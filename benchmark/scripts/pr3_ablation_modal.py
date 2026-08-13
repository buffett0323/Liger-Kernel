"""Ablation: which half of the modulated_rms_norm fix causes the nondeterminism?

The fix has two independent parts:

  FWD  add an ``else:`` branch casting eps/offset to fp32 in the forward kernel
  BWD  change ``W_row = W_row + offset`` to ``offset.to(tl.float32)`` in backward

The full fix removes the fp64 contamination but made the *compiled* dScale
gradient differ run-to-run. This script builds four trees from the baseline ref
(none / FWD / BWD / both), and for each reports fp64 counts and, critically, a
compiled-vs-compiled rerun diff that exposes nondeterminism.

Run::

    modal run benchmark/scripts/pr3_ablation_modal.py
"""

import os
import pathlib
import subprocess

import modal

BASE_ROOT = "/root/liger_baseline"
BASELINE_REF = "HEAD"
ONE_HOUR = 60 * 60

IS_LOCAL = modal.is_local()
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2] if IS_LOCAL else pathlib.Path("/root")

app = modal.App("liger-pr3-ablation")


def _export_ref(ref: str) -> pathlib.Path:
    export_root = pathlib.Path("/tmp") / f"liger-abl-{ref.replace('/', '_')}"
    package = export_root / "src" / "liger_kernel"
    if package.is_dir():
        subprocess.run(["rm", "-rf", str(export_root)], check=True)
    export_root.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", ref, "src/liger_kernel"], cwd=REPO_ROOT, check=True, capture_output=True
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(export_root)], input=archive, check=True)
    return package


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.11.0+cu130", "numpy", extra_index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("triton", "packaging")
)

if IS_LOCAL:
    image = image.add_local_dir(str(_export_ref(BASELINE_REF)), f"{BASE_ROOT}/liger_kernel")

FWD_OLD = """    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols"""

FWD_NEW = """    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)
    else:
        eps = eps.to(tl.float32)
        offset = offset.to(tl.float32)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols"""

BWD_OLD = """        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
        W_row = W_row + offset"""

BWD_NEW = """        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
        W_row = W_row + offset.to(tl.float32)"""

# Alternative backward fix: make the scalar a compile-time constant so no fp64
# parameter exists at all, and no ``.to()`` is called on it (which is what
# breaks torch's Triton mutation analysis).
BWDCE_OLD = """    n_cols,
    offset,
    rows_per_program,"""

BWDCE_NEW = """    n_cols,
    offset: tl.constexpr,
    rows_per_program,"""

PROBE = r"""
import sys, os, re
sys.path.insert(0, os.environ["LIGER_ROOT"])
import torch
from liger_kernel.ops.modulated_rms_norm import LigerModulatedRMSNormFunction

ROWS, COLS = 4096, 8192
NAMES = ["X", "W", "scale"]

f64 = {"n": 0}
import triton
_orig = triton.compile
def _patched(*a, **k):
    c = _orig(*a, **k)
    try:
        ptx = c.asm.get("ptx", "")
        if "modulated" in ptx:
            f64["n"] += len(re.findall(r"\.f64", ptx))
    except Exception:
        pass
    return c
triton.compile = _patched

def make(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    X = torch.randn(ROWS, COLS, dtype=torch.bfloat16, device="cuda", generator=g, requires_grad=True)
    W = torch.randn(COLS, dtype=torch.bfloat16, device="cuda", generator=g, requires_grad=True)
    sc = torch.randn(ROWS, COLS, dtype=torch.bfloat16, device="cuda", generator=g, requires_grad=True)
    return [X, W, sc]

def drive(fn, inputs):
    for t in inputs:
        t.grad = None
    out = fn(*inputs)
    out.sum().backward()
    return [t.grad.detach().clone() for t in inputs]

mode = os.environ["MODE"]
fn = lambda x, w, sc: LigerModulatedRMSNormFunction.apply(x, w, sc, None, 1e-6, 0.0, mode, False)
cfn = torch.compile(fn, fullgraph=False, dynamic=False)

inputs = make(0)
e1 = drive(fn, inputs)
runs = [drive(cfn, inputs) for _ in range(4)]

worst_cvc = 0.0
worst_name = ""
for i, name in enumerate(NAMES):
    base = runs[0][i].float()
    for r in runs[1:]:
        d = (r[i].float() - base).abs().max().item()
        if d > worst_cvc:
            worst_cvc, worst_name = d, name
ec = max((runs[0][i].float() - e1[i].float()).abs().max().item() for i in range(3))
print(f"  f64={f64['n']:<5} compiled-vs-compiled(max over 4 runs)={worst_cvc:10.4f} on d{worst_name or '-'}   eager-vs-compiled={ec:10.4f}", flush=True)
"""


@app.function(image=image, gpu="H100", timeout=ONE_HOUR)
def ablate():
    import shutil
    import subprocess as sp

    src = pathlib.Path(BASE_ROOT) / "liger_kernel"
    variants = {
        "none": (False, False, False),
        "FWD": (True, False, False),
        "BWD": (False, True, False),
        "FWDBWD": (True, True, False),
        "BWDCE": (False, False, True),
        "FWDBWDCE": (True, False, True),
    }

    env = dict(os.environ, TORCHINDUCTOR_COMPILE_THREADS="1", TORCHINDUCTOR_FORCE_DISABLE_CACHES="1")

    for label, (fwd, bwd, bwdce) in variants.items():
        root = pathlib.Path("/root/variants") / label
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        shutil.copytree(src, root / "liger_kernel")
        f = root / "liger_kernel" / "ops" / "modulated_rms_norm.py"
        s = f.read_text()
        if fwd:
            assert FWD_OLD in s, "fwd anchor missing"
            s = s.replace(FWD_OLD, FWD_NEW)
        if bwd:
            assert BWD_OLD in s, "bwd anchor missing"
            s = s.replace(BWD_OLD, BWD_NEW)
        if bwdce:
            assert BWDCE_OLD in s, "bwdce anchor missing"
            s = s.replace(BWDCE_OLD, BWDCE_NEW)
        f.write_text(s)

        for mode in ("llama", "none"):
            print(f"\n[{label}] casting_mode={mode}", flush=True)
            sp.run(
                ["python", "-c", PROBE],
                env=dict(env, LIGER_ROOT=str(root), MODE=mode),
                check=False,
            )


@app.local_entrypoint()
def main():
    ablate.remote()
