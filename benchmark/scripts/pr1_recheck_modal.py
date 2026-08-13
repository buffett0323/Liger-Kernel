"""Does PR1's backward `offset.to(tl.float32)` corrupt fused_add_rms_norm?

PR3 established that `.to()` on a scalar Triton param makes torch's Triton
mutation analysis fail (it traces with a raw Python float), after which torch
conservatively treats every pointer argument as mutated. In
`_modulated_rms_norm_backward_kernel` that either raised "a leaf Variable that
requires grad is being used in an in-place operation" or silently returned a
nondeterministic dScale.

`_fused_add_rms_norm_backward_kernel` (changed by PR1 / #1358) has the same
shape of risk: `W` is a leaf requiring grad, and `dX = dY` when in_place=True.
The PR1 audit only measured fp64 counts and eager-vs-compiled diffs with
in_place=False, so this failure mode was never tested.

Compares the PR1 branch against its parent, for in_place False and True.

Run::

    modal run benchmark/scripts/pr1_recheck_modal.py
"""

import os
import pathlib
import subprocess

import modal

FIXED_ROOT = "/root/liger_pr1"
BASE_ROOT = "/root/liger_base"
PR1_REF = "fix/fp64-scalar-norm-ops"
BASE_REF = "fix/fp64-scalar-norm-ops~1"
ONE_HOUR = 60 * 60

IS_LOCAL = modal.is_local()
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2] if IS_LOCAL else pathlib.Path("/root")

app = modal.App("liger-pr1-recheck")


def _export_ref(ref: str) -> pathlib.Path:
    export_root = pathlib.Path("/tmp") / f"liger-pr1-{ref.replace('/', '_').replace('~', '-')}"
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
    image = image.add_local_dir(str(_export_ref(PR1_REF)), f"{FIXED_ROOT}/liger_kernel").add_local_dir(
        str(_export_ref(BASE_REF)), f"{BASE_ROOT}/liger_kernel"
    )

PROBE = r"""
import sys, os, re
sys.path.insert(0, os.environ["LIGER_ROOT"])
import torch
import triton
from liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction

f64 = {"n": 0}
_orig = triton.compile
def _patched(*a, **k):
    c = _orig(*a, **k)
    try:
        ptx = c.asm.get("ptx", "")
        if "fused_add_rms_norm" in ptx:
            f64["n"] += len(re.findall(r"\.f64", ptx))
    except Exception:
        pass
    return c
triton.compile = _patched

ROWS, COLS = 4096, 8192
IN_PLACE = os.environ["IN_PLACE"] == "1"
NAMES = ["X", "R", "W"]

def make():
    g = torch.Generator(device="cuda").manual_seed(0)
    X = torch.randn(ROWS, COLS, dtype=torch.bfloat16, device="cuda", generator=g, requires_grad=True)
    R = torch.randn(ROWS, COLS, dtype=torch.bfloat16, device="cuda", generator=g, requires_grad=True)
    W = torch.randn(COLS, dtype=torch.bfloat16, device="cuda", generator=g, requires_grad=True)
    return [X, R, W]

inputs = make()
# clone so an in-place kernel cannot corrupt the inputs between arms
fn = lambda x, r, w: LigerFusedAddRMSNormFunction.apply(x.clone(), r.clone(), w, 1e-6, 0.0, "llama", IN_PLACE)[0]
cfn = torch.compile(fn, fullgraph=False, dynamic=False)

def drive(f):
    for t in inputs:
        t.grad = None
    f(*inputs).sum().backward()
    return [t.grad.detach().clone().float() for t in inputs]

try:
    e1 = drive(fn)
    runs = [drive(cfn) for _ in range(4)]
except Exception as exc:
    print(f"  in_place={IN_PLACE}  RAISED: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
    raise SystemExit(0)

worst_cvc, worst_name = 0.0, "-"
for i, nm in enumerate(NAMES):
    for r in runs[1:]:
        d = (r[i] - runs[0][i]).abs().max().item()
        if d > worst_cvc:
            worst_cvc, worst_name = d, nm
ec = max((runs[0][i] - e1[i]).abs().max().item() for i in range(3))
mag = max(e1[i].abs().max().item() for i in range(3))
print(f"  in_place={str(IN_PLACE):<5} f64={f64['n']:<5} comp-vs-comp={worst_cvc:9.4f} on d{worst_name:<3}"
      f" eager-vs-comp={ec:9.4f}  |grad|max={mag:9.2f}", flush=True)
"""


@app.function(image=image, gpu="H100", timeout=ONE_HOUR)
def recheck():
    import subprocess as sp

    env = dict(os.environ, TORCHINDUCTOR_COMPILE_THREADS="1", TORCHINDUCTOR_FORCE_DISABLE_CACHES="1")
    for label, root in (("BASELINE (PR1 parent)", BASE_ROOT), ("PR1 (#1358, with .to())", FIXED_ROOT)):
        print(f"\n=== {label} ===", flush=True)
        for ip in ("0", "1"):
            sp.run(["python", "-c", PROBE], env=dict(env, LIGER_ROOT=root, IN_PLACE=ip), check=False)


@app.local_entrypoint()
def main():
    recheck.remote()
