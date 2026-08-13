"""Determinism re-check for PR2 (cross_entropy, #1362).

PR3 showed that `.to()` on a scalar Triton param makes torch's Triton mutation
analysis fail, after which torch conservatively assumes every pointer argument is
mutated -- which silently corrupted dScale in modulated_rms_norm. PR2 adds
exactly that kind of cast (`softcap`, `sum_non_ignore_weight`, `weight_sum`) to a
kernel that *writes gradients in place into X*, so it is the remaining
unverified surface.

PR2's original audit measured fp64 counts and eager-vs-compiled diffs but never
compiled-vs-compiled determinism. This adds that, per scalar path.

Run::

    modal run benchmark/scripts/pr2_recheck_modal.py
"""

import os
import pathlib
import subprocess

import modal

FIXED_ROOT = "/root/liger_pr2"
BASE_ROOT = "/root/liger_base"
PR2_REF = "fix/fp64-scalar-cross-entropy"
BASE_REF = "fix/fp64-scalar-cross-entropy^"
ONE_HOUR = 60 * 60

IS_LOCAL = modal.is_local()
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2] if IS_LOCAL else pathlib.Path("/root")

app = modal.App("liger-pr2-recheck")


def _export_ref(ref: str) -> pathlib.Path:
    export_root = pathlib.Path("/tmp") / f"liger-pr2-{ref.replace('/', '_').replace('^', '-p')}"
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
    image = image.add_local_dir(str(_export_ref(PR2_REF)), f"{FIXED_ROOT}/liger_kernel").add_local_dir(
        str(_export_ref(BASE_REF)), f"{BASE_ROOT}/liger_kernel"
    )

PROBE = r"""
import sys, os, re
sys.path.insert(0, os.environ["LIGER_ROOT"])
import torch
import triton
from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction

f64 = {"n": 0}
_orig = triton.compile
def _patched(*a, **k):
    c = _orig(*a, **k)
    try:
        ptx = c.asm.get("ptx", "")
        if "cross_entropy" in ptx:
            f64["n"] += len(re.findall(r"\.f64", ptx))
    except Exception:
        pass
    return c
triton.compile = _patched

ROWS, COLS = 4096, 32000
VARIANT = os.environ["VARIANT"]

g = torch.Generator(device="cuda").manual_seed(0)
X = torch.randn(ROWS, COLS, dtype=torch.bfloat16, device="cuda", generator=g, requires_grad=True)
target = torch.randint(0, COLS, (ROWS,), device="cuda", generator=g)
weight = torch.rand(COLS, dtype=torch.float32, device="cuda", generator=g) + 0.5

kw = dict(weight=None, ignore_index=-100, lse_square_scale=0.0, label_smoothing=0.0,
          reduction="mean", softcap=None, return_z_loss=False)
if VARIANT == "softcap":
    kw["softcap"] = 30.0
elif VARIANT == "weight":
    kw["weight"] = weight
elif VARIANT == "weight_smoothing":
    kw["weight"] = weight
    kw["label_smoothing"] = 0.1

# Liger CE writes the gradient in place into its input, so every call must get a
# fresh clone or later arms see corrupted data. clone() is differentiable.
def fn(x):
    return LigerCrossEntropyFunction.apply(
        x.clone(), target, kw["weight"], kw["ignore_index"], kw["lse_square_scale"],
        kw["label_smoothing"], kw["reduction"], kw["softcap"], kw["return_z_loss"],
    )[0]

cfn = torch.compile(fn, fullgraph=False, dynamic=False)

def drive(f):
    X.grad = None
    f(X).sum().backward()
    return X.grad.detach().clone().float()

try:
    e1 = drive(fn)
    runs = [drive(cfn) for _ in range(4)]
except Exception as exc:
    print(f"  {VARIANT:<18} RAISED: {type(exc).__name__}: {str(exc)[:150]}", flush=True)
    raise SystemExit(0)

cvc = max((r - runs[0]).abs().max().item() for r in runs[1:])
ec = (runs[0] - e1).abs().max().item()
mag = e1.abs().max().item()
print(f"  {VARIANT:<18} f64={f64['n']:<5} comp-vs-comp={cvc:.3e}  eager-vs-comp={ec:.3e}  |grad|max={mag:.3e}", flush=True)
"""


@app.function(image=image, gpu="H100", timeout=ONE_HOUR)
def recheck():
    import subprocess as sp

    env = dict(os.environ, TORCHINDUCTOR_COMPILE_THREADS="1", TORCHINDUCTOR_FORCE_DISABLE_CACHES="1")
    for label, root in (("BASELINE (PR2 parent)", BASE_ROOT), ("PR2 (#1362, fixed)", FIXED_ROOT)):
        print(f"\n=== {label} ===", flush=True)
        for v in ("plain", "softcap", "weight", "weight_smoothing"):
            sp.run(["python", "-c", PROBE], env=dict(env, LIGER_ROOT=root, VARIANT=v), check=False)


@app.local_entrypoint()
def main():
    recheck.remote()
