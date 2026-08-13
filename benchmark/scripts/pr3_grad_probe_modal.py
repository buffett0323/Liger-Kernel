"""Diagnose the large absolute eager-vs-compiled grad diff on modulated_rms_norm.

The fp64 audit reports max |grad_compiled - grad_eager| as a raw absolute
number. For this op dW is a reduction over all rows, so its magnitude is huge
and an absolute diff is meaningless on its own. This probe reports, per input
tensor and per arm (baseline ref vs fixed worktree):

  * the tensor's own magnitude
  * absolute and RELATIVE eager-vs-compiled diff
  * an eager-vs-eager rerun diff, to separate real error from nondeterminism

Run::

    modal run benchmark/scripts/pr3_grad_probe_modal.py
"""

import os
import pathlib
import subprocess

import modal

REMOTE_ROOT = "/root/liger"
BASELINE_ROOT = "/root/liger_baseline"
BASELINE_REF = "HEAD"
ONE_HOUR = 60 * 60

IS_LOCAL = modal.is_local()
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2] if IS_LOCAL else pathlib.Path(REMOTE_ROOT)

app = modal.App("liger-pr3-grad-probe")


def _export_ref(ref: str) -> pathlib.Path:
    export_root = pathlib.Path("/tmp") / f"liger-pr3-{ref.replace('/', '_')}"
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
    image = image.add_local_dir(str(REPO_ROOT / "src" / "liger_kernel"), f"{REMOTE_ROOT}/liger_kernel").add_local_dir(
        str(_export_ref(BASELINE_REF)), f"{BASELINE_ROOT}/liger_kernel"
    )

PROBE = r"""
import sys, os
sys.path.insert(0, os.environ["LIGER_ROOT"])
import torch
import liger_kernel
print("  liger_kernel from", os.path.dirname(liger_kernel.__file__), flush=True)

from liger_kernel.ops.modulated_rms_norm import LigerModulatedRMSNormFunction
from liger_kernel.ops.attn_res import LigerAttnResFunction

ROWS, COLS = 4096, 8192
NAMES = ["X", "W", "scale"]


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
    return out.detach().clone(), [t.grad.detach().clone() for t in inputs]


def attn_inputs(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    V = torch.randn(4, 2, 2048, 4096, dtype=torch.bfloat16, device="cuda", generator=g, requires_grad=True)
    wq = torch.randn(4096, dtype=torch.bfloat16, device="cuda", generator=g, requires_grad=True)
    wn = torch.randn(4096, dtype=torch.bfloat16, device="cuda", generator=g, requires_grad=True)
    return [V, wq, wn]


afn = lambda v, wq, wn: LigerAttnResFunction.apply(v, wq, wn, 1e-6)
acfn = torch.compile(afn, fullgraph=False, dynamic=False)
ai = attn_inputs(0)
_, ae = drive(afn, ai)
aruns = [drive(acfn, ai)[1] for _ in range(4)]
print("\n=== attn_res ===", flush=True)
for i, nm in enumerate(["V", "w_query", "w_norm"]):
    mag = ae[i].float().abs().max().item()
    cvc = max((r[i].float() - aruns[0][i].float()).abs().max().item() for r in aruns[1:])
    ec = (aruns[0][i].float() - ae[i].float()).abs().max().item()
    print(f"  d{nm:<8} |grad|max={mag:12.4f}  comp-vs-comp={cvc:10.4f}  eager-vs-comp={ec:10.4f}", flush=True)

for mode in ["llama", "gemma", "none"]:
    fn = lambda x, w, sc: LigerModulatedRMSNormFunction.apply(x, w, sc, None, 1e-6, 0.0, mode, False)
    cfn = torch.compile(fn, fullgraph=False, dynamic=False)

    inputs = make(0)
    _, e1 = drive(fn, inputs)
    _, e2 = drive(fn, inputs)          # eager rerun -> nondeterminism floor
    _, c1 = drive(cfn, inputs)
    _, c2 = drive(cfn, inputs)         # compiled rerun

    print(f"\n=== casting_mode={mode} ===", flush=True)
    for name, a, b, cc, dd in zip(NAMES, e1, e2, c1, c2):
        mag = a.float().abs().max().item()
        ee = (b.float() - a.float()).abs().max().item()          # eager vs eager
        cvc = (dd.float() - cc.float()).abs().max().item()       # compiled vs compiled
        ec = (cc.float() - a.float()).abs().max().item()         # eager vs compiled
        rel = ec / mag if mag else 0.0
        print(
            f"  d{name:<6} |grad|max={mag:12.4f}  eager-vs-eager={ee:10.4f}  "
            f"comp-vs-comp={cvc:10.4f}  eager-vs-comp={ec:10.4f}  rel={rel:.3e}",
            flush=True,
        )
"""


@app.function(image=image, gpu="H100", timeout=ONE_HOUR)
def probe():
    import subprocess as sp

    env = dict(os.environ, TORCHINDUCTOR_COMPILE_THREADS="1", TORCHINDUCTOR_FORCE_DISABLE_CACHES="1")
    for label, root in (("BASELINE (HEAD)", BASELINE_ROOT), ("WORKTREE (fixed)", REMOTE_ROOT)):
        print("\n" + "=" * 78, flush=True)
        print(label, flush=True)
        print("=" * 78, flush=True)
        sp.run(["python", "-c", PROBE], env=dict(env, LIGER_ROOT=root), check=False)


@app.local_entrypoint()
def main():
    probe.remote()
