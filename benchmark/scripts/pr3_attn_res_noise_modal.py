"""Show that attn_res dW_norm run-to-run variation is pre-existing atomics noise.

`_attn_res_bwd_kernel` accumulates dW_query/dW_norm with `tl.atomic_add`, so
float addition order varies between launches and the result is nondeterministic
by construction -- in eager as well as compiled, on main as well as on this
branch. This samples the spread on both arms so the numbers can be compared
like-for-like instead of from a single draw.

Run::

    modal run benchmark/scripts/pr3_attn_res_noise_modal.py
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

app = modal.App("liger-pr3-attn-noise")


def _export_ref(ref: str) -> pathlib.Path:
    export_root = pathlib.Path("/tmp") / f"liger-noise-{ref.replace('/', '_')}"
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
from liger_kernel.ops.attn_res import LigerAttnResFunction

g = torch.Generator(device="cuda").manual_seed(0)
V = torch.randn(4, 2, 2048, 4096, dtype=torch.bfloat16, device="cuda", generator=g, requires_grad=True)
wq = torch.randn(4096, dtype=torch.bfloat16, device="cuda", generator=g, requires_grad=True)
wn = torch.randn(4096, dtype=torch.bfloat16, device="cuda", generator=g, requires_grad=True)
inputs = [V, wq, wn]

fn = lambda v, a, b: LigerAttnResFunction.apply(v, a, b, 1e-6)
cfn = torch.compile(fn, fullgraph=False, dynamic=False)

def run(f):
    for t in inputs:
        t.grad = None
    f(*inputs).sum().backward()
    return wn.grad.detach().clone().float()

N = 6
for label, f in (("eager", fn), ("compiled", cfn)):
    runs = [run(f) for _ in range(N)]
    base = runs[0]
    mag = base.abs().max().item()
    spread = max((r - base).abs().max().item() for r in runs[1:])
    print(f"  dW_norm {label:<9} |grad|max={mag:10.2f}  max run-to-run spread over {N} runs={spread:8.4f}"
          f"  rel={spread / mag:.2e}", flush=True)
"""


@app.function(image=image, gpu="H100", timeout=ONE_HOUR)
def noise():
    import subprocess as sp

    env = dict(os.environ, TORCHINDUCTOR_COMPILE_THREADS="1", TORCHINDUCTOR_FORCE_DISABLE_CACHES="1")
    for label, root in (("BASELINE (HEAD)", BASELINE_ROOT), ("WORKTREE (fixed)", REMOTE_ROOT)):
        print(f"\n=== {label} ===", flush=True)
        sp.run(["python", "-c", PROBE], env=dict(env, LIGER_ROOT=root), check=False)


@app.local_entrypoint()
def main():
    noise.remote()
