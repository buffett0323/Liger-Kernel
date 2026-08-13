"""Diagnose why Liger group_norm is slow at large batch / small hidden_size.

Splits forward from backward, sweeps batch size and dtype, and compares against
``torch.nn.GroupNorm`` so the result is interpretable rather than an isolated
number.

Hypothesis under test: ``_group_norm_backward_kernel`` reduces dW/dB with
``tl.atomic_add`` onto only ``num_channels`` distinct addresses. Every one of the
``batch_size * num_groups`` programs hits those same few addresses, so cost
should grow with batch size independently of total FLOPs, and should be far worse
in bf16 (where the atomic is emulated with a CAS retry loop) than in fp32.

Run::

    modal run benchmark/scripts/group_norm_slowdown_probe_modal.py --gpu H100
"""

import pathlib

import modal

REMOTE_ROOT = "/root/liger"
ONE_HOUR = 60 * 60

IS_LOCAL = modal.is_local()
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2] if IS_LOCAL else pathlib.Path(REMOTE_ROOT)

app = modal.App("liger-group-norm-probe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.11.0+cu130",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install("triton", "packaging")
)

if IS_LOCAL:
    image = image.add_local_dir(str(REPO_ROOT / "src" / "liger_kernel"), f"{REMOTE_ROOT}/liger_kernel")

PROBE = r"""
import statistics, time
import torch
from liger_kernel.ops.group_norm import LigerGroupNormFunction

def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)

def run(batch, channels, groups, hidden, dtype):
    X = torch.randn(batch, channels, hidden, dtype=dtype, device="cuda", requires_grad=True)
    W = torch.randn(channels, dtype=dtype, device="cuda", requires_grad=True)
    B = torch.randn(channels, dtype=dtype, device="cuda", requires_grad=True)

    def fwd():
        return LigerGroupNormFunction.apply(X, W, B, channels, groups, 1e-6)

    out = fwd()
    g = torch.randn_like(out)

    def fwd_only():
        LigerGroupNormFunction.apply(X, W, B, channels, groups, 1e-6)

    def fwd_bwd():
        for t in (X, W, B):
            t.grad = None
        LigerGroupNormFunction.apply(X, W, B, channels, groups, 1e-6).backward(g)

    tor = torch.nn.GroupNorm(groups, channels, eps=1e-6).to(dtype).cuda()
    Xt = X.detach().clone().requires_grad_(True)

    def torch_fwd_bwd():
        Xt.grad = None
        tor(Xt).backward(g)

    f = bench(fwd_only)
    fb = bench(fwd_bwd)
    tf = bench(torch_fwd_bwd)
    ctas = batch * groups
    print(f"  {str(dtype).replace('torch.',''):<9} batch={batch:<6} chan={channels:<4} grp={groups:<3} "
          f"hidden={hidden:<6} CTAs={ctas:<7} fwd={f:8.4f} ms  fwd+bwd={fb:9.4f} ms  "
          f"bwd~={fb-f:9.4f} ms  torch={tf:7.4f} ms  liger/torch={fb/tf:7.2f}x")
    return fb - f

print("=" * 150)
print("A) batch sweep at fixed work per CTA -- if atomics contention dominates, bwd grows with CTA count")
print("=" * 150)
for batch in (16, 64, 256, 1024, 2048):
    run(batch, 32, 4, 128, torch.bfloat16)

print()
print("=" * 150)
print("B) same total elements, batch traded against hidden_size -- isolates CTA count from data volume")
print("=" * 150)
for batch, hidden in ((2048, 128), (512, 512), (128, 2048), (32, 8192)):
    run(batch, 32, 4, hidden, torch.bfloat16)

print()
print("=" * 150)
print("C) dtype: bf16 atomics are emulated with a CAS retry loop, fp32 atomics are native")
print("=" * 150)
for dt in (torch.bfloat16, torch.float32):
    run(2048, 32, 4, 128, dt)

print()
print("=" * 150)
print("D) repo-representative shapes (from test/transformers/test_group_norm.py)")
print("=" * 150)
for batch, chan, grp, hid in ((16, 32, 1, 4096), (16, 48, 12, 8192), (2, 63, 21, 2163)):
    run(batch, chan, grp, hid, torch.float32)
"""


def _run(label):
    import subprocess as sp

    print(f"[{label}] {'=' * 60}", flush=True)
    sp.run(
        ["python", "-c", PROBE],
        cwd=REMOTE_ROOT,
        env={"PYTHONPATH": REMOTE_ROOT, "PATH": "/usr/local/bin:/usr/bin:/bin"},
    )


@app.function(image=image, gpu="H100", timeout=ONE_HOUR)
def run_h100():
    _run("H100")


@app.function(image=image, gpu="B300", timeout=ONE_HOUR)
def run_b300():
    _run("B300")


ENTRYPOINTS = {"H100": run_h100, "B300": run_b300}


@app.local_entrypoint()
def main(gpu: str = "H100"):
    for label in [g.strip().upper() for g in gpu.split(",") if g.strip()]:
        if label in ENTRYPOINTS:
            ENTRYPOINTS[label].remote()
