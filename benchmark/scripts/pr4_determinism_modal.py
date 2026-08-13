"""PR4: determinism + perf probe for the grpo_loss fp64-scalar fix.

PR3 taught that a *falling* .f64 count can coexist with silently corrupted
gradients: making a scalar fp32 via ``.to()`` breaks torch's Triton mutation
analysis, after which torch assumes every pointer arg is mutated and the
compiled backward becomes nondeterministic. PTX counting alone did not catch it
-- only re-running the *same* compiled graph and diffing the gradients did.

So this script reports, per variant:

  * comp-vs-comp : max |grad| diff over N identical compiled reruns. Must be 0.
  * eager-vs-comp: max |grad| diff between the eager and compiled arms.
  * timing       : eager vs compiled GPU time, to size the win.

grpo_loss defaults to ``inplace=True`` (the forward writes the gradient into
logits), which is exactly the aliasing situation that made modulated_rms_norm
fail, so this check matters more here than usual.
"""

import pathlib

import modal

REMOTE_ROOT = "/root/liger"
ONE_HOUR = 60 * 60
RERUNS = 5

IS_LOCAL = modal.is_local()
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2] if IS_LOCAL else pathlib.Path(REMOTE_ROOT)

app = modal.App("liger-pr4-grpo-determinism")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.11.0+cu130",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install("triton", "packaging")
)

if IS_LOCAL:
    image = image.add_local_dir(str(REPO_ROOT / "src" / "liger_kernel"), f"{REMOTE_ROOT}/liger_kernel")

VARIANTS = [
    ("dapo", dict(loss_type="dapo")),
    ("cispo", dict(loss_type="cispo")),
    ("sapo", dict(loss_type="sapo")),
    ("delta", dict(loss_type="grpo", delta=2.0)),
    ("seq", dict(loss_type="grpo", importance_sampling_level="sequence")),
]


@app.function(image=image, gpu="H100", timeout=ONE_HOUR)
def probe(rows: int = 2048, vocab: int = 32000):
    import sys

    sys.path.insert(0, REMOTE_ROOT)

    import torch

    from liger_kernel.transformers.grpo_loss import triton_grpo_loss

    dev = "cuda"
    print(f"# torch {torch.__version__} | {torch.cuda.get_device_name()}")
    print(f"# B*L={rows} vocab={vocab} bfloat16 | {RERUNS} compiled reruns\n")

    batch = 4
    seq = rows // batch

    def build(**kw):
        torch.manual_seed(0)
        logits = torch.randn(batch, seq + 1, vocab, dtype=torch.bfloat16, device=dev, requires_grad=True)
        ids = torch.randint(0, vocab, (batch, seq), device=dev)
        old_logp = torch.randn(batch, seq, dtype=torch.float32, device=dev) * 0.1 - 2.0
        ref_logp = torch.randn(batch, seq, dtype=torch.float32, device=dev) * 0.1 - 2.0
        adv = torch.randn(batch, dtype=torch.float32, device=dev)
        mask = torch.ones(batch, seq, dtype=torch.int32, device=dev)

        def fn(x):
            return triton_grpo_loss(
                x.clone(),
                old_logp,
                ref_logp,
                ids,
                adv,
                completion_mask=mask,
                temperature=0.9,
                beta=0.04,
                eps_low=0.2,
                eps_high=0.4,
                reduce=False,
                **kw,
            )

        return fn, logits

    def grads(fn, x):
        if x.grad is not None:
            x.grad = None
        out = fn(x)
        (out[0] if isinstance(out, (tuple, list)) else out).sum().backward()
        return x.grad.detach().clone()

    def bench(fn, x, iters=10):
        for _ in range(3):
            grads(fn, x)
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        for _ in range(iters):
            grads(fn, x)
        ev1.record()
        torch.cuda.synchronize()
        return ev0.elapsed_time(ev1) / iters

    hdr = f"{'variant':<8} {'compiles':<9} {'comp-vs-comp':>13} {'eager-vs-comp':>14} {'eager ms':>9} {'comp ms':>9} {'speedup':>8}"
    print(hdr)
    print("-" * len(hdr))

    for name, kw in VARIANTS:
        fn, x = build(**kw)
        try:
            g_eager = grads(fn, x)
            torch._dynamo.reset()
            cfn = torch.compile(fn)
            runs = [grads(cfn, x) for _ in range(RERUNS)]
        except Exception as e:
            msg = str(e).split("\n")[0][:60]
            print(f"{name:<8} {'NO':<9} {'-':>13} {'-':>14} {'-':>9} {'-':>9} {'-':>8}   {type(e).__name__}: {msg}")
            continue

        cvc = max((runs[i].float() - runs[0].float()).abs().max().item() for i in range(1, len(runs)))
        evc = (runs[0].float() - g_eager.float()).abs().max().item()
        t_e = bench(fn, x)
        t_c = bench(cfn, x)
        flag = "" if cvc == 0.0 else "   <-- NONDETERMINISTIC"
        print(f"{name:<8} {'yes':<9} {cvc:>13.4f} {evc:>14.4f} {t_e:>9.3f} {t_c:>9.3f} {t_e / t_c:>7.2f}x{flag}")

    print("\n(comp-vs-comp must be exactly 0; any nonzero value means the compiled")
    print(" backward is nondeterministic -- the PR3 failure mode.)")


@app.local_entrypoint()
def main(rows: int = 2048, vocab: int = 32000):
    probe.remote(rows=rows, vocab=vocab)
