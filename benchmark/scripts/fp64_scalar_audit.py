"""Detect fp64 scalar contamination in Liger Triton kernels under ``torch.compile``.

Background
----------
A Python ``float`` passed to a non-``constexpr`` Triton kernel parameter is
specialized to **fp32** by Triton's own JIT, but to **fp64** by Inductor. Triton's
standard type promotion then widens any expression mixing that scalar with fp32
tensors -- including transcendentals such as ``rsqrt``, ``tanh`` and ``log`` -- to
float64. The kernel silently runs in double precision under ``torch.compile``.

The severity is architecture-dependent (catastrophic on B300, ~2x on H100, roughly
free on B200) but the *promotion itself happens everywhere*. So timing is a poor
detector and generated PTX is a perfect one: this script counts ``.f64``
instructions in the PTX of each Liger kernel and reports any non-zero count.

A correct kernel emits 0 ``.f64`` instructions eager, and at most a couple
compiled (the incoming ``.param .f64`` plus a single ``cvt.rn.f32.f64`` demotion).

Usage
-----
    python benchmark/scripts/fp64_scalar_audit.py                 # all ops
    python benchmark/scripts/fp64_scalar_audit.py --ops layer_norm
    python benchmark/scripts/fp64_scalar_audit.py --json out.json

Exit code is non-zero if any op is contaminated, so this doubles as a regression
guard. Requires a CUDA GPU.
"""

import argparse
import collections
import json
import logging
import os
import re
import sys

# Inductor compiles in worker subprocesses by default, where an in-process hook
# would never fire. Must be set before torch is imported.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# Two shapes that round to the same BLOCK_SIZE produce byte-identical Triton
# source, so the second one is served from the on-disk FX graph cache and
# triton.compile never fires -- the PTX counter would report a misleading 0.
# torch._dynamo.reset() does not clear that cache; this does.
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402

# torch's Triton mutation analysis logs a long traceback whenever it cannot
# constant-fold a kernel body; it is noisy and orthogonal to what we measure.
logging.getLogger("torch._higher_order_ops.triton_kernel_wrap").setLevel(logging.ERROR)

# Only PTX for kernels whose name matches one of these is attributed to an op.
# Anything else (Inductor's own generated kernels, e.g. pointwise fusions) is
# reported separately and is not our concern.
_LIGER_KERNEL_RE = re.compile(
    r"^_?(liger|element_mul|_kldiv|_tvd|_attn_res|.*(rms_norm|layer_norm|group_norm|poly_norm"
    r"|swiglu|geglu|cross_entropy|kl_div|kldiv|tvd|attn_res))"
)

# `.param .f64` in the signature is expected and harmless once the body casts to
# fp32, so only count *instructions* that compute in f64.
_F64_INSTR_RE = re.compile(r"^\s*(?!\.)\S*\.f64\b", re.MULTILINE)
_F32_INSTR_RE = re.compile(r"^\s*(?!\.)\S*\.f32\b", re.MULTILINE)


def _count(pattern, ptx):
    return len(pattern.findall(ptx))


class PTXRecorder:
    """Captures PTX for every Triton kernel compiled inside the ``with`` block.

    Patches both ``triton.compiler.compiler.compile`` and the ``compile`` name
    bound into ``triton.runtime.jit``'s module namespace. The latter matters
    because eager ``JITFunction.run`` resolves ``compile`` at import time, so
    patching only the ``triton.compile`` alias silently records nothing for the
    eager path.
    """

    def __init__(self):
        self.records = []
        self._saved = []

    def __enter__(self):
        import triton
        import triton.compiler.compiler as tcc
        import triton.runtime.jit as tjit

        original = tcc.compile

        def hooked(*args, **kwargs):
            compiled = original(*args, **kwargs)
            try:
                ptx = compiled.asm.get("ptx")
                if ptx:
                    self.records.append((compiled.name, ptx))
            except Exception:
                pass
            return compiled

        for module in (tcc, tjit, triton):
            if getattr(module, "compile", None) is original:
                self._saved.append((module, original))
                module.compile = hooked
        return self

    def __exit__(self, *exc):
        for module, original in self._saved:
            module.compile = original
        self._saved = []
        return False

    def summary(self):
        """Aggregate f64/f32 instruction counts over recorded Liger kernels."""
        per_kernel = collections.OrderedDict()
        for name, ptx in self.records:
            if not _LIGER_KERNEL_RE.match(name):
                continue
            entry = per_kernel.setdefault(name, {"f64": 0, "f32": 0})
            entry["f64"] += _count(_F64_INSTR_RE, ptx)
            entry["f32"] += _count(_F32_INSTR_RE, ptx)
        return per_kernel


# --------------------------------------------------------------------------
# Op definitions. Each returns (callable, inputs) where callable(*inputs) runs a
# forward pass; the harness drives backward itself.
# --------------------------------------------------------------------------


def _make_inputs(rows, cols, dtype, device, n=1):
    return [torch.randn(rows, cols, dtype=dtype, device=device, requires_grad=True) for _ in range(n)]


def op_layer_norm(rows, cols, dtype, device):
    from liger_kernel.ops.layer_norm import LigerLayerNormFunction

    (X,) = _make_inputs(rows, cols, dtype, device)
    W = torch.randn(cols, dtype=dtype, device=device, requires_grad=True)
    B = torch.randn(cols, dtype=dtype, device=device, requires_grad=True)
    return (lambda x, w, b: LigerLayerNormFunction.apply(x, w, b, 1e-6)), (X, W, B)


def op_group_norm(rows, cols, dtype, device):
    from liger_kernel.ops.group_norm import LigerGroupNormFunction

    # rows == batch; cols is split into (num_channels, hidden). Keep hidden large
    # enough that the op is bandwidth-bound rather than atomics-bound (see the
    # backward's tl.atomic_add onto num_channels addresses).
    num_channels, num_groups = 32, 4
    X = torch.randn(rows, num_channels, cols // num_channels, dtype=dtype, device=device, requires_grad=True)
    W = torch.randn(num_channels, dtype=dtype, device=device, requires_grad=True)
    B = torch.randn(num_channels, dtype=dtype, device=device, requires_grad=True)
    return (lambda x, w, b: LigerGroupNormFunction.apply(x, w, b, num_channels, num_groups, 1e-6)), (X, W, B)


def op_poly_norm(rows, cols, dtype, device):
    from liger_kernel.ops.poly_norm import LigerPolyNormFunction

    (X,) = _make_inputs(rows, cols, dtype, device)
    W = torch.randn(3, dtype=dtype, device=device, requires_grad=True)
    # bias is a 0-dim scalar parameter (see LigerPolyNorm), not shape (1,).
    B = torch.tensor(1.0, dtype=dtype, device=device, requires_grad=True)
    # in_place=False so the compiled path does not trip the mutation checker.
    return (lambda x, w, b: LigerPolyNormFunction.apply(x, w, b, 1e-6, False)), (X, W, B)


def op_fused_add_rms_norm(rows, cols, dtype, device):
    from liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction

    X, R = _make_inputs(rows, cols, dtype, device, n=2)
    W = torch.randn(cols, dtype=dtype, device=device, requires_grad=True)
    return (lambda x, r, w: LigerFusedAddRMSNormFunction.apply(x, r, w, 1e-6, 0.0, "llama", False)), (X, R, W)


def op_rms_norm(rows, cols, dtype, device):
    """Already fixed upstream by PR #1350 -- included as a positive control."""
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction

    (X,) = _make_inputs(rows, cols, dtype, device)
    W = torch.randn(cols, dtype=dtype, device=device, requires_grad=True)
    return (lambda x, w: LigerRMSNormFunction.apply(x, w, 1e-6, 0.0, "llama", False)), (X, W)


def _cross_entropy(rows, cols, dtype, device, *, weight=None, label_smoothing=0.0, softcap=None):
    """cross_entropy with one fp64-scalar path enabled at a time.

    Here `cols` is the vocab size and `rows` is BT. The three non-``constexpr``
    float scalars each gate a different arithmetic path, so they are exercised
    separately rather than all at once:

    * ``softcap``      -> ``softcap * tanh(X / softcap)``  (fp64 tanh, the worst)
    * ``weight_sum``   -> only read when label_smoothing > 0
    * ``sum_non_ignore_weight`` -> only a divisor when a class weight is given
    """
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction

    X = torch.randn(rows, cols, dtype=dtype, device=device, requires_grad=True)
    target = torch.randint(0, cols, (rows,), device=device)
    W = torch.rand(cols, dtype=torch.float32, device=device) + 0.5 if weight else None

    def fn(x):
        # Liger CE writes the gradient into its input in place, so the eager arm
        # would corrupt X before the compiled arm runs and every variant would
        # report a bogus output diff. clone() is differentiable, so grads still
        # reach x while the mutation lands on the copy.
        return LigerCrossEntropyFunction.apply(x.clone(), target, W, -100, 0.0, label_smoothing, "mean", softcap)

    return fn, (X,)


def op_cross_entropy_softcap(rows, cols, dtype, device):
    return _cross_entropy(rows, cols, dtype, device, softcap=30.0)


def op_cross_entropy_weighted(rows, cols, dtype, device):
    return _cross_entropy(rows, cols, dtype, device, weight=True)


def op_cross_entropy_smoothing(rows, cols, dtype, device):
    return _cross_entropy(rows, cols, dtype, device, label_smoothing=0.1)


def op_cross_entropy_weighted_smoothing(rows, cols, dtype, device):
    """The only combination that actually reads ``weight_sum``."""
    return _cross_entropy(rows, cols, dtype, device, weight=True, label_smoothing=0.1)


def op_cross_entropy_plain(rows, cols, dtype, device):
    """No optional scalar paths -- shows the floor before any of them are on."""
    return _cross_entropy(rows, cols, dtype, device)


# --------------------------------------------------------------------------
# PR3 / PR4 candidates: loss and norm ops with non-constexpr float scalars.
# --------------------------------------------------------------------------


def op_kl_div(rows, cols, dtype, device):
    """`eps` feeds tl.log(tl.maximum(y_true, eps)) on an fp32 tensor."""
    from liger_kernel.ops.kl_div import LigerKLDivLossFunction

    y_pred = torch.randn(rows, cols, dtype=dtype, device=device).log_softmax(-1).requires_grad_(True)
    y_true = torch.randn(rows, cols, dtype=dtype, device=device).softmax(-1)
    return (lambda x: LigerKLDivLossFunction.apply(x, y_true, "batchmean", False, 1e-10)), (y_pred,)


def op_tvd(rows, cols, dtype, device):
    """`scale` multiplies fp32 gradients and the reduced loss."""
    from liger_kernel.ops.tvd import LigerTVDLossFunction

    p_ = torch.randn(rows, cols, dtype=dtype, device=device).softmax(-1).requires_grad_(True)
    q_ = torch.randn(rows, cols, dtype=dtype, device=device).softmax(-1)
    return (lambda x: LigerTVDLossFunction.apply(x, q_, None, "batchmean", -100)), (p_,)


def _modulated_rms_norm(rows, cols, dtype, device, casting_mode):
    from liger_kernel.ops.modulated_rms_norm import LigerModulatedRMSNormFunction

    (X,) = _make_inputs(rows, cols, dtype, device)
    W = torch.randn(cols, dtype=dtype, device=device, requires_grad=True)
    scale = torch.randn(rows, cols, dtype=dtype, device=device, requires_grad=True)
    return (lambda x, w, sc: LigerModulatedRMSNormFunction.apply(x, w, sc, None, 1e-6, 0.0, casting_mode, False)), (
        X,
        W,
        scale,
    )


def op_modulated_rms_norm(rows, cols, dtype, device):
    """Same shape of bug as fused_add_rms_norm: the eps cast has no else branch,
    so llama/gemma casting modes leave eps fp64."""
    return _modulated_rms_norm(rows, cols, dtype, device, "llama")


def op_modulated_rms_norm_gemma(rows, cols, dtype, device):
    """The other contaminated casting mode."""
    return _modulated_rms_norm(rows, cols, dtype, device, "gemma")


def op_modulated_rms_norm_none(rows, cols, dtype, device):
    """Control: casting_mode=none already casts eps/offset to the input dtype,
    so this path is untouched by the fix and must stay clean and unchanged."""
    return _modulated_rms_norm(rows, cols, dtype, device, "none")


def op_attn_res(rows, cols, dtype, device):
    """`eps` feeds tl.rsqrt(ms + eps). Takes [N, B, T, D]; map cols -> D."""
    from liger_kernel.ops.attn_res import LigerAttnResFunction

    n_blocks, batch = 4, 2
    d = min(cols, 4096)
    seq = max(1, rows // batch)
    V = torch.randn(n_blocks, batch, seq, d, dtype=dtype, device=device, requires_grad=True)
    w_query = torch.randn(d, dtype=dtype, device=device, requires_grad=True)
    w_norm = torch.randn(d, dtype=dtype, device=device, requires_grad=True)
    return (lambda v, wq, wn: LigerAttnResFunction.apply(v, wq, wn, 1e-6)), (V, w_query, w_norm)


def _grpo_loss(
    rows,
    cols,
    dtype,
    device,
    *,
    loss_type="dapo",
    importance_sampling_level="token",
    beta=0.04,
    delta=None,
):
    """grpo_loss with one fp64-scalar path enabled at a time.

    `cols` is the vocab size; `rows` is B*L. Only `logits` carries a gradient.

    grpo_loss has six non-``constexpr`` float scalars spread over five kernels
    (``BETA`` is already ``tl.constexpr``). They do not all reach every kernel,
    so the variants below select between them:

    * ``TEMPERATURE``  -> divides every logit in all five kernels (always live)
    * ``EPS_LOW/HIGH`` -> ``tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)`` for
      grpo/dapo; for cispo ``EPS_HIGH`` is instead a raw one-sided bound
    * ``SAPO_TEMP_POS/NEG`` -> only read when loss_type="sapo"
    * ``DELTA``        -> only read when two-sided clipping is on (delta != None)

    ``importance_sampling_level="sequence"`` swaps in the separate ``_seq``
    forward/backward kernels, which take a different (smaller) scalar set.
    """
    from liger_kernel.transformers.grpo_loss import triton_grpo_loss

    batch = 4
    seq = max(2, rows // batch)

    logits = torch.randn(batch, seq + 1, cols, dtype=dtype, device=device, requires_grad=True)
    completion_ids = torch.randint(0, cols, (batch, seq), device=device)
    old_logp = torch.randn(batch, seq, dtype=torch.float32, device=device) * 0.1 - 2.0
    # beta != 0 requires a reference model's log probs.
    ref_logp = torch.randn(batch, seq, dtype=torch.float32, device=device) * 0.1 - 2.0 if beta != 0.0 else None
    advantages = torch.randn(batch, dtype=torch.float32, device=device)
    completion_mask = torch.ones(batch, seq, dtype=torch.int32, device=device)

    def fn(x):
        # inplace=True makes the forward write the gradient into logits, so the
        # eager arm would corrupt x before the compiled arm runs. clone() is
        # differentiable, so grads still reach x while the mutation lands on the
        # copy (same reason as cross_entropy above).
        return triton_grpo_loss(
            x.clone(),
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask=completion_mask,
            temperature=0.9,
            beta=beta,
            eps_low=0.2,
            eps_high=0.4,
            loss_type=loss_type,
            importance_sampling_level=importance_sampling_level,
            delta=delta,
            reduce=False,
        )

    return fn, (logits,)


def op_grpo_loss_dapo(rows, cols, dtype, device):
    return _grpo_loss(rows, cols, dtype, device, loss_type="dapo")


def op_grpo_loss_cispo(rows, cols, dtype, device):
    return _grpo_loss(rows, cols, dtype, device, loss_type="cispo")


def op_grpo_loss_sapo(rows, cols, dtype, device):
    return _grpo_loss(rows, cols, dtype, device, loss_type="sapo")


def op_grpo_loss_delta(rows, cols, dtype, device):
    """Two-sided clipping, the only path that reads DELTA."""
    return _grpo_loss(rows, cols, dtype, device, loss_type="grpo", delta=2.0)


def op_grpo_loss_seq(rows, cols, dtype, device):
    """Sequence-level IS, which routes through the separate `_seq` kernels."""
    return _grpo_loss(rows, cols, dtype, device, loss_type="grpo", importance_sampling_level="sequence")


OPS = collections.OrderedDict(
    [
        ("fused_add_rms_norm", op_fused_add_rms_norm),
        ("layer_norm", op_layer_norm),
        ("group_norm", op_group_norm),
        ("poly_norm", op_poly_norm),
        ("cross_entropy_plain", op_cross_entropy_plain),
        ("cross_entropy_softcap", op_cross_entropy_softcap),
        ("cross_entropy_weighted", op_cross_entropy_weighted),
        ("cross_entropy_weighted_smoothing", op_cross_entropy_weighted_smoothing),
        ("cross_entropy_smoothing", op_cross_entropy_smoothing),
        ("kl_div", op_kl_div),
        ("tvd", op_tvd),
        ("modulated_rms_norm", op_modulated_rms_norm),
        ("modulated_rms_norm_gemma", op_modulated_rms_norm_gemma),
        ("modulated_rms_norm_none", op_modulated_rms_norm_none),
        ("attn_res", op_attn_res),
        ("grpo_loss_dapo", op_grpo_loss_dapo),
        ("grpo_loss_cispo", op_grpo_loss_cispo),
        ("grpo_loss_sapo", op_grpo_loss_sapo),
        ("grpo_loss_delta", op_grpo_loss_delta),
        ("grpo_loss_seq", op_grpo_loss_seq),
        ("rms_norm", op_rms_norm),
    ]
)

# Shapes to sweep per op, as (rows, cols). Row-wise norms use realistic
# (tokens, hidden) pairs, including 8192x3072 to match the isolated repro in
# #1350. group_norm takes (batch, num_channels * hidden), so cols is the
# per-sample element count; batch is kept modest because its backward becomes
# atomics-bound at large batch with small hidden (a separate pre-existing issue).
_ROW_SHAPES = [(2048, 2048), (8192, 3072), (4096, 4096), (2048, 8192), (16384, 4096)]

SWEEP_SHAPES = {
    "fused_add_rms_norm": _ROW_SHAPES,
    "layer_norm": _ROW_SHAPES,
    "poly_norm": _ROW_SHAPES,
    "rms_norm": _ROW_SHAPES,
    "group_norm": [(16, 131072), (8, 327680), (4, 262144), (32, 65536)],
    "modulated_rms_norm": _ROW_SHAPES,
    "modulated_rms_norm_gemma": _ROW_SHAPES,
    "modulated_rms_norm_none": _ROW_SHAPES,
    # attn_res takes [N, B, T, D]; the driver maps cols -> D (capped at 4096)
    # and rows -> B*T, so keep cols at or below 4096 to avoid silent clamping.
    "attn_res": [(2048, 2048), (4096, 4096), (8192, 2048), (4096, 1024)],
    # cross_entropy takes (BT, vocab). Real vocabularies: 32k (Llama-2),
    # 128256 (Llama-3), 152064 (Qwen). Kept under ~0.5 GB of logits per shape.
    **{
        name: [(4096, 32000), (8192, 32000), (2048, 128256), (2048, 152064)]
        for name in (
            "cross_entropy_plain",
            "cross_entropy_softcap",
            "cross_entropy_weighted",
            "cross_entropy_weighted_smoothing",
            "cross_entropy_smoothing",
        )
    },
    # grpo_loss takes logits (B, L+1, vocab) with B fixed at 4 in the builder,
    # so the driver's `rows` maps to B*L and `cols` to the vocab. Logits here are
    # 3-D, so element counts grow much faster than the 2-D ops above -- shapes
    # are kept under ~0.5 GB of logits (the backward allocates dlogits too).
    **{
        name: [(2048, 32000), (4096, 32000), (1024, 128256), (1024, 152064)]
        for name in (
            "grpo_loss_dapo",
            "grpo_loss_cispo",
            "grpo_loss_sapo",
            "grpo_loss_delta",
            "grpo_loss_seq",
        )
    },
}


def _bytes_moved(op, rows, cols, itemsize):
    """Approximate bytes read+written by one forward+backward step.

    Every one of these ops is bandwidth-bound, so bytes/time is the meaningful
    efficiency number. Counts the dominant X-sized traffic only: forward reads X
    and writes Y; backward reads dY and X and writes dX. fused_add_rms_norm also
    reads a residual and writes the summed output.
    """
    elems = rows * cols
    passes = {
        "fused_add_rms_norm": 8,
        "layer_norm": 5,
        "poly_norm": 5,
        "rms_norm": 5,
        "group_norm": 7,
        # Liger CE writes the gradient into X during forward (read X + write X),
        # then element_mul scales it in backward (read + write).
        "cross_entropy_plain": 4,
        "cross_entropy_softcap": 4,
        "cross_entropy_weighted": 4,
        "cross_entropy_weighted_smoothing": 4,
        "cross_entropy_smoothing": 4,
    }.get(op, 5)
    return elems * itemsize * passes


def _drive(fn, inputs):
    """Forward + backward, returning (output, grads) detached for comparison."""
    for t in inputs:
        if t.grad is not None:
            t.grad = None
    out = fn(*inputs)
    # Some ops (e.g. fused_add_rms_norm) return several tensors; the first is Y.
    if isinstance(out, (tuple, list)):
        out = out[0]
    out.sum().backward()
    return out.detach().clone(), [t.grad.detach().clone() for t in inputs if t.grad is not None]


def _bench(fn, inputs, warmup=5, iters=20):
    """Summed GPU time (ms/step) of the Liger kernels only.

    Wall-clock timing around ``backward()`` is dominated by Python and autograd
    overhead and by torch.compile's dispatch, which swamps the kernel signal --
    it happily reports a *slower* time for a build whose PTX is strictly better.
    So aggregate self-CUDA-time per kernel from the profiler instead, and keep
    only kernels that are ours.
    """
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile

    for _ in range(warmup):
        _drive(fn, inputs)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            _drive(fn, inputs)
        torch.cuda.synchronize()

    total_us = 0.0
    for evt in prof.key_averages():
        if _LIGER_KERNEL_RE.match(evt.key):
            total_us += evt.self_device_time_total
    return total_us / 1e3 / iters


def audit_op(name, builder, rows, cols, dtype, device, bench=False):
    fn, inputs = builder(rows, cols, dtype, device)

    with PTXRecorder() as eager_rec:
        eager_out, eager_grads = _drive(fn, inputs)

    torch._dynamo.reset()
    compiled_fn = torch.compile(fn, fullgraph=False)
    with PTXRecorder() as comp_rec:
        comp_out, comp_grads = _drive(compiled_fn, inputs)

    eager_ptx = eager_rec.summary()
    comp_ptx = comp_rec.summary()

    eager_f64 = sum(v["f64"] for v in eager_ptx.values())
    comp_f64 = sum(v["f64"] for v in comp_ptx.values())

    # One demotion (cvt.rn.f32.f64) per fp64 scalar param is the expected residue
    # of a correct fix. Real contamination shows up as tens-to-hundreds of ops.
    contaminated = comp_f64 > 8

    max_diff = (comp_out.float() - eager_out.float()).abs().max().item()
    grad_diff = max((c.float() - e.float()).abs().max().item() for c, e in zip(comp_grads, eager_grads))

    return {
        "op": name,
        "eager_f64": eager_f64,
        "compiled_f64": comp_f64,
        "contaminated": contaminated,
        "eager_kernels": {k: v for k, v in eager_ptx.items()},
        "compiled_kernels": {k: v for k, v in comp_ptx.items()},
        "max_out_diff": max_diff,
        "max_grad_diff": grad_diff,
        "eager_ms": _bench(fn, inputs) if bench else None,
        "compiled_ms": _bench(compiled_fn, inputs) if bench else None,
        "rows": rows,
        "cols": cols,
    }


def run_sweep(op_names, dtype, device, itemsize):
    """Per-op shape sweep: contamination and bandwidth at several realistic sizes."""
    results = []
    for name in op_names:
        if name not in OPS:
            print(f"  {name:<24} SKIP (unknown op)")
            continue
        print(f"\n{name}")
        print(
            f"  {'shape':<16} {'.f64':>6}  {'eager':>9} {'compiled':>9}  "
            f"{'compiled BW':>12}  {'eager BW':>10}  {'max|d out|':>11}"
        )
        for rows, cols in SWEEP_SHAPES.get(name, [(2048, 4096)]):
            try:
                res = audit_op(name, OPS[name], rows, cols, dtype, device, bench=True)
            except Exception as exc:
                print(f"  {rows}x{cols:<10} ERROR {type(exc).__name__}: {exc}")
                continue
            gb = _bytes_moved(name, rows, cols, itemsize) / 1e9
            cbw = gb / (res["compiled_ms"] / 1e3) if res["compiled_ms"] else 0.0
            ebw = gb / (res["eager_ms"] / 1e3) if res["eager_ms"] else 0.0
            flag = "!" if res["contaminated"] else " "
            print(
                f"  {f'{rows}x{cols}':<16} {res['compiled_f64']:>5}{flag}  "
                f"{res['eager_ms']:>7.4f}ms {res['compiled_ms']:>7.4f}ms  "
                f"{cbw:>9.0f} GB/s  {ebw:>7.0f} GB/s  {res['max_out_diff']:>11.3e}"
            )
            results.append(res)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ops", default=",".join(OPS), help="comma-separated op names")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--json", default=None, help="write results to this path")
    parser.add_argument("--bench", action="store_true", help="also time eager vs compiled")
    parser.add_argument("--sweep", action="store_true", help="sweep several shapes per op")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: this audit requires a CUDA GPU (Triton must emit PTX).", file=sys.stderr)
        return 2

    dtype = getattr(torch, args.dtype)
    device = "cuda"
    import liger_kernel

    print(f"# fp64 scalar audit | {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    print(f"# liger_kernel from {os.path.dirname(liger_kernel.__file__)}")

    op_names = [o.strip() for o in args.ops.split(",") if o.strip()]

    if args.sweep:
        itemsize = torch.empty((), dtype=dtype).element_size()
        print(f"# shape sweep, {args.dtype}; '!' marks fp64 contamination\n")
        results = run_sweep(op_names, dtype, device, itemsize)
        bad = [r for r in results if r.get("contaminated")]
        print()
        print(
            f"FAIL: {len(bad)} shape(s) contaminated"
            if bad
            else "PASS: no fp64 scalar contamination detected at any swept shape."
        )
        if args.json:
            with open(args.json, "w") as fh:
                json.dump({"device": torch.cuda.get_device_name(0), "results": results}, fh, indent=2)
        return 1 if bad else 0

    print(f"# shape {args.rows}x{args.cols} {args.dtype}\n")

    results = []
    for name in op_names:
        if name not in OPS:
            print(f"  {name:<24} SKIP (unknown op)")
            continue
        try:
            res = audit_op(name, OPS[name], args.rows, args.cols, dtype, device, bench=args.bench)
        except Exception as exc:  # keep auditing the remaining ops
            print(f"  {name:<24} ERROR {type(exc).__name__}: {exc}")
            results.append({"op": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        results.append(res)
        verdict = "CONTAMINATED" if res["contaminated"] else "clean"
        print(
            f"  {name:<24} eager .f64={res['eager_f64']:<5} compiled .f64={res['compiled_f64']:<5} "
            f"{verdict:<13} max|d out|={res['max_out_diff']:.3e} grad={res['max_grad_diff']:.3e}"
        )
        for kname, counts in res["compiled_kernels"].items():
            print(f"        {kname:<52} .f64={counts['f64']:<5} .f32={counts['f32']}")
        if res.get("compiled_ms"):
            print(
                f"        {'liger kernel GPU time / step':<52} "
                f"eager={res['eager_ms']:.4f} ms  compiled={res['compiled_ms']:.4f} ms"
            )

    bad = [r for r in results if r.get("contaminated")]
    print()
    if bad:
        print(f"FAIL: {len(bad)} op(s) run in float64 under torch.compile: {', '.join(r['op'] for r in bad)}")
    else:
        print("PASS: no fp64 scalar contamination detected.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"device": torch.cuda.get_device_name(0), "results": results}, fh, indent=2)
        print(f"wrote {args.json}")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
