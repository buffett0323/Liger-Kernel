"""Modal launcher for the fp64 scalar audit and the norm-op unit tests.

Runs, on a real GPU:

1. ``fp64_scalar_audit.py`` against a *baseline* git ref (default ``main``,
   i.e. before this branch's fix) -- expected to report CONTAMINATED.
2. The same audit against the live worktree -- expected to report clean.
3. ``pytest`` for the affected ops against the worktree, to prove the casts do
   not change numerics.

Together those three give a complete before/after for a PR: proof the bug was
real, proof it is gone, and proof nothing else moved.

Setup::

    pip install modal
    modal setup

Run::

    modal run benchmark/scripts/fp64_audit_modal_runner.py
    modal run benchmark/scripts/fp64_audit_modal_runner.py --gpu H100
    modal run benchmark/scripts/fp64_audit_modal_runner.py --gpu B200 --skip-tests

Any GPU works -- the fp64 promotion shows up in PTX on every architecture, even
those where it happens to cost nothing at runtime. H100 is the cheapest choice.
"""

import pathlib
import subprocess

import modal

TOOLS_ROOT = "/root/tools"
REMOTE_ROOT = "/root/liger"
BASELINE_ROOT = "/root/liger_baseline"
ONE_HOUR = 60 * 60

# The ref to compare the worktree against: `main` still has the unfixed norm ops.
BASELINE_REF = "HEAD"

IS_LOCAL = modal.is_local()
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2] if IS_LOCAL else pathlib.Path(REMOTE_ROOT)

app = modal.App("liger-fp64-scalar-audit")


def _export_ref(ref: str) -> pathlib.Path:
    """Materialize ``src/liger_kernel`` at ``ref`` into /tmp via ``git archive``."""
    export_root = pathlib.Path("/tmp") / f"liger-fp64-{ref.replace('/', '_')}"
    package = export_root / "src" / "liger_kernel"
    if package.is_dir():
        subprocess.run(["rm", "-rf", str(export_root)], check=True)
    export_root.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", ref, "src/liger_kernel"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(export_root)], input=archive, check=True)
    return package


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.11.0+cu130",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install("triton", "packaging", "pytest", "datasets", "transformers")
)

if IS_LOCAL:
    image = (
        image.add_local_dir(str(REPO_ROOT / "src" / "liger_kernel"), f"{REMOTE_ROOT}/liger_kernel")
        .add_local_dir(str(_export_ref(BASELINE_REF)), f"{BASELINE_ROOT}/liger_kernel")
        .add_local_file(
            str(REPO_ROOT / "benchmark" / "scripts" / "fp64_scalar_audit.py"),
            f"{TOOLS_ROOT}/fp64_scalar_audit.py",
        )
        .add_local_dir(str(REPO_ROOT / "test"), f"{REMOTE_ROOT}/test")
    )

OPS = "grpo_loss_dapo,grpo_loss_cispo,grpo_loss_sapo,grpo_loss_delta,grpo_loss_seq,rms_norm"

# rms_norm is the positive control: already fixed upstream by PR #1350, so it
# should read clean in *both* arms.
TEST_FILES = [
    "test/transformers/test_grpo_loss.py",
    "test/chunked_loss/test_grpo_loss.py",
    "test/transformers/test_rms_norm.py",
]


def _run(gpu_label: str, skip_tests: bool, ops: str = None, rows: int = 2048, cols: int = 4096, sweep: bool = False):
    import os
    import subprocess as sp

    env = dict(os.environ, TORCHINDUCTOR_COMPILE_THREADS="1")

    def section(title):
        print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)

    # The audit script lives in its own directory on purpose: Python puts the
    # script's directory at sys.path[0], which would otherwise shadow PYTHONPATH
    # and make both arms import the same liger_kernel.
    def audit(pythonpath):
        cmd = [
            "python",
            f"{TOOLS_ROOT}/fp64_scalar_audit.py",
            "--ops",
            ops or OPS,
            "--bench",
            "--rows",
            str(rows),
            "--cols",
            str(cols),
        ]
        if sweep:
            cmd.append("--sweep")
        return sp.run(
            cmd,
            env=dict(env, PYTHONPATH=pythonpath),
            cwd=TOOLS_ROOT,
        )

    section(f"[{gpu_label}] BASELINE ({BASELINE_REF}) -- expect CONTAMINATED on the unfixed ops")
    print(f"importing liger_kernel from {BASELINE_ROOT}", flush=True)
    audit(BASELINE_ROOT)

    section(f"[{gpu_label}] WORKTREE (fixed) -- expect all clean")
    print(f"importing liger_kernel from {REMOTE_ROOT}", flush=True)
    fixed = audit(REMOTE_ROOT)

    if not skip_tests:
        section(f"[{gpu_label}] UNIT TESTS (worktree) -- expect all pass")
        existing = [f for f in TEST_FILES if os.path.isfile(os.path.join(REMOTE_ROOT, f))]
        sp.run(
            ["python", "-m", "pytest", "-q", "--no-header", *existing],
            env=dict(env, PYTHONPATH=REMOTE_ROOT),
            cwd=REMOTE_ROOT,
        )

    print(f"\naudit exit code (worktree arm, 0 == clean): {fixed.returncode}", flush=True)


@app.function(image=image, gpu="H100", timeout=ONE_HOUR)
def run_h100(skip_tests: bool = False, ops: str = None, rows: int = 2048, cols: int = 4096, sweep: bool = False):
    _run("H100", skip_tests, ops, rows, cols, sweep)


@app.function(image=image, gpu="B200", timeout=ONE_HOUR)
def run_b200(skip_tests: bool = False, ops: str = None, rows: int = 2048, cols: int = 4096, sweep: bool = False):
    _run("B200", skip_tests, ops, rows, cols, sweep)


@app.function(image=image, gpu="B300", timeout=ONE_HOUR)
def run_b300(skip_tests: bool = False, ops: str = None, rows: int = 2048, cols: int = 4096, sweep: bool = False):
    _run("B300", skip_tests, ops, rows, cols, sweep)


ENTRYPOINTS = {"H100": run_h100, "B200": run_b200, "B300": run_b300}


@app.local_entrypoint()
def main(
    gpu: str = "H100",
    skip_tests: bool = False,
    ops: str = None,
    rows: int = 2048,
    cols: int = 4096,
    sweep: bool = False,
):
    for label in [g.strip().upper() for g in gpu.split(",") if g.strip()]:
        if label not in ENTRYPOINTS:
            print(f"unknown gpu {label!r}; choose from {', '.join(ENTRYPOINTS)}")
            continue
        ENTRYPOINTS[label].remote(skip_tests=skip_tests, ops=ops, rows=rows, cols=cols, sweep=sweep)
