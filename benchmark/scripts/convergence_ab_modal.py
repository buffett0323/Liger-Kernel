"""A/B a subset of convergence tests between a baseline git ref and the worktree.

``make test-convergence`` reports failures, but a failure list alone cannot say
whether a branch caused them. This runs the *same* tests twice in one container
-- once with ``src/liger_kernel`` swapped to a baseline ref, once with the
worktree -- so identical failures on both sides prove the branch is innocent.

Run::

    modal run --detach benchmark/scripts/convergence_ab_modal.py
    modal run benchmark/scripts/convergence_ab_modal.py --show-log verdict

``--detach`` keeps the run alive server-side if the local machine sleeps or
drops off the network; the verdict and both pytest logs are written to a Modal
Volume so they can be retrieved afterwards.
"""

import pathlib
import subprocess

import modal

REMOTE_ROOT = "/root/liger"
BASELINE_SRC = "/root/baseline_liger_kernel"
RESULTS_DIR = "/results"
THREE_HOURS = 3 * 60 * 60
# The parent of the fix commit -- NOT "main". A topic branch is often built on a
# newer upstream tip than the local main, and diffing against main would fold
# unrelated upstream commits into the "branch" arm and misattribute their
# failures to this change.
BASELINE_REF = "HEAD~1"

# Set via --test to isolate a single case; otherwise the full bf16 set.
TESTS = [
    "test/convergence/bf16/test_mini_models.py::test_mini_model",
    "test/convergence/bf16/test_mini_models_multimodal.py::test_mini_model_multimodal",
    "test/convergence/bf16/test_mini_models_with_logits.py::test_mini_model",
]

IS_LOCAL = modal.is_local()
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2] if IS_LOCAL else pathlib.Path(REMOTE_ROOT)

app = modal.App("liger-convergence-ab")
results_volume = modal.Volume.from_name("liger-convergence-ab-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "make")
    .pip_install(
        "torch==2.11.0+cu130",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install("triton", "packaging")
)


def _export_ref(ref: str) -> pathlib.Path:
    """Materialize ``src/liger_kernel`` at ``ref`` into /tmp via ``git archive``."""
    export_root = pathlib.Path("/tmp") / f"liger-conv-{ref.replace('/', '_')}"
    if export_root.is_dir():
        subprocess.run(["rm", "-rf", str(export_root)], check=True)
    export_root.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", ref, "src/liger_kernel"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(export_root)], input=archive, check=True)
    return export_root / "src" / "liger_kernel"


if IS_LOCAL:
    image = image.add_local_dir(
        str(REPO_ROOT),
        REMOTE_ROOT,
        ignore=[
            "**/.venv/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/.git/**",
            "**/benchmark/data/**",
            "**/*.egg-info/**",
            "**/build/**",
            "**/.pytest_cache/**",
            "**/.ruff_cache/**",
        ],
    ).add_local_dir(str(_export_ref(BASELINE_REF)), BASELINE_SRC)


def _pytest(env, label, slug, tests, repeat=1):
    import re
    import subprocess as sp

    print("\n" + "=" * 78 + f"\n{label}\n" + "=" * 78, flush=True)
    all_runs = []
    full_out = []
    for i in range(repeat):
        proc = sp.run(
            ["python", "-m", "pytest", "--disable-warnings", "-q", "--no-header", "-rf", *tests],
            cwd=REMOTE_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        out = proc.stdout + proc.stderr
        full_out.append(f"----- run {i + 1}/{repeat} -----\n{out}")
        failed = sorted(set(re.findall(r"^FAILED (\S+)", out, re.M)))
        summary = [ln for ln in out.splitlines() if " passed" in ln or " failed" in ln]
        tag = f"  run {i + 1}/{repeat}:" if repeat > 1 else " "
        for f in failed:
            print(f"{tag} FAILED {f}", flush=True)
        print(f"{tag} {summary[-1] if summary else 'no summary'}", flush=True)
        all_runs.append(set(failed))

    with open(f"{RESULTS_DIR}/{slug}.log", "w") as fh:
        fh.write("\n".join(full_out))
    results_volume.commit()
    return all_runs


@app.function(image=image, gpu="H100", timeout=THREE_HOURS, volumes={RESULTS_DIR: results_volume})
def _run(tests=None, repeat: int = 1):
    import os
    import shutil
    import subprocess as sp

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sp.run(["pip", "install", "-e", ".[dev]"], cwd=REMOTE_ROOT, check=True)
    env = dict(os.environ, HF_DATASETS_OFFLINE="1")

    # `.[dev]` is unpinned and silently upgrades torch/triton/transformers past
    # the image pins, so record what actually ran -- it changes how results
    # should be reported.
    versions = sp.run(
        [
            "python",
            "-c",
            "import torch, triton, transformers;"
            "print(f'torch={torch.__version__} triton={triton.__version__} transformers={transformers.__version__}')",
        ],
        cwd=REMOTE_ROOT,
        env=env,
        text=True,
        capture_output=True,
    ).stdout.strip()
    print(f"environment: {versions}", flush=True)

    live = f"{REMOTE_ROOT}/src/liger_kernel"
    stash = f"{REMOTE_ROOT}/src/_liger_kernel_worktree"

    tests = list(tests) if tests else list(TESTS)
    branch_runs = _pytest(env, f"WORKTREE (branch) x{repeat}", "branch", tests, repeat)

    # Swap in the baseline package in place, so the *same* installed distribution
    # and the same test files are used; only the kernel source differs.
    shutil.move(live, stash)
    shutil.copytree(BASELINE_SRC, live)
    try:
        baseline_runs = _pytest(env, f"BASELINE ({BASELINE_REF}) x{repeat}", "baseline", tests, repeat)
    finally:
        shutil.rmtree(live)
        shutil.move(stash, live)

    print("\n" + "=" * 78 + "\nVERDICT\n" + "=" * 78, flush=True)

    def rate(runs, test):
        return sum(test in r for r in runs)

    seen = sorted({t for r in branch_runs + baseline_runs for t in r})
    lines = [f"environment: {versions}", f"repeats per arm: {repeat}", ""]
    lines.append(f"{'test':<70} branch  baseline")
    regressions, flaky = [], []
    for t in seen:
        b, m = rate(branch_runs, t), rate(baseline_runs, t)
        short = t.split("::")[-1][:68]
        lines.append(f"{short:<70} {b}/{repeat}     {m}/{repeat}")
        # A regression must fail on EVERY branch run and pass on EVERY baseline
        # run. Anything in between is nondeterminism, not attribution.
        if b == repeat and m == 0:
            regressions.append(t)
        elif b != m:
            flaky.append(t)

    lines += ["", f"consistent regressions (branch {repeat}/{repeat}, baseline 0/{repeat}): {len(regressions)}"]
    lines += [f"  {t}" for t in regressions]
    lines += [f"inconsistent / flaky (differ but not consistently): {len(flaky)}"]
    lines += [f"  {t}" for t in flaky]
    lines.append(
        "=> this branch INTRODUCED convergence failures"
        if regressions
        else "=> no convergence regression attributable to this branch"
    )
    for ln in lines:
        print(ln, flush=True)
    with open(f"{RESULTS_DIR}/verdict.log", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    results_volume.commit()
    return regressions


@app.function(image=image, volumes={RESULTS_DIR: results_volume})
def show(name: str = "verdict", lines: int = 60):
    import os

    results_volume.reload()
    path = f"{RESULTS_DIR}/{name}.log"
    if not os.path.exists(path):
        print(f"no such log: {name}; available: {sorted(os.listdir(RESULTS_DIR))}")
        return
    with open(path) as fh:
        body = fh.read().splitlines()
    print("\n".join(body[-lines:]))


@app.local_entrypoint()
def main(show_log: str = "", lines: int = 60, test: str = "", repeat: int = 1):
    if show_log:
        show.remote(name=show_log, lines=lines)
        return
    tests = [t for t in test.split(",") if t] or None
    regressions = _run.remote(tests=tests, repeat=repeat)
    print(f"\nconsistent regressions: {regressions}")
