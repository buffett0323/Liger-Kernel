"""Run the repo's ``make test`` / ``make test-convergence`` on a GPU via Modal.

Local dev machines without CUDA cannot tick the PR checklist boxes honestly.
This mounts the working tree, installs it with ``pip install -e ".[dev]"`` and
runs the real make targets so results can be reported as measured.

Use ``--detach`` and retrieve results from the Modal volume afterwards, so a
flaky local network cannot kill a long test run.

Run::

    modal run --detach benchmark/scripts/run_make_test_modal.py --gpu H100 --target test
    modal run benchmark/scripts/run_make_test_modal.py --show-log test_H100
"""

import pathlib

import modal

REMOTE_ROOT = "/root/liger"
RESULTS_DIR = "/results"
THREE_HOURS = 3 * 60 * 60

IS_LOCAL = modal.is_local()
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2] if IS_LOCAL else pathlib.Path(REMOTE_ROOT)

app = modal.App("liger-make-test")
results_volume = modal.Volume.from_name("liger-make-test-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "make")
    .pip_install(
        "torch==2.11.0+cu130",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install("triton", "packaging")
)

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
    )


def _run(gpu_label: str, target: str):
    import os
    import subprocess as sp

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[{gpu_label}] installing liger-kernel with dev extras", flush=True)
    sp.run(["pip", "install", "-e", ".[dev]"], cwd=REMOTE_ROOT, check=True)

    targets = ["test", "test-convergence"] if target == "all" else [target]

    summary = []
    for tgt in targets:
        log_path = f"{RESULTS_DIR}/{tgt}_{gpu_label}.log"
        print("\n" + "=" * 78 + f"\n[{gpu_label}] make {tgt} -> {log_path}\n" + "=" * 78, flush=True)
        with open(log_path, "w") as fh:
            proc = sp.Popen(
                ["make", tgt],
                cwd=REMOTE_ROOT,
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                fh.write(line)
                # Echo only summary-ish lines; the full log lives in the volume.
                if any(k in line for k in ("passed", "failed", "error", "FAILED", "ERROR")):
                    print(line.rstrip(), flush=True)
            proc.wait()

        with open(log_path) as fh:
            tail = [ln for ln in fh.read().strip().splitlines() if ln.strip()]
        summary.append((tgt, proc.returncode, tail[-1] if tail else ""))
        results_volume.commit()

    print("\n" + "=" * 78, flush=True)
    for tgt, code, tail in summary:
        print(f"[{gpu_label}] make {tgt}: exit={code} | {tail}", flush=True)
    return summary


@app.function(image=image, gpu="H100", timeout=THREE_HOURS, volumes={RESULTS_DIR: results_volume})
def run_h100(target: str = "all"):
    return _run("H100", target)


@app.function(image=image, gpu="B300", timeout=THREE_HOURS, volumes={RESULTS_DIR: results_volume})
def run_b300(target: str = "all"):
    return _run("B300", target)


@app.function(image=image, volumes={RESULTS_DIR: results_volume})
def show(name: str = "", lines: int = 40):
    """Print the tail of a stored log; ``name`` is e.g. ``test_H100``."""
    import os

    results_volume.reload()
    available = os.listdir(RESULTS_DIR)
    path = f"{RESULTS_DIR}/{name}.log"
    if not name or not os.path.isfile(path):
        print(f"available logs: {available}")
        return
    with open(path) as fh:
        content = fh.read().splitlines()
    print("\n".join(content[-lines:]))


ENTRYPOINTS = {"H100": run_h100, "B300": run_b300}


@app.local_entrypoint()
def main(gpu: str = "H100", target: str = "all", show_log: str = "", lines: int = 40):
    if show_log:
        show.remote(show_log, lines)
        return
    for label in [g.strip().upper() for g in gpu.split(",") if g.strip()]:
        if label not in ENTRYPOINTS:
            print(f"unknown gpu {label!r}; choose from {', '.join(ENTRYPOINTS)}")
            continue
        ENTRYPOINTS[label].remote(target=target)
