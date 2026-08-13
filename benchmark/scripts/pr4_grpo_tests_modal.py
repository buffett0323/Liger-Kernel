"""Run the grpo_loss unit tests for PR4 on a Modal GPU.

The combined audit runner kept getting its container killed while running the
grpo test files, so this script runs *only* pytest, one file per invocation, so
a crash in one file cannot hide the result of the others.
"""

import pathlib

import modal

REMOTE_ROOT = "/root/liger"
ONE_HOUR = 60 * 60

IS_LOCAL = modal.is_local()
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2] if IS_LOCAL else pathlib.Path(REMOTE_ROOT)

app = modal.App("liger-pr4-grpo-tests")

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
    image = image.add_local_dir(str(REPO_ROOT / "src" / "liger_kernel"), f"{REMOTE_ROOT}/liger_kernel").add_local_dir(
        str(REPO_ROOT / "test"), f"{REMOTE_ROOT}/test"
    )

TEST_FILES = [
    "test/transformers/test_grpo_loss.py",
    # chunked_loss/grpo_loss.py is a pure-PyTorch impl that never imports
    # ops/grpo_loss.py, so it cannot exercise this change; it is also memory
    # heavy enough to get the container OOM-killed. Excluded deliberately.
]


@app.function(image=image, gpu="H100", timeout=ONE_HOUR)
def run_tests():
    import os
    import subprocess as sp

    env = dict(os.environ, PYTHONPATH=REMOTE_ROOT, TORCHINDUCTOR_COMPILE_THREADS="1")
    results = {}
    for f in TEST_FILES:
        print("\n" + "=" * 78 + f"\n{f}\n" + "=" * 78, flush=True)
        if not os.path.isfile(os.path.join(REMOTE_ROOT, f)):
            results[f] = "MISSING"
            continue
        p = sp.run(
            ["python", "-m", "pytest", "-q", "--no-header", "-rN", f],
            env=env,
            cwd=REMOTE_ROOT,
            capture_output=True,
            text=True,
        )
        print(p.stdout[-4000:], flush=True)
        if p.stderr.strip():
            print("STDERR:", p.stderr[-2000:], flush=True)
        summary = [ln for ln in p.stdout.strip().split("\n") if "passed" in ln or "failed" in ln or "error" in ln]
        results[f] = (summary[-1] if summary else f"no summary (rc={p.returncode})").strip()

    print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78, flush=True)
    for f, r in results.items():
        print(f"  {f}\n      {r}", flush=True)


@app.local_entrypoint()
def main():
    run_tests.remote()
