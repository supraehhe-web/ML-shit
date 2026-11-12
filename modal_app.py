"""Thin Modal wrapper. All library code runs without Modal.

Usage examples:
  modal run --detach modal_app.py::download_data
  modal run --detach modal_app.py::vit_train --arm A --seed 42
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import modal

APP_NAME = "dm-ahr-vit"
VOLUME_NAME = "dmahr-vit-vol"
REMOTE_ROOT = "/root/proj"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git", "ffmpeg")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "pillow==11.0.0",
        "opencv-python-headless==4.10.0.84",
        "scikit-image==0.24.0",
        "scikit-learn==1.5.2",
        "scipy==1.14.1",
        "matplotlib==3.9.2",
        "tqdm==4.67.1",
        "requests==2.32.3",
        "gdown==5.2.0",
    )
    .env(
        {
            "DATA_ROOT": "/vol/data",
            "ARTIFACT_ROOT": "/vol/artifacts",
            "PYTHONPATH": f"{REMOTE_ROOT}/src",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir("src", f"{REMOTE_ROOT}/src")
    .add_local_dir("scripts", f"{REMOTE_ROOT}/scripts")
    .add_local_dir("configs", f"{REMOTE_ROOT}/configs")
    .add_local_dir("third_party", f"{REMOTE_ROOT}/third_party")
)


def _run(script: str, args: list[str] | None = None) -> int:
    cmd = [sys.executable, f"{REMOTE_ROOT}/scripts/{script}", *(args or [])]
    print("RUN", " ".join(cmd), flush=True)
    started = time.time()
    # Commit periodically so checkpoints written by the child survive
    # preemption. The interval is generous because committing a volume holding
    # tens of thousands of files is expensive, and several shards committing
    # every minute will starve the workers of I/O.
    commit_interval = int(os.environ.get("VOLUME_COMMIT_SECONDS", "900"))
    stop = threading.Event()

    def _committer() -> None:
        while not stop.wait(commit_interval):
            try:
                vol.commit()
            except Exception as exc:  # noqa: BLE001
                print("volume commit failed:", exc, flush=True)

    committer = threading.Thread(target=_committer, daemon=True)
    committer.start()
    tail: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd, cwd=REMOTE_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line)
            del tail[:-60]
        code = proc.wait()
    finally:
        stop.set()
        committer.join(timeout=120)
    print(f"EXIT {code} in {time.time() - started:.1f}s", flush=True)
    vol.commit()
    if code != 0:
        raise RuntimeError(
            f"{script} failed with exit code {code}\n" + "".join(tail[-25:])
        )
    return code


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={"/vol": vol})
def hello_gpu():
    import torch

    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("cuda available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
        print("capability", torch.cuda.get_device_capability(0))
        print("memory_gb", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    return {"cuda": torch.cuda.is_available()}


@app.function(image=image, gpu="A100-80GB", timeout=900, volumes={"/vol": vol})
def gpu_probe():
    import json
    import platform

    import torch

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        if torch.cuda.is_available()
        else None,
        "nvidia_smi": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    os.makedirs("/vol/artifacts/logs", exist_ok=True)
    with open("/vol/artifacts/logs/environment_report.json", "w") as fh:
        json.dump(info, fh, indent=2)
    vol.commit()
    print(json.dumps(info, indent=2))
    return info


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": vol})
def download_data(limit: int | None = None):
    args = ["--workers", "24"]
    if limit:
        args += ["--limit", str(limit)]
    return _run("00_download_data.py", args)


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": vol})
def build_manifests(force_manifest: bool = False, force_confirm: bool = False):
    args = []
    if force_manifest:
        args.append("--force-manifest")
    if force_confirm:
        args.append("--force-confirm")
    return _run("01_audit_and_splits.py", args)


@app.function(image=image, cpu=8, memory=32768, timeout=14400, volumes={"/vol": vol})
def hair_pipeline():
    return _run("02_hair_pipeline.py")


@app.function(image=image, cpu=8, memory=16384, timeout=3600, volumes={"/vol": vol})
def fill_c_passthrough():
    return _run("05b_fill_c_passthrough.py")


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def dmahr_smoke():
    return _run("03_dmahr_smoke.py")


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def dmahr_bench():
    """Measure whether bf16 / torch.compile cut per-image sampling cost."""
    return _run("03b_dmahr_bench.py")


# The sampler uses ~1.5 GB and runs 2,000 small UNet calls, so it is likely
# launch-overhead bound rather than compute bound. These variants measure the
# real cost on cheaper GPUs, which is what decides the budget.
@app.function(image=image, gpu="L4", timeout=3600, volumes={"/vol": vol})
def dmahr_bench_l4():
    return _run("03b_dmahr_bench.py", ["--tag", "l4"])


@app.function(image=image, gpu="A10", timeout=3600, volumes={"/vol": vol})
def dmahr_bench_a10():
    return _run("03b_dmahr_bench.py", ["--tag", "a10"])


@app.function(image=image, gpu="T4", timeout=3600, volumes={"/vol": vol})
def dmahr_bench_t4():
    return _run("03b_dmahr_bench.py", ["--tag", "t4"])


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def dmahr_batch_sweep(batches: str = "32,64,128,256"):
    return _run("03c_batch_sweep.py", ["--batches", batches, "--tag", "a100"])


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": vol})
def dmahr_train(overfit: bool = False):
    return _run("04_dmahr_train.py", ["--overfit"] if overfit else [])


# Arm C diagnosis. Runs on validation synthetic pairs only, so no test-set
# choice depends on it. A100-40GB is the cheapest card measured to run the
# sampler without becoming the bottleneck.
@app.function(image=image, gpu="A100-40GB", timeout=3600, volumes={"/vol": vol})
def arm_c_diagnose(identity_n: int = 50, pairs_n: int = 100):
    return _run("17_arm_c_diagnose.py",
                ["--identity-n", str(identity_n), "--pairs-n", str(pairs_n)])


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": vol})
def dmahr_overfit():
    return _run("04_dmahr_train.py", ["--overfit"])


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": vol})
def dmahr_sample(shard: int, n_shards: int, pairs_split: str = ""):
    args = ["--shard", str(shard), "--n-shards", str(n_shards)]
    if pairs_split:
        args += ["--pairs-split", pairs_split]
    return _run("05_dmahr_sample.py", args)


# Sampling peaks at ~1.5 GB, so a mid-range GPU is enough and has capacity when
# the A100 pool is saturated.
@app.function(image=image, gpu="A10G", timeout=86400, volumes={"/vol": vol})
def dmahr_sample_a10g(shard: int, n_shards: int, pairs_split: str = ""):
    args = ["--shard", str(shard), "--n-shards", str(n_shards)]
    if pairs_split:
        args += ["--pairs-split", pairs_split]
    return _run("05_dmahr_sample.py", args)


# L4 is $0.80/hr vs $2.50/hr for A100-80GB. Sampling peaks at 1.5 GB and is
# launch-overhead bound, so L4 is the budget-viable choice for Arm C.
@app.function(image=image, gpu="L4", timeout=86400, volumes={"/vol": vol})
def dmahr_sample_l4(shard: int, n_shards: int, pairs_split: str = ""):
    args = ["--shard", str(shard), "--n-shards", str(n_shards)]
    if pairs_split:
        args += ["--pairs-split", pairs_split]
    return _run("05_dmahr_sample.py", args)


# A100 40GB is $2.10/hr vs $2.50/hr for 80GB with identical compute, and
# sampling peaks at 1.5 GB, so 40GB is the budget-optimal choice.
@app.function(image=image, gpu="A100-40GB", timeout=86400, volumes={"/vol": vol})
def dmahr_sample_a100_40(shard: int, n_shards: int, pairs_split: str = ""):
    args = ["--shard", str(shard), "--n-shards", str(n_shards)]
    if pairs_split:
        args += ["--pairs-split", pairs_split]
    return _run("05_dmahr_sample.py", args)


@app.function(image=image, gpu="A100-40GB", timeout=86400, volumes={"/vol": vol})
def dmahr_sample_bf16(shard: int, n_shards: int, pairs_split: str = ""):
    args = ["--shard", str(shard), "--n-shards", str(n_shards), "--bf16"]
    if pairs_split:
        args += ["--pairs-split", pairs_split]
    return _run("05_dmahr_sample.py", args)


# Budget-forced ablation: 1,000 sampling steps instead of the reference 2,000.
# Halves Arm C cost so the full 8,993-image run fits the remaining credits.
# Recorded as an ablation in the decision log and the report, never as the
# paper's method.
@app.function(image=image, gpu="A100-40GB", timeout=86400, volumes={"/vol": vol})
def dmahr_sample_ablation(shard: int, n_shards: int, pairs_split: str = ""):
    args = ["--shard", str(shard), "--n-shards", str(n_shards),
            "--bf16", "--num-timesteps", "1000"]
    if pairs_split:
        args += ["--pairs-split", pairs_split]
    return _run("05_dmahr_sample.py", args)


@app.function(image=image, cpu=1, memory=2048, timeout=86400, volumes={"/vol": vol})
def dmahr_sample_driver_ablation(n_shards: int = 4, pairs_split: str = "test"):
    handles = [dmahr_sample_ablation.spawn(i, n_shards, "") for i in range(n_shards)]
    if pairs_split:
        handles.append(dmahr_sample_ablation.spawn(0, 1, pairs_split))
    results = [h.get() for h in handles]
    print("all shards finished:", results, flush=True)
    return results


@app.function(image=image, cpu=1, memory=2048, timeout=86400, volumes={"/vol": vol})
def dmahr_sample_driver_bf16(n_shards: int = 4, pairs_split: str = "test"):
    handles = [dmahr_sample_bf16.spawn(i, n_shards, "") for i in range(n_shards)]
    if pairs_split:
        handles.append(dmahr_sample_bf16.spawn(0, 1, pairs_split))
    results = [h.get() for h in handles]
    print("all shards finished:", results, flush=True)
    return results


@app.function(image=image, cpu=1, memory=2048, timeout=86400, volumes={"/vol": vol})
def dmahr_sample_driver_a100_40(n_shards: int = 4, pairs_split: str = "test"):
    handles = [dmahr_sample_a100_40.spawn(i, n_shards, "") for i in range(n_shards)]
    if pairs_split:
        handles.append(dmahr_sample_a100_40.spawn(0, 1, pairs_split))
    results = [h.get() for h in handles]
    print("all shards finished:", results, flush=True)
    return results


@app.function(image=image, cpu=1, memory=2048, timeout=86400, volumes={"/vol": vol})
def dmahr_sample_driver_l4(n_shards: int = 4, pairs_split: str = "test"):
    handles = [dmahr_sample_l4.spawn(i, n_shards, "") for i in range(n_shards)]
    if pairs_split:
        handles.append(dmahr_sample_l4.spawn(0, 1, pairs_split))
    results = [h.get() for h in handles]
    print("all shards finished:", results, flush=True)
    return results


@app.function(image=image, cpu=1, memory=2048, timeout=86400, volumes={"/vol": vol})
def dmahr_sample_driver_a10g(n_shards: int = 4, pairs_split: str = "test"):
    handles = [dmahr_sample_a10g.spawn(i, n_shards, "") for i in range(n_shards)]
    if pairs_split:
        handles.append(dmahr_sample_a10g.spawn(0, 1, pairs_split))
    results = [h.get() for h in handles]
    print("all shards finished:", results, flush=True)
    return results


@app.local_entrypoint()
def sample_all(n_shards: int = 4):
    """Run Arm C for every hair-present image, then for the held-out test pairs."""
    print("launching", n_shards, "shards for hair-present images", flush=True)
    for result in dmahr_sample.map(range(n_shards), [n_shards] * n_shards, [""] * n_shards):
        print("shard result", result, flush=True)
    print("processing held-out synthetic test pairs", flush=True)
    print(dmahr_sample.remote(0, 1, "test"), flush=True)


@app.function(image=image, cpu=1, memory=2048, timeout=86400, volumes={"/vol": vol})
def dmahr_sample_driver(n_shards: int = 4, pairs_split: str = "test"):
    """Spawn Arm C shards and block until every one finishes.

    Returning immediately would let the app tear down and kill the spawned
    shards, so the driver waits on the handles.
    """
    handles = [dmahr_sample.spawn(i, n_shards, "") for i in range(n_shards)]
    if pairs_split:
        handles.append(dmahr_sample.spawn(0, 1, pairs_split))
    results = [h.get() for h in handles]
    print("all shards finished:", results, flush=True)
    return results


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": vol})
def vit_train(arm: str, seed: int):
    return _run("06_train_vit.py", ["--arm", arm, "--seed", str(seed)])


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": vol})
def vit_sweep(runs_json: str):
    """Train several (arm, seed) pairs in one container, sequentially."""
    import json

    runs = json.loads(runs_json)
    results = []
    for arm, seed in runs:
        results.append(_run("06_train_vit.py", ["--arm", arm, "--seed", str(seed)]))
    return results


@app.function(image=image, gpu="A100-80GB", timeout=14400, volumes={"/vol": vol})
def vit_eval(arm: str, seed: int):
    return _run("07_eval_vit.py", ["--arm", arm, "--seed", str(seed)])


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": vol})
def image_quality(split: str = "test"):
    return _run("07b_image_quality.py", ["--split", split])


@app.function(image=image, gpu="A100-80GB", timeout=14400, volumes={"/vol": vol})
def xai_run(arm: str, seed: int):
    return _run("08_xai.py", ["--arm", arm, "--seed", str(seed)])


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": vol})
def stats_run():
    return _run("09_stats.py")


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes={"/vol": vol})
def acceptance():
    return _run("check_acceptance.py")


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes={"/vol": vol})
def finalize():
    return _run("10_finalize.py")


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": vol})
def eval_and_xai_all(arms: str = "A,B,Astar,C"):
    """Locked test evaluation then XAI for every checkpoint that exists."""
    done = []
    for arm in [a for a in arms.split(",") if a]:
        for seed in (42, 123, 2026):
            ckpt = f"/vol/artifacts/checkpoints/vit/{arm}_seed{seed}/best.pt"
            if not os.path.exists(ckpt):
                print("skip (no checkpoint):", ckpt, flush=True)
                continue
            pred = f"/vol/artifacts/predictions/metrics_{arm}_seed{seed}.json"
            if not os.path.exists(pred):
                _run("07_eval_vit.py", ["--arm", arm, "--seed", str(seed)])
            xai = f"/vol/artifacts/logs/xai_summary_{arm}_seed{seed}_X1.json"
            if not os.path.exists(xai):
                _run("08_xai.py", ["--arm", arm, "--seed", str(seed)])
            done.append(f"{arm}/{seed}")
    return done


@app.function(image=image, gpu="A100-40GB", timeout=86400, volumes={"/vol": vol})
def arm_c_pipeline():
    """Train, evaluate and explain all three Arm C seeds in one container.

    One container avoids paying the image-build and cold-start cost three times,
    which matters because the remaining budget is small.
    """
    # The Arm C shards committed after this container's volume snapshot was
    # taken, so reload before reading their outputs.
    vol.reload()
    done = []
    for seed in (42, 123, 2026):
        _run("06_train_vit.py", ["--arm", "C", "--seed", str(seed)])
        _run("07_eval_vit.py", ["--arm", "C", "--seed", str(seed)])
        _run("08_xai.py", ["--arm", "C", "--seed", str(seed)])
        done.append(seed)
    return done


@app.function(image=image, gpu="A100-40GB", timeout=7200, volumes={"/vol": vol})
def reproduction_check():
    """G13: re-run one recorded ViT arm/seed from scratch and compare.

    Runs seed 42 of arm A into a fresh directory so the recorded result is not
    disturbed, then writes a reproduction log comparing the two.
    """
    return _run("11_reproduction_check.py")


# Separate image: rclone must be installed before the local-file mounts, which
# Modal requires to come last in the build chain.
drive_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "unzip")
    .run_commands(
        "curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip "
        "-o /tmp/rclone.zip",
        "unzip -q /tmp/rclone.zip -d /tmp",
        "install -m 755 /tmp/rclone-*/rclone /usr/local/bin/rclone",
        "rclone version",
    )
    .pip_install("modal==1.6.3", "numpy==1.26.4", "pandas==2.2.3")
    .env(
        {
            "DATA_ROOT": "/vol/data",
            "ARTIFACT_ROOT": "/vol/artifacts",
            "PYTHONPATH": f"{REMOTE_ROOT}/src",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir("src", f"{REMOTE_ROOT}/src")
    .add_local_dir("scripts", f"{REMOTE_ROOT}/scripts")
)


# Modal resolves a named secret at app-load time, so referencing one that does
# not exist aborts every function in this file, not just the Drive upload.
# The upload is optional, so it is declared only when explicitly enabled:
#   ENABLE_GDRIVE_UPLOAD=1 modal run modal_app.py::upload_to_drive
_GDRIVE_SECRET = (
    modal.Secret.from_name("gdrive-sa")
    if os.environ.get("ENABLE_GDRIVE_UPLOAD") == "1"
    else None
)


@app.function(
    image=drive_image,
    cpu=4,
    memory=8192,
    timeout=21600,
    volumes={"/vol": vol},
    # Create this once with the service-account JSON:
    #   modal secret create gdrive-sa GDRIVE_SA_JSON="$(cat sa.json)"
    secrets=[_GDRIVE_SECRET] if _GDRIVE_SECRET is not None else [],
)
def upload_to_drive(tier: str = "results", dest: str = "dmahr-vit", dry_run: bool = False):
    """Stream volume artifacts to Google Drive (cloud to cloud)."""
    if _GDRIVE_SECRET is None:
        raise RuntimeError(
            "the gdrive-sa secret is not configured; create it with "
            'modal secret create gdrive-sa GDRIVE_SA_JSON="$(cat sa.json)"')
    args = ["--tier", tier, "--dest", dest]
    if dry_run:
        args.append("--dry-run")
    return _run("12_upload_to_drive.py", args)


@app.local_entrypoint()
def drive_smoke():
    """Verify the Drive remote works before moving 40 GB."""
    print(upload_to_drive.remote("results", "dmahr-vit-smoke", True), flush=True)

