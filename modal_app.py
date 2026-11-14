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
        "python-docx==1.1.2",
        "pytest==8.3.4",
        "pymupdf==1.24.14",
        "lpips==0.1.4",
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
    .add_local_dir("reports", f"{REMOTE_ROOT}/reports")
    .add_local_dir("docs", f"{REMOTE_ROOT}/docs")
    .add_local_dir("tests", f"{REMOTE_ROOT}/tests")
    .add_local_file("README.md", f"{REMOTE_ROOT}/README.md")
    .add_local_file("requirements.txt", f"{REMOTE_ROOT}/requirements.txt")
    .add_local_file("requirements.lock", f"{REMOTE_ROOT}/requirements.lock")
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


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": vol})
def dmahr_train_continue(iterations: int = 100000):
    """Resume the fixed 20k run toward `iterations`, after preserving the 20k files.

    The continuation overwrites best_gen.pth, train_state.pt and the run logs, so
    the 20,000-iteration originals are copied aside first, once.
    """
    import shutil  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    root = Path(os.environ.get("ARTIFACT_ROOT", "/vol/artifacts"))
    src, dst = root / "checkpoints" / "dmahr", root / "checkpoints" / "dmahr_fixed20k"
    if not dst.exists():
        dst.mkdir(parents=True)
        for name in ("best_gen.pth", "final_gen.pth", "train_state.pt",
                     "train_log.json", "training_manifest.json"):
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        old_log = root / "logs" / "dmahr_train.json"
        if old_log.exists():
            shutil.copy2(old_log, dst / "dmahr_train_20k.json")
        vol.commit()
        print("preserved 20k files in", dst, flush=True)
    return _run("04_dmahr_train.py", ["--iterations", str(iterations)])


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


@app.function(image=image, gpu="A100-80GB", timeout=7200, volumes={"/vol": vol})
def vit_eval_repro(arm: str = "A", seed: int = 42):
    """Re-evaluate into a scratch artifact root, then diff against the recorded run."""
    repro_root = "/vol/artifacts/repro_check"
    _run("07_eval_vit.py", ["--arm", arm, "--seed", str(seed),
                            "--out-dir", os.path.join(repro_root, "predictions")])
    return _run("21_verify_reconstruction.py", ["--arm", arm, "--seed", str(seed)])


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def transform_sweep():
    return _run("23_transform_sweep.py")


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def resize_sweep():
    return _run("24_resize_sweep.py")


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def oneshot_diagnostic(pairs: int = 50):
    # Small batches at 128 px fragment the allocator; expandable segments keeps
    # the sweep inside the 80 GB budget.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return _run("25_oneshot_diagnostic.py", ["--pairs", str(pairs)])


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def convention_check(pairs: int = 50):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return _run("26_convention_check.py", ["--pairs", str(pairs)])


@app.function(image=image, gpu="A100-80GB", timeout=7200, volumes={"/vol": vol})
def warmstart_check(iterations: int = 1000):
    """Warm-start the released weights with the fixed trainer, then re-score."""
    released = "/vol/data/raw/dmahr_weights/I725000_E1083_gen.pth"
    _run("04_dmahr_train.py", ["--warm-start", released,
                               "--iterations", str(iterations),
                               "--out-name", "dmahr_warmstart_check"])
    return _run("27_warmstart_check.py")


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def respacing_check():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return _run("28_respacing_check.py")


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def paper_t10_check():
    return _run("57_paper_t10_check.py")


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def trainer_gates(gate: str = "1"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return _run("29_trainer_gates.py", ["--gate", gate])


@app.function(image=image, cpu=8, memory=16384, timeout=7200, volumes={"/vol": vol})
def harvest_hair(workers: int = 8, limit: int = 0):
    return _run("40_harvest_hair.py", ["--workers", str(workers)]
                + (["--limit", str(limit)] if limit else []))


@app.function(image=image, cpu=4, memory=8192, timeout=1800, volumes={"/vol": vol})
def harvest_review():
    return _run("43_harvest_review.py")


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": vol})
def dmahr_train_v2(iterations: int = 20000, out_name: str = "dmahr_v2", workers: int = 8,
                   warm_start: str = "/vol/artifacts/checkpoints/dmahr/gen_iter050000.pth"):
    """Fine-tune on real-hair composites. Never writes into dmahr/ or dmahr_fixed20k/."""
    assert out_name not in ("dmahr", "dmahr_fixed20k"), "refusing to overwrite a recorded run"
    return _run("04_dmahr_train.py", ["--composite", "--warm-start", warm_start,
                                      "--out-name", out_name, "--iterations", str(iterations),
                                      "--num-workers", str(workers)])


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def real_hair_diffusion(ckpt: str, name: str, ids: str, t0s: str = "600,1000,1500",
                        steps: int = 50):
    return _run("39_real_hair_diffusion.py",
                ["--ckpt", ckpt, "--name", name, "--ids", ids, "--t0s", t0s,
                 "--steps", str(steps)])


@app.function(image=image, gpu="A100-80GB", timeout=7200, volumes={"/vol": vol})
def partial_noise_eval(ckpt: str, name: str, tag: str, t0s: str = "600,1000,1500,1999",
                       steps: int = 50, seeds: int = 4, pairs: int = 24,
                       commit: str = "unknown", offset: int = 0):
    return _run("38_partial_noise_eval.py",
                ["--ckpt", ckpt, "--name", name, "--tag", tag, "--t0s", t0s,
                 "--steps", str(steps), "--seeds", str(seeds), "--pairs", str(pairs),
                 "--offset", str(offset), "--commit", commit])


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def composite_eval(ckpts: str, tag: str, t0s: str = "600,1000", items: int = 48,
                   seeds: int = 2, commit: str = "unknown"):
    args = ["--tag", tag, "--t0s", t0s, "--items", str(items), "--seeds", str(seeds),
            "--commit", commit]
    for spec in ckpts.split(","):
        args += ["--ckpt", spec]
    return _run("45_composite_eval.py", args)


CV2_CKPT = "/vol/artifacts/checkpoints/dmahr_v2/final_gen.pth"


@app.function(image=image, cpu=8, memory=16384, timeout=3600, volumes={"/vol": vol})
def fill_cv2_passthrough():
    return _run("05b_fill_c_passthrough.py", ["--arm-name", "Cv2"])


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": vol})
def cv2_sample(shard: int, n_shards: int, pairs_split: str = "", start_t: int = 600,
               steps: int = 50, checkpoint: str = CV2_CKPT):
    """Stage C sampler: strided DDIM from the noised input, fp32. Writes the Cv2 arm only."""
    args = ["--shard", str(shard), "--n-shards", str(n_shards), "--arm-name", "Cv2",
            "--respaced", "--steps", str(steps), "--start-t", str(start_t),
            "--checkpoint", checkpoint]
    if pairs_split:
        args += ["--pairs-split", pairs_split]
    return _run("05_dmahr_sample.py", args)


@app.function(image=image, cpu=1, memory=2048, timeout=86400, volumes={"/vol": vol})
def cv2_sample_driver(n_shards: int = 4, pairs_split: str = ""):
    """Spawn the Cv2 shards (real hair-present images, plus a synthetic pair split if named)."""
    handles = [cv2_sample.spawn(i, n_shards, "") for i in range(n_shards)]
    if pairs_split:
        handles.append(cv2_sample.spawn(0, 1, pairs_split))
    results = [h.get() for h in handles]
    print("all shards finished:", results, flush=True)
    return results


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": vol})
def vit_arm_seed(arm: str, seed: int, xai: bool = True):
    """Train, evaluate on the locked test split, and (optionally) explain one arm/seed."""
    vol.reload()
    _run("06_train_vit.py", ["--arm", arm, "--seed", str(seed)])
    _run("07_eval_vit.py", ["--arm", arm, "--seed", str(seed)])
    if xai:
        _run("08_xai.py", ["--arm", arm, "--seed", str(seed)])
    return [arm, seed]


@app.function(image=image, cpu=1, memory=2048, timeout=86400, volumes={"/vol": vol})
def vit_arm_driver(arm: str, seeds: str = "42,123,2026"):
    handles = [vit_arm_seed.spawn(arm, int(s)) for s in seeds.split(",")]
    results = [h.get() for h in handles]
    print("all seeds finished:", results, flush=True)
    return results


@app.function(image=image, cpu=1, memory=2048, timeout=86400, volumes={"/vol": vol})
def vit_extra_seeds_driver(arms: str = "A,Astar,B,Cv2", seeds: str = "7,11,17,23,31"):
    """Stage E: extra seeds for the pooled analysis, evaluated once each, no explanation maps."""
    handles = [vit_arm_seed.spawn(a, int(s), False) for a in arms.split(",")
               for s in seeds.split(",")]
    results = [h.get() for h in handles]
    print("all runs finished:", results, flush=True)
    return results


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": vol})
def image_quality_arm(split: str = "test", c_arm: str = "Cv2"):
    return _run("07b_image_quality.py", ["--split", split, "--c-arm", c_arm])


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": vol})
def stats_arm(c_arm: str = "Cv2"):
    return _run("09_stats.py", ["--c-arm", c_arm])


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": vol})
def pasteback_tune(variant: str = "coarse"):
    """Stage D parameter search on the validation synthetic pairs only."""
    return _run("48_pasteback.py", ["tune", "--variant", variant])


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": vol})
def pasteback_real(ids: str, name: str, tau: float, dilate: int, refine: bool = True,
                    thin_t: float = 0.0, grow: int = 2):
    args = ["real", "--ids", ids, "--name", name, "--tau", str(tau), "--dilate", str(dilate),
            "--thin-T", str(thin_t), "--grow", str(grow)]
    return _run("48_pasteback.py", args + (["--refine"] if refine else []))


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": vol})
def pasteback_build(target: str, tau: float, dilate: int, refine: bool = True,
                    thin_t: float = 0.0, grow: int = 2):
    args = ["build", "--target", target, "--tau", str(tau), "--dilate", str(dilate),
            "--thin-T", str(thin_t), "--grow", str(grow)]
    return _run("48_pasteback.py", args + (["--refine"] if refine else []))


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def composite_arms(items: int = 96, commit: str = "unknown"):
    """Stage E V1: DullRazor and Cv2 on validation composites with lesion-damage metrics."""
    return _run("50_composite_arms.py", ["--items", str(items), "--commit", commit])


@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def condition_usage(pairs: int = 48, commit: str = "unknown"):
    """Stage E V4: epsilon error with the condition kept, zeroed and shuffled."""
    return _run("51_condition_usage.py", ["--pairs", str(pairs), "--commit", commit])


@app.function(image=image, cpu=8, memory=16384, timeout=3600, volumes={"/vol": vol})
def residual_hair_split(ids_gate: str, ids_dense: str, n: int = 300, commit: str = "unknown"):
    """Stage E V2 and V3: lesion change on real validation photos; residual hair by split."""
    return _run("52_residual_hair_split.py", ["--ids-gate", ids_gate, "--ids-dense", ids_dense,
                                              "--n", str(n), "--commit", commit])


@app.function(image=image, cpu=4, memory=8192, timeout=1800, volumes={"/vol": vol})
def review_kit_build():
    """Stage E: blinded human-review kit on validation photos."""
    return _run("54_review_kit.py", ["build"])


@app.function(image=image, gpu="A100-80GB", timeout=7200, volumes={"/vol": vol})
def gate3_eval(pairs: int = 182, checkpoint: str = "", tag: str = "gate3",
               precision: str = "bf16", replicate_first: int = 0,
               no_tf32: bool = False):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = ["--pairs", str(pairs), "--tag", tag, "--precision", precision]
    if no_tf32:
        args += ["--no-tf32"]
    if replicate_first:
        args += ["--replicate-first", str(replicate_first)]
    if checkpoint:
        args += ["--checkpoint", checkpoint]
    return _run("30_gate3_eval.py", args)


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": vol})
def image_quality(split: str = "test"):
    return _run("07b_image_quality.py", ["--split", split])


@app.function(image=image, cpu=4, memory=8192, timeout=3600, volumes={"/vol": vol})
def xai_registration_check():
    """Mask-only geometry check on the validation split (no model, no test data)."""
    return _run("56_xai_registration_check.py")


def _clean_commit() -> str:
    """Current commit; refuses to run with uncommitted changes to code or configs."""
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", "scripts", "src",
         "configs", "tests", "modal_app.py"], capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit("commit scripts before running:\n" + dirty)
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/vol": vol},
              retries=modal.Retries(max_retries=2, initial_delay=5.0))
def xai_v2(arm: str, seed: int, split: str, commit: str):
    """Corrected explanation analysis for one checkpoint (see scripts/58_xai_corrected.py)."""
    return _run("58_xai_corrected.py", ["--arm", arm, "--seed", str(seed), "--split", split,
                                        "--commit", commit])


@app.local_entrypoint()
def xai_v2_all(split: str = "test"):
    commit = _clean_commit()
    if split == "val":
        runs = [(a, s) for a in ("A", "Astar", "B", "Cv2") for s in (42, 123, 2026)]
    else:
        eight = (42, 123, 2026, 7, 11, 17, 23, 31)
        runs = ([(a, s) for a in ("A", "Astar", "B", "Cv2") for s in eight]
                + [(a, s) for a in ("C", "Creleased") for s in (42, 123, 2026)])
    print(f"commit {commit}: {len(runs)} runs on split={split}", flush=True)
    for res in xai_v2.starmap([(a, s, split, commit) for a, s in runs]):
        print("done", res, flush=True)


@app.function(image=image, gpu="A100-40GB", timeout=14400, volumes={"/vol": vol})
def dermahair_repro(commit: str):
    """Released checkpoint on DERMAHAIR (scripts/62_dermahair_repro.py)."""
    return _run("62_dermahair_repro.py", ["--commit", commit])


@app.local_entrypoint()
def dermahair_run():
    print(dermahair_repro.remote(_clean_commit()))


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={"/vol": vol})
def contract_tests():
    """Trainer/sampler contract tests, including the two that need the real pair manifest."""
    diag = "/tmp/diag"
    os.makedirs(diag, exist_ok=True)
    for link, target in (("man", "/vol/data/manifests"), ("data", "/vol/data")):
        if not os.path.exists(f"{diag}/{link}"):
            os.symlink(target, f"{diag}/{link}")
    os.environ["DMAHR_DIAG_ROOT"] = diag
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
           "tests/test_trainer_sampler_contract.py", "tests/test_noise_level_convention.py",
           "tests/test_trainer_resume.py", "tests/test_partial_noise_sampler.py"]
    proc = subprocess.run(cmd, cwd=REMOTE_ROOT, capture_output=True, text=True)
    print(proc.stdout[-3000:], proc.stderr[-1000:], flush=True)
    if proc.returncode != 0:
        raise RuntimeError("contract tests failed")
    return proc.stdout.strip().splitlines()[-1]


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": vol},
              retries=modal.Retries(max_retries=3, initial_delay=10.0))
def lc_train(out_name: str, iterations: int, grad_clip: float, commit: str):
    """Learning curve: continue the repaired-trainer run from its 56,000-iteration state into a
    new directory (nothing recorded is overwritten). Resumable: reruns continue from train_state.pt."""
    import shutil  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import torch  # noqa: PLC0415

    root = Path(os.environ.get("ARTIFACT_ROOT", "/vol/artifacts")) / "checkpoints"
    dst = root / out_name
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "train_state.pt").exists():
        state = torch.load(root / "dmahr" / "train_state.pt", map_location="cpu", weights_only=False)
        print("start state: iteration", state["iteration"], "commit", commit, flush=True)
        if int(state["iteration"]) != 56000:
            raise RuntimeError(f"unexpected start iteration {state['iteration']}")
        shutil.copy2(root / "dmahr" / "train_state.pt", dst / "train_state.pt")
        vol.commit()
    return _run("04_dmahr_train.py", ["--out-name", out_name, "--iterations", str(iterations),
                                      "--grad-clip", str(grad_clip)])


@app.function(image=image, gpu="A100-40GB", timeout=14400, volumes={"/vol": vol})
def lc_eval(ckpts: str, commit: str):
    return _run("63_learning_curve_eval.py", ["--ckpts", ckpts, "--commit", commit])


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": vol})
def export_checkpoints(commit: str):
    """Weights-only ViT exports and diffusion weights with a SHA-256 manifest, on the volume."""
    return _run("71_export_checkpoints.py", ["--commit", commit])


@app.local_entrypoint()
def lc_start(iterations: int = 100000):
    """Clipped continuation and the no-clip comparison, same start state, run side by side."""
    commit = _clean_commit()
    calls = [lc_train.spawn("dmahr_lc", iterations, 1.0, commit),
             lc_train.spawn("dmahr_lc_noclip", iterations, 0.0, commit)]
    for c in calls:
        print("finished", c.get(), flush=True)


@app.function(image=image, gpu="A100-80GB", timeout=14400, volumes={"/vol": vol})
def xai_run(arm: str, seed: int):
    return _run("08_xai.py", ["--arm", arm, "--seed", str(seed)])


@app.function(image=image, gpu="A100-80GB", timeout=14400, volumes={"/vol": vol})
def xai_all(arms: str = "A,Astar,B,C"):
    """Regenerate XAI outputs for every seed of the given arms."""
    done = []
    for arm in [a for a in arms.split(",") if a]:
        for seed in (42, 123, 2026):
            _run("08_xai.py", ["--arm", arm, "--seed", str(seed)])
            done.append(f"{arm}/{seed}")
    return done


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": vol})
def stats_run():
    return _run("09_stats.py")


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": vol})
def exploratory_stats():
    return _run("22_exploratory_stats.py")


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes={"/vol": vol})
def acceptance():
    return _run("check_acceptance.py")


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes={"/vol": vol})
def deliverable_gates():
    return _run("20_deliverable_gates.py")


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes={"/vol": vol})
def results_report():
    return _run("14_results_report.py")


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes={"/vol": vol})
def finalize():
    return _run("10_finalize.py")


@app.function(image=image, gpu="A100-80GB", timeout=86400, volumes={"/vol": vol})
def eval_and_xai_all(arms: str = "A,B,Astar,C", force: bool = False):
    """Locked test evaluation then XAI for every checkpoint that exists."""
    done = []
    for arm in [a for a in arms.split(",") if a]:
        for seed in (42, 123, 2026):
            ckpt = f"/vol/artifacts/checkpoints/vit/{arm}_seed{seed}/best.pt"
            if not os.path.exists(ckpt):
                print("skip (no checkpoint):", ckpt, flush=True)
                continue
            pred = f"/vol/artifacts/predictions/metrics_{arm}_seed{seed}.json"
            if force or not os.path.exists(pred):
                _run("07_eval_vit.py", ["--arm", arm, "--seed", str(seed)])
            xai = f"/vol/artifacts/logs/xai_summary_{arm}_seed{seed}_X1.json"
            if force or not os.path.exists(xai):
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

