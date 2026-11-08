"""DEPTHOPEN — driving the makers' own trainer for the five arms.

WHAT IS THE MAKERS' AND WHAT IS OURS. The training code, the LoRA recipe, the checkpoint layout and the
HF export are theirs, at POSTCHECK's pinned commit, unmodified: the arms differ only in the four flags
SPEC §2 names, which their launcher already exposes. Ours is the launch wrapper, and it does exactly two
things their launcher cannot: it registers a mixture pointing at our banked subset, and it instruments the
optimizer so G-grad is a measurement of realized gradient rather than a claim about intent.

WHY A WRAPPER AND NOT A PATCH. `MOLMOACT2_LEROBOT_MIXTURES` is read inside
`get_lerobot_training_data_plan`, so a builder registered at runtime is picked up; the tag is `libero`,
the same tag the built-in `libero_goal` mixture uses, so the setup/control text and the action
normalization metadata are the released ones. Nothing in their tree is edited, which keeps the arms
comparable to the recipe the README publishes.

WHY THE INSTALL IS AN IMPORT PROBE. The AMI carries torch 2.12+cu130; `pip install -e experiments`
without `--no-deps` would resolve `torch` again and can replace a working CUDA build. So the package is
installed with no dependencies and then the import chain is walked, installing exactly what it asks for,
and the list of what was missing is banked. A guess about the dependency set would be a silent difference
between our arms and theirs.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

BUCKET = "pg-lineage-769063704696"
REGION = "us-east-1"
MO2_SHA = "66b87e64efd99dfd103241418113955cf64dfa9c"
S3_SRC = f"s3://{BUCKET}/depthopen/src/molmoact2_at_{MO2_SHA[:12]}.tar.gz"
SUBSET_REPO = "allenai/MolmoAct2-LIBERO-depthopen"
S3_SUBSET = f"s3://{BUCKET}/depthopen/subset/v1/{SUBSET_REPO}"
S3_DEPTH = f"s3://{BUCKET}/depthopen/subset_depth/v1/{SUBSET_REPO}"
BASE_CKPT = "allenai/MolmoAct2-Think"
MIXTURE = "depthopen"

# SPEC §2. The four flags are the only difference between arms; everything else is shared below.
ARMS: Dict[str, Dict[str, object]] = {
    "A0": {"gate": True, "per_layer": True, "init_bias": -4.0, "noise": 0.10,
           "what": "the released condition reproduced at our scale: the control"},
    "A1": {"gate": True, "per_layer": True, "init_bias": 0.0, "noise": 0.10,
           "what": "opened gate, noise as shipped"},
    "A2": {"gate": True, "per_layer": True, "init_bias": -4.0, "noise": 0.00,
           "what": "shipped gate, no depth-code noise"},
    "A3": {"gate": True, "per_layer": True, "init_bias": 0.0, "noise": 0.00,
           "what": "opened gate, no noise: the primary treatment"},
    "A4": {"gate": False, "per_layer": False, "init_bias": -4.0, "noise": 0.10,
           "what": "no gate at all: the base model's own architecture, prefix at full weight"},
}

# module -> pip package, for the names that differ. Anything not here is installed under its own name.
PACKAGE_FOR_MODULE = {
    # the import name and the distribution name differ, and for `beaker` a DIFFERENT project squats the
    # import name on PyPI — the repo means Ai2's client, pinned in its own `train` extra
    "beaker": "beaker-py==1.36.2",
    "gantry": "beaker-gantry==1.17.0",
    "cv2": "opencv-python-headless",
    "google.protobuf": "protobuf",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "Levenshtein": "python-Levenshtein",
    "decord": "decord2",
    "jsonlines": "jsonlines",
    "cached_property": "cached_property",
    "einops_exts": "einops-exts",
    "json_numpy": "json_numpy",
    "ffmpeg": "ffmpeg-python",
    "torchcodec": "torchcodec",
}
# torch and its companions are the AMI's; a pip install of these would swap a working CUDA build
NEVER_INSTALL = {"torch", "torchvision", "torchaudio", "triton", "nvidia"}


def s3(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["aws", "s3", *args, "--region", REGION], check=check)


def fetch_source(work: pathlib.Path) -> pathlib.Path:
    """The makers' tree at the pinned commit, from the recon job's tarball."""
    work.mkdir(parents=True, exist_ok=True)
    tar = work / "molmoact2.tar.gz"
    if not tar.exists():
        s3("cp", S3_SRC, str(tar))
    root = work / "molmoact2"
    if not (root / "experiments").is_dir():
        subprocess.run(["tar", "xzf", str(tar), "-C", str(work)], check=True)
    if not (root / "experiments" / "launch_scripts" / "train_lerobot.py").is_file():
        raise FileNotFoundError(f"unpacked tree looks wrong: {sorted(p.name for p in root.iterdir())}")
    return root


APT_PACKAGES = "ffmpeg"


def install_system_packages() -> Dict[str, object]:
    """`ffmpeg`, because the trainer's own torchcodec preflight writes a smoke video before step 0.

    `launch_scripts/lerobot_utils/env.py:_run_torchcodec_preflight` calls
    `olmo/torchcodec_utils.py:_create_smoke_video`, which raises "ffmpeg binary was not found on PATH"
    and takes the run with it — 19 seconds in, on a box that had just synced 47 GB of weights. Our subset
    has no video at all (the release stores frames as parquet image bytes), but the preflight is
    unconditional, so the binary has to be there.
    """
    cmd = (f"apt-get -o DPkg::Lock::Timeout=900 update -y && DEBIAN_FRONTEND=noninteractive "
           f"apt-get -o DPkg::Lock::Timeout=900 install -y {APT_PACKAGES}")
    r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"apt failed ({r.returncode}): {r.stdout[-600:]}{r.stderr[-600:]}")
    which = subprocess.run(["bash", "-lc", "command -v ffmpeg && ffmpeg -version | head -1"],
                           capture_output=True, text=True)
    if which.returncode != 0:
        raise RuntimeError(f"ffmpeg still not on PATH after apt: {which.stderr[-300:]}")
    return {"packages": APT_PACKAGES, "ffmpeg": which.stdout.strip().splitlines()[:2]}


def pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)


def declared_dependencies(exp: pathlib.Path) -> Dict[str, object]:
    """The makers' own dependency lists, read from their `pyproject.toml`.

    Discovering the environment one `ModuleNotFoundError` at a time guesses distribution names, and one
    guess was wrong in a way that cost a job: the trainer imports `beaker`, and PyPI's `beaker` is a
    caching library, while the repo means `beaker-py==1.36.2` — declared right there in its `train` extra.
    So the declared lists are installed first and the probe loop only mops up what they miss.
    """
    import tomllib
    payload = tomllib.loads((exp / "pyproject.toml").read_text())
    project = payload.get("project") or {}
    base = list(project.get("dependencies") or [])
    train = list((project.get("optional-dependencies") or {}).get("train") or [])
    def keep(spec: str) -> bool:
        name = re.split(r"[<>=!\[;~ ]", spec.strip(), 1)[0].lower().replace("_", "-")
        return name not in {"torch", "torchvision", "torchaudio", "numpy"}
    wanted = [s for s in base + train if keep(s)]
    dropped = [s for s in base + train if not keep(s)]
    return {"install": wanted, "dropped": dropped}


def install_stack(src: pathlib.Path, *, max_rounds: int = 40) -> Dict[str, object]:
    """Install the makers' declared dependencies, then the two packages, then walk the import chain.

    torch, torchvision and numpy are dropped from the list: the AMI's CUDA build is the one the science
    runs on and a resolver is free to replace it. The install is checked afterwards — if torch's version
    or its CUDA availability moved, the job stops rather than quietly continuing on a different stack.
    """
    exp = src / "experiments"
    apt = install_system_packages()
    declared = declared_dependencies(exp)
    torch_before = _torch_state()
    # PyPI `beaker` squats the import name the trainer wants; remove it if an earlier attempt left it
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "beaker"],
                   capture_output=True, text=True)
    pip(*declared["install"])
    pip("-e", str(exp), "--no-deps")
    pip("-e", str(exp / "lerobot"), "--no-deps")
    torch_after = _torch_state()
    if torch_after != torch_before:
        raise RuntimeError(f"the dependency install moved torch: {torch_before} -> {torch_after}")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{exp}:{exp / 'lerobot' / 'src'}:" + env.get("PYTHONPATH", "")
    rounds: List[Dict[str, object]] = []
    for _ in range(max_rounds):
        probe = subprocess.run([sys.executable, "-c",
                                "import launch_scripts.train_lerobot as m; print('IMPORT_OK', m.__file__)"],
                               cwd=str(exp), env=env, capture_output=True, text=True)
        if probe.returncode == 0:
            rounds.append({"round": len(rounds), "result": "IMPORT_OK"})
            break
        m = re.search(r"No module named '([^']+)'", probe.stderr)
        if not m:
            raise RuntimeError(f"the trainer import failed for a reason that is not a missing module:\n"
                               f"{probe.stderr[-3000:]}")
        module = m.group(1)
        top = module.split(".")[0]
        if top in NEVER_INSTALL:
            raise RuntimeError(f"the trainer wants to install {module}, which would replace the AMI's "
                               f"CUDA stack; refusing")
        package = PACKAGE_FOR_MODULE.get(module, PACKAGE_FOR_MODULE.get(top, top))
        rounds.append({"round": len(rounds), "missing": module, "installing": package})
        print(f"[install] missing {module} -> pip install {package}", flush=True)
        pip(package)
    else:
        raise RuntimeError(f"still not importable after {max_rounds} rounds: {rounds[-5:]}")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    return {"apt": apt, "declared": declared, "torch_state": torch_after,
            "rounds": rounds, "n_installed": sum(1 for r in rounds if "installing" in r),
            "pythonpath": env["PYTHONPATH"],
            "freeze_lines": len(freeze.stdout.splitlines()),
            "torch": _version("torch"), "transformers": _version("transformers"),
            "peft": _version("peft"), "lerobot": _version("lerobot")}


def _torch_state() -> Dict[str, object]:
    """torch's version and whether CUDA is live — the two things a resolver can silently take away."""
    r = subprocess.run([sys.executable, "-c",
                        "import json, torch; print(json.dumps({'v': torch.__version__, "
                        "'cuda': torch.version.cuda, 'available': bool(torch.cuda.is_available())}))"],
                       capture_output=True, text=True)
    return json.loads(r.stdout.strip()) if r.returncode == 0 else {"error": r.stderr[-200:]}


def _version(module: str) -> Optional[str]:
    r = subprocess.run([sys.executable, "-c",
                        f"import importlib.metadata as m; print(m.version('{module}'))"],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def sync_data(work: pathlib.Path) -> Dict[str, object]:
    """The banked subset and its depth companions, into the two roots the loader reads."""
    data_root = work / "lerobot_data"
    depth_root = work / "lerobot_depth"
    (data_root / SUBSET_REPO).mkdir(parents=True, exist_ok=True)
    (depth_root / SUBSET_REPO).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    s3("sync", S3_SUBSET, str(data_root / SUBSET_REPO))
    s3("sync", S3_DEPTH, str(depth_root / SUBSET_REPO))
    info = json.loads((data_root / SUBSET_REPO / "meta" / "info.json").read_text())
    depth_info = json.loads((depth_root / SUBSET_REPO / "meta" / "info.json").read_text())
    n_data = len(list((data_root / SUBSET_REPO / "data" / "chunk-000").glob("*.parquet")))
    n_depth = len(list((depth_root / SUBSET_REPO / "data" / "chunk-000").glob("*.parquet")))
    if n_data != n_depth:
        raise RuntimeError(f"subset has {n_data} data files, companions have {n_depth}")
    return {"data_root": str(data_root), "depth_root": str(depth_root),
            "n_files": n_data, "total_episodes": info["total_episodes"],
            "total_frames": info["total_frames"],
            "depth_has_vae_block": "depth_vae" in depth_info,
            "seconds": round(time.perf_counter() - t0, 1)}


WRAPPER = '''\
"""Launch wrapper, written by infra/pg/depthopen_train.py. Registers the DEPTHOPEN mixture and
instruments the optimizer, then calls the makers' launcher unchanged."""
import atexit, json, os, pathlib, sys

EXP = pathlib.Path(os.environ["DEPTHOPEN_EXPERIMENTS"])
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "lerobot" / "src"))
PROBE = pathlib.Path(os.environ["DEPTHOPEN_PROBE"])
SUBSET_REPO = os.environ["DEPTHOPEN_SUBSET_REPO"]
MIXTURE = os.environ["DEPTHOPEN_MIXTURE"]

from launch_scripts import data_mixtures as dm


def _builder():
    # the built-in `libero_goal` mixture's own arguments, on our subset: same tag, so the setup and
    # control text and the action normalization metadata are the released ones
    return dm.build_single_lerobot_mixture(
        name=MIXTURE,
        tag="libero",
        repo_ids=[SUBSET_REPO],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=["observation.images.image", "observation.images.wrist_image"],
        normalize_gripper=False,
        setup_type="single franka robotic arm in libero",
        control_mode="delta end-effector pose",
        action_horizon=10,
        n_action_steps=10,
    )


dm.MOLMOACT2_LEROBOT_MIXTURES[MIXTURE] = _builder

# ---- G-grad: the gate's realized gradient, not its intended trainability -----------------------
from olmo.train.optim import OptimizerConfig

RECORD = {"gate_params": [], "group_of": {}, "grad_norms": {}, "n_backwards": 0}
_orig_groups = OptimizerConfig.get_param_groups


def _patched(self, max_grad_norm, max_grad_norm_ratio, model):
    groups = _orig_groups(self, max_grad_norm, max_grad_norm_ratio, model)
    named = {id(p): n for n, p in model.named_parameters()}
    for n, p in model.named_parameters():
        if "depth_gate" in n:
            RECORD["gate_params"].append({"name": n, "shape": list(p.shape),
                                          "requires_grad": bool(p.requires_grad),
                                          "value_mean": float(p.detach().float().mean())})
            if p.requires_grad:
                def _hook(grad, name=n):
                    prev = RECORD["grad_norms"].get(name, 0.0)
                    RECORD["grad_norms"][name] = max(prev, float(grad.detach().float().norm()))
                    RECORD["n_backwards"] += 1
                    return grad
                p.register_hook(_hook)
    for g in groups:
        for p in g["params"]:
            n = named.get(id(p))
            if n and "depth_gate" in n:
                RECORD["group_of"][n] = g.get("group_name")
    RECORD["n_gate_params"] = len(RECORD["gate_params"])
    RECORD["n_gate_trainable"] = sum(1 for e in RECORD["gate_params"] if e["requires_grad"])
    _dump()
    return groups


OptimizerConfig.get_param_groups = _patched


def _dump():
    import torch
    RECORD["cuda_max_memory_allocated_gb"] = (
        round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None)
    RECORD["cuda_max_memory_reserved_gb"] = (
        round(torch.cuda.max_memory_reserved() / 1e9, 2) if torch.cuda.is_available() else None)
    PROBE.write_text(json.dumps(RECORD, indent=1))


atexit.register(_dump)

from launch_scripts.train_lerobot import main

if __name__ == "__main__":
    sys.argv = ["train_lerobot.py"] + json.loads(os.environ["DEPTHOPEN_ARGV"])
    main()
'''


def write_wrapper(work: pathlib.Path) -> pathlib.Path:
    path = work / "depthopen_launch.py"
    path.write_text(WRAPPER)
    return path


def train_argv(arm: str, *, steps: int, device_batch: int, global_batch: int,
               save_folder: pathlib.Path, seed: int, save_interval: Optional[int] = None,
               save_checkpoints: bool = True, extra: Sequence[str] = ()) -> List[str]:
    """The makers' LoRA depth-fine-tuning command, with only SPEC §2's four flags varying.

    `save_checkpoints=False` for the batch probes: each save writes a full unsharded merged checkpoint
    (~24 GB) plus the sharded one, and three probe runs would fill the 200 GB root before the real run
    started. `save_num_checkpoints_to_keep=0` is the trainer's own switch for "do not save".
    """
    a = ARMS[arm]
    argv = [
        BASE_CKPT, MIXTURE,
        f"--max_duration={steps}",
        f"--device_batch_size={device_batch}",
        f"--global_batch_size={global_batch}",
        "--num_workers=2", "--pin_memory=false",
        # the console prints metrics every `log_interval` steps and nothing between, so a short run with
        # the default 20 emits no step line at all
        f"--log_interval={max(1, min(10, steps // 4 or 1))}",
        # our subset has NO video: the release stores both camera streams as parquet image bytes, so the
        # frame-loading backend only decides which decoder is preflighted. The default `torchcodec_exact`
        # makes the launcher demand a torchcodec runtime that this AMI has no matching build for, and the
        # run dies before step 0. PyAV is one of the launcher's own choices and a declared dependency.
        "--frame_loading_backend=av",
        f"--save_folder={save_folder}",
        f"--save_interval={save_interval or steps}",
        f"--save_num_checkpoints_to_keep={1 if save_checkpoints else 0}",
        "--packing=false", "--dynamic_seq_len=true",
        "--ft_vlm=true", "--ft_action_expert=true", "--ft_embedding=lm_head",
        "--lora_enable=true", "--lora_rank=64",
        "--llm_learning_rate=1e-5", "--vit_learning_rate=5e-6",
        "--connector_learning_rate=5e-6", "--action_expert_learning_rate=5e-5",
        # the README's depth fine-tuning block
        "--enable_depth_reasoning=true", "--num_depth_tokens=128",
        "--num_depth_tokens_per_image=100",
        "--style_robot_depth=0.0", "--style_robot_depth_action=1.0",
        # SPEC amendment A5. The launcher's `--style_robot_action` defaults to 1.0, so the sampler mixes
        # action-only samples with depth-and-action samples, and the collator raises "Examples in batch
        # are missing key 'depth_updated_mask'" the moment a batch contains one of each — measured at
        # device_batch_size 2. Every sample here carries the depth prefix, which is also what a study of
        # that channel wants. Applied identically to all five arms.
        "--style_robot_action=0.0",
        # SPEC §2: the four flags that make an arm
        f"--action_expert_depth_gate={'true' if a['gate'] else 'false'}",
        f"--action_expert_depth_gate_per_layer={'true' if a['per_layer'] else 'false'}",
        f"--action_expert_depth_gate_init_bias={a['init_bias']}",
        f"--depth_code_input_noise_rate={a['noise']}",
        f"--seed={seed}",
        "--wandb=null",
    ]
    return argv + list(extra)


STEP_RE = re.compile(r"\[step=(\d+)/\d+")
LOSS_RE = re.compile(r"train/CE loss=([0-9.]+)")
ANY_LOSS_RE = re.compile(r"loss=([0-9.eE+-]+)")
# the trainer's own timestamps, which exist whatever its metric format is
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+\t")
START_RE = re.compile(r"Starting training\.\.\.")
END_RE = re.compile(r"Training complete")
TRAINABLE_RE = re.compile(r"Number of trainable parameters: ([\d,]+)")
# strict: the loose version backtracked and reported "2" from "... after FSDP: 23622"
PEAK_RE = re.compile(r"Peak GPU Memory \(MB\)(?: after FSDP)?\s*[=:]\s*([\d,]+)")


def timings_from_log(text: str, steps: int) -> Dict[str, object]:
    """Seconds per step from the trainer's own log timestamps.

    The console metric lines only appear every `log_interval` steps, so a short probe run emits none and a
    regex over them measures nothing — the first smoke banked `seconds_per_step: null` for exactly that
    reason. The wall time between "Starting training..." and "Training complete" is there regardless.
    """
    import datetime as dt
    start = end = None
    trainable = peak = None
    for line in text.splitlines():
        m = TS_RE.match(line)
        stamp = (dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None)
        if stamp and START_RE.search(line) and start is None:
            start = stamp
        if stamp and END_RE.search(line):
            end = stamp
        tm = TRAINABLE_RE.search(line)
        if tm:
            trainable = int(tm.group(1).replace(",", ""))
        pm = PEAK_RE.search(line)
        if pm:
            peak = max(peak or 0, int(pm.group(1).replace(",", "")))
    out: Dict[str, object] = {"trainable_parameters": trainable, "peak_gpu_mb_reported": peak}
    if start and end and steps > 0:
        seconds = (end - start).total_seconds()
        out.update({"train_wall_seconds": seconds,
                    "seconds_per_step_from_timestamps": round(seconds / steps, 3)})
    return out


def run_training(src: pathlib.Path, work: pathlib.Path, argv: List[str], *,
                 data: Dict[str, object], log_path: pathlib.Path,
                 artifact_dir: Optional[pathlib.Path] = None,
                 on_line=None) -> Dict[str, object]:
    """torchrun on one GPU, with the log streamed to disk and the step/loss lines parsed as they pass."""
    exp = src / "experiments"
    wrapper = write_wrapper(work)
    probe = work / "grad_probe.json"
    # every path the makers' own Setup section exports. `MOLMO_DATA_DIR` is not optional even for a
    # LeRobot-only mixture: `olmo/data/get_dataset.py:get_dataset_by_name` reads it on its FIRST line,
    # before any branch, so a run without it dies inside `build_train_dataloader`. Nothing of ours lives
    # under these two roots — they exist so the makers' code finds the shape it expects.
    molmo_data = pathlib.Path(work) / "molmo_data"
    spatial_data = pathlib.Path(work) / "spatial_data"
    for d in (molmo_data, spatial_data):
        d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": f"{exp}:{exp / 'lerobot' / 'src'}:" + env.get("PYTHONPATH", ""),
        "DEPTHOPEN_EXPERIMENTS": str(exp),
        "DEPTHOPEN_PROBE": str(probe),
        "DEPTHOPEN_SUBSET_REPO": SUBSET_REPO,
        "DEPTHOPEN_MIXTURE": MIXTURE,
        "DEPTHOPEN_ARGV": json.dumps(argv),
        "LEROBOT_DATA_ROOT": str(data["data_root"]),
        "LEROBOT_DEPTH_DATA_ROOT": str(data["depth_root"]),
        "LEROBOT_VIDEO_BACKEND": "pyav",
        "MOLMO_DATA_DIR": str(molmo_data),
        "SPATIAL_DATA_HOME": str(spatial_data),
        "HF_HUB_OFFLINE": "0",
        "TOKENIZERS_PARALLELISM": "false",
    })
    env.pop("TRANSFORMERS_OFFLINE", None)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
           str(wrapper)]
    t0 = time.perf_counter()
    steps: List[Dict[str, object]] = []
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=str(exp), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            m = STEP_RE.search(line)
            if m:
                loss = LOSS_RE.search(line) or ANY_LOSS_RE.search(line)
                steps.append({"step": int(m.group(1)),
                              "loss": float(loss.group(1)) if loss else None,
                              "t": round(time.perf_counter() - t0, 1)})
                if on_line:
                    on_line(steps[-1], line)
        rc = proc.wait()
    seconds = time.perf_counter() - t0
    out = {"returncode": rc, "seconds": round(seconds, 1), "n_step_lines": len(steps),
           "steps": steps[-200:], "log": str(log_path),
           "argv": argv}
    if probe.is_file():
        out["grad_probe"] = json.loads(probe.read_text())
    if artifact_dir is not None:
        # the child's traceback lives in this log and nowhere else; run_job.sh syncs out/ every 30 s, so
        # copying it there is the difference between a diagnosable failure and another instance
        dest = pathlib.Path(artifact_dir) / f"trainlog_{log_path.stem}.log"
        text = log_path.read_text()
        dest.write_text(text if len(text) < 4_000_000 else text[:2_000_000] + "\n...\n" + text[-2_000_000:])
        out["log_artifact"] = str(dest)
    if rc != 0:
        tail = log_path.read_text()[-12000:]
        raise RuntimeError(f"training exited {rc}. log tail:\n{tail}")
    out.update(timings_from_log(log_path.read_text(), _steps_requested(argv)))
    if out.get("seconds_per_step") is None and out.get("seconds_per_step_from_timestamps"):
        out["seconds_per_step"] = out["seconds_per_step_from_timestamps"]
    if steps:
        first, last = steps[0], steps[-1]
        out["seconds_per_step"] = round((last["t"] - first["t"]) / max(1, last["step"] - first["step"]), 2)
        losses = [s["loss"] for s in steps if s["loss"] is not None]
        if losses:
            half = max(1, len(losses) // 2)
            out["loss_first_half_mean"] = round(sum(losses[:half]) / half, 4)
            out["loss_second_half_mean"] = round(sum(losses[-half:]) / half, 4)
            out["loss_decreasing"] = bool(out["loss_second_half_mean"] < out["loss_first_half_mean"])
    return out


def weights_moved(base_snapshot: pathlib.Path, arm_dir: pathlib.Path, *,
                  n_tensors: int = 12) -> Dict[str, object]:
    """Did training change the model? Read from the two checkpoints' safetensors, no forward pass.

    This replaces G-smoke's "loss decreasing" clause. The makers' trainer sends per-step loss to W&B and
    this study runs with `--wandb=null`, so the console carries step headers and no loss at all — measured.
    A weights diff is a realized effect rather than a reported one, which is the house preference anyway,
    and "is the fine-tune broken" is carried by G-known on the closed loop.
    """
    from safetensors import safe_open
    import numpy as np

    def index(root: pathlib.Path) -> Dict[str, str]:
        idx = json.loads((root / "model.safetensors.index.json").read_text())
        return idx["weight_map"]

    base_map, arm_map = index(pathlib.Path(base_snapshot)), index(pathlib.Path(arm_dir))
    shared = sorted(set(base_map) & set(arm_map))
    gate = [k for k in shared if "action_expert_depth_gate" in k]
    expert = [k for k in shared if "action_expert" in k and "depth_gate" not in k]
    llm = [k for k in shared if ".blocks." in k or "transformer" in k][:n_tensors]
    picks = gate[:6] + expert[:n_tensors] + llm[:n_tensors]
    rows = []
    for key in picks:
        vals = {}
        for label, root, m in (("base", pathlib.Path(base_snapshot), base_map),
                               ("arm", pathlib.Path(arm_dir), arm_map)):
            with safe_open(str(root / m[key]), framework="np") as f:
                vals[label] = np.asarray(f.get_tensor(key), dtype=np.float64)
        if vals["base"].shape != vals["arm"].shape:
            rows.append({"tensor": key, "shape_changed": [list(vals["base"].shape),
                                                          list(vals["arm"].shape)]})
            continue
        delta = float(np.abs(vals["arm"] - vals["base"]).max())
        rows.append({"tensor": key, "max_abs_delta": delta,
                     "base_l2": float(np.linalg.norm(vals["base"])),
                     "arm_l2": float(np.linalg.norm(vals["arm"]))})
    moved = [r for r in rows if float(r.get("max_abs_delta") or 0.0) > 0.0]
    gate_rows = [r for r in rows if "depth_gate" in r["tensor"]]
    return {"n_tensors_compared": len(rows), "n_moved": len(moved),
            "max_abs_delta_overall": max((float(r.get("max_abs_delta") or 0.0) for r in rows),
                                         default=0.0),
            "gate_tensors_moved": sum(1 for r in gate_rows
                                      if float(r.get("max_abs_delta") or 0.0) > 0.0),
            "gate_max_abs_delta": max((float(r.get("max_abs_delta") or 0.0) for r in gate_rows),
                                      default=0.0),
            "per_tensor": rows,
            "passes": bool(len(moved) >= max(3, len(rows) // 3))}


def _steps_requested(argv: Sequence[str]) -> int:
    for a in argv:
        if str(a).startswith("--max_duration="):
            return int(str(a).split("=", 1)[1])
    return 0


def checkpoint_listing(save_folder: pathlib.Path) -> Dict[str, object]:
    """What the trainer actually wrote. The merged directory is the deployment checkpoint per the
    makers' own layout table, and its presence is what the export step depends on."""
    out = {}
    for d in sorted(p for p in save_folder.glob("*") if p.is_dir()):
        files = sorted(p.name for p in d.rglob("*") if p.is_file())
        out[d.name] = {"n_files": len(files), "bytes": sum(p.stat().st_size for p in d.rglob("*") if p.is_file()),
                       "sample": files[:8]}
    return out


RELEASED_CODE_S3 = f"s3://{BUCKET}/postcheck/code/v1/think_libero"
REMOTE_CODE_FILES = ("configuration_molmoact2.py", "modeling_molmoact2.py", "processing_molmoact2.py",
                     "image_processing_molmoact2.py", "video_processing_molmoact2.py", "inference.py",
                     "chat_template.jinja", "processor_config.json")


def adopt_released_remote_code(hf_dir: pathlib.Path, work: pathlib.Path) -> Dict[str, object]:
    """Replace the exported remote code with the RELEASED checkpoint's own inference code.

    The training repo's `convert_molmoact2_to_hf` copies its own `modeling_molmoact2.py`, whose
    `predict_action` has no `enable_adaptive_depth` argument. The released checkpoint's does, and the
    adaptive-replay path is how every readout in this study injects a grid — DEPTHANCHOR's battery, the
    rollout loop and the attention probe all go through it. So an arm is evaluated with the same inference
    code the released checkpoint runs, which is also the only reading of "does the policy read the grid at
    inference" that means anything.

    The weights are untouched; only the code and the processor description are replaced. `config.json`
    stays the export's, because that is where the arm's own gate flags live, and its keys are diffed
    against the released config so a field the released code needs cannot go missing silently.
    """
    stage = pathlib.Path(work) / "released_code"
    stage.mkdir(parents=True, exist_ok=True)
    s3("sync", RELEASED_CODE_S3, str(stage), "--only-show-errors")
    copied, missing = [], []
    for name in REMOTE_CODE_FILES:
        srcf = stage / name
        if srcf.is_file():
            (pathlib.Path(hf_dir) / name).write_bytes(srcf.read_bytes())
            copied.append(name)
        else:
            missing.append(name)
    exported_cfg = json.loads((pathlib.Path(hf_dir) / "config.json").read_text())
    released_cfg = json.loads((stage / "config.json").read_text()) if (stage / "config.json").is_file() else {}
    only_released = sorted(set(released_cfg) - set(exported_cfg))
    only_exported = sorted(set(exported_cfg) - set(released_cfg))
    return {"copied": copied, "missing": missing,
            "config_keys_only_in_released": only_released,
            "config_keys_only_in_export": only_exported,
            "why": "the released predict_action takes enable_adaptive_depth; the export's does not"}


def convert_to_hf(src: pathlib.Path, merged: pathlib.Path, out_dir: pathlib.Path,
                  work: Optional[pathlib.Path] = None,
                  adopt_released_code: bool = True) -> Dict[str, object]:
    """`convert_molmoact2_to_hf`, so this study's own instruments can read the arm."""
    exp = src / "experiments"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{exp}:{exp / 'lerobot' / 'src'}:" + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "olmo.hf_model.convert_molmoact2_to_hf",
                           str(merged), str(out_dir)],
                          cwd=str(exp), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"HF conversion failed ({proc.returncode}):\n{proc.stdout[-2000:]}\n"
                           f"{proc.stderr[-3000:]}")
    adopted = (adopt_released_remote_code(out_dir, work or out_dir.parent)
               if adopt_released_code else {"copied": []})
    cfg = json.loads((out_dir / "config.json").read_text())
    return {"out_dir": str(out_dir), "seconds": round(time.perf_counter() - t0, 1),
            "released_code_adopted": adopted,
            "files": sorted(p.name for p in out_dir.iterdir()),
            "gate_in_config": {k: cfg.get(k) for k in
                               ("action_expert_depth_gate", "action_expert_depth_gate_per_layer",
                                "action_expert_depth_gate_init_bias")},
            "stdout_tail": proc.stdout[-1500:]}
