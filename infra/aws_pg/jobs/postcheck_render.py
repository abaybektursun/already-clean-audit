"""POSTCHECK cell B — P5, and the 40 frames the model has never seen.

Three things, in this order, because each licenses the next:

  P5(b) THE ARBITER, first, because §4 makes it the precondition for everything else here: ten episodes
        of LIBERO-Object under each of the four (vertical, horizontal) transforms of the render feed.
        The transform that reaches the released evaluation's ceiling is the dataset's. If 180° does not
        give at least 8 of 10, this environment is not the one DEPTHANCHOR validated and nothing else
        in P5 is trusted — the cell says so and stops.
  P5(a) THE IMAGE MATCH: for 20 episodes, the dataset's own first frame against renders of that task's
        shipped initial states, under all four transforms, by normalized correlation. DEPTHANCHOR found
        the released episodes do not start from the shipped init states, so this may fail to identify
        any transform; that outcome is recorded, not forced. The dataset frames are read fresh from the
        pinned revision, at the row locations the P4 scan MEASURED rather than the ones the metadata
        claims — which is the only way to read them correctly, and is itself P4's point.
  FRAMES the 40 rendered frames cells N1 and P3 need: 10 per suite, at steps 0, 20 and 40 of episodes
        started from held-out initial states, under the arbiter's winning transform. The checkpoint has
        never seen these images; that is the whole reason for rendering them.

  pg run infra.aws_pg.jobs.postcheck_render --type g6e.2xlarge \\
     --weights models--allenai--MolmoAct2-Think-LIBERO --s3-prefix results/postcheck/render/1 \\
     --budget-min 180 --args " --p4-prefix results/postcheck/files/1"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from infra.aws_pg.jobs.postcheck_files import Bank, _s3_json

BUCKET = "pg-lineage-769063704696"
REGION = "us-east-1"
S3_FRAMES = f"s3://{BUCKET}/postcheck/frames/v1"
DATASET_REPO = "allenai/MolmoAct2-LIBERO-Dataset"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ARBITER_SUITE = "libero_object"
ARBITER_EPISODES = 10
ARBITER_FLOOR = 8               # §4: 180° must reach this or nothing else in P5 is trusted
TRANSFORMS = ((True, True), (False, False), (False, True), (True, False))   # (vflip, hflip)
FRAME_STEPS = (0, 20, 40)
FRAMES_PER_SUITE = 10
HELD_OUT_INIT_FROM = 40         # DEPTHANCHOR's loop used init state 0; these are the far end
IMAGE_MATCH_EPISODES = 20


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized correlation of two images as flat float vectors."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"image sizes differ: {x.size} vs {y.size}")
    x = x - x.mean()
    y = y - y.mean()
    d = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if d < 1e-12 else float(x @ y / d)


def _transform(img: np.ndarray, vflip: bool, hflip: bool) -> np.ndarray:
    out = np.asarray(img)
    if vflip:
        out = out[::-1]
    if hflip:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


# ------------------------------------------------------------------ P5(b): the arbiter
def stage_arbiter(bank: Bank, actor, lib) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    per_transform: List[Dict[str, object]] = []
    for vflip, hflip in TRANSFORMS:
        policy = lib.ArmNormal()
        ok = 0
        for task_id in range(ARBITER_EPISODES):
            r = lib.rollout(actor, ARBITER_SUITE, task_id=task_id, episode=0, policy=policy,
                            flip=vflip, hflip=hflip, max_steps=lib.HORIZON[ARBITER_SUITE])
            rec = r.as_record()
            rec["render_flip"], rec["render_hflip"] = bool(vflip), bool(hflip)
            rows.append({k: v for k, v in rec.items()
                         if k not in ("eef_path", "bins_per_call", "gate_per_call", "tail_per_call")})
            ok += int(r.success)
            print(f"[arbiter] vflip={vflip} hflip={hflip} task {task_id}: success={r.success} "
                  f"steps={r.n_steps}", flush=True)
            bank.stage("arbiter_partial", {"done": len(rows), "rows": rows[-4:]})
        per_transform.append({"render_flip": bool(vflip), "render_hflip": bool(hflip),
                              "successes": ok, "episodes": ARBITER_EPISODES})
        print(f"[arbiter] vflip={vflip} hflip={hflip}: {ok}/{ARBITER_EPISODES}", flush=True)
    bank.artifact("p5b_arbiter_rows.json", rows)
    best = max(per_transform, key=lambda t: t["successes"])
    rot180 = next(t for t in per_transform if t["render_flip"] and t["render_hflip"])
    passing = [t for t in per_transform if t["successes"] >= ARBITER_FLOOR]
    return {
        "suite": ARBITER_SUITE, "episodes_per_transform": ARBITER_EPISODES,
        "floor": ARBITER_FLOOR, "by_transform": per_transform,
        "winner": {"render_flip": best["render_flip"], "render_hflip": best["render_hflip"],
                   "successes": best["successes"]},
        "rot180_successes": rot180["successes"],
        "rot180_passes": bool(rot180["successes"] >= ARBITER_FLOOR),
        "n_transforms_passing": len(passing),
        "unique_transform_passes": bool(len(passing) == 1),
        # §4's precondition and P5's kill condition, in one place
        "trusted": bool(rot180["successes"] >= ARBITER_FLOOR and len(passing) == 1
                        and best["render_flip"] and best["render_hflip"]),
    }


# ------------------------------------------------------------------ P5(a): the image match
def stage_image_match(bank: Bank, lib, work: pathlib.Path, p4: Optional[Dict[str, object]],
                      cam: Dict[str, object]) -> Dict[str, object]:
    """The dataset's own first frames against renders of the shipped init states, four transforms."""
    if not p4:
        return {"measured": False, "why": "no P4 scan available, so no measured row locations"}
    locations = _episode_locations(p4)
    if not locations:
        return {"measured": False, "why": "the P4 artifact carries no per-file episode listing"}
    frames = _fetch_dataset_first_frames(work, locations[:IMAGE_MATCH_EPISODES])
    bank.artifact("p5a_dataset_frames_index.json",
                  [{k: v for k, v in f.items() if k != "image"} for f in frames])
    if not frames:
        return {"measured": False, "why": "no dataset frames could be read"}

    rows = []
    unmatched = []
    for f in frames:
        # the dataset row carries a task_index and the task STRING; the camera bank is keyed by the
        # same string, so the language is the join. (The first draft looked for a `task_key` the
        # dataset does not carry, and matched nothing.)
        key = f.get("task_key") or (cam.get("by_language") or {}).get(f.get("language"))
        entry = cam["tasks"].get(key) if key else None
        if entry is None:
            unmatched.append({"episode_index": f.get("episode_index"),
                              "language": f.get("language"), "task_index": f.get("task_index")})
            continue
        env, bench, task = lib.make_env(entry["suite"], entry["task_id"], camera_depths=False,
                                        image_size=int(cam["image_size"]))
        try:
            init_states = bench.get_task_init_states(entry["task_id"])
            best = {name: {"corr": -2.0, "init": None} for name in ("id", "vflip", "hflip", "rot180")}
            for i in range(len(init_states)):
                env.reset()
                env.set_init_state(init_states[i])
                obs, _, _, _ = env.step(lib.NOOP)
                raw = np.asarray(obs["agentview_image"], dtype=np.uint8)
                for name, (vf, hf) in zip(("id", "vflip", "hflip", "rot180"),
                                          ((False, False), (True, False), (False, True), (True, True))):
                    c = _corr(_transform(raw, vf, hf), f["image"])
                    if c > best[name]["corr"]:
                        best[name] = {"corr": c, "init": i}
            rows.append({"episode_index": f["episode_index"], "task_key": key,
                         "language": str(task.language), "n_init_states": len(init_states),
                         "best": best})
            print(f"[p5a] ep {f['episode_index']} {key}: "
                  + " ".join(f"{k}={v['corr']:.3f}" for k, v in best.items()), flush=True)
        finally:
            env.close()
    bank.artifact("p5a_rows.json", rows)
    if not rows:
        return {"measured": False, "why": "no episode could be matched to a task",
                "n_frames_fetched": len(frames), "unmatched": unmatched[:10]}
    med = {name: float(np.median([r["best"][name]["corr"] for r in rows]))
           for name in ("id", "vflip", "hflip", "rot180")}
    order = sorted(med, key=lambda k: med[k], reverse=True)
    return {"measured": True, "n_episodes": len(rows), "n_unmatched": len(unmatched),
            "median_best_correlation": med,
            "winner": order[0], "margin": med[order[0]] - med[order[1]],
            # DEPTHANCHOR measured 0.085 as the best correlation over 50 init states: the released
            # episodes do not start from the shipped states, so a low ceiling here is the expected
            # outcome and is reported as such rather than read as an orientation answer
            "decisive": bool(med[order[0]] >= 0.5 and med[order[0]] - med[order[1]] >= 0.1),
            "note": ("a low best correlation means the released episodes do not start from the shipped "
                     "initial states, which is a fact about the dataset and not about the orientation")}


def _episode_locations(p4: Dict[str, object]) -> List[Dict[str, object]]:
    """Which file actually holds each episode's rows, from the P4 scan's per-file listing."""
    out = []
    for entry in (p4.get("per_file") or []):
        for e in (entry.get("episodes") or [])[:1]:
            out.append({"episode_index": int(e), "file_index": int(entry["file_index"])})
    return sorted(out, key=lambda r: r["episode_index"])


def _fetch_dataset_first_frames(work: pathlib.Path, locations: List[Dict[str, object]]
                                ) -> List[Dict[str, object]]:
    """Read the first frame of each named episode from the released files, fresh and pinned.

    Runs in a child process: `run_job.sh` exports `HF_HUB_OFFLINE=1` for the weight cache and
    `huggingface_hub` reads that into a module constant at import time.
    """
    import os
    import sys
    script = work / "fetch_frames.py"
    script.write_text(_FETCH_CHILD)
    spec = work / "fetch_spec.json"
    spec.write_text(json.dumps(locations))
    dest = work / "dataset_frames.npz"
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "0"
    if "HF_TOKEN" not in env:
        tok = subprocess.run(["aws", "ssm", "get-parameter", "--name", "/pg-lineage/hf-token",
                              "--with-decryption", "--region", REGION,
                              "--query", "Parameter.Value", "--output", "text"],
                             capture_output=True, text=True)
        if tok.returncode == 0:
            env["HF_TOKEN"] = tok.stdout.strip()
    proc = subprocess.run([sys.executable, str(script), DATASET_REPO, str(spec), str(dest)],
                          env=env, capture_output=True, text=True)
    print(proc.stdout[-3000:], flush=True)
    if proc.returncode != 0:
        print(f"[p5a] fetching dataset frames failed: {proc.stderr[-1500:]}", flush=True)
        return []
    data = np.load(dest, allow_pickle=False)
    meta = json.loads((work / "dataset_frames_meta.json").read_text())
    return [dict(m, image=data[f"img{i}"]) for i, m in enumerate(meta)]


_FETCH_CHILD = '''\
"""Child: the first frame of each named episode, from the released files at a pinned revision."""
import io, json, pathlib, sys
import numpy as np
from huggingface_hub import HfApi, HfFileSystem
import pyarrow.parquet as pq
from PIL import Image

repo, spec_p, dest = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
want = json.loads(spec_p.read_text())
api = HfApi(); sha = api.dataset_info(repo).sha
print("DATASET_REVISION", sha, flush=True)
fs = HfFileSystem()
tasks = None
try:
    with fs.open(f"datasets/{repo}@{sha}/meta/tasks.parquet", "rb") as fh:
        tasks = pq.read_table(fh).to_pylist()
except Exception as exc:
    print("TASKS_UNREADABLE", exc, flush=True)

imgs, meta = {}, []
by_file = {}
for w in want:
    by_file.setdefault(int(w["file_index"]), []).append(int(w["episode_index"]))
n = 0
for fi, eps in sorted(by_file.items()):
    path = f"datasets/{repo}@{sha}/data/chunk-000/file-{fi:03d}.parquet"
    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        names = pf.schema_arrow.names
        img_col = next((c for c in names if c.endswith("images.image")), None)
        if img_col is None:
            print("NO_IMAGE_COLUMN", names[:12], flush=True); break
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["episode_index", "frame_index", "task_index", img_col])
            e = t.column("episode_index").to_pylist()
            fr = t.column("frame_index").to_pylist()
            ti = t.column("task_index").to_pylist()
            im = t.column(img_col).to_pylist()
            for k in range(len(e)):
                if int(e[k]) in eps and int(fr[k]) == 0:
                    raw = im[k]
                    b = raw["bytes"] if isinstance(raw, dict) else raw
                    arr = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)
                    imgs[f"img{n}"] = arr
                    lang = None
                    if tasks:
                        for row in tasks:
                            if int(row.get("task_index", -1)) == int(ti[k]):
                                lang = row.get("task"); break
                    meta.append({"episode_index": int(e[k]), "file_index": fi,
                                 "task_index": int(ti[k]), "language": lang,
                                 "shape": list(arr.shape)})
                    n += 1
                    print("FRAME", meta[-1], flush=True)
np.savez_compressed(dest, **imgs)
pathlib.Path(dest.parent / "dataset_frames_meta.json").write_text(json.dumps(meta, indent=1))
print("FETCH_DONE", n, flush=True)
'''


# ------------------------------------------------------------------ the 40 unseen frames
def stage_frames(bank: Bank, actor, lib, cam: Dict[str, object],
                 vflip: bool, hflip: bool) -> Dict[str, object]:
    """Roll the released policy from HELD-OUT initial states and keep the observation at steps 0, 20
    and 40. These images exist only because this rollout happened, so no part of them is in training."""
    frames: List[Dict[str, object]] = []
    index: List[Dict[str, object]] = []
    for suite in SUITES:
        kept = 0
        task_id = 0
        while kept < FRAMES_PER_SUITE and task_id < 10:
            entry = cam["tasks"][f"{suite}/{task_id}"]
            env, bench, task = lib.make_env(suite, task_id, camera_depths=False,
                                            image_size=int(cam["image_size"]))
            try:
                init_states = bench.get_task_init_states(task_id)
                init_i = (HELD_OUT_INIT_FROM + task_id) % len(init_states)
                env.reset()
                obs = env.set_init_state(init_states[init_i])
                for _ in range(lib.SETTLE_STEPS):
                    obs, _, _, _ = env.step(lib.NOOP)
                policy = lib.ArmNormal()
                policy.reset({"env": env, "task": task, "episode": init_i, "suite": suite,
                              "task_id": task_id, "obs": obs})
                step = 0
                want = set(FRAME_STEPS)
                while step <= max(FRAME_STEPS) and kept < FRAMES_PER_SUITE:
                    if step in want:
                        f = lib.obs_to_frame(obs, flip=vflip, hflip=hflip, task=task.language)
                        f["spec"] = {"frame_id": f"render:{suite}/{task_id}:init{init_i}:s{step}",
                                     "suite": suite, "task_key": f"{suite}/{task_id}",
                                     "task_id": task_id, "init_state": init_i, "step": step,
                                     "language": str(task.language)}
                        frames.append(f)
                        index.append(dict(f["spec"]))
                        kept += 1
                        want.discard(step)
                    ctx = {"env": env, "task": task, "episode": init_i, "suite": suite,
                           "task_id": task_id, "obs": obs, "step": step}
                    res = policy.act(actor, lib.obs_to_frame(obs, flip=vflip, hflip=hflip,
                                                             task=task.language), step, ctx)
                    for a in res.a_robot:
                        obs, _, done, _ = env.step(np.asarray(a, dtype=np.float64))
                        step += 1
                        if step > max(FRAME_STEPS):
                            break
            finally:
                env.close()
            task_id += 1
        print(f"[frames] {suite}: {kept} frames", flush=True)
    _save_frames(bank.out / "frames", frames)
    subprocess.run(["aws", "s3", "cp", str(bank.out / "frames" / "frames.npz"),
                    f"{S3_FRAMES}/frames.npz", "--region", REGION], check=True)
    subprocess.run(["aws", "s3", "cp", str(bank.out / "frames" / "frames_index.json"),
                    f"{S3_FRAMES}/frames_index.json", "--region", REGION], check=True)
    return {"n_frames": len(frames), "per_suite": {s: sum(1 for i in index if i["suite"] == s)
                                                   for s in SUITES},
            "steps": sorted({i["step"] for i in index}),
            "render_flip": bool(vflip), "render_hflip": bool(hflip),
            "held_out_init_from": HELD_OUT_INIT_FROM,
            "index": index}


def _save_frames(dest: pathlib.Path, frames: List[Dict[str, object]]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    index = []
    for i, f in enumerate(frames):
        arrays[f"agentview{i}"] = np.asarray(f["agentview"], dtype=np.uint8)
        arrays[f"wrist{i}"] = np.asarray(f["wrist"], dtype=np.uint8)
        arrays[f"state{i}"] = np.asarray(f["state"], dtype=np.float32)
        index.append({"i": i, "task": f["task"], "spec": f["spec"]})
    np.savez_compressed(dest / "frames.npz", **arrays)
    (dest / "frames_index.json").write_text(json.dumps({"n_frames": len(frames), "index": index},
                                                       indent=1))


def load_frames(src: pathlib.Path) -> List[Dict[str, object]]:
    """The counterpart read, used by cells N1 and P3."""
    data = np.load(src / "frames.npz", allow_pickle=False)
    idx = json.loads((src / "frames_index.json").read_text())
    out = []
    for row in idx["index"]:
        i = row["i"]
        out.append({"agentview": data[f"agentview{i}"], "wrist": data[f"wrist{i}"],
                    "state": data[f"state{i}"], "task": row["task"], "spec": row["spec"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/work/postcheck")
    ap.add_argument("--p4-prefix", required=True,
                    help="the files cell whose measured episode locations P5(a) reads")
    ap.add_argument("--stages", default="env,cam,arbiter,frames,imagematch")
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    bank = Bank(pathlib.Path(args.out), "postcheck_render")
    bank.top("args", vars(args))
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    from infra.pg import depthanchor_libero as lib
    if "env" in stages:
        bank.stage("env", {"install": lib.install(extra_numpy=None)})

    cam_path = bank.out / "camera_bank.json"
    if "cam" in stages:
        cam = lib.camera_bank()
        cam_path.write_text(json.dumps(cam, indent=1, default=str))
        bank.stage("cam", {"n_tasks": len(cam["tasks"]), "image_size": cam["image_size"]})
    else:
        cam = json.loads(cam_path.read_text())
    cam = json.loads(cam_path.read_text())

    actor = None
    if {"arbiter", "frames"} & set(stages):
        from infra.pg import depthanchor_model as dam
        model, processor, path = dam.load_model()
        actor = dam.Actor(model, processor, enable_cuda_graph=False)
        bank.stage("model", {"snapshot": str(path), "cuda_graph": False})
    else:
        # the image match is simulator and arithmetic only; loading 21.8 GB to correlate images would
        # be a waste of the one thing this lane is short of
        bank.stage("model", {"loaded": False, "why": "no stage in this run needs the checkpoint"})

    t0 = time.perf_counter()
    arb = None
    if "arbiter" in stages:
        arb = stage_arbiter(bank, actor, lib)
        bank.stage("p5b_arbiter", arb)
        if not arb["trusted"]:
            bank.top("verdict", {"P5b_trusted": False, "arbiter": arb,
                                 "blocked": "the arbiter did not put 180° alone above the floor; "
                                            "P5 is not trusted and the frames are not banked"})
            raise SystemExit(f"P5(b) arbiter failed: {json.dumps(arb['by_transform'])}")
    vflip, hflip = (arb["winner"]["render_flip"], arb["winner"]["render_hflip"]) if arb else (True, True)

    if "frames" in stages:
        bank.stage("frames", stage_frames(bank, actor, lib, cam, vflip, hflip))
    if "imagematch" in stages:
        p4 = _s3_json(f"s3://{BUCKET}/{args.p4_prefix}/out/p4_scan.json", work / "p4_scan.json")
        bank.stage("p5a_image_match", stage_image_match(bank, lib, work, p4, cam))

    st = bank.payload["stages"]
    bank.top("verdict", {
        "P5b_trusted": (st.get("p5b_arbiter") or {}).get("trusted"),
        "P5b_by_transform": (st.get("p5b_arbiter") or {}).get("by_transform"),
        "P5a_winner": (st.get("p5a_image_match") or {}).get("winner"),
        "P5a_decisive": (st.get("p5a_image_match") or {}).get("decisive"),
        "P5a_median_best_correlation": (st.get("p5a_image_match") or {}).get("median_best_correlation"),
        "frames_banked": (st.get("frames") or {}).get("n_frames"),
        "seconds_total": round(time.perf_counter() - t0, 1),
    })
    print(json.dumps(bank.payload["verdict"], indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
