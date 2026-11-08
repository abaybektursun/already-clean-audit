"""PLENUM Arm 3 — the CONTROL cells at step 0 (spec §2.3).

`behavior/control/task_native` is defined for every unit with an action expert (pi0, pi05, u3, u4,
u3c). I had been treating all five as blocked behind Arm-3 training, because §2.3's *savings*
contrast is stated at budgets {B*/4, B*}. That was a misreading of my own frozen spec: the §2.3
instrument clause says, verbatim,

    "EGL-before-CUDA; step-0 evals double as the on-chain zero-shot cells."

So the step-0 (pre-training) control eval IS a pre-registered cell, and four of the five are
measurable with no training run at all. The instrument is `lerobot-eval` exactly as the map's probe
table names it, and `n5a_pi0_libero_eval` already proved the LIBERO/MuJoCo/EGL stack renders on the
pg AMI (probe 3, non-degenerate frames at std≈50) and already takes an arbitrary `--policy` path.
This job is the roster loop over it, at the frozen §2.3 primary population: LIBERO-Spatial, 10 tasks
× 5 episodes.

WHAT THIS NUMBER IS, AND WHAT IT IS NOT. It is zero-shot control competence on LIBERO-Spatial under
each unit's OWN normalization statistics. None of these units is LIBERO-trained: π0/π0.5 ship norm
stats for their own training mixes, and U3/U4 carry freshly initialised action experts. So a reading
at or near zero is expected for the fresh-expert units and is a real floor measurement (it is the
floor the §2.3 savings contrast is measured FROM), not a failed eval. What it is NOT is a statement
about these policies' trained control ability, and it must never be quoted as one — the scope string
travels with every cell.

The discriminating content is the ORDERING: U3/U4 have randomly initialised experts and therefore
cannot exceed chance-of-accident, while π0/π0.5 are trained generalist policies whose zero-shot
transfer to an unseen suite is an open quantity. If the trained policies read *identically* to the
random-expert units, that is informative about the eval's reach and is reported as such.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.plenum_control --type g6e.2xlarge --no-weights \
      --args "--units pi0" --s3-prefix results/plenum_ctl_pi0 --budget-min 400
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import subprocess
import sys
import time

# §2.3 frozen primary eval population.
SUITE = "libero_spatial"
TASK_IDS = "[0,1,2,3,4,5,6,7,8,9]"
EPISODES = 5
N_ACTION_STEPS = 10          # matches the OpenPI / lerobot LIBERO reproduction

# unit -> policy path (HF id, or an S3 prefix synced to a local dir)
UNIT_POLICY = {
    "pi0": "lerobot/pi0_base",
    "pi05": "lerobot/pi05_base",
    "u3": "s3://pg-lineage-769063704696/weights/lacuna_units/U3",
    "u4": "s3://pg-lineage-769063704696/weights/lacuna_units/U4",
    "u3c": "s3://pg-lineage-769063704696/weights/lacuna_units/U3c",
}


def unit_etag(uri):
    """S3 ETag of a unit's weights, recorded so a ROW is self-verifying.

    Two cells of the same unit's row must have read the same weights. Today a battery straddled a
    re-publish — `fc`/`gen` synced the previous run's weights and `text`/`ctl` synced the new ones — and
    NOTHING in the artifacts flagged it, because every job recorded the same unit *URI* while reading
    different *objects*. A URI names a location; an ETag names the bytes.
    """
    import subprocess
    if not str(uri).startswith("s3://"):
        return None
    b = str(uri)[len("s3://"):].split("/", 1)
    r = subprocess.run(["aws", "s3api", "head-object", "--bucket", b[0],
                        "--key", f"{b[1].rstrip('/')}/model.safetensors",
                        "--query", "ETag", "--output", "text"],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def _run(cmd, **kw):
    print(f"[PCTL] $ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def _install():
    _run(["bash", "-lc",
          "apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y "
          "libegl1 libgles2 libglvnd0 libgl1 libosmesa6 libglfw3 libglib2.0-0 libsm6 "
          "libxext6 libxrender1 libgomp1 git python3-dev python3.12-dev build-essential"])
    _run([sys.executable, "-m", "pip", "install", "-q",
          "lerobot[libero,pi] @ git+https://github.com/huggingface/lerobot.git"])
    # EGL BEFORE CUDA (§2.3, and the Modal-B200 ogbench lesson: the reverse order segfaults).
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
    # LIBERO writes an interactive config on FIRST import; there is no stdin under lerobot-eval.
    subprocess.run('echo "N" | %s -c "import libero.libero"' % sys.executable, shell=True)


def _pc_success(eval_dir):
    """pc_success + the per-episode successes, read from lerobot-eval's real schema.

    The schema is nested: `{"per_task": [...], "per_group": {suite: {...}}, "overall": {...}}`, and
    the aggregate lives at `overall.pc_success`. My first version looked for a top-level
    `aggregated` block with `pc_success` in it and asserted when it found neither — which is the
    right failure (a control cell whose success rate could not be located is NOT a zero, and
    silently reporting one would be the worst possible failure mode for this particular number).

    The per-episode `successes` lists are banked too: pc_success is a mean over 50 Bernoulli
    episodes, so a CI on it needs the episodes, not the mean.
    """
    found = sorted(glob.glob(str(pathlib.Path(eval_dir) / "**" / "*.json"), recursive=True))
    assert found, f"lerobot-eval wrote no json under {eval_dir}"
    for f in found:
        d = json.loads(pathlib.Path(f).read_text())
        if not (isinstance(d, dict) and isinstance(d.get("overall"), dict)
                and "pc_success" in d["overall"]):
            continue
        eps = []
        for t in d.get("per_task", []):
            eps.extend(int(bool(x)) for x in (t.get("metrics", {}).get("successes") or []))
        return float(d["overall"]["pc_success"]), {
            "file": f, "overall": d["overall"].get("pc_success"),
            "n_episodes": d["overall"].get("n_episodes"),
            "eval_s": d["overall"].get("eval_s"),
            "per_group": {k: v.get("pc_success") for k, v in (d.get("per_group") or {}).items()},
            "per_episode_success": eps}
    raise AssertionError(
        f"no json under {eval_dir} has overall.pc_success; files seen: {found}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--suite", default=SUITE)
    ap.add_argument("--task-ids", default=TASK_IDS)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--n-action-steps", type=int, default=N_ACTION_STEPS)
    ap.add_argument("--as-unit", default=None,
                    help="map-unit id this eval fills, when the policy is a ckpt:. Needed for the "
                         "§2.3 step-0 cells: pi0/pi05 CANNOT be evaluated from their published "
                         "weights (lerobot-eval refuses the feature-schema pairing), so their cell is "
                         "read from a feature-adapted near-step-0 checkpoint. Without this the row is "
                         "tagged is_checkpoint and routed to the film clock, filling no map cell.")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    units = [u.strip() for u in a.units.split(",") if u.strip()]
    for u in units:
        assert u in UNIT_POLICY or u.startswith("ckpt:"), (
            f"{u}: no policy path known (has_expert units, or ckpt:<s3-uri> for the film clock)")

    t0 = time.time()
    rep = {"object": "plenum_control_v1", "units": units, "suite": a.suite,
           "task_ids": a.task_ids, "episodes": a.episodes,
           "readout": "zero-shot pc_success at step 0 (§2.3: step-0 evals double as the on-chain "
                      "zero-shot cells)",
           "scope_caveat": "each unit's OWN normalization stats; no unit is LIBERO-trained. A "
                           "near-zero reading for a fresh-expert unit is the floor, not a failure.",
           "as_unit": a.as_unit,
           "label": "EXPLORATORY (§7) — never in a BH family"}
    cells = []
    rep["cells"] = cells                      # bound BEFORE the loop (fleet lesson)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    _install()
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import lerobot
    import transformers
    rep["transformers"] = transformers.__version__
    rep["lerobot"] = getattr(lerobot, "__version__", "?")
    _w()

    binp = pathlib.Path(sys.executable).parent / "lerobot-eval"
    base = [str(binp)] if binp.exists() else [sys.executable, "-m", "lerobot.scripts.eval"]

    for uid in units:
        # `ckpt:` gives the film its ACQUISITION clock: pc_success at each checkpoint, which is
        # what `acquisition_step` needs to be defined on the same grid as the onset.
        src = uid[len("ckpt:"):] if uid.startswith("ckpt:") else UNIT_POLICY[uid]
        if src.startswith("s3://"):
            import hashlib as _h
            local = pathlib.Path("/opt/dlami/nvme/plenum_ctl/"
                                 + _h.sha256(uid.encode()).hexdigest()[:12])
            local.mkdir(parents=True, exist_ok=True)
            _run(["aws", "s3", "sync", src, str(local)])
            pol = str(local)
            # PROCESSOR MIGRATION. `policy.save_pretrained()` writes weights + config but NOT the
            # processor/normalization files, and lerobot-eval refuses such a checkpoint:
            #   ProcessorMigrationError: Model '...' requires migration to processor format.
            # The banked U3/U4 units carry `policy_preprocessor.json` (lacuna_units wrote them that
            # way), which is why they evaluate as-is while every checkpoint this program trains does
            # not. lerobot ships the converter and names it in the error; run it rather than
            # hand-copying normalization stats from the source unit, which would silently substitute
            # one policy's statistics for another's if training ever recomputed them.
            if not (local / "policy_preprocessor.json").exists():
                print(f"[PCTL] migrating {pol} to processor format", flush=True)
                _run([sys.executable, "-m", "lerobot.processor.migrate_policy_normalization",
                      "--pretrained-path", pol])
                assert (local / "policy_preprocessor.json").exists(), (
                    f"migration reported success but wrote no policy_preprocessor.json under {pol}")
        else:
            pol = src
        # a `ckpt:s3://.../00000512` uid cannot be a directory name
        safe = uid.replace("ckpt:", "ck_").replace("s3://", "").replace("/", "_").replace(":", "_")
        eval_out = out / f"eval_{safe}"
        _run(base + [
            f"--output_dir={eval_out}",
            "--env.type=libero",
            f"--env.task={a.suite}",
            f"--env.task_ids={a.task_ids}",
            "--eval.batch_size=1",
            f"--eval.n_episodes={a.episodes}",
            f"--policy.path={pol}",
            f"--policy.n_action_steps={a.n_action_steps}",
            "--env.max_parallel_tasks=1",
        ], env={**os.environ})
        pc, hits = _pc_success(eval_out)
        # A checkpoint is NOT a map unit: it is a point on the film's acquisition clock. Tagging it
        # as one would mint a garbage cell id like `ckpt:s3://...|behavior/control/task_native`.
        cells.append({"unit": a.as_unit or uid,
                      "weights_etag": unit_etag(src if src.startswith("s3://") else ""),
                      "is_checkpoint": uid.startswith("ckpt:") and not a.as_unit,
                      "from_checkpoint": uid if a.as_unit else None,
                      "probe": "behavior/control/task_native", "policy": pol,
                      "pc_success": pc,
                      "n_episodes": hits.get("n_episodes") or a.episodes * 10,
                      "aggregated": hits})
        print(f"[PCTL] {uid:>6s} pc_success={pc:.4f} t={time.time()-t0:.0f}s", flush=True)
        _w()

    rep["elapsed_s"] = round(time.time() - t0, 1)
    _w()
    print("PLENUM_CONTROL_OK " + json.dumps({c["unit"]: c["pc_success"] for c in cells}), flush=True)


if __name__ == "__main__":
    main()
