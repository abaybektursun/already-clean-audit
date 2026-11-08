"""DEPTHOFF — the released policy on standard LIBERO with the depth pathway on, repeated, zeroed and absent.

SPEC §1-§2: five conditions on ONE suite per instance, every condition run back to back on the SAME task,
the SAME banked initial state and the SAME flow-matching seed, so an episode is a paired unit. Running the
conditions in one process is not a convenience — DEPTHANCHOR's 7.4 % run-to-run flip rate was measured
ACROSS instances, and ON2 exists to measure that floor here, inside the same process, where it should be
smaller.

  ON      the released deployment path, untouched
  ON2     the same path again — the noise floor every removal arm is compared against
  ZERO    the 36 per-layer depth scalars forced to 0.0: depth positions present, carrying nothing
  NOGEN   `enable_depth_reasoning=False`: no depth block generated at all (the deployment removal)
  BROKEN  another task's agentview image, depth normal — the positive control (one suite only)

ZERO and NOGEN are different operations and are never pooled: a zeroed-but-present prefix perturbs the
expert more than an absent one (DEPTHOPEN measured 0.022 against 0.0094 on an ungated arm).

Rows bank after EVERY rollout, not after the loop, so a job killed at hour two still yields its pairs.

  pg run infra.aws_pg.jobs.depthoff_loop --type g6e.2xlarge \\
     --weights models--allenai--MolmoAct2-Think-LIBERO --budget-min 300 \\
     --s3-prefix results/depthoff/loop/libero_object/1 \\
     --args " --suite libero_object --episodes-per-task 5 --broken"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, List, Optional

import numpy as np

from infra.aws_pg.jobs import _depthanchor_shared as sh
from infra.pg import depthanchor_core as core
from infra.pg import depthoff_rules as rules

CONDITIONS = ("ON", "ON2", "ZERO", "NOGEN")
BROKEN = "BROKEN"
BROKEN_SUITE = "libero_object"
TASKS_PER_SUITE = 10


# ---------------------------------------------------------------- the conditions as arm policies
def build_condition(name: str, *, broken_image: Optional[np.ndarray] = None):
    """One closed-loop condition. `force_gate` is set by the caller around the rollout, not in here,
    because it is actor state rather than policy state."""
    from infra.pg import depthanchor_libero as lib

    if name in ("ON", "ON2", "ZERO"):
        # all three take the released deployment path; ZERO differs only in the actor's forced scalar
        arm = lib.ArmNormal()
        arm.name = name
        return arm

    if name == "NOGEN":
        class ArmNoGen(lib.ArmPolicy):
            """`enable_depth_reasoning=False` through POSTCHECK's own call, which mirrors the Actor's
            path with that one flag changed. The Actor hard-codes depth reasoning on."""
            name = "NOGEN"

            def act(self, actor, frame, step, ctx):
                from infra.pg.postcheck_battery import call_no_depth
                return call_no_depth(actor, frame, seed=0)

        return ArmNoGen()

    if name == BROKEN:
        if broken_image is None:
            raise ValueError("BROKEN needs an image from another task")

        class ArmBroken(lib.ArmPolicy):
            """The positive control: the agentview replaced by another task's frame every step, depth
            pathway untouched. If this does NOT collapse, the loop is not sensitive to what it sees and
            every equivalence verdict in this study is cancelled."""
            name = "BROKEN"

            def reset(self, ctx):
                self._cache = None

            def act(self, actor, frame, step, ctx):
                swapped = dict(frame)
                swapped["agentview"] = broken_image
                res, self._cache = actor.adaptive(swapped, self._cache)
                return res

        return ArmBroken()

    raise ValueError(f"unknown condition {name!r}")


def grab_other_task_image(suite: str, task_id: int, *, flip: bool, hflip: bool) -> np.ndarray:
    """One agentview frame from a DIFFERENT task in the same suite, for the positive control."""
    from infra.pg import depthanchor_libero as lib
    env, bench, task = lib.make_env(suite, task_id, camera_depths=False, image_size=lib.IMAGE_SIZE)
    try:
        init_states = bench.get_task_init_states(task_id)
        env.reset()
        obs = env.set_init_state(init_states[0])
        for _ in range(lib.SETTLE_STEPS):
            obs, _, _, _ = env.step(lib.NOOP)
        frame = lib.obs_to_frame(obs, flip=flip, hflip=hflip, task=task.language)
        return np.ascontiguousarray(np.asarray(frame["agentview"], dtype=np.uint8))
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/work/depthoff")
    ap.add_argument("--suite", required=True, choices=list(core.SUITES))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--tasks", type=int, default=TASKS_PER_SUITE)
    ap.add_argument("--episodes-per-task", type=int, default=5,
                    help="SPEC §2: banked init states 0..N-1, so every condition sees the same starts")
    ap.add_argument("--first-episode", type=int, default=0,
                    help="SPEC §4's pre-registered extension starts at 5")
    ap.add_argument("--broken", action="store_true",
                    help="also run the positive control (SPEC §1: libero_object only)")
    ap.add_argument("--broken-episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = the suite's own horizon")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    bank = sh.Bank(out, f"depthoff:{args.suite}")
    bank.top("args", vars(args))
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in CONDITIONS:
            raise ValueError(f"unknown condition {c!r}; known: {CONDITIONS}")

    # ---- environment, exactly DEPTHANCHOR's, including the numpy-pin fallback it needed
    from infra.pg import depthanchor_libero as lib
    env_report = lib.install(extra_numpy=None)
    smoke_ok = True
    try:
        e, _b, _t = lib.make_env(args.suite, 0, camera_depths=False, image_size=lib.IMAGE_SIZE)
        e.close()
    except Exception as exc:
        print(f"[depthoff] simulator import failed without the numpy pin: {exc}", flush=True)
        smoke_ok = False
    if not smoke_ok:
        env_report = lib.install(extra_numpy="numpy<2")
        e, _b, _t = lib.make_env(args.suite, 0, camera_depths=False, image_size=lib.IMAGE_SIZE)
        e.close()
    bank.stage("env", {"install": env_report, "numpy_pin_needed": not smoke_ok})

    # ---- the render transform, read from the banked artifact and never defaulted
    orient = json.loads(sh.pull_json(f"{sh.S3_CAMERA}/orientation.json",
                                     work / "orientation.json").read_text())
    for key in ("render_flip", "render_hflip"):
        if orient.get(key) is None:
            raise RuntimeError(f"the banked orientation artifact has no `{key}`")
    flip, hflip = bool(orient["render_flip"]), bool(orient["render_hflip"])
    bank.stage("inputs", {
        "render_flip": flip, "render_hflip": hflip,
        "render_transform_decided_by": orient.get("render_transform_decided_by"),
        "suite": args.suite, "horizon": lib.HORIZON[args.suite], "settle_steps": lib.SETTLE_STEPS,
        "conditions": conditions, "tasks": args.tasks,
        "episodes": list(range(args.first_episode, args.first_episode + args.episodes_per_task)),
        "frozen_rules": {"delta_points": rules.DELTA_POINTS, "on_floor": rules.ON_FLOOR,
                         "broken_ceiling": rules.BROKEN_CEILING},
        "pairing": "same task, same banked init state, same seed; conditions run back to back in one "
                   "process so the ON/ON2 floor is measured where the comparison is made"})

    # ---- the released checkpoint, and its depth scalars read from the weights before any rollout
    from infra.pg import depthanchor_model as dam
    model, processor, path = dam.load_model()
    actor = dam.Actor(model, processor, enable_cuda_graph=False)
    gate = dam.gate_weights_readout(model)
    bank.stage("model", {"snapshot": str(path), "cuda_graph": "off", "gate_weights": gate})

    broken_image = None
    if args.broken:
        if args.suite != BROKEN_SUITE:
            raise ValueError(f"SPEC §1 runs the positive control on {BROKEN_SUITE} only")
        # task 9's frame shown to tasks 0..8 — always a different task from the one being scored
        broken_image = grab_other_task_image(args.suite, args.tasks - 1, flip=flip, hflip=hflip)
        bank.stage("broken_control", {"image_from_task": args.tasks - 1,
                                      "image_shape": list(broken_image.shape)})

    t0 = time.perf_counter()
    rows: List[Dict[str, object]] = []
    episodes = list(range(args.first_episode, args.first_episode + args.episodes_per_task))

    def run_one(cond: str, task_id: int, ep: int) -> None:
        arm = build_condition(cond, broken_image=broken_image)
        # ZERO is the only condition that touches actor state; set it immediately around the rollout and
        # clear it after, so a raised exception cannot leak a forced scalar into the next condition
        actor.force_gate(0.0 if cond == "ZERO" else None)
        try:
            r = lib.rollout(actor, args.suite, task_id=task_id, episode=ep, policy=arm,
                            flip=flip, hflip=hflip, film=False, want_depth=False,
                            max_steps=args.max_steps or None)
        finally:
            actor.force_gate(None)
        rec = r.as_record()
        rec["condition"] = cond
        rec["pair_id"] = f"{args.suite}/{task_id}/ep{ep}"
        rec["episode_index_used"] = ep
        # the realized scalar the model actually applied, per call — the audit that ZERO took effect
        gates = [g for g in (rec.get("gate_per_call") or []) if g is not None]
        rec["realized_gate_mean"] = float(np.mean([np.mean(g) for g in gates])) if gates else None
        rows.append(rec)
        print(f"[depthoff] {cond} {args.suite}/{task_id} ep{ep}: success={r.success} "
              f"steps={r.n_steps} gate={rec['realized_gate_mean']} {r.seconds:.0f}s", flush=True)
        bank.stage("progress", {"rollouts_done": len(rows), "condition": cond,
                                "elapsed_s": round(time.perf_counter() - t0, 1),
                                "seconds_per_rollout": round(
                                    (time.perf_counter() - t0) / len(rows), 1)})
        bank.artifact("rollouts.json", [{k: v for k, v in x.items()
                                         if k not in ("bins_per_call", "gate_per_call")} for x in rows])

    # every condition of one episode back to back, so a killed job loses whole episodes, never half a pair
    for task_id in range(args.tasks):
        for ep in episodes:
            for cond in conditions:
                run_one(cond, task_id, ep)

    if args.broken:
        for task_id in range(min(args.broken_episodes, args.tasks)):
            run_one(BROKEN, task_id, episodes[0])

    bank.stage("summary", summarize(rows, suite=args.suite))
    bank.flush()


def summarize(rows: List[Dict[str, object]], *, suite: str) -> Dict[str, object]:
    """Per-condition rates, the paired grid, and the frozen rules run on whatever landed."""
    by_cond: Dict[str, Dict[str, int]] = {}
    for r in rows:
        c = str(r["condition"])
        by_cond.setdefault(c, {})[str(r["pair_id"])] = int(bool(r["success"]))

    # paired episodes = those present in EVERY non-BROKEN condition, in one fixed order
    paired_conds = [c for c in CONDITIONS if c in by_cond]
    common = None
    for c in paired_conds:
        keys = set(by_cond[c])
        common = keys if common is None else (common & keys)
    pair_ids = sorted(common or [])
    outcomes = {c: [by_cond[c][p] for p in pair_ids] for c in paired_conds}
    if BROKEN in by_cond:
        outcomes[BROKEN] = [by_cond[BROKEN][p] for p in sorted(by_cond[BROKEN])]

    latency: Dict[str, object] = {}
    for c in sorted(by_cond):
        mine = [r for r in rows if r["condition"] == c]
        secs = [float(r["seconds"]) for r in mine]
        calls = [int((r.get("meta") or {}).get("calls") or 0) for r in mine]
        per_call = [s / k for s, k in zip(secs, calls) if k > 0]
        latency[c] = {"episodes": len(mine),
                      "median_seconds_per_episode": round(float(np.median(secs)), 2) if secs else None,
                      "median_seconds_per_call": round(float(np.median(per_call)), 4) if per_call else None,
                      "median_calls": int(np.median(calls)) if calls else None}

    realized = {}
    for c in sorted(by_cond):
        vals = [r["realized_gate_mean"] for r in rows
                if r["condition"] == c and r.get("realized_gate_mean") is not None]
        realized[c] = round(float(np.median(vals)), 6) if vals else None

    out: Dict[str, object] = {
        "suite": suite, "n_paired_episodes": len(pair_ids), "pair_ids": pair_ids,
        "rates": {c: rules.wilson(sum(v), len(v)) for c, v in outcomes.items()},
        "outcomes": outcomes, "latency": latency, "realized_gate_median": realized,
    }
    # the rules need ON and ON2; a partial job may not have them yet, and that is not an error here
    if "ON" in outcomes and "ON2" in outcomes and len(pair_ids) > 0:
        out["decided"] = rules.decide(outcomes)
    else:
        out["decided"] = {"verdict": "not_decidable_yet",
                          "why": "ON and ON2 are both needed on at least one paired episode"}
    return out


if __name__ == "__main__":
    main()
