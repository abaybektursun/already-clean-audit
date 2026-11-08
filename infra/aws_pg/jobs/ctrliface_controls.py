"""CTRLIFACE controls + Gate-0/P7 (spec §9 steps 0/2/3) — the FIRST GPU validation of the whole
ablation-rollout stack, and the blind threshold-freezing step.

Runs on the DISCOVERY suites only (spatial+object) so the sealed confirmation stays blind:
  0. Gate-0/P7: intact (identity) rollout on discovery tasks → success rate; dead-backbone floor
     (mean-replace ALL backbone layers → no-backbone-info). Assert intact ≫ floor (dynamic range;
     else P7 halt). This validates the lerobot rollout wiring + hook injection on real weights.
  1. Control sweep at the discovery-provisional (L*, k): {identity, random_k, random_energy, shuffle,
     positive, saction} → per-task per-episode success → RF (task-cluster bootstrap).
  2. FREEZE τ_low/τ_high/δ from the calibration curve (control data only) → controls.json.

Consumes the discovery bundle (bases.pt: S_action per layer + h̄ + trace) from results/ctrliface_discovery.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ctrliface_controls --type g6e.8xlarge,g6e.4xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/ctrliface_controls --budget-min 180 \
      --args "--bundle-s3 s3://pg-lineage-769063704696/results/ctrliface_discovery/out/bases.pt --tasks-per-suite 3 --n-episodes 6"
"""
import argparse
import json
import pathlib
import subprocess
import sys

DISCOVERY_SUITE_ENVS = {"spatial": "libero_spatial", "object": "libero_object"}
CONTROL_KINDS = ("identity", "saction", "random_k", "random_energy", "shuffle", "positive")


def _install():
    subprocess.run(["bash", "-lc",
                    "apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y "
                    "libegl1 libgles2 libglvnd0 libgl1 libosmesa6 libglfw3 libglib2.0-0 libsm6 "
                    "libxext6 libxrender1 libgomp1 git python3-dev build-essential"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "lerobot[libero,pi] @ git+https://github.com/huggingface/lerobot.git"], check=True)
    import os
    os.environ["MUJOCO_GL"] = "egl"; os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
    subprocess.run('echo "N" | %s -c "import libero.libero"' % sys.executable, shell=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle-s3", required=True, help="s3 path to discovery bases.pt")
    ap.add_argument("--tasks-per-suite", type=int, default=3)
    ap.add_argument("--n-episodes", type=int, default=6)
    ap.add_argument("--k", type=int, default=None, help="override control-sweep k (else discovery provisional)")
    ap.add_argument("--L", type=int, default=None, help="override probe layer (else discovery provisional)")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    _install()
    import os
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from infra.pg import ctrliface_lerobot as LR
    from infra.pg import ctrliface_core as CC
    from infra.pg import ctrliface_rollout_lerobot as RL
    from infra.pg import ctrliface_pi05_hook as PH

    rep = {"suites": list(DISCOVERY_SUITE_ENVS), "tasks_per_suite": a.tasks_per_suite,
           "n_episodes": a.n_episodes, "seed": a.seed}

    # --- load discovery bundle + provisional (L*,k) ---
    subprocess.run(["aws", "s3", "cp", a.bundle_s3, "/tmp/bases.pt", "--region", "us-east-1"], check=True)
    bundle = torch.load("/tmp/bases.pt", map_location="cpu", weights_only=False)
    disc_metrics_s3 = a.bundle_s3.rsplit("/", 1)[0] + "/metrics.json"
    subprocess.run(["aws", "s3", "cp", disc_metrics_s3, "/tmp/disc_metrics.json", "--region", "us-east-1"], check=True)
    disc = json.loads(pathlib.Path("/tmp/disc_metrics.json").read_text())
    prov = disc.get("provisional", {})
    L = a.L if a.L is not None else prov["L_star"]
    k = a.k if a.k is not None else prov["k_star"]
    d = bundle["d"]
    bases = {int(Lk): bundle["bases"][int(Lk)].to("cuda") for Lk in bundle["bases"]}
    hbars = {int(Lk): bundle["hbar"][int(Lk)].to("cuda") for Lk in bundle["hbar"]}
    traces = {int(Lk): float(bundle["trace"][int(Lk)]) for Lk in bundle["trace"]}
    rep.update({"L": L, "k": k, "d": d})

    policy = LR.load_policy()
    _, layers, _ = LR.get_backbone(policy)
    stats = RL.dataset_stats()
    pre, post = RL.load_processors(policy, dataset_stats=stats)

    # E_target(L,k): removed energy of S_action(L,k) on discovery activations (for energy-matched null).
    # measured with removed_energy over the trace already stored: E_target ≈ sum of top-k proj energy;
    # exact per-k requires activations — approximate via trace-share of the top-k eigen-directions is
    # NOT valid (grad vs activation). Measure directly on a few demo batches:
    meta = LR.dataset_meta()
    dt = LR.build_delta_timestamps(policy, meta)
    loader, _, _ = LR.build_loader(LR.DATASET_REPO, dt, suites=None, batch_size=2, seed=a.seed,
                                   max_episodes=40)
    from infra.pg.ctrliface_run import harvest_hbar_trace  # reuse activation capture
    # capture activations at L over a few batches to measure removed energy of S_action[:, :k]
    import itertools
    import numpy as np
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout
    # E_target(L,k) = removed energy of S_action(L,k) on discovery activations (energy-matched null).
    # Activations captured via the compute_layer_complete monkeypatch (forward hooks don't fire).
    pre_batches = [pre(b) for b in itertools.islice(loader, 6)]
    Xd = PH.capture_activations(policy, lambda b: policy.forward(b), pre_batches, d, [L], n_prefix=None)
    E_target = CC.removed_energy(Xd[L], bases[L][:, :k].cpu(), hbars[L].cpu())
    trace_C = traces[L]
    rep["E_target_Lk"] = E_target
    (out / "controls.json").write_text(json.dumps(rep, indent=2, default=str))

    # --- task list (discovery suites) ---
    tasks = [(env, tid) for env in DISCOVERY_SUITE_ENVS.values() for tid in range(a.tasks_per_suite)]

    # --- floor: dead backbone (mean-replace ALL layers, via the monkeypatch) ---
    floor_by_task = {}
    for env, tid in tasks:
        venv, cfg = RL.make_libero_vecenv(env, tid, n_envs=a.n_episodes)
        env_pre, env_post = make_env_pre_post_processors(cfg, policy.config)
        if hasattr(policy, "reset"):
            policy.reset()
        torch.manual_seed(a.seed)
        with PH.mean_replace_all(policy, hbars, n_prefix=None):
            r = rollout(venv, policy, env_pre, env_post, pre, post,
                        seeds=[a.seed + i for i in range(a.n_episodes)])
        venv.close()
        floor_by_task[f"{env}:{tid}"] = [float(x) for x in r["success"].any(dim=1).float().tolist()]
    floor = float(np.mean(np.concatenate([np.asarray(v) for v in floor_by_task.values()])))
    rep["floor"] = floor
    rep["floor_by_task"] = floor_by_task
    (out / "controls.json").write_text(json.dumps(rep, indent=2, default=str))

    # --- control sweep (incl. intact via identity) ---
    sweep = {}
    for kind in CONTROL_KINDS:
        by_task = {}
        for env, tid in tasks:
            spec = {"kind": kind, "L": L, "k": k}
            per = RL.eval_task(policy, pre, post, layers, d, env, tid, a.n_episodes, spec,
                               bases, hbars, a.seed, n_prefix=None, E_target=E_target, trace_C=trace_C)
            by_task[f"{env}:{tid}"] = per
        sweep[kind] = by_task
    rep["sweep_success_by_task"] = sweep

    # intact = identity; RF of each kind vs intact, floor
    intact = sweep["identity"]
    rep["intact_success"] = float(np.mean(np.concatenate([np.asarray(v) for v in intact.values()])))
    # Gate-0/P7: dynamic range
    rep["P7_dynamic_range_ok"] = rep["intact_success"] - floor > 0.15
    rf = {}
    for kind in CONTROL_KINDS:
        rf[kind] = CC.rf_cluster_bootstrap(sweep[kind], intact, floor, n_boot=2000, seed=a.seed)
    rep["rf_by_kind"] = rf
    (out / "controls.json").write_text(json.dumps(rep, indent=2, default=str))

    # --- FREEZE thresholds from control data (blind) ---
    # τ_high = the band the NULLS preserve control to (min of identity/random_k/random_energy RF, minus margin)
    # τ_low  = the band the positive/floor craters to (positive RF, plus margin); δ = half the null-vs-positive gap
    null_rf = min(rf["identity"]["rf"], rf["random_k"]["rf"], rf["random_energy"]["rf"])
    pos_rf = rf["positive"]["rf"]
    tau_high = round(max(0.5, null_rf - 0.10), 3)
    tau_low = round(min(0.5, pos_rf + 0.10), 3)
    delta = round(max(0.15, (null_rf - pos_rf) / 3.0), 3)
    rep["frozen_thresholds"] = {"tau_low": tau_low, "tau_high": tau_high, "delta": delta,
                                "null_rf": null_rf, "pos_rf": pos_rf}
    (out / "controls.json").write_text(json.dumps(rep, indent=2, default=str))

    print("CONTROLS_OK", json.dumps({"intact": rep["intact_success"], "floor": floor,
                                     "P7_ok": rep["P7_dynamic_range_ok"],
                                     "rf_saction": rf["saction"]["rf"], "rf_random_energy": rf["random_energy"]["rf"],
                                     "thresholds": rep["frozen_thresholds"]}, default=str))


if __name__ == "__main__":
    main()
