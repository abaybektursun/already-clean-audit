"""CTRLIFACE confirmation — Arm A, the MDE (spec §9 step 5, UNBLIND ONCE). On the SEALED suites
(goal + libero_10/Long), roll out {S_action(k) / energy-matched-random / identity} across the frozen
k-grid at L*, plus the depth profile S_action(k*,L) across all layers. The headline is the
task-cluster-bootstrap RF gap (RF_random_energy − RF_Saction) with ci_lo vs the frozen δ.

Loads the discovery bundle (bases.pt) + the FROZEN thresholds (controls.json). Fills the §11 grid.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ctrliface_confirm --type g6e.8xlarge,g6e.4xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/ctrliface_confirm --budget-min 300 \
      --args "--bundle-s3 s3://.../ctrliface_discovery/out/bases.pt --controls-s3 s3://.../ctrliface_controls/out/controls.json --tasks-per-suite 4 --n-episodes 8"
"""
import argparse
import itertools
import json
import pathlib
import subprocess
import sys

SEALED_SUITE_ENVS = {"goal": "libero_goal", "long": "libero_10"}
K_GRID = (1, 2, 4, 8, 16, 32, 64, 128)


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
    ap.add_argument("--bundle-s3", required=True)
    ap.add_argument("--controls-s3", required=True)
    ap.add_argument("--tasks-per-suite", type=int, default=4)
    ap.add_argument("--n-episodes", type=int, default=8)
    ap.add_argument("--k-star", type=int, default=None)
    ap.add_argument("--L-star", type=int, default=None)
    ap.add_argument("--k-grid", default="1,2,4,8,16,32,64,128", help="csv k-sweep points at L*")
    ap.add_argument("--depth-stride", type=int, default=1, help="depth-profile every Nth layer")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    global K_GRID
    K_GRID = tuple(int(x) for x in a.k_grid.split(","))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    _install()
    import os
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.pg import ctrliface_lerobot as LR
    from infra.pg import ctrliface_core as CC
    from infra.pg import ctrliface_rollout_lerobot as RL

    subprocess.run(["aws", "s3", "cp", a.bundle_s3, "/tmp/bases.pt", "--region", "us-east-1"], check=True)
    subprocess.run(["aws", "s3", "cp", a.controls_s3, "/tmp/controls.json", "--region", "us-east-1"], check=True)
    bundle = torch.load("/tmp/bases.pt", map_location="cpu", weights_only=False)
    ctrl = json.loads(pathlib.Path("/tmp/controls.json").read_text())
    thr = ctrl["frozen_thresholds"]
    floor = ctrl["floor"]
    L_star = a.L_star if a.L_star is not None else ctrl["L"]
    k_star = a.k_star if a.k_star is not None else ctrl["k"]
    d = bundle["d"]
    bases = {int(L): bundle["bases"][int(L)].to("cuda") for L in bundle["bases"]}
    hbars = {int(L): bundle["hbar"][int(L)].to("cuda") for L in bundle["hbar"]}
    traces = {int(L): float(bundle["trace"][int(L)]) for L in bundle["trace"]}

    rep = {"ws": "CTRLIFACE", "policy": LR.POLICY_REPO, "sealed_suites": list(SEALED_SUITE_ENVS),
           "L_star": L_star, "k_star": k_star, "d": d, "floor": floor, "thresholds": thr,
           "n_episodes": a.n_episodes, "tasks_per_suite": a.tasks_per_suite, "seed": a.seed}

    policy = LR.load_policy()
    _, layers, _ = LR.get_backbone(policy)
    stats = RL.dataset_stats()
    pre, post = RL.load_processors(policy, dataset_stats=stats)

    # cache discovery activations per layer (for energy-matched-null E_target per (L,k)) — via the
    # compute_layer_complete monkeypatch (π0.5's fused forward bypasses layer-module forward hooks).
    from infra.pg import ctrliface_pi05_hook as PH
    meta = LR.dataset_meta()
    dt = LR.build_delta_timestamps(policy, meta)
    loader, _, _ = LR.build_loader(LR.DATASET_REPO, dt, suites=None, batch_size=2, seed=1,
                                   max_episodes=40)
    pre_batches = [pre(b) for b in itertools.islice(loader, 6)]
    Xcat = PH.capture_activations(policy, lambda b: policy.forward(b), pre_batches, d,
                                  list(range(len(layers))), n_prefix=None)

    def E_target(L, k):
        return CC.removed_energy(Xcat[L], bases[L][:, :k].cpu(), hbars[L].cpu())

    tasks = [(env, tid) for env in SEALED_SUITE_ENVS.values() for tid in range(a.tasks_per_suite)]

    def _roll(kind, L, k):
        by_task = {}
        for env, tid in tasks:
            spec = {"kind": kind, "L": L, "k": k}
            by_task[f"{env}:{tid}"] = RL.eval_task(
                policy, pre, post, layers, d, env, tid, a.n_episodes, spec, bases, hbars, a.seed,
                n_prefix=None, E_target=E_target(L, k), trace_C=traces[L])
        return by_task

    # intact once (identity at L*, k*)
    intact = _roll("identity", L_star, k_star)
    intact_mean = float(np.mean(np.concatenate([np.asarray(v) for v in intact.values()])))
    rep["intact_success"] = intact_mean

    # --- Arm A, k-sweep at L* ---
    rf_by_k = {}
    for k in K_GRID:
        cell = {}
        sa = _roll("saction", L_star, k)
        re = _roll("random_energy", L_star, k)
        cell["saction"] = CC.rf_cluster_bootstrap(sa, intact, floor, seed=a.seed)
        cell["random_energy"] = CC.rf_cluster_bootstrap(re, intact, floor, seed=a.seed)
        cell["identity"] = {"rf": 1.0, "ci": [1.0, 1.0]}
        if k == k_star:
            cell["gap"] = CC.gap_cluster_bootstrap(sa, re, intact, floor, seed=a.seed)
            rep["gap_at_kstar"] = cell["gap"]
            rep["_sa_kstar"], rep["_re_kstar"] = sa, re
        rf_by_k[str(k)] = cell
    rep["arm_a"] = {"rf_by_k": rf_by_k}

    # --- Arm A, depth profile at k* (every depth-stride-th layer; L* always included) ---
    rf_by_L = {}
    depth_layers = sorted(set(list(range(0, len(layers), a.depth_stride)) + [L_star]))
    for L in depth_layers:
        sa = _roll("saction", L, k_star)
        rf_by_L[str(L)] = {"saction": CC.rf_cluster_bootstrap(sa, intact, floor, seed=a.seed)}
    rep["arm_a"]["rf_by_L"] = rf_by_L
    rep["arm_a"]["gap_at_kstar"] = rep.get("gap_at_kstar")

    # --- frozen verdict (axis 1) ---
    rf_sa = rf_by_k[str(k_star)]["saction"]["rf"]
    gap_lo = rep["gap_at_kstar"]["ci_lo"]
    ks_sorted = [k for k in K_GRID if k <= k_star]
    monotone = all(rf_by_k[str(ks_sorted[i])]["saction"]["rf"] >= rf_by_k[str(ks_sorted[i + 1])]["saction"]["rf"] - 0.05
                   for i in range(len(ks_sorted) - 1))
    rep["verdict"] = {"axis1": CC.interface_verdict(rf_sa, gap_lo, thr["tau_low"], thr["delta"], monotone),
                      "monotone_ok": monotone}
    rep.pop("_sa_kstar", None); rep.pop("_re_kstar", None)
    (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    print("CONFIRM_OK", json.dumps({"verdict_axis1": rep["verdict"]["axis1"], "rf_saction_kstar": rf_sa,
                                    "gap_ci_lo": gap_lo, "intact": intact_mean, "floor": floor,
                                    "k_star": k_star, "L_star": L_star}, default=str))


if __name__ == "__main__":
    main()
