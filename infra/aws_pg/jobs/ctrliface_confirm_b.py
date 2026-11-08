"""CTRLIFACE Arm-B, part 2 — the disjointness mechanism (spec §3 Arm B, §6 axis 2). Answers: is the
control interface S_action geometrically DISJOINT from the text-sensitivity subspace S_text, and does
ablating text directions leave control intact? (The mechanistic explanation for N5a+PCR/RFP: text lived
where the controller does not read.)

  (i)  overlap o(S_action, S_text) at L* vs a matched-random overlap null (principal-angle).
  (ii) causal cross-check: ablate S_text(k*,L*) during sealed rollout → RF_control(S_text).
Verdict axis 2 = disjoint_verdict(overlap, null_p95, RF_control(S_text), τ_high). Per P6 (π0 text
floored) the 'ablate→text' cell is dropped by design; the informative cells are overlap + S_text→control.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ctrliface_confirm_b --type g6e.2xlarge,g5.4xlarge,g5.2xlarge \
      --no-weights --s3-prefix results/ctrliface_confirm_b --budget-min 180 \
      --args "--bundle-s3 s3://.../ctrliface_discovery/out/bases.pt --stext-s3 s3://.../ctrliface_stext/out/stext_bases.pt --controls-s3 s3://.../ctrliface_controls/out/controls.json --confirm-s3 s3://.../ctrliface_confirm/out/metrics.json"
"""
import argparse
import json
import pathlib
import subprocess
import sys

SEALED_SUITE_ENVS = {"goal": "libero_goal", "long": "libero_10"}


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


def _cp(s3, local):
    subprocess.run(["aws", "s3", "cp", s3, local, "--region", "us-east-1"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle-s3", required=True)
    ap.add_argument("--stext-s3", required=True)
    ap.add_argument("--controls-s3", required=True)
    ap.add_argument("--confirm-s3", required=True)
    ap.add_argument("--tasks-per-suite", type=int, default=4)
    ap.add_argument("--n-episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
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

    for s3, loc in [(a.bundle_s3, "/tmp/bases.pt"), (a.stext_s3, "/tmp/stext.pt"),
                    (a.controls_s3, "/tmp/controls.json"), (a.confirm_s3, "/tmp/confirm.json")]:
        _cp(s3, loc)
    bundle = torch.load("/tmp/bases.pt", map_location="cpu", weights_only=False)
    stext = torch.load("/tmp/stext.pt", map_location="cpu", weights_only=False)
    ctrl = json.loads(pathlib.Path("/tmp/controls.json").read_text())
    conf = json.loads(pathlib.Path("/tmp/confirm.json").read_text())
    thr, floor = ctrl["frozen_thresholds"], ctrl["floor"]
    L_star, k_star, d = conf["L_star"], conf["k_star"], bundle["d"]

    sa = {int(L): bundle["bases"][int(L)] for L in bundle["bases"]}
    st = {int(L): stext["stext_bases"][int(L)] for L in stext["stext_bases"]}
    hbars = {int(L): bundle["hbar"][int(L)].to("cuda") for L in bundle["hbar"]}
    traces = {int(L): float(bundle["trace"][int(L)]) for L in bundle["trace"]}

    rep = {"L_star": L_star, "k_star": k_star, "d": d, "thresholds": thr, "floor": floor,
           "sealed_suites": list(SEALED_SUITE_ENVS)}

    # --- (i) geometric overlap at L* (+ full depth profile) + matched-random null ---
    def _overlap(L, k):
        Qa = CC.orthonormalize(sa[L][:, :k]); Qt = CC.orthonormalize(st[L][:, :k])
        ov, cos = CC.principal_angle_overlap(Qa, Qt)
        null = CC.matched_random_overlap(d, k, k, n_draws=300, seed=a.seed, ref=Qa)
        return ov, [float(x) for x in cos], float(null.mean()), float(null.quantile(0.95))
    ov, cos, null_mean, null_p95 = _overlap(L_star, k_star)
    depth = {}
    for L in range(len(sa)):
        o, _, _, p95 = _overlap(L, k_star)
        depth[str(L)] = {"overlap": o, "null_p95": p95}
    rep["arm_b"] = {"overlap": {"value": ov}, "overlap_null": {"mean": null_mean, "p95": null_p95},
                    "cos_spectrum": cos, "overlap_by_L": depth}

    # --- (ii) S_text → control rollout on sealed suites (kind='saction' with S_text bases) ---
    policy = LR.load_policy()
    _, layers, _ = LR.get_backbone(policy)
    stats = RL.dataset_stats()
    pre, post = RL.load_processors(policy, dataset_stats=stats)
    tasks = [(env, tid) for env in SEALED_SUITE_ENVS.values() for tid in range(a.tasks_per_suite)]

    intact_by_task, stext_by_task = {}, {}
    for env, tid in tasks:
        key = f"{env}:{tid}"
        intact_by_task[key] = RL.eval_task(policy, pre, post, layers, d, env, tid, a.n_episodes,
                                           {"kind": "identity", "L": L_star, "k": k_star},
                                           sa, hbars, a.seed)
        stext_by_task[key] = RL.eval_task(policy, pre, post, layers, d, env, tid, a.n_episodes,
                                          {"kind": "saction", "L": L_star, "k": k_star},
                                          st, hbars, a.seed)          # S_text bases → ablate S_text
    rf_stext = CC.rf_cluster_bootstrap(stext_by_task, intact_by_task, floor, seed=a.seed)
    rep["arm_b"]["rf_control_stext"] = rf_stext

    # --- verdict axis 2 ---
    rep["arm_b"]["twoby2"] = {"saction_control": conf.get("arm_a", {}).get("rf_by_k", {})
                              .get(str(k_star), {}).get("saction", {}).get("rf"),
                              "stext_control": rf_stext["rf"],
                              "saction_text": "N/A (π0 text floored; P6)", "stext_text": "N/A (P6)"}
    rep["verdict"] = {"axis2": CC.disjoint_verdict(ov, null_p95, rf_stext["rf"], thr["tau_high"])}
    (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    print("CONFIRM_B_OK", json.dumps({"axis2": rep["verdict"]["axis2"], "overlap": ov,
                                      "null_p95": null_p95, "rf_control_stext": rf_stext["rf"]},
                                     default=str))


if __name__ == "__main__":
    main()
