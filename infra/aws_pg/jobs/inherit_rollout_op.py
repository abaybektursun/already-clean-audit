"""INHERIT Arms O + P (π0.5 rollout cells; spec §3, §9 step 7). Reuses the CTRLIFACE-validated
lerobot rollout binding (correct obs/action/normalization by construction) with hook-instrumented
ablations. Both arms are π0.5 rollouts on the SEALED CTRLIFACE suites (goal + libero_10) — the same
instrument, the same firewall, stated not re-derived.

  Arm O (type): ablate the residual entering the π0.5 bite layer L14 with basis families
    {top-k S_action, mass-matched low-σ tail, mid-band} from the SELF-CONSISTENT re-estimated full
    Gram (bases_full.pt; the frozen bank lacks a full spectrum — disclosed), RF of rollout success
    vs captured Gram mass. H_privileged iff top-k craters below the mass-matched tail at matched
    mass (gap ci-lo > δ_O); H_spectral iff the (RF, mass) points of all families collapse.

  Arm P-causal (identity): ablate U_unt on the q/k INPUT of every π0.5 backbone layer (the
    invariant_core-FUNC contrast, transported to the control readout) vs its energy-matched random
    null. Plus the FREE weight-side check: did robot training WRITE into U_unt?
    ‖Δq_proj·U_unt‖/‖Δq_proj‖ vs a random-subspace null (no rollout).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.inherit_rollout_op --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/inherit_rollout_op --budget-min 300 \
      --args "--full-bank-s3 s3://.../inherit_pi05_bank/out/bases_full.pt --arm both"
"""
import argparse
import contextlib
import json
import pathlib
import subprocess
import sys

SEALED_SUITE_ENVS = {"goal": "libero_goal", "long": "libero_10"}
DELTA_O_FLOOR = 0.15                 # pre-registered minimal meaningful rollout RF gap (CTRLIFACE convention)
ARM_O_LAYER = 14                     # the CTRLIFACE π0.5 bite layer L* (recomputed/asserted from controls.json)
ARM_O_KGRID = (4, 16, 64)


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


@contextlib.contextmanager
def _ablate_qk_input(layers, U_by_layer, device):
    """Project the q/k INPUT onto the complement of U (x' = x − U(Uᵀx)) on every π0.5 backbone layer
    that has a U — the invariant_core-FUNC _InputProject contract, transported to the deployed policy.
    q_proj/k_proj submodules are called directly in BOTH forward paths → a submodule pre-hook fires
    for the Gram-time joint forward AND the inference forward."""
    import torch
    handles = []
    Us = {L: torch.as_tensor(U, device=device, dtype=torch.float32) for L, U in U_by_layer.items()}

    def mk(Uq):
        def hook(_m, args):
            x = args[0]
            coeff = x.float() @ Uq
            return ((x.float() - coeff @ Uq.transpose(0, 1)).to(x.dtype),) + tuple(args[1:])
        return hook

    for L, Uq in Us.items():
        for proj in ("q_proj", "k_proj"):
            m = getattr(layers[L].self_attn, proj)
            handles.append(m.register_forward_pre_hook(mk(Uq)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _roll(policy, pre, post, L, Q, hbar, ctxmaker, seed, tasks, n_episodes):
    """One arm's rollouts over the sealed tasks. ctxmaker() → the ablation context (or nullcontext).
    Returns {f'{env}:{tid}': [per-episode success]}."""
    import torch
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout
    from infra.pg import ctrliface_rollout_lerobot as RL
    by_task = {}
    for env, tid in tasks:
        venv, cfg = RL.make_libero_vecenv(env, tid, n_envs=n_episodes)
        env_pre, env_post = make_env_pre_post_processors(cfg, policy.config)
        if hasattr(policy, "reset"):
            policy.reset()
        torch.manual_seed(seed)
        with ctxmaker():
            r = rollout(venv, policy, env_pre, env_post, pre, post,
                        seeds=[seed + i for i in range(n_episodes)])
        venv.close()
        by_task[f"{env}:{tid}"] = [float(x) for x in r["success"].any(dim=1).float().tolist()]
    return by_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--full-bank-s3", required=True)
    ap.add_argument("--arm", choices=["O", "P", "both"], default="both")
    ap.add_argument("--tasks-per-suite", type=int, default=2)
    ap.add_argument("--n-episodes", type=int, default=6)
    ap.add_argument("--uunt-k", type=int, default=128)
    ap.add_argument("--n-cap-batches", type=int, default=6, help="forwards for the q/k E_target capture")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    _install()
    import os
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _inherit_shared as SH
    from infra.pg import ctrliface_core as CC
    from infra.pg import ctrliface_lerobot as LR
    from infra.pg import ctrliface_pi05_hook as PH
    from infra.pg import ctrliface_rollout_lerobot as RL
    from infra.pg import inherit_core as IC

    # frozen bank (h̄/trace, π0.5-native) + the self-consistent re-estimated full Gram
    bank = torch.load(SH.fetch(SH.CTRLIFACE_BANK, "/tmp/inherit/bases.pt"),
                      map_location="cpu", weights_only=False)
    full = torch.load(SH.fetch(a.full_bank_s3, "/tmp/inherit/bases_full.pt"),
                      map_location="cpu", weights_only=False)["full"]
    ctrl = json.loads(SH.fetch(SH.CTRLIFACE_CONTROLS, "/tmp/inherit/ctrliface_controls.json").read_text())
    L_bite = int(ctrl["L"]); floor = float(ctrl["floor"])
    assert L_bite == ARM_O_LAYER, f"CTRLIFACE bite layer {L_bite} != {ARM_O_LAYER}"
    d = int(bank["d"])
    hbar_L = torch.as_tensor(bank["hbar"][L_bite], dtype=torch.float64)
    rep = {"arm": a.arm, "L_bite": L_bite, "sealed_suites": list(SEALED_SUITE_ENVS),
           "n_episodes": a.n_episodes, "tasks_per_suite": a.tasks_per_suite, "delta_O": DELTA_O_FLOOR}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    policy = LR.load_policy()
    _bk, layers, _d = LR.get_backbone(policy)
    meta = LR.dataset_meta()
    pre, post = RL.load_processors(policy, dataset_stats=meta.stats)
    tasks = [(env, tid) for env in SEALED_SUITE_ENVS.values() for tid in range(a.tasks_per_suite)]

    def intact_ctx():
        return contextlib.nullcontext()
    intact = _roll(policy, pre, post, L_bite, None, None, intact_ctx, a.seed, tasks, a.n_episodes)
    rep["intact_by_task"] = intact
    intact_pool = np.concatenate([np.asarray(v) for v in intact.values()])
    rep["intact_success"] = float(intact_pool.mean())
    _w()

    def rf(by_task):
        return CC.rf_cluster_bootstrap(by_task, intact, floor, seed=a.seed)

    # ---------------- Arm O ----------------
    if a.arm in ("O", "both"):
        evals = torch.as_tensor(full[L_bite]["evals"], dtype=torch.float64)  # descending
        vecs = torch.as_tensor(full[L_bite]["vecs"], dtype=torch.float64)    # (d,d)
        total_mass = float(evals.sum())
        arm_o = {}
        rep["arm_o"] = {"by_k": arm_o, "verdict": None}          # write progressively (survive a crash)
        for k in ARM_O_KGRID:
            topk_mass = float(evals[:k].sum())
            families = {"topk": vecs[:, :k]}
            cell = {"captured_mass_frac": topk_mass / total_mass}
            # a mass-matched tail/mid may be INFEASIBLE — top-k already holds > the complement/mid
            # mass (spectral concentration, itself H_privileged-consistent). Record + skip, no crash.
            if topk_mass <= float(evals[k:].sum()):
                j, tmass, _ = IC.mass_matched_tail(evals, k)
                families["tail"] = vecs[:, j:]
                cell.update(tail_start=j, tail_dims=int(vecs.shape[1] - j))
            else:
                cell["tail_infeasible"] = f"top-{k} mass {topk_mass:.3g} > complement {float(evals[k:].sum()):.3g}"
            if k < (vecs.shape[1] // 2) and topk_mass <= float(evals[vecs.shape[1] // 2:].sum()):
                lo, hi, mmass, _ = IC.mass_matched_mid(evals, k)
                families["mid"] = vecs[:, lo:hi]
                cell.update(mid=[int(lo), int(hi)], mid_dims=int(hi - lo))
            else:
                cell["mid_infeasible"] = True
            byfam = {}
            for fam, Q in families.items():
                def mk(Qf=Q):
                    return PH.ablate_residual(policy, L_bite, Qf, hbar_L, None)
                bt = _roll(policy, pre, post, L_bite, Q, hbar_L, mk, a.seed, tasks, a.n_episodes)
                byfam[fam] = bt
                cell[fam] = rf(bt)
            if "tail" in byfam:                                  # decisive contrast (matched mass)
                cell["gap_tail_minus_topk"] = CC.gap_cluster_bootstrap(
                    byfam["topk"], byfam["tail"], intact, floor, seed=a.seed)  # RF(tail)−RF(topk)
            arm_o[f"k{k}"] = cell
            _w()
        # verdict: H_privileged iff top-k craters BELOW a FEASIBLE mass-matched tail (gap ci_lo>δ_O);
        # infeasible tails (top-k mass > complement) are themselves concentration evidence, noted.
        feasible = [k for k in ARM_O_KGRID if "gap_tail_minus_topk" in arm_o.get(f"k{k}", {})]
        privileged = any(arm_o[f"k{k}"]["gap_tail_minus_topk"]["ci_lo"] > DELTA_O_FLOOR for k in feasible)
        rep["arm_o"]["verdict"] = "H_privileged" if privileged else "H_spectral"
        rep["arm_o"]["feasible_k"] = feasible
        rep["arm_o"]["concentration_note"] = (
            "mass-matched tail INFEASIBLE at k where top-k mass > complement — the Gram is top-heavy "
            "(spectral concentration), consistent with a privileged-subspace read")
        _w()

    # ---------------- Arm P-causal ----------------
    if a.arm in ("P", "both"):
        uunt = SH.derive_uunt(list(range(18)), k=a.uunt_k)
        U_by_layer = {L: uunt[L][0].numpy() for L in range(18)}

        # (a) FREE weight-side check: did robot training write into U_unt?  ‖ΔW·U‖/‖ΔW‖ vs random
        model_pg1, _proc = None, None
        from infra.pg import inherit_pg1 as IP
        model_pg1, _proc = IP.load_pg1(device="cpu")
        pg1_layers = IP.decoder_layers(model_pg1)
        write = {}
        for L in range(18):
            Wq_pi = layers[L].self_attn.q_proj.weight.detach().to(torch.float64).cpu()
            Wq_pg = pg1_layers[L].self_attn.q_proj.weight.detach().to(torch.float64).cpu()
            dW = Wq_pi - Wq_pg                                   # (out,in): robot-training write
            U = torch.as_tensor(U_by_layer[L], dtype=torch.float64)  # (in,k)
            frac_unt = float((dW @ U).norm() / dW.norm())        # energy of the write inside U_unt
            rng = np.random.default_rng(1000 + L)
            fr = []
            for s in range(5):
                R = IC.random_basis(d, a.uunt_k, 1000 + L * 7 + s, "cpu", torch.float64)
                fr.append(float((dW @ R).norm() / dW.norm()))
            write[str(L)] = {"frac_in_uunt": frac_unt, "rand_frac_mean": float(np.mean(fr)),
                             "rand_frac_std": float(np.std(fr))}
        rep["uunt_weight_write"] = write
        del model_pg1
        _w()

        # (b) energy-matched null rank for the q/k-input ablation (capture q_proj-input Gram)
        dt = LR.build_delta_timestamps(policy, meta)
        loader, _, _ = LR.build_loader(LR.DATASET_REPO, dt, suites=None, batch_size=2, seed=1,
                                       max_episodes=40)
        cap_batches = [pre(b) for b in __import__("itertools").islice(loader, a.n_cap_batches)]
        from infra.aws_pg.jobs.invariant_core_func import _InputCapture
        with _InputCapture(list(layers)) as capt:
            with torch.no_grad():
                for b in cap_batches:
                    policy.forward(b)
        # per-layer removed energy of U_unt on the q/k input, matched-random rank
        R_by_layer = {}
        em_ranks = {}
        for L in range(18):
            G = torch.as_tensor(capt.gram[L], dtype=torch.float64)
            U = torch.as_tensor(U_by_layer[L], dtype=torch.float64)
            E_unt = float((U.transpose(0, 1) @ G @ U).trace())
            trace_C = float(G.trace())
            r_e = CC.energy_matched_rank(E_unt, trace_C, d)
            R_by_layer[L] = IC.random_basis(d, r_e, 5000 + L, "cpu", torch.float64).numpy()
            em_ranks[str(L)] = r_e
        rep["uunt_em_ranks"] = em_ranks
        _w()

        def uunt_ctx():
            return _ablate_qk_input(layers, U_by_layer, "cuda")

        def emrand_ctx():
            return _ablate_qk_input(layers, R_by_layer, "cuda")

        bt_unt = _roll(policy, pre, post, L_bite, None, None, uunt_ctx, a.seed, tasks, a.n_episodes)
        bt_em = _roll(policy, pre, post, L_bite, None, None, emrand_ctx, a.seed, tasks, a.n_episodes)
        rf_unt = rf(bt_unt)
        rf_em = rf(bt_em)
        gap = CC.gap_cluster_bootstrap(bt_unt, bt_em, intact, floor, seed=a.seed)  # RF(em)−RF(unt)
        # H_primitive-read CAUSAL leg: U_unt cell beats its em-null (ci_lo > δ_P); reuse δ_O floor
        rep["arm_p_causal"] = {"rf_uunt": rf_unt, "rf_em": rf_em, "gap_em_minus_uunt": gap,
                               "causal_leg_pass": bool(gap["ci_lo"] > DELTA_O_FLOOR)}
        _w()

    print("ROLLOUT_OP_OK", json.dumps({"arm": a.arm,
                                        "arm_o": rep.get("arm_o", {}).get("verdict"),
                                        "arm_p": rep.get("arm_p_causal", {}).get("causal_leg_pass")},
                                       default=str))


if __name__ == "__main__":
    main()
