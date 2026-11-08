"""CTRLIFACE discovery (spec §9 step 4, Arm A identification) — estimate the action-sensitivity
subspace S_action(k,L) of the π0.5 backbone and pick provisional (k*, L*) by the cheap
action-deviation proxy. ROLLOUT-FREE (forward/backward only) → cheap; the sealed rollout confirms.

Firewall: demos filtered to the DISCOVERY suites (spatial+object) only — the sealed goal+long
rollouts never influence k*/L* (spec §Freeze-resolutions, §5).

Pipeline (all on the tested pure core + shared lerobot bindings):
  1. Gram G_L = Σ ∇_{h_L}(flow-matching loss) ∇ᵀ over N demo batches, ALL 18 layers in ONE
     backward each (accumulate_gram_multi) → top-k eigvecs S_action[L] (ctrliface_core.topk_eigvecs);
     mean-ablation target h̄_L + tr(C_L) (harvest_hbar_trace_multi).
  2. Proxy sweep: for each (k,L), D = ‖a_ablate − a_intact‖ under seeded noise (proxy_deviation),
     S_action vs equal-k random. Provisional L* = argmax_L max_k (D_S − D_rand); k* = smallest k
     reaching 80% of D_S(L*)'s max (saturation). PROVISIONAL — the sealed rollout is the headline.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ctrliface_discovery --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/ctrliface_discovery --budget-min 120 \
      --args "--n-grad-batches 48 --n-proxy-batches 16"
"""
import argparse
import json
import pathlib
import subprocess
import sys

K_GRID = (1, 2, 4, 8, 16, 32, 64, 128)


def _install():
    subprocess.run(["bash", "-lc",
                    "apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y "
                    "git python3-dev build-essential libgl1 libglib2.0-0"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "lerobot[libero,pi] @ git+https://github.com/huggingface/lerobot.git"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--n-grad-batches", type=int, default=48)
    ap.add_argument("--n-proxy-batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=128)
    ap.add_argument("--proxy-k", type=int, default=16, help="single k for the cheap L-selection proxy")
    ap.add_argument("--max-episodes", type=int, default=60, help="demo-episode cap (fast partial download)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    _install()
    import os
    os.environ["TORCHDYNAMO_DISABLE"] = "1"   # eager forward; skip lerobot's compiled-matmul triton autotune
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from infra.pg import ctrliface_lerobot as LR
    from infra.pg import ctrliface_run as CR
    from infra.pg import ctrliface_core as CC

    dataset = a.dataset or LR.DATASET_REPO
    rep = {"dataset": dataset, "suites": list(LR.DISCOVERY_SUITES), "k_grid": list(K_GRID),
           "seed": a.seed, "n_grad_batches": a.n_grad_batches, "n_proxy_batches": a.n_proxy_batches}

    policy = LR.load_policy()
    backbone, layers, d = LR.get_backbone(policy)
    # NOTE: do NOT freeze params. PI05Policy.forward's flow-matching loss only requires grad when
    # some param does (freezing → loss.requires_grad=False; the hook's requires_grad injection does
    # not propagate through the policy's forward the way it does for a plain nn.Module). The graph
    # still connects h_L → loss (the "frozen-ish" backbone is in the forward graph), so h_L.grad is
    # well-defined. Needs a ≥48 GB GPU for the backward (param grads). torch-compile disabled below.
    meta = LR.dataset_meta(dataset)
    dt = LR.build_delta_timestamps(policy, meta)
    # Firewall note (post-freeze substrate): HuggingFaceVLA/libero is a COMBINED dataset whose
    # per-episode `tasks` column holds natural-language task descriptions, NOT suite names (Gate-0c),
    # so a demo-level spatial/object filter is not available. S_action is therefore estimated on ALL
    # demos (a controller-wide subspace; the Gram sees demo ACTIONS, not rollout success → no
    # test-leakage). The discovery/confirmation firewall is enforced at the ROLLOUT-SUITE level:
    # controls calibrate thresholds on the spatial+object ENV rollouts; confirmation is sealed on
    # goal+long ENV rollouts. The frozen decision rules are unchanged.
    loader, n_ep, filtered = LR.build_loader(dataset, dt, suites=None,
                                             batch_size=a.batch_size, seed=a.seed,
                                             max_episodes=a.max_episodes)
    rep.update({"d": d, "n_episodes_used": n_ep, "demo_suite_filter": "ALL (rollout-level firewall)",
                "max_episodes": a.max_episodes, "delta_timestamps": dt})
    (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    # The policy PROCESSOR (make_pre_post_processors) tokenizes `task`→observation.language.tokens
    # + normalizes + moves to device — REQUIRED before policy.forward (Gate-0b KeyError lesson).
    from infra.pg import ctrliface_rollout_lerobot as RL
    pre, _post = RL.load_processors(policy, dataset_stats=meta.stats)

    def _take_pre(loader, n):
        out, it = [], iter(loader)
        for _ in range(n):
            out.append(pre(next(it)))               # pre → normalized+tokenized batch on device
        return out

    grad_batches = _take_pre(loader, a.n_grad_batches)
    proxy_batches = _take_pre(loader, a.n_proxy_batches)

    from infra.pg import ctrliface_pi05_hook as PH
    loss_fn = LR.loss_fn_factory(policy)
    action_fn = LR.action_fn_factory(policy)
    NL = len(layers)

    # --- Stage 1: Gram (all layers, one backward each, via the compute_layer_complete monkeypatch)
    #     + h̄/trace. The forward-hook path does NOT work for π0.5's fused backbone+expert forward. ---
    G, npos = PH.capture_gram(policy, loss_fn, grad_batches, d, n_prefix=None, n_layers=NL)
    hbar_trace, n_h = PH.harvest_hbar_trace(policy, loss_fn, grad_batches, d, n_prefix=None, n_layers=NL)
    rep["gram_positions"] = npos
    bases, spectra = {}, {}
    for L in range(NL):
        Q, evals = CC.topk_eigvecs(G[L].cpu(), a.k_max)     # eigh on CPU (robust; frees GPU)
        bases[L] = Q.cpu()                                  # (d, k_max) fp64
        spectra[str(L)] = [float(x) for x in evals[:min(32, a.k_max)]]
    torch.save({"bases": bases,
                "hbar": {L: hbar_trace[L][0].cpu() for L in range(NL)},
                "trace": {L: hbar_trace[L][1] for L in range(NL)},
                "d": d, "n_prefix": None}, out / "bases.pt")
    rep["eigval_spectra_top32"] = spectra
    (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    # --- Stage 2: MINIMAL proxy — single k across all 18 L (cheap L-selection) via ablate_residual.
    # The full K_GRID×L proxy is ~2300 predict_action_chunk calls (too slow); the sealed confirmation
    # k-sweep is the real k* determination (P3 fallback). This single-k L-sweep also DIAGNOSES whether
    # the ablation reaches inference: D_saction≈D_random≈0 ⇒ the monkeypatch does not fire in the
    # inference forward path (→ add forward-hook rollout ablation). k* provisional = proxy_k. ---
    pk = min(a.proxy_k, d)
    grid = {}
    for L in range(NL):
        hbar = hbar_trace[L][0].to("cuda")
        Qs = bases[L][:, :pk].to("cuda")
        Ds, _ = PH.proxy_deviation(policy, action_fn, proxy_batches, L, Qs, hbar, seed=a.seed, n_prefix=None)
        Qr = CC.random_basis(d, pk, a.seed * 991 + L * 17, "cuda", torch.float64)
        Dr, _ = PH.proxy_deviation(policy, action_fn, proxy_batches, L, Qr, hbar, seed=a.seed, n_prefix=None)
        grid[str(L)] = {"D_saction": Ds, "D_random": Dr, "gap": Ds - Dr}
    rep["proxy_grid_at_k"] = {"k": pk, "by_L": grid}
    L_star = max(range(NL), key=lambda L: grid[str(L)]["gap"])
    rep["provisional"] = {"L_star": L_star, "k_star": pk, "peak_gap": grid[str(L_star)]["gap"],
                          "D_saction_Lstar": grid[str(L_star)]["D_saction"],
                          "D_random_Lstar": grid[str(L_star)]["D_random"],
                          "note": "MINIMAL proxy (single k); sealed rollout k-sweep is the headline; "
                                  "D≈0 ⇒ ablation not reaching inference (needs forward-hook path)"}
    (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    print("DISCOVERY_OK", json.dumps({"L_star": L_star, "k_star": pk,
                                      "gram_positions": npos, "n_episodes": n_ep,
                                      "peak_gap": rep["provisional"]["peak_gap"]}, default=str))


if __name__ == "__main__":
    main()
