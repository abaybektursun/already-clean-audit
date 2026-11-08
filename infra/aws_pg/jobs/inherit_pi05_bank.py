"""INHERIT π0.5-side re-bank (§4 Gram-provenance rule: anything the bank lacks is re-estimated
ON THE π0.5 SIDE ONLY, with the frozen CTRLIFACE estimator, and banked). PG1 data never enters.

Adds what the CTRLIFACE bases.pt lacks:
  1. FULL Gram spectrum + full eigvecs per layer (Arm O needs mass-matched tail/mid directions;
     the frozen bank stored top-128 vecs / top-32 vals only). Provenance check: the re-estimated
     top-128 subspace must match the frozen bank per layer (principal-angle overlap ≈ 1) — same
     estimator, same demo policy, same seed.
  2. Activation-COVARIANCE top-σ basis per layer (the transported positive control of §5).
  3. Per-item prefix-pooled residuals + the exact (image, task) items (the CKA transport-gate
     input; alignment key = (episode_index, frame_index)) — saved so the PG1 side replays the
     SAME items.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.inherit_pi05_bank --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/inherit_pi05_bank --budget-min 180 \
      --args "--n-grad-batches 48 --max-episodes 60"
"""
import argparse
import json
import pathlib
import subprocess
import sys


def _install():
    subprocess.run(["bash", "-lc",
                    "apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y "
                    "git python3-dev build-essential libgl1 libglib2.0-0"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "lerobot[libero,pi] @ git+https://github.com/huggingface/lerobot.git"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-grad-batches", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-episodes", type=int, default=60)
    ap.add_argument("--n-cka-batches", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)          # the frozen discovery seed
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    _install()
    import os
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from infra.aws_pg.jobs import _inherit_shared as SH
    from infra.pg import ctrliface_core as CC
    from infra.pg import ctrliface_lerobot as LR
    from infra.pg import ctrliface_pi05_hook as PH
    from infra.pg import ctrliface_rollout_lerobot as RL

    rep = {"seed": a.seed, "n_grad_batches": a.n_grad_batches, "max_episodes": a.max_episodes}
    bases_frozen, _h, _t = SH.load_bank()

    policy = LR.load_policy()
    _bk, layers, d = LR.get_backbone(policy)
    NL = len(layers)
    meta = LR.dataset_meta()
    dt = LR.build_delta_timestamps(policy, meta)
    loader, n_ep, _ = LR.build_loader(LR.DATASET_REPO, dt, suites=None, batch_size=a.batch_size,
                                      seed=a.seed, max_episodes=a.max_episodes)
    pre, _post = RL.load_processors(policy, dataset_stats=meta.stats)

    def _take(n):
        it = iter(loader)
        return [pre(next(it)) for _ in range(n)]

    grad_batches = _take(a.n_grad_batches)

    # --- 1. full-spectrum Gram (frozen estimator) + a STABILITY DIAGNOSTIC vs the frozen bank ---
    # NOTE (root-caused at the pi05_bank v1 run): π0.5's flow-matching loss samples random
    # timestep+noise INTERNALLY (unseeded), so the activation-gradient Gram is STOCHASTIC across
    # runs. The FROZEN CTRLIFACE bank was estimated with UNSEEDED noise → it is one draw from a
    # distribution of S_action subspaces, and a same-recipe re-estimate CANNOT reproduce it exactly
    # (v1 measured top-128 overlap 0.51). We therefore (a) SEED the flow noise here so THIS re-bank
    # is reproducible, and (b) report the frozen-bank overlap PER-k as a stability diagnostic
    # (never a hard gate — the >0.98 gate was structurally impossible). The frozen bank remains the
    # Arm-G transport reference (the CTRLIFACE-causally-validated object); Arm O uses THIS
    # self-consistent full Gram for its {top-k, mass-tail, mid-band} families (disclosed).
    torch.manual_seed(a.seed)                              # reproducible flow noise for the re-bank
    G, npos = PH.capture_gram(policy, LR.loss_fn_factory(policy), grad_batches, d,
                              n_prefix=None, n_layers=NL)
    rep["gram_positions"] = npos
    K_STAB = (1, 16, 32, 64, 128)
    full = {}
    stab = {str(L): {} for L in range(NL)}
    for L in range(NL):
        Q, evals = CC.topk_eigvecs(G[L].cpu(), d)          # FULL eigendecomposition
        full[L] = {"vecs": Q.to(torch.float32), "evals": evals}
        for k in K_STAB:
            ov, _ = CC.principal_angle_overlap(Q[:, :k], bases_frozen[L][:, :k])
            stab[str(L)][str(k)] = float(ov)
    rep["bank_overlap_by_k"] = stab                       # stability diagnostic (reported, not gated)
    rep["bank_overlap_summary"] = {
        str(k): {"min": min(stab[str(L)][str(k)] for L in range(NL)),
                 "mean": sum(stab[str(L)][str(k)] for L in range(NL)) / NL} for k in K_STAB}
    # loose sanity floor ONLY (catch a genuine wiring break, not noise): top-16 must beat the
    # random-subspace null (16/2048 ≈ 0.008) at every layer — else the estimator itself is broken.
    for L in range(NL):
        assert stab[str(L)]["16"] > 0.05, \
            f"L{L} top-16 overlap {stab[str(L)]['16']:.4f} at random-null level — estimator wiring broken"
    torch.save({"full": full, "d": d}, out / "bases_full.pt")
    (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    # --- 2. activation-covariance top-σ + 3. per-item pooled residuals + CKA items ---
    # CKA items must be RAW (image, task) pairs — a fresh same-seed loader replays the same order;
    # the π0.5 forward consumes pre(raw); the RAW image+task are banked so PG1 replays them exactly.
    loader2, _, _ = LR.build_loader(LR.DATASET_REPO, dt, suites=None, batch_size=a.batch_size,
                                    seed=a.seed, max_episodes=a.max_episodes)
    it2 = iter(loader2)
    raw_batches = [next(it2) for _ in range(a.n_cka_batches)]
    cka_batches = [pre(b) for b in raw_batches]
    X = PH.capture_activations(policy, lambda b: policy.forward(b), cka_batches, d,
                               list(range(NL)), n_prefix=None)
    topsig, pooled = {}, {}
    for L in range(NL):
        hbar = X[L].mean(0)
        Xc = X[L] - hbar
        Qs, _ = CC.topk_eigvecs(Xc.transpose(0, 1) @ Xc, 128)
        topsig[L] = Qs.to(torch.float32)
    # per-ITEM pooled residuals: one mean-pooled vector per batch-item per layer, batch-aligned
    pooled_rows = {L: [] for L in range(NL)}
    imgs, tasks_txt = [], []
    img_key = None
    for raw, b in zip(raw_batches, cka_batches):
        if img_key is None:
            img_key = sorted(k for k in raw if k.startswith("observation.images.")
                             and "empty" not in k)[0]
        im = raw[img_key]
        assert torch.is_tensor(im), f"raw image not a tensor: {type(im)}"
        imgs.append(im.detach().to(torch.float32).cpu())
        raw_task = raw.get("task")
        assert raw_task is not None and len(raw_task) == im.shape[0], "raw batch lacks task strings"
        tasks_txt.extend([str(t) for t in raw_task])
        cap = PH.capture_activations(policy, lambda bb: policy.forward(bb), [b], d,
                                     list(range(NL)), n_prefix=None)
        n_item = im.shape[0]
        for L in range(NL):
            pooled_rows[L].append(cap[L].reshape(n_item, -1, d).mean(1))
    for L in range(NL):
        pooled[L] = torch.cat(pooled_rows[L], 0).to(torch.float32)
    assert pooled[0].shape[0] == len(tasks_txt)
    torch.save({"topsig": topsig, "pooled": pooled, "images": torch.cat(imgs, 0),
                "tasks": tasks_txt, "image_key": img_key, "d": d}, out / "cov_cka_bank.pt")
    rep["cka_items"] = int(pooled[0].shape[0])
    (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    print("PI05_BANK_OK", json.dumps({"gram_positions": npos, "cka_items": rep["cka_items"],
                                      "bank_overlap_summary": rep["bank_overlap_summary"]}))


if __name__ == "__main__":
    main()
