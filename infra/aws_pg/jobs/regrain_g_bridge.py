"""REGRAIN Arm G (CPU, banks only) — the bridge from fine-tuning's WRITE to the dictionary's atoms.

COMWRITE's anatomy estimand (verified in its own artifact):
    U_wr(L, t; k) = top-k eigvecs of the residual-facing, per-FT-unit-Frobenius consensus Gram,
    output side o_proj / down_proj — on `google/paligemma-3b-pt-224` at base_sha 35e4f464…,
    i.e. EXACTLY REGRAIN's pinned model. The leave-one-out bases `loo__<t>__L<L>__<i>.npy` are
    (2048, 128) fp32 subspaces of the residual stream.

The question: **does the consensus fine-tuning write land in a few nameable dictionary atoms —
more concentrated than the task's own activation footprint?**

    write footprint    e_j = ‖U_wrᵀ d_j‖²      over unit decoder columns d_j
    task footprint     a_j = mean |z_j|        over the site's harvest rows
    concentration      top-8 share of the normalized footprint, atom-bootstrap CI
    null               Haar (2048,128) subspaces → a flat e_j ≈ k/d baseline

SITE. Arm G runs at **L14 only, and this is structural, not a budget choice**: the dictionary at
L0 featurizes the residual ENTERING layer 0, which is the embedding output — no decoder
`down_proj`/`o_proj` write reaches it. Any "damage footprint" at L0 would be identically zero by
construction. Pairing: the L14 dictionary sees writes from layers < 14, so U_wr is read at layer 13
(the most proximate writer) for both output-side matrices.

Labelled EXPLORATORY (spec §10).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_g_bridge --type r7i.4xlarge --no-weights \
      --s3-prefix results/regrain_g --budget-min 90 \
      --args "--harvest-s3 s3://…/regrain_harvest/out"
"""
import argparse
import json
import pathlib

LOO_IDX = (0, 1, 2, 3)          # leave-one-out draws → the write subspace's own spread
TOPK = 8
N_BOOT = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--harvest-s3", default=None,
                    help="defaults to the GRAIN L14 harvest — Arm G needs only the site's usage, "
                         "not the expanded row count, so it does not wait on I1")
    ap.add_argument("--writer-layer", type=int, default=13)
    ap.add_argument("--n-usage-rows", type=int, default=30000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC
    from infra.pg.ctrliface_core import orthonormalize

    rep = {"object": "regrain_g_v1", "label": "EXPLORATORY", "writer_layer": a.writer_layer,
           "topk": TOPK,
           "site_note": ("Arm G runs at L14 only: the L0 dictionary featurizes the residual "
                         "ENTERING layer 0 (the embedding output), which no decoder down_proj/"
                         "o_proj write can reach. A damage footprint there is zero by "
                         "construction, not by measurement.")}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- provenance: COMWRITE's anatomy must be the SAME base model + sha ----
    anat = json.loads(RS.fetch(f"{RS.COMWRITE_ANATOMY}/metrics.json",
                               "/tmp/regrain/anatomy.json").read_text())
    sha, _ = RS.pinned_pg1_sha("/tmp/regrain")
    assert anat["base_sha"] == sha, \
        f"anatomy base_sha {anat['base_sha'][:10]} != REGRAIN pin {sha[:10]} — provenance fork"
    assert anat["base_repo"] == IP.PG1_REPO
    rep["provenance"] = {"anatomy_base_repo": anat["base_repo"], "base_sha": sha,
                         "k_star": anat["k_star"], "k_loo": anat["k_loo"],
                         "estimand": anat["estimand"]}
    _w()

    banks = RS.load_read_banks("/tmp/regrain")
    Ls = banks["svl"]["L"]
    harvest = a.harvest_s3 or RS.GRAIN_HARVEST
    meta = json.loads(RS.fetch(f"{harvest}/harvest_meta.json",
                               "/tmp/regrain/harvest_meta.json").read_text())
    key = f"L{Ls}"
    # `load_shards` (no per-row stream) — Arm G reads usage only, so GRAIN's shards suffice.
    X, labs = RS.load_shards(harvest, meta, key, workdir="/tmp/regrain/shards")
    rep["harvest"] = harvest

    nm = RS.dict_bank_name("secondary", 1, 0)
    ae, _c, _q, _m = GB.load_dict_bank(RS.fetch(f"{RS.GRAIN_DICTS_L14}/{nm}",
                                                f"/tmp/regrain/{nm}"))
    U_dec = RC.unit_decoder(RC.decoder_matrix(ae))                    # (d, W) fp64 unit columns
    usage = GB.block_usage(ae, X[: a.n_usage_rows], labs[: a.n_usage_rows].long(), "cpu")
    live = usage["freq"] > 0
    n_live = int(live.sum())
    rep["dictionary"] = {"site": Ls, "bank": nm, "n_atoms": int(U_dec.shape[1]),
                         "n_live": n_live}
    _w()

    def footprint_concentration(e, mask):
        """top-k share of a non-negative per-atom footprint, restricted to live atoms."""
        v = np.asarray(e, float)[mask]
        v = np.clip(v, 0, None)
        tot = v.sum()
        assert tot > 0, "degenerate footprint"
        top = np.sort(v)[::-1][:TOPK].sum()
        return float(top / tot)

    def boot_share(e, mask, seed=1337):
        v = np.asarray(e, float)[mask]
        v = np.clip(v, 0, None)
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(N_BOOT):
            s = rng.choice(v, size=v.size, replace=True)
            tot = s.sum()
            if tot > 0:
                vals.append(np.sort(s)[::-1][:TOPK].sum() / tot)
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    livem = live.numpy()

    # ---- the TASK footprint ----
    task = usage["latent_mean_act"].numpy()
    rep["task_footprint"] = {"top8_share": footprint_concentration(task, livem),
                             "ci95": boot_share(task, livem)}
    _w()

    # ---- the WRITE footprint, per output-side matrix, across LOO draws ----
    write = {}
    for mtype in ("down_proj", "o_proj"):
        shares, es = [], []
        for i in LOO_IDX:
            fn = f"loo__{mtype}__L{a.writer_layer}__{i}.npy"
            p = RS.fetch(f"{RS.COMWRITE_ANATOMY}/scratch/{fn}", f"/tmp/regrain/{fn}")
            Uw = torch.as_tensor(np.load(p), dtype=torch.float64)
            assert Uw.shape[0] == IP.D, f"{fn} lives in R^{Uw.shape[0]}, not R^{IP.D}"
            Uw = orthonormalize(Uw)
            e = (Uw.transpose(0, 1) @ U_dec).pow(2).sum(0).numpy()     # ‖U_wrᵀ d_j‖², per atom
            es.append(e)
            shares.append(footprint_concentration(e, livem))
        e_mean = np.mean(es, axis=0)
        write[mtype] = {"loo_top8_shares": shares, "top8_share_mean": float(np.mean(shares)),
                        "top8_share_spread": float(np.max(shares) - np.min(shares)),
                        "ci95_meanLOO": boot_share(e_mean, livem),
                        "k": int(Uw.shape[1]),
                        "top8_atoms": [int(i) for i in
                                       np.argsort(np.where(livem, e_mean, -1))[::-1][:TOPK]]}
        rep["write_footprint"] = write
        _w()

    # ---- Haar null: a random k-subspace's footprint is flat at k/d ----
    k = write["down_proj"]["k"]
    haar = []
    for s in range(4):
        g = torch.Generator().manual_seed(31 + s)
        Uh = orthonormalize(torch.randn(IP.D, k, generator=g, dtype=torch.float64))
        eh = (Uh.transpose(0, 1) @ U_dec).pow(2).sum(0).numpy()
        haar.append(footprint_concentration(eh, livem))
    rep["haar_null"] = {"top8_shares": haar, "mean": float(np.mean(haar)),
                        "analytic_flat_e_j": k / IP.D,
                        "note": "a Haar k-subspace projects every unit atom to ≈ k/d energy, so "
                                "its top-8 share is the unconcentrated baseline for n_live atoms"}

    # ---- the comparison Arm G exists to make ----
    t = rep["task_footprint"]["top8_share"]
    w = write["down_proj"]["top8_share_mean"]
    rep["headline"] = {
        "write_more_concentrated_than_task": bool(w > t),
        "write_top8_share_down_proj": w, "task_top8_share": t,
        "haar_top8_share": rep["haar_null"]["mean"],
        "shared_atoms_write_vs_task": sorted(
            set(write["down_proj"]["top8_atoms"]) &
            set(int(i) for i in np.argsort(np.where(livem, task, -1))[::-1][:TOPK])),
        "R11_prediction": "damage footprint MORE concentrated than task footprint"}
    print("REGRAIN_G_OK", json.dumps(rep["headline"], default=str))
    # green written LAST: a crash after this point must not leave a green
    # artifact -- the drivers gate on `green` (regrain_drive*.sh wait_for_green).
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
