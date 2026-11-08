"""GRAIN offline gate (CPU, --no-weights) — the full pipeline MINUS the model forward, on real
banks + real tokenizer/dataset item streams + synthetic activations (the house offline-gate
convention: GPU minutes only validate what only GPU can).

Exercises end-to-end: read banks load+validate → sites resolve → item halves (grain| salt,
real streamed rows) → micro dictionary training on synthetic block-sparse rows at d=2048 →
quality → block alignment vs the REAL S_action bank → union assembly → support spread →
δ_G-style band arithmetic → adjudication on the resulting numbers → Jaccard machinery →
dict bank round-trip. Every downstream code path that does not need a PG1 forward runs here.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.grain_offline --type r7i.4xlarge --no-weights \
      --s3-prefix results/grain_offline --budget-min 90
"""
import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-items", type=int, default=256, help="real streamed items (split check)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"object": "grain_offline_v1"}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from infra.aws_pg.jobs import _grain_shared as GS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg.inherit_core import rf_item_bootstrap

    # --- real banks + sites ---
    banks = GS.load_read_banks()
    rep["sites"] = {"primary": banks["saction"]["L"], "secondary": banks["svl"]["L"]}
    _w()

    # --- real item stream + grain split (tokenizer/dataset path without the model) ---
    vqa, _coco = IP.pg1_tasks()
    rows, hashes = IP.load_items(vqa, a.n_items)
    split = GS.grain_split(hashes, name="grain_vqa_offline")
    rep["items"] = {"n": len(hashes), "n_disc": int(split["n_disc"]),
                    "n_sealed": int(split["n_conf"])}
    _w()

    # --- micro end-to-end at real d=2048: planted data → train → align vs REAL bank ---
    d = 2048
    X, U_planted = GB.gen_planted(d, n_planted=4, m_true=4, n_samples=8000, k_active=2,
                                  seed=7, noise=0.05)
    ae, cfg = GB.train_block_sae(X, m=4, seed=0, device="cpu", dict_size=256, r_active=8,
                                 steps=1200, lr=1e-3, batch=256, warmup=100)
    q = GB.block_quality(ae, X, "cpu", n_tokens=2000)
    rep["micro_train"] = {"cfg": {k: cfg[k] for k in ("m", "k_blk", "dict_size", "threshold")},
                          "quality": q}
    assert q["fvu"] < 0.9, f"micro training degenerate (FVU {q['fvu']:.3f})"
    _w()

    Wd = ae.decoder.weight.detach()
    align = GB.block_alignment(Wd, 4, banks["saction"]["V"])       # real bank, synthetic dict
    spread = GB.support_spread(align, n_active=ae.n_blocks)
    rep["align_vs_real_bank"] = {"max": float(align.max()), "n_for_90": spread["n_for_90"]}
    Q = GB.union_Q(Wd, 4, [0, 1, 2])
    assert Q.shape == (d, 12)
    _w()

    # --- band + adjudication arithmetic on synthetic per-item scores ---
    g = torch.Generator().manual_seed(3)
    intact = (torch.rand(200, generator=g) < 0.6).float().numpy()
    ablate = (torch.rand(200, generator=g) < 0.4).float().numpy()
    floor = 0.05
    rf = rf_item_bootstrap(ablate, intact, floor)
    label = GB.adjudicate_grain(effect_bstar=1 - rf["rf"], effect_full=0.9, effect_em=0.05,
                                effect_grain1=0.1, effect_rand_mean=0.05,
                                gap_cilo=0.2, delta_G=0.05, rho=0.5, bh_pass=True,
                                support_frac_90=0.1)
    assert label in GB.G_LABELS
    null = GB.jaccard_null(4096, 8, 500, seed=1)
    s2 = GB.adjudicate_s2(jaccard_cilo=0.4, null_p95=float(torch.quantile(null, 0.95)),
                          subspace_canonicity=0.35)
    rep["adjudication_smoke"] = {"g_label": label, "s2_label": s2,
                                 "rf": rf["rf"], "rf_ci": rf["ci"]}
    _w()

    # --- bank round-trip ---
    GB.save_dict_bank(ae, cfg, q, out / "micro_dict.pt", GS.manifest_sha(hashes[:8], cfg))
    ae2, cfg2, _q2, _sha = GB.load_dict_bank(out / "micro_dict.pt")
    x = torch.randn(4, d)
    assert torch.equal(ae2.encode(x, use_threshold=True), ae.encode(x, use_threshold=True))
    rep["bank_roundtrip"] = True
    rep["green"] = True
    _w()
    print("GRAIN_OFFLINE_OK", json.dumps({"sites": rep["sites"],
                                          "micro_fvu": q["fvu"]}, default=str))


if __name__ == "__main__":
    main()
