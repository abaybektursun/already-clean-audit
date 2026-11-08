"""HUMBOLDT B2 — derived series statistics (spec §3/B2). CPU only; no new forwards.

Reads B1's per-token e_U recordings from S3 and derives, per (model, layer, site, type, bank_id):
  * autocorrelation of e_U over token index, lags 1..256, and the top spectral-density peaks
  * cross-language paired deltas on the parallel corpus (same sentence id, different language)
  * image-vs-text contrast, and the real-vs-gray-128 paired delta, on VQAv2
  * the position curve over the long document
  * the period response on the repetition suite, against its shuffled control
Every row carries its twin columns (the twin banks are separate rows of the same statistic) and a
bootstrap CI.

  pg run infra.aws_pg.jobs.humboldt_b2 --type r7i.2xlarge --no-weights \
      --s3-prefix results/humboldt_b2 --budget-min 180
"""
import argparse
import json
import pathlib
import subprocess

from infra.pg import humboldt_act as A
from infra.pg import humboldt_core as H
from infra.pg import humboldt_smoke as S

MAX_LAG = 256
N_POSITION_BINS = 32


def _load_b1(dest, b1_s3):
    subprocess.run(["aws", "s3", "sync", b1_s3, str(dest), "--region", "us-east-1"], check=True)
    rows = [json.loads(x) for x in (dest / "atlas.jsonl").read_text().splitlines() if x.strip()]
    series_rows = [r for r in rows if r["statistic"] == "e_U_series"]
    assert series_rows, f"no e_U_series rows in {dest}/atlas.jsonl"
    return rows, series_rows


def _meta(dest, mid, cid):
    z = dict(__import__("numpy").load(dest / f"meta__{mid}__{cid}.npz"))
    j = json.loads((dest / f"meta__{mid}__{cid}.json").read_text())
    return {**z, **j}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--b1-s3", default="s3://pg-lineage-769063704696/humboldt/humboldt_b1")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-id", default="humboldt_b2")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    gates = H.Gates(out)
    S.run_all_offline(gates)

    src = H.cache_dir(out, "b1_in")
    _rows, series_rows = _load_b1(src, a.b1_s3)
    b1_manifest = json.loads((src / "manifest.json").read_text())
    par = b1_manifest["resolved_corpora"]["parallel"]
    langs, ref_lang, par_src = par["langs"], par["langs"][0], par["id"]

    # group the e_U_series rows by the block they reference (one .npy per model/corpus/layer/site).
    # The SITE belongs in the block key: there is one block per (model, corpus, layer, SITE), so a
    # key without it makes the vqav2 real/gray lookup ambiguous three ways.
    blocks = {}
    for r in series_rows:
        ref, col = r["vector_ref"].split("#")
        site = json.loads(r["flags"]).get("site", "site")
        blocks.setdefault((r["model_id"], r["corpus_id"], r["layer"], site, ref),
                          []).append((int(col), r))
    print(f"[B2] {len(series_rows)} series rows in {len(blocks)} blocks", flush=True)

    manifest = {"arm": "B2", "run_id": a.run_id, "commit_sha": H.commit_sha(),
                "b1_source": a.b1_s3, "parallel_source": par_src, "parallel_langs": langs,
                "reference_lang": ref_lang, "max_lag": MAX_LAG, "n_position_bins": N_POSITION_BINS,
                "seed": a.seed, "n_blocks": len(blocks)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    atlas = H.Atlas(out, arm="B2", run_id=a.run_id)      # bound BEFORE any loop (§0.2.4)

    meta_cache = {}
    done = 0
    for (mid, cid, L, site, ref), cols in sorted(blocks.items()):
        arr = np.load(src / ref)        # ref is "vectors/<name>.npy", relative to the B1 out/
        if (mid, cid) not in meta_cache:
            meta_cache[(mid, cid)] = _meta(src, mid, cid)
        m = meta_cache[(mid, cid)]
        tclass = np.asarray(m["token_class"], dtype=object)
        item = np.asarray(m["item_id"], dtype=object)
        pos = np.asarray(m["position"])
        # One block per (model, corpus, layer, site) rather than one file per bank: the naive
        # layout would put ~30 000 two-kilobyte .npy files on S3 for the ACFs alone, and syncing
        # them costs more than computing them.
        acf_block = np.stack([H.autocorr(arr[:, c].astype(np.float64), MAX_LAG)
                              for c, _ in sorted(cols)], axis=1)
        acf_ref = atlas.vector(f"B2__acf__{mid}__{cid}__L{L}__{site}", acf_block)
        pos_block = None
        if cid == "longdoc":
            bins = np.array_split(np.argsort(pos), N_POSITION_BINS)
            pos_block = np.stack([[float(arr[b, c].mean()) for b in bins]
                                  for c, _ in sorted(cols)], axis=1)
            pos_ref = atlas.vector(f"B2__poscurve__{mid}__L{L}__{site}", pos_block)
        for j, (col, r) in enumerate(sorted(cols)):
            x = arr[:, col].astype(np.float64)
            base = dict(layer=L, type=r["type"], bank_id=r["bank_id"], model_id=mid,
                        model_sha=r["model_sha"], corpus_id=cid, flags=json.loads(r["flags"]))
            # --- series shape (every corpus).
            atlas.add("e_U_autocorr", vector_ref=f"{acf_ref}#{j}", n=MAX_LAG, seed=a.seed, **base)
            pk = H.spectral_peaks(x, top=5)
            if pk["period"]:
                atlas.add("e_U_spectral_top_period", value=pk["period"][0], seed=a.seed,
                          **{**base, "flags": {**base["flags"], "top_power": pk["power"][0],
                                               "peaks": pk["period"]}})
            # --- per-corpus contrasts.
            if cid == "parallel":
                eng = {}
                for iid, cl, v in zip(item, tclass, x):
                    if cl == ref_lang:
                        eng.setdefault(iid, []).append(v)
                eng_mean = {k: float(np.mean(v)) for k, v in eng.items()}
                for lang in langs:
                    if lang == ref_lang:
                        continue
                    per = {}
                    for iid, cl, v in zip(item, tclass, x):
                        if cl == lang:
                            per.setdefault(iid, []).append(v)
                    ids = sorted(set(per) & set(eng_mean))
                    assert ids, f"no shared parallel-corpus ids for {lang}"
                    d = [float(np.mean(per[i])) - eng_mean[i] for i in ids]
                    atlas.add_ci("parallel_paired_delta_vs_reference", d, token_class=lang,
                                 seed=a.seed,
                                 **{**base, "flags": {**base["flags"], "reference": ref_lang,
                                                      "source_dataset": par_src,
                                                      "n_sentences": len(ids)}})
            if cid.startswith("vqav2"):
                img = x[tclass == "image_patch"]
                txt = x[tclass == "text"]
                if img.size and txt.size:
                    atlas.add("vqav2_image_minus_text", value=float(img.mean() - txt.mean()),
                              n=int(img.size + txt.size), seed=a.seed,
                              **{**base, "flags": {**base["flags"], "n_image": int(img.size),
                                                   "n_text": int(txt.size)}})
            if cid == "longdoc":
                atlas.add("longdoc_position_curve", vector_ref=f"{pos_ref}#{j}",
                          n=N_POSITION_BINS, seed=a.seed,
                          **{**base, "flags": {**base["flags"], "n_bins": N_POSITION_BINS,
                                               "max_position": int(pos.max())}})
            if cid == "repetition":
                arms = ([(f"period{p}", p) for p in A.REPETITION_PERIODS]
                        + [("monotone_numeric", None), ("monotone_length", None)])
                for cls_name, period in arms:
                    sel = x[tclass == cls_name]
                    ctl = x[tclass == f"shuffled_{cls_name}"]
                    if sel.size and ctl.size:
                        atlas.add("repetition_period_response",
                                  value=float(sel.mean() - ctl.mean()), n=int(sel.size),
                                  token_class=cls_name, seed=a.seed,
                                  **{**base, "flags": {**base["flags"], "period": period,
                                                       "control": f"shuffled_{cls_name}",
                                                       "mean_structured": float(sel.mean()),
                                                       "mean_shuffled": float(ctl.mean())}})
        done += 1
        if done % 25 == 0:
            print(f"[B2] {done}/{len(blocks)} blocks ({atlas.n_rows} rows)", flush=True)
            H.s3_sync(out, f"humboldt/{a.run_id}")

    # --- the real-vs-gray paired delta needs BOTH vqav2 blocks, so it runs after the block pass.
    pairs = 0
    for (mid, cid, L, site, ref), cols in sorted(blocks.items()):
        if cid != "vqav2_real":
            continue
        gkey = [k for k in blocks
                if k[0] == mid and k[1] == "vqav2_gray" and k[2] == L and k[3] == site]
        assert len(gkey) == 1, \
            f"expected exactly one vqav2_gray block for {mid} L{L} site={site}, found {len(gkey)}"
        greal = np.load(src / ref)
        ggray = np.load(src / gkey[0][4])
        mr, mg = meta_cache[(mid, "vqav2_real")], meta_cache[(mid, "vqav2_gray")]
        ir = np.asarray(mr["item_id"], dtype=object)
        ig = np.asarray(mg["item_id"], dtype=object)
        cr = np.asarray(mr["token_class"], dtype=object)
        cg = np.asarray(mg["token_class"], dtype=object)
        for col, r in sorted(cols):
            for cls in ("text", "image_patch"):
                per_r, per_g = {}, {}
                for iid, cl, v in zip(ir, cr, greal[:, col]):
                    if cl == cls:
                        per_r.setdefault(iid, []).append(float(v))
                for iid, cl, v in zip(ig, cg, ggray[:, col]):
                    if cl == cls:
                        per_g.setdefault(iid, []).append(float(v))
                ids = sorted(set(per_r) & set(per_g))
                if not ids:
                    continue
                d = [float(np.mean(per_r[i]) - np.mean(per_g[i])) for i in ids]
                atlas.add_ci("vqav2_real_minus_gray", d, layer=L, type=r["type"],
                             bank_id=r["bank_id"], model_id=mid, model_sha=r["model_sha"],
                             corpus_id="vqav2_paired", token_class=cls, seed=a.seed,
                             flags={**json.loads(r["flags"]), "n_items": len(ids),
                                    "pairing": "question_id"})
        pairs += 1

    atlas.close()
    rep = atlas.to_parquet()
    (out / "b2_summary.json").write_text(json.dumps(
        {"manifest": manifest, "atlas": rep, "gates": gates.gates, "n_paired_blocks": pairs},
        indent=2, default=str))
    H.s3_sync(out, f"humboldt/{a.run_id}")
    print("=" * 78)
    print(f"HUMBOLDT_B2_OK rows={rep['n_rows']} blocks={len(blocks)} paired={pairs}")


if __name__ == "__main__":
    main()
