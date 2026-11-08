"""HUMBOLDT B1 — the core MRI / corpus atlas (spec §3/B1). GPU, forwards only, no training.

Five models × six corpora. Per token, per layer, per bank in the battery: e_U = ‖UUᵀh‖²/‖h‖², read
at the bank's own site (journal C2). Plus token metadata (position, token id, language / modality /
period label, item id) so every series is joinable.

Gates run here: G3 (activation projector reproduces the FUNC +49-ppl anchor, recomputed from its
artifact), G4 (twin plumbing on the activation side), G5-offline, base_ppl anchor (±2%), and the
paired-condition token-alignment hash.

  pg run infra.aws_pg.jobs.humboldt_b1 --type p5.4xlarge,g6e.8xlarge --no-weights \
      --s3-prefix results/humboldt_b1 --budget-min 300
"""
import argparse
import json
import pathlib

from infra.pg import humboldt_act as A
from infra.pg import humboldt_core as H
from infra.pg import humboldt_smoke as S
from infra.pg import humboldt_weights as W

# `pg run` packages ONLY infra/, so the repo copy of the FUNC artifact is not on the instance. Read
# the canonical one from S3 — which is the run's own output and the better provenance source anyway
# (the repo copy is a convenience mirror). The repo path stays as a local fallback.
FUNC_ARTIFACT_S3 = "s3://pg-lineage-769063704696/results/invcore_func/out/invariant_core_func.json"
FUNC_ARTIFACT_REPO = "studies/lineage/paligemma1-pi0/results/invcore_func/invariant_core_func.json"
G3_REL_TOL = 0.10          # §1 kill-table row 3
BASE_PPL_REL_TOL = 0.02    # §3/Phase B base_ppl anchor


def _func_anchor(out):
    """Recompute the G3 anchors FROM THE FUNC ARTIFACT (never from prose, §0.12). Returns the
    k=128 all-layer q/k ablation Δppl and the clean blank-image base ppl."""
    import subprocess
    local = pathlib.Path(out) / "invariant_core_func.json"
    repo = pathlib.Path(__file__).resolve().parents[3] / FUNC_ARTIFACT_REPO
    if repo.exists():
        src, text = FUNC_ARTIFACT_REPO, repo.read_text()
    else:
        subprocess.run(["aws", "s3", "cp", FUNC_ARTIFACT_S3, str(local), "--region", "us-east-1"],
                       check=True)
        src, text = FUNC_ARTIFACT_S3, local.read_text()
    local.write_text(text)                       # the anchor's source rides along in out/
    d = json.loads(text)
    vb = d["verdict_basis"]
    # The ppl protocol is DERIVED from the artifact, never assumed: n_windows is recorded on the
    # paired bootstrap, and the window length follows from the token count (each window loses one
    # token to the next-token shift). Both are asserted, so a protocol change fails loud here
    # instead of silently making G3 compare two different measurements.
    n_win = int(vb["ci_unt_minus_rand"]["n_windows"])
    n_tok = int(d["capability_baseline"]["ppl_n_tokens"])
    window = round(n_tok / n_win) + 1
    assert n_win > 0 and window in (256, 512, 1024, 2048), \
        f"unexpected FUNC ppl protocol: {n_win} windows, {n_tok} tokens → window {window}"
    return {"k": int(vb["k"]), "dppl_unt": float(vb["dppl_unt"]),
            "base_ppl": float(d["capability_baseline"]["ppl"]),
            "n_ppl_seq": n_win, "ppl_window": window, "n_tokens": n_tok,
            "artifact": src, "derivation": "n_windows + ppl_n_tokens read from artifact"}


def _g3(out, gates, model, processor, layers, bank_dir, device, anchor, smoke=False):
    """G3 — identify the activation projector with the weight-space object. Rebuild the FUNC band
    (bottom-128 eigvecs of the q_proj consensus Gram) FROM THE SHIPPED A1 EIGENBASIS, ablate it at
    the q_proj/k_proj inputs of every layer exactly as FUNC did, and require the Δppl to reproduce
    the recomputed anchor within ±10% relative."""
    import numpy as np
    import torch
    from infra.aws_pg.jobs.invariant_core_func import _InputProject, _ppl_blankimg

    k = anchor["k"]
    Q = {}
    for L in range(len(layers)):
        V = np.load(pathlib.Path(bank_dir) / f"q_proj__L{L}__gramV_bottom{k}.npy")
        assert V.shape == (2048, k), f"G3 band L{L} shape {V.shape}"
        Q[L] = V
    n_seq = 8 if smoke else anchor["n_ppl_seq"]          # the protocol comes from the artifact
    # In smoke mode the ppl is measured on 8 windows instead of the anchor's 120, so the ±10%
    # comparison would fail for a sampling reason rather than an instrument reason. Record the
    # numbers, do not gate on them: a smoke proves the PLUMBING, and gating it here would make the
    # smoke fail for the one reason it is not testing.
    gate = gates.record if smoke else gates.require
    clean = _ppl_blankimg(model, processor, device, n_seq=n_seq, window=anchor["ppl_window"])
    d_base = abs(clean["ppl"] - anchor["base_ppl"]) / anchor["base_ppl"]
    gate("G3a_base_ppl_anchor", d_base <= BASE_PPL_REL_TOL, measured_ppl=clean["ppl"],
         anchor_ppl=anchor["base_ppl"], rel_delta=d_base, tol=BASE_PPL_REL_TOL,
         n_tokens=clean["n_tokens"], n_windows=n_seq, smoke=smoke, source=anchor["artifact"])
    with _InputProject(layers, Q, device, torch.bfloat16):
        abl = _ppl_blankimg(model, processor, device, n_seq=n_seq, window=anchor["ppl_window"])
    dppl = abl["ppl"] - clean["ppl"]
    rel = abs(dppl - anchor["dppl_unt"]) / abs(anchor["dppl_unt"])
    gate("G3_activation_projector_identification", rel <= G3_REL_TOL,
         measured_dppl=dppl, anchor_dppl=anchor["dppl_unt"], rel_delta=rel, tol=G3_REL_TOL, k=k,
         clean_ppl=clean["ppl"], ablated_ppl=abl["ppl"], n_windows=n_seq, smoke=smoke,
         source=anchor["artifact"],
         criterion="weight-space core ablated through the activation-side projector reproduces "
                   "the FUNC anchor")
    return {"clean_ppl": clean["ppl"], "ablated_ppl": abl["ppl"], "dppl": dppl, "rel": rel}


def _forward_rows(model, proc, rows, is_mm, device, image_token_id, gray=128):
    """Forward every row with the projector armed. Returns token metadata aligned to the series."""
    import numpy as np
    import torch
    meta = {"token_id": [], "position": [], "window_idx": [], "token_class": [], "item_id": [],
            "align_hash": {}}
    blank = None
    if is_mm:
        from infra.aws_pg.jobs.invariant_core_func import _blank_image
        blank = _blank_image(gray)
    for wi, r in enumerate(rows):
        if is_mm:
            img = r.get("image", blank)
            inp = proc(text=[r["text"]], images=[img], return_tensors="pt", padding=False).to(device)
            kw = {"pixel_values": inp["pixel_values"].to(torch.bfloat16)}
        else:
            inp = proc(r["text"], return_tensors="pt", add_special_tokens=True).to(device)
            kw = {}
        ids = inp["input_ids"][0]
        with torch.inference_mode():
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], **kw)
        idc = ids.cpu().numpy()
        if is_mm:
            n_img = int((idc == image_token_id).sum())
            assert n_img > 0, (f"no image tokens found at id {image_token_id} — the config field "
                               f"holding the placeholder id has moved; every image token would be "
                               f"mislabelled as text")
            cls = np.where(idc == image_token_id, "image_patch", r["token_class"])
        else:
            cls = np.full(idc.shape, r["token_class"], dtype=object)
        meta["token_id"] += idc.tolist()
        meta["position"] += list(range(len(idc)))
        meta["window_idx"] += [wi] * len(idc)
        meta["token_class"] += [str(c) for c in cls]
        meta["item_id"] += [r["item_id"]] * len(idc)
        meta["align_hash"][r["item_id"]] = A.alignment_hash(idc.tolist())
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(m for m, _, _ in A.ROSTER))
    ap.add_argument("--layers", default="")
    ap.add_argument("--n-tokens", type=int, default=50_000)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--long-doc-tokens", type=int, default=4096)
    ap.add_argument("--n-vqa", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-id", default="humboldt_b1")
    ap.add_argument("--probe-s3", default="s3://pg-lineage-769063704696/humboldt/humboldt_probe/corpus_resolution.json")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    want_models = [m.strip() for m in a.models.split(",") if m.strip()]
    if a.smoke:
        want_models = want_models[:2]
        a.n_tokens, a.n_vqa, a.long_doc_tokens = 3000, 4, 1024

    H.hf_setup()                       # FIRST — XET off + gated-repo auth (§0.2.7)
    import subprocess
    import numpy as np
    import torch
    assert torch.cuda.is_available(), "GPU required for Phase B forwards"
    device = "cuda"

    gates = H.Gates(out)
    S.run_all_offline(gates)

    subprocess.run(["aws", "s3", "cp", a.probe_s3, str(out / "corpus_resolution.json"),
                    "--region", "us-east-1"], check=True)
    resolution = json.loads((out / "corpus_resolution.json").read_text())
    resolved = {k: v["chosen"] for k, v in resolution["slots"].items()}
    bank_dir = W.fetch_banks(H.cache_dir(out, "banks"))
    anchor = _func_anchor(out)

    manifest = {"arm": "B1", "run_id": a.run_id, "commit_sha": H.commit_sha(),
                "models": want_models, "resolved_corpora": resolved, "seed": a.seed,
                "n_tokens_per_corpus": a.n_tokens, "window": a.window,
                "long_doc_tokens": a.long_doc_tokens, "n_vqa_items": a.n_vqa,
                "func_anchor": anchor, "sites": list(H.SITES_PHASE_B),
                "corpus_probe_run": resolution["run_id"],
                "gpu": torch.cuda.get_device_name(0)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    atlas = H.Atlas(out, arm="B1", run_id=a.run_id)      # bound BEFORE any loop (§0.2.4)
    coverage = []                                        # bound BEFORE any loop (§0.2.4)
    g4_act = {}                                          # bound BEFORE any loop (§0.2.4)

    for mid, repo, kind in A.ROSTER:
        if mid not in want_models:
            continue
        print(f"[B1] === {mid} ({repo}, {kind}) ===", flush=True)
        # §9 pre-mortem pivot, verbatim: "any single-model failure in B1 drops that model with a
        # journaled coverage-table hole, never the whole arm". The pivot covers model CONSTRUCTION
        # (a re-shell key mismatch, an unavailable checkpoint); everything after it is fail-loud,
        # because a broken recording must not be written as if it were data.
        try:
            model, proc, layers, sha, is_mm = A.load_model(mid, repo, kind, device=device)
        except Exception as e:      # noqa: BLE001 — see the §9 pivot above
            coverage.append({"model": mid, "corpus": None, "status": "DROPPED",
                             "reason": f"{type(e).__name__}: {str(e)[:300]}"})
            (out / "coverage_b1.json").write_text(json.dumps(coverage, indent=2))
            print(f"[B1][DROP] {mid}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            torch.cuda.empty_cache()
            continue
        tok = proc.tokenizer if is_mm else proc
        image_token_id = -1
        if is_mm:
            tid = getattr(model.config, "image_token_index", None)
            if tid is None:
                tid = getattr(model.config, "image_token_id", None)
            assert tid is not None, "PaliGemma config exposes no image-token id under either name"
            image_token_id = int(tid)
        energy = A.SiteEnergy(layers, bank_dir, device)

        if mid == "pg1-pt224":
            g3 = _g3(out, gates, model, proc, layers, bank_dir, device, anchor, smoke=a.smoke)
            (out / "g3_detail.json").write_text(json.dumps(g3, indent=2))

        corpora = A.build_text_corpora(tok, resolved, n_tokens=a.n_tokens, window=a.window,
                                       long_doc_tokens=a.long_doc_tokens, seed=a.seed)
        if is_mm:
            items = A.build_vqav2_items(resolved, n_items=a.n_vqa, seed=a.seed)
            from infra.aws_pg.jobs.invariant_core_func import _blank_image
            corpora["vqav2_real"] = [{"corpus_id": "vqav2_real", "item_id": it["item_id"],
                                      "token_class": "text", "text": it["question"],
                                      "image": it["image"].convert("RGB")} for it in items]
            corpora["vqav2_gray"] = [{"corpus_id": "vqav2_gray", "item_id": it["item_id"],
                                      "token_class": "text", "text": it["question"],
                                      "image": _blank_image(128)} for it in items]

        for cid, rows in corpora.items():
            with energy:
                meta = _forward_rows(model, proc, rows, is_mm, device, image_token_id)
            series = energy.collect()
            T = len(meta["token_id"])
            np.savez(out / f"meta__{mid}__{cid}.npz",
                     token_id=np.asarray(meta["token_id"], dtype=np.int64),
                     position=np.asarray(meta["position"], dtype=np.int32),
                     window_idx=np.asarray(meta["window_idx"], dtype=np.int32))
            (out / f"meta__{mid}__{cid}.json").write_text(json.dumps(
                {"token_class": meta["token_class"], "item_id": meta["item_id"],
                 "align_hash": meta["align_hash"], "n_tokens": T,
                 "content_hash": A.content_hash([r["text"][:64] for r in rows])}))
            # Hoisted out of the bank loop: the class list and its boolean masks are the SAME for
            # every one of the ~750 (layer, bank) columns. Recomputing set() over 75k strings per
            # column costs more than the forwards did.
            tclass = np.asarray(meta["token_class"], dtype=object)
            classes = sorted(set(meta["token_class"]))
            masks = {cl: (tclass == cl) for cl in classes}
            masks = {cl: m2 for cl, m2 in masks.items() if m2.any()}
            for (L, site), arr in series.items():
                assert arr.shape[0] == T, f"{mid}/{cid} L{L} {site}: {arr.shape[0]} rows vs {T} tokens"
                ref = atlas.vector(f"B1__eU__{mid}__{cid}__L{L}__{site}", arr, dtype="float32")
                for col, (t, bid) in enumerate(energy.order[site]):
                    # §2.3's alignment key: a per-token row must be joinable across models. The
                    # series is a column of a block; the token axis is described by the metadata
                    # files, named EXPLICITLY here rather than left to a naming convention.
                    fl = {**H.type_flags(t), "model_kind": kind, "corpus": cid, "col": col,
                          "meta_ref": f"meta__{mid}__{cid}.npz",
                          "meta_labels_ref": f"meta__{mid}__{cid}.json",
                          "align_key": "(model_sha, layer, type, bank_id, corpus_id, token_index)"}
                    atlas.add("e_U_series", vector_ref=f"{ref}#{col}", layer=L, type=t,
                              bank_id=bid, model_id=mid, model_sha=sha, n=int(T), corpus_id=cid,
                              flags=fl)
                    atlas.add_ci("e_U_mean", arr[:, col], layer=L, type=t, bank_id=bid,
                                 model_id=mid, model_sha=sha, corpus_id=cid, seed=a.seed,
                                 flags={**fl, "chance": H.BANK_K / H.IN_DIM[t]})
                    for cl, sel in masks.items():
                        atlas.add_ci("e_U_mean_by_class", arr[sel, col], layer=L, type=t,
                                     bank_id=bid, model_id=mid, model_sha=sha, corpus_id=cid,
                                     token_class=cl, seed=a.seed,
                                     flags={**fl, "chance": H.BANK_K / H.IN_DIM[t]})
            if cid == "wikitext":
                g4_act[mid] = {
                    "core32": [float(series[(L, "q_proj_in")]
                                     [:, energy.order["q_proj_in"].index(("q_proj", "core32"))].mean())
                               for L in range(len(layers))],
                    "haar32_s0": [float(series[(L, "q_proj_in")]
                                        [:, energy.order["q_proj_in"].index(("q_proj", "haar32_s0"))].mean())
                                  for L in range(len(layers))]}
            # §8 storage row: the budget line estimated ~200 MB for the whole atlas at ~12 banks;
            # the site fix (C2) makes it 42 banks per layer, so the real figure is recorded per cell
            # rather than assumed. Nothing is capped — this is accounting, not truncation.
            n_bytes = sum(int(v.nbytes) for v in series.values())
            coverage.append({"model": mid, "corpus": cid, "n_tokens": T,
                             "n_series": len(series), "n_bank_columns": sum(
                                 energy.n_banks(s2) for s2 in H.SITES_PHASE_B) * len(layers),
                             "series_bytes_fp32": n_bytes, "status": "recorded"})
            (out / "coverage_b1.json").write_text(json.dumps(coverage, indent=2))
            print(f"[B1] {mid}/{cid}: {T} tokens, {len(series)} series ({atlas.n_rows} rows)",
                  flush=True)
            H.s3_sync(out, f"humboldt/{a.run_id}")

        # paired-condition token alignment (§1 kill-table, last row): real vs gray must be
        # token-IDENTICAL per item; FLORES languages must cover the same sentence-id set.
        if is_mm:
            r_meta = json.loads((out / f"meta__{mid}__vqav2_real.json").read_text())["align_hash"]
            g_meta = json.loads((out / f"meta__{mid}__vqav2_gray.json").read_text())["align_hash"]
            bad = [k for k in r_meta if r_meta[k] != g_meta.get(k)]
            gates.require(f"G_corpus_alignment_vqav2__{mid}", not bad, n_items=len(r_meta),
                          n_mismatched=len(bad), mismatched=bad[:5],
                          criterion="real and gray-128 twins are token-identical per item")
        # the parallel corpus is named `parallel`, not `flores` — FLORES-200 is gated on the Hub
        # and the slot resolved to a different source (journal, corpus probe entry).
        pmeta_path = out / f"meta__{mid}__parallel.json"
        assert pmeta_path.exists(), f"parallel-corpus metadata missing at {pmeta_path}"
        f_meta = json.loads(pmeta_path.read_text())
        by_lang = {}
        for iid, cl in zip(f_meta["item_id"], f_meta["token_class"]):
            by_lang.setdefault(cl, set()).add(iid)
        sets = list(by_lang.values())
        gates.require(f"G_corpus_alignment_parallel__{mid}",
                      bool(sets) and all(s == sets[0] for s in sets), n_langs=len(sets),
                      n_ids=len(sets[0]) if sets else 0,
                      source=resolved["parallel"]["id"],
                      criterion="every language of the parallel corpus covers the same sentence-id "
                                "set")

        # G4 on the activation side: the real bank's e_U must separate from its Haar twin.
        # Gate scoped to pg1-pt224 — the model the object is DEFINED on. On any other model the same
        # comparison is a scientific reading, not a plumbing check, so it is recorded, not gated.
        if mid in g4_act:
            r = H.mean_ci(g4_act[mid]["core32"], seed=a.seed)
            h = H.mean_ci(g4_act[mid]["haar32_s0"], seed=a.seed)
            ok = r["ci_lo"] > h["ci_hi"]
            ev = dict(model=mid, core32_e_U=r, haar32_s0_e_U=h, n_layers=len(layers),
                      haar_expectation=H.BANK_K / 2048, per_layer=g4_act[mid],
                      criterion="q_proj core32 mean e_U CI strictly above the haar32_s0 CI "
                                "(wikitext, all layers)")
            if mid == "pg1-pt224":
                gates.require("G4_twin_plumbing_activation", ok, **ev)
            else:
                gates.record(f"G4_twin_plumbing_activation__{mid}", ok, gate_scope="evidence only",
                             **ev)
        del model, energy
        torch.cuda.empty_cache()

    atlas.close()
    rep = atlas.to_parquet()
    total_bytes = sum(c.get("series_bytes_fp32", 0) for c in coverage)
    (out / "b1_summary.json").write_text(json.dumps(
        {"manifest": manifest, "atlas": rep, "gates": gates.gates, "coverage": coverage,
         "series_bytes_total_fp32": total_bytes,
         "series_gib": round(total_bytes / 2 ** 30, 3),
         "storage_note": ("§8 budgeted ~200 MB assuming ~12 banks; the C2 site fix records 42 banks "
                          "per layer, so the realised size is larger. Recorded, never capped.")},
        indent=2, default=str))
    H.s3_sync(out, f"humboldt/{a.run_id}")
    print("=" * 78)
    print(f"HUMBOLDT_B1_OK rows={rep['n_rows']} vectors={rep['n_vectors']} "
          f"cells={len(coverage)} gates={ {k: v['ok'] for k, v in gates.gates.items()} }")


if __name__ == "__main__":
    main()
