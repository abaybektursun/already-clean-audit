"""STAGEDIFF fit — one checkpoint: J-lens fit (+ size sweep), J-lens B1-int + shuffle control, band stats,
TRACE / null activation banks, top fit-corpus tokens (spec §2, W2).

    python -m infra.aws_pg.jobs.stagediff_fit --ckpt C5 --out /work/out --n-seqs 64 --seq-len 128 --dim-batch 32
    python -m infra.aws_pg.jobs.stagediff_fit --ckpt C0 --corpus-half a --n-seqs 64 ...   # 32-sequence half (C0 stability)
    python -m infra.aws_pg.jobs.stagediff_fit --dryrun --out <dir>

Outputs: out/lens.pt (fp16), out/lens_<n>.pt (sweep), out/trace_bank.pt [200,36,5,d] fp16,
out/null_bank.pt [64,36,d] fp16, per_item/B1-int_jlens*.json, metrics.json (fit timing, stats, pass@k).
"""
from __future__ import annotations

import argparse
import os
import sys

MODULE = "infra.aws_pg.jobs.stagediff_fit"


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="C0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seqs", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--dim-batch", type=int, default=32)
    ap.add_argument("--corpus-half", choices=["a", "b"], default=None, help="fit on half a (first 32) or b (last 32) of the 64")
    ap.add_argument("--sizes", default="16,32", help="fit-size sweep checkpoints (prompt counts)")
    ap.add_argument("--trace-seqs", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--stub-seed", type=int, default=0, help="dry-run only: stub model seed")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if not a.dryrun:
        from infra.aws_pg.jobs._stagediff_shared import boot
        if boot(MODULE, a.ckpt, sys.argv[1:]):
            return
    import torch
    from infra.pg import stagediff_bank as SB, stagediff_interv as SV, stagediff_items as SI, stagediff_lens as SL, stagediff_models as SM
    from infra.aws_pg.jobs import _stagediff_shared as SH
    bank = SB.Bank(a.out, "stagediff_fit", vars(a))
    with SH.Timer(bank, "load"):
        member = SM.stub_member(bos_id=1, seed=a.stub_seed) if a.dryrun else SM.load_ckpt(a.ckpt, a.device)
    tok = member["tok"]
    bank.manifest({"ckpt": member["id"], "repo": member["repo"], "repo_sha": member["sha"], "hashes": member["hashes"],
                   "bos_id": member["bos_id"], "dtype": member["dtype"], "attn": member["attn"],
                   "transformers_in_process": member["transformers"], "jlens": "vendored anthropics/jacobian-lens@581d3986",
                   "lens_eval_sha256": SI.FILE_SHA256})
    lm = SL.AnatomyLensModel(member)

    # ---- corpora (hashed) ---------------------------------------------------------------------
    with SH.Timer(bank, "corpora"):
        if a.dryrun:
            docs_train, docs_val = SH.fake_docs(40), SH.fake_docs(30)
            fc = SL.fit_corpus(tok, n_seqs=4, seq_len=48, seed=0, docs=docs_train)
            tc = SL.trace_corpus(tok, n_seqs=6, seq_len=24, seed=0, docs=docs_val)
        else:
            fc = SL.fit_corpus(tok, n_seqs=a.n_seqs, seq_len=a.seq_len, seed=SL.CORPUS_SEED)
            tc = SL.trace_corpus(tok, n_seqs=a.trace_seqs, seq_len=SL.TRACE_LEN, seed=SL.CORPUS_SEED)
    seqs = fc["seqs"]
    if a.corpus_half:
        h = len(seqs) // 2
        seqs = seqs[:h] if a.corpus_half == "a" else seqs[h:]
    if a.smoke:
        seqs, tc["seqs"] = seqs[:2], tc["seqs"][:4]
    bank.put("controls", "fit_corpus", {k: v for k, v in fc.items() if k != "seqs"} | {"half": a.corpus_half, "n_used": len(seqs), "sha256_used": SL.corpus_sha256(seqs)})
    bank.put("controls", "trace_corpus", {k: v for k, v in tc.items() if k != "seqs"} | {"n_used": len(tc["seqs"])})
    sets = SI.load_sets()
    beh, ints = SI.beh_items(tok, sets), SI.int_items(tok, sets)
    texts = [tok.decode(s) for s in fc["seqs"] + tc["seqs"]]
    bank.put("controls", "contamination", SI.contamination([it["prompt"] for it in beh + ints], texts))
    bank.put("controls", "top_tokens", SL.top_tokens(fc["seqs"], 5000))

    # ---- fit -----------------------------------------------------------------------------------
    sizes = tuple(int(x) for x in a.sizes.split(",") if x.strip()) if not (a.smoke or a.dryrun) else (1,)
    with SH.Timer(bank, "fit"):
        lens, info = SL.fit_lens(lm, seqs, dim_batch=a.dim_batch, ckpt_path=os.path.join(a.out, "fit_ckpt.pt"),
                                 sizes=sizes, save_dir=a.out)
    lens.save(os.path.join(a.out, "lens.pt"), dtype=torch.float16)
    os.remove(os.path.join(a.out, "fit_ckpt.pt"))
    st = SL.lens_stats(lens)
    SL.lens_to(lens, lm.device)
    last3 = sorted(st)[-3:]
    diag_ok = all(0.5 <= st[L]["diag_mean"] <= 1.5 for L in last3)
    finite = all(v["finite"] for v in st.values())
    total_s = sum(s["seconds"] for s in info["stages"])
    from jlens.fitting import SKIP_FIRST_N_POSITIONS
    used_idx = fc["doc_window_index"][:len(seqs)] if a.corpus_half in (None, "a") else fc["doc_window_index"][len(fc["seqs"]) - len(seqs):]
    bank.put("readouts", "fit", {**info, "seconds_total": total_s, "seconds_per_prompt": total_s / max(1, len(seqs)),
                                 "n_prompts": lens.n_prompts, "source_layers": lens.source_layers, "stats": st,
                                 "late_diag_ok(0.5-1.5)": diag_ok, "finite": finite,
                                 "skip_first": SKIP_FIRST_N_POSITIONS, "target_layer": lm.n_layers - 1,   # CK-3: logged numbers, not defaults
                                 "seq_len": a.seq_len, "doc_window_index_used": used_idx})
    assert finite, "non-finite J — §14: dim_batch 4, seq 64, 32 sequences"
    if not diag_ok:
        bank.event("FLAG", what="late-layer diag(J) outside 0.5–1.5", layers={L: st[L]["diag_mean"] for L in last3})

    # ---- B1-int under the J-lens + shuffle control ---------------------------------------------
    if a.smoke or a.dryrun:
        ints = SI.smoke_subset(ints, 4)
    band = SM.BAND if not a.dryrun else (1, 2)
    with SH.Timer(bank, "B1-int-jlens"):
        acts = SL.item_acts(lm, ints)
        prof = SL.rank_profile(lm, ints, acts, lens=lens)
        prof_sh = SL.rank_profile(lm, ints, acts, lens=lens, partner=SI.derange_within_set(ints))
    bank.per_item("B1-int_jlens", prof)
    bank.per_item("B1-int_jlens_shuffle", prof_sh)
    res = {"lens": "jlens", "band": [band[0], band[-1]], "discovery": {}, "layer_curve_k10_discovery": SL.layer_curve(prof, 10, "discovery")}
    sealed = {"confirmation": {}}
    for k in (1, 10, 25):
        d, c = SL.pass_at_k(prof, band, k, "discovery"), SL.pass_at_k(prof, band, k, "confirmation")
        res["discovery"][f"pass@{k}"] = {kk: v for kk, v in d.items() if kk != "per_item"}
        sealed["confirmation"][f"pass@{k}"] = {kk: v for kk, v in c.items() if kk != "per_item"}
    bank.put("readouts", "B1-int", {"jlens": res})
    bank.metrics["sealed"].setdefault("B1-int", {})["jlens"] = sealed
    sh = SL.pass_at_k(prof_sh, band, 10, "discovery")
    real = res["discovery"]["pass@10"]["overall"]
    ctrl = {"shuffle_pass@10_discovery": sh["overall"], "real_pass@10_discovery": real, "n_items": sh["n_items"],
            "pass": (sh["overall"] is not None and sh["overall"] <= 0.05 and sh["n_items"] >= 20 and real is not None and real - sh["overall"] >= 0.10),
            "rule": "shuffle ≤ 0.05 (ceiling), n ≥ 20 (else VOID), real − shuffle ≥ 0.10 (informative)"}
    bank.put("controls", "shuffle_jlens", ctrl)
    if not ctrl["pass"]:
        bank.stamp_void("B1-int/jlens", f"shuffle control: {ctrl}")
    for want in (True, False):
        for p, it in zip(prof, ints):
            hit = min(p["ranks"][0][L] for L in band) <= 10
            if hit == want:
                bank.sample("B1-int", "jlens", {"hash": p["hash"], "set": p["set"], "prompt": it["prompt"],
                                                "readout_idx": it["readout_idx"], "intermediate": it["intermediates_scored"][0]["key"],
                                                "ids": it["intermediates_scored"][0]["ids"], "ranks_by_layer": p["ranks"][0],
                                                "band_min_rank": min(p["ranks"][0][L] for L in band), "hit_k10": hit, "correct": hit})
                break

    # ---- TRACE bank + null bank -----------------------------------------------------------------
    with SH.Timer(bank, "banks"):
        tb = SV.trace_bank(lm, tc["seqs"], n_pos=SL.TRACE_POSITIONS)
        nb = SV.null_bank(lm, tc["seqs"], n=min(64, 4 * len(tc["seqs"])), seed=0, min_pos=SL.TRACE_POSITIONS)
    torch.save({"acts": tb, "positions": list(range(SL.TRACE_POSITIONS)), "bos_offset": lm.bos_offset,
                "corpus_sha256": tc["sha256"], "ckpt": member["id"]}, os.path.join(a.out, "trace_bank.pt"))
    torch.save({**nb, "corpus_sha256": tc["sha256"], "ckpt": member["id"]}, os.path.join(a.out, "null_bank.pt"))
    bank.put("readouts", "banks", {"trace_bank_shape": list(tb.shape), "null_bank_shape": list(nb["acts"].shape),
                                   "trace_mean_norm_by_layer": tb.float().norm(dim=-1).mean(dim=(0, 2)).tolist()})
    bank.top("status", "done")
    bank.event("done", out=a.out)


if __name__ == "__main__":
    main()
