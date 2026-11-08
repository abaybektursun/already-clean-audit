"""STAGEDIFF battery — one checkpoint, text-only: B1-beh, B2, B3, logit-lens B1-int + shuffle control (spec §2, W1).

    python -m infra.aws_pg.jobs.stagediff_battery --ckpt C0 --out /work/out [--smoke] [--bos-ablation]
    python -m infra.aws_pg.jobs.stagediff_battery --dryrun --out <dir>        # stub model, no weights, no datasets

Discovery-half numbers are printed; confirmation-half numbers go to metrics["sealed"] and are not read
before W4 (spec §9.3). Every readout banks per-item vectors and samples (§10).
"""
from __future__ import annotations

import argparse
import os
import sys

MODULE = "infra.aws_pg.jobs.stagediff_battery"


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="C0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="24 B1 items, 12 MMLU items, 4 ppl windows, 12 int items")
    ap.add_argument("--bos-ablation", action="store_true", help="also run B1-beh/B2 with the BOS rule flipped")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dryrun", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if not a.dryrun:
        from infra.aws_pg.jobs._stagediff_shared import boot
        if boot(MODULE, a.ckpt, sys.argv[1:]):
            return
    import torch
    from infra.pg import stagediff_bank as SB, stagediff_items as SI, stagediff_lens as SL, stagediff_models as SM
    from infra.aws_pg.jobs import _stagediff_shared as SH
    bank = SB.Bank(a.out, "stagediff_battery", vars(a))
    with SH.Timer(bank, "load"):
        member = SM.stub_member(bos_id=1) if a.dryrun else SM.load_ckpt(a.ckpt, a.device, batch_size=a.batch_size)
    tok = member["tok"]
    bank.manifest({"ckpt": member["id"], "repo": member["repo"], "repo_sha": member["sha"], "stage": member["stage"],
                   "hashes": member["hashes"], "bos_id": member["bos_id"], "dtype": member["dtype"], "attn": member["attn"],
                   "transformers_in_process": member["transformers"], "n_lm_head_rows": member["n_lm_head_rows"],
                   "lens_eval_sha256": SI.FILE_SHA256, "robot_vocab_sha256": SI.ROBOT_VOCAB_SHA256,
                   "forward": "StageForward: text tokenizer, BOS per roster, left pad, content position_ids, logits_to_keep=1, shared-prefix slice"})
    sets = SI.load_sets()
    beh, ints = SI.beh_items(tok, sets), SI.int_items(tok, sets)
    bank.put("controls", "item_counts", SI.counts(beh, ints))
    bank.put("controls", "robot_vocab", SI.robot_vocab_ids(tok))
    if a.smoke or a.dryrun:
        beh, ints = SI.smoke_subset(beh, 8), SI.smoke_subset(ints, 4)
    tracked = SH.tracked_hashes(beh)

    # ---- B1-beh (clean) ---------------------------------------------------------------------
    with SH.Timer(bank, "B1-beh"):
        rec = SH.score_b1(member, beh)
    SH.bank_records(bank, "B1-beh", "clean", rec, tracked, sample_all=True, keys=("correct", "correct_first"))

    # ---- B2 (MMLU cloze, both halves) --------------------------------------------------------
    ld = None
    if a.dryrun:
        from infra.pg.matrix_items import fake_load_dataset as ld
    with SH.Timer(bank, "B2"):
        mm = SI.mmlu_cloze_items("discovery", ld) + SI.mmlu_cloze_items("confirmation", ld)
        if a.smoke or a.dryrun:
            mm = SI.smoke_subset([m for m in mm if m["split"] == "discovery"], 1)[:12] + \
                 SI.smoke_subset([m for m in mm if m["split"] == "confirmation"], 1)[:12]
        rec2 = SH.score_b2(member, mm)
    SH.bank_records(bank, "B2", "clean", rec2, SH.tracked_hashes(mm, 1))

    # ---- B3 (wikitext-2 perplexity through the shell) ----------------------------------------
    with SH.Timer(bank, "B3"):
        ppl = (SH.dryrun_ppl if a.dryrun else SH.run_b3)(member, 8192 if a.smoke else 120_000)
    for s in ppl.pop("samples", []):
        bank.sample("B3", "clean", s)
    bank.per_item("B3_clean", {"per_window": ppl.pop("per_window")})
    bank.put("readouts", "B3", {"clean": ppl})

    # ---- B1-int, logit lens (J = I) + shuffle control ---------------------------------------
    with SH.Timer(bank, "B1-int-logitlens"):
        lm = SL.AnatomyLensModel(member)
        acts = SL.item_acts(lm, ints)
        prof = SL.rank_profile(lm, ints, acts, lens=None)
        partner = SI.derange_within_set(ints)
        prof_sh = SL.rank_profile(lm, ints, acts, lens=None, partner=partner)
    bank.per_item("B1-int_logitlens", prof)
    bank.per_item("B1-int_logitlens_shuffle", prof_sh)
    band = SM.BAND if not a.dryrun else (1, 2)
    res = {"lens": "logit", "band": [band[0], band[-1]], "discovery": {}, "layer_curve_k10_discovery": SL.layer_curve(prof, 10, "discovery")}
    sealed = {"confirmation": {}}
    for k in (1, 10, 25):
        d, c = SL.pass_at_k(prof, band, k, "discovery"), SL.pass_at_k(prof, band, k, "confirmation")
        res["discovery"][f"pass@{k}"] = {kk: v for kk, v in d.items() if kk != "per_item"}
        sealed["confirmation"][f"pass@{k}"] = {kk: v for kk, v in c.items() if kk != "per_item"}
    bank.put("readouts", "B1-int", {"logitlens": res})
    bank.metrics["sealed"].setdefault("B1-int", {})["logitlens"] = sealed
    sh = SL.pass_at_k(prof_sh, band, 10, "discovery")
    real = res["discovery"]["pass@10"]["overall"]
    ctrl = {"shuffle_pass@10_discovery": sh["overall"], "real_pass@10_discovery": real, "n_items": sh["n_items"],
            "pass": (sh["overall"] is not None and sh["overall"] <= 0.05 and sh["n_items"] >= 20 and real is not None and real - sh["overall"] >= 0.10),
            "rule": "shuffle ≤ 0.05 (ceiling), n ≥ 20 (else VOID), real − shuffle ≥ 0.10 (informative)"}
    bank.put("controls", "shuffle_logitlens", ctrl)
    if not ctrl["pass"]:
        bank.stamp_void("B1-int/logitlens", f"shuffle control: {ctrl}")
    # two intermediate-rank samples (one hit, one miss at k=10 in band)
    for want in (True, False):
        for p, it in zip(prof, ints):
            hit = min(p["ranks"][0][L] for L in band) <= 10
            if hit == want:
                bank.sample("B1-int", "logitlens", {"hash": p["hash"], "set": p["set"], "prompt": it["prompt"],
                                                    "readout_idx": it["readout_idx"], "intermediate": it["intermediates_scored"][0]["key"],
                                                    "ids": it["intermediates_scored"][0]["ids"], "ranks_by_layer": p["ranks"][0],
                                                    "band_min_rank": min(p["ranks"][0][L] for L in band), "hit_k10": hit, "correct": hit})
                break

    # ---- optional BOS ablation (instrument control for the H1 position-0 confound) -----------
    if a.bos_ablation:
        flipped = SM.MOLMO_BOS if member["bos_id"] is None else None
        fwd0 = member["fwd"]
        member["fwd"] = type(fwd0)(member["model"], tok, fwd0.device, bos_id=flipped)
        member["fwd"].batch_size = fwd0.batch_size
        member["bos_id"], keep = flipped, member["bos_id"]
        with SH.Timer(bank, "B1-beh-bosflip"):
            SH.bank_records(bank, "B1-beh", "bosflip", SH.score_b1(member, beh), tracked, sample_all=True, keys=("correct", "correct_first"))
        with SH.Timer(bank, "B2-bosflip"):
            SH.bank_records(bank, "B2", "bosflip", SH.score_b2(member, mm), SH.tracked_hashes(mm, 1))
        member["fwd"], member["bos_id"] = fwd0, keep
        bank.put("controls", "bos_ablation", {"flipped_bos_id": flipped, "note": "B1-beh/B2 under the other BOS setting"})

    bank.top("status", "done")
    bank.event("done", out=a.out)


if __name__ == "__main__":
    main()
