"""STAGEDIFF pair — one hop, parent then child resident (never both): clean B1/B2 on each, I1 (+ same-norm
head control), I2 (+ 8 norm-matched random-difference controls), I3 (+ parent positive control and the
random-direction τ_live), parent-lens reading of child activations, TRACE readout (+ null, sequence
bootstrap), ROT under both heads (spec §2, §5, W3).

    python -m infra.aws_pg.jobs.stagediff_pair --hop H5 --out /work/out \
        --parent-fit s3://…/w2/fit-C4/1/out --child-fit s3://…/w2/fit-C5/1/out
    (C0 as parent: --parent-fit <fit-C0a/out>,<fit-C0b/out> — the two halves are merged)
    python -m infra.aws_pg.jobs.stagediff_pair --dryrun --out <dir>

Two stages in one job: `parent` (under the parent's transformers pin) banks clean B1/B2, I3 positive
control + τ_live + parent hallucination rate, the head snapshot; `child` (under the child's pin) banks
everything else. Stages hand over through out/ (parent_head.pt, pair_state.json, per_item/*_parent.json).
Confirmation-half numbers are banked per item and sealed; only discovery R's are printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

MODULE = "infra.aws_pg.jobs.stagediff_pair"
INPUT_FILES = ("lens.pt", "trace_bank.pt", "null_bank.pt", "metrics.json")


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop", default="H5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent-fit", default=None, help="S3 prefix (…/out) of the parent's fit; two comma-separated for C0")
    ap.add_argument("--child-fit", default=None)
    ap.add_argument("--stage", choices=["parent", "child"], default=None)
    ap.add_argument("--n-rand", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tau-pct", type=float, default=95.0)
    ap.add_argument("--n-boot-trace", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dryrun", action="store_true")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------- inputs
def fetch_inputs(a):
    from infra.aws_pg.jobs._stagediff_shared import s3_fetch
    for side, spec in (("parent", a.parent_fit), ("child", a.child_fit)):
        assert spec, f"--{side}-fit is required"
        for i, prefix in enumerate(spec.split(",")):
            s3_fetch(prefix, os.path.join(a.out, "inputs", side, str(i)), INPUT_FILES)


def load_side(out: str, side: str):
    """(lens, trace_bank dict, null_bank dict, fit metrics) — lenses of several parts are merged (C0 halves)."""
    import torch
    from infra.pg.stagediff_lens import merge_lenses
    base = os.path.join(out, "inputs", side)
    parts = sorted(os.listdir(base))
    lens = merge_lenses([os.path.join(base, p, "lens.pt") for p in parts])
    tb = torch.load(os.path.join(base, parts[0], "trace_bank.pt"), map_location="cpu", weights_only=True)
    nb = torch.load(os.path.join(base, parts[0], "null_bank.pt"), map_location="cpu", weights_only=True)
    met = json.load(open(os.path.join(base, parts[0], "metrics.json")))
    if len(parts) > 1:      # the halves saw the same trace corpus: banks identical by construction, assert it
        tb2 = torch.load(os.path.join(base, parts[1], "trace_bank.pt"), map_location="cpu", weights_only=True)
        assert tb2["corpus_sha256"] == tb["corpus_sha256"] and torch.equal(tb2["acts"], tb["acts"]), "C0 halves disagree on the trace bank"
    return lens, tb, nb, met


# ------------------------------------------------------------------------------------- I3 (swap loop)
def swap_records(member, lm, lens, beh_items, int_by_hash, partner, alpha, band, seed=0):
    """For every B1-beh item with a strict target and a scorable intermediate: clean / swapped(old→alt) /
    random-direction logits at the last position → Δ, flips, band rank of the swapped intermediate."""
    import torch
    from infra.pg.stagediff_interv import Swap
    from infra.pg.stagediff_lens import item_acts, rank_profile
    from infra.pg.stagediff_models import V_SHARED
    fwd = member["fwd"]
    rows = [it for it in beh_items if it["scorable"] and it["hash"] in int_by_hash and it["hash"] in partner]
    ints = [int_by_hash[it["hash"]] for it in rows]
    acts = item_acts(lm, ints)
    prof = {p["hash"]: p for p in rank_profile(lm, ints, acts, lens=lens)}
    g, W = lm.head()
    sw = Swap(lm, lens, g, W, band=band, alpha=alpha)
    out, skipped = [], []
    for k, it in enumerate(rows):
        pr = prof[it["hash"]]
        inter = int_by_hash[it["hash"]]["intermediates_scored"][0]
        band_ranks = [min(pr["ranks"][0][L] for L in band)]
        # the id of the first intermediate with the lowest band rank
        best_id = inter["ids"][0]
        if len(inter["ids"]) > 1:
            per_id = []
            for tid in inter["ids"]:
                pp = rank_profile(lm, [dict(int_by_hash[it["hash"]], intermediates_scored=[{"key": inter["key"], "ids": [tid]}])], acts, lens=lens)[0]
                per_id.append(min(pp["ranks"][0][L] for L in band))
            best_id = inter["ids"][int(torch.tensor(per_id).argmin())]
        # the alternative must be a DIFFERENT intermediate: walk the derangement cycle past same-key partners
        # (order-ops items share keys like "5"); an item whose whole cycle shares its key is skipped and counted
        alt_hash, hops = partner[it["hash"]], 0
        while int_by_hash[alt_hash]["intermediates_scored"][0]["key"] == inter["key"] and hops < len(rows):
            alt_hash, hops = partner[alt_hash], hops + 1
        if int_by_hash[alt_hash]["intermediates_scored"][0]["key"] == inter["key"]:
            skipped.append(it["hash"]); continue
        alt_int = int_by_hash[alt_hash]["intermediates_scored"][0]
        alt_beh = next(b for b in beh_items if b["hash"] == alt_hash)
        alt_id = alt_int["ids"][0]
        orig_t = it["target_ids"][0]
        alt_t = (alt_beh["target_ids"] or alt_beh["first_ids"])[0]
        with torch.inference_mode():
            lg_clean = fwd.next_token_logits([it["prompt"]])[0, :V_SHARED]
            with sw.set_pair(best_id, new_id=alt_id):
                lg_swap = fwd.next_token_logits([it["prompt"]])[0, :V_SHARED]
            cos_on = dict(sw.cos_old_new)
            with sw.set_pair(best_id, random_seed=seed * 100003 + k):
                lg_rand = fwd.next_token_logits([it["prompt"]])[0, :V_SHARED]
        d = lambda lg: float(lg[alt_t] - lg[orig_t])   # noqa: E731
        tc, ts, tr = int(lg_clean.argmax()), int(lg_swap.argmax()), int(lg_rand.argmax())
        out.append({"hash": it["hash"], "set": it["set"], "split": it["split"], "prompt": it["prompt"],
                    "old_key": inter["key"], "old_id": best_id, "old_token": lm.tokenizer.decode([best_id]),
                    "alt_hash": alt_hash, "alt_key": alt_int["key"], "alt_id": alt_id, "alt_token": lm.tokenizer.decode([alt_id]),
                    "orig_target_id": orig_t, "alt_target_id": alt_t, "band_min_rank": band_ranks[0],
                    "lens_sees_k10": band_ranks[0] <= 10, "clean_correct": tc in it["target_ids"],
                    "top_clean": tc, "top_swap": ts, "top_rand": tr,
                    "top_clean_token": lm.tokenizer.decode([tc]), "top_swap_token": lm.tokenizer.decode([ts]),
                    "delta_swap": d(lg_swap) - d(lg_clean), "delta_rand": d(lg_rand) - d(lg_clean),
                    "cos_old_new_band_mean": sum(cos_on.values()) / len(cos_on),
                    "flip_any_swap": ts != tc, "flip_to_alt_swap": ts == alt_t, "flip_any_rand": tr != tc,
                    "p_top_clean": float(torch.softmax(lg_clean.float(), -1)[tc]),
                    "correct": None})
    return out, {"cond": sw.cond, "band": list(sw.band), "alpha": alpha, "n_rows": len(rows),
                 "n_skipped_same_key_cycle": len(skipped), "construction": "residual-space read-out direction swap (§15.19)"}


def i3_summary(recs, tau_live, which):
    from infra.pg.stagediff_stats import wilson
    import numpy as np
    right = [r for r in recs if r["clean_correct"]]
    wrong_seen = [r for r in recs if (not r["clean_correct"]) and r["lens_sees_k10"]]
    pos = {"n": len(right), "flip_any_swap": wilson(sum(r["flip_any_swap"] for r in right), len(right)),
           "flip_to_alt_swap": wilson(sum(r["flip_to_alt_swap"] for r in right), len(right)),
           "flip_any_rand": wilson(sum(r["flip_any_rand"] for r in right), len(right))}
    pos["pass"] = (len(right) >= 20 and (pos["flip_any_swap"]["rate"] or 0) >= 0.20
                   and (pos["flip_any_swap"]["rate"] or 0) > (pos["flip_any_rand"]["rate"] or 0) + 0.05)
    pos["rule"] = "n ≥ 20 (else VOID); flip_any(swap) ≥ 0.20 (floor); flip_any(swap) > flip_any(random) + 0.05 (informative)"
    live = [r["delta_swap"] >= tau_live for r in wrong_seen] if tau_live is not None else []
    hal = {"n": len(wrong_seen), "tau_live": tau_live,
           "live": wilson(sum(live), len(live)) if live else None,
           "hallucination_rate": (1 - sum(live) / len(live)) if live else None,
           "delta_swap_median": float(np.median([r["delta_swap"] for r in wrong_seen])) if wrong_seen else None}
    return {"which": which, "positive_control": pos, "hallucination": hal,
            "n_records": len(recs), "n_lens_sees_k10": sum(r["lens_sees_k10"] for r in recs)}


# ------------------------------------------------------------------------------------- TRACE
def trace_block(lm, tbP, tbC, nbP, robot_ids, band, lens, n_rand, n_boot, seed=0):
    import numpy as np
    import torch
    from infra.pg.stagediff_interv import delta_bar, norm_match, random_diffs
    from infra.pg.stagediff_lens import lens_logits, relevance, topk_ids, trace_readout
    dbar = delta_bar(tbC["acts"], tbP["acts"])                       # [L, P, d]
    res = {"delta_norm_by_layer_pos": dbar.norm(dim=-1).tolist()}
    for name, ln in (("jlens", lens), ("logitlens", None)):
        res[name] = trace_readout(lm, dbar, robot_ids, band, ln)
        rd = random_diffs(nbP["acts"], n_rand, seed)                # [n, L, d]
        nulls = []
        for k in range(n_rand):
            v = rd[k].unsqueeze(1).expand(-1, dbar.shape[1], -1)     # [L, P, d] same vector at every position
            v = norm_match(v.transpose(0, 1), dbar.transpose(0, 1)).transpose(0, 1)   # per-(L,P) norm match
            nulls.append(trace_readout(lm, v, robot_ids, band, ln)["best"]["relevance"])
        res[name]["null_best_relevance"] = {"draws": nulls, "mean": float(np.mean(nulls)), "max": float(max(nulls)),
                                            "cos_with_delta_pooled_mean": float(torch.nn.functional.cosine_similarity(
                                                rd.mean(0).flatten(), dbar.mean(1).flatten(), dim=0))}
    # sequence bootstrap of the best J-lens relevance (re-selected per draw, matched to the null's selection)
    rng = np.random.default_rng(seed)
    nS = tbC["acts"].shape[0]
    accP, accC = tbP["acts"].to(lm.device), tbC["acts"].to(lm.device)
    Js = {L: lens.jacobians[L].to(lm.device) for L in band if L in lens.jacobians}
    best = []
    robot = torch.zeros(lens_logits(lm, dbar[0, 0].to(lm.device), 0, None).shape[-1], dtype=torch.bool, device=lm.device)
    robot[torch.as_tensor(sorted(robot_ids), dtype=torch.long, device=lm.device)] = True
    for _ in range(n_boot):
        idx = torch.as_tensor(rng.integers(0, nS, size=nS), device=lm.device)
        db = accC[idx].float().mean(0) - accP[idx].float().mean(0)   # [L, P, d]
        xs = []
        for L in band:
            x = db[L]
            xs.append(x @ Js[L].T if L in Js else x)
        X = torch.stack(xs, 0).reshape(-1, db.shape[-1])              # [len(band)*P, d]
        top = lm.unembed(X).topk(25).indices                          # [n, 25]
        rel = robot[top].float().mean(1)
        best.append(float(rel.max()))
    res["boot_best_relevance_jlens"] = {"lo": float(np.percentile(best, 2.5)), "hi": float(np.percentile(best, 97.5)),
                                        "mean": float(np.mean(best)), "n_boot": n_boot}
    return res, dbar


# ------------------------------------------------------------------------------------- stages
def run_stage(a, stage: str, stub_seed=None):
    import torch
    from infra.pg import stagediff_bank as SB, stagediff_interv as SV, stagediff_items as SI, stagediff_lens as SL, stagediff_models as SM
    from infra.pg import stagediff_stats as ST
    from infra.aws_pg.jobs import _stagediff_shared as SH
    P, C = SM.HOPS[a.hop]
    cid = P if stage == "parent" else C
    bank = SB.Bank(a.out, "stagediff_pair", vars(a), resume=(stage == "child"))
    band = SM.BAND if not a.dryrun else (1, 2)
    with SH.Timer(bank, f"load_{stage}"):
        member = SM.stub_member(bos_id=1, seed=stub_seed) if a.dryrun else SM.load_ckpt(cid, a.device)
    tok = member["tok"]
    lm = SL.AnatomyLensModel(member)
    bank.put("controls", f"member_{stage}", {"ckpt": member["id"], "repo": member["repo"], "sha": member["sha"],
                                             "hashes": member["hashes"], "bos_id": member["bos_id"], "transformers": member["transformers"]})
    lens_P, tbP, nbP, metP = load_side(a.out, "parent")
    lens_C, tbC, nbC, metC = load_side(a.out, "child")
    SL.lens_to(lens_P, lm.device); SL.lens_to(lens_C, lm.device)
    assert tbP["corpus_sha256"] == tbC["corpus_sha256"], "parent and child trace banks come from different corpora"
    sets = SI.load_sets()
    beh, ints = SI.beh_items(tok, sets), SI.int_items(tok, sets)
    ld = None
    if a.dryrun:
        from infra.pg.matrix_items import fake_load_dataset as ld
    mm = SI.mmlu_cloze_items("discovery", ld) + SI.mmlu_cloze_items("confirmation", ld)
    if a.smoke or a.dryrun:
        beh, ints = SI.smoke_subset(beh, 10), SI.smoke_subset(ints, 6)
        mm = SI.smoke_subset([m for m in mm if m["split"] == "discovery"], 1)[:12] + SI.smoke_subset([m for m in mm if m["split"] == "confirmation"], 1)[:12]
    if a.dryrun:      # the stub reads only ' 1'..' 4': make every beh item strictly scorable on the stub
        for it in beh:
            it["target_ids"] = it["target_ids"] or SI.single_token_ids(tok, "1"); it["scorable"] = True
    int_by_hash = {it["hash"]: it for it in ints}
    beh_hashes = {it["hash"] for it in beh}
    ints_beh = [it for it in ints if it["hash"] in beh_hashes]
    partner = SI.derange_within_set(ints_beh)
    tracked = SH.tracked_hashes(beh)
    tracked_mm = SH.tracked_hashes(mm, 1)
    lens_own = lens_P if stage == "parent" else lens_C
    arm = stage

    def b1b2(arm_name):
        with SH.Timer(bank, f"B1-beh_{arm_name}"):
            SH.bank_records(bank, "B1-beh", arm_name, SH.score_b1(member, beh), tracked, sample_all=True, keys=("correct", "correct_first"))
        with SH.Timer(bank, f"B2_{arm_name}"):
            SH.bank_records(bank, "B2", arm_name, SH.score_b2(member, mm), tracked_mm)

    b1b2(arm)

    # ---- I3 on this member (positive control + τ_live on the parent; hallucination on both) -----
    with SH.Timer(bank, f"I3_{stage}"):
        recs, swmeta = swap_records(member, lm, lens_own, beh, int_by_hash, partner, a.alpha, band, seed=0)
    bank.per_item(f"I3_{stage}", recs)
    for want in (True, False):
        r = next((x for x in recs if x["flip_any_swap"] == want), None)
        if r:
            bank.sample("I3", stage, {**r, "correct": r["flip_any_swap"]})
    state_path = os.path.join(a.out, "pair_state.json")
    disc = [r for r in recs if r["split"] == "discovery"]
    conf = [r for r in recs if r["split"] == "confirmation"]
    if stage == "parent":
        import numpy as np
        right = [r for r in disc if r["clean_correct"]]          # τ_live is control data: parent, DISCOVERY half
        tau = float(np.percentile([r["delta_rand"] for r in right], a.tau_pct)) if right else None
        summ = i3_summary(disc, tau, "parent")
        bank.put("controls", "I3_parent", {**summ, "swap": swmeta, "half": "discovery"})
        bank.metrics["sealed"].setdefault("I3", {})["parent"] = i3_summary(conf, tau, "parent")
        bank.write()
        if not summ["positive_control"]["pass"]:
            bank.stamp_void("I3", f"parent positive control did not fire: {summ['positive_control']}")
        json.dump({"tau_live": tau, "tau_pct": a.tau_pct, "n_tau": len(right), "half": "discovery"}, open(state_path, "w"))
        torch.save(SV.head_snapshot(member), os.path.join(a.out, "parent_head.pt"))
        bank.put("controls", "parent_head", {"n_rows": SM.V_SHARED, "ln_f_norm": float(SV.head_snapshot(member)["ln_f"].norm())})
        bank.top("status", "parent-done")
        bank.event("stage-done", stage="parent")
        return

    # ---------------------------------------------------------------- child stage ------------------
    st = json.load(open(state_path))
    summ = i3_summary(disc, st["tau_live"], "child")
    bank.put("readouts", "I3", {"child": {**summ, "swap": swmeta, "half": "discovery"}, "parent": bank.metrics["controls"]["I3_parent"]})
    bank.metrics["sealed"].setdefault("I3", {})["child"] = i3_summary(conf, st["tau_live"], "child")
    bank.write()
    snap = torch.load(os.path.join(a.out, "parent_head.pt"), map_location="cpu", weights_only=True)
    own = SV.head_snapshot(member)
    # ---- I1 ----
    restore = SV.transplant_head(member, snap)
    b1b2("I1"); restore()
    restore, hstats = SV.random_head_control(member, snap, seed=0)
    b1b2("I1rand"); restore()
    bank.put("controls", "I1_random_head", hstats)
    after = SV.head_snapshot(member)
    assert torch.equal(after["ln_f"], own["ln_f"]) and torch.equal(after["lm_head"], own["lm_head"]), "head restore is not exact"
    # ---- I2 ----
    dbar = SV.delta_bar(tbC["acts"], tbP["acts"])                 # [L, P, d]
    pooled = dbar.mean(1)                                          # [L, d]
    u = SV.unit_rows(pooled)
    with SV.ProjectOut(lm.layers, {L: u[L] for L in band}):
        b1b2("I2")
    rd = SV.norm_match(SV.random_diffs(nbP["acts"], a.n_rand, seed=0), pooled)
    ur = SV.unit_rows(rd)
    cos = torch.nn.functional.cosine_similarity(ur, u.unsqueeze(0).expand_as(ur), dim=-1)   # [n, L]
    bank.put("controls", "I2_vectors", {"delta_pooled_norm_by_layer": pooled.norm(dim=-1).tolist(),
                                        "rand_cos_with_delta_band_mean": float(cos[:, list(band)].mean()),
                                        "rand_norm_ratio_band": float((rd.norm(dim=-1) / pooled.norm(dim=-1).clamp_min(1e-12))[:, list(band)].mean()),
                                        "n_rand": a.n_rand})
    for k in range(a.n_rand):
        with SV.ProjectOut(lm.layers, {L: ur[k, L] for L in band}):
            b1b2(f"I2rand{k}")
    # ---- parent-lens reading of CHILD activations (confound control) + child-lens re-read ----
    with SH.Timer(bank, "parent_lens_readout"):
        acts = SL.item_acts(lm, ints)                       # CHILD activations, captured once with the child's own head
        prof_c = SL.rank_profile(lm, ints, acts, lens=lens_C)
        restore = SV.transplant_head(member, snap)          # read the same activations through the PARENT head
        prof_p = SL.rank_profile(lm, ints, acts, lens=lens_P)
        prof_ph_only = SL.rank_profile(lm, ints, acts, lens=lens_C)
        restore()
        gP, WP = snap["ln_f"], snap["lm_head"]
    after = SV.head_snapshot(member)
    assert torch.equal(after["ln_f"], own["ln_f"]) and torch.equal(after["lm_head"], own["lm_head"]), "head restore is not exact"
    bank.per_item("B1-int_child_lens_in_pair", prof_c)
    bank.per_item("B1-int_parent_lens_on_child", prof_p)
    bank.per_item("B1-int_child_lens_parent_head", prof_ph_only)
    out = {}
    for name, prof in (("child_lens", prof_c), ("parent_lens_parent_head", prof_p), ("child_lens_parent_head", prof_ph_only)):
        for want in (True, False):          # CK-6 W3: one hit and one miss sample per lens arm (§6.8)
            for pp, it in zip(prof, ints):
                hit = min(pp["ranks"][0][L] for L in band) <= 10
                if hit == want:
                    bank.sample("B1-int-pair", name, {"hash": pp["hash"], "set": pp["set"], "split": pp["split"], "prompt": it["prompt"],
                                                     "readout_idx": it["readout_idx"], "intermediate": it["intermediates_scored"][0]["key"],
                                                     "ids": it["intermediates_scored"][0]["ids"], "ranks_by_layer": pp["ranks"][0],
                                                     "band_min_rank": min(pp["ranks"][0][L] for L in band), "hit_k10": hit, "correct": hit})
                    break
        d = {f"pass@{k}": {kk: v for kk, v in SL.pass_at_k(prof, band, k, "discovery").items() if kk != "per_item"} for k in (1, 10, 25)}
        s = {f"pass@{k}": {kk: v for kk, v in SL.pass_at_k(prof, band, k, "confirmation").items() if kk != "per_item"} for k in (1, 10, 25)}
        out[name] = {"discovery": d, "layer_curve_k10_discovery": SL.layer_curve(prof, 10, "discovery")}
        bank.metrics["sealed"].setdefault("B1-int-pair", {})[name] = {"confirmation": s}
    bank.put("readouts", "B1-int-pair", out)
    # ---- TRACE ----
    rv = SI.robot_vocab_ids(tok)
    robot = set(rv["ids"])
    bank.put("controls", "robot_vocab", rv)
    if len(robot) < 20 and not a.dryrun:
        bank.stamp_void("TRACE", f"ROBOT_VOCAB resolves to {len(robot)} single-token ids (< 20)")
    with SH.Timer(bank, "TRACE"):
        tr, _ = trace_block(lm, tbP, tbC, nbP, robot, band, lens_C, a.n_rand, a.n_boot_trace if not a.dryrun else 20)
    bank.put("readouts", "TRACE", tr)
    # ---- ROT ----
    with SH.Timer(bank, "ROT"):
        toks = metP["controls"]["top_tokens"]
        gC, WC = lm.head()
        rot = SL.rot_by_layer(lens_P, lens_C, (gP.to(lm.device), WP.to(lm.device)), (gC, WC), toks, layers=range(lm.n_layers))
    band_own = [rot[L]["own_heads"]["point"] for L in band]
    band_ph = [rot[L]["parent_head"]["point"] for L in band]
    bank.put("readouts", "ROT", {"by_layer": rot, "n_tokens": len(toks),
                                 "band_median_own_heads": sorted(band_own)[len(band_own) // 2],
                                 "band_median_parent_head": sorted(band_ph)[len(band_ph) // 2]})
    # ---- discovery-half R's (printed; confirmation is the verdict's) ----
    def per_item(readout, arm_name, key="correct"):
        d = json.load(open(os.path.join(a.out, "per_item", f"{readout}_{arm_name}.json")))
        return {h: float(v[key]) for h, v in d.items() if v["split"] == "discovery" and v[key] is not None}
    rs = {}
    for readout in ("B1-beh", "B2"):
        p, c = per_item(readout, "parent"), per_item(readout, "child")
        keys = sorted(set(p) & set(c))
        if len(keys) < 20:
            rs[readout] = {"note": f"only {len(keys)} paired discovery items"}; continue
        P_, C_ = ST.align({k: p[k] for k in keys}, {k: c[k] for k in keys})
        row = {"n": len(keys), "acc_parent": float(P_.mean()), "acc_child": float(C_.mean())}
        for arm_name in ["I1", "I1rand", "I2"] + [f"I2rand{k}" for k in range(a.n_rand)]:
            x = per_item(readout, arm_name)
            X_ = ST.align({k: p[k] for k in keys}, {k: x[k] for k in keys})[1]
            row[arm_name] = ST.boot_recovery(P_, C_, X_)
        rs[readout] = row
    bank.put("readouts", "R_discovery", rs)
    bank.top("status", "done")
    bank.event("stage-done", stage="child")


def main(argv=None):
    a = parse(argv)
    if a.dryrun:
        from infra.aws_pg.jobs import stagediff_fit
        for side, seed in (("parent", 0), ("child", 1)):
            d = os.path.join(a.out, "inputs", side, "0")
            if not os.path.exists(os.path.join(d, "lens.pt")):
                stagediff_fit.main(["--dryrun", "--out", d, "--stub-seed", str(seed)])
        run_stage(a, "parent", stub_seed=0)
        run_stage(a, "child", stub_seed=1)
        return
    from infra.pg.diff_harvest import hf_online_setup
    if os.environ.get("PG_S3_PREFIX") or os.environ.get("HF_TOKEN"):
        hf_online_setup()
    from infra.pg import stagediff_models as SM
    if a.stage:
        run_stage(a, a.stage)
        return
    fetch_inputs(a)
    P, C = SM.HOPS[a.hop]
    for stage, cid in (("parent", P), ("child", C)):
        argv_stage = [x for x in sys.argv[1:]] + ["--stage", stage]
        if SM.ensure_pin(cid, MODULE, argv_stage) is None:
            run_stage(a, stage)


if __name__ == "__main__":
    main()
