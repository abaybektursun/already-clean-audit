"""STAGEDIFF E1 — what is lost: candidate-rank accuracy, target rank and next-token entropy under three start tokens
(remediation plan 2026-09-05 §3 E1; fresh prefixes results/stagediff/w6/rank-<C>/<n>). Exploratory, post-unblind.

    python -m infra.aws_pg.jobs.stagediff_rank --ckpt C4 --out /work/out [--conventions none,im_end,endoftext]
    python -m infra.aws_pg.jobs.stagediff_rank --dryrun --out <dir>

Per prompt of the three target sets (255 items; 136 with a one-token in-context target = "strict") and per start-token
convention, at the last prompt token over the shared prefix (151 669 ids):
  correct              the frozen strict readout (top-1 == the in-context target token)          — must reproduce W1/W5
  cand_correct         the target's logit is the highest among the SET's candidate targets (the union of the strict
                       in-context targets of that set over both halves, banked in inputs/candidates.json)
  cand_rank            1-based rank of the target among those candidates
  target_rank          1-based rank of the target in the shared-prefix distribution
  entropy_prefix/full  next-token entropy in nats over the shared prefix / the full vocabulary
  p_target, top_p, top_id, full_argmax_above_prefix, correct_first, cand_first_correct (first-token variants, all 255)
A set with fewer than MIN_CANDIDATES strict candidates is VOID for the candidate readouts (registered failing input).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

MODULE = "infra.aws_pg.jobs.stagediff_rank"
MIN_CANDIDATES = 4


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conventions", default="none,im_end,endoftext")
    ap.add_argument("--min-candidates", type=int, default=MIN_CANDIDATES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dryrun", action="store_true")
    return ap.parse_args(argv)


def candidate_sets(beh: list, min_n: int) -> dict:
    """{set: {"strict": sorted ids, "first": sorted ids, "void": reason|None}} from the items' own in-context targets."""
    out = {}
    for s in sorted({it["set"] for it in beh}):
        rows = [it for it in beh if it["set"] == s]
        strict = sorted({it["target_ids"][0] for it in rows if it["scorable"]})
        first = sorted({it["first_ids"][0] for it in rows if it["first_ids"]})
        out[s] = {"strict": strict, "first": first, "n_strict_items": sum(it["scorable"] for it in rows), "n_items": len(rows),
                  "void": None if len(strict) >= min_n else f"{len(strict)} strict candidates < {min_n}"}
    return out


def score_rank(member, items: list, cands: dict) -> list:
    import torch
    from infra.pg.stagediff_models import V_SHARED
    fwd, tok = member["fwd"], member["tok"]
    full = fwd.next_token_logits([it["prompt"] for it in items])           # [N, V_full] fp32
    pref = full[:, :V_SHARED]
    lp_pref, lp_full = torch.log_softmax(pref, -1), torch.log_softmax(full, -1)
    ent_pref = -(lp_pref.exp() * lp_pref).sum(-1)
    ent_full = -(lp_full.exp() * lp_full).sum(-1)
    out = []
    for i, it in enumerate(items):
        lg = pref[i]
        top = int(lg.argmax())
        t_ids, f_ids = it["target_ids"], it["first_ids"]
        cs = cands[it["set"]]
        rec = {"hash": it["hash"], "set": it["set"], "split": it["split"], "name": it["name"], "prompt": it["prompt"],
               "bos_id": member["bos_id"], "top_id": top, "top_token": tok.decode([top]), "top_p": float(lp_pref[i, top].exp()),
               "target": it["target"], "target_ids": t_ids, "first_ids": f_ids, "scorable": bool(t_ids),
               "entropy_prefix": float(ent_pref[i]), "entropy_full": float(ent_full[i]),
               "full_argmax_above_prefix": bool(int(full[i].argmax()) >= V_SHARED),
               "correct": (top in t_ids) if t_ids else None, "correct_first": top in f_ids if f_ids else None,
               "p_target": float(lp_pref[i, t_ids[0]].exp()) if t_ids else None,
               "target_rank": int(1 + (lg > lg[t_ids[0]]).sum()) if t_ids else None,
               "cand_correct": None, "cand_rank": None, "n_candidates": len(cs["strict"]), "cand_first_correct": None}
        if t_ids and cs["void"] is None:
            c_ids = torch.as_tensor(cs["strict"], dtype=torch.long)
            v = lg[c_ids]
            rec["cand_rank"] = int(1 + (v > lg[t_ids[0]]).sum())
            rec["cand_correct"] = rec["cand_rank"] == 1
        if f_ids and len(cs["first"]) >= MIN_CANDIDATES:
            c_ids = torch.as_tensor(cs["first"], dtype=torch.long)
            rec["cand_first_correct"] = bool(int(1 + (lg[c_ids] > lg[f_ids[0]]).sum()) == 1)
        out.append(rec)
    return out


def summarize(recs: list, split: str) -> dict:
    rows = [r for r in recs if r["split"] == split]
    def mean(key, pred=lambda r: True):
        xs = [r[key] for r in rows if pred(r) and r[key] is not None]
        return (sum(float(x) for x in xs) / len(xs)) if xs else None
    strict = lambda r: r["scorable"]   # noqa: E731
    out = {"n": len(rows), "n_strict": sum(strict(r) for r in rows),
           "acc_strict": mean("correct", strict), "acc_cand": mean("cand_correct", strict), "acc_first": mean("correct_first"),
           "acc_cand_first": mean("cand_first_correct"), "entropy_prefix": mean("entropy_prefix"), "entropy_full": mean("entropy_full"),
           "p_target": mean("p_target", strict), "top_p": mean("top_p"), "target_rank_median": None,
           "share_top_p_below_0.2": (sum(r["top_p"] < 0.2 for r in rows if strict(r)) / max(1, sum(strict(r) for r in rows))),
           "full_argmax_above_prefix": mean("full_argmax_above_prefix"), "per_set": {}}
    tr = sorted(r["target_rank"] for r in rows if strict(r) and r["target_rank"] is not None)
    if tr:
        out["target_rank_median"] = tr[len(tr) // 2]
    for s in sorted({r["set"] for r in rows}):
        sr = [r for r in rows if r["set"] == s]
        out["per_set"][s] = {"n_strict": sum(strict(r) for r in sr),
                             "acc_strict": (sum(r["correct"] for r in sr if strict(r)) / max(1, sum(strict(r) for r in sr))) if any(strict(r) for r in sr) else None,
                             "acc_cand": (sum(bool(r["cand_correct"]) for r in sr if strict(r) and r["cand_correct"] is not None) / max(1, sum(1 for r in sr if strict(r) and r["cand_correct"] is not None))) if any(r["cand_correct"] is not None for r in sr) else None,
                             "entropy_prefix": sum(r["entropy_prefix"] for r in sr) / len(sr)}
    return out


def main(argv=None):
    a = parse(argv)
    if not a.dryrun:
        from infra.aws_pg.jobs._stagediff_shared import boot
        if boot(MODULE, a.ckpt, sys.argv[1:]):
            return
    from infra.pg import stagediff_bank as SB, stagediff_items as SI, stagediff_models as SM
    from infra.aws_pg.jobs import _stagediff_shared as SH
    bank = SB.Bank(a.out, "stagediff_rank", vars(a))
    with SH.Timer(bank, "load"):
        member = SM.stub_member(bos_id=1, seed=1) if a.dryrun else SM.load_ckpt(a.ckpt, a.device)
    tok = member["tok"]
    bank.manifest({"ckpt": member["id"], "repo": member["repo"], "repo_sha": member["sha"], "hashes": member["hashes"], "bos_id_roster": member["bos_id"]})
    beh = SI.beh_items(tok, SI.load_sets())
    if a.dryrun:
        for it in beh:
            it["target_ids"] = it["target_ids"] or SI.single_token_ids(tok, "1"); it["scorable"] = True
    if a.smoke or a.dryrun:
        beh = SI.smoke_subset(beh, 12)
    cands = candidate_sets(beh, a.min_candidates)
    os.makedirs(os.path.join(a.out, "inputs"), exist_ok=True)
    SB._write_json(os.path.join(a.out, "inputs", "candidates.json"),
                   {s: {**c, "strict_tokens": [tok.decode([i]) for i in c["strict"]]} for s, c in cands.items()})
    for s, c in cands.items():
        if c["void"]:
            bank.stamp_void(f"E1_cand[{s}]", c["void"])
    bank.put("controls", "candidate_sets", {s: {"n_strict_candidates": len(c["strict"]), "n_first_candidates": len(c["first"]), "void": c["void"]} for s, c in cands.items()})
    variants = {"none": None, "im_end": SM.MOLMO_BOS, "endoftext": 151643} if not a.dryrun else {"none": None, "one": 1, "two": 2}
    Fwd = type(member["fwd"])
    keys = ("correct", "cand_correct", "cand_rank", "target_rank", "correct_first", "cand_first_correct", "entropy_prefix", "entropy_full",
            "p_target", "top_p", "top_id", "full_argmax_above_prefix", "set", "n_candidates", "scorable")
    for name in (a.conventions.split(",") if not a.dryrun else list(variants)):
        assert name in variants, (name, list(variants))
        bid = variants[name]
        f = Fwd(member["model"], tok, member["fwd"].device, bos_id=bid); f.batch_size = member["fwd"].batch_size
        keep = member["fwd"], member["bos_id"]
        member["fwd"], member["bos_id"] = f, bid
        with SH.Timer(bank, f"E1_{name}"):
            recs = score_rank(member, beh, cands)
        member["fwd"], member["bos_id"] = keep
        bank.per_item(f"E1_{name}", {r["hash"]: {**{k: r[k] for k in keys}, "split": r["split"]} for r in recs})
        bank.put("readouts", f"E1_{name}", {"discovery": summarize(recs, "discovery"), "n_records": len(recs)})
        bank.metrics["sealed"].setdefault("E1", {})[name] = {"confirmation": summarize(recs, "confirmation")}
        bank.write()
        for s in cands:                                   # one candidate-right and one candidate-wrong sample per set per convention
            for want in (True, False):
                r = next((x for x in recs if x["set"] == s and x["split"] == "discovery" and x["cand_correct"] == want), None)
                if r:      # `correct` stays the STRICT readout; the selection key is named (CK review-3, D9)
                    bank.sample("E1", name, {**r, "sample_key": "cand_correct", "sample_key_value": r["cand_correct"], "selection": f"first cand_correct={want} in {s}"})
            for want in (True, False):     # and one strict-right / strict-wrong pair per set
                r = next((x for x in recs if x["set"] == s and x["split"] == "discovery" and x["correct"] == want), None)
                if r:
                    bank.sample("E1", name, {**r, "sample_key": "correct", "sample_key_value": r["correct"], "selection": f"first strict correct={want} in {s}"})
        d = summarize(recs, "discovery")
        bank.event("readout", readout="E1", arm=name, discovery_acc_strict=d["acc_strict"], discovery_acc_cand=d["acc_cand"],
                   discovery_entropy_prefix=d["entropy_prefix"])
    bank.top("status", "done")
    bank.event("done", out=a.out)


def _dryrun_checks(out: str):
    import json
    import torch
    from infra.pg import stagediff_items as SI, stagediff_models as SM
    m = json.load(open(os.path.join(out, "metrics.json")))
    assert m["status"] == "done" and set(m["readouts"]) >= {"E1_none", "E1_one", "E1_two"}, list(m["readouts"])
    pi = json.load(open(os.path.join(out, "per_item", "E1_none.json")))
    for h, r in pi.items():
        assert math.isfinite(r["entropy_prefix"]) and 0 <= r["entropy_prefix"] <= math.log(256) + 1e-6, r
        if r["correct"] is True and r["cand_correct"] is not None:
            assert r["cand_correct"] is True, (h, r)          # strict-correct ⇒ candidate-correct
        if r["cand_rank"] is not None:
            assert 1 <= r["cand_rank"] <= r["n_candidates"], r
    assert os.path.exists(os.path.join(out, "inputs", "candidates.json"))
    # registered failing input: an impossible candidate floor stamps every set VOID
    tok = SM.stub_member(bos_id=1)["tok"]
    beh = SI.smoke_subset(SI.beh_items(tok, SI.load_sets()), 12)
    for it in beh:
        it["target_ids"] = it["target_ids"] or SI.single_token_ids(tok, "1"); it["scorable"] = True
    cs = candidate_sets(beh, 1000)
    assert all(c["void"] for c in cs.values()), cs
    member = SM.stub_member(bos_id=1)
    recs = score_rank(member, beh, cs)
    assert all(r["cand_correct"] is None for r in recs)      # a VOID set scores no candidate readout
    assert torch.isfinite(torch.tensor([r["entropy_full"] for r in recs])).all()
    print("PASS stagediff_rank dry-run (three conventions banked, entropy bounded, strict ⇒ candidate, VOID on an impossible candidate floor)")


if __name__ == "__main__":
    args = parse()
    main()
    if args.dryrun:
        _dryrun_checks(args.out)
