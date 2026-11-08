"""SINKPATCH cell — one hop, parent AND child resident (same transformers pin), the arms of SPEC §2.3 on B1-beh and B3.

    python -m infra.aws_pg.jobs.sinkpatch_hop --hop H5p --out /work/out [--smoke]
    python -m infra.aws_pg.jobs.sinkpatch_hop --dryrun --out <dir>          # two stub members, no weights, no datasets

Per B1 prompt: one parent forward (block outputs at every position recorded), one child clean forward (recorded), then
every patch arm as a child forward with `PatchPositions` hooks reading from the captures; calibration arms are
transforms of the clean logits. Per wikitext window the same, with all-position logits and the window's own tokens
as targets (`sinkpatch.window_targets`). Batch 1 everywhere (the hooks address absolute positions).

Blind protocol (SPEC §9): printed = B1 discovery half and B3 fit windows 0–4; B1 confirmation and B3 scored windows
5–58 go to metrics["sealed"]; per-item / per-window values to per_item/. B3 samples are taken from the fit windows only.
B1 top-1 / entropy over the shared prefix (parity with STAGEDIFF B1-beh); B3 cross-entropy over the model's full
vocabulary (parity with STAGEDIFF B3). Gates computed here: G0 (known number), G1 (identity), G3 (prompt-independence);
G2 / G4 belong to the verdict (they need sealed values).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

MODULE = "infra.aws_pg.jobs.sinkpatch_hop"
W1_DISCOVERY_STRICT = {"C3": 28, "C4": 23, "C5": 14, "C6": 13}     # of 63: W1 bat-C*/2 discovery strict accuracy × 63 (SPEC §6 G0)
N_FIT_WINDOWS = 5
SEED = 1337


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop", default="H5p")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="12 B1 items, 7 windows of 256 tokens")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dryrun", action="store_true")
    return ap.parse_args(argv)


def entropy_nats(logits):
    import torch
    lp = torch.log_softmax(logits.float(), -1)
    return float(-(lp.exp() * lp).sum())


def b1_record(it, logits_full, tok, bos_id):
    """One B1-beh record from full-vocabulary last-position logits: top-1 / entropy over the shared prefix."""
    import torch
    from infra.pg.stagediff_models import V_SHARED
    lg = logits_full[:V_SHARED]
    probs = torch.softmax(lg, -1)
    top = int(lg.argmax())
    t_ids, f_ids = it["target_ids"], it["first_ids"]
    top5 = torch.topk(lg, 5).indices.tolist()
    return {"hash": it["hash"], "set": it["set"], "split": it["split"], "name": it["name"], "prompt": it["prompt"],
            "bos_id": bos_id, "top_id": top, "top_token": tok.decode([top]), "top_p": float(probs[top]),
            "top5": [(t, tok.decode([t]), float(probs[t])) for t in top5],
            "target": it["target"], "target_ids": t_ids, "first_ids": f_ids, "scorable": bool(t_ids),
            "entropy_prefix": entropy_nats(lg), "full_argmax_above_prefix": bool(int(logits_full.argmax()) >= V_SHARED),
            "p_target": float(probs[t_ids[0]]) if t_ids else None,
            "correct": (top in t_ids) if t_ids else None, "correct_first": (top in f_ids) if f_ids else None}


def sink_diagnostics(acts, band):
    """Position-0 state per layer: norm and the share of ‖s‖² on the three largest coordinates (description only)."""
    import torch
    s0 = acts[:, 0]                                                          # [L, d]
    n = s0.norm(dim=-1)
    top3 = torch.topk(s0.abs(), 3, dim=-1).values ** 2
    return {"norm_by_layer": n.tolist(), "top3_coord_energy_share_by_layer": (top3.sum(-1) / (n ** 2).clamp_min(1e-12)).tolist(),
            "norm_band_median": float(n[list(band)].median())}


def rel_shift(a, b):
    """‖a − b‖ / ‖b‖ per layer for [L, d] tensors."""
    return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12)).tolist()


def main(argv=None):
    a = parse(argv)
    import torch
    from infra.pg import sinkpatch as SP, stagediff_bank as SB, stagediff_items as SI, stagediff_models as SM
    from infra.aws_pg.jobs import _stagediff_shared as SH
    P_id, C_id = SM.HOPS[a.hop]
    if not a.dryrun:
        assert SM.ROSTER[P_id]["pin"] == SM.ROSTER[C_id]["pin"], (P_id, C_id, "parent and child need the same transformers pin to be co-resident")
        from infra.aws_pg.jobs._stagediff_shared import boot
        if boot(MODULE, C_id, sys.argv[1:]):
            return
    bank = SB.Bank(a.out, "sinkpatch_hop", vars(a))
    band = SM.BAND if not a.dryrun else (1, 2)
    with SH.Timer(bank, "load"):
        if a.dryrun:
            P, C = SM.stub_member(bos_id=1, seed=0), SM.stub_member(bos_id=1, seed=1)
        else:
            P, C = SM.load_ckpt(P_id, a.device, batch_size=1), SM.load_ckpt(C_id, a.device, batch_size=1)
    for m in (P, C):
        m["fwd"].batch_size = 1
    tok = C["tok"]
    assert P["bos_id"] == C["bos_id"], (P["bos_id"], C["bos_id"])
    assert P["n_lm_head_rows"] == C["n_lm_head_rows"], (P["n_lm_head_rows"], C["n_lm_head_rows"])
    if not a.dryrun:
        assert P["hashes"]["prefix_vocab_sha256"] == C["hashes"]["prefix_vocab_sha256"], "shared prefix differs"
    bank.manifest({"hop": a.hop, "parent": {k: P[k] for k in ("id", "repo", "sha", "hashes", "transformers")},
                   "child": {k: C[k] for k in ("id", "repo", "sha", "hashes", "transformers")}, "bos_id": C["bos_id"],
                   "band": [band[0], band[-1]], "arms": list(SP.ARMS), "forward": "StageForward batch 1, start token per roster, content position ids, use_cache=False; hooks on block outputs",
                   "items_sha256": SI.FILE_SHA256})
    lp, lc = P["an"]["layers"], C["an"]["layers"]
    n_layers = len(lc)

    # ---- items ---------------------------------------------------------------------------------
    beh = SI.beh_items(tok, SI.load_sets())
    if a.dryrun:
        for it in beh:
            it["target_ids"] = it["target_ids"] or SI.single_token_ids(tok, "1"); it["scorable"] = True
    if a.smoke or a.dryrun:
        beh = SI.smoke_subset(beh, 4)
    tracked = SH.tracked_hashes(beh)
    bank.put("controls", "item_counts", {"beh": len(beh), "strict": sum(1 for it in beh if it["scorable"]),
                                         "discovery": sum(1 for it in beh if it["split"] == "discovery")})

    # ---- position-0 state on the unrelated text; prompt-independence (G3) ----------------------
    with SH.Timer(bank, "sink_states"):
        _, _, aP_u = SP.capture(P["fwd"], lp, SP.UNRELATED_TEXT)
        _, _, aC_u = SP.capture(C["fwd"], lc, SP.UNRELATED_TEXT)
        _, _, aP_0 = SP.capture(P["fwd"], lp, beh[0]["prompt"])
        _, _, aC_0 = SP.capture(C["fwd"], lc, beh[0]["prompt"])
    s0P, s0C = aP_u[:, 0], aC_u[:, 0]
    g3 = {"parent_rel_diff_by_layer": rel_shift(aP_0[:, 0], s0P), "child_rel_diff_by_layer": rel_shift(aC_0[:, 0], s0C)}
    g3["max_band"] = max(max(g3["parent_rel_diff_by_layer"][L], g3["child_rel_diff_by_layer"][L]) for L in band)
    g3["pass"] = g3["max_band"] <= 0.01
    g3["rule"] = "position-0 state on a B1 prompt vs on the unrelated text: relative difference ≤ 0.01 at every band layer, both models"
    bank.put("controls", "G3_prompt_independence", g3)
    diag = {"parent": sink_diagnostics(aP_u, band), "child": sink_diagnostics(aC_u, band),
            "rel_shift_by_layer": rel_shift(s0C, s0P),
            "cos_by_layer": torch.nn.functional.cosine_similarity(s0C, s0P, dim=-1).tolist()}
    diag["rel_shift_band_median"] = float(torch.tensor(diag["rel_shift_by_layer"])[list(band)].median())
    diag["norm_ratio_band_median"] = float((s0C.norm(dim=-1) / s0P.norm(dim=-1))[list(band)].median())
    bank.put("controls", "sink_diagnostics", diag)
    torch.save({"parent_pos0": s0P.cpu(), "child_pos0": s0C.cpu(), "hop": a.hop, "text": SP.UNRELATED_TEXT},
               os.path.join(a.out, "pos0_states.pt"))

    # ---- B1-beh -------------------------------------------------------------------------------
    recs = {arm: [] for arm in SP.ARMS}
    recs["parent"] = []
    meta_items, pos1_norms, rest_med = {}, {"parent": [], "child": []}, {"parent": [], "child": []}
    logits_clean, logits_parent = [], []
    with SH.Timer(bank, "B1"):
        for i, it in enumerate(beh):
            encP, lgP, aP = SP.capture(P["fwd"], lp, it["prompt"])
            encC, lgC, aC = SP.capture(C["fwd"], lc, it["prompt"])
            assert torch.equal(encP["input_ids"].cpu(), encC["input_ids"].cpu()), "parent and child tokenise the prompt differently"
            pl, meta = SP.plans(aP, aC, s0P, band=band, n_layers=n_layers, seed_rand=SEED, seed_pos=SEED + i)
            meta_items[it["hash"]] = meta
            pos1_norms["parent"].append(aP[list(band), 1].norm(dim=-1).tolist())
            pos1_norms["child"].append(aC[list(band), 1].norm(dim=-1).tolist())
            if aP.shape[1] > 2:                      # median norm of positions ≥ 2 at the first band layer (the "massive" reference)
                rest_med["parent"].append(float(aP[band[0], 2:].norm(dim=-1).median()))
                rest_med["child"].append(float(aC[band[0], 2:].norm(dim=-1).median()))
            recs["parent"].append(b1_record(it, lgP, tok, P["bos_id"]))
            recs["clean"].append(b1_record(it, lgC, tok, C["bos_id"]))
            logits_clean.append(lgC.cpu()); logits_parent.append(lgP.cpu())
            for arm in SP.ARMS_PATCH:
                if arm not in pl:
                    continue
                with SP.PatchPositions(lc, pl[arm]) as pp:
                    lg = SP.forward_last(C["fwd"], encC)
                assert pp.n_calls == len(pl[arm]), (arm, pp.n_calls, len(pl[arm]))
                recs[arm].append({**b1_record(it, lg, tok, C["bos_id"]), "rand_position": meta["rand_position"] if arm == "posrand_parent" else None})
            del aP, aC
    # calibration on the discovery half (all prompts), applied to every prompt
    acc = SP.BiasAccumulator()
    for it, lp_, lc_ in zip(beh, logits_parent, logits_clean):
        if it["split"] == "discovery":
            acc.add(lp_, lc_)
    b1_bias = acc.bias
    fit_rows = [(lc_.unsqueeze(0), torch.tensor([it["target_ids"][0]])) for it, lc_ in zip(beh, logits_clean)
                if it["split"] == "discovery" and it["target_ids"]]
    b1_T = SP.fit_temperature(fit_rows, b1_bias)
    for it, lc_ in zip(beh, logits_clean):
        recs["calib_bias"].append(b1_record(it, lc_ + b1_bias, tok, C["bos_id"]))
        recs["calib_bias_temp"].append(b1_record(it, (lc_ + b1_bias) / b1_T["T"], tok, C["bos_id"]))
    bank.put("controls", "B1_calibration", {"fit": "discovery half, all prompts (bias); discovery strict targets (temperature)",
                                            "n_bias_rows": acc.n, "bias_norm": float(b1_bias.norm()), "bias_max_abs": float(b1_bias.abs().max()),
                                            "temperature": b1_T["T"], "nll_by_T": b1_T["nll_by_T"], "n_T_rows": b1_T["n"]})
    for arm, rr in recs.items():
        if not rr:
            continue
        SH.bank_records(bank, "B1-beh", arm, rr, tracked, sample_all=True, keys=("correct", "correct_first"))
        bank.per_item(f"B1-beh_{arm}_extra", {r["hash"]: {"entropy_prefix": r["entropy_prefix"], "p_target": r["p_target"], "top_id": r["top_id"],
                                                          "split": r["split"], "rand_position": r.get("rand_position")} for r in rr})
        disc = [r["entropy_prefix"] for r in rr if r["split"] == "discovery"]
        bank.metrics["readouts"]["B1-beh"][arm]["discovery"]["entropy_prefix_mean"] = sum(disc) / len(disc)
        bank.write()
    bank.per_item("B1_arm_meta", meta_items)
    pn = {k: torch.tensor(v) for k, v in pos1_norms.items()}                       # [n_items, band]
    massive = {k: float((pn[k][:len(rest_med[k]), 0] > 5 * torch.tensor(rest_med[k])).float().mean()) if rest_med[k] else None for k in pn}
    bank.put("controls", "pos1_norms", {k: {"band_mean_by_layer": pn[k].mean(0).tolist(), "first_band_layer_mean": float(pn[k][:, 0].mean()),
                                            "rest_median_first_band_layer_mean": (sum(rest_med[k]) / len(rest_med[k])) if rest_med[k] else None,
                                            "massive_fraction_first_band_layer": massive[k]} for k in pn}
                | {"note": "norm of the residual at absolute position 1 (first content token) on the B1 prompts, band layers; massive = position-1 norm > 5 × the median norm of positions ≥ 2 at the first band layer, per prompt"})
    # ---- gates G0, G1 on B1 (discovery-visible) -----------------------------------------------
    disc_strict = [r for r in recs["clean"] if r["split"] == "discovery" and r["correct"] is not None]
    n_right = sum(int(r["correct"]) for r in disc_strict)
    expect = W1_DISCOVERY_STRICT.get(C_id)
    g0 = {"n_right": n_right, "n": len(disc_strict), "expected_w1": expect, "pass": ((expect is None) or (abs(n_right - expect) <= 2)) if not (a.smoke or a.dryrun) else None,
          "rule": "clean strict discovery count equals the W1 count within two items (bat-C*/2; batch-1 vs batched bf16); the verdict lists per-item disagreements",
          "note": "not evaluated on smoke/dry-run" if (a.smoke or a.dryrun) else None}
    bank.put("controls", "G0_known_number", g0)
    agree = sum(1 for rc, rs in zip(recs["clean"], recs["bos_self"]) if rc["top_id"] == rs["top_id"]) / len(recs["clean"])
    ent_diff = max(abs(rc["entropy_prefix"] - rs["entropy_prefix"]) for rc, rs in zip(recs["clean"], recs["bos_self"]))
    g1 = {"b1_top1_agreement": agree, "b1_max_abs_entropy_diff": ent_diff, "n": len(recs["clean"])}
    unrel_agree = sum(1 for rc, rs in zip(recs["bos_parent"], recs["bos_parent_unrelated"]) if rc["top_id"] == rs["top_id"]) / len(recs["clean"])
    bank.put("controls", "bos_parent_vs_unrelated_top1_agreement", unrel_agree)

    # ---- B3 -----------------------------------------------------------------------------------
    with SH.Timer(bank, "B3_text"):
        if a.dryrun:
            text = " ".join(SH.fake_docs(6))
            wins = SP.window_ids(tok, text, window=24, max_tokens=24 * 8)
        else:
            text = SP.wikitext_text()
            wins = SP.window_ids(tok, text, window=256 if a.smoke else 2048, max_tokens=256 * 7 if a.smoke else 120_000)
    assert len(wins) > N_FIT_WINDOWS, len(wins)
    per_win = {arm: [] for arm in SP.ARMS}
    per_win["parent"] = []
    win_meta, win_samples = [], {arm: [] for arm in SP.ARMS + ("parent",)}
    acc3 = SP.BiasAccumulator()
    b3_bias = b3_T = None

    def score_window(w, chunk):
        win_text = tok.decode(chunk.tolist())
        encC, lo, hi, tg, off = SP.window_targets(C["fwd"], win_text)
        encP = SP.encode1(P["fwd"], win_text)
        assert torch.equal(encP["input_ids"].cpu(), encC["input_ids"].cpu())
        with SP.Recorder(lp) as rec:
            lgP = SP.forward_all(P["fwd"], encP)
        aP = rec.stack()
        with SP.Recorder(lc) as rec:
            lgC = SP.forward_all(C["fwd"], encC)
        aC = rec.stack()
        pl, meta = SP.plans(aP, aC, s0P, band=band, n_layers=n_layers, seed_rand=SEED, seed_pos=SEED + 1000 + w)
        out = {"parent": SP.nll_of(lgP, lo, hi, tg), "clean": SP.nll_of(lgC, lo, hi, tg)}
        for arm in SP.ARMS_PATCH:
            if arm not in pl:
                continue
            with SP.PatchPositions(lc, pl[arm]):
                lg = SP.forward_all(C["fwd"], encC)
            out[arm] = SP.nll_of(lg, lo, hi, tg)
            del lg
        if b3_bias is not None:
            out["calib_bias"] = SP.nll_of(lgC, lo, hi, tg, transform=SP.calib_transform(b3_bias, 1.0))
            out["calib_bias_temp"] = SP.nll_of(lgC, lo, hi, tg, transform=SP.calib_transform(b3_bias, b3_T["T"]))
        fit = (lgP[lo:hi].cpu(), lgC[lo:hi].cpu(), tg.cpu()) if w < N_FIT_WINDOWS else None
        del aP, aC, lgP, lgC
        return out, {"window": w, "n_targets": int(tg.numel()), "dropped_leading": off, "rand_position": meta["rand_position"],
                     "n_tokens": meta["n_tokens"], "last_gold": tok.decode([int(tg[-1])]), "prompt_tail": win_text[-300:]}, fit

    with SH.Timer(bank, "B3"):
        fit_rows_C = []
        for w, chunk in enumerate(wins):
            out, meta, fit = score_window(w, chunk)
            if fit is not None:
                acc3.add(fit[0], fit[1]); fit_rows_C.append((fit[1], fit[2]))
            if w == N_FIT_WINDOWS - 1:                      # fit set complete: calibration parameters, then the fit windows' calib arms
                b3_bias = acc3.bias
                b3_T = SP.fit_temperature(fit_rows_C, b3_bias)
                assert len(fit_rows_C) == len(win_meta) + 1 == N_FIT_WINDOWS
                for (lgc, tg_), meta_ in zip(fit_rows_C, win_meta + [meta]):
                    for arm, T in (("calib_bias", 1.0), ("calib_bias_temp", b3_T["T"])):
                        loss, n, top, p = SP.nll_of(lgc, 0, lgc.shape[0], tg_, transform=SP.calib_transform(b3_bias, T))
                        per_win[arm].append([loss, n])
                        win_samples[arm].append({**meta_, "nll_per_token": loss / n, "model_top_token_id": top, "model_top_token_p": p,
                                                 "model_top_token_text": tok.decode([top]), "predicted": tok.decode([top]), "gold": meta_["last_gold"],
                                                 "score_kind": "per-token NLL (not right/wrong)", "correct": None})
                fit_rows_C = []
                bank.put("controls", "B3_calibration", {"fit": f"windows 0..{N_FIT_WINDOWS - 1} (all targets)", "n_bias_rows": acc3.n,
                                                        "bias_norm": float(b3_bias.norm()), "bias_max_abs": float(b3_bias.abs().max()),
                                                        "temperature": b3_T["T"], "nll_by_T": b3_T["nll_by_T"], "n_T_rows": b3_T["n"]})
            for arm, (loss, n, top, p) in out.items():
                per_win[arm].append([loss, n])
                if w < N_FIT_WINDOWS:
                    win_samples[arm].append({**meta, "nll_per_token": loss / n, "model_top_token_id": top, "model_top_token_p": p,
                                             "model_top_token_text": tok.decode([top]), "predicted": tok.decode([top]), "gold": meta["last_gold"],
                                             "score_kind": "per-token NLL (not right/wrong)", "correct": None})
            win_meta.append(meta)
            if w % 10 == 0:
                bank.event("B3-progress", window=w, of=len(wins))
    for arm, rows in per_win.items():
        if not rows:
            continue
        assert len(rows) == len(wins), (arm, len(rows), len(wins))
        bank.per_item(f"B3_{arm}", {"per_window": rows, "fit_windows": list(range(N_FIT_WINDOWS)), "scored_windows": list(range(N_FIT_WINDOWS, len(wins)))})
        fitv = [l / n for l, n in rows[:N_FIT_WINDOWS]]
        scv = [l / n for l, n in rows[N_FIT_WINDOWS:]]
        bank.metrics["readouts"].setdefault("B3", {})[arm] = {"fit_windows": {"mean_nll": sum(fitv) / len(fitv), "n_windows": len(fitv)}}
        bank.metrics["sealed"].setdefault("B3", {})[arm] = {"scored": {"mean_nll": sum(scv) / len(scv), "n_windows": len(scv), "ppl": math.exp(sum(scv) / len(scv))}}
        bank.write()
        ss = sorted(win_samples.get(arm, []), key=lambda s: s["nll_per_token"])
        for s, tag in ([(ss[0], "best-scoring fit window")] if ss else []) + ([(ss[-1], "worst-scoring fit window")] if len(ss) > 1 else []):
            bank.sample("B3", arm, {**s, "selection": tag})
    bank.per_item("B3_window_meta", win_meta)
    # G1 on B3 (all windows: the identity arm reveals nothing about the loss)
    d = [abs(a_[0] / a_[1] - b_[0] / b_[1]) for a_, b_ in zip(per_win["clean"], per_win["bos_self"])]
    g1.update({"b3_mean_abs_nll_diff": sum(d) / len(d), "b3_max_abs_nll_diff": max(d), "n_windows": len(d)})
    g1["pass"] = g1["b1_top1_agreement"] >= 0.98 and g1["b3_mean_abs_nll_diff"] <= 0.005
    g1["rule"] = "bos_self vs clean: B1 top-1 agreement ≥ 0.98 and B3 mean per-window |ΔNLL| ≤ 0.005 nats"
    bank.put("controls", "G1_identity", g1)
    if not g1["pass"]:
        bank.stamp_void("patched_arms", f"G1 identity failed: {g1}")
    if g0["pass"] is False:
        bank.stamp_void("cell", f"G0 known number failed: {g0}")
    bank.top("status", "done")
    bank.event("done", out=a.out, G0=g0["pass"], G1=g1["pass"], G3=g3["pass"])


def _dryrun_checks(out: str):
    import json as _j
    m = _j.load(open(os.path.join(out, "metrics.json")))
    assert m["status"] == "done" and m["controls"]["G1_identity"]["pass"] and m["controls"]["G3_prompt_independence"]["pass"], m["controls"]
    assert set(m["readouts"]["B1-beh"]) >= set(("clean",) + tuple(a for a in ("bos_parent", "posrand_parent", "all_parent", "calib_bias")))
    assert set(m["sealed"]["B3"]) >= {"clean", "parent", "bos_parent", "calib_bias_temp"}
    for arm in ("clean", "bos_parent", "all_parent", "calib_bias"):
        assert os.path.exists(os.path.join(out, "samples", f"B1-beh__{arm}.jsonl")) and os.path.exists(os.path.join(out, "samples", f"B3__{arm}.jsonl")), arm
        pi = _j.load(open(os.path.join(out, "per_item", f"B3_{arm}.json")))
        assert len(pi["per_window"]) == len(pi["fit_windows"]) + len(pi["scored_windows"])
    b1c = _j.load(open(os.path.join(out, "per_item", "B1-beh_clean.json")))
    b1s = _j.load(open(os.path.join(out, "per_item", "B1-beh_bos_self.json")))
    assert b1c == b1s, "identity arm differs from clean on the stub"
    assert "entropy_prefix_mean" in m["readouts"]["B1-beh"]["clean"]["discovery"]
    assert m["controls"]["B3_calibration"]["temperature"] in [round(0.5 + 0.05 * i, 2) for i in range(31)]
    print("PASS sinkpatch_hop dry-run: metrics/per_item/samples for every arm, identity exact, gates stamped")


if __name__ == "__main__":
    _a = parse()
    main()
    if _a.dryrun:
        _dryrun_checks(_a.out)
