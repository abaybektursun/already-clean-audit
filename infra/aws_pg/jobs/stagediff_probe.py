"""STAGEDIFF review probes (2026-09-05, after the W4 unblind; fresh prefixes results/stagediff/w5/<cell>/<n>).

Sub-probes, each independent, all banked under out/ with per-item vectors and samples:
  --bos          B1-beh under three start-token conventions: none, <|im_end|> (151645, the Molmo processors' BOS),
                 <|endoftext|> (151643); plus the fraction of prompts whose FULL-vocabulary argmax lies above the shared
                 prefix (action / added tokens) — a slice that hides such a prediction inflates the child.
  --lens-sizes   J-lens fit convergence: band pass@k of the item set under the banked lens_16 / lens_32 / lens_64 of this
                 checkpoint's own fit (C0: half a's 16 and 32, merged a+b as 64), the logit lens, and (--released, C0 only)
                 the Neuronpedia 479-prompt Qwen3-4B lens as a cross-checkpoint reading.
  --hop H --parent-fit … --child-fit …   I2 and TRACE redone with the position-0 term removed: δ̄ pooled over content
                 positions 1..4, the position-0 direction alone, the child's MEAN-activation direction (μ_C, positions
                 1..4) as the matched control, two random-difference draws; B1-beh and B2 under each projection. TRACE
                 top tokens banked per (layer, position) for δ̄, μ_C and the null under both lenses.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

MODULE = "infra.aws_pg.jobs.stagediff_probe"


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bos", action="store_true")
    ap.add_argument("--lens-sizes", action="store_true")
    ap.add_argument("--fit", default=None, help="S3 prefix (…/out) of this checkpoint's fit; two comma-separated for C0 (a,b)")
    ap.add_argument("--released", action="store_true")
    ap.add_argument("--hop", default=None)
    ap.add_argument("--parent-fit", default=None)
    ap.add_argument("--child-fit", default=None)
    ap.add_argument("--n-rand", type=int, default=2)
    ap.add_argument("--i2-matched", action="store_true", help="plan E3: energy-matched I2 directions instead of the position split")
    ap.add_argument("--n-em", type=int, default=4, help="energy-matched random draws")
    ap.add_argument("--em-target", choices=["asrun", "perp"], default="asrun", help="E3: match δ̄ as run (mean family); E3b: match δ̄ ⊥ μ_C (top-PC family)")
    ap.add_argument("--n-pcs", type=int, default=3, help="E3b: number of centred PCs spanning the control's base direction")
    ap.add_argument("--em-base", default="span", help="E3b/E3c/E3d control bases, comma list of span|pc1|orth|mu")
    ap.add_argument("--em-match", default="layer", help="E3b/E3c matching, comma list of layer|band")
    ap.add_argument("--trace-fullvocab", action="store_true", help="plan E3: TRACE with μ_P, δ̄ ⊥ μ_C, full-vocabulary relevance")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dryrun", action="store_true")
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
    from infra.aws_pg.jobs.stagediff_pair import load_side
    bank = SB.Bank(a.out, "stagediff_probe", vars(a))
    with SH.Timer(bank, "load"):
        member = SM.stub_member(bos_id=1, seed=1) if a.dryrun else SM.load_ckpt(a.ckpt, a.device)
    tok = member["tok"]
    lm = SL.AnatomyLensModel(member)
    band = SM.BAND if not a.dryrun else (1, 2)
    bank.manifest({"ckpt": member["id"], "repo": member["repo"], "repo_sha": member["sha"], "hashes": member["hashes"], "bos_id": member["bos_id"]})
    sets = SI.load_sets()
    beh, ints = SI.beh_items(tok, sets), SI.int_items(tok, sets)
    ld = None
    if a.dryrun:
        from infra.pg.matrix_items import fake_load_dataset as ld
        for it in beh:
            it["target_ids"] = it["target_ids"] or SI.single_token_ids(tok, "1"); it["scorable"] = True
    mm = SI.mmlu_cloze_items("discovery", ld) + SI.mmlu_cloze_items("confirmation", ld)
    if a.smoke or a.dryrun:
        beh, ints = SI.smoke_subset(beh, 10), SI.smoke_subset(ints, 6)
        mm = SI.smoke_subset([m for m in mm if m["split"] == "discovery"], 1)[:12] + SI.smoke_subset([m for m in mm if m["split"] == "confirmation"], 1)[:12]
    tracked, tracked_mm = SH.tracked_hashes(beh), SH.tracked_hashes(mm, 1)
    Fwd = type(member["fwd"])

    # ---- BOS conventions + full-vocabulary argmax --------------------------------------------
    if a.bos:
        variants = {"none": None, "im_end": SM.MOLMO_BOS, "endoftext": 151643} if not a.dryrun else {"none": None, "one": 1, "two": 2}
        for name, bid in variants.items():
            f = Fwd(member["model"], tok, member["fwd"].device, bos_id=bid); f.batch_size = member["fwd"].batch_size
            keep = member["fwd"], member["bos_id"]
            member["fwd"], member["bos_id"] = f, bid
            with SH.Timer(bank, f"bos_{name}"):
                rec = SH.score_b1(member, beh)
                full = f.next_token_logits([it["prompt"] for it in beh])         # full vocabulary, no slice
                above = (full.argmax(-1) >= SM.V_SHARED).float()
                top_full = full.argmax(-1).tolist()
            SH.bank_records(bank, "B1-beh", f"bos_{name}", rec, tracked, sample_all=True, keys=("correct", "correct_first"))
            bank.put("controls", f"fullvocab_argmax_above_prefix_bos_{name}",
                     {"fraction": float(above.mean()), "n": int(above.numel()),
                      "per_half": {h: float(above[[i for i, it in enumerate(beh) if it["split"] == h]].mean()) for h in ("discovery", "confirmation")},
                      "examples": [{"hash": it["hash"], "prompt": it["prompt"][-40:], "full_argmax_id": t, "token": tok.decode([t])}
                                   for it, t, ab in zip(beh, top_full, above.tolist()) if ab][:5]})
            member["fwd"], member["bos_id"] = keep

    # ---- J-lens fit-size convergence -----------------------------------------------------------
    if a.lens_sizes:
        lenses = {}
        if a.dryrun:
            from infra.pg.stagediff_lens import fit_corpus, fit_lens
            seqs = fit_corpus(tok, n_seqs=4, seq_len=40, seed=0, docs=SH.fake_docs(20))["seqs"]
            l, _ = fit_lens(lm, seqs, dim_batch=4, ckpt_path=os.path.join(a.out, "ck.pt"), sizes=(2,), save_dir=a.out)
            lenses = {"lens_2": SL.load_lens(os.path.join(a.out, "lens_2.pt")), "lens_4": l}
        else:
            parts = a.fit.split(",")
            for i, prefix in enumerate(parts):
                SH.s3_fetch(prefix, os.path.join(a.out, "inputs", "fit", str(i)), ["lens_16.pt", "lens_32.pt", "lens.pt"])
            d0 = os.path.join(a.out, "inputs", "fit", "0")
            lenses["lens_16"] = SL.load_lens(os.path.join(d0, "lens_16.pt"))
            lenses["lens_32"] = SL.load_lens(os.path.join(d0, "lens_32.pt"))
            lenses["lens_64"] = SL.merge_lenses([os.path.join(a.out, "inputs", "fit", str(i), "lens.pt") for i in range(len(parts))])
            if a.released:
                from infra.aws_pg.jobs.stagediff_preflight import released_lens
                lenses["released_qwen3_4b_base_479"], meta = released_lens(a.out)
                bank.put("controls", "released_lens", {"hf_model_name": meta["hf_model_name"], "n_prompts": lenses["released_qwen3_4b_base_479"].n_prompts})
        for l in lenses.values():
            SL.lens_to(l, lm.device)
        with SH.Timer(bank, "lens_sizes"):
            acts = SL.item_acts(lm, ints)
            out = {}
            for name, l in list(lenses.items()) + [("logit_lens", None)]:
                prof = SL.rank_profile(lm, ints, acts, lens=l)
                bank.per_item(f"B1-int_{name}", prof)
                out[name] = {half: {f"pass@{k}": SL.pass_at_k(prof, band, k, half)["overall"] for k in (1, 10, 25)} for half in ("discovery", "confirmation")}
                out[name]["n_prompts"] = getattr(l, "n_prompts", None)
                out[name]["layer_curve_k10_all"] = SL.layer_curve(prof, 10)
        bank.put("readouts", "lens_sizes", out)

    # ---- I2 / TRACE without position 0 --------------------------------------------------------
    if a.hop:
        if not a.dryrun:
            from infra.aws_pg.jobs.stagediff_pair import fetch_inputs
            fetch_inputs(a)
        else:
            from infra.aws_pg.jobs import stagediff_fit
            for side, seed in (("parent", 0), ("child", 1)):
                d = os.path.join(a.out, "inputs", side, "0")
                if not os.path.exists(os.path.join(d, "lens.pt")):
                    stagediff_fit.main(["--dryrun", "--out", d, "--stub-seed", str(seed)])
        lens_P, tbP, nbP, _ = load_side(a.out, "parent")
        lens_C, tbC, nbC, _ = load_side(a.out, "child")
        SL.lens_to(lens_C, lm.device)
        AP, AC = tbP["acts"].float(), tbC["acts"].float()                 # [n, L, 5, d]
        dbar = AC.mean(0) - AP.mean(0)                                     # [L, 5, d]
        muC = AC[:, :, 1:].mean((0, 2))                                    # child mean activation, positions 1..4
        rd = SV.norm_match(SV.random_diffs(nbP["acts"], a.n_rand, seed=0), dbar[:, 1:].mean(1))
        if a.i2_matched:      # plan E3 / E3b: energy-matched controls (2026-09-05)
            dirs, em = matched_directions(AP, AC, dbar, nbP, band, a.n_em, a.n_rand, em_target=a.em_target, n_pcs=a.n_pcs,
                                          em_bases=tuple(a.em_base.split(",")), em_matches=tuple(a.em_match.split(",")))
            bank.put("controls", "I2_matched", em)
            if "families" in em:      # review-4 D4: stamp VOID per family, so a cell with one matched and one failed family says both
                for fam, info in em["families"].items():
                    if not info["matched"]:
                        bank.stamp_void(f"I2_matched:{fam}", f"energy match failed on more than half the band layers: {info['unmatched_layers']}")
            elif not em["matched_ok"]:
                bank.stamp_void("I2_matched", f"energy match failed on more than half the band layers: {em['em_unmatched_layers']}")
        else:
            dirs = {"I2_pos1to4": SV.unit_rows(dbar[:, 1:].mean(1)), "I2_pos0only": SV.unit_rows(dbar[:, 0]),
                    "I2_meanC": SV.unit_rows(muC), "I2_pos0to4_asrun": SV.unit_rows(dbar.mean(1))}
            for k in range(a.n_rand):
                dirs[f"I2rand{k}"] = SV.unit_rows(rd)[k]
            cosm = {n: float(torch.nn.functional.cosine_similarity(u[list(band)], dirs["I2_meanC"][list(band)], dim=-1).mean()) for n, u in dirs.items()}
            bank.put("controls", "I2_directions", {"cos_with_meanC_band": cosm,
                                                   "delta_norm_by_pos_L20": dbar[20 if not a.dryrun else 1].norm(dim=-1).tolist(),
                                                   "pos0_energy_share_band": float(sum((dbar[L, 0].norm() ** 2) / (dbar[L].norm(dim=-1) ** 2).sum() for L in band) / len(band))})
        for name, u in dirs.items():
            with SV.ProjectOut(lm.layers, {L: u[L] for L in band}):
                with SH.Timer(bank, f"B1-beh_{name}"):
                    SH.bank_records(bank, "B1-beh", name, SH.score_b1(member, beh), tracked, sample_all=True, keys=("correct", "correct_first"))
                with SH.Timer(bank, f"B2_{name}"):
                    SH.bank_records(bank, "B2", name, SH.score_b2(member, mm), tracked_mm)
        with SH.Timer(bank, "B1-beh_clean"):
            SH.bank_records(bank, "B1-beh", "clean", SH.score_b1(member, beh), tracked, sample_all=True, keys=("correct", "correct_first"))
        with SH.Timer(bank, "B2_clean"):
            SH.bank_records(bank, "B2", "clean", SH.score_b2(member, mm), tracked_mm)
        robot = set(SI.robot_vocab_ids(tok)["ids"])
        if a.trace_fullvocab:     # plan E3: μ_P beside μ_C, δ̄ ⊥ μ_C, relevance over the full vocabulary with the added rows counted
            perp = dirs["I2_perp_muC"] if a.i2_matched else SV.unit_rows(dbar.mean(1))
            vecs = {"delta": dbar, "perp_muC": perp.unsqueeze(1).expand(-1, 5, -1), "meanC": AC.mean(0), "meanP": AP.mean(0),
                    "null0": rd[0].unsqueeze(1).expand(-1, 5, -1)}
            bank.put("readouts", "TRACE_fullvocab", trace_fullvocab(lm, tok, vecs, robot, band, member["n_lm_head_rows"]))
        else:
            # TRACE tokens per (layer, position) under both lenses, for δ̄, μ_C and one null draw
            layers = [L for L in (12, 16, 20, 24, 28, 31) if L in lens_C.jacobians] if not a.dryrun else [1, 2]
            tr = {}
            for vname, V in (("delta", dbar), ("meanC", muC.unsqueeze(1).expand(-1, 5, -1)), ("null0", rd[0].unsqueeze(1).expand(-1, 5, -1))):
                for lname, ln in (("jlens", lens_C), ("logitlens", None)):
                    cells = {}
                    for L in layers:
                        for j in range(5):
                            lg = SL.lens_logits(lm, V[L, j].to(lm.device), L, ln)
                            ids = SL.topk_ids(lg, 10)
                            cells[f"{L},{j}"] = {"tokens": [tok.decode([i]) for i in ids], "relevance": SL.relevance(ids, robot)}
                    tr[f"{vname}_{lname}"] = cells
            best14 = {ln: max((c["relevance"], k) for k, c in tr[f"delta_{ln}"].items() if int(k.split(",")[1]) >= 1) for ln in ("jlens", "logitlens")}
            bank.put("readouts", "TRACE_probe", {"cells": tr, "best_relevance_positions1to4": {ln: {"relevance": v[0], "cell": v[1]} for ln, v in best14.items()}})
    bank.top("status", "done")
    bank.event("done", out=a.out)


# ------------------------------------------------------------------------------------- E3 helpers
def energy_fraction(H, u) -> float:
    """Share of Σ‖h‖² that projecting the unit direction u out of every row of H [m, d] removes."""
    return float(((H @ u) ** 2).mean() / (H.norm(dim=-1) ** 2).mean())


def match_theta(H, mu, r, target: float, iters: int = 40):
    """θ ∈ [0, π/2] with energy_fraction(H, cos θ·mu + sin θ·r) = target (mu, r orthonormal). The fraction falls from
    f(mu) to f(r) as θ grows; a target above f(mu) has no solution → (0.0, False): the registered failing input."""
    f = lambda th: energy_fraction(H, math.cos(th) * mu + math.sin(th) * r)   # noqa: E731
    if f(0.0) < target:
        return 0.0, False
    lo, hi = 0.0, math.pi / 2
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    th = (lo + hi) / 2
    tol = min(0.05, max(0.01, 0.25 * target))      # absolute 0.05 for large targets; 25 % relative for small ones (E3b, target ≈ 0.07)
    return th, abs(f(th) - target) <= tol


def matched_directions(AP, AC, dbar, nbP, band, n_em: int, n_rand: int, seed: int = 0, em_target: str = "asrun", n_pcs: int = 3,
                       em_bases=("span",), em_matches=("layer",)):
    """Unit directions per layer for the energy-matched I2 (plan E3): δ̄ as run; δ̄ ⊥ μ_C; μ_C (all positions and
    1..4); μ_P; PC1 of the child's centred activations (positions 1..4); n_em energy-matched random directions; n_rand
    plain ADL random differences. Two families (plan E3 / E3b, 2026-09-05):
      em_target="asrun": v = cos θ·μ̂_C + sin θ·r̂, θ per layer so the removed energy equals δ̄'s (the mean-direction family);
      em_target="perp":  v = cos φ·p̂ + sin φ·r̂ with p̂ a random unit vector in the span of the top-`n_pcs` centred PCs
                         (μ̂_C component removed) and r̂ random ⊥ (μ̂_C, p̂), φ per layer so the removed energy equals
                         δ̄ ⊥ μ_C's — the structured control for a direction that carries energy through the other
                         high-variance axes rather than through the mean.
    Returns (dirs, meta) with per-direction removed-energy fractions and cosines banked."""
    import torch
    from infra.pg import stagediff_interv as SV
    cos = torch.nn.functional.cosine_similarity
    band = list(band)
    L_all, d = dbar.shape[0], dbar.shape[-1]
    asrun = SV.unit_rows(dbar.mean(1))
    muC_all, muC_14, muP_all = SV.unit_rows(AC.mean((0, 2))), SV.unit_rows(AC[:, :, 1:].mean((0, 2))), SV.unit_rows(AP.mean((0, 2)))
    perp = SV.unit_rows(asrun - (asrun * muC_all).sum(-1, keepdim=True) * muC_all)
    pc1, pcs = torch.zeros(L_all, d), {}
    for L in band:
        X = AC[:, L, 1:].reshape(-1, d)
        X = X - X.mean(0, keepdim=True)
        Vh = torch.linalg.svd(X, full_matrices=False).Vh
        pc1[L] = Vh[0]; pcs[L] = Vh[:n_pcs]
    ref = asrun if em_target == "asrun" else perp
    target = {L: energy_fraction(AC[:, L].reshape(-1, d), ref[L]) for L in band}
    g = torch.Generator().manual_seed(seed)
    rd = SV.unit_rows(SV.norm_match(SV.random_diffs(nbP["acts"], n_rand, seed=0), dbar.mean(1)))
    meta = {"em_target": em_target, "n_pcs": n_pcs, f"target_energy_fraction_by_layer_{em_target}": target, "per_direction": {}}
    if em_target == "asrun":
        ems, thetas, unmatched = [], {}, {}
        for k in range(n_em):
            v = torch.zeros(L_all, d); thetas[k] = {}; unmatched[k] = []
            for L in band:
                H, mu = AC[:, L].reshape(-1, d), muC_all[L]
                r = torch.randn(d, generator=g); r = r - (r @ mu) * mu; r = r / r.norm()
                th, ok = match_theta(H, mu, r, target[L])
                v[L] = math.cos(th) * mu + math.sin(th) * r; thetas[k][L] = th
                if not ok:
                    unmatched[k].append(L)
            ems.append(v)
        dirs = {"I2_asrun": asrun, "I2_perp_muC": perp, "I2_muC_all": muC_all, "I2_muC_1to4": muC_14, "I2_muP_all": muP_all, "I2_pc1": pc1}
        for k in range(n_em):
            dirs[f"I2_em{k}"] = ems[k]
        meta.update({"em_theta_by_layer": thetas, "em_unmatched_layers": unmatched})
    else:
        # E3b/E3c (2026-09-05): control families for δ̄ ⊥ μ_C. Base per layer:
        #   span — a random direction in the top-n_pcs centred-PC span (μ̂_C removed)           [E3b; could not reach the target]
        #   pc1  — the mean-removed PC1 of the centred positions 1..4                             [director's suggestion]
        #   orth — the maximum-energy axis orthogonal to BOTH μ̂_C and δ̄ ⊥ μ_C (all positions)     [a control that is not δ̄ ⊥ μ_C]
        # Match per layer (θ_L so the removed energy equals the target where the base carries ≥ it) or per band (one θ so
        # the band-mean removed energy equals the band-mean target). A family whose base cannot reach the target is banked
        # as unmatched, never silently interpolated.
        #   mu   — μ̂_C itself (energy 0.8–0.9 everywhere), with r̂ ⊥ (μ̂_C, δ̄ ⊥ μ_C): the control is orthogonal to δ̄ ⊥ μ_C by
        #          construction and the downward interpolation reaches any target                        [review-4, E3d]
        # NOTE the "orth" base is constrained orthogonal to BOTH μ̂_C and δ̄ ⊥ μ_C (review-4 D2: a design choice, stated here).
        families = {}
        target_band = sum(target.values()) / len(target)
        bases = {}
        for L in band:
            H, mu, p = AC[:, L].reshape(-1, d), muC_all[L], perp[L]
            b = {}
            w = torch.randn(n_pcs, generator=g)
            s_ = (w @ pcs[L]); s_ = s_ - (s_ @ mu) * mu; b["span"] = s_ / s_.norm()
            p1 = pcs[L][0] - (pcs[L][0] @ mu) * mu; b["pc1"] = p1 / p1.norm()
            Q = torch.eye(d) - torch.outer(mu, mu) - torch.outer(p, p)
            HP = H @ Q
            b["orth"] = torch.linalg.eigh(HP.T @ HP / H.shape[0]).eigenvectors[:, -1]
            b["mu"] = mu
            bases[L] = b
        dirs = {"I2_asrun": asrun, "I2_perp_muC": perp}
        for bname in em_bases:
            for match in em_matches:
                fam = f"{bname}{'L' if match == 'layer' else 'B'}"
                info = {"base": bname, "match": match, "base_energy_by_layer": {L: energy_fraction(AC[:, L].reshape(-1, d), bases[L][bname]) for L in band},
                        "cos_base_perp_by_layer": {L: float(abs(bases[L][bname] @ perp[L])) for L in band}, "theta": {}, "unmatched_layers": {},
                        "band_target": target_band, "max_abs_layer_deviation": {}, "r_orthogonal_to_perp": bname == "mu",
                        "constraints": {"span": "⊥ μ̂_C", "pc1": "⊥ μ̂_C", "orth": "⊥ μ̂_C and ⊥ δ̄ ⊥ μ_C (top eigenvector after projecting both out)", "mu": "μ̂_C itself; r̂ ⊥ μ̂_C and ⊥ δ̄ ⊥ μ_C"}[bname]}
                vs = []
                for k in range(n_em):
                    v = torch.zeros(L_all, d)
                    rs = {}
                    for L in band:
                        mu, base, p = muC_all[L], bases[L][bname], perp[L]
                        r = torch.randn(d, generator=g); r = r - (r @ mu) * mu; r = r - (r @ base) * base
                        if bname == "mu":
                            r = r - (r @ p) * p
                        r = r / r.norm(); rs[L] = r
                    if match == "layer":
                        info["theta"][k], info["unmatched_layers"][k] = {}, []
                        for L in band:
                            th, ok = match_theta(AC[:, L].reshape(-1, d), bases[L][bname], rs[L], target[L])
                            v[L] = math.cos(th) * bases[L][bname] + math.sin(th) * rs[L]; info["theta"][k][L] = th
                            if not ok:
                                info["unmatched_layers"][k].append(L)
                    else:
                        th, ok = match_theta_band({L: AC[:, L].reshape(-1, d) for L in band}, {L: bases[L][bname] for L in band}, rs, target_band)
                        for L in band:
                            v[L] = math.cos(th) * bases[L][bname] + math.sin(th) * rs[L]
                        info["theta"][k] = th; info["unmatched_layers"][k] = [] if ok else list(band)
                    info["max_abs_layer_deviation"][k] = max(abs(energy_fraction(AC[:, L].reshape(-1, d), v[L]) - target[L]) for L in band)
                    dirs[f"I2_emp_{fam}{k}"] = v
                    vs.append(v)
                info["matched"] = all(len(info["unmatched_layers"][k]) <= len(band) // 2 for k in range(n_em))
                # an unmatched family sits at θ = 0 on every draw and its draws coincide: count the distinct ones (review-4 D6)
                info["n_distinct_draws"] = len({tuple(torch.round(vv[list(band)[0]][:8] * 1e4).tolist()) for vv in vs})
                families[fam] = info
        meta["families"] = families
        meta["em_unmatched_layers"] = {fam: f_["unmatched_layers"] for fam, f_ in families.items()}
    for k in range(n_rand):
        dirs[f"I2rand{k}"] = rd[k]
    for name, u in dirs.items():
        fa = {L: energy_fraction(AC[:, L].reshape(-1, d), u[L]) for L in band}
        f14 = {L: energy_fraction(AC[:, L, 1:].reshape(-1, d), u[L]) for L in band}
        meta["per_direction"][name] = {"energy_frac_band_mean_allpos": sum(fa.values()) / len(fa), "energy_frac_band_mean_pos1to4": sum(f14.values()) / len(f14),
                                       "energy_frac_by_layer_allpos": fa,
                                       "cos_muC_all_band": float(cos(u[band], muC_all[band], dim=-1).mean()),
                                       "cos_delta_band": float(cos(u[band], asrun[band], dim=-1).mean()),
                                       "cos_perp_band": float(cos(u[band], perp[band], dim=-1).abs().mean())}
    if em_target == "asrun":
        meta["matched_ok"] = all(len(meta["em_unmatched_layers"][k]) <= len(band) // 2 for k in range(n_em))
    else:
        meta["matched_ok"] = any(f_["matched"] for f_ in meta["families"].values())
    return dirs, meta


def match_theta_band(Hs: dict, bases: dict, rs: dict, target_band: float, iters: int = 40):
    """One θ for the whole band so that the band-mean removed energy of cos θ·base_L + sin θ·r_L equals `target_band`.
    (0.0, False) when even θ = 0 (the bases themselves) falls short — the registered failing input of the band match."""
    def f(th):
        return sum(energy_fraction(Hs[L], math.cos(th) * bases[L] + math.sin(th) * rs[L]) for L in Hs) / len(Hs)
    if f(0.0) < target_band:
        return 0.0, False
    lo, hi = 0.0, math.pi / 2
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) > target_band:
            lo = mid
        else:
            hi = mid
    th = (lo + hi) / 2
    tol = min(0.05, max(0.01, 0.25 * target_band))
    return th, abs(f(th) - target_band) <= tol


def trace_fullvocab(lm, tok, vectors: dict, robot_prefix: set, band, n_rows: int, k: int = 25) -> dict:
    """Logit-lens read-out (top-k) of each [L, 5, d] vector at every band (layer, position): relevance over the shared
    prefix (ROBOT_VOCAB) and over the FULL vocabulary with every row above the shared prefix counted as robot/added
    vocabulary; μ_C − μ_P relevance per cell so a robot word in the child's mean counts only if absent from the parent's."""
    import torch
    an = lm.an
    W = an["lm_head"].weight
    from infra.pg.stagediff_models import V_SHARED
    added = set(range(V_SHARED, n_rows))
    out = {"n_added_rows": len(added), "k": k, "cells": {}, "best": {}}
    for vname, V in vectors.items():
        cells, best_p, best_f = {}, (0.0, None), (0.0, None)
        for L in band:
            for j in range(V.shape[1]):
                with torch.inference_mode():
                    lg = an["lm_head"](an["norm"](V[L, j].to(W.device, W.dtype))).float()
                top_full = lg.topk(k).indices.tolist()
                top_pref = lg[:V_SHARED].topk(k).indices.tolist()
                rel_p = sum(1 for i in top_pref if i in robot_prefix) / k
                rel_f = sum(1 for i in top_full if (i in robot_prefix or i in added)) / k
                cells[f"{L},{j}"] = {"relevance_prefix": rel_p, "relevance_full": rel_f, "n_added_in_top": sum(1 for i in top_full if i in added),
                                     "tokens_full": [tok.decode([i]) for i in top_full[:10]]}
                if rel_p > best_p[0]:
                    best_p = (rel_p, f"{L},{j}")
                if rel_f > best_f[0]:
                    best_f = (rel_f, f"{L},{j}")
        out["cells"][vname] = cells
        out["best"][vname] = {"prefix": {"relevance": best_p[0], "cell": best_p[1]}, "full": {"relevance": best_f[0], "cell": best_f[1]}}
    if "meanC" in out["cells"] and "meanP" in out["cells"]:
        diff = {c: out["cells"]["meanC"][c]["relevance_prefix"] - out["cells"]["meanP"][c]["relevance_prefix"] for c in out["cells"]["meanC"]}
        out["meanC_minus_meanP_prefix"] = {"cells": diff, "max": max(diff.values()), "argmax": max(diff, key=diff.get)}
    return out


if __name__ == "__main__":
    main()
