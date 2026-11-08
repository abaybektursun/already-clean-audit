"""read-from-passage control (prereg 2026-06-20) — is the deployed-policy "deletion" a gone FACT or a
dead READER?

PCR found π0/π0.5 floor BOTH readouts on MMLU, AND that a matched-random-ΔW decoder (generic damage of
π0's magnitude) floors them too. So "floors everything" is what generic damage looks like, and PCR
cannot yet tell its "erasure" apart from a backbone that can no longer read text. This job runs the
IDENTICAL cloze / MCQ-letter readout on a task whose answer is IN the prompt and needs NO stored fact
(fictional entities), under each policy's OWN-EMBED Gemma-1 shell, against {base CEILING, random-decoder
FLOOR, matched-random-ΔW generic-damage null}, and recomputes the MMLU-recall floor in-run.

  read-from-passage ABOVE FLOOR (CI excludes FLOOR and the matched-random-ΔW null), MMLU at FLOOR
        ⇒ H_reader-intact : the reader works; PCR's deletion is knowledge-specific → headline LOCKS.
  read-from-passage AT FLOOR
        ⇒ H_reader-dead   : the backbone cannot read; PCR measured a dead reader → headline RETRACTS.
  read-from-passage above FLOOR but below the CEILING band
        ⇒ H_partial-reader: reading degraded not dead → headline QUALIFIES.

Reuses the ENTIRE PCR orchestration (own-embed shell, matched-random-ΔW null, paired item-bootstrap,
discovery/confirmation split, own-vs-base-embed control, label-shuffle) verbatim; the only NEW science is
the read-from-passage item set + readout pair (infra.pg.read_from_passage) and `reader_verdict` below.

Firewall (CLAUDE.md): Gemma-1 (google/gemma-2b, 18L) ONLY. Every baseline recomputed in-run; PCR cited
only via its frozen artifact, never transported. Fail-loud: any mismatch RAISES.

  python -m infra.aws_pg.jobs.read_from_passage_control --out <dir> \
      --checkpoints base,floor,pi0,pi05,pi0_baseembed,pi05_baseembed,matched_dw \
      --n 2000 --mmlu-recall base,floor,pi0,pi05 --nopass base,pi0
"""
from __future__ import annotations

import argparse
import json
import pathlib

# REUSED verbatim from PCR (verified @ 87a1a65) — the pure analysis core + the model builders.
from infra.aws_pg.jobs.policy_content_readout import (
    cloze_acc_lambda, label_shuffle_acc, matched_dw_decoder,
    paired_item_bootstrap, split_masks, _extract_own, _build_ownembed_model,
)
from infra.pg import read_from_passage as RFP

PI0_REPO = "lerobot/pi0_base"
PI05_REPO = "lerobot/pi05_base"

# Controls before policies; matched_dw last (needs base+pi0 decoders staged). base-embed arms are the
# own-vs-base-embed interface control (prereg §5).
ARM_ORDER = ("base", "floor", "pi0", "pi05", "pi0_baseembed", "pi05_baseembed", "matched_dw")
CEILING_ARM = "base"
FLOOR_ARM = "floor"
BASEEMBED_OF = {"pi0_baseembed": "pi0", "pi05_baseembed": "pi05"}


def reader_verdict(name, rfp_acc, rfp_ci, floor, ceiling, matched_dw_acc,
                   mmlu_at_floor=None):
    """Frozen §6 decision rule, applied per policy on the headline (confirmation) half.
      H_reader-intact : rfp CI lo > FLOOR AND > matched-random-ΔW AND rfp closer to CEILING than FLOOR
                        (the reader works → PCR deletion is knowledge-specific; LOCKS, esp. with MMLU@floor).
      H_reader-dead   : rfp CI lo ≤ FLOOR (at the floor → the backbone cannot read; PCR headline RETRACTS).
      H_partial-reader: rfp CI lo > FLOOR & > null but rfp ≤ midpoint (reads, degraded; QUALIFIES).
      INDETERMINATE   : rfp CI lo > FLOOR but ≤ the matched-random-ΔW null (above floor, not above generic
                        damage — cannot separate a live reader from generic damage; report raw, no relabel).
    rfp_ci = [mean, lo95, hi95] of raw read-from-passage cloze accuracy (item-bootstrap)."""
    mid = 0.5 * (ceiling + floor)
    ci_lo = rfp_ci[1]
    excludes_floor = ci_lo > floor
    beats_generic = ci_lo > matched_dw_acc
    closer_to_ceiling = rfp_acc > mid
    if not excludes_floor:
        verdict = "H_reader-dead (read-from-passage at FLOOR → backbone cannot read text; PCR headline RETRACTS to 'no text ability survives')"
    elif not beats_generic:
        verdict = "INDETERMINATE (above FLOOR but not above the matched-random-ΔW generic-damage null — cannot separate a live reader from generic damage; report raw, do not relabel)"
    elif closer_to_ceiling:
        verdict = "H_reader-intact (read-from-passage above FLOOR and the generic-damage null, closer to CEILING → the reader works; PCR's deletion is knowledge-specific → headline LOCKS)"
    else:
        verdict = "H_partial-reader (read-from-passage above FLOOR/null but below the CEILING band → reading degraded not dead → headline QUALIFIES)"
    return {"name": name, "verdict": verdict, "rfp_cloze_acc": rfp_acc, "rfp_cloze_ci": rfp_ci,
            "floor": floor, "ceiling": ceiling, "midpoint": mid, "matched_dw_acc": matched_dw_acc,
            "excludes_floor": bool(excludes_floor), "beats_generic_damage": bool(beats_generic),
            "closer_to_ceiling": bool(closer_to_ceiling), "mmlu_at_floor": mmlu_at_floor}


def rfp_vocab_check(tok, items):
    """Prereg §13 gate 2 (read-from-passage form): every choice text tokenizes into shared base-vocab
    rows (<256000) under the base gemma-2b tokenizer. Weight-free. RAISES on any ≥256000."""
    max_id, n = 0, 0
    for it in items:
        for c in it["choices"]:
            ids = tok.encode(" " + str(c).strip(), add_special_tokens=False)
            max_id = max(max_id, max(ids))
            n += 1
    assert max_id < 256000, f"vocab-parity FAIL: a choice routed through extended-vocab row {max_id}"
    return {"max_token_id": max_id, "n_candidates_checked": n, "shared_vocab_ceiling": 256000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints",
                    default="base,floor,pi0,pi05,pi0_baseembed,pi05_baseembed,matched_dw")
    ap.add_argument("--n", type=int, default=2000, help="synthetic read-from-passage item count")
    ap.add_argument("--synth-seed", type=int, default=20260620)
    ap.add_argument("--mmlu-recall", default="base,floor,pi0,pi05",
                    help="comma list of arms to also score on MMLU cloze (the dissociation floor side); '' to skip")
    ap.add_argument("--mmlu-n-per-subject", type=int, default=0, help="0 = full MMLU test split")
    ap.add_argument("--nopass", default="base,pi0",
                    help="comma list of arms to also score with passages REMOVED (leakage control); '' to skip")
    ap.add_argument("--floor-seed", type=int, default=1337)
    ap.add_argument("--matched-dw-seed", type=int, default=20260620)
    ap.add_argument("--shuffle-seed", type=int, default=4242)
    ap.add_argument("--boot-seed", type=int, default=1337)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--theta", type=float, default=0.10)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    requested = [c.strip() for c in a.checkpoints.split(",") if c.strip()]
    arms = [x for x in ARM_ORDER if x in requested] + [x for x in requested if x not in ARM_ORDER]
    mmlu_arms = [c.strip() for c in a.mmlu_recall.split(",") if c.strip()]
    nopass_arms = [c.strip() for c in a.nopass.split(",") if c.strip()]

    # ---- read-from-passage item set (deterministic, content-hashed) ---------
    items, exemplars, synth_meta = RFP.build_synth(a.synth_seed, a.n)
    print(f"[RFP] synth n={len(items)} content_sha1={synth_meta['content_sha1'][:12]} "
          f"exemplars={len(exemplars)}")

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from infra.pg import text_eval as TE
    from infra.pg import vlat_swapback as SB
    from infra.pg.entities import ENTITIES
    from infra.aws_pg.jobs.vlat_retention import _bare_decoder, _random_decoder_sd

    assert torch.cuda.is_available(), "read-from-passage control needs GPU"
    device = "cuda"
    n_per = a.mmlu_n_per_subject or None
    L0g1 = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(L0g1.repo_id, revision=L0g1.revision)
    print(f"[RFP] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} arms={arms}")

    # ---- gate 2 (vocab parity) — weight-free, before any forward ------------
    vp = rfp_vocab_check(tok, items)
    print(f"[RFP] vocab-parity OK: max_token_id={vp['max_token_id']} < 256000 (n={vp['n_candidates_checked']})")

    rfp_cz, rfp_mc = {}, {}        # name -> read-from-passage cloze / mcq eval dict
    rfp_cz_nopass = {}             # name -> no-passage cloze eval dict (leakage control)
    mmlu_cz = {}                   # name -> MMLU cloze eval dict (dissociation floor side)
    meta, stash = {}, {}
    for name in arms:
        if name == "base":
            model = AutoModelForCausalLM.from_pretrained(
                L0g1.repo_id, revision=L0g1.revision, torch_dtype=torch.bfloat16,
                device_map=device, attn_implementation="eager").eval()
            stash["base_dec"] = {k: v.detach().cpu().clone()
                                 for k, v in _bare_decoder(model.state_dict()).items()}
            meta[name] = {"readout": "base-embed (own); read-from-passage CEILING"}
        elif name == "floor":
            b = AutoModelForCausalLM.from_pretrained(
                L0g1.repo_id, revision=L0g1.revision, torch_dtype=torch.bfloat16, device_map="cpu")
            dec = _random_decoder_sd(_bare_decoder(b.state_dict()), a.floor_seed)
            del b
            model, info = SB.load_swapback_model(dec, device)
            meta[name] = {"readout": "random-decoder + base-embed; read-from-passage FLOOR", **info}
        elif name == "matched_dw":
            assert "base_dec" in stash and "pi0_dec" in stash, \
                "matched_dw needs base + pi0 decoders staged first (request base,pi0 before matched_dw)"
            mdec, frob = matched_dw_decoder(stash["base_dec"], stash["pi0_dec"], a.matched_dw_seed)
            model, info = SB.load_swapback_model(mdec, device)
            tot = sum(v * v for v in frob.values()) ** 0.5
            meta[name] = {"readout": "base-decoder + π0-norm-matched random ΔW + base-embed (generic-damage null)",
                          "matched_dw_seed": a.matched_dw_seed, "total_frob_dW": tot,
                          "n_matched_matrices": len(frob), **info}
            print(f"[RFP] matched_dw: total ‖ΔW‖_F={tot:.3f} over {len(frob)} matrices")
        elif name in BASEEMBED_OF:
            pol = BASEEMBED_OF[name]
            dec, _own_head, sha = _extract_own(pol, device)
            model, info = SB.load_swapback_model(dec, device)
            meta[name] = {"readout": f"{pol} decoder + BASE-embed (own-vs-base-embed equivalence control)",
                          "source_sha": sha, **info}
        else:                                    # pi0 / pi05 under their OWN embed
            dec, own_head, sha = _extract_own(name, device)
            if name == "pi0":
                stash["pi0_dec"] = {k: v.detach().cpu().clone() for k, v in dec.items()}
            model, info = _build_ownembed_model(dec, own_head, device)
            meta[name] = {"readout": "own-embed (tied lm_head)", "source_sha": sha, **info}
            print(f"[RFP] {name}: own-embed shell vocab={info['vocab']} decoder swapped={info['n_decoder_keys_swapped']}")

        # ---- read-from-passage axes on the SAME model instance, SAME items --
        cz = RFP.passage_cloze(model, tok, device, items, exemplars)
        mc = RFP.passage_choice_logits(model, tok, device, items, exemplars)
        assert cz["n"] == mc["n"] == len(items), f"{name}: rfp item-count mismatch"
        assert cz["per_item_hash"] == mc["per_item_hash"], f"{name}: rfp cloze/mcq order mismatch"
        rfp_cz[name], rfp_mc[name] = cz, mc
        line = f"[RFP] {name:14s} rfp_cloze={cz['acc_pct']:.2f}  rfp_mcq={mc['acc_pct']:.2f}"
        if name in nopass_arms:
            czn = RFP.passage_cloze(model, tok, device, items, exemplars, drop_passage=True)
            rfp_cz_nopass[name] = czn
            line += f"  rfp_cloze_NOPASS={czn['acc_pct']:.2f}"
        if name in mmlu_arms:
            mm = TE.mmlu_cloze(model, tok, device, n_per_subject=n_per)
            mmlu_cz[name] = mm
            line += f"  MMLU_cloze={mm['acc_pct']:.2f} (n={mm['n']})"
        meta[name]["rfp_cloze_acc_pct"] = cz["acc_pct"]
        meta[name]["rfp_mcq_acc_pct"] = mc["acc_pct"]
        print(line)
        del model
        torch.cuda.empty_cache()

    # ---- position-pairing tripwire across ALL arms (hash sequence equality) --
    ref_hash = rfp_cz[arms[0]]["per_item_hash"]
    for name in arms:
        assert rfp_cz[name]["per_item_hash"] == ref_hash, f"{name}: rfp hash sequence differs (reorder!)"
    N = len(ref_hash)
    disc, conf = split_masks(ref_hash)
    print(f"[RFP] N={N} items; discovery={sum(disc)} confirmation={sum(conf)}")

    def mask(arr, m):
        return np.asarray(arr, dtype=int)[np.asarray(m, dtype=bool)]

    # ---- per-item 0/1 arrays (position-aligned) -----------------------------
    rfp_item = {name: cloze_acc_lambda(rfp_cz[name]["cloze_logprobs"], rfp_cz[name]["cloze_ntoks"],
                                       rfp_cz[name]["correct_idx"], lam=1.0) for name in arms}
    rfp_mcq_item = {name: np.asarray(rfp_mc[name]["per_item"], dtype=int) for name in arms}

    result = {"torch": torch.__version__, "n_items": N, "arms": arms,
              "synth_meta": {k: synth_meta[k] for k in ("seed", "n", "n_exemplars", "content_sha1", "exemplar_sha1")},
              "seeds": {"synth": a.synth_seed, "floor": a.floor_seed, "matched_dw": a.matched_dw_seed,
                        "shuffle": a.shuffle_seed, "boot": a.boot_seed},
              "vocab_parity": vp, "meta": meta,
              "acc_pct": {"rfp_cloze": {n: rfp_cz[n]["acc_pct"] for n in arms},
                          "rfp_mcq": {n: rfp_mc[n]["acc_pct"] for n in arms},
                          "rfp_cloze_nopass": {n: rfp_cz_nopass[n]["acc_pct"] for n in rfp_cz_nopass},
                          "mmlu_cloze": {n: mmlu_cz[n]["acc_pct"] for n in mmlu_cz}}}

    if CEILING_ARM in arms and FLOOR_ARM in arms:
        cz_ceiling = float(rfp_item[CEILING_ARM].mean())
        cz_floor = float(rfp_item[FLOOR_ARM].mean())
        span = cz_ceiling - cz_floor
        boot_span = paired_item_bootstrap(
            {"cz_ceiling": rfp_item[CEILING_ARM], "cz_floor": rfp_item[FLOOR_ARM]},
            [("ceiling_minus_floor", "cz_ceiling", "cz_floor")], n_boot=a.n_boot, seed=a.boot_seed)
        span_ci = boot_span["contrast"]["ceiling_minus_floor"]
        # §13 gate 1: base CEILING must be clearly above the FLOOR (power to detect a live reader)
        adequate = span_ci[1] > 0.20
        result["rfp_dynamic_range"] = {"ceiling": cz_ceiling, "floor": cz_floor, "span": span,
                                       "span_ci": span_ci, "adequate": bool(adequate),
                                       "note": ("read-from-passage has power" if adequate else
                                                "DEGENERATE: base CEILING not clearly above FLOOR — re-tune item difficulty on discovery (prereg §13.1)")}
        print(f"[RFP] dynamic range: CEILING={cz_ceiling:.4f} FLOOR={cz_floor:.4f} span={span:.4f} "
              f"CI={span_ci[1]:.4f}..{span_ci[2]:.4f} adequate={adequate}")

        # ---- label-shuffle validity control on the CEILING (prereg §5) -------
        sh_ceiling, _ = label_shuffle_acc(rfp_cz[CEILING_ARM]["cloze_logprobs"],
                                          rfp_cz[CEILING_ARM]["cloze_ntoks"],
                                          rfp_cz[CEILING_ARM]["correct_idx"], a.shuffle_seed)
        cloze_axis_valid = abs(sh_ceiling - 0.25) < 0.05
        result["label_shuffle"] = {"ceiling_shuffled_acc": sh_ceiling, "chance": 0.25,
                                   "cloze_axis_valid": bool(cloze_axis_valid)}
        print(f"[RFP] label-shuffle CEILING acc={sh_ceiling:.4f} (chance 0.25) valid={cloze_axis_valid}")

        # ---- no-passage leakage gate (prereg §5/§13.4) -----------------------
        if CEILING_ARM in rfp_cz_nopass:
            np_item = cloze_acc_lambda(rfp_cz_nopass[CEILING_ARM]["cloze_logprobs"],
                                       rfp_cz_nopass[CEILING_ARM]["cloze_ntoks"],
                                       rfp_cz_nopass[CEILING_ARM]["correct_idx"], lam=1.0)
            np_acc = float(np_item.mean())
            mid = 0.5 * (cz_ceiling + cz_floor)
            leaks = np_acc > mid                  # base answers WITHOUT the passage → items leak priors
            result["no_passage_leakage"] = {"base_nopass_acc": np_acc, "base_withpass_acc": cz_ceiling,
                                            "floor": cz_floor, "midpoint": mid, "leaks": bool(leaks),
                                            "note": ("CLEAN: base drops to FLOOR without the passage → answer needs reading, not priors"
                                                     if not leaks else "LEAK: base answers without the passage → items leak world knowledge (prereg §13.4: drop)")}
            print(f"[RFP] no-passage: base_nopass={np_acc:.4f} vs withpass={cz_ceiling:.4f} floor={cz_floor:.4f} leaks={leaks}")

        # ---- matched-random-ΔW generic-damage null on read-from-passage ------
        matched_dw_acc = float(rfp_item["matched_dw"].mean()) if "matched_dw" in arms else cz_floor
        result["matched_dw_rfp"] = {"rfp_cloze_acc": matched_dw_acc, "floor": cz_floor,
                                    "note": "the generic-damage null a live reader must exceed (prereg §6)"}

        # ---- MMLU-recall floor side of the dissociation (recomputed in-run) --
        mmlu_floor = float(np.asarray(mmlu_cz[FLOOR_ARM]["per_item"]).mean()) if FLOOR_ARM in mmlu_cz else None
        mmlu_ceiling = float(np.asarray(mmlu_cz[CEILING_ARM]["per_item"]).mean()) if CEILING_ARM in mmlu_cz else None
        mmlu_mid = 0.5 * (mmlu_ceiling + mmlu_floor) if (mmlu_floor is not None and mmlu_ceiling is not None) else None
        if mmlu_mid is not None:
            result["mmlu_recall"] = {"ceiling": mmlu_ceiling, "floor": mmlu_floor, "midpoint": mmlu_mid,
                                     "per_arm_acc": {n: float(np.asarray(mmlu_cz[n]["per_item"]).mean()) for n in mmlu_cz}}

        # ---- per-policy reader verdict on the headline (confirmation) half ---
        result["policies"] = {}
        for name in ("pi0", "pi05"):
            if name not in arms:
                continue
            block = {}
            policy_mmlu = float(np.asarray(mmlu_cz[name]["per_item"]).mean()) if name in mmlu_cz else None
            mmlu_at_floor = (policy_mmlu <= mmlu_mid) if (policy_mmlu is not None and mmlu_mid is not None) else None
            for half, m in (("confirmation", conf), ("discovery", disc), ("full", [True] * N)):
                p_item = mask(rfp_item[name], m)
                f_item = mask(rfp_item[FLOOR_ARM], m)
                ce_item = mask(rfp_item[CEILING_ARM], m)
                mdw_item = mask(rfp_item["matched_dw"], m) if "matched_dw" in arms else f_item
                bt = paired_item_bootstrap(
                    {"rfp_policy": p_item, "rfp_floor": f_item, "rfp_ceiling": ce_item, "rfp_mdw": mdw_item},
                    [("rfp_minus_floor", "rfp_policy", "rfp_floor"),
                     ("rfp_minus_mdw", "rfp_policy", "rfp_mdw"),
                     ("ceiling_minus_rfp", "rfp_ceiling", "rfp_policy")],
                    n_boot=a.n_boot, seed=a.boot_seed)
                v = reader_verdict(name, float(p_item.mean()), bt["acc"]["rfp_policy"],
                                   float(f_item.mean()), float(ce_item.mean()), float(mdw_item.mean()),
                                   mmlu_at_floor=mmlu_at_floor)
                v["bootstrap"] = bt
                v["n"] = int(len(p_item))
                v["rfp_mcq_acc"] = float(mask(rfp_mcq_item[name], m).mean())
                v["policy_mmlu_recall_acc"] = policy_mmlu
                block[half] = v
            result["policies"][name] = block
            head = block["confirmation"]
            print(f"[RFP] {name} HEADLINE(confirmation, n={head['n']}): rfp_cloze={head['rfp_cloze_acc']:.4f} "
                  f"CI={head['rfp_cloze_ci'][1]:.4f}..{head['rfp_cloze_ci'][2]:.4f} "
                  f"mmlu={policy_mmlu} mmlu_at_floor={mmlu_at_floor}\n       → {head['verdict']}")

        # ---- own-vs-base-embed read-from-passage equivalence (prereg §5) -----
        if any(b in arms for b in BASEEMBED_OF):
            result["own_vs_base_embed_rfp"] = {}
            for be, pol in BASEEMBED_OF.items():
                if be not in arms or pol not in arms:
                    continue
                own, base_e = rfp_item[pol], rfp_item[be]
                bt = paired_item_bootstrap({"own": own, "base_embed": base_e},
                                           [("base_minus_own", "base_embed", "own")], n_boot=a.n_boot, seed=a.boot_seed)
                diff_ci = bt["contrast"]["base_minus_own"]
                artifact = diff_ci[1] > 0.03
                result["own_vs_base_embed_rfp"][pol] = {
                    "own_embed_rfp": float(own.mean()), "base_embed_rfp": float(base_e.mean()),
                    "base_minus_own_ci": diff_ci, "interface_artifact": bool(artifact),
                    "note": ("INTERFACE ARTIFACT: base-embed recovers read-from-passage the own-embed suppressed"
                             if artifact else "EQUIVALENT: own-embed ≈ base-embed → a read-from-passage result is genuine, not an interface artifact")}
                print(f"[RFP] own-vs-base {pol}: own={float(own.mean()):.4f} base-embed={float(base_e.mean()):.4f} artifact={artifact}")

    # ---- artifact registry (prereg §10) -------------------------------------
    result["registry"] = {
        "read_from_passage_control.json": "this file — generated by infra.aws_pg.jobs.read_from_passage_control",
        "axes": {"rfp_cloze": "read_from_passage.passage_cloze (text-only, λ=1)",
                 "rfp_mcq": "read_from_passage.passage_choice_logits",
                 "mmlu_cloze": "text_eval.mmlu_cloze (dissociation floor side)"},
        "pairing": "by POSITION; per_item_hash sequence asserted equal across all arms",
        "headline": "confirmation half (is_confirmation(hash)); discovery + full also banked",
        "tag": "CONFIRMATORY",
    }
    (out / "read_from_passage_control.json").write_text(json.dumps(result, indent=2, default=str))
    banks = {name: {"per_item_hash": rfp_cz[name]["per_item_hash"],
                    "correct_idx": rfp_cz[name]["correct_idx"],
                    "rfp_cloze_logprobs": rfp_cz[name]["cloze_logprobs"],
                    "rfp_cloze_ntoks": rfp_cz[name]["cloze_ntoks"],
                    "rfp_cloze_per_item": [int(x) for x in rfp_item[name]],
                    "rfp_mcq_choice_logits": rfp_mc[name]["choice_logits"],
                    "rfp_mcq_per_item": rfp_mc[name]["per_item"]} for name in arms}
    for name in mmlu_cz:                          # MMLU per-item for the dissociation figure (F2) CIs
        banks[name]["mmlu_cloze_per_item"] = mmlu_cz[name]["per_item"]
        banks[name]["mmlu_correct_idx"] = mmlu_cz[name]["correct_idx"]
    for name in rfp_cz_nopass:                    # no-passage per-item (leakage figure / audit)
        banks[name]["rfp_cloze_nopass_per_item"] = [
            int(x) for x in cloze_acc_lambda(rfp_cz_nopass[name]["cloze_logprobs"],
                                             rfp_cz_nopass[name]["cloze_ntoks"],
                                             rfp_cz_nopass[name]["correct_idx"], lam=1.0)]
    (out / "read_from_passage_control_banks.json").write_text(json.dumps(banks, default=str))
    # the read-from-passage item set itself (reproducibility)
    (out / "read_from_passage_items.json").write_text(json.dumps(
        {"items": items, "exemplars": exemplars, "meta": synth_meta}, default=str))
    print("[RFP] DONE", json.dumps({"rfp_cloze": result["acc_pct"]["rfp_cloze"],
                                    "mmlu_cloze": result["acc_pct"].get("mmlu_cloze", {})}))


if __name__ == "__main__":
    main()
