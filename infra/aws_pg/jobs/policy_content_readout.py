"""policy-content-readout (prereg 2026-06-19) — does the deployed-policy "deletion" survive a CLOZE read?

Every readout ever run on the π0/π0.5 *policies* (VLAT, OWN-EMBED) is the restricted A/B/C/D letter-logit
MCQ argmax. REPRICE established on the firewalled Gemma-2 hop that this exact letter axis collapses to
chance while a full-answer CLOZE content read RETAINS most of the capability. So the program's "genuine
deletion on the deployed policy" rests on the one readout shown (elsewhere) to read "deleted" while
content is intact — and the cloze counterpart on the policy is UNRUN. This job runs it.

Design (MDE, prereg §3): read π0/π0.5 under their OWN tied lm_head-as-embed Gemma-1 shell and score BOTH
  (Z) cloze content   — text_eval.mmlu_cloze   (rank the 4 candidate TEXTS, choices hidden)
  (M) MCQ letter      — text_eval.mmlu_choice_logits (rank the 4 letters)  ── REUSED, paired by POSITION
against {base gemma-2b CEILING, random-decoder FLOOR, matched-random-ΔW readout-hardness null}.

  cloze near CEILING while MCQ at FLOOR ⇒ H_head-on-policy: the §10 "deletion" headline collapses to
                                          "letter readout died, content survived" (REPRICE on the policy).
  cloze ALSO at FLOOR                   ⇒ H_genuine-deletion: a real single-hop-vs-deployed dissociation.
  matched-random-ΔW reproduces the split⇒ H_readout-hardness: the dissociation is generic, not knowledge.

Firewall (prereg §0): Gemma-1 (google/gemma-2b, 18L) ONLY. Every baseline recomputed in-run; NEVER
transport REPRICE's Gemma-2 cloze number as a threshold. Faithful: own-embed was co-trained with the
decoder (not a proxy). MCQ anchors (recompute, never transcribe): BASE 40.97, FLOOR 24.33, θ=0.10 from
VLAT @ c9c8773. Discipline (CLAUDE.md): no try/except hiding — any mismatch RAISES with a trace.

  python -m infra.aws_pg.jobs.policy_content_readout --out <dir> \
      --checkpoints base,floor,pi0,pi05,matched_dw [--mmlu-n-per-subject N (0=full)]
"""
from __future__ import annotations

import argparse
import json
import pathlib

PI0_REPO = "lerobot/pi0_base"
PI05_REPO = "lerobot/pi05_base"

# Canonical arm order (controls before the policies; matched_dw last — it needs base+pi0 decoders).
# pi0_baseembed/pi05_baseembed = the policy DECODER read under BASE embed (the prereg §5/§8 own-vs-base-embed
# cloze equivalence control — rules out an own-embed interface artifact as the cause of cloze-at-floor).
ARM_ORDER = ("base", "floor", "pi0", "pi05", "pi0_baseembed", "pi05_baseembed", "matched_dw")
CEILING_ARM = "base"
FLOOR_ARM = "floor"
BASEEMBED_OF = {"pi0_baseembed": "pi0", "pi05_baseembed": "pi05"}


# ===========================================================================
# Pure analysis core (weight-free; unit-tested by pcr_dryrun.py). Heavy imports
# live inside each fn so this module py_compiles on a torch-free control plane.
# ===========================================================================
def cloze_acc_lambda(cloze_logprobs, cloze_ntoks, correct_idx, lam=1.0):
    """Length-normalized cloze accuracy: argmax_c (Σlogp_c / ntok_c**λ) == gold. λ=1 = per-token mean
    (the banked default). Pure function of the banked (Σlogprob, ntok) — any λ is a recompute."""
    import numpy as np
    lp = np.asarray(cloze_logprobs, dtype=float)          # (N, 4)
    nt = np.asarray(cloze_ntoks, dtype=float)             # (N, 4)
    ci = np.asarray(correct_idx, dtype=int)               # (N,)
    assert lp.shape == nt.shape and lp.shape[0] == ci.shape[0], "cloze arrays misaligned"
    normed = lp / np.power(nt, lam)
    pred = normed.argmax(axis=1)
    return (pred == ci).astype(int)                        # (N,) per-item 0/1


def label_shuffle_acc(cloze_logprobs, cloze_ntoks, correct_idx, seed, lam=1.0):
    """Structure-collapse control (prereg §5): reassign each item's gold to a RANDOM candidate slot
    (independent of content), recompute cloze accuracy. Must collapse to chance (≈1/4) — proving the
    readout tracks the question→answer-CONTENT mapping, not a length/frequency/position artifact."""
    import numpy as np
    per_item = cloze_acc_lambda(cloze_logprobs, cloze_ntoks, correct_idx, lam=lam)  # validates shapes
    lp = np.asarray(cloze_logprobs, dtype=float)
    nt = np.asarray(cloze_ntoks, dtype=float)
    rng = np.random.default_rng(seed)
    shuffled_gold = rng.integers(0, 4, lp.shape[0])        # gold ⟂ content → E[acc]=1/4 exactly
    pred = (lp / np.power(nt, lam)).argmax(axis=1)
    return float((pred == shuffled_gold).mean()), per_item


def matched_dw_decoder(base_dec, target_dec, seed):
    """The readout-hardness null decoder (prereg §5): base decoder + a Gaussian ΔW matched in SHAPE and
    per-matrix FROBENIUS NORM to the target's (π0's) REAL decoder ΔW. Tests whether a generic decoder
    change of the SAME magnitude as π0's adaptation reproduces cloze-spared/MCQ-floored. 1-D norms kept
    at base (matching the FLOOR convention). Distinct from the FLOOR, which REPLACES the decoder."""
    import torch
    from infra.pg import gain_routing as GR
    g = torch.Generator().manual_seed(seed)
    out, frob = {}, {}
    for k, w0 in base_dec.items():
        if w0.ndim == 2:
            assert k in target_dec, f"matched_dw: base decoder key {k!r} absent from target decoder"
            dW = target_dec[k].detach().to(torch.float64) - w0.detach().to(torch.float64)
            fn = float(dW.norm().item())
            delta = GR.matched_random_delta(tuple(w0.shape), fn, g, dtype=torch.float64)
            out[k] = (w0.detach().to(torch.float64) + delta).to(w0.dtype)
            frob[k] = fn
        else:
            out[k] = w0.detach().clone()
    assert frob, "matched_dw: no 2-D decoder matrices found"
    return out, frob


def paired_item_bootstrap(per_item_arrays, contrasts, n_boot=1000, seed=1337):
    """Paired nonparametric item-bootstrap (prereg §7): resample item indices ONCE per draw, apply to
    every arm/axis (all position-aligned), recompute. Returns raw-accuracy CIs per key + contrast CIs.
    `per_item_arrays`: {key: 0/1 array, length N}. `contrasts`: [(name, plus_key, minus_key), ...]."""
    import numpy as np
    keys = list(per_item_arrays)
    arrs = {k: np.asarray(v, dtype=float) for k, v in per_item_arrays.items()}
    n = arrs[keys[0]].size
    for k in keys:
        assert arrs[k].size == n, f"bootstrap arrays misaligned: {k} has {arrs[k].size} vs {n}"
    rng = np.random.default_rng(seed)
    acc_draws = {k: [] for k in keys}
    con_draws = {name: [] for name, _, _ in contrasts}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means = {k: float(arrs[k][idx].mean()) for k in keys}
        for k in keys:
            acc_draws[k].append(means[k])
        for name, plus, minus in contrasts:
            con_draws[name].append(means[plus] - means[minus])

    def ci(xs):
        xs = np.asarray(xs)
        return [float(xs.mean()), float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]

    return {"acc": {k: ci(acc_draws[k]) for k in keys},
            "contrast": {name: ci(con_draws[name]) for name, _, _ in contrasts},
            "n_boot": n_boot, "seed": seed, "n_items": int(n)}


def split_masks(per_item_hash):
    """Discovery/confirmation split (prereg §7) by is_discovery(hash)=int(h,16)%2==0. Position-stable
    (hash is deterministic per item). Returns (disc_mask, conf_mask) as boolean lists."""
    from infra.pg import w16_probes as WP
    disc = [WP.is_discovery(h) for h in per_item_hash]
    conf = [WP.is_confirmation(h) for h in per_item_hash]
    assert all(d ^ c for d, c in zip(disc, conf)), "discovery/confirmation not a partition"
    return disc, conf


def policy_verdict(name, mcq_S, cloze_acc, cloze_ci, ceiling_cloze, floor_cloze,
                   matched_dw_cloze_retained, theta_mcq=0.10):
    """Frozen §6 decision rule, applied per policy on the headline (confirmation) half.
      H_head-on-policy : MCQ S ≤ θ (floored) AND cloze closer to CEILING than FLOOR AND boot-CI lo
                         excludes the FLOOR (content retained) AND matched-random-ΔW does NOT reproduce
                         a cloze-retained pattern.
      H_genuine        : cloze at the FLOOR (CI excludes CEILING-side retention) — content also gone.
      H_readout-hardness: matched-random-ΔW reproduces cloze-retained → dissociation not knowledge-specific.
    cloze_ci = [mean, lo95, hi95] of raw cloze accuracy (item-bootstrap)."""
    mid = 0.5 * (ceiling_cloze + floor_cloze)
    mcq_floored = mcq_S <= theta_mcq
    closer_to_ceiling = cloze_acc > mid
    ci_excludes_floor = cloze_ci[1] > floor_cloze          # bootstrap lo above the cloze FLOOR
    ci_excludes_ceiling_retention = cloze_ci[2] < mid      # bootstrap hi below the midpoint (at FLOOR side)
    if matched_dw_cloze_retained:
        verdict = "H_readout-hardness (matched-random-ΔW reproduces cloze-spared/MCQ-floored → not knowledge-specific)"
    elif mcq_floored and closer_to_ceiling and ci_excludes_floor:
        verdict = "H_head-on-policy (MCQ-letter floored, CLOZE CONTENT RETAINED → §10 headline collapses)"
    elif ci_excludes_ceiling_retention:
        verdict = "H_genuine-deletion (cloze ALSO at FLOOR → content gone at the policy)"
    else:
        verdict = "INDETERMINATE (cloze CI straddles FLOOR↔midpoint — report raw accuracy, do not relabel)"
    return {"name": name, "verdict": verdict, "mcq_S_text": mcq_S, "mcq_floored": bool(mcq_floored),
            "cloze_acc": cloze_acc, "cloze_ci": cloze_ci, "cloze_floor": floor_cloze,
            "cloze_ceiling": ceiling_cloze, "cloze_midpoint": mid,
            "closer_to_ceiling": bool(closer_to_ceiling), "ci_excludes_floor": bool(ci_excludes_floor)}


# ===========================================================================
# Model builders (REUSED verbatim from ownembed_retention; matched_dw is NEW).
# ===========================================================================
def _own_head_key(keys):
    cands = [k for k in keys if "lm_head" in k and "gemma_expert" not in k
             and "action_expert" not in k and "vision_tower" not in k]
    assert len(cands) >= 1, f"no backbone lm_head key found (keys e.g. {list(keys)[:4]})"
    return sorted(cands, key=len)[0]


def _extract_own(name, device):
    """(bare_decoder_sd, own_head_tensor, sha) for π0/π0.5 — decoder via pi0_port.remap, own embed via
    the tied lm_head. Verbatim from ownembed_retention._extract_own."""
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    from safetensors.torch import load_file
    from infra.pg import pi0_port
    api = HfApi()
    repo = PI0_REPO if name == "pi0" else PI05_REPO
    if name == "pi0":
        sd = load_file(hf_hub_download(repo, "model.safetensors"))
    else:
        sd = {}
        for f in [x for x in list_repo_files(repo) if x.endswith(".safetensors")]:
            sd.update(load_file(hf_hub_download(repo, f)))
    dec = pi0_port.remap_backbone_lm(sd)
    head = sd[_own_head_key(sd.keys())]
    return dec, head, api.model_info(repo, revision="main").sha


def _build_ownembed_model(decoder_sd, own_head, device):
    """The own-embed shell (verbatim from ownembed_retention): base Gemma-1 resized to the own-embed
    vocab, own tied embed/head attached, decoder swapped in (leak-gated). Returns (model, info)."""
    import torch
    from transformers import AutoModelForCausalLM
    from infra.pg.entities import ENTITIES
    from infra.pg import vlat_swapback as SB

    ent = ENTITIES["L0_g1"]
    m = AutoModelForCausalLM.from_pretrained(
        ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    V = int(own_head.shape[0])
    m.resize_token_embeddings(V)
    with torch.no_grad():
        m.get_input_embeddings().weight.copy_(own_head.to(device).to(torch.bfloat16))
    m.tie_weights()
    base_sd = dict(m.state_dict())
    dec = SB.prefix_decoder_sd({k: v.to(device) for k, v in decoder_sd.items()})
    new_sd, info = SB.build_swapback_sd(base_sd, dec)
    missing, unexpected = m.load_state_dict(new_sd, strict=True)
    assert not missing and not unexpected, f"own-embed load not strict-clean: {missing} {unexpected}"
    return m, {**info, "vocab": V}


def vocab_parity_gate(tok, n_per_subject=2):
    """Prereg §13 gate 2: every cloze candidate text must tokenize into shared base-vocab rows (<256000)
    under the base gemma-2b tokenizer, so they resolve correctly under the 257152 own-embed shell.
    Cheap, weight-free. Returns the max token id seen. RAISES if any candidate routes through ≥256000."""
    from collections import defaultdict
    from datasets import load_dataset
    from infra.pg.text_eval import MMLU_REPO
    ds = load_dataset(MMLU_REPO, "all")
    by_subj = defaultdict(list)
    for r in ds["test"]:
        by_subj[r["subject"]].append(r)
    max_id, n_checked = 0, 0
    for subj in sorted(by_subj):
        rows = by_subj[subj][:n_per_subject] if n_per_subject else by_subj[subj]
        for r in rows:
            for c in r["choices"]:
                ids = tok.encode(" " + str(c).strip(), add_special_tokens=False)
                max_id = max(max_id, max(ids))
                n_checked += 1
    assert max_id < 256000, f"vocab-parity FAIL: a cloze candidate routed through extended-vocab row {max_id} (≥256000)"
    return {"max_token_id": max_id, "n_candidates_checked": n_checked, "shared_vocab_ceiling": 256000}


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", default="base,floor,pi0,pi05,matched_dw")
    ap.add_argument("--mmlu-n-per-subject", type=int, default=0, help="0 = full split (n≈14,042)")
    ap.add_argument("--floor-seed", type=int, default=1337)
    ap.add_argument("--matched-dw-seed", type=int, default=20260619)
    ap.add_argument("--shuffle-seed", type=int, default=4242)
    ap.add_argument("--boot-seed", type=int, default=1337)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--theta", type=float, default=0.10, help="VLAT θ_forget (inherited, MCQ axis)")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    requested = [c.strip() for c in a.checkpoints.split(",") if c.strip()]
    # canonical order (matched_dw last; it needs base+pi0 decoders staged)
    arms = [x for x in ARM_ORDER if x in requested] + [x for x in requested if x not in ARM_ORDER]

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from infra.pg import text_eval as TE
    from infra.pg import vlat_swapback as SB
    from infra.pg.entities import ENTITIES
    from infra.aws_pg.jobs.vlat_retention import _bare_decoder, _random_decoder_sd

    assert torch.cuda.is_available(), "policy-content-readout needs GPU"
    device = "cuda"
    n_per = a.mmlu_n_per_subject or None
    L0g1 = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(L0g1.repo_id, revision=L0g1.revision)  # base gemma-2b (rows <256000)
    print(f"[PCR] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} arms={arms} n_per={n_per}")

    # ---- gate 2 (vocab parity) — weight-free, before any forward ------------
    vp = vocab_parity_gate(tok, n_per_subject=(n_per or 2))
    print(f"[PCR] vocab-parity OK: max_token_id={vp['max_token_id']} < 256000 (n={vp['n_candidates_checked']})")

    mcq, cloze = {}, {}     # name -> eval dict
    meta = {}
    stash = {}              # base/pi0 bare decoders for matched_dw
    for name in arms:
        if name == "base":
            model = AutoModelForCausalLM.from_pretrained(
                L0g1.repo_id, revision=L0g1.revision, torch_dtype=torch.bfloat16,
                device_map=device, attn_implementation="eager").eval()
            stash["base_dec"] = {k: v.detach().cpu().clone()
                                 for k, v in _bare_decoder(model.state_dict()).items()}
            meta[name] = {"readout": "base-embed (own); CLOZE+MCQ CEILING"}
        elif name == "floor":
            b = AutoModelForCausalLM.from_pretrained(
                L0g1.repo_id, revision=L0g1.revision, torch_dtype=torch.bfloat16, device_map="cpu")
            dec = _random_decoder_sd(_bare_decoder(b.state_dict()), a.floor_seed)
            del b
            model, info = SB.load_swapback_model(dec, device)
            meta[name] = {"readout": "random-decoder + base-embed; CLOZE+MCQ FLOOR", **info}
        elif name == "matched_dw":
            assert "base_dec" in stash and "pi0_dec" in stash, \
                "matched_dw needs base + pi0 decoders staged first (request base,pi0 before matched_dw)"
            mdec, frob = matched_dw_decoder(stash["base_dec"], stash["pi0_dec"], a.matched_dw_seed)
            model, info = SB.load_swapback_model(mdec, device)
            tot = sum(v * v for v in frob.values()) ** 0.5
            meta[name] = {"readout": "base-decoder + π0-norm-matched random ΔW + base-embed (readout-hardness null)",
                          "matched_dw_seed": a.matched_dw_seed, "total_frob_dW": tot,
                          "n_matched_matrices": len(frob), **info}
            print(f"[PCR] matched_dw: total ‖ΔW‖_F={tot:.3f} over {len(frob)} matrices")
        elif name in BASEEMBED_OF:               # policy DECODER under BASE embed (own-vs-base cloze control)
            pol = BASEEMBED_OF[name]
            dec, _own_head, sha = _extract_own(pol, device)
            model, info = SB.load_swapback_model(dec, device)
            meta[name] = {"readout": f"{pol} decoder + BASE-embed (own-vs-base-embed cloze equivalence control)",
                          "source_sha": sha, **info}
            print(f"[PCR] {name}: base-embed shell, {pol} decoder swapped={info['n_decoder_keys_swapped']}")
        else:                                    # pi0 / pi05 under their OWN embed
            dec, own_head, sha = _extract_own(name, device)
            if name == "pi0":
                stash["pi0_dec"] = {k: v.detach().cpu().clone() for k, v in dec.items()}
            model, info = _build_ownembed_model(dec, own_head, device)
            meta[name] = {"readout": "own-embed (tied lm_head)", "source_sha": sha, **info}
            print(f"[PCR] {name}: own-embed shell vocab={info['vocab']} decoder swapped={info['n_decoder_keys_swapped']}")

        # ---- the two paired axes on the SAME model instance, SAME items -----
        mc = TE.mmlu_choice_logits(model, tok, device, n_per_subject=n_per)
        cz = TE.mmlu_cloze(model, tok, device, n_per_subject=n_per)
        assert mc["n"] == cz["n"] > 0, f"{name}: MCQ/cloze item-count mismatch {mc['n']} vs {cz['n']}"
        assert mc["per_item_hash"] == cz["per_item_hash"], f"{name}: MCQ/cloze item order mismatch (position pairing broken)"
        mcq[name], cloze[name] = mc, cz
        meta[name]["mcq_acc_pct"] = mc["acc_pct"]
        meta[name]["cloze_acc_pct"] = cz["acc_pct"]
        print(f"[PCR] {name:10s} MCQ={mc['acc_pct']:.2f}  CLOZE={cz['acc_pct']:.2f}  (n={mc['n']})")
        del model
        torch.cuda.empty_cache()

    # ---- position-pairing tripwire across ALL arms (hash sequence equality) --
    ref_hash = mcq[arms[0]]["per_item_hash"]
    for name in arms:
        assert mcq[name]["per_item_hash"] == ref_hash, f"{name}: per-item hash sequence differs from {arms[0]} (reorder!)"
    N = len(ref_hash)
    disc, conf = split_masks(ref_hash)
    print(f"[PCR] N={N} items; discovery={sum(disc)} confirmation={sum(conf)}")

    import numpy as np
    def mask(arr, m):
        return np.asarray(arr, dtype=int)[np.asarray(m, dtype=bool)]

    # ---- per-axis per-item 0/1 arrays (position-aligned) --------------------
    mcq_item = {name: np.asarray(mcq[name]["per_item"], dtype=int) for name in arms}
    cloze_item = {name: cloze_acc_lambda(cloze[name]["cloze_logprobs"], cloze[name]["cloze_ntoks"],
                                         cloze[name]["correct_idx"], lam=1.0) for name in arms}

    result = {"torch": torch.__version__, "n_items": N, "arms": arms,
              "theta_mcq_inherited": a.theta,
              "seeds": {"floor": a.floor_seed, "matched_dw": a.matched_dw_seed,
                        "shuffle": a.shuffle_seed, "boot": a.boot_seed},
              "mcq_anchors_recompute": {"vlat_base": 40.97, "vlat_floor": 24.33, "theta": 0.10,
                                        "source": "VLAT @ c9c8773 — recomputed in-run as base/floor arms"},
              "vocab_parity": vp, "meta": meta,
              "acc_pct": {"mcq": {n: mcq[n]["acc_pct"] for n in arms},
                          "cloze": {n: cloze[n]["acc_pct"] for n in arms}}}

    if CEILING_ARM in arms and FLOOR_ARM in arms:
        # ---- cloze [FLOOR, CEILING] span + dynamic-range gate (prereg §13.1) -
        cz_ceiling = float(cloze_item[CEILING_ARM].mean())
        cz_floor = float(cloze_item[FLOOR_ARM].mean())
        mcq_ceiling = float(mcq_item[CEILING_ARM].mean())
        mcq_floor = float(mcq_item[FLOOR_ARM].mean())
        span = cz_ceiling - cz_floor
        # bootstrap on the span to judge whether it clears noise
        boot_span = paired_item_bootstrap(
            {"cz_ceiling": cloze_item[CEILING_ARM], "cz_floor": cloze_item[FLOOR_ARM]},
            [("ceiling_minus_floor", "cz_ceiling", "cz_floor")], n_boot=a.n_boot, seed=a.boot_seed)
        span_ci = boot_span["contrast"]["ceiling_minus_floor"]
        adequate = span_ci[1] > 0.05               # lo of CEILING−FLOOR clears a 5pp usable-range bar
        result["cloze_dynamic_range"] = {"ceiling": cz_ceiling, "floor": cz_floor, "span": span,
                                         "span_ci": span_ci, "adequate_for_S": bool(adequate),
                                         "note": ("S_cloze usable" if adequate else
                                                  "span small — PRIMARY=raw cloze accuracy with item-bootstrap CIs (S_cloze low-power)")}
        result["mcq_span"] = {"ceiling": mcq_ceiling, "floor": mcq_floor, "span": mcq_ceiling - mcq_floor}
        print(f"[PCR] cloze span: CEILING={cz_ceiling:.4f} FLOOR={cz_floor:.4f} span={span:.4f} "
              f"CI={span_ci[1]:.4f}..{span_ci[2]:.4f} adequate={adequate}")

        # ---- label-shuffle validity control on the CEILING (prereg §5) -------
        sh_ceiling, _ = label_shuffle_acc(cloze[CEILING_ARM]["cloze_logprobs"], cloze[CEILING_ARM]["cloze_ntoks"],
                                          cloze[CEILING_ARM]["correct_idx"], a.shuffle_seed)
        cloze_axis_valid = abs(sh_ceiling - 0.25) < 0.05
        result["label_shuffle"] = {"ceiling_shuffled_acc": sh_ceiling, "chance": 0.25,
                                   "cloze_axis_valid": bool(cloze_axis_valid),
                                   "note": ("cloze tracks content (collapses to chance under shuffle)"
                                            if cloze_axis_valid else "VOID: cloze axis does not collapse under shuffle")}
        print(f"[PCR] label-shuffle CEILING cloze acc={sh_ceiling:.4f} (chance 0.25) valid={cloze_axis_valid}")

        # ---- matched-random-ΔW readout-hardness check ------------------------
        mdw_retained = None
        if "matched_dw" in arms:
            mdw_cloze = float(cloze_item["matched_dw"].mean())
            mdw_mcq_S = (mcq_item["matched_dw"].mean() - mcq_floor) / (mcq_ceiling - mcq_floor) if mcq_ceiling > mcq_floor else float("nan")
            mdw_mid = 0.5 * (cz_ceiling + cz_floor)
            mdw_retained = bool(mdw_cloze > mdw_mid and mdw_mcq_S <= a.theta)
            result["matched_dw"] = {"cloze_acc": mdw_cloze, "mcq_S_text": float(mdw_mcq_S),
                                    "reproduces_cloze_spared_mcq_floored": mdw_retained,
                                    "note": ("readout-hardness CONFOUND active — generic ΔW reproduces the split"
                                             if mdw_retained else "generic ΔW does NOT reproduce the split (a cloze-retained policy result would be knowledge-specific)")}
            print(f"[PCR] matched_dw cloze={mdw_cloze:.4f} mcq_S={mdw_mcq_S:.3f} reproduces_split={mdw_retained}")

        # ---- per-policy verdict on the headline (confirmation) half ----------
        result["policies"] = {}
        for name in ("pi0", "pi05"):
            if name not in arms:
                continue
            policy_block = {}
            for half, m in (("confirmation", conf), ("discovery", disc), ("full", [True] * N)):
                ci_item, c_item, f_item, ce_item = (mask(cloze_item[name], m), None, mask(cloze_item[FLOOR_ARM], m),
                                                    mask(cloze_item[CEILING_ARM], m))
                mq_item = mask(mcq_item[name], m)
                cz_acc = float(ci_item.mean())
                mcq_S = (mq_item.mean() - f_item.mean()) / (ce_item.mean() - f_item.mean()) if ce_item.mean() > f_item.mean() else float("nan")
                bt = paired_item_bootstrap(
                    {"cloze_policy": ci_item, "cloze_floor": f_item, "cloze_ceiling": ce_item, "mcq_policy": mq_item},
                    [("cloze_minus_floor", "cloze_policy", "cloze_floor"),
                     ("ceiling_minus_cloze", "cloze_ceiling", "cloze_policy")],
                    n_boot=a.n_boot, seed=a.boot_seed)
                cloze_ci = bt["acc"]["cloze_policy"]
                v = policy_verdict(name, float(mcq_S), cz_acc, cloze_ci,
                                   float(ce_item.mean()), float(f_item.mean()),
                                   bool(mdw_retained) if mdw_retained is not None else False,
                                   theta_mcq=a.theta)
                v["bootstrap"] = bt
                v["n"] = int(len(mq_item))
                policy_block[half] = v
            result["policies"][name] = policy_block
            head = policy_block["confirmation"]
            print(f"[PCR] {name} HEADLINE(confirmation, n={head['n']}): cloze={head['cloze_acc']:.4f} "
                  f"CI={head['cloze_ci'][1]:.4f}..{head['cloze_ci'][2]:.4f} mcq_S={head['mcq_S_text']:.3f}\n"
                  f"       → {head['verdict']}")

        # ---- own-vs-base-embed cloze equivalence (prereg §5/§8 interface-artifact control) ----
        if any(b in arms for b in BASEEMBED_OF):
            result["own_vs_base_embed_cloze"] = {}
            for be, pol in BASEEMBED_OF.items():
                if be not in arms or pol not in arms:
                    continue
                own, base_e = cloze_item[pol], cloze_item[be]
                bt = paired_item_bootstrap({"own": own, "base_embed": base_e},
                                           [("base_minus_own", "base_embed", "own")], n_boot=a.n_boot, seed=a.boot_seed)
                own_acc, base_acc = float(own.mean()), float(base_e.mean())
                diff_ci = bt["contrast"]["base_minus_own"]
                # interface artifact iff base-embed recovers cloze the own-embed suppressed (CI lo > 3pp)
                artifact = diff_ci[1] > 0.03
                result["own_vs_base_embed_cloze"][pol] = {
                    "own_embed_cloze": own_acc, "base_embed_cloze": base_acc,
                    "base_minus_own_ci": diff_ci, "cloze_floor": cz_floor, "cloze_ceiling": cz_ceiling,
                    "interface_artifact": bool(artifact),
                    "note": ("INTERFACE ARTIFACT: base-embed recovers cloze the own-embed suppressed → reconsider H_genuine"
                             if artifact else
                             "EQUIVALENT: own-embed cloze ≈ base-embed cloze (both at floor) → cloze-at-floor is genuine, NOT an own-embed artifact")}
                print(f"[PCR] own-vs-base cloze {pol}: own={own_acc:.4f} base-embed={base_acc:.4f} "
                      f"base−own CI={diff_ci[1]:.4f}..{diff_ci[2]:.4f} artifact={artifact}")

    # ---- artifact registry (prereg §10) -------------------------------------
    result["registry"] = {
        "policy_content_readout.json": "this file — generated by infra.aws_pg.jobs.policy_content_readout",
        "axes": {"cloze": "text_eval.mmlu_cloze (text-only, λ=1)", "mcq": "text_eval.mmlu_choice_logits"},
        "pairing": "by POSITION (sorted-subject, row); per_item_hash sequence asserted equal across all arms",
        "headline": "confirmation half (is_confirmation(hash)); discovery + full also banked",
        "tag": "CONFIRMATORY",
    }
    (out / "policy_content_readout.json").write_text(json.dumps(result, indent=2, default=str))
    # per-item banks (separate file — keeps the metrics json readable)
    banks = {name: {"per_item_hash": mcq[name]["per_item_hash"],
                    "correct_idx": cloze[name]["correct_idx"],
                    "mcq_choice_logits": mcq[name]["choice_logits"],
                    "mcq_per_item": mcq[name]["per_item"],
                    "cloze_logprobs": cloze[name]["cloze_logprobs"],
                    "cloze_ntoks": cloze[name]["cloze_ntoks"],
                    "cloze_per_item": [int(x) for x in cloze_item[name]]} for name in arms}
    (out / "policy_content_readout_banks.json").write_text(json.dumps(banks, default=str))
    print("[PCR] DONE", json.dumps({"mcq": result["acc_pct"]["mcq"], "cloze": result["acc_pct"]["cloze"]}))


if __name__ == "__main__":
    main()
