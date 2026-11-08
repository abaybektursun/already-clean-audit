"""W16 PREFLIGHT GATE (fail-fast, before the W16 prereg freeze / controls / any arm).

Clones the w11/w14 preflight pattern, scoped to the W16 dispositional audit
(program spec 2026-06-09 §4): validates the HARNESS + the two NEW read/probe
surfaces on real weights. Gates (all GPU; HALT/raise on failure, no try/except):

  (1a) Generation firewall: base gemma-2-2b AND PG2 text_config both 26 layers
       (Gemma-2, not the Gemma-1/18L sibling).
  (1b) Harness validity + cached-number reproduction: base gemma-2-2b text-only
       MMLU within --tol-mmlu of published 51.3 AND within --tol-cached of the
       frozen W14 controls' base acc (s3 legA_controls.json — the "preflight
       reproduces a cached number" rule, §11). Tokenizer parity + finite ppl.
  (1c) Logit-capture parity (W16-A read-side addition): mmlu_choice_logits on
       base over a small cap — derived per_item EXACTLY equals mmlu_accuracy's
       per_item on the same cap; hashes identical; margins/Brier/AUC finite.
  (1d) Probe wiring on the W16 target (PG2 AFTER, text-only forward):
       snapshot → int8 quant → capped eval finite → restore(bitwise) →
       int4 quant → capped eval finite → restore(bitwise) →
       noise(σ=1e-3, seed=1337) → capped eval finite → restore; noise re-applied
       from the same snapshot/seed is bitwise-identical (determinism); after the
       final restore the capped unprobed eval reproduces its pre-probe per_item
       exactly (probe cycles leave the AFTER readout untouched).
  (1e) Item-split firewall: hash-parity discovery/confirmation halves non-empty
       + disjoint (swap_back.subset_by_hash mechanics).

    python -m infra.aws_pg.jobs.w16_preflight --out <dir> [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

W14_CONTROLS_S3 = "s3://pg-lineage-769063704696/results/w14_arm23_controls/out/legA_controls.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", default=W14_CONTROLS_S3,
                    help="frozen W14 legA_controls.json (cached base MMLU to reproduce)")
    ap.add_argument("--tol-mmlu", type=float, default=3.0,
                    help="abs tol (pct pts) base MMLU vs published 51.3")
    ap.add_argument("--tol-cached", type=float, default=1.0,
                    help="abs tol (pct pts) base MMLU vs the cached controls number")
    ap.add_argument("--mmlu-n-per-subject", type=int, default=0,
                    help="cap for the base validity MMLU (0 = full test split)")
    ap.add_argument("--probe-n-per-subject", type=int, default=2,
                    help="small per-subject cap for the 1c/1d instrument probes")
    ap.add_argument("--ppl-max-tokens", type=int, default=120_000)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny sizes, wiring only (cached/published gates relaxed)")
    args = ap.parse_args()
    if args.smoke:
        args.mmlu_n_per_subject = 2
        args.probe_n_per_subject = 2
        args.ppl_max_tokens = 4_000
        args.tol_mmlu = 100.0     # smoke exercises WIRING; n=2/subject is too noisy to gate
        args.tol_cached = 100.0   # the real run uses 3.0 / 1.0

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()                                # MUST precede transformers/datasets

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PaliGemmaProcessor

    from infra.pg import text_eval as TE
    from infra.pg import w16_probes as W
    from infra.pg.entities import ENTITIES
    from infra.pg.swap_back import subset_by_hash

    assert torch.cuda.is_available(), "W16 preflight needs GPU (model forwards)."
    device = "cuda"
    print(f"[W16-PRE] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} smoke={args.smoke}")

    L0 = ENTITIES["L0"]            # google/gemma-2-2b
    PG = ENTITIES["PG"]            # google/paligemma2-3b-pt-224

    # ---- (1a) generation firewall: 26 decoder layers on both sides ----
    l0_cfg = AutoConfig.from_pretrained(L0.repo_id, revision=L0.revision)
    pg_cfg = AutoConfig.from_pretrained(PG.repo_id)
    pg_text_cfg = getattr(pg_cfg, "text_config", pg_cfg)
    l0_layers = int(l0_cfg.num_hidden_layers)
    pg_layers = int(pg_text_cfg.num_hidden_layers)
    gate_1a = bool(l0_layers == 26 and pg_layers == 26)
    print(f"[W16-PRE][1a] firewall: L0 layers={l0_layers} PG text layers={pg_layers} → ok={gate_1a}")

    # ---- cached number to reproduce (W14 frozen controls; fail-loud fetch) ----
    ctrl_local = out / "legA_controls_cached.json"
    subprocess.run(["aws", "s3", "cp", args.controls_s3, str(ctrl_local), "--region", "us-east-1"],
                   check=True)
    ctrl = json.loads(ctrl_local.read_text())
    cached_base_pct = float(ctrl["mmlu_text"]["base"]["acc_pct"])   # W14 schema: text-only base anchor
    cached_n_per = int(ctrl["mmlu_n_per_subject"])
    print(f"[W16-PRE] cached base MMLU={cached_base_pct:.2f} (controls n_per_subject={cached_n_per})")

    # ---- (1b) base gemma-2-2b harness validity (eager: Gemma2 soft-capping; W14 convention) ----
    base = AutoModelForCausalLM.from_pretrained(
        L0.repo_id, revision=L0.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    tok_g = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    tok_pg = PaliGemmaProcessor.from_pretrained(PG.repo_id).tokenizer
    parity = TE.tokenizer_parity(tok_g, tok_pg)
    n_per = args.mmlu_n_per_subject or None
    mmlu = TE.mmlu_accuracy(base, tok_g, device, n_per_subject=n_per)
    d_pub = abs(mmlu["acc_pct"] - mmlu["published"])
    # cached-number comparison is only meaningful when the caps match the controls run
    cap_matches_cached = bool(args.mmlu_n_per_subject == cached_n_per)
    d_cached = abs(mmlu["acc_pct"] - cached_base_pct) if cap_matches_cached else None
    cached_ok = bool(cap_matches_cached and d_cached <= args.tol_cached) or args.smoke
    ppl = TE.clean_text_perplexity(base, tok_g, device, max_tokens=args.ppl_max_tokens)
    ppl_ok = bool(np.isfinite(ppl["ppl"]))
    gate_1b = bool(parity["match"] and d_pub <= args.tol_mmlu and cached_ok and ppl_ok)
    print(f"[W16-PRE][1b] parity={parity['match']} MMLU={mmlu['acc_pct']:.2f} "
          f"|Δpub|={d_pub:.2f}(tol {args.tol_mmlu}) |Δcached|={d_cached} (tol {args.tol_cached}) "
          f"ppl={ppl['ppl']:.2f} → ok={gate_1b}")

    # ---- (1c) logit-capture parity on base (small cap) ----
    cap = args.probe_n_per_subject
    acc_small = TE.mmlu_accuracy(base, tok_g, device, n_per_subject=cap)
    cl = TE.mmlu_choice_logits(base, tok_g, device, n_per_subject=cap)
    per_item_match = bool(cl["per_item"] == acc_small["per_item"])
    hash_match = bool(cl["per_item_hash"] == acc_small["per_item_hash"])
    margins = W.margin_per_item(cl["choice_logits"], cl["correct_idx"])
    briers = W.brier_per_item(cl["choice_logits"], cl["correct_idx"])
    auc = W.rank_auc(cl["choice_logits"], cl["correct_idx"])
    cont_finite = bool(np.isfinite(margins).all() and np.isfinite(briers).all()
                       and 0.0 <= auc <= 1.0)
    gate_1c = bool(per_item_match and hash_match and cont_finite)
    print(f"[W16-PRE][1c] per_item_match={per_item_match} hash_match={hash_match} "
          f"margin_mean={margins.mean():.3f} brier_mean={briers.mean():.3f} auc={auc:.3f} → ok={gate_1c}")
    del base
    torch.cuda.empty_cache()

    # ---- (1d) probe wiring on PG2 AFTER (text-only, the W11-A convention) ----
    pg_after, pg_before, _proc, paths, _it, build_gate = H.build_before_after(device)
    del pg_before
    torch.cuda.empty_cache()
    lm = pg_after.get_submodule(paths["language_model"])

    scope = W.probe_param_names(lm)
    assert scope and all("embed_tokens" not in n and "lm_head" not in n for n in scope)
    snap = W.snapshot_probe_params(lm)
    ref_name = scope[0]

    unprobed = TE.mmlu_accuracy(pg_after, tok_g, device, n_per_subject=cap)
    print(f"[W16-PRE][1d] unprobed AFTER capped MMLU={unprobed['acc_pct']:.2f} (n={unprobed['n']})")

    probe_reports = {}
    for bits in W.QUANT_BITS:
        rep = W.quantize_decoder_(lm, bits)
        ev = TE.mmlu_accuracy(pg_after, tok_g, device, n_per_subject=cap)
        rep["capped_acc_pct"] = ev["acc_pct"]
        probe_reports[f"int{bits}"] = rep
        W.restore_probe_params_(lm, snap)
        assert torch.equal(lm.get_parameter(ref_name).detach().cpu(), snap[ref_name]), \
            f"int{bits} restore not bitwise on {ref_name}"
        print(f"[W16-PRE][1d] int{bits}: rel_frob_max={rep['rel_frob_max']:.4f} "
              f"capped MMLU={ev['acc_pct']:.2f} → restored")

    rep_n = W.noise_decoder_(lm, 1e-3, seed=W.NOISE_SEEDS[0])
    noised_1 = lm.get_parameter(ref_name).detach().cpu().clone()
    ev_n = TE.mmlu_accuracy(pg_after, tok_g, device, n_per_subject=cap)
    rep_n["capped_acc_pct"] = ev_n["acc_pct"]
    probe_reports["noise_1e-3_s1337"] = rep_n
    W.restore_probe_params_(lm, snap)
    W.noise_decoder_(lm, 1e-3, seed=W.NOISE_SEEDS[0])
    noise_deterministic = bool(torch.equal(lm.get_parameter(ref_name).detach().cpu(), noised_1))
    W.restore_probe_params_(lm, snap)
    print(f"[W16-PRE][1d] noise σ=1e-3 s1337: rel_frob_max={rep_n['rel_frob_max']:.6f} "
          f"capped MMLU={ev_n['acc_pct']:.2f} deterministic={noise_deterministic}")

    reprobed = TE.mmlu_accuracy(pg_after, tok_g, device, n_per_subject=cap)
    roundtrip_ok = bool(reprobed["per_item"] == unprobed["per_item"]
                        and reprobed["per_item_hash"] == unprobed["per_item_hash"])
    gate_1d = bool(noise_deterministic and roundtrip_ok
                   and all(np.isfinite(r["capped_acc_pct"]) for r in probe_reports.values()))
    print(f"[W16-PRE][1d] roundtrip per_item identical={roundtrip_ok} → ok={gate_1d}")
    del pg_after
    torch.cuda.empty_cache()

    # ---- (1e) item-split firewall on the captured hashes ----
    hashes = mmlu["per_item_hash"]
    disc = subset_by_hash(hashes, hashes, "discovery")
    conf = subset_by_hash(hashes, hashes, "confirmation")
    gate_1e = bool(disc and conf and set(disc).isdisjoint(conf)
                   and len(disc) + len(conf) == len(hashes))
    print(f"[W16-PRE][1e] split disc={len(disc)} conf={len(conf)} → ok={gate_1e}")

    ok = bool(gate_1a and gate_1b and gate_1c and gate_1d and gate_1e)
    result = {
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "smoke": args.smoke,
        "entities": {"L0": L0.repo_id, "PG": PG.repo_id},
        "gate_1a_firewall": {"L0_num_layers": l0_layers, "PG_num_layers": pg_layers, "ok": gate_1a},
        "gate_1b_harness": {
            "tokenizer_parity": parity["match"],
            "mmlu_acc_pct": mmlu["acc_pct"], "published": mmlu["published"],
            "abs_delta_published": d_pub, "tol_mmlu": args.tol_mmlu,
            "cached_controls_s3": args.controls_s3, "cached_base_pct": cached_base_pct,
            "cap_matches_cached": cap_matches_cached,
            "abs_delta_cached": d_cached, "tol_cached": args.tol_cached,
            "ppl": ppl["ppl"], "ok": gate_1b},
        "gate_1c_logit_capture": {
            "per_item_match": per_item_match, "hash_match": hash_match,
            "margin_mean": float(margins.mean()), "brier_mean": float(briers.mean()),
            "rank_auc": auc, "n": cl["n"], "ok": gate_1c},
        "gate_1d_probe_wiring": {
            "n_scope_tensors": len(scope),
            "unprobed_capped_acc_pct": unprobed["acc_pct"],
            "probes": probe_reports,
            "noise_deterministic": noise_deterministic,
            "restore_roundtrip_per_item_identical": roundtrip_ok, "ok": gate_1d},
        "gate_1e_item_split": {"n_disc": len(disc), "n_conf": len(conf), "ok": gate_1e},
        "build_gate": build_gate,
        "PREFLIGHT_OK": ok,
    }
    (out / "preflight.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W16-PRE] PREFLIGHT_OK={ok} (1a={gate_1a} 1b={gate_1b} 1c={gate_1c} "
          f"1d={gate_1d} 1e={gate_1e})")
    if not ok:
        raise RuntimeError(f"W16 PREFLIGHT HALT: see {out / 'preflight.json'}")
    print("W16_PREFLIGHT_OK", json.dumps({"mmlu_base_pct": mmlu["acc_pct"],
                                          "cached_base_pct": cached_base_pct}))


if __name__ == "__main__":
    main()
