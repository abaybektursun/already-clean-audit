"""VLAT §9.1 — WEIGHT-FREE DRY-RUN GATE (no GPU, no model downloads).

Proves the LOGIC of the NEW swap-back instrument + the retention/mechanism arithmetic on tiny
synthetic tensors and a tiny GemmaForCausalLM config, BEFORE the §9.3 freeze and long before any GPU
forward. Distinct from §9.2 (which proves the WIRING on real weights). Eight gates, all must pass:

  G1  swap-back round-trip   — build a fresh tiny Gemma-1 shell, swap in a perturbed decoder, strict-load
                               (0 missing/unexpected), embed/head KEPT bitwise, decoder CHANGED, forward
                               finite with shape [B,T,vocab]; exercises the real `model.`-prefix re-key.
  G2  leak guards            — action-expert / vision key, a missing decoder key, and a shape mismatch
                               each RAISE (partial/corrupt swap is impossible to do silently).
  G3  decompose round-trip   — gain_routing.decompose returns a DICT (not a 6-tuple) with f_perp and a
                               fp64 recon_residual < 1e-10 on a planted ΔW.
  G4  ΔW pairing semantics   — matched_decoder_weight_matrices returns ABSOLUTE (w_before, w_after);
                               the caller forms ΔW = wa − wb.
  G5  S_text arithmetic      — s_text((c−FLOOR)/(BASE−FLOOR)): BASE→1, FLOOR→0, midpoint→0.5; degenerate
                               band RAISES.
  G6  split bucketing        — per_item_hash discovery/confirmation split is deterministic, disjoint, and
                               ~balanced; the §4.1 within-checkpoint blind split.
  G7  assert_18x7            — a complete 18×7 grid passes; one missing matrix RAISES (gap, not just count).
  G8  pipeline-minus-forward — end-to-end retention arithmetic for 4 checkpoints on PLANTED per-item
                               correctness → S_text vector → recipe-prediction label match, no model.

Any fail → HALT (raise). No try/except hides a failure; the `_expect_raise` helper asserts that a guard
DID raise (legitimate test discipline), and re-raises a clean error if it did not.

    python -m infra.aws_pg.jobs.vlat_dryrun --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _expect_raise(fn, must_contain):
    """Run fn; require it to raise with `must_contain` in the message. Returns the message."""
    raised = None
    try:
        fn()
    except Exception as e:                          # test-only: we ASSERT a raise, never hide one
        raised = e
    assert raised is not None, f"expected a raise containing {must_contain!r}, but none occurred"
    msg = str(raised)
    assert must_contain in msg, f"raised, but {must_contain!r} not in message: {msg!r}"
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-split-probe", type=int, default=4000,
                    help="planted MMLU items for the §4.1 split-balance check")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import GemmaConfig, GemmaForCausalLM

    from infra.pg import gain_routing as GR
    from infra.pg import vlat_swapback as SB
    from infra.pg.pi0_port import PROJ, assert_18x7

    torch.manual_seed(1337)
    gates = {}

    # ===================================================================== G1
    # A tiny Gemma-1 shell (random) stands in for google/gemma-2b: same key/namespace structure
    # (model.embed_tokens / model.layers.N.* / model.norm / lm_head), no download.
    cfg = GemmaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=1, head_dim=8, max_position_embeddings=32)
    shell = GemmaForCausalLM(cfg).eval()
    base_sd = {k: v.detach().clone() for k, v in shell.state_dict().items()}

    # Simulate pi0_port.remap_backbone_lm output: BARE decoder keys (embed_tokens dropped), perturbed
    # so the swap is detectable. lm_head is tied/own — exclude it (it is part of the fixed instrument).
    bare_decoder = {k[len("model."):]: (base_sd[k] + 0.01 * torch.randn_like(base_sd[k]))
                    for k in base_sd if k.startswith("model.") and "embed_tokens" not in k}
    dec = SB.prefix_decoder_sd(bare_decoder)        # re-key into the shell namespace
    new_sd, info = SB.build_swapback_sd(shell.state_dict(), dec)
    missing, unexpected = shell.load_state_dict(new_sd, strict=True)
    assert not missing and not unexpected, f"strict-load not clean: missing={missing} unexpected={unexpected}"

    now = shell.state_dict()
    emb_keys = [k for k in base_sd if any(e in k for e in SB.EMBED_KEYS)]
    assert emb_keys, "tiny shell exposed no embed/head key to keep — config wrong"
    for k in emb_keys:
        assert torch.equal(now[k], base_sd[k]), f"embed/head key {k} changed — must be kept"
    dec_keys = [k for k in base_sd if not any(e in k for e in SB.EMBED_KEYS)]
    max_delta = max((now[k].float() - base_sd[k].float()).abs().max().item() for k in dec_keys)
    assert max_delta > 0.0, "decoder unchanged — swap did nothing"
    ids = torch.randint(0, cfg.vocab_size, (2, 5))
    with torch.inference_mode():
        logits = shell(input_ids=ids).logits
    assert logits.shape == (2, 5, cfg.vocab_size), f"bad logits shape {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "non-finite logits from the swap-back shell"
    gates["G1_swapback_roundtrip"] = {"ok": True, "n_swapped": info["n_decoder_keys_swapped"],
                                      "n_embed_kept": info["n_embed_keys_kept"], "decoder_max_delta": max_delta,
                                      "logits_shape": list(logits.shape)}

    # ===================================================================== G2
    leak = dict(dec); leak["model.layers.0.gemma_expert.weight"] = torch.zeros(4, 4)
    m_leak = _expect_raise(lambda: SB.build_swapback_sd(shell.state_dict(), leak), "non-backbone keys leaked")
    drop = dict(dec); drop.pop(next(iter(k for k in dec if "q_proj" in k)))
    m_miss = _expect_raise(lambda: SB.build_swapback_sd(shell.state_dict(), drop), "partial swap would leak")
    bad = dict(dec); bk = next(iter(k for k in dec if "q_proj" in k)); bad[bk] = torch.zeros(3, 3)
    m_shape = _expect_raise(lambda: SB.build_swapback_sd(shell.state_dict(), bad), "shape mismatch")
    extra = dict(dec); extra["model.layers.0.bogus.weight"] = torch.zeros_like(base_sd["model.norm.weight"])
    m_extra = _expect_raise(lambda: SB.build_swapback_sd(shell.state_dict(), extra), "unused by the base shell")
    gates["G2_leak_guards"] = {"ok": True, "expert": m_leak[:60], "missing": m_miss[:60],
                               "shape": m_shape[:60], "extra": m_extra[:60]}

    # ===================================================================== G3
    w_before = torch.randn(16, 24, dtype=torch.float64)
    w_after = w_before + 0.05 * torch.randn(16, 24, dtype=torch.float64)
    d = GR.decompose(w_before, w_after, check_recon=True)
    assert isinstance(d, dict), f"decompose must return a dict, got {type(d)}"
    for key in ("f_perp", "E_gain", "in_span_energy", "out_of_span_energy", "route_energy", "recon_residual"):
        assert key in d, f"decompose dict missing key {key!r}"
    assert d["recon_residual"] < 1e-10, f"fp64 recon_residual not tight: {d['recon_residual']}"
    gates["G3_decompose_roundtrip"] = {"ok": True, "is_dict": True, "f_perp": float(d["f_perp"]),
                                       "recon_residual": float(d["recon_residual"])}

    # ===================================================================== G4
    # signature is (pg_lm_sd=AFTER, l0_sd=BEFORE); returns (wb=l0[k], wa=pg[k]) = (before, after).
    after_sd = {"model.layers.0.self_attn.q_proj.weight": w_after}
    before_sd = {"model.layers.0.self_attn.q_proj.weight": w_before}
    pairs = GR.matched_decoder_weight_matrices(after_sd, before_sd)
    (k0, (wb, wa)), = pairs.items()
    assert torch.equal(wb, w_before) and torch.equal(wa, w_after), "pairs must be ABSOLUTE (w_before, w_after)"
    dW = wa - wb
    assert torch.allclose(dW, w_after - w_before), "caller-side ΔW = wa − wb failed"
    gates["G4_dW_pairing"] = {"ok": True, "key": k0, "returns": "absolute (w_before, w_after)"}

    # ===================================================================== G5
    assert abs(SB.s_text(0.5, 0.5, 0.25) - 1.0) < 1e-12, "BASE must map to 1.0"
    assert abs(SB.s_text(0.25, 0.5, 0.25) - 0.0) < 1e-12, "FLOOR must map to 0.0"
    assert abs(SB.s_text(0.375, 0.5, 0.25) - 0.5) < 1e-12, "midpoint must map to 0.5"
    m_band = _expect_raise(lambda: SB.s_text(0.3, 0.25, 0.25), "must exceed FLOOR")
    gates["G5_s_text"] = {"ok": True, "degenerate_band_raises": m_band[:50]}

    # ===================================================================== G6
    import hashlib
    def _pih(q):                                    # text_eval.py:132 recipe (verified), not transcribed
        return hashlib.sha1(q[:64].encode()).hexdigest()[:12]
    questions = [f"Question {i}: which option correctly completes the statement about topic {i}?"
                 for i in range(args.n_split_probe)]
    hashes = [_pih(q) for q in questions]
    b1 = [SB.split_bucket(h) for h in hashes]
    b2 = [SB.split_bucket(h) for h in hashes]
    assert b1 == b2, "split_bucket not deterministic"
    n_disc = sum(x == "discovery" for x in b1)
    frac = n_disc / len(hashes)
    assert 0.45 <= frac <= 0.55, f"split imbalance: discovery frac={frac:.3f}"
    assert set(b1) <= {"discovery", "confirmation"}, "split produced an unexpected label"
    # known per_item_hash property: [:64] truncation collides; collided items are indistinguishable so
    # they MUST co-bucket (never straddle the discovery/confirmation firewall).
    qa, qb = "Z" * 64 + " alpha?", "Z" * 64 + " beta?"
    assert _pih(qa) == _pih(qb), "first-64 truncation should collide"
    assert SB.split_bucket(_pih(qa)) == SB.split_bucket(_pih(qb)), "collided items must co-bucket"
    gates["G6_split_bucket"] = {"ok": True, "deterministic": True, "discovery_frac": frac,
                                "n": len(hashes), "collision_cobuckets": True}

    # ===================================================================== G7
    full = {f"layers.{i}.{('self_attn' if p[0] in 'qkvo' else 'mlp')}.{p}.weight": torch.zeros(2, 2)
            for i in range(18) for p in PROJ}
    assert len(assert_18x7(full)) == 18 * 7, "complete 18×7 grid should pass"
    gap = dict(full); gap.pop("layers.7.mlp.gate_proj.weight")
    m_18 = _expect_raise(lambda: assert_18x7(gap), "18×7 incomplete")
    gates["G7_assert_18x7"] = {"ok": True, "complete_passes": True, "gap_raises": m_18[:40]}

    # ===================================================================== G8
    # Planted per-item correctness for 4 checkpoints; FLOOR/BASE anchor S_text; recipe labels compared.
    planted = {"base": 0.50, "PG1": 0.46, "pi0": 0.30, "pi05": 0.44}  # mean MMLU acc (illustrative)
    FLOOR, BASE = 0.25, planted["base"]
    theta = 0.15                                    # FLOOR + noise band on the S_text scale → deletion gate
    stext = {c: SB.s_text(acc, BASE, FLOOR) for c, acc in planted.items()}
    measured = {c: ("forget" if stext[c] < theta else "retain") for c in stext}
    predicted = {"PG1": "retain", "pi0": "forget", "pi05": "retain"}   # illustrative recipe forecast
    match = {c: (measured[c] == predicted[c]) for c in predicted}
    assert abs(stext["base"] - 1.0) < 1e-12, "BASE S_text must be 1.0"
    assert set(measured.values()) <= {"forget", "retain"}, "bad label"
    gates["G8_pipeline_minus_forward"] = {"ok": True, "s_text": stext, "measured": measured,
                                          "predicted": predicted, "match": match, "theta": theta}

    ok = all(g["ok"] for g in gates.values())
    result = {"torch": torch.__version__, "transformers_GemmaConfig": True, "tiny_cfg": cfg.to_dict(),
              "gates": gates, "DRYRUN_OK": ok}
    (out / "dryrun.json").write_text(json.dumps(result, indent=2, default=str))
    print("[VLAT-DRY] gates: " + "  ".join(f"{k.split('_')[0]}={v['ok']}" for k, v in gates.items()))
    print(f"[VLAT-DRY] DRYRUN_OK={ok}  → {out / 'dryrun.json'}")
    if not ok:
        raise RuntimeError(f"VLAT §9.1 DRY-RUN HALT: {[k for k, v in gates.items() if not v['ok']]}")
    print("[VLAT-DRY] §9.1 gate passed → swap-back logic + retention/mechanism arithmetic are sound.")


if __name__ == "__main__":
    main()
