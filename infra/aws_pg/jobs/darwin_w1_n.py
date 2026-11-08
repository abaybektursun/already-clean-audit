"""DARWIN W1-N + W1-N4 — conservedness quantization and core-exact quantization (CEQ). GPU.

W1-N is the frozen spec's arm; W1-N4 is the post-freeze ADDENDUM
(`2026-08-12_darwin_addendum_ceq.md`). Both live here because the addendum's arms d-f only mean
anything against a-c measured in the same job, on the same data, with the same bulk quantizer —
§3's matched-incumbent rule.

Arms (all on PG1's gemma-2b decoder; wikitext blank-image ppl, the harness G2 anchored):
  a  uniform INT4                                   incumbent floor
  b  INT4 + fp16 outlier-COLUMN list                THE HONEST INCUMBENT (the standard trick)
  c  INT4 + fp16 core-COLUMN list                   W1-N's own arm (approximate, no routing)
  d  CEQ projector-routed                           ADDENDUM
  e  CEQ rotated-block                              ADDENDUM, licensed by W0-2 (0.574 >= 0.5)
  f  vault reuse: ft-vqav2 bulk + BASE vault        ADDENDUM, the family-vault claim
  kv activation/KV variant, first-k dims fp16       ADDENDUM (the T7 mechanism)

Every arm reduces to ONE effective matrix, which is why they are comparable at all:

    a/b/c   W_eff = Q(W) with some columns restored to fp16
    d       W_eff = A Uᵀ + Q(W_bulk)(I − U Uᵀ)        (the routed form, materialised)
    e       W_eff = rotate, keep first k columns fp16, quantize the rest, rotate back
    f       W_eff = A_base Uᵀ + Q(W_ft,bulk)(I − U Uᵀ)

so each is measured by swapping W_eff into the model and running the SAME ppl path. No arm gets
a bespoke evaluator.

"gemma-2b" in the addendum: PG1's language decoder IS gemma-2b, and the banks are defined in that
decoder's input space. Running on PG1 therefore evaluates the addendum's model in the banks'
native consensus frame rather than transplanting them into a separately-trained checkpoint. The
manifest records this reading explicitly.

BINDING (addendum's corrected register): per-matrix core action exactness MAY be recorded;
end-to-end core-lane purity MUST NOT be — nothing here computes it.

  pg run infra.aws_pg.jobs.darwin_w1_n --args "--arm W1-N4" \
      --type g6e.2xlarge,g6e.4xlarge,g6e.8xlarge,g6e.xlarge --no-weights ...
"""
import argparse
import json
import pathlib

from infra.pg import darwin_ceq as C
from infra.pg import darwin_core as D
from infra.pg import darwin_w0 as DW
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as W

FT_REPO = "google/paligemma-3b-ft-vqav2-224"       # arm f's fine-tune
BITS_LADDER = (3, 4, 5, 6)                          # >=4 doses, the arm's intensity knob (§3)
SPEC_EXPECTATIONS = {
    "W1-N": ("fp16-core INT4 no better than the fp16-outlier-list trick at equal memory",
             "keeping the conserved coordinates in fp16 beats uniform INT4 and matches or beats "
             "the standard outlier-list trick"),
    "W1-N4": ("arm b (INT4 + fp16 outlier-column list) >= CEQ everywhere AND the leak-law "
              "prediction misses by >2x",
              "d/e are EXPECTED to land close to b on ppl-at-matched-bits; the decisive novelty "
              "rides on f (vault reuse, predicted by the leak law) and the KV variant"),
}


def outlier_columns_from_w0_1b(b_dir, model_id, layer, site, k):
    """Arm b's column list: the measured massive-activation coordinates from W0-1b, truncated to
    the matched budget k. Taken from the ARTIFACT, not re-derived, so arm b is the trick as this
    program actually measured it (§0.18)."""
    import numpy as np
    df = D.load_atlas(b_dir)
    med = df[(df["statistic"] == "activation_median_abs") & (df["model_id"] == model_id)
             & (df["layer"] == layer) & (df["corpus_id"] == "wikitext")]
    med = med[med["flags"].map(lambda s: json.loads(s).get("site") == site)]
    assert len(med) == 1, f"expected 1 W0-1b median row for {model_id}/L{layer}/{site}, got {len(med)}"
    v = D.vector_of(b_dir, med.iloc[0])
    return np.asarray(DW.top_k_coords(v, min(k, v.size)), dtype=np.int64)


def _ppl_of(per_window):
    """Perplexity from a per-window [[nll_sum, n_tok], ...] block — exp of the token-weighted
    mean NLL, which is how _ppl_blankimg defines it. Recomputed from the same blocks the paired
    bootstrap resamples, so the point estimate and its interval cannot come from different
    objects."""
    import math
    nll = sum(w[0] for w in per_window)
    tok = sum(w[1] for w in per_window)
    assert tok > 0, "per-window block has no tokens"
    return math.exp(nll / tok)


def build_variants(W_mat, U, outlier_cols, bits, k, *, W_ft=None):
    """Every arm's effective matrix for one (type, layer). Returns {arm_id: (W_eff, storage)}."""
    import numpy as np
    m, n = W_mat.shape
    k_matched = C.matched_k_for_vault(m=m, n=n, k_vault=k)      # arm b/c get the WIDER list
    A, bulk = C.vault_split(W_mat, U)
    Qb = C.quantize_uniform(bulk, bits)
    P_perp = np.eye(n) - U @ U.T

    core_cols = DW.top_k_coords(H.leverage(U), min(k_matched, n))
    out = {
        "a_uniform_int4": (C.quantize_uniform(W_mat, bits),
                           C.storage_bits(m=m, n=n, k=0, scheme="uniform", bits=bits)),
        "b_int4_fp16_outlier_cols": (
            C.quantize_with_fp16_columns(W_mat, outlier_cols[:k_matched], bits),
            C.storage_bits(m=m, n=n, k=k_matched, scheme="column_list", bits=bits)),
        "c_int4_fp16_core_cols": (
            C.quantize_with_fp16_columns(W_mat, core_cols, bits),
            C.storage_bits(m=m, n=n, k=k_matched, scheme="column_list", bits=bits)),
        "d_ceq_routed": (A @ U.T + Qb @ P_perp,
                         C.storage_bits(m=m, n=n, k=k, scheme="vault", bits=bits)),
        "e_ceq_rotated_block": (
            C.rotated_block_quantize(W_mat, U, bits),
            C.storage_bits(m=m, n=n, k=k, scheme="rotated_block", bits=bits,
                           n_layers_sharing_rotation=H.N_LAYERS * len(H.TYPES_RESID))),
    }
    if W_ft is not None:
        _A_ft, bulk_ft = C.vault_split(W_ft, U)
        out["f_vault_reuse"] = (A @ U.T + C.quantize_uniform(bulk_ft, bits) @ P_perp,
                                C.storage_bits(m=m, n=n, k=k, scheme="vault", bits=bits))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="W1-N4", choices=sorted(SPEC_EXPECTATIONS))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--types", default="q_proj,o_proj,gate_proj,up_proj,v_proj,k_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--bits", default=",".join(str(b) for b in BITS_LADDER))
    ap.add_argument("--k", type=int, default=H.BANK_K)
    ap.add_argument("--n-seq", type=int, default=40)
    ap.add_argument("--w0-1b-prefix", default="darwin/darwin_w0_1b")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.run_id = a.run_id or f"darwin_{a.arm.lower().replace('-', '_')}"
    a.s3_prefix = a.s3_prefix or f"{D.PROGRAM}/{a.run_id}"
    types = [t.strip() for t in a.types.split(",") if t.strip()]
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    bits_grid = [int(b) for b in a.bits.split(",") if b.strip()]
    if a.smoke:
        types, layers, bits_grid, a.n_seq = types[:1], layers[:2], bits_grid[:2], 4

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    from infra.aws_pg.jobs.invariant_core_func import (PG1_REPO, _load_full_pg1, _paired_contrast_ci,
                                                       _ppl_blankimg)

    # §3: the intensity knob sweeps >=4 doses. Bit-depth is linear, not log-spaced.
    D.check_dose_grid(bits_grid, arm=a.arm, min_doses=2 if a.smoke else 4, log_spaced=False)

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        refuter, direction = SPEC_EXPECTATIONS[a.arm]
        exp.log(a.arm, refuter=refuter, direction=direction, utc=utc)
    ref = exp.require(a.arm)

    api = HfApi()
    base_sha = api.model_info(PG1_REPO).sha
    ft_sha = api.model_info(FT_REPO).sha
    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))
    b_dir = D.fetch_prior_arm(H.cache_dir(out, "w0_1b"), a.w0_1b_prefix)

    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="GRAD", wave="W1",
                    expected_ref=ref)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "GRAD",
                "wave": "W1", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "source": ("ADDENDUM 2026-08-12_darwin_addendum_ceq.md" if a.arm == "W1-N4"
                           else "spec 2026-08-12_darwin_program_spec.md"),
                "base_repo": PG1_REPO, "base_sha": base_sha, "ft_repo": FT_REPO,
                "ft_sha": ft_sha, "types": types, "layers": layers, "bits_grid": bits_grid,
                "k_vault": a.k, "n_seq": a.n_seq, "expected_ref": ref, "started_utc": utc,
                "seed": a.seed,
                "gemma2b_reading": ("PG1's language decoder IS gemma-2b and the banks are defined "
                                    "in that decoder's input space; running on PG1 evaluates the "
                                    "addendum's model in the banks' native frame rather than "
                                    "transplanting them into a separate checkpoint"),
                "rotation_licensed_by": "W0-2 mean shared-coordinate fraction 0.574 >= 0.5",
                "binding_register": ("per-matrix core action exactness MAY be recorded; "
                                     "end-to-end core-lane purity MUST NOT be and is not computed")}
    D.write_json(out, "manifest.json", manifest)

    model = _load_full_pg1(base_sha, "cuda")
    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    layer_mods = {}
    from infra.aws_pg.jobs.invariant_core_func import _decoder_layers
    dl = _decoder_layers(model)
    for t in types:
        for L in layers:
            parent = dl[L].self_attn if t.startswith(("q", "k", "v", "o")) else dl[L].mlp
            layer_mods[(t, L)] = getattr(parent, t)

    base_res = _ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)
    base_ppl = float(base_res["ppl"])
    atlas.add("base_ppl", value=base_ppl, model_id=PG1_REPO, model_sha=base_sha,
              flags={"n_tokens": base_res["n_tokens"], "n_seq": a.n_seq})
    print(f"[W1-N] fp16 base ppl = {base_ppl:.5f} over {base_res['n_tokens']} tokens", flush=True)

    w_ft = W._load_decoder(FT_REPO, ft_sha)
    originals = {k: m.weight.detach().clone() for k, m in layer_mods.items()}
    leak_rows = []

    INCUMBENT = "b_int4_fp16_outlier_cols"
    for bits in bits_grid:
        # Per-window NLL for every arm at this dose, so the comparison against the incumbent
        # carries a PAIRED bootstrap CI. The addendum's graduation clause is stated in units of
        # 'more than the bootstrap 95% CI half-width', so an arm that reports only a point ppl
        # cannot evaluate its own rule -- it would be VOID-underpowered by construction while the
        # data to compute the interval sat in _ppl_blankimg's discarded per_window (defect D14).
        per_window = {}
        # Swap EVERY targeted matrix to the arm's variant, measure ppl once, restore. Per-matrix
        # ppl would answer a different question (one quantized layer in an fp16 model).
        variant_ids = None
        for arm_id in ("a_uniform_int4", "b_int4_fp16_outlier_cols", "c_int4_fp16_core_cols",
                       "d_ceq_routed", "e_ceq_rotated_block", "f_vault_reuse"):
            if a.arm == "W1-N" and arm_id.startswith(("d_", "e_", "f_")):
                continue                     # the addendum's arms belong to W1-N4
            storages = []
            for (t, L), mod in layer_mods.items():
                Wm = originals[(t, L)].to(torch.float64).cpu().numpy()
                U = W.load_bank(bank_dir, t, L, "core32")
                site = H.SITE[t]
                oc = outlier_columns_from_w0_1b(b_dir, "pg1-pt224", L, site,
                                                C.matched_k_for_vault(m=Wm.shape[0],
                                                                      n=Wm.shape[1], k_vault=a.k))
                Wft = (w_ft[W.matrix_key(t, L)].to(torch.float64).numpy()
                       if arm_id == "f_vault_reuse" else None)
                variants = build_variants(Wm, U, oc, bits, a.k, W_ft=Wft)
                variant_ids = sorted(variants)
                W_eff, storage = variants[arm_id]
                storages.append(storage["bits_per_weight"])
                mod.weight.data.copy_(torch.as_tensor(W_eff, dtype=mod.weight.dtype,
                                                      device=mod.weight.device))
                if arm_id in ("d_ceq_routed", "f_vault_reuse"):
                    A, bulk = C.vault_split(Wm, U)
                    err = C.core_action_error(Wm, U, A, C.quantize_uniform(bulk, bits))
                    atlas.add("core_action_error", value=err, layer=L, type=t, bank_id="core32",
                              dose=float(bits), incumbent_id="b_int4_fp16_outlier_cols",
                              model_id=PG1_REPO, model_sha=base_sha,
                              flags={"arm_id": arm_id, "claim": "T1 per-matrix exactness"})
                if arm_id == "f_vault_reuse":
                    lk = C.leak_law_check(Wm, Wft, U)
                    leak_rows.append(lk["within_2x"])
                    atlas.add("leak_law_predicted", value=lk["predicted_leak_norm"], layer=L,
                              type=t, bank_id="core32", dose=float(bits), model_id=FT_REPO,
                              model_sha=ft_sha, flags={**lk, "arm_id": arm_id})
                    atlas.add("leak_law_realized", value=lk["realized_vault_error"], layer=L,
                              type=t, bank_id="core32", dose=float(bits), model_id=FT_REPO,
                              model_sha=ft_sha, flags={**lk, "arm_id": arm_id})
            res = _ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)
            ppl = float(res["ppl"])
            per_window[arm_id] = res["per_window"]
            atlas.add("wikitext_ppl", value=ppl, bank_id=arm_id, dose=float(bits),
                      incumbent_id="b_int4_fp16_outlier_cols", model_id=PG1_REPO,
                      model_sha=base_sha,
                      flags={"arm_id": arm_id, "bits": bits, "base_ppl": base_ppl,
                             "delta_ppl": ppl - base_ppl,
                             "mean_bits_per_weight": float(np.mean(storages)),
                             "n_tokens": res["n_tokens"], "k_vault": a.k})
            print(f"[W1-N] bits={bits} {arm_id}: ppl={ppl:.5f} (base {base_ppl:.5f}, "
                  f"{np.mean(storages):.3f} bits/wt)", flush=True)
            for (t, L), mod in layer_mods.items():
                mod.weight.data.copy_(originals[(t, L)])

        # Paired contrast against the honest incumbent, same windows, same order.
        if INCUMBENT in per_window:
            for arm_id, pw in sorted(per_window.items()):
                if arm_id == INCUMBENT:
                    continue
                ci = _paired_contrast_ci(pw, per_window[INCUMBENT])
                hw = (ci["hi"] - ci["lo"]) / 2.0
                atlas.add("wikitext_ppl_contrast_vs_incumbent", value=ci["contrast"],
                          bank_id=arm_id, dose=float(bits), incumbent_id=INCUMBENT,
                          ci_lo=ci["lo"], ci_hi=ci["hi"], n=ci["n_windows"],
                          model_id=PG1_REPO, model_sha=base_sha,
                          flags={**ci, "bits": bits, "ci_halfwidth": hw,
                                 "note": "positive contrast = this arm is WORSE (higher ppl)"})
                if arm_id in ("d_ceq_routed", "e_ceq_rotated_block"):
                    ppl_arm, ppl_b = _ppl_of(pw), _ppl_of(per_window[INCUMBENT])
                    g = C.ceq_graduates(ppl_ceq=ppl_arm, ppl_ceq_ci_lo=ppl_arm - hw,
                                        ppl_ceq_ci_hi=ppl_arm + hw, ppl_incumbent_b=ppl_b,
                                        leak_within_2x=bool(leak_rows) and all(leak_rows))
                    atlas.add("ceq_graduation_clause", value=float(g["graduates"]),
                              bank_id=arm_id, dose=float(bits), incumbent_id=INCUMBENT,
                              model_id=PG1_REPO, model_sha=base_sha, flags=g)
                    print(f"[W1-N4] bits={bits} {arm_id} vs b: contrast {ci['contrast']:+.4f} "
                          f"CI[{ci['lo']:+.4f},{ci['hi']:+.4f}] -> {g['path']}", flush=True)
        D.s3_sync(out, a.s3_prefix)
        gates.record(f"W1N_variants_built_bits{bits}", True, variants=variant_ids)

    # --- the KV / activation variant (T7 mechanism), measured on real captured activations.
    from infra.pg import humboldt_act as A_
    sampler_layers = dl
    for L in layers[:4]:
        U = W.load_bank(bank_dir, "q_proj", L, "core32")
        cap = {}

        def _hook(_m, args, _L=L):
            cap[_L] = args[0].reshape(-1, args[0].shape[-1]).float().detach().cpu().numpy()
        h = getattr(sampler_layers[L].self_attn,
                    A_.SITE_MODULE["q_proj_in"]).register_forward_pre_hook(_hook)
        _ppl_blankimg(model, processor, "cuda", n_seq=min(4, a.n_seq))
        h.remove()
        for bits_a in (4, 8):
            r = C.routed_activation_error(cap[L], U, bits=bits_a)
            atlas.add("kv_routed_activation_error_ratio", value=r["ratio"], layer=L,
                      type="q_proj", bank_id="core32", dose=float(bits_a),
                      incumbent_id="uniform_activation_quant", model_id=PG1_REPO,
                      model_sha=base_sha, flags={**r, "claim": "T7 mechanism", "site": "q_proj_in"})
    D.s3_sync(out, a.s3_prefix)

    if leak_rows:
        gates.record("W1N4_leak_law_within_2x", bool(all(leak_rows)),
                     n_cells=len(leak_rows), n_within=sum(leak_rows),
                     note="addendum graduation clause reads this alongside the ppl comparison")

    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, f"{a.arm.lower().replace('-', '_')}_summary.json",
                 {"manifest": manifest, "atlas": rep, "gates": gates.gates,
                  "base_ppl": base_ppl, "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_{a.arm.replace('-', '_')}_OK rows={rep['n_rows']} base_ppl={base_ppl:.5f}")


if __name__ == "__main__":
    main()
