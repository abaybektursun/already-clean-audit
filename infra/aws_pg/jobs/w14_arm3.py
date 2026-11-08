"""W14 ARM 3 — minimal targeted edit that restores MMLU; does Arm-2-informed targeting beat the
untargeted graft? (the Hase null-of-record).

Weight-swap / SVD-truncation grafts onto AFTER (no training). Restore = move the adapted decoder
back toward base gemma-2-2b. Four families (prereg §A3.2):

  T       — targeted layer-restore: reset the Arm-2 module's matrices at the top-L `arm2_layer_rank`
            layers (L∈{1,2,4,8,16,26}). The payoff arm.
  T-lr    — targeted rank-k SVD restore confined to the Arm-2 module.
  U       — untargeted full-decoder rank-k restore (all 182 matrices). "What you'd do without Arm 2."
  U-layer — untargeted layer-restore by descending |ΔW| rel-Frob (W1-ranked) — the discriminating
            control: if T beats U-layer, Arm-2 causal credit beats raw-magnitude targeting.

R_restore = (m(edit) − FLOOR)/(BEFORE − FLOOR) on blank-image MMLU (PRIMARY), re-read from the
re-frozen legA_controls.json. Decisive scalars: Π* (cheapest targeted reaching τ*_restore=0.50,
monotone "stays ≥ τ at and above" — §A3.6B) vs Π_untarg (cheapest untargeted). Cells (BH-FDR q=0.05
over exactly {A,B,C,D}, a bootstrap CI per cell — §A3.4/§A3.7): A log10(Π_untarg/Π*)>0; B
R_restore(T@Π*)−matched-numel-random@Π*>δ_target=0.10; C min(Π*,Π_untarg)<0.5·Π_full; D
log10(k*_untarg/k*_targeted)>0. The targeting claim also requires a MONOTONE dose-response of
R_restore in L that the layer-label shuffle does not exhibit (§A3.7). Nulls: matched-numel random
+ layer-label shuffle (T) + within-subspace direction-shuffle (T-lr), ≥10 seeds/budget (§A3.5). fp64
FULL SVD for the graft tail (WL.restore_svd_graft); the positive control asserts fp64↔deploy
agreement and HALTs on disagreement (§A3.3). Fail-loud; no try/except.

If arm2_locus is 'mixed' → targeting is unfalsifiable: run U / positive-control frontier only, disclosed.

Amendment (2026-06-02, restore ≠ unmasking): the winning restoring edit is re-scored under the
W14-R bounded transform set (infra.pg.recovery_transforms) + on a held-out knowledge set, reporting
transform-robust R_restore (genuine, the min over transforms / held-out-generalizing) vs transform-
fragile (unmask-like). Arm 3 records that it ran under the SATISFIED W14-R H_deletion gate.

    pg run infra.aws_pg.jobs.w14_arm3 --type g6e.8xlarge \
        --args="--arm2-input s3://pg-lineage-769063704696/results/w14_arm2_module/out/arm2_module_reset.json \
                --arm2-layer-rank-s3 s3://.../arm2_layer_sweep.json \
                --controls-s3 s3://.../w14_arm23_controls/out/legA_controls.json \
                --w14r-gate-s3 s3://.../w14r_recovery/out/metrics.json" \
        --s3-prefix results/w14_arm3_edit --budget-min 180
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

from infra.aws_pg.jobs.e4_compress_curve import _set_decoder
from infra.aws_pg.jobs.w11_legA_unblind import _boot_S
from infra.pg import w14_localize as WL
from infra.pg import recovery_transforms as RT      # W14-R transform set (Finding #12; pure, no GPU)
from infra.pg.w5_verdict import _bh_fdr             # BH-FDR over the 4 cells (prereg §A3.4)

SEED = 1337
N_BOOT = 2000
TAU_RESTORE = 0.50          # frozen now, unconditional (prereg §A3.6A)
DELTA_TARGET = 0.10         # Cell B gap (prereg §A3.6C)
FP64_DEPLOY_TOL = 1e-2      # fp64↔deploy graft RELATIVE max|Δ| HALT tol (bf16 ~2^-8 round-off; §A3.3)
LAYER_LADDER = (1, 2, 4, 8, 16, 26)
RANK_LADDER = (8, 32, 128, 512, 1024)
NULL_SEEDS = 10             # ≥10 seeds/budget for null + shuffle (prereg §A3.5)

# W14-R §4 frozen held-out knowledge subjects (restore≠unmasking generalization set; amendment §1).
HELD_OUT = ["college_chemistry", "astronomy", "machine_learning", "college_mathematics",
            "philosophy", "high_school_world_history", "professional_law", "econometrics",
            "us_foreign_policy", "nutrition"]


def _cp(s3, dst):
    subprocess.run(["aws", "s3", "cp", s3, str(dst), "--region", "us-east-1"], check=True)
    return json.loads(pathlib.Path(dst).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", required=True)
    ap.add_argument("--arm2-input", required=True, help="arm2_module_reset.json (locus verdict)")
    ap.add_argument("--arm2-layer-rank-s3", default=None, help="arm2_layer_sweep.json (ordered layers)")
    ap.add_argument("--w14r-gate-s3", required=True,
                    help="W14-R metrics.json; Arm 3 is gated on its verdict == H_deletion (amendment §1)")
    ap.add_argument("--mmlu-n-per-subject", type=int, default=0)
    ap.add_argument("--null-seeds", type=int, default=NULL_SEEDS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from infra.pg import diff_harvest as H
    H.hf_online_setup()
    from infra.pg import text_eval as TE

    device = "cuda"
    assert torch.cuda.is_available(), "W14 Arm 3 needs GPU."
    n_per = (2 if a.smoke else (a.mmlu_n_per_subject or None))
    null_seeds = (2 if a.smoke else a.null_seeds)

    ctrl = _cp(a.controls_s3, out / "legA_controls.json")
    assert ctrl["CONTROLS_OK"] is True
    item_hashes = ctrl["item_hashes"]
    before_pi = ctrl["mmlu_blank"]["BEFORE"]["per_item"]
    floor_pi = ctrl["mmlu_blank"]["FLOOR"]["per_item"]

    arm2 = _cp(a.arm2_input, out / "arm2_module_reset.json")
    locus = {"H_MLP": "mlp", "H_attn": "attn"}.get(arm2.get("VERDICT"), "mixed")
    print(f"[W14-A3] arm2 verdict={arm2.get('VERDICT')} → locus={locus} gpu={torch.cuda.get_device_name(0)}",
          flush=True)

    # W14-R gate (amendment §1): Arm 3 is well-posed only if the capability did NOT return from a
    # prompt alone. The primary (text) forward verdict must be H_deletion; HALT otherwise (no fold).
    w14r = _cp(a.w14r_gate_s3, out / "w14r_recovery.json")
    w14r_verdict = w14r["verdict"]["text"]
    assert w14r_verdict == "H_deletion", (
        f"W14-R gate NOT satisfied: verdict(text)={w14r_verdict} (need H_deletion) — Arm 3 is the "
        f"wrong question if the capability returns from a prompt alone (amendment §1).")
    print(f"[W14-A3] W14-R gate satisfied: verdict(text)={w14r_verdict}", flush=True)
    # arm2 layer ranking (descending credit); fall back to depth order if absent
    if a.arm2_layer_rank_s3:
        ls = _cp(a.arm2_layer_rank_s3, out / "arm2_layer_sweep.json")
        layer_rank = [int(L) for L, _ in sorted(ls["per_layer_R_mmlu"].items(),
                                                key=lambda kv: -kv[1]["R_mmlu"])]
    else:
        layer_rank = list(range(26))

    pg_after, pg_before, proc, paths, _it, gate = H.build_pair("base-pt", device)
    lm_edit = pg_before.get_submodule(paths["language_model"])
    after_sd = {k: v.detach().clone().cpu() for k, v in pg_after.get_submodule(paths["language_model"]).state_dict().items()}
    base_sd = {k: v.detach().clone().cpu() for k, v in lm_edit.state_dict().items()}
    del pg_after
    torch.cuda.empty_cache()
    key_numel = {k: int(after_sd[k].numel()) for k in WL.decoder_keys()}
    pi_full = WL.numel_of(after_sd, WL.decoder_keys() + WL.norm_keys(after_sd.keys()))

    def set_layer_restore(module, layers):
        keys = [k for k in WL.module_keys(module) if int(k.split(".")[1]) in set(layers)]
        _set_decoder(lm_edit, after_sd, WL.reset_overrides(after_sd, base_sd, keys), device)
        return WL.numel_of(after_sd, keys), keys

    # rank-k restore via the registered fp64 FULL SVD graft (prereg §A3.3 — WL.restore_svd_graft uses
    # torch.linalg.svd, NOT svd_lowrank/fp32). Deploy dtype = bf16; fp64↔deploy agreement is asserted
    # on the positive control below (Finding #11).
    def set_rank_restore(keys, k):
        ov = {key: WL.restore_svd_graft(after_sd[key], base_sd[key], rank=k).to(torch.bfloat16)
              for key in keys}
        _set_decoder(lm_edit, after_sd, ov, device)
        pi = sum(int(min(k, min(after_sd[key].shape))) * (after_sd[key].shape[0] + after_sd[key].shape[1])
                 for key in keys)
        return pi

    def score_R(which="confirmation"):
        b = TE.mmlu_blankimg(pg_before, proc, device, n_per_subject=n_per, gray=128)
        assert b["per_item_hash"] == item_hashes, "MMLU item order differs from controls — abort."
        sel = [i for i, h in enumerate(item_hashes) if (int(h, 16) % 2 == 1)] if which == "confirmation" else list(range(len(item_hashes)))
        a_pi = [b["per_item"][i] for i in sel]
        bf = [before_pi[i] for i in sel]
        fl = [floor_pi[i] for i in sel]
        mean, lo, hi, _ = _boot_S(a_pi, bf, fl, n_boot=N_BOOT, seed=SEED)
        denom = np.mean(bf) - np.mean(fl)
        point = (np.mean(a_pi) - np.mean(fl)) / denom if denom > 0 else float("nan")
        # keep the confirmation-selected per-item edit array so the Π* crossing point is bootstrapped
        # PAIRED against before/floor (prereg §A3.7: no bare point estimates on the crossing).
        return {"R_restore": float(point), "ci95": [lo, hi], "acc_pct": b["acc_pct"],
                "per_item": a_pi}

    # confirmation-half before/floor per-item arrays — the paired bootstrap denominator anchors.
    conf_sel = [i for i, h in enumerate(item_hashes) if (int(h, 16) % 2 == 1)]
    conf_before = np.asarray([before_pi[i] for i in conf_sel], float)
    conf_floor = np.asarray([floor_pi[i] for i in conf_sel], float)
    # §A3.6A degenerate-denom HALT (fail loud, no silent nan): the R_restore scale is BEFORE−FLOOR.
    _denom = float(conf_before.mean() - conf_floor.mean())
    assert _denom > 0.05, (f"degenerate R_restore denom: BEFORE−FLOOR={_denom:.4f} ≤ 0.05 on the "
                           f"confirmation half — HALT (§A3.6A); the re-frozen controls have no dynamic range.")

    families = {}

    # positive control: full restore (= base decoder). The full-decoder restore at full rank IS the
    # rank-k graft with k→full, so it exercises the fp64 graft path; ASSERT the fp64 graft and the
    # deploy-precision (bf16) graft AGREE within tol over the whole decoder, HALT on disagreement
    # (prereg §A3.3 / Finding #11 — no try/except, fail loud).
    # agreement = relative max|Δ| (bf16 ~2^-8 round-off scales with weight magnitude; an absolute tol
    # would false-HALT on large-magnitude weights).
    fp64_agree = 0.0
    for key in WL.decoder_keys():
        graft64 = WL.restore_svd_graft(after_sd[key], base_sd[key], rank=min(after_sd[key].shape))
        graft_deploy = graft64.to(torch.bfloat16).double()
        scale = float(graft64.abs().max().item()) + 1e-12
        fp64_agree = max(fp64_agree, float((graft64 - graft_deploy).abs().max().item()) / scale)
    assert fp64_agree <= FP64_DEPLOY_TOL, (
        f"fp64↔deploy graft DISAGREE: rel max|Δ|={fp64_agree:.3e} > tol {FP64_DEPLOY_TOL:.0e} — HALT (§A3.3).")
    # full restore = base decoder (matrices + norms); the positive control R_restore ≡ 1.
    _set_decoder(lm_edit, after_sd, WL.reset_overrides(after_sd, base_sd, WL.decoder_keys() + WL.norm_keys(after_sd.keys())), device)
    pc = score_R()
    print(f"[W14-A3] positive control (full restore) R_restore={pc['R_restore']:.3f} "
          f"fp64↔deploy max|Δ|={fp64_agree:.2e} (≤{FP64_DEPLOY_TOL:.0e})", flush=True)
    # §A3.5 positive-control HALT (fail loud): a FULL reset to the base decoder must recover ≈ the
    # whole gap (R_restore ≈ 1). If it does not, the restore/scoring harness is broken — HALT.
    assert abs(pc["R_restore"] - 1.0) < 0.05, (
        f"positive control (full restore = base decoder) R_restore={pc['R_restore']:.3f} ≠ 1.0±0.05 — "
        f"HALT (§A3.5): the restore harness is broken (a full reset must recover ~the full gap).")

    # U — untargeted full-decoder rank ladder
    fam_U = []
    for k in (RANK_LADDER if not a.smoke else (8, 128)):
        pi = set_rank_restore(WL.decoder_keys(), k)
        r = score_R()
        fam_U.append({"rank": k, "Pi_params": int(pi), **r})
        print(f"[W14-A3] U rank={k} Π={pi} R={r['R_restore']:.3f}", flush=True)
    families["U_untargeted_rank"] = fam_U

    layer_ladder = (LAYER_LADDER if not a.smoke else (1, 8))
    rank_ladder = (RANK_LADDER if not a.smoke else (8, 128))

    # direction-shuffle T-lr null: graft k RANDOM singular directions of the restore SVD (not the
    # top-k), matched rank k — the structure-destroying null for the Cell-D rank claim (prereg §A3.5).
    def set_rank_restore_shuffled(keys, k, seed):
        g = torch.Generator().manual_seed(int(seed))
        ov = {}
        for key in keys:
            aW = after_sd[key].double()
            U, S, Vh = torch.linalg.svd(base_sd[key].double() - aW, full_matrices=False)
            r = S.shape[0]
            kk = min(k, r)
            pick = torch.randperm(r, generator=g)[:kk]
            dWk = (U[:, pick] * S[pick]) @ Vh[pick, :]
            ov[key] = (aW + dWk).to(torch.bfloat16)
        _set_decoder(lm_edit, after_sd, ov, device)

    if locus != "mixed":
        mkeys = WL.module_keys(locus)
        # T — targeted layer-restore at top-L arm2-ranked layers
        fam_T = []
        for L in layer_ladder:
            top_layers = layer_rank[:L]
            pi, _ = set_layer_restore(locus, top_layers)
            r = score_R()
            fam_T.append({"L": L, "layers": top_layers, "Pi_params": int(pi), **r})
            print(f"[W14-A3] T L={L} layers={top_layers[:4]}.. Π={pi} R={r['R_restore']:.3f}", flush=True)
        families["T_targeted_layer"] = fam_T

        # T-shuffle — layer-label shuffle null: same module, same L, RANDOM L locus layers (≥10 seeds
        # per L). The monotone-dose gate (§A3.7) is: T's R_restore rises monotonically in L while the
        # shuffle dose does not — that is what makes arm2_layer_rank load-bearing rather than assumed.
        all_locus_layers = list(range(WL.N_LAYERS))
        fam_Tsh = []
        for L in layer_ladder:
            for s in range(null_seeds):
                g = np.random.default_rng(SEED + s)
                sh_layers = sorted(g.choice(all_locus_layers, size=L, replace=False).tolist())
                pi, _ = set_layer_restore(locus, sh_layers)
                r = score_R()
                fam_Tsh.append({"L": L, "seed": SEED + s, "layers": sh_layers, "Pi_params": int(pi), **r})
            print(f"[W14-A3] T-shuffle L={L} ({null_seeds} seeds) "
                  f"mean R={np.mean([e['R_restore'] for e in fam_Tsh if e['L'] == L]):.3f}", flush=True)
        families["Tshuffle_layer_null"] = fam_Tsh

        # T-lr — targeted rank-k restore confined to the locus module (fp64 full-SVD graft)
        fam_Tlr = []
        for k in rank_ladder:
            pi = set_rank_restore(mkeys, k)
            r = score_R()
            fam_Tlr.append({"rank": k, "Pi_params": int(pi), **r})
            print(f"[W14-A3] T-lr rank={k} Π={pi} R={r['R_restore']:.3f}", flush=True)
        families["Tlr_targeted_rank"] = fam_Tlr

        # T-lr direction-shuffle — matched-rank random-direction null for the Cell-D rank claim
        # (§A3.5; ≥10 seeds per rank).
        fam_Tlr_sh = []
        for k in rank_ladder:
            for s in range(null_seeds):
                set_rank_restore_shuffled(mkeys, k, seed=SEED + s)
                r = score_R()
                fam_Tlr_sh.append({"rank": k, "seed": SEED + s, **r})
            print(f"[W14-A3] T-lr dir-shuffle rank={k} ({null_seeds} seeds) "
                  f"mean R={np.mean([e['R_restore'] for e in fam_Tlr_sh if e['rank'] == k]):.3f}", flush=True)
        families["Tlr_dirshuffle_null"] = fam_Tlr_sh

        # U-layer — untargeted layer-restore by descending |ΔW| rel-Frob (magnitude targeting)
        relfrob = {}
        for L in range(26):
            num = sum(float((after_sd[k].float() - base_sd[k].float()).norm() ** 2)
                      for k in WL.module_keys(locus) if int(k.split(".")[1]) == L)
            den = sum(float(after_sd[k].float().norm() ** 2)
                      for k in WL.module_keys(locus) if int(k.split(".")[1]) == L)
            relfrob[L] = (num / den) ** 0.5 if den > 0 else 0.0
        mag_rank = [L for L, _ in sorted(relfrob.items(), key=lambda kv: -kv[1])]
        fam_Ul = []
        for L in layer_ladder:
            pi, _ = set_layer_restore(locus, mag_rank[:L])
            r = score_R()
            fam_Ul.append({"L": L, "layers": mag_rank[:L], "Pi_params": int(pi), **r})
            print(f"[W14-A3] U-layer L={L} Π={pi} R={r['R_restore']:.3f}", flush=True)
        families["Ulayer_magnitude"] = fam_Ul

    # ---- Π* (monotone "stays ≥ τ at and above"; prereg §A3.6B) ----
    # Π* = smallest budget B such that EVERY frontier entry with Π ≥ B has R_restore ≥ τ (not merely
    # the first crossing). non_monotone flags when that differs from the first crossing (Finding #10).
    def _pi_star(fam, r_of):
        """fam = list of entries sorted by Π; r_of(entry) → R_restore. Returns (pi_star, first_cross,
        non_monotone). pi_star uses the monotone rule on the given R values."""
        es = sorted(fam, key=lambda e: e["Pi_params"])
        first_cross = next((e["Pi_params"] for e in es if r_of(e) >= TAU_RESTORE), None)
        pi_star = None
        for i, e in enumerate(es):                          # smallest B s.t. all entries ≥ B clear τ
            if all(r_of(e2) >= TAU_RESTORE for e2 in es[i:]):
                pi_star = e["Pi_params"]
                break
        return pi_star, first_cross, bool(pi_star != first_cross)

    def _point(e):
        return e["R_restore"]
    pi_star_T, first_cross_T, nonmono_T = (_pi_star(families["T_targeted_layer"], _point)
                                           if locus != "mixed" else (None, None, False))
    pi_untarg, first_cross_U, nonmono_U = _pi_star(families["U_untargeted_rank"], _point)

    # matched-Σnumel random null at Π* (the T crossing budget — prereg §A3.6C / Finding #8), ≥10 seeds
    null = []
    if locus != "mixed" and pi_star_T is not None:
        for s in range(null_seeds):
            chosen, achieved = WL.matched_numel_random_keys(WL.decoder_keys(), key_numel, pi_star_T, seed=SEED + s)
            _set_decoder(lm_edit, after_sd, WL.reset_overrides(after_sd, base_sd, chosen), device)
            r = score_R()
            null.append({"seed": SEED + s, "Pi_params": achieved, "target": int(pi_star_T), **r})
            print(f"[W14-A3] null@Π* seed={SEED+s} Π={achieved}/{pi_star_T} R={r['R_restore']:.3f}", flush=True)

    # ---- paired bootstrap over the four cells (prereg §A3.7: crossing/ratios bootstrapped, BH-FDR
    # over exactly {A,B,C,D}, a CI per cell — Findings #5/#6) ----
    def _pi_star_boot(fam, idx):
        """Monotone Π* on a single resample (R recomputed paired against before/floor on idx)."""
        es = sorted(fam, key=lambda e: e["Pi_params"])
        rs = [_boot_R(e["per_item"], idx) for e in es]
        for i, e in enumerate(es):
            if all(rs[j] >= TAU_RESTORE for j in range(i, len(es))):
                return e["Pi_params"]
        return None

    def _kstar_boot(fam, idx):
        """Monotone k* (rank crossing) on a resample; fam entries carry 'rank'."""
        es = sorted(fam, key=lambda e: e["rank"])
        rs = [_boot_R(e["per_item"], idx) for e in es]
        for i, e in enumerate(es):
            if all(rs[j] >= TAU_RESTORE for j in range(i, len(es))):
                return e["rank"]
        return None

    def _boot_R(per_item, idx):
        den = conf_before[idx].mean() - conf_floor[idx].mean()
        return ((np.asarray(per_item, float)[idx].mean() - conf_floor[idx].mean()) / den
                if den > 0 else float("nan"))

    n_conf = conf_before.shape[0]
    rng = np.random.default_rng(SEED)
    cellA_boot, cellD_boot, cellB_boot, cellC_boot = [], [], [], []
    # the T@Π* entry (its per-item array) for Cell B
    tstar = next((e for e in families.get("T_targeted_layer", []) if e["Pi_params"] == pi_star_T), None)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_conf, n_conf)
        psU = _pi_star_boot(families["U_untargeted_rank"], idx)            # untargeted Π* (always run)
        psT = _pi_star_boot(families["T_targeted_layer"], idx) if locus != "mixed" else None
        if psT and psU:
            cellA_boot.append(np.log10(psU / psT))
        pi_any_b = min([p for p in (psT, psU) if p], default=None)        # Cell C existence (bootstrapped)
        if pi_any_b is not None:
            cellC_boot.append(1.0 if pi_any_b < 0.5 * pi_full else 0.0)
        if locus != "mixed":
            ksT = _kstar_boot(families["Tlr_targeted_rank"], idx)
            ksU = _kstar_boot(families["U_untargeted_rank"], idx)
            if ksT and ksU:
                cellD_boot.append(np.log10(ksU / ksT))
            if tstar is not None and null:
                r_t = _boot_R(tstar["per_item"], idx)
                r_null = np.mean([_boot_R(e["per_item"], idx) for e in null])
                cellB_boot.append(r_t - r_null)

    def _ci(arr):
        return ([float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if len(arr) else None)

    cellA_point = (float(np.log10(pi_untarg / pi_star_T)) if (pi_star_T and pi_untarg) else None)
    cellA_ci = _ci(cellA_boot)
    cellA_p = (float(np.mean(np.asarray(cellA_boot) <= 0.0)) if cellA_boot else float("nan"))     # H0: ratio ≤ 0
    cellB_point = (float(_point(tstar) - np.mean([e["R_restore"] for e in null]))
                   if (tstar is not None and null) else None)
    cellB_ci = _ci(cellB_boot)
    cellB_p = (float(np.mean(np.asarray(cellB_boot) <= DELTA_TARGET)) if cellB_boot else float("nan"))  # H0: gap ≤ δ
    # Cell D point: log10(k*_untarg / k*_targeted) on the full sample, k* = monotone rank crossing.
    def _kstar_point(fam):
        es = sorted(fam, key=lambda e: e["rank"])
        for i, e in enumerate(es):
            if all(_point(es[j]) >= TAU_RESTORE for j in range(i, len(es))):
                return e["rank"]
        return None
    k_targeted = (_kstar_point(families["Tlr_targeted_rank"]) if locus != "mixed" else None)
    k_untarg = _kstar_point(families["U_untargeted_rank"])
    cellD_point = (float(np.log10(k_untarg / k_targeted)) if (k_targeted and k_untarg) else None)
    cellD_ci = _ci(cellD_boot)
    cellD_p = (float(np.mean(np.asarray(cellD_boot) <= 0.0)) if cellD_boot else float("nan"))     # H0: ratio ≤ 0

    pi_any = min([p for p in (pi_star_T, pi_untarg) if p], default=None)
    cellC_point = (bool(pi_any < 0.5 * pi_full) if pi_any else None)
    cellC_ci = _ci(cellC_boot)                                            # bootstrap existence rate
    cellC_p = (float(1.0 - np.mean(cellC_boot)) if cellC_boot else float("nan"))   # H0: NOT cheap (frac not-<0.5Πfull)

    # monotone dose-response gate (prereg §A3.7): T's R_restore rises (weakly) monotonically in L,
    # the layer-shuffle dose does not. Targeting claim requires (i) Cell A, (ii) Cell B, AND (iii) this.
    def _monotone_in_L(by_L):
        Ls = sorted(by_L)
        return all(by_L[Ls[i + 1]] >= by_L[Ls[i]] - 1e-9 for i in range(len(Ls) - 1))
    t_dose = {e["L"]: e["R_restore"] for e in families.get("T_targeted_layer", [])}
    sh_by_L = {}
    for e in families.get("Tshuffle_layer_null", []):
        sh_by_L.setdefault(e["L"], []).append(e["R_restore"])
    sh_dose = {L: float(np.mean(v)) for L, v in sh_by_L.items()}
    t_dose_monotone = bool(t_dose and _monotone_in_L(t_dose))
    shuffle_dose_monotone = bool(sh_dose and _monotone_in_L(sh_dose))

    # BH-FDR over EXACTLY the 4 cells {A,B,C,D} (prereg §A3.4 / Finding #6).
    cell_pairs = [("A", cellA_p), ("B", cellB_p), ("C", cellC_p), ("D", cellD_p)]
    rejected = _bh_fdr(cell_pairs, q=0.05)
    cells = {
        "A": {"name": "log10(Pi_untarg/Pi_star) > 0", "point": cellA_point, "ci95": cellA_ci,
              "p": cellA_p, "fdr_reject": "A" in rejected},
        "B": {"name": "R_restore(T@Pi*) - random@Pi* > delta_target", "point": cellB_point,
              "ci95": cellB_ci, "p": cellB_p, "fdr_reject": "B" in rejected},
        "C": {"name": "min(Pi*,Pi_untarg) < 0.5*Pi_full", "point": cellC_point, "ci95": cellC_ci,
              "p": cellC_p, "fdr_reject": "C" in rejected},
        "D": {"name": "log10(k*_untarg/k*_targeted) > 0", "point": cellD_point, "ci95": cellD_ci,
              "p": cellD_p, "fdr_reject": "D" in rejected},
    }
    # H_targeted requires Cell A (FDR) AND Cell B (FDR) AND monotone dose that the shuffle lacks.
    h_targeted = bool(cells["A"]["fdr_reject"] and cells["B"]["fdr_reject"]
                      and t_dose_monotone and not shuffle_dose_monotone)

    # ---- Finding #12: restore ≠ unmasking (amendment §1) ----
    # Re-score the BEST restoring edit under the W14-R FORMAT/reframe/few-shot transform set + a
    # held-out knowledge set. A genuine capability restore is format-robust; an unmask is fragile
    # under format change. We score the no-i18n transforms (format f1_* + reframe f2_textbook/booklet
    # + few-shot f3_*) — the confound-free unmask probe; the translation transforms need MMMLU i18n
    # AND degrade the BASE too (W14-R), so they are excluded here and logged. RT identity is byte-exact
    # to text_eval canonical (W14-R 8/8 regression), so the held-out canonical score reuses it.
    # Transform-robust R_restore = min over transforms; normalized by the canonical blank-image
    # before/floor (the W14-R R_recover convention). Winning edit = cheapest-targeted T@Π* if
    # H_targeted else the cheapest edit reaching τ (NEVER a shuffle/null family).
    edit_families = ("T_targeted_layer", "Tlr_targeted_rank", "U_untargeted_rank", "Ulayer_magnitude")
    edits = [dict(e, _fam=fam) for fam in edit_families for e in families.get(fam, []) if "Pi_params" in e]
    reaching = [e for e in edits if e["R_restore"] >= TAU_RESTORE]
    best_edit = (dict(tstar, _fam="T_targeted_layer") if (h_targeted and tstar is not None)
                 else min(reaching or edits, key=lambda e: e["Pi_params"], default=None))

    from collections import defaultdict
    from datasets import load_dataset
    from PIL import Image
    tok = proc.tokenizer
    ds_mmlu = load_dataset(TE.MMLU_REPO, "all")
    shots_by_subj = defaultdict(list)
    for r in ds_mmlu["dev"]:
        shots_by_subj[r["subject"]].append({"question": r["question"], "options": r["choices"],
                                            "answer": int(r["answer"]), "subject": r["subject"]})
    items_by_hash = {}
    for r in ds_mmlu["test"]:
        h = hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12]
        items_by_hash.setdefault(h, {"question": r["question"], "options": r["choices"],
                                     "answer": int(r["answer"]), "subject": r["subject"]})
    gray_img = Image.fromarray(np.full((224, 224, 3), 128, np.uint8))
    h2i = {h: i for i, h in enumerate(item_hashes)}

    def _score_under(t, hashes):
        cand = [tok.encode(c, add_special_tokens=False) for c in RT.answer_candidates(t)]
        assert all(len(c) == 1 for c in cand), f"answer candidates not single-token for {t}"
        cand = [c[0] for c in cand]
        peritem, B = [], 4
        for s0 in range(0, len(hashes), B):
            hb = hashes[s0:s0 + B]
            prompts = [RT.apply_transform(items_by_hash[h], t, shots_by_subj[items_by_hash[h]["subject"]]) for h in hb]
            ans = [items_by_hash[h]["answer"] for h in hb]
            inp = proc(text=prompts, images=[gray_img] * len(hb), return_tensors="pt", padding="longest").to(device)
            with torch.inference_mode():
                last = pg_before(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                                 pixel_values=inp["pixel_values"].to(torch.bfloat16), logits_to_keep=1).logits[:, -1, :]
            preds = last[:, cand].argmax(dim=-1).tolist()
            peritem += [int(p == a_) for p, a_ in zip(preds, ans)]
        return peritem

    def _R(peritem, hashes):
        bsel = np.asarray([before_pi[h2i[h]] for h in hashes], float)
        fsel = np.asarray([floor_pi[h2i[h]] for h in hashes], float)
        den = bsel.mean() - fsel.mean()
        return float((np.asarray(peritem, float).mean() - fsel.mean()) / den) if den > 0 else float("nan")

    restore_vs_unmask = {
        "held_out_subjects": HELD_OUT, "w14r_gate_verdict": w14r_verdict,
        "ran_under_satisfied_H_deletion_gate": True,
        "excluded_translations": [t[0] for t in RT.transform_set() if t[2] in RT.CONJ_LANG],
        "exclusion_reason": "translations need MMMLU i18n + degrade the BASE (W14-R confound); "
                            "the format arm is the confound-free unmask probe.",
        "best_edit": None, "per_transform_R": {}, "held_out_R": None,
        "transform_robust_R": None, "transform_min_t": None, "verdict": None,
    }

    # ---- DURABLE pre-#12 checkpoint (the arm2 write-before-risk lesson): serialize the core result
    # with the restore_vs_unmask PLACEHOLDER *before* the GPU-heavy #12 rescore, so a #12 failure
    # never discards the frontier scoring. restore_vs_unmask is mutated IN PLACE below and `result`
    # holds the same object, so the final re-write captures the filled values. ----
    def _slim(entries):
        return [{k: v for k, v in e.items() if k != "per_item"} for e in entries]
    families_out = {fam: _slim(es) for fam, es in families.items()}
    null_out = _slim(null)
    result = {
        "experiment": "w14_arm3_edit", "status": "confirmatory",
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        "forward_primary": "blank-image", "controls_s3": a.controls_s3, "arm2_input": a.arm2_input,
        "w14r_gate_s3": a.w14r_gate_s3, "w14r_gate_verdict": w14r_verdict,
        "arm2_locus": locus, "arm2_layer_rank": layer_rank, "Pi_full": int(pi_full),
        "tau_restore": TAU_RESTORE, "delta_target": DELTA_TARGET, "n_boot": N_BOOT,
        "positive_control_full_restore": {k: v for k, v in pc.items() if k != "per_item"},
        "fp64_deploy_rel_max_delta": fp64_agree,
        "families": families_out, "matched_numel_null": null_out,
        "Pi_star_targeted": pi_star_T, "Pi_star_first_crossing": first_cross_T, "Pi_star_non_monotone": nonmono_T,
        "Pi_untargeted": pi_untarg, "Pi_untarg_first_crossing": first_cross_U, "Pi_untarg_non_monotone": nonmono_U,
        "k_star_targeted": k_targeted, "k_star_untargeted": k_untarg,
        "cells": cells, "fdr_rejected": sorted(rejected),
        "t_dose_monotone": t_dose_monotone, "shuffle_dose_monotone": shuffle_dose_monotone,
        "H_targeted": h_targeted,
        "restore_vs_unmask": restore_vs_unmask,
        "swap_gate": gate,
        "honest_scope": ("targeting unfalsifiable (arm2 locus mixed) — U/positive frontier only"
                         if locus == "mixed" else "targeting tested by intervention vs U-layer + matched-null"),
    }
    (out / "arm3_minimal_edit.json").write_text(json.dumps(result, indent=2, default=str))
    print("[W14-A3] core result checkpoint written (pre-#12 rescore)", flush=True)

    orig_side = tok.padding_side
    tok.padding_side = "left"
    if best_edit is not None and best_edit.get("R_restore", 0.0) >= TAU_RESTORE:
        if best_edit["_fam"] in ("T_targeted_layer", "Ulayer_magnitude"):
            set_layer_restore(locus, best_edit["layers"])          # re-apply the winning edit
        elif best_edit["_fam"] == "Tlr_targeted_rank":
            set_rank_restore(mkeys, best_edit["rank"])
        else:
            set_rank_restore(WL.decoder_keys(), best_edit["rank"])
        conf_h = [h for h in item_hashes if int(h, 16) % 2 == 1]
        for t in [tr for tr in RT.transform_set() if tr[2] not in RT.CONJ_LANG]:
            R = _R(_score_under(t, conf_h), conf_h)
            restore_vs_unmask["per_transform_R"][t[0]] = R
            print(f"[W14-A3] #12 transform {t[0]} R_restore={R:.3f}", flush=True)
        ident = next(tr for tr in RT.transform_set() if tr[0] == "identity")
        ho_h = [h for h in item_hashes if items_by_hash.get(h, {}).get("subject") in set(HELD_OUT)]
        if ho_h:
            restore_vs_unmask["held_out_R"] = _R(_score_under(ident, ho_h), ho_h)
        rmin = min(restore_vs_unmask["per_transform_R"].values())
        restore_vs_unmask["transform_robust_R"] = rmin
        restore_vs_unmask["transform_min_t"] = min(restore_vs_unmask["per_transform_R"],
                                                   key=restore_vs_unmask["per_transform_R"].get)
        restore_vs_unmask["verdict"] = ("genuine_restore" if rmin >= TAU_RESTORE
                                        else "transform_fragile_unmask_like")
        restore_vs_unmask["best_edit"] = {k: v for k, v in best_edit.items() if k != "per_item"}
        print(f"[W14-A3] #12 transform-robust R={rmin:.3f} ({restore_vs_unmask['transform_min_t']}) "
              f"held_out_R={restore_vs_unmask['held_out_R']} → {restore_vs_unmask['verdict']}", flush=True)
    else:
        restore_vs_unmask["verdict"] = "no_restoring_edit_reached_tau"
        print("[W14-A3] #12: no edit reached τ_restore — restore≠unmask check N/A", flush=True)
    tok.padding_side = orig_side

    # #12 filled restore_vs_unmask IN PLACE (same object referenced by `result`); persist the final
    # result + a standalone restore_vs_unmask.json (the core result was already checkpointed pre-#12).
    (out / "arm3_minimal_edit.json").write_text(json.dumps(result, indent=2, default=str))
    (out / "restore_vs_unmask.json").write_text(json.dumps(restore_vs_unmask, indent=2, default=str))
    print(f"[W14-A3] Π*_T={pi_star_T}(first={first_cross_T},nonmono={nonmono_T}) Π_untarg={pi_untarg} "
          f"cellA={cellA_point} ci={cellA_ci} FDR_reject={sorted(rejected)} "
          f"dose(T_mono={t_dose_monotone},sh_mono={shuffle_dose_monotone}) H_targeted={h_targeted}")
    print("W14_ARM3_OK", json.dumps({"locus": locus, "H_targeted": h_targeted,
                                     "fdr_rejected": sorted(rejected)}))


if __name__ == "__main__":
    main()
