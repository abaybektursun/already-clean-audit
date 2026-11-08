"""W14 ARM 2 — localize the MMLU loss: MLP vs attention (the project MDE).

Reset a SUBSET of PG2-pt's decoder weights back to base gemma-2-2b and measure how much
free-standing MMLU competence returns. Recovery-by-restoration (Hase T1): a module is
load-bearing iff resetting ONLY it recovers a pre-specified fraction of the gap. Inference-only
weight-swaps, no training. Blank-image MMLU is PRIMARY (§1); text-only is secondary.

Stages (--stage):
  module     — THE PROJECT MDE. 4 arms {none, mlp_only, attn_only, full(=base decoder, pos ctrl)}
               + matched-Σnumel random null (≥5 seeds, per single-module budget, GQA-asymmetric)
               + structure-shuffle reset + norm-only sensitivity. Orders H_MLP / H_attn /
               H_distributed via the MLP−attn difference CI. Run ONCE post-freeze.
  layers     — per-layer reset of the winning --module across the confirmation layer-half, +
               cumulative early→late, + random-single-layer null. Locates the load-bearing band.
  withinmlp  — within-MLP component necessity scored on MMLU: top-N SVD of ΔW grafted onto base,
               necessity gap = R_mmlu(top-N) − R_mmlu(random-N-in-span) (re-derived on MMLU, never
               reusing W13's grounding-derived set).

R_mmlu = (acc_arm − acc_floor)/(acc_before − acc_floor), paired bootstrap over items vs the
re-frozen legA_controls.json (read from --controls-s3, NOT W11's verbatim file). Confirmation
item-half (hash parity) is binding; discovery + full reported for context. Fail-loud; no try/except.

    pg run infra.aws_pg.jobs.w14_arm2 \
        --args="--stage module --controls-s3 s3://pg-lineage-769063704696/results/w14_arm23_controls/out/legA_controls.json" \
        --type g6e.8xlarge --s3-prefix results/w14_arm2_module --budget-min 120
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

from infra.aws_pg.jobs.e4_compress_curve import _set_decoder
from infra.aws_pg.jobs.w11_legA_unblind import _boot_S
from infra.pg import w14_localize as WL

SEED = 1337
N_BOOT = 2000
DELTA_MOD = 0.10                  # §A2.6-6A τ_loc margin = Arm-2 SESOI δ_mod (inherited frozen δ=0.10)


def _read_controls(out, controls_s3):
    """Read the re-frozen legA_controls.json (instance-role aws cli; provenance, never transcribed)."""
    local = out / "legA_controls.json"
    subprocess.run(["aws", "s3", "cp", controls_s3, str(local), "--region", "us-east-1"], check=True)
    ctrl = json.loads(local.read_text())
    assert ctrl["CONTROLS_OK"] is True, "controls did NOT pass (CONTROLS_OK=False) — fix harness, do not unblind."
    return ctrl


def _subset(arr, item_hashes, which):
    """Item-half by hash parity: confirmation=odd, discovery=even, all=full (prereg §4.2)."""
    if which == "all":
        return list(arr)
    par = 1 if which == "confirmation" else 0
    return [v for v, h in zip(arr, item_hashes) if int(h, 16) % 2 == par]


def _boot_delta(a_pi, b_pi, before_pi, floor_pi, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap of R_mmlu(a) − R_mmlu(b) on the SAME item resample (the MLP−attn cut)."""
    import numpy as np
    a = np.asarray(a_pi, float); b = np.asarray(b_pi, float)
    bf = np.asarray(before_pi, float); fl = np.asarray(floor_pi, float)
    n = a.size
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        den = bf[idx].mean() - fl[idx].mean()
        if den > 0:
            ra = (a[idx].mean() - fl[idx].mean()) / den
            rb = (b[idx].mean() - fl[idx].mean()) / den
            ds.append(ra - rb)
    ds = np.asarray(ds)
    return float(ds.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def _boot_null_ci_upper(null_per_item, before_pi, floor_pi, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI-upper of the matched-numel-null R_mmlu (§A2.6-6A τ_loc input; null seeds pooled,
    paired item resample shared across seeds)."""
    import numpy as np
    seeds = [np.asarray(p, float) for p in null_per_item]
    bf = np.asarray(before_pi, float); fl = np.asarray(floor_pi, float)
    n = bf.size
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        den = bf[idx].mean() - fl[idx].mean()
        if den > 0:
            for a in seeds:
                rs.append((a[idx].mean() - fl[idx].mean()) / den)
    rs = np.asarray(rs)
    return float(np.percentile(rs, 97.5))


def _boot_superadd(full_pi, mlp_pi, attn_pi, before_pi, floor_pi, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap of the super-additivity statistic R_mmlu(full) − [R_mmlu(mlp) + R_mmlu(attn)]
    on the SAME item resample (§A2.6-6A H_distributed: must lie OUTSIDE its CI, i.e. CI-lower > 0)."""
    import numpy as np
    f = np.asarray(full_pi, float); m = np.asarray(mlp_pi, float); a = np.asarray(attn_pi, float)
    bf = np.asarray(before_pi, float); fl = np.asarray(floor_pi, float)
    n = f.size
    rng = np.random.default_rng(seed)
    ss = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        den = bf[idx].mean() - fl[idx].mean()
        if den > 0:
            flm = fl[idx].mean()
            rf = (f[idx].mean() - flm) / den
            rm = (m[idx].mean() - flm) / den
            ra = (a[idx].mean() - flm) / den
            ss.append(rf - (rm + ra))
    ss = np.asarray(ss)
    return float(ss.mean()), float(np.percentile(ss, 2.5)), float(np.percentile(ss, 97.5))


def _score(model, proc, gemma_tok, device, n_per, text_secondary=True):
    from infra.pg import text_eval as TE
    blank = TE.mmlu_blankimg(model, proc, device, n_per_subject=n_per, gray=128)
    text = TE.mmlu_accuracy(model, gemma_tok, device, n_per_subject=n_per) if text_secondary else None
    return blank, text


def _r_block(arm_pi, item_hashes, before_pi, floor_pi):
    """R_mmlu point + paired-bootstrap CI on confirmation (binding), discovery, and full halves."""
    blocks = {}
    for which in ("confirmation", "discovery", "all"):
        a = _subset(arm_pi, item_hashes, which)
        bf = _subset(before_pi, item_hashes, which)
        fl = _subset(floor_pi, item_hashes, which)
        mean, lo, hi, nb = _boot_S(a, bf, fl, n_boot=N_BOOT, seed=SEED)
        import numpy as np
        denom = np.mean(bf) - np.mean(fl)
        point = (np.mean(a) - np.mean(fl)) / denom if denom > 0 else float("nan")
        blocks[which] = {"R_mmlu": float(point), "boot_mean": mean, "ci95": [lo, hi], "n": len(a)}
    return blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pg-variant", choices=["pt", "mix"], default="pt",
                    help="base→pt (W14) or base→mix (second-ΔW generality gate)")
    ap.add_argument("--controls-s3", required=True)
    ap.add_argument("--stage", choices=["module", "layers", "withinmlp"], default="module")
    ap.add_argument("--module", choices=["mlp", "attn"], default="mlp",
                    help="winning module from the module stage (for layers/withinmlp)")
    ap.add_argument("--null-seeds", type=int, default=5, help="matched-numel random null seeds")
    ap.add_argument("--n-top", type=int, default=64, help="within-mlp top-N components")
    ap.add_argument("--draws", type=int, default=5, help="within-mlp random-N draws")
    ap.add_argument("--mmlu-n-per-subject", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from infra.pg import diff_harvest as H
    H.hf_online_setup()
    from transformers import AutoTokenizer
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available(), "W14 Arm 2 needs GPU."
    device = "cuda"
    n_per = (2 if a.smoke else (a.mmlu_n_per_subject or None))
    null_seeds = (2 if a.smoke else a.null_seeds)
    print(f"[W14-A2] stage={a.stage} module={a.module} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0)} smoke={a.smoke}", flush=True)

    ctrl = _read_controls(out, a.controls_s3)
    item_hashes = ctrl["item_hashes"]
    before_pi = ctrl["mmlu_blank"]["BEFORE"]["per_item"]      # PRIMARY = blank-image denominators
    floor_pi = ctrl["mmlu_blank"]["FLOOR"]["per_item"]
    assert len(before_pi) == len(floor_pi) == len(item_hashes), "controls per-item misaligned."

    gemma_tok = AutoTokenizer.from_pretrained(ENTITIES["L0"].repo_id, revision=ENTITIES["L0"].revision)

    # --- build the pair; clone decoder state_dicts; the edit host is pg_before.LM ---
    pg_key = "PG_mix" if a.pg_variant == "mix" else "PG"
    print(f"[W14-A2] pg_variant={a.pg_variant} (pg_key={pg_key})", flush=True)
    pg_after, pg_before, proc, paths, _it, gate = H.build_before_after(device, pg_key=pg_key)
    lm_edit = pg_before.get_submodule(paths["language_model"])
    after_sd = {k: v.detach().clone().cpu() for k, v in pg_after.get_submodule(paths["language_model"]).state_dict().items()}
    base_sd = {k: v.detach().clone().cpu() for k, v in lm_edit.state_dict().items()}
    key_numel = {k: int(after_sd[k].numel()) for k in WL.decoder_keys()}
    norm_ks = WL.norm_keys(after_sd.keys())

    def set_arm(reset_keys, shuffle_seed=None):
        if shuffle_seed is None:
            ov = WL.reset_overrides(after_sd, base_sd, reset_keys)
        else:
            ov = WL.shuffled_reset_overrides(after_sd, base_sd, reset_keys, shuffle_seed)
        _set_decoder(lm_edit, after_sd, ov, device)

    def hash_assert(d):
        assert d["per_item_hash"] == item_hashes, "MMLU item order differs from controls — abort."

    # ============================ STAGE: module (MDE) ============================
    if a.stage == "module":
        arms = {}

        # none = AFTER (the gap source); score on pg_after directly, then free it.
        nb, nt = _score(pg_after, proc, gemma_tok, device, n_per)
        hash_assert(nb)
        arms["none"] = {"blank": nb, "text": nt}
        del pg_after
        torch.cuda.empty_cache()

        reset_sets = {
            "mlp_only": WL.module_keys("mlp"),               # 78 matrices, norms held at PG
            "attn_only": WL.module_keys("attn"),             # 104 matrices, norms held
            "full": WL.module_keys("all") + norm_ks,         # 287 keys = base decoder (positive control)
            "norm_only": norm_ks,                            # H_norm sensitivity arm
        }
        for name, keys in reset_sets.items():
            set_arm(keys)
            b, t = _score(pg_before, proc, gemma_tok, device, n_per)
            hash_assert(b)
            arms[name] = {"blank": b, "text": t, "n_reset_keys": len(keys)}
            print(f"[W14-A2] {name}: blank={b['acc_pct']:.2f} ({len(keys)} keys)", flush=True)

        # matched-Σnumel random null (≥5 seeds) for each single-module budget (GQA-asymmetric).
        nulls = {}
        for mod in ("mlp", "attn"):
            target = WL.numel_of(after_sd, WL.module_keys(mod))
            seeds_out = []
            for s in range(null_seeds):
                chosen, achieved = WL.matched_numel_random_keys(WL.decoder_keys(), key_numel, target, seed=SEED + s)
                set_arm(chosen)
                b, _ = _score(pg_before, proc, gemma_tok, device, n_per, text_secondary=False)
                hash_assert(b)
                seeds_out.append({"seed": SEED + s, "achieved_numel": achieved, "target_numel": target,
                                  "blank": {"acc_pct": b["acc_pct"], "per_item": b["per_item"]}})
                print(f"[W14-A2] null({mod}) seed={SEED+s} achieved/target={achieved}/{target} "
                      f"blank={b['acc_pct']:.2f}", flush=True)
            nulls[mod] = seeds_out

        # structure-destroying shuffle reset (Frobenius-preserved) for each single module.
        shuffles = {}
        for mod in ("mlp", "attn"):
            set_arm(WL.module_keys(mod), shuffle_seed=SEED)
            b, _ = _score(pg_before, proc, gemma_tok, device, n_per, text_secondary=False)
            hash_assert(b)
            shuffles[mod] = {"blank": {"acc_pct": b["acc_pct"], "per_item": b["per_item"]}}
            print(f"[W14-A2] shuffle({mod}): blank={b['acc_pct']:.2f}", flush=True)

        # ---- analysis: R_mmlu per arm (blank primary), MLP−attn delta, additivity, τ_loc ----
        R = {name: _r_block(arms[name]["blank"]["per_item"], item_hashes, before_pi, floor_pi)
             for name in arms}
        Rt = {name: _r_block(arms[name]["text"]["per_item"], item_hashes, before_pi, floor_pi)
              for name in arms if arms[name].get("text")}      # text-only secondary (vs blank denom; sensitivity)
        delta = {which: _boot_delta(_subset(arms["mlp_only"]["blank"]["per_item"], item_hashes, which),
                                    _subset(arms["attn_only"]["blank"]["per_item"], item_hashes, which),
                                    _subset(before_pi, item_hashes, which),
                                    _subset(floor_pi, item_hashes, which))
                 for which in ("confirmation", "discovery", "all")}
        # POSITIVE-CONTROL + IDENTITY controls (§A2.5 / §A2.9 Step 2). The method must detect a known
        # effect: full reset (=base decoder) ⇒ R_mmlu(full)≈1 (two-sided). IDENTITY: no reset (=AFTER)
        # ⇒ AFTER shows NO RECOVERED capability ⇒ R_mmlu(none) ≤ δ_mod — ONE-SIDED: AFTER scoring BELOW
        # the random floor on the blank-image (PRIMARY) forward is the EXPECTED H_deletion signature
        # (W14-R saw blank identity R≈−0.16), NOT a leak, so the lower side is not gated. Computed here;
        # ASSERTED below AFTER the result JSON is durably written (a control HALT must never discard the
        # multi-hour scoring). No try/except.
        pos_ctrl_ok = bool(abs(R["full"]["all"]["R_mmlu"] - 1.0) <= DELTA_MOD)
        identity_ok = bool(R["none"]["all"]["R_mmlu"] <= DELTA_MOD)

        # τ_loc on the DISCOVERY half: max(0.5, matched-numel-null CI-upper + δ_mod) (§A2.6-6A) — controls-beat-null.
        def _null_ci_upper(mod):
            null_pi = [_subset(s["blank"]["per_item"], item_hashes, "discovery") for s in nulls[mod]]
            return _boot_null_ci_upper(null_pi, _subset(before_pi, item_hashes, "discovery"),
                                       _subset(floor_pi, item_hashes, "discovery"))
        tau_inputs = {mod: _null_ci_upper(mod) for mod in ("mlp", "attn")}
        tau_loc = max(0.5, max(tau_inputs["mlp"], tau_inputs["attn"]) + DELTA_MOD)

        rc = lambda name: R[name]["confirmation"]            # binding half
        mlp_lb, attn_lb = rc("mlp_only")["ci95"][0], rc("attn_only")["ci95"][0]
        d_conf = delta["confirmation"]
        # H_distributed super-additivity: R(full) − [R(mlp)+R(attn)] must lie OUTSIDE its bootstrap CI
        # (§A2.6-6A) — i.e. CI-lower > 0 (super-additive), not a bare point comparison.
        superadd_conf = _boot_superadd(_subset(arms["full"]["blank"]["per_item"], item_hashes, "confirmation"),
                                       _subset(arms["mlp_only"]["blank"]["per_item"], item_hashes, "confirmation"),
                                       _subset(arms["attn_only"]["blank"]["per_item"], item_hashes, "confirmation"),
                                       _subset(before_pi, item_hashes, "confirmation"),
                                       _subset(floor_pi, item_hashes, "confirmation"))
        if mlp_lb >= tau_loc and d_conf[1] > 0:
            verdict = "H_MLP"
        elif attn_lb >= tau_loc and d_conf[2] < 0:
            verdict = "H_attn"
        elif mlp_lb < tau_loc and attn_lb < tau_loc and superadd_conf[1] > 0:
            verdict = "H_distributed"
        else:
            verdict = "inconclusive"

        result = {
            "experiment": "w14_arm2_module", "status": "confirmatory", "stage": "module",
            "pg_variant": a.pg_variant,
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
            "forward_primary": "blank-image", "controls_s3": a.controls_s3,
            "frozen_denominator_blank": ctrl["frozen_denominator_mmlu_acc"],
            "arms_acc_pct": {name: arms[name]["blank"]["acc_pct"] for name in arms},
            "R_mmlu_blank": R, "R_mmlu_text": Rt,
            "mlp_minus_attn_delta": {"confirmation": d_conf, "discovery": delta["discovery"], "all": delta["all"]},
            "superadditivity_full_minus_sum_conf": {"mean": superadd_conf[0], "ci95": [superadd_conf[1], superadd_conf[2]],
                                                    "outside_ci": bool(superadd_conf[1] > 0)},
            "matched_numel_null": nulls, "shuffle_reset": shuffles,
            "tau_loc_discovery": tau_loc, "tau_margin": DELTA_MOD, "null_ci_upper_discovery": tau_inputs,
            "VERDICT": verdict, "swap_gate": gate,
            "pos_ctrl_ok": pos_ctrl_ok, "identity_ok": identity_ok,
            "R_mmlu_full_all": R["full"]["all"]["R_mmlu"], "R_mmlu_none_all": R["none"]["all"]["R_mmlu"],
            "per_item": {name: arms[name]["blank"]["per_item"] for name in arms},
        }
        (out / "arm2_module_reset.json").write_text(json.dumps(result, indent=2, default=str))
        print(f"[W14-A2] MDE VERDICT={verdict} (MLP R={rc('mlp_only')['R_mmlu']:.3f} "
              f"attn R={rc('attn_only')['R_mmlu']:.3f} full R={rc('full')['R_mmlu']:.3f} τ_loc={tau_loc:.3f})")
        print("W14_ARM2_MODULE_OK", json.dumps({"verdict": verdict, "pos_ctrl_ok": pos_ctrl_ok,
                                               "identity_ok": identity_ok}))
        # fail-loud control HALTs — AFTER the JSON is durably written (never discard the scoring).
        assert pos_ctrl_ok, (f"POS-CTRL FAIL: full-reset R_mmlu={R['full']['all']['R_mmlu']:.3f} not ≈1 "
                             f"(±{DELTA_MOD}) — instrument cannot restore; HALT (result JSON written).")
        assert identity_ok, (f"IDENTITY FAIL: none R_mmlu={R['none']['all']['R_mmlu']:.3f} > {DELTA_MOD} "
                             f"— AFTER shows recovered capability (leak); HALT (result JSON written).")
        return

    # ============================ STAGE: layers ============================
    if a.stage == "layers":
        del pg_after
        torch.cuda.empty_cache()
        conf_layers = list(range(0, 26, 2))                  # confirmation = even {0,2,..,24}
        if a.smoke:
            conf_layers = [12]
        mod_keys_by_layer = {L: [k for k in WL.module_keys(a.module) if int(k.split(".")[1]) == L]
                             for L in range(26)}
        per_layer, cumulative = {}, {}
        for L in conf_layers:
            set_arm(mod_keys_by_layer[L])
            b, _ = _score(pg_before, proc, gemma_tok, device, n_per, text_secondary=False)
            hash_assert(b)
            per_layer[L] = _r_block(b["per_item"], item_hashes, before_pi, floor_pi)["confirmation"]
            print(f"[W14-A2-L] layer {L}: R_mmlu={per_layer[L]['R_mmlu']:.3f}", flush=True)
        cum_keys = []
        for L in range(26):                                  # cumulative early→late
            cum_keys += mod_keys_by_layer[L]
            if L in (3, 7, 12, 17, 21, 25):
                set_arm(cum_keys)
                b, _ = _score(pg_before, proc, gemma_tok, device, n_per, text_secondary=False)
                hash_assert(b)
                cumulative[L] = _r_block(b["per_item"], item_hashes, before_pi, floor_pi)["confirmation"]
                print(f"[W14-A2-L] cumulative 0..{L}: R_mmlu={cumulative[L]['R_mmlu']:.3f}", flush=True)
        result = {"experiment": "w14_arm2_layers", "status": "confirmatory", "stage": "layers",
                  "module": a.module, "torch": torch.__version__, "controls_s3": a.controls_s3,
                  "per_layer_R_mmlu": per_layer, "cumulative_R_mmlu": cumulative, "swap_gate": gate}
        (out / "arm2_layer_sweep.json").write_text(json.dumps(result, indent=2, default=str))
        print("W14_ARM2_LAYERS_OK", json.dumps({"module": a.module, "n_layers": len(per_layer)}))
        return

    # ============================ STAGE: withinmlp ============================
    if a.stage == "withinmlp":
        del pg_after
        torch.cuda.empty_cache()
        mkeys = WL.module_keys(a.module)
        N = a.n_top
        draws = (2 if a.smoke else a.draws)
        # install-direction SVD per matrix (ΔW = after − base), generous q
        svd = {}
        for k in mkeys:
            dW = (after_sd[k].float() - base_sd[k].float()).to(device)
            U, S, V = torch.svd_lowrank(dW, q=min(N + 256, min(dW.shape)), niter=8)
            svd[k] = (U.cpu(), S.cpu(), V.cpu())
            del dW, U, S, V
            torch.cuda.empty_cache()

        def graft(use_random, gen=None):
            ov = {}
            for k in mkeys:
                U, S, V = svd[k]
                kk = min(N, S.shape[0])
                if not use_random:                            # top-N: base + rank-N truncation of ΔW
                    dWk = (U[:, :kk].to(device) * S[:kk].to(device)) @ V[:, :kk].to(device).T
                else:                                         # random-N IN-SPAN (w13_necessity form)
                    r = S.shape[0]
                    Ur = U[:, :r].to(device)
                    R = torch.randn(r, kk, generator=gen, device=device)
                    Q, _ = torch.linalg.qr(R)
                    Qd = Ur @ Q
                    dWfull = (after_sd[k].float() - base_sd[k].float()).to(device)
                    dWk = Qd @ (Qd.T @ dWfull)
                ov[k] = (base_sd[k].float().to(device) + dWk).to(torch.bfloat16)
            _set_decoder(lm_edit, after_sd, ov, device)

        graft(use_random=False)
        b, _ = _score(pg_before, proc, gemma_tok, device, n_per, text_secondary=False)
        hash_assert(b)
        top = _r_block(b["per_item"], item_hashes, before_pi, floor_pi)["confirmation"]
        gen = torch.Generator(device=device).manual_seed(SEED)
        rand_R = []
        for di in range(draws):
            graft(use_random=True, gen=gen)
            b, _ = _score(pg_before, proc, gemma_tok, device, n_per, text_secondary=False)
            hash_assert(b)
            rand_R.append(_r_block(b["per_item"], item_hashes, before_pi, floor_pi)["confirmation"]["R_mmlu"])
            print(f"[W14-A2-W] random-{N} draw {di+1}/{draws}: R_mmlu={rand_R[-1]:.3f}", flush=True)
        gap = top["R_mmlu"] - float(np.mean(rand_R))
        result = {"experiment": "w14_arm2_withinmlp", "status": "confirmatory", "stage": "withinmlp",
                  "module": a.module, "N_top": N, "draws": draws, "torch": torch.__version__,
                  "controls_s3": a.controls_s3, "top_N_R_mmlu": top,
                  "random_N_R_mmlu_mean": float(np.mean(rand_R)), "random_N_R_mmlu_draws": rand_R,
                  "necessity_gap": gap, "gap_positive": bool(gap > 0), "swap_gate": gate}
        (out / "arm2_within_mlp.json").write_text(json.dumps(result, indent=2, default=str))
        print("W14_ARM2_WITHINMLP_OK", json.dumps({"necessity_gap": round(gap, 4)}))
        return


if __name__ == "__main__":
    main()
