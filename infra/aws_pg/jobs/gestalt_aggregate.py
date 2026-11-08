"""GESTALT aggregate — recomputes every verdict from the banked artifacts (never from logs).
Implements the FROZEN Arm-S analysis rule (journal 2026-08-04, pre-unblind) VERBATIM, assembles
the BH F-shape family from whatever members exist (absent/VOID members listed, never silently
dropped), and emits the machine-readable aggregate the figures + result doc cite.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.gestalt_aggregate --type r7i.2xlarge --no-weights \
      --args "--mde site9=results/gestalt_arm_s/out/mde_s9 --field results/gestalt_field/out" \
      --s3-prefix results/gestalt_aggregate --budget-min 60

Prefixes point at the OUT dir of the producing run (s3://BUCKET/<prefix>/...).
No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

BONF_K = 7                 # the frozen K grid size (gestalt_splat.K_GRID)
BH_Q = 0.05


def _pull(prefix, dest):
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", f"s3://pg-lineage-769063704696/{prefix}", str(dest),
                    "--region", "us-east-1", "--only-show-errors",
                    "--exclude", "*", "--include", "metrics.json",
                    "--include", "eval_banks.json"], check=True)


def arm_s_member(mde_dir, site):
    """The frozen rule: p_K = max over seed pairings of paired p_le0(real−twin); member
    p = min_K p_K * 7, capped. Plus K_D (FR-5) and per-K descriptives."""
    import numpy as np

    from infra.pg.lacuna_core import boot_diff_ci

    met = json.loads((mde_dir / "metrics.json").read_text())
    banks = json.loads((mde_dir / "eval_banks.json").read_text())
    k_grid = met["k_grid"]
    seeds = met["seeds"]
    scores = {c["key"]: c for c in met["cells"] if c["kind"] == "eval"}
    per_k = {}
    for K in k_grid:
        pair_stats = []
        for s_r in seeds:
            kr = f"real|K{K}|s{s_r}"
            if kr not in banks:
                continue
            for s_t in seeds:
                kt = f"twin|K{K}|s{s_t}"
                if kt not in banks:
                    continue
                d = boot_diff_ci(banks[kr]["per_item"], banks[kt]["per_item"])
                pair_stats.append({"s_real": s_r, "s_twin": s_t, **d})
        if not pair_stats:
            per_k[K] = {"VOID": "no seed pairing measured"}
            continue
        # seed-MIN tail per the frozen wording (critic F23): the pairing with the smallest
        # |diff| is the one whose CI-overlap-with-0 the H_no-shape condition reads
        e_min = min(pair_stats, key=lambda p: abs(p["diff"]))
        per_k[K] = {"pairings": pair_stats,
                    "p_K": max(p["p_le0"] for p in pair_stats),
                    "diff_min": min(p["diff"] for p in pair_stats),
                    "diff_mean": float(np.mean([p["diff"] for p in pair_stats])),
                    "ci_overlap_seedmin": bool(e_min["lo95"] <= 0 <= e_min["hi95"]),
                    "ci_overlap_all_pairings": all(p["lo95"] <= 0 for p in pair_stats),
                    "real_scores": {s: scores.get(f"real|K{K}|s{s}", {}).get("score")
                                    for s in seeds},
                    "twin_scores": {s: scores.get(f"twin|K{K}|s{s}", {}).get("score")
                                    for s in seeds}}
    live = {K: v for K, v in per_k.items() if "p_K" in v}
    assert live, f"site{site}: no live K cells"
    member_p = min(1.0, min(v["p_K"] for v in live.values()) * BONF_K)
    k_best_p = min(live, key=lambda K: live[K]["p_K"])
    # K_D (FR-5): smallest K whose real CI overlaps the best real K's CI
    real_ci = {K: scores[f"real|K{K}|s{seeds[0]}"]["ci95"] for K in k_grid
               if f"real|K{K}|s{seeds[0]}" in scores}
    real_pt = {K: scores[f"real|K{K}|s{seeds[0]}"]["score"] for K in real_ci}
    k_star = max(real_pt, key=lambda K: real_pt[K])
    k_d = None
    for K in sorted(real_ci):
        if real_ci[K][1] >= real_ci[k_star][0]:
            k_d = K
            break
    every_k_overlaps = all(v.get("ci_overlap_seedmin", True) for v in live.values())
    # corroborating two-sided contrast (critic F2 fold): K1->K64 LL gain per point, per source
    fitc = {c["key"]: c for c in met["cells"] if c.get("kind") == "fit" and not c.get("void")}
    ll_gain = {}
    for src in ("real", "twin"):
        k1 = fitc.get(f"{src}|K1|s{seeds[0]}")
        k64 = fitc.get(f"{src}|K64|s{seeds[0]}")
        if k1 and k64:
            n_fit = met.get("n_fit") or 1
            ll_gain[src] = (k64["ll_final"] - k1["ll_final"]) / n_fit
    return {"site": site, "per_k": {str(k): v for k, v in per_k.items()},
            "member_p": member_p, "k_at_min_p": k_best_p,
            "member_p_floor": 7.0 / 10001,   # bootstrap resolution floor x7 Bonferroni (F19)
            "ll_gain_k1_to_k64_per_point": ll_gain,
            "k_d_fr5": k_d, "k_best_real": k_star,
            "every_k_ci_overlap": every_k_overlaps,
            "unswapped": scores.get("unswapped"), "p3_identity": scores.get("P3_identity"),
            "p3_gate": met.get("P3_gate"), "ppl_gate": met.get("ppl_gate"),
            "n_fidelity": met.get("fidelity", {}).get("n"),
            "gpu": met.get("gpu"), "twin_spec": met.get("twin_spec")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mde", default="", help="site9=prefix[,site14=prefix] (OUT dirs)")
    ap.add_argument("--field", default="", help="Arm F OUT prefix")
    ap.add_argument("--census", default="",
                    help="census-only OUT prefix (the main field run banked F1/F2 before its "
                         "census crash; the census cell lives in the rerun's metrics)")
    ap.add_argument("--diff", default="", help="Arm D adjudication OUT prefix")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.lacuna_core import bh_reject

    agg = {"object": "GESTALT aggregate (recomputed from banks)",
           "inputs": {"mde": a.mde, "field": a.field, "diff": a.diff}}
    family = {}

    if a.mde:
        for kv in a.mde.split(","):
            site_tag, prefix = kv.split("=", 1)
            d = pathlib.Path(f"/tmp/agg/{site_tag}")
            _pull(prefix, d)
            m = arm_s_member(d, site_tag)
            agg[f"arm_s_{site_tag}"] = m
            family[f"S_twin_separation_{site_tag}"] = m["member_p"]
            print(f"[AGG] {site_tag}: member_p={m['member_p']:.5f} K@minp={m['k_at_min_p']} "
                  f"K_D={m['k_d_fr5']} every-K-overlap={m['every_k_ci_overlap']}", flush=True)
    else:
        family["S_twin_separation_site9"] = None

    if a.field:
        d = pathlib.Path("/tmp/agg/field")
        _pull(a.field, d)
        fm = json.loads((d / "metrics.json").read_text())
        census = fm.get("census_pi05")
        if a.census:
            dc = pathlib.Path("/tmp/agg/census")
            _pull(a.census, dc)
            cm = json.loads((dc / "metrics.json").read_text())
            census = cm.get("census_pi05")
            assert (cm.get("probe_manifest") or {}).get("sha") == \
                (fm.get("probe_manifest") or {}).get("sha"), \
                "census probes differ from the field banks' probes"
        agg["arm_f"] = {"F1": fm.get("F1"), "F2_verdict": fm.get("F2_verdict"),
                        "F2_ladder": fm.get("F2_ladder"), "P5": fm.get("P5"),
                        "census_pi05": census,
                        "expert_params": fm.get("expert_params"),
                        "probe_manifest": fm.get("probe_manifest")}
        # FR-3(a) split-half measurement band at the endpoint, computed POST-HOC from the
        # banked v arrays (critic F4: the in-job check had silently substituted an unfailable
        # CI-width test — journaled D23; this is the frozen null actually computed)
        import numpy as np
        from infra.pg.gestalt_field import field_diff
        bdir = pathlib.Path("/tmp/agg/field_banks")
        bdir.mkdir(parents=True, exist_ok=True)
        for tag in ("U3_00004096", "U3c_00004096"):
            subprocess.run(["aws", "s3", "cp",
                            f"s3://pg-lineage-769063704696/{a.field}/bank_{tag}.npz",
                            str(bdir / f"bank_{tag}.npz"), "--region", "us-east-1",
                            "--only-show-errors"], check=True)
        bank_cells = {c["tag"]: c for c in fm.get("cells", []) if c.get("kind") == "bank"}
        adim_valid = int(bank_cells["pi0"]["adim_valid"])
        vA = np.load(bdir / "bank_U3c_00004096.npz")["v"]
        vB = np.load(bdir / "bank_U3_00004096.npz")["v"]
        vA = vA.reshape(-1, vA.shape[-2], vA.shape[-1]).astype(np.float32)
        vB = vB.reshape(-1, vB.shape[-2], vB.shape[-1]).astype(np.float32)
        d_even = field_diff(vA[0::2], vB[0::2], adim_valid, n_boot=400)
        d_odd = field_diff(vA[1::2], vB[1::2], adim_valid, n_boot=400)
        halfband = abs(d_even["d_rel"] - d_odd["d_rel"])
        agg["fr3a_splithalf_endpoint"] = {
            "d_rel_even_half": d_even["d_rel"], "d_rel_odd_half": d_odd["d_rel"],
            "band_width": halfband, "zero_width": bool(halfband <= 0.0),
            "note": "post-hoc repair of D23; frozen rule: zero width => F1 VOID"}

        f1 = fm.get("F1") or {}
        per_step = f1.get("per_step") or {}
        dev_steps = [s for s, e in per_step.items() if isinstance(e, dict) and e.get("develops")]
        # F1's family member: the matched-step contrast at the ENDPOINT step (4096 — the last
        # step both films define; frozen in the journal before any film bank existed); its p is
        # the banked bootstrap mass at/below the drift band (field_diff p_le_ref).
        keys_int = sorted(int(s) for s in per_step)
        if keys_int:
            last = keys_int[-1]
            e = per_step.get(last, per_step.get(str(last)))
            family["F1_field_develops_endpoint"] = e["diff"].get("p_le_ref")
            agg["arm_f"]["f1_develop_steps"] = dev_steps
    else:
        family["F1_field_develops_laststep"] = None

    if a.diff:
        import numpy as np
        d = pathlib.Path("/tmp/agg/diff")
        _pull(a.diff, d)
        dm = json.loads((d / "metrics.json").read_text())
        agg["arm_d"] = dm["cells"]

        rng = np.random.default_rng(20260804)
        n_boot = 4000

        def snorm_draws(site_rec, tgt_rec):
            """Component-bootstrap S_norm draws pooled equally over seed pairs (frozen rule)."""
            self_surv = np.array([lb == "survived" for lb in site_rec["self_labels_list"]],
                                 float)
            pools = []
            pairs = [p for p in tgt_rec["seed_pairs"] if "labels" in p]
            for p in pairs:
                sv = np.array([lb == "survived" for lb in p["labels"]], float)
                fl = np.array([lb == "survived" for lb in p["labels_floor"]], float)
                per = n_boot // max(len(pairs), 1)
                Kn = sv.size
                for _ in range(per):
                    s_d = sv[rng.integers(0, Kn, Kn)].mean()
                    f_d = fl[rng.integers(0, Kn, Kn)].mean()
                    c_d = self_surv[rng.integers(0, self_surv.size, self_surv.size)].mean()
                    den = c_d - f_d
                    pools.append((s_d - f_d) / den if den > 0 else float("nan"))
            return np.asarray(pools)

        # POST-CRITIC (F1/F7 fold): a site is VOID for BOTH D members when the MAJORITY of its
        # cross-model pairs are void_banded (band ordering inverted) — applied symmetrically,
        # with per-site pair/NaN counts disclosed. The earlier unfrozen 50%-finite guard (which
        # excluded site 9 from one member but not the other) is retired.
        p_pi0_sites, p_grad_sites, site_disclosure = [], [], []
        for site_rec in dm["cells"]:
            if "VOID" in site_rec:
                continue
            tgts = {t["target"]: t for t in site_rec.get("targets", [])}
            pairs_all = [p for t in site_rec.get("targets", []) for p in t["seed_pairs"]
                         if "void_banded" in p]
            n_void = sum(p["void_banded"] for p in pairs_all)
            disc = {"site": site_rec["site"], "n_pairs": len(pairs_all),
                    "n_void_banded": n_void}
            if pairs_all and n_void > len(pairs_all) / 2:
                disc["site_void_for_D_members"] = True
                site_disclosure.append(disc)
                continue
            disc["site_void_for_D_members"] = False
            if "pi0" in tgts:
                dr = snorm_draws(site_rec, tgts["pi0"])
                ok = dr[np.isfinite(dr)]
                disc["pi0_nan_draws"] = int(dr.size - ok.size)
                p_pi0_sites.append(float(((ok < 0.5).sum() + 1) / (ok.size + 1)))
                if "pi05" in tgts:
                    dr5 = snorm_draws(site_rec, tgts["pi05"])
                    ok5 = dr5[np.isfinite(dr5)]
                    disc["pi05_nan_draws"] = int(dr5.size - ok5.size)
                    m = min(ok.size, ok5.size)
                    # DEGENERACY GUARD (F1): a strict-inequality one-sided statistic on two
                    # (near-)identical constant draw vectors cannot fire in the registered
                    # direction — the member is VOID-degenerate, never "not rejected"
                    if m > 100 and (ok.std() > 1e-9 or ok5.std() > 1e-9
                                    or abs(float(ok.mean() - ok5.mean())) > 1e-9):
                        p_grad_sites.append(float(((ok5[:m] < ok[:m]).sum() + 1) / (m + 1)))
                    else:
                        disc["gradation_degenerate"] = True
            site_disclosure.append(disc)
        family["D_pi0_survival_vs_sealed"] = max(p_pi0_sites) if p_pi0_sites else None
        family["D_pi05_gradation"] = max(p_grad_sites) if p_grad_sites else None
        agg["arm_d_members"] = {"p_pi0_per_site": p_pi0_sites,
                                "p_grad_per_site": p_grad_sites,
                                "site_disclosure": site_disclosure}
    else:
        family["D_pi0_survival_vs_sealed"] = None
        family["D_pi05_gradation"] = None

    agg["bh_family_F_shape"] = bh_reject(family, q=BH_Q)
    agg["elapsed_s"] = time.time() - t0
    (out / "aggregate.json").write_text(json.dumps(agg, indent=1, default=str))
    print("GESTALT_AGGREGATE_OK", json.dumps(agg["bh_family_F_shape"], default=str), flush=True)


if __name__ == "__main__":
    main()
