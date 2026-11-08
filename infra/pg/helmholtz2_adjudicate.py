"""HELMHOLTZ-2 adjudication — offline fold of arm metrics into the frozen §6 verdict.

Inputs: local copies of the frozen thresholds file plus, per arm (benign/know/skill; seeds
pooled), the arm job's metrics.json + per_item.json. Computes per instrument the two
separations Δ^know = S(D_know)−S(B) and Δ^skill = S(D_skill)−S(B) with item-bootstrap CIs
(behavior/cloze; per-item units) and band-vs-point comparisons (CKA, SAE stats; per-site),
then calls helmholtz2_core.discrimination_verdict. Pure json+numpy — runs anywhere.

  python -m infra.pg.helmholtz2_adjudicate --thresholds th.json \
      --benign b_s0/metrics.json,b_s1/metrics.json --know k_s0/...,k_s1/... \
      --skill s_s0/...,s_s1/... --out verdict.json
(per_item.json is expected next to each metrics.json)
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from infra.pg import helmholtz2_core as HC


def _load_arm(paths):
    out = []
    for p in paths.split(","):
        p = pathlib.Path(p)
        m = json.loads(p.read_text())
        pi = json.loads((p.parent / "per_item.json").read_text())
        out.append((m, pi))
    return out


def _beh_sep(d_runs, b_runs, key, n_boot, seed):
    """Per-item separation of the behavior DROP (before-after per item, pooled over seeds)."""
    def drops(runs):
        v = []
        for m, pi in runs:
            a = np.array(pi[f"{key}_before"], dtype=float) - np.array(pi[f"{key}_after"], dtype=float)
            v.append(a)
        return np.concatenate(v)
    return HC.bootstrap_mean_diff(drops(d_runs), drops(b_runs), n_boot=n_boot, seed=seed)


def _site_stat(runs, site, stat):
    return float(np.mean([m["sites"][site]["diff"][stat] if stat in m["sites"][site]["diff"]
                          else m["sites"][site][stat] for m, _ in runs]))


def adjudicate(thresholds, benign, know, skill, n_boot=10_000, seed=0):
    rep = {"inputs": {"n_benign": len(benign), "n_know": len(know), "n_skill": len(skill)}}
    # gates + margins (from the arm jobs' frozen computations)
    for name, runs in (("benign", benign), ("know", know), ("skill", skill)):
        rep[name] = {"gates": [m["margins"]["gate_pass"] for m, _ in runs],
                     "mcq_drop": [m["margins"]["mcq_margin_drop"] for m, _ in runs],
                     "cloze_drop": [m["margins"]["cloze_margin_drop"] for m, _ in runs]}
    assert all(rep["benign"]["gates"]), f"G2 failed: {rep['benign']}"
    assert all(rep["know"]["gates"]), f"G1k failed: {rep['know']}"
    assert all(rep["skill"]["gates"]), f"G1s failed: {rep['skill']}"
    # plant validity: EVERY scored run's plant controls must fire (else VOID)
    plants = [m["plant_validity"] for m, _ in benign + know + skill if "plant_validity" in m]
    assert plants, "no plant_validity blocks — arms were not scored with --score-sites"
    plants_valid = all(p["cka_fires"] and p["sae_fires"] for p in plants)
    rep["plants_valid"] = plants_valid

    # behavior separations (per-item bootstrap)
    beh_know = _beh_sep(know, benign, "mcq", n_boot, seed)
    beh_skill = _beh_sep(skill, benign, "mcq", n_boot, seed + 1)
    clz_know = _beh_sep(know, benign, "cloze", n_boot, seed + 2)
    clz_skill = _beh_sep(skill, benign, "cloze", n_boot, seed + 3)
    rep["separations"] = {"behavior_know": beh_know, "behavior_skill": beh_skill,
                          "cloze_know": clz_know, "cloze_skill": clz_skill}
    beh_fires_know = bool(beh_know["lo"] > 0)
    beh_fires_skill = bool(beh_skill["lo"] > 0)

    # SAE + CKA per site: fires ⟺ the D-vs-B stat gap clears 2x the frozen identity band
    bands = thresholds["per_site_identity_bands"]
    site_rows = {}
    sae_fires_know = sae_fires_skill = False
    cka_fires_know = cka_fires_skill = False
    for site, band in bands.items():
        row = {}
        for stat in ("lost", "dlog_rate"):
            sk = _site_stat(know, site, stat) - _site_stat(benign, site, stat)
            ss = _site_stat(skill, site, stat) - _site_stat(benign, site, stat)
            row[stat] = {"know_minus_benign": sk, "skill_minus_benign": ss,
                         "band": band[stat]}
            if sk > 2 * band[stat]:
                sae_fires_know = True
            if ss > 2 * band[stat]:
                sae_fires_skill = True
        ck = _site_stat(know, site, "cka") - _site_stat(benign, site, "cka")
        cs = _site_stat(skill, site, "cka") - _site_stat(benign, site, "cka")
        # CKA fires when the D arm moved the representation MORE than benign by > identity slack
        cka_slack = 1.0 - thresholds.get("cka_identity_floor", 0.999)
        row["cka"] = {"know_minus_benign": ck, "skill_minus_benign": cs, "slack": cka_slack}
        if ck < -2 * cka_slack:
            cka_fires_know = True
        if cs < -2 * cka_slack:
            cka_fires_skill = True
        site_rows[site] = row
    rep["sites"] = site_rows
    rep["fires"] = {"sae_know": sae_fires_know, "sae_skill": sae_fires_skill,
                    "cka_know": cka_fires_know, "cka_skill": cka_fires_skill,
                    "beh_know": beh_fires_know, "beh_skill": beh_fires_skill}
    rep["verdict"] = HC.discrimination_verdict(
        sae_fires_know, sae_fires_skill, cka_fires_know, cka_fires_skill,
        beh_fires_know, beh_fires_skill,
        cloze_drop_know=float(np.mean(rep["know"]["cloze_drop"])),
        cloze_drop_skill=float(np.mean(rep["skill"]["cloze_drop"])),
        plants_valid=plants_valid)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--benign", required=True)
    ap.add_argument("--know", required=True)
    ap.add_argument("--skill", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    th = json.loads(pathlib.Path(a.thresholds).read_text())
    rep = adjudicate(th, _load_arm(a.benign), _load_arm(a.know), _load_arm(a.skill))
    pathlib.Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print("VERDICT:", rep["verdict"])


def _dryrun():
    """Synthetic end-to-end: H_skill-blind world -> the verdict must land there."""
    rng = np.random.default_rng(0)
    n_items = 400
    def mk(arm, mcq_drop, cloze_drop, lost_gap, cka, plant=True):
        before = (rng.random(n_items) < 0.6).astype(int)
        after_m = before.copy()
        flip = rng.random(n_items) < mcq_drop * 0.35
        after_m[flip] = 0
        c_before = (rng.random(n_items) < 0.55).astype(int)
        c_after = c_before.copy()
        cflip = rng.random(n_items) < cloze_drop * 0.3
        c_after[cflip] = 0
        sites = {"17": {"cka": cka, "diff": {"lost": 0.001 + lost_gap, "dlog_rate": 0.01,
                                             "survival": 1 - lost_gap, "emergent": 0.001},
                        "fvu_before": 0.2, "fvu_after": 0.25, "band": None, "sae_fires": None}}
        m = {"margins": {"gate_pass": True, "mcq_margin_drop": mcq_drop,
                         "cloze_margin_drop": cloze_drop, "ce_final_over_initial": 0.5},
             "sites": sites,
             "plant_validity": {"cka_fires": plant, "sae_fires": plant}}
        pi = {"mcq_before": before.tolist(), "mcq_after": after_m.tolist(),
              "cloze_before": c_before.tolist(), "cloze_after": c_after.tolist()}
        return (m, pi)
    th = {"per_site_identity_bands": {"17": {"lost": 0.002, "emergent": 0.002,
                                             "dlog_rate": 0.02, "survival_floor": 0.995}},
          "tau": 1e-3, "cka_identity_floor": 0.999}
    benign = [mk("b", 0.02, 0.02, 0.0, 0.999)]
    know = [mk("k", 0.7, 0.5, 0.05, 0.85)]        # SAE sees it (lost gap >> band), CKA moved
    skill = [mk("s", 0.7, 0.05, 0.0, 0.998)]      # SAE + CKA silent, behavior fires
    rep = adjudicate(th, benign, know, skill, n_boot=2000)
    assert rep["verdict"].startswith("H_skill-blind"), rep["verdict"]
    know2 = [mk("k", 0.7, 0.5, 0.0, 0.85)]        # SAE silent on BOTH, CKA fires -> H_blind
    rep2 = adjudicate(th, benign, know2, skill, n_boot=2000)
    assert rep2["verdict"].startswith("H_blind"), rep2["verdict"]
    skill_v = [mk("s", 0.7, 0.05, 0.0, 0.998, plant=False)]
    rep3 = adjudicate(th, benign, know, skill_v, n_boot=2000)
    assert rep3["verdict"].startswith("VOID"), rep3["verdict"]
    print("HELMHOLTZ-2 adjudicate dry-run: ALL PASS (H_skill-blind / H_blind / VOID branches)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _dryrun()
    else:
        main()
