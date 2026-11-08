"""PRAGNANZ aggregate — the ONE formal pass (spec §7 SAP + the journaled pre-aggregate
micro-rule amendments, applied verbatim; CPU-only, reads banks from S3, no weights).

F-read: 4 p-valued members {ladder PG1, ladder ft-max (both reproduction-gated vs the
journaled unblind), BSF corroboration, restore-vs-random} — BH q=0.05, m=4.
F-deadlock: 2 p-valued members {mean-cancel sign test, same-object null test} — BH m=2 —
plus {M4 savings, M5 ordering} categorical claim rules reported beside (no p exists at
2 seeds). Known resolution limit, disclosed not patched: the same-object member's 24-draw
banked null floors p at 1/25 = 0.04 > the m=2 BH k=1 threshold 0.025 — the confirmatory
member is instrument-resolution-limited regardless of effect size; census reported beside.

    python -m infra.aws_pg.jobs.pragnanz_aggregate --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "s3://pg-lineage-769063704696/results"
N_BOOT = 10000
BOOT_SEED = 20260805
BH_Q = 0.05
LADDER_REPRO = {"pg1": 0.4218, "ft-aokvqa-da": 0.5004}   # journaled unblind, tolerance 0.02
M4_ARMS = ["baseline", "core", "random", "dome"]
M5_ARMS = ["m5_incore", "m5_perp", "m5_rand"]
SEEDS = [1337, 1338]


def _pull(rel, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "cp", f"{BUCKET}/{rel}", str(dest),
                    "--region", "us-east-1"], check=True, capture_output=True)
    return json.loads(dest.read_text())


def boot_p_greater(a, b, rng):
    """One-sided item-paired bootstrap p for mean(a-b) > 0; p=(1+#{boot<=0})/(B+1)."""
    import numpy as np
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    assert n > 10, f"too few items ({n})"
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = d[idx].mean(axis=1)
    return {"d": float(d.mean()), "lo95": float(np.percentile(boots, 2.5)),
            "hi95": float(np.percentile(boots, 97.5)),
            "p": float((1 + (boots <= 0).sum()) / (N_BOOT + 1)), "n": int(n)}


def bh(members, q=BH_Q):
    """BH over p-valued members: returns per-member reject flags (name-keyed)."""
    ps = sorted(((m["p"], m["name"]) for m in members))
    m = len(ps)
    k_star = 0
    for i, (p, _) in enumerate(ps, start=1):
        if p <= q * i / m:
            k_star = i
    rejected = {name for _, name in ps[:k_star]}
    return {mm["name"]: (mm["name"] in rejected) for mm in members}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    rep = {"n_boot": N_BOOT, "boot_seed": BOOT_SEED, "bh_q": BH_Q, "families": {}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    # ================= F-read ===============================================================
    lb = _pull("pragnanz_ladder/out/ladder_banks.json", out / "in/ladder_banks.json")
    fread = []
    for unit in ("pg1", "ft-aokvqa-da"):
        ps = []
        pairs = []
        for s1 in (0, 1):
            for s2 in (0, 1):
                r1 = lb[f"{unit}|K32|r1|s{s1}"]["per_item"]
                r2 = lb[f"{unit}|K32|r2|s{s2}"]["per_item"]
                st = boot_p_greater(r1, r2, np.random.default_rng(BOOT_SEED))
                pairs.append({"s_r1": s1, "s_r2": s2, **st})
                ps.append(st["p"])
        member_p = max(ps)                                     # worst-of-4 (frozen)
        repro = abs(member_p - LADDER_REPRO[unit]) <= 0.02
        assert repro, f"ladder reproduction FAIL {unit}: {member_p} vs {LADDER_REPRO[unit]}"
        fread.append({"name": f"ladder_{unit}_K32", "p": member_p, "pairs": pairs,
                      "reproduces_journal": repro})
    # K16 sensitivity rung (critic #14: previously computed off-aggregate) — NOT a family
    # member; reported beside per the frozen micro-rule ("K=16 is the sensitivity rung").
    sens = {}
    for unit in ("pg1", "ft-aokvqa-da"):
        ps, pairs = [], []
        for s1 in (0, 1):
            for s2 in (0, 1):
                st = boot_p_greater(lb[f"{unit}|K16|r1|s{s1}"]["per_item"],
                                    lb[f"{unit}|K16|r2|s{s2}"]["per_item"],
                                    np.random.default_rng(BOOT_SEED))
                pairs.append({"s_r1": s1, "s_r2": s2, **st})
                ps.append(st["p"])
        sens[unit] = {"member_p_worst4": max(ps), "pairs": pairs}
    rep["sensitivity_k16"] = sens

    r2b = _pull("pragnanz_r2/out/r2_banks.json", out / "in/r2_banks.json")
    ps, det = [], []
    for s in (0, 1):
        st = boot_p_greater(r2b["pg1|rt"]["per_item"], r2b[f"pg1|shuf|s{s}"]["per_item"],
                            np.random.default_rng(BOOT_SEED))
        det.append({"shuf_seed": s, **st})
        ps.append(st["p"])
    fread.append({"name": "bsf_corroboration", "p": max(ps), "seeds": det})
    ps, det = [], []
    for s in (0, 1):
        st = boot_p_greater(r2b["pi0|transport"]["per_item"],
                            r2b[f"pi0|randtrans|s{s}"]["per_item"],
                            np.random.default_rng(BOOT_SEED))
        det.append({"rand_seed": s, **st})
        ps.append(st["p"])
    fread.append({"name": "restore_vs_random", "p": max(ps), "seeds": det,
                  "registered": "NULL"})
    rep["families"]["F-read"] = {"members": fread, "m": len(fread), "bh": bh(fread)}
    _w()

    # ================= F-deadlock (p-valued members) ========================================
    fdead = []
    from scipy.stats import binomtest
    ps, det = [], []
    for dst in ("textvqa", "cococap"):
        m1 = _pull(f"pragnanz_m1_{dst}/out/metrics.json", out / f"in/m1_{dst}.json")["m1"]
        wins = sum(1 for v in m1.values()
                   if v["core"]["ratio"] < np.mean([v[f"rand{j}"]["ratio"] for j in range(3)]))
        p = binomtest(wins, len(m1), 0.5, alternative="greater").pvalue
        det.append({"dst": dst, "wins": wins, "n": len(m1), "p": float(p)})
        ps.append(float(p))
    fdead.append({"name": "mean_cancel_sign", "p": max(ps), "dsts": det})
    ps, det = [], []
    for dst in ("textvqa", "cococap"):
        m2 = json.loads((out / f"in/m1_{dst}.json").read_text())["m2"]
        key = next(k for k in m2 if "__9__" in k or "layers__9" in k)
        v = m2[key]
        cf, lam = np.asarray(v["dome_core_fracs"]), np.asarray(v["lam"])
        obs = float((cf * lam).sum() / lam.sum())
        null = v["null"]
        draws = null.get("draws")
        if draws is not None:
            p = (1 + sum(1 for x in draws if x >= obs)) / (len(draws) + 1)
        else:
            # band-only bank: obs above the whole band ⇒ p at the 1/(n+1) resolution floor;
            # obs inside/below ⇒ VOID (band cannot resolve it) — disclosed rather than guessed.
            p = (1.0 / (null["n"] + 1)) if obs > null["hi"] else float("nan")
        det.append({"dst": dst, "obs_weighted": obs, "null_band": [null["lo"], null["hi"]],
                    "null_branch": "draws" if draws is not None else "band_only",
                    "p": float(p)})
        ps.append(float(p))
    member = {"name": "same_object_L9", "dsts": det}
    if any(np.isnan(ps)):
        member.update({"p": float("nan"), "VOID": "band-only null could not resolve obs"})
        fdead.append(member)
    else:
        member.update({"p": max(ps)})
        fdead.append(member)
    pvalued = [m for m in fdead if not np.isnan(m["p"])]
    rep["families"]["F-deadlock"] = {"members": fdead, "m_pvalued": len(pvalued),
                                     "bh": bh(pvalued) if pvalued else {}}
    _w()

    # ================= M4/M5 categorical claim rules ========================================
    sav = [_pull(f"sav_cellsE_PG1/out/PG1_n16000_s{s}_confirmation/"
                 f"cell_PG1_cap_r512_n16000_s{s}_confirmation.json", out / f"in/sav_{s}.json")
           for s in (1337, 1338, 1339)]
    banked_S = [c["eval"]["S_recover"] for c in sav]
    banked_ppl = [c["eval"]["wikitext_ppl"] for c in sav]
    s_lo, s_hi = 0.9 * min(banked_S), 0.9 * float(np.mean(banked_S))
    rep["criterion"] = {"S_star_lo": s_lo, "S_star_hi": s_hi, "banked_S": banked_S,
                        "banked_ppl_max": max(banked_ppl)}

    def crossing(traj, s_star):
        for row in traj:
            if row["S"] >= s_star:
                return row["step"], False
        return None, True                                     # censored

    m4 = {}
    for arm in M4_ARMS + M5_ARMS:
        for seed in SEEDS:
            if arm == "baseline":                        # standalone P4-anchor runs
                rel = f"pragnanz_m4_baseline_s{seed}/out/metrics.json"
            elif arm in M4_ARMS:                         # server-side pipeline steps
                rel = f"pragnanz_m4arms_s{seed}/out/{arm}/metrics.json"
            else:
                rel = f"pragnanz_m5arms_s{seed}/out/{arm.removeprefix('m5_')}/metrics.json"
            r = _pull(rel, out / f"in/{arm}_s{seed}.json")
            c_lo, cen_lo = crossing(r["trajectory"], s_lo)
            c_hi, cen_hi = crossing(r["trajectory"], s_hi)
            m4[f"{arm}_s{seed}"] = {
                "steps_lo": c_lo, "censored_lo": cen_lo,
                "steps_hi": c_hi, "censored_hi": cen_hi,
                "S_end": r["endpoint"]["S"], "ppl": r["endpoint"]["wikitext_ppl"],
                "per_item": r["endpoint"]["per_item"], "p4": r.get("p4"),
                "trajectory": r["trajectory"]}
    rep["m4_cells"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_item"}
                       for k, v in m4.items()}
    _w()

    # frozen VOID branches
    void = []
    if any(m4[f"baseline_s{s}"]["censored_lo"] for s in SEEDS):
        void.append("M4 VOID: baseline failed to cross S*_lo (harness drift)")
    if all(not m4[f"{arm}_s{s}"]["censored_lo"] and m4[f"{arm}_s{s}"]["steps_lo"] <= 250
           for arm in M4_ARMS for s in SEEDS):
        void.append("M4 resolution-VOID: all arms cross at first eval")
    rep["m4_void"] = void

    def not_worse(arm, seed):
        st = boot_p_greater(m4[f"{arm}_s{seed}"]["per_item"],
                            m4[f"baseline_s{seed}"]["per_item"],
                            np.random.default_rng(BOOT_SEED))
        return (not st["hi95"] < 0), st

    m4_verdict = {"void": void}
    if not void:
        rows = {}
        for arm in ("core", "random", "dome"):
            per_seed = []
            for s in SEEDS:
                sav_steps = None
                if not m4[f"{arm}_s{s}"]["censored_lo"]:
                    sav_steps = m4[f"baseline_s{s}"]["steps_lo"] - m4[f"{arm}_s{s}"]["steps_lo"]
                nw, st = not_worse(arm, s)
                ppl_ok = m4[f"{arm}_s{s}"]["ppl"] <= max(banked_ppl)
                per_seed.append({"seed": s, "savings_lo": sav_steps, "endpoint_not_worse": nw,
                                 "endpoint_diff": st, "ppl_ok": ppl_ok,
                                 "censored": m4[f"{arm}_s{s}"]["censored_lo"]})
            rows[arm] = per_seed
        beats_random = all(
            (rows["core"][i]["savings_lo"] is not None
             and rows["random"][i]["savings_lo"] is not None
             and rows["core"][i]["savings_lo"] > rows["random"][i]["savings_lo"])
            for i in range(len(SEEDS)))
        clauses_ok = all(r["endpoint_not_worse"] and r["ppl_ok"] for r in rows["core"])
        m4_verdict.update({"arms": rows, "core_beats_random_both_seeds": beats_random,
                           "core_clauses_ok": clauses_ok,
                           "CLAIM": bool(beats_random and clauses_ok)})
    rep["m4_verdict"] = m4_verdict

    m5_order = {}
    for s in SEEDS:
        st = {arm: m4[f"{arm}_s{s}"]["steps_lo"] for arm in M5_ARMS}
        cen = {arm: m4[f"{arm}_s{s}"]["censored_lo"] for arm in M5_ARMS}
        big = 10 ** 9
        v = {arm: (big if cen[arm] else st[arm]) for arm in M5_ARMS}
        m5_order[s] = {"steps": st, "censored": cen,
                       "ordering_holds": bool(v["m5_perp"] <= v["m5_rand"] < v["m5_incore"])}
    rep["m5_verdict"] = {"per_seed": m5_order,
                         "CLAIM": all(m5_order[s]["ordering_holds"] for s in SEEDS),
                         "registered": "perp <= rand < incore on BOTH seeds"}

    rep["elapsed_note"] = "aggregate complete"
    _w()
    print("PRAGNANZ_AGGREGATE_OK", json.dumps(
        {f["name"]: round(f["p"], 4) for fam in rep["families"].values()
         for f in fam["members"] if f.get("p") == f.get("p")}), flush=True)


if __name__ == "__main__":
    main()
