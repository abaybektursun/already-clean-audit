"""FAIRSCRATCH §9.0 weight-free dry-run gate — proves the LOGIC, no model/GPU.

Unit-tests the pure cores on tiny planted inputs (seconds on a laptop), exactly the
four checks the prereg §9.0 names:
  (a) s_text + paired-bootstrap ΔS on synthetic per-item arrays → the known analytic ΔS;
  (b) §6.1 admissibility + rejection + selection on a toy CE table, incl. the
      "no admissible recipe" → INDETERMINATE branch and the all-admissible-unsafe branch;
  (c) ladder_subset nesting + dedup_against_eval audit counts on toy rows;
  (d) §6.4 verdict map on hand-built ΔS/CE inputs — every row hit.

Distinct from the §9.1 runtime preflight (which proves the WIRING on real weights).
Fail-loud: any mismatch RAISES.  Run:  python -m infra.aws_pg.jobs.fairscratch_dryrun
"""
from __future__ import annotations

from infra.aws_pg.jobs.fairscratch_recipe import (
    LN4, admissible, is_safe, select_recipe, recipe_key,
)
from infra.aws_pg.jobs.fairscratch_unblind import fairscratch_verdict
from infra.pg.relearn import dedup_against_eval, ladder_subset, qa_item_hash
from infra.pg.swap_back import boot_s_text, s_text
from infra.pg.w17_verdict import paired_arm_diff_s


def t_s_and_paired():
    """(a) s_text + paired ΔS recover the analytic values; bootstrap CI brackets the point."""
    before = [1] * 800 + [0] * 200      # mean .8
    floor = [1] * 200 + [0] * 800       # mean .2  → scale (BEFORE−FLOOR)=.6
    a = [1] * 600 + [0] * 400           # pg  mean .6 → S_pg=(.6−.2)/.6=.6667
    b = [1] * 300 + [0] * 700           # scr mean .3 → S_scr=(.3−.2)/.6=.1667
    s_pg = s_text(a, before, floor)
    assert abs(s_pg - (0.4 / 0.6)) < 1e-9, f"s_text wrong: {s_pg}"
    bm = boot_s_text(a, before, floor, n_boot=500)
    assert bm[1] <= bm[0] <= bm[2] and abs(bm[0] - s_pg) < 0.03, f"boot_s_text off: {bm} vs {s_pg}"
    d = paired_arm_diff_s([a], [b], before, floor, n_boot=500)
    assert abs(d["point"] - 0.5) < 1e-9, f"paired ΔS wrong: {d['point']} (want 0.5)"
    assert d["ci95"][0] <= d["point"] <= d["ci95"][1], f"CI doesn't bracket: {d['ci95']}"
    print(f"  [ok] s_text={s_pg:.4f}  ΔS_paired={d['point']:.4f} CI[{d['ci95'][0]:.4f},{d['ci95'][1]:.4f}]")


def t_recipe_gates():
    """(b) §6.1 admissibility / rejection / selection + INDETERMINATE + all-unsafe branches."""
    ce_pg, band = 0.93, 0.50           # admissible bar = min(ln4−0.30, 0.93+0.15) = 1.08
    assert abs(LN4 - 1.3862943611) < 1e-6
    assert admissible(1.70, ce_pg)[0] is False     # W17 random arm — never enters label-fitting
    assert admissible(1.05, ce_pg)[0] is True       # below ln4−0.30 AND within 0.15 of pg
    assert admissible(1.09, ce_pg)[0] is False      # above ln4−0.30 (1.086) → cond1 fails
    assert is_safe(0.55, band) is True and is_safe(0.40, band) is False

    cands = [
        {"recipe_id": "C1_w17like", "lr": 5e-6, "epochs": 1, "alpha": 1024, "steps": 2000, "ce_rnd": 1.70, "before_acc": 0.55},
        {"recipe_id": "C2", "lr": 2e-5, "epochs": 3, "alpha": 1024, "steps": 6000, "ce_rnd": 1.05, "before_acc": 0.55},
        {"recipe_id": "C3_unsafe", "lr": 1e-4, "epochs": 6, "alpha": 1024, "steps": 12000, "ce_rnd": 0.95, "before_acc": 0.40},
        {"recipe_id": "C4_winner", "lr": 5e-5, "epochs": 6, "alpha": 1024, "steps": 12000, "ce_rnd": 1.00, "before_acc": 0.52},
    ]
    res = select_recipe(cands, ce_pg, band)
    assert res["status"] == "admissible", res["status"]
    assert res["chosen"]["recipe_id"] == "C4_winner", res["chosen"]   # lowest CE among admissible+safe
    # C3 admissible but unsafe → rejected; C1 inadmissible
    rk = {c["recipe_id"]: c for c in res["ranked"]}
    assert rk["C3_unsafe"]["admissible"] is True and rk["C3_unsafe"]["safe"] is False
    assert rk["C1_w17like"]["admissible"] is False

    # tie on CE → fewest steps wins
    tie = [
        {"recipe_id": "T_more", "lr": 5e-5, "epochs": 6, "alpha": 1024, "steps": 12000, "ce_rnd": 1.00, "before_acc": 0.52},
        {"recipe_id": "T_few", "lr": 5e-5, "epochs": 3, "alpha": 1024, "steps": 6000, "ce_rnd": 1.00, "before_acc": 0.52},
    ]
    assert select_recipe(tie, ce_pg, band)["chosen"]["recipe_id"] == "T_few"

    # no admissible → INDETERMINATE
    none_adm = [{"recipe_id": "x", "lr": 5e-6, "epochs": 1, "alpha": 1024, "steps": 2000, "ce_rnd": 1.5, "before_acc": 0.55}]
    assert select_recipe(none_adm, ce_pg, band)["status"] == "INDETERMINATE-untrainable"

    # admissible but all unsafe → all-admissible-unsafe
    all_unsafe = [{"recipe_id": "y", "lr": 1e-4, "epochs": 6, "alpha": 1024, "steps": 12000, "ce_rnd": 0.9, "before_acc": 0.30}]
    assert select_recipe(all_unsafe, ce_pg, band)["status"] == "all-admissible-unsafe"
    print("  [ok] §6.1 gates: admissible/unsafe/INDETERMINATE/all-unsafe + tie-break all resolve")


def t_ladder_dedup():
    """(c) ladder_subset is nested (subset10 ⊂ subset20); dedup audit counts are exact."""
    rows = [{"question": f"Distinct question number {i} concerning topic alpha-{i}",
             "choices": ["w", "x", "y", "z"], "answer": i % 4} for i in range(100)]
    s10, s20 = ladder_subset(rows, 10), ladder_subset(rows, 20)
    h10 = {qa_item_hash(r["question"]) for r in s10}
    h20 = {qa_item_hash(r["question"]) for r in s20}
    assert len(s10) == 10 and len(s20) == 20 and h10 <= h20, "ladder not nested"
    eval_hashes = [qa_item_hash(rows[i]["question"]) for i in range(5)]
    ded = dedup_against_eval(rows, eval_hashes)
    assert ded["n_in"] == 100 and ded["n_dropped"] == 5, ded
    assert all(qa_item_hash(r["question"]) not in set(eval_hashes) for r in ded["kept"])
    print(f"  [ok] ladder nested (10⊂20); dedup dropped {ded['n_dropped']}/{ded['n_in']} colliding rows")


def t_recipe_key():
    """recipe_key pairs random↔before cells by (lr, epochs, alpha, max_steps)."""
    def row(lr, ep, alpha, ms):
        return {"train": {"recipe": {"lr": lr, "epochs": ep, "max_steps": ms}}, "adapter": {"alpha": alpha}}
    assert recipe_key(row(5e-5, 6, 1024, None)) == recipe_key(row(5e-5, 6, 1024.0, None))
    assert recipe_key(row(5e-5, 6, 1024, None)) != recipe_key(row(2e-5, 6, 1024, None))
    print("  [ok] recipe_key pairs by (lr,epochs,alpha,max_steps)")


def t_verdicts():
    """(d) §6.4 verdict map — every row hit."""
    deq = 0.12
    assert fairscratch_verdict(0.5, True, 0.5, deq, True, False)["verdict"] == "HALT"                # preflight fail
    assert fairscratch_verdict(0.5, True, 0.5, deq, False, True)["verdict"] == "INDETERMINATE-untrainable"
    assert fairscratch_verdict(0.5, True, 0.5, deq, True, True, gap_stable=False)["verdict"] == "non-result"
    assert fairscratch_verdict(0.50, True, 0.50, deq, True, True)["verdict"] == "H_construction"     # both gaps survive
    assert fairscratch_verdict(0.50, True, 0.05, deq, True, True)["verdict"] == "H_access"           # matched-CE closes it
    assert fairscratch_verdict(0.05, False, None, deq, True, True)["verdict"] == "H_recipe-artifact" # gap collapses
    assert fairscratch_verdict(0.30, False, None, deq, True, True)["verdict"] == "indeterminate"     # >δeq but ns
    print("  [ok] §6.4 verdict map: HALT/INDETERMINATE/non-result/H_construction/H_access/H_recipe-artifact/indeterminate")


def main():
    print("[FS-DRYRUN] weight-free logic gate (prereg §9.0)")
    t_s_and_paired()
    t_recipe_gates()
    t_ladder_dedup()
    t_recipe_key()
    t_verdicts()
    print("[FS-DRYRUN] ALL PASS")


if __name__ == "__main__":
    main()
