"""FAIRSCRATCH §6.1 recipe scorer — the blind, pre-stated fair-scratch recipe search.

Control-plane ONLY (no GPU, no model): the candidate recipes run as ordinary
`w17_cell` jobs (`--decoder random` for admissibility; `--decoder before` for the
BEFORE-side safety check), each banking a `cell_*.json` with `train.mean_loss_last_epoch`
and `eval.acc_pct`. THIS script reads those rows and applies the FROZEN §6.1 procedure
mechanically:

  * admissibility (label-fitting gate, train-CE only): the random arm's last-epoch
    letter-CE drops ≥ 0.30 below ln4 (the uniform-over-4 chance floor) AND lands within
    0.15 of the pg arm's last-epoch CE at the same n — i.e. it ENTERS the label-fitting
    regime W17's random arm never reached (the M2 confound this study closes).
  * rejection (instrument safety): a recipe is rejected if, applied to the intact
    BEFORE decoder, it degrades the BEFORE-side bucket MMLU below the W17 frozen
    no-degradation band (forbids recipes that "win" by perturbing the readout).
  * selection: among admissible-and-safe recipes, the LOWEST last-epoch CE (best label
    fit); ties → fewest steps. If NONE is admissible → INDETERMINATE-untrainable
    (bounds the construction cost from below; do NOT relax the gate post-hoc).

No favourite recipe; the grid is pre-stated in the prereg (§6.1) and decided here from
train-CE on the discovery/bucket half only — the readout is never consulted in selection.

    python -m infra.aws_pg.jobs.fairscratch_recipe --out <dir> \
        --cells-dir <dir with **/cell_*.json> --controls <w17_controls.json>
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

LN4 = math.log(4.0)            # 1.3862943611… — uniform-over-4 chance CE on the answer letter
LN4_MARGIN = 0.30              # §6.1 cond1: last-epoch CE ≤ ln4 − 0.30
PG_CE_TOL = 0.15              # §6.1 cond2: last-epoch CE ≤ pg-arm CE + 0.15


# ===========================================================================
# PURE §6.1 gate + selection logic — the §9.0 weight-free dry-run exercises this.
# ===========================================================================
def admissible(ce_rnd: float, ce_pg: float) -> tuple:
    """§6.1 convergence-acceptance: did the random arm enter the label-fitting regime?
    Returns (bool, detail). cond1 = CE ≥0.30 below ln4; cond2 = within 0.15 of pg CE."""
    cond1 = ce_rnd <= LN4 - LN4_MARGIN
    cond2 = ce_rnd <= ce_pg + PG_CE_TOL
    return bool(cond1 and cond2), {"cond1_below_ln4": bool(cond1), "cond2_near_pg_ce": bool(cond2)}


def is_safe(before_acc, before_band: float):
    """§6.1 rejection: the recipe must keep the intact BEFORE-side bucket MMLU at or
    above the frozen W17 no-degradation band. `before_acc` is a fraction (0..1) or None
    (not yet measured — only admissible candidates need the safety run)."""
    if before_acc is None:
        return None
    return bool(before_acc >= before_band)


def select_recipe(candidates: list, ce_pg: float, before_band: float) -> dict:
    """Apply admissibility + rejection + selection to the candidate rows.

    candidates: list of {recipe_id, lr, epochs, alpha, steps, ce_rnd, before_acc|None}.
    Returns {chosen, ranked, status, ce_pg, before_band}. status ∈
      {"admissible" (chosen set), "INDETERMINATE-untrainable" (none admissible),
       "all-admissible-unsafe" (admissible exist but all degrade the instrument)}.
    """
    ranked = []
    for c in candidates:
        adm, detail = admissible(c["ce_rnd"], ce_pg)
        ranked.append({**c, "admissible": adm, **detail,
                       "safe": is_safe(c.get("before_acc"), before_band)})
    ranked.sort(key=lambda c: (c["ce_rnd"], c["steps"]))
    eligible = [c for c in ranked if c["admissible"] and c["safe"] is True]
    if eligible:
        chosen = min(eligible, key=lambda c: (c["ce_rnd"], c["steps"]))
        return {"chosen": chosen, "ranked": ranked, "status": "admissible",
                "ce_pg": ce_pg, "before_band": before_band}
    status = "all-admissible-unsafe" if any(c["admissible"] for c in ranked) \
        else "INDETERMINATE-untrainable"
    return {"chosen": None, "ranked": ranked, "status": status,
            "ce_pg": ce_pg, "before_band": before_band}


def recipe_key(row: dict) -> tuple:
    """Identify a candidate by its (lr, epochs, alpha, max_steps) — the cell_id does NOT
    encode the recipe, so cells are paired random↔before by this key."""
    r = row["train"]["recipe"]
    return (round(float(r["lr"]), 12), int(r["epochs"]),
            round(float(row["adapter"]["alpha"]), 6), r.get("max_steps"))


# ===========================================================================
# I/O main — read banked cells, build candidate table, freeze the recipe.
# ===========================================================================
def _load_cells(cells_dir: pathlib.Path) -> list:
    rows = [json.loads(p.read_text()) for p in sorted(cells_dir.glob("**/cell_*.json"))]
    assert rows, f"no cell_*.json under {cells_dir}"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells-dir", required=True,
                    help="dir holding **/cell_*.json for the random recipe ladder, the "
                         "before-side safety cells, and ≥1 pg@16k cell (the CE reference)")
    ap.add_argument("--controls", required=True, help="w17_controls.json (no_degradation_floor_bucket)")
    ap.add_argument("--n", type=int, default=16000, help="decision budget the recipe is frozen at")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    C = json.loads(pathlib.Path(args.controls).read_text())
    assert C["CONTROLS_OK"] is True, "controls not OK"
    before_band = float(C["no_degradation_floor_bucket"])   # frozen W17 BEFORE-side band (fraction)

    rows = _load_cells(pathlib.Path(args.cells_dir))

    def cap_cells(decoder):
        return [r for r in rows if r["decoder"] == decoder and r["arm"] == "cap"
                and r["n"] == args.n and r["rank"] == 512]

    pg = cap_cells("pg")
    assert pg, f"no pg cap@n{args.n} cell for the CE reference"
    pg_ces = [r["train"]["mean_loss_last_epoch"] for r in pg]
    ce_pg = float(sum(pg_ces) / len(pg_ces))                # seed-mean pg last-epoch CE

    rand = {recipe_key(r): r for r in cap_cells("random")}
    bef = {recipe_key(r): r for r in cap_cells("before")}
    assert rand, "no random-decoder recipe cells found"

    candidates = []
    for k, rc in rand.items():
        ce_rnd = rc["train"]["mean_loss_last_epoch"]
        assert ce_rnd is not None, f"random cell {rc['cell_id']} has no last-epoch CE"
        bc = bef.get(k)
        lr, epochs, alpha, msteps = k
        candidates.append({
            "recipe_id": f"lr{lr:g}_ep{epochs}_a{alpha:g}" + (f"_ms{msteps}" if msteps else ""),
            "lr": lr, "epochs": epochs, "alpha": alpha, "max_steps": msteps,
            "steps": int(rc["train"]["steps"]),
            "ce_rnd": float(ce_rnd),
            "before_acc": (float(bc["eval"]["acc_pct"]) / 100.0) if bc else None,
            "before_acc_pct": (float(bc["eval"]["acc_pct"]) if bc else None),
            "rnd_cell": rc["cell_id"], "before_cell": (bc["cell_id"] if bc else None),
        })

    result = select_recipe(candidates, ce_pg, before_band)
    result["ln4"] = LN4
    result["gates"] = {"ln4_margin": LN4_MARGIN, "pg_ce_tol": PG_CE_TOL,
                       "admissible_ce_max": min(LN4 - LN4_MARGIN, ce_pg + PG_CE_TOL),
                       "before_band": before_band}
    result["controls"] = args.controls
    result["FROZEN"] = result["chosen"] is not None
    (out / "fairscratch_recipe.json").write_text(json.dumps(result, indent=1, default=str))

    print(f"[FS-RECIPE] ce_pg={ce_pg:.4f} before_band={before_band:.4f} "
          f"admissible_ce_max={result['gates']['admissible_ce_max']:.4f}")
    for c in result["ranked"]:
        print(f"[FS-RECIPE] {c['recipe_id']}: CE={c['ce_rnd']:.4f} "
              f"adm={c['admissible']} safe={c['safe']} "
              f"before={c['before_acc_pct']}")
    print(f"[FS-RECIPE] STATUS={result['status']} CHOSEN={result['chosen']}")
    if result["chosen"] is None:
        raise RuntimeError(f"FAIRSCRATCH recipe search: {result['status']} — "
                           f"no fair scratch recipe frozen (do NOT relax the gate post-hoc)")


if __name__ == "__main__":
    main()
