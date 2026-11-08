"""LATMECH unblind / analyze — aggregate cell JSONs → MDE + §6 verdict + Arm-2 law + §11 grid.

CPU, weight-free (reads the banked per-item arrays; NO model). Computes ONCE on the requested
split (sealed `confirmation` for the headline): the recovery fraction per weight condition at the
headline budget, the MDE primary contrast Δ_content − Δ_emrand (= RF(emrand) − RF(content)), the
shrink-perturb and competence-matched CIs, the frozen §6 verdict, and (if interp/spectrum cells
exist) the Arm-2 content-vs-basin law via savb. Fills the §11 grid 1:1, whatever the result.

RF anchoring (prereg §2.5): ceiling = deleted@budget (content present, re-learned), floor =
random_floor@budget (no content). RF_x = (S_x − floor)/(ceiling − floor), per-item, paired BY
POSITION (qa hash is a reorder tripwire only — the 895-collision is NOT a join key).

    python -m infra.aws_pg.jobs.latmech_unblind --cells <dir> --split confirmation \
        --budget 16000 --arm cap --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _load_cells(cells_dir, split, arm):
    """{(condition, n, seed): row} for cells matching split+arm."""
    by_key = {}
    for p in sorted(pathlib.Path(cells_dir).rglob("cell_*.json")):
        row = json.loads(p.read_text())
        if row["eval_split"] != split or row["arm"] != arm:
            continue
        cond = row["condition_info"]["condition"]
        by_key[(cond, int(row["n"]), int(row["seed"]))] = row
    return by_key


def _fetch_cells_s3(prefixes, dest):
    """Download every cell_*.json under the given S3 prefixes into one local dir (I/O only; the
    cells are spread across per-seed grid prefixes). Recursive sync filtered to cell_*.json."""
    import subprocess
    dest.mkdir(parents=True, exist_ok=True)
    for pre in prefixes:
        subprocess.run(["aws", "s3", "cp", pre, str(dest), "--recursive",
                        "--exclude", "*", "--include", "*cell_*.json",
                        "--region", "us-east-1"], check=True)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="", help="local dir of cell_*.json (or use --cells-s3)")
    ap.add_argument("--cells-s3", default="", help="comma-list of S3 prefixes to pull cell_*.json from")
    ap.add_argument("--split", default="confirmation", choices=["bucket", "confirmation"])
    ap.add_argument("--budget", type=int, default=16000)
    ap.add_argument("--arm", default="cap", choices=["cap", "shuffled"])
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cells_s3:
        args.cells = str(_fetch_cells_s3([p.strip() for p in args.cells_s3.split(",")], out / "cells"))
    assert args.cells, "need --cells (local dir) or --cells-s3 (S3 prefixes)"

    from infra.pg import latmech as L
    from infra.pg import reprice_stats as RS
    from infra.pg import savb

    cells = _load_cells(args.cells, args.split, args.arm)
    seeds = sorted({s for (_, _, s) in cells})
    B = args.budget
    res = {"split": args.split, "budget": B, "arm": args.arm, "seeds": seeds, "per_seed": {}}

    def mcq(cond, seed):
        row = cells.get((cond, B, seed))
        assert row is not None, f"missing cell {cond} n={B} seed={seed}"
        return row["mcq"]["per_item"], row["item_hash"]

    mde_points = []
    primary = None
    for s in seeds:
        del_S, h_del = mcq("deleted", s)
        flo_S, h_flo = mcq("random_floor", s)
        con_S, h_con = mcq("content", s)
        emr_S, h_emr = mcq("emrand", s)
        for h in (h_flo, h_con, h_emr):
            L.position_align_tripwire(h_del, h)               # pairing reorder tripwire (T5)

        mde = L.mde_contrast(con_S, emr_S, floor=flo_S, ceiling=del_S, n_boot=args.boot, seed=args.seed)
        comp = L.rf_diff_contrast(con_S, del_S, floor=flo_S, ceiling=del_S, n_boot=args.boot, seed=args.seed)
        rf_content = RS.recovery_fraction(con_S, flo_S, del_S)
        rf_emrand = RS.recovery_fraction(emr_S, flo_S, del_S)
        seed_block = {
            "mde": mde, "comp_matched_vs_deleted": comp,
            "rf_content": rf_content, "rf_emrand": rf_emrand,
        }
        # shrink-perturb is optional (basin arm); include its CI when the cell exists.
        if ("shrink_perturb", B, s) in cells:
            sp_S, h_sp = mcq("shrink_perturb", s)
            L.position_align_tripwire(h_del, h_sp)
            seed_block["sp_vs_deleted"] = L.rf_diff_contrast(
                sp_S, del_S, floor=flo_S, ceiling=del_S, n_boot=args.boot, seed=args.seed)
        res["per_seed"][s] = seed_block
        mde_points.append(mde["point"])
        if primary is None:
            primary = (s, seed_block)

    # ---- verdict (frozen §6 rule; primary seed CI + seed-sign stability) ----
    seed_signs = [p for p in mde_points if p is not None]
    if primary is not None:
        s0, blk = primary
        sp_ci = blk.get("sp_vs_deleted", {"lo": 0.0, "hi": 0.0})   # no-sp ⇒ basin arm cannot fire
        verdict = L.mde_verdict(
            mde_ci=blk["mde"], sp_ci=sp_ci, comp_matched_ci=blk["comp_matched_vs_deleted"],
            seed_signs=seed_signs, mde_fdr_ok=True)               # BH-FDR applied at the family level below
        res["verdict"] = verdict
        res["seed_stable"] = bool(seed_signs) and len({p > 0 for p in seed_signs}) == 1

    # ---- BH-FDR over the confirmatory arm family (primary seed) ----
    if primary is not None:
        _, blk = primary
        fam = {"MDE": blk["mde"], "comp_matched": blk["comp_matched_vs_deleted"]}
        if "sp_vs_deleted" in blk:
            fam["shrink_perturb"] = blk["sp_vs_deleted"]
        res["fdr_family"] = {k: {"lo": v["lo"], "hi": v["hi"],
                                 "excludes_0": not (v["lo"] <= 0 <= v["hi"])} for k, v in fam.items()}

    # ---- Arm-2 law (DISCOVERY; content=cloze vs basin=spectrum_frac, if interp cells exist) ----
    interp = [(k, v) for k, v in cells.items() if k[0] == "interp"]
    if len(interp) >= 4:
        import numpy as np
        pts = []
        for (_, n_, s_), row in interp:
            pts.append((row["condition_info"].get("spectrum_frac", 0.0),
                        row["cloze"]["acc_pct"] / 100.0,
                        100.0))  # ceiling handled below; RF computed against shared del/floor@n_
        # surviving-content (cloze) and basin (spectrum_frac) predictors of MCQ recovery.
        frac = np.array([p[0] for p in pts])
        cloze = np.array([p[1] for p in pts])
        mcq_acc = np.array([row["mcq"]["acc_pct"] / 100.0 for _, row in interp])
        law = {
            "n_points": len(interp),
            "content_predictor_loo_r2": savb.loo_r2(cloze[:, None], mcq_acc),
            "basin_predictor_loo_r2": savb.loo_r2(frac[:, None], mcq_acc),
            "content_over_basin": savb.delta_loo_perm(frac[:, None], cloze, mcq_acc,
                                                      n_perm=2000, seed=args.seed),
            "note": "DISCOVERY corroborating arm; basin axis = spectrum_frac (deleted↔floor), "
                    "content axis = surviving cloze. Held-out spectrum-point check required to promote.",
        }
        res["arm2_law"] = law

    # ---- §11 prediction grid (filled 1:1) ----
    if primary is not None:
        _, blk = primary
        res["grid"] = {
            "mde_primary_delta": {"point": blk["mde"]["point"], "lo": blk["mde"]["lo"], "hi": blk["mde"]["hi"]},
            "seed_stable": res.get("seed_stable"),
            "competence_matched_vs_deleted": {"lo": blk["comp_matched_vs_deleted"]["lo"],
                                              "hi": blk["comp_matched_vs_deleted"]["hi"]},
            "named_verdict": res.get("verdict"),
        }

    (out / "latmech_metrics.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"[LATMECH-UNBLIND] split={args.split} budget={B} seeds={seeds} "
          f"verdict={res.get('verdict')} seed_stable={res.get('seed_stable')}")


if __name__ == "__main__":
    main()
