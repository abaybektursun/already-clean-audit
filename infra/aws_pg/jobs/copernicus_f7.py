"""CO-F7 (item 43) — the M-6 overshoot, characterized from banked ladder artifacts. CHARTER.

QUESTION. M-6's `m6_ladder_reaches_the_true_finetune` recorded VOID-INSTRUMENT at a relative gap of
-0.394: the decoded r=128 model scores BETTER on the task than the fine-tune it reconstructs. Item
43 asks what the overshoot IS, and floats "reconstruction as regularization". This job reads M-6's
own artifact and asks which account it supports. CHARTER: no verdict token, and the precondition is
carried, not repaired -- M-6 stays VOID-INSTRUMENT whatever this dossier finds.

TWO ACCOUNTS, EACH WITH THE OBSERVATION THAT KILLS IT, named before the numbers are read:

  (A) RECONSTRUCTION AS REGULARIZATION. A low-rank reconstruction of dW drops noise the full
      fine-tune carried, so the decoded model is genuinely better. This predicts the decoded model
      beats the BASE, not merely the fine-tune. Killed by: mean(decoded - base) > 0 at every rank.

  (B) A RATIO WITH A VANISHING DENOMINATOR. The relative gap is normalised by the base->fine-tune
      distance. If fine-tuning barely moves the task loss on this eval, that denominator is near
      zero and a large relative gap can sit on top of almost nothing. Killed by: a base->fine-tune gap that is
      large and consistently signed, with the overshoot a real fraction of it.

WHY (B) IS WORTH TAKING SERIOUSLY: a relative gap is only interpretable when its denominator has a
stable sign. A fine-tune that is BETTER than base has a negative base->true distance, so the same
absolute overshoot enters a pooled mean with the opposite sign. This job therefore reports the
denominator distribution and splits the cohort by its sign, rather than pooling.

The absolute overshoot in nats is reported next to every relative one. A ratio whose numerator and
denominator are both ~1e-3 is a number the reader must be able to see the scale of.

INPUT: `results/mercator_wpb5/out/M-6/M-6_result.json`, M-6's own immutable output.

    python3 -m infra.aws_pg.jobs.copernicus_f7 --out out
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import copernicus_core as C

ARM = "CO-F7"
QUEUE_ITEM = "43"
WAVE = "WF"
REGISTER = "CHARTER (described artifact, no verdict)"
PRECONDITION = ("M-6's recorded verdict is VOID-INSTRUMENT and stays so: this dossier "
                "characterizes the overshoot, it cannot rehabilitate the arm.")

M6_RESULT = ("s3://pg-lineage-769063704696/results/mercator_wpb5/out/M-6/M-6_result.json")


def s3_json(uri):
    r = subprocess.run(["aws", "s3", "cp", uri, "-", "--region", "us-east-1"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout, f"unreadable: {uri}"
    return json.loads(r.stdout)


def denominators(base_loss, true_loss):
    """base -> fine-tune task-loss distance, per fine-tune. This is what a relative gap divides by,
    so its spread and its SIGN decide whether a relative gap means anything."""
    d = {k: float(v) - float(base_loss) for k, v in true_loss.items()}
    vals = np.array(list(d.values()))
    return {"per_finetune": d,
            "n": len(d),
            "n_worse_than_base": int((vals > 0).sum()),
            "n_better_than_base": int((vals < 0).sum()),
            "mean": float(vals.mean()), "median": float(np.median(vals)),
            "min": float(vals.min()), "max": float(vals.max()),
            "sign_is_stable": bool((vals > 0).all() or (vals < 0).all())}


def rank_row(row, r, base_loss, true_loss):
    """One rung: where the decoded models sit relative to base and to their own fine-tunes."""
    pf = row["per_finetune"]
    to_base = np.array([float(v["task_loss"]) - float(base_loss) for v in pf.values()])
    to_true = np.array([float(v["task_loss"]) - float(true_loss[k]) for k, v in pf.items()])
    den = np.array([float(true_loss[k]) - float(base_loss) for k in pf])
    # relative gap ONLY where the denominator's sign is meaningful (fine-tune worse than base);
    # pooling a negative denominator flips the sign of the same absolute overshoot.
    mask = den > 0
    return {"r": int(r), "n_finetunes": len(pf),
            "abs_overshoot_nats_mean": float(to_true.mean()),
            "abs_overshoot_nats_median": float(np.median(to_true)),
            "n_beating_their_finetune": int((to_true < 0).sum()),
            "decoded_minus_base_mean": float(to_base.mean()),
            "n_beating_base": int((to_base < 0).sum()),
            "relative_gap_worse_than_base_only": (
                float((to_true[mask] / den[mask]).mean()) if mask.any() else None),
            "n_in_relative_gap": int(mask.sum()),
            "fraction_of_write_recovered": float((to_base[mask] / den[mask]).mean())
                                            if mask.any() else None}


def assess(rows, den):
    """Which account the artifact supports. Not a verdict — a CHARTER row states what it sees."""
    beats_base = any(r["decoded_minus_base_mean"] < 0 for r in rows)
    a_killed = not beats_base
    b_supported = (not den["sign_is_stable"]) or abs(den["mean"]) < 1e-2
    return {
        "A_reconstruction_as_regularization": {
            "prediction": "decoded beats BASE, not merely the fine-tune",
            "observed_decoded_beats_base_at_any_rank": beats_base,
            "killed_at_its_named_killer": a_killed},
        "B_ratio_with_vanishing_denominator": {
            "prediction": "base->fine-tune distance near zero and/or unstably signed",
            "denominator_sign_is_stable": den["sign_is_stable"],
            "denominator_mean_nats": den["mean"],
            "supported": b_supported},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    m6 = s3_json(M6_RESULT)
    base_loss, true_loss = m6["base_task_loss"], m6["true_task_loss"]
    curve = m6["curve"]
    cap_rows = m6["capability_rows"]
    assert len(curve) == len(cap_rows), (
        f"M-6 curve has {len(curve)} rungs but {len(cap_rows)} capability rows")

    den = denominators(base_loss, true_loss)
    rows = [rank_row(cr, c["r"], base_loss, true_loss) for c, cr in zip(curve, cap_rows)]

    rep = {"object": "copernicus_f7_v1", "arm": ARM, "program": "copernicus",
           "queue_item": QUEUE_ITEM, "register": REGISTER, "precondition": PRECONDITION,
           "input_artifact": M6_RESULT,
           "m6_recorded": {"ladder_reaches_true_finetune": m6["ladder_reaches_true_finetune"],
                           "ranks": m6["ranks"], "base_task_loss": base_loss,
                           "base_ppl": m6["base_ppl"]},
           "denominator": den, "rungs": rows, "accounts": assess(rows, den),
           "provenance": C.provenance(), "utc": C.utc_now()}
    (out / "copernicus_f7.json").write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps({"denominator": {k: den[k] for k in
                                      ("n", "n_worse_than_base", "mean", "sign_is_stable")},
                      "accounts": rep["accounts"]}, indent=2, default=float), flush=True)
    for r in rows:
        print(f"[f7] r={r['r']:4d} abs_overshoot={r['abs_overshoot_nats_mean']:+.6f} nats  "
              f"beat_ft={r['n_beating_their_finetune']}/{r['n_finetunes']}  "
              f"beat_base={r['n_beating_base']}/{r['n_finetunes']}", flush=True)


if __name__ == "__main__":
    main()
