#!/usr/bin/env python3
"""AM-6's $0 pre-gate: does the SINKFORM closed form retrodict the V3 noise trend?

If it does, AM-6's two new seeds are not the cheapest way to learn anything and
the arm is SKIPPED with its budget returned to reserve (pre-specified in the
spec's §2 AM-6 gate — a planned branch, not an improvisation).

THE MODEL (SINKFORM Result 6, recomputed from that project's own artifacts, not
quoted from prose). For Adam, accumulated squared motion per coordinate after T
steps is proportional to

    lambda(r) = (T^2 r^2 + T) / (1 + r^2),   r = mu/sigma  (consistency)

and per-step gradient energy per coordinate is proportional to mu^2 + sigma^2.
Writing rho = sigma^2/mu^2 = 1/r^2 — which is exactly what the V3 cell banks per
bank (`bank_ratio`) and per layer (`median_elementwise_ratio`) — this becomes

    lambda(rho) = T (T + rho) / (1 + rho),      grad(rho) = mu^2 (1 + rho)

Batch size enters ONLY through rho (minibatch noise falls with B; mu does not),
so for each bank b the B-dependence of the delivery ratio is predicted, free of
every fitted constant except one per-bank scale:

    delivery(b, B)  ~  [lambda(rho_b(B)) / lambda(rho_pop(B))]
                       / [(1 + rho_b(B)) / (1 + rho_pop(B))]

The scale is removed by normalising every curve to its own B = 2 value, so the
remaining rungs are OUT-OF-SAMPLE predictions of the trend.

FROZEN GATE (written before the script was ever run; §2 AM-6):
    FIT   iff  the predicted relative change in delivery(core32) from the lowest
               to the highest rung is within +-50% of the observed relative
               change, AND the predicted sign matches the observed sign for >= 3
               of the 4 structured banks.
    -> FIT: AM-6 is SKIPPED, budget to reserve, the closed form is recorded as
            the account of the trend (with its n=1-seed caveat stated).
    -> NO-FIT: AM-6 RUNS.

RESOLUTION GATE (the SINKFORM lesson: show the check can resolve its target
before trusting it): the same comparison is run against a NULL model (delivery
constant in B) and against a MUTANT of the closed form (lambda with r^2 replaced
by r, i.e. the wrong power). The check is only trusted if it separates the true
model from at least one of them.
"""
import argparse
import json

import numpy as np

BANKS = ("core32", "anti32", "topsigma32", "bottomsigma32")
HAAR = ("haar32_s0", "haar32_s1", "haar32_s2")


def lam(rho, T, power=2.0):
    """lambda_p(r) = (T^2 r^p + T)/(1 + r^p) written in rho = 1/r^2, so
    r^p = rho^(-p/2) and lambda_p = T (T + rho^(p/2)) / (1 + rho^(p/2)).
    p = 2 is SINKFORM Result 6; p = 1 is the MUTANT (wrong power on r)."""
    rho = np.asarray(rho, dtype=np.float64)
    x = rho ** (power / 2.0)
    return T * (T + x) / (1.0 + x)


def predict(rows, T, power=2.0):
    """Predicted delivery shape per bank per batch, normalised at the lowest B."""
    by_layer = [r for r in rows if r.get("layer") is not None]
    Bs = sorted({r["batch"] for r in by_layer})
    rho_pop = {B: float(np.mean([r["median_elementwise_ratio"]
                                 for r in by_layer if r["batch"] == B])) for B in Bs}
    out = {}
    for b in BANKS + HAAR:
        rho_b = {B: float(np.mean([r["bank_ratio"][b] for r in by_layer
                                   if r["batch"] == B])) for B in Bs}
        pred = {B: float((lam(rho_b[B], T, power) / lam(rho_pop[B], T, power))
                         / ((1 + rho_b[B]) / (1 + rho_pop[B]))) for B in Bs}
        out[b] = {"rho_bank": rho_b, "pred_raw": pred,
                  "pred_norm": {B: pred[B] / pred[Bs[0]] for B in Bs}}
    return out, Bs, rho_pop


def observed(rows):
    d = {r["batch"]: r["delivery"] for r in rows if r.get("layer") is None}
    Bs = sorted(d)
    return {b: {B: d[B][b] / d[Bs[0]][b] for B in Bs} for b in d[Bs[0]]}, Bs, d


def score(rows, T, tol=0.5, min_banks=3):
    obs, Bs, level = observed(rows)
    pred, Bs2, rho_pop = predict(rows, T)
    assert Bs == Bs2, (Bs, Bs2)
    lo, hi = Bs[0], Bs[-1]
    per_bank = {}
    for b in BANKS:
        o = obs[b][hi] - 1.0
        p = pred[b]["pred_norm"][hi] - 1.0
        per_bank[b] = {"obs_rel_change": o, "pred_rel_change": p,
                       "sign_match": bool(np.sign(o) == np.sign(p) and o != 0)}
    core = per_bank["core32"]
    within = bool(abs(core["pred_rel_change"] - core["obs_rel_change"])
                  <= tol * abs(core["obs_rel_change"]))
    n_sign = sum(v["sign_match"] for v in per_bank.values())
    fit = bool(within and n_sign >= min_banks)

    # resolution: can this comparison separate the model from a null / a mutant?
    null_err = abs(0.0 - core["obs_rel_change"])
    mut, _, _ = predict(rows, T, power=1.0)
    mut_err = abs((mut["core32"]["pred_norm"][hi] - 1.0) - core["obs_rel_change"])
    true_err = abs(core["pred_rel_change"] - core["obs_rel_change"])
    resolves = bool(null_err > 1.5 * true_err or mut_err > 1.5 * true_err)

    # the diagnostic that explains WHY the prediction comes out where it does:
    # rho_bank / rho_pop is the only bank-specific quantity the model sees
    rho_tbl = {b: {B: pred[b]["rho_bank"][B] / rho_pop[B] for B in Bs}
               for b in BANKS + HAAR}
    return {"gate": "FIT -> SKIP AM-6" if fit else "NO-FIT -> RUN AM-6",
            "fit": fit, "core_within_tol": within, "n_sign_match": n_sign,
            "per_bank": per_bank, "rho_pop": rho_pop, "batches": Bs, "T": T,
            "rho_bank_over_rho_pop": rho_tbl,
            "observed_delivery_level_core_over_haar":
                {B: level[B]["core32"] / float(np.mean([level[B][h] for h in HAAR]))
                 for B in Bs},
            "haar_pred_norm": {h: pred[h]["pred_norm"] for h in HAAR},
            "haar_obs_norm": {h: obs[h] for h in HAAR},
            "resolution": {"resolves": resolves, "err_true_model": true_err,
                           "err_null_constant": null_err, "err_mutant_power1": mut_err}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="banked v3_rows.jsonl")
    ap.add_argument("--steps", type=int, required=True, help="T used in that run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.rows) if l.strip()]
    res = score(rows, a.steps)
    txt = json.dumps(res, indent=2, default=float)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)


if __name__ == "__main__":
    main()
