"""TESSERA aggregate — fill the §11 prediction grid from the arm artifacts. Spec §9 step 7, §11.

The grid is filled MECHANICALLY from whatever arm artifacts exist, and a row with no artifact is
written `PENDING` rather than omitted. That is the whole point: a grid that silently drops the rows
that did not run reads as a completed program, and this project has a standing rule against
exactly that (`pending` is a terminal-looking state only when it is shown).

BH q=0.05 over the three §7 families, m disclosed per family:
  F-K {K1 ΔLOO, K2 two sealed-prediction scores, K3 two footprint→n* predictions}  m=5
  F-X {install emergent, π0 survival, π0.5 survival}                                m=3
  F-S {core-survives, hop-disjoint}                                                 m=2
Everything else is census and is never in a family.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_aggregate --type r7i.2xlarge --no-weights \
      --args "--s s3://.../tessera_s_core/out/metrics.json --k1 ... --k24 ... --k3 ... --x ..." \
      --s3-prefix results/tessera_agg --budget-min 60
  # or locally, since it only reads JSON

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

FAMILIES = {"F-K": 5, "F-X": 3, "F-S": 2}
BH_Q = 0.05


def _read(p):
    if p is None:
        return None
    if str(p).startswith("s3://"):
        return json.loads(subprocess.run(["aws", "s3", "cp", str(p), "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def bh(pvals, m, q=BH_Q):
    """Benjamini-Hochberg at family size m (m DISCLOSED, not len(pvals) — a family whose members
    did not all run keeps its declared size, or the correction silently gets easier)."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    rej = [False] * len(pvals)
    kmax = -1
    for rank, i in enumerate(idx, start=1):
        if pvals[i] <= q * rank / m:
            kmax = rank
    for rank, i in enumerate(idx, start=1):
        if rank <= kmax:
            rej[i] = True
    return rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    for f in ("s", "k1", "k24", "k3", "x", "u3", "u3ck", "offline"):
        ap.add_argument(f"--{f}", default=None)
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    art = {k: _read(getattr(a, k)) for k in ("s", "k1", "k24", "k3", "x", "u3", "u3ck", "offline")}
    grid, pv = [], {"F-K": [], "F-X": [], "F-S": []}

    def row(n, quantity, hspace, observed, verdict, family=None, p=None):
        grid.append({"#": n, "quantity": quantity, "H_space": hspace,
                     "observed": observed, "verdict": verdict, "family": family, "p": p})
        if family and p is not None:
            pv[family].append((len(grid) - 1, p))

    # 1 — K1
    if art["k1"] and "law" in art["k1"]:
        law = art["k1"]["law"]
        d, ci = law["delta_loo_footprint"], law["delta_loo_footprint_ci"]
        row(1, "K1 ΔLOO(footprint) + CI + sign", "destroys / protects / nothing-still",
            f"ΔLOO {d['delta']:+.4f}, CI [{ci['lo95']:+.4f}, {ci['hi95']:+.4f}], "
            f"p_perm {d['p_perm']:.4f}, N3(permuted) {law['N3_permuted_labels']['delta']:+.4f}",
            law["verdict"]["verdict"], "F-K", d["p_perm"])
    else:
        row(1, "K1 ΔLOO(footprint) + CI + sign", "destroys / protects / nothing-still",
            "PENDING", "PENDING")

    # 2 — K2 sealed
    if art["k24"] and "K2_score" in art["k24"]:
        sc = art["k24"]["K2_score"]
        # Finding 11: K2 is a declared member of F-K and shipped with no family tag, so the
        # artifact could not be audited for family membership. Tagged with p=None — it is a
        # verdict-rule row, so it enters the family's ACCOUNTING without entering its BH ranking.
        row(2, "K2 sealed scores (π0, π0.5)", "hit / miss",
            f"{sc['n_hit']}/{sc['n_scored']} faculty rankings hit; the outcome (π0 more damaged "
            f"than π0.5 on every faculty) was already public in LACUNA before the seal was "
            f"written, so chance is near 1 and the information content is ~0 bits",
            ("HIT" if sc["n_scored"] and sc["n_hit"] > sc["n_scored"] / 2 else "MISS"),
            "F-K", None)
    else:
        row(2, "K2 sealed scores (π0, π0.5)", "hit / miss", "PENDING", "PENDING")

    # 3-5 — Arm X
    xr = (art["x"] or {}).get("pairs", {})
    band = (art["x"] or {}).get("identity_band")
    for n, pair, label, hspace in (
            (3, "install", "install emergent-feature fraction", "≈0 (PG2 verdict replicates) / >null"),
            (4, "pi0", "π0 feature survival", "present-unread / erased / VOID"),
            (5, "pi05", "π0.5 feature survival", "rides PG1 features / new-features surprise / VOID")):
        rec = xr.get(pair)
        if rec and band:
            sites = {L: s for L, s in rec.get("sites", {}).items() if "seeds" in s}
            if sites:
                em = {L: s.get("mean_emergent") for L, s in sites.items()}
                # `survival_fraction` is DEFINED as 1 - emergent_fraction in
                # `tessera_crosscoder.pair_fractions`, so it carries no information the emergent
                # fraction does not already carry. The quantity that answers the ERASURE question
                # -- are PG1's features gone from the child's stream? -- is `lost_fraction`, the
                # BEFORE-only latents. Reporting survival here would present an arithmetic
                # identity as a measurement of feature persistence, which it is not.
                lost = {L: (sum(x["lost_fraction"] for x in s["seeds"]) / len(s["seeds"]))
                        for L, s in sites.items()}
                shuf = {L: [x["emergent_fraction"] for x in s.get("shuffle_null", [])]
                        for L, s in sites.items()}
                vd_em = {L: (s.get("emergent_verdict") or {}).get("verdict")
                         for L, s in sites.items()}
                vd_lo = {L: (s.get("lost_verdict") or {}).get("verdict") for L, s in sites.items()}
                shl = {L: [z.get("lost_fraction") for z in (s.get("shuffle_null_full") or [])]
                       for L, s in sites.items()}
                # Each row's VERDICT must be the verdict of the quantity the row asks about. Rows 4
                # and 5 ask "present-unread / erased", which is the LOST question; row 3 asks about
                # emergent features. Reporting the emergent verdict on an erasure row would label
                # π0 by a statistic that is 0 for every unit in the lineage.
                is_erasure = hspace.startswith("present-unread") or "rides PG1" in hspace
                row(n, label, hspace,
                    f"emergent {em} (verdict {vd_em}) · LOST — the erasure signal — {lost} "
                    f"(verdict {vd_lo}) · shuffle-null emergent {shuf} · shuffle-null LOST {shl} · "
                    f"identity band emergent [{band['p2.5']:.4f}, {band['p97.5']:.4f}]",
                    str(vd_lo if is_erasure else vd_em), "F-X", None)
                continue
            row(n, label, hspace, "VOID — every site below MIN_PAIRED_TOKENS", "VOID")
            continue
        row(n, label, hspace, "PENDING", "PENDING")

    # 6-7 — Arm S
    if art["s"]:
        p_s1 = art["s"]["P_S1_control"]
        if p_s1["verdict"] != "PASS":
            row(6, "26-unit core consensus", "survives / VLA breaks it",
                f"P-S1 control {p_s1['verdict']} — n24 beyond-null on "
                f"{p_s1['n24_beyond_null_at_k8']}/{p_s1['n_matrices']} q/k matrices",
                "VOID-INSTRUMENT")
        else:
            row(6, "26-unit core consensus", "survives / VLA breaks it",
                f"n24 beyond-null {p_s1['n24_beyond_null_at_k8']}/{p_s1['n_matrices']} → "
                f"n26 beyond-null {p_s1['n26_beyond_null_at_k8']}/{p_s1['n_matrices']} (k=8)",
                ("SURVIVES" if p_s1["n26_beyond_null_at_k8"] > p_s1["n_matrices"] // 2
                 else "VLA-BREAKS-IT"), "F-S", None)
        ov = art["s"].get("overlaps", [])
        if ov:
            summ = {"~".join(o["pair"]): {"median": o["median_ci"]["point"],
                                          "shared": o["n_shared_channel"],
                                          "energy_aligned": o.get("n_energy_aligned", 0),
                                          "anti": o["n_anti_aligned"],
                                          "haar_disjoint": o["n_haar_disjoint"]} for o in ov}
            n_sh = sum(o["n_shared_channel"] for o in ov)
            n_en = sum(o.get("n_energy_aligned", 0) for o in ov)
            n_dj = sum(o["n_haar_disjoint"] for o in ov)
            n_an = sum(o["n_anti_aligned"] for o in ov)
            tot = n_sh + n_en + n_dj + n_an
            # A12 added the ENERGY-ALIGNED category and this row predated it. Labelling an
            # all-above-Haar result "HAAR-DISJOINT" because no cell cleared the SECOND null would
            # invert the finding: the overlaps are far above chance, they are simply no larger than
            # two UNRELATED fine-tunes reach.
            if n_sh > tot / 2:
                v7 = "SHARED-CHANNEL"
            elif n_en > tot / 2:
                v7 = "ENERGY-ALIGNED (above Haar everywhere, inside the unrelated-FT reference)"
            elif n_dj > tot / 2:
                v7 = "HAAR-DISJOINT"
            else:
                v7 = "MIXED"
            row(7, "cross-hop subspace overlap",
                "Haar-disjoint / energy-aligned / shared channel",
                json.dumps(summ) + f" · totals shared={n_sh} energy={n_en} disjoint={n_dj} "
                                   f"anti={n_an} of {tot}", v7, "F-S", None)
        else:
            row(7, "cross-hop subspace overlap", "Haar-disjoint / shared channel",
                "PENDING", "PENDING")
    else:
        row(6, "26-unit core consensus", "survives / VLA breaks it", "PENDING", "PENDING")
        row(7, "cross-hop subspace overlap", "Haar-disjoint / shared channel", "PENDING", "PENDING")

    # 8 — K3
    if art["k3"] and art["k3"].get("arms_result"):
        ar = art["k3"]["arms_result"]
        p3 = art["k3"].get("P3", {})
        measured = {k: v.get("n_star") for k, v in ar.items() if v.get("n_star") is not None}
        norange = {k: round(v.get("gain_full", 0.0), 4) for k, v in ar.items()
                   if v.get("n_star") is None}
        if p3.get("verdict") != "PASS":
            v8 = f"VOID — P3 {p3.get('verdict')}"
        elif not measured:
            v8 = "VOID — no arm cleared MIN_GAIN_FULL"
        else:
            # n* well above the search rank = redundant storage; near 1 = compact (QUORUM's frame)
            lab = ["%s n*=%.0f (%s)" % (k, n, "REDUNDANT" if n > 20 else "COMPACT")
                   for k, n in measured.items()]
            if norange:
                lab.append("no range: " + ", ".join(f"{k} gain={g}" for k, g in norange.items()))
            v8 = " · ".join(lab)
        row(8, "n*(PG1-VQA), n*(π0.5-VQA) + footprint predictions",
            "compact / redundant; hit / miss",
            json.dumps({k: {"n_star": v.get("n_star"), "gain_full": v.get("gain_full"),
                            "censored": v.get("censored"), "skipped": bool(v.get("skipped"))}
                        for k, v in ar.items()})
            + f" · P3 {p3.get('verdict')} n*={p3.get('n_star')} @scale {p3.get('scale_used')}",
            v8, "F-K", None)
    else:
        row(8, "n*(PG1-VQA), n*(π0.5-VQA) + footprint predictions",
            "compact / redundant; hit / miss", "PENDING", "PENDING")

    # 9 — gates and VOIDs
    gates = []
    if art["offline"]:
        gates.append(f"offline gate {art['offline']['n_checks']} checks "
                     f"all_passed={art['offline']['all_passed']}")
    if art["u3"]:
        gates.append(f"§0.1 U3 adjudication: {art['u3']['verdict']['verdict']}")
    if art["u3ck"]:
        gates.append(f"§0.1b artifact identity: {art['u3ck']['verdict']['verdict']}")
    if art["s"]:
        gates.append(f"P-S1: {art['s']['P_S1_control']['verdict']}")
    if art["k1"]:
        gates.append(f"P4 void faculties: {art['k1'].get('void_faculties')}")
    if art["x"]:
        voids = [f"{p}@L{L}" for p, r in xr.items()
                 for L, s in r.get("sites", {}).items() if "VOID" in s]
        gates.append(f"Arm X VOID sites: {voids or 'none'}")
    row(9, "gates + VOIDs fired", "enumerated", " · ".join(gates) or "PENDING",
        "ENUMERATED" if gates else "PENDING")

    # ---- BH over the three families, m disclosed ----------------------------------------------
    bh_out = {}
    for fam, m in FAMILIES.items():
        entries = [(i, p) for i, p in pv[fam] if p is not None]
        if not entries:
            bh_out[fam] = {"m_declared": m, "n_with_p": 0,
                           "note": "no member of this family reported a p-value; BH not applied"}
            continue
        rej = bh([p for _, p in entries], m)
        for (i, p), r in zip(entries, rej):
            grid[i]["bh_reject"] = bool(r)
        bh_out[fam] = {"m_declared": m, "n_with_p": len(entries),
                       "p": {grid[i]["quantity"]: p for i, p in entries},
                       "reject": {grid[i]["quantity"]: bool(r)
                                  for (i, _), r in zip(entries, rej)}}

    # Finding 11: state what each family DECLARED versus what it produced. A family whose
    # declared members did not all run keeps its declared m (the correction stays conservative),
    # but the gap must be visible in the artifact rather than inferred from the row list.
    for fam, m in FAMILIES.items():
        bh_out.setdefault(fam, {})["members_present_in_grid"] = sum(
            1 for g in grid if g.get("family") == fam)
        bh_out[fam]["declared_m"] = m
        bh_out[fam]["accounting"] = (
            f"{m} members declared; {sum(1 for g in grid if g.get('family') == fam)} present in "
            f"this grid. m is held at the DECLARED value — shrinking it to what ran would ease "
            f"the threshold after seeing which members survived.")

    n_pending = sum(1 for g in grid if g["verdict"] == "PENDING")
    rep = {"object": "TESSERA — §11 prediction grid, filled from the arm artifacts",
           "q": BH_Q, "families": FAMILIES, "bh": bh_out, "grid": grid,
           "n_rows": len(grid), "n_pending": n_pending,
           "complete": bool(n_pending == 0),
           "note": "a row with no artifact is PENDING, never omitted — a grid that drops its "
                   "unrun rows reads as a completed program",
           "elapsed_s": time.time() - t0}
    (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    for g in grid:
        print(f"[{g['#']}] {g['quantity']:<48s} {g['verdict']}", flush=True)
    print(f"\nTESSERA_AGG pending={n_pending}/{len(grid)}", flush=True)


if __name__ == "__main__":
    main()
