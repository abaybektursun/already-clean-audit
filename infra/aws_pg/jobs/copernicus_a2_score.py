"""CO-A2 scoring — the §6.2 decision procedure over the artifacts `copernicus_a2.py` banked.

CPU-only and separate from the training job on purpose: the fit, the prediction intervals and the
permutation test are cheap, and iterating on them must never cost GPU time. It takes the producer's
prefix as an ARGUMENT and asserts on an empty dependency (M-D99), and it re-reads thresholds.json
by hash so the admissibility band it scores against is provably the one frozen before any ladder
arm's delivery was read.

`assess` is a PURE function of (artifact, thresholds, anchor rows) so both substantive outcomes can
be planted and checked in the §14.2 dry-run before a real artifact exists.

Also fills the CO-B3 (item 24) depth gate from the per-layer rows the training job banked.

    pg run infra.aws_pg.jobs.copernicus_a2_score \
        --args "--a2-s3 s3://.../results/copernicus_a2c/out --thresholds-sha256 <sha>" \
        --no-weights --type r7i.2xlarge --s3-prefix results/copernicus_a2_score --budget-min 30
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import copernicus_core as C

ARM = "CO-A2"
QUEUE_ITEM = "86"
WAVE = "W0"
BANK = "core32"
INTERIOR = tuple(range(1, 17))          # §6.3 CO-B3: interior = L1..L16
BOUNDARY_HI = 17                        # boundary = {L0, L17}; the depth gate reads L17
PHI_LOW, PHI_HIGH = 0.25, 0.75          # §2.2 REPORTING bands only, never CI gates
MECHANISMS = ("lr", "noise")


def by_arm(rows):
    d = {}
    for r in rows:
        d.setdefault(r["arm"], []).append(r)
    return d


def seed_band(values, stored: dict) -> dict:
    """§6.2.3's acceptance region from the reference arm's seed spread.

    THE 2.5–97.5 QUANTILE IS THE WRONG SHAPE HERE, and the first scoring proved it: over three
    seeds that interval lies strictly INSIDE the observed range, so the calibration arm's own
    extreme seeds fall outside their own band and the reference arm reads as inadmissible. A gate
    the calibration arm cannot pass is degenerate in exactly the way §6.5 forbids a gate that
    cannot fail — the same defect, mirrored.

    `C.band` is built for ≥5 NULL draws, where a quantile band is right. An ACCEPTANCE region over
    3 seeds is the RANGE widened by its own width — the construction §9.1c already uses for the
    anchor seed-band and `anchor_check` already applies here. It contains every calibration seed by
    construction, which is the minimum a self-consistent acceptance region must do.

    The raw per-seed values are read from the FROZEN thresholds file, so this changes the shape of
    the region and not the data behind it: those values were frozen after the unrotated reference
    and before any ladder arm's delivery was read.
    """
    v = [float(x) for x in values] if values else []
    if len(v) < 2:
        return {**stored, "construction": "stored quantile band (no raw seed values available)"}
    lo, hi = min(v), max(v)
    span = hi - lo
    return {"lo": lo - span, "hi": hi + span, "seeds": v, "raw_range": [lo, hi],
            "stored_quantile_band": [stored.get("lo"), stored.get("hi")],
            "construction": ("reference-arm seed RANGE widened by its own width (§9.1c); the "
                             "stored 2.5-97.5 quantile band is reported beside it but is not "
                             "scored — over 3 seeds it excludes its own calibration points")}


def admissibility(arms: dict, thr: dict) -> dict:
    """§6.2.3 — all three axis-aligned v̂ statistics inside the reference arm's OWN spread. Every
    bound comes from the frozen file's raw seed values; nothing is invented here."""
    disp_band = seed_band(thr.get("vhat_dispersion_core"),
                          thr["admissibility"]["dispersion_band"])
    lev_band = seed_band(thr.get("vhat_leverage_share_core"),
                         thr["admissibility"]["leverage_share_band"])
    sp_band = seed_band(thr.get("vhat_pairwise_spearman"),
                        thr["admissibility"]["spearman_band"])
    out = {}
    for name, st in arms.items():
        disp = [s["vhat"]["dispersion_core_mean"] for s in st]
        lev = [s["vhat"]["leverage_share_core_mean"] for s in st]
        sp = [s["vhat"]["spearman_mean"] for s in st if s["vhat"]["spearman_mean"] is not None]
        d_ok = all(disp_band["lo"] <= x <= disp_band["hi"] for x in disp)
        l_ok = all(lev_band["lo"] <= x <= lev_band["hi"] for x in lev)
        s_ok = (not sp) or all(x >= sp_band["lo"] for x in sp)
        out[name] = {"ok": bool(d_ok and l_ok and s_ok), "dispersion": disp,
                     "leverage_share": lev, "spearman": sp, "dispersion_ok": d_ok,
                     "leverage_ok": l_ok, "spearman_ok": s_ok,
                     "bands": {"dispersion": [disp_band["lo"], disp_band["hi"]],
                               "leverage_share": [lev_band["lo"], lev_band["hi"]],
                               "spearman_lo": sp_band["lo"],
                               "construction": disp_band["construction"],
                               "dispersion_detail": disp_band,
                               "leverage_detail": lev_band,
                               "spearman_detail": sp_band}}
    return out


def build_ladders(arms, adm, had_ppl, had_S, release_gap, void_confound):
    """One fit per ladder MECHANISM, on admissible arms only, with the unrotated reference as a
    point on every ladder.

    The above/inside calls are lifted to the TOP level of each entry because that is the contract
    `copernicus_core.a2_verdict` reads. Leaving them nested under S_net/S_path (as the first draft
    did) makes the verdict raise KeyError at the very last step of the primary.
    """
    ladders = {}
    for mech in MECHANISMS:
        pts = [s for n, st in arms.items() for s in st
               if s["kind"] == mech and n not in void_confound]
        pts += list(arms["unrotated"])
        names = sorted({s["arm"] for s in pts})
        adm_names = [n for n in names if adm[n]["ok"]]
        pts = [s for s in pts if s["arm"] in adm_names]
        ppls = [s["ppl"] for s in pts]
        brackets = bool(pts and min(ppls) <= had_ppl <= max(ppls))
        entry = {"mechanism": mech, "arms": names, "admissible_arms": adm_names,
                 "n_points": len(pts), "ppl_range": [min(ppls), max(ppls)] if pts else None,
                 "hadamard_ppl": had_ppl, "brackets": brackets,
                 "admissible": bool(adm_names),
                 "above_net": False, "above_path": False,
                 "inside_net": False, "inside_path": False, "pi_ok": False}
        if len(pts) >= 4 and brackets:
            for key in ("S_net", "S_path"):
                pi = C.linear_prediction_interval([s["ppl"] for s in pts],
                                                  [s[key] for s in pts], x0=had_ppl)
                above = bool(had_S[key] > pi["pi_hi"])
                inside = bool(pi["pi_lo"] <= had_S[key] <= pi["pi_hi"])
                entry[key] = {**pi, "hadamard": had_S[key], "above": above, "inside": inside}
                suffix = "net" if key == "S_net" else "path"
                entry[f"above_{suffix}"] = above
                entry[f"inside_{suffix}"] = inside
            # §2.2 precision floor: a PI wider than the effect it must adjudicate cannot decide.
            entry["pi_ok"] = bool(entry["S_net"]["half_width"] <= release_gap["S_net"])
        else:
            entry["reason"] = ("fewer than 4 admissible points" if len(pts) < 4
                               else "ladder does not bracket hadamard's ppl")
        ladders[mech] = entry
    return ladders


def anchor_check(arms, anchor_rows) -> dict:
    """§6.2.1 — the unrotated and hadamard replicas must reproduce the banked delivery ratio within
    the banked SEED-BAND (§9.1c: the band over ALL banked seeds of the anchor rows)."""
    detail, ok = {}, True
    for g in ("unrotated", "hadamard"):
        banked = [r["delivery_ratio"] for r in anchor_rows
                  if r["bank"] == BANK and r["gauge"] == g]
        assert banked, f"anchor carries no {g}/{BANK} rows"
        got = [s["delivery_net"][BANK] for s in arms[g]]
        lo, hi = min(banked), max(banked)
        span = hi - lo
        good = all(lo - span <= v <= hi + span for v in got)
        detail[g] = {"banked_band": [lo, hi], "recomputed": got, "ok": bool(good),
                     "band": "banked seed range, widened by its own width"}
        ok = ok and good
    detail["ok"] = bool(ok)
    return detail


def depth_gate(per_layer_rows) -> dict:
    """CO-B3 (item 24): per-layer release = delivery(core32) in hadamard minus the same layer in
    unrotated. DEPTH-NONUNIFORM fires iff ≥12 of the 16 interior layers exceed L17's release in
    ALL seeds — a sign-test construction, because n=3 seeds does not support a bootstrap."""
    core_pl = [r for r in per_layer_rows if r["bank"] == BANK]
    if not core_pl:
        return {"fires": False, "reason": "no per-layer rows banked"}

    def dl(arm_name, seed, L):
        hit = [r["delivery_ratio"] for r in core_pl
               if r["arm"] == arm_name and r["seed"] == seed and r["layer"] == L]
        return hit[0] if hit and hit[0] is not None else None

    counts = []
    for sd in sorted({r["seed"] for r in core_pl}):
        h17, u17 = dl("hadamard", sd, BOUNDARY_HI), dl("unrotated", sd, BOUNDARY_HI)
        if h17 is None or u17 is None:
            continue
        rel17 = h17 - u17
        n_above = 0
        for L in INTERIOR:
            h, u = dl("hadamard", sd, L), dl("unrotated", sd, L)
            if h is not None and u is not None and (h - u) > rel17:
                n_above += 1
        counts.append(n_above)
    return {"per_seed_interior_above_L17": counts, "n_interior": len(INTERIOR),
            "fires": bool(counts and all(c >= 12 for c in counts)),
            "rule": ">=12 of 16 interior layers above L17's release in ALL seeds (sign test)"}


def merge_extension(primary: dict, ext: dict) -> dict:
    """Fold §2.2's ONE bisection extension into the primary artifact.

    Only arms the primary does NOT already have are added — that is the new dose point, and it is
    the only thing the extension exists to provide. The extension's own unrotated and hadamard arms
    are an INDEPENDENT REPLICATION and are recorded separately, never pooled: the thresholds and the
    anchor comparison belong to the run they were calibrated on, and silently averaging two runs'
    reference arms would both move the frozen band and inflate n. Same non-pooling discipline the
    spec applies to CO-A2 versus SHACKLETON V2a.
    """
    for key in ("steps", "reference_lr", "base_sha"):
        assert primary.get(key) == ext.get(key), (
            f"extension disagrees with the primary on {key}: {ext.get(key)} vs {primary.get(key)} "
            f"— it is not the same recipe and cannot sit on the same ladder")
    have = {r["arm"] for r in primary["rows"]}
    new_rows = [r for r in ext["rows"] if r["arm"] not in have]
    assert new_rows, "extension added no new arm — nothing to bracket with"
    merged = dict(primary)
    merged["rows"] = list(primary["rows"]) + new_rows
    merged["extension"] = {
        "added_arms": sorted({r["arm"] for r in new_rows}),
        "replicate_arms_not_pooled": sorted({r["arm"] for r in ext["rows"] if r["arm"] in have}),
        "replicates": [r for r in ext["rows"] if r["arm"] in have],
        "note": ("§2.2 allows exactly one bisection extension, assigned to the lr ladder; the "
                 "extension's own reference arms are an independent replication and are NOT pooled")}
    return merged


def assess(rep_a2: dict, thr: dict, anchor_rows: list) -> dict:
    """The whole §6.2 decision procedure as a pure function of banked inputs."""
    rows = rep_a2["rows"]
    arms = by_arm(rows)
    assert "unrotated" in arms and "hadamard" in arms, sorted(arms)

    anc = anchor_check(arms, anchor_rows)
    adm = admissibility(arms, thr)
    out = {"anchor": anc, "admissibility": adm}

    # The positive control: the composite gate MUST fail on the rotated arm. A validity gate that
    # cannot fail is a defect, so an admissible hadamard voids the instrument rather than passing.
    if adm["hadamard"]["ok"]:
        out["verdict"] = C.VOID_INSTRUMENT
        out["verdict_reason"] = ("§6.2.3 positive control: the v̂ admissibility gate did NOT fail "
                                 "on hadamard, so it cannot discriminate and no admissibility "
                                 "claim ships")
        return out

    had_ppl = float(np.mean([s["ppl"] for s in arms["hadamard"]]))
    unr_S = {k: float(np.mean([s[k] for s in arms["unrotated"]])) for k in ("S_net", "S_path")}
    had_S = {k: float(np.mean([s[k] for s in arms["hadamard"]])) for k in ("S_net", "S_path")}
    release_gap = {k: abs(had_S[k] - unr_S[k]) for k in ("S_net", "S_path")}

    # VOID-FOR-CONFOUND first: an arm whose whole eight-bank battery rose together is a
    # global-motion artifact and is removed BEFORE anything else is evaluated (§6.2.4 branch 1).
    banks = rep_a2["banks"]
    unr_delivery = {b: float(np.mean([s["delivery_net"][b] for s in arms["unrotated"]]))
                    for b in banks}
    void_confound = [n for n in arms if n != "unrotated"
                     and C.whole_battery_rose(
                         {b: float(np.mean([s["delivery_net"][b] for s in arms[n]]))
                          for b in banks}, unr_delivery)]

    ladders = build_ladders(arms, adm, had_ppl, had_S, release_gap, void_confound)

    cands = [n for n in arms if n not in ("unrotated", "hadamard")
             and adm[n]["ok"] and n not in void_confound]
    if cands:
        near = min(cands, key=lambda n: abs(float(np.mean([s["ppl"] for s in arms[n]])) - had_ppl))
        perm = C.exact_permutation_p([s["S_net"] for s in arms["hadamard"]],
                                     [s["S_net"] for s in arms[near]])
        perm["comparator"] = near
    else:
        perm = {"complete_separation": False, "note": "no admissible comparator arm"}

    vd = C.a2_verdict({"anchor_ok": anc["ok"], "ladders": ladders, "permutation": perm,
                       "void_for_confound": void_confound})
    out.update({"verdict": vd["verdict"], "verdict_detail": vd, "ladders": ladders,
                "permutation": perm, "void_for_confound": void_confound,
                "release": {"unrotated_S": unr_S, "hadamard_S": had_S, "gap": release_gap,
                            "phi_reporting_band": [PHI_LOW, PHI_HIGH],
                            "note": "phi bands are REPORTING only, never CI gates (§2.2)"},
                # §6.2.5: the ladder-shape row ships whichever verdict lands.
                "ladder_shape": [{"arm": s["arm"], "kind": s["kind"], "seed": s["seed"],
                                  "ppl": s["ppl"], "S_net": s["S_net"], "S_path": s["S_path"],
                                  "core_delivery": s["delivery_net"][BANK],
                                  "admissible": adm[s["arm"]]["ok"]} for s in rows]})
    return out


def fetch(a2_s3: str, dest: pathlib.Path):
    assert a2_s3.startswith("s3://"), f"--a2-s3 must be an s3 uri, got {a2_s3!r}"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", a2_s3, str(dest), "--region", "us-east-1"], check=True)
    main_j = dest / "copernicus_a2.json"
    assert main_j.exists(), f"A2 dependency empty at {a2_s3}"
    rep = json.loads(main_j.read_text())
    assert rep.get("rows"), "A2 artifact carries no rows — nothing to score"
    return rep, dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--a2-s3", required=True)
    ap.add_argument("--thresholds-sha256", required=True)
    ap.add_argument("--extend-s3", default="",
                    help="comma-separated prefixes holding §2.2's ONE bisection extension")
    ap.add_argument("--mq2-anchor-s3",
                    default="s3://pg-lineage-769063704696/results/mercator_mq2/out/M-Q_stage2_result.json")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rep_a2, dest = fetch(a.a2_s3, out / "_a2")
    # The thresholds come from the PRIMARY run and only the primary run: they were frozen after its
    # unrotated reference, before any ladder delivery was read, and an extension cannot move them.
    thr = C.load_thresholds(dest / "thresholds.json", a.thresholds_sha256)
    for i, ex in enumerate([x.strip() for x in a.extend_s3.split(",") if x.strip()]):
        rep_ext, _ = fetch(ex, out / f"_ext{i}")
        rep_a2 = merge_extension(rep_a2, rep_ext)
        print(f"[a2-score] merged extension {ex}: added {rep_a2['extension']['added_arms']}, "
              f"replicates not pooled {rep_a2['extension']['replicate_arms_not_pooled']}",
              flush=True)
    anchor = json.loads(subprocess.run(
        ["aws", "s3", "cp", a.mq2_anchor_s3, "-", "--region", "us-east-1"],
        check=True, capture_output=True, text=True).stdout)

    # The scoring artifact must carry the expectations it SPENT, or the §11 grid cannot pair a
    # prediction with the gate that consumed it from this prefix alone. Copy, do not re-log: these
    # rows were written by the training job before any of it was measured.
    exp_src = dest / "copernicus_expectations.jsonl"
    if exp_src.exists():
        (out / "copernicus_expectations.jsonl").write_text(exp_src.read_text())
    exp = C.Expectations(exp_src)
    at = C.Atlas(out, arm=ARM, run_id="copernicus_a2_score", wave=WAVE, queue_item=QUEUE_ITEM,
                 expected_ref=exp.require(ARM, "primary"))
    gates = C.Gates(out)

    res = assess(rep_a2, thr, anchor["rows"])
    gates.record("a2_anchor_recompute", bool(res["anchor"]["ok"]),
                 **{k: v for k, v in res["anchor"].items() if k != "ok"})
    # `ok` is Gates.record's own positional argument, so an evidence dict that carries its own "ok"
    # key collides with it. The admissibility record does. Rename it rather than splat blindly —
    # the same bookkeeping-vs-evidence collision the Gates docstring already names three instances of.
    _had = {("hadamard_admissible" if k == "ok" else k): v
            for k, v in res["admissibility"]["hadamard"].items()}
    gates.record("a2_vhat_gate_fails_on_hadamard", not res["admissibility"]["hadamard"]["ok"],
                 **_had,
                 band=("the composite denominator-intactness gate MUST fail on the rotated arm; "
                       "a validity gate that cannot fail is a defect (§6.5)"))
    if "ladders" in res:
        gates.record("a2_primary", res["verdict"] == C.A2_MECHANISM_CLEAN,
                     verdict=res["verdict"], surviving=res["verdict_detail"].get("surviving", []),
                     hadamard_S=res["release"]["hadamard_S"],
                     unrotated_S=res["release"]["unrotated_S"],
                     void_for_confound=res["void_for_confound"],
                     band="per-ladder 95% prediction interval at hadamard's own ppl (§2.2)")

    pl_path = dest / "a2_per_layer_rows.json"
    if pl_path.exists():
        b3 = depth_gate(json.loads(pl_path.read_text()))
        res["b3"] = b3
        gates.record("b3_depth_nonuniform", b3["fires"],
                     per_seed_interior_above_L17=b3.get("per_seed_interior_above_L17", []),
                     n_interior=b3.get("n_interior", 0), rule=b3.get("rule", ""),
                     band="sign-test construction; no seed bootstrap at n=3",
                     prediction="release strong interior, weak L17 (§11, mid)")
        if b3.get("per_seed_interior_above_L17"):
            at.add("b3_interior_release_count",
                   value=float(np.mean(b3["per_seed_interior_above_L17"])), bank_id=BANK,
                   n=len(b3["per_seed_interior_above_L17"]),
                   flags={"rider": "CO-B3 (item 24)",
                          "per_seed": b3["per_seed_interior_above_L17"]})

    res.update({"object": "copernicus_a2_score_v1", "arm": ARM, "program": C.PROGRAM,
                "a2_s3": a.a2_s3, "extend_s3": a.extend_s3,
                "thresholds_sha256": a.thresholds_sha256,
                "extension": rep_a2.get("extension"),
                "mq2_anchor_s3": a.mq2_anchor_s3,
                "provenance": C.provenance(), "utc": C.utc_now()})
    (out / "copernicus_a2_score.json").write_text(json.dumps(res, indent=2, default=float))
    at.close()
    at.to_parquet()
    print("=" * 78, flush=True)
    print(f"COPERNICUS_A2_SCORE_OK verdict={res['verdict']} "
          f"surviving={res.get('verdict_detail', {}).get('surviving')} "
          f"void_for_confound={res.get('void_for_confound')}", flush=True)


if __name__ == "__main__":
    main()
