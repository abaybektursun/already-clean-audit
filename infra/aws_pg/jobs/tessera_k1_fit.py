"""TESSERA K1 — THE MDE. Does the curvature footprint predict faculty survival? Spec §2 K1, §3.

PLENUM's headline is that faculty survival is predicted by nothing in its frozen set: recipe
membership ΔLOO −0.004340 (p 0.5347), ‖ΔW‖ −0.859324, similarity −0.091806, out-of-sample
R² −0.995717 (recomputed here from `plenum_agg_fc`, never quoted). But the one predictor QUORUM
actually validated — the curvature footprint, ρ = 0.654 FDR on n* — was not in that set. K1 adds
it as a fourth predictor to the SAME frozen pipeline and reads the one CI test.

WHAT IS REUSED VERBATIM, AND WHY THAT IS THE WHOLE POINT
  `plenum_core.predictor_matrix`   builds (membership, ‖ΔW‖, similarity, y, groups) in a fixed
                                   row order; the footprint is aligned by REPLAYING that exact
                                   nested loop and asserting equal length.
  `plenum_core.delta_loo_grouped`  leave-one-UNIT-out ΔLOO with the across-unit permutation null,
                                   both tails.
  `plenum_core.boot_delta_loo`     the unit-bootstrap CI. §6's rule is CI-based (CI-over-p).
  `cleave_core.mechanism_census`   the footprint itself — QUORUM's own construction.
Nothing about the fit is re-implemented here. If K1 lands positive, it lands through the pipeline
that reported the null.

THE FOOTPRINT, AND ITS RANK
  footprint(u, f) = Σ_M ‖ΔW_M‖² · energy_fraction(ΔW_M, U_sus^f[L(M)], r) / Σ_M ‖ΔW_M‖²
i.e. the energy-weighted mean over the 126 decoder matrices of the fraction of that matrix's write
energy lying in faculty f's top-r curvature directions at PG1 — `mechanism_census(...)`'s
`susceptibility` entry, which is exactly the quantity QUORUM's Arm 3 correlated with n*.

**The registered rank is r = 679** (amendment A6b), because that is the rank at which QUORUM
validated the predictor, and because CLEAVE R1 showed the low-rank Q subspace is NOT reproducible
across measurements (18.6% agreement at r=20) — a footprint read at small r would inherit that
instability. The other ranks in `CENSUS_RANKS` are computed and reported as a labelled census and
are NEVER searched for a significant one: the §7 F-K family contains the r=679 test only.

CONTROLS THAT CAN FAIL
  N3   the same fit on PERMUTED labels must not reward the footprint (a pipeline that rewards any
       added column would report K1 live whatever the footprint was).
  P4   a faculty whose curvature bank failed `fd_directional_check` is VOID, not fitted.
  floor  every footprint is reported beside its Haar chance floor r/d; a footprint AT the floor is
       a write with no curvature preference, and the fit is told so by the number, not by a flag.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_k1_fit --type r7i.2xlarge --no-weights \
      --args "--banks-s3 s3://.../tessera_k1_banks_{f}/out --agg-fc s3://.../plenum_agg_fc/out" \
      --s3-prefix results/tessera_k1_fit --budget-min 600

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time

BUCKET = "pg-lineage-769063704696"
PG1_REPO = "google/paligemma-3b-pt-224"
FACULTIES = ("vqa", "ocr", "caption")
CENSUS_RANKS = (20, 50, 350, 679)
REGISTERED_RANK = 679          # amendment A6b — QUORUM's validated rank
N_LAYERS = 18
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _read_json(p):
    if str(p).startswith("s3://"):
        return json.loads(subprocess.run(["aws", "s3", "cp", str(p), "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def _fetch(uri, dest):
    pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "cp", uri, dest, "--only-show-errors"], check=True)
    return dest


def _delete_cache(repo_id):
    import shutil
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", ""), "hub")
    d = os.path.join(hub, "models--" + repo_id.replace("/", "--"))
    if os.path.isdir(d):
        shutil.rmtree(d)


def _decoder_sd(repo):
    """{(layer, type): tensor} of one repo's LM decoder matrices, canonical keys."""
    import re
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file
    from infra.pg.lacuna_shell import canon
    pat = re.compile(r"^(?:language_model\.)?layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)\.weight$")
    sd = {}
    for f in [x for x in list_repo_files(repo) if x.endswith(".safetensors")]:
        for k, v in load_file(hf_hub_download(repo, f)).items():
            c = canon(k)
            m = pat.match(c)
            if m and m.group(2) in PROJ and v.ndim == 2:
                sd[(int(m.group(1)), m.group(2))] = v
    assert len(sd) == N_LAYERS * len(PROJ), f"{repo}: {len(sd)} decoder matrices, want 126"
    return sd


def footprints_for_unit(base_sd, ft_sd, banks_by_faculty, ranks):
    """{faculty: {rank: energy-weighted susceptibility fraction}} + the chance floor.

    Energy weighting (‖ΔW_M‖²) is `quorum_arm0`'s convention: an unweighted mean over 126 matrices
    would let a near-zero matrix count as much as the one carrying the write.
    """
    from infra.pg import cleave_core as CL
    out = {}
    for fac, banks in banks_by_faculty.items():
        acc = {r: 0.0 for r in ranks}
        floor = {r: None for r in ranks}
        wsum = 0.0
        for (L, t), W in base_sd.items():
            dW = (ft_sd[(L, t)].float() - W.float()).numpy()
            fro2 = float((dW ** 2).sum())
            if fro2 <= 0:
                continue
            for r in ranks:
                cen = CL.mechanism_census(dW, banks[L], r, t)
                acc[r] += cen["susceptibility"] * fro2
                floor[r] = cen["chance_floor"]
            wsum += fro2
        assert wsum > 0, f"{fac}: all-zero ΔW reached the footprint"
        out[fac] = {"footprint": {r: acc[r] / wsum for r in ranks},
                    "chance_floor": floor, "total_fro2": wsum}
    return out


def aligned_footprint(unit_rows, faculty_order):
    """Replay `predictor_matrix`'s nested loop so the footprint column matches its rows exactly."""
    import numpy as np
    col = []
    for u in unit_rows:
        for f in faculty_order:
            if u["S"].get(f) is None:
                continue                      # dropped by predictor_matrix too
            col.append(float(u["footprint"][f]))
    return np.asarray(col, dtype=np.float64)


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--agg-fc", required=True, help="plenum_agg_fc metrics.json (s3 or local)")
    ap.add_argument("--census", required=True, help="comma list of plenum census metrics.json")
    ap.add_argument("--predictions", required=True, help="sealed predictions json (similarity)")
    ap.add_argument("--banks", required=True,
                    help="comma list faculty=uri of tessera_k1_banks_<f>.pt")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    from infra.pg import plenum_core as PC
    from infra.pg import plenum_map as PM

    rep = {"object": "TESSERA K1 — the MDE", "registered_rank": REGISTERED_RANK,
           "census_ranks": list(CENSUS_RANKS),
           "label": "CONFIRMATORY through the frozen PLENUM pipeline; the outcome labels are "
                    "already known to us, so the clean out-of-sample test is K2 and every future "
                    "unit (spec §2 K1 'honest scope')"}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    # ---- the curvature banks, one per faculty ------------------------------------------------
    banks_by_faculty, cert = {}, {}
    for spec in a.banks.split(","):
        fac, uri = spec.split("=", 1)
        assert fac in FACULTIES, f"unknown faculty {fac!r}"
        p = _fetch(uri, f"/tmp/k1banks_{fac}.pt")
        banks_by_faculty[fac] = torch.load(p, map_location="cpu", weights_only=False)
        m = _read_json(uri.rsplit("/", 1)[0] + "/metrics.json")
        cert[fac] = {"faculty_certified": m.get("faculty_certified"),
                     "certified_layers": m.get("certified_layers"),
                     "banks_sha256": m.get("banks_sha256"),
                     "base_ppl": m.get("base_ppl")}
        print(f"[banks] {fac}: {len(banks_by_faculty[fac])} layers, "
              f"certified={cert[fac]['faculty_certified']}", flush=True)
    # P4 — a faculty whose operator failed the fp32 certify is VOID, not fitted
    void_facs = [f for f in banks_by_faculty if cert[f]["faculty_certified"] is False]
    facs_ok = [f for f in FACULTIES if f in banks_by_faculty and f not in void_facs]
    rep["bank_certification"] = cert
    rep["void_faculties"] = void_facs
    assert facs_ok, "every faculty bank failed the fp32 certify — K1 is VOID"

    # ---- the footprints ----------------------------------------------------------------------
    from infra.pg.plenum_map import FT_REPO, FT_SLUGS
    base_sd = _decoder_sd(PG1_REPO)
    fps = {}
    for slug in FT_SLUGS:
        repo = FT_REPO.format(slug=slug)
        ft_sd = _decoder_sd(repo)
        fps[f"ft-{slug}"] = footprints_for_unit(base_sd, ft_sd,
                                                {f: banks_by_faculty[f] for f in facs_ok},
                                                CENSUS_RANKS)
        r = REGISTERED_RANK
        print(f"[fp] ft-{slug:<14s} " + " ".join(
            f"{f}={fps[f'ft-{slug}'][f]['footprint'][r]:.4f}" for f in facs_ok)
            + f"  floor={fps[f'ft-{slug}'][facs_ok[0]]['chance_floor'][r]:.4f}", flush=True)
        del ft_sd
        _delete_cache(repo)
        rep["footprints"] = fps
        _w()

    # ---- the frozen fit ----------------------------------------------------------------------
    agg = _read_json(a.agg_fc)
    cen_cells = []
    for cf in a.census.split(","):
        cen_cells.extend(_read_json(cf).get("cells", []))
    dw = {c["unit"]: c["dw_norm_lm"] for c in cen_cells}
    sim = _read_json(a.predictions)["similarity"]
    law_readout = agg["law_readout_per_faculty"]
    ub = PM.unit_by_id()

    S_by = {}
    for c in agg["cells"]:
        if law_readout.get(c["faculty"]) == c["readout"]:
            S_by.setdefault(c["unit"], {})[c["faculty"]] = c["S"]

    rows, excluded = [], []
    for uid in [u["id"] for u in PM.units() if u["role"] == "fit"]:
        if uid not in S_by or uid not in dw or uid not in sim or uid not in fps:
            excluded.append({"unit": uid, "why": "no measured grid row / no census / no sim / "
                                                 "no footprint (not a CARRY-banked FT)"})
            continue
        rows.append({"unit": uid, "dw_norm": dw[uid],
                     "recipe_faculties": set(ub[uid]["recipe_faculties"]),
                     "similarity": sim[uid], "S": S_by[uid],
                     "footprint": {f: fps[uid][f]["footprint"][REGISTERED_RANK] for f in facs_ok}})
    rep["law_excluded"] = excluded

    mem, dwv, simv, y, grp, info = PC.predictor_matrix(rows, facs_ok)
    fp_col = aligned_footprint(rows, facs_ok)
    assert fp_col.size == y.size, (
        f"footprint column {fp_col.size} != {y.size} outcome rows — the alignment replay of "
        f"predictor_matrix's loop is wrong, and a misaligned predictor is a silent scramble")

    X_base = np.column_stack([mem, dwv, simv])
    law = {"n_units_fit": len(rows), "faculties": facs_ok, **info}
    # the three PLENUM predictors, RECOMPUTED here (never quoted) so K1's baseline is its own
    law["delta_loo_membership"] = PC.delta_loo_grouped(np.column_stack([dwv, simv]), mem, y, grp,
                                                       n_perm=a.n_perm, seed=a.seed)
    law["delta_loo_dw"] = PC.delta_loo_grouped(np.column_stack([mem, simv]), dwv, y, grp,
                                               n_perm=a.n_perm, seed=a.seed)
    law["delta_loo_similarity"] = PC.delta_loo_grouped(np.column_stack([mem, dwv]), simv, y, grp,
                                                       n_perm=a.n_perm, seed=a.seed)
    # THE K1 TEST
    law["delta_loo_footprint"] = PC.delta_loo_grouped(X_base, fp_col, y, grp,
                                                      n_perm=a.n_perm, seed=a.seed)
    law["delta_loo_footprint_ci"] = PC.boot_delta_loo(X_base, fp_col, y, grp)

    # N3 — the same test on permuted labels
    rng = np.random.default_rng(a.seed + 1)
    law["N3_permuted_labels"] = PC.delta_loo_grouped(X_base, fp_col, y[rng.permutation(y.size)],
                                                     grp, n_perm=a.n_perm, seed=a.seed)

    d, ci = law["delta_loo_footprint"], law["delta_loo_footprint_ci"]
    if not ci["excludes_zero"]:
        verdict = {"verdict": "H_nothing-still", "why":
                   "the footprint's ΔLOO CI includes 0 under the frozen pipeline — the fourth "
                   "predictor family is dead too, and PLENUM's unpredictability upgrades to "
                   "'not even the QUORUM-validated predictor'"}
    elif d["delta"] < 0:
        verdict = {"verdict": "H_curvature-protects", "why":
                   "curvature-aligned writes SPARE the faculty (registered as the alternative "
                   "direction, opposite sign)"}
    else:
        verdict = {"verdict": "H_curvature-destroys", "why":
                   "writes landing in a faculty's curvature subspace destroy it — the registered "
                   "direction, and the program's first pre-training triage signal"}
    law["verdict"] = verdict
    rep["law"] = law
    rep["elapsed_s"] = time.time() - t0
    _w()
    print("TESSERA_K1", json.dumps({"delta": d["delta"], "ci": [ci["lo95"], ci["hi95"]],
                                    "p_perm": d["p_perm"], "N3": law["N3_permuted_labels"]["delta"],
                                    "verdict": verdict["verdict"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
