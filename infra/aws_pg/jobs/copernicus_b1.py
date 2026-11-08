"""CO-B1 (item 22 + item 75 rider) — is the L17 gate/up agreement STRUCTURAL or CONTAMINATION?

Spec §3 Tier B / §6.3. CPU-only. One re-stream of the frozen 24-delta cloud, three instruments.

THE OBJECT. Item 22's "second discriminator": the L17 gate_proj and up_proj AVOIDED subspaces (the
bottom-32 eigenvectors of each type's consensus ROW Gram) agree far above chance. Item 75's reframe
is carried: what is being compared at L17 is CONCENTRATED-WRITE structure, not "avoidance", and the
avoidance-ratio language does not appear in this arm's output.

WHY THREE INSTRUMENTS. Leave-one-out alone is underpowered at 1/24 mass and detects only a single
dominant carrier, so §3 requires all three in ONE re-stream:
  (i)   jackknife influence — CONTAMINATION fires iff any single fold drops the agreement by more
        than 4 jackknife SDs below the 24-fold mean, and the fold NAMES the repo;
  (ii)  subsample-stability curve over subsets of 4/8/12/18 — descriptive, banked;
  (iii) disjoint half-split — two 12-repo consensuses; STRUCTURAL requires the half-split agreement
        CI-separated above the Haar band.
POSITIVE CONTROL: a shared-rowspace carrier planted AT the observed effect size in ONE repo must be
caught by (i). A contamination detector that cannot catch planted contamination is not a detector.

ARITHMETIC. G_S = Σ_{i∈S} A_i A_iᵀ / ‖A_i‖²_F. Each ΔW is cached to fp16 scratch on the first pass
so any subset's Gram is a re-accumulation rather than a re-download; LOO uses G_total − term_i.
Extremes come from `mercator_rowgram.eig_extremes` (Lanczos at m = 16384 — a dense decomposition
there is O(n³) for 2k vectors nobody needs).

    pg run infra.aws_pg.jobs.copernicus_b1 \
        --args "--manifest-sha256 <sha>" --no-weights \
        --type r7i.2xlarge --s3-prefix results/copernicus_b1 --budget-min 360
"""
import argparse
import json
import pathlib

import numpy as np

from infra.pg import copernicus_core as C
from infra.pg import humboldt_core as H
from infra.pg import mercator_rowgram as RG

from infra.aws_pg.jobs.invariant_core import (
    PG1_REPO, _load_decoder, _decoder_matrices, _ftype, _delete_cache,
)
from infra.aws_pg.jobs.copernicus_a1_banks import load_manifest, MANIFEST

ARM = "CO-B1"
QUEUE_ITEM = "22"
WAVE = "WF"
LAYER = 17
TYPES = ("gate_proj", "up_proj")
K = 32
JACKKNIFE_SD = 4.0                      # §6.3 (i) — frozen
SUBSAMPLE_SIZES = (4, 8, 12, 18)        # §3 (ii)
SUBSAMPLE_DRAWS = 2
N_HAAR = 12                             # Haar reference draws for the agreement statistic


def agreement(V_gate, V_up) -> dict:
    """Item 22's statistic, both halves: mean cos² between the two avoided subspaces, and the
    overlap of their top-32 coordinate sets (the director note records 14/32 shared coordinates and
    cos² 0.196 ≈ 100× Haar at L17 — RECOMPUTED here, never quoted)."""
    cos2 = C.mean_cos2(V_gate, V_up)
    lev_g = (np.asarray(V_gate) ** 2).sum(axis=1)
    lev_u = (np.asarray(V_up) ** 2).sum(axis=1)
    top_g = set(np.argsort(-lev_g)[:K].tolist())
    top_u = set(np.argsort(-lev_u)[:K].tolist())
    inter = len(top_g & top_u)
    return {"mean_cos2": float(cos2), "top_coord_shared": int(inter),
            "top_coord_jaccard": float(inter / len(top_g | top_u)),
            "construction": f"mean cos² of the two bottom-{K} row subspaces + top-{K} leverage overlap"}


def bottom_subspace(G):
    lam_b, V_b, _lam_t, _V_t, _tr, _full = RG.eig_extremes(G, K)
    return np.ascontiguousarray(V_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--subsample-draws", type=int, default=SUBSAMPLE_DRAWS)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = pathlib.Path(H.cache_dir(out, "b1_scratch"))
    scratch.mkdir(parents=True, exist_ok=True)

    H.hf_setup()
    import torch

    man = load_manifest(a.manifest, a.manifest_sha256)
    repos, shas, base_sha = man["repos"], man["shas"], man["base_sha"]

    exp = C.Expectations(out / "copernicus_expectations.jsonl")
    exp.log(ARM, "b1_contamination_loo",
            "the L17 gate/up agreement SURVIVES leave-one-out — structural, not carried by one repo",
            confidence="mid-high", basis="spec §11 row CO-B1", family="boundary")
    exp.log(ARM, "b1_halfsplit_structural",
            "two disjoint 12-repo consensuses agree above the Haar band",
            confidence="mid-high", basis="spec §6.3 CO-B1 (iii)", family="boundary")
    at = C.Atlas(out, arm=ARM, run_id="copernicus_b1", wave=WAVE, queue_item=QUEUE_ITEM,
                 expected_ref=exp.require(ARM, "b1_contamination_loo"))
    gates = C.Gates(out)

    rep = {"object": "copernicus_b1_v1", "arm": ARM, "program": C.PROGRAM,
           "queue_item": QUEUE_ITEM, "layer": LAYER, "types": list(TYPES), "k": K,
           "base_repo": PG1_REPO, "base_sha": base_sha, "n_ft": len(repos),
           "manifest_sha256": man["sha256"],
           "reframe": ("item 75: the object at L17 is CONCENTRATED-WRITE structure; the "
                       "avoidance-ratio label is not used in this arm"),
           "provenance": C.provenance(), "utc": C.utc_now()}

    def save():
        (out / "copernicus_b1.json").write_text(json.dumps(rep, indent=2, default=float))
    save()

    # ---- ONE re-stream: accumulate G_total per type and cache each ΔW to fp16 scratch ----
    w_base = _load_decoder(PG1_REPO, base_sha)
    m_dim = {t: int(w_base[f"layers.{LAYER}.mlp.{t}.weight"].shape[0]) for t in TYPES}
    G = {t: np.zeros((m_dim[t], m_dim[t]), dtype=np.float64) for t in TYPES}
    fro2 = {t: {} for t in TYPES}
    for ridx, repo in enumerate(repos):
        print(f"[b1] stream {repo}@{shas[repo][:8]} ({ridx + 1}/{len(repos)})", flush=True)
        w_ft = _load_decoder(repo, shas[repo])
        matched = _decoder_matrices(w_base, w_ft)
        for key, (wb, wf) in matched.items():
            t = _ftype(key)
            if t not in TYPES or int(key.split("layers.")[1].split(".")[0]) != LAYER:
                continue
            A = (wf.to(torch.float64) - wb.to(torch.float64)).numpy()
            f2 = float((A ** 2).sum())
            assert f2 > 0, f"zero ΔW {key} in {repo}"
            G[t] += (A @ A.T) / f2
            fro2[t][repo] = f2
            np.save(scratch / f"{t}__{ridx}.npy", A.astype(np.float16))
            del A
        del w_ft, matched
        _delete_cache(repo)
    for t in TYPES:
        assert len(fro2[t]) == len(repos), f"{t}: {len(fro2[t])} terms, expected {len(repos)}"
    print(f"[b1] streamed {len(repos)} deltas; Gram dims {m_dim}", flush=True)

    def term(t, ridx):
        A = np.load(scratch / f"{t}__{ridx}.npy").astype(np.float64)
        return (A @ A.T) / fro2[t][repos[ridx]]

    def gram_subset(t, idxs):
        acc = np.zeros_like(G[t])
        for i in idxs:
            acc += term(t, i)
        return acc

    # ---- the full-24 agreement, and the Haar reference for it ---------------------------
    V_full = {t: bottom_subspace(G[t]) for t in TYPES}
    full = agreement(V_full[TYPES[0]], V_full[TYPES[1]])
    haar = [agreement(H.haar_subspace(m_dim[TYPES[0]], K, a.seed + 7 * d),
                      H.haar_subspace(m_dim[TYPES[1]], K, a.seed + 7 * d + 1))["mean_cos2"]
            for d in range(N_HAAR)] if m_dim[TYPES[0]] == m_dim[TYPES[1]] else []
    haar_band = C.band(haar) if haar else None
    rep["full24"] = full
    rep["haar_band"] = haar_band
    print(f"[b1] FULL-24 agreement: cos²={full['mean_cos2']:.4f} "
          f"shared top-{K}={full['top_coord_shared']}/{K} | Haar band="
          f"{[round(haar_band['lo'], 5), round(haar_band['hi'], 5)] if haar_band else 'n/a'}",
          flush=True)
    at.add("b1_agreement_full24", value=full["mean_cos2"], layer=LAYER, n=len(repos),
           flags={**full, "haar_band": haar_band})
    save()

    # ---- (i) jackknife influence -------------------------------------------------------
    loo = []
    for i, repo in enumerate(repos):
        Vs = {t: bottom_subspace(G[t] - term(t, i)) for t in TYPES}
        ag = agreement(Vs[TYPES[0]], Vs[TYPES[1]])
        loo.append({"held_out": repo, "idx": i, **ag})
        print(f"[b1][LOO] -{repo.split('/')[-1]}: cos²={ag['mean_cos2']:.4f} "
              f"shared={ag['top_coord_shared']}", flush=True)
        save()
    vals = np.array([r["mean_cos2"] for r in loo], dtype=np.float64)
    jk_mean, jk_sd = float(vals.mean()), float(vals.std(ddof=1))
    floor = jk_mean - JACKKNIFE_SD * jk_sd
    carriers = [loo[i]["held_out"] for i in range(len(loo)) if vals[i] < floor]
    rep["loo"] = {"folds": loo, "mean": jk_mean, "sd": jk_sd, "floor": floor,
                  "carriers": carriers}
    gates.record("b1_contamination_loo", not carriers, jackknife_mean=jk_mean, jackknife_sd=jk_sd,
                 floor=floor, n_folds=len(loo), carriers=carriers,
                 min_fold=float(vals.min()), max_fold=float(vals.max()),
                 band=f"CONTAMINATION fires iff any fold sits > {JACKKNIFE_SD} jackknife SDs below "
                      f"the {len(loo)}-fold mean (§6.3 (i)); a firing fold NAMES its repo",
                 prediction="agreement survives LOO (§11, mid-high)")

    # ---- POSITIVE CONTROL: plant a carrier at the observed effect size in ONE repo ------
    # A single repo is given a shared gate/up rowspace block scaled to reproduce the measured
    # agreement; the influence rule must catch it. Without this the LOO gate cannot be trusted
    # when it does NOT fire.
    rng = np.random.default_rng(a.seed + 99)
    planted_idx = 0
    Qs = np.linalg.qr(rng.standard_normal((m_dim[TYPES[0]], K)))[0][:, :K]

    def _isotropic_minus_block(acc, Q, c, block=2048):
        """acc += c·(I − QQᵀ), in row blocks. The plant must make span(Q) a shared LOW-energy
        block — the avoided subspace is the BOTTOM of the spectrum, so adding energy along Q
        would push it out of the bottom-k rather than into it. Blocked because QQᵀ at m=16384 is
        another 2.1 GB temporary nobody needs."""
        m = acc.shape[0]
        acc.flat[::m + 1] += c
        for i0 in range(0, m, block):
            i1 = min(i0 + block, m)
            acc[i0:i1] -= c * (Q[i0:i1] @ Q.T)
        return acc

    G_pl = {}
    for t in TYPES:
        tr = float(np.trace(G[t]))
        # the plant carries the same trace share ONE repo does, so it is AT the observed effect
        # size rather than an easy giant
        acc = G[t] - term(t, planted_idx)
        G_pl[t] = _isotropic_minus_block(acc, Qs, (tr / len(repos)) / m_dim[t])
    V_pl = {t: bottom_subspace(G_pl[t]) for t in TYPES}
    ag_pl_full = agreement(V_pl[TYPES[0]], V_pl[TYPES[1]])
    # the same cloud WITHOUT the plant: holding the planted repo out and adding nothing
    V_drop = {t: bottom_subspace(G[t] - term(t, planted_idx)) for t in TYPES}
    ag_drop = agreement(V_drop[TYPES[0]], V_drop[TYPES[1]])
    caught = bool(abs(ag_pl_full["mean_cos2"] - ag_drop["mean_cos2"]) > JACKKNIFE_SD * jk_sd)
    rep["positive_control"] = {"planted_repo": repos[planted_idx],
                               "with_plant": ag_pl_full, "plant_held_out": ag_drop,
                               "delta": ag_pl_full["mean_cos2"] - ag_drop["mean_cos2"],
                               "caught": caught}
    gates.record("b1_positive_control_catches_planted_carrier", caught,
                 planted_repo=repos[planted_idx],
                 agreement_with_plant=ag_pl_full["mean_cos2"],
                 agreement_plant_removed=ag_drop["mean_cos2"],
                 threshold=JACKKNIFE_SD * jk_sd,
                 band="a planted shared-rowspace carrier AT the observed effect size, in ONE repo, "
                      "must move the agreement by more than the influence rule's own threshold")
    del V_pl, G_pl
    save()

    # ---- (iii) disjoint half-split ------------------------------------------------------
    order = list(rng.permutation(len(repos)))
    h1, h2 = sorted(order[:len(repos) // 2]), sorted(order[len(repos) // 2:])
    halves = {}
    for name, idxs in (("half1", h1), ("half2", h2)):
        Vs = {t: bottom_subspace(gram_subset(t, idxs)) for t in TYPES}
        halves[name] = {"idxs": idxs, "repos": [repos[i] for i in idxs],
                        **agreement(Vs[TYPES[0]], Vs[TYPES[1]])}
        print(f"[b1][{name}] cos²={halves[name]['mean_cos2']:.4f}", flush=True)
        save()
    hs_min = min(halves[n]["mean_cos2"] for n in halves)
    structural = bool(haar_band is not None and hs_min > haar_band["hi"])
    rep["half_split"] = halves
    gates.record("b1_halfsplit_structural", structural,
                 half1=halves["half1"]["mean_cos2"], half2=halves["half2"]["mean_cos2"],
                 haar_band=[haar_band["lo"], haar_band["hi"]] if haar_band else None,
                 n_haar_draws=N_HAAR,
                 band="both disjoint 12-repo consensuses must clear the Haar band's upper edge",
                 prediction="both halves agree above Haar (§6.3 (iii))")

    # ---- (ii) subsample-stability curve (descriptive, banked) ---------------------------
    curve = []
    for size in SUBSAMPLE_SIZES:
        for d in range(a.subsample_draws):
            idxs = sorted(rng.choice(len(repos), size, replace=False).tolist())
            Vs = {t: bottom_subspace(gram_subset(t, idxs)) for t in TYPES}
            ag = agreement(Vs[TYPES[0]], Vs[TYPES[1]])
            curve.append({"size": size, "draw": d, "idxs": idxs, **ag})
            print(f"[b1][sub{size}#{d}] cos²={ag['mean_cos2']:.4f}", flush=True)
            at.add("b1_agreement_subsample", value=ag["mean_cos2"], layer=LAYER, n=size,
                   flags={**ag, "draw": d, "role": "descriptive row (§3 (ii))"})
            save()
    rep["subsample_curve"] = curve

    # ---- item 75 rider: effective rank on every wide-type cell --------------------------
    ranks = {}
    for t in TYPES:
        tr = float(np.trace(G[t]))
        tr2 = float(np.sum(G[t] * G[t]))          # tr(G²) for a symmetric G
        ranks[t] = {"trace": tr, "trace_sq": tr2, "effective_rank": float(tr * tr / tr2),
                    "m": m_dim[t],
                    "construction": "tr²/tr(G²) (item 75 standing rule); full spectra not computed "
                                    "at m=16384 and their absence is stated, not implied"}
        at.add("b1_effective_rank", value=ranks[t]["effective_rank"], type=t, layer=LAYER,
               flags=ranks[t])
    rep["effective_rank"] = ranks

    # THE VALIDITY GATE IS CONSUMED HERE, not merely recorded beside the verdict. §6.3 says the
    # planted carrier "must be caught by (i)"; this line used to compose from `carriers` and
    # `structural` alone, so a run whose positive control read False would still have banked a
    # clean REFUTER-DID-NOT-FIRE off a detector that provably cannot detect. It nearly did: the
    # 2026-08-18 run was SIGKILLed by its budget five minutes short of this line with
    # `caught=False` already in its artifact (journal, CO-B1). The rule is imported from the
    # scorer so there is exactly ONE composition in the program, not two that can drift.
    from infra.aws_pg.jobs.copernicus_b1_score import compose as _compose
    rep["verdict"], rep["verdict_basis"] = _compose(
        {"verdict": ("CONTAMINATION" if carriers else "NO-CARRIER"),
         "valid": bool(caught),
         "positive_control": {"caught": bool(caught)}},
        {"verdict": "STRUCTURAL" if structural else "NOT-SEPARATED"},
        True)
    save()
    at.close()
    at.to_parquet()
    print("=" * 78, flush=True)
    eff = {t: round(v["effective_rank"], 1) for t, v in ranks.items()}
    print(f"COPERNICUS_B1_OK verdict={rep['verdict']} full24_cos2={full['mean_cos2']:.4f} "
          f"carriers={carriers} structural={structural} eff_rank={eff}", flush=True)


if __name__ == "__main__":
    main()
