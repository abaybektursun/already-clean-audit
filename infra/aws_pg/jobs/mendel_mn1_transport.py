"""MENDEL MN-1 — the MDE: sink transport along the lineage vs the traffic-null (queue item 26a).

THE QUESTION. A hop's sink is where that hop declined to write. Is it in the same place the
PARENT's sink was — beyond where the parent's activation TRAFFIC alone would put it?

Per (type, layer) cell of each hop, at k = 32 (primary) and k = 8 (kernel-resolved secondary):

    A_parent  = top-k    coords by leverage of the parent's consensus core32   (transport predictor)
    A_traffic = top-k    coords of the parent's activation RMS                 (the traffic null)
    A_hop     = bottom-k coords by column write energy of this hop's ΔW        (the measurand)
    delta     = Jaccard(A_parent, A_hop) − Jaccard(A_traffic, A_hop)

FROZEN RULE (spec §2, unchanged): INHERITED-BEYOND-TRAFFIC iff the paired CI on delta clears 0 at
>= 2 independent hops; TRAFFIC-ONLY iff no hop clears; else AMBIGUOUS.

WHY THE NULL IS INSIDE THE STATISTIC. From the banked M-8 table (`mercator_w0gpu4`), the
activation-magnitude address already predicts PG1's measured core32 at mean Jaccard 0.4160 — it
BEAT the mechanism-derived predictor (0.3068) and M-8's refuter fired. A raw parent↔hop overlap in
the 0.4s would therefore look like powerful inheritance evidence and be almost entirely activation
energy. H_inert is the null this arm is built to be able to report.

THE CIRCULARITY FIX (freeze A2). PG1's core32 is the consensus over exactly the 24 descendants, so
for hop `pg1→ft_i` the parent bank is rebuilt LEAVE-ONE-OUT:
`G_loo(i) = G_all − ΔW_iᵀΔW_i/‖ΔW_i‖²_F`. The Gram is a sum, so this is a subtraction plus one
eigendecomposition per (cell, i) — not a second pass over the weights.

THE VALIDITY GATE (freeze A7). Each cell reports the Jaccard between the addresses of its two
row-parity halves. Cells at or below their own haar band are EXCLUDED and counted; a hop that
loses more than half its cells is VOID-INSTRUMENT, consumed inside the verdict expression.

CPU only (r7i, `--no-weights`): it downloads decoders from HF, differences them, and never runs a
forward pass.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import darwin_w0 as W0
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mendel_core as MC
from infra.pg import mendel_program as MP
from infra.pg import mendel_transport as MT

ARM = "MN-1"
TYPES = H.TYPES_RESID                      # 6 residual types, n = 2048 (freeze A3)
KS = (32, 8)                               # primary, kernel-resolved secondary
# The A7 gate excludes a cell whose split-half self-consistency does not beat the MAX of N_HAAR
# random addresses, so its false-pass rate on a structureless ΔW is ~1/(N_HAAR+1) — measured 0.030
# at 16 and 0.000/300 at 64 (tests/test_mendel_mn1_job.py). 64 keeps the gate's own error well
# below any effect the arm could claim, and haar draws are a set intersection, so they are free.
N_HAAR = 64

# The parent's traffic vectors, banked by G4 on the frozen wikitext batch (freeze A8).
G4_PREFIX = f"{MC.RESULTS_S3}/mendel_g4_v1/out"
# gemma-2b's traffic, banked by SHACKLETON's anchor cell (sites: q_proj_in, o_proj_in,
# gate_proj_in — and its banks cover q_proj alone, so that hop is q_proj x 18 anyway, freeze A4).
G2B_ACTSTATS = (f"{MC.RESULTS_S3}/shackleton_r_v1_anchor_gemma2b/cells/"
                f"anchor_gemma2b_actstats/actstats.npz")
G2B_BANKS = f"{MC.RESULTS_S3}/shackleton_census_g2b/out/banks"
G2B_REPO = "google/gemma-2b"


def _bare(sd):
    """Strip a leading 'model.' and drop embed_tokens/lm_head. REUSED from vlat_retention so every
    lineage member's decoder is keyed identically and a diff is a diff of the same object."""
    from infra.aws_pg.jobs.vlat_retention import _bare_decoder
    return _bare_decoder(sd)


def _lineage_decoder(name: str):
    """(bare decoder state dict, source sha) for PG1 / pi0 / pi05.

    REUSED verbatim from `vlat_retention._extract_decoder`, which already carries the π0/π0.5
    backbone remap and its compartment-leak assert (an action-expert key entering the backbone set
    fails loud rather than silently widening the diff).
    """
    from infra.aws_pg.jobs.vlat_retention import _extract_decoder
    dec, sha = _extract_decoder(name, "cpu", None)
    return _bare(dec), sha


def _gemma2b_decoder():
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM
    sha = HfApi().model_info(G2B_REPO).sha
    m = AutoModelForCausalLM.from_pretrained(G2B_REPO, revision=sha, torch_dtype=torch.float32,
                                             device_map="cpu")
    inner = m.model if hasattr(m, "model") else m
    sd = _bare(inner.state_dict())
    del m
    return sd, sha


def _ft_decoder(repo: str):
    """One of the 24 official PG1 fine-tunes as a bare decoder, then deleted from the HF cache.

    The delete matters: 24 x ~5.6 GB would fill the root volume long before the last fine-tune.
    """
    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaForConditionalGeneration
    from infra.aws_pg.jobs.invariant_core import _delete_cache
    from infra.pg.entities import resolve_submodule_paths
    sha = HfApi().model_info(repo).sha
    m = PaliGemmaForConditionalGeneration.from_pretrained(
        repo, revision=sha, torch_dtype=torch.float32, device_map="cpu",
        attn_implementation="eager")
    lm = m.get_submodule(resolve_submodule_paths(m)["language_model"])
    if hasattr(lm, "lm_head") and hasattr(lm, "model"):
        lm = lm.model
    sd = _bare(lm.state_dict())
    del m
    _delete_cache(repo)
    return sd, sha


def _delta(child_sd, parent_sd, t: str, layer: int) -> np.ndarray:
    """ΔW = W_child − W_parent for one decoder matrix, in fp64.

    fp64 because the operands are near-identical over most of the matrix and the quantity of
    interest is their difference: differencing in bf16 and promoting afterwards would quantise
    exactly the small column norms this arm reads.
    """
    import torch
    key = WT.matrix_key(t, layer)
    for name, sd in (("child", child_sd), ("parent", parent_sd)):
        assert key in sd, f"{name} decoder has no {key} (keys e.g. {sorted(sd)[:3]})"
    c, p = child_sd[key], parent_sd[key]
    assert c.shape == p.shape, f"{key}: child {tuple(c.shape)} vs parent {tuple(p.shape)}"
    return (c.to(torch.float64) - p.to(torch.float64)).numpy()


def _gemma2b_traffic(npz, t: str, layer: int):
    """gemma-2b's per-coordinate activation traffic from the SHACKLETON anchor bank.

    The bank stores Σ_tokens x², not the RMS. The address is the top-k, and sqrt is strictly
    increasing on non-negatives, so Σx² and RMS give the SAME address; the constant token count is
    likewise irrelevant to a ranking. Returning the square root anyway keeps this vector on the
    same scale as G4's `act_rms__*`, so the two parents' traffic scores are the same kind of
    object if they are ever compared directly.
    """
    key = f"L{layer:02d}_{t}_in_sum_sq"
    assert key in npz, f"{key} absent from the g2b actstats bank (have e.g. {list(npz)[:3]})"
    v = np.asarray(npz[key], dtype=np.float64)
    assert v.ndim == 1 and (v >= 0).all() and v.sum() > 0, f"{key}: not a positive energy profile"
    return np.sqrt(v)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _write_cell(at, hop_id, t, L, k, cell, excluded, model_id, model_sha):
    """One cell's rows: the treatment and §5's structured + haar controls beside it, under ONE
    statistic name so `check_control_trinity` can see them as one contrast."""
    base = {"k": k, "self_jaccard": cell["stability"]["self_jaccard"], "excluded": excluded}
    at.add("transport_jaccard", value=cell["observed"]["jaccard"], layer=L, type=t,
           bank_id="core32", hop_id=hop_id, control_id=MC.TREATMENT, model_id=model_id,
           model_sha=model_sha,
           flags={**base, "predictor": "parent_sink",
                  "n_intersection": cell["observed"]["n_intersection"]})
    at.add("transport_jaccard", value=cell["traffic"]["jaccard"], layer=L, type=t,
           bank_id="core32", hop_id=hop_id, control_id=MC.STRUCTURED, model_id=model_id,
           model_sha=model_sha,
           flags={**base, "predictor": "parent_activation_rms",
                  "n_intersection": cell["traffic"]["n_intersection"],
                  "why": "M-8's named incumbent; it BEAT the mechanism predictor in mercator_w0gpu4"})
    at.add("transport_jaccard", value=cell["haar_mean"], layer=L, type=t, bank_id="core32",
           hop_id=hop_id, control_id=MC.HAAR, model_id=model_id, model_sha=model_sha,
           n=cell["n_haar"], ci_lo=cell["haar_lo"], ci_hi=cell["haar_hi"],
           flags={**base, "predictor": "uniform_random_address"})
    at.add("transport_stability_self_jaccard", value=cell["stability"]["self_jaccard"], layer=L,
           type=t, hop_id=hop_id, control_id=MC.TREATMENT, model_id=model_id, model_sha=model_sha,
           flags={"k": k, "n_rows": cell["stability"]["n_rows"], "haar_hi": cell["haar_hi"],
                  "gate": "freeze A7 split-half", "excluded": excluded})


def _score_hop(at, hop_id, deltas, bank_of, traffic_of, k, seed, model_id, model_sha):
    """One hop at one k: every cell, the A7 exclusion gate, the paired CI, the shuffle null."""
    obs, tra, shuffle_cells = [], [], []
    n_cells = n_excluded = 0
    for (t, L), dW in sorted(deltas.items()):
        U = bank_of(t, L)
        cell = MT.transport_cell(U_parent=U, dW_hop=dW, traffic_score=traffic_of(t, L), k=k,
                                 seed=H.cell_seed(seed, ARM, hop_id, t, L), n_haar=N_HAAR)
        # Freeze A7: a cell whose own address does not survive a row-parity split cannot support a
        # transport claim in EITHER direction. Excluded and COUNTED, never silently dropped —
        # dropping it silently would manufacture TRAFFIC-ONLY, since an unstable cell contributes a
        # near-chance observed overlap that drags delta negative for instrumental reasons.
        excluded = bool(cell["stability"]["self_jaccard"] <= cell["haar_hi"])
        n_cells += 1
        n_excluded += int(excluded)
        _write_cell(at, hop_id, t, L, k, cell, excluded, model_id, model_sha)
        if not excluded:
            obs.append(cell["observed"]["jaccard"])
            tra.append(cell["traffic"]["jaccard"])
            shuffle_cells.append({"parent": MT.bank_address(U, k), "child": cell["hop_address"],
                                  "n_total": cell["n_total"]})

    kept = n_cells - n_excluded
    row = {"hop_id": hop_id, "k": int(k), "n_cells": n_cells, "n_excluded": n_excluded,
           "n_kept": kept, "void_instrument": bool(kept <= n_cells / 2)}
    if row["void_instrument"] or kept < 2:
        row.update({"clears_zero": False,
                    "reason": "VOID-INSTRUMENT: at most half the cells survived the A7 gate"})
        return row

    ci = MC.paired_delta_ci(obs, tra, seed=seed)
    shuffled = MT.shuffle_transport(shuffle_cells, seed=seed)
    perm = MC.permutation_pvalue(list(obs) + list(shuffled),
                                 ["observed"] * len(obs) + ["shuffled"] * len(shuffled), seed=seed)
    at.add_ci("transport_jaccard", shuffled, hop_id=hop_id, bank_id="core32",
              control_id=MC.SHUFFLE, model_id=model_id, model_sha=model_sha, seed=seed,
              flags={"k": k, "p_vs_observed": perm["p_value"], "perm_method": perm["method"],
                     "predictor": "parent_sink_of_a_DIFFERENT_cell",
                     "null": "derangement re-pairing of parent addresses with other cells' hops"})
    at.add_ci("transport_delta_vs_traffic", np.asarray(obs) - np.asarray(tra), hop_id=hop_id,
              bank_id="core32", control_id=MC.TREATMENT, model_id=model_id, model_sha=model_sha,
              seed=seed, flags={"k": k, "n_excluded": n_excluded, "rule": "CI clear of 0"})
    row.update({**ci, "shuffle_mean": float(np.mean(shuffled)),
                "shuffle_p_value": perm["p_value"], "shuffle_perm_method": perm["method"]})
    return row


def _score_ft_family(at, at_k, repos, energies, stability, g_all, gram_dir, traffic_of, cells,
                     seed, pg1_sha):
    """The 24 PG1 fine-tunes, each against a LEAVE-ONE-OUT parent bank (freeze A2).

    Counted as ONE hop for the >=2-hop rule — the 24 share a parent and are not independent of one
    another — but the spread across fine-tunes is reported, because a family whose members agree is
    a different object from one whose members disagree.

    One eigendecomposition per (fine-tune, cell), scoring BOTH k values from it. Doing the loop per
    k instead would double 2592 eigh of a 2048x2048 matrix for no new information.
    """
    acc = {k: {} for k in at_k}
    for i, repo in enumerate(repos):
        per_k = {k: {"obs": [], "tra": [], "excluded": 0} for k in at_k}
        for (t, L) in cells:
            g_loo = g_all[(t, L)] - np.load(gram_dir / f"{i}__{t}__L{L}.npy").astype(np.float64)
            U = np.ascontiguousarray(np.linalg.eigh(g_loo)[1][:, :H.BANK_K])
            n = int(g_loo.shape[0])
            traffic = traffic_of(t, L)
            for k in at_k:
                hop_addr = W0.top_k_coords(-energies[repo][(t, L)], k)
                haar_hi = max(W0.overlap_stats(
                    MT.haar_address(n, k, H.cell_seed(seed, "ftfam", repo, t, L) + j),
                    hop_addr, n)["jaccard"] for j in range(N_HAAR))
                if stability[repo][(t, L)][k]["self_jaccard"] <= haar_hi:
                    per_k[k]["excluded"] += 1
                    continue
                per_k[k]["obs"].append(
                    W0.overlap_stats(MT.bank_address(U, k), hop_addr, n)["jaccard"])
                per_k[k]["tra"].append(
                    W0.overlap_stats(MT.score_address(traffic, k), hop_addr, n)["jaccard"])
        for k in at_k:
            d = per_k[k]
            if len(d["obs"]) >= 2:
                acc[k][repo] = {**MC.paired_delta_ci(d["obs"], d["tra"], seed=seed),
                                "n_kept": len(d["obs"]), "n_excluded": d["excluded"]}
        print(f"[mn1/ft] {i + 1}/{len(repos)} {repo}", flush=True)

    rows = []
    for k in at_k:
        per_ft = acc[k]
        assert per_ft, f"the FT family kept no fine-tune with >=2 stable cells at k={k}"
        deltas = [p["delta_mean"] for p in per_ft.values()]
        # The family CI is over FINE-TUNES, not over cells: the unit that could have come out
        # differently is the fine-tune, and pooling cells would count 24x too many.
        fam = MC.paired_delta_ci(deltas, [0.0] * len(deltas), seed=seed)
        at.add_ci("transport_delta_vs_traffic", deltas, hop_id="pg1->ft_family", bank_id="core32",
                  control_id=MC.TREATMENT, model_id=WT.PG1_REPO, model_sha=pg1_sha, seed=seed,
                  flags={"k": k, "n_fine_tunes": len(per_ft), "parent_bank": "leave-one-out",
                         "why": "freeze A2 — the pooled bank contains this child's own ΔW",
                         "unit": "fine-tune"})
        rows.append({"hop_id": "pg1->ft_family", "k": int(k), "n_fine_tunes": len(per_ft),
                     "n_cells": len(cells), "void_instrument": False, **fam,
                     "per_fine_tune_delta_mean": {r: p["delta_mean"] for r, p in per_ft.items()},
                     "n_kept": int(sum(p["n_kept"] for p in per_ft.values())),
                     "n_excluded": int(sum(p["n_excluded"] for p in per_ft.values()))})
    return rows


def main():
    ap = argparse.ArgumentParser(description="MENDEL MN-1 — sink transport vs the traffic null")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-ft", type=int, default=24, help="fine-tunes in the PG1 family hop")
    ap.add_argument("--skip-ft-family", action="store_true",
                    help="the three backbone hops only (the family needs the LOO pass)")
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_mn1", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_mn1"), out)

    # ---- parents' banks and traffic ----
    pg1_bank_dir = WT.fetch_banks(H.cache_dir(out, "banks"))
    g4_vec = man.require("g4_vectors", f"{G4_PREFIX}/vectors", recursive=True)
    g4 = json.loads(man.require("g4_metrics", f"{G4_PREFIX}/metrics.json").read_text())
    assert g4["verdict"] == "REPRODUCED", (
        f"G4 verdict is {g4['verdict']!r} — MN-1 is gated on the preflight reproducing M-8's "
        f"activation-magnitude mean (spec §2: G4 gates MN-1)")
    g2b_npz = np.load(man.require("g2b_actstats", G2B_ACTSTATS))
    g2b_bank_dir = man.require("g2b_banks", G2B_BANKS, recursive=True)

    def pg1_traffic(t, L):
        return np.load(g4_vec / f"act_rms__{t}__L{L}.npy")

    def pg1_bank(t, L):
        return WT.load_bank(pg1_bank_dir, t, L, "core32")

    def g2b_bank(t, L):
        return WT.load_bank(g2b_bank_dir, t, L, "core32")

    def g2b_traffic(t, L):
        return _gemma2b_traffic(g2b_npz, t, L)

    layers = list(range(H.N_LAYERS))
    cells = [(t, L) for t in TYPES for L in layers]
    pg1_sd, pg1_sha = _lineage_decoder("PG1")

    # ---- the three backbone hops ----
    hops = {}
    g2b_sd, g2b_sha = _gemma2b_decoder()
    hops["g2b->pg1"] = dict(                       # q_proj only: freeze A4
        deltas={("q_proj", L): _delta(pg1_sd, g2b_sd, "q_proj", L) for L in layers},
        bank=g2b_bank, traffic=g2b_traffic, model_id=f"{G2B_REPO}->{WT.PG1_REPO}",
        model_sha=f"{g2b_sha[:12]}->{pg1_sha[:12]}")
    del g2b_sd

    for name, repo in (("pi0", "lerobot/pi0_base"), ("pi05", "lerobot/pi05_base")):
        child_sd, child_sha = _lineage_decoder(name)
        hops[f"pg1->{name}"] = dict(
            deltas={c: _delta(child_sd, pg1_sd, *c) for c in cells},
            bank=pg1_bank, traffic=pg1_traffic, model_id=f"{WT.PG1_REPO}->{repo}",
            model_sha=f"{pg1_sha[:12]}->{child_sha[:12]}")
        del child_sd
        print(f"[mn1] hop pg1->{name} differenced ({len(cells)} cells)", flush=True)

    results = {}
    for k in KS:
        for hop_id, h in hops.items():
            r = _score_hop(at, hop_id, h["deltas"], h["bank"], h["traffic"], k, a.seed,
                           h["model_id"], h["model_sha"])
            results[f"{hop_id}@k{k}"] = r
            print(f"[mn1] {hop_id} k={k}: " + json.dumps(
                {q: r[q] for q in ("n_kept", "n_excluded", "delta_mean", "ci_lo", "ci_hi",
                                   "clears_zero") if q in r}), flush=True)

    # ---- the PG1 -> 24 fine-tune family, leave-one-out (freeze A2) ----
    if not a.skip_ft_family:
        from huggingface_hub import HfApi
        repos = WT.ft_cloud(HfApi())[: a.n_ft]
        assert len(repos) >= 2, f"degenerate fine-tune cloud: {repos}"
        g_all = {c: np.zeros((H.IN_DIM[c[0]],) * 2, dtype=np.float64) for c in cells}
        energies, stability = {}, {}
        gram_dir = H.cache_dir(out, "ft_grams")
        for i, repo in enumerate(repos):
            sd, _sha = _ft_decoder(repo)
            energies[repo], stability[repo] = {}, {}
            for c in cells:
                dW = _delta(sd, pg1_sd, *c)
                g = dW.T @ dW / float((dW ** 2).sum())
                g_all[c] += g
                np.save(gram_dir / f"{i}__{c[0]}__L{c[1]}.npy", g.astype(np.float32))
                energies[repo][c] = MT.column_write_energy(dW)
                stability[repo][c] = {k: MT.split_half_stability(dW, k) for k in KS}
            del sd
            print(f"[mn1/gram] {i + 1}/{len(repos)} {repo}", flush=True)
        man.entries["ft_cloud"] = {"uri": "hf://google/paligemma-3b-ft-*-224", "kind": "roster",
                                   "n_files": len(repos), "bytes": 0, "sha256": "n/a",
                                   "repos": repos}
        for r in _score_ft_family(at, KS, repos, energies, stability, g_all, gram_dir,
                                  pg1_traffic, cells, a.seed, pg1_sha):
            results[f"pg1->ft_family@k{r['k']}"] = r

    # ---- the frozen rule, on the primary k ----
    k0 = KS[0]
    primary = {h: r for h, r in results.items()
               if h.endswith(f"@k{k0}") and "clears_zero" in r}
    assert primary, "no hop produced a decidable CI at the primary k"
    verdict_doc = MT.mn1_verdict(primary)
    void_hops = sorted(h for h, r in primary.items() if r.get("void_instrument"))
    verdict = MP.check_verdict(
        ARM, "VOID-INSTRUMENT" if len(void_hops) == len(primary) else verdict_doc["verdict"])

    gates.record("mn1_instrument_stable", not void_hops, void_hops=void_hops,
                 note="freeze A7: a hop losing >half its cells to the split-half gate is VOID")
    trinity = at.check_control_trinity(["transport_jaccard"])
    info = at.to_parquet()
    man.write()
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "queue_item": meta["item"],
        "rule": "INHERITED-BEYOND-TRAFFIC iff the paired CI clears 0 at >=2 independent hops",
        "k_primary": k0, "k_secondary": KS[1], "types": list(TYPES), "n_haar": N_HAAR,
        "seed": a.seed, "verdict_detail": verdict_doc, "control_trinity": trinity,
        "hops": results, "atlas": info,
        "traffic_null": ("parent activation RMS — M-8's named incumbent, mean Jaccard 0.4160 "
                         "against PG1 core32 in mercator_w0gpu4, which BEAT the predictor"),
        "excluded_types": ["down_proj"], "exclusion_reason": "freeze A3 (LOO eigh cost at 16384^2)",
        "g4_reference": g4.get("reproduction")})
    print("MN1_OK " + json.dumps({"verdict": verdict, "n_hops": len(primary),
                                  "void_hops": void_hops}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
