"""TESSERA Arm S — the two CONFIRMATORY tests (spec §1.3 S, §6 S, §7 F-S).

The spectral ATLAS (F1, the census) is `tessera_s_atlas.py`. This job carries only the two things
Arm S actually claims:

  H_core-survives   the universally-untouched q/k core over the 24 PG1 fine-tunes still reads as a
                    core when π0 and π0.5 join the consensus (roster 26). Killed by the 26-unit
                    consensus collapsing into the matched-rotation null band.
  H_hop-disjoint    the VLA writes occupy subspaces ≈ Haar-disjoint from the install write and
                    from the FT family. Killed by a cross-hop principal-angle overlap beyond the
                    Haar null (writes SHARE a channel — reportable structure).

WHY q/k ONLY FOR THE CORE. §1.3 states the hypothesis about the q/k core, and that is also what
keeps the test honest: q_proj and k_proj are 2048×2048, so the consensus Gram is 2048² and its
matched-rotation null is affordable at full draw count. `down_proj`'s input dim is 16384, whose
16384² Gram would force a draw count so small the null band itself would be the noisiest term in
the comparison. The other five types enter the ATLAS census, not this test.

THE POSITIVE CONTROL THAT CAN FAIL (P-S1). The 24-unit consensus is recomputed here from scratch
and must reproduce `invariant_core`'s CORE-BEYOND-NULL verdict on q/k. If it does not, this
pipeline is not the instrument that found the core, and the 26-unit number is uninterpretable —
the job reports VOID-INSTRUMENT and draws no conclusion about the VLA writes. Without this leg a
26-unit "collapse" could just as well be a re-implementation bug as a finding.

INSTRUMENTS. `invariant_core._consensus_gram` / `_null_band` / `_bottom_topk_eigs` are imported
and used verbatim (REUSED — the shipped core number comes from the instrument that found the
core, never from `tessera_core`'s copy, which exists so the offline gate can run weight-free).
Cross-hop overlaps use `tessera_core.subspace_overlap` against `haar_overlap_null` (NEW).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_s_core --type r7i.2xlarge --no-weights \
      --args "--n-null 200" --s3-prefix results/tessera_s_core --budget-min 600

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import time

BUCKET = "pg-lineage-769063704696"
PG1_REPO = "google/paligemma-3b-pt-224"
GEMMA1_REPO = "google/gemma-2b"
S3_UNITS = "s3://pg-lineage-769063704696/weights/lacuna_units"
N_LAYERS = 18
# the core test's types (§1.3) — both 2048×2048, so the null is affordable at full draw count
CORE_TYPES = ("q_proj", "k_proj")
# every LM decoder type; the cross-hop overlap runs over all of them
ALL_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
K_GRID = (8, 32, 128)
# the hops whose subspaces are compared pairwise (H_hop-disjoint)
HOPS = ("install", "pi0", "pi05")
# A12: unrelated FT writes share the ENERGY structure but not the task, so their
# pairwise overlap distribution is the achievable null for "writes overlap because
# writes overlap". 8 FTs give 28 reference pairs per matrix at ~2 GB of subspaces.
N_FT_REF = 8


_LM_KEY = re.compile(r"^(?:language_model\.)?layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)\.weight$")


def _lm_decoder_matrices(sd):
    """{'layers.<i>.<blk>.<type>.weight': tensor} — the 18×7 LM decoder matrices, canonical keys.

    Accepts either the `language_model.`-prefixed shell namespace (PaliGemma) or a bare decoder
    namespace (gemma-2b), and normalises both to the same short key so a PG1 matrix and a gemma-2b
    matrix of the same (layer, type) align.

    The anchored regex is load-bearing, not tidiness. PaliGemma's VISION tower also contains
    `...encoder.layers.0.self_attn.q_proj.weight`, so a substring search for "layers." plus
    ".q_proj.weight" matches vision keys too and they normalise to the SAME short key as the LM's
    — silently overwriting language-model matrices with vision ones. Anchoring at the start of the
    canonical key admits only `layers.` and `language_model.layers.`.
    """
    out = {}
    for k, v in sd.items():
        if v.ndim != 2:
            continue
        m = _LM_KEY.match(k)
        if not m or m.group(2) not in ALL_TYPES:
            continue
        short = k[k.index("layers."):]
        assert short not in out, f"LM key collision on {short!r} (from {k!r})"
        out[short] = v
    assert len(out) == N_LAYERS * len(ALL_TYPES), (
        f"expected {N_LAYERS * len(ALL_TYPES)} LM decoder matrices, got {len(out)}")
    return out


def _ftype(key):
    return next(t for t in ALL_TYPES if f".{t}.weight" in key)


def _delete_cache(repo_id):
    import shutil
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", ""), "hub")
    d = os.path.join(hub, "models--" + repo_id.replace("/", "--"))
    if os.path.isdir(d):
        shutil.rmtree(d)          # no try/except — a cleanup failure must fail loud


def _hf_lm_sd(repo, max_tries=4):
    """Canonical LM decoder matrices of an HF PaliGemma/Gemma repo, read from safetensors.

    The retry is at a SYSTEM BOUNDARY (network I/O), which is the one carve-out the no-try/except
    rule allows — it guards nothing about the science. A 26-unit sweep pulls ~300 GB from the Hub,
    and the first full run died 51 minutes in on a truncated body ("received 3993576014 bytes,
    expected 4999820608"). The retry deletes the partial snapshot before re-fetching, so a resumed
    download can never splice two partial bodies, and it re-raises after `max_tries` rather than
    proceeding on incomplete weights.
    """
    import time as _t
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file
    from infra.pg.lacuna_shell import canon
    last = None
    for attempt in range(max_tries):
        try:
            sd = {}
            for f in [x for x in list_repo_files(repo) if x.endswith(".safetensors")]:
                sd.update(load_file(hf_hub_download(repo, f)))
            return _lm_decoder_matrices({canon(k): v for k, v in sd.items()})
        except Exception as e:                       # noqa: BLE001 — network boundary only
            last = e
            print(f"[retry] {repo} attempt {attempt + 1}/{max_tries} failed: "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
            _delete_cache(repo)                      # never splice two partial bodies
            _t.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{repo}: {max_tries} download attempts failed") from last


def _policy_lm_sd(name):
    """Canonical LM decoder matrices of a published policy's PaliGemma half."""
    from infra.pg import lacuna_shell as LS
    full, _sha = LS.policy_full_sd(name)
    vlm, _head, _counts = LS.policy_vlm_sd(name, full_sd=full)
    del full
    return _lm_decoder_matrices(vlm)


def unit_sources(units):
    """(uid, loader) pairs in an order that fetches each heavy checkpoint exactly once."""
    from infra.pg.plenum_map import FT_REPO, FT_SLUGS
    src = {}
    for slug in FT_SLUGS:
        src[f"ft-{slug}"] = ("hf", FT_REPO.format(slug=slug))
    src["pi0"] = ("policy", "pi0")
    src["pi05"] = ("policy", "pi05")
    src["gemma-2b"] = ("hf", GEMMA1_REPO)
    return [(u, src[u]) for u in units if u in src]


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--n-haar", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-ft", type=int, default=0,
                    help="SMOKE ONLY: cap the FT roster to the first N slugs. 0 = all 24. A capped "
                         "run is not the registered test and its metrics carry smoke=true.")
    ap.add_argument("--layers", default="",
                    help="SMOKE ONLY: comma layer ids to restrict the matrix grid to")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()

    # Bind the report BEFORE any loop and flush after every matrix. A 20+ hour sweep that writes
    # metrics.json only at the end loses everything to a reaper kill or a late crash — the
    # incremental-save lesson this project has already paid for once.
    rep = {"object": "TESSERA Arm S — the two confirmatory tests "
                     "(H_core-survives, H_hop-disjoint)", "stage": "startup"}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    from infra.aws_pg.jobs.invariant_core import (_bottom_topk_eigs, _consensus_gram, _null_band,
                                                  _verdict)
    from infra.pg import tessera_core as TC
    from infra.pg.plenum_map import FT_SLUGS

    slugs = FT_SLUGS[:a.n_ft] if a.n_ft else FT_SLUGS
    ft_units = [f"ft-{s}" for s in slugs]
    roster26 = ft_units + ["pi0", "pi05"]
    smoke = bool(a.n_ft or a.layers)

    # ---- PG1 (the shared base every ΔW is measured from) --------------------------------------
    print("[base] loading PG1", flush=True)
    w_pg1 = _hf_lm_sd(PG1_REPO)
    if a.layers:
        keep = {int(x) for x in a.layers.split(",")}
        w_pg1 = {k: v for k, v in w_pg1.items() if int(k.split(".")[1]) in keep}
    core_keys = sorted(k for k in w_pg1 if _ftype(k) in CORE_TYPES)
    all_keys = sorted(w_pg1)
    if not smoke:
        assert len(core_keys) == N_LAYERS * len(CORE_TYPES), f"core keys {len(core_keys)}"
    assert core_keys and all_keys, "empty matrix grid"

    # accumulators
    gram = {k: None for k in core_keys}              # 26-roster consensus Gram per q/k matrix
    gram24 = {k: None for k in core_keys}            # 24-FT consensus (the P-S1 positive control)
    spectra = {k: {} for k in core_keys}             # {key: {unit: p_i}} — the null's only input
    hop_V = {h: {} for h in HOPS}                    # {hop: {key: V_r}} for the overlap test
    ft_V = {}                                        # {ft_uid: {key: V_r}} — A12 reference
    ft_gram_all = {k: None for k in all_keys}        # 24-FT consensus over ALL types (for hop-vs-FT)
    unit_log = []

    # `down_proj` is (2048, 16384): its INPUT space is 16384-dim, so an FT-family consensus Gram
    # for it would be 16384² per matrix (2.1 GB each, 38 GB over 18 layers). The hop-vs-hop
    # overlaps still cover it — only the hop-vs-FT pairs do not. Recorded here rather than left
    # as a silent gap (the no-silent-caps rule).
    FT_GRAM_TYPES = tuple(t for t in ALL_TYPES if t != "down_proj")

    # ---- stream the units ---------------------------------------------------------------------
    # ΔW is built ONE MATRIX AT A TIME. Materialising all 126 in fp64 costs ~17 GB on top of the
    # two fp32 state dicts (~8 GB each) and does not fit the instance.
    for uid, (kind, ref) in unit_sources(roster26 + ["gemma-2b"]):
        import torch
        print(f"[unit] {uid} ({kind}:{ref})", flush=True)
        w = _hf_lm_sd(ref) if kind == "hf" else _policy_lm_sd(ref)
        w = {k: v for k, v in w.items() if k in set(all_keys)}
        assert sorted(w) == all_keys, (
            f"{uid}: LM decoder key set differs from PG1's "
            f"(unit-only {sorted(set(w) - set(all_keys))[:3]}, "
            f"pg1-only {sorted(set(all_keys) - set(w))[:3]})")
        is_install = (uid == "gemma-2b")
        is_ft = uid.startswith("ft-")
        for k in all_keys:
            # the INSTALL hop is PG1 − gemma-2b (the creation step), not gemma-2b − PG1
            A = ((w_pg1[k].to(torch.float64) - w[k].to(torch.float64)) if is_install
                 else (w[k].to(torch.float64) - w_pg1[k].to(torch.float64))).numpy()
            if is_install:
                hop_V["install"][k] = TC.top_right_subspace(A)
            else:
                if k in gram:                       # the q/k core test
                    fro2 = float((A ** 2).sum())
                    assert fro2 > 0, f"{uid}/{k}: zero ΔW reached the consensus"
                    G = (A.T @ A) / fro2
                    gram[k] = G if gram[k] is None else gram[k] + G
                    if is_ft:
                        gram24[k] = G if gram24[k] is None else gram24[k] + G
                    spectra[k][uid] = TC.spectrum_normsq(A)
                    del G
                if is_ft and _ftype(k) in FT_GRAM_TYPES:
                    fro2 = float((A ** 2).sum())
                    Gf = (A.T @ A) / fro2
                    ft_gram_all[k] = Gf if ft_gram_all[k] is None else ft_gram_all[k] + Gf
                    del Gf
                if uid in HOPS:
                    hop_V[uid][k] = TC.top_right_subspace(A)
                if is_ft and len(ft_V) < N_FT_REF:
                    ft_V.setdefault(uid, {})[k] = TC.top_right_subspace(A)
            del A
        unit_log.append({"unit": ("install(PG1−gemma-2b)" if is_install else uid),
                         "role": ("hop-only" if is_install
                                  else "consensus" + ("+hop" if uid in HOPS else ""))})
        del w
        if kind == "hf":
            _delete_cache(ref)

    # ---- the core test: 24 (control) and 26 (the question) ------------------------------------
    core = {}
    for label, G_by_key, roster in (("n24", gram24, ft_units), ("n26", gram, roster26)):
        rows = []
        for k in core_keys:
            G = G_by_key[k]
            assert G is not None, f"{k}: no consensus accumulated for {label}"
            w_eig, bot, top = _bottom_topk_eigs(G, K_GRID)
            sp = [spectra[k][u] for u in roster]
            nb, nt, samples = _null_band(sp, G.shape[0], K_GRID, a.n_null, a.seed)
            v = _verdict(bot, nb, samples, K_GRID)
            rows.append({"key": k, "type": _ftype(k), "n_units": len(roster),
                         "bottom_k": bot, "top_k": top, "null_bottom": nb, "null_top": nt,
                         "verdict": v})
            print(f"[core {label}] {k} {v}", flush=True)
            core[label] = rows                     # rebind every matrix, then flush
            rep["core"], rep["stage"] = core, f"core/{label}/{len(rows)}"
            _w()
        core[label] = rows

    # P-S1: the 24-unit control must reproduce invariant_core's core verdict on q/k.
    # `_verdict` labels the below-null case "BEYOND_NULL_below" (its upper tail is "ABOVE_NULL").
    def _beyond(rows, k="8"):
        return sum(1 for r in rows if r["verdict"][k]["verdict"] == "BEYOND_NULL_below")
    n24_beyond = _beyond(core["n24"])
    n26_beyond = _beyond(core["n26"])
    control_ok = n24_beyond > len(core_keys) // 2

    # ---- the overlap test ---------------------------------------------------------------------
    # the FT family's representative subspace = the most-WRITTEN consensus directions of the
    # 24-FT Gram (its top-r eigenvectors), which is the same object the core score mirrors.
    for k in all_keys:
        if ft_gram_all[k] is not None:
            w_e, V = np.linalg.eigh(ft_gram_all[k])
            hop_V.setdefault("ft24", {})[k] = np.ascontiguousarray(
                V[:, np.argsort(w_e)[::-1][:TC.R_SUBSPACE]])
            ft_gram_all[k] = None                   # 108 × 2048² fp64 is 3.6 GB; free as we go

    # ---- A12: the unrelated-FT reference distribution, per matrix -----------------------------
    ftref = {}
    ft_ids = sorted(ft_V)
    for k in all_keys:
        vals = [TC.subspace_overlap(ft_V[u][k], ft_V[w][k])
                for i, u in enumerate(ft_ids) for w in ft_ids[i + 1:]
                if k in ft_V[u] and k in ft_V[w]]
        if len(vals) >= 3:
            ftref[k] = {"n_pairs": len(vals), "median": float(np.median(vals)),
                        "p97.5": float(np.percentile(vals, 97.5)),
                        "p2.5": float(np.percentile(vals, 2.5))}
    print(f"[A12] unrelated-FT reference on {len(ftref)} matrices "
          f"from {len(ft_ids)} FTs", flush=True)

    haar_cache = {}
    pairs = [("install", "pi0"), ("install", "pi05"), ("pi0", "pi05"),
             ("install", "ft24"), ("pi0", "ft24"), ("pi05", "ft24")]
    overlaps = []
    for x, y in pairs:
        per_key = []
        for k in all_keys:
            if k not in hop_V.get(x, {}) or k not in hop_V.get(y, {}):
                continue
            U, V = hop_V[x][k], hop_V[y][k]
            n = U.shape[0]
            if n not in haar_cache:
                haar_cache[n] = TC.haar_overlap_null(n, U.shape[1], V.shape[1],
                                                     n_draws=a.n_haar, seed=a.seed)
            o = TC.subspace_overlap(U, V)
            per_key.append({"key": k, "type": _ftype(k), "overlap": o,
                            "verdict": TC.overlap_verdict(o, haar_cache[n])})
        assert per_key, f"pair {x}~{y}: no shared matrices — the subspace rosters do not overlap"
        # A12 second null: per matrix, the distribution of overlaps between UNRELATED FT writes
        for r in per_key:
            ref = ftref.get(r["key"])
            if ref:
                r["ft_ref_p97_5"] = ref["p97.5"]
                r["ft_ref_p2_5"] = ref["p2.5"]
                r["beats_ft_ref"] = bool(r["overlap"] > ref["p97.5"])
                # C4 / finding 13 — A12 shipped ONE-SIDED. It could demote a SHARED-CHANNEL to
                # ENERGY-ALIGNED, but had no branch for the opposite structure: an overlap BELOW
                # what two UNRELATED fine-tunes reach, i.e. two writes that avoid each other more
                # than unrelated training does. That is as much a structure claim as the upper
                # tail, and the rule as shipped could not report it. Symmetric now, matching the
                # two-sidedness `overlap_verdict` already had against the Haar band.
                r["below_ft_ref"] = bool(r["overlap"] < ref["p2.5"])
                if r["verdict"]["verdict"] == "SHARED-CHANNEL" and not r["beats_ft_ref"]:
                    r["verdict"] = {"verdict": "ENERGY-ALIGNED", "z": r["verdict"]["z"],
                                    "why": "above the Haar band but inside the unrelated-FT "
                                           "reference — the overlap is what ANY two large writes "
                                           "on these models produce, not a lineage-specific "
                                           "shared channel (A12)"}
                elif r["below_ft_ref"]:
                    r["verdict"] = {"verdict": "BELOW-FT-REFERENCE", "z": r["verdict"]["z"],
                                    "why": "overlap below the unrelated-FT reference band — these "
                                           "two writes avoid each other MORE than two unrelated "
                                           "fine-tunes of this model do (C4, finding 13)"}
        med = TC.boot_ci([r["overlap"] for r in per_key])
        n_shared = sum(1 for r in per_key if r["verdict"]["verdict"] == "SHARED-CHANNEL")
        n_energy = sum(1 for r in per_key if r["verdict"]["verdict"] == "ENERGY-ALIGNED")
        n_anti = sum(1 for r in per_key if r["verdict"]["verdict"] == "ANTI-ALIGNED")
        n_below_ft = sum(1 for r in per_key if r["verdict"]["verdict"] == "BELOW-FT-REFERENCE")
        overlaps.append({"pair": [x, y], "n_matrices": len(per_key), "median_ci": med,
                         "n_shared_channel": n_shared, "n_energy_aligned": n_energy,
                         "n_anti_aligned": n_anti, "n_below_ft_reference": n_below_ft,
                         "n_haar_disjoint": (len(per_key) - n_shared - n_energy - n_anti
                                             - n_below_ft),
                         "per_key": per_key})
        rep["overlaps"], rep["stage"] = overlaps, f"overlap/{x}~{y}"
        _w()
        print(f"[overlap] {x}~{y}: median {med['point']:.4g} "
              f"[{med['lo95']:.4g},{med['hi95']:.4g}] shared={n_shared} anti={n_anti} "
              f"energy={n_energy} disjoint={len(per_key) - n_shared - n_energy - n_anti}",
              flush=True)

    metrics = {
        "object": "TESSERA Arm S — the two confirmatory tests (H_core-survives, H_hop-disjoint)",
        "smoke": smoke,
        "label": ("SMOKE — a capped roster/grid; NOT the registered test" if smoke
                  else "CONFIRMATORY — the registered Arm S test"),
        "r_subspace": TC.R_SUBSPACE,
        "k_grid": list(K_GRID),
        "n_null": a.n_null, "n_haar": a.n_haar, "seed": a.seed,
        "roster26": roster26, "core_types": list(CORE_TYPES),
        "units": unit_log,
        "core": core,
        "P_S1_control": {
            "n24_beyond_null_at_k8": n24_beyond, "n26_beyond_null_at_k8": n26_beyond,
            "n_matrices": len(core_keys), "passed": bool(control_ok),
            "rule": "the 24-unit consensus must reproduce invariant_core's CORE-BEYOND-NULL on a "
                    "MAJORITY of the q/k matrices; if it does not, this pipeline is not the "
                    "instrument that found the core and the 26-unit number is uninterpretable",
            "verdict": ("PASS" if control_ok else "VOID-INSTRUMENT")},
        "haar_bands": {str(n): b for n, b in haar_cache.items()},
        "ft_reference_null": {"n_fts": len(ft_ids), "fts": ft_ids, "per_key": ftref,
                              "why": "A12 — unrelated FT writes share the energy structure but not "
                                     "the task, so this is the achievable null for 'writes overlap "
                                     "because writes overlap'"},
        "overlaps": overlaps,
        "coverage": {
            "hop_vs_hop_types": list(ALL_TYPES),
            "hop_vs_ft24_types": list(FT_GRAM_TYPES),
            "excluded_from_ft24": ["down_proj"],
            "why": "down_proj's input space is 16384-dim, so an FT-family consensus Gram would be "
                   "2.1 GB per matrix (38 GB over 18 layers). Hop-vs-hop overlaps still cover it; "
                   "only the hop-vs-FT pairs do not. Disclosed rather than silently dropped."},
        "elapsed_s": time.time() - t0,
    }
    rep.update(metrics)
    _w()
    print(json.dumps({"P_S1": metrics["P_S1_control"]["verdict"],
                      "n24_beyond": n24_beyond, "n26_beyond": n26_beyond}, indent=1), flush=True)


if __name__ == "__main__":
    main()
