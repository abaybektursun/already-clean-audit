"""GAUGE — shared spine (pure, importable, weight-free).

The GAUGE program (studies/lineage/paligemma1-pi0/research/2026-07-04_gauge_program_spec.md)
measures the 24-FT PG1 ΔW family with instruments whose invariance groups are LARGER than the
program's default energy-ranked SVD. Every probe (gauge_cs/fix/rmt/tree/fun/dist) shares:

  * the SAME weight-matrix scope (18 layers × 7 decoder projection types — Gemma-1 / `gemma-2b`);
  * the SAME deterministic discovery/confirmation split of the 24 fine-tunes;
  * the SAME generation firewall (18-layer asserts everywhere; Gemma-2 never crosses).

This module holds ONLY the pure, model-free helpers those probes agree on. Heavy imports
(numpy) live inside functions so the module py_compiles on the control plane (house style:
infra/pg/relaw.py, infra/pg/reuse_estimators.py). No try/except — a wrong assumption RAISES.

Gate-0 provenance note (2026-07-04). The spec §14 said the 24-FT repo+SHA manifest lives in
`2026-06-13_invariant_core_metrics.json`. It does NOT: that file is the phase-1 SUMMARY
(`fine_tunes` is the string "24 google/paligemma-3b-ft-*-224"). The roster of 24 repo-ids lives in
`2026-06-02_n5c_cloud.json::members` (with `task_distance` near/far labels reused by P4); per-FT
SHAs are pinned LIVE via HfApi exactly as `invariant_core.py::main` does. The 12/12 split is a
deterministic sha1(repo_id), needing only repo-ids — so no banked SHA list is required. The job
layer (gauge_* jobs) does the live enumerate+pin; this pure module does the roster→split.
"""
from __future__ import annotations

import hashlib
import re

# The Gemma-1 decoder weight-matrix projection types — the unit of analysis across the whole
# program (18 layers × 7 = 126 matrices). Identical set to invariant_core.py / reuse_estimators.py;
# the COUNT (18 layers) is what the firewall asserts, never the generation-2 26.
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
N_LAYERS = 18                                   # Gemma-1 (paligemma-3b-pt-224 language_model)
N_DISCOVERY = 12                                # 12/12 split of the 24-FT family (§2)


def matrix_type(key: str) -> str:
    """The projection type of a decoder weight key ('...q_proj.weight' → 'q_proj'). RAISES if the
    key is not one of the 7 (never guesses)."""
    for t in PROJ:
        if t in key:
            return t
    raise ValueError(f"key {key!r} matches none of the 7 projection types {PROJ}")


def layer_index(key: str) -> int:
    """The decoder layer index of a key ('...layers.7.self_attn.q_proj.weight' → 7). RAISES if
    absent — a matrix outside the 'layers.<N>' decoder scope must fail loud, not map to -1."""
    m = re.search(r"layers\.(\d+)\.", key)
    assert m is not None, f"no decoder layer index in key {key!r}"
    return int(m.group(1))


def assert_gemma1(keys) -> int:
    """Generation firewall (§2). Count distinct decoder layer indices in a key list; assert ==18
    (Gemma-1). Returns the count. Mirrors relaw.n_decoder_layers but asserts the GAUGE invariant so
    a Gemma-2 (26L) weight cannot silently enter a GAUGE statistic."""
    idx = {layer_index(k) for k in keys if re.search(r"layers\.(\d+)\.", k)}
    assert idx, f"no decoder layer indices in {len(list(keys))} keys"
    n = len(idx)
    assert n == N_LAYERS, f"GAUGE firewall: expected {N_LAYERS}-layer Gemma-1 decoder, got {n}"
    return n


def sha1_split(repo_ids, n_discovery: int = N_DISCOVERY):
    """Deterministic discovery/confirmation split of the fine-tune roster (§2 'sha1-split 12/12 by
    FT repo-id'). Sort each repo-id by the hex sha1 of its own id; the first `n_discovery` are the
    discovery half, the rest are the sealed confirmation half. Depends ONLY on the repo-id strings,
    so it is reproducible without any banked SHA. Returns (discovery_sorted, confirmation_sorted),
    each a list of repo-ids sorted alphabetically for stable downstream iteration.

    RAISES on duplicate repo-ids or if the roster is smaller than n_discovery (never silently
    truncates the split)."""
    repos = list(repo_ids)
    assert len(set(repos)) == len(repos), "duplicate repo-id in roster"
    assert len(repos) >= n_discovery, f"roster {len(repos)} < n_discovery {n_discovery}"

    def _h(r):
        return hashlib.sha1(r.encode("utf-8")).hexdigest()

    by_hash = sorted(repos, key=_h)
    discovery = sorted(by_hash[:n_discovery])
    confirmation = sorted(by_hash[n_discovery:])
    return discovery, confirmation


def gini(x) -> float:
    """Gini coefficient of a NON-NEGATIVE vector (the P1/P4 concentration primitive). 0 = perfectly
    uniform, →1 = all mass on one entry. Sorted-order formula, O(n log n). RAISES on any negative
    entry (Gini is only defined for non-negative magnitudes — squared coefficients here). A zero
    vector returns 0.0 (degenerate-uniform, disclosed by the caller)."""
    import numpy as np
    a = np.asarray(x, dtype=np.float64).ravel()
    assert (a >= 0).all(), "gini requires non-negative entries"
    n = a.size
    assert n > 0, "empty vector"
    s = a.sum()
    if s == 0.0:
        return 0.0
    a = np.sort(a)
    idx = np.arange(1, n + 1, dtype=np.float64)
    # G = (2 Σ i·a_i)/(n Σ a_i) − (n+1)/n   (standard sorted-order Gini).
    return float((2.0 * (idx * a).sum()) / (n * s) - (n + 1.0) / n)


def participation_ratio(x) -> float:
    """Participation ratio (Σ a)² / Σ a² of a NON-NEGATIVE vector ∈ [1, n]; the P1 secondary
    concentration statistic (must agree in SIGN of excess with Gini for 'robust'). Low ⇒
    concentrated, high ⇒ spread. Numpy twin of weight_diff._participation_ratio (which is torch).
    A zero vector returns 0.0."""
    import numpy as np
    a = np.asarray(x, dtype=np.float64).ravel()
    assert (a >= 0).all(), "participation_ratio requires non-negative entries"
    num = float(a.sum()) ** 2
    den = float((a * a).sum())
    if den == 0.0:
        return 0.0
    return num / den
