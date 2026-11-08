"""conserved-core Tier 2 — PURE per-subject policy-erasure analysis (no datasets, no GPU).

Re-analysis of the PCR banks (policy_content_readout_banks.json): is the deployed-policy content
erasure UNIFORM across the 57 MMLU subjects or STRUCTURED? Descriptive / exploratory (spec §3.2) — no
causal claim. The thin job (jobs/policy_subject_structure.py) re-derives the per-item subject labels by
replaying mmlu_cloze's sorted(test_by_subj) order over cais/mmlu and aligns to the banks via
per_item_hash; this module holds the frozen distance proxy + grouping + Spearman + bootstrap, all
weight-free and unit-tested.

FROZEN distance-from-grounded-task proxy (set before looking, §3.2): a subject is "spatial/embodied"
(proxy=1) if its name contains any SPATIAL_KEYWORD (natural science / math / engineering / medicine /
geography — physically grounded), else "abstract/verbal" (proxy=0; law / moral / history / philosophy /
politics / social). One specific operationalization; reported as exploratory.
"""
import numpy as np

SPATIAL_KEYWORDS = (
    "physics", "chemistry", "biology", "anatomy", "engineering", "mathematics", "algebra",
    "astronomy", "geography", "medicine", "medical", "clinical", "nutrition", "virology",
    "genetics", "computer", "machine_learning", "statistics", "econometrics",
)


def is_spatial(subject):
    """1 if the subject name names a physically-grounded / spatial domain, else 0 (abstract/verbal)."""
    s = subject.lower()
    return int(any(kw in s for kw in SPATIAL_KEYWORDS))


def per_subject_accuracy(subjects, per_item):
    """{subject: mean per_item} grouping a 0/1 per-item array by its aligned subject label."""
    subjects = list(subjects)
    per_item = np.asarray(per_item, dtype=np.float64)
    assert len(subjects) == per_item.shape[0], (len(subjects), per_item.shape)
    out = {}
    for subj in sorted(set(subjects)):
        mask = np.array([s == subj for s in subjects])
        out[subj] = float(per_item[mask].mean())
    return out


def _avg_rank(a):
    """Average (tie-corrected) 1-based ranks — for Spearman on a binary proxy with many ties."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(a.shape[0], dtype=np.float64)
    i = 0
    while i < a.shape[0]:
        j = i
        while j + 1 < a.shape[0] and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman ρ = Pearson of average ranks (pure numpy, tie-corrected)."""
    rx, ry = _avg_rank(x), _avg_rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx @ rx) * (ry @ ry))
    assert denom > 0, "degenerate ranks (constant variable)"
    return float((rx @ ry) / denom)


def gap_proxy_spearman(subjects, per_item_arm, per_item_floor, n_boot=2000, seed=1337):
    """The §3.2 readout: per-subject cloze−floor GAP vs the frozen spatial proxy, Spearman ρ + item-
    bootstrap 95% CI. subjects: per-item subject labels (len N). per_item_arm/floor: 0/1 per-item arrays.
    GAP(subject) = acc_arm(subject) − acc_floor(subject) (how far above the deletion floor the policy
    retains the content for that subject). Returns ρ, CI, per-subject gaps + proxy, and the uniform-vs-
    structured verdict (structured iff the ρ CI excludes 0)."""
    subjects = list(subjects)
    arm = np.asarray(per_item_arm, dtype=np.float64)
    floor = np.asarray(per_item_floor, dtype=np.float64)
    N = len(subjects)
    assert arm.shape[0] == floor.shape[0] == N, (arm.shape, floor.shape, N)
    subj_arr = np.asarray(subjects)
    uniq = sorted(set(subjects))
    proxy = {s: is_spatial(s) for s in uniq}

    def _gaps(idx):
        sa = subj_arr[idx]
        a, f = arm[idx], floor[idx]
        g, px = [], []
        for s in uniq:
            m = sa == s
            if m.sum() == 0:
                continue
            g.append(a[m].mean() - f[m].mean())
            px.append(proxy[s])
        return np.asarray(g), np.asarray(px)

    full = np.arange(N)
    g0, px0 = _gaps(full)
    rho = spearman(g0, px0)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        g, px = _gaps(rng.integers(0, N, N))
        if len(set(px.tolist())) < 2 or np.ptp(g) == 0:
            continue
        boots.append(spearman(g, px))
    boots = np.asarray(boots, dtype=np.float64)
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots.size else (float("nan"),) * 2
    gaps = {s: (float(np.mean(arm[subj_arr == s]) - np.mean(floor[subj_arr == s]))) for s in uniq}
    return {"rho": rho, "lo": lo, "hi": hi, "n_boot_used": int(boots.size),
            "structured": bool(boots.size and (lo > 0 or hi < 0)),
            "per_subject_gap": gaps, "proxy": proxy,
            "n_spatial": int(sum(proxy.values())), "n_verbal": int(len(uniq) - sum(proxy.values()))}
