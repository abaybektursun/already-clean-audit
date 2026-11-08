"""BROCA pure core — restoration curves, null bands, band-finding, and the frozen verdict.

Spec: studies/lineage/cross-lineage/research/2026-09-02_broca_spec.md §5. Pure numpy; asserts;
run `python -m infra.pg.broca_core` for the dry-run gate.
"""
import numpy as np

CHANCE = 0.25


def restoration_fraction(acc_patched: float, acc_after: float, acc_before: float) -> float:
    """R = recovered fraction of the lost above-chance margin. Can exceed 1 or go <0; clipped
    only at reporting, never here (diagnostics need the raw value)."""
    lost = (acc_before - CHANCE) - (acc_after - CHANCE)
    assert lost > 1e-6, f"no lost margin to restore (before={acc_before}, after={acc_after})"
    return float((acc_patched - acc_after) / lost)


def null_band(null_Rs) -> float:
    """Band = max |R| over the mismatched-item null cuts (the achievable-null floor)."""
    a = np.asarray(list(null_Rs), dtype=float)
    assert a.size >= 3, "need >=3 null cuts"
    return float(np.abs(a).max())


def threshold(band: float) -> float:
    """Frozen: T = max(2x null band, 0.10)."""
    return float(max(2.0 * band, 0.10))


def transition(layers, Rs, T: float, hi: float = 0.6) -> dict:
    """Band-finder on a (layer, R) curve sorted by layer. Returns first layer above T, first
    layer with R>=hi sustained to the end (L*), and the band width in LAYER-INDEX units."""
    L = list(layers); R = list(Rs)
    assert len(L) == len(R) >= 4 and L == sorted(L)
    first_above = next((l for l, r in zip(L, R) if r > T), None)
    l_star = None
    for i in range(len(L)):
        if all(r >= hi for r in R[i:]):
            l_star = L[i]
            break
    width = (l_star - first_above) if (l_star is not None and first_above is not None) else None
    return {"first_above_null": first_above, "l_star": l_star, "band_width": width,
            "max_R": float(max(R))}


def broca_verdict(layers, Rs, T: float, n_layers: int) -> dict:
    """§5 frozen mapping. layers/Rs = the full (pass-1+2) prefix-graft curve."""
    tr = transition(layers, Rs, T)
    L = list(layers); R = list(Rs)
    early = [r for l, r in zip(L, R) if l < n_layers / 3]
    early_quiet = bool(early) and all(r <= T for r in early)
    spread = sum(1 for r in R if r > T)
    narrow = tr["band_width"] is not None and tr["band_width"] <= round(n_layers / 6) + 1
    if tr["l_star"] is not None and narrow and early_quiet:
        v = "H_reader (intact computation, localized late reader lesion)"
    elif tr["max_R"] <= T:
        v = "H_lost (nothing upstream restores — content consumed)"
    elif spread >= round(n_layers / 3) and tr["max_R"] > T:
        v = "H_distributed (gradual, wide restoration — damage spread through depth)"
    else:
        v = "MIXED (report components; no branch's signature)"
    return {"verdict": v, **tr, "early_quiet": early_quiet, "n_above_null": spread, "T": T}


def _dryrun():
    # restoration arithmetic
    assert abs(restoration_fraction(0.47, 0.281, 0.47) - 1.0) < 1e-9
    assert abs(restoration_fraction(0.281, 0.281, 0.47)) < 1e-9
    # bands + threshold floor
    b = null_band([0.02, -0.03, 0.01, 0.04]); assert abs(b - 0.04) < 1e-12
    assert threshold(0.02) == 0.10 and abs(threshold(0.08) - 0.16) < 1e-12
    n = 26; L = list(range(0, 26, 2))
    # H_reader: quiet early, sharp jump at 18, sustained
    R = [0.01, 0.02, 0.01, 0.03, 0.02, 0.02, 0.02, 0.05, 0.2, 0.75, 0.85, 0.9, 0.95]
    v = broca_verdict(L, R, T=0.10, n_layers=n); assert v["verdict"].startswith("H_reader"), v
    assert v["l_star"] == 18 and v["band_width"] == 2, v
    # H_lost: flat under T
    v = broca_verdict(L, [0.02] * 13, T=0.10, n_layers=n); assert v["verdict"].startswith("H_lost"), v
    # H_distributed: slow wide rise from early on
    Rd = [0.05, 0.12, 0.2, 0.28, 0.35, 0.42, 0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.75]
    v = broca_verdict(L, Rd, T=0.10, n_layers=n); assert v["verdict"].startswith("H_distributed"), v
    # MIXED: early leak + late jump (fails early_quiet)
    Rm = [0.3, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.2, 0.75, 0.85, 0.9, 0.95]
    v = broca_verdict(L, Rm, T=0.10, n_layers=n); assert v["verdict"].startswith("MIXED"), v
    print("BROCA core dry-run: ALL PASS")


if __name__ == "__main__":
    _dryrun()
