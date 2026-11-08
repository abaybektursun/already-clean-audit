"""Weight-free tests for the DARWIN figure suite (§10).

Run: PYTHONPATH=. python3 infra/pg/tests/test_darwin_figures.py

The figures are an artifact the director reads, so the properties tested here are the ones that
would silently mislead a reader: a missing control curve, a missing CI band, a missing provenance
footer, a figure that renders from an empty frame.
"""
import json
import pathlib
import tempfile

import numpy as np
import pandas as pd

from infra.pg import darwin_core as D
from infra.pg import darwin_figures as F

F.use_agg()

BANKS = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
         "haar32_s0", "haar32_s1", "haar32_s2")


def _frame(n_layers=6, with_ci=True, banks=BANKS):
    rows = []
    rng = np.random.default_rng(0)
    for b in banks:
        base = 0.4 if b == "core32" else 0.02
        for L in range(n_layers):
            v = base + 0.01 * L + rng.normal(0, 0.002)
            rows.append({"arm": "W0-1a", "statistic": "leverage_spike_count", "value": v,
                         "layer": L, "type": "q_proj", "bank_id": b,
                         "ci_lo": v - 0.01 if with_ci else None,
                         "ci_hi": v + 0.01 if with_ci else None,
                         "run_id": "darwin_w0_1a", "commit_sha": "abc123def456",
                         "register": "ATLAS", "wave": "W0"})
    return pd.DataFrame(rows)


def test_sweep_draws_every_bank_and_writes_a_file():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "f.png"
        drawn = F.bank_sweep(_frame(), x="layer", y="value", title="t", xlabel="layer",
                             ylabel="v", out_path=p, reference=32 / 2048,
                             reference_label="Haar k/n")
        assert drawn == set(BANKS)
        assert p.exists() and p.stat().st_size > 5000        # a real raster, not a stub
        F.assert_controls_drawn(drawn, figure="f.png")


def test_missing_control_curves_fail_loud():
    """The defect this guards: a figure that plots the treatment against Haar alone."""
    with tempfile.TemporaryDirectory() as td:
        thin = _frame(banks=("core32", "haar32_s0", "haar32_s1", "haar32_s2"))
        drawn = F.bank_sweep(thin, x="layer", y="value", title="t", xlabel="l", ylabel="v",
                             out_path=pathlib.Path(td) / "thin.png")
        try:
            F.assert_controls_drawn(drawn, figure="thin.png")
            raise SystemExit("a Haar-only figure must fail loud")
        except AssertionError as ex:
            assert "topsigma32" in str(ex) and "control curves" in str(ex)


def test_sweep_without_ci_columns_still_renders():
    """Not every statistic is a sampled one; a point-valued sweep must not crash the suite."""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "noci.png"
        F.bank_sweep(_frame(with_ci=False), x="layer", y="value", title="t", xlabel="l",
                     ylabel="v", out_path=p)
        assert p.exists()


def test_bars_orders_treatment_first_and_plots_all_points():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "b.png"
        drawn = F.bank_bars(_frame(), y="value", title="t", ylabel="v", out_path=p)
        assert drawn == set(BANKS) and p.exists()
        F.assert_controls_drawn(drawn, figure="b.png")


def test_heatmap_counts_filled_cells():
    rows = []
    for t in ("q_proj", "k_proj"):
        for L in range(4):
            rows.append({"type": t, "layer": L, "value": 1e-8 * (L + 1),
                         "run_id": "r", "commit_sha": "sha"})
    with tempfile.TemporaryDirectory() as td:
        info = F.layer_type_heatmap(pd.DataFrame(rows), value="value", title="t",
                                    out_path=pathlib.Path(td) / "h.png")
        assert info == {"n_types": 2, "n_layers": 4, "n_filled": 8}


def test_empty_frames_fail_loud_rather_than_producing_a_blank_figure():
    empty = _frame().iloc[0:0]
    with tempfile.TemporaryDirectory() as td:
        for fn, kw in ((F.bank_sweep, dict(x="layer", y="value", xlabel="l", ylabel="v")),
                       (F.bank_bars, dict(y="value", ylabel="v"))):
            try:
                fn(empty, title="t", out_path=pathlib.Path(td) / "e.png", **kw)
                raise SystemExit("an empty frame must fail loud, not render a blank axes")
            except AssertionError as ex:
                assert "empty frame" in str(ex)


def test_provenance_footer_carries_runs_and_sha():
    import matplotlib.pyplot as plt
    fig, _ax = plt.subplots()
    F.provenance_footer(fig, _frame(), extra="extra-token")
    texts = [t.get_text() for t in fig.texts]
    plt.close(fig)
    assert len(texts) == 1
    assert "n_rows=48" in texts[0] and "darwin_w0_1a" in texts[0]
    assert "abc123def4" in texts[0] and "extra-token" in texts[0]


def test_every_style_key_is_a_known_bank_id():
    """A style table that drifts from the bank battery silently greys out a real control."""
    assert set(F.STYLE) == set(D.BANK_IDS)
    for bid, st in F.STYLE.items():
        assert set(st) == {"color", "lw", "z", "marker", "ms"}, bid
    # Treatments must sit above their controls in z-order, or the controls occlude them.
    assert min(F.STYLE[b]["z"] for b in F.REAL_BANKS) > max(F.STYLE[b]["z"] for b in F.HAAR)


def test_manifest_round_trips():
    with tempfile.TemporaryDirectory() as td:
        p = F.manifest({"a.png": {"kind": "sweep", "banks": ["core32"]}}, td)
        assert json.loads(pathlib.Path(p).read_text())["a.png"]["kind"] == "sweep"


def test_non_bank_series_get_distinct_stable_colours():
    """The first W1-N4 contrast figure rendered six arms in one dark colour, because none of them
    is a bank id and every fallback was identical."""
    arms = ["a_uniform_int4", "b_int4_fp16_outlier_cols", "c_int4_fp16_core_cols",
            "d_ceq_routed", "e_ceq_rotated_block", "f_vault_reuse"]
    smap = F.style_map(arms)
    styles = [smap[a] for a in arms]
    assert len({s["color"] for s in styles}) == len(arms), "arm colours collide"
    assert len({s["marker"] for s in styles}) == len(arms), "arm markers collide"
    assert all(s["color"] != F.FALLBACK["color"] for s in styles)
    # Stable across calls, so the same arm keeps its colour across figures and runs.
    assert [F.style_map(arms)[a]["color"] for a in arms] == [s["color"] for s in styles]
    # Real bank ids still take their fixed role-based style.
    assert F.style_map(["core32", "d_ceq_routed"])["core32"] is F.STYLE["core32"]
    # Overflowing the palette must fail loud rather than silently reuse a colour.
    try:
        F.style_map([f"arm{i}" for i in range(len(F.FALLBACK_PALETTE) + 1)])
        raise SystemExit("palette overflow must fail loud")
    except AssertionError as ex:
        assert "exceeds the" in str(ex)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
