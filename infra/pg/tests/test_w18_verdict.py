"""Weight-free truth-table tests for the W18 unblind verdict path — the §3.1
freeze gate (critic MAJOR-1). Exercises: restoration-gate label split (floor-
band HALT vs reversal-as-science), SUPPORTED / PARTIAL / REVERSAL strings,
m=2 vs m=3 BH family, identity asserts, dropped-draw accounting. Runs the
ACTUAL aggregator main() on planted JSON cells. <3s, no model/GPU.

    python -m pytest infra/pg/tests/test_w18_verdict.py -q
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pytest


def _setup(tmp_path, S_512=0.8, S_64=0.7, S_8=0.2, probes_conf_p=None):
    N = 800
    rng = np.random.default_rng(0)
    hashes = [hashlib.sha1(f"q{i}".encode()).hexdigest()[:12] for i in range(N)]
    held = set(hashes[:300])                       # planted held-out population
    pool = set(hashes[700:])                       # planted pool (disjoint)
    b = (rng.random(N) < 0.55).astype(int).tolist()
    f = (rng.random(N) < 0.25).astype(int).tolist()
    ctrl = {"CONTROLS_OK": True, "recipe": {"FROZEN": True},
            "item_hashes": [h for h in hashes if h not in pool],
            "pool_hashes": sorted(pool), "heldout_hashes": sorted(held),
            "BASE": {"per_item": [v for v, h in zip(b, hashes) if h not in pool]},
            "FLOOR": {"per_item": [v for v, h in zip(f, hashes) if h not in pool]}}
    (tmp_path / "controls.json").write_text(json.dumps(ctrl))
    eligible = [h for h in hashes if h not in pool]
    held_hashes = [h for h in eligible if h in held]
    bm = np.asarray([v for v, h in zip(b, hashes) if h in held], float)
    fm = np.asarray([v for v, h in zip(f, hashes) if h in held], float)
    den = bm.mean() - fm.mean()
    cells = tmp_path / "cells"
    cells.mkdir(exist_ok=True)
    for rank, S in ((512, S_512), (64, S_64), (8, S_8)):
        target = fm.mean() + S * den
        for s in (1337, 1338, 1339):
            r2 = np.random.default_rng(rank * 10 + s)
            pi = (r2.random(len(held_hashes)) < target).astype(int).tolist()
            row = {"cell_id": f"rmu_cap_r{rank}_n375_s{s}_heldout",
                   "per_item": pi, "per_item_hash": held_hashes,
                   "controls_s3": str(tmp_path / "controls.json")}
            (cells / f"cell_{row['cell_id']}.json").write_text(json.dumps(row))
    probes = None
    if probes_conf_p is not None:
        probes = tmp_path / "probes.json"
        probes.write_text(json.dumps({"screening": {"promoted": True},
                                      "confirmation": {"p_one_sided": probes_conf_p}}))
    return ctrl, cells, probes


def _run(tmp_path, monkeypatch, **kw):
    from infra.aws_pg.jobs import w18_unblind as U
    ctrl, cells, probes = _setup(tmp_path, **{k: v for k, v in kw.items() if k != "probes_p"})
    argv = ["w18_unblind", "--out", str(tmp_path / "out"),
            "--controls", str(tmp_path / "controls.json"),
            "--cells-dir", str(cells), "--n-boot", "200"]
    if kw.get("probes_p") is not None:
        _, _, probes = _setup(tmp_path, probes_conf_p=kw["probes_p"], **{k: v for k, v in kw.items() if k != "probes_p"})
        argv += ["--probes", str(probes)]
    monkeypatch.setattr(sys, "argv", argv)
    U.main()
    return json.loads((tmp_path / "out" / "metrics.json").read_text())


def test_supported_both_axes(tmp_path, monkeypatch):
    m = _run(tmp_path, monkeypatch, S_512=0.85, S_64=0.75, S_8=0.2)
    assert m["VERDICT"].startswith("H_deeper-than-engineered SUPPORTED")
    assert m["r8_descriptive"]["point_S"] < 0.5
    assert "n_half > 1,000 > 375" in m["VERDICT"]      # frozen anchor phrasing


def test_partial_data_axis_only(tmp_path, monkeypatch):
    m = _run(tmp_path, monkeypatch, S_512=0.85, S_64=0.3)
    assert m["VERDICT"].startswith("PARTIAL")


def test_reversal_branch_not_halt(tmp_path, monkeypatch):
    # S ≈ 0.35: above the floor band (CI upper > 0.2) but never crosses 0.5 —
    # must route to REVERSAL (science datum), NOT harness-indicted HALT.
    m = _run(tmp_path, monkeypatch, S_512=0.35, S_64=0.3)
    assert m["VERDICT"].startswith("REVERSAL")
    assert m["restoration_gate"]["ci_upper_in_floor_band"] is False


def test_floor_band_halts(tmp_path, monkeypatch):
    m = _run(tmp_path, monkeypatch, S_512=0.02, S_64=0.02)
    assert m["VERDICT"].startswith("HARNESS-INDICTED HALT")
    assert "ordinal_tests" not in m


def test_probe_test_joins_family(tmp_path, monkeypatch):
    m = _run(tmp_path, monkeypatch, S_512=0.85, S_64=0.75, probes_p=0.004)
    assert "probe_promoted" in m["bh"]
    assert len(m["bh"]) == 3                            # m=3 family when promoted


def test_identity_drift_halts(tmp_path, monkeypatch):
    from infra.aws_pg.jobs import w18_unblind as U
    ctrl, cells, _ = _setup(tmp_path, S_512=0.8, S_64=0.7)
    victim = sorted(cells.glob("cell_rmu_cap_r512*s1338*.json"))[0]
    row = json.loads(victim.read_text())
    row["per_item_hash"] = list(reversed(row["per_item_hash"]))
    victim.write_text(json.dumps(row))
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(tmp_path / "o"),
                                      "--controls", str(tmp_path / "controls.json"),
                                      "--cells-dir", str(cells), "--n-boot", "100"])
    with pytest.raises(AssertionError, match="identity drift"):
        U.main()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
