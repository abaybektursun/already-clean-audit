"""Fixture test for the device check: two synthetic harvest dirs that differ by known amounts."""
import json

import numpy as np
import pytest

from infra.pg import neurons_device_check as DC

POLES = ["L00N0+", "L00N1-", "L01N2+", "L01N0-"]
E = 50


def write_dir(root, a, q999, max_abs, tok, positions, sha="s" * 64):
    (root / "poles").mkdir(parents=True)
    (root / "stats").mkdir()
    (root / "topk").mkdir()
    (root / "manifest.json").write_text(json.dumps({"tokens_sha256": sha, "n_tokens_scored": 1000,
                                                    "branch": "stage1-step1000000", "device_name": root.name}))
    (root / "poles" / "poles.json").write_text(json.dumps(POLES))
    np.savez(root / "poles" / "positions.npz", tok=positions)
    np.savez(root / "poles" / "exemplar_a.npz", a=a, g=a, u=a)
    for lyr in (0, 1):
        np.savez(root / "stats" / f"layer_{lyr:02d}.npz", q999=q999[lyr], max_abs=max_abs[lyr])
        np.savez(root / "topk" / f"layer_{lyr:02d}.npz", tok=tok[lyr])


def fixture_arrays():
    mag = 1 + 0.01 * np.arange(E)                                # |a| 1.00..1.49, median 1.245
    a = np.stack([s * mag for s in (1, -1, 1, -1)]).astype(np.float32)
    q999 = {lyr: np.ones((3, 2), np.float32) for lyr in (0, 1)}
    max_abs = {lyr: np.full((3, 2), 2.0, np.float32) for lyr in (0, 1)}
    q999[0][1, 1] = 1.30                                         # L00N1-: median 1.245 < 1.30, absent on A
    max_abs[1][2, 0] = 4.0                                       # L01N2+: 0.3 * 4.0 = 1.2 <= 1.245, present on A
    q999[1][0, 1] = max_abs[1][0, 1] = 0.0                       # L01N0-: thresholds 0, so only a+ and med > 0 decide
    tok = {lyr: np.arange(3 * 2 * E).reshape(3, 2, E) + 1000 * lyr for lyr in (0, 1)}
    positions = np.arange(len(POLES) * E).reshape(len(POLES), E) * 7 + 1
    return a, q999, max_abs, tok, positions


def test_known_differences_give_the_expected_flips(tmp_path):
    a, q999, max_abs, tok, pos = fixture_arrays()
    write_dir(tmp_path / "blackwell", a, q999, max_abs, tok, pos)
    b = a.copy()
    b[0] *= 1.1                                                  # L00N0+: stays present (q999 1.0)
    b[1] *= 1.1                                                  # L00N1-: median 1.3695 >= 1.30, percentile flips
    b[3, :30] *= -1                                              # L01N0-: 30 of 50 flip sign: |a| median unchanged,
                                                                 # a+ = max(-a, 0) median 0 on B -> absent on both rules
    max_abs_b = {k: v.copy() for k, v in max_abs.items()}
    max_abs_b[1][2, 0] = 4.5                                     # L01N2+: 0.3 * 4.5 = 1.35 > 1.245, raw-ratio flips
    tok_b = {k: v.copy() for k, v in tok.items()}
    tok_b[0][0, 0, :6] = 900_000 + np.arange(6)                  # L00N0+: 6 of its top-20 replaced -> overlap 14
    write_dir(tmp_path / "l40s", b, q999, max_abs_b, tok_b, pos)

    rows = [{"pole": "L00N1-", "sample": 0, "clusters": [
                {"c": 0, "exemplar_ranks": list(range(20, 40)), "echo": False},   # a+ median 1.295 < 1.30 -> 1.4245
                {"c": 1, "exemplar_ranks": list(range(40, 50)), "echo": False},   # 1.445 -> 1.5895, present on both
                {"c": 2, "exemplar_ranks": [0, 1, 2], "echo": True}]},              # echo: not read
            {"pole": "L00N0+", "sample": 1, "clusters": [{"c": 0, "exemplar_ranks": [0], "echo": False}]},
            {"pole": "L01N2+", "sample": 0, "declined": "no jobs", "clusters": []}]
    (tmp_path / "clusters.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    r = DC.main(["--a", str(tmp_path / "blackwell"), "--b", str(tmp_path / "l40s"),
                 "--poles", str(write_poles(tmp_path)), "--out", str(tmp_path / "report.json"),
                 "--clusters", str(tmp_path / "clusters.jsonl")])
    assert json.loads((tmp_path / "report.json").read_text())["n_poles"] == 4
    assert r["unit"].startswith("pole (proxy")
    assert r["presence"]["percentile"]["flipped"] == ["L00N1-", "L01N0-"]
    assert r["presence"]["raw_ratio"]["flipped"] == ["L01N2+", "L01N0-"]
    assert r["presence"]["percentile"]["present_A"] == 3 and r["presence"]["percentile"]["present_B"] == 3
    d = {x["pole"]: x for x in r["presence"]["percentile"]["detail"]}["L01N0-"]
    assert d["A"]["median_a_plus"] == pytest.approx(1.245, rel=1e-6) and d["B"]["median_a_plus"] == 0
    e = r["exemplar_a"]
    assert e["n_entries"] == 200 and e["sign_differs_share"] == pytest.approx(30 / 200)
    assert e["rel_diff"]["median"] == pytest.approx(0.1 / 1.1, rel=1e-5)    # 70 zeros, 100 at 0.1/1.1, 30 at 2
    assert e["rel_diff"]["max"] == pytest.approx(2.0)
    assert r["stats"]["max_abs_rel_diff"]["max"] == pytest.approx(0.5 / 4.5, rel=1e-6)
    assert r["stats"]["q999_rel_diff"]["max"] == 0
    assert r["topk_top20"]["below"] == [{"pole": "L00N0+", "overlap": 14}]
    assert r["topk_top20"]["overlap_median"] == 20 and r["topk_top20"]["share_below_15"] == 0.25
    c = r["clusters"]
    assert c["n_poles_with_clusters"] == 1 and c["n_clusters"] == 2
    assert c["percentile"]["flipped"] == [["L00N1-", 0]] and c["percentile"]["flip_share"] == 0.5
    assert c["raw_ratio"]["flipped"] == []


def write_poles(tmp_path):
    p = tmp_path / "grid_poles.json"
    p.write_text(json.dumps(POLES))
    return p


def test_different_exemplar_columns_fail_loudly(tmp_path):
    a, q999, max_abs, tok, pos = fixture_arrays()
    write_dir(tmp_path / "x", a, q999, max_abs, tok, pos)
    write_dir(tmp_path / "y", a, q999, max_abs, tok, pos + 1)
    with pytest.raises(AssertionError, match="positions.npz differs"):
        DC.compare(tmp_path / "x", tmp_path / "y", POLES)
    write_dir(tmp_path / "z", a, q999, max_abs, tok, pos, sha="t" * 64)
    with pytest.raises(AssertionError, match="tokens_sha256 differs"):
        DC.compare(tmp_path / "x", tmp_path / "z", POLES)
