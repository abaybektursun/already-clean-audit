"""Weight-free DRY-RUN gate for the DARWIN Wave-0 job functions (§9 step 2, §13 NEW row).

Run: PYTHONPATH=. python3 infra/pg/tests/test_darwin_w0_job.py

This is the gate the runbook means by "weight-free dry-run of every NEW pure instrument": it runs
the REAL `run_w0_1a` / `run_w0_2` bodies — the same code the instance executes — against a
synthetic bank library at the real ambient dimensions, with S3 and the Hub stubbed out. What it
proves is that the arms produce a schema-valid atlas, satisfy their own control-completeness
check, and emit the routing decision, without spending a cent or touching a weight.

Its own failure mode is worth naming: it does NOT prove the banks are the right objects (that is
G1a, on the instance, against HUMBOLDT's published library). It proves the plumbing.
"""
import argparse
import json
import pathlib
import tempfile

import numpy as np

from infra.aws_pg.jobs import darwin_w0 as JOB
from infra.pg import darwin_core as D
from infra.pg import darwin_w0 as DW
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as W

TYPES = ("q_proj", "k_proj")          # one 2048-in attention type + one MQA/rank-deficient type
LAYERS = (0, 1)


def _synthetic_bank_library(dest):
    """A full battery for TYPES x LAYERS at the REAL ambient dimensions.

    `core32` is given a deliberately SPIKY leverage profile (its mass concentrated on a few
    coordinates) and the haar twins are left flat, so the arms have a signal to separate and a
    control that must not show it. A library of pure-Haar banks would let a broken leverage
    statistic pass unnoticed.
    """
    d = pathlib.Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    for t in TYPES:
        n = H.IN_DIM[t]
        for L in LAYERS:
            for i, bid in enumerate(D.bank_ids_for(t)):
                if bid == "core32":
                    # Concentrate on 64 coordinates: leverage there is ~32/64, elsewhere ~0.
                    Z = np.zeros((n, H.BANK_K))
                    rng = np.random.default_rng(1000 + L)
                    hot = rng.choice(n, 64, replace=False)
                    Z[hot] = rng.standard_normal((64, H.BANK_K))
                    U = np.linalg.qr(Z)[0][:, :H.BANK_K]
                else:
                    U = H.haar_subspace(n, H.BANK_K, seed=7000 + 100 * i + L)
                np.save(d / W.bank_name(t, L, bid), np.ascontiguousarray(U))
    return d


def _args(out):
    return argparse.Namespace(out=str(out), arm=None, run_id="dryrun",
                              s3_prefix="darwin/dryrun", layers=list(LAYERS),
                              types=list(TYPES), seed=1337, smoke=True)


def _patched(bank_dir, monkey):
    """Stub the three system boundaries: bank fetch, S3 sync, and the on-instance cache root."""
    monkey["fetch"] = D.fetch_all_banks
    monkey["sync"] = D.s3_sync
    monkey["pub"] = D.publish_banks
    D.fetch_all_banks = lambda dest, require_topsigma=True: bank_dir
    D.s3_sync = lambda out, prefix: None
    D.publish_banks = lambda out: None


def _restore(monkey):
    D.fetch_all_banks = monkey["fetch"]
    D.s3_sync = monkey["sync"]
    D.publish_banks = monkey["pub"]


def _run(arm, fn):
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "out"
        out.mkdir()
        bank_dir = _synthetic_bank_library(pathlib.Path(td) / "banks")
        monkey = {}
        _patched(bank_dir, monkey)
        try:
            exp = D.Expectations(out.parent / "expectations.jsonl")
            ref = JOB._log_expectation(exp, arm, "2026-08-12T00:00:00Z")
            gates = H.Gates(out)
            atlas = D.Atlas(out, arm=arm, run_id="dryrun", register="ATLAS", wave="W0",
                            expected_ref=ref)
            result = fn(out, atlas, gates, _args(out), "deadbeef")
            atlas.close()
            rep = atlas.to_parquet()
            import pandas as pd
            df = pd.read_parquet(rep["path"])
            # Read every side-car artifact HERE: `out` lives inside the TemporaryDirectory and is
            # gone by the time the caller runs its assertions.
            files = {p.name: json.loads(p.read_text()) for p in sorted(out.glob("*.json"))}
        finally:
            _restore(monkey)
        return result, df, gates.gates, files


def test_w0_1a_emits_a_schema_valid_atlas_with_a_complete_control_battery():
    result, df, _gates, _files = _run("W0-1a", JOB.run_w0_1a)
    assert list(df.columns) == list(D.ATLAS_COLUMNS)
    assert set(df["register"]) == {"ATLAS"} and set(df["wave"]) == {"W0"}
    assert set(df["claimed_by"]) == {D.CLAIMED_BY}
    assert set(df["arm"]) == {"W0-1a"}
    # Every bank in the battery, including the DARWIN-derived structured control, has rows.
    banks = set(df["bank_id"].dropna())
    assert banks == set(D.bank_ids_for("q_proj")), banks
    assert "topsigma32" in banks and "anti32" in banks
    # Every (type, layer, bank) produced the same statistic set — no silently skipped cell.
    per_cell = df.groupby(["type", "layer", "bank_id"])["statistic"].nunique()
    assert per_cell.nunique() == 1, per_cell.to_dict()
    assert set(df["statistic"]) == {"leverage", "leverage_participation_ratio",
                                    "leverage_spike_count", "leverage_spike_coords",
                                    "leverage_topk_coords"}
    # The planted signal must be visible: the spiky core is spikier than its flat haar twins.
    counts = result["spike_counts_by_bank"]
    assert counts["core32"] > 0
    assert counts["core32"] > max(counts[b] for b in ("haar32_s0", "haar32_s1", "haar32_s2"))
    # ...and the participation ratio, the twin-free way of saying the same thing, agrees.
    pr = df[df["statistic"] == "leverage_participation_ratio"]
    assert (pr[pr["bank_id"] == "core32"]["value"].mean()
            < pr[pr["bank_id"] == "haar32_s0"]["value"].mean())


def test_w0_1a_refuses_an_incomplete_control_battery():
    """The control-completeness check must actually fire — a gate that cannot fail is a defect."""
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "out"
        out.mkdir()
        bank_dir = _synthetic_bank_library(pathlib.Path(td) / "banks")
        for p in bank_dir.glob("*__topsigma32.npy"):      # remove the structured control
            p.unlink()
        monkey = {}
        _patched(bank_dir, monkey)
        try:
            exp = D.Expectations(out.parent / "expectations.jsonl")
            ref = JOB._log_expectation(exp, "W0-1a", "2026-08-12T00:00:00Z")
            atlas = D.Atlas(out, arm="W0-1a", run_id="dryrun", register="ATLAS", wave="W0",
                            expected_ref=ref)
            JOB.run_w0_1a(out, atlas, H.Gates(out), _args(out), "deadbeef")
            raise SystemExit("a missing bank must fail loud, not produce a short table")
        except AssertionError as ex:
            assert "missing bank" in str(ex) or "topsigma32" in str(ex)
        finally:
            _restore(monkey)


def test_w0_2_emits_pairs_and_a_routing_decision():
    result, df, gates, files = _run("W0-2", JOB.run_w0_2)
    assert list(df.columns) == list(D.ATLAS_COLUMNS)
    n_pairs = len(LAYERS) * (len(LAYERS) - 1) // 2
    pair_rows = df[df["statistic"] == "cross_layer_mean_cos2"]
    assert len(pair_rows) == n_pairs * len(TYPES) * len(D.bank_ids_for("q_proj"))
    # Independent Haar banks must sit near k/n; a broken angle statistic would not.
    haar = pair_rows[(pair_rows["bank_id"] == "haar32_s0") & (pair_rows["type"] == "q_proj")]
    assert abs(float(haar["value"].mean()) - H.BANK_K / H.IN_DIM["q_proj"]) < 0.02
    routing = result["w1r_routing"]
    assert routing["variant"] in ("global", "per_layer")
    assert gates["W1R_routing"]["variant"] == routing["variant"]
    assert files["w1r_routing.json"]["variant"] == routing["variant"]
    # The summary CI rows exist for every bank — §3's "no point estimates without their interval".
    summ = df[df["statistic"] == "cross_layer_mean_cos2_summary"]
    assert len(summ) == len(TYPES) * len(D.bank_ids_for("q_proj"))
    assert summ["ci_lo"].notna().all() and summ["ci_hi"].notna().all()


def test_expectation_is_logged_before_any_row_exists():
    """§11's ordering requirement, checked mechanically rather than trusted."""
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "out"
        out.mkdir()
        exp = D.Expectations(out.parent / "expectations.jsonl")
        ref = JOB._log_expectation(exp, "W0-2", "2026-08-12T00:00:00Z")
        assert (out.parent / "expectations.jsonl").exists()
        assert not (out / "atlas.jsonl").exists()          # expectation strictly precedes rows
        # Re-logging is idempotent: a resumed job must not create a second expectation.
        assert JOB._log_expectation(exp, "W0-2", "2026-08-12T09:00:00Z") == ref
        assert len(exp.by_arm("W0-2")) == 1


def test_every_spec_arm_has_a_refuter_and_a_runner():
    for arm in JOB.ARMS:
        assert arm in JOB.SPEC_EXPECTATIONS, arm
        refuter, direction = JOB.SPEC_EXPECTATIONS[arm]
        assert refuter.strip() and direction.strip()
        assert arm in JOB.RUNNERS, arm


def test_frozen_rules_are_the_spec_rules():
    assert DW.OUTLIER_MULT == 5.0                     # §4/W0-1's own criterion
    assert DW.LEVERAGE_MULT == 5.0                    # journal amendment: same functional form
    assert DW.W1R_SHARED_FRACTION_GATE == 0.5         # §4/W1-R


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
