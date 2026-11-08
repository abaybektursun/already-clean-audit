"""neurons_verdict: every frozen rule on constructed inputs with known answers, the PREDICTIONS
gate, p_u recovery on synthetic judgments, and the unblind CLI end to end on synthetic banked
files. No model, no downloads.

    uv run --no-project --with pytest --with transformers --with tokenizers --with anthropic \
        --with numpy pytest infra/pg/tests/test_neurons_verdict.py -q
"""
from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from infra.pg import neurons_explain as NE
from infra.pg import neurons_figures as FG
from infra.pg import neurons_llm as llm
from infra.pg import neurons_verdict as NV
from infra.pg.tests import neurons_fixture as F


# --- Q1 rules --------------------------------------------------------------------------------

def test_score_by_pole_sign():
    assert NV.score(3.0, -1.0, 6.0, 0) == 0.5
    assert NV.score(3.0, -1.5, 6.0, 1) == 0.25
    assert NV.score(-0.1, -2.0, 6.0, 0) < 0                           # a text that never fires positive
    with pytest.raises(ValueError, match="dead pole"):
        NV.score(1.0, 0.0, 0.0, 0)
    with pytest.raises(ValueError, match="non-finite"):
        NV.score(float("nan"), 0.0, 6.0, 0)


def test_theta_s_on_known_scores_and_its_gate():
    scores = [i / 100 for i in range(100)]
    assert NV.theta_s(scores) == pytest.approx(0.9801)              # linear: 0.98 + 0.01 * 0.01
    assert NV.theta_gate(NV.theta_s(scores)) == "fail_ceiling"       # > 0.5: W1-C does not start
    assert NV.theta_s([0.3 * s for s in scores]) == pytest.approx(0.29403)
    assert NV.theta_gate(0.29403) == "pass"
    assert NV.theta_gate(NV.theta_s([0.0] * 100)) == "fail_floor"
    with pytest.raises(ValueError, match="at least 100"):
        NV.theta_s(scores[:99])


def test_pass_rule_and_the_essay_rule():
    th = 0.2
    pos, near, dist = [0.5, 0.5, 0.5, 0.0, 0.0], [0.3, 0.3, 0.0, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0, 0.0]
    assert NV.explanation_passes(pos, near, dist, th)
    assert NV.explanation_passes([0.2, 0.2, 0.2, 0, 0], near, dist, th)          # a score at θ_s is strong
    assert not NV.explanation_passes([0.5, 0.5, 0, 0, 0], near, dist, th)       # 2/5 positives
    assert not NV.explanation_passes(pos, [0.3] * 3 + [0, 0], dist, th)         # 3/5 near misses
    assert not NV.explanation_passes(pos, near, [0.3, 0.3, 0, 0, 0], th)        # 2/5 distant
    with pytest.raises(ValueError, match="exactly 5"):
        NV.explanation_passes(pos[:4], near, dist, th)
    assert NV.essay_rule([0.3] * 2 + [0] * 3, [0.3] + [0] * 4) == "win"
    assert NV.essay_rule([0.3] + [0] * 4, [0.31] + [0] * 4) == "tie"
    assert NV.essay_rule([0.29] * 5, [0.3] + [0] * 4) == "lose"


WORDS = ("first", "second", "third", "fourth")


def J(*related):
    return [{"name": f"the {WORDS[i]} job", "related_to_others": r, "reasoning": ""} for i, r in enumerate(related)]


def test_class_assignment_on_constructed_cases():
    A = NV.assign_class
    assert A(True, "ONE", J(False), True, {}) == "RARE"
    assert NV.is_rare(49, 3.0) and NV.is_rare(500, 0.0) and not NV.is_rare(50, 3.0)
    assert A(False, "ONE", J(False), True, {}) == "ONE"
    assert A(False, "ONE", J(False), False, {}) == "UNK"                       # fails §6.1
    assert A(False, "ONE", J(False, False), True, {}) == "UNK"                 # label and jobs disagree
    assert A(False, "TWO-REL", J(True, True), True, {}) == "TWO-REL"
    assert A(False, "TWO-REL", J(True, True, True), True, {}) == "TWO-REL"
    assert A(False, "TWO-REL", J(True, True, True, True), True, {}) == "UNK"   # four jobs
    assert A(False, "TWO-REL", J(True, False), True, {}) == "UNK"
    assert A(False, "TWO-REL", J(True, True), False, {}) == "UNK"
    # UNREL: support clause, explanation pass not required
    assert A(False, "UNREL", J(False, False), False, {0: True, 1: True}) == "UNREL"
    assert A(False, "UNREL", J(False, False), True, {0: True, 1: False}) == "UNK"
    assert A(False, "UNREL", J(False, False, False), True, {0: False, 1: True, 2: True}) == "UNREL"
    assert A(False, "UNREL", J(True, True, False), True, {2: True}) == "UNK"   # one unrelated job only
    with pytest.raises(KeyError):                                              # support never computed
        A(False, "UNREL", J(False, False), True, {0: True})
    assert A(False, "UNK", J(False), True, {}) == "UNK"


def test_wilson_against_a_hand_value():
    lo, hi = NV.wilson(8, 10)                    # (0.8 + 0.19207)/1.38415 -/+ 0.22658
    assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
    assert NV.wilson(0, 10)[0] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        NV.wilson(1, 0)


def test_kappa_and_holm_hand_values():
    assert NV.cohen_kappa(["ONE", "ONE", "UNK", "UNK"], ["ONE", "UNK", "UNK", "UNK"]) == pytest.approx(0.5)
    assert NV.cohen_kappa(["ONE"] * 3, ["ONE"] * 3) is None
    assert NV.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_single_token_suppressor():
    assert NV.single_token_suppressor({"promoted": [["a", 0.4]], "suppressed": [["b", -0.9], ["c", -0.3]]})
    assert not NV.single_token_suppressor({"promoted": [["a", 0.5]], "suppressed": [["b", -0.9]]})


def test_pole_vocab_flips_the_negative_pole_and_the_suppressor_flag_follows():
    """vocab.json is per pole: the "-" entry is already the sign flip of the "+" entry (JOURNAL 2026-10-25, F41), and
    pole_vocab returns the file's own entry. A neuron that strongly promotes "a" makes its negative pole a single-token
    suppressor of "a". A "-" entry that is not the flip (the per-neuron layout) is refused."""
    neuron = {"promoted": [["a", 0.9], ["c", 0.2]], "suppressed": [["b", -0.3]]}
    flipped = {"promoted": [["b", 0.3]], "suppressed": [["a", -0.9], ["c", -0.2]]}
    vocab = {"L35N7+": neuron, "L35N7-": flipped}
    assert NV.pole_vocab(vocab, "L35N7+") == neuron
    assert NV.pole_vocab(vocab, "L35N7-") == flipped
    assert not NV.single_token_suppressor(NV.pole_vocab(vocab, "L35N7+"))
    assert NV.single_token_suppressor(NV.pole_vocab(vocab, "L35N7-"))
    with pytest.raises(ValueError, match="not the sign flip"):
        NV.pole_vocab({"L35N7+": neuron, "L35N7-": neuron}, "L35N7-")


def test_a_nan_or_missing_probe_value_raises_naming_text_and_pole(tmp_path):
    corpus = F.build_corpus(tmp_path)
    harvest = F.build_harvest(tmp_path / "h", corpus, poles=["L00N0+"])
    texts = [{"text_id": "L00N0+:pos:0"}, {"text_id": "L00N0+:pos:1"}]
    probe = F.write_probe(tmp_path / "p", texts, ["L00N0+", "L00N1-"],
                          lambda t, p: (float("nan"), 0.0) if (t["text_id"], p) == ("L00N0+:pos:1", "L00N1-") else (1.0, -1.0))
    sc = NV.scorer(harvest, probe)
    assert sc("L00N0+:pos:0", "L00N1-") > 0
    with pytest.raises(ValueError, match=r"non-finite probe value for text L00N0\+:pos:1 at pole L00N1-"):
        sc("L00N0+:pos:1", "L00N1-")
    with pytest.raises(ValueError, match=r"no probe value for text L00N0\+:pos:9 at pole L00N0\+"):
        sc("L00N0+:pos:9", "L00N0+")
    with pytest.raises(ValueError, match="no probe value"):
        sc("L00N0+:pos:0", "L01N2+")



def test_pos0_share_missing_from_the_stats_file_is_an_error_naming_it(tmp_path):
    corpus = F.build_corpus(tmp_path, n_docs=30)
    harvest = F.build_harvest(tmp_path / "h", corpus, poles=["L01N2+"])
    assert NV.pole_pos0_share(harvest, "L01N2+") == pytest.approx(0.04)
    st = NV.load_npz(harvest / "stats" / "layer_01.npz")
    del st["pos0_share"]
    np.savez(harvest / "stats" / "layer_01.npz", **st)
    NV.layer_file.cache_clear()
    with pytest.raises(ValueError, match=r"stats/layer_01.npz: no pos0_share"):
        NV.pole_pos0_share(harvest, "L01N2+")


def test_pass_rate_by_encoding_flag():
    recs = [{"passes": True, "encoding_flag": True}, {"passes": False, "encoding_flag": True},
            {"passes": True, "encoding_flag": False}, {"encoding_flag": True}]          # the last is unscored
    r = NV.pass_rate_by_encoding_flag(recs)
    assert (r["flagged"]["k"], r["flagged"]["n"], r["unflagged"]["k"], r["unflagged"]["n"]) == (1, 2, 1, 1)


def test_a_probe_without_the_text_position_rule_is_refused(tmp_path):
    """Probe outputs from before neurons/harvest 62d865d3 carry no probe_rule: theta, discovery and
    unblind refuse them (all score through `scorer`), naming the file."""
    corpus = F.build_corpus(tmp_path, n_docs=30)
    harvest = F.build_harvest(tmp_path / "h", corpus, poles=["L00N0+"])
    texts = [{"text_id": "L00N0+:pos:0"}]
    old = F.write_probe(tmp_path / "old", texts, ["L00N0+"], lambda t, p: (1.0, -1.0), rule=False)
    with pytest.raises(ValueError, match=r"old/poles/probe/scores.npz: no probe_rule"):
        NV.scorer(harvest, old)
    new = F.write_probe(tmp_path / "new", texts, ["L00N0+"], lambda t, p: (1.0, -1.0))
    assert NV.probe_rule(new) == {"probe_rule": F.PROBE_RULE, "probe_excluded_positions": "text position 0"}
    assert NV.scorer(harvest, new)("L00N0+:pos:0", "L00N0+") > 0
    m = json.loads((new / "manifest.json").read_text())
    del m["probe_excluded_positions"]
    (new / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match=r"new/manifest.json: probe_excluded_positions is None"):
        NV.scorer(harvest, new)


def test_draw_s_takes_every_non_rare_pole_from_stats_only(tmp_path, capsys, monkeypatch):
    """§4.4: S = the non-RARE audit poles, S_gate = the non-RARE extra poles, grid_poles = their union;
    RARE comes from stats/ alone (here one pole with density 10 and one dead pole)."""
    corpus = F.build_corpus(tmp_path, n_docs=30)
    harvest = F.build_harvest(tmp_path / "h", corpus, n_layers=2, n_neurons=6, dead=("L01N5-",))
    st = NV.load_npz(harvest / "stats" / "layer_00.npz")
    st["density"][2, 0] = 10                                          # L00N2+: fewer than 50 tokens above 1%
    F.set_stats(harvest, 0, density=st["density"])
    NV.layer_file.cache_clear()
    audit, extra = ["L01N1-", "L00N2+", "L00N0+", "L00N1-"], ["L01N5-", "L01N3+"]
    (tmp_path / "audit.json").write_text(json.dumps(audit))
    (tmp_path / "extra.json").write_text(json.dumps(extra))
    args = ["draw-s", "--stats", str(harvest), "--audit", str(tmp_path / "audit.json"), "--extra",
            str(tmp_path / "extra.json"), "--seed", "0", "--out", str(tmp_path / "w3")]
    NV.main(args)
    draw = tmp_path / "w3" / "draw"
    S, S_gate, grid = (NV.read_json(draw / f) for f in ("S.json", "S_gate.json", "grid_poles.json"))
    assert S == ["L00N0+", "L00N1-", "L01N1-"] and S_gate == ["L01N3+"]
    assert grid == ["L00N0+", "L00N1-", "L01N1-", "L01N3+"] and set(grid) == set(S) | set(S_gate)  # the W3 --poles
    record = NV.read_json(draw / "draw.json")
    assert record["counts"] == {"audit": 4, "extra": 2, "rare_excluded_audit": 1, "rare_excluded_extra": 1,
                                "S": 3, "S_gate": 1, "grid_poles": 4}
    assert record["rare_excluded"]["audit"]["L00N2+"]["density"] == 10
    assert record["rare_excluded"]["extra"] == {"L01N5-": {"density": 0, "max_abs": 0.0}}
    assert record["seed"] == 0 and "no random step" in record["rule"] and len(record["inputs"]["audit"]["sha256"]) == 64
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == record["counts"]
    with pytest.raises(FileExistsError):
        NV.main(args)                                                 # drawn once
    (tmp_path / "extra2.json").write_text(json.dumps(["L00N0+"]))
    with pytest.raises(ValueError, match="share poles"):
        NV.main([*args[:5], "--extra", str(tmp_path / "extra2.json"), *args[7:-1], str(tmp_path / "w3b")])
    monkeypatch.setattr(NV, "S_SIZE", 2)
    with pytest.raises(ValueError, match=r"3 non-RARE audit poles > \|S\| = 2"):
        NV.main([*args[:-1], str(tmp_path / "w3c")])

# --- presence and births ---------------------------------------------------------------------

def test_birth_requires_staying_present():
    assert NV.birth([0, 1, 0, 1, 1, 0]) == 3
    assert NV.birth([0] * 15 + [1]) == 15                             # T_15 only: birth T_15 (amended)
    assert NV.birth([0, 1, 0, 1, 0]) is None
    assert NV.late_flags([0, 3, None, 0]) == [False, True, False, False]
    pct, raw = NV.presence([0.0, 2.0, 5.0], [0.0, 1.0, 1.0], [0.0, 10.0, 10.0])
    assert pct.tolist() == [False, True, True] and raw.tolist() == [False, False, True]   # zero is never present


def table_case():
    """One pole, five sample-0 clusters of ten exemplars each; q99.9 = 4 everywhere; max|a| = 6 up
    to T_4 and 20 from T_5 (an activation-scale change)."""
    T = len(NV.GRID_STEPS)
    med = np.zeros((T, 5))
    med[:, 0] = 10.0                                       # c0: present from T_0 under both readings
    med[6:, 1] = 6.0                                       # c1: born T_6 under both readings
    med[:5, 2], med[5:, 2] = 2.5, 8.0                      # c2: raw says T_0, percentile says T_5
    med[1:, 3] = 5.0                                       # c3: born T_1 under both readings
    med[7:, 4] = 9.0                                       # c4: born T_7, split in sample 1
    mag = np.repeat(med, 10, axis=1)
    max_abs = np.where(np.arange(T) <= 4, 6.0, 20.0)
    c0 = [{"c": i, "name": f"c{i}", "echo": False, "exemplar_ranks": list(range(10 * i, 10 * i + 10))} for i in range(5)]
    c1 = c0[:4] + [{"c": 4, "name": "a", "echo": False, "exemplar_ranks": list(range(40, 45))},
                   {"c": 5, "name": "b", "echo": False, "exemplar_ranks": list(range(45, 50))}]
    return mag, np.full(T, 4.0), max_abs, c0, c1


def test_cluster_table_statuses_including_a_scale_drift_exclusion():
    mag, q999, max_abs, c0, c1 = table_case()
    rows = NV.cluster_table("L00N0+", c0, c1, mag, q999, max_abs, q999_alt=np.full(16, 7.0), max_abs_alt=max_abs)
    by = {r["c"]: r for r in rows}
    assert [by[c]["status"] for c in range(5)] == ["earliest", "late_primary", "readings_disagree",
                                                   "formed_before_step_1000", "unmatched_in_sample_1"]
    assert (by[2]["birth_pct"], by[2]["birth_raw"]) == (5, 0)          # the two readings disagree
    assert by[2]["late_pct"] and not by[2]["late_raw"] and by[2]["t_before"] is None
    assert by[1]["t_before"] == 5 and by[1]["r_at_birth"] == pytest.approx(6.0 / 20.0)
    assert by[0]["is_main"] and sum(r["is_main"] for r in rows) == 1
    assert by[1]["split_changed_10m"] and not by[0]["split_changed_10m"]   # c1 never reaches 7 on 10M
    bad = mag.copy()
    bad[:, 30:40] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        NV.cluster_table("L00N0+", c0, c1, bad, q999, max_abs)


def test_echo_clusters_take_no_part_in_main_births_or_counts():
    mag, q999, max_abs, c0, c1 = table_case()
    mag = np.concatenate([mag, np.zeros((16, 10))], axis=1)
    mag[9:, 50:] = 50.0                                     # an echo: the largest, born T_9
    echo = {"c": 5, "name": "echo", "echo": True, "exemplar_ranks": list(range(50, 60))}
    rows = NV.cluster_table("L00N0+", c0 + [echo], c1 + [{**echo, "c": 6}], mag, q999, max_abs)
    by = {r["c"]: r for r in rows}
    assert by[5]["status"] == "echo" and not by[5]["is_main"] and by[0]["is_main"]
    assert by[1]["status"] == "late_primary" and NV.vestigial_counts(rows) == NV.vestigial_counts(rows[:5])
    only_echo = NV.cluster_table("L00N0+", [echo], [echo], mag, q999, max_abs)
    assert [r["status"] for r in only_echo] == ["echo"] and not any(r["is_main"] for r in only_echo)


def test_undefined_growth_is_excluded_and_counted():
    mag, q999, max_abs, c0, c1 = table_case()
    rows = NV.cluster_table("L00N0+", c0, c1, mag, q999, max_abs)
    assert NV.vestigial_counts(rows) == (0, 4, 0)
    zero = max_abs.copy()
    zero[NV.T_LATE_HALF] = 0.0
    rows = NV.cluster_table("L00N0+", c0, c1, mag, q999, zero)
    assert all(r["growth_last_half"] is None for r in rows)
    assert NV.vestigial_counts(rows) == (0, 0, 4)
    at_birth = max_abs.copy()
    at_birth[6] = 0.0                                       # c1 is present at T_6 with max|a| = 0 there
    with pytest.raises(ValueError, match="stats and exemplar files disagree"):
        NV.cluster_table("L00N0+", c0, c1, mag, q999, at_birth)


def test_vestigial_counts_born_at_or_before_main_and_shrinking():
    T = len(NV.GRID_STEPS)
    med = np.zeros((T, 3))
    med[3:, 0] = 10.0                                      # main, born T_3
    med[2:, 1] = 8.0                                       # born before main ...
    med[-1, 1] = 6.0                                       # ... and shrinking T_12 -> T_15: vestigial
    med[5:, 2] = 4.0                                       # born after main: not vestigial
    mag = np.repeat(med, 10, axis=1)
    cs = [{"c": i, "name": "", "echo": False, "exemplar_ranks": list(range(10 * i, 10 * i + 10))} for i in range(3)]
    rows = NV.cluster_table("L00N0+", cs, cs, mag, np.full(T, 1.0), np.full(T, 20.0))
    assert [r["birth_pct"] for r in rows] == [3, 2, 5] and rows[0]["is_main"]
    assert NV.vestigial_counts(rows) == (1, 2, 0)


def test_signed_magnitude_counts_only_the_pole_direction():
    assert NV.signed_magnitude([2.0, -3.0], 0).tolist() == [2.0, 0.0]
    assert NV.signed_magnitude([2.0, -3.0], 1).tolist() == [0.0, 3.0]


# --- p_u, birth order, vestigial -------------------------------------------------------------

def synthetic(p_u, seed, n=400, a_ctrl=0.95):
    rng = np.random.default_rng(seed)
    unrelated = rng.random(n) < p_u
    acc = np.where(unrelated, rng.binomial(4, 0.5, n), rng.binomial(4, a_ctrl, n)) / 4
    return [[x] for x in acc], list(rng.binomial(4, a_ctrl, n) / 4)


@pytest.mark.parametrize("truth", [0.05, 0.30])
def test_p_u_recovery_over_seeds(truth):
    """Power and coverage at n = 400 clusters, a_ctrl = 0.95, four judgments each, data seeds 0-199
    (bootstrap seed = data seed, 10,000 draws). Coverage >= 0.90; at 0.05 H_RG fires in 55-85% of
    datasets (SPEC §15.7: about 70%); at 0.30 H_SP fires in at least 85%. 200 seeds, not 50: on
    seeds 0-49 coverage at 0.05 is 0.88 (binomial sd about 0.036 around a rate near 0.93), so a
    50-seed test of a 0.90 floor measures sampling noise as much as the estimator."""
    covered, rules = 0, []
    for seed in range(200):
        r = NV.p_u_bootstrap(*synthetic(truth, seed), n_boot=NV.N_BOOT, seed=seed)
        covered += r["ci95"][0] <= truth <= r["ci95"][1]
        rules.append(NV.p_u_rule(r["ci95"]))
    assert covered / 200 >= 0.90
    if truth == 0.05:
        assert 0.55 <= rules.count("H_RG") / 200 <= 0.85 and "H_SP" not in rules
    else:
        assert rules.count("H_SP") / 200 >= 0.85 and "H_RG" not in rules


def test_p_u_regression_pin():
    """A regression pin on the exact output for data seed 0, bootstrap seed 0. Not a power claim."""
    r = NV.p_u_bootstrap(*synthetic(0.05, 0), n_boot=NV.N_BOOT, seed=0)
    assert (r["A"], r["a_ctrl"]) == pytest.approx((0.9275, 0.948125))
    assert r["point"] == pytest.approx(0.04602510460251047, abs=1e-12)
    assert r["ci95"] == pytest.approx([0.005722052493984807, 0.08551724137931024], abs=1e-12)
    assert "p_RG" not in r and "p_SP" not in r                        # p_u carries no p-value (§7)
    assert r["degenerate_draws"] == 0                                 # every pole has a cluster: no redraw


def test_p_u_conditional_bootstrap_redraws_degenerate_draws():
    """The W0 case: one pole of four has no late-born cluster, so a draw of four poles holds none with
    probability (1/4)^4 = 0.39%. Such draws are redrawn and counted; the interval is finite."""
    r = NV.p_u_bootstrap([[1.0, 0.75], [0.5], [0.75], []], [1.0, 0.75, 1.0, 1.0], n_boot=NV.N_BOOT, seed=0)
    assert 0 < r["degenerate_draws"] <= NV.DEGENERATE_LIMIT * NV.N_BOOT
    assert all(np.isfinite(r["ci95"])) and r["ci95"][0] <= r["point"] <= r["ci95"][1]


def test_p_u_refuses_when_degenerate_draws_exceed_one_percent():
    with pytest.raises(ValueError, match=r"no late-born cluster or no control: 10000 degenerate draws exceed 1%"):
        NV.p_u_bootstrap([[]] * 5, [1.0] * 5, n_boot=NV.N_BOOT, seed=0)        # no pole has a late-born cluster
    # Only one pole of twenty has late-born clusters: a draw misses it with probability
    # (19/20)^20 = 36%, far above the 1% limit, so this also refuses.
    with pytest.raises(ValueError, match="degenerate draws exceed 1%"):
        NV.p_u_bootstrap([[1.0]] + [[]] * 19, [1.0] * 20, n_boot=NV.N_BOOT, seed=0)


@pytest.mark.parametrize("truth,rule", [(0.05, "H_RG"), (0.30, "H_SP")])
def test_p_u_at_the_expected_realization(truth, rule):
    """Exact proportions: related clusters at accuracy 0.95 (3/4 or 4/4), unrelated at 0.5 (2/4),
    controls at 0.95. A and a_ctrl equal their §7 values, so the point is the truth."""
    n_un = int(round(400 * truth))
    clusters = [[0.75] if i % 5 == 0 else [1.0] for i in range(400 - n_un)] + [[0.5]] * n_un
    ctrl = [0.75 if i % 5 == 0 else 1.0 for i in range(400)]
    r = NV.p_u_bootstrap(clusters, ctrl, n_boot=NV.N_BOOT, seed=0)
    assert r["a_ctrl"] == pytest.approx(0.95) and r["point"] == pytest.approx(truth)
    assert r["ci95"][0] <= truth <= r["ci95"][1] and NV.p_u_rule(r["ci95"]) == rule


def test_p_u_moves_clusters_with_their_pole_and_takes_weights():
    """200 poles x 3 identical clusters: resampling poles widens the interval by about sqrt(3)
    against 600 poles x 1 cluster with the same accuracies."""
    rng = np.random.default_rng(2)
    acc = rng.binomial(4, 0.85, 200) / 4
    grouped = NV.p_u_bootstrap([[x] * 3 for x in acc], [1.0] * 200, n_boot=4000, seed=0)
    flat = NV.p_u_bootstrap([[x] for x in np.repeat(acc, 3)], [1.0] * 600, n_boot=4000, seed=0)
    width = lambda r: r["ci95"][1] - r["ci95"][0]  # noqa: E731
    assert grouped["point"] == pytest.approx(flat["point"]) and 1.4 < width(grouped) / width(flat) < 2.1
    same = NV.p_u_bootstrap([[x] for x in acc], [1.0] * 200, n_boot=500, seed=1, weights=np.ones(200))
    assert same == NV.p_u_bootstrap([[x] for x in acc], [1.0] * 200, n_boot=500, seed=1)
    heavy = NV.p_u_bootstrap([[1.0], [0.5]], [1.0, 1.0], n_boot=100, seed=0, weights=[3.0, 1.0])
    assert heavy["A"] == pytest.approx(0.875)


def test_p_u_refuses_a_chance_level_control():
    with pytest.raises(ValueError, match="a_ctrl"):
        NV.p_u_bootstrap([[0.5]] * 10, [0.5] * 10, n_boot=100)
    assert NV.p_u_rule([0.0, 0.099]) == "H_RG" and NV.p_u_rule([0.201, 0.4]) == "H_SP"
    assert NV.p_u_rule([0.05, 0.15]) == "mixed" and NV.p_u_rule([0.1, 0.2]) == "mixed"


def test_birth_order_permutation_test_and_interval():
    rng = np.random.default_rng(1)
    planted = [([1, 4, 7], 0)] * 100                                 # main always earliest of three
    r = NV.birth_order_test(planted, n_perm=NV.N_PERM, seed=0)
    assert r["b"] == 1.0 and r["null_mean"] == pytest.approx(1 / 3, abs=0.01) and r["ci95"] == [1.0, 1.0]
    assert r["supports_main_oldest"] and r["p"] == pytest.approx(1 / (NV.N_PERM + 1))
    shuffled = [([1, 4, 7], int(rng.integers(3))) for _ in range(100)]   # main at a random rank
    r = NV.birth_order_test(shuffled, n_perm=NV.N_PERM, seed=0)
    assert not r["supports_main_oldest"] and r["drift_share"] == pytest.approx(1 - r["b"])
    assert r["ci95"][0] < r["b"] < r["ci95"][1]
    # The null permutes within each pole: 50 poles where every birth ties (main always earliest) and
    # 50 two-cluster poles with main the later one. Within-pole expectation (1 + 0.5) / 2 = 0.75.
    mixed = [([1, 1, 1, 1], 0)] * 50 + [([1, 5], 1)] * 50
    r = NV.birth_order_test(mixed, n_perm=NV.N_PERM, seed=0)
    assert r["b"] == 0.5 and r["null_mean"] == pytest.approx(0.75, abs=0.005) and r["p"] == 1.0


def test_vestigial_rule():
    assert NV.vestigial([(3, 3)] * 50, n_boot=1000)["label"] == "H_VEST"
    r = NV.vestigial([(1, 3)] * 50, n_boot=1000)
    assert r["label"] == "not H_VEST" and r["v"] == pytest.approx(1 / 3)


def test_stratum_weights():
    w, info = NV.stratum_weights(["a", "b", "c"], {"a": "UNREL", "b": "UNREL", "c": "ONE"},
                                 {"ONE": 90, "TWO-REL": 7, "UNREL": 3, "UNK": 0})
    assert w.tolist() == [1.5, 1.5, 90.0] and info["unrepresented_strata"] == ["TWO-REL"]
    with pytest.raises(ValueError, match="without a Q1 class"):
        NV.stratum_weights(["a"], {"a": "RARE"}, {"ONE": 1})


def test_g2_against_its_line(tmp_path):
    e = tmp_path / "explain"
    e.mkdir()
    rows = [{"pole": f"L00N{i}+", "sample": 0, "confidence": 0.9 if i < 2 else 0.2} for i in range(10)]
    llm.write_jsonl_new(e / "explanations.jsonl", rows + [{"pole": "L00N0+", "sample": 1, "confidence": 0.9}])
    g = NV.g2_readout(e)
    assert g["n_poles"] == 10 and g["share"] == 0.2 and not g["within_line"]


# --- §9.3 gate -------------------------------------------------------------------------------

def manifest(path, started):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"started_utc": started}))
    return path


def refused(capsys, journal, manifests, needle):
    with pytest.raises(SystemExit) as e:
        NV.predictions_gate(journal, manifests)
    err = capsys.readouterr().err
    assert e.value.code == 2 and needle in err, err


def test_predictions_gate_refuses_and_accepts(tmp_path, capsys):
    m1 = manifest(tmp_path / "a" / "manifest.json", "2026-10-20T00:00:00Z")
    m2 = manifest(tmp_path / "b" / "manifest.json", "2026-10-21T09:00:00Z")
    ok = "## PREDICTIONS 2026-10-01T00:00Z"
    j = F.git_journal(tmp_path / "r1", "## 2026-10-19 notes", body=["text"])
    refused(capsys, j, [m1], "no '## PREDICTIONS' heading")
    j = F.git_journal(tmp_path / "r2", "## PREDICTIONS 2026-10-01T10:00")                 # no zone
    refused(capsys, j, [m1], "explicit zone")
    j = F.git_journal(tmp_path / "r3", "## PREDICTIONS 2026-10-01")                       # date only
    refused(capsys, j, [m1], "explicit zone")
    j = F.git_journal(tmp_path / "r4", ok, body=["(to be written)"])                      # empty
    refused(capsys, j, [m1], "no line for ['p_u', 'b', 'v', 'H_MIG', 'hard-decoy']")
    j = F.git_journal(tmp_path / "r5", ok, commit_date=None)                               # never committed
    refused(capsys, j, [m1], "not in any commit")
    j = F.git_journal(tmp_path / "r6", ok)
    j.write_text(j.read_text().replace("- v: low", "- v: high"))                           # edited after commit
    refused(capsys, j, [m1], "uncommitted changes")
    j = F.git_journal(tmp_path / "r7", ok, commit_date="2026-10-20T06:00:00+00:00")         # committed late
    refused(capsys, j, [m1, m2], f"{m1}: started_utc 2026-10-20T00:00:00Z does not follow the PREDICTIONS commit")
    j = F.git_journal(tmp_path / "r8", "## PREDICTIONS 2026-10-19T20:00-10:00")            # 06:00Z on the 20th
    refused(capsys, j, [m1], "PREDICTIONS heading time")
    j = F.git_journal(tmp_path / "r9", "## PREDICTIONS 2026-09-01T00:00Z",                # a stub first,
                      body=["(stub)", "", ok, *F.READOUT_LINES])                           # the real entry after
    refused(capsys, j, [m1], "no line for")                                               # no fall-through
    other = tmp_path / "r10" / "JOURNAL.md"
    other.parent.mkdir()
    other.write_text("## PREDICTIONS 2026-10-01T00:00Z\n")
    refused(capsys, other, [m1], "studies/already-clean/JOURNAL.md")
    j = F.git_journal(tmp_path / "r11", "## PREDICTIONS 2026-10-01T02:00+02:00")           # 00:00Z
    got = NV.predictions_gate(j, [m1, m2])
    assert got["heading_utc"] == "2026-10-01T00:00:00+00:00" and got["commit_utc"] == "2026-10-01T00:00:00+00:00"
    refused(capsys, j, [tmp_path / "missing.json"], "manifest not found")


# --- judge gate ------------------------------------------------------------------------------

S = ["L00N0+", "L00N1-", "L00N2+"]


def design(t, pole):
    mag = np.zeros(50)
    if t >= 1:
        mag[:25] = 8.0                                                # c0: main, born T_1
    if t >= 4:
        mag[25:] = 5.0                                                # c1: late, born T_4
    return mag, 1.0, 10.0


def judged_dir(out, grid, corpus, batch, right_random, seed, gate):
    """positions, two clustering samples (two halves plus the echo), the judge plan and answers in
    which each pole's late cluster gets right_random[pole] of its four random-decoy judgments right;
    controls and hard decoys are always right. The directory's poles are right_random's keys; `gate`
    is prepare-judge's --gate (the directory itself for the gate directory)."""
    (out / "S.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "S.json").write_text(json.dumps(list(right_random)))
    NE.main(["positions", "--harvest", str(grid[-1]), "--out", str(out), "--poles", str(out / "S.json")])
    NE.main(["prepare-clusters", "--harvest", str(grid[-1]), "--corpus", str(corpus), "--out", str(out),
             "--batch", "c", "--model", "test-model", "--seed", str(seed)])
    explain = out / "explain"
    cplans = F.plans_by_id(explain, "c-clusterer")
    for s in (0, 1):
        F.answer_batch(explain, f"c-clusterer-s{s}", lambda r: F.two_halves(cplans[r["id"]]))
    NE.main(["collect-clusters", "--out", str(out), "--batch", "c", "--backend", "manual"])
    NE.main(["prepare-judge", "--grid", ",".join(map(str, grid)), "--corpus", str(corpus), "--out", str(out),
             "--batch", batch, "--model", "test-model", "--seed", str(seed), "--gate", str(gate)])
    plans = F.plans_by_id(explain, f"{batch}-judge")

    def answer(req):
        p = plans[req["id"]]
        truth = "A" if p["shown_order"][0] == "true" else "B"
        correct = p["control"] or p["decoy_type"] == "hard" or 2 * p["draw"] + p["sample"] < right_random[p["cluster"].split(":")[0]]
        return {"answer": truth if correct else ("B" if truth == "A" else "A"), "reason": "r"}
    for s in (0, 1):
        F.answer_batch(explain, f"{batch}-judge-s{s}", answer)
    NE.main(["collect-judge", "--out", str(out), "--batch", batch, "--backend", "manual"])
    return explain


@pytest.fixture(scope="module")
def q2_bank(tmp_path_factory):
    root = tmp_path_factory.mktemp("q2")
    corpus = F.build_corpus(root)
    grid = F.build_grid(root / "grid", corpus, S, design)
    dev = judged_dir(root / "dev", grid, corpus, "jd", {p: 4 for p in S}, seed=6, gate=root / "dev")
    NV.main(["judge-gate", "--explain", str(dev.parent), "--out", str(dev.parent)])   # before the S judge batch
    s_dir = judged_dir(root / "w4", grid, corpus, "j", {"L00N0+": 4, "L00N1-": 2, "L00N2+": 0}, seed=5, gate=root / "dev")
    return root, corpus, grid, dev, s_dir


def test_judge_gate_requires_the_full_control_plan(q2_bank, tmp_path):
    root, _, _, dev, _ = q2_bank
    gate = NV.read_json(root / "dev" / "verdict" / "judge_gate.json")
    assert gate["gate"] == "pass" and gate["a_ctrl"] == 1.0 and gate["f_ctrl"] == 0.0 and gate["n_poles"] == 3
    assert gate["created_utc"] and set(gate["inputs"]) >= {"judgments", "clusters", "plan:jd-judge.jsonl"}
    assert gate["judge"] == {"models": ["test-model"], "prompt_sha256": NE.load_prompt("judge")[2]}
    assert len(gate["inputs"]["judgments"]["sha256"]) == 64
    short = tmp_path / "short"
    shutil.copytree(dev.parent, short)
    rows = llm.read_jsonl(short / "explain" / "judgments.jsonl")
    dropped = next(r for r in rows if r["control"])
    (short / "explain" / "judgments.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r is not dropped))
    with pytest.raises(ValueError, match=f"missing \\['{dropped['id']}'\\]"):
        NV.main(["judge-gate", "--explain", str(short), "--out", str(tmp_path / "v1")])
    plan = short / "explain" / "plans" / "jd-judge.jsonl"
    plans = llm.read_jsonl(plan)
    cut = [p for p in plans if not (p["control"] and p["cluster"].startswith("L00N1-") and p["decoy_type"] == "hard")]
    plan.write_text("".join(json.dumps(p) + "\n" for p in cut))
    with pytest.raises(ValueError, match=r"L00N1-: planned controls \{'random': 4\}"):
        NV.main(["judge-gate", "--explain", str(short), "--out", str(tmp_path / "v2")])


# --- one grid harvest serving several explain directories ------------------------------------

def per_pole_design(t, pole):
    """Births differ by pole, so reading another pole's exemplar row changes the table."""
    _, n, s = NV.parse_pole(pole)
    mag = np.zeros(50)
    if t >= 1 + n:
        mag[:25] = 8.0 + n
    if t >= 4 + 2 * n + s:
        mag[25:] = 5.0
    return mag, 1.0, 20.0


def hand_explain(root, poles):
    """An explain directory holding only what grid_cluster_tables reads: positions_poles.json and
    two clustering samples per pole (ranks 0-24 and 25-49)."""
    e = root / "explain"
    e.mkdir(parents=True)
    (e / "positions_poles.json").write_text(json.dumps(poles))
    halves = [{"c": 0, "name": "a", "echo": False, "exemplar_ranks": list(range(25))},
              {"c": 1, "name": "b", "echo": False, "exemplar_ranks": list(range(25, 50))}]
    llm.write_jsonl_new(e / "clusters.jsonl", [{"pole": p, "sample": s, "clusters": halves, "pairs": [[0, 1, "related"]]}
                                               for p in poles for s in (0, 1)])
    return e


def test_grid_tables_unchanged_when_the_grid_holds_more_poles_in_another_order(tmp_path):
    corpus = F.build_corpus(tmp_path, n_docs=60)
    S = ["L00N0+", "L00N1-", "L00N2+"]
    alone = F.build_grid(tmp_path / "alone", corpus, S, per_pole_design)
    wider = F.build_grid(tmp_path / "wider", corpus, S, per_pole_design,
                         grid_poles=["L00N4-", "L00N2+", "L00N5+", "L00N0+", "L00N3+", "L00N1-"])
    e = hand_explain(tmp_path / "e", S)
    base = NV.grid_cluster_tables(alone, e)
    assert NV.grid_cluster_tables(wider, e) == base
    assert [r["birth_pct"] for p in S for r in base[p]] == [1, 4, 2, 7, 3, 8]   # the rows are told apart


def test_grid_missing_a_pole_of_the_explain_directory_is_refused(tmp_path):
    corpus = F.build_corpus(tmp_path, n_docs=60)
    S = ["L00N0+", "L00N1-", "L00N2+"]
    short = F.build_grid(tmp_path / "short", corpus, S, per_pole_design, grid_poles=["L00N2+", "L00N0+", "L00N4-"])
    e = hand_explain(tmp_path / "e", S)
    with pytest.raises(ValueError, match=r"T00: poles/poles.json lacks \['L00N1-'\]"):
        NV.grid_cluster_tables(short, e)
    full = F.build_grid(tmp_path / "full", corpus, S, per_pole_design, grid_poles=S + ["L00N4-"])
    (full[7] / "poles" / "poles.json").write_text(json.dumps(S[::-1] + ["L00N4-"]))   # T_7 lists another order
    with pytest.raises(ValueError, match=r"T07: poles/poles.json differs from .*T00"):
        NV.grid_cluster_tables(full, e)

def split_grid(root, corpus, s, grid_poles):
    """build_grid moved into the W3 layout: job a's prefix 1/out holds T_0..T_6 and T_15, job b's 1b/out
    T_7..T_14, each step as t<NN>_step<n>/. Returns (the two prefixes listed in turn, the grid in T order)."""
    flat = F.build_grid(root / "flat", corpus, s, design, grid_poles=grid_poles)
    for t, (d, step) in enumerate(zip(flat, NV.GRID_STEPS)):
        dest = root / "w3" / "grid" / ("1" if t <= 6 or t == 15 else "1b") / "out" / f"t{t:02d}_step{step}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(dest))
    listed = [p for job in ("1", "1b") for p in sorted((root / "w3" / "grid" / job / "out").iterdir())]
    return listed, listed[:7] + listed[8:] + [listed[7]]              # T_0..T_6 (a), T_7..T_14 (b), T_15 (a)


def test_w4_on_a_grid_split_over_two_job_prefixes(tmp_path):
    """The W3 layout: job a's prefix holds T_15 and T_0..T_6, job b's T_7..T_14, each step as
    out/t<NN>_step<n>/, and one grid pole list serves S_gate and S, two explain directories with disjoint
    positions lists. W4 in order: positions, clusters and judge on S_gate, judge-gate, then the S judge
    batch, then unblind; the grid is passed in T order across the two prefixes."""
    corpus = F.build_corpus(tmp_path)
    s_gate, s = ["L00N3+", "L00N4-"], ["L00N0+", "L00N1-", "L00N2+"]
    listed, grid = split_grid(tmp_path, corpus, s, ["L00N4-", *s, "L00N3+"])
    with pytest.raises(ValueError, match=r"t15_step1413814: branch 'stage1-step1413814', grid position expects stage1-step20000"):
        NV.check_grid(listed)                                         # the prefixes listed in turn put T_15 eighth
    NV.check_grid(grid)
    other = F.build_corpus(tmp_path / "other", seed=1)                # a corpus the grid did not score
    with pytest.raises(ValueError, match=r"scored the corpus"):
        NE.main(["render", "--harvest", str(grid[-1]), "--corpus", str(other), "--pole", "L00N0+", "--n", "1",
                 "--before", "64"])
    gate_out = tmp_path / "w4" / "o7_gate"
    s_right = {"L00N0+": 4, "L00N1-": 2, "L00N2+": 0}
    gate_dir = judged_dir(gate_out, grid, corpus, "w4j", {p: 4 for p in s_gate}, seed=6, gate=gate_out)
    early = tmp_path / "w4" / "o7_s_early"                            # S's judge batch before the gate is written
    with pytest.raises(ValueError, match=r"o7_gate/verdict/judge_gate.json does not exist"):
        judged_dir(early, grid, corpus, "w4j", s_right, seed=5, gate=gate_out)
    assert not llm.requests_path(early / "explain", "w4j-judge-s0").exists()
    assert "judge" not in {r for b in NV.read_json(early / "explain" / "manifest.json")["batches"] for r in b["roles"]}
    NV.main(["judge-gate", "--explain", str(gate_out), "--out", str(gate_out)])
    gate = NV.read_json(gate_out / "verdict" / "judge_gate.json")
    assert gate["gate"] == "pass" and gate["n_poles"] == 2 and gate["judge"]["models"] == ["test-model"]
    assert gate["declined_judgments"]["n"] == 0 and gate["declined_clusterer"] == {"n": 0, "poles": {}}
    s_out = tmp_path / "w4" / "o7_s"
    s_dir = judged_dir(s_out, grid, corpus, "w4j", s_right, seed=5, gate=gate_out)
    assert NV.read_json(gate_dir / "manifest.json")["judge_gates"]["w4j"]["self"] is True
    recorded = NV.read_json(s_dir / "manifest.json")["judge_gates"]["w4j"]
    assert recorded["self"] is False and recorded["gate_dir"] == str(gate_out)
    assert recorded["judge_gate_sha256"] == NV.file_sha256(gate_out / "verdict" / "judge_gate.json")
    for d, poles in ((gate_dir, s_gate), (s_dir, s)):
        assert NV.read_json(d / "positions_poles.json") == poles
        assert NV.read_json(d / "manifest.json")["positions"]["poles_sha256"] == NV.file_sha256(d.parent / "S.json")
        assert {p["cluster"].split(":")[0] for p in F.plans_by_id(d, "w4j-judge").values()} == set(poles)
    v = tmp_path / "v"
    (v / "verdict").mkdir(parents=True)
    (v / "verdict" / "theta.json").write_text(json.dumps({"theta_s": 0.2, "gate": "pass", "created_utc": "2026-10-01T00:00:00Z"}))
    journal = F.git_journal(tmp_path / "repo", "## PREDICTIONS 2026-10-01T00:00Z")
    NV.main(["unblind", "--journal", str(journal), "--theta", str(v / "verdict" / "theta.json"),
             "--judge-gate", str(gate_out / "verdict" / "judge_gate.json"), "--q2-grid", ",".join(map(str, grid)),
             "--q2-explain", str(s_out), "--seed", "0", "--out", str(v)])
    q2 = NV.read_json(v / "verdict" / "metrics.json")["q2"]
    assert q2["n_poles"] == 3 and q2["status_counts"]["late_primary"] == 3
    assert q2["p_u"]["a_ctrl"] == 1.0 and q2["p_u"]["A"] == pytest.approx(0.5)
    assert q2["center_birth_curve"] == NV.CENTER_NOT_COMPUTED          # a recorded absence (W3 scores T_15 only)
    grid_md = (v / "verdict" / "grid.md").read_text()
    assert f"| {NV.CENTER_NOT_COMPUTED} | not computed |" in grid_md
    assert "Left out of Q2 (declined after the attempts cap): 0 poles at the clusterer; 0 judge lines" in grid_md


def test_declined_clusterer_and_judge_lines_leave_q2_and_are_counted(tmp_path):
    """Declined rule (2026-10-21): a pole whose clustering sample is declined at the attempts cap leaves
    Q2; a cluster whose judgment is declined leaves p_u, a pole whose control judgment is declined leaves
    a_ctrl. prepare-judge, judge-gate and the Q2 readout go on with the rest and report the ids."""
    corpus = F.build_corpus(tmp_path)
    poles = [f"L00N{n}{s}" for n in range(4) for s in "+-"]
    grid = F.build_grid(tmp_path / "grid", corpus, poles, design)
    out = tmp_path / "d"
    out.mkdir()
    (out / "S.json").write_text(json.dumps(poles))
    NE.main(["positions", "--harvest", str(grid[-1]), "--out", str(out), "--poles", str(out / "S.json")])
    NE.main(["prepare-clusters", "--harvest", str(grid[-1]), "--corpus", str(corpus), "--out", str(out),
             "--batch", "c", "--model", "test-model", "--seed", "1"])
    explain = out / "explain"
    cplans = F.plans_by_id(explain, "c-clusterer")
    bad = next(rid for rid, p in cplans.items() if p["pole"] == "L00N3+" and p["sample"] == 1)

    def cluster(req):
        x = F.two_halves(cplans[req["id"]])
        if req["id"] == bad:                                          # rank 24 twice: refused, then declined
            x["clusters"][1]["exemplar_ranks"] = [24] + x["clusters"][1]["exemplar_ranks"]
        return x
    for s in (0, 1):
        F.answer_batch(explain, f"c-clusterer-s{s}", cluster)
    one = ["--out", str(out), "--backend", "manual", "--max-attempts", "1"]
    NE.main(["reject", *one, "--batch", "c", "--roles", "clusterer"])
    NE.main(["collect-clusters", "--out", str(out), "--batch", "c", "--backend", "manual"])
    gone = NV.declined_clusterer(explain)
    assert list(gone) == ["L00N3+"] and gone["L00N3+"].startswith("sample 1: clusterer declined (")
    assert "L00N3+" not in NV.clusters_by_pole(explain)
    NE.main(["prepare-judge", "--grid", ",".join(map(str, grid)), "--corpus", str(corpus), "--out", str(out),
             "--batch", "j", "--model", "test-model", "--seed", "2", "--gate", str(out)])
    plans = F.plans_by_id(explain, "j-judge")
    assert {p["cluster"].split(":")[0] for p in plans.values()} == set(poles) - {"L00N3+"}
    ctrl_line = next(i for i, p in plans.items() if p["control"] and p["cluster"].startswith("L00N1-:")
                     and p["decoy_type"] == "random")
    test_line = next(i for i, p in plans.items() if not p["control"] and p["cluster"] == "L00N2+:c1"
                     and p["decoy_type"] == "hard")

    def judge(req):
        if req["id"] in (ctrl_line, test_line):
            return {"answer": "C", "reason": "r"}                       # outside the schema: refused, then declined
        return {"answer": "A" if plans[req["id"]]["shown_order"][0] == "true" else "B", "reason": "r"}
    for s in (0, 1):
        F.answer_batch(explain, f"j-judge-s{s}", judge)
    NE.main(["reject", *one, "--batch", "j", "--roles", "judge"])
    NE.main(["collect-judge", "--out", str(out), "--batch", "j", "--backend", "manual"])
    rows = {j["id"]: j for j in llm.read_jsonl(explain / "judgments.jsonl")}
    assert len(rows) == len(plans) and rows[ctrl_line]["correct"] is None and rows[ctrl_line]["declined"]
    NV.main(["judge-gate", "--explain", str(out), "--out", str(out)])
    gate = NV.read_json(out / "verdict" / "judge_gate.json")
    assert gate["n_poles"] == 6 and gate["a_ctrl"] == 1.0              # 8 - L00N3+ (clusterer) - L00N1- (control)
    assert list(gate["declined_judgments"]["ids"]) == [ctrl_line] and gate["declined_judgments"]["control_poles"] == ["L00N1-"]
    assert list(gate["declined_clusterer"]["poles"]) == ["L00N3+"]
    q2, _ = NV.q2_readouts(grid, explain, gate, 0, False)
    assert (q2["n_positions"], q2["n_poles"], q2["declined_clusterer"]["n"]) == (8, 7, 1)
    d = q2["declined_judgments"]
    assert d["n"] == 2 and sorted(d["ids"]) == sorted([ctrl_line, test_line])
    assert d["clusters"] == ["L00N2+:c1"] and d["control_poles"] == ["L00N1-"]
    assert q2["p_u"]["n_clusters"] == 6 and q2["p_u"]["n_poles"] == 7 and q2["p_u"]["a_ctrl"] == 1.0
    assert q2["flags"]["n_flagged"] == 0


# --- unblind end to end on synthetic banked files --------------------------------------------

POLES = ["L00N0+", "L00N0-", "L00N1+", "L00N2-", "L01N0+", "L01N1-", "L01N3+", "L01N4-"]
PLAN = {"L00N0+": ("ONE", "pass"), "L00N0-": ("ONE", "fail"), "L00N1+": ("TWO-REL", "pass"),
        "L00N2-": ("UNREL", "both"), "L01N0+": ("UNREL", "one"), "L01N1-": ("UNK", "pass"),
        "L01N3+": ("ONE", "pass"), "L01N4-": ("ONE", "pass")}
EXPECTED = {"L00N0+": "ONE", "L00N0-": "UNK", "L00N1+": "TWO-REL", "L00N2-": "UNREL", "L01N0+": "UNK",
            "L01N1-": "UNK", "L01N3+": "ONE", "L01N4-": "RARE"}


def explanation(req):
    pole = req["id"].split(":")[1]
    label = PLAN[pole][0]
    jobs = {"ONE": J(False), "TWO-REL": J(True, True), "UNREL": J(False, False), "UNK": J(False)}[label]
    confidence = 0.5
    if req["role"] == "second_reader":
        confidence = 0.4
        if pole == "L01N3+":
            label = "UNK"
    text = ("fires on the first token of a document; this cannot be reproduced in a short text"
            if pole == "L01N1-" else "fires on boats at sea")
    return {"explanation": text, "jobs": jobs, "class": label, "confidence": confidence}


def writer(req):
    return {"positives": ["the boat sailed"] * 5, "near_misses": ["the boat in a park"] * 5, "distant_concept": "bread",
            "distant": ["the bread on the hill"] * 5, "max": "the ship sails the sea"}


def explained_dir(out, harvest, corpus, poles_file, probe_dir):
    """prepare -> answers -> collect -> writer -> texts -> designed probe scores."""
    NE.main(["prepare", "--harvest", str(harvest), "--corpus", str(corpus), "--out", str(out), "--batch", "a",
             "--roles", "explainer,second_reader", "--poles", str(poles_file), "--model", "test-model"])
    explain = out / "explain"
    for role in ("explainer", "second_reader"):
        F.answer_batch(explain, f"a-{role}", explanation)
    NE.main(["collect", "--out", str(out), "--batch", "a", "--roles", "explainer,second_reader", "--backend", "manual"])
    NE.main(["prepare-writer", "--out", str(out), "--batch", "w", "--model", "test-model"])
    F.answer_batch(explain, "w-writer", writer)
    F.answer_batch(explain, "w-writer_job", lambda r: {"positives": ["rain on the town"] * 5})
    NE.main(["collect-writer", "--out", str(out), "--batch", "w", "--backend", "manual",
             "--harvest", str(harvest), "--corpus", str(corpus), "--seed", "11"])

    def activation(t, pole):
        """The explained pole fires per PLAN; every other pole gets a deterministic value in
        [0, 0.2) of its max|a| (the shuffle null)."""
        _, _, s = NV.parse_pole(pole)
        max_abs = NV.pole_stats(harvest, pole)["max_abs"]
        v = (sum(map(ord, t["text_id"] + pole)) % 97) / 97 * 0.2
        if pole == t["pole"]:
            outcome = PLAN[pole][1]
            v = {"pos": 0.8, "max": 1.2 if pole == "L00N0+" else 0.9, "distant": 0.0,
                 "near": 0.8 if outcome == "fail" else 0.05}.get(t["kind"])
            if t["kind"] == "job_pos":
                v = 0.8 if outcome == "both" or t["job"] == 0 else 0.05
        return (v * max_abs, -0.01) if s == 0 else (0.01, -v * max_abs)
    F.write_probe(probe_dir, llm.read_jsonl(explain / "texts.jsonl"), NV.read_json(explain / "probe_poles.json"),
                  activation)
    return explain


@pytest.fixture(scope="module")
def q1_bank(tmp_path_factory):
    """Discovery directory, θ_s, then the audit directory (after θ_s, as the run orders them)."""
    root = tmp_path_factory.mktemp("q1")
    corpus = F.build_corpus(root)
    harvest = F.build_harvest(root / "harvest", corpus, poles=POLES)
    st = NV.load_npz(harvest / "stats" / "layer_01.npz")
    st["density"][4, 1] = 10                                          # L01N4- is RARE
    F.set_stats(harvest, 1, density=st["density"])
    (root / "poles.json").write_text(json.dumps(POLES))
    explained_dir(root / "disc", harvest, corpus, root / "poles.json", root / "probe_disc")
    NV.main(["theta", "--stats", str(harvest), "--probe", str(root / "probe_disc"), "--explain", str(root / "disc"),
             "--out", str(root / "v")])
    explained_dir(root / "w2", harvest, corpus, root / "poles.json", root / "probe")
    journal = F.git_journal(root / "repo", "## PREDICTIONS 2026-10-01T00:00Z")
    return root, harvest, journal


def test_discovery_report_matches_q1_metrics_on_the_same_records(q1_bank, tmp_path):
    root, harvest, _ = q1_bank
    theta_path = root / "v" / "verdict" / "theta.json"
    args = ["discovery", "--theta", str(theta_path), "--stats", str(harvest), "--probe", str(root / "probe_disc"),
            "--explain", str(root / "disc"), "--out", str(tmp_path)]
    NV.main(args)
    report = NV.read_json(tmp_path / "verdict" / "discovery.json")
    assert set(report) == {"set", "created_utc", "theta", "pass_rate", "fractions", "declined", "agreement",
                           "by_explainer_model", "X7_pass_rate_by_encoding_flag", "strong_rates",
                           "X3_blind_max_ge_corpus_max", "prompts", "batches", "inputs"}
    assert list(report["by_explainer_model"]) == ["test-model"]                  # one explainer model here
    assert report["by_explainer_model"]["test-model"]["fractions"] == report["fractions"]
    assert report["agreement"]["kappa_models"]["cross_model_pairs"] == 0
    assert report["declined"] == {"n": 0, "poles": {}, "second_reader": 0}
    assert report["inputs"]["probe_rule"] == F.PROBE_RULE and report["inputs"]["probe_excluded_positions"] == "text position 0"
    assert report["set"] == "discovery" and report["theta"]["created_utc"] == NV.read_json(theta_path)["created_utc"]
    theta = report["theta"]["theta_s"]
    recs = NV.q1_records(harvest, root / "probe_disc", root / "disc" / "explain", theta)
    q1 = json.loads(json.dumps(NV.q1_metrics(recs, theta)))
    for key in ("fractions", "X7_pass_rate_by_encoding_flag", "strong_rates", "X3_blind_max_ge_corpus_max"):
        assert report[key] == q1[key]
    passes = [r["passes"] for r in recs if "passes" in r and not r["rare"]]
    # every non-RARE pole but L00N0- (its near misses fire) passes §6.1; passing is not the class
    # (L01N0+ passes but is UNK, its second job unsupported; L01N1- passes but the explainer said UNK)
    assert (report["pass_rate"]["k"], report["pass_rate"]["n"]) == (sum(passes), len(passes)) == (6, 7)
    assert report["prompts"]["version"] == "v3" and set(report["prompts"]["sha256"]) == {"explainer", "second_reader", "writer"}
    assert report["batches"]["a-explainer"] == {"requests": 8, "responses": 8, "rejected_lines": 0, "declined": 0,
                                                "requested_models": ["test-model"],
                                                "backend": "manual (no backend sidecar)", "reported_models": None}
    samples = llm.read_jsonl(tmp_path / "verdict" / "discovery_samples.jsonl")
    assert {r["pole"]: r["class"] for r in samples} == EXPECTED
    assert next(r for r in samples if r["pole"] == "L00N0-")["strong"] == {"pos": 5, "near": 5, "distant": 0, "max": 1}
    with pytest.raises(FileExistsError):
        NV.main(args)


def test_unblind_q1_end_to_end(q1_bank, tmp_path, capsys):
    root, harvest, journal = q1_bank
    theta_path = root / "v" / "verdict" / "theta.json"
    theta = NV.read_json(theta_path)
    # shuffle values lie in [0, 0.2); the designed weak (0.05) and strong (0.8) texts straddle θ_s
    assert theta["gate"] == "pass" and theta["n_scores"] == 8 * 15 and 0.05 < theta["theta_s"] < 0.2
    assert theta["created_utc"] and len(theta["inputs"]["texts"]["sha256"]) == 64
    assert theta["probe"] == {"probe_rule": F.PROBE_RULE, "probe_excluded_positions": "text position 0"}
    v = tmp_path / "v"
    args = ["unblind", "--journal", str(journal), "--theta", str(theta_path),
            "--q1", f"TOY={harvest},{root / 'probe'},{root / 'w2'}", "--seed", "0", "--out", str(v)]
    NV.main(args)
    metrics = NV.read_json(v / "verdict" / "metrics.json")
    q1 = metrics["q1"]["TOY"]
    poles = llm.read_jsonl(v / "verdict" / "samples" / "q1_TOY_poles.jsonl")
    assert {r["pole"]: r["class"] for r in poles} == EXPECTED                       # every pole
    assert {c: q1["fractions"][c]["k"] for c in NV.CLASSES} == {"ONE": 2, "TWO-REL": 1, "UNREL": 1, "UNK": 3, "RARE": 1}
    # §2.3 as amended: UNK = UNK-N (explainer said UNK) + UNK-F (an explanation was given and failed)
    assert (q1["fractions"]["UNK-N"]["k"], q1["fractions"]["UNK-F"]["k"]) == (1, 2)
    assert {r["pole"]: r["unk_part"] for r in poles if r["class"] == "UNK"} == \
        {"L01N1-": "UNK-N", "L00N0-": "UNK-F", "L01N0+": "UNK-F"}
    assert q1["unk_reasons"] == {"explainer said UNK": 1, "failed §6.1": 1, "UNREL support clause failed": 1}
    # X3: one of the two ONE poles has a blind maximum above the corpus max
    assert (q1["X3_blind_max_ge_corpus_max"]["ONE"]["k"], q1["X3_blind_max_ge_corpus_max"]["ONE"]["n"]) == (1, 2)
    assert q1["X3_blind_max_ge_corpus_max"]["UNK-F"]["k"] == 0
    # DATA_CONTRACT §8: pos0_share banked per pole (fixture value = (2 * neuron + sign) / 100)
    assert {r["pole"]: r["pos0_share"] for r in poles}["L01N3+"] == pytest.approx(0.06)
    assert q1["pos0_share_by_class"]["ONE"]["n"] == 2 and q1["pos0_share_by_class"]["RARE"]["quantiles_0_25_50_75_100"][2] == pytest.approx(0.09)
    # X7: no encodings in the toy corpus, so every pole is unflagged
    assert q1["X7_pass_rate_by_encoding_flag"]["flagged"] == {"n": 0}
    assert q1["X7_pass_rate_by_encoding_flag"]["unflagged"]["n"] == 8
    assert q1["kappa_label"] < 1 and q1["n_rated_twice"] == 8
    assert q1["strong_rates"]["pos"]["n"] == 35 and q1["strong_rates_including_rare"]["pos"]["n"] == 40
    assert q1["strong_rates"]["pos"]["rate"] == 1.0 and q1["strong_rates"]["distant"]["rate"] == 0.0
    assert q1["essay_rule"] == {"win": 7, "tie": 1, "lose": 0}
    assert all(t["strong"] == (t["score"] >= theta["theta_s"]) for r in poles if "texts" in r
               for rows in r["texts"].values() for t in rows)
    assert metrics["G2"].startswith("PENDING") and "holm" not in metrics
    grid = (v / "verdict" / "grid.md").read_text()
    assert "| Q1 fractions (descriptive)" in grid and "PENDING (no Q2 input)" in grid and "(UNK-N 1, UNK-F 2)" in grid
    gallery = llm.read_jsonl(v / "verdict" / "samples" / "q1_TOY.jsonl")
    one_pass = next(s for s in gallery if s["class"] == "ONE" and s["which"] == "passing")
    assert "⟦" in one_pass["explainer_saw"] and len(one_pass["texts"]["pos"]) == 5
    with pytest.raises(FileExistsError):
        NV.main(args)                                                 # the grid is filled once
    bad = tmp_path / "theta_fail.json"
    bad.write_text(json.dumps({**theta, "gate": "fail_ceiling"}))
    with pytest.raises(SystemExit) as e:
        NV.main([*args[:3], "--theta", str(bad), *args[5:-1], str(tmp_path / "v2")])
    assert e.value.code == 2 and "theta_fail.json" in capsys.readouterr().err
    late = tmp_path / "theta_late.json"                               # θ_s written after the audit started
    late.write_text(json.dumps({**theta, "created_utc": "2100-01-01T00:00:00Z"}))
    with pytest.raises(SystemExit) as e:
        NV.main([*args[:3], "--theta", str(late), *args[5:-1], str(tmp_path / "v3")])
    assert e.value.code == 2 and "after" in capsys.readouterr().err


def test_unblind_q2_end_to_end(q2_bank, tmp_path, capsys):
    root, _, grid, dev, s_dir = q2_bank
    v = tmp_path / "v"
    (v / "verdict").mkdir(parents=True)
    (v / "verdict" / "theta.json").write_text(json.dumps({"theta_s": 0.2, "gate": "pass", "created_utc": "2026-10-01T00:00:00Z"}))
    t0 = tmp_path / "t0" / "explain"
    t0.mkdir(parents=True)
    llm.write_jsonl_new(t0 / "explanations.jsonl",
                        [{"pole": p, "sample": 0, "confidence": 0.2} for p in ("L00N0+", "L00N1-", "L00N2+")])
    journal = F.git_journal(tmp_path / "repo", "## PREDICTIONS 2026-10-01T00:00Z")
    args = ["unblind", "--journal", str(journal), "--theta", str(v / "verdict" / "theta.json"),
            "--judge-gate", str(root / "dev" / "verdict" / "judge_gate.json"), "--q2-grid", ",".join(map(str, grid)),
            "--q2-explain", str(s_dir.parent), "--t0-explanations", str(t0.parent), "--seed", "0", "--out", str(v)]
    NV.main(args)
    metrics = NV.read_json(v / "verdict" / "metrics.json")
    q2 = metrics["q2"]
    n_echo = sum(bool(p["double_negative_ranks"]) for p in F.plans_by_id(s_dir, "c-clusterer").values() if p["sample"] == 0)
    assert q2["echo_clusters"] == n_echo and q2["poles_all_echo"] == []
    assert q2["status_counts"]["late_primary"] == 3 and q2["status_counts"]["earliest"] == 3
    assert q2["p_u"]["A"] == pytest.approx(0.5) and q2["p_u"]["a_ctrl"] == 1.0 and q2["f_ctrl"] == 0.0
    assert q2["p_u"]["point"] == pytest.approx(1.0) and q2["p_u"]["underpowered"] and "p_RG" not in q2["p_u"]
    assert q2["flags"]["n_flagged"] == 1 and q2["H_MIG"]["count"] == 0
    assert q2["hard"]["accuracy"] == 1.0 and q2["hard"]["measurable"]
    assert q2["birth_order"]["b"] == 1.0 and q2["birth_order"]["ci95"] == [1.0, 1.0] and q2["vestigial"]["v"] == 0.0
    assert q2["vestigial"]["n_excluded_growth_undefined"] == 0
    assert set(metrics["holm"]) == {"b", "v"}                         # always both, never p_u (§7)
    assert q2["r_at_birth"]["median_late_primary"] == pytest.approx(0.5)
    assert metrics["G2"]["share"] == 0.0 and metrics["G2"]["within_line"]
    assert q2["population_weighted"].startswith("PENDING")
    grid_md = (v / "verdict" / "grid.md").read_text()
    assert "| `p_u` (§6.3)" in grid_md and "Holm" in grid_md
    parent = llm.read_jsonl(v / "verdict" / "samples" / "q2_parent.jsonl")
    assert any(s.get("correct") is False and "Record A" in s["judge_saw"] for s in parent)
    clusters = llm.read_jsonl(v / "verdict" / "samples" / "q2_clusters.jsonl")
    assert all(len(r["judgments"]) == 6 for r in clusters if r["status"] == "late_primary")
    # population weights from Q1 classes (§4.4)
    gate = NV.read_json(root / "dev" / "verdict" / "judge_gate.json")
    strata = {"L00N0+": "ONE", "L00N1-": "UNREL", "L00N2+": "UNK"}
    weighted, _ = NV.q2_readouts(grid, s_dir, gate, 0, False, strata,
                                 {"ONE": 90, "TWO-REL": 7, "UNREL": 2, "UNK": 1})
    pw = weighted["population_weighted"]
    assert pw["strata"]["unrepresented_strata"] == ["TWO-REL"]
    assert pw["p_u"]["A"] == pytest.approx((90 * 1.0 + 2 * 0.5 + 1 * 0.0) / 93)
    # judge validity gate failed: the parent test is not read
    failed, _ = NV.q2_readouts(grid, s_dir, {"gate": "fail", "a_ctrl": 0.7}, 0, False)
    assert failed["p_u"]["valid"] is False and failed["H_MIG"]["valid"] is False
    assert failed["birth_order"]["b"] == 1.0                          # birth order is still read
    # a judge gate written after the S judge batch is refused
    late = tmp_path / "gate_late.json"
    late.write_text(json.dumps({**gate, "created_utc": "2100-01-01T00:00:00Z"}))
    with pytest.raises(SystemExit) as e:
        NV.main([*args[:5], "--judge-gate", str(late), *args[7:-1], str(tmp_path / "v2")])
    assert e.value.code == 2 and "after the first S judge batch" in capsys.readouterr().err
    # a gate measured on another judge (model or prompt) does not validate the S judgments
    for judge in ({**gate["judge"], "models": ["claude-opus-5-5"]}, {**gate["judge"], "prompt_sha256": "0" * 64}):
        other = tmp_path / "gate_other_judge.json"
        other.write_text(json.dumps({**gate, "judge": judge}))
        with pytest.raises(SystemExit) as e:
            NV.main([*args[:5], "--judge-gate", str(other), *args[7:-1], str(tmp_path / "v3")])
        assert e.value.code == 2 and "the gate measured the judge" in capsys.readouterr().err


# --- W5: unblind on the W4 layout, figures F1-F6, RESULTS.md --------------------------------------

@pytest.fixture(scope="module")
def w5_bank(q1_bank, tmp_path_factory):
    """One unblind on the W4 layout: three Q1 cells (the q1_bank audit directory under three keys; O7 read over
    a 6-pole list, as the 600 audit poles, and giving the strata), S_gate and S on a grid split over two job
    prefixes. S holds one ONE, one TWO-REL and one UNREL pole of the list, so F6 has a pole per group."""
    root, harvest, journal = q1_bank
    w = tmp_path_factory.mktemp("w5")
    corpus = F.build_corpus(w)
    s_gate, s = ["L00N3+", "L00N4-"], ["L00N0+", "L00N1+", "L00N2-"]
    _, grid = split_grid(w, corpus, s, ["L00N4-", *s, "L00N3+"])
    gate_out, s_out = w / "w4" / "o7_gate", w / "w4" / "o7_s"
    judged_dir(gate_out, grid, corpus, "w4j", {p: 4 for p in s_gate}, seed=6, gate=gate_out)
    NV.main(["judge-gate", "--explain", str(gate_out), "--out", str(gate_out)])
    judged_dir(s_out, grid, corpus, "w4j", {"L00N0+": 4, "L00N1+": 2, "L00N2-": 0}, seed=5, gate=gate_out)
    cell = f"{harvest},{root / 'probe'},{root / 'w2'}"
    (w / "audit_poles.json").write_text(json.dumps([p for p in POLES if p not in ("L01N1-", "L01N4-")]))
    out = w / "w5"
    NV.main(["unblind", "--journal", str(journal), "--theta", str(root / "v" / "verdict" / "theta.json"),
             "--q1", f"Q4B={cell}", "--q1", f"G4={cell}", "--q1", f"O7={cell},poles={w / 'audit_poles.json'}",
             "--q2-strata", "O7",
             "--judge-gate", str(gate_out / "verdict" / "judge_gate.json"), "--q2-grid", ",".join(map(str, grid)),
             "--q2-explain", str(s_out), "--seed", "0", "--out", str(out)])
    return out


def test_w5_metrics_carry_every_number_the_figures_cite(w5_bank):
    m = NV.read_json(w5_bank / "verdict" / "metrics.json")
    q2 = m["q2"]
    assert q2["births"]["main"]["counts"][1] == 3 and q2["births"]["non_main"]["counts"][4] == 3   # design: T_1, T_4
    assert q2["births"]["main"]["never_born"] == 0 and q2["births"]["non_main"]["n"] == 3
    pts = q2["grid_population"]["per_grid_point"]
    assert [p["step"] for p in pts] == list(NV.GRID_STEPS) and pts[0]["n_poles"] == 12 and pts[0]["dead_share"] == 0.0
    assert pts[0]["density"] == [500.0] * 5 and len(pts[0]["double_negative_share"]) == 5
    groups = {e["group"]: e for e in q2["gallery"]}
    assert {g: e["pole"] for g, e in groups.items()} == {"ONE": "L00N0+", "TWO-REL": "L00N1+", "UNREL or UNK": "L00N2-"}
    main = next(c for c in groups["ONE"]["clusters"] if c["is_main"])
    assert main["birth_pct"] == 1 and len(main["exemplars"]) == 2 and all("⟦" in x for x in main["exemplars"])
    assert m["q1"]["Q4B"]["untestable_phrase"]["k"] == 1 and m["q1"]["Q4B"]["untestable_phrase"]["n"] == 8
    assert m["q1"]["O7"]["untestable_phrase"]["k"] == 0                # L01N1- is outside the O7 list
    assert m["q1"]["O7"]["poles_list"]["excluded_by_list"] == {"n": 2, "poles": ["L01N1-", "L01N4-"]}
    assert m["q2"]["population_weighted"]["strata"]["population"] == {"ONE": 2, "TWO-REL": 1, "UNREL": 1, "UNK": 2}
    assert q2["center_birth_curve"] == NV.CENTER_NOT_COMPUTED


def test_figures_render_to_scale_from_metrics_alone(tmp_path, monkeypatch):
    """Since 2026-11-05 F4 adds the chance level of random same-size groups, computed on the seventh unblind run
    (studies/already-clean/runbook/f4_births.py), so the figures are drawn from that run's metrics.json: the local bank
    copy the script reads. Run as `--unblind bank/w5`, the F4 written equals the committed one byte for byte."""
    (tmp_path / "bank" / "w5" / "verdict").mkdir(parents=True)
    shutil.copy(FG.load_f4_births().METRICS, tmp_path / "bank" / "w5" / "verdict" / "metrics.json")
    monkeypatch.chdir(tmp_path)
    unblind = Path("bank/w5")
    paths = FG.build(unblind, Path("figures"))
    assert [p.name for p in paths] == [f"{n}.svg" for n in FG.FIGURES]
    source = "bank/w5/verdict/metrics.json"
    texts = {}
    for p in paths:
        root = ET.fromstring(p.read_text(encoding="utf-8"))           # well-formed SVG
        texts[p.stem] = " ".join(t.text or "" for t in root.iter("{http://www.w3.org/2000/svg}text"))
        assert f"source: {source}" in texts[p.stem]
        bars = [(float(r.get("height")), float(r.get("data-value"))) for r in root.iter("{http://www.w3.org/2000/svg}rect")
                if r.get("data-value") is not None and float(r.get("data-value")) > 0]
        if p.stem in ("F1_audit_classes", "F2_strong_rates", "F4_births", "F5_parent_test"):
            h0, v0 = max(bars, key=lambda b: b[1])                       # one scale per figure (heights to 0.01 px)
            assert bars and all(abs(h - v * h0 / v0) <= 0.01 for h, v in bars), p.stem
    m = NV.read_json(source)
    one = m["q1"]["O7"]["fractions"]["ONE"]
    assert f"O7 {100 * one['p']:.1f}%" in texts["F1_audit_classes"]
    pu = m["q2"]["p_u"]
    assert f"p_u = {pu['point']:.3f} [{pu['ci95'][0]:.3f}, {pu['ci95'][1]:.3f}]: {pu['rule']}" in texts["F5_parent_test"]
    assert "chance level: random same-size groups" in texts["F4_births"] and "so b is below chance" in texts["F4_births"]
    committed = FG.F4_BIRTHS.parents[1] / "results" / "figures" / "F4_births.svg"
    assert paths[FG.FIGURES.index("F4_births")].read_bytes() == committed.read_bytes()
    assert all(f"{e['pole']} ({e['class']})" in texts["F6_gallery"] for e in m["q2"]["gallery"] if e["pole"])
    with pytest.raises(FileExistsError):
        FG.build(unblind, Path("figures"))


def test_figures_refuse_missing_inputs_and_write_nothing(w5_bank, tmp_path):
    m = NV.read_json(w5_bank / "verdict" / "metrics.json")
    seventh = NV.read_json(FG.load_f4_births().METRICS)      # F4 draws only the births its chance level was computed on
    for base, change, key in ((seventh, lambda x: x["q2"].update(gallery="PENDING (run with --q2-strata)"), "q2.gallery"),
                              (m, lambda x: x["q2"].pop("grid_population"), "q2.grid_population"),
                              (m, lambda x: x.pop("q2"), "no q2")):
        broken = json.loads(json.dumps(base))
        change(broken)
        d = tmp_path / key.replace(" ", "_")
        (d / "verdict").mkdir(parents=True)
        (d / "verdict" / "metrics.json").write_text(json.dumps(broken))
        with pytest.raises(ValueError, match=key.replace(".", r"\.")):
            FG.build(d, d / "figures")
        assert not (d / "figures").exists()
    with pytest.raises(AssertionError, match="births"):        # the synthetic bank's births are not the seventh run's
        FG.build(w5_bank, tmp_path / "synthetic")
    assert not (tmp_path / "synthetic").exists()


def test_results_template_lays_out_q1_tables_grid_declines_and_blank_costs(w5_bank, tmp_path):
    out = tmp_path / "RESULTS.md"
    NV.main(["results-template", "--unblind", str(w5_bank), "--out", str(out)])
    text = out.read_text()
    m = NV.read_json(w5_bank / "verdict" / "metrics.json")
    one = m["q1"]["O7"]["fractions"]["ONE"]
    assert (f"On model O7, {100 * one['p']:.1f}% of poles ({100 * one['wilson95'][0]:.1f}–{100 * one['wilson95'][1]:.1f}%, "
            "Wilson 95%) have one job under explanations that pass §6.1 (n = 6).") in text
    assert "2 poles of the explain directory left out by the list: L01N1-, L01N4-." in text
    assert "Poles: every pole of the explain directory (no list given)." in text
    assert "| depth tercile 0 |" in text and "| explainer test-model |" in text and "### Q4B" in text
    assert "| double-negative share < 0.8 | 6 |" in text
    assert "Label agreement: kappa" in text and "second reader {'test-model': 8}" in text
    assert "Kappa by explainer model: test-model 0.822 (n = 8)." in text
    assert "Declined at W4: 0 poles at the clusterer (none); 0 judge lines" in text
    assert "| `p_u` (§6.3) |" in text and NV.CENTER_NOT_COMPUTED in text
    assert "Q4B explanations naming an untestable job: 1/8" in text and "O7 X7 (pass rate by encoding flag)" in text
    assert "- GPU: ____" in text and "- API: ____" in text and "Finding: ____" in text
    with pytest.raises(FileExistsError):
        NV.main(["results-template", "--unblind", str(w5_bank), "--out", str(out)])


def test_q1_cell_with_a_pole_list_reads_only_the_listed_poles(q1_bank, tmp_path):
    """§6.2: the O7 readout is over the 600 audit poles; the explain directory's other poles (the S_gate draw)
    are counted as excluded_by_list with their ids, and the strata come from the listed set."""
    root, harvest, journal = q1_bank
    listed = ["L00N0+", "L00N0-", "L00N1+", "L00N2-", "L01N0+", "L01N3+"]       # 6 of the 8 poles
    (tmp_path / "audit_poles.json").write_text(json.dumps(listed))
    cell = f"{harvest},{root / 'probe'},{root / 'w2'}"
    args = ["unblind", "--journal", str(journal), "--theta", str(root / "v" / "verdict" / "theta.json"),
            "--q1", f"ALL={cell}", "--q1", f"O7={cell},poles={tmp_path / 'audit_poles.json'}", "--seed", "0"]
    NV.main([*args, "--out", str(tmp_path / "v")])
    q1 = NV.read_json(tmp_path / "v" / "verdict" / "metrics.json")["q1"]
    lst = q1["O7"]["poles_list"]
    assert lst["n"] == 6 and lst["sha256"] == NV.file_sha256(tmp_path / "audit_poles.json")
    assert lst["excluded_by_list"] == {"n": 2, "poles": ["L01N1-", "L01N4-"]}
    assert q1["O7"]["fractions"]["n"] == 6 and q1["ALL"]["fractions"]["n"] == 8 and q1["ALL"]["poles_list"] is None
    want = {c: sum(EXPECTED[p] == c for p in listed) for c in NV.CLASSES}
    assert {c: q1["O7"]["fractions"][c]["k"] for c in NV.CLASSES} == want
    assert q1["O7"]["n_rated_twice"] == 6
    assert {r["pole"] for r in llm.read_jsonl(tmp_path / "v" / "verdict" / "samples" / "q1_O7_poles.jsonl")} == set(listed)
    (tmp_path / "bad.json").write_text(json.dumps(listed + ["L05N9+"]))
    with pytest.raises(ValueError, match=r"1 listed poles have no Q1 record: \['L05N9\+'\]"):
        NV.main([*args[:-4], "--q1", f"O7={cell},poles={tmp_path / 'bad.json'}", "--seed", "0", "--out", str(tmp_path / "v2")])
    with pytest.raises(ValueError, match="expected KEY=STATS,PROBE,EXPLAIN"):
        NV.main([*args[:-4], "--q1", f"O7={cell},{tmp_path / 'bad.json'}", "--seed", "0", "--out", str(tmp_path / "v3")])


def test_fork_clusters_gates_another_judge_on_the_same_clusters(tmp_path):
    """§4.4: a judge is tuned on S_gate at W4. fork-clusters copies the collected clusters byte for byte
    without the first judge batch or its verdict, and the fork is a fresh gate directory for another judge."""
    corpus = F.build_corpus(tmp_path)
    grid = F.build_grid(tmp_path / "grid", corpus, S, design)
    src = tmp_path / "gate"
    explain = judged_dir(src, grid, corpus, "jf", {p: 4 for p in S}, seed=6, gate=src)
    NV.main(["judge-gate", "--explain", str(src), "--out", str(src)])
    with pytest.raises(ValueError, match="no clusters.jsonl"):
        NE.main(["fork-clusters", "--source", str(tmp_path / "grid"), "--out", str(tmp_path / "x")])
    fork = tmp_path / "gate_judge2"
    NE.main(["fork-clusters", "--source", str(src), "--out", str(fork)])
    fe = fork / "explain"
    for name in ("clusters.jsonl", "positions.npz", "positions_poles.json", "plans/c-clusterer.jsonl",
                 "requests/c-clusterer-s0.jsonl", "responses/c-clusterer-s1.jsonl"):
        assert (fe / name).read_bytes() == (explain / name).read_bytes(), name
    assert not (fork / "verdict").exists() and not (fe / "judgments.jsonl").exists()
    assert not list((fe / "plans").glob("*-judge.jsonl")) and not list((fe / "requests").glob("*judge*"))
    assert {r["role"] for r in llm.read_jsonl(fe / "calls.jsonl")} == {"clusterer"}
    m = NV.read_json(fe / "manifest.json")
    assert m["forked_from"]["manifest_sha256"] == NV.file_sha256(explain / "manifest.json")
    assert m["forked_from"]["dropped_batches"] == ["jf-judge-s0", "jf-judge-s1"] and "judge_gates" not in m
    assert not [b for b in m["batches"] if "judge" in b["roles"]] and "judge:jf" not in m["seeds"]
    assert set(m["models_by_role"]["answered"]) == {"clusterer"}
    NE.main(["prepare-judge", "--grid", ",".join(map(str, grid)), "--corpus", str(corpus), "--out", str(fork),
             "--batch", "j2", "--model", "other-model", "--seed", "7", "--gate", str(fork)])
    plans = F.plans_by_id(fe, "j2-judge")
    for s in (0, 1):
        F.answer_batch(fe, f"j2-judge-s{s}", lambda r: {"answer": "A" if plans[r["id"]]["shown_order"][0] == "true"
                                                         else "B", "reason": "r"}, model="other-model")
    NE.main(["collect-judge", "--out", str(fork), "--batch", "j2", "--backend", "manual"])
    NV.main(["judge-gate", "--explain", str(fork), "--out", str(fork)])
    gate = NV.read_json(fork / "verdict" / "judge_gate.json")
    assert gate["judge"]["models"] == ["other-model"] and gate["a_ctrl"] == 1.0 and gate["n_poles"] == 3
    with pytest.raises(FileExistsError):
        NE.main(["fork-clusters", "--source", str(src), "--out", str(fork)])
