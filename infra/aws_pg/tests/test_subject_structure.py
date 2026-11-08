"""Weight-free test for conserved-core Tier 2 (per-subject policy structure). No datasets, no GPU.

    python3 -m pytest infra/aws_pg/tests/test_subject_structure.py -q
"""
import numpy as np

from infra.pg import subject_structure as S


def test_is_spatial_frozen_proxy():
    assert S.is_spatial("college_physics") == 1
    assert S.is_spatial("anatomy") == 1
    assert S.is_spatial("high_school_mathematics") == 1
    assert S.is_spatial("electrical_engineering") == 1
    assert S.is_spatial("professional_law") == 0
    assert S.is_spatial("moral_scenarios") == 0
    assert S.is_spatial("high_school_world_history") == 0
    assert S.is_spatial("philosophy") == 0


def test_per_subject_accuracy():
    subs = ["a", "a", "b", "b", "b"]
    acc = S.per_subject_accuracy(subs, [1, 0, 1, 1, 1])
    assert acc == {"a": 0.5, "b": 1.0}


def test_spearman_monotone_and_anti():
    assert abs(S.spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(S.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    # with ties (binary proxy): higher group has higher values => positive rho
    assert S.spearman([0, 0, 1, 1], [1, 2, 8, 9]) > 0.8


def test_gap_proxy_spearman_structured_vs_uniform():
    rng = np.random.default_rng(0)
    # 20 subjects, 100 items each; STRUCTURED: spatial subjects retain MORE above floor.
    subs, arm, floor = [], [], []
    names = [f"college_physics_{i}" for i in range(10)] + [f"professional_law_{i}" for i in range(10)]
    for nm in names:
        spatial = S.is_spatial(nm)
        for _ in range(100):
            subs.append(nm)
            floor.append(int(rng.uniform() < 0.25))
            arm.append(int(rng.uniform() < (0.6 if spatial else 0.26)))   # spatial retains, verbal at floor
    out = S.gap_proxy_spearman(subs, arm, floor, n_boot=1000, seed=1)
    assert out["rho"] > 0.5 and out["structured"], out
    # UNIFORM: same retention regardless of proxy => rho ~ 0, CI straddles.
    arm_u = [int(rng.uniform() < 0.4) for _ in subs]
    out_u = S.gap_proxy_spearman(subs, arm_u, floor, n_boot=1000, seed=2)
    assert not out_u["structured"], out_u
