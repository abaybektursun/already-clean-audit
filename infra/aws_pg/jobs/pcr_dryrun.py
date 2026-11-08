"""policy-content-readout — weight-free dry-run (prereg §9.1 / §13 gate 3).

Unit-tests the PURE logic of policy_content_readout on tiny planted tensors — NO model, NO GPU, seconds
on a laptop. Distinct from the runtime preflight (§9.2, which proves the WIRING on real weights); this
proves the LOGIC. Fail-loud: any mismatch RAISES. Run:  python -m infra.aws_pg.jobs.pcr_dryrun
"""
from __future__ import annotations

import numpy as np
import torch

from infra.aws_pg.jobs.policy_content_readout import (
    cloze_acc_lambda, label_shuffle_acc, matched_dw_decoder,
    paired_item_bootstrap, split_masks, policy_verdict,
)


def t_cloze_lambda():
    """λ-recompute of banked (Σlogprob, ntok): λ=1 ranks by per-token mean; a different λ can flip the
    winner when candidates differ in length — proves the bank supports any length-normalization."""
    # item 0: cand 1 has best per-token mean (-0.5) → λ=1 winner = 1; gold = 1 → correct
    # item 1: cand 0 has best SUM (-1.0) but worst MEAN; cand 2 best MEAN (-0.4); gold = 2
    logprobs = [[-3.0, -1.0, -2.0, -4.0],
                [-1.0, -3.0, -2.0, -5.0]]
    ntoks = [[3, 2, 2, 4],
             [1, 3, 5, 5]]
    gold = [1, 2]
    per = cloze_acc_lambda(logprobs, ntoks, gold, lam=1.0)
    assert per.tolist() == [1, 1], f"λ=1 per-item wrong: {per.tolist()}"
    # λ=0 ranks by raw SUM → item1 winner becomes cand 0 (sum -1.0) != gold 2 → 0
    per0 = cloze_acc_lambda(logprobs, ntoks, gold, lam=0.0)
    assert per0.tolist() == [1, 0], f"λ=0 per-item wrong: {per0.tolist()}"
    print("  [ok] cloze_acc_lambda: λ=1 mean-rank, λ=0 sum-rank (length-norm is a pure recompute)")


def t_label_shuffle():
    """Label-shuffle reassigns gold to a random slot ⟂ content → accuracy must collapse to chance 1/4."""
    rng = np.random.default_rng(0)
    N = 4000
    logprobs = rng.normal(size=(N, 4)).tolist()
    ntoks = np.ones((N, 4), dtype=int).tolist()
    gold = rng.integers(0, 4, N).tolist()
    sh, _ = label_shuffle_acc(logprobs, ntoks, gold, seed=4242)
    assert abs(sh - 0.25) < 0.02, f"label-shuffle did not collapse to chance: {sh}"
    print(f"  [ok] label_shuffle_acc → {sh:.4f} ≈ 0.25 (structure-collapse control valid)")


def t_matched_dw_norm():
    """matched_dw_decoder: per 2-D matrix, ‖out−base‖_F must equal ‖target−base‖_F (shape+Frobenius
    matched random ΔW); 1-D norms kept at base."""
    g = torch.Generator().manual_seed(7)
    base = {"layers.0.self_attn.q_proj.weight": torch.randn(8, 6, generator=g),
            "layers.0.mlp.gate_proj.weight": torch.randn(5, 9, generator=g),
            "norm.weight": torch.randn(8, generator=g)}
    target = {"layers.0.self_attn.q_proj.weight": base["layers.0.self_attn.q_proj.weight"] + 0.3 * torch.randn(8, 6, generator=g),
              "layers.0.mlp.gate_proj.weight": base["layers.0.mlp.gate_proj.weight"] + 1.1 * torch.randn(5, 9, generator=g),
              "norm.weight": base["norm.weight"]}
    out, frob = matched_dw_decoder(base, target, seed=99)
    for k in base:
        if base[k].ndim == 2:
            got = (out[k].double() - base[k].double()).norm().item()
            want = (target[k].double() - base[k].double()).norm().item()
            assert abs(got - want) < 1e-4, f"{k}: ‖Δ‖ mismatch got={got} want={want}"
        else:
            assert torch.equal(out[k], base[k]), f"{k}: 1-D norm not kept at base"
    print(f"  [ok] matched_dw_decoder: Frobenius-matched ΔW on {len(frob)} matrices; 1-D kept at base")


def t_paired_bootstrap():
    """Paired item-bootstrap: shared resample index per draw; contrast mean ≈ analytic difference; CI
    brackets the point estimate — the §7 [FLOOR,CEILING] arithmetic."""
    N = 2000
    rng = np.random.default_rng(1)
    ceiling = (rng.random(N) < 0.38).astype(int)
    floor = (rng.random(N) < 0.34).astype(int)
    policy = (rng.random(N) < 0.37).astype(int)
    bt = paired_item_bootstrap({"ceiling": ceiling, "floor": floor, "policy": policy},
                               [("policy_minus_floor", "policy", "floor"),
                                ("ceiling_minus_policy", "ceiling", "policy")], n_boot=500, seed=1337)
    pm = bt["contrast"]["policy_minus_floor"]
    analytic = policy.mean() - floor.mean()
    assert abs(pm[0] - analytic) < 0.01, f"contrast mean off: {pm[0]} vs {analytic}"
    assert pm[1] < pm[0] < pm[2], f"CI does not bracket mean: {pm}"
    assert bt["n_items"] == N
    print(f"  [ok] paired_item_bootstrap: contrast={pm[0]:.4f} CI={pm[1]:.4f}..{pm[2]:.4f} brackets analytic {analytic:.4f}")


def t_split_partition():
    """split_masks must be a clean discovery/confirmation partition; reorder tripwire = hash inequality."""
    import hashlib
    hashes = [hashlib.sha1(f"q{i}".encode()).hexdigest()[:12] for i in range(500)]
    disc, conf = split_masks(hashes)
    assert all(d ^ c for d, c in zip(disc, conf)), "not a partition"
    assert 150 < sum(disc) < 350, f"split wildly imbalanced: {sum(disc)}/500"
    # reorder tripwire: a permuted hash list is detectably != the original (the main() assert)
    assert hashes != hashes[::-1], "reorder tripwire degenerate"
    print(f"  [ok] split_masks: clean partition (disc={sum(disc)} conf={sum(conf)}); reorder→hash inequality")


def t_verdicts():
    """policy_verdict resolves the three pre-registered scenarios from planted summaries."""
    # H_head-on-policy: MCQ floored, cloze near ceiling, CI excludes floor, matched_dw NOT retained
    v = policy_verdict("pi0", mcq_S=0.05, cloze_acc=0.372, cloze_ci=[0.372, 0.360, 0.384],
                       ceiling_cloze=0.382, floor_cloze=0.334, matched_dw_cloze_retained=False)
    assert v["verdict"].startswith("H_head-on-policy"), v["verdict"]
    # H_genuine: cloze at floor, CI below midpoint
    v2 = policy_verdict("pi0", mcq_S=0.04, cloze_acc=0.336, cloze_ci=[0.336, 0.330, 0.342],
                        ceiling_cloze=0.382, floor_cloze=0.334, matched_dw_cloze_retained=False)
    assert v2["verdict"].startswith("H_genuine-deletion"), v2["verdict"]
    # H_readout-hardness: matched_dw reproduces the split → dominates
    v3 = policy_verdict("pi0", mcq_S=0.05, cloze_acc=0.372, cloze_ci=[0.372, 0.360, 0.384],
                        ceiling_cloze=0.382, floor_cloze=0.334, matched_dw_cloze_retained=True)
    assert v3["verdict"].startswith("H_readout-hardness"), v3["verdict"]
    print("  [ok] policy_verdict: H_head / H_genuine / H_readout-hardness all resolve from planted summaries")


def main():
    print("[PCR-DRYRUN] weight-free logic gate (prereg §9.1 / §13 gate 3)")
    t_cloze_lambda()
    t_label_shuffle()
    t_matched_dw_norm()
    t_paired_bootstrap()
    t_split_partition()
    t_verdicts()
    print("[PCR-DRYRUN] ALL PASS")


if __name__ == "__main__":
    main()
