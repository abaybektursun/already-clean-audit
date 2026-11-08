"""Weight-free CPU dry-run gate for forgetting_locus — the prereg §9.1 logic test.
NO model, NO datasets, NO GPU — pure surgery logic on tiny synthetic tensors.

    python3 -m pytest infra/aws_pg/tests/test_forgetting_locus.py -q

Covers: the surgery identity W_ft − ΔW QQᵀ == W0 + ΔW(I−QQᵀ); removal fraction; snapshot/restore;
set-qk-to-base; and the PLANTED core-locus mechanic (removing the subspace ΔW reads recovers all;
removing an orthogonal subspace recovers nothing).
"""
import numpy as np
import torch
import torch.nn as nn

from infra.aws_pg.jobs import forgetting_locus as F


def _layer(n, m, Wq, Wk):
    class Attn(nn.Module):
        def __init__(s):
            super().__init__()
            s.q_proj = nn.Linear(n, m, bias=False)
            s.k_proj = nn.Linear(n, m, bias=False)
            with torch.no_grad():
                s.q_proj.weight.copy_(torch.tensor(Wq, dtype=torch.float32))
                s.k_proj.weight.copy_(torch.tensor(Wk, dtype=torch.float32))

    class Layer(nn.Module):
        def __init__(s):
            super().__init__()
            s.self_attn = Attn()

    return Layer()


def test_surgery_identity_and_removal_fraction():
    rng = np.random.default_rng(0)
    n, m, k = 16, 20, 4
    W0q = rng.standard_normal((m, n)); W0k = rng.standard_normal((m, n))
    dWq = rng.standard_normal((m, n)); dWk = rng.standard_normal((m, n))
    Wftq = W0q + dWq; Wftk = W0k + dWk
    layers = [_layer(n, m, Wftq, Wftk)]
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q = Q[:, :k]
    rem = F._surgery(layers, {0: Wftq}, {0: Wftk}, {0: W0q}, {0: W0k}, {0: Q}, [0], "cpu")
    QQt = Q @ Q.T
    exp_q = Wftq - dWq @ QQt                      # == W0q + dWq (I − QQᵀ)
    got_q = layers[0].self_attn.q_proj.weight.detach().numpy()
    assert np.abs(got_q - exp_q).max() < 1e-4, np.abs(got_q - exp_q).max()
    assert np.allclose(exp_q, W0q + dWq @ (np.eye(n) - QQt), atol=1e-9)   # the identity
    num = ((dWq @ QQt) ** 2).sum() + ((dWk @ QQt) ** 2).sum()
    den = (dWq ** 2).sum() + (dWk ** 2).sum()
    assert abs(rem - num / den) < 1e-9
    assert 0.0 <= rem <= 1.0


def test_snapshot_restore_roundtrip():
    rng = np.random.default_rng(1)
    n, m, k = 12, 12, 3
    W0q = rng.standard_normal((m, n)); W0k = rng.standard_normal((m, n))
    Wftq = W0q + rng.standard_normal((m, n)); Wftk = W0k + rng.standard_normal((m, n))
    layers = [_layer(n, m, Wftq, Wftk)]
    snap = F._snapshot_qk(layers, [0])
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    F._surgery(layers, {0: Wftq}, {0: Wftk}, {0: W0q}, {0: W0k}, {0: Q[:, :k]}, [0], "cpu")
    F._restore_qk(layers, snap, [0])
    assert np.abs(layers[0].self_attn.q_proj.weight.detach().numpy() - Wftq).max() < 1e-5
    assert np.abs(layers[0].self_attn.k_proj.weight.detach().numpy() - Wftk).max() < 1e-5


def test_set_qk_to_base():
    rng = np.random.default_rng(2)
    n, m = 10, 10
    W0q = rng.standard_normal((m, n)); W0k = rng.standard_normal((m, n))
    Wftq = W0q + 5.0; Wftk = W0k - 3.0
    layers = [_layer(n, m, Wftq, Wftk)]
    F._set_qk_to_base(layers, {0: W0q}, {0: W0k}, [0], "cpu")
    assert np.abs(layers[0].self_attn.q_proj.weight.detach().numpy() - W0q).max() < 1e-5
    assert np.abs(layers[0].self_attn.k_proj.weight.detach().numpy() - W0k).max() < 1e-5


def test_planted_core_locus_removal():
    """ΔW reads ONLY span(U): removing U removes all of ΔW (frac≈1) and restores q to base; removing an
    orthogonal subspace removes nothing (frac≈0). The core-locus mechanic the verdict rests on."""
    rng = np.random.default_rng(3)
    n, m, k = 24, 20, 4
    Qfull, _ = np.linalg.qr(rng.standard_normal((n, n)))
    U = Qfull[:, :k]                                   # the subspace ΔW reads
    Qperp = Qfull[:, k:2 * k]                          # orthogonal to U
    dWq = rng.standard_normal((m, n)) @ (U @ U.T)      # ΔW acts only on span(U)
    W0q = rng.standard_normal((m, n)); W0k = rng.standard_normal((m, n))
    Wftq = W0q + dWq
    layers = [_layer(n, m, Wftq, W0k)]                 # ΔWk = 0
    rem_U = F._surgery(layers, {0: Wftq}, {0: W0k}, {0: W0q}, {0: W0k}, {0: U}, [0], "cpu")
    got = layers[0].self_attn.q_proj.weight.detach().numpy()
    assert rem_U > 0.99, rem_U                          # removing U removes ~all of ΔW
    assert np.abs(got - W0q).max() < 1e-4               # ... and restores q to base
    # removal fraction of an orthogonal subspace (math uses the unchanged Wft/W0 dicts):
    rem_perp = F._surgery(layers, {0: Wftq}, {0: W0k}, {0: W0q}, {0: W0k}, {0: Qperp}, [0], "cpu")
    assert rem_perp < 0.01, rem_perp
