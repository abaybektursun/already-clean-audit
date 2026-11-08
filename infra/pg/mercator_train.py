"""M-A2 — the breathe valve, and the damage-onset reading (queue A2).

W1-S opened the valve to 21x realized core motion without damage; REFEREE's full SGD dose destroyed
the model. The band between is unmapped, and mapping it is what M-A2 is for.

THE VALVE. `SubspaceProjector` CONFINES the update to a subspace (g <- g U U^T). The valve does the
opposite: it AMPLIFIES the component inside the subspace and leaves the complement alone,

    g  <-  g + (gain - 1) * (g U U^T)

so gain 1 is exactly the untouched optimizer and gain > 1 pushes harder on the sink specifically.
Written as an additive correction rather than as a two-term reconstruction because the complement
must be BIT-IDENTICAL to the unmodified gradient at gain 1 — a valve whose "off" position perturbs
the run would confound every dose on the ladder with its own implementation.

Applied to the GRADIENT, not the weight, for the reason SubspaceProjector gives: the amplification
must hold for every optimizer step including momentum, which a post-hoc weight edit would not.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations


class BreatheValve:
    """Amplify each parameter's gradient INSIDE a fixed input subspace by `gain`.

    `basis_by_key` maps the trainable-module key to that cell's (n x k) orthonormal basis, exactly
    as `SubspaceProjector` takes it, so the two are drop-in alternatives in the same loop and the
    same bank library feeds both.
    """

    def __init__(self, basis_by_key, device, gain: float):
        import torch
        assert gain >= 1.0, (
            f"gain {gain} < 1 would ATTENUATE the subspace, which is a different experiment "
            f"(that is W1-T's restriction, not W1-S's valve)")
        self.gain = float(gain)
        self.U = {k: torch.as_tensor(U, device=device, dtype=torch.float32)
                  for k, U in basis_by_key.items()}

    def apply(self, modules) -> int:
        """Amplify in place. Returns the number of modules touched, so a silently empty basis map
        cannot look like a successful application."""
        if self.gain == 1.0:
            # EXACT no-op, and it returns before touching torch at all: the "off" position of the
            # valve must be indistinguishable from not having a valve, right down to not entering
            # the no_grad block.
            return 0
        import torch
        n = 0
        with torch.no_grad():
            for key, mod in modules.items():
                if mod.weight.grad is None or key not in self.U:
                    continue
                U = self.U[key]
                g = mod.weight.grad.to(torch.float32)
                inside = (g @ U) @ U.T
                mod.weight.grad = (g + (self.gain - 1.0) * inside).to(mod.weight.grad.dtype)
                n += 1
        return n


def gain_ladder(start: float = 8.0, n: int = 5) -> list:
    """W1-S ran 1/2/4/8; M-A2 continues the same doubling from where it stopped.

    Log-spaced by construction, which is what §3's dose rule asks for, and continuing the SAME
    geometric sequence means the two arms' ladders join into one curve rather than two.
    """
    assert start >= 1.0 and n >= 4, f"§3 needs >= 4 doses, got {n}"
    return [float(start * (2 ** i)) for i in range(n)]


def damage_onset(gains, damages, *, band_hi: float) -> dict:
    """The first gain whose damage exceeds the control band, with the bracketing pair.

    `band_hi` is the upper edge of the matched-control damage band — the level below which a
    damage reading is indistinguishable from what the controls do at the same dose. Returning the
    BRACKET rather than a single number is the honest form: the ladder is geometric, so the onset
    is known only to lie between two rungs, and reporting the upper rung alone would imply a
    resolution the sweep does not have.
    """
    import numpy as np
    g = np.asarray(gains, dtype=np.float64)
    d = np.asarray(damages, dtype=np.float64)
    assert g.shape == d.shape and g.size >= 2, f"need >=2 paired doses, got {g.shape}"
    order = np.argsort(g)
    g, d = g[order], d[order]
    over = np.flatnonzero(d > band_hi)
    # A SUSTAINED onset is the first rung above the band that STAYS above it for every higher
    # rung. The distinction is not pedantry: with a five-point series wandering inside a noise
    # band, SOME rung crosses almost surely, and reporting that crossing as an onset would turn
    # noise into a threshold. M-A2's first run crossed by 0.0008 at gain 16 and fell back below at
    # 32 and 64 — first-crossing said "onset at 16", sustained said "none" (defect M-D22). Both
    # are returned; the sustained one is the reading that supports a claim.
    sustained = None
    for j in range(len(d)):
        if bool((d[j:] > band_hi).all()):
            sustained = j
            break
    common = {"max_gain_tested": float(g[-1]), "max_damage": float(d[-1]),
              "band_hi": float(band_hi), "n_rungs_over_band": int(over.size),
              "sustained_onset": float(g[sustained]) if sustained is not None else None,
              "sustained_bracket": ([float(g[sustained - 1]) if sustained else None,
                                     float(g[sustained])] if sustained is not None else None)}
    if not over.size:
        return {"onset": None, "bracket": None, **common,
                "reading": "no damage onset within the swept ladder"}
    i = int(over[0])
    return {"onset": float(g[i]), "bracket": [float(g[i - 1]) if i else None, float(g[i])],
            "damage_at_onset": float(d[i]), **common,
            "reading": ("damage first exceeds the control band between the bracketing gains"
                        if sustained is not None else
                        "damage crosses the control band but does not stay above it: the "
                        "crossing is not an onset")}


def valve_saturation(gains, motion_units) -> dict:
    """How much of the REQUESTED amplification the optimizer actually delivered.

    The valve multiplies the in-subspace gradient by `gain`, so a naive reading expects realized
    motion to scale with it. Adam does not work that way: multiplying g inside a subspace multiplies
    both m_hat and sqrt(v_hat) there, and the update m_hat/sqrt(v_hat) is unchanged to first order.
    So the delivered/requested ratio is the arm's own measurement of how self-defeating a
    gradient-side valve is under a per-coordinate denominator — and if it collapses, "no damage at
    gain 128" says nothing about gain 128, only about what gain 128 delivered.
    """
    import numpy as np
    g = np.asarray(gains, dtype=np.float64)
    m = np.asarray(motion_units, dtype=np.float64)
    assert g.shape == m.shape and g.size >= 2, f"need >=2 paired rungs, got {g.shape}"
    order = np.argsort(g)
    g, m = g[order], m[order]
    delivered = m / g
    # A log-log slope of 1 would be full delivery; 0 would be complete absorption.
    slope = float(np.polyfit(np.log(g), np.log(m), 1)[0])
    return {"gains": g.tolist(), "motion_units": m.tolist(),
            "delivered_over_requested": delivered.tolist(),
            "delivered_over_requested_at_max_gain": float(delivered[-1]),
            "log_log_slope": slope,
            "reading": ("slope 1 means the optimizer delivers the requested amplification; "
                        "slope 0 means it absorbs all of it")}


def power_law_pickup(motion, damage, *, exponent: float, coefficient: float) -> dict:
    """Does the observed damage follow REFEREE's motion power law at these doses?

    `exponent` and `coefficient` are ARGUMENTS, recomputed from REFEREE's artifact at preflight —
    and the caller must also state which damage SCALE they were fitted on, because MERCATOR's own
    anchor recompute found the exponent is 1.4175 on ppl/ppl_base and 1.9921 on the excess scale
    over the same nine rows. A prediction quoted without its scale is not a prediction.
    """
    import numpy as np
    m = np.asarray(motion, dtype=np.float64)
    d = np.asarray(damage, dtype=np.float64)
    assert m.shape == d.shape and m.size >= 3, f"need >=3 paired points, got {m.shape}"
    assert (m > 0).all(), "motion must be positive to evaluate a power law"
    pred = coefficient * m ** exponent
    resid = d - pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((d - d.mean()) ** 2).sum())
    return {"predicted": pred.tolist(), "observed": d.tolist(),
            "r2_against_referee_law": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "mean_ratio_observed_over_predicted": float(np.mean(d / pred)),
            "exponent": float(exponent), "coefficient": float(coefficient), "n": int(m.size)}


class ScalarDenomAdam:
    """M-A3 stage 2's counterfactual: Adam with the per-coordinate denominator replaced by ONE
    scalar per parameter.

        Adam        w <- w - lr * m_hat / (sqrt(v_hat) + eps)           per-coordinate denominator
        this        w <- w - lr * m_hat / (mean(sqrt(v_hat)) + eps)     one denominator per matrix

    EVERYTHING ELSE IS IDENTICAL to `darwin_train.ExactAdam` — same betas, same bias correction,
    same fp32 master weights, same decoupled decay — because the arm's whole claim is about the
    DENOMINATOR'S SHAPE and nothing else. A counterfactual that also changed the momentum or the
    correction would not isolate it.

    Why mean(sqrt(v_hat)) and not sqrt(mean(v_hat)): the quantity Adam divides by is sqrt(v_hat),
    so the scalar that replaces it is an average OF THAT, not of its square. The two differ by
    Jensen whenever the second moment is unevenly spread — which is exactly the regime under study
    — so the choice is not innocuous and is stated rather than defaulted.

    THE IDENTITY THAT MAKES IT A CONTROLLED SWAP: when v_hat is the same in every coordinate the
    two rules coincide exactly. So any difference they produce is attributable to the SPREAD of the
    second moment and to nothing else, and the job that uses this asserts that identity on
    synthetic gradients before it trains anything.
    """

    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        import torch
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.t = 0
        self.master = [p.detach().to(torch.float32).clone() for p in self.params]
        self.m = [torch.zeros_like(x) for x in self.master]
        self.v = [torch.zeros_like(x) for x in self.master]
        self.last_denoms = []

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        import torch
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        self.last_denoms = []
        with torch.no_grad():
            for p, w, m, v in zip(self.params, self.master, self.m, self.v):
                if p.grad is None:
                    continue
                g = p.grad.detach().to(torch.float32)
                m.mul_(self.b1).add_(g, alpha=1.0 - self.b1)
                v.mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
                denom = float((v / bc2).sqrt().mean()) + self.eps
                self.last_denoms.append(denom)
                if self.wd:
                    w.add_(w, alpha=-self.lr * self.wd)
                w.add_(m / bc1, alpha=-self.lr / denom)
                p.copy_(w.to(p.dtype))
