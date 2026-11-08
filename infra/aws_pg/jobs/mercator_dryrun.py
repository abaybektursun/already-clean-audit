"""MERCATOR §8 step 3 — the weight-free dry-run of every NEW instrument.

Runs on `r7i.2xlarge --no-weights`: no model, no GPU, no dataset. Its job is to prove the LOGIC of
every instrument the §11 reuse ledger marks NEW, before any of them is trusted with GPU hours.

Covers, in order:
  1. the pure-numpy instruments (the same assertions as infra/pg/tests/test_mercator_*.py, run
     here so the instance's numpy/pyarrow build is exercised too, not just the laptop's);
  2. the TORCH half of M-10/M-11, which cannot run on the control plane at all: model build, the
     gauge rotation, the equivariance gate on BOTH twins, BlockAdam's block second moment, and the
     RoPE commutation check;
  3. an atlas/parquet round trip, so the deliverable renderer is proven in the first minute of the
     program rather than after an arm's worth of compute.

The 8-test CEQ dry-run suite (§11 reuse ledger: "re-run 8/8 pre-freeze") is NOT here: it lives
under `studies/`, and `pg run` tars only `infra/`, so the file does not exist on the instance. It
runs in scripts/mercator_check.sh instead — on the control plane, before EVERY launch rather than
once pre-freeze, which is strictly stronger. Putting it here would have produced a gate that
silently could not run.

THE EQUIVARIANCE GATE IS RUN ON BOTH TWINS ON PURPOSE. A gate exercised only where it is expected
to pass has not been tested. The grid-aware twin MUST fail it; if it does not, the equivariance
measurement is not measuring what M-11 needs it to measure, and that is a stop-and-debug event.

FAIL LOUD: every check is an assert. The job's value is that it CANNOT report a pass it did not
earn.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import mercator_arch as A
from infra.pg import mercator_assemble as AS
from infra.pg import mercator_compress as C
from infra.pg import mercator_core as M
from infra.pg import mercator_erosion as E
from infra.pg import mercator_format as F
from infra.pg import mercator_induction as I
from infra.pg import mercator_predict as PR
from infra.pg import mercator_program as P
from infra.pg import mercator_rowgram as RG


def _pure_instruments(gates) -> dict:
    """(1) The pure-numpy instruments, on this machine's numpy."""
    res = {}

    r = RG.planted_rowspace_null()
    gates.require("m_b1_planted_rowspace", r["planted_capture"] > 0.9, **r)
    gates.require("m_b1_null_at_chance", abs(r["null_capture"] - r["chance_capture"]) < 0.05, **r)
    res["m_b1"] = r

    n, k = 128, 16
    rng = np.random.default_rng(7)
    truth = np.arange(k)
    U = np.zeros((n, k))
    U[truth, np.arange(k)] = 1.0
    mag = rng.uniform(0.1, 0.2, n)
    cons = rng.uniform(0.8, 1.0, n)
    mag[truth], cons[truth] = 1.0, 0.05
    cmp_ = PR.compare_to_baselines(score=PR.suppression_score(mag, cons),
                                   act_rms=rng.uniform(0.1, 0.2, n), U_bank=U, k=k, seed=3)
    gates.require("m_8_predictor_recovers_planted_address",
                  cmp_["predictor"]["jaccard"] == 1.0 and cmp_["beats_activation_baseline"],
                  jaccard=cmp_["predictor"]["jaccard"])
    flat = PR.compare_to_baselines(score=rng.uniform(0, 1, n), act_rms=rng.uniform(0.1, 0.2, n),
                                   U_bank=U, k=k, seed=3)
    gates.require("m_8_uninformative_score_does_not_win",
                  flat["predictor"]["jaccard"] < 0.5, jaccard=flat["predictor"]["jaccard"])
    res["m_8"] = {"planted": cmp_["predictor"], "uninformative": flat["predictor"]}

    L, nh = 6, 2
    attn = np.zeros((nh, 2 * L, 2 * L))
    for i in range(L + 1, 2 * L):
        attn[0, i, i - L + 1] = 1.0
    attn[1] = np.ones((2 * L, 2 * L)) / (2 * L)
    s = I.prefix_matching_score(attn, L)
    gates.require("m_12_head_identification", s[0] > 0.99 and s[1] < 0.2,
                  planted=float(s[0]), uniform=float(s[1]))
    e_haar = I.head_read_energy(np.random.default_rng(11).standard_normal((16, 256)),
                                H.haar_subspace(256, 32, 5))
    gates.require("m_12_read_energy_at_k_over_n", abs(e_haar - 32 / 256) < 0.05,
                  energy=float(e_haar), expected=32 / 256)
    res["m_12"] = {"scores": s.tolist(), "haar_read_energy": float(e_haar)}

    traffic = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
    fit = E.fit_decay_vs_traffic(traffic, 0.3 * traffic ** 1.5)
    gates.require("m_4_erosion_fit_recovers_planted_power_law",
                  abs(fit["slope"] - 1.5) < 1e-6 and fit["r2_log"] > 0.999, **fit)
    noise = E.fit_decay_vs_traffic(traffic, np.random.default_rng(5).uniform(0.1, 0.2, 6))
    gates.require("m_4_unrelated_data_is_not_monotone", abs(noise["spearman"]) < 0.95,
                  spearman=noise["spearman"])
    res["m_4"] = {"planted": fit, "noise_spearman": noise["spearman"]}

    rng = np.random.default_rng(13)
    Wb = rng.standard_normal((24, 32))
    Ub = np.linalg.qr(rng.standard_normal((32, 6)))[0]
    Wf = Wb + 0.05 * rng.standard_normal((24, 32))
    lo, hi = F.roundtrip(Wb, Wf, Ub, r=2), F.roundtrip(Wb, Wf, Ub, r=16)
    gates.require("m_6_roundtrip_improves_with_rank",
                  hi["rel_error_delta"] < lo["rel_error_delta"],
                  r2=lo["rel_error_delta"], r16=hi["rel_error_delta"])
    gates.require("m_6_vault_exact_on_stable_block", lo["core_action_error"] < 1e-10,
                  core_action_error=lo["core_action_error"])
    res["m_6"] = {"r2": lo, "r16": hi}

    g = np.random.default_rng(19).standard_normal((8, 32))
    arms = [C.compress(g, 40, arm=a, sink_coords=list(range(4)), seed=1) for a in C.ARMS]
    bw = C.matched_bandwidth_check(arms)
    gates.require("m_5_matched_bandwidth",
                  all(int((r["sent"] != 0).sum()) == 40 for r in arms), **bw)
    res["m_5"] = {a["arm"]: a["sent_energy_fraction"] for a in arms}

    x = np.linspace(0, 1, 12).reshape(-1, 1)
    y = 3 * x.ravel() + 1

    def fit_predict(tx, ty, testx):
        c = np.linalg.lstsq(np.c_[tx, np.ones(len(tx))], ty, rcond=None)[0]
        return float(np.r_[testx, 1.0] @ c)

    loo = AS.leave_one_out(x, y, fit_predict)
    gates.require("m_c1_loo_beats_no_model_baseline_on_a_solvable_problem",
                  loo["mae_model"] < 1e-8 < loo["mae_baseline"],
                  mae_model=loo["mae_model"], mae_baseline=loo["mae_baseline"])
    res["m_c1"] = loo
    return res


def _torch_instruments(gates) -> dict:
    """(2) The torch half of M-10/M-11 — the part the control plane cannot run at all."""
    import torch

    rope = A.rope_commutation_check()
    gates.require("m_11_rope_commutes_with_residual_rotation", rope["commutes"], **rope)

    cfg_eq = A.ArchConfig(vocab=256, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq=32,
                          norm="scalar", dropout=0.0, seed=1234)
    cfg_grid = A.ArchConfig(vocab=256, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq=32,
                            norm="per_coord", dropout=0.0, seed=1234)

    torch.manual_seed(0)
    idx = torch.randint(0, 256, (2, 32))
    R = A.random_rotation(64, 99)

    def _train(cfg, rotate: bool, optim: str, steps: int = 3):
        torch.manual_seed(cfg.seed)
        m = A.build_model(cfg).to(torch.float64)
        if rotate:
            A.rotate_model_(m, R.to(torch.float64))
        params = list(m.parameters())
        assert optim in ("sgdm", "adam"), f"unknown optimizer {optim}"
        opt = A.SGDM(params, lr=1e-3, momentum=0.9) if optim == "sgdm" \
            else A.ExactAdam(params, lr=1e-3)
        for _ in range(steps):
            opt.zero_grad()
            _logits, loss = m(idx, labels=idx)
            loss.backward()
            opt.step()
        return m, float(loss)

    # EQUIVARIANT twin: scalar gain + SGD+momentum. train(R.theta0) must equal R.train(theta0).
    m_plain, loss_eq = _train(cfg_eq, False, "sgdm")
    m_rot, _ = _train(cfg_eq, True, "sgdm")
    eq = A.equivariance_residual(dict(m_plain.named_parameters()),
                                 dict(m_rot.named_parameters()), R.to(torch.float64),
                                 m_plain.residual_spec)
    gates.require("m_11_equivariant_twin_passes_the_gate", eq["max_rel_residual"] < 1e-8, **eq)

    # GRID-AWARE arm: same model, Adam. The gate MUST fail here, or it is not a gate.
    m_plain_a, _ = _train(cfg_eq, False, "adam")
    m_rot_a, _ = _train(cfg_eq, True, "adam")
    ga = A.equivariance_residual(dict(m_plain_a.named_parameters()),
                                 dict(m_rot_a.named_parameters()), R.to(torch.float64),
                                 m_plain_a.residual_spec)
    gates.require("m_11_adam_breaks_equivariance_so_the_gate_can_fail",
                  ga["max_rel_residual"] > 1e-4, **ga)

    # A per-coordinate norm gain cannot be rotated at all — refusing is the correct behaviour.
    m_grid = A.build_model(cfg_grid)
    refused = False
    try:
        A.rotate_model_(m_grid, R)
    except AssertionError as ex:
        refused = "does not commute" in str(ex)
    gates.require("m_11_per_coordinate_gain_refuses_rotation", refused,
                  n_params=int(sum(p.numel() for p in m_grid.parameters())),
                  rotation_dim=int(R.shape[0]),
                  note="diag(gamma) does not commute with R (inventory item 1)")

    # M-10 switch (b): the reserved block gets ONE second moment; outside it, per-coordinate.
    cfg_b = A.ArchConfig(vocab=256, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq=32,
                         reserved_k=8, reserve_block_v=True, seed=1234)
    torch.manual_seed(cfg_b.seed)
    mb = A.build_model(cfg_b)
    names = [n for n, _ in mb.named_parameters()]
    params = [p for _, p in mb.named_parameters()]
    roles = [mb.residual_spec[n] for n in names]
    opt = A.BlockAdam(params, roles, reserved_k=8, lr=1e-3)
    for _ in range(2):
        opt.zero_grad()
        _logits, loss_b = mb(idx, labels=idx)
        loss_b.backward()
        opt.step()
    gates.require("m_10_block_adam_steps", all(torch.isfinite(p).all() for p in params),
                  n_params=len(params), loss=float(loss_b))
    gates.require("m_10_block_second_moment_is_scalar",
                  all(v.ndim == 0 for v in opt.v_blk), n_blocks=len(opt.v_blk))

    # M-10 switch (c): scaling writer rows must actually change the init.
    c0 = A.ArchConfig(vocab=256, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq=32,
                      reserved_k=8, writer_init_scale=0.0, seed=1234)
    m0 = A.build_model(c0)
    zero_rows = float(m0.blocks[0].attn.o.weight[:8, :].abs().max())
    gates.require("m_10_writer_init_switch_bites", zero_rows == 0.0, max_abs_in_block=zero_rows)

    # M-10 switch (a): the pinned gains must not move under training.
    ca = A.ArchConfig(vocab=256, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq=32,
                      reserved_k=8, reserve_gamma_fixed=True, seed=1234)
    ma = A.build_model(ca)
    before = ma.blocks[0].n1.weight.detach().clone()
    o = A.ExactAdam(list(ma.parameters()), lr=1e-2)
    for _ in range(2):
        o.zero_grad()
        _logits, la = ma(idx, labels=idx)
        la.backward()
        o.step()
    after = ma.blocks[0].n1.weight.detach()
    moved_in = float((after[:8] - before[:8]).abs().max())
    moved_out = float((after[8:] - before[8:]).abs().max())
    gates.require("m_10_pinned_gains_do_not_move", moved_in == 0.0 and moved_out > 0.0,
                  moved_inside=moved_in, moved_outside=moved_out)

    return {"rope": rope, "equivariant_residual": eq, "adam_residual": ga,
            "loss_equivariant": loss_eq, "torch": torch.__version__}


def _atlas_roundtrip(gates, out: pathlib.Path) -> dict:
    """(3) The deliverable renderer, proven in the first minute."""
    exp = M.Expectations(out / "expectations.jsonl")
    rid = exp.log("M-K1", refuter="n/a (gate arm; deliverable = the critic report + the fold)",
                  direction="dry-run instruments behave as their smokes specify", utc=M.utc_now())
    at = M.Atlas(out, arm="M-K1", run_id="dryrun", register="ATLAS", wave="K",
                 expected_ref=rid, queue_item="gate")
    at.add("dryrun_marker", value=1.0, basis=M.BASIS_COORD)
    at.add("dryrun_marker", value=1.0, basis=M.BASIS_RANDOM_REF)
    at.check_dual_basis(["dryrun_marker"])
    info = at.to_parquet()
    gates.require("atlas_parquet_renders", info["n_rows"] == 2, **info)
    return info


def main():
    ap = argparse.ArgumentParser(description="MERCATOR weight-free dry-run of every NEW instrument")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-torch", action="store_true",
                    help="control-plane use only: torch is deliberately absent there")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gates = H.Gates(out)

    # The §5 and §10 contracts must hold before any instrument is trusted.
    t = P.totals()
    gates.require("coverage_no_uncovered_queue_items", t["n_uncovered"] == 0,
                  uncovered=P.uncovered(), n_arms=t["n_arms"], n_items=t["n_queue_items"])
    (out / "COVERAGE.md").write_text(P.render_coverage_md())

    res = {"provenance": M.provenance(), "totals": t,
           "wave_cap_audit": P.wave_cap_audit(),
           "pure": _pure_instruments(gates),
           "atlas": _atlas_roundtrip(gates, out)}
    res["torch"] = None if a.skip_torch else _torch_instruments(gates)
    if a.skip_torch:
        print("[dryrun] TORCH SECTION SKIPPED — this run does NOT satisfy §8 step 3", flush=True)

    res["all_gates_passed"] = all(g["ok"] for g in gates.gates.values())
    M.write_json(out, "dryrun.json", res)
    print("DRYRUN_OK " + json.dumps({"n_gates": len(gates.gates),
                                     "all_passed": res["all_gates_passed"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
