"""BISPEC Arm V1 (GPU) — the positive control, re-scoped per amendment A1. Runs FIRST among GPU arms.

**What the spec asked for, and why it could not work.** §8 gates the whole program on the
instrument "re-detecting `U_unt`'s beyond-energy criticality" by separating `span(U_unt)` deletion
from its energy-band twin **on ppl**. But `conserved_core` established that criticality on
**position-decodability**: `U_unt` drives position-corr 0.933 → 0.096 while the energy-matched null,
removing the same 0.464 activation energy, leaves it at 0.804. On **ppl** the same artifact records
the *opposite* ordering — `U_unt` costs **+49.14**, its energy-matched twin **+260**. A gate that
expects "more critical ⇒ more ppl damage" fires backwards, and §3's instrument identity contains no
position probe at all.

**A1, as three legs with distinct diagnostic meaning.** The probe site is held FIXED (the q_proj
input at layer L, exactly where `conserved_core` reads it); only the DELETION SITE moves.

  V1a-native     `U_unt` projected out of the q/k input at L (the parent's own intervention).
                 Reproduces `conserved_core`. A miss here means the PROBE PORT is broken.
  V1a-transport  the same subspace carried to the RESIDUAL stream and deleted there. A miss here,
                 with native green, means the TRANSPORT failed — not that the residual-site
                 instrument is blind to beyond-energy structure.
  V1b            the ppl leg, direction RECOMPUTED from the artifact (`U_unt` < energy-twin).

**The transport is exact, not an approximation.** PaliGemma's `q_proj` input is the post-
`input_layernorm` residual: `z = γ ⊙ x/rms(x)` with `γ = 1 + input_layernorm.weight` (Gemma RMSNorm).
For any subspace `U` of z-space,

    z ⊥ span(U)  ⟺  Uᵀ D_γ (x/rms(x)) = 0  ⟺  x ⊥ span(D_γ U),

and the per-token positive scalar `1/rms(x)` drops out because orthogonality is scale-invariant. So
the residual-space object is `orth(D_γ U_unt)` — computed, not approximated. §5.1's "transported to
the residual site as disclosed approximation" is replaced by this identity.

**Therefore the transport leg carries a CHECKABLE PREDICTION**, and that is what makes it a control
rather than a hope: after deleting `span(D_γ U_unt)` from the residual, the q_proj input is
orthogonal to `span(U_unt)`, so the position probe read at that same site must land at (very nearly)
the native ablation's position-corr. If the two legs disagree, the transport identity is wrong and
we say so.

**The disclosure that IS owed** (`reference_ablation_site_confound`): deleting from the residual
ENTERING `layers[L]` also removes the subspace from the MLP path and from the stream carried
forward, not only from what attention reads. The residual-site intervention is a strict SUPERSET of
the q/k-input one. Expect the ppl legs to differ even where the position legs agree; the two are
never differenced.

**The energy-matched twin** is drawn on the residual level set at the *measured* removed energy of
the transported subspace — matched by construction, not by a rank proxy. Gate-0a showed that whether
such a twin has room to differ is itself measurable (E/E_max), so the twins' energy-dispersion is
reported with the cell: a null whose dispersion is unmeasured cannot license a beyond-energy claim.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.bispec_v1 --type g6e.8xlarge,p5.4xlarge --no-weights \
      --s3-prefix results/bispec_v1 --budget-min 300 --args "--k 128 --u-layer 9"
"""
import argparse
import contextlib
import json
import math
import os
import pathlib

FUNC_BASE_PPL = 22.4258
FUNC_UNT_DPPL_K128 = 49.14          # artifact anchor; replication band ±10 (conserved_core)
CC_POS_BASE = 0.933                 # conserved_core Tier-1 position-corr, baseline (confirmation half)
CC_POS_UNT = 0.096                  # ... under U_unt (k=128, L9)
CC_POS_EM = 0.804                   # ... under the energy-matched null at the same removed energy
POS_TOL = 0.10                      # replication band on the position-corr anchors
TRANSPORT_AGREE_TOL = 0.10          # the transport identity's own prediction band
N_EM_DRAWS = 3
RIDGE_LAM = 1.0
SEED = 1337


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # HF's Xet transfer backend stalls indefinitely on this instance class: two separate
    # runs wedged inside huggingface_hub.file_download.xet_get at "Fetching 3 files: 33%",
    # log fresh but byte-count frozen for ~50 min. Force the plain HTTP path.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--u-layer", type=int, default=9)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--n-ppl-seq", type=int, default=120)
    ap.add_argument("--ppl-window", type=int, default=512)
    ap.add_argument("--ft-tasks", default="")
    ap.add_argument("--n-sigma-windows", type=int, default=40)
    ap.add_argument("--repaired", action="store_true",
                    help="V1r: scope-matched ppl anchor + pure-projection transport (see §V1r below)")
    ap.add_argument("--all-layers", type=int, default=18)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor

    from infra.aws_pg.jobs.conserved_core import _build_u_unt, _probe_forward, _wikitext_windows
    from infra.aws_pg.jobs.invariant_core import PG1_REPO
    from infra.aws_pg.jobs.invariant_core_func import (
        _blank_image, _decoder_layers, _InputProject, _load_full_pg1,
    )
    from infra.pg import bispec_core as BC
    from infra.pg import conserved_core_probes as P
    from infra.pg.inherit_core import _split_hidden

    L, k, dev = a.u_layer, a.k, "cuda"
    rep = {"object": "bispec_v1_v1", "u_layer": L, "k": k, "n_ppl_seq": a.n_ppl_seq,
           "amendment": "A1: probe site FIXED at the q_proj input; only the DELETION site moves. "
                        "native (q/k) certifies the probe port; transport (residual) certifies the "
                        "residual-site instrument; V1b reports ppl with the artifact's direction "
                        "(U_unt < energy-twin).",
           "artifact_anchors": {"base_ppl": FUNC_BASE_PPL, "unt_dppl_k128": FUNC_UNT_DPPL_K128,
                                "pos_base": CC_POS_BASE, "pos_unt": CC_POS_UNT, "pos_em": CC_POS_EM}}

    def _w():
        (out / "v1.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    api = HfApi()
    # V1r: build U_unt at EVERY layer in one pass. The consensus Gram is accumulated per layer
    # anyway, so the all-layer wiring anchor costs nothing extra — and without it the ppl clause
    # compares a 1-layer ablation against FUNC's ALL-LAYER +49.14 constant (bug 1).
    want = list(range(a.all_layers)) if a.repaired else [L]
    print(f"[v1] building U_unt at layers {want} from the consensus q_proj Gram ...", flush=True)
    U_unt, repos, shas, base_sha = _build_u_unt(api, [k], want, a.ft_tasks)
    U = torch.as_tensor(np.asarray(U_unt[(L, k)]), dtype=torch.float64)     # (2048,k) in q_proj-INPUT space
    rep["u_unt"] = {"n_ft": len(repos), "base_sha": base_sha[:12], "shape": list(U.shape)}
    # bank IMMEDIATELY: the consensus Gram is the expensive half of this job and a later crash
    # must not discard it (it did, once, on a bf16/float32 dtype mismatch in _InputProject).
    torch.save({"U_unt_qk": U, "layer": L, "k": k, "base_sha": base_sha}, out / "u_unt.pt")
    _w()

    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    model = _load_full_pg1(base_sha, dev)
    layers = _decoder_layers(model)
    # _InputProject multiplies the layer's own activations, which are bf16 here. conserved_core
    # passes `next(model.parameters()).dtype`; hardcoding float32 raises deep inside q_proj.
    mdtype = next(model.parameters()).dtype
    windows = _wikitext_windows(processor, a.n_ppl_seq, a.ppl_window)
    rep["n_windows"] = len(windows)

    # ---- the EXACT transport ---------------------------------------------------
    gamma = (1.0 + layers[L].input_layernorm.weight.detach().double().cpu())   # Gemma RMSNorm
    Ures, _ = torch.linalg.qr(gamma.unsqueeze(1) * U)                          # orth(D_γ U)
    orth_err = float((Ures.transpose(0, 1) @ Ures - torch.eye(k, dtype=torch.float64)).abs().max())
    assert orth_err < 1e-8, f"transported basis not orthonormal: {orth_err}"
    th = BC.principal_angles(U, Ures)
    rep["transport"] = {"gamma_min": float(gamma.min()), "gamma_max": float(gamma.max()),
                        "orthonormality_err": orth_err,
                        "principal_angles_deg": {"min": float(th[0]) * 57.29578,
                                                 "median": float(th[k // 2]) * 57.29578,
                                                 "max": float(th[-1]) * 57.29578},
                        "identity": "z ⊥ span(U) ⟺ x ⊥ span(D_γ U); the rms scalar drops out",
                        "prediction": "after residual deletion of span(D_γU), the q_proj input is ⊥ "
                                      "span(U), so the position probe must reproduce the native cell",
                        "disclosure": "residual deletion is a SUPERSET of the q/k-input deletion (it "
                                      "also strips the MLP path and the carried stream); the ppl legs "
                                      "may differ where the position legs agree; never differenced"}
    _w()

    # ---- one pass over the windows under an intervention ----------------------
    def run(ctx_factory, tag):
        with ctx_factory():
            pw = _probe_forward(model, processor, dev, layers, windows, [L], capture_proj="q_proj")
        arr = np.asarray([w["ppl_pw"] for w in pw], dtype=np.float64)
        ppl = math.exp(arr[:, 0].sum() / arr[:, 1].sum())
        print(f"[v1] pass {tag}: ppl={ppl:.4f}", flush=True)
        return pw, ppl

    base_pw, base_ppl = run(contextlib.nullcontext, "baseline")
    assert abs(base_ppl - FUNC_BASE_PPL) < 0.05, f"ppl wiring drift {base_ppl} vs {FUNC_BASE_PPL}"

    # discovery/confirmation firewall: the parent's own salt. Probe TUNED on discovery, the headline
    # contrast read on the confirmation half only.
    def split(pw):
        d = [w for w in pw if not w["is_conf"]]
        c = [w for w in pw if w["is_conf"]]
        assert d and c, "empty probe half"
        return d, c

    def stack(ws):
        X = np.concatenate([w["feats"][L] for w in ws], 0)
        y = np.concatenate([w["pos_labels"] for w in ws], 0)
        return X, y

    d_ws, c_ws = split(base_pw)
    Xd, yd = stack(d_ws)
    Xc, yc = stack(c_ws)
    w_pos = P.ridge_fit(Xd, yd, lam=RIDGE_LAM)
    w_shuf = P.ridge_fit(Xd, P.shuffle_labels(yd, SEED), lam=RIDGE_LAM)
    pos_base = P.corr_score(yc, P.linear_predict(w_pos, Xc))
    pos_shuf = P.corr_score(yc, P.linear_predict(w_shuf, Xc))
    rep["baseline"] = {"ppl": base_ppl, "pos_corr": pos_base, "pos_corr_shuffle": pos_shuf,
                       "n_disc_windows": len(d_ws), "n_conf_windows": len(c_ws)}
    assert pos_base - pos_shuf > 0.05, "position probe has no construct validity at baseline"
    _w()
    print(f"[v1] baseline ppl={base_ppl:.4f} pos_corr={pos_base:.4f} (shuf {pos_shuf:.4f})", flush=True)

    def pos_of(pw):
        X, y = stack([w for w in pw if w["is_conf"]])
        return P.corr_score(y, P.linear_predict(w_pos, X))

    # ---- V1r wiring anchor: reproduce FUNC's ALL-LAYER Δppl (+49.14 ± 10) --------------------
    if a.repaired:
        Qall = {LL: torch.as_tensor(np.asarray(U_unt[(LL, k)]), dtype=torch.float64) for LL in want}
        _pw, ppl_all = run(lambda: _InputProject(layers, Qall, dev, mdtype), "wiring/all-layers")
        rep["wiring_anchor"] = {
            "n_layers": len(want), "dppl_all_layers": ppl_all - base_ppl,
            "func_anchor": FUNC_UNT_DPPL_K128, "band": 10.0,
            "reproduces": bool(abs((ppl_all - base_ppl) - FUNC_UNT_DPPL_K128) < 10.0),
            "why": "conserved_core's _InputProject ablates EVERY layer in Q_by_layer, and the 49.14 "
                   "constant is FUNC's all-layer number. The first V1 run gated a SINGLE-layer "
                   "ablation (+7.465) against it — a cross-object comparison, the same failure A1 "
                   "removed from the spec."}
        _w()
        print(f"[v1] all-layer wiring anchor: Δppl={ppl_all-base_ppl:+.3f} "
              f"(FUNC {FUNC_UNT_DPPL_K128})", flush=True)

    # ---- the activation Gram at the q/k input → U_unt's removed energy + the native em null ----
    # V1r: the parent builds its em null on the ALL-POSITION q_proj-input Gram (_activation_grams),
    # not on the text-block probe features. Using the text-only Gram gave an em null 200x gentler
    # than the parent's (+1.2 ppl vs +260) — a second cross-object slip.
    if a.repaired:
        from infra.aws_pg.jobs.conserved_core import _activation_grams
        Gq = _activation_grams(model, processor, dev, layers, windows, [L])[L]
    else:
        Gq = (Xd.astype(np.float64).T @ Xd.astype(np.float64)) / Xd.shape[0]
    trG = float(np.trace(Gq))
    e_unt = float(np.einsum("ai,ab,bi->", U.numpy(), Gq, U.numpy()) / trG)
    ev, V = np.linalg.eigh(Gq)                       # ascending
    V_desc, mu_desc = V[:, ::-1], ev[::-1]
    Qem, e_em = P.energy_matched_subspace(V_desc, mu_desc, e_unt, k, seed=501)
    Qem = torch.as_tensor(np.asarray(Qem), dtype=torch.float64)

    native = {"removed_energy": {"U_unt": e_unt, "em_matched": float(e_em)}}
    for name, Q in (("U_unt", U), ("em_matched", Qem)):
        pw, ppl = run(lambda Qq=Q: _InputProject(layers, {L: Qq}, dev, mdtype), f"native/{name}")
        native[name] = {"pos_corr": pos_of(pw), "ppl": ppl, "dppl": ppl - base_ppl}
        print(f"[v1][native] {name}: pos={native[name]['pos_corr']:.4f} "
              f"dppl={native[name]['dppl']:+.3f}", flush=True)
    native["beyond_energy_excess"] = native["em_matched"]["pos_corr"] - native["U_unt"]["pos_corr"]
    # A1's object is POSITION-decodability. The ppl clause is a wiring check and, in V1r, it is
    # applied at the scope it was measured at (all layers) rather than at L9.
    pos_clauses = (abs(pos_base - CC_POS_BASE) < POS_TOL
                   and abs(native["U_unt"]["pos_corr"] - CC_POS_UNT) < POS_TOL
                   and abs(native["em_matched"]["pos_corr"] - CC_POS_EM) < POS_TOL)
    if a.repaired:
        port_ok = pos_clauses and rep["wiring_anchor"]["reproduces"]
    else:
        port_ok = pos_clauses and abs(native["U_unt"]["dppl"] - FUNC_UNT_DPPL_K128) < 10.0
    rep["port_clauses"] = {"position_anchors": bool(pos_clauses),
                           "ppl_anchor_scope": "all-layers" if a.repaired else "L9-only (MIS-SCOPED)"}
    native["port_certified"] = bool(port_ok)
    rep["v1a_native"] = native
    _w()

    # ---- the residual Σ at L, for the level-set twins --------------------------
    class _ResidualCapture:
        def __init__(self):
            self.rows = []
            self._h = None

        def __enter__(self):
            def hook(_m, args, kwargs):
                hs, _rb = _split_hidden(args, kwargs)
                self.rows.append(hs[0].detach().float().cpu().numpy())
                return None
            self._h = layers[L].register_forward_pre_hook(hook, with_kwargs=True)
            return self

        def __exit__(self, *e):
            self._h.remove()

    blank = _blank_image(128)
    cap = _ResidualCapture()
    with cap, torch.inference_mode():
        for wt, _wi in windows[:a.n_sigma_windows]:
            inp = processor(text=[wt], images=[blank], return_tensors="pt", padding=False).to(dev)
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"].to(torch.bfloat16))
    R = torch.as_tensor(np.concatenate(cap.rows, 0), dtype=torch.float64)
    hbar_r = R.mean(0)
    Z = R - hbar_r
    Sig_r = Z.transpose(0, 1) @ Z / R.shape[0]
    Sig_r = 0.5 * (Sig_r + Sig_r.transpose(0, 1))
    eig_r = torch.linalg.eigh(Sig_r)
    E_res = BC.removed_energy_fraction(Ures, Sig_r)
    E_min_r, E_max_r, E_max_an = BC.reachable_energy_range(Sig_r, k, eig=eig_r)
    trans = {"residual_sigma_rows": int(R.shape[0]),
             "removed_energy_of_transported_U": E_res,
             "E_max_reachable_at_k": E_max_r, "E_max_analytic_at_k": E_max_an,
             "E_over_Emax": E_res / E_max_r}
    rep["v1a_transport"] = trans
    _w()
    print(f"[v1] transported U removes E={E_res:.4f} of the residual trace "
          f"(E/E_max={E_res/E_max_r:.3f})", flush=True)

    twins, tw_E = [], []
    for s in range(N_EM_DRAWS):
        Q, Er = BC.sample_level_set(Sig_r, k, E_res, seed=900 + s, eig=eig_r, tol=1e-3)
        twins.append(Q)
        tw_E.append(Er)
    disp = BC.energy_dispersion(twins, Sig_r)
    trans["twins"] = {"realized_E": tw_E,
                      "energy_mean_alignment_among_twins": disp["mean_alignment"],
                      "energy_alignment_to_U_res": [BC.energy_alignment(Ures, T, Sig_r) for T in twins],
                      "reading": "alignment ≈ 1 among twins (or to U_res) ⇒ the null had no room to "
                                 "differ and no beyond-energy claim is testable at this (k, E)"}
    _w()

    # V1r bug 2: the transport identity requires a PURE PROJECTION on the residual side.
    # make_hook_band_deletion removes UUᵀ(x − h̄), so x' − h̄ ⊥ span(U) but x' ⊥̸ span(U) — it leaves
    # P·h̄ behind. _InputProject does x − QQᵀx. Different operators, so z' ⊥ span(U) never held, and
    # transport_identity_check correctly reported holds=False (native pos 0.117 vs transport 0.864).
    zero_h = torch.zeros_like(hbar_r)

    def res_del(Q):
        h = zero_h if a.repaired else hbar_r
        return lambda: BC.hook_at(layers, L, BC.make_hook_band_deletion(Q.float(), h.float(), "full"))

    pw, ppl = run(res_del(Ures), "transport/U_res")
    trans["U_res"] = {"pos_corr": pos_of(pw), "ppl": ppl, "dppl": ppl - base_ppl}
    print(f"[v1][transport] U_res: pos={trans['U_res']['pos_corr']:.4f} "
          f"dppl={trans['U_res']['dppl']:+.3f}", flush=True)
    tw_pos, tw_dppl = [], []
    for i, T in enumerate(twins):
        pw, ppl = run(res_del(T), f"transport/twin{i}")
        tw_pos.append(pos_of(pw))
        tw_dppl.append(ppl - base_ppl)
        print(f"[v1][transport] twin{i}: pos={tw_pos[-1]:.4f} dppl={tw_dppl[-1]:+.3f}", flush=True)
    trans["em_twins"] = {"pos_corr": tw_pos, "dppl": tw_dppl,
                         "pos_corr_mean": float(np.mean(tw_pos)), "dppl_mean": float(np.mean(tw_dppl))}
    trans["beyond_energy_excess"] = float(np.mean(tw_pos)) - trans["U_res"]["pos_corr"]
    # the transport identity's own prediction: the two deletion sites must agree ON THE PROBE
    trans["transport_identity_check"] = {
        "native_pos_corr": native["U_unt"]["pos_corr"], "transport_pos_corr": trans["U_res"]["pos_corr"],
        "abs_diff": abs(native["U_unt"]["pos_corr"] - trans["U_res"]["pos_corr"]),
        "tol": TRANSPORT_AGREE_TOL,
        "holds": bool(abs(native["U_unt"]["pos_corr"] - trans["U_res"]["pos_corr"]) < TRANSPORT_AGREE_TOL)}
    _w()

    # ---- adjudication (A1) ----------------------------------------------------
    sat = BC.saturation_guard(damage_treat=trans["U_res"]["pos_corr"],
                              damage_null=float(np.mean(tw_pos)),
                              floor=0.0, intact=pos_base, name="v1a_transport_position")
    transport_ok = (not sat["void"]) and trans["beyond_energy_excess"] > POS_TOL

    rep["v1b_ppl"] = {
        "expected_direction": "U_unt < energy-twin (conserved_core: +49.14 vs +260)",
        "native": {"U_unt": native["U_unt"]["dppl"], "em": native["em_matched"]["dppl"],
                   "direction_holds": bool(native["U_unt"]["dppl"] < native["em_matched"]["dppl"])},
        "transport": {"U_res": trans["U_res"]["dppl"], "em_mean": float(np.mean(tw_dppl)),
                      "direction_holds": bool(trans["U_res"]["dppl"] < float(np.mean(tw_dppl)))},
        "note": "reported, never read as 'more damage ⇒ more critical' — that inversion is exactly "
                "what amendment A1 removed from §8"}

    if port_ok and transport_ok:
        verdict, route = "V1_GREEN", "surface cells may be interpreted"
    elif port_ok:
        verdict, route = "V1_TRANSPORT_MISS", ("probe port certified; the residual-site instrument did "
                                               "not re-detect beyond-energy structure ⇒ TRANSPORT "
                                               "failure, not instrument failure; S/A run DESCRIPTIVE")
    else:
        verdict, route = "V1_PORT_MISS", ("the ported probe does not reproduce conserved_core ⇒ "
                                          "instrument repair before any surface cell is read")
    rep["verdict"] = {"label": verdict, "route": route, "saturation": sat,
                      "port_certified": bool(port_ok), "transport_certified": bool(transport_ok)}
    _w()

    # Bank the bases so Arm S's named rows do not require rebuilding the 24-fine-tune consensus
    # Gram (the expensive half of this job). U_res is the exact residual-space transport.
    torch.save({"U_unt_qk": U, "U_unt_residual": Ures, "gamma": gamma,
                "hbar_residual": hbar_r, "layer": L, "k": k}, out / "named_rows.pt")

    print("BISPEC_V1_OK", json.dumps(
        {"verdict": verdict, "port": port_ok, "transport": transport_ok,
         "native_excess": native["beyond_energy_excess"],
         "transport_excess": trans["beyond_energy_excess"],
         "transport_identity_holds": trans["transport_identity_check"]["holds"]}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
