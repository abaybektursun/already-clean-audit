"""ATLAS Arm X (GPU) — T8: interventions transport by CONJUGATION through the RMSNorm Jacobian,
tested on the exact V1r case where the frame carry left a 0.355 identity gap.

Setup identical to bispec_v1 (probe FIXED at the q_proj input at L9; U_unt loaded from the V1r
bank, not rebuilt): the native cell projects the q/k INPUT off span(U); the frame cell deletes
span(orth(D_γU)) from the residual (the V1r carry, re-run); the CONJUGATED cell applies the
closed-form constrained-least-squares carry Δ = argmin_{Δ⊥x̂}‖J(x)Δ + UUᵀφ(x)‖ per token
(atlas_core.conjugated_removal_delta, one Newton refinement). R-X: closure = 1 − gap_conj/gap_frame.

gap_frame is RECOMPUTED live and asserted against the banked artifact value (0.355 ± 0.1); the
banked value itself is re-read from `bispec_v1r/out/v1.json` at run time, never from this file.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.atlas_x --type p5.4xlarge,g6e.8xlarge --no-weights \
      --s3-prefix results/atlas_x --budget-min 240
"""
import argparse
import contextlib
import json
import math
import os
import pathlib

RIDGE_LAM = 1.0
SEED = 1337


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--u-layer", type=int, default=9)
    ap.add_argument("--n-ppl-seq", type=int, default=120)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _atlas_shared as AS
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.aws_pg.jobs.conserved_core import _probe_forward, _wikitext_windows
    from infra.aws_pg.jobs.invariant_core import PG1_REPO
    from infra.aws_pg.jobs.invariant_core_func import _decoder_layers, _InputProject, _load_full_pg1
    from infra.pg import atlas_core as AC
    from infra.pg import atlas_fit as AF
    from infra.pg import bispec_core as BC
    from infra.pg import conserved_core_probes as P

    AS.assert_frozen("x")
    RS.install_pg()
    g0 = json.loads(RS.fetch(f"{AS.ATLAS_GATE0}/gate0.json", "/tmp/atlas/gate0.json").read_text())
    assert g0.get("green"), "Gate-0 not green — Arm X refuses to run"
    L = a.u_layer
    v1r = json.loads(RS.fetch(AS.BISPEC_V1R, "/tmp/atlas/v1r.json").read_text())
    gap_frame_banked = float(v1r["v1a_transport"]["transport_identity_check"]["abs_diff"])
    nr = torch.load(RS.fetch(AS.BISPEC_V1R_ROWS, "/tmp/atlas/named_rows.pt"),
                    map_location="cpu", weights_only=False)
    U = torch.as_tensor(nr["U_unt_qk"], dtype=torch.float64)
    assert nr["layer"] == L and U.shape == (2048, 128)
    rep = {"object": "atlas_x_v1", "u_layer": L, "k": 128,
           "gap_frame_banked": gap_frame_banked,
           "u_source": AS.BISPEC_V1R_ROWS}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    sha, _ = RS.pinned_pg1_sha("/tmp/atlas")
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    base_sha = HfApi().model_info(PG1_REPO).sha
    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    model = _load_full_pg1(base_sha, "cuda")
    layers = _decoder_layers(model)
    mdtype = next(model.parameters()).dtype
    windows = _wikitext_windows(processor, a.n_ppl_seq, 512)
    gamma_raw = layers[L].input_layernorm.weight.detach().double().cpu()      # RAW weight; D = 1+γ

    def run(ctx_factory, tag):
        with ctx_factory():
            pw = _probe_forward(model, processor, "cuda", layers, windows, [L], capture_proj="q_proj")
        arr = np.asarray([w["ppl_pw"] for w in pw], dtype=np.float64)
        ppl = math.exp(arr[:, 0].sum() / arr[:, 1].sum())
        print(f"[x] pass {tag}: ppl={ppl:.4f}", flush=True)
        return pw, ppl

    def stack(ws):
        X = np.concatenate([w["feats"][L] for w in ws], 0)
        y = np.concatenate([w["pos_labels"] for w in ws], 0)
        return X, y

    base_pw, base_ppl = run(contextlib.nullcontext, "baseline")
    assert abs(base_ppl - AS.FUNC_BASE_PPL) < 0.05, f"ppl wiring drift {base_ppl}"
    d_ws = [w for w in base_pw if not w["is_conf"]]
    c_ws = [w for w in base_pw if w["is_conf"]]
    Xd, yd = stack(d_ws)
    w_pos = P.ridge_fit(Xd, yd, lam=RIDGE_LAM)
    Xc, yc = stack(c_ws)
    pos_base = P.corr_score(yc, P.linear_predict(w_pos, Xc))
    rep["baseline"] = {"ppl": base_ppl, "pos_corr": pos_base,
                       "banked_pos_base": float(v1r["baseline"]["pos_corr"])}
    assert abs(pos_base - v1r["baseline"]["pos_corr"]) < 0.10, "probe port drift vs V1r — halt"
    _w()

    def pos_of(pw):
        X, y = stack([w for w in pw if w["is_conf"]])
        return P.corr_score(y, P.linear_predict(w_pos, X))

    # ---- native (the target the carries must reproduce) --------------------------
    pw, ppl = run(lambda: _InputProject(layers, {L: U}, "cuda", mdtype), "native")
    native = {"pos_corr": pos_of(pw), "ppl": ppl, "dppl": ppl - base_ppl}
    rep["native"] = {**native, "banked": float(v1r["v1a_native"]["U_unt"]["pos_corr"])}
    assert abs(native["pos_corr"] - v1r["v1a_native"]["U_unt"]["pos_corr"]) < 0.10, \
        "native cell does not reproduce V1r — halt"
    _w()

    # ---- frame carry (the V1r cell, re-run: the identity check that D_γ failed) ---
    Ures, _ = torch.linalg.qr((1.0 + gamma_raw).unsqueeze(1) * U)

    def res_del_pure(Q):
        Qd = Q.to("cuda", torch.float32)

        def hook(_m, args, kwargs):
            from infra.pg.inherit_core import _split_hidden
            hs, rebuild = _split_hidden(args, kwargs)
            x = hs.float()
            return rebuild((x - (x @ Qd) @ Qd.transpose(0, 1)).to(hs.dtype))
        return BC.hook_at(layers, L, hook)

    pw, ppl = run(lambda: res_del_pure(Ures), "frame_carry")
    frame = {"pos_corr": pos_of(pw), "ppl": ppl, "dppl": ppl - base_ppl}
    gap_frame = abs(native["pos_corr"] - frame["pos_corr"])
    rep["frame_carry"] = {**frame, "gap_frame_live": gap_frame,
                          "gap_frame_banked": gap_frame_banked}
    assert abs(gap_frame - gap_frame_banked) < 0.10, \
        f"frame-carry gap does not reproduce the banked 0.355 ({gap_frame}) — halt"
    _w()

    # ---- conjugated carry ----------------------------------------------------------
    def conj_ctx():
        return BC.hook_at(layers, L, AC.make_hook_conjugated_removal(U, gamma_raw, newton_steps=2))

    # per-token telemetry on a residual sample first (fail-loud before the full pass)
    cap_rows = []

    def cap_hook(_m, args, kwargs):
        from infra.pg.inherit_core import _split_hidden
        hs, _ = _split_hidden(args, kwargs)
        if len(cap_rows) < 4:
            cap_rows.append(hs.reshape(-1, hs.shape[-1]).float().cpu())

    h = layers[L].register_forward_pre_hook(cap_hook, with_kwargs=True)
    wt0 = windows[0][0]
    from infra.aws_pg.jobs.invariant_core_func import _blank_image
    inp = processor(text=[wt0], images=[_blank_image()], return_tensors="pt", padding=False).to("cuda")
    with torch.inference_mode():
        model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
              pixel_values=inp["pixel_values"].to(mdtype))
    h.remove()
    Xs = torch.cat(cap_rows, 0)[:2000].double()
    _delta, tel = AC.conjugated_removal_delta(Xs, U, gamma_raw, newton_steps=2)
    rep["conjugation_telemetry"] = tel
    assert tel["in_span_after_mean"] < 0.05, \
        f"the conjugated carry leaves {tel['in_span_after_mean']:.3f} of post-norm energy in span(U)"
    _w()

    pw, ppl = run(conj_ctx, "conjugated")
    conj = {"pos_corr": pos_of(pw), "ppl": ppl, "dppl": ppl - base_ppl}
    gap_conj = abs(native["pos_corr"] - conj["pos_corr"])
    rep["conjugated"] = {**conj, "gap_conjugated": gap_conj}
    rep["rx"] = AF.adjudicate_rx(gap_frame=gap_frame, gap_conjugated=gap_conj)
    print(f"[x] R-X: {rep['rx']['verdict']} closure={rep['rx']['closure']:.3f} "
          f"gap_conj={gap_conj:.4f} (frame {gap_frame:.4f})", flush=True)
    _w()

    print("ATLAS_X_OK", json.dumps({"verdict": rep["rx"]["verdict"],
                                    "closure": rep["rx"]["closure"]}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
