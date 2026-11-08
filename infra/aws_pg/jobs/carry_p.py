"""CARRY Arm P (GPU) — position-stratify each of the 8 promoted SPECTER spikes (amendment
A3: the pre-approved recompute battery — per-position Rayleigh terms are NOT banked).

The only arm that can RE-SCOPE SPECTER's spike claims before anything builds on them
(§3.2). For each promoted spike (its eigvec on its own sequence mode) we measure, on the
SPECTER operator ppl batch (24 wikitext windows, injector at L0):
  q(δ·1_sink), q(δ·1_content), q(δ)  →  sink_frac = q_sink / q_total  (injection-side,
primary; the R-P 0.3/0.7 lines read on it), with the cross term q_total−q_sink−q_content
reported. Plus a loss-side additive attribution (per-loss-position Fisher terms) as a
secondary diagnostic. Sink positions (frozen A3 rule): mean received-attention mass ≥ 10×
the uniform share 1/T (L0 self-attention telemetry, selection blind to curvature) ∪ {first
position, BOS}.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.carry_p --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/carry_p --budget-min 180
"""
import argparse
import json
import os
import pathlib


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-windows", type=int, default=24)
    ap.add_argument("--sink-mult", type=float, default=10.0)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _carry_shared as CS
    from infra.pg import carry_core as CC
    from infra.pg import inherit_pg1 as IP
    from infra.pg import specter_core as SC

    freeze = CS.assert_frozen("p")
    CS.install_pg()
    rep = {"object": "carry_p_v1", "freeze": freeze, "sink_mult": a.sink_mult}
    tm = CS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = "/tmp/carry_p"
    g0 = json.loads(CS.fetch(f"{CS.CARRY_GATE0}/gate0.json",
                             pathlib.Path(wd) / "gate0.json").read_text())
    assert g0.get("green"), "carry gate0 not green"
    anat = CS.load_roster(wd)

    with tm.stage("load"):
        model, proc = IP.load_pg1(sha=anat["base_sha"])
        layers = IP.decoder_layers(model)
        dev = "cuda"
        d = IP.D
        wins = CS.wikitext_windows(proc, a.n_windows)
        install, holder = SC.make_residual_injector(layers, 0)
        items = [CS.build_ppl_item(model, proc, dev, wdict, install, holder)
                 for wdict in wins]
    _w()

    # ---- telemetry: L0 received-attention → per-item sink masks (blind to curvature) ----
    with tm.stage("telemetry"):
        bos_id = proc.tokenizer.bos_token_id
        sink_masks, sink_meta = [], []
        for wdict in wins:
            inp = proc(text=[wdict["text"]], images=[wdict["blank"]],
                       return_tensors="pt", padding=False).to(dev)
            inp["pixel_values"] = inp["pixel_values"].to(next(model.parameters()).dtype)
            with torch.no_grad():
                att = model(input_ids=inp["input_ids"],
                            attention_mask=inp["attention_mask"],
                            pixel_values=inp["pixel_values"],
                            output_attentions=True).attentions[0]
            T = att.shape[-1]
            received = att.float().mean(dim=(0, 1, 2))          # (T,) key-position mass
            mask = (received >= a.sink_mult / T)
            mask[0] = True
            ids0 = inp["input_ids"][0]
            bos_pos = (ids0 == bos_id).nonzero(as_tuple=True)[0]
            for p in bos_pos.tolist():
                mask[p] = True
            sink_masks.append(mask.float())
            sink_meta.append({"T": int(T), "n_sink": int(mask.sum()),
                              "sink_positions": mask.nonzero(as_tuple=True)[0].tolist()[:16]})
        rep["sink_meta"] = sink_meta
        rep["mean_sink_fraction_of_positions"] = float(
            sum(m["n_sink"] for m in sink_meta) / sum(m["T"] for m in sink_meta))
    _w()

    # ---- per-spike stratified quadforms -------------------------------------------------
    with tm.stage("stratify"):
        spikes = CS.load_promoted_spikes(wd)
        rep["white_sign_seed"] = SC.WHITE_SIGN_SEEDS[0]
        per_spike = []
        for sp in spikes:
            mode = sp["key"].split("/")[1]
            sign_seed = SC.WHITE_SIGN_SEEDS[0] if mode == "white" else None
            v = sp["evec"].to(dev).float()
            q_full = q_sink = q_content = 0.0
            for it, mask in zip(items, sink_masks):
                T = it["T"]
                assert mask.shape[0] == T, f"sink mask T={mask.shape[0]} vs item T={T}"
                c = SC.env_coeffs(mode, T, sign_seed=sign_seed, device=dev)
                delta_full = c.unsqueeze(1) * v.unsqueeze(0)             # (T,d)
                m = mask.to(dev).unsqueeze(1)
                q_full += SC.ggn_quadform(it, delta_full) / T
                q_sink += SC.ggn_quadform(it, delta_full * m) / T
                q_content += SC.ggn_quadform(it, delta_full * (1.0 - m)) / T
            q_full, q_sink, q_content = (q_full / len(items), q_sink / len(items),
                                         q_content / len(items))
            sf = CC.sink_frac(q_sink, q_full) if q_full > 0 else float("nan")
            cross = 1.0 - (q_sink + q_content) / q_full if q_full > 0 else float("nan")
            per_spike.append({"key": sp["key"], "idx": sp["idx"], "eval": sp["eval"],
                              "mode": mode, "q_full": q_full, "q_sink": q_sink,
                              "q_content": q_content, "sink_frac": sf, "cross_frac": cross})
            print(f"[carry-p] {sp['key']}#{sp['idx']}: sink_frac={sf:.3f} "
                  f"cross={cross:+.3f} q_full={q_full:.3e}", flush=True)
            rep["per_spike"] = per_spike
            _w()
    _w()

    # ---- loss-side additive attribution (secondary diagnostic) --------------------------
    with tm.stage("loss_side"):
        import torch.func as F
        for row, sp in zip(per_spike, spikes):
            mode = row["mode"]
            sign_seed = SC.WHITE_SIGN_SEEDS[0] if mode == "white" else None
            v = sp["evec"].to(dev).float()
            m_sink_loss = m_tot_loss = 0.0
            for it, mask in zip(items, sink_masks):
                T = it["T"]
                c = SC.env_coeffs(mode, T, sign_seed=sign_seed, device=dev)
                delta = c.unsqueeze(1) * v.unsqueeze(0)

                def f(dd, _it=it):
                    _it["holder"]["delta"] = dd
                    return _it["forward_fn"]()
                with torch.no_grad():
                    logits, u = F.jvp(f, (torch.zeros_like(delta),), (delta,))
                per_pos = _per_position_fisher(logits[0], u[0], it["loss_w"])
                loss_mask = (it["loss_w"] > 0).float()
                sink_loss = (per_pos * mask.to(dev) * loss_mask).sum()
                m_sink_loss += float(sink_loss)
                m_tot_loss += float((per_pos * loss_mask).sum())
            row["loss_side_sink_frac"] = (m_sink_loss / m_tot_loss
                                          if m_tot_loss > 0 else float("nan"))
        rep["per_spike"] = per_spike
    _w()

    rep["rp"] = CC.adjudicate_rp([r["sink_frac"] for r in per_spike])
    CS.assert_gpu_seconds(rep)
    print("CARRY_P_OK", json.dumps({"verdict": rep["rp"]["verdict"],
                                    "sink_fracs": [round(r["sink_frac"], 3)
                                                   for r in per_spike]}))
    rep["green"] = True
    _w()


def _per_position_fisher(logits, u, loss_w, chunk=65536):
    """Per-position wₜ·(Σ_v p·u² − (Σ_v p·u)²) — fisher_quad_from_tangent WITHOUT the final
    sum over positions (the additive per-token Rayleigh attribution). Returns (T,) fp64."""
    import torch
    T, V = logits.shape
    w = loss_w.to(logits.device, torch.float32)
    live = w > 0
    out = torch.zeros(T, dtype=torch.float64, device=logits.device)
    if not bool(live.any()):
        return out
    lg = logits[live].float()
    uu = u[live].float()
    lse = torch.logsumexp(lg, dim=-1, keepdim=True)
    s1 = torch.zeros(lg.shape[0], dtype=torch.float64, device=lg.device)
    s2 = torch.zeros(lg.shape[0], dtype=torch.float64, device=lg.device)
    for c0 in range(0, V, chunk):
        p = (lg[:, c0:c0 + chunk] - lse).exp().double()
        uc = uu[:, c0:c0 + chunk].double()
        s1 += (p * uc).sum(-1)
        s2 += (p * uc * uc).sum(-1)
    out[live] = w[live].double() * (s2 - s1 ** 2)
    return out


if __name__ == "__main__":
    main()
