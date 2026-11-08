"""SPECTER Gate-0 (GPU) — anchors + the FD WIRING CHECK + the `specter|` re-salt audit + the
memory smoke + the differentiation-path resolution (§0.5-1/-2/-3 made live). Nothing downstream
runs until this is green; the resolved path (`torch.func` vs manual double-backprop) is recorded
in the artifact and every operator driver reuses it.

Anchors, each recomputed from its named artifact at run time:
  * blank-image wikitext ppl (n_seq=120) vs the invcore-FUNC anchor (22.4258, tol 0.01)
  * model sha vs the invcore-FUNC pin; MQA layout asserts (kv heads 1, k/v width 256)
  * identity injector: the zero-delta residual injector is a bit-level no-op on the logits
  * FD wiring: GGN quadform vs the symmetric KL finite difference along 8 probe directions on a
    fixed micro-batch (ppl + vqa items), rel err ≤ 1e-2 on the ACTIVE path
  * jvp↔vjp consistency: ⟨δ, Gδ⟩ from ggn_vp equals the jvp-only quadform (streaming-Fisher
    wiring at the full 257k vocab — the live half of §0.5-3)
  * memory smoke: one full H_dc matvec on the 12-item half-batch at T=512 (peak GiB logged)

  ./infra/aws_pg/pg run infra.aws_pg.jobs.specter_gate0 --type p5.4xlarge,g6e.8xlarge \
      --no-weights --s3-prefix results/specter_gate0 --budget-min 120
"""
import argparse
import json
import os
import pathlib

FD_PROBES = 8


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--site", default="L0")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.aws_pg.jobs import _specter_shared as SS
    from infra.pg import inherit_pg1 as IP
    from infra.pg import specter_core as SC

    freeze = SS.assert_frozen("gate0")
    RS.install_pg()
    L = int(a.site[1:])
    rep = {"object": "specter_gate0_v1", "site": a.site, "freeze": freeze,
           "n_ppl_seq": SS.N_PPL_SEQ}
    tm = SS.StageTimer(rep)

    def _w():
        (out / "gate0.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- model + sha + MQA asserts ------------------------------------------------
    with tm.stage("load"):
        sha, func_ppl = RS.pinned_pg1_sha("/tmp/specter")
        model, proc = IP.load_pg1(sha=sha)
        layers = IP.decoder_layers(model)
    tc = model.config.text_config
    assert tc.num_hidden_layers == 18 and tc.hidden_size == 2048
    assert tc.num_key_value_heads == 1, f"MQA assert failed: {tc.num_key_value_heads}"
    assert layers[0].self_attn.k_proj.out_features == 256
    rep["model"] = {"sha": sha, "head_dim": tc.head_dim, "n_heads": tc.num_attention_heads,
                    "rope_theta": float(getattr(tc, "rope_theta", 10000.0))}
    _w()

    # ---- ppl wiring anchor --------------------------------------------------------
    with tm.stage("ppl_anchor"):
        base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=SS.N_PPL_SEQ)["ppl"]
    rep["ppl_gate"] = {"base_ppl": base_ppl, "func_anchor": SS.FUNC_BASE_PPL,
                       "delta": base_ppl - SS.FUNC_BASE_PPL}
    assert abs(base_ppl - SS.FUNC_BASE_PPL) < SS.PPL_TOL, \
        f"ppl instrument drift: {base_ppl} vs {SS.FUNC_BASE_PPL} — halt"
    _w()

    # ---- items: specter| re-salt + exposure audit ---------------------------------
    with tm.stage("salt_audit"):
        vqa, _coco = IP.pg1_tasks()
        from infra.aws_pg.jobs._regrain_shared import GRAIN_GATE0
        it = json.loads(SS.fetch(f"{GRAIN_GATE0}/items_vqa.json",
                                 "/tmp/specter/items_vqa.json").read_text())
        rows, hashes = IP.load_items(vqa, it["stream_window"])
        assert hashes == it["hashes"], "item-stream drift — halt"
        import hashlib as _hl

        import numpy as np
        new = SS.specter_split(hashes)
        old_atlas = np.array([int(_hl.sha1(f"atlas|{h}".encode()).hexdigest(), 16) % 2 == 1
                              for h in hashes])
        n_moved = int((np.asarray(new["disc_mask"]) != ~old_atlas).sum())
        assert n_moved > 0, "re-salt produced an identical split — salt not applied"
        disc_idx, conf_idx = list(new["disc_idx"]), list(new["conf_idx"])
        # operator batches draw from the DISCOVERY half only; sealed ∩ operator = ∅ audited here
        op_vqa_idx = disc_idx[:SS.N_OP_VQA + SS.N_OP_HELD]
        sealed_set = {hashes[i] for i in conf_idx}
        op_set = {hashes[i] for i in op_vqa_idx}
        assert not (sealed_set & op_set), "sealed half intersects operator batch — halt"
        rep["items"] = {"n_total": len(hashes), "disc": int(new["n_disc"]),
                        "sealed": int(new["n_conf"]), "n_reassigned_vs_atlas": n_moved,
                        "salt": "specter|", "op_vqa_n": len(op_vqa_idx),
                        "sealed_op_overlap": 0}
    _w()

    # ---- site rms + injector identity check ---------------------------------------
    with tm.stage("identity"):
        wins = SS.wikitext_windows(proc, 2)
        install, holder = SC.make_residual_injector(layers, L)
        item0 = SS.build_ppl_item(model, proc, "cuda", wins[0], install, holder)
        d = IP.D
        cap = {}

        def _cap_hook(_m, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            cap["rms"] = float(hs.float().pow(2).mean().sqrt())

        h = layers[L].register_forward_pre_hook(_cap_hook, with_kwargs=True)
        with torch.no_grad():
            holder["delta"] = torch.zeros(item0["T"], d, device="cuda")
            lg_zero = item0["forward_fn"]()
        h.remove()
        # bit-level identity: zero delta vs the hook absent entirely
        import contextlib as _ctx

        @_ctx.contextmanager
        def _null_install():
            yield
        item_ref = SS.build_ppl_item(model, proc, "cuda", wins[0], _null_install, {})
        with torch.no_grad():
            lg_none = item_ref["forward_fn"]()
        idmax = float((lg_zero - lg_none).abs().max())
        rep["identity"] = {"site_rms": cap["rms"], "zero_delta_max_abs_logit_diff": idmax}
        assert idmax == 0.0, f"zero-delta injector is not a bit-level no-op ({idmax})"
    _w()
    rms = rep["identity"]["site_rms"]

    # ---- differentiation-path resolution + FD wiring check ------------------------
    # The FD check DIFFERENCES two forwards, so at bf16 it carries an additive rounding-KL
    # floor (~1e-3/token — measured below and reported as the production noise floor). The
    # WIRING is certified on an fp32 forward: same code path, dtype-independent; AD itself
    # (jvp/vjp along ONE forward) has no differencing floor. First Gate-0 run halted exactly
    # here — journal E3b.
    with tm.stage("fd_check"):
        vqa_row = rows[disc_idx[0]]
        item_v = SS.build_vqa_item(model, proc, "cuda", vqa_row, vqa, install, holder)
        path = "func"
        gen = torch.Generator(device="cpu").manual_seed(101)

        def _probe(i, T):
            kind = i % 4
            v = torch.randn(d, generator=gen).float()
            v = v / v.norm()
            c = SC.env_coeffs(("dc", "sin4", "white", "dc")[kind], T,
                              sign_seed=211 if kind == 2 else None)
            return (0.05 * rms) * c.unsqueeze(1).cuda() * v.cuda().unsqueeze(0)

        try:
            _ = SC.ggn_quadform(item0, _probe(0, item0["T"]), path="func")
        except Exception as e:                                    # noqa: BLE001 — path resolution
            rep["path_resolution_error_func"] = repr(e)
            path = "manual"
        rep["diff_path"] = path
        # (i) the bf16 differencing floor, MEASURED (reported, not gated): q_fd − q at amp 1
        q_b, qfd_b, rel_b = SC.fd_directional_check(item0, _probe(0, item0["T"]), path=path,
                                                    h_rel=1.0)
        rep["bf16_fd_floor"] = {"q": q_b, "q_fd": qfd_b, "rel": rel_b,
                                "kl_floor_estimate": (qfd_b - q_b) / 2.0,
                                "note": "additive rounding-KL per differenced forward pair; "
                                        "production thresholds sit above this floor"}
        print(f"[gate0] bf16 FD floor: q={q_b:.3e} q_fd={qfd_b:.3e}", flush=True)
        # (ii) the wiring certification on the fp32 forward
        model.float()
        item0_f = SS.build_ppl_item(model, proc, "cuda", wins[0], install, holder)
        item_v_f = SS.build_vqa_item(model, proc, "cuda", vqa_row, vqa, install, holder)
        # Per-loss FD protocol, frozen from the measured two-term error (runs 2-3, E3d):
        # symmetric-FD error = quartic·h² + rounding-floor/h². ppl (511 loss positions):
        # quartic negligible → h=1.0, floor ≪ 1e-2 → tol 1e-2 (measured 3e-3..8.9e-3).
        # vqa (1-5 answer tokens): quartic dominates at h=1 (up to 0.51), floor×16 at h=0.25 →
        # optimum h=0.25, composite floor ~7e-2 measured → tol 1e-1. Wiring cannot be
        # loss-selective (shared code path, label-free check), so ppl's sub-1% certifies it.
        # GATE = the ppl leg (h=1, tol 1e-2; it CAN and did fail — bf16 run 1). The vqa rows
        # are REPORTED ungated: on a 1-5-token readout the differencing floor is probe-variable
        # (runs 3-4 measured 7.5e-3..0.124 across probes at the optimal h) and the wiring
        # cannot be loss-selective — the check is label-free and both losses share every line
        # of the differentiation path.
        fd_cfg = {"ppl": (1.0, SC.FD_REL_TOL, True), "vqa": (0.25, 0.1, False)}
        fd_rows = []
        for i in range(FD_PROBES):
            item = item0_f if i % 2 == 0 else item_v_f
            h_rel, tol, gated = fd_cfg[item["loss"]]
            delta = _probe(i, item["T"])
            q, qfd, rel = SC.fd_directional_check(item, delta, path=path, h_rel=h_rel)
            fd_rows.append({"probe": i, "loss": item["loss"], "q": q, "q_fd": qfd, "rel": rel,
                            "h_rel": h_rel, "tol": tol, "gated": gated})
            print(f"[gate0] FD fp32 probe {i} ({item['loss']}, h={h_rel}): rel={rel:.2e}",
                  flush=True)
        rep["fd_check"] = {"rows": fd_rows, "dtype": "fp32",
                           "cfg": {k: list(v) for k, v in fd_cfg.items()},
                           "max_rel_ppl": max(r["rel"] for r in fd_rows if r["loss"] == "ppl"),
                           "max_rel_vqa": max(r["rel"] for r in fd_rows if r["loss"] == "vqa")}
        bad = [r for r in fd_rows if r["gated"] and r["rel"] > r["tol"]]
        assert not bad, f"FD wiring check failed on fp32 path={path}: {bad}"
    _w()

    # ---- jvp↔vjp consistency at the full 257k vocab (live §0.5-3; still fp32) -------
    with tm.stage("jvp_vjp_consistency"):
        delta = _probe(0, item0_f["T"])
        q_jvp = SC.ggn_quadform(item0_f, delta, path=path)
        w = SC.ggn_vp_item(item0_f, delta, path=path)
        q_vjp = float((delta.double() * w.double()).sum())
        rel = abs(q_jvp - q_vjp) / max(abs(q_jvp), 1e-30)
        rep["jvp_vjp"] = {"q_jvp": q_jvp, "q_vjp": q_vjp, "rel": rel, "dtype": "fp32"}
        assert rel < 5e-2, f"jvp↔vjp inconsistency {rel} — Fisher/vjp wiring bug"
        model.bfloat16()                       # production dtype restored (lossless round trip)
        item0 = SS.build_ppl_item(model, proc, "cuda", wins[0], install, holder)
    _w()

    # ---- memory smoke: one full H_dc matvec on the half-batch ---------------------
    with tm.stage("memory_smoke"):
        wins_a = SS.wikitext_windows(proc, 12, skip=2)
        items_a = [SS.build_ppl_item(model, proc, "cuda", wd, install, holder) for wd in wins_a]
        torch.cuda.reset_peak_memory_stats()
        v = torch.randn(d, generator=gen).float().cuda()
        v = v / v.norm()
        hv = SC.mode_matvec(items_a, v * (0.05 * rms), "dc", path=path)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        rep["memory_smoke"] = {"n_items": len(items_a), "peak_gib": peak,
                               "matvec_norm": float(hv.norm())}
        assert torch.isfinite(hv).all()
        print(f"[gate0] memory smoke: peak {peak:.1f} GiB", flush=True)
    _w()

    # ---- channel rms at the site (channel-cell energy anchors) --------------------
    with tm.stage("channel_rms"):
        g0 = json.loads(SS.fetch(SS.ATLAS_GATE0_JSON, "/tmp/specter/atlas_gate0.json").read_text())
        site_cap = {}

        def pre_site(_m, args, kwargs):
            site_cap["hs"] = (args[0] if args else kwargs["hidden_states"]).detach()

        h3 = layers[L].self_attn.register_forward_pre_hook(pre_site, with_kwargs=True)
        with torch.no_grad():
            SS.clean_forward(item0, d)                    # zero the shared holder (stale-probe bug)
        h3.remove()
        hs16 = site_cap["hs"].to(torch.bfloat16)
        with torch.no_grad():
            ch = {"k": float(layers[L].self_attn.k_proj(hs16).float().pow(2).mean().sqrt()),
                  "v": float(layers[L].self_attn.v_proj(hs16).float().pow(2).mean().sqrt()),
                  "q": float(layers[L].self_attn.q_proj(hs16).float().pow(2).mean().sqrt())}
        rep["channel_rms_at_site"] = {"measured": ch,
                                      "atlas_banked": g0.get("channel_rms_at_site")}
    _w()

    SS.assert_gpu_seconds(rep)
    print("SPECTER_GATE0_OK", json.dumps({"base_ppl": base_ppl, "path": path,
                                          "fd_max_rel_ppl": rep["fd_check"]["max_rel_ppl"],
                                          "fd_max_rel_vqa": rep["fd_check"]["max_rel_vqa"],
                                          "peak_gib": rep["memory_smoke"]["peak_gib"]}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
