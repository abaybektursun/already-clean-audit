"""SPECTER Arm O (GPU) — the object itself: Lanczos top-64 per (site, ω, loss) with split-half
A1/A2 + held-out B, kernel Rayleigh censuses (radial family, counter-rotated key-coherent slice),
channel censuses (M4's measured channel constants), Hutchinson sub-band traces, tr(H²), the
GGN-vs-Hessian gap, and — on the L0 pass — the C-4 LIBRARY bank (every named object as a frame,
content-hashed) plus u_dc for C-1.

Frozen protocol (journal E3):
  * batch A = 24 items/loss (A1 = first 12, A2 = next 12), held-out B = 12; wikitext windows for
    ppl (invcore construction), teacher-forced VQA items from the specter| DISCOVERY half.
  * dc/sin1/sin4: Lanczos(m=72) on A1 and A2. white: sign-set 211 on A1+A2, sign-set 212 on A1
    (the agreement row). Held-out Rayleigh evaluated for stable-cluster members within the top
    16 Ritz pairs. Convergence: top-8 residuals ≤ 0.1·λ₁; one restart at m=144 (R-O1 protocol).
  * traces + tr(H²) + censuses measured on FULL batch A; gap (H_full − GGN) on 4 dc probes, ppl.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.specter_o --args "--sites L0" \
      --type p5.4xlarge,g6e.8xlarge --no-weights --s3-prefix results/specter_o --budget-min 300
"""
import argparse
import json
import os
import pathlib

M_RESTART = 144
TOP_EVAL_HELDOUT = 16
CONV_RESID_FRAC = 0.10
N_GAP_PROBES = 4


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", default="L0")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.aws_pg.jobs import _specter_shared as SS
    from infra.pg import atlas_core as AC
    from infra.pg import atlas_fit as AF
    from infra.pg import bispec_core as BC
    from infra.pg import inherit_pg1 as IP
    from infra.pg import specter_core as SC

    freeze = SS.assert_frozen("o")
    RS.install_pg()
    sites = [s.strip() for s in a.sites.split(",")]
    rep = {"object": "specter_o_v1", "sites": sites, "freeze": freeze}
    tm = SS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    g0 = json.loads(SS.fetch(f"{SS.SPECTER_GATE0}/gate0.json", "/tmp/specter/g0.json").read_text())
    assert g0.get("green"), "Gate-0 not green — Arm O refuses to run"
    path = g0["diff_path"]
    rep["diff_path"] = path

    with tm.stage("load"):
        sha, _ = RS.pinned_pg1_sha("/tmp/specter")
        model, proc = IP.load_pg1(sha=sha)
        layers = IP.decoder_layers(model)
        assert sha == g0["model"]["sha"], "model sha drift vs Gate-0"
    d = IP.D
    dev = "cuda"

    # ---- items (manifests hashed; zero-overlap re-assert) --------------------------
    with tm.stage("items"):
        vqa_task, _ = IP.pg1_tasks()
        from infra.aws_pg.jobs._regrain_shared import GRAIN_GATE0
        it = json.loads(SS.fetch(f"{GRAIN_GATE0}/items_vqa.json",
                                 "/tmp/specter/items_vqa.json").read_text())
        rows, hashes = IP.load_items(vqa_task, it["stream_window"])
        split = SS.specter_split(hashes)
        disc_idx = list(split["disc_idx"])
        sealed_set = {hashes[i] for i in split["conf_idx"]}
        op_idx = disc_idx[:SS.N_OP_VQA + SS.N_OP_HELD]
        assert not (sealed_set & {hashes[i] for i in op_idx}), "sealed∩operator — halt"
        wins_a = SS.wikitext_windows(proc, SS.N_OP_PPL, skip=0)
        wins_b = SS.wikitext_windows(proc, SS.N_OP_HELD, skip=SS.N_OP_PPL)
        rep["manifests"] = {
            "ppl_A": SS.batch_manifest([w["hash"] for w in wins_a]),
            "ppl_B": SS.batch_manifest([w["hash"] for w in wins_b]),
            "vqa_A": SS.batch_manifest([hashes[i] for i in op_idx[:SS.N_OP_VQA]]),
            "vqa_B": SS.batch_manifest([hashes[i] for i in op_idx[SS.N_OP_VQA:]])}
    _w()

    bank = {}
    lib = {}

    for site in sites:
        L = int(site[1:])
        install, holder = SC.make_residual_injector(layers, L)
        with tm.stage(f"{site}/items"):
            items = {"ppl": [SS.build_ppl_item(model, proc, dev, wd, install, holder)
                             for wd in wins_a],
                     "vqa": [SS.build_vqa_item(model, proc, dev, rows[i], vqa_task, install,
                                               holder) for i in op_idx[:SS.N_OP_VQA]]}
            items_b = {"ppl": [SS.build_ppl_item(model, proc, dev, wd, install, holder)
                               for wd in wins_b],
                       "vqa": [SS.build_vqa_item(model, proc, dev, rows[i], vqa_task, install,
                                                 holder) for i in op_idx[SS.N_OP_VQA:]]}
        # bands: banked Σ (L0/L14 bispec bank; L8 the atlas Gate-0e bank)
        with tm.stage(f"{site}/bands"):
            if site == "L8":
                import torch as _t
                bank_l8 = _t.load(SS.fetch(f"{SS.ATLAS_L8_SIGMA}/sigma_bank.pt",
                                           "/tmp/specter/sigma_l8.pt"),
                                  map_location="cpu", weights_only=False)
                sig = bank_l8["L8"]
            else:
                sig = SS.load_sigma_bank(site, workdir="/tmp/specter")
            bands = BC.eig_bands(sig["Sigma"])
            sb = AF.subband_slices(bands["band_slices"], bands["spectrum"])
            basis32 = bands["basis"].float()
            rms = float((torch.diagonal(torch.as_tensor(sig["Sigma"])).sum() / d).sqrt())
        site_rms = g0["identity"]["site_rms"] if site == "L0" else rms
        eps_probe = 0.05 * site_rms

        def matvec_factory(its, mode, ss_seed):
            # mode_matvec is exactly LINEAR in v (jvp/vjp are linear maps), so the eps_probe
            # scaling cancels exactly; it is applied and divided out only to keep the bf16
            # forward tangents well-conditioned.
            def mv(v):
                return SC.mode_matvec(its, v.to(dev) * eps_probe, mode,
                                      sign_seed=ss_seed, path=path) / eps_probe
            return mv

        for loss in SS.LOSSES:
            A = items[loss]
            A1, A2 = A[:len(A) // 2], A[len(A) // 2:]
            B = items_b[loss]
            for mode in SC.MODES:
                key = f"{site}/{mode}/{loss}"
                runs = {}
                with tm.stage(f"{key}/lanczos"):
                    specs = [("A1", A1, SC.WHITE_SIGN_SEEDS[0]), ("A2", A2, SC.WHITE_SIGN_SEEDS[0])]
                    if mode == "white":
                        specs.append(("A1ss2", A1, SC.WHITE_SIGN_SEEDS[1]))
                    for name, its, ss_seed in specs:
                        ss = ss_seed if mode == "white" else None
                        res = SC.lanczos(matvec_factory(its, mode, ss), d, m=SC.LANCZOS_M,
                                         seed=17, device=dev)
                        top8_ok = bool((res["resid"][:8] <= CONV_RESID_FRAC *
                                        max(float(res["evals"][0]), 1e-30)).all())
                        if not top8_ok:
                            res = SC.lanczos(matvec_factory(its, mode, ss), d, m=M_RESTART,
                                             seed=17, device=dev)
                            res["restarted"] = True
                        runs[name] = res
                        print(f"[o] {key}/{name}: top evals "
                              f"{[round(float(x), 5) for x in res['evals'][:4]]}", flush=True)
                with tm.stage(f"{key}/stability"):
                    ss1 = SC.WHITE_SIGN_SEEDS[0] if mode == "white" else None

                    def ray_b(u, _its=B, _m=mode, _ss=ss1):
                        return SC.mode_quadform(_its, u.to(dev) * eps_probe, _m, sign_seed=_ss,
                                                path=path) / (eps_probe ** 2)
                    spikes, clusters = SC.stable_spikes(runs["A1"], runs["A2"], ray_b,
                                                        k=TOP_EVAL_HELDOUT)
                    conv = bool((runs["A1"]["resid"][:8] <= CONV_RESID_FRAC *
                                 max(float(runs["A1"]["evals"][0]), 1e-30)).all())
                    agree = None
                    if mode == "white":
                        agree = SC.match_cluster_overlaps(runs["A1"], runs["A1ss2"], k=8)
                with tm.stage(f"{key}/traces"):
                    traces_sb, permode_sb, probe_qs_sb = [], [], []
                    gen = torch.Generator(device="cpu").manual_seed(310 + L)
                    for b, (lo, hi) in enumerate(sb):
                        kb = hi - lo
                        qs = []
                        for p in range(SC.N_TRACE_PROBES):
                            g = torch.randn(kb, generator=gen)
                            g = (g / g.norm()).to(dev)
                            z = (basis32[:, lo:hi].to(dev) @ g)
                            qs.append(SC.mode_quadform(A, z * eps_probe, mode, sign_seed=ss1,
                                                       path=path) / (eps_probe ** 2))
                        probe_qs_sb.append(qs)
                        permode_sb.append(sum(qs) / len(qs))
                        traces_sb.append(permode_sb[-1] * kb)
                    tr_total = sum(traces_sb)
                    g2 = torch.Generator(device="cpu").manual_seed(410 + L)
                    h2 = []
                    for p in range(SC.N_TR2_PROBES):
                        z = (torch.randint(0, 2, (d,), generator=g2) * 2 - 1).float().to(dev)
                        hz = matvec_factory(A, mode, ss1)(z)
                        h2.append(float((hz.double() ** 2).sum()))
                    tr_h2 = sum(h2) / len(h2)
                bank[key] = {
                    "evals_A1": runs["A1"]["evals"], "evecs_A1": runs["A1"]["evecs"].cpu(),
                    "resid_A1": runs["A1"]["resid"], "evals_A2": runs["A2"]["evals"],
                    "evecs_A2": runs["A2"]["evecs"].cpu(), "resid_A2": runs["A2"]["resid"],
                    "spikes": spikes, "clusters": clusters, "converged": conv,
                    "restarted": bool(runs["A1"].get("restarted")),
                    "white_signset_agreement": agree,
                    "traces_sb": traces_sb, "permode_sb": permode_sb,
                    "probe_qs_sb": probe_qs_sb, "trace_total": tr_total,
                    "tr_h2": tr_h2, "participation_ratio": SC.participation_ratio(tr_total, tr_h2),
                    "eps_probe": eps_probe}
                rep.setdefault("ro1", {})[key] = SC.adjudicate_ro1(
                    n_stable=len(spikes), converged=conv,
                    restart_used=bool(runs["A1"].get("restarted")))
                rep.setdefault("summary", {})[key] = {
                    "top4_evals": [float(x) for x in runs["A1"]["evals"][:4]],
                    "n_stable": len(spikes), "trace_total": tr_total,
                    "pr": bank[key]["participation_ratio"]}
                _w()

            # ---- R(site) per loss: the quadratic coherence dial ---------------------
            rep.setdefault("dial", {})[f"{site}/{loss}"] = {
                "R": bank[f"{site}/dc/{loss}"]["trace_total"] /
                     max(bank[f"{site}/white/{loss}"]["trace_total"], 1e-30)}
            _w()

            # ---- kernel censuses (R-O2) ---------------------------------------------
            with tm.stage(f"{site}/{loss}/census"):
                gen = torch.Generator(device="cpu").manual_seed(510 + L)
                # radial family: field δ_t = x̂_t (clean-forward frozen fields) vs the
                # coherence-matched tangential twin (AM-6) — construction-matched bulk
                xcache = {}

                def _cap(i):
                    def hook(_m, args, kwargs):
                        hs = args[0] if args else kwargs["hidden_states"]
                        xcache[i] = hs[0].detach().float()
                    return hook
                for i, itx in enumerate(A):
                    h = layers[L].register_forward_pre_hook(_cap(i), with_kwargs=True)
                    with torch.no_grad():
                        SS.clean_forward(itx, d)          # zero the shared holder (stale-probe bug)
                    h.remove()
                rad_fields = [eps_probe * xcache[i] / xcache[i].norm(dim=-1, keepdim=True)
                              for i in range(len(A))]
                q_rad = SC.field_quadform(A, rad_fields, path=path) / (eps_probe ** 2)
                q_tang = []
                for s in range(8):
                    g = torch.randn(d, generator=gen)
                    u = (g / g.norm()).to(dev)
                    tf = []
                    for i in range(len(A)):
                        x = xcache[i]
                        xh = x / x.norm(dim=-1, keepdim=True)
                        t = u.unsqueeze(0) - (xh @ u).unsqueeze(1) * xh
                        tf.append(eps_probe * t / t.norm(dim=-1, keepdim=True))
                    q_tang.append(SC.field_quadform(A, tf, path=path) / (eps_probe ** 2))
                bulk_tang = sum(q_tang) / len(q_tang)
                rep.setdefault("ro2", {})[f"{site}/{loss}/radial"] = dict(
                    SC.adjudicate_ro2_radial(mean_rayleigh=q_rad, bulk_mean=bulk_tang),
                    q_radial=q_rad, bulk_tangential=bulk_tang, q_tang_draws=q_tang)
                # key-coherent slice at k_proj (the exact theorem) vs Haar-dc bulk at k_proj
                inst_k, hold_k = SC.make_proj_injector(layers[L], "k_proj")
                if loss == "ppl":
                    items_k = [SS.build_ppl_item(model, proc, dev, wd, inst_k, hold_k)
                               for wd in wins_a]
                else:
                    items_k = [SS.build_vqa_item(model, proc, dev, rows[i], vqa_task, inst_k,
                                                 hold_k) for i in op_idx[:SS.N_OP_VQA]]
                hd = g0["model"]["head_dim"]
                theta = g0["model"]["rope_theta"]
                crms = g0["channel_rms_at_site"]["measured"]["k"]
                epsk = 0.05 * crms * (hd ** 0.5)
                q_key, q_bulk_k = [], []
                for s in range(SC.CENSUS_N_PROBES // 2):
                    g = torch.randn(hd, generator=gen, dtype=torch.float64)
                    u = g / g.norm()
                    fn = AC.delta_postrope_coherent(u * epsk, base=theta)
                    fields = [fn(torch.arange(itk["T"]), dev) for itk in items_k]
                    q_key.append(SC.field_quadform(items_k, fields, path=path) / (epsk ** 2))
                    gb = torch.randn(hd, generator=gen, dtype=torch.float64)
                    ub = (gb / gb.norm()) * epsk
                    fb = [ub.float().to(dev).unsqueeze(0).expand(itk["T"], hd).contiguous()
                          for itk in items_k]
                    q_bulk_k.append(SC.field_quadform(items_k, fb, path=path) / (epsk ** 2))
                mean_key = sum(q_key) / len(q_key)
                mean_bulk = sum(q_bulk_k) / len(q_bulk_k)
                rep["ro2"][f"{site}/{loss}/key_coherent"] = dict(
                    SC.adjudicate_ro2_key(mean_rayleigh=mean_key, bulk_mean=mean_bulk),
                    q_key_draws=q_key, q_bulk_draws=q_bulk_k)
                halt = rep["ro2"][f"{site}/{loss}/key_coherent"]["verdict"] == "HALT-instrument"
                assert not halt, "R-O2 key-coherent ≥0.1×bulk — instrument halt (spec §6)"
                # channel censuses: M4's measured constants (k_low/k_high dc+white, v, q)
                chan = {}
                chan_seed = {"k_low": 701, "k_high": 702, "v": 703, "q": 704}   # frozen (never
                for ch in ("k_low", "k_high", "v", "q"):                        # hash(): A3)
                    for cmode in ("dc", "white"):
                        crms_c = g0["channel_rms_at_site"]["measured"]["k" if ch.startswith("k")
                                                                       else ch]
                        epsc = 0.05 * crms_c * (hd ** 0.5)
                        attr, fn = AC.channel_delta_fn(ch, cmode, epsc, chan_seed[ch], hd,
                                                       g0["model"]["n_heads"], theta)
                        inst_c, hold_c = SC.make_proj_injector(layers[L], attr)
                        if loss == "ppl":
                            items_c = [SS.build_ppl_item(model, proc, dev, wd, inst_c, hold_c)
                                       for wd in wins_a[:6]]
                        else:
                            items_c = [SS.build_vqa_item(model, proc, dev, rows[i], vqa_task,
                                                         inst_c, hold_c)
                                       for i in op_idx[:6]]
                        fields = [fn(torch.arange(itc["T"]), dev) for itc in items_c]
                        chan[f"{ch}/{cmode}"] = SC.field_quadform(items_c, fields,
                                                                  path=path) / (epsc ** 2)
                rep.setdefault("channel_census", {})[f"{site}/{loss}"] = chan
            _w()

        # ---- GGN-vs-Hessian gap (dc probes, ppl loss) --------------------------------
        with tm.stage(f"{site}/gap"):
            gen = torch.Generator(device="cpu").manual_seed(610 + L)
            gaps = []
            for p in range(N_GAP_PROBES):
                g = torch.randn(d, generator=gen)
                v = (g / g.norm()).to(dev) * eps_probe
                itx = items["ppl"][p % len(items["ppl"])]
                c = SC.env_coeffs("dc", itx["T"], device=dev)
                delta = c.unsqueeze(1) * v.unsqueeze(0)
                qg = SC.ggn_quadform(itx, delta, path=path) / itx["T"] / (eps_probe ** 2)
                hv = SC.hessian_vp_item(itx, delta)
                qh = float((delta.double() * hv.double()).sum()) / itx["T"] / (eps_probe ** 2)
                gaps.append({"q_ggn": qg, "q_hess": qh, "gap": qh - qg})
            rep.setdefault("ggn_gap", {})[site] = gaps
        _w()

        # ---- the L0 library bank + u_dc (C-1/C-4 inputs) ------------------------------
        if site == "L0":
            with tm.stage("library"):
                key = "L0/dc/ppl"
                sp = bank[key]["spikes"]
                if sp:
                    lib["u_dc"] = bank[key]["evecs_A1"][:, sp[0]["idx"]].clone()
                    rep["u_dc_stable"] = True
                else:
                    # no stable spike: bank the top Ritz vector, flagged — R-C1's u_dc_stable
                    # input goes False and its branches route accordingly
                    lib["u_dc"] = bank[key]["evecs_A1"][:, 0].clone()
                    rep["u_dc_stable"] = False
                nr = torch.load(SS.fetch(SS.BISPEC_V1R_ROWS, "/tmp/specter/named_rows.pt"),
                                map_location="cpu", weights_only=False)
                lib["U_unt_residual"] = nr["U_unt_residual"].float()
                lib["U_unt_qk"] = nr["U_unt_qk"].float()
                lib["gamma_dir"] = (nr["gamma"] / nr["gamma"].norm()).float()
                lib["mean_dir"] = (nr["hbar_residual"] / nr["hbar_residual"].norm()).float()
                lib["e1"] = basis32[:, :1].clone()
                lib["sigma_basis_L0"] = basis32.cpu()
                lib["band_projectors_meta"] = {"band_slices": bands["band_slices"],
                                               "sb_slices": sb}
                import numpy as np
                man = json.loads(SS.fetch(f"{SS.COMWRITE_ANATOMY}/metrics.json",
                                          "/tmp/specter/cw.json").read_text())
                for t in ("o_proj", "down_proj"):
                    arr = np.load(SS.fetch(f"{SS.COMWRITE_ANATOMY}/scratch/ufull__{t}__L0.npy",
                                           f"/tmp/specter/ufull_{t}_L0.npy"))
                    lib[f"U_wr_{t}"] = torch.from_numpy(arr).float()
                masks = torch.load(SS.fetch(SS.ATLAS_B_MASKS, "/tmp/specter/gate_masks.pt"),
                                   map_location="cpu", weights_only=False)
                for Lg, m in masks["flipped"].items():
                    if int(Lg) == 0 and bool(m.any()):
                        Wg = layers[0].mlp.gate_proj.weight.detach().float().cpu()[m]
                        Q, _ = torch.linalg.qr(Wg.transpose(0, 1))
                        lib["dormant_gate_L0"] = Q[:, :min(Q.shape[1], 64)]
                for Ls in (0, 8, 14):
                    for wname in ("k_proj", "v_proj"):
                        W = getattr(layers[Ls].self_attn, wname).weight.detach().float().cpu()
                        _, _, Vh = torch.linalg.svd(W, full_matrices=False)
                        lib[f"{wname}_rowspace_L{Ls}"] = Vh.transpose(0, 1)
                v1, sigma1, v1_meta = SS.build_v1(f"{SS.S3_BUCKET}/results/regrain_pipeA/out/harvest",
                                                  f"{SS.S3_BUCKET}/results/regrain_c_disc/out",
                                                  device=dev, workdir="/tmp/specter")
                lib["v1"] = v1.float()
                rep["v1_meta"] = v1_meta
                # B*-union frame + task footprint (regrain_g recipe): same banks build_v1 fetched
                from infra.aws_pg.jobs._regrain_shared import GRAIN_DICTS_L0, dict_bank_name
                from infra.pg import grain_bsf as GB
                from infra.pg import regrain_core as RC
                cd = json.loads(pathlib.Path("/tmp/specter/c_disc.json").read_text())
                bstar = [int(x) for x in cd["bstar"]]
                med_s = int(cd["median_seed"])
                nm = dict_bank_name("primary", 1, med_s)
                src = (f"{GRAIN_DICTS_L0}/{nm}" if med_s == 0
                       else f"{SS.S3_BUCKET}/results/regrain_c_disc/out/{nm}")
                ae, _c, _q, _m2 = GB.load_dict_bank(SS.fetch(src, f"/tmp/specter/{nm}"))
                ae = ae.to(dev).eval()
                D_full = RC.decoder_matrix(ae).float()
                Qb, _ = torch.linalg.qr(D_full[:, bstar].double().cpu())
                lib["bstar_union"] = Qb.float()
                meta_h = json.loads(pathlib.Path("/tmp/specter/harvest_meta.json").read_text())
                shard0 = meta_h["shards"]["L0"][0]
                X = torch.load(f"/tmp/specter/shards/{shard0}", map_location="cpu",
                               weights_only=False)["X"][:40000].float()
                with torch.inference_mode():
                    z = ae.encode(X.to(dev), use_threshold=True)
                    a_j = z.abs().mean(0)
                top8 = torch.topk(a_j, 8).indices.cpu()
                Qf, _ = torch.linalg.qr(D_full[:, top8].double().cpu())
                lib["task_footprint"] = Qf.float()
                rep["task_footprint_atoms"] = [int(i) for i in top8]
            _w()

    with tm.stage("save_bank"):
        bank["_meta"] = {"T_mean_ppl": sum(it["T"] for it in items["ppl"]) / len(items["ppl"]),
                         "T_mean_vqa": sum(it["T"] for it in items["vqa"]) / len(items["vqa"]),
                         "sites": sites, "diff_path": path}
        bank_hash = SS.save_operator_bank(out / "operator_bank.pt", bank)
        lib_hash = SS.save_operator_bank(out / "library_bank.pt", lib) if lib else None
        rep["bank_sha256"] = bank_hash
        rep["library_sha256"] = lib_hash
    SS.assert_gpu_seconds(rep)
    print("SPECTER_O_OK", json.dumps({k: v for k, v in rep["summary"].items()}, default=str)[:2000])
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
