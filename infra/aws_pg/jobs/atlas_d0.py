"""ATLAS Arm D0 (CPU, zero forwards) — everything the banked artifacts already contain:

  * R-D: dispersion-dimension calibration on synthetic forms + the d_eff sweep over every
    reconstructible banked multi-draw family (the program's first quantitative map of
    susceptibility dimensionality by mechanism)
  * R-B1: the 9 banked fully-coherent P1 draws regenerated bit-exactly from the seed formula
    (`manual_seed(11 + 1000·ss)`, AM-3), verified against the banked realized R², then regressed:
    τ(damage, ⟨δ̄,v₁⟩₊) vs the e₁ and norm comparators — the one-boundary test, zero forwards
  * MQA K-read/band overlap from the weights + banked Σ
  * T7/R-N: the Bingham dispersion curve VERIFIED on the banked L0/L14 frontiers, then the L8
    curve PREDICTED from the L8 spectrum and written to `pre_unblind_predictions.json` BEFORE the
    L8 house-sampler curve is drawn (both in this job; the prediction block is written and
    uploaded first — §9.3)
  * the corpus manifest `atlas_corpus.json`: every banked (perturbation → damage) cell at
    L0/n_seq=120 with a reconstructible descriptor; every exclusion counted with its reason

  ./infra/aws_pg/pg run infra.aws_pg.jobs.atlas_d0 --type r7i.4xlarge --no-weights \
      --s3-prefix results/atlas_d0 --budget-min 240 \
      --args "--l8-sigma-s3 s3://pg-lineage-769063704696/results/atlas_gate0/out/sigma_l8"
"""
import argparse
import json
import math
import os
import pathlib

E_GRID = (0.05, 0.15, 0.30, 0.50, 0.70, 0.90)     # E/E_max rungs for the R-N curves
BINGHAM_K = 25
N_HOUSE_DRAWS = 8


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--l8-sigma-s3", required=True, help="the atlas_gate0 L8 sigma.json/bank prefix")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # BEFORE any torch/transformers/huggingface_hub import: HF_HUB_OFFLINE is frozen into a
    # module constant at first import (reference_pg_hf_online_before_transformers)
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _atlas_shared as AS
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import atlas_bingham as AB
    from infra.pg import atlas_core as AC
    from infra.pg import atlas_fit as AF
    from infra.pg import bispec_core as BC
    from infra.pg import regrain_core as RC

    AS.assert_frozen("d0")
    rep = {"object": "atlas_d0_v1"}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    def _upload_now(path):
        # the §9.3 pre-unblind write: push the prediction to S3 BEFORE the sampled curve exists
        os.system(f"aws s3 cp {path} {AS.ATLAS_D0}/{path.name} --region us-east-1")

    # ================= banked inputs, all fetched + hashed =================
    def fetch_json(uri, name):
        p = RS.fetch(uri, f"/tmp/atlas/{name}")
        return json.loads(p.read_text()), AF.sha256_file(p), uri

    k_art, k_sha, _ = fetch_json(AS.BISPEC_K, "bispec_k.json")
    ks_art, ks_sha, _ = fetch_json(AS.BISPEC_K_SEALED, "bispec_k_sealed.json")
    a_art, a_sha, _ = fetch_json(AS.BISPEC_A, "bispec_a.json")
    s_art, s_sha, _ = fetch_json(AS.BISPEC_S, "bispec_s.json")
    sig_art, sig_sha, _ = fetch_json(f"{AS.BISPEC_SIGMA}/sigma.json", "sigma.json")
    fr_art, fr_sha, _ = fetch_json(AS.REGRAIN_FRONTIER, "frontier.json")
    ce_art, ce_sha, _ = fetch_json(AS.REGRAIN_C_ENERGY, "regrain_c_energy.json")

    bank = AS.load_sigma_bank("L0")
    Sigma, hbar, basis, spectrum = bank["Sigma"], bank["hbar"], bank["basis"], bank["spectrum"]
    d = Sigma.shape[0]
    bands5 = [tuple(s) for s in bank["band_slices"]]
    sb = AF.subband_slices(bands5, spectrum)
    rep["subbands"] = {"slices": sb, "counts": AF.SUBBAND_COUNTS,
                      "ranks": [hi - lo for lo, hi in sb]}
    base_ppl = float(k_art["reference"]["base_ppl"])
    rms = float(k_art["site_rms"])
    _w()

    # harvest rows (the same 8192-row shard slice bispec_k measured realized R² on)
    meta = json.loads(RS.fetch(f"{AS.REGRAIN_HARVEST}/harvest_meta.json",
                               "/tmp/atlas/harvest_meta.json").read_text())
    shard0 = meta["shards"]["L0"][0]
    X = torch.load(RS.fetch(f"{AS.REGRAIN_HARVEST}/{shard0}", f"/tmp/atlas/shards/{shard0}"),
                   map_location="cpu", weights_only=False)["X"][:8192].double()
    xbar = X.mean(0)
    xhat = X / X.norm(dim=1, keepdim=True)
    radial_mass_by_sb = []
    for s_i in range(len(sb)):
        U = AF.subband_projector(basis, sb, s_i)
        radial_mass_by_sb.append(float(((xhat @ U) ** 2).sum(1).mean()))
    rep["radial_mass_by_subband"] = radial_mass_by_sb
    _w()

    # ================= v₁ (CPU rebuild) + e₁ =================
    v1, sigma1, v1_meta = AS.build_v1(AS.REGRAIN_HARVEST, AS.REGRAIN_CDISC, device="cpu")
    e1 = basis[:, 0].double()
    rep["v1"] = {**v1_meta, "cos_with_e1": float(abs(v1 @ e1))}
    torch.save({"v1": v1, "sigma1": sigma1}, out / "v1.pt")
    _w()

    # ================= R-B1: regenerate the 9 coherent draws (AM-3) =================
    def coherent_delta(ss, sigma_abs, dtype=torch.float32):
        """The banked coherent draw. PARENT INCONSISTENCY, verified here: bispec_k's hook injected
        the FP32 randn from seed 11+1000·ss, but its realized-R² row used the FP64 randn from the
        same seed — different vectors of the same law (Δr² up to ~2e-3). The fp64 path reproduces
        the banked invariant bit-exactly; the fp32 path is the vector the model actually saw and
        is the R-B1 regressor."""
        gen = torch.Generator(device="cpu").manual_seed(11 + 1000 * ss)
        return (sigma_abs * torch.randn(d, generator=gen, dtype=dtype)).double()

    lv = k_art["frozen_levels"]
    lev_sigma = {"benign": lv["benign"]["level"] * rms, "mid": lv["mid"]["level"] * rms}
    draws, recon = [], []
    for half, art in (("discovery", k_art), ("sealed", ks_art)):
        for c in art["cells"]:
            if c["family"] in ("P1", "P1g") and c["arm"] == "coherent" and c.get("g") == -1:
                if half == "discovery" and c["level"] not in ("benign", "mid"):
                    continue
                if half == "sealed" and c.get("level") != "mid":
                    continue
                delta = coherent_delta(int(c["draw"]), lev_sigma[c["level"]])
                if c.get("realized_r2") is not None:
                    delta64 = coherent_delta(int(c["draw"]), lev_sigma[c["level"]],
                                             dtype=torch.float64)
                    r2_exact = RC.r2_of(X, X + delta64.unsqueeze(0), xbar)
                    r2_hook = RC.r2_of(X, X + delta.unsqueeze(0), xbar)
                    recon.append({"cell": c["cell"], "half": half, "draw": int(c["draw"]),
                                  "banked_r2": c["realized_r2"],
                                  "recon_r2_fp64_row_path": r2_exact,
                                  "recon_r2_fp32_hook_path": r2_hook,
                                  "exact_diff": abs(r2_exact - c["realized_r2"]),
                                  "fp32_fp64_draw_gap": abs(r2_hook - r2_exact)})
                    assert abs(r2_exact - c["realized_r2"]) < 1e-6, \
                        f"seed-formula reconstruction failed for {c['cell']}: " \
                        f"{r2_exact} vs banked {c['realized_r2']}"
                draws.append({"cell": c["cell"], "half": half, "draw": int(c["draw"]),
                              "level": c["level"], "dppl": c["dppl"],
                              "log1p_dppl": math.log1p(max(c["dppl"], -0.99)),
                              "v1_overlap_pos": float(max(float(delta @ v1), 0.0)),
                              "e1_overlap_abs": float(abs(delta @ e1)),
                              "norm": float(delta.norm())})
    assert len(draws) == 9, f"expected the 9 coherent draws (AM-3), found {len(draws)}"
    rep["draw_reconstruction"] = recon
    rb1 = AF.adjudicate_rb1(
        damages=np.array([r["log1p_dppl"] for r in draws]),
        v1_overlap_pos=np.array([r["v1_overlap_pos"] for r in draws]),
        e1_overlap_abs=np.array([r["e1_overlap_abs"] for r in draws]),
        norms=np.array([r["norm"] for r in draws]))
    rep["rb1"] = {"draws": draws, "adjudication": rb1}
    print(f"[d0] R-B1: {rb1['verdict']} tau_v1={rb1['tau_v1']:.3f} p={rb1['p_v1']:.4f} "
          f"tau_e1={rb1['tau_e1']:.3f}", flush=True)
    _w()

    # ================= R-D: calibration + the banked-family sweep =================
    cal = AC.calibrate_d_eff()
    rep["deff_calibration"] = {f"pr{pr}_n{n}": v for (pr, n), v in cal.items()}

    def fam(vals, name, mechanism):
        r = AC.d_eff_from_damages(vals)
        key = min((abs(math.log(max(r["d_eff"], 1e-9) / pr)), pr) for pr in (1, 4, 32, 256))[1]
        n = min((3, 8), key=lambda x: abs(x - r["n"]))
        c = cal[(key, n)]
        return {"family": name, "mechanism": mechanism, **r,
                "calibrated_interval_at_nearest_pr": [c["q05"], c["q95"]], "nearest_pr": key}

    sweep = []
    for lev in ("benign", "mid"):
        for arm, g in (("coherent", -1), ("independent", 1)):
            v = [c["dppl"] for c in k_art["cells"]
                 if c["family"] in ("P1", "P1g") and c["arm"] == arm and c.get("g") == g
                 and c.get("level") == lev]
            if len(v) >= 2:
                sweep.append(fam(v, f"P1_{lev}_{arm}", "full-space gaussian"))
    for r_ in (64, 128):
        for arm in ("coherent", "independent"):
            v = [c["dppl"] for c in k_art["cells"]
                 if c["family"] == "P3" and c["arm"] == arm and c.get("rank") == r_]
            if len(v) >= 2:
                sweep.append(fam(v, f"P3_r{r_}_{arm}", "subspace removal"))
    ce_cells = ce_art.get("cells", {})
    um = [v["dppl"] for k, v in ce_cells.items() if k.startswith("union_matched")]
    if len(um) >= 2:
        sweep.append(fam(um, "R1b_union_matched_dppl", "energy-matched union (delete)"))
    mm = [v["dppl"] for k, v in ce_cells.items() if k.startswith("marginal_matched")]
    if len(mm) >= 2:
        sweep.append(fam(mm, "R1b_marginal_matched_dppl", "marginal-matched union (delete)"))
    rep["deff_sweep"] = sweep
    _w()

    # ================= MQA K-read / band overlap (weights + Σ) =================
    from infra.aws_pg.jobs.invariant_core import PG1_REPO, _load_decoder
    from huggingface_hub import HfApi
    sha = HfApi().model_info(PG1_REPO).sha
    w = _load_decoder(PG1_REPO, sha)
    Wk = w["layers.0.self_attn.k_proj.weight"].double()          # (256, 2048)
    tot = float((Wk ** 2).sum())
    overlap = []
    for b in range(len(bands5)):
        U = BC.band_projector({"basis": basis, "band_slices": bands5}, b)
        overlap.append(float(((Wk @ U) ** 2).sum()) / tot)
    rep["k_read_band_overlap_L0"] = {"per_band": overlap,
                                     "note": "fraction of k_proj row-space Frobenius energy per Σ "
                                             "band (input side); the K-channel's exposure to the "
                                             "banded metric, from weights alone"}
    _w()

    # ================= T7 / R-N: verify on L0/L14, PREDICT L8 =================
    lam_l0 = spectrum.double()
    bank14 = AS.load_sigma_bank("L14")
    lam_l14 = bank14["spectrum"].double()
    l8_sig = json.loads(RS.fetch(f"{a.l8_sigma_s3}/sigma.json", "/tmp/atlas/l8_sigma.json").read_text())
    bank8 = torch.load(RS.fetch(f"{a.l8_sigma_s3}/sigma_bank.pt", "/tmp/atlas/l8_bank.pt"),
                       map_location="cpu", weights_only=False)["L8"]
    lam_l8 = bank8["spectrum"].double()

    def house_curve(Sig, name):
        eig = torch.linalg.eigh(0.5 * (Sig + Sig.transpose(0, 1)))
        ev = eig[0].clamp_min(0.0)
        E_max = float(ev.flip(0)[:BINGHAM_K].sum() / ev.sum())
        rows = []
        for r in E_GRID:
            E = r * E_max
            frames = [BC.sample_level_set(Sig, BINGHAM_K, E, seed=200 + s, eig=eig)[0]
                      for s in range(N_HOUSE_DRAWS)]
            disp = BC.energy_dispersion(frames, Sig)
            rows.append({"E_over_Emax": r, "target_E": E,
                         "energy_mean_distance": disp["mean_distance"]})
            print(f"[d0] house {name} E/Emax={r}: disp={disp['mean_distance']:.4f}", flush=True)
        return {"E_max": E_max, "curve": rows}

    def curve_gaps(model_c, house_c):
        """Per-rung |model − house| over rungs the tilt CERTIFIED (target_missed rungs are
        excluded and COUNTED — a rung the sampler could not reach is not evidence)."""
        gaps, missed = [], 0
        for m, h in zip(model_c["curve"], house_c["curve"]):
            if m.get("target_missed"):
                missed += 1
                continue
            gaps.append(abs(m["energy_mean_distance"] - h["energy_mean_distance"]))
        assert gaps, "every rung missed its tilt target — no comparable curve"
        return gaps, missed

    # (1) verification legs at L0/L14 — model curve vs house curve, residuals disclosed
    verify = {}
    for name, lam, Sig in (("L0", lam_l0, Sigma.double()), ("L14", lam_l14, bank14["Sigma"].double())):
        model_c = AB.bingham_dispersion_curve(lam, BINGHAM_K, E_GRID, n_draws=4, seed=50)
        house_c = house_curve(Sig, name)
        gaps, missed = curve_gaps(model_c, house_c)
        verify[name] = {"model": model_c, "house": house_c, "abs_gaps": gaps,
                        "max_gap": max(gaps), "n_rungs_missed": missed}
        rep["bingham_verify"] = verify
        _w()

    # (2) the L8 PREDICTION — journaled + uploaded BEFORE the L8 house curve is sampled
    pred_l8 = AB.bingham_dispersion_curve(lam_l8, BINGHAM_K, E_GRID, n_draws=4, seed=50)
    const_baseline = [{"E_over_Emax": r,
                       "energy_mean_distance": 0.5 * (verify["L0"]["house"]["curve"][i]["energy_mean_distance"]
                                                      + verify["L14"]["house"]["curve"][i]["energy_mean_distance"])}
                      for i, r in enumerate(E_GRID)]
    pred_block = {"object": "atlas_d0_pre_unblind", "rule": "R-N",
                  "prediction_L8_curve": pred_l8, "constant_baseline": const_baseline,
                  "grid_row": "P9: gap <= 0.10 (H_maxent-sufficient)",
                  "note": "written and uploaded BEFORE the L8 house curve is sampled (§9.3)"}
    pp = out / "pre_unblind_predictions.json"
    pp.write_text(json.dumps(pred_block, indent=2, default=str))
    _upload_now(pp)
    print("[d0] L8 prediction journaled + uploaded; NOW sampling the L8 house curve", flush=True)

    house_l8 = house_curve(bank8["Sigma"].double(), "L8")
    gaps, missed_l8 = curve_gaps(pred_l8, house_l8)
    cgaps = [abs(c["energy_mean_distance"] - h["energy_mean_distance"])
             for c, h, m in zip(const_baseline, house_l8["curve"], pred_l8["curve"])
             if not m.get("target_missed")]
    rep["rn"] = {"prediction": pred_l8, "sampled_L8": house_l8, "abs_gaps": gaps,
                 "constant_gaps": cgaps, "n_rungs_missed": missed_l8,
                 "constant_baseline": const_baseline,
                 "adjudication": AF.adjudicate_rn(gap=max(gaps), model_gap=max(gaps),
                                                  constant_gap=max(cgaps))}
    print(f"[d0] R-N: {rep['rn']['adjudication']['verdict']} gap={max(gaps):.4f} "
          f"const={max(cgaps):.4f}", flush=True)
    _w()

    # ================= the corpus manifest =================
    corpus, excluded = [], []
    exp_sb_full = [(hi - lo) / d for lo, hi in sb]              # full-space expected fracs

    def sb_fracs_of_vec(v):
        return AF.subband_fractions_of_vector(v, basis.double(), sb)

    def sb_fracs_of_span(Q):
        """Removed-energy fraction per sub-band of ablating span(Q): tr(Qᵀ U_b Λ_b U_bᵀ Q),
        normalized over sub-bands."""
        Qd = torch.as_tensor(Q, dtype=torch.float64)
        parts = []
        for s_i in range(len(sb)):
            lo, hi = sb[s_i]
            lamb = spectrum.double()[lo:hi]
            A = basis.double()[:, lo:hi].transpose(0, 1) @ Qd          # (r_b, k)
            parts.append(float((lamb.unsqueeze(1) * A ** 2).sum()))
        tot = sum(parts)
        return [p / max(tot, 1e-300) for p in parts]

    def desc_base(**kw):
        dd = {k: 0.0 for k in AF.DESCRIPTOR_FIELDS}
        dd.update({"chan_none": 1.0, "retention": 1.0})
        dd.update(kw)
        return dd

    def add_row(cell_id, source, family, tag, sign, dppl, dvqa_effect, desc, fr, path, sha_):
        corpus.append(AF.manifest_entry(
            cell_id=cell_id, source=source, family=family, tag=tag, sign=sign,
            damage_ppl=math.log1p(max(dppl, -0.99)), damage_vqa=dvqa_effect,
            descriptor={**desc, "raw_dppl": dppl}, subband16=fr,
            artifact_path=path, artifact_sha256=sha_))

    T_EFF = 271.0
    # ---- bispec_k / sealed: P1/P1g/P3 (P2 excluded: its coordinate is outside every model) ----
    for art, sha_, path, tag in ((k_art, k_sha, AS.BISPEC_K, "DISCOVERY"),
                                 (ks_art, ks_sha, AS.BISPEC_K_SEALED, "CONFIRMATORY")):
        for c in art["cells"]:
            famly, cell = c["family"], c["cell"]
            cid = f"{path.split('/')[-3]}/{cell}/d{c['draw']}"
            if famly in ("anchor",):
                excluded.append({"cell": cid, "reason": "anchor row, not a perturbation cell"})
                continue
            if famly == "P2":
                excluded.append({"cell": cid, "reason": "P2's alignment coordinate is outside the "
                                                        "descriptor grammar (A9 axis)"})
                continue
            if famly in ("P1", "P1g"):
                lev = c.get("level")
                eps = lv[lev]["level"] if lev in ("benign", "mid") else None
                if eps is None:
                    excluded.append({"cell": cid, "reason": f"unmapped level {lev}"})
                    continue
                g = c.get("g")
                coh = 1.0 if g == -1 else (min(float(g), T_EFF) / T_EFF)
                if g == -1:
                    delta = coherent_delta(int(c["draw"]), lev_sigma[lev])
                    fr = sb_fracs_of_vec(delta)
                    v1p = float(max(float(delta @ v1), 0.0)) / (rms * math.sqrt(d))
                else:
                    fr = exp_sb_full
                    v1p = 0.0
                dd = desc_base(log_energy=math.log10(eps ** 2), dc_frac=coh,
                               shared_dir=1.0 if g == -1 else 0.0,
                               mean_shift=eps * math.sqrt(coh),
                               radial_frac=1.0 / d, v1_overlap_pos=v1p)
                add_row(cid, path.split("/")[-3], f"P1_{lev}", tag, "add", c["dppl"],
                        c.get("effect", 0.0), dd, fr, path, sha_)
            elif famly == "P3":
                r_ = int(c["rank"])
                if c["arm"] == "coherent":
                    from infra.pg.ctrliface_core import random_basis
                    Q = random_basis(d, r_, 777 + int(c["draw"]), "cpu", torch.float64)
                    fr = sb_fracs_of_span(Q)
                    E = BC.removed_energy_fraction(Q, Sigma)
                    dcf, shd = 1.0, 1.0
                else:
                    fr = [float((spectrum.double()[lo:hi]).sum() / spectrum.double().sum() * r_ / d)
                          for lo, hi in sb]
                    ss = sum(fr)
                    fr = [x / ss for x in fr]
                    E = r_ / d
                    dcf, shd = 0.0, 0.0
                dd = desc_base(is_delete=1.0, log_energy=math.log10(max(E, 1e-9)), dc_frac=dcf,
                               shared_dir=shd)
                add_row(cid, path.split("/")[-3], f"P3_r{r_}", tag, "delete", c["dppl"],
                        c.get("effect", 0.0), dd, fr, path, sha_)
    _w()

    # ---- bispec_s: band add/delete cells at amp 0.2 (0.4 rows enter, guard flags them) ----
    band_of = {f"band{b}": b for b in range(5)}
    for c in s_art["cells"]:
        rowname, cell_mode, sign = c.get("row"), c.get("mode"), c.get("sign")
        cid = f"bispec_s/{rowname}/{cell_mode}/{sign}/a{c.get('amp_over_rms')}/d{c.get('draw')}"
        if rowname not in band_of:
            excluded.append({"cell": cid, "reason": "named row (transport-bounded, V1r) — not a "
                                                    "band cell"})
            continue
        b = band_of[rowname]
        lo, hi = bands5[b]
        seg = spectrum.double()[lo:hi]
        segsum = float(seg.sum())
        fr = []
        for s_i in range(len(sb)):
            slo, shi = sb[s_i]
            inter = max(0, min(hi, shi) - max(lo, slo))
            if inter <= 0:
                fr.append(0.0)
            else:
                fr.append(max(0.0, min(1.0, inter / (hi - lo))))
        ss = sum(fr)
        fr = [x / ss for x in fr]
        if sign == "add":
            eps = float(c["amp_over_rms"])
            dd = desc_base(log_energy=math.log10(eps ** 2),
                           dc_frac=1.0 if cell_mode == "dc" else 0.0,
                           shared_dir=0.0 if cell_mode == "white" else 1.0,
                           mean_shift=eps if cell_mode == "dc" else 0.0)
            add_row(cid, "bispec_s", f"S_add_band{b}", "DISCOVERY", "add", c["dppl"],
                    c.get("effect", 0.0), dd, fr, AS.BISPEC_S, s_sha)
        else:
            E = c.get("removed_energy")
            if E is None:
                excluded.append({"cell": cid, "reason": "delete cell without banked removed_energy"})
                continue
            ms = 1.0 if cell_mode == "full" else float(c.get("envelope_ms", BC.DEL_MS_TARGET))
            dd = desc_base(is_delete=1.0, log_energy=math.log10(max(E * ms, 1e-9)),
                           dc_frac=1.0 if cell_mode in ("dc", "full") else 0.0, shared_dir=1.0)
            add_row(cid, "bispec_s", f"S_del_band{b}", "DISCOVERY", "delete", c["dppl"],
                    c.get("effect", 0.0), dd, fr, AS.BISPEC_S, s_sha)
    _w()

    # ---- bispec_a: dose cells (v₁ family, rotated, random, gain excluded) ----
    fr_v1 = sb_fracs_of_vec(v1)
    Qrot = RC.haar_rotation(d, RS.ROT_SEED, dtype=torch.float64)
    v_rot = Qrot @ v1
    fr_rot = sb_fracs_of_vec(v_rot)
    rand_frs, rand_vs = [], []
    for s_i in range(3):
        g = torch.Generator().manual_seed(4242 + s_i)
        r = torch.randn(d, generator=g, dtype=torch.float64)
        r = r / r.norm()
        rand_vs.append(r)
        rand_frs.append(sb_fracs_of_vec(r))
    for c in a_art["cells"]:
        if c["family"] != "A1_dose":
            excluded.append({"cell": f"bispec_a/{c['family']}/{c['direction']}/{c['alpha']}",
                             "reason": "gain-matched null is a rescale, not an additive cell"})
            continue
        dirn = c["direction"]
        alpha = float(c["alpha"])
        amp = abs(alpha) * sigma1 / (rms * math.sqrt(d))
        vmap = {"pc1": (v1, fr_v1), "pc1_orthogonalized": (v1, fr_v1), "pc1_rotated": (v_rot, fr_rot),
                "rand0": (rand_vs[0], rand_frs[0]), "rand1": (rand_vs[1], rand_frs[1]),
                "rand2": (rand_vs[2], rand_frs[2])}
        if dirn not in vmap:
            excluded.append({"cell": f"bispec_a/{dirn}", "reason": "unmapped direction"})
            continue
        vv, fr = vmap[dirn]
        ip = float(alpha * sigma1 * float(vv @ v1))
        dd = desc_base(log_energy=math.log10(max(amp ** 2, 1e-12)), dc_frac=1.0, shared_dir=1.0,
                       mean_shift=amp, radial_frac=float((xhat @ vv).pow(2).mean()),
                       v1_overlap_pos=max(ip, 0.0) / (rms * math.sqrt(d)))
        add_row(f"bispec_a/{dirn}/a{alpha}", "bispec_a", f"A_dose_{dirn}", "DISCOVERY", "add",
                c["dppl"], c.get("effect", 0.0), dd, fr, AS.BISPEC_A, a_sha)
    _w()

    # ---- frontier: gauss + quant rows (em_abl/rot_err constructions not in the grammar) ----
    for p in fr_art["points"]:
        cid = f"frontier/{p['family']}/{p['level']}"
        if p["family"] == "gauss":
            dd = desc_base(log_energy=math.log10(p["level"] ** 2), dc_frac=0.0, shared_dir=0.0)
            add_row(cid, "regrain_i_frontier", "frontier_gauss", "DISCOVERY", "add", p["dppl"],
                    -p["dvqa"], dd, exp_sb_full, AS.REGRAIN_FRONTIER, fr_sha)
        elif p["family"] == "quant":
            eps = p["level"] / math.sqrt(12.0)
            dd = desc_base(log_energy=math.log10(max(eps ** 2, 1e-12)), dc_frac=0.0, shared_dir=0.0)
            add_row(cid, "regrain_i_frontier", "frontier_quant", "DISCOVERY", "add", p["dppl"],
                    -p["dvqa"], dd, exp_sb_full, AS.REGRAIN_FRONTIER, fr_sha)
        else:
            excluded.append({"cell": cid, "reason": f"{p['family']} construction outside the "
                                                    f"descriptor grammar"})
    _w()

    rep["corpus_counts"] = {"included": len(corpus), "excluded": len(excluded),
                            "by_family": {}, "exclusion_reasons": {}}
    for r in corpus:
        rep["corpus_counts"]["by_family"][r["family"]] = \
            rep["corpus_counts"]["by_family"].get(r["family"], 0) + 1
    for e in excluded:
        key = e["reason"][:60]
        rep["corpus_counts"]["exclusion_reasons"][key] = \
            rep["corpus_counts"]["exclusion_reasons"].get(key, 0) + 1
    manifest = {"object": "atlas_corpus_v1", "rows": corpus, "excluded": excluded,
                "site": "L0", "n_ppl_seq": 120,
                "subband_slices": sb, "radial_mass_by_subband": radial_mass_by_sb}
    manifest["content_hash"] = AF.manifest_hash(corpus)
    (out / "atlas_corpus.json").write_text(json.dumps(manifest, indent=2, default=str))
    rep["corpus_hash"] = manifest["content_hash"]
    print("ATLAS_D0_OK", json.dumps({"rb1": rb1["verdict"], "rn": rep["rn"]["adjudication"]["verdict"],
                                     "corpus": len(corpus), "excluded": len(excluded)}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
