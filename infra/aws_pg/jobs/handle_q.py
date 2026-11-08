"""HANDLE Arm Q (GPU) — the benignity-mechanism race + the δ-dependence dose-response.

CARRY R9 showed the global operator does NOT predict which structured activation error is
behaviourally benign (ρ=0.238). Arm Q races FOUR named candidate properties — each a frozen
zero-forward computation on the error bank — to rank benignity, and adds the δ-mixture ladder as
the CAUSAL leg (the A9 error–token dependence axis, dose-responsive, with a rotated-twin control).

Families (≥12): CARRY's 8 (sae_error, quant, quant_signflip, quant_qshuffle, quant_perm,
gaussian_iid, rotated_error, quant_incoherent — reconstructed bit-for-bit from carry_e) + the 5
δ-mixture rungs (δ ∈ {0,.25,.5,.75,1}). Error fields built at L0 exactly as carry_e; damage =
log1p_dppl under the L0 injection hook (safe_ppl_gate). The 4 properties (H-A7, dated 2026-07-13):
  * dependence_fraction — mean_t cos²(e[t], the token's OWN quant residual). δ-ladder: ≈ δ.
  * feature_alignment (S3) — mean_t energy fraction of e[t] on the token's ACTIVE SAE feature
    directions (top-K decoder atoms by |code|).
  * massive_coord (S17) — mean_t energy fraction of e[t] on the L0 massive coords (carry_gate0).
  * incoherence (S18) — mean_t max_c|e[t,c]| / rms(e[t]) (spikiness; the QuaRot gauge axis).
Feeds handle_core.property_race + monotone_negative + adjudicate_rq. δ-ladder = Kendall τ(δ,damage),
required monotone-negative, rotated twin must break it.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.handle_q --type g6e.8xlarge,p5.4xlarge --no-weights \
      --s3-prefix results/handle_q --budget-min 300
"""
import argparse
import json
import os
import pathlib

HEADLINE = 0.8            # ×rms headline amplitude (carry_e convention — reproduce the 8)
N_PROP_ROWS = 512         # rows sampled for the zero-forward property scores
K_ACTIVE = 32             # top-K active SAE atoms for the feature-alignment property


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ppl-seq", type=int, default=60)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _handle_shared as HS
    from infra.pg import bispec_core as BC
    from infra.pg import carry_core as CC
    from infra.pg import grain_bsf as GB
    from infra.pg import handle_core as HC
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC
    from infra.pg.inherit_core import _split_hidden

    freeze = HS.assert_frozen("q")
    HS.install_pg()
    rep = {"object": "handle_q_v1", "freeze": freeze}
    tm = HS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = pathlib.Path("/tmp/handle_q")
    wd.mkdir(parents=True, exist_ok=True)
    g0 = json.loads(HS.fetch(f"{HS.HANDLE_GATE0}/gate0.json", wd / "g0.json").read_text())
    assert g0.get("green"), "handle gate0 not green"
    carry_g0 = json.loads(HS.fetch(f"{HS.CARRY_GATE0}/gate0.json", wd / "carry_g0.json").read_text())
    fts = HS.load_carry_fts(str(wd))
    D = IP.D
    Hd = CC.hadamard_matrix(D).float()
    # massive coords at L0 (recomputed from the carry_gate0 census, never prose)
    mass_coords = [int(c) for c in carry_g0["gamma_sink_readout"]["L0"]["coords"]]
    rep["massive_coords"] = mass_coords

    with tm.stage("load"):
        model, proc = IP.load_pg1(sha=fts["base_sha"])
        layers = IP.decoder_layers(model)
        dev = "cuda"
        X = torch.load(HS.fetch(f"{HS.S3_BUCKET}/results/regrain_pipeA/out/harvest/shard_L0_0.pt",
                                wd / "shard_L0.pt"),
                       map_location="cpu", weights_only=False)["X"][:8192].float().cuda()
        rms = float(X.pow(2).mean().sqrt())
        step = HEADLINE * rms
        Xr = X @ Hd.cuda(); rms_r = float(Xr.pow(2).mean().sqrt()); step_r = HEADLINE * rms_r
        ae, _cfg, _q, _sha = GB.load_dict_bank(
            HS.fetch(f"{HS.GRAIN_DICT_L0}", wd / "dict_m1.pt"))
        ae = ae.cuda()
        Rrot = RC.haar_rotation(D, 4242, device="cuda", dtype=torch.float32)
        pp = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
        rep["base_ppl"] = pp["ppl"]
        rep["site"] = {"rms": rms, "step": step, "step_r": step_r}
    _w()

    # ---- family constructions (reconstructed to match carry_e bit-for-bit) --------------
    def e_sae(x):
        return x - ae.decode(ae.encode(x.float(), use_threshold=True))

    def _qshuffle(x, st, seed):
        e = BC.quant_error(x.float(), st); T = x.shape[0]
        if T <= 1:
            return e
        stat = x.float().mean(-1)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        q = torch.quantile(stat.double().cpu(), torch.linspace(0, 1, 11, dtype=torch.float64))
        binid = torch.bucketize(stat.double().cpu(), q[1:-1]); perm = torch.arange(T)
        for b in range(10):
            idx = torch.nonzero(binid == b).flatten()
            if idx.numel() > 1:
                perm[idx] = idx[torch.randperm(idx.numel(), generator=gen)]
        return e[perm.to(e.device)]

    def build_field(name, x, mult=1.0, seed=0):
        x = x.float()
        if name == "sae_error":
            return -e_sae(x)
        if name == "quant":
            return BC.quant_error(x, mult * step)
        if name == "quant_signflip":
            return -BC.quant_error(x, mult * step)
        if name == "quant_qshuffle":
            return _qshuffle(x, mult * step, 40 + seed)
        if name == "quant_perm":
            e = BC.quant_error(x, mult * step); T = x.shape[0]
            if T > 1:
                gen = torch.Generator(device="cpu").manual_seed(5 + seed)
                e = e[torch.randperm(T, generator=gen).to(e.device)]
            return e
        if name == "gaussian_iid":
            gen = torch.Generator(device="cpu").manual_seed(11 + 1000 * seed)
            return (mult * HEADLINE * rms) * torch.randn(x.shape, generator=gen).to(x.device)
        if name == "rotated_error":
            return BC.quant_error(x, mult * step) @ Rrot.T
        if name == "quant_incoherent":
            xr = x @ Hd.to(x.device)
            return BC.quant_error(xr, mult * step_r) @ Hd.to(x.device).T
        raise AssertionError(f"unknown family {name}")

    CARRY_FAMILIES = ["sae_error", "quant", "quant_signflip", "quant_qshuffle", "quant_perm",
                      "gaussian_iid", "rotated_error", "quant_incoherent"]

    def delta_field(x, delta, twin=False):
        """The δ-mixture error at L0: δ·(token's own quant residual) + (1−δ)·(matched off-token
        residual), per-token energy-matched (HC.delta_mix). twin=True Haar-rotates the off-token
        component (destroys direction, keeps norm)."""
        x = x.float(); T = x.shape[0]
        r_own = BC.quant_error(x, step)
        gen = torch.Generator(device="cpu").manual_seed(777)
        r_off = r_own[torch.randperm(T, generator=gen).to(r_own.device)] if T > 1 else r_own
        if twin:
            r_off = HC.rotate_offtoken(r_off.double(), 4242).float().to(x.device)
        return HC.delta_mix(r_own.double(), r_off.double(), delta).float().to(x.device)

    def make_hook(field_fn):
        def hook(_m, args, kwargs):
            hs, rebuild = _split_hidden(args, kwargs)
            fld = field_fn(hs[0])
            return rebuild((hs.float() + fld.unsqueeze(0)).to(hs.dtype))
        return hook

    # ---- the 4 zero-forward properties on the error bank --------------------------------
    def _props(field):
        """Mean over N_PROP_ROWS sampled tokens of the 4 property scalars for a family whose
        per-token error field on X is `field` (8192,D)."""
        idx = torch.randperm(X.shape[0])[:N_PROP_ROWS].tolist()
        r_own = BC.quant_error(X, step)
        dep, feat, mass, incoh = [], [], [], []
        codes = ae.encode(X.float(), use_threshold=True)          # (8192, n_atoms)
        Wd = ae.decoder.weight if hasattr(ae, "decoder") else ae.W_dec.T  # (D, n_atoms)
        for i in idx:
            e = field[i].double()
            n2 = float((e ** 2).sum())
            if n2 <= 0:
                continue
            ro = r_own[i].double()
            if float((ro ** 2).sum()) > 0:
                dep.append(float((e @ ro) ** 2 / (n2 * float((ro ** 2).sum()))))
            mass.append(sum(float(e[c] ** 2) for c in mass_coords) / n2)
            incoh.append(float(e.abs().max()) / (n2 / D) ** 0.5)
            # feature alignment: energy fraction on the token's top-K active decoder atoms
            c = codes[i].abs()
            top = torch.topk(c, min(K_ACTIVE, c.numel())).indices
            Da = torch.as_tensor(Wd[:, top], dtype=torch.float64)
            Qa, _ = torch.linalg.qr(Da)
            feat.append(float(((Qa.T @ e) ** 2).sum() / n2))
        return {"dependence_fraction": float(np.mean(dep)), "feature_alignment": float(np.mean(feat)),
                "massive_coord": float(np.mean(mass)), "incoherence": float(np.mean(incoh))}

    # ---- damage + properties per family -------------------------------------------------
    base_ppl = rep["base_ppl"]
    fam_damage, fam_props = {}, {}
    with tm.stage("carry_families"):
        for name in CARRY_FAMILIES:
            with BC.hook_at(layers, 0, make_hook(lambda x, n=name: build_field(n, x))):
                pg = HS.safe_ppl_gate(model, proc, dev, n_seq=a.n_ppl_seq)
            fam_damage[name] = HS.safe_log1p_dppl(pg["mean_nll"], base_ppl)
            fam_props[name] = _props(build_field(name, X))
            print(f"[handle-q] {name}: dppl={fam_damage[name]:+.3f} "
                  f"dep={fam_props[name]['dependence_fraction']:.3f}", flush=True)
            rep["fam_damage"], rep["fam_props"] = fam_damage, fam_props
            _w()

    with tm.stage("delta_ladder"):
        delta_damage, twin_damage = {}, {}
        for d in HC.RQ_DELTA_GRID:
            fname = f"delta@{d}"
            with BC.hook_at(layers, 0, make_hook(lambda x, dd=d: delta_field(x, dd))):
                pg = HS.safe_ppl_gate(model, proc, dev, n_seq=a.n_ppl_seq)
            fam_damage[fname] = HS.safe_log1p_dppl(pg["mean_nll"], base_ppl)
            delta_damage[d] = fam_damage[fname]
            fam_props[fname] = _props(delta_field(X, d))
            with BC.hook_at(layers, 0, make_hook(lambda x, dd=d: delta_field(x, dd, twin=True))):
                pgt = HS.safe_ppl_gate(model, proc, dev, n_seq=a.n_ppl_seq)
            twin_damage[d] = HS.safe_log1p_dppl(pgt["mean_nll"], base_ppl)
            print(f"[handle-q] delta@{d}: dppl={delta_damage[d]:+.3f} twin={twin_damage[d]:+.3f}",
                  flush=True)
            rep["fam_damage"], rep["fam_props"] = fam_damage, fam_props
            rep["delta_damage"], rep["twin_damage"] = delta_damage, twin_damage
            _w()

    # ---- race + causal ladder + verdict -------------------------------------------------
    with tm.stage("adjudicate"):
        families = list(fam_damage)
        damages = [fam_damage[f] for f in families]
        props = ["dependence_fraction", "feature_alignment", "massive_coord", "incoherence"]
        scores_by_prop = {p: [fam_props[f][p] for f in families] for p in props}
        race = HC.property_race(scores_by_prop, damages, families)
        rep["race"] = race
        dgrid = list(HC.RQ_DELTA_GRID)
        ladder = HC.monotone_negative(dgrid, [delta_damage[d] for d in dgrid])
        twin = HC.monotone_negative(dgrid, [twin_damage[d] for d in dgrid])
        pos = [d for d in damages if d > 0]
        span = (max(pos) / min(pos)) if len(pos) >= 2 and min(pos) > 0 else 0.0
        rep["rq"] = HC.adjudicate_rq(race=race, delta_ladder=ladder, delta_ladder_twin=twin,
                                     damage_span=span)
        rep["delta_ladder"], rep["delta_twin"], rep["damage_span"] = ladder, twin, span
    _w()

    HS.assert_gpu_seconds(rep)
    print("HANDLE_Q_OK", json.dumps({"verdict": rep["rq"]["verdict"],
                                     "delta_tau": rep["rq"]["delta_tau"]}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
