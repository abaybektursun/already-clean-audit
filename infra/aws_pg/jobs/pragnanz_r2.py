"""PRAGNANZ Wing-R completion — (A) the crosscoder-family corroboration via the house BSF
featurizer (PZ-FR pivot, journaled: TESSERA banked no dictionaries) and (B) the Q2 restore.

A: train BlockTopK SAE (m=1) on the banked PG1 L9 discovery harvest; rungs on the SAME 800
   confirmation items as the ladder: `rt` (encode-decode roundtrip — the r1-analogue) vs
   `shuf` (per-FEATURE value permutation across the forward's tokens — marginals exact,
   co-occurrence destroyed — the r2-analogue). Sign of Δ(shuf−rt) vs the splat Δ(r2−r1).
   P-live: identity perms ≡ roundtrip bit-level.
B: π0-reshell; transport hook at L9 moving each token by μ_PG1[σ(c)] − μ_π0[c] under the
   INVERTED GESTALT site-9 correspondence (DISCLOSED void-banded — the only band-valid site has
   no behavioral range); arms {none, transport, random-transport ×2}; forced-choice VQA + gray
   (Δ_perception); registered NULL; any revival → P7 (never reported from this run alone).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.pragnanz_r2 --type g6e.2xlarge,g6e.4xlarge \
      --s3-prefix results/pragnanz_r2 --budget-min 360

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import time

SITE = 9
N_A = 800
N_B = 400
CHOICE_SEED = 20260728
SHUF_SEEDS = (0, 1)


def _pull(path, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "cp", f"s3://pg-lineage-769063704696/{path}", str(dest),
                    "--region", "us-east-1", "--only-show-errors"], check=True)
    return dest


class FeatHook:
    """mode 'rt' | 'shuf': h -> decode(encode(h)) with optional per-feature value permutation
    across the forward's flattened tokens. Persistent perm generator; identity_perm for P-live."""

    def __init__(self, layer, ae, mode, seed=0):
        import torch
        self.layer, self.ae, self.mode = layer, ae, mode
        self.g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.identity_perm = False
        self._handle = None

    def __enter__(self):
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        return False

    def _hook(self, module, args, output):
        import torch
        hs = output[0] if isinstance(output, tuple) else output
        flat = hs.reshape(-1, hs.shape[-1]).float()
        f = self.ae.encode(flat)
        if self.mode == "shuf":
            n = f.shape[0]
            if self.identity_perm:
                _ = torch.randperm(n, generator=self.g)          # burn identically
            else:
                # ONE permutation applied per-feature via independent column rolls would need
                # F perms; frozen realization: independent per-feature permutations, drawn from
                # the persistent generator (marginals exact per feature; co-occurrence dead)
                idx = torch.stack([torch.randperm(n, generator=self.g)
                                   for _ in range(f.shape[1])], dim=1).to(f.device)
                f = torch.gather(f, 0, idx)
        elif self.mode == "jointperm":
            # post-critic control (critic #5): ONE shared permutation over the flattened pool —
            # every token keeps its full feature vector (co-occurrence intact); only
            # position/candidate-row binding is destroyed. Separates the shuf arm's
            # co-occurrence cost from its cross-row mixing cost.
            n = f.shape[0]
            if self.identity_perm:
                _ = torch.randperm(n, generator=self.g)
            else:
                f = f[torch.randperm(n, generator=self.g).to(f.device)]
        new = self.ae.decode(f).to(hs.dtype).reshape(hs.shape)
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new


class TransportHook:
    """x' = x + (mu_dst[sigma(c)] - mu_src[c]); 'random' arm re-directions the shift per-token
    at equal norm (pragnanz_core.random_transport_shift)."""

    def __init__(self, layer, atlas_src, atlas_dst, sigma, mode="real", seed=0):
        self.layer, self.src, self.dst, self.sigma = layer, atlas_src, atlas_dst, sigma
        self.mode, self.seed = mode, seed
        self._handle = None

    def __enter__(self):
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        return False

    def _hook(self, module, args, output):
        from infra.pg import pragnanz_core as PC
        from infra.pg.gestalt_splat import _log_resp
        hs = output[0] if isinstance(output, tuple) else output
        flat = hs.reshape(-1, hs.shape[-1]).float()
        lg, _ = _log_resp(flat, self.src)
        assign = lg.argmax(1)
        shift = PC.transport_shift(self.src, self.dst, assign, self.sigma)
        if self.mode == "random":
            shift = PC.random_transport_shift(shift, seed=self.seed)
        new = (flat + shift).to(hs.dtype).reshape(hs.shape)
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new


def main():
    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="24 items, 1 shuf seed, short SAE")
    ap.add_argument("--followup", action="store_true",
                    help="post-critic pass: P3 shuffler plant test + jointperm arm only")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs.plenum_fc import PPL_ANCHOR, PPL_REL_TOL
    from infra.pg import gestalt_splat as GS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_core as LC
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_core as PC_fc
    from infra.pg import plenum_units as PU
    from infra.pg.plenum_core import score_item_fc

    assert torch.cuda.is_available()
    device = "cuda"
    n_a = 24 if a.smoke else N_A
    n_b = 24 if a.smoke else N_B
    shuf_seeds = (0,) if a.smoke else SHUF_SEEDS
    rep = {"object": "PRAGNANZ Wing-R completion (corroboration + restore)", "site": SITE,
           "n_a": n_a, "n_b": n_b, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0)}
    cells = []
    rep["cells"] = cells
    banks = {}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
        (out / "r2_banks.json").write_text(json.dumps(banks, default=str))
    _w()

    # ---- items (confirmation half; SAME frozen construction as the ladder) ------------------
    tasks = LS.mm_tasks()
    vqa = tasks["vqav2"]
    rows, hashes = IP.load_items(vqa, 4000)
    disc, _ = LC.parity_split(hashes)
    c_rows = [r for r, m in zip(rows, disc) if not m]
    c_hashes = [h for h, m in zip(hashes, disc) if not m]
    rows_a, rows_b = c_rows[:n_a], c_rows[:n_b]
    choices_a = PC_fc.choices_for_task("vqav2", rows_a, CHOICE_SEED)
    choices_b = choices_a[:n_b] if n_b <= n_a else PC_fc.choices_for_task("vqav2", rows_b,
                                                                          CHOICE_SEED)
    rep["items_sha"] = hashlib.sha256("".join(c_hashes[:max(n_a, n_b)]).encode()).hexdigest()[:16]
    _w()

    # ---- SAE training on the banked PG1 L9 discovery harvest --------------------------------
    sdir = pathlib.Path("/tmp/pz_h")
    sdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync",
                    "s3://pg-lineage-769063704696/results/gestalt_arm_s8/out/harvest_pg1/site9",
                    str(sdir), "--region", "us-east-1", "--only-show-errors",
                    "--exclude", "item_idx*"], check=True)
    shards = sorted(sdir.glob("acts_*.npy"))
    X = torch.cat([torch.from_numpy(np.load(p)).float() for p in shards], 0)
    steps = 800 if a.smoke else 8000
    ae, cfg = GB.train_block_sae(GS.fit_subsample(X, seed=0, n_fit=400_000), m=1, seed=0,
                                 device=device, steps=steps)
    q = GB.block_quality(ae, X, device)
    ae = ae.to(device)                      # trainer returns CPU eval mode; hooks feed CUDA
    rep["sae"] = {"cfg": cfg, "quality": q}
    del X
    torch.cuda.empty_cache()
    print(f"[R2] SAE trained: {json.dumps(q, default=str)[:200]}", flush=True)
    _w()

    # ---- shell + anchor + hooks -------------------------------------------------------------
    model, proc = LS.load_pg1(device=device)
    pg1_sd = LS.pg1_baseline_sd(model)
    layers = IP.decoder_layers(model)
    ppl = IP.ppl_gate(model, proc, device)
    val = float(ppl["ppl"] if isinstance(ppl, dict) else ppl)
    assert abs(val - PPL_ANCHOR) / PPL_ANCHOR < PPL_REL_TOL
    site_layer = layers[SITE]

    def run_eval(tag, ctx, eval_rows, eval_choices):
        import contextlib
        t1 = time.time()
        per_item = []
        with (ctx if ctx is not None else contextlib.nullcontext()):
            for r, (cands, gold) in zip(eval_rows, eval_choices):
                img = vqa.image_fn(r).convert("RGB")
                s, nt = score_item_fc(model, proc, device, img,
                                      PC_fc.fc_prompt(vqa.prompt_fn(r)), cands)
                if s is None:
                    per_item.append(float("nan"))
                    continue
                mlp = np.asarray(s) / np.asarray(nt, dtype=float)
                per_item.append(int(int(mlp.argmax()) == gold))
        acc = float(np.nanmean(per_item))
        cells.append({"key": tag, "acc": acc, "n": len(per_item),
                      "eval_s": round(time.time() - t1, 1)})
        banks[tag] = {"per_item": per_item}
        _w()
        print(f"[R2] {tag:>26s} acc={acc:.4f} ({time.time() - t1:.0f}s)", flush=True)

    # ---- A: corroboration (pg1) -------------------------------------------------------------
    # P-live: identity-perm shuf == rt bit-level on one forward
    img0 = vqa.image_fn(rows_a[0]).convert("RGB")
    inp = proc(text=[PC_fc.fc_prompt(vqa.prompt_fn(rows_a[0]))], images=[img0],
               return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
    cap = {}
    for tag, mode, ident in (("rt", "rt", False), ("shufid", "shuf", True)):
        h = FeatHook(site_layer, ae, mode, seed=55)
        h.identity_perm = ident
        ch = layers[SITE + 1].register_forward_pre_hook(
            lambda m, args, _t=tag: cap.__setitem__(_t, args[0].detach().clone()))
        with h, torch.inference_mode():
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"])
        ch.remove()
    assert torch.equal(cap["rt"], cap["shufid"]), "P-live FAIL: identity shuf != rt"
    rep["p_live_corr"] = "bit-exact"
    _w()

    if a.followup:
        # ---- P3 (PZ-D4 remediation): the shuffle-OPERATOR plant test, fail-loud, N1-printed --
        gsyn = torch.Generator().manual_seed(7)
        n_t, n_f = 5000, 64
        gate_v = (torch.rand(n_t, generator=gsyn) < 0.3).float()
        f_syn = torch.rand(n_t, n_f, generator=gsyn) * 0.1
        f_syn[:, 0] = gate_v * (0.5 + torch.rand(n_t, generator=gsyn))   # planted co-active pair
        f_syn[:, 1] = gate_v * (0.5 + torch.rand(n_t, generator=gsyn))

        def phi(u, v):
            u, v = (u > 0.25).float(), (v > 0.25).float()
            return float(((u - u.mean()) * (v - v.mean())).mean()
                         / (u.std(unbiased=False) * v.std(unbiased=False) + 1e-12))

        phi0 = phi(f_syn[:, 0], f_syn[:, 1])
        gp = torch.Generator().manual_seed(11)
        idx = torch.stack([torch.randperm(n_t, generator=gp) for _ in range(n_f)], dim=1)
        f_sh = torch.gather(f_syn, 0, idx)
        marg_ok = all(torch.equal(f_sh[:, j].sort().values, f_syn[:, j].sort().values)
                      for j in range(n_f))
        phi_sh = phi(f_sh[:, 0], f_sh[:, 1])
        f_jp = f_syn[torch.randperm(n_t, generator=gp)]
        phi_jp = phi(f_jp[:, 0], f_jp[:, 1])
        print(f"[R2-FU] P3: phi0={phi0:.4f} shuf={phi_sh:.4f} joint={phi_jp:.4f} "
              f"marginals_exact={marg_ok}", flush=True)
        assert phi0 > 0.9, f"P3 plant too weak ({phi0})"
        assert marg_ok, "P3 FAIL: shuffle does not preserve per-feature marginals"
        assert abs(phi_sh) < 0.05, f"P3 FAIL: plant survives the shuffle ({phi_sh})"
        assert abs(phi_jp - phi0) < 1e-6, f"P3 FAIL: joint perm altered co-occurrence ({phi_jp})"
        rep["p3"] = {"phi_plant": phi0, "phi_shuf": phi_sh, "phi_joint": phi_jp,
                     "marginals_exact": marg_ok}

        # P-live for the jointperm path
        cap2 = {}
        for tag, ident in (("rt", None), ("jpid", True)):
            h = FeatHook(site_layer, ae, "rt" if ident is None else "jointperm", seed=66)
            if ident:
                h.identity_perm = True
            ch = layers[SITE + 1].register_forward_pre_hook(
                lambda m, args, _t=tag: cap2.__setitem__(_t, args[0].detach().clone()))
            with h, torch.inference_mode():
                model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                      pixel_values=inp["pixel_values"])
            ch.remove()
        assert torch.equal(cap2["rt"], cap2["jpid"]), "P-live FAIL: identity jointperm != rt"
        rep["p_live_jointperm"] = "bit-exact"
        _w()

        run_eval("pg1|rt", FeatHook(site_layer, ae, "rt", seed=100), rows_a, choices_a)
        for s in shuf_seeds:
            run_eval(f"pg1|jointperm|s{s}", FeatHook(site_layer, ae, "jointperm", seed=200 + s),
                     rows_a, choices_a)
        rep["elapsed_s"] = time.time() - t0
        _w()
        print("PRAGNANZ_R2_FOLLOWUP_OK",
              json.dumps({c["key"]: c["acc"] for c in cells}), flush=True)
        return

    run_eval("pg1|rt", FeatHook(site_layer, ae, "rt", seed=100), rows_a, choices_a)
    for s in shuf_seeds:
        run_eval(f"pg1|shuf|s{s}", FeatHook(site_layer, ae, "shuf", seed=100 + s),
                 rows_a, choices_a)

    # ---- B: restore (pi0-reshell) -----------------------------------------------------------
    # PZ-D3: the GESTALT diff banked labels/costs but NOT the assignment columns — the
    # correspondence is RECOMPUTED here from the two banked atlases under the frozen FR-6 rule
    # (cost matrix -> raw-Hungarian for chance exclusion -> tau_chance -> CAPPED assignment);
    # deterministic, artifact-derived.
    from infra.pg import gestalt_corr as GC

    def _atl(unit, tag):
        raw = np.load(_pull(f"results/gestalt_arm_d/out/atlas_{unit}/site{SITE}/{tag}_s0.npz",
                            pathlib.Path(f"/tmp/pz_{unit}_{tag}.npz")))
        return {k: torch.from_numpy(raw[k])
                for k in ("log_pi", "mu", "U", "lam", "sigma2")} | {"void": False}

    a_pi0_cpu, a_pg1_cpu = _atl("pi0", "full"), _atl("pg1", "full")
    band = GC.selfhalf_band(GC.hungarian(GC.cost_matrix(
        GC.atlas_np(_atl("pg1", "halfA")), GC.atlas_np(_atl("pg1", "halfB"))))[2])
    assert not band["zero_width"], "P4 zero-width band at site 9"
    Cx = GC.cost_matrix(GC.atlas_np(a_pg1_cpu), GC.atlas_np(a_pi0_cpu))
    _, ci_x, _ = GC.hungarian(Cx)
    tau_chance = float(np.percentile(GC.shuffle_chance_costs(Cx, assign_cols=ci_x),
                                     GC.TAU_CHANCE_PCTL))
    lab = GC.label_pair(Cx, band["tau_move"], tau_chance)
    sigma = [0] * len(lab["assign_rows"])
    for i_pg1, j_pi0 in zip(lab["assign_rows"], lab["assign_cols"]):
        sigma[j_pi0] = i_pg1                          # invert: pi0 comp -> pg1 comp
    a_pi0 = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in a_pi0_cpu.items()}
    a_pg1 = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in a_pg1_cpu.items()}
    rep["restore"] = {"void_banded_map": bool(lab["void_banded"]),
                      "tau_move": band["tau_move"], "tau_chance": tau_chance,
                      "frac_survived": lab["frac_survived"], "sigma": sigma,
                      "note": "PZ-D3: correspondence recomputed in-job from banked atlases "
                              "(frozen FR-6 sequence, gestalt_diff.py:88-128); the site-9 map "
                              "carries GESTALT's disclosure — the registered restore prediction "
                              "is NULL; a revival hits P7"}
    _w()

    PU.install_unit("pi0", model, proc, pg1_sd, device,
                    {"stitch": 0, "matched_dw": 0, "rand_dec": 0})
    run_eval("pi0|none", None, rows_b, choices_b)
    run_eval("pi0|transport",
             TransportHook(site_layer, a_pi0, a_pg1, sigma, mode="real"), rows_b, choices_b)
    for s in shuf_seeds:
        run_eval(f"pi0|randtrans|s{s}",
                 TransportHook(site_layer, a_pi0, a_pg1, sigma, mode="random", seed=300 + s),
                 rows_b, choices_b)

    rep["elapsed_s"] = time.time() - t0
    _w()
    print("PRAGNANZ_R2_OK", json.dumps({c["key"]: c["acc"] for c in cells}), flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)     # GESTALT D7
