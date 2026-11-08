"""GESTALT Arm F — the flow-field program: probe bank, P5, film field banks (U3/U3c x 8 steps),
F1 diff stats (FR-3 nulls), F2 surrogate ladder + FR-7 behavioral gate, pi0.5 census.
Spec §1.4/§2 Arm F; instrument core = infra/pg/gestalt_field.py (offline-gated).

Order (incremental metrics.json throughout; each leg gated):
  0. lerobot install (house git-extra pattern), version + API asserts banked.
  1. Probe rows: HuggingFaceVLA/libero DISCOVERY suites (ctrliface split), N_OBS frames with
     action chunks; manifest sha'd. The SAME raw rows feed every policy (each consumes them
     through its OWN preprocessor — the field belongs to the policy, preprocessing included).
  2. pi0 (pi0_base re-shimmed to the LIBERO feature layout via lacuna_units.build_published —
     weights untouched): **P5** (own integrator == model.sample_actions, same noise, exact);
     field bank at T_GRID x N_NOISE; behavioral references (10-step + 5-step integrations).
  3. F2: ridge ladder fit on fit-half obs, proxy MSE on held half, FR-7 behavioral gate on held
     obs; Borges ratio vs the counted expert params.
  4. Films: {U3, U3c} x steps 512..4096 — field bank per checkpoint (P5-lite on the first of
     each unit); preprocessor sha equality asserted for the F1 pair (else F1 VOID, disclosed).
  5. F1: D_rel(U3c@s, U3@s) + probe-bootstrap CI vs the FR-3 nulls (split-half width via CI;
     adjacent-step drift bands).
  6. Census: pi05_base (re-shimmed) field summary stats (its own prefix adapter; no state arg).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.gestalt_field_job --type p5.4xlarge,g6e.8xlarge \
      --s3-prefix results/gestalt_field --budget-min 420

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

S3_FILM = "s3://pg-lineage-769063704696/weights/plenum_film"
FILM_STEPS = (512, 1024, 1536, 2048, 2560, 3072, 3584, 4096)
FILM_UNITS = ("U3", "U3c")
P5_ATOL = 1e-3          # fp32 action-space; bf16 backbone kernels differ at ~1e-5 rms


def _sh(cmd):
    print("[sh]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _pull_ckpt(uri, dest):
    d = pathlib.Path(dest)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    _sh(["aws", "s3", "sync", uri, str(d), "--region", "us-east-1", "--only-show-errors"])
    assert (d / "model.safetensors").exists(), f"no model.safetensors under {uri}"
    return d


def _file_sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def probe_rows(n_obs, seed):
    """N_OBS frames (with action chunks) from the first 48 LIBERO episodes, deterministic.

    NOT suite-scoped: HuggingFaceVLA/libero has no per-episode suite tag (the suite lives in the
    ENV — `reference_libero_no_suite_tag`, which bit CTRLIFACE and LACUNA), so a suite firewall
    is inexpressible here. Arm F makes no held-episode behavioral claim; its only split is F2's
    within-probe fit/held halves. The manifest is content-hashed (spec Arm F)."""
    import torch

    from infra.pg import ctrliface_lerobot as CL
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    cfg = PI0Config.from_pretrained("lerobot/pi0_base")
    meta = CL.dataset_meta()
    dt = {"action": [i / meta.fps for i in range(cfg.chunk_size)]}
    loader, n_ep, _filtered = CL.build_loader(CL.DATASET_REPO, dt, suites=None,
                                              batch_size=1, shuffle=True, seed=seed,
                                              num_workers=2, max_episodes=48)
    rows = []
    for b in loader:
        rows.append({"episode_index": int(b["episode_index"][0]),
                     "frame_index": int(b["frame_index"][0]),
                     "task": b["task"][0],
                     "state": b["observation.state"][0].clone(),
                     "action": b["action"][0].clone(),
                     "image": b["observation.images.image"][0].clone(),
                     "image2": b["observation.images.image2"][0].clone()})
        if len(rows) >= n_obs:
            break
    assert len(rows) == n_obs, f"only {len(rows)} probe rows (< {n_obs})"
    return rows, n_ep


def raw_batch(rows_slice, n_repeat):
    """A raw lerobot-style batch of ONE obs repeated n_repeat times (the noise-batch trick)."""
    import torch
    r = rows_slice
    return {"observation.state": r["state"].unsqueeze(0).repeat(n_repeat, 1),
            "observation.images.image": r["image"].unsqueeze(0).repeat(n_repeat, 1, 1, 1),
            "observation.images.image2": r["image2"].unsqueeze(0).repeat(n_repeat, 1, 1, 1),
            "action": r["action"].unsqueeze(0).repeat(n_repeat, 1, 1),
            "task": [r["task"]] * n_repeat}


def to_dev(batch, device):
    import torch
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def build_unit_dir(kind, workdir):
    """pi0/pi05 published weights re-shimmed to the LIBERO feature layout (weights untouched)."""
    from infra.pg import lacuna_units as LU
    d = pathlib.Path(workdir)
    if (d / "lacuna_unit_manifest.json").exists():
        return d
    repo = {"pi0": "lerobot/pi0_base", "pi05": "lerobot/pi05_base"}[kind]
    LU.build_published(repo, d)
    return d


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-obs", type=int, default=0, help="0 = frozen N_OBS")
    ap.add_argument("--units", default=",".join(FILM_UNITS))
    ap.add_argument("--steps", default=",".join(str(s) for s in FILM_STEPS))
    ap.add_argument("--skip-census", action="store_true")
    ap.add_argument("--census-only", action="store_true",
                    help="run ONLY probes + the pi05 census (banks/F1/F2 already on S3)")
    ap.add_argument("--smoke", action="store_true", help="4 obs, 2 steps, U3 only")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # PINNED to the film-TRAINING-era git sha (2026-07-29; the 0.5.1 wheel's dep pins conflict
    # with the AMI stack — D9). API surface at this sha verified: denoise_step(state, ...),
    # euler_integrate == our integrate convention, module-level prepare_attention_masks_4d.
    # `av` explicit: newer builds gate it behind the `dataset` extra, older need it directly.
    _sh([sys.executable, "-m", "pip", "install", "-q",
         "lerobot[pi,dataset] @ git+https://github.com/huggingface/lerobot.git@35339d31e53d",
         "av"])
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import lerobot
    import numpy as np
    import torch

    from infra.pg import gestalt_field as GF

    assert torch.cuda.is_available(), "gestalt_field_job needs CUDA"
    device = "cuda"
    n_obs = a.n_obs or (4 if a.smoke else GF.N_OBS)
    units = ["U3"] if a.smoke else [u for u in a.units.split(",") if u]
    steps = [512, 1024] if a.smoke else [int(s) for s in a.steps.split(",") if s]
    rep = {"object": "GESTALT Arm F", "n_obs": n_obs, "units": units, "steps": steps,
           "t_grid": list(GF.T_GRID), "n_noise": GF.N_NOISE,
           "lerobot": getattr(lerobot, "__version__", "git"),
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    cells = []
    rep["cells"] = cells

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    # ---- 1. probe rows ----------------------------------------------------------------------
    rows, n_ep = probe_rows(n_obs, GF.PROBE_SEED)
    rep["probe_manifest"] = {"n_obs": n_obs, "n_episodes_pool": n_ep,
                            "sha": GF.manifest_sha(rows)}
    print(f"[F] probes: {n_obs} obs from {n_ep} discovery episodes sha={rep['probe_manifest']['sha']}",
          flush=True)
    _w()

    # ---- field banking for one pi0-family policy --------------------------------------------
    def bank_policy(tag, ckpt_dir, do_p5, do_behavior):
        from infra.pg.gestalt_field import (field_eval, integrate, prefix_cache,
                                            probe_noise, probe_points)
        policy, pre = GF.load_pi0_policy(str(ckpt_dir), device=device)
        cfg = policy.config
        chunk, adim = cfg.chunk_size, cfg.max_action_dim
        nsteps = cfg.num_inference_steps
        z_all = probe_noise(n_obs, GF.N_NOISE, chunk, adim)
        v_bank = np.zeros((n_obs, GF.N_NOISE, len(GF.T_GRID), chunk, adim), dtype=np.float16)
        z_obs_bank = np.zeros((n_obs, 2048), dtype=np.float32)
        aref_bank = np.zeros((n_obs, chunk, adim), dtype=np.float32)
        act10 = np.zeros((n_obs, GF.N_NOISE, chunk, adim), dtype=np.float32)
        act5 = np.zeros_like(act10)
        pproc_sha = _file_sha(pathlib.Path(ckpt_dir) / "policy_preprocessor.json")
        for i, row in enumerate(rows):
            # to_dev AFTER pre as well: a processor's tokenizer step can emit fresh CPU tensors
            batch = to_dev(pre(to_dev(raw_batch(row, GF.N_NOISE), device)), device)
            pfx, kv, state, z_obs = prefix_cache(policy, batch)
            a_ref = policy.prepare_action(batch)[0]                     # (chunk, adim) normalized
            aref_bank[i] = a_ref.float().cpu().numpy()
            z_obs_bank[i] = z_obs[0].cpu().numpy()[:2048]
            z_i = z_all[i].to(device)                                   # (N_NOISE, chunk, adim)
            if do_p5 and i == 0:
                ref = policy.model.sample_actions(
                    *_prefix_args(policy, batch), noise=z_i, num_steps=nsteps)
                mine = integrate(policy, pfx, kv, state, z_i, nsteps)
                d = float((ref - mine).abs().max())
                rep.setdefault("P5", {})[tag] = {"max_abs_delta": d, "atol": P5_ATOL,
                                                 "passes": d <= P5_ATOL}
                assert d <= P5_ATOL, f"P5 FAIL ({tag}): own integrator differs by {d}"
                print(f"[F] P5 {tag} PASS max|Δ|={d:.2e}", flush=True)
            for ti, t in enumerate(GF.T_GRID):
                x = probe_points(a_ref.unsqueeze(0).expand(1, chunk, adim),
                                 z_i.unsqueeze(0), t)[0]               # (N_NOISE, chunk, adim)
                v = field_eval(policy, pfx, kv, state, x, t)
                v_bank[i, :, ti] = v.cpu().numpy().astype(np.float16)
            if do_behavior:
                act10[i] = integrate(policy, pfx, kv, state, z_i, 10).cpu().numpy()
                act5[i] = integrate(policy, pfx, kv, state, z_i, 5).cpu().numpy()
            if i % 8 == 0:
                print(f"[F] {tag}: obs {i + 1}/{n_obs}", flush=True)
        np.savez_compressed(out / f"bank_{tag}.npz", v=v_bank, z_obs=z_obs_bank,
                            a_ref=aref_bank, act10=act10, act5=act5)
        adim_valid = cfg.output_features["action"].shape[0] if hasattr(
            cfg.output_features["action"], "shape") else cfg.output_features["action"]["shape"][0]
        cells.append({"kind": "bank", "tag": tag, "pproc_sha": pproc_sha,
                      "chunk": chunk, "adim": adim, "adim_valid": int(adim_valid),
                      "num_inference_steps": nsteps,
                      "v_rms": float(np.sqrt((v_bank.astype(np.float64) ** 2).mean()))})
        _w()
        n_expert, _keys = GF.expert_param_count(policy)
        del policy
        torch.cuda.empty_cache()
        return n_expert, int(adim_valid)

    def _prefix_args(policy, batch):
        from lerobot.utils.constants import (OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS)
        images, img_masks = policy._preprocess_images(batch)
        return (images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK], policy.prepare_state(batch))

    if a.census_only:
        units, steps = [], []          # skip pi0/F2/films/F1 below; census block still runs
    else:
        # ---- 2. pi0 bank + P5 + behavioral refs ---------------------------------------------
        pi0_dir = build_unit_dir("pi0", "/tmp/gestalt_pi0")
        n_expert, adim_valid = bank_policy("pi0", pi0_dir, do_p5=True, do_behavior=True)
        rep["expert_params"] = n_expert
        _w()

    if not a.census_only:
        # ---- 3. F2 surrogate ladder -------------------------------------------------------------
        bank = np.load(out / "bank_pi0.npz")
        fit_idx = list(range(0, n_obs, 2))
        held_idx = list(range(1, n_obs, 2))
        pca_m, pca_c = GF.pca_fit(bank["z_obs"][fit_idx], q=64)
        z_red = GF.pca_apply(bank["z_obs"], pca_m, pca_c)                   # (n_obs, 64)
        chunk, adim = bank["v"].shape[3], bank["v"].shape[4]
        Dx = chunk * adim

        def _rows_for(idx_list):
            Phi = {lvl: [] for lvl in GF.LADDER}
            V = []
            z_all = GF.probe_noise(n_obs, GF.N_NOISE, chunk, adim).numpy()
            for i in idx_list:
                for j in range(GF.N_NOISE):
                    for ti, t in enumerate(GF.T_GRID):
                        x = (t * z_all[i, j] + (1 - t) * bank["a_ref"][i]).reshape(-1)
                        for lvl in GF.LADDER:
                            Phi[lvl].append(GF.features(x, t, z_red[i], lvl))
                        V.append(bank["v"][i, j, ti].astype(np.float64).reshape(-1))
            return {lvl: np.stack(Phi[lvl]) for lvl in GF.LADDER}, np.stack(V)

        Phi_fit, V_fit = _rows_for(fit_idx)
        Phi_held, V_held = _rows_for(held_idx)
        f2 = {}
        W_by_lvl = {}
        for lvl in GF.LADDER:
            W = GF.fit_ridge(Phi_fit[lvl], V_fit)
            W_by_lvl[lvl] = W
            mse_held = float(((Phi_held[lvl] @ W - V_held) ** 2).mean())
            f2[lvl] = {"params": GF.surrogate_params(lvl, Dx, Dx),
                       "proxy_mse_held": mse_held,
                       "field_var_held": float((V_held ** 2).mean())}
            print(f"[F] F2 {lvl}: params={f2[lvl]['params']} held-MSE={mse_held:.5f}", flush=True)
        rep["F2_ladder"] = f2
        _w()

        # FR-7 behavioral gate on held obs: surrogate 10-step integration vs true 10-step actions;
        # threshold = the true field's OWN 5-vs-10-step deviation.
        z_np = GF.probe_noise(n_obs, GF.N_NOISE, chunk, adim).numpy()
        thresh = float(((bank["act5"][held_idx] - bank["act10"][held_idx]) ** 2).mean())
        for lvl in GF.LADDER:
            W = W_by_lvl[lvl]
            mses = []
            for i in held_idx:
                for j in range(GF.N_NOISE):
                    x = z_np[i, j].reshape(-1).astype(np.float64)
                    for step in range(10):
                        t = 1.0 + step * (-0.1)
                        v = GF.features(x, t, z_red[i], lvl) @ W
                        x = x + (-0.1) * v
                    mses.append(((x - bank["act10"][i, j].reshape(-1)) ** 2).mean())
            f2[lvl]["behavior_mse"] = float(np.mean(mses))
            f2[lvl]["behavior_thresh_5v10"] = thresh
            f2[lvl]["gate_passes"] = bool(np.mean(mses) <= thresh)
            print(f"[F] F2 {lvl}: behavior={f2[lvl]['behavior_mse']:.5f} vs thresh={thresh:.5f} "
                  f"passes={f2[lvl]['gate_passes']}", flush=True)
        passing = [lvl for lvl in GF.LADDER if f2[lvl]["gate_passes"]]
        rep["F2_verdict"] = {
            "passing_levels": passing,
            "borges_ratio": (min(f2[lvl]["params"] for lvl in passing) / n_expert) if passing else None,
            "note": ("no rung passed the FR-7 gate at any tested budget — the field's Borges ratio "
                     "is ~1 on this ladder (a result either way, spec §1.4)" if not passing else
                     "smallest gate-passing rung over the counted expert params")}
        _w()

    # ---- 4. films ---------------------------------------------------------------------------
    for unit in units:
        for s in steps:
            tag = f"{unit}_{s:08d}"
            d = _pull_ckpt(f"{S3_FILM}/{unit}/seed0/{s:08d}/pretrained_model",
                           f"/tmp/gestalt_film/{tag}")
            bank_policy(tag, d, do_p5=(s == steps[0]), do_behavior=False)
            shutil.rmtree(d)

    # ---- 5. F1 ------------------------------------------------------------------------------
    if len(units) == 2:
        shas = {c["tag"]: c["pproc_sha"] for c in cells if c["kind"] == "bank"}
        pair_ok = all(shas[f"{units[0]}_{s:08d}"] == shas[f"{units[1]}_{s:08d}"] for s in steps)
        f1 = {"preprocessor_pair_match": pair_ok, "per_step": {}}
        if not pair_ok:
            f1["note"] = "VOID: the two units' preprocessors differ — x_t points not shared"
        def _v(tag):
            return np.load(out / f"bank_{tag}.npz")["v"].reshape(-1, chunk, adim).astype(
                np.float32)

        # drifts FIRST (they are the FR-3 (b) reference the pair diff's p is computed against)
        for u in units:
            for s in steps[1:]:
                f1["per_step"].setdefault(s, {})[f"drift_{u}"] = GF.field_diff(
                    _v(f"{u}_{s:08d}"), _v(f"{u}_{s - 512:08d}"), adim_valid, n_boot=400)
        for s in steps:
            e = f1["per_step"].setdefault(s, {})
            drift = (max(e[f"drift_{u}"]["d_rel"] for u in units)
                     if all(f"drift_{u}" in e for u in units) else None)
            e["drift_band"] = drift
            e["diff"] = GF.field_diff(_v(f"{units[1]}_{s:08d}"), _v(f"{units[0]}_{s:08d}"),
                                      adim_valid, ref_null=drift)
            if drift is not None:
                width_ok = e["diff"]["hi95"] > e["diff"]["lo95"] > 0
                e["develops"] = bool(pair_ok and width_ok and e["diff"]["lo95"] > drift)
        rep["F1"] = f1
        _w()
        print(f"[F] F1: {json.dumps({s: f1['per_step'][s].get('develops') for s in steps[1:]})}",
              flush=True)

    # ---- 6. census (pi05) -------------------------------------------------------------------
    if not a.skip_census and not a.smoke:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        from lerobot.policies.pi0 import modeling_pi0 as MP
        make_att_2d_masks = MP.make_att_2d_masks
        p5dir = build_unit_dir("pi05", "/tmp/gestalt_pi05")
        pol = PI05Policy.from_pretrained(str(p5dir)).to(device).eval()
        prep4d5 = (pol.model._prepare_attention_masks_4d
                   if hasattr(pol.model, "_prepare_attention_masks_4d")
                   else MP.prepare_attention_masks_4d)
        pre5, _ = make_pre_post_processors(pol.config, pretrained_path=str(p5dir))
        chunk5, adim5 = pol.config.chunk_size, pol.config.max_action_dim
        z5 = GF.probe_noise(n_obs, GF.N_NOISE, chunk5, adim5)
        stats = {t: [] for t in GF.T_GRID}
        for i, row in enumerate(rows):
            batch = to_dev(pre5(to_dev(raw_batch(row, GF.N_NOISE), device)), device)
            images, img_masks = pol._preprocess_images(batch)
            toks, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
            with torch.no_grad():
                pe, ppm, pam = pol.model.embed_prefix(images, img_masks, toks, masks)
                att = prep4d5(make_att_2d_masks(ppm, pam))
                pos = torch.cumsum(ppm, dim=1) - 1
                pol.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
                _, kv = pol.model.paligemma_with_expert.forward(
                    attention_mask=att, position_ids=pos, past_key_values=None,
                    inputs_embeds=[pe, None], use_cache=True)
                a_ref5 = pol.prepare_action(batch)[0] if "action" in batch else None
                z_i = z5[i].to(device)
                for t in GF.T_GRID:
                    x = t * z_i + (1 - t) * (a_ref5 if a_ref5 is not None else 0.0)
                    ts = torch.full((x.shape[0],), float(t), dtype=torch.float32, device=device)
                    v = pol.model.denoise_step(prefix_pad_masks=ppm, past_key_values=kv,
                                               x_t=x, timestep=ts)
                    stats[t].append(float(v.float().pow(2).mean().sqrt()))
        rep["census_pi05"] = {"v_rms_by_t": {t: float(np.mean(v)) for t, v in stats.items()},
                              "n_obs": n_obs, "note": "own-preprocessing field summary; no cross-model diff"}
        _w()

    rep["elapsed_s"] = time.time() - t0
    _w()
    print("GESTALT_FIELD_OK", flush=True)


if __name__ == "__main__":
    main()
    # D7: datasets/cuda extension threads abort the interpreter during finalization
    # (PyGILState_Release fatal, rc=-6) AFTER all artifacts are banked; every write is
    # explicit and already streamed, so skip finalization outright.
    import os as _os
    _os._exit(0)
