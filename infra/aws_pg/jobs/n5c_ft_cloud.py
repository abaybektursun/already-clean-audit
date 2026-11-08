"""N5c — the official-ft reference cloud (pi-trajectory N5 prereg §3). CPU, fp64, fail-loud.

Convert E2b's "π0 reuse > Haar (an arbitrary ROTATION null)" into "π0 reuse > a parallel FULL fine-tune
of the SAME pt-224 init". Hold ΔW₁ = (pt-224.LM − gemma-2b) FIXED as the projection basis (re-SVD'd by the
FROZEN instrument, deterministic ⇒ U_r(ΔW₁) bit-identical to E2b); swap ONLY ΔW₂ → ΔW_other^task =
(ft-task.LM − pt-224.LM) over the ~24 official google/paligemma-3b-ft-{task}-224 checkpoints (each a
full-parameter FT of the identical pt-224 init). Each ft scored by the frozen E2b instrument on the same
18×7 spine — CPU-only, no training. π0 is the (N+1)-th member (its frozen per_tensor reused, NOT recomputed
here; the rank/cut lives in n5c_reagg.py).

Instrument FROZEN, unchanged (no estimator edit ⇒ no E1 re-calibration; firewall + all E1 gates carry).
Load-bearing nulls computed only at r=32 and r=64 (the G-power-resolvable r); deterministic reuse_stats at
all --r-grid. Per-(tensor,estimand,r) Haar seed is CHECKPOINT-INDEPENDENT → a paired contrast against a
shared null realization (deliberate variance reduction, §3). GEOMETRY-ONLY.

  # smoke (1 ft, time it): --args "--ft-tasks vqav2 --n-null-draws 1000 --seed 1337" --budget-min 120
  # full cloud (resumable; incremental write every ft; r7i.4xlarge = 16 vCPU = Standard quota):
  pg run infra.aws_pg.jobs.n5c_ft_cloud --type r7i.4xlarge --no-weights \
      --args "--r-grid 32,64 --n-null-draws 1000 --seed 1337" --budget-min 600
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess

BUCKET = "pg-lineage-769063704696"
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MLP = ("down_proj", "gate_proj", "up_proj")
LOAD_BEARING_R = (32, 64)               # the G-power-resolvable r — the only r the nulls/decision read
INSTR_SHA = "693a7d04"                  # reuse_estimators.py (pi-e1-instrument-freeze-v2)
GR_SHA = "c7389652"                     # gain_routing.py (vendored @ w12 12399e2)
PG1_REPO = "google/paligemma-3b-pt-224"
BASE_REPO = "google/gemma-2b"
MLP_FT_SCALE_FLOOR = 1e-3               # pooled MLP rel-Frobenius floor: a frozen-MLP/QLoRA ft reads ≈0 → drop


def _aws_json(key):
    r = subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{key}", "-", "--region", "us-east-1"],
                       check=True, capture_output=True)
    return json.loads(r.stdout)


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _seed(base, key, est, r):
    """Per-(tensor,estimand,r) null seed — CHECKPOINT-INDEPENDENT (no ft term) ⇒ identical random r-frames
    reused across all checkpoints (the §3 shared null realization / common-random-numbers paired contrast)."""
    return int(hashlib.md5(f"{key}|{est}|{r}|{base}".encode()).hexdigest(), 16) % (2 ** 31)


def _empirical_p(value, samples, side):
    import numpy as np
    s = np.asarray(samples, dtype=float)
    n = s.size
    if side == "upper":
        return float((np.sum(s >= value) + 1) / (n + 1))
    return float((np.sum(s <= value) + 1) / (n + 1))


def _norm_keys(sd):
    return {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}


def _lm_decoder_sd(model):
    from infra.pg.entities import resolve_submodule_paths
    lm = model.get_submodule(resolve_submodule_paths(model)["language_model"])
    if hasattr(lm, "lm_head") and hasattr(lm, "model"):
        lm = lm.model
    return _norm_keys(lm.state_dict())


def _matched_18x7(w0, w1, w2, n_layers):
    """The 18×7 decoder matrices present in all three (gemma-2b, pt-224.LM, ft.LM). Asserts layer count +
    per-(type,layer) presence explicitly (key-intersection alone would miss a gap)."""
    keys = set(w0) & set(w1) & set(w2)
    out = {}
    for k in sorted(keys):
        if any(t in k for t in PROJ) and w0[k].ndim == 2:
            assert w0[k].shape == w1[k].shape == w2[k].shape, f"shape mismatch {k}"
            out[k] = (w0[k], w1[k], w2[k])
    expect = {f"layers.{i}.{('self_attn' if p.startswith(('q','k','v','o')) else 'mlp')}.{p}.weight"
              for i in range(n_layers) for p in PROJ}
    missing = sorted(expect - set(out))
    assert not missing, f"18×7 incomplete: {len(missing)} missing (e.g. {missing[:5]})"
    assert len(out) == n_layers * 7, f"expected {n_layers*7}, got {len(out)}"
    return out


def _enumerate_ft_repos(api):
    """ALL transformers-format google/paligemma-3b-ft-{task}-224 repos (enumerate, NOT curate post-hoc).
    Excludes -jax (big_vision), -448/-896 (different vision regime), paligemma2-* (generation boundary)."""
    repos = []
    for m in api.list_models(author="google", search="paligemma-3b-ft"):
        rid = m.id
        name = rid.split("/")[-1]
        if name.endswith("-224") and "-jax" not in name and name.startswith("paligemma-3b-ft-"):
            repos.append(rid)
    return sorted(set(repos))


def _delete_cache(repo_id):
    """Free the ft's HF snapshot after differencing (24×~12 GB fp32 would overflow the 200 GB root)."""
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", ""), "hub")
    d = os.path.join(hub, "models--" + repo_id.replace("/", "--"))
    if os.path.isdir(d):
        shutil.rmtree(d)                                 # no try/except: a cleanup failure must fail loud


def _participation_ratio(spectrum):
    """PR = (Σσ)²/Σσ² — the eff-rank proxy (control 2). High PR = spread spectrum; low = concentrated."""
    import numpy as np
    s = np.asarray(spectrum, dtype=np.float64)
    num = float(s.sum()) ** 2
    den = float((s ** 2).sum())
    return num / den if den > 0 else float("nan")


def _write_cloud(out, rep, members, pi0_member, dropped, preflights):
    """Incremental write after EVERY ft (the 30s S3 sync uploads it) → a timeout never loses completed
    members, and a re-launch resumes from this file. Idempotent; the final write is just the last one."""
    rep = {**rep, "n_members": len(members), "dropped": dropped, "preflights": preflights}
    (out / "n5c_cloud.json").write_text(json.dumps(
        {"summary": rep, "members": members, "pi0_member": pi0_member}, indent=2))


def _resume_prior(out):
    """Fetch any prior n5c_cloud.json for this prefix from S3 (a re-launch after a timeout). Returns
    (members, dropped, preflights, done_set, pi0_member) or empties if absent — `aws s3 cp` nonzero ⇒
    start fresh. A prior π0 with a stored bit-identity proof is reused (skip the ~1-ft recompute)."""
    dst = out / "n5c_cloud.json"
    r = subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/results/n5c_ft_cloud/out/n5c_cloud.json",
                        str(dst), "--region", "us-east-1"], capture_output=True)
    if r.returncode != 0:
        return [], {}, {}, set(), None
    d = json.loads(dst.read_text())
    members = d.get("members", [])
    dropped = d.get("summary", {}).get("dropped", {})
    preflights = d.get("summary", {}).get("preflights", {})
    done = {m["repo_id"] for m in members} | set(dropped)
    pi0_prev = d.get("pi0_member") if d.get("summary", {}).get("pi0_freeze_max_abs_diff", 1.0) < 1e-9 else None
    print(f"[N5c] RESUME: {len(members)} members + {len(dropped)} dropped done; π0 reused={pi0_prev is not None}")
    return members, dropped, preflights, done, pi0_prev


def _difference_matched(matched, r_grid, n_draws, seed, e1_gpower):
    """The per-tensor differencing loop, shared by every checkpoint (ft cloud AND π0). ΔW₁=W1−W0 is the
    FIXED basis (recomputed inside the frozen reuse_stats); ΔW₂=W2−W1 is the per-checkpoint edit. Nulls
    ONLY at the load-bearing r (the cost driver); deterministic reuse_stats at every r in r_grid."""
    import torch
    from infra.pg import reuse_estimators as re

    per_tensor = []
    for key, (a0, a1, a2) in matched.items():
        W0 = a0.double(); W1 = a1.double(); W2 = a2.double()
        dW1 = W1 - W0
        dW2 = W2 - W1
        ftype = next(t for t in PROJ if t in key)
        layer = int(key.split("layers.")[1].split(".")[0])
        sig2 = torch.linalg.svdvals(dW2).tolist()
        rec = {"key": key, "type": ftype, "layer": layer, "shape": list(dW1.shape),
               "frob_dW1": dW1.norm().item(), "frob_dW2": dW2.norm().item(),
               "pr_dW2": _participation_ratio(sig2)}
        do, di = dW1.shape[0], dW1.shape[1]
        per_r = {}
        for r in r_grid:
            st = re.reuse_stats(dW1, dW2, r)              # ONE ΔW₁ SVD per (tensor,r); returns dw1_sigma_top
            assert st["recon_resid"] < 1e-9, f"ρ_L+ρ_⊥ identity FAIL {key} r={r}: {st['recon_resid']}"
            entry = {"rho_L": st["rho_L"], "rho_R": st["rho_R"], "rho_J": st["rho_J"], "a": st["a"],
                     "rho_perp_L": st["rho_perp_L"], "forced_floor_L": st["forced_floor_L"],
                     "trunc_gap_rel": st["trunc_gap_rel"], "dw1_inblock_frac_flagged": st["dw1_inblock_frac_flagged"]}
            if r in LOAD_BEARING_R:                       # nulls ONLY at the load-bearing r (the cost driver)
                st_top = st["dw1_sigma_top"]              # = s[:r] (r≤rank) — no extra SVD beyond reuse_stats
                hL = re.haar_null(sig2, st_top, r, do, di, "rho_L", n_draws, _seed(seed, key, "rho_L", r))
                hA = re.haar_null(sig2, st_top, r, do, di, "a", n_draws, _seed(seed, key, "a", r))
                cls = re.classify_reuse(st["rho_L"], st["a"], hL, hA, st["forced_floor_L"], e1_gpower[r])
                entry |= {
                    "p_rho_L": _empirical_p(st["rho_L"], hL["samples"], "upper"),
                    "p_a_pos": _empirical_p(st["a"], hA["samples"], "upper"),
                    "haar_rho_L": {"p95.0": hL["pcts"]["p95.0"], "mean": hL["mean"], "std": hL["std"]},
                    "a_band": [hA["pcts"]["p2.5"], hA["pcts"]["p97.5"]], "a_halfwidth_frozen": e1_gpower[r],
                    "bin": cls["bin"], "a_resolvable": cls["a_resolvable"], "rho_margin_sigma": cls["rho_margin_sigma"],
                }
            per_r[str(r)] = entry
        rec["reuse"] = per_r
        per_tensor.append(rec)
    return per_tensor


def _process_ft(repo_id, sha, w0, w1, w1_norm_mlp, r_grid, n_draws, seed, e1_gpower):
    """Difference ONE ft through the frozen instrument → (per_tensor table, frame-preflight dict) OR
    (None, drop-reason). Loads the ft, runs the §3 frame preflight (drop on fail), then _difference_matched."""
    import torch
    from transformers import AutoConfig, PaliGemmaForConditionalGeneration
    from infra.pg import reuse_estimators as re

    cfg = AutoConfig.from_pretrained(repo_id, revision=sha)
    ft = PaliGemmaForConditionalGeneration.from_pretrained(
        repo_id, revision=sha, torch_dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    w2 = _lm_decoder_sd(ft)
    pf = {"dtype_hint": str(getattr(cfg, "torch_dtype", None))}

    matched = _matched_18x7(w0, w1, w2, 18)
    pf["n_matched"] = len(matched)

    # frame preflight (control 5): MQA k/v single-head rows; fp64 recon-identity on q_proj L0. Drop on fail.
    k0 = matched["layers.0.self_attn.k_proj.weight"]
    v0 = matched["layers.0.self_attn.v_proj.weight"]
    pf["kv_rows"] = [int(k0[2].shape[0]), int(v0[2].shape[0])]
    if not (k0[2].shape[0] == v0[2].shape[0] == 256 and k0[2].shape[1] == 2048):
        return None, {"drop": "mqa_misaligned", **pf}
    q0 = matched["layers.0.self_attn.q_proj.weight"]
    recon = re.reuse_stats((q0[1] - q0[0]).double(), (q0[2] - q0[1]).double(), 32)["recon_resid"]
    pf["q0_recon_resid"] = recon
    if recon >= 1e-10:
        return None, {"drop": "recon_identity_fail", **pf}

    # MLP FT-scale provenance assert (§3 EXCLUDE): a frozen-vision/QLoRA ft has ≈0 MLP edit and would RIG
    # the MLP-extend null. ΔW₂ MLP rel-Frobenius must be FT-scale. Drop (do NOT silently degrade) if not.
    mlp_d2 = sum((m[2].double() - m[1].double()).norm().item() ** 2 for k, m in matched.items()
                 if any(t in k for t in MLP)) ** 0.5
    pf["mlp_dW2_rel_frob"] = mlp_d2 / w1_norm_mlp
    if pf["mlp_dW2_rel_frob"] < MLP_FT_SCALE_FLOOR:
        return None, {"drop": "mlp_not_ft_scale", **pf}

    per_tensor = _difference_matched(matched, r_grid, n_draws, seed, e1_gpower)
    del ft, w2, matched
    return per_tensor, {"drop": None, **pf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ft-tasks", default="", help="comma slugs (e.g. vqav2,gqa); empty ⇒ enumerate all -224")
    ap.add_argument("--r-grid", default="32,64", help="reuse_stats r's; default = the load-bearing 32,64 only")
    ap.add_argument("--n-null-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r_grid = [int(x) for x in a.r_grid.split(",")]
    assert set(LOAD_BEARING_R) <= set(r_grid), "r-grid must include the load-bearing 32,64"

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                                    # FIRST — gated repos 401 otherwise
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, PaliGemmaForConditionalGeneration
    from infra.pg import n5c_aggregate
    import infra.pg.reuse_estimators as _re_mod
    import infra.pg.gain_routing as _gr_mod

    rep = {"GEOMETRY_ONLY": True, "seed": a.seed, "n_null_draws": a.n_null_draws, "r_grid": r_grid,
           "load_bearing_r": list(LOAD_BEARING_R)}

    # provenance gate — frozen instrument, unchanged (no estimator edit ⇒ E1 carries; firewall holds).
    rep["instrument_sha256"] = _sha256(_re_mod.__file__)
    rep["gain_routing_sha256"] = _sha256(_gr_mod.__file__)
    assert rep["instrument_sha256"].startswith(INSTR_SHA), \
        f"reuse_estimators not the v2 freeze: {rep['instrument_sha256'][:12]} != {INSTR_SHA}"
    assert rep["gain_routing_sha256"].startswith(GR_SHA), \
        f"gain_routing drifted: {rep['gain_routing_sha256'][:12]} != {GR_SHA}"
    e1 = _aws_json("results/e1_calibration/out/calibration_e1.json")
    assert e1["all_pass"], "E1 calibration not green — HALT (instrument carries from E1)"
    e1_gpower = {int(k): v for k, v in e1["gates"]["G-power"]["observed"]["half_width"].items()}
    rep["e1_gpower_halfwidth"] = e1_gpower

    # FIXED basis: W0 = gemma-2b, W1 = pt-224.LM (same SHAs as E2b ⇒ ΔW₁ + its SVD bit-identical).
    api = HfApi()
    rep["base_sha"] = api.model_info(BASE_REPO).sha
    rep["pg1_sha"] = api.model_info(PG1_REPO).sha
    base = AutoModelForCausalLM.from_pretrained(BASE_REPO, torch_dtype=torch.float32,
                                                device_map="cpu", attn_implementation="eager")
    w0 = _norm_keys(base.model.state_dict())
    assert int(base.config.num_hidden_layers) == 18, "expected Gemma-1-2B 18 layers — wrong base"
    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        PG1_REPO, torch_dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    w1 = _lm_decoder_sd(pg)
    rep["pt224_dtype_hint"] = str(getattr(pg.config, "torch_dtype", None))
    del pg
    # the FIXED-basis MLP norms used by the FT-scale provenance assert.
    w1_norm_mlp = sum(w1[k].double().norm().item() ** 2 for k in w1
                      if any(t in k for t in MLP) and w1[k].ndim == 2) ** 0.5

    # π0 — the (N+1)-th member. Re-difference from the SAME sealed E1.5 backbone sd under the FROZEN
    # instrument and ASSERT bit-identity (≤1e-9) of per-tensor a/ρ_L at the load-bearing r to the frozen
    # e2_geometry.json — strictly stronger than "reuse": it proves the freeze reproduces, and yields π0 in
    # the identical schema (pr_dW2 / enrichment the frozen json lacks). Reused as-is on a resume launch.
    members, dropped, preflights, done, pi0_prev = _resume_prior(out)
    if pi0_prev is not None:                              # validated π0 from a prior launch — skip the recompute
        pi0_member, rep["pi0_freeze_max_abs_diff"], rep["pi0_reused_from_resume"] = pi0_prev, 0.0, True
        print(f"[N5c] π0 reused from resume; a_exL17mlp r32={pi0_member['A']['32']['a_pooled_exL17mlp']:+.4f}")
    else:
        from safetensors.torch import load_file
        sealed = out / "pi0_backbone_lm_sd.safetensors"
        subprocess.run(["aws", "s3", "cp",
                        f"s3://{BUCKET}/results/e1_5_port/out/pi0_backbone_lm_sd.safetensors",
                        str(sealed), "--region", "us-east-1"], check=True)
        matched_pi0 = _matched_18x7(w0, w1, _norm_keys(load_file(str(sealed))), 18)
        pt_pi0 = _difference_matched(matched_pi0, r_grid, a.n_null_draws, a.seed, e1_gpower)
        assert len(pt_pi0) == 126, f"π0: expected 126 per-tensor rows, got {len(pt_pi0)}"
        frozen = {r["key"]: r for r in _aws_json("results/e2_lm_geometry/out/e2_geometry.json")["per_tensor"]}
        maxdiff = max(abs(rec["reuse"][str(r)][m] - frozen[rec["key"]]["reuse"][str(r)][m])
                      for rec in pt_pi0 for r in LOAD_BEARING_R for m in ("a", "rho_L"))
        rep["pi0_freeze_max_abs_diff"] = maxdiff
        assert maxdiff < 1e-9, f"π0 re-difference DRIFTED from the freeze: max|Δ|={maxdiff} — not bit-identical"
        pi0_member = n5c_aggregate.cloud_member(pt_pi0, "lerobot/pi0_base", r_grid=LOAD_BEARING_R, seed=a.seed)
        print(f"[N5c] π0 member built; freeze max|Δ|={maxdiff:.2e}  a_exL17mlp r32={pi0_member['A']['32']['a_pooled_exL17mlp']:+.4f}")

    # the cloud: enumerate (default) or pin an explicit --ft-tasks list; resolve+pin SHAs at run (the deposit).
    if a.ft_tasks.strip():
        repos = [f"google/paligemma-3b-ft-{t.strip()}-224" for t in a.ft_tasks.split(",") if t.strip()]
    else:
        repos = _enumerate_ft_repos(api)
    rep["cloud_repos"] = repos
    rep["cloud_shas"] = {r: api.model_info(r).sha for r in repos}
    print(f"[N5c] cloud = {len(repos)} ft checkpoints; FIXED basis ΔW₁ = pt-224.LM − gemma-2b")

    _write_cloud(out, rep, members, pi0_member, dropped, preflights)   # ensure π0 lands before the loop
    for repo_id in repos:
        if repo_id in done:
            continue
        sha = rep["cloud_shas"][repo_id]
        print(f"[N5c] {repo_id}@{sha[:8]} ...", flush=True)
        per_tensor, pf = _process_ft(repo_id, sha, w0, w1, w1_norm_mlp,
                                     r_grid, a.n_null_draws, a.seed, e1_gpower)
        preflights[repo_id] = pf                          # dtype_hint recorded; uniform fp32 load is the guard
        if per_tensor is None:
            dropped[repo_id] = pf["drop"]
            print(f"       DROP ({pf['drop']})")
        else:
            assert len(per_tensor) == 126, f"{repo_id}: expected 126 per-tensor rows, got {len(per_tensor)}"
            mem = n5c_aggregate.cloud_member(per_tensor, repo_id, r_grid=LOAD_BEARING_R, seed=a.seed)
            members.append(mem)
            a32 = mem["A"]["32"]; print(f"       a_exL17mlp r32={a32['a_pooled_exL17mlp']:+.4f} "
                                        f"survives={a32['survives_exL17mlp']} enr32={mem['rhoL_enrichment']['32']:.2f}")
        _delete_cache(repo_id)
        _write_cloud(out, rep, members, pi0_member, dropped, preflights)   # incremental durability

    print("=" * 78)
    print(f"N5C_CLOUD_OK members={len(members)} dropped={len(dropped)} "
          f"pi0_freeze_maxdiff={rep['pi0_freeze_max_abs_diff']:.2e} "          # rep field, defined on both paths
          f"null_ft={[m['repo_id'] for m in members if all(m['null_ft_on_rhoL'].values())]}")


if __name__ == "__main__":
    main()
