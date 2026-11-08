"""TESSERA K3 — necessary rank n* of the RETAINED multimodal capability. Spec §2 K3, §5 P3, §6 K3.

THE QUESTION (§1.3 K3). H_compact vs H_redundant for VQA: is the capability stored in few
directions or many? And — the sharper form amendment A10 names — does π0.5's RETAINED VQA ride the
same directions PG1's VQA needs?

THE REMOVAL OBJECT (amendment A10). QUORUM's n* is always "rank removed from the write that
CREATED the capability". VQA was created by the INSTALL hop, not by the π0.5 hop; removing the
π0.5 write would move π0.5 back toward PG1 and *raise* VQA, so the naive reading is backwards, not
merely different. Both arms therefore remove rank from the SAME install write Δ_install =
PG1 − gemma-2b, and differ only in which unit is scored:

  PG1 arm    cell = gemma-2b + (Δ_install − proj_r Δ_install)  =  PG1 − proj_r Δ_install
  π0.5 arm   cell = π0.5 − proj_r Δ_install

The second is built by handing `build_decoder` a base of (π0.5 − Δ_install) and a target of π0.5,
so its Δ is again Δ_install and the SAME surgery code produces both — a difference between the two
arms cannot be a difference between two implementations.

  same n*      π0.5's retained VQA rides the directions the install created (retained-by-riding)
  different n* π0.5 re-encoded it somewhere else

π0 is an impossible cell (§2 K3): nothing retained to break.

THE READOUT, AND WHY IT IS QUORUM'S (amendment A11). §2 K3 says "forced-choice VQA under the
LACUNA shell". This job uses QUORUM's own `_quorum_shared.score`/`gain_of` instead, because every
threshold the ladder applies — ρ_break, `MIN_GAIN_FULL`, the planted-rank-1 acceptance band — was
calibrated on THAT readout. Keeping the frozen thresholds while swapping the readout underneath
them would import one instrument's calibration into another, which is the firewall violation the
house rule names explicitly. The deviation is recorded in the result rather than silently taken.

P3, FIRST IN THE ORDER. A planted rank-1 capability must read n* ≈ 1-2. If it does not, the
readout is VOID by QUORUM's own rule and no n* here is reported as a measurement.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_k3 \
      --args "--banks-s3 s3://.../cleave_banks2.pt --rho-break-s3 s3://.../rb.json" \
      --type g6e.2xlarge,p5.4xlarge --s3-prefix results/tessera_k3 --budget-min 900

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

GEMMA1_REPO = "google/gemma-2b"


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--banks-s3", required=True)
    ap.add_argument("--rho-break-s3", required=True)
    ap.add_argument("--n-items", type=int, default=500)
    ap.add_argument("--arms", default="pg1,pi05")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _quorum_shared as QS
    from infra.pg import comwrite_surgery as SURG
    from infra.pg import quorum_core as QC

    device = "cuda"
    rep = {"object": "TESSERA K3 — n* of the retained multimodal capability",
           "removal_object": "the INSTALL write PG1 − gemma-2b (amendment A10)",
           "readout": "QUORUM's own score/gain_of (amendment A11 — the ladder's thresholds were "
                      "calibrated on this readout; swapping it under frozen thresholds would "
                      "import one instrument's calibration into another)",
           "arms": a.arms.split(",")}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    banks = torch.load(QS.fetch_s3(a.banks_s3, "/tmp/tessera_k3/banks.pt"),
                       map_location="cpu", weights_only=False)
    rho_break = float(json.loads(pathlib.Path(
        QS.fetch_s3(a.rho_break_s3, "/tmp/tessera_k3/rb.json")).read_text())["freeze"]["rho_break"])
    rep["rho_break"] = rho_break

    model, proc, layers, sha, base_ppl = QS.load_model("/tmp/tessera_k3")
    rep["pinned_sha"], rep["base_ppl"] = sha, base_ppl
    lm, paths, vt_orig = QS.lm_bundle(model)
    pg1_lm_sd = QS.decoder_sd(lm, device=device)
    QS.assert_grid(pg1_lm_sd)
    install = QS.make_installer(model, lm, paths, vt_orig)
    snap = QS.snapshot_base(model)

    def reset():
        QS.reset_model(model, snap)

    tmap = QS.build_task_map()
    task = tmap["vqav2"] if "vqav2" in tmap else tmap[sorted(tmap)[0]]
    rows = QS.task_rows(task, a.n_items)
    rep["task"] = getattr(task, "name", str(task))

    # ---- the install write: base = gemma-2b decoder, target = PG1 ----------------------------
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file
    from infra.pg.lacuna_shell import canon
    g1_raw = {}
    for f in [x for x in list_repo_files(GEMMA1_REPO) if x.endswith(".safetensors")]:
        g1_raw.update(load_file(hf_hub_download(GEMMA1_REPO, f)))
    g1_canon = {canon(k): v for k, v in g1_raw.items()}
    del g1_raw
    gemma1_lm_sd, excluded = {}, []
    for k in pg1_lm_sd:
        ck = canon(k)
        # `x or y` on TENSORS invokes __bool__ on a multi-element tensor and raises. The lookup
        # must fall through on ABSENCE, not on truthiness.
        src = g1_canon.get(ck)
        if src is None:
            src = g1_canon.get(ck.replace("language_model.", ""))
        assert src is not None, f"gemma-2b has no counterpart for {k!r} (canonical {ck!r})"
        if tuple(src.shape) != tuple(pg1_lm_sd[k].shape):
            # PG1's embedding is [257216, 2048]; gemma-2b's is [256000, 2048] — PG1 EXTENDED the
            # vocabulary for image tokens. Δ_install is therefore undefined on that table: there is
            # no gemma-2b row for the image placeholders. Excluding it is not a workaround, it is
            # the only defined choice, and the rank-removal surgery operates on the 18×7 decoder
            # grid regardless (`comwrite_surgery.is_target`). Recorded, never silently dropped.
            # The state dict handed to the installer must be COMPLETE — `load_state_dict` is
            # strict and a dropped key raises "Missing key(s)". So the key is KEPT, set to PG1's
            # OWN value, which makes Δ_install exactly zero on it. That is the semantically right
            # choice as well as the mechanically necessary one: the embedding contributes nothing
            # to the install write being removed, because that write is undefined there.
            excluded.append({"key": k, "pg1": list(pg1_lm_sd[k].shape),
                             "gemma2b": list(src.shape),
                             "handling": "set to PG1's own value ⇒ Δ_install = 0 on this key"})
            gemma1_lm_sd[k] = pg1_lm_sd[k].clone()
            continue
        gemma1_lm_sd[k] = src.to(pg1_lm_sd[k].dtype).to(device)
    # the surgery grid must survive the exclusion intact, or the install write is not the object
    grid_keys = [k for k, w in pg1_lm_sd.items() if SURG.is_target(k, w)]
    missing_grid = [k for k in grid_keys if k not in gemma1_lm_sd]
    assert not missing_grid, (
        f"the install write is missing {len(missing_grid)} of {len(grid_keys)} SURGERY-GRID "
        f"matrices, e.g. {missing_grid[:3]} — excluding the embedding must not touch the grid")
    rep["install_write_matrices"] = len(gemma1_lm_sd)
    rep["install_write_excluded"] = excluded
    rep["surgery_grid_matrices"] = len(grid_keys)
    _w()

    def sweep_arm(name, base_sd, ft_sd):
        # The reference must be the PRE-INSTALL state, i.e. the model carrying `base_sd`. A plain
        # reset() leaves PG1 in the shell, and installing base + (ft − base) reconstructs `ft`; for
        # the pg1 arm ft IS PG1, so the cell would equal the reference and gain_full ≈ 0 BY
        # CONSTRUCTION. The first K3 run measured exactly that (-0.0153) and correctly reported the
        # arm as having no range — the arm had no range because the reference was wrong.
        reset()
        install({k: v for k, v in base_sd.items()}, f"{name}/reference")
        base_res = QS.score(model, proc, task, rows, device)
        reset()

        def read(tag):
            return QS.gain_of(QS.score(model, proc, task, rows, device), base_res)

        full, _fro = QS.install_full_delta(base_sd, ft_sd, banks, install=install, read=read,
                                           reset=reset, seed=QS.SEED, device=device)
        g = full["gain"]
        if abs(g) < QC.MIN_GAIN_FULL:
            return {"arm": name, "gain_full": g,
                    "skipped": f"|gain_full| {abs(g):.4f} < MIN_GAIN_FULL {QC.MIN_GAIN_FULL} — "
                               f"this arm has no measurable range and no n* is reported"}
        sw = QS.rank_sweep(base_sd, ft_sd, banks, policy="energy", ladder=QC.RANK_LADDER,
                           install=install, read=read, reset=reset, gain_full=g, seed=QS.SEED,
                           device=device)
        ns = QC.necessary_rank([s["retention"] for s in sw], [s["rank"] for s in sw],
                               rho_break=rho_break)
        return {"arm": name, "gain_full": g, "n_star": ns["n_star"], "censored": ns["censored"],
                "sweep": sw}

    # ---- P3 FIRST: the planted rank-1 positive control ---------------------------------------
    # `inject_rank1_delta` takes the top-energy eigenvector MATRIX for ONE layer (d, k) — not a
    # dict over layers — and RETURNS a ΔW that lies entirely in the top-1 energy direction, so
    # `off_bases("energy", r>=1)` annihilates it and n* = 1 analytically. Recording the injection
    # without sweeping it would not have been a control at all; the sweep is the control.
    L0, MT = 0, "q_proj"
    # SCALE LADDER. At scale=1.0 the planted rank-1 moved the readout by |gain| = 0.0067, far under
    # MIN_GAIN_FULL = 0.08, so the sweep never ran and n* came back None. A plant that does not
    # perturb behaviour tests NOTHING — it has no dynamic range, exactly the failure P1's noise leg
    # had. Calibrating the plant until it is measurable is not relaxing the gate: the gate is
    # "given a plant the readout CAN see, does it read n* ≈ 1-2?", and an invisible plant cannot
    # pose that question. If NO scale on the ladder produces a measurable gain, the readout VOIDs
    # for real and that is reported.
    P3_SCALES = (1.0, 4.0, 16.0, 64.0, 256.0)
    tgt = [k for k in pg1_lm_sd if f"layers.{L0}." in k and f".{MT}." in k]
    assert len(tgt) == 1, f"P3: expected one (L{L0}, {MT}) matrix, found {tgt}"
    tk = tgt[0]
    p3_tries, with_p3, used_scale = [], None, None
    for sc in P3_SCALES:
        dW1 = QC.inject_rank1_delta(banks[L0]["sigma_evecs"], layer=L0, mtype=MT, scale=sc)
        planted_sd = {k: v.clone() for k, v in pg1_lm_sd.items()}
        assert tuple(dW1.shape) == tuple(planted_sd[tk].shape), (
            f"P3: planted ΔW {tuple(dW1.shape)} vs target {tuple(planted_sd[tk].shape)}")
        planted_sd[tk] = (planted_sd[tk].float()
                          + torch.as_tensor(dW1, dtype=torch.float32,
                                            device=planted_sd[tk].device)).to(planted_sd[tk].dtype)
        r = sweep_arm(f"P3_planted_rank1_s{sc:g}", pg1_lm_sd, planted_sd)
        p3_tries.append({"scale": sc, "gain_full": r.get("gain_full"),
                         "n_star": r.get("n_star"), "skipped": r.get("skipped")})
        print(f"[K3] P3 scale={sc:g} gain_full={r.get('gain_full')} n*={r.get('n_star')}",
              flush=True)
        if r.get("n_star") is not None:
            with_p3, used_scale = r, sc
            break
    ns_p3 = (with_p3 or {}).get("n_star")
    p3_ok = (ns_p3 is not None) and (not (with_p3 or {}).get("censored", False)) and ns_p3 <= 2.0
    rep["P3"] = {
        "rule": "given a plant the readout CAN see (|gain_full| ≥ MIN_GAIN_FULL), a rank-1 "
                "capability must read n* ≈ 1-2; otherwise the readout cannot resolve compact "
                "storage and every n* here VOIDs (QUORUM's frozen rule, spec §5)",
        "target": {"layer": L0, "mtype": MT, "key": tk},
        "scale_ladder": P3_SCALES, "tries": p3_tries, "scale_used": used_scale,
        "n_star": ns_p3, "censored": (with_p3 or {}).get("censored"),
        "gain_full": (with_p3 or {}).get("gain_full"),
        "passed": bool(p3_ok),
        "verdict": ("PASS" if p3_ok else
                    ("VOID-READOUT-NO-RANGE" if with_p3 is None else "VOID-READOUT"))}
    rep["P3_sweep"] = (with_p3 or {}).get("sweep")
    print(f"[K3] P3 -> {rep['P3']['verdict']} (n*={ns_p3}, scale={used_scale})", flush=True)
    _w()
    assert p3_ok, (
        f"P3 {rep['P3']['verdict']}: tries={p3_tries} — no n* in this job is a measurement and K3 "
        f"VOIDs by QUORUM's frozen rule")

    arms = {}
    if "pg1" in a.arms:
        arms["pg1"] = sweep_arm("pg1", gemma1_lm_sd, pg1_lm_sd)
        print(f"[K3] pg1 n*={arms['pg1'].get('n_star')} "
              f"censored={arms['pg1'].get('censored')}", flush=True)
        rep["arms_result"] = arms
        _w()
    if "pi05" in a.arms:
        # base = π0.5 − Δ_install, target = π0.5  ⇒  Δ = Δ_install, cell = π0.5 − proj_r Δ_install
        from infra.pg import lacuna_shell as LS
        full_sd, _s = LS.policy_full_sd("pi05")
        vlm, _h, _c = LS.policy_vlm_sd("pi05", full_sd=full_sd)
        del full_sd
        pi05_lm_sd, missing = {}, []
        for k in pg1_lm_sd:
            ck = canon(k)
            src = vlm.get(ck)
            if src is None:
                src = vlm.get("language_model." + ck)
            if src is None:
                # the published policies drop the tied `embed_tokens` under the openpi convention;
                # PG1's own value keeps the state dict COMPLETE for the strict installer and makes
                # this key contribute nothing to the contrast, which is correct either way
                pi05_lm_sd[k] = pg1_lm_sd[k].clone()
                missing.append(k)
                continue
            pi05_lm_sd[k] = src.to(pg1_lm_sd[k].dtype).to(device)
        rep["pi05_filled_from_pg1"] = missing
        shifted = {k: (pi05_lm_sd[k].float() - (pg1_lm_sd[k].float() - gemma1_lm_sd[k].float()))
                   .to(pg1_lm_sd[k].dtype) for k in pg1_lm_sd}
        arms["pi05"] = sweep_arm("pi05", shifted, pi05_lm_sd)
        print(f"[K3] pi05 n*={arms['pi05'].get('n_star')} "
              f"censored={arms['pi05'].get('censored')}", flush=True)

    rep["arms_result"] = arms
    ns = {k: v.get("n_star") for k, v in arms.items()}
    if all(v is not None for v in ns.values()) and len(ns) == 2:
        ratio = max(ns.values()) / max(min(ns.values()), 1e-9)
        rep["A10_read"] = {
            "n_star": ns, "ratio": ratio,
            "verdict": ("RIDES-THE-INSTALL" if ratio < 2.0 else "RE-ENCODED"),
            "why": ("π0.5's retained VQA breaks at the same install-write rank PG1's does — it "
                    "rides the directions the install created" if ratio < 2.0 else
                    "π0.5's retained VQA breaks at a different rank — it is not simply riding the "
                    "install's directions")}
    rep["elapsed_s"] = time.time() - t0
    _w()
    print("TESSERA_K3_OK", json.dumps(ns, default=str), flush=True)


if __name__ == "__main__":
    main()
