"""PRAGNANZ M4 — ONE {arm, seed} A/B cell on the FROZEN SAV recipe (spec §2 M4, §5 P4/P5).

Reproduces the SAV PG1 r512 confirmation cell (sav_cell.py) exactly, plus a gradient-space
intervention on the 18 q_proj adapter B-factors (IN axis; B init = 0 so every projected
step keeps rowspace(B) — hence the adapter's effective read-directions — out of the subspace):

  baseline  none (the P4 anchor: endpoint S must land in the banked same-seed boot CI)
  core      ERLANGEN consensus core basis (r=32/matrix, IN axis, sha-stable npz)
  random    rank-matched Haar, per-matrix seeds (the FULCRUM control, non-negotiable)
  dome      online DOME filter, k=16 (=C^2 for 4-choice MMLU), update-then-filter,
            samples = the 4 micro-batch grad deltas per step, scoped to the same 18 B-factors

M5 placement arms (same scaffold, mode='into', rank-matched 32-dim constraint on B):
  m5_incore  B constrained INTO span(core)      m5_rand  Haar 32-dim (per-matrix seeds)
  m5_perp    Haar 32-dim drawn INSIDE the core's complement

Trajectory readout: S on the deterministic 1/8 confirmation sub-bucket (hash % 16 == 1)
every EVAL_EVERY steps; endpoint: FULL confirmation split + wikitext ppl (collateral).
Frozen criterion (journaled pre-dispatch): S*_lo = 0.9 x min(banked seed S) primary,
S*_hi = 0.9 x mean(banked) sensitivity — crossing computed in the AGGREGATE job, not here.

    python -m infra.aws_pg.jobs.pragnanz_m4 --out <dir> --arm core --seed 1337 [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

CONTROLS_S3 = "s3://pg-lineage-769063704696/results/sav_controls_a2/out/sav_controls.json"
BASIS_S3 = "s3://pg-lineage-769063704696/results/erlangen_m_textvqa/out/erlangen_core_basis.npz"
BANKED_CELL_S3 = ("s3://pg-lineage-769063704696/results/sav_cellsE_PG1/out/"
                  "PG1_n16000_s{seed}_confirmation/cell_PG1_cap_r512_n16000_s{seed}_confirmation.json")
N_TRAIN = 16000
RANK = 512
EVAL_EVERY = 250
DOME_K = 16                      # C^2 for 4-choice MMLU (the paper's guidance; M2 froze 32
                                 # for open-vocab tasks — different C, disclosed)
RAND_SEED0 = 20260806            # per-matrix Haar seeds (FULCRUM per-matrix convention)
DOME_SEED0 = 20260806


def traj_pred(h):
    """Deterministic 1/8 sub-bucket of the CONFIRMATION half (h%2==1); pure hash rule so
    every arm/step evaluates the identical item set in identical order."""
    return int(h, 16) % 16 == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=["baseline", "core", "random", "dome",
                                      "m5_incore", "m5_perp", "m5_rand"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from datasets import load_dataset

    from infra.pg import pragnanz_core as PC
    from infra.pg import relearn as R
    from infra.pg import text_eval as TE
    from infra.pg import vlat_swapback as SB
    from infra.pg import w16_probes as W
    from infra.pg.entities import ENTITIES
    from infra.pg.lacuna_shell import canon
    from infra.pg.swap_back import boot_s_text, s_text
    from infra.aws_pg.jobs.vlat_retention import _extract_decoder
    from transformers import AutoTokenizer

    assert torch.cuda.is_available()
    device = "cuda"
    cell_id = f"m4_{a.arm}_s{a.seed}"
    rep = {"cell_id": cell_id, "arm": a.arm, "seed": a.seed, "smoke": a.smoke,
           "controls_s3": CONTROLS_S3, "basis_s3": BASIS_S3,
           "recipe_frozen": None, "trajectory": [], "torch": torch.__version__}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    # ---- controls (frozen recipe + BASE/FLOOR per-item) --------------------------------------
    ctrl_local = out / "sav_controls.json"
    subprocess.run(["aws", "s3", "cp", CONTROLS_S3, str(ctrl_local),
                    "--region", "us-east-1"], check=True)
    C = json.loads(ctrl_local.read_text())
    assert C["CONTROLS_OK"] is True and C["recipe"]["FROZEN"] is True
    fr = C["recipe"]
    rep["recipe_frozen"] = fr
    hashes = C["item_hashes"]
    b_pi, f_pi = C["BASE"]["per_item"], C["FLOOR"]["per_item"]
    conf_pred = W.is_confirmation
    b_conf = W.mask_by(b_pi, hashes, conf_pred)
    f_conf = W.mask_by(f_pi, hashes, conf_pred)
    conf_hashes = W.mask_by(hashes, hashes, conf_pred)
    b_traj = W.mask_by(b_pi, hashes, traj_pred)
    f_traj = W.mask_by(f_pi, hashes, traj_pred)
    traj_hashes = W.mask_by(hashes, hashes, traj_pred)
    rep["n_items"] = {"confirmation": len(conf_hashes), "trajectory": len(traj_hashes)}
    assert len(traj_hashes) > 500, f"trajectory sub-bucket too small: {len(traj_hashes)}"

    ent = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(ent.repo_id, revision=ent.revision)

    # ---- shell: PG1 decoder through the Gemma-1 swap-back (sav_cell verbatim) ----------------
    dec, src_sha = _extract_decoder("PG1", device, "")
    assert dec is not None
    model, swap_info = SB.load_swapback_model(dec, device)
    del dec
    torch.cuda.empty_cache()
    rep["decoder_source_sha"] = str(src_sha)
    rep["swap_gate"] = {k: v for k, v in swap_info.items()}

    arep = R.attach_adapters(model.model, rank=RANK, seed=a.seed, alpha=float(RANK))
    model.to(device)
    rep["adapter"] = {k: arep[k] for k in ("n_modules", "trainable_params", "rank", "alpha")}

    # ---- q_proj B-factor map + arm bases -----------------------------------------------------
    # basis npz keys are canon("language_model.<module>.weight") with "." -> "__" (as in M1);
    # here <module> iterates model.model (the LM) so the same construction applies.
    local = out / "core_basis.npz"
    subprocess.run(["aws", "s3", "cp", BASIS_S3, str(local), "--region", "us-east-1"],
                   check=True)
    z = np.load(local)
    core_keys = {k for k in z.files if "q_proj" in k}
    bmods = {}                                  # canonical key -> the q_proj RankRAdapter
    for n, m in model.model.named_modules():
        if n.endswith("self_attn.q_proj") and hasattr(m, "B"):
            key = canon(f"language_model.{n}.weight").replace(".", "__")
            if key in core_keys:
                bmods[key] = m
    assert len(bmods) == len(core_keys), \
        f"basis/module mismatch: {len(bmods)} modules vs {len(core_keys)} basis keys"
    rep["n_qproj"] = len(bmods)
    core_bases = {}                             # B-param name -> Q (d_in, r_core), IN axis
    pname_by_id = {id(p): n for n, p in model.model.named_parameters()}
    for key, m in bmods.items():
        Q = torch.from_numpy(z[key]).float()
        assert Q.shape[0] == m.B.shape[1], f"{key}: basis dim {Q.shape} vs B {tuple(m.B.shape)}"
        core_bases[pname_by_id[id(m.B)]] = Q
    r_core = int(next(iter(core_bases.values())).shape[1])
    rep["r_core"] = r_core

    # M5 placement arms: constrain B INTO a rank-matched 32-dim subspace (B init = 0 keeps it
    # there): m5_incore = span(core); m5_rand = Haar 32-dim; m5_perp = Haar 32-dim drawn INSIDE
    # the core's complement. Per-matrix seeds.
    mode = "into" if a.arm.startswith("m5_") else "out"
    if a.arm in ("core", "m5_incore"):
        bases = core_bases
    elif a.arm == "random":
        bases = PC.random_bases_like(core_bases, RAND_SEED0)
    elif a.arm == "m5_rand":
        bases = PC.random_bases_like(core_bases, 20260807)
    elif a.arm == "m5_perp":
        bases = {}
        for i, (pn, Q) in enumerate(sorted(core_bases.items())):
            Rm = PC.haar_basis(Q.shape[0], Q.shape[1], 20260808 + i)
            bases[pn] = torch.linalg.qr(Rm - Q @ (Q.T @ Rm))[0]
    else:
        bases = None
    proj = PC.GradProjector(bases, mode=mode) if bases is not None else None

    # ---- P5: projector sanity on synthetic updates (exact removal / exact preservation) ------
    if proj is not None:
        pname, Q = next(iter(sorted(bases.items())))
        G_in = torch.randn(RANK, Q.shape[0]) @ (Q @ Q.T)       # rows inside span(Q)
        G_perp = torch.randn(RANK, Q.shape[0])
        G_perp = G_perp - (G_perp @ Q) @ Q.T                   # rows orthogonal to span(Q)
        fn = PC.project_out_grad if mode == "out" else PC.project_into_grad
        killed, kept = (G_in, G_perp) if mode == "out" else (G_perp, G_in)
        r_k = fn(killed, Q)
        assert float(r_k.norm()) <= 1e-4 * float(killed.norm()), \
            f"P5 FAIL ({mode}): removed component survives ({float(r_k.norm()):.2e})"
        r_p = fn(kept, Q)
        assert float((r_p - kept).norm()) <= 1e-4 * float(kept.norm()), \
            f"P5 FAIL ({mode}): kept component altered"
        rep["p5"] = {"mode": mode, "removed_residual": float(r_k.norm()),
                     "kept_drift": float((r_p - kept).norm()), "param": pname}
        if a.arm == "m5_perp":                                 # placement really ⊥ core
            for pn2, Qp in bases.items():
                ov = float((core_bases[pn2].T @ Qp).abs().max())
                assert ov <= 1e-5, f"P5 FAIL: m5_perp basis overlaps core ({ov:.2e}) at {pn2}"
        print(f"[M4] P5 projector sanity OK ({a.arm}, mode={mode})", flush=True)
    _w()

    # ---- DOME online state (dome arm only) ---------------------------------------------------
    dome, prev, deltas, hooks = {}, {}, {}, []
    if a.arm == "dome":
        for i, (key, m) in enumerate(sorted(bmods.items())):
            dome[key] = PC.DomeEstimator(int(m.B.numel()), DOME_K, seed=DOME_SEED0 + i,
                                         device=device)
            prev[key], deltas[key] = None, []

            def _cap(p, _key=key):
                g = p.grad.detach().float().reshape(-1)
                base = prev[_key] if prev[_key] is not None else torch.zeros_like(g)
                deltas[_key].append(g - base)
                prev[_key] = g.clone()

            hooks.append(m.B.register_post_accumulate_grad_hook(_cap))
        rep["dome"] = {"k": DOME_K, "n_matrices": len(dome),
                      "note": "update-then-filter; samples = 4 micro-batch grad deltas/step"}

    def grad_hook():
        if proj is not None:
            proj.apply(model.model)
        elif a.arm == "dome":
            for key, m in bmods.items():
                S = torch.stack(deltas[key])                   # (4, numel) on CPU
                dome[key].update(S)
                m.B.grad.copy_(dome[key].filter(m.B.grad))
                deltas[key].clear()
                prev[key] = None

    # ---- data (sav_cell verbatim) ------------------------------------------------------------
    ds = load_dataset(TE.MMLU_REPO, "all")
    pool = [dict(r) for r in ds["auxiliary_train"]]
    ded = R.dedup_against_eval(pool, hashes)
    sub = R.ladder_subset(ded["kept"], N_TRAIN)
    examples = [R.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]
    rep["dedup"] = {"n_in": ded["n_in"], "n_dropped": ded["n_dropped"]}

    # ---- periodic trajectory eval ------------------------------------------------------------
    def traj_eval(step):
        model.eval()
        side = tok.padding_side
        ev = TE.mmlu_choice_logits(model, tok, device, item_filter=traj_pred)
        tok.padding_side = side
        model.train()
        assert ev["per_item_hash"] == traj_hashes, f"step {step}: trajectory item order drift"
        S = s_text(ev["per_item"], b_traj, f_traj)
        row = {"step": step, "S": S, "acc_pct": ev["acc_pct"], "t_s": round(time.time() - t0)}
        rep["trajectory"].append(row)
        _w()
        print(f"[M4] {cell_id} step {step}: S_traj={S:.4f} acc={ev['acc_pct']:.2f}", flush=True)

    eval_every = 10 if a.smoke else EVAL_EVERY

    def step_hook(step):
        if step % eval_every == 0:
            traj_eval(step)

    traj_eval(0)
    max_steps = 30 if a.smoke else None
    tr = R.train_adapter_qa(model, tok, device, examples, lr=fr["lr"], epochs=fr["epochs"],
                            batch_size=fr["batch"], warmup_ratio=fr["warmup"], seed=a.seed,
                            max_steps=max_steps, micro_batch=2,
                            grad_hook=grad_hook if a.arm != "baseline" else None,
                            step_hook=step_hook)
    rep["train"] = tr
    for h in hooks:
        h.remove()
    _w()

    # ---- endpoint: full confirmation split + collateral --------------------------------------
    model.eval()
    ev = TE.mmlu_choice_logits(model, tok, device,
                               n_per_subject=(2 if a.smoke else None), item_filter=conf_pred)
    if not a.smoke:
        assert ev["per_item_hash"] == conf_hashes, "endpoint item order != controls — HALT"
        S_end = s_text(ev["per_item"], b_conf, f_conf)
        Sb = boot_s_text(ev["per_item"], b_conf, f_conf)
    else:
        bd = dict(zip(hashes, zip(b_pi, f_pi)))
        pairs = [bd[h] for h in ev["per_item_hash"]]
        S_end = s_text(ev["per_item"], [p[0] for p in pairs], [p[1] for p in pairs])
        Sb = (S_end, float("nan"), float("nan"), 0)
    ppl = None if a.smoke else TE.clean_text_perplexity(model, tok, device,
                                                       max_tokens=60_000)["ppl"]
    rep["endpoint"] = {"S": S_end, "S_boot": {"mean": Sb[0], "lo95": Sb[1], "hi95": Sb[2]},
                      "acc_pct": ev["acc_pct"], "n_items": ev["n"], "wikitext_ppl": ppl,
                      "per_item": ev["per_item"], "per_item_hash": ev["per_item_hash"]}

    # ---- P4 anchor (baseline arm only, full runs only) ---------------------------------------
    if a.arm == "baseline" and not a.smoke:
        bl = out / "banked_cell.json"
        subprocess.run(["aws", "s3", "cp", BANKED_CELL_S3.format(seed=a.seed), str(bl),
                        "--region", "us-east-1"], check=True)
        bk = json.loads(bl.read_text())["eval"]
        within = bool(bk["S_boot"]["lo95"] <= S_end <= bk["S_boot"]["hi95"])
        rep["p4"] = {"banked_S": bk["S_recover"],
                     "banked_ci": [bk["S_boot"]["lo95"], bk["S_boot"]["hi95"]],
                     "own_S": S_end, "within_banked_ci": within}
        print(f"[M4] P4 anchor: own S={S_end:.4f} vs banked {bk['S_recover']:.4f} "
              f"[{bk['S_boot']['lo95']:.4f},{bk['S_boot']['hi95']:.4f}] within={within}",
              flush=True)

    rep["elapsed_s"] = time.time() - t0
    _w()
    print(f"PRAGNANZ_M4_OK {cell_id} S_end={S_end:.4f}", flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)     # GESTALT D7
