"""W11 §9 Leg-B UNBLIND (step 5: read the concentration DECISION once, on the sealed half).

Load PG2-mix + the FROZEN projector_svd.pt (U,S,h̄) from controls. Reconstruct the SAME eval
items + split deterministically; use the SEALED CONFIRMATION half only. For each task, at every
frozen k in K_GRID:

  structured  Q = U[:, :k]                  → structured_drop = baseline − structured_score
  random      k random orthonormal × 3      → random_drop     = baseline − mean_seeds(score)

Per-item delta_i = random_score_i − structured_score_i  (>0 ⇒ structured ablation hurts MORE than
a rank-matched random ablation ⇒ info CONCENTRATED in the top-k bridge directions). Paired
bootstrap over confirmation items; one-sided p = P(boot Δ ≤ 0); BH-FDR q=0.05 across the
(task × k) family. Knee = smallest k reaching 50% of the all-ablate (k=1152) structured drop.

Frozen §6B rule:
  small-k structured ≫ random (FDR-sig, CI-separated) → CONCENTRATED → leans H_bridge-installs
  structured ≈ random until near-full k                → DIFFUSE re-basis → leans H_already-readable
  structured drop ≤ random floor everywhere            → H_null
Honest scope (§6B): this is NECESSITY of the bridge directions, not an already-readable proof.

    python -m infra.aws_pg.jobs.w11_legB_unblind --out <dir> \
        --svd-s3 s3://pg-lineage-769063704696/results/w11_legB_controls/out/projector_svd.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys

SMALL_K = 16            # "small-k" boundary for the concentrated verdict (§6B)
KNEE_FRAC = 0.5
N_RANDOM_SEEDS = 3
BATCH = 16


def _ensure_pycocoevalcap():
    if importlib.util.find_spec("pycocoevalcap") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocoevalcap"], check=True)


def _bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg: return a boolean 'significant' list aligned to pvals."""
    import numpy as np
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    sig = np.zeros(m, bool)
    if passed.any():
        kmax = np.where(passed)[0].max()
        sig[order[: kmax + 1]] = True
    return sig.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--svd-s3", required=True, help="frozen projector_svd.pt from controls")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()
    _ensure_pycocoevalcap()

    import numpy as np
    import torch
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    from infra.pg import projector_ablation as PA
    from infra.pg.entities import ENTITIES, resolve_submodule_paths

    assert torch.cuda.is_available(), "Leg-B unblind needs GPU."
    device = "cuda"

    # --- frozen instrument (instance-role cli; provenance) ---
    svd_local = out / "projector_svd.pt"
    subprocess.run(["aws", "s3", "cp", args.svd_s3, str(svd_local), "--region", "us-east-1"], check=True)
    frozen = torch.load(svd_local, map_location=device)
    assert frozen.get("controls_ok") is True, \
        "Leg-B controls did NOT pass (controls_ok=False in frozen SVD) — fix harness, do not unblind."
    U = frozen["U"].to(device).to(torch.float32)
    hbar = frozen["hbar"].to(device).to(torch.float32)
    out_dim = int(frozen["out_dim"]); rank = int(frozen["rank"])
    k_grid = [k for k in frozen["k_grid"] if k <= rank]
    split_seed = int(frozen["split_seed"]); load_seed = int(frozen["load_seed"])
    n_total = int(frozen["n_total"]); mix_repo = frozen["mix_repo"]
    item_hashes_frozen = frozen["item_hashes"]; splits_frozen = frozen["splits"]
    print(f"[W11-B-UNB] frozen SVD rank={rank} out_dim={out_dim} k_grid={k_grid} "
          f"pt→mix proj drift={frozen.get('proj_pt_mix_rel_frob')}")

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        mix_repo, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa").eval()
    proc = PaliGemmaProcessor.from_pretrained(mix_repo)
    paths = resolve_submodule_paths(model)
    proj_module, _, _ = PA.get_projector(model, paths)

    tasks = PA.build_tasks()
    per_task = {}
    fam_keys, fam_p = [], []     # BH-FDR family: per-item (task,k) one-sided p
    for ti, task in enumerate(tasks):
        rows = PA.load_eval_items(task, n_total, seed=load_seed)
        # alignment: the reloaded items MUST match the frozen content hashes (same dataset/order),
        # else the frozen confirmation split would index different rows (critic determinism fix).
        assert [PA._content_hash(task, r) for r in rows] == item_hashes_frozen[task.name], (
            f"{task.name}: reloaded eval items differ from frozen controls (content-hash mismatch).")
        conf_idx = splits_frozen[task.name]["conf_idx"]
        conf_rows = [rows[i] for i in conf_idx]
        base = PA.eval_task(model, proc, task, conf_rows, device, batch_size=BATCH)
        base_score = base["score"]
        base_pi = np.asarray(base["per_item"], float) if not base["corpus"] else None

        curve = {}
        for k in k_grid:
            Qk = U[:, :k]
            st = PA.eval_task(model, proc, task, conf_rows, device,
                              hook_fn=PA.make_ablation_hook(Qk, hbar), proj_module=proj_module,
                              batch_size=BATCH)
            rand_scores, rand_pis = [], []
            for s in range(N_RANDOM_SEEDS):
                Q = PA.random_basis(out_dim, k, seed=1000 * k + s, device=device, dtype=torch.float32)
                rr = PA.eval_task(model, proc, task, conf_rows, device,
                                  hook_fn=PA.make_ablation_hook(Q, hbar), proj_module=proj_module,
                                  batch_size=BATCH)
                rand_scores.append(rr["score"])
                if not rr["corpus"]:
                    rand_pis.append(np.asarray(rr["per_item"], float))
            rand_mean = float(np.mean(rand_scores))
            entry = {"structured": st["score"], "random_mean": rand_mean,
                     "random_seeds": rand_scores,
                     "structured_drop": base_score - st["score"],
                     "random_drop": base_score - rand_mean}
            if base_pi is not None:
                st_pi = np.asarray(st["per_item"], float)
                rand_pi_mean = np.mean(rand_pis, axis=0)
                delta = rand_pi_mean - st_pi               # >0 ⇒ structured hurts more
                # distinct resampling per (task,k) → independent draws across the BH-FDR family
                # (critic: a single shared seed correlates all family p-values).
                rng = np.random.default_rng(1337 + ti * 100_000 + k)
                n = delta.size
                boot = np.array([delta[rng.integers(0, n, n)].mean() for _ in range(args.n_boot)])
                p_one = float((boot <= 0).mean())
                entry.update({"delta_mean": float(delta.mean()),
                              "delta_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                              "p_onesided": p_one})
                fam_keys.append((task.name, k)); fam_p.append(p_one)
            curve[k] = entry

        drops = {k: curve[k]["structured_drop"] for k in k_grid}
        max_drop = max(drops.values()) if drops else 0.0
        knee = next((k for k in k_grid if max_drop > 0 and drops[k] >= KNEE_FRAC * max_drop), None)
        per_task[task.name] = {"corpus": base["corpus"], "n_conf": base["n"],
                               "baseline": base_score, "curve": curve, "knee_k": knee,
                               "max_structured_drop": max_drop}
        print(f"[W11-B-UNB] {task.name}: base={base_score:.4f} knee_k={knee} "
              f"struct_drop@k1={drops.get(1):.4f} @all={drops.get(rank if rank in drops else k_grid[-1]):.4f}")

    # BH-FDR across the per-item (task,k) family
    sig = _bh_fdr(fam_p, q=0.05)
    sig_map = {f"{t}@k{k}": bool(s) for (t, k), s in zip(fam_keys, sig)}

    # verdict (§6B): concentrated if a small-k structured ablation is FDR-sig AND the knee is small
    per_item_tasks = [t for t in per_task if not per_task[t]["corpus"]]
    concentrated = []
    for t in per_item_tasks:
        knee = per_task[t]["knee_k"]
        small_sig = any(sig_map.get(f"{t}@k{k}", False) for k in [k for k in per_task[t]["curve"] if k <= SMALL_K])
        concentrated.append(bool(small_sig and knee is not None and knee <= SMALL_K))
    n_conc = sum(concentrated)
    any_sig = any(sig)
    if not any_sig:
        verdict = "H_null"
    elif n_conc >= (len(per_item_tasks) + 1) // 2:
        verdict = "H_bridge-installs (concentrated)"
    else:
        verdict = "H_already-readable (diffuse re-basis)"

    result = {
        "torch": torch.__version__, "mix_repo": mix_repo, "svd_s3": args.svd_s3,
        "k_grid": k_grid, "rank": rank, "small_k_boundary": SMALL_K, "knee_frac": KNEE_FRAC,
        "per_task": per_task,
        "fdr": {"family": [f"{t}@k{k}" for (t, k) in fam_keys], "pvals": fam_p, "significant": sig_map},
        "concentration": {"per_item_tasks": per_item_tasks, "concentrated": concentrated,
                          "n_concentrated": n_conc},
        "VERDICT": verdict,
        "scope_note": ("necessity of the bridge directions, NOT an already-readable proof; a diffuse "
                       "bridge is consistent-with (not proof-of) SigLIP being 'born linguistic'."),
    }
    (out / "legB_unblind.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W11-B-UNB] VERDICT={verdict} (n_concentrated={n_conc}/{len(per_item_tasks)}, any_FDR_sig={any_sig})")


if __name__ == "__main__":
    main()
