"""W11 Leg-B SUPPLEMENT (non-verdict-altering refinements; critic-recommended).

The registered unblind verdict (H_already-readable / diffuse) STANDS. Two refinements, on the
SAME frozen SVD/h̄ + sealed confirmation half, for a complete result doc:

  (1) Knee re-grid — the registered K_GRID jumps 128→512→1152; the true knee (50% of all-ablate
      structured drop) lies in [128, 512]. Add k ∈ {192,256,320,384,448} structured top-k to pin it.
  (2) Bottom-k basis check — ablate the LOW-variance directions Q=U[:,-k:] at k∈{8,64,128}. If
      bottom-k drops ≈ top-k drops ⇒ the diffuseness is basis-agnostic (not an SVD-ordering
      artifact); if top-k ≫ bottom-k ⇒ the top singular directions do carry more (weak concentration).

Reuses the frozen projector_svd.pt; asserts controls_ok + item-hash alignment. Fail-loud.

    python -m infra.aws_pg.jobs.w11_legB_supplement --out <dir> \
        --svd-s3 s3://pg-lineage-769063704696/results/w11_legB_controls_v3/out/projector_svd.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys

EXTRA_K = (192, 256, 320, 384, 448)
BOTTOM_K = (8, 64, 128)
BATCH = 16


def _ensure_pycocoevalcap():
    if importlib.util.find_spec("pycocoevalcap") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocoevalcap"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--svd-s3", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()
    _ensure_pycocoevalcap()

    import torch
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    from infra.pg import projector_ablation as PA
    from infra.pg.entities import resolve_submodule_paths

    assert torch.cuda.is_available(), "Leg-B supplement needs GPU."
    device = "cuda"

    svd_local = out / "projector_svd.pt"
    subprocess.run(["aws", "s3", "cp", args.svd_s3, str(svd_local), "--region", "us-east-1"], check=True)
    frozen = torch.load(svd_local, map_location=device)
    assert frozen.get("controls_ok") is True, "frozen controls_ok=False — do not run."
    U = frozen["U"].to(device).to(torch.float32)
    hbar = frozen["hbar"].to(device).to(torch.float32)
    rank = int(frozen["rank"])
    split_seed = int(frozen["split_seed"]); load_seed = int(frozen["load_seed"])
    n_total = int(frozen["n_total"]); mix_repo = frozen["mix_repo"]
    item_hashes_frozen = frozen["item_hashes"]; splits_frozen = frozen["splits"]

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        mix_repo, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa").eval()
    proc = PaliGemmaProcessor.from_pretrained(mix_repo)
    paths = resolve_submodule_paths(model)
    proj_module, _, _ = PA.get_projector(model, paths)

    per_task = {}
    for task in PA.build_tasks():
        rows = PA.load_eval_items(task, n_total, seed=load_seed)
        assert [PA._content_hash(task, r) for r in rows] == item_hashes_frozen[task.name], \
            f"{task.name}: item-hash mismatch vs frozen."
        conf_rows = [rows[i] for i in splits_frozen[task.name]["conf_idx"]]
        base = PA.eval_task(model, proc, task, conf_rows, device, batch_size=BATCH)["score"]

        extra = {}
        for k in EXTRA_K:
            sc = PA.eval_task(model, proc, task, conf_rows, device,
                              hook_fn=PA.make_ablation_hook(U[:, :k], hbar),
                              proj_module=proj_module, batch_size=BATCH)["score"]
            extra[k] = {"structured": sc, "structured_drop": base - sc}
        bottom = {}
        for k in BOTTOM_K:
            sc = PA.eval_task(model, proc, task, conf_rows, device,
                              hook_fn=PA.make_ablation_hook(U[:, rank - k:], hbar),
                              proj_module=proj_module, batch_size=BATCH)["score"]
            bottom[k] = {"bottom_structured": sc, "bottom_drop": base - sc}
        per_task[task.name] = {"baseline": base, "knee_regrid": extra, "bottom_k": bottom}
        print(f"[W11-B-SUP] {task.name}: base={base:.4f} "
              f"regrid={ {k: round(v['structured_drop'],3) for k,v in extra.items()} } "
              f"bottomk={ {k: round(v['bottom_drop'],3) for k,v in bottom.items()} }")

    result = {"torch": torch.__version__, "mix_repo": mix_repo, "svd_s3": args.svd_s3,
              "extra_k": list(EXTRA_K), "bottom_k": list(BOTTOM_K), "rank": rank,
              "note": "non-verdict-altering refinement of the registered Leg-B unblind "
                      "(H_already-readable stands); pins the knee + tests basis-specificity.",
              "per_task": per_task}
    (out / "legB_supplement.json").write_text(json.dumps(result, indent=2, default=str))
    print("[W11-B-SUP] done.")


if __name__ == "__main__":
    main()
