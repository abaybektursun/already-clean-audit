"""TESSERA §0.1b — WHICH artifact is the map's `u3`: the construction, or the trained endpoint?

`tessera_u3` measured that U3's VLM half MOVES at every film step (547/604 tensors by step 1024,
max|Δ| growing to 5.8e-3 by 4096). PLENUM's census of the SAME named unit found its whole VLM half
bit-identical to PG1 — 291 matrices at exactly zero. Both cannot describe the same weights.

The hypothesis this job tests is that `s3://…/weights/lacuna_units/U3` is the PRE-TRAINING
CONSTRUCTION (PG1's 604 backbone keys installed into a π0 shell) rather than the 4096-step
endpoint. If so, its `rel_frob = 0.000000` is true BY CONSTRUCTION and cannot support PLENUM
§4.3's inference that U3 "loses no faculty for co-training to spare".

THE THREE-WAY CHECK, and the four outcomes it can produce (frozen before the run):

  banked := lacuna_units/U3 (the artifact the map censused)
  f512   := plenum_film/U3/seed0/00000512 (the earliest trained checkpoint)
  f4096  := plenum_film/U3/seed0/00004096 (the endpoint the budget sentence is about)
  PG1    := google/paligemma-3b-pt-224 (the base every ΔW is measured from)

  CONSTRUCTION   banked ≡ PG1 (VLM bit-exact) AND f512 ≢ PG1
                 => the banked unit is the initialisation; the map's u3 weight cells describe the
                    pre-training artifact, and the endpoint is f4096, which moved.
  ENDPOINT       banked ≡ f4096
                 => the banked unit IS the trained endpoint; then the film and the census disagree
                    about the same bytes and the CONTRADICTION is in the instrument, not the naming.
  THIRD-ARTIFACT banked ≢ PG1 and banked ≢ f4096 and banked ≢ f512
                 => the banked unit is neither; whatever the map censused has no place in this
                    trajectory and must be identified before any u3 cell is read.
  DEGENERATE     f512 ≡ PG1 (VLM bit-exact)
                 => the trained trajectory never left PG1 at its first checkpoint either, which
                    would contradict tessera_u3's own measurement and voids this comparison.

Only the VLM half is compared: PG1 has no action expert, so the expert has no PG1 counterpart.
The comparison is on CANONICAL keys (`lacuna_shell.canon`) because the three artifacts use three
different namespaces — the banked unit nests under `model.`, the film checkpoints likewise, and
PG1 uses the plain HF PaliGemma namespace.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_u3ck --type r7i.2xlarge --no-weights \
      --s3-prefix results/tessera_u3ck --budget-min 120

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import time

S3_FILM = "s3://pg-lineage-769063704696/weights/plenum_film"
S3_UNITS = "s3://pg-lineage-769063704696/weights/lacuna_units"
PG1_REPO = "google/paligemma-3b-pt-224"
PALI_PREFIX = "paligemma_with_expert.paligemma."


def _sync(uri, dest):
    d = pathlib.Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", uri, str(d), "--exclude", "*",
                    "--include", "*.safetensors", "--only-show-errors"], check=True)
    shards = sorted(d.glob("*.safetensors"))
    assert shards, f"no safetensors under {uri}"
    return d, shards


def _policy_vlm(shards):
    """{canonical VLM key without the policy prefix: tensor} from a policy-shaped checkpoint."""
    from safetensors.torch import load_file
    from infra.pg.lacuna_shell import canon
    sd = {}
    for p in shards:
        sd.update(load_file(str(p)))
    c = {canon(k): v for k, v in sd.items()}
    vlm = {k[len(PALI_PREFIX):]: v for k, v in c.items() if k.startswith(PALI_PREFIX)}
    assert vlm, f"no {PALI_PREFIX!r} compartment; sample keys {sorted(c)[:4]}"
    return vlm


def _pg1_vlm():
    """PG1's own VLM state dict on the same canonical keys."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file
    from infra.pg.lacuna_shell import canon
    sd = {}
    for f in [x for x in list_repo_files(PG1_REPO) if x.endswith(".safetensors")]:
        sd.update(load_file(hf_hub_download(PG1_REPO, f)))
    return {canon(k): v for k, v in sd.items()}


def compare(a_sd, b_sd, tag):
    """Movement of b away from a over their SHARED canonical keys."""
    import numpy as np
    import torch
    shared = sorted(set(a_sd) & set(b_sd))
    assert shared, (f"{tag}: no shared keys "
                    f"(a-sample {sorted(a_sd)[:3]}, b-sample {sorted(b_sd)[:3]})")
    n_moved, rels, maxabs, shape_diff = 0, [], 0.0, []
    for k in shared:
        x = a_sd[k].detach().to(torch.float32)
        y = b_sd[k].detach().to(torch.float32)
        if x.shape != y.shape:
            # A shape difference is NOT weight movement, and conflating the two is how the first
            # run of this job produced THIRD-ARTIFACT off a single key. PaliGemma's embedding table
            # is padded to 257216 rows while a policy's tied head is 257152 (`lacuna_shell`
            # AMENDMENT-2) — an architectural vocabulary-padding difference that exists between
            # EVERY PaliGemma checkpoint and EVERY policy checkpoint, carrying no information about
            # whether training moved anything. It is recorded separately and excluded from the
            # value comparison, never silently dropped.
            shape_diff.append({"key": k, "a_shape": list(x.shape), "b_shape": list(y.shape)})
            continue
        d = y - x
        m = float(d.abs().max())
        if m > 0:
            n_moved += 1
        maxabs = max(maxabs, m)
        if x.ndim == 2:
            nx = float(torch.linalg.norm(x))
            rels.append(float(torch.linalg.norm(d)) / nx if nx > 0 else float("nan"))
    fin = [r for r in rels if r == r]
    return {"tag": tag, "n_shared": len(shared), "n_only_a": len(set(a_sd) - set(b_sd)),
            "n_only_b": len(set(b_sd) - set(a_sd)),
            "n_compared": len(shared) - len(shape_diff),
            "n_shape_diff": len(shape_diff), "shape_diff": shape_diff,
            "n_moved": n_moved, "max_abs_delta": maxabs,
            "median_rel_frob": float(np.median(fin)) if fin else None,
            # "identical" = every COMPARABLE tensor is bit-equal. Shape-differing keys are reported
            # in `shape_diff` and adjudicated separately by the reader.
            "identical": bool(n_moved == 0)}


def adjudicate(c):
    """The four outcomes of the module docstring, in the order that lets a dead comparison VOID.

    AMENDED 2026-08-01, BEFORE the TESSERA freeze and after the first run of this pre-freeze
    checklist job. v1 counted a SHAPE difference as movement, so the single padded-embedding key
    (257216 vs 257152) made `banked_vs_pg1.identical` false even though every one of the 602
    comparable tensors was bit-equal (max|Δ| = 0.0, median rel-Frobenius = 0.0) — and the rule
    returned THIRD-ARTIFACT off that one key. The amendment separates shape differences from value
    movement; both readings are reported in the result doc, and the v1 verdict is not deleted.
    """
    if c["f512_vs_pg1"]["identical"]:
        return {"verdict": "DEGENERATE", "why":
                "the first trained checkpoint is bit-identical to PG1, which contradicts "
                "tessera_u3's measurement that the VLM moves — this comparison is void"}
    if c["banked_vs_pg1"]["identical"] and not c["f512_vs_pg1"]["identical"]:
        return {"verdict": "CONSTRUCTION", "why":
                "the banked unit is bit-identical to PG1 while the earliest TRAINED checkpoint is "
                "not — the map censused the pre-training construction, so u3's rel_frob = 0 is "
                "true by construction and cannot support a statement about what training consumed"}
    if c["banked_vs_f4096"]["identical"]:
        return {"verdict": "ENDPOINT", "why":
                "the banked unit IS the 4096-step endpoint — the census and the film describe the "
                "same bytes and the contradiction is in the instrument, not the naming"}
    return {"verdict": "THIRD-ARTIFACT", "why":
            "the banked unit matches neither PG1 nor either film checkpoint — whatever the map "
            "censused must be identified before any u3 weight cell is read"}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default="/tmp/tessera_u3ck")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()

    print("[load] PG1", flush=True)
    pg1 = _pg1_vlm()
    print(f"[load] PG1 canonical keys: {len(pg1)}", flush=True)

    arts = {}
    for name, uri in (("banked", f"{S3_UNITS}/U3"),
                      ("f512", f"{S3_FILM}/U3/seed0/00000512/pretrained_model"),
                      ("f4096", f"{S3_FILM}/U3/seed0/00004096/pretrained_model")):
        d, shards = _sync(uri, f"{a.workdir}/{name}")
        arts[name] = _policy_vlm(shards)
        print(f"[load] {name}: {len(arts[name])} VLM keys from {uri}", flush=True)
        shutil.rmtree(d)

    cmp_ = {
        "banked_vs_pg1": compare(pg1, arts["banked"], "banked vs PG1"),
        "f512_vs_pg1": compare(pg1, arts["f512"], "film@512 vs PG1"),
        "f4096_vs_pg1": compare(pg1, arts["f4096"], "film@4096 vs PG1"),
        "banked_vs_f512": compare(arts["f512"], arts["banked"], "banked vs film@512"),
        "banked_vs_f4096": compare(arts["f4096"], arts["banked"], "banked vs film@4096"),
    }
    for k, v in cmp_.items():
        print(f"[cmp] {k:<18s} shared={v['n_shared']:<4d} moved={v['n_moved']:<4d} "
              f"max|Δ|={v['max_abs_delta']:.4e} med_rel_frob={v['median_rel_frob']} "
              f"identical={v['identical']}", flush=True)

    verdict = adjudicate(cmp_)
    metrics = {
        "object": "TESSERA §0.1b — which artifact is the map's u3",
        "label": "ADJUDICATION under four outcomes frozen in the module docstring before the run",
        "pg1_repo": PG1_REPO,
        "comparisons": cmp_,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(verdict, indent=1), flush=True)


if __name__ == "__main__":
    main()
