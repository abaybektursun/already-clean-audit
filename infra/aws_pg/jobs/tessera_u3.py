"""TESSERA §0 freeze-blocking checklist, item 1 — close the PLENUM U3 residual.

WHAT IS BEING ADJUDICATED
-------------------------
PLENUM §4.3 reports that U3's LM is byte-identical to PG1 (`rel_frob = 0.000000`) after 4096
LIBERO steps, and reads that as a statement about the BUDGET: "neither arm consumed anything at
this budget". The competing reading is that the executed recipe sent **no gradient to the VLM at
all** — a CONFIG property, which would make the budget sentence wrong rather than merely weak.

WHY THE BANKED CENSUS CANNOT SETTLE IT (the spec's "one census read" assumption is false)
-----------------------------------------------------------------------------------------
TESSERA §0 says this is "one census read + one sentence; no new compute", on the theory that the
banked census already shows the action expert moving while the LM sits at zero. It does not:

  * `plenum_census.py:332` censuses the expert with `_absolute_census`, which is a census of the
    matrices THEMSELVES (there is no pre-install reference for a compartment PG1 does not have).
    Its `n_nonzero` is literally `len(rows)` (`plenum_census.py:102`) — the matrix COUNT. So
    `expert n_nonzero=132` for u3 carries no information about whether training moved the expert.
  * The banked `lacuna_units/U3` census is a single endpoint. No census over the film checkpoints
    exists, so "the LM sits at zero AT EVERY STEP" has never been measured.

So the adjudication needs exactly one CPU job over already-banked film checkpoints. No GPU, no
training, no new data. This is recorded as TESSERA pre-freeze amendment A1.

THE MEASUREMENT
---------------
For each film unit, every banked checkpoint is differenced against the EARLIEST banked checkpoint
of the same unit, split into the two compartments that the recipe can dissociate:

  vlm     — every key under `paligemma_with_expert.paligemma.` (the PG1 half: LM, vision,
            projector, embedding). This is the compartment whose movement the budget sentence is
            about.
  expert  — everything else (the gemma_expert decoder + action/state projections + time MLP).
            This is the compartment the action loss trains directly.

Cross-step differencing needs no external base, so PG1 never has to be downloaded and the two
compartments are measured through IDENTICAL code — a difference between them cannot be a
difference between two instruments.

FROZEN DECISION RULES (registered here, in code, BEFORE the job is run)
-----------------------------------------------------------------------
Every branch below can fire; the two VOID branches are what make the verdict falsifiable.

  V1 VOID-film       every delta of every half of every unit is exactly 0
                     => the checkpoints are copies of one another; the film measures nothing and
                        no reading of U3 is available from it.
  V2 VOID-control    U3c's vlm half shows NO movement at any step
                     => the instrument cannot see VLM movement, since PLENUM measured u3c's LM
                        moved 0.35% at the endpoint. Any U3 zero would then be uninformative.
  D  DEGENERATE      U3's expert half also shows no movement at any step
                     => the training run moved nothing anywhere; that is a broken run, NOT a
                        statement about which compartment received gradient.
  C  CONFIG          U3's vlm delta is exactly 0 at every step AND U3's expert delta is nonzero
                     => gradient reached the expert and not the VLM. The executed recipe froze the
                        VLM; PLENUM's "statement about the budget" sentence is wrong as written.
  B  BUDGET          U3's vlm delta is nonzero at any step
                     => the VLM did receive gradient and simply moved little (or moved and
                        returned). PLENUM's sentence stands and only its magnitude is at issue.

Order matters: V1 and V2 are checked BEFORE C, so a dead instrument can never be read as a
finding. This is the §5-N-style discipline applied to an adjudication instead of a contrast.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_u3 --type r7i.2xlarge --no-weights \
      --args "--units U3,U3c" --s3-prefix results/tessera_u3 --budget-min 180

No try/except (CLAUDE.md): a missing checkpoint or a key-set mismatch must fail loud.
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
PALI_PREFIX = "paligemma_with_expert.paligemma."


def film_steps(unit, seed=0):
    """Banked checkpoint steps for a film unit, in numeric order."""
    r = subprocess.run(["aws", "s3", "ls", f"{S3_FILM}/{unit}/seed{seed}/"],
                       check=True, capture_output=True, text=True)
    steps = sorted(int(ln.split()[-1].strip("/")) for ln in r.stdout.splitlines()
                   if ln.strip().endswith("/"))
    assert steps, f"{unit}: no banked film checkpoints under {S3_FILM}/{unit}/seed{seed}/"
    return steps


def fetch(unit, step, seed, workdir):
    """Sync one checkpoint's safetensors to local disk and return the shard paths."""
    d = pathlib.Path(workdir) / f"{unit}_{step:08d}"
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync",
                    f"{S3_FILM}/{unit}/seed{seed}/{step:08d}/pretrained_model", str(d),
                    "--exclude", "*", "--include", "*.safetensors",
                    # the sync progress meter is what makes a pg job.log 16 MB of noise and
                    # buries the traceback that matters
                    "--only-show-errors"], check=True)
    shards = sorted(d.glob("*.safetensors"))
    assert shards, f"{unit}@{step}: no safetensors synced"
    return d, shards


def open_shards(shards):
    """{canonical_key: (handle, stored_key)} across a checkpoint's shards, opened lazily.

    safetensors' `safe_open` reads one tensor at a time, so a 2 x 8 GB comparison never holds two
    whole checkpoints in RAM.

    Keys are canonicalised through `lacuna_shell.canon` (REUSED) before use. A film checkpoint is
    written by `policy.save_pretrained`, which nests every compartment under `model.`, so the raw
    keys carry NO `paligemma_with_expert.paligemma.` prefix and a prefix split on them finds an
    empty VLM compartment. That is not a cosmetic difference: an empty compartment reads as "did
    not move", which is the exact CONFIG verdict this job exists to test, arrived at from a string
    mismatch. The assertion in `compare` is what caught it; canonicalising is what fixes it.
    """
    from safetensors import safe_open
    from infra.pg.lacuna_shell import canon
    index, handles = {}, []
    for p in shards:
        f = safe_open(str(p), framework="pt", device="cpu")
        handles.append(f)
        for k in f.keys():
            c = canon(k)
            assert c not in index, f"canonicalisation collision on {c!r} (from {k!r}) in {p.name}"
            index[c] = (f, k)
    return index, handles


def compare(ref_index, cur_index, tag):
    """Per-compartment movement of `cur` away from `ref`, over the keys they share.

    Reports BOTH the all-tensor movement count (norms and biases are where a tiny gradient shows up
    first, so restricting to matrices could hide a live path) and the matrix-only Frobenius
    profile.
    """
    import numpy as np
    import torch
    assert set(ref_index) == set(cur_index), (
        f"{tag}: checkpoint key sets differ "
        f"(ref-only {sorted(set(ref_index) - set(cur_index))[:3]}, "
        f"cur-only {sorted(set(cur_index) - set(ref_index))[:3]})")
    out = {}
    for half in ("vlm", "expert"):
        keys = sorted(k for k in ref_index
                      if (k.startswith(PALI_PREFIX)) == (half == "vlm"))
        assert keys, (f"{tag}: compartment {half!r} is empty — the key split is wrong. "
                      f"Sample canonical keys: {sorted(ref_index)[:4]}")
        n_moved, mats, maxabs = 0, [], 0.0
        for k in keys:
            rf, rk = ref_index[k]
            cf, ck = cur_index[k]
            a = rf.get_tensor(rk).to(torch.float32)
            b = cf.get_tensor(ck).to(torch.float32)
            assert a.shape == b.shape, f"{tag}: {k} shape {tuple(a.shape)} vs {tuple(b.shape)}"
            d = (b - a)
            m = float(d.abs().max())
            if m > 0:
                n_moved += 1
            maxabs = max(maxabs, m)
            if a.ndim == 2:
                na = float(torch.linalg.norm(a))
                mats.append(float(torch.linalg.norm(d)) / na if na > 0 else float("nan"))
            del a, b, d
        out[half] = {
            "n_tensors": len(keys),
            "n_tensors_moved": n_moved,
            "max_abs_delta": maxabs,
            "n_matrices": len(mats),
            "median_rel_frob": float(np.median(mats)) if mats else None,
            "max_rel_frob": float(np.max(mats)) if mats else None,
            "moved": bool(n_moved > 0),
        }
    return out


def adjudicate(per_unit):
    """The frozen rules of the module docstring, applied mechanically."""
    def moved(unit, half):
        return any(s["halves"][half]["moved"] for s in per_unit[unit]["steps"])

    any_movement = any(s["halves"][h]["moved"]
                       for u in per_unit for s in per_unit[u]["steps"] for h in ("vlm", "expert"))
    if not any_movement:
        return {"verdict": "VOID-film", "rule": "V1",
                "why": "every delta of every compartment of every unit is exactly zero — the "
                       "banked film checkpoints are copies and measure nothing"}
    if "U3c" in per_unit and not moved("U3c", "vlm"):
        return {"verdict": "VOID-control", "rule": "V2",
                "why": "the U3c positive control shows no VLM movement at any step, yet PLENUM "
                       "measured u3c's LM moved 0.35% at the endpoint — this instrument cannot "
                       "see VLM movement, so U3's zero is uninformative"}
    if not moved("U3", "expert"):
        return {"verdict": "DEGENERATE-TRAINING", "rule": "D",
                "why": "U3's expert did not move either — the run moved nothing anywhere, which "
                       "is a broken run rather than a statement about gradient routing"}
    if not moved("U3", "vlm"):
        return {"verdict": "CONFIG-NOT-BUDGET", "rule": "C",
                "why": "U3's expert moved at every measured step while its VLM half is bit-exactly "
                       "unchanged — gradient reached the expert and not the VLM, so the executed "
                       "recipe froze the VLM. PLENUM §4.3's 'statement about the budget' is wrong "
                       "as written: no budget would have moved a compartment that receives no "
                       "gradient"}
    return {"verdict": "BUDGET-STANDS", "rule": "B",
            "why": "U3's VLM half did move — the compartment receives gradient and simply moved "
                   "little, so PLENUM's budget reading stands and only its magnitude is at issue"}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", default="U3,U3c")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workdir", default="/tmp/tessera_u3")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    per_unit = {}
    for unit in a.units.split(","):
        steps = film_steps(unit, a.seed)
        ref_step = steps[0]
        ref_dir, ref_shards = fetch(unit, ref_step, a.seed, a.workdir)
        ref_index, _rh = open_shards(ref_shards)
        n_vlm = sum(1 for k in ref_index if k.startswith(PALI_PREFIX))
        # fail BEFORE syncing seven more checkpoints if the namespace is not what the split assumes
        assert 0 < n_vlm < len(ref_index), (
            f"{unit}@{ref_step}: canonical key split gives {n_vlm} VLM of {len(ref_index)} keys — "
            f"one compartment is empty. Sample: {sorted(ref_index)[:4]}")
        rec = {"unit": unit, "ref_step": ref_step, "all_steps": steps,
               "n_keys": len(ref_index), "n_keys_vlm": n_vlm,
               "n_keys_expert": len(ref_index) - n_vlm, "steps": []}
        for s in steps[1:]:
            cur_dir, cur_shards = fetch(unit, s, a.seed, a.workdir)
            cur_index, _ch = open_shards(cur_shards)
            halves = compare(ref_index, cur_index, f"{unit}@{s}vs{ref_step}")
            rec["steps"].append({"step": s, "halves": halves})
            print(f"[{unit} {s:>6} vs {ref_step}] "
                  f"vlm moved={halves['vlm']['moved']} "
                  f"({halves['vlm']['n_tensors_moved']}/{halves['vlm']['n_tensors']} tensors, "
                  f"max|Δ|={halves['vlm']['max_abs_delta']:.3e}) | "
                  f"expert moved={halves['expert']['moved']} "
                  f"({halves['expert']['n_tensors_moved']}/{halves['expert']['n_tensors']}, "
                  f"max|Δ|={halves['expert']['max_abs_delta']:.3e})", flush=True)
            del cur_index, _ch
            shutil.rmtree(cur_dir)
        per_unit[unit] = rec
        del ref_index, _rh
        shutil.rmtree(ref_dir)

    verdict = adjudicate(per_unit)
    metrics = {
        "object": "TESSERA §0 checklist item 1 — the PLENUM U3 budget-vs-config residual",
        "label": "ADJUDICATION under rules frozen in the module docstring before the run",
        "amendment": "A1 — the banked expert census is ABSOLUTE (plenum_census.py:332), so its "
                     "n_nonzero is the matrix count, not a movement count; and no film-checkpoint "
                     "census existed. The spec's 'no new compute' was wrong; this is one CPU job "
                     "over already-banked checkpoints.",
        "reference": "each unit is differenced against its OWN earliest banked checkpoint, so no "
                     "external base is needed and both compartments pass through identical code",
        "units": per_unit,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(verdict, indent=1), flush=True)


if __name__ == "__main__":
    main()
