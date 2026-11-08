"""CO-X1a (item 40) — smolVLA per-repo key-aligned shape validation. CHARTER row, no verdict.

Spec §3 Tier X / §6.3: "CO-X1 is TWO halves with two registers: X1a (shape validation) is CHARTER;
X1b (the census, runs iff >=12 survive) is VERDICT-BEARING." This is the first half only. It
produces a described artifact and a COUNT; it does not score a hypothesis, and it must not be read
as one.

WHAT "SURVIVES" MEANS, frozen here so the count is not a judgement call:
  1. the repo resolves and exposes a safetensors index (weights are never downloaded — this is a
     metadata pass and downloading 200 checkpoints to answer a shape question would be absurd);
  2. its parameter KEY SET matches the base's on the shared keys, and every shared key has an
     IDENTICAL shape — a delta is only defined between aligned matrices;
  3. it is NOT adapter-contaminated. LoRA/PEFT repos carry `adapter_config.json` or expose
     `lora_`/`adapter` keys; their "weights" are a low-rank side-car, so a delta against the base
     would measure the adapter's rank, not the fine-tune's geometry. §3 makes adapter and rank
     flags MANDATORY rather than advisory for exactly this reason.

Every rejected repo is listed WITH its reason. A survivor count with no reject list is a number
nobody can check.

    pg run infra.aws_pg.jobs.copernicus_x1a --args "" --no-weights \
        --type r7i.2xlarge --s3-prefix results/copernicus_x1a --budget-min 60
"""
import argparse
import collections
import json
import pathlib

from infra.pg import copernicus_core as C
from infra.pg import humboldt_core as H

ARM = "CO-X1a"
QUEUE_ITEM = "40"
WAVE = "WF"
BASE_CANDIDATES = ("lerobot/smolvla_base",)
CENSUS_FLOOR = 12          # §6.3: X1b runs iff >=12 survive
SEARCH_TERMS = ("smolvla", "SmolVLA")
MAX_REPOS = 400


def param_shapes(repo):
    """{parameter name -> shape} from the safetensors HEADERS. Metadata only; no weights.

    Uses `get_safetensors_metadata`, not the index file, for two reasons the first attempt at this
    arm found the hard way: `lerobot/smolvla_base` is SINGLE-SHARD and therefore publishes no
    `model.safetensors.index.json` at all, and the index — where it exists — carries only a
    weight_map with NO SHAPES. An arm named "key-aligned shape validation" cannot run off a file
    that has no shapes in it.
    """
    from huggingface_hub import get_safetensors_metadata
    try:
        meta = get_safetensors_metadata(repo)
    except Exception:
        return None
    shapes = {}
    for fmeta in meta.files_metadata.values():
        for name, info in fmeta.tensors.items():
            shapes[name] = tuple(int(x) for x in info.shape)
    return shapes or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-repos", type=int, default=MAX_REPOS)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    H.hf_setup()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()

    exp = C.Expectations(out / "copernicus_expectations.jsonl")
    exp.log(ARM, "x1a_roster_survives",
            "at least 12 smolVLA repos survive key-aligned shape validation, unlocking X1b",
            confidence="low", basis="spec §11 (X1b carries a directional prior, low confidence)",
            family="census")
    at = C.Atlas(out, arm=ARM, run_id="copernicus_x1a", wave=WAVE, queue_item=QUEUE_ITEM,
                 expected_ref=exp.require(ARM), tag=H.DISCOVERY)
    gates = C.Gates(out)

    base = None
    for cand in BASE_CANDIDATES:
        try:
            api.model_info(cand)
            base = cand
            break
        except Exception:
            continue
    assert base is not None, f"no smolVLA base resolved from {BASE_CANDIDATES}"
    base_shapes = param_shapes(base)
    assert base_shapes, f"{base} exposes no readable safetensors metadata"
    base_keys = set(base_shapes)
    base_meta = {"n_params": len(base_keys)}
    print(f"[x1a] base {base}: {len(base_keys)} parameter tensors", flush=True)

    seen = {}
    for term in SEARCH_TERMS:
        for m in api.list_models(search=term, limit=a.max_repos):
            if m.id != base:
                seen[m.id] = m
    print(f"[x1a] {len(seen)} candidate repos", flush=True)

    rows, survivors, reasons = [], [], collections.Counter()
    for i, repo in enumerate(sorted(seen)):
        row = {"repo": repo, "survives": False, "reason": None, "adapter_flag": False,
               "rank": None, "n_keys": None, "n_shared": None, "n_shape_mismatch": None}
        try:
            files = set(api.list_repo_files(repo))
        except Exception as e:
            row["reason"] = f"unreadable: {type(e).__name__}"
            reasons[row["reason"]] += 1
            rows.append(row)
            continue
        # (3) adapter contamination — MANDATORY flag, checked before anything else
        if "adapter_config.json" in files:
            row["adapter_flag"] = True
            try:
                p = hf_hub_download(repo_id=repo, filename="adapter_config.json")
                cfg = json.loads(pathlib.Path(p).read_text())
                row["rank"] = cfg.get("r") or cfg.get("rank")
            except Exception:
                pass
            row["reason"] = "adapter/LoRA repo (delta would measure the adapter's rank)"
            reasons[row["reason"]] += 1
            rows.append(row)
            continue
        shapes = param_shapes(repo)
        if not shapes:
            row["reason"] = "no readable safetensors metadata"
            reasons[row["reason"]] += 1
            rows.append(row)
            continue
        keys = set(shapes)
        shared = keys & base_keys
        row["n_keys"], row["n_shared"] = len(keys), len(shared)
        if any(k.startswith("lora_") or "lora" in k.lower() or "adapter" in k.lower()
               for k in keys):
            row["adapter_flag"] = True
            row["reason"] = "lora/adapter keys present"
            reasons[row["reason"]] += 1
            rows.append(row)
            continue
        if not shared or len(shared) < 0.9 * len(base_keys):
            row["reason"] = f"key set does not align ({len(shared)}/{len(base_keys)} shared)"
            reasons[row["reason"]] += 1
            rows.append(row)
            continue
        # (2) SHAPES, which is what the arm is named for: a delta is only defined between
        # aligned matrices, so one mismatched shared key disqualifies the repo.
        mismatched = [k for k in shared if shapes[k] != base_shapes[k]]
        row["n_shape_mismatch"] = len(mismatched)
        row["shape_mismatch_sample"] = mismatched[:3]
        if mismatched:
            row["reason"] = f"shape mismatch on {len(mismatched)} shared keys"
            reasons[row["reason"]] += 1
            rows.append(row)
            continue
        row["survives"] = True
        survivors.append(repo)
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"[x1a] {i + 1}/{len(seen)} checked, {len(survivors)} survivors so far",
                  flush=True)
            (out / "copernicus_x1a.json").write_text(json.dumps(
                {"partial": True, "rows": rows}, indent=2))

    unlocked = bool(len(survivors) >= CENSUS_FLOOR)
    gates.record("x1a_roster_survives", unlocked, n_candidates=len(seen),
                 n_survivors=len(survivors), floor=CENSUS_FLOOR,
                 reject_reasons=dict(reasons),
                 band=f"X1b (VERDICT-BEARING) runs iff >= {CENSUS_FLOOR} repos survive key-aligned "
                      f"shape validation with no adapter contamination (§6.3)",
                 prediction="census existence predicted present, low confidence (§11)")
    at.add("x1a_survivor_count", value=float(len(survivors)), n=len(seen),
           flags={"survivors": survivors[:50], "reject_reasons": dict(reasons),
                  "register": "CHARTER — described artifact, not a verdict"})

    rep = {"object": "copernicus_x1a_v1", "arm": ARM, "program": C.PROGRAM,
           "queue_item": QUEUE_ITEM, "register": "CHARTER (no verdict; X1b is the scored half)",
           "base": base, "n_base_keys": len(base_keys), "base_metadata": base_meta,
           "n_candidates": len(seen), "n_survivors": len(survivors),
           "census_floor": CENSUS_FLOOR, "x1b_unlocked": unlocked,
           "survivors": survivors, "reject_reasons": dict(reasons), "rows": rows,
           "provenance": C.provenance(), "utc": C.utc_now()}
    (out / "copernicus_x1a.json").write_text(json.dumps(rep, indent=2, default=str))
    at.close()
    at.to_parquet()
    print("=" * 78, flush=True)
    print(f"COPERNICUS_X1A_OK survivors={len(survivors)}/{len(seen)} floor={CENSUS_FLOOR} "
          f"x1b_unlocked={unlocked} reasons={dict(reasons)}", flush=True)


if __name__ == "__main__":
    main()
