"""CO-X2 preflight — build and freeze a Llama-3.2-3B delta cloud. CHARTER, no verdict.

CO-X2 needs a 24-member Llama-3.2-3B fine-tune cloud, and none exists. This job builds one the same
way the PG1 manifest was built and freezes it by content hash, so the arm that consumes it can
assert the cloud has not moved.

IT REUSES TWO INSTRUMENTS THIS PROGRAM ALREADY VALIDATED, rather than inventing selection rules:

  * CO-F9's LINEAGE-SCOPED enumeration — `base_model:finetune:<base>`, exhaustive. Not a name
    search: `search="Llama-3.2-3B"` returns repos that MENTION the base and caps at its own limit
    (journal D100). The relation matters too — of Llama-3.2-3B's declared derivatives only a
    minority are finetunes; adapters, quantizations and merges have no full-weight delta.
  * CO-X1a's SHAPE VALIDATION from safetensors headers — key alignment AND per-key shapes, read
    from `get_safetensors_metadata`, no weights moved. On smolVLA this rejected a repo whose key
    set aligned while 136 shared keys had mismatched shapes (journal D99); a delta against that
    repo is undefined.

SELECTION IS DETERMINISTIC AND STATED IN ADVANCE: survivors sorted by repo id, at most
MAX_PER_OWNER per owner, first 24 taken. Not "the 24 largest" or "the 24 most downloaded" — a
popularity-ordered cloud would confound the census with whatever makes a repo popular, and a rule
chosen after seeing the candidates is not a rule. Every rejection is recorded with its reason.

THE ONE-PER-OWNER CAP IS THE LOAD-BEARING PART, and it was added after the first build of this
manifest showed why. Sorting 518 lineage-declared finetunes by repo id and taking the first 24 gave
a cloud that was 9/24 one owner's `llama32-3b-redo-{dpo,kto}_*` sweep, 3/24 another's
`subject{2,3,4}-test1`, and 2/24 a third's V1/V2 — 14 of 24 within-sweep siblings from three
owners. Members of one sweep share initialisation, data and hyperparameters, so their deltas are
near-duplicates. X2's row-Gram treatment asks what structure is shared across INDEPENDENT
fine-tunes; a consensus dominated by one lab's KTO sweep would answer a different question and
would answer it with inflated agreement. The cap is a pre-freeze amendment to a CHARTER preflight —
no X2 measurement exists yet, so nothing is being reinterpreted after the fact.

    python3 -m infra.aws_pg.jobs.copernicus_x2_manifest --out out
"""
import argparse
import collections
import hashlib
import json
import pathlib

from infra.pg import copernicus_core as C
from infra.pg import humboldt_core as H

ARM = "CO-X2-preflight"
QUEUE_ITEM = "53+82"
REGISTER = "CHARTER (frozen input artifact, no verdict)"
BASE = "meta-llama/Llama-3.2-3B"
N_FT = 24
MAX_PER_OWNER = 1
ADAPTER_FILES = ("adapter_config.json",)


def param_shapes(repo, revision=None):
    """{name: shape} from safetensors HEADERS — metadata only, no weights.

    `get_safetensors_metadata`, not the index file: single-shard repos publish no index, and the
    index carries a weight_map with NO SHAPES (CO-X1a, journal D99)."""
    from huggingface_hub import get_safetensors_metadata
    try:
        meta = get_safetensors_metadata(repo, revision=revision)
    except Exception:
        return None
    shapes = {}
    for fmeta in meta.files_metadata.values():
        for name, info in fmeta.tensors.items():
            shapes[name] = tuple(int(x) for x in info.shape)
    return shapes or None


def is_adapter(api, repo):
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(repo_id=repo, filename=ADAPTER_FILES[0])
    except Exception:
        return False
    return True


def manifest_hash(repos, shas, base_sha):
    """The same body and ordering `copernicus_a1_banks.load_manifest` hashes, so the consuming arm
    can verify this cloud with the loader it already uses."""
    body = json.dumps({"repos": repos, "shas": shas, "base_sha": base_sha},
                      sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--n-ft", type=int, default=N_FT)
    ap.add_argument("--max-per-owner", type=int, default=MAX_PER_OWNER)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    from huggingface_hub import HfApi
    api = HfApi()
    base_sha = api.model_info(a.base).sha
    base_shapes = param_shapes(a.base, base_sha)
    assert base_shapes, f"{a.base} exposes no readable safetensors metadata"
    base_keys = set(base_shapes)
    print(f"[x2m] base {a.base}@{base_sha[:8]}: {len(base_keys)} tensors", flush=True)

    rep = {"object": "copernicus_x2_manifest_v1", "arm": ARM, "program": "copernicus",
           "queue_item": QUEUE_ITEM, "register": REGISTER,
           "base_repo": a.base, "base_sha": base_sha, "n_base_keys": len(base_keys),
           "selection_rule": ("lineage-scoped exhaustive enumeration (base_model:finetune), "
                              "adapter-rejected, key-and-shape validated, survivors sorted by "
                              f"repo id, at most {a.max_per_owner} per owner, first N taken"),
           "max_per_owner": a.max_per_owner,
           "why_owner_cap": ("without it the cloud was 14/24 within-sweep siblings from three "
                             "owners (9 from one KTO sweep); sweep members share init, data and "
                             "hyperparameters, so a consensus over them measures one recipe"),
           "provenance": C.provenance(), "utc": C.utc_now(),
           "candidates": [], "rejected": []}

    def save():
        (out / "copernicus_x2_manifest.json").write_text(json.dumps(rep, indent=2, default=float))
    save()

    cands = sorted({m.id for m in api.list_models(
        filter=f"base_model:finetune:{a.base}", limit=None) if m.id != a.base})
    rep["n_candidates"] = len(cands)
    print(f"[x2m] {len(cands)} lineage-declared finetunes", flush=True)
    save()

    survivors, per_owner = [], collections.Counter()
    for i, repo in enumerate(cands):
        if len(survivors) >= a.n_ft:
            break
        row = {"repo": repo}
        owner = repo.split("/")[0]
        if per_owner[owner] >= a.max_per_owner:
            row["reason"] = f"owner quota ({a.max_per_owner}) already filled"
            rep["rejected"].append(row); save(); continue
        if is_adapter(api, repo):
            row["reason"] = "adapter/LoRA repo"
            rep["rejected"].append(row); save(); continue
        shapes = param_shapes(repo)
        if not shapes:
            row["reason"] = "no readable safetensors metadata"
            rep["rejected"].append(row); save(); continue
        shared = set(shapes) & base_keys
        row["n_keys"], row["n_shared"] = len(shapes), len(shared)
        if len(shared) < len(base_keys):
            row["reason"] = f"key set incomplete ({len(shared)}/{len(base_keys)})"
            rep["rejected"].append(row); save(); continue
        bad = [k for k in shared if shapes[k] != base_shapes[k]]
        if bad:
            row["reason"] = f"shape mismatch on {len(bad)} keys"
            row["sample"] = bad[:3]
            rep["rejected"].append(row); save(); continue
        row["sha"] = api.model_info(repo).sha
        per_owner[owner] += 1
        survivors.append(row)
        rep["candidates"].append(row)
        print(f"[x2m] accept {repo}@{row['sha'][:8]} ({len(survivors)}/{a.n_ft}) "
              f"[scanned {i + 1}]", flush=True)
        save()

    assert len(survivors) == a.n_ft, (
        f"only {len(survivors)} of {a.n_ft} survived shape validation over {len(cands)} "
        f"candidates; the cloud cannot be frozen at this size")
    repos = sorted(r["repo"] for r in survivors)
    shas = {r["repo"]: r["sha"] for r in survivors}
    owners = collections.Counter(r.split("/")[0] for r in repos)
    assert max(owners.values()) <= a.max_per_owner, f"owner cap violated: {owners.most_common(3)}"
    rep["n_distinct_owners"] = len(owners)
    h = manifest_hash(repos, shas, base_sha)
    frozen = {"base_repo": a.base, "base_sha": base_sha, "repos": repos, "shas": shas,
              "n_ft": len(repos), "sha256": h, "max_per_owner": a.max_per_owner,
              "n_distinct_owners": len(owners),
              "note": ("Llama-3.2-3B delta cloud for CO-X2; built by "
                       "copernicus_x2_manifest with F9's lineage enumeration and X1a's "
                       "shape validation")}
    (out / "copernicus_llama_delta_manifest.json").write_text(json.dumps(frozen, indent=2))
    rep["frozen_manifest_sha256"] = h
    rep["n_rejected"] = len(rep["rejected"])
    save()
    print(f"[x2m] FROZEN {len(repos)} repos, sha256={h}", flush=True)


if __name__ == "__main__":
    main()
