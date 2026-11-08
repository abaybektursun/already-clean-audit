"""OWN-EMBED feasibility gate (RESEARCH_INDEX §5.1, step 1.5) — CPU, --no-weights.

Decides whether the π0/π0.5 own-embed readout is BUILDABLE before the prereg specifies it
(the charter-retraction lesson: don't design the estimand ahead of the artifact). Pulls ONLY
the backbone embed_tokens (+ lm_head) tensors from HF for four checkpoints and runs the
sharpened checklist on the base-vocab rows (0..255,999):

  1. embed_tokens present in the LM backbone? tied to lm_head? vocab dim (257k ext vs 256k base)?
  2. bitwise-distinct from base gemma-2b embed on base-vocab rows? (precondition for own-vs-base signal)
  3. bitwise-distinct from PG1 embed on base-vocab rows? (π0-era co-adaptation vs PG1-inherited)

Pre-stated fork (journal 2026-06-17_ownembed_journal.md, logged blind):
  (a) present + distinct-from-base → buildable (design prereg + PG1-vs-π0 attribution arm)
  (b) present but ≈ base       → informative null (no embed co-adaptation → VLAT deletion stands)
  (c) absent                   → own-embed not constructible → reframe

  pg run infra.aws_pg.jobs.ownembed_extract_gate --type r7i.2xlarge --no-weights --budget-min 40
"""
from __future__ import annotations

import argparse
import json
import pathlib

BASE_VOCAB = 256_000   # gemma-2b (Gemma-1) text vocab; PG1/π0/π0.5 extend it (~257,152)

# repo -> role. PG1/base are the anchors; pi0/pi05 are the subjects.
REPOS = {
    "base": "google/gemma-2b",                # Gemma-1, 18L
    "PG1":  "google/paligemma-3b-pt-224",
    "pi0":  "lerobot/pi0_base",
    "pi05": "lerobot/pi05_base",
}


def _backbone(k: str) -> bool:
    """True iff key belongs to the LM backbone (not vision / not the action expert)."""
    return not any(m in k for m in ("vision_tower", "gemma_expert", "action_expert"))


def _pick_embed_key(keys):
    cands = [k for k in keys if "embed_tokens" in k and _backbone(k)]
    if not cands:
        return None
    # prefer the language_model-scoped key when several match
    lm = [k for k in cands if "language_model" in k]
    return sorted(lm or cands, key=len)[0]


def _pick_head_keys(keys):
    return sorted(k for k in keys if "lm_head" in k and _backbone(k))


def load_embed_head(repo):
    """Return {present, embed_key, vocab, hidden, tied, embed(tensor|None), head_key}.
    Uses the safetensors index for a partial (shard-only) download when available."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors import safe_open

    files = [f for f in list_repo_files(repo) if f.endswith(".safetensors")]
    idx = [f for f in list_repo_files(repo) if f.endswith(".safetensors.index.json")]
    assert files, f"{repo}: no .safetensors files"

    weight_map = None
    all_keys = []
    if idx:
        wm = json.loads(pathlib.Path(hf_hub_download(repo, idx[0])).read_text())["weight_map"]
        weight_map = wm
        all_keys = list(wm)
    else:
        # no index: must open each file (single-file checkpoints, e.g. pi0 model.safetensors)
        key_file = {}
        for f in files:
            p = hf_hub_download(repo, f)
            with safe_open(p, framework="pt") as st:
                for k in st.keys():
                    all_keys.append(k); key_file[k] = p
        weight_map = key_file   # key -> local path here

    ek = _pick_embed_key(all_keys)
    hk = _pick_head_keys(all_keys)

    def _get(key):
        if idx:
            path = hf_hub_download(repo, weight_map[key])
        else:
            path = weight_map[key]
        with safe_open(path, framework="pt") as st:
            return st.get_tensor(key)

    import torch
    # The TEXT EMBEDDING is the tied [vocab, hidden] weight. It survives under EITHER
    # embed_tokens OR — when embed_tokens was dropped at export (openpi #669) — under lm_head
    # (Gemma ties them: lm_head.weight IS embed_tokens.weight). So recover the tied weight from
    # whichever key is present; only genuinely ABSENT iff neither exists.
    if ek is not None:
        emb = _get(ek)
        head = _get(hk[0]) if hk else None
        source = "embed_tokens"; emb_key = ek
        tied = (not hk) or (head is not None and head.shape == emb.shape
                            and torch.equal(head.float(), emb.float()))
    elif hk:
        emb = _get(hk[0])                       # tied-weight recovered from lm_head
        source = "lm_head_tied"; emb_key = hk[0]; tied = True
    else:
        return {"repo": repo, "present": False, "source": "absent", "embed_key": None,
                "head_keys": hk, "tied": None, "vocab": None, "hidden": None, "embed": None}

    return {"repo": repo, "present": True, "source": source, "embed_key": emb_key,
            "head_keys": hk, "tied": bool(tied),
            "vocab": int(emb.shape[0]), "hidden": int(emb.shape[1]), "embed": emb}


def compare(a, b, nrows=BASE_VOCAB):
    """Distinctness of two embeddings on the shared base-vocab rows [0:nrows], fp32."""
    import torch
    assert a.shape[1] == b.shape[1], f"hidden mismatch {a.shape} vs {b.shape}"
    n = min(nrows, a.shape[0], b.shape[0])
    a32 = a[:n].float(); b32 = b[:n].float()
    na = a32.norm().item()
    return {"rows": n,
            "bitwise_equal": bool(torch.equal(a32, b32)),
            "rel_frob": ((a32 - b32).norm().item() / na) if na > 0 else float("nan"),
            "max_abs": (a32 - b32).abs().max().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    inv = {}
    for name, repo in REPOS.items():
        info = load_embed_head(repo)
        print(f"[OWNEMBED] {name:5s} {repo}: present={info['present']} source={info.get('source')} "
              f"tied={info['tied']} vocab={info['vocab']} hidden={info['hidden']} embed_key={info['embed_key']}")
        inv[name] = info

    assert inv["base"]["present"], "base gemma-2b embed not found — cannot anchor"
    assert inv["PG1"]["present"], "PG1 embed not found — cannot anchor attribution"
    base_e, pg1_e = inv["base"]["embed"], inv["PG1"]["embed"]

    result = {"base_vocab_rows": BASE_VOCAB, "inventory": {
        k: {kk: v.get(kk) for kk in ("repo", "present", "source", "tied", "vocab", "hidden", "embed_key", "head_keys")}
        for k, v in inv.items()}}

    # anchor sanity: PG1 vs base (the VLM hop already moves the embed — context for attribution)
    result["pg1_vs_base"] = compare(pg1_e, base_e)
    print(f"[OWNEMBED] PG1 vs base (context): rel_frob={result['pg1_vs_base']['rel_frob']:.4g} "
          f"bitwise_equal={result['pg1_vs_base']['bitwise_equal']}")

    result["checks"] = {}
    for name in ("pi0", "pi05"):
        info = inv[name]
        if not info["present"]:
            result["checks"][name] = {"present": False, "branch": "c",
                                      "note": "embed absent from backbone → own-embed not constructible"}
            print(f"[OWNEMBED] {name}: ABSENT → branch (c)")
            continue
        vs_base = compare(info["embed"], base_e)
        vs_pg1 = compare(info["embed"], pg1_e)
        # branch: (b) if ≈base on base-vocab rows; else (a). tied is fine (noted, not (c)).
        near_base = vs_base["bitwise_equal"] or vs_base["rel_frob"] < 1e-4
        branch = "b" if near_base else "a"
        result["checks"][name] = {
            "present": True, "source": info["source"], "tied": info["tied"], "vocab": info["vocab"],
            "vs_base": vs_base, "vs_pg1": vs_pg1, "branch": branch,
            "pi0era_moved_embed": (not (vs_pg1["bitwise_equal"] or vs_pg1["rel_frob"] < 1e-4)),
        }
        print(f"[OWNEMBED] {name}: source={info['source']} tied={info['tied']} vocab={info['vocab']} | "
              f"vs_base rel_frob={vs_base['rel_frob']:.4g} (bitwise={vs_base['bitwise_equal']}) | "
              f"vs_PG1 rel_frob={vs_pg1['rel_frob']:.4g} (bitwise={vs_pg1['bitwise_equal']}) "
              f"→ branch ({branch}); pi0era_moved={result['checks'][name]['pi0era_moved_embed']}")

    (out / "ownembed_gate.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"[OWNEMBED] DONE → branches: " +
          ", ".join(f"{n}={result['checks'][n].get('branch')}" for n in ('pi0', 'pi05')))


if __name__ == "__main__":
    main()
