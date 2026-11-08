"""HUMBOLDT — pre-freeze corpus + roster reach probe. CPU, no GPU, no model weights loaded.

§9 step 2 ("reuse-ledger verification against the tree + weight-free dry-runs"). Phase B's corpus
manifest is frozen at `humboldt-freeze`, so it must be RESOLVED before the freeze — freezing a
dataset id that does not load would freeze an unexecutable plan.

Probes, and writes out/corpus_resolution.json:
  * each §3/B1 corpus slot, over its candidate list, first-that-loads wins (the house
    battery_tasks._survey_candidates pattern: enumeration is reported, never hidden);
  * tokenizer parity between PG1 and gemma-2b (the cross-model token-alignment key);
  * π0 re-shell reachability: repo file listing + the LACUNA §4.1 key counts, WITHOUT downloading
    the multi-GB weights.

Raises only if a slot has NO usable candidate.

  pg run infra.aws_pg.jobs.humboldt_corpus_probe --type r7i.2xlarge --no-weights \
      --s3-prefix results/humboldt_probe --budget-min 60
"""
import argparse
import json
import pathlib

from infra.pg import humboldt_act as A
from infra.pg import humboldt_core as H


def _probe_slot(slot, cands):
    """Survey every candidate; the FIRST usable one wins. Enumeration is reported, never hidden
    (the house battery_tasks._survey_candidates pattern). Never raises — main() asserts once, after
    every slot has been surveyed, so one gated dataset does not hide the state of the others."""
    from datasets import load_dataset
    report, chosen = [], None
    for c in cands:
        entry = dict(c)
        try:
            cfg = c["config"].format(lang=c["langs"][0]) if slot == "parallel" else c["config"]
            ds = (load_dataset(c["id"], cfg, split=c["split"], streaming=True) if cfg
                  else load_dataset(c["id"], split=c["split"], streaming=True))
            row = next(iter(ds))
            entry["row_keys"] = sorted(row.keys())
            field = c["field"] if c["field"] in row else next(
                (f for f in A.FIELD_ALIASES if f in row), None)
            entry["resolved_field"] = field
            usable = field is not None
            if slot == "vqav2":
                img = row.get(c["image_field"])
                entry["image_inline"] = bool(img is not None and hasattr(img, "size"))
                entry["id_present"] = bool(c.get("id_field") and c["id_field"] in row)
                usable = usable and entry["image_inline"]     # a missing id falls back to row index
            entry["sample"] = {k: str(v)[:100] for k, v in row.items() if k != "image"}
            entry["usable"] = bool(usable)
            if usable and chosen is None:
                chosen = entry
        except Exception as e:      # noqa: BLE001 — probe enumeration: report availability
            entry["usable"] = False
            entry["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        report.append(entry)
    return {"chosen": chosen, "candidates": report}


def _probe_parallel_langs(chosen):
    """Every language of the chosen parallel corpus must load, and all must have the same length or
    the row-index pairing is not a pairing."""
    from datasets import load_dataset
    per = {}
    for lang in chosen["langs"]:
        d = load_dataset(chosen["id"], chosen["config"].format(lang=lang), split=chosen["split"])
        per[lang] = {"n": len(d), "keys": sorted(d.column_names)}
    ns = {k: v["n"] for k, v in per.items()}
    assert len(set(ns.values())) == 1, f"parallel languages have different lengths: {ns}"
    return per


def _probe_pi0():
    """π0 re-shell reachability: repo listing + the safetensors HEADER key counts against the
    LACUNA §4.1 gate. `hf_hub_download` fetches whole files (there is no header-range API in
    huggingface_hub), so the weights DO land in the HF cache; only their headers are parsed here."""
    import json as _json
    import struct
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    from infra.pg import lacuna_shell as LS

    repo = LS.POLICY_REPOS["pi0"]
    files = list_repo_files(repo)
    sfs = [f for f in files if f.endswith(".safetensors")]
    assert sfs, f"{repo}: no safetensors"
    keys = []
    for f in sfs:
        p = hf_hub_download(repo, f)
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = _json.loads(fh.read(n))
        keys += [k for k in hdr if k != "__metadata__"]
    pali = [k[len(LS.PALI_PREFIX):] for k in keys if k.startswith(LS.PALI_PREFIX)]
    canon = [LS.canon(k) for k in pali]
    counts = {"vision": sum(1 for k in canon if k.startswith("vision_tower")),
              "projector": sum(1 for k in canon if k.startswith("multi_modal_projector")),
              "language_model": sum(1 for k in canon if k.startswith("language_model")),
              "total_keys": len(keys)}
    ok = (counts["vision"] == LS.N_VISION_KEYS and counts["projector"] == LS.N_PROJ_KEYS
          and counts["language_model"] == LS.N_LM_KEYS)
    return {"repo": repo, "sha": HfApi().model_info(repo).sha, "counts": counts,
            "expected": {"vision": LS.N_VISION_KEYS, "projector": LS.N_PROJ_KEYS,
                         "language_model": LS.N_LM_KEYS},
            "key_gate_ok": bool(ok),
            "note": ("safetensors headers parsed for the key counts; the files themselves are "
                     "downloaded to the HF cache by hf_hub_download — no tensor is loaded into "
                     "memory and no model is constructed")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default="humboldt_probe")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    H.hf_setup()                       # FIRST — XET off + gated-repo auth (§0.2.7)
    from huggingface_hub import HfApi
    from transformers import AutoProcessor, AutoTokenizer
    from infra.pg import text_eval as TE

    api = HfApi()
    res = {"run_id": a.run_id, "commit_sha": H.commit_sha(), "slots": {}}
    for slot, cands in A.CORPUS_CANDIDATES.items():
        print(f"[probe] {slot} …", flush=True)
        res["slots"][slot] = _probe_slot(slot, cands)
        (out / "corpus_resolution.json").write_text(json.dumps(res, indent=2, default=str))
    unresolved = [k for k, v in res["slots"].items() if v["chosen"] is None]
    assert not unresolved, (f"corpus slots with NO usable candidate: {unresolved}\n"
                            + json.dumps({k: res["slots"][k]["candidates"] for k in unresolved},
                                         indent=2, default=str))
    res["parallel_languages"] = _probe_parallel_langs(res["slots"]["parallel"]["chosen"])

    pg1_sha = api.model_info(A.PG1_REPO).sha
    g1_sha = api.model_info(A.GEMMA1_REPO).sha
    tok_pg1 = AutoProcessor.from_pretrained(A.PG1_REPO, revision=pg1_sha).tokenizer
    tok_g1 = AutoTokenizer.from_pretrained(A.GEMMA1_REPO, revision=g1_sha)
    res["tokenizer_parity"] = TE.tokenizer_parity(tok_pg1, tok_g1)
    res["shas"] = {"pg1": pg1_sha, "gemma-2b": g1_sha}
    res["roster_shas"] = {mid: api.model_info(repo).sha for mid, repo, kind in A.ROSTER
                          if kind != "policy_reshell"}
    res["pi0"] = _probe_pi0()

    # a text-corpus build dry-run on the PG1 tokenizer: proves the window/pairing logic runs on the
    # REAL tokenizer before the freeze (weight-free — no model is loaded).
    corp = A.build_text_corpora(tok_pg1, {k: v["chosen"] for k, v in res["slots"].items()},
                               n_tokens=4000, window=256, long_doc_tokens=1024)
    res["corpus_dryrun"] = {cid: {"n_windows": len(rows),
                                  "token_classes": sorted({r["token_class"] for r in rows}),
                                  "content_hash": A.content_hash([r["text"][:64] for r in rows])}
                            for cid, rows in corp.items()}
    (out / "corpus_resolution.json").write_text(json.dumps(res, indent=2, default=str))
    H.s3_sync(out, f"humboldt/{a.run_id}")
    print("=" * 78)
    print("HUMBOLDT_PROBE_OK " + json.dumps(
        {"chosen": {k: v["chosen"]["id"] for k, v in res["slots"].items()},
         "pi0_key_gate": res["pi0"]["key_gate_ok"],
         "tokenizer_parity": res["tokenizer_parity"]["match"],
         "corpus_dryrun": {k: v["n_windows"] for k, v in res["corpus_dryrun"].items()}},
        default=str))


if __name__ == "__main__":
    main()
