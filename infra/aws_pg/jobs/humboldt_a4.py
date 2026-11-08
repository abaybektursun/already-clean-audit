"""HUMBOLDT A4 — external-relative projection, phylogenetics-lite (spec §3/A4). CPU.

Frozen mechanical selection rule (selection happens BEFORE any projection is computed):
  1. query HF for models tagged with base_model {google/gemma-2b, google/paligemma-3b-pt-224}
     across the three tag forms HF emits (base_model:, base_model:finetune:, base_model:merge:);
  2. dedup, sort by downloads DESCENDING;
  3. walk the list in order; accept a model iff all 126 decoder matrices key-align with its base at
     the right shapes; take the first N_ACCEPT;
  4. record the FULL selection log, accepts and rejects with reasons.

Per accepted model, per (type, layer): ‖ΔW_ext U‖²_F / ‖ΔW_ext‖²_F for every bank AND every twin
(the paired chance column is what makes a row interpretable later), plus ΔW_ext's full singular
spectrum.

Smoke (§3/A4): a HOUSE fine-tune HELD OUT of the consensus must reproduce the avoidance pattern and
its Haar-rotated copy must not. The held-out consensus is built leave-one-out from the cached fp16
q_proj ΔW scratch — fp16 is ample for an order-of-magnitude smoke separation and is flagged as such;
it is never used to define a shipped bank (journal C3).

Pivot (§3/A4): if fewer than 3 external models key-align, record however many do and journal.

  pg run infra.aws_pg.jobs.humboldt_a4 --type r7i.2xlarge --no-weights \
      --s3-prefix results/humboldt_a4 --budget-min 300
"""
import argparse
import json
import pathlib
import subprocess

from infra.pg import humboldt_core as H
from infra.pg import humboldt_smoke as S
from infra.pg import humboldt_weights as W

BASE_TAGS = ("google/gemma-2b", "google/paligemma-3b-pt-224")
TAG_FORMS = ("base_model:{}", "base_model:finetune:{}", "base_model:merge:{}")
N_ACCEPT = 8
N_EXAMINE = 60                 # hard cap on candidates walked, so the arm cannot run away
CACHED_SCRATCH_S3 = "s3://pg-lineage-769063704696/results/invcore_full2/out/scratch"


def _cfg_prefilter(cfg):
    """Cheap architecture check from the config alone — avoids downloading multi-GB junk. Returns
    (ok, reason)."""
    # A quantised checkpoint has the right config geometry but stores `qweight`/scales instead of
    # dense matrices, so it would be accepted, downloaded, and then dropped at load. Reject on the
    # config, which is free.
    if getattr(cfg, "quantization_config", None):
        q = getattr(cfg, "quantization_config", None)
        method = q.get("quant_method") if isinstance(q, dict) else getattr(q, "quant_method", "?")
        return False, f"quantised checkpoint (quant_method={method}) — no dense ΔW to project"
    tc = getattr(cfg, "text_config", cfg)
    want = {"hidden_size": 2048, "num_hidden_layers": H.N_LAYERS, "intermediate_size": 16384,
            "head_dim": 256, "num_key_value_heads": 1, "num_attention_heads": 8}
    for k, v in want.items():
        got = getattr(tc, k, None)
        if got != v:
            return False, f"config.{k}={got} != {v}"
    return True, "config matches PG1 decoder geometry"


def _load_any_decoder(repo, sha, model_type):
    """{(type, layer): W fp32} for either a PaliGemma checkpoint or a plain causal LM."""
    import torch
    if model_type == "paligemma":
        sd = W._load_decoder(repo, sha)
        return {(t, L): sd[W.matrix_key(t, L)].to(torch.float32)
                for t in H.TYPES for L in range(H.N_LAYERS)}
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(repo, revision=sha, torch_dtype=torch.float32,
                                             device_map="cpu", attn_implementation="eager")
    sd = m.model.state_dict()
    out = {}
    for t in H.TYPES:
        for L in range(H.N_LAYERS):
            k = W.matrix_key(t, L)
            assert k in sd, f"{repo}: missing {k}"
            out[(t, L)] = sd[k].detach().to(torch.float32)
    del m
    return out


def _select(api, out, n_accept):
    """The frozen mechanical rule. Returns (accepted, log)."""
    from transformers import AutoConfig
    cands, seen = [], set()
    for tag in BASE_TAGS:
        for form in TAG_FORMS:
            # No `direction=` — huggingface_hub 1.x dropped it. The frozen rule's ordering comes
            # from the explicit client-side sort below, which does not depend on API defaults.
            for m in api.list_models(filter=form.format(tag), sort="downloads", limit=100):
                if m.id in seen:
                    continue
                seen.add(m.id)
                cands.append({"id": m.id, "downloads": int(m.downloads or 0), "base_tag": tag,
                              "tag_form": form.format(tag)})
    cands.sort(key=lambda c: -c["downloads"])
    log, accepted = [], []
    for c in cands[:N_EXAMINE]:
        if len(accepted) >= n_accept:
            log.append({**c, "decision": "not_examined",
                        "reason": f"quota of {n_accept} already filled"})
            continue
        # Candidate INSPECTION is enumeration over arbitrary community repos, so an unreadable
        # config is a rejection reason, not an arm-ending error (the same sanctioned pattern as
        # battery_tasks._survey_candidates). The projection code below is still fail-loud.
        try:
            files = {x.rfilename for x in api.model_info(c["id"], files_metadata=False).siblings}
            if "adapter_config.json" in files:
                log.append({**c, "decision": "reject",
                            "reason": "adapter-only repo (no full weights)", "is_adapter": True})
                continue
            if not any(f.endswith((".safetensors", ".bin")) for f in files):
                log.append({**c, "decision": "reject", "reason": "no weight files"})
                continue
            cfg = AutoConfig.from_pretrained(c["id"])
        except Exception as e:      # noqa: BLE001 — candidate enumeration: report unreadability
            log.append({**c, "decision": "reject",
                        "reason": f"unreadable repo: {type(e).__name__}: {str(e)[:200]}"})
            continue
        ok, why = _cfg_prefilter(cfg)
        rec = {**c, "model_type": cfg.model_type, "is_adapter": False,
               "is_merge": "merge" in c["tag_form"],
               # the checkpoint's storage dtype rides as a flag: a bf16-saved external model carries
               # bf16 rounding into ΔW relative to the fp32 base, which is a property of the
               # checkpoint, not of the projection (§8 "flag columns, never prose-only footnotes").
               "ckpt_dtype": str(getattr(cfg, "torch_dtype", None))}
        if not ok:
            log.append({**rec, "decision": "reject", "reason": why})
            continue
        log.append({**rec, "decision": "accept", "reason": why})
        accepted.append(rec)
    (pathlib.Path(out) / "a4_selection_log.json").write_text(
        json.dumps({"rule": {"base_tags": list(BASE_TAGS), "tag_forms": list(TAG_FORMS),
                             "sort": "downloads desc", "n_accept": n_accept,
                             "n_examine_cap": N_EXAMINE},
                    "n_candidates": len(cands), "log": log,
                    "accepted": [r["id"] for r in accepted]}, indent=2))
    return accepted, log


def _held_out_smoke(out, gates, api, repos, seed):
    """Leave-one-out held-out-fine-tune smoke on q_proj from the cached fp16 scratch."""
    import numpy as np
    cache = H.cache_dir(out, "loo_scratch")
    subprocess.run(["aws", "s3", "sync", CACHED_SCRATCH_S3, str(cache), "--region", "us-east-1",
                    "--exclude", "*", "--include", "q_proj__L0__*"], check=True)
    held = 0                                    # leave out ft index 0 (deterministic, logged)
    G = None
    for ridx in range(len(repos)):
        if ridx == held:
            continue
        A = np.load(cache / f"q_proj__L0__{ridx}.npy").astype(np.float64)
        G = (A.T @ A) / (A ** 2).sum() if G is None else G + (A.T @ A) / (A ** 2).sum()
    V = np.linalg.eigh(G)[1]
    U = V[:, :H.BANK_K]
    dW = np.load(cache / f"q_proj__L0__{held}.npy").astype(np.float64)
    rng = np.random.default_rng(seed)
    R = np.linalg.qr(rng.standard_normal((dW.shape[1], dW.shape[1])))[0]

    def frac(A):
        return float(np.linalg.norm(A @ U) ** 2 / np.linalg.norm(A) ** 2)

    f_held, f_rot = frac(dW), frac(dW @ R)
    chance = H.BANK_K / dW.shape[1]
    sep = f_rot / max(f_held, 1e-300)
    gates.require("G5_A4_held_out_house_ft", sep >= S.G5_SEPARATION,
                  held_out_repo=repos[held], frac_held_out=f_held, frac_haar_rotated=f_rot,
                  chance=chance, separation=sep, required=S.G5_SEPARATION,
                  consensus="23-ft leave-one-out, q_proj L0",
                  fp16_scratch_smoke_only=True,
                  note="fp16 cached ΔW is used HERE ONLY (smoke separation), never for a shipped bank")
    return {"frac_held_out": f_held, "frac_haar_rotated": f_rot, "chance": chance,
            "separation": sep, "held_out_repo": repos[held]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default=",".join(H.TYPES))
    ap.add_argument("--layers", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-id", default="humboldt_a4")
    ap.add_argument("--n-accept", type=int, default=N_ACCEPT)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = ([int(x) for x in a.layers.split(",") if x.strip()] if a.layers.strip()
              else list(range(H.N_LAYERS)))
    if a.smoke:
        types, layers = ("q_proj", "gate_proj"), layers[:2]

    H.hf_setup()                       # FIRST — XET off + gated-repo auth (§0.2.7)
    import numpy as np
    from huggingface_hub import HfApi

    gates = H.Gates(out)
    S.run_all_offline(gates)

    api = HfApi()
    repos = W.ft_cloud(api)
    smoke = _held_out_smoke(out, gates, api, repos, a.seed)

    accepted, log = _select(api, out, a.n_accept)
    print(f"[A4] selected {len(accepted)}/{len(log)} examined: {[r['id'] for r in accepted]}",
          flush=True)

    bank_dir = W.fetch_banks(H.cache_dir(out, "banks"))
    base_cache = {}
    manifest = {"arm": "A4", "run_id": a.run_id, "commit_sha": H.commit_sha(),
                "types": list(types), "layers": layers, "seed": a.seed,
                "selection_rule": {"base_tags": list(BASE_TAGS), "tag_forms": list(TAG_FORMS),
                                   "sort": "downloads desc", "n_accept": a.n_accept},
                "accepted": [r["id"] for r in accepted], "n_examined": len(log),
                "held_out_smoke": smoke,
                "pivot_triggered": bool(len(accepted) < 3)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if len(accepted) < 3:
        print(f"[A4][PIVOT] only {len(accepted)} external models key-align — recording those and "
              f"journaling the count (§3/A4 pivot)", flush=True)

    atlas = H.Atlas(out, arm="A4", run_id=a.run_id)      # bound BEFORE any loop (§0.2.4)

    dropped = []                                         # bound BEFORE any loop (§0.2.4)
    for rec in accepted:
        rid = rec["id"]
        base_tag = rec["base_tag"]
        try:
            sha = api.model_info(rid).sha
            if base_tag not in base_cache:
                b_sha = api.model_info(base_tag).sha
                base_cache[base_tag] = (_load_any_decoder(
                    base_tag, b_sha,
                    "paligemma" if "paligemma" in base_tag else "gemma"), b_sha)
            base_w, base_sha = base_cache[base_tag]
            print(f"[A4] {rid}@{sha[:8]} (base {base_tag})", flush=True)
            ext = _load_any_decoder(rid, sha, rec["model_type"])
        except Exception as e:      # noqa: BLE001 — §9 pivot: a single model drops with a
            # journaled coverage hole, never the whole arm.
            dropped.append({"id": rid, "reason": f"{type(e).__name__}: {str(e)[:300]}"})
            print(f"[A4][DROP] {rid}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            (out / "a4_dropped.json").write_text(json.dumps(dropped, indent=2))
            continue
        for t in types:
            for L in layers:
                dW = (ext[(t, L)].numpy().astype(np.float64)
                      - base_w[(t, L)].numpy().astype(np.float64))
                fro2 = float((dW ** 2).sum())
                fl = {**H.type_flags(t), "ext_base": base_tag, "is_merge": rec["is_merge"],
                      "is_adapter": rec["is_adapter"], "model_type": rec["model_type"],
                      "downloads": rec["downloads"], "dW_fro2": fro2,
                      "ckpt_dtype": rec.get("ckpt_dtype")}
                if fro2 == 0.0:
                    atlas.add("ext_dW_fro2", value=0.0, layer=L, type=t, model_id=rid,
                              model_sha=sha, flags={**fl, "identical_to_base": True})
                    continue
                atlas.add("ext_dW_fro2", value=fro2, layer=L, type=t, model_id=rid, model_sha=sha,
                          flags=fl)
                sv = np.linalg.svd(dW, compute_uv=False)
                atlas.add("ext_dW_singular_spectrum",
                          vector_ref=atlas.vector(f"A4__extsv__{rid.replace('/', '_')}__{t}__L{L}",
                                                  sv),
                          layer=L, type=t, model_id=rid, model_sha=sha, n=int(sv.size), flags=fl)
                for bid in H.BANK_IDS:
                    U = W.load_bank(bank_dir, t, L, bid)
                    frac = float(np.linalg.norm(dW @ U) ** 2 / fro2)
                    atlas.add("ext_dW_bank_energy_fraction", value=frac, layer=L, type=t,
                              bank_id=bid, model_id=rid, model_sha=sha,
                              flags={**fl, "chance": H.BANK_K / H.IN_DIM[t]})
                del dW
            print(f"[A4]   {t}: {len(layers)} layers ({atlas.n_rows} rows)", flush=True)
        del ext
        W._delete_cache(rid)
        H.s3_sync(out, f"humboldt/{a.run_id}")

    atlas.close()
    rep = atlas.to_parquet()
    (out / "a4_summary.json").write_text(json.dumps(
        {"manifest": manifest, "atlas": rep, "gates": gates.gates,
         "selection_log_len": len(log), "dropped_models": dropped,
         "n_recorded": len(accepted) - len(dropped)}, indent=2, default=str))
    H.s3_sync(out, f"humboldt/{a.run_id}")
    print("=" * 78)
    print(f"HUMBOLDT_A4_OK rows={rep['n_rows']} accepted={len(accepted)} "
          f"dropped={len(dropped)} examined={len(log)} pivot={manifest['pivot_triggered']}")


if __name__ == "__main__":
    main()
