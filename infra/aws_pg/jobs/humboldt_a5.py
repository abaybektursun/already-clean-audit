"""HUMBOLDT A5 — closed-form-editor deltas (spec §3/A5, OPTIONAL, hard cap 4h wall-clock).

"If a maintained ROME/MEMIT implementation supports gemma-2b within the cap: apply 10 standard
CounterFact edits, record the edit ΔW projections onto all banks/twins (same schema as A4).
If the cap is hit: skip entirely, journal one line. No adaptation heroics."

This job runs that conditional exactly:
  1. a BOUNDED reach test — can a maintained implementation be installed, and does it ship (or
     generically construct) hyper-parameters for gemma-2b? Evidence recorded either way.
  2. if the reach test passes, apply 10 CounterFact edits and record ΔW projections in A4's schema;
  3. if it fails, record the skip with the evidence that produced it. No adaptation is attempted.

  pg run infra.aws_pg.jobs.humboldt_a5 --type g6e.8xlarge,p5.4xlarge --no-weights \
      --s3-prefix results/humboldt_a5 --budget-min 240
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

from infra.pg import humboldt_core as H
from infra.pg import humboldt_smoke as S
from infra.pg import humboldt_weights as W

# The spec's conditional is "does a maintained ROME/MEMIT implementation SUPPORT gemma-2b". That is
# settled by whether the implementation ships gemma hyper-parameters, which its repository answers
# directly and for free — no install required, and no inference from an install failure. Each
# candidate names the repo paths that would hold them.
EDITOR_CANDIDATES = (
    {"name": "EasyEdit", "repo": "zjunlp/EasyEdit",
     "hparam_paths": ("hparams/ROME", "hparams/MEMIT"),
     "pip": "git+https://github.com/zjunlp/EasyEdit.git", "module": "easyeditor"},
    {"name": "rome-kmeng01", "repo": "kmeng01/rome", "hparam_paths": ("hparams/ROME",),
     "pip": "git+https://github.com/kmeng01/rome.git", "module": "rome"},
    {"name": "memit-kmeng01", "repo": "kmeng01/memit",
     "hparam_paths": ("hparams/ROME", "hparams/MEMIT"),
     "pip": "git+https://github.com/kmeng01/memit.git", "module": "memit"},
    {"name": "easyeditor-pypi", "repo": None, "hparam_paths": (),
     "pip": "easyeditor", "module": "easyeditor"},
)
COUNTERFACT_CANDIDATES = ("azhx/counterfact", "NeelNanda/counterfact-tracing")
N_EDITS = 10
REACH_CAP_S = 3600          # 1h of the 4h cap for the reach test


def _pip(spec, timeout):
    t = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", spec],
                       capture_output=True, text=True, timeout=timeout)
    return {"spec": spec, "rc": r.returncode, "secs": round(time.time() - t, 1),
            "stderr_tail": r.stderr[-400:]}


def _list_repo_dir(repo, path):
    """File names under a GitHub repo directory, via the contents API. Returns None if absent."""
    import json as _json
    import urllib.request
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "humboldt-a5-reach"})
    with urllib.request.urlopen(req, timeout=30) as r:      # noqa: S310 — fixed api.github.com URL
        d = _json.loads(r.read())
    return None if isinstance(d, dict) else [x["name"] for x in d]


def _reach(out, deadline):
    """Bounded reach test (§3/A5). Two stages, cheapest-decisive first:

      1. Does ANY maintained implementation ship gemma hyper-parameters? Answered from each repo's
         hparams directory over the GitHub contents API — free, and it settles the spec's actual
         conditional rather than inferring it from whether pip happened to succeed.
      2. Only if one does, attempt the install.

    Never raises on a candidate failure — enumeration, like the corpus probe; the decision it feeds
    (run vs skip) is the spec's own conditional."""
    import importlib
    import glob
    log = []
    chosen = None
    for c in EDITOR_CANDIDATES:
        rec = dict(c)
        if time.time() > deadline:
            log.append({**rec, "decision": "not_attempted", "usable": False,
                        "reason": "reach cap reached"})
            continue
        # --- stage 1: does the repo ship gemma hyper-parameters at all?
        listings, gemma = {}, []
        if c["repo"]:
            for hp in c["hparam_paths"]:
                try:
                    names = _list_repo_dir(c["repo"], hp)
                except Exception as e:      # noqa: BLE001 — reach enumeration: report availability
                    listings[hp] = f"unreadable: {type(e).__name__}: {str(e)[:120]}"
                    continue
                listings[hp] = names
                gemma += [n for n in (names or []) if "gemma" in n.lower()]
        rec["hparam_listings"] = listings
        rec["gemma_hparams"] = sorted(set(gemma))
        if c["repo"] and not gemma:
            rec["usable"] = False
            rec["reason"] = ("the repository ships NO gemma hyper-parameters (see hparam_listings); "
                             "authoring them plus the layer covariance statistics is adaptation, "
                             "which §3/A5 forbids")
            log.append(rec)
            continue
        # --- stage 2: it claims gemma support (or has no inspectable repo) — try to install it.
        try:
            rec["pip"] = _pip(c["pip"], timeout=max(60, int(deadline - time.time())))
            mod = importlib.import_module(c["module"])
            root = pathlib.Path(mod.__file__).resolve().parent
            hpf = sorted(glob.glob(str(root / "hparams/*/*.yaml"))
                         + glob.glob(str(root.parent / "hparams/*/*.yaml"))
                         + glob.glob(str(root / "hparams/*/*.json"))
                         + glob.glob(str(root.parent / "hparams/*/*.json")))
            rec["hparams_installed"] = [pathlib.Path(x).name for x in hpf]
            rec["gemma_hparams"] = [x for x in hpf if "gemma" in pathlib.Path(x).name.lower()]
            rec["usable"] = bool(rec["gemma_hparams"])
            rec["reason"] = ("installs AND ships gemma hyper-parameters" if rec["usable"] else
                             "installs, but no gemma hyper-parameters are present after install")
            if rec["usable"] and chosen is None:
                chosen = rec
        except Exception as e:      # noqa: BLE001 — reach enumeration: report availability
            rec["usable"] = False
            rec["reason"] = f"not installable: {type(e).__name__}: {str(e)[:300]}"
        log.append(rec)
    n_with_gemma = sum(1 for r in log if r.get("gemma_hparams"))
    ev = {"candidates": log, "chosen": chosen, "reachable": chosen is not None,
          "cap_s": REACH_CAP_S, "n_candidates": len(log),
          "n_shipping_gemma_hparams": n_with_gemma,
          "question": ("§3/A5: does a maintained ROME/MEMIT implementation support gemma-2b? "
                       "Settled by whether it ships gemma hyper-parameters.")}
    (pathlib.Path(out) / "a5_reach.json").write_text(json.dumps(ev, indent=2, default=str))
    return ev


def _counterfact(n):
    from datasets import load_dataset
    log = []
    for cid in COUNTERFACT_CANDIDATES:
        try:
            d = load_dataset(cid, split="train", streaming=True)
            rows = [r for _, r in zip(range(n), d)]
            log.append({"id": cid, "usable": True, "keys": sorted(rows[0].keys())})
            return rows, log
        except Exception as e:      # noqa: BLE001 — probe enumeration
            log.append({"id": cid, "usable": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return None, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-min", type=float, default=240.0, help="§3/A5 hard wall-clock cap")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-id", default="humboldt_a5")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cap_at = t0 + a.cap_min * 60

    H.hf_setup()                       # FIRST — XET off + gated-repo auth (§0.2.7)
    import numpy as np
    gates = H.Gates(out)
    S.run_all_offline(gates)

    reach = _reach(out, min(t0 + REACH_CAP_S, cap_at))
    cf_rows, cf_log = (_counterfact(N_EDITS) if reach["reachable"] else (None, []))
    manifest = {"arm": "A5", "run_id": a.run_id, "commit_sha": H.commit_sha(),
                "cap_min": a.cap_min, "reach": reach, "counterfact": cf_log,
                "n_edits_planned": N_EDITS, "seed": a.seed}

    if not reach["reachable"] or cf_rows is None:
        manifest["status"] = "SKIPPED"
        # Derived from the evidence, not asserted: say exactly what was observed.
        if not reach["reachable"]:
            det = "; ".join(f"{r['name']}: {r.get('reason', '?')}" for r in reach["candidates"])
            manifest["skip_reason"] = (
                f"{reach['n_shipping_gemma_hparams']} of {reach['n_candidates']} candidate "
                f"implementations ship gemma hyper-parameters. §3/A5 forbids authoring them "
                f"('no adaptation heroics'). Per candidate — {det}")
        else:
            manifest["skip_reason"] = "CounterFact unreachable"
        # `reach` already carries n_candidates / n_shipping_gemma_hparams; passing either again
        # explicitly collides with the ** expansion (the same defect class as the earlier
        # require(name, ok, **ev) collision).
        gates.record("A5_reach", False, **{k: v for k, v in reach.items() if k != "candidates"},
                     executed_pivot=True, skip_reason=manifest["skip_reason"])
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        H.s3_sync(out, f"humboldt/{a.run_id}")
        print("=" * 78)
        print(f"HUMBOLDT_A5_SKIPPED reason={manifest['skip_reason']}")
        return

    # ---- the editor is reachable: apply the edits and record ΔW projections in A4's schema.
    gates.record("A5_reach", True, chosen=reach["chosen"]["name"],
                 gemma_hparams=reach["chosen"]["gemma_hparams"])
    import torch
    from transformers import AutoModelForCausalLM
    from easyeditor import BaseEditor, ROMEHyperParams          # noqa: F401 — reach-gated import
    # The SKIP path is pure HTTP + pip and runs anywhere, so this arm rides the CPU pipeline. The
    # RECORDED path applies closed-form edits and must not silently crawl on a CPU box.
    assert torch.cuda.is_available(), \
        ("A5 became reachable but is running without CUDA — closed-form editing needs a GPU. "
         "Relaunch on g6e.8xlarge,p5.4xlarge.")

    bank_dir = W.fetch_banks(H.cache_dir(out, "banks"))
    hp_path = reach["chosen"]["gemma_hparams"][0]
    hparams = ROMEHyperParams.from_hparams(hp_path)
    before = AutoModelForCausalLM.from_pretrained(H.GEMMA1_REPO, torch_dtype=torch.float32,
                                                  device_map="cpu")
    W0 = {(t, L): before.model.state_dict()[W.matrix_key(t, L)].clone()
          for t in H.TYPES for L in range(H.N_LAYERS)}
    del before

    atlas = H.Atlas(out, arm="A5", run_id=a.run_id)   # bound BEFORE any loop (§0.2.4)
    editor = BaseEditor.from_hparams(hparams)
    _m, edited, _ = editor.edit(
        prompts=[r.get("prompt") or r["requested_rewrite"]["prompt"] for r in cf_rows],
        target_new=[r.get("target_new") or r["requested_rewrite"]["target_new"]["str"]
                    for r in cf_rows],
        subject=[r.get("subject") or r["requested_rewrite"]["subject"] for r in cf_rows],
        keep_original_weight=False)
    sd = edited.model.state_dict() if hasattr(edited, "model") else edited.state_dict()
    for t in H.TYPES:
        for L in range(H.N_LAYERS):
            k = W.matrix_key(t, L)
            if k not in sd:
                continue
            dW = (sd[k].detach().float().cpu().numpy().astype(np.float64)
                  - W0[(t, L)].numpy().astype(np.float64))
            fro2 = float((dW ** 2).sum())
            fl = {**H.type_flags(t), "editor": reach["chosen"]["name"], "n_edits": len(cf_rows),
                  "dataset": cf_log[-1]["id"], "dW_fro2": fro2}
            atlas.add("ext_dW_fro2", value=fro2, layer=L, type=t, model_id="gemma-2b+ROME",
                      model_sha="closed-form-edit", flags=fl)
            if fro2 == 0.0:
                continue
            sv = np.linalg.svd(dW, compute_uv=False)
            atlas.add("ext_dW_singular_spectrum",
                      vector_ref=atlas.vector(f"A5__sv__{t}__L{L}", sv), layer=L, type=t,
                      model_id="gemma-2b+ROME", model_sha="closed-form-edit", n=int(sv.size),
                      flags=fl)
            for bid in H.BANK_IDS:
                U = W.load_bank(bank_dir, t, L, bid)
                atlas.add("ext_dW_bank_energy_fraction",
                          value=float(np.linalg.norm(dW @ U) ** 2 / fro2), layer=L, type=t,
                          bank_id=bid, model_id="gemma-2b+ROME", model_sha="closed-form-edit",
                          flags={**fl, "chance": H.BANK_K / H.IN_DIM[t]})
        H.s3_sync(out, f"humboldt/{a.run_id}")
        assert time.time() < cap_at, "A5 hit its 4h cap mid-recording — partial output is on S3"

    atlas.close()
    rep = atlas.to_parquet()
    manifest["status"] = "RECORDED"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "a5_summary.json").write_text(json.dumps(
        {"manifest": manifest, "atlas": rep, "gates": gates.gates,
         "elapsed_min": (time.time() - t0) / 60}, indent=2, default=str))
    H.s3_sync(out, f"humboldt/{a.run_id}")
    print("=" * 78)
    print(f"HUMBOLDT_A5_OK rows={rep['n_rows']} edits={len(cf_rows)}")


if __name__ == "__main__":
    main()
