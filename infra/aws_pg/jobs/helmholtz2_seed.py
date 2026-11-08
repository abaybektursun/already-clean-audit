"""HELMHOLTZ-2 seeding + BIND-recon job (CPU, --no-weights).

Downloads gemma-3-4b-pt / gemma-3-4b-it and the SELECTED Gemma Scope 2 SAE folders into
the instance HF cache, syncs the cache up to the S3 weights plane, and records every
[BIND] the spec left open: repo revisions, text-config n_layers/d_model, the exact SAE
directories found+chosen, and the transformers version (fail-fast assert that it knows
gemma3, so R0 cannot burn GPU time on an unsupported env).

  pg run infra.aws_pg.jobs.helmholtz2_seed --no-weights --type r7i.2xlarge \
      --s3-prefix results/h2_seed --budget-min 90
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess

BUCKET = "pg-lineage-769063704696"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gemma-repos", default="google/gemma-3-4b-pt,google/gemma-3-4b-it")
    ap.add_argument("--scope-repo", default="google/gemma-scope-2-4b-pt")
    ap.add_argument("--primary-width", default="65k")
    ap.add_argument("--stability-widths", default="16k,262k")
    ap.add_argument("--l0", default="medium")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()

    import transformers
    from transformers import AutoConfig
    from huggingface_hub import HfApi, snapshot_download

    assert "gemma3" in transformers.models.auto.configuration_auto.CONFIG_MAPPING_NAMES, \
        f"transformers {transformers.__version__} does not know gemma3 — rebake env"
    rep = {"object": "HELMHOLTZ-2 seed + BIND recon",
           "transformers": transformers.__version__, "repos": {}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    for repo in a.gemma_repos.split(","):
        p = snapshot_download(repo)
        rev = pathlib.Path(p).name
        rep["repos"][repo] = {"revision": rev, "path": p}
        print(f"[SEED] {repo} @ {rev}", flush=True)
        _w()

    cfg = AutoConfig.from_pretrained(a.gemma_repos.split(",")[0])
    tc = getattr(cfg, "text_config", cfg)
    n_layers = int(tc.num_hidden_layers)
    d_model = int(tc.hidden_size)
    rep["bind"] = {"n_layers": n_layers, "d_model": d_model,
                   "model_type": cfg.model_type,
                   "depth_layers": {p: max(0, round(p * n_layers) - 1)
                                    for p in (0.25, 0.50, 0.65, 0.85)}}
    print(f"[SEED] BIND n_layers={n_layers} d_model={d_model} "
          f"depths={rep['bind']['depth_layers']}", flush=True)
    _w()

    # ---- introspect the Scope-2 repo layout and choose SAE dirs ----
    files = HfApi().list_repo_files(a.scope_repo)
    param_files = [f for f in files if f.endswith("params.npz") or f.endswith(".safetensors")]
    dirs = sorted({str(pathlib.PurePosixPath(f).parent) for f in param_files})
    rep["scope_repo"] = {"repo": a.scope_repo, "n_param_dirs": len(dirs)}
    pat = re.compile(r"layer_(\d+)")
    def _layers_in(ds):
        got = {}
        for d in ds:
            m = pat.search(d)
            if m:
                got.setdefault(int(m.group(1)), []).append(d)
        return got

    single = [d for d in dirs if d.startswith("resid_post/") or d == "resid_post"
              or ("resid_post" in d and "_all" not in d)]
    alldirs = [d for d in dirs if "resid_post_all" in d]
    l0 = a.l0
    prim = [d for d in single if f"width_{a.primary_width}" in d and f"l0_{l0}" in d]
    prim_layers = _layers_in(prim)
    assert prim_layers, (f"no single-layer resid_post dirs at width_{a.primary_width} l0_{l0}; "
                         f"single-layer dirs: {single[:40]}")
    chosen = sorted(set(sum(prim_layers.values(), [])))
    mid = sorted(prim_layers)[len(prim_layers) // 2]
    for w in a.stability_widths.split(","):
        sw = [d for d in single if f"width_{w}" in d and f"l0_{l0}" in d and f"layer_{mid}_" in d]
        assert sw, f"no width_{w} l0_{l0} dir at layer {mid}; have {single[:40]}"
        chosen += sw
    all_layers = _layers_in(alldirs)
    assert all_layers, f"no resid_post_all dirs found; dirs sample: {dirs[:40]}"
    final_l = max(all_layers)
    final_dirs = sorted(all_layers[final_l])[:1]
    chosen += final_dirs
    chosen = sorted(set(chosen))
    rep["scope_repo"]["single_layers_at_primary"] = sorted(prim_layers)
    rep["scope_repo"]["mid_layer"] = mid
    rep["scope_repo"]["final_all_layer"] = final_l
    rep["scope_repo"]["chosen_dirs"] = chosen
    print(f"[SEED] scope-2: layers@{a.primary_width}/{l0}={sorted(prim_layers)} "
          f"final_all={final_l} -> {len(chosen)} dirs", flush=True)
    _w()

    p = snapshot_download(a.scope_repo, allow_patterns=[d + "/*" for d in chosen])
    rep["scope_repo"]["revision"] = pathlib.Path(p).name
    _w()

    import os
    hf_home = os.environ.get("HF_HOME")
    assert hf_home, "HF_HOME not set (run_job.sh should set it)"
    print(f"[SEED] s3 sync {hf_home} -> s3://{BUCKET}/weights/hf/", flush=True)
    subprocess.run(["aws", "s3", "sync", hf_home, f"s3://{BUCKET}/weights/hf/",
                    "--region", "us-east-1", "--no-progress"], check=True)
    rep["synced"] = True
    _w()
    print("HELMHOLTZ2_SEED_OK " + json.dumps(
        {k: rep["bind"][k] for k in ("n_layers", "d_model")} |
        {"chosen": len(chosen), "revision": rep["scope_repo"]["revision"]}), flush=True)


if __name__ == "__main__":
    main()
