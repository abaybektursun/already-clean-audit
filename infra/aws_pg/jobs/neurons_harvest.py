"""D1 harvest job (N5 for N1): one checkpoint per call, DATA_CONTRACT §2.

    python -m infra.aws_pg.jobs.neurons_harvest --model-key Q4B --repo Qwen/Qwen3-4B-Base \
        --corpus /work/out/corpus/corpus --tokname qwen3 --max-tokens 2000000 --sample-neurons 10 --out <dir>

Weights: without `--revision` the single snapshot of `--repo` in the HF cache (`pg run --weights`); with
`--revision` (O7 branches) a Hub download after `hf_online_setup()`, timed, and deleted after the harvest.
`--corpus`, `--poles`, `--positions`, `--probe-texts`: an S3 URI or a local path (a previous pipeline step's output).
`--poles` without `--positions`/`--positions-from` writes the poles' own top-50 as poles/positions.npz and
poles/exemplar_a.npz (DATA_CONTRACT §7, the T_15 rule).
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import time

FETCH = pathlib.Path("/work/fetch")


def fetch(src: str, rel: str, recursive: bool = False) -> pathlib.Path:
    """S3 URIs are copied under /work/fetch; local paths are used as they are."""
    if not src.startswith("s3://"):
        p = pathlib.Path(src)
        assert p.exists(), f"local input missing: {p}"
        return p
    dst = FETCH / rel
    cmd = ["aws", "s3", "cp", src, str(dst), "--region", "us-east-1", "--only-show-errors"]
    subprocess.run(cmd + (["--recursive"] if recursive else []), check=True)
    return dst


def fetch_corpus(src: str, tokname: str) -> pathlib.Path:
    if not src.startswith("s3://"):
        return fetch(src, "")
    src = src.rstrip("/")
    fetch(f"{src}/manifest.json", "corpus/manifest.json")
    fetch(f"{src}/tok/{tokname}", f"corpus/tok/{tokname}", recursive=True)
    return FETCH / "corpus"


def download(repo: str, revision: str) -> tuple[str, str, dict]:
    from huggingface_hub import snapshot_download
    from infra.pg.matrix_models import HF_ALLOW
    t0 = time.time()
    local = snapshot_download(repo, revision=revision, allow_patterns=HF_ALLOW)
    dt = time.time() - t0
    n = sum(f.stat().st_size for f in pathlib.Path(local).rglob("*") if f.is_file())
    rec = {"repo": repo, "branch": revision, "sha": pathlib.Path(local).name, "bytes": n, "seconds": round(dt, 1),
           "MB_per_s": round(n / 1e6 / dt, 1), "xet_disabled": os.environ.get("HF_HUB_DISABLE_XET")}
    print(f"[download] {repo}@{revision}: {n / 1e9:.2f} GB in {dt:.0f} s = {rec['MB_per_s']} MB/s", flush=True)
    return local, rec["sha"], rec


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default=None, help="Hub branch or sha to download; absent = the cached snapshot")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokname", required=True, choices=["olmo3", "qwen3", "gemma3"])
    ap.add_argument("--max-tokens", required=True, help="an integer, or 'all' for every full window of the corpus")
    ap.add_argument("--snapshot-bands", default="")
    ap.add_argument("--poles", default=None, help="JSON list of pole ids, or a samples/sample_draw.json")
    ap.add_argument("--positions", default=None, help="positions.npz with tok [P, E]")
    ap.add_argument("--positions-from", default=None, help="a previous harvest's out dir (poles/positions.npz, else topk/)")
    ap.add_argument("--probe-texts", default=None)
    ap.add_argument("--sample-neurons", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wikitext-parity", type=float, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--window", type=int, default=2048)
    return ap


def main():
    a = parser().parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.pg import neurons_corpus as NC
    from infra.pg import neurons_harvest as NH
    assert torch.cuda.is_available(), "no CUDA on this instance"
    assert not (a.positions and a.positions_from), "--positions and --positions-from are exclusive"

    extra = {"args": vars(a), "device_name": torch.cuda.get_device_name(0), "pg_git_sha": os.environ.get("PG_GIT_SHA")}
    if a.revision:
        local, sha, extra["download"] = download(a.repo, a.revision)
    else:
        local, sha = NC.local_snapshot(a.repo)
    corpus = NC.load_tok(fetch_corpus(a.corpus, a.tokname), a.tokname)
    poles = NH.load_poles(fetch(a.poles, "poles.json")) if a.poles else None
    positions = None
    if a.positions_from:
        positions = NH.positions_from(a.positions_from, poles)
    elif a.positions:
        positions = np.load(fetch(a.positions, "positions.npz"))["tok"]
    texts = None
    if a.probe_texts:
        texts = [json.loads(x) for x in fetch(a.probe_texts, "probe.jsonl").read_text().splitlines() if x.strip()]

    t0 = time.time()
    model, tok = NH.load_model(local, a.model_key)
    extra["load_s"] = round(time.time() - t0, 1)
    bands = [int(b) for b in a.snapshot_bands.split(",") if b]
    man = NH.run(out, model=model, tok=tok, corpus=corpus, model_key=a.model_key, repo=a.repo, revision=sha,
                 branch=a.revision, max_tokens=None if a.max_tokens == "all" else int(a.max_tokens), window=a.window, batch=a.batch, snapshot_bands=bands,
                 poles=poles, positions=positions, probe_texts=texts, sample_neurons=a.sample_neurons, seed=a.seed,
                 wikitext_parity=a.wikitext_parity, device="cuda", extra_manifest=extra)
    if a.revision:
        del model
        torch.cuda.empty_cache()
        repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / ("models--" + a.repo.replace("/", "--"))
        shutil.rmtree(repo_dir)
        assert not repo_dir.exists()
        man["download"]["deleted_after_harvest"] = str(repo_dir)
        NH.dump(out / "manifest.json", man)
    print("NEURONS_HARVEST_OK", json.dumps({"model_key": a.model_key, "revision": sha,
                                            "tokens_per_second": man["tokens_per_second"]}), flush=True)


if __name__ == "__main__":
    main()
