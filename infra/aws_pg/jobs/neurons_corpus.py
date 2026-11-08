"""D1 corpus job (N5 for N2): writes `<out>/corpus/` per DATA_CONTRACT §1.

    python -m infra.aws_pg.jobs.neurons_corpus --tokens-per-shard 60000 --olmo-revision stage1-step1413814 --out <dir>

Needs the Qwen/Qwen3-4B-Base and google/gemma-3-4b-pt snapshots in the HF cache (`pg run --weights`) for
their tokenizers; the OLMo-3 tokenizer and the Dolma 3 shards come from the Hub after `hf_online_setup()`.
"""
import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens-per-shard", type=int, default=1_000_000)
    ap.add_argument("--olmo-revision", default="stage1-step1413814")
    ap.add_argument("--n-dirs", type=int, default=50)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    preinstalled = importlib.util.find_spec("zstandard") is not None
    subprocess.run([sys.executable, "-m", "pip", "install", "zstandard"], check=True)
    from infra.pg import neurons_corpus as NC
    man = NC.build(out / "corpus", a.tokens_per_shard, a.olmo_revision, a.n_dirs)
    man["zstandard_preinstalled"] = preinstalled
    man["args"] = vars(a)
    (out / "corpus" / "manifest.json").write_text(json.dumps(man, indent=1))
    print("NEURONS_CORPUS_OK", json.dumps({"n_docs": man["n_docs"],
                                           "n_tokens": {k: v["n_tokens"] for k, v in man["tokenizers"].items()}}),
          flush=True)


if __name__ == "__main__":
    main()
