"""The one job shape of the clean-neuron line (studies/neurons/CONSOLIDATION_2026-11-04.md section 4): an instrument module defines INSTRUMENT,
check_items(items), Job(Run) with run(), and dry_items(model, tokens, offsets, hw); this module gives the command line, the inputs, the model, the
metrics record and the weight-free dry run. Written against infra/neurons.

    python -m infra.aws_pg.jobs.neurons_<instrument> --items <s3 or local json> --corpus <s3 corpus dir> --out <dir>
    uv run --offline --no-project --with "transformers>=5.5" --with torch --with tokenizers --with numpy \
        python -m infra.aws_pg.jobs.neurons_<instrument> --dry-run <dir>

Items (JSON): {"instrument", "model": {"key", "repo", "revision"}, "corpus": {"tokens_sha256"}, "hw", "length", "bos", ...the instrument's own}.
The model's repo and revision come from the items (no model name is written here); the corpus is checked against the items' sha256.
metrics.json is rewritten after every case and holds status, the items' sha256, every gate record (rules.gate: value, failing input, identity)
and the case records; arrays go to <out>/<name>.npz before metrics.json names them.
Dry run: a tiny random Olmo 3 built from its transformers config (neuron_off_dryrun.tiny_model, eager attention, bf16, nothing downloaded), a
fake corpus of random ids in documents, the instrument's dry items, Job.run, then one GATE line per gate: its value passes and its failing input
fails. Exit by rules.exit_code (0 green). Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import torch

from infra.neurons import capture as C
from infra.neurons import rules as R

DRY = {"vocab": 384, "hw": 64, "n_windows": 48, "n_layers": 4, "bos": 1}


class Run:
    """The output folder and its metrics record."""

    def __init__(self, out, items: dict, model, tokens, offsets, device: str, batch: int = 32):
        self.out = pathlib.Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.items, self.model, self.tokens, self.offsets, self.device, self.batch = items, model, tokens, offsets, device, batch
        self.ad, self.tok = C.Adapter(model), None          # the tokenizer is set by real(); a dry run has none
        self.kw = dict(tokens=tokens, length=items.get("length"), bos=items.get("bos"), batch=batch, hw=items["hw"], device=device)
        self.m = {"instrument": items["instrument"], "status": "running", "items_sha256": items.get("sha256"), "adapter": {"arch": self.ad.arch,
                  "n_layers": len(self.ad.layers), "width": self.ad.width, "act": self.ad.act}, "gates": {}, "cases": {}, "utc_start": time.time()}

    def gate(self, name: str, rec: dict):
        self.m["gates"][name] = rec
        self.write()

    def save(self, name: str, **arrays) -> str:
        p = self.out / f"{name}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **arrays)
        return str(p.relative_to(self.out))

    def write(self):
        tmp = self.out / "metrics.json.tmp"
        tmp.write_text(json.dumps(self.m, indent=1, default=float))
        tmp.replace(self.out / "metrics.json")

    def done(self):
        self.m["status"], self.m["utc_end"] = "done", time.time()
        self.write()


def report(m: dict) -> tuple:
    """One GATE line per gate, then the exit code by rules.exit_code (every gate a pass whose failing input fails: 0)."""
    lines = []
    for k, g in m["gates"].items():
        lines.append(f"GATE {k}: {g['value']} -> {'pass' if g['pass'] else 'FAIL'} (failing input {g['failing_input']['input']}: "
                     f"{g['failing_input']['value']} -> {'fails' if g['failing_input_failed'] else 'DID NOT FAIL'}); identity {g['identity']['value']}"
                     + (f" ({g['identity']['reason']})" if g['identity']['reason'] else ""))
    bad = {k: g for k, g in m["gates"].items() if not g["pass"]}
    code, why = R.exit_code(m["gates"], {k: {"reading": "cancelled"} for k in bad}, complete=m["status"] == "done")
    return lines, code, why


def fake_corpus(vocab: int, n_windows: int, hw: int, seed: int = 0) -> tuple:
    """Random ids (never 0 to 2, the special ids) in documents of 20 to 120 tokens."""
    rng = np.random.default_rng(seed)
    n = n_windows * hw
    tokens = rng.integers(3, vocab, n).astype(np.int64)
    cuts = np.cumsum(rng.integers(20, 120, n // 20))
    return tokens, np.r_[0, cuts[cuts < n], n].astype(np.int64)


def dry(mod, out):
    from infra.aws_pg.jobs.neuron_off_dryrun import tiny_model
    out = pathlib.Path(out)
    model = tiny_model("olmo3", seed=0, n_layers=DRY["n_layers"]).requires_grad_(False)
    model.config._attn_implementation = "eager"
    tokens, offsets = fake_corpus(DRY["vocab"], DRY["n_windows"], DRY["hw"])
    items = {"instrument": mod.INSTRUMENT, "hw": DRY["hw"], "length": None, "bos": DRY["bos"], "sha256": "dry", **mod.dry_items(model, tokens, offsets, DRY)}
    mod.check_items(items)
    (out / "items.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps(items, indent=1))
    job = mod.Job(out / "job", items, model, tokens, offsets, "cpu", batch=8)
    job.run()
    lines, code, why = report(job.m)
    lines += mod.dry_checks(model, tokens, offsets, items) if hasattr(mod, "dry_checks") else []
    for x in lines:
        print(x)
    print(f"DRY RUN {mod.INSTRUMENT}: exit {code} ({why})")
    return code


def real(mod, a):
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.neurons import banks as B
    from infra.pg import neurons_harvest as NH
    assert torch.cuda.is_available(), "no CUDA on this instance"
    p = fetch(a.items, f"{mod.INSTRUMENT}_items.json")
    items = {**json.loads(pathlib.Path(p).read_text()), "sha256": hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()}
    assert items["instrument"] == mod.INSTRUMENT, (items["instrument"], mod.INSTRUMENT)
    mod.check_items(items)
    local, sha, rec = download(items["model"]["repo"], items["model"]["revision"])
    corpus = B.load_corpus(fetch_corpus(a.corpus, items["corpus"]["tokname"]), items["corpus"]["tokname"])
    assert corpus["tokens_sha256"] == items["corpus"]["tokens_sha256"], "the corpus is not the items' corpus"
    model, tok = NH.load_model(local, items["model"]["key"])
    job = mod.Job(a.out, items, model.requires_grad_(False), corpus["tokens"], corpus["doc_offsets"], "cuda", batch=a.batch)
    job.tok = tok
    job.m.update({"download": rec, "sha": sha, "device_name": torch.cuda.get_device_name(0), "torch": torch.__version__,
                  "transformers": __import__("transformers").__version__})
    job.run()
    job.m["memory_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    job.write()
    lines, code, why = report(job.m)
    print("\n".join(lines) + f"\nJOB {mod.INSTRUMENT}: exit {code} ({why})")
    return code


def main(mod):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--items")
    ap.add_argument("--corpus")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dry-run", default=None, metavar="DIR")
    a = ap.parse_args()
    if a.dry_run:
        sys.exit(dry(mod, a.dry_run))
    assert a.out and a.items and a.corpus, "--out, --items and --corpus are required outside --dry-run"
    sys.exit(real(mod, a))
