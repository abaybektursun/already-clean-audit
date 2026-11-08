"""Shared scaffolding for the STAGEDIFF jobs: boot (HF online + pin re-exec), the B1-beh / B2 / B3 scorers,
and the banking policy (per-item vectors, samples with one right and one wrong line, discovery printed,
confirmation sealed). Every job imports the science from `infra/pg/stagediff_*`; nothing here decides.
"""
from __future__ import annotations

import os
import sys
import time

TRACKED_PER_SET = 2       # items whose samples are banked in EVERY arm (the same instance through every arm)


def boot(module: str, cid: str, argv) -> bool:
    """HF online mode first (token from SSM), then the transformers pin: True when the job already ran as a
    child under the pin venv (the caller returns), False to continue in-process."""
    from infra.pg.diff_harvest import hf_online_setup
    if os.environ.get("PG_S3_PREFIX") or os.environ.get("HF_TOKEN"):
        hf_online_setup()
    from infra.pg import stagediff_models as SM
    return SM.ensure_pin(cid, module, list(argv)) is not None


def tracked_hashes(items: list, per_set: int = TRACKED_PER_SET) -> set:
    seen, out = {}, set()
    for it in items:
        k = it.get("set") or it.get("subject")
        if it.get("split") == "discovery" and seen.get(k, 0) < per_set:
            out.add(it["hash"]); seen[k] = seen.get(k, 0) + 1
    return out


# ------------------------------------------------------------------------------------- scorers
def score_b1(member: dict, items: list) -> list:
    """B1-beh records: top-1 over the shared prefix at the last prompt token; `correct` (strict single-token
    target ids; None when the item has none) and `correct_first` (first-token ids, every item)."""
    import torch
    from infra.pg.stagediff_models import V_SHARED
    fwd, tok = member["fwd"], member["tok"]
    prompts = [it["prompt"] for it in items]
    logits = fwd.next_token_logits(prompts)[:, :V_SHARED]
    probs = torch.softmax(logits.float(), dim=-1)
    rows_ids = fwd.encode_rows(prompts)
    out = []
    for i, it in enumerate(items):
        top = int(logits[i].argmax())
        t_ids = it["target_ids"]
        out.append({"hash": it["hash"], "set": it["set"], "split": it["split"], "name": it["name"],
                    "prompt": it["prompt"], "prompt_ids": rows_ids[i], "n_prompt_tokens": len(rows_ids[i]),
                    "bos_present": member["bos_id"] is not None,
                    "top_id": top, "top_token": tok.decode([top]), "top_p": float(probs[i, top]),
                    "target": it["target"], "target_ids": t_ids, "first_ids": it["first_ids"],
                    "target_ids_in_context": it["target_ids_in_context"],
                    "target_tokens_in_context": [tok.decode([t]) for t in it["target_ids_in_context"]],
                    "p_target": float(max((probs[i, t] for t in t_ids), default=0.0)),
                    "p_first": float(max((probs[i, t] for t in it["first_ids"]), default=0.0)),
                    "scorable": bool(t_ids),
                    "correct": (top in t_ids) if t_ids else None,
                    "correct_first": top in it["first_ids"]})
    return out


def score_b2(member: dict, items: list) -> list:
    """B2 records: 5-shot MMLU cloze, per-token-mean log-likelihood argmax over the 4 option texts."""
    fwd = member["fwd"]
    out = []
    for it in items:
        lp = fwd.continuation_logprobs([it["prompt"]] * 4, it["choices"])
        means = [s / n for s, n in lp]
        pred = max(range(4), key=lambda k: means[k])
        out.append({"hash": it["hash"], "subject": it["subject"], "split": it["split"],
                    "prompt_tail": it["prompt"][-400:], "prompt_chars": len(it["prompt"]), "choices": it["choices"],
                    "scored": [{"sum_logprob": s, "n_tok": n, "mean_logprob": s / n} for s, n in lp],
                    "pred": pred, "answer": it["answer"], "correct": pred == it["answer"],
                    "bos_present": member["bos_id"] is not None})
    return out


def run_b3(member: dict, max_tokens: int) -> dict:
    from infra.pg.text_eval import shell_text_perplexity
    return shell_text_perplexity(member["fwd"], window=2048, max_tokens=max_tokens)


# ------------------------------------------------------------------------------------- banking
def acc_summary(records: list, key: str, split: str) -> dict:
    rows = [r for r in records if r["split"] == split and r[key] is not None]
    by = {}
    for r in rows:
        by.setdefault(r.get("set") or "mmlu", []).append(int(r[key]))
    return {"n": len(rows), "acc": (sum(int(r[key]) for r in rows) / len(rows)) if rows else None,
            "per_set": {s: {"n": len(v), "acc": sum(v) / len(v)} for s, v in sorted(by.items())}}


def bank_records(bank, readout: str, arm: str, records: list, tracked: set, sample_all: bool = False,
                 keys=("correct",)):
    """per_item/<readout>_<arm>.json = {hash: {key: value, split}}; metrics.readouts[readout][arm] holds the
    DISCOVERY summary, metrics.sealed[readout][arm] the confirmation one (never printed); samples: every
    record when `sample_all`, else the first right + first wrong per half plus the tracked items."""
    per_item = {r["hash"]: {**{k: r[k] for k in keys}, "split": r["split"]} for r in records}
    bank.per_item(f"{readout}_{arm}", per_item)
    disc = {k: acc_summary(records, k, "discovery") for k in keys}
    conf = {k: acc_summary(records, k, "confirmation") for k in keys}
    sec = bank.metrics["readouts"].setdefault(readout, {})
    sec[arm] = {"discovery": disc, "n_records": len(records)}
    bank.metrics["sealed"].setdefault(readout, {})[arm] = {"confirmation": conf}
    bank.write()
    if sample_all:
        picks = list(range(len(records)))
    else:
        picks, got = [], set()
        for i, r in enumerate(records):
            tag = (r["split"], r[keys[0]])
            if r["hash"] in tracked or (r[keys[0]] is not None and tag not in got):
                picks.append(i); got.add(tag)
    for i in picks:
        bank.sample(readout, arm, {**records[i], "selection": "all" if sample_all else
                                   ("tracked" if records[i]["hash"] in tracked else "first-right/first-wrong")})
    # counts over the DISCOVERY half only: a both-half count beside the discovery n would let the sealed
    # accuracy be recovered by subtraction (CK-10 pre-freeze FLAG)
    n_right = sum(1 for r in records if r["split"] == "discovery" and r[keys[0]] is True)
    n_wrong = sum(1 for r in records if r["split"] == "discovery" and r[keys[0]] is False)
    bank.event("readout", readout=readout, arm=arm, n=len(records),
               discovery_acc=disc[keys[0]]["acc"], discovery_n_right=n_right, discovery_n_wrong=n_wrong)
    if not (n_right and n_wrong):
        bank.event("sample-gap", readout=readout, arm=arm,
                   note="discovery half has fewer than one right AND one wrong record — §6.8 sample rule not met by the data itself")
    return per_item


class Timer:
    def __init__(self, bank, key):
        self.bank, self.key = bank, key

    def __enter__(self):
        self.t0 = time.perf_counter(); return self

    def __exit__(self, *exc):
        self.bank.put("timing_s", self.key, round(time.perf_counter() - self.t0, 2))


def s3_fetch(prefix: str, dest: str, names: list):
    """`aws s3 cp` each named file under an S3 prefix (or copy from a local dir) into `dest`."""
    import shutil
    import subprocess
    os.makedirs(dest, exist_ok=True)
    for n in names:
        src = f"{prefix.rstrip('/')}/{n}"
        tgt = os.path.join(dest, n)
        if prefix.startswith("s3://"):
            subprocess.run(["aws", "s3", "cp", src, tgt, "--region", "us-east-1"], check=True)
        else:
            shutil.copyfile(src, tgt)
        assert os.path.exists(tgt) and os.path.getsize(tgt) > 0, tgt
    return dest


def fake_docs(n: int = 40) -> list:
    """Synthetic wikitext-shaped documents for the dry-runs (no dataset load on the laptop)."""
    import random
    rng = random.Random(0)
    words = "the of and a to in is was for on that with as by at from it his an be this which or".split()
    docs = []
    for i in range(n):
        body = " ".join(rng.choice(words) for _ in range(400))
        docs.append(f" = Article {i} = \n {body} . \n")
    return docs


def dryrun_ppl(member: dict, max_tokens: int) -> dict:
    return {"task": "wikitext-2-raw perplexity (DRY-RUN STUB)", "ppl": 7.39, "mean_nll": 2.0, "n_tokens": 110,
            "window": 2048, "n_windows": 2, "per_window": [[100.0, 50], [120.0, 60]], "samples": [], "dropped_leading_target": False,
            "n_dropped_leading_targets": 0, "protocol": "dry-run stub", "forward": "stub"}
