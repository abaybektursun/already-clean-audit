"""POSTCHECK cell A — P1, P2 and P4: everything that can be settled from files.

No GPU work and no forward pass. Three claims, each re-measured from a fresh read at a pinned revision:

  P1  the action expert's depth gate, per layer, out of both checkpoints' safetensors, plus the same
      read over every released allenai MolmoAct model whose config the seeder banked
  P2  the base config against the fine-tune's config against the published recipe, from the seeded
      text trees and the repository clone
  P4  `meta/episodes` against the data files' own columns, over an HTTP range read of four int64
      columns rather than 23.8 GB of images

The scanner and the weights reader are checked on planted inputs before they touch anything released
(`postcheck_dataset.selftest`, `postcheck_weights.selftest`), and both self-tests run here so their
result is in this cell's artifact rather than in a developer's terminal.

  pg run infra.aws_pg.jobs.postcheck_files --type g5.xlarge \\
     --weights models--allenai--MolmoAct2-Think-LIBERO,models--allenai--MolmoAct2-Think \\
     --s3-prefix results/postcheck/files/1 --budget-min 120
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Dict, List, Optional

from infra.pg import postcheck_dataset as pcd
from infra.pg import postcheck_weights as pcw

BUCKET = "pg-lineage-769063704696"
S3_CODE = f"s3://{BUCKET}/postcheck/code/v1"
REGION = "us-east-1"
DATASET_REPO = "allenai/MolmoAct2-LIBERO-Dataset"
N_DATA_FILES = 378


class Bank:
    """One growing `metrics.json`, flushed after every stage, because `run_job.sh` syncs `out/` every
    30 s and a cell killed mid-scan must leave what it finished readable."""

    def __init__(self, out: pathlib.Path, cell: str):
        self.out = pathlib.Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.payload: Dict[str, object] = {"cell": cell, "started_utc": _utc(), "stages": {}}
        self.flush()

    def stage(self, name: str, payload) -> None:
        self.payload["stages"][name] = payload
        self.payload["updated_utc"] = _utc()
        self.flush()
        print(f"[bank] stage {name}", flush=True)

    def top(self, key: str, value) -> None:
        self.payload[key] = value
        self.flush()

    def artifact(self, name: str, payload) -> pathlib.Path:
        p = self.out / name
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=str))
        os.replace(tmp, p)
        return p

    def flush(self) -> None:
        tmp = self.out / "metrics.json.tmp"
        tmp.write_text(json.dumps(self.payload, indent=1, default=str))
        os.replace(tmp, self.out / "metrics.json")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _s3_json(uri: str, dest: pathlib.Path) -> Optional[Dict[str, object]]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["aws", "s3", "cp", uri, str(dest), "--region", REGION],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[s3] MISSING {uri}: {r.stderr[-200:]}", flush=True)
        return None
    return json.loads(dest.read_text())


def _s3_text(uri: str, dest: pathlib.Path) -> Optional[str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["aws", "s3", "cp", uri, str(dest), "--region", REGION],
                       capture_output=True, text=True)
    return dest.read_text() if r.returncode == 0 else None


def _snapshot_dir(repo: str) -> Optional[pathlib.Path]:
    """The restored HF cache directory for a repo, found on disk rather than downloaded."""
    root = pathlib.Path(os.environ.get("HF_HOME", "/opt/hf-cache/hf")) / "hub"
    d = root / ("models--" + repo.replace("/", "--"))
    if not d.is_dir():
        for cand in root.glob("models--*"):
            if cand.name.replace("models--", "").replace("--", "/") == repo:
                d = cand
                break
    snaps = sorted((d / "snapshots").glob("*")) if (d / "snapshots").is_dir() else []
    return snaps[-1] if snaps else None


# ------------------------------------------------------------------ P1
def stage_p1(bank: Bank, work: pathlib.Path) -> Dict[str, object]:
    pcw.selftest()
    out: Dict[str, object] = {"selftest": "ok", "checkpoints": {}}
    for repo in ("allenai/MolmoAct2-Think-LIBERO", "allenai/MolmoAct2-Think"):
        snap = _snapshot_dir(repo)
        if snap is None:
            out["checkpoints"][repo] = {"error": "no restored snapshot on this instance"}
            continue
        read = pcw.read_gate(snap)
        summary = pcw.summarize(read)
        bank.artifact(f"p1_gate_{repo.split('/')[-1]}.json", read)
        out["checkpoints"][repo] = dict(summary, snapshot=str(snap),
                                        n_gate_tensors=read["n_gate_tensors"])
        print(f"[p1] {repo}: {json.dumps(summary)[:300]}", flush=True)
    coll = _fetch_collection(work)
    if coll:
        rows = []
        for mid, entry in sorted((coll.get("models") or {}).items()):
            gate = entry.get("gate") or {}
            rows.append({
                "model": mid,
                "architectures": entry.get("architectures"),
                "action_expert_depth_gate": gate.get("action_expert_depth_gate"),
                "per_layer": gate.get("action_expert_depth_gate_per_layer"),
                "init_bias": gate.get("action_expert_depth_gate_init_bias"),
                "enable_depth_reasoning": gate.get("enable_depth_reasoning"),
                "num_depth_tokens": gate.get("num_depth_tokens"),
                "norm_tags": entry.get("norm_tags"),
                "config_error": entry.get("config_error"),
            })
        bank.artifact("p1_collection.json", {"listed_utc": coll.get("listed_utc"), "rows": rows})
        with_gate = [r["model"] for r in rows if r["action_expert_depth_gate"] is True]
        without = [r["model"] for r in rows if r["action_expert_depth_gate"] is False]
        silent = [r["model"] for r in rows if r["action_expert_depth_gate"] is None]
        out["collection"] = {"n_models": len(rows), "gate_true": with_gate,
                             "gate_false": without, "gate_absent_from_config": silent}
    return out


# ------------------------------------------------------------------ P2
def stage_p2(bank: Bank, work: pathlib.Path) -> Dict[str, object]:
    out: Dict[str, object] = {}
    rev = _s3_json(f"{S3_CODE}/revisions.json", work / "revisions.json")
    out["revisions"] = {k: v.get("sha") for k, v in ((rev or {}).get("repos") or {}).items()}
    for label, repo in (("base", "think_base"), ("tuned", "think_libero")):
        cfg = _s3_json(f"{S3_CODE}/{repo}/config.json", work / repo / "config.json")
        norm = _s3_json(f"{S3_CODE}/{repo}/norm_stats.json", work / repo / "norm_stats.json")
        entry: Dict[str, object] = {"config_found": cfg is not None}
        if cfg:
            entry["gate_keys"] = {k: v for k, v in cfg.items()
                                  if "depth" in k.lower() or "gate" in k.lower()}
            entry["architectures"] = cfg.get("architectures")
        if norm:
            # norm_stats.json is {norm_mode, metadata_by_tag: {tag: {...}}}, so the TAGS are one level
            # down; reading the top level would report "metadata_by_tag" as a normalization tag
            entry["norm_mode"] = norm.get("norm_mode")
            entry["norm_tags"] = sorted((norm.get("metadata_by_tag") or {}).keys())
            entry["libero_tag_present"] = "libero" in (norm.get("metadata_by_tag") or {})
        out[label] = entry
    # the published recipe, as seeded from the repository itself
    grep = _s3_text(f"{S3_CODE}/molmoact2_repo/grep_gate.txt", work / "grep_gate.txt")
    commits = _s3_text(f"{S3_CODE}/molmoact2_repo/commits_touching_gate.txt", work / "commits_gate.txt")
    head = _s3_text(f"{S3_CODE}/molmoact2_repo/HEAD.txt", work / "HEAD.txt")
    readme = _s3_text(f"{S3_CODE}/molmoact2_repo/experiments/README.md", work / "experiments_README.md")
    out["repository"] = {
        "head": (head or "").strip(),
        "grep_action_expert_depth_gate_lines": len([l for l in (grep or "").splitlines() if l.strip()]),
        "grep_sample": [l for l in (grep or "").splitlines() if l.strip()][:12],
        "commits_touching_the_flag": [l for l in (commits or "").splitlines() if l.strip()],
        "readme_found": readme is not None,
    }
    if readme:
        blocks = [b for b in readme.split("```") if "depth" in b.lower() and "--" in b]
        out["repository"]["depth_command_blocks"] = len(blocks)
        gate_in_cmds = [b for b in blocks if "action_expert_depth_gate" in b]
        out["repository"]["depth_command_blocks_setting_the_gate"] = len(gate_in_cmds)
        out["repository"]["depth_command_sample"] = (blocks[0][:1200] if blocks else None)
    return out


# ------------------------------------------------------------------ P4
def stage_p4(bank: Bank, work: pathlib.Path) -> Dict[str, object]:
    planted = pcd.selftest()
    bank.artifact("p4_selftest.json", planted)

    # HF must be ONLINE for this scan and `run_job.sh` exports HF_HUB_OFFLINE=1 for the weight cache;
    # huggingface_hub reads that into a module constant at import, so the scan runs in a CHILD process
    # with the variable cleared (the lesson DEPTHANCHOR paid for four times).
    script = pathlib.Path(work / "p4_scan.py")
    script.write_text(_P4_CHILD)
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "0"
    env.pop("TRANSFORMERS_OFFLINE", None)
    if "HF_TOKEN" not in env:
        tok = subprocess.run(["aws", "ssm", "get-parameter", "--name", "/pg-lineage/hf-token",
                              "--with-decryption", "--region", REGION,
                              "--query", "Parameter.Value", "--output", "text"],
                             capture_output=True, text=True)
        if tok.returncode == 0:
            env["HF_TOKEN"] = tok.stdout.strip()
    dest = work / "p4_scan.json"
    proc = subprocess.run([sys.executable, str(script), DATASET_REPO, str(N_DATA_FILES), str(dest)],
                          env=env, capture_output=True, text=True)
    print(proc.stdout[-4000:], flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"the P4 scan failed (rc={proc.returncode}): {proc.stderr[-2000:]}")
    result = json.loads(dest.read_text())
    bank.artifact("p4_scan.json", result)
    return {"selftest": {"planted_found": planted["planted"]["file_index_mismatches"],
                         "planted_absent": planted["planted"]["episodes_listed_but_absent"],
                         "clean_found": planted["clean"]["file_index_mismatches"]},
            **{k: v for k, v in result.items()
               if k not in ("absent_episodes", "file_index_mismatch_examples",
                            "span_mismatch_examples", "multi_file_examples", "per_file")}}


def _fetch_collection(work: pathlib.Path) -> Optional[Dict[str, object]]:
    """Every released allenai MolmoAct model's config, fetched by the cell that makes the claim.

    P1's sentence is about the collection, so the collection is enumerated here rather than taken from
    the seed: the cell that says "in all released Think fine-tunes" is the cell that read them. Runs in
    a child process for the same reason the P4 scan does — `HF_HUB_OFFLINE` is a module constant.
    """
    script = work / "collection.py"
    script.write_text(_COLLECTION_CHILD)
    dest = work / "molmoact2_configs.json"
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "0"
    if "HF_TOKEN" not in env:
        tok = subprocess.run(["aws", "ssm", "get-parameter", "--name", "/pg-lineage/hf-token",
                              "--with-decryption", "--region", REGION,
                              "--query", "Parameter.Value", "--output", "text"],
                             capture_output=True, text=True)
        if tok.returncode == 0:
            env["HF_TOKEN"] = tok.stdout.strip()
    proc = subprocess.run([sys.executable, str(script), str(dest)], env=env,
                          capture_output=True, text=True)
    print(proc.stdout[-3000:], flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"listing the collection failed (rc={proc.returncode}): "
                           f"{proc.stderr[-1500:]}")
    return json.loads(dest.read_text())


_COLLECTION_CHILD = '''\
"""Child: every released allenai MolmoAct model's config and normalization tags."""
import json, sys, datetime, pathlib
from huggingface_hub import HfApi, hf_hub_download
dest = pathlib.Path(sys.argv[1])
api = HfApi()
models = sorted({m.id for m in api.list_models(author="allenai", search="MolmoAct")})
out = {"listed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "n_models": len(models), "models": {}}
for mid in models:
    entry = {}
    try:
        info = api.model_info(mid)
        entry["sha"] = info.sha
        entry["last_modified"] = str(info.lastModified)
        entry["n_files"] = len(info.siblings or [])
    except Exception as exc:
        entry["info_error"] = f"{type(exc).__name__}: {exc}"
    for fname, key in (("config.json", "config"), ("norm_stats.json", "norm")):
        try:
            p = hf_hub_download(mid, fname, revision=entry.get("sha"))
            payload = json.loads(pathlib.Path(p).read_text())
            if key == "config":
                entry["architectures"] = payload.get("architectures")
                entry["gate"] = {k: v for k, v in payload.items()
                                 if "depth" in k.lower() or "gate" in k.lower()}
            else:
                entry["norm_tags"] = sorted((payload.get("metadata_by_tag") or payload).keys())
        except Exception as exc:
            entry[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
    out["models"][mid] = entry
    print("COLLECTION", mid, entry.get("gate"), file=sys.stderr, flush=True)
dest.write_text(json.dumps(out, indent=1))
print("COLLECTION_DONE", len(models))
'''


_P4_CHILD = '''\
"""Child process: fresh, ONLINE, pinned read of the dataset's four index columns."""
import json, sys, pathlib
sys.path.insert(0, "/opt/pg/repo")
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
import pyarrow.parquet as pq
from infra.pg import postcheck_dataset as pcd

repo, n_files, dest = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])
api = HfApi()
info = api.dataset_info(repo)
sha = info.sha
print("DATASET_REVISION", sha, flush=True)

meta_p = hf_hub_download(repo, pcd.META_EPISODES, repo_type="dataset", revision=sha)
info_p = hf_hub_download(repo, pcd.META_INFO, repo_type="dataset", revision=sha)
meta_tbl = pq.read_table(meta_p)
meta_rows = meta_tbl.to_pylist()
info_json = json.loads(pathlib.Path(info_p).read_text())
print("META", len(meta_rows), "episodes listed;", "columns:", meta_tbl.column_names, flush=True)

fs = HfFileSystem()
def opener(i):
    path = f"datasets/{repo}@{sha}/" + pcd.DATA_GLOB.format(i=i)
    return lambda: fs.open(path, "rb")

def on_progress(p):
    if p["files_done"] % 25 == 0 or p["files_done"] == p["n_files"]:
        print("SCAN", p, flush=True)

scan = pcd.scan_data_files(opener, n_files, on_progress=on_progress)
out = pcd.compare(meta_rows, scan, info=info_json)
out["dataset_revision"] = sha
out["meta_columns"] = meta_tbl.column_names
out["per_file"] = scan["per_file"]
dest.write_text(json.dumps(out, indent=1, default=str))
print("P4_DONE", json.dumps({k: out[k] for k in (
    "episodes_listed", "episodes_present", "rows_present", "file_index_mismatches",
    "from_index_or_length_mismatches", "episodes_listed_but_absent")}), flush=True)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/work/postcheck")
    ap.add_argument("--stages", default="p1,p2,p4")
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    bank = Bank(pathlib.Path(args.out), "postcheck_files")
    bank.top("args", vars(args))
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    if "p1" in stages:
        bank.stage("p1", stage_p1(bank, work))
    if "p2" in stages:
        bank.stage("p2", stage_p2(bank, work))
    if "p4" in stages:
        bank.stage("p4", stage_p4(bank, work))

    st = bank.payload["stages"]
    p1 = (st.get("p1") or {}).get("checkpoints", {})
    tuned = p1.get("allenai/MolmoAct2-Think-LIBERO", {})
    base = p1.get("allenai/MolmoAct2-Think", {})
    p4 = st.get("p4") or {}
    verdict = {
        "P1_tuned_gate_present": tuned.get("present"),
        "P1_tuned_n_layers": tuned.get("n_layers"),
        "P1_tuned_max_abs_gate_minus_init": tuned.get("max_abs_gate_minus_init"),
        "P1_tuned_claim_within_0_003": tuned.get("claim_within_0_003_of_sigmoid_init"),
        "P1_base_gate_present": base.get("present"),
        "P4_episodes_listed": p4.get("episodes_listed"),
        "P4_episodes_present": p4.get("episodes_present"),
        "P4_file_index_mismatches": p4.get("file_index_mismatches"),
        "P4_absent": p4.get("episodes_listed_but_absent"),
        "P4_span_mismatches": p4.get("from_index_or_length_mismatches"),
    }
    bank.top("verdict", verdict)
    print(json.dumps(verdict, indent=1), flush=True)


if __name__ == "__main__":
    main()
