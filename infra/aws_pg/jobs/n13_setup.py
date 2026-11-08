"""N13 setup step (studies/next/07_PICKS/N13_SPEC.md §9.4 S2 step 1, §9.2 n13_setup, §4.1, §6.1 G-GRADER, G-TEMPLATE, G-INIT (a)).

    python -m infra.aws_pg.jobs.n13_setup --pair pair1 --models base,rl [--olmes] --out <dir>

Order (out/metrics.json rewritten after every stage; status running -> done):
  1. /opt/n13: python3 -m venv, pip install vllm==0.32.0 math-verify==0.9.0; pip freeze banked; versions asserted.
  2. /opt/olmes-venv (with --olmes): git clone allenai/olmes to /opt/olmes, checkout 5a51f502…, pip install -e ".[gpu]"; pip freeze banked (olmes pulls
     ai2-olmo from a git main); vllm 0.11.0, lm_eval 0.4.3, antlr4-python3-runtime 4.11 asserted; the Minerva task config read back
     from the installed package (n13_known.olmes_config) and compared with n13_known.EXPECTED (§0.3 item 1 at the pinned commit).
  3. Data: MATH-500 test.jsonl and GSM8K test parquet at the pinned commits; MATH-500 checked by git blob sha1 against the HF tree API
     (it is not an LFS file), GSM8K by its LFS sha256; row counts asserted; problems.json banks every row and its sha256.
  4. G-GRADER under /opt/n13 (math-verify as installed there), both grading rules recorded; the active rule must pass.
  5. Weights: snapshot_download at the pinned commits; every .safetensors sha256 checked against the LFS oid of the HF tree API read
     now; the base shards also against the logged 3.1 init revision 39ff1a89… (G-INIT (a)), with stage2-step47684's oids as the
     failing input.
  6. G-TEMPLATE under /opt/n13 with the RL tokenizer: the §2.2 bytes for every MATH-500 problem; prompt ids banked in prompt_ids.json.
Writes setup.json (paths, hashes, versions, gate records). Fail-loud: no try/except, no fallback.
--local runs the same stages on local files without venvs or network (the offline gate).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

from infra.pg import n13_arms as A
from infra.pg import n13_score as S

N13_PY, OLMES_DIR, OLMES_VENV = "/opt/n13/bin/python", "/opt/olmes", "/opt/olmes-venv"
OLMES_PY = OLMES_VENV + "/bin/python"
PINS_N13 = {"vllm": "0.32.0", "math-verify": "0.9.0"}
OLMES_REPO, OLMES_COMMIT = "https://github.com/allenai/olmes", "5a51f502d463b8cdc4a2dcad7d7096c41ff1197e"
PINS_OLMES = {"vllm": "0.11.0", "lm_eval": "0.4.3", "antlr4-python3-runtime": "4.11"}
# olmes leaves pyarrow unpinned and pins numpy<2. pyarrow 26.0.0 (2026-10-09) added an import-time check for NumPy >= 2
# (python/pyarrow/lib.pyx at apache-arrow-26.0.0; absent at 25.0.1), so `import datasets` failed in s2setup-04. 23.0.1 is the last
# pyarrow release before the olmes commit (2026-03-24).
OLMES_EXTRA = ("pyarrow==23.0.1",)
MODELS = {
    "base": ("allenai/Olmo-3-1025-7B", "a81bae42db3975be1671e27b9c9a56da1a9f980f"),
    "stage2": ("allenai/Olmo-3-1025-7B", "c3c800dc900f3fecc112e8cd6b2a13edabc096e1"),
    "rl": ("allenai/Olmo-3.1-7B-RL-Zero-Math", "364820b002fc8e81d11a81c7fc9dd6b49c83cd05"),
    "rl30": ("allenai/Olmo-3-7B-RL-Zero-Math", "dee26a4074b88215cd1326472e1ab14d0259d892"),
}
LOGGED_INIT = ("allenai/Olmo-3-1025-7B", "39ff1a89dc6950c25f4a435a993904f8587c6a3a")
MATH500 = ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test.jsonl", 500)
GSM8K = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/test-00000-of-00001.parquet", 1319)
HF = "https://huggingface.co"


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dump(path, obj):
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def sh(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=True, **kw)


def hf_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {})
    return urllib.request.urlopen(req, timeout=60).read()


def tree(repo: str, rev: str, kind: str = "models") -> list:
    return json.loads(hf_get(f"{HF}/api/{kind}/{repo}/tree/{rev}?recursive=1"))


def lfs_oids(tree_rows: list) -> dict:
    return {r["path"]: r["lfs"]["oid"] for r in tree_rows if r["path"].endswith(".safetensors")}


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def canon(name: str) -> str:
    return name.lower().replace("_", "-")


def freeze(py: str) -> dict:
    out = subprocess.run([py, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    return {canon(k): v for k, v in (l.split("==", 1) for l in out.splitlines() if "==" in l)}, out


# ------------------------------------------------------------------------------------------------ stages
def python_headers() -> str:
    """The Python 3.12 headers. The AMI has none. olmes' pytrec_eval build needs them (s2setup-01), and so does the n13 engine: at
    start-up torch inductor's Triton backend compiles its CUDA driver stub (cuda_utils.c, #include <Python.h>) with gcc (s2smoke-08
    smoke_base: "fatal error: Python.h: No such file or directory"; s2setup-07 passed this stage only because the olmes stage had
    installed the headers first)."""
    sh(["bash", "-lc", "apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y python3.12-dev"])
    inc = subprocess.run(["python3", "-c", "import sysconfig; print(sysconfig.get_paths()['include'])"], check=True, capture_output=True,
                         text=True).stdout.strip()
    assert pathlib.Path(inc, "Python.h").exists(), ("no Python.h after installing python3.12-dev", inc)
    return inc


def venv_n13(out: pathlib.Path) -> dict:
    inc = python_headers()                         # before the engine's first start (s2smoke-08)
    sh(["python3", "-m", "venv", "/opt/n13"])
    sh([N13_PY, "-m", "pip", "install", "--no-cache-dir", *[f"{k}=={v}" for k, v in PINS_N13.items()]])
    pins, text = freeze(N13_PY)
    (out / "pip_freeze_n13.txt").write_text(text)
    for k, v in PINS_N13.items():
        assert pins.get(canon(k)) == v, (k, pins.get(canon(k)), v)
    return {**{k: pins.get(canon(k)) for k in ("vllm", "math-verify", "torch", "transformers", "latex2sympy2_extended", "antlr4-python3-runtime",
                                                "sympy")}, "python_include": inc}


def venv_olmes(out: pathlib.Path) -> dict:
    sh(["git", "clone", "--quiet", OLMES_REPO, OLMES_DIR])
    sh(["git", "-C", OLMES_DIR, "checkout", "--quiet", OLMES_COMMIT])
    head = subprocess.run(["git", "-C", OLMES_DIR, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    assert head == OLMES_COMMIT, head
    sh(["python3", "-m", "venv", OLMES_VENV])
    py = OLMES_PY
    # olmes needs pytrec_eval, which has no wheel and compiles a C extension; the AMI has no Python.h (s2setup-01 failed here).
    # venv_n13 runs first and installs the headers (python_headers); checked again here.
    inc = subprocess.run(["python3", "-c", "import sysconfig; print(sysconfig.get_paths()['include'])"], check=True, capture_output=True,
                         text=True).stdout.strip()
    assert pathlib.Path(inc, "Python.h").exists(), ("no Python.h: venv_n13 must run before venv_olmes", inc)
    sh([py, "-m", "pip", "install", "--no-cache-dir", "-e", f"{OLMES_DIR}[gpu]", *OLMES_EXTRA])
    pins, text = freeze(py)
    (out / "pip_freeze_olmes.txt").write_text(text)
    from infra.aws_pg.jobs import n13_known as K
    for k, v in PINS_OLMES.items():
        assert K.release(pins[canon(k)]) == K.release(v), (k, pins.get(canon(k)), v)
    assert pins[canon("pyarrow")] == OLMES_EXTRA[0].split("==")[1], pins[canon("pyarrow")]
    sh([py, "-c", "import oe_eval.run_eval"])     # the module olmes' launcher runs: every import it needs, before the GPU step
    cfg = json.loads(subprocess.run([py, "-m", "infra.aws_pg.jobs.n13_known", "--print-olmes-config"], check=True, capture_output=True,
                                    text=True, cwd="/opt/pg/repo").stdout)
    chk = K.check_config(cfg)
    assert chk["pass"], chk
    return {"commit": head, "pins": {k: pins.get(canon(k)) for k in ("vllm", "lm_eval", "antlr4-python3-runtime", "torch", "transformers")},
            "config_check": chk}


def data(out: pathlib.Path, local_math=None, local_gsm=None) -> dict:
    repo, rev, path, n = MATH500
    if local_math:
        raw, t = pathlib.Path(local_math).read_bytes(), None
    else:
        raw = hf_get(f"{HF}/datasets/{repo}/resolve/{rev}/{path}")
        t = {r["path"]: r for r in tree(repo, rev, "datasets")}[path]
        assert git_blob_sha1(raw) == t["oid"] and len(raw) == t["size"], ("MATH-500 file", git_blob_sha1(raw), t)
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    assert local_math or len(rows) == n, len(rows)
    math = [{"index": i, "problem": r["problem"], "answer": r["answer"], "unique_id": r["unique_id"], "row_sha256": S.row_sha256(r)}
            for i, r in enumerate(rows)]
    repo, rev, path, n = GSM8K
    if local_gsm:
        g_rows = json.loads(pathlib.Path(local_gsm).read_text())
        g_sha = hashlib.sha256(pathlib.Path(local_gsm).read_bytes()).hexdigest()
    else:
        rawp = hf_get(f"{HF}/datasets/{repo}/resolve/{rev}/{path}")
        t = {r["path"]: r for r in tree(repo, rev, "datasets")}[path]
        g_sha = hashlib.sha256(rawp).hexdigest()
        assert g_sha == t["lfs"]["oid"], ("GSM8K parquet", g_sha, t)
        import io
        import pyarrow.parquet as pq
        g_rows = pq.read_table(io.BytesIO(rawp)).to_pylist()
        assert len(g_rows) == n, len(g_rows)
    gsm = [{"index": i, "problem": r["question"], "answer": S.gsm8k_gold(r["answer"]), "unique_id": f"gsm8k/test/{i}",
            "row_sha256": S.row_sha256(r)} for i, r in enumerate(g_rows)]
    rec = {"math500": math, "gsm8k": gsm, "math500_file_sha256": hashlib.sha256(raw).hexdigest(), "gsm8k_file_sha256": g_sha,
           "math500_source": MATH500[:3], "gsm8k_source": GSM8K[:3]}
    dump(out / "problems.json", rec)
    return {"math500_rows": len(math), "gsm8k_rows": len(gsm), "math500_file_sha256": rec["math500_file_sha256"], "gsm8k_file_sha256": g_sha}


def weights(out: pathlib.Path, names: list, local=None) -> dict:
    snaps, hashes, checks = {}, {}, {}
    for nm in names:
        repo, rev = MODELS[nm]
        if local:
            snaps[nm] = str(local[nm])
            hashes[nm] = {p.name: A.file_sha256(p) for p in sorted(pathlib.Path(local[nm]).glob("*.safetensors"))}
            continue
        from huggingface_hub import snapshot_download
        from infra.pg.mercator_core import retry_transport
        d, n_retry = retry_transport(lambda: snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.jinja",
                                                                                                   "*.txt"], max_workers=8),
                                     what=f"{repo}@{rev}")
        snaps[nm] = d
        oids = lfs_oids(tree(repo, rev))
        hashes[nm] = {p: A.file_sha256(pathlib.Path(d) / p) for p in sorted(oids)}
        bad = [p for p in oids if hashes[nm][p] != oids[p]]
        assert not bad, (f"{repo}@{rev}: sha256 differs from the LFS oid", bad)
        checks[nm] = {"n_shards": len(oids), "retries": n_retry, "pass": True}
    return {"snapshots": snaps, "shard_sha256": hashes, "checks": checks}


def g_init_a(hashes_base: dict, init_oids: dict, stage2_oids: dict) -> dict:
    g = A.g_init_hash(hashes_base, init_oids)
    g["failing_input"] = {"what": "stage2-step47684's shard oids in place of base main's", "failed": not A.g_init_hash(stage2_oids, init_oids)["pass"]}
    return g


# ------------------------------------------------------------------------------------------------ the parts run under /opt/n13
def part_grader(problems_json: str, out_json: str):
    golds = [r["answer"] for r in json.loads(pathlib.Path(problems_json).read_text())["math500"]]
    import math_verify
    rec = {"active_rule": S.GRADE_RULE, "math_verify": getattr(math_verify, "__version__", None),
           "rules": {r: S.g_grader(golds, r) for r in ("registered", "dollar_both", "boxed")}}
    rec["pass"] = rec["rules"][S.GRADE_RULE]["pass"] and rec["rules"][S.GRADE_RULE]["failing_input"]["failed"]
    dump(out_json, rec)


def part_template(problems_json: str, tok_dir: str, pair: str, out_json: str, ids_json: str):
    from transformers import AutoTokenizer
    import transformers
    tok = AutoTokenizer.from_pretrained(tok_dir)
    probs = [r["problem"] for r in json.loads(pathlib.Path(problems_json).read_text())["math500"]]
    g, ids = S.g_template(tok, probs, pair)
    g["transformers"] = transformers.__version__
    g["pass_all"] = g["pass"] and g["failing_input"]["failed"]
    dump(ids_json, {"pair": pair, "ids": ids})
    dump(out_json, g)


def run_part(py: str, *args):
    sh([py, "-m", "infra.aws_pg.jobs.n13_setup", *args])


# ------------------------------------------------------------------------------------------------ main
def run(a):
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = {"step": "setup", "status": "running", "started_utc": utc(), "pair": a.pair, "models": a.models, "stages": {}}
    w = lambda: dump(out / "metrics.json", {**m, "updated_utc": utc()})
    w()
    local = a.local
    py = sys.executable if local else N13_PY
    if not local:
        m["stages"]["venv_n13"] = venv_n13(out)
        w()
        if a.olmes:
            m["stages"]["venv_olmes"] = venv_olmes(out)
            w()
    m["stages"]["data"] = data(out, a.local_math500, a.local_gsm8k)
    w()
    run_part(py, "--part", "grader", "--problems", out / "problems.json", "--part-out", out / "grader.json")
    m["stages"]["G-GRADER"] = json.loads((out / "grader.json").read_text())
    m["stages"]["G-GRADER"] = {k: m["stages"]["G-GRADER"][k] for k in ("active_rule", "pass")} | {
        r: {k: v[k] for k in ("pass", "planted_eq_true", "planted_neq_false", "self_equal", "derangement_rate", "identical_normalised_rate")}
        for r, v in m["stages"]["G-GRADER"]["rules"].items()}
    w()
    assert m["stages"]["G-GRADER"]["pass"], ("G-GRADER failed", m["stages"]["G-GRADER"])
    loc = None
    if local:
        loc = dict(kv.split("=", 1) for kv in a.local_snapshots.split(","))
    names = a.models.split(",")
    wt = weights(out, names, loc)
    m["stages"]["weights"] = {"snapshots": wt["snapshots"], "checks": wt["checks"]}
    w()
    if "base" in names:
        if local:
            init_oids = json.loads(pathlib.Path(a.local_init_oids).read_text())
        else:
            init_oids, s2 = lfs_oids(tree(*LOGGED_INIT)), lfs_oids(tree(*MODELS["stage2"]))
            init_oids = {"init": init_oids, "stage2": s2}
        m["stages"]["G-INIT_a"] = g_init_a(wt["shard_sha256"]["base"], init_oids["init"], init_oids["stage2"])
        w()
        assert m["stages"]["G-INIT_a"]["pass"] and m["stages"]["G-INIT_a"]["failing_input"]["failed"], m["stages"]["G-INIT_a"]
    tok_dir = wt["snapshots"]["rl"]
    run_part(py, "--part", "template", "--problems", out / "problems.json", "--tok-dir", tok_dir, "--pair", a.pair,
             "--part-out", out / "template.json", "--ids-out", out / "prompt_ids.json")
    m["stages"]["G-TEMPLATE"] = json.loads((out / "template.json").read_text())
    w()
    assert m["stages"]["G-TEMPLATE"]["pass_all"], ("G-TEMPLATE failed", m["stages"]["G-TEMPLATE"])
    setup = {"pair": a.pair, "snapshots": wt["snapshots"], "shard_sha256": wt["shard_sha256"], "tokenizer_dir": tok_dir,
             "problems": str(out / "problems.json"), "prompt_ids": str(out / "prompt_ids.json"), "grade_rule": S.GRADE_RULE,
             "n13_python": py, "olmes_python": None if local else (OLMES_PY if a.olmes else None)}
    dump(out / "setup.json", setup)
    m["status"] = "done"
    m["finished_utc"] = utc()
    w()


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--pair", default="pair1", choices=sorted(S.TEMPLATE))
    ap.add_argument("--models", default="base,rl")
    ap.add_argument("--olmes", action="store_true")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--local-snapshots")
    ap.add_argument("--local-math500")
    ap.add_argument("--local-gsm8k")
    ap.add_argument("--local-init-oids")
    ap.add_argument("--part", choices=("grader", "template"))
    ap.add_argument("--problems")
    ap.add_argument("--tok-dir")
    ap.add_argument("--part-out")
    ap.add_argument("--ids-out")
    return ap


def main():
    a = parser().parse_args()
    if a.part == "grader":
        return part_grader(a.problems, a.part_out)
    if a.part == "template":
        return part_template(a.problems, a.tok_dir, a.pair, a.part_out, a.ids_out)
    if not a.local:
        from infra.pg.humboldt_core import hf_setup
        hf_setup()
    run(a)


if __name__ == "__main__":
    main()
