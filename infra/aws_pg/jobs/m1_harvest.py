"""M1 job (studies/next/07_PICKS/M1_SPEC.md §9): one checkpoint per call through the banked grid's harvest, unchanged.

Modes
  harvest  Download one registered branch: only its registered weight files (the mix branches also carry the three
           bf16 shards of `main`; they are never fetched), each sha256-checked against its LFS oid (G-SHARD). For the
           mix ends, write a local copy of config.json with `olmo2-retrofit` / `Olmo2RetrofitForCausalLM` rewritten
           to `olmo3` / `Olmo3ForCausalLM` (G-CFG); every other file is a symlink into the HF cache. Load through
           `neurons_harvest.load_model`, which passes `dtype=torch.bfloat16`: that casts the fp32 mix shards on load,
           so no separate cast step is written; G-CAST reads registered tensors back from the shard files and
           requires the loaded bf16 tensor to equal `file_tensor.to(torch.bfloat16)` bit for bit. Then
           `neurons_harvest.run` with the grid's arguments (the 640 grid poles, T_15's positions, 20,570,743 tokens,
           window 2,048, batch 8, snapshot band 1), which writes metrics.json every 50 batches inside its loop. Then
           m1/counts.npz (per pole and corpus directory: the pole's filled top-50 slots there), m1/quantiles.npz and
           the positive control (per-window corpus NLL on 32 code+math and 32 web windows). Weights are deleted after.
  known    The known number K0: the same path on one banked grid point over the first 2^20 corpus tokens, compared
           in-job with the bank's own files (exemplar activations at the bank's positions inside that prefix; the
           rule that a global top-50 position inside the prefix is a prefix top-50 position). Each comparison also
           runs on its registered failing input, which must fail. Exits non-zero unless all of that holds.
  refs     m1/counts.npz of banked harvests (their topk/ on S3), for the references frozen before any mix is read.

    python -m infra.aws_pg.jobs.m1_harvest --mode harvest --branch stage1-step476838 \
        --corpus <corpus uri> --inputs <inputs uri> --out <dir>

m1/metrics.json is rewritten at every phase; the harvest's own metrics.json is rewritten inside its loop.
Fail-loud: no try/except, no fallback, no retry.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import time

import numpy as np

from infra.aws_pg.jobs.neurons_harvest import fetch, fetch_corpus
from infra.pg import m1

HUB = None          # snapshot_download; the dry run replaces it with a local stand-in
WORK = pathlib.Path(os.environ.get("M1_WORK", "/work/m1"))


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def status(out: pathlib.Path, rec: dict, phase: str):
    rec["phase"], rec["updated_utc"] = phase, utc()
    from infra.pg.neurons_harvest import dump
    dump(out / "m1" / "metrics.json", rec)
    print(f"[m1] {phase}", flush=True)


def fetch_inputs(uri: str, names: list, reg: dict) -> dict:
    """The laptop bank copies uploaded before launch; each must have the sha256 of the laptop file."""
    paths = {}
    for n in names:
        p = fetch(f"{uri.rstrip('/')}/{n}", f"inputs/{n}")
        got = m1.sha256_file(p)
        assert got == reg["bank_sha"][n], f"input {n}: sha256 {got} != registered {reg['bank_sha'][n]}"
        paths[n] = p
    return paths


def grid_poles(contexts: pathlib.Path) -> tuple[list, np.ndarray]:
    """The grid's pole list and T_15's top-50 positions, in the order of the banked T_15 contexts.jsonl (the order
    neurons_harvest writes them: the --poles order)."""
    poles, pos = [], []
    for line in contexts.read_text().splitlines():
        r = json.loads(line)
        poles.append(r["pole"])
        t = [c["tok"] for c in r["topk"]]
        pos.append(t + [-1] * (50 - len(t)))
    return poles, np.array(pos, np.int64)


def download(branch: str, entry: dict, reg: dict, out: pathlib.Path) -> tuple[pathlib.Path, dict]:
    from infra.pg.neurons_harvest import dump
    hub = HUB
    if hub is None:
        from huggingface_hub import snapshot_download as hub
    allow = list(reg["nonweight"]) + sorted(entry["weights"])
    t0 = time.time()
    local = pathlib.Path(hub(m1.REPO, revision=branch, allow_patterns=allow))
    dt = time.time() - t0
    n = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
    rec = {"branch": branch, "commit": local.name, "allow_patterns": allow, "bytes": n, "seconds": round(dt, 1),
           "MB_per_s": round(n / 1e6 / max(dt, 1e-9), 1), "xet_disabled": os.environ.get("HF_HUB_DISABLE_XET")}
    assert local.name == entry["commit"], f"G-REV: snapshot {local.name} != registered commit {entry['commit']}"
    t0 = time.time()
    found = {f.name: m1.sha256_file(f) for f in sorted(local.glob("*.safetensors"))}
    index = set(json.loads((local / "model.safetensors.index.json").read_text())["weight_map"].values())
    rec["sha256_s"] = round(time.time() - t0, 1)
    g = m1.shard_gate(found, index, entry, reg["stale"])
    g["failing_input"] = m1.shard_gate(dict(reg["stale"]), index, entry, reg["stale"])   # main's bf16 shards
    rec["G_SHARD"] = g
    dump(out / "m1" / "download.json", rec)
    assert g["pass"], f"G-SHARD failed: {g}"
    assert not g["failing_input"]["pass"], "G-SHARD: its failing input (the stale shards of main) passed"
    print(f"[m1] download {branch}: {n / 1e9:.2f} GB in {dt:.0f} s ({rec['MB_per_s']} MB/s); shards verified", flush=True)
    return local, rec


def load_dir(local: pathlib.Path, entry: dict, reg: dict) -> tuple[pathlib.Path, dict]:
    """The directory the model is loaded from. For a mix end: a new directory whose config.json is the rewritten
    copy and whose other files are symlinks to the snapshot (the cache blob of config.json is never written)."""
    cfg = json.loads((local / "config.json").read_text())
    rec = {"rewrite": entry["rewrite"], "as_released": {k: cfg.get(k) for k in ("model_type", "architectures",
                                                                                "sliding_window", "torch_dtype")}}
    rec["failing_input"] = m1.config_gate(cfg, reg["ref_config"]) if entry["rewrite"] else \
        m1.config_gate({**cfg, "model_type": "olmo2-retrofit", "architectures": ["Olmo2RetrofitForCausalLM"]},
                       reg["ref_config"])
    if not entry["rewrite"]:
        rec["G_CFG"] = m1.config_gate(cfg, reg["ref_config"])
        assert rec["G_CFG"]["pass"] and rec["G_CFG"]["diff_keys"] == [], f"G-CFG failed: {rec['G_CFG']}"
        assert not rec["failing_input"]["pass"], "G-CFG: its failing input passed"
        return local, rec
    new = m1.rewrite_config(cfg)
    rec["G_CFG"] = m1.config_gate(new, reg["ref_config"])
    assert rec["G_CFG"]["pass"], f"G-CFG failed: {rec['G_CFG']}"
    assert not rec["failing_input"]["pass"], "G-CFG: the unrewritten config passed"
    d = WORK / "load" / local.name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in local.iterdir():
        if f.name != "config.json":
            (d / f.name).symlink_to(f.resolve())
    (d / "config.json").write_text(json.dumps(new, indent=2))
    rec["load_dir"] = str(d)
    return d, rec


def cast_check(model, src: pathlib.Path, names: list) -> dict:
    """G-CAST: every parameter is bf16, and each registered tensor equals the file's tensor cast with
    torch's round-to-nearest-even `.to(torch.bfloat16)`. Failing input: the same comparison against the next
    layer's tensor of the same name (or, for a non-layer tensor, the first layer's norm of the same width).
    For fp32 files the mismatch share of a truncating cast is recorded as well."""
    import torch
    from safetensors import safe_open
    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    params = dict(model.named_parameters())
    dtypes = sorted({str(p.dtype) for p in params.values()})
    rows, fail_rows = [], []

    def file_tensor(name):
        with safe_open(str(src / wm[name]), framework="pt") as f:
            return f.get_tensor(name)

    def other(name):
        if name.startswith("model.layers."):
            parts = name.split(".")
            parts[2] = str((int(parts[2]) + 1) % model.config.num_hidden_layers)
            return ".".join(parts)
        return "model.layers.0.post_feedforward_layernorm.weight"

    for name in names:
        t = file_tensor(name)
        p = params[name].detach().cpu()
        rne = t.to(torch.bfloat16)
        row = {"name": name, "file_dtype": str(t.dtype), "shape": list(t.shape),
               "equal_rne": bool(torch.equal(rne.view(torch.int16), p.view(torch.int16)))}
        if t.dtype == torch.float32:
            trunc = (t.view(torch.int32) >> 16).to(torch.int16)
            row["trunc_mismatch_share"] = float((trunc != p.view(torch.int16)).float().mean())
        rows.append(row)
        o = file_tensor(other(name)).to(torch.bfloat16)
        fail_rows.append({"name": name, "against": other(name),
                          "equal": bool(o.shape == p.shape and torch.equal(o.view(torch.int16), p.view(torch.int16)))})
    ok = dtypes == ["torch.bfloat16"] and all(r["equal_rne"] for r in rows)
    return {"pass": bool(ok), "param_dtypes": dtypes, "tensors": rows,
            "failing_input": {"pass": bool(all(r["equal"] for r in fail_rows)), "tensors": fail_rows}}


def counts_and_quantiles(src: pathlib.Path, dst: pathlib.Path, corpus: dict, n_layers: int,
                         quantiles: bool = True) -> dict:
    """dst/m1/counts.npz from src/topk/layer_LL.npz `tok`; dst/m1/quantiles.npz from src/stats/ when asked."""
    man = corpus["manifest"]
    docdir, grp = m1.doc_dirs(man), m1.groups(man)
    n_dirs = len(grp)
    c, f, q = [], [], {k: [] for k in ("q99", "q999", "q9999", "max_abs", "density")}
    for lyr in range(n_layers):
        cl, fl = m1.pole_counts(np.load(src / "topk" / f"layer_{lyr:02d}.npz")["tok"], corpus["doc_offsets"],
                                docdir, n_dirs)
        c.append(cl)
        f.append(fl)
        if quantiles:
            st = np.load(src / "stats" / f"layer_{lyr:02d}.npz")
            for k in q:
                q[k].append(st[k])
    (dst / "m1").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst / "m1" / "counts.npz", counts=np.stack(c), filled=np.stack(f), cm_mask=m1.cm_mask(man),
                        dirs=np.array(man["dirs"]), groups=np.array(grp))
    if quantiles:
        np.savez_compressed(dst / "m1" / "quantiles.npz", **{k: np.stack(v) for k, v in q.items()})
    return {"n_dirs": n_dirs, "cm_dirs": [d for d, g in zip(man["dirs"], grp) if g in ("code", "math")],
            "min_filled": int(np.stack(f).min())}


def slot_samples(out: pathlib.Path, corpus: dict) -> list:
    """One top-50 slot counted as code+math and one counted as other, with the decoded context, from the
    harvest's own samples/contexts.jsonl (the 640 grid poles): the first such slot in file order."""
    man = corpus["manifest"]
    docdir, grp, dirs = m1.doc_dirs(man), m1.groups(man), man["dirs"]
    picked = {}
    for line in (out / "samples" / "contexts.jsonl").read_text().splitlines():
        r = json.loads(line)
        for c in r["topk"]:
            d = int(docdir[c["doc"]])
            kind = "code+math" if grp[d] in ("code", "math") else "other"
            if kind not in picked:
                picked[kind] = {"counted_as": kind, "pole": r["pole"], "rank": c["rank"], "tok": c["tok"], "a": c["a"],
                                "doc": c["doc"], "dir": dirs[d], "before": c["before"], "token": c["token"],
                                "after": c["after"]}
        if len(picked) == 2:
            break
    assert len(picked) == 2, f"samples: found only {sorted(picked)}"
    return [picked["code+math"], picked["other"]]


def pc_nll(model, corpus: dict, windows: dict, bos, window: int, batch: int, device: str) -> dict:
    """Mean next-token NLL per window, the harvest's window rule: BOS prepended when defined, targets
    input[:, 1:] from logits[:, :-1] (every corpus token of the window is a target)."""
    import torch
    import torch.nn.functional as F
    from infra.pg.neurons_harvest import forward
    toks = corpus["tokens"]
    res = {}
    with torch.inference_mode():
        for g in ("cm", "web"):
            vals = []
            ws = windows[g]
            for i in range(0, len(ws), batch):
                rows = [toks[w * window:(w + 1) * window].astype(np.int64) for w in ws[i:i + batch]]
                x = torch.from_numpy(np.stack(rows))
                if bos is not None:
                    x = torch.cat([torch.full((x.shape[0], 1), bos, dtype=torch.long), x], 1)
                x = x.to(device)
                lg = forward(model, x, True).logits
                for r in range(x.shape[0]):
                    vals.append(float(F.cross_entropy(lg[r, :-1].float(), x[r, 1:]).item()))
            res[g] = vals
    return res


def run_harvest(a, reg: dict, out: pathlib.Path, rec: dict, known: bool):
    import torch
    from infra.pg import neurons_corpus as NC
    from infra.pg import neurons_harvest as NH
    entry = reg["branches"][a.branch]
    same = reg["same"]
    status(out, rec, "inputs")
    names = ["t15_contexts.jsonl"] + ([a.bank_exemplar, a.fail_exemplar, a.bank_contexts] if known else [])
    inp = fetch_inputs(a.inputs, list(dict.fromkeys(names)), reg)
    poles, positions = grid_poles(inp["t15_contexts.jsonl"])
    assert len(poles) == len(set(poles)) == a.n_poles_grid and (positions >= 0).all(), "grid poles/positions"
    corpus = NC.load_tok(fetch_corpus(a.corpus, same["tokname"]), same["tokname"])
    assert corpus["manifest_sha256"] == same["corpus_manifest_sha256"], "G-CORPUS: corpus manifest sha256"
    assert corpus["tokens_sha256"] == same["tokens_sha256"], "G-CORPUS: tokens sha256"
    max_tokens = reg["k0_tokens"] if known else same["max_tokens"]
    bands = [] if known else list(same["snapshot_bands"])
    if known:
        positions = np.where(positions < max_tokens, positions, -1)
    status(out, rec, "download")
    local, rec["download"] = download(a.branch, entry, reg, out)
    src, rec["config"] = load_dir(local, entry, reg)
    status(out, rec, "load")
    t0 = time.time()
    model, tok = NH.load_model(str(src), same["model_key"], device=a.device)
    rec["load_s"] = round(time.time() - t0, 1)
    rec["G_CAST"] = m1_cast = cast_check(model, src, reg["cast_names"])
    assert m1_cast["pass"], f"G-CAST failed: {m1_cast}"
    assert not m1_cast["failing_input"]["pass"], "G-CAST: its failing input passed"
    status(out, rec, "harvest")
    extra = {"args": vars(a), "device_name": torch.cuda.get_device_name(0) if a.device == "cuda" else a.device,
             "pg_git_sha": os.environ.get("PG_GIT_SHA"), "download": rec["download"], "load_s": rec["load_s"],
             "m1": {"config": rec["config"], "G_CAST": rec["G_CAST"]}}
    man = NH.run(out, model=model, tok=tok, corpus=corpus, model_key=same["model_key"], repo=m1.REPO,
                 revision=local.name, branch=a.branch, max_tokens=max_tokens, window=same["window"],
                 batch=same["batch"], snapshot_bands=bands, poles=poles, positions=positions, device=a.device,
                 extra_manifest=extra)
    status(out, rec, "after_harvest")
    if known:
        k0(out, rec, inp, a, positions, max_tokens)
    else:
        rec["counts"] = counts_and_quantiles(out, out, corpus, man["n_layers"])
        rec["samples"] = slot_samples(out, corpus)
        docdir, grp = m1.doc_dirs(corpus["manifest"]), m1.groups(corpus["manifest"])
        rec["token_share_per_dir"] = m1.token_share_per_dir(corpus["doc_offsets"], docdir, len(grp),
                                                            man["n_tokens_windowed"])
        win = m1.pc_windows(corpus["doc_offsets"], docdir, grp, same["window"], man["n_windows"], k=reg["pc_k"])
        rec["pc_nll"] = {"windows": win, "nll": pc_nll(model, corpus, win, tok.bos_token_id, same["window"],
                                                       same["batch"], a.device)}
        rec["fit"] = m1.fit_gate(json.loads((out / "metrics.json").read_text()),
                                 json.loads((out / "controls.json").read_text()), man, same, reg["nll_band"])
        assert rec["fit"]["pass"], f"G-FIT/G-SAME failed: {rec['fit']}"
    rec["gates_pass"] = True
    del model
    if a.device == "cuda":
        torch.cuda.empty_cache()
    repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / ("models--" + m1.REPO.replace("/", "--"))
    shutil.rmtree(repo_dir)
    if src != local:
        shutil.rmtree(src)
    rec["deleted_after_harvest"] = str(repo_dir)
    status(out, rec, "done")


def k0(out: pathlib.Path, rec: dict, inp: dict, a, positions: np.ndarray, n_tokens: int):
    mask = positions >= 0
    new = np.load(out / "poles" / "exemplar_a.npz")["a"]
    bank = np.load(inp[a.bank_exemplar])["a"]
    fail = np.load(inp[a.fail_exemplar])["a"]
    g_val = m1.k0_values_gate(new, bank, mask)
    g_val["failing_input"] = m1.k0_values_gate(fail, bank, mask)
    _, bank_top = grid_poles(inp[a.bank_contexts])
    poles = json.loads((out / "poles" / "poles.json").read_text())
    new_top = []
    cache = {}
    for p in poles:
        lyr, n, s = (int(p[1:3]), int(p[4:-1]), "+-".index(p[-1]))
        if lyr not in cache:
            cache[lyr] = np.load(out / "topk" / f"layer_{lyr:02d}.npz")["tok"]
        new_top.append(cache[lyr][n, s])
    new_top = np.stack(new_top)
    g_sub = m1.k0_subset_gate(new_top, bank_top, n_tokens)
    shifted = np.where(bank_top >= 0, bank_top + 1, -1)
    g_sub["failing_input"] = m1.k0_subset_gate(new_top, shifted, n_tokens)
    worst = np.unravel_index(np.argmax(np.where(mask, m1.rel_diff(new, bank), -1)), mask.shape)
    best = np.unravel_index(np.argmin(np.where(mask, m1.rel_diff(new, bank), np.inf)), mask.shape)
    rec["K0"] = {"values": g_val, "subset": g_sub, "n_tokens": n_tokens, "bank": a.bank_exemplar,
                 "failing_input_exemplar": a.fail_exemplar,
                 "samples": [{"which": w, "pole": poles[i], "slot": int(j), "tok": int(positions[i, j]),
                              "a_bank": float(bank[i, j]), "a_new": float(new[i, j])}
                             for w, (i, j) in (("closest", best), ("farthest", worst))],
                 "pass": bool(g_val["pass"] and g_sub["pass"] and not g_val["failing_input"]["pass"]
                              and not g_sub["failing_input"]["pass"])}
    status(out, rec, "k0_compared")
    assert rec["K0"]["pass"], f"K0 failed: {json.dumps(rec['K0'])[:2000]}"


def run_refs(a, reg: dict, out: pathlib.Path, rec: dict):
    from infra.pg import neurons_corpus as NC
    same = reg["same"]
    corpus = NC.load_tok(fetch_corpus(a.corpus, same["tokname"]), same["tokname"])
    assert corpus["manifest_sha256"] == same["corpus_manifest_sha256"], "G-CORPUS: corpus manifest sha256"
    rec["refs"] = {}
    for spec in a.refs.split(","):
        name, uri = spec.split("=", 1)
        status(out, rec, f"refs:{name}")
        d = out / name
        (d / "m1").mkdir(parents=True, exist_ok=True)
        bman = json.loads(fetch(f"{uri.rstrip('/')}/manifest.json", f"refs/{name}/manifest.json").read_text())
        bctl = json.loads(fetch(f"{uri.rstrip('/')}/controls.json", f"refs/{name}/controls.json").read_text())
        assert bman["tokens_sha256"] == same["tokens_sha256"] and \
            bman["corpus_manifest_sha256"] == same["corpus_manifest_sha256"], f"ref {name} is not on this corpus"
        topk = fetch(f"{uri.rstrip('/')}/topk/", f"refs/{name}/topk", recursive=True)
        rec["refs"][name] = {"uri": uri, "branch": bman["branch"], "revision": bman["revision"],
                             "device_name": bman.get("device_name"),
                             "nll_first_120k": bctl["nll_corpus_first_120k"]["mean_nll"],
                             **counts_and_quantiles(topk.parent, d, corpus, bman["n_layers"], quantiles=False)}
        if uri.startswith("s3://"):
            shutil.rmtree(topk)                  # a fetched copy; a local bank is never deleted
        (d / "m1" / "metrics.json").write_text(json.dumps(rec["refs"][name], indent=1))
    rec["gates_pass"] = True
    status(out, rec, "done")


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["harvest", "known", "refs"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--branch", default=None)
    ap.add_argument("--inputs", default=None, help="S3 prefix or directory with the laptop bank copies")
    ap.add_argument("--bank-exemplar", default="t15_exemplar_a.npz")
    ap.add_argument("--fail-exemplar", default="t14_exemplar_a.npz")
    ap.add_argument("--bank-contexts", default="t15_contexts.jsonl")
    ap.add_argument("--refs", default=None, help="name=uri,name=uri: banked harvest directories")
    ap.add_argument("--registry", default=None, help="dry run only: a planted registry JSON")
    ap.add_argument("--n-poles-grid", type=int, default=m1.N_POLES_GRID)
    ap.add_argument("--device", default="cuda")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    out = pathlib.Path(a.out)
    (out / "m1").mkdir(parents=True, exist_ok=True)
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    import torch
    import transformers
    reg = m1.load_registry(a.registry)
    rec = {"mode": a.mode, "branch": a.branch, "key": reg["branches"][a.branch]["key"] if a.branch else None,
           "started_utc": utc(), "gates_pass": False, "same": reg["same"], "nll_band": reg["nll_band"],
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
    assert rec["versions"] == {"torch": reg["same"]["torch"], "transformers": reg["same"]["transformers"]}, \
        f"G-SAME: versions {rec['versions']} differ from the banked grid harvest's"
    if a.device == "cuda":
        assert torch.cuda.is_available(), "no CUDA on this instance"
    if a.mode == "refs":
        run_refs(a, reg, out, rec)
    else:
        assert a.branch in reg["branches"] and a.inputs, "--branch (registered) and --inputs are required"
        run_harvest(a, reg, out, rec, known=a.mode == "known")
    print("M1_JOB_OK", json.dumps({"mode": a.mode, "branch": a.branch}), flush=True)


if __name__ == "__main__":
    main()
