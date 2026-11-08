"""Weight-free offline gate of M1 (studies/next/07_PICKS/M1_OFFLINE_GATE.md). No weight download, no GPU, no AWS
call. CPU torch 2.12.0 and transformers 5.9.0 (the versions of the banked grid harvest).

    python -m infra.aws_pg.jobs.m1_dryrun --hf <HF text files> --grid <banked grid> \
        --corpus-manifest <corpus manifest.json> --t15-manifest <T_15 manifest.json> --out <scratch dir>

--hf: the Hugging Face API reads (refs.json, tree_<branch>.json, config_<branch>.json, index_<branch>.json,
headers/hdr__<branch>__<file>.json, tok/ tokenizer files). --grid: BANK/neurons/w3/grid (16 banked points).
Sections, one CHECK line per check:
  REAL      the registry against the HF reads; the config rewrite on the three real mix configs; the model
            built from the rewritten config on the meta device has exactly the index's 355 tensor names.
  WIN       sliding_window 4096 and 4097 give identical logits on a 2,049-position forward (and 1,000 does not).
  ROOM      on the 640 banked grid poles: the lead labels on neighbouring banked points reach both SUPPORTED and
            REFUTED; the identity gate passes on an identity pair and fails on its failing input; K0 passes on the
            bank against itself and fails on its failing input (T_14's values at T_15's positions).
  JOB       a tiny random Olmo 3 (4 layers, hidden 64, MLP width 96, the real 100,278-token tokenizer) on a tiny
            corpus, through m1_harvest.main from its entry point: known, harvest x5, refs. metrics.json is
            rewritten inside the harvest loop.
  FAIL      each instance gate on its planted failing input, through the entry point in a child process
            (exit non-zero) or as a direct call.
  VERDICT   freeze and verdict on the tiny job's banks; then on synthetic banks: one that satisfies the
            ordering, two that violate it, and the does-not-count and ceiling branches.
  PROJECT   GPU-hours of the known number and of the full run.
Ends with M1_DRYRUN_OK when every check holds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RESULTS = []
MIX = ["stage2-step47684-mix-gen-mc-only-from-2T-ckpt", "stage2-step47684-mix-round5-from-2T-ckpt",
       "stage2-step47684-mix-math-code-reasoning-web-from-2T-ckpt"]


def check(name, ok, value=""):
    RESULTS.append((name, bool(ok)))
    print(f"CHECK {name:44s} {'PASS' if ok else 'FAIL'} {value}", flush=True)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------ REAL
def real(hf: pathlib.Path, scratch: pathlib.Path):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from infra.pg import m1
    import infra.aws_pg.jobs.m1_harvest as J
    refs = {b["name"]: b["targetCommit"] for b in json.loads((hf / "refs.json").read_text())["branches"]}
    for br, e in m1.BRANCHES.items():
        tree = json.loads((hf / f"tree_{br}.json").read_text())
        idx = json.loads((hf / f"index_{br}.json").read_text())
        files = set(idx["weight_map"].values())
        lfs = {f["path"]: f["lfs"]["oid"] for f in tree if f["path"] in files}
        check(f"REAL registry {e['key']} commit and oids", refs[br] == e["commit"] and lfs == e["weights"],
              f"{e['commit'][:12]}, {len(lfs)} weight files")
        found = {f["path"]: f["lfs"]["oid"] for f in tree if f["path"].endswith(".safetensors") and f["path"] in files}
        g = m1.shard_gate(found, files, e, m1.STALE)
        check(f"REAL G-SHARD {e['key']} on its own tree", g["pass"])
        check(f"REAL G-SHARD {e['key']} on main's shards fails", not m1.shard_gate(dict(m1.STALE), files, e,
                                                                                    m1.STALE)["pass"])
    main_tree = json.loads((hf / "tree_main.json").read_text())
    check("REAL stale oids = main's shards", {f["path"]: f["lfs"]["oid"] for f in main_tree
                                               if f["path"].endswith(".safetensors")} == m1.STALE)
    for br in MIX:
        mt = {f["path"]: f["lfs"]["oid"] for f in json.loads((hf / f"tree_{br}.json").read_text())
              if f["path"].endswith("-of-00003.safetensors")}
        check(f"REAL {br[19:30]} carries main's 3 bf16 shards", mt == m1.STALE)
    for h in sorted((hf / "headers").glob("hdr__*.json")):
        d = {v["dtype"] for k, v in json.loads(h.read_text()).items() if k != "__metadata__"}
        want = {"F32"} if "-of-00006" in h.name else {"BF16"}
        check(f"REAL header dtype {h.name[5:60]}", d == want, sorted(d))
    start = json.loads((hf / "config_stage1-step476838.json").read_text())
    check("REAL start config == REF_CONFIG", start == m1.REF_CONFIG and m1.config_gate(start, m1.REF_CONFIG)["pass"])
    for br in MIX:
        raw = json.loads((hf / f"config_{br}.json").read_text())
        new = m1.rewrite_config(raw)
        g, f = m1.config_gate(new, m1.REF_CONFIG), m1.config_gate(raw, m1.REF_CONFIG)
        check(f"REAL G-CFG rewritten {br[19:30]}", g["pass"] and g["diff_keys"] == ["sliding_window"], g)
        check(f"REAL G-CFG as released fails {br[19:30]}", not f["pass"], f["diff_keys"])
    # the job's own load_dir on a snapshot holding the real mix config, index and tokenizer (no weights)
    br = MIX[2]
    snap = scratch / "real_snap" / m1.BRANCHES[br]["commit"]
    snap.mkdir(parents=True, exist_ok=True)
    blob = scratch / "real_snap" / "config_blob.json"
    shutil.copy(hf / f"config_{br}.json", blob)
    (snap / "config.json").unlink(missing_ok=True)
    (snap / "config.json").symlink_to(blob)
    shutil.copy(hf / f"index_{br}.json", snap / "model.safetensors.index.json")
    for t in (hf / "tok").iterdir():
        shutil.copy(t, snap / t.name)
    before = sha(blob)
    d, rec = J.load_dir(snap, m1.BRANCHES[br], m1.load_registry())
    check("REAL load_dir: rewritten copy, blob untouched", sha(blob) == before and d != snap
          and json.loads((d / "config.json").read_text())["model_type"] == "olmo3", rec["G_CFG"])
    cfg = AutoConfig.from_pretrained(d)
    ref = AutoConfig.from_pretrained(hf / "config_stage1-step476838.json")
    diff = sorted(k for k in set(cfg.to_dict()) | set(ref.to_dict())
                  if cfg.to_dict().get(k) != ref.to_dict().get(k) and k != "_name_or_path")
    check("REAL AutoConfig(rewritten) is Olmo3Config", type(cfg).__name__ == "Olmo3Config" and diff == ["sliding_window"],
          diff)
    bad = scratch / "real_unrewritten"
    bad.mkdir(exist_ok=True)
    shutil.copy(hf / f"config_{br}.json", bad / "config.json")
    rc = subprocess.run([sys.executable, "-c", "import sys; from transformers import AutoConfig; "
                         "AutoConfig.from_pretrained(sys.argv[1])", str(bad)], capture_output=True, text=True)
    check("REAL AutoConfig(as released) exits non-zero", rc.returncode != 0, rc.stderr.strip().splitlines()[-1][:100])
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    keys = set(model.state_dict())
    n = sum(p.numel() for p in model.parameters())
    check("REAL meta model keys == index weight_map", keys == set(idx["weight_map"]), f"{len(keys)} tensors")
    check("REAL parameter count == index total_parameters", n == idx["metadata"]["total_parameters"], n)
    check("REAL cast names exist in the index", set(m1.CAST_NAMES) <= set(idx["weight_map"]))
    tok_vocab = __import__("transformers").AutoTokenizer.from_pretrained(hf / "tok")
    from infra.pg.matrix_models import vocab_sha256
    check("REAL mix tokenizer vocab == corpus record", vocab_sha256(tok_vocab) ==
          "2f3c015ee8742712a402874d88560643d1071060e12647737928c6c1f80f2710")


def real_t15(t15_manifest: pathlib.Path, grid: pathlib.Path):
    """The banked T_15 harvest passes G-FIT and G-SAME against the registered settings (its NLL, 2.675, is quoted
    from 00_ASSETS.md line 54: controls.json is on S3 only)."""
    from infra.pg import m1
    man = json.loads(t15_manifest.read_text())
    met = json.loads((grid / "1" / "out" / "t15_step1413814" / "metrics.json").read_text())
    g = m1.fit_gate(met, {"nll_corpus_first_120k": {"mean_nll": 2.675}}, man, m1.SAME)
    check("REAL banked T_15 manifest passes G-SAME and G-FIT", g["pass"] and g["same_diff"] == [], g["same_diff"])


# ------------------------------------------------------------------------------------------ WIN
def win():
    import torch
    from transformers import Olmo3Config, Olmo3ForCausalLM
    base = dict(vocab_size=512, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=4, max_position_embeddings=8192,
                layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"])
    torch.manual_seed(0)
    x = torch.randint(0, 512, (1, 2049))
    outs = {}
    sd = None
    for w in (4096, 4097, 1000):
        m = Olmo3ForCausalLM(Olmo3Config(**base, sliding_window=w, attn_implementation="sdpa")).eval()
        if sd is None:
            sd = m.state_dict()
        m.load_state_dict(sd)
        with torch.inference_mode():
            outs[w] = m(input_ids=x, use_cache=False).logits
    check("WIN sliding 4096 vs 4097 at 2,049 positions", torch.equal(outs[4096], outs[4097]))
    check("WIN control: sliding 1000 differs", not torch.equal(outs[4096], outs[1000]))


# ------------------------------------------------------------------------------------------ ROOM (control data)
def grid_counts(path: pathlib.Path, docdir, n_dirs):
    rows = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows[r["pole"]] = [c["doc"] for c in r["topk"]]
    layers = sorted({int(p[1:3]) for p in rows})
    neurons = {lyr: sorted({int(p[4:-1]) for p in rows if int(p[1:3]) == lyr}) for lyr in layers}
    n = len(neurons[layers[0]])
    assert all(len(v) == n for v in neurons.values())
    counts = np.zeros((len(layers), n, 2, n_dirs), np.uint8)
    for li, lyr in enumerate(layers):
        for ni, nn in enumerate(neurons[lyr]):
            for s, sg in enumerate("+-"):
                d = docdir[np.array(rows[f"L{lyr:02d}N{nn}{sg}"])]
                counts[li, ni, s] = np.bincount(d, minlength=n_dirs)
    return counts


def room(grid: pathlib.Path, manifest: dict):
    from infra.pg import m1
    docdir, cm = m1.doc_dirs(manifest), m1.cm_mask(manifest)
    pts = sorted([p for sub in ("1", "1b") for p in (grid / sub / "out").iterdir()], key=lambda p: int(p.name[1:3]))
    c = {}
    for p in pts:
        k = grid_counts(p / "samples" / "contexts.jsonl", docdir, len(cm))
        c[p.name[:3]] = {"counts": k, "filled": k.sum(-1).astype(np.uint8), "cm": cm}
    s = {k: m1.share(v) for k, v in c.items()}
    C = {k: m1.band_means(v) for k, v in s.items()}
    names = list(C)
    pairs = [("t12", "t13"), ("t13", "t14"), ("t14", "t15")]
    tau = [max(abs(C[j][b] - C[i][b]) for i, j in pairs) for b in range(4)]
    kappa = [max(abs(C[i][b] - C[j][b]) for i in names for j in names) for b in range(4)]
    print("ROOM C per band (640 banked poles):", json.dumps({k: [round(x, 4) for x in v] for k, v in C.items()}))
    print("ROOM tau_640", [round(x, 4) for x in tau], "kappa_640", [round(x, 4) for x in kappa])
    labels = {}
    for i, j in zip(names[:-1], names[1:]):
        for g, m in ((i, j), (j, i)):
            bd = m1.band_diff(s[m], s[g])
            labels[f"{g}->{m}"] = [m1.label_lead(x, lo, hi, tau[b], kappa[b]) for b, (x, lo, hi) in enumerate(bd)]
    flat = [x for v in labels.values() for x in v]
    print("ROOM labels on neighbouring banked pairs, as (G, M):", json.dumps(labels))
    check("ROOM SUPPORTED reachable on banked pairs", "SUPPORTED" in flat, flat.count("SUPPORTED"))
    check("ROOM REFUTED reachable on banked pairs", "REFUTED" in flat, flat.count("REFUTED"))
    check("ROOM tau < kappa in every band", all(t < k for t, k in zip(tau, kappa)))
    late = {k: v for k, v in labels.items() if k.split("->")[0] in ("t12", "t13", "t14", "t15")}
    print("ROOM labels on the late pairs (the stage-1 drift reference):", json.dumps(late))
    gid = m1.identity_gate(c["t15"], c["t15"], min(tau))
    gid_f = m1.identity_gate(c["t15"], c["t13"], min(tau))
    check("ROOM G-ID identity pair T15/T15 passes", gid["pass"], gid["I"])
    check("ROOM G-ID failing input T13 as T15's twin fails", not gid_f["pass"], [round(x, 4) for x in gid_f["I"]])
    return {"tau_640": tau, "kappa_640": kappa, "C": C, "labels": labels}


def k0_offline(grid: pathlib.Path):
    from infra.pg import m1
    import infra.aws_pg.jobs.m1_harvest as J
    t15, t14 = grid / "1" / "out" / "t15_step1413814", grid / "1b" / "out" / "t14_step1000000"
    for f, key in ((t15 / "samples" / "contexts.jsonl", "t15_contexts.jsonl"),
                   (t15 / "poles" / "exemplar_a.npz", "t15_exemplar_a.npz"),
                   (t14 / "poles" / "exemplar_a.npz", "t14_exemplar_a.npz"),
                   (t14 / "samples" / "contexts.jsonl", "t14_contexts.jsonl"),
                   (grid / "1b" / "out" / "t13_step560000" / "poles" / "exemplar_a.npz", "t13_exemplar_a.npz")):
        check(f"K0 bank file {key} exists, sha256 registered", f.exists() and sha(f) == m1.BANK_SHA[key], str(f)[-60:])
    poles, pos = J.grid_poles(t15 / "samples" / "contexts.jsonl")
    mask = (pos >= 0) & (pos < m1.K0_TOKENS)
    a15 = np.load(t15 / "poles" / "exemplar_a.npz")["a"]
    a14 = np.load(t14 / "poles" / "exemplar_a.npz")["a"]
    ok = m1.k0_values_gate(a15, a15, mask)
    bad = m1.k0_values_gate(a14, a15, mask)
    check("K0 bank against itself passes", ok["pass"], f"{ok['n']} entries in {int(mask.any(1).sum())} poles")
    check("K0 failing input (T14 values) fails", not bad["pass"],
          f"median {bad['median_rel']:.4f} p99 {bad['p99_rel']:.4f}")
    check("K0 subset rule, bank in itself", m1.k0_subset_gate(pos, pos, m1.K0_TOKENS)["pass"])
    _, t14top = J.grid_poles(t14 / "samples" / "contexts.jsonl")
    sh = np.where(pos >= 0, pos + 1, -1)
    g1, g2 = m1.k0_subset_gate(pos, sh, m1.K0_TOKENS), m1.k0_subset_gate(pos, t14top, m1.K0_TOKENS)
    check("K0 subset failing input (positions + 1) fails", not g1["pass"], f"{g1['share']:.3f}")
    check("K0 subset with T14's own top-50 fails", not g2["pass"], f"{g2['share']:.3f}")
    return {"n": ok["n"], "fail_median": bad["median_rel"], "fail_p99": bad["p99_rel"]}


# ------------------------------------------------------------------------------------------ fixtures
DIRS = ["data/common_crawl-a_topic-0001", "data/common_crawl-b_topic-0002", "data/common_crawl-c_topic-0003",
        "data/common_crawl-d_topic-0004", "data/finemath-3plus", "data/olmocr_science_pdfs-x1",
        "data/olmocr_science_pdfs-x2", "data/stack_edu-JavaScript", "data/stack_edu-Python",
        "data/stack_edu-SQL"]
WINDOW, BATCH = 128, 4


def make_corpus(d: pathlib.Path, vocab_sha: str):
    rng = np.random.default_rng(1)
    docs = {b: {i: [int(x) for x in rng.integers(380, 640, 16)] for i in range(len(DIRS))} for b in ("1", "2")}
    toks, offs = [], [0]
    for b in ("1", "2"):
        for i in range(len(DIRS)):
            for n in docs[b][i]:
                lo = 1000 + 3000 * (i % 5)                         # directories differ in their token range
                toks.append(rng.integers(lo, lo + 3000, n).astype(np.uint32))
                offs.append(offs[-1] + n)
    tokens = np.concatenate(toks)
    tok_dir = d / "tok" / "olmo3"
    tok_dir.mkdir(parents=True, exist_ok=True)
    np.save(tok_dir / "tokens.npy", tokens)
    np.save(tok_dir / "doc_offsets.npy", np.array(offs, np.int64))
    n1 = sum(len(docs["1"][i]) for i in range(len(DIRS)))
    t1 = int(offs[n1])
    man = {"dirs": DIRS, "n_docs": len(offs) - 1,
           "shards": [{"dir": x, "n_docs_selected": {"1": len(docs["1"][i]), "2": len(docs["2"][i])}}
                      for i, x in enumerate(DIRS)],
           "bands": {"1": {"doc_start": 0, "doc_end": n1, "tok_start": {"olmo3": 0}, "tok_end": {"olmo3": t1}},
                     "2": {"doc_start": n1, "doc_end": len(offs) - 1, "tok_start": {"olmo3": t1},
                           "tok_end": {"olmo3": int(tokens.size)}}},
           "tokenizers": {"olmo3": {"n_tokens": int(tokens.size), "sha256_of_tokens_npy": sha(tok_dir / "tokens.npy"),
                                    "vocab_sha256": vocab_sha, "bos_id": 100257, "eos_id": 100257}}}
    (d / "manifest.json").write_text(json.dumps(man))
    return man, int(tokens.size)


def tiny_cfg(hf: pathlib.Path) -> dict:
    c = json.loads((hf / "config_stage1-step476838.json").read_text())
    c.update(hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
             layer_types=c["layer_types"][:4])
    return c


def tiny_model(cfg: dict, seed: int, base=None, noise: float = 0.0):
    import torch
    from transformers import Olmo3Config, Olmo3ForCausalLM
    torch.manual_seed(seed)
    m = Olmo3ForCausalLM(Olmo3Config(**{k: v for k, v in cfg.items() if k not in ("architectures", "model_type",
                                                                                   "transformers_version")}))
    if base is not None:
        sd = {k: v + noise * torch.randn_like(v) * v.std().clamp_min(1e-3) for k, v in base.state_dict().items()}
        m.load_state_dict(sd)
    return m.eval()


def save_shards(model, d: pathlib.Path, n: int, dtype) -> dict:
    from safetensors.torch import save_file
    sd = {k: v.detach().to(dtype).contiguous().clone() for k, v in model.state_dict().items()}
    names = sorted(sd)
    wm, total = {}, 0
    for i, part in enumerate(np.array_split(np.array(names), n), 1):
        f = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file({k: sd[k] for k in part}, str(d / f), metadata={"format": "pt"})
        wm.update({str(k): f for k in part})
        total += sum(sd[k].numel() * sd[k].element_size() for k in part)
    return {"metadata": {"total_size": total}, "weight_map": wm}


def make_remote(d: pathlib.Path, model, cfg: dict, hf: pathlib.Path, dtype_name: str, stale_model=None) -> dict:
    import torch
    d.mkdir(parents=True, exist_ok=True)
    n, dt = (6, torch.float32) if dtype_name == "F32" else (3, torch.bfloat16)
    idx = save_shards(model, d, n, dt)
    (d / "model.safetensors.index.json").write_text(json.dumps(idx))
    if stale_model is not None:                    # the mix branches' extra copy of main's three bf16 shards
        save_shards(stale_model, d, 3, torch.bfloat16)
    c = dict(cfg)
    if dtype_name == "F32":
        c.update(model_type="olmo2-retrofit", architectures=["Olmo2RetrofitForCausalLM"], sliding_window=4097)
    (d / "config.json").write_text(json.dumps(c))
    for t in (hf / "tok").iterdir():
        shutil.copy(t, d / t.name)
    return {f"model-{i:05d}-of-{n:05d}.safetensors": sha(d / f"model-{i:05d}-of-{n:05d}.safetensors")
            for i in range(1, n + 1)}


def fake_hub(remotes: dict, commits: dict, cache: pathlib.Path):
    from huggingface_hub.utils import filter_repo_objects

    def hub(repo, revision, allow_patterns):
        src = pathlib.Path(remotes[revision])
        names = list(filter_repo_objects([p.name for p in src.iterdir()], allow_patterns=allow_patterns))
        root = cache / ("models--" + repo.replace("/", "--"))
        snap = root / "snapshots" / commits[revision]
        snap.mkdir(parents=True, exist_ok=True)
        (root / "blobs").mkdir(exist_ok=True)
        for nm in names:
            blob = root / "blobs" / sha(src / nm)
            if not blob.exists():
                shutil.copy(src / nm, blob)
            if not (snap / nm).exists():
                (snap / nm).symlink_to(os.path.relpath(blob, snap))
        return str(snap)
    return hub


def child(spec_path: str):
    """Entry point of a child process: the job's main() with the local stand-in hub."""
    spec = json.loads(pathlib.Path(spec_path).read_text())
    import infra.aws_pg.jobs.m1_harvest as J
    J.HUB = fake_hub(spec["remotes"], spec["commits"], pathlib.Path(os.environ["HF_HUB_CACHE"]))
    J.main(spec["argv"])


def run_child(scratch: pathlib.Path, name: str, spec: dict, env: dict) -> subprocess.CompletedProcess:
    p = scratch / f"child_{name}.json"
    p.write_text(json.dumps(spec))
    return subprocess.run([sys.executable, "-c", "import sys; from infra.aws_pg.jobs import m1_dryrun as D; "
                           "D.child(sys.argv[1])", str(p)], capture_output=True, text=True, env=env, cwd=REPO_ROOT)


def last_error(rc) -> str:
    lines = [x for x in rc.stderr.strip().splitlines() if x.strip()]
    return lines[-1][:150] if lines else ""


# ------------------------------------------------------------------------------------------ JOB + FAIL
def job(hf: pathlib.Path, scratch: pathlib.Path) -> pathlib.Path:
    import torch
    import transformers
    from infra.pg import m1
    from infra.pg import neurons_harvest as NH
    from infra.pg import neurons_corpus as NC
    import infra.aws_pg.jobs.m1_harvest as J
    from infra.pg.matrix_models import vocab_sha256
    tok = transformers.AutoTokenizer.from_pretrained(hf / "tok")
    corpus_dir = scratch / "corpus"
    man, n_tok = make_corpus(corpus_dir, vocab_sha256(tok))
    cfg = tiny_cfg(hf)
    models = {"T15": tiny_model(cfg, 15)}
    models["T14"] = tiny_model(cfg, 14, models["T15"], 0.30)
    models["T13"] = tiny_model(cfg, 13, models["T15"], 0.40)
    models["T12"] = tiny_model(cfg, 12, models["T15"], 0.50)
    models["T00"] = tiny_model(cfg, 99)
    models["start"] = tiny_model(cfg, 2, models["T13"], 0.10)
    for k, sd in (("G", 21), ("R", 22), ("M", 23)):
        models[k] = tiny_model(cfg, sd, models["start"], 0.20)
    models["main"] = tiny_model(cfg, 77)
    max_tokens = n_tok
    n_windows = max_tokens // WINDOW
    corpus = NC.load_tok(corpus_dir, "olmo3")
    same = dict(m1.SAME, max_tokens=max_tokens, batch=BATCH, window=WINDOW, n_windows=n_windows,
                n_tokens_scored=n_windows * (WINDOW - 1), corpus_manifest_sha256=corpus["manifest_sha256"],
                tokens_sha256=corpus["tokens_sha256"], torch=torch.__version__, transformers=transformers.__version__)
    # ---- the planted banks: a T_15-rule harvest of the tiny "T15", and T14, T13 at T15's positions
    poles = [NH.pole_id(lyr, n, s) for lyr in range(4) for n in (3, 17, 40, 63, 90) for s in (0, 1)]
    banks = scratch / "grid_bank"
    hdirs = {}

    def harvest(key, out, positions=None):
        # through the same loader as the real grid (from_pretrained, bf16): the load path sets buffer dtypes
        d = scratch / "bank_models" / out.name
        make_remote(d, models[key], cfg, hf, "BF16")
        model, _ = NH.load_model(str(d), "O7", device="cpu")
        NH.run(out, model=model, tok=tok, corpus=corpus, model_key="O7", repo=m1.REPO, revision="bank", branch=key,
               max_tokens=max_tokens, window=WINDOW, batch=BATCH, snapshot_bands=[1], poles=poles, positions=positions,
               device="cpu")
        hdirs[key] = out
        shutil.rmtree(d)
    harvest("T15", banks / "t15")
    pos15 = np.load(banks / "t15" / "poles" / "positions.npz")["tok"]
    for key, name in (("T14", "t14b"), ("T14", "t14c"), ("T13", "t13"), ("T12", "t12"), ("T00", "t00")):
        harvest(key, banks / name, pos15)
    inputs = scratch / "inputs"
    inputs.mkdir(exist_ok=True)
    shutil.copy(banks / "t15" / "samples" / "contexts.jsonl", inputs / "t15_contexts.jsonl")
    shutil.copy(banks / "t15" / "poles" / "exemplar_a.npz", inputs / "t15_exemplar_a.npz")
    shutil.copy(banks / "t14b" / "poles" / "exemplar_a.npz", inputs / "t14_exemplar_a.npz")
    shutil.copy(banks / "t14b" / "samples" / "contexts.jsonl", inputs / "t14_contexts.jsonl")
    shutil.copy(banks / "t13" / "poles" / "exemplar_a.npz", inputs / "t13_exemplar_a.npz")
    # ---- remotes (what the hub would serve) and the planted registry
    remotes, commits, branches = {}, {}, {}
    plan = {"stage1-step476838": ("start", "BF16"), MIX[0]: ("G", "F32"), MIX[1]: ("R", "F32"), MIX[2]: ("M", "F32"),
            "stage1-step1413814": ("T15", "BF16")}
    stale_dir = scratch / "remote" / "main"
    stale = make_remote(stale_dir, models["main"], cfg, hf, "BF16")
    for br, (key, dt) in plan.items():
        d = scratch / "remote" / br
        w = make_remote(d, models[key], cfg, hf, dt, models["main"] if dt == "F32" else None)
        commits[br] = hashlib.sha1(br.encode()).hexdigest()
        remotes[br] = str(d)
        branches[br] = {"key": key, "commit": commits[br], "dtype": dt, "rewrite": dt == "F32", "weights": w}
    ref_cfg = dict(cfg)
    reg = {"branches": branches, "stale": stale, "ref_config": ref_cfg, "same": same,
           "bank_sha": {p.name: sha(p) for p in inputs.iterdir()}, "nonweight": m1.NONWEIGHT,
           "cast_names": ["model.layers.0.mlp.gate_proj.weight", "model.layers.3.mlp.down_proj.weight",
                          "model.layers.1.self_attn.q_proj.weight", "model.norm.weight"],
           "pc_k": 3, "k0_tokens": 940 * WINDOW, "nll_band": [0.0, 20.0]}     # K0 must cover the 120,000 NLL targets
    reg_path = scratch / "registry.json"
    reg_path.write_text(json.dumps(reg))
    cache = scratch / "hf_cache"
    J.HUB = fake_hub(remotes, commits, cache)
    common = ["--corpus", str(corpus_dir), "--inputs", str(inputs), "--registry", str(reg_path), "--device", "cpu",
              "--n-poles-grid", str(len(poles))]
    out = scratch / "banks"
    # count metrics.json rewrites inside the harvest loop
    writes = {}
    real_dump = NH.dump

    def counting_dump(path, obj):
        path = pathlib.Path(path)
        if path.name == "metrics.json" and path.parent.name != "m1":
            writes[str(path.parent)] = writes.get(str(path.parent), 0) + 1
        real_dump(path, obj)
    NH.dump = counting_dump
    NH.METRICS_EVERY = 40
    J.main(["--mode", "known", "--branch", "stage1-step1413814", "--out", str(out / "k0")] + common)
    k0 = json.loads((out / "k0" / "m1" / "metrics.json").read_text())["K0"]
    check("JOB known: K0 passes on the tiny T15", k0["pass"], f"n {k0['values']['n']}, exact share "
          f"{k0['values']['exact_share']:.3f}, failing-input median {k0['values']['failing_input']['median_rel']:.3f}")
    for key, br in (("start_a", "stage1-step476838"), ("start_b", "stage1-step476838"), ("G", MIX[0]),
                    ("R", MIX[1]), ("M", MIX[2])):
        J.main(["--mode", "harvest", "--branch", br, "--out", str(out / key)] + common)
        rec = json.loads((out / key / "m1" / "metrics.json").read_text())
        n_w = writes.get(str(out / key), 0)
        check(f"JOB harvest {key}: gates pass, outputs written", rec["gates_pass"] and rec["phase"] == "done"
              and (out / key / "m1" / "counts.npz").exists() and (out / key / "m1" / "quantiles.npz").exists(),
              f"metrics.json writes {n_w}; cast trunc mismatch "
              f"{[t.get('trunc_mismatch_share') for t in rec['G_CAST']['tensors']][:2]}")
        check(f"JOB harvest {key}: metrics.json rewritten in the loop", n_w >= 3, n_w)
        sm = rec["samples"]
        check(f"JOB harvest {key}: one code+math and one other slot banked", [x["counted_as"] for x in sm] ==
              ["code+math", "other"] and sm[0]["dir"].split("/")[1].startswith(("stack_edu", "finemath")),
              f"{sm[0]['dir']} / {sm[1]['dir']}")
        check(f"JOB harvest {key}: weights deleted", not (cache / "models--allenai--Olmo-3-1025-7B").exists())
    NH.dump = real_dump
    refs_arg = ",".join(f"{n}={banks / n}" for n in ("t00", "t12", "t13", "t14b", "t14c", "t15"))
    J.main(["--mode", "refs", "--refs", refs_arg, "--out", str(out / "refs")] + common)
    check("JOB refs: six reference count files", all((out / "refs" / n / "m1" / "counts.npz").exists()
                                                       for n in ("t00", "t12", "t13", "t14b", "t14c", "t15")))
    cs = [np.load(out / k / "m1" / "counts.npz") for k in ("start_a", "start_b")]
    check("JOB identity pair: start_a and start_b counts equal", np.array_equal(cs[0]["counts"], cs[1]["counts"]))
    check("JOB counts sum to filled slots", bool((cs[0]["counts"].sum(-1) == cs[0]["filled"]).all()))
    man_a = json.loads((out / "start_a" / "manifest.json").read_text())
    check("JOB manifest settings equal the registered grid settings",
          m1.fit_gate(json.loads((out / "start_a" / "metrics.json").read_text()),
                      json.loads((out / "start_a" / "controls.json").read_text()), man_a, same, [0, 20])["pass"])
    # ---- FAIL: each gate on its planted input
    env = dict(os.environ, HF_HUB_CACHE=str(scratch / "hf_cache_child"), M1_WORK=str(scratch / "work_child"),
               PYTHONPATH=str(REPO_ROOT))
    spec = {"remotes": remotes, "commits": commits}

    def bad_reg(name, mutate):
        r = json.loads(reg_path.read_text())
        mutate(r)
        p = scratch / f"registry_{name}.json"
        p.write_text(json.dumps(r))
        return p

    def args_with(p, mode="harvest", br=MIX[2], extra=()):
        a = ["--mode", mode, "--branch", br, "--out", str(scratch / "fail" / p.stem)] + common
        a[a.index("--registry") + 1] = str(p)
        return a + list(extra)
    cases = {
        "G-REV wrong commit": (bad_reg("rev", lambda r: r["branches"][MIX[2]].update(commit="0" * 40)), "G-REV"),
        "G-SHARD one oid changed": (bad_reg("shard", lambda r: r["branches"][MIX[2]]["weights"].update(
            {"model-00001-of-00006.safetensors": "0" * 64})), "G-SHARD"),
        "G-SHARD stale shards registered": (bad_reg("stale", lambda r: r["branches"][MIX[2]].update(
            weights=dict(r["stale"]))), "G-SHARD"),
        "G-CFG unrewritten config": (bad_reg("cfg", lambda r: r["branches"][MIX[2]].update(rewrite=False)), "G-CFG"),
        "G-CFG width differs": (bad_reg("cfgw", lambda r: r["ref_config"].update(intermediate_size=97)), "G-CFG"),
        "G-SAME versions differ": (bad_reg("ver", lambda r: r["same"].update(transformers="0.0.0")), "G-SAME"),
        "input sha256 differs": (bad_reg("input", lambda r: r["bank_sha"].update({"t15_contexts.jsonl": "0" * 64})),
                                 "input t15_contexts.jsonl"),
    }
    for name, (p, needle) in cases.items():
        env["HF_HUB_CACHE"] = str(scratch / f"hf_cache_{p.stem}")       # a fresh cache per child
        rc = run_child(scratch, p.stem, {**spec, "argv": args_with(p)}, env)
        check(f"FAIL {name}", rc.returncode != 0 and needle in rc.stderr, last_error(rc))
    bad_corpus = scratch / "corpus_bad"
    shutil.copytree(corpus_dir, bad_corpus, dirs_exist_ok=True)
    (bad_corpus / "manifest.json").write_text((corpus_dir / "manifest.json").read_text() + " ")
    a = args_with(reg_path)
    a[a.index("--corpus") + 1] = str(bad_corpus)
    env["HF_HUB_CACHE"] = str(scratch / "hf_cache_corpus")
    rc = run_child(scratch, "corpus", {**spec, "argv": a}, env)
    check("FAIL G-CORPUS manifest changed by one byte", rc.returncode != 0 and "G-CORPUS" in rc.stderr, last_error(rc))
    a = args_with(reg_path, mode="known", br="stage1-step1413814", extra=["--bank-exemplar", "t14_exemplar_a.npz",
                                                                         "--fail-exemplar", "t15_exemplar_a.npz"])
    env["HF_HUB_CACHE"] = str(scratch / "hf_cache_k0")
    rc = run_child(scratch, "k0", {**spec, "argv": a}, env)
    check("FAIL K0 against the wrong bank (T14 values)", rc.returncode != 0 and "K0 failed" in rc.stderr, last_error(rc))
    # in-process: the cast gate on a perturbed parameter; the fit gate's failing inputs
    J.HUB = fake_hub(remotes, commits, cache)
    local, _ = J.download(MIX[2], branches[MIX[2]], reg, scratch / "castcheck")
    src, _ = J.load_dir(local, branches[MIX[2]], reg)
    model, _ = NH.load_model(str(src), "O7", device="cpu")
    g = J.cast_check(model, src, reg["cast_names"])
    check("FAIL G-CAST passes on the loaded model, failing input fails", g["pass"] and not g["failing_input"]["pass"])
    with torch.no_grad():
        dict(model.named_parameters())["model.layers.0.mlp.gate_proj.weight"][0, 0] += 1.0
    check("FAIL G-CAST one entry changed", not J.cast_check(model, src, reg["cast_names"])["pass"])
    shutil.rmtree(cache / "models--allenai--Olmo-3-1025-7B")
    met = json.loads((out / "M" / "metrics.json").read_text())
    ctl = json.loads((out / "M" / "controls.json").read_text())
    manM = json.loads((out / "M" / "manifest.json").read_text())
    check("FAIL G-FIT complete false", not m1.fit_gate({**met, "complete": False}, ctl, manM, same, [0, 20])["pass"])
    check("FAIL G-FIT NLL of T_0 (12.223) outside the band", not m1.fit_gate(
        met, {"nll_corpus_first_120k": {"mean_nll": 12.223}}, manM, same)["pass"])
    check("FAIL G-SAME batch 16", not m1.fit_gate(met, ctl, {**manM, "batch": 16}, same, [0, 20])["pass"])
    return out


# ------------------------------------------------------------------------------------------ VERDICT
def synth_bank(root: pathlib.Path, spec: dict, seed: int = 0):
    """Synthetic banks: 8 layers x 200 neurons x 2 poles, 10 directories (3 code+math). Pole i has a base code+math
    probability p_i; checkpoint c adds shift[c][band]. One binomial draw of 50 slots per pole and checkpoint."""
    from infra.pg import m1
    rng = np.random.default_rng(seed)
    L, n, nd = 8, 200, 10
    cm = np.zeros(nd, bool)
    cm[7:] = True
    p = np.clip(rng.beta(2, 16, (L, n, 2)), 0.01, 0.9)
    band = m1.band_index(L)[:, None, None]
    draws = {}
    for key, sh in spec["shift"].items():
        if key in spec.get("copy", {}):
            draws[key] = draws[spec["copy"][key]]
            continue
        q = np.clip(p + np.array(sh)[band], 0.0, 1.0)
        k_cm = rng.binomial(50, q)
        c = np.zeros((L, n, 2, nd), np.int64)
        for idx in np.ndindex(L, n, 2):
            c[idx][7:] = rng.multinomial(k_cm[idx], [1 / 3] * 3)
            c[idx][:7] = rng.multinomial(50 - k_cm[idx], [1 / 7] * 7)
        draws[key] = c.astype(np.uint8)
    for key, c in draws.items():
        d = root / ("refs/" + key if key.startswith("t") else key)
        (d / "m1").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "m1" / "counts.npz", counts=c, filled=c.sum(-1).astype(np.uint8), cm_mask=cm)
        if key.startswith("t"):
            continue
        same = m1.SAME
        (d / "metrics.json").write_text(json.dumps({"complete": True, "G0_pass": True, "dead_poles": 0,
                                                    "n_tokens_scored": same["n_tokens_scored"]}))
        (d / "controls.json").write_text(json.dumps({"nll_corpus_first_120k": {"mean_nll": 2.75}}))
        (d / "manifest.json").write_text(json.dumps({**{k: v for k, v in same.items() if k != "max_tokens"},
                                                     "max_tokens_arg": same["max_tokens"]}))
        nl = spec["nll"][key]
        r = np.random.default_rng(seed + 1)
        (d / "m1" / "metrics.json").write_text(json.dumps({
            "same": same, "nll_band": list(m1.NLL_BAND), "gates_pass": True,
            "pc_nll": {"nll": {"cm": (nl[0] + 0.01 * r.standard_normal(32)).tolist(),
                               "web": (nl[1] + 0.01 * r.standard_normal(32)).tolist()}}}))


def verdict_cases(scratch: pathlib.Path, tiny_banks: pathlib.Path):
    from infra.pg import m1
    th = m1.freeze(tiny_banks)
    v = m1.verdict(tiny_banks, th)
    check("VERDICT runs on the tiny job's banks", "headline" in v and v["present"] == ["start_a", "start_b", "G", "R", "M"],
          v["headline"][:90])
    check("VERDICT tiny: every harvest gate read and passed", all(g["pass"] for g in v["gates"].values()))
    check("VERDICT tiny: samples carried for every harvest", all(v["samples"][k] for k in v["present"]))
    refs = {"t00": [0.20] * 4, "t12": [0.0] * 4, "t13": [0.010] * 4, "t14b": [0.0] * 4, "t15": [0.012] * 4,
            "t14c": [0.0] * 4}
    nll_ok = {"start_a": (3.0, 2.8), "start_b": (3.0, 2.8), "G": (3.1, 2.7), "R": (2.8, 2.75), "M": (2.7, 2.8)}
    base = {"start_a": [0.0] * 4, "start_b": [0.0] * 4}
    cases = {
        "satisfies the ordering": ({**base, "G": [-0.03] * 4, "R": [0.02] * 4, "M": [0.06] * 4}, nll_ok, {},
                                   {"H_lead": "SUPPORTED", "H_order": "SUPPORTED", "H_specific": "SUPPORTED"}),
        "violates: R above M in band 2": ({**base, "G": [-0.03] * 4, "R": [0.02, 0.02, 0.10, 0.02], "M": [0.06] * 4},
                                          nll_ok, {}, {"H_lead": "SUPPORTED", "H_order": "REFUTED"}),
        "violates: M equals G": ({**base, "G": [0.01] * 4, "R": [0.01] * 4, "M": [0.01] * 4}, nll_ok, {},
                                 {"H_lead": "REFUTED", "H_order": "REFUTED"}),
        "does not count: identity pair differs": ({**base, "start_b": [0.02] * 4, "G": [-0.03] * 4, "R": [0.02] * 4,
                                                   "M": [0.06] * 4}, nll_ok, {}, {"headline": "DOES NOT COUNT"}),
        "does not count: positive control fails": ({**base, "G": [-0.03] * 4, "R": [0.02] * 4, "M": [0.06] * 4},
                                                   {**nll_ok, "M": (3.2, 2.7)}, {}, {"headline": "DOES NOT COUNT"}),
        "ceiling: M far above G": ({**base, "G": [-0.03] * 4, "R": [0.02] * 4, "M": [0.6] * 4}, nll_ok, {},
                                   {"H_lead": "ABOVE_CEILING", "headline": "DOES NOT COUNT until inspected"}),
    }
    for name, (shift, nll, _, want) in cases.items():
        root = scratch / "synth" / name.replace(" ", "_").replace(":", "")
        if root.exists():
            shutil.rmtree(root)
        cp = {"t14c": "t14b", **({"start_b": "start_a"} if shift["start_b"] == shift["start_a"] else {})}
        synth_bank(root, {"shift": {**refs, **shift}, "copy": cp, "nll": nll})
        th = m1.freeze(root)
        r = m1.verdict(root, th)
        got = {k: r.get(k, "") for k in want}
        ok = all(got[k].startswith(want[k]) for k in want)
        check(f"VERDICT synthetic, {name}", ok, f"{got} | lead {[round(x['mean'], 3) for x in r['lead']]} "
              f"bar {[round(x, 4) for x in r['bar']]}")
        if name == "satisfies the ordering":
            check("VERDICT freeze room on synthetic refs", th["room"]["supported_reachable"]
                  and th["room"]["refuted_reachable"] and th["room"]["tau_below_kappa"])
            check("VERDICT G-ID and PC failing inputs fail", not r["G_ID"]["failing_input_pass"]
                  and not r["PC_NLL"]["failing_input_pass"])


# ------------------------------------------------------------------------------------------ PROJECT
def project():
    """GPU-hours from banked rates (sources in M1_SPEC.md §9.7)."""
    from infra.pg import m1
    rows = {}
    boot_h = 164 / 3600                       # mechanics screen: launch to STATUS 650 s minus job wall 486 s
    fixed_s = 155 + 145 + 60 + 60             # band-1 snapshot write, final write, counts, positive control
    for lane, tps, mbps in (("AWS L40S", 3334.3, 234.3), ("AWS L40S low", 3215.0, 143.0),
                            ("Vast RTX PRO 5000", 4612.0, 234.3)):
        k0_s = 14.61e9 / (mbps * 1e6) + (m1.K0_TOKENS // 2048) * 2047 / tps + 145 + 60
        refs_s = 6 * 0.6e9 / 150e6 + 6 * 30
        h_bf16 = (14.61e9 / (mbps * 1e6) + 20560068 / tps + fixed_s + 30) / 3600
        h_f32 = (29.19e9 / (mbps * 1e6) + 20560068 / tps + fixed_s + 90) / 3600
        l1 = boot_h + (k0_s + refs_s) / 3600
        a = boot_h + h_bf16 + 2 * h_f32          # start_a, M, R
        b = boot_h + h_bf16 + h_f32              # start_b, G
        rows[lane] = {"K0_launch": round(l1, 2), "harvest_bf16": round(h_bf16, 2), "harvest_f32": round(h_f32, 2),
                      "instance_A": round(a, 2), "instance_B": round(b, 2), "total": round(l1 + a + b, 2)}
        print(f"PROJECT {lane}: {json.dumps(rows[lane])}")
    check("PROJECT total at the low L40S rate within the 12 GPU-h ceiling", rows["AWS L40S low"]["total"] <= 12.0,
          rows["AWS L40S low"]["total"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--corpus-manifest", required=True, help="the real corpus manifest.json (sha256 0c99a8e8...)")
    ap.add_argument("--t15-manifest", required=True, help="the banked T_15 harvest's manifest.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scratch = pathlib.Path(a.out)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    os.environ.update(HF_TOKEN="dryrun-no-network", HF_HUB_CACHE=str(scratch / "hf_cache"),
                      M1_WORK=str(scratch / "work"), HF_HUB_DISABLE_TELEMETRY="1")
    hf, grid = pathlib.Path(a.hf), pathlib.Path(a.grid)
    from infra.pg import m1
    cman = pathlib.Path(a.corpus_manifest)
    check("REAL corpus manifest sha256 registered", m1.sha256_file(cman) == m1.SAME["corpus_manifest_sha256"])
    manifest = json.loads(cman.read_text())
    print("== REAL")
    real(hf, scratch)
    real_t15(pathlib.Path(a.t15_manifest), grid)
    print("== WIN")
    win()
    print("== ROOM")
    r = room(grid, manifest)
    k = k0_offline(grid)
    print("== JOB and FAIL")
    banks = job(hf, scratch)
    print("== VERDICT")
    verdict_cases(scratch, banks)
    print("== PROJECT")
    p = project()
    (scratch / "dryrun_summary.json").write_text(json.dumps({"room": r, "k0": k, "project": p,
                                                             "checks": RESULTS}, indent=1))
    failed = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS)} checks, {len(failed)} failed: {failed}")
    assert not failed, failed
    print("M1_DRYRUN_OK")


if __name__ == "__main__":
    main()
