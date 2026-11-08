"""M1: the three stage-2 data mixes of OLMo-3-7B in the neuron basis (studies/next/07_PICKS/M1_SPEC.md).

Numpy only (the job imports torch itself). Users:
  - the job `infra/aws_pg/jobs/m1_harvest.py` on the instance: registry, gates, the per-pole directory counts;
  - the laptop CLI below: `freeze` (thresholds from the banked references, before any mix is read) and
    `verdict` (run once, after freeze 2);
  - the dry run `infra/aws_pg/jobs/m1_dryrun.py`.

    python -m infra.pg.m1 freeze  --banks <dir> --out <thresholds.json>
    python -m infra.pg.m1 verdict --banks <dir> --thresholds <thresholds.json> --out <verdict_m1.json>

<dir> holds one sub-directory per checkpoint: start_a, start_b, G, R, M (each a harvest step's out dir:
metrics.json, controls.json, manifest.json, m1/counts.npz, m1/metrics.json) and refs/<name>/m1/counts.npz for
the banked references t00, t12, t13, t14b, t14c, t15.

The quantity. For pole p (layer l, neuron n, sign s) at checkpoint c: the harvest's exact top-50 positions
`topk/layer_LL.npz` -> `tok[n, s, :]` (global token indices of the fixed corpus). Each position lies in one
corpus document and so in one of the 50 corpus directories. counts[l, n, s, d] = number of the pole's filled
top-50 slots in directory d. cm_share = (slots in the 5 code and math directories) / (filled slots).

Fail loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

REPO = "allenai/Olmo-3-1025-7B"

# ---------------------------------------------------------------- registry (HF API, read 2026-11-05)
# commit: /api/models/<repo>/refs; weights: LFS sha256 from /api/models/<repo>/tree/<branch> for exactly the
# files the branch's model.safetensors.index.json names. Saved reads: the scratchpad m1_gate/hf/ folder named in
# studies/next/07_PICKS/M1_OFFLINE_GATE.md.
_B3 = [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]
_F6 = [f"model-0000{i}-of-00006.safetensors" for i in range(1, 7)]
BRANCHES = {
    "stage1-step476838": {"key": "start", "commit": "31bf90b3211621ce8154e55e7fec5b5eda08e620", "dtype": "BF16",
                          "rewrite": False, "weights": dict(zip(_B3, (
                              "6005c96a297cb1960f5e763deb331b1c1cc4c9ae7037889637395abfaae49ee7",
                              "da953fe9917731e78d1305e6df4349530f60d15205ad4610612552d10e277d40",
                              "c5696461572e9e71c03a44050e9932c430bee38249c3e3e5a068e6eda33d909d")))},
    "stage2-step47684-mix-gen-mc-only-from-2T-ckpt": {
        "key": "G", "commit": "45786899c6c427d17f9291f3d97c7e4fcbd4962b", "dtype": "F32", "rewrite": True,
        "weights": dict(zip(_F6, (
            "e8e8abbfd63fe4dad103b43f9c433c01fe5f7da7ed6ff87c5ac9ece30a043b9a",
            "453c2ae4ebcbdcac9c4c98a063a1d5bd73eba7db9d6438a7a117edbda6b73d46",
            "982c106e6e5eb7babe8cb48f0ad17e1152ef42912576f371a03954c5dd61865e",
            "3ec0366cc7e8a4877a864a789a249a77d903d3c408d44713d3449c9f0eef0e9c",
            "e50e3dfb3866c4ff1787e278333d94f836b45b7aa3e92857c730adb03f806b06",
            "81c6f17da24e3927c6416a17067e91ea4249970b1c8037dfd41461570e271bc2")))},
    "stage2-step47684-mix-round5-from-2T-ckpt": {
        "key": "R", "commit": "34e13d5fae4d1aacf4aa019c377a332ed09ff136", "dtype": "F32", "rewrite": True,
        "weights": dict(zip(_F6, (
            "8e8c22d2fe10c0cb72b3992ed07b142b8809d0818ddd02efa9ea4be3e8d9b0db",
            "89b4cbf3cfcb77048d235df63de1b5c6b297ca47243f9ad3b830fe48e6c647f4",
            "8d2219cefc41a5a4a66cb520d0f3d96d66decffaa14a16dabf1f38df895215f2",
            "a0e4d0ef57c64ee86ad0d9100b3aaa44e0a9e6660a7d6cd1503aedf6a9cb7be0",
            "f081c89197f1eaefa3c352a9ae76d06d877959085a5dd4cf2f6d89a81ca98fe2",
            "55509870bf443ccaa8b5a9d5a13977a565d869d3ed60a4332dd17dfb5817b55d")))},
    "stage2-step47684-mix-math-code-reasoning-web-from-2T-ckpt": {
        "key": "M", "commit": "ab9f4b070e9125de257bfbe31bf8cc777daadfb4", "dtype": "F32", "rewrite": True,
        "weights": dict(zip(_F6, (
            "7c4231e0123f6252253522cda5869d021597a8f5db19899c8d071862bc7fe16a",
            "fe69ddc329e5e8d56b61ed096f8f610ac5a397dc364b0e117fd363759e32b5a9",
            "717bb44705fa0d5ab69ea4d6e28783041ee361b43adb14c96f254cd11acc55e9",
            "dd44c4be4081100375a9699587f7611c28f65c7e8ba757fec7ad98d0c67ae482",
            "9bbe71d55586ba1705c32218e7771db57e4e34a77df1e943ee27c85579e9cb56",
            "ebc7c8f5e092adaa8f6c82ce7fb596c54511d973449c8b612afeef7a9023aaaf")))},
    "stage1-step1413814": {"key": "T15", "commit": "373bad25002f1624757a73235c5ca844c6375c25", "dtype": "BF16",
                           "rewrite": False, "weights": dict(zip(_B3, (
                               "f256015ffbd26278172c0832bde17448526dd880430c70035a1ec7e98e0f3756",
                               "de327e942a56a61d8ab2648b61606c5daa96f41b090cd18a6f159f4ab17d4676",
                               "0262ab848daa983ba79ddfe2929f58a4614c84e3f70b972ca40700cbab0937fb")))},
    "stage1-step1000000": {"key": "T14", "commit": "cb286dcdaaf3f92b3c419dd77439078fdebffe53", "dtype": "BF16",
                           "rewrite": False, "weights": dict(zip(_B3, (    # the known number on the Vast lane only
                               "2940a4230fddc76bc64486542144f496fed4de7ca88b175392323577c567c151",
                               "a75b7b4afa9a191f0b8d7f3b69ba3b6ce680d18f6595122dd027c109255453c1",
                               "cf359431b10df43f955fe5d3d00be42f57f9a0ddf138c18be7c52fbee19022b8")))},
}
# The three bf16 shards every mix branch also carries: their LFS oids equal `main` (= stage3-step11921), the final
# base model. Loading any of them for a mix end is the failure the shard gate exists to catch.
STALE = dict(zip(_B3, ("0490d6668e613a29b23367e3a7aa9cc6aced3d162694445bb969ed7622b3c4e2",
                        "e127ea479fb6e208fe9d48d23b11212b5722f4873f6eef9c009b7a855866c641",
                        "f3ddff10052ffe5de5c6b4cad45c422c0d898acc6beb21b1b8531244adfb3c70")))
NONWEIGHT = ["config.json", "generation_config.json", "model.safetensors.index.json", "special_tokens_map.json",
             "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
# config.json of stage1-step476838 (git blob 62e31c349819; identical blob on stage1-step1413814, stage1-step476000,
# stage2-step47684). The mix ends' config (blob a9df2fe3b454) differs in exactly architectures, model_type and
# sliding_window (4097).
REF_CONFIG = json.loads(
    '{"architectures": ["Olmo3ForCausalLM"], "attention_bias": false, "attention_dropout": 0, "eos_token_id": 100257,'
    ' "hidden_act": "silu", "hidden_size": 4096, "initializer_range": 0.02, "intermediate_size": 11008, "layer_types":'
    ' ' + json.dumps((["sliding_attention"] * 3 + ["full_attention"]) * 8) + ', "max_position_embeddings": 8192,'
    ' "model_type": "olmo3", "num_attention_heads": 32, "num_hidden_layers": 32, "num_key_value_heads": 32,'
    ' "pad_token_id": 100277, "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 500000, "sliding_window": 4096,'
    ' "tie_word_embeddings": false, "torch_dtype": "bfloat16", "transformers_version": "4.54.0.dev0", "use_cache": true,'
    ' "vocab_size": 100278}')
# Tensors the cast gate reads back from the shard files (MLP first: the harvest reads the MLP).
CAST_NAMES = ["model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight",
              "model.layers.15.mlp.down_proj.weight", "model.layers.31.mlp.down_proj.weight",
              "model.layers.16.self_attn.q_proj.weight", "model.layers.31.post_feedforward_layernorm.weight",
              "model.norm.weight"]
# The banked grid harvest these runs must equal in every setting (T_15 manifest, s3 grid/1, read 2026-11-05).
SAME = {"model_key": "O7", "tokname": "olmo3", "max_tokens": 20570743, "snapshot_bands": [1], "batch": 8,
        "window": 2048, "n_windows": 10044, "n_tokens_scored": 20560068, "topk_K": 50, "poles_K": 1000,
        "corpus_manifest_sha256": "0c99a8e86856f1d94f08535b252d541f2ab18c4cbcb486c9e4993d6eb70a305c",
        "tokens_sha256": "2aca80ca6537f46b8f60a550b4a3f2f13be2d18c5402be8f1af5c00caa86f531",
        "bos_id": 100257, "model_type": "olmo3", "model_class": "Olmo3ForCausalLM", "act_fn_measured": "silu",
        "dtype": "torch.bfloat16", "attn": "sdpa", "transformers": "5.9.0", "torch": "2.12.0+cu130"}
# Laptop banks the job's inputs are copied from (sha256 of the files under
# /Users/abaybektursun/projects/interp_research_banks/neurons/w3/grid/, read 2026-11-05).
BANK_SHA = {"t15_contexts.jsonl": "a4d46cd837f7a7532c300cac7372df1908e37f25b5acf57b6ea1ea37493e3269",
            "t15_exemplar_a.npz": "84bd59fdaf6b45727778b86ae82602b6815d4b8cda591a1237f022aa09ace300",
            "t14_exemplar_a.npz": "9fabd78be6d81b5c4cecbc52e9ca883ec9a0ea3db5f2a8067c86e8e25c37e09e",
            "t14_contexts.jsonl": "692e85854bebb5446522b6c3cb16138bee0d57c581de5d1ecd48595f645ddc9a",
            "t13_exemplar_a.npz": "7976ca719702c6cdb1dc7db5ea95c296e43893be76438c3f9c2173c257e35d8d"}
N_POLES_GRID = 640
K0_TOKENS = 1 << 20                     # 512 windows of 2,048: the known-number prefix
# K0 tolerance: the cross-device repeat of T_14 (Blackwell against L40S, same code) gave relative differences of
# the 640 x 50 exemplar activations of median 0.43 % and p99 2.22 % (studies/already-clean/JOURNAL.md,
# 2026-10-21T19:31:14Z, artifact results/neurons/w3/device_check/t14_1b_vs_1c.json). K0 runs on the same device
# type as its bank, so these cross-device values are an upper bound.
K0_TOL = {"median_rel": 0.0043, "p99_rel": 0.0222, "subset_share": 0.99}
NLL_BAND = (2.40, 3.20)  # fit gate: corpus NLL, first 120,000 tokens; banked T_8..T_15 span 3.013..2.675
PC_FLOOR_MIN = 0.01      # nats; the positive control's floor is max(this, the identity pair's NLL difference)
PC_CEIL = 2.0            # nats; a larger mix difference on the corpus means a broken model
PC_K = 32                # windows per group for the positive control
N_BOOT, SEED = 2000, 0
N_RANDOM_SETS = 1000
REF_PAIRS = [("t12", "t13"), ("t13", "t14b"), ("t14c", "t15")]   # neighbouring banked points, same device type
REF_IDENTITY = ("t14b", "t14c")                                   # banked cross-device identity pair


def load_registry(path=None) -> dict:
    """The registered facts; `path` (dry run only) replaces them with a planted set of the same shape."""
    if path is None:
        return {"branches": BRANCHES, "stale": STALE, "ref_config": REF_CONFIG, "same": SAME,
                "bank_sha": BANK_SHA, "cast_names": CAST_NAMES, "nonweight": NONWEIGHT, "pc_k": PC_K,
                "k0_tokens": K0_TOKENS, "nll_band": list(NLL_BAND)}
    return json.loads(pathlib.Path(path).read_text())


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- corpus: document -> directory -> group
def group_of(d: str) -> str:
    n = d.split("/", 1)[1]
    if n.startswith("stack_edu-"):
        return "code"
    if n == "finemath-3plus":
        return "math"
    if n.startswith("olmocr_science_pdfs-"):
        return "pdf"
    if n.startswith("common_crawl-"):
        return "web"
    raise AssertionError(f"directory {d} has no registered group")


def doc_dirs(manifest: dict) -> np.ndarray:
    """Directory index of every corpus document. Corpus order (manifest band_rule): band 1 of every shard in shard
    order, then band 2, then band 3; a shard is one directory."""
    docdir = np.full(manifest["n_docs"], -1, np.int16)
    d0 = 0
    for band in sorted(manifest["bands"], key=int):
        assert d0 == manifest["bands"][band]["doc_start"], f"band {band} starts at document {d0}"
        for i, s in enumerate(manifest["shards"]):
            n = s["n_docs_selected"][band]
            docdir[d0:d0 + n] = i
            d0 += n
        assert d0 == manifest["bands"][band]["doc_end"], f"band {band} ends at document {d0}"
    assert (docdir >= 0).all()
    return docdir


def groups(manifest: dict) -> list:
    dirs = [s["dir"] for s in manifest["shards"]]
    assert dirs == manifest["dirs"], "manifest shards are not in directory order"
    return [group_of(d) for d in dirs]


def cm_mask(manifest: dict) -> np.ndarray:
    return np.array([g in ("code", "math") for g in groups(manifest)])


def pole_counts(tok: np.ndarray, doc_offsets: np.ndarray, docdir: np.ndarray, n_dirs: int):
    """tok [..., K] global token indices (-1 = empty slot) -> (counts uint8 [..., n_dirs], filled uint8 [...])."""
    valid = tok >= 0
    doc = np.searchsorted(doc_offsets, np.where(valid, tok, 0), "right") - 1
    d = np.where(valid, docdir[doc], n_dirs).reshape(-1, tok.shape[-1]).astype(np.int64)
    rows = d.shape[0]
    flat = (d + (np.arange(rows) * (n_dirs + 1))[:, None]).ravel()
    c = np.bincount(flat, minlength=rows * (n_dirs + 1)).reshape(rows, n_dirs + 1)[:, :n_dirs]
    assert c.max() <= 255
    return c.astype(np.uint8).reshape(tok.shape[:-1] + (n_dirs,)), valid.sum(-1).astype(np.uint8)


def token_share_per_dir(doc_offsets, docdir, n_dirs, n_tokens) -> list:
    """Tokens of the scored range [0, n_tokens) per directory (whole documents up to n_tokens)."""
    lens = np.diff(np.minimum(doc_offsets, n_tokens))
    return np.bincount(docdir.astype(np.int64), weights=lens, minlength=n_dirs).astype(np.int64).tolist()


def pc_windows(doc_offsets, docdir, grp: list, window: int, n_windows: int, k: int = PC_K, seed: int = SEED) -> dict:
    """The positive control's windows: harvest windows w (tokens [w*window, (w+1)*window)) whose every token lies
    in documents of one group; k per group, drawn by numpy default_rng(seed) without replacement, sorted."""
    starts = np.arange(n_windows) * window
    d_first = docdir[np.searchsorted(doc_offsets, starts, "right") - 1]
    d_last = docdir[np.searchsorted(doc_offsets, starts + window - 1, "right") - 1]
    g = np.array(grp)
    out = {}
    for name, members in (("cm", ("code", "math")), ("web", ("web",))):
        ok = np.isin(g[d_first], members) & np.isin(g[d_last], members)
        # every document between the first and last must also be in the group: check each window's span
        cand = []
        for w in np.flatnonzero(ok):
            a = np.searchsorted(doc_offsets, starts[w], "right") - 1
            b = np.searchsorted(doc_offsets, starts[w] + window - 1, "right") - 1
            if np.isin(g[docdir[a:b + 1]], members).all():
                cand.append(int(w))
        assert len(cand) >= k, f"group {name}: {len(cand)} candidate windows, need {k}"
        rng = np.random.default_rng(seed)
        out[name] = sorted(int(x) for x in rng.choice(cand, size=k, replace=False))
        out[f"n_candidates_{name}"] = len(cand)
    return out


# ---------------------------------------------------------------- gates on the instance (each returns a record)
def rewrite_config(cfg: dict) -> dict:
    """The loader fix: `olmo2-retrofit` / `Olmo2RetrofitForCausalLM` (unknown to transformers 5.9.0) become
    `olmo3` / `Olmo3ForCausalLM`. Every other key, sliding_window 4097 included, is kept."""
    assert cfg["model_type"] == "olmo2-retrofit" and cfg["architectures"] == ["Olmo2RetrofitForCausalLM"], \
        f"rewrite asked for a config of type {cfg['model_type']}"
    return {**cfg, "model_type": "olmo3", "architectures": ["Olmo3ForCausalLM"]}


def config_gate(cfg: dict, ref: dict) -> dict:
    """G-CFG: the config that will be loaded equals the registered olmo3 config except sliding_window, which may
    be 4096 or 4097 (no effect at a 2,049-position forward; offline gate G-WIN)."""
    diff = sorted(k for k in set(cfg) | set(ref) if cfg.get(k) != ref.get(k))
    ok = (cfg.get("model_type") == "olmo3" and cfg.get("architectures") == ["Olmo3ForCausalLM"]
          and set(diff) <= {"sliding_window"} and cfg.get("sliding_window") in (4096, 4097))
    return {"pass": bool(ok), "diff_keys": diff, "sliding_window": cfg.get("sliding_window")}


def shard_gate(found: dict, index_files: set, entry: dict, stale: dict) -> dict:
    """G-SHARD: the weight files on disk are exactly the registered ones, each with its registered sha256; the
    index names exactly those files; none of them is a stale shard of `main` ({file name: sha256})."""
    want, bad = entry["weights"], set(stale.values())
    ok = (set(found) == set(want) and set(index_files) == set(want)
          and all(found[f] == want[f] for f in want) and not (set(found.values()) & bad))
    return {"pass": bool(ok), "files": sorted(found), "index_files": sorted(index_files),
            "mismatch": sorted(f for f in want if found.get(f) != want[f]),
            "stale_present": sorted(f for f, h in found.items() if h in bad)}


def rel_diff(new: np.ndarray, bank: np.ndarray) -> np.ndarray:
    return np.abs(new.astype(np.float64) - bank) / np.abs(bank.astype(np.float64))


def k0_values_gate(new_a: np.ndarray, bank_a: np.ndarray, mask: np.ndarray, tol=K0_TOL) -> dict:
    r = rel_diff(new_a[mask], bank_a[mask])
    med, p99 = float(np.median(r)), float(np.quantile(r, 0.99))
    return {"pass": bool(med <= tol["median_rel"] and p99 <= tol["p99_rel"]), "n": int(mask.sum()),
            "median_rel": med, "p99_rel": p99, "max_rel": float(r.max()),
            "exact_share": float((new_a[mask] == bank_a[mask]).mean())}


def k0_subset_gate(new_tok: np.ndarray, bank_tok: np.ndarray, n_tokens: int, tol=K0_TOL) -> dict:
    """Every banked global top-50 position inside the prefix must be in the prefix run's top-50 of that pole
    (a global top-50 position in the prefix is a prefix top-50 position)."""
    hit = tot = 0
    for i in range(bank_tok.shape[0]):
        b = bank_tok[i][(bank_tok[i] >= 0) & (bank_tok[i] < n_tokens)]
        tot += b.size
        hit += int(np.isin(b, new_tok[i]).sum())
    share = hit / max(tot, 1)
    return {"pass": bool(tot > 0 and share >= tol["subset_share"]), "n": tot, "share": share}


def fit_gate(metrics: dict, controls: dict, manifest: dict, same: dict, nll_band=NLL_BAND) -> dict:
    """G-FIT and G-SAME for one harvest: complete, G0, no dead pole, all tokens scored, corpus NLL in band, and the
    manifest equal to the banked grid harvest in every registered setting."""
    nll = controls["nll_corpus_first_120k"]["mean_nll"]
    same_diff = sorted(k for k in same if manifest.get({"max_tokens": "max_tokens_arg"}.get(k, k)) != same[k])
    ok_m = (metrics["complete"] is True and metrics["G0_pass"] is True and metrics["dead_poles"] == 0
            and metrics["n_tokens_scored"] == same["n_tokens_scored"])
    ok_nll = nll_band[0] <= nll <= nll_band[1]
    return {"pass": bool(ok_m and ok_nll and not same_diff), "metrics_ok": bool(ok_m), "nll": nll,
            "nll_band": list(nll_band), "nll_ok": bool(ok_nll), "same_diff": same_diff}


# ---------------------------------------------------------------- analysis (laptop)
def load_counts(d) -> dict:
    z = np.load(pathlib.Path(d) / "m1" / "counts.npz")
    return {"counts": z["counts"], "filled": z["filled"], "cm": z["cm_mask"], "sha256": sha256_file(
        pathlib.Path(d) / "m1" / "counts.npz")}


def share(c: dict, mask=None) -> np.ndarray:
    """[L, n, 2] share of filled top-50 slots in the directories of `mask` (default: code + math)."""
    m = c["cm"] if mask is None else mask
    f = c["filled"].astype(np.float64)
    assert (f > 0).all(), "a pole with no filled slot"
    return c["counts"][..., m].sum(-1) / f


def band_index(n_layers: int) -> np.ndarray:
    return (np.arange(n_layers) * 4) // n_layers


def boot_mean(v: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple:
    rng = np.random.default_rng(seed)
    n = v.size
    m = np.array([v[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(v.mean()), float(lo), float(hi)


def band_diff(x: np.ndarray, y: np.ndarray) -> list:
    """Per layer band: mean over poles of x - y (shares, [L, n, 2]), paired by pole; 95 % bootstrap interval over
    neurons (the two poles of a neuron resampled together), 2,000 resamples, numpy default_rng(0)."""
    d = (x - y).mean(-1)                      # per neuron: mean of its two poles
    b = band_index(d.shape[0])
    return [boot_mean(d[b == k].ravel()) for k in range(4)]


def band_means(x: np.ndarray) -> list:
    b = band_index(x.shape[0])
    return [float(x[b == k].mean()) for k in range(4)]


def label_lead(m, lo, hi, bar, ceil) -> str:
    if m > ceil:
        return "ABOVE_CEILING"
    if lo > bar:
        return "SUPPORTED"
    if hi < bar:
        return "REFUTED"
    return "UNDECIDED"


def label_between(g1, g2, bar) -> str:
    """R between G and M: g1 = R - G, g2 = M - R, each (mean, lo, hi)."""
    if g1[1] > bar and g2[1] > bar:
        return "SUPPORTED"
    if g1[2] < -bar or g2[2] < -bar:
        return "REFUTED"
    return "UNDECIDED"


def combine(labels: list) -> str:
    if "ABOVE_CEILING" in labels:
        return "ABOVE_CEILING"
    if all(x == "SUPPORTED" for x in labels):
        return "SUPPORTED"
    if "REFUTED" in labels:
        return "REFUTED"
    return "UNDECIDED"


def dir_band_diff(a: dict, b: dict) -> np.ndarray:
    """[4, n_dirs]: per band, mean over poles of (a's share in directory d) - (b's share in d)."""
    fa, fb = a["filled"].astype(np.float64)[..., None], b["filled"].astype(np.float64)[..., None]
    d = a["counts"] / fa - b["counts"] / fb
    bi = band_index(d.shape[0])
    return np.stack([d[bi == k].mean(axis=(0, 1, 2)) for k in range(4)])


def specificity(a: dict, b: dict, n_sets: int = N_RANDOM_SETS, seed: int = SEED) -> list:
    """The code+math contrast against random sets of 5 other directories (equal directory count; each directory
    holds about 0.4M tokens of bands 1-2 by the band rule), 1,000 sets, numpy default_rng(0)."""
    dd = dir_band_diff(a, b)
    cm = a["cm"]
    others = np.flatnonzero(~cm)
    k = int(cm.sum())
    rng = np.random.default_rng(seed)
    sets = np.array([rng.choice(others, size=k, replace=False) for _ in range(n_sets)])
    out = []
    for band in range(4):
        x = float(dd[band, cm].sum())
        r = np.abs(dd[band][sets].sum(-1))
        q99, q50 = float(np.quantile(r, 0.99)), float(np.median(r))
        lab = "SUPPORTED" if x > q99 else ("REFUTED" if x <= q50 else "UNDECIDED")
        out.append({"cm": x, "random_abs_q99": q99, "random_abs_median": q50, "label": lab})
    return out


def pc_gate(nll: dict, floor: float) -> dict:
    """Positive control: on the corpus's own code+math windows the math-code mix M must predict better than the
    general mix G, and by more than on web windows. nll: {'M': {'cm': [...], 'web': [...]}, 'G': {...}}."""
    rng = np.random.default_rng(SEED)
    mc = np.array(nll["M"]["cm"]) - np.array(nll["G"]["cm"])
    mw = np.array(nll["M"]["web"]) - np.array(nll["G"]["web"])
    bc, bd = [], []
    for _ in range(N_BOOT):
        i, j = rng.integers(0, mc.size, mc.size), rng.integers(0, mw.size, mw.size)
        bc.append(mc[i].mean())
        bd.append(mc[i].mean() - mw[j].mean())
    cm = (float(mc.mean()),) + tuple(float(x) for x in np.quantile(bc, [0.025, 0.975]))
    dd = (float(mc.mean() - mw.mean()),) + tuple(float(x) for x in np.quantile(bd, [0.025, 0.975]))
    f = max(floor, PC_FLOOR_MIN)
    ok = cm[2] < -f and dd[2] < -f and abs(cm[0]) < PC_CEIL
    return {"pass": bool(ok), "cm_M_minus_G": cm, "cm_minus_web": dd, "floor": f, "ceiling": PC_CEIL}


def identity_gate(a: dict, b: dict, tau_min: float) -> dict:
    """G-ID: two harvests of the same weights. Band-mean share difference within tau_min / 4."""
    I = [m for m, _, _ in band_diff(share(b), share(a))]
    changed = float((a["counts"] != b["counts"]).any(-1).mean())
    return {"pass": bool(max(abs(x) for x in I) <= tau_min / 4), "I": I, "line": tau_min / 4,
            "changed_pole_share": changed}


def freeze(banks: pathlib.Path) -> dict:
    """Freeze 2: tau, kappa and the banked identity spread from the references (all poles), before any mix is
    read. The room check runs the lead labels on every reference pair as if it were (G, M), both ways."""
    refs = {n: load_counts(banks / "refs" / n) for n in ("t00", "t12", "t13", "t14b", "t14c", "t15")}
    C = {n: band_means(share(c)) for n, c in refs.items()}
    tau = [max(abs(C[j][k] - C[i][k]) for i, j in REF_PAIRS) for k in range(4)]
    eps_dev = [abs(C[REF_IDENTITY[1]][k] - C[REF_IDENTITY[0]][k]) for k in range(4)]
    names = list(refs)
    kappa = [max(abs(C[i][k] - C[j][k]) for i in names for j in names) for k in range(4)]
    bar = [max(t, e) for t, e in zip(tau, eps_dev)]
    room = []
    for i in names:
        for j in names:
            if i == j:
                continue
            bd = band_diff(share(refs[j]), share(refs[i]))
            room.append({"as_G": i, "as_M": j, "labels": [label_lead(m, lo, hi, bar[k], kappa[k])
                                                          for k, (m, lo, hi) in enumerate(bd)]})
    flat = [x for r in room for x in r["labels"]]
    return {"tau": tau, "eps_dev": eps_dev, "kappa": kappa, "bar_before_identity": bar, "C_refs": C,
            "room": {"pairs": room, "supported_reachable": "SUPPORTED" in flat, "refuted_reachable": "REFUTED" in flat,
                     "tau_below_kappa": all(t < k for t, k in zip(tau, kappa))},
            "refs_sha256": {n: c["sha256"] for n, c in refs.items()}}


def verdict(banks: pathlib.Path, th: dict, keys=("start_a", "start_b", "G", "R", "M")) -> dict:
    """The verdict, run once after freeze 2. Gates first; every readout is reported; the headline reads the gates."""
    present = [k for k in keys if (banks / k / "m1" / "counts.npz").exists()]
    c = {k: load_counts(banks / k) for k in present}
    gates = {}
    for k in present:
        rec = json.loads((banks / k / "m1" / "metrics.json").read_text())
        g = fit_gate(json.loads((banks / k / "metrics.json").read_text()),
                     json.loads((banks / k / "controls.json").read_text()),
                     json.loads((banks / k / "manifest.json").read_text()), rec["same"], rec["nll_band"])
        gates[k] = {"fit": g, "instance": rec["gates_pass"], "pass": bool(g["pass"] and rec["gates_pass"])}
    out = {"present": present, "parked": [k for k in keys if k not in present], "gates": gates, "thresholds": th}
    need = {"start_a", "start_b", "G", "M"}
    if not need <= set(present):
        out["headline"] = f"DOES NOT COUNT: missing {sorted(need - set(present))}"
        return out
    tau_min = min(th["tau"])
    gid = identity_gate(c["start_a"], c["start_b"], tau_min)
    gid_fail = identity_gate(c["start_a"], load_counts(banks / "refs" / "t15"), tau_min)
    gid["failing_input_pass"] = gid_fail["pass"]
    out["G_ID"] = gid
    bar = [max(b, abs(i)) for b, i in zip(th["bar_before_identity"], gid["I"])]
    out["bar"] = bar
    nll = {k: json.loads((banks / k / "m1" / "metrics.json").read_text())["pc_nll"]["nll"] for k in present}
    id_nll = float(np.mean(np.r_[nll["start_b"]["cm"], nll["start_b"]["web"]]
                           - np.r_[nll["start_a"]["cm"], nll["start_a"]["web"]]))
    pc = pc_gate({"M": nll["M"], "G": nll["G"]}, abs(id_nll))
    pc["identity_nll_diff"] = id_nll
    pc["failing_input_pass"] = pc_gate({"M": nll["G"], "G": nll["M"]}, abs(id_nll))["pass"]
    out["PC_NLL"] = pc
    s = {k: share(v) for k, v in c.items()}
    lead = band_diff(s["M"], s["G"])
    out["lead"] = [{"band": k, "mean": m, "lo": lo, "hi": hi, "bar": bar[k], "ceiling": th["kappa"][k],
                    "label": label_lead(m, lo, hi, bar[k], th["kappa"][k])} for k, (m, lo, hi) in enumerate(lead)]
    out["H_lead"] = combine([x["label"] for x in out["lead"]])
    if "R" in present:
        g1, g2 = band_diff(s["R"], s["G"]), band_diff(s["M"], s["R"])
        out["between"] = [{"band": k, "R_minus_G": g1[k], "M_minus_R": g2[k],
                           "label": (label_between(g1[k], g2[k], bar[k]) if out["lead"][k]["label"] == "SUPPORTED"
                                     else "NOT_EVALUATED")} for k in range(4)]
        out["H_order"] = ("REFUTED" if out["H_lead"] == "REFUTED" else
                          combine([x["label"] for x in out["between"]]) if out["H_lead"] == "SUPPORTED" else
                          out["H_lead"])
    else:
        out["H_order"] = "NOT_RUN (R parked)"
    out["specificity"] = specificity(c["M"], c["G"])
    out["H_specific"] = combine([x["label"] for x in out["specificity"]])
    out["samples"] = {k: json.loads((banks / k / "m1" / "metrics.json").read_text()).get("samples") for k in present}
    out["exploratory"] = {"shift_from_start": {k: band_diff(s[k], s["start_a"]) for k in ("G", "R", "M") if k in s},
                          "C": {k: band_means(v) for k, v in s.items()}}
    failed = [k for k, g in gates.items() if not g["pass"]]
    no_room = [n for n, g in (("G_ID", gid), ("PC_NLL", pc)) if g["pass"] and g["failing_input_pass"]]
    if failed or not gid["pass"] or not pc["pass"] or no_room:
        out["headline"] = (f"DOES NOT COUNT: harvest gates {failed}, G_ID {gid['pass']}, PC_NLL {pc['pass']}, "
                           f"failing input that did not fail {no_room}")
    elif out["H_lead"] == "ABOVE_CEILING":
        out["headline"] = "DOES NOT COUNT until inspected: lead contrast above the ceiling"
    else:
        out["headline"] = f"H_lead {out['H_lead']}; H_order {out['H_order']}; H_specific {out['H_specific']}"
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--banks", required=True)
    f.add_argument("--out", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--banks", required=True)
    v.add_argument("--thresholds", required=True)
    v.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "freeze":
        th = freeze(pathlib.Path(a.banks))
        assert th["room"]["supported_reachable"] and th["room"]["refuted_reachable"] and th["room"]["tau_below_kappa"], \
            f"no room: {th['room']}"
        pathlib.Path(a.out).write_text(json.dumps(th, indent=1))
        print("M1_FREEZE_OK", json.dumps({k: th[k] for k in ("tau", "eps_dev", "kappa")}))
    else:
        r = verdict(pathlib.Path(a.banks), json.loads(pathlib.Path(a.thresholds).read_text()))
        pathlib.Path(a.out).write_text(json.dumps(r, indent=1))
        print("M1_VERDICT", r["headline"])


if __name__ == "__main__":
    main()
