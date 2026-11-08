#!/usr/bin/env python3
"""Confound check: is the measured top-32 activation concentration inherited
from the RMSNorm gain vector γ (a weight property), rather than activation
dynamics? q_proj_in reads γ_in ⊙ rmsnorm(x); gate_proj_in reads γ_post ⊙ ...;
o_proj_in has no norm in front (internal control).

Fetches ONLY the tiny norm-gain tensors via HTTP byte-range from the HF
safetensors shards (no model download), then per (layer, norm-site):
  overlap@32   |top32(sum_sq) ∩ top32(γ²)| / 32
  spearman     rank correlation between sum_sq and γ² across coordinates

Usage: python3 gamma_check.py <actstats.npz> <flags.json> <hf_repo> --out out.json
"""
import argparse
import json
import struct
import urllib.request

import numpy as np

SITE_NORM = {"q_proj_in": "input_layernorm",
             "gate_proj_in": "post_attention_layernorm"}


def fetch(url, start=None, end=None):
    req = urllib.request.Request(url, headers={"User-Agent": "gamma-check"})
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    return urllib.request.urlopen(req).read()


def shard_header(url):
    n = struct.unpack("<Q", fetch(url, 0, 7))[0]
    return json.loads(fetch(url, 8, 8 + n - 1)), 8 + n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("flags")
    p.add_argument("repo")
    p.add_argument("--out", default="gamma_check.json")
    a = p.parse_args()

    base = f"https://huggingface.co/{a.repo}/resolve/main"
    index = json.loads(fetch(f"{base}/model.safetensors.index.json"))
    wmap = index["weight_map"]

    # group wanted tensors by shard, fetch each shard's header once
    wanted = {k: v for k, v in wmap.items() if k.endswith("norm.weight")}
    headers = {}
    gammas = {}
    for name, shard in sorted(wanted.items()):
        url = f"{base}/{shard}"
        if shard not in headers:
            headers[shard] = shard_header(url)
        hdr, data0 = headers[shard]
        meta = hdr[name]
        assert meta["dtype"] == "BF16", (name, meta["dtype"])
        s, e = meta["data_offsets"]
        raw = fetch(url, data0 + s, data0 + e - 1)
        u16 = np.frombuffer(raw, dtype=np.uint16)
        gammas[name] = (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)

    z = np.load(a.npz)
    flags = json.load(open(a.flags))
    out = []
    for row in flags["per_site"]:
        site = row["site"]
        if site not in SITE_NORM:
            continue
        li = row["layer"]
        g = gammas[f"model.layers.{li}.{SITE_NORM[site]}.weight"]
        key = f"L{li:02d}_{site}"
        ss = z[f"{key}_sum_sq"]
        assert ss.shape == g.shape
        g2 = g * g
        top_ss = set(np.argsort(ss)[::-1][:32])
        top_g = set(np.argsort(g2)[::-1][:32])
        rs = np.argsort(np.argsort(ss)).astype(np.float64)
        rg = np.argsort(np.argsort(g2)).astype(np.float64)
        spearman = float(np.corrcoef(rs, rg)[0, 1])
        out.append({"layer": li, "site": site,
                    "overlap_at_32": len(top_ss & top_g) / 32,
                    "spearman_sumsq_gamma2": round(spearman, 4)})
    ov = [r["overlap_at_32"] for r in out]
    sp = [r["spearman_sumsq_gamma2"] for r in out]
    summary = {"repo": a.repo, "n_cells": len(out),
               "overlap_at_32": {"median": float(np.median(ov)), "max": max(ov)},
               "spearman": {"median": float(np.median(sp)), "max": max(sp)},
               "rows": out}
    json.dump(summary, open(a.out, "w"), indent=2)
    print(f"[gamma] overlap@32 median={np.median(ov):.3f} max={max(ov):.3f} | "
          f"spearman median={np.median(sp):.3f} max={max(sp):.3f} -> {a.out}")


if __name__ == "__main__":
    main()
