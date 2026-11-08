#!/usr/bin/env python3
"""Standard γ-corrected concentration column (E8): divide banked sum_sq by the
RMSNorm gain² (byte-range fetched, no model download) and recompute top-32
share + random-band excess. o_proj_in passes through unchanged (no norm).

Usage: python3 analysis_gamma.py <npz> <flags.json> <hf_repo> --out out.json
"""
import argparse
import json

import numpy as np

import gamma_check as gc

SITE_NORM = {"q_proj_in": "input_layernorm",
             "gate_proj_in": "post_attention_layernorm"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("flags")
    p.add_argument("repo")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()

    base = f"https://huggingface.co/{a.repo}/resolve/main"
    index = json.loads(gc.fetch(f"{base}/model.safetensors.index.json"))
    wanted = {k: v for k, v in index["weight_map"].items()
              if k.endswith("norm.weight")}
    headers, gammas = {}, {}
    for name, shard in sorted(wanted.items()):
        url = f"{base}/{shard}"
        if shard not in headers:
            headers[shard] = gc.shard_header(url)
        hdr, d0 = headers[shard]
        m = hdr[name]
        s, e = m["data_offsets"]
        raw = gc.fetch(url, d0 + s, d0 + e - 1)
        u16 = np.frombuffer(raw, dtype=np.uint16)
        gammas[name] = (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)

    z = np.load(a.npz)
    flags = json.load(open(a.flags))
    rng = np.random.default_rng(a.seed)

    def excess(vec):
        d = vec.size
        tot = vec.sum()
        top = vec[np.argsort(vec)[::-1][:32]].sum() / tot
        draws = np.array([vec[rng.choice(d, 32, replace=False)].sum() / tot
                          for _ in range(1000)])
        return float(top), float(top / np.quantile(draws, 0.99))

    rows = []
    for row in flags["per_site"]:
        li, site = row["layer"], row["site"]
        ss = z[f"L{li:02d}_{site}_sum_sq"].copy()
        g2 = None
        if site in SITE_NORM:
            g2 = gammas[f"model.layers.{li}.{SITE_NORM[site]}.weight"] ** 2
            ss = ss / np.maximum(g2, 1e-12)
        top, ex = excess(ss)
        rec = {"layer": li, "site": site, "corrected_top32": top,
               "excess_p99": ex}
        if g2 is not None:
            # director rider (queue, 2026-08-16): dividing by gamma^2 can
            # MANUFACTURE spikes at tiny-gamma coordinates. Flag a corrected
            # max living in the bottom gamma decile, and ship a gamma-floor
            # robustness row (bottom decile excluded) beside every table.
            mx = int(np.argmax(ss))
            pct = float((g2 <= g2[mx]).mean())
            floor = g2 >= np.quantile(g2, 0.10)
            _, ex_floor = excess(ss[floor])
            rec.update({"max_coord_gamma_pct": round(pct, 4),
                        "tiny_gamma_max_flag": pct < 0.10,
                        "excess_p99_gamma_floor": ex_floor})
        else:
            rec.update({"excess_p99_gamma_floor": ex})
        rows.append(rec)
    json.dump({"repo": a.repo, "rows": rows}, open(a.out, "w"), indent=1)
    ex = [r["excess_p99"] for r in rows]
    print(f"[gamma-corrected] {a.repo}: excess med={np.median(ex):.1f}x max={max(ex):.1f}x -> {a.out}")


if __name__ == "__main__":
    main()
