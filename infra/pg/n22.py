"""N22 with N25 (studies/next/07_PICKS/N22_SPEC.md): slot-dependent arithmetic on OLMo-3 7B over training, with the
lens instrument checks of N25. Pure logic and the laptop CLI; the instance job is infra/aws_pg/jobs/n22_score.py.

Objects (SPEC §0.5). An item is (a + b) * c + d with s1 = a + b, s2 = s1 * c, y = s2 + d, every number a single
OLMo-3 token ("0".."999"). The prompt is BOS, three fixed solved lines, then "(a + b) * c + d = " (63 tokens); the
answer is the next token. For a value v, f1(v) = v * c + d (v put in slot 1) and f2(v) = v + d (v put in slot 2).
An edit X applied to slot k with value v gives log-probabilities lp at the answer position. The 2 x 2 interaction is
I = [lp(f1 | X1) - lp(f1 | X2)] + [lp(f2 | X2) - lp(f2 | X1)] (nats; the log-partitions cancel, so it equals the
same expression in logits).

    PYTHONPATH=. python -m infra.pg.n22 items --tok <dir with tokenizer.json> --out studies/next/n22/items.json
    PYTHONPATH=. python -m infra.pg.n22 project > studies/next/n22/projection.json
    PYTHONPATH=. python -m infra.pg.n22 plan-lens --metrics <L2 n22/metrics.json>
    PYTHONPATH=. python -m infra.pg.n22 freeze --banks <dir> --out studies/next/n22/thresholds.json
    PYTHONPATH=. python -m infra.pg.n22 verdict --banks <dir> --thresholds <json> --out <json>

Fail-loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

from infra.pg import m2

REPO, POINTS, BOS = m2.REPO, m2.POINTS, m2.BOS
N_LAYERS = 32
TARGET = 30                                   # penultimate block output (01_JLENS.md §1.1; 03_IDEAS.md N22)
SOURCES = list(range(TARGET))                 # lens source layers 0..29
UPPER = list(range(N_LAYERS // 2, TARGET))    # the paper's "upper half of the network's layers" (§A.24.2): 16..29
IC_SITES = [-1, 0, 4, 8, 12, 16, 20, 24, 28]  # -1 = the embedding output; else a block output
DOM_SWEEP = list(range(0, TARGET, 2))         # the paper's single-layer mean-difference sweep (§A.24.1), freeze half
SEED_POOL, SEED_CLASS, SEED_LENS, SEED_TOKS = 20261106, 20261107, 20261108, 20261109
POOL_N, N_V, N_V_IC, N_CLASS = 1200, 5, 2, 16
N_MIN = 50                                    # answered items per half, else the point's labels do not count
N_SWAP_MAX = 200                              # answered items per half that enter the edits (first by index)
K_READ = 10                                   # readout hit: the token in the lens top 10 of the full vocabulary
N_LENS_PROMPTS, LENS_LEN, DIM_BATCH = 24, 128, 16   # two halves of 12; BOS + 127 corpus tokens
N_BOOT, SEED = 10000, 0
# Known number (controls.json nll_corpus_first_120k), frozen on control pairs read 2026-11-06:
#   same device: T_15 harvest vs M1 K0 equal (2.675372037760417 both); stage1-step476838 start_a vs start_b equal
#   (2.781479736328125 both); cross device: T_14 Blackwell 2.714761865234375 vs L40S 2.714700182088216 (6.168e-05).
K0_TOL_SAME = 1e-6
K0_REF_CROSS = 2 * 6.1683e-05                 # reported at the Blackwell-banked points T_7-T_14 (lesson 3, M2 A3)
K0_TOL_COARSE = 0.01                          # the gate there: 162 x the device pair, 3.4 x below the smallest gap
K0_BANK = {0: 12.2233822265625, 1: 4.563403108723958, 2: 3.9356994303385417, 3: 3.6571566162109375,
           4: 3.4425349283854167, 5: 3.2878328369140624, 6: 3.1876436604817706, 7: 3.0880104614257813,
           8: 3.013055735270182, 9: 2.9398125, 10: 2.895995829264323, 11: 2.8503530151367187,
           12: 2.8164237569173176, 13: 2.7708053771972656, 14: 2.714761865234375, 15: 2.675372037760417}
K0_TARGETS, K0_WINDOW, K0_BATCH = 120_000, 2048, 8
TOL_FWD = 1e-4          # nats; fp32 identity pairs (registered a priori: no fp32 forward of this model is banked)
TOL_GRAD = 1e-6         # relative Frobenius of J between two passes of one prompt (lesson 2; expected 0)
MASSIVE_K, MASSIVE_X = 3, 10.0                # top-3 coordinates; "formed" at 10 x the step-0 ratio
EXEMPLARS = [(3, 5, 4, 2), (12, 7, 3, 9), (25, 16, 2, 8)]
LINE_TOKENS, QUERY_TOKENS = 16, 14
Q0 = 1 + LINE_TOKENS * len(EXEMPLARS)         # first query position (BOS at 0)
QPOS = {"a": Q0 + 1, "b": Q0 + 4, "c": Q0 + 8, "d": Q0 + 11, "last": Q0 + 13}
IC_POS = {1: [QPOS["a"], QPOS["b"]], 2: [QPOS["a"], QPOS["b"], QPOS["c"]]}
N_TOKENS = Q0 + QUERY_TOKENS                  # 63
ARMS = ["LENS_U", "LENS_M", "DOM_M", "ANS_U", "ANS_M", "RAND_U", "RAND_M", "RAND_DOM"]
SCORED = ["f1", "f2", "v", "y", "s1", "s2", "f1m", "f1p", "f2m", "f2p", "f1x", "f2x"]


def sha256_file(p) -> str:
    return m2.sha256_file(p)


def utc():
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------------------------------------ items
def line(a, b, c, d, answer=True) -> str:
    q = f"({a} + {b}) * {c} + {d} = "
    return q + f"{(a + b) * c + d}\n" if answer else q


def exemplar_text() -> str:
    return "".join(line(*e) for e in EXEMPLARS)


def exemplar_numbers() -> set:
    return {x for a, b, c, d in EXEMPLARS for x in (a, b, c, d, (a + b) * c + d)}


def factorizations(x: int) -> list:
    """(s1, c) with s1 * c = x, c in 2..9, s1 >= 2."""
    return [(x // c, c) for c in range(2, 10) if x % c == 0 and x // c >= 2]


def targets(it: dict, v: int, part: dict | None = None) -> dict:
    c, d = (it["c"], it["d"]) if part is None else (part["c"], part["d"])
    return {"f1": v * c + d, "f2": v + d}


def make_pool(n: int = POOL_N, seed: int = SEED_POOL) -> list:
    """The registered pool (SPEC §4.1). Deterministic; halves by index parity."""
    rng = np.random.default_rng(seed)
    ex, seen, items = exemplar_numbers(), set(EXEMPLARS), []
    while len(items) < n:
        s1 = int(rng.integers(11, 61))
        a = int(rng.integers(1, s1))
        b, c, d = s1 - a, int(rng.integers(2, 10)), int(rng.integers(1, 100))
        s2 = s1 * c
        y = s2 + d
        nums = {a, b, c, d, s1, s2, y}
        if len(nums) < 7 or (a, b, c, d) in seen or {s1, s2, y} & ex:
            continue
        cand = []
        for x in range(11, 100):
            f1, f2 = x * c + d, x + d
            if x in nums or x in ex or f1 > 999 or {f1, f2} & (nums | ex | {x}) or not factorizations(x):
                continue
            cand.append(x)
        if len(cand) < N_V:
            continue
        vs = [int(x) for x in rng.choice(cand, size=N_V, replace=False)]
        ic = []
        for v in vs[:N_V_IC]:
            a1 = int(rng.integers(1, v))
            s1b, c2 = factorizations(v)[int(rng.integers(0, len(factorizations(v))))]
            a2 = int(rng.integers(1, s1b))
            ic.append({"v": v, "slot1": [a1, v - a1, c, d], "slot2": [a2, s1b - a2, c2, d]})
        seen.add((a, b, c, d))
        items.append({"i": len(items), "half": "freeze" if len(items) % 2 == 0 else "test", "a": a, "b": b, "c": c,
                      "d": d, "s1": s1, "s2": s2, "y": y, "v": vs, "ic": ic})
    for h in ("freeze", "test"):
        idx = [it["i"] for it in items if it["half"] == h]
        for j, i in enumerate(idx):
            it = items[i]
            nums = {it[k] for k in ("a", "b", "c", "d", "s1", "s2", "y")}
            for step in range(1, len(idx)):
                p = items[idx[(j + step) % len(idx)]]
                ok = (p["c"], p["d"]) != (it["c"], it["d"]) and not ({p["s1"], p["s2"], p["y"]} & nums)
                for v in it["v"]:
                    t, tx = targets(it, v), targets(it, v, p)
                    ok = ok and tx["f1"] <= 999 and tx["f1"] != t["f1"] and tx["f2"] != t["f2"]
                if ok:
                    it["partner"] = p["i"]
                    break
            assert "partner" in it, f"item {i}: no partner"
    return items


def class_prompts(slot: int, x: int, n: int = N_CLASS, seed: int = SEED_CLASS) -> list:
    """The mean-difference classes (SPEC §2.4, the paper's §A.24.1): n expressions whose slot-`slot` intermediate is x,
    every other operand drawn at random; (a, b, c, d) with every number a single token."""
    rng = np.random.default_rng([seed, slot, x])
    out = []
    while len(out) < n:
        if slot == 1:
            a = int(rng.integers(1, x))
            cs = [c for c in range(2, 10) if x * c + 99 <= 999]
            c, d = cs[int(rng.integers(0, len(cs)))], int(rng.integers(1, 100))
            out.append((a, x - a, c, d))
        else:
            f = factorizations(x)
            s1, c = f[int(rng.integers(0, len(f)))]
            a, d = int(rng.integers(1, s1)), int(rng.integers(1, 100))
            assert x + d <= 999
            out.append((a, s1 - a, c, d))
    return out


def class_values(items: list) -> dict:
    """Every value whose class mean an edit needs: each item's own s_k and its v values, per slot."""
    return {1: sorted({it["s1"] for it in items} | {v for it in items for v in it["v"]}),
            2: sorted({it["s2"] for it in items} | {v for it in items for v in it["v"]})}


def prompt_text(abcd) -> str:
    return exemplar_text() + line(*abcd, answer=False)


def num_id(vocab: dict, x: int) -> int:
    assert 0 <= x <= 999, x
    return vocab[str(x)]


_VOCAB = {}


def vocab_of(tok) -> dict:
    if id(tok) not in _VOCAB:
        _VOCAB[id(tok)] = tok.get_vocab()
    return _VOCAB[id(tok)]


def prompt_ids(tok, abcd, bos: int = BOS) -> list:
    """BOS + the tokens of the prompt text; asserts the registered structure (63 tokens, operands where QPOS says)."""
    ids = [bos] + tok.encode(prompt_text(abcd), add_special_tokens=False)
    v = vocab_of(tok)
    assert len(ids) == N_TOKENS, f"{abcd}: {len(ids)} tokens"
    for k, x in zip("abcd", abcd):
        assert ids[QPOS[k]] == v[str(x)], (abcd, k)
    assert ids[QPOS["last"]] == v["Ġ"] and ids[QPOS["last"] - 1] == v["Ġ="], abcd
    return ids


def items_record(tok, items: list) -> dict:
    v = vocab_of(tok)
    num = [v[str(x)] for x in range(1000)]
    for it in items:
        it["ids"] = prompt_ids(tok, (it["a"], it["b"], it["c"], it["d"]))
        for ic in it["ic"]:
            ic["ids1"] = prompt_ids(tok, ic["slot1"])
            ic["ids2"] = prompt_ids(tok, ic["slot2"])
    cv = class_values(items)
    cls = {str(k): {str(x): [{"abcd": list(q), "ids": prompt_ids(tok, q)} for q in class_prompts(k, x)] for x in xs}
           for k, xs in cv.items()}
    from infra.pg.matrix_models import vocab_sha256
    return {"what": "N22 pool (SPEC §4.1): (a + b) * c + d, single-token numbers, halves by index parity",
            "seed": SEED_POOL, "exemplars": EXEMPLARS, "exemplar_text": exemplar_text(), "qpos": QPOS,
            "n_tokens": N_TOKENS, "number_ids": num, "vocab_sha256": vocab_sha256(tok),
            "class_values": {str(k): x for k, x in cv.items()}, "class_prompts": cls, "items": items}


# ------------------------------------------------------------------------------------------------ statistics
def boot_mean(x, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    x = np.asarray(x, np.float64)
    assert x.size > 0, "nothing to summarize"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, (n_boot, x.size))
    m = x[idx].mean(1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return {"n": int(x.size), "mean": float(x.mean()), "lo": float(lo), "hi": float(hi)}


def wilson(h: int, n: int) -> tuple:
    return m2.wilson(int(h), int(n))


def interaction(lp1: dict, lp2: dict, key1: str = "f1", key2: str = "f2") -> float:
    """lp1, lp2: {token key: log-prob} under the slot-1 and slot-2 edit of one (item, v)."""
    return (lp1[key1] - lp2[key1]) + (lp2[key2] - lp1[key2])


def item_interactions(rows: list, arm: str, shuffled: bool = False) -> dict:
    """rows: edit records {item, vi, arm, slot, lp}; returns {item: mean over its v of I} for `arm`."""
    by = {}
    for r in rows:
        if r["arm"] == arm:
            by.setdefault((r["item"], r["vi"]), {})[r["slot"]] = r["lp"]
    out = {}
    for (i, _), d in by.items():
        assert set(d) == {1, 2}, f"item {i}: arm {arm} lacks a slot"
        k1, k2 = ("f1x", "f2x") if shuffled else ("f1", "f2")
        out.setdefault(i, []).append(interaction(d[1], d[2], k1, k2))
    return {i: float(np.mean(v)) for i, v in out.items()}


def paired_vs(x: dict, y: dict) -> dict:
    ids = sorted(set(x) & set(y))
    assert ids, "no paired items"
    return boot_mean([x[i] - y[i] for i in ids])


def label3(d_rand: dict, d_shuf: dict) -> str:
    """ABOVE_FLOORS: both paired lower bounds > 0; BELOW_FLOOR: either upper bound < 0; AT_FLOOR otherwise."""
    if d_rand["lo"] > 0 and d_shuf["lo"] > 0:
        return "ABOVE_FLOORS"
    if d_rand["hi"] < 0 or d_shuf["hi"] < 0:
        return "BELOW_FLOOR"
    return "AT_FLOOR"


def slot_label(rows: list, arm: str, rand_arm: str, ceiling: dict, n_min: int = N_MIN) -> dict:
    """The label of one edit arm on one half (SPEC §6.2)."""
    real = item_interactions(rows, arm)
    if len(real) < n_min:
        return {"label": "DOES NOT COUNT: fewer than N_MIN items", "n": len(real)}
    rnd, shf = item_interactions(rows, rand_arm), item_interactions(rows, arm, shuffled=True)
    d_r, d_s = paired_vs(real, rnd), paired_vs(real, shf)
    lab = label3(d_r, d_s)
    m = boot_mean(list(real.values()))
    out = {"n": len(real), "I": m, "I_rand": boot_mean(list(rnd.values())), "I_shuf": boot_mean(list(shf.values())),
           "minus_rand": d_r, "minus_shuf": d_s, "label": lab}
    if m["lo"] > ceiling["hi"]:
        out["label"] = "DOES NOT COUNT: above the ceiling (inspect)"
    out["ceiling"] = ceiling
    return out


def band_rule(r1: np.ndarray, r2: np.ndarray) -> dict:
    """r1, r2 [layers]: net readout shares (hit minus shuffled hit) of s1 and s2 on the freeze half. Ties: lower layer."""
    l1, l2 = int(np.argmax(r1)), int(np.argmax(r2))
    mn = np.minimum(r1, r2)
    lm = int(np.argmax(mn))
    band = [x for x in (lm - 1, lm, lm + 1) if 0 <= x < len(r1)]
    return {"l1": l1, "l2": l2, "lm": lm, "band_m": band, "matched": float(mn[lm])}


def readout_present(hit: np.ndarray, shuf: np.ndarray, delta: float) -> dict:
    """hit, shuf: 0/1 per item at one layer. PRESENT if the net share exceeds delta and the Wilson intervals separate."""
    n = int(hit.size)
    assert n > 0
    p, lo, _ = wilson(hit.sum(), n)
    q, _, hi = wilson(shuf.sum(), n)
    ok = (p - q) > delta and lo > hi
    return {"n": n, "share": p, "shuffled": q, "net": p - q, "delta": delta, "label": "PRESENT" if ok else "NOT_PRESENT"}


def ranks_of(logits, ids):
    """1 + the number of vocabulary entries with a strictly larger logit; logits [..., V] (torch or numpy)."""
    import torch
    lg = torch.as_tensor(logits)
    sel = lg.gather(-1, torch.as_tensor(ids, device=lg.device).long())
    return 1 + (lg.unsqueeze(-2) > sel.unsqueeze(-1)).sum(-1)


def linear_cka(gx, gy) -> float:
    """Linear CKA from two Gram matrices over the same rows (numpy or torch, float64 recommended)."""
    import torch
    gx, gy = torch.as_tensor(gx, dtype=torch.float64), torch.as_tensor(gy, dtype=torch.float64)
    n = gx.shape[0]
    h = torch.eye(n, dtype=torch.float64, device=gx.device) - 1.0 / n
    a, b = h @ gx @ h, h @ gy @ h
    return float((a * b).sum() / (a.norm() * b.norm()))


def cka_block(cka: np.ndarray) -> float:
    """Block-diagonal structure: mean CKA of layer pairs 1..L/10 apart minus mean of pairs >= L/2 apart (L = 30:
    1-3 apart against >= 15 apart)."""
    L = cka.shape[0]
    near, far = max(1, L // 10), L // 2
    d = np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    return float(cka[(d >= 1) & (d <= near)].mean() - cka[d >= far].mean())


def excess_kurtosis(x):
    import torch
    x = torch.as_tensor(x, dtype=torch.float64)
    m = x.mean(-1, keepdim=True)
    v = ((x - m) ** 2).mean(-1)
    return ((x - m) ** 4).mean(-1) / v ** 2 - 3.0


def lens_swap_delta(h, V, sigma: bool = True):
    """The paper's swap (§2.5): c = V+ h, h' = h + V (sigma(c) - c). h [B, P, d], V [B, d, 2]. Returns h' - h."""
    import torch
    G = V.transpose(1, 2) @ V                                  # [B, 2, 2]
    c = torch.linalg.solve(G, V.transpose(1, 2) @ h.transpose(1, 2))   # [B, 2, P]
    sc = c.flip(1) if sigma else c
    return (V @ (sc - c)).transpose(1, 2)                      # [B, P, d]


# ------------------------------------------------------------------------------------------------ N25 labels
def n25_labels(stats: dict) -> dict:
    """stats {t: {stat: {"A", "B", "full", "proj"}}}. H_arch per statistic (T_0 against T_15 within split-half noise);
    H_sink per statistic and point; H_form: the intervals between consecutive lens points whose change exceeds the noise."""
    pts = sorted(stats)
    out = {"arch": {}, "sink": {}, "form": {}}
    names = sorted(stats[pts[-1]])
    noise = {t: {s: abs(stats[t][s]["A"] - stats[t][s]["B"]) for s in names} for t in pts}
    for s in names:
        if 0 in stats and 15 in stats:
            dd = abs(stats[0][s]["full"] - stats[15][s]["full"])
            room = noise[0][s] + noise[15][s]
            out["arch"][s] = {"diff": dd, "noise": room, "label": "ARCHITECTURAL" if dd <= room else "LEARNED"}
        out["sink"][s] = {str(t): {"diff": abs(stats[t][s]["proj"] - stats[t][s]["full"]), "noise": noise[t][s],
                                   "label": "SINK_DEPENDENT" if abs(stats[t][s]["proj"] - stats[t][s]["full"]) >
                                   noise[t][s] else "SURVIVES"} for t in pts}
        steps = [(t1, t2, abs(stats[t2][s]["full"] - stats[t1][s]["full"]), noise[t1][s] + noise[t2][s])
                 for t1, t2 in zip(pts, pts[1:])]
        jumps = [(t1, t2) for t1, t2, dd, nz in steps if dd > nz]
        total = abs(stats[pts[-1]][s]["full"] - stats[pts[0]][s]["full"])
        out["form"][s] = {"steps": [{"from": t1, "to": t2, "diff": dd, "noise": nz} for t1, t2, dd, nz in steps],
                          "first_interval": list(jumps[0]) if jumps else None,
                          "label": ("FORMS_BETWEEN" if jumps else
                                    ("SMOOTH" if total > noise[pts[0]][s] + noise[pts[-1]][s] else "NO_CHANGE"))}
    return out


def massive_formed(ratio: dict) -> dict:
    """ratio {t: max over layers of max|h_i| / median|h|}; formed at the first t with ratio >= MASSIVE_X x T_0's."""
    base = ratio[0]
    first = next((t for t in sorted(ratio) if ratio[t] >= MASSIVE_X * base), None)
    return {"t0_ratio": base, "rule": f">= {MASSIVE_X} x the step-0 ratio", "first": first}


# ------------------------------------------------------------------------------------------------ verdict
def read_jsonl(p) -> list:
    return m2.read_jsonl(p)


def gates_ok(metrics: dict) -> list:
    bad = [n for n, g in metrics["gates"].items() if not g["pass"]]
    bad += [n for n, g in metrics["gates"].items() if "failing_input" in g and g["failing_input"]["pass"]]
    return bad


def plan_lens(acc: dict, max_points: int = 4) -> dict:
    """L3's lens points from L2's accuracy scan (SPEC §9.5). acc {t: {"freeze": n, "test": n}} for t = 0..14.
    C = points with >= N_MIN answered in both halves; if more than four, the earliest, the latest and the two nearest
    the thirds of the index range between them. N25 adds T_0 and the point just before the earliest of C."""
    C = sorted(t for t in range(1, 15) if acc[t]["freeze"] >= N_MIN and acc[t]["test"] >= N_MIN)
    if len(C) > max_points:
        lo, hi = C[0], C[-1]
        mids = []
        for f in (1 / 3, 2 / 3):
            want = lo + f * (hi - lo)
            mids.append(min((t for t in C if t not in (lo, hi) and t not in mids), key=lambda t: (abs(t - want), t)))
        C = sorted({lo, hi, *mids})
    early = (min(C) - 1) if C and min(C) - 1 > 0 else None
    order = [0] + sorted(C, reverse=True) + ([early] if early is not None and early not in C else [])
    return {"n22_points": C, "n25_points": [0] + ([early] if early is not None else []), "order": order,
            "rule": "SPEC §9.5; park from the end of `order`"}


def per_item_pc(rows: list) -> dict:
    """The positive control of the lens-swap instrument: per item, mean over (v, slot) of
    lp(f_k | ANS_U_k) - lp(f_k | RAND_U_k)."""
    a, r = {}, {}
    for x in rows:
        if x["arm"] in ("ANS_U", "RAND_U"):
            (a if x["arm"] == "ANS_U" else r)[(x["item"], x["vi"], x["slot"])] = x["lp"][f"f{x['slot']}"]
    out = {}
    for k in a:
        out.setdefault(k[0], []).append(a[k] - r[k])
    return {i: float(np.mean(v)) for i, v in out.items()}


def ic_items(rows: list, site: int, kind: str, shuffled: bool = False) -> dict:
    by = {}
    for r in rows:
        if r["site"] == site and r["kind"] == kind:
            by.setdefault((r["item"], r["vi"]), {})[r["slot"]] = r["lp"]
    out = {}
    for (i, _), d in by.items():
        assert set(d) == {1, 2}
        out.setdefault(i, []).append(interaction(d[1], d[2], *(("f1x", "f2x") if shuffled else ("f1", "f2"))))
    return {i: float(np.mean(v)) for i, v in out.items()}


def ic_go(rows: list) -> dict:
    """The interchange control at the embedding (SPEC §5.1): fires if I_IC exceeds both floors (paired lower bounds > 0)."""
    real, rnd, shf = ic_items(rows, -1, "IC"), ic_items(rows, -1, "RAND_IC"), ic_items(rows, -1, "IC", True)
    if len(real) < N_MIN:
        return {"n": len(real), "fires": False, "note": "fewer than N_MIN items"}
    d_r, d_s = paired_vs(real, rnd), paired_vs(real, shf)
    return {"n": len(real), "I_IC_emb": boot_mean(list(real.values())), "minus_rand": d_r, "minus_shuf": d_s,
            "fires": bool(d_r["lo"] > 0 and d_s["lo"] > 0)}


def ic_profile(rows: list) -> dict:
    sites = sorted({r["site"] for r in rows})
    out = {}
    for s in sites:
        real = ic_items(rows, s, "IC")
        rnd = ic_items(rows, s, "RAND_IC")
        if real and rnd:
            out[str(s)] = {"I": boot_mean(list(real.values())), "minus_rand": paired_vs(real, rnd)}
    return out


def go_nogo(n_answered_total: int, ic: dict) -> dict:
    return {"n_answered": n_answered_total, "ic_fires": ic["fires"], "go": bool(n_answered_total >= 100 and ic["fires"]),
            "rule": "answered >= 100 over the pool and the interchange control above both floors (freeze half)"}


def load_point(d) -> dict:
    d = pathlib.Path(d)
    return {"design": json.loads((d / "design.json").read_text()),
            "accuracy": json.loads((d / "accuracy.json").read_text()),
            "readout": np.load(d / "readout.npz"), "dir": d}


def freeze(banks, out, points=(15,)) -> dict:
    """Freeze 2 (SPEC §6.1): from the freeze halves only. Writes delta_read, the bands, the matched-layer decision, the
    DoM guard, the positive control and (T_15) the go/no-go; the sha256 of every test-half file."""
    banks = pathlib.Path(banks)
    m = json.loads((banks / "n22" / "metrics.json").read_text())
    bad = gates_ok(m)
    assert not bad, f"gates failed: {bad}"
    thr = {"what": "N22 freeze 2 (freeze halves only)", "points": {}, "metrics_sha256": sha256_file(banks / "n22" / "metrics.json")}
    for t in points:
        d = banks / "n22" / f"t{t:02d}"
        P = load_point(d)
        de = P["design"]
        rec = {"t": t, "bands": de.get("bands"), "delta_read": de.get("delta_read"), "n_answered": de["n_answered"],
               "n_swap": de.get("n_swap", {})}
        if "bands" in de:
            rec["matched"] = bool(de["bands"]["matched"] > de["delta_read"])
            rec["dom_guard_answer_readable"] = bool(de["answer_net_at_lm"] > de["delta_read"])
        if (d / "edits_freeze.jsonl").exists():
            fe = read_jsonl(d / "edits_freeze.jsonl")
            pc = per_item_pc(fe)
            rec["pc_ans"] = boot_mean(list(pc.values()))
            rec["pc_ans_fires"] = bool(rec["pc_ans"]["lo"] > 0)
            disc = {}
            for arm in ("LENS_M", "LENS_U", "DOM_M", "RAND_M", "RAND_U", "RAND_DOM", "ANS_M", "ANS_U"):
                x = list(item_interactions(fe, arm).values())
                disc[arm] = boot_mean(x) | {"sd_items": float(np.std(x, ddof=1))}
            rec["DISCOVERY_freeze_interactions"] = disc     # the freeze half's values: discovery, never a label
            ic = ic_go(read_jsonl(d / "ic_freeze.jsonl"))
            rec["ic"] = ic
            rec["test_sha256"] = {f: sha256_file(d / f) for f in ("edits_test.jsonl", "ic_test.jsonl")}
            if t == 15:
                rec["go_nogo"] = go_nogo(sum(de["n_answered"].values()), ic)
        elif t == 15:
            rec["go_nogo"] = {"go": False, "n_answered": sum(de["n_answered"].values()),
                              "rule": "no edits ran (fewer than N_MIN answered in a half)"}
        thr["points"][str(t)] = rec
    pathlib.Path(out).write_text(json.dumps(thr, indent=1))
    return thr


def edit_reports(rows: list, num_ids: list, items: dict) -> dict:
    """Reported, no label: the top-1 double flip (argmax X1 = f1 and argmax X2 = f2), the share of slot edits whose
    top-1 is v, and the exact-target margin (the target above f +- 1 and v)."""
    by = {}
    for r in rows:
        by.setdefault((r["arm"], r["item"], r["vi"]), {})[r["slot"]] = r
    out = {}
    for arm in sorted({k[0] for k in by}):
        ks = [k for k in by if k[0] == arm]
        dbl, inj, mar, n = 0, 0, 0, 0
        for k in ks:
            d = by[k]
            it = items[k[1]]
            v = it["v"][k[2]]
            f = {1: v * it["c"] + it["d"], 2: v + it["d"]}
            dbl += int(d[1]["top1"] == num_ids[f[1]] and d[2]["top1"] == num_ids[f[2]])
            for s in (1, 2):
                n += 1
                inj += int(d[s]["top1"] == num_ids[v])
                lp = d[s]["lp"]
                mar += int(lp[f"f{s}"] > max(lp[f"f{s}m"], lp[f"f{s}p"], lp["v"]))
        out[arm] = {"n_pairs": len(ks), "double_flip": wilson(dbl, len(ks)), "top1_is_v": wilson(inj, n),
                    "exact_target_margin": wilson(mar, n)}
    return out


def point_verdict(d, thr: dict, items_rec: dict) -> dict:
    """The test-half labels at one point (SPEC §6.2), under the frozen point record `thr`."""
    d = pathlib.Path(d)
    if (d / "edits_test.jsonl").exists():
        for f, h in thr.get("test_sha256", {}).items():
            assert sha256_file(d / f) == h, f"{f} changed after freeze"
    P = load_point(d)
    de, acc, rd = P["design"], P["accuracy"], P["readout"]
    items = {it["i"]: it for it in items_rec["items"]}
    ranks, variants, names = rd["ranks"], list(rd["variants"]), list(rd["targets"])
    out = {"t": thr["t"], "n_answered": de["n_answered"]}
    te_ix = np.array([i for i, it in items.items() if it["half"] == "test" and acc["correct"][i]])
    if de["kind"] == "n25":
        te_ix = np.array([i for i, it in items.items() if it["half"] == "test"])
    if te_ix.size == 0 or "bands" not in de:
        out["labels"] = "NOT_RUN: no item"
        return out
    b, delta = de["bands"], thr["delta_read"]
    vf, vp = variants.index("full"), variants.index("proj")
    read = {}
    for k in ("s1", "s2"):
        lyr = b["l1" if k == "s1" else "l2"]
        hit = ranks[te_ix, lyr, vf, names.index(k)] <= K_READ
        sh = ranks[te_ix, lyr, vf, names.index(k + "x")] <= K_READ
        r = readout_present(hit, sh, delta)
        hp = ranks[te_ix, lyr, vp, names.index(k)] <= K_READ
        shp = ranks[te_ix, lyr, vp, names.index(k + "x")] <= K_READ
        r["net_proj"] = float(hp.mean() - shp.mean())
        r["sink"] = "READOUT_SINK" if abs(r["net_proj"] - r["net"]) > delta else "READOUT_SURVIVES"
        r["layer"] = lyr
        read[k] = r
    out["readout"] = read
    out["readout_both"] = "PRESENT" if all(read[k]["label"] == "PRESENT" for k in read) else "NOT_PRESENT"
    if not (d / "edits_test.jsonl").exists():
        out["slot"] = "NOT_RUN: no edits at this point"
        return out
    rows = read_jsonl(d / "edits_test.jsonl")
    ic = read_jsonl(d / "ic_test.jsonl")
    ceil_items = ic_items(ic, -1, "IC")
    ceiling = boot_mean(list(ceil_items.values()))
    lab = {}
    for arm, rnd in (("LENS_M", "RAND_M"), ("LENS_U", "RAND_U"), ("DOM_M", "RAND_DOM")):
        lab[arm] = slot_label(rows, arm, rnd, ceiling)
    pc = boot_mean(list(per_item_pc(rows).values()))
    pc_ok = bool(pc["lo"] > 0 and thr.get("pc_ans_fires", False))
    out["pc_ans_test"], out["pc_ans_fires_both"] = pc, pc_ok
    for arm in ("LENS_M", "LENS_U"):
        if not pc_ok:
            lab[arm]["label"] = "DOES NOT COUNT: the lens-swap positive control did not fire"
    for arm in ("LENS_M", "DOM_M"):
        if not thr.get("matched", False):
            lab[arm]["label"] = "DOES NOT COUNT: no matched layer"
    if thr.get("dom_guard_answer_readable", False):
        lab["DOM_M"]["label"] = "DOES NOT COUNT: the answer is readable at the matched layer"
    lab["LENS_M"]["minus_ans_m"] = paired_vs(item_interactions(rows, "LENS_M"), item_interactions(rows, "ANS_M"))
    lab["LENS_M"]["above_answer_substitution"] = bool(lab["LENS_M"]["minus_ans_m"]["lo"] > 0)
    out["slot"] = lab
    out["H_slot"] = {"ABOVE_FLOORS": "SLOT_DEPENDENT", "AT_FLOOR": "NOT_SHOWN", "BELOW_FLOOR": "BELOW_FLOOR"}.get(
        lab["LENS_M"]["label"], lab["LENS_M"]["label"])
    out["reports"] = edit_reports(rows, items_rec["number_ids"], items)
    out["depth_reading"] = ("DEPTH_NOT_EXCLUDED" if lab["LENS_U"]["label"] == "ABOVE_FLOORS" and
                            lab["LENS_M"]["label"] in ("AT_FLOOR", "BELOW_FLOOR") else None)
    return out


def dating_arm(t15: dict) -> str | None:
    """The arm that dates slot dependence (SPEC §6.3): LENS_M if SLOT_DEPENDENT at T_15, else LENS_U, else DOM_M."""
    for arm in ("LENS_M", "LENS_U", "DOM_M"):
        if t15.get("slot", {}).get(arm, {}).get("label") == "ABOVE_FLOORS":
            return arm
    return None


def order_label(per_point: dict, arm: str | None) -> dict:
    """H_order over the scored lens points in time order (SPEC §6.3)."""
    pts = sorted(int(t) for t, v in per_point.items() if isinstance(v.get("slot"), dict))
    if arm is None or not pts:
        return {"label": "NOT_RUN", "points": pts}
    rd = {t: per_point[str(t)]["readout_both"] == "PRESENT" for t in pts}
    sl = {t: per_point[str(t)]["slot"][arm]["label"] == "ABOVE_FLOORS" for t in pts}
    tr = next((t for t in pts if rd[t]), None)
    ts = next((t for t in pts if sl[t]), None)
    if ts is None:
        lab = "NOT_RUN"
    elif tr is None or ts < tr:
        lab = "REFUTED_CONTRARY"
    elif tr < ts:
        lab = "SUPPORTED"
    else:
        lab = "CENSORED" if tr == pts[0] else "REFUTED_SAME"
    return {"label": lab, "arm": arm, "points": pts, "readout_first": tr, "slot_first": ts, "readout": rd, "slot": sl}


def project(n_answered: int = 400, n_class_prompts: int = 5056, rates: dict | None = None) -> dict:
    """GPU-hours per launch from the measured or registered rates (SPEC §9.6)."""
    R = rates or {"center": {"dl_s": 82, "load_s": 60, "k0_s": 15, "lens_prompt_s": 78, "fwd_s": 0.018, "boot_s": 224},
                  "high": {"dl_s": 192, "load_s": 60, "k0_s": 15, "lens_prompt_s": 120, "fwd_s": 0.04, "boot_s": 224}}
    out = {}
    for k, r in R.items():
        base = r["dl_s"] + r["load_s"] + r["k0_s"] + 30
        lens = (N_LENS_PROMPTS + 1) * r["lens_prompt_s"]
        fz = n_answered // 2
        edits = n_answered * N_V * 2 * len(ARMS) + fz * 2 * len(DOM_SWEEP) + \
            fz * N_V_IC * 2 * (1 + 2 * len(IC_SITES)) + (n_answered - fz) * N_V_IC * 2 * 3
        full = base + lens + (2 * POOL_N + n_class_prompts + edits) * r["fwd_s"] + 300
        acc_pt = base + POOL_N * r["fwd_s"] + 20
        n25_pt = base + lens + (POOL_N + n_class_prompts) * r["fwd_s"] + 200
        l1 = r["boot_s"] + full
        l2 = r["boot_s"] + 15 * acc_pt
        l3 = r["boot_s"] + 4 * full + 2 * n25_pt
        out[k] = {"per_point_full_s": round(full), "per_point_accuracy_s": round(acc_pt), "per_point_n25_s": round(n25_pt),
                  "forwards_full_point": int(2 * POOL_N + n_class_prompts + edits),
                  "L1_h": round(l1 / 3600, 2), "L2_h": round(l2 / 3600, 2), "L3_h": round(l3 / 3600, 2),
                  "total_h": round((l1 + l2 + l3) / 3600, 2)}
    for L in ("L1", "L2", "L3"):
        out[f"{L}_kill_min"] = int(np.ceil(1.3 * out["high"][f"{L}_h"] * 60))
    out["usd_per_h"] = 2.24
    out["usd"] = {k: round(v["total_h"] * 2.24, 2) for k, v in out.items() if isinstance(v, dict) and "total_h" in v}
    out["assumptions"] = {"n_answered_per_full_point": n_answered, "class_prompts": n_class_prompts, "rates": R,
                          "sources": "download 76 MB/s (M1 K0) to 178-321 MB/s (M2 score-06); fixed and boot from N12 "
                                     "score-02; lens per prompt from N12's measured 0.5832 s per 2,049-token gradient "
                                     "window scaled by FLOPs (an estimate); fp32 forward of 63 tokens an estimate"}
    return out


def cli():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("items")
    p.add_argument("--tok", required=True)
    p.add_argument("--out", required=True)
    sub.add_parser("project")
    p = sub.add_parser("plan-lens")
    p.add_argument("--metrics", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--banks", required=True)
    p.add_argument("--points", default="15")
    p.add_argument("--out", required=True)
    p = sub.add_parser("verdict")
    p.add_argument("--banks", required=True)
    p.add_argument("--thresholds", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "items":
        import transformers
        tok = transformers.AutoTokenizer.from_pretrained(a.tok)
        rec = items_record(tok, make_pool())
        pathlib.Path(a.out).write_text(json.dumps(rec))
        print(hashlib.sha256(pathlib.Path(a.out).read_bytes()).hexdigest())
    elif a.cmd == "project":
        print(json.dumps(project(), indent=1))
    elif a.cmd == "plan-lens":
        m = json.loads(pathlib.Path(a.metrics).read_text())
        acc = {int(t): v["answered"] for t, v in m["points"].items()}
        print(json.dumps(plan_lens(acc), indent=1))
    elif a.cmd == "freeze":
        freeze(a.banks, a.out, [int(x) for x in a.points.split(",")])
        print("N22_FREEZE_OK")
    elif a.cmd == "verdict":
        thr = json.loads(pathlib.Path(a.thresholds).read_text())
        it = json.loads(pathlib.Path(a.items).read_text())
        res = {t: point_verdict(pathlib.Path(a.banks) / "n22" / f"t{int(t):02d}", r, it) for t, r in thr["points"].items()}
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
        print(json.dumps({t: r.get("H_slot", r.get("slot")) for t, r in res.items()}))


if __name__ == "__main__":
    cli()
