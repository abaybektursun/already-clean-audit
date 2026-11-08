"""Dated exploratory analysis (2026-11-05) on the closed M1 result. Run from the repo root:
python3 studies/next/m1/reading_control/target_g_m.py > studies/next/m1/reading_control/target_g_m_output.txt
Part 1: L04N3542- top contexts at start_a, G, M and its directory counts. Part 2: token families and position overlap."""
import json, collections, numpy as np
B = "studies/next/m1/banks"
def clip(s, n, left):
    s = s.replace("\n", "⏎")
    return ("…" + s[-n:] if len(s) > n else s) if left else (s[:n] + "…" if len(s) > n else s)
for k in ["start_a", "G", "M"]:
    r = [json.loads(l) for l in open(f"{B}/{k}/samples/contexts.jsonl") if l.startswith('{"pole": "L04N3542-"')][0]
    print(f"\n===== {k}: max_abs {r['max_abs']:.4f}")
    toks = collections.Counter(t["token"] for t in r["topk"])
    print("top-50 token counts:", toks.most_common(12))
    for t in r["topk"][:20]:
        print(f"{t['rank']:2d} {t['a']:+.3f} doc {t['doc']:6d} | {clip(t['before'], 80, True)}[[{t['token']}]]{clip(t['after'], 40, False)}")
    z = np.load(f"{B}/{k}/m1/counts.npz")
    for s in (0, 1):
        c = z["counts"][4, 3542, s]; m = z["cm_mask"]
        nz = {str(d).replace("data/", ""): int(v) for d, v in zip(z["dirs"], c) if v}
        print(f"  counts.npz sign index {s}: cm_share {c[m].sum() / z['filled'][4, 3542, s]:.2f}", dict(sorted(nz.items(), key=lambda x: -x[1])))

# ---- part 2
R = {k: [json.loads(l) for l in open(f"{B}/{k}/samples/contexts.jsonl") if l.startswith('{"pole": "L04N3542-"')][0] for k in ["start_a", "G", "M"]}
div = {"erg", "-div", "gent", " conver", " divergence", " diver", "iver", "ver", "Diver", " Diver", "gence", " converge", "erging"}  # word-pieces of diverge/converge; ".div" (a division function) is not one
prod = lambda t: any(w in t.lower() for w in ("prod", "cross", "dot", "ultiply")) and "dotted" not in t  # " dotted" is a music term
for k, r in R.items():
    toks = [t["token"] for t in r["topk"]]
    nd = sum(t in div for t in toks); npd = sum(prod(t) for t in toks)
    print(f"{k:8s} max_abs {r['max_abs']:.3f}  50th |a| {abs(r['topk'][-1]['a']):.3f}  div-family {nd}/50  product-family {npd}/50  other {50 - nd - npd}: {[t for t in toks if t not in div and not prod(t)]}")
    print("   all tokens:", collections.Counter(toks).most_common())
s = {t["tok"]: t for t in R["start_a"]["topk"]}
for k in ["G", "M"]:
    pos = {t["tok"]: t["rank"] for t in R[k]["topk"]}
    kept = [(s[p]["rank"], s[p]["token"], pos[p]) for p in s if p in pos]
    print(f"\nstart_a top-50 positions still in {k}'s top-50: {len(kept)}/50 ; (start rank, token, {k} rank):", sorted(kept)[:25])
    print(f"  start_a top-10 positions in {k}'s top-50:", [(s[p]['rank'], s[p]['token'], pos.get(p)) for p in s if s[p]['rank'] < 10])

# ---- part 3: directory groups (web, pdf, code, math) of the six blind poles' top-50 at start_a (sign index 0 = +, 1 = -)
z = np.load(f"{B}/start_a/m1/counts.npz")
for label, pole in json.load(open("studies/next/m1/reading_control/map.json"))["map"].items():
    n, s = int(pole[4:-1]), 0 if pole.endswith("+") else 1
    c = z["counts"][4, n, s]
    g = {grp: int(c[z["groups"] == grp].sum()) for grp in ("web", "pdf", "code", "math")}
    print(f"{label} {pole:11s} top-50 slots by group at start_a: {g}")
