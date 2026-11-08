"""Weight-free dry run of SPEC12 LOWLAYER stage 2 (studies/reuse/scripts/build_items_lowlayer.py and verdict_lowlayer.py), on CPU. Nothing is downloaded.

    python -m infra.aws_pg.jobs.reuse_dryrun_lowlayer2 --out <dir>

Bank: the capture job (infra/aws_pg/jobs/reuse_lowlayer.Job) run at all 16 loads on the stage-1 dry run's tiny OLMo 3 (reuse_dryrun_lowlayer:
tiny_olmo3, reference_windows, reference_instrument, top_tok), with 8 neurons per layer 0-3 (64 new poles) and the 8 known poles with one SPEC8 case; positions drawn among window positions 1..255.
The load-t model is (1 - w_t) theta_15 + w_t theta_seed_t with w_t = 0.1 + 0.9 ((15 - t) / 15)^4 (w_15 = 0), so the T_15 top positions stay
positive at late loads (the scale validity line) while adjacent loads still differ (the known numbers' failing inputs). A corpus of 12 windows of 512 tokens with 400 document bounds.
Plants, from the banked a_h and aP (after the job): per pole a birth t_b in 4..13 (cycled), O* = 6 ranks with a_h > 0 at t_b - 2 and t_b - 1 and
aP(t_b) > 0, J = 6 ranks with a_h > 0 at t_b and t_b + 1 and the smallest a_h before, J and O* with no shared token id and no shared document; the
pole's q999 per load: huge before t_b - 2 and after t_b + 1, between J's and O*'s medians at t_b - 2 and t_b - 1, half J's median at t_b and t_b + 1;
max_abs = q999 / 0.3 (the raw reading equals the percentile reading). The other ranks in clusters of 2. Special poles: one pole whose J shares a
document with O* (must be filtered), one pole with an echo cluster holding its three double-negative ranks (g < 0, u < 0), one pole declined in
sample 1. The T_15 dating source: the first pole's poles-file a at ranks 48 and 49 doubled before the bank, those two ranks one cluster, and that
pole's q999 at T_15 set to 1.5 times their captured median, so the cluster is born at T_15 on topk.a and never on the capture's a_h: the items
must date T_15 on the poles file and count exactly 1 differing birth. Re-asks: a clusterer directory with 3 rejected rows in the s0 file, no s1
file and reasks.json withheld 2, so the §7 rate must read 5 / 128; the same directory without reasks.json must stop the verdict.
Clusters: sample 1 equal to sample 0. Then the items builder (real dating known number on the 80 grid poles: it reads only the real
banks, so it runs here as on the real bank; no --skip-known is needed) and the verdict: exit 0 and every planted birth recovered; each registered
failing input exits 1. Prints REUSE_DRYRUN_LOWLAYER2_OK.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import torch

from infra.aws_pg.jobs import reuse_dryrun_lowlayer as D1
from infra.aws_pg.jobs import reuse_lowlayer as LL
from infra.aws_pg.jobs.neuron_off_dryrun import tiny_tokenizer

REPO = pathlib.Path(__file__).resolve().parents[3]
TS = D1.TS
HW, N_WIN, K, WIDTH = D1.HW, D1.N_WIN, D1.K, D1.WIDTH
NEURONS = range(10, 18)
HUGE = 1e9
MODES = ("known_prev_load", "known_next_load", "identity_layer_above", "completeness_minus_one", "routes_disagree", "clusterer_missing_sample",
         "dating_next_q999", "scale_other_load_sha", "validity_zeroed", "nullpower_dates_observed_shuffled", "cases_under_floor")


def mixed_models() -> dict:
    m15 = D1.tiny_olmo3(15)
    p15 = {k: v.float().clone() for k, v in m15.state_dict().items()}
    out = {}
    for t in TS:
        w = 0.0 if t == "T_15" else 0.1 + 0.9 * ((15 - int(t[2:])) / 15) ** 4
        m = D1.tiny_olmo3(int(t[2:]))
        m.load_state_dict({k: ((1 - w) * p15[k] + w * v.float()).to(v.dtype) for k, v in m.state_dict().items()})
        out[t] = m
    return out


def top_tok(ref15, L, n, s) -> np.ndarray:
    """reuse_dryrun_lowlayer.top_tok over window positions 1..255 only, where the 512-token harvest window and the 256-token instrument window see
    the same tokens, so s_P (the instrument's median) follows the harvest's top contexts as on the real bank's late loads."""
    sc = (1.0 if s == "+" else -1.0) * ref15[L, :, :, n].astype(np.float64)
    sc[:, 0] = -np.inf
    sc[:, 256:] = -np.inf
    flat = np.argsort(-sc.ravel(), kind="stable")[:K]
    return (flat // HW * HW + flat % HW).astype(np.int64)


def build(tok, tokens, offsets, models) -> dict:
    bos = tok.bos_token_id
    refs = {t: D1.reference_windows(models[t], tokens, bos) for t in TS}
    at = lambda ref, L, n, tk: ref[L, tk // HW, tk % HW, n].astype(np.float64)
    doc = lambda tk: (np.searchsorted(offsets, tk, side="right") - 1).astype(np.int64)
    poles = []
    for L in range(4):
        for n in NEURONS:
            for s in "+-":
                tk = top_tok(refs["T_15"], L, n, s)
                poles.append({"pole": f"L{L:02d}N{n}{s}", "layer": L, "neuron": n, "sign": s, "tok": tk.tolist(), "doc": doc(tk).tolist(),
                              "a": at(refs["T_15"], L, n, tk).tolist(), "g": [1.0] * K, "u": [1.0] * K})
    known = []
    for L in range(4):
        n = 70 + L
        for s in "+-":
            tk = top_tok(refs["T_15"], L, n, s)
            rec = {"pole": f"L{L:02d}N{n}{s}", "layer": L, "neuron": n, "sign": s, "tok": tk.tolist(), "doc": doc(tk).tolist(),
                   "a_bank": {t: at(refs[t], L, n, tk).tolist() for t in TS}, "spec8_cases": []}
            if L == 3 and s == "+":
                ctx = np.concatenate([tk[:7], tk[7:13]])
                aP = {t: D1.reference_instrument(models[t], tokens, bos, ctx, L, n, 1.0) for t in TS}
                rec["spec8_cases"].append({"c": 0, "side": ["J"] * 7 + ["O"] * 6, "tok": ctx.tolist(), "aP": {t: v.tolist() for t, v in aP.items()},
                                           "sha256": {t: hashlib.sha256(v.tobytes()).hexdigest() for t, v in aP.items()}})
            known.append(rec)
    return {"poles": poles, "known": known, "upto_layer": D1.UPTO, "grid": {t: f"t{int(t[2:]):02d}_step{int(t[2:])}" for t in TS},
            "corpus": {"tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest()}}


def choose(A, P, ids, docs, tb, skip, shared: bool):
    """J and O* for a birth at tb (see the module doc); None if the bank does not allow it."""
    pre = [tb - 2, tb - 1]
    oc = [r for r in np.argsort(-A[pre].min(0)) if r not in skip and A[pre, r].min() > 0 and P[tb, r] > 0]
    O = oc[:6]
    if len(O) < 6:
        return None
    jc = [r for r in np.argsort(A[pre].max(0)) if r not in skip and r not in O and A[tb, r] > 0 and A[tb + 1, r] > 0 and ids[r] not in set(ids[O])]
    dO = set(docs[O])
    J = ([r for r in jc if docs[r] in dO][:1] if shared else []) + [r for r in jc if docs[r] not in dO]
    J = J[:6]
    if len(J) < 6 or (shared and not set(docs[J]) & dO):
        return None
    mJ, mO = np.median(A[:, J], 1), np.median(A[:, O], 1)
    if not (all(mO[t] > max(mJ[t], 0) for t in pre) and mJ[tb] > 0 and mJ[tb + 1] > 0):
        return None
    q = np.full(16, HUGE)
    for t in pre:
        q[t] = (max(mJ[t], 0) + mO[t]) / 2
    for t in (tb, tb + 1):
        q[t] = 0.5 * mJ[t]
    return sorted(int(r) for r in J), sorted(int(r) for r in O), q


def run(cmd: list) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tok = tiny_tokenizer(bos=True)
    rng = np.random.default_rng(0)
    tokens = rng.integers(3, len(tok), size=N_WIN * HW + 7).astype(np.int64)
    offsets = np.concatenate([[0], np.sort(rng.choice(np.arange(5, len(tokens) - 5), 400, replace=False)), [len(tokens)]]).astype(np.int64)
    corpus = {"tokens": tokens, "doc_offsets": offsets}
    (out / "corpus").mkdir()
    np.save(out / "corpus" / "tokens.npy", tokens)
    np.save(out / "corpus" / "doc_offsets.npy", offsets)
    models = mixed_models()
    items = build(tok, tokens, offsets, models)
    special = {"echo": items["poles"][5]["pole"], "shared": items["poles"][7]["pole"], "declined": items["poles"][9]["pole"]}
    dn = [44, 46, 48]
    for r in dn:
        items["poles"][5]["g"][r], items["poles"][5]["u"][r] = -1.0, -1.0
    t15_pole, t15_ranks = items["poles"][0]["pole"], [48, 49]
    for r in t15_ranks:
        items["poles"][0]["a"][r] *= 2.0
    ip = out / "items.json"
    ip.write_text(json.dumps(items))
    sha = hashlib.sha256(ip.read_bytes()).hexdigest()

    # ---------------------------------------------------------------- the bank: the capture job at every load
    with torch.inference_mode():
        for t in TS:
            LL.Job(out / "run" / f"t{int(t[2:]):02d}", D1.args_for(t, sha), items, models[t], tok, corpus, "cpu").run()
    print("bank: 16 loads written")

    # ---------------------------------------------------------------- plants: births, clusters, stats
    q999 = {t: {L: np.full((WIDTH, 2), HUGE) for L in range(4)} for t in range(16)}
    rows, planted, cyc = [], {}, list(range(4, 14))
    for i, p in enumerate(items["poles"]):
        sg = 1.0 if p["sign"] == "+" else -1.0
        A = np.zeros((16, K))
        P = np.zeros((16, K))
        for t in range(16):
            with np.load(out / "run" / f"t{t:02d}" / "lowlayer" / f"{p['pole']}.npz") as z:
                A[t], P[t] = sg * z["a_h"], z["aP"]
        ids, docs = tokens[np.asarray(p["tok"])], np.asarray(p["doc"])
        skip = set(dn) if p["pole"] == special["echo"] else set()
        hold = set(t15_ranks) if p["pole"] == t15_pole else set()
        got = None
        for k in range(len(cyc)):
            tb = cyc[(i + k) % len(cyc)]
            got = choose(A, P, ids, docs, tb, skip | hold, p["pole"] == special["shared"])
            if got:
                break
        if got is None:
            print("no plant possible on", p["pole"])
            cl = [{"c": c, "name": f"pair {c}", "echo": False, "exemplar_ranks": [2 * c, 2 * c + 1]} for c in range(K // 2)]
        else:
            J, O, q = got
            planted[p["pole"]] = {"birth": tb, "J": J, "O": O}
            q999_si = 0 if p["sign"] == "+" else 1
            for t in range(16):
                q999[t][p["layer"]][p["neuron"], q999_si] = q[t]
            rest = [r for r in range(K) if r not in J and r not in O and r not in skip and r not in hold]
            cl = [{"c": 0, "name": "older", "echo": False, "exemplar_ranks": O}, {"c": 1, "name": "newer", "echo": False, "exemplar_ranks": J}]
            if skip:
                cl.append({"c": 2, "name": "echo", "echo": True, "exemplar_ranks": sorted(skip)})
            if hold:
                m15 = float(np.median(A[15, sorted(hold)]))
                assert m15 > 0, (p["pole"], m15)
                cl.append({"c": len(cl), "name": "T_15 only on topk.a", "echo": False, "exemplar_ranks": sorted(hold)})
                q999[15][p["layer"]][p["neuron"], q999_si] = 1.5 * m15
            cl += [{"c": len(cl) + k, "name": f"pair {k}", "echo": False, "exemplar_ranks": rest[2 * k:2 * k + 2]} for k in range((len(rest) + 1) // 2)]
        rows.append({"pole": p["pole"], "sample": 0, "clusters": cl, "pairs": []})
        rows.append({"pole": p["pole"], "sample": 1, "declined": "attempts cap (planted)"} if p["pole"] == special["declined"] else
                    {"pole": p["pole"], "sample": 1, "clusters": cl, "pairs": []})
    (out / "clusters.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for t in range(16):
        d = out / "stats" / f"t{t:02d}" / "stats"
        d.mkdir(parents=True)
        for L in range(4):
            np.savez(d / f"layer_{L:02d}.npz", q999=q999[t][L], max_abs=q999[t][L] / 0.3)
    assert t15_pole in planted, (t15_pole, "the T_15 plant needs a planted pole")
    print(f"planted: {len(planted)} poles; special {special}; T_15-only cluster on {t15_pole} ranks {t15_ranks}")
    cdir = out / "clusterer"
    (cdir / "explain" / "responses").mkdir(parents=True)
    (cdir / "explain" / "responses" / "llc-clusterer-s0.rejected.jsonl").write_text("\n".join(json.dumps({"id": f"planted:{k}"}) for k in range(3)) + "\n")
    (cdir / "reasks.json").write_text(json.dumps({"withheld": 2, "withheld_ids": ["planted:a", "planted:b"], "rule": "planted", "note": "dry run"}))
    (out / "clusterer_no_reasks" / "explain" / "responses").mkdir(parents=True)

    # ---------------------------------------------------------------- the items builder
    py = sys.executable
    b = run([py, "studies/reuse/scripts/build_items_lowlayer.py", "--run", str(out / "run"), "--clusters", str(out / "clusters.jsonl"), "--out", str(out / "inputs"),
             "--poles", str(ip), "--corpus", str(out / "corpus"), "--stats-root", str(out / "stats")])
    print(b.stdout[-2500:], b.stderr[-2500:])
    assert b.returncode == 0, "the items builder failed"
    it = json.loads((out / "inputs" / "lowlayer_items.json").read_text())
    got = {c["pole"]: c for c in it["cases"]}
    for pole, pl in planted.items():
        if pole in (special["shared"], special["declined"]):
            assert pole not in got, (pole, "a planted case that must be filtered is a case")
            continue
        c = got[pole]
        assert c["birth"] == pl["birth"] and c["J_ranks"] == pl["J"] and c["O_ranks"] == pl["O"], (pole, c["birth"], pl)
    assert it["counts"]["dropped_by_filter_overlapping"].get("shares_document", 0) >= 1, it["counts"]["dropped_by_filter_overlapping"]
    assert special["declined"] in it["counts"]["declined"] and it["counts"]["status"].get("echo", 0) == 1, (it["counts"]["declined"], it["counts"]["status"])
    assert len(it["cases"]) == len(planted) - 2, (len(it["cases"]), len(planted))
    t15 = it["t15_dating"]
    assert t15["n_births_differ_if_capture_T15_a_h"] == 1 and t15["differing"][0]["cluster"].startswith(t15_pole + ":"), t15
    assert t15["differing"][0]["birth_topk_a"] is not None and t15["differing"][0]["birth_capture_a_h"] is None, t15
    print(f"items: T_15 dated on the poles file's a; {t15['n_births_differ_if_capture_T15_a_h']} birth would differ on the capture's a_h: {t15['differing']}")
    print(f"items: {len(it['cases'])} cases, every planted birth recovered; shared-document job filtered; echo cluster dated as echo; declined pole counted")

    # ---------------------------------------------------------------- the verdict and its failing inputs
    vcmd = [py, "studies/reuse/scripts/verdict_lowlayer.py", "--run", str(out / "run"), "--items", str(out / "inputs" / "lowlayer_items.json"), "--out",
            str(out / "verdict"), "--poles", str(ip), "--clusters", str(out / "clusters.jsonl")]
    v = run(vcmd + ["--clusterer-dir", str(cdir)])
    print(v.stdout[-4000:], v.stderr[-3000:])
    assert v.returncode == 0, f"the verdict exits {v.returncode} on the planted bank"
    ra = json.loads((out / "verdict" / "verdict_lowlayer.json").read_text())["reported"]["clusterer_reasks"]
    assert ra["reasks_per_request"] == 5 / (2 * len(items["poles"])) and ra["rejected_rows"] == {"s0": 3, "s1": 0}, ra
    print(f"re-asks: {ra['rejected_rows']} rejected + {ra['withheld']} withheld over {ra['n_requests']} requests = {ra['reasks_per_request']:.4f}")
    nr = run(vcmd + ["--clusterer-dir", str(out / "clusterer_no_reasks")])
    assert nr.returncode != 0 and "reasks.json" in nr.stderr, (nr.returncode, nr.stderr[-600:])
    print(f"re-asks without reasks.json: exit {nr.returncode}; {nr.stderr.strip().splitlines()[-1]}")
    vcmd += ["--clusterer-dir", str(cdir)]
    for mode in MODES:
        f = run(vcmd + ["--failing-input", mode])
        last = [x for x in f.stdout.split("\n") if x.startswith("FAILING_INPUT")]
        print(f"failing input {mode}: exit {f.returncode}; {last[-1] if last else f.stderr[-600:]}")
        assert f.returncode == 1, (mode, f.returncode)
    print("REUSE_DRYRUN_LOWLAYER2_OK")


if __name__ == "__main__":
    main()
