"""Weight-free dry run of INSTRUMENTS E3's capture job (infra/aws_pg/jobs/instruments_e3.py), on CPU. Nothing is downloaded.

    python -m infra.aws_pg.jobs.instruments_dryrun_e3 --out <dir>

A tiny OLMo 3 built in code (6 layers, hidden 64, 256 MLP neurons, bf16, seed 15), a synthetic corpus of 12 windows of 512 tokens (the dry harvest
window) with random document bounds, and items in e3_items.json's format: 16 poles (neurons 10 + L and 40 + L, both signs, layers 1 to 4), 50 random
contexts each, a random clusterer partition C and a perturbed second sample C1, B_t = reuse_birth.high_background of ā on 100 random positions (the
V flag) and of ā on 100 other positions (the known number's exclusion, as SPEC10's own load), 4 known cases whose banked D'(J) is made by SPEC10's
route: a capture of J's contexts alone (batch 3), reuse_ablate.drivers_direct, reuse_birth.top_m_excluded. The bank's a per context is a_P from an
independent route (each context alone, [BOS] + at most 256 tokens, g and u from forward hooks; reuse_dryrun_lowlayer.reference_instrument).
Checks on the normal run: status done; every gate passing and its failing input failing as the job records them; layer 5 never runs (a forward hook
counts its calls); per pole npz: keys, dtypes, shapes, tok and token strings equal to the items', U the union of top_idx, U_in_B the B_t flag, top_A
equal to top_a * top_c (1e-5 relative), A_U and top_idx equal to direct_A from a second capture of the same contexts at the job's batch (5e-3 of the
row-set max: fp16 storage and CPU bf16 noise; top-200 sets, Jaccard above 0.95), a_P against the independent route (median relative difference under 0.02); samples/e3.jsonl holds the 5 sample poles with 50 contexts
and 3 x 5 drivers each.
Gates fed their failing inputs: the job stops (AssertionError; the gate's record pass False in metrics.json) on a known bank shifted by one case and on
an items token id that is not the corpus token; gate_ac fails on c read from layer l + 1; gate_complete fails on a missing pole and on a non-finite
A_U. Prints INSTRUMENTS_DRYRUN_E3_OK.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import numpy as np
import torch

from infra.aws_pg.jobs import instruments_e3 as E3
from infra.aws_pg.jobs.neuron_off_dryrun import VOCAB, tiny_tokenizer
from infra.aws_pg.jobs.reuse_dryrun_lowlayer import reference_instrument
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import unite_route as UR

N_LAYERS, HIDDEN, WIDTH = 6, 64, 256
HW, N_WIN, N = 512, 12, E3.N_CTX
POLE_LAYERS = (1, 2, 3, 4)


def tiny_olmo3(seed: int):
    import transformers as T
    torch.manual_seed(seed)
    cfg = T.Olmo3Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=WIDTH, num_hidden_layers=N_LAYERS, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, max_position_embeddings=1024, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    return T.Olmo3ForCausalLM(cfg).to(torch.bfloat16).eval().requires_grad_(False)


def kw(tokens, bos, batch):
    return dict(tokens=tokens, length=RA.LENGTH, bos=bos, batch=batch, hw=HW, device="cpu")


def partition(rng, k: int) -> list:
    lab = np.concatenate([np.arange(k), rng.integers(k, size=N - k)])
    rng.shuffle(lab)
    return lab


def clusters(lab, echo_c=None) -> list:
    return [{"c": int(c), "name": f"dry cluster {c}", "ranks": np.flatnonzero(lab == c).tolist(), "echo": bool(c == echo_c)} for c in np.unique(lab)]


def one_line(text: str) -> str:
    """judge_items.one_line: a line break shown as ↵."""
    return text.replace("\r\n", "\n").replace("\n", "↵")


def build(model, tok, tokens, offsets) -> dict:
    rng = np.random.default_rng(0)
    bos = tok.bos_token_id
    rows = UR.Rows(model)
    valid = np.array([t for t in range(N_WIN * HW) if t % HW != 0])
    abar = lambda pos: RA.background_mean_a(model, pos, N_LAYERS - 1, **kw(tokens, bos, 16))
    hb4 = RB.high_background(abar(rng.choice(valid, 100, replace=False)), N_LAYERS - 1)
    hb10 = RB.high_background(abar(rng.choice(valid, 100, replace=False)), N_LAYERS - 1)
    poles = []
    for L in POLE_LAYERS:
        for n in (10 + L, 40 + L):
            for s in "+-":
                pole = f"L{L:02d}N{n}{s}"
                tk = np.sort(rng.choice(valid, N, replace=False))
                tk = tk[rng.permutation(N)]
                ref = reference_instrument(model, tokens, bos, tk, L, n, 1.0 if s == "+" else -1.0)
                k = 1 if (L, n, s) == (4, 44, "-") else int(rng.integers(2, 6))
                lab = partition(rng, k)
                lab1 = np.where(rng.random(N) < 0.2, rng.integers(k, size=N), lab)
                if len(np.unique(lab1)) < k:
                    lab1 = lab
                ctx = [{"rank": r, "tok": int(t), "doc": int(np.searchsorted(offsets, t, side="right") - 1), "token_id": int(tokens[t]),
                        "token_string": one_line(tok.decode([int(tokens[t])])), "a_bank": float((1.0 if s == "+" else -1.0) * ref[r])} for r, t in enumerate(tk)]
                poles.append({"pole": pole, "layer": L, "neuron": n, "sign": s, "band": "low" if L <= 3 else "deep",
                              "sources": ["lowlayer"] if n == 10 + L and L <= 3 else ["dating", "primary"], "cluster_bank": "lowlayer" if n == 10 + L and L <= 3 else "w4",
                              "C": clusters(lab, echo_c=0 if pole == "L02N12+" else None), "C1": clusters(lab1), "contexts": ctx})
    mask10 = torch.as_tensor(hb10["mask"])
    known = []
    for p in [q for q in poles if q["sign"] == "+" and q["neuron"] == 10 + q["layer"]]:
        unit = UR.pole_unit(p["pole"])
        J = sorted(rng.choice(N, 8, replace=False).tolist())
        cap = UR.capture(model, np.asarray([p["contexts"][r]["tok"] for r in J]), p["layer"], {p["layer"]}, **kw(tokens, bos, 3))
        _, A_J = RA.drivers_direct(rows, cap, unit, WIDTH, ms=(E3.M_KNOWN,))
        D = RB.top_m_excluded(A_J.sum(0), E3.M_KNOWN, WIDTH, mask10)
        known.append({"pole": p["pole"], "c": 0, "J_ranks": J, "D": [[int(l), int(k)] for l, k in D], "bank": "dryrun: SPEC10's route", "bank_sha256": "dry"})
    members = lambda hb: [[int(l), int(k)] for l, k, _ in hb["members"]]
    return {"t": "T_15", "grid": "t15_step1413814", "counts": {"poles": len(poles), "known_poles": len(known)}, "corpus": {"tokens_sha256": "dry"},
            "B_t": {"path": "dryrun", "sha256": "dry", "cut": hb4["cut"], "count": hb4["count"], "n_up": N_LAYERS - 1, "members": members(hb4)},
            "known": {"cases": known, "B_t_path": "dryrun", "B_t_sha256": "dry", "B_t_members": members(hb10), "m": E3.M_KNOWN, "line": E3.KNOWN_LINE},
            "sample_poles": ["L01N11+", "L02N42-", "L03N13+", "L04N14-", "L04N44+"], "dropped_layer0": [], "poles": poles}


def args_for():
    return argparse.Namespace(t="T_15", revision="dry", items_sha256="dry", harvest_window=HW, batch=8, full_batch=8)


def stops(out, items, model, tok, corpus, key, text: str) -> dict:
    """The job ends with an AssertionError naming the gate; with a key, its record in metrics.json says pass False."""
    raised = None
    try:
        E3.Job(out, args_for(), items, model, tok, corpus, "cpu").run()
    except AssertionError as e:
        raised = str(e)
    m = json.loads((out / "metrics.json").read_text())
    assert raised is not None and text in raised, (text, raised)
    assert key is None or m[key]["pass"] is False, (key, m.get(key))
    return {"gate": key or "marked token", "message": raised[:160], **({"recorded": {k: m[key][k] for k in ("median_overlap", "line", "pass")}} if key else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tok = tiny_tokenizer(bos=True)
    rng = np.random.default_rng(1)
    tokens = rng.integers(3, len(tok), size=N_WIN * HW + 7).astype(np.int64)
    offsets = np.concatenate([[0], np.sort(rng.choice(np.arange(100, len(tokens) - 100), 20, replace=False)), [len(tokens)]]).astype(np.int64)
    corpus = {"tokens": tokens, "doc_offsets": offsets}
    model = tiny_olmo3(15)
    items = build(model, tok, tokens, offsets)
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False))
    report = {}

    # ---------------------------------------------------------------- the normal run
    calls = []
    h = model.model.layers[N_LAYERS - 1].register_forward_hook(lambda m, a, o: calls.append(1))
    bank = out / "bank"
    E3.Job(bank, args_for(), items, model, tok, corpus, "cpu").run()
    h.remove()
    m = json.loads((bank / "metrics.json").read_text())
    assert m["status"] == "done" and not calls, (m.get("status"), len(calls))
    for g in ("known_number", "ac"):
        assert m[g]["pass"] is True and m[g]["failing_input_failed"] is True, (g, m[g])
    assert m["completeness"]["pass"] and m["n_done"]["poles"] == len(items["poles"]) and m["captures"] == N * len(items["poles"]), m["completeness"]
    rows, bos, flag = UR.Rows(model), tok.bos_token_id, RB.mask_of_members([(l, k, 0) for l, k in items["B_t"]["members"]], N_LAYERS - 1, WIDTH).reshape(-1)
    worst = {"A_U_vs_second_capture": 0.0, "top200_jaccard_min": 1.0, "ac_banked": 0.0, "aP_vs_reference_median": 0.0}
    for p in items["poles"]:
        with np.load(bank / "e3" / f"{p['pole']}.npz") as z:
            assert sorted(z.files) == sorted(E3.NPZ_KEYS), z.files
            assert z["A_U"].dtype == np.float16 and np.abs(z["A_U"]).max() == 1.0 and z["top_idx"].dtype == np.int64 and z["aP"].dtype == np.float64 and z["U_in_B"].dtype == bool
            assert np.array_equal(z["tok"], [c["tok"] for c in p["contexts"]]) and [json.loads(x) for x in z["token_json"]] == [c["token_string"] for c in p["contexts"]]
            assert np.array_equal(z["U"], np.unique(z["top_idx"])) and np.array_equal(z["U_in_B"], flag[z["U"]])
            rel = np.abs(z["top_A"] - z["top_a"] * z["top_c"]).max(1) / np.abs(z["top_A"]).max(1)
            unit = UR.pole_unit(p["pole"])
            cap = UR.capture(model, z["tok"], p["layer"], {p["layer"]}, **kw(tokens, bos, 8))
            A2 = RA.direct_A(rows, cap, unit, WIDTH).numpy()
            du = np.abs(z["A_U"].astype(np.float64) * z["A_U_scale"][:, None] - A2[:, z["U"]]).max() / np.abs(A2).max()
            t2 = np.argsort(-A2, axis=1, kind="stable")[:, :E3.K_TOP]
            jac = min(len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(z["top_idx"], t2))
            ref = np.asarray([c["a_bank"] for c in p["contexts"]]) * (1.0 if p["sign"] == "+" else -1.0)
            rd = float(np.median(np.abs(z["aP"] - ref) / np.abs(ref)))
            for k, v in (("A_U_vs_second_capture", du), ("ac_banked", float(rel.max())), ("aP_vs_reference_median", rd)):
                worst[k] = max(worst[k], float(v))
            worst["top200_jaccard_min"] = min(worst["top200_jaccard_min"], jac)
    assert worst["A_U_vs_second_capture"] < 5e-3 and worst["top200_jaccard_min"] > 0.95 and worst["ac_banked"] < 1e-5 and worst["aP_vs_reference_median"] < 0.02, worst
    smp = [json.loads(x) for x in (bank / "samples" / "e3.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [s["pole"] for s in smp] == sorted(items["sample_poles"]), [s["pole"] for s in smp]
    assert all(len(s["contexts"]) == N and len(s["drivers"]) == 3 and all(len(d["top"]) == E3.SAMPLE_TOP for d in s["drivers"]) and s["V"] is None for s in smp)
    report["normal"] = {"known_number": {k: m["known_number"][k] for k in ("median_overlap", "min_overlap", "failing_input_median_overlap", "pass", "failing_input_failed")},
                        "ac": m["ac"], "layer5_calls": len(calls), "checks": worst, "n_U_median": float(np.median([v["n_U"] for v in m["poles"].values()])),
                        "n_U_in_B_total": int(sum(v["n_U_in_B"] for v in m["poles"].values())), "samples": len(smp)}
    print("normal run:", json.dumps(report["normal"]))
    print("sample row:", json.dumps({k: smp[0][k] for k in ("pole", "layer", "band")}), json.dumps(smp[0]["drivers"][0], ensure_ascii=False)[:300])

    # ---------------------------------------------------------------- each gate fed its failing input
    fi, fed = out / "failing", []
    w = json.loads(json.dumps(items))
    Ds = [c["D"] for c in w["known"]["cases"]]
    for c, D in zip(w["known"]["cases"], Ds[1:] + Ds[:1]):
        c["D"] = D
    fed.append(stops(fi / "known", w, model, tok, corpus, "known_number", "known number failed"))
    w = json.loads(json.dumps(items))
    w["poles"][3]["contexts"][7]["token_id"] = (w["poles"][3]["contexts"][7]["token_id"] + 1) % len(tok)
    fed.append(stops(fi / "token", w, model, tok, corpus, None, "the marked token differs"))
    p = items["poles"][5]
    unit = UR.pole_unit(p["pole"])
    cap = UR.capture(model, np.asarray([c["tok"] for c in p["contexts"]]), p["layer"], {p["layer"]}, **kw(tokens, bos, 8))
    A = RA.direct_A(rows, cap, unit, WIDTH)
    a_all = torch.cat([cap["a"][l] for l in range(p["layer"])], 1)
    rw = E3.ac_rel(A, a_all, E3.coefficient(rows, cap, unit, shift=1))
    ga, gw = E3.gate_ac(E3.ac_rel(A, a_all, E3.coefficient(rows, cap, unit)), rw), E3.gate_ac(rw, rw)
    assert ga["pass"] is True and gw["pass"] is False, (ga, gw)
    fed.append({"gate": "A = a * c", "c from layer l + 1": {k: gw[k] for k in ("max_rel", "tol", "pass")}, "c as built": ga["max_rel"]})
    banked = {}
    for q in items["poles"]:
        with np.load(bank / "e3" / f"{q['pole']}.npz") as z:
            banked[q["pole"]] = {k: z[k] for k in z.files}
    assert E3.gate_complete(banked, items["poles"], "T_15")["pass"] is True
    q = items["poles"][2]["pole"]
    bad = banked[q]["A_U"].copy()
    bad[0, 0] = np.inf
    gc1 = E3.gate_complete({**banked, q: {**banked[q], "A_U": bad}}, items["poles"], "T_15")
    gc2 = E3.gate_complete({k: v for k, v in banked.items() if k != q}, items["poles"], "T_15")
    assert gc1["pass"] is False and gc1["bad"] and gc2["pass"] is False and gc2["missing"] == [q], (gc1, gc2)
    fed.append({"gate": "completeness", "non_finite": gc1["bad"], "missing": gc2["missing"]})
    report["gates_fed_failing_inputs"] = fed
    for r in fed:
        print("failing input:", json.dumps(r, ensure_ascii=False)[:300])
    (out / "dryrun_report.json").write_text(json.dumps(report, indent=1))
    print("INSTRUMENTS_DRYRUN_E3_OK", out)


if __name__ == "__main__":
    main()
