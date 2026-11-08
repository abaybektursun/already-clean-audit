"""Weight-free dry run of SPEC12 LOWLAYER's capture job (infra/aws_pg/jobs/reuse_lowlayer.py), on CPU. Nothing is downloaded.

    python -m infra.aws_pg.jobs.reuse_dryrun_lowlayer --out <dir>

A tiny OLMo 3 built in code (5 layers, hidden 64, 96 MLP neurons, bf16), one random model per load T_0..T_15 (seed = load index), a synthetic
corpus of 12 windows of 512 tokens (the dry harvest window) with random document bounds, and a planted bank made by an independent route:
  harvest condition: each window [BOS] + 512 tokens through the whole model, the model's own input to mlp.down_proj (bf16) read by a forward
      pre-hook (not MLPTap, not the fp32 recomputation); the T_15 model's top-50 positions per pole (window position 0 excluded) are the pole's
      positions, its values there the T_15 topk.a, every load's values there the known poles' exemplar_a;
  SPEC8 instrument: each context alone, [BOS] + at most 256 tokens ending at it, through the whole model; g and u the bf16 outputs of the pole's
      gate_proj and up_proj at the last position (forward hooks), a_P = sign silu(g) u in float64 -> a planted dating/<pole>__c<c>.npz per load.
Items in lowlayer_poles.json's format: 16 new poles (2 neurons x 2 signs per layer 0-3; the first a layer-0 pole), 8 known poles (both poles of one
neuron per layer), one SPEC8 case (7 J and 6 O* contexts) on a layer-3 known pole. The job runs at T_15, T_0, T_1 and T_14 (both ends of each
failing input's load rule). Checks per load: status done; every gate passing and its failing input failing as the job records them; capture
identity exactly 0 for a_h; a_h against the independent reference (median relative difference under 0.02); the npz format (keys, dtypes,
shapes, tok and doc equal to the items'); layer 4 never runs (a forward hook counts its calls); the harvest windows forwarded equal the distinct
windows per pole plus the identity's; the two windows: a_P equals sign a_h where the 512-window prefix fits in 256 tokens, and differs beyond.
Gates fed their failing inputs: the job stops (AssertionError, the gate's record pass False in metrics.json) on the previous load's exemplar bank,
on the next load's SPEC8 bank and on the reversed topk.a at T_15; gate_identity fails on the layer L + 1 reading as the second capture;
gate_complete fails on a non-finite a_h and on a missing pole. Prints REUSE_DRYRUN_LOWLAYER_OK.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil

import numpy as np
import torch

from infra.aws_pg.jobs import reuse_lowlayer as LL
from infra.aws_pg.jobs.neuron_off_dryrun import VOCAB, tiny_tokenizer
from infra.pg import unite_route as UR

N_LAYERS, HIDDEN, WIDTH = 5, 64, 96
HW, N_WIN, K = 512, 12, 50
UPTO = 3
LOADS = ("T_15", "T_0", "T_1", "T_14")
TS = [f"T_{t}" for t in range(16)]


def tiny_olmo3(seed: int):
    import transformers as T
    torch.manual_seed(seed)
    cfg = T.Olmo3Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=WIDTH, num_hidden_layers=N_LAYERS, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, max_position_embeddings=1024, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    return T.Olmo3ForCausalLM(cfg).to(torch.bfloat16).eval().requires_grad_(False)


def silu64(g):
    return g / (1 + np.exp(-g))


@torch.inference_mode()
def reference_windows(model, tokens, bos) -> np.ndarray:
    """[layer, window, window position] of every neuron: the model's own down_proj input (bf16 -> fp32) over [BOS] + each window."""
    layers = model.model.layers
    out = np.zeros((N_LAYERS, N_WIN, HW, WIDTH), np.float32)
    for w in range(N_WIN):
        seen = {}
        hs = [layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: seen.__setitem__(l, a[0][0, 1:].float().numpy())) for l in range(N_LAYERS)]
        x = torch.tensor([[bos] + tokens[w * HW:(w + 1) * HW].tolist()])
        model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
        for h in hs:
            h.remove()
        for l in range(N_LAYERS):
            out[l, w] = seen[l]
    return out


@torch.inference_mode()
def reference_instrument(model, tokens, bos, toks, L, n, sign) -> np.ndarray:
    """a_P per context, each context alone: [BOS] + at most 256 tokens ending at it; g, u the bf16 gate_proj/up_proj outputs at the last position."""
    mlp = model.model.layers[L].mlp
    res = []
    for t in toks:
        t = int(t)
        start = max(t - t % HW, t - 255)
        got = {}
        hs = [mlp.gate_proj.register_forward_hook(lambda m, a, o: got.__setitem__("g", float(o[0, -1, n]))),
              mlp.up_proj.register_forward_hook(lambda m, a, o: got.__setitem__("u", float(o[0, -1, n])))]
        x = torch.tensor([[bos] + tokens[start:t + 1].tolist()])
        model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
        for h in hs:
            h.remove()
        g, u = np.float32(got["g"]), np.float32(got["u"])
        res.append(sign * silu64(np.float64(g)) * np.float64(u))
    return np.asarray(res)


def top_tok(ref15, L, n, s) -> np.ndarray:
    sc = (1.0 if s == "+" else -1.0) * ref15[L, :, :, n].astype(np.float64)
    sc[:, 0] = -np.inf                                         # window position 0 is not a context
    flat = np.argsort(-sc.ravel(), kind="stable")[:K]
    return (flat // HW * HW + flat % HW).astype(np.int64)


def build(tok, tokens, offsets, models) -> tuple:
    bos = tok.bos_token_id
    refs = {t: reference_windows(models[t], tokens, bos) for t in TS}
    at = lambda ref, L, n, toks: ref[L, toks // HW, toks % HW, n].astype(np.float64)
    doc = lambda toks: (np.searchsorted(offsets, toks, side="right") - 1).astype(np.int64)
    poles = []
    for L in range(4):
        for n in (10 + L, 40 + L):
            for s in "+-":
                tk = top_tok(refs["T_15"], L, n, s)
                poles.append({"pole": f"L{L:02d}N{n}{s}", "layer": L, "neuron": n, "sign": s, "tok": tk.tolist(), "doc": doc(tk).tolist(),
                              "a": at(refs["T_15"], L, n, tk).tolist()})
    known = []
    for L in range(4):
        n = 70 + L
        for s in "+-":
            tk = top_tok(refs["T_15"], L, n, s)
            rec = {"pole": f"L{L:02d}N{n}{s}", "layer": L, "neuron": n, "sign": s, "tok": tk.tolist(), "doc": doc(tk).tolist(),
                   "a_bank": {t: at(refs[t], L, n, tk).tolist() for t in TS}, "spec8_cases": []}
            if L == 3 and s == "+":
                ctx = np.concatenate([tk[:7], tk[7:13]])
                aP = {t: reference_instrument(models[t], tokens, bos, ctx, L, n, 1.0) for t in TS}
                rec["spec8_cases"].append({"c": 0, "side": ["J"] * 7 + ["O"] * 6, "tok": ctx.tolist(), "aP": {t: v.tolist() for t, v in aP.items()},
                                           "sha256": {t: hashlib.sha256(v.tobytes()).hexdigest() for t, v in aP.items()}})
            known.append(rec)
    items = {"poles": poles, "known": known, "upto_layer": UPTO, "grid": {t: f"t{int(t[2:]):02d}_step{int(t[2:])}" for t in TS},
             "corpus": {"tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest()}}
    return items, refs


def args_for(t: str, items_sha: str):
    return argparse.Namespace(t=t, revision=f"dry-{t}", items_sha256=items_sha, harvest_window=HW, harvest_batch=3, batch=8, full_batch=8)


def run_job(out, t, items, model, tok, corpus):
    return LL.Job(out, args_for(t, "dry"), items, model, tok, corpus, "cpu").run()


def stops(out, t, items, model, tok, corpus, key: str, text: str):
    """The job ends with an AssertionError naming the gate, and its record in metrics.json says pass False."""
    raised = None
    try:
        run_job(out, t, items, model, tok, corpus)
    except AssertionError as e:
        raised = str(e)
    m = json.loads((out / "metrics.json").read_text())
    assert raised is not None and text in raised and m[key]["pass"] is False, (key, raised)
    return {"gate": key, "message": raised, "recorded": {k: m[key][k] for k in ("median_rel_diff", "tol", "pass")}}


def windows_expected(items) -> int:
    per = sum(len({t // HW for t in p["tok"]}) for p in items["known"] + items["poles"])
    return per + 2 * len({t // HW for t in items["poles"][0]["tok"][:LL.N_ID]})


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
    offsets = np.concatenate([[0], np.sort(rng.choice(np.arange(100, len(tokens) - 100), 20, replace=False)), [len(tokens)]]).astype(np.int64)
    corpus = {"tokens": tokens, "doc_offsets": offsets}
    models = {t: tiny_olmo3(int(t[2:])) for t in TS}
    items, refs = build(tok, tokens, offsets, models)
    (out / "items.json").write_text(json.dumps(items))
    report = {"loads": {}, "gates_fed_failing_inputs": []}

    for t in LOADS:
        model = models[t]
        calls = []
        h = model.model.layers[4].register_forward_hook(lambda m, a, o: calls.append(1))
        d = out / "run" / t
        run_job(d, t, items, model, tok, corpus)
        h.remove()
        m = json.loads((d / "metrics.json").read_text())
        assert m["status"] == "done" and not calls, (m.get("status"), len(calls))
        for g in ("known_harvest", "known_instrument", "capture_identity"):
            assert m[g]["pass"] is True and m[g]["failing_input_failed"] is True, (t, g, m[g])
        assert m["capture_identity"]["a_h"]["max_abs_diff"] == 0.0 and m["capture_identity"]["a_P"]["max_abs_diff"] < 1e-5, m["capture_identity"]
        assert m["completeness"]["pass"] and m["completeness"]["n_contexts_complete"] == 24 * K, m["completeness"]
        assert (t == "T_15") == ("known_harvest_new_T15" in m), t
        if t == "T_15":
            assert m["known_harvest_new_T15"]["pass"] and m["known_harvest_new_T15"]["failing_input_failed"], m["known_harvest_new_T15"]
        prev = {"T_0": "T_1"}.get(t, f"T_{int(t[2:]) - 1}")
        nxt = {"T_15": "T_14"}.get(t, f"T_{int(t[2:]) + 1}")
        assert prev in m["known_harvest"]["failing_input"] and nxt in m["known_instrument"]["failing_input"], (m["known_harvest"]["failing_input"], m["known_instrument"]["failing_input"])
        assert m["harvest_windows"] == windows_expected(items), (m["harvest_windows"], windows_expected(items))
        ref_rd, short, far = [], [], []
        for p in items["known"] + items["poles"]:
            with np.load(d / "lowlayer" / f"{p['pole']}.npz") as z:
                assert sorted(z.files) == sorted(LL.NPZ_KEYS), z.files
                assert z["a_h"].dtype == np.float32 and z["gP"].dtype == np.float32 and z["uP"].dtype == np.float32 and z["aP"].dtype == np.float64
                assert z["tok"].dtype == np.int64 and np.array_equal(z["tok"], p["tok"]) and np.array_equal(z["doc"], p["doc"]) and str(z["t"]) == t
                tk = np.asarray(p["tok"])
                ref = refs[t][p["layer"], tk // HW, tk % HW, p["neuron"]]
                ref_rd.append(LL.rel_diff(z["a_h"], ref))
                sg = 1.0 if p["sign"] == "+" else -1.0
                rd = np.abs(z["aP"] - sg * z["a_h"]) / np.abs(z["a_h"])
                short += rd[tk % HW <= 255].tolist()
                far += rd[tk % HW > 255].tolist()
        assert max(ref_rd) < 0.02, max(ref_rd)
        assert short and far and np.median(short) < 0.02 and np.median(far) > 10 * np.median(short), (np.median(short), np.median(far))
        report["loads"][t] = {"known_harvest": m["known_harvest"]["median_rel_diff"], "known_harvest_failing": m["known_harvest"]["failing_input_median_rel_diff"],
                              "known_instrument": m["known_instrument"]["median_rel_diff"], "known_instrument_failing": m["known_instrument"]["failing_input_median_rel_diff"],
                              "identity_a_h": m["capture_identity"]["a_h"]["max_abs_diff"], "identity_a_P": m["capture_identity"]["a_P"]["max_abs_diff"],
                              "identity_failing_a_h": m["capture_identity"]["a_h"]["failing_input_max_abs_diff"],
                              "identity_failing_a_P": m["capture_identity"]["a_P"]["failing_input_max_abs_diff"],
                              "a_h_vs_reference_max_median_rel": max(ref_rd), "aP_vs_a_h_prefix_le_256": float(np.median(short)),
                              "aP_vs_a_h_prefix_gt_256": float(np.median(far)), "harvest_windows": m["harvest_windows"], "layer4_calls": len(calls),
                              **({"known_harvest_new_T15": m["known_harvest_new_T15"]["median_rel_diff"],
                                  "known_harvest_new_T15_failing": m["known_harvest_new_T15"]["failing_input_median_rel_diff"]} if t == "T_15" else {})}
        print(t, json.dumps(report["loads"][t]))

    # ---------------------------------------------------------------- each gate fed its failing input
    model, fi = models["T_15"], out / "failing"
    w = json.loads(json.dumps(items))
    for k in w["known"]:
        k["a_bank"]["T_15"] = k["a_bank"]["T_14"]
    report["gates_fed_failing_inputs"].append(stops(fi / "known_harvest", "T_15", w, model, tok, corpus, "known_harvest", "known number (harvest condition) failed"))
    w = json.loads(json.dumps(items))
    for k in w["known"]:
        for c in k["spec8_cases"]:
            c["aP"]["T_15"] = c["aP"]["T_14"]
    report["gates_fed_failing_inputs"].append(stops(fi / "known_instrument", "T_15", w, model, tok, corpus, "known_instrument", "known number (SPEC8 instrument) failed"))
    w = json.loads(json.dumps(items))
    for p in w["poles"]:
        p["a"] = p["a"][::-1]
    report["gates_fed_failing_inputs"].append(stops(fi / "known_new", "T_15", w, model, tok, corpus, "known_harvest_new_T15", "new poles at T_15) failed"))
    job = LL.Job(fi / "identity", args_for("T_15", "dry"), items, model, tok, corpus, "cpu")
    p = items["poles"][0]
    toks, L, n = np.asarray(p["tok"][:LL.N_ID]), p["layer"], p["neuron"]
    h1, h2 = job.harvest_a(toks, n, [L])[L], job.harvest_a(toks, n, [L, L + 1])
    gi = LL.gate_identity(h1, h2[L + 1], h2[L + 1], LL.ID_TOL_H)
    assert gi["pass"] is False, gi
    report["gates_fed_failing_inputs"].append({"gate": "capture_identity", "second_capture": "layer L + 1", "record": gi})
    d = out / "run" / "T_15"
    banked = {}
    for q in items["known"] + items["poles"]:
        with np.load(d / "lowlayer" / f"{q['pole']}.npz") as z:
            banked[q["pole"]] = {k: z[k] for k in z.files}
    assert LL.gate_complete(banked, items["known"] + items["poles"], "T_15")["pass"] is True
    nan = {**banked, p["pole"]: {**banked[p["pole"]], "a_h": np.where(np.arange(K) == 3, np.nan, banked[p["pole"]]["a_h"]).astype(np.float32)}}
    gc1 = LL.gate_complete(nan, items["known"] + items["poles"], "T_15")
    gone = {k: v for k, v in banked.items() if k != p["pole"]}
    gc2 = LL.gate_complete(gone, items["known"] + items["poles"], "T_15")
    assert gc1["pass"] is False and gc1["bad"] and gc2["pass"] is False and gc2["missing"] == [p["pole"]], (gc1, gc2)
    report["gates_fed_failing_inputs"].append({"gate": "completeness", "non_finite": gc1["bad"], "missing": gc2["missing"]})
    for r in report["gates_fed_failing_inputs"]:
        print("failing input:", json.dumps(r)[:300])
    (out / "metrics.json").write_text(json.dumps(report, indent=1))
    print("REUSE_DRYRUN_LOWLAYER_OK")


if __name__ == "__main__":
    main()
