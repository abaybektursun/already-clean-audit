"""Weight-free dry run of the MECHANICS screen (studies/mechanics/SPEC_mechanics.md §3, §8, §9 step 1). No download, no network, no GPU: a tiny
random Olmo 3 (6 layers, 4 heads of 16, MLP width 128, bf16) built from its transformers config, the byte-level tokenizer of the earlier dry runs
(neuron_off_dryrun), a fake corpus in 256-token harvest windows, and a tiny grid bank harvested from that model with neurons_harvest's own
MLPTap, LayerState and write_layers (stats/ and topk/ in the real layout; manifest.json with the bank's fields).

    python -m infra.aws_pg.jobs.mechanics_dryrun --out <dir>

The fake corpus mixes the tokenized LINES with ids drawn with weights 1 / rank, never from the first 20 non-special ids, so the sampling
vocabulary must drop zero-count ids and the n_top (10 here) most frequent ones. The bank's q999 is set to (0, 0) for neuron (2, 5) and to
(nan, x) for (3, 7): both must be excluded.
Stand-in for an induction head: a random model has none. Run A patches Olmo 3's eager attention so that head 0 of layer 1 puts 0.9 of its weight on
the position after the latest earlier occurrence of the current 2-gram and writes there the direction W_O_h^T (g * W_U[attended token]), so that it
copies (the planted head); the hook points (down_proj input, the attention module's weights, o_proj's input) are the real model's. Run B is the
same job on the unplanted model.
Lines printed:
  SELFCHECK  mechanics.selfcheck().
  RUN A      Job.run on the planted model with a small plan (16 rows per set, batch 8, 64 narrow ids, family caps 10, n_id 8, E floor 0 so the
             families fill); one GATE line per gate of §3: its value passes and its failing input fails; the detector's top head is the planted one.
  SAMPLING   the job's V equals a recount from the corpus counts; no drawn id lies outside V.
  HEADS      the planted head is the first induction head; every induction head copies positively, every negative-copy head negatively.
  RECOUNT    E, E', E_early, E_early', P, P', T, U, T', U', R, F, the weight-side std ratio, skew and kurtosis, the head prefix scores and the
             copying scores recomputed from full stored activations (neurons_harvest.MLPTap, output_attentions, an o_proj hook, full-vocabulary
             logits; numpy float64; independent of mechanics.py's accumulators) equal the job's npz.
  SCALE      bank neurons (2, 5) and (3, 7) are excluded: s NaN, every signature NaN, in no family.
  LAYOUT     verdict_mechanics.load_bank reads the bank; its gate readers agree with the job; the samples gate fails on a copy without the file.
  RUN B      the unplanted model: every gate passes except the detector, which fails.
  ESTIMATE   the real plan's tokens and wall time.
Then MECHANICS_DRYRUN_OK.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

import numpy as np
import torch

from infra.aws_pg.jobs import mechanics_screen as JS
from infra.aws_pg.jobs.neuron_off_dryrun import LINES, tiny_model, tiny_tokenizer
from infra.pg import mechanics as M
from infra.pg import neurons_harvest as NH

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "studies/mechanics/scripts"))
import verdict_mechanics as VM  # noqa: E402

HW, N_WINDOWS, N_LAYERS = 256, 40, 6
PLANT, PLANT_GAIN = (1, 0), 8.0         # the planted head's write: its original norm times PLANT_GAIN
ZERO, NANC = (2, 5), (3, 7)             # bank q999 set to (0, 0) and to (nan, x): both must be excluded
UNSEEN = 20                             # the first 20 non-special ids never occur in the fake corpus (zero count)
DRY_PLAN = {**JS.PLAN, "n_seq": 16, "n_narrow": 64, "n_nat": 8, "family_cap": 10, "random_n": 10, "floors": {"E": 0.0}, "n_id": 8, "batch": 8, "hw": HW,
            "n_top_excl": 10}
E1_RATE = 15904.6            # E1's measured units rate on an L40S (tokens/s), forwards stopped after the pole's layer
FAILING_NAMES = {"known_activation": "known_next_pole", "identity": "identity_one_neuron", "ablation_identity": "ablation_all_heads_one_layer",
                 "detector": "detector_shuffled"}


# ------------------------------------------------------------------------------------------ the planted head (stand-in)
def plant(model):
    """Patch Olmo 3's eager attention: head PLANT[1] of layer PLANT[0] moves 0.9 of its weight to the position after the latest earlier
    occurrence of the current 2-gram, and at those positions writes (as its slice of o_proj's input, PLANT_GAIN times its norm) the direction
    W_O_h^T (g * W_U[attended token]), so that it copies; the input ids reach it through a pre-hook on the embedding."""
    import transformers.models.olmo3.modeling_olmo3 as MO
    orig, state = MO.eager_attention_forward, {}
    tower = NH.text_tower(model)
    lay = tower.layers[PLANT[0]]
    D = model.config.head_dim
    WO_h = lay.self_attn.o_proj.weight[:, PLANT[1] * D:(PLANT[1] + 1) * D].float()
    gU = model.get_output_embeddings().weight.float() * (tower.norm.weight * lay.post_attention_layernorm.weight).float()
    model.model.embed_tokens.register_forward_pre_hook(lambda m, args: state.__setitem__("ids", args[0].clone()))

    def targets(ids):
        t = torch.full(ids.shape, -1, dtype=torch.long)
        for b, row in enumerate(ids.tolist()):
            last = {}
            for q in range(1, len(row)):
                g = (row[q - 1], row[q])
                if g in last:
                    t[b, q] = last[g] + 1
                last[g] = q
        return t

    def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out, w = orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
        if module.layer_idx != PLANT[0]:
            return out, w
        t = targets(state["ids"]).to(w.device)
        wh = w[:, PLANT[1]].float()
        new = 0.1 * wh
        b, q = (t >= 0).nonzero(as_tuple=True)
        new[b, q, t[b, q]] += 0.9
        w = w.clone()
        w[:, PLANT[1]] = torch.where((t >= 0)[..., None], new, wh).to(w.dtype)
        out = torch.matmul(w, MO.repeat_kv(value, module.num_key_value_groups)).transpose(1, 2).contiguous()
        ids = state["ids"].to(w.device)
        d = gU[ids[b, t[b, q]]] @ WO_h                                   # [n, D]: W_O_h^T (g * W_U[token])
        cur = out[b, q, PLANT[1]].float()
        out = out.clone()
        out[b, q, PLANT[1]] = (d / d.norm(dim=-1, keepdim=True) * PLANT_GAIN * cur.norm(dim=-1, keepdim=True)).to(out.dtype)
        return out, w
    MO.eager_attention_forward = eager
    return lambda: setattr(MO, "eager_attention_forward", orig)


# ------------------------------------------------------------------------------------------ the tiny corpus and bank
def fake_corpus(tok) -> dict:
    """Chunks of the tokenized LINES alternating with chunks of ids drawn with weights 1 / rank from the non-special ids after the first
    UNSEEN (those never occur unless LINES uses them; the job must drop every zero-count id and the n_top most frequent)."""
    text = tok(" ".join(LINES * 400), add_special_tokens=False).input_ids
    V0 = M.nonspecial_ids(tok)
    pool = V0[UNSEEN:]
    rng = np.random.default_rng(99)
    wts = 1.0 / np.arange(1, pool.size + 1)
    n = N_WINDOWS * HW
    parts, it = [], 0
    while sum(map(len, parts)) < n:
        parts.append(np.asarray(text[it:it + 200], np.int64))
        parts.append(rng.choice(pool, 200, p=wts / wts.sum()))
        it += 200
    tokens = np.concatenate(parts)[:n]
    return {"tokens": tokens, "doc_offsets": np.array([0, 3001, 7313, n], np.int64), "tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest()}


@torch.no_grad()
def tiny_bank(model, tok, corpus, root: pathlib.Path):
    """The harvest's rule on the tiny model: 256-token windows, BOS first, a = act(g) * u in fp32 at window positions >= 1 into LayerState
    (top 50 per pole, the |a| histogram), written by write_layers; then q999 of neuron ZERO set to (0, 0) and of NANC to (nan, its + value)."""
    tokens = corpus["tokens"]
    layers = list(NH.text_tower(model).layers)
    W = int(layers[0].mlp.gate_proj.weight.shape[0])
    states = {l: NH.LayerState(W, "cpu") for l in range(len(layers))}
    tap = NH.MLPTap(layers)
    acts = [l.mlp.act_fn for l in layers]
    for w in range(N_WINDOWS):
        x = torch.tensor([[tok.bos_token_id] + tokens[w * HW:(w + 1) * HW].tolist()])
        tg = torch.arange(w * HW + 1, (w + 1) * HW)

        def fn(lyr, g, u, a_model):
            gg, uu = g[0, 2:].float(), u[0, 2:].float()
            states[lyr].update(acts[lyr](gg) * uu, gg, uu, tg)
        tap.run(fn, lambda: NH.forward(model, x, False))
    shutil.rmtree(root, ignore_errors=True)
    NH.write_layers(root, {l: s.snapshot() for l, s in states.items()}, tokens)
    for (L, k), val in ((ZERO, (0.0, 0.0)), (NANC, (np.nan, None))):
        p = root / "stats" / f"layer_{L:02d}.npz"
        with np.load(p) as z:
            st = {key: z[key] for key in z.files}
        st["q999"][k, 0] = val[0]
        if val[1] is not None:
            st["q999"][k, 1] = val[1]
        np.savez_compressed(p, **st)
    NH.dump(root / "manifest.json", {"revision": "tiny-olmo3", "branch": "tiny", "tokens_sha256": corpus["tokens_sha256"], "n_windows": N_WINDOWS,
                                     "window": HW, "bos_id": tok.bos_token_id, "n_layers": len(layers), "width": W})


def run_job(out: pathlib.Path, planted: bool):
    tok = tiny_tokenizer(True)
    model = tiny_model("olmo3", seed=0, dtype=torch.bfloat16, n_layers=N_LAYERS)
    model.requires_grad_(False)
    JS.use_eager(model)
    undo = plant(model) if planted else (lambda: None)
    corpus = fake_corpus(tok)
    tiny_bank(model, tok, corpus, out / "bank")
    shutil.rmtree(out / "job", ignore_errors=True)
    job = JS.Job(out / "job", model, tok, corpus, out / "bank", DRY_PLAN, "cpu", {"sha": "tiny-olmo3", "t": "dry", "revision": "tiny"})
    job.run()
    return job, undo


# ------------------------------------------------------------------------------------------ the recount (independent of mechanics.py's accumulators)
@torch.no_grad()
def recount(job) -> dict:
    layers, B, h, s = job.layers, job.plan["batch"], job.half, job.s
    tap = NH.MLPTap(layers)

    def acts(ids):
        got = [[] for _ in layers]
        for b0 in range(0, len(ids), B):
            tap.run(lambda lyr, g, u, a: got[lyr].append(a[:, 1:].double().numpy()), lambda: NH.forward(job.model, job.x_of(ids[b0:b0 + B]), False))
        return [np.concatenate(x) for x in got]
    X = {k: acts(job.seq[k]) for k in ("rep", "A", "B", "narrow")}
    L, W = s.shape
    perm = np.random.default_rng(M.SEEDS["bin_perm"]).permutation(2 * h)
    lab = M.narrow_labels(job.seq["narrow"], job.seq["narrow_ids"], job.plan["min_occ"])
    R = {k: np.zeros((L, W)) for k in ("E", "E_null", "E_early", "E_early_null", "P", "P_null", "R", "F", "T", "U", "T_null", "U_null")}
    e = job.plan["early"]
    for l in range(L):
        d = {k: X[k][l][:, h:].mean((0, 1)) - X[k][l][:, :h].mean((0, 1)) for k in ("rep", "A", "B")}
        R["E"][l], R["E_null"][l] = (d["rep"] - d["A"]) / s[l], (d["B"] - d["A"]) / s[l]
        de = {k: X[k][l][:, h:h + e].mean((0, 1)) - X[k][l][:, :h].mean((0, 1)) for k in ("rep", "A", "B")}
        R["E_early"][l], R["E_early_null"][l] = (de["rep"] - de["A"]) / s[l], (de["B"] - de["A"]) / s[l]
        ab = np.abs(X["A"][l])
        R["F"][l] = np.where(np.isfinite(s[l]), ab[:, :4].mean((0, 1)) / ab[:, 4:].mean((0, 1)), np.nan)
        bA = X["A"][l].mean(0).reshape(8, 2 * h // 8, W).mean(1)
        bB = X["B"][l].mean(0)[perm].reshape(8, 2 * h // 8, W).mean(1)
        R["P"][l], R["P_null"][l] = (bA.max(0) - bA.min(0)) / s[l], (bB.max(0) - bB.min(0)) / s[l]
        R["R"][l] = np.abs(X["A"][l]).mean((0, 1)) / s[l]
        x = X["narrow"][l].reshape(-1, W)
        for key, lk in (("T", "id"), ("U", "prev"), ("T_null", "id_perm"), ("U_null", "prev_perm")):
            ll = lab[lk].reshape(-1)
            xv, lv = x[ll >= 0], ll[ll >= 0]
            mu = xv.mean(0)
            bw = sum((lv == c).sum() * (xv[lv == c].mean(0) - mu) ** 2 for c in np.unique(lv))
            R[key][l] = np.where(np.isfinite(s[l]), bw / ((xv - mu) ** 2).sum(0), np.nan)
    q, k = M.head_index(h)
    hs = {}
    for name in ("rep", "A"):
        tot = np.zeros((L, job.H))
        for b0 in range(0, len(job.seq[name]), B):
            x = job.x_of(job.seq[name][b0:b0 + B])
            att = job.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, output_attentions=True, logits_to_keep=1).attentions
            for l in range(L):
                tot[l] += sum(att[l][:, :, q[i], k[i]].double().sum(0).numpy() for i in range(len(q)))
        hs[name] = tot / (len(job.seq[name]) * len(q))
    R["head_score"] = hs["rep"] - hs["A"]
    # copying score: o_proj inputs from a hook written here, full-vocabulary logits in float64
    WU = job.model.get_output_embeddings().weight.double().numpy()
    gf = job.tower.norm.weight.double().numpy()
    Z = [[] for _ in layers]
    hk = [l.self_attn.o_proj.register_forward_pre_hook(lambda m, a, i=i: Z[i].append(a[0].double().numpy())) for i, l in enumerate(layers)]
    for b0 in range(0, len(job.seq["rep"]), B):
        NH.forward(job.model, job.x_of(job.seq["rep"][b0:b0 + B]), False)
    for x in hk:
        x.remove()
    tg = job.seq["rep"][:, h + 2:2 * h]                               # row[h + i + 1], i = 1..h-2
    D = job.hd
    R["copy"] = np.zeros((L, job.H))
    for l in range(L):
        z = np.concatenate(Z[l])[:, q]
        WO = layers[l].self_attn.o_proj.weight.double().numpy()
        g = gf * layers[l].post_attention_layernorm.weight.double().numpy()
        for hh in range(job.H):
            lg = ((z[..., hh * D:(hh + 1) * D] @ WO[:, hh * D:(hh + 1) * D].T) * g) @ WU.T
            R["copy"][l, hh] = (np.take_along_axis(lg, tg[..., None], -1)[..., 0] - lg[..., job.copy_rand].mean(-1)).mean()
    # weight side: the full logit vectors in float64
    for k in ("wu_std_ratio", "wu_skew", "wu_kurtosis"):
        R[k] = np.zeros((L, W))
    for l in range(L):
        Wd = layers[l].mlp.down_proj.weight.double().numpy()
        lg = WU @ Wd
        c = lg - lg.mean(0)
        m2 = (c ** 2).mean(0)
        R["wu_std_ratio"][l], R["wu_skew"][l], R["wu_kurtosis"][l] = np.sqrt(m2) / np.linalg.norm(Wd, axis=0), (c ** 3).mean(0) / m2 ** 1.5, (c ** 4).mean(0) / m2 ** 2
    return R


def max_rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "NaN patterns differ"
    m = np.isfinite(a)
    return float(np.max(np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-3)))


def gate_line(name, g) -> str:
    if name == "scale":
        return f"GATE scale: {g['n_read']} scales read, {g['n_excluded']} excluded {g['excluded']}, NaN at excluded {g['nan_at_excluded']}; " \
               f"natural exceedance median {g['scale_known']['median_exceedance']:.4f}, mean {g['scale_known']['mean_exceedance']:.5f} (expected at most " \
               f"{g['scale_known']['expected_at_most']:.3f}); pass {g['pass']}; failing input none"
    if name == "samples":
        return f"GATE samples: lines {g['lines']} want {g['want']} pass {g['pass']}; failing input the file missing (LAYOUT line)"
    if name == "detector":
        f = g["failing_input"]
        return f"GATE detector: top head {g['top_head']} score {g['score']:.3f} E_top {g['E_top']:.4f} (line {g['line']}) pass {g['pass']}; " \
               f"failing input (pass 2 shuffled) score {f['score']:.3f} E_top {f['E_top']:.4f} fails {g['failing_input_failed']}"
    return f"GATE {name}: value {g['value']:.2e} (line {g['tol']}) pass {g['pass']}; failing input ({g['failing_input']['input']}) " \
           f"{g['failing_input']['value']:.2e} fails {g['failing_input_failed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    sc = M.selfcheck()
    print(f"SELFCHECK: mechanics.selfcheck passes {sc}", flush=True)

    # ---- run A: the planted model
    job, undo = run_job(out / "A", planted=True)
    m = json.loads((out / "A/job/metrics.json").read_text())
    g = m["gates"]
    sel = m["selection"]
    print(f"RUN A (planted head {list(PLANT)}, 6-layer bf16 Olmo 3, {DRY_PLAN['n_seq']} rows per set): status {m['status']}, {m['forwards']} forwards, "
          f"{m['tokens_forwarded']} tokens; families {sel['family_sizes']}; lines " + ", ".join(f"{k} {v:.4f}" for k, v in sel["lines"].items()), flush=True)
    for name in VM.GATES:
        print("  " + gate_line(name, g[name]), flush=True)
    for name in ("known_activation", "identity", "ablation_identity", "detector"):
        assert g[name]["pass"] and g[name]["failing_input_failed"], (name, g[name])
    assert g["detector"]["top_head"] == list(PLANT), g["detector"]["top_head"]
    assert g["scale"]["pass"] and g["samples"]["pass"] and g["scale"]["n_excluded"] == 2 and g["scale"]["excluded"] == [list(ZERO), list(NANC)]

    # ---- sampling vocabulary (amendment 5), recounted from the corpus
    cnt = np.bincount(job.tokens, minlength=len(job.tok))
    V0 = M.nonspecial_ids(job.tok)
    top = set(np.argsort(-cnt, kind="stable")[:DRY_PLAN["n_top_excl"]].tolist())
    want_V = np.array([v for v in V0 if cnt[v] > 0 and v not in top])
    allseq = np.concatenate([job.seq[k].ravel() for k in ("rep", "A", "B", "narrow")])
    print(f"SAMPLING: {m['vocab']}; V equals the recount {np.array_equal(job.V, want_V)}; ids drawn outside V: {int((~np.isin(allseq, job.V)).sum())}", flush=True)
    assert np.array_equal(job.V, want_V) and np.isin(allseq, job.V).all() and m["vocab"]["n_zero_count"] > 0 and m["vocab"]["n_top_excluded"] > 0

    # ---- heads (amendment 2)
    hd = m["heads"]
    fmt = lambda xs: ", ".join(f"({x['layer']}, {x['head']}) prefix {x['prefix']:.3f} copy {x['copy']:.2e}" for x in xs[:3])
    with np.load(out / "A/job/heads.npz") as z:
        cmax = np.unravel_index(int(np.argmax(z["copy"])), z["copy"].shape)
    print(f"HEADS: induction {fmt(hd['induction'])} ...; negative copy {fmt(hd['negative_copy'])} ...; {hd['n_positive_copy']} of {job.nL * job.H} "
          f"heads copy positively; largest copying score at {tuple(int(c) for c in cmax)}", flush=True)
    assert tuple(int(c) for c in cmax) == PLANT
    assert (hd["induction"][0]["layer"], hd["induction"][0]["head"]) == PLANT and all(x["copy"] > 0 for x in hd["induction"])
    assert all(x["copy"] < 0 for x in hd["negative_copy"]) and len(hd["induction"]) == len(hd["negative_copy"]) == 8

    # ---- recount
    rc = recount(job)
    with np.load(out / "A/job/signatures.npz") as z:
        sig = {k: z[k] for k in z.files}
    with np.load(out / "A/job/heads.npz") as z:
        hscore = z["score"]
    with np.load(out / "A/job/heads.npz") as z:
        hcopy = z["copy"]
    errs = {k: max_rel(sig[k], rc[k]) for k in ("E", "E_null", "E_early", "E_early_null", "P", "P_null", "T", "U", "T_null", "U_null", "R", "F",
                                                 "wu_std_ratio", "wu_kurtosis")}
    errs["wu_skew (absolute)"] = float(np.nanmax(np.abs(sig["wu_skew"] - rc["wu_skew"])))
    errs["head_score"] = float(np.abs(hscore - rc["head_score"]).max())
    errs["copy"] = float(np.abs(hcopy - rc["copy"]).max() / np.abs(rc["copy"]).max())
    undo()
    print("RECOUNT (float64 from stored activations vs the job's accumulators; max relative difference): "
          + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()), flush=True)
    assert all(v < 1e-4 for v in errs.values()), errs

    # ---- scale exclusion
    fam = json.loads((out / "A/job/families.json").read_text())
    for cell, what in ((ZERO, "q999 (0, 0)"), (NANC, "q999 (nan, x)")):
        z0 = {k: float(sig[k][cell]) for k in ("s", "E", "E_early", "P", "T", "U", "R", "F", "E_abl", "E_rand", "E_neg")}
        in_fam = [f for f, v in fam["families"].items() if list(cell) in v]
        print(f"SCALE: neuron {list(cell)} ({what} in the bank): valid {bool(sig['valid'][cell])}, every signature NaN "
              f"{all(np.isnan(v) for v in z0.values())}, in families {in_fam}", flush=True)
        assert not sig["valid"][cell] and all(np.isnan(v) for v in z0.values()) and not in_fam

    # ---- layout and the verdict's gate readers
    bank = VM.load_bank(out / "A/job")
    rd = {n: VM.read_gate(bank, n) for n in VM.GATES}
    rf = {n: VM.read_gate(bank, n, failing=True) for n in FAILING_NAMES}
    cp = out / "A/job_no_samples"
    shutil.rmtree(cp, ignore_errors=True)
    shutil.copytree(out / "A/job", cp)
    (cp / "samples" / "mechanics.jsonl").unlink()
    gs = VM.gate_samples(cp)
    n_ctx = sum(1 for _ in (out / "A/job/contexts").glob("*.jsonl"))
    print(f"LAYOUT: verdict_mechanics.load_bank reads {len(bank['sig'])} signature arrays, {len(bank['flagged']['layer'])} flagged neurons, "
          f"{len(bank['flags']['pole'])} poles ({n_ctx} context files, repeat flags {int((bank['flags']['repeat'] == 1).sum())} of "
          f"{int((bank['flags']['repeat'] >= 0).sum())}); gate readers {rd}; failing inputs read as passing {rf}; samples gate on a copy without the file: "
          f"{gs}", flush=True)
    assert all(rd.values()) and not any(rf.values()) and not gs["pass"]

    # ---- run B: no planted head
    jobB, _ = run_job(out / "B", planted=False)
    gB = json.loads((out / "B/job/metrics.json").read_text())["gates"]
    print(f"RUN B (no planted head): " + "; ".join(f"{n} pass {gB[n]['pass']}" for n in VM.GATES) + f"; detector top head {gB['detector']['top_head']} "
          f"score {gB['detector']['score']:.4f} (line 0.3)", flush=True)
    assert not gB["detector"]["pass"] and all(gB[n]["pass"] for n in VM.GATES if n != "detector")

    # ---- the real run's size
    p = JS.PLAN
    rows = 7 * p["n_seq"] + p["n_nat"] + 3 * p["n_id"] + 6 + 2 + len(JS.SAMPLE_ROWS) + JS.N_SAMPLE_NAT
    tok = rows * (2 * p["half"] + 1) + 3 * p["known_ctx"] * p["hw"] // 2
    rate = E1_RATE / 2
    gpu_min = tok / rate / 60
    wall = gpu_min * 1.5 + 2 + 3 + 5.5 + 4
    print(f"ESTIMATE: {rows} rows of {2 * p['half'] + 1} tokens + 48 known prefixes = {tok / 1e6:.2f} M tokens; forwards {gpu_min:.1f} min at half E1's "
          f"measured rate ({rate:.0f} tokens/s full-depth, an assumption), x1.5 for eager attention, the accumulators and the copying score; + 2 min weight side (580 TFLOP fp32), 3 min "
          f"contexts, 5.5 min download, corpus and load (E1: 162 s + 125 s), 4 min boot: about {wall:.0f} min end to end, {wall / 60:.2f} GPU-h", flush=True)
    print("MECHANICS_DRYRUN_OK", flush=True)


if __name__ == "__main__":
    main()
