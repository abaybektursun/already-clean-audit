"""Weight-free dry run of INSTRUMENTS E5 (studies/instruments/SPEC_instruments.md, section "E5"). No download and no network: tiny random Olmo 3,
Gemma 3 and Qwen 3 models built from their transformers configs, a byte-level tokenizer and a fake corpus, as the E1 and E1b dry runs build them.

    PYTHONPATH=. <harvest venv python> -m infra.aws_pg.jobs.instruments_dryrun_e5 --out <dir>

Part 1: instruments_e5.selfcheck (the hold, the marked write, the member write on tiny tensors); the real inputs (e1_items.json's 56 pairs, e5_items.json's
o7 items and T5 members) and the run's size: readouts and tokens per step from the job's arm plan, timed at E1's measured L40S rate.
Part 2, t12 end to end (jobs.instruments_e5.T12.run) on a tiny bf16 Olmo 3 (6 layers; E1's dry-run items, 6 pairs; batch 8, so every readout
replays two batches): every pair banked with every arm; the held clean forward equal to the clean forward exactly, the shifted scales not; the inf
fill non-finite; an independent probe sees each held reader's input at the clean rms; readout_marked's down_proj input equal to the clean input
except at (row, marked position, idx); the free arms equal an independent reuse_ablate.readout of the banked sets at the banked ā; 21 sample lines.
Part 3, t4 (jobs.instruments_e5.t4_job = jobs.instruments_e1b.Job with model key o7) on a tiny jittered Olmo 3 (9 layers; E1b's dry-run items),
checked by the E1b dry run's check_tiny. Part 4, t5 (run_t5) on tiny Gemma 3 and Qwen 3 and the tiny t12's Olmo bank; the residual reconstruction
against output_hidden_states on all three architectures.
Part 5: verdict_e5.py on the tiny bank (exit 0, 1 or 4: random weights need not pass the room gates), then a constructed bank (the tiny bank with
planted a_P): exit 0 with the planted readings (T1 refuted, T2 holds, T4 not separated) and the planted numbers; T4's other two readings; exit 5 on a
missing pair; exit 3 with shifted known targets; exit 1 when the shifted scales reproduce the clean a_P; exit 4 with T1 cancelled and T2 read when
the held own fall is 0.1; each --failing-input makes its gate fail (exit 3 or 4) and leaves the others passing. figures_e5.py on the constructed
verdict (PNG and SVG). Prints one line per check and INSTRUMENTS_DRYRUN_E5_OK.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from infra.aws_pg.jobs import instruments_dryrun_e1 as D1
from infra.aws_pg.jobs import instruments_dryrun_e1b as D1B
from infra.aws_pg.jobs import instruments_e1 as JE
from infra.aws_pg.jobs import instruments_e1b as JE1B
from infra.aws_pg.jobs import instruments_e5 as J5
from infra.aws_pg.jobs.neuron_off_dryrun import WINDOW, fake_corpus, reference_a, tiny_model, tiny_tokenizer
from infra.pg import instruments_adapter as AD
from infra.pg import instruments_e5 as E5
from infra.pg import instruments_values as IV
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import unite_route as UR

REPO = pathlib.Path(__file__).resolve().parents[3]
ITEMS1 = REPO / "studies/instruments/inputs/e1_items.json"
ITEMS5 = REPO / "studies/instruments/inputs/e5_items.json"
E1_METRICS = REPO / "studies/instruments/results/e1/1/t15/metrics.json"
E1B_METRICS = {mk: REPO / f"studies/instruments/results/e1b/1/{mk}/metrics.json" for mk in ("g4", "q4b")}
VERDICT = REPO / "studies/instruments/scripts/verdict_e5.py"
FIGURES = REPO / "studies/instruments/scripts/figures_e5.py"
HOLD_OVERHEAD = 1.5                                   # upper bound on a held readout's time over a free one (five extra pre-hooks per layer)
BUDGET_MIN = 120
SPEC_GPU_H = 0.6 + 0.15
TINY5 = {"g4": "gemma3", "q4b": "qwen3"}


def sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------ part 1: the real inputs and the size
def t12_reads(first: bool) -> dict:
    """The t12 arm plan per pair: pass 1 before, identity, 4 free; pass 2 before2, heldclean, heldclean1, shift (inf on the first pair), 4 held,
    4 held1, 4 marked."""
    return {"free": 6 + 1 + 4, "held": 3 + (1 if first else 0) + 8}


def real_inputs() -> dict:
    it1, it5 = json.loads(ITEMS1.read_text()), json.loads(ITEMS5.read_text())
    P = it1["planted"]
    assert len(P) == 56 and all(len(p["J_tok"]) >= 12 and len(p["O_tok"]) >= 12 for p in P), len(P)
    o7 = it5["models"]["o7"]
    assert len(o7["planted"]) == 56 and len(o7["singles"]) == 30 and o7["revision"].startswith("373bad25") and o7["branch"] == "stage1-step1413814", o7["revision"]
    for mk in J5.T5_MODELS:
        t = it5["t5"][mk]
        assert len(t["background"]["positions"]) == 200 and t["members"] and all(l < t["n_layers"] - 1 for l, _, _ in t["members"]), mk
    e1 = json.loads(E1_METRICS.read_text())
    rate = e1["units_tokens_per_second"]
    bos, hw = 100257, UR.HARVEST_WINDOW
    nf = lambda t, length=RA.LENGTH: RA.n_forwarded(np.asarray(t, np.int64)[:J5.CAP] if length else np.asarray(t, np.int64), length, bos, hw)
    reads = {"free": 0, "held": 0}
    tok = {"free": 0, "held": 0}
    for i, p in enumerate(P):
        r = t12_reads(i == 0)
        for k in r:
            reads[k] += r[k]
            tok[k] += r[k] * nf(p["J_tok"])
    cap = sum(nf(p["J_tok"]) + nf(p["O_tok"]) for p in P)
    pos = UR.background_positions(15, 20_570_743 // hw * hw, hw)["e1"]
    fixed = 3 * nf(pos) + sum(nf(np.asarray(x)[:16], None) for x in it1["known"]["tok"])
    t12_s = (tok["free"] + HOLD_OVERHEAD * tok["held"] + cap + fixed) / rate
    t12_wall = t12_s + e1["download_and_corpus_s"] + e1["load_s"] + e1["timing_s"]["known"]
    # t4: E1b's arm plan on the o7 items (upper bound: every pair with members and inverted sets; jobs.instruments_dryrun_e1b.count_reads)
    c = D1B.count_reads(True, True, True)
    nfo = lambda t, length=RA.LENGTH: RA.n_forwarded(np.asarray(t, np.int64), length, o7["bos_id"], o7["window"])
    t4_reads = len(o7["planted"]) * (c["J"] + c["O"]) + len(o7["singles"]) * D1B.SINGLE_READS
    t4_tok = sum(c["J"] * nfo(p["J_tok"]) + c["O"] * nfo(p["O_tok"]) for p in o7["planted"]) + sum(D1B.SINGLE_READS * nfo(s["J_tok"]) for s in o7["singles"])
    t4_cap = (sum(nfo(p["J_tok"]) + nfo(p["O_tok"]) for p in o7["planted"]) + sum(nfo(s["J_tok"]) for s in o7["singles"])
              + 2 * nfo(o7["background"]["positions"]) + sum(nfo(t, None) for t in o7["known"]["tok"]))
    t4_s = (t4_tok + t4_cap) / rate
    t4_wall = t4_s + e1["load_s"] + 30                    # the hub cache from t12: a re-check of the snapshot, then the load
    # t5: two loads and 200 positions each (E1b's measured loads), timed at the Olmo rate (an upper bound for 4B models)
    t5_tok = sum(RA.n_forwarded(np.asarray(it5["t5"][mk]["background"]["positions"]), RA.LENGTH, it5["t5"][mk]["bos_id"], hw) for mk in J5.T5_MODELS)
    t5_load = sum(json.loads(E1B_METRICS[mk].read_text())["load_s"] for mk in J5.T5_MODELS)
    t5_wall = t5_tok / rate + t5_load + 60
    boot = 300                                             # instance start, weight sync of gemma and Qwen from S3, venv
    total = t12_wall + t4_wall + t5_wall + boot
    return {"rate": rate, "t12": {"readouts": reads["free"] + reads["held"], "held_readouts": reads["held"], "forwards": J5.CAP * (reads["free"] + reads["held"]),
                                  "captures": 24 * len(P) + 3 * 200 + 48, "tokens": int(tok["free"] + tok["held"] + cap + fixed), "units_min": t12_s / 60, "wall_min": t12_wall / 60},
            "t4": {"readouts_upper": t4_reads, "forwards_upper": 12 * t4_reads, "tokens": int(t4_tok + t4_cap), "units_min": t4_s / 60, "wall_min": t4_wall / 60},
            "t5": {"forwards": 400, "tokens": int(t5_tok), "wall_min": t5_wall / 60},
            "boot_min": boot / 60, "total_min": total / 60, "gpu_h": total / 3600,
            "members": {mk: len(it5["t5"][mk]["members"]) for mk in J5.T5_MODELS}}


# ------------------------------------------------------------------------------------------ part 2: t12 on a tiny Olmo 3
def part_t12(run: pathlib.Path, out: pathlib.Path) -> tuple:
    tok = tiny_tokenizer(bos=True)
    tokens = fake_corpus(tok, D1.N_WINDOWS + 2)
    offsets = np.arange(0, tokens.size + 37, 37)
    offsets[-1] = tokens.size
    corpus = {"tokens": tokens, "doc_offsets": offsets}
    model = tiny_model("olmo3", seed=0, n_layers=D1.N_LAYERS).requires_grad_(False)
    other = tiny_model("olmo3", seed=1, n_layers=D1.N_LAYERS).requires_grad_(False)
    ref, ref2 = (reference_a(x, tok, tokens, list(range(D1.N_WINDOWS))) for x in (model, other))
    items = D1.tiny_items(ref, ref2, model, tok, tokens)
    ip = out / "e1_items_tiny.json"
    NH.dump(ip, items)
    d = run / "t12"
    a = J5.parser().parse_args(["--out", str(d), "--stage", "t12", "--items", str(ip), "--corpus", "tiny", "--t", "T_15", "--revision", "stage1-step1413814",
                                "--harvest-window", str(WINDOW), "--scan-tokens", str(D1.N_WINDOWS * WINDOW), "--batch", "8", "--full-batch", "4"])
    a.items_sha256 = sha(ip)
    job = J5.T12(d, a, items, model, tok, corpus, "cpu")
    job.run()
    return ip, items, check_t12(d, items, job, model, tok, tokens)


def check_t12(d: pathlib.Path, items: dict, job, model, tok, tokens) -> dict:
    m = json.loads((d / "metrics.json").read_text())
    g = m["gates"]
    assert m["status"] == "done" and g["held_identity"]["value"] == 0.0 and g["held_identity"]["failing_input_failed"], g["held_identity"]
    assert g["nonfinite"]["value"] == 0.0 and g["nonfinite"]["failing_input"]["value"] == 1.0, g["nonfinite"]
    assert g["t5_capture"]["pass"] and g["t5_capture"]["failing_input_failed"] and g["samples"]["pass"], (g["t5_capture"], g["samples"])
    assert m["pairs"][items["planted"][0]["unit"]]["nonfinite"]["heldclean_inf"]["by_layer"], "the inf fill counted no layer"
    lines = [json.loads(x) for x in (d / "samples" / "e5.jsonl").read_text().splitlines()]
    assert len(lines) == g["samples"]["want"] and all(x["contexts"][0]["marked_token"] for x in lines), len(lines)
    assert all("held_rms_clean_ablated" in x["contexts"][0] for x in lines if x["test"] == "T1"), "a T1 sample lacks the held scales"
    kw = dict(tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=8, hw=WINDOW, device="cpu")
    vz = np.load(d / "values.npz")
    ab = torch.tensor(vz["abar"])
    layers = list(NH.text_tower(model).layers)
    n_arms, probe_err, marked_changed = 0, 0.0, 0
    for i, p in enumerate(items["planted"]):
        z = np.load(d / "e5" / "pairs" / f"{p['unit']}.npz")
        names = [str(x) for x in z["J_arms"]]
        assert set(names) == J5_arms(i == 0), (p["unit"], sorted(set(names) ^ J5_arms(i == 0)))
        n_arms += len(names)
        unit = UR.planted_unit(p)
        L = unit["layer"]
        S = {x: [(int(l), int(k)) for l, k in zip(z[f"set_{x}_layer"], z[f"set_{x}_neuron"])] for x in J5.SETS}
        assert np.array_equal(z["J_a_before"], z["J_a_before2"]), (p["unit"], "pass 2's clean a_P differs from pass 1's")
        for x in ("own_u", "cross_u"):
            want = RA.readout(model, z["J_tok"], unit, IV.written(S[x], "mean", {"mean": ab}, L, "cpu"), **kw)["a"]
            assert np.array_equal(want, z["J_a_after"][names.index(f"free__{x}")]), (p["unit"], x, "the free arm is not an independent readout of the banked set")
        assert not np.array_equal(z["J_a_after"][names.index("marked__own_u")], z["J_a_after"][names.index("free__own_u")]), "the marked write equals the free write"
        if i == 0:
            # an independent probe: under the hold each reader's input has the clean rms at every window position
            toks = z["J_tok"]
            seen = {}
            with E5.held_input(layers, range(L + 1), 1) as H:
                hs = [E5.reader(layers[l], r).register_forward_pre_hook(lambda mm, aa, key=(l, r): seen.setdefault(key, []).append(aa[0].clone()))
                      for l in range(L + 1) for r in E5.NAMES]
                H.record()
                RA.readout(model, toks, unit, None, **kw)
                clean = {k: [E5.rms(x) for x in v] for k, v in seen.items()}
                seen.clear()
                H.hold(E5.held_layers(S["cross_u"], L))
                RA.readout(model, toks, unit, IV.written(S["cross_u"], "mean", {"mean": ab}, L, "cpu"), **kw)
                H.done()
                for h in hs:
                    h.remove()
            for l in E5.held_layers(S["cross_u"], L):
                for r in E5.NAMES:
                    for c, x in zip(clean[(l, r)], seen[(l, r)]):
                        probe_err = max(probe_err, float(((E5.rms(x)[:, 1:] - c[:, 1:]).abs() / c[:, 1:]).max()))
            assert 0 < probe_err <= 2 ** -7, (probe_err, "a held reader's input is not at the clean rms")
            # readout_marked: down_proj's input equals the clean input except at (row, marked position, idx)
            dn = {}
            l0 = min(l for l, _ in S["cross_u"])                # the lowest written layer: its MLP input is the clean one everywhere
            abl = IV.written(S["cross_u"], "mean", {"mean": ab}, L, "cpu")
            h = layers[l0].mlp.down_proj.register_forward_hook(lambda mm, aa, o: dn.setdefault("x", []).append(aa[0].clone()))
            RA.readout(model, toks, unit, None, **kw)
            clean_in = dn.pop("x")
            E5.readout_marked(model, toks, unit, abl, **kw)
            marked_in = dn.pop("x")
            h.remove()
            idx = abl[l0][0]
            order = np.argsort(np.asarray(toks) % WINDOW, kind="stable")
            for b, (c, mk_) in enumerate(zip(clean_in, marked_in)):
                _, pos = UR.pack(tokens, np.asarray(toks)[order[b * 8:(b + 1) * 8]], RA.LENGTH, tok.bos_token_id, WINDOW, "cpu")
                diff = (c != mk_)
                rows = torch.arange(len(pos))
                diff[rows[:, None], pos[:, None], idx[None, :]] = False
                assert not diff.any(), "the marked write changed an entry away from (row, marked position, idx)"
                marked_changed += int((c[rows[:, None], pos[:, None], idx[None, :]] != mk_[rows[:, None], pos[:, None], idx[None, :]]).sum())
            assert marked_changed > 0, "the marked write changed nothing"
    return {"m": m, "n_arms": n_arms, "probe_err": probe_err, "marked_changed": marked_changed}


def J5_arms(first: bool) -> set:
    return ({"identity", "heldclean", "heldclean1", "heldclean_shift"} | {f"{c}__{x}" for c in ("free", "held", "held1", "marked") for x in J5.SETS}
            | ({"heldclean_inf"} if first else set()))


# ------------------------------------------------------------------------------------------ part 3: t4 on a tiny Olmo 3 (E1b's job, model key o7)
def part_t4(run: pathlib.Path) -> tuple:
    tok = tiny_tokenizer(bos=True)
    tokens = fake_corpus(tok, D1B.N_WINDOWS + 2)
    offsets = np.arange(0, tokens.size + 37, 37)
    offsets[-1] = tokens.size
    corpus = {"tokens": tokens, "doc_offsets": offsets}
    model = D1B.jitter(tiny_model("olmo3", seed=0, n_layers=D1B.N_LAYERS).requires_grad_(False), 1)
    ref = reference_a(model, tok, tokens, list(range(D1B.N_WINDOWS)))
    rec = D1B.tiny_items("o7", "olmo3", ref, model, tok)
    rec["branch"] = "tiny"
    return rec, (model, tok, tokens, corpus)


def run_t4(run: pathlib.Path, rec: dict, ip5: pathlib.Path, built) -> dict:
    model, tok, tokens, corpus = built
    d = run / "t4"
    J5.t4_job(d, rec, sha(ip5), model, tok, corpus, "cpu", 8, 4).run()
    return D1B.check_tiny(d, "o7", rec, model, tok, tokens)


# ------------------------------------------------------------------------------------------ part 4: t5 on tiny Gemma 3 and Qwen 3
def t5_tiny() -> tuple[dict, dict]:
    recs, built = {}, {}
    for mk, arch in TINY5.items():
        tok = tiny_tokenizer(bos=arch == "gemma3")
        tokens = fake_corpus(tok, D1B.N_WINDOWS + 2)
        offsets = np.arange(0, tokens.size + 37, 37)
        offsets[-1] = tokens.size
        model = D1B.jitter(tiny_model(arch, seed=0, n_layers=D1B.N_LAYERS).requires_grad_(False), 1)
        ad = AD.Adapter(model, D1B.N_LAYERS)
        pos = UR.background_positions(-1000, D1B.N_WINDOWS * WINDOW, WINDOW)["e1"]
        kw = dict(tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=8, hw=WINDOW, device="cpu")
        ab = AD.background_mean_a(ad, model, pos, D1B.N_LAYERS - 1, **kw)
        mem = RB.high_background(ab, D1B.N_LAYERS - 1)["members"][:12]
        recs[mk] = {"model_key": mk, "repo": f"tiny/{arch}", "revision": "tiny", "arch": arch, "nh_key": "tiny", "tokname": "tiny", "n_layers": D1B.N_LAYERS,
                    "width": D1B.W_TINY, "hidden": ad.hidden, "bos_id": tok.bos_token_id, "window": WINDOW, "tokens_sha256": "tiny",
                    "background": {"positions": pos.tolist(), "seed": 0}, "members": [[int(l), int(k), float(a)] for l, k, a in mem]}
        built[mk] = (model, tok, {"tokens": tokens, "doc_offsets": offsets})
    return recs, built


def residual_check() -> dict:
    """E5.residual_rms from instruments_adapter.capture against output_hidden_states at the marked token, on the three tiny architectures."""
    out = {}
    for arch in ("olmo3", "gemma3", "qwen3"):
        tok = tiny_tokenizer(bos=arch != "qwen3")
        tokens = fake_corpus(tok, 20)
        model = D1B.jitter(tiny_model(arch, seed=0, n_layers=D1B.N_LAYERS).requires_grad_(False), 1)
        ad = AD.Adapter(model, D1B.N_LAYERS)
        toks = np.array([w * WINDOW + 30 + 7 * w for w in range(10)])
        kw = dict(tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=4, hw=WINDOW, device="cpu")
        cap = AD.capture(ad, model, toks, D1B.N_LAYERS - 2, set(), **kw)
        rr = E5.residual_rms(cap, D1B.N_LAYERS - 2)
        x, pos = UR.pack(tokens, toks, RA.LENGTH, tok.bos_token_id, WINDOW, "cpu")
        with torch.no_grad():
            hs = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, output_hidden_states=True).hidden_states
        rows = torch.arange(len(toks))
        want = torch.stack([E5.rms(hs[l + 1][rows, pos]) for l in range(D1B.N_LAYERS - 1)], 1)
        e = float(((rr - want).abs() / want).max())
        wrong = float(((E5.residual_rms({**cap, "mlp": {l: 0 * v for l, v in cap["mlp"].items()}}, D1B.N_LAYERS - 2) - want).abs() / want).max())
        assert e <= 2e-2 < wrong, (arch, e, wrong, "the residual reconstruction does not match the model's hidden states")
        out[arch] = {"err": e, "without_mlp": wrong}
    return out


def part_t5(run: pathlib.Path, recs: dict, built: dict, ip5: pathlib.Path) -> dict:
    load = lambda mk, rec: built[mk]
    m = J5.run_t5(run / "t5", sha(ip5), recs, "t12", load, "cpu", 8, {"device_name": "cpu"})
    for mk in ("o7",) + J5.T5_MODELS:
        z = np.load(run / "t5" / "t5" / f"{mk}.npz")
        assert z["write"].shape == (200, len(z["members_layer"])) and np.isfinite(z["write"]).all() and (z["write"] > 0).all(), mk
    for mk in J5.T5_MODELS:
        model, tok, corpus = built[mk]
        ad = AD.Adapter(model, D1B.N_LAYERS)
        z = np.load(run / "t5" / "t5" / f"{mk}.npz")
        for j in range(min(3, len(z["members_layer"]))):
            l, k = int(z["members_layer"][j]), int(z["members_neuron"][j])
            d = float((ad.post_gain(l) * ad.layers[l].mlp.down_proj.weight[:, k].float()).norm())
            assert abs(d - float(z["dnorm"][j])) <= 1e-5 * d, (mk, l, k, d, float(z["dnorm"][j]))
    return m


# ------------------------------------------------------------------------------------------ part 5: the verdict, a constructed bank
def fake_harvest_o7(root: pathlib.Path) -> pathlib.Path:
    """stats/layer_<l>.npz for the tiny o7 model in the E1b dry run's construction (even neurons dense, odd sparse)."""
    d = root / "stats"
    d.mkdir(parents=True, exist_ok=True)
    edges = NH.hist_edges().astype(np.float32)
    for l in range(D1B.N_LAYERS):
        h = np.zeros((D1B.W_TINY, 2, 400), np.int32)
        h[0::2, 0, 200:280] = 100
        h[1::2, 0, 80:100] = 4950
        h[1::2, 0, 200:280] = 12
        np.savez_compressed(d / f"layer_{l:02d}.npz", hist=h, hist_edges=edges, density=h.sum(-1).astype(np.int64), q999=np.ones((D1B.W_TINY, 2), np.float32),
                            n_pole_tokens=h.sum(-1).astype(np.int64))
    return root


def verdict(run, ip1, ip5, vout, hroot, targets, *extra) -> tuple:
    cmd = [sys.executable, str(VERDICT), "--run", str(run), "--items-e1", str(ip1), "--items-e5", str(ip5), "--out", str(vout), "--harvest-o7", str(hroot),
           "--known-targets", ",".join(f"{x:.12g}" for x in targets), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 5 or f"VERDICT_E5 exit {r.returncode}" in r.stdout, (r.returncode, r.stdout[-2000:], r.stderr[-3000:], "the verdict crashed")
    return r.returncode, r.stdout, r.stderr


def recount_known(d: pathlib.Path, items: dict) -> tuple:
    f = lambda arm: float(np.median([RA.job_fall(z["J_a_before"], z["J_a_after"][[str(x) for x in z["J_arms"]].index(arm)])["mean"]
                                     for z in (np.load(d / "e5" / "pairs" / f"{p['unit']}.npz") for p in items["planted"])]))
    return f("free__cross_u"), f("free__own_u")


PLAN12 = {"free": {"own_u": 0.5, "cross_u": (0.08, 0.002), "own_e": 0.45, "cross_e": (0.03, 0.001)},
          "held": {"own_u": 0.4, "cross_u": (0.06, 0.002), "own_e": 0.38, "cross_e": (0.02, 0.001)},
          "held1": {"own_u": 0.42, "cross_u": (0.07, 0.002), "own_e": 0.4, "cross_e": (0.025, 0.001)},
          "marked": {"own_u": 0.25, "cross_u": (0.01, 0.001), "own_e": 0.22, "cross_e": (0.005, 0.001)}}
EXTRA12 = {"identity": 0.0, "heldclean": 0.0, "heldclean1": 0.0, "heldclean_shift": 0.01, "heldclean_inf": float("nan")}
PLAN4 = {"mean": (0.08, 0.04), "zero": (0.07, 0.03), "median": (0.075, 0.035), "res": (0.09, 0.05)}


def fall12(plan, extra, i: int, arm: str) -> float:
    if arm in extra:
        return extra[arm]
    c, x = arm.split("__")
    v = plan[c][x]
    return v[0] + v[1] * i if isinstance(v, tuple) else v


def synth(dst: pathlib.Path, src: pathlib.Path, it1: dict, it5: dict, plan=PLAN12, extra=EXTRA12, plan4=PLAN4) -> pathlib.Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    d = dst / "t12"
    m = json.loads((d / "metrics.json").read_text())
    for i, p in enumerate(it1["planted"]):
        f = d / "e5" / "pairs" / f"{p['unit']}.npz"
        z = dict(np.load(f))
        b = (1.0 + 0.05 * np.arange(len(z["J_tok"]))).astype(np.float32)
        z["J_a_before"] = z["J_a_before2"] = b
        names = [str(x) for x in z["J_arms"]]
        z["J_a_after"] = np.stack([(b * (1 - fall12(plan, extra, i, n))).astype(np.float32) for n in names])
        np.savez_compressed(f, **z)
        r = m["pairs"][p["unit"]]
        r["falls"] = {n: {k: RA.job_fall(b, z["J_a_after"][j])[k] for k in ("mean", "median", "n_pos")} for j, n in enumerate(names)}
        r["identity"].update({x: J5.rel(z["J_a_after"][names.index(x)], b) for x in ("heldclean", "heldclean1", "heldclean_shift")})
    P = m["pairs"].values()
    m["gates"]["held_identity"]["value"] = max(max(v["identity"]["heldclean"], v["identity"]["heldclean1"]) for v in P)
    m["gates"]["held_identity"]["failing_input"]["value"] = max(v["identity"]["heldclean_shift"] for v in P)
    (d / "metrics.json").write_text(json.dumps(m))
    # t4: E1b's bank for o7 with planted falls (jobs.instruments_dryrun_e1b's planted_fall and singles), ā with a high-background tail
    d4 = dst / "t4"
    m4 = json.loads((d4 / "metrics.json").read_text())
    rec = it5["models"]["o7"]
    for i, p in enumerate(rec["planted"]):
        f = d4 / "e1b" / "o7" / "pairs" / f"{p['unit']}.npz"
        z = dict(np.load(f))
        for s in ("J", "O"):
            b = (1.0 + 0.05 * np.arange(len(z[f"{s}_tok"]))).astype(np.float32)
            z[f"{s}_a_before"] = b
            z[f"{s}_a_after"] = np.stack([(b * (1 - D1B.planted_fall(plan4, i, s, str(arm)))).astype(np.float32) for arm in z[f"{s}_arms"]])
        np.savez_compressed(f, **z)
        m4["pairs"][p["unit"]]["sides"] = JE.record_of(z, ("J", "O"))
    for s in rec["singles"]:
        f = d4 / "e1b" / "o7" / "singles" / f"{s['pole']}.npz"
        z = dict(np.load(f))
        b = (1.0 + 0.05 * np.arange(len(z["J_tok"]))).astype(np.float32)
        z["J_a_before"] = b
        z["J_a_after"] = np.stack([(b * (1 - D1B.SINGLE[str(arm)])).astype(np.float32) for arm in z["J_arms"]])
        np.savez_compressed(f, **z)
        m4["singles"][s["pole"]]["falls"] = {str(n): {k: RA.job_fall(b, z["J_a_after"][j])[k] for k in ("mean", "median", "n_pos")} for j, n in enumerate(z["J_arms"])}
    (d4 / "metrics.json").write_text(json.dumps(m4))
    return dst


def expected4(rec: dict, plan4: dict) -> float:
    ps = rec["planted"]
    off = np.mean([min(0.004 * i for i, p in enumerate(ps) if p["tercile"] == t) for t in (0, 1, 2)])
    return plan4["mean"][0] + off


def part_verdict(out: pathlib.Path, run: pathlib.Path, ip1: pathlib.Path, ip5: pathlib.Path, it1: dict, it5: dict) -> dict:
    hroot = fake_harvest_o7(out / "fake_harvest_o7")
    kt = recount_known(run / "t12", it1)
    code, so, se = verdict(run, ip1, ip5, out / "tiny" / "verdict", hroot, kt)
    assert code in (0, 1, 4) and "E5_BANK_INCOMPLETE" not in se, (code, so[-3000:], se[-3000:])
    tiny_lines = [x for x in so.splitlines() if x.startswith(("GATE", "RULE", "REPORTED", "VERDICT"))]
    sd = synth(out / "synth" / "run", run, it1, it5)
    kt_s = recount_known(sd / "t12", it1)
    vout = out / "synth" / "verdict"
    code, so, se = verdict(sd, ip1, ip5, vout, hroot, kt_s)
    lines = [x for x in so.splitlines() if x.startswith(("GATE", "RULE", "REPORTED", "VERDICT"))]
    assert code == 0, (code, so[-4000:], se[-3000:])
    v = json.loads((vout / "verdict_e5.json").read_text())
    R = v["readings"]
    n = len(it1["planted"])
    med = lambda base, step: float(np.median([base + step * i for i in range(n)]))
    close = lambda a, b: abs(a - b) < 1e-6
    assert close(R["T1"]["C"]["value"], med(0.06, 0.002)) and close(R["T2"]["C"]["value"], med(0.01, 0.001)), (R["T1"]["C"], R["T2"]["C"])
    assert close(R["T4"]["N"]["value"], expected4(it5["models"]["o7"], PLAN4)), (R["T4"]["N"], expected4(it5["models"]["o7"], PLAN4))
    assert [R[t]["reading"] for t in ("T1", "T2", "T4")] == ["cause 1 refuted", "the channel is the other positions (holds)", "not separated"], [R[t]["reading"] for t in R]
    md = v["reported"]["t12"]["medians"]
    assert close(md["held"]["own_u"]["value"], 0.4) and close(md["marked"]["own_u"]["value"], 0.25) and close(md["free"]["cross_u"]["value"], med(0.08, 0.002))
    assert close(v["reported"]["t12"]["paired"]["free_minus_held_cross_u"]["value"], 0.02), v["reported"]["t12"]["paired"]
    exits = {"green": code}
    readings = {"green": [R[t]["reading"] for t in ("T1", "T2", "T4", "T5")]}
    for tag, p4, want in (("t4_high", {**PLAN4, "mean": (0.2, 0.05)}, "the construction is not the difference"), ("t4_low", {**PLAN4, "mean": (0.01, 0.0)}, "the construction is the difference")):
        s4 = synth(out / f"synth_{tag}" / "run", run, it1, it5, plan4=p4)
        c, so4, _ = verdict(s4, ip1, ip5, out / f"synth_{tag}" / "verdict", hroot, kt_s)
        r4 = json.loads((out / f"synth_{tag}" / "verdict" / "verdict_e5.json").read_text())["readings"]["T4"]
        assert c == 0 and r4["reading"] == want and close(r4["N"]["value"], expected4(it5["models"]["o7"], p4)), (tag, c, r4)
        exits[tag], readings[tag] = c, r4["reading"]
    # T1 at the other two lines
    for tag, h, want in (("t1_low", (0.005, 0.001), "cause 1 holds"), ("t1_mid", (0.03, 0.002), "not separated")):
        s1 = synth(out / f"synth_{tag}" / "run", run, it1, it5, plan={**PLAN12, "held": {**PLAN12["held"], "cross_u": h}})
        c, _, _ = verdict(s1, ip1, ip5, out / f"synth_{tag}" / "verdict", hroot, kt_s)
        r1 = json.loads((out / f"synth_{tag}" / "verdict" / "verdict_e5.json").read_text())["readings"]["T1"]
        assert c == 0 and r1["reading"] == want, (tag, c, r1)
        exits[tag], readings[tag] = c, r1["reading"]
    inc = synth(out / "synth_incomplete" / "run", run, it1, it5)
    (inc / "t12" / "e5" / "pairs" / f"{it1['planted'][-1]['unit']}.npz").unlink()
    c5, _, e5 = verdict(inc, ip1, ip5, out / "synth_incomplete" / "verdict", hroot, kt_s)
    assert c5 == 5 and "E5_BANK_INCOMPLETE" in e5 and not (out / "synth_incomplete" / "verdict").exists(), (c5, e5[-2000:])
    exits["incomplete"] = c5
    c3, so3, _ = verdict(sd, ip1, ip5, out / "synth" / "verdict_known_off", hroot, (kt_s[0] + 0.02, kt_s[1]))
    assert c3 == 3, (c3, so3[-2000:])
    exits["known_off"] = c3
    nf = synth(out / "synth_nofail" / "run", run, it1, it5, extra={**EXTRA12, "heldclean_shift": 0.0})
    c1, so1, _ = verdict(nf, ip1, ip5, out / "synth_nofail" / "verdict", hroot, kt_s)
    assert c1 == 1 and "t12.held_identity" in so1, (c1, so1[-2000:])
    exits["failing_input_does_not_fail"] = c1
    rt = synth(out / "synth_room" / "run", run, it1, it5, plan={**PLAN12, "held": {**PLAN12["held"], "own_u": 0.1}})
    c4, so4, _ = verdict(rt, ip1, ip5, out / "synth_room" / "verdict", hroot, kt_s)
    vr = json.loads((out / "synth_room" / "verdict" / "verdict_e5.json").read_text())
    assert c4 == 4 and vr["readings"]["T1"]["reading"].startswith("cancelled") and vr["readings"]["T2"]["reading"] == "the channel is the other positions (holds)", (c4, so4[-2000:])
    exits["room_T1_0.1"] = c4
    fails = {}
    for fi, (grp, target) in sorted(VFAIL().items()):
        cf, sof, sef = verdict(sd, ip1, ip5, vout, hroot, kt_s, "--failing-input", fi)
        f = json.loads((vout / f"failing_input_{fi}" / "verdict_e5.json").read_text())
        want = 3 if (grp, target) in VKNOWN() else 4
        assert cf == want and not f["gates"][grp][target]["pass"] and all(r["reading"].startswith("cancelled") for r in f["readings"].values()), (fi, cf, sof[-2000:], sef[-2000:])
        others = {f"{g_}.{k}": x["pass"] for g_, gs in f["gates"].items() for k, x in gs.items() if (g_, k) != (grp, target)}
        assert all(others.values()), (fi, others)
        fails[fi] = [x for x in sof.splitlines() if x.startswith("FAILING_INPUT")][0]
        exits[fi] = cf
    fig = subprocess.run(["uv", "run", "--no-project", "--with", "matplotlib", "--with", "numpy", "python", str(FIGURES), "--verdict", str(vout / "verdict_e5.json"),
                          "--out", str(out / "figures")], capture_output=True, text=True)
    assert fig.returncode == 0 and (out / "figures" / "fig_e5.png").exists() and (out / "figures" / "fig_e5.svg").exists(), (fig.stdout[-2000:], fig.stderr[-3000:])
    return {"tiny_lines": tiny_lines, "lines": lines, "exits": exits, "readings": readings, "fails": fails, "figure": fig.stdout.strip().splitlines()[-1]}


def VFAIL() -> dict:
    sys.path.insert(0, str(REPO / "studies/instruments/scripts"))
    import verdict_e5 as V5
    return V5.FAILING


def VKNOWN() -> set:
    import verdict_e5 as V5
    return V5.KNOWN_GATES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    t0 = time.time()
    print("SELFCHECK instruments_e5:", json.dumps(E5.selfcheck(), ensure_ascii=False))
    e = real_inputs()
    print(f"REAL INPUTS: E1's 56 pairs; e5_items o7 56 pairs (18/20/18) and 30 singles at revision 373bad25 (stage1-step1413814); T5 members g4 "
          f"{e['members']['g4']}, q4b {e['members']['q4b']}")
    print(f"SIZE t12: {e['t12']['readouts']} readouts ({e['t12']['held_readouts']} under the hold or marked) = {e['t12']['forwards']} forwards + {e['t12']['captures']} "
          f"captures, {e['t12']['tokens'] / 1e6:.1f} M tokens; units {e['t12']['units_min']:.1f} min (held readouts x {HOLD_OVERHEAD}), wall {e['t12']['wall_min']:.1f} min")
    print(f"SIZE t4: at most {e['t4']['readouts_upper']} readouts = {e['t4']['forwards_upper']} forwards, {e['t4']['tokens'] / 1e6:.1f} M tokens; units "
          f"{e['t4']['units_min']:.1f} min, wall {e['t4']['wall_min']:.1f} min")
    print(f"SIZE t5: {e['t5']['forwards']} forwards, {e['t5']['tokens'] / 1e6:.2f} M tokens, two loads; wall {e['t5']['wall_min']:.1f} min")
    print(f"ESTIMATE: at E1's measured L40S rate {e['rate']:.0f} tokens/s, with {e['boot_min']:.0f} min of instance start and weight sync: {e['total_min']:.0f} min, "
          f"{e['gpu_h']:.2f} GPU-h (spec {SPEC_GPU_H:.2f}; budget {BUDGET_MIN} min)")
    run = out / "tiny" / "run"
    ip1, it1, t = part_t12(run, out)
    mt = t["m"]
    print(f"TINY t12 (6-layer bf16 Olmo 3, batch 8): {len(it1['planted'])} pairs, {t['n_arms']} arms banked, readouts {mt['readouts']}; held clean = clean "
          f"{mt['gates']['held_identity']['value']} (shifted scales {mt['gates']['held_identity']['failing_input']['value']:.2e}); inf fill non-finite "
          f"{mt['gates']['nonfinite']['failing_input']['value']}; held readers' input rms = clean rms to {t['probe_err']:.1e} (probe); the marked write changed "
          f"{t['marked_changed']} entries, all at (row, marked position, idx); free arms = an independent readout of the banked sets; pass 2's clean a_P = "
          f"pass 1's; T5 capture mean = ā to {mt['gates']['t5_capture']['max_abs']:.1e}; {mt['gates']['samples']['lines']} sample lines")
    rec, built4 = part_t4(run)
    recs5, built5 = t5_tiny()
    ip5 = out / "e5_items_tiny.json"
    NH.dump(ip5, {"models": {"o7": rec}, "t5": recs5})
    it5 = json.loads(ip5.read_text())
    t4 = run_t4(run, rec, ip5, built4)
    g4 = t4["m"]["gates"]
    print(f"TINY t4 (E1b's job, model key o7, 9-layer jittered Olmo 3): readouts {t4['m']['readouts']} + {t4['m']['readouts_cached']} cached = the arm plan "
          f"{t4['asked']}; identity {g4['identity']['value']}, resample known {g4['resample_known']['value']}, known ratio {g4['known_activation']['ratio']:.1e}, "
          f"completeness {g4['completeness']['median_err_over_pole_max']:.1e}; {t4['n_mem']} pairs with a member; E1b's check_tiny passes")
    m5 = part_t5(run, recs5, built5, ip5)
    rc = residual_check()
    print("TINY t5: " + "; ".join(f"{mk} {x['summary']['n_members']} members, median write {x['summary']['median']:.3f}, ā check {x['check']['max_abs']:.1e}"
                                    for mk, x in m5["models"].items()) + "; ||d_k|| = the weights' column norm; residual rms vs output_hidden_states "
          + ", ".join(f"{k} {v['err']:.1e} (without the MLP terms {v['without_mlp']:.2f})" for k, v in rc.items()))
    s = part_verdict(out, run, ip1, ip5, it1, it5)
    for x in s["tiny_lines"]:
        print("  tiny", x)
    print("CONSTRUCTED BANK (planted a_P), verdict lines:")
    for x in s["lines"]:
        print("  ", x)
    print("planted answers reproduced to 1e-6: C_T1, C_T2, N_T4, the own falls, free - held; readings " + json.dumps(s["readings"]))
    for x in s["fails"].values():
        print("  ", x)
    print("EXITS:", json.dumps(s["exits"]))
    print("FIGURE:", s["figure"])
    (out / "metrics.json").write_text(json.dumps({"real_inputs": e, "synth_exits": s["exits"], "readings": s["readings"], "failing_inputs": s["fails"],
                                                  "residual_check": rc, "seconds": round(time.time() - t0, 1)}, indent=1, default=str))
    print("INSTRUMENTS_DRYRUN_E5_OK")


if __name__ == "__main__":
    main()
