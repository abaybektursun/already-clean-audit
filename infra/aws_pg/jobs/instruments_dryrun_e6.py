"""Weight-free dry run of INSTRUMENTS E6 (studies/instruments/SPEC_instruments.md, section "E6"). No download and no network: tiny random OLMo 2 and
Olmo 3 models built from their transformers configs, a byte-level tokenizer and a fake corpus, as the E1, E1b and E5 dry runs build them.

    PYTHONPATH=. <harvest venv python> -m infra.aws_pg.jobs.instruments_dryrun_e6 --out <dir>

Part 1, the olmo2 adapter entry: on a tiny OLMo 2 (9 layers, jittered norms) instruments_adapter.Adapter reads arch olmo2 with olmo3's hook points;
olmo_equality (the adapter path equal to unite_route's, exactly: columns, capture, loads, direct_A, c, completeness, ā, the background capture); the
residual reconstruction emb + sum(attention + MLP terms) against output_hidden_states (the MLP and the attention read the raw residual); failing
inputs: a config declaring a sliding window, and a decoder layer carrying an input_layernorm, each stop the adapter.
Part 2, the harvest step: the manifest's harvest args parsed by jobs.neurons_harvest.parser and compared with the q4b harvest's banked args (only
the model key, the repo, the revision, the tokenizer name and the corpus path differ: the S3 copy of the q4b run's corpus step, as the o7_t15
harvest read it); the model key O2 -> olmo2 = matrix_models.KNOWN's and the facts file's model
type, the revision = KNOWN's sha, the tokenizer olmo3 (the corpus tokenization OLMo-2 shares); neurons_harvest.load_model's config check on a saved
tiny OLMo 2 (O2 passes, O7 fails); neurons_harvest.run on the tiny OLMo 2 (G0 passes, its two failing inputs fail); the items and e1b steps' args
parsed by jobs.instruments_e6b.parser.
Part 3, the items step: instruments_e6b.build equals build_items_e1b.build exactly on the g4, q4b and o7_t15 harvest banks, and its o7_t15 record
equals e5_items.json's o7 record (the code-level known number of the items step).
Part 4, the e1b step on the tiny OLMo 2: jobs.instruments_e1b.Job (model key o2) checked by the E1b dry run's check_tiny, then instruments_e6b.t5_stage
(||d_k|| against the weights, the capture check); verdict_e6 --part b on that bank (exit 0, 1 or 4), then constructed banks with planted falls: N_O2
planted high, low and between read "the term is in OLMo-2 too", "specific to Olmo-3's run" and "not separated" at the planted values (exit 0); exit
5 on a missing pair; every part b --failing-input makes its gate fail (exit 3 or 4) and leaves the others passing.
Part 5, E6c: the job's KNOWN_CASES equal E1's tracked metrics.json values at the real items' positions 0, n // 3, 2n // 3, and its steps file
reads E1's items, corpus and revision; on a tiny Olmo 3 (E1's dry-run items, 6 cases) E1's job on the same model gives the reference bank; the E6c job's known cases read
the same free arms bitwise and the known gate is 0; the marked arms equal an independent readout_marked; identity 0; the marked-write probe passes and
its failing input fails; verdict_e6 --part c on the tiny bank (exit 0 or 4), then constructed banks with planted I_marked of 0.25, 0.05 and 0.01
read "survives", "does not separate" and "the other-positions channel" (exit 0); exit 5 on a missing case; exit 3 with a shifted anchor; every part
c --failing-input fails its gate.
Part 6, E6a on a fabricated T5 bank: Olmo's members planted at 2.0 times gemma's top 34 and 1.0 times Qwen's (low spread) read "Olmo's units write
more" and "they do not", the positive control fires, each failing input fails, a missing bank exits 5.
Part 7: figures_e6.py on the constructed verdicts. Part 8: the size of E6b and E6c from banked timings. Prints one line per check and
INSTRUMENTS_DRYRUN_E6_OK.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from infra.aws_pg.jobs import instruments_dryrun_e1 as D1
from infra.aws_pg.jobs import instruments_dryrun_e1b as D1B
from infra.aws_pg.jobs import instruments_dryrun_e5 as D5
from infra.aws_pg.jobs import instruments_e1 as JE
from infra.aws_pg.jobs import instruments_e1b as JE1B
from infra.aws_pg.jobs import instruments_e6b as J6B
from infra.aws_pg.jobs import instruments_e6c as J6C
from infra.aws_pg.jobs import neurons_harvest as NHJ
from infra.aws_pg.jobs.neuron_off_dryrun import VOCAB, WINDOW, fake_corpus, reference_a, tiny_model, tiny_tokenizer
from infra.pg import instruments_adapter as AD
from infra.pg import instruments_e5 as E5
from infra.pg import instruments_values as IV
from infra.pg import matrix_models as MM
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import unite_route as UR

REPO = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "studies/instruments/scripts"
VERDICT = SCRIPTS / "verdict_e6.py"
FIGURES = SCRIPTS / "figures_e6.py"
E6A = SCRIPTS / "e6a_writes.py"
STEPS_B = REPO / "infra/aws_pg/steps/instruments_e6b.json"
STEPS_C = REPO / "infra/aws_pg/steps/instruments_e6c.json"
HV = pathlib.Path("/Users/abaybektursun/projects/interp_research-neurons-harvest/bank/neurons/w1")
E1_METRICS = REPO / "studies/instruments/results/e1/1/t15/metrics.json"
E5_T4 = REPO / "studies/instruments/results/e5/1/t4/metrics.json"
NULL = REPO / "studies/instruments/results/e5/marked_null_lines.json"
UV = ["uv", "run", "--no-project", "--with", "matplotlib", "--with", "numpy", "python"]


def sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def V6():
    sys.path.insert(0, str(SCRIPTS))
    import verdict_e6 as V
    return V


def tiny_olmo2(seed: int = 0, n_layers: int = D1B.N_LAYERS, sliding: int | None = None):
    import transformers as T
    torch.manual_seed(seed)
    cfg = T.Olmo2Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=D1B.W_TINY, num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=16, max_position_embeddings=512, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    if sliding is not None:
        cfg.sliding_window = sliding
    return D1B.jitter(T.Olmo2ForCausalLM(cfg).to(torch.bfloat16).eval().requires_grad_(False), 1)


def corpus_of(tok, n_windows: int) -> dict:
    tokens = fake_corpus(tok, n_windows)
    offsets = np.arange(0, tokens.size + 37, 37)
    offsets[-1] = tokens.size
    return {"tokens": tokens, "doc_offsets": offsets, "record": {"vocab_sha256": MM.vocab_sha256(tok)}, "band_end": {}, "manifest_sha256": "tiny",
            "tokname": "tiny", "tokens_sha256": "tiny"}


def run_v(*args) -> tuple:
    r = subprocess.run([sys.executable, str(VERDICT), *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 5 or "VERDICT_E6" in r.stdout, (r.returncode, r.stdout[-2000:], r.stderr[-3000:], "the verdict crashed")
    return r.returncode, r.stdout, r.stderr


# ------------------------------------------------------------------------------------------ part 1: the olmo2 adapter entry
def part_adapter() -> dict:
    model = tiny_olmo2()
    ad = AD.Adapter(model, D1B.N_LAYERS)
    assert ad.describe()["arch"] == "olmo2" and AD.ARCH["olmo2"] == AD.ARCH["olmo3"], ad.describe()
    tok = tiny_tokenizer(bos=True)
    tokens = fake_corpus(tok, 30)
    kw = dict(tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=4, hw=WINDOW, device="cpu")
    toks = np.array([w * WINDOW + 30 + 7 * w for w in range(10)])
    units = [UR.pole_unit("L07N11+"), UR.pole_unit("L04N5-"), UR.planted_unit({"unit": "PLX", "layer": 6, "a": 3, "b": 9, "lam": 0.7})]
    eq = AD.olmo_equality(model, toks, units, UR.background_positions(-1000, 30 * WINDOW, WINDOW)["e1"][:40], **kw)
    cap = AD.capture(ad, model, toks, D1B.N_LAYERS - 2, set(), **kw)
    rr = E5.residual_rms(cap, D1B.N_LAYERS - 2)
    x, pos = UR.pack(tokens, toks, RA.LENGTH, tok.bos_token_id, WINDOW, "cpu")
    with torch.no_grad():
        hs = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, output_hidden_states=True).hidden_states
    rows = torch.arange(len(toks))
    want = torch.stack([E5.rms(hs[l + 1][rows, pos]) for l in range(D1B.N_LAYERS - 1)], 1)
    err = float(((rr - want).abs() / want).max())
    wrong = float(((E5.residual_rms({**cap, "mlp": {l: 0 * v for l, v in cap["mlp"].items()}}, D1B.N_LAYERS - 2) - want).abs() / want).max())
    assert err <= 2e-2 < wrong, (err, wrong, "the residual reconstruction does not match OLMo 2's hidden states")
    stops = {}
    for tag, m in (("sliding_window", tiny_olmo2(sliding=16)), ("input_layernorm", tiny_olmo2())):
        if tag == "input_layernorm":
            m.model.layers[0].input_layernorm = torch.nn.Identity()
        raised = False
        try:
            AD.Adapter(m, D1B.N_LAYERS)
        except AssertionError as e:
            raised, stops[tag] = True, str(e)[:90]
        assert raised, f"the olmo2 layout check passed a model with {tag}"
    return {"describe": ad.describe(), "equality": eq["compared"], "residual_err": err, "residual_err_without_mlp": wrong, "stops": stops}


# ------------------------------------------------------------------------------------------ part 2: the harvest step
def part_harvest(out: pathlib.Path) -> dict:
    steps = json.loads(STEPS_B.read_text())
    assert [s["name"] for s in steps] == ["harvest", "items", "e1b"], [s["name"] for s in steps]
    a = NHJ.parser().parse_args(shlex.split(steps[0]["args"]) + ["--out", "x"])
    q = next(s for s in json.loads((HV / "q4b/1/out/steps.json").read_text()) if s["name"] == "q4b")
    aq = NHJ.parser().parse_args(shlex.split(q["args"]) + ["--out", "x"])
    diff = sorted(k for k in vars(a) if vars(a)[k] != vars(aq)[k])
    assert diff == ["corpus", "model_key", "repo", "revision", "tokname"], diff
    o7 = NHJ.parser().parse_args(shlex.split(json.loads((HV / "o7_t15/1/out/steps.json").read_text())[0]["args"]) + ["--out", "x"])
    assert a.corpus == o7.corpus and aq.corpus == "/work/out/corpus/corpus", (a.corpus, o7.corpus, "the corpus is not the S3 copy of the q4b run's corpus step")
    K, F = MM.KNOWN[a.repo], MM.FACTS[a.repo]
    assert NH.EXPECTED_MODEL_TYPE[a.model_key] == "olmo2" == K["model_type"] == F["config"]["model_type"] and a.revision == K["sha"] == F["revision"], (a, K)
    assert (K["n_layers"], K["hidden"], K["vocab"]) == (32, 4096, 100352) and a.tokname == "olmo3" and a.max_tokens == "all" and a.sample_neurons == 360
    ai = J6B.parser().parse_args(shlex.split(steps[1]["args"]) + ["--out", "x"])
    ae = J6B.parser().parse_args(shlex.split(steps[2]["args"]) + ["--out", "x"])
    assert ai.stage == "items" and ai.harvest == "/work/out/harvest" and ae.stage == "e1b" and ae.items == "/work/out/items/e6b_items.json" and ae.corpus == a.corpus
    # neurons_harvest.load_model's first check (config model_type against the key) on a saved tiny OLMo 2; the load itself needs accelerate's device_map
    from transformers import AutoConfig
    model, tok = tiny_olmo2(), tiny_tokenizer(bos=True)
    model.save_pretrained(out / "tiny_olmo2")
    cfg = AutoConfig.from_pretrained(out / "tiny_olmo2")
    assert cfg.model_type == NH.EXPECTED_MODEL_TYPE["O2"] and cfg.model_type != NH.EXPECTED_MODEL_TYPE["O7"], cfg.model_type
    t0 = time.time()
    man = NH.run(out / "harvest", model=model, tok=tok, corpus=corpus_of(tok, D1B.N_WINDOWS + 2), model_key="O2", repo="tiny/olmo2", revision="tiny",
                 max_tokens=D1B.N_WINDOWS * WINDOW, window=WINDOW, batch=4, sample_neurons=18, seed=0, device="cpu", nll_tokens=D1B.N_WINDOWS * WINDOW)
    g0 = json.loads((out / "harvest" / "controls.json").read_text())["G0"]
    assert man["model_type"] == "olmo2" and g0["pass"] and not g0["failing_input"]["pass"] and not g0["failing_input_2"]["pass"], g0
    return {"args": vars(a), "differs_from_q4b": diff, "g0_max": max(g0["ratio"]), "g0_fail_max": max(g0["failing_input"]["ratio"]), "harvest_s": round(time.time() - t0, 1),
            "model": model, "tok": tok}


# ------------------------------------------------------------------------------------------ part 3: the items step
def part_items() -> dict:
    sys.path.insert(0, str(SCRIPTS))
    import build_items_e1b as B
    B.MODELS["o7_t15"] = {"repo": "allenai/Olmo-3-1025-7B", "arch": "olmo3", "nh_key": "O7"}
    out = {}
    for key in ("g4", "q4b", "o7_t15"):
        r1, s1 = B.build(key)
        r2, s2 = J6B.build(B.HV / key / "1" / "out" / key, key, B.MODELS[key])
        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True) and s1 == s2, (key, "instruments_e6b.build differs from build_items_e1b.build")
        out[key] = r2["counts"]["planted"]
    e5 = json.loads((REPO / "studies/instruments/inputs/e5_items.json").read_text())["models"]["o7"]
    r2.update({"model_key": "o7", "harvest_key": "o7_t15", "branch": "stage1-step1413814"})
    assert json.dumps(r2, sort_keys=True) == json.dumps(e5, sort_keys=True), "the o7_t15 record differs from e5_items.json's"
    return out


# ------------------------------------------------------------------------------------------ part 4: the e1b step on the tiny OLMo 2, verdict part b
def part_e1b(out: pathlib.Path, model, tok) -> dict:
    corpus = corpus_of(tok, D1B.N_WINDOWS + 2)
    ref = reference_a(model, tok, corpus["tokens"], list(range(D1B.N_WINDOWS)))
    rec = D1B.tiny_items("o2", "olmo2", ref, model, tok)
    rec["branch"] = "tiny"
    run = out / "e6b_run"
    ip = run / "items" / "e6b_items.json"
    NH.dump(ip, {"models": {"o2": rec}})
    d = run / "e1b"
    ns = argparse.Namespace(model_key="o2", items_sha256=sha(ip), harvest_window=WINDOW, batch=8, full_batch=4)
    job = JE1B.Job(d, ns, rec, model, tok, corpus, "cpu")
    job.run()
    chk = D1B.check_tiny(d, "o2", rec, model, tok, corpus["tokens"])
    (d / "samples").mkdir(exist_ok=True)
    t5 = J6B.t5_stage(d, rec, job, model, tok, corpus, "cpu", 8, {"device_name": "cpu"})
    z = np.load(d / "t5" / "o2.npz")
    ad = AD.Adapter(model, D1B.N_LAYERS)
    for j in range(min(3, len(z["members_layer"]))):
        l, k = int(z["members_layer"][j]), int(z["members_neuron"][j])
        dn = float((ad.post_gain(l) * ad.layers[l].mlp.down_proj.weight[:, k].float()).norm())
        assert abs(dn - float(z["dnorm"][j])) <= 1e-5 * dn, (l, k, dn, float(z["dnorm"][j]))
    shutil.copytree(out / "harvest", run / "harvest")
    o3 = d / "values.npz"                                  # the tiny job's own ā stands in for Olmo-3's E5 T4 values
    vout = out / "vb_tiny"
    code, so, se = run_v("--part", "b", "--run", run, "--out", vout, "--olmo3-values", o3)
    assert code in (0, 1, 4) and "BANK_INCOMPLETE" not in se, (code, so[-3000:], se[-3000:])
    tiny_lines = [x for x in so.splitlines() if x.startswith(("RULE", "VERDICT"))]
    V = V6()
    exits, readings = {"tiny": code}, {}
    close = lambda a, b: abs(a - b) < 1e-5
    for tag, mean, want in (("high", (0.16, 0.06), V.B_WORDS[0]), ("low", (0.01, 0.005), V.B_WORDS[1]), ("mid", (0.06, 0.03), V.B_WORDS[2])):
        plan = {"mean": mean, "zero": (0.14, 0.05), "median": (0.12, 0.05), "res": (0.13, 0.05)}
        sd = synth_b(out / f"synth_b_{tag}", run, rec, plan)
        c, sob, seb = run_v("--part", "b", "--run", sd, "--out", out / f"vb_{tag}", "--olmo3-values", o3)
        v = json.loads((out / f"vb_{tag}" / "verdict_e6b.json").read_text())
        assert c == 0 and v["reading"]["reading"] == want and close(v["reading"]["N_O2"]["value"], D5.expected4(rec, plan)), (tag, c, v["reading"], sob[-2000:], seb[-2000:])
        exits[tag], readings[tag] = c, v["reading"]["reading"]
    sd = out / "synth_b_high"
    inc = out / "synth_b_incomplete"
    shutil.copytree(sd, inc)
    (inc / "e1b" / "e1b" / "o2" / "pairs" / f"{rec['planted'][-1]['unit']}.npz").unlink()
    c5, _, e5 = run_v("--part", "b", "--run", inc, "--out", out / "vb_incomplete", "--olmo3-values", o3)
    assert c5 == 5 and "E6B_BANK_INCOMPLETE" in e5 and not (out / "vb_incomplete").exists(), (c5, e5[-2000:])
    exits["incomplete"] = c5
    fails = {}
    for fi, (grp, target) in sorted(V.FAILING_B.items()):
        cf, sof, sef = run_v("--part", "b", "--run", sd, "--out", out / "vb_high", "--olmo3-values", o3, "--failing-input", fi)
        f = json.loads((out / "vb_high" / f"failing_input_{fi}" / "verdict_e6b.json").read_text())
        want = 3 if (grp, target) in V.KNOWN_B else 4
        assert cf == want and not f["gates"][grp][target]["pass"] and f["reading"]["reading"].startswith("cancelled"), (fi, cf, sof[-2000:], sef[-2000:])
        others = {f"{g_}.{k}": x["pass"] for g_, gs in f["gates"].items() for k, x in gs.items() if (g_, k) != (grp, target)}
        assert all(others.values()), (fi, others)
        fails[fi] = cf
    return {"check_tiny": sorted(chk) if isinstance(chk, dict) else chk, "t5": t5["models"]["o2"]["summary"], "tiny_lines": tiny_lines, "exits": exits,
            "readings": readings, "fails": fails, "vout": out / "vb_high"}


def synth_b(dst: pathlib.Path, src: pathlib.Path, rec: dict, plan: dict) -> pathlib.Path:
    """The tiny E6b run with planted falls on the o2 bank (jobs.instruments_dryrun_e1b.planted_fall and SINGLE), as E5's dry run plants its T4 bank."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    d = dst / "e1b"
    m = json.loads((d / "metrics.json").read_text())
    for i, p in enumerate(rec["planted"]):
        f = d / "e1b" / "o2" / "pairs" / f"{p['unit']}.npz"
        z = dict(np.load(f))
        for s in ("J", "O"):
            b = (1.0 + 0.05 * np.arange(len(z[f"{s}_tok"]))).astype(np.float32)
            z[f"{s}_a_before"] = b
            z[f"{s}_a_after"] = np.stack([(b * (1 - D1B.planted_fall(plan, i, s, str(arm)))).astype(np.float32) for arm in z[f"{s}_arms"]])
        np.savez_compressed(f, **z)
        m["pairs"][p["unit"]]["sides"] = JE.record_of(z, ("J", "O"))
    for s in rec["singles"]:
        f = d / "e1b" / "o2" / "singles" / f"{s['pole']}.npz"
        z = dict(np.load(f))
        b = (1.0 + 0.05 * np.arange(len(z["J_tok"]))).astype(np.float32)
        z["J_a_before"] = b
        z["J_a_after"] = np.stack([(b * (1 - D1B.SINGLE[str(arm)])).astype(np.float32) for arm in z["J_arms"]])
        np.savez_compressed(f, **z)
        m["singles"][s["pole"]]["falls"] = {str(n): {k: RA.job_fall(b, z["J_a_after"][j])[k] for k in ("mean", "median", "n_pos")} for j, n in enumerate(z["J_arms"])}
    (d / "metrics.json").write_text(json.dumps(m))
    return dst


# ------------------------------------------------------------------------------------------ part 5: E6c on a tiny Olmo 3, verdict part c
def known_constants() -> dict:
    """The E6c job's KNOWN_CASES against E1's tracked metrics.json (sha256 9e7acee8...) and the real items' positions 0, n // 3, 2n // 3."""
    C = json.loads((REPO / "studies/instruments/inputs/e1_items.json").read_text())["primary"]
    m = json.loads(E1_METRICS.read_text())
    assert sha(E1_METRICS).startswith("9e7acee8f754"), sha(E1_METRICS)
    want = {C[i]["job"]: m["cases"][C[i]["job"]]["sides"]["J"]["I"]["mean__e"] for i in J6C.known_index(len(C))}
    assert J6C.KNOWN_CASES == want, (J6C.KNOWN_CASES, want, "the job's known cases are not E1's banked values")
    anchor = float(np.median([v["sides"]["J"]["I"]["mean__e"] for v in m["cases"].values() if not v["cancel"]["mean__e"]["cancelled"]]))
    assert abs(anchor - J6C.KNOWN_I) < 1e-4, anchor
    steps = json.loads(STEPS_C.read_text())
    a = J6C.parser().parse_args(shlex.split(steps[0]["args"]) + ["--out", "x"])
    e1 = json.loads((REPO / "studies/instruments/results/e1/1/steps.json").read_text())[0]["args"]
    assert a.corpus in e1 and a.items in e1 and a.revision in e1, (a, e1, "E6c does not read E1's items, corpus and revision")
    return {"known_cases": want, "anchor": anchor, "rotated_max": max(abs(v - list(want.values())[(i + 1) % 3]) for i, v in enumerate(want.values()))}


def part_e6c(out: pathlib.Path) -> dict:
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
    common = ["--t", "T_15", "--items", str(ip), "--harvest-window", str(WINDOW), "--scan-tokens", str(D1.N_WINDOWS * WINDOW), "--batch", "8", "--full-batch", "4"]
    d1 = out / "e1_tiny"
    a1 = JE.parser().parse_args(["--out", str(d1), "--stage", "e1"] + common)
    a1.items_sha256 = sha(ip)
    JE.Job(d1, a1, items, model, tok, corpus, "cpu").run()
    m1 = json.loads((d1 / "metrics.json").read_text())
    C = items["primary"]
    kj = [C[i]["job"] for i in J6C.known_index(len(C))]
    known = {k: m1["cases"][k]["sides"]["J"]["I"]["mean__e"] for k in kj}
    anchor = float(np.median([v["sides"]["J"]["I"]["mean__e"] for v in m1["cases"].values() if not v["cancel"]["mean__e"]["cancelled"]]))
    d = out / "e6c_run"
    a = J6C.parser().parse_args(["--out", str(d), "--revision", "stage1-step1413814", "--corpus", "tiny"] + common)
    a.items_sha256 = sha(ip)
    J6C.Job(d, a, items, model, tok, corpus, "cpu", known_cases=known).run()
    m = json.loads((d / "metrics.json").read_text())
    g = m["gates"]
    assert m["status"] == "done" and g["known_number"]["value"] == 0.0 and g["known_number"]["failing_input_failed"], g["known_number"]
    assert g["identity"]["value"] == 0.0 and g["identity"]["failing_input_failed"] and g["marked_write"]["pass"] and g["marked_write"]["failing_input_failed"], (g["identity"], g["marked_write"])
    assert g["exclusion"]["pass"] and g["samples"]["pass"], (g["exclusion"], g["samples"])
    kw = dict(tokens=tokens, length=RA.LENGTH, bos=tok.bos_token_id, batch=8, hw=WINDOW, device="cpu")
    ab = torch.tensor(np.load(d / "values.npz")["abar"])
    n_bitwise = 0
    for j in C:
        key = f"{j['pole']}__c{j['c']}"
        z, z1 = np.load(d / "e6c" / "cases" / f"{key}.npz"), np.load(d1 / "e1" / "cases" / f"{key}.npz")
        names, n1 = [str(x) for x in z["J_arms"]], [str(x) for x in z1["J_arms"]]
        for x in JE.CASE_SETS:
            assert np.array_equal(z[f"set_J_{x}_layer"], z1[f"set_J_{x}_layer"]) and np.array_equal(z[f"set_J_{x}_neuron"], z1[f"set_J_{x}_neuron"]), (key, x, "sets differ from E1's")
        if j["job"] in kj:
            for x in JE.CASE_SETS:
                assert np.array_equal(z["J_a_after"][names.index(f"free__{x}__mean")], z1["J_a_after"][n1.index(f"{x}__mean")]), (key, x, "the free arm differs from E1's")
                n_bitwise += 1
            assert not np.array_equal(z["J_a_after"][names.index("own_e__mean")], z["J_a_after"][names.index("free__own_e__mean")]), (key, "marked equals free")
        if j["job"] == kj[0]:
            unit = UR.pole_unit(j["pole"])
            S = [(int(l), int(k)) for l, k in zip(z["set_J_own_e_layer"], z["set_J_own_e_neuron"])]
            want = E5.readout_marked(model, z["J_tok"], unit, IV.written(S, "mean", {"mean": ab}, unit["layer"], "cpu"), **kw)["a"]
            assert np.array_equal(want, z["J_a_after"][names.index("own_e__mean")]), (key, "the marked arm is not an independent readout_marked")
    tiny_args = ["--part", "c", "--items", ip, "--e1-metrics", d1 / "metrics.json", "--known-anchor", f"{anchor:.12g}", "--null", NULL]
    code, so, se = run_v(*tiny_args, "--run", d, "--out", out / "vc_tiny")
    assert code in (0, 4) and "BANK_INCOMPLETE" not in se, (code, so[-3000:], se[-3000:])
    tiny_lines = [x for x in so.splitlines() if x.startswith(("GATE", "RULE", "VERDICT"))]
    # constructed banks
    V = V6()
    e1s, anchor_s = synth_e1(out / "e1_synth_metrics.json", m1, C)
    sargs = ["--part", "c", "--items", ip, "--e1-metrics", e1s, "--known-anchor", f"{anchor_s:.12g}", "--null", NULL]
    nl = json.loads(NULL.read_text())["marked_null_excluded"]
    exits, readings = {"tiny": code}, {}
    for tag, base, step, want in (("survives", 0.25, 0.002, "the reuse result survives the marked-token instrument"),
                                  ("between", 0.05, 0.001, "it does not separate from the null under this instrument"),
                                  ("channel", 0.01, 0.0005, "the reuse was the other-positions channel")):
        sd = synth_c(out / f"synth_c_{tag}", d, C, kj, base, step)
        c, soc, sec = run_v(*sargs, "--run", sd, "--out", out / f"vc_{tag}")
        v = json.loads((out / f"vc_{tag}" / "verdict_e6c.json").read_text())
        r = v["readings"]["I_marked_mean_e"]
        assert c == 0 and r["reading"].startswith(want) and abs(r["I"]["value"] - float(np.median([base + step * i for i in range(len(C))]))) < 1e-5, (tag, c, r, soc[-2000:], sec[-2000:])
        assert abs(v["reported"]["own_e_fall_marked"]["mean"]["value"] - 0.4) < 1e-5 and r["lines"]["null_p95"] == nl["p95"], v["reported"]["own_e_fall_marked"]
        exits[tag], readings[tag] = c, r["reading"]
    sd = out / "synth_c_survives"
    inc = out / "synth_c_incomplete"
    shutil.copytree(sd, inc)
    (inc / "e6c" / "cases" / f"{C[-1]['pole']}__c{C[-1]['c']}.npz").unlink()
    c5, _, e5 = run_v(*sargs, "--run", inc, "--out", out / "vc_incomplete")
    assert c5 == 5 and "E6C_BANK_INCOMPLETE" in e5 and not (out / "vc_incomplete").exists(), (c5, e5[-2000:])
    c3, so3, _ = run_v(*sargs[:-4], "--known-anchor", f"{anchor_s + 0.05:.12g}", "--null", NULL, "--run", sd, "--out", out / "vc_anchor_off")
    assert c3 == 3, (c3, so3[-2000:])
    exits.update({"incomplete": c5, "anchor_off": c3})
    fails = {}
    for fi, target in sorted(V.FAILING_C.items()):
        cf, sof, sef = run_v(*sargs, "--run", sd, "--out", out / "vc_survives", "--failing-input", fi)
        f = json.loads((out / "vc_survives" / f"failing_input_{fi}" / "verdict_e6c.json").read_text())
        want = 3 if target == "known_number" else 4
        assert cf == want and not f["gates"]["e6c"][target]["pass"] and f["readings"]["I_marked_mean_e"]["reading"].startswith("cancelled"), (fi, cf, sof[-2000:], sef[-2000:])
        others = {k: x["pass"] for k, x in f["gates"]["e6c"].items() if k != target}
        assert all(others.values()), (fi, others)
        fails[fi] = cf
    return {"tiny_lines": tiny_lines, "known": g["known_number"]["value"], "marked_write": g["marked_write"]["value"], "n_free_bitwise": n_bitwise,
            "exits": exits, "readings": readings, "fails": fails, "vout": out / "vc_survives", "readouts": m["readouts"], "marked_readouts": m["marked_readouts"]}


def synth_e1(path: pathlib.Path, m1: dict, C: list) -> tuple[pathlib.Path, float]:
    """E1's tiny metrics with each case's I_e(mean) = 0.30 + 0.05 i (live), the reference the constructed E6c bank's free arms reproduce."""
    m = json.loads(json.dumps(m1))
    for i, j in enumerate(C):
        c = m["cases"][j["job"]]
        c["sides"]["J"]["I"]["mean__e"] = 0.30 + 0.05 * i
        c["sides"]["J"]["falls"]["own_e__mean"]["mean"] = 0.5
        c["cancel"]["mean__e"]["cancelled"] = False
    path.write_text(json.dumps(m))
    return path, float(np.median([0.30 + 0.05 * i for i in range(len(C))]))


def synth_c(dst: pathlib.Path, src: pathlib.Path, C: list, kj: list, base: float, step: float) -> pathlib.Path:
    """The tiny E6c bank with planted falls: own 0.4, random 0.01, cross 0.4 (base + step i) under the marked write; free own 0.5, cross 0.5 (0.30 +
    0.05 i), random 0.01; identity 0, identity_fail 0.01."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    m = json.loads((dst / "metrics.json").read_text())
    for i, j in enumerate(C):
        f = dst / "e6c" / "cases" / f"{j['pole']}__c{j['c']}.npz"
        z = dict(np.load(f))
        names = [str(x) for x in z["J_arms"]]
        b = (1.0 + 0.05 * np.arange(len(z["J_tok"]))).astype(np.float32)

        def fall(n: str) -> float:
            if n == "identity":
                return 0.0
            if n == "identity_fail":
                return 0.01
            free = n.startswith("free__")
            x = n.split("__")[-2]
            if x.startswith("own"):
                return 0.5 if free else 0.4
            if x.startswith("random"):
                return 0.01
            return 0.5 * (0.30 + 0.05 * i) if free else 0.4 * (base + step * i)
        z["J_a_before"] = b
        z["J_a_after"] = np.stack([(b * (1 - fall(n))).astype(np.float32) for n in names])
        np.savez_compressed(f, **z)
        m["cases"][j["job"]]["falls"] = {n: {k: RA.job_fall(b, z["J_a_after"][q])[k] for k in ("mean", "median", "n_pos")} for q, n in enumerate(names)}
    (dst / "metrics.json").write_text(json.dumps(m))
    return dst


# ------------------------------------------------------------------------------------------ part 6: E6a on a fabricated T5 bank
def fabricate_e6a(root: pathlib.Path) -> dict:
    """T5 npz for o7 (34 members), g4 (189), q4b (61) and 56 pair npz each, with o7's per-member writes 2.0 x g4's top 34 and 1.0 x q4b's top 34."""
    rng = np.random.default_rng(1)
    d = {"o7": 64, "g4": 64, "q4b": 64}
    n = {"o7": 34, "g4": 189, "q4b": 61}
    base = np.exp(rng.normal(np.log(0.01), 0.05, 34))
    w = {"o7": 2.0 * base, "g4": np.concatenate([base, np.exp(rng.normal(np.log(0.001), 0.3, 189 - 34))]),
         "q4b": np.concatenate([2.0 * base, np.exp(rng.normal(np.log(0.002), 0.3, 61 - 34))])}
    P = {"o7": root / "e5/1/t12/e5/pairs", "g4": root / "e1b/1/g4/e1b/g4/pairs", "q4b": root / "e1b/1/q4b/e1b/q4b/pairs"}
    key = {"o7": "set_members", "g4": "set_J_members", "q4b": "set_J_members"}
    (root / "e5/1/t5/t5").mkdir(parents=True, exist_ok=True)
    for mk in ("o7", "g4", "q4b"):
        L = np.arange(n[mk]) % 20
        K = np.arange(n[mk]) + 1000
        s = np.full((200, 21), 2.0)
        r = np.tile(1.0 + 0.1 * np.arange(21), (200, 1))      # the residual rms grows with depth, so the next layer's rms is a wrong input
        ab = w[mk] * np.sqrt(d[mk]) * r[0, L] / 2.0             # write = |ā| ||d|| s / rms with ||d|| = 1, s = 2 -> w = write / sqrt(d)
        write = (ab[None] * 1.0 * s[:, L] / r[:, L]).astype(np.float32)
        np.savez_compressed(root / "e5/1/t5/t5" / f"{mk}.npz", members_layer=L.astype(np.int16), members_neuron=K.astype(np.int32), abar=ab.astype(np.float32),
                            abar_capture=ab.astype(np.float32), dnorm=np.ones(n[mk], np.float32), s=s.astype(np.float32), rms=r.astype(np.float32), write=write,
                            positions=np.arange(200))
        P[mk].mkdir(parents=True, exist_ok=True)
        order = np.argsort(-w[mk])
        for u in range(56):
            take = order[[(u + q) % min(34, n[mk]) for q in range(3)]] if u % 2 == 0 else order[[(u + q) % min(34, n[mk]) for q in range(2)]]
            np.savez_compressed(P[mk] / f"PL{u:02d}.npz", **{f"{key[mk]}_layer": L[take].astype(np.int16), f"{key[mk]}_neuron": K[take].astype(np.int32)})
    return d


def part_e6a(out: pathlib.Path) -> dict:
    root = out / "e6a_bank"
    d = fabricate_e6a(root)
    hid = ",".join(str(d[mk]) for mk in ("o7", "g4", "q4b"))
    r = subprocess.run(UV + [str(E6A), "--banks", str(root), "--out", str(out / "e6a"), "--hidden", hid, "--dry"], capture_output=True, text=True)
    assert r.returncode == 0, (r.returncode, r.stdout[-3000:], r.stderr[-3000:])
    v = json.loads((out / "e6a" / "e6a_writes.json").read_text())
    R = v["readings"]
    assert R["g4"]["reading"] == "Olmo's units write more" and R["q4b"]["reading"] == "they do not", R
    assert abs(R["g4"]["ratio"]["value"] - 2.0) < 1e-4 and abs(R["q4b"]["ratio"]["value"] - 1.0) < 1e-4, (R["g4"]["ratio"], R["q4b"]["ratio"])
    assert (out / "e6a" / "figures" / "fig_e6a.png").exists() and (out / "e6a" / "figures" / "fig_e6a.svg").exists()
    fails = {}
    for fi in ("recompute_next_layer", "positive_control_halved"):
        rf = subprocess.run(UV + [str(E6A), "--banks", str(root), "--out", str(out / "e6a"), "--hidden", hid, "--failing-input", fi], capture_output=True, text=True)
        f = json.loads((out / "e6a" / f"failing_input_{fi}" / "e6a_writes.json").read_text())
        target = {"recompute_next_layer": "recompute", "positive_control_halved": "positive_control"}[fi]
        assert rf.returncode == 4 and not f["gates"][target]["pass"], (fi, rf.returncode, rf.stdout[-2000:])
        fails[fi] = rf.returncode
    shutil.rmtree(root / "e5/1/t5/t5")
    r5 = subprocess.run(UV + [str(E6A), "--banks", str(root), "--out", str(out / "e6a_missing"), "--hidden", hid], capture_output=True, text=True)
    assert r5.returncode == 5 and not (out / "e6a_missing").exists(), (r5.returncode, r5.stderr[-1000:])
    return {"lines": [x for x in r.stdout.splitlines() if x.startswith(("RULE", "CONTROL", "E6A"))], "fails": fails, "missing": r5.returncode}


# ------------------------------------------------------------------------------------------ part 8: the size of the runs
def sizes() -> dict:
    o7 = json.loads((HV / "o7_t15/1/out/o7_t15/manifest.json").read_text())
    q4 = json.loads((HV / "q4b/1/out/q4b/manifest.json").read_text())
    t4 = json.loads(E5_T4.read_text())
    e1 = json.loads(E1_METRICS.read_text())
    fp32_gb = MM.FACTS["allenai/OLMo-2-1124-7B"]["config"]["total_size"] / 1e9
    dl_s = fp32_gb * 1e9 / (o7["download"]["MB_per_s"] * 1e6)
    harvest_s = o7["wall_s"] + q4["timing_s"]["snapshot_write_s"] + dl_s + 2 * o7["load_s"]
    e1b_s = t4["timing_s"]["wall_after_load"] + dl_s + 2 * t4["load_s"] + 120
    boot = 300
    items = json.loads((REPO / "studies/instruments/inputs/e1_items.json").read_text())
    nf = lambda t: RA.n_forwarded(np.asarray(t, np.int64)[:12], RA.LENGTH, 100257, UR.HARVEST_WINDOW)
    # per case: 2 captures (J and O*), before, identity, 6 sets x 2 values marked (excluded = unexcluded below the first B_t layer: cached), known: 6 free + 1
    tok = sum(nf(j["J_tok"]) * (1 + 1 + 1 + 12) + nf(j["O_tok"]) for j in items["primary"]) + 3 * 7 * 12 * 256
    e6c_s = tok / e1["units_tokens_per_second"] + e1["download_and_corpus_s"] + e1["load_s"] + e1["timing_s"]["known"] + e1["timing_s"]["background"] + e1["timing_s"]["values"]
    return {"o7_harvest_wall_h": o7["wall_s"] / 3600, "o7_tokens_per_s": o7["tokens_per_second"], "q4b_harvest_wall_h": q4["wall_s"] / 3600,
            "q4b_tokens_per_s": q4["tokens_per_second"], "o2_download_gb": fp32_gb, "o2_download_min": dl_s / 60, "e6b_harvest_h": harvest_s / 3600,
            "e6b_e1b_h": e1b_s / 3600, "e6b_total_h": (harvest_s + e1b_s + boot + 60) / 3600, "e6c_tokens": tok, "e6c_wall_h": (e6c_s + boot) / 3600,
            "e5_t4_wall_after_load_s": t4["timing_s"]["wall_after_load"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    t0 = time.time()
    print("SELFCHECK instruments_adapter:", json.dumps(AD.selfcheck(), ensure_ascii=False))
    p1 = part_adapter()
    print(f"ADAPTER olmo2: {p1['describe']}; olmo_equality on a tiny OLMo 2: {', '.join(p1['equality'])} equal exactly; residual reconstruction "
          f"{p1['residual_err']:.1e} (without the MLP terms {p1['residual_err_without_mlp']:.2f}); stops: {json.dumps(p1['stops'])}")
    p2 = part_harvest(out)
    print(f"HARVEST STEP: args differ from the q4b harvest only in {p2['differs_from_q4b']}; O2 -> olmo2 = KNOWN = facts; revision "
          f"{p2['args']['revision'][:12]} = KNOWN's sha; tokname olmo3; sample-neurons {p2['args']['sample_neurons']}, seed {p2['args']['seed']}, max-tokens "
          f"{p2['args']['max_tokens']}, snapshot-bands {p2['args']['snapshot_bands']}; load_model's config check passes O2 and fails O7; tiny OLMo 2 harvest: G0 max "
          f"{p2['g0_max']:.4f} (swapped activation {p2['g0_fail_max']:.3f}, fails) in {p2['harvest_s']} s; items and e1b args parse")
    p3 = part_items()
    print(f"ITEMS STEP: instruments_e6b.build = build_items_e1b.build exactly on {json.dumps(p3)} (pairs per bank); the o7_t15 record = e5_items.json's o7 record")
    p4 = part_e1b(out, p2["model"], p2["tok"])
    print(f"E1B STEP (tiny OLMo 2, key o2): E1b's job passes check_tiny; T5 {json.dumps(p4['t5'])}, ||d_k|| = the weights' to 1e-5")
    for x in p4["tiny_lines"]:
        print("  tiny b", x)
    print(f"VERDICT B constructed: {json.dumps(p4['readings'])}; exits {json.dumps(p4['exits'])}; failing inputs {json.dumps(p4['fails'])}")
    kc = known_constants()
    print(f"E6C KNOWN: the job's three known cases = E1's banked I_e(mean) {json.dumps(kc['known_cases'])}; E1's median over live cases {kc['anchor']:.4f}; "
          f"rotated by one case they miss by up to {kc['rotated_max']:.3f}; the steps file reads E1's items, corpus and revision")
    p5 = part_e6c(out)
    print(f"E6C (tiny Olmo 3, 6 cases): {p5['readouts']} readouts ({p5['marked_readouts']} marked); the known cases' {p5['n_free_bitwise']} free arms = E1's bank bitwise, "
          f"known gate {p5['known']}; marked write probe {json.dumps(p5['marked_write'])}; the marked arm = an independent readout_marked")
    for x in p5["tiny_lines"]:
        print("  tiny c", x)
    print(f"VERDICT C constructed: {json.dumps(p5['readings'])}; exits {json.dumps(p5['exits'])}; failing inputs {json.dumps(p5['fails'])}")
    p6 = part_e6a(out)
    for x in p6["lines"]:
        print("  e6a fabricated", x)
    print(f"E6A fabricated: failing inputs {json.dumps(p6['fails'])}; missing bank exit {p6['missing']}")
    figs = {}
    for part, vout in (("b", p4["vout"]), ("c", p5["vout"])):
        r = subprocess.run(UV + [str(FIGURES), "--part", part, "--verdict", str(vout / f"verdict_e6{part}.json"), "--out", str(out / "figures")], capture_output=True, text=True)
        assert r.returncode == 0 and (out / "figures" / f"fig_e6{part}.png").exists() and (out / "figures" / f"fig_e6{part}.svg").exists(), (r.stdout[-2000:], r.stderr[-3000:])
        figs[part] = str(out / "figures" / f"fig_e6{part}.png")
    print(f"FIGURES: {json.dumps(figs)}")
    s = sizes()
    print(f"SIZE E6b: Olmo-3-7B's full-corpus harvest (same dims, same tokenizer and corpus) took {s['o7_harvest_wall_h']:.2f} h at {s['o7_tokens_per_s']:.0f} tokens/s "
          f"(Qwen3-4B's {s['q4b_harvest_wall_h']:.2f} h at {s['q4b_tokens_per_s']:.0f}); OLMo-2 downloads {s['o2_download_gb']:.1f} GB (float32) twice, "
          f"{s['o2_download_min']:.1f} min each; harvest step {s['e6b_harvest_h']:.2f} h, e1b step {s['e6b_e1b_h']:.2f} h (E5 T4's {s['e5_t4_wall_after_load_s']:.0f} s), "
          f"pipeline {s['e6b_total_h']:.2f} GPU-h")
    print(f"SIZE E6c: {s['e6c_tokens'] / 1e6:.1f} M tokens at E1's rate: {s['e6c_wall_h']:.2f} GPU-h with boot and load")
    (out / "metrics.json").write_text(json.dumps({"adapter": {k: v for k, v in p1.items()}, "harvest": {k: v for k, v in p2.items() if k not in ("model", "tok")},
                                                  "items": p3, "e1b": {k: v for k, v in p4.items() if k != "vout"}, "e6c": {k: v for k, v in p5.items() if k != "vout"},
                                                  "e6a": p6, "sizes": s, "seconds": round(time.time() - t0, 1)}, indent=1, default=str))
    print(f"seconds {time.time() - t0:.0f}")
    print("INSTRUMENTS_DRYRUN_E6_OK")


if __name__ == "__main__":
    main()
