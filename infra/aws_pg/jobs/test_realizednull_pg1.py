"""REALIZEDNULL N3 job: pure-logic tests on tiny tensors, and its integration with N1, with no model.

The integration tests at the end call every `# N1:` call site's real N1 function (infra/pg/realizednull_core.py)
on inputs built by N3's own helpers, from Gram and probe passes over a toy 18-layer attention model with
PaliGemma's call signature, and feed N3's job_metrics to N1's gates().
Run: python3 -m pytest infra/aws_pg/jobs/test_realizednull_pg1.py -q
"""
import inspect
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from infra.aws_pg.jobs import gutenberg_gb1, invariant_core_func
from infra.aws_pg.jobs import realizednull_pg1 as rn
from infra.pg import gutenberg_core as G
from infra.pg import realizednull_core as R

T0 = "2026-11-01T00:00:00Z"


# ---------------- masks, positions, units ----------------

def test_forced_masks_on_toy_sequence_with_image_block():
    S, pos = 8, 3                                        # positions 0-2 = image block + bos
    lo = torch.finfo(torch.bfloat16).min
    allowed = {}
    for m in rn.FORCED:
        mk = rn.forced_mask(m, S, pos)
        assert mk.shape == (1, 1, S, S) and mk.dtype == torch.bfloat16
        assert set(mk.unique().tolist()) <= {0.0, lo}
        allowed[m] = mk[0, 0] == 0
    tri = torch.tril(torch.ones(S, S, dtype=torch.bool))
    assert torch.equal(allowed["causal"], tri)
    assert bool(allowed["bidir"].all())
    pl = tri.clone()
    pl[:pos, :pos] = True
    assert torch.equal(allowed["prefixlm"], pl)
    assert allowed["prefixlm"][0, 2] and not allowed["causal"][0, 2]       # image sees later image
    assert not allowed["prefixlm"][3, 4] and allowed["prefixlm"][7, 0]     # text stays causal


def test_position_bookkeeping_image_bos_text_newline():
    n_text, V = 7, 50
    win = torch.arange(10, 10 + n_text)
    ids = torch.cat([torch.full((rn.N_IMAGE,), 40), torch.tensor([2]), win, torch.tensor([3])])
    pos = rn.block_at(ids, win)
    assert pos == rn.N_PREFIX == 257 and ids.numel() == 256 + 1 + n_text + 1
    torch.manual_seed(0)
    logits = torch.randn(ids.numel(), V)
    for t in range(ids.numel() - 1):
        logits[t, ids[t + 1]] += 20.0                     # row t predicts token t+1
    logits = logits.to(torch.bfloat16)
    r = rn.score_logits(logits, pos, n_text, win)
    assert r["ntok"] == n_text - 1
    assert (r["top_id"] == win[1:].numpy()).all()         # scored rows pos..pos+n_text-2 → win[1:]
    ref = torch.log_softmax(logits[pos:pos + n_text - 1].float(), -1)[torch.arange(n_text - 1), win[1:]]
    assert np.allclose(r["lp"], ref.numpy(), atol=1e-5)
    assert r["nll"] == pytest.approx(-float(ref.sum()), rel=1e-5)


class _Tok:
    """Character tokenizer whose "x\\n" merges into one token, the processor's trailing-newline merge."""
    MERGE = 999

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            if text[i:i + 2] == "x\n":
                ids.append(self.MERGE)
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, ids, skip_special_tokens=False):
        return "".join("x\n" if t == self.MERGE else chr(t) for t in ids)

    def convert_ids_to_tokens(self, ids):
        return self.decode([ids]) if isinstance(ids, int) else [self.decode([t]) for t in ids]


class _Proc:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, text, images, return_tensors):
        ids = torch.tensor([[500] * 256 + [2] + self.tok(text[0] + "\n").input_ids[0].tolist()])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "pixel_values": torch.zeros(1, 3, 2, 2)}


def test_units_and_trailing_token_trim_match_gb1():
    assert rn.build_windows is gutenberg_gb1.build_windows
    assert rn._locate_text_block is invariant_core_func._locate_text_block
    tok = _Tok()
    units, trimmed = rn.make_units(["abc", "abx"], tok, _Proc(tok), image=None)
    assert trimmed == [1]                                 # "abx\n" merged → final token dropped once
    assert [u["text"] for u in units] == ["abc", "ab"]
    assert [u["n_text"] for u in units] == [3, 2]
    assert [u["pos"] for u in units] == [257, 257]
    with pytest.raises(AssertionError):                   # still absent after the trim → fail loud
        rn.make_units(["axx"], tok, _Proc(tok), image=None)


# ---------------- hooks ----------------

def test_probe_hooks_out_of_place():
    n, S = 4, 260
    torch.manual_seed(1)
    q, k, v = (torch.nn.Linear(n, n, bias=False) for _ in range(3))
    u = torch.as_tensor(rn.probe_vector(n), dtype=torch.float32)
    seen = {}
    k.register_forward_pre_hook(rn.probe_add_hook(u))
    v.register_forward_pre_hook(rn.probe_add_hook(u))
    for name, mod in (("q", q), ("k", k), ("v", v)):
        mod.register_forward_pre_hook(lambda _m, a, name=name: seen.__setitem__(name, a[0].clone()))
    x = torch.randn(1, S, n)
    x0 = x.clone()
    k(x), v(x), q(x)                                      # one shared input tensor, as at layer 0
    assert torch.equal(x, x0) and torch.equal(seen["q"], x0)
    g = x0[:, 200:256].norm(dim=-1, keepdim=True) * u
    for name in ("k", "v"):
        d = seen[name] - x0
        assert torch.equal(d[:, :200], torch.zeros_like(d[:, :200]))
        assert torch.equal(d[:, 256:], torch.zeros_like(d[:, 256:]))
        assert torch.allclose(d[:, 200:256], g, atol=1e-6)


def test_ablation_hook_identity_projection_and_realized_perturbation():
    n, S, p, nt = 6, 12, 5, 5
    torch.manual_seed(2)
    h = torch.randn(1, S, n).to(torch.bfloat16)
    h32 = h.float()
    state = {"pos": p, "n_text": nt, "acc": False}
    mu = torch.cat([h32[0, :p], h32[0, p:].mean(0, keepdim=True)])
    for op in ("zero", "zero_text", "mean"):                       # k = 0 is the identity, bitwise
        assert torch.equal(rn.ablation_hook(torch.zeros(n, 0), op, mu, state)(None, (h,))[0], h)
    U = torch.linalg.qr(torch.randn(n, 2))[0]
    z = rn.ablation_hook(U, "zero", mu, state)(None, (h,))[0].float()
    assert float((z @ U).abs().max()) < 0.05                       # span removed up to the bf16 cast
    zt = rn.ablation_hook(U, "zero_text", mu, state)(None, (h,))[0]
    assert torch.equal(zt[:, :p], h[:, :p]) and float((zt[:, p:].float() @ U).abs().max()) < 0.05
    mean = rn.ablation_hook(U, "mean", mu, state)(None, (h,))[0]
    assert torch.equal(mean[:, :p], h[:, :p])                      # prefix at its class mean: unchanged
    W = torch.randn(n, n)
    acc = {"num": [0.0], "den": [0.0]}
    out = rn.ablation_hook(U, "zero", mu, {**state, "acc": True}, W, acc, 0)(None, (h,))[0].float()
    t = slice(p, p + nt - 1)
    assert float(acc["num"][0]) == pytest.approx(float(((h32 - out)[0, t] @ W.T).pow(2).sum()), rel=1e-6)
    assert float(acc["den"][0]) == pytest.approx(float((h32[0, t] @ W.T).pow(2).sum()), rel=1e-6)


# ---------------- writers ----------------

ARMS = ["none", "none_causal", "band", "em_s0", "ident_zero", "band_dup", "topenergy32", "haar_s0",
        "ident_mean", "band_mean", "topsigma32_mean", "haar_mean_s0", "rt_s0"]


def _scores(arm, windows, f_shift):
    """Synthetic per-window NLL and per-position arrays. Windows >= 98 carry f_shift, so any number that
    reads them changes between two runs that differ only in f_shift."""
    k = ARMS.index(arm)
    out = {}
    for w in windows:
        rng = np.random.default_rng(1000 * k + w)
        tgt = np.arange(511) % 7
        top = np.where(rng.random(511) < 0.4, tgt, tgt + 1)
        shift = f_shift if w >= rn.F0 else 0.0
        out[w] = {"nll": 1500.0 + 3.0 * k + 0.01 * w + shift + (0 if k < 2 else rng.random()),
                  "ntok": 511, "lp": np.log(rng.uniform(0.01, 1.0, 511)) - shift,
                  "top_id": top, "top_lp": np.log(rng.uniform(0.3, 1.0, 511))}
    return out


def _run_writers(out, f_shift):
    sp = rn.splits(196, 32)
    sink = rn.Sink(out, "discovery", now=lambda: T0)
    rec = rn.Recorder(sink, sp, ARMS)
    targets = {w: np.arange(511) % 7 for w in range(196)}
    rec.stage("corpus")
    sink.json("sealed/corpus_f.json", {"window_text_sha256": "f" * 64, "trimmed": [101]})
    for arm in ARMS:
        sc = _scores(arm, sp["R"] if arm == "none_causal" else range(196), f_shift)
        sink.event("arm", arm=arm)
        rec.nll(arm, "zero", sc)
        sel, Y = rec.select(sc)

        def payload(s):                    # stands in for the N1 sample records: index and numbers
            if s is None:
                return None
            wins = [{"index": s[r], "role": r, "Y": Y[s[r]], "nll_sum_arm": sc[s[r]]["nll"],
                     "logp_arm": sc[s[r]]["lp"][:5]} for r in ("max", "min", "common")]
            ex = rn.sample_examples([s["common"], s["max"], s["min"]], targets, rec.none, sc,
                                    rn.N_PREFIX, str)
            return {"arm": arm, "windows": wins, "examples": ex}
        rec.write_samples(arm, payload(sel["open"]), payload(sel["sealed"]))
        rec.done(arm)
    sink.json("metrics.json", rn.metrics_doc(rec, {"G-PRE1": rn.pre1_gate(
        rec.summary["none"][rn.wkey("ppl", sp["R"])], sp["R"])}, {"smoke": False}, {"mask": "native"}))


def _open_files(root):
    return sorted(p.relative_to(root) for p in root.rglob("*")
                  if p.is_file() and p.relative_to(root).parts[0] != "sealed")


def _walk(obj, rel):
    if isinstance(obj, dict):
        for k, v in obj.items():
            for m in rn.WKEY.finditer(str(k)):
                assert int(m.group(2)) < rn.F0, (rel, k)
            if k in ("window", "windows", "index"):
                assert all(w < rn.F0 for w in (v if isinstance(v, list) else [v]) if isinstance(w, int)), (rel, k)
            _walk(v, rel)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, rel)


def test_no_confirm_stat_in_job_outputs(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    _run_writers(a, 0.0)
    log_a = capsys.readouterr().out
    _run_writers(b, 1000.0)
    log_b = capsys.readouterr().out
    # nothing outside sealed/ depends on windows >= 98: same files, same bytes, same log
    assert _open_files(a) == _open_files(b)
    for rel in _open_files(a):
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel
    assert log_a == log_b and all(int(m.group(2)) < rn.F0 for m in rn.WKEY.finditer(log_a))
    # no key, window field or sample index outside sealed/ names a window >= 98
    for rel in _open_files(a):
        text = (a / rel).read_text()
        docs = [json.loads(x) for x in text.splitlines()] if rel.suffix == ".jsonl" else [json.loads(text)]
        for d in docs:
            _walk(d, rel)
    # the sealed side holds windows 98-195 for every arm scored on them, and only those
    rows = [json.loads(x) for x in (a / "sealed/window_nll_confirm.jsonl").read_text().splitlines()]
    assert [r["arm"] for r in rows] == [x for x in ARMS if x != "none_causal"]
    assert all(r["windows"] == list(range(98, 196)) for r in rows)
    rd = [json.loads(x) for x in (a / "window_nll_rd.jsonl").read_text().splitlines()]
    assert [r["arm"] for r in rd] == ARMS and all(r["windows"] == list(range(98)) for r in rd)
    assert (a / "sealed/samples/band.json").exists() and not (a / "sealed/samples/none_causal.json").exists()
    sealed = json.loads((a / "sealed/samples/band.json").read_text())
    assert {w["index"] for w in sealed["windows"]} <= set(range(98, 196))
    assert (a / "sealed/window_nll_confirm.jsonl").read_bytes() != (b / "sealed/window_nll_confirm.jsonl").read_bytes()
    # the detectors fire on a leak: the write guard, and the two-run comparison
    sink = rn.Sink(a, "discovery", now=lambda: T0)
    for leak in ({"ppl_w0_195": 1.0}, {"windows": [97, 98]}, {"x": {"index": 150}}):
        with pytest.raises(AssertionError):
            sink.json("metrics.json", leak)
    leaky = lambda f: sum(s["nll"] for s in _scores("band", range(196), f).values())
    assert leaky(0.0) != leaky(1000.0)


def test_event_markers_in_order(tmp_path):
    sink = rn.Sink(tmp_path, "smoke", now=lambda: T0)
    sink.event("start", budget_s=2400)
    for st in ("model_load", "model_loaded", "probes", "forced_mask_selftest", "forced_mask_selftest_done"):
        sink.event(st)
    sink.event("arm", arm="none")
    rn.Sink(tmp_path, "smoke", now=lambda: T0).event("arm", arm="band")       # append-only
    rows = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["start", "model_load", "model_loaded", "probes",
                                          "forced_mask_selftest", "forced_mask_selftest_done", "arm", "arm"]
    assert rows[0] == {"stage": "start", "utc": T0, "budget_s": 2400}
    # the job itself emits them in this order, every arm after the self-test
    src = inspect.getsource(rn.main)
    at = [src.index(s) for s in ('sink.event("start"', 'sink.event("model_load")',
                                 'sink.event("model_loaded")', 'sink.event("forced_mask_selftest")',
                                 'sink.event("forced_mask_selftest_done")', 'sink.event("arm"')]
    assert at == sorted(at)
    assert src.index('_load_full_pg1(') > at[1] and src.index('_load_full_pg1(') < at[2]


def test_progress_rewritten_per_arm(tmp_path):
    sp = rn.splits(6, 4)
    rec = rn.Recorder(rn.Sink(tmp_path, "smoke", now=lambda: T0), sp, ["none", "band", "em_s0"])
    for k, arm in enumerate(["none", "band", "em_s0"], start=1):
        rec.nll(arm, "zero", {w: {"nll": 100.0 + k + w, "ntok": 511} for w in range(6)})
        rec.done(arm)
        p = json.loads((tmp_path / "progress.json").read_text())      # one document, replaced
        assert (p["arm"], p["n_done"], p["arms_done"]) == (arm, k, ["none", "band", "em_s0"][:k])
    assert not list(tmp_path.glob("*.tmp"))


def test_sample_writer_right_and_wrong_examples():
    targets = {4: np.array([5, 6, 7, 8]), 5: np.array([1, 2, 3, 4])}

    def sc(top):
        return {w: {"top_id": np.array(t), "lp": np.log(np.full(4, 0.25)), "top_lp": np.log(np.full(4, 0.5))}
                for w, t in top.items()}
    none = sc({4: [9, 6, 9, 9], 5: [1, 0, 0, 0]})
    arm = sc({4: [0, 0, 0, 0], 5: [0, 0, 3, 0]})
    ex = rn.sample_examples([4, 5], targets, none, arm, 257, lambda i: f"t{i}")
    r, w = ex["none"]["right"], ex["none"]["wrong"]
    assert (r["window"], r["scored_index"], r["position"], r["target"], r["right_none"]) == (4, 1, 258, "t6", True)
    assert (w["window"], w["scored_index"], w["target"], w["top1_none"], w["right_none"]) == (4, 0, "t5", "t9", False)
    ar = ex["arm"]["right"]
    assert (ar["window"], ar["scored_index"], ar["target"], ar["right_arm"], ar["right_none"]) == (5, 2, "t3", True, False)
    assert ex["arm"]["wrong"]["window"] == 4 and ex["arm"]["wrong"]["scored_index"] == 0
    assert ar["logp_target_arm"] == pytest.approx(np.log(0.25)) and ar["p_top1_none"] == pytest.approx(0.5)
    assert rn.sample_examples([4], targets, none, sc({4: [0, 0, 0, 0]}), 257, str)["arm"]["right"] is None
    with pytest.raises(AssertionError):                   # the unablated model must show both
        rn.sample_examples([4], targets, sc({4: [0, 0, 0, 0]}), arm, 257, str)


# ---------------- flags, roster, gates ----------------

def test_smoke_sets_windows_and_draws():
    a = rn.parse_args(["--out", "x", "--smoke"])
    assert (a.n_windows, a.n_construct) == (6, 4)
    sp = rn.splits(a.n_windows, a.n_construct)
    assert sp["R"] == list(range(6)) and sp["C"] == [0, 1, 2, 3] and sp["F"] == []
    names = [x[0] for x in rn.roster(a)]
    count = lambda pat: sum(bool(re.fullmatch(pat, n)) for n in names)
    assert {"em": count(r"em_s\d+"), "spec": count(r"spec_s\d+"), "haar": count(r"haar_s\d+"),
            "rt": count(r"rt_s\d+"), "emt": count(r"emt_s\d+"), "rtall": count(r"rtall_s\d+"),
            "rthalf": count(r"rthalf_s\d+"), "rt2x": count(r"rt2x_s\d+"),
            "rt_mean": count(r"rt_mean_s\d+"), "spec_mean": count(r"spec_mean_s\d+"),
            "haar_mean": count(r"haar_mean_s\d+"), "em_zt": count(r"em_zt_s\d+"),
            "rt_zt": count(r"rt_zt_s\d+")} == {
        "em": 4, "spec": 4, "haar": 4, "rt": 2, "emt": 2, "rtall": 1, "rthalf": 1, "rt2x": 1,
        "rt_mean": 1, "spec_mean": 1, "haar_mean": 1, "em_zt": 1, "rt_zt": 1}
    singles = {"ident_zero", "band", "band_dup", "topenergy32", "topsigma32", "bottomsigma32", "anti32",
               "band_zt", "ident_mean", "band_mean", "topenergy32_mean", "topsigma32_mean"}
    assert singles <= set(names) and len(names) == len(set(names)) == 40
    full = rn.roster(rn.parse_args(["--out", "x"]))
    assert len(full) == 214 and full[0][:2] == ("band", "zero")
    assert [x[0] for x in full if x[3] == "pre"][1:] == [
        "em_s0", "em_s1", "em_s2", "spec_s0", "spec_s1", "spec_s2", "topenergy32", "topsigma32",
        "bottomsigma32", "anti32", "haar_s0", "haar_s1", "haar_s2"]
    top = [x[0] for x in rn.roster(rn.parse_args(["--out", "x", "--topup-families", "em,rt",
                                                   "--slot-offset", "40"]))]
    assert len(top) == 65 and "em_mean_s59" in top and "rt_s40" in top and "emt_s40" not in top


def test_family_gates_count_as_n1_does():
    """G-PIVOT counts every record with pivot = True, matched or not (N1's gates, §6.1 "share of a family's
    draws with pivot = True"); an unreachable summed target is reported as "sum", not cast to a layer."""
    base = {"pivot": False, "matched": True, "reachable": True, "exhausted": False, "unreachable_layers": []}
    cons = {"rt_s0": {**base, "pivot": True, "matched": False, "exhausted": True},
            "rt_s1": {**base, "matched": False, "reachable": False, "exhausted": True, "unreachable_layers": [3, "sum"]},
            "rt_s2": {**base, "pivot": True}, "rt_s3": base}
    fg = rn.family_gates(cons, list(cons))
    assert fg["G-PIVOT"]["share"] == 0.5 and fg["G-EXHAUST"]["share"] == 0.5
    assert not fg["G-REACH"]["pass"] and fg["G-REACH"]["unreachable_layers"] == [3, "sum"]
    assert fg["G-REACH"]["unreachable_draws"] == [1]


def test_budget_from_parent_cmdline():
    raw = b"timeout\x00--signal=KILL\x002400s\x00/opt/pg/venv/bin/python\x00-m\x00" \
          b"infra.aws_pg.jobs.realizednull_pg1\x00--smoke\x00--out\x00/work/out\x00"
    assert rn.budget_s_from_cmdline(raw) == 2400
    with pytest.raises(AssertionError):
        rn.budget_s_from_cmdline(b"bash\x00run_job.sh\x00")


def test_preflight_bands_match_spec():
    R = list(range(98))
    want = {"band": (0.149161, 0.155250), "em_s0": (0.444819, 0.462975), "em_s1": (0.321908, 0.335048),
            "em_s2": (0.532018, 0.553733), "spec_s0": (0.435909, 0.453702), "spec_s1": (0.263437, 0.274189),
            "spec_s2": (0.390815, 0.406767), "topenergy32": (6.076418, 6.324435),
            "topsigma32": (4.015479, 4.179377), "anti32": (3.997623, 4.160791)}
    summary = {arm: {"dnll_w0_97": (lo + hi) / 2} for arm, (lo, hi) in want.items()}
    g = rn.pre_gate(summary, list(want), R)
    for arm, band in want.items():
        assert g["arms"][arm]["band"] == pytest.approx(list(band), abs=1e-6)
    assert g["pass"]
    summary["band"]["dnll_w0_97"] = 0.1560
    assert not rn.pre_gate(summary, ["band"], R)["pass"]
    assert rn.pre_gate(summary, ["em_s0"], R)["pass"]
    assert rn.pre1_gate(22.03, R)["pass"]
    for bad in (36.8, 22.08, 21.98):                      # 36.8: the sdpa failure mode
        assert not rn.pre1_gate(bad, R)["pass"]


# ---------------- integration with N1: every `# N1:` call site on inputs N3 builds ----------------

N_TOY, VOCAB_TOY, K_TOY = 16, 1000, 2
KW_TOY = {"k": K_TOY, "n_grid": 8, "n_probe": 8, "tol_sum": 0.005, "tol_layer": 0.05, "heavy_frac": 0.01,
          "max_resample": 2000, "max_closing": 50, "max_redraws": 3}
THE, A = ord("e"), ord("a")                              # the text probe's "▁the" / "▁a" stand-ins


class _Attn(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (torch.nn.Linear(n, n, bias=False) for _ in range(4))


class _Layer(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.self_attn = _Attn(n)


class _Toy(torch.nn.Module):
    """18 one-head attention layers of width N_TOY on a token embedding, called as PaliGemma is. A 2-D
    attention_mask (the processor's) is read as causal; a 4-D float mask (forced_mask) is added to the
    scores as given. Token id+1 gets +8 on the logits, so top-1 is right where the next character is the
    successor (the "abcdefgh" run) and wrong elsewhere."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(VOCAB_TOY, N_TOY)
        self.layers = torch.nn.ModuleList(_Layer(N_TOY) for _ in range(rn.N_LAYERS))
        self.out = torch.nn.Linear(N_TOY, VOCAB_TOY, bias=False)

    def forward(self, input_ids, attention_mask, pixel_values):
        h = self.emb(input_ids)
        S = h.shape[1]
        add = (torch.full((S, S), float("-inf")).triu(1) if attention_mask.dim() == 2
               else attention_mask[0, 0].float())
        for layer in self.layers:
            a = layer.self_attn
            w = torch.softmax(a.q_proj(h) @ a.k_proj(h).transpose(-1, -2) / N_TOY ** 0.5 + add, dim=-1)
            h = h + a.o_proj(w @ a.v_proj(h))
        succ = torch.nn.functional.one_hot((input_ids + 1) % VOCAB_TOY, VOCAB_TOY)
        return SimpleNamespace(logits=self.out(h) + 8.0 * succ)


def _toy_texts(n):
    rng = np.random.default_rng(5)
    letters = list("bcdfghjklmnpqrstvwz")
    return ["".join(rng.choice(letters, 40 + 4 * w)) + "abcdefgh" + "".join(rng.choice(list("eiouy "), 40)) + "q"
            for w in range(n)]


@pytest.fixture(scope="module")
def toy():
    """main()'s pipeline from the probes to verification and the D Gram, on the toy model: 6 windows
    (C = 0-3, D = 4-5, the smoke's split), a k = 2 band drawn inside the top-6 eigenvectors of Σ_all, two
    draws of every constructed family. In this toy the rtall and emt draws match and the other families are
    unreachable at some heavy layer, so both paths run."""
    tok = _Tok()
    model = _Toy().eval()
    pg = rn.PG1(model, model.layers, "cpu")
    units, trimmed = rn.make_units(_toy_texts(6), tok, _Proc(tok), image=None)
    assert trimmed == [] and all(u["pos"] == rn.N_PREFIX for u in units)
    sp = rn.splits(6, 4)
    uvec = torch.as_tensor(rn.probe_vector(N_TOY), dtype=torch.float32)
    rec_probe = {"native": rn.probe_mode(pg, units[0], "native", THE, A, uvec),
                 "forced": {m: rn.probe_mode(pg, units[0], m, THE, A, uvec) for m in rn.FORCED}}
    gC, nC = rn.gram_pass(pg, units, sp["C"], "native", class_sums=True)
    gD, _ = rn.gram_pass(pg, units, sp["D"], "native", class_sums=False)
    W64 = [layer.self_attn.q_proj.weight.detach().double().numpy() for layer in model.layers]
    Ms = [W.T @ W for W in W64]
    ecT, ecA = rn.eigen_sets(gC, iter(W64))
    rng = np.random.default_rng(7)
    band = [e["V"][:, :6] @ G.haar_frame(6, K_TOY, rng) for e in ecT]
    fam_in = rn.quantity_inputs(ecT, ecA)
    tgt = rn.band_targets(fam_in, band)
    grids = rn.quantity_grids(fam_in, sorted(tgt), KW_TOY)
    cons, bases, names = {}, {}, {}
    for fam in rn.FAMILIES:
        recs, b, _ = rn.construct_family(fam, [0, 1], fam_in, tgt, grids, KW_TOY)
        cons.update(recs)
        bases.update(b)
        names[fam] = list(recs)
    hookq = rn.verify_pass(pg, units, sp["C"], "native", rn.verify_entries(band, bases, cons))
    return SimpleNamespace(tok=tok, model=model, pg=pg, units=units, sp=sp, rec_probe=rec_probe, gC=gC, nC=nC,
                           gD=gD, W64=W64, Ms=Ms, ecT=ecT, ecA=ecA, VT=[e["V"] for e in ecT], band=band,
                           fam_in=fam_in, tgt=tgt, grids=grids, cons=cons, bases=bases, names=names, hookq=hookq)


def test_n1_mask_probe_verdict_on_the_toy_masks(toy):
    """mask_probe_verdict on probe_mode's numbers: the toy's native mask reads causal, and each forced mask
    passes its own self-test, as main() records it; probe_numbers carries exactly N1's four arguments."""
    rp = toy.rec_probe
    assert rp["native"]["mode"] == "causal" and not rp["native"]["image_sees_text"]
    assert {m: r["mode"] for m, r in rp["forced"].items()} == {m: m for m in rn.FORCED}
    assert rp["forced"]["bidir"]["image_sees_text"] and rp["forced"]["prefixlm"]["r_prior"] >= 1e-3
    mp = rn.probe_numbers(rp, "on")
    assert R.mask_probe_verdict(**mp["native"]) == "causal"
    assert {m: R.mask_probe_verdict(**v) for m, v in mp["forced"].items()} == {m: m for m in R.MODES}


def _capture(toy, L, windows):
    """Every q_proj input row at layer L over the windows, with its class (prefix position, or the text
    block) and whether it is a scored text position."""
    H, cls, inT = [], [], []
    hk = toy.model.layers[L].self_attn.q_proj.register_forward_pre_hook(
        lambda _m, a: H.append(a[0][0].double().clone()))
    with torch.inference_mode():
        for w in windows:
            u = toy.units[w]
            toy.pg.forward(u, "native")
            S, p = u["input_ids"].shape[1], u["pos"]
            cls += [min(t, p) for t in range(S)]
            inT += [p <= t < p + u["n_text"] - 1 for t in range(S)]
    hk.remove()
    return torch.cat(H).numpy(), np.array(cls), np.array(inT)


def test_n1_class_means_eigencoords_and_arm_stats_call_sites(toy):
    """class_means on class_inputs, eigencoords on eigen_sets' inputs, arm_stats on arm_stats_grams and
    arm_stats_means: the returned keys N3 reads exist, and the numbers equal direct sums over the hook's
    rows at one layer (class means, centered energy share, realized perturbation, denominators)."""
    sums, counts = rn.class_inputs(toy.gC, len(toy.sp["C"]), toy.nC["text"])
    means = [R.class_means(sums[L], counts) for L in range(rn.N_LAYERS)]
    mu = [rn.class_rows(m) for m in means]
    assert mu[0].shape == (rn.N_PREFIX + 1, N_TOY)
    assert {"V", "mu", "Mp", "Sp", "denom"} <= set(toy.ecT[0]) and {"V", "Mp", "Sp"} <= set(toy.ecA[0])
    gramsC = rn.arm_stats_grams(toy.gC, toy.VT, toy.Ms)
    gramsD = rn.arm_stats_grams(toy.gD, toy.VT, toy.Ms)
    assert gramsC["denom_T"] == pytest.approx([e["denom"] for e in toy.ecT], rel=1e-12)
    assert gramsC["denom_all"] == pytest.approx([e["denom"] for e in toy.ecA], rel=1e-12)
    meansC = rn.arm_stats_means(mu, counts, toy.gC, toy.nC["T"])
    stC = R.arm_stats(toy.band, gramsC, toy.W64, meansC)
    stD = R.arm_stats(toy.band, gramsD, toy.W64, None)
    cols = {"e_all", "e_T", "E_T", "N_T", "rho_T", "N_all", "rho_all", "w_top1", "w_top5", "leverage_max", "gain"}
    assert cols | {"e_all_centered", "N_T_centered"} <= set(stC) and cols <= set(stD)
    assert "e_all_centered" not in stD
    with pytest.raises(ValueError):                       # main() never calls it for ident_*
        R.arm_stats([np.zeros((N_TOY, 0))] * rn.N_LAYERS, gramsC, toy.W64, None)
    L = 5
    H, cls, inT = _capture(toy, L, toy.sp["C"])
    mu_true = np.stack([H[cls == c].mean(axis=0) for c in range(rn.N_PREFIX + 1)])
    assert np.allclose(mu[L], mu_true, rtol=1e-4, atol=1e-5)
    U, W = toy.band[L], toy.W64[L]
    Hc = H - mu_true[cls]
    assert stC["e_all_centered"][L] == pytest.approx(((Hc @ U) ** 2).sum() / (Hc ** 2).sum(), rel=1e-4)
    HT = H[inT]
    assert stC["N_T"][L] == pytest.approx(((HT @ U @ U.T @ W.T) ** 2).sum(), rel=1e-4)
    assert stC["N_T_centered"][L] == pytest.approx((((HT - mu_true[-1]) @ U @ U.T @ W.T) ** 2).sum(), rel=1e-4)
    assert stC["rho_T"][L] == pytest.approx(stC["N_T"][L] / ((HT @ W.T) ** 2).sum(), rel=1e-4)
    HD, _, inTD = _capture(toy, L, toy.sp["D"])
    assert stD["rho_all"][L] == pytest.approx(((HD @ U @ U.T @ W.T) ** 2).sum() / ((HD @ W.T) ** 2).sum(),
                                              rel=1e-4)


def test_n1_realized_quantity_reach_grid_construct_draw_call_sites(toy):
    """band_targets, quantity_grids and construct_family call N1 as main() does: grids per quantity, seeds
    190_000 + L and the §3.2 base seeds; each record keeps every construct_draw key but Ue; a matched
    draw's basis, mapped to model coordinates, reproduces its constructed N through the hooks (G-MATCH)."""
    for q, t in toy.tgt.items():
        assert t.shape == (rn.N_LAYERS,) and np.all(t > 0)
        assert toy.hookq[f"band|{q}"] == pytest.approx(t, rel=1e-4)          # Gram == hooks for the band
    for q, g in toy.grids.items():
        assert len(g) == rn.N_LAYERS and {"m", "median_top", "median_bottom"} <= set(g[0])
        Mp, Sp, _ = toy.fam_in[q]
        again = R.reach_grid(Mp[3], Sp[3], k=K_TOY, n_grid=8, n_probe=8, seed=R.GRID_SEED_BASE + 3)
        assert all(np.array_equal(again[key], g[3][key]) for key in again)
    want = {"N", "N_sum", "family", "m", "draws", "pivot", "matched", "reachable", "unreachable_layers",
            "redraw", "exhausted", "seconds", "fail_reason", "family_name", "slot", "base_seed", "wall_seconds"}
    for fam, names in toy.names.items():
        assert names == [f"{fam}_s0", f"{fam}_s1"]
        assert [toy.cons[n]["base_seed"] for n in names] == [rn.FAMILIES[fam]["seed"] + 1000 * s for s in (0, 1)]
        for n in names:
            assert set(toy.cons[n]) == want
        rows = rn.verify_rows(toy.hookq, names, fam)
        for n, r in zip(names, rows):
            assert (r is None) == (n not in toy.bases) == (not toy.cons[n]["matched"])
            if r is not None:
                assert r["N"] == pytest.approx(toy.cons[n]["N"], rel=1e-4)
                assert rn.match_gate(r["N"], r["targets"], toy.sp["C"])["pass"]
    matched = sorted(n for n, r in toy.cons.items() if r["matched"])
    assert 0 < len(matched) < len(toy.cons), matched                     # both paths ran
    for r in toy.cons.values():
        assert r["matched"] or (r["exhausted"] and r["fail_reason"] and r["N"] is None)


def test_n1_sample_record_call_site(toy, tmp_path, monkeypatch):
    """sample_payload hands N1 sample_record a window built by sample_window and per-position arrays from
    PG1.scores: each window record has its role, the input as seen, N1's NLL sums equal N3's readout, and
    the file has a right and a wrong example."""
    monkeypatch.setattr(rn, "VOCAB", VOCAB_TOY)
    sp, units, tok = toy.sp, toy.units, toy.tok
    none = toy.pg.scores(units, range(6), "native")
    state = {}
    U = [torch.as_tensor(b, dtype=torch.float32) for b in toy.band]
    hooks = [(layer.self_attn.q_proj, rn.ablation_hook(U[L], "zero", None, state))
             for L, layer in enumerate(toy.model.layers)]
    arm = toy.pg.scores(units, range(6), "native", hooks, state, set())
    sink = rn.Sink(tmp_path, "smoke", now=lambda: T0)
    rec = rn.Recorder(sink, sp, ["none", "band"])
    rec.nll("none", "none", none)
    rec.nll("band", "zero", arm)
    sel, Y = rec.select(arm)
    targets = [u["win_ids"][1:].numpy() for u in units]
    pay = rn.sample_payload({"arm": "band"}, sel["open"], arm, rec.none, units, targets, tok)
    assert [w["role"] for w in pay["windows"]] == ["max", "min", "common"]
    for w in pay["windows"]:
        i = w["index"]
        assert w["text_seen"] == f"<image>×{rn.N_IMAGE} " + chr(2) + units[i]["text"] + "\n"
        assert w["ntok"] == units[i]["n_text"] - 1 == len(w["positions"])
        assert w["nll_sum_none"] == pytest.approx(none[i]["nll"], rel=1e-5)
        assert w["nll_sum_arm"] == pytest.approx(arm[i]["nll"], rel=1e-5)
        assert w["Y"] == pytest.approx(Y[i], rel=1e-4, abs=1e-6)
        assert w["right_example"]["top1_arm_is_target"] and not w["wrong_example"]["top1_arm_is_target"]
    assert pay["examples"]["none"]["right"]["right_none"] and not pay["examples"]["none"]["wrong"]["right_none"]
    rec.write_samples("band", pay, None)
    assert json.loads((tmp_path / "samples/smoke/band.json").read_text())["windows"][2]["index"] == sp["common_open"]


def test_n1_provenance_and_env_keys(tmp_path, monkeypatch):
    """main() reads provenance() through _n1() and stamps env.json with N1's key names, which G-ENV reads."""
    p = tmp_path / "realizednull_provenance.json"
    p.write_text(json.dumps({"program": "realizednull", "commit_sha": "c" * 40}))
    monkeypatch.setattr(R, "PROVENANCE_FILE", p)
    assert rn._n1() is R and rn._n1().provenance()["commit_sha"] == "c" * 40
    env = {"transformers": "5.9.0", "torch": "2.12.0", "numpy": "2.4.6", "cuda": "13.0", "gpu_name": "NVIDIA L4",
           "instance_type": "g6.xlarge", "_attn_implementation": "eager", "mask": "native"}
    assert rn.env_gate(env)["pass"] and set(rn.env_gate(env)) >= {"missing", "_attn_implementation"}
    assert not rn.env_gate({**env, "_attn_implementation": "sdpa"})["pass"]
    assert rn.env_gate({**env, "gpu_name": None})["missing"] == ["gpu_name"]
    src = inspect.getsource(rn.main)
    assert '"gpu_name":' in src and '"_attn_implementation":' in src and "R.ENV_KEYS" in src


def _recorder(sink, sp):
    """A Recorder whose arms (none, every G-PRE2-4 arm, the identity pair, the positive controls) carry
    synthetic per-window NLL on the smoke's windows."""
    arms = ["none", *R.PRE_ARMS, "ident_zero", "band_dup", "ident_mean", "topsigma32_mean", "haar_mean_s0"]
    rec = rn.Recorder(sink, sp, arms)
    for k, arm in enumerate(arms):
        rec.nll(arm, "zero", {w: {"nll": 1500.0 + 7.0 * k + w, "ntok": 511} for w in sp["R"]})
    return rec


def test_job_metrics_is_n1_gates_input(toy, tmp_path):
    """metrics.json → job_metrics, assembled as main() assembles it and read back from the file, is accepted
    by N1's gates(); N1's G-ENV, G-MASK, G-PRE0, G-PRE2, G-REACH, G-PIVOT, G-EXHAUST, G-MATCH and G-GEN agree
    with N3's own records; a missing identity arm makes N1's G-READOUT fire."""
    sp, C_ = toy.sp, toy.sp["C"]
    sink = rn.Sink(tmp_path, "smoke", now=lambda: T0)
    rec = _recorder(sink, sp)
    gramsD = rn.arm_stats_grams(toy.gD, toy.VT, toy.Ms)
    opD = {"N_T": (toy.Ms, gramsD["T"]), "N_all": (toy.Ms, gramsD["all"]),
           "E_T": ([np.eye(N_TOY)] * rn.N_LAYERS, gramsD["T"])}
    bandD = {q: rn.summed_quantity(toy.band, *opD[q]) for q in opD}
    env = {"transformers": "5.9.0", "torch": "2.12.0", "numpy": "2.4.6", "cuda": "13.0", "gpu_name": "NVIDIA L4",
           "instance_type": "g6.xlarge", "_attn_implementation": "eager", "mask": "native", "args": {"smoke": True}}
    gates = {"G-ENV": rn.env_gate(env), "G-EMATCH": {"draws": {}, "n": 1, "n_pass": 1},
             "G-MATCH": {}, "G-GEN": {}, "G-REACH": {}, "G-PIVOT": {}, "G-EXHAUST": {}}
    jm = {"env": {k: env[k] for k in R.ENV_KEYS + ("_attn_implementation",)}, "mask": "native",
          "mask_probe": rn.probe_numbers(toy.rec_probe, "on"),
          "ematch": [None] * 3 + [[{"e_target": 0.02, "e_realized": 0.0205}] * rn.N_LAYERS],
          "construction": {f: [toy.cons[n] for n in names] for f, names in toy.names.items()},
          "verify_C": {}, "gen_D": {}}
    for fam, names in toy.names.items():
        for gid, v in rn.family_gates(toy.cons, names).items():
            gates[gid][fam] = v
        vrows = jm["verify_C"][fam] = rn.verify_rows(toy.hookq, names, fam)
        grows = jm["gen_D"][fam] = rn.gen_rows(names, toy.bases, fam, opD, bandD)
        gates["G-MATCH"][fam] = {"draws": {n: rn.match_gate(r["N"], r["targets"], C_)
                                           for n, r in zip(names, vrows) if r is not None}, "n": len(names)}
        gates["G-GEN"][fam] = {"draws": {n: {"pass": abs(r["N_sum"] / r["target_sum"] - 1) <= 0.05}
                                         for n, r in zip(names, grows) if r is not None}, "n": len(names)}
        for gid in ("G-MATCH", "G-GEN"):
            gates[gid][fam]["n_pass"] = sum(d["pass"] for d in gates[gid][fam]["draws"].values())
    gb1_nll = {"per_window_by_site": {"q": {rn.GB1_ARM[arm]: {"nll": [rec.nll_open[arm][w] for w in sp["R"]]}
                                            for arm in rn.PRE0_ARMS}}}
    jm["pre0"] = rn.pre0_input(rec.nll_open, gb1_nll, sp["R"])
    gates["G-PRE0"] = rn.pre0_rows(rec.nll_open, gb1_nll, sp["R"])
    gates["G-PRE2"] = rn.pre_gate(rec.summary, ["band"], sp["R"])
    dc = {"c_D": {"band": 0.15, "band_dup": 0.15, "ident_mean": 0.0, "topenergy32": 6.2, "topsigma32_mean": 2.0},
          "c_D_draws": {"haar": [0.006, 0.009], "haar_mean": [0.004, 0.005], "em": [0.45] * 4,
                        "em_mean": [0.30] * 4, **{f: [0.15] * 2 for f in R.CONSTRUCTED}},
          "Y_band_mean_D": [0.10, 0.11, 0.09, 0.10], "id_max": {"ident_zero": 0.0, "ident_mean": 0.0},
          "id_windows": "0-97", "boot": {s: {"n_degenerate": 0, "n_boot": 10000} for s in "EMRT"},
          "f_H2": 0.05, "noleak": {"leaks": []}, "planned_draws": {s: 2 for s in "EMRT"}}

    def n1_gates():
        sink.json("metrics.json", rn.metrics_doc(rec, gates, {"smoke": True}, jm))     # the job's writer
        got = json.loads((tmp_path / "metrics.json").read_text())["job_metrics"]
        assert "run" not in got and "pre1_reference" not in got
        return R.gates({**got, "run": "R1", "pre1_reference": None}, dc)

    g = n1_gates()
    assert g["G-ENV"]["pass"] and gates["G-ENV"]["pass"]
    assert g["G-MASK"]["value"]["native"] == toy.rec_probe["native"]["mode"] == "causal"
    assert all(g["G-MASK"]["value"]["selftest"].values()) and not g["G-MASK"]["mask_unknown"]
    assert g["G-PRE0"]["pass"] and gates["G-PRE0"]["pass"]
    assert g["G-PRE2"]["value"]["band"]["dnll"] == pytest.approx(gates["G-PRE2"]["arms"]["band"]["dnll_w0_5"])
    assert g["G-EMATCH"]["pass"]
    for fam, names in toy.names.items():
        assert g[f"G-REACH:{fam}"]["pass"] == gates["G-REACH"][fam]["pass"]
        assert g[f"G-REACH:{fam}"]["value"]["unreachable_layers"] == gates["G-REACH"][fam]["unreachable_layers"]
        assert g[f"G-PIVOT:{fam}"]["value"] == gates["G-PIVOT"][fam]["share"]
        assert g[f"G-EXHAUST:{fam}"]["value"] == gates["G-EXHAUST"][fam]["share"]
        for gid in ("G-MATCH", "G-GEN"):
            fails = [i for i, n in enumerate(names) if n in gates[gid][fam]["draws"]
                     and not gates[gid][fam]["draws"][n]["pass"]]
            assert g[f"{gid}:{fam}"]["value"] == fails, (gid, fam)
    assert all(g[f"G-READOUT:{s}"]["pass"] for s in "EMRT")
    rec.summary.pop("ident_mean")                         # the identity pair is now missing a number
    g = n1_gates()
    assert not any(g[f"G-READOUT:{s}"]["pass"] for s in "EMRT")


class _BeforeModel(Exception):
    """Raised in place of hf_setup(), main()'s first step that needs the box."""


def test_smoke_main_parses_and_writes_start_marker_without_a_model(tmp_path, monkeypatch):
    """main() with --smoke as run_job.sh launches it: the budget from the parent `timeout` command line,
    the smoke's sample directory, and `start` as the first event, before any model step."""
    raw = (b"timeout\x00--signal=KILL\x002400s\x00/opt/pg/venv/bin/python\x00-m\x00"
           b"infra.aws_pg.jobs.realizednull_pg1\x00--smoke\x00--out\x00" + str(tmp_path).encode() + b"\x00")
    monkeypatch.setattr(rn, "parent_cmdline", lambda: raw)

    def before_model():
        raise _BeforeModel
    monkeypatch.setattr(rn, "hf_setup", before_model)
    argv = [x.decode() for x in raw.split(b"\x00")[6:-1]]                # --smoke --out <dir>
    with pytest.raises(_BeforeModel):
        rn.main(argv)
    rows = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["start"] and rows[0]["budget_s"] == 2400
    assert (tmp_path / "samples/smoke").is_dir() and (tmp_path / "sealed/samples").is_dir()
    assert not (tmp_path / "samples/discovery").exists()


def test_gram_only_flags_and_gencal_writer_on_tiny_grams(tmp_path, monkeypatch):
    """--gram-only parses to the registered draw counts and refuses the smoke, a top-up and a forced mask; main()
    writes `start` and makes no samples/ or sealed/ directory. On tiny synthetic Grams over GRAM_BLOCKS:
    gen_quantities equals Σ_t ||W U Uᵀ h_t||² over the rows, gencal_entry's summed ratio is gen_rows' on the same
    windows, a draw at 0.96 passes G-GEN and one at 0.94 does not, G-DRAWS counts matched, G-MATCH and G-GEN
    draws, C = 0-97 has no held-out set, and the writer refuses a window set reaching 98."""
    a = rn.parse_args(["--out", "x", "--gram-only"])
    assert a.gram_only and not rn.parse_args(["--out", "x"]).gram_only
    assert {f: len(rn.draw_slots(a)[f]) for f in rn.FAMILIES} == {"rt": 40, "rtall": 5, "rthalf": 3, "rt2x": 3,
                                                                  "emt": 40}
    for bad in (["--smoke"], ["--mask", "causal"], ["--topup-families", "rt", "--slot-offset", "40"]):
        with pytest.raises(AssertionError):
            rn.parse_args(["--out", "x", "--gram-only", *bad])
    out = tmp_path / "out"
    raw = (b"timeout\x00--signal=KILL\x002700s\x00/opt/pg/venv/bin/python\x00-m\x00"
           b"infra.aws_pg.jobs.realizednull_pg1\x00--gram-only\x00--out\x00" + str(out).encode() + b"\x00")
    monkeypatch.setattr(rn, "parent_cmdline", lambda: raw)

    def before_model():
        raise _BeforeModel
    monkeypatch.setattr(rn, "hf_setup", before_model)
    with pytest.raises(_BeforeModel):
        rn.main(["--gram-only", "--out", str(out)])
    assert [json.loads(x)["stage"] for x in (out / "events.jsonl").read_text().splitlines()] == ["start"]
    assert not (out / "sealed").exists() and not (out / "samples").exists()

    n, k, nL = 6, 2, rn.N_LAYERS
    rng = np.random.default_rng(11)
    H = [[rng.standard_normal((20, n)) for _ in range(nL)] for _ in rn.GRAM_BLOCKS]   # rows per block, layer
    inT = slice(5, 15)                                                                # scored text rows
    blocks = [{"all": [h.T @ h for h in Hb], "T": [h[inT].T @ h[inT] for h in Hb]} for Hb in H]
    Ws = [rng.standard_normal((n, n)) for _ in range(nL)]
    Ms = [W.T @ W for W in Ws]
    band = [np.linalg.qr(rng.standard_normal((n, k)))[0] for _ in range(nL)]
    scale = {"rt_s0": 1.0, "rt_s1": 0.96, "rt_s2": 0.94, "rt_s4": 1.0, "rthalf_s0": 0.5, "emt_s0": 1.0}
    bases = {x: [U * c ** 0.25 for U in band] for x, c in scale.items()}             # N(c^¼ U) = c N(U)
    ok = {"pivot": False, "matched": True, "reachable": True, "exhausted": False, "unreachable_layers": []}
    cons = {x: {**ok, "family_name": x.rsplit("_s", 1)[0]} for x in [*scale, "rt_s3"]}
    cons["rt_s3"].update(matched=False, reachable=False, exhausted=True, unreachable_layers=[17])
    fam_draws = {"rt": [f"rt_s{s}" for s in range(5)], "rtall": [], "rthalf": ["rthalf_s0"], "rt2x": [],
                 "emt": ["emt_s0"]}
    hookq = {"band|N_T": np.ones(nL), "band|N_all": np.ones(nL), "band|E_T": np.ones(nL),
             **{x: rn.FAMILIES[cons[x]["family_name"]]["dose"] * np.ones(nL) for x in bases}}
    hookq["rt_s4"] = 1.02 * np.ones(nL)                                              # G-MATCH fails
    C_w = list(range(32))
    sets = rn.gen_sets(32)
    assert sets == [list(range(98)), list(range(32, 98))] and rn.gen_sets(98) == [list(range(98))]
    gen = {rn.wkey("D", ws): rn.gen_quantities(bases, cons, band, Ms, rn.add_grams(
        [b for b, r in zip(blocks, rn.GRAM_BLOCKS) if r.start >= ws[0]])) for ws in sets}
    L = 4
    HT = np.concatenate([H[b][L][inT] for b in (1, 2)])
    assert gen["D_w32_97"]["band"]["N_T"][L] == pytest.approx(((HT @ band[L] @ band[L].T @ Ws[L].T) ** 2).sum(),
                                                             rel=1e-9)
    assert gen["D_w32_97"]["band"]["E_T"][L] == pytest.approx(((HT @ band[L]) ** 2).sum(), rel=1e-9)
    gD = rn.arm_stats_grams(rn.add_grams(blocks[1:]), None, Ms)
    opD = {"N_T": (Ms, gD["T"]), "N_all": (Ms, gD["all"]), "E_T": ([np.eye(n)] * nL, gD["T"])}
    rows = rn.gen_rows(fam_draws["rt"], bases, "rt", opD, {q: rn.summed_quantity(band, *opD[q]) for q in opD})

    entry = rn.gencal_entry(C_w, fam_draws, cons, hookq, gen)
    rt = entry["families"]["rt"]
    assert (rt["attempted"], rt["matched"], rt["G-MATCH"]["n_pass"]) == (5, 4, 3)
    assert rt["G-REACH"]["unreachable_layers"] == [17] and set(entry["families"]) == {"rt", "rthalf", "emt"}
    g = rt["G-GEN"]["D_w32_97"]["draws"]
    assert [g[x]["ratio_sum"] for x in g] == [r["N_sum"] / r["target_sum"] for r in rows if r is not None]
    assert [g[x]["pass"] for x in ("rt_s0", "rt_s1", "rt_s2", "rt_s4")] == [True, True, False, True]
    assert g["rt_s2"]["ratio_sum"] == pytest.approx(0.94) and g["rt_s2"]["worst_heavy_ratio"] == pytest.approx(0.94)
    assert rt["G-GEN"]["D_w32_97"]["worst_ratio_sum"] == g["rt_s2"]["ratio_sum"]
    assert entry["families"]["rthalf"]["G-GEN"]["D_w0_97"]["draws"]["rthalf_s0"]["ratio_sum"] == pytest.approx(1.0)
    assert rt["valid"] == {"D_w0_97": ["rt_s0", "rt_s1"], "D_w32_97": ["rt_s0", "rt_s1"]}
    assert entry["G-DRAWS"]["R"]["D_w32_97"] == {"valid": 2, "planned": 5, "pass": False}         # 2 < 0.8 × 5
    assert entry["G-DRAWS"]["T"]["D_w32_97"] == {"valid": 1, "planned": 1, "pass": True}
    sink = rn.Sink(tmp_path / "w", None, now=lambda: T0)
    sink.json("gencal.json", {"by_C": {rn.wkey("C", C_w): entry}})
    back = json.loads((tmp_path / "w/gencal.json").read_text())["by_C"]["C_w0_31"]
    assert back["G-DRAWS"] == entry["G-DRAWS"] and not (tmp_path / "w/sealed").exists()
    leak = rn.gencal_entry(C_w, fam_draws, cons, hookq, {rn.wkey("D", range(32, 99)): gen["D_w32_97"]})
    with pytest.raises(AssertionError, match="G-NOLEAK"):
        sink.json("gencal.json", leak)
