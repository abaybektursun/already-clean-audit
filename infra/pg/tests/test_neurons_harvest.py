"""Offline gate for the D1 harvest (CLAUDE.md "Before any compute" 4): no weights, no downloads.

A tiny random LlamaForCausalLM (hidden 32, MLP 64, 2 layers, vocab 256) on CPU in fp32, a byte tokenizer, a
synthetic corpus of 12 documents. Everything the GPU job writes is checked against an independent read:
the model's own down_proj input captured by a separate hook over the same batches.
"""
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from infra.pg import neurons_harvest as NH
from infra.pg.matrix_models import vocab_sha256

WINDOW, BATCH, N_WINDOWS = 16, 2, 16
K = NH.K_TOP


class ByteTok:
    bos_token_id, eos_token_id = 1, 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3 + b % 253 for b in text.encode()]}

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(33 + int(i) % 90) for i in ids)

    def get_vocab(self):
        return {f"t{i}": i for i in range(256)}


@pytest.fixture(scope="module")
def model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, vocab_size=256, max_position_embeddings=64, initializer_range=0.2)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def corpus():
    rng = np.random.default_rng(0)
    docs = [rng.integers(3, 256, size=int(rng.integers(10, 40))).tolist() for _ in range(12)]
    lens = np.array([len(d) + 1 for d in docs])
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    tokens = np.concatenate([d + [2] for d in docs]).astype(np.int32)
    assert tokens.size >= WINDOW * N_WINDOWS
    return {"tokname": "byte", "tokens": tokens, "doc_offsets": offsets, "manifest_sha256": "test",
            "tokens_sha256": "test", "record": {"vocab_sha256": vocab_sha256(ByteTok())},
            "band_end": {1: 70, 2: 150, 3: int(tokens.size)}}


def harvest(tmp, model, corpus, **kw):
    return NH.run(tmp, model=model, tok=ByteTok(), corpus=corpus, model_key="test", repo="tiny", revision="0",
                  max_tokens=WINDOW * N_WINDOWS, window=WINDOW, batch=BATCH, device="cpu", nll_tokens=100, **kw)


def brute_force(model, corpus):
    """[n_windowed, width] per layer: the model's own down_proj input, BOS dropped, same batches as the job;
    row index = global token index (window position 0 included; the caller masks it)."""
    caps = {}
    hs = [lay.mlp.down_proj.register_forward_pre_hook(
        lambda m, a, lyr=lyr: caps.setdefault(lyr, []).append(a[0][:, 1:].reshape(-1, a[0].shape[-1])))
        for lyr, lay in enumerate(model.model.layers)]
    tokens = corpus["tokens"]
    with torch.inference_mode():
        for w0 in range(0, N_WINDOWS, BATCH):
            x = torch.from_numpy(tokens[w0 * WINDOW:(w0 + BATCH) * WINDOW].astype(np.int64)).view(BATCH, WINDOW)
            x = torch.cat([torch.ones(BATCH, 1, dtype=torch.long), x], 1)
            NH.forward(model, x, False)
    for h in hs:
        h.remove()
    return {lyr: torch.cat(v).numpy() for lyr, v in caps.items()}


def exact_topk(A, keep):
    """Brute force over the rows `keep`: per neuron and pole the K largest |a| of that sign, as (tok, a)."""
    rows = np.flatnonzero(keep)
    out_tok = np.full((A.shape[1], 2, K), -1)
    out_a = np.full((A.shape[1], 2, K), np.nan)
    for n in range(A.shape[1]):
        for s, sign in ((0, 1), (1, -1)):
            idx = [i for i in rows[np.argsort(-sign * A[rows, n], kind="stable")] if sign * A[i, n] > 0][:K]
            out_tok[n, s, :len(idx)] = idx
            out_a[n, s, :len(idx)] = A[idx, n]
    return out_tok, out_a


@pytest.fixture(scope="module")
def run1(tmp_path_factory, model, corpus):
    out = tmp_path_factory.mktemp("run1")
    harvest(out, model, corpus, sample_neurons=4, seed=0, snapshot_bands=(1, 2))
    return out


def test_topk_equals_brute_force_and_band_snapshots(run1, model, corpus):
    A = brute_force(model, corpus)
    t = np.arange(WINDOW * N_WINDOWS)
    for lyr in range(2):
        for root, n_tok in ((run1, WINDOW * N_WINDOWS), (run1 / "snapshots" / "band1", 70),
                            (run1 / "snapshots" / "band2", 150)):
            keep = (t < n_tok) & (t % WINDOW != 0)                   # window position 0 excluded
            got = np.load(root / "topk" / f"layer_{lyr:02d}.npz")
            want_tok, want_a = exact_topk(A[lyr], keep)
            assert np.array_equal(got["tok"], want_tok), (root, lyr)
            np.testing.assert_allclose(got["a"], want_a, rtol=1e-5, atol=1e-6)
            assert got["tok"].max() < n_tok and not (got["tok"][got["tok"] >= 0] % WINDOW == 0).any()
            st = np.load(root / "stats" / f"layer_{lyr:02d}.npz")
            a = A[lyr][keep]
            np.testing.assert_allclose(st["max_abs"][:, 0], np.clip(a.max(0), 0, None), rtol=1e-5)
            np.testing.assert_allclose(st["max_abs"][:, 1], np.clip(-a.min(0), 0, None), rtol=1e-5)
            assert np.array_equal(st["n_pole_tokens"][:, 0], (a > 0).sum(0))
            assert np.array_equal(st["n_pole_tokens"][:, 1], (a < 0).sum(0))
            # density brackets the exact count of |a| >= 0.01 max_abs within one histogram bin
            edges = st["hist_edges"]
            for s, pole in ((0, a * (a > 0)), (1, -a * (a < 0))):
                thr = 0.01 * st["max_abs"][:, s]
                lo = edges[NH.hist_bin(torch.from_numpy(thr)).numpy()]
                exact = (pole >= thr).sum(0)
                assert np.all(st["density"][:, s] >= exact) and np.all(st["density"][:, s] <= (pole >= lo).sum(0))


def test_pos0_share_is_the_pre_exclusion_topk_share(run1, model, corpus):
    """stats/pos0_share = share of the top-K over all tokens (position 0 included) that sat at position 0."""
    A = brute_force(model, corpus)
    t = np.arange(WINDOW * N_WINDOWS)
    seen_nonzero = False
    for lyr in range(2):
        for root, n_tok in ((run1, WINDOW * N_WINDOWS), (run1 / "snapshots" / "band1", 70)):
            full_tok, _ = exact_topk(A[lyr], t < n_tok)
            valid = full_tok >= 0
            want = ((full_tok % WINDOW == 0) & valid).sum(-1) / valid.sum(-1)
            got = np.load(root / "stats" / f"layer_{lyr:02d}.npz")["pos0_share"]
            assert got.dtype == np.float32 and got.shape == want.shape
            np.testing.assert_allclose(got, want, rtol=1e-6)
            seen_nonzero |= bool((got > 0).any())
    assert seen_nonzero, "no pole had position 0 in its pre-exclusion top-K: the test cannot fail"


def test_histogram_quantiles_bracket_exact():
    rng = np.random.default_rng(1)
    a = (rng.lognormal(0, 2, size=(200_000, 3)) * rng.choice([-1, 1], size=(200_000, 3))).astype(np.float32)
    h = NH.Hist(3, "cpu")
    for i in range(0, a.shape[0], 50_000):
        h.update(torch.from_numpy(a[i:i + 50_000]))
    hist, edges = h.array(), NH.hist_edges()
    step = edges[1] / edges[0]
    for q in (0.5, 0.99, 0.999, 0.9999):
        got = NH.hist_quantile_edge(hist, q, edges)
        for n in range(3):
            for s, vals in ((0, a[a[:, n] > 0, n]), (1, -a[a[:, n] < 0, n])):
                exact = np.quantile(vals, q, method="inverted_cdf")
                assert got[n, s] / step * (1 - 1e-5) <= exact <= got[n, s] * (1 + 1e-5), (q, n, s, exact, got[n, s])


def test_g0_passes_on_the_module_and_fails_on_a_wrong_activation(model, corpus, run1):
    x = torch.cat([torch.ones(2, 1, dtype=torch.long),
                   torch.from_numpy(corpus["tokens"][:32].astype(np.int64)).view(2, 16)], 1)
    tap = NH.MLPTap(model.model.layers)
    acts = [lay.mlp.act_fn for lay in model.model.layers]
    with torch.inference_mode():
        good = NH.g0_ratios(model, tap, acts, x)
        bad = NH.g0_ratios(model, tap, [NH.WRONG_ACT[NH.act_name(acts[0])]] * 2, x)
    assert NH.g0_pass(good) and max(good) < 1e-6
    assert not NH.g0_pass(bad), bad
    c = json.loads((run1 / "controls.json").read_text())["G0"]
    assert c["pass"] and not c["failing_input"]["pass"] and c["failing_input"]["n_layers_over_tol"] >= 1


def test_g0_second_failing_input_has_power_at_small_g():
    """At small |g| (early checkpoints: stage1-step1000 moved 3/32 layers) silu and gelu_tanh nearly agree,
    so the activation swap loses power; scaling u by 1.1 fails every layer whatever |g| is."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(1)
    small = LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=4, vocab_size=256,
                                         max_position_embeddings=64, initializer_range=0.002)).eval()
    tap = NH.MLPTap(small.model.layers)
    acts = [lay.mlp.act_fn for lay in small.model.layers]
    x = torch.randint(3, 256, (2, 17))
    with torch.inference_mode():
        good = NH.g0_ratios(small, tap, acts, x)
        swap = NH.g0_ratios(small, tap, [NH.WRONG_ACT["silu"]] * 2, x)
        scaled = NH.g0_ratios(small, tap, acts, x, u_scale=NH.WRONG_U_SCALE)
    assert NH.g0_pass(good)
    assert NH.g0_pass(swap), swap                               # the swap cannot fail here
    assert not NH.g0_pass(scaled) and min(scaled) > 5 * NH.G0_TOL, scaled


def test_drawn_poles_get_the_poles_directory(run1):
    """--sample-neurons without --poles: the drawn poles are the pole list for poles/."""
    draw = json.loads((run1 / "samples" / "sample_draw.json").read_text())
    assert json.loads((run1 / "poles" / "poles.json").read_text()) == draw["poles"] and not draw["from_poles_arg"]
    pos = np.load(run1 / "poles" / "positions.npz")["tok"]
    ex = np.load(run1 / "poles" / "exemplar_a.npz")
    assert np.array_equal(pos, topk_positions(run1, draw["poles"]))
    for j, p in enumerate(draw["poles"]):
        lyr, n, s = NH.parse_pole(p)
        tk = np.load(run1 / "topk" / f"layer_{lyr:02d}.npz")
        for key in ("a", "g", "u"):
            np.testing.assert_array_equal(ex[key][j], tk[key][n, s])
    assert sorted(json.loads((run1 / "poles" / "vocab.json").read_text())) == sorted(draw["poles"])
    top = np.load(run1 / "poles" / "top1000.npz")
    assert top["tok"].shape == (len(draw["poles"]), NH.K_POLES)
    np.testing.assert_array_equal(top["a"][:, :K], np.stack([ex["a"][j] for j in range(len(draw["poles"]))]))
    c = json.loads((run1 / "controls.json").read_text())["G0"]
    assert c["failing_input_2"]["u_scale"] == 1.1 and not c["failing_input_2"]["pass"]


def test_g1_fails_on_a_wrong_banked_value():
    assert NH.g1_pass(2.2444 + 0.01, 2.2444)
    assert not NH.g1_pass(2.2444, 2.2444 + 0.1)
    assert not NH.g1_pass(2.2444, 2.2444 - 0.03)


def test_g1_runs_on_the_tapped_model(model, monkeypatch):
    """G1 is the STAGEDIFF B3 call through StageForward (no BOS). It runs a plain forward of the same model
    after the tap has been used; verify/2 died there when the hooks stayed registered."""
    import math
    import sys
    import types
    from infra.pg.matrix_forward import _StubTok

    class NoBosTok(_StubTok):
        bos_token_id = None

        def _ids(self, text, add_special_tokens):
            return super()._ids(text, False)

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        load_dataset=lambda *a, **k: {"text": ["Paris is the capital of France. " * 6, "", "Rain falls. " * 4]}))
    tap = NH.MLPTap(model.model.layers)
    x = torch.randint(3, 200, (2, 17))
    with torch.inference_mode():
        NH.g0_ratios(model, tap, [lay.mlp.act_fn for lay in model.model.layers], x)
        r = NH.g1_wikitext(model, NoBosTok(), "cpu")
    assert tap.fired == [0, 1] and math.isfinite(r["mean_nll"]) and r["n_dropped_leading_targets"] == 1
    assert r["forward"] == "text shell" and r["n_tokens"] == len(NoBosTok().encode(
        "Paris is the capital of France. " * 6 + "\n\n" + "Rain falls. " * 4, False)) - 1


POLES = ["L00N3+", "L00N3-", "L01N10+", "L01N60-"]
TEXTS = [{"text_id": "t0", "text": "The cat sat on the mat."}, {"text_id": "t1", "text": "x = 3 * (y + 4)"},
         {"text_id": "t2", "text": "Ab"}]


def topk_positions(root, poles):
    return np.stack([np.load(root / "topk" / f"layer_{NH.parse_pole(p)[0]:02d}.npz")["tok"][NH.parse_pole(p)[1],
                                                                                      NH.parse_pole(p)[2]] for p in poles])


@pytest.fixture(scope="module")
def run2(tmp_path_factory, model, corpus, run1):
    out = tmp_path_factory.mktemp("run2")
    harvest(out, model, corpus, poles=POLES, positions=topk_positions(run1, POLES), probe_texts=TEXTS)
    return out


def test_exemplar_a_equals_topk_values(run1, run2):
    pos = np.load(run2 / "poles" / "positions.npz")["tok"]
    ex = np.load(run2 / "poles" / "exemplar_a.npz")
    for j, p in enumerate(POLES):
        lyr, n, s = NH.parse_pole(p)
        tk = np.load(run1 / "topk" / f"layer_{lyr:02d}.npz")
        assert np.array_equal(pos[j], tk["tok"][n, s])
        for key in ("a", "g", "u"):
            np.testing.assert_allclose(ex[key][j], tk[key][n, s], rtol=1e-5, atol=1e-6)
    top = np.load(run2 / "poles" / "top1000.npz")
    np.testing.assert_allclose(top["a"][:, :K], np.stack([np.load(run1 / "topk" / f"layer_{NH.parse_pole(p)[0]:02d}.npz")
                                                          ["a"][NH.parse_pole(p)[1], NH.parse_pole(p)[2]]
                                                          for p in POLES]), rtol=1e-5, atol=1e-6)


@pytest.fixture(scope="module")
def run3(tmp_path_factory, model, corpus):
    out = tmp_path_factory.mktemp("run3")
    harvest(out, model, corpus, poles=POLES)
    return out


def test_t15_rule_exemplar_is_own_topk(run3):
    """--poles without positions (T_15): positions.npz = the poles' own top-K tok, exemplar_a = its a, g, u."""
    pos = np.load(run3 / "poles" / "positions.npz")["tok"]
    ex = np.load(run3 / "poles" / "exemplar_a.npz")
    assert pos.shape == (len(POLES), K) and pos.dtype == np.int64
    for j, p in enumerate(POLES):
        lyr, n, s = NH.parse_pole(p)
        tk = np.load(run3 / "topk" / f"layer_{lyr:02d}.npz")
        assert np.array_equal(pos[j], tk["tok"][n, s])
        for key in ("a", "g", "u"):
            assert ex[key].dtype == np.float32
            np.testing.assert_array_equal(ex[key][j], tk[key][n, s])
    assert json.loads((run3 / "poles" / "poles.json").read_text()) == POLES


def test_positions_from_prefers_poles_positions(run1, run3, tmp_path):
    assert np.array_equal(NH.positions_from(run3, POLES), np.load(run3 / "poles" / "positions.npz")["tok"])
    (tmp_path / "poles").mkdir()
    (tmp_path / "topk").symlink_to(run1 / "topk")
    planted = topk_positions(run1, POLES) + 1
    np.savez(tmp_path / "poles" / "positions.npz", tok=planted)
    (tmp_path / "poles" / "poles.json").write_text(json.dumps(POLES))
    assert np.array_equal(NH.positions_from(tmp_path, POLES), planted)
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "topk").symlink_to(run1 / "topk")
    assert np.array_equal(NH.positions_from(bare, POLES), topk_positions(run1, POLES))
    with pytest.raises(AssertionError):                       # run1's positions.npz belongs to its drawn poles
        NH.positions_from(run1, POLES)


def test_probe_scores_equal_a_direct_forward(model, run2):
    sc = np.load(run2 / "poles" / "probe" / "scores.npz")
    tok = ByteTok()
    for t, rec in enumerate(TEXTS):
        ids = tok(rec["text"])["input_ids"]
        assert sc["n_tokens"][t] == len(ids) and sc["text_ids"][t] == rec["text_id"]
        for j, p in enumerate(POLES):
            lyr, n, _ = NH.parse_pole(p)
            cap = []
            h = model.model.layers[lyr].mlp.down_proj.register_forward_pre_hook(
                lambda m, a: cap.append(a[0][0, 1:, n].clone()))
            with torch.inference_mode():
                model(input_ids=torch.tensor([[1] + ids]))
            h.remove()
            col = cap[0].numpy()[1:]                              # text position 0 is not scored
            np.testing.assert_allclose([sc["max_a"][t, j], sc["min_a"][t, j]], [col.max(), col.min()], rtol=1e-5,
                                       atol=1e-6)
            assert sc["argmax_tok"][t, j] == 1 + col.argmax() and sc["argmin_tok"][t, j] == 1 + col.argmin()
    assert str(sc["probe_rule"]) == NH.PROBE_RULE
    m = json.loads((run2 / "manifest.json").read_text())
    assert m["probe_excluded_positions"] == "text position 0" and m["probe_rule"] == NH.PROBE_RULE
    vocab = json.loads((run2 / "poles" / "vocab.json").read_text())
    assert set(vocab) == set(POLES) and all(len(v["promoted"]) == 10 for v in vocab.values())
    lyr, n, _ = NH.parse_pole("L00N3+")
    w = model.lm_head.weight @ (model.model.norm.weight * model.model.layers[lyr].mlp.down_proj.weight[:, n])
    assert vocab["L00N3+"]["promoted"][0][1] == pytest.approx(float(w.max()), rel=1e-4)
    assert vocab["L00N3-"]["promoted"][0][1] == pytest.approx(float(-w.min()), rel=1e-4)


def own_a(model, lyr, n, ids):
    """The model's own down_proj input at neuron n over the text's tokens (BOS 1 prepended and dropped)."""
    cap = []
    h = model.model.layers[lyr].mlp.down_proj.register_forward_pre_hook(lambda m, a: cap.append(a[0][0, 1:, n].clone()))
    with torch.inference_mode():
        model(input_ids=torch.tensor([[1] + ids]))
    h.remove()
    return cap[0].numpy()


def score(model, texts, poles):
    tap = NH.MLPTap(model.model.layers)
    with torch.inference_mode():
        return NH.probe_scores(model, tap, [lay.mlp.act_fn for lay in model.model.layers], ByteTok(), 1, texts, poles,
                               "cpu")[0]


def test_probe_skips_a_maximum_at_text_position_0(model):
    text = "Zebras graze while the river rises."
    ids = ByteTok()(text)["input_ids"]
    hits = [n for n in range(64) if own_a(model, 0, n, ids).argmax() == 0 and own_a(model, 0, n, ids).max() > 0]
    assert hits, "no layer-0 neuron peaks at text position 0 on this text: the test cannot fail"
    n = hits[0]
    col = own_a(model, 0, n, ids)
    sc = score(model, [{"text_id": "z", "text": text}], [f"L00N{n}+"])
    assert sc["max_a"][0, 0] == pytest.approx(float(col[1:].max()), rel=1e-5) and sc["max_a"][0, 0] < col[0]
    assert sc["argmax_tok"][0, 0] == 1 + col[1:].argmax() >= 1


def test_probe_one_token_text_fails_loudly(model):
    with pytest.raises(AssertionError, match="'solo'.*1 token"):
        score(model, [{"text_id": "ok", "text": "Ab"}, {"text_id": "solo", "text": "A"}], ["L00N3+"])


def test_probe_two_token_text_scores_its_second_token(model):
    ids = ByteTok()("Ab")["input_ids"]
    sc = score(model, [{"text_id": "two", "text": "Ab"}], ["L01N10+"])
    col = own_a(model, 1, 10, ids)
    assert sc["max_a"][0, 0] == pytest.approx(float(col[1]), rel=1e-5) == pytest.approx(float(sc["min_a"][0, 0]), rel=1e-5)
    assert sc["argmax_tok"][0, 0] == sc["argmin_tok"][0, 0] == 1 and sc["n_tokens"][0] == 2


def test_contexts_are_document_bounded():
    tokens = np.arange(100)
    offsets = np.array([0, 10, 50, 100])
    ident = list
    c = NH.context(tokens, offsets, 12, ident)
    assert c["doc"] == 1 and c["pos"] == 2 and list(c["before"]) == [10, 11] and list(c["after"]) == list(range(13, 29))
    c = NH.context(tokens, offsets, 48, ident)
    assert list(c["before"]) == list(range(10, 48)) and list(c["after"]) == [49]
    c = NH.context(tokens, offsets, 99, ident)
    assert list(c["before"]) == list(range(50, 99)) and list(c["after"]) == [] and list(c["token"]) == [99]
    c = NH.context(tokens, offsets, 9, ident)
    assert c["doc"] == 0 and list(c["after"]) == []


def test_contexts_file_matches_topk(run1, corpus):
    lines = [json.loads(x) for x in (run1 / "samples" / "contexts.jsonl").read_text().splitlines()]
    draw = json.loads((run1 / "samples" / "sample_draw.json").read_text())
    assert [x["pole"] for x in lines] == draw["poles"] and len(draw["poles"]) == 8
    for x in lines:
        lyr, n, s = NH.parse_pole(x["pole"])
        tk = np.load(run1 / "topk" / f"layer_{lyr:02d}.npz")
        assert [r["tok"] for r in x["topk"]] == [int(t) for t in tk["tok"][n, s] if t >= 0]
        for r in x["topk"]:
            d = r["doc"]
            assert corpus["doc_offsets"][d] <= r["tok"] < corpus["doc_offsets"][d + 1]
    m = json.loads((run1 / "metrics.json").read_text())
    assert m["complete"] and m["n_tokens_scored"] == (WINDOW - 1) * N_WINDOWS
