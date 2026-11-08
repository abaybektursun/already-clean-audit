"""Weight-free dry run of UNITE (studies/unite/PREREG.md §9): tiny random OLMo 3 and a stub text tower with MolmoAct2's parameter names, nothing downloaded. It runs the jobs' own code and the offline
analysis end to end, with each gate's failing input and the sample-writing path.

    python -m infra.aws_pg.jobs.unite_dryrun --out <dir> [--part arm1|arm2|rules]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from infra.aws_pg.jobs import unite_route as JR
from infra.aws_pg.jobs.neuron_off_dryrun import WINDOW, fake_corpus, reference_a, tiny_model, tiny_tokenizer
from infra.pg import neurons_harvest as NH
from infra.pg import unite_analysis as UA
from infra.pg import unite_route as UR
from infra.pg import unite_stats as US

N_LAYERS = 6
N_WINDOWS = 60


def expect_raises(fn, why):
    raised = False
    try:
        fn()
    except AssertionError:
        raised = True
    assert raised, f"{why}: nothing raised"


# ------------------------------------------------------------------------------------------ statistic and gate rules
def part_rules():
    rng = np.random.default_rng(0)
    d = 4000

    def gram(A):
        e1, e2 = rng.normal(0, 0.01, d), rng.normal(0, 0.01, d)
        return (A - e1) @ (A - e2).T
    a, b = rng.normal(0, 1, d), rng.normal(0, 1, d)
    mk = lambda v, n: v + rng.normal(0, 0.5, (n, d))
    M_same = gram(np.vstack([mk(a, 10), mk(a, 7)]))
    M_diff = gram(np.vstack([mk(a, 10), mk(b, 7)]))
    t_same, t_diff = US.stat_T(M_same, range(10), range(10, 17))["T"], US.stat_T(M_diff, range(10), range(10, 17))["T"]
    assert 0.9 < t_same < 1.1 and abs(t_diff) < 0.1, (t_same, t_diff)
    t3 = US.stat_T(gram(np.vstack([mk(a, 3), mk(a, 40)])), range(3), range(3, 43))["T"]
    assert 0.85 < t3 < 1.15, f"T moves with counts: {t3}"
    M_none = gram(rng.normal(0, 1, (17, d)))
    assert abs(US.within(M_none, range(10))) < 0.01 * US.within(M_same, range(10)), "no job structure must give W near 0"
    # every gate's failing branch
    assert not US.gate_known(0.5, 0.5)["pass"] and US.gate_known(1e-3, 0.5)["failing_input_failed"] and not US.gate_known(1e-3, 1e-3)["failing_input_failed"]
    assert not US.gate_complete(0.5, 0.5)["pass"] and US.gate_complete(1e-3, 0.5)["failing_input_failed"]
    assert not US.gate_identity([0.5] * 5, [0.1] * 5)["pass"] and US.gate_identity([1.0] * 5, [0.1] * 5)["pass"] and not US.gate_identity([1.0] * 5, [0.9] * 5)["failing_input_failed"]
    assert not US.gate_noise(0.5, 0.9, 1.0)["pass"] and US.gate_noise(0.1, 0.9, 1.0)["pass"] and not US.gate_noise(0.1, 0.1, 1.0)["failing_input_failed"]
    assert US.gate_window([0.01] * 9, [0.9] * 9)["window"] == 256 and US.gate_window([0.5] * 9, [0.9] * 9)["window"] == "full" and not US.gate_window([0.01] * 9, [0.01] * 9)["failing_input_failed"]
    g = US.gate_room(0.5, 0.1, 1.0, 0.0)
    assert g["pass"] and g["failing_input_failed"] and not US.gate_room(0.1, 0.05, 1.0, 0.0)["pass"] and not US.gate_room(0.5, 0.1, 1.3, 0.0)["pass"] and not US.gate_room(0.5, 0.1, 1.0, 0.6)["failing_input_failed"]
    assert US.gate_mlp(0.7, 0.2)["direct_primary"] and not US.gate_mlp(0.3, 0.2)["direct_primary"] and US.gate_mlp(1.7, 0.2)["cancelled"]
    no_fail = US.gate_mlp(0.7, 0.8)
    assert not no_fail["direct_primary"] and not no_fail["failing_input_failed"] and "no working failing input" in no_fail["branch"]
    assert US.rule_packed(0.0, 0.05) == "refuted" and US.rule_packed(0.3, 0.5) == "holds" and US.rule_packed(0.05, 0.3) == "mixed"
    assert abs(US.packed_share(0.5, 0.1) - 4 / 9) < 1e-9
    b1 = US.boot_windows([3.0, 3.1, 2.9, 3.0], [2.0, 2.1, 1.9, 2.0], [2.5, 2.6, 2.4, 2.5], 200)
    assert abs(b1["R"] - 0.5) < 1e-9 and b1["R_lo"] <= 0.5 <= b1["R_hi"]
    print("PASS rules: T against planted truth, count independence, W near 0 with no jobs, every gate's failing branch")


# ------------------------------------------------------------------------------------------ Arm 1
def build_arm1(model, other, tok, tokens):
    """Items in the real formats, from the model's own whole-window activations (`reference_a`, the harvest's rule): the independent reference for the known number."""
    ref, ref2 = reference_a(model, tok, tokens, list(range(N_WINDOWS))), reference_a(other, tok, tokens, list(range(N_WINDOWS)))
    offsets = np.arange(0, tokens.size + 37, 37)
    offsets[-1] = tokens.size
    poles, jobs, tok_map, half, exemplar, room_c = [], [], {}, {}, {}, {}
    neurons = [(3, 5), (3, 9), (4, 11), (5, 2), (5, 7), (4, 20)]
    for i, (L, n) in enumerate(neurons):
        for s, sign in ((0, 1.0), (1, -1.0)):
            p = NH.pole_id(L, n, s)
            a = ref[L][:, :, n].numpy().copy() if hasattr(ref[L], "numpy") else np.asarray(ref[L])[:, :, n].copy()
            a[:, 0] = -np.inf if sign > 0 else np.inf
            order = np.argsort(-(sign * a).ravel())[:50]
            toks = order.astype(np.int64)                # window * 128 + position, the corpus index
            ids = np.asarray(tokens[toks])
            uniq = np.unique(ids)
            J = [int(r) for r in np.flatnonzero(np.isin(ids, uniq[::2]))]
            O = [int(r) for r in np.flatnonzero(np.isin(ids, uniq[1::2]))]
            poles.append(p)
            tok_map[p] = toks.tolist()
            half[p] = "AB"[i % 2]
            exemplar[p] = a.ravel()[order].tolist()
            if len(J) >= 3 and len(O) >= 3:
                big, small = (O, J) if len(O) >= len(J) else (J, O)
                ceiling = {"c": 0, "sides": [big[:len(big) // 2], big[len(big) // 2:]]} if len(big) >= 6 else None
                jobs.append({"job": f"{p}:c1", "pole": p, "c": 1, "ranks": small, "birth": 13 if i % 3 else 15, "t_before": 12, "flagged": i % 2 == 0, "primary": bool(i % 3), "main_older": True,
                             "older": [{"c": 0, "ranks": big, "is_main": True}], "o_star": 0, "ceiling": ceiling})
                if ceiling:
                    room_c[p] = ceiling
    assert len(jobs) >= 6, len(jobs)
    plants = []
    rng = np.random.default_rng(0)
    for L in (3, 4, 5):                                       # lambda as the real builder sets it: the ratio of the two neurons' 25th-ranked values
        r = np.asarray(ref[L])[:, 1:, :].reshape(-1, np.asarray(ref[L]).shape[-1])
        v = -np.sort(-r, axis=0)[24]
        draw = rng.permutation(np.flatnonzero(v > 0))[:80]
        for i in range(40):
            x, y = int(draw[2 * i]), int(draw[2 * i + 1])
            plants.append({"unit": f"PL{L:02d}_{i:02d}", "layer": L, "a": x, "b": y, "lam": float(v[x] / v[y]), "reserve": i >= 20, "half": "AB"[i % 2], "order": i})
    known_p = poles[:3]
    known_tok = [tok_map[p] for p in known_p]
    bank, wrong = [], []
    for p in known_p:
        L, n, _ = NH.parse_pole(p)
        toks = np.asarray(tok_map[p])
        w, q = toks // WINDOW, toks % WINDOW
        bank.append(np.asarray(ref[L])[w, q, n].tolist())
        wrong.append(np.asarray(ref2[L])[w, q, n].tolist())
    items = {"tok": tok_map, "half": half, "jobs": jobs, "units": {"T_15": poles, "T_13": sorted({j["pole"] for j in jobs if j["birth"] == 13})}, "room_ceiling": room_c, "planted": plants, "identity": [],
             "job_counts": [[len(j["ranks"]), len(j["older"][0]["ranks"])] for j in jobs], "exemplar_t15": exemplar,
             "known": {"poles": known_p, "tok": known_tok, "a_bank": {"T_15": bank, "T_13": bank}, "a_wrong": {"T_15": wrong, "T_13": wrong}, "wrong_rule": "another seed's model"}}
    return items, {"tokens": tokens, "doc_offsets": offsets}


def args_for(*extra):
    a = JR.parser().parse_args(["--out", "x", "--items", "x", "--harvest-window", str(WINDOW), "--scan-tokens", str(N_WINDOWS * WINDOW), "--batch", "8", "--full-batch", "4", "--grad-batch", "4", "--scan-batch", "8", *extra])
    a.items_sha256 = "dry"
    return a


def finite_difference(tok, tokens):
    """The gradient readout against a real perturbation of one upstream activation, in fp32 so the difference is resolvable."""
    model = tiny_model("olmo3", seed=3, dtype=torch.float32, n_layers=N_LAYERS).requires_grad_(False)
    tower = NH.text_tower(model)
    t = 3 * WINDOW + 40
    unit = UR.pole_unit("L04N11+")
    kw = dict(tokens=tokens, length=32, bos=tok.bos_token_id, batch=1, hw=WINDOW, device="cpu")
    A = UR.grad_attr(model, [t], unit, **kw)[0]
    W = tower.layers[0].mlp.down_proj.weight.shape[1]
    x, pos = UR.pack(tokens, [t], 32, tok.bos_token_id, WINDOW, "cpu")
    l, k = 2, int(A[2 * W:3 * W].abs().argmax())

    def value(eps):
        rec = {}

        def bump(m, args):
            a = args[0].clone()
            a[0, :, k] = a[0, :, k] * (1 + eps)
            return (a,)
        h = [tower.layers[l].mlp.down_proj.register_forward_pre_hook(bump),
             tower.layers[4].mlp.gate_proj.register_forward_hook(lambda m, a, o: rec.__setitem__("g", o)),
             tower.layers[4].mlp.up_proj.register_forward_hook(lambda m, a, o: rec.__setitem__("u", o))]
        with torch.no_grad():
            NH.forward(model, x, False)
        for z in h:
            z.remove()
        return float(tower.layers[4].mlp.act_fn(rec["g"][0, pos[0], 11]) * rec["u"][0, pos[0], 11])
    eps = 1e-3
    fd = (value(eps) - value(-eps)) / (2 * eps)
    got = float(A[l * W + k])
    assert abs(fd - got) <= 0.03 * abs(got) + 1e-7, (fd, got)
    print(f"gradient readout against a finite difference on layer {l} neuron {k}: {got:.6g} vs {fd:.6g}")


def part_arm1(out: pathlib.Path):
    tok = tiny_tokenizer(bos=True)
    tokens = fake_corpus(tok, N_WINDOWS + 2)
    model = tiny_model("olmo3", seed=0, n_layers=N_LAYERS).requires_grad_(False)
    other = tiny_model("olmo3", seed=1, n_layers=N_LAYERS).requires_grad_(False)
    items, corpus = build_arm1(model, other, tok, tokens)
    d15, d13, dgrad, dsmoke = out / "t15", out / "t13", out / "grad", out / "smoke"
    JR.Job(dsmoke, args_for("--stage", "smoke", "--t", "T_15"), items, model, tok, corpus, "cpu").run()
    s = json.loads((dsmoke / "metrics.json").read_text())
    print("smoke known", s["known_number"]["ratio"], "wrong", s["known_number"]["wrong_ratio"], "completeness", s["completeness"]["median_err_over_pole_max"], "wrong layer", s["completeness"]["wrong_layer"])
    assert s["known_number"]["pass"] and s["known_number"]["failing_input_failed"] and s["completeness"]["pass"] and s["completeness"]["failing_input_failed"] and s["grad_smoke"]["finite"]
    # a failing input through the job's own gate ends the job: the known number against the wrong bank
    wrong = json.loads(json.dumps(items))
    wrong["known"]["a_bank"]["T_15"] = wrong["known"]["a_wrong"]["T_15"]
    expect_raises(lambda: JR.Job(out / "wrong", args_for("--stage", "smoke", "--t", "T_15"), wrong, model, tok, corpus, "cpu").run(), "wrong bank")
    assert json.loads((out / "wrong" / "metrics.json").read_text())["known_number"]["pass"] is False

    JR.Job(d15, args_for("--stage", "route", "--t", "T_15", "--scan", "--window-gate", "--noise"), items, model, tok, corpus, "cpu").run()
    m = json.loads((d15 / "metrics.json").read_text())
    assert m["status"] == "done" and m["n_units_done"] == m["n_units"]
    print("window gate", {k: m["window_gate"][k] for k in ("median", "p90", "median_16", "p90_16", "pass", "failing_input_failed")}, "scan", m["scan"]["kept_by_layer"], m["scan"]["why_dropped"])
    print("identity", {k: m["identity"][k] for k in ("median_T", "two_pole_median_T", "pass", "failing_input_failed", "primary")})
    assert m["identity"]["pass"], m["identity"]
    planted = json.loads((d15 / "planted_contexts.json").read_text())
    if not any(v["kept"] for v in planted.values()):            # a random tiny model may keep no pair; force the kept path so the analysis meets planted Grams
        raise AssertionError(f"no planted pair kept on the tiny model: {m['scan']}")
    pairs = [json.loads(x) for x in (d15 / "samples" / "pairs.jsonl").read_text().splitlines()] if (d15 / "samples" / "pairs.jsonl").exists() else []
    z = np.load(d15 / "grams" / f"{items['units']['T_15'][0]}.npz")
    for k in ("M", "M_foreign", "M_fixed", "M_noise", "a_terms", "g", "u", "c_g", "c_u", "s", "rise", "rise_by_layer", "top_layer", "top_neuron", "top_A", "top_a"):
        assert k in z, k
    assert z["M"].shape == (50, 50) and z["top_A"].shape == (50, UR.TOP_INPUTS) and z["rise"].shape == (50, 4)

    JR.Job(d13, args_for("--stage", "route", "--t", "T_13", "--from-step", str(d15), "--also-grad", "--grad-foreign"), items, model, tok, corpus, "cpu").run()
    res = UA.pilot(items, {"T_15": str(d15), "T_13": str(d13)})
    print("pilot gates:", json.dumps({k: {x: v[x] for x in v if x in ("pass", "failing_input_failed", "branch", "delta_B", "line", "W_noise", "share", "restricted_share", "share_cancelled_half_B", "share_cancelled_no_job")} for k, v in res["gates"].items()}, default=float))
    assert set(res["freeze2"]) == {"m", "W_noise", "band_edges", "primary_readout"}
    assert res["gates"]["W_noise"]["failing_input_failed"], "noise sets from no job must be cancelled"
    assert np.isfinite(res["gates"]["room"]["delta_labels_shuffled"])

    # the gradient readout's own pass, then the pilot read through it
    JR.Job(dgrad, args_for("--stage", "grad", "--t", "T_15", "--from-step", str(d15), "--noise", "--grad-foreign"), items, model, tok, corpus, "cpu").run()
    assert np.load(dgrad / "grams_grad" / f"{items['units']['T_15'][0]}.npz")["M"].shape == (50, 50)
    (dgrad / "grams").symlink_to((d15 / "grams").resolve()) if not (dgrad / "grams").exists() else None
    resg = UA.pilot(items, {"T_15": str(dgrad)}, sub="grams_grad", planted_from=str(d15))
    print("pilot through the gradient readout: room", {k: resg["gates"]["room"][k] for k in ("pass", "delta_B", "line")}, "W_noise", resg["gates"]["W_noise"]["W_noise"])
    finite_difference(tok, tokens)

    # the frozen verdict end to end on the dry banks
    freeze = {**res["freeze2"], "band_edges": {k: (v if v[0] is not None else [0.0, 0.0]) for k, v in res["freeze2"]["band_edges"].items()}}
    jobs = [j for j in items["jobs"] if j["birth"] in (13, 15)]
    v = UA.verdict({"jobs": jobs}, {"T_13": str(d13), "T_15": str(d15)}, freeze, n_boot=200, planted_rows=res["planted_rows"])
    print("verdict on dry banks:", {k: v["primary"][k] for k in ("n_jobs", "n_cancelled", "f", "f_c", "rules")})
    assert set(v["primary"]["rules"]) == {"H_grown", "no_shared_input", "H_context", "H_own_route", "H_grown_original_rule_superseded_before_unblinding", "H_own_route_original_rule_superseded_before_unblinding"}
    mc = v["primary"]["median_comparison"]
    print("median comparison on dry banks:", {k: mc[k] for k in ("birth_vs_planted", "T15_vs_planted")}, "planted", round(mc["planted_median"]["point"], 4), [round(x, 4) for x in (mc["planted_median"]["lo"], mc["planted_median"]["hi"])], "position", mc["position_between_planted_and_ceiling"])
    assert abs(sum(mc["planted_median"]["weights"].values()) - 1) < 1e-9
    # the amended rule's three outcomes on known intervals, and the weighting against a hand count
    assert US.rule_median(0.1, 0.2, 0.15, 0.3)["overlap"] and US.rule_median(0.5, 0.6, 0.1, 0.2)["side"] == "real T higher" and US.rule_median(0.0, 0.05, 0.1, 0.2)["side"] == "real T lower"
    hand, _ = UA.planted_median({0: np.array([1.0, 1.0]), 1: np.array([3.0, 3.0]), 2: np.array([9.0])}, {0: 0.5, 1: 0.5, 2: 0.0}, 50)
    assert abs(hand - 2.0) < 1e-9, hand
    vg = UA.verdict({"jobs": jobs}, {"T_13": str(d13), "T_15": str(dgrad)}, freeze, sub="grams_grad", n_boot=200, planted_rows=resg["planted_rows"])
    assert vg["job_rows"][0]["T_fixed"] is None and any(r["T_foreign"] is not None for r in vg["job_rows"]), "gradient verdict: foreign at birth, no fixed-coefficient Gram"
    print("verdict through the gradient Grams:", {k: vg["primary"][k] for k in ("n_jobs", "n_cancelled", "f", "f_c", "rules")})
    print(f"PASS arm1: smoke, route, scan, window, identity, noise, room with labels shuffled, MLP share, gradient readout, verdict; samples written: {len(pairs)} pairs")


# ------------------------------------------------------------------------------------------ Arm 2
class StubBlock(nn.Module):
    def __init__(self, d, W):
        super().__init__()
        self.attn_norm, self.ff_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = nn.ModuleDict({"att_proj": nn.Linear(d, d, bias=False), "attn_out": nn.Linear(d, d, bias=False)})
        self.mlp = nn.ModuleDict({"ff_proj": nn.Linear(d, 2 * W, bias=False), "ff_out": nn.Linear(W, d, bias=False)})

    def forward(self, h):
        h = h + self.self_attn["attn_out"](torch.tanh(self.self_attn["att_proj"](self.attn_norm(h))))
        up, gate = self.mlp["ff_proj"](self.ff_norm(h)).chunk(2, -1)
        return h + self.mlp["ff_out"](F.silu(gate) * up)


class StubMolmo(nn.Module):
    """36 blocks with MolmoAct2's parameter names, so every name rule meets a real module tree."""

    def __init__(self, seed, d=8, W=12, vocab=256):
        super().__init__()
        torch.manual_seed(seed)
        self.model = nn.ModuleDict({"embed_tokens": nn.Embedding(vocab, d), "layers": nn.ModuleList([StubBlock(d, W) for _ in range(36)]), "norm": nn.LayerNorm(d)})
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.config = types.SimpleNamespace(model_type="llama", num_hidden_layers=36, hidden_size=d, vocab_size=vocab, text_config=None)
        self.dtype = torch.float32

    def forward(self, input_ids, attention_mask=None, position_ids=None, logits_to_keep=0, **kw):
        h = self.model["embed_tokens"](input_ids)
        for b in self.model["layers"]:
            h = b(h)
        h = self.model["norm"](h)
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            h = h[:, -logits_to_keep:, :]
        return types.SimpleNamespace(logits=self.lm_head(h))


def stub_member(seed, parent=None, drift=0.05):
    from infra.pg.matrix_forward import _StubTok, lm_anatomy
    from infra.pg.stagediff_models import _text_forward_cls
    m = StubMolmo(seed).eval().requires_grad_(False)
    if parent is not None:
        m.load_state_dict(parent["model"].state_dict())
        with torch.no_grad():
            for p in m.parameters():
                p += drift * p.abs().mean() * torch.randn_like(p)
    tok = _StubTok()
    fwd = _text_forward_cls()(m, tok, "cpu", bos_id=1)
    fwd.batch_size = 1
    return {"model": m, "tok": tok, "fwd": fwd, "an": lm_anatomy(m)}


def part_arm2(out: pathlib.Path):
    from infra.aws_pg.jobs import unite_swap as JS
    from infra.aws_pg.jobs import unite_topk as JT
    from infra.pg import stagediff_bank as SB
    from infra.pg import unite_swap as SW
    P = stub_member(0)
    C = stub_member(1, parent=P)
    third = {n: p.detach().clone() for n, p in stub_member(2)["model"].named_parameters()}
    text = ("the quick brown fox jumps over the lazy dog and runs far away. " * 60)
    wa = dict(window=24, max_tokens=24 * 9)
    prompts = {"B1": ["Q: 1+1?\nAnswer:", "Q: capital of France? Answer:"], "LIBERO": ["put both moka pots on the stove", "open the drawer"]}
    d = out / "swap"
    bank = SB.Bank(str(d), "unite_swap", {"dry": True})
    r0 = SW.prepared_windows(C, text, **wa)
    known = (sum(x[0] for x in SW.score(P, r0)) / len(r0), sum(x[0] for x in SW.score(C, r0)) / len(r0))
    JS.run_hop(bank, "H5", P, C, text, prompts, third, 256, banked=known, window_args=wa)
    m = json.loads((d / "metrics.json").read_text())
    g = m["controls"]["H5__gates"]
    assert g["known_pass"] and g["k0_pass"] and g["all_text_pass"], g
    assert m["controls"]["H5__failing_input"]["failing_input_failed"], m["controls"]["H5__failing_input"]
    assert len([a for a, r in m["readouts"]["H5"].items() if isinstance(r, dict) and r.get("kind") == "forward"]) == 16
    assert len(m["readouts"]["H5"]["reverse_top1_agreement_with_child"]["arms"]) == 7
    assert (d / "samples" / "H5__last_4_head.jsonl").exists() and (d / "samples" / "H5_reverse_LIBERO__rev_last_3.jsonl").exists()
    assert not JS.BANKED["H5"][0] - 2.7149177 > 1e-5
    bank2 = SB.Bank(str(out / "swap_wrong"), "unite_swap", {"dry": True})
    JS.run_hop(bank2, "H5", P, C, text, prompts, third, 256, banked=(known[0] + 0.1, known[1]), window_args=wa)
    assert not json.loads((out / "swap_wrong" / "metrics.json").read_text())["controls"]["H5__gates"]["known_pass"], "a wrong banked number must fail"
    # the failing input fails to fail when the 'foreign' blocks are the parent's own and every other text part is too: the gate can report R near 1
    rep = UA.swap_report(str(d), hops=("H5",), n_boot=200)
    a = rep["hops"]["H5"]["arms"]["last_36"]
    print("swap: k0", g["k0_nll"], "child", known[1], "all text", g["all_text_nll"], "parent", known[0], "last_36 R", a["R"], "failing input R", m["controls"]["H5__failing_input"]["R"])

    dk = out / "topk"
    bk = SB.Bank(str(dk), "unite_topk", {"dry": True})
    ks = (1, 3, 10, 100)
    JT.run_hop(bk, "H5", P, C, text, ks, True, window_args=wa)
    bk3 = SB.Bank(str(out / "topk_h4"), "unite_topk", {"dry": True})
    JT.run_hop(bk3, "H4", P, C, text, ks, False, window_args=wa)
    mk = json.loads((dk / "metrics.json").read_text())
    tab = mk["controls"]["H5__mass_table"]
    print("topk mass table:", {k: (round(v["top_mass"], 3), v["tail_exists"], v.get("tail_n")) for k, v in tab["k"].items()})
    for arm, r in mk["readouts"]["H5"].items():
        if isinstance(r, dict) and r.get("kind") == "f2" and arm.endswith("both"):
            k = arm.split("_")[0].replace("top", "").replace("tail", "")
            want = tab["k"][k]["top_mass"] if arm.startswith("top") else tab["k"][k].get("tail_mass")
            assert abs(r["mass_restored"] - want) <= 1e-2 * want + 1e-6, (arm, r["mass_restored"], want)         # realized, not intended
    for k in ks:
        r = mk["readouts"]["H5"][f"sv{k}_both"]
        assert abs(r["mass_restored"] - r["realized_mass"]) <= 0.02 * r["realized_mass"] + 1e-6, (k, r)
    allm = mk["readouts"]["H5"]["all_mlp"]
    assert allm["mass_restored"] > 0 and mk["readouts"]["H5"]["all_attention"]["mass_restored"] > 0
    rk = UA.topk_report(str(dk), ks=ks, n_boot=200)
    print("topk reading:", json.dumps({k: (v.get("reading") or {s: x["reading"] for s, x in v["sides"].items()}) for k, v in rk["hops"]["H5"]["reading_by_k"].items()})[:400])
    assert (dk / "samples" / "H5__all_mlp.jsonl").exists()
    print("PASS arm2: forward and reverse arms, gates and their failing inputs, realized mass per arm, singular-value control, samples, reports")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.part in ("all", "rules"):
        part_rules()
    if a.part in ("all", "arm1"):
        part_arm1(out / "arm1")
    if a.part in ("all", "arm2"):
        part_arm2(out / "arm2")
    print("UNITE_DRYRUN_OK", a.part)


if __name__ == "__main__":
    main()
