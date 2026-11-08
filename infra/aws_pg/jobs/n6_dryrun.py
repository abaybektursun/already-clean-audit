"""N6 offline gate, the weight-free dry run (studies/next/07_PICKS/N6_OFFLINE_GATE.md). CPU, no download, no GPU.

    PYTHONPATH=. <venv>/bin/python -m infra.aws_pg.jobs.n6_dryrun --out <dir>

Part A, the pure logic on the real banks: the critic's statistic reproduced, pairs.json rebuilt bit for bit, the balance gate on
both sides, the Wilson interval against known values. Part B, the job end to end on a tiny random OLMo 3 (6 layers, hidden 32,
vocabulary 256, 2,049-token windows, the sliding window set to 64 so that the layer types matter) in bf16 and then fp32: every gate's
pass side, and every registered failing input planted and shown to fail. Part C, freeze and verdict: every label branch from
planted records, DOES NOT COUNT from a planted gate failure, parking. The tiny world checks the code path, not the science.
Prints N6_DRYRUN_OK with the count of checks; any failed check ends the run with an AssertionError after the summary.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pathlib
import shlex
import types

import numpy as np

from infra.pg import n6

CHECKS = []
A4_WINDOWS_SHA = "5822191728d49aba45817f8479058c266bcaaa057dc1037fa0d5509507d1b223"    # sha256 of json.dumps(n6.eval_windows(1024))
HF_CFG = pathlib.Path("studies/next/m2/offline_gate/hf/config_stage1-step1413814.json")


def same_json(a, b, rel=1e-6) -> bool:
    """Two JSON objects equal, floats within rel (strings, ints and the structure exact)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and sorted(a) == sorted(b) and all(same_json(a[k], b[k], rel) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same_json(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)
    return a == b


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": str(detail)[:400]})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:200]}]" if detail != "" else ""), flush=True)


def expect_assert(name, fn, needle):
    """A registered failing input must stop the job: fn() raises AssertionError whose message holds `needle`."""
    try:
        fn()
    except AssertionError as e:
        check(name, needle in str(e), str(e)[:160])
        return
    check(name, False, "no AssertionError")


# ------------------------------------------------------------------------------------------ part A
def part_a(pairs_path: pathlib.Path):
    poles = n6.load_dated()
    lay = np.array([p["layer"] for p in poles])
    age = np.array([p["age"] for p in poles], float)
    r = n6.stratified_spearman(age, np.log([p["max_abs"] for p in poles]), lay)
    check("A1 critic's within-layer Spearman(age, ln max|a|) reproduced (-0.592, age_energy_out.txt)", abs(r + 0.592) < 5e-4, r)
    r2 = n6.stratified_spearman(age, np.log([p["density"] for p in poles]), lay)
    check("A2 critic's Spearman(age, ln density) reproduced (+0.502)", abs(r2 - 0.502) < 5e-4, r2)
    built = n6.build_pairs()
    banked = json.loads(pairs_path.read_text())
    check("A3 pairs.json rebuilt from the banks equals the committed file", json.dumps(json.loads(json.dumps(built)), sort_keys=True) == json.dumps(banked, sort_keys=True),
          f"{built['n_pairs']} pairs")
    b = built["balance"]
    check("A4 balance gate pass side: |rho_pair| <= 0.10 and |SMD| <= 0.10 on ln max|a| and ln density",
          n6.balanced(b), {k: (round(b[k]["rho_pair"], 3), round(b[k]["smd"], 3)) for k in ("ln_max_abs", "ln_density")})
    u = built["failing_input_unmatched"]["balance"]
    check("A5 balance gate failing input (unmatched pairs) fails", not n6.balanced(u),
          {k: (round(u[k]["rho_pair"], 3), round(u[k]["smd"], 3)) for k in ("ln_max_abs", "ln_density")})
    check("A6 every pair: same layer, different neurons, |d age| >= 3, early younger in index",
          all(p["layer"] == n6.parse_pole(p["early"])[0] == n6.parse_pole(p["late"])[0] and
              n6.parse_pole(p["early"])[1] != n6.parse_pole(p["late"])[1] and p["age_late"] - p["age_early"] >= 3 for p in built["pairs"]))
    used = [p[m] for p in built["pairs"] for m in ("early", "late")]
    check("A7 every pole in at most one pair", len(used) == len(set(used)), len(used))
    lm = {p["pole"]: math.log(p["max_abs"]) for p in poles}
    ld = {p["pole"]: math.log(p["density"]) for p in poles}
    c = built["caliper_scan"]["chosen"]
    check("A8 every pair inside the chosen calipers", all(abs(lm[p["early"]] - lm[p["late"]]) <= c["ln_max_abs"] and
                                                            abs(ld[p["early"]] - ld[p["late"]]) <= c["ln_density"] for p in built["pairs"]), c)
    from infra.aws_pg.jobs import n6_edit as J
    reg = J.INPUT_SHA["L1"]
    check("A15 the L1 inputs' sha256 registered in the job equal the committed files",
          reg["pairs.json"] == n6.sha256_file(pairs_path) and reg["e1_known.json"] == n6.sha256_file("studies/next/n6/inputs/e1_known.json"))
    lo, hi = n6.wilson(5, 10)
    check("A9 Wilson(5, 10) = [0.2366, 0.7634]", abs(lo - 0.23659) < 1e-4 and abs(hi - 0.76341) < 1e-4, (lo, hi))
    lo, hi = n6.wilson(0, 10)
    check("A10 Wilson(0, 10) upper = 0.2775", lo == 0.0 and abs(hi - 0.27754) < 1e-4, hi)
    rng = np.random.default_rng(5)
    cn = np.exp(rng.normal(0, 0.15, (32, 11008)))
    en = np.zeros((32, 11008, 2))
    for p in poles:
        en[p["layer"], p["neuron"], 0 if p["sign"] > 0 else 1] = math.exp(2 * math.log(p["max_abs"]) + math.log(p["density"]) - 4 +
                                                                          rng.normal(0, 0.3))
    fp = n6.final_pairs(built, cn, en)
    bf = fp["balance"]
    check("A13 stage 2 on planted L1 arrays (a simulation, not data): balanced on all four quantities, >= 40 pairs, a subset of the graph",
          n6.balanced(bf) and fp["n_pairs"] >= 40 and set(fp["poles"]) <= set(built["poles"]) and fp["stage"] == "L2",
          (fp["n_pairs"], fp["caliper_scan"]["chosen"], {k: round(bf[k]["smd"], 3) for k in ("ln_colnorm", "ln_E_plus")}))
    cn2 = cn.copy()
    for p in poles:
        cn2[p["layer"], p["neuron"]] = math.exp(-0.2 * p["age"])          # planted: column norm set by age
    expect_assert("A14 stage 2 planted: column norm a function of age leaves no balanced point, L2 does not launch",
                  lambda: n6.final_pairs(built, cn2, en), "L2 does not launch")
    # ---- amendment A3: the stage-2 split on the real L1 banks, and a planted L1-edited pole
    l1 = pathlib.Path("studies/next/n6/banks/L1-02")
    edited = {json.loads(x)["pole"] for x in open(l1 / "edits.jsonl") if x.strip()}
    fr = n6.final_pairs(banked, np.load(l1 / "colnorms.npy"), np.load(l1 / "energies.npy"), edited)
    committed = json.loads(pathlib.Path("studies/next/n6/pairs_L2.json").read_text())
    check("A16 A3: pairs_L2.json rebuilt from L1-02's banks equals the committed file (floats within rel 1e-6: the file was written "
          "under numpy 2.4.3, this venv has 1.26.4, and float32 sums differ in the 8th digit; every string and integer exact)",
          same_json(json.loads(json.dumps(fr)), committed), (fr["n_pairs"], fr["n_freeze"], fr["n_test"]))
    touched = [q["pair"] for q in fr["pairs"] if {q["early"], q["late"]} & edited]
    check("A16 A3: 45 pairs, 24 test and 21 freeze; the pairs holding an L1-edited pole are freeze; every other pair is test iff "
          "its early member's neuron index is even",
          fr["n_pairs"] == 45 and fr["n_test"] == 24 and fr["n_freeze"] == 21 and len(touched) == 4 and
          all(q["half"] == ("freeze" if q["pair"] in touched else ("test" if n6.parse_pole(q["early"])[1] % 2 == 0 else "freeze"))
              for q in fr["pairs"]), touched)
    check("A16 A3: the freeze half still holds the probes and the ten raw examples (>= 5 layers, two probe poles)",
          len(n6.raw_example_pairs(fr)) == 5 and len(set(n6.probes(fr))) == 2, (n6.probes(fr), n6.raw_example_pairs(fr)))
    reg2 = J.INPUT_SHA["L2"]
    check("A17 the L2 inputs' sha256 registered in the job equal the committed files",
          reg2["pairs_L2.json"] == n6.sha256_file("studies/next/n6/pairs_L2.json") and
          reg2["e1_known.json"] == n6.sha256_file("studies/next/n6/inputs/e1_known.json"), reg2)
    ev_pair = next(q for q in fr["pairs"] if q["half"] == "test")
    fr2 = n6.final_pairs(banked, np.load(l1 / "colnorms.npy"), np.load(l1 / "energies.npy"), edited | {ev_pair["late"]})
    check("A18 A3 planted: marking a test pair's late pole as L1-edited moves that pair, and only it, to the freeze half",
          [q["pair"] for q in fr2["pairs"] if q["half"] != next(r for r in fr["pairs"] if r["pair"] == q["pair"])["half"]] == [ev_pair["pair"]]
          and fr2["n_test"] == 23, ev_pair["pair"])
    w = n6.eval_windows()
    check("A11 eval windows: 96 distinct, wholly in band 3", len(set(w)) == 96 and w[0] * 2048 >= n6.BAND3_TOK[0] and
          (w[-1] + 1) * 2048 <= n6.BAND3_TOK[1], (w[0], w[-1]))
    # ---- amendment A4: L2's eval set and roster
    w2 = committed["eval_windows"]
    sha_w2 = hashlib.sha256(json.dumps(w2).encode()).hexdigest()
    check("A19 A4: pairs_L2.json's eval set is the registered rule at 1,024 windows (n6.eval_windows(1024)): distinct, ascending, wholly "
          "in band 3, after the grid harvest's windows 0-10,043; the list's sha256 is the registered one; inst_windows (G-INST) are "
          "pairs.json's 96 windows",
          w2 == n6.eval_windows(1024) == sorted(set(w2)) and len(w2) == committed["w_eval"] == 1024 and w2[0] >= 10_044 and
          w2[0] * 2048 >= n6.BAND3_TOK[0] and (w2[-1] + 1) * 2048 <= n6.BAND3_TOK[1] and sha_w2 == A4_WINDOWS_SHA and
          committed["inst_windows"] == banked["eval_windows"] == w, (w2[0], w2[-1], sha_w2[:12], len(set(w2) & set(w))))
    pl = n6.plan(committed)
    roles = {r_: sum(e["role"] == r_ for e in pl) for r_ in ("pc", "probe_pc", "pair", "probe_pair", "zero")}
    pc_l = sorted({e["layer"] for e in pl if e["role"] == "pc"})
    rand_ok = all(f"{q['pair']}_{m}_{k}" in {e["edit"] for e in pl} for q in committed["pairs"] for m in ("early", "late")
                  for k in ("col", "rand"))
    check("A20 A4: the L2 plan holds 209 edits: the positive control (col, rand, col_half) in the 8 layers 0, 4, ..., 28; the probes on "
          "the positive controls of layers 0 and 31 and on two pair poles; the zero edit; col and rand of both members of all 45 pairs "
          "in both halves", len(pl) == 209 and roles == {"pc": 24, "probe_pc": 2, "pair": 180, "probe_pair": 2, "zero": 1} and
          pc_l == list(range(0, 32, 4)) and sorted(e["layer"] for e in pl if e["role"] == "probe_pc") == [0, 31] and rand_ok, roles)
    p6 = sum(math.comb(8, k_) for k_ in range(6, 9)) / 2 ** 8
    check("A21 A4: G-PC needs 6 of 8 layers; P(X >= 6 | n = 8, p = 0.5) = 37/256 = 0.1445", n6.PC_MIN == 6 and
          len(n6.pc_layers()) == 8 and abs(p6 - 37 / 256) < 1e-15 and round(p6, 3) == 0.145, p6)
    pa, pb = n6.plan(committed, half="freeze"), n6.plan(committed, half="test")
    na, nb = [e["edit"] for e in pa], [e["edit"] for e in pb]
    ra = {r_: sum(e["role"] == r_ for e in pa) for r_ in ("pc", "probe_pc", "pair", "probe_pair", "zero")}
    rb = {r_: sum(e["role"] == r_ for e in pb) for r_ in ("pc", "probe_pc", "pair", "probe_pair", "zero")}
    both = sorted(set(na) & set(nb))
    check("A22 A5: the L2 plan split by half. L2a (freeze) 113 edits: the positive control 24, its probes 2, the 21 freeze pairs' col "
          "and rand 84, the two pair probes and the zero edit; L2b (test) 99 edits: the 24 test pairs' col and rand 96 and the same "
          "pair probes and zero edit; together the 209-edit plan in its order; in both only those 3; every test-pair edit in L2b only",
          len(pa) == 113 and len(pb) == 99 and ra == {"pc": 24, "probe_pc": 2, "pair": 84, "probe_pair": 2, "zero": 1} and
          rb == {"pc": 0, "probe_pc": 0, "pair": 96, "probe_pair": 2, "zero": 1} and set(na) | set(nb) == {e["edit"] for e in pl} and
          [e["edit"] for e in pl if e["edit"] in set(na) | set(nb)] == [e["edit"] for e in pl] and
          both == sorted(e["edit"] for e in pl if e["role"] in ("probe_pair", "zero")) and
          all((e["half"] == "test") == (e["edit"] in nb) for e in pl if e["role"] == "pair") and
          [e for e in pa if e["edit"] in both] == [e for e in pb if e["edit"] in both], (len(pa), len(pb), both))
    s = np.array([0, 1, 0, 1])
    check("A12 pair-stratified Spearman = 1 - 2f (f = share of pairs whose earlier member has the larger value)",
          abs(n6.stratified_spearman([1, 5, 2, 9], [3.0, 1.0, 2.0, 4.0], s) - 0.0) < 1e-12 and
          abs(n6.stratified_spearman([1, 5, 2, 9], [3.0, 1.0, 4.0, 2.0], s) - 1.0) < 1e-12)
    return built


# ------------------------------------------------------------------------------------------ part B: the tiny world
def train_tiny(m, tokens, steps=500, seed=2):
    """A few hundred Adam steps on 256-token windows of the first 30 windows, so that the tiny model predicts something (an untrained
    model gives every target the same loss, and K3's off-by-one failing input would have no power there)."""
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    m.train()
    for _ in range(steps):
        st = rng.integers(0, 30 * 2048 - 257, 8)
        x = torch.tensor(np.stack([tokens[s:s + 257] for s in st]).astype(np.int64))
        lg = m(input_ids=x[:, :-1], use_cache=False).logits
        loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    return float(loss.detach())


def tiny_model(seed=0, tokens=None):
    import torch
    from transformers import Olmo3Config, Olmo3ForCausalLM
    c = json.loads(HF_CFG.read_text())
    c.update(hidden_size=32, intermediate_size=48, num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=2,
             vocab_size=256, layer_types=c["layer_types"][:6], sliding_window=64, pad_token_id=0, eos_token_id=1, bos_token_id=1)
    cfg = Olmo3Config(**{k: v for k, v in c.items() if k not in ("architectures", "model_type", "transformers_version", "torch_dtype")})
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(seed)
    m = Olmo3ForCausalLM(cfg)
    tiny_model.train_loss = train_tiny(m, tokens)
    m.requires_grad_(False)
    return m.to(torch.bfloat16).eval()


def tiny_tokens(n_windows=70, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.integers(2, 256, 4096)
    t = np.concatenate([np.roll(base, k * 7)[:2048] for k in range(n_windows)])
    noise = rng.random(t.size) < 0.1
    t[noise] = rng.integers(2, 256, noise.sum())
    return t.astype(np.int32)


def tiny_world(model, tokens):
    """Pairs, poles and E1 item for the tiny model, the 'banked' values computed by the harvest's rule on the bf16 model."""
    import torch
    from infra.aws_pg.jobs import n6_edit as J
    from infra.pg import reuse_ablate as RA
    from infra.pg import unite_route as UR
    M = J.Model(model)
    L, W = M.L, int(M.layers[0].mlp.down_proj.weight.shape[1])
    ev = [40, 44, 48, 52]
    # energies in fp32 on the eval windows, by an independent loop (no job code)
    mf = copy.deepcopy(model).float()
    E = np.zeros((L, W, 2))
    caps = {}
    hs = [lay.mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: caps.__setitem__(l, a[0].float()))
          for l, lay in enumerate(J.Model(mf).layers)]
    with torch.inference_mode():
        for w in ev:
            x = torch.tensor([[1] + tokens[w * 2048:(w + 1) * 2048].tolist()])
            mf(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
            for l in range(L):
                a = caps[l][0, 1:]
                E[l, :, 0] += (torch.relu(a) ** 2).sum(0).numpy()
                E[l, :, 1] += (torch.relu(-a) ** 2).sum(0).numpy()
    for h in hs:
        h.remove()
    norms = torch.stack([lay.mlp.down_proj.weight.float().norm(dim=0) for lay in M.layers]).numpy()
    pairs, poles = [], {}
    k = 0
    for l in range(L):
        # matched like the real pairs, here on ln E+ and ln ||w|| (the instance's two balance quantities): over the + poles of
        # neurons 2..33, the two closest disjoint (even, odd) pairs; the first is a freeze pair (early = the even neuron), the second a
        # test pair (early = the odd neuron)
        lE, lw = np.log(E[l, :, 0]), np.log(norms[l])
        sE, sw = lE[2:34].std(), lw[2:34].std()
        cand = sorted((abs(lE[e_] - lE[o_]) / sE + abs(lw[e_] - lw[o_]) / sw, e_, o_) for e_ in range(2, 34, 2) for o_ in range(3, 34, 2))
        used, chosen = set(), []
        for _, e_, o_ in cand:
            if e_ in used or o_ in used:
                continue
            used |= {e_, o_}
            chosen.append((e_, o_))
            if len(chosen) == 2:
                break
        for j, (ev_, od_) in enumerate(chosen):
            e_, l_ = (ev_, od_) if j == 0 else (od_, ev_)
            pe, pl = n6.pole_id(l, e_, 1), n6.pole_id(l, l_, 1)
            pairs.append({"pair": f"P{k:03d}", "layer": l, "half": "freeze" if e_ % 2 == 0 else "test", "early": pe, "late": pl,
                          "age_early": 4 + (k % 3), "age_late": 9 + (k % 4), "adj_early": -1.0, "adj_late": 0.5 * (k % 2),
                          "concordant_adj": k % 2 == 1})
            k += 1
    rng = np.random.default_rng(3)
    kw = dict(tokens=tokens, length=None, bos=1, batch=4, hw=2048, device="cpu")
    for p in pairs:
        for m in ("early", "late"):
            l, n, s = n6.parse_pole(p[m])
            tok = sorted(rng.choice(np.arange(10 * 2048), 12, replace=False))
            tok = [int(t) for t in tok if t % 2048 != 0][:10]
            r = RA.readout(model, tok, {"layer": l, "terms": [{"n": n, "w": 1.0, "sign": 1.0}]}, None, **kw)
            a = (M.layers[l].mlp.act_fn(torch.from_numpy(r["g"][:, 0])) * torch.from_numpy(r["u"][:, 0])).numpy()
            poles[p[m]] = {"layer": l, "neuron": n, "sign": s, "age": p[f"age_{m}"], "adj": 0.0, "ref_mean": None, "max_abs": 1.0,
                           "density": 1, "q999": 0.1, "tok15": tok, "a15": a.tolist()}
    P = {"stage": "L1", "pairs": pairs, "poles": poles, "eval_windows": ev, "w_eval": len(ev)}
    unit = UR.planted_unit({"unit": "PLT", "layer": 3, "a": 5, "b": 7, "lam": 2.0})
    toks = [int(w * 2048 + 300 + 37 * i) for i, w in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4])]
    members = [[0, 3], [0, 9], [1, 4], [1, 11], [2, 6], [2, 20], [2, 21], [1, 30], [0, 31], [2, 40]]
    kw1 = dict(tokens=tokens, length=256, bos=1, batch=32, hw=2048, device="cpu")
    before = RA.readout(model, toks, unit, None, **kw1)["a"]
    after = RA.readout(model, toks, unit, RA.ablation_dict([tuple(m) for m in members], "zero", None, 3, "cpu"), **kw1)["a"]
    ok = before > 0
    jf = float(((before - after) / before)[ok].mean())
    e1 = {"unit": {"unit": "PLT", "layer": 3, "a": 5, "b": 7, "lam": 2.0}, "members": members, "toks": toks, "length": 256,
          "batch": 32, "a_before": before.tolist(), "a_after": after.tolist(), "job_fall": jf, "failing_input_cross_job_fall": jf + 0.5}
    return P, e1, E, norms


def independent_nll(model, tokens, n_tok=120_000):
    """K2's target in the tiny world, by a second implementation of the harvest's rule (one window at a time)."""
    import torch
    import torch.nn.functional as F
    s, n, w = 0.0, 0, 0
    with torch.inference_mode():
        while n < n_tok:
            x = torch.tensor([[1] + tokens[w * 2048:(w + 1) * 2048].tolist()])
            lg = model(input_ids=x, use_cache=False).logits[0]
            nll = F.cross_entropy(lg[:-1].float(), x[0, 1:], reduction="none")
            take = min(n_tok - n, nll.numel())
            s += float(nll[:take].sum())
            n += take
            w += 1
    return s / n


def plant_pole(P, v):
    """Amendment A1's planted G-POLE value: a pairs file whose bank differs, for every pole, at the exemplar of smallest |a| by v of
    the pole's largest |a| (moved toward zero, so the largest |a| is unchanged). In the tiny world the recompute equals the bank
    (G-POLE 0.0), so the job's statistic on this file is v."""
    Pv = copy.deepcopy(P)
    for q in Pv["poles"].values():
        a = np.array(q["a15"])
        M = np.abs(a[:10]).max()
        j = int(np.argmin(np.abs(a[:10])))
        a[j] -= (np.sign(a[j]) or 1.0) * v * M
        q["a15"] = a.tolist()
    return Pv


def make_job(out, model, tokens, P, e1, reg, stage="L2", budget=1e9, chunk=2, half=None, against=None):
    from infra.aws_pg.jobs import n6_edit as J
    a = types.SimpleNamespace(stage=stage, budget_min=budget, batch=2, chunk=chunk, half=half, against=against)
    return J.Job(a, model, None, tokens, P, e1, out, "cpu", lambda ids: " ".join(str(int(i)) for i in ids), reg)


def part_b(out: pathlib.Path):
    import torch
    from infra.aws_pg.jobs import n6_edit as J
    tokens = tiny_tokens()
    model0 = tiny_model(tokens=tokens)
    check("B the tiny model was trained (last training loss below ln 256 - 1)", tiny_model.train_loss < math.log(256) - 1,
          tiny_model.train_loss)
    P, e1, E_ind, norms_ind = tiny_world(model0, tokens)
    k2 = independent_nll(model0, tokens)
    reg = {"versions": {"torch": "t", "transformers": "x", "device": "cpu"}, "snapshot": "tiny", "bos": 1, "k2_target": k2,
           "k2_fail": k2 + 0.04, "input_sha": {}}
    vers = dict(reg["versions"])
    # ---- L1 on the stage-1 pairs, then the stage-2 pairs from L1's arrays, then L2
    j1 = make_job(out / "L1", copy.deepcopy(model0), tokens, P, e1, reg, stage="L1", chunk=4)      # one chunk; L2 below: two
    j1.run(vers, "tiny")
    cn, en1 = np.load(out / "L1" / "colnorms.npy"), np.load(out / "L1" / "energies.npy")
    P1 = P
    P = {**copy.deepcopy(P1), "stage": "L2", "instance": {q: {"colnorm": float(cn[v["layer"], v["neuron"]]),
                                                               "E_plus": float(en1[v["layer"], v["neuron"], 0 if v["sign"] > 0 else 1])}
                                                           for q, v in P1["poles"].items()}, "inst_windows": P1["eval_windows"]}
    model = copy.deepcopy(model0)
    job = make_job(out / "full", model, tokens, P, e1, reg)
    job.run(vers, "tiny")
    m = json.loads((out / "full" / "metrics.json").read_text())
    g = m["gates"]
    for name in ("G_SAME", "G_REV", "K1", "K2", "G_POLE", "G_FP32", "K3", "G_INST", "G_ID", "G_CACHE", "G_EPS", "G_ZN",
                 "G_NORM", "G_SAMPLES"):
        check(f"B {name}: pass side passes and its failing input fails", g[name]["pass"] and not g[name]["failing_input"]["pass"],
              {k: v for k, v in g[name].items() if k in ("value", "target", "value_max_abs", "n", "failing_input_min", "bf16",
                                                           "failing_value_off_by_one", "max_E_rel_err")})
    z = np.load(out / "full" / "g_pole.npz")
    names = [str(x) for x in z["poles"]]
    act = J.Model(model0).layers[0].mlp.act_fn
    stat = [float(np.abs(z["a"][i, :, 0] - z["a_bank"][i]).max() / np.abs(z["a_bank"][i]).max()) for i in range(len(names))]
    check("B A1 g_pole.npz: per pole the recomputed g, u, a of the neuron and of n + 1 beside the bank's a at the 10 positions; "
          "a = act(g) u; the bank's a and positions equal the pairs file; the statistic from the file equals metrics' per_pole",
          z["g"].shape == z["u"].shape == z["a"].shape == (len(names), J.POLE_EX, 2) and names == sorted(P["poles"]) and
          all(z["tok"][i].tolist() == P["poles"][p]["tok15"][:J.POLE_EX] and np.array_equal(z["a_bank"][i], P["poles"][p]["a15"][:J.POLE_EX])
              and z["neuron"][i] == P["poles"][p]["neuron"] and z["neighbour"][i] == P["poles"][p]["neuron"] + 1 for i, p in enumerate(names))
          and all(np.array_equal(z["a"][i], (act(torch.from_numpy(z["g"][i])) * torch.from_numpy(z["u"][i])).numpy()) for i in range(len(names)))
          and stat == [r[1] for r in g["G_POLE"]["per_pole"]], (z["g"].shape, max(stat)))
    phases = ["load_K1", "K2", "G_POLE", "bf16_probes", "cast", "K3", "pass1", "inst", "pass2", "progression"]   # inst: A4's G-INST pass
    check("B A1 peak GPU memory recorded per phase in metrics.json (None on the CPU)", list(m["measured"]["peak_mem_gib"]) == phases,
          m["measured"]["peak_mem_gib"])
    for v, ok in ((0.03, True), (0.06, False)):
        jv = make_job(out / f"pole_{v}", copy.deepcopy(model0), tokens, plant_pole(P, v), e1, reg)
        if ok:
            jv.pole_check()
        else:
            expect_assert(f"B A1 G_POLE planted {v} of max|a| stops the job (tolerance {J.POLE_TOL})", jv.pole_check, "G_POLE FAILED")
        gv = jv.m["gates"]["G_POLE"]
        check(f"B A1 G_POLE planted {v} of max|a|: value {v}, {'passes' if ok else 'fails'} at the tolerance {J.POLE_TOL}",
              gv["pass"] == ok and abs(gv["value"] - v) < 1e-6 and not gv["failing_input"]["pass"], (gv["value"], gv["pass"]))
    # ---- amendment A2: G-ZN and G-CACHE at the tolerances frozen from L1-02, planted values on both sides
    first = next(r for r in map(json.loads, open(out / "full" / "edits.jsonl")) if r["kind"] == "col")
    for v, ok in ((5e-8, True), (5e-7, False)):
        zg = J.zn_gate([v], first["dL"])
        check(f"B A2 G_ZN planted zero edit dL {v:g}: {'passes' if ok else 'fails'} at {J.TAU_EPS_L1:.4g}; failing input (first col "
              f"edit dL {first['dL']:.3g}) fails", zg["pass"] == ok and not zg["failing_input"]["pass"], (zg["value_max_abs"], zg["pass"]))
    zf = J.zn_gate([0.0], 5e-8)
    check("B A2 G_ZN planted failing input: a first column edit at 5e-8 passes the bar, so the failing side registers it",
          zf["failing_input"]["pass"])
    rng = np.random.default_rng(7)
    base = torch.from_numpy(rng.uniform(1, 10, (4, 2048)).astype(np.float32))
    other = base + 1.0
    for v, ok in ((5e-6, True), (5e-5, False)):
        fin = base.clone()
        fin[1, 100] += v
        cg = J.cache_gate(fin, base, other)
        check(f"B A2 G_CACHE planted largest per-token difference {v:g}: {'passes' if ok else 'fails'} at {J.CACHE_MAX_TOL:g}; "
              "failing input fails", cg["pass"] == ok and not cg["failing_input"]["pass"], (cg["value_max_abs"], cg["value_mean"]))
    cg = J.cache_gate(base.double() + 2e-7, base, other)
    check("B A2 G_CACHE planted uniform offset 2e-7 (largest 2e-7, under 1e-5): the mean condition fails it", not cg["pass"],
          (cg["value_mean"], cg["value_max_abs"]))
    hs = g["G_CACHE"]["head_input_shapes"]
    check("B A2 the head's input shapes of the two paths are recorded: base pass 1 a batch, the progression one window",
          hs["base_pass1 (model forward)"] == [[2, 2049, 32]] and hs["progression (head per window)"] == [[1, 2049, 32]] and
          g["G_ZN"]["head_input_shapes"] == hs, hs)
    check("B A2 G_CACHE records the mean and the largest per-token difference and its failing input's values",
          all(k in g["G_CACHE"] for k in ("value_mean", "value_max_abs", "value_mean_abs")) and
          g["G_CACHE"]["failing_input"]["value_max_abs"] > J.CACHE_MAX_TOL, {k: g["G_CACHE"][k] for k in ("value_mean", "value_max_abs")})
    check("B K2: the job's corpus NLL equals a second implementation (one window at a time)", abs(g["K2"]["value"] - k2) < 1e-4,
          (g["K2"]["value"], k2))
    check("B G_EPS: the bf16 probe moves the loss by more than the bound, the fp32 probe by less (the floor is the precision's)",
          max(abs(v) for v in g["G_EPS"]["bf16"].values()) > n6.EPS_ABS >= max(abs(v) for v in g["G_EPS"]["fp32"].values()),
          (g["G_EPS"]["bf16"], g["G_EPS"]["fp32"]))
    check("B G_PC and G_DOSE computed on the positive-control layers (A4: every fourth, layers 0 and 4 of 6; values only; a tiny model "
          "need not pass)", g["G_PC"]["n_layers"] == 2 and g["G_PC"]["layers"] == [0, 4] and "failing_input_swapped" in g["G_DOSE"],
          (g["G_PC"]["n_col_gt_rand"], g["G_DOSE"]["n_full_gt_half"]))
    recs = [json.loads(x) for x in open(out / "full" / "edits.jsonl")]
    plan = n6.plan(P, 6)
    check("B every planned edit has one record, in plan order", [r["edit"] for r in recs] == [e["edit"] for e in plan], len(recs))
    check("B m['complete'] and nothing parked", m["complete"] and not m["parked"])
    en = np.load(out / "full" / "energies.npy")
    check("B the job's E+ equals an independent fp32 loop over the same windows (rel 1e-5)",
          np.allclose(en, E_ind, rtol=1e-5, atol=1e-6), float(np.abs(en - E_ind).max()))
    check("B the job's column norms equal the independent ones", np.array_equal(np.load(out / "full" / "colnorms.npy"), norms_ind))
    zero = [r for r in recs if r["kind"] == "zero"]
    check("B zero-norm edit: bitwise equal to base, dL exactly 0", zero and all(r["bitwise_equal_base"] and r["dL"] == 0.0 for r in zero))
    cols = [r for r in recs if r["kind"] == "col"]
    check("B every col edit changes the loss (not bitwise equal to base)", all(not r["bitwise_equal_base"] for r in cols))
    pr = [r for r in recs if r["role"] == "pair"]
    c_ok = all(abs(r["c"] - min(norms_ind[r["layer"], n6.parse_pole(P_["early"])[1]], norms_ind[r["layer"], n6.parse_pole(P_["late"])[1]])) < 1e-6
               for r in pr for P_ in [next(p for p in P["pairs"] if p["pair"] == r["pair"])])
    check("B pair edit size c = min of the two column norms; equal within each pair", c_ok)
    check("B kappa = c / ||w|| and kappa = 1 on the smaller-norm member", all(abs(r["kappa"] - r["c"] / r["norm_w"]) < 1e-9 for r in pr)
          and all(max(r["kappa"] for r in pr if r["pair"] == q and r["kind"] == "col") == 1.0 for q in {r["pair"] for r in pr}))
    check("B realized: the hook's E equals pass 1's E+ (rel 1e-4) on every edit", max(r["E_rel_err"] for r in recs if r["kind"] != "zero") <= 1e-4,
          max(r["E_rel_err"] for r in recs if r["kind"] != "zero"))
    check("B every edit read 50 eval exemplars and the KL is >= 0", all(r["n_ex"] == 50 and r["kl_eff"] >= -1e-7 for r in recs))
    raw = [json.loads(x) for x in open(out / "full" / "samples" / "raw_examples.jsonl")]
    check("B ten raw examples: pole, partner, c, dL, rand dL, 5 tokens with context",
          len(raw) == 10 and all(len(x["top5_tokens"]) == 5 and x["dL_rand_same_norm"] is not None and x["partner"] for x in raw))
    # ---- equivalence of the col edit at kappa = 1 with zero written into the pole's own-sign activations
    mf = copy.deepcopy(model0).float()
    Mf = J.Model(mf)
    l, n = 2, 5
    x = torch.tensor([[1] + tokens[40 * 2048:41 * 2048].tolist()])
    w = Mf.layers[l].mlp.down_proj.weight[:, n].detach().float()
    ed = J.Edit(Mf.layers[l].mlp.down_proj, n, 1, -w, "add")
    with torch.inference_mode(), ed:
        a1 = mf(input_ids=x, use_cache=False).logits

    def pre(mod, args):
        a = args[0].clone()
        on = a[..., n] > 0
        on[:, 0] = False
        a[..., n] = torch.where(on, torch.zeros_like(a[..., n]), a[..., n])
        return (a,)
    hh = Mf.layers[l].mlp.down_proj.register_forward_pre_hook(pre)
    with torch.inference_mode():
        a2 = mf(input_ids=x, use_cache=False).logits
    hh.remove()
    check("B col edit (kappa 1) equals zero written into the pole's own-sign inputs (E1's operator restricted to the sign), fp32",
          float((a1 - a2).abs().max()) < 1e-4, float((a1 - a2).abs().max()))
    # ---- the progression and its failing input
    with torch.inference_mode():
        h0 = Mf.embed(x)
        pr_ = Mf.prep(h0)
        full = mf(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0]
        mine = Mf.window_logits(Mf.run(h0, 0, 6, pr_)[0])
        skip = Mf.window_logits(Mf.run(h0, 1, 6, pr_)[0])
    check("B forward from a kept residual (layer 0) equals the model's forward bitwise", torch.equal(full, mine))
    check("B its failing input (layer 0 skipped) differs", not torch.equal(full, skip), float((full - skip).abs().max()))
    # ---- planted failures, each must stop the job with the gate's name
    expect_assert("B G_SAME planted wrong torch version stops", lambda: make_job(out / "f1", copy.deepcopy(model0), tokens, P, e1, reg)
                  .run({**vers, "torch": "wrong"}, "tiny"), "G_SAME FAILED")
    expect_assert("B G_REV planted wrong snapshot stops", lambda: make_job(out / "f2", copy.deepcopy(model0), tokens, P, e1, reg)
                  .run(vers, "0" * 40), "G_REV FAILED")
    expect_assert("B K1 planted wrong banked value stops", lambda: make_job(out / "f3", copy.deepcopy(model0), tokens, P,
                  {**e1, "job_fall": e1["job_fall"] + 0.1}, reg).run(vers, "tiny"), "K1 FAILED")
    expect_assert("B K1 failing input planted equal to the target stops", lambda: make_job(out / "f4", copy.deepcopy(model0), tokens, P,
                  {**e1, "failing_input_cross_job_fall": e1["job_fall"]}, reg).run(vers, "tiny"), "K1: its failing input passed")
    expect_assert("B K2 planted wrong target stops", lambda: make_job(out / "f5", copy.deepcopy(model0), tokens, P, e1,
                  {**reg, "k2_target": k2 + 0.01}).run(vers, "tiny"), "K2 FAILED")
    Pw = copy.deepcopy(P)
    for q in Pw["poles"].values():
        q["a15"] = (0.7 * np.array(q["a15"])).tolist()
    expect_assert("B G_POLE planted wrong bank (0.7 x a, the T_14 analog) stops", lambda: make_job(out / "f6", copy.deepcopy(model0),
                  tokens, Pw, e1, reg).run(vers, "tiny"), "G_POLE FAILED")
    Pb = copy.deepcopy(P)
    for q in Pb["instance"].values():
        q["colnorm"] *= 1.1
    expect_assert("B G_INST planted: column norms 1.1 x L1's stop L2", lambda: make_job(out / "f7", copy.deepcopy(model0), tokens, Pb,
                  e1, reg).run(vers, "tiny"), "G_INST FAILED")
    expect_assert("B a stage-1 pairs file given to L2 stops", lambda: make_job(out / "f8", copy.deepcopy(model0), tokens, P1, e1, reg)
                  .run(vers, "tiny"), "pairs file of stage L1 given to stage L2")
    # ---- G-NORM's failing input: the hook on the neighbouring neuron reads another E
    r0 = next(r for r in recs if r["kind"] == "col")
    ed2 = J.Edit(Mf.layers[r0["layer"]].mlp.down_proj, r0["n"] + 1, r0["sign"], torch.zeros(32), "add")
    with torch.inference_mode(), ed2:
        for wv in P["eval_windows"]:
            xx = torch.tensor([[1] + tokens[wv * 2048:(wv + 1) * 2048].tolist()])
            mf(input_ids=xx, use_cache=False)
    check("B G_NORM failing input: the neighbouring neuron's E differs from the pole's by more than 1e-4",
          abs(ed2.E - r0["E_plus_pass1"]) / r0["E_plus_pass1"] > 1e-4, (ed2.E, r0["E_plus_pass1"]))
    # ---- the stop rule: a budget that parks the later layers
    jb = make_job(out / "park", copy.deepcopy(model0), tokens, P, e1, reg, budget=(m["elapsed_s"] * 0.55) / 60)
    jb.run(vers, "tiny")
    mp = json.loads((out / "park" / "metrics.json").read_text())
    check("B stop rule: a short budget parks the remaining edits and the job ends normally", mp["parked"] and not mp["complete"],
          (len(mp["parked"]), mp["n_done"]))
    # ---- L1: the ten raw column edits, probes and the zero edit only
    r1 = [json.loads(x) for x in open(out / "L1" / "edits.jsonl")]
    check("B L1 runs the 10 raw column edits, 2 probes and the zero edit; the ten raw examples written",
          sum(r["kind"] == "col" for r in r1) == 10 and sum(r["kind"] == "probe" for r in r1) == 2 and
          len(open(out / "L1" / "samples" / "raw_examples.jsonl").read().splitlines()) == 10, len(r1))
    same = {r["edit"]: r["dL"] for r in r1}
    check("B L1 and L2 give the same dL for the same edits (bit for bit)", all(same[r["edit"]] == r["dL"] for r in recs if r["edit"] in same))
    keys = ("dL", "dL_win", "R", "E_plus", "realized_sq", "kl_eff", "dl_eff", "top5", "bitwise_equal_base", "hook_calls", "n_ex")
    by1 = {r["edit"]: r for r in r1}
    common = [r for r in recs if r["edit"] in by1]
    check("B A4 chunks: L1 (one chunk of 4 windows) and L2 (two chunks of 2) give equal records for the same edits, field by field "
          "(dL, per-window dL, R, E+, KL, top-5 tokens, bitwise flag, hook calls)", len(common) == len(r1) and m["chunk"] == 2 and
          json.loads((out / "L1" / "metrics.json").read_text())["chunk"] == 4 and
          all(r[k] == by1[r["edit"]][k] for r in common for k in keys), len(common))
    # ---- amendment A4: an L2 eval set larger than L1's; G-INST reads L1's windows
    P8 = {**copy.deepcopy(P), "eval_windows": list(range(40, 56, 2)), "w_eval": 8}
    j8 = make_job(out / "w8", copy.deepcopy(model0), tokens, P8, e1, reg, chunk=4)
    j8.run(vers, "tiny")
    m8 = json.loads((out / "w8" / "metrics.json").read_text())
    r8 = [json.loads(x) for x in open(out / "w8" / "edits.jsonl")]
    e8 = np.load(out / "w8" / "energies.npy")
    check("B A4 an L2 eval set of 8 windows (two chunks of 4) beside L1's 4: G-INST reads L1's 4 windows (inst_windows) and passes, its "
          "failing input fails; every record has 8 per-window values; pass 1's E+ is over the 8 windows (differs from L1's)",
          m8["gates"]["G_INST"]["pass"] and not m8["gates"]["G_INST"]["failing_input"]["pass"] and m8["w_eval"] == 8 and
          all(len(r["dL_win"]) == 8 for r in r8) and m8["complete"] and not np.allclose(e8, en1, rtol=1e-3),
          (m8["gates"]["G_INST"]["value"], len(r8)))
    P8b = {**copy.deepcopy(P8), "inst_windows": P8["eval_windows"]}
    expect_assert("B A4 G-INST planted on L2's eval set (8 windows) instead of L1's 4 stops the job",
                  lambda: make_job(out / "w8b", copy.deepcopy(model0), tokens, P8b, e1, reg, chunk=4).run(vers, "tiny"), "G_INST FAILED")
    # ---- the PC gate's logic on planted records
    by = {f"PC{l_:02d}_{k}": {"dL": v} for l_ in range(32) for k, v in (("col", 2e-3), ("rand", 1e-4), ("col_half", 1e-3))}
    gp, gd = J.pc_gates(by, 32, 1e-8)
    check("B G_PC / G_DOSE planted: col > rand everywhere passes, its swapped labels fail", gp["pass"] and not gp["failing_input"]["pass"]
          and gd["pass"] and not gd["failing_input"]["pass"])
    by2 = {f"PC{l_:02d}_{k}": {"dL": v * (1 if l_ % 2 else -1)} for l_ in range(32) for k, v in (("col", 2e-3), ("rand", 1e-4),
                                                                                                   ("col_half", 1e-3))}
    gp2, _ = J.pc_gates(by2, 32, 1e-8)
    check("B G_PC planted: col > rand only in the odd layers, none of the 8 positive-control layers (A4), fails", not gp2["pass"],
          gp2["n_col_gt_rand"])
    for k_ok, ok in ((6, True), (5, False)):
        win_l = set(n6.pc_layers()[:k_ok])
        by3 = {f"PC{l_:02d}_{k}": {"dL": (v if l_ in win_l else {"col": 1e-4, "rand": 2e-3, "col_half": 1e-3}[k])} for l_ in range(32)
               for k, v in (("col", 2e-3), ("rand", 1e-4), ("col_half", 1e-3))}
        gp3, gd3 = J.pc_gates(by3, 32, 1e-8)
        check(f"B A4 G_PC planted: col > rand in {k_ok} of the 8 positive-control layers {'passes' if ok else 'fails'} (need 6); "
              "the swapped labels fail", gp3["pass"] == ok and gp3["n_col_gt_rand"] == k_ok and gp3["n_layers"] == 8 and
              gp3["need"] == 6 and not gp3["failing_input"]["pass"], (gp3["n_col_gt_rand"], gp3["failing_input_swapped"]))
        check(f"B A4 G_DOSE planted: full > half in {k_ok} of 8 {'passes' if ok else 'fails'}",
              gd3["pass"] == ok and gd3["n_full_gt_half"] == k_ok, gd3["n_full_gt_half"])
    split_a5(out, model0, tokens, P, e1, reg, vers, recs, m)
    return P, recs, m


def split_a5(out, model0, tokens, P, e1, reg, vers, recs, m):
    """Amendment A5: L2 as two launches. A tiny L2a (half freeze), freeze 2 from its bank alone, a tiny L2b (half test) checked
    against L2a's metrics.json, the verdict reading both banks through the CLI; each against the single launch (`full`, the same
    chunk and batch); then the planted failures: an L2b with another eval set, an incomplete L2a, the wrong banks."""
    import shutil
    from infra.aws_pg.jobs import n6_edit as J
    keys = ("dL", "dL_win", "R", "E_plus", "realized_sq", "kl_eff", "dl_eff", "top5", "bitwise_equal_base", "hook_calls", "n_ex", "c",
            "kappa", "delta_norm", "E_plus_pass1")
    full = {r["edit"]: r for r in recs}
    ja = make_job(out / "L2a", copy.deepcopy(model0), tokens, P, e1, reg, half="freeze")
    ja.run(vers, "tiny")
    jb = make_job(out / "L2b", copy.deepcopy(model0), tokens, P, e1, reg, half="test", against=str(out / "L2a" / "metrics.json"))
    jb.run(vers, "tiny")
    ma, mb = (json.loads((out / x / "metrics.json").read_text()) for x in ("L2a", "L2b"))
    ra, rb = ([json.loads(x) for x in open(out / d / "edits.jsonl")] for d in ("L2a", "L2b"))
    check("B A5 L2a and L2b run their halves of the plan (n6.plan with half), in plan order; together every edit of the single launch",
          [r["edit"] for r in ra] == [e["edit"] for e in n6.plan(P, 6, "freeze")] and
          [r["edit"] for r in rb] == [e["edit"] for e in n6.plan(P, 6, "test")] and
          {r["edit"] for r in ra} | {r["edit"] for r in rb} == set(full) and ma["complete"] and mb["complete"], (len(ra), len(rb), len(full)))
    check("B A5 every L2a and L2b record equals the single launch's record of the same edit, field by field (dL, per-window dL, R, E+, "
          "KL, top-5 tokens, c, kappa, the bitwise flag, hook calls); the record format is unchanged (same keys)",
          all(r[k] == full[r["edit"]][k] for r in ra + rb for k in keys) and all(sorted(r) == sorted(full[r["edit"]]) for r in ra + rb),
          len(ra) + len(rb))
    ga, gb = ma["gates"], mb["gates"]
    l2b_gates = ("G_SAME", "G_REV", "K1", "K2", "G_POLE", "G_FP32", "K3", "G_INST", "G_ID", "G_CACHE", "G_EPS", "G_ZN", "G_NORM")
    check("B A5 L2b re-runs the 13 gates that do not depend on the half, each passes and its failing input fails; G-EPS reads the two "
          "pair probes; no G-PC, G-DOSE, G-SAMPLES, no positive-control or freeze-pair record in L2b",
          sorted(gb) == sorted(l2b_gates) and all(gb[k]["pass"] and not gb[k]["failing_input"]["pass"] for k in l2b_gates) and
          len(gb["G_EPS"]["fp32"]) == 2 and not [r for r in rb if r["role"] in ("pc", "probe_pc") or r.get("half") == "freeze"
                                                 and r["role"] == "pair"], sorted(gb))
    check("B A5 L2a holds every gate of L2, G-PC, G-DOSE and G-SAMPLES equal to the single launch's; G-EPS reads the four probes",
          sorted(ga) == sorted(m["gates"]) and all(ga[k] == m["gates"][k] for k in ("G_PC", "G_DOSE", "G_SAMPLES")) and
          len(ga["G_EPS"]["fp32"]) == 4 and (out / "L2a" / "samples" / "raw_examples.jsonl").read_text() ==
          (out / "full" / "samples" / "raw_examples.jsonl").read_text(), sorted(ga))
    check("B A5 L2b's metrics record L2a's eval windows, eval tokens and pairs content (sha256) beside its own, equal; halves recorded",
          ma["half"] == "freeze" and mb["half"] == "test" and all(mb["against"][k] == ma[k] == mb[k] for k in n6.SAME_KEYS) and
          mb["against"]["path"] == str(out / "L2a" / "metrics.json"), {k: str(mb[k])[:12] for k in n6.SAME_KEYS})
    # ---- freeze 2 from L2a alone, then the verdict on both banks, through the CLI the head runs
    pp = out / "pairs_tiny.json"
    pp.write_text(json.dumps(P))
    n6.main(["freeze", "--bank", str(out / "L2a"), "--pairs", str(pp), "--out", str(out / "th.json")])
    th = json.loads((out / "th.json").read_text())
    th_full = n6.freeze(P, recs, m["gates"])
    check("B A5 freeze 2 from L2a's bank alone (n6 freeze --bank L2a) equals freeze on the single launch's records and gates",
          json.dumps(th, sort_keys=True) == json.dumps(json.loads(json.dumps(th_full)), sort_keys=True),
          (th["tau_eps"], th["tau_s"], th["G_POWER"]["ratio"]))
    thp = {**th, "G_POWER": {**th["G_POWER"], "pass": True}}            # planted pass, so that the test half is read
    (out / "th_pass.json").write_text(json.dumps(thp))
    n6.main(["verdict", "--bank", str(out / "L2a"), "--bank", str(out / "L2b"), "--pairs", str(pp), "--thresholds",
             str(out / "th_pass.json"), "--out", str(out / "v.json")])
    v = json.loads((out / "v.json").read_text())
    v_full = json.loads(json.dumps(n6.verdict(P, recs, m["gates"], thp)))
    n_test = sum(q["half"] == "test" for q in P["pairs"])
    check("B A5 the verdict on both banks (n6 verdict --bank L2a --bank L2b, G-POWER planted to pass) reads every test pair and equals "
          "the single launch's verdict, every field", v == v_full and v["n_test_read"] == n_test and not v["parked"],
          (v["headline"], v["n_test_read"], v["failed_gates"]))
    rec2, g2 = n6.read_banks([out / "L2a", out / "L2b"], ("freeze", "test"))
    g2["L2b:K3"] = {**g2["L2b:K3"], "pass": False}
    v2 = n6.verdict(P, rec2, g2, thp)
    check("B A5 the verdict reads L2b's gates: a planted failure of L2b's K3 makes every label DOES NOT COUNT ('L2b:K3' listed)",
          "L2b:K3" in v2["failed_gates"] and all(v2[h]["label"] == "DOES NOT COUNT" for h in ("H_age", "H_adj", "H_eff")), v2["failed_gates"])
    thf = {**th, "G_POWER": {**th["G_POWER"], "pass": False}}
    (out / "th_fail.json").write_text(json.dumps(thf))
    n6.main(["verdict", "--bank", str(out / "L2a"), "--pairs", str(pp), "--thresholds", str(out / "th_fail.json"), "--out",
             str(out / "v_fail.json")])
    vf = json.loads((out / "v_fail.json").read_text())
    check("B A5 G-POWER failed: the verdict on L2a alone is UNDECIDED_NO_POWER with no test pair read (L2b is never launched)",
          all(vf[h].get("realized", vf[h]["label"]) == "UNDECIDED_NO_POWER" for h in ("H_age", "H_adj", "H_eff")) and
          vf["n_test_read"] == 0,
          vf["headline"])
    # ---- planted failures
    Palt = {**copy.deepcopy(P), "eval_windows": [40, 44, 48, 54]}
    expect_assert("B A5 planted: an L2b whose eval set differs from L2a's (window 52 replaced by 54) stops before its first gate",
                  lambda: make_job(out / "L2b_alt", copy.deepcopy(model0), tokens, Palt, e1, reg, half="test",
                                   against=str(out / "L2a" / "metrics.json")).run(vers, "tiny"), "not one comparison")
    check("B A5 planted: that L2b wrote no gate and no record (it stopped before G-SAME)",
          not (out / "L2b_alt" / "metrics.json").exists() and (out / "L2b_alt" / "edits.jsonl").read_text() == "")
    shutil.copytree(out / "L2b", out / "L2b_planted")
    mp = json.loads((out / "L2b_planted" / "metrics.json").read_text())
    mp["eval_windows_sha256"] = hashlib.sha256(json.dumps(Palt["eval_windows"]).encode()).hexdigest()
    (out / "L2b_planted" / "metrics.json").write_text(json.dumps(mp))
    expect_assert("B A5 planted: the verdict given an L2b bank whose eval windows differ from L2a's stops",
                  lambda: n6.main(["verdict", "--bank", str(out / "L2a"), "--bank", str(out / "L2b_planted"), "--pairs", str(pp),
                                   "--thresholds", str(out / "th_pass.json"), "--out", str(out / "v_planted.json")]), "not one comparison")
    mi = {**ma, "complete": False, "parked": ["P011_early_col"]}
    (out / "L2a_incomplete.json").write_text(json.dumps(mi))
    expect_assert("B A5 planted: an L2b against an L2a that parked edits stops", lambda: make_job(
        out / "L2b_inc", copy.deepcopy(model0), tokens, P, e1, reg, half="test", against=str(out / "L2a_incomplete.json")).run(vers, "tiny"),
        "L2b does not run")
    expect_assert("B A5 planted: n6 freeze given L2b's bank stops (freeze reads L2a only)",
                  lambda: n6.main(["freeze", "--bank", str(out / "L2a"), "--bank", str(out / "L2b"), "--pairs", str(pp), "--out",
                                   str(out / "th_x.json")]), "this step reads ['freeze']")
    expect_assert("B A5 planted: the verdict with G-POWER passed and L2a alone stops (L2b's bank is needed)",
                  lambda: n6.main(["verdict", "--bank", str(out / "L2a"), "--pairs", str(pp), "--thresholds", str(out / "th_pass.json"),
                                   "--out", str(out / "v_x.json")]), "this step reads ['freeze', 'test']")
    proj = json.loads(pathlib.Path("studies/next/n6/projection_L1.json").read_text())
    want = {"next_n6_L2a.json": ("freeze", None, proj["L2a"]["job_budget_min"]),
            "next_n6_L2b.json": ("test", "s3://pg-lineage-769063704696/results/next_n6/L2a-01/out/edit/n6/metrics.json",
                                 proj["L2b"]["job_budget_min"])}
    got = {}
    for f in want:
        st = json.loads((pathlib.Path("infra/aws_pg/steps") / f).read_text())
        a_ = J.parser().parse_args(shlex.split(st[0]["args"]) + ["--out", "x"])
        J.check_args(a_)
        got[f] = (a_.half, a_.against, a_.budget_min)
        assert len(st) == 1 and st[0]["module"] == "infra.aws_pg.jobs.n6_edit" and a_.stage == "L2" and \
            a_.inputs == "s3://pg-lineage-769063704696/results/next_n6/inputs_L2", st
    check("B A5 the manifests pass the job's argument checks: next_n6_L2a.json --half freeze, no --against, --budget-min = "
          "projection_L1.json L2a's job stop rule; next_n6_L2b.json --half test, --against L2a-01's metrics.json, L2b's job stop rule; "
          "both stage L2 on inputs_L2", got == want, got)
    expect_assert("B A5 planted: the job's CLI refuses stage L2 without --half",
                  lambda: (J.main(["--out", str(out / "cli"), "--stage", "L2", "--inputs", "x", "--corpus", "x", "--budget-min", "1"])),
                  "L2 runs as L2a (freeze) and L2b (test)")
    expect_assert("B A5 planted: the job's CLI refuses --half test without --against",
                  lambda: (J.main(["--out", str(out / "cli"), "--stage", "L2", "--half", "test", "--inputs", "x", "--corpus", "x",
                                   "--budget-min", "1"])), "given to L2b and only to L2b")


# ------------------------------------------------------------------------------------------ part C: freeze and verdict
def synth(n_freeze=39, n_test=35, d_test=None, d_rand=None, seed=0, kl=None):
    """A planted pairs file and records with chosen D_col and D_rand per test pair."""
    rng = np.random.default_rng(seed)
    pairs, recs = [], []
    for i in range(n_freeze + n_test):
        half = "freeze" if i < n_freeze else "test"
        pid = f"P{i:03d}"
        pairs.append({"pair": pid, "layer": i % 32, "half": half, "early": f"L{i % 32:02d}N{2 * i}+", "late": f"L{i % 32:02d}N{2 * i + 1}+",
                      "concordant_adj": i % 3 != 0})
        j = i - n_freeze
        dc = (d_test[j] if half == "test" else rng.normal(0, 1e-4)) if d_test is not None else rng.normal(0, 1e-4)
        dr = (d_rand[j] if half == "test" else rng.normal(0, 1e-5)) if d_rand is not None else rng.normal(0, 1e-5)
        base = 1e-3 + abs(rng.normal(0, 2e-4))
        win = rng.normal(0, 2e-5, 96)
        for m, sgn in (("early", 0.5), ("late", -0.5)):
            for k, dd in (("col", dc), ("rand", dr)):
                v = (base if k == "col" else base / 10) + sgn * dd
                recs.append({"edit": f"{pid}_{m}_{k}", "dL": v, "dL_win": (v + sgn * win).tolist(),
                             "kl_eff": (kl[m] if kl else 0.1) if k == "col" else 0.01, "E_plus": 1.0 + (0.1 if m == "early" else 0.0)})
    return {"pairs": pairs}, recs


def good_gates():
    ok = {"pass": True, "failing_input": {"pass": False}}
    return {"K1": ok, "K2": ok, "G_EPS": {**ok, "fp32": {"a": 1e-9, "b": -2e-9}}, "G_PC": {**ok, "n_col_gt_rand": 31, "n_layers": 32}}


def part_c():
    n = 35
    cases = {
        "SUPPORTED": (np.full(n, 3e-4), np.zeros(n)),
        "SUPPORTED_UNSPECIFIC": (np.full(n, 3e-4), np.full(n, 3e-4)),
        "REVERSED": (np.full(n, -3e-4), np.zeros(n)),
        "FLAT": (np.array([3e-4 if i % 2 else -3e-4 for i in range(n)]), np.zeros(n)),
        "UNDECIDED_NO_POWER": (np.array([2e-6 if i % 2 else -2e-6 for i in range(n)]), np.zeros(n)),
        "UNDECIDED": (np.array([3e-4 if i % 10 < 6 else -3e-4 for i in range(n)]), np.zeros(n)),
    }
    for want, (dc, dr) in cases.items():
        P, recs = synth(d_test=dc, d_rand=dr)
        th = n6.freeze(P, recs, good_gates())
        v = n6.verdict(P, recs, good_gates(), th)
        check(f"C label branch {want}", v["H_age"]["label"] == want, (v["H_age"]["label"], round(v["H_age"]["p"], 3), v["H_age"]["ci"]))
    P, recs = synth(d_test=np.full(n, 3e-4), d_rand=np.zeros(n))
    g = good_gates()
    g["G_PC"] = {**g["G_PC"], "n_col_gt_rand": 8}
    v = n6.verdict(P, recs, g, n6.freeze(P, recs, g))
    check("C ABOVE_CEILING when the age ordering is more reliable than the positive control's (8 of 32)", v["H_age"]["label"] == "ABOVE_CEILING",
          v["H_age"]["ci"])
    g = good_gates()
    g["K1"] = {"pass": False, "failing_input": {"pass": False}}
    v = n6.verdict(P, recs, g, n6.freeze(P, recs, g))
    check("C a failed gate makes every label DOES NOT COUNT (realized label kept beside it)", v["H_age"]["label"] == "DOES NOT COUNT"
          and v["H_age"]["realized"] == "SUPPORTED" and "K1" in v["failed_gates"])
    g = good_gates()
    g["K2"] = {"pass": True, "failing_input": {"pass": True}}
    v = n6.verdict(P, recs, g, n6.freeze(P, recs, g))
    check("C a failing input that passed makes every label DOES NOT COUNT", v["H_age"]["label"] == "DOES NOT COUNT"
          and "K2:failing_input_passed" in v["failed_gates"])
    g = good_gates()
    g["G_EPS"] = {**g["G_EPS"], "fp32": {"a": 5e-5}}
    th = n6.freeze(P, recs, g)
    check("C G_RES fails when the rounding floor exceeds min(1e-6, 0.02 x q10 |dL_col|) of the freeze half (A4: the factor 50)",
          not th["G_RES"]["pass"], (th["G_RES"]["value"], th["G_RES"]["bound"]))
    small = [{**r, "dL": r["dL"] * 1e-3, "dL_win": [x * 1e-3 for x in r["dL_win"]]} for r in recs]     # |dL_col| near 1e-6
    q10 = n6.freeze(P, small, good_gates())["q10_abs_dL_col_freeze"]
    for f_, ok in ((0.015, True), (0.05, False)):
        g = good_gates()
        g["G_EPS"] = {**g["G_EPS"], "fp32": {"a": f_ * q10}}
        th_ = n6.freeze(P, small, g)
        check(f"C A4 G_RES planted floor {f_} x q10 (q10 {q10:.3g}): {'passes' if ok else 'fails'} at q10 / 50"
              + (" (it would fail at freeze 1's q10 / 100)" if ok else " (the registered failing input)"),
              th_["G_RES"]["pass"] == ok and th_["G_RES"]["bound"] < n6.EPS_ABS, (th_["G_RES"]["value"], th_["G_RES"]["bound"]))
    th = n6.freeze(P, recs, good_gates())
    check("C room: at 35 test pairs the FLAT band is reachable and 24 wins lift the lower bound above 0.5",
          th["room"]["flat_reachable"] and th["room"]["wins_needed_for_lower_bound_above_half"] is not None,
          (th["room"]["wilson_at_half"], th["room"]["wins_needed_for_lower_bound_above_half"]))
    for nf, nt, flat, sup, rev in ((21, 24, True, 17, 7), (17, 28, True, 20, 8), (28, 17, False, 13, 4)):
        Pr, rr = synth(n_freeze=nf, n_test=nt)
        tr = n6.freeze(Pr, rr, good_gates())
        rv = max(w_ for w_ in range(nt + 1) if n6.wilson(w_, nt)[1] < 0.5)
        check(f"C A3 room at {nt} test pairs ({nf} freeze): FLAT {'reachable' if flat else 'unreachable'}, SUPPORTED needs {sup} wins, "
              f"REVERSED at most {rev}; freeze reads {nf} pairs (tau_s and G-RES's q10 from {2 * nf} column edits)",
              tr["room"]["flat_reachable"] == flat and tr["room"]["wins_needed_for_lower_bound_above_half"] == sup and rv == rev and
              tr["discovery"]["n"] == nf and tr["tau_s"] > 0, (tr["room"]["wilson_at_half"], tr["tau_s"]))
    rec2 = [r for r in recs if not r["edit"].startswith("P070")]
    v = n6.verdict(P, rec2, good_gates(), th)
    check("C one parked test pair is listed; the label still reads (1 of 35 <= 0.2)", v["parked"] == ["P070"] and "G_PARK" not in v["failed_gates"])
    rec3 = [r for r in recs if int(r["edit"][1:4]) < 39 + 27]
    v = n6.verdict(P, rec3, good_gates(), th)
    check("C parked above 0.2 of the test half: DOES NOT COUNT (G_PARK)", "G_PARK" in v["failed_gates"])
    g = good_gates()
    del g["G_PC"]
    v = n6.verdict(P, rec3, g, th)
    check("C a run parked before every positive control was read: DOES NOT COUNT (G_PC_missing)", "G_PC_missing" in v["failed_gates"]
          and v["H_age"]["label"] == "DOES NOT COUNT")
    P, recs = synth(d_test=np.full(n, 3e-4), d_rand=np.zeros(n), kl={"early": 1.0, "late": 0.1})
    v = n6.verdict(P, recs, good_gates(), n6.freeze(P, recs, good_gates()))
    check("C H_eff reads the collateral per unit efficacy: an early pole ten times as effective at 3e-4 nats per token more cost is REVERSED",
          v["H_eff"]["label"] == "REVERSED", v["H_eff"])
    check("C practical: the age pick saves D/2 against a random pick", abs(v["practical"]["gain_vs_random_pick"] - 1.5e-4) < 1e-9,
          v["practical"])
    th = n6.freeze(P, recs, good_gates())
    check("C freeze reads only the freeze half (test D planted at +3e-4 does not move the freeze median)",
          abs(th["discovery"]["median_D_col"]) < 1e-4 and th["discovery"]["n"] == 39, th["discovery"]["median_D_col"])
    # ---- amendment A4: the power gate at freeze 2
    P, recs = synth(n_freeze=21, n_test=24, d_test=np.full(24, 3e-4), d_rand=np.zeros(24))
    for ratio, ok in ((0.4, True), (0.6, False)):
        rp = plant_power(P, recs, ratio)
        th = n6.freeze(P, rp, good_gates())
        gp = th["G_POWER"]
        check(f"C A4 G_POWER planted: window-bootstrap sd of D_col at {ratio} x the median |D_col| over the 21 freeze pairs "
              f"{'passes' if ok else 'fails'} (bar 0.5)", gp["pass"] == ok and abs(gp["ratio"] - ratio) < 1e-9,
              (gp["value"], gp["median_abs_D_col"], gp["ratio"]))
        v = n6.verdict(P, rp, good_gates(), th)
        if ok:
            check("C A4 G_POWER passes: the verdict reads the test half (planted SUPPORTED)", v["H_age"]["label"] == "SUPPORTED" and
                  v["n_test_read"] == 24, v["headline"])
        else:
            check("C A4 G_POWER fails: H_age, H_adj and H_eff read UNDECIDED_NO_POWER and no test pair is read",
                  all(v[h]["label"] == "UNDECIDED_NO_POWER" for h in ("H_age", "H_adj", "H_eff")) and v["n_test_read"] == 0, v["headline"])
            fz = [r for r in rp if next(q for q in P["pairs"] if q["pair"] == r["edit"][:4])["half"] == "freeze"]
            v2 = n6.verdict(P, fz, good_gates(), th)
            check("C A4 G_POWER fails: with every test record removed the verdict is the same (the test half is not read; no G_PARK)",
                  v2["headline"] == v["headline"] and "G_PARK" not in v2["failed_gates"], v2["failed_gates"])
            g = good_gates()
            g["K1"] = {"pass": False, "failing_input": {"pass": False}}
            v3 = n6.verdict(P, rp, g, th)
            check("C A4 G_POWER fails and a gate failed: DOES NOT COUNT, UNDECIDED_NO_POWER kept as the realized label",
                  v3["H_age"]["label"] == "DOES NOT COUNT" and v3["H_age"]["realized"] == "UNDECIDED_NO_POWER" and "K1" in v3["failed_gates"])


def plant_power(P, recs, ratio):
    """Amendment A4's planted power gate: the freeze pairs' per-window D_col rescaled about its mean so that each pair's
    window-bootstrap sd (n6.boot_sd_win with freeze's seed) is `ratio` x the median |D_col| over the freeze pairs; D_col unchanged."""
    out = copy.deepcopy(recs)
    ix = n6.index(out)
    rows = [p for p in P["pairs"] if p["half"] == "freeze"]
    D = [ix[f"{p['pair']}_early_col"]["dL"] - ix[f"{p['pair']}_late_col"]["dL"] for p in rows]
    target = ratio * float(np.median(np.abs(D)))
    for i, p in enumerate(rows):
        e, l_ = ix[f"{p['pair']}_early_col"], ix[f"{p['pair']}_late_col"]
        dw = np.array(e["dL_win"]) - np.array(l_["dL_win"])
        z = dw - dw.mean()
        e["dL_win"] = (np.array(l_["dL_win"]) + dw.mean() + z * (target / n6.boot_sd_win(z, n6.SEED + i))).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default="studies/next/n6/pairs.json")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    part_a(pathlib.Path(a.pairs))
    part_b(out)
    part_c()
    n_fail = sum(not c["ok"] for c in CHECKS)
    (out / "dryrun_summary.json").write_text(json.dumps({"n_checks": len(CHECKS), "n_failed": n_fail, "checks": CHECKS}, indent=1))
    print(f"{len(CHECKS)} checks, {n_fail} failed")
    assert n_fail == 0, [c["check"] for c in CHECKS if not c["ok"]]
    print("N6_DRYRUN_OK", len(CHECKS), "checks", flush=True)


if __name__ == "__main__":
    main()
