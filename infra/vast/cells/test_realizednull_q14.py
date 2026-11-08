"""REALIZEDNULL session 2: tests for the Qwen2.5-14B cell (N7), its manifest (N8) and the corpus job's pure parts (N9).

Tiny tensors and a toy text-only model; no model weights, no S3, no GPU. The integration tests at the end call every
`# N1:` call site of N7, and the N1 calls N7 makes through N3's helpers, on inputs built by N7's own functions from
Gram and verification passes over the toy, as infra/aws_pg/jobs/test_realizednull_pg1.py does for N3. The last section
reads a session-2 run with N1's gates() and N5 (scripts/realizednull_analyze.py), the run directory written by N7's
own writers.
Run: python3 -m pytest infra/vast/cells/test_realizednull_q14.py -q
"""
import ast
import contextlib
import inspect
import io
import json
import math
import pathlib
import re
import shutil
import tarfile
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from infra.aws_pg.jobs import realizednull_pg1 as P
from infra.aws_pg.jobs import realizednull_q14_corpus as N9
from infra.pg import gutenberg_core as G
from infra.pg import realizednull_core as R
from infra.vast.cells import realizednull_q14 as Q
from scripts import realizednull_analyze as A

ROOT = pathlib.Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "studies/lineage/paligemma1-pi0/research"
T0 = "2026-11-02T00:00:00Z"


# ---------------- manifest (N8) ----------------

def test_manifest_matches_registered_fields(monkeypatch):
    """§9.2 N8 with §0.4 U5: image shackleton:v1-14b, disk 70, the gpu_query, cells q14_smoke then q14_full, watch
    {90, 7}, in gb3_am1b_harden.json's shape; baked weights; results land at results/mdl/mendeleev_realizednull_q14
    (§9.6 step 7); the cells parse to the smoke's windows 0-7 and the full run's C = 0-63 (§0.4 P7). expected_gpu
    "RTX A6000": the controller's offer search (plan and auto-respawn) keeps only RTX A6000 offers, and lifecycle.sh's
    `grep -qi` accepts the A6000's nvidia-smi name."""
    m = json.loads((ROOT / "infra/vast/manifests/realizednull_q14.json").read_text())
    gb3 = json.loads((ROOT / "infra/vast/manifests/gb3_am1b_harden.json").read_text())
    assert list(m) == list(gb3)
    assert m["image"] == gb3["image"] == "769063704696.dkr.ecr.us-east-1.amazonaws.com/shackleton:v1-14b"
    assert m["disk"] == 70 and m["gpu_query"] == gb3["gpu_query"] == \
        "gpu_ram>=48 num_gpus=1 cuda_vers>=12.8 compute_cap>=800"
    assert m["watch"] == {"load_backstop_min": 90, "max_billed_h": 7}
    assert (m["expected_gpu"], m["bound"]) == ("RTX A6000", gb3["bound"]) and "weights_mode" not in m
    from infra.vast import controller
    offers = [{"id": i, "gpu_name": g, "dph_total": p, "machine_id": i} for i, (g, p) in enumerate(
        [("CMP 170HX", 0.40), ("RTX A6000", 0.55), ("RTX PRO 6000 WS", 1.57), ("RTX 6000Ada", 0.70)])]
    monkeypatch.setattr(controller, "vast", lambda *a, **k: offers)
    monkeypatch.setattr(controller, "wedged_machines", set)
    assert [o["gpu_name"] for o in controller.search_offers(m)] == ["RTX A6000"]
    boot_ok = lambda gpu: re.search(m["expected_gpu"], gpu, re.I)              # lifecycle.sh: grep -qi on nvidia-smi
    assert boot_ok("NVIDIA RTX A6000") and not boot_ok("NVIDIA CMP 170HX")
    assert [c["id"] for c in m["cells"]] == ["q14_smoke", "q14_full"]
    for c in m["cells"]:
        assert c["module"] == "infra.vast.cells.realizednull_q14" and c["model_dir"] == gb3["cells"][0]["model_dir"]
        assert (ROOT / (c["module"].replace(".", "/") + ".py")).exists()        # preflight.sh T1's check
    program = json.loads((ROOT / "infra/vast/program.json").read_text())
    assert f"{program['s3_results']}/{program['program']}_{m['rental']}" == \
        "s3://pg-lineage-769063704696/results/mdl/mendeleev_realizednull_q14"
    smoke, full = (Q.cell_args(c) for c in m["cells"])
    assert (smoke["smoke"], smoke["n_windows"], smoke["n_construct"]) == (True, 8, 4)
    assert (full["smoke"], full["n_windows"], full["n_construct"]) == (False, 196, 64)
    assert P.splits(196, 64)["D"] == list(range(64, 98))
    for bad in ({"n_construct": 98}, {"n_windows": 98}, {"smoke": 1}):
        with pytest.raises((AssertionError, KeyError)):
            Q.cell_args({**m["cells"][1], **bad})


# ---------------- roster, slots, batches ----------------

def test_roster_is_session_one_minus_rtall_and_k1_plus_topenergy128():
    full = Q.roster(Q.cell_args({"smoke": False, "n_windows": 196, "n_construct": 64, "model_dir": "m"}))
    names = [x[0] for x in full]
    assert len(names) == len(set(names)) == 210            # session 1's 214, minus RT_all's 5, plus topenergy128
    assert [x[0] for x in full if x[3] == "pre"] == list(Q.PRE) and full[0][:3] == ("band", "zero", "band")
    count = lambda pat: sum(bool(re.fullmatch(pat, n)) for n in names)
    assert {f: count(rf"{f}_s\d+") for f in ("em", "spec", "haar", "rt", "rthalf", "rt2x", "emt", "rtall", "em_zt",
                                             "rt_zt", "em_mean", "rt_mean", "spec_mean", "haar_mean")} == {
        "em": 40, "spec": 5, "haar": 5, "rt": 40, "rthalf": 3, "rt2x": 3, "emt": 40, "rtall": 0, "em_zt": 5,
        "rt_zt": 5, "em_mean": 40, "rt_mean": 5, "spec_mean": 3, "haar_mean": 3}
    ops = {n: op for n, op, _, _ in full}
    assert ops["band_zt"] == ops["em_zt_s0"] == ops["rt_zt_s4"] == "zero_text" and ops["em_mean_s39"] == "mean"
    assert {"none_causal", "none_bidir", "none_prefixlm"}.isdisjoint(names)
    smoke = Q.roster(Q.cell_args({"smoke": True, "n_windows": 6, "n_construct": 4, "model_dir": "m"}))
    assert len(smoke) == 45 and {f"em_s{s}" for s in range(6)} <= {x[0] for x in smoke}


def test_two_arm_phases_cover_the_roster_once():
    """run() scores the arms whose bases are not constructed first (phase A, after G14-PRE) and the RT-family and
    EM_T arms after (phase B, while arm_stats runs): each arm once, roster order in a phase."""
    full = Q.roster(Q.cell_args({"smoke": False, "n_windows": 196, "n_construct": 64, "model_dir": "m"}))
    rest = [x for x in full if x[3] == "rest"]
    a = [x[0] for x in rest if not Q.needs_construction(x[2])]
    b = [x[0] for x in rest if Q.needs_construction(x[2])]
    assert sorted(a + b) == sorted(x[0] for x in rest) and not set(a) & set(b)
    assert len(b) == 40 + 3 + 3 + 40 + 5 + 5 and {"rt_zt_s0", "rt_mean_s4", "emt_s39", "rthalf_s2"} <= set(b)
    assert {"em_s0", "em_mean_s39", "spec_mean_s2", "haar_s4", "band_zt", "topsigma32_mean", "ident_mean"} <= set(a)
    assert not any(Q.needs_construction(x) for x in ("topsigma32", "bottomsigma32", "topenergy32", "ident", "band"))


def test_batches_follow_gb3_and_never_mix_open_and_sealed_windows():
    b = Q.batches(range(196))
    assert b[:2] == [[0, 1, 2, 3], [4, 5, 6, 7]] and [96, 97] in b and [98, 99, 100, 101] in b
    assert all(max(x) < Q.F0 or min(x) >= Q.F0 for x in b) and sum(b, []) == list(range(196))
    assert Q.batches(range(98)) == [list(range(i, min(i + 4, 98))) for i in range(0, 98, 4)]   # GB-3's corpus[:98]
    assert Q.batches(range(64, 98))[0] == [64, 65, 66, 67]


def test_smoke_scores_windows_0_to_7_in_gb3s_batches():
    """§0.4 U5: the smoke scores windows 0-7 (C = 0-3, D = 4-7) in GB-3's batches [0-3], [4-7], so its G14-PRE on 0-7
    compares like-for-like with GB-3's per-window NLL (windows 0-5 batched [4, 5] alone); its other settings stand."""
    m = json.loads((ROOT / "infra/vast/manifests/realizednull_q14.json").read_text())
    a = Q.cell_args(next(c for c in m["cells"] if c["id"] == "q14_smoke"))
    assert (a["smoke"], a["n_windows"], a["n_construct"]) == (True, 8, 4)
    assert {k: a[k] for k in Q.SMOKE_DRAWS} == Q.SMOKE_DRAWS
    sp = P.splits(a["n_windows"], a["n_construct"])
    assert (sp["C"], sp["D"], sp["R"], sp["F"]) == ([0, 1, 2, 3], [4, 5, 6, 7], list(range(8)), [])
    assert Q.batches(sp["R"]) == [[0, 1, 2, 3], [4, 5, 6, 7]] == Q.batches(range(Q.F0))[:2]
    assert Q.batches(range(6)) == [[0, 1, 2, 3], [4, 5]]                   # the S4c smoke's batches
    ref = Q.g14pre_references(sp["R"], None, _gb3_wnll())
    assert ref["source"] == "gb3_full v1abl_window_nll.json, same windows"
    assert set(ref["ppl"]) == set(Q.G14_ARMS.values())


# ---------------- hooks ----------------

def test_ablation_hook_identity_x_rows_mean_classes_and_realized_change():
    B_, S, n = 3, 7, 6
    torch.manual_seed(2)
    h = torch.randn(B_, S, n).to(torch.bfloat16)
    h32 = h.float()
    mu = torch.stack([h32[:, 0].mean(0), h32[:, 1:].mean((0, 1))])
    for op in ("zero", "zero_text", "mean"):                          # k = 0 is the identity, bitwise
        assert torch.equal(Q.ablation_hook(torch.zeros(n, 0), op, mu, {})(None, (h,))[0], h)
    U = torch.linalg.qr(torch.randn(n, 2))[0]
    zt = Q.ablation_hook(U, "zero_text", mu, {})(None, (h,))[0]
    assert torch.equal(zt[:, 0], h[:, 0]) and float((zt[:, 1:].float() @ U).abs().max()) < 0.05
    one = h.clone()
    one[:, 0] = mu[0].to(torch.bfloat16)                              # position 0 at its class mean: unchanged
    mean = Q.ablation_hook(U, "mean", mu.to(torch.bfloat16).float(), {})(None, (one,))[0]
    assert torch.equal(mean[:, 0], one[:, 0])
    W = torch.randn(n, n)
    acc = {"num": [0.0], "den": [0.0]}
    rows = torch.tensor([True, False, True])
    out = Q.ablation_hook(U, "zero", None, {"acc_rows": rows}, W, acc, 0)(None, (h,))[0].float()
    assert float(acc["num"][0]) == pytest.approx(float(((h32 - out)[rows] @ W.T).pow(2).sum()), rel=1e-6)
    assert float(acc["den"][0]) == pytest.approx(float((h32[rows] @ W.T).pow(2).sum()), rel=1e-6)
    acc0 = {"num": [0.0], "den": [0.0]}
    Q.ablation_hook(U, "zero", None, {"acc_rows": torch.zeros(3, dtype=torch.bool)}, W, acc0, 0)(None, (h,))
    assert acc0 == {"num": [0.0], "den": [0.0]}


# ---------------- writers: sealed routing ----------------

class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"t{int(i)}" for i in ids)

    def convert_ids_to_tokens(self, ids):
        return f"t{int(ids)}" if np.ndim(ids) == 0 else [f"t{int(i)}" for i in ids]


CORPUS = (np.arange(196 * 512, dtype=np.int64).reshape(196, 512) * 7) % 1000
ARMS = ["none", "band", "anti32", "haar_s0", "em_s3", "em_s4", "em_s5", "topenergy128", "ident_zero", "rt_s0",
        "band_mean"]


def _scores(arm, windows, f_shift):
    """Synthetic per-window NLL and per-position arrays. Windows >= 98 carry f_shift, so any number that reads them
    changes between two runs that differ only in f_shift."""
    k = ARMS.index(arm)
    out = {}
    for w in windows:
        rng = np.random.default_rng(1000 * k + w)
        tgt = CORPUS[w][1:]
        shift = f_shift if w >= Q.F0 else 0.0
        out[w] = {"nll": 1500.0 + 3.0 * k + 0.01 * w + shift + (0 if k < 1 else rng.random()), "ntok": 511,
                  "lp": np.log(rng.uniform(0.01, 1.0, 511)) - shift,
                  "top_id": np.where(rng.random(511) < 0.4, tgt, tgt + 1),
                  "top_lp": np.log(rng.uniform(0.3, 1.0, 511))}
    return out


def _run_writers(out, f_shift):
    """N7's per-arm writes for a full cell's windows, then G14-PRE's record, arm_stats.json and metrics.json."""
    sp = P.splits(196, 64)
    sink = P.Sink(out, "discovery", now=lambda: T0)
    rec = P.Recorder(sink, sp, ARMS)
    sink.json("sealed/corpus_f.json", {"array_sha256": Q.array_sha256(CORPUS[98:])})
    ast = {P.wkey("under_ablation", sp["D"]): {}, "per_basis": None}
    for arm in ARMS:
        sc = _scores(arm, sp["R"] if arm in Q.R_ONLY else range(196), f_shift)
        sink.event("arm", arm=arm)
        Q.record_arm(rec, arm, "zero", sc, {"arm": arm}, CORPUS, _Tok())
        ast[P.wkey("under_ablation", sp["D"])][arm] = {"N": np.ones(3), "den": np.ones(3), "ratio": np.ones(3)}
        sink.json("arm_stats.json", ast)
    R_w = sp["R"]
    ref = {"ppl": {g: 6.0 for g in Q.G14_ARMS.values()}, "dnll_core32": 0.024, "E_over_C": 5.5, "source": "test"}
    run_ppl = {a: rec.summary[a][P.wkey("ppl", R_w)] if a in rec.summary else 6.0 for a in Q.G14_ARMS}
    g = Q.g14pre_gate(run_ppl, rec.summary["band"][P.wkey("dnll", R_w)], Q.run_e_over_c(rec.nll_open, R_w), ref, R_w)
    sink.json("g14pre.json", g)
    sink.json("metrics.json", Q.metrics_doc(rec, {"G14-PRE": g}, {"smoke": False, "stopped": None}, {"mask": "native"}))


def _open_files(root):
    return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file() and p.relative_to(root).parts[0] != "sealed")


def test_writers_route_windows_from_98_only_into_sealed(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    _run_writers(a, 0.0)
    log_a = capsys.readouterr().out
    _run_writers(b, 1000.0)
    log_b = capsys.readouterr().out
    assert _open_files(a) == _open_files(b) and (a / "samples/discovery/band.json").exists()
    for rel in _open_files(a):                       # nothing outside sealed/ depends on windows >= 98
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel
    assert log_a == log_b and all(int(m.group(2)) < Q.F0 for m in P.WKEY.finditer(log_a))
    rows = [json.loads(x) for x in (a / "sealed/window_nll_confirm.jsonl").read_text().splitlines()]
    assert [r["arm"] for r in rows] == [x for x in ARMS if x not in Q.R_ONLY]
    assert all(r["windows"] == list(range(98, 196)) for r in rows)
    rd = [json.loads(x) for x in (a / "window_nll_rd.jsonl").read_text().splitlines()]
    assert [r["arm"] for r in rd] == ARMS and all(r["windows"] == list(range(98)) for r in rd)
    sealed = json.loads((a / "sealed/samples/band.json").read_text())
    assert {w["index"] for w in sealed["windows"]} <= set(range(98, 196))
    assert {w["index"] for w in json.loads((a / "samples/discovery/band.json").read_text())["windows"]} <= \
        set(range(64, 98))
    assert not (a / "sealed/samples/topenergy128.json").exists()
    assert (a / "sealed/window_nll_confirm.jsonl").read_bytes() != (b / "sealed/window_nll_confirm.jsonl").read_bytes()
    sink = P.Sink(a, "discovery", now=lambda: T0)        # the write guard fires on a leak
    for leak in ({"ppl_w0_195": 1.0}, {"windows": [97, 98]}, {"x": {"index": 150}}):
        with pytest.raises(AssertionError):
            sink.json("metrics.json", leak)


# ---------------- G14-PRE ----------------

def test_g14pre_constants_are_gb3s_banked_numbers():
    v = json.loads((RESEARCH / "gutenberg_gb3_verdict.json").read_text())
    assert Q.GB3_DNLL_CORE == v["dnll"]["core32"][0] and Q.GB3_E_OVER_C == v["E_over_C"][0]
    assert Q.G14_TOL == v["reproduction_gate"]["tol"] == 1e-3
    assert sorted(Q.G14_ARMS.values()) == v["reproduction_gate"]["checked"]
    assert list(Q.GB3_EM.values()) == v["fresh_draw_arms"]["ematch"] and v["n_windows"] == Q.F0
    assert list(Q.GB3_SPEC.values()) == v["fresh_draw_arms"]["spec"]


def _gb3_wnll(n=98, seed=3):
    """GB-3-shaped per-window NLL (v1abl_window_nll.json) with arm-dependent costs."""
    rng = np.random.default_rng(seed)
    base = 1500 + rng.random(n)
    cost = {"none": 0.0, "core32": 12.0, **{a: 66.0 + 4 * i for i, a in enumerate(Q.GB3_EM.values())},
            **{a: 60.0 for a in Q.GB3_SPEC.values()},
            **{g: 5.0 for g in Q.G14_ARMS.values() if g not in ("none", "core32")}}
    pw = {a: {"nll": (base + c * (1 + 0.1 * rng.random(n))).tolist(), "ntok": [511] * n} for a, c in cost.items()}
    return {"per_window": pw, "n_abl_windows": 98}


def test_g14pre_references_follow_gb3s_own_aggregation():
    """On fewer windows (the smoke), the references are gutenberg_core.build_gb3_metrics' dnll.core32[0] and
    E_over_C[0] on the same windows, and ppl over them; on windows 0-97 a per-window file that does not reproduce
    GB-3's banked numbers is refused."""
    wn = _gb3_wnll()
    R_w = list(range(6))
    sub = {a: {"nll": v["nll"][:6], "ntok": v["ntok"][:6]} for a, v in wn["per_window"].items()}
    gb3 = G.build_gb3_metrics(sub, None, {}, n_boot=10)
    ref = Q.g14pre_references(R_w, None, wn)
    assert ref["dnll_core32"] == pytest.approx(gb3["dnll"]["core32"][0], rel=1e-12)
    assert ref["E_over_C"] == pytest.approx(gb3["E_over_C"][0], rel=1e-12)
    assert ref["ppl"]["none"] == pytest.approx(np.exp(sum(sub["none"]["nll"]) / (6 * 511)))
    nll_open = {ours: dict(enumerate(sub[g]["nll"])) for ours, g in
                [("none", "none"), ("band", "core32")] + [(f"em_s{s}", g) for s, g in Q.GB3_EM.items()]}
    assert Q.run_e_over_c(nll_open, R_w) == pytest.approx(gb3["E_over_C"][0], rel=1e-12)
    rows = {"arms": [{"arm": g, "ppl": 6.0} for g in Q.G14_ARMS.values()], "n_abl_windows": 98}
    with pytest.raises(AssertionError, match="banked"):
        Q.g14pre_references(list(range(98)), rows, wn)


def test_g14pre_fails_on_a_planted_wrong_number():
    R_w = list(range(98))
    ref = {"ppl": {g: 6.0 + i for i, g in enumerate(Q.G14_ARMS.values())}, "dnll_core32": Q.GB3_DNLL_CORE,
           "E_over_C": Q.GB3_E_OVER_C, "source": "test"}
    ppl = {a: ref["ppl"][g] for a, g in Q.G14_ARMS.items()}
    ok = Q.g14pre_gate(ppl, Q.GB3_DNLL_CORE, Q.GB3_E_OVER_C, ref, R_w)
    assert ok["pass"] and ok["band_dnll"]["tol"] == 0.002 and "ppl_w0_97" in ok["rows"]["band"]
    bad = Q.g14pre_gate({**ppl, "topsigma32": ppl["topsigma32"] * 1.0015}, Q.GB3_DNLL_CORE, Q.GB3_E_OVER_C, ref, R_w)
    assert not bad["pass"] and [a for a, r in bad["rows"].items() if not r["pass"]] == ["topsigma32"]
    assert Q.g14pre_gate({**ppl, "none": ppl["none"] * 1.0009}, Q.GB3_DNLL_CORE, Q.GB3_E_OVER_C, ref, R_w)["pass"]
    assert not Q.g14pre_gate(ppl, Q.GB3_DNLL_CORE + 0.0021, Q.GB3_E_OVER_C, ref, R_w)["pass"]
    assert Q.g14pre_gate(ppl, Q.GB3_DNLL_CORE - 0.0019, Q.GB3_E_OVER_C, ref, R_w)["pass"]
    assert not Q.g14pre_gate(ppl, Q.GB3_DNLL_CORE, Q.GB3_E_OVER_C * 1.06, ref, R_w)["pass"]
    assert Q.g14pre_gate(ppl, Q.GB3_DNLL_CORE, Q.GB3_E_OVER_C * 0.96, ref, R_w)["pass"]


# ---------------- inputs, corpus, provenance, host ----------------

def test_input_table_covers_every_key_the_cell_reads():
    t = Q.input_table()
    banks = {Q.BANK.format(L=L, bank=b) for L in range(Q.N_LAYERS) for b in ("core", "anti")}
    assert set(t) == banks | {Q.TOPE, Q.V1A_ROWS, Q.GB3_BANKS, Q.GB3_WNLL}
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in t.values()) and len(set(t.values())) == len(t)
    assert Q.INPUTS == "s3://pg-lineage-769063704696/results/mdl/realizednull_q14_inputs"


def test_fetch_reads_the_mirror_and_refuses_a_wrong_hash(tmp_path, monkeypatch):
    """fetch_inputs copies exactly INPUTS/<original key> for every input and the corpus, and verify_inputs fails on a
    file whose sha256 differs from the one recorded at the original prefix."""
    mirror = tmp_path / "mirror"
    table = {}
    for k, rel in enumerate([Q.BANK.format(L=0, bank="core"), Q.BANK.format(L=0, bank="anti"), Q.TOPE, Q.V1A_ROWS,
                             Q.GB3_BANKS, Q.GB3_WNLL, Q.CORPUS, Q.CORPUS_META]):
        (mirror / rel).parent.mkdir(parents=True, exist_ok=True)
        (mirror / rel).write_bytes(f"input {k}".encode())
        if rel not in (Q.CORPUS, Q.CORPUS_META):
            table[rel] = P.sha256_file(mirror / rel)
    seen = []

    def fake_run(cmd, check):
        assert cmd[:3] == ["aws", "s3", "cp"] and check
        args = [x for x in cmd[3:] if not x.startswith("--")]
        src, dst = args
        assert src.startswith(Q.INPUTS + "/")
        seen.append(src[len(Q.INPUTS) + 1:])
        s = mirror / src[len(Q.INPUTS) + 1:]
        shutil.copytree(s, dst) if "--recursive" in cmd else (
            pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True), shutil.copy(s, dst))
    monkeypatch.setattr(Q.subprocess, "run", fake_run)
    monkeypatch.setattr(Q, "input_table", lambda: table)
    got = Q.fetch_inputs(tmp_path / "in")
    assert got == table and seen == [Q.BANKS_DIR + "/", Q.TOPE, Q.V1A_ROWS, Q.GB3_BANKS, Q.GB3_WNLL, Q.CORPUS,
                                     Q.CORPUS_META]
    (tmp_path / "in" / Q.TOPE).write_bytes(b"a partial copy")
    with pytest.raises(AssertionError, match="stale or partial"):
        Q.verify_inputs(tmp_path / "in", table)


def test_corpus_check_and_the_corpus_writer(tmp_path):
    """N9: windows_from_ids is prepare_corpus.py's no-BOS cut, the writer writes (196, 512) int64 with its file hash,
    windows 0-97 must hash to GB-3's baked corpus; N7's check_corpus refuses a planted mismatch."""
    ids = np.random.default_rng(0).integers(0, 152064, 196 * 512 + 77)
    w = N9.windows_from_ids(ids)
    assert w.shape == (196, 512) and w.dtype == np.int64 and np.array_equal(w.ravel(), ids[:196 * 512])
    with pytest.raises(AssertionError, match="too short"):
        N9.windows_from_ids(ids[:1000])
    doc = N9.write_corpus(w, tmp_path, {"tokenizer": "tok@rev", "dataset": "wikitext"})
    back = np.load(tmp_path / "corpus196.npy")
    assert back.shape == (196, 512) and back.dtype == np.int64 and np.array_equal(back, w)
    assert doc == json.loads((tmp_path / "corpus196.json").read_text())
    assert doc["file_sha256"] == P.sha256_file(tmp_path / "corpus196.npy") and doc["shape"] == [196, 512]
    assert doc["first98_array_sha256"] == Q.array_sha256(w[:98]) == N9.check_baked(w, Q.array_sha256(w[:98]))
    with pytest.raises(AssertionError):
        N9.check_baked(w)                                   # random ids are not GB-3's corpus
    with pytest.raises(AssertionError):
        N9.write_corpus(w[:98], tmp_path, {})
    meta = json.loads((tmp_path / "corpus196.json").read_text())
    rec = Q.check_corpus(back, meta, doc["file_sha256"], back[:98].copy())
    assert rec["first98_equal_to_baked"] and rec["file_sha256"] == doc["file_sha256"]
    planted = back[:98].copy()
    planted[50, 7] += 1
    for args in ((back, meta, "0" * 64, back[:98]), (back[:98], {**meta, "shape": [98, 512]}, doc["file_sha256"],
                                                     back[:98]), (back, meta, doc["file_sha256"], planted)):
        with pytest.raises(AssertionError):
            Q.check_corpus(*args)


def test_corpus_job_constants_and_order():
    """N9's tokenizer revision is the commit shackleton:v1-14b baked, its baked-corpus hash has the prefix the
    shackleton manifest records, and main() calls hf_setup() before it loads the tokenizer or the dataset."""
    man = json.loads((RESEARCH / "shackleton_manifest.json").read_text())
    assert N9.REVISION == man["models"]["Qwen/Qwen2.5-14B"]["sha"]
    assert N9.GB3_CORPUS98_SHA256.startswith(man["corpus"]["sha256_prefix"])
    assert man["corpus"]["revision"] == N9.WIKITEXT_REV
    src = inspect.getsource(N9.main)
    assert src.index("hf_setup()") < src.index("AutoTokenizer.from_pretrained") < src.index("load_dataset(")
    assert src.index("check_baked(") < src.index("write_corpus(")


def test_code_provenance_reads_the_shipped_commit(tmp_path):
    sha = "aa583c922d57bd36d2ac8160f032e54158db51bc"
    p = tmp_path / "c.tgz"
    with tarfile.open(p, "w:gz", format=tarfile.PAX_FORMAT, pax_headers={"comment": sha}) as t:
        data = b"{}"
        info = tarfile.TarInfo("infra/vast/program.json")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    prov = Q.code_provenance(p)
    assert prov["commit_sha"] == sha and prov["program"] == "realizednull"
    assert prov["code_archive_sha256"] == P.sha256_file(p)
    q = tmp_path / "plain.tgz"
    with tarfile.open(q, "w:gz") as t:
        t.add(p, arcname="x")
    with pytest.raises(AssertionError, match="no commit id"):
        Q.code_provenance(q)


def test_host_ram_reads_the_smallest_limit(tmp_path):
    mi = tmp_path / "meminfo"
    mi.write_text("MemTotal:       65970000 kB\nMemFree:  1 kB\n")
    v2, v1 = tmp_path / "memory.max", tmp_path / "limit"
    v2.write_text("max\n")
    v1.write_text(str(40 * 2 ** 30) + "\n")
    assert Q.host_ram_gib(mi, (v2,)) == pytest.approx(65970000 / 2 ** 20)
    assert Q.host_ram_gib(mi, (v2, v1, tmp_path / "absent")) == pytest.approx(40.0)


def test_ematch_floor_is_k_over_5120():
    t = [0.005, 0.01, 0.02]
    assert Q.ematch_gate(t, [0.009, 0.0105, 0.02], 5120)["pass"]               # 0.005 < 32/5120: not checked
    g = Q.ematch_gate(t, [0.005, 0.0112, 0.02], 5120)
    assert not g["pass"] and g["failing_layers"] == [1] and g["floor_k_over_d"] == 32 / 5120


# ---------------- integration with N1: every N1 call N7 makes, on inputs N7 builds ----------------

N_TOY, V_TOY, S_TOY, NL_TOY, K_TOY = 16, 64, 24, 6, 2
KW_TOY = {"k": K_TOY, "n_grid": 8, "n_probe": 8, "tol_sum": 0.005, "tol_layer": 0.05, "heavy_frac": 0.01,
          "max_resample": 2000, "max_closing": 50, "max_redraws": 3}


class _Attn(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.q_proj = torch.nn.Linear(n, n, bias=True)                # Qwen2's q_proj has a bias
        self.k_proj, self.v_proj, self.o_proj = (torch.nn.Linear(n, n, bias=False) for _ in range(3))


class _Layer(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.self_attn = _Attn(n)


class _ToyLM(torch.nn.Module):
    """A text-only causal LM called as AutoModelForCausalLM is (input_ids only), its decoder layers at model.layers.
    Token id+1 gets +8 on the logits, so top-1 is right where the next token is the successor."""

    def __init__(self, causal=True):
        super().__init__()
        torch.manual_seed(0)
        self.causal = causal
        self.emb = torch.nn.Embedding(V_TOY, N_TOY)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(_Layer(N_TOY) for _ in range(NL_TOY))
        self.out = torch.nn.Linear(N_TOY, V_TOY, bias=False)

    def forward(self, input_ids):
        h = self.emb(input_ids)
        S = h.shape[1]
        add = torch.full((S, S), float("-inf")).triu(1) if self.causal else torch.zeros(S, S)
        for layer in self.model.layers:
            a = layer.self_attn
            w = torch.softmax(a.q_proj(h) @ a.k_proj(h).transpose(-1, -2) / N_TOY ** 0.5 + add, dim=-1)
            h = h + a.o_proj(w @ a.v_proj(h))
        succ = torch.nn.functional.one_hot((input_ids + 1) % V_TOY, V_TOY)
        return SimpleNamespace(logits=self.out(h) + 8.0 * succ)


def _toy_corpus():
    rng = np.random.default_rng(5)
    c = rng.integers(0, V_TOY - 10, (6, S_TOY))
    c[:, 8:14] = np.arange(30, 36)                         # a successor run: right top-1 predictions exist
    return c.astype(np.int64)


@pytest.fixture(scope="module")
def toy():
    """N7's pipeline on the toy, as run() orders it: Grams on C (0-3) and D (4-5), eigen_stage, class means, a k = 2
    band inside the top-6 eigenvectors, band targets, grids and two draws of every family, verification on C, and
    the per-layer arm_stats / G-GEN pass the daemon thread runs."""
    model = _ToyLM().eval()
    layers = model.model.layers
    corpus = _toy_corpus()
    fw = Q.Forward(model, layers, corpus, "cpu", V_TOY)
    C_, D_ = [0, 1, 2, 3], [4, 5]
    Gm, sums = Q.gram_pass(fw, C_, class_sums=True)
    GD, _ = Q.gram_pass(fw, D_, class_sums=False)
    SC = [g.clone().numpy() for g in Gm]
    W64 = [layer.self_attn.q_proj.weight.detach().double().numpy() for layer in layers]
    Ws32 = [layer.self_attn.q_proj.weight.detach().float() for layer in layers]
    V0 = [Q.eigen_layer(Gm[L], layers[L].self_attn.q_proj.weight)["V"] for L in range(NL_TOY)]
    rng = np.random.default_rng(7)
    band = [V0[L][:, :6] @ G.haar_frame(6, K_TOY, rng) for L in range(NL_TOY)]
    spill = pathlib.Path(__import__("tempfile").mkdtemp())
    tope32 = [np.arange(K_TOY) for _ in range(NL_TOY)]
    eg = Q.eigen_stage(Gm, [layer.self_attn.q_proj.weight for layer in layers], band, tope32, spill)   # as run()
    mu, counts = Q.class_means_q14(sums, S_TOY)
    fam_in = {"N_T": (eg["Mp"], eg["Sp"], eg["V"]), "E_T": ([np.eye(N_TOY)] * NL_TOY, eg["Sp"], eg["V"])}
    built = Q.construct_all(fam_in, band, {f: [0, 1] for f in Q.FAMS}, KW_TOY)      # as run(): in the main process
    tgt, grids, cons, bases, names = (built[k] for k in ("tgt", "grids", "cons", "bases", "fam_draws"))
    assert built["t_grid"] > 0 and built["t_con_s"] > 0
    hookq = Q.verify_pass(fw, C_, P.verify_entries(band, bases, cons), Ws32)
    SD = [g.numpy() for g in GD]
    U_by_basis = {"band": band, **{n: [u.astype(np.float32) for u in U] for n, U in bases.items()}}
    gen_keys = {"band|N_T": ("band", "N_T"), "band|E_T": ("band", "E_T"),
                **{n: (n, P.FAMILIES[cons[n]["family_name"]]["q"]) for n in bases}}
    means = Q.stats_means(mu, counts, sums, S_TOY)
    V32 = [v.astype(np.float32) for v in eg["V"]]
    load = lambda L: (np.load(spill / f"sigmaC_L{L}.npy").astype(np.float64), SD[L].astype(np.float64),
                      V32[L].astype(np.float64), W64[L])
    th, box = Q.start_thread("q14-stats", Q.stats_all_layers, NL_TOY, load, U_by_basis, means, gen_keys)
    stats, gen = Q.assemble_stats(Q.thread_result(th, box, wait=True))
    shutil.rmtree(spill)
    return SimpleNamespace(model=model, layers=layers, corpus=corpus, fw=fw, C=C_, D=D_, SC=SC, SD=SD, sums=sums,
                           W64=W64, Ws32=Ws32, band=band, eg=eg, mu=mu, counts=counts, fam_in=fam_in, tgt=tgt,
                           grids=grids, cons=cons, bases=bases, names=names, hookq=hookq, U_by_basis=U_by_basis,
                           V32=V32, stats=stats, gen=gen, tope32=tope32)


def _capture(toy, L, windows):
    """Every q_proj input row at layer L over the windows, with its position."""
    H = []
    hk = toy.layers[L].self_attn.q_proj.register_forward_pre_hook(lambda _m, a: H.append(a[0].double().clone()))
    toy.fw.plain(windows, [])
    hk.remove()
    H = torch.cat([h.reshape(-1, N_TOY) for h in H]).numpy()
    return H, np.tile(np.arange(S_TOY), len(windows))


def test_eigen_layer_gives_n1_eigencoords_quantities(toy):
    """eigen_layer (fp32 eigh and products, the §2.2 path for d = 5120) against N1's eigencoords in fp64: V
    orthonormal, the same eigenvalues and denominator, and the same realized quantity N(U) for any basis, which is
    the number every construction and target reads."""
    rng = np.random.default_rng(1)
    for L in (0, 3):
        S64, M64 = toy.SC[L].astype(np.float64), toy.W64[L].T @ toy.W64[L]
        ref = R.eigencoords(S64, M64, S64)
        ec = {k: toy.eg[k][L] for k in ("V", "mu", "Mp", "Sp", "denom")}
        assert np.abs(ec["V"].T @ ec["V"] - np.eye(N_TOY)).max() < 1e-12
        assert ec["mu"] == pytest.approx(ref["mu"], rel=1e-4, abs=1e-5 * ref["mu"][0])
        assert ec["denom"] == pytest.approx(ref["denom"], rel=1e-10)
        U = np.linalg.qr(rng.standard_normal((N_TOY, K_TOY)))[0]
        for Mp, Mp_ref in ((ec["Mp"], ref["Mp"]), (np.eye(N_TOY), np.eye(N_TOY))):
            got = R.realized_quantity(ec["V"].T @ U, Mp, ec["Sp"])
            want = R.realized_quantity(ref["V"].T @ U, Mp_ref, ref["Sp"])
            assert got == pytest.approx(want, rel=1e-4)


def test_n1_class_means_and_arm_stats_call_sites(toy):
    """class_means on gram_pass' sums (two classes: position 0, positions 1-23 here), and arm_stats on one-layer
    lists, assembled over layers, equal to N1's arm_stats called once on every layer, and to direct sums over the
    hook's rows."""
    assert toy.mu[0].shape == (2, N_TOY) and toy.counts == {0: 4, "text": 4 * (S_TOY - 1)}
    L = 2
    H, pos = _capture(toy, L, toy.C)
    mu_true = np.stack([H[pos == 0].mean(0), H[pos > 0].mean(0)])
    assert np.allclose(toy.mu[L], mu_true, rtol=1e-4, atol=1e-5)
    S64 = [s.astype(np.float64) for s in toy.SC]
    SD64 = [s.astype(np.float64) for s in toy.SD]
    Ms = [W.T @ W for W in toy.W64]
    V = [v.astype(np.float64) for v in toy.V32]
    g = lambda S: {"all": S, "T": S, "V": V, "denom_all": [float((M * s).sum()) for M, s in zip(Ms, S)],
                   "denom_T": [float((M * s).sum()) for M, s in zip(Ms, S)]}
    meansC = {"mu": toy.mu, "counts": [np.array([4.0, 4.0 * (S_TOY - 1)])] * NL_TOY, "text": [1] * NL_TOY,
              "T_sum": [toy.sums["pos0"][L_] + toy.sums["rest"][L_] for L_ in range(NL_TOY)],
              "T_count": [4 * S_TOY] * NL_TOY}
    for b in ("band", sorted(toy.bases)[0]):
        wantC = R.arm_stats(toy.U_by_basis[b], g(S64), toy.W64, meansC)
        wantD = R.arm_stats(toy.U_by_basis[b], g(SD64), toy.W64, None)
        for got, want in ((toy.stats["C"][b], wantC), (toy.stats["D"][b], wantD)):
            assert set(got) == set(want)
            for k in want:
                assert np.allclose(got[k], want[k], rtol=1e-10, atol=0), (b, k)
    st = toy.stats["C"]["band"]
    U, W = toy.band[L], toy.W64[L]
    Hc = H - mu_true[(pos > 0).astype(int)]
    assert st["e_all_centered"][L] == pytest.approx(((Hc @ U) ** 2).sum() / (Hc ** 2).sum(), rel=1e-4)
    assert st["N_T"][L] == pytest.approx(((H @ U @ U.T @ W.T) ** 2).sum(), rel=1e-4)
    assert st["rho_T"][L] == pytest.approx(st["N_T"][L] / ((H @ W.T) ** 2).sum(), rel=1e-4)
    HD, _ = _capture(toy, L, toy.D)
    assert toy.stats["D"]["band"]["rho_all"][L] == pytest.approx(
        ((HD @ U @ U.T @ W.T) ** 2).sum() / ((HD @ W.T) ** 2).sum(), rel=1e-4)
    assert toy.eg["e_target"][L] == pytest.approx(((H @ U) ** 2).sum() / (H ** 2).sum(), rel=1e-4)
    assert toy.eg["tope_overlap"][L] == len(set(np.argsort(np.diag(toy.SC[L]))[::-1][:Q.K].tolist()) & {0, 1})


def test_n1_realized_quantity_reach_grid_construct_draw_call_sites(toy):
    """band_targets, quantity_grids and construct_family on eigen_stage's matrices: targets equal the hook-measured
    band quantity over every position, grids reproduce with seed 190_000 + L, base seeds are §3.2's, and a matched
    draw's basis reproduces its constructed N through verify_pass and passes G-MATCH (N1's rule)."""
    for q, t in toy.tgt.items():
        assert t.shape == (NL_TOY,) and np.all(t > 0)
        assert toy.hookq[f"band|{q}"] == pytest.approx(t, rel=1e-4)
    Mp, Sp, _ = toy.fam_in["N_T"]
    again = R.reach_grid(Mp[3], Sp[3], k=K_TOY, n_grid=8, n_probe=8, seed=R.GRID_SEED_BASE + 3)
    assert all(np.array_equal(again[k], toy.grids["N_T"][3][k]) for k in again)
    assert list(toy.names) == list(Q.FAMS) == ["rt", "rthalf", "rt2x", "emt"]           # §3.8: no RT_all
    assert all(names == [f"{fam}_s0", f"{fam}_s1"] for fam, names in toy.names.items())
    matched = []
    for fam, names in toy.names.items():
        assert [toy.cons[n]["base_seed"] for n in names] == [P.FAMILIES[fam]["seed"] + 1000 * s for s in (0, 1)]
        for n, r in zip(names, P.verify_rows(toy.hookq, names, fam)):
            assert (r is None) == (not toy.cons[n]["matched"])
            if r is not None:
                matched.append(n)
                assert r["N"] == pytest.approx(toy.cons[n]["N"], rel=1e-4)
                assert P.match_gate(r["N"], r["targets"], toy.C)["pass"] == R._match_ok(r)
    assert matched, "no draw matched in the toy: the verification path did not run"


def test_n1_gen_rows_equal_n3s_on_whole_layer_lists(toy):
    """G-GEN from the per-layer quantities the thread computes equals N3's gen_rows over whole-layer lists, and its
    pass agrees with N1's rule."""
    Ms = [W.T @ W for W in toy.W64]
    SD64 = [s.astype(np.float64) for s in toy.SD]
    opD = {"N_T": (Ms, SD64), "E_T": ([np.eye(N_TOY)] * NL_TOY, SD64)}
    bandD = {q: P.summed_quantity(toy.band, *opD[q]) for q in opD}
    bases32 = {n: toy.U_by_basis[n] for n in toy.bases}
    for fam, names in toy.names.items():
        got = Q.gen_rows(names, fam, toy.gen)
        want = P.gen_rows(names, bases32, fam, opD, bandD)
        for x, y in zip(got, want):
            assert (x is None) == (y is None)
            if x is not None:
                assert x["N_sum"] == pytest.approx(y["N_sum"], rel=1e-10)
                assert x["target_sum"] == pytest.approx(y["target_sum"], rel=1e-10)
                assert R._gen_ok(x) == (abs(x["N_sum"] / x["target_sum"] - 1) <= R.GEN_TOL)
    gen = {"band|N_T": [1.0, 2.0], "band|E_T": [4.0, 4.0], "rthalf_s0": [0.5, 1.0], "rt2x_s0": [3.0, 3.0],
           "emt_s0": [4.0, 4.4]}
    assert Q.gen_rows(["rthalf_s0", "rthalf_s1"], "rthalf", gen) == [{"N_sum": 1.5, "target_sum": 1.5}, None]
    assert Q.gen_rows(["rt2x_s0"], "rt2x", gen) == [{"N_sum": 6.0, "target_sum": 6.0}]
    assert Q.gen_rows(["emt_s0"], "emt", gen) == [{"N_sum": 8.4, "target_sum": 8.0}]


def test_n1_sample_record_call_site(toy, tmp_path):
    """sample_payload hands N1 sample_record a text-only window and Forward.scores' per-position arrays: roles, the
    window as decoded, N1's NLL sums equal the readout, Y, and a right and a wrong example."""
    tok = _Tok()
    sp = P.splits(6, 4)
    none = toy.fw.scores(range(6))
    U = [torch.as_tensor(b, dtype=torch.float32) for b in toy.band]
    state = {}
    hooks = [(layer.self_attn.q_proj, Q.ablation_hook(U[L], "zero", None, state, toy.Ws32[L],
                                                      {"num": [0.0] * NL_TOY, "den": [0.0] * NL_TOY}, L))
             for L, layer in enumerate(toy.layers)]
    arm = toy.fw.scores(range(6), hooks, state, set(sp["D"]))
    rec = P.Recorder(P.Sink(tmp_path, "smoke", now=lambda: T0), sp, ["none", "band"])
    rec.nll("none", "none", none)
    rec.nll("band", "zero", arm)
    sel, Y = rec.select(arm)
    pay = Q.sample_payload({"arm": "band"}, sel["open"], arm, rec.none, toy.corpus, tok)
    assert [w["role"] for w in pay["windows"]] == ["max", "min", "common"] and sel["open"]["common"] == 4
    for w in pay["windows"]:
        i = w["index"]
        assert w["text_seen"] == tok.decode(toy.corpus[i]) and w["ntok"] == S_TOY - 1 == len(w["positions"])
        assert w["nll_sum_none"] == pytest.approx(none[i]["nll"], rel=1e-5)
        assert w["nll_sum_arm"] == pytest.approx(arm[i]["nll"], rel=1e-5)
        assert w["Y"] == pytest.approx(Y[i], rel=1e-4, abs=1e-6)
    ex = pay["examples"]["none"]
    assert ex["right"]["right_none"] and not ex["wrong"]["right_none"] and ex["right"]["position"] == \
        ex["right"]["scored_index"]


def test_n1_mask_probe_verdict_on_the_text_probe():
    """text_probe → mask_probe_verdict with NO_IMAGE_BLOCK: the causal toy reads causal; a bidirectional toy, the
    registered failing input, reads bidir."""
    corpus = _toy_corpus()
    for causal, want in ((True, "causal"), (False, "bidir")):
        m = _ToyLM(causal=causal).eval()
        fw = Q.Forward(m, m.model.layers, corpus, "cpu", V_TOY)
        p = Q.text_probe(fw, torch.as_tensor(corpus[0]), 3, 4, edit=8)
        assert (p["edit_start"], p["r_prior"], p["r_self"]) == (S_TOY - 8, 0.0, 1.0)
        assert (p["d_before"] == 0.0) == causal and p["d_after"] >= 0.05
        assert Q.probe_mode(p) == want


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_thread_failure_is_loud():
    """An error in the arm_stats / G-GEN thread fails the cell at the next check; its traceback goes to stderr."""
    def planted():
        raise ValueError("planted")
    th, box = Q.start_thread("q14-test", planted)
    with pytest.raises(AssertionError, match="thread q14-test died"):
        Q.thread_result(th, box, wait=True)
    th, box = Q.start_thread("q14-test", max, 3, 7)
    assert Q.thread_result(th, box, wait=True) == 7


def test_job_metrics_family_records_in_n1s_shape(toy):
    """The construction, verify_C and gen_D records run() writes into job_metrics have the keys and lengths N1's
    gates() reads, and N3's G-REACH/G-PIVOT/G-EXHAUST on them is N1's reading (tested in N3)."""
    gates = {}
    Q.family_records(toy.cons, toy.names, gates)
    for fam, names in toy.names.items():
        recs = [toy.cons[n] for n in names]
        ver = P.verify_rows(toy.hookq, names, fam)
        gen = Q.gen_rows(names, fam, toy.gen)
        assert len(recs) == len(ver) == len(gen) == 2
        assert {"reachable", "unreachable_layers", "pivot", "exhausted", "matched"} <= set(recs[0])
        assert gates["G-PIVOT"][fam]["share"] == sum(r["pivot"] for r in recs) / 2
        assert gates["G-REACH"][fam]["pass"] == all(r["reachable"] for r in recs)
    env = {k: "x" for k in R.ENV_KEYS}
    assert Q.env_gate({**env, "_attn_implementation": "sdpa"})["pass"]
    assert not Q.env_gate({**env, "_attn_implementation": "eager"})["pass"]
    assert Q.env_gate({**env, "gpu_name": None, "_attn_implementation": "sdpa"})["missing"] == ["gpu_name"]


# ---------------- run(): order of stages, and the start before any box step ----------------

def test_run_orders_its_stages():
    src = inspect.getsource(Q.run)
    order = ['sink.event("start"', "host_ram_gib()", "ensure_scipy()", "fetch_inputs(", "check_corpus(",
             'sink.event("model_load")', "AutoModelForCausalLM.from_pretrained(", 'sink.event("model_loaded")',
             'sink.event("probes")', 'rec.stage("gram_c")', "em_fresh(", 'sink.event("construct_all")',
             "construct_all(fam_in", 'sink.event("constructed"', 'sink.event("arm", arm="none")', "g14pre_gate(",
             'return stop("G14-PRE"', "if not needs_construction(", 'rec.stage("verify_c")',
             'rec.stage("gram_d")', 'start_thread("q14-stats"', "if needs_construction(",
             "thread_result(th, box, wait=True)"]
    at = [src.index(s) for s in order] + [src.rindex('sink.json("metrics.json"')]     # the first one is stop()'s
    assert at == sorted(at)


def test_construction_runs_in_the_main_process_before_any_arm_record():
    """§0.4 U5: run() calls construct_all itself (no construction thread), after the eigen-coordinates, and writes
    compute.json with t_con and t_grid_per_layer set before the first arm record (none's per-window NLL, or any
    run_arm) exists. Read from run()'s own statements; its nested writers (run_arm, stop) are set aside."""
    fn = ast.parse(textwrap.dedent(inspect.getsource(Q.run))).body[0]
    body = [x for x in fn.body if not isinstance(x, ast.FunctionDef)]

    def calls(node):
        return {c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}

    def assigns(node, name):
        return isinstance(node, ast.Assign) and name in {t.id for t in node.targets if isinstance(t, ast.Name)}
    first = lambda pred, lo=0: next(i for i in range(lo, len(body)) if pred(body[i]))
    i_con = first(lambda x: "construct_all" in calls(x))
    i_tcon, i_tgrid = (first(lambda x, n=n: assigns(x, n), i_con) for n in ("t_con", "t_grid"))
    i_write = first(lambda x: "write_compute" in calls(x), i_con)
    i_arm = first(lambda x: calls(x) & {"record_arm", "run_arm"})
    assert first(lambda x: "eigen_stage" in calls(x)) < i_con < max(i_tcon, i_tgrid) < i_write < i_arm
    starts = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "start_thread"]
    assert starts and not any(getattr(x, "id", None) == "construct_all" for c in starts for x in c.args)


class _BeforeBox(Exception):
    pass


def test_run_writes_start_then_stops_at_the_first_box_step(tmp_path, monkeypatch):
    def before_box():
        raise _BeforeBox
    monkeypatch.setattr(Q, "host_ram_gib", before_box)
    with pytest.raises(_BeforeBox):
        Q.run({"id": "q14_smoke", "smoke": True, "n_windows": 6, "n_construct": 4, "model_dir": "m"}, str(tmp_path))
    rows = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["start"] and rows[0]["cell"] == "q14_smoke"
    assert (tmp_path / "samples/smoke").is_dir() and (tmp_path / "sealed/samples").is_dir()


# ---------------- N1 and N5 on a session-2 run: the q14 parameter set ----------------

FULL = {"id": "q14_full", "smoke": False, "n_windows": 196, "n_construct": 64, "model_dir": "/models/Qwen_Qwen2.5-14B"}


def test_family_parameter_set_from_env_json():
    """N1's family_of picks the parameter set from env.json: N3's base for pg1 (R2's own env.json), N7's family field
    for q14. The q14 set carries GB-3's numbers read by key path, equal to N7's own constants, N7's shapes and arms,
    and no RT_all."""
    r2 = json.loads((ROOT / "studies/realizednull/results/pg1_full_a1/env.json").read_text())
    assert R.family_of(r2) is R.PG1 and R.PG1["attn"] == "eager" and R.PG1["ematch_floor"] == 32 / 2048
    assert '"family": "q14"' in inspect.getsource(Q.run)
    q = R.family_of({"family": "q14", "d_model": 5120, "n_layers": 48})
    with pytest.raises(ValueError, match="n_layers 18"):
        R.family_of({"family": "q14", "d_model": 5120, "n_layers": 18})
    with pytest.raises(ValueError, match="no parameter set"):
        R.family_params("q7")
    v = json.loads((RESEARCH / "gutenberg_gb3_verdict.json").read_text())
    pre = q["preflight"]
    assert {k: pre[k] for k in R.GB3_KEYS} == {k: R.key_path(v, path) for k, path in R.GB3_KEYS.items()}
    assert (pre["dnll_core32"], pre["E_over_C"], pre["ppl_rel"]) == (Q.GB3_DNLL_CORE, Q.GB3_E_OVER_C, Q.G14_TOL)
    assert (pre["dnll_tol"], pre["ec_tol"]) == (Q.G14_DNLL_TOL, Q.G14_EC_TOL)
    assert sorted(pre["checked"]) == sorted(Q.G14_ARMS.values()) and pre["em_arms"] == list(Q.GB3_EM.values())
    assert (q["attn"], q["d_model"], q["n_layers"], q["ematch_floor"]) == ("sdpa", Q.D_MODEL, Q.N_LAYERS, 32 / 5120)
    assert set(q["constructed"]) == set(Q.FAMS) and "rtall" not in q["args"]
    a, sl = Q.cell_args(FULL), Q.slots(Q.cell_args(FULL))
    assert set(q["args"]) == set(sl) and all(a[q["args"][f]] == len(sl[f]) for f in sl)
    assert q["scale_range"] == R.PG1["scale_range"] == (5.0, 500.0)
    assert "G14-PRE" in q["run_gates"] and not [g for g in q["run_gates"] if g.startswith("G-PRE")]


def _rec_q14(**kw):
    return {"reachable": True, "matched": True, "pivot": False, "exhausted": False, "unreachable_layers": [],
            "redraw": 0, **kw}


def _q14_inputs(ppl_edit=None):
    """N1's gates() input for a session-2 run that passes every gate: job_metrics as N7 writes it (48 layers, no RT_all,
    EM s3-s5 banked, sdpa), its g14pre record from N7's g14pre_gate on the same ppl, and the discovery side."""
    pre = R.family_params("q14")["preflight"]
    d = pre["dnll_core32"]
    base = {"none": 6.0, "band": 6.0 * math.exp(d),
            **{f"em_s{s}": 6.0 * math.exp(pre["E_over_C"] * d) for s in Q.GB3_EM},
            **{a: 6.0 * (1.5 + i) for i, a in enumerate(a for a in Q.G14_ARMS if a not in ("none", "band"))}}
    ppl = dict(base)
    if ppl_edit:
        ppl_edit(ppl)
    ec = np.mean([math.log(ppl[f"em_s{s}"] / ppl["none"]) for s in Q.GB3_EM]) / math.log(ppl["band"] / ppl["none"])
    ref = {"ppl": {g: base[a] for a, g in Q.G14_ARMS.items()}, "dnll_core32": Q.GB3_DNLL_CORE,
           "E_over_C": Q.GB3_E_OVER_C, "source": "test"}        # GB-3's rows (v1abl_rows.json) at the unedited run
    g14 = Q.g14pre_gate({a: ppl[a] for a in Q.G14_ARMS}, math.log(ppl["band"] / ppl["none"]), ec, ref, list(range(98)))
    sizes = {"rt": 40, "emt": 40, "rthalf": 3, "rt2x": 3}
    tg = list(np.linspace(1.0, 3.0, 48))
    jm = {"env": {"transformers": "4.51.3", "torch": "2.6.0", "numpy": "2.2.4", "cuda": "12.4",
                  "gpu_name": "NVIDIA RTX A6000", "instance_type": "vast.ai", "_attn_implementation": "sdpa"},
          "run": "R2", "mask": "native", "pre1_reference": None,
          "mask_probe": {"native": {"d_before": 0.0, "d_after": 1.5, **Q.NO_IMAGE_BLOCK}, "forced_masks": "off",
                         "forced": {}},
          "ppl_R": ppl, "g14pre": json.loads(json.dumps(g14, default=P._jsonable)),
          "ematch": [None if s in Q.GB3_EM else [{"e_target": 0.02, "e_realized": 0.0205}] * 48 for s in range(40)],
          "construction": {f: [_rec_q14() for _ in range(n)] for f, n in sizes.items()},
          "verify_C": {f: [{"N": tg, "targets": tg} for _ in range(n)] for f, n in sizes.items()},
          "gen_D": {f: [{"N_sum": 100.0, "target_sum": 100.0} for _ in range(n)] for f, n in sizes.items()},
          "readout_validity": {s: {"fit": 1.0, "positive_control": 2.0, "identity": 0.0} for s in "EMRT"}}
    dc = {"c_D": {"band": 0.024, "band_dup": 0.024, "ident_mean": 0.0, "topenergy32": 2.0, "topsigma32_mean": 1.0},
          "c_D_draws": {"haar": [0.0012, 0.0015, 0.0011, 0.0014, 0.0013], "haar_mean": [0.0008, 0.001, 0.0009],
                        "em": [0.13] * 40, "em_mean": [0.046] * 40, "rt": [0.024] * 40, "emt": [0.07] * 40,
                        "rthalf": [0.012] * 3, "rt2x": [0.048] * 3, "spec": [0.12] * 5},
          "Y_band_mean_D": list(0.016 * (1 + 0.1 * np.sin(np.arange(34)))),
          "id_max": {"ident_zero": 0.0, "ident_mean": 0.0}, "id_windows": "0-97",
          "boot": {s: {"n_degenerate": 0, "n_boot": 10000} for s in "EMRT"}, "f_H2": 0.05, "noleak": {"leaks": []},
          "planned_draws": {s: 40 for s in "EMRT"}, "families": {s: {"s": 0.10, "D": 40} for s in "EMRT"}}
    return jm, dc


def _em_ratio(f):
    def edit(ppl):
        d = math.log(ppl["band"] / ppl["none"])
        for s in Q.GB3_EM:
            ppl[f"em_s{s}"] = ppl["none"] * math.exp(f * R.family_params("q14")["preflight"]["E_over_C"] * d)
    return edit


def _band_shift(x):
    def edit(ppl):
        ppl["band"] *= math.exp(x)
        _em_ratio(1.0)(ppl)
    return edit


def _set(path, value):
    def edit(jm, dc):
        obj = {"jm": jm, "dc": dc}
        *keys, last = path
        for k in keys:
            obj = obj[k]
        if value is _set.POP:
            obj.pop(last)
        else:
            obj[last] = value(obj[last]) if callable(value) else value
    return edit


_set.POP = object()
# §6.1's registered failing inputs as they apply to O3 (§3.8): (case, ppl edit or None, job/discovery edit, gate)
Q14_FAILING = [
    ("G-ENV: eager at 14B", None, _set(("jm", "env", "_attn_implementation"), "eager"), "G-ENV"),
    ("G-ENV: missing version", None, _set(("jm", "env", "transformers"), _set.POP), "G-ENV"),
    ("G14-PRE: one ppl row 0.15% off", lambda p: p.update(topsigma32=p["topsigma32"] * 1.0015), None, "G14-PRE"),
    ("G14-PRE: band dNLL +0.0021", _band_shift(0.0021), None, "G14-PRE"),
    ("G14-PRE: E/C 6% high", _em_ratio(1.06), None, "G14-PRE"),
    ("G-ID1: leaky hook", None, _set(("dc", "id_max", "ident_zero"), 2e-3), "G-ID1"),
    ("G-ID3: leaky hook", None, _set(("dc", "id_max", "ident_mean"), 2e-3), "G-ID3"),
    ("G-ID2: nondeterministic", None, _set(("dc", "c_D", "band_dup"), 0.024 * 1.03), "G-ID2"),
    ("G-SCALE: band equals Haar", None, lambda jm, dc: dc["c_D"].update(band=0.0015, band_dup=0.0015), "G-SCALE"),
    ("G-SCALE: Haar zero", None, _set(("dc", "c_D_draws", "haar"), [0.0] * 5), "G-SCALE"),
    ("G-SCALE: GB-3's own 4.13", None, _set(("dc", "c_D_draws", "haar"), [0.024 / 4.13] * 5), "G-SCALE"),
    ("G-POS: dead", None, _set(("dc", "c_D", "topenergy32"), 0.01), "G-POS"),
    ("G-POS: nan", None, _set(("dc", "c_D", "topenergy32"), float("nan")), "G-POS"),
    ("G-EMATCH: 20% off above k/5120", None, lambda jm, dc: jm["ematch"][6].__setitem__(
        10, {"e_target": 0.01, "e_realized": 0.012}), "G-EMATCH"),
    ("G-REACH:rt", None, lambda jm, dc: jm["construction"].update(
        rt=[_rec_q14(reachable=False, matched=False, unreachable_layers=["sum"])] * 40), "G-REACH:rt"),
    ("G-PIVOT:rt", None, lambda jm, dc: jm["construction"]["rt"].__setitem__(
        slice(0, 11), [_rec_q14(pivot=True)] * 11), "G-PIVOT:rt"),
    ("G-EXHAUST:rt", None, lambda jm, dc: jm["construction"]["rt"].__setitem__(
        slice(0, 11), [_rec_q14(matched=False, exhausted=True, redraw=3)] * 11), "G-EXHAUST:rt"),
    ("G-MATCH:rt: 2% off", None, lambda jm, dc: jm["verify_C"]["rt"].__setitem__(
        0, {"N": list(1.02 * np.linspace(1.0, 3.0, 48)), "targets": list(np.linspace(1.0, 3.0, 48))}), "G-MATCH:rt"),
    ("G-GEN:rt: 6% off", None, lambda jm, dc: jm["gen_D"]["rt"].__setitem__(
        0, {"N_sum": 106.0, "target_sum": 100.0}), "G-GEN:rt"),
    ("G-DRAWS:R: 31 of 40", None, lambda jm, dc: jm["gen_D"]["rt"].__setitem__(
        slice(0, 9), [{"N_sum": 106.0, "target_sum": 100.0}] * 9), "G-DRAWS:R"),
    ("G-DOSE: flat", None, _set(("dc", "c_D_draws", "rthalf"), [0.024] * 3), "G-DOSE"),
    ("G-MEANPOS: dead", None, _set(("dc", "c_D", "topsigma32_mean"), 0.05), "G-MEANPOS"),
    ("G-BRACKET:R", None, _set(("dc", "c_D_draws", "rt"), [7.0] * 5 + [0.024] * 35), "G-BRACKET:R"),
    ("G-BOOT:E", None, _set(("dc", "boot", "E"), {"n_degenerate": 200, "n_boot": 10000}), "G-BOOT:E"),
    ("G-READOUT:R", None, lambda jm, dc: jm["readout_validity"]["R"].update(identity=None), "G-READOUT:R"),
    ("G-PSEUDO", None, _set(("dc", "f_H2"), 0.15), "G-PSEUDO"),
    ("G-NOLEAK", None, _set(("dc", "noleak", "leaks"), ["metrics.json: c_w0_195"]), "G-NOLEAK"),
]


def test_gates_on_a_q14_run_pass_and_each_registered_input_fails():
    """N1's gates() with the q14 parameter set: a session-2 run passes every gate (G14-PRE in place of G-PRE1-4, no
    RT_all records), and each registered failing input fails its gate; a run gate's failure is the verdict's
    RUN-DOES-NOT-COUNT. A g14pre.json that disagrees with G14-PRE read again raises."""
    q = R.family_params("q14")
    g = R.gates(*_q14_inputs(), q)
    assert [k for k, v in g.items() if not v["pass"]] == []
    assert "G14-PRE" in g and not [k for k in g if k.startswith("G-PRE") or k.endswith(":rtall")]
    assert g["G-SCALE"]["value"] == pytest.approx(0.024 / 0.0015) and g["G-MASK"]["value"]["native"] == "causal"
    for case, ppl_edit, edit, gid in Q14_FAILING:
        jm, dc = _q14_inputs(ppl_edit)
        if edit:
            edit(jm, dc)
        g = R.gates(jm, dc, q)
        assert not g[gid]["pass"], case
        if gid in q["run_gates"]:
            th = R.thresholds_from_controls(dc)
            v = R.verdict({"x_rt": [], "x_band": math.log(0.024)}, g, th, q)
            assert v["label"] == f"RUN-DOES-NOT-COUNT ({gid})", (case, v["label"])
    assert R.gates(*_q14_inputs(_band_shift(0.0021)), q)["G14-PRE"]["label"] == \
        "RUN-DOES-NOT-COUNT (G14-PRE: band, band ΔNLL)"                # the band's ppl row moves 0.21% too
    assert R.gates(*_q14_inputs(_em_ratio(1.06)), q)["G14-PRE"]["label"] == "RUN-DOES-NOT-COUNT (G14-PRE: E/C)"
    jm, dc = _q14_inputs()
    jm["g14pre"]["band_dnll"]["gb3"] = 0.025                    # the cell's reference differs from GB-3's JSON
    with pytest.raises(ValueError, match="g14pre.json differs"):
        R.gates(jm, dc, q)
    jm, dc = _q14_inputs()
    jm["g14pre"]["pass"] = False
    with pytest.raises(ValueError, match="g14pre.json differs"):
        R.gates(jm, dc, q)


# ---------------- N5 on a session-2 run directory written by N7's writers ----------------

NT = 8                    # tokens per window here (N7's are 512), so the 400-odd sample files stay small


def _q14_Y(a, pre):
    """Planted per-window Y (nats/token) for every arm of N7's full roster on windows 0-195. Draw i of a family costs
    level·θ·exp(s·z_i) in every window (θ_E 5.5, θ_M 2.9, θ_R 1.0, θ_T 2.9; s 0.10); then the C windows of the band and
    of EM s3-s5 are set so that their windows 0-97 give GB-3's dnll.core32[0] and E_over_C[0] (G14-PRE)."""
    w = np.arange(196)
    sb, sm = 1 + 0.2 * np.sin(w), 1 + 0.2 * np.cos(w)
    band = 0.0241
    Y = {"none": np.zeros(196), "band": band * sb, "band_dup": band * sb, "band_zt": 0.8 * band * sb,
         "band_mean": 0.016 * sm, "ident_zero": np.zeros(196), "ident_mean": np.zeros(196), "topenergy32": 2.0 * sb,
         "topsigma32": 1.5 * sb, "bottomsigma32": 5e-4 * sb, "anti32": 1.4 * sb, "topenergy128": 3.0 * sb[:98],
         "topenergy32_mean": 0.5 * sm, "topsigma32_mean": 1.0 * sm}
    fams = {"em": (band * 5.5, sb, 0.10), "em_mean": (0.016 * 2.9, sm, 0.10), "rt": (band, sb, 0.10),
            "emt": (band * 2.9, sb, 0.10), "spec": (band * 5.0, sb, 0.05), "rthalf": (band / 2, sb, 0.05),
            "rt2x": (band * 2, sb, 0.05), "haar": (0.0015, sb, 0.05), "em_zt": (0.1, sb, 0.05),
            "rt_zt": (0.02, sb, 0.05), "rt_mean": (0.016, sm, 0.05), "spec_mean": (0.06, sm, 0.05),
            "haar_mean": (0.001, sm, 0.05)}
    for fam, slots in Q.slots(a).items():
        lvl, shape, s = fams[fam]
        for i, z in enumerate(A._z(len(slots))):
            Y[f"{fam}_s{i}"] = lvl * math.exp(s * z) * shape
    targets = {"band": pre["dnll_core32"], **{f"em_s{s}": pre["E_over_C"] * pre["dnll_core32"] for s in Q.GB3_EM}}
    for arm, t in targets.items():
        Y[arm][:64] = (98 * t - Y[arm][64:98].sum()) / 64
    return Y


def _q14_scores(y, nll_none, corpus):
    """N7's per-window scores (score_row's keys) for one arm: NLL_sum = NLL_sum(none) + ntok·Y; top-1 right at even
    positions and wrong at odd ones."""
    out = {}
    for w in range(len(y)):
        tgt = corpus[w][1:]
        out[w] = {"nll": float(nll_none[w] + (NT - 1) * y[w]), "ntok": NT - 1,
                  "lp": -np.linspace(0.2, 2.0, NT - 1) - y[w], "top_id": np.where(np.arange(NT - 1) % 2, tgt + 1, tgt),
                  "top_lp": np.log(np.linspace(0.3, 0.9, NT - 1))}
    return out


def _q14_run_dir(out):
    """A full session-2 cell's out/ written by N7's own writers (Sink, Recorder, record_arm, g14pre_gate,
    family_records, verify_rows, gen_rows, metrics_doc) on planted per-window costs, with job.log beside it."""
    a, sl, pre = Q.cell_args(FULL), Q.slots(Q.cell_args(FULL)), R.family_params("q14")["preflight"]
    sp, arms = P.splits(196, 64), Q.roster(Q.cell_args(FULL))
    corpus = (np.arange(196 * NT, dtype=np.int64).reshape(196, NT) * 7) % 1000
    Y = _q14_Y(a, pre)
    nll_none = (NT - 1) * (math.log(6.0) + 0.05 * np.cos(np.arange(196)))
    sink = P.Sink(out, "discovery", now=lambda: T0)
    env = {"transformers": "4.51.3", "torch": "2.6.0", "numpy": "2.2.4", "scipy": "1.17.1",
           "scipy_installed_at_start": False, "cuda": "12.4", "gpu_name": "NVIDIA RTX A6000",
           "instance_type": "vast.ai",
           "_attn_implementation": "sdpa", "mask": "native", "forced_masks": "off", "args": a, "cell": FULL["id"],
           "provenance": {"program": "realizednull", "commit_sha": "0" * 40}, "family": "q14",
           "model_dir": FULL["model_dir"], "n_layers": 48, "d_model": 5120}
    sink.json("env.json", env)
    probe = {"d_before": 0.0, "d_after": 1.5, "edit_start": NT - 4, **Q.NO_IMAGE_BLOCK}
    probe["mode"] = Q.probe_mode(probe)
    sink.json("mask_probe.json", {"native": probe, "forced_masks": "off", "image_probe": "none: text-only model"})
    jm = {"env": {k: env[k] for k in R.ENV_KEYS + ("_attn_implementation",)}, "mask": "native",
          "mask_probe": {"native": {k: probe[k] for k in P.PROBE_KEYS}, "forced_masks": "off", "forced": {}}}
    gates = {"G-ENV": Q.env_gate(env), "G-MASK": {"native_mode": probe["mode"], "pass": True}}
    rec = P.Recorder(sink, sp, ["none"] + [x[0] for x in arms])
    sink.json("sealed/corpus_f.json", {"array_sha256": Q.array_sha256(corpus[98:])})
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        for arm, op, basis, _ in [("none", "none", None, None)] + arms:
            sc = _q14_scores(Y[arm], nll_none, corpus)
            Q.record_arm(rec, arm, op, sc, {"arm": arm, "op": op, "basis": basis}, corpus, _Tok())
    Rw = sp["R"]
    ref = {"ppl": {g: rec.summary[x][P.wkey("ppl", Rw)] * (1 + 2e-4) for x, g in Q.G14_ARMS.items()},
           "dnll_core32": Q.GB3_DNLL_CORE, "E_over_C": Q.GB3_E_OVER_C, "source": "synthetic v1abl_rows.json"}
    gates["G14-PRE"] = Q.g14pre_gate({x: rec.summary[x][P.wkey("ppl", Rw)] for x in Q.G14_ARMS},
                                     rec.summary["band"][P.wkey("dnll", Rw)], Q.run_e_over_c(rec.nll_open, Rw), ref, Rw)
    sink.json("g14pre.json", gates["G14-PRE"])
    jm["ematch"] = [None if s in Q.GB3_EM else [{"e_target": 0.02, "e_realized": 0.0205}] * 48 for s in sl["em"]]
    gates["G-EMATCH"] = {"n": len(sl["em"]) - 3, "n_pass": len(sl["em"]) - 3}
    fam_draws = {f: [f"{f}_s{s}" for s in sl[f]] for f in Q.FAMS}
    cons = {n: _rec_q14(family_name=f, slot=i, base_seed=P.FAMILIES[f]["seed"] + 1000 * i, fail_reason=None)
            for f, names in fam_draws.items() for i, n in enumerate(names)}
    jm["construction"] = {f: [cons[n] for n in names] for f, names in fam_draws.items()}
    Q.family_records(cons, fam_draws, gates)
    prof = np.concatenate([[1e-4], np.linspace(1.0, 3.0, 47)])
    band_q = {"N_T": prof, "N_all": prof, "E_T": 0.3 * prof}
    hookq = {**{f"band|{k}": v for k, v in band_q.items()},
             **{n: P.FAMILIES[cons[n]["family_name"]]["dose"] * band_q[P.FAMILIES[cons[n]["family_name"]]["q"]]
                for n in cons}}
    gen = {"band|N_T": list(1.01 * prof), "band|E_T": list(0.303 * prof),
           **{n: list(1.01 * hookq[n]) for n in cons}}
    for key in ("G-MATCH", "G-GEN"):
        gates[key] = {}
    jm["verify_C"], jm["gen_D"] = {}, {}
    for f, names in fam_draws.items():
        jm["verify_C"][f] = P.verify_rows(hookq, names, f)
        jm["gen_D"][f] = Q.gen_rows(names, f, gen)
        n_m = sum(P.match_gate(r["N"], r["targets"], sp["C"])["pass"] for r in jm["verify_C"][f])
        gates["G-MATCH"][f] = {"n": len(names), "n_pass": n_m}
        gates["G-GEN"][f] = {"n": len(names), "n_pass": sum(abs(r["N_sum"] / r["target_sum"] - 1) <= R.GEN_TOL
                                                            for r in jm["gen_D"][f])}
    level = {"band": 1.0, "em": 2.6, "emt": 1.8, "rt": 1.0, "rthalf": 0.5, "rt2x": 2.0, "haar": 0.05, "spec": 2.4}
    used = sorted({b for _, _, b, _ in arms if b != "ident" and b.rsplit("_s", 1)[0] in level})
    N = {b: level[b.rsplit("_s", 1)[0]] * prof for b in used}
    ast = {"per_basis": {P.wkey("C", sp["C"]): {b: {"N_T": N[b], "N_T_sum": float(N[b].sum())} for b in used},
                         P.wkey("D", sp["D"]): {b: {"N_T_sum": 1.01 * float(N[b].sum())} for b in used}},
           "construction": cons}
    sink.json("arm_stats.json", ast)
    info = {"smoke": False, "cell": FULL["id"], "session": 2, "mask": "native", "forced_masks": "off",
            "topup_families": "", "slot_offset": 0, "n_windows": 196, "n_construct": 64,
            "provenance": env["provenance"], "stopped": None}
    sink.json("metrics.json", Q.metrics_doc(rec, gates, info, jm))
    (out / "job.log").write_text(log.getvalue() + "[q14] G14-PRE pass on windows 0-97\nREALIZEDNULL_Q14_OK\n")
    return Y


def test_analysis_reads_a_q14_run_directory(tmp_path):
    """N5 on a session-2 cell directory written by N7's writers: discovery runs end to end with the q14 parameter set
    (no RT_all, 48 layers, D = 64-97), G14-PRE is read from g14pre.json and passes, and G-SCALE is c_D(band) over the
    largest Haar c_D on D. The unblind phase refuses an uncommitted thresholds sha256 before reading sealed/, and with
    it committed reaches the planted label."""
    d = tmp_path / "q14_full"
    Y = _q14_run_dir(d)
    sha, disc = A.discovery(d, quiet=True)
    assert disc["windows"] == {"C": "0-63", "D": "64-97", "G-ID1/3": "0-97"}
    g = disc["gates"]
    assert [k for k, v in g.items() if not v["pass"]] == []
    assert "G14-PRE" in g and not [k for k in g if k.startswith("G-PRE") or "rtall" in k]
    assert "rtall" not in disc["families"] and len(disc["N"]["bases"]["band"]["C_layers"]) == 48
    rec = json.loads((d / "g14pre.json").read_text())
    assert g["G14-PRE"]["value"]["band_dnll"]["dnll"] == pytest.approx(rec["band_dnll"]["dnll_w0_97"], rel=1e-9)
    assert g["G14-PRE"]["value"]["rows"]["topsigma32"]["gb3_ppl"] == rec["rows"]["topsigma32"]["gb3_ppl"]
    assert disc["preflight"]["G14-PRE"]["pass"] and disc["preflight"]["text_only"]
    scale = Y["band"][64:98].mean() / max(Y[f"haar_s{s}"][64:98].mean() for s in range(5))
    assert g["G-SCALE"]["value"] == pytest.approx(scale, rel=1e-9) and 5 <= scale <= 500
    assert g["G-EMATCH"]["pass"] and g["G-DRAWS:E"]["value"] == {"valid": 40, "planned": 40}
    journal = A._journal_repo(tmp_path)
    sealed = d / "sealed" / "window_nll_confirm.jsonl"
    kept = sealed.read_bytes()
    sealed.unlink()                                          # a read of sealed/ would fail differently
    with pytest.raises(RuntimeError, match="G-SEAL: .* is on no line a commit added"):
        A.unblind(d, sha, journal=journal, quiet=True)
    sealed.write_bytes(kept)
    A.commit_sha(journal, "q14_full", sha)
    v = A.unblind(d, sha, journal=journal, quiet=True)
    assert (v["label"], v["notes"]) == ("CONTROL-RULE", ["T-QUANTITY"])
    assert v["numbers"]["stats"]["E"]["theta"] == pytest.approx(5.5, rel=0.02)
    assert "G14-PRE" in v["figures"]["F5"]["claim"] and "RT_all" not in v["figures"]["F1"]["caption"]
