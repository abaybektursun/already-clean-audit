"""REALIZEDNULL PG1 (N3): every arm of SPEC §3.2 on PG1 over 196 wikitext windows, one model load.

Spec: studies/realizednull/SPEC.md — §2 (readout, instrument, means, mask identification), §3.2 (arms,
draws, seeds), §3.6 (arm_stats.json), §4 (manifest, splits, parity), §6.1 (job-side gates), §9.1-9.4
(runbook, job order), §10 (artifacts). The pure core is infra/pg/realizednull_core.py (N1); every N1 call
is marked `# N1: <function>`. GB-1's instrument (gutenberg_gb1.py) and FUNC's forward helpers
(invariant_core_func.py) are imported, not copied.

Sealing (G-NOLEAK): windows >= 98 are written only under out/sealed/. Nothing over them is printed,
logged or summarized anywhere else, and `Sink` asserts this on every unsealed write.

Fail loud: nothing in the job catches an error. A registered preflight stop (§9.4: G-PRE1, G-PRE2-4 under
the native mask) writes metrics.json with `stopped` and returns.

`metrics.json → job_metrics` is the job side of N1's `gates(job_metrics, discovery)` input, with N1's key
names (env, mask, mask_probe, pre0, ppl_R, ematch, construction, verify_C, gen_D, readout_validity). The
analysis adds `run` and `pre1_reference`, which the job cannot know. Every number in it is on windows < 98.

`--gram-only` (G-GEN calibration, not a registered run): the Grams over windows 0-97, then for C = windows 0-31,
0-63 and 0-97 every RT-family and EM_T construction, its verification on C and G-GEN on 0-97 and on the windows
of 0-97 outside C; writes gencal.json and compute.json only (gram_only).

    python3 scripts/realizednull_stamp.py
    ./infra/aws_pg/pg run infra.aws_pg.jobs.realizednull_pg1 --args " --smoke" \
        --type g6.xlarge,g6e.xlarge --weights models--google--paligemma-3b-pt-224 \
        --s3-prefix results/realizednull_pg1_smoke_a1 --budget-min 40 --detach
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import resource
import time
import urllib.request

import numpy as np
import torch

from infra.aws_pg.jobs.gutenberg_gb1 import (A1_MANIFEST, A1_NPZ, K, N_IN, N_LAYERS, PPL_NONE_BAND,
                                            WIKITEXT_REV, _fetch, build_windows)
from infra.aws_pg.jobs.gutenberg_gb2c_o1twin import GB1_BANKS
from infra.aws_pg.jobs.invariant_core_func import (_blank_image, _decoder_layers, _load_full_pg1,
                                                   _locate_text_block)
from infra.pg import copernicus_core as C
from infra.pg import gutenberg_core as G
from infra.pg.conserved_core_probes import energy_matched_subspace
from infra.pg.humboldt_core import hf_setup

BASE_REPO = "google/paligemma-3b-pt-224"
BASE_SHA = "35e4f46485b4d07967e7e9935bc3786aad50687c"        # §0.3
GB1_NLL = "s3://pg-lineage-769063704696/results/gutenberg_gb1_v2/out/gb1_window_nll.json"
REPO_FACTS = pathlib.Path(__file__).resolve().parents[2] / "pg" / "matrix_repo_facts.json"
WINDOW = 512                     # gutenberg_gb1.py:65
F0 = 98                          # first confirmation window (§4.2); windows >= F0 are sealed
N_IMAGE, N_PREFIX = 256, 257     # 256 <image> tokens + <bos>; the text block starts at 257 (§2.3)
VOCAB = 257216                   # §4.1
FORCED = ("causal", "bidir", "prefixlm")
REL = 2e-3                       # G-PRE0, G-PRE1, P1 tolerance

# GB-1's arm perplexities, research/gutenberg_gb1_verdict.json → sites.q.ppl (§6.1 G-PRE1..4 read them).
GB1_PPL_Q = {
    "none": 22.02970644601963, "core32": 25.651377364420576, "anti32": 1301.924405786236,
    "bottomsigma32": 22.021591340103345, "topsigma32": 1325.8643332245006,
    "topenergy32": 10859.74604175714,
    "energymatched32_s0": 34.684349555832725, "energymatched32_s1": 30.59601856648921,
    "energymatched32_s2": 37.9119918265937,
    "specmatched32_s0": 34.3704542304772, "specmatched32_s1": 28.823903724709215,
    "specmatched32_s2": 32.82475075634106,
    "haar32_s0": 22.21881215040791, "haar32_s1": 22.07144858628743, "haar32_s2": 22.235379406931226,
}
GB1_ARM = {"none": "none", "band": "core32", "anti32": "anti32", "bottomsigma32": "bottomsigma32",
           "topsigma32": "topsigma32", "topenergy32": "topenergy32",
           **{f"em_s{s}": f"energymatched32_s{s}" for s in range(3)},
           **{f"spec_s{s}": f"specmatched32_s{s}" for s in range(3)},
           **{f"haar_s{s}": f"haar32_s{s}" for s in range(3)}}
PRE_GATING = {"G-PRE2": ("band",), "G-PRE3": ("em_s0",),
              "G-PRE4": ("em_s1", "em_s2", "spec_s0", "spec_s1", "spec_s2", "topenergy32",
                         "topsigma32", "anti32")}
PRE_RECORDED = ("haar_s0", "haar_s1", "haar_s2", "bottomsigma32")      # same rule, not gating
PRE0_ARMS = ("none", "band", "em_s0")                                   # G-PRE0, smoke only

# RT-family and EM_T constructions (§2.2, §3.2): base seed, the band quantity matched, the dose.
FAMILIES = {
    "rt": {"seed": 200_000, "q": "N_T", "dose": 1.0},
    "rtall": {"seed": 300_000, "q": "N_all", "dose": 1.0},
    "rthalf": {"seed": 400_000, "q": "N_T", "dose": 0.5},
    "rt2x": {"seed": 500_000, "q": "N_T", "dose": 2.0},
    "emt": {"seed": 600_000, "q": "E_T", "dose": 1.0},
}
CONSTRUCT_KW = {"k": K, "tol_sum": 0.005, "tol_layer": 0.05, "heavy_frac": 0.01, "n_grid": 64,
                "n_probe": 16, "max_resample": 20000, "max_closing": 50}
TOPUP_FAMILIES = ("em", "em_mean", "rt", "emt")
TOPUP_DRAWS = 20                 # §6.5: seed slots 40-59
SMOKE = {"n_windows": 6, "n_construct": 4, "draws_em": 4, "draws_spec": 4, "draws_haar": 4,
         "draws_rt": 2, "draws_emt": 2, "draws_rtall": 1, "draws_dose": 1, "draws_zt": 1,
         "draws_mean_rt": 1, "draws_mean_spec": 1, "draws_mean_haar": 1}       # §9.3 step 6
PROBE_EDIT, PROBE_PERTURB, PROBE_PRIOR, PROBE_SEED = 64, (200, 256), (0, 200), 7    # §2.4


def _n1():
    """N1, imported on first use so the writers and pure helpers here load and test without it."""
    from infra.pg import realizednull_core
    return realizednull_core


# ===========================================================================
# Flags, windows, roster
# ===========================================================================

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mask", choices=("native",) + FORCED, default="native")
    ap.add_argument("--forced-masks", choices=("on", "off"), default="on")
    ap.add_argument("--n-windows", type=int, default=196)
    ap.add_argument("--n-construct", type=int, default=32)
    ap.add_argument("--draws-em", type=int, default=40)
    ap.add_argument("--draws-rt", type=int, default=40)
    ap.add_argument("--draws-emt", type=int, default=40)
    ap.add_argument("--draws-spec", type=int, default=5)
    ap.add_argument("--draws-haar", type=int, default=5)
    ap.add_argument("--draws-rtall", type=int, default=5)
    ap.add_argument("--draws-dose", type=int, default=3)
    ap.add_argument("--draws-zt", type=int, default=5)
    ap.add_argument("--draws-mean-rt", type=int, default=5)
    ap.add_argument("--draws-mean-spec", type=int, default=3)
    ap.add_argument("--draws-mean-haar", type=int, default=3)
    ap.add_argument("--topup-families", default="")
    ap.add_argument("--slot-offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--gram-only", action="store_true")
    a = ap.parse_args(argv)
    if a.gram_only:
        assert not (a.smoke or a.topup_families), "--gram-only is its own run"
        assert a.mask == "native", "--gram-only runs no forced-mask self-test"
    if a.smoke:
        assert not a.topup_families, "--smoke and --topup-families are separate runs"
        for k, v in SMOKE.items():
            setattr(a, k, v)
    assert a.mask == "native" or a.forced_masks == "on", "a forced --mask needs --forced-masks on"
    assert a.smoke or a.n_windows > F0, "a full run scores windows 0-97 and at least one window >= 98"
    fams = [f for f in a.topup_families.split(",") if f]
    assert set(fams) <= set(TOPUP_FAMILIES), fams
    if fams:
        assert a.slot_offset >= 3 and a.slot_offset + TOPUP_DRAWS <= 60, a.slot_offset   # §3.2 s <= 59
    else:
        assert a.slot_offset == 0, "--slot-offset is a top-up flag"
        assert min(a.draws_em, a.draws_spec, a.draws_haar) >= 3, "s0-s2 are the banked bases"
        assert a.draws_zt <= min(a.draws_em, a.draws_rt) and a.draws_mean_rt <= a.draws_rt
        assert a.draws_mean_spec <= a.draws_spec and a.draws_mean_haar <= a.draws_haar
    return a


def splits(n_windows, n_construct):
    """C, R, D, F of §4.2 for this run. The smoke (6 windows) has no D or F proper: its D is windows
    n_construct..5 and its F is empty. The common sample window is n_construct (32) in the open split and
    F0 (98) in the sealed one (§10.1)."""
    top = min(n_windows, F0)
    assert 0 < n_construct < top, (n_construct, top)
    return {"C": list(range(n_construct)), "R": list(range(top)), "D": list(range(n_construct, top)),
            "F": list(range(F0, n_windows)), "common_open": n_construct, "common_sealed": F0}


def draw_slots(a):
    """Seed slots per family. A top-up run (§6.5) draws TOPUP_DRAWS new slots for the listed families
    only; a new EM basis runs under zero and mean ablation when either EM family is listed."""
    fams = {f for f in a.topup_families.split(",") if f}
    if fams:
        top = list(range(a.slot_offset, a.slot_offset + TOPUP_DRAWS))
        em = top if fams & {"em", "em_mean"} else []
        none = []
        return {"em": em, "em_mean": em, "spec": none, "haar": none,
                "rt": top if "rt" in fams else none, "rtall": none, "rthalf": none, "rt2x": none,
                "emt": top if "emt" in fams else none, "em_zt": none, "rt_zt": none,
                "rt_mean": none, "spec_mean": none, "haar_mean": none}
    r = lambda n: list(range(n))
    return {"em": r(a.draws_em), "em_mean": r(a.draws_em), "spec": r(a.draws_spec),
            "haar": r(a.draws_haar), "rt": r(a.draws_rt), "rtall": r(a.draws_rtall),
            "rthalf": r(a.draws_dose), "rt2x": r(a.draws_dose), "emt": r(a.draws_emt),
            "em_zt": r(a.draws_zt), "rt_zt": r(a.draws_zt), "rt_mean": r(a.draws_mean_rt),
            "spec_mean": r(a.draws_mean_spec), "haar_mean": r(a.draws_mean_haar)}


def roster(a):
    """Ablation arms in §9.4 step-1 order, as (arm, op, basis, group): `band` (G-PRE2), the banked-basis
    arms (G-PRE3/4), then every remaining arm in §3.2 table order. `none` and K1 run before these."""
    sl = draw_slots(a)
    Z, M, ZT = "zero", "mean", "zero_text"
    full = not a.topup_families
    out = [("band", Z, "band", "pre")]
    if full:
        out += [(f"em_s{s}", Z, f"em_s{s}", "pre") for s in range(3)]
        out += [(f"spec_s{s}", Z, f"spec_s{s}", "pre") for s in range(3)]
        out += [(n, Z, n, "pre") for n in ("topenergy32", "topsigma32", "bottomsigma32", "anti32")]
        out += [(f"haar_s{s}", Z, f"haar_s{s}", "pre") for s in range(3)]
    rest = [("ident_zero", Z, "ident"), ("band_dup", Z, "band")]
    for fam in ("haar", "em", "spec"):
        rest += [(f"{fam}_s{s}", Z, f"{fam}_s{s}") for s in sl[fam] if s >= 3]
    for fam in FAMILIES:
        rest += [(f"{fam}_s{s}", Z, f"{fam}_s{s}") for s in sl[fam]]
    rest += [("band_zt", ZT, "band")] if full else []
    rest += [(f"em_zt_s{s}", ZT, f"em_s{s}") for s in sl["em_zt"]]
    rest += [(f"rt_zt_s{s}", ZT, f"rt_s{s}") for s in sl["rt_zt"]]
    rest += [("ident_mean", M, "ident"), ("band_mean", M, "band")]
    for fam, src in (("em_mean", "em"), ("rt_mean", "rt"), ("spec_mean", "spec"),
                     ("haar_mean", "haar")):
        rest += [(f"{fam}_s{s}", M, f"{src}_s{s}") for s in sl[fam]]
    rest += [("topenergy32_mean", M, "topenergy32"), ("topsigma32_mean", M, "topsigma32")] if full else []
    return out + [(n, op, b, "rest") for n, op, b in rest]


def budget_s_from_cmdline(raw):
    """The budget run_job.sh hands `timeout --signal=KILL <budget>s python -m <job>`, read from the job's
    parent's /proc cmdline (the job is not told it any other way)."""
    argv = raw.decode().split("\0")
    assert os.path.basename(argv[0]) == "timeout", argv[:4]
    secs = [x for x in argv[1:] if re.fullmatch(r"\d+s", x)]
    assert secs, argv
    return int(secs[0][:-1])


def parent_cmdline():
    """The job's parent's command line (run_job.sh's `timeout`), from /proc."""
    return pathlib.Path(f"/proc/{os.getppid()}/cmdline").read_bytes()


def instance_type():
    """EC2 instance type from IMDSv2 (G-ENV)."""
    tok = urllib.request.urlopen(urllib.request.Request(
        "http://169.254.169.254/latest/api/token", method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}), timeout=5).read().decode()
    return urllib.request.urlopen(urllib.request.Request(
        "http://169.254.169.254/latest/meta-data/instance-type",
        headers={"X-aws-ec2-metadata-token": tok}), timeout=5).read().decode()


# ===========================================================================
# Inputs, masks, scoring, hooks (pure torch; tested on tiny tensors)
# ===========================================================================

def block_at(hay, needle):
    """Rightmost start of `needle` in `hay` (1-D LongTensors), or None. `_locate_text_block`'s search,
    used only to decide GB-1's trailing-token trim without catching its AssertionError."""
    n = int(needle.shape[0])
    for i in range(hay.shape[0] - n, -1, -1):
        if torch.equal(hay[i:i + n], needle):
            return i
    return None


def make_units(texts, tok, processor, image):
    """GB-1's window units and trailing-token trim (gutenberg_gb1.py:110-127): a window whose exact token
    block is absent from the processor output loses its final token once (AM-0.5.8), and
    `_locate_text_block` raises if it is still absent. Returns (units, indices of trimmed windows)."""
    units, trimmed = [], []
    for w, text in enumerate(texts):
        win_ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        inp = processor(text=[text], images=[image], return_tensors="pt")
        if block_at(inp["input_ids"][0].cpu(), win_ids) is None:
            text = tok.decode(win_ids[:-1].tolist())
            trimmed.append(w)
            win_ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
            inp = processor(text=[text], images=[image], return_tensors="pt")
        pos = _locate_text_block(inp["input_ids"][0], win_ids)
        units.append({"text": text, "n_text": int(win_ids.shape[0]), "pos": pos, "win_ids": win_ids,
                      "input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"],
                      "pixel_values": inp["pixel_values"]})
    return units, trimmed


def forced_mask(mode, S, pos, dtype=torch.bfloat16):
    """The explicit 4-D float mask of §2.4, shape (1, 1, S, S): 0 where query i may attend key j,
    finfo(dtype).min where it may not. causal: j <= i. bidir: every pair. prefixlm: full attention among
    positions 0..pos-1, causal over the text."""
    i = torch.arange(S)[:, None]
    j = torch.arange(S)[None, :]
    allowed = {"causal": j <= i,
               "bidir": torch.ones(S, S, dtype=torch.bool),
               "prefixlm": (j <= i) | ((i < pos) & (j < pos))}[mode]
    m = torch.where(allowed, torch.zeros((), dtype=dtype), torch.tensor(torch.finfo(dtype).min, dtype=dtype))
    return m[None, None]


def score_logits(logits, pos, n_text, win_ids):
    """GB-1's readout (gutenberg_gb1.py:185-194) plus the per-position arrays the samples need: the logit at
    position pos+i scores target win_ids[i+1], i = 0..n_text-2. logits: (S, V); win_ids on logits' device."""
    text_logits = logits[pos:pos + n_text - 1, :].float()
    targets = win_ids[1:]
    assert text_logits.shape[0] == targets.shape[0] == n_text - 1, (text_logits.shape, targets.shape)
    nll = float(torch.nn.functional.cross_entropy(text_logits, targets, reduction="sum").item())
    lse = torch.logsumexp(text_logits, dim=-1)
    lp = text_logits.gather(1, targets[:, None])[:, 0] - lse
    top_val, top_id = text_logits.max(dim=-1)
    return {"nll": nll, "ntok": int(targets.numel()), "lp": lp.cpu().numpy(),
            "top_id": top_id.cpu().numpy(), "top_lp": (top_val - lse).cpu().numpy()}


def ablation_hook(U, op, mu, state, W=None, acc=None, L=None):
    """forward_pre_hook on q_proj (§2.1). h32 = fp32 input; returns, cast back to the input dtype,
    zero: h32 - (h32 @ U) @ U.T; zero_text: the same at text-block positions (pos..end) only;
    mean: h32 - ((h32 - mu_c(t)) @ U) @ U.T with mu rows 0..pos-1 per prefix position and row pos the text
    class. U: (n, k) fp32, k = 0 gives the identity. When state["acc"] is set, accumulates the realized
    perturbation sum_t ||W (h32 - out)||^2 and sum_t ||W h32||^2 over the scored text positions (§3.6)."""
    def hook(_m, args):
        h = args[0]
        h32 = h.to(torch.float32)
        p = state["pos"]
        if op == "zero":
            out = h32 - (h32 @ U) @ U.T
        elif op == "zero_text":
            out = h32.clone()
            out[:, p:] = h32[:, p:] - (h32[:, p:] @ U) @ U.T
        elif op == "mean":
            mu_full = torch.cat([mu[:p], mu[p:p + 1].expand(h32.shape[1] - p, -1)], dim=0)
            out = h32 - ((h32 - mu_full) @ U) @ U.T
        else:
            raise ValueError(op)
        hb = out.to(h.dtype)
        if state["acc"]:
            t = slice(p, p + state["n_text"] - 1)
            d = (h32 - hb.to(torch.float32))[0, t]
            acc["num"][L] = acc["num"][L] + (d @ W.T).pow(2).sum().double()
            acc["den"][L] = acc["den"][L] + (h32[0, t] @ W.T).pow(2).sum().double()
        return (hb,) + tuple(args[1:])
    return hook


def probe_add_hook(u, lo=PROBE_PERTURB[0], hi=PROBE_PERTURB[1]):
    """Image-block probe perturbation (§2.4) for layer 0's k_proj / v_proj input: adds g_i = ||h_i|| u at
    positions lo..hi-1, OUT OF PLACE (q, k and v share one input tensor; an in-place add would move the
    query too)."""
    def hook(_m, args):
        h = args[0]
        g = torch.zeros_like(h)
        g[:, lo:hi] = (h[:, lo:hi].to(torch.float32).norm(dim=-1, keepdim=True) * u).to(h.dtype)
        return (h + g,) + tuple(args[1:])
    return hook


def probe_vector(n=N_IN):
    v = np.random.default_rng(PROBE_SEED).standard_normal(n)
    return v / np.linalg.norm(v)


# ===========================================================================
# Writers (routing by window range) and per-arm bookkeeping
# ===========================================================================

WKEY = re.compile(r"_w(\d+)_(\d+)")


def wkey(name, windows):
    """A statistic's key carries its window range (G-NOLEAK)."""
    w = list(windows)
    assert w and w == list(range(w[0], w[-1] + 1)), "window ranges are contiguous"
    return f"{name}_w{w[0]}_{w[-1]}"


def assert_open(obj, where):
    """An unsealed record names no window range reaching F0 and no window index >= F0."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            for m in WKEY.finditer(str(k)):
                assert int(m.group(2)) < F0, f"G-NOLEAK: {where} key {k}"
            if k in ("window", "windows", "index"):
                ws = v if isinstance(v, (list, tuple)) else [v]
                assert all(w < F0 for w in ws if isinstance(w, (int, np.integer))), f"G-NOLEAK: {where} {k}"
            assert_open(v, where)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            assert_open(v, where)


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(f"not JSON-serializable: {type(x)}")


class Sink:
    """Every file the job writes. Windows >= F0 go only under sealed/ (G-NOLEAK) and every unsealed write
    passes assert_open. JSON files are replaced atomically so the 30 s S3 sync never ships half a file."""

    def __init__(self, out, sample_split, now=C.utc_now):
        self.out = pathlib.Path(out)
        self.sample_split = sample_split          # "smoke" | "discovery" | None (--gram-only: no samples)
        self.now = now
        self.out.mkdir(parents=True, exist_ok=True)
        for d in ("sealed/samples", f"samples/{sample_split}") if sample_split else ():
            (self.out / d).mkdir(parents=True, exist_ok=True)

    def json(self, rel, obj):
        if not rel.startswith("sealed/"):
            assert_open(obj, rel)
        p = self.out / rel
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=1, default=_jsonable))
        os.replace(tmp, p)

    def _append(self, rel, row):
        with open(self.out / rel, "a") as f:
            f.write(json.dumps(row, default=_jsonable) + "\n")

    def event(self, stage, **kw):
        row = {"stage": stage, "utc": self.now(), **kw}
        assert_open(row, "events.jsonl")
        self._append("events.jsonl", row)

    def window_nll(self, arm, op, windows, nll, ntok):
        """One line per arm per file (§10.1): windows < F0 → window_nll_rd.jsonl, >= F0 → sealed/."""
        for rel, keep in (("window_nll_rd.jsonl", lambda w: w < F0),
                          ("sealed/window_nll_confirm.jsonl", lambda w: w >= F0)):
            idx = [i for i, w in enumerate(windows) if keep(w)]
            if idx:
                row = {"arm": arm, "op": op, "windows": [windows[i] for i in idx],
                       "nll_sum": [nll[i] for i in idx], "ntok": [ntok[i] for i in idx]}
                if not rel.startswith("sealed/"):
                    assert_open(row, rel)
                self._append(rel, row)

    def samples(self, arm, open_payload, sealed_payload):
        paths = {}
        if open_payload is not None:
            paths["open"] = f"samples/{self.sample_split}/{arm}.json"
            self.json(paths["open"], open_payload)
        if sealed_payload is not None:
            paths["sealed"] = f"sealed/samples/{arm}.json"
            self.json(paths["sealed"], sealed_payload)
        return paths


def summarize(scores, none, sp):
    """Absolute R and D statistics of one arm against `none` (§2.1: Y = ΔNLL_sum / ntok); every key
    carries its window range. Windows >= F0 are never read."""
    R, D = sp["R"], sp["D"]
    for w in R:
        assert scores[w]["ntok"] == none[w]["ntok"], (w, scores[w]["ntok"], none[w]["ntok"])
    n = sum(scores[w]["nll"] for w in R)
    t = sum(scores[w]["ntok"] for w in R)
    Y = {w: (scores[w]["nll"] - none[w]["nll"]) / scores[w]["ntok"] for w in R}
    return {wkey("ppl", R): math.exp(n / t),
            wkey("dnll", R): (n - sum(none[w]["nll"] for w in R)) / t,
            wkey("max_abs_Y", R): max(abs(Y[w]) for w in R),
            wkey("c", D): float(np.mean([Y[w] for w in D]))}


def sample_windows(Y, windows, common):
    """The max-Y, min-Y and common windows of one split (§10.1)."""
    ws = list(windows)
    assert common in ws, (common, ws[:3])
    return {"max": max(ws, key=lambda w: Y[w]), "min": min(ws, key=lambda w: Y[w]), "common": common}


def pick_examples(order, targets, top_ids):
    """First scored position whose top-1 token is the target ("right") and first where it is not
    ("wrong"), searching windows in `order`. Returns {"right": (w, i) | None, "wrong": (w, i) | None}."""
    found = {"right": None, "wrong": None}
    for w in order:
        hit = np.asarray(top_ids[w]) == np.asarray(targets[w])
        for key, mask in (("right", hit), ("wrong", ~hit)):
            if found[key] is None and mask.any():
                found[key] = (w, int(np.argmax(mask)))
    return found


def sample_examples(order, targets, none, arm, pos, to_str):
    """One right and one wrong scored position under the arm and under `none`, each with the target,
    log p(target) and the top-1 token and its probability under both models (§10.1). `none` must have
    both; an arm that never (or always) predicts the target has None in that slot."""
    def example(w, i):
        t = int(targets[w][i])
        row = {"window": w, "scored_index": i, "position": pos + i, "target_id": t, "target": to_str(t)}
        for name, sc in (("none", none), ("arm", arm)):
            top = int(sc[w]["top_id"][i])
            row.update({f"logp_target_{name}": float(sc[w]["lp"][i]), f"top1_{name}": to_str(top),
                        f"p_top1_{name}": float(np.exp(sc[w]["top_lp"][i])), f"right_{name}": top == t})
        return row
    out = {}
    for name, sc in (("arm", arm), ("none", none)):
        hits = pick_examples(order, targets, {w: sc[w]["top_id"] for w in order})
        out[name] = {k: (None if v is None else example(*v)) for k, v in hits.items()}
    assert out["none"]["right"] is not None and out["none"]["wrong"] is not None, \
        "the unablated model has no right or no wrong top-1 in the sampled windows"
    return out


class Recorder:
    """Per-arm bookkeeping: per-window NLL rows (routed by Sink), R/D summaries for metrics.json, sample
    window selection, coverage and progress. Windows >= F0 are read only to route them to sealed/ and to
    pick the sealed samples."""

    def __init__(self, sink, sp, all_arms):
        self.sink, self.sp = sink, sp
        self.summary, self.nll_open, self.done_arms, self.skipped = {}, {}, [], {}
        self.none = None
        sealed = "PENDING" if sp["F"] else "no confirmation windows in this run"
        self.coverage = {arm: {"open": "PENDING", "sealed": sealed} for arm in all_arms}

    def stage(self, name):
        self.sink.event(name)
        self.sink.json("progress.json", {"stage": name, "arms_done": self.done_arms,
                                         "n_done": len(self.done_arms), "n_arms": len(self.coverage),
                                         "skipped": self.skipped, "utc": self.sink.now()})

    def nll(self, arm, op, scores):
        windows = sorted(scores)
        self.sink.window_nll(arm, op, windows, [scores[w]["nll"] for w in windows],
                             [scores[w]["ntok"] for w in windows])
        if arm == "none":
            self.none = scores
        s = summarize(scores, self.none, self.sp)
        self.summary[arm] = s
        self.nll_open[arm] = {w: scores[w]["nll"] for w in windows if w < F0}
        print(f"[realizednull] {arm} op={op} "
              + " ".join(f"{k}={v:.6g}" for k, v in sorted(s.items())), flush=True)
        return s

    def select(self, scores):
        """Sample windows for the open split (D) and the sealed split (F), with Y per window."""
        Y = {w: (scores[w]["nll"] - self.none[w]["nll"]) / scores[w]["ntok"] for w in scores}
        D = [w for w in self.sp["D"] if w in Y]
        F = [w for w in self.sp["F"] if w in Y]
        return {"open": sample_windows(Y, D, self.sp["common_open"]) if D else None,
                "sealed": sample_windows(Y, F, self.sp["common_sealed"]) if F else None}, Y

    def write_samples(self, arm, open_payload, sealed_payload):
        self.coverage[arm].update(self.sink.samples(arm, open_payload, sealed_payload))

    def skip(self, arm, reason):
        self.skipped[arm] = reason
        self.done(arm)

    def done(self, arm):
        self.done_arms.append(arm)
        self.sink.json("samples/coverage.json", {"arms": self.coverage, "skipped": self.skipped})
        self.sink.json("progress.json", {"stage": "arms", "arm": arm, "arms_done": self.done_arms,
                                         "n_done": len(self.done_arms), "n_arms": len(self.coverage),
                                         "skipped": self.skipped, "utc": self.sink.now()})


def present(d):
    """d when every number in it exists, else None. N1's G-READOUT reads None as a missing number (§6.1);
    an empty family counts as missing."""
    def full(x):
        if isinstance(x, dict):
            return bool(x) and all(full(v) for v in x.values())
        return x is not None
    return d if full(d) else None


def readout_validity(summary, gates, sp):
    """The three numbers G-READOUT reads per confirmatory statistic (§6.1): fit on both sides, the
    positive control, the identity pair, on R/D windows. A part with any number missing (an arm that did
    not run, a gate with no draws) is None, so N1's G-READOUT fires on it."""
    R, D = sp["R"], sp["D"]
    c = lambda arm: summary.get(arm, {}).get(wkey("c", D))
    y = lambda arm: summary.get(arm, {}).get(wkey("max_abs_Y", R))
    fam = lambda pat: {x: c(x) for x in summary if re.fullmatch(pat, x)}
    identity = {wkey("max_abs_Y_ident_zero", R): y("ident_zero"),
                wkey("max_abs_Y_ident_mean", R): y("ident_mean"),
                wkey("c_band", D): c("band"), wkey("c_band_dup", D): c("band_dup")}
    pos_zero = {wkey("c_topenergy32", D): c("topenergy32"), wkey("c_haar", D): fam(r"haar_s\d+")}
    pos_mean = {wkey("c_topsigma32_mean", D): c("topsigma32_mean"),
                wkey("c_haar_mean", D): fam(r"haar_mean_s\d+")}

    def counts(gate, family=None):
        g = gates.get(gate, {})
        g = g.get(family, {}) if family else g
        return {"n_pass": g.get("n_pass"), "n": g.get("n") or None}        # n = 0: no draw to fit

    def part(fit, pos):
        return {"fit": present(fit), "positive_control": present(pos), "identity": present(identity)}
    fit_em = {"G-EMATCH": counts("G-EMATCH")}
    return {"E": part(fit_em, pos_zero), "M": part(fit_em, pos_mean),
            "R": part({g: counts(g, "rt") for g in ("G-MATCH", "G-GEN")}, pos_zero),
            "T": part({g: counts(g, "emt") for g in ("G-MATCH", "G-GEN")}, pos_zero)}


def metrics_doc(rec, gates, info, jm):
    """metrics.json: the job-side gate records, every arm's R/D summary, readout_validity, and job_metrics
    (jm plus ppl_R and readout_validity), the job side of N1's gates() input."""
    rv = readout_validity(rec.summary, gates, rec.sp)
    ppl_R = {arm: s[wkey("ppl", rec.sp["R"])] for arm, s in rec.summary.items()}
    return {"program": "realizednull", "job": "realizednull_pg1", **info, "gates": gates,
            "arms": rec.summary, "arms_done": rec.done_arms, "skipped": rec.skipped,
            "readout_validity": rv, "job_metrics": {**jm, "ppl_R": ppl_R, "readout_validity": rv}}


# ===========================================================================
# Job-side gates (§6.1), pure
# ===========================================================================

def env_gate(env):
    """G-ENV on env.json with N1's key names: every N1 ENV_KEYS entry, and _attn_implementation == "eager"."""
    missing = [k for k in _n1().ENV_KEYS if not env.get(k)]
    return {"missing": missing, "_attn_implementation": env.get("_attn_implementation"),
            "pass": not missing and env.get("_attn_implementation") == "eager"}


def pre1_gate(ppl_none, R):
    """G-PRE1: ppl(none) on windows 0-97 inside GB-1's frozen band and within rel 2e-3 of 22.0297."""
    ok = (PPL_NONE_BAND[0] <= ppl_none <= PPL_NONE_BAND[1]
          and abs(ppl_none - GB1_PPL_Q["none"]) <= REL * GB1_PPL_Q["none"])
    return {wkey("ppl_none", R): ppl_none, "frozen_band": list(PPL_NONE_BAND),
            "gb1_ppl_none": GB1_PPL_Q["none"], "rel_tol": REL, "pass": ok}


def pre_gate(summary, arms, R):
    """G-PRE2..4: |ΔNLL_run − ΔNLL_GB1| <= max(0.002, 0.02·ΔNLL_GB1) on windows 0-97, ΔNLL_GB1 =
    ln(GB-1 ppl(arm) / ppl(none)). Evaluates the listed arms that ran."""
    out = {}
    for arm in arms:
        if arm not in summary:
            continue
        ref = math.log(GB1_PPL_Q[GB1_ARM[arm]] / GB1_PPL_Q["none"])
        tol = max(0.002, 0.02 * ref)
        x = summary[arm][wkey("dnll", R)]
        out[arm] = {wkey("dnll", R): x, "dnll_gb1": ref, "band": [ref - tol, ref + tol],
                    "pass": abs(x - ref) <= tol}
    return {"arms": out, "pass": all(v["pass"] for v in out.values())}


def pre0_rows(nll_open, gb1_nll, windows):
    """G-PRE0 (smoke): per-window NLL of none, band, em_s0 within rel 2e-3 of GB-1's, windows 0-5."""
    q = gb1_nll["per_window_by_site"]["q"]
    out = {}
    for arm in PRE0_ARMS:
        ref = q[GB1_ARM[arm]]["nll"]
        rel = [abs(nll_open[arm][w] - ref[w]) / abs(ref[w]) for w in windows]
        out[arm] = {wkey("rel_err", windows): rel, "pass": max(rel) <= REL}
    return {"arms": out, "pass": all(v["pass"] for v in out.values())}


def pre0_input(nll_open, gb1_nll, windows):
    """job_metrics → pre0 (smoke) as N1's gates() reads it: per-window NLL of none, band, em_s0 under this
    run and under GB-1 (its arm names)."""
    q = gb1_nll["per_window_by_site"]["q"]
    return {"run": {arm: [nll_open[arm][w] for w in windows] for arm in PRE0_ARMS},
            "gb1": {GB1_ARM[arm]: [q[GB1_ARM[arm]]["nll"][w] for w in windows] for arm in PRE0_ARMS}}


def p1_rows(nll_open, ppl_by_mode, gb1_nll, R, smoke):
    """P1 / P1-smoke inputs (§6.6): which forced mode reproduces GB-1's `none`. Smoke: per-window NLL on
    windows 0-5 within rel 2e-3; full: ppl on 0-97 within rel 2e-3 of 22.0297."""
    out = {}
    for m, ppl in ppl_by_mode.items():
        if smoke:
            ref = gb1_nll["per_window_by_site"]["q"]["none"]["nll"]
            ok = all(abs(nll_open[f"none_{m}"][w] - ref[w]) <= REL * abs(ref[w]) for w in R)
        else:
            ok = abs(ppl - GB1_PPL_Q["none"]) <= REL * GB1_PPL_Q["none"]
        out[m] = {wkey("ppl", R): ppl, "matches_gb1_none": ok}
    return out


def ematch_gate(e_target, e_real):
    """G-EMATCH: realized energy within 10% of target at every layer whose target is not below k/d."""
    floor = K / N_IN
    bad = [L for L, (t, e) in enumerate(zip(e_target, e_real)) if t >= floor and abs(e - t) > 0.1 * t]
    return {"failing_layers": bad, "pass": not bad}


def match_gate(N, targets, windows):
    """G-MATCH as N1 reads a verify_C row: hook-verified summed quantity N within 1% of the summed target
    (dose × the hook-verified band quantity, verify_rows), heavy layers (target >= 1% of the summed target)
    within 5.5%."""
    ref = np.asarray(targets)
    per = np.asarray(N) / ref
    heavy = ref >= 0.01 * np.sum(ref)
    ratio = float(np.sum(N) / np.sum(ref))
    bad = [int(L) for L in np.flatnonzero(heavy & (np.abs(per - 1) > 0.055))]
    return {wkey("ratio_sum", windows): ratio, wkey("ratio_per_layer", windows): per,
            "heavy_layers": np.flatnonzero(heavy), "failing_heavy_layers": bad,
            "pass": abs(ratio - 1) <= 0.01 and not bad}


def family_gates(cons, names):
    """G-REACH, G-PIVOT, G-EXHAUST for one family's draw indices (§6.1), as N1's gates() computes them:
    the pivot share counts every record with pivot = True; unreachable_layers is the first unreachable
    draw's list (layer indices, and "sum" when the summed target is out of range)."""
    recs = [cons[n] for n in names]
    n = len(recs)
    unreach = [i for i, r in enumerate(recs) if not r["reachable"]]
    layers = recs[unreach[0]]["unreachable_layers"] if unreach else []
    pivot = sum(bool(r["pivot"]) for r in recs) / n
    exhausted = sum(bool(r["exhausted"]) for r in recs) / n
    return {"G-REACH": {"pass": not unreach, "unreachable_draws": unreach, "unreachable_layers": layers},
            "G-PIVOT": {"share": pivot, "pass": pivot <= 0.25},
            "G-EXHAUST": {"share": exhausted, "pass": exhausted <= 0.25}}


# ===========================================================================
# N1 inputs: each dict N3 hands to realizednull_core is built here, in one place
# ===========================================================================

PROBE_KEYS = ("d_before", "d_after", "r_prior", "r_self")       # mask_probe_verdict's arguments


def probe_numbers(rec_probe, forced_masks):
    """job_metrics → mask_probe as N1's gates() reads it: the four §2.4 probe numbers per mode."""
    pick = lambda r: {k: r[k] for k in PROBE_KEYS}
    return {"native": pick(rec_probe["native"]), "forced_masks": forced_masks,
            "forced": {m: pick(r) for m, r in rec_probe.get("forced", {}).items()}}


def class_inputs(g, n_construct, n_text_positions):
    """class_means' sums and counts per layer (§2.3) from gram_pass: one class per prefix position (row i
    of g["prefix"][L], one position per window, so n_construct positions) and one text-block class
    (positions pos..end of every window, n_text_positions in all)."""
    n_prefix = g["prefix"][0].shape[0]
    sums = [{**{i: P[i].astype(np.float64) for i in range(n_prefix)}, "text": t.astype(np.float64)}
            for P, t in zip(g["prefix"], g["text"])]
    counts = {**{i: n_construct for i in range(n_prefix)}, "text": n_text_positions}
    return sums, counts


def class_rows(means):
    """class_means' result as an (n_prefix + 1, n) array: prefix positions 0..pos-1, the text class last
    (row pos), the layout ablation_hook's "mean" and arm_stats' means["mu"] read."""
    n_prefix = len(means) - 1
    return np.stack([means[i] for i in range(n_prefix)] + [means["text"]])


def eigen_sets(g, Ws):
    """eigencoords per layer on C (§2.2), M = W_qᵀW_q in fp64 from each fp64 W_q that Ws yields: sigma_S =
    Σ_T (for N_T and E_T) and, in its own call with its own V, Σ_all (for N_all)."""
    R = _n1()
    ecT, ecA = [], []
    for L, W in enumerate(Ws):
        M = W.T @ W
        SA = g["all"][L].astype(np.float64)
        ecT.append(R.eigencoords(SA, M, g["T"][L].astype(np.float64)))  # N1: eigencoords
        ecA.append(R.eigencoords(SA, M, SA))  # N1: eigencoords
    return ecT, ecA


def quantity_inputs(ecT, ecA):
    """(Mp, Sp, V) per matched quantity: N on T, N on all positions, E on T (W_q replaced by I)."""
    eye = np.eye(ecT[0]["V"].shape[0])
    get = lambda ec, key: [e[key] for e in ec]
    return {"N_T": (get(ecT, "Mp"), get(ecT, "Sp"), get(ecT, "V")),
            "N_all": (get(ecA, "Mp"), get(ecA, "Sp"), get(ecA, "V")),
            "E_T": ([eye] * len(ecT), get(ecT, "Sp"), get(ecT, "V"))}


def band_targets(fam_in, band):
    """The band's per-layer quantity in eigen-coordinates, per matched quantity (§2.2 targets)."""
    R = _n1()
    out = {}
    for q, (Mp, Sp, V) in fam_in.items():
        out[q] = np.array([R.realized_quantity(V[L].T @ band[L], Mp[L], Sp[L])  # N1: realized_quantity
                           for L in range(len(band))])
    return out


def quantity_grids(fam_in, qs, kw):
    """reach_grid per layer for each matched quantity in qs, seed 190_000 + L (§2.2, §3.2); every family on
    that quantity, and every attempt, reuses them (construct_draw's grids=)."""
    R = _n1()
    out = {}
    for q in qs:
        Mp, Sp, _ = fam_in[q]
        out[q] = [R.reach_grid(Mp[L], Sp[L], k=kw["k"], n_grid=kw["n_grid"],  # N1: reach_grid
                               n_probe=kw["n_probe"], seed=R.GRID_SEED_BASE + L) for L in range(len(Mp))]
    return out


def construct_family(fam, slots, fam_in, tgt, grids, kw):
    """Every draw index s of one RT-family or EM_T family (§2.2, §3.2): construct_draw at base seed
    FAMILIES[fam]["seed"] + 1000·s on the family's quantity × dose, with that quantity's grids. Returns
    {name: record} (construct_draw's result without Ue, plus family_name, slot, base_seed, wall_seconds),
    {name: per-layer model-space basis V @ Ue} for the matched draws, and the construction seconds."""
    R = _n1()
    f = FAMILIES[fam]
    Mp, Sp, V = fam_in[f["q"]]
    recs, bases, t_con = {}, {}, 0.0
    for s in slots:
        name, base_seed = f"{fam}_s{s}", f["seed"] + 1000 * s
        t0 = time.time()
        res = R.construct_draw(Mp, Sp, f["dose"] * tgt[f["q"]], base_seed,  # N1: construct_draw
                               grids=grids[f["q"]], **kw)
        dt = time.time() - t0
        t_con += dt
        recs[name] = {**{k: v for k, v in res.items() if k != "Ue"}, "family_name": fam, "slot": s,
                      "base_seed": base_seed, "wall_seconds": dt}
        if res["matched"]:
            bases[name] = [V[L] @ res["Ue"][L] for L in range(len(V))]
            for L, U in enumerate(bases[name]):
                C.orthonormal_or_die(U, f"{name} L{L}")
    return recs, bases, t_con


def verify_entries(band, bases, cons):
    """verify_pass entries (§2.2): the band under each matched quantity, then every constructed basis under
    its family's quantity: N on T or on all positions (through W_q), E on T (identity)."""
    how = {"N_T": ("T", True), "N_all": ("all", True), "E_T": ("T", False)}
    return ([(f"band|{q}", band, *how[q]) for q in how]
            + [(n, U, *how[FAMILIES[cons[n]["family_name"]]["q"]]) for n, U in bases.items()])


def verify_rows(hookq, names, fam):
    """job_metrics → verify_C[fam] as N1's gates() reads it (G-MATCH): per draw index, the hook-measured
    per-layer quantity N on C and its targets, dose × the hook-measured band quantity (heavy layers come
    from the same targets); None for a draw with no basis."""
    f = FAMILIES[fam]
    ref = f["dose"] * np.asarray(hookq["band|" + f["q"]])
    return [{"N": hookq[n], "targets": ref} if n in hookq else None for n in names]


def summed_quantity(U, Mq, Sq):
    """Σ_L N_L(U) in model coordinates (the trace is basis-invariant), for G-GEN on D."""
    R = _n1()
    return float(sum(R.realized_quantity(U[L], Mq[L], Sq[L])  # N1: realized_quantity
                     for L in range(len(U))))


def gen_rows(names, bases, fam, opD, bandD):
    """job_metrics → gen_D[fam] as N1's gates() reads it (G-GEN): per draw index, the summed quantity on D
    and dose × the band's; None for a draw with no basis."""
    f = FAMILIES[fam]
    return [{"N_sum": summed_quantity(bases[n], *opD[f["q"]]), "target_sum": f["dose"] * bandD[f["q"]]}
            if n in bases else None for n in names]


# ---- --gram-only: G-GEN at construction sizes 32, 64 and 98 windows (pure; gram_only runs them) ----

GRAM_BLOCKS = (range(0, 32), range(32, 64), range(64, F0))     # disjoint; one gram_pass each
GRAM_C = (32, 64, F0)                                          # C = windows 0..n-1, the blocks below n


def add_grams(blocks):
    """gram_pass's all-position and text-position Grams summed over disjoint window blocks (fp32)."""
    return {k: [sum(b[k][L] for b in blocks) for L in range(len(blocks[0][k]))] for k in ("all", "T")}


def gen_sets(n_construct):
    """G-GEN's window sets for C = windows 0..n_construct-1: windows 0-97, and the windows of 0-97 outside C
    (main()'s D, the set the registered G-GEN reads) when there are any."""
    return [list(range(F0))] + ([list(range(n_construct, F0))] if n_construct < F0 else [])


def gen_quantities(bases, cons, band, Ms, g):
    """G-GEN's per-layer inputs on one window set's Grams g, through main()'s Gram-on-D operators (opD): the
    band's quantity on each matched quantity, and each basis's on its family's. summed_quantity is the sum of
    these terms in this order."""
    R = _n1()
    grams = arm_stats_grams(g, None, Ms)
    eye = np.eye(Ms[0].shape[0])
    op = {"N_T": (Ms, grams["T"]), "N_all": (Ms, grams["all"]), "E_T": ([eye] * len(Ms), grams["T"])}

    def per(U, q):
        Mq, Sq = op[q]
        return [R.realized_quantity(U[L], Mq[L], Sq[L]) for L in range(len(U))]  # N1: realized_quantity
    return {"band": {q: per(band, q) for q in op},
            "draws": {n: per(U, FAMILIES[cons[n]["family_name"]]["q"]) for n, U in bases.items()}}


def gencal_entry(C_w, fam_draws, cons, hookq, gen):
    """One construction size of gencal.json, from the job's gate functions. Per family: draws attempted and
    matched; G-REACH, G-PIVOT, G-EXHAUST (family_gates); G-MATCH on C (match_gate on verify_rows); on each
    G-GEN window set (gen: {key: gen_quantities}) each draw's summed ratio to dose × the band's (gen_rows'
    ratio; pass within 5%), its per-layer ratios and its worst heavy layer (target >= 1% of the sum); the
    valid draws (matched, G-MATCH and G-GEN, as N1's gates() counts them). Then G-DRAWS:R and G-DRAWS:T
    (valid >= 0.8 × attempted) on each set."""
    R = _n1()
    fams = {}
    for fam, names in fam_draws.items():
        if not names:
            continue
        f = FAMILIES[fam]
        match = {n: match_gate(r["N"], r["targets"], C_w)
                 for n, r in zip(names, verify_rows(hookq, names, fam)) if r is not None}
        row = {"attempted": len(names), "matched": sum(bool(cons[n]["matched"]) for n in names),
               **family_gates(cons, names),
               "G-MATCH": {"n_pass": sum(m["pass"] for m in match.values()), "draws": match},
               "G-GEN": {}, "valid": {}}
        for kD, q in gen.items():
            band = q["band"][f["q"]]
            t = f["dose"] * np.asarray(band)
            heavy = np.flatnonzero(t >= 0.01 * t.sum())
            draws = {}
            for n in names:
                if n in q["draws"]:
                    ratio = float(sum(q["draws"][n])) / (f["dose"] * float(sum(band)))
                    per = np.asarray(q["draws"][n]) / t
                    Lw = int(heavy[np.argmax(np.abs(per[heavy] - 1))])
                    draws[n] = {"ratio_sum": ratio, "pass": abs(ratio - 1) <= R.GEN_TOL, "ratio_per_layer": per,
                                "worst_heavy_layer": Lw, "worst_heavy_ratio": float(per[Lw])}
            rs = [d["ratio_sum"] for d in draws.values()]
            row["G-GEN"][kD] = {"n": len(draws), "n_pass": sum(d["pass"] for d in draws.values()),
                                "median_ratio_sum": float(np.median(rs)) if rs else None,
                                "worst_ratio_sum": max(rs, key=lambda r: abs(r - 1)) if rs else None,
                                "draws": draws}
            row["valid"][kD] = [n for n in names if n in match and match[n]["pass"] and draws[n]["pass"]]
        fams[fam] = row
    need = lambda fam: R.DRAWS_MIN_SHARE * fams[fam]["attempted"]
    draws_gate = {stat: {kD: {"valid": len(v), "planned": fams[fam]["attempted"], "pass": len(v) >= need(fam)}
                         for kD, v in fams[fam]["valid"].items()}
                  for stat, fam in (("R", "rt"), ("T", "emt")) if fam in fams}
    return {"families": fams, "G-DRAWS": draws_gate}


def arm_stats_grams(g, V, Ms):
    """arm_stats' grams on one split (N1): Σ_all and Σ_T in fp64, C's eigenbasis V, and this split's
    realized-share denominators tr(M Σ), the sum eigencoords returns as denom."""
    S_all = [x.astype(np.float64) for x in g["all"]]
    S_T = [x.astype(np.float64) for x in g["T"]]
    return {"all": S_all, "T": S_T, "V": V,
            "denom_all": [float((M * S).sum()) for M, S in zip(Ms, S_all)],
            "denom_T": [float((M * S).sum()) for M, S in zip(Ms, S_T)]}


def arm_stats_means(mu, counts, g, n_T):
    """arm_stats' means on C (N1): class means as rows (class_rows), positions per class in the
    all-position Gram in the same order (class_inputs' counts), the text row, Σ_{t∈T} h_t and |T|. D gets
    None: its Gram does not hold the positions these means were taken over."""
    n_prefix = len(counts) - 1
    cnt = np.array([counts[i] for i in range(n_prefix)] + [counts["text"]], dtype=np.float64)
    return {"mu": mu, "counts": [cnt] * len(mu), "text": [n_prefix] * len(mu),
            "T_sum": [x.astype(np.float64) for x in g["sT"]], "T_count": [n_T] * len(mu)}


def sample_window(w, u, target, tok):
    """sample_record's window (N1, §10.1): the processor input as the model saw it, the 256 <image> tokens
    shown as a count, then <bos>, the window text and the "\n" decoded; the scored targets win_ids[1:]."""
    ids = u["input_ids"][0]
    return {"index": w, "n_text": u["n_text"], "target_ids": target,
            "target_strs": tok.convert_ids_to_tokens(target.tolist()),
            "text_seen": f"<image>×{N_IMAGE} " + tok.decode(ids[N_IMAGE:].tolist(), skip_special_tokens=False)}


# ===========================================================================
# The GPU side
# ===========================================================================

class PG1:
    """The before-model and its forward conditions (§2.1, §4.3): batch 1, one mask mode per call."""

    def __init__(self, model, layers, dev):
        self.model, self.layers, self.dev = model, layers, dev
        self._masks = {}
        self.fwd_s, self.n_fwd = 0.0, 0

    def mask(self, u, mode):
        if mode == "native":
            return u["attention_mask"].to(self.dev)
        key = (mode, u["input_ids"].shape[1], u["pos"])
        if key not in self._masks:
            self._masks[key] = forced_mask(mode, key[1], key[2]).to(self.dev)
        return self._masks[key]

    def forward(self, u, mode, input_ids=None):
        ids = u["input_ids"] if input_ids is None else input_ids
        return self.model(input_ids=ids.to(self.dev), attention_mask=self.mask(u, mode),
                          pixel_values=u["pixel_values"].to(self.dev).to(torch.bfloat16)).logits[0]

    def scores(self, units, windows, mode, hooks=(), state=None, acc_windows=()):
        """Score windows with optional forward_pre_hooks [(module, fn)]; state carries pos, n_text and
        whether this window accumulates (acc_windows)."""
        handles = [m.register_forward_pre_hook(fn) for m, fn in hooks]
        out = {}
        t0 = time.time()
        with torch.inference_mode():
            for w in windows:
                u = units[w]
                if state is not None:
                    state.update(pos=u["pos"], n_text=u["n_text"], acc=w in acc_windows)
                logits = self.forward(u, mode)
                assert logits.shape[-1] == VOCAB, logits.shape
                out[w] = score_logits(logits, u["pos"], u["n_text"], u["win_ids"].to(self.dev))
                del logits
        self.fwd_s += time.time() - t0
        self.n_fwd += len(out)
        for h in handles:
            h.remove()
        return out

    def plain(self, units, windows, mode, hooks, state):
        """Forward windows for record-only hooks (Gram, verification); logits unused."""
        handles = [m.register_forward_pre_hook(fn) for m, fn in hooks]
        with torch.inference_mode():
            for w in windows:
                u = units[w]
                state.update(pos=u["pos"], n_text=u["n_text"])
                self.forward(u, mode)
        for h in handles:
            h.remove()


def text_probe(pg, u, mode, the_id, a_id, read_layer=N_LAYERS - 1):
    """§2.4 text causality probe: replace the last 64 window tokens by "▁the" ("▁a" where the original is
    "▁the"); d_before / d_after = max |Δ fp32 logit| at text positions before / at-and-after the edit;
    max |Δh| at layer 17's q_proj input over the 257 prefix positions."""
    p, nt = u["pos"], u["n_text"]
    e = p + nt - PROBE_EDIT
    ids = u["input_ids"].clone()
    ids[0, e:p + nt] = torch.where(ids[0, e:p + nt] == the_id, torch.tensor(a_id), torch.tensor(the_id))
    cap = {}

    def rec(key):
        def hook(_m, args):
            cap[key] = args[0][0, :p].to(torch.float32).clone()
        return hook
    q = pg.layers[read_layer].self_attn.q_proj
    with torch.inference_mode():
        h = q.register_forward_pre_hook(rec("orig"))
        l0 = pg.forward(u, mode)[p:].float()
        h.remove()
        h = q.register_forward_pre_hook(rec("edit"))
        l1 = pg.forward(u, mode, ids)[p:].float()
        h.remove()
    d = (l1 - l0).abs().amax(dim=-1)
    k = e - p
    dh = float((cap["edit"] - cap["orig"]).abs().max())
    return {"d_before": float(d[:k].max()), "d_after": float(d[k:].max()), "edit_start": e,
            "max_dh_prefix_L17": dh, "image_sees_text": dh >= 1e-3}


def image_probe(pg, u, mode, uvec):
    """§2.4 image-block probe: perturb layer 0's k_proj and v_proj inputs at image positions 200-255
    (out of place); r_prior / r_self = relative Frobenius change of the layer-0 attention output (o_proj
    input) at positions 0-199 / 200-255."""
    att = pg.layers[0].self_attn
    cap = {}

    def rec(key):
        def hook(_m, args):
            cap[key] = args[0][0].to(torch.float32).clone()
        return hook
    with torch.inference_mode():
        h = att.o_proj.register_forward_pre_hook(rec("base"))
        pg.forward(u, mode)
        h.remove()
        hs = [att.o_proj.register_forward_pre_hook(rec("pert")),
              att.k_proj.register_forward_pre_hook(probe_add_hook(uvec)),
              att.v_proj.register_forward_pre_hook(probe_add_hook(uvec))]
        pg.forward(u, mode)
        for h in hs:
            h.remove()
    d, base = cap["pert"] - cap["base"], cap["base"]
    r = lambda lo, hi: float(d[lo:hi].norm() / base[lo:hi].norm())
    return {"r_prior": r(*PROBE_PRIOR), "r_self": r(*PROBE_PERTURB)}


def probe_mode(pg, u, mode, the_id, a_id, uvec):
    R = _n1()
    tp, ip = text_probe(pg, u, mode, the_id, a_id), image_probe(pg, u, mode, uvec)
    read = R.mask_probe_verdict(tp["d_before"], tp["d_after"],  # N1: mask_probe_verdict
                                ip["r_prior"], ip["r_self"])
    return {**tp, **ip, "mode": read}


def gram_pass(pg, units, windows, mode, class_sums):
    """Σ_all and Σ_T (scored text positions) per layer in fp32 on device (gutenberg_gb1.py:164-165), and
    with class_sums the per-class sums for the mean-ablation means (§2.3) and the T-position sum."""
    dev, n_layers = pg.dev, len(pg.layers)
    d = pg.layers[0].self_attn.q_proj.in_features
    z = lambda *s: [torch.zeros(*s, dtype=torch.float32, device=dev) for _ in range(n_layers)]
    G_ = {"all": z(d, d), "T": z(d, d)}
    if class_sums:
        G_.update(prefix=z(N_PREFIX, d), text=z(d), sT=z(d))
    n = {"text": 0, "T": 0}
    state = {}

    def mk(L):
        def hook(_m, args):
            h = args[0][0].to(torch.float32)
            p, nt = state["pos"], state["n_text"]
            hT = h[p:p + nt - 1]
            G_["all"][L] += h.T @ h
            G_["T"][L] += hT.T @ hT
            if class_sums:
                G_["prefix"][L] += h[:p]
                G_["text"][L] += h[p:].sum(dim=0)
                G_["sT"][L] += hT.sum(dim=0)
        return hook
    pg.plain(units, windows, mode, [(pg.layers[L].self_attn.q_proj, mk(L)) for L in range(n_layers)],
             state)
    for w in windows:
        n["text"] += units[w]["input_ids"].shape[1] - units[w]["pos"]
        n["T"] += units[w]["n_text"] - 1
    host = {k: [v.cpu().numpy() for v in vs] for k, vs in G_.items()}
    return host, n


def verify_pass(pg, units, windows, mode, entries):
    """§2.2 verification through the hooks, independent of the Gram: per layer Σ_t ||W_q U Uᵀ h_t||² (or
    Σ_t ||Uᵀ h_t||² for the identity) over the entry's positions ("T" or "all"), fp32 per window, summed
    in fp64. entries: [(key, U_by_layer, positions, use_W)] → {key: ndarray[18]}."""
    dev, n_layers = pg.dev, len(pg.layers)
    W = [pg.layers[L].self_attn.q_proj.weight.detach().to(torch.float32) for L in range(n_layers)]
    per_layer = [[] for _ in range(n_layers)]
    for key, U, positions, use_W in entries:
        for L in range(n_layers):
            Ud = torch.as_tensor(np.ascontiguousarray(U[L]), dtype=torch.float32, device=dev)
            per_layer[L].append((key, Ud, (W[L] @ Ud) if use_W else None, positions))
    acc = {key: [0.0] * n_layers for key, *_ in entries}
    state = {}

    def mk(L):
        def hook(_m, args):
            h = args[0][0].to(torch.float32)
            p, nt = state["pos"], state["n_text"]
            hs = {"T": h[p:p + nt - 1], "all": h}
            for key, Ud, WU, positions in per_layer[L]:
                c = hs[positions] @ Ud
                v = c @ WU.T if WU is not None else c
                acc[key][L] = acc[key][L] + v.pow(2).sum().double()
        return hook
    pg.plain(units, windows, mode, [(pg.layers[L].self_attn.q_proj, mk(L)) for L in range(n_layers)],
             state)
    return {key: np.array([float(x) for x in v]) for key, v in acc.items()}


def sha256_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def sample_payload(header, sel, scores, none, units, targets, tok):
    """One §10.1 sample file: the header, the max-Y, min-Y and common windows (N1 sample_record each, with
    its role), and one right and one wrong example."""
    R = _n1()

    def top(s):
        return (s["top_id"], tok.convert_ids_to_tokens(s["top_id"].tolist()), np.exp(s["top_lp"]))
    recs = []
    for role in ("max", "min", "common"):
        w = sel[role]
        rec = R.sample_record(sample_window(w, units[w], targets[w], tok),  # N1: sample_record
                              none[w]["lp"], scores[w]["lp"], top(none[w]), top(scores[w]))
        recs.append({"role": role, **rec})
    order = [sel["common"], sel["max"], sel["min"]]
    ex = sample_examples(order, targets, none, scores, N_PREFIX,
                         lambda i: tok.convert_ids_to_tokens(int(i)))
    return {**header, "windows": recs, "examples": ex}


def gram_only(a, sink, pg, units, band, gate_env, lap, stages):
    """--gram-only: gram_pass over each of GRAM_BLOCKS (windows 0-97), then for each C of GRAM_C: eigen_sets,
    grids, every RT-family and EM_T draw at the registered seeds (construct_family), verify_pass on that C,
    and G-GEN on each gen_sets window set (gencal_entry). gencal.json and compute.json are rewritten after
    each C. No NLL pass, no samples, nothing under sealed/."""
    def stage(name, **kw):
        sink.event(name, **kw)
        sink.json("progress.json", {"stage": name, **kw, "utc": sink.now()})

    stage("gram")
    blocks = [gram_pass(pg, units, list(r), a.mask, class_sums=False)[0] for r in GRAM_BLOCKS]
    lap("gram")
    W64 = [layer.self_attn.q_proj.weight.detach().to(torch.float64).cpu().numpy() for layer in pg.layers]
    Ms = [W.T @ W for W in W64]
    del W64
    sl = draw_slots(a)
    used_q = sorted({FAMILIES[f]["q"] for f in FAMILIES if sl[f]})
    doc = {"program": "realizednull", "job": "realizednull_pg1", "mode": "gram_only", "G-ENV": gate_env,
           "gram_blocks": [wkey("block", r) for r in GRAM_BLOCKS], "by_C": {}}
    comp = {"t_con": {}, "construction_wall_s": {}}
    for n_c in GRAM_C:
        C_w = list(range(n_c))
        kC = wkey("C", C_w)
        stage("construct", n_construct=n_c)
        t0 = time.time()
        gC = add_grams([b for b, r in zip(blocks, GRAM_BLOCKS) if r.stop <= n_c])
        ecT, ecA = eigen_sets(gC, (layer.self_attn.q_proj.weight.detach().to(torch.float64).cpu().numpy()
                                   for layer in pg.layers))
        del gC
        fam_in = quantity_inputs(ecT, ecA)
        tgt = band_targets(fam_in, band)
        grids = quantity_grids(fam_in, used_q, CONSTRUCT_KW)
        lap(f"grid_{kC}")
        cons, fam_draws, built, t_con = {}, {}, {}, 0.0
        for fam in FAMILIES:
            recs, bases, t = construct_family(fam, sl[fam], fam_in, tgt, grids, {**CONSTRUCT_KW, "max_redraws": 3})
            cons.update(recs)
            fam_draws[fam] = list(recs)
            built.update(bases)
            t_con += t
        del ecT, ecA, fam_in
        n_ld = sum(len(v) for v in fam_draws.values()) * len(pg.layers)
        comp["t_con"][kC] = t_con / n_ld if n_ld else None
        comp["construction_wall_s"][kC] = time.time() - t0
        lap(f"construct_{kC}")
        stage("verify_c", n_construct=n_c)
        hookq = verify_pass(pg, units, C_w, a.mask, verify_entries(band, built, cons))
        lap(f"verify_c_{kC}")
        stage("gen", n_construct=n_c)
        gen = {}
        for ws in gen_sets(n_c):
            g = add_grams([b for b, r in zip(blocks, GRAM_BLOCKS) if r.start >= ws[0]])
            assert sorted(w for r in GRAM_BLOCKS if r.start >= ws[0] for w in r) == ws, ws
            gen[wkey("D", ws)] = gen_quantities(built, cons, band, Ms, g)
            del g
        entry = gencal_entry(C_w, fam_draws, cons, hookq, gen)
        doc["by_C"][kC] = {**entry, "n_construct": n_c, "t_con": comp["t_con"][kC],
                           "construction_wall_s": comp["construction_wall_s"][kC],
                           "gen_sets": {k: "outside C" if k != wkey("D", range(F0)) else "contains C" for k in gen},
                           "construction": cons, wkey("band_targets", C_w): tgt, "grids": grids,
                           wkey("band_hook", C_w): {k.split("|")[1]: v for k, v in hookq.items()
                                                    if k.startswith("band|")}}
        lap(f"gen_{kC}")
        sink.json("gencal.json", doc)
        sink.json("compute.json", {"seconds_per_stage": stages, **comp,
                                   "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                                   "peak_gpu_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30})
        for fam, r in entry["families"].items():
            print(f"[gencal] {kC} {fam}: attempted {r['attempted']} matched {r['matched']} "
                  f"G-MATCH {r['G-MATCH']['n_pass']} " + " ".join(
                      f"| {k} G-GEN {v['n_pass']}/{v['n']} median {v['median_ratio_sum']} worst {v['worst_ratio_sum']}"
                      for k, v in r["G-GEN"].items()), flush=True)
        del built, hookq, gen, cons
    sink.event("done")
    print(f"REALIZEDNULL_PG1_GENCAL_OK sizes={list(doc['by_C'])}", flush=True)


def main(argv=None):
    a = parse_args(argv)
    sp = splits(a.n_windows, a.n_construct)
    sink = Sink(a.out, None if a.gram_only else "smoke" if a.smoke else "discovery")
    budget_s = budget_s_from_cmdline(parent_cmdline())
    sink.event("start", budget_s=budget_s)
    t_stage, stages = time.time(), {}

    def lap(name):
        nonlocal t_stage
        stages[name] = stages.get(name, 0.0) + time.time() - t_stage
        t_stage = time.time()

    hf_setup()
    import transformers
    from huggingface_hub import try_to_load_from_cache
    from transformers import PaliGemmaProcessor
    R = _n1()
    prov = R.provenance()  # N1: provenance

    # ---- banks: band, anti (hash-verified), Haar s0-s2, GB-1's banked twins (§4.1) ----
    wk = pathlib.Path("/tmp/realizednull_pg1")
    wk.mkdir(parents=True, exist_ok=True)
    man = json.loads(_fetch(A1_MANIFEST, wk / "a1_manifest.json").read_text())
    assert man["base_sha"] == BASE_SHA, man["base_sha"]
    want = {(b["bank"], b["layer"]): b["sha256"] for b in man["banks"] if b["k"] == K}
    a1 = np.load(_fetch(A1_NPZ, wk / "a1_banks.npz"))
    gb1_path = _fetch(GB1_BANKS, wk / "gb1_banks.npz")
    gb1 = np.load(gb1_path)
    B = {"ident": [np.zeros((N_IN, 0)) for _ in range(N_LAYERS)]}
    src = {"ident": ("U with 0 columns", None)}
    rng = np.random.default_rng(1337)                 # gutenberg_gb1.py:89-97
    for L in range(N_LAYERS):
        for bid, name in (("core", "band"), ("anti", "anti32")):
            Q = np.asarray(a1[f"{bid}/{K}/{L}"], dtype=np.float64)
            assert C.bank_hash(Q) == want[(bid, L)], f"{bid}/32/{L} hash mismatch"
            C.orthonormal_or_die(Q, f"{bid}32 L{L}")
            B.setdefault(name, []).append(Q)
        for s in range(3):
            B.setdefault(f"haar_s{s}", []).append(G.haar_frame(N_IN, K, rng))
        for s in range(3):
            for gname, name in ((f"energymatched32_s{s}", f"em_s{s}"), (f"specmatched32_s{s}", f"spec_s{s}")):
                Q = np.asarray(gb1[f"{gname}/{L}"], dtype=np.float64)
                C.orthonormal_or_die(Q, f"{gname} L{L}", tol=1e-5)       # banked fp32
                B.setdefault(name, []).append(Q)
        B.setdefault("topenergy32", []).append(G.coordinate_bank(N_IN, gb1[f"topenergy32_idx/{L}"]))
    src.update({"band": ("a1_banks.npz core/32/{L}", None), "anti32": ("a1_banks.npz anti/32/{L}", None),
                "topenergy32": ("gb1_banks.npz topenergy32_idx/{L} via coordinate_bank", None)})
    for s in range(3):
        src[f"haar_s{s}"] = ("default_rng(1337), 3 frames per layer in layer order", 1337)
        src[f"em_s{s}"] = (f"gb1_banks.npz energymatched32_s{s}/{{L}}", None)
        src[f"spec_s{s}"] = (f"gb1_banks.npz specmatched32_s{s}/{{L}}", None)
    gb1_nll, gb1_nll_sha = None, None
    if a.smoke:
        p = _fetch(GB1_NLL, wk / "gb1_window_nll.json")
        gb1_nll, gb1_nll_sha = json.loads(p.read_text()), sha256_file(p)
    lap("banks")

    # ---- model, processor, env (G-ENV) ----
    dev = "cuda"
    assert torch.cuda.is_available()
    sink.event("model_load")
    model = _load_full_pg1(BASE_SHA, dev)
    sink.event("model_loaded")
    assert model.config._attn_implementation == "eager", model.config._attn_implementation
    processor = PaliGemmaProcessor.from_pretrained(BASE_REPO, revision=BASE_SHA)
    tok = processor.tokenizer
    layers = _decoder_layers(model)
    blank = _blank_image()
    pg = PG1(model, layers, dev)
    cfg = try_to_load_from_cache(BASE_REPO, "config.json", revision=BASE_SHA)
    assert isinstance(cfg, str), cfg
    cfg_sha = sha256_file(cfg)
    assert cfg_sha == json.loads(REPO_FACTS.read_text())[BASE_REPO]["config_sha256"], cfg_sha
    env = {"transformers": transformers.__version__, "torch": torch.__version__,
           "numpy": np.__version__, "cuda": torch.version.cuda,
           "gpu_name": torch.cuda.get_device_name(0), "instance_type": instance_type(),
           "_attn_implementation": model.config._attn_implementation, "mask": a.mask,
           "forced_masks": a.forced_masks, "args": vars(a), "budget_s": budget_s,
           "pg_git_sha": os.environ.get("PG_GIT_SHA"), "provenance": prov,
           "base": f"{BASE_REPO}@{BASE_SHA}", "snapshot_config": cfg, "config_sha256": cfg_sha,
           "a1_manifest_sha256": man["manifest_sha256"], "gb1_banks_sha256": sha256_file(gb1_path),
           "gb1_window_nll_sha256": gb1_nll_sha, "wikitext_rev": WIKITEXT_REV}
    gates = {"G-ENV": env_gate(env)}
    jm = {"env": {k: env[k] for k in R.ENV_KEYS + ("_attn_implementation",)}, "mask": a.mask}
    sink.json("env.json", env)
    lap("model")
    if a.gram_only:
        sink.event("corpus")
        texts = build_windows(tok, a.n_windows, WINDOW)
        units, _ = make_units(texts, tok, processor, blank)
        image_id = tok.convert_tokens_to_ids("<image>")
        for u in units:
            assert u["pos"] == N_PREFIX, u["pos"]
            assert bool((u["input_ids"][0, :N_IMAGE] == image_id).all()) and \
                int(u["input_ids"][0, N_IMAGE]) == tok.bos_token_id
        lap("corpus")
        return gram_only(a, sink, pg, units, B["band"], gates["G-ENV"], lap, stages)

    # ---- probes (window 0), then the forced-mask self-test step (§2.4) ----
    rec_probe = {}
    sink.event("probes")
    w0_text = build_windows(tok, 1, WINDOW)[0]
    u0 = make_units([w0_text], tok, processor, blank)[0][0]
    the_id, a_id = tok.convert_tokens_to_ids("▁the"), tok.convert_tokens_to_ids("▁a")
    assert tok.unk_token_id not in (the_id, a_id), (the_id, a_id)
    uvec = torch.as_tensor(probe_vector(), dtype=torch.float32, device=dev)
    native = probe_mode(pg, u0, "native", the_id, a_id, uvec)
    rec_probe["native"] = native
    passed = []
    if a.forced_masks == "on":
        sink.event("forced_mask_selftest")
        rec_probe["forced"] = {}
        for m in FORCED:
            r = probe_mode(pg, u0, m, the_id, a_id, uvec)
            rec_probe["forced"][m] = {**r, "pass": r["mode"] == m}
        sink.event("forced_mask_selftest_done")
        passed = [m for m in FORCED if rec_probe["forced"][m]["pass"]]
    sink.json("mask_probe.json", rec_probe)
    jm["mask_probe"] = probe_numbers(rec_probe, a.forced_masks)
    gates["G-MASK"] = {"native_mode": native["mode"], "pass": native["mode"] in FORCED,
                       "image_sees_text": native["image_sees_text"], "forced_masks": a.forced_masks,
                       "forced_selftest_passed": passed}
    assert a.mask == "native" or a.mask in passed, f"--mask {a.mask} failed its forced-mask self-test"
    env.update(native_mode=native["mode"], forced_selftest_passed=passed)
    sink.json("env.json", env)
    lap("probes")

    # ---- corpus (§4.1) ----
    arms = roster(a)
    k1_arms = [f"none_{m}" for m in passed]
    rec = Recorder(sink, sp, ["none"] + k1_arms + [x[0] for x in arms])
    rec.stage("corpus")
    texts = build_windows(tok, a.n_windows, WINDOW)
    assert texts[0] == w0_text
    units, trimmed = make_units(texts, tok, processor, blank)
    image_id = tok.convert_tokens_to_ids("<image>")
    for u in units:
        assert u["pos"] == N_PREFIX, u["pos"]
        assert bool((u["input_ids"][0, :N_IMAGE] == image_id).all()) and \
            int(u["input_ids"][0, N_IMAGE]) == tok.bos_token_id
    targets = [u["win_ids"][1:].numpy() for u in units]
    join = lambda ws: hashlib.sha256("\x00".join(texts[w] for w in ws).encode("utf-8")).hexdigest()
    corpus = {wkey("window_text_sha256", sp["R"]): join(sp["R"]),
              wkey("trimmed", sp["R"]): [w for w in trimmed if w < F0],
              "recipe": f"wikitext-103-raw-v1:test@{WIKITEXT_REV} {a.n_windows}x{WINDOW} (build_windows)"}
    if sp["F"]:
        sink.json("sealed/corpus_f.json", {"window_text_sha256": join(sp["F"]),
                                           "trimmed": [w for w in trimmed if w >= F0]})
    info = {"smoke": a.smoke, "mask": a.mask, "forced_masks": a.forced_masks,
            "topup_families": a.topup_families, "slot_offset": a.slot_offset, "n_windows": a.n_windows,
            "corpus": corpus, "provenance": prov, "stopped": None}
    lap("corpus")

    def header(arm, op, basis, mode):
        source, seed = src.get(basis, (None, None))
        after = {"none": f"the before-model under mask {mode}",
                 "zero": "the before-model + forward_pre_hook on self_attn.q_proj of all 18 layers: "
                         "h32 - (h32 @ U_L) @ U_L.T, cast to bf16",
                 "zero_text": "as zero, at text-block positions (pos..end) only",
                 "mean": "the before-model + forward_pre_hook on self_attn.q_proj of all 18 layers: "
                         "h32 - ((h32 - mu_{L,c(t)}) @ U_L) @ U_L.T, cast to bf16"}[op]
        return {"arm": arm, "op": op, "basis": basis, "basis_source": source, "seed": seed,
                "basis_sha256": None if basis is None else C.bank_hash(np.concatenate(B[basis], axis=1)),
                "before_model": f"PG1 {BASE_REPO}@{BASE_SHA}, bf16, eager, mask {a.mask}, input = 256 "
                                "<image> + <bos> + window text + newline (PaliGemmaProcessor, gray-128 "
                                "224x224 image), batch 1, inference_mode, no hooks",
                "after_model": after}

    def samples(arm, hdr, scores):
        sel, _ = rec.select(scores)
        pay = lambda s: None if s is None else sample_payload(hdr, s, scores, rec.none, units, targets, tok)
        rec.write_samples(arm, pay(sel["open"]), pay(sel["sealed"]))

    def write_compute(t_con):
        sink.json("compute.json", {
            "seconds_per_stage": stages, "s_fwd": pg.fwd_s / max(pg.n_fwd, 1),
            "n_window_forwards_scored": pg.n_fwd, "t_con": t_con,
            "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            "peak_gpu_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30})

    def stop(gate, after):
        info["stopped"] = {"gate": gate, "after": after}
        sink.json("metrics.json", metrics_doc(rec, gates, info, jm))
        write_compute(None)
        sink.event("stop", gate=gate)
        print(f"REALIZEDNULL_PG1_STOP gate={gate} after={after}", flush=True)

    # ---- none on all windows, then K1 on windows 0-97 (§9.4) ----
    sink.event("arm", arm="none")
    none = pg.scores(units, range(a.n_windows), a.mask)
    rec.nll("none", "none", none)
    samples("none", header("none", "none", None, a.mask), none)
    rec.done("none")
    for m in passed:
        arm = f"none_{m}"
        sink.event("arm", arm=arm)
        sc = pg.scores(units, sp["R"], m)
        rec.nll(arm, "none", sc)
        samples(arm, header(arm, "none", None, m), sc)
        rec.done(arm)
    gates["P1"] = p1_rows(rec.nll_open, {m: rec.summary[f"none_{m}"][wkey("ppl", sp["R"])] for m in passed},
                          gb1_nll, sp["R"], a.smoke)
    lap("none_k1")
    if not a.smoke:
        gates["G-PRE1"] = pre1_gate(rec.summary["none"][wkey("ppl", sp["R"])], sp["R"])
        if a.mask == "native" and not gates["G-PRE1"]["pass"]:
            return stop("G-PRE1", "K1")

    # ---- Grams on C (all, text), class means (§2.2, §2.3) ----
    rec.stage("gram_c")
    gC, nC = gram_pass(pg, units, sp["C"], a.mask, class_sums=True)
    sums, counts = class_inputs(gC, len(sp["C"]), nC["text"])
    mu = [class_rows(R.class_means(sums[L], counts)) for L in range(N_LAYERS)]  # N1: class_means
    assert all(m.shape == (N_PREFIX + 1, N_IN) for m in mu), mu[0].shape
    del sums
    tope = {}
    for L in range(N_LAYERS):                         # recompute-and-compare only (§9.2)
        idx = np.argsort(np.diag(gC["all"][L]))[::-1][:K]
        tope[L] = len(set(idx.tolist()) & set(np.asarray(gb1[f"topenergy32_idx/{L}"]).tolist()))
    np.savez(sink.out / "grams_c.npz", **{f"all/{L}": gC["all"][L] for L in range(N_LAYERS)},
             **{f"T/{L}": gC["T"][L] for L in range(N_LAYERS)})
    lap("gram_c")

    # ---- grid scan and every construction (§2.2, §3.2) ----
    rec.stage("construct")
    sl = draw_slots(a)
    ecT, ecA = eigen_sets(gC, (layers[L].self_attn.q_proj.weight.detach().to(torch.float64).cpu().numpy()
                               for L in range(N_LAYERS)))
    VT = [e["V"] for e in ecT]                        # C's eigenbasis, kept for arm_stats
    band = B["band"]
    fam_in = quantity_inputs(ecT, ecA)
    tgt = band_targets(fam_in, band)                  # the band's per-layer quantity (§2.2 targets)
    used_q = sorted({FAMILIES[f]["q"] for f in FAMILIES if sl[f]})
    grids = quantity_grids(fam_in, used_q, CONSTRUCT_KW)
    lap("grid")
    e_tgt = [float(np.sum((gC["all"][L] @ band[L]) * band[L]) / np.trace(gC["all"][L]))
             for L in range(N_LAYERS)]
    gates["G-EMATCH"], ematch = {"draws": {}}, {}
    for s in [s for s in sl["em"] if s >= 3]:
        Qs, er = [], []
        for L in range(N_LAYERS):
            Q, e = energy_matched_subspace(VT[L], ecT[L]["mu"], e_tgt[L], K, seed=70_000 + 1000 * s + L,
                                           tol=max(0.02 * e_tgt[L], 1e-6), max_resample=200)
            Qs.append(np.ascontiguousarray(Q))
            er.append(e)
        B[f"em_s{s}"] = Qs
        src[f"em_s{s}"] = ("energy_matched_subspace, seed 70_000 + 1000*s + L", 70_000 + 1000 * s)
        gates["G-EMATCH"]["draws"][f"em_s{s}"] = {**ematch_gate(e_tgt, er),
                                                  wkey("e_target", sp["C"]): e_tgt,
                                                  wkey("e_real", sp["C"]): er}
        ematch[s] = [{"e_target": t, "e_realized": e} for t, e in zip(e_tgt, er)]
    jm["ematch"] = [ematch.get(s) for s in sl["em"]]       # None for the banked s0-s2
    gm = gates["G-EMATCH"]["draws"].values()
    gates["G-EMATCH"].update(n=len(gm), n_pass=sum(g["pass"] for g in gm))
    for s in [s for s in sl["spec"] if s >= 3]:
        Us = []
        for L in range(N_LAYERS):
            A = VT[L].T @ band[L]
            srng = np.random.default_rng(60_000 + 1000 * s + L)
            Qs, _ = np.linalg.qr(VT[L] @ (A * srng.choice([-1.0, 1.0], size=A.shape)))
            Us.append(np.ascontiguousarray(Qs[:, :K]))
        B[f"spec_s{s}"] = Us
        src[f"spec_s{s}"] = ("QR(V (A∘S)), default_rng(60_000 + 1000*s + L)", 60_000 + 1000 * s)
    for s in [s for s in sl["haar"] if s >= 3]:
        B[f"haar_s{s}"] = [G.haar_frame(N_IN, K, np.random.default_rng(1_000_000 + 1000 * s + L))
                           for L in range(N_LAYERS)]
        src[f"haar_s{s}"] = ("haar_frame, default_rng(1_000_000 + 1000*s + L)", 1_000_000 + 1000 * s)
    for L in range(N_LAYERS):
        W32 = layers[L].self_attn.q_proj.weight.detach().to(torch.float32)
        assert W32.shape == (N_IN, N_IN), W32.shape
        _, _, Vh = torch.linalg.svd(W32, full_matrices=False)        # gutenberg_gb1.py:129-136
        B.setdefault("bottomsigma32", []).append(Vh[-K:, :].T.cpu().numpy().astype(np.float64))
        B.setdefault("topsigma32", []).append(Vh[:K, :].T.cpu().numpy().astype(np.float64))
        del Vh
    src.update({"topsigma32": ("top-32 right singular vectors of W_q", None),
                "bottomsigma32": ("bottom-32 right singular vectors of W_q", None)})
    cons, fam_draws, t_con = {}, {}, 0.0
    for fam, f in FAMILIES.items():
        recs, bases, t = construct_family(fam, sl[fam], fam_in, tgt, grids, {**CONSTRUCT_KW, "max_redraws": 3})
        cons.update(recs)
        fam_draws[fam] = list(recs)
        t_con += t
        for name, U in bases.items():
            B[name] = U
            src[name] = (f"construct_draw on the {f['q']} quantity × {f['dose']}, "
                         f"base seed {f['seed']} + 1000*s (+ 10_000_000*r per redraw)", recs[name]["base_seed"])
    jm["construction"] = {fam: [cons[n] for n in names] for fam, names in fam_draws.items()}
    n_ld = sum(len(v) for v in fam_draws.values()) * N_LAYERS
    t_con = t_con / n_ld if n_ld else None
    for g in ("G-REACH", "G-PIVOT", "G-EXHAUST"):
        gates[g] = {}
    for fam, names in fam_draws.items():
        if names:
            for g, v in family_gates(cons, names).items():
                gates[g][fam] = v
    del ecT, ecA, fam_in
    lap("construct")

    # ---- verification pass on C (G-MATCH) ----
    rec.stage("verify_c")
    built = {n: B[n] for names in fam_draws.values() for n in names if n in B}
    hookq = verify_pass(pg, units, sp["C"], a.mask, verify_entries(band, built, cons))
    gates["G-MATCH"], jm["verify_C"] = {}, {}
    for fam, names in fam_draws.items():
        rows = verify_rows(hookq, names, fam)
        jm["verify_C"][fam] = rows
        if names:
            draws = {n: match_gate(r["N"], r["targets"], sp["C"]) for n, r in zip(names, rows) if r is not None}
            gates["G-MATCH"][fam] = {"draws": draws, "n": len(names),
                                     "n_pass": sum(d["pass"] for d in draws.values())}
    lap("verify_c")

    # ---- bank every constructed basis and μ (§10.2) ----
    loaded = {"ident", "band", "anti32", "topenergy32", "em_s0", "em_s1", "em_s2", "spec_s0", "spec_s1",
              "spec_s2"}                              # fetched banks; the rest were built here
    banked = {f"{n}/{L}": np.asarray(B[n][L], dtype=np.float32) for n in B if n not in loaded
              for L in range(N_LAYERS)}
    banked.update({f"mu/{L}": mu[L].astype(np.float32) for L in range(N_LAYERS)})
    np.savez(sink.out / "banks.npz", **banked)
    sink.json("banks_manifest.json", {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                                      for k, v in banked.items()})

    # ---- Gram on D (G-GEN, D columns of arm_stats) ----
    rec.stage("gram_d")
    gD, _ = gram_pass(pg, units, sp["D"], a.mask, class_sums=False)
    W64 = [layers[L].self_attn.q_proj.weight.detach().to(torch.float64).cpu().numpy()
           for L in range(N_LAYERS)]
    Ms = [W.T @ W for W in W64]
    gramsD = arm_stats_grams(gD, VT, Ms)
    del gD
    opD = {"N_T": (Ms, gramsD["T"]), "N_all": (Ms, gramsD["all"]),
           "E_T": ([np.eye(N_IN)] * N_LAYERS, gramsD["T"])}
    bandD = {q: summed_quantity(band, *opD[q]) for q in used_q}
    gates["G-GEN"], jm["gen_D"] = {}, {}
    for fam, names in fam_draws.items():
        rows = gen_rows(names, built, fam, opD, bandD)
        jm["gen_D"][fam] = rows
        if names:
            draws = {n: {wkey("ratio_sum", sp["D"]): r["N_sum"] / r["target_sum"],
                         "pass": abs(r["N_sum"] / r["target_sum"] - 1) <= 0.05}
                     for n, r in zip(names, rows) if r is not None}
            gates["G-GEN"][fam] = {"draws": draws, "n": len(names),
                                   "n_pass": sum(d["pass"] for d in draws.values())}
    gates["family_marks"] = {fam: [g for g in ("G-REACH", "G-PIVOT", "G-EXHAUST")
                                   if fam in gates[g] and not gates[g][fam]["pass"]]
                             for fam in fam_draws if fam_draws[fam]}
    lap("gram_d")

    # ---- arm_stats (§3.6) on C and D for every basis an arm uses ----
    rec.stage("arm_stats")
    gramsC = arm_stats_grams(gC, VT, Ms)
    meansC = arm_stats_means(mu, counts, gC, nC["T"])
    kC, kD = wkey("C", sp["C"]), wkey("D", sp["D"])
    stats = {kC: {}, kD: {}}
    for basis in sorted({b for _, _, b, _ in arms if b in B and b != "ident"}):
        stats[kC][basis] = R.arm_stats(B[basis], gramsC, W64, meansC)  # N1: arm_stats
        stats[kD][basis] = R.arm_stats(B[basis], gramsD, W64, None)  # N1: arm_stats
    arm_stats = {"per_basis": stats, "construction": cons, "grids": grids,
                 wkey("band_targets", sp["C"]): tgt, wkey("band_hook", sp["C"]): {
                     k.split("|")[1]: v for k, v in hookq.items() if k.startswith("band|")},
                 wkey("topenergy32_overlap_with_banked", sp["C"]): tope,
                 wkey("under_ablation", sp["D"]): {}}
    sink.json("arm_stats.json", arm_stats)
    del gramsC, gramsD, meansC, Ms, opD, W64, gC, VT
    lap("arm_stats")

    # ---- every ablation arm in §9.4 order; preflight checkpoint after the banked arms ----
    W32 = [layers[L].self_attn.q_proj.weight.detach().to(torch.float32) for L in range(N_LAYERS)]
    mu_dev = [torch.as_tensor(mu[L], dtype=torch.float32, device=dev) for L in range(N_LAYERS)]
    last_pre = max(i for i, x in enumerate(arms) if x[3] == "pre")
    for i, (arm, op, basis, group) in enumerate(arms):
        sink.event("arm", arm=arm, i=i, n=len(arms), skipped=basis not in B)
        if basis not in B:
            rec.skip(arm, f"basis {basis} not constructed: " + json.dumps(
                {**{k: bool(cons[basis][k]) for k in ("matched", "reachable", "exhausted")},
                 "fail_reason": cons[basis]["fail_reason"]}))
            continue
        U = [torch.as_tensor(np.ascontiguousarray(B[basis][L]), dtype=torch.float32, device=dev)
             for L in range(N_LAYERS)]
        state = {}
        acc = {"num": [0.0] * N_LAYERS, "den": [0.0] * N_LAYERS}
        hooks = [(layers[L].self_attn.q_proj, ablation_hook(U[L], op, mu_dev[L], state, W32[L], acc, L))
                 for L in range(N_LAYERS)]
        sc = pg.scores(units, range(a.n_windows), a.mask, hooks, state, set(sp["D"]))
        rec.nll(arm, op, sc)
        samples(arm, header(arm, op, basis, a.mask), sc)
        num, den = np.array([float(x) for x in acc["num"]]), np.array([float(x) for x in acc["den"]])
        arm_stats[wkey("under_ablation", sp["D"])][arm] = {"N": num, "den": den, "ratio": num / den}
        sink.json("arm_stats.json", arm_stats)
        rec.done(arm)
        write_compute(t_con)
        if i == last_pre:
            if a.smoke:
                gates["G-PRE0"] = pre0_rows(rec.nll_open, gb1_nll, sp["R"])
                jm["pre0"] = pre0_input(rec.nll_open, gb1_nll, sp["R"])
            else:
                for g, names in PRE_GATING.items():
                    gates[g] = pre_gate(rec.summary, names, sp["R"])
                gates["G-PRE4-recorded"] = pre_gate(rec.summary, PRE_RECORDED, sp["R"])
                miss = [g for g in PRE_GATING if gates[g]["arms"] and not gates[g]["pass"]]
                if a.mask == "native" and miss:
                    return stop(",".join(miss), arm)
    lap("arms")

    sink.json("metrics.json", metrics_doc(rec, gates, info, jm))
    write_compute(t_con)
    sink.event("done")
    print(f"REALIZEDNULL_PG1_OK arms={len(rec.done_arms)} skipped={len(rec.skipped)}", flush=True)


if __name__ == "__main__":
    main()
