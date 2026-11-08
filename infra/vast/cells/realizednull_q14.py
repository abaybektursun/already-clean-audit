"""REALIZEDNULL session 2 (N7): the Qwen2.5-14B census-core cell of SPEC §3.8, run on vast.ai.

Spec: studies/realizednull/SPEC.md §3.8 (object, arms, environment), §2 (readout and instrument), §3.2 (arms and
seeds), §3.6 (arm_stats.json), §6.1 (G14-PRE and the job-side gates), §9.2 N7, §9.6, §10 (artifacts), and §0.4 P7
(C = windows 0-63, D = 64-97). Contract: run(cell, out_dir), as infra/vast/cells/v1_ablation.py:64, called by
infra/vast/run_cells.py. Manifest: infra/vast/manifests/realizednull_q14.json (N8).

GB-3's forward (v1_ablation.py): AutoModelForCausalLM, bf16, attn_implementation="sdpa" (:96), batches of 4 windows
of input_ids (:175), the projection in fp32 on the bf16 stream (:150-166), per-window NLL from log_softmax over
logits[:-1] (:180-183). GB-3's bases: census core and anti with QR (:105-109), Haar s0-s2 from default_rng(1337),
three per layer in layer order (:103, :110-112), top/bottom-σ from the fp32 SVD on the GPU (:121-124), topenergy32
and topenergy128 from the banked indices (:126-129), EM seed 70_000 + 1000·slot + L (:232), spec seed
60_000 + 1000·slot + L (:243). The slots GB-3 banked (EM s3-s5, spec s3-s5) are loaded from its
v1abl_gb_banks.npz; every other slot is constructed at the same seed formula on this cell's Gram.

Text-only model (§3.8): T is all 512 positions of a window, so Σ_T = Σ_all and one set of 48 Grams sits on the GPU
at a time. The mean classes are position 0 and positions 1-511. The X-rows keep op "zero_text" and zero-ablate every
position other than 0. RT_all and K1 are not run. G-MASK is the text causality probe alone.

N1 (infra/pg/realizednull_core.py) does every construction, gate rule and sample record. Each call is marked
`# N1:` here, or `# N1 via N3:` where the N3 helper this module reuses makes it (band_targets, quantity_grids,
construct_family). The eigen-coordinates follow §2.2 for d = 5120: fp32 eigh and fp32 products on the GPU, cast to
fp64 (eigen_layer, tested equal to N1's eigencoords). The grid scan and every construction (construct_all) run in the
main process after the eigen-coordinates and before any arm forward (§0.4 U5). arm_stats (§3.6) and G-GEN's quantities
on D run in a daemon thread, one layer at a time, while the GPU scores the RT-family and EM_T arms.

Inputs come from the mirror INPUTS (the four §9.6 step 1 prefixes copied under results/mdl/, decided 2026-11-02), and
each file's sha256 is asserted against realizednull_q14_inputs.json, recorded from the original prefixes. The corpus
is N9's corpus196.npy (infra/aws_pg/jobs/realizednull_q14_corpus.py), checked against its sidecar hash and against
the 98 windows baked into the image.

Sealing (G-NOLEAK): N3's Sink and Recorder route windows >= 98 to sealed/ only, and nothing over them is printed.
G14-PRE (§6.1) reproduces GB-3's banked rows before any new arm cost. In the full cell a miss writes metrics.json with
`stopped` and returns. env.json carries family "q14", from which N1's family_of picks the session-2 parameter set for
the analysis (scripts/realizednull_analyze.py). Fail loud: nothing here catches an error.
"""
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import resource
import shutil
import subprocess
import sys
import tarfile
import threading
import time

import numpy as np
import torch

from infra.aws_pg.jobs import realizednull_pg1 as P
from infra.aws_pg.jobs.gutenberg_gb1 import WIKITEXT_REV
from infra.pg import copernicus_core as C
from infra.pg import gutenberg_core as G
from infra.pg.conserved_core_probes import energy_matched_subspace

# ---- inputs: the mirror of §9.6 step 1 and the keys the cell reads under it (original relative keys) ----
BUCKET = "s3://pg-lineage-769063704696"
INPUTS = f"{BUCKET}/results/mdl/realizednull_q14_inputs"
INPUT_TABLE = pathlib.Path(__file__).with_name("realizednull_q14_inputs.json")
BANKS_DIR = "shackleton_census14b_banks/out/banks"
BANK = BANKS_DIR + "/q_proj__L{L}__{bank}32.npy"
TOPE = "shackleton_r_v1_14b/topenergy_14b.npz"
V1A_ROWS = "shackleton_r_v1a_14b/cells/v1a_full/v1abl_rows.json"
GB3_BANKS = "gutenberg_gb3_am1b_harden/cells/gb3_full/v1abl_gb_banks.npz"
GB3_WNLL = "gutenberg_gb3_am1b_harden/cells/gb3_full/v1abl_window_nll.json"
CORPUS, CORPUS_META = "out/corpus196.npy", "out/corpus196.json"       # N9's output (pg job R6a, §9.6 step 3)
CODE_ARCHIVE = pathlib.Path("/tmp/c.tgz")        # the onstart's copy of the shipped code (controller.py:233)
PIP_FREEZE = pathlib.Path("/app/pip_freeze.txt")  # written at image build (gen_dockerfile.py:78)
WORK = pathlib.Path("/workspace/q14_work")       # inputs and the Gram spill; not under out_dir, never uploaded
SCIPY_PIN = "scipy==1.17.1"                       # the AMI freeze (infra/vast/operations/2026-09-03_mendeleev_infra.md:90)

# ---- the model and the instrument (§3.8) ----
D_MODEL, N_LAYERS, K = 5120, 48, 32
WINDOW, F0, BATCH = 512, P.F0, 4
P0 = 1                           # position 0 is its own mean class; the X-rows keep it (§3.8)
MIN_HOST_RAM_GIB = 56.0          # peak about 41 GiB while construct_all runs: its V, M′, S′ in fp64 (30.4 GB at
                                 # d = 5120, 48 layers) and fp64 bases (<= 5.4 GB), plus V in fp32 and the fp32 bases
PROBE_EDIT = 64                  # §2.4
# §3.8: no image block. These two values make mask_probe_verdict's image label IMAGE-CAUSAL with its control
# fired, so the verdict is the text label alone. mask_probe.json says so beside them.
NO_IMAGE_BLOCK = {"r_prior": 0.0, "r_self": 1.0}

# ---- GB-3's banked numbers (studies/lineage/paligemma1-pi0/research/gutenberg_gb3_verdict.json) ----
GB3_DNLL_CORE = 0.02412969044657522          # → dnll.core32[0]
GB3_E_OVER_C = 5.516048806709715             # → E_over_C[0]
G14_TOL = 1e-3                               # → reproduction_gate.tol (the vast path, §6.1)
G14_DNLL_TOL = (0.002, 0.02)                 # band ΔNLL within max(0.002, 2%) of dnll.core32[0]
G14_EC_TOL = 0.05                            # E/C within 5% of E_over_C[0]
G14_ARMS = {"none": "none", "band": "core32", "anti32": "anti32", "haar_s0": "haar32_s0", "haar_s1": "haar32_s1",
            "haar_s2": "haar32_s2", "bottomsigma32": "bottomsigma32", "topsigma32": "topsigma32",
            "topenergy32": "topenergy32", "topenergy128": "topenergy128"}
GB3_EM = {3: "energymatched32_s3", 4: "energymatched32_s4", 5: "energymatched32_s5"}
GB3_SPEC = {3: "specmatched32_s3", 4: "specmatched32_s4", 5: "specmatched32_s5"}

# ---- arms (§3.2 minus A8 and K1, §3.8) ----
FAMS = ("rt", "rthalf", "rt2x", "emt")
DRAWS = {"em": 40, "spec": 5, "haar": 5, "rt": 40, "emt": 40, "dose": 3, "zt": 5, "mean_rt": 5, "mean_spec": 3,
         "mean_haar": 3}
SMOKE_DRAWS = {"em": 6, "spec": 5, "haar": 4, "rt": 2, "emt": 2, "dose": 1, "zt": 1, "mean_rt": 1, "mean_spec": 1,
               "mean_haar": 1}
PRE = ("band", "anti32", "haar_s0", "haar_s1", "haar_s2", "bottomsigma32", "topsigma32", "topenergy32",
       "topenergy128") + tuple(f"em_s{s}" for s in GB3_EM)       # G14-PRE's arms, run right after `none`
R_ONLY = ("topenergy128",)       # a reproduction arm only: scored on windows 0-97


def _n1():
    """N1, imported after ensure_scipy(): realizednull_core imports scipy at module level."""
    from infra.pg import realizednull_core
    return realizednull_core


def ensure_scipy():
    """N1 needs scipy, and the image's pip list (gen_dockerfile.py:22-23) does not name it. When it is absent, the AMI
    freeze's pin is installed without dependencies, so numpy is not touched mid-process. Returns (version, whether it
    was installed here); env.json records both."""
    installed = importlib.util.find_spec("scipy") is None
    if installed:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", SCIPY_PIN],
                       check=True)
    import scipy
    return scipy.__version__, installed


# ===========================================================================
# Cell parameters, slots, roster, batches
# ===========================================================================

def cell_args(cell):
    """The cell's parameters (N8): smoke, n_windows, n_construct, model_dir. Draw counts are §3.2's; the smoke's are
    SMOKE_DRAWS (windows 0-7, batched [0-3], [4-7] as GB-3, §0.4 U5)."""
    smoke = cell["smoke"]
    assert isinstance(smoke, bool), smoke
    a = {"smoke": smoke, "n_windows": int(cell["n_windows"]), "n_construct": int(cell["n_construct"]),
         "model_dir": str(cell["model_dir"]), **(SMOKE_DRAWS if smoke else DRAWS)}
    if smoke:
        assert a["n_windows"] < F0, a
    else:
        assert F0 < a["n_windows"] <= 2 * F0, a
    assert 0 < a["n_construct"] < min(a["n_windows"], F0), a
    assert a["em"] > max(GB3_EM) and a["haar"] >= 3, "G14-PRE needs EM s3-s5 and Haar s0-s2"
    assert a["zt"] <= min(a["em"], a["rt"]) and a["mean_rt"] <= a["rt"]
    assert a["mean_spec"] <= a["spec"] and a["mean_haar"] <= a["haar"]
    return a


def slots(a):
    """Seed slots per family (§3.2). em_mean shares em's bases; the X- and mean rows take the first slots."""
    r = lambda n: list(range(n))
    return {"em": r(a["em"]), "em_mean": r(a["em"]), "spec": r(a["spec"]), "haar": r(a["haar"]), "rt": r(a["rt"]),
            "rthalf": r(a["dose"]), "rt2x": r(a["dose"]), "emt": r(a["emt"]), "em_zt": r(a["zt"]),
            "rt_zt": r(a["zt"]), "rt_mean": r(a["mean_rt"]), "spec_mean": r(a["mean_spec"]),
            "haar_mean": r(a["mean_haar"])}


def roster(a):
    """Ablation arms as (arm, op, basis, group): G14-PRE's arms ("pre"), then every other arm of §3.2 minus A8 and K1
    in §9.4's order ("rest"). `none` runs before these."""
    sl = slots(a)
    Z, M, ZT = "zero", "mean", "zero_text"
    pre = [(n, Z, n, "pre") for n in PRE]
    rest = [("ident_zero", Z, "ident"), ("band_dup", Z, "band")]
    rest += [(f"haar_s{s}", Z, f"haar_s{s}") for s in sl["haar"] if s >= 3]
    rest += [(f"em_s{s}", Z, f"em_s{s}") for s in sl["em"] if s not in GB3_EM]
    rest += [(f"spec_s{s}", Z, f"spec_s{s}") for s in sl["spec"]]
    for fam in FAMS:
        rest += [(f"{fam}_s{s}", Z, f"{fam}_s{s}") for s in sl[fam]]
    rest += [("band_zt", ZT, "band")]
    rest += [(f"em_zt_s{s}", ZT, f"em_s{s}") for s in sl["em_zt"]]
    rest += [(f"rt_zt_s{s}", ZT, f"rt_s{s}") for s in sl["rt_zt"]]
    rest += [("ident_mean", M, "ident"), ("band_mean", M, "band")]
    for fam, src in (("em_mean", "em"), ("rt_mean", "rt"), ("spec_mean", "spec"), ("haar_mean", "haar")):
        rest += [(f"{fam}_s{s}", M, f"{src}_s{s}") for s in sl[fam]]
    rest += [("topenergy32_mean", M, "topenergy32"), ("topsigma32_mean", M, "topsigma32")]
    return pre + [(n, op, b, "rest") for n, op, b in rest]


def batches(windows, size=BATCH):
    """GB-3's batches of 4 consecutive windows (v1_ablation.py:175), cut separately below and above window 98, so
    windows 0-97 batch exactly as GB-3's and no batch mixes open and sealed windows."""
    ws = list(windows)
    out = []
    for part in ([w for w in ws if w < F0], [w for w in ws if w >= F0]):
        out += [part[i:i + size] for i in range(0, len(part), size)]
    return out


# ===========================================================================
# Inputs, corpus, provenance, host
# ===========================================================================

def array_sha256(x):
    """sha256 of an array's bytes, C order: prepare_corpus.py:46's corpus hash and N3's banks_manifest entries."""
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def input_table():
    return json.loads(INPUT_TABLE.read_text())["sha256"]


def verify_inputs(root, table):
    """Every input under root (at its original relative key) has the sha256 recorded from its original prefix."""
    got = {}
    for rel, want in table.items():
        h = P.sha256_file(root / rel)
        assert h == want, f"input {rel}: sha256 {h} differs from {want} recorded at results/{rel} " \
                          "(stale or partial mirror copy)"
        got[rel] = h
    return got


def fetch_inputs(root):
    """Copy every input from the mirror: the census bank directory in one recursive copy, then each other key and
    the corpus; then verify the recorded hashes."""
    def cp(*args):
        subprocess.run(["aws", "s3", "cp", *args, "--only-show-errors"], check=True)
    cp("--recursive", f"{INPUTS}/{BANKS_DIR}/", str(root / BANKS_DIR))
    for rel in (TOPE, V1A_ROWS, GB3_BANKS, GB3_WNLL, CORPUS, CORPUS_META):
        cp(f"{INPUTS}/{rel}", str(root / rel))
    return verify_inputs(root, input_table())


def check_corpus(arr, meta, file_sha256, baked):
    """N9's corpus: the file hash its sidecar records, shape (196, 512) int64, and windows 0-97 equal to the 98x512
    corpus baked into shackleton:v1-14b (prepare_corpus.py:19, 38-42), the windows GB-3 scored."""
    assert file_sha256 == meta["file_sha256"], (file_sha256, meta["file_sha256"])
    assert arr.dtype == np.int64 and arr.shape == tuple(meta["shape"]) == (2 * F0, WINDOW), (arr.dtype, arr.shape)
    assert baked.shape == (F0, WINDOW) and np.array_equal(arr[:F0], baked), "windows 0-97 differ from the baked corpus"
    return {"file_sha256": file_sha256, "shape": list(arr.shape), "first98_equal_to_baked": True,
            "tokenizer": meta["tokenizer"], "dataset": meta["dataset"]}


def code_provenance(path):
    """The commit the controller shipped: controller.py ship() archives `git archive HEAD infra configs scripts`,
    the onstart fetches it to /tmp/c.tgz (controller.py:233), and git writes the commit id into the archive's pax
    header. N4's stamp is not committed, so it is not in the shipped tree and N1's provenance() has nothing to read."""
    with tarfile.open(path, "r:gz") as t:
        sha = t.pax_headers.get("comment")
    assert sha and len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), f"{path}: no commit id ({sha!r})"
    return {"program": "realizednull", "commit_sha": sha, "source": f"{path} pax header (controller.py ship())",
            "code_archive_sha256": P.sha256_file(path)}


def host_ram_gib(meminfo="/proc/meminfo", limits=("/sys/fs/cgroup/memory.max",
                                                   "/sys/fs/cgroup/memory/memory.limit_in_bytes")):
    """The container's memory in GiB: the smallest of MemTotal and any cgroup limit that is set."""
    total = [int(ln.split()[1]) / 2 ** 20 for ln in pathlib.Path(meminfo).read_text().splitlines()
             if ln.startswith("MemTotal:")]
    assert len(total) == 1, meminfo
    vals = total + [int(v) / 2 ** 30 for p in limits if os.path.exists(p)
                    for v in [pathlib.Path(p).read_text().strip()] if v != "max"]
    return min(vals)


# ===========================================================================
# Hooks, forward, scoring (torch; tested on a toy model)
# ===========================================================================

def ablation_hook(U, op, mu, state, W=None, acc=None, L=None):
    """forward_pre_hook on q_proj for a batch of windows (§2.1 at 14B). zero: h32 − (h32 U) Uᵀ; zero_text: the same at
    positions >= P0 only (§3.8's X-rows); mean: h32 − ((h32 − μ_c(t)) U) Uᵀ, μ row 0 at position 0 and row 1 at
    positions 1-511. U (n, k) fp32; k = 0 is the identity. state["acc_rows"] (a bool per batch row) marks the rows
    that accumulate, per layer, the realized change Σ_t ||W (h32 − out)||² and Σ_t ||W h32||² over every position
    (§3.6 under ablation)."""
    def hook(_m, args):
        h = args[0]
        h32 = h.to(torch.float32)
        if op == "zero":
            out = h32 - (h32 @ U) @ U.T
        elif op == "zero_text":
            out = h32.clone()
            out[:, P0:] = h32[:, P0:] - (h32[:, P0:] @ U) @ U.T
        elif op == "mean":
            mu_full = torch.cat([mu[:P0], mu[P0:P0 + 1].expand(h32.shape[1] - P0, -1)], dim=0)
            out = h32 - ((h32 - mu_full) @ U) @ U.T
        else:
            raise ValueError(op)
        hb = out.to(h.dtype)
        rows = state.get("acc_rows")
        if rows is not None and bool(rows.any()):
            d = (h32 - hb.to(torch.float32))[rows]
            acc["num"][L] = acc["num"][L] + (d @ W.T).pow(2).sum().double()
            acc["den"][L] = acc["den"][L] + (h32[rows] @ W.T).pow(2).sum().double()
        return (hb,) + tuple(args[1:])
    return hook


def score_row(logits_row, ids_row):
    """GB-3's readout for one window (v1_ablation.py:180-183): NLL summed over the 511 next-token targets, plus the
    per-position log p(target), top-1 id and top-1 log-probability the samples need."""
    lp_all = torch.log_softmax(logits_row[:-1].to(torch.float32), dim=-1)
    tgt = ids_row[1:]
    lpg = lp_all.gather(-1, tgt[:, None])
    top_lp, top_id = lp_all.max(dim=-1)
    return {"nll": float(-lpg.sum()), "ntok": int(tgt.numel()), "lp": lpg[:, 0].cpu().numpy(),
            "top_id": top_id.cpu().numpy(), "top_lp": top_lp.cpu().numpy()}


class Forward:
    """GB-3's before-model forward (v1_ablation.py:173-186): batches of input_ids only, no attention_mask, no_grad."""

    def __init__(self, model, layers, corpus, dev, vocab):
        self.model, self.layers, self.corpus, self.dev, self.vocab = model, layers, corpus, dev, vocab
        self.fwd_s, self.n_fwd = 0.0, 0

    def scores(self, windows, hooks=(), state=None, acc_windows=()):
        handles = [m.register_forward_pre_hook(fn) for m, fn in hooks]
        out = {}
        t0 = time.time()
        with torch.no_grad():
            for b in batches(windows):
                ids = torch.as_tensor(self.corpus[b], device=self.dev)
                if state is not None:
                    state["acc_rows"] = torch.tensor([w in acc_windows for w in b], device=self.dev)
                res = self.model(input_ids=ids)
                logits = res.logits
                assert logits.shape[-1] == self.vocab, logits.shape
                for r, w in enumerate(b):
                    out[w] = score_row(logits[r], ids[r])
                del res, logits
        self.fwd_s += time.time() - t0
        self.n_fwd += len(out)
        for h in handles:
            h.remove()
        return out

    def plain(self, windows, hooks):
        """Forward windows for record-only hooks (Gram, verification); logits unused."""
        handles = [m.register_forward_pre_hook(fn) for m, fn in hooks]
        with torch.no_grad():
            for b in batches(windows):
                self.model(input_ids=torch.as_tensor(self.corpus[b], device=self.dev))
        for h in handles:
            h.remove()


def text_probe(fw, ids0, the_id, a_id, edit=PROBE_EDIT):
    """§2.4's text causality probe on window 0, batch 1 (§3.8: G-MASK is this probe alone): the last `edit` tokens
    replaced by " the" (" a" where the original is " the"); d_before and d_after are the max |Δ fp32 logit| at
    positions before, and at or after, the edit."""
    e = ids0.shape[0] - edit
    ids = ids0.clone()
    ids[e:] = torch.where(ids[e:] == the_id, torch.tensor(a_id), torch.tensor(the_id))
    with torch.no_grad():
        l0 = fw.model(input_ids=ids0[None].to(fw.dev)).logits[0].float()
        l1 = fw.model(input_ids=ids[None].to(fw.dev)).logits[0].float()
    d = (l1 - l0).abs().amax(dim=-1)
    return {"d_before": float(d[:e].max()), "d_after": float(d[e:].max()), "edit_start": e, **NO_IMAGE_BLOCK}


def probe_mode(probe):
    R = _n1()
    return R.mask_probe_verdict(probe["d_before"], probe["d_after"],  # N1: mask_probe_verdict
                                probe["r_prior"], probe["r_self"])


def gram_pass(fw, windows, class_sums):
    """Σ = Σ_t h_t h_tᵀ over every position (T = all, §3.8) per layer, fp32 on the device, one set at a time
    (§9.2 N7). With class_sums: the sums of h at position 0 and at positions 1-511, the two mean classes (§2.3)."""
    layers, dev = fw.layers, fw.dev
    n = layers[0].self_attn.q_proj.in_features
    Gm = [torch.zeros(n, n, dtype=torch.float32, device=dev) for _ in layers]
    s0 = [torch.zeros(n, dtype=torch.float32, device=dev) for _ in layers]
    s1 = [torch.zeros(n, dtype=torch.float32, device=dev) for _ in layers]

    def mk(L):
        def hook(_m, args):
            h = args[0].to(torch.float32)
            flat = h.reshape(-1, h.shape[-1])
            Gm[L] += flat.T @ flat
            if class_sums:
                s0[L] += h[:, :P0].sum(dim=(0, 1))
                s1[L] += h[:, P0:].sum(dim=(0, 1))
        return hook
    fw.plain(windows, [(layer.self_attn.q_proj, mk(L)) for L, layer in enumerate(layers)])
    sums = None
    if class_sums:
        sums = {"pos0": [x.double().cpu().numpy() for x in s0], "rest": [x.double().cpu().numpy() for x in s1],
                "n_windows": len(windows)}
    return Gm, sums


def verify_pass(fw, windows, entries, Ws32):
    """§2.2 verification through the hooks, independent of the Gram: per layer Σ_t ||W_q U Uᵀ h_t||² (or
    Σ_t ||Uᵀ h_t||² for the identity) over every position (T = all), fp32 per batch, summed in fp64. entries:
    P.verify_entries' [(key, U_by_layer, positions, use_W)] → {key: ndarray[n_layers]}."""
    layers, dev = fw.layers, fw.dev
    per_layer = [[] for _ in layers]
    for key, U, positions, use_W in entries:
        assert positions in ("T", "all"), positions
        for L in range(len(layers)):
            Ud = torch.as_tensor(np.ascontiguousarray(U[L]), dtype=torch.float32, device=dev)
            per_layer[L].append((key, Ud, (Ws32[L] @ Ud) if use_W else None))
    acc = {key: [0.0] * len(layers) for key, *_ in entries}

    def mk(L):
        def hook(_m, args):
            h = args[0].to(torch.float32)
            for key, Ud, WU in per_layer[L]:
                c = h @ Ud
                v = c @ WU.T if WU is not None else c
                acc[key][L] = acc[key][L] + v.pow(2).sum().double()
        return hook
    fw.plain(windows, [(layer.self_attn.q_proj, mk(L)) for L, layer in enumerate(layers)])
    return {key: np.array([float(x) for x in v]) for key, v in acc.items()}


# ===========================================================================
# Eigen-coordinates, means, fresh EM and spec draws
# ===========================================================================

def eigen_layer(S, W):
    """§2.2's eigen-coordinates at d = 5120 for one layer, on S's device: eigh of Σ in fp32 (v1_ablation.py:223-225:
    descending, eigenvalues clamped >= 0); V re-orthonormalized in fp64 (QR, column signs kept), so V Ue is
    orthonormal to N1's tolerances; M = W_qᵀ W_q in fp64 from the bf16 weight; M′ = Vᵀ M V and S′ = Vᵀ Σ V as fp32
    products, cast to fp64 and symmetrized (§2.2); denom = tr(M Σ) in fp64. T is every position, so Σ_S = Σ_all.
    Returns N1 eigencoords(Σ, M, Σ)'s keys. W may be the model's Parameter: it is read detached."""
    W = W.detach()
    mu, V = torch.linalg.eigh(S)
    mu = mu.flip(0).clamp_min(0.0)
    Q, Rq = torch.linalg.qr(V.flip(1).double())
    Q = Q * torch.sign(torch.diagonal(Rq))[None, :]
    W64 = W.double()
    M = W64.T @ W64
    V32 = Q.float()
    Mp = (V32.T @ M.float() @ V32).double()
    Sp = (V32.T @ S @ V32).double()

    def sym(X):
        return ((X + X.T) / 2.0).cpu().numpy()
    return {"V": Q.cpu().numpy(), "mu": mu.double().cpu().numpy(), "Mp": sym(Mp), "Sp": sym(Sp),
            "denom": float((M * S.double()).sum())}


def eigen_stage(Gm, Ws, band, tope32, spill):
    """eigen_layer per layer, then each Gram leaves the device: spilled to spill/sigmaC_L{L}.npy (fp32) for the
    arm_stats thread. Also per layer: the band's energy share on C (G-EMATCH's e_target) and the overlap of the 32
    largest Gram diagonal entries with the banked topenergy32 indices (recompute-and-compare only, §9.2)."""
    out = {"V": [], "mu": [], "Mp": [], "Sp": [], "denom": [], "e_target": [], "tope_overlap": []}
    for L in range(len(Gm)):
        ec = eigen_layer(Gm[L], Ws[L])
        S = Gm[L].cpu().numpy()
        Gm[L] = None
        np.save(spill / f"sigmaC_L{L}.npy", S)
        for key in ("V", "mu", "Mp", "Sp", "denom"):
            out[key].append(ec[key])
        S64 = S.astype(np.float64)
        out["e_target"].append(float(np.sum((S64 @ band[L]) * band[L]) / np.trace(S64)))
        idx = np.argsort(np.diag(S))[::-1][:K]
        out["tope_overlap"].append(len(set(idx.tolist()) & set(np.asarray(tope32[L]).tolist())))
    return out


def class_means_q14(sums, S_len):
    """§2.3 at 14B (§3.8): two mean classes per layer, position 0 and positions 1-511, through N1's class_means.
    Returns the (2, n) rows per layer in P.class_rows' layout (position 0, then the rest) and the class counts."""
    R = _n1()
    nw = sums["n_windows"]
    counts = {0: nw * P0, "text": nw * (S_len - P0)}
    mu = [P.class_rows(R.class_means({0: s0, "text": s1}, counts))  # N1: class_means
          for s0, s1 in zip(sums["pos0"], sums["rest"])]
    return mu, counts


def stats_means(mu, counts, sums, S_len):
    """arm_stats' means on C, one single-layer dict per layer: class rows, positions per class, the rest class's row,
    Σ_{t∈T} h_t and |T| with T every position."""
    cnt = np.array([counts[0], counts["text"]], dtype=np.float64)
    return [{"mu": [mu[L]], "counts": [cnt], "text": [P0], "T_sum": [sums["pos0"][L] + sums["rest"][L]],
             "T_count": [sums["n_windows"] * S_len]} for L in range(len(mu))]


def ematch_gate(e_target, e_real, d):
    """G-EMATCH at d = 5120: realized energy within N1's 10% of target at every layer whose target is not below k/d."""
    floor = K / d
    tol = _n1().EMATCH_TOL
    bad = [L for L, (t, e) in enumerate(zip(e_target, e_real)) if t >= floor and abs(e - t) > tol * t]
    return {"failing_layers": bad, "floor_k_over_d": floor, "pass": not bad}


def em_fresh(V, mu, e_tgt, s):
    """A fresh energy-matched draw at slot s (v1_ablation.py:231-233): energy_matched_subspace at
    seed 70_000 + 1000·s + L. Returns per-layer bases and realized energies."""
    Qs, er = [], []
    for L in range(len(V)):
        Q, e = energy_matched_subspace(V[L], mu[L], e_tgt[L], K, seed=70_000 + 1000 * s + L,
                                       tol=max(0.02 * e_tgt[L], 1e-6), max_resample=200)
        Qs.append(np.ascontiguousarray(Q))
        er.append(e)
    return Qs, er


def spec_fresh(V, band, s):
    """A fresh spectrum-matched draw at slot s (v1_ablation.py:242-247): QR(V (A∘S)), A = Vᵀ U_band, random signs
    from default_rng(60_000 + 1000·s + L)."""
    Us = []
    for L in range(len(V)):
        A = V[L].T @ band[L]
        srng = np.random.default_rng(60_000 + 1000 * s + L)
        Q, _ = np.linalg.qr(V[L] @ (A * srng.choice([-1.0, 1.0], size=A.shape)))
        Us.append(np.ascontiguousarray(Q[:, :K]))
    return Us


# ===========================================================================
# arm_stats (§3.6) and G-GEN's quantities on D: one layer at a time, in a daemon thread
# ===========================================================================

def layer_stats(L, U_by_basis, S_C, S_D, V, W, means_L, gen_keys):
    """Layer L's §3.6 columns for every basis on C and on D (N1 arm_stats on one-layer lists; T is every position, so
    grams "all" and "T" are one matrix), and layer L's quantity on D for G-GEN (N1 realized_quantity) per gen_keys
    {key: (basis, "N_T" | "E_T")}. S_C, S_D, V, W in fp64."""
    R = _n1()
    M = W.T @ W
    eye = np.eye(W.shape[1])

    def grams(S):
        den = float((M * S).sum())
        return {"all": [S], "T": [S], "V": [V], "denom_all": [den], "denom_T": [den]}
    gC, gD = grams(S_C), grams(S_D)
    stats = {"C": {b: R.arm_stats([U[L]], gC, [W], means_L) for b, U in U_by_basis.items()},  # N1: arm_stats
             "D": {b: R.arm_stats([U[L]], gD, [W], None) for b, U in U_by_basis.items()}}  # N1: arm_stats
    gen = {key: R.realized_quantity(U_by_basis[b][L], M if q == "N_T" else eye, S_D)  # N1: realized_quantity
           for key, (b, q) in gen_keys.items()}
    return stats, gen


def stats_all_layers(n_layers, load_layer, U_by_basis, means, gen_keys):
    """layer_stats for every layer; load_layer(L) → (S_C, S_D, V, W) in fp64, built once per layer."""
    return [layer_stats(L, U_by_basis, *load_layer(L), means[L], gen_keys) for L in range(n_layers)]


def assemble_stats(per_layer):
    """layer_stats' results → N1 arm_stats' layout over all layers: each column concatenated in layer order and each
    `<column>_sum` recomputed as N1 does (np.sum over the layers); G-GEN quantities as per-layer lists."""
    stats = {}
    for split in ("C", "D"):
        stats[split] = {}
        for b, first in per_layer[0][0][split].items():
            cols = {k: [st[split][b][k][0] for st, _ in per_layer] for k in first if not k.endswith("_sum")}
            cols.update({k: float(np.sum(cols[k[:-4]])) for k in first if k.endswith("_sum")})
            stats[split][b] = cols
    gen = {key: [g[key] for _, g in per_layer] for key in per_layer[0][1]}
    return stats, gen


def gen_rows(names, fam, gen):
    """job_metrics → gen_D[fam] (G-GEN) from the per-layer quantities on D, as N3's gen_rows: N_sum = Σ_L N_L(draw),
    target_sum = dose × Σ_L N_L(band); None for a draw with no basis."""
    f = P.FAMILIES[fam]
    band = float(sum(gen["band|" + f["q"]]))
    return [{"N_sum": float(sum(gen[n])), "target_sum": f["dose"] * band} if n in gen else None for n in names]


def start_thread(name, fn, *args):
    """fn(*args) in a daemon thread (arm_stats and G-GEN) while the GPU scores arms. The
    arguments are bound here, so the caller may drop its own names for them. A crash of the cell is not held up by
    the thread, and the thread's traceback reaches job.log through the threading module's default hook for an
    uncaught error. thread_result reads the result."""
    box = {}
    t = threading.Thread(target=lambda: box.update(result=fn(*args)), name=name, daemon=True)
    t.start()
    return t, box


def thread_result(t, box, wait):
    """The thread's result once it has finished, None while it runs; asserts at once if it died."""
    if wait:
        t.join()
    if t.is_alive():
        return None
    assert "result" in box, f"thread {t.name} died; its traceback is in job.log"
    return box["result"]


def needs_construction(basis):
    """Whether a basis is an RT-family or EM_T draw (§3.2 A7, A9, A10, A15), built by construct_all."""
    return basis.rsplit("_s", 1)[0] in FAMS


def construct_all(fam_in, band, sl, kw=P.CONSTRUCT_KW):
    """§2.2 for every RT-family and EM_T draw (N1 via N3's helpers): the band's per-layer targets, one grid scan per
    layer and quantity (seed 190_000 + L), and construct_draw at §3.2's base seeds with up to three redraws. kw: N3's
    CONSTRUCT_KW (§2.2's k, tolerances, grid and resampling sizes). Returns the targets, grids, records, the matched
    draws' model-space bases (fp64), and seconds per grid and in all."""
    t0 = time.time()
    tgt = P.band_targets(fam_in, band)  # N1 via N3: realized_quantity
    grids = P.quantity_grids(fam_in, sorted(fam_in), kw)  # N1 via N3: reach_grid
    n_layers = len(band)
    out = {"tgt": tgt, "grids": grids, "t_grid": (time.time() - t0) / (len(fam_in) * n_layers),
           "cons": {}, "fam_draws": {}, "bases": {}, "t_con_s": 0.0}
    for fam in FAMS:
        recs, bases, t = P.construct_family(fam, sl[fam], fam_in, tgt, grids,  # N1 via N3: construct_draw
                                            {**kw, "max_redraws": 3})
        out["cons"].update(recs)
        out["fam_draws"][fam] = list(recs)
        out["bases"].update(bases)
        out["t_con_s"] += t
    return out


# ===========================================================================
# G14-PRE (§6.1)
# ===========================================================================

def g14pre_references(R_w, v1a_rows, gb3_wnll):
    """G14-PRE's references on the windows R_w this cell scores. On windows 0-97 (the full cell): ppl from
    shackleton_r_v1a_14b v1a_full v1abl_rows.json, and band ΔNLL and E/C at GB-3's banked values, which GB-3's own
    per-window NLL must reproduce (an input check). On fewer windows (the smoke): every reference from GB-3's
    per-window NLL on the same windows (gb3_full v1abl_window_nll.json, gutenberg_core.build_gb3_metrics' pooling)."""
    pw = gb3_wnll["per_window"]
    assert gb3_wnll["n_abl_windows"] == F0, gb3_wnll["n_abl_windows"]
    none = pw["none"]

    def dn(arm):
        return G.dnll_per_window([pw[arm]["nll"][w] for w in R_w], [none["nll"][w] for w in R_w],
                                 [none["ntok"][w] for w in R_w])
    core = dn("core32")
    em = np.mean([dn(a) for a in GB3_EM.values()], axis=0)
    dnll, ec = float(core.mean()), float(em.mean() / core.mean())
    if list(R_w) == list(range(F0)):
        assert v1a_rows["n_abl_windows"] == F0, v1a_rows["n_abl_windows"]
        ppl = {r["arm"]: r["ppl"] for r in v1a_rows["arms"]}
        assert math.isclose(dnll, GB3_DNLL_CORE, rel_tol=1e-9) and math.isclose(ec, GB3_E_OVER_C, rel_tol=1e-9), \
            ("GB-3's per-window NLL does not give its banked dnll.core32[0] and E_over_C[0]", dnll, ec)
        return {"ppl": ppl, "dnll_core32": GB3_DNLL_CORE, "E_over_C": GB3_E_OVER_C,
                "source": "v1a_full v1abl_rows.json; gutenberg_gb3_verdict.json dnll.core32[0], E_over_C[0]"}
    ppl = {a: math.exp(sum(pw[a]["nll"][w] for w in R_w) / sum(pw[a]["ntok"][w] for w in R_w))
           for a in G14_ARMS.values()}
    return {"ppl": ppl, "dnll_core32": dnll, "E_over_C": ec, "source": "gb3_full v1abl_window_nll.json, same windows"}


def run_e_over_c(nll_open, R_w):
    """E/C as GB-3 pooled it: the mean over windows of the mean per-window ΔNLL of EM s3-s5 over the band's."""
    none = [nll_open["none"][w] for w in R_w]
    ntok = [WINDOW - 1] * len(R_w)
    core = G.dnll_per_window([nll_open["band"][w] for w in R_w], none, ntok)
    em = np.mean([G.dnll_per_window([nll_open[f"em_s{s}"][w] for w in R_w], none, ntok) for s in GB3_EM], axis=0)
    return float(em.mean() / core.mean())


def g14pre_gate(run_ppl, run_dnll_band, run_ec, ref, R_w):
    """G14-PRE (§6.1, the vast path): every reproduction arm's ppl within rel 1e-3 of GB-3's banked row, the band's
    ΔNLL within max(0.002, 2%) of GB-3's, and E/C with GB-3's banked EM s3-s5 bases within 5% of GB-3's."""
    wk = lambda name: P.wkey(name, R_w)
    rows = {}
    for arm, gb3 in G14_ARMS.items():
        rel = abs(run_ppl[arm] / ref["ppl"][gb3] - 1.0)
        rows[arm] = {wk("ppl"): run_ppl[arm], "gb3_arm": gb3, "gb3_ppl": ref["ppl"][gb3], "rel": rel,
                     "pass": rel <= G14_TOL}
    tol = max(G14_DNLL_TOL[0], G14_DNLL_TOL[1] * ref["dnll_core32"])
    band = {wk("dnll"): run_dnll_band, "gb3": ref["dnll_core32"], "tol": tol,
            "pass": abs(run_dnll_band - ref["dnll_core32"]) <= tol}
    ec_rel = abs(run_ec / ref["E_over_C"] - 1.0)
    ec = {wk("E_over_C"): run_ec, "gb3": ref["E_over_C"], "rel": ec_rel, "pass": ec_rel <= G14_EC_TOL}
    ok = all(r["pass"] for r in rows.values()) and band["pass"] and ec["pass"]
    return {"rows": rows, "band_dnll": band, "E_over_C": ec, "ppl_rel_tol": G14_TOL,
            "references": ref["source"], "pass": ok}


# ===========================================================================
# Bases, samples, per-arm writes, documents
# ===========================================================================

def load_bases(root, Ws32, d, NL, sl):
    """GB-3's bases (v1_ablation.py:102-129) and GB-3's banked EM and spec slots, each per layer as fp32 (n, 32):
    band and anti32 (bank, QR), Haar s0-s2 (default_rng(1337)) and fresh Haar s>=3 (default_rng(1_000_000 + 1000·s
    + L), §3.2), topenergy32 and topenergy128 (coordinate_bank), top/bottom-σ (fp32 SVD on the GPU), and "ident"
    (0 columns). Returns (bases, sources, the band in fp64)."""
    B = {"ident": [np.zeros((d, 0), dtype=np.float32)] * NL}
    band64 = []
    tope = np.load(root / TOPE)
    gb3 = np.load(root / GB3_BANKS)
    rng = np.random.default_rng(1337)
    banked = [(f"{fam}_s{s}", g) for fam, table in (("em", GB3_EM), ("spec", GB3_SPEC))
              for s, g in table.items() if s in sl[fam]]
    for L in range(NL):
        for bank, name in (("core", "band"), ("anti", "anti32")):
            raw = np.load(root / BANK.format(L=L, bank=bank))
            assert raw.shape == (d, K), (bank, L, raw.shape)
            Q = np.linalg.qr(np.asarray(raw, dtype=np.float64))[0]
            if name == "band":
                band64.append(Q)
            B.setdefault(name, []).append(Q.astype(np.float32))
        for s in range(3):
            B.setdefault(f"haar_s{s}", []).append(G.haar_frame(d, K, rng).astype(np.float32))
        for k in (32, 128):
            B.setdefault(f"topenergy{k}", []).append(
                G.coordinate_bank(d, tope[f"L{L:02d}_top{k}"]).astype(np.float32))
        for name, gname in banked:
            Q = np.asarray(gb3[f"{gname}/{L}"], dtype=np.float64)
            C.orthonormal_or_die(Q, f"{gname} L{L}", tol=1e-5)          # banked fp32
            B.setdefault(name, []).append(Q.astype(np.float32))
        _, _, Vh = torch.linalg.svd(Ws32[L], full_matrices=False)
        B.setdefault("bottomsigma32", []).append(np.ascontiguousarray(Vh[-K:, :].T.cpu().numpy()))
        B.setdefault("topsigma32", []).append(np.ascontiguousarray(Vh[:K, :].T.cpu().numpy()))
        del Vh
    for s in [s for s in sl["haar"] if s >= 3]:
        B[f"haar_s{s}"] = [G.haar_frame(d, K, np.random.default_rng(1_000_000 + 1000 * s + L)).astype(np.float32)
                           for L in range(NL)]
    src = {"ident": ("U with 0 columns", None),
           "band": ("census bank q_proj__L{L}__core32.npy, QR (v1_ablation.py:105-108)", None),
           "anti32": ("census bank q_proj__L{L}__anti32.npy, QR", None),
           "topenergy32": ("topenergy_14b.npz L{L:02d}_top32 via coordinate_bank", None),
           "topenergy128": ("topenergy_14b.npz L{L:02d}_top128 via coordinate_bank", None),
           "topsigma32": ("top-32 right singular vectors of W_q (fp32 SVD on the GPU)", None),
           "bottomsigma32": ("bottom-32 right singular vectors of W_q (fp32 SVD on the GPU)", None)}
    for s in range(3):
        src[f"haar_s{s}"] = ("default_rng(1337), 3 frames per layer in layer order (v1_ablation.py:103)", 1337)
    for s in [s for s in sl["haar"] if s >= 3]:
        src[f"haar_s{s}"] = ("haar_frame, default_rng(1_000_000 + 1000*s + L)", 1_000_000 + 1000 * s)
    for name, gname in banked:
        base = 70_000 if name.startswith("em") else 60_000
        src[name] = (f"GB-3 v1abl_gb_banks.npz {gname}/{{L}}", base + 1000 * int(name.rsplit("_s", 1)[1]))
    return B, src, band64


def sample_window(w, ids, tok):
    """sample_record's window (N1, §10.1) for a text-only window: the 512 tokens as the model saw them, decoded;
    the 511 targets ids[1:]."""
    return {"index": w, "n_text": int(len(ids)), "target_ids": ids[1:],
            "target_strs": tok.convert_ids_to_tokens(ids[1:].tolist()), "text_seen": tok.decode(ids.tolist())}


def sample_payload(header, sel, scores, none, corpus, tok):
    """One §10.1 sample file: the header, the max-Y, min-Y and common windows (N1 sample_record each, with its
    role), and one right and one wrong example (N3's sample_examples at position offset 0)."""
    R = _n1()

    def top(s):
        return (s["top_id"], tok.convert_ids_to_tokens(s["top_id"].tolist()), np.exp(s["top_lp"]))
    recs = []
    for role in ("max", "min", "common"):
        w = sel[role]
        rec = R.sample_record(sample_window(w, corpus[w], tok),  # N1: sample_record
                              none[w]["lp"], scores[w]["lp"], top(none[w]), top(scores[w]))
        recs.append({"role": role, **rec})
    order = [sel["common"], sel["max"], sel["min"]]
    ex = P.sample_examples(order, {w: corpus[w][1:] for w in order}, none, scores, 0,
                           lambda i: tok.convert_ids_to_tokens(int(i)))
    return {**header, "windows": recs, "examples": ex}


def record_arm(rec, arm, op, sc, header, corpus, tok):
    """One arm's per-arm saves (§10.1): per-window NLL routed by window range (windows >= 98 to sealed/), the R/D
    summary, the open (D) and sealed (F) sample files, coverage and progress."""
    rec.nll(arm, op, sc)
    sel, _ = rec.select(sc)
    pay = lambda s: None if s is None else sample_payload(header, s, sc, rec.none, corpus, tok)
    rec.write_samples(arm, pay(sel["open"]), pay(sel["sealed"]))
    rec.done(arm)


def env_gate(env):
    """G-ENV for session 2: every N1 ENV_KEYS entry present, and GB-3's attention, sdpa (v1_ablation.py:96)."""
    missing = [k for k in _n1().ENV_KEYS if not env.get(k)]
    return {"missing": missing, "_attn_implementation": env.get("_attn_implementation"),
            "pass": not missing and env.get("_attn_implementation") == "sdpa"}


def metrics_doc(rec, gates, info, jm):
    """metrics.json as N3 writes it (gates, R/D summaries, readout_validity, job_metrics), named for this job."""
    return {**P.metrics_doc(rec, gates, info, jm), "job": "realizednull_q14"}


def family_records(cons, fam_draws, gates):
    for g in ("G-REACH", "G-PIVOT", "G-EXHAUST"):
        gates[g] = {}
    for fam, names in fam_draws.items():
        if names:
            for g, v in P.family_gates(cons, names).items():
                gates[g][fam] = v


# ===========================================================================
# The cell
# ===========================================================================

def run(cell, out_dir):
    a = cell_args(cell)
    cid = cell["id"]
    sp = P.splits(a["n_windows"], a["n_construct"])
    sl = slots(a)
    arms = roster(a)
    sink = P.Sink(out_dir, "smoke" if a["smoke"] else "discovery")
    sink.event("start", cell=cid, rental=os.environ.get("RENTAL"))
    t_stage, stages = time.time(), {}

    def lap(name):
        nonlocal t_stage
        stages[name] = stages.get(name, 0.0) + time.time() - t_stage
        t_stage = time.time()

    ram = host_ram_gib()
    assert ram >= MIN_HOST_RAM_GIB, f"host memory {ram:.1f} GiB < {MIN_HOST_RAM_GIB} GiB (HANDOFF_q14.md, offer)"
    scipy_version, scipy_installed = ensure_scipy()
    R = _n1()
    prov = code_provenance(CODE_ARCHIVE)
    work = WORK / cid
    root = work / "inputs"
    spill = work / "spill"
    spill.mkdir(parents=True, exist_ok=True)
    inputs = fetch_inputs(root)
    lap("inputs")

    # ---- corpus (§3.8, §9.6 step 3) and the model directory (§3.8 prerequisite 3) ----
    model_dir = pathlib.Path(a["model_dir"])
    cfg = json.loads((model_dir / "config.json").read_text())
    assert (cfg["hidden_size"], cfg["num_hidden_layers"]) == (D_MODEL, N_LAYERS), cfg
    corpus_file = np.load(root / CORPUS)
    cinfo = check_corpus(corpus_file, json.loads((root / CORPUS_META).read_text()), P.sha256_file(root / CORPUS),
                         np.load(model_dir / "shackleton_corpus.npy"))
    corpus = np.ascontiguousarray(corpus_file[:a["n_windows"]])
    lap("corpus")

    # ---- model (GB-3's load, v1_ablation.py:94-97), env (G-ENV) ----
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    sink.event("model_load")
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16, device_map=dev,
                                                 attn_implementation="sdpa")
    model.eval()
    sink.event("model_loaded")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    layers = model.model.layers
    NL = len(layers)
    d = layers[0].self_attn.q_proj.in_features
    assert (NL, d) == (N_LAYERS, D_MODEL), (NL, d)
    fw = Forward(model, layers, corpus, dev, cfg["vocab_size"])
    Ws32 = [layer.self_attn.q_proj.weight.detach().to(torch.float32) for layer in layers]
    env = {"transformers": transformers.__version__, "torch": torch.__version__, "numpy": np.__version__,
           "scipy": scipy_version, "scipy_installed_at_start": scipy_installed, "cuda": torch.version.cuda,
           "gpu_name": torch.cuda.get_device_name(0), "instance_type": "vast.ai",
           "vast_container": os.environ.get("CONTAINER_ID") or os.environ.get("VAST_CONTAINERLABEL"),
           "rental": os.environ.get("RENTAL"), "_attn_implementation": model.config._attn_implementation,
           "mask": "native", "forced_masks": "off", "args": a, "cell": cid, "provenance": prov,
           "family": "q14", "model_dir": str(model_dir), "n_layers": NL, "d_model": d,
           "config_sha256": P.sha256_file(model_dir / "config.json"),
           "image_pip_freeze_sha256": P.sha256_file(PIP_FREEZE), "inputs_sha256": inputs, "corpus": cinfo,
           "inputs_base": INPUTS, "wikitext_rev": WIKITEXT_REV, "host_ram_gib": ram}
    gates = {"G-ENV": env_gate(env)}
    jm = {"env": {k: env[k] for k in R.ENV_KEYS + ("_attn_implementation",)}, "mask": "native"}
    sink.json("env.json", env)
    lap("model")

    B, src, band64 = load_bases(root, Ws32, d, NL, sl)
    tope = np.load(root / TOPE)
    tope32 = [tope[f"L{L:02d}_top32"] for L in range(NL)]
    lap("bases")

    # ---- text causality probe (window 0), §2.4 / §3.8 ----
    sink.event("probes")
    the_id, a_id = tok.convert_tokens_to_ids("Ġthe"), tok.convert_tokens_to_ids("Ġa")
    assert tok.decode([the_id]) == " the" and tok.decode([a_id]) == " a", (the_id, a_id)
    probe = text_probe(fw, torch.as_tensor(corpus[0]), the_id, a_id)
    probe["mode"] = probe_mode(probe)
    sink.json("mask_probe.json", {"native": probe, "forced_masks": "off",
                                  "image_probe": "none: text-only model (§3.8); r_prior and r_self are "
                                                 "NO_IMAGE_BLOCK, so the mode is the text label alone"})
    jm["mask_probe"] = {"native": {k: probe[k] for k in P.PROBE_KEYS}, "forced_masks": "off", "forced": {}}
    gates["G-MASK"] = {"native_mode": probe["mode"], "pass": probe["mode"] in P.FORCED, "text_only": True,
                       "forced_masks": "off"}
    lap("probes")

    rec = P.Recorder(sink, sp, ["none"] + [x[0] for x in arms])
    rec.stage("corpus")
    if sp["F"]:
        sink.json("sealed/corpus_f.json", {"array_sha256": array_sha256(corpus[F0:])})
    info = {"smoke": a["smoke"], "cell": cid, "session": 2, "mask": "native", "forced_masks": "off",
            "topup_families": "", "slot_offset": 0, "n_windows": a["n_windows"], "n_construct": a["n_construct"],
            "corpus": {P.wkey("array_sha256", sp["R"]): array_sha256(corpus[:len(sp["R"])]),
                       "recipe": f"wikitext-103-raw-v1:test@{WIKITEXT_REV}, Qwen2.5-14B tokenizer, "
                                 f"prepare_corpus.py:38-42 (N9)", "first98_equal_to_baked": True},
            "provenance": prov, "stopped": None}
    hashes = {}
    ast = {P.wkey("under_ablation", sp["D"]): {}, "per_basis": None}
    mu_dev = None
    t_con = t_grid = stats_s = None

    def header(arm, op, basis):
        source, seed = src.get(basis, (None, None))
        if basis is not None and basis not in hashes:
            hashes[basis] = C.bank_hash(np.concatenate(B[basis], axis=1))
        after = {"none": "the before-model",
                 "zero": f"the before-model + forward_pre_hook on self_attn.q_proj of all {NL} layers: "
                         "h32 - (h32 @ U_L) @ U_L.T, cast to bf16",
                 "zero_text": "as zero, at positions 1-511 only (position 0 unchanged)",
                 "mean": f"the before-model + forward_pre_hook on self_attn.q_proj of all {NL} layers: "
                         "h32 - ((h32 - mu_{L,c(t)}) @ U_L) @ U_L.T, classes position 0 and positions 1-511, "
                         "cast to bf16"}[op]
        return {"arm": arm, "op": op, "basis": basis, "basis_source": source, "seed": seed,
                "basis_sha256": None if basis is None else hashes[basis],
                "before_model": f"Qwen2.5-14B ({model_dir}, baked in shackleton:v1-14b), AutoModelForCausalLM, bf16, "
                                "sdpa, input_ids of 512-token windows in batches of 4, no attention_mask, no_grad, "
                                "no hooks", "after_model": after}

    def write_compute():
        sink.json("compute.json", {
            "seconds_per_stage": stages, "s_fwd": fw.fwd_s / max(fw.n_fwd, 1),
            "n_window_forwards_scored": fw.n_fwd, "t_con": t_con, "t_grid_per_layer": t_grid,
            "stats_thread_s": stats_s, "host_ram_gib": ram,
            "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            "peak_gpu_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30})

    def run_arm(arm, op, basis, i, n):
        sink.event("arm", arm=arm, i=i, n=n)
        assert op != "mean" or mu_dev is not None, arm
        U = [torch.as_tensor(np.ascontiguousarray(B[basis][L]), dtype=torch.float32, device=dev) for L in range(NL)]
        state, acc = {}, {"num": [0.0] * NL, "den": [0.0] * NL}
        hooks = [(layers[L].self_attn.q_proj,
                  ablation_hook(U[L], op, None if mu_dev is None else mu_dev[L], state, Ws32[L], acc, L))
                 for L in range(NL)]
        windows = sp["R"] if arm in R_ONLY else range(a["n_windows"])
        sc = fw.scores(windows, hooks, state, set(sp["D"]))
        record_arm(rec, arm, op, sc, header(arm, op, basis), corpus, tok)
        num, den = np.array([float(x) for x in acc["num"]]), np.array([float(x) for x in acc["den"]])
        ast[P.wkey("under_ablation", sp["D"])][arm] = {"N": num, "den": den, "ratio": num / den}
        sink.json("arm_stats.json", ast)
        write_compute()

    def stop(gate, after):
        info["stopped"] = {"gate": gate, "after": after}
        sink.json("metrics.json", metrics_doc(rec, gates, info, jm))
        write_compute()
        sink.event("stop", gate=gate)
        shutil.rmtree(work)
        print(f"REALIZEDNULL_Q14_STOP cell={cid} gate={gate} after={after}", flush=True)

    # ---- Gram on C, eigen-coordinates, class means (§2.2, §2.3) ----
    rec.stage("gram_c")
    Gm, sums = gram_pass(fw, sp["C"], class_sums=True)
    lap("gram_c")
    eg = eigen_stage(Gm, [layer.self_attn.q_proj.weight for layer in layers], band64, tope32, spill)
    del Gm
    torch.cuda.empty_cache()
    mu, counts = class_means_q14(sums, WINDOW)
    mu_dev = [torch.as_tensor(m, dtype=torch.float32, device=dev) for m in mu]
    V64, Mp, Sp = eg.pop("V"), eg.pop("Mp"), eg.pop("Sp")
    lap("eigen")

    # ---- fresh EM and spec draws (§3.2) ----
    rec.stage("construct")
    gates["G-EMATCH"], ematch = {"draws": {}}, {}
    for s in [s for s in sl["em"] if s not in GB3_EM]:
        Qs, er = em_fresh(V64, eg["mu"], eg["e_target"], s)
        B[f"em_s{s}"] = [Q.astype(np.float32) for Q in Qs]
        src[f"em_s{s}"] = ("energy_matched_subspace, seed 70_000 + 1000*s + L", 70_000 + 1000 * s)
        gates["G-EMATCH"]["draws"][f"em_s{s}"] = {**ematch_gate(eg["e_target"], er, d),
                                                  P.wkey("e_target", sp["C"]): eg["e_target"],
                                                  P.wkey("e_real", sp["C"]): er}
        ematch[s] = [{"e_target": t, "e_realized": e} for t, e in zip(eg["e_target"], er)]
    jm["ematch"] = [ematch.get(s) for s in sl["em"]]              # None for GB-3's banked s3-s5
    gm = gates["G-EMATCH"]["draws"].values()
    gates["G-EMATCH"].update(n=len(gm), n_pass=sum(g["pass"] for g in gm))
    for s in [s for s in sl["spec"] if s not in GB3_SPEC]:
        B[f"spec_s{s}"] = [U.astype(np.float32) for U in spec_fresh(V64, band64, s)]
        src[f"spec_s{s}"] = ("QR(V (A∘S)), default_rng(60_000 + 1000*s + L)", 60_000 + 1000 * s)
    V32 = [v.astype(np.float32) for v in V64]
    fam_in = {"N_T": (Mp, Sp, V64), "E_T": ([np.eye(d)] * NL, Sp, V64)}
    del Mp, Sp, V64                       # fam_in holds them until construct_all returns
    lap("em_spec")

    # ---- the grid scan and every construction (§2.2, §3.2), in the main process before any arm forward, so
    # nothing contends with it; t_con and t_grid_per_layer reach compute.json before `none` runs (§0.4 U5) ----
    sink.event("construct_all")
    built_all = construct_all(fam_in, band64, sl)
    del fam_in                            # M′, S′ and V in fp64
    tgt, grids, cons, fam_draws = (built_all[k] for k in ("tgt", "grids", "cons", "fam_draws"))
    for name, U in built_all["bases"].items():
        f = P.FAMILIES[cons[name]["family_name"]]
        B[name] = [u.astype(np.float32) for u in U]
        src[name] = (f"construct_draw on the {f['q']} quantity × {f['dose']}, base seed {f['seed']} + 1000*s "
                     "(+ 10_000_000*r per redraw)", cons[name]["base_seed"])
    jm["construction"] = {fam: [cons[n] for n in names] for fam, names in fam_draws.items()}
    n_ld = sum(len(v) for v in fam_draws.values()) * NL
    t_con = built_all["t_con_s"] / n_ld if n_ld else None
    t_grid = built_all["t_grid"]
    del built_all                         # the fp64 bases
    family_records(cons, fam_draws, gates)
    lap("construct")
    write_compute()
    sink.event("constructed", t_con=t_con, t_grid_per_layer=t_grid)

    # ---- none on every window, then G14-PRE's arms; G14-PRE before any arm-cost number (§6.1) ----
    sink.event("arm", arm="none")
    none = fw.scores(range(a["n_windows"]))
    record_arm(rec, "none", "none", none, header("none", "none", None), corpus, tok)
    pre = [x for x in arms if x[3] == "pre"]
    for i, (arm, op, basis, _) in enumerate(pre):
        run_arm(arm, op, basis, i, len(arms))
    Rw = sp["R"]
    ref = g14pre_references(Rw, json.loads((root / V1A_ROWS).read_text()), json.loads((root / GB3_WNLL).read_text()))
    gates["G14-PRE"] = g14pre_gate({arm: rec.summary[arm][P.wkey("ppl", Rw)] for arm in G14_ARMS},
                                   rec.summary["band"][P.wkey("dnll", Rw)], run_e_over_c(rec.nll_open, Rw), ref, Rw)
    sink.json("g14pre.json", gates["G14-PRE"])
    print(f"[q14] G14-PRE {'pass' if gates['G14-PRE']['pass'] else 'FAIL'} on windows {Rw[0]}-{Rw[-1]}", flush=True)
    lap("g14pre")
    if not a["smoke"] and not gates["G14-PRE"]["pass"]:
        return stop("G14-PRE", pre[-1][0])

    # ---- phase A: every remaining arm whose basis is not constructed (§9.4 order within each phase) ----
    rest = [(i, x) for i, x in enumerate(arms) if x[3] == "rest"]
    for i, (arm, op, basis, _) in [(i, x) for i, x in rest if not needs_construction(x[2])]:
        run_arm(arm, op, basis, i, len(arms))
    lap("arms_a")

    # ---- verification on C (G-MATCH) ----
    rec.stage("verify_c")
    built = {n: B[n] for names in fam_draws.values() for n in names if n in B}
    hookq = verify_pass(fw, sp["C"], P.verify_entries(band64, built, cons), Ws32)
    gates["G-MATCH"], jm["verify_C"] = {}, {}
    for fam, names in fam_draws.items():
        rows = P.verify_rows(hookq, names, fam)
        jm["verify_C"][fam] = rows
        draws = {n: P.match_gate(r["N"], r["targets"], sp["C"]) for n, r in zip(names, rows) if r is not None}
        gates["G-MATCH"][fam] = {"draws": draws, "n": len(names), "n_pass": sum(x["pass"] for x in draws.values())}
    lap("verify_c")

    # ---- bank every basis built here and μ (§10.2) ----
    loaded = {"ident", "band", "anti32", "topenergy32", "topenergy128"} | {
        f"{fam}_s{s}" for fam, table in (("em", GB3_EM), ("spec", GB3_SPEC)) for s in table}
    banked = {f"{n}/{L}": np.asarray(B[n][L], dtype=np.float32) for n in B if n not in loaded for L in range(NL)}
    banked.update({f"mu/{L}": mu[L].astype(np.float32) for L in range(NL)})
    np.savez(sink.out / "banks.npz", **banked)
    sink.json("banks_manifest.json", {k: array_sha256(v) for k, v in banked.items()})
    del banked
    lap("banks")

    # ---- Gram on D; arm_stats and G-GEN's quantities in a daemon thread during the remaining arms ----
    rec.stage("gram_d")
    GD, _ = gram_pass(fw, sp["D"], class_sums=False)
    SD = [g.cpu().numpy() for g in GD]
    del GD
    torch.cuda.empty_cache()
    lap("gram_d")
    used = sorted({b for _, _, b, _ in arms if b in B and b != "ident"})
    U_by_basis = {b: B[b] for b in used}
    gen_keys = {"band|N_T": ("band", "N_T"), "band|E_T": ("band", "E_T"),
                **{n: (n, P.FAMILIES[cons[n]["family_name"]]["q"]) for n in built}}
    Wh = [layer.self_attn.q_proj.weight.detach().cpu() for layer in layers]
    means = stats_means(mu, counts, sums, WINDOW)

    def load_layer(L):
        return (np.load(spill / f"sigmaC_L{L}.npy").astype(np.float64), SD[L].astype(np.float64),
                V32[L].astype(np.float64), Wh[L].to(torch.float64).numpy())
    rec.stage("arms_b")
    t_thread = time.time()
    th, box = start_thread("q14-stats", stats_all_layers, NL, load_layer, U_by_basis, means, gen_keys)

    # ---- phase B: the arms on RT-family and EM_T bases, while arm_stats runs ----
    for i, (arm, op, basis, _) in [(i, x) for i, x in rest if needs_construction(x[2])]:
        if basis not in B:
            rec.skip(arm, f"basis {basis} not constructed: " + json.dumps(
                {**{k: bool(cons[basis][k]) for k in ("matched", "reachable", "exhausted")},
                 "fail_reason": cons[basis]["fail_reason"]}))
            continue
        run_arm(arm, op, basis, i, len(arms))
        thread_result(th, box, wait=False)
    lap("arms_b")
    stats, gen = assemble_stats(thread_result(th, box, wait=True))
    stats_s = time.time() - t_thread
    lap("arm_stats_wait")

    # ---- G-GEN (on D) and arm_stats.json (§3.6) ----
    gates["G-GEN"], jm["gen_D"] = {}, {}
    for fam, names in fam_draws.items():
        rows = gen_rows(names, fam, gen)
        jm["gen_D"][fam] = rows
        draws = {n: {P.wkey("ratio_sum", sp["D"]): r["N_sum"] / r["target_sum"],
                     "pass": abs(r["N_sum"] / r["target_sum"] - 1) <= R.GEN_TOL}
                 for n, r in zip(names, rows) if r is not None}
        gates["G-GEN"][fam] = {"draws": draws, "n": len(names), "n_pass": sum(x["pass"] for x in draws.values())}
    gates["family_marks"] = {fam: [g for g in ("G-REACH", "G-PIVOT", "G-EXHAUST")
                                   if fam in gates[g] and not gates[g][fam]["pass"]]
                             for fam in fam_draws if fam_draws[fam]}
    kC, kD = P.wkey("C", sp["C"]), P.wkey("D", sp["D"])
    ast.update({"per_basis": {kC: stats["C"], kD: stats["D"]}, "construction": cons, "grids": grids,
                P.wkey("band_targets", sp["C"]): tgt,
                P.wkey("band_hook", sp["C"]): {k.split("|")[1]: v for k, v in hookq.items() if k.startswith("band|")},
                P.wkey("topenergy32_overlap_with_banked", sp["C"]): eg["tope_overlap"],
                P.wkey("gen_quantities", sp["D"]): gen})
    sink.json("arm_stats.json", ast)
    sink.json("metrics.json", metrics_doc(rec, gates, info, jm))
    write_compute()
    sink.event("done")
    shutil.rmtree(work)
    print(f"REALIZEDNULL_Q14_OK cell={cid} arms={len(rec.done_arms)} skipped={len(rec.skipped)}", flush=True)
