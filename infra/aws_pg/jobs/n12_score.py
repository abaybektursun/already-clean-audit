"""N12 job (studies/next/07_PICKS/N12_SPEC.md section 9): one launch, every unit of the plan at every point.

Per point t, in the order n12.ORDER (T_14, T_15, T_13, ..., T_5):
  1. Stop rule (from the second point on): if the elapsed time plus the projection of every remaining point (the fixed
     cost, seconds per forward row, per gradient window, measured at the first point) exceeds 0.9 of --budget-min,
     this and every later point is PARKED and the job ends normally.
  2. topk/ and stats/ of the needed layers (S3, the banked grid harvest); the download, G-REV, G-CFG, the load
     (neurons_harvest.load_model, bf16, sdpa), every parameter frozen (requires_grad False), G0 with u x 1.1 as its
     failing input and the activation swap reported (M2's A5).
  3. K0 first, at T_14 only: the first 10 real carriers (by unit id) whose pre point is T_14: their windows forwarded
     (batches of 8), B recomputed, against M2's banked pre-role B (ranks files) within K0_TOL_B and their activations
     against M2's banked vectors within the same-device tolerance; failing input: the same carriers' banked birth-role B.
  4. The forward (batches of exactly 8, the M2 shape, all logits): a = act(g) * u in fp32 for every neuron of each
     needed layer at each needed position (neurons_harvest.MLPTap), and each window's summed next-token loss in fp32.
     G-M2V: every (unit, role) M2 banked at t against M2's vectors (same device, A1 statistic; failing input: the
     neuron axis rolled by one). G-EX (A3: gated at T_5, T_6, T_15, reported at T_7-T_14) and G-TOPK, as M2.
  5. Per (unit, role): B and the other M2 scores (m2.scores), an M2-format record (ranks_<half>.jsonl), the arrays
     (arrays/tNN/<unit>_<role>.npz: B, a). Pre roles: the competitor set (n12.competitors).
  6. The gradient pass at pre roles (batch 1, every window holding a pre-role position): the summed next-token loss
     L of the window, autograd from the lowest needed layer's down-projection input; G = dL/dh and h at every exemplar
     position, and G averaged over the M_LM loss-matched positions of each exemplar (n12.loss_matched). The pull P,
     P' (n12.pull) into the arrays. Amendment A1 (2026-11-06): this pass, and its I-GRAD re-run, runs under
     grad_mode(): torch.use_deterministic_algorithms(True) and the sdpa math kernel only, with bf16 reductions;
     CUBLAS_WORKSPACE_CONFIG is set to CUBLAS_CFG by main() before CUDA starts. Every other step keeps the harvest's
     kernels. The peak memory of the pass is written per point (grad_max_mem_gb).
  7. Ablation at real pre roles (batches of 8 rows): per window of the unit, the carrier and its top N_ABL competitors
     zeroed (the neuron's down-projection input) at the unit's positions; one clean row per unit (G-ABL-ID: equal to
     the forward's loss of that window within ABL_ID_TOL; failing input: another window's loss).
  8. At T_13 (m2.IDENTITY_T): I-ID (the forward again, reversed; failing input: M2's birth-role vectors at T_14) and
     I-GRAD (the gradient pass again, reversed; failing input: the neuron axis rolled). A1: I-GRAD also writes the
     largest |dG| and |dS| between the passes to metrics.json, the count of changed units per half, and every unit
     whose u_S or S ranks (carrier and competitors) differ to igrad_changed_<half>.jsonl (unit, half, both values).
After the last point: one record per unit with pre and birth arrays (n12_<half>.jsonl, n12.unit_record) and the raw
examples (samples/raw_examples_<half>.jsonl). out/n12/metrics.json is rewritten after every phase and never carries
a rank, a hit or a label.

    python -m infra.aws_pg.jobs.n12_score --inputs <uri> --exemplars <uri> --m2 <uri> --corpus <uri> --grid <uri> \
        --budget-min <min> --out <dir>

Fail-loud: no try/except, no fallback, no retry.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import shutil
import time

import numpy as np

from infra.aws_pg.jobs import m2_score as M
from infra.aws_pg.jobs.neurons_harvest import FETCH, fetch, fetch_corpus
from infra.pg import m2, n12

UNITS_SHA = "c223dfee261c3f6019ad46d08cccc23859143e6f59698aa0ef24de769f95508a"   # studies/next/n12/units.json
M2_RANKS_SHA = {"freeze": "864fd0dd7cf84cebfcc99d0f1a166f67908ed4c9bcabc68e65c37156f9ad6000",
                "test": "dc247ea35089dc0aead2777e55a6f43bd4faf9db3a9f9a58fc27a10e9406d0c5"}   # studies/next/m2/banks/score-06/m2/ranks_*.jsonl
CUBLAS_CFG = ":4096:8"      # A1 (head's decision): in torch 2.12 it sets the cuBLAS workspace to 32 MiB (SPEC section 14)


@contextlib.contextmanager
def grad_mode():
    """A1 (2026-11-06): the gradient pass with torch.use_deterministic_algorithms(True) and scaled-dot-product attention
    on the math kernel only, its reductions kept in bf16 (allow_fp16_bf16_reduction_math_sdp): the math kernel's
    default fp32 upcast saves 0.959 GB per layer at 2,049 tokens against 0.640 GB, a peak of about 46 GB of the L40S's
    48 GB for a window whose lowest needed layer is 0 (SPEC section 14, A1). score-01's batch-1 backward on the L40S
    was not bit-deterministic (I-GRAD: 4 of 98 u_S changed). The forward, G0, K0, G-M2V, G-EX, G-TOPK and the ablation
    rows run outside it, on the harvest's kernels."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_CFG, "CUBLAS_WORKSPACE_CONFIG not set before CUDA started"
    assert not torch.are_deterministic_algorithms_enabled(), "deterministic mode is on outside the gradient pass"
    assert not torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(), "bf16 math reduction on outside the pass"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    with sdpa_kernel([SDPBackend.MATH]):
        yield
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    torch.use_deterministic_algorithms(False)


class Run(M.Run):
    def save(self, phase: str):
        self.rec["phase"], self.rec["updated_utc"] = phase, M.utc()
        self.rec["elapsed_s"] = round(time.time() - self.t0, 1)
        M.dump(self.out / "n12" / "metrics.json", self.rec)
        print(f"[n12] {phase}", flush=True)


def ce_rows(logits, x) -> list:
    """Summed next-token cross-entropy of each row, fp32: input index j predicts input j + 1, j = 0 .. W - 1."""
    import torch.nn.functional as F
    return [float(F.cross_entropy(logits[b, :-1].float(), x[b, 1:], reduction="sum")) for b in range(x.shape[0])]


def forward_loss(model, tap, acts, tokens, need: dict, ws: list, reg: dict, device: str, cap: dict, loss: dict) -> float:
    """Windows ws in batches of reg['batch'] (the last padded with its own last window): a [N] fp32 for every
    (layer, position) of need[w], and the window's summed loss."""
    import torch
    from infra.pg.neurons_harvest import forward
    t0 = time.time()
    B = reg["batch"]
    with torch.inference_mode():
        for i in range(0, len(ws), B):
            x, padded = M.windows_tensor(tokens, ws[i:i + B], reg["window"], reg["bos"], B)
            x = x.to(device)
            want = {}
            for b, w in enumerate(padded[:len(ws[i:i + B])]):
                for lyr, idx, e in need[w]:
                    want.setdefault(lyr, []).append((b, idx, e))

            def fn(lyr, g, u, a_model):
                for b, idx, e in want.get(lyr, []):
                    cap[(lyr, e)] = (acts[lyr](g[b, idx].float()) * u[b, idx].float()).cpu().numpy()
            out = tap.run(fn, lambda: forward(model, x, True))
            ls = ce_rows(out.logits, x)
            for b, w in enumerate(ws[i:i + B]):
                loss[w] = ls[b]
    return time.time() - t0


def grad_capture(model, tower, tokens, need_g: dict, ws: list, excl: dict, reg: dict, device: str, into: dict) -> float:
    """Batch 1 per window: L = summed next-token loss; autograd from the lowest needed layer's down-projection input
    (a detached leaf; higher needed layers keep their grad). For each (layer, input index, position) of need_g[w]:
    G and h at the exemplar, the mean G over its loss-matched positions, the losses. Runs under grad_mode() (A1)."""
    import torch
    import torch.nn.functional as F
    from infra.pg.neurons_harvest import forward
    t0 = time.time()
    with grad_mode():
        _grad_windows(model, tower, tokens, need_g, ws, excl, reg, device, into, torch, F, forward)
    return time.time() - t0


def _grad_windows(model, tower, tokens, need_g, ws, excl, reg, device, into, torch, F, forward):
    for w in ws:
        x, _ = M.windows_tensor(tokens, [w], reg["window"], reg["bos"], 1)
        x = x.to(device)
        layers = sorted({lyr for lyr, _, _ in need_g[w]})
        store = {}

        def hook_for(lyr):
            def hook(module, args):
                if lyr == layers[0]:
                    h = args[0].detach().requires_grad_(True)
                    store[lyr] = h
                    return (h,)
                args[0].retain_grad()
                store[lyr] = args[0]
            return hook
        hs = [tower.layers[lyr].mlp.down_proj.register_forward_pre_hook(hook_for(lyr)) for lyr in layers]
        with torch.enable_grad():
            logits = forward(model, x, True).logits
            ce = F.cross_entropy(logits[0, :-1].float(), x[0, 1:], reduction="none")
            ce.sum().backward()
        for h in hs:
            h.remove()
        ell = ce.detach().cpu().numpy().astype(np.float64)
        assert ell.size == reg["window"], ell.size
        for lyr, idx, e in need_g[w]:
            g = store[lyr].grad[0]
            hv = store[lyr].detach()[0]
            valid = idx <= reg["window"] - 1
            lm = n12.loss_matched(ell, idx, excl[w]) if valid else []
            into[(lyr, e)] = {"G": g[idx].float().cpu().numpy(), "H": hv[idx].float().cpu().numpy(),
                              "Glm": (g[lm].float().mean(0).cpu().numpy() if lm else np.zeros(g.shape[-1], np.float32)),
                              "valid": bool(valid), "lm": lm, "ell": float(ell[idx]) if valid else None,
                              "dell": float(np.mean(np.abs(ell[lm] - ell[idx]))) if lm else None, "w": w}
        del logits, ce


def ablate(model, tower, tokens, rows: list, reg: dict, device: str) -> list:
    """rows: (window, layer, input indices, neuron or None). Batches of reg['batch'] rows (the last padded with its own
    last row); the neuron's down-projection input set to 0 at the indices; the summed next-token loss of each row."""
    import torch
    from infra.pg.neurons_harvest import forward
    B = reg["batch"]
    out = []
    with torch.inference_mode():
        for i in range(0, len(rows), B):
            chunk = rows[i:i + B]
            padded = chunk + [chunk[-1]] * (-len(chunk) % B)
            x, _ = M.windows_tensor(tokens, [r[0] for r in padded], reg["window"], reg["bos"], len(padded))
            x = x.to(device)
            by = {}
            for b, (w, lyr, idxs, n) in enumerate(padded):
                if n is not None:
                    by.setdefault(lyr, []).append((b, idxs, n))
            hs = []
            for lyr, items in by.items():
                def hook(module, args, items=items):
                    h = args[0].clone()
                    for b, idxs, n in items:
                        h[b, idxs, n] = 0
                    return (h,)
                hs.append(tower.layers[lyr].mlp.down_proj.register_forward_pre_hook(hook))
            ls = ce_rows(forward(model, x, True).logits, x)
            for h in hs:
                h.remove()
            out += ls[:len(chunk)]
    return out


def m2_name(u: dict, role: str, t: int) -> str | None:
    """M2's vector file for this (unit, role) at t, or None (dt units; pb birth below T_15)."""
    if u["kind"] == "real":
        return f"{u['unit'].replace(':', '_')}_{role}"
    if u["kind"] == "pb" and role == "pre":
        return f"{u['unit'].replace(':', '_')}_pre"
    if u["kind"] == "pb" and role == "birth" and t == 15:
        return f"{u['unit'].replace(':', '_')}_end"
    return None


def m2v_gate(pairs: list) -> dict:
    """G-M2V: [(new a [m, N], M2 a [m, N])] pooled; same-device A1 tolerance; failing input: M2 rolled one neuron."""
    new = np.concatenate([a.ravel() for a, _ in pairs])
    old = np.concatenate([b.ravel() for _, b in pairs])
    g = m2.values_gate(new, old, m2.TOL["same"])
    g["n_units"] = len(pairs)
    g["failing_input"] = m2.values_gate(new, np.concatenate([np.roll(b, 1, axis=1).ravel() for _, b in pairs]),
                                        m2.TOL["same"])
    return g


def unit_pull(u: dict, gc: dict, q999: np.ndarray) -> dict:
    """P, P' [N, 2] for a pre-role unit from its captured gradient rows (valid exemplars only)."""
    rows = [gc[(u["layer"], e)] for e in u["tok"]]
    ok = [r for r in rows if r["valid"]]
    assert ok, f"{u['unit']}: no exemplar with a next-token loss"
    P = n12.pull(np.stack([r["G"] for r in ok]), q999).mean(0)
    Pp = n12.pull(np.stack([r["Glm"] for r in ok]), q999).mean(0)
    return {"P": P, "Pp": Pp, "n_valid": len(ok), "dell": float(np.mean([r["dell"] for r in ok]))}


def run(a, reg: dict, out: pathlib.Path):
    import torch
    import transformers
    from infra.pg import m1
    from infra.pg import neurons_corpus as NC
    from infra.pg import neurons_harvest as NH
    R = Run(out, a.budget_min * 60)
    R.rec["grad_pass_settings"] = {"amendment": "A1", "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                                   "use_deterministic_algorithms": True, "sdpa_backends": ["MATH"],
                                   "allow_fp16_bf16_reduction_math_sdp": True}
    v = {"torch": torch.__version__, "transformers": transformers.__version__}
    R.gate("G_SAME", {"pass": v == reg["versions"], "versions": v, "registered": reg["versions"],
                      "failing_input": {"pass": {**v, "transformers": "0.0.0"} == reg["versions"]}})
    paths = {"units": fetch(f"{a.inputs.rstrip('/')}/units.json", "inputs/units.json"),
             **{f"m2_{h}": fetch(f"{a.inputs.rstrip('/')}/m2_ranks_{h}.jsonl", f"inputs/m2_ranks_{h}.jsonl")
                for h in ("freeze", "test")}}
    for t in range(16):
        paths[t] = fetch(f"{a.exemplars.rstrip('/')}/exemplar_a_t{t:02d}.npz", f"inputs/exemplar_a_t{t:02d}.npz")
    got = {k: m2.sha256_file(p) for k, p in paths.items()}
    want = {"units": reg["units_sha"], "m2_freeze": reg["m2_ranks_sha"]["freeze"], "m2_test": reg["m2_ranks_sha"]["test"],
            **{t: reg["exemplar_sha"][t] for t in range(16)}}
    R.gate("INPUTS", {"pass": all(got[k] == want[k] for k in want), "mismatch": sorted(str(k) for k in want
                                                                                       if got[k] != want[k]),
                      "failing_input": {"pass": got["units"] == "0" * 64}})
    uj = json.loads(paths["units"].read_text())
    units = {u["unit"]: u for u in uj["units"]}
    m2rec = {}
    for h in ("freeze", "test"):
        for r in m2.read_jsonl(paths[f"m2_{h}"]):
            m2rec[(r["unit"], r["role"])] = r
    exa = {t: np.load(paths[t])["a"] for t in range(16)}
    corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
    R.gate("G_CORPUS", {"pass": corpus["tokens_sha256"] == reg["tokens_sha256"], "got": corpus["tokens_sha256"],
                        "failing_input": {"pass": "0" * 64 == reg["tokens_sha256"]}})
    dec = NH.decoder(transformers.AutoTokenizer.from_pretrained(corpus["tokenizer_dir"]))
    pl = n12.plan(list(units.values()))
    ex_ids = set(uj["examples"]["freeze"]) | set(uj["examples"]["test"])
    for d in ("samples", "arrays"):
        (out / "n12" / d).mkdir(parents=True, exist_ok=True)
    fh = {h: open(out / "n12" / f"ranks_{h}.jsonl", "w") for h in ("freeze", "test")}
    side = {}                   # per unit: matched list (real pre), ablation pairs, raw-example texts, dt end ranks
    order = list(pl["points"])
    cost = {}
    for t in order:
        p = pl["points"][t]
        elapsed = time.time() - R.t0
        if t != order[0]:
            rest = order[order.index(t):]
            rem = sum(cost["fixed_s"] + (len(pl["points"][x]["windows"]) + pl["points"][x]["abl_rows"]) *
                      cost["per_row_s"] + len(pl["points"][x]["grad_windows"]) * cost["per_grad_s"] for x in rest)
            if m2.IDENTITY_T in rest:
                q = pl["points"][m2.IDENTITY_T]
                rem += len(q["windows"]) * cost["per_row_s"] + len(q["grad_windows"]) * cost["per_grad_s"]
            R.rec["projection"] = {"at": t, "elapsed_s": round(elapsed), "remaining_s": round(rem),
                                   "budget_s": R.budget_s}
            if elapsed + rem > 0.9 * R.budget_s:
                R.rec["parked"] = rest
                R.save(f"parked from T_{t}")
                break
        pt = {"t": t, "windows": len(p["windows"]), "grad_windows": len(p["grad_windows"]),
              "abl_rows": p["abl_rows"], "records": {"freeze": 0, "test": 0}}
        R.rec["points"][str(t)] = pt
        t_fix = time.time()
        layers = sorted({lyr for w in p["need"].values() for lyr, _, _ in w})
        files = M.layer_files(a.grid, t, layers, reg)
        deg = [m2.pole_id(lyr, int(i), int(s)) for lyr in layers for i, s in np.argwhere(m2.degenerate(files[lyr][1]))]
        pt["n_degenerate_poles"], pt["degenerate_poles"] = len(deg), deg
        local, pt["download"] = M.download(reg["points"][t][2], reg["points"][t][3])
        R.gate(f"t{t:02d}_G_REV", pt["download"].pop("G_REV"))
        cfg = json.loads((local / "config.json").read_text())
        g = m1.config_gate(cfg, reg["ref_config"])
        g["pass"] = bool(g["pass"] and g["diff_keys"] == [])
        g["failing_input"] = m1.config_gate({**cfg, "model_type": "olmo2-retrofit"}, reg["ref_config"])
        R.gate(f"t{t:02d}_G_CFG", g)
        model, _ = NH.load_model(str(local), "O7", device=a.device)
        for prm in model.parameters():
            prm.requires_grad_(False)
        tower = NH.text_tower(model)
        acts = [layer.mlp.act_fn for layer in tower.layers]
        tap = NH.MLPTap(tower.layers)
        act = NH.act_name(acts[0])
        x0, _ = M.windows_tensor(corpus["tokens"], p["windows"][:reg["batch"]], reg["window"], reg["bos"], reg["batch"])
        with torch.inference_mode():
            r0 = NH.g0_ratios(model, tap, acts, x0.to(a.device))
            r1 = NH.g0_ratios(model, tap, [NH.WRONG_ACT[act]] * len(acts), x0.to(a.device))
            r2 = NH.g0_ratios(model, tap, acts, x0.to(a.device), u_scale=NH.WRONG_U_SCALE)
        R.report(f"t{t:02d}_G0_act_swap", {"max_ratio": max(r1), "under_tol": NH.g0_pass(r1)})
        R.gate(f"t{t:02d}_G0", {"pass": NH.g0_pass(r0), "max_ratio": max(r0), "failing_input": {
            "u_scale": NH.WRONG_U_SCALE, "pass": NH.g0_pass(r2), "max_ratio": max(r2), "min_ratio": min(r2)}})
        colnorm = {lyr: tower.layers[lyr].mlp.down_proj.weight.detach().float().norm(dim=0).cpu().numpy()
                   for lyr in layers}
        pt["fixed_s"] = round(time.time() - t_fix, 1)
        roles = [(units[uid], role) for uid, role in p["roles"]]
        cap, loss, fwd = {}, {}, 0.0
        ws = p["windows"]
        if t == n12.KNOWN_T:
            k0 = sorted((u for u, role in roles if u["kind"] == "real" and role == "pre"), key=lambda u: u["unit"])
            k0 = k0[:n12.K0_N]
            assert len(k0) == n12.K0_N, f"K0: {len(k0)} carriers at T_{t}"
            k0_ws = sorted({e // reg["window"] for u in k0 for e in u["tok"]})
            fwd += forward_loss(model, tap, acts, corpus["tokens"], p["need"], k0_ws, reg, a.device, cap, loss)
            bn, bm, bb, pairs = [], [], [], []
            for u in k0:
                tk, st = files[u["layer"]]
                sc = m2.scores(M.a_matrix(cap, u), tk["tok"], st, colnorm[u["layer"]], u["tok"])
                n, s = u["carrier"]
                bn.append(float(sc["B"][n, s]))
                bm.append(m2rec[(u["unit"], "pre")]["value"]["B"])
                bb.append(m2rec[(u["unit"], "birth")]["value"]["B"])
                old = np.load(fetch(f"{a.m2.rstrip('/')}/vectors/t{t:02d}/{m2_name(u, 'pre', t)}.npz",
                                    f"m2v/t{t:02d}/{m2_name(u, 'pre', t)}.npz"))["a"]
                pairs.append((M.a_matrix(cap, u)[:, n], old[:, n]))
            dB = np.abs(np.array(bn) - np.array(bm))
            ga = m2.values_gate(np.concatenate([x for x, _ in pairs]), np.concatenate([y for _, y in pairs]),
                                m2.TOL["same"])
            fb = np.abs(np.array(bn) - np.array(bb))
            R.gate("K0", {"pass": bool(dB.max() <= n12.K0_TOL_B and ga["pass"]), "units": [u["unit"] for u in k0],
                          "B_new": bn, "B_m2": bm, "max_abs_dB": float(dB.max()), "tol_B": n12.K0_TOL_B,
                          "activations": ga, "failing_input": {"what": "M2's birth-role B of the same carriers",
                                                               "pass": bool(fb.max() <= n12.K0_TOL_B),
                                                               "min_abs_dB": float(fb.min())}})
            ws = [w for w in ws if w not in set(k0_ws)]
        fwd += forward_loss(model, tap, acts, corpus["tokens"], p["need"], ws, reg, a.device, cap, loss)
        pt["forward_s"] = round(fwd, 1)
        # G-M2V, G-EX, G-TOPK
        mp = []
        for u, role in roles:
            nm = m2_name(u, role, t)
            if nm is not None:
                mp.append((M.a_matrix(cap, u), np.load(fetch(f"{a.m2.rstrip('/')}/vectors/t{t:02d}/{nm}.npz",
                                                             f"m2v/t{t:02d}/{nm}.npz"))["a"]))
        R.gate(f"t{t:02d}_G_M2V", m2v_gate(mp))
        here = [u for u, _ in roles]
        nb = t + 1 if t < 15 else t - 1
        gex = M.ex_gate(M.ex_entries(here, cap), exa[t], exa[nb], m2.tol_ex(t, a.device_name),
                        names={u["row"]: u["pole"] for u in units.values()})
        (R.gate if m2.ex_is_gate(t, a.device_name) else R.report)(f"t{t:02d}_G_EX", gex)
        R.gate(f"t{t:02d}_G_TOPK", M.topk_gate(cap, files, m2.tol_for(t, a.device_name)))
        # records and arrays per (unit, role)
        vd = out / "n12" / "arrays" / f"t{t:02d}"
        vd.mkdir(parents=True, exist_ok=True)
        comp = {}
        for u, role in roles:
            tk, st = files[u["layer"]]
            am = M.a_matrix(cap, u)
            sc = m2.scores(am, tk["tok"], st, colnorm[u["layer"]], u["tok"])
            rec = m2.record(u, role, t, sc, st["max_abs"])
            fh[u["half"]].write(json.dumps(rec) + "\n")
            pt["records"][u["half"]] += 1
            np.savez_compressed(vd / f"{u['unit'].replace(':', '_')}_{role}.npz", B=sc["B"], a=am,
                                tok=np.array(u["tok"], np.int64))
            sd = side.setdefault(u["unit"], {})
            if role == "pre":
                comp[u["unit"]] = n12.competitors(sc["B"], u["carrier"][0])
                if u["kind"] == "real":
                    sd["matched"] = [list(m2.parse_pole(x)[1:]) for x in rec["matched"]]
            if role == "end":
                sd["end"] = {"rank": rec["rank"]["B"], "rank_opposite": rec["rank_opposite"]["B"],
                             "B": rec["value"]["B"]}
        for f in fh.values():
            f.flush()
        # the gradient pass at pre roles
        pre = [u for u, role in roles if role == "pre"]
        gc = {}
        if pre:
            if a.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            pt["grad_s"] = round(grad_capture(model, tower, corpus["tokens"], p["grad"], p["grad_windows"],
                                              pl["excl"], reg, a.device, gc), 1)
            if a.device == "cuda":
                pt["grad_max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            for u in pre:
                st = files[u["layer"]][1]
                pu = unit_pull(u, gc, st["q999"])
                side[u["unit"]]["grad"] = {"n_valid": pu["n_valid"], "dell": pu["dell"]}
                np.savez_compressed(vd / f"{u['unit'].replace(':', '_')}_pull.npz", P=pu["P"], Pp=pu["Pp"],
                                    G=np.stack([gc[(u["layer"], e)]["G"] for e in u["tok"]]),
                                    H=np.stack([gc[(u["layer"], e)]["H"] for e in u["tok"]]),
                                    Glm=np.stack([gc[(u["layer"], e)]["Glm"] for e in u["tok"]]),
                                    valid=np.array([gc[(u["layer"], e)]["valid"] for e in u["tok"]]))
                if u["unit"] in ex_ids or u["birth"] in ex_ids:
                    side[u["unit"]]["lm_text"] = [
                        {"exemplar": M.decode_ctx(corpus, dec, e), "ell": gc[(u["layer"], e)]["ell"],
                         "matched": [M.decode_ctx(corpus, dec, gc[(u["layer"], e)]["w"] * reg["window"] + j - 1)
                                     for j in gc[(u["layer"], e)]["lm"][:2]]}
                        for e in u["tok"][:3]]
        # ablation at real pre roles
        rp = [u for u in pre if u["kind"] == "real"]
        if rp:
            rows, meta = [], []
            for u in rp:
                n, s = u["carrier"]
                poles = [[n, s]] + comp[u["unit"]][:n12.N_ABL]
                byw = {}
                for e in u["tok"]:
                    if gc[(u["layer"], e)]["valid"]:
                        byw.setdefault(e // reg["window"], []).append(1 + e % reg["window"])
                w0 = sorted(byw)[0]
                rows.append((w0, u["layer"], [], None))
                meta.append((u["unit"], None, w0))
                for w in sorted(byw):
                    for q in poles:
                        rows.append((w, u["layer"], byw[w], q[0]))
                        meta.append((u["unit"], tuple(q), w))
            t_ab = time.time()
            ls = ablate(model, tower, corpus["tokens"], rows, reg, a.device)
            pt["abl_s"] = round(time.time() - t_ab, 1)
            idd, fail_idd = [], []
            meas = {}
            for (uid, q, w), lv in zip(meta, ls):
                if q is None:
                    idd.append(abs(lv - loss[w]))
                    other = [x for x in loss if x != w]
                    fail_idd.append(abs(lv - loss[other[0]]))
                else:
                    meas[(uid, q)] = meas.get((uid, q), 0.0) + lv - loss[w]
            R.gate(f"t{t:02d}_G_ABL_ID", {"pass": bool(max(idd) <= n12.ABL_ID_TOL), "max_abs": float(max(idd)),
                                          "n": len(idd), "tol": n12.ABL_ID_TOL,
                                          "failing_input": {"what": "another window's loss",
                                                            "pass": bool(min(fail_idd) <= n12.ABL_ID_TOL),
                                                            "min_abs": float(min(fail_idd))}})
            for u in rp:
                n, s = u["carrier"]
                ab = []
                for q in [[n, s]] + comp[u["unit"]][:n12.N_ABL]:
                    pred = float(sum(-gc[(u["layer"], e)]["H"][q[0]] * gc[(u["layer"], e)]["G"][q[0]]
                                     for e in u["tok"] if gc[(u["layer"], e)]["valid"]))
                    ab.append([m2.pole_id(u["layer"], q[0], q[1]), pred, meas[(u["unit"], tuple(q))]])
                side[u["unit"]]["abl"] = ab
        # examples: contexts of the top three competitors and the first matched pole at the pre point
        for u in pre:
            if u["birth"] in ex_ids:
                tk = files[u["layer"]][0]

                def top3(n_, s_):
                    return [{"rank": j, "a": float(tk["a"][n_, s_, j]),
                             "text": M.decode_ctx(corpus, dec, tk["tok"][n_, s_, j])}
                            for j in range(3) if tk["tok"][n_, s_, j] >= 0]
                sd = side[u["unit"]]
                sd["exemplars"] = [M.decode_ctx(corpus, dec, e) for e in u["tok"]]
                sd["comp_top3"] = {m2.pole_id(u["layer"], c[0], c[1]): top3(*c) for c in comp[u["unit"]][:3]}
                sd["own_top3"] = top3(*u["carrier"])
                if "matched" in sd:
                    sd["matched_top3"] = {m2.pole_id(u["layer"], *sd["matched"][0]): top3(*sd["matched"][0])}
        # the identity pair
        if t == m2.IDENTITY_T:
            cap2, loss2 = {}, {}
            pre_ws = sorted({e // reg["window"] for u in pre for e in u["tok"]}, reverse=True)
            forward_loss(model, tap, acts, corpus["tokens"], p["need"], pre_ws, reg, a.device, cap2, loss2)
            first = np.concatenate([M.a_matrix(cap, u).ravel() for u in pre])
            second = np.concatenate([M.a_matrix(cap2, u).ravel() for u in pre])
            flips = 0
            for u in pre:
                tk, st = files[u["layer"]]
                b1 = m2.scores(M.a_matrix(cap, u), tk["tok"], st, colnorm[u["layer"]], u["tok"])["B"]
                b2 = m2.scores(M.a_matrix(cap2, u), tk["tok"], st, colnorm[u["layer"]], u["tok"])["B"]
                n, s = u["carrier"]
                flips += int((m2.ranks(b1)[n, s] <= m2.TOP) != (m2.ranks(b2)[n, s] <= m2.TOP))
            g = m2.values_gate(second, first, m2.TOL["same"])
            g["pass"] = bool(g["pass"] and flips == 0)
            g["carrier_top10_flips_B"], g["n_units"] = flips, len(pre)
            fn_, fo_ = [], []
            for u in pre:
                if u["kind"] == "real":
                    nm = m2_name(u, "birth", t + 1)
                    fn_.append(np.load(fetch(f"{a.m2.rstrip('/')}/vectors/t{t + 1:02d}/{nm}.npz",
                                             f"m2v/t{t + 1:02d}/{nm}.npz"))["a"].ravel())
                    fo_.append(M.a_matrix(cap, u).ravel())
            assert fn_, "I-ID: no birth-role vectors at the next point"
            g["failing_input"] = m2.values_gate(np.concatenate(fn_), np.concatenate(fo_), m2.TOL["same"])
            R.gate("I_ID", g)
            gc2 = {}
            grad_capture(model, tower, corpus["tokens"], p["grad"], list(reversed(p["grad_windows"])), pl["excl"], reg,
                         a.device, gc2)
            k1 = sorted(gc)
            g1 = np.concatenate([gc[k]["G"] for k in k1])
            g2 = np.concatenate([gc2[k]["G"] for k in k1])
            flips, dS, changed = 0, 0.0, {"freeze": [], "test": []}
            for u in pre:
                st = files[u["layer"]][1]
                p1, p2 = unit_pull(u, gc, st["q999"]), unit_pull(u, gc2, st["q999"])
                c = comp[u["unit"]]
                S1, S2 = p1["P"] - p1["Pp"], p2["P"] - p2["Pp"]
                u1, u2 = n12.u_share(S1, c, u["carrier"]), n12.u_share(S2, c, u["carrier"])
                q = [list(u["carrier"])] + [x for x in c if x != list(u["carrier"])]
                v1, v2 = np.array([S1[x, y] for x, y in q]), np.array([S2[x, y] for x, y in q])
                r1, r2 = n12.order_ranks(v1), n12.order_ranks(v2)
                dS = max(dS, float(np.abs(v1 - v2).max()))
                flips += int(u1 != u2)
                if u1 != u2 or r1 != r2:
                    changed[u["half"]].append({"unit": u["unit"], "half": u["half"], "kind": u["kind"],
                                               "u_S": [u1, u2], "S_order": [r1, r2],
                                               "poles": [m2.pole_id(u["layer"], x, y) for x, y in q]})
            for h, rows in changed.items():
                with open(out / "n12" / f"igrad_changed_{h}.jsonl", "w") as f:
                    f.writelines(json.dumps(r) + "\n" for r in rows)
            rho = n12.spearman(g1, g2)
            rho_f = n12.spearman(g1, np.concatenate([np.roll(gc2[k]["G"], 1) for k in k1]))
            R.gate("I_GRAD", {"pass": bool(rho >= n12.IGRAD_RHO_MIN and flips == 0), "rho": rho,
                              "exact_share": float((g1 == g2).mean()), "u_S_flips": flips, "n": int(g1.size),
                              "n_units": len(pre), "changed_units": {h: len(r) for h, r in changed.items()},
                              "max_abs_dG": float(np.abs(g1 - g2).max()), "max_abs_G": float(np.abs(g1).max()),
                              "max_abs_dS": dS,
                              "failing_input": {"what": "the neuron axis rolled by one", "rho": rho_f,
                                                "pass": bool(rho_f >= n12.IGRAD_RHO_MIN)}})
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()
        repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / ("models--" + m2.REPO.replace("/", "--"))
        shutil.rmtree(repo_dir)
        for sub in (FETCH / "grid" / f"t{t:02d}", FETCH / "m2v" / f"t{t:02d}"):
            if sub.exists():
                shutil.rmtree(sub)
        if not cost:
            n_rows = -(-len(p["windows"]) // reg["batch"]) * reg["batch"] + p["abl_rows"]
            cost = {"fixed_s": pt["fixed_s"] + 30.0,
                    "per_row_s": (fwd + pt.get("abl_s", 0.0)) / max(n_rows, 1),
                    "per_grad_s": pt.get("grad_s", 0.0) / max(len(p["grad_windows"]), 1)}
            R.rec["measured"] = {k: round(v, 4) for k, v in cost.items()}
            R.rec["measured"]["tokens_per_s_forward"] = round(n_rows * (reg["window"] + 1) / max(
                fwd + pt.get("abl_s", 0.0), 1e-9), 1)
        R.save(f"done T_{t}")
    for f in fh.values():
        f.close()
    # one record per unit
    vroot = out / "n12" / "arrays"
    nh = {h: open(out / "n12" / f"n12_{h}.jsonl", "w") for h in ("freeze", "test")}
    eh = {h: open(out / "n12" / "samples" / f"raw_examples_{h}.jsonl", "w") for h in ("freeze", "test")}
    recs = {}
    for u in units.values():
        f = {r: vroot / f"t{t:02d}" / f"{u['unit'].replace(':', '_')}_{r}.npz" for r, t in u["roles"].items()}
        if not (f["pre"].exists() and f["birth"].exists()):
            continue
        Bp, Bb = np.load(f["pre"])["B"], np.load(f["birth"])["B"]
        pz = np.load(vroot / f"t{u['roles']['pre']:02d}" / f"{u['unit'].replace(':', '_')}_pull.npz")
        sd = side[u["unit"]]
        grad = {"P": pz["P"], "Pp": pz["Pp"], **sd["grad"]}
        rec = n12.unit_record(u, {"t": u["roles"]["pre"], "B": Bp}, {"t": u["roles"]["birth"], "B": Bb}, grad,
                              sd.get("abl"), sd.get("matched"))
        idn = n12.cf_ranks(Bp, Bp, *u["carrier"])
        rec["cf_identity"] = {"r_c": idn["r_c"], "r_o": idn["r_o"], "F": idn["F"]}
        if "end" in sd:
            rec["end"] = sd["end"]
        nh[u["half"]].write(json.dumps(rec) + "\n")
        recs[u["unit"]] = rec
    R.rec["n_unit_records"] = {h: sum(r["half"] == h for r in recs.values()) for h in ("freeze", "test")}
    for h in ("freeze", "test"):
        for bid in uj["examples"][h]:
            ex = {"birth": bid}
            for kind, suf in (("real", ""), ("pb", ":pb"), ("dt", ":dt")):
                if bid + suf in recs:
                    ex[kind] = {**recs[bid + suf], **{k: v for k, v in side[bid + suf].items()
                                                      if k in ("exemplars", "comp_top3", "own_top3", "matched_top3",
                                                               "lm_text")}}
            eh[h].write(json.dumps(ex, ensure_ascii=False) + "\n")
    for f in list(nh.values()) + list(eh.values()):
        f.close()
    R.rec["complete"] = True
    R.save("done")


def registry(path=None) -> dict:
    """The registered facts; `path` (dry run only) replaces them with a planted set of the same shape."""
    if path is not None:
        r = json.loads(pathlib.Path(path).read_text())
        r["points"] = {int(k): v for k, v in r["points"].items()}
        return r
    from infra.pg import m1
    return {"versions": M.VERSIONS, "units_sha": UNITS_SHA, "m2_ranks_sha": M2_RANKS_SHA,
            "exemplar_sha": m2.EXEMPLAR_SHA, "tokens_sha256": M.TOKENS_SHA, "points": {p[0]: list(p) for p in m2.POINTS},
            "ref_config": m1.REF_CONFIG, "window": m2.WINDOW, "batch": 8, "bos": m2.BOS}


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", required=True, help="S3 prefix or directory: units.json, m2_ranks_{freeze,test}.jsonl")
    ap.add_argument("--exemplars", required=True, help="S3 prefix or directory: exemplar_a_tNN.npz (M2's inputs)")
    ap.add_argument("--m2", required=True, help="M2's banked job output (holds vectors/tNN/)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--budget-min", type=float, required=True)
    ap.add_argument("--registry", default=None, help="dry run only: a planted registry JSON")
    ap.add_argument("--device", default="cuda")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CFG          # A1: before CUDA starts
    out = pathlib.Path(a.out)
    (out / "n12").mkdir(parents=True, exist_ok=True)
    if a.registry is None:
        from infra.pg.diff_harvest import hf_online_setup
        hf_online_setup()
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    import torch
    reg = registry(a.registry)
    assert reg["units_sha"] and reg["m2_ranks_sha"]["freeze"] and reg["m2_ranks_sha"]["test"], \
        "units.json and the M2 ranks files are not registered (freeze 1 sets them)"
    if a.device == "cuda":
        assert not torch.cuda.is_initialized(), "CUDA started before CUBLAS_WORKSPACE_CONFIG was set"
        assert torch.cuda.is_available(), "no CUDA on this instance"
        a.device_name = torch.cuda.get_device_name(0)
    else:
        a.device_name = reg.get("cpu_device_name", "cpu")
    run(a, reg, out)
    print("N12_JOB_OK", flush=True)


if __name__ == "__main__":
    main()
