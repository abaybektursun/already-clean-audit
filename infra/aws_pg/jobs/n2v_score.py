"""N2V job (studies/next/07_PICKS/N2V_SPEC.md section 9): one launch on one L40S, N12's ablated (unit, pole) pairs read in
fp32 at every pre point.

Per point t, in the order n2v.ORDER (T_14, T_13, T_12, ..., T_7, T_5):
  1. Stop rule (from the second point on): elapsed time plus the projection of every remaining point (fixed cost, seconds
     per plain row and per forward-mode row, measured at the first point) above 0.9 of --budget-min: this point and every
     later one PARKED; the job ends normally.
  2. Download, G-REV, G-CFG, the load (neurons_harvest.load_model: bf16, sdpa), every parameter frozen.
  3. At T_14 only, in bf16 on N12's code: K-A. N12's ablation rows of its K0 units (n12_score.ablate, batches of 8) and
     the windows' losses (n12_score.forward_loss, batches of 8) are run again; each pair's dL must equal N12's banked dL
     (n12_<half>.jsonl -> abl[i][2]) within n2v.K_A_TOL; failing input: the banked dL with the poles rolled inside each
     unit. The same units' probe rows in bf16 are written (reported: the bf16 floor at the same positions).
  4. model.float(), float32 matmul precision "highest", TF32 off: G-FP32 (every parameter fp32).
  5. Plain rows (batches of 4, the last padded with its own last row; default kernels): per unit window the rows clean,
     zero q, fd q (q = carrier and its top 3 competitors, N12's abl order) and probe; every row's per-position loss
     l(j) in fp32. G-CAUSAL: every changed row equals its clean row bitwise at input indices j < i0 (the first changed
     index of that window); failing input: the comparison through j = i0 for the zero rows, which must differ.
     A1 (2026-11-07, SPEC section 14): the loss reads each row's logits from a copy of their own (ce_pos), so row j's
     log-softmax starts at the same 16-byte alignment in every batch slot; G-CAUSAL records every changed row's largest
     difference before i0 with its batch and slot; at the first point the report slot_control runs one window's clean
     row in all four slots through the copy and through the row view (score-01's path).
  6. Forward-mode rows (part "jvp"; batches of 2; deterministic algorithms, the sdpa math kernel): the tangent of every
     l(j) along the zero-ablation direction of pole q (tangent -h at the unit's entries).
  7. At T_13: I-FWD (the clean rows again, reversed, in batches of clean rows only; bitwise; failing input: each clean
     row against its window's first zero row) and I-JVP (the first n2v.N_JVP_ID unit windows' forward-mode rows again,
     reversed; bitwise; failing input: pole 1's tangent against pole 0's).
  8. Per pair (n2v.pair_values): dL, dL_own, dL_fd, dL_fd_own, pred_jvp, pred_jvp_own, the unit's probe; the per-row
     losses and tangents to losses/tNN.npz.
After the last point: n2v_<half>.jsonl (one line per pair) and samples/raw_examples_<half>.jsonl. out/n2v/metrics.json
is rewritten after every phase and never carries a Spearman, a room value or a label.

    python -m infra.aws_pg.jobs.n2v_score --inputs <uri> --corpus <uri> --budget-min <min> --parts core,fd,jvp --out <dir>

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
from infra.aws_pg.jobs import n12_score as J12
from infra.aws_pg.jobs.neurons_harvest import fetch, fetch_corpus
from infra.pg import m2, n2v

UNITS_SHA = "c223dfee261c3f6019ad46d08cccc23859143e6f59698aa0ef24de769f95508a"      # studies/next/n12/units.json
N12_SHA = {"freeze": "30f31cb1b96403aa1e0344d86fb607022d77786b7f333f7a5943556bcaadb391",
           "test": "877926da34620a121067788b3c4736b986cba7c94fdc5bbc4f98e40975599d12"}  # n12/banks/score-02/n12/n12_*.jsonl
CUBLAS_CFG = ":4096:8"
DEVICE = "NVIDIA L40S"


@contextlib.contextmanager
def jvp_mode():
    """The forward-mode rows: deterministic algorithms and the sdpa math kernel (N12 A1's settings for its gradient
    pass; in fp32 the math kernel needs no reduction setting). Everything else runs outside it."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_CFG, "CUBLAS_WORKSPACE_CONFIG not set before CUDA started"
    assert not torch.are_deterministic_algorithms_enabled(), "deterministic mode is on outside the forward-mode rows"
    torch.use_deterministic_algorithms(True)
    with sdpa_kernel([SDPBackend.MATH]):
        yield
    torch.use_deterministic_algorithms(False)


class Run(M.Run):
    def save(self, phase: str):
        self.rec["phase"], self.rec["updated_utc"] = phase, M.utc()
        self.rec["elapsed_s"] = round(time.time() - self.t0, 1)
        M.dump(self.out / "n2v" / "metrics.json", self.rec)
        print(f"[n2v] {phase}", flush=True)


def ce_pos(logits, x):
    """Per-position next-token cross-entropy of each row: [B, W] fp32 (input j predicts input j + 1).
    A1: each row's logits are copied into a buffer of their own before the log-softmax. The vocabulary's 100,278 fp32
    values are 401,112 bytes per position, 8 bytes past a multiple of 16, so in the row view logits[b, :-1] of an fp32
    batch the start of position j moves between two 16-byte alignments with the parity of b * 2049 + j, and the
    vectorized log-softmax sums a row in an order set by that alignment. In a fresh buffer position j's alignment
    depends on j alone. score-01 used the view (in bf16, .float() had always made the copy)."""
    import torch
    import torch.nn.functional as F
    return [F.cross_entropy(logits[b, :-1].to(torch.float32, copy=True), x[b, 1:], reduction="none")
            for b in range(x.shape[0])]


def slot_control(model, tokens, w: int, reg: dict, device: str) -> dict:
    """A1, a report: window w's clean row in every slot of one batch. Per slot, the largest |difference| against slot 0
    of the logits, of the loss through ce_pos (the copy) and of the loss through the row view (score-01's path).
    The diagnosis predicts on the L40S: logits 0 at every slot; copy 0 at every slot; view 0 at slot 2 and above 0 at
    slots 1 and 3. A view at 0 in every slot refutes it."""
    import torch
    import torch.nn.functional as F
    from infra.pg.neurons_harvest import forward
    B = reg["batch"]
    x, _ = M.windows_tensor(tokens, [w] * B, reg["window"], reg["bos"], B)
    x = x.to(device)
    with torch.inference_mode():
        lg = forward(model, x, True).logits
        cp = [v.cpu().numpy().astype(np.float64) for v in ce_pos(lg, x)]
        vw = [F.cross_entropy(lg[b, :-1], x[b, 1:], reduction="none").cpu().numpy().astype(np.float64) for b in range(B)]
        return {"window": int(w), "dtype": str(lg.dtype), "vocab": int(lg.shape[-1]),
                "view_offset_mod16_bytes": [int(lg[b, :-1].data_ptr() % 16) for b in range(B)],
                "logits_max_abs_vs_slot0": [float((lg[b] - lg[0]).abs().max()) for b in range(B)],
                "copy_max_abs_vs_slot0": [float(np.abs(c - cp[0]).max()) for c in cp],
                "view_max_abs_vs_slot0": [float(np.abs(c - vw[0]).max()) for c in vw]}


def plain_rows(model, tower, tokens, specs: list, reg: dict, device: str, hcap: dict | None = None) -> list:
    """specs: (window, layer, indices, kind, neuron) with kind clean/zero/fd/probe. Batches of reg['batch'] (the last
    padded with its own last row). Returns the per-position losses [W] fp32 of each spec. hcap, if given, receives
    h[indices, :] of every clean row at its layer (keyed by (window, layer))."""
    import torch
    from infra.pg.neurons_harvest import forward
    B = reg["batch"]
    out = []
    with torch.inference_mode():
        for i in range(0, len(specs), B):
            chunk = specs[i:i + B]
            padded = chunk + [chunk[-1]] * (-len(chunk) % B)
            x, _ = M.windows_tensor(tokens, [s[0] for s in padded], reg["window"], reg["bos"], len(padded))
            x = x.to(device)
            by = {}
            for b, s in enumerate(padded):
                by.setdefault(s[1], []).append((b,) + tuple(s))
            hs = []
            for lyr, items in by.items():
                def pre(module, args, items=items):
                    h = args[0].clone()
                    for b, w, _, idxs, kind, n in items:
                        if kind == "zero":
                            h[b, idxs, n] = 0
                        elif kind == "fd":
                            h[b, idxs, n] = h[b, idxs, n] * n2v.FD_SCALE
                        elif kind == "clean" and hcap is not None:
                            hcap[(w, lyr)] = args[0][b, idxs, :].float().cpu().numpy()
                    return (h,)

                def post(module, args, output, items=items):
                    o = output.clone()
                    for b, w, _, idxs, kind, n in items:
                        if kind == "probe":
                            v = -args[0][b, idxs, n].unsqueeze(-1) * module.weight[:, n]
                            o[b, idxs] = (o[b, idxs] + v) - v
                    return o
                dp = tower.layers[lyr].mlp.down_proj
                hs += [dp.register_forward_pre_hook(pre), dp.register_forward_hook(post)]
            ls = ce_pos(forward(model, x, True).logits, x)
            for h in hs:
                h.remove()
            out += [ls[b].cpu().numpy() for b in range(len(chunk))]
    return out


def jvp_rows(model, tower, tokens, specs: list, reg: dict, device: str) -> list:
    """specs: (window, layer, indices, neuron). Batches of reg['batch_jvp']. Returns, per spec, the forward-mode
    derivative of every l(j) along the tangent -h at (indices, neuron): the first-order prediction per position."""
    import torch
    import torch.autograd.forward_ad as fwAD
    from infra.pg.neurons_harvest import forward
    B = reg["batch_jvp"]
    out = []
    with jvp_mode(), torch.no_grad():
        for i in range(0, len(specs), B):
            chunk = specs[i:i + B]
            padded = chunk + [chunk[-1]] * (-len(chunk) % B)
            x, _ = M.windows_tensor(tokens, [s[0] for s in padded], reg["window"], reg["bos"], len(padded))
            x = x.to(device)
            by = {}
            for b, s in enumerate(padded):
                by.setdefault(s[1], []).append((b,) + tuple(s))
            with fwAD.dual_level():
                hs = []
                for lyr, items in by.items():
                    def pre(module, args, items=items):
                        h = args[0]
                        t = torch.zeros_like(h)
                        for b, _, _, idxs, n in items:
                            t[b, idxs, n] = -h[b, idxs, n]
                        return (fwAD.make_dual(h, t),)
                    hs.append(tower.layers[lyr].mlp.down_proj.register_forward_pre_hook(pre))
                ls = ce_pos(forward(model, x, True).logits, x)
                for h in hs:
                    h.remove()
                out += [fwAD.unpack_dual(ls[b]).tangent.cpu().numpy() for b in range(len(chunk))]
    return out


def run(a, reg: dict, out: pathlib.Path):
    import torch
    import transformers
    from infra.pg import m1
    from infra.pg import neurons_corpus as NC
    from infra.pg import neurons_harvest as NH
    R = Run(out, a.budget_min * 60)
    parts = a.parts.split(",")
    assert parts[0] == "core" and set(parts) <= {"core", "fd", "jvp"}, parts
    R.rec["parts"] = parts
    v = {"torch": torch.__version__, "transformers": transformers.__version__}
    dev = torch.cuda.get_device_name(0) if a.device == "cuda" else reg.get("cpu_device_name", "cpu")
    R.gate("G_SAME", {"pass": v == reg["versions"] and dev == reg["device"], "versions": v, "device": dev,
                      "registered": [reg["versions"], reg["device"]],
                      "failing_input": {"pass": {**v, "transformers": "0.0.0"} == reg["versions"]}})
    paths = {"units": fetch(f"{a.inputs.rstrip('/')}/units.json", "inputs/units.json"),
             **{h: fetch(f"{a.inputs.rstrip('/')}/n12_{h}.jsonl", f"inputs/n12_{h}.jsonl") for h in ("freeze", "test")}}
    got = {k: m2.sha256_file(p) for k, p in paths.items()}
    want = {"units": reg["units_sha"], **reg["n12_sha"]}
    R.gate("INPUTS", {"pass": got == want, "mismatch": sorted(k for k in want if got[k] != want[k]),
                      "failing_input": {"pass": got["units"] == "0" * 64}})
    units = json.loads(paths["units"].read_text())["units"]
    n12r = [r for h in ("freeze", "test") for r in m2.read_jsonl(paths[h])]
    banked = {(r["unit"], p[0]): (p[1], p[2]) for r in n12r if r["kind"] == "real" for p in r["abl"]}
    corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
    R.gate("G_CORPUS", {"pass": corpus["tokens_sha256"] == reg["tokens_sha256"], "got": corpus["tokens_sha256"],
                        "failing_input": {"pass": "0" * 64 == reg["tokens_sha256"]}})
    dec = NH.decoder(transformers.AutoTokenizer.from_pretrained(corpus["tokenizer_dir"]))
    pl = n2v.plan(units, n12r)
    uj = json.loads(paths["units"].read_text())
    ex_ids = set(uj["examples"]["freeze"]) | set(uj["examples"]["test"])
    (out / "n2v" / "losses").mkdir(parents=True, exist_ok=True)
    (out / "n2v" / "samples").mkdir(parents=True, exist_ok=True)
    pairs, examples, cost = [], {}, {}
    order = list(pl)
    for t in order:
        items = pl[t]
        n_uw = sum(len(x["uw"]) for x in items)
        n_plain = n_uw * (10 if "fd" in parts else 6)
        n_jvp = n_uw * n2v.N_POLES if "jvp" in parts else 0
        if t != order[0]:
            rest = order[order.index(t):]
            rem = 0.0
            for x in rest:
                uw = sum(len(i["uw"]) for i in pl[x])
                rem += cost["fixed_s"] + uw * (10 if "fd" in parts else 6) * cost["plain_s"] + \
                    (uw * n2v.N_POLES * cost["jvp_s"] if "jvp" in parts else 0.0)
                if x == n2v.IDENTITY_T:
                    rem += uw * cost["plain_s"] + n2v.N_JVP_ID * n2v.N_POLES * cost["jvp_s"]
            elapsed = time.time() - R.t0
            R.rec["projection"] = {"at": t, "elapsed_s": round(elapsed), "remaining_s": round(rem), "budget_s": R.budget_s}
            if elapsed + rem > 0.9 * R.budget_s:
                R.rec["parked"] = rest
                R.save(f"parked from T_{t}")
                break
        pt = {"t": t, "units": len(items), "unit_windows": n_uw, "plain_rows": n_plain, "jvp_rows": n_jvp}
        R.rec["points"][str(t)] = pt
        t_fix = time.time()
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
        if t == n2v.KNOWN_T:
            ka = items[:n2v.K_A_N]
            rows, meta = [], []
            for it in ka:
                rows.append((it["uw"][0][0], it["layer"], [], None))
                meta.append((it["unit"], None, it["uw"][0][0]))
                for w, idxs in it["uw"]:
                    for n, s in it["poles"]:
                        rows.append((w, it["layer"], idxs, n))
                        meta.append((it["unit"], m2.pole_id(it["layer"], n, s), w))
            ws = sorted({w for it in ka for w, _ in it["uw"]})
            acts = [layer.mlp.act_fn for layer in tower.layers]
            loss = {}
            J12.forward_loss(model, NH.MLPTap(tower.layers), acts, corpus["tokens"], {w: [] for w in ws}, ws,
                             {**reg, "batch": 8}, a.device, {}, loss)
            ls = J12.ablate(model, tower, corpus["tokens"], rows, {**reg, "batch": 8}, a.device)
            meas = {}
            for (uid, pole, w), lv in zip(meta, ls):
                if pole is not None:
                    meas[(uid, pole)] = meas.get((uid, pole), 0.0) + lv - loss[w]
            keys = sorted(meas)
            d = np.array([abs(meas[k] - banked[k][1]) for k in keys])
            rolled = {}
            for it in ka:
                ids = [m2.pole_id(it["layer"], n, s) for n, s in it["poles"]]
                for j, p in enumerate(ids):
                    rolled[(it["unit"], p)] = banked[(it["unit"], ids[(j + 1) % len(ids)])][1]
            fd_ = np.array([abs(meas[k] - rolled[k]) for k in keys])
            pr16 = plain_rows(model, tower, corpus["tokens"],
                              [(w, it["layer"], idxs, kind, it["poles"][0][0]) for it in ka for w, idxs in it["uw"]
                               for kind in ("clean", "probe")], {**reg, "batch": 8}, a.device)
            p16 = [float((pr16[2 * j + 1].astype(np.float64) - pr16[2 * j].astype(np.float64)).sum())
                   for j in range(len(pr16) // 2)]
            R.gate("K_A", {"pass": bool(d.max() <= n2v.K_A_TOL), "units": [it["unit"] for it in ka], "n_pairs": len(keys),
                           "max_abs_diff": float(d.max()), "n_exact": int((d == 0).sum()), "tol": n2v.K_A_TOL,
                           "failing_input": {"what": "N12's banked dL with the poles rolled inside each unit",
                                             "pass": bool(fd_.max() <= n2v.K_A_TOL), "min_abs_diff": float(fd_.min())}})
            R.report("bf16_probe_KA_units", {"per_unit_window_window_sum": p16,
                                             "median_abs": float(np.median(np.abs(p16)))})
        pre_dts = sorted({str(p.dtype) for p in model.parameters()})
        model.float()
        torch.set_float32_matmul_precision("highest")
        if a.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        dts = sorted({str(p.dtype) for p in model.parameters()})
        R.gate(f"t{t:02d}_G_FP32", {"pass": dts == ["torch.float32"] and torch.get_float32_matmul_precision() == "highest"
                                    and not torch.backends.cuda.matmul.allow_tf32, "dtypes": dts,
                                    "failing_input": {"what": "the parameter dtypes before the cast", "dtypes": pre_dts,
                                                      "pass": pre_dts == ["torch.float32"]}})
        pt["fixed_s"] = round(time.time() - t_fix, 1)
        if t == order[0]:
            R.report("slot_control", slot_control(model, corpus["tokens"], items[0]["uw"][0][0], reg, a.device))
        # plain rows
        kinds = [("clean", None)] + [("zero", q) for q in range(n2v.N_POLES)] + \
            ([("fd", q) for q in range(n2v.N_POLES)] if "fd" in parts else []) + [("probe", 0)]
        specs, keyz = [], []
        for ii, it in enumerate(items):
            for w, idxs in it["uw"]:
                for kind, q in kinds:
                    specs.append((w, it["layer"], idxs, kind, None if q is None else it["poles"][q][0]))
                    keyz.append((ii, w, kind, q))
        hcap = {}
        t0 = time.time()
        ls = plain_rows(model, tower, corpus["tokens"], specs, reg, a.device, hcap)
        pt["plain_s"] = round(time.time() - t0, 1)
        L = {k: v for k, v in zip(keyz, ls)}
        pos = {k: i for i, k in enumerate(keyz)}
        B = reg["batch"]
        ok, fail_eq, crow = [], [], []
        for (ii, w, kind, q), v in L.items():
            if kind == "clean":
                continue
            i0 = int(min(dict(items[ii]["uw"])[w]))
            lc = L[(ii, w, "clean", None)]
            p_, f_ = n2v.causal_ok(lc, v, i0)
            ok.append(p_)
            if kind == "zero":
                fail_eq.append(f_)
            d = np.abs(v[:i0].astype(np.float64) - lc[:i0].astype(np.float64))
            r, c = pos[(ii, w, kind, q)], pos[(ii, w, "clean", None)]
            crow.append({"unit": items[ii]["unit"], "window": int(w), "kind": kind, "q": q, "i0": i0, "bitwise": p_,
                         "max_abs": float(d.max()), "n_positions_differ": int((d != 0).sum()),
                         "batch": r // B, "slot": r % B, "clean_batch": c // B, "clean_slot": c % B})
        diff = [x for x in crow if not x["bitwise"]]
        R.gate(f"t{t:02d}_G_CAUSAL", {"pass": all(ok), "n_rows": len(ok), "n_bitwise_before_i0": int(sum(ok)),
                                      "tol": 0.0, "max_abs_before_i0": max(x["max_abs"] for x in crow),
                                      "n_over_tol": len(diff),
                                      "n_over_tol_by_slot_parity": {
                                          "same_as_clean": sum((x["slot"] - x["clean_slot"]) % 2 == 0 for x in diff),
                                          "other": sum((x["slot"] - x["clean_slot"]) % 2 == 1 for x in diff)},
                                      "worst10": sorted(crow, key=lambda x: -x["max_abs"])[:10],
                                      "failing_input": {"what": "zero rows compared through j = i0",
                                                        "pass": all(fail_eq), "n_equal": int(sum(fail_eq))}})
        # forward-mode rows
        T = {}
        if "jvp" in parts:
            js, jk = [], []
            for ii, it in enumerate(items):
                for w, idxs in it["uw"]:
                    for q in range(n2v.N_POLES):
                        js.append((w, it["layer"], idxs, it["poles"][q][0]))
                        jk.append((ii, w, "jvp", q))
            if a.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            T = dict(zip(jk, jvp_rows(model, tower, corpus["tokens"], js, reg, a.device)))
            pt["jvp_s"] = round(time.time() - t0, 1)
            if a.device == "cuda":
                pt["jvp_max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        # the identity pairs
        if t == n2v.IDENTITY_T:
            cs = [(ii, w) for ii, it in enumerate(items) for w, _ in it["uw"]][::-1]
            again = plain_rows(model, tower, corpus["tokens"],
                               [(w, items[ii]["layer"], dict(items[ii]["uw"])[w], "clean", None) for ii, w in cs],
                               reg, a.device)
            eq = [np.array_equal(L[(ii, w, "clean", None)], v) for (ii, w), v in zip(cs, again)]
            fe = [np.array_equal(L[(ii, w, "zero", 0)], v) for (ii, w), v in zip(cs, again)]
            R.gate("I_FWD", {"pass": all(eq), "n": len(eq), "n_bitwise": int(sum(eq)),
                             "max_abs_diff": float(max(np.abs(L[(ii, w, "clean", None)].astype(np.float64) - v).max()
                                                       for (ii, w), v in zip(cs, again))),
                             "failing_input": {"what": "each clean row against its window's first zero row",
                                               "pass": all(fe)}})
            if "jvp" in parts:
                sub = [k for k in T if k[3] in (0, 1)][:2 * n2v.N_JVP_ID]
                jj = [(w, items[ii]["layer"], dict(items[ii]["uw"])[w], items[ii]["poles"][q][0]) for ii, w, _, q in sub]
                again = jvp_rows(model, tower, corpus["tokens"], jj[::-1], reg, a.device)[::-1]
                eq = [np.array_equal(T[k], v) for k, v in zip(sub, again)]
                fe = [np.array_equal(T[(ii, w, "jvp", 0)], T[(ii, w, "jvp", 1)]) for ii, w, _, q in sub if q == 0]
                R.gate("I_JVP", {"pass": all(eq), "n": len(eq), "n_bitwise": int(sum(eq)),
                                 "max_abs_diff": float(max(np.abs(T[k].astype(np.float64) - v).max()
                                                           for k, v in zip(sub, again))),
                                 "failing_input": {"what": "pole 1's tangent against pole 0's", "pass": all(fe)}})
        # per-pair values
        arr = {}
        for ii, it in enumerate(items):
            uws = it["uw"]
            lc = {w: L[(ii, w, "clean", None)] for w, _ in uws}
            rows = {(k[2], k[3], k[1]): v for k, v in L.items() if k[0] == ii and k[2] != "clean"}
            rows.update({(k[2], k[3], k[1]): v for k, v in T.items() if k[0] == ii})
            pv = n2v.pair_values(lc, rows, dict(uws))
            for q, (n, s) in enumerate(it["poles"]):
                pole = m2.pole_id(it["layer"], n, s)
                rec = {"unit": it["unit"], "half": it["half"], "t": t, "layer": it["layer"], "pole": pole, "rank": q,
                       "n_windows": len(uws), "pred_bf16": banked[(it["unit"], pole)][0],
                       "meas_bf16": banked[(it["unit"], pole)][1], **pv["pairs"][q], **pv["probe"],
                       "sum_abs_h": float(sum(np.abs(hcap[(w, it["layer"])][:, n]).sum() for w, _ in uws)),
                       "n_later": int(sum(m2.WINDOW - min(ix) for _, ix in uws))}
                pairs.append(rec)
            for k, v in L.items():
                if k[0] == ii:
                    arr[f"{it['unit']}|{k[1]}|{k[2]}|{k[3]}"] = v
            for k, v in T.items():
                if k[0] == ii:
                    arr[f"{it['unit']}|{k[1]}|{k[2]}|{k[3]}"] = v
            if it["unit"] in ex_ids:
                w0, ix0 = uws[0]
                dl = L[(ii, w0, "zero", 0)].astype(np.float64) - lc[w0].astype(np.float64)
                top = np.argsort(-np.abs(dl))[:5]
                examples[it["unit"]] = {
                    "unit": it["unit"], "half": it["half"], "t": t,
                    "exemplars": [M.decode_ctx(corpus, dec, w * m2.WINDOW + i - 1) for w, ix in uws for i in ix],
                    "pairs": [p for p in pairs if p["unit"] == it["unit"]],
                    "carrier_zero_top5": [{"window": w0, "j": int(j), "dl": float(dl[j]), "clean_loss": float(lc[w0][j]),
                                           "predicts": M.decode_ctx(corpus, dec, w0 * m2.WINDOW + int(j))} for j in top]}
        np.savez_compressed(out / "n2v" / "losses" / f"t{t:02d}.npz", **arr)
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()
        shutil.rmtree(pathlib.Path(os.environ["HF_HUB_CACHE"]) / ("models--" + m2.REPO.replace("/", "--")))
        if not cost:
            cost = {"fixed_s": pt["fixed_s"] + 30.0, "plain_s": pt["plain_s"] / max(n_plain, 1),
                    "jvp_s": pt.get("jvp_s", 0.0) / max(n_jvp, 1)}
            R.rec["measured"] = {k: round(v, 4) for k, v in cost.items()}
        R.save(f"done T_{t}")
    for h in ("freeze", "test"):
        with open(out / "n2v" / f"n2v_{h}.jsonl", "w") as f:
            f.writelines(json.dumps(p) + "\n" for p in pairs if p["half"] == h)
        with open(out / "n2v" / "samples" / f"raw_examples_{h}.jsonl", "w") as f:
            f.writelines(json.dumps(examples[b], ensure_ascii=False) + "\n" for b in uj["examples"][h] if b in examples)
    R.rec["n_pairs"] = {h: sum(p["half"] == h for p in pairs) for h in ("freeze", "test")}
    R.rec["complete"] = True
    R.save("done")


def registry(path=None) -> dict:
    if path is not None:
        r = json.loads(pathlib.Path(path).read_text())
        r["points"] = {int(k): v for k, v in r["points"].items()}
        return r
    from infra.pg import m1
    return {"versions": M.VERSIONS, "device": DEVICE, "units_sha": UNITS_SHA, "n12_sha": N12_SHA,
            "tokens_sha256": M.TOKENS_SHA, "points": {p[0]: list(p) for p in m2.POINTS}, "ref_config": m1.REF_CONFIG,
            "window": m2.WINDOW, "batch": 4, "batch_jvp": 2, "bos": m2.BOS}


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", required=True, help="S3 prefix or directory: units.json, n12_{freeze,test}.jsonl")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--budget-min", type=float, required=True)
    ap.add_argument("--parts", default="core,fd,jvp")
    ap.add_argument("--registry", default=None, help="dry run only: a planted registry JSON")
    ap.add_argument("--device", default="cuda")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CFG
    out = pathlib.Path(a.out)
    (out / "n2v").mkdir(parents=True, exist_ok=True)
    if a.registry is None:
        from infra.pg.diff_harvest import hf_online_setup
        hf_online_setup()
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    import torch
    reg = registry(a.registry)
    if a.device == "cuda":
        assert not torch.cuda.is_initialized(), "CUDA started before CUBLAS_WORKSPACE_CONFIG was set"
        assert torch.cuda.is_available(), "no CUDA on this instance"
    run(a, reg, out)
    print("N2V_JOB_OK", flush=True)


if __name__ == "__main__":
    main()
