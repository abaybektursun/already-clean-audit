"""NEURONOFF job (studies/neuronoff/SCHEMA.md v4 §2-3, §6; SPEC.md §2, §5.3, §6 as amended A1-A12): one pole at a time.

    python -m infra.aws_pg.jobs.neuron_off --model-key Q4B --repo Qwen/Qwen3-4B-Base \
        --corpus s3://pg-lineage-769063704696/results/neurons/w1/q4b/1/out/corpus/corpus --tokname qwen3 \
        --plan s3://pg-lineage-769063704696/results/neurons/off/plans/<attempt>/plan_Q4B.json \
        [--revision stage1-step1413814] [--max-poles N] [--max-arm-b K] [--skip-arm-b] [--skip-arm-a] --out <dir>

Order: load -> gates (controls.json) -> Arm B (the first K entries of plan.arm_b) -> Arm A (plan.order filtered to
arm "A", the first N) -> status complete. A gate failure writes metrics.json with status "aborted_gate" and
raises. metrics.json is rewritten (temp file, then rename) after the gates, after every Arm B prompt and every
Arm A pole, and at the end. Outputs under --out as SCHEMA §3; NaN and inf are written as null.

Memory (SCHEMA §2): held-out readouts compute logits only at the readout positions (`logits_to_keep` index tensor);
whole-window reads take one window per forward and log-softmax CHUNK = 1024 positions at a time in fp32; kept logits
stay in the model's dtype and are compared CHUNK positions at a time.

Rules this file fixes where SPEC and SCHEMA name a quantity but not every step (listed for ratification):
- A1 Mask domain (SCHEMA §6): no hook edits the BOS or window (text) position 0; generated positions are inside it.
  own.excluded_positions_max_abs_a = max |a| of the pole's sign outside the domain. The identity hook (core mode
  "identity") runs the pole path on the pole's real firing mask writing the original values.
- A2 Determinism: torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=:4096:8, torch.manual_seed(SEED).
  G-det runs on corpus windows 0 and 1 over the prediction columns 0..T-2 of each; pass max |dlogit| == 0; its
  failing input (SPEC §6) is window 1 with the token at window position 100 replaced by (id + 1) mod V, compared
  with the clean window 1 over the positions from 100 on.
- A3 G-ident (SPEC §6, A7): per pole, identity max |dlogit| == 0 over every position of the first held-out window
  prefix (targets: the next corpus token at each position); the clean-repeat value is banked beside it. Failing
  input: a := 1.5 a at every neuron of the layer of the first Arm A pole of plan.order (the whole plan, not
  --max-poles), on that pole's first held-out window prefix; it fails when max |dlogit| > 0.
- A4 G-floor: on prefix + tokens[0:s] for s in plan.floor_slices in order, the lowest neuron j with no sign-sigma
  entry in the mask domain while its other sign has at least one (+ before - on a tie, which cannot happen); pass =
  max |dlogit| == 0 over the prediction columns; failing input = (j, -sigma). The job aborts if a constructible layer
  fails, its failing input passes, or fewer than 80% of used layers are constructible.
- A5 G-fit failing input: neuron i+1's a (mod width) at the same readout positions against a_banked, same rule;
  banked as reproduction.failing_input with the tolerances used. An exact match is flagged (n_exact).
- A6 G-pc failing input: the identity hook's dNLL at the first held-out context, read from the G-ident forward;
  banked as positive_control.failing_input (0 < floor, so it fails).
- A7 Arm B: a record without the prefix property is still scored (tok(continuation) appended) and flagged; the
  analysis excludes it. pole_a_inputs (clean line) is read by the ablate_pole hook before it edits: nothing upstream
  of the pole's down_proj input changes, so it is the clean a. Prompt 0 is also sampled under the identity hook
  (condition "identity", SPEC §8 T11 and §9.4 L3); the generation config used is banked in manifest.json.
- A8 own.n_forward_tokens = tokens in the own-context forwards of one condition.

Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import time

from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus

METRICS_SCHEMA = "neuronoff/metrics/v1"
PLAN_SCHEMA = "neuronoff/plan/v2"
KNOWN_TOL = 0.005
DET_BOUND = 0.0
DET_FAIL_POS = 100
SEED = 20261025
FLOOR_MIN_SHARE = 0.8
IDENTITY_FAIL_SCALE = 1.5
N_BEFORE = 64
TEACHER_CONDITIONS = ("clean", "identity", "ablate_pole", "ablate_ctrl1", "ablate_ctrl2", "ablate_layer")
SAMPLE_CONDITIONS = ("clean", "ablate_pole", "ablate_ctrl1")
KNOWN_RULE = "targets input[:, 1:] from logits[:, :-1] per window, windows in order, BOS prepended when defined"


def stem(pole: str) -> str:
    return pole.replace("+", "p").replace("-", "m")


def finite(o):
    """NaN and inf -> None, recursively (SCHEMA conventions)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    return o


def dump(path, obj):
    from infra.pg.neurons_harvest import dump as atomic_dump
    atomic_dump(path, finite(obj))


def append_jsonl(path, rows):
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(finite(r)) + "\n")


def sign_str(s: int) -> str:
    assert s in (1, -1), f"plan sign must be +1 or -1, got {s!r}"
    return "+" if s == 1 else "-"


def percentile(xs, q):
    import numpy as np
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q)) if len(xs) else float("nan")


def fit(a_ours: list, a_banked: list, sign: int, median_tol: float, p95_tol: float) -> dict:
    """G-fit on paired lists: |a_ours / a_banked - 1| median and 95th percentile, sign mismatches against the pole."""
    rel = [abs(x / b - 1.0) for x, b in zip(a_ours, a_banked)]
    mism = sum(not (x > 0 if sign == 1 else x < 0) for x in a_ours)
    r = {"median_rel_err": percentile(rel, 50), "p95_rel_err": percentile(rel, 95), "n_sign_mismatch": mism}
    r["pass"] = r["median_rel_err"] <= median_tol and r["p95_rel_err"] <= p95_tol and mism == 0
    return r


class Job:
    def __init__(self, out, *, model, tok, tokens, plan: dict, plan_sha256: str, meta: dict, known_tol: float = KNOWN_TOL):
        from infra.pg import neuron_off_core as C
        from infra.pg.neurons_harvest import utc
        self.C, self.utc = C, utc
        self.out = pathlib.Path(out)
        for d in ("poles", "arm_b", "samples"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.model, self.tok, self.tokens, self.plan, self.known_tol = model, tok, tokens, plan, known_tol
        import torch
        if model.device.type == "cuda":
            assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "set CUBLAS_WORKSPACE_CONFIG before CUDA starts"
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(SEED)
        assert plan["schema"] == PLAN_SCHEMA, plan["schema"]
        assert tok.bos_token_id == plan["bos_id"], f"tokenizer BOS {tok.bos_token_id} != plan bos_id {plan['bos_id']}"
        self.W = int(plan["window"])
        self.prefix = [] if tok.bos_token_id is None else [tok.bos_token_id]
        self.excl = len(self.prefix) + 1
        for pid, p in plan["poles"].items():
            assert pid == C.pole_id(p["layer"], p["neuron"], sign_str(p["sign"])), f"pole id {pid} vs its fields"
        S = plan["sampling"]
        gen = C.generation_config(tok, S["temperature"], S["max_new_tokens"], S["top_k"], S["top_p"])
        self.tokens_forwarded, self.forward_s, self.stage_s = 0, 0.0, {}
        self.m = {"schema": METRICS_SCHEMA, "status": "running", "started_utc": utc(), "updated_utc": None,
                  "model_key": plan["model_key"], **{k: meta.get(k) for k in ("repo", "revision", "branch", "device_name",
                                                                              "transformers", "torch", "pg_git_sha")},
                  "plan_sha256": plan_sha256, "gates": None,
                  "progress": {"n_poles_planned": 0, "n_poles_done": 0, "arm_b_done": [], "tokens_forwarded": 0,
                               "forward_s": 0.0, "tokens_per_second": None},
                  "poles": {}, "arm_b": {}}
        self.manifest = {**meta, "plan_sha256": plan_sha256, "dtype": str(model.dtype),
                         "attn_implementation": model.config._attn_implementation,
                         "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "argv": sys.argv,
                         "generation_config": gen.to_dict(), "chunk_positions": C.CHUNK,
                         "deterministic_algorithms": True, "seed": SEED,
                         "started_utc": self.m["started_utc"], "finished_utc": None, "stage_s": self.stage_s}

    # ---------------------------------------------------------------------------------- plumbing
    def save(self):
        pr = self.m["progress"]
        pr["tokens_forwarded"], pr["forward_s"] = self.tokens_forwarded, round(self.forward_s, 3)
        pr["tokens_per_second"] = self.tokens_forwarded / self.forward_s if self.forward_s > 0 else None
        self.m["updated_utc"] = self.utc()
        dump(self.out / "metrics.json", self.m)

    def sync(self):
        if self.model.device.type == "cuda":
            import torch
            torch.cuda.synchronize()

    def fwd(self, ids, abl=None, keep_logits=False, positions=None, targets=None):
        t0 = time.time()
        r = self.C.next_token(self.model, ids, abl, keep_logits, positions, targets)
        self.sync()
        self.forward_s += time.time() - t0
        self.tokens_forwarded += ids.numel()
        return r

    def ids(self, start: int, n: int):
        import torch
        assert 0 <= start and start + n <= self.tokens.size, f"tokens [{start}, {start + n}) outside the corpus"
        return torch.tensor([self.prefix + self.tokens[start:start + n].tolist()])

    def window_prefix(self, t: int):
        """prefix + the corpus window of t from its first token to t inclusive; every position's next corpus token."""
        w0 = (t // self.W) * self.W
        ids = self.ids(w0, t - w0 + 1)
        nxt = ([int(self.tokens[w0])] if self.prefix else []) + [int(x) for x in self.tokens[w0 + 1:t + 2]]
        return ids, list(range(ids.shape[1])), nxt

    def pole(self, pid, mode="pole"):
        p = self.plan["poles"][pid]
        return self.C.PoleAblator(self.model, mode, p["layer"], p["neuron"], sign_str(p["sign"]), exclude_first=self.excl)

    def layer_abl(self, mode, layer):
        return self.C.PoleAblator(self.model, mode, layer, exclude_first=self.excl)

    def dlogit(self, a, b) -> float:
        return float(self.C.max_abs_dlogit(a["logits"], b["logits"]).max())

    def dec(self, ids):
        return self.tok.decode([int(i) for i in ids])

    def side(self, nll, top_id, top_p):
        return {"nll": float(nll), "p_target": math.exp(-float(nll)),
                "top5": [[self.dec([i]), float(p)] for i, p in zip(top_id, top_p)]}

    def sample_line(self, pid, rank, tok_index, before, token, target, clean, ablated):
        """clean / ablated: (nll, top5 ids, top5 p) of the prediction made at `token` for `target`."""
        return {"pole": pid, "rank": rank, "tok": tok_index, "text_before": self.dec(before[-N_BEFORE:]),
                "token": self.dec([token]), "target_str": self.dec([target]),
                "clean": self.side(*clean), "ablated": self.side(*ablated), "dnll": float(ablated[0]) - float(clean[0])}

    def stage(self, name, t0):
        self.stage_s[name] = round(self.stage_s.get(name, 0.0) + time.time() - t0, 2)

    # ---------------------------------------------------------------------------------- gates (SCHEMA §3.1)
    def corpus_nll(self, n_targets: int, abl=None) -> float:
        s, n, w = 0.0, 0, 0
        while n < n_targets:
            nll = self.fwd(self.ids(w * self.W, self.W), abl)["nll"][0].double()
            take = min(n_targets - n, nll.numel())
            s, n, w = s + float(nll[:take].sum()), n + take, w + 1
        return s / n

    def floor_layer(self, layer: int) -> dict:
        down = self.C.decoder_layers(self.model)[layer].mlp.down_proj
        for s in self.plan["floor_slices"]:
            ids = self.ids(0, s)
            cap = {}
            h = down.register_forward_pre_hook(lambda mod, args: cap.__setitem__("a", args[0][0].float()))
            clean = self.fwd(ids, keep_logits=True)
            h.remove()
            a = cap["a"][self.excl:]
            pos, neg = (a > 0).sum(0), (a < 0).sum(0)
            cands = [(int(n), 1) for n in ((pos == 0) & (neg > 0)).nonzero()[:, 0][:1]] + \
                    [(int(n), -1) for n in ((neg == 0) & (pos > 0)).nonzero()[:, 0][:1]]
            if not cands:
                continue
            n, sg = min(cands)
            counts = {1: int(pos[n]), -1: int(neg[n])}

            def read(sign):
                abl = self.C.PoleAblator(self.model, "pole", layer, n, sign_str(sign), exclude_first=self.excl)
                r = self.fwd(ids, abl, keep_logits=True)
                mx = self.dlogit(r, clean)
                return {"n_sign_positions": counts[sign], "max_abs_dlogit": mx,
                        "dnll_mean": float((r["nll"] - clean["nll"]).double().mean()), "pass": mx == 0.0}
            return {"neuron": n, "sign": sg, "slice_tokens": s, **read(sg), "failing_input": {"sign": -sg, **read(-sg)}}
        return {"not_constructible": True, "searched_slices": list(self.plan["floor_slices"])}

    def gates(self, first_a_pole: str):
        t0 = time.time()
        plan, W = self.plan, self.W
        known = plan["known"]
        banked = known["nll_corpus_first_120k"]
        value = self.corpus_nll(known["n_targets"])
        fvalue = self.corpus_nll(known["n_targets"], self.layer_abl("mlp_zero", 0))
        kn = {"value": value, "banked": banked, "abs_diff": abs(value - banked), "tol": self.known_tol,
              "pass": abs(value - banked) <= self.known_tol, "n_targets": known["n_targets"], "rule": KNOWN_RULE,
              "failing_input": {"rule": "same targets, layer 0 whole-MLP ablation on the mask domain", "value": fvalue,
                                "abs_diff": abs(fvalue - banked), "pass": abs(fvalue - banked) <= self.known_tol}}

        self.coll = []
        for w in range(plan["collateral"]["n_windows"]):
            ids = self.ids(w * W, W)
            r = self.fwd(ids)
            self.coll.append({"w": w, "ids": ids, "nll": r["nll"][0], "top5_id": r["top5_id"][0], "top5_p": r["top5_p"][0]})
        n_coll = sum(c["nll"].numel() for c in self.coll)
        coll = {"n_windows": len(self.coll), "n_targets": n_coll,
                "mean_nll": sum(float(c["nll"].double().sum()) for c in self.coll) / n_coll,
                "per_window_mean_nll": [float(c["nll"].double().mean()) for c in self.coll]}

        rep, n_cmp = 0.0, 0
        for w in (0, 1):
            ids = self.ids(w * W, W)
            c1, c2 = self.fwd(ids, keep_logits=True), self.fwd(ids, keep_logits=True)
            rep, n_cmp = max(rep, self.dlogit(c1, c2)), n_cmp + c1["logits"].shape[1]
            if w == 1:
                j = len(self.prefix) + DET_FAIL_POS
                assert j < ids.shape[1] - 1, f"window {W} too short for the G-det failing input at position {DET_FAIL_POS}"
                bad = ids.clone()
                bad[0, j] = (bad[0, j] + 1) % self.model.config.get_text_config().vocab_size
                fail = float(self.C.max_abs_dlogit(self.fwd(bad, keep_logits=True)["logits"][:, j:],
                                                   c1["logits"][:, j:]).max())
            del c1, c2
        det = {"max_abs_dlogit_clean_repeat": rep, "positions_compared": n_cmp, "windows": [0, 1], "bound": DET_BOUND,
               "pass": rep <= DET_BOUND,
               "failing_input": {"rule": f"window 1, token {DET_FAIL_POS} replaced by the next token id",
                                 "max_abs_dlogit": fail, "pass": fail <= DET_BOUND}}

        p0 = plan["poles"][first_a_pole]
        ids, pos, nxt = self.window_prefix(p0["heldout"][0]["tok"])
        c1 = self.fwd(ids, keep_logits=True, positions=pos, targets=nxt)
        bound = self.dlogit(c1, self.fwd(ids, keep_logits=True, positions=pos, targets=nxt))
        down = self.C.decoder_layers(self.model)[p0["layer"]].mlp.down_proj
        h = down.register_forward_pre_hook(lambda mod, args: ((args[0].float() * IDENTITY_FAIL_SCALE).to(args[0].dtype),))
        mx = self.dlogit(self.fwd(ids, keep_logits=True, positions=pos, targets=nxt), c1)
        h.remove()
        del c1
        idf = {"hook": f"a := {IDENTITY_FAIL_SCALE} * a at every neuron of layer {p0['layer']}", "pole": first_a_pole,
               "rank": p0["heldout"][0]["rank"], "clean_repeat": bound, "max_abs_dlogit": mx, "pass": mx == 0.0}

        layers = sorted({p["layer"] for p in plan["poles"].values()})
        floor = {str(l): self.floor_layer(l) for l in layers}
        built = [f for f in floor.values() if not f.get("not_constructible")]
        dump(self.out / "controls.json", {"known_number": kn, "collateral_clean": coll, "determinism": det,
                                          "identity_failing_input": idf, "floor": floor})
        g = {"known_number_pass": kn["pass"] and not kn["failing_input"]["pass"],
             "determinism_pass": det["pass"] and not det["failing_input"]["pass"],
             "identity_failing_input_fails": not idf["pass"],
             "floor": {"n_layers": len(layers), "n_pass": sum(f["pass"] and not f["failing_input"]["pass"] for f in built),
                       "n_not_constructible": len(layers) - len(built)}}
        self.m["gates"] = g
        self.stage("gates", t0)
        ok = g["known_number_pass"] and g["determinism_pass"] and g["identity_failing_input_fails"] and \
            g["floor"]["n_pass"] == len(built) and len(built) >= FLOOR_MIN_SHARE * len(layers)
        if not ok:
            self.m["status"] = "aborted_gate"
            self.save()
            raise SystemExit(f"[neuron_off] gate failure: {json.dumps(finite(g))}")
        self.save()
        print("[neuron_off] gates", json.dumps(finite(g)), flush=True)

    # ---------------------------------------------------------------------------------- Arm B (SCHEMA §3.4-3.5)
    def arm_b(self, e: dict):
        C, S, pid = self.C, self.plan["sampling"], e["pole"]
        layer = self.plan["poles"][pid]["layer"]
        abls = {"clean": None, "identity": self.pole(pid, "identity"), "ablate_pole": self.pole(pid),
                "ablate_ctrl1": self.pole(e["ctrl1"]), "ablate_ctrl2": self.pole(e["ctrl2"]),
                "ablate_layer": self.layer_abl("mlp_zero", layer)}
        f_t, f_s = self.out / "arm_b" / f"{stem(pid)}_teacher.jsonl", self.out / "arm_b" / f"{stem(pid)}_samples.jsonl"
        f_t.write_text("")
        f_s.write_text("")
        rec = {"file_teacher": str(f_t.relative_to(self.out)), "file_samples": str(f_s.relative_to(self.out)),
               "mean_logp_sum": None, "mean_logp_first": None, "n_samples_written": 0, "done": False}
        self.m["arm_b"][pid] = rec
        sums, firsts = {c: [] for c in TEACHER_CONDITIONS}, {c: [] for c in TEACHER_CONDITIONS}
        for j, pr in enumerate(e["prompts"]):
            t0, lines, clean_logits = time.time(), [], None
            for c in TEACHER_CONDITIONS:
                r = C.teacher(self.model, self.tok, pr["prompt"], pr["continuation"], abls[c])
                self.tokens_forwarded += r["prompt_n_tokens"] + len(r["cont_token_ids"])
                cl, a_in = r.pop("cont_logits"), r.pop("a_inputs")
                clean_logits = cl if c == "clean" else clean_logits
                lines.append({"pole": pid, "i": pr["i"], "condition": c, **r,
                              "target_matches_first_token": r["first_token_str"] == pr["target"],
                              "max_abs_dlogit_vs_clean": float(C.max_abs_dlogit(cl[None], clean_logits[None]).max())})
                if c == "ablate_pole":
                    assert len(a_in) == len(r["cont_token_ids"])
                    lines[0]["pole_a_inputs"] = a_in
                sums[c].append(r["logp_sum"])
                firsts[c].append(r["logp_first"])
            append_jsonl(f_t, lines)
            seed = S["base_seed"] + 1000 * pr["i"]
            for c in SAMPLE_CONDITIONS + (("identity",) if j == 0 else ()):
                rows = C.sample(self.model, self.tok, pr["prompt"], abls[c], S["n_samples"], seed, S["temperature"],
                                S["max_new_tokens"], S["top_k"], S["top_p"])
                self.tokens_forwarded += sum(r["n_new_tokens"] for r in rows)
                append_jsonl(f_s, [{"pole": pid, "i": pr["i"], "condition": c, "seed": seed, "k": k, **r}
                                   for k, r in enumerate(rows)])
                rec["n_samples_written"] += len(rows)
            self.sync()
            self.forward_s += time.time() - t0
            rec["mean_logp_sum"] = {c: sum(v) / len(v) for c, v in sums.items()}
            rec["mean_logp_first"] = {c: sum(v) / len(v) for c, v in firsts.items()}
            self.save()
        rec["done"] = True
        self.m["progress"]["arm_b_done"].append(pid)

    # ---------------------------------------------------------------------------------- Arm A (SCHEMA §3.3)
    def arm_a(self, pid: str):
        import torch
        p, W, P = self.plan["poles"][pid], self.W, len(self.prefix)
        rows = p["heldout"]
        assert rows, f"{pid}: no held-out rows"
        abl, zero = self.pole(pid), self.layer_abl("mlp_zero", p["layer"])
        down = self.C.decoder_layers(self.model)[p["layer"]].mlp.down_proj
        w_norm = float(down.weight.detach()[:, p["neuron"]].float().norm())
        pc_plan, rp = self.plan["positive_control"], self.plan["reproduction"]
        pc_ranks = [r["rank"] for r in rows[:pc_plan["n_contexts"]]]
        by_w = {}
        for r in rows:
            by_w.setdefault(r["tok"] // W, []).append(r)
        ctx, pcs, keep, a_next = {}, {}, {}, {}
        n_abl, n_fwd, excl_max, t_pc = 0, 0, 0.0, 0.0
        t_own = time.time()
        for w in sorted(by_w):
            rs = by_w[w]
            ids = self.ids(w * W, max(r["tok"] % W for r in rs) + 1)
            pos = [P + r["tok"] % W for r in rs]
            tgt = [int(self.tokens[r["tok"] + 1]) for r in rs]
            clean, hooked = self.fwd(ids, positions=pos, targets=tgt), self.fwd(ids, abl, positions=pos, targets=tgt)
            n_abl, n_fwd = n_abl + abl.n_edited, n_fwd + ids.numel()
            lead = hooked["a"][0, :self.excl][hooked["fire"][0, :self.excl]]
            excl_max = max(excl_max, float(lead.abs().max()) if lead.numel() else 0.0)
            zeroed = None
            if any(r["rank"] in pc_ranks for r in rs):
                t1 = time.time()
                zeroed = self.fwd(ids, zero, positions=pos, targets=tgt)
                t_pc += time.time() - t1
            for k, r in enumerate(rs):
                t, c = r["tok"], pos[k]
                assert int(ids[0, c]) == int(self.tokens[t]), f"{pid} rank {r['rank']}: context ids misaligned"
                a = float(hooked["a"][0, c])
                cn, an = float(clean["nll"][0, k]), float(hooked["nll"][0, k])
                ctx[r["rank"]] = {"rank": r["rank"], "tok": t, "window": w, "pos": t % W, "target_id": tgt[k],
                                  "target_str": self.dec([tgt[k]]), "a_banked": r["a_banked"], "a_ours": a,
                                  "rel_err": abs(a / r["a_banked"] - 1.0), "clean_nll": cn, "ablated_nll": an,
                                  "dnll": an - cn, "removed_norm": abs(a) * w_norm}
                a_next[r["rank"]] = float(hooked["a_next"][0, c])
                keep[r["rank"]] = {"before": ids[0, :c].tolist(), "token": int(ids[0, c]), "target": tgt[k],
                                   "clean": (cn, clean["top5_id"][0, k], clean["top5_p"][0, k]),
                                   "ablated": (an, hooked["top5_id"][0, k], hooked["top5_p"][0, k])}
                if zeroed is not None and r["rank"] in pc_ranks:
                    zn = float(zeroed["nll"][0, k])
                    pcs[r["rank"]] = {"rank": r["rank"], "clean_nll": cn, "layer_ablated_nll": zn, "dnll": zn - cn}
                    keep[r["rank"]]["layer"] = (zn, zeroed["top5_id"][0, k], zeroed["top5_p"][0, k])
        ctx = [ctx[r["rank"]] for r in rows]

        # G-ident (and the G-pc failing input) on the first held-out window prefix, every position
        ids, pos, nxt = self.window_prefix(rows[0]["tok"])
        c1 = self.fwd(ids, keep_logits=True, positions=pos, targets=nxt)
        rep = self.dlogit(self.fwd(ids, keep_logits=True, positions=pos, targets=nxt), c1)
        i1 = self.fwd(ids, self.pole(pid, "identity"), keep_logits=True, positions=pos, targets=nxt)
        ident = self.dlogit(i1, c1)
        pc_fail_dnll = float(i1["nll"][0, -1] - c1["nll"][0, -1])
        del c1, i1
        t_own = time.time() - t_own - t_pc

        t_coll = time.time()
        s_d = s_c = rn = 0.0
        n = n_abl_c = 0
        per_window, best = [], None
        for cw in self.coll:
            h = self.fwd(cw["ids"], abl)
            d = (h["nll"][0] - cw["nll"]).double()
            per_window.append(float(d.mean()))
            s_d, s_c, n = s_d + float(d.sum()), s_c + float(cw["nll"].double().sum()), n + d.numel()
            n_abl_c += abl.n_edited
            rn += float(h["a"][0][h["edit"][0]].abs().sum()) * w_norm
            j = int(d.abs().argmax())
            if best is None or abs(float(d[j])) > abs(best["d"]):
                best = {"d": float(d[j]), "cw": cw, "j": j, "abl": (float(h["nll"][0, j]), h["top5_id"][0, j], h["top5_p"][0, j])}
        t_coll = time.time() - t_coll

        a_ours, a_bank = [x["a_ours"] for x in ctx], [x["a_banked"] for x in ctx]
        repro = {**fit(a_ours, a_bank, p["sign"], rp["median_rel_tol"], rp["p95_rel_tol"]),
                 "median_rel_tol": rp["median_rel_tol"], "p95_rel_tol": rp["p95_rel_tol"],
                 "n_exact": sum(x["rel_err"] == 0.0 for x in ctx),
                 "failing_input": {"rule": "neuron i+1 (mod width) a at the same positions against a_banked",
                                   **fit([a_next[r["rank"]] for r in rows], a_bank, p["sign"], rp["median_rel_tol"],
                                         rp["p95_rel_tol"])}}
        pc_list = [pcs[k] for k in pc_ranks]
        pc_mean = sum(x["dnll"] for x in pc_list) / len(pc_list)
        lo_hi = (pc_plan["floor_nats"], pc_plan["ceiling_nats"])
        pc = {"contexts": pc_list, "mean_dnll": pc_mean, "floor_nats": lo_hi[0], "ceiling_nats": lo_hi[1],
              "pass": lo_hi[0] <= pc_mean <= lo_hi[1],
              "failing_input": {"hook": "identity", "rank": rows[0]["rank"], "mean_dnll": pc_fail_dnll,
                                "pass": lo_hi[0] <= pc_fail_dnll <= lo_hi[1]}}
        identity = {"rank": rows[0]["rank"], "max_abs_dlogit_identity": ident, "max_abs_dlogit_clean_repeat": rep,
                    "positions_compared": len(pos), "pass": ident == 0.0}
        dn = [x["dnll"] for x in ctx]
        collateral = {"n_targets": n, "clean_mean_nll": s_c / n, "ablated_mean_nll": (s_c + s_d) / n, "dnll": s_d / n,
                      "per_window_dnll": per_window, "n_ablated_positions": n_abl_c, "removed_norm_sum": rn}
        rec = {"pole": pid, "layer": p["layer"], "neuron": p["neuron"], "sign": p["sign"], "w_down_norm": w_norm,
               "own": {"n_windows": len(by_w), "n_forward_tokens": n_fwd, "n_ablated_positions": n_abl,
                       "excluded_positions_max_abs_a": excl_max, "contexts": ctx},
               "positive_control": pc, "identity": identity, "reproduction": repro, "collateral": collateral,
               "timing_s": {"own": round(t_own, 3), "positive_control": round(t_pc, 3), "collateral": round(t_coll, 3)}}
        dump(self.out / "poles" / f"{stem(pid)}.json", rec)

        def line(x, cond="ablated"):
            k = keep[x["rank"]]
            return self.sample_line(pid, x["rank"], x["tok"], k["before"], k["token"], k["target"], k["clean"], k[cond])
        hi, lo = max(ctx, key=lambda x: x["dnll"]), min(ctx, key=lambda x: x["dnll"])
        append_jsonl(self.out / "samples" / "own_ablation.jsonl", [line(hi), line(lo)])
        append_jsonl(self.out / "samples" / "positive_control.jsonl", [line(ctx[0], "layer")])
        cw, j = best["cw"], best["j"]
        ci = cw["ids"][0].tolist()
        append_jsonl(self.out / "samples" / "collateral.jsonl",
                     [self.sample_line(pid, None, cw["w"] * W + j - P if j >= P else None, ci[:j], ci[j], ci[j + 1],
                                       (float(cw["nll"][j]), cw["top5_id"][j], cw["top5_p"][j]), best["abl"])])

        self.m["poles"][pid] = {
            "file": f"poles/{stem(pid)}.json", "own_dnll_mean": sum(dn) / len(dn), "own_dnll_median": percentile(dn, 50),
            "own_clean_nll_mean": sum(x["clean_nll"] for x in ctx) / len(ctx), "own_n_contexts": len(ctx),
            "own_removed_norm_median": percentile([x["removed_norm"] for x in ctx], 50), "w_down_norm": w_norm,
            "coll_dnll": collateral["dnll"], "coll_removed_norm_sum": rn, "coll_n_ablated_positions": n_abl_c,
            "pc_dnll_mean": pc_mean, "identity_max_abs_dlogit": ident,
            "repro_median_rel_err": repro["median_rel_err"], "repro_p95_rel_err": repro["p95_rel_err"],
            "repro_n_sign_mismatch": repro["n_sign_mismatch"],
            "gates": {"reproduction": repro["pass"], "identity": identity["pass"], "positive_control": pc["pass"]}}
        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------------- run
    def run(self, max_poles=None, max_arm_b=None, skip_arm_a=False, skip_arm_b=False):
        plan = self.plan
        assert set(plan["order"]) == set(plan["poles"]), "plan.order and plan.poles name different poles"
        all_a = [q for q in plan["order"] if plan["poles"][q]["arm"] == "A"]
        arm_a = [] if skip_arm_a else all_a[:max_poles]
        arm_b = [] if skip_arm_b else plan["arm_b"][:max_arm_b]
        self.m["progress"]["n_poles_planned"] = len(arm_a) + 3 * len(arm_b)
        dump(self.out / "manifest.json", self.manifest)
        self.save()
        self.gates(all_a[0])
        t0 = time.time()
        for e in arm_b:
            self.arm_b(e)
            self.m["progress"]["n_poles_done"] += 3
            self.save()
            print(f"[neuron_off] arm B {e['pole']}: {json.dumps(finite(self.m['arm_b'][e['pole']]['mean_logp_sum']))}",
                  flush=True)
        self.stage("arm_b", t0)
        t0 = time.time()
        for pid in arm_a:
            self.arm_a(pid)
            self.m["progress"]["n_poles_done"] += 1
            self.save()
            print(f"[neuron_off] arm A {pid}: {json.dumps(finite(self.m['poles'][pid]))}", flush=True)
        self.stage("arm_a", t0)
        self.m["status"] = "complete"
        self.save()
        self.manifest["finished_utc"] = self.utc()
        dump(self.out / "manifest.json", self.manifest)
        return self.m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-key", required=True, choices=["Q4B", "G4", "O7"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokname", required=True, choices=["olmo3", "qwen3", "gemma3"])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--revision", default=None, help="Hub branch to download (O7); absent = the cached snapshot")
    ap.add_argument("--max-poles", type=int, default=None, help="Arm A: the first N poles of plan.order with arm A")
    ap.add_argument("--max-arm-b", type=int, default=None, help="Arm B: the first K entries of plan.arm_b")
    ap.add_argument("--skip-arm-a", action="store_true")
    ap.add_argument("--skip-arm-b", action="store_true")
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"           # before CUDA starts; deterministic cuBLAS (A2)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    import transformers
    from infra.pg import neurons_corpus as NC
    from infra.pg import neurons_harvest as NH
    from infra.pg.matrix_models import vocab_sha256
    assert torch.cuda.is_available(), "no CUDA on this instance"

    raw = fetch(a.plan, "plan.json").read_bytes()
    plan = json.loads(raw)
    assert (plan["model_key"], plan["repo"], plan["branch"], plan["tokname"], plan["corpus"]) == \
        (a.model_key, a.repo, a.revision, a.tokname, a.corpus), "plan and arguments name different models or corpora"
    meta = {"model_key": a.model_key, "repo": a.repo, "branch": a.revision, "device_name": torch.cuda.get_device_name(0),
            "transformers": transformers.__version__, "torch": torch.__version__, "pg_git_sha": os.environ.get("PG_GIT_SHA")}
    if a.revision:
        local, sha, meta["download"] = download(a.repo, a.revision)
    else:
        local, sha = NC.local_snapshot(a.repo)
    assert sha == plan["revision_expected"], f"{a.repo}: snapshot {sha}, plan expects {plan['revision_expected']}"
    meta["revision"] = sha
    corpus = NC.load_tok(fetch_corpus(a.corpus, a.tokname), a.tokname)
    assert corpus["tokens_sha256"] == plan["corpus_tokens_sha256"], "corpus tokens differ from the plan's"
    meta["corpus_tokens_sha256"] = corpus["tokens_sha256"]
    t0 = time.time()
    model, tok = NH.load_model(local, a.model_key)
    meta["load_s"] = round(time.time() - t0, 1)
    assert vocab_sha256(tok) == corpus["record"]["vocab_sha256"], "model and corpus tokenizer vocabularies differ"
    m = Job(a.out, model=model, tok=tok, tokens=corpus["tokens"], plan=plan, plan_sha256=hashlib.sha256(raw).hexdigest(),
            meta=meta).run(a.max_poles, a.max_arm_b, a.skip_arm_a, a.skip_arm_b)
    print("NEURON_OFF_OK", json.dumps({"model_key": a.model_key, "status": m["status"],
                                       "n_poles_done": m["progress"]["n_poles_done"]}), flush=True)


if __name__ == "__main__":
    main()
