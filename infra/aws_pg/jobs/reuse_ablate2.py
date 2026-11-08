"""REUSE heads and output stages (studies/reuse/SPEC_B_heads.md, studies/reuse/SPEC_A_output.md), one O7 checkpoint per call, the stages named in
--stage run in order on one loaded model, each into <out>/<stage>/.

    python -m infra.aws_pg.jobs.reuse_ablate2 --stage heads output --arms sniper --t T_15 --revision stage1-step1413814 --items <sniper_items s3> --corpus <s3> --from-step <S1 step, s3> --out <dir>
    python -m infra.aws_pg.jobs.reuse_ablate2 --stage heads output --arms primary --t T_12 --revision stage1-step320000 --items <primary_items s3> --setting <setting_items s3> --corpus <s3> --from-step <primary T_15 step, s3> --h 20 --r 20 --out <dir>

Q, the foreign pole, is never chosen here: every load reads foreign.json from --from-step (REUSE's own choice: S1's for the sniper, the primary T_15
step's for the primary). Contexts per side capped at 12 by rank. Mean replacement over UNITE's e1 positions at the load.
heads (SPEC B §3), per job: identity; own H(J), cross H(O*), random-matched R_H (per head of H(O*): same layer, outside H(J) and H(O*), nearest in
  ||W_o[:, h] z̄_h||), foreign H_foreign (Q's top heads on Q's contexts), activity-matched R_HJ (the same matching on the mean ||W_o[:, h] z_h|| over J's
  contexts), each at every h of the arm set, read on J. When a layer has no head left to match, the nearest layer below L that has one gives it (the
  job record counts these). Ceiling splits: side 2 read with identity, own H(side 2), cross H(side 1). Planted pairs: identity, own H(side 1), cross
  H(side 2), R_H on the labelled sides; identity, own, cross on side labels shuffled within the pair (seed 0, as reuse_ablate.Job.planted).
output (SPEC A §3), per read set: one capture through the last layer (the readers' g and u, and every downstream a), one identity forward (the hook on
  layer L, nothing written) and one held forward (P at its background mean at window positions); every reader set is read from these. Per job:
  own R(J), cross R(O*), random-matched (per reader of R(O*): same layer, outside both sets, nearest in |ā|), foreign R_foreign (Q's top readers on Q's
  contexts), at every r of the arm set; a reader layer with no neuron left gives the match from the nearest layer above L that has one (counted, as
  for heads). A job at the last layer has no MLP reader and is recorded as skipped. Ceilings and planted pairs as heads.
Arm sets: heads_sniper (h 10, 20, 40) and output_sniper (r 10, 20, 50): every item job at every load; the items' planted pairs and the ceiling splits of
  the jobs with in_s1 at T_15 only. heads_primary and output_primary (h from --h, r from --r): T_15 every job and every distinct own ceiling split; a
  birth checkpoint T_n the jobs born there, each job's own split once per load, and the planted pairs of --setting.
Gates, asserted after their numbers are written so a failure ends the job non-zero: the known number and completeness (reuse_ablate.Job.known), identity,
and the head or reader capture check, each with its failing input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import time

import numpy as np
import torch

from infra.aws_pg.jobs import reuse_ablate as JR
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_ablate2 as RA2
from infra.pg import unite_route as UR

ARMSETS = {"heads_sniper": {"xs": RA2.HS, "sniper": True}, "heads_primary": {"xs": None, "sniper": False},
           "output_sniper": {"xs": RA2.RS, "sniper": True}, "output_primary": {"xs": None, "sniper": False}}
CAP = 12
TOP_SAMPLE = JR.TOP_SAMPLE
ratio = JR.ratio


def reader_stats(a_before, a_id, a_held, tot_d) -> tuple:
    """From the banked arrays, in float64: per context E = sum |a_id - a_held| / sum |a_id| over the set and share = the same numerator over the total
    downstream |a_id - a_held|; the read's effect is the mean over contexts with a_P > 0. Returns (summary, E, share)."""
    a_id, a_held = np.asarray(a_id, np.float64), np.asarray(a_held, np.float64)
    d = np.abs(a_id - a_held).sum(1)
    E, share = d / np.abs(a_id).sum(1), d / np.asarray(tot_d, np.float64)
    pos = np.asarray(a_before) > 0
    nan = float("nan")
    return ({"E_mean": float(E[pos].mean()) if pos.any() else nan, "E_median": float(np.median(E[pos])) if pos.any() else nan,
             "share_mean": float(share[pos].mean()) if pos.any() else nan, "n_pos": int(pos.sum()), "n": int(pos.size)}, E, share)


class Base(JR.Job):
    """reuse_ablate.Job's known number (known), metrics (write), packing (kw, prompts, t_index) and splits (primary_splits), unchanged; the units here."""

    def __init__(self, out, a, stage: str, items: dict, model, tok, corpus: dict, device: str):
        self.out, self.a, self.stage, self.items, self.model, self.tok, self.device = pathlib.Path(out), a, stage, items, model, tok, device
        for d in ("", "falls", "samples"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.tokens, self.offsets = corpus["tokens"], corpus["doc_offsets"]
        self.bos = tok.bos_token_id
        self.tower = NH.text_tower(model)
        self.layers = list(self.tower.layers)
        self.n_layers = len(self.layers)
        self.W = int(self.layers[0].mlp.gate_proj.weight.shape[0])
        self.rows = UR.Rows(model)
        self.hw = a.harvest_window
        self.spec = ARMSETS[f"{stage}_{a.arms}"]
        x = a.h if stage == "heads" else a.r
        assert (self.spec["xs"] is None) == (x is not None), f"{stage} {a.arms}: the frozen {'h' if stage == 'heads' else 'r'} is given for the primary and only there"
        self.xs = tuple(self.spec["xs"]) if self.spec["xs"] else (int(x),)
        self.dec = NH.decoder(tok)
        assert NH.act_name(self.layers[0].mlp.act_fn) == "silu", "the coefficients assume silu"
        self.m = {"stage": stage, "arms": a.arms, "t": a.t, "revision": a.revision, "items_sha256": a.items_sha256, "setting_sha256": a.setting_sha256,
                  "from_step": a.from_step, "harvest_window": self.hw, "window": RA.LENGTH, "cap": CAP, "xs": list(self.xs), "started_utc": NH.utc(),
                  "pg_git_sha": os.environ.get("PG_GIT_SHA"), "n_layers": self.n_layers, "width": self.W, "dtype": str(next(model.parameters()).dtype),
                  "replacement": "mean over UNITE's e1 positions at this checkpoint; BOS left as it is", "timing_s": {}, "tokens_forwarded": 0, "forwards": 0,
                  "token_layers": 0, "jobs": {}, "planted": {}, "ceiling": {}}
        self.id_rows, self.id_fail_rows, self.chk, self.chk_fail = [], [], [], []
        self.write()

    def count(self, toks, depth: int, forwards: bool):
        """Tokens and token-layers (tokens x layers run) of one pass over `toks`; `forwards` also counts its contexts as readout forwards."""
        n = RA.n_forwarded(np.asarray(toks), RA.LENGTH, self.bos, self.hw)
        self.m["tokens_forwarded"] += n
        self.m["token_layers"] += n * depth
        if forwards:
            self.m["forwards"] += len(toks)

    def e1(self):
        return UR.background_positions(self.t_index, self.a.scan_tokens // self.hw * self.hw, self.hw)["e1"]

    def foreign(self) -> dict:
        """Q per job, read from --from-step's foreign.json (REUSE's choice) at every load, copied into this stage's directory."""
        p = pathlib.Path(self.a.from_step) / "foreign.json"
        assert p.exists(), f"REUSE's foreign choice is not at {p}"
        q = json.loads(p.read_text())
        missing = [j["job"] for j in self.items["jobs"] if j["job"] not in q]
        assert not missing, f"foreign.json lacks {missing[:5]}"
        NH.dump(self.out / "foreign.json", q)
        return q

    # -------------------------------------------------------------------------------------- which units a load reads
    def jobs_here(self) -> list:
        jobs = self.items["jobs"]
        if not self.spec["sniper"] and self.a.t != "T_15":
            jobs = [j for j in jobs if j["birth"] == self.t_index]
            assert jobs, f"no primary job is born at {self.a.t}"
        return jobs

    def refs_here(self) -> tuple:
        """(planted pairs, ceiling splits {name: (pole, ceiling)}) read at this load."""
        if self.spec["sniper"]:
            if self.a.t != "T_15":
                return [], {}
            return self.items["planted"], {j["job"]: (j["pole"], j["ceiling"]) for j in self.items["jobs"] if j["ceiling"]["in_s1"]}
        if self.a.t == "T_15":
            return [], self.primary_splits()
        out = {}
        for j in self.jobs_here():
            key = f"{j['pole']}__c{j['ceiling']['c']}"
            if key in out:
                assert out[key][1]["sides_tok"] == j["ceiling"]["sides_tok"], f"{key}: two jobs name this split with different sides"
            out[key] = (j["pole"], j["ceiling"])
        return self.items["planted"], out

    def layers_read(self) -> list:
        planted, splits = self.refs_here()
        return [j["layer"] for j in self.jobs_here()] + [p["layer"] for p in planted] + [UR.pole_unit(p)["layer"] for p, _ in splits.values()]

    def check_gate(self, name: str, scale: str):
        e, w = np.array([r["err"] for r in self.chk]), np.array([r["err"] for r in self.chk_fail])
        tol = RA2.CHECK_TOLS[name]
        self.m[name] = {"max_err": float(e.max()), "median_err": float(np.median(e)), "n_contexts": int(e.size), "n_over_tol": int((e > tol).sum()),
                        "tol": tol, "pass": bool((e <= tol).all()), "failing_input": "the wrong layer",
                        "failing_input_median_err": float(np.median(w)), "failing_input_min_err": float(w.min()),
                        "failing_input_failed": bool(np.median(w) > tol),
                        "rule": f"every context's error <= {tol:g} ({scale}); the failing input fails when its median error > {tol:g}"}

    def run(self):
        self.known()
        self.background()
        q = self.foreign()
        t0, tok0, tl0 = time.time(), self.m["tokens_forwarded"], self.m["token_layers"]
        jobs = self.jobs_here()
        planted, splits = self.refs_here()
        self.m.update({"n_jobs_to_read": len(jobs), "n_ceilings_to_read": len(splits), "n_planted_to_read": len(planted)})
        for i, j in enumerate(jobs):
            self.pole_job(j, q[j["job"]])
            if "label" in j:
                self.m["jobs"][j["job"]]["label"] = j["label"]
            if not self.spec["sniper"] and self.a.t != "T_15":
                self.m["jobs"][j["job"]]["ceiling_split"] = f"{j['pole']}__c{j['ceiling']['c']}"
            self.m["n_jobs_done"] = i + 1
            self.write()
        for i, (name, (pole, c)) in enumerate(splits.items()):
            self.ceiling(name, pole, c)
            self.m["n_ceilings_done"] = i + 1
            self.write()
        for i, p in enumerate(planted):
            self.planted(p)
            self.m["n_planted_done"] = i + 1
            self.write()
        dt = time.time() - t0
        self.m["timing_s"]["units"] = round(dt, 1)
        self.m["units_tokens_per_second"] = (self.m["tokens_forwarded"] - tok0) / dt
        self.m["units_token_layers_per_second"] = (self.m["token_layers"] - tl0) / dt
        self.summarize()
        self.m["status"] = "done"
        self.write()


# ------------------------------------------------------------------------------------------ heads
class HeadsJob(Base):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.hrows = RA2.HeadRows(self.model)
        self.H, self.hd = self.hrows.H, self.hrows.hd
        self.id_fail_arm = f"own_direct_m{max(self.xs)}"
        self.m.update({"n_heads": self.H, "head_dim": self.hd, "n_kv_heads": self.hrows.kv,
                       "job_fall_rule": "mean over contexts with a_P > 0", "matching": "same layer, else the nearest layer below L with a head left (lower first)"})
        self.write()

    def background(self):
        t0 = time.time()
        n = max(self.layers_read())
        e1 = self.e1()
        self.zbar = RA2.background_mean_z(self.model, e1, n, **self.kw(RA.LENGTH))
        self.bnorm = torch.stack([self.hrows.out_norm(l, self.zbar[l][None])[0] for l in range(n)])
        self.count(e1, n, False)
        zb, bn = self.zbar.cpu().numpy(), self.bnorm.cpu().numpy()
        np.savez_compressed(self.out / "zbar.npz", zbar=zb, out_norm=bn, e1=e1)
        NH.dump(self.out / "background.json", {"e1": e1.tolist(), "t_index": self.t_index, "n_layers": n, "out_norm_median_by_layer": np.median(bn, 1).tolist()})
        self.m["background"] = {"n_positions": int(len(e1)), "n_layers": n, "finite": bool(np.isfinite(zb).all()), "out_norm_median": float(np.median(bn))}
        self.m["timing_s"]["background"] = round(time.time() - t0, 1)
        self.write()
        assert self.m["background"]["finite"], "z̄ is not finite"

    def capture(self, toks, unit):
        cap = RA2.capture(self.model, np.asarray(toks), unit["layer"], {unit["layer"]}, **self.kw(RA.LENGTH))
        self.count(toks, unit["layer"] + 1, False)
        return cap

    def attr(self, cap, unit):
        return RA2.direct_H(self.hrows, self.rows, cap, unit)

    def check(self, cap, unit, key):
        e, w = RA2.head_completeness(self.hrows, self.rows, cap, unit), RA2.head_completeness(self.hrows, self.rows, cap, unit, shift=1)
        self.chk += [{"unit": key, "err": float(x)} for x in e]
        self.chk_fail += [{"unit": key, "err": float(x)} for x in w]

    def evaluate(self, key: str, unit: dict, toks, cap: dict, A: torch.Tensor, arms: list) -> dict:
        """`before` and one forward per arm on `toks`; banks falls/<key>.npz in reuse_ablate.Job.evaluate's fields (set_neuron holds the head),
        appends samples/arms.jsonl, returns per-arm summaries."""
        L, H = unit["layer"], self.H
        kw = self.kw(RA.LENGTH)
        toks = np.asarray(toks)
        N, M, T = len(toks), max(self.xs), len(unit["terms"])
        before = RA.readout(self.model, toks, unit, None, **kw)
        self.count(toks, L + 1, True)
        norm = torch.cat([self.hrows.out_norm(l, cap["z"][l]) for l in range(L)], 1)
        nA = len(arms)
        bank = {"tok": toks, "arm": np.array([x["arm"] for x in arms]), "ranking": np.array([x.get("ranking", "") for x in arms]),
                "m": np.array([x.get("m", 0) for x in arms]), "replace": np.array(["mean"] * nA), "off_layer": np.array([x.get("off", 0) for x in arms]),
                "a_before": before["a"], "g_before": before["g"], "u_before": before["u"],
                "a_after": np.full((nA, N), np.nan, np.float32), "g_after": np.full((nA, N, T), np.nan, np.float32), "u_after": np.full((nA, N, T), np.nan, np.float32),
                "set_layer": np.full((nA, M), -1, np.int16), "set_neuron": np.full((nA, M), -1, np.int32), "set_A": np.full((nA, N, M), np.nan, np.float32),
                "set_norm": np.full((nA, N, M), np.nan, np.float32), "set_rnorm": np.full((nA, M), np.nan, np.float32)}
        pr = self.prompts(toks)
        res, samples = {}, []
        for i, arm in enumerate(arms):
            S, ident = arm["set"], arm["arm"] == "identity"
            after = RA2.readout_heads(self.model, toks, unit, RA2.head_ablation(S, self.zbar, L, self.device, identity=ident), **kw)
            self.count(toks, L + 1, True)
            bank["a_after"][i], bank["g_after"][i], bank["u_after"][i] = after["a"], after["g"], after["u"]
            jf, f = RA.job_fall(before["a"], after["a"]), RA.fall(before["a"], after["a"])
            res[arm["arm"]] = {**jf, "ranking": arm.get("ranking"), "m": arm.get("m"), "replace": "mean", "a_before_mean": float(before["a"].mean()),
                               "a_after_mean": float(after["a"].mean()), "max_abs_fall": float(np.nanmax(np.abs(f))),
                               "max_abs_change": float(np.abs(before["a"].astype(np.float64) - after["a"]).max()), "n_off_layer": arm.get("off", 0)}
            if ident:
                self.id_rows += [{"unit": key, "tok": int(t), "fall": float(x), "change": float(abs(b - c))} for t, x, b, c in zip(toks, f, before["a"], after["a"])]
                continue
            if arm["arm"] == self.id_fail_arm:
                self.id_fail_rows += [{"unit": key, "tok": int(t), "fall": float(x)} for t, x in zip(toks, f)]
            flat = torch.tensor([l * H + h for l, h in S], device=self.device)
            ns = len(S)
            bank["set_layer"][i, :ns], bank["set_neuron"][i, :ns] = [l for l, _ in S], [h for _, h in S]
            sA, sn = A[:, flat].cpu().numpy(), norm[:, flat].cpu().numpy()
            rn = self.bnorm[[l for l, _ in S], [h for _, h in S]].cpu().numpy()
            bank["set_A"][i, :, :ns], bank["set_norm"][i, :, :ns], bank["set_rnorm"][i, :ns] = sA, sn, rn
            ctx = []
            for c in range(N):
                top = np.argsort(-np.abs(sA[c]))[:TOP_SAMPLE]
                ctx.append({"tok": int(toks[c]), **pr[c], "a_before": float(before["a"][c]), "a_after": float(after["a"][c]), "fall": float(f[c]),
                            "top_ablated": [{"layer": int(S[k][0]), "head": int(S[k][1]), "A": float(sA[c, k]), "out_norm": float(sn[c, k]), "replacement_out_norm": float(rn[k])}
                                            for k in top]})
            samples.append({"unit": key, "arm": arm["arm"], "m": arm.get("m"), "set_size": ns, "n_off_layer": arm.get("off", 0), "job_fall": jf, "contexts": ctx})
        np.savez_compressed(self.out / "falls" / f"{key}.npz", **bank)
        with open(self.out / "samples" / "arms.jsonl", "a") as fh:
            for s in samples:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        return res

    def pole_job(self, j: dict, q: dict):
        t0 = time.time()
        unit, qunit = UR.pole_unit(j["pole"]), UR.pole_unit(q["pole"])
        L = unit["layer"]
        assert L == qunit["layer"] == j["layer"], (j["pole"], q["pole"])
        Jt, Ot, Qt = (np.asarray(x)[:CAP] for x in (j["J_tok"], j["O_tok"], q["tok"]))
        key = f"{j['pole']}__c{j['c']}"
        capJ = self.capture(Jt, unit)
        AJ = self.attr(capJ, unit)
        AO = self.attr(self.capture(Ot, unit), unit).sum(0)
        AQ = self.attr(self.capture(Qt, qunit), qunit).sum(0)
        self.check(capJ, unit, key)
        bg = self.bnorm[:L].cpu().numpy()
        act = torch.stack([self.hrows.out_norm(l, capJ["z"][l]).mean(0) for l in range(L)]).cpu().numpy()
        M = max(self.xs)
        D = {s: RA.top_m(sc, M, self.H) for s, sc in (("J", AJ.sum(0)), ("O", AO), ("F", AQ))}
        arms = [{"arm": "identity", "set": []}]
        s_, last4, off = {}, {}, {}
        for x in self.xs:
            own, cross, fo = D["J"][:x], D["O"][:x], D["F"][:x]
            rnd, off_r = RA2.matched(cross, own, bg, range(L))
            act_, off_a = RA2.matched(cross, own, act, range(L))
            arms += [{"arm": f"{nm}_direct_m{x}", "set": S, "ranking": "direct", "m": x, "off": o}
                     for nm, S, o in (("own", own, 0), ("cross", cross, 0), ("random", rnd, off_r), ("foreign", fo, 0), ("active", act_, off_a))]
            s_[f"direct_m{x}"] = len(set(own) & set(cross)) / x
            last4[f"direct_m{x}"] = float(np.mean([l >= L - 4 for l, _ in own]))
            off[f"direct_m{x}"] = {"random": off_r, "active": off_a}
        res = self.evaluate(key, unit, Jt, capJ, AJ, arms)
        I, canc, fc, ac = {}, {}, {}, {}
        for x in self.xs:
            own, cr, rd, fo, av = (res[f"{nm}_direct_m{x}"] for nm in ("own", "cross", "random", "foreign", "active"))
            I[f"direct_m{x}"] = ratio(cr["mean"], own["mean"])
            canc[f"direct_m{x}"] = RA.cancel(own["mean"], rd["mean"], own["n_pos"])
            fc[f"cross_minus_foreign_m{x}"] = cr["mean"] - fo["mean"]
            ac[f"cross_minus_active_m{x}"] = cr["mean"] - av["mean"]
        self.m["jobs"][j["job"]] = {"pole": j["pole"], "layer": L, "tercile": j["tercile"], "birth": j["birth"], "n_J": int(len(Jt)), "n_O": int(len(Ot)),
                                    "foreign": {"pole": q["pole"], "c": q["c"], "n": int(len(Qt))}, "falls": res, "I": I, "s": s_, "cancel": canc,
                                    "foreign_contrast": {"direct": fc}, "active_contrast": {"direct": ac}, "last4_share": last4, "off_layer": off,
                                    "falls_path": f"falls/{key}.npz", "seconds": round(time.time() - t0, 1)}

    def ref_arms(self, own: list, cross: list, random: bool, L: int) -> list:
        arms = [{"arm": "identity", "set": []}]
        for x in self.xs:
            arms += [{"arm": f"own_direct_m{x}", "set": own[:x], "ranking": "direct", "m": x}, {"arm": f"cross_direct_m{x}", "set": cross[:x], "ranking": "direct", "m": x}]
            if random:
                rnd, o = RA2.matched(cross[:x], own[:x], self.bnorm[:L].cpu().numpy(), range(L))
                arms.append({"arm": f"random_direct_m{x}", "set": rnd, "ranking": "direct", "m": x, "off": o})
        return arms

    def ref_I(self, res: dict, own: list, cross: list) -> tuple:
        return ({f"m{x}": ratio(res[f"cross_direct_m{x}"]["mean"], res[f"own_direct_m{x}"]["mean"]) for x in self.xs},
                {f"m{x}": len(set(own[:x]) & set(cross[:x])) / x for x in self.xs})

    def ceiling(self, name: str, pole: str, c: dict):
        t0 = time.time()
        unit = UR.pole_unit(pole)
        s1, s2 = (np.asarray(x)[:CAP] for x in c["sides_tok"])
        A1 = self.attr(self.capture(s1, unit), unit).sum(0)
        cap2 = self.capture(s2, unit)
        A2 = self.attr(cap2, unit)
        M = max(self.xs)
        D1, D2 = RA.top_m(A1, M, self.H), RA.top_m(A2.sum(0), M, self.H)
        key = f"{pole}__c{c['c']}_ceiling_side2"
        res = self.evaluate(key, unit, s2, cap2, A2, self.ref_arms(D2, D1, False, unit["layer"]))
        I_m, s_m = self.ref_I(res, D2, D1)
        self.m["ceiling"][name] = {"pole": pole, "c": c["c"], "n_sides": [int(len(s1)), int(len(s2))], "s": s_m[f"m{M}"], "s_m": s_m, "I_ceil": I_m[f"m{M}"], "I_ceil_m": I_m,
                                   "side2": {"falls": res, "I": I_m[f"m{M}"], "I_m": I_m, "n_read": int(len(s2)), "falls_path": f"falls/{key}.npz"},
                                   "seconds": round(time.time() - t0, 1)}

    def planted(self, p: dict):
        t0 = time.time()
        unit = UR.planted_unit(p)
        Jt, Ot = np.asarray(p["J_tok"])[:CAP], np.asarray(p["O_tok"])[:CAP]
        pool = np.concatenate([Jt, Ot])
        cap = self.capture(pool, unit)
        A = self.attr(cap, unit)
        nJ, M = len(Jt), max(self.xs)
        perm = np.random.default_rng(0).permutation(len(pool))
        out = {}
        for tag, jr, orr in (("", np.arange(nJ), np.arange(nJ, len(pool))), ("shuffled", np.sort(perm[:nJ]), np.sort(perm[nJ:]))):
            tj, to = (torch.as_tensor(x, device=A.device) for x in (jr, orr))
            DJ, DO = RA.top_m(A[tj].sum(0), M, self.H), RA.top_m(A[to].sum(0), M, self.H)
            key = f"{p['unit']}__planted" + (f"_{tag}" if tag else "")
            res = self.evaluate(key, unit, pool[jr], RA2.cap_rows(cap, jr), A[tj], self.ref_arms(DJ, DO, not tag, unit["layer"]))
            I_m, s_m = self.ref_I(res, DJ, DO)
            out[tag or "labels"] = {"falls": res, "I": I_m[f"m{M}"], "s": s_m[f"m{M}"], "I_m": I_m, "s_m": s_m, "n_J": int(len(jr)), "n_O": int(len(orr)),
                                    "falls_path": f"falls/{key}.npz"}
        self.m["planted"][p["unit"]] = {"layer": p["layer"], "a": p["a"], "b": p["b"], "lam": p["lam"], **out, "seconds": round(time.time() - t0, 1)}

    def summarize(self):
        J = self.m["jobs"]
        med = lambda xs: float(np.nanmedian(xs)) if len(xs) else float("nan")
        tab = {}
        for x in self.xs:
            k = f"direct_m{x}"
            tab[k] = {nm: med([v["falls"][f"{nm}_{k}"]["mean"] for v in J.values()]) for nm in ("own", "cross", "random", "foreign", "active")}
            tab[k].update({"I_median": med([v["I"][k] for v in J.values()]), "s_median": med([v["s"][k] for v in J.values()]),
                           "n_cancelled": sum(v["cancel"][k]["cancelled"] for v in J.values()),
                           "cross_minus_foreign": med([v["foreign_contrast"]["direct"][f"cross_minus_foreign_m{x}"] for v in J.values()]),
                           "cross_minus_active": med([v["active_contrast"]["direct"][f"cross_minus_active_m{x}"] for v in J.values()]),
                           "last4_share_median": med([v["last4_share"][k] for v in J.values()]),
                           "n_off_layer": {nm: sum(v["off_layer"][k][nm] for v in J.values()) for nm in ("random", "active")}})
        idf, idc = np.array([r["fall"] for r in self.id_rows]), np.array([r["change"] for r in self.id_rows])
        fl = np.array([r["fall"] for r in self.id_fail_rows])
        self.m["identity"] = {"max_abs_fall": float(np.nanmax(np.abs(idf))), "max_abs_change": float(idc.max()), "n_contexts": int(idf.size),
                              "n_over_tol": int((np.abs(idf) > RA.ID_TOL).sum()), "tol": RA.ID_TOL, "pass": bool((np.abs(idf) <= RA.ID_TOL).all()),
                              "failing_input": self.id_fail_arm, "failing_input_n_over_tol": int((np.abs(fl) > RA.ID_TOL).sum()), "failing_input_n": int(fl.size),
                              "failing_input_median_abs_fall": float(np.median(np.abs(fl))) if fl.size else float("nan"),
                              "failing_input_failed": bool((np.abs(fl) > RA.ID_TOL).any())}
        self.check_gate("head_check", "relative to the pole's max |g| (|u|) at the job's contexts, as unite_route.completeness")
        summ = {"by_ranking_m": tab}
        if self.m["planted"]:
            summ["planted_I_m"] = {u: v["labels"]["I_m"] for u, v in self.m["planted"].items()}
            summ["planted_I_shuffled_m"] = {u: v["shuffled"]["I_m"] for u, v in self.m["planted"].items()}
        if self.m["ceiling"]:
            summ["ceiling_I_m"] = {u: v["I_ceil_m"] for u, v in self.m["ceiling"].items()}
        self.m["summary"] = summ
        self.write()
        assert self.m["identity"]["pass"], f"identity failed: {self.m['identity']}"
        assert self.m["identity"]["failing_input_failed"], f"identity's failing input passed: {self.m['identity']}"
        assert self.m["head_check"]["pass"], f"head capture check failed: {self.m['head_check']}"
        assert self.m["head_check"]["failing_input_failed"], f"head capture check's failing input passed: {self.m['head_check']}"


# ------------------------------------------------------------------------------------------ output
class OutputJob(Base):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.m.update({"effect_rule": "E(S; x) = sum_S |a_id - a_held| / sum_S |a_id| (identity forward); job effect = mean over contexts with a_P > 0",
                       "share_rule": "share(S; x) = sum_S |a_id - a_held| / sum over every MLP neuron above L of |a_id - a_held|",
                       "identity_rule": "per context max over every MLP neuron above L of |a_identity - a_capture| / max |a_capture|"})
        self.write()

    def background(self):
        t0 = time.time()
        e1 = self.e1()
        self.abar = RA.background_mean_a(self.model, e1, self.n_layers, **self.kw(RA.LENGTH))
        self.count(e1, self.n_layers, False)
        ab = self.abar.cpu().numpy()
        np.savez_compressed(self.out / "abar.npz", abar=ab, e1=e1)
        NH.dump(self.out / "background.json", {"e1": e1.tolist(), "t_index": self.t_index, "n_layers": self.n_layers, "abar_abs_median_by_layer": np.median(np.abs(ab), 1).tolist()})
        self.m["background"] = {"n_positions": int(len(e1)), "n_layers": self.n_layers, "finite": bool(np.isfinite(ab).all()), "abar_abs_median": float(np.median(np.abs(ab)))}
        self.m["timing_s"]["background"] = round(time.time() - t0, 1)
        self.write()
        assert self.m["background"]["finite"], "ā is not finite"

    def capture(self, toks, unit):
        cap = RA2.capture_down(self.model, np.asarray(toks), unit, None, True, **self.kw(RA.LENGTH))
        self.count(toks, self.n_layers, False)
        return cap

    def top(self, score, x: int, L: int) -> list:
        return [(L + 1 + l, k) for l, k in RA.top_m(score, x, self.W)]

    def flat(self, S: list, L: int) -> list:
        return [(l - L - 1) * self.W + k for l, k in S]

    def check(self, cap, unit, key):
        e, w = RA2.reader_completeness(self.rows, cap, unit), RA2.reader_completeness(self.rows, cap, unit, shift=1)
        self.chk += [{"unit": key, "err": float(x)} for x in e]
        self.chk_fail += [{"unit": key, "err": float(x)} for x in w]

    def forwards(self, toks, unit: dict, cap: dict) -> dict:
        """The identity and held forwards on `toks` (cap: the capture of the same contexts) and every downstream a of each; per context the identity
        value (identity forward against the capture) and its failing input (the held forward against the capture), each over the context's max |a|."""
        L = unit["layer"]
        kw = self.kw(RA.LENGTH)
        capI = RA2.capture_down(self.model, np.asarray(toks), unit, RA2.hold_dict(unit, self.abar, self.device, identity=True), False, **kw)
        capH = RA2.capture_down(self.model, np.asarray(toks), unit, RA2.hold_dict(unit, self.abar, self.device), False, **kw)
        for _ in range(2):
            self.count(toks, self.n_layers, True)
        a_cap, a_id, a_held = (RA2.down_a(c, L, self.n_layers) for c in (cap, capI, capH))
        scale = a_cap.abs().amax(1)
        return {"a_cap": a_cap, "a_id": a_id, "a_held": a_held, "id_rel": ((a_id - a_cap).abs().amax(1) / scale).cpu().numpy(),
                "held_rel": ((a_held - a_cap).abs().amax(1) / scale).cpu().numpy(), "tot_d": (a_id - a_held).abs().sum(1), "tot_a": a_id.abs().sum(1)}

    def reads(self, key: str, unit: dict, toks, idx, cap: dict, F: dict, C: torch.Tensor, sets: dict) -> dict:
        """Every reader set of `sets` {arm: [(layer, neuron)]} on the rows `idx` of F, C and cap. Banks falls/<key>.npz, appends samples, returns summaries."""
        L = unit["layer"]
        toks, idx = np.asarray(toks), np.asarray(idx)
        i = torch.as_tensor(idx, device=self.device)
        N, R, nS = len(idx), max(self.xs), len(sets)
        ns = torch.tensor([t["n"] for t in unit["terms"]], device=self.device)
        ws = torch.tensor([t["w"] * t["sign"] for t in unit["terms"]], device=self.device, dtype=torch.float32)
        a_before = (cap["a"][L][i][:, ns] * ws).sum(1).cpu().numpy()
        a_after = float((self.abar[L, ns].float() * ws).sum())
        a_cap, a_id, a_held, Ci = F["a_cap"][i], F["a_id"][i], F["a_held"][i], C[i]
        tot_d, tot_a = F["tot_d"][i], F["tot_a"][i]
        bank = {"tok": toks[idx], "a_before": a_before, "a_after": np.full(N, a_after, np.float32), "arm": np.array(list(sets)),
                "m": np.array([int(k.rsplit("_r", 1)[1]) for k in sets]), "set_layer": np.full((nS, R), -1, np.int16), "set_neuron": np.full((nS, R), -1, np.int32),
                "set_C": np.full((nS, N, R), np.nan, np.float32), "set_a_cap": np.full((nS, N, R), np.nan, np.float32), "set_a_id": np.full((nS, N, R), np.nan, np.float32),
                "set_a_held": np.full((nS, N, R), np.nan, np.float32), "total_abs_delta": tot_d.cpu().numpy(), "total_abs_a": tot_a.cpu().numpy(),
                "id_max_rel": F["id_rel"][idx], "held_max_rel": F["held_rel"][idx]}
        pr = self.prompts(toks[idx])
        self.id_rows += [{"unit": key, "tok": int(t), "rel": float(x)} for t, x in zip(bank["tok"], bank["id_max_rel"])]
        self.id_fail_rows += [{"unit": key, "tok": int(t), "rel": float(x)} for t, x in zip(bank["tok"], bank["held_max_rel"])]
        res, samples = {}, []
        for s, (arm, S) in enumerate(sets.items()):
            fl = torch.tensor(self.flat(S, L), device=self.device)
            k = len(S)
            bank["set_layer"][s, :k], bank["set_neuron"][s, :k] = [l for l, _ in S], [n for _, n in S]
            for f, v in (("set_C", Ci), ("set_a_cap", a_cap), ("set_a_id", a_id), ("set_a_held", a_held)):
                bank[f][s, :, :k] = v[:, fl].cpu().numpy()
            st, E, share = reader_stats(a_before, bank["set_a_id"][s, :, :k], bank["set_a_held"][s, :, :k], bank["total_abs_delta"])
            res[arm] = {**st, "m": int(bank["m"][s])}
            sC = bank["set_C"][s, :, :k]
            ctx = []
            for c in range(N):
                top = np.argsort(-np.abs(sC[c]))[:TOP_SAMPLE]
                ctx.append({"tok": int(toks[idx][c]), **pr[c], "a_before": float(a_before[c]), "a_after": a_after, "E": float(E[c]), "share": float(share[c]),
                            "top_readers": [{"layer": int(S[q][0]), "neuron": int(S[q][1]), "C": float(sC[c, q]), "a_id": float(bank["set_a_id"][s, c, q]),
                                             "a_held": float(bank["set_a_held"][s, c, q])} for q in top]})
            samples.append({"unit": key, "arm": arm, "m": int(bank["m"][s]), "set_size": k, "E_mean": res[arm]["E_mean"], "share_mean": res[arm]["share_mean"], "contexts": ctx})
        np.savez_compressed(self.out / "falls" / f"{key}.npz", **bank)
        with open(self.out / "samples" / "arms.jsonl", "a") as fh:
            for s in samples:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        return res

    def logit_top(self, unit: dict, caps: list) -> list:
        """Per capture: the top 50 vocabulary items of P's direct logit direction, signed by the sum over its contexts of s_L raw_a (the final norm's
        1 / rms is a positive scale per context, so it does not change the sign)."""
        L, n = unit["layer"], unit["terms"][0]["n"]
        u = RA2.logit_direction(self.model, self.rows, L, n)
        return [torch.topk(u if float((c["s"][L] * c["a"][L][:, n]).sum()) >= 0 else -u, RA2.TOP_LOGIT).indices.tolist() for c in caps]

    def pole_job(self, j: dict, q: dict):
        t0 = time.time()
        unit, qunit = UR.pole_unit(j["pole"]), UR.pole_unit(q["pole"])
        L = unit["layer"]
        assert L == qunit["layer"] == j["layer"], (j["pole"], q["pole"])
        base = {"pole": j["pole"], "layer": L, "tercile": j["tercile"], "birth": j["birth"]}
        if L == self.n_layers - 1:
            self.m["jobs"][j["job"]] = {**base, "skipped": "no MLP layer above L", "seconds": 0.0}
            return
        Jt, Ot, Qt = (np.asarray(x)[:CAP] for x in (j["J_tok"], j["O_tok"], q["tok"]))
        key = f"{j['pole']}__c{j['c']}"
        capJ, capO = self.capture(Jt, unit), self.capture(Ot, unit)
        CJ = RA2.direct_C(self.rows, capJ, unit)
        SO = RA2.direct_C(self.rows, capO, unit).abs().sum(0)
        SQ = RA2.direct_C(self.rows, self.capture(Qt, qunit), qunit).abs().sum(0)
        self.check(capJ, unit, key)
        F = self.forwards(Jt, unit, capJ)
        M = max(self.xs)
        D = {s: self.top(sc, M, L) for s, sc in (("J", CJ.abs().sum(0)), ("O", SO), ("F", SQ))}
        sets, s_, depth, within4, off = {}, {}, {}, {}, {}
        for x in self.xs:
            own, cross = D["J"][:x], D["O"][:x]
            rnd, off[f"r{x}"] = RA2.matched(cross, own, self.abar.abs().cpu().numpy(), range(L + 1, self.n_layers))
            sets.update({f"own_r{x}": own, f"cross_r{x}": cross, f"random_r{x}": rnd, f"foreign_r{x}": D["F"][:x]})
            s_[f"r{x}"] = len(set(own) & set(cross)) / x
            depth[f"r{x}"] = float(np.median([l - L for l, _ in own]))
            within4[f"r{x}"] = float(np.mean([l - L <= 4 for l, _ in own]))
        res = self.reads(key, unit, Jt, np.arange(len(Jt)), capJ, F, CJ, sets)
        I, canc, fc = {}, {}, {}
        for x in self.xs:
            own, cr, rd, fo = (res[f"{nm}_r{x}"] for nm in ("own", "cross", "random", "foreign"))
            I[f"r{x}"] = ratio(cr["E_mean"], own["E_mean"])
            canc[f"r{x}"] = RA.cancel(own["E_mean"], rd["E_mean"], own["n_pos"])
            fc[f"cross_minus_foreign_r{x}"] = cr["E_mean"] - fo["E_mean"]
        tJ, tO = self.logit_top(unit, [capJ, capO])
        self.m["jobs"][j["job"]] = {**base, "n_J": int(len(Jt)), "n_O": int(len(Ot)), "foreign": {"pole": q["pole"], "c": q["c"], "n": int(len(Qt))}, "reads": res,
                                    "I": I, "s": s_, "cancel": canc, "foreign_contrast": fc, "depth_median": depth, "within4_share": within4, "off_layer": off,
                                    "logit_top50_J": tJ, "logit_top50_O": tO, "logit_overlap": len(set(tJ) & set(tO)) / RA2.TOP_LOGIT,
                                    "falls_path": f"falls/{key}.npz", "seconds": round(time.time() - t0, 1)}

    def ceiling(self, name: str, pole: str, c: dict):
        t0 = time.time()
        unit = UR.pole_unit(pole)
        L = unit["layer"]
        if L == self.n_layers - 1:
            self.m["ceiling"][name] = {"pole": pole, "c": c["c"], "skipped": "no MLP layer above L"}
            return
        s1, s2 = (np.asarray(x)[:CAP] for x in c["sides_tok"])
        C1 = RA2.direct_C(self.rows, self.capture(s1, unit), unit).abs().sum(0)
        cap2 = self.capture(s2, unit)
        C2 = RA2.direct_C(self.rows, cap2, unit)
        key = f"{pole}__c{c['c']}_ceiling_side2"
        F = self.forwards(s2, unit, cap2)
        M = max(self.xs)
        D1, D2 = self.top(C1, M, L), self.top(C2.abs().sum(0), M, L)
        sets = {}
        for x in self.xs:
            sets.update({f"own_r{x}": D2[:x], f"cross_r{x}": D1[:x]})
        res = self.reads(key, unit, s2, np.arange(len(s2)), cap2, F, C2, sets)
        I_m = {f"r{x}": ratio(res[f"cross_r{x}"]["E_mean"], res[f"own_r{x}"]["E_mean"]) for x in self.xs}
        s_m = {f"r{x}": len(set(D2[:x]) & set(D1[:x])) / x for x in self.xs}
        self.m["ceiling"][name] = {"pole": pole, "c": c["c"], "n_sides": [int(len(s1)), int(len(s2))], "I_ceil_m": I_m, "s_m": s_m,
                                   "side2": {"reads": res, "n_read": int(len(s2)), "falls_path": f"falls/{key}.npz"}, "seconds": round(time.time() - t0, 1)}

    def planted(self, p: dict):
        t0 = time.time()
        unit = UR.planted_unit(p)
        L = unit["layer"]
        if L == self.n_layers - 1:
            self.m["planted"][p["unit"]] = {"layer": L, "skipped": "no MLP layer above L"}
            return
        Jt, Ot = np.asarray(p["J_tok"])[:CAP], np.asarray(p["O_tok"])[:CAP]
        pool = np.concatenate([Jt, Ot])
        cap = self.capture(pool, unit)
        C = RA2.direct_C(self.rows, cap, unit)
        F = self.forwards(pool, unit, cap)
        nJ, M = len(Jt), max(self.xs)
        perm = np.random.default_rng(0).permutation(len(pool))
        out = {}
        for tag, jr, orr in (("", np.arange(nJ), np.arange(nJ, len(pool))), ("shuffled", np.sort(perm[:nJ]), np.sort(perm[nJ:]))):
            tj, to = (torch.as_tensor(x, device=C.device) for x in (jr, orr))
            DJ, DO = self.top(C[tj].abs().sum(0), M, L), self.top(C[to].abs().sum(0), M, L)
            sets = {}
            for x in self.xs:
                sets.update({f"own_r{x}": DJ[:x], f"cross_r{x}": DO[:x]})
                if not tag:
                    sets[f"random_r{x}"] = RA2.matched(DO[:x], DJ[:x], self.abar.abs().cpu().numpy(), range(L + 1, self.n_layers))[0]
            key = f"{p['unit']}__planted" + (f"_{tag}" if tag else "")
            res = self.reads(key, unit, pool, jr, cap, F, C, sets)
            out[tag or "labels"] = {"reads": res, "I_m": {f"r{x}": ratio(res[f"cross_r{x}"]["E_mean"], res[f"own_r{x}"]["E_mean"]) for x in self.xs},
                                    "s_m": {f"r{x}": len(set(DJ[:x]) & set(DO[:x])) / x for x in self.xs}, "n_J": int(len(jr)), "n_O": int(len(orr)),
                                    "falls_path": f"falls/{key}.npz"}
        self.m["planted"][p["unit"]] = {"layer": L, "a": p["a"], "b": p["b"], "lam": p["lam"], **out, "seconds": round(time.time() - t0, 1)}

    def summarize(self):
        J = {k: v for k, v in self.m["jobs"].items() if "skipped" not in v}
        med = lambda xs: float(np.nanmedian(xs)) if len(xs) else float("nan")
        tab = {}
        for x in self.xs:
            k = f"r{x}"
            tab[k] = {nm: med([v["reads"][f"{nm}_{k}"]["E_mean"] for v in J.values()]) for nm in ("own", "cross", "random", "foreign")}
            tab[k].update({"own_share": med([v["reads"][f"own_{k}"]["share_mean"] for v in J.values()]), "random_share": med([v["reads"][f"random_{k}"]["share_mean"] for v in J.values()]),
                           "I_median": med([v["I"][k] for v in J.values()]), "s_median": med([v["s"][k] for v in J.values()]),
                           "n_cancelled": sum(v["cancel"][k]["cancelled"] for v in J.values()),
                           "cross_minus_foreign": med([v["foreign_contrast"][f"cross_minus_foreign_{k}"] for v in J.values()]),
                           "depth_median": med([v["depth_median"][k] for v in J.values()]), "within4_share_median": med([v["within4_share"][k] for v in J.values()])})
        idr, fl = np.array([r["rel"] for r in self.id_rows]), np.array([r["rel"] for r in self.id_fail_rows])
        self.m["identity"] = {"max_rel": float(idr.max()), "n_contexts": int(idr.size), "n_over_tol": int((idr > RA.ID_TOL).sum()), "tol": RA.ID_TOL,
                              "pass": bool((idr <= RA.ID_TOL).all()), "failing_input": "the held forward in place of the identity forward",
                              "failing_input_n_over_tol": int((fl > RA.ID_TOL).sum()), "failing_input_n": int(fl.size),
                              "failing_input_median_rel": float(np.median(fl)), "failing_input_failed": bool((fl > RA.ID_TOL).any())}
        self.check_gate("reader_check", "relative to the max over the contexts and the neurons of layer L+1 of the weights-only dg (du)")
        summ = {"by_r": tab, "n_jobs_read": len(J), "n_jobs_skipped": len(self.m["jobs"]) - len(J)}
        if self.m["planted"]:
            summ["planted_I_m"] = {u: v["labels"]["I_m"] for u, v in self.m["planted"].items() if "skipped" not in v}
            summ["planted_I_shuffled_m"] = {u: v["shuffled"]["I_m"] for u, v in self.m["planted"].items() if "skipped" not in v}
        if self.m["ceiling"]:
            summ["ceiling_I_m"] = {u: v["I_ceil_m"] for u, v in self.m["ceiling"].items() if "skipped" not in v}
        self.m["summary"] = summ
        self.write()
        assert self.m["identity"]["pass"], f"identity failed: {self.m['identity']}"
        assert self.m["identity"]["failing_input_failed"], f"identity's failing input passed: {self.m['identity']}"
        assert self.m["reader_check"]["pass"], f"reader capture check failed: {self.m['reader_check']}"
        assert self.m["reader_check"]["failing_input_failed"], f"reader capture check's failing input passed: {self.m['reader_check']}"


STAGES = {"heads": HeadsJob, "output": OutputJob}


def run_load(out, a, items: dict, model, tok, corpus: dict, device: str, info: dict | None = None) -> dict:
    """Every stage of a.stage in order on one loaded model, each into <out>/<stage>/; returns {stage: its metrics}."""
    res = {}
    for stage in a.stage:
        job = STAGES[stage](pathlib.Path(out) / stage, a, stage, items, model, tok, corpus, device)
        if info:
            job.m.update(info)
            job.write()
        job.run()
        res[stage] = job.m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return res


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, nargs="+", choices=sorted(STAGES))
    ap.add_argument("--arms", required=True, choices=["sniper", "primary"])
    ap.add_argument("--t", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--setting", default=None, help="primary: setting_items.json, whose planted pairs the birth loads read")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--from-step", required=True, help="the REUSE step whose foreign.json names Q (a local dir or an S3 prefix)")
    ap.add_argument("--h", type=int, default=None, help="heads_primary: the frozen h")
    ap.add_argument("--r", type=int, default=None, help="output_primary: the frozen r")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--full-batch", type=int, default=8)
    ap.add_argument("--scan-tokens", type=int, default=20_570_743)
    ap.add_argument("--harvest-window", type=int, default=UR.HARVEST_WINDOW)
    return ap


def main():
    a = parser().parse_args()
    assert JR.REVISION[a.t] == a.revision, f"{a.t} is {JR.REVISION[a.t]}, not {a.revision}"
    assert len(set(a.stage)) == len(a.stage), a.stage
    assert (a.arms == "primary") == (a.setting is not None), "--setting is the primary's planted pairs, and only its"
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    if a.from_step.startswith("s3://"):
        a.from_step = str(fetch(a.from_step.rstrip("/") + "/foreign.json", f"reuse2_{a.arms}_{a.t}_from_step/foreign.json").parent)
    items_path = fetch(a.items, f"reuse2_{a.arms}_{a.t}_items.json")
    a.items_sha256 = hashlib.sha256(pathlib.Path(items_path).read_bytes()).hexdigest()
    items = json.loads(pathlib.Path(items_path).read_text())
    a.setting_sha256 = None
    if a.arms == "primary":
        setting_path = fetch(a.setting, f"reuse2_{a.arms}_{a.t}_setting.json")
        a.setting_sha256 = hashlib.sha256(pathlib.Path(setting_path).read_bytes()).hexdigest()
        assert "planted" not in items, "the primary items carry planted pairs; the primary reads them from --setting only"
        items["planted"] = json.loads(pathlib.Path(setting_path).read_text())["planted"]
        grid = {f"T_{int(g[1:3])}": "stage1-" + g.split("_")[1] for g in items["sources"]["grid"]}
        assert grid[a.t] == a.revision, f"the items' grid has {grid[a.t]} at {a.t}, not {a.revision}"
    t0 = time.time()
    local, sha, rec = download("allenai/Olmo-3-1025-7B", a.revision)
    corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
    t1 = time.time()
    model, tok = NH.load_model(local, "O7")
    model.requires_grad_(False)
    info = {"download": rec, "download_and_corpus_s": round(t1 - t0, 1), "load_s": round(time.time() - t1, 1), "sha": sha,
            "device_name": torch.cuda.get_device_name(0), "transformers": __import__("transformers").__version__, "torch": torch.__version__}
    res = run_load(a.out, a, items, model, tok, corpus, "cuda", info)
    del model
    torch.cuda.empty_cache()
    repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / "models--allenai--Olmo-3-1025-7B"
    shutil.rmtree(repo_dir)
    assert not repo_dir.exists()
    print("REUSE_ABLATE2_OK", json.dumps({"t": a.t, "arms": a.arms, "stages": list(res)}), flush=True)


if __name__ == "__main__":
    main()
