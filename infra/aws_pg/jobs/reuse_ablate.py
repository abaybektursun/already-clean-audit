"""REUSE job, one O7 checkpoint per call: the sniper stage (studies/reuse/SPEC.md §7), the primary stage (§9 step 4, §13.1) and the amend2 stage
(studies/reuse/SPEC2.md §3).

    python -m infra.aws_pg.jobs.reuse_ablate --stage sniper --arms s1 --t T_15 --revision stage1-step1413814 --items <s3> --corpus <s3> --out <dir>
    python -m infra.aws_pg.jobs.reuse_ablate --stage sniper --arms s2 --t T_10 --revision stage1-step100000 --items <s3> --corpus <s3> --from-step /work/out/s1 --out <dir>
    python -m infra.aws_pg.jobs.reuse_ablate --stage primary --arms primary --t T_12 --revision stage1-step320000 --items <s3> --corpus <s3> --from-step /work/out/t15 --out <dir>
    python -m infra.aws_pg.jobs.reuse_ablate --stage amend2 --arms amend2_birth --t T_12 --revision stage1-step320000 --items <primary_items s3> --setting <setting_items s3> --corpus <s3> --out <dir>

S1 (T_15), per sniper job: identity; own D(J) and cross D(O*) at m 20, 50, 100 under the direct and the gradient ranking; random-matched R at each m and
ranking; D_foreign at m 50 under both rankings; zero replacement of own at m 50 direct. Three planted pairs on a_plant (identity, own, cross, R at m 50
direct, and the room gate's failing input: side labels shuffled within the pair). Three ceiling splits (identity, own and cross between the sides, m 50
direct). S2 (T_10): identity, own and cross at m 50 under both rankings, R, D_foreign; Q is S1's choice, read from --from-step.
S3 (SPEC §7), every side capped at 12 contexts by rank, direct ranking, mean replacement:
  s3_t15: the 56 planted pairs (identity, own, cross, R at m 20 and 50; side labels shuffled within the pair, own and cross at both m), the setting
          set's ceiling splits read on side 2 (identity, own D(side 2), cross D(side 1), R at m 20 and 50), then per planted pair 40 count-matched
          draws: sides subsampled to a primary job's (|J|, |O*|), drivers from the subsampled sides, own and cross read on the 12 J contexts.
  s3_t10: the 6 sniper jobs at m 20: identity, own, cross, R, D_foreign; Q is S1's choice (--from-step, a local dir or an S3 prefix).
primary (SPEC §13.1 item 1), m 50, direct ranking, mean replacement, J capped at 12 by rank; the arm functions are the sniper code's, the stage only
  selects which jobs run at a load. T_15: every job of primary_items.json, arms identity, own, cross, R, D_foreign; Q is chosen here. A birth
  checkpoint T_n: only the jobs with birth == n, the same five arms (Q read from --from-step, the T_15 step), plus the job's own ceiling split read on
  side 2 (identity, own D(side 2), cross D(side 1), no R). Jobs sharing a pole and a ceiling c share one split, read once per load under the key
  "<pole>__c<ceiling c>", written into each job record as ceiling_split; each job record also carries the item's label.
amend2 (SPEC2 §3), m 50, direct ranking, mean replacement, every side capped at 12 by rank; no existing arm changes. Planted pairs come from
  --setting (setting_items.json -> planted), jobs and splits from --items (primary_items.json). No step reads T_15 state, so no --from-step.
  amend2_t15 (T_15): B2 for every job: R_J on J, a_P before and one arm `rj_direct_m50` (no identity arm); B3: every distinct own ceiling split
          of the primary jobs (129 keys for 150 jobs) read on side 2 (identity, own D(side 2), cross D(side 1)), as the primary birth ceilings.
  amend2_birth (T_n): B2 for the jobs born at T_n; B1: the 56 planted pairs as S3 read them at m 50 (identity, own D(side 1), cross D(side 2),
          random-matched R on the full sides; side labels shuffled within the pair: identity, own, cross), no count-matched draws.
  R_J: for each k in D(O*), the neuron of k's layer outside D(J), D(O*) and the neurons already picked, nearest to k in mean |a| at the marked
  token over J's contexts, read from the unablated capture of J (a at every layer below L, one row per context), deterministic. The job npz banks
  D(J) (own_layer, own_neuron), D(O*) (cross_layer, cross_neuron: the matched k, in order), and the mean |a| over J of each R_J neuron
  (rj_energy) and of its matched k (match_energy).
`Job.run` takes a loaded model, so the dry run drives the same code on a tiny random OLMo 3. metrics.json is rewritten after every unit; falls/ and
samples/ are written per unit. Gates assert after their numbers are written, so a failed gate ends the job with a non-zero status.
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

from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import unite_route as UR
from infra.pg import unite_stats as US

# the O7 grid (primary_items.json -> sources.grid); T_4, T_5 and T_14 hold no primary birth and are not loaded (journal, before §13.1)
REVISION = {"T_4": "stage1-step5000", "T_5": "stage1-step8000", "T_6": "stage1-step12000", "T_7": "stage1-step20000", "T_8": "stage1-step35000",
            "T_9": "stage1-step60000", "T_10": "stage1-step100000", "T_11": "stage1-step180000", "T_12": "stage1-step320000", "T_13": "stage1-step560000",
            "T_14": "stage1-step1000000", "T_15": "stage1-step1413814"}
KNOWN_CTX = 16
# ms: the jobs' m; ref_ms: the planted pairs' and ceilings' m; cap: contexts per side, by rank (None: as the items give them); draws: count-matched
# planted draws per pair; id_fail_arm: the identity gate's failing input.
_BASE = {"ms": (), "rankings": ("direct",), "zero": False, "jobs": False, "planted": False, "ceiling": None, "ref_ms": (), "ceiling_sides": ("side2",),
         "ceiling_random": False, "m_foreign": None, "cap": None, "draws": 0, "id_fail_arm": "own_direct_m50"}
ARMSETS = {"s1": {**_BASE, "ms": (20, 50, 100), "rankings": ("direct", "grad"), "zero": True, "jobs": True, "planted": True, "ceiling": "jobs",
                  "ref_ms": (50,), "ceiling_sides": ("side2", "side1"), "m_foreign": 50},
           "s2": {**_BASE, "ms": (50,), "rankings": ("direct", "grad"), "jobs": True, "m_foreign": 50},
           "s3_t15": {**_BASE, "planted": True, "ceiling": "setting", "ref_ms": (20, 50), "ceiling_random": True, "cap": 12, "draws": 40},
           "s3_t10": {**_BASE, "ms": (20,), "jobs": True, "m_foreign": 20, "cap": 12, "id_fail_arm": "own_direct_m20"},
           "primary": {**_BASE, "ms": (50,), "jobs": True, "m_foreign": 50, "cap": 12, "ceiling": "primary", "ref_ms": (50,)},
           "amend2_t15": {**_BASE, "jobs": True, "cap": 12, "ceiling": "all_primary", "ref_ms": (50,)},
           "amend2_birth": {**_BASE, "jobs": True, "planted": True, "cap": 12, "ref_ms": (50,)}}
M_AMEND2 = 50
TOP_SAMPLE = 5
DRAW_SEED = 0


def ratio(x: float, y: float) -> float:
    return x / y if y != 0 else float("nan")


class Job:
    def __init__(self, out, a, items: dict, model, tok, corpus: dict, device: str):
        self.out, self.a, self.items, self.model, self.tok, self.device = pathlib.Path(out), a, items, model, tok, device
        for d in ("", "falls", "samples"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.tokens, self.offsets = corpus["tokens"], corpus["doc_offsets"]
        self.bos = tok.bos_token_id
        self.tower = NH.text_tower(model)
        self.layers = list(self.tower.layers)
        self.W = int(self.layers[0].mlp.gate_proj.weight.shape[0])
        self.rows = UR.Rows(model)
        self.hw = a.harvest_window
        self.spec = ARMSETS[a.arms]
        assert (a.stage == "primary") == (a.arms == "primary"), f"stage {a.stage} with arm set {a.arms}"
        assert (a.stage == "amend2") == a.arms.startswith("amend2"), f"stage {a.stage} with arm set {a.arms}"
        assert a.stage != "amend2" or (a.arms == "amend2_t15") == (a.t == "T_15"), f"arm set {a.arms} at {a.t}"
        self.dec = NH.decoder(tok)
        assert NH.act_name(self.layers[0].mlp.act_fn) == "silu", "the coefficients assume silu"
        self.m = {"stage": a.stage, "arms": a.arms, "t": a.t, "revision": a.revision, "items_sha256": a.items_sha256, "harvest_window": self.hw,
                  "window": RA.LENGTH, "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"), "n_layers": len(self.layers), "width": self.W,
                  "dtype": str(next(model.parameters()).dtype), "replacement": "mean over UNITE's e1 positions at this checkpoint; BOS left as it is",
                  "job_fall_rule": "mean over contexts with a_P > 0", "timing_s": {}, "tokens_forwarded": 0, "forwards": 0,
                  "jobs": {}, "planted": {}, "ceiling": {}}
        if a.stage == "amend2":
            self.m["setting_sha256"] = a.setting_sha256
        self.lin = []
        self.id_rows, self.id_fail_rows = [], []
        self.split_sides = {}
        self.write()

    def write(self):
        self.m["updated_utc"] = NH.utc()
        NH.dump(self.out / "metrics.json", self.m)

    def kw(self, length, batch=None):
        return dict(tokens=self.tokens, length=length, bos=self.bos, batch=batch or (self.a.batch if length is not None else self.a.full_batch), hw=self.hw, device=self.device)

    @property
    def t_index(self):
        return int(self.a.t.split("_")[1])

    # -------------------------------------------------------------------------------------- the known number, as unite_route.Job.known
    def known(self, n_ctx: int = KNOWN_CTX):
        t0 = time.time()
        k = self.items["known"]
        out = {}
        for i, p in enumerate(k["poles"]):
            L, n, _ = NH.parse_pole(p)
            toks = np.asarray(k["tok"][i])[:n_ctx]
            cap = UR.capture(self.model, toks, L, {L}, **self.kw(None))
            a = cap["a"][L][:, n].cpu().numpy()
            b, w = np.asarray(k["a_bank"][self.a.t][i], float)[:n_ctx], np.asarray(k["a_wrong"][self.a.t][i], float)[:n_ctx]
            c = UR.completeness(self.rows, cap, L, n)
            cw = UR.completeness(self.rows, cap, L, n, shift=1)
            out[p] = {"known_ratio": US.known_ratio(a, b), "wrong_ratio": US.known_ratio(a, w), "err_g_median": float(np.median(c["err_g"])),
                      "err_u_median": float(np.median(c["err_u"])), "wrong_layer_err_g_median": float(np.median(cw["err_g"])),
                      "wrong_layer_err_u_median": float(np.median(cw["err_u"])),
                      "samples": [{"tok": int(toks[j]), **NH.context(self.tokens, self.offsets, int(toks[j]), self.dec), "a_ours": float(a[j]), "a_bank": float(b[j])} for j in range(2)]}
            self.m["tokens_forwarded"] += RA.n_forwarded(toks, None, self.bos, self.hw)
        ratio = float(np.median([v["known_ratio"] for v in out.values()]))
        wr = float(np.median([v["wrong_ratio"] for v in out.values()]))
        errs = [v[x] for v in out.values() for x in ("err_g_median", "err_u_median")]
        werrs = [v[x] for v in out.values() for x in ("wrong_layer_err_g_median", "wrong_layer_err_u_median")]
        self.m["known_number"] = {**US.gate_known(ratio, wr), "per_pole": {p: {x: v[x] for x in v if x != "samples"} for p, v in out.items()}, "n_ctx": n_ctx,
                                  "wrong_rule": k["wrong_rule"], "failing_input": f"a_wrong at {self.a.t}: {k['wrong_rule']}"}
        self.m["completeness"] = US.gate_complete(float(np.median(errs)), float(np.median(werrs)))
        self.m["timing_s"]["known"] = round(time.time() - t0, 1)
        self.write()
        NH.dump(self.out / "samples" / "known_number.json", out)
        assert self.m["known_number"]["pass"], f"known number failed: {ratio}"
        assert self.m["known_number"]["failing_input_failed"], f"known number's failing input passed: {wr}"
        assert self.m["completeness"]["pass"], f"completeness failed: {self.m['completeness']}"
        assert self.m["completeness"]["failing_input_failed"], f"completeness's failing input passed: {self.m['completeness']}"

    # -------------------------------------------------------------------------------------- ā and the foreign pole
    def background(self):
        t0 = time.time()
        Ls = ([j["layer"] for j in self.items["jobs"]] if self.spec["jobs"] else []) + ([p["layer"] for p in self.items["planted"]] if self.spec["planted"] else []) \
            + ([c["layer"] for c in self.items["ceilings"]] if self.spec["ceiling"] == "setting" else [])
        n_layers = max(Ls)
        sets = UR.background_positions(self.t_index, self.a.scan_tokens // self.hw * self.hw, self.hw)
        self.abar = RA.background_mean_a(self.model, sets["e1"], n_layers, **self.kw(RA.LENGTH))
        self.m["tokens_forwarded"] += RA.n_forwarded(sets["e1"], RA.LENGTH, self.bos, self.hw)
        ab = self.abar.cpu().numpy()
        np.savez_compressed(self.out / "abar.npz", abar=ab, e1=sets["e1"])
        NH.dump(self.out / "background.json", {"e1": sets["e1"].tolist(), "t_index": self.t_index, "n_layers": n_layers,
                                               "abar_abs_median_by_layer": np.median(np.abs(ab), 1).tolist(), "abar_abs_max_by_layer": np.abs(ab).max(1).tolist()})
        self.m["background"] = {"n_positions": int(len(sets["e1"])), "n_layers": n_layers, "finite": bool(np.isfinite(ab).all()), "abar_abs_median": float(np.median(np.abs(ab)))}
        self.m["timing_s"]["background"] = round(time.time() - t0, 1)
        self.write()
        assert self.m["background"]["finite"], "ā is not finite"

    def foreign(self) -> dict:
        """Q per job: the item pole of P's layer and sign nearest in ||G||^2 + ||U||^2 at T_15, P excluded. Chosen at T_15; any other checkpoint reads it."""
        if self.a.t != "T_15":
            p = pathlib.Path(self.a.from_step) / "foreign.json"
            assert p.exists(), f"the foreign choice of T_15 is not at {p}"
            q = json.loads(p.read_text())
            NH.dump(self.out / "foreign.json", q)
            return q
        out = {}
        for j in self.items["jobs"]:
            mlp = self.layers[j["layer"]].mlp
            load = lambda n: float(mlp.gate_proj.weight[n].float().pow(2).sum() + mlp.up_proj.weight[n].float().pow(2).sum())
            lp = load(j["neuron"])
            cands = [{**c, "load": load(NH.parse_pole(c["pole"])[1])} for c in j["foreign_candidates"]]
            assert cands, f"{j['job']}: no foreign candidate"
            best = min(cands, key=lambda c: (abs(c["load"] - lp), c["pole"]))
            out[j["job"]] = {"pole": best["pole"], "c": best["c"], "tok": best["tok"], "load": best["load"], "P_load": lp,
                             "candidates": {c["pole"]: c["load"] for c in cands}}
        NH.dump(self.out / "foreign.json", out)
        return out

    # -------------------------------------------------------------------------------------- one read set, all its arms
    def prompts(self, toks):
        x, pos = UR.pack(self.tokens, np.asarray(toks), RA.LENGTH, self.bos, self.hw, "cpu")
        return [{"prompt": self.dec(x[i, :int(pos[i]) + 1].tolist()), "marked_token": self.dec([int(x[i, int(pos[i])])]), "n_tokens": int(pos[i]) + 1} for i in range(len(toks))]

    def evaluate(self, key: str, unit: dict, toks, cap: dict, A: torch.Tensor, Ag, arms: list, linear: bool, extra: dict | None = None) -> dict:
        """`before` and one forward per arm on the read contexts `toks`. Banks falls/<key>.npz (with `extra` arrays, if any), appends samples/arms.jsonl,
        returns per-arm summaries."""
        L = unit["layer"]
        kw = self.kw(RA.LENGTH)
        toks = np.asarray(toks)
        N, M, T = len(toks), RA.MS[-1], len(unit["terms"])
        before = RA.readout(self.model, toks, unit, None, **kw)
        nfw = RA.n_forwarded(toks, RA.LENGTH, self.bos, self.hw)
        all_a = torch.cat([cap["a"][l] for l in range(L)], 1)
        loads = self.rows.loads(L, unit["terms"][0]["n"]) if linear else None
        nA = len(arms)
        bank = {"tok": toks, "arm": np.array([x["arm"] for x in arms]), "ranking": np.array([x.get("ranking", "") for x in arms]),
                "m": np.array([x.get("m", 0) for x in arms]), "replace": np.array([x["replace"] for x in arms]),
                "a_before": before["a"], "g_before": before["g"], "u_before": before["u"],
                "a_after": np.full((nA, N), np.nan, np.float32), "g_after": np.full((nA, N, T), np.nan, np.float32), "u_after": np.full((nA, N, T), np.nan, np.float32),
                "set_layer": np.full((nA, M), -1, np.int16), "set_neuron": np.full((nA, M), -1, np.int32), "set_r": np.full((nA, M), np.nan, np.float32),
                "set_A": np.full((nA, N, M), np.nan, np.float32), "set_a": np.full((nA, N, M), np.nan, np.float32), "set_Ag": np.full((nA, N, M), np.nan, np.float32),
                "lin_sum_A": np.full((nA, N), np.nan, np.float32), "lin_sum_A_c": np.full((nA, N), np.nan, np.float32),
                "lin_pred_g": np.full((nA, N), np.nan, np.float32), "lin_pred_u": np.full((nA, N), np.nan, np.float32)}
        pr = self.prompts(toks)
        res = {}
        samples = []
        for i, arm in enumerate(arms):
            S, ident = arm["set"], arm["arm"].startswith("identity")
            abl = RA.ablation_dict(S, arm["replace"], self.abar, L, self.device, identity=ident)
            after = RA.readout(self.model, toks, unit, abl, **kw)
            nfw += RA.n_forwarded(toks, RA.LENGTH, self.bos, self.hw)
            bank["a_after"][i], bank["g_after"][i], bank["u_after"][i] = after["a"], after["g"], after["u"]
            jf = RA.job_fall(before["a"], after["a"])
            f = RA.fall(before["a"], after["a"])
            res[arm["arm"]] = {**jf, "ranking": arm.get("ranking"), "m": arm.get("m"), "replace": arm["replace"], "a_before_mean": float(before["a"].mean()),
                               "a_after_mean": float(after["a"].mean()), "max_abs_fall": float(np.nanmax(np.abs(f))),
                               "max_abs_change": float(np.abs(before["a"].astype(np.float64) - after["a"]).max())}
            if ident:
                self.id_rows += [{"unit": key, "tok": int(t), "fall": float(x), "change": float(abs(b - c))} for t, x, b, c in zip(toks, f, before["a"], after["a"])]
                continue
            if arm["arm"] == self.spec["id_fail_arm"]:
                self.id_fail_rows += [{"unit": key, "tok": int(t), "fall": float(x)} for t, x in zip(toks, f)]
            flat = torch.tensor([l * self.W + k for l, k in S], device=self.device)
            r = torch.stack([abl[l][1][abl[l][0].tolist().index(k)] for l, k in S]).float()
            ns = len(S)
            bank["set_layer"][i, :ns] = [l for l, _ in S]
            bank["set_neuron"][i, :ns] = [k for _, k in S]
            bank["set_r"][i, :ns] = r.cpu().numpy()
            sA, sa = A[:, flat].cpu().numpy(), all_a[:, flat].cpu().numpy()
            bank["set_A"][i, :, :ns], bank["set_a"][i, :, :ns] = sA, sa
            if Ag is not None:
                bank["set_Ag"][i, :, :ns] = Ag[:, flat].cpu().numpy()
            if linear:
                lt = RA.linear_terms(self.rows, cap, unit, S, r, self.W, loads)
                sum_direct = sA.sum(1)
                assert np.allclose(lt["sum_A"], sum_direct, rtol=1e-3, atol=1e-3 * np.abs(sum_direct).max() + 1e-6), "linear terms disagree with the attribution"
                for x in ("sum_A", "sum_A_c", "pred_g", "pred_u"):
                    bank[f"lin_{x}"][i] = lt[x]
                for c in range(N):
                    self.lin.append({"unit": key, "arm": arm["arm"], "tok": int(toks[c]), "sum_A": float(lt["sum_A"][c]), "sum_A_c": float(lt["sum_A_c"][c]),
                                     "pred_g": float(lt["pred_g"][c]), "pred_u": float(lt["pred_u"][c]),
                                     "d_a": float(before["a"][c] - after["a"][c]), "d_g": float(before["g"][c, 0] - after["g"][c, 0]), "d_u": float(before["u"][c, 0] - after["u"][c, 0])})
            ctx = []
            for c in range(N):
                top = np.argsort(-np.abs(sA[c]))[:TOP_SAMPLE]
                ctx.append({"tok": int(toks[c]), **pr[c], "a_before": float(before["a"][c]), "a_after": float(after["a"][c]), "fall": float(f[c]),
                            "top_ablated": [{"layer": int(S[q][0]), "neuron": int(S[q][1]), "A": float(sA[c, q]), "a": float(sa[c, q]), "r": float(r[q])} for q in top]})
            samples.append({"unit": key, "arm": arm["arm"], "ranking": arm.get("ranking"), "m": arm.get("m"), "replace": arm["replace"], "set_size": ns,
                            "job_fall": jf, "contexts": ctx})
        if extra:
            assert not set(extra) & set(bank), sorted(set(extra) & set(bank))
            bank.update(extra)
        np.savez_compressed(self.out / "falls" / f"{key}.npz", **bank)
        with open(self.out / "samples" / "arms.jsonl", "a") as fh:
            for s in samples:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        self.m["tokens_forwarded"] += nfw
        self.m["forwards"] += N * (1 + nA)
        return res

    # -------------------------------------------------------------------------------------- the units
    def capture(self, toks, unit):
        return UR.capture(self.model, np.asarray(toks), unit["layer"], {unit["layer"]}, **self.kw(RA.LENGTH))

    def grad(self, toks, unit):
        return RA.grad_attr_window(self.model, np.asarray(toks), unit, **self.kw(RA.LENGTH, self.a.grad_batch))

    def pole_job(self, j: dict, q: dict):
        t0 = time.time()
        unit, qunit = UR.pole_unit(j["pole"]), UR.pole_unit(q["pole"])
        assert unit["layer"] == qunit["layer"] == j["layer"], (j["pole"], q["pole"])
        Jt, Ot, Qt = (np.asarray(x)[:self.spec["cap"]] for x in (j["J_tok"], j["O_tok"], q["tok"]))
        fm = self.spec["m_foreign"]
        ms, M = self.spec["ms"], max(self.spec["ms"] + (fm,))
        capJ = self.capture(Jt, unit)
        AJ = RA.direct_A(self.rows, capJ, unit, self.W)
        AO = RA.direct_A(self.rows, self.capture(Ot, unit), unit, self.W).sum(0)
        AQ = RA.direct_A(self.rows, self.capture(Qt, qunit), qunit, self.W).sum(0)
        score = {"direct": {"J": AJ.sum(0), "O": AO, "F": AQ}}
        GJ = None
        if "grad" in self.spec["rankings"]:
            GJ = self.grad(Jt, unit)
            score["grad"] = {"J": GJ.sum(0), "O": self.grad(Ot, unit).sum(0), "F": self.grad(Qt, qunit).sum(0)}
        self.m["tokens_forwarded"] += RA.n_forwarded(np.concatenate([Jt, Ot, Qt]), RA.LENGTH, self.bos, self.hw) * len(score)
        D = {rk: {s: RA.top_m(score[rk][s], M, self.W) for s in ("J", "O", "F")} for rk in self.spec["rankings"]}
        arms = [{"arm": "identity", "set": [], "replace": "mean"}]
        overlap = {}
        for rk in self.spec["rankings"]:
            for m in ms:
                own, cross = D[rk]["J"][:m], D[rk]["O"][:m]
                arms += [{"arm": f"own_{rk}_m{m}", "set": own, "replace": "mean", "ranking": rk, "m": m},
                         {"arm": f"cross_{rk}_m{m}", "set": cross, "replace": "mean", "ranking": rk, "m": m},
                         {"arm": f"random_{rk}_m{m}", "set": RA.random_matched(cross, own, self.abar), "replace": "mean", "ranking": rk, "m": m}]
                overlap[f"{rk}_m{m}"] = len(set(own) & set(cross)) / m
            arms.append({"arm": f"foreign_{rk}_m{fm}", "set": D[rk]["F"][:fm], "replace": "mean", "ranking": rk, "m": fm})
            overlap[f"foreign_{rk}_m{fm}"] = len(set(D[rk]["J"][:fm]) & set(D[rk]["F"][:fm])) / fm
            overlap[f"cross_foreign_{rk}_m{fm}"] = len(set(D[rk]["O"][:fm]) & set(D[rk]["F"][:fm])) / fm
        if self.spec["zero"]:
            arms.append({"arm": "own_direct_m50_zero", "set": D["direct"]["J"][:50], "replace": "zero", "ranking": "direct", "m": 50})
        key = f"{j['pole']}__c{j['c']}"
        res = self.evaluate(key, unit, Jt, capJ, AJ, GJ, arms, linear=True)
        I, canc, spec_ = {}, {}, {}
        for rk in self.spec["rankings"]:
            for m in ms:
                own, cr, rd = (res[f"{x}_{rk}_m{m}"] for x in ("own", "cross", "random"))
                I[f"{rk}_m{m}"] = ratio(cr["mean"], own["mean"])
                canc[f"{rk}_m{m}"] = RA.cancel(own["mean"], rd["mean"], own["n_pos"])
            if fm in ms:
                fo = res[f"foreign_{rk}_m{fm}"]
                spec_[rk] = {f"cross_minus_foreign_m{fm}": res[f"cross_{rk}_m{fm}"]["mean"] - fo["mean"], f"I_foreign_m{fm}": ratio(fo["mean"], res[f"own_{rk}_m{fm}"]["mean"])}
        self.m["jobs"][j["job"]] = {"pole": j["pole"], "layer": j["layer"], "tercile": j["tercile"], "birth": j["birth"], "n_J": int(len(Jt)), "n_O": int(len(Ot)),
                                    "foreign": {"pole": q["pole"], "c": q["c"], "n": int(len(Qt))}, "falls": res, "I": I, "s": overlap, "cancel": canc,
                                    "foreign_contrast": spec_, "falls_path": f"falls/{key}.npz", "seconds": round(time.time() - t0, 1)}

    def active_job(self, j: dict):
        """SPEC2 §3 B2: D(J) and D(O*) at m 50 direct as `pole_job` ranks them; R_J matched to D(O*) in mean |a| over J's contexts from the
        unablated capture of J; a_P before and the one arm R_J, read on J."""
        t0 = time.time()
        unit, m = UR.pole_unit(j["pole"]), M_AMEND2
        L = unit["layer"]
        assert L == j["layer"], (j["pole"], j["layer"])
        Jt, Ot = (np.asarray(x)[:self.spec["cap"]] for x in (j["J_tok"], j["O_tok"]))
        capJ = self.capture(Jt, unit)
        AJ = RA.direct_A(self.rows, capJ, unit, self.W)
        AO = RA.direct_A(self.rows, self.capture(Ot, unit), unit, self.W).sum(0)
        self.m["tokens_forwarded"] += RA.n_forwarded(np.concatenate([Jt, Ot]), RA.LENGTH, self.bos, self.hw)
        DJ, DO = RA.top_m(AJ.sum(0), m, self.W), RA.top_m(AO, m, self.W)
        act = torch.stack([capJ["a"][l].abs().mean(0) for l in range(L)]).cpu().numpy().astype(np.float64)   # [L, W] mean |a_k| over J
        RJ = RA.random_matched(DO, DJ, act)
        e_rj, e_k = act[tuple(np.array(RJ).T)], act[tuple(np.array(DO).T)]
        extra = {"own_layer": np.array([l for l, _ in DJ], np.int16), "own_neuron": np.array([k for _, k in DJ], np.int32),
                 "cross_layer": np.array([l for l, _ in DO], np.int16), "cross_neuron": np.array([k for _, k in DO], np.int32),
                 "rj_energy": e_rj.astype(np.float32), "match_energy": e_k.astype(np.float32)}
        key = f"{j['pole']}__c{j['c']}"
        arm = {"arm": f"rj_direct_m{m}", "set": RJ, "replace": "mean", "ranking": "direct", "m": m}
        res = self.evaluate(key, unit, Jt, capJ, AJ, None, [arm], linear=True, extra=extra)
        self.m["jobs"][j["job"]] = {"pole": j["pole"], "layer": L, "tercile": j["tercile"], "birth": j["birth"], "label": j["label"], "n_J": int(len(Jt)),
                                    "n_O": int(len(Ot)), "falls": res, "s": len(set(DJ) & set(DO)) / m,
                                    "rj": {"m": m, "n": len(RJ), "energy_median": float(np.median(e_rj)), "match_energy_median": float(np.median(e_k)),
                                           "rel_gap_median": float(np.median(np.abs(e_rj - e_k) / e_k))},
                                    "falls_path": f"falls/{key}.npz", "seconds": round(time.time() - t0, 1)}

    def ref_arms(self, own: list, cross: list, random: bool) -> list:
        """identity, then own and cross (and R matched to cross) at each of the references' m, direct ranking, mean replacement."""
        arms = [{"arm": "identity", "set": [], "replace": "mean"}]
        for m in self.spec["ref_ms"]:
            arms += [{"arm": f"own_direct_m{m}", "set": own[:m], "replace": "mean", "ranking": "direct", "m": m},
                     {"arm": f"cross_direct_m{m}", "set": cross[:m], "replace": "mean", "ranking": "direct", "m": m}]
            if random:
                arms.append({"arm": f"random_direct_m{m}", "set": RA.random_matched(cross[:m], own[:m], self.abar), "replace": "mean", "ranking": "direct", "m": m})
        return arms

    def ref_I(self, res: dict, own: list, cross: list) -> tuple:
        """I = cross / own and s = |own & cross| / m at each of the references' m."""
        ms = self.spec["ref_ms"]
        return ({f"m{m}": ratio(res[f"cross_direct_m{m}"]["mean"], res[f"own_direct_m{m}"]["mean"]) for m in ms},
                {f"m{m}": len(set(own[:m]) & set(cross[:m])) / m for m in ms})

    def planted_sides(self, p: dict):
        unit = UR.planted_unit(p)
        Jt, Ot = np.asarray(p["J_tok"])[:self.spec["cap"]], np.asarray(p["O_tok"])[:self.spec["cap"]]
        pool = np.concatenate([Jt, Ot])
        cap = self.capture(pool, unit)
        return unit, Jt, Ot, pool, cap, RA.direct_A(self.rows, cap, unit, self.W)

    def planted(self, p: dict):
        t0 = time.time()
        unit, Jt, Ot, pool, cap, A = self.planted_sides(p)
        nJ, M = len(Jt), max(self.spec["ref_ms"])
        out = {}
        perm = np.random.default_rng(0).permutation(len(pool))
        for tag, jr, orr in (("", np.arange(nJ), np.arange(nJ, len(pool))), ("shuffled", np.sort(perm[:nJ]), np.sort(perm[nJ:]))):
            tj, to = (torch.as_tensor(x, device=A.device) for x in (jr, orr))
            DJ = RA.top_m(A[tj].sum(0), M, self.W)
            DO = RA.top_m(A[to].sum(0), M, self.W)
            key = f"{p['unit']}__planted" + (f"_{tag}" if tag else "")
            res = self.evaluate(key, unit, pool[jr], UR.cap_rows(cap, jr), A[tj], None, self.ref_arms(DJ, DO, random=not tag), linear=False)
            I_m, s_m = self.ref_I(res, DJ, DO)
            out[tag or "labels"] = {"falls": res, "I": I_m[f"m{M}"], "s": s_m[f"m{M}"], "I_m": I_m, "s_m": s_m,
                                    "n_J": int(len(jr)), "n_O": int(len(orr)), "falls_path": f"falls/{key}.npz"}
        self.m["planted"][p["unit"]] = {"layer": p["layer"], "a": p["a"], "b": p["b"], "lam": p["lam"], **out, "seconds": round(time.time() - t0, 1)}

    def planted_draws(self, p: dict):
        """Count-matched planted I (SPEC §3): per draw, a primary job's (|J|, |O*|) from the items' job_counts, the sides subsampled to it without
        replacement, D from the subsampled sides at each m, own and cross read on all J contexts (the before of the planted unit), the fall of a draw
        the mean over its subsampled J contexts. One rng per pair, seed 0, as unite_stats.planted_count_matched. Readouts of a set already read are reused."""
        t0 = time.time()
        unit, Jt, Ot, pool, cap, A = self.planted_sides(p)
        nJ, nO, K, ms = len(Jt), len(Ot), self.spec["draws"], self.spec["ref_ms"]
        kw = self.kw(RA.LENGTH)
        before = RA.readout(self.model, Jt, unit, None, **kw)["a"]
        banked = np.load(self.out / "falls" / f"{p['unit']}__planted.npz")["a_before"]
        assert np.array_equal(before, banked), "the draws' a_P before differs from the planted unit's"
        self.m["forwards"] += nJ
        self.m["tokens_forwarded"] += RA.n_forwarded(Jt, RA.LENGTH, self.bos, self.hw)
        cache = {}

        def after(S):
            k = frozenset(S)
            if k not in cache:
                cache[k] = RA.readout(self.model, Jt, unit, RA.ablation_dict(S, "mean", self.abar, unit["layer"], self.device), **kw)["a"]
                self.m["forwards"] += nJ
                self.m["tokens_forwarded"] += RA.n_forwarded(Jt, RA.LENGTH, self.bos, self.hw)
            return cache[k]
        counts = self.items["job_counts"]
        rng = np.random.default_rng(DRAW_SEED)
        bank = {"tok": Jt, "a_before": before, "j": np.zeros((K, nJ), bool), "o": np.zeros((K, nO), bool), "count": np.zeros((K, 2), np.int16)}
        for m in ms:
            for x in ("own", "cross"):
                bank[f"{x}_after_m{m}"] = np.full((K, nJ), np.nan, np.float32)
                bank[f"{x}_set_m{m}"] = np.full((K, m), -1, np.int64)
        I = {f"m{m}": [] for m in ms}
        for d in range(K):
            nj, no = counts[rng.integers(len(counts))]
            j = np.sort(rng.choice(nJ, min(nj, nJ), replace=False))
            o = np.sort(rng.choice(nO, min(no, nO), replace=False))
            bank["j"][d, j], bank["o"][d, o], bank["count"][d] = True, True, (nj, no)
            for m in ms:
                DJ = RA.top_m(A[torch.as_tensor(j, device=A.device)].sum(0), m, self.W)
                DO = RA.top_m(A[torch.as_tensor(nJ + o, device=A.device)].sum(0), m, self.W)
                own, cross = after(DJ), after(DO)
                bank[f"own_after_m{m}"][d], bank[f"cross_after_m{m}"][d] = own, cross
                bank[f"own_set_m{m}"][d], bank[f"cross_set_m{m}"][d] = [l * self.W + k for l, k in DJ], [l * self.W + k for l, k in DO]
                I[f"m{m}"].append(ratio(RA.job_fall(before[j], cross[j])["mean"], RA.job_fall(before[j], own[j])["mean"]))
        key = f"{p['unit']}__planted_draws"
        np.savez_compressed(self.out / "falls" / f"{key}.npz", **bank)
        med = {k: float(np.median([x for x in v if np.isfinite(x)])) if np.isfinite(v).any() else float("nan") for k, v in I.items()}
        self.m["planted"][p["unit"]]["count_matched"] = {"n_draws": K, "seed": DRAW_SEED, "I_median": med, "I_draws": I, "n_nonfinite": {k: int((~np.isfinite(v)).sum()) for k, v in I.items()},
                                                         "n_readouts": len(cache), "falls_path": f"falls/{key}.npz", "seconds": round(time.time() - t0, 1)}

    def ceiling(self, name: str, pole: str, c: dict):
        """A ceiling split read on side 2 (own D(side 2), cross D(side 1)), and on side 1 the other way when the arm set asks for it."""
        t0 = time.time()
        unit = UR.pole_unit(pole)
        s1, s2 = (np.asarray(x)[:self.spec["cap"]] for x in c["sides_tok"])
        cap1, cap2 = self.capture(s1, unit), self.capture(s2, unit)
        A1, A2 = RA.direct_A(self.rows, cap1, unit, self.W), RA.direct_A(self.rows, cap2, unit, self.W)
        M = max(self.spec["ref_ms"])
        D1, D2 = RA.top_m(A1.sum(0), M, self.W), RA.top_m(A2.sum(0), M, self.W)
        out = {}
        for side, toks, cap, A, own, cross in (("side2", s2, cap2, A2, D2, D1), ("side1", s1, cap1, A1, D1, D2)):
            if side not in self.spec["ceiling_sides"]:
                continue
            key = f"{pole}__c{c['c']}_ceiling_{side}"
            res = self.evaluate(key, unit, toks, cap, A, None, self.ref_arms(own, cross, random=self.spec["ceiling_random"]), linear=True)
            I_m, s_m = self.ref_I(res, own, cross)
            out[side] = {"falls": res, "I": I_m[f"m{M}"], "I_m": I_m, "n_read": int(len(toks)), "falls_path": f"falls/{key}.npz"}
        rev = {"I_ceil_reverse": out["side1"]["I"]} if "side1" in out else {}
        self.m["ceiling"][name] = {"pole": pole, "c": c["c"], "n_sides": [int(len(s1)), int(len(s2))], "s": s_m[f"m{M}"], "s_m": s_m, "I_ceil": out["side2"]["I"],
                                   "I_ceil_m": out["side2"]["I_m"], **rev, **out, "seconds": round(time.time() - t0, 1)}

    # -------------------------------------------------------------------------------------- the primary stage: which jobs a load reads
    def jobs_here(self) -> list:
        """Every item job, except at a primary or amend2 birth checkpoint: there only the jobs born at it."""
        jobs = self.items["jobs"] if self.spec["jobs"] else []
        if self.a.stage in ("primary", "amend2") and self.a.t != "T_15":
            jobs = [j for j in jobs if j["birth"] == self.t_index]
            assert jobs, f"no primary job is born at {self.a.t}"
        return jobs

    def primary_record(self, j: dict):
        """The item's label into the job record; at a birth checkpoint the job's ceiling split, read once per load, and its key."""
        rec = self.m["jobs"][j["job"]]
        rec["label"] = j["label"]
        if self.a.t == "T_15":
            return
        key = f"{j['pole']}__c{j['ceiling']['c']}"
        if key in self.split_sides:
            assert self.split_sides[key] == j["ceiling"]["sides_tok"], f"{key}: two jobs name this split with different sides"
        else:
            self.split_sides[key] = j["ceiling"]["sides_tok"]
            self.ceiling(key, j["pole"], j["ceiling"])
        rec["ceiling_split"] = key

    def primary_splits(self) -> dict:
        """Every distinct own ceiling split of the item jobs, "<pole>__c<ceiling c>" -> (pole, ceiling); jobs naming one split carry the same sides."""
        out = {}
        for j in self.items["jobs"]:
            key = f"{j['pole']}__c{j['ceiling']['c']}"
            if key in out:
                assert out[key][0] == j["pole"] and out[key][1]["sides_tok"] == j["ceiling"]["sides_tok"], f"{key}: two jobs name this split with different sides"
            else:
                out[key] = (j["pole"], j["ceiling"])
        return out

    # -------------------------------------------------------------------------------------- the summary
    def summarize(self):
        ms, rks = self.spec["ms"], self.spec["rankings"]
        J = self.m["jobs"]
        med = lambda xs: float(np.nanmedian(xs)) if len(xs) else float("nan")
        tab = {}
        for rk in rks:
            for m in ms:
                k = f"{rk}_m{m}"
                tab[k] = {x: med([v["falls"][f"{x}_{k}"]["mean"] for v in J.values()]) for x in ("own", "cross", "random")}
                tab[k]["I_median"] = med([v["I"][k] for v in J.values()])
                tab[k]["s_median"] = med([v["s"][k] for v in J.values()])
                tab[k]["n_cancelled"] = sum(v["cancel"][k]["cancelled"] for v in J.values())
            fm = self.spec["m_foreign"]
            if fm in ms:
                tab[f"{rk}_m{fm}"]["foreign"] = med([v["falls"][f"foreign_{rk}_m{fm}"]["mean"] for v in J.values()])
                tab[f"{rk}_m{fm}"]["cross_minus_foreign"] = med([v["foreign_contrast"][rk][f"cross_minus_foreign_m{fm}"] for v in J.values()])
        idf = np.array([r["fall"] for r in self.id_rows])
        idc = np.array([r["change"] for r in self.id_rows])
        fl = np.array([r["fall"] for r in self.id_fail_rows])
        self.m["identity"] = {"max_abs_fall": float(np.nanmax(np.abs(idf))), "max_abs_change": float(idc.max()), "n_contexts": int(idf.size),
                              "n_over_tol": int((np.abs(idf) > RA.ID_TOL).sum()), "tol": RA.ID_TOL, "pass": bool((np.abs(idf) <= RA.ID_TOL).all()),
                              "failing_input": self.spec["id_fail_arm"], "failing_input_n_over_tol": int((np.abs(fl) > RA.ID_TOL).sum()),
                              "failing_input_n": int(fl.size), "failing_input_median_abs_fall": float(np.median(np.abs(fl))) if fl.size else float("nan"),
                              "failing_input_failed": bool((np.abs(fl) > RA.ID_TOL).any())}
        lin = self.lin
        fit = lambda x, y: RA.linfit([r[x] for r in lin], [r[y] for r in lin])
        self.m["linearity"] = {"n_rows": len(lin), "d_a_vs_sum_A": fit("sum_A", "d_a"), "d_a_vs_sum_A_c": fit("sum_A_c", "d_a"),
                               "d_g_vs_pred_g": fit("pred_g", "d_g"), "d_u_vs_pred_u": fit("pred_u", "d_u"),
                               "by_arm_kind": {kind: RA.linfit([r["sum_A_c"] for r in lin if r["arm"].startswith(kind)], [r["d_a"] for r in lin if r["arm"].startswith(kind)])
                                               for kind in ("own", "cross", "random", "foreign")}}
        with open(self.out / "linearity.jsonl", "w") as fh:
            for r in lin:
                fh.write(json.dumps(r) + "\n")
        summ = {"by_ranking_m": tab}
        if J and self.a.stage == "sniper":
            summ["freeze_rule"] = RA.freeze_rule({rk: {m: [v["falls"][f"own_{rk}_m{m}"]["mean"] for v in J.values()] for m in ms} for rk in rks}, ms)
        if self.spec["zero"]:
            summ["zero_vs_mean_own_direct_m50"] = {"mean": med([v["falls"]["own_direct_m50"]["mean"] for v in J.values()]),
                                                   "zero": med([v["falls"]["own_direct_m50_zero"]["mean"] for v in J.values()])}
        if self.m["planted"]:
            summ["planted_I"] = {u: v["labels"]["I"] for u, v in self.m["planted"].items()}
            summ["planted_I_shuffled"] = {u: v["shuffled"]["I"] for u, v in self.m["planted"].items()}
            summ["planted_I_m"] = {u: v["labels"]["I_m"] for u, v in self.m["planted"].items()}
            summ["planted_I_shuffled_m"] = {u: v["shuffled"]["I_m"] for u, v in self.m["planted"].items()}
            if self.spec["draws"]:
                summ["planted_I_count_matched"] = {u: v["count_matched"]["I_median"] for u, v in self.m["planted"].items()}
        if self.m["ceiling"]:
            summ["ceiling_I"] = {u: v["I_ceil"] for u, v in self.m["ceiling"].items()}
            summ["ceiling_I_m"] = {u: v["I_ceil_m"] for u, v in self.m["ceiling"].items()}
        if self.a.stage == "amend2":
            summ["rj_direct_m50"] = {"median_fall": med([v["falls"]["rj_direct_m50"]["mean"] for v in J.values()]), "n_jobs": len(J)}
        self.m["summary"] = summ
        self.write()
        assert self.m["identity"]["pass"], f"identity failed: {self.m['identity']}"
        assert self.m["identity"]["failing_input_failed"], f"identity's failing input passed: {self.m['identity']}"

    def run(self):
        self.known()
        self.background()
        q = self.foreign() if self.spec["jobs"] and self.a.stage != "amend2" else {}
        t0 = time.time()
        tok0 = self.m["tokens_forwarded"]
        jobs = self.jobs_here()
        self.m["n_jobs_to_read"] = len(jobs)
        for i, j in enumerate(jobs):
            if self.a.stage == "amend2":
                self.active_job(j)
            else:
                self.pole_job(j, q[j["job"]])
            if self.spec["ceiling"] == "primary":
                self.primary_record(j)
            self.m["n_jobs_done"] = i + 1
            self.write()
        if self.spec["planted"]:
            for i, p in enumerate(self.items["planted"]):
                self.planted(p)
                self.m["n_planted_done"] = i + 1
                self.write()
        if self.spec["ceiling"] == "jobs":
            for j in self.items["jobs"]:
                if j["ceiling"]["in_s1"]:
                    self.ceiling(j["job"], j["pole"], j["ceiling"])
                    self.write()
        if self.spec["ceiling"] == "setting":
            for i, c in enumerate(self.items["ceilings"]):
                self.ceiling(c["split"], c["pole"], c)
                self.m["n_ceilings_done"] = i + 1
                self.write()
        if self.spec["ceiling"] == "all_primary":
            splits = self.primary_splits()
            self.m["n_ceilings_to_read"] = len(splits)
            for i, (key, (pole, c)) in enumerate(splits.items()):
                self.ceiling(key, pole, c)
                self.m["n_ceilings_done"] = i + 1
                self.write()
        if self.spec["draws"]:
            for i, p in enumerate(self.items["planted"]):
                self.planted_draws(p)
                self.m["n_planted_draws_done"] = i + 1
                self.write()
        dt = time.time() - t0
        self.m["timing_s"]["units"] = round(dt, 1)
        self.m["units_tokens_per_second"] = (self.m["tokens_forwarded"] - tok0) / dt
        self.summarize()
        self.m["status"] = "done"
        self.write()


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["sniper", "primary", "amend2"])
    ap.add_argument("--arms", required=True, choices=sorted(ARMSETS))
    ap.add_argument("--t", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--setting", default=None, help="amend2: setting_items.json, whose planted pairs B1 reads")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--from-step", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--full-batch", type=int, default=8)
    ap.add_argument("--grad-batch", type=int, default=8)
    ap.add_argument("--scan-tokens", type=int, default=20_570_743)
    ap.add_argument("--harvest-window", type=int, default=UR.HARVEST_WINDOW)
    return ap


def main():
    a = parser().parse_args()
    assert REVISION[a.t] == a.revision, f"{a.t} is {REVISION[a.t]}, not {a.revision}"
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    if a.from_step and a.from_step.startswith("s3://"):
        a.from_step = str(fetch(a.from_step.rstrip("/") + "/foreign.json", f"reuse_{a.arms}_{a.t}_from_step/foreign.json").parent)
    items_path = fetch(a.items, f"reuse_{a.arms}_{a.t}_items.json")
    a.items_sha256 = hashlib.sha256(pathlib.Path(items_path).read_bytes()).hexdigest()
    items = json.loads(pathlib.Path(items_path).read_text())
    assert (a.stage == "amend2") == (a.setting is not None), "--setting is the amend2 stage's planted pairs, and only its"
    if a.stage == "amend2":
        setting_path = fetch(a.setting, f"reuse_{a.arms}_{a.t}_setting.json")
        a.setting_sha256 = hashlib.sha256(pathlib.Path(setting_path).read_bytes()).hexdigest()
        assert "planted" not in items, "the primary items carry planted pairs; the amend2 stage reads them from --setting only"
        items["planted"] = json.loads(pathlib.Path(setting_path).read_text())["planted"]
    if a.stage in ("primary", "amend2"):
        grid = {f"T_{int(g[1:3])}": "stage1-" + g.split("_")[1] for g in items["sources"]["grid"]}
        assert grid[a.t] == a.revision, f"the items' grid has {grid[a.t]} at {a.t}, not {a.revision}"
    t0 = time.time()
    local, sha, rec = download("allenai/Olmo-3-1025-7B", a.revision)
    corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
    t1 = time.time()
    model, tok = NH.load_model(local, "O7")
    model.requires_grad_(False)
    job = Job(a.out, a, items, model, tok, corpus, "cuda")
    job.m.update({"download": rec, "download_and_corpus_s": round(t1 - t0, 1), "load_s": round(time.time() - t1, 1), "sha": sha,
                  "device_name": torch.cuda.get_device_name(0), "transformers": __import__("transformers").__version__, "torch": torch.__version__})
    job.write()
    job.run()
    job.m["timing_s"]["wall_after_load"] = round(time.time() - t1, 1)
    job.write()
    del model
    torch.cuda.empty_cache()
    repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / "models--allenai--Olmo-3-1025-7B"
    shutil.rmtree(repo_dir)
    assert not repo_dir.exists()
    print("REUSE_ABLATE_OK", json.dumps({"t": a.t, "arms": a.arms}), flush=True)


if __name__ == "__main__":
    main()
