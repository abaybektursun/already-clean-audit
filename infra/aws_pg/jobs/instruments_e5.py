"""INSTRUMENTS E5 job (studies/instruments/SPEC_instruments.md, section "E5: the cause of Olmo-3's generic term"): tests T1, T2, T4 and T5 in three
pipeline steps on one instance. jobs.instruments_e1 (the model load, the known activation number and completeness, ā, the capture X, the values, B_t,
the planted sides and the sets), jobs.instruments_e1b (T4, unchanged) and reuse_ablate, reuse_birth, reuse_generic, unite_route, instruments_values,
instruments_adapter are imported unchanged; infra/pg/instruments_e5.py adds the input hold, the marked-token write and the member write.

    python -m infra.aws_pg.jobs.instruments_e5 --stage t12 --t T_15 --revision stage1-step1413814 --items <s3 e1_items.json> --corpus <s3> --out <dir>
    python -m infra.aws_pg.jobs.instruments_e5 --stage t4 --items <s3 e5_items.json> --corpus <s3> --out <dir>
    python -m infra.aws_pg.jobs.instruments_e5 --stage t5 --items <s3 e5_items.json> --corpus <s3> --olmo-step t12 --out <dir>

t12 (one Olmo-3 load at T_15, fetched from the hub as E1 fetches it, kept in the cache for t4): E1's known activation number and completeness
  (asserted), E1's ā on its 200 e1 positions, the capture X and B_t (E1's background, unchanged); T5's Olmo capture: the same 200 positions through
  instruments_adapter.capture, s_l and the residual rms per layer and position, its mean asserted equal to ā (failing input: the first half's mean).
  Then E1's 56 planted pairs on side J (12 contexts each; sets as E1's pair_pass1 builds them: own D(J), cross D(O) under u = unexcluded and e = B_t
  excluded, members = the B_t members of the unexcluded cross set), the mean value written:
  pass 1, every pair: a_P before, E1's identity arm, and the free knock-outs own_u, cross_u, own_e, cross_e (the known numbers: E1's side-J medians
    of the cross_u and own_u falls, 0.0741 and 0.4737); metrics.json then carries the known-number gate before any new number.
  pass 2, every pair, with the input hold (instruments_e5.held_input on q/k/v/gate/up of layers 0..L): a_P before again with the clean rms recorded
    (its equality with pass 1 is reported as determinism); heldclean (the identity ablation under the hold of layers 0..L; asserted equal to the
    recorded clean a_P to 1e-4), heldclean1 (the hold of layer L alone), heldclean_shift (the failing input: the rms of position t - 1 at t),
    heldclean_inf (the first pair only: the clean rms set to inf, the non-finite counter's failing input); held__<set> (the hold of the lowest
    ablated layer .. L) and held1__<set> (layer L alone) for the four sets; marked__<set> (instruments_e5.readout_marked: the value written at the
    marked token only) for the four sets.
  After pass 1 the T5 Olmo bank: the distinct members of the side-J unexcluded cross sets, ||d_k|| from the weights, the write per position.
t4 (T4): jobs.instruments_e1b.Job, unchanged, on Olmo-3 at T_15 with model key o7 and the items e5_items.json["models"]["o7"] (build_items_e5.py:
  build_items_e1b's rule on the o7_t15 harvest bank); the weights from the hub cache t12 filled (downloaded again if absent), deleted at the end.
t5 (T5): the Olmo bank of the sibling t12 step, then gemma-3-4b-pt and Qwen3-4B-Base from the S3 weight cache (pg run --weights), each loaded once:
  one capture of E1b's 200 background positions (s_l, the residual rms, ā asserted equal to E1b's banked ā at every member to 1e-5 (1 + |ā|);
  failing input the first half's mean), ||d_k|| from the weights, the write per member and position. No other forward.
Bank: t12/e5/pairs/<unit>.npz (J_tok, J_a_before, J_a_before2, J_arms, J_a_after [arms, 12], set_<x>_layer/_neuron/_abar for own_u, cross_u,
own_e, cross_e, members; held_lo per set), t12/e5/t5_o7.npz, t5/t5/<mk>.npz (members_layer, members_neuron, abar, abar_capture, dnorm, s, rms,
write, positions), values.npz and the rest of E1's background files in t12; t4 is E1b's bank (e1b/o7/...). samples/e5.jsonl: t12 the first 3 pairs
x (T1 held, T1 held1, T2 marked) x (own_u, cross_u) = 18 lines with the contexts, the set, the written positions, a_P before and after (free and
under the test), the held clean and ablated rms at the marked token per held layer for v and g (T1), and 3 T5 lines for Olmo; t5 3 lines per model.
metrics.json after every pair (t12), every item (t4, E1b's) and every model (t5). The job asserts instrument checks only; the verdict decides.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time

import numpy as np
import torch

from infra.aws_pg.jobs import instruments_e1 as JE
from infra.aws_pg.jobs import instruments_e1b as JE1B
from infra.pg import instruments_adapter as AD
from infra.pg import instruments_e5 as E5
from infra.pg import instruments_values as IV
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import reuse_generic as RG
from infra.pg import unite_route as UR

M = RB.M
CAP = 12
SETS = ("own_u", "cross_u", "own_e", "cross_e")
KNOWN = {"cross_u": 0.0741, "own_u": 0.4737}          # E1's side-J medians over 56 pairs of the mean-value job falls (results/e1/1/t15/metrics.json)
KNOWN_TOL = {"cross_u": 0.005, "own_u": 0.02}
N_SAMPLE_PAIRS, N_T5_SAMPLES = 3, 3
SAMPLE_ARMS = (("T1", "held"), ("T1", "held1"), ("T2", "marked"))
N_T12_LINES = N_SAMPLE_PAIRS * len(SAMPLE_ARMS) * 2                  # plus min(3, members) T5 lines for Olmo
OLMO = "allenai/Olmo-3-1025-7B"
T5_MODELS = ("g4", "q4b")


def rel(a, ref) -> float:
    """max |a - ref| / max |ref| (the identity gate's units)."""
    a, ref = np.asarray(a, np.float64), np.asarray(ref, np.float64)
    return float(np.abs(a - ref).max() / np.abs(ref).max())


def known_gate(cross: dict, own: dict) -> dict:
    """E1's known numbers from per-pair side-J job falls {unit: fall}: the medians of cross_u and own_u against 0.0741 and 0.4737."""
    v = {"cross_u": float(np.median(list(cross.values()))), "own_u": float(np.median(list(own.values())))}
    return {**v, "pass": all(abs(v[k] - KNOWN[k]) <= KNOWN_TOL[k] for k in KNOWN), "targets": KNOWN, "tol": KNOWN_TOL, "n": len(cross)}


def t5_bank(path, ad, members: list, abar_m, abar_cap_m, abar_half_m, sc: dict, positions) -> dict:
    """The member write per position for `members` [(l, k)] and the ā check at those members; banks the npz, returns the record."""
    L = np.array([l for l, _ in members], np.int64)
    K = np.array([k for _, k in members], np.int64)
    dn = np.zeros(len(members))
    for l in sorted(set(L.tolist())):
        j = np.nonzero(L == l)[0]
        dn[j] = E5.column_norms(ad, l, K[j].tolist()).cpu().numpy()
    w = E5.member_writes(dn, abar_m, L, sc["s"], sc["rms"])
    chk = E5.abar_check(abar_cap_m, abar_half_m, abar_m)
    np.savez_compressed(path, members_layer=L.astype(np.int16), members_neuron=K.astype(np.int32), abar=np.asarray(abar_m, np.float32),
                        abar_capture=np.asarray(abar_cap_m, np.float32), dnorm=dn.astype(np.float32), s=sc["s"].cpu().numpy(), rms=sc["rms"].cpu().numpy(),
                        write=w.astype(np.float32), positions=np.asarray(positions, np.int64))
    return {"check": chk, "summary": {k: v for k, v in E5.summary(w).items() if k != "per_member_median"}, "n_positions": int(len(positions)),
            "dnorm_median": float(np.median(dn)) if dn.size else float("nan"), "s_median_by_layer": sc["s"].median(0).values.cpu().numpy().tolist(),
            "rms_median_by_layer": sc["rms"].median(0).values.cpu().numpy().tolist(), "write": w}


def t5_samples(fh, mk: str, members: list, rec: dict, positions, tokens, bos, hw, dec) -> int:
    """N_T5_SAMPLES lines: the first members with their numbers and three positions with the context text."""
    x, pos = UR.pack(tokens, np.asarray(positions[:3]), RA.LENGTH, bos, hw, "cpu")
    ctx = [{"position": int(positions[i]), "prompt": dec(x[i, :int(pos[i]) + 1].tolist()), "marked_token": dec([int(x[i, int(pos[i])])])} for i in range(3)]
    n = 0
    for j, (l, k) in enumerate(members[:N_T5_SAMPLES]):
        w = rec["write"][:, j]
        fh.write(json.dumps({"test": "T5", "model": mk, "member": [int(l), int(k)], "abar": float(rec["abar_m"][j]), "dnorm": float(rec["dnorm"][j]),
                             "write_median": float(np.median(w)), "write_at_contexts": [float(w[i]) for i in range(3)], "contexts": ctx}, ensure_ascii=False) + "\n")
        n += 1
    return n


# ------------------------------------------------------------------------------------------ t12: E1's pairs under the hold and the marked write
class T12(JE.Job):
    def __init__(self, out, a, items: dict, model, tok, corpus: dict, device: str):
        self.out, self.a, self.items, self.model, self.tok, self.device = pathlib.Path(out), a, items, model, tok, device
        for d in ("", "e5/pairs", "samples"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.tokens, self.offsets = corpus["tokens"], corpus["doc_offsets"]
        self.bos = tok.bos_token_id
        self.tower = NH.text_tower(model)
        self.layers = list(self.tower.layers)
        self.W = int(self.layers[0].mlp.gate_proj.weight.shape[0])
        self.rows = UR.Rows(model)
        self.ad = AD.Adapter(model)
        assert self.ad.arch == "olmo3" and a.t == "T_15", (self.ad.arch, a.t)
        self.hw = a.harvest_window
        self.spec = {"cap": CAP}
        self.dec = NH.decoder(tok)
        assert NH.act_name(self.layers[0].mlp.act_fn) == "silu", "the coefficients assume silu"
        self.start = RA.window_start(self.bos)
        self.m = {"stage": "e5_t12", "t": a.t, "revision": a.revision, "items_sha256": a.items_sha256, "harvest_window": self.hw, "window": RA.LENGTH,
                  "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"), "n_layers": len(self.layers), "width": self.W,
                  "dtype": str(next(model.parameters()).dtype), "status": "running", "m": M, "cap": CAP, "value": "mean", "side": "J",
                  "job_fall_rule": "mean over contexts with a_P > 0", "hold": "q/k/v/gate/up inputs at window positions rescaled to the clean rms per position",
                  "marked": "the value written into down_proj's input at the marked token only", "timing_s": {}, "tokens_forwarded": 0, "forwards": 0,
                  "captures": 0, "readouts": 0, "n_to_read": {"pass1": len(items["planted"]), "pass2": len(items["planted"])}, "n_done": {"pass1": 0, "pass2": 0},
                  "gates": {}, "pairs": {}}
        self.state = {}
        self.excl = {"n_sets": 0, "n_members": 0, "n_in_B": 0}
        self.n_samples = 0
        self.write()

    # -------------------------------------------------------------------------------------- T5's Olmo capture
    def t5_capture(self):
        t0 = time.time()
        pos = UR.background_positions(self.t_index, self.a.scan_tokens // self.hw * self.hw, self.hw)["e1"]
        n_up = len(self.layers) - 1
        self.t5 = {"pos": pos, "sc": E5.layer_scales(self.ad, self.model, pos, n_up, **self.kw(RA.LENGTH))}
        self.m["captures"] += len(pos)
        self.m["tokens_forwarded"] += RA.n_forwarded(pos, RA.LENGTH, self.bos, self.hw)
        sc = self.t5["sc"]
        chk = E5.abar_check(sc["abar"].cpu().numpy(), sc["abar_half"].cpu().numpy(), self.abar.cpu().numpy())
        self.m["gates"]["t5_capture"] = {**chk, "rule": "the T5 capture's mean over E1's 200 e1 positions equals E1's ā at every neuron"}
        self.m["timing_s"]["t5_capture"] = round(time.time() - t0, 1)
        self.write()
        assert chk["pass"] and chk["failing_input_failed"], f"T5's Olmo capture: {chk}"

    def t5_members(self):
        mem = sorted({x for st in self.state.values() for x in st["sets"]["members"]})
        sc = self.t5["sc"]
        L = [l for l, _ in mem]
        K = [k for _, k in mem]
        ab = self.abar.cpu().numpy()
        r = t5_bank(self.out / "e5" / "t5_o7.npz", self.ad, mem, ab[L, K], sc["abar"].cpu().numpy()[L, K], sc["abar_half"].cpu().numpy()[L, K], sc, self.t5["pos"])
        z = np.load(self.out / "e5" / "t5_o7.npz")
        self.m["t5_o7"] = {k: v for k, v in r.items() if k != "write"}
        self.m["t5_o7"]["members"] = [[int(l), int(k), float(a)] for l, k, a in zip(L, K, ab[L, K])]
        with open(self.out / "samples" / "e5.jsonl", "a") as fh:
            self.n_samples += t5_samples(fh, "o7", mem, {"write": r["write"], "abar_m": z["abar"], "dnorm": z["dnorm"]}, self.t5["pos"], self.tokens, self.bos,
                                         self.hw, self.dec)
        self.write()

    # -------------------------------------------------------------------------------------- readouts
    def readout(self, st: dict, ablation, marked: bool = False) -> np.ndarray:
        f = E5.readout_marked if marked else RA.readout
        a = f(self.model, st["tok"], st["unit"], ablation, **self.kw(RA.LENGTH))["a"]
        self.m["readouts"] += 1
        self.m["forwards"] += len(st["tok"])
        self.m["tokens_forwarded"] += RA.n_forwarded(st["tok"], RA.LENGTH, self.bos, self.hw)
        return a

    def ablation(self, st: dict, S: list) -> dict:
        L = st["unit"]["layer"]
        return IV.written(S, "mean", self.tab, L, self.device) if S else IV.written([], "mean", self.tab, L, self.device, identity=True)

    def held(self, st: dict, H, name: str, S: list, layers, **hk):
        H.hold(layers, **hk)
        a = self.readout(st, self.ablation(st, S))
        nf = H.done()
        st["after"][name] = a
        st["nonfinite"][name] = {"a_P": int((~np.isfinite(a)).sum()), "by_layer": {str(l): n for l, n in nf.items()}}

    def marked_scales(self, H, st: dict, layers) -> list:
        """Per context: {layer: {v: [clean rms, ablated rms], g: [...]}} at the marked token, from the last held forward (its batches in readout order)."""
        toks = np.asarray(st["tok"])
        order = np.argsort(toks % self.hw, kind="stable")
        _, pos = UR.pack(self.tokens, toks[order], RA.LENGTH, self.bos, self.hw, "cpu")
        B = self.a.batch
        got = {key: [(c.cpu().numpy(), h.cpu().numpy()) for c, h in H.last[key]] for key in ((l, r) for l in layers for r in ("v", "g"))}
        out = [None] * len(toks)
        for j, i in enumerate(order):
            b, row = divmod(j, B)
            t = int(pos[j])
            out[i] = {str(l): {r: [float(got[(l, r)][b][0][row, t]), float(got[(l, r)][b][1][row, t])] for r in ("v", "g")} for l in layers}
        return out

    # -------------------------------------------------------------------------------------- pairs
    def pass1(self, p: dict):
        unit, Jt, Ot, pool, cap, A = self.planted_sides(p)
        nJ, W = len(Jt), self.W
        sJ, sO = A[:nJ].sum(0), A[nJ:].sum(0)
        Ju, Ou = RA.top_m(sJ, M, W), RA.top_m(sO, M, W)
        sets = {"own_u": Ju, "cross_u": Ou, "own_e": self.rank(sJ), "cross_e": self.rank(sO), "members": RG.members_alone(Ou, self.mask, self.abar)}
        self.excluded([sets["own_e"], sets["cross_e"]])
        del cap, A
        st = {"unit": unit, "tok": Jt, "sets": sets, "after": {}, "nonfinite": {}, "samples": {}, "t0": time.time()}
        st["before"] = self.readout(st, None)
        st["after"]["identity"] = self.readout(st, self.ablation(st, []))
        for x in SETS:
            st["after"][f"free__{x}"] = self.readout(st, self.ablation(st, sets[x]))
        self.state[p["unit"]] = st
        self.finish(p, st, 1)

    def pass2(self, p: dict, first: bool, sample: bool):
        st = self.state[p["unit"]]
        L = st["unit"]["layer"]
        S = st["sets"]
        everything = list(range(L + 1))
        with E5.held_input(self.layers, everything, self.start) as H:
            H.record()
            st["before2"] = self.readout(st, None)
            H.free()
            self.held(st, H, "heldclean", [], everything)
            self.held(st, H, "heldclean1", [], [L])
            for x in ("heldclean", "heldclean1"):
                d = rel(st["after"][x], st["before2"])
                assert d <= E5.ID_TOL, (p["unit"], x, d, "the held clean forward does not reproduce the clean a_P")
            self.held(st, H, "heldclean_shift", [], everything, shift=1)
            if first:
                self.held(st, H, "heldclean_inf", [], everything, fill=float("inf"))
            for x in SETS:
                self.held(st, H, f"held__{x}", S[x], E5.held_layers(S[x], L))
                if sample and x in ("own_u", "cross_u"):
                    st["samples"][f"held__{x}"] = self.marked_scales(H, st, E5.held_layers(S[x], L))
                self.held(st, H, f"held1__{x}", S[x], [L])
                if sample and x in ("own_u", "cross_u"):
                    st["samples"][f"held1__{x}"] = self.marked_scales(H, st, [L])
        for x in SETS:
            st["after"][f"marked__{x}"] = self.readout(st, self.ablation(st, S[x]), marked=True)
        self.finish(p, st, 2)
        if sample:
            self.samples(p, st)
        del self.state[p["unit"]]

    def finish(self, p: dict, st: dict, npass: int):
        S = st["sets"]
        names = list(st["after"])
        L = st["unit"]["layer"]
        z = {"unit": p["unit"], "layer": p["layer"], "tercile": p["tercile"], "npass": npass, "J_tok": np.asarray(st["tok"], np.int64), "J_a_before": st["before"],
             "J_arms": np.array(names), "J_a_after": np.stack([st["after"][n] for n in names]).astype(np.float32)}
        if "before2" in st:
            z["J_a_before2"] = st["before2"]
        ab = self.abar.cpu().numpy()
        for x, Sx in S.items():
            li, ki = np.array([l for l, _ in Sx], np.int64), np.array([k for _, k in Sx], np.int64)
            z.update({f"set_{x}_layer": li.astype(np.int16), f"set_{x}_neuron": ki.astype(np.int32), f"set_{x}_abar": ab[li, ki],
                      f"held_lo_{x}": np.int16(E5.held_layers(Sx, L)[0])})
        np.savez_compressed(self.out / "e5" / "pairs" / f"{p['unit']}.npz", **z)
        b = st["before"]
        rec = {"layer": p["layer"], "tercile": p["tercile"], "a": p["a"], "b": p["b"], "lam": p["lam"], "pass": npass, "npz": f"e5/pairs/{p['unit']}.npz",
               "falls": {n: {k: f[k] for k in ("mean", "median", "n_pos")} for n, f in ((n, RA.job_fall(b, st["after"][n])) for n in names)},
               "nonfinite": st["nonfinite"], "identity": {"free_identity": rel(st["after"]["identity"], b)},
               "set_sizes": {x: len(v) for x, v in S.items()}, "held_layers": {x: [E5.held_layers(S[x], L)[0], L] for x in SETS},
               "seconds": round(time.time() - st["t0"], 1)}
        if npass == 2:
            b2 = st["before2"]
            rec["identity"].update({x: rel(st["after"][x], b2) for x in ("heldclean", "heldclean1", "heldclean_shift")})
            rec["determinism"] = rel(b2, b)
        self.m["pairs"][p["unit"]] = rec

    def samples(self, p: dict, st: dict):
        pr = self.prompts(st["tok"])
        b = st["before"]
        ab = self.abar.cpu().numpy()
        L = st["unit"]["layer"]
        with open(self.out / "samples" / "e5.jsonl", "a") as fh:
            for test, cond in SAMPLE_ARMS:
                for x in ("own_u", "cross_u"):
                    S = st["sets"][x]
                    aft, free = st["after"][f"{cond}__{x}"], st["after"][f"free__{x}"]
                    sc = st["samples"].get(f"{cond}__{x}")
                    ctx = [{"tok": int(t), **pr[i], "a_before": float(b[i]), "a_after": float(aft[i]), "a_after_free": float(free[i]),
                            "fall": float((b[i] - aft[i]) / b[i]), "fall_free": float((b[i] - free[i]) / b[i]),
                            **({"held_rms_clean_ablated": sc[i]} if sc is not None else {})} for i, t in enumerate(st["tok"])]
                    where = ("every window position >= 1 of each row (BOS left), as E1" if test == "T1" else
                             "the marked token only (position n_tokens - 1 of each row); every other position untouched")
                    hl = E5.held_layers(S, L) if cond == "held" else [L] if cond == "held1" else []
                    fh.write(json.dumps({"test": test, "condition": cond, "pair": p["unit"], "layer": p["layer"], "side": "J", "value": "mean", "set": x,
                                         "written_positions": where, "held_layers": hl, "members": [[int(l), int(k), float(ab[l, k])] for l, k in S],
                                         "job_fall": RA.job_fall(b, aft)["mean"], "job_fall_free": RA.job_fall(b, free)["mean"], "contexts": ctx},
                                        ensure_ascii=False) + "\n")
                    self.n_samples += 1

    # -------------------------------------------------------------------------------------- gates as they become readable
    def gates(self):
        P, g = self.m["pairs"], self.m["gates"]
        fall = lambda arm, ps: {u: v["falls"][arm]["mean"] for u, v in ps.items()}
        kn = known_gate(fall("free__cross_u", P), fall("free__own_u", P))
        kf = known_gate(fall("free__cross_e", P), fall("free__own_e", P))
        g["known_number"] = {**kn, "failing_input": {"input": "the excluded construction's falls (cross_e, own_e) in place of cross_u, own_u", "cross_u": kf["cross_u"],
                                                     "own_u": kf["own_u"]}, "failing_input_failed": not kf["pass"], "identity": None,
                             "rule": "medians over pairs of the side-J free mean-value job falls within 0.005 of 0.0741 (cross_u) and 0.02 of 0.4737 (own_u)"}
        g["free_identity"] = {"value": max(v["identity"]["free_identity"] for v in P.values()), "tol": JE.ID_TOL,
                              "rule": "E1's identity arm (hooks below the pole, nothing written): reported"}
        P2 = {u: v for u, v in P.items() if v["pass"] == 2}
        if not P2:
            return
        idv = max(max(v["identity"]["heldclean"], v["identity"]["heldclean1"]) for v in P2.values())
        sh = [v["identity"]["heldclean_shift"] for v in P2.values()]
        g["held_identity"] = {"value": idv, "tol": E5.ID_TOL, "pass": idv <= E5.ID_TOL, "n_pairs": len(P2),
                              "failing_input": {"input": "the held scales from the wrong position (the clean rms of position t - 1 at t)", "value": max(sh), "min": min(sh)},
                              "failing_input_failed": max(sh) > E5.ID_TOL, "identity": {"value": idv, "reason": "the gate is the identity check of the hold"},
                              "rule": "max over pairs of |a_P(held clean) - a_P(clean, recorded)| / max |a_P| for the all-layer and the target-layer hold <= 1e-4"}
        held = [(u, n, nf) for u, v in P2.items() for n, nf in v["nonfinite"].items() if n.startswith(("held__", "held1__", "heldclean")) and n != "heldclean_inf"]
        share = float(np.mean([nf["a_P"] > 0 for _, _, nf in held]))
        inf = [nf for v in P2.values() for n, nf in v["nonfinite"].items() if n == "heldclean_inf"]
        by = {}
        for _, n, nf in held:
            c = n.split("__")[0]
            for l, k in nf["by_layer"].items():
                by.setdefault(c, {}).setdefault(l, 0)
                by[c][l] += k
        fi = float(np.mean([x["a_P"] > 0 for x in inf])) if inf else float("nan")
        g["nonfinite"] = {"value": share, "line": E5.NONFINITE_LINE, "pass": share <= E5.NONFINITE_LINE, "n_readouts": len(held), "by_layer": by,
                          "failing_input": {"input": "the clean rms set to inf (the first pair's held clean forward)", "value": fi},
                          "failing_input_failed": bool(fi > E5.NONFINITE_LINE), "identity": None,
                          "rule": "share of held readouts (held, held1, heldclean, heldclean1, heldclean_shift) with a non-finite a_P <= 0.05"}
        g["determinism"] = {"value": max(v["determinism"] for v in P2.values()), "rule": "pass 2's clean a_P against pass 1's: reported"}
        want = N_T12_LINES + min(N_T5_SAMPLES, len(self.m["t5_o7"]["members"]))
        g["samples"] = {"lines": self.n_samples, "want": want, "pass": self.n_samples == want, "path": "samples/e5.jsonl"}

    # -------------------------------------------------------------------------------------- the load
    def run(self):
        self.known()
        self.m["captures"] += sum(min(JE.JR.KNOWN_CTX, len(x)) for x in self.items["known"]["tok"])
        self.background()
        self.t5_capture()
        t0 = time.time()
        tok0 = self.m["tokens_forwarded"]
        P = self.items["planted"]
        for p in P:
            self.pass1(p)
            self.m["n_done"]["pass1"] += 1
            self.gates()
            self.write()
        self.m["timing_s"]["pass1"] = round(time.time() - t0, 1)
        self.t5_members()
        for i, p in enumerate(P):
            self.pass2(p, first=i == 0, sample=i < N_SAMPLE_PAIRS)
            self.m["n_done"]["pass2"] += 1
            self.gates()
            self.write()
        assert self.m["n_done"] == self.m["n_to_read"], (self.m["n_done"], self.m["n_to_read"])
        dt = time.time() - t0
        self.m["timing_s"]["units"] = round(dt, 1)
        self.m["units_tokens_per_second"] = (self.m["tokens_forwarded"] - tok0) / dt
        self.m["exclusion"] = {**self.excl, "pass": self.excl["n_in_B"] == 0}
        self.m["memory_max_allocated_gb"] = RB.memory_gb(self.device)
        self.m["status"] = "done"
        self.write()
        g = self.m["gates"]
        assert g["held_identity"]["pass"] and g["held_identity"]["failing_input_failed"], f"held identity: {g['held_identity']}"
        assert g["nonfinite"]["failing_input_failed"], f"the non-finite counter's failing input passed: {g['nonfinite']}"
        assert g["samples"]["pass"], g["samples"]


# ------------------------------------------------------------------------------------------ t4: E1b's job on Olmo-3
def t4_job(out, rec: dict, items_sha256: str, model, tok, corpus: dict, device: str, batch: int, full_batch: int):
    ns = argparse.Namespace(model_key="o7", items_sha256=items_sha256, harvest_window=rec["window"], batch=batch, full_batch=full_batch)
    return JE1B.Job(out, ns, rec, model, tok, corpus, device)


# ------------------------------------------------------------------------------------------ t5: the member write on gemma and Qwen
def t5_model(out: pathlib.Path, mk: str, rec: dict, model, tok, corpus: dict, device: str, batch: int, fh) -> dict:
    t0 = time.time()
    ad = AD.Adapter(model, n_layers=rec["n_layers"])
    assert (ad.arch, ad.width, ad.hidden) == (rec["arch"], rec["width"], rec["hidden"]), (ad.describe(), rec["arch"])
    assert tok.bos_token_id == rec["bos_id"], (tok.bos_token_id, rec["bos_id"])
    n_up = rec["n_layers"] - 1
    pos = np.asarray(rec["background"]["positions"], np.int64)
    kw = dict(tokens=corpus["tokens"], length=RA.LENGTH, bos=tok.bos_token_id, batch=batch, hw=rec["window"], device=device)
    sc = E5.layer_scales(ad, model, pos, n_up, **kw)
    mem = [(int(l), int(k)) for l, k, _ in rec["members"]]
    L, K = [l for l, _ in mem], [k for _, k in mem]
    ab = np.array([a for _, _, a in rec["members"]], np.float64)
    r = t5_bank(out / "t5" / f"{mk}.npz", ad, mem, ab, sc["abar"].cpu().numpy()[L, K], sc["abar_half"].cpu().numpy()[L, K], sc, pos)
    z = np.load(out / "t5" / f"{mk}.npz")
    n = t5_samples(fh, mk, mem, {"write": r["write"], "abar_m": z["abar"], "dnorm": z["dnorm"]}, pos, corpus["tokens"], tok.bos_token_id, rec["window"],
                   NH.decoder(tok))
    out_rec = {k: v for k, v in r.items() if k != "write"}
    out_rec.update({"repo": rec["repo"], "revision": rec["revision"], "adapter": ad.describe(), "n_samples": n, "seconds": round(time.time() - t0, 1),
                    "forwards": int(len(pos)), "tokens_forwarded": RA.n_forwarded(pos, RA.LENGTH, tok.bos_token_id, rec["window"])})
    return out_rec


def t5_olmo(out: pathlib.Path, step: str) -> dict:
    d = out.parent / step
    m = json.loads((d / "metrics.json").read_text())
    assert m["stage"] == "e5_t12" and m["status"] == "done" and (d / "e5" / "t5_o7.npz").exists(), (d, m.get("status"))
    shutil.copy(d / "e5" / "t5_o7.npz", out / "t5" / "o7.npz")
    return {**m["t5_o7"], "from": str(d / "e5" / "t5_o7.npz"), "check_capture": m["gates"]["t5_capture"]}


def run_t5(out: pathlib.Path, items_sha256: str, t5items: dict, olmo_step: str, load, device: str, batch: int, extra: dict) -> dict:
    """T5: Olmo's bank from the sibling t12 step, then each model of T5_MODELS loaded by load(mk, rec) -> (model, tok, corpus), measured, freed."""
    (out / "t5").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    m = {"stage": "e5_t5", "items_sha256": items_sha256, "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"), "status": "running",
         "rule": "per member |ā_k| ||gamma_post ⊙ W_down[:, k]|| s_l(x) / rms(h_l(x)) at the last token of 200 background windows; the median over positions "
                 "per member, then the median and 90th percentile over members", "models": {}, "gates": {}, **extra}
    NH.dump(out / "metrics.json", m)
    m["models"]["o7"] = t5_olmo(out, olmo_step)
    NH.dump(out / "metrics.json", m)
    with open(out / "samples" / "e5.jsonl", "a") as fh:
        for mk in T5_MODELS:
            rec = t5items[mk]
            t0 = time.time()
            model, tok, corpus = load(mk, rec)
            load_s = round(time.time() - t0, 1)
            m["models"][mk] = {**t5_model(out, mk, rec, model, tok, corpus, device, batch, fh), "load_s": load_s}
            m["gates"][f"t5_capture_{mk}"] = m["models"][mk]["check"]
            NH.dump(out / "metrics.json", m)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    m["status"] = "done"
    m["memory_max_allocated_gb"] = RB.memory_gb(device) if device == "cuda" else None
    NH.dump(out / "metrics.json", m)
    for mk in T5_MODELS:
        chk = m["models"][mk]["check"]
        assert chk["pass"] and chk["failing_input_failed"], (mk, chk)
    return m


# ------------------------------------------------------------------------------------------ entry points
def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["t12", "t4", "t5"])
    ap.add_argument("--items", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--t", default=None, choices=["T_15"])
    ap.add_argument("--revision", default=None)
    ap.add_argument("--olmo-step", default="t12", help="t5: the sibling step directory that holds t12's T5 bank")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--full-batch", type=int, default=8)
    ap.add_argument("--scan-tokens", type=int, default=20_570_743)
    ap.add_argument("--harvest-window", type=int, default=UR.HARVEST_WINDOW)
    return ap


def env() -> dict:
    return {"device_name": torch.cuda.get_device_name(0), "transformers": __import__("transformers").__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()}


def main():
    a = parser().parse_args()
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg.diff_harvest import hf_online_setup
    from infra.pg.matrix_models import KNOWN as MKNOWN, vocab_sha256
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    out = pathlib.Path(a.out)
    items_path = fetch(a.items, f"instruments_e5_{a.stage}_items.json")
    a.items_sha256 = hashlib.sha256(pathlib.Path(items_path).read_bytes()).hexdigest()
    items = json.loads(pathlib.Path(items_path).read_text())
    RB.memory_start("cuda")
    if a.stage == "t12":
        assert a.t == "T_15" and JE.JR.REVISION[a.t] == a.revision, (a.t, a.revision)
        grid = {f"T_{int(g[1:3])}": "stage1-" + g.split("_")[1] for g in items["sources"]["grid"]}
        assert grid[a.t] == a.revision, f"the items' grid has {grid[a.t]} at {a.t}, not {a.revision}"
        t0 = time.time()
        local, sha, rec = download(OLMO, a.revision)
        corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
        t1 = time.time()
        model, tok = NH.load_model(local, "O7")
        model.requires_grad_(False)
        job = T12(out, a, items, model, tok, corpus, "cuda")
        job.m.update({"download": rec, "download_and_corpus_s": round(t1 - t0, 1), "load_s": round(time.time() - t1, 1), "sha": sha, **env()})
        job.write()
        job.run()
        job.m["timing_s"]["wall_after_load"] = round(time.time() - t1, 1)
        job.write()
        print("INSTRUMENTS_E5_T12_OK", json.dumps({"t": a.t, "sha": sha}), flush=True)
    elif a.stage == "t4":
        rec = items["models"]["o7"]
        t0 = time.time()
        local, sha, drec = download(rec["repo"], rec["branch"])
        assert sha == rec["revision"], (sha, rec["revision"])
        corpus = NC.load_tok(fetch_corpus(a.corpus, rec["tokname"]), rec["tokname"])
        assert corpus["tokens_sha256"] == rec["tokens_sha256"], "the corpus is not the harvest bank's"
        t1 = time.time()
        model, tok = NH.load_model(local, rec["nh_key"])
        model.requires_grad_(False)
        job = t4_job(out, rec, a.items_sha256, model, tok, corpus, "cuda", a.batch, a.full_batch)
        job.m.update({"download": drec, "corpus_s": round(t1 - t0, 1), "load_s": round(time.time() - t1, 1), "sha": sha, "e5_stage": "t4", **env()})
        job.write()
        job.run()
        job.m["timing_s"]["wall_after_load"] = round(time.time() - t1, 1)
        job.m["memory_max_allocated_gb"] = RB.memory_gb("cuda")
        job.write()
        del model, job
        torch.cuda.empty_cache()
        repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / "models--allenai--Olmo-3-1025-7B"
        shutil.rmtree(repo_dir)
        assert not repo_dir.exists()
        print("INSTRUMENTS_E5_T4_OK", json.dumps({"sha": sha}), flush=True)
    else:
        def load(mk: str, rec: dict):
            assert MKNOWN[rec["repo"]]["n_layers"] == rec["n_layers"], rec["repo"]
            local, rev = NC.local_snapshot(rec["repo"])
            assert rev == rec["revision"] == MKNOWN[rec["repo"]]["sha"], f"{rec['repo']}: cached {rev}, bank {rec['revision']}"
            corpus = NC.load_tok(fetch_corpus(a.corpus, rec["tokname"]), rec["tokname"])
            assert corpus["tokens_sha256"] == rec["tokens_sha256"], "the corpus is not E1b's"
            model, tok = NH.load_model(local, rec["nh_key"])
            model.requires_grad_(False)
            assert vocab_sha256(tok) == corpus["record"]["vocab_sha256"], "the model's vocabulary differs from the corpus tokenizer's"
            return model, tok, corpus
        m = run_t5(out, a.items_sha256, items["t5"], a.olmo_step, load, "cuda", a.batch, env())
        print("INSTRUMENTS_E5_T5_OK", json.dumps({mk: x["summary"] for mk, x in m["models"].items()}), flush=True)


if __name__ == "__main__":
    main()
