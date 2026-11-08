"""INSTRUMENTS E3 capture job, one O7 T_15 load (studies/instruments/SPEC_instruments.md §5.2, §5.3; amendments 15 to 19). A subclass of
jobs.reuse_ablate.Job like jobs.reuse_lowlayer's; unite_route.capture, reuse_ablate.direct_A, reuse_birth.top_m_excluded and mask_of_members and
reuse_dating.Job.gu are reused unchanged.

    python -m infra.aws_pg.jobs.instruments_e3 --t T_15 --revision stage1-step1413814 --items <s3 or local e3_items.json> --corpus <s3> --out <dir>

For every pole of the items (run order: the known-number poles, then the rest in item order), its 50 T_15 contexts through unite_route.capture to the
pole's layer L (256-token window ending at the marked token, BOS first, the harvest's 2,048-token window bounds; forward stopped after L), and
    A = reuse_ablate.direct_A [50, L W] fp32: A_k(x) = sigma a_k(x) s_l(x) [c_g(x) G.d_k + c_u(x) U.d_k] over every upstream MLP neuron k = l W + n.
    c [50, L W] = sigma s_l(x) [c_g(x) G.d_k + c_u(x) U.d_k] (coefficient()), so A = a * c; c_wrong the same with d_k read from layer l + 1.
Banked per pole, e3/<pole>.npz (written before metrics.json is): per context the 200 largest signed A (top_idx flat index l W + n, top_A, top_a,
top_c, top_c_wrong); U = the sorted union of the 50 top-200 sets; A_U = A[:, U] / A_U_scale fp16 [50, |U|] with A_U_scale [50] fp32
the row's max |A[:, U]| (fp16 keeps 11 bits relative to the row max; raw A of 1e-6 would sit in fp16's subnormal range); U_in_B = SPEC4's T_15 B_t flag per column of U;
A_norm (L2 of the full row) and U_energy (share of the row's squared L2 in U); g, u, a_P = sign silu(g) u (float64) of the pole at the marked token;
rank, tok, doc, token_id, token_json (each marked-token string JSON-encoded: a numpy str array drops a trailing NUL); ac_rel and ac_rel_wrong per context.
Gates, asserted (each failing input computed and asserted to fail):
  marked token: the corpus token at every context equals the items' token_id (asserted per pole before the capture).
  A = a * c: per context max_k |A - a c| / max_k |A| <= 1e-5 on every context of every pole; failing input c_wrong (d_k from layer l + 1), which must
      exceed 1e-5. Asserted after every pole.
  known number: for the 20 known poles, D = top 50 of the sum over the case's J ranks of A with SPEC10's own load's B_t excluded
      (reuse_birth.top_m_excluded), overlap |D & D'(J)| / 50 with SPEC10's banked own_direct_m50 set; median over the 20 cases >= 0.9; failing
      input the contexts in reversed rank order (rank r read as rank 49 - r), whose median must be < 0.9. Asserted after the 20 known poles.
  completeness: every pole's npz read back: keys, shapes, U the union of top_idx, finite values, tok equal to the items'.
metrics.json rewritten after every pole. samples/e3.jsonl (rewritten when a sample pole is done): for the items' 5 sample poles the 50 contexts
(marked token, 64 tokens before and 16 after, a_P, the bank's a) and the top-5 drivers of ranks 0, 1 and 2; C from the items; V, K and the reader's
fields null (driver_partition.py fills V and K offline).
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

from infra.aws_pg.jobs import reuse_ablate as JR
from infra.aws_pg.jobs import reuse_dating as RD
from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg import unite_route as UR

N_CTX, K_TOP, M_KNOWN = 50, 200, 50
AC_TOL, KNOWN_LINE = 1e-5, 0.9
SAMPLE_RANKS, SAMPLE_TOP = (0, 1, 2), 5
NPZ_KEYS = ("t", "pole", "layer", "width", "rank", "tok", "doc", "token_id", "token_json", "gP", "uP", "aP", "top_idx", "top_A", "top_a", "top_c",
            "top_c_wrong", "U", "A_U", "A_U_scale", "U_in_B", "A_norm", "U_energy", "ac_rel", "ac_rel_wrong")


# ------------------------------------------------------------------------------------------ pure pieces (the dry run calls them directly)
def coefficient(rows: UR.Rows, cap: dict, unit: dict, shift: int = 0) -> torch.Tensor:
    """c [N, L W]: sigma s_l(x) (c_g(x) G.d_k + c_u(x) U.d_k), so that direct_A = a * c. shift != 0 reads d_k from layer l + shift (the failing input)."""
    assert len(unit["terms"]) == 1, "a single pole"
    L, t = unit["layer"], unit["terms"][0]
    gc, uc = UR.term_gu(cap, unit)
    cg, cu = UR.coefficients(gc[:, 0], uc[:, 0])
    lo = rows.loads(L, t["n"], shift)
    return torch.cat([t["w"] * t["sign"] * cap["s"][l][:, None] * (cg[:, None] * lo[l][0][None] + cu[:, None] * lo[l][1][None]) for l in range(L)], 1)


def ac_rel(A: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> np.ndarray:
    """Per row max |A - a c| / max |A|."""
    den = A.abs().amax(1)
    assert bool((den > 0).all()), "a context with A identically 0"
    return ((A - a * c).abs().amax(1) / den).cpu().numpy().astype(np.float64)


def gate_ac(rel, rel_wrong, tol: float = AC_TOL) -> dict:
    rel, rel_wrong = np.asarray(rel, np.float64), np.asarray(rel_wrong, np.float64)
    return {"max_rel": float(rel.max()), "tol": tol, "n_contexts": int(rel.size), "pass": bool(rel.max() <= tol),
            "failing_input_max_rel": float(rel_wrong.max()), "failing_input_min_rel": float(rel_wrong.min()), "failing_input_failed": bool(rel_wrong.max() > tol)}


def overlap(D, bank) -> float:
    a, b = {tuple(map(int, x)) for x in D}, {tuple(map(int, x)) for x in bank}
    assert len(a) == len(D) and len(b) == len(bank) == M_KNOWN, (len(a), len(D), len(b))
    return len(a & b) / len(b)


def gate_known(per_case: dict, line: float = KNOWN_LINE) -> dict:
    ov = np.array([v["overlap"] for v in per_case.values()])
    ow = np.array([v["failing_input_overlap"] for v in per_case.values()])
    return {"median_overlap": float(np.median(ov)), "min_overlap": float(ov.min()), "n_below_line": int((ov < line).sum()), "line": line,
            "n_cases": int(ov.size), "pass": bool(np.median(ov) >= line), "failing_input_median_overlap": float(np.median(ow)),
            "failing_input_failed": bool(np.median(ow) < line)}


def gate_complete(banked: dict, poles: list, t: str) -> dict:
    """banked {pole: {key: array}} read back from e3/<pole>.npz; poles the items' pole records."""
    missing = [p["pole"] for p in poles if p["pole"] not in banked]
    bad = {}
    for p in poles:
        if p["pole"] in missing:
            continue
        z, L, why = banked[p["pole"]], p["layer"], []
        nU = len(z["U"]) if "U" in z else -1
        if sorted(z) != sorted(NPZ_KEYS):
            why.append(f"keys {sorted(z)}")
        elif str(z["t"]) != t or str(z["pole"]) != p["pole"] or int(z["layer"]) != L:
            why.append("t, pole or layer")
        elif any(z[k].shape != (N_CTX, K_TOP) for k in ("top_idx", "top_A", "top_a", "top_c", "top_c_wrong")) or z["A_U"].shape != (N_CTX, nU) \
                or z["U_in_B"].shape != (nU,) or any(z[k].shape != (N_CTX,) for k in ("rank", "tok", "aP", "A_U_scale", "A_norm", "U_energy", "ac_rel")):
            why.append("shape")
        elif not np.array_equal(z["tok"], [c["tok"] for c in p["contexts"]]) or not np.array_equal(z["rank"], np.arange(N_CTX)):
            why.append("tok or rank differ from the items")
        elif not np.array_equal(z["U"], np.unique(z["top_idx"])) or int(z["U"].max()) >= L * int(z["width"]):
            why.append("U is not the union of the top-200 sets")
        elif not all(np.isfinite(z[k].astype(np.float64)).all() for k in ("aP", "top_A", "top_a", "top_c", "A_U", "A_U_scale", "A_norm", "U_energy")):
            why.append("non-finite")
        if why:
            bad[p["pole"]] = why
    return {"n_poles_expected": len(poles), "n_poles_present": len(poles) - len(missing), "missing": missing, "bad": bad, "pass": not missing and not bad}


# ------------------------------------------------------------------------------------------ the job
class Job(JR.Job):
    def __init__(self, out, a, items: dict, model, tok, corpus: dict, device: str):
        self.out, self.a, self.items, self.model, self.tok, self.device = pathlib.Path(out), a, items, model, tok, device
        for d in ("", "e3", "samples"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.tokens, self.offsets = corpus["tokens"], corpus["doc_offsets"]
        self.bos = tok.bos_token_id
        assert self.bos is not None, "the instrument prepends a BOS; this tokenizer defines none"
        self.tower = NH.text_tower(model)
        self.layers = list(self.tower.layers)
        self.W = int(self.layers[0].mlp.gate_proj.weight.shape[0])
        self.rows = UR.Rows(model)
        self.hw = a.harvest_window
        self.dec = NH.decoder(tok)
        assert NH.act_name(self.layers[0].mlp.act_fn) == "silu", "a_P and the coefficients assume silu"
        n_up = len(self.layers) - 1
        assert items["B_t"]["n_up"] == n_up, (items["B_t"]["n_up"], n_up)
        self.flag_B = RB.mask_of_members([(l, k, 0.0) for l, k in items["B_t"]["members"]], n_up, self.W).reshape(-1)
        self.mask_known = torch.as_tensor(RB.mask_of_members([(l, k, 0.0) for l, k in items["known"]["B_t_members"]], n_up, self.W), device=device)
        self.known_cases = {c["pole"]: c for c in items["known"]["cases"]}
        poles = items["poles"]
        assert all(p["layer"] >= 1 for p in poles), "a pole at layer 0 has no upstream MLP neuron"
        assert set(self.known_cases) <= {p["pole"] for p in poles}, "a known pole is not an item"
        self.order = [p for p in poles if p["pole"] in self.known_cases] + [p for p in poles if p["pole"] not in self.known_cases]
        self.sample_poles = set(items["sample_poles"])
        self.samples_rows = {}
        self.m = {"stage": "instruments_e3", "t": a.t, "revision": a.revision, "items_sha256": a.items_sha256, "harvest_window": self.hw,
                  "window": RA.LENGTH, "batch": a.batch, "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"), "n_layers": len(self.layers),
                  "width": self.W, "dtype": str(next(model.parameters()).dtype), "k_top": K_TOP, "timing_s": {}, "tokens_forwarded": 0, "captures": 0,
                  "B_t": {k: items["B_t"][k] for k in ("path", "sha256", "count", "cut")}, "known_B_t": {"path": items["known"]["B_t_path"],
                  "sha256": items["known"]["B_t_sha256"]}, "poles": {}, "known_cases": {},
                  "A": "reuse_ablate.direct_A at the marked token of the 256-token window (unite_route.capture, BOS first), forward stopped after L",
                  "drivers": f"per context the {K_TOP} largest signed A; U the union over the {N_CTX} contexts"}
        self.write()

    # -------------------------------------------------------------------------------------- one pole
    @torch.inference_mode()
    def pole(self, p: dict) -> dict:
        t0 = time.time()
        unit = UR.pole_unit(p["pole"])
        L, n, sign = unit["layer"], unit["terms"][0]["n"], unit["terms"][0]["sign"]
        assert (L, n, "+" if sign > 0 else "-") == (p["layer"], p["neuron"], p["sign"]), p["pole"]
        ctx = p["contexts"]
        assert [c["rank"] for c in ctx] == list(range(N_CTX)), (p["pole"], "contexts not in rank order 0..49")
        toks = np.asarray([c["tok"] for c in ctx], np.int64)
        assert np.array_equal(np.asarray(self.tokens[toks], np.int64), [c["token_id"] for c in ctx]), (p["pole"], "the marked token differs from the items'")
        cap = UR.capture(self.model, toks, L, {L}, **self.kw(RA.LENGTH))
        self.m["captures"] += len(toks)
        self.m["tokens_forwarded"] += RA.n_forwarded(toks, RA.LENGTH, self.bos, self.hw)
        A = RA.direct_A(self.rows, cap, unit, self.W)
        assert A.shape == (N_CTX, L * self.W), A.shape
        a_all = torch.cat([cap["a"][l] for l in range(L)], 1)
        c = coefficient(self.rows, cap, unit)
        cw = coefficient(self.rows, cap, unit, shift=1)
        rel, relw = ac_rel(A, a_all, c), ac_rel(A, a_all, cw)
        top_A, top_idx = torch.topk(A, K_TOP, dim=1, sorted=True)
        U = torch.unique(top_idx)
        A_U = A[:, U]
        scale = A_U.abs().amax(1)
        assert bool((scale > 0).all()), (p["pole"], "a context with A identically 0 on U")
        norm = A.norm(dim=1)
        g, u, aP = RD.Job.gu(cap, unit)
        Un = U.cpu().numpy().astype(np.int64)
        rec_known = None
        if p["pole"] in self.known_cases:
            k = self.known_cases[p["pole"]]
            J, rev = list(k["J_ranks"]), [N_CTX - 1 - r for r in k["J_ranks"]]
            D = RB.top_m_excluded(A[J].sum(0), M_KNOWN, self.W, self.mask_known)
            Dw = RB.top_m_excluded(A[rev].sum(0), M_KNOWN, self.W, self.mask_known)
            rec_known = {"c": k["c"], "n_J": len(J), "overlap": overlap(D, k["D"]), "failing_input_overlap": overlap(Dw, k["D"]), "bank": k["bank"],
                         "bank_sha256": k["bank_sha256"]}
            self.m["known_cases"][p["pole"]] = rec_known
        z = {"t": self.a.t, "pole": p["pole"], "layer": L, "width": self.W, "rank": np.arange(N_CTX), "tok": toks, "doc": np.asarray([c["doc"] for c in ctx], np.int64),
             "token_id": np.asarray([c["token_id"] for c in ctx], np.int64), "token_json": np.asarray([json.dumps(c["token_string"], ensure_ascii=False) for c in ctx]),
             "gP": g, "uP": u, "aP": aP, "top_idx": top_idx.cpu().numpy().astype(np.int64), "top_A": top_A.cpu().numpy().astype(np.float32),
             "top_a": a_all.gather(1, top_idx).cpu().numpy().astype(np.float32), "top_c": c.gather(1, top_idx).cpu().numpy().astype(np.float32),
             "top_c_wrong": cw.gather(1, top_idx).cpu().numpy().astype(np.float32), "U": Un, "A_U": (A_U / scale[:, None]).cpu().numpy().astype(np.float16), "A_U_scale": scale.cpu().numpy().astype(np.float32),
             "U_in_B": self.flag_B[Un], "A_norm": norm.cpu().numpy().astype(np.float32),
             "U_energy": (A_U.pow(2).sum(1) / norm.pow(2)).cpu().numpy().astype(np.float32), "ac_rel": rel, "ac_rel_wrong": relw}
        assert np.isfinite(z["A_U"].astype(np.float32)).all(), (p["pole"], "A_U overflows fp16")
        path = self.out / "e3" / f"{p['pole']}.npz"
        np.savez_compressed(path, **z)
        gate = gate_ac(rel, relw)
        a_bank = np.asarray([c["a_bank"] for c in ctx], np.float64)
        self.m["poles"][p["pole"]] = {"layer": L, "n": N_CTX, "n_U": int(len(Un)), "n_U_in_B": int(z["U_in_B"].sum()), "median_U_energy": float(np.median(z["U_energy"])),
                                      "s_P": float(np.median(aP)), "median_rel_diff_aP_vs_signed_bank_a": float(np.median(np.abs(aP - sign * a_bank) / np.abs(a_bank))),
                                      "ac": gate, "known": rec_known, "path": str(path.relative_to(self.out)), "seconds": round(time.time() - t0, 2)}
        acs = self.m["ac"] = self.m.get("ac", {"max_rel": 0.0, "failing_input_min_of_max_rel": float("inf"), "n_contexts": 0, "tol": AC_TOL})
        acs["max_rel"] = max(acs["max_rel"], gate["max_rel"])
        acs["failing_input_min_of_max_rel"] = min(acs["failing_input_min_of_max_rel"], gate["failing_input_max_rel"])
        acs["n_contexts"] += gate["n_contexts"]
        acs["pass"], acs["failing_input_failed"] = acs["max_rel"] <= AC_TOL, acs["failing_input_min_of_max_rel"] > AC_TOL
        self.m["n_done"]["poles"] += 1
        if p["pole"] in self.sample_poles:
            self.sample(p, z, sign, a_bank)
        self.write()
        assert gate["pass"], f"A = a * c failed on {p['pole']}: {gate['max_rel']}"
        assert gate["failing_input_failed"], f"A = a * c's failing input (c from layer l + 1) passed on {p['pole']}: {gate['failing_input_max_rel']}"
        del cap, A, a_all, c, cw, A_U
        return rec_known

    def sample(self, p: dict, z: dict, sign: float, a_bank):
        W = self.W
        drivers = []
        for r in SAMPLE_RANKS:
            drivers.append({"rank": r, "token_string": json.loads(str(z["token_json"][r])), "top": [
                {"layer": int(j // W), "neuron": int(j % W), "A": float(z["top_A"][r, i]), "a": float(z["top_a"][r, i]), "c": float(z["top_c"][r, i]),
                 "in_B_t": bool(self.flag_B[j])} for i, j in enumerate(z["top_idx"][r, :SAMPLE_TOP])]})
        self.samples_rows[p["pole"]] = {
            "pole": p["pole"], "layer": p["layer"], "sign": p["sign"], "band": p["band"], "sources": p["sources"], "t": self.a.t,
            "contexts": [{"rank": r, "tok": int(z["tok"][r]), "token_string": json.loads(str(z["token_json"][r])), "aP": float(z["aP"][r]), "a_bank": float(a_bank[r]),
                          **NH.context(self.tokens, self.offsets, int(z["tok"][r]), self.dec)} for r in range(N_CTX)],
            "drivers": drivers, "C": p["C"], "V": None, "K": None, "driver_clusters": None, "reader": None,
            "note": "C from the items; V, K and driver_clusters filled by studies/instruments/scripts/driver_partition.py; reader by the E3 reader stage"}
        rows = [self.samples_rows[q] for q in sorted(self.samples_rows)]
        tmp = self.out / "samples" / "e3.jsonl.tmp"
        tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
        os.replace(tmp, self.out / "samples" / "e3.jsonl")

    # -------------------------------------------------------------------------------------- gates read after the loop parts
    def known(self):
        rec = gate_known(self.m["known_cases"])
        rec.update({"rule": f"D = top {M_KNOWN} of the sum over J's ranks of A, SPEC10's load's B_t excluded; overlap with SPEC10's own_direct_m50 set; "
                            f"median over the cases >= {KNOWN_LINE}", "failing_input": "the contexts in reversed rank order (rank r read as 49 - r)"})
        self.m["known_number"] = rec
        self.write()
        assert rec["pass"], f"known number failed: median overlap {rec['median_overlap']}"
        assert rec["failing_input_failed"], f"known number's failing input passed: median overlap {rec['failing_input_median_overlap']}"

    def completeness(self):
        banked = {}
        for p in self.items["poles"]:
            path = self.out / "e3" / f"{p['pole']}.npz"
            if path.exists():
                with np.load(path) as z:
                    banked[p["pole"]] = {k: z[k] for k in z.files}
        rec = gate_complete(banked, self.items["poles"], self.a.t)
        self.m["completeness"] = rec
        self.write()
        assert rec["pass"], f"completeness failed: missing {rec['missing'][:5]}, bad {list(rec['bad'].items())[:5]}"

    def run(self):
        self.m["n_to_read"] = {"poles": len(self.order), "contexts": len(self.order) * N_CTX, "known_poles": len(self.known_cases)}
        self.m["n_done"] = {"poles": 0}
        t0 = time.time()
        for p in self.order[:len(self.known_cases)]:
            self.pole(p)
        self.known()
        self.m["timing_s"]["known"] = round(time.time() - t0, 1)
        t1, tok1 = time.time(), self.m["tokens_forwarded"]
        for p in self.order[len(self.known_cases):]:
            self.pole(p)
        dt = time.time() - t1
        self.m["timing_s"]["rest"] = round(dt, 1)
        self.m["rest_tokens_per_second"] = (self.m["tokens_forwarded"] - tok1) / dt
        assert set(self.samples_rows) == self.sample_poles, (sorted(self.samples_rows), sorted(self.sample_poles))
        self.completeness()
        assert self.m["n_done"]["poles"] == self.m["n_to_read"]["poles"], (self.m["n_done"], self.m["n_to_read"])
        self.m["memory_max_allocated_gb"] = RB.memory_gb(self.device)
        self.m["status"] = "done"
        self.write()


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--t", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--batch", type=int, default=32, help="contexts per unite_route.capture forward (REUSE's)")
    ap.add_argument("--full-batch", type=int, default=8)
    ap.add_argument("--harvest-window", type=int, default=UR.HARVEST_WINDOW)
    return ap


def check_items(items: dict, t: str):
    assert items["t"] == t == "T_15" and items["grid"] == "t15_step1413814", (items["t"], items["grid"], t)
    assert len(items["poles"]) == items["counts"]["poles"] and len(items["known"]["cases"]) == items["counts"]["known_poles"] == 20
    assert len(items["sample_poles"]) == 5 and items["known"]["m"] == M_KNOWN and items["known"]["line"] == KNOWN_LINE


def main():
    a = parser().parse_args()
    assert JR.REVISION[a.t] == a.revision, f"{a.t} is {JR.REVISION[a.t]}, not {a.revision}"
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    items_path = fetch(a.items, "instruments_e3_items.json")
    a.items_sha256 = hashlib.sha256(pathlib.Path(items_path).read_bytes()).hexdigest()
    items = json.loads(pathlib.Path(items_path).read_text())
    check_items(items, a.t)
    t0 = time.time()
    local, sha, rec = download("allenai/Olmo-3-1025-7B", a.revision)
    corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
    assert corpus["tokens_sha256"] == items["corpus"]["tokens_sha256"], "the corpus is not the one the grid scored"
    t1 = time.time()
    RB.memory_start("cuda")
    model, tok = NH.load_model(local, "O7")
    model.requires_grad_(False)
    job = Job(a.out, a, items, model, tok, corpus, "cuda")
    job.m.update({"download": rec, "download_and_corpus_s": round(t1 - t0, 1), "load_s": round(time.time() - t1, 1), "sha": sha,
                  "device_name": torch.cuda.get_device_name(0), "transformers": __import__("transformers").__version__, "torch": torch.__version__,
                  "cuda": torch.version.cuda, "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9})
    job.m["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    job.write()
    t2 = time.time()
    job.run()
    job.m["timing_s"]["wall_after_load"] = round(time.time() - t2, 1)
    job.write()
    del model
    torch.cuda.empty_cache()
    repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / "models--allenai--Olmo-3-1025-7B"
    shutil.rmtree(repo_dir)
    assert not repo_dir.exists()
    print("INSTRUMENTS_E3_OK", json.dumps({"t": a.t, "poles": job.m["n_done"]["poles"], "captures": job.m["captures"]}), flush=True)


if __name__ == "__main__":
    main()
