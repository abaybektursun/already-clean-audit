"""UNITE Arm 1 job, one O7 checkpoint per call (studies/unite/PREREG.md §9).

    python -m infra.aws_pg.jobs.unite_route --stage smoke --t T_15 --revision stage1-step1413814 --items <s3> --corpus <s3> --out <dir>      # J0
    python -m infra.aws_pg.jobs.unite_route --stage route --t T_15 --revision ... --scan --window-gate --noise --items <s3> --corpus <s3> --out <dir>      # J1 first step
    python -m infra.aws_pg.jobs.unite_route --stage route --t T_13 --revision ... --from-step <T_15 out dir> --items <s3> --corpus <s3> --out <dir>
    python -m infra.aws_pg.jobs.unite_route --stage grad --t T_15 --revision ... --from-step <dir> --items <s3> --corpus <s3> --out <dir>      # gradient readout

`Job.run` takes a loaded model, so the dry run (`unite_dryrun`) drives the same code on a tiny random OLMo 3. metrics.json is rewritten inside each loop.
Gates known number and completeness assert after their numbers are written, so a failed gate ends the job with a non-zero status.
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

from infra.pg import neurons_harvest as NH
from infra.pg import unite_route as UR
from infra.pg import unite_stats as US

KNOWN_CTX = {"smoke": 50, "route": 16, "grad": 16}


class Job:
    def __init__(self, out, a, items: dict, model, tok, corpus: dict, device: str):
        self.out, self.a, self.items, self.model, self.tok, self.device = pathlib.Path(out), a, items, model, tok, device
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "grams").mkdir(exist_ok=True)
        (self.out / "samples").mkdir(exist_ok=True)
        self.tokens, self.offsets = corpus["tokens"], corpus["doc_offsets"]
        self.bos = tok.bos_token_id
        self.tower = NH.text_tower(model)
        self.layers = list(self.tower.layers)
        self.W = int(self.layers[0].mlp.gate_proj.weight.shape[0])
        self.rows = UR.Rows(model)
        self.hw = a.harvest_window
        assert NH.act_name(self.layers[0].mlp.act_fn) == "silu", "the coefficients assume silu"
        self.m = {"stage": a.stage, "t": a.t, "revision": a.revision, "items_sha256": a.items_sha256, "harvest_window": self.hw, "started_utc": NH.utc(),
                  "pg_git_sha": os.environ.get("PG_GIT_SHA"), "n_layers": len(self.layers), "width": self.W, "dtype": str(next(model.parameters()).dtype),
                  "timing_s": {}, "tokens_forwarded": 0}
        self.write()

    def write(self):
        self.m["updated_utc"] = NH.utc()
        NH.dump(self.out / "metrics.json", self.m)

    def kw(self, length):
        return dict(tokens=self.tokens, length=length, bos=self.bos, batch=self.a.batch if length is not None else self.a.full_batch, hw=self.hw, device=self.device)

    # -------------------------------------------------------------------------------------- gates at every load
    def known(self, n_ctx: int) -> dict:
        t0 = time.time()
        k = self.items["known"]
        out = {}
        ours, bank, wrong, errs, wrong_errs = [], [], [], [], []
        for i, p in enumerate(k["poles"]):
            L, n, _ = NH.parse_pole(p)
            toks = np.asarray(k["tok"][i])[:n_ctx]
            cap = UR.capture(self.model, toks, L, {L}, **self.kw(None))
            a = cap["a"][L][:, n].cpu().numpy()
            b, w = np.asarray(k["a_bank"][self.a.t][i], float)[:n_ctx], np.asarray(k["a_wrong"][self.a.t][i], float)[:n_ctx]
            c = UR.completeness(self.rows, cap, L, n)
            cw = UR.completeness(self.rows, cap, L, n, shift=1)
            out[p] = {"known_ratio": US.known_ratio(a, b), "wrong_ratio": US.known_ratio(a, w), "err_g_median": float(np.median(c["err_g"])), "err_u_median": float(np.median(c["err_u"])),
                      "wrong_layer_err_g_median": float(np.median(cw["err_g"])), "wrong_layer_err_u_median": float(np.median(cw["err_u"])),
                      "samples": [{"tok": int(toks[j]), **NH.context(self.tokens, self.offsets, int(toks[j]), NH.decoder(self.tok)), "a_ours": float(a[j]), "a_bank": float(b[j])} for j in range(2)]}
            errs += [c["err_g"], c["err_u"]]
            wrong_errs += [cw["err_g"], cw["err_u"]]
            ours.append(a), bank.append(b), wrong.append(w)
        # a pole's own max is the unit, so the ratio is the median over poles of the per-pole ratio
        ratio = float(np.median([v["known_ratio"] for v in out.values()]))
        wr = float(np.median([v["wrong_ratio"] for v in out.values()]))
        self.m["known_number"] = {**US.gate_known(ratio, wr), "per_pole": out, "n_ctx": n_ctx, "wrong_rule": k["wrong_rule"]}
        self.m["completeness"] = US.gate_complete(float(np.median(np.concatenate(errs))), float(np.median(np.concatenate(wrong_errs))))
        self.m["timing_s"]["known"] = round(time.time() - t0, 1)
        self.write()
        NH.dump(self.out / "samples" / "known_number.json", out)
        assert self.m["known_number"]["pass"], f"known number failed: {self.m['known_number']['ratio']}"
        assert self.m["known_number"]["failing_input_failed"], f"known number's failing input passed: {wr}"
        assert self.m["completeness"]["pass"], f"completeness failed: {self.m['completeness']}"
        assert self.m["completeness"]["failing_input_failed"], f"completeness's failing input passed: {self.m['completeness']}"
        return out

    # -------------------------------------------------------------------------------------- window gate
    def window_gate(self) -> dict:
        t0 = time.time()
        rel, rel16 = [], []
        for p, toks in self.items["tok"].items():
            L, n, _ = NH.parse_pole(p)
            toks = np.asarray(toks)
            bank = np.asarray(self.items["exemplar_t15"][p], float)
            ok = toks >= 0
            for length, sink in ((256, rel), (16, rel16)):
                a = UR.capture(self.model, toks[ok], L, {L}, **self.kw(length))["a"][L][:, n].cpu().numpy()
                sink.append(np.abs(a - bank[ok]) / np.abs(bank[ok]))
        g = US.gate_window(np.concatenate(rel), np.concatenate(rel16))
        self.m["window_gate"] = g
        self.m["timing_s"]["window_gate"] = round(time.time() - t0, 1)
        self.write()
        NH.dump(self.out / "window.json", {"window": g["window"]})
        return g

    # -------------------------------------------------------------------------------------- planted scan
    def scan(self):
        t0 = time.time()
        plants = self.items["planted"]
        n_windows = self.a.scan_tokens // self.hw

        def progress(w1):
            if w1 % (self.a.scan_batch * 50) == 0:
                self.m["scan_progress"] = {"windows": w1, "of": n_windows, "tokens_per_second": w1 * self.hw / (time.time() - t0), "utc": NH.utc()}
                self.write()
        res = UR.scan_planted(self.model, plants, tokens=self.tokens, n_windows=n_windows, window=self.hw, bos=self.bos, batch=self.a.scan_batch, device=self.device, progress=progress)
        sel = UR.select_planted(plants, res, self.tokens)
        NH.dump(self.out / "planted_contexts.json", sel)
        kept = [v for v in sel.values() if v["kept"]]
        self.m["scan"] = {"tokens": n_windows * self.hw, "seconds": round(time.time() - t0, 1), "tokens_per_second": n_windows * self.hw / (time.time() - t0),
                          "kept_by_layer": {str(L): sum(v["layer"] == L for v in kept) for L in sorted({p["layer"] for p in plants})},
                          "reserve_used": sum(v.get("used_as") == "reserve" for v in kept), "why_dropped": {w: sum(v.get("why") == w for v in sel.values()) for w in {v.get("why") for v in sel.values()} if w}}
        self.write()
        return sel

    # -------------------------------------------------------------------------------------- the Grams
    def background(self, gu_layers, upto, length):
        sets = UR.background_positions(self.t_index, self.tokens_windowed, self.hw)
        toks = np.concatenate([sets["e1"], sets["e2"], sets["noise"]])
        t0 = time.time()
        bg = UR.capture(self.model, toks, upto, gu_layers, **self.kw(length))
        self.m["timing_s"]["background"] = round(time.time() - t0, 1)
        return sets, bg

    @property
    def t_index(self):
        return int(self.a.t.split("_")[1])

    @property
    def tokens_windowed(self):
        return self.a.scan_tokens // self.hw * self.hw

    def route(self, length, planted: dict | None):
        units = [(UR.pole_unit(p), np.asarray(self.items["tok"][p])) for p in self.items["units"][self.a.t]]
        kept = {} if planted is None else {u: r for u, r in planted.items() if r["kept"]}
        units += [(UR.planted_unit(r), np.asarray(r["tok"])) for r in kept.values()]
        assert units, "no units at this checkpoint"
        upto = max(u["layer"] for u, _ in units)
        sets, bg = self.background({u["layer"] for u, _ in units}, upto, length)
        t0, n_tok = time.time(), 0
        self.m["n_units"] = len(units)
        grams = {}
        for i, (u, toks) in enumerate(units):
            valid = np.flatnonzero(toks >= 0)
            assert u["unit"].startswith("PL") or valid.size == toks.size, f"{u['unit']}: an empty top-50 slot; job ranks would not index the Gram"
            cap = UR.capture(self.model, toks[valid], u["layer"], {u["layer"]}, **self.kw(length))
            bank = UR.unit_bank(self.rows, u, cap, bg, sets, noise=self.a.noise and not u["unit"].startswith("PL"))
            np.savez_compressed(self.out / "grams" / f"{u['unit']}.npz", ranks=valid, tok=toks[valid], **bank)
            grams[u["unit"]] = {"M": bank["M"]}
            n_tok += int(valid.size) * (length or self.hw)
            del cap
            if (i + 1) % 10 == 0 or i + 1 == len(units):
                self.m.update({"n_units_done": i + 1, "route_tokens_per_second": n_tok / (time.time() - t0), "route_seconds": round(time.time() - t0, 1)})
                self.write()
        self.m["tokens_forwarded"] += n_tok
        self.m["window_used"] = length or "full"
        self.write()
        NH.dump(self.out / "background.json", {k: v.tolist() for k, v in sets.items()})
        self.samples(units, grams, length)
        return grams

    def pair_record(self, unit: str, side: list, ranks_all, n=3):
        z = np.load(self.out / "grams" / f"{unit}.npz")
        rows = []
        index = {int(r): i for i, r in enumerate(z["ranks"])}
        dec = NH.decoder(self.tok)
        for r in side[:n]:
            i = index[int(r)]
            rec = {"rank": int(r), "tok": int(z["tok"][i]), **NH.context(self.tokens, self.offsets, int(z["tok"][i]), dec), "a_terms": z["a_terms"][i].tolist(),
                   "g": z["g"][i].tolist(), "u": z["u"][i].tolist(), "rise": dict(zip(UR.SOURCES, z["rise"][i].tolist()))}
            if "top_A" in z:
                rec["top_inputs"] = [{"layer": int(z["top_layer"][i, k]), "neuron": int(z["top_neuron"][i, k]), "A": float(z["top_A"][i, k]), "a": float(z["top_a"][i, k])} for k in range(5)]
            rows.append(rec)
        return rows

    def samples(self, units, grams, length):
        """One job scored most shared and one scored least shared per depth tercile, with prompts, top input neurons and activations. Planted pairs likewise."""
        rows = []
        for j in self.items.get("jobs", []):
            if j["pole"] in grams and (f"T_{j['birth']}" == self.a.t or self.a.t == "T_15"):
                o = next(o for o in j["older"] if o["c"] == j["o_star"])
                r = US.stat_T(grams[j["pole"]]["M"], j["ranks"], o["ranks"])
                if np.isfinite(r["T"]):
                    rows.append({"job": j["job"], "pole": j["pole"], "J": j["ranks"], "O": o["ranks"], "tercile": US.tercile(NH.parse_pole(j["pole"])[0]), **r})
        out = []
        for ter in (0, 1, 2):
            sel = sorted((r for r in rows if r["tercile"] == ter), key=lambda r: r["T"])
            for r, tag in ([(sel[-1], "scored most shared"), (sel[0], "scored least shared")] if sel else []):
                out.append({**r, "scored": tag, "J_contexts": self.pair_record(r["pole"], r["J"], None), "O_contexts": self.pair_record(r["pole"], r["O"], None)})
        with open(self.out / "samples" / "pairs.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.m["samples"] = {"pairs_written": len(out), "jobs_scored_for_samples": len(rows), "path": "samples/pairs.jsonl"}
        self.write()

    def identity(self, length):
        """Real pairs with identical prior tokens in different documents, the same context forwarded twice in different batches, and the failing input: pairs from two poles."""
        rec = {"real_pairs": [], "twice": [], "two_poles": []}
        by = {}
        for x in self.items["identity"]:
            by.setdefault(x["pole"], []).append(x["tok"])
        sets, bg = self.background({NH.parse_pole(p)[0] for p in self.items["tok"]}, max(NH.parse_pole(p)[0] for p in self.items["tok"]), length)
        poles = [p for p in self.items["tok"] if NH.parse_pole(p)[0] >= 1]
        pick = poles[:: max(1, len(poles) // 10)][:10]
        rng = np.random.default_rng(0)

        def pair_gram(pole, t1, t2):
            u = UR.pole_unit(pole)
            cap = UR.capture(self.model, np.array([t1, t2]), u["layer"], {u["layer"]}, **{**self.kw(length), "batch": 1})
            return float(US.identity_T(UR.unit_bank(self.rows, u, cap, bg, sets)["M"], 0, 1))
        for p, pairs in by.items():
            if NH.parse_pole(p)[0] >= 1:
                rec["real_pairs"] += [{"pole": p, "tok": x, "T": pair_gram(p, *x)} for x in pairs]
        for i, p in enumerate(pick):
            toks = np.asarray(self.items["tok"][p])
            for t in toks[toks >= 0][:3]:
                rec["twice"].append({"pole": p, "tok": int(t), "T": pair_gram(p, int(t), int(t))})
            q = pick[(i + 1) % len(pick)]
            other = np.asarray(self.items["tok"][q])
            for t, o in zip(toks[toks >= 0][:3], rng.permutation(other[other >= 0])[:3]):
                rec["two_poles"].append({"pole": p, "other": q, "tok": [int(t), int(o)], "T": pair_gram(p, int(t), int(o))})
        primary = rec["real_pairs"] if len(rec["real_pairs"]) >= 20 else rec["twice"]
        self.m["identity"] = {**US.gate_identity([r["T"] for r in primary], [r["T"] for r in rec["two_poles"]]), "n_real_pairs": len(rec["real_pairs"]),
                              "primary": "real pairs" if primary is rec["real_pairs"] else "same context forwarded twice: under 20 real pairs exist",
                              "real_pairs_median_T": float(np.nanmedian([r["T"] for r in rec["real_pairs"]])) if rec["real_pairs"] else None}
        NH.dump(self.out / "identity.json", rec)
        self.write()

    # -------------------------------------------------------------------------------------- stages
    def run_smoke(self):
        self.known(KNOWN_CTX["smoke"])
        # one Gram and one gradient Gram on one known pole, so the whole path meets the real model before J1
        p = self.items["known"]["poles"][0]
        u = UR.pole_unit(p)
        toks = np.asarray(self.items["known"]["tok"][0])[:8]
        sets, bg = self.background({u["layer"]}, u["layer"], 256)
        cap = UR.capture(self.model, toks, u["layer"], {u["layer"]}, **self.kw(256))
        b = UR.unit_bank(self.rows, u, cap, bg, sets)
        t0 = time.time()
        Mg = UR.grad_attr(self.model, toks, u, **self.kw(256))
        self.m["grad_smoke"] = {"shape": list(Mg.shape), "seconds": round(time.time() - t0, 1), "finite": bool(torch.isfinite(Mg).all()), "max_abs": float(Mg.abs().max())}
        self.m["gram_smoke"] = {"pole": p, "M_shape": list(b["M"].shape), "M_diag": np.diag(b["M"]).tolist(), "rise_first": dict(zip(UR.SOURCES, b["rise"][0].tolist())),
                                "transformers": __import__("transformers").__version__}
        self.m["status"] = "done"
        self.write()

    def run_route(self):
        self.known(KNOWN_CTX["route"])
        planted = None
        if self.a.scan:
            planted = self.scan()
        elif self.a.from_step:
            planted = self.from_step_json("planted_contexts.json")
        if self.a.window_gate:
            length = 256 if self.window_gate()["window"] == 256 else None
        elif self.a.from_step:
            length = 256 if self.from_step_json("window.json")["window"] == 256 else None
        else:
            length = 256 if self.a.window == "256" else None
        self.route(length, planted)
        if "identity" in self.items and self.a.t == self.a.identity_at:
            self.identity(length)
        if self.a.also_grad:
            units = [(UR.pole_unit(p), np.asarray(self.items["tok"][p])) for p in self.items["units"][self.a.t]]
            self.grad_units(units, length, self.a.grad_foreign)
        self.m["status"] = "done"
        self.write()

    def grad_units(self, units, length, foreign: bool):
        """The gradient readout's Gram per unit under the frozen centring sets. `foreign` adds a Gram under the foreign neuron's own activation."""
        sets = UR.background_positions(self.t_index, self.tokens_windowed, self.hw)
        (self.out / "grams_grad").mkdir(exist_ok=True)
        t0, n_tok = time.time(), 0
        self.m["n_grad_units"] = len(units)
        kw = {**self.kw(length), "batch": self.a.grad_batch}
        for i, (u, toks) in enumerate(units):
            valid = np.flatnonzero(toks >= 0)
            noise = self.a.noise and not u["unit"].startswith("PL")
            bank = UR.grad_bank(self.model, u, toks[valid], sets, noise=noise, **kw)
            n_pos = valid.size + len(sets["e1"]) + len(sets["e2"]) + (len(sets["noise"]) if noise else 0)
            if foreign and not u["unit"].startswith("PL"):
                t_ = u["terms"][0]
                fu = {"unit": u["unit"], "layer": u["layer"], "terms": [{**t_, "n": self.rows.foreign(u["layer"], t_["n"])}]}
                bank["M_foreign"] = UR.grad_bank(self.model, fu, toks[valid], sets, **kw)["M"]
                n_pos += valid.size + len(sets["e1"]) + len(sets["e2"])
            np.savez_compressed(self.out / "grams_grad" / f"{u['unit']}.npz", ranks=valid, tok=toks[valid], **bank)
            n_tok += n_pos * (length or self.hw)
            self.m.update({"n_grad_units_done": i + 1, "grad_seconds": round(time.time() - t0, 1), "grad_tokens_per_second": n_tok / (time.time() - t0), "grad_batch": self.a.grad_batch})
            self.write()

    def from_step_json(self, name: str):
        """A file of an earlier step: a local directory on the same instance, or an S3 prefix from an earlier job. None if it is not there."""
        src = self.a.from_step
        if src.startswith("s3://"):
            local = self.out / "from_step" / name
            local.parent.mkdir(exist_ok=True)
            subprocess.run(["aws", "s3", "cp", f"{src.rstrip('/')}/{name}", str(local), "--region", "us-east-1", "--only-show-errors"], check=True)
            return json.loads(local.read_text())
        p = pathlib.Path(src) / name
        return json.loads(p.read_text()) if p.exists() else None

    def run_grad(self):
        self.known(KNOWN_CTX["grad"])
        length = 256 if self.from_step_json("window.json")["window"] == 256 else None
        planted = self.from_step_json("planted_contexts.json") or {}
        units = [(UR.pole_unit(p), np.asarray(self.items["tok"][p])) for p in self.items["units"][self.a.t]]
        units += [(UR.planted_unit(r), np.asarray(r["tok"])) for r in planted.values() if r["kept"]]
        self.grad_units(units, length, self.a.grad_foreign)
        self.m["status"] = "done"
        self.write()

    def run(self):
        {"smoke": self.run_smoke, "route": self.run_route, "grad": self.run_grad}[self.a.stage]()


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["smoke", "route", "grad"])
    ap.add_argument("--t", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--window-gate", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--grad-foreign", action="store_true")
    ap.add_argument("--also-grad", action="store_true")
    ap.add_argument("--from-step", default=None)
    ap.add_argument("--window", default="256", choices=["256", "full"])
    ap.add_argument("--identity-at", default="T_15")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--full-batch", type=int, default=8)
    ap.add_argument("--grad-batch", type=int, default=8)
    ap.add_argument("--scan-batch", type=int, default=16)
    ap.add_argument("--scan-tokens", type=int, default=20_570_743)
    ap.add_argument("--harvest-window", type=int, default=UR.HARVEST_WINDOW)
    return ap


def main():
    a = parser().parse_args()
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    items_path = fetch(a.items, f"{a.stage}_{a.t}_items.json")
    a.items_sha256 = hashlib.sha256(pathlib.Path(items_path).read_bytes()).hexdigest()
    items = json.loads(pathlib.Path(items_path).read_text())
    local, sha, rec = download("allenai/Olmo-3-1025-7B", a.revision)
    corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
    t0 = time.time()
    model, tok = NH.load_model(local, "O7")
    model.requires_grad_(False)
    job = Job(a.out, a, items, model, tok, corpus, "cuda")
    job.m.update({"download": rec, "load_s": round(time.time() - t0, 1), "sha": sha, "device_name": torch.cuda.get_device_name(0)})
    job.write()
    job.run()
    del model
    torch.cuda.empty_cache()
    repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / "models--allenai--Olmo-3-1025-7B"
    shutil.rmtree(repo_dir)
    assert not repo_dir.exists()
    print("UNITE_ROUTE_OK", json.dumps({"t": a.t, "stage": a.stage}), flush=True)


if __name__ == "__main__":
    main()
