"""NEURONSHOW job (studies/neuronshow/SPEC.md): NEURONRW's residual push on six question prompts.

    python -m infra.aws_pg.jobs.neuronshow --out DIR --model-key Q4B|G4|O7 [--weights DIR] [--corpus PATH]

Weights: --weights is a local snapshot directory (holding config.json; its name must be the plan's commit sha) or an HF
hub cache root holding exactly one snapshot of the repo; without --weights the plan's repo is downloaded from the Hub
at the plan's revision (O7: branch stage1-step1413814), and the commit sha is asserted either way. HF_TOKEN in the
environment is used as is (G4 is gated); without it the token is read from SSM (infra/pg/diff_harvest.hf_online_setup).
--corpus (Q4B only, for the known number) is the NEURONRW corpus directory, an s3:// URI (copied with the aws CLI) or a
local directory with manifest.json and tok/qwen3/.

Order (every stage appends its rows to DIR/samples.jsonl as each generate call returns and rewrites DIR/metrics.json):
  0. Q4B: the known number (neuron_rw.Job.g_known on the Arm 0 known windows, A26): own clean NLL 1.3216137 and own
     dNLL 0.3082006 within 0.002 nats, its layers-0..n/4 failing input shifting >= 0.5 nats; else status aborted_gate.
  1. Directions, projections (top 20 over the full vocabulary; G-proj: top-10 overlap >= 9 with the banked list, the
     neighbouring column i + 1 must fail), matched controls (neuronshow_plan.draw_control), the realized-change
     failing input (hook keyed per forward: its decode-step-1 cosine must be < 0.95). Any check landing wrong aborts.
  2. gate: 20 unsteered samples per prompt; the frozen gate rule keeps or drops each prompt (DIR/gate.json).
  3. dose: every target pole at every grid dose, 20 samples on P1; the frozen dose rule picks the dose (DIR/dose.json).
  4. main: per kept prompt, every target pole and its control at the chosen dose, 20 samples each; the dose-story pole
     (G4 L28N5924+) also at every other grid dose.
Every steered cell (stage, condition, pole, dose, prompt) gets the dose-0 identity check (logits with the hook at alpha
0 equal the unhooked logits exactly; failing input: the cell's dose must differ) before its rows are generated, and its
decode-step-1 realized cosine (median over its rows >= 0.95) after; both in DIR/cells.json. Seeds: torch.manual_seed(
base_seed + 1000 * prompt_index + c) before the c-th generate call of prompt_index, c counted over the whole job.
At exit DIR/STATUS holds exit=0 (complete) or exit=1 (any exception, which is re-raised).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"      # before torch is imported (neuronshow_plan imports it)
from infra.aws_pg.jobs.neuron_off import append_jsonl, dump  # noqa: E402
from infra.pg import neuronshow_plan as P  # noqa: E402

SCHEMA = "neuronshow/metrics/v1"
REPO = pathlib.Path(__file__).resolve().parents[3]


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Show:
    def __init__(self, model, tok, model_key: str, plan: dict, out, meta: dict, known=None):
        import torch
        from infra.pg import neuron_rw_core as R
        self.torch, self.R = torch, R
        self.model, self.tok, self.M, self.plan = model, tok, model_key, plan
        self.spec, self.rules, self.S = plan["models"][model_key], plan["rules"], dict(plan["sampling"])
        self.poles = P.model_poles(plan, model_key)
        self.out = pathlib.Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.samples = self.out / "samples.jsonl"
        assert not self.samples.exists(), f"{self.samples} exists; use a fresh output directory"
        self.known = known
        self.calls = {i: 0 for i in range(len(plan["prompts"]))}
        self.cells, self.base_logits = [], {}
        assert meta.get("plan_sha256"), "meta must carry the plan's sha256"
        self.m = {"schema": SCHEMA, "status": "running", "started_utc": utc(), "updated_utc": utc(), **meta,
                  "stage": "load", "gates": {}, "gate": None, "dose": None,
                  "progress": {"calls_done": 0, "rows_written": 0},
                  "throughput": {"gen_s": 0.0, "gen_rows": 0, "gen_new_tokens": 0}}
        self.save()

    # ------------------------------------------------------------------ plumbing
    def save(self):
        self.m["updated_utc"] = utc()
        dump(self.out / "metrics.json", self.m)

    def abort(self, why: str):
        self.m["status"], self.m["abort_reason"] = "aborted_gate", why
        self.save()
        raise AssertionError(f"gate failed: {why}")

    def rms(self, layer: int) -> float:
        return self.spec["rms"][layer]

    def pid(self, i: int) -> str:
        return self.plan["prompts"][i]["id"]

    # ------------------------------------------------------------------ stage 1: directions and checks
    def setup(self):
        torch, R = self.torch, self.R
        tower = R.text_tower(self.model)
        assert len(tower.layers) == self.spec["n_layers"] == len(self.spec["rms"]), "model, plan and rms layer counts differ"
        study = {}
        for p in self.poles:
            study.setdefault(p["layer"], set()).add(p["neuron"])
        self.w, self.ctrl = {}, {}
        draws = {}
        for p in self.poles:
            l, n, s = p["layer"], p["neuron"], p["sign"]
            self.w[p["pole"]] = (l, R.pole_direction(self.model, l, n, s))
            W = tower.layers[l].mlp.down_proj.weight
            norms = W.detach().float().norm(dim=0).cpu().numpy()
            d = P.draw_control(norms, l, n, s, study[l], self.spec["model_index"], self.plan)
            assert abs(d["ratio"] - 1.0) <= self.rules["control_norm_tol"], d
            if p["w_down_norm_banked"] is not None:
                d["target_norm_banked"] = p["w_down_norm_banked"]
                d["target_norm_rel_err"] = abs(d["target_norm"] / p["w_down_norm_banked"] - 1.0)
            self.ctrl[p["pole"]] = d["control"]
            self.w[d["control"]] = (l, R.pole_direction(self.model, l, d["neuron"], s))
            draws[p["pole"]] = d
        assert len(set(self.ctrl.values())) == len(self.ctrl), "two targets drew the same control"
        dump(self.out / "controls.json", draws)
        # projections: top 20 over the full vocabulary for every target, control and neighbour column
        allv = torch.arange(R.unembedding(self.model).shape[0])
        targets = [(p["pole"], p["layer"], p["neuron"], p["sign"]) for p in self.poles]
        ctrls = [(draws[p["pole"]]["control"], p["layer"], draws[p["pole"]]["neuron"], p["sign"]) for p in self.poles]
        width = {p["layer"]: tower.layers[p["layer"]].mlp.down_proj.weight.shape[1] for p in self.poles}
        neigh = [(f"neighbour:{p['pole']}", p["layer"], (p["neuron"] + 1) % width[p["layer"]], p["sign"]) for p in self.poles]
        proj = R.vocab_projection(self.model, self.tok, targets + ctrls + neigh, allv, k=20)
        rows, gp = [], {}
        for pid_, l, n, s in targets + ctrls + neigh:
            rows.append({"pole": pid_, "layer": l, "neuron": n, "sign": s, **proj[pid_]})
        append_jsonl(self.out / "projections.jsonl", rows)
        for p in self.poles:
            if p["top_full10_banked"] is None:
                continue
            mine = {t["id"] for t in proj[p["pole"]]["top_full"][:10]}
            nb = {t["id"] for t in proj[f"neighbour:{p['pole']}"]["top_full"][:10]}
            gp[p["pole"]] = {"overlap": len(mine & set(p["top_full10_banked"])),
                             "neighbour_overlap": len(nb & set(p["top_full10_banked"]))}
        self.top_ids = {p["pole"]: [t["id"] for t in (p["top20"] if p["top20"] is not None else proj[p["pole"]]["top_full"])]
                        for p in self.poles}
        g_proj = {"per_pole": gp, "pass": all(v["overlap"] >= 9 for v in gp.values()),
                  "failing_input_pass": any(v["neighbour_overlap"] >= 9 for v in gp.values()), "rule":
                  "top-10 (full vocabulary) shares >= 9 ids with the banked top_full; failing input: column i + 1"}
        # realized-change failing input: hook keyed per forward never steers a decode step
        p0 = self.poles[0]
        l0, w0 = self.w[p0["pole"]]
        top = max(self.spec["doses"])
        res = R.generate_rows(self.model, self.tok, self.plan["prompts"][0]["text"], [(l0, w0, top * self.rms(l0))] * 4,
                              self.S["base_seed"], {**self.S, "max_new_tokens": 2}, keyed="per_forward")
        cos1 = [c for r in res for (st, c, _) in r["realized_steps"] if st == 1]
        assert len(cos1) == 4, f"step 1 read on {len(cos1)} of 4 rows"
        gen_fail = {"pole": p0["pole"], "dose": top, "hook": "per-forward index", "cos_step1": cos1,
                    "fired": max(cos1) < self.rules["cos_min"]}
        self.m["gates"].update({"g_proj": g_proj, "realized_failing_input": gen_fail, "controls": draws})
        self.save()
        if not g_proj["pass"] or g_proj["failing_input_pass"]:
            self.abort(f"G-proj {g_proj}")
        if not gen_fail["fired"]:
            self.abort(f"realized-change failing input did not fire: {gen_fail}")

    # ------------------------------------------------------------------ cells
    def identity(self, cell: dict, i: int, layer: int, w, dose: float) -> dict:
        R = self.R
        ids = self.torch.tensor([R.prompt_ids(self.tok, self.plan["prompts"][i]["text"])])
        start = ids.shape[1] - 1
        if i not in self.base_logits:
            self.base_logits[i] = R.logits_with(self.model, ids).float()
        base = self.base_logits[i]
        zero = R.logits_with(self.model, ids, R.steer_one(self.model, layer, w, 0.0, start)).float()
        fail = R.logits_with(self.model, ids, R.steer_one(self.model, layer, w, dose * self.rms(layer), start)).float()
        cell["identity_max_abs_diff"] = float((zero - base).abs().max())
        cell["identity_pass"] = cell["identity_max_abs_diff"] == 0.0
        cell["identity_failing_input_diff"] = float((fail - base).abs().max())
        cell["identity_failing_input_fired"] = cell["identity_failing_input_diff"] > 0.0
        return cell

    def generate(self, stage: str, i: int, rows: list):
        """rows: (condition, pole or None, target pole or None, dose, k) for prompt i, in plan order; identity per new
        cell before generation, realized cosine per cell after."""
        import numpy as np
        R, prompt = self.R, self.plan["prompts"][i]["text"]
        cells = {}
        for cond, pole, target, dose, _ in rows:
            if pole is None or (cond, pole, dose) in cells:
                continue
            layer, w = self.w[pole]
            cells[(cond, pole, dose)] = self.identity({"stage": stage, "condition": cond, "pole": pole, "target": target,
                                                       "dose": dose, "prompt_id": self.pid(i), "cos_step1": []}, i, layer, w, dose)
        bad = [c for c in cells.values() if not (c["identity_pass"] and c["identity_failing_input_fired"])]
        if bad:
            self.cells += list(cells.values())
            dump(self.out / "cells.json", self.cells)
            self.abort(f"identity check: {bad[0]}")
        n = self.S["max_rows"]
        for s in range(0, len(rows), n):
            chunk = rows[s:s + n]
            spec = [(None, None, 0.0) if pole is None else (self.w[pole][0], self.w[pole][1], dose * self.rms(self.w[pole][0]))
                    for _, pole, _, dose, _ in chunk]
            c = self.calls[i]
            seed = self.S["base_seed"] + 1000 * i + c
            t0 = time.time()
            res = R.generate_rows(self.model, self.tok, prompt, spec, seed, self.S)
            if self.model.device.type == "cuda":
                self.torch.cuda.synchronize()
            dt = time.time() - t0
            recs = []
            for (cond, pole, target, dose, k), (_, _, a_abs), r in zip(chunk, spec, res):
                s1 = [(cs, ra) for st, cs, ra in r["realized_steps"] if st == 1]
                assert len(s1) <= 1
                if pole is not None:
                    assert s1, f"{pole}: no decode-step-1 reading"
                    cells[(cond, pole, dose)]["cos_step1"].append(s1[0][0])
                text = r["completion"]
                recs.append({"model_key": self.M, "stage": stage, "condition": cond, "pole": pole, "target": target,
                             "prompt_id": self.pid(i), "prompt_index": i, "prompt": prompt, "dose": dose,
                             "alpha_abs": a_abs, "layer": None if pole is None else self.w[pole][0], "k": k, "seed": seed,
                             "call": c, "completion": text, "completion_token_ids": r["completion_token_ids"],
                             "n_new_tokens": r["n_new_tokens"], "stopped_eos": r["stopped_eos"],
                             "realized_cos": r["realized_cos"], "cos_step1": s1[0][0] if s1 else None,
                             "ratio_step1": s1[0][1] if s1 else None,
                             "bigram_ratio": P.distinct_bigram_ratio(text), "nonrep": P.nonrep(text, self.rules)})
            append_jsonl(self.samples, recs)
            self.calls[i] += 1
            self.m["progress"]["calls_done"] += 1
            self.m["progress"]["rows_written"] += len(recs)
            tp = self.m["throughput"]
            tp["gen_s"] = round(tp["gen_s"] + dt, 2)
            tp["gen_rows"] += len(recs)
            tp["gen_new_tokens"] += sum(r["n_new_tokens"] for r in res)
            self.save()
        for c in cells.values():
            v = c.pop("cos_step1")
            c.update({"n_rows": len(v), "cos_step1_median": float(np.median(v)), "cos_step1_min": float(min(v)),
                      "cos_pass": float(np.median(v)) >= self.rules["cos_min"]})
        self.cells += list(cells.values())
        dump(self.out / "cells.json", self.cells)
        return recs

    def read(self, stage: str) -> list:
        return [r for r in map(json.loads, self.samples.read_text().splitlines()) if r["stage"] == stage]

    # ------------------------------------------------------------------ stages
    def run(self) -> dict:
        n = self.rules["n_samples"]
        if self.known is not None:
            self.m["stage"] = "known"
            self.save()
            kn = self.known()
            self.m["gates"]["known"] = kn
            self.save()
            if not (kn["pass"] and not kn["failing_input"]["pass"]):
                self.abort(f"known number: pass {kn['pass']}, failing input pass {kn['failing_input']['pass']}")
        self.m["stage"] = "setup"
        self.setup()
        # gate
        self.m["stage"] = "gate"
        for i in range(len(self.plan["prompts"])):
            self.generate("gate", i, [("unsteered", None, None, 0, k) for k in range(n)])
        rows = self.read("gate")
        gate = {self.pid(i): P.gate([r for r in rows if r["prompt_index"] == i], self.plan["prompts"][i]["text"], self.rules)
                for i in range(len(self.plan["prompts"]))}
        self.kept = [i for i in range(len(self.plan["prompts"])) if gate[self.pid(i)]["kept"]]
        self.m["gate"] = {"per_prompt": gate, "kept": [self.pid(i) for i in self.kept]}
        dump(self.out / "gate.json", self.m["gate"])
        self.save()
        # dose
        self.m["stage"] = "dose"
        i1 = [p["id"] for p in self.plan["prompts"]].index(self.rules["dose_prompt"])
        grid = self.spec["doses"]
        self.generate("dose", i1, [("target", p["pole"], p["pole"], d, k) for p in self.poles for d in grid for k in range(n)])
        drows = self.read("dose")
        counts = {d: {p["pole"]: sum(r["nonrep"] for r in drows if r["pole"] == p["pole"] and r["dose"] == d)
                      for p in self.poles} for d in grid}
        dose = P.choose_dose(counts, grid, self.rules)
        self.m["dose"] = dose
        dump(self.out / "dose.json", dose)
        self.save()
        # main
        self.m["stage"] = "main"
        d0, story = dose["chosen"], self.spec.get("dose_story")
        for i in self.kept:
            rows = []
            for p in self.poles:
                rows += [("target", p["pole"], p["pole"], d0, k) for k in range(n)]
                rows += [("control", self.ctrl[p["pole"]], p["pole"], d0, k) for k in range(n)]
            if story is not None:
                rows += [("target", story, story, d, k) for d in grid if d != d0 for k in range(n)]
            self.generate("main", i, rows)
        self.m["cells"] = {"n": len(self.cells), "identity_pass": sum(c["identity_pass"] for c in self.cells),
                           "cos_pass": sum(c["cos_pass"] for c in self.cells),
                           "cos_step1_min": min(c["cos_step1_min"] for c in self.cells)}
        self.m["status"], self.m["stage"] = "complete", "done"
        self.save()
        return self.m


def known_reader(model, tok, plan: dict, corpus_dir: str, out: pathlib.Path):
    """A callable running neuron_rw.Job.g_known on the Arm 0 known block (Q4B), as NEURONRW's Arm 1 and Arm 2 did."""
    from infra.pg import neurons_corpus as NC
    from infra.pg.matrix_models import vocab_sha256
    k = plan["known"]
    raw2, raw0 = (REPO / k["arm2_plan"]).read_bytes(), (REPO / k["arm0_plan"]).read_bytes()
    assert hashlib.sha256(raw2).hexdigest() == k["arm2_plan_sha256"] and hashlib.sha256(raw0).hexdigest() == k["arm0_plan_sha256"]
    plan2, plan0 = json.loads(raw2), json.loads(raw0)
    corpus = NC.load_tok(corpus_dir, plan2["tokname"])
    assert corpus["tokens_sha256"] == plan2["corpus_tokens_sha256"], "corpus tokens differ from the plan's"
    assert vocab_sha256(tok) == corpus["record"]["vocab_sha256"], "model and corpus tokenizer vocabularies differ"
    assert plan0["known"]["own_clean_nll_mean_banked"] == k["own_clean_nll_mean"]
    assert plan0["known"]["own_dnll_mean_banked"] == k["own_dnll_mean"]
    return known_from(model, tok, plan2, k["arm2_plan_sha256"], plan0["known"], corpus["tokens"], out)


def known_from(model, tok, plan2: dict, plan2_sha256: str, known0: dict, tokens, out: pathlib.Path):
    """The G-known reading of NEURONRW's Arm 2 job (A26: the Arm 0 known windows) as a callable."""
    from infra.aws_pg.jobs import neuron_rw as J

    def run():
        job = J.Job(out / "known", stage="arm2", model=model, tok=tok, tokens=tokens, plan=plan2,
                    plan_sha256=plan2_sha256, meta={}, rms_banked={"model_key": plan2["model_key"]}, known_arm0=known0)
        return job.g_known()
    return run


def resolve_weights(spec: dict, weights: str | None) -> tuple:
    """(local snapshot dir, commit sha, record)."""
    from infra.aws_pg.jobs.neurons_harvest import download
    if weights is None:
        from infra.pg.diff_harvest import hf_online_setup
        hf_online_setup()
        local, sha, rec = download(spec["repo"], spec["branch"] or spec["revision_expected"])
        return local, sha, {"source": "hub", **rec}
    d = pathlib.Path(weights)
    if not (d / "config.json").exists():
        snaps = sorted((d / ("models--" + spec["repo"].replace("/", "--")) / "snapshots").iterdir())
        assert len(snaps) == 1, f"{weights}: expected one snapshot of {spec['repo']}, found {[s.name for s in snaps]}"
        d = snaps[0]
    return str(d), d.name, {"source": "local", "path": str(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-key", required=True, choices=["Q4B", "G4", "O7"])
    ap.add_argument("--weights", default=None, help="local snapshot dir or HF hub cache root; absent = Hub download")
    ap.add_argument("--corpus", default=None, help="Q4B: NEURONRW corpus dir (s3:// or local); default the plan's")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        _main(a, out)
    except BaseException:
        (out / "STATUS").write_text("exit=1\n")
        raise
    (out / "STATUS").write_text("exit=0\n")


def _main(a, out: pathlib.Path):
    import torch
    import transformers
    from infra.aws_pg.jobs.neurons_harvest import fetch_corpus
    from infra.pg import neurons_harvest as NH
    assert torch.cuda.is_available(), "no CUDA on this machine"
    torch.use_deterministic_algorithms(True)
    plan = P.load()
    spec = plan["models"][a.model_key]
    local, sha, wrec = resolve_weights(spec, a.weights)
    assert sha == spec["revision_expected"], f"{spec['repo']}: snapshot {sha}, plan expects {spec['revision_expected']}"
    meta = {"model_key": a.model_key, "repo": spec["repo"], "branch": spec["branch"], "revision": sha, "weights": wrec,
            "device_name": torch.cuda.get_device_name(0), "transformers": transformers.__version__,
            "torch": torch.__version__, "git_sha": os.environ.get("PG_GIT_SHA"),
            "plan_sha256": hashlib.sha256(P.PLAN_PATH.read_bytes()).hexdigest()}
    t0 = time.time()
    model, tok = NH.load_model(local, a.model_key)
    meta["load_s"] = round(time.time() - t0, 1)
    known = None
    if a.model_key == plan["known"]["model_key"]:
        src = a.corpus or plan["known"]["corpus"]
        meta["corpus"] = src
        known = known_reader(model, tok, plan, str(fetch_corpus(src, "qwen3")), out)
    m = Show(model, tok, a.model_key, plan, out, meta, known).run()
    print("NEURONSHOW_OK", json.dumps({"model_key": a.model_key, "status": m["status"], "progress": m["progress"],
                                       "dose": m["dose"]["chosen"], "kept": m["gate"]["kept"]}), flush=True)


if __name__ == "__main__":
    main()
