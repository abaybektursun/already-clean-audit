"""M2 job (studies/next/07_PICKS/M2_SPEC.md §9): one launch scores every unit of the plan at every checkpoint.

Per checkpoint t, in the plan's order T_15, T_14, ..., T_5 (`m2.plan`):
  1. Stop rule: from T_14 on, if the elapsed time plus the projection of every remaining checkpoint (fixed cost and
     seconds per window measured at T_15) exceeds 0.9 of `--budget-min`, this and every later checkpoint is PARKED
     and the job ends normally. The births those checkpoints serve are then not usable.
  2. Zero-GPU reads: topk/ and stats/ of the layers the plan needs at t, from the banked grid harvest (S3).
  3. Download the branch (the banked harvest's own allow patterns), G-REV (the snapshot commit is the registered
     revision; failing input: a zero commit), G-CFG (config = the registered stage-1 config; failing input: the
     model type renamed), load through `neurons_harvest.load_model` (bf16, sdpa), G0 on the first batch
     (`neurons_harvest.g0_ratios`). Amendment A5: G0's failing input is u scaled by 1.1 (`WRONG_U_SCALE`, the banked
     harvest's second registered input), whose ratio lies within 0.1 +- 1.1 x the gate's own ratio on every layer at
     every point; the wrong activation (the failing input to freeze 1f, without power at early points) is still run
     and REPORTED (metrics.json -> reports -> tNN_G0_act_swap), never gated.
  4. Forward the plan's harvest windows (BOS + 2,048 corpus tokens; batches of exactly 8, the last one padded with
     its own last window) and capture a = act(g) * u in fp32 for all neurons of each needed layer at each needed
     position, through `neurons_harvest.MLPTap`. At T_15 the windows of the first K0_POLES carriers run first and K0,
     the known number, must pass before anything else is computed: their activations against the banked
     exemplar_a at T_15 (failing input: T_14's banked values at the same entries).
  5. G-EX: every grid pole forwarded at t against the banked exemplar_a at t (failing input: the neighbouring point's
     banked values). G-TOPK: every captured position that is in some pole's banked top-50 at t (failing input: the
     neuron axis shifted by one). K0, G-EX, G-TOPK and I-ID use the statistic of amendment A1 (`m2.values_gate`:
     the 99th percentile of |new - bank| / (0.05 + |bank|)); K0 and G-TOPK within `m2.tol_for`, G-EX within
     `m2.tol_ex` (amendment A2: across devices only the entries with |bank| >= 0.1 are read, the rest reported).
     Amendment A3: at a point banked on another device (T_7-T_14) G-EX is REPORTED, not gated (metrics.json ->
     reports: the statistic, the count of read entries above the tolerance and their list); the job continues.
     At a same-device point it stays a gate. G-TOPK stays a gate everywhere.
  6. Records: `m2.scores` and `m2.record` for every (unit, role) at t, appended to out/m2/ranks_freeze.jsonl or
     out/m2/ranks_test.jsonl by the unit's half; the activations to out/m2/vectors/tNN/<unit>_<role>.npz.
     Amendment A4: a pole with no token of its sign in the banked layer files at t (`m2.degenerate`) gets B = 0 and
     ratio = 0 and is not a matched-pole candidate; metrics.json -> points -> tNN lists them (n_degenerate_poles,
     degenerate_poles) for the layers t scores.
  7. At T_13 (`m2.IDENTITY_T`), I-ID: the pre-role windows again in reversed order; the activations must agree with
     the first pass at the same-device tolerance and no carrier may change its top-10 membership under B; failing
     input: the same births' birth-role activations at T_14.
  8. The raw examples of the selected births (births.json `examples`) at their pre role: top-10 lists under A and B
     with each listed pole's top-3 contexts at t, the decoded exemplars, the first matched pole's top-3 contexts.
out/m2/metrics.json is rewritten after every phase. It carries gates, counts and timings, never a hit rate.

    python -m infra.aws_pg.jobs.m2_score --inputs <uri> --corpus <uri> --grid <uri> --budget-min 150 --out <dir>

Fail-loud: no try/except, no fallback, no retry.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import time

import numpy as np

from infra.aws_pg.jobs.neurons_harvest import FETCH, fetch, fetch_corpus
from infra.pg import m2

HUB = None              # snapshot_download; the dry run replaces it with a local stand-in
ALLOW = ["*.safetensors", "*.json", "*.py", "*.model", "*.txt", "*.tiktoken", "*.jinja"]   # matrix_models.HF_ALLOW
K0_POLES = 10
VERSIONS = {"torch": "2.12.0+cu130", "transformers": "5.9.0"}          # the banked grid harvest's (T_15 manifest)
TOKENS_SHA = "2aca80ca6537f46b8f60a550b4a3f2f13be2d18c5402be8f1af5c00caa86f531"
BIRTHS_SHA = "fdca794bc98a78ecf7aa360cad33ca598b633830e2c9891f8dbb7ca73aaf45b7"   # sha256 of studies/next/m2/births.json (written by `m2 births`)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dump(path: pathlib.Path, obj):
    from infra.pg.neurons_harvest import dump as d
    d(path, obj)


class Run:
    def __init__(self, out: pathlib.Path, budget_s: float):
        self.out, self.t0, self.budget_s = out, time.time(), budget_s
        self.rec = {"started_utc": utc(), "complete": False, "gates": {}, "points": {}, "parked": [], "phase": "start"}

    def gate(self, name: str, g: dict):
        self.rec["gates"][name] = g
        self.save(f"gate {name}")
        assert g["pass"], f"{name} failed: {json.dumps(g)[:1500]}"
        assert "failing_input" not in g or not g["failing_input"]["pass"], f"{name}: its failing input passed"

    def report(self, name: str, g: dict):
        """A3: a readout written to metrics.json -> reports; it never stops the job."""
        self.rec.setdefault("reports", {})[name] = g
        self.save(f"report {name}")

    def save(self, phase: str):
        self.rec["phase"], self.rec["updated_utc"] = phase, utc()
        self.rec["elapsed_s"] = round(time.time() - self.t0, 1)
        dump(self.out / "m2" / "metrics.json", self.rec)
        print(f"[m2] {phase}", flush=True)


def check_inputs(inputs: str, reg: dict) -> dict:
    """births.json and the 16 exemplar_a.npz files from the laptop banks, each with its registered sha256."""
    paths = {"births": fetch(f"{inputs.rstrip('/')}/births.json", "inputs/births.json")}
    for t in range(16):
        paths[t] = fetch(f"{inputs.rstrip('/')}/exemplar_a_t{t:02d}.npz", f"inputs/exemplar_a_t{t:02d}.npz")
    got = {k: m2.sha256_file(p) for k, p in paths.items()}
    want = {"births": reg["births_sha"], **{t: reg["exemplar_sha"][t] for t in range(16)}}
    bad = sorted(str(k) for k in want if got[k] != want[k])
    return {"pass": not bad, "mismatch": bad, "paths": {str(k): str(p) for k, p in paths.items()},
            "failing_input": {"pass": got["births"] == "0" * 64}}


def download(branch: str, revision: str) -> tuple[pathlib.Path, dict]:
    hub = HUB
    if hub is None:
        from huggingface_hub import snapshot_download as hub
    t0 = time.time()
    local = pathlib.Path(hub(m2.REPO, revision=branch, allow_patterns=ALLOW))
    dt = time.time() - t0
    n = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
    rec = {"branch": branch, "commit": local.name, "bytes": n, "seconds": round(dt, 1),
           "MB_per_s": round(n / 1e6 / max(dt, 1e-9), 1)}
    rec["G_REV"] = {"pass": local.name == revision, "commit": local.name, "registered": revision,
                    "failing_input": {"pass": local.name == "0" * 40}}
    return local, rec


def layer_files(grid: str, t: int, layers: list, reg: dict) -> dict:
    """topk/ and stats/ of the needed layers at t, from the banked grid harvest (S3 or a local directory)."""
    root = f"{grid.rstrip('/')}/{reg['points'][t][1]}"
    out = {}
    for lyr in layers:
        tk = np.load(fetch(f"{root}/topk/layer_{lyr:02d}.npz", f"grid/t{t:02d}/topk_{lyr:02d}.npz"))
        st = np.load(fetch(f"{root}/stats/layer_{lyr:02d}.npz", f"grid/t{t:02d}/stats_{lyr:02d}.npz"))
        out[lyr] = ({"tok": tk["tok"], "a": tk["a"]},
                    {k: st[k] for k in ("hist", "hist_edges", "q999", "max_abs", "density", "n_pole_tokens")})
        assert (out[lyr][1]["hist"].sum(-1) == out[lyr][1]["n_pole_tokens"]).all(), f"t{t} L{lyr}: hist sums"
    return out


def windows_tensor(tokens: np.ndarray, ws: list, window: int, bos: int, batch: int):
    import torch
    ws = list(ws) + [ws[-1]] * (-len(ws) % batch)
    rows = np.stack([tokens[w * window:(w + 1) * window].astype(np.int64) for w in ws])
    x = torch.from_numpy(rows)
    return torch.cat([torch.full((x.shape[0], 1), bos, dtype=torch.long), x], 1), ws


def forward_capture(model, tap, acts, tokens, need: dict, ws: list, reg: dict, device: str, into: dict) -> float:
    """Run the windows `ws` in batches of reg['batch'] and store a [width] fp32 for every (layer, tok) of `need`."""
    import torch
    from infra.pg.neurons_harvest import forward
    t0 = time.time()
    B = reg["batch"]
    with torch.inference_mode():
        for i in range(0, len(ws), B):
            x, padded = windows_tensor(tokens, ws[i:i + B], reg["window"], reg["bos"], B)
            x = x.to(device)
            want = {}
            for b, w in enumerate(padded[:len(ws[i:i + B])]):
                for lyr, idx, e in need[w]:
                    want.setdefault(lyr, []).append((b, idx, e))

            def fn(lyr, g, u, a_model):
                for b, idx, e in want.get(lyr, []):
                    into[(lyr, e)] = (acts[lyr](g[b, idx].float()) * u[b, idx].float()).cpu().numpy()
            tap.run(fn, lambda: forward(model, x, False))
    return time.time() - t0


def a_matrix(cap: dict, u: dict) -> np.ndarray:
    return np.stack([cap[(u["layer"], e)] for e in u["tok"]])


def ex_entries(us: list, cap: dict) -> list:
    """(grid row, T_15 rank, forwarded a of that pole) for every unit's carrier and exemplar, deduplicated."""
    seen = {}
    for u in us:
        n = u["carrier"][0]
        for e, r in zip(u["tok"], u["ranks"]):
            if (u["layer"], e) in cap:
                seen[(u["row"], r)] = float(cap[(u["layer"], e)][n])
    return [(row, r, v) for (row, r), v in sorted(seen.items())]


def ex_gate(entries: list, bank: np.ndarray, fail_bank: np.ndarray, tol: dict, min_poles: int = 1,
            names: dict | None = None) -> dict:
    rows = np.array([x[0] for x in entries])
    cols = np.array([x[1] for x in entries])
    new = np.array([x[2] for x in entries])
    g = m2.values_gate(new, bank[rows, cols], tol, min_poles, len(set(rows.tolist())))
    g["failing_input"] = m2.values_gate(new, fail_bank[rows, cols], tol, min_poles, len(set(rows.tolist())))
    d = np.where(m2.gate_mask(bank[rows, cols], tol), m2.mixed_diff(new, bank[rows, cols]), -1.0)
    i = int(np.argmax(d))                       # the farthest gated entry
    g["farthest"] = {"row": int(rows[i]), "slot": int(cols[i]), "a_bank": float(bank[rows[i], cols[i]]),
                     "a_new": float(new[i])}
    above = np.flatnonzero(d > tol["p99"])      # A3: every read entry above the tolerance, largest first
    above = above[np.argsort(-d[above], kind="stable")]
    g["n_above_tol"] = int(above.size)
    g["above"] = [{"row": int(rows[j]), "pole": (names or {}).get(int(rows[j])), "slot": int(cols[j]),
                   "a_bank": float(bank[rows[j], cols[j]]), "a_new": float(new[j]), "d": float(d[j])} for j in above]
    return g


def topk_gate(cap: dict, files: dict, tol: dict) -> dict:
    new, bank, fail = [], [], []
    for (lyr, e), vec in cap.items():
        tk = files[lyr][0]
        n_i, s_i, j_i = np.nonzero(tk["tok"] == e)
        for n, s, j in zip(n_i, s_i, j_i):
            new.append(float(vec[n]))
            bank.append(float(tk["a"][n, s, j]))
            fail.append(float(np.roll(vec, 1)[n]))
    if not new:
        return {"pass": True, "n": 0, "note": "no captured position is in any pole's banked top-50 at this point"}
    g = m2.values_gate(np.array(new), np.array(bank), tol)
    g["failing_input"] = m2.values_gate(np.array(fail), np.array(bank), tol)
    return g


def decode_ctx(corpus: dict, dec, t: int) -> str:
    from infra.pg.neurons_harvest import context
    c = context(corpus["tokens"], corpus["doc_offsets"], int(t), dec)
    return c["before"][-160:] + " [[" + c["token"] + "]] " + c["after"]


def example(u: dict, rec: dict, files: dict, corpus: dict, dec) -> dict:
    tk = files[u["layer"]][0]

    def top3(pole):
        _, n, s = m2.parse_pole(pole)
        return [{"rank": j, "a": float(tk["a"][n, s, j]), "text": decode_ctx(corpus, dec, tk["tok"][n, s, j])}
                for j in range(3) if tk["tok"][n, s, j] >= 0]
    return {"unit": u["unit"], "t": rec["t"], "carrier": u["pole"], "rank": rec["rank"], "value": rec["value"],
            "exemplars": [{"rank": r, "tok": e, "text": decode_ctx(corpus, dec, e)} for e, r in zip(u["tok"], u["ranks"])],
            "top10": {k: [{"pole": p, "score": v, "top3": top3(p)} for p, v in rec["top10"][k]] for k in rec["top10"]},
            "matched_first": {"pole": rec["matched"][0], "rank": {k: v[0] for k, v in rec["matched_rank"].items()},
                              "top3": top3(rec["matched"][0])}}


def run(a, reg: dict, out: pathlib.Path):
    import torch
    import transformers
    from infra.pg import m1
    from infra.pg import neurons_corpus as NC
    from infra.pg import neurons_harvest as NH
    R = Run(out, a.budget_min * 60)
    v = {"torch": torch.__version__, "transformers": transformers.__version__}
    R.gate("G_SAME", {"pass": v == reg["versions"], "versions": v, "registered": reg["versions"],
                      "failing_input": {"pass": {**v, "transformers": "0.0.0"} == reg["versions"]}})
    R.gate("INPUTS", check_inputs(a.inputs, reg))
    births = json.loads(pathlib.Path(R.rec["gates"]["INPUTS"]["paths"]["births"]).read_text())
    exa = {t: np.load(R.rec["gates"]["INPUTS"]["paths"][str(t)])["a"] for t in range(16)}
    corpus = NC.load_tok(fetch_corpus(a.corpus, "olmo3"), "olmo3")
    R.gate("G_CORPUS", {"pass": corpus["tokens_sha256"] == reg["tokens_sha256"], "got": corpus["tokens_sha256"],
                        "failing_input": {"pass": "0" * 64 == reg["tokens_sha256"]}})
    tok = transformers.AutoTokenizer.from_pretrained(corpus["tokenizer_dir"])
    dec = NH.decoder(tok)
    pl = m2.plan(births["births"])
    units = {u["unit"]: u for u in pl["units"]}
    examples = set(births["examples"]["freeze"]) | set(births["examples"]["test"])
    (out / "m2").mkdir(parents=True, exist_ok=True)
    (out / "m2" / "samples").mkdir(parents=True, exist_ok=True)
    fh = {h: open(out / "m2" / f"ranks_{h}.jsonl", "w") for h in ("freeze", "test")}
    eh = {h: open(out / "m2" / "samples" / f"raw_examples_{h}.jsonl", "w") for h in ("freeze", "test")}
    order = list(pl["points"])
    assert order[0] == 15, "the plan must start at T_15 (K0)"
    cost = {}
    for t in order:
        p = pl["points"][t]
        elapsed = time.time() - R.t0
        if t != 15:
            rem = sum(cost["fixed_s"] + len(pl["points"][x]["windows"]) * cost["per_window_s"] for x in order[order.index(t):])
            rem += cost["per_window_s"] * len(pl["points"].get(m2.IDENTITY_T, {"windows": []})["windows"]) \
                if m2.IDENTITY_T in order[order.index(t):] else 0.0
            R.rec["projection"] = {"at": t, "elapsed_s": round(elapsed), "remaining_s": round(rem),
                                   "budget_s": R.budget_s}
            if elapsed + rem > 0.9 * R.budget_s:
                R.rec["parked"] = order[order.index(t):]
                R.save(f"parked from T_{t}")
                break
        pt = {"t": t, "branch": reg["points"][t][2], "windows": len(p["windows"]),
              "entries": sum(len(x) for x in p["need"].values()), "records": {"freeze": 0, "test": 0}}
        R.rec["points"][str(t)] = pt
        t_fix = time.time()
        layers = sorted({lyr for w in p["need"].values() for lyr, _, _ in w})
        files = layer_files(a.grid, t, layers, reg)
        deg = [m2.pole_id(lyr, int(i), int(s)) for lyr in layers for i, s in np.argwhere(m2.degenerate(files[lyr][1]))]
        pt["n_degenerate_poles"], pt["degenerate_poles"] = len(deg), deg      # A4: B = 0, ratio = 0, not matched
        local, pt["download"] = download(reg["points"][t][2], reg["points"][t][3])
        R.gate(f"t{t:02d}_G_REV", pt["download"].pop("G_REV"))
        cfg = json.loads((local / "config.json").read_text())
        g = m1.config_gate(cfg, reg["ref_config"])
        g["pass"] = bool(g["pass"] and g["diff_keys"] == [])
        g["failing_input"] = m1.config_gate({**cfg, "model_type": "olmo2-retrofit"}, reg["ref_config"])
        R.gate(f"t{t:02d}_G_CFG", g)
        t_load = time.time()
        model, _ = NH.load_model(str(local), "O7", device=a.device)
        pt["load_s"] = round(time.time() - t_load, 1)
        tower = NH.text_tower(model)
        acts = [layer.mlp.act_fn for layer in tower.layers]
        tap = NH.MLPTap(tower.layers)
        act = NH.act_name(acts[0])
        x0, _ = windows_tensor(corpus["tokens"], p["windows"][:reg["batch"]], reg["window"], reg["bos"], reg["batch"])
        with torch.inference_mode():
            r0 = NH.g0_ratios(model, tap, acts, x0.to(a.device))
            r1 = NH.g0_ratios(model, tap, [NH.WRONG_ACT[act]] * len(acts), x0.to(a.device))
            r2 = NH.g0_ratios(model, tap, acts, x0.to(a.device), u_scale=NH.WRONG_U_SCALE)
        R.report(f"t{t:02d}_G0_act_swap", {"act": {"silu": "gelu_tanh", "gelu_tanh": "silu"}[act], "max_ratio": max(r1),
                                           "n_layers_over_tol": sum(r > NH.G0_TOL for r in r1),
                                           "under_tol": NH.g0_pass(r1)})   # A5: reported, never gated
        R.gate(f"t{t:02d}_G0", {"pass": NH.g0_pass(r0), "max_ratio": max(r0), "failing_input": {
            "u_scale": NH.WRONG_U_SCALE, "pass": NH.g0_pass(r2), "max_ratio": max(r2), "min_ratio": min(r2)}})
        colnorm = {lyr: tower.layers[lyr].mlp.down_proj.weight.detach().float().norm(dim=0).cpu().numpy() for lyr in layers}
        fixed_s = time.time() - t_fix
        cap = {}
        roles = [(units[uid], role) for uid, role in p["roles"]]
        here = [u for u, _ in roles]
        tol = m2.tol_for(t, a.device_name)
        fwd = 0.0
        ws = p["windows"]
        if t == 15:
            k0_units = []
            for u in sorted((u for u in here if u["kind"] == "real"), key=lambda u: u["unit"]):
                if u["pole"] not in {x["pole"] for x in k0_units}:
                    k0_units.append(u)
                if len(k0_units) == K0_POLES:
                    break
            k0_ws = sorted({e // reg["window"] for u in k0_units for e in u["tok"]})
            fwd += forward_capture(model, tap, acts, corpus["tokens"], p["need"], k0_ws, reg, a.device, cap)
            R.gate("K0", ex_gate(ex_entries(k0_units, cap), exa[15], exa[14], tol, K0_POLES))
            ws = [w for w in ws if w not in set(k0_ws)]
        fwd += forward_capture(model, tap, acts, corpus["tokens"], p["need"], ws, reg, a.device, cap)
        pt["forward_s"] = round(fwd, 1)
        nb = t + 1 if t < 15 else t - 1
        gex = ex_gate(ex_entries(here, cap), exa[t], exa[nb], m2.tol_ex(t, a.device_name),
                      names={u["row"]: u["pole"] for u in units.values()})
        if m2.ex_is_gate(t, a.device_name):
            R.gate(f"t{t:02d}_G_EX", gex)
        else:
            R.report(f"t{t:02d}_G_EX", gex)    # A3: reported at a point banked on another device; never stops
        R.gate(f"t{t:02d}_G_TOPK", topk_gate(cap, files, tol))
        vec_dir = out / "m2" / "vectors" / f"t{t:02d}"
        vec_dir.mkdir(parents=True, exist_ok=True)
        pre_recs = {}
        for u, role in roles:
            tk, st = files[u["layer"]]
            am = a_matrix(cap, u)
            sc = m2.scores(am, tk["tok"], st, colnorm[u["layer"]], u["tok"])
            rec = m2.record(u, role, t, sc, st["max_abs"])
            fh[u["half"]].write(json.dumps(rec) + "\n")
            pt["records"][u["half"]] += 1
            np.savez_compressed(vec_dir / f"{u['unit'].replace(':', '_')}_{role}.npz", a=am,
                                tok=np.array(u["tok"], np.int64))
            if role == "pre":
                pre_recs[u["unit"]] = (u, sc)
            if role == "pre" and u["kind"] == "real" and u["unit"] in examples:
                eh[u["half"]].write(json.dumps(example(u, rec, files, corpus, dec), ensure_ascii=False) + "\n")
        for f in list(fh.values()) + list(eh.values()):
            f.flush()
        if t == m2.IDENTITY_T:
            twin = {}
            pre_ws = sorted({e // reg["window"] for u, _ in pre_recs.values() for e in u["tok"]}, reverse=True)
            forward_capture(model, tap, acts, corpus["tokens"], p["need"], pre_ws, reg, a.device, twin)
            first = np.concatenate([a_matrix(cap, u).ravel() for u, _ in pre_recs.values()])
            second = np.concatenate([a_matrix(twin, u).ravel() for u, _ in pre_recs.values()])
            flips = 0
            for u, sc in pre_recs.values():
                tk, st = files[u["layer"]]
                s2 = m2.scores(a_matrix(twin, u), tk["tok"], st, colnorm[u["layer"]], u["tok"])
                n, s = u["carrier"]
                flips += int((m2.ranks(sc["B"])[n, s] <= m2.TOP) != (m2.ranks(s2["B"])[n, s] <= m2.TOP))
            g = m2.values_gate(second, first, m2.TOL["same"])
            g["pass"] = bool(g["pass"] and flips == 0)
            g["carrier_top10_flips_B"] = flips
            g["n_units"] = len(pre_recs)
            nxt = out / "m2" / "vectors" / f"t{t + 1:02d}"
            fail_new, fail_old = [], []
            for uid, (u, _) in pre_recs.items():
                f = nxt / f"{uid.replace(':', '_')}_birth.npz"
                if f.exists():
                    fail_new.append(np.load(f)["a"].ravel())
                    fail_old.append(a_matrix(cap, u).ravel())
            assert fail_new, "I-ID: no birth-role activations at the next point for its failing input"
            g["failing_input"] = m2.values_gate(np.concatenate(fail_new), np.concatenate(fail_old), m2.TOL["same"])
            R.gate("I_ID", g)
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()
        repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / ("models--" + m2.REPO.replace("/", "--"))
        shutil.rmtree(repo_dir)
        fetched = FETCH / "grid" / f"t{t:02d}"
        if fetched.exists():                     # S3 copies only; a local bank is never deleted
            shutil.rmtree(fetched)
        pt["fixed_s"] = round(fixed_s, 1)
        if t == 15:
            n_w = len(p["windows"])
            cost = {"fixed_s": fixed_s + 30.0, "per_window_s": fwd / max(n_w, 1)}
            R.rec["measured"] = {"fixed_s": round(cost["fixed_s"], 1), "per_window_s": round(cost["per_window_s"], 4),
                                 "tokens_per_s": round(-(-n_w // reg["batch"]) * reg["batch"] * (reg["window"] + 1)
                                                       / max(fwd, 1e-9), 1)}
        R.save(f"done T_{t}")
    for f in list(fh.values()) + list(eh.values()):
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
    return {"versions": VERSIONS, "births_sha": BIRTHS_SHA, "exemplar_sha": m2.EXEMPLAR_SHA,
            "tokens_sha256": TOKENS_SHA, "points": {p[0]: list(p) for p in m2.POINTS}, "ref_config": m1.REF_CONFIG,
            "window": m2.WINDOW, "batch": 8, "bos": m2.BOS}


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", required=True, help="S3 prefix or directory: births.json, exemplar_a_tNN.npz")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--grid", required=True, help="the banked grid root (S3 or a directory) holding 1/out, 1b/out")
    ap.add_argument("--budget-min", type=float, required=True)
    ap.add_argument("--registry", default=None, help="dry run only: a planted registry JSON")
    ap.add_argument("--device", default="cuda")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    out = pathlib.Path(a.out)
    (out / "m2").mkdir(parents=True, exist_ok=True)
    if a.registry is None:
        from infra.pg.diff_harvest import hf_online_setup
        hf_online_setup()
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    import torch
    reg = registry(a.registry)
    assert reg["births_sha"], "BIRTHS_SHA is not registered (freeze 1 sets it)"
    if a.device == "cuda":
        assert torch.cuda.is_available(), "no CUDA on this instance"
        a.device_name = torch.cuda.get_device_name(0)
    else:
        a.device_name = reg.get("cpu_device_name", "cpu")
    run(a, reg, out)
    print("M2_JOB_OK", flush=True)


if __name__ == "__main__":
    main()
