"""D1 data explorer: compiles every banked artifact of the already-clean study into the data files that
`infra/pg/neurons_explore_site/index.html` reads.

Reads the explain bank, the harvest bank and the study documents in the sibling worktrees; writes
`neurons_explore_site/data/*.js`. Scores, classes, pass rules and presence readings come from
`neurons_verdict` and contexts from `neurons_explain`, so every number on the page is the pipeline's own.

Sealed until W5 (SPEC_D1 §9.3): the per-pole explanations, texts, scores and classes of the 1,800 audit
poles, and the grid trajectories, clusters and judgments of the development sample S. They are compiled once
the unblind has written `bank/w5/verdict/metrics.json`; before that the page shows their counts and harvest
statistics.

Run on the laptop (no model weights; the tokenizers decode corpus contexts):
    uv run --no-project --with transformers --with tokenizers --with numpy \
        python -m infra.pg.neurons_explore
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from infra.pg import neurons_explain as X
from infra.pg import neurons_verdict as V

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = ROOT.parent
SITE = ROOT / "infra" / "pg" / "neurons_explore_site"

# (id, explain dir under the explain bank, probe job under w2d/q4b_probe, theta file dir, journal entry, what changed)
ROUNDS = (
    ("d1", "w1/q4b_disc", 1, "w1/q4b_disc", "20 Oct 07:50Z", "Prompt v2, every role on DeepSeek-V4.1-Flash."),
    ("d2", "w1/q4b_disc_opus", 2, "w1/q4b_disc_opus", "20 Oct 08:20Z", "Explainer and second reader moved to Opus 5.5."),
    ("d2w", "w1/q4b_disc_opus_w", 3, "w1/q4b_disc_opus_w", "20 Oct 08:38Z", "Writer moved to Opus 5.5 as well."),
    ("d2w-r", "w1/q4b_disc_opus_w", 5, "w1/q4b_disc_opus_w_r5", "20 Oct 16:48Z",
     "The d2w texts rescored once text position 0 stopped counting."),
    ("d3", "w1/q4b_disc_v3", 4, "w1/q4b_disc_v3", "20 Oct 16:48Z", "Prompt v3 with Opus 5.5. Frozen."),
    ("r6", "w1/q4b_disc_sonnet55", 6, "w1/q4b_disc_sonnet55", "20 Oct 18:40Z", "Every Q1 role on Sonnet 5.5."),
    ("r7", "w1/q4b_disc_dsv4pro", 7, "w1/q4b_disc_dsv4pro", "20 Oct 18:40Z", "Every Q1 role on DeepSeek-V4-Pro."),
    ("r9", "w1/q4b_disc_gem37f", 9, "w1/q4b_disc_gem37f", "20 Oct 18:40Z", "Every Q1 role on Gemini 3.7 Flash."),
    ("r13", "w1/q4b_disc_v3_sw", 13, "w1/q4b_disc_v3_sw_own", "21 Oct 17:53Z",
     "The frozen d3 explanations (Opus 5.5) with a Sonnet 5.5 writer: the audit's main configuration."),
)
CANCELLED_ROUNDS = (("r8", "zai-org/GLM-5.3", "20 Oct 18:05Z", "No answer within 15 minutes at concurrency 8; cancelled."),)
FROZEN_ROUND = "d3"
OLD_PROBE_RULE = "text position 0 scored (before the 62d865d3 rule)"

MODELS = (  # key, label, W1 harvest, audit poles, explain cell
    ("Q4B", "Qwen3-4B-Base", "w1/q4b/1/out/q4b", "w1/q4b_split/audit_poles.json", "w2c/q4b", "w2c/q4b_probe/1/out/probe"),
    ("G4", "gemma-3-4b-pt", "w1/g4/1/out/g4", "w1/g4_split/audit_poles.json", "w2c/g4", "w2c/g4_probe/1/out/probe"),
    ("O7", "OLMo-3-1025-7B", "w1/o7_t15/1/out/o7_t15", "w1/o7_split/audit_poles.json", "w2c/o7", "w2c/o7_probe/1/out/probe"),
)
C0_HARVEST = "w1/q4b/1/out/c0"
CORPUS = "w1/q4b/1/out/corpus/corpus"
GRID_DIRS = ("w3/grid/1/out", "w3/grid/1b/out")
N_CONTEXTS = 30
PROMPT_ROLES = ("explainer", "second_reader", "writer", "writer_job", "clusterer", "judge")
PROMPT_VERSIONS = {"9733a5b5": "draft", "17b1b673": "draft", "24dbf384": "v1", "016c11bf": "v2", "e36cf31d": "v3", "c8976b3a": "v1–v2", "ed68dcec": "v3",
                   "6d032b9a": "v1–v2", "3ab19637": "v3", "11b5fd81": "v1"}
CLUSTERER_VERSIONS = {sha: v for v, sha in X.CLUSTERER_PROMPTS.items()}   # the pipeline's own clusterer versions
DOCS = ("SPEC_D1.md", "JOURNAL.md", "DATA_CONTRACT.md", "DIRECTIONS.md", "IDEAS.md", "THEORIES.md", "CHECKS.md",
        "checks/CHECKERS.md", "checks/W1-W3.md", "verify/VERIFY.md", "verify/HARVEST.md", "verify/EXPLAIN.md",
        "verify/REVIEW_EXPLAIN.md", "handoff/S2-S4.md", "handoff/S5.md", "runbook/README.md", "INDEX.md",
        "results/RESULTS.md", "results/REGISTRY.md", "results/README.md", "checks/W4-W5.md", "checks/closeout_W5.md",
        "checks/closeout_W5_recheck.md", "checks/acceptance_final.md", "checks/literature_W5.md", "handoff/S6.md",
        "handoff/judge_incident_log.txt")
W5 = "w5"                                   # the unblind's output directory under the explain bank
CENTER_PROBE = "x/center_probe/1/out"       # the per-checkpoint probe run of the §2.4 center birth curve
FIGURES = ("F1_audit_classes", "F2_strong_rates", "F3_grid_population", "F4_births", "F5_parent_test", "F6_gallery")


def read_json(p):
    return json.loads(Path(p).read_text())


def read_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text().split("\n") if line.strip()]


def r4(x):
    return None if x is None else float(f"{float(x):.4g}")


# --- harvest-side per-pole facts -------------------------------------------------------------

def pole_facts(harvest: Path, pole: str) -> dict:
    st = V.pole_stats(harvest, pole)
    return {"max": r4(st["max_abs"]), "q99": r4(st["q99"]), "q999": r4(st["q999"]), "q9999": r4(st["q9999"]),
            "density": st["density"], "n": st["n_pole_tokens"], "same": r4(st["same_token_frac"]),
            "regime": st["sign_counts"], "dn": r4(st["double_negative_share"]),
            "pos0": r4(V.pole_pos0_share(harvest, pole)), "rare": V.is_rare(st["density"], st["max_abs"])}


def pole_hist(harvest: Path, pole: str) -> dict:
    """The pole's own |a| histogram over the corpus (400 log bins), trimmed to its non-zero span."""
    layer, neuron, s = V.parse_pole(pole)
    h = V.layer_file(str(harvest), "stats", layer)["hist"][neuron, s]
    nz = np.flatnonzero(h)
    if not len(nz):
        return {"lo": 0, "c": []}
    return {"lo": int(nz[0]), "c": h[nz[0]:nz[-1] + 1].astype(int).tolist()}


def pole_detail(harvest: Path, pole: str, vocab: dict) -> dict:
    # vocab.json already holds each pole's own write direction (manifest vocab_rule: sign = -1 for the − pole);
    # neurons_verdict.pole_vocab flips the − pole a second time, so it is not used here
    v = vocab[pole]
    return {**pole_facts(harvest, pole), "hist": pole_hist(harvest, pole),
            "up": [[t, r4(x)] for t, x in v["promoted"]], "down": [[t, r4(x)] for t, x in v["suppressed"]]}


def contexts(harvest: Path, corpus: Path, pole: str) -> list[list]:
    """[a, g, u, doc, pos, before, token, after] for the pole's top 30, as the explainer saw them (256 before)."""
    out = []
    for c in X.pole_contexts(harvest, corpus, pole, N_CONTEXTS, X.EXPLAIN_BEFORE):
        out.append([r4(c["a"]), r4(c["g"]), r4(c["u"]), c["doc"], c["pos"], c["before"], c["token"], c["after"]])
    return out


# --- explain side: one round or one cell -------------------------------------------------------

def old_rule(probe_dir: Path) -> bool:
    with np.load(probe_dir / V.HARVEST_FILES["probe"], allow_pickle=False) as z:
        return "probe_rule" not in z.files


def q1(stats: Path, probe: Path, explain: Path, theta: float) -> list[dict]:
    """neurons_verdict.q1_records; a probe job from before the text-position-0 rule is read under its own
    (old) rule, which the round carries as `probe_rule`."""
    if not old_rule(probe):
        return V.q1_records(stats, probe, explain, theta)
    keep = V.probe_rule
    V.probe_rule = lambda d: {"probe_rule": [OLD_PROBE_RULE], "probe_excluded_positions": None}
    try:
        return V.q1_records(stats, probe, explain, theta)
    finally:
        V.probe_rule = keep


def peaks(probe: Path) -> tuple[dict, dict, dict, np.ndarray, np.ndarray]:
    z = V.load_npz(probe / V.HARVEST_FILES["probe"])
    tix = {str(t): i for i, t in enumerate(z["text_ids"])}
    pix = {p: j for j, p in enumerate(V.harvest_poles(probe))}
    toks = read_json(probe / "poles" / "probe" / "tokens.json")
    return tix, pix, toks, z["argmax_tok"], z["argmin_tok"]


def usage(explain: Path) -> dict:
    """Calls, tokens and latency per role from responses/*.usage.jsonl, with the models that answered."""
    out: dict[str, dict] = {}
    for p in sorted((explain / "responses").glob("*.usage.jsonl")):
        role = p.name.split(".")[0].split("-", 1)[1]
        u = out.setdefault(role, {"calls": 0, "in": 0, "out": 0, "s": 0.0, "models": Counter()})
        for row in read_jsonl(p):
            u["calls"] += 1
            u["in"] += row["prompt_tokens"]
            u["out"] += row["completion_tokens"]
            u["s"] += row["latency_s"]
            u["models"][row["model"]] += 1
    return {k: {**v, "s": round(v["s"]), "models": dict(v["models"])} for k, v in out.items()}


def explain_cell(stats: Path, probe: Path, explain: Path, theta_own: float, theta_frozen: float,
                 poles: set[str] | None = None) -> tuple[dict, dict]:
    """Per pole: the compact record (scores, labels, classes at both θ) and its words (explanations,
    texts as the probe's tokens with the peak token). `poles` keeps a subset of the cell."""
    own = {r["pole"]: r for r in q1(stats, probe, explain, theta_own)}
    frz = {r["pole"]: r for r in q1(stats, probe, explain, theta_frozen)}
    agent = {0: {}, 1: {}}
    for row in read_jsonl(explain / "explanations.jsonl"):
        agent[row["sample"]][row["pole"]] = row
    tix, pix, toks, amax, amin = peaks(probe)
    recs, words = {}, {}
    for pole, r in sorted(own.items()):
        if poles is not None and pole not in poles:
            continue
        sign = V.parse_pole(pole)[2]
        a1, a2 = agent[0][pole], agent[1].get(pole, {})
        rec = {"label": r["label"], "conf": a1.get("confidence"), "related": [j["related_to_others"] for j in r["jobs"]],
               "model": r["explainer_model"], "label2": r.get("label_2"), "model2": r.get("second_reader_model"),
               "declined": r["declined"], "declined2": r.get("second_reader_declined"), "rare": r["rare"],
               "enc": r["encoding_flag"], "c_own": r["class"], "c_frz": frz[pole]["class"],
               "p_own": r.get("passes"), "p_frz": frz[pole].get("passes"), "bmax": r4(r.get("blind_max_score")),
               "unk_own": r["unk_part"], "unk_frz": frz[pole]["unk_part"], "essay": r.get("essay_rule")}
        w = {"e": a1.get("explanation"), "jobs": [[j["name"], j.get("reasoning"), j["related_to_others"]] for j in a1.get("jobs", [])],
             "e2": a2.get("explanation"), "conf2": a2.get("confidence"),
             "jobs2": [[j["name"], j.get("reasoning"), j["related_to_others"]] for j in a2.get("jobs", [])]}
        if "texts" in r:
            rec["s"] = {k: [r4(t["score"]) for t in rows] for k, rows in r["texts"].items()}
            rec["sh"] = [r4(t["score"]) for t in r["shuffled"]]
            rec["sp"] = r["shuffle_pole"]
            w["t"] = {}
            for k, rows in r["texts"].items():
                w["t"][k] = []
                for t in rows:
                    i, j = tix[t["text_id"]], pix[pole]
                    w["t"][k].append([toks[t["text_id"]], int((amax if sign == 0 else amin)[i, j])])
        recs[pole], words[pole] = rec, w
    return recs, words


# --- prompts -------------------------------------------------------------------------------------

def prompt_versions(repo: Path) -> dict:
    """Every committed text of every role's prompt file, keyed by SHA-256, with the first commit that has it."""
    out = {}
    for role in PROMPT_ROLES:
        path = f"infra/pg/neurons_prompts/{role}.md"
        log = subprocess.run(["git", "-C", str(repo), "log", "--reverse", "--format=%h %cI", "--", path],
                             check=True, capture_output=True, text=True).stdout.split("\n")
        for line in filter(None, log):
            commit, when = line.split()
            text = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"], check=True,
                                  capture_output=True, text=True).stdout
            sha = hashlib.sha256(text.encode()).hexdigest()
            key = f"{role}:{sha[:8]}"
            if key not in out:
                version = CLUSTERER_VERSIONS.get(sha, "draft") if role == "clusterer" else PROMPT_VERSIONS.get(sha[:8])
                out[key] = {"role": role, "sha": sha, "version": version, "commit": commit,
                            "when": when, "text": text}
    return out


# --- grid ----------------------------------------------------------------------------------------

def grid_dirs(harvest_bank: Path) -> list[Path]:
    found = {}
    for base in GRID_DIRS:
        for d in sorted((harvest_bank / base).glob("t*_step*")):
            found[int(d.name.split("_step")[1])] = d
    missing = [s for s in V.GRID_STEPS if s not in found]
    if missing:
        raise ValueError(f"grid steps {missing} are not in {harvest_bank}/{{{','.join(GRID_DIRS)}}}")
    dirs = [found[s] for s in V.GRID_STEPS]
    V.check_grid(dirs)
    return dirs


def grid_meta(dirs: list[Path]) -> list[dict]:
    out = []
    for i, d in enumerate(dirs):
        m, met, ctl = (read_json(d / f) for f in ("manifest.json", "metrics.json", "controls.json"))
        out.append({"t": i, "step": V.GRID_STEPS[i], "dir": str(d.relative_to(d.parents[4])), "revision": m["revision"],
                    "device": m.get("device_name"), "tokens": met["n_tokens_scored"], "tps": round(met["tokens_per_second"]),
                    "g0": max(ctl["G0"]["ratio"]), "g0_pass": met["G0_pass"], "dead": met.get("dead_poles"),
                    "started": m.get("started_utc"), "git": m.get("pg_git_sha")})
    return out


def trajectories(dirs: list[Path], poles: list[str]) -> dict:
    """Per pole and checkpoint: the median of a⁺ over the pole's 50 T_15 exemplars, the pole's own q99.9 and
    max|a| there, and the two §2.4 presence readings of that median."""
    order = V.harvest_poles(dirs[-1])
    out = {p: {"med": [], "q999": [], "max": []} for p in poles}
    for d in dirs:
        if V.harvest_poles(d) != order:
            raise ValueError(f"{d}: poles.json differs from T_15's")
        a = V.load_npz(d / V.HARVEST_FILES["exemplar"])["a"]
        for p in poles:
            _, _, s = V.parse_pole(p)
            st = V.pole_stats(d, p)
            out[p]["med"].append(float(np.median(V.signed_magnitude(a[order.index(p)], s))))
            out[p]["q999"].append(st["q999"])
            out[p]["max"].append(st["max_abs"])
    for p, tr in out.items():
        pct, raw = V.presence(tr["med"], tr["q999"], tr["max"])
        tr.update({k: [r4(x) for x in tr[k]] for k in ("med", "q999", "max")})
        tr["pct"], tr["raw"] = pct.astype(int).tolist(), raw.astype(int).tolist()
    return out


# --- models and atlas ----------------------------------------------------------------------------

def atlas(harvest: Path, n_layers: int) -> dict:
    """Per layer, over every pole: histograms of log10 max|a| and log10 density, and the RARE count."""
    max_edges = np.linspace(-3, 4, 71)
    den_edges = np.linspace(0, 8, 49)
    rows = []
    for layer in range(n_layers):
        with np.load(harvest / "stats" / f"layer_{layer:02d}.npz") as z:
            mx, den = z["max_abs"].ravel(), z["density"].ravel()
        if mx.max() >= 10 ** max_edges[-1] or den.max() >= 10 ** den_edges[-1]:
            raise ValueError(f"{harvest} layer {layer}: max|a| {mx.max()} or density {den.max()} beyond the atlas edges")
        rare = (mx == 0) | (den < V.RARE_DENSITY)
        rows.append({"max": np.histogram(np.log10(np.maximum(mx, 1e-3)), max_edges)[0].tolist(),
                     "den": np.histogram(np.log10(np.maximum(den, 1)), den_edges)[0].tolist(),
                     "rare": int(rare.sum()), "n": int(mx.size), "med_max": r4(np.median(mx))})
    return {"max_edges": [r4(x) for x in max_edges], "den_edges": [r4(x) for x in den_edges], "layers": rows}


def model_meta(harvest: Path) -> dict:
    m, met, ctl = (read_json(harvest / f) for f in ("manifest.json", "metrics.json", "controls.json"))
    keys = ("repo", "revision", "branch", "model_type", "n_layers", "width", "hidden", "act_fn_measured", "dtype",
            "transformers", "torch", "tokname", "device_name", "started_utc", "pg_git_sha", "n_tokens_scored")
    g0 = ctl["G0"]
    return {**{k: m.get(k) for k in keys}, "tps": round(met["tokens_per_second"]), "dead": met["dead_poles"],
            "n_poles": met["n_poles"], "g0": [r4(x) for x in g0["ratio"]], "g0_pass": g0["pass"], "g0_tol": g0["tol"],
            "g0_failing": [g0["failing_input"], g0["failing_input_2"]], "g1": ctl.get("G1"),
            "nll": ctl["nll_corpus_first_120k"]["mean_nll"]}


def n_lines(p: Path) -> int:
    return p.read_bytes().count(b"\n") if p.exists() else 0


def w4_dirs(ebank: Path, sealed: tuple[str, ...]) -> list[dict]:
    """W4 explain directories as they stand: manifest batches, answers kept per batch and collected clusters.
    A directory whose name starts with a sealed prefix (every run on S) gives only its manifest and the line
    counts of its response files; its clusters, judgments and readouts are not read."""
    out = []
    for d in sorted((ebank / "w4").glob("*/explain")):
        m = read_json(d / "manifest.json") if (d / "manifest.json").exists() else {}
        clusters, closed = d / "clusters.jsonl", d.parent.name.startswith(sealed)
        batches = m.get("batches", [])
        row = {"dir": f"w4/{d.parent.name}", "sealed": closed, "started": m.get("started_utc"),
               "clusterer_prompt": CLUSTERER_VERSIONS.get(m["prompts"]["clusterer"]) if m else None,
               "batches": [{"batch": b["batch"], "n": b["n"], "roles": b.get("roles"), "models": b.get("models"),
                            **{k: n_lines(d / "responses" / f"{b['batch']}{suffix}")
                               for k, suffix in (("kept", ".jsonl"), ("rejected", ".rejected.jsonl"), ("declined", ".declined.jsonl"))}}
                           for b in batches],
               "n_clusters": len(read_jsonl(clusters)) if clusters.exists() and not closed else None,
               "n_judgments": None if closed else n_lines(d / "judgments.jsonl")}
        gate = d.parent / "verdict" / "judge_gate.json"
        if gate.exists() and not closed:
            g = read_json(gate)
            row["judge_gate"] = {**{k: g[k] for k in ("gate", "a_ctrl", "f_ctrl", "hard_ctrl", "n_poles", "threshold", "created_utc")},
                                 "judge": g["judge"]["models"], "declined_clusterer": g["declined_clusterer"]["n"]}
        if any(b["batch"].startswith("g2-") for b in batches) and not closed:
            row["g2"] = V.g2_readout(d)
        out.append(row)
    return out


def device_checks(harvest_bank: Path, sealed: bool) -> list[dict]:
    """W3 device checks (neurons_device_check): one checkpoint harvested on two GPUs. While S is sealed only the
    aggregates; the list of flipped clusters of S once it is open."""
    out = []
    for p in sorted((harvest_bank / "w3" / "device_check").glob("*.json")):
        r = read_json(p)
        c = r["clusters"]
        out.append({"file": str(p.relative_to(harvest_bank.parent)), "a": r["a"], "b": r["b"], "devices": r["devices"],
                    "n_poles": r["n_poles"], "unit": r["unit"], "checks": r["checks"], "definitions": r["definitions"],
                    "exemplar_a": r["exemplar_a"], "stats": r["stats"],
                    "presence": {k: {x: v[x] for x in ("present_A", "present_B", "n_flipped", "flip_share")} for k, v in r["presence"].items()},
                    "topk": {k: r["topk_top20"][k] for k in ("overlap_median", "share_below_15")},
                    "clusters": None if c is None else {"unit": c["unit"], "n_poles": c["n_poles_with_clusters"], "n": c["n_clusters"],
                                                        **{k: {x: c[k][x] for x in ("n_flipped", "flip_share", *(() if sealed else ("flipped",)))}
                                                           for k in ("percentile", "raw_ratio")}}})
    return out


# --- W5: the unblind -----------------------------------------------------------------------------

def bits(xs) -> str:
    return "".join("1" if x else "0" for x in xs)


def clusters(samples: Path) -> dict:
    """Per pole of S, its clusters as the verdict tabled them (`samples/q2_clusters.jsonl`): the clusterer's name and
    exemplar ranks, the median a⁺ at each checkpoint, both presence readings, births, status, and every judgment
    as [decoy type, decoy pole, t_before, sample, shown order, answer, correct, control]; not the records as the
    judge saw them."""
    out: dict[str, list] = {}
    with open(samples / "q2_clusters.jsonl") as f:
        for line in f:
            r = json.loads(line)
            out.setdefault(r["pole"], []).append({
                "c": r["c"], "name": r["name"], "echo": r["echo"], "ranks": r["ranks"], "med": [r4(x) for x in r["med"]],
                "pct": bits(r["present_pct"]), "raw": bits(r["present_raw"]), "bp": r["birth_pct"], "br": r["birth_raw"],
                "main": r["is_main"], "matched": r["matched"], "status": r["status"], "tb": r["t_before"],
                "r": r4(r["r_at_birth"]), "growth": r4(r["growth_last_half"]), "flag": r.get("flagged"),   # judged clusters only
                "acc": [r4(r.get("acc_random")), r4(r.get("acc_hard"))],
                "j": [[j["decoy_type"], j["decoy_pole"], j["t_before"], j["sample"], j["shown_order"], j["answer"], j["correct"], j["control"]]
                      for j in r.get("judgments", [])]})
    return out


def center_series(center: Path, grid: list[Path], texts_explain: Path, poles: list[str]) -> dict:
    """Per pole of S, the blind maximum's and the five positives' mean score at every grid point, read exactly as
    neurons_verdict.center_birth_curve reads them (each probe value over that checkpoint's own max|a|)."""
    texts = V.texts_by_pole(texts_explain)
    ids = {p: (texts[p]["max"][0]["text_id"], [r["text_id"] for r in texts[p]["pos"]])
           for p in poles if texts.get(p, {}).get("max") and len(texts[p].get("pos", [])) == 5}
    out = {p: {"max": [], "pos": []} for p in ids}
    for g in grid:
        probe = center / g.name
        if read_json(probe / V.HARVEST_FILES["manifest"])["revision"] != read_json(g / V.HARVEST_FILES["manifest"])["revision"]:
            raise ValueError(f"{probe} and {g} are different model revisions")
        tix, pix, max_a, min_a = V.probe_scores(probe)
        for p, (tmax, tpos) in ids.items():
            mab, sign = V.pole_stats(g, p)["max_abs"], V.parse_pole(p)[2]
            sc = (lambda tid: V.score(float(max_a[tix[tid], pix[p]]), float(min_a[tix[tid], pix[p]]), mab, sign)) if mab > 0 else (lambda tid: 0.0)
            out[p]["max"].append(r4(sc(tmax)))
            out[p]["pos"].append(r4(float(np.mean([sc(t) for t in tpos]))))
    return out


def unblind_runs(ebank: Path) -> list[dict]:
    """Every unblind run kept beside the final one, with the readouts that moved between them."""
    out = []
    for d in sorted(ebank.glob(f"{W5}_run*")) + [ebank / W5]:
        p = d / "verdict" / "metrics.json"
        m, q2 = read_json(p), read_json(p)["q2"]
        sens = q2.get("sensitivity_F2_excl_T7_T15")
        out.append({"dir": d.name, "created": m["created_utc"], "sha": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "p_u": [q2["p_u"]["point"], *q2["p_u"]["ci95"]], "n_clusters": q2["p_u"]["n_clusters"],
                    "A": q2["p_u"]["A"], "a_ctrl": q2["p_u"]["a_ctrl"], "b": q2["birth_order"]["b"], "h_mig": q2["H_MIG"]["count"],
                    "hard": [q2["hard"]["accuracy"], q2["hard"]["accuracy_on_flagged"], q2["hard"]["ctrl_accuracy"]],
                    "sens_p_u": sens and sens["p_u"]["point"], "center": "center_birth_curve" in q2})
    return out


def results(ebank: Path) -> dict:
    """The unblind's own output: metrics.json whole, grid.md, the six figures and the parent-test samples."""
    v = ebank / W5 / "verdict"
    return {"metrics": read_json(v / "metrics.json"), "sha": hashlib.sha256((v / "metrics.json").read_bytes()).hexdigest(),
            "grid_md": (v / "grid.md").read_text(), "figures": {f: (ebank / W5 / "figures" / f"{f}.svg").read_text() for f in FIGURES},
            "parent": read_jsonl(v / "samples" / "q2_parent.jsonl"), "runs": unblind_runs(ebank)}


def first_vocab_token(user: str, which: str) -> str:
    line = next(x for x in user.split("\n") if x.startswith(f"Output vocabulary, most {which}: "))
    return json.JSONDecoder().raw_decode(line.split(": ", 1)[1])[0]


def vocab_check(explain: Path, vocab: dict, recs: dict) -> dict:
    """The explainer's view of a pole's output vocabulary against vocab.json, which holds each pole in its own sign:
    for every explainer request, whether its most-promoted line opens with the pole's own top promoted token or with
    its top suppressed one (the mirror pole's promoted), by sign; and, per neuron with both poles audited, the frozen-θ
    pass of each pole."""
    pairs = [p for p in vocab if p.endswith("+") and f"{p[:-1]}-" in vocab]
    mirror = sum([t for t, _ in vocab[p]["promoted"]] == [t for t, _ in vocab[f"{p[:-1]}-"]["suppressed"]] for p in pairs)
    seen = {"+": Counter(), "-": Counter()}
    for row in read_jsonl(explain / "requests" / "c1-explainer.jsonl"):
        pole = row["id"].split(":")[1]
        if pole not in recs:
            continue
        top = first_vocab_token(row["user"], "promoted")
        own, other = vocab[pole]["promoted"][0][0], vocab[pole]["suppressed"][0][0]
        seen[pole[-1]]["own" if top == own else "mirror" if top == other else "neither"] += 1
    paired = Counter()
    for p, r in recs.items():
        q = f"{p[:-1]}-"
        if p.endswith("+") and q in recs:
            paired[f"{'+' if r['p_frz'] else ''}{'-' if recs[q]['p_frz'] else ''}" or "none"] += 1
    by_sign = {s: {"n": sum(p.endswith(s) for p in recs), "pass": sum(bool(r["p_frz"]) for p, r in recs.items() if p.endswith(s)),
                   "one": sum(r["c_frz"] == "ONE" for p, r in recs.items() if p.endswith(s))} for s in "+-"}
    return {"pairs": len(pairs), "mirror": mirror, "prompts": {s: dict(c) for s, c in seen.items()},
            "paired": dict(paired), "by_sign": by_sign}


# --- documents -----------------------------------------------------------------------------------

def documents(study: Path) -> list[dict]:
    out = []
    for name in DOCS:
        p = study / name
        text = p.read_text() if p.suffix == ".md" else f"```\n{p.read_text().rstrip()}\n```\n"
        out.append({"name": name, "text": text, "bytes": len(text.encode()),
                    "modified": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%MZ")})
    return out



def commits(repo: Path, docs: list[dict]) -> dict:
    """Every hex string in the documents that names a commit of this repository: its date and subject."""
    hexes = sorted({h for d in docs for h in re.findall(r"(?<![0-9a-zA-Z_])[0-9a-f]{7,40}(?![0-9a-zA-Z_])", d["text"])
                    if re.search(r"[a-f]", h) and re.search(r"[0-9]", h)})
    check = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch-check"], input="".join(f"{h}^{{commit}}\n" for h in hexes),
                           check=True, capture_output=True, text=True).stdout.split("\n")
    found = {h: line.split()[0] for h, line in zip(hexes, check) if line.split()[1:2] == ["commit"]}
    if not found:
        return {}
    log = subprocess.run(["git", "-C", str(repo), "show", "-s", "--format=%H%x09%cI%x09%s", *sorted(set(found.values()))],
                         check=True, capture_output=True, text=True).stdout
    meta = {full: (when, subject) for full, when, subject in (line.split("\t", 2) for line in log.split("\n") if line)}
    return {h: [*meta[full]] for h, full in found.items()}

# --- runs ----------------------------------------------------------------------------------------

def runs(harvest_bank: Path, spend: Path) -> dict:
    jobs = []
    for status in sorted(harvest_bank.glob("*/*/*/out/pipeline_status.json")):
        out_dir = status.parent
        st = read_json(status)
        for step in st["steps"]:
            d = out_dir / step["name"]
            row = {"job": str(out_dir.relative_to(harvest_bank).parent), "step": step["name"], "state": step["state"],
                   "module": step["module"].rsplit(".", 1)[-1], "args": step["args"]}
            if (d / "manifest.json").exists() and (d / "metrics.json").exists():
                m, met = read_json(d / "manifest.json"), read_json(d / "metrics.json")
                row.update({"model": m.get("model_key"), "branch": m.get("branch"), "device": m.get("device_name"),
                            "tokens": met.get("n_tokens_scored"), "tps": r4(met.get("tokens_per_second")),
                            "started": m.get("started_utc"), "updated": met.get("updated_utc"), "git": m.get("pg_git_sha")})
            jobs.append(row)
    record = read_json(spend)
    rentals = [r for r in record if r["rental"].startswith("neurons")]
    return {"jobs": jobs, "rentals": rentals, "spend_file": str(spend),
            "record": {"n": len(record), "usd": sum(r["realized_usd"] or 0 for r in record)}}


# --- main ----------------------------------------------------------------------------------------

def write_js(path: Path, key: str, value) -> int:
    body = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    path.write_text(f"window.D=window.D||{{}};D[{json.dumps(key)}]={body};\n")
    return path.stat().st_size


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--explain", default=str(PROJECTS / "interp_research-neurons-explain"))
    ap.add_argument("--harvest", default=str(PROJECTS / "interp_research-neurons-harvest"))
    ap.add_argument("--study", default=str(PROJECTS / "interp_research-clean-neurons" / "studies" / "already-clean"))
    a = ap.parse_args(argv)
    ebank, hbank = Path(a.explain) / "bank", Path(a.harvest) / "bank" / "neurons"
    corpus = hbank / CORPUS
    out = SITE / "data"
    out.mkdir(parents=True, exist_ok=True)
    sealed = not (ebank / W5 / "verdict" / "metrics.json").exists()

    harvest = {k: hbank / h for k, _, h, *_ in MODELS}
    theta_frozen = read_json(ebank / "w1/q4b_disc_v3/verdict/theta.json")
    core = {"built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "sealed": sealed,
            "theta_frozen": theta_frozen["theta_s"], "frozen_round": FROZEN_ROUND, "grid_steps": V.GRID_STEPS,
            "rules": {"pass": [3, 2, 1], "essay_strong": V.ESSAY_STRONG, "rare_density": V.RARE_DENSITY,
                      "ctrl_gate": V.CTRL_GATE, "pu": [V.PU_RG_UPPER, V.PU_SP_LOWER], "raw_presence": V.RAW_PRESENCE}}
    files = {}   # lazily loaded data files, by name

    # discovery rounds
    q4b = harvest["Q4B"]
    rounds, words = [], {}
    for rid, edir, probe_n, tdir, journal, note in ROUNDS:
        explain = ebank / edir / "explain"
        probe = hbank / f"w2d/q4b_probe/{probe_n}/out/probe"
        theta = read_json(ebank / tdir / "verdict" / "theta.json")
        recs, w = explain_cell(q4b, probe, explain, theta["theta_s"], theta_frozen["theta_s"])
        man = read_json(explain / "manifest.json")
        rounds.append({"id": rid, "note": note, "journal": journal, "explain": edir, "probe": f"w2d/q4b_probe/{probe_n}",
                       "probe_rule": OLD_PROBE_RULE if old_rule(probe) else "text position 0 excluded",
                       "theta": theta["theta_s"], "theta_n": theta.get("n_scores"), "prompt": man.get("prompt_version"),
                       "prompts": man["prompts"], "started": man["started_utc"], "usage": usage(explain),
                       "forked": man.get("forked_from"),
                       "device": read_json(probe / "manifest.json").get("device_name"), "poles": recs})
        words[rid] = w
        print(f"round {rid}: {len(recs)} poles, θ_s {theta['theta_s']:.4f}")
    core["rounds"] = rounds
    core["cancelled_rounds"] = [dict(zip(("id", "model", "journal", "note"), r)) for r in CANCELLED_ROUNDS]
    files["words"] = words

    # poles: discovery set, their shuffle partners, S_gate
    vocab = {k: read_json(h / V.HARVEST_FILES["vocab"]) for k, h in harvest.items()}
    disc = read_json(ebank / "w1/q4b_split/discovery_poles.json")
    partners = sorted({r["sp"] for rd in rounds for r in rd["poles"].values() if "sp" in r} - set(disc))
    s_gate = read_json(ebank / "w3/o7/draw/S_gate.json")
    s_main = read_json(ebank / "w3/o7/draw/S.json")
    poles = {p: {"model": "Q4B", "set": "discovery", **pole_detail(q4b, p, vocab["Q4B"])} for p in disc}
    poles.update({p: {"model": "Q4B", "set": "shuffle partner", **pole_facts(q4b, p)} for p in partners})
    poles.update({f"O7:{p}": {"model": "O7", "set": "S_gate", **pole_detail(harvest["O7"], p, vocab["O7"])} for p in s_gate})
    core["poles"] = poles
    core["hist_edges"] = {k: [r4(x) for x in V.layer_file(str(h), "stats", 0)["hist_edges"]] for k, h in harvest.items()}
    ctx = {p: contexts(q4b, corpus, p) for p in disc}
    ctx.update({f"O7:{p}": contexts(harvest["O7"], corpus, p) for p in s_gate})
    files["contexts"] = ctx
    print(f"contexts: {len(ctx)} poles")

    # audit cells (W2-C) and S_gate's Q1 records; once unblinded, every audit pole's words, details and contexts
    cells, audit_detail = [], {}
    for key, label, h, audit_file, cell, probe in MODELS:
        explain, probe_dir = ebank / cell / "explain", hbank / probe
        audit = read_json(ebank / audit_file)
        answered = Counter(m for rid, m in V.answering_models(explain).items() if rid.startswith("explainer:"))
        c = {"key": key, "label": label, "explain": cell, "probe": probe, "n_audit": len(audit),
             "explainers": dict(answered), "n_texts": len(read_jsonl(explain / "texts.jsonl")),
             "n_probe_poles": len(V.harvest_poles(probe_dir)), "device": read_json(probe_dir / "manifest.json").get("device_name"),
             "started": read_json(explain / "manifest.json")["started_utc"], "prompts": read_json(explain / "manifest.json")["prompts"],
             "n_tokens": read_json(harvest[key] / "manifest.json")["n_tokens_scored"],
             "audit": {p: pole_facts(harvest[key], p) for p in audit}}
        keep = (set(s_gate) if key == "O7" else set()) | (set() if sealed else set(audit))
        if keep:
            recs, w = explain_cell(harvest[key], probe_dir, explain, theta_frozen["theta_s"], theta_frozen["theta_s"], keep)
            c["poles"] = recs
            files[f"words_{key}"] = w
        if not sealed:
            audit_detail.update({f"{key}:{p}": {"model": key, "set": "audit", **pole_detail(harvest[key], p, vocab[key])} for p in audit})
            by_layer: dict[int, dict] = {}
            for p in audit:
                by_layer.setdefault(V.parse_pole(p)[0], {})[f"{key}:{p}"] = contexts(harvest[key], corpus, p)
            for layer, cx in by_layer.items():
                files[f"ctx_{key}_L{layer:02d}"] = cx
            c["vocab_check"] = vocab_check(explain, vocab[key], {p: r for p, r in c["poles"].items() if p in set(audit)})
        cells.append(c)
        print(f"cell {key}: {len(audit)} audit poles, {len(c.get('poles', {}))} compiled")
    core["cells"] = cells
    if audit_detail:
        files["audit"] = audit_detail

    # development grid; once unblinded, S's trajectories, clusters, judgments and center-probe scores
    dirs = grid_dirs(hbank)
    core["grid"] = {"checkpoints": grid_meta(dirs), "s_gate": s_gate, "s": s_main, "n_s": len(s_main),
                    "traj": trajectories(dirs, s_gate + ([] if sealed else s_main))}
    core["w4"] = w4_dirs(ebank, ("o7_s",) if sealed else ())
    core["device_checks"] = device_checks(hbank, sealed)
    if not sealed:
        core["results"] = results(ebank)
        core["grid"]["center"] = center_series(hbank / CENTER_PROBE, dirs, ebank / "w2c/o7/explain", s_main)
        files["clusters"] = clusters(ebank / W5 / "verdict" / "samples")
    print(f"grid: {len(dirs)} checkpoints")

    # models, prompts, runs, documents
    core["models"] = [{"key": k, "label": lab, "harvest": h, **model_meta(harvest[k]),
                       "atlas": atlas(harvest[k], read_json(harvest[k] / "manifest.json")["n_layers"])}
                      for k, lab, h, *_ in MODELS]
    core["c0"] = model_meta(hbank / C0_HARVEST)
    core["prompts"] = prompt_versions(Path(a.explain))
    core["runs"] = runs(hbank, Path(a.harvest) / "studies/lineage/cross-lineage/research/mendeleev_spend.json")
    docs = documents(Path(a.study))
    core["commits"] = commits(Path(a.study), docs)
    files["docs"] = docs
    core["files"] = sorted(files)

    sizes = {name: write_js(out / f"{name}.js", name, value) for name, value in (("core", core), *files.items())}
    print(json.dumps({k: f"{v / 1e6:.2f} MB" for k, v in sizes.items() if v > 1e6}), f"{len(sizes)} files, {sum(sizes.values()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
