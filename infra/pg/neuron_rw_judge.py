"""NEURONRW judge and RE-EXPLAIN records (studies/neuronrw/SCHEMA.md v5 §1.6-§1.7, §4; SPEC §5.0, §5.1, §5.7, §5.8,
§9.5). Laptop only: builds request files for the parent's client (`infra/pg/neurons_llm.py`, unchanged from
`neurons/explain`), runs RE-EXPLAIN through it when asked, and collects answers through `neurons_llm.collect`.

    python -m infra.pg.neuron_rw_judge reexplain --model M --dry-run            # rebuild, verify, hash; no API call
    python -m infra.pg.neuron_rw_judge reexplain --model M [--out bank/rw/reexplain] [--concurrency 8]   # + run
    python -m infra.pg.neuron_rw_judge build --step RX --model M [--out bank/rw/reexplain]
    python -m infra.pg.neuron_rw_judge build --step J0|A0 --model M --bank <arm0 out> [--out studies/neuronrw/judge]
    python -m infra.pg.neuron_rw_judge build --step J2|A2 --model M --bank <arm2 out> --arm0-bank <arm0 out> [...]
    python -m infra.pg.neuron_rw_judge run --step J0|A0|J2|A2 --model M [--out ...] [--concurrency 8]   # A22 stop rule, then calls
    python -m infra.pg.neuron_rw_judge collect --step RX|J0|A0|J2|A2 --model M [--out ...] [--results-dir ...]

A22 (judge budget): A0 judges the subsample of judge_draws_<M>.json -> a22 (items, per-class task lists, the
with-vocabulary and original-label tasks on its ONE items, judge0_pc_poles, retest items and tasks); A2 judges the
neutral prompts of judge_prompts.json -> arm2_judged (fc3 cells and guesses) and retests round(retest_share x its
fc3_steer and fc3_alpha0 requests); J0 and J2 are unchanged. Every number comes from those plan files. Cost rule
(judge_prompts.json -> budget): `build` and `run` of A0, J2 and A2 price every answered judge call of the judge
directory from its usage.jsonl tokens, project the step (A0 at J0's mean yn cost; J2 and A2 at J0's mean fc3 cost plus
a2_extra_input_tokens, or at J2's mean cost once J2 has usage), and before the first A0 also the rest of the study
(calls_worst_case); they print the projection and refuse when spent + projection > stop_usd.

RE-EXPLAIN (SPEC §5.0, A1, A16, A19): two batches per model, `rx-<M>` (ids `explainer:rx:<pole>`, vocabulary lines
removed, every audit pole) and `rxv-<M>` (ids `explainer:rxv:<pole>`, the parent request verbatim with only the model
replaced, ONE poles; B1). Every request is passed through `neurons_llm.make_request`, so the client's role-prefix and
schema asserts hold. `plan/make_plan.py:reexplain_requests` rebuilds every request from the parent's
explainer requests; `verify_reexplain` checks each one independently against its parent (every field equal except
the model, which is read from the parent's judge record file; the user text equals the parent's minus exactly the two
lines `Output vocabulary, most promoted: ` and `Output vocabulary, most suppressed: `; the system text equals the
parent's with the clause ", and the ten vocabulary items ... unembedding)." replaced by "." and the bullet "- The
promoted and suppressed vocabulary is the output view. ..." deleted, each found exactly once, whitespace-tolerant
patterns written from the SPEC text); `verify_reexplain_vocab` checks that a with-vocabulary request equals its parent
in every field but the id and the model; the canonical lines of each batch must hash to `plan/reexplain_<M>.json →
requests_sha256` and `→ with_vocab.requests_sha256`.

Judge requests: every line is made by the parent's `neurons_llm.make_request` (role "judge", so ids start with
"judge:"); id = "judge:<batch>:<6-digit index>" in a seeded order; model, temperature and max_tokens from
`plan/judge_prompts.json → judge` (read from the parent's records by make_plan.py), asserted equal to the settings
file. Keys (never shown to the judge) in <out>/keys/<batch>.jsonl with the SCHEMA §4 fields.

Rules fixed here where SCHEMA, SPEC and the plan files are silent (for the spec writer to ratify):
- K1 Request order: numpy default_rng([20261025, model_index, ORDER_SEED[batch kind]]).permutation of the batch's
  requests in build order.
- K2 Labels: "reexplained" = results/reexplain_<M>.jsonl `explanation`; "with_vocab" = results/reexplain_vocab_<M>.jsonl
  `explanation` (a0-yn-vocab-<M>: yn_true, yn_shuf1, yn_shuf2 for the ONE items, A16); "original" = roles poles `label`. A request
  whose shown label (or list) belongs to a pole without a re-explanation is not built, and counted.
- K3 G-judge0 goes to a0-pc-<M> (tasks pc_true, pc_shuf; re-explained labels; firing-token lists of
  arm0/firing_tokens.jsonl); a0-retest-<M> repeats the re-explained yn_true and fc3 of judge_draws.retest.
- K4 Arm 2 (SPEC §5.7; A27: when this G-coh, computed as freeze2 computes it, fails, no coherence filter is applied and
  every sample of a cell is shown): the Arm 2 coherence band (the sweep's and Arm 2's alpha = 0 rows pooled in file order, split
  by neuron_rw_core.split_halves; natural calibration spans) decides which samples are shown; a cell needs >= 3
  coherent samples; texts in k order; alpha = 0 cells (shared rows) use each pole's own triple (task fc3_alpha0); the
  label-prompted cell (condition label_prompt) is task fc3_label_prompt; guesses at plan.primary_dose.
- K5 a2-retest (A22): round(arm2_judged.retest_share x the a2-fc3 requests of arm2_judged.retest_tasks), half to even,
  by default_rng([20261025, model_index, arm2_judged.retest_seed_purpose]).choice
  without replacement (SPEC A11).
Fail loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re

import numpy as np

from infra.pg.neuron_rw_core import MODEL_INDEX, SEED, is_coherent, read_jsonl

REQUEST_KEYS = ("id", "role", "model", "temperature", "max_tokens", "system", "user", "schema")
KEY_FIELDS = ("id", "label_set", "task", "model_key", "pole", "shown_label_pole", "list_pole", "candidates", "truth",
              "alpha", "prompt_i", "n_texts_shown", "retest_of", "preflight")
ORDER_SEED = {"j0": 19, "a0-yn": 20, "a0-yn-vocab": 29, "a0-yn-orig": 28, "a0-fc3": 21, "a0-retest": 22, "a0-pc": 23, "j2": 27, "a2-fc3": 24,
              "a2-guess": 25, "a2-retest": 26}
SETTINGS = pathlib.Path(__file__).resolve().parent / "settings" / "deepinfra_neuronrw_judge.json"
PLAN_DIR = pathlib.Path(__file__).resolve().parents[2] / "studies" / "neuronrw" / "plan"
N_PREFLIGHT = 20
MIN_SHOWN = 3
LETTERS = ("A", "B", "C")
STEP_BATCHES = {"RX": ("rx", "rxv"), "J0": ("j0",), "A0": ("a0-yn", "a0-yn-vocab", "a0-yn-orig", "a0-pc", "a0-retest"),
                "J2": ("j2",),
                "A2": ("a2-fc3", "a2-guess", "a2-retest")}
USER_PREFIXES = ("Output vocabulary, most promoted: ", "Output vocabulary, most suppressed: ")
RX_ID = {"rx": "explainer:rx:", "rxv": "explainer:rxv:"}          # SPEC A19
SYS_CLAUSE_RE = re.compile(r",\s+and\s+the\s+ten\s+vocabulary\s+items\s+its\s+output\s+most\s+promotes\s+and\s+most\s+"
                           r"suppresses\s+\(the\s+direct\s+effect\s+of\s+the\s+neuron's\s+output\s+weights\s+read\s+through\s+"
                           r"the\s+unembedding\)\.")
SYS_BULLET_RE = re.compile(r"- The promoted and suppressed vocabulary is the output view\.[^\n]*\n(?:  [^\n]*\n)*")


def make_plan_module(plan_dir=PLAN_DIR):
    spec = importlib.util.spec_from_file_location("neuronrw_make_plan", pathlib.Path(plan_dir) / "make_plan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render(strings: list) -> str:
    """judge_prompts.json render rule: one JSON string per line, json.dumps(s, ensure_ascii=False)."""
    return "\n".join(json.dumps(s, ensure_ascii=False) for s in strings)


def load_prompts(plan_dir) -> dict:
    jp = json.loads((pathlib.Path(plan_dir) / "judge_prompts.json").read_text())
    assert jp["schema"] == "neuronrw/judge-prompts/v1", jp["schema"]
    st, j = json.loads(SETTINGS.read_text()), jp["judge"]
    assert j["model"] == st["models"]["judge"] == st["models"]["explainer"], f"judge model {j['model']} vs settings {st['models']}"
    assert (j["backend"], j["base_url"], j["temperature"], j["key_env"]) == \
        (st["backend"], st["base_url"], st["temperature"], st["key_env"]), "judge_prompts.json judge and the settings file differ"
    return jp


# ------------------------------------------------------------------------------------------ RE-EXPLAIN (§5.0)
def verify_reexplain(rx: list, parent: dict, model: str) -> dict:
    """Every rebuilt request against its parent request (`explainer:<pole>:0`): only the model, the two user lines
    and the two system passages differ (module docstring). Returns counts; raises on the first difference."""
    n_sys = 0
    for r in rx:
        assert r["id"].startswith(RX_ID["rx"]), f"{r['id']}: not an {RX_ID['rx']} id"
        pole = r["id"][len(RX_ID["rx"]):]
        q = parent[f"explainer:{pole}:0"]
        assert tuple(r) == REQUEST_KEYS, f"{r['id']}: keys {tuple(r)}"
        assert r["model"] == model, f"{r['id']}: model {r['model']} is not the record's {model}"
        for k in ("role", "temperature", "max_tokens", "schema"):
            assert r[k] == q[k], f"{r['id']}: {k} differs from the parent request"
        lines = q["user"].split("\n")
        drop = [x for x in lines if x.startswith(USER_PREFIXES)]
        assert sorted(x[:len(USER_PREFIXES[0])] == USER_PREFIXES[0] for x in drop) == [False, True], \
            f"{r['id']}: the two vocabulary lines are not found once each"
        assert r["user"].split("\n") == [x for x in lines if not x.startswith(USER_PREFIXES)], f"{r['id']}: user text edited beyond the two lines"
        sys_ = q["system"]
        assert len(SYS_CLAUSE_RE.findall(sys_)) == 1 and len(SYS_BULLET_RE.findall(sys_)) == 1, f"{r['id']}: system passages not found once"
        assert r["system"] == SYS_BULLET_RE.sub("", SYS_CLAUSE_RE.sub(".", sys_)), f"{r['id']}: system text edited beyond the two passages"
        n_sys += r["system"] != sys_
    return {"n_verified": len(rx), "n_system_edited": n_sys}


def verify_reexplain_vocab(rxv: list, parent: dict, model: str) -> dict:
    """A16: each with-vocabulary request equals its parent request in every field but the id and the model."""
    for r in rxv:
        assert r["id"].startswith(RX_ID["rxv"]), f"{r['id']}: not an {RX_ID['rxv']} id"
        q = parent[f"explainer:{r['id'][len(RX_ID['rxv']):]}:0"]
        assert tuple(r) == REQUEST_KEYS, f"{r['id']}: keys {tuple(r)}"
        assert r["model"] == model, f"{r['id']}: model {r['model']} is not the record's {model}"
        for k in REQUEST_KEYS[1:]:
            assert k == "model" or r[k] == q[k], f"{r['id']}: {k} differs from the parent request"
    return {"n_verified": len(rxv)}


def through_make_request(rows: list) -> int:
    """Every line rebuilt by the parent client's make_request (role prefix, Opus refusal, schema check) must come back
    unchanged (SPEC A19)."""
    from infra.pg import neurons_llm as L
    for r in rows:
        assert L.make_request(r["id"], r["role"], r["model"], r["temperature"], r["max_tokens"], r["system"], r["user"],
                              r["schema"]) == r, f"{r['id']}: make_request changes the line"
    return len(rows)


def canonical_sha256(rx: list) -> str:
    return hashlib.sha256("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rx).encode("utf-8")).hexdigest()


def rebuild_reexplain(model_key: str, plan_dir=PLAN_DIR) -> tuple:
    """(rx requests, rxv requests, report): make_plan.reexplain_requests on roles.reexplain_poles (rx) and roles.one with
    keep_vocab=True (rxv), the judge settings read from the parent record file, each verified against the parent
    requests, passed through make_request, and hashed against the manifest."""
    pd = pathlib.Path(plan_dir)
    mp = make_plan_module(pd)
    roles = json.loads((pd / f"roles_{model_key}.json").read_text())
    man = json.loads((pd / f"reexplain_{model_key}.json").read_text())
    judge = mp.judge_settings()
    jp = load_prompts(pd)
    assert {k: v for k, v in judge.items() if k != "sources"} == {k: v for k, v in jp["judge"].items() if k != "sources"}, \
        "judge settings read now differ from plan/judge_prompts.json"
    rec = json.loads(pathlib.Path(judge["sources"]["model_backend_temperature"]["path"]).read_text())
    assert judge["model"] == rec["reported_models"][0] == man["model"], "re-explain model is not the record's reported model"
    rx, parent_path = mp.reexplain_requests(model_key, roles["reexplain_poles"], judge)
    assert hashlib.sha256(pathlib.Path(parent_path).read_bytes()).hexdigest() == man["parent_requests"]["sha256"], \
        f"{parent_path} changed since the manifest"
    parent = {q["id"]: q for q in read_jsonl(parent_path)}
    rep = verify_reexplain(rx, parent, judge["model"])
    rxv, _ = mp.reexplain_requests(model_key, roles["one"], judge, keep_vocab=True)
    rep_v = verify_reexplain_vocab(rxv, parent, judge["model"])
    wv = man["with_vocab"]
    for rows, m_ in ((rx, man), (rxv, wv)):
        sha = canonical_sha256(rows)
        assert sha == m_["requests_sha256"] == hashlib.sha256(mp.reexplain_lines(rows).encode("utf-8")).hexdigest(), \
            f"{m_['batch']}: rebuilt requests hash {sha}, manifest {m_['requests_sha256']}"
        assert [r["id"] for r in rows] == m_["ids"] and len(rows) == m_["n_requests"], f"{m_['batch']}: ids differ from the manifest"
        through_make_request(rows)
    cls = {}
    for p in roles["reexplain_poles"]:
        c = roles["poles"][p]["class"]
        cls[c] = cls.get(c, 0) + 1
    assert all(roles["poles"][p]["class"] == "ONE" for p in roles["one"]), "roles.one holds a pole that is not ONE"
    return rx, rxv, {"model_key": model_key, "batch": man["batch"], "n_requests": len(rx), "by_class": cls,
                     "requests_sha256": canonical_sha256(rx), "model": judge["model"], **rep, "n_through_make_request": len(rx),
                     "with_vocab": {"batch": wv["batch"], "n_requests": len(rxv), "requests_sha256": canonical_sha256(rxv),
                                    **rep_v, "n_through_make_request": len(rxv)}}


def cmd_reexplain(a):
    from infra.pg import neurons_llm as L
    rx, rxv, rep = rebuild_reexplain(a.model, a.plan_dir)
    print("REEXPLAIN_VERIFIED", json.dumps(rep))
    if a.dry_run:
        return
    for batch, rows in ((rep["batch"], rx), (rep["with_vocab"]["batch"], rxv)):
        L.write_requests(a.out, batch, rows)
        L.run_batch(a.out, batch, L.load_settings(SETTINGS), a.concurrency)
        print("REEXPLAIN_RUN_OK", batch, a.out)


# ------------------------------------------------------------------------------------------ judge builder
class Builder:
    def __init__(self, model_key: str, jp: dict, preflight: bool):
        self.mk, self.jp, self.pre = model_key, jp, preflight

    def req(self, template: str, schema: str, **fields) -> dict:
        from infra.pg import neurons_llm as L
        t, j = self.jp[template], self.jp["judge"]
        return L.make_request("judge:pending", "judge", j["model"], j["temperature"], j["max_tokens"], t["system"],
                              t["user"].format(**fields), self.jp["schemas"][schema])

    def key(self, label_set, task, pole, shown=None, list_pole=None, candidates=None, truth=None, alpha=None, prompt_i=None,
            n_texts=None) -> dict:
        return dict(zip(KEY_FIELDS, (None, label_set, task, self.mk, pole, shown, list_pole, candidates, truth, alpha,
                                     prompt_i, n_texts, None, self.pre)))

    def finish(self, kind: str, pairs: list) -> tuple:
        """K1: permute (request, key) pairs and assign ids judge:<batch>:<6 digits>; returns (batch, requests, keys)."""
        batch = f"{kind}-{self.mk}"
        perm = np.random.default_rng([SEED, MODEL_INDEX[self.mk], ORDER_SEED[kind]]).permutation(len(pairs))
        reqs, keys = [], []
        for j, i in enumerate(perm):
            r, k = pairs[i]
            rid = f"judge:{batch}:{j:06d}"
            reqs.append({**r, "id": rid})
            keys.append({**k, "id": rid})
            assert tuple(reqs[-1]) == REQUEST_KEYS
        through_make_request(reqs)                          # final ids pass the client's role-prefix assert (A19)
        return batch, reqs, keys

    def retest(self, kind: str, src: list) -> tuple:
        return self.finish(kind, [({**r, "id": None}, {**k, "id": None, "retest_of": r["id"]}) for r, k in src])


def arm0_pairs(b: Builder, draws: dict, projections: dict, firing_tokens: dict, labels: dict, items: list, pc: list,
               tasks: tuple, label_set: str) -> tuple:
    """(pairs by kind, n skipped for a missing label)."""
    def toks(p):
        top = projections[p]["top"]
        assert len(top) == 20, f"{p}: {len(top)} attested top tokens"
        return render([x["str"] for x in top])
    out, skipped = {"yn": [], "fc3": [], "pc": []}, 0
    for p in items:
        d = draws["draws"][p]
        specs = {"yn_true": (p, p), "yn_shuf1": (d["shuf1"], p), "yn_shuf2": (d["shuf2"], p), "yn_fixed": (p, d["fixed_list"])}
        for task in tasks:
            if task == "fc3":
                c = d["fc3_candidates"]
                assert c[LETTERS.index(d["fc3_truth"])] == p, f"{p}: the truth letter does not point at the pole"
                if not all(q in labels for q in c):
                    skipped += 1
                    continue
                out["fc3"].append((b.req("a0_fc3", "fc3", tokens=toks(p), A=labels[c[0]], B=labels[c[1]], C=labels[c[2]]),
                                   b.key(label_set, "fc3", p, list_pole=p, candidates=list(c), truth=d["fc3_truth"])))
                continue
            shown, lst = specs[task]
            if shown not in labels:
                skipped += 1
                continue
            out["yn"].append((b.req("a0_yn", "yn", tokens=toks(lst), label=labels[shown]), b.key(label_set, task, p, shown, lst)))
    for e in pc:
        ft = firing_tokens[e["pole"]]["tokens"]
        assert 10 <= len(ft) <= 20, f"{e['pole']}: {len(ft)} firing tokens (A20: 10 to 20)"
        for task, shown in (("pc_true", e["pole"]), ("pc_shuf", e["shuf1"])):
            if shown not in labels:
                skipped += 1
                continue
            out["pc"].append((b.req("a0_yn", "yn", tokens=render(ft), label=labels[shown]),
                              b.key(label_set, task, e["pole"], shown, e["pole"])))
    return out, skipped


PRIMARY0 = ("yn_true", "yn_shuf1", "yn_shuf2", "yn_fixed", "fc3")


def build0_compute(step: str, model_key: str, roles: dict, draws: dict, jp: dict, projections: dict, firing_tokens: dict,
                   reexplained: dict, with_vocab: dict) -> tuple:
    """([(batch, requests, keys)], counts) for J0 or A0 (SPEC §5.1, A1, A8, A16)."""
    assert draws["model_key"] == roles["model_key"] == model_key, "draws, roles and --model differ"
    original = {p: roles["poles"][p]["label"] for p in roles["poles"]}
    if step == "J0":
        b = Builder(model_key, jp, True)
        x, sk = arm0_pairs(b, draws, projections, firing_tokens, reexplained, draws["items"][:N_PREFLIGHT],
                           draws["judge0_pc"][:N_PREFLIGHT], PRIMARY0, "reexplained")
        return [b.finish("j0", x["yn"] + x["fc3"] + x["pc"])], {"skipped_no_label": sk}
    assert step == "A0", step
    sub = draws["a22"]                                        # A22: the judged subsample and its task lists
    cls = {p: roles["poles"][p]["class"] for p in sub["items"]}
    one_sub = [p for p in sub["items"] if cls[p] == "ONE"]
    b = Builder(model_key, jp, False)
    yn_p, fc3_p, sk = [], [], 0
    for c in ("ONE", "UNK"):
        x, k = arm0_pairs(b, draws, projections, firing_tokens, reexplained, [p for p in sub["items"] if cls[p] == c], [],
                          tuple(sub["tasks"][c]), "reexplained")
        yn_p, fc3_p, sk = yn_p + x["yn"], fc3_p + x["fc3"], sk + k
    pc_draw = {e["pole"]: e for e in draws["judge0_pc"]}
    xpc, kpc = arm0_pairs(b, draws, projections, firing_tokens, reexplained, [], [pc_draw[p] for p in sub["judge0_pc_poles"]],
                          (), "reexplained")
    xv, skv = arm0_pairs(b, draws, projections, firing_tokens, with_vocab, one_sub, [], tuple(sub["secondary_tasks"]["with_vocab"]),
                         "with_vocab")
    xo, sko = arm0_pairs(b, draws, projections, firing_tokens, original, one_sub, [], tuple(sub["secondary_tasks"]["original"]),
                         "original")
    yn = b.finish("a0-yn", yn_p)
    out = [yn, b.finish("a0-yn-vocab", xv["yn"]), b.finish("a0-yn-orig", xo["yn"]), b.finish("a0-pc", xpc["pc"])]
    out += [b.finish("a0-fc3", fc3_p)] if fc3_p else []
    orig = {(k["task"], k["pole"]): (r, k) for _, reqs, keys in [yn] + out[4:] for r, k in zip(reqs, keys)}
    src = [orig[(t, p)] for p in sub["retest"] for t in sub["retest_tasks"] if (t, p) in orig]
    return out + [b.retest("a0-retest", src)], {"skipped_no_label": sk + kpc, "skipped_no_label_with_vocab": skv,
                                                "skipped_no_label_original": sko}


def a0_kinds(sub: dict) -> list:
    """The A0 batch kinds the a22 block implies (a0-fc3 only when a task list holds fc3)."""
    return list(STEP_BATCHES["A0"]) + (["a0-fc3"] if any("fc3" in t for t in sub["tasks"].values()) else [])


def arm2_pairs(b: Builder, roles: dict, plan: dict, samples: list, band: dict, labels: dict, poles: list, steered: bool,
               judged: dict | None = None) -> dict:
    """judged: judge_prompts.json -> arm2_judged (A22: the prompts A2 judges); None (J2) judges every prompt."""
    ent = {e["pole"]: e for e in roles["arm2"]}
    cells = {}
    for r in samples:
        key = ("label", r["pole"], None) if r["condition"] == "label_prompt" else (r["pole"], r["alpha"], r["prompt_i"])
        cells.setdefault(key, []).append(r)

    def shown(rows):
        rows = sorted((r for r in rows if band is None or is_coherent(r, band)), key=lambda r: r["k"])   # band None: A27
        return rows if len(rows) >= MIN_SHOWN else None
    out = {"fc3": [], "guess": [], "skipped_no_label": 0}
    d_p = float(plan["primary_dose"])
    for p in poles:
        e = ent[p]
        c = e["candidates"]
        assert c[LETTERS.index(e["truth"])] == p, f"{p}: the truth letter does not point at the pole"
        if not all(q in labels for q in c):
            out["skipped_no_label"] += 1
            continue
        lab3 = {"A": labels[c[0]], "B": labels[c[1]], "C": labels[c[2]]}
        for alpha in [0.0] + ([float(x) for x in plan["doses"]] if steered else []):
            for i in (range(len(plan["neutral_prompts"])) if judged is None else judged["prompts"]):
                rows = shown(cells.get((None if alpha == 0 else p, alpha, i), []))
                if rows is None:
                    continue
                texts = render([r["completion"] for r in rows])
                out["fc3"].append((b.req("a2_fc3", "fc3", texts=texts, **lab3),
                                   b.key("reexplained", "fc3_alpha0" if alpha == 0 else "fc3_steer", p, candidates=list(c),
                                         truth=e["truth"], alpha=alpha, prompt_i=i, n_texts=len(rows))))
                if steered and alpha == d_p and i in judged["guess_prompts"]:
                    out["guess"].append((b.req("a2_guess", "guess", texts=texts),
                                         b.key("reexplained", "guess", p, alpha=alpha, prompt_i=i, n_texts=len(rows))))
        rows = shown(cells.get(("label", p, None), []))
        if rows is not None:
            out["fc3"].append((b.req("a2_fc3", "fc3", texts=render([r["completion"] for r in rows]), **lab3),
                               b.key("reexplained", "fc3_label_prompt", p, candidates=list(c), truth=e["truth"], alpha=0.0,
                                     n_texts=len(rows))))
    return out


def build2_compute(step: str, model_key: str, roles: dict, jp: dict, plan: dict, samples: list, band: dict, labels: dict) -> tuple:
    assert {e["pole"] for e in roles["arm2"]} == {q["pole"] for q in plan["poles"]}, "roles.arm2 and the plan differ"
    assert float(plan["primary_dose"]) in [float(x) for x in plan["doses"]], "primary dose not among the plan's doses"
    if step == "J2":
        b = Builder(model_key, jp, True)
        first = [e["pole"] for e in sorted((e for e in roles["arm2"] if e["role"] == "ONE"), key=lambda e: e["draw_index"])][:N_PREFLIGHT]
        x = arm2_pairs(b, roles, plan, samples, band, labels, first, False)
        return [b.finish("j2", x["fc3"])], {"skipped_no_label": x["skipped_no_label"]}
    assert step == "A2", step
    b = Builder(model_key, jp, False)
    aj = jp["arm2_judged"]
    x = arm2_pairs(b, roles, plan, samples, band, labels, [q["pole"] for q in plan["poles"]], True, aj)
    fc3 = b.finish("a2-fc3", x["fc3"])
    pool = [j for j, k in enumerate(fc3[2]) if k["task"] in aj["retest_tasks"]]          # A22: 10% of the fc3 cells
    n = int(round(aj["retest_share"] * len(pool)))
    g = np.random.default_rng([SEED, MODEL_INDEX[model_key], aj["retest_seed_purpose"]])
    pick = sorted(g.choice(pool, n, replace=False).tolist())
    return [fc3, b.finish("a2-guess", x["guess"]), b.retest("a2-retest", [(fc3[1][j], fc3[2][j]) for j in pick])], \
        {"skipped_no_label": x["skipped_no_label"]}


def arm2_coherence(arm0_bank, bank, samples: list, model_key: str) -> dict:
    """SPEC §5.2 and freeze2's G-coh, computed the same way (neuron_rw_analysis.coherence_from): the sweep's and Arm 2's
    alpha = 0 rows pooled in file order, the natural calibration spans, the Arm 2 bank's coherence inputs and the
    label-prompted rows."""
    from infra.pg.neuron_rw_analysis import coherence_from
    b0 = pathlib.Path(arm0_bank)
    a0 = [r for r in read_jsonl(b0 / "sweep" / "samples.jsonl") if r["alpha"] == 0] + \
        [r for r in samples if r["condition"] == "steer" and r["pole"] is None]
    cal = [r for r in read_jsonl(b0 / "arm0" / "natural_spans.jsonl") if r["set"] == "calibration"]
    return coherence_from(a0, cal, read_jsonl(pathlib.Path(bank) / "samples" / "coherence_inputs.jsonl"), model_key,
                          [r for r in samples if r["condition"] == "label_prompt"])


def reexplained_labels(results_dir, model_key: str, variant: str = "") -> dict:
    return {r["pole"]: r["explanation"] for r in read_jsonl(pathlib.Path(results_dir) / f"reexplain_{variant}{model_key}.jsonl")}


# ------------------------------------------------------------------------------------------ A22 cost rule
def usage_rows(judge_dir) -> dict:
    """{batch: [usage rows]} of every responses/<batch>.usage.jsonl in the judge directory."""
    d = pathlib.Path(judge_dir) / "responses"
    return {f.name[:-len(".usage.jsonl")]: read_jsonl(f) for f in sorted(d.glob("*.usage.jsonl"))} if d.exists() else {}


def call_usd(u: dict, bud: dict) -> float:
    return u["prompt_tokens"] * bud["usd_per_m_input"] / 1e6 + u["completion_tokens"] * bud["usd_per_m_output"] / 1e6


def projection(step: str, judge_dir, bud: dict, n_calls: int) -> dict:
    """SPEC §14.2 A22 (5): spent (every answered judge call), the step's projected cost and, before the first A0, the
    rest of the study; refuse = spent + projection > stop_usd."""
    use = usage_rows(judge_dir)
    spent = sum(call_usd(u, bud) for rows in use.values() for u in rows)
    j0 = [b for b in use if b.startswith("j0-")]
    assert len(j0) == 1, f"the A22 projection needs the J0 usage (one j0-<M> usage file), found {j0}"
    task = {k["id"]: k["task"] for k in read_jsonl(pathlib.Path(judge_dir) / "keys" / f"{j0[0]}.jsonl")}
    yn = [call_usd(u, bud) for u in use[j0[0]] if task[u["id"]] != "fc3"]
    fc3 = [call_usd(u, bud) for u in use[j0[0]] if task[u["id"]] == "fc3"]
    assert yn and fc3, "J0 usage lacks yn or fc3 calls"
    j2 = [call_usd(u, bud) for b in use if b.startswith("j2-") for u in use[b]]
    a2 = float(np.mean(j2)) if j2 else float(np.mean(fc3)) + bud["a2_extra_input_tokens"] * bud["usd_per_m_input"] / 1e6
    unit = float(np.mean(yn)) if step == "A0" else a2
    out = {"step": step, "spent_usd": spent, "n_answered": sum(map(len, use.values())), "unit_usd": unit,
           "unit_source": "J0 mean yn" if step == "A0" else ("J2 mean" if j2 else "J0 mean fc3 + a2_extra_input_tokens"),
           "step_calls": n_calls, "step_usd": n_calls * unit, "stop_usd": bud["stop_usd"], "rest_usd": None}
    if step == "A0" and not any(b.startswith("a0-") for b in use):
        c = bud["calls_worst_case"]
        out["rest_usd"] = c["A0"] * float(np.mean(yn)) + (c["J2"] + c["A2"]) * a2
    out["refuse"] = spent + out["step_usd"] > bud["stop_usd"] or (out["rest_usd"] is not None and spent + out["rest_usd"] > bud["stop_usd"])
    return out


def budget_gate(step: str, judge_dir, jp: dict, n_calls: int):
    """Print the projection; raise SystemExit when the step must not start (A22). J0 and RX are not gated."""
    if step not in ("A0", "J2", "A2"):
        return
    pr = projection(step, judge_dir, jp["budget"], n_calls)
    print("JUDGE_BUDGET", json.dumps(pr))
    if pr["refuse"]:
        raise SystemExit(f"A22 stop rule: {step} refused (spent {pr['spent_usd']:.2f} + projected step {pr['step_usd']:.2f}"
                         f" / rest {pr['rest_usd']} USD against stop_usd {pr['stop_usd']})")


# ------------------------------------------------------------------------------------------ files and commands
def write_new(path, rows: list):
    path = pathlib.Path(path)
    assert not path.exists(), f"refusing to overwrite {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def cmd_build(a):
    from infra.pg import neurons_llm as L
    pd = pathlib.Path(a.plan_dir)
    if a.step == "RX":
        rx, rxv, rep = rebuild_reexplain(a.model, pd)
        L.write_requests(a.out, rep["batch"], rx)
        L.write_requests(a.out, rep["with_vocab"]["batch"], rxv)
        print("BUILD_OK RX", rep["batch"], rep["with_vocab"]["batch"], json.dumps(rep))
        return
    roles = json.loads((pd / f"roles_{a.model}.json").read_text())
    jp, bank = load_prompts(pd), pathlib.Path(a.bank)
    labels = reexplained_labels(a.results_dir, a.model)
    if a.step in ("J0", "A0"):
        draws = json.loads((pd / f"judge_draws_{a.model}.json").read_text())
        proj = {r["pole"]: r for r in read_jsonl(bank / "arm0" / "projections.jsonl")}
        ft = {r["pole"]: r for r in read_jsonl(bank / "arm0" / "firing_tokens.jsonl")}
        vocab = reexplained_labels(a.results_dir, a.model, "vocab_") if a.step == "A0" else {}
        out, counts = build0_compute(a.step, a.model, roles, draws, jp, proj, ft, labels, vocab)
    else:
        assert a.arm0_bank, "J2 and A2 need --arm0-bank (the sweep's alpha = 0 samples and the calibration spans)"
        plan = json.loads((pd / f"arm2_{a.model}.json").read_text())
        samples = read_jsonl(bank / "arm2" / "samples.jsonl")
        gc = arm2_coherence(a.arm0_bank, bank, samples, a.model)
        band = gc["band"] if gc["pass"] else None           # A27: G-coh failed -> every sample shown, labels stamped later
        out, counts = build2_compute(a.step, a.model, roles, jp, plan, samples, band, labels)
        counts = {**counts, "g_coh_pass": gc["pass"], "coherence_filter": band is not None}
    budget_gate(a.step, a.out, jp, sum(len(r) for _, r, _ in out))
    for batch, reqs, keys in out:
        write_new(pathlib.Path(a.out) / "requests" / f"{batch}.jsonl", reqs)
        write_new(pathlib.Path(a.out) / "keys" / f"{batch}.jsonl", keys)
    print("BUILD_OK", a.step, ",".join(b for b, _, _ in out), json.dumps({**{b: len(r) for b, r, _ in out}, **counts}))


def collect_compute(keys: list, answers: dict, declined: set) -> list:
    """One line per key: key fields + answer + correct + declined (answers: neurons_llm.collect's {id: object})."""
    out = []
    for k in keys:
        v = answers.get(k["id"])
        assert v is not None or k["id"] in declined, f"{k['id']}: neither answered nor declined"
        if v is None:
            ans, correct = None, None
        elif "match" in v:
            ans, correct = v["match"], v["match"] == "yes"
        elif "choice" in v:
            ans, correct = v["choice"], v["choice"] == k["truth"]
        else:
            ans, correct = v["guess"], None
        out.append({**k, "answer": ans, "correct": correct, "declined": v is None})
    return out


def collect_reexplain(answers: dict, declined: set, ids: list, model: str, prefix: str) -> tuple:
    """SCHEMA §4: results/reexplain_[vocab_]<M>.jsonl lines {pole, explanation, class, confidence, jobs, model} and the
    declined poles."""
    rows, dec = [], []
    for rid in ids:
        assert rid.startswith(prefix), f"{rid}: not a {prefix} id"
        v, pole = answers.get(rid), rid[len(prefix):]
        assert v is not None or rid in declined, f"{rid}: neither answered nor declined"
        if v is None:
            dec.append({"pole": pole, "id": rid})
            continue
        rows.append({"pole": pole, "explanation": v["explanation"], "class": v["class"], "confidence": v["confidence"],
                     "jobs": v["jobs"], "model": model})
    return rows, dec


def step_kinds(step: str, plan_dir, model: str) -> list:
    if step != "A0":
        return list(STEP_BATCHES[step])
    return a0_kinds(json.loads((pathlib.Path(plan_dir) / f"judge_draws_{model}.json").read_text())["a22"])


def cmd_run(a):
    """A22: the stop rule on the step's unanswered requests, then neurons_llm.run_batch for each of its batches."""
    from infra.pg import neurons_llm as L
    jd = pathlib.Path(a.out)
    batches = [f"{k}-{a.model}" for k in step_kinds(a.step, a.plan_dir, a.model)]
    done = {u["id"] for rows in usage_rows(jd).values() for u in rows}
    todo = sum(1 for bt in batches for r in read_jsonl(jd / "requests" / f"{bt}.jsonl") if r["id"] not in done)
    budget_gate(a.step, jd, load_prompts(a.plan_dir), todo)
    for bt in batches:
        L.run_batch(jd, bt, L.load_settings(SETTINGS), a.concurrency)
        print("RUN_OK", bt)


def cmd_collect(a):
    from infra.pg import neurons_llm as L
    backend = a.backend or json.loads(SETTINGS.read_text())["backend"]
    rd = pathlib.Path(a.results_dir)
    if a.step == "RX":
        man = json.loads((pathlib.Path(a.plan_dir) / f"reexplain_{a.model}.json").read_text())
        for m_, kind, name in ((man, "rx", f"reexplain_{a.model}"), (man["with_vocab"], "rxv", f"reexplain_vocab_{a.model}")):
            answers = L.collect(a.out, m_["batch"], backend)
            rows, dec = collect_reexplain(answers, set(L.declined(a.out, m_["batch"])), m_["ids"], man["model"], RX_ID[kind])
            write_new(rd / f"{name}.jsonl", rows)
            write_new(rd / f"{name}_declined.jsonl", dec)
            print("COLLECT_OK", m_["batch"], json.dumps({"answered": len(rows), "declined": len(dec)}))
        return
    for kind in step_kinds(a.step, a.plan_dir, a.model):
        batch = f"{kind}-{a.model}"
        answers = L.collect(a.out, batch, backend)
        rows = collect_compute(read_jsonl(pathlib.Path(a.out) / "keys" / f"{batch}.jsonl"), answers, set(L.declined(a.out, batch)))
        path = rd / f"judge_{batch}.jsonl"
        write_new(path, rows)
        print("COLLECT_OK", batch, json.dumps({"n": len(rows), "declined": sum(r["declined"] for r in rows),
                                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="neuron_rw_judge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reexplain")
    r.add_argument("--model", required=True)
    r.add_argument("--dry-run", action="store_true", help="rebuild, verify and hash the requests; write nothing, call nothing")
    r.add_argument("--plan-dir", default=str(PLAN_DIR))
    r.add_argument("--out", default="bank/rw/reexplain")
    r.add_argument("--concurrency", type=int, default=8)
    b = sub.add_parser("build")
    b.add_argument("--step", required=True, choices=sorted(STEP_BATCHES))
    b.add_argument("--model", required=True)
    b.add_argument("--bank", default=None, help="J0, A0: the arm0 out dir; J2, A2: the arm2 out dir")
    b.add_argument("--arm0-bank", default=None)
    b.add_argument("--plan-dir", default=str(PLAN_DIR))
    b.add_argument("--results-dir", default="studies/neuronrw/results")
    b.add_argument("--out", default="studies/neuronrw/judge")
    u = sub.add_parser("run")
    u.add_argument("--step", required=True, choices=["J0", "A0", "J2", "A2"])
    u.add_argument("--model", required=True)
    u.add_argument("--plan-dir", default=str(PLAN_DIR))
    u.add_argument("--out", default="studies/neuronrw/judge")
    u.add_argument("--concurrency", type=int, default=8)
    c = sub.add_parser("collect")
    c.add_argument("--step", required=True, choices=sorted(STEP_BATCHES))
    c.add_argument("--model", required=True)
    c.add_argument("--plan-dir", default=str(PLAN_DIR))
    c.add_argument("--out", default="studies/neuronrw/judge")
    c.add_argument("--results-dir", default="studies/neuronrw/results")
    c.add_argument("--backend", default=None, help="neurons_llm collect backend; default the settings file's")
    a = ap.parse_args(argv)
    if a.cmd == "build" and a.step != "RX":
        assert a.bank, f"build --step {a.step} needs --bank"
    {"reexplain": cmd_reexplain, "build": cmd_build, "run": cmd_run, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
