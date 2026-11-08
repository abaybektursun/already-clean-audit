"""INSTRUMENTS E6b job: the OLMo-2 checkpoint (studies/instruments/SPEC_instruments.md, section "E6", part E6b). Two pipeline stages after the
harvest step (jobs.neurons_harvest, unchanged, model key O2); the pipeline's steps share the instance's /work/out, each step writing /work/out/<name>.

    python -m infra.aws_pg.jobs.instruments_e6b --stage items --harvest /work/out/harvest --out <dir>
    python -m infra.aws_pg.jobs.instruments_e6b --stage e1b --items /work/out/items/e6b_items.json --corpus <s3 corpus> --out <dir>

items  E1b's item rule (studies/instruments/scripts/build_items_e1b.py `build`, which the code tarball does not ship) applied to the harvest step's
       bank: 56 planted pairs (seed 4; 18/20/18 by layer thirds; each side its own neuron's top-50 contexts, separate documents, shared marked
       tokens dropped, 12 per side), 30 singles (seed 5), 3 known poles (16 contexts), 200 background positions (seed 0). build() below is that
       function with the bank directory passed in; the dry run asserts it returns build_items_e1b.build's record exactly on the g4, q4b and o7_t15
       banks. Writes <out>/e6b_items.json in E1b's format ({"models": {"o2": record}, ...}); the record adds model_key o2 and branch (the sha
       downloaded by the next stage).
e1b    jobs.instruments_e1b.Job, unchanged, model key o2, on OLMo-2 downloaded in-job from the Hub at the record's sha (jobs.neurons_harvest.download
       after hf_online_setup; the sha asserted against matrix_models.KNOWN); every value E1b reads (mean, zero, median, res0..res3; the verdict
       reads mean and zero); E1b's gates in its metrics.json. Then T5 on OLMo-2 (jobs.instruments_e5.t5_model, unchanged): the distinct
       high-background members of the side-J unexcluded cross sets (set_J_members of every pair npz), one capture of E1b's 200 background
       positions (s_l, the residual rms; its mean asserted equal to E1b's ā at every member; failing input the first half's mean), ||d_k|| from the
       weights; banked as t5/o2.npz and t5/metrics.json. The per-layer |ā| profile is values.npz's ā [n_up, W] (E1b's bank). The weights are
       deleted at the end.
Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import shutil
import time

import numpy as np

from infra.pg import unite_route as UR

REPO_O2 = "allenai/OLMo-2-1124-7B"
SPEC_O2 = {"repo": REPO_O2, "arch": "olmo2", "nh_key": "O2"}
MK = "o2"
CORPUS = "s3://pg-lineage-769063704696/results/neurons/w1/q4b/1/out/corpus/corpus"
# build_items_e1b's constants, unchanged
CAP, N_KNOWN_CTX = 12, 16
PAIR_SEED, SINGLE_SEED, BG_SEED = 4, 5, 0
PAIRS_BY_TERCILE = (18, 20, 18)
N_SINGLE_LAYERS = 15
LAMBDA_RANK = 24


# ------------------------------------------------------------------------------------------ items: build_items_e1b's rule on a bank directory
def sha256(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def tercile(L: int, n_layers: int) -> int:
    return 3 * L // n_layers


def parse(p: str):
    L, n, s = p[1:].split("N")[0], p.split("N")[1][:-1], p[-1]
    return int(L), int(n), s


def valid(ctx: list, sign: str) -> list:
    return [c for c in ctx if c["tok"] >= 0 and np.isfinite(c["a"]) and (c["a"] > 0 if sign == "+" else c["a"] < 0)]


def one_per_doc(ctx: list, drop_tok=frozenset(), drop_doc=frozenset(), cap: int = CAP) -> list:
    out, seen = [], set()
    for c in ctx:
        if c["token"] in drop_tok or c["doc"] in drop_doc or c["doc"] in seen:
            continue
        out.append(c)
        seen.add(c["doc"])
        if len(out) == cap:
            break
    return out


def side_rec(prefix: str, ctx: list) -> dict:
    return {f"{prefix}_tok": [c["tok"] for c in ctx], f"{prefix}_rank": [c["rank"] for c in ctx], f"{prefix}_doc": [c["doc"] for c in ctx],
            f"{prefix}_token": [c["token"] for c in ctx], f"{prefix}_a_bank": [c["a"] for c in ctx]}


def build(d: pathlib.Path, key: str, spec: dict) -> tuple[dict, dict]:
    """build_items_e1b.build(key) with the bank directory `d` in place of HV / key / 1 / out / key and the model spec passed in."""
    files = {x: d / x for x in ("manifest.json", "controls.json", "poles/poles.json", "poles/positions.npz", "poles/exemplar_a.npz",
                                 "samples/contexts.jsonl", "samples/sample_draw.json")}
    man = json.loads(files["manifest.json"].read_text())
    assert man["repo"] == spec["repo"] and man["model_type"] == spec["arch"] and man["window"] == UR.HARVEST_WINDOW, (key, man["repo"], man["model_type"])
    nL = int(man["n_layers"])
    poles = json.loads(files["poles/poles.json"].read_text())
    rows = [json.loads(x) for x in files["samples/contexts.jsonl"].read_text().splitlines() if x.strip()]
    assert [r["pole"] for r in rows] == poles, "contexts.jsonl is not in poles.json order"
    pos, ex = np.load(files["poles/positions.npz"])["tok"], np.load(files["poles/exemplar_a.npz"])["a"]
    for i, r in enumerate(rows):
        assert [c["tok"] for c in r["topk"]] == pos[i, :len(r["topk"])].tolist() and (pos[i, len(r["topk"]):] < 0).all(), r["pole"]
        assert [c["a"] for c in r["topk"]] == ex[i, :len(r["topk"])].tolist(), r["pole"]
    ctx = {r["pole"]: r["topk"] for r in rows}
    neurons = sorted({parse(p)[:2] for p in poles})
    by_layer = collections.defaultdict(list)
    for L, n in neurons:
        by_layer[L].append(n)

    rng = np.random.default_rng(PAIR_SEED)
    planted, used = [], set()
    for t, want in enumerate(PAIRS_BY_TERCILE):
        cands = [(L, a, b) for L in sorted(by_layer) if L >= 1 and tercile(L, nL) == t for i, a in enumerate(by_layer[L]) for b in by_layer[L][i + 1:]]
        kept = 0
        for i in rng.permutation(len(cands)):
            if kept == want:
                break
            L, a, b = cands[i]
            if (L, a) in used or (L, b) in used:
                continue
            ca, cb = valid(ctx[f"L{L:02d}N{a}+"], "+"), valid(ctx[f"L{L:02d}N{b}+"], "+")
            if len(ca) <= LAMBDA_RANK or len(cb) <= LAMBDA_RANK:
                continue
            J = one_per_doc(ca, frozenset(c["token"] for c in cb), frozenset(c["doc"] for c in cb))
            O = one_per_doc(cb, frozenset(c["token"] for c in ca), frozenset(c["doc"] for c in ca))
            if len(J) < CAP or len(O) < CAP:
                continue
            assert not ({c["doc"] for c in J} & {c["doc"] for c in O}) and not ({c["token"] for c in J} & {c["token"] for c in O})
            lam = float(ca[LAMBDA_RANK]["a"] / cb[LAMBDA_RANK]["a"])
            planted.append({"unit": f"PL{L:02d}_{len(planted):02d}", "layer": L, "tercile": t, "a": a, "b": b, "lam": lam,
                            "a_rank25": ca[LAMBDA_RANK]["a"], "b_rank25": cb[LAMBDA_RANK]["a"], **side_rec("J", J), **side_rec("O", O)})
            used |= {(L, a), (L, b)}
            kept += 1
        assert kept == want, (key, t, kept, want, "the tercile has too few valid pairs")

    elig = collections.defaultdict(dict)
    for p in poles:
        L, n, s = parse(p)
        if L >= 1 and (L, n) not in used:
            side = one_per_doc(valid(ctx[p], s))
            if len(side) == CAP:
                elig[L].setdefault(n, {})[s] = side
    rng = np.random.default_rng(SINGLE_SEED)
    lay = sorted(L for L in elig if len(elig[L]) >= 2)
    assert len(lay) >= N_SINGLE_LAYERS, (key, len(lay))
    singles = []
    for L in sorted(rng.choice(lay, N_SINGLE_LAYERS, replace=False).tolist()):
        two = sorted(rng.choice(sorted(elig[L]), 2, replace=False).tolist())
        got = []
        for n in two:
            signs = sorted(elig[L][n])
            s = signs[int(rng.integers(len(signs)))]
            got.append({"pole": f"L{L:02d}N{n}{s}", "layer": L, "neuron": n, "sign": s, "tercile": tercile(L, nL), **side_rec("J", elig[L][n][s])})
        got[0]["partner"], got[1]["partner"] = got[1]["pole"], got[0]["pole"]
        singles += got
    taken = used | {(s["layer"], s["neuron"]) for s in singles}

    kp = []
    for t in (0, 1, 2):
        p = next(p for p in poles if p.endswith("+") and parse(p)[0] >= 1 and tercile(parse(p)[0], nL) == t and parse(p)[:2] not in taken
                 and len(valid(ctx[p], "+")) >= N_KNOWN_CTX)
        kp.append(p)
    ktok = [[c["tok"] for c in ctx[p][:N_KNOWN_CTX]] for p in kp]
    ka = [[c["a"] for c in ctx[p][:N_KNOWN_CTX]] for p in kp]
    known = {"poles": kp, "tok": ktok, "a_bank": ka, "a_wrong": [ka[(i + 1) % 3] for i in range(3)], "n_ctx": N_KNOWN_CTX,
             "wrong_rule": "the next known pole's bank a (cyclic)", "rule": "the first + pole per tercile in poles.json order, layer >= 1, in no pair or single"}

    nw = int(man["n_tokens_windowed"])
    bg = UR.background_positions(BG_SEED - 1000, nw, UR.HARVEST_WINDOW)["e1"]
    assert (bg % UR.HARVEST_WINDOW != 0).all() and bg.max() < nw and len(set(bg.tolist())) == 200

    rec = {"model_key": key, "repo": man["repo"], "revision": man["revision"], "arch": spec["arch"], "nh_key": spec["nh_key"], "tokname": man["tokname"],
           "n_layers": nL, "width": int(man["width"]), "hidden": int(man["hidden"]), "bos_id": man["bos_id"], "window": int(man["window"]),
           "n_tokens_windowed": nw, "tokens_sha256": man["tokens_sha256"], "corpus_manifest_sha256": man["corpus_manifest_sha256"], "act_fn_measured": man["act_fn_measured"],
           "planted": planted, "singles": singles, "known": known,
           "background": {"positions": bg.tolist(), "seed": BG_SEED, "rule": "unite_route.background_positions(t_index=-1000) -> default_rng(0); window position 0 excluded"},
           "counts": {"planted": len(planted), "planted_by_tercile": [sum(p["tercile"] == t for p in planted) for t in (0, 1, 2)],
                      "planted_layers": sorted({p["layer"] for p in planted}), "singles": len(singles), "single_layers": sorted({s["layer"] for s in singles}),
                      "singles_by_sign": dict(collections.Counter(s["sign"] for s in singles)), "sampled_neurons": len(neurons),
                      "lambda_median": float(np.median([p["lam"] for p in planted])), "lambda_range": [min(p["lam"] for p in planted), max(p["lam"] for p in planted)]}}
    src = {f"{key}/{k}": {"path": str(v), "sha256": sha256(v)} for k, v in files.items()}
    return rec, src


def items_stage(harvest: pathlib.Path, out: pathlib.Path) -> dict:
    from infra.pg.matrix_models import KNOWN
    rec, src = build(harvest, MK, SPEC_O2)
    assert rec["revision"] == KNOWN[REPO_O2]["sha"] and rec["n_layers"] == KNOWN[REPO_O2]["n_layers"], (rec["revision"], rec["n_layers"])
    assert rec["counts"]["planted"] == 56 and rec["counts"]["planted_by_tercile"] == [18, 20, 18] and rec["counts"]["singles"] == 30, rec["counts"]
    rec.update({"harvest_dir": str(harvest), "branch": rec["revision"]})
    rules = {"o2": "build_items_e1b.build's rule on the harvest step's bank: 56 pairs seed 4 (18/20/18 by layer thirds), 30 singles seed 5, 3 known poles, "
                   "200 background positions seed 0"}
    obj = {"models": {MK: rec}, "rules": rules, "corpus": CORPUS, "sources": src, "script": "infra/aws_pg/jobs/instruments_e6b.py --stage items"}
    body = json.dumps(obj, sort_keys=True).encode()
    out.mkdir(parents=True, exist_ok=True)
    (out / "e6b_items.json").write_bytes(body)
    print(MK, json.dumps(rec["counts"]), flush=True)
    print("INSTRUMENTS_E6B_ITEMS_OK", hashlib.sha256(body).hexdigest(), flush=True)
    return obj


# ------------------------------------------------------------------------------------------ e1b: E1b's job and T5 on OLMo-2
def members_of(bank: pathlib.Path, rec: dict, abar) -> list:
    """The distinct (layer, neuron) of set_J_members over the 56 pairs, with ā from the job: [[l, k, ā], ...] in (l, k) order."""
    mem = set()
    for p in rec["planted"]:
        z = np.load(bank / "pairs" / f"{p['unit']}.npz")
        mem |= {(int(l), int(k)) for l, k in zip(z["set_J_members_layer"], z["set_J_members_neuron"])}
    return [[l, k, float(abar[l, k])] for l, k in sorted(mem)]


def t5_stage(out: pathlib.Path, rec: dict, job, model, tok, corpus: dict, device: str, batch: int, extra: dict) -> dict:
    from infra.aws_pg.jobs import instruments_e5 as J5
    from infra.pg import neurons_harvest as NH
    (out / "t5").mkdir(parents=True, exist_ok=True)
    mem = members_of(job.bank_dir, rec, job.abar.cpu().numpy())
    m = {"stage": "e6b_t5", "items_sha256": job.a.items_sha256, "started_utc": NH.utc(), "pg_git_sha": os.environ.get("PG_GIT_SHA"), "status": "running",
         "rule": "per member |ā_k| ||gamma_post ⊙ W_down[:, k]|| s_l(x) / rms(h_l(x)) at the last token of E1b's 200 background windows (E5's T5 statistic); "
                 "divide by sqrt(hidden) for ||write|| / ||h||", "members": mem, "models": {}, "gates": {}, **extra}
    NH.dump(out / "t5" / "metrics.json", m)
    t0 = time.time()
    with open(out / "samples" / "e6b_t5.jsonl", "a") as fh:
        r = J5.t5_model(out, MK, {**rec, "members": mem}, model, tok, corpus, device, batch, fh)
    m["models"][MK] = {**r, "hidden": rec["hidden"], "seconds_total": round(time.time() - t0, 1)}
    m["gates"][f"t5_capture_{MK}"] = r["check"]
    m["status"] = "done"
    NH.dump(out / "t5" / "metrics.json", m)
    assert r["check"]["pass"] and r["check"]["failing_input_failed"], (MK, r["check"])
    return m


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["items", "e1b"])
    ap.add_argument("--harvest", default=None, help="items: the harvest step's out dir")
    ap.add_argument("--items", default=None, help="e1b: the items step's e6b_items.json")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--model-key", default=MK, choices=[MK])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--full-batch", type=int, default=8)
    return ap


def main():
    a = parser().parse_args()
    out = pathlib.Path(a.out)
    if a.stage == "items":
        items_stage(pathlib.Path(a.harvest), out)
        return
    import torch
    from infra.aws_pg.jobs import instruments_e1b as JE1B
    from infra.aws_pg.jobs import instruments_e5 as J5
    from infra.aws_pg.jobs.neurons_harvest import download, fetch, fetch_corpus
    from infra.pg import neurons_corpus as NC
    from infra.pg import neurons_harvest as NH
    from infra.pg import reuse_birth as RB
    from infra.pg.diff_harvest import hf_online_setup
    from infra.pg.matrix_models import KNOWN, vocab_sha256
    hf_online_setup()
    assert torch.cuda.is_available(), "no CUDA on this instance"
    items_path = fetch(a.items, "instruments_e6b_items.json")
    a.items_sha256 = sha256(items_path)
    rec = json.loads(pathlib.Path(items_path).read_text())["models"][MK]
    assert rec["repo"] == REPO_O2 and rec["revision"] == rec["branch"] == KNOWN[REPO_O2]["sha"] and rec["n_layers"] == KNOWN[REPO_O2]["n_layers"], \
        (rec["repo"], rec["revision"], KNOWN[REPO_O2]["sha"])
    t0 = time.time()
    local, sha, drec = download(rec["repo"], rec["branch"])
    assert sha == rec["revision"], (sha, rec["revision"])
    corpus = NC.load_tok(fetch_corpus(a.corpus, rec["tokname"]), rec["tokname"])
    assert corpus["tokens_sha256"] == rec["tokens_sha256"], "the corpus is not the harvest bank's"
    t1 = time.time()
    RB.memory_start("cuda")
    model, tok = NH.load_model(local, rec["nh_key"])
    model.requires_grad_(False)
    assert vocab_sha256(tok) == corpus["record"]["vocab_sha256"], "the model's vocabulary differs from the corpus tokenizer's"
    ns = argparse.Namespace(model_key=MK, items_sha256=a.items_sha256, harvest_window=rec["window"], batch=a.batch, full_batch=a.full_batch)
    job = JE1B.Job(out, ns, rec, model, tok, corpus, "cuda")
    env = J5.env()
    job.m.update({"download": drec, "corpus_s": round(t1 - t0, 1), "load_s": round(time.time() - t1, 1), "sha": sha, "e6_stage": "e1b", **env})
    job.write()
    job.run()
    job.m["timing_s"]["wall_after_load"] = round(time.time() - t1, 1)
    job.m["memory_max_allocated_gb"] = RB.memory_gb("cuda")
    job.write()
    t5 = t5_stage(out, rec, job, model, tok, corpus, "cuda", a.batch, env)
    del model, job
    torch.cuda.empty_cache()
    repo_dir = pathlib.Path(os.environ["HF_HUB_CACHE"]) / ("models--" + REPO_O2.replace("/", "--"))
    shutil.rmtree(repo_dir)
    assert not repo_dir.exists()
    print("INSTRUMENTS_E6B_E1B_OK", json.dumps({"sha": sha, "t5": t5["models"][MK]["summary"]}), flush=True)


if __name__ == "__main__":
    main()
