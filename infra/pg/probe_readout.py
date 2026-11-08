"""N3 — the P (probe) readout, spec §1.2 / §6.7.

A ridge-logistic classifier on the residual activation at the FINAL token (the readout position of
S), trained on MMLU test rows[10:] per subject (≈ 13k items; position-disjoint from both eval halves
by construction) to predict the correct option index, tested on each eval half. Decisive layer
⌊n/3⌋, ⌊2n/3⌋ reported. Controls: shuffle-label null (labels permuted, seed 1337, refit) and the
random-init lower bound (same architecture, decoder re-drawn N(0, per-tensor std), seed 0 — see
`reinit_decoder` for why this replaced `_init_weights` on 2026-09-03 and what is unchanged).
Verdict inputs go to `matrix_core.probe_recovered`.

Sequential loading (the behaviour job never holds both models): `probe_features(fwd, ...)` harvests
train + eval features for ONE resident model; `probe_column(feats_before, feats_after, ...)` fits and
scores from features alone. `random_init_bound` is destructive — run it last on the healthy model.

Estimator: multinomial logistic regression with an L2 penalty on the weights (intercept unpenalised),
objective 0.5·‖W‖²_F + C·Σ_i CE_i with C = 1.0 on standardised features, minimised by L-BFGS
(max_iter 2000, float64) — the objective of sklearn.linear_model.LogisticRegression(C=1.0,
max_iter=2000) after StandardScaler, implemented in torch so the dry-run is dependency-free and the
fit is deterministic on CPU. Features are harvested in batches through `Forward.final_residuals`
and stored float16.
"""
from __future__ import annotations


C_RIDGE = 1.0
MAX_ITER = 2000
SHUFFLE_SEED = 1337
RANDOM_INIT_SEED = 0
PROBE_TRAIN_ROW_START = 10       # MMLU test rows[10:] per subject


def probe_layers(n_layers: int) -> list:
    """Decisive layer ⌊n/3⌋ first, ⌊2n/3⌋ second (both 0-indexed post-block residuals)."""
    return [n_layers // 3, (2 * n_layers) // 3]


def mmlu_probe_train_items(n_per_subject: int | None = None, load_dataset=None) -> list:
    """MMLU test rows[10:] per subject in text_eval order (sorted subject, row order); item dicts
    {subject, question, choices, answer, hash}, keyed by `matrix_items.item_hash` (the one alignment
    key: full subject + question). Disjointness from the eval halves holds by POSITION (the halves are
    rows[:10]). These hashes are compared as a SEQUENCE between the two models, never used as a pairing
    key, so they carry no identity gate: MMLU test repeats 78 (subject, question) rows, all of them in
    rows[10:], i.e. inside this set and outside every scored one."""
    from collections import defaultdict
    from infra.pg import matrix_items as MI
    from infra.pg.text_eval import MMLU_REPO
    test = MI._rows(load_dataset, MMLU_REPO, "all", "test")
    by = defaultdict(list)
    for r in test:
        by[r["subject"]].append(r)
    items = []
    for subj in sorted(by):
        rows = by[subj][PROBE_TRAIN_ROW_START:]
        if n_per_subject:
            rows = rows[:n_per_subject]
        for r in rows:
            items.append({"subject": subj, "question": r["question"], "choices": list(r["choices"]),
                          "answer": int(r["answer"]), "hash": MI.item_hash(subj, r["question"])})
    assert items, "no probe-train items"
    return items


def mmlu_prompt(item, shots: int, dev_shots_fn) -> str:
    """The S prompt: subject header + k dev exemplars + the item ending at 'Answer:' (text_eval)."""
    from infra.pg.text_eval import _q_block, _subject_header
    prefix = _subject_header(item["subject"]) + "".join(
        _q_block(s["question"], s["choices"], s["answer"]) for s in dev_shots_fn(item["subject"], shots))
    return prefix + _q_block(item["question"], item["choices"])


def harvest(fwd, items, layers, shots: int, dev_shots_fn=None, batch: int | None = None,
            keep_prompts: bool = False) -> dict:
    """Final-token residuals at `layers` for every item, streamed through `fwd` in batches.
    Returns {"X": {L: (N, d) float16 tensor}, "y": (N,) int tensor, "hash": [..], "n": N}.
    `dev_shots_fn(subject, k)` defaults to matrix_items.mmlu_dev_shots.

    `keep_prompts` returns the prompts as well, for §12.11 sample capture. It is set on the EVAL sets
    only, never on the ≈13 k-item train harvest, so the memory cost is a few hundred prompts."""
    import torch
    if dev_shots_fn is None:
        from infra.pg.matrix_items import mmlu_dev_shots as dev_shots_fn
    layers = [int(L) for L in layers]
    prompts = [mmlu_prompt(it, shots, dev_shots_fn) for it in items]
    if batch is not None:
        fwd.batch_size = batch
    X = {L: [] for L in layers}
    step = max(fwd.batch_size * 8, 8)
    for s in range(0, len(prompts), step):
        res = fwd.final_residuals(prompts[s:s + step], layers)
        for L in layers:
            X[L].append(res[L])
    X = {L: torch.cat(v, 0) for L, v in X.items()}
    y = torch.tensor([int(it["answer"]) for it in items], dtype=torch.long)
    for L in layers:
        assert X[L].shape[0] == y.numel() and X[L].dtype == torch.float16, (L, X[L].shape, X[L].dtype)
    return {"X": X, "y": y, "hash": [it["hash"] for it in items], "n": int(y.numel()),
            "prompt": (prompts if keep_prompts else None)}


def fit_probe(X_train, y_train, C: float = C_RIDGE, max_iter: int = MAX_ITER, n_classes: int = 4) -> dict:
    """Standardise (train mean/std) then multinomial L2-logistic by L-BFGS in float64.
    Returns {"W": (d, K), "b": (K,), "mu", "sd", "C", "n_iter", "loss"}."""
    import torch
    X = torch.as_tensor(X_train).to(torch.float64)
    y = torch.as_tensor(y_train).long()
    assert X.ndim == 2 and X.shape[0] == y.numel() and y.min() >= 0 and y.max() < n_classes
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, unbiased=False, keepdim=True).clamp_min(1e-12)
    Z = (X - mu) / sd
    d = Z.shape[1]
    W = torch.zeros(d, n_classes, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, tolerance_grad=1e-6,
                            tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")
    state = {"n": 0, "loss": None}

    def closure():
        opt.zero_grad()
        logits = Z @ W + b
        loss = C * torch.nn.functional.cross_entropy(logits, y, reduction="sum") + 0.5 * (W * W).sum()
        loss.backward()
        state["n"] += 1
        state["loss"] = float(loss.detach())
        return loss

    opt.step(closure)
    return {"W": W.detach(), "b": b.detach(), "mu": mu, "sd": sd, "C": C,
            "n_iter": state["n"], "loss": state["loss"]}


def predict(probe, X):
    import torch
    Z = (torch.as_tensor(X).to(torch.float64) - probe["mu"]) / probe["sd"]
    return (Z @ probe["W"] + probe["b"]).argmax(-1)


def evaluate(probe, X, y, prompts=None) -> dict:
    """`prompts` turns on §12.11 sample capture. P had NO capture path at all — `infra/pg/probe_readout`
    contained the string "sample" zero times, and `5c_P` already has a real number banked with nothing
    behind it. P is a trained readout of the residual, so there is no top token over the vocabulary to
    report and the sample says so explicitly rather than leaving the field silently absent; what it does
    show is real: the actual prompt whose final-token residual was read, the probe's four class scores,
    the class it picked and the gold."""
    import torch
    y = torch.as_tensor(y).long()
    Z = (torch.as_tensor(X).to(torch.float64) - probe["mu"]) / probe["sd"]
    logits = Z @ probe["W"] + probe["b"]
    pred = logits.argmax(-1)
    per_item = (pred == y).long().tolist()
    out = {"acc": float(sum(per_item) / len(per_item)), "n": len(per_item),
           "per_item": per_item, "pred": pred.tolist()}
    if prompts is not None:
        from infra.pg.format_readouts import _samples
        assert len(prompts) == len(per_item), (len(prompts), len(per_item))
        out["samples"] = _samples(prompts, per_item, pred.tolist(), y.tolist(), logits.tolist(),
                                  labels=["A", "B", "C", "D"],
                                  extra_per_item={"readout_kind": ["ridge-logistic probe on the residual "
                                                                   "stream — no vocabulary top token exists"]
                                                                  * len(per_item)})
    return out


def shuffle_null(X_train, y_train, X_eval, y_eval, seed: int = SHUFFLE_SEED) -> dict:
    """Labels permuted (seed 1337), refit, evaluated — must land within 0.05 of chance (§6.7)."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    y = torch.as_tensor(y_train).long()
    perm = torch.randperm(y.numel(), generator=g)
    p = fit_probe(X_train, y[perm])
    r = evaluate(p, X_eval, y_eval)
    r["seed"] = int(seed)
    return r


def reinit_decoder(model, seed: int = RANDOM_INIT_SEED) -> dict:
    """Random-init lower bound: re-initialise the LM decoder (layers + final norm; embeddings/head
    untouched so the readout position and vocabulary stay defined) and PROVE every tensor changed.
    DESTRUCTIVE — call last. Returns the re-init record.

    THE DEFECT THIS REPLACES (2026-09-03). This function used to call `model._init_weights(mod)` on
    every module under the decoder. Under transformers 5.9.0 that call is a NO-OP on a
    `from_pretrained` model — the loaded weights carry HF's already-initialised flag, so the init
    entry point returns without touching them. It was handed 580 modules per run and changed
    nothing, and on all 60 banked wave-1 probe records the "random-init lower bound" was per-item
    IDENTICAL to the healthy before-probe: a control that could not fail. The mechanism above is
    inferred (transformers is absent from the control plane by design and was not stepped through
    here); the fix does not depend on it, because it no longer calls `_init_weights` and it verifies
    the change tensor by tensor.

    RULE — the house fair-scratch rule, identical to `matrix_recovery_core.reinit_decoder` /
    `diff_harvest.build_random_after`: every decoder parameter with ndim >= 2 is redrawn
    N(0, std of that same tensor) from one CPU generator seeded `seed`, in named-parameter order.
    1-D parameters (RMSNorm gains, biases) KEEP their values, and that is deliberate rather than an
    omission: the fresh value of a custom RMSNorm gain is architecture-dependent (Gemma normalises
    by `1 + w`, so fresh is 0; Llama/Qwen by `w`, so fresh is 1) and guessing it wrong yields a
    numerically dead network rather than an untrained one. Redrawing every attention/MLP weight is
    what makes the network untrained; the gains only rescale a stream that now carries no learned
    structure. NOTE for the reader of the spec: §6.7 words this control as "`_init_weights` re-init",
    which is the mechanism that did not work — the estimand (a linear probe read off an untrained
    decoder of the same architecture) is unchanged."""
    import torch
    from infra.pg.matrix_forward import lm_anatomy
    an = lm_anatomy(model)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    rec = {"seed": int(seed), "rule": "ndim>=2 -> N(0, std of the source tensor); 1-D kept (see docstring)",
           "n_random": 0, "n_kept": 0, "randomised": []}
    with torch.no_grad():
        for root, path in ((an["layers"], an["layers_path"]), (an["norm"], an["norm_path"])):
            for name, p in root.named_parameters():
                if p.ndim < 2:
                    rec["n_kept"] += 1
                    continue
                old = p.detach().clone()
                # `.cpu()` is load-bearing, not decoration. `g` is a CPU generator (the house rule
                # above), so `torch.randn(..., generator=g)` lands on CPU; without `.cpu()` here `std`
                # is a CUDA 0-dim tensor and the multiply crosses devices. That killed row 3b — the
                # Molmo install row, the program's only non-Google modern VLM — 73 minutes in on
                # 2026-09-04, after it had banked every column of BOTH healthy arms, with
                # "Expected all tensors to be on the same device, but found cuda:0 and cpu".
                # The two sibling implementations this function's docstring calls identical
                # (`matrix_recovery_core.reinit_decoder`, `diff_harvest.build_random_after`) both do
                # `v.detach().cpu().float()`; this one omitted the `.cpu()` and so was never actually
                # identical. It had only ever run on CPU stubs.
                std = p.detach().cpu().float().std().clamp_min(1e-8)
                p.copy_((torch.randn(p.shape, generator=g) * std).to(dtype=p.dtype, device=p.device))
                assert not torch.equal(old, p.detach()), f"re-init left {path}.{name} bit-identical"
                rec["n_random"] += 1
                rec["randomised"].append(f"{path}.{name}")
    assert rec["n_random"] > 0, f"decoder re-init touched no 2-D parameter under {an['layers_path']}"
    return rec


def probe_features(fwd, train_items, eval_sets, layers, shots, dev_shots_fn=None) -> dict:
    """Everything P needs from ONE resident model: {"train": harvest, "eval": {half: harvest}}."""
    layers = [int(L) for L in layers]
    return {"train": harvest(fwd, train_items, layers, shots, dev_shots_fn),
            "eval": {name: harvest(fwd, items, layers, shots, dev_shots_fn, keep_prompts=True)
                     for name, items in eval_sets.items()},
            "layers": layers, "shots": int(shots)}


def _acc_ci_lo(per_item, n_boot, seed=SHUFFLE_SEED) -> float:
    """2.5-percentile of the item-bootstrap mean accuracy (matrix_core.probe_recovered's `ci_lo_after`, D30)."""
    import numpy as np
    a = np.asarray(per_item, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(int(n_boot), a.size))
    return float(np.percentile(a[idx].mean(1), 2.5))


def _feature_shift(feats_healthy, feats_rand) -> dict:
    """Mean |Δ| of the train features per layer, healthy → re-initialised, over the SAME prompts, and
    the assertion that the re-init reached the readout position at all."""
    import torch
    assert feats_healthy["train"]["hash"] == feats_rand["train"]["hash"], "random-init harvest read other items"
    out = {}
    for L in feats_rand["layers"]:
        a = feats_healthy["train"]["X"][L].float()
        b = feats_rand["train"]["X"][L].float()
        out[str(L)] = {"mean_abs_delta": float((b - a).abs().mean()), "healthy_std": float(a.std())}
        assert not torch.equal(a, b), (
            f"random-init bound: layer {L} residuals are bit-identical to the healthy model's — the "
            "decoder re-init did not reach the readout position (the 2026-09-03 defect)")
    return out


def random_init_bound(fwd, train_items, eval_sets, layers, shots, dev_shots_fn=None, *, feats_healthy) -> dict:
    """Re-init `fwd.model`'s decoder (seed 0), re-harvest, refit, evaluate on each eval set.
    eval_sets: {name: items}. Returns {name: {L: evaluate-record}} plus the harvest sizes.

    `feats_healthy` is THIS model's `probe_features` from before the re-init, and it is required, not
    optional: the whole worth of this control is that the features moved, and on 2026-09-03 sixty
    banked records said they had not. The delta is asserted and banked. It is asserted on the
    FEATURES rather than on the probe's accuracy because two probes can agree item-for-item on
    features that are entirely different — the stub in `_dryrun` does exactly that at 3 of its 4
    (half, layer) cells — so an accuracy-difference assertion would fire on correct code."""
    rei = reinit_decoder(fwd.model)
    f = probe_features(fwd, train_items, eval_sets, layers, shots, dev_shots_fn)
    out = {"reinit": rei, "n_reinit_params": rei["n_random"], "eval": {},
           "feature_shift": _feature_shift(feats_healthy, f)}
    for name in eval_sets:
        out["eval"][name] = {}
        for L in f["layers"]:
            p = fit_probe(f["train"]["X"][L], f["train"]["y"])
            out["eval"][name][str(L)] = evaluate(p, f["eval"][name]["X"][L], f["eval"][name]["y"])
    return out


def probe_column(feats_before, feats_after, n_boot: int = 2000, save_dir=None) -> dict:
    """The full P column for one row from harvested features: fit per layer on `before` AND on `after`
    (each model's own probe — P is a trained readout of THAT model's residual), evaluate on every eval
    half, shuffle-null on the before-probe, item-bootstrap CI-low of the after-probe accuracy, and return
    `matrix_core.probe_recovered` inputs per (half, layer). The random-init bound is run separately
    (destructive) via `random_init_bound`."""
    import torch
    layers = feats_before["layers"]
    assert layers == feats_after["layers"] and feats_before["shots"] == feats_after["shots"]
    assert feats_before["train"]["hash"] == feats_after["train"]["hash"], "probe train items differ"
    tr = {"before": feats_before["train"], "after": feats_after["train"]}
    out = {"layers": layers, "shots": feats_before["shots"], "n_train": tr["before"]["n"], "halves": {}}
    probes = {}
    for side in ("before", "after"):
        for L in layers:
            probes[(side, L)] = fit_probe(tr[side]["X"][L], tr[side]["y"])
    for half in feats_before["eval"]:
        ev = {"before": feats_before["eval"][half], "after": feats_after["eval"][half]}
        assert ev["before"]["hash"] == ev["after"]["hash"], f"{half}: eval items differ"
        rec = {}
        for L in layers:
            eb = evaluate(probes[("before", L)], ev["before"]["X"][L], ev["before"]["y"])
            # §12.11: the AFTER probe is the arm every P verdict reads (`acc_after_probe`), so it is the
            # arm that carries the samples.
            ea = evaluate(probes[("after", L)], ev["after"]["X"][L], ev["after"]["y"],
                          prompts=ev["after"].get("prompt"))
            sh = shuffle_null(tr["before"]["X"][L], tr["before"]["y"], ev["before"]["X"][L], ev["before"]["y"])
            rec[str(L)] = {"before": eb, "after": ea, "shuffle": sh,
                           "inputs": {"acc_after_probe": ea["acc"], "acc_before_probe": eb["acc"],
                                      "acc_shuffle": sh["acc"], "ci_lo_after": _acc_ci_lo(ea["per_item"], n_boot)}}
        out["halves"][half] = rec
    if save_dir is not None:
        import pathlib
        p = pathlib.Path(save_dir); p.mkdir(parents=True, exist_ok=True)
        torch.save({f"{s}_L{L}": {k: v for k, v in pr.items()} for (s, L), pr in probes.items()},
                   p / "probe_weights.pt")
        torch.save({s: {"X": tr[s]["X"], "y": tr[s]["y"], "hash": tr[s]["hash"]} for s in tr},
                   p / "probe_train_features_fp16.pt")
    return out


def finalize_with_random_init(record: dict, rand: dict, chance: float = 0.25) -> dict:
    """Attach the random-init lower bound and compute matrix_core.probe_recovered per (half, layer).

    Each record also carries `random_init_identical_to_before`: the 2026-09-03 defect made that True on
    all 60 banked cells, so a reader (and `matrix_verdicts`) can see the degenerate case in the
    artifact. It is banked and not asserted — the hard checks are on the weights (`reinit_decoder`)
    and on the features (`random_init_bound`), because two probes CAN agree item-for-item on entirely
    different features and an assertion here would fire on correct code."""
    from infra.pg import matrix_core as MC
    record["random_init_feature_shift"] = rand.get("feature_shift")
    record["random_init_reinit"] = {k: v for k, v in (rand.get("reinit") or {}).items() if k != "randomised"}
    for half, rec in record["halves"].items():
        for L, r in rec.items():
            rb = rand["eval"][half][L]
            r["random_init"] = rb
            r["random_init_identical_to_before"] = bool(rb["pred"] == r["before"]["pred"])
            i = r["inputs"]
            r["verdict"] = MC.probe_recovered(i["acc_after_probe"], i["acc_before_probe"], i["acc_shuffle"],
                                              rb["acc"], i["ci_lo_after"], chance=chance)
    record["decisive_layer"] = record["layers"][0]
    return record


def _dryrun():
    import torch
    from infra.pg.matrix_forward import TextForward, _StubTok, stub_text_model
    torch.manual_seed(0)
    # synthetic activations: class signal in a 2-D subspace of d=16, N=600 train / 200 eval
    d, N, K = 16, 600, 4
    centers = torch.randn(K, d, dtype=torch.float64) * 3
    ytr = torch.randint(0, K, (N,)); Xtr = (centers[ytr] + torch.randn(N, d, dtype=torch.float64)).half()
    yev = torch.randint(0, K, (200,)); Xev = (centers[yev] + torch.randn(200, d, dtype=torch.float64)).half()
    p = fit_probe(Xtr, ytr)
    assert p["W"].shape == (d, K) and p["n_iter"] <= MAX_ITER
    ev = evaluate(p, Xev, yev)
    assert ev["acc"] > 0.9, ev["acc"]
    sh = shuffle_null(Xtr, ytr, Xev, yev)
    assert abs(sh["acc"] - 0.25) < 0.12, sh["acc"]
    # determinism
    p2 = fit_probe(Xtr, ytr)
    assert torch.allclose(p["W"], p2["W"]) and p["loss"] == p2["loss"]
    # end-to-end through the stub Forward (harvest → fit → evaluate → reinit → bound)
    items = [{"subject": "s", "question": f"q{i} {'ab'*(i%3)}", "choices": ["w", "x", "y", "z"],
              "answer": i % 4, "hash": f"h{i}"} for i in range(40)]
    shots_fn = lambda subject, k: [{"question": "dq", "choices": ["a", "b", "c", "d"], "answer": 0}] * k
    fb = TextForward(stub_text_model(seed=1), _StubTok(), device="cpu")
    fa = TextForward(stub_text_model(seed=2), _StubTok(), device="cpu")
    layers = probe_layers(fb.n_layers)
    assert layers == [1, 2]
    halves = {"discovery": items[:20], "confirmation": items[20:]}
    feats_b = probe_features(fb, items, halves, layers, 1, shots_fn)       # sequential: before first ...
    feats_a = probe_features(fa, items, halves, layers, 1, shots_fn)       # ... then after, from features
    rec = probe_column(feats_b, feats_a, n_boot=50)
    assert set(rec["halves"]) == {"discovery", "confirmation"} and rec["n_train"] == 40
    r = rec["halves"]["discovery"]["1"]
    assert set(r["inputs"]) == {"acc_after_probe", "acc_before_probe", "acc_shuffle", "ci_lo_after"}
    assert r["inputs"]["ci_lo_after"] <= r["inputs"]["acc_after_probe"]
    assert len(r["before"]["per_item"]) == 20
    # NEGATIVE CONTROL for the control (2026-09-03): a re-init that does nothing must RAISE, not bank.
    # This is the exact wave-1 failure — the bound re-reporting the healthy probe on 60 records — and
    # before today nothing in this file or in the behaviour job could tell the two apart.
    g_mod = globals()
    real_reinit = g_mod["reinit_decoder"]
    g_mod["reinit_decoder"] = lambda model, seed=RANDOM_INIT_SEED: {"n_random": 0, "no_op": True}
    try:
        random_init_bound(fb, items, halves, layers, 1, shots_fn, feats_healthy=feats_b)  # fb untouched
        raise SystemExit("FAIL: a no-op decoder re-init was accepted as a lower bound")
    except AssertionError as e:
        assert "bit-identical to the healthy model" in str(e), e
    finally:
        g_mod["reinit_decoder"] = real_reinit
    w0 = fb.model.model.layers[1].lin.weight.clone()
    rand = random_init_bound(fb, items, halves, layers, 1, shots_fn, feats_healthy=feats_b)
    assert rand["n_reinit_params"] == fb.n_layers and rand["reinit"]["n_kept"] >= fb.n_layers   # Linear w vs bias+LN
    assert not torch.equal(w0, fb.model.model.layers[1].lin.weight)
    assert all(v["mean_abs_delta"] > 0.1 * v["healthy_std"] for v in rand["feature_shift"].values()), rand["feature_shift"]
    fin = finalize_with_random_init(rec, rand)
    # the degenerate case is BANKED, not asserted: this stub's probe agrees item-for-item at 3 of the 4
    # (half, layer) cells on features that moved by ~0.8 of a standard deviation.
    assert isinstance(fin["halves"]["discovery"]["1"]["random_init_identical_to_before"], bool)
    assert fin["random_init_feature_shift"] and fin["random_init_reinit"]["n_random"] == fb.n_layers
    v = fin["halves"]["confirmation"]["1"]["verdict"]
    assert {"P_recovered", "void", "verdict"} <= set(v) and fin["decisive_layer"] == 1     # VOID-probe is a legal landing
    assert v["void"] or {"lower_bound", "linearly_present", "P_ci_lo"} <= set(v), v
    print(f"PASS probe_readout dry-run (synthetic acc={ev['acc']:.3f} shuffle={sh['acc']:.3f} "
          f"iters={p['n_iter']}; stub end-to-end incl. random-init bound)")


if __name__ == "__main__":
    _dryrun()
