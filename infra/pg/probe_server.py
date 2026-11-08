"""Probe server: load Q4B, G4 and O7 once on one GPU, then answer probe requests from a directory.

    python -m infra.pg.probe_server --requests DIR --responses DIR [--models Q4B,G4,O7] [--weights Q4B=PATH ...]

Models (studies/neuronshow/plan/plan.json -> models): Q4B Qwen/Qwen3-4B-Base, G4 google/gemma-3-4b-pt, O7
allenai/Olmo-3-1025-7B (branch stage1-step1413814), each asserted at its pinned commit. Without --weights a model is
downloaded from the Hub (HF_TOKEN from the environment; G4 is gated); --weights M=PATH takes a local snapshot directory
or an HF hub cache root. Weights bf16, SDPA attention; every residual add and readout is done in fp32 (neuron_rw_core).
Doses are in residual-RMS units of the pole's layer (NEURONRW Arm 0 rms.json, copied into plan.json): the added vector is
dose * RMS_layer * w, w = sign * W_down[:, i] / ||W_down[:, i]||.

Loop (the probe station's protocol, infra/vast/cells/probe_station.py): every --requests/<rid>.json without a
--responses/<rid>.json is answered, oldest first, into --responses/<rid>.json (written to a temp name, then renamed);
requests are never moved or deleted; rid must match [A-Za-z0-9_.-]+ (other file names are ignored). A request that fails (bad JSON, unknown op or field, bad pole, any exception) gets a response with
"ok": false, the error and the traceback; the loop goes on. At startup the server writes --responses/startup.json: per
model the identity check (dose 0 through the push hook and the NEURONOFF ablator in identity mode reproduce the unhooked
logits exactly; failing input: dose 1 must differ). If any identity check fails, startup.json says so and the server exits.

Poles are written as NEURONRW names them: "L07N10202-" (layer 7, neuron 10202, negative pole). Positions are absolute
token indices of BOS + prompt + generated tokens (BOS is index 0 for G4 and O7; Q4B has no BOS); P = prompt length with
BOS. Requests (unknown fields are rejected):

  steer   {"op": "steer", "model", "pushes": [{"pole", "dose"}, ...], "prompt",
           "n": 4, "max_new_tokens": 80, "temperature": 0.7, "seed": 20261025,
           "positions": "gen" | "all" | "first" | "last" | [int, ...], "layer_override": null | int}
          positions: "gen" (default; P - 1 and every generated token, NEURONRW's instrument), "all" (every prompt
          token and every generated token), "first" (the first non-BOS prompt token only), "last" (P - 1 only), or a
          list of absolute indices (indices >= P land on generated tokens). Several pushes at one layer add up.
          layer_override: add every push's vector (sized by its own layer's RMS) at this layer instead.
          dose may be negative. Returns samples [{text, token_ids, n_new_tokens, stopped_eos, bigram_ratio,
          top5_step1}], top5_prefill (the first generated token's distribution, shared by all samples), the resolved
          pushes, positions_pushed_prefill and n_decode_steps_pushed. top5 = softmax of the raw logits (temperature 1).
  ablate  {"op": "ablate", "model", "poles": [pole, ...], "prompt", "n", "max_new_tokens", "temperature", "seed",
           "exclude_first": null | int}
          neuron_off_core.PoleAblator(mode "pole"): a_i set to 0 where sign * a_i > 0, at every position >=
          exclude_first (default: the BOS length, so every prompt and generated token). Same generation outputs as
          steer, plus per pole the number of (row, position) pairs zeroed.
  read    {"op": "read", "model", "pole", "text"}
          the neuron's activation a_i (the down_proj input coordinate) at every token of BOS + text, with the token
          strings, signed = sign * a_i (the pole fires where signed > 0), and the maximum.
  vocab   {"op": "vocab", "model", "pole", "k": 20}
          the Arm 0 projection W_U (gain_final * w) over the full vocabulary: top-k promoted and suppressed tokens, the
          column norm ||W_down[:, i]||, and the banked attested top 20 when the pole is in plan.json.
  lens    {"op": "lens", "model", "pole", "dose", "prompt", "positions": "gen"}
          one forward of BOS + prompt; at the last prompt position, the decoder output of every layer from the pole's
          layer to the last read through the final norm and the unembedding (fp32), top 5 with and without the push;
          plus the model's own top 5 with and without the push.
  nll     {"op": "nll", "model", "text", "pushes": [] , "positions": "all", "ablate": []}
          teacher-forced NLL (nats, fp32 log-softmax) of every token of the text given its prefix, unhooked and with
          the pushes and/or ablations active; per token: string, nll_clean, nll_hooked, delta.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import pathlib
import re
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from infra.aws_pg.jobs.neuron_off import finite  # noqa: E402
from infra.pg import neuron_off_core as C  # noqa: E402
from infra.pg import neuron_rw_core as R  # noqa: E402
from infra.pg import neuronshow_plan as P  # noqa: E402

ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
GEN_DEFAULTS = {"n": 4, "max_new_tokens": 80, "temperature": 0.7, "seed": 20261025}
FIELDS = {   # op -> (required, optional with defaults)
    "steer": (("model", "pushes", "prompt"), {**GEN_DEFAULTS, "positions": "gen", "layer_override": None}),
    "ablate": (("model", "poles", "prompt"), {**GEN_DEFAULTS, "exclude_first": None}),
    "read": (("model", "pole", "text"), {}),
    "vocab": (("model", "pole"), {"k": 20}),
    "lens": (("model", "pole", "dose", "prompt"), {"positions": "gen"}),
    "nll": (("model", "text"), {"pushes": [], "positions": "all", "ablate": []}),
}
MAX_TEXT_TOKENS = 2048
IDENTITY_PROMPT = "Q: Why is the sky blue?\nA:"


def parse_pole(p: str) -> tuple:
    m = re.fullmatch(r"L(\d+)N(\d+)([+-])", p)
    assert m, f"bad pole id {p!r} (expected like L07N10202-)"
    return int(m[1]), int(m[2]), 1 if m[3] == "+" else -1


def top5(logits: torch.Tensor, tok) -> list:
    """Top 5 of one logit vector: token string, id, logit, softmax probability (fp32, temperature 1)."""
    lf = logits.float()
    p = F.softmax(lf, -1)
    v, i = lf.topk(5)
    return [{"str": tok.decode([int(j)]), "id": int(j), "logit": float(x), "p": float(p[j])} for x, j in zip(v, i)]


class Push:
    """Forward hooks adding fp32 vectors to decoder-layer outputs at chosen absolute positions, every forward of one
    `with` block (prefill, then one position per decode step). pushes: [(layer, v fp32 [d])]; positions: a predicate
    on an absolute position tensor. Records the positions pushed in the first forward and the number of later forwards
    with at least one position pushed."""

    def __init__(self, model, pushes: list, positions):
        self.model, self.positions = model, positions
        self.delta = {}
        for layer, v in pushes:
            self.delta[layer] = self.delta.get(layer, 0) + v.float().to(model.device)
        self.handles, self.offset, self.first, self.n_later = [], {}, None, 0

    def __enter__(self):
        tower = R.text_tower(self.model)
        self.offset = {l: 0 for l in self.delta}
        self.handles = [tower.layers[l].register_forward_hook(lambda m, a, o, l=l: self._hook(l, o)) for l in self.delta]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _hook(self, l, out):
        h = R.hidden_of(out)
        T = h.shape[1]
        absolute = self.offset[l] + torch.arange(T, device=h.device)
        mask = self.positions(absolute)
        hf = h.to(torch.float32, copy=True)
        hf[:, mask] += self.delta[l]
        if l == min(self.delta):
            if self.offset[l] == 0:
                self.first = absolute[mask].tolist()
            elif bool(mask.any()):
                self.n_later += 1
        self.offset[l] += T
        return R.with_hidden(out, hf.to(h.dtype))


def position_rule(spec, P_len: int, n_bos: int):
    if spec == "gen":
        return lambda a: a >= P_len - 1
    if spec == "all":
        return lambda a: torch.ones_like(a, dtype=torch.bool)
    if spec == "first":
        return lambda a: a == n_bos
    if spec == "last":
        return lambda a: a == P_len - 1
    assert isinstance(spec, list) and spec and all(isinstance(x, int) and x >= 0 for x in spec), \
        f"positions must be gen, all, first, last or a non-empty list of non-negative ints, got {spec!r}"
    s = torch.tensor(sorted(set(spec)))
    return lambda a: torch.isin(a, s.to(a.device))


class ProbeServer:
    def __init__(self, models: dict, plan: dict | None = None, identity_poles: dict | None = None):
        """models: {key: (model, tok, rms list per layer)}; identity_poles: {key: pole} for the startup check (default:
        the model's first NEURONSHOW pole in plan.json)."""
        self.models = models
        self.plan = plan or P.load()
        self.banked = {(p["model_key"], p["pole"]): p for p in self.plan["poles"]}
        self.identity_poles = identity_poles or {k: P.model_poles(self.plan, k)[0]["pole"] for k in models}

    # ------------------------------------------------------------------ helpers
    def get(self, key: str) -> tuple:
        assert key in self.models, f"model {key!r} not loaded (loaded: {sorted(self.models)})"
        return self.models[key]

    def direction(self, key: str, pole: str) -> tuple:
        model, _, rms = self.get(key)
        layer, neuron, sign = parse_pole(pole)
        n_layers = len(R.text_tower(model).layers)
        assert layer < n_layers, f"{pole}: layer {layer} >= {n_layers}"
        return layer, neuron, sign, R.pole_direction(model, layer, neuron, sign), rms[layer]

    def pushes(self, key: str, pushes: list, layer_override=None) -> tuple:
        model = self.get(key)[0]
        out, resolved = [], []
        assert isinstance(pushes, list), "pushes must be a list of {pole, dose}"
        for q in pushes:
            assert isinstance(q, dict) and set(q) == {"pole", "dose"}, f"each push is {{pole, dose}}, got {q!r}"
            layer, neuron, sign, w, rms = self.direction(key, q["pole"])
            dose = float(q["dose"])
            assert math.isfinite(dose), f"dose {q['dose']!r}"
            at = layer if layer_override is None else int(layer_override)
            assert 0 <= at < len(R.text_tower(model).layers), f"layer_override {layer_override} out of range"
            out.append((at, dose * rms * w))
            resolved.append({"pole": q["pole"], "dose": dose, "layer": layer, "layer_injected": at, "rms": rms,
                             "alpha_abs": dose * rms})
        return out, resolved

    def ids(self, key: str, text: str) -> list:
        tok = self.get(key)[1]
        ids = R.prompt_ids(tok, text)
        assert 0 < len(ids) <= MAX_TEXT_TOKENS, f"{len(ids)} tokens (limit {MAX_TEXT_TOKENS})"
        return ids

    def generate(self, key: str, prompt: str, hooks: list, r: dict) -> dict:
        model, tok, _ = self.get(key)
        n, mnt, temp, seed = int(r["n"]), int(r["max_new_tokens"]), float(r["temperature"]), int(r["seed"])
        assert 1 <= n <= 64 and 1 <= mnt <= 512 and temp > 0, f"n {n}, max_new_tokens {mnt}, temperature {temp}"
        p = self.ids(key, prompt)
        x = torch.tensor([p], device=model.device).repeat(n, 1)
        cfg = C.generation_config(tok, temp, mnt, 0, 1.0)
        torch.manual_seed(seed)
        with contextlib.ExitStack() as st:
            for h in hooks:
                st.enter_context(h)
            with torch.no_grad():
                out = model.generate(x, attention_mask=torch.ones_like(x), generation_config=cfg,
                                     return_dict_in_generate=True, output_logits=True)
        eos, samples = tok.eos_token_id, []
        logits = out.logits
        for b, o in enumerate(out.sequences[:, len(p):].tolist()):
            stop = eos in o
            new = o[:o.index(eos)] if stop else o
            text = tok.decode(new, skip_special_tokens=True)
            samples.append({"text": text, "token_ids": new, "n_new_tokens": len(new), "stopped_eos": stop,
                            "bigram_ratio": P.distinct_bigram_ratio(text),
                            "top5_step1": top5(logits[1][b], tok) if len(logits) > 1 and len(new) >= 1 else None})
        return {"prompt_tokens": len(p), "top5_prefill": top5(logits[0][0], tok), "samples": samples}

    # ------------------------------------------------------------------ operations
    def op_steer(self, r: dict) -> dict:
        key = r["model"]
        model, tok, _ = self.get(key)
        vecs, resolved = self.pushes(key, r["pushes"], r["layer_override"])
        assert vecs, "steer needs at least one push"
        p = self.ids(key, r["prompt"])
        hook = Push(model, vecs, position_rule(r["positions"], len(p), len(C.bos_prefix(tok))))
        g = self.generate(key, r["prompt"], [hook], r)
        return {**g, "pushes": resolved, "positions": r["positions"], "positions_pushed_prefill": hook.first,
                "n_decode_steps_pushed": hook.n_later}

    def ablators(self, key: str, poles: list, exclude_first) -> tuple:
        model, tok, _ = self.get(key)
        assert isinstance(poles, list), "poles must be a list of pole ids"
        ex = len(C.bos_prefix(tok)) if exclude_first is None else int(exclude_first)
        abl = []
        for pole in poles:
            layer, neuron, sign = parse_pole(pole)
            abl.append(C.PoleAblator(model, "pole", layer, neuron, "+" if sign == 1 else "-", exclude_first=ex))
        return abl, ex

    def op_ablate(self, r: dict) -> dict:
        abl, ex = self.ablators(r["model"], r["poles"], r["exclude_first"])
        assert abl, "ablate needs at least one pole"
        g = self.generate(r["model"], r["prompt"], abl, r)
        return {**g, "exclude_first": ex,
                "zeroed": [{"pole": p, "n_row_positions_zeroed": a.n_edited} for p, a in zip(r["poles"], abl)]}

    def op_read(self, r: dict) -> dict:
        key = r["model"]
        model, tok, _ = self.get(key)
        layer, neuron, sign = parse_pole(r["pole"])
        ids = self.ids(key, r["text"])
        down = R.text_tower(model).layers[layer].mlp.down_proj
        assert neuron < down.in_features, f"neuron {neuron} >= {down.in_features}"
        cap = {}
        h = down.register_forward_pre_hook(lambda m, a: cap.__setitem__("a", a[0][0, :, neuron].float().cpu()))
        try:
            with torch.no_grad():
                t = torch.tensor([ids], device=model.device)
                model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False, logits_to_keep=1)
        finally:
            h.remove()
        a = cap["a"].tolist()
        rows = [{"i": i, "id": int(t_), "str": tok.decode([int(t_)]), "a": x, "signed": sign * x}
                for i, (t_, x) in enumerate(zip(ids, a))]
        j = max(range(len(rows)), key=lambda i: rows[i]["signed"])
        return {"layer": layer, "neuron": neuron, "sign": sign, "tokens": rows,
                "max_signed": {"i": j, "str": rows[j]["str"], "signed": rows[j]["signed"]}}

    def op_vocab(self, r: dict) -> dict:
        key = r["model"]
        model, tok, _ = self.get(key)
        layer, neuron, sign = parse_pole(r["pole"])
        k = int(r["k"])
        assert 1 <= k <= 200, f"k {k}"
        allv = torch.arange(R.unembedding(model).shape[0])
        pr = R.vocab_projection(model, tok, [(r["pole"], layer, neuron, sign)], allv, k=k)[r["pole"]]
        b = self.banked.get((key, r["pole"]))
        return {"layer": layer, "neuron": neuron, "sign": sign, "column_norm": pr["w_down_norm"],
                "promoted": pr["top_full"], "suppressed": pr["bottom_full"], "vocabulary": "full",
                "banked_top20_attested": None if b is None else b["top20"]}

    def lens_logits(self, model, h: torch.Tensor) -> torch.Tensor:
        """Final norm (fp32, the model's gain and epsilon) and unembedding of residual vectors h [K, d]."""
        norm = R.text_tower(model).norm
        x = h.float() * R.inv_rms(h, R.norm_eps(norm))[..., None] * R.norm_gain_exact(norm)
        return R.softcap(model, R.unembed(model, x))

    def op_lens(self, r: dict) -> dict:
        key = r["model"]
        model, tok, _ = self.get(key)
        vecs, resolved = self.pushes(key, [{"pole": r["pole"], "dose": r["dose"]}])
        layer = resolved[0]["layer"]
        ids = self.ids(key, r["prompt"])
        tower = R.text_tower(model)
        t = torch.tensor([ids], device=model.device)

        def run(hook):
            cap = {}
            hs = []
            with hook if hook is not None else contextlib.nullcontext():
                for l in range(layer, len(tower.layers)):
                    hs.append(tower.layers[l].register_forward_hook(
                        lambda m, a, o, l=l: cap.__setitem__(l, R.hidden_of(o)[0, -1].float().clone())))
                try:
                    with torch.no_grad():
                        logits = model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False).logits[0, -1]
                finally:
                    for h in hs:
                        h.remove()
            return cap, logits
        clean, lc = run(None)
        hook = Push(model, vecs, position_rule(r["positions"], len(ids), len(C.bos_prefix(tok))))
        pushed, lp = run(hook)
        rows = []
        for l in range(layer, len(tower.layers)):
            lg = self.lens_logits(model, torch.stack([clean[l], pushed[l]]))
            rows.append({"layer": l, "top5_clean": top5(lg[0], tok), "top5_pushed": top5(lg[1], tok)})
        return {"pushes": resolved, "position": len(ids) - 1, "positions_pushed": hook.first, "layers": rows,
                "model_top5_clean": top5(lc, tok), "model_top5_pushed": top5(lp, tok)}

    def op_nll(self, r: dict) -> dict:
        key = r["model"]
        model, tok, _ = self.get(key)
        ids = self.ids(key, r["text"])
        assert len(ids) >= 2, "nll needs at least two tokens"
        vecs, resolved = self.pushes(key, r["pushes"])
        abl, ex = self.ablators(key, r["ablate"], None)
        hooks = abl + ([Push(model, vecs, position_rule(r["positions"], len(ids), len(C.bos_prefix(tok))))] if vecs else [])
        t = torch.tensor([ids], device=model.device)

        def nll(hs):
            with contextlib.ExitStack() as st:
                for h in hs:
                    st.enter_context(h)
                with torch.no_grad():
                    lg = model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False).logits[0, :-1]
            return (-F.log_softmax(lg.float(), -1).gather(-1, t[0, 1:, None])[:, 0]).cpu()
        clean = nll([])
        hooked = nll(hooks) if hooks else clean
        start = len(C.bos_prefix(tok))
        rows = [{"i": i, "str": tok.decode([ids[i]]), "nll_clean": float(clean[i - 1]), "nll_hooked": float(hooked[i - 1]),
                 "delta": float(hooked[i - 1] - clean[i - 1])} for i in range(max(1, start), len(ids))]
        mean = lambda k: sum(x[k] for x in rows) / len(rows)  # noqa: E731
        return {"pushes": resolved, "ablate": r["ablate"], "positions": r["positions"] if vecs else None,
                "exclude_first": ex if abl else None, "tokens": rows, "mean_nll_clean": mean("nll_clean"),
                "mean_nll_hooked": mean("nll_hooked"), "mean_delta": mean("delta")}

    # ------------------------------------------------------------------ requests
    def handle(self, req) -> dict:
        assert isinstance(req, dict), "a request is a JSON object"
        op = req.get("op")
        assert op in FIELDS, f"unknown op {op!r} (ops: {sorted(FIELDS)})"
        need, opt = FIELDS[op]
        extra = set(req) - set(need) - set(opt) - {"op", "id"}
        assert not extra, f"unknown fields for {op}: {sorted(extra)}"
        missing = [k for k in need if k not in req]
        assert not missing, f"missing fields for {op}: {missing}"
        r = {**opt, **req}
        return getattr(self, f"op_{op}")(r)

    def respond(self, f: pathlib.Path, resp_dir: pathlib.Path):
        """Answer requests/<rid>.json into responses/<rid>.json (the station's protocol: same name, never delete)."""
        rid, t0, req = f.stem, time.time(), None
        try:
            req = json.loads(f.read_text())
            out = {"id": rid, "ok": True, "op": req["op"], "request": req, "result": self.handle(req)}
        except Exception as e:                                   # one request fails loud; the loop goes on
            out = {"id": rid, "ok": False, "request": req, "error": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc()}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        out["elapsed_s"] = round(time.time() - t0, 3)
        out["answered_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(resp_dir / f.name, out)
        return out

    def serve_once(self, req_dir: pathlib.Path, resp_dir: pathlib.Path) -> int:
        """Answer every request that has no response yet, oldest first; returns how many were answered."""
        files = [f for f in req_dir.glob("*.json") if ID_RE.fullmatch(f.stem) and not (resp_dir / f.name).exists()]
        for f in sorted(files, key=lambda p: (p.stat().st_mtime, p.name)):
            self.respond(f, resp_dir)
        return len(files)

    def serve(self, req_dir, resp_dir, poll: float = 0.5, stop=None):
        req_dir, resp_dir = pathlib.Path(req_dir), pathlib.Path(resp_dir)
        req_dir.mkdir(parents=True, exist_ok=True)
        resp_dir.mkdir(parents=True, exist_ok=True)
        while stop is None or not stop.is_set():
            if self.serve_once(req_dir, resp_dir) == 0:
                time.sleep(poll)

    # ------------------------------------------------------------------ startup
    def identity(self, key: str) -> dict:
        model, tok, rms = self.get(key)
        ids = R.prompt_ids(tok, IDENTITY_PROMPT)
        t = torch.tensor([ids], device=model.device)
        pole = self.identity_poles[key]
        layer, neuron, sign, w, r_ = self.direction(key, pole)

        def logits(hook):
            with hook if hook is not None else contextlib.nullcontext():
                with torch.no_grad():
                    return model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False).logits.float()
        base = logits(None)
        everywhere = lambda a: torch.ones_like(a, dtype=torch.bool)  # noqa: E731
        d0 = float((logits(Push(model, [(layer, 0.0 * r_ * w)], everywhere)) - base).abs().max())
        d1 = float((logits(Push(model, [(layer, 1.0 * r_ * w)], everywhere)) - base).abs().max())
        da = float((logits(C.PoleAblator(model, "identity", layer, neuron, "+" if sign == 1 else "-", 0)) - base).abs().max())
        out = {"prompt": IDENTITY_PROMPT, "pole": pole, "push_dose0_max_abs_diff": d0, "ablator_identity_max_abs_diff": da,
               "failing_input_dose1_max_abs_diff": d1}
        out["pass"] = d0 == 0.0 and da == 0.0 and d1 > 0.0
        return out

    def startup(self, resp_dir, meta: dict | None = None) -> dict:
        resp_dir = pathlib.Path(resp_dir)
        resp_dir.mkdir(parents=True, exist_ok=True)
        rep = {"models": {}, "meta": meta or {}, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        for key, (model, tok, rms) in self.models.items():
            tower = R.text_tower(model)
            rep["models"][key] = {"n_layers": len(tower.layers), "d_model": R.d_model(model),
                                  "d_mlp": tower.layers[0].mlp.down_proj.weight.shape[1], "dtype": str(model.dtype),
                                  "device": str(model.device), "bos_id": tok.bos_token_id, "identity": self.identity(key)}
        rep["pass"] = all(m["identity"]["pass"] for m in rep["models"].values())
        write_json(resp_dir / "startup.json", rep)
        return rep


def write_json(path: pathlib.Path, obj):
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(finite(obj), ensure_ascii=False, allow_nan=False, default=str))   # NaN, inf -> null
    os.replace(tmp, path)


def load(keys: list, weights: dict) -> dict:
    from infra.aws_pg.jobs.neuronshow import resolve_weights
    from infra.pg import neurons_harvest as NH
    plan, models, meta = P.load(), {}, {}
    for key in keys:
        spec = plan["models"][key]
        local, sha, rec = resolve_weights(spec, weights.get(key))
        assert sha == spec["revision_expected"], f"{spec['repo']}: snapshot {sha}, plan expects {spec['revision_expected']}"
        t0 = time.time()
        model, tok = NH.load_model(local, key)
        assert len(R.text_tower(model).layers) == spec["n_layers"] == len(spec["rms"])
        models[key] = (model, tok, spec["rms"])
        meta[key] = {"repo": spec["repo"], "revision": sha, "weights": rec, "load_s": round(time.time() - t0, 1)}
    return models, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--models", default="Q4B,G4,O7")
    ap.add_argument("--weights", action="append", default=[], help="M=PATH (local snapshot dir or HF cache root)")
    ap.add_argument("--poll", type=float, default=0.5)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "no CUDA on this machine"
    torch.use_deterministic_algorithms(True)
    weights = dict(w.split("=", 1) for w in a.weights)
    keys = a.models.split(",")
    assert set(weights) <= set(keys), f"--weights names a model not in --models: {sorted(set(weights) - set(keys))}"
    models, meta = load(keys, weights)
    import transformers
    meta.update({"device_name": torch.cuda.get_device_name(0), "torch": torch.__version__,
                 "transformers": transformers.__version__})
    srv = ProbeServer(models)
    rep = srv.startup(a.responses, meta)
    print("PROBE_SERVER_STARTUP", json.dumps({k: v["identity"]["pass"] for k, v in rep["models"].items()}), flush=True)
    if not rep["pass"]:
        raise SystemExit("identity check failed at startup; see startup.json")
    srv.serve(a.requests, a.responses, a.poll)


if __name__ == "__main__":
    main()
