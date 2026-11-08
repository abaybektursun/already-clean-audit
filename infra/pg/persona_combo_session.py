"""Manual replication of the two Qwen3-8B cards in nautamine's post.

Default implements the published residual-addition formula. The input-offset
variant checks sensitivity to the algebraically equivalent bf16 implementation.
All model work runs on the rented GPU. No judging or automatic dose selection.
"""
import argparse
import contextlib
import json
import math
import os
import pathlib

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TARGETS = ((8, 9994), (16, 1887))


def validate(r):
    assert r.get("op") == "combo"
    assert set(r) <= {"op", "id", "a", "b", "prompt", "max_new_tokens", "implementation", "batch", "mode"}
    for k in ("a", "b"):
        assert math.isfinite(float(r.get(k, 0))) and abs(float(r.get(k, 0))) <= 1.25
    assert r.get("implementation", "residual") in ("residual", "input")
    assert r.get("batch", 1) in (1, 2)
    assert 1 <= r.get("max_new_tokens", 160) <= 320
    assert r.get("mode", "generate") in ("generate", "inspect")
    assert isinstance(r.get("prompt", "Who are you?"), str)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--responses", required=True)
    args = ap.parse_args()
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from infra.pg.probe_server import ProbeServer, write_json

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    layers = model.model.layers
    assert len(layers) == 36 and layers[8].mlp.down_proj.weight.shape == (4096, 12288)

    class Session(ProbeServer):
        def __init__(self):
            super().__init__({}, plan={"poles": []})

        @torch.no_grad()
        def handle(self, r):
            validate(r)
            prompt = r.get("prompt", "Who are you?")
            chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(chat, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
            norms, hs = {}, []
            def capture(l):
                def hook(m, a, o):
                    h = o[0] if isinstance(o, tuple) else o
                    norms[l] = h[0, 1:].float().norm(dim=-1).cpu().numpy()
                return hook
            try:
                for l, _ in TARGETS:
                    hs.append(layers[l].register_forward_hook(capture(l)))
                clean = model(ids, use_cache=False).logits[0, -1].float()
            finally:
                for h in hs:
                    h.remove()
            spec = []
            for (l, n), alpha in zip(TARGETS, (float(r.get("a", 0)), -float(r.get("b", 0)))):
                w = layers[l].mlp.down_proj.weight[:, n].detach().float()
                median = float(np.median(norms[l]))
                spec.append({"layer": l, "neuron": n, "alpha": alpha,
                    "median_residual_norm": median, "column_norm": float(w.norm()),
                    "coefficient": alpha * median / float(w.norm())})
            realized = []
            @contextlib.contextmanager
            def pushes():
                hooks = []
                for s in spec:
                    l, n, c = s["layer"], s["neuron"], s["coefficient"]
                    if r.get("implementation", "residual") == "input":
                        def pre(m, a, n=n, c=c):
                            x = a[0].clone()
                            x[..., n] = (x[..., n].float() + c).to(x.dtype)
                            return (x, *a[1:])
                        hooks.append(layers[l].mlp.down_proj.register_forward_pre_hook(pre))
                    else:
                        delta = layers[l].mlp.down_proj.weight[:, n].detach().float() * c
                        def post(m, a, o, delta=delta, l=l):
                            x = o[0] if isinstance(o, tuple) else o
                            y = (x.float() + delta).to(x.dtype)
                            if len(realized) < 2:
                                d = y[0, -1].float() - x[0, -1].float()
                                dn = float(delta.norm())
                                realized.append({"layer": l, "intended_norm": dn,
                                    "actual_norm": float(d.norm()), "relative_error": float((d-delta).norm()) / dn if dn else float(d.norm()),
                                    "cosine": float(torch.nn.functional.cosine_similarity(d, delta, dim=0)) if dn else None})
                            return (y, *o[1:]) if isinstance(o, tuple) else y
                        hooks.append(layers[l].register_forward_hook(post))
                try:
                    yield
                finally:
                    for h in hooks:
                        h.remove()
            common = {"model": MODEL, "revision": REVISION, "chat_prompt": chat,
                "prompt_token_ids": ids[0].tolist(), "spec": spec, "implementation": r.get("implementation", "residual")}
            if r.get("mode", "generate") == "inspect":
                with pushes():
                    pushed = model(ids, use_cache=False).logits[0, -1].float()
                def top(x):
                    vals, ix = x.softmax(-1).topk(5)
                    return [{"id": int(i), "text": tok.decode([i]), "p": float(v)} for i, v in zip(ix, vals)]
                return {**common, "max_abs_logit_delta": float((pushed-clean).abs().max()),
                    "clean_top5": top(clean), "pushed_top5": top(pushed), "realized": realized}
            batch = int(r.get("batch", 1))
            inputs = ids.repeat(batch, 1)
            with pushes():
                outputs = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                    do_sample=False, max_new_tokens=int(r.get("max_new_tokens", 160)),
                    pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                    use_cache=True)
            samples = []
            for row in outputs[:, ids.shape[1]:]:
                ts = row.tolist()
                samples.append({"token_ids": ts, "text": tok.decode(ts, skip_special_tokens=False),
                    "n_new_tokens": len(ts), "stopped_eos": tok.eos_token_id in ts})
            return {**common, "samples": samples, "realized": realized, "greedy": True, "batch": batch}

    srv = Session()
    zero = srv.handle({"op": "combo", "mode": "inspect"})
    fired = srv.handle({"op": "combo", "mode": "inspect", "a": 1, "b": .75})
    passed = zero["max_abs_logit_delta"] == 0 and 0 < fired["max_abs_logit_delta"] < 1000
    passed = passed and {x["layer"] for x in fired["realized"]} == {8, 16}
    passed = passed and all(0 <= x["relative_error"] < .02 and .99 < x["cosine"] <= 1.00001 for x in fired["realized"])
    startup = {"pass": passed, "identity": zero, "nonzero_fails_identity": fired,
        "meta": {"torch": torch.__version__, "transformers": transformers.__version__,
                 "gpu": torch.cuda.get_device_name(), "git_sha": os.environ.get("PG_GIT_SHA")}}
    path = pathlib.Path(args.responses)
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "startup_combo.json", startup)
    assert passed, "startup gates failed; see startup_combo.json"
    srv.serve(args.requests, args.responses)


if __name__ == "__main__":
    main()
