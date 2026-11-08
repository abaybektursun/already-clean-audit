"""A1 (2026-11-06): memory of the gradient pass per attention kernel, weight-free, on the laptop CPU.

OLMo-3-7B's layer at one eighth of its width (hidden 512 = 4 heads of 128, MLP 1376; the real model has hidden 4096 =
32 heads of 128, MLP 11008), random bf16 weights, one window of 2,049 tokens, the gradient leaf at layer 0's
down-projection input (as n12_score.grad_capture), every parameter frozen. Every tensor autograd saves for the backward
is proportional to the hidden size, the MLP width or the number of heads (times 2,049 squared for the attention
matrix), so the saved bytes of one layer at full width are 8 times the measured ones. One layer's bytes are the
difference between a 3-layer and a 2-layer model. Kernels: the default sdpa (CPU: the fused flash kernel), the math
kernel with its default fp32 upcast, the math kernel with bf16 reductions. Also: how transformers calls sdpa (mask or
is_causal), and the gradient's agreement between kernels.

    python studies/next/n12/offline_gate/a1_memory.py --hf <dir with config_stage1-step560000.json> --out a1_memory.json
"""
import argparse
import contextlib
import json

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Olmo3Config, Olmo3ForCausalLM

MODES = ("default", "math_fp32", "math_bf16")


def model(cfg: dict, n_layers: int):
    c = dict(cfg, num_hidden_layers=n_layers, layer_types=["sliding_attention"] * n_layers, vocab_size=1024,
             pad_token_id=1023, eos_token_id=1022, hidden_size=512, num_attention_heads=4, num_key_value_heads=4,
             intermediate_size=1376)
    torch.manual_seed(0)
    m = Olmo3ForCausalLM(Olmo3Config(**{k: v for k, v in c.items() if k not in
                                        ("architectures", "model_type", "transformers_version")})).to(torch.bfloat16)
    m.config._attn_implementation = "sdpa"
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def run(m, x, mode: str, calls: list):
    store, seen = {}, {}

    def pack(t):
        s = t.untyped_storage()
        seen[s.data_ptr()] = s.nbytes()
        return t

    def hook(module, args):
        h = args[0].detach().requires_grad_(True)
        store[0] = h
        return (h,)
    hd = m.model.layers[0].mlp.down_proj.register_forward_pre_hook(hook)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(mode == "math_bf16")
    ctx = contextlib.nullcontext() if mode == "default" else sdpa_kernel([SDPBackend.MATH])
    orig = F.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        calls.append({"attn_mask": None if attn_mask is None else list(attn_mask.shape), "is_causal": is_causal,
                      "q": list(q.shape), "dtype": str(q.dtype)})
        return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)
    F.scaled_dot_product_attention = spy
    with ctx, torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        logits = m(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
        ce = F.cross_entropy(logits[0, :-1].float(), x[0, 1:], reduction="none")
    F.scaled_dot_product_attention = orig
    ce.sum().backward()
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    hd.remove()
    return sum(seen.values()), store[0].grad.float().clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    cfg = json.load(open(f"{a.hf}/config_stage1-step560000.json"))
    x = torch.randint(0, 1000, (1, 2049), generator=torch.Generator().manual_seed(0))
    out = {"torch": torch.__version__, "window_tokens": 2049, "width_factor": 8, "saved_bytes": {}, "grad": {}}
    calls = []
    for nl in (2, 3):
        m = model(cfg, nl)
        res = {mode: run(m, x, mode, calls) for mode in MODES}
        out["saved_bytes"][nl] = {mode: res[mode][0] for mode in MODES}
        g = {mode: res[mode][1] for mode in MODES}
        out["grad"][nl] = {f"{p}_vs_{q}": {"exact_share": float((g[p] == g[q]).float().mean()),
                                           "rel_l2": float((g[p] - g[q]).norm() / g[q].norm())}
                           for p, q in (("math_fp32", "default"), ("math_bf16", "default"), ("math_bf16", "math_fp32"))}
        out["grad"][nl]["max_abs_G"] = float(g["default"].abs().max())
    out["sdpa_call"] = calls[0]
    out["sdpa_call_forms"] = sorted({json.dumps({k: v for k, v in c.items() if k != "q"}) for c in calls})
    per_layer = {mode: 8 * (out["saved_bytes"][3][mode] - out["saved_bytes"][2][mode]) / 1e9 for mode in MODES}
    out["per_layer_full_width_GB"] = per_layer
    # A window whose lowest needed layer is 0 puts 31 full layers in the graph. Beside them: the weights (7.298e9
    # bf16 parameters), the fp32 log-softmax of the 2,048 x 100,278 logits (saved), the bf16 logits (alive), the
    # final norm's fp32 input, and at the start of the backward the fp32 gradient of the logits.
    fixed = {"weights": 7.298e9 * 2 / 1e9, "log_softmax_fp32": 2048 * 100278 * 4 / 1e9,
             "logits_bf16": 2049 * 100278 * 2 / 1e9, "final_norm_fp32": 2049 * 4096 * 4 / 1e9,
             "backward_logit_grad_fp32": 2048 * 100278 * 4 / 1e9}
    out["fixed_GB"] = fixed
    out["peak_estimate_GB_lowest_layer_0"] = {mode: round(31 * per_layer[mode] + sum(fixed.values()), 2) for mode in MODES}
    out["l40s_total_GB"] = "46,068 MiB = 48.3 GB as nvidia-smi reports it on g6e; less the CUDA context"
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("per_layer_full_width_GB", "peak_estimate_GB_lowest_layer_0", "sdpa_call_forms")}, indent=1))


if __name__ == "__main__":
    main()
