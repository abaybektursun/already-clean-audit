"""N2V: does N12's gradient pass fit the L40S in fp32? Weight-free, laptop CPU, the method of N12's A1 memory estimate
(studies/next/n12/offline_gate/a1_memory.py), whose bf16 estimate (36.51 GB) the instance then measured (36.04 GB,
studies/next/n12/banks/score-02/n12/metrics.json -> points.14.grad_max_mem_gb).

OLMo-3-7B's layer at one eighth of its width, one window of 2,049 tokens, the gradient leaf at layer 0's down-projection
input, the sdpa math kernel (N12 A1). Saved bytes per full-width layer = 8 x (3-layer model - 2-layer model). Modes:
  bf16_math_bf16red   N12 score-02's pass (bf16 model, math kernel, bf16 reductions): the control, against 36.04 GB;
  fp32_math           the same pass on the fp32 model (N2V's question).
Peak = layers in the graph x saved bytes per layer + the weights + the logits' log-softmax, logits and their gradient
(fp32 when the model is fp32). The layers in the graph for a unit of layer L are 32 - L (layer L's down projection and
everything above); the real units' layers come from studies/next/n12/units.json.

    python studies/next/n2v/offline_gate/mem_fp32.py --hf <dir with config_stage1-step560000.json> --out mem_fp32.json
"""
import argparse
import json
import pathlib

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Olmo3Config, Olmo3ForCausalLM

ROOT = pathlib.Path(__file__).resolve().parents[4]
N_PARAMS = 7.298e9
L40S_GB = 46068 * 2 ** 20 / 1e9            # nvidia-smi on g6e: 46,068 MiB


def model(cfg: dict, n_layers: int, dtype):
    c = dict(cfg, num_hidden_layers=n_layers, layer_types=["sliding_attention"] * n_layers, vocab_size=1024,
             pad_token_id=1023, eos_token_id=1022, hidden_size=512, num_attention_heads=4, num_key_value_heads=4,
             intermediate_size=1376)
    torch.manual_seed(0)
    m = Olmo3ForCausalLM(Olmo3Config(**{k: v for k, v in c.items() if k not in
                                        ("architectures", "model_type", "transformers_version")})).to(dtype)
    m.config._attn_implementation = "sdpa"
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def saved_bytes(m, x, bf16_red: bool) -> int:
    seen = {}

    def pack(t):
        s = t.untyped_storage()
        seen[s.data_ptr()] = s.nbytes()
        return t

    def hook(module, args):
        return (args[0].detach().requires_grad_(True),)
    hd = m.model.layers[0].mlp.down_proj.register_forward_pre_hook(hook)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(bf16_red)
    with sdpa_kernel([SDPBackend.MATH]), torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        logits = m(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
        ce = F.cross_entropy(logits[0, :-1].float(), x[0, 1:], reduction="none")
    ce.sum().backward()
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    hd.remove()
    return sum(seen.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    cfg = json.load(open(f"{a.hf}/config_stage1-step560000.json"))
    x = torch.randint(0, 1000, (1, 2049), generator=torch.Generator().manual_seed(0))
    modes = {"bf16_math_bf16red": (torch.bfloat16, True, 2), "fp32_math": (torch.float32, False, 4)}
    out = {"torch": torch.__version__, "window_tokens": 2049, "width_factor": 8, "per_layer_full_width_GB": {},
           "peak_lowest_layer_0_GB": {}, "l40s_GB": round(L40S_GB, 2)}
    V, T = 100278, 2048
    for name, (dt, red, nbytes) in modes.items():
        b2 = saved_bytes(model(cfg, 2, dt), x, red)
        b3 = saved_bytes(model(cfg, 3, dt), x, red)
        per = 8 * (b3 - b2) / 1e9
        fixed = N_PARAMS * nbytes / 1e9 + 3 * T * V * 4 / 1e9 + 2049 * 4096 * 4 / 1e9
        out["per_layer_full_width_GB"][name] = round(per, 3)
        out["fixed_GB"] = out.get("fixed_GB", {})
        out["fixed_GB"][name] = round(fixed, 2)
        out["peak_lowest_layer_0_GB"][name] = round(31 * per + fixed, 2)
    units = json.loads((ROOT / "studies/next/n12/units.json").read_text())["units"]
    layers = [u["layer"] for u in units if u["kind"] == "real"]
    per, fx = out["per_layer_full_width_GB"]["fp32_math"], out["fixed_GB"]["fp32_math"]
    peaks = [(32 - L) * per + fx for L in layers]
    out["fp32_math_real_units"] = {"n": len(peaks), "fit_under_l40s_less_2GB": sum(p <= L40S_GB - 2.0 for p in peaks),
                                   "lowest_layer_that_fits": min(L for L, p in zip(layers, peaks) if p <= L40S_GB - 2.0)
                                   if any(p <= L40S_GB - 2.0 for p in peaks) else None,
                                   "max_peak_GB": round(max(peaks), 2)}
    out["control"] = {"n12_score02_measured_grad_max_mem_gb": 36.04, "a1_estimate_math_bf16": 36.51,
                      "this_estimate_bf16_math_bf16red": out["peak_lowest_layer_0_GB"]["bf16_math_bf16red"]}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
