"""N13 R0 compatibility shim for olmes 5a51f502 on its own pins, vllm 0.11.0 and lm_eval 0.4.3 (studies/next/07_PICKS/N13_SPEC.md A18 (b)).

lm_eval 0.4.3 calls LLM.generate(prompt_token_ids=ids, ...); vLLM 0.11.0's LLM.generate has no such parameter. The shim converts that call
to LLM.generate([TokensPrompt(prompt_token_ids=x) for x in ids], ...), the conversion vLLM made itself through v0.10.1
(LLM._convert_v1_inputs), and passes every other argument through unchanged. A call without prompt_token_ids is not touched.

Python imports sitecustomize at start-up. n13_known puts this file's directory first on PYTHONPATH and sets N13_VLLM_SHIM_LOG only in the
olmes subprocess's environment, so the olmes processes import vllm and install the shim; every other process finds nothing to do.
Each converted call appends one line to the N13_VLLM_SHIM_LOG file after vLLM returns (A19 revision, 2026-11-05): n_prompts, the wall
times t_start and t_end around vLLM's own generate, and output_tokens, the sum of len(token_ids) over every completion of every returned
RequestOutput (vllm/outputs.py at v0.11.0: RequestOutput.outputs, CompletionOutput.token_ids). The return value is passed back unchanged.
n13_known writes the counts into out/r0_known/metrics.json and projects the full task from them. No olmes or lm_eval file is edited.
Fail-loud: no try/except.
"""
import json
import os
import sys
import time


def install(llm_cls, tokens_prompt, log_path):
    orig = llm_cls.generate

    def generate(self, *args, prompt_token_ids=None, **kwargs):
        if prompt_token_ids is None:
            return orig(self, *args, **kwargs)
        assert not args and "prompts" not in kwargs, "both prompts and prompt_token_ids given"
        assert isinstance(prompt_token_ids, list) and all(isinstance(x, list) for x in prompt_token_ids), type(prompt_token_ids)
        prompts = [tokens_prompt(prompt_token_ids=x) for x in prompt_token_ids]
        t_start = time.time()
        out = orig(self, prompts, **kwargs)
        t_end = time.time()
        n_tokens = sum(len(c.token_ids) for o in out for c in o.outputs)
        with open(log_path, "a") as f:
            f.write(json.dumps({"pid": os.getpid(), "n_prompts": len(prompts), "t_start": t_start, "t_end": t_end,
                                "output_tokens": n_tokens}) + "\n")
        return out

    llm_cls.generate = generate
    print(f"N13 vLLM shim installed: {llm_cls.__module__}.{llm_cls.__qualname__}.generate converts prompt_token_ids (pid {os.getpid()})",
          file=sys.stderr, flush=True)


if os.environ.get("N13_VLLM_SHIM_LOG"):
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    install(LLM, TokensPrompt, os.environ["N13_VLLM_SHIM_LOG"])
