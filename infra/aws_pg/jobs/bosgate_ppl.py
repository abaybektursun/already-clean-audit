"""BOSGATE — how large is the BOS correction on the clean-text perplexity readout?

`infra/pg/text_eval.clean_text_perplexity` scored raw wikitext windows with no BOS until
2026-09-03 (`add_special_tokens=False`, no prefix). That is the protocol ERLANGEN c6b voided:
`infra/aws_pg/jobs/erlangen_c6b.py:92-109` prefixes every window with BOS because "scoring raw
windows without it destabilises the forward", and `RESEARCH_INDEX.md:133` carries a
`VOID-INSTRUMENT-SUPERSEDED` tag on a whole execution for it (no-BOS base ppl 51.8 vs a corrected
22.222). The fix was never propagated to the shared readout, so every MENDELEEV `ppl` cell banked
before 2026-09-03 is on the voided protocol.

WHY IT MATTERS FOR THE GRID, not just for hygiene. The spec's §1.2 ppl rule is ONE-SIDED:
`after/before > 20` flags a row "fluency-broken". A BOS-fragile BEFORE inflates the denominator,
deflates the ratio, and SUPPRESSES the flag. Row 2 — the flagship — has `google/gemma-2-2b` as its
before-model, and the banked ctrl-g2 cell already shows that model is exactly the fragile case: no
BOS, gemma-2-2b reads 64.07 while its own instruction tune gemma-2-2b-it reads 13.47, i.e. the
"benign tune" appears to IMPROVE perplexity 4.8x. That is an instrument statement, not a finding.

This job measures the correction directly. For five models it computes clean-text perplexity on the
SAME corpus twice:
  * `nobos` — the pre-2026-09-03 protocol, replicated inline verbatim (this is the only copy left;
    text_eval no longer implements it), and
  * `bos`   — the ERLANGEN protocol, by calling the fixed `text_eval.clean_text_perplexity`.
Then it reports, under BOTH protocols, the after/before ratio for row 2, ctrl-g1 and ctrl-g2, and
whether each crosses the §1.2 threshold of 20.

PREDICTION, recorded before the run (the observation that would kill it): gemma-2-2b corrects far
more than PaliGemma-2 does, so row 2's ratio rises from ~9 toward ~40 and CROSSES 20. It is killed
if gemma-2-2b's BOS correction is small (< 1.5x) or if PaliGemma-2 corrects by a similar factor —
in either case the flag stays off and the §1.2 verdict for row 2 is unchanged by the BOS bug.

    python -m infra.aws_pg.jobs.bosgate_ppl --out DIR [--max-tokens 20000] [--window 2048]
    python -m infra.aws_pg.jobs.bosgate_ppl --dryrun          # weight-free, no GPU, no downloads

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

# (repo, attn_implementation) — attn matches infra/pg/matrix_models.ROWS for each row's members.
MODELS = [
    ("google/gemma-2-2b", "eager"),               # row 2 BEFORE, ctrl-g2 BEFORE
    ("google/paligemma2-3b-pt-224", "eager"),     # row 2 AFTER
    ("google/gemma-2-2b-it", "eager"),            # ctrl-g2 AFTER
    ("google/gemma-2b", "eager"),                 # ctrl-g1 BEFORE
    ("google/gemma-1.1-2b-it", "eager"),          # ctrl-g1 AFTER
]

# row -> (before repo, after repo); §1.2 flags a row when after/before > 20.
PAIRS = {
    "2": ("google/gemma-2-2b", "google/paligemma2-3b-pt-224"),
    "ctrl-g2": ("google/gemma-2-2b", "google/gemma-2-2b-it"),
    "ctrl-g1": ("google/gemma-2b", "google/gemma-1.1-2b-it"),
}
PPL_RATIO_FLAG = 20.0    # spec §1.2, one-sided


def ppl_nobos(model, tokenizer, device, window, max_tokens):
    """The pre-2026-09-03 protocol, verbatim: no BOS, windows shifted by one inside the window.

    Kept ONLY here, as the control arm of this job. `text_eval.clean_text_perplexity` no longer
    implements it; every number it produces now is BOS-prefixed."""
    import torch
    from datasets import load_dataset
    from infra.pg.text_eval import WIKITEXT_REPO, WIKITEXT_CONFIG

    ds = load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG, split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if max_tokens:
        ids = ids[:max_tokens]

    nll_sum, n_tok, per_window = 0.0, 0, []
    for start in range(0, ids.size(0) - 1, window):
        chunk = ids[start:start + window + 1]
        if chunk.size(0) < 2:
            break
        inp = chunk[:-1].unsqueeze(0).to(device)
        tgt = chunk[1:].unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(inp).logits
        loss = torch.nn.functional.cross_entropy(logits[0].float(), tgt[0], reduction="sum")
        per_window.append([float(loss.item()), int(tgt.numel())])
        nll_sum += float(loss.item())
        n_tok += int(tgt.numel())

    return {"ppl": math.exp(nll_sum / n_tok), "mean_nll": nll_sum / n_tok, "n_tokens": n_tok,
            "window": window, "per_window": per_window,
            "protocol": "raw windows, NO BOS (pre-2026-09-03, VOIDed by ERLANGEN c6b)"}


def ratios(ppl_by_repo: dict) -> dict:
    out = {}
    for row, (before, after) in PAIRS.items():
        if before not in ppl_by_repo or after not in ppl_by_repo:
            continue
        e = {}
        for prot in ("nobos", "bos"):
            b, a = ppl_by_repo[before][prot], ppl_by_repo[after][prot]
            e[prot] = {"before": b, "after": a, "ratio": a / b,
                       "fluency_broken": bool(a / b > PPL_RATIO_FLAG)}
        e["flag_flipped_by_bos"] = bool(e["nobos"]["fluency_broken"] != e["bos"]["fluency_broken"])
        out[row] = e
    return out


def _dryrun():
    """Weight-free: stub model + stub tokenizer + stubbed dataset. Asserts the two protocols agree on
    token accounting, that only the BOS arm feeds a BOS, and that `ratios` applies §1.2 correctly."""
    import types

    import torch
    import datasets as _ds
    _ds.load_dataset = lambda *a, **k: {"text": ["stub"]}
    V, BOS, N, W = 64, 2, 500, 128

    class Tok:
        bos_token_id = BOS

        def __call__(self, text, return_tensors=None, add_special_tokens=None):
            assert add_special_tokens is False
            return types.SimpleNamespace(input_ids=(torch.arange(N) % (V - 1) + 1).unsqueeze(0))

    seen = []

    class Model:
        def __call__(self, input_ids=None):
            if input_ids is None:
                raise AssertionError("model called without input_ids")
            seen.append(int(input_ids[0, 0]))
            return types.SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], V))

    from infra.pg import text_eval as TE
    m, t = Model(), Tok()
    nb = ppl_nobos(m, t, "cpu", W, N)
    assert BOS not in seen, "the no-BOS arm must never feed a BOS"
    seen.clear()
    b = TE.clean_text_perplexity(m, t, "cpu", window=W, max_tokens=N)
    assert set(seen) == {BOS}, f"the BOS arm must feed BOS at position 0 of every window: {set(seen)}"
    assert abs(nb["ppl"] - V) < 1e-4 and abs(b["ppl"] - V) < 1e-4, (nb["ppl"], b["ppl"])
    assert nb["n_tokens"] == N - 1 and b["n_tokens"] == N, (nb["n_tokens"], b["n_tokens"])
    assert b["protocol"] == "bos-prefixed windows (ERLANGEN c6b)", b["protocol"]

    fake = {"google/gemma-2-2b": {"nobos": 64.0, "bos": 13.5},
            "google/paligemma2-3b-pt-224": {"nobos": 576.0, "bos": 576.0}}
    r = ratios(fake)["2"]
    assert abs(r["nobos"]["ratio"] - 9.0) < 1e-9 and not r["nobos"]["fluency_broken"]
    assert r["bos"]["ratio"] > 42 and r["bos"]["fluency_broken"] and r["flag_flipped_by_bos"]
    print("DRYRUN OK: both protocols wired, BOS only on the BOS arm, §1.2 rule applied one-sided")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--max-tokens", type=int, default=20_000)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--dryrun", action="store_true")
    a = ap.parse_args()
    if a.dryrun:
        _dryrun()
        return
    assert a.out, "--out is required unless --dryrun"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import torch
    from infra.pg import matrix_models as MM
    from infra.pg import text_eval as TE

    assert torch.cuda.is_available(), "BOSGATE needs a GPU"
    device = "cuda"
    t0 = time.time()
    rep = {"object": "bosgate_ppl_v1",
           "question": "how large is the BOS correction on clean_text_perplexity, and does row 2 "
                       "cross the §1.2 fluency-broken threshold once the readout is fixed?",
           "corpus": f"{TE.WIKITEXT_REPO}/{TE.WIKITEXT_CONFIG} test",
           "max_tokens": a.max_tokens, "window": a.window, "ppl_ratio_flag": PPL_RATIO_FLAG,
           "erlangen_reference": "infra/aws_pg/jobs/erlangen_c6b.py:92-109",
           "models": {}, "rows": {}}

    def bank():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2))

    bank()
    ppl_by_repo: dict = {}
    for repo, attn in MODELS:
        model, local = MM._load_model(repo, device, attn)
        tok = MM._tokenizer(local)
        assert tok.bos_token_id is not None, f"{repo}: no BOS token"
        nb = ppl_nobos(model, tok, device, a.window, a.max_tokens)
        bo = TE.clean_text_perplexity(model, tok, device, window=a.window, max_tokens=a.max_tokens)
        rep["models"][repo] = {
            "attn_implementation": attn, "sha": MM.KNOWN[repo]["sha"],
            "bos_token_id": int(tok.bos_token_id),
            "nobos": {k: nb[k] for k in ("ppl", "mean_nll", "n_tokens", "protocol")},
            "bos": {k: bo[k] for k in ("ppl", "mean_nll", "n_tokens", "protocol")},
            "bos_correction_x": nb["ppl"] / bo["ppl"],
            "elapsed_s": round(time.time() - t0, 1),
        }
        ppl_by_repo[repo] = {"nobos": nb["ppl"], "bos": bo["ppl"]}
        rep["rows"] = ratios(ppl_by_repo)
        bank()
        print(f"[BOSGATE] {repo}: nobos {nb['ppl']:.3f}  bos {bo['ppl']:.3f}  "
              f"correction {nb['ppl'] / bo['ppl']:.2f}x", flush=True)
        del model
        torch.cuda.empty_cache()

    rep["elapsed_s"] = round(time.time() - t0, 1)
    bank()
    for row, e in rep["rows"].items():
        print(f"[BOSGATE] row {row}: nobos ratio {e['nobos']['ratio']:.2f} "
              f"(flag {e['nobos']['fluency_broken']}) → bos ratio {e['bos']['ratio']:.2f} "
              f"(flag {e['bos']['fluency_broken']}); flipped={e['flag_flipped_by_bos']}", flush=True)


if __name__ == "__main__":
    main()
