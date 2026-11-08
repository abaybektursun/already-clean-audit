"""CTRLIFACE Arm-B, part 1 — estimate the TEXT-sensitivity subspace S_text(k,L) of the π0.5
backbone decoder (spec §3 Arm B). Reuses the VLAT swap-back instrument: the π0.5 decoder layers
loaded into a base Gemma-1 text shell (fixed base embed/head), run as a TEXT LM. S_text[L] = top-k
eigenvectors of the activation-space Gram of the next-token CE loss w.r.t. that layer's residual h_L
— the SAME layer index + d=2048 space as S_action, so the Arm-B overlap o(S_action,S_text) is
well-defined (T-parity disclosed: S_action = obs-forward, S_text = text-forward, both ⊂ R^d at L).

Text corpus: the LIBERO task-description strings (natural language, self-contained — no extra
download). S_text is the decoder's text-CE sensitivity; a diverse NL corpus suffices for the subspace.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ctrliface_stext --type g6e.2xlarge,g5.2xlarge,g5.xlarge \
      --no-weights --s3-prefix results/ctrliface_stext --budget-min 90 --args "--k-max 128"
"""
import argparse
import json
import pathlib
import subprocess
import sys


def _install():
    subprocess.run(["bash", "-lc",
                    "apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y "
                    "git python3-dev build-essential libgl1 libglib2.0-0"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "lerobot[libero,pi] @ git+https://github.com/huggingface/lerobot.git"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-max", type=int, default=128)
    ap.add_argument("--n-text", type=int, default=512, help="# text strings from the corpus")
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    _install()
    import os
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from infra.pg import ctrliface_lerobot as LR
    from infra.pg import ctrliface_run as CR
    from infra.pg import ctrliface_core as CC
    from infra.pg import vlat_swapback as VS

    rep = {"k_max": a.k_max, "n_text": a.n_text, "corpus": "libero_task_descriptions"}

    # --- π0.5 decoder → text-runnable Gemma-1 shell (base embed/head, π0.5 decoder layers) ---
    # Load the policy on CPU (we only need its decoder state_dict); the GPU is reserved for the
    # swapback text model. Avoids the 24 GB OOM from holding policy + swapback on-GPU together.
    policy = LR.load_policy(device="cpu")
    backbone, _layers, d = LR.get_backbone(policy)
    # EXCLUDE embed_tokens/lm_head — the swapback keeps the BASE Gemma-1 embed/head as the fixed
    # readout instrument and swaps only the decoder matrices+norms (vlat_swapback contract).
    decoder_sd = {k: v.detach().cpu() for k, v in backbone.state_dict().items()
                  if not any(e in k for e in ("embed_tokens", "lm_head"))}
    del backbone, policy
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    text_model, info = VS.load_swapback_model(decoder_sd, "cuda")
    # do NOT freeze — loss.requires_grad needs a grad-requiring param (see discovery note); the
    # text model is 2.5B so param grads fit easily. h_L.grad still flows to layers[L].
    layers = text_model.model.layers
    assert len(layers) == 18, f"swapback layers {len(layers)} != 18"
    rep["d"], rep["swapback_info"] = d, {k: str(v) for k, v in info.items()}

    from infra.pg.entities import ENTITIES
    tok = AutoTokenizer.from_pretrained(ENTITIES["L0_g1"].repo_id, revision=ENTITIES["L0_g1"].revision)

    # --- text corpus: LIBERO task descriptions (natural language) ---
    meta = LR.dataset_meta()
    tasks_tbl = meta.tasks
    if hasattr(tasks_tbl, "column_names"):
        tcol = "task" if "task" in tasks_tbl.column_names else tasks_tbl.column_names[0]
        corpus = [str(tasks_tbl[i][tcol]) for i in range(min(a.n_text, len(tasks_tbl)))]
    else:
        # episodes.tasks fallback
        ep = meta.episodes
        corpus = []
        for i in range(min(a.n_text, len(ep))):
            v = ep[i]["tasks"]
            corpus.append(" ".join(v) if isinstance(v, (list, tuple)) else str(v))
    corpus = [c for c in corpus if c.strip()]
    rep["n_corpus"] = len(corpus)
    assert len(corpus) >= 32, f"text corpus too small: {len(corpus)}"

    def _batches():
        for s in range(0, len(corpus), a.batch_size):
            enc = tok(corpus[s:s + a.batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=64)
            yield {k: v.to("cuda") for k, v in enc.items()}

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def text_loss_fn(batch):
        out = text_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        V = out.logits.shape[-1]
        sl = out.logits[:, :-1, :].reshape(-1, V)
        lab = batch["input_ids"][:, 1:].reshape(-1)
        return F.cross_entropy(sl.float(), lab, ignore_index=pad_id)

    batches = list(_batches())
    layers_dict = {str(L): layers[L] for L in range(len(layers))}
    G, npos = CR.accumulate_gram_multi(text_model, layers_dict, text_loss_fn, batches, d, n_prefix=None)
    rep["gram_positions"] = npos

    bases, spectra = {}, {}
    for L in range(len(layers)):
        Q, evals = CC.topk_eigvecs(G[str(L)], a.k_max)
        bases[L] = Q.cpu()
        spectra[str(L)] = [float(x) for x in evals[:min(32, a.k_max)]]
    torch.save({"stext_bases": bases, "d": d}, out / "stext_bases.pt")
    rep["eigval_spectra_top32"] = spectra
    (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    print("STEXT_OK", json.dumps({"gram_positions": npos, "n_corpus": rep["n_corpus"], "d": d}))


if __name__ == "__main__":
    main()
