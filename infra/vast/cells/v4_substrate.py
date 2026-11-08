"""V4 — substrate scaling ladder (spec §1/§2/§3: infrastructure, banks not claims).

From-scratch tiny decoders (mercator_pretrain.tiny_config), width/depth doubling
from the 4x256 substrate, ONE seed per rung, byte-level tokenization of the
frozen wikitext text (vocab 256, deterministic). Per rung: train with plain
Adam, snapshot every ckpt_every steps, and evaluate the SPIKE CRITERION per
checkpoint suite:

  emergent bank = bottom-32 eigvecs of sum_t dW_t^T dW_t / ||dW_t||_F^2 over
  consecutive q_proj checkpoint deltas (consensus_from_snapshots — the SAME
  construction as the census estimand, per mercator_pretrain's docstring).
  CRITERION FIRES when leverage_max(emergent bank) > p99 of 64 matched haar32
  draws at that width (the census spikiness object on the pretrain's own
  checkpoint cloud).

Honesty note (banked in config): the spec's "M-B2 substrate gate" exists in the
tree as the construction above, not as a named numeric constant; this frozen
form was recorded in the expectations row BEFORE any V4 run. Deliverable is the
banked ladder + checkpoints regardless of firing (spec §1: "criterion not
reached at banked scales" is a recorded outcome). FAIL LOUD.
"""
import json
import os
import time


def run(cell, out_dir):
    import numpy as np
    import torch
    from infra.pg import mercator_pretrain as MP
    from infra.pg import humboldt_core as HC

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rungs = json.loads(cell["rungs"])            # [[4,256],[4,512],[8,512],[8,1024]]
    steps = int(cell.get("steps", 3000))
    ckpt_every = int(cell.get("ckpt_every", 300))
    seq, bsz = int(cell.get("seq", 256)), int(cell.get("batch", 32))
    K, NDRAW = 32, 64

    # byte-level corpus from the frozen wikitext windows' SOURCE text bytes:
    # deterministic, no tokenizer dependency at tiny vocab
    corpus_txt = cell.get("text_s3")
    import subprocess
    subprocess.run(["aws", "s3", "cp", corpus_txt, "/workspace/v4_text.bin"],
                   check=True, capture_output=True)
    data = np.frombuffer(open("/workspace/v4_text.bin", "rb").read(),
                         dtype=np.uint8).astype(np.int64)

    from transformers import GemmaForCausalLM   # gemma-shaped per tiny_config

    results = []
    for (nl, dm) in rungs:
        rng = np.random.default_rng(HC.cell_seed(2026, "V4", nl, dm))
        torch.manual_seed(int(rng.integers(0, 2**31)))
        cfg = MP.tiny_config(vocab_size=256, d_model=dm, n_layers=nl,
                             n_heads=max(4, dm // 64), max_positions=seq)
        model = GemmaForCausalLM(cfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        keys = [f"model.layers.{L}.self_attn.q_proj.weight" for L in range(nl)]
        snaps, fired, fire_step = [], False, None
        model.train()
        t0 = time.time()
        for step in range(1, steps + 1):
            idx = rng.integers(0, len(data) - seq - 1, size=bsz)
            ids = torch.as_tensor(np.stack([data[i:i + seq] for i in idx]),
                                  device=dev)
            opt.zero_grad()
            model(input_ids=ids, labels=ids).loss.backward()
            opt.step()
            if step % ckpt_every == 0:
                sd = {k: v.detach().cpu().numpy().astype(np.float64)
                      for k, v in model.state_dict().items() if k in keys}
                snaps.append(sd)
                np.savez_compressed(os.path.join(
                    out_dir, f"ckpt_{nl}x{dm}_s{step}.npz"), **sd)
                if len(snaps) >= 3 and not fired:
                    cons = MP.consensus_from_snapshots(snaps, keys, K)
                    lev_maxes = [float(((c["U"] ** 2).sum(axis=1)).max())
                                 for c in cons.values()]
                    med_lev = float(np.median(lev_maxes))
                    hs = []
                    hrng = np.random.default_rng(1337)
                    n = next(iter(cons.values()))["U"].shape[0]
                    for _ in range(NDRAW):
                        g = hrng.standard_normal((n, K))
                        q, _ = np.linalg.qr(g)
                        hs.append(float(((q ** 2).sum(axis=1)).max()))
                    p99 = float(np.quantile(hs, 0.99))
                    # frozen rule: fires when the MEDIAN-over-layers leverage_max
                    # exceeds the haar p99 at that width
                    if med_lev > p99:
                        fired, fire_step = True, step
                    results.append({"rung": f"{nl}x{dm}", "step": step,
                                    "median_leverage_max": med_lev,
                                    "per_layer_leverage_max": lev_maxes,
                                    "haar_p99": p99, "fired": bool(med_lev > p99)})
        loss = None
        model.eval()
        with torch.no_grad():
            ids = torch.as_tensor(np.stack([data[i:i + seq] for i in
                                            rng.integers(0, len(data) - seq - 1,
                                                         size=8)]), device=dev)
            loss = float(model(input_ids=ids, labels=ids).loss)
        results.append({"rung": f"{nl}x{dm}", "final_loss": loss,
                        "fired": fired, "fire_step": fire_step,
                        "elapsed_s": round(time.time() - t0, 1)})
        print(f"[v4] {nl}x{dm}: fired={fired}@{fire_step} loss={loss:.3f}")
        del model, opt
        if dev == "cuda":
            torch.cuda.empty_cache()
        if fired and cell.get("stop_on_fire", True):
            break
    with open(os.path.join(out_dir, "v4_results.json"), "w") as f:
        json.dump({"rows": results, "criterion":
                   "leverage_max(bottom-32 consensus over ckpt deltas) > "
                   "haar32 p99 (64 draws) — frozen pre-run; M-B2 named gate "
                   "not present as code, construction per mercator_pretrain "
                   "estimand", "steps": steps, "ckpt_every": ckpt_every}, f,
                  indent=2)
