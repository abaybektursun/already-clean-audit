"""DARWIN G2 — battery preflight (§9 step 6). GPU (g6e L40S).

The gate: reproduce PG1's base perplexity from the FUNC/HUMBOLDT artifact to within +-2% relative,
using the SAME code path, before any GPU arm is allowed to quote a battery number.

Anchor provenance (§0.18 — recomputed from its artifact, never quoted from prose or memory):
  s3://pg-lineage-769063704696/humboldt/humboldt_b1/invariant_core_func.json
      .base_repo                 google/paligemma-3b-pt-224
      .base_sha                  35e4f46485b4d07967e7e9935bc3786aad50687c
      .capability_baseline.ppl   <read at runtime>
      .capability_baseline.ppl_n_tokens
The number is FETCHED at runtime and compared; it is deliberately not written into this file, so
the gate cannot pass against a stale constant someone pasted.

Why this gate is worth a GPU-hour: PG1-class forwards must use EAGER attention (sdpa mishandles
the prefix-LM mask) and gemma-2b text forwards must prepend BOS. Both are silent failures — the
model runs, the loss is finite, and every downstream battery number is quietly wrong. A ppl
anchor catches exactly that class.

  pg run infra.aws_pg.jobs.darwin_g2 --type g6e.2xlarge,g6e.4xlarge,g6e.8xlarge,g6e.xlarge \
      --no-weights --s3-prefix results/darwin_g2 --budget-min 90
"""
import argparse
import json
import pathlib
import subprocess

from infra.pg import darwin_core as D
from infra.pg import humboldt_core as H

FUNC_ANCHOR_S3 = ("s3://pg-lineage-769063704696/humboldt/humboldt_b1/invariant_core_func.json")
REL_TOL = 0.02                      # §9 step 6: +-2% relative


def fetch_anchor(out) -> dict:
    """Pull the FUNC artifact and extract the anchor. Fails loud if either is absent."""
    p = pathlib.Path(out) / "func_anchor.json"
    subprocess.run(["aws", "s3", "cp", FUNC_ANCHOR_S3, str(p), "--region", "us-east-1"],
                   check=True, capture_output=True, text=True)
    d = json.loads(p.read_text())
    cb = d.get("capability_baseline") or {}
    assert "ppl" in cb, f"no capability_baseline.ppl in {FUNC_ANCHOR_S3}"
    return {"base_repo": d["base_repo"], "base_sha": d["base_sha"],
            "ppl": float(cb["ppl"]), "n_tokens": int(cb.get("ppl_n_tokens", 0)),
            "source": FUNC_ANCHOR_S3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="G2")
    ap.add_argument("--run-id", default="darwin_g2")
    ap.add_argument("--s3-prefix", default="darwin/darwin_g2")
    ap.add_argument("--n-seq", type=int, default=0,
                    help="0 = match the anchor's token count as closely as the window allows")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    anchor = fetch_anchor(out)
    print(f"[G2] anchor: ppl={anchor['ppl']:.6f} over {anchor['n_tokens']} tokens "
          f"({anchor['base_repo']}@{anchor['base_sha'][:8]})", flush=True)

    from transformers import AutoProcessor
    from infra.aws_pg.jobs.invariant_core_func import PG1_REPO, _load_full_pg1, _ppl_blankimg

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        exp.log(a.arm,
                refuter="the reproduced base perplexity misses the FUNC anchor by more than 2% "
                        "relative",
                direction="the same code path reproduces the banked base perplexity to well "
                          "inside 2% relative",
                utc=utc)
    ref = exp.require(a.arm)

    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="ATLAS", wave="W0",
                    expected_ref=ref)

    # n_seq is chosen to land on the anchor's token budget: the anchor was computed over
    # ppl_n_tokens tokens in `window`-token blocks, so matching the count matters more than
    # matching a sequence count that was never recorded.
    window = 512
    n_seq = a.n_seq or max(1, round(anchor["n_tokens"] / window))
    # _load_full_pg1 returns the MODEL only; the processor is loaded separately at the same
    # revision so the tokenisation the anchor used is the tokenisation reproduced here.
    model = _load_full_pg1(anchor["base_sha"], "cuda")
    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=anchor["base_sha"])
    res = _ppl_blankimg(model, processor, "cuda", n_seq=n_seq, window=window)

    got, want = float(res["ppl"]), anchor["ppl"]
    rel = abs(got - want) / want
    flags = {"anchor_source": anchor["source"], "anchor_ppl": want,
             "anchor_n_tokens": anchor["n_tokens"], "n_seq": n_seq, "window": window,
             "n_tokens": res.get("n_tokens"), "rel_tol": REL_TOL,
             "attn_implementation": "eager", "note": "PG1-class forwards: eager only"}
    atlas.add("base_ppl_reproduced", value=got, model_id=anchor["base_repo"],
              model_sha=anchor["base_sha"], flags=flags)
    atlas.add("base_ppl_relative_error", value=rel, model_id=anchor["base_repo"],
              model_sha=anchor["base_sha"], flags=flags)

    gates.require("G2_battery_preflight_base_ppl", rel < REL_TOL,
                  reproduced_ppl=got, anchor_ppl=want, relative_error=rel, tol=REL_TOL,
                  n_tokens_reproduced=res.get("n_tokens"), n_tokens_anchor=anchor["n_tokens"],
                  anchor_source=anchor["source"])

    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, "g2_summary.json",
                 {"arm": a.arm, "run_id": a.run_id, "commit_sha": D.commit_sha(),
                  "provenance": D.provenance(), "anchor": anchor, "reproduced": {k: v for k, v in res.items() if k != "per_window"},
                  "relative_error": rel, "atlas": rep, "gates": gates.gates,
                  "expected_ref": ref, "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_G2_OK ppl={got:.6f} anchor={want:.6f} rel_err={rel:.5f} tol={REL_TOL}")


if __name__ == "__main__":
    main()
