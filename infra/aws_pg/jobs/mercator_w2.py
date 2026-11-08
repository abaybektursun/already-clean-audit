"""M-10 and M-11 — the two Wave-2 pretrains that ask about the sink's ADDRESS.

    --arm M-10   REGISTERS IN WIDTH. Offer the network a sanctioned 32-coordinate register block
                 and see whether training adopts it as its sink. Three switches, each a plausible
                 reason adoption might fail or succeed:
                     none                 the plain pretrain, the reference
                     fixed_gamma          the RMSNorm gains are frozen at 1, so no per-coordinate
                                          gain can select coordinates for the network
                     writer_init_ablated  writers are initialised to write nothing into the block,
                                          so adoption has to be learned rather than inherited
                 and an OPTIMIZER ladder — Adam, block-Adam, scalar-Adam, SGD — because M-A3 stage
                 2 measured that the denominator's SHAPE is what selects coordinates, and block
                 size is that shape made continuous.
                 REFUTER: adoption angle stays at the Haar band (the network ignores the block).

    --arm M-11   THE ROTATION-EQUIVARIANT TWIN. SGD's update IS the gradient, so rotating the
                 parameter space rotates the update with it: an SGD-trained model has no preferred
                 coordinate frame for the optimizer to imprint. M-Q stage 2 found the sink's
                 address is a property of the frame Adam counts in. So the equivariant twin should
                 show no spiky leverage structure.
                 REFUTER of the grid theory: spiky structure in the equivariant twin.

ONE HARNESS, TWO ARMS, and the shared substrate is `mercator_pretrain` so a difference between
them cannot be a difference of harness. The models are small and the arms are switches on them;
that is stated rather than buried, because the question is about what training DOES and not about
scale.

THE ADOPTION AND SPIKINESS CLAIMS ARE COORDINATE CLAIMS, so §3's dual-basis rule binds: each is
reported in the coordinate basis AND in one random reference basis, and the reference row is what
shows the coordinate row was not an artifact of the estimator.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import mercator_core as M
from infra.pg import mercator_pretrain as PT
from infra.pg import mercator_program as P

ARMS = ("M-10", "M-11")
N_HAAR_DRAWS = 8
SPIKE_MULT = 5.0
# "Spiky" means the participation ratio sits below this fraction of its flat value n.
SPIKY_FRACTION = 0.9
# An optimizer arm whose final loss exceeds this multiple of the Adam reference did not train.
DIVERGENCE_MULTIPLE = 3.0


def spikiness(U) -> dict:
    """Leverage concentration of a subspace in the coordinate basis.

    THE PARTICIPATION RATIO'S FLAT VALUE IS n, NOT 1. Leverage sums to k over n coordinates, so a
    subspace spread evenly has l_i = k/n and participation ratio (sum l)^2 / sum l^2 = n. A spiky
    one is far below n. The first run of M-11 gated "is Adam spiky at all" on `PR < 1.0`, which is
    unreachable for this statistic and so read False on an Adam run whose PR was 197 against a flat
    256 — i.e. genuinely concentrated (defect M-D31). The normalised form is banked alongside so a
    reader never has to know n to read the number.
    """
    lev = H.leverage(U)
    n, k = U.shape
    pr = float(H.participation_ratio(lev))
    return {"leverage_max": float(lev.max()), "participation_ratio": pr,
            "participation_fraction": pr / float(n),
            "flat_participation_ratio": float(n),
            "spike_count": int((lev > SPIKE_MULT * (k / n)).sum()),
            "flat_expectation": float(k) / float(n), "n": int(n), "k": int(k)}


def main():
    ap = argparse.ArgumentParser(description="M-10 / M-11 — Wave-2 pretrains")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sgd-lr", type=float, default=0.1)
    ap.add_argument("--n-snapshots", type=int, default=9)
    ap.add_argument("--n-docs", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--seeds", default="1337,1338")
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    meta = P.ARMS[a.arm]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w2") / "expectations.jsonl")
    expected_ref = exp.require(a.arm)
    at = M.Atlas(out, arm=a.arm, run_id=f"m_w2_{a.arm.lower().replace('-', '_')}",
                 register=meta["register"], wave=meta["wave"], expected_ref=expected_ref,
                 queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)
    optinfo = M.check_optimizer(a.arm, "mercator_pretrain.build_optimizer", offload_exact_fp32=False)

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, GemmaForCausalLM

    assert torch.cuda.is_available(), "no CUDA"
    tok = AutoTokenizer.from_pretrained("google/gemma-2b")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
                      streaming=True)
    ids, need = [], a.n_docs * a.seq_len
    for row in ds:
        t = str(row["text"]).strip()
        if not t:
            continue
        ids.extend(tok(t, add_special_tokens=False).input_ids)
        if len(ids) >= need:
            break
    assert len(ids) >= need, f"corpus gave {len(ids)} tokens, need {need}"
    data = torch.tensor(ids[:need], dtype=torch.long).reshape(-1, a.seq_len)
    print(f"[{a.arm}] {data.shape[0]} windows of {a.seq_len} tokens, vocab {len(tok)}", flush=True)

    sanctioned = PT.sanctioned_block(a.d_model, H.BANK_K, seed=0, kind="coordinate")

    def build(switch, seed):
        torch.manual_seed(seed)
        cfg = PT.tiny_config(len(tok), a.d_model, a.n_layers, a.n_heads,
                             max_positions=a.seq_len)
        model = GemmaForCausalLM(cfg).to("cuda")
        if switch == "fixed_gamma":
            # Freeze every RMSNorm gain at its initial value. Gemma's norm applies (1 + weight),
            # so zeroing the weight and freezing it makes the gain exactly 1 everywhere: no
            # per-coordinate gain exists for the network to select coordinates with.
            for name, mod in model.named_modules():
                if "norm" in name and hasattr(mod, "weight"):
                    with torch.no_grad():
                        mod.weight.zero_()
                    mod.weight.requires_grad_(False)
        elif switch == "writer_init_ablated":
            # Writers (o_proj, down_proj) start with their contribution to the sanctioned block
            # projected out, so any energy that appears there later was LEARNED.
            Ub = torch.as_tensor(sanctioned, device="cuda", dtype=torch.float32)
            with torch.no_grad():
                for layer in model.model.layers:
                    for w in (layer.self_attn.o_proj, layer.mlp.down_proj):
                        W = w.weight.data.to(torch.float32)
                        w.weight.data.copy_((W - Ub @ (Ub.T @ W)).to(w.weight.dtype))
        return model

    def trainable(model):
        return {("q_proj", i): layer.self_attn.q_proj
                for i, layer in enumerate(model.model.layers)}

    def pretrain(model, opt_name, seed):
        mods = trainable(model)
        keys = sorted(mods)
        lr = a.sgd_lr if opt_name == "sgd" else a.lr
        opt = PT.build_optimizer(opt_name, [p for p in model.parameters() if p.requires_grad],
                                 lr=lr, block=a.block_size)
        rng = np.random.default_rng(H.cell_seed(seed, a.arm, opt_name, "pretrain"))
        every = max(1, a.steps // (a.n_snapshots - 1))
        snaps, losses = [], []
        model.train()
        for step in range(a.steps + 1):
            if step % every == 0 or step == a.steps:
                snaps.append({k: mods[k].weight.detach().to(torch.float32).cpu().numpy()
                              .astype(np.float64) for k in keys})
            if step == a.steps:
                break
            idx = rng.integers(0, data.shape[0], a.batch)
            b = data[idx].to("cuda")
            out_ = model(input_ids=b, labels=b)
            opt.zero_grad()
            out_.loss.backward()
            opt.step()
            losses.append(float(out_.loss.detach()))
            if (step + 1) % 500 == 0:
                print(f"    {opt_name:12s} step {step+1}/{a.steps} "
                      f"loss={np.mean(losses[-100:]):.4f}", flush=True)
        return snaps, keys, float(np.mean(losses[-100:]))

    # ---- the grid ----------------------------------------------------------------------------
    if a.arm == "M-10":
        grid = [(sw, op) for sw in PT.SWITCHES for op in ("adam", "block_adam", "scalar_adam")]
    else:
        grid = [("none", op) for op in ("adam", "sgd", "block_adam", "scalar_adam")]

    cells = {}
    for switch, opt_name in grid:
        for seed in seeds:
            model = build(switch, seed)
            snaps, keys, final_loss = pretrain(model, opt_name, seed)
            cons = PT.consensus_from_snapshots(snaps, keys, H.BANK_K)
            per_layer = {}
            for k in keys:
                U = cons[k]["U"]
                ad = PT.adoption_angle(U, sanctioned)
                band = PT.haar_adoption_band(U, N_HAAR_DRAWS, seed=1000 + k[1])
                sp = spikiness(U)
                # §3 dual basis: the SAME emergent subspace conjugated by a random reference
                # basis. Leverage is a coordinate statistic and must move; adoption of a
                # coordinate block must collapse to the independent level.
                Q = M.random_reference_basis(U.shape[0], seed + k[1])
                Ur = Q @ U
                per_layer[f"L{k[1]}"] = {
                    "adoption": ad, "haar_band": band, "spikiness": sp,
                    "adoption_reference_basis": PT.adoption_angle(Ur, sanctioned),
                    "spikiness_reference_basis": spikiness(Ur),
                    "n_terms": cons[k]["n_terms"]}
                at.add("adoption_angle", value=ad["mean_cos2"], layer=k[1], type=k[0],
                       bank_id="emergent_bottom32", basis=M.BASIS_COORD,
                       flags={**ad, "switch": switch, "optimizer": opt_name, "seed": seed,
                              "haar_band": band, "final_loss": final_loss})
                at.add("adoption_angle", value=per_layer[f"L{k[1]}"][
                    "adoption_reference_basis"]["mean_cos2"], layer=k[1], type=k[0],
                    bank_id="emergent_bottom32", basis=M.BASIS_RANDOM_REF,
                    flags={"switch": switch, "optimizer": opt_name, "seed": seed})
                at.add("emergent_leverage_participation_ratio", value=sp["participation_ratio"],
                       layer=k[1], type=k[0], bank_id="emergent_bottom32",
                       basis=M.BASIS_COORD,
                       flags={**sp, "switch": switch, "optimizer": opt_name, "seed": seed})
                at.add("emergent_leverage_participation_ratio",
                       value=per_layer[f"L{k[1]}"]["spikiness_reference_basis"][
                           "participation_ratio"], layer=k[1], type=k[0],
                       bank_id="emergent_bottom32", basis=M.BASIS_RANDOM_REF,
                       flags={"switch": switch, "optimizer": opt_name, "seed": seed})
            mean_ad = float(np.mean([v["adoption"]["mean_cos2"] for v in per_layer.values()]))
            band_hi = float(np.max([v["haar_band"]["hi"] for v in per_layer.values()]))
            mean_sp = float(np.mean([v["spikiness"]["participation_ratio"]
                                     for v in per_layer.values()]))
            cells[f"{switch}/{opt_name}/s{seed}"] = {
                "switch": switch, "optimizer": opt_name, "seed": seed,
                "final_loss": final_loss, "mean_adoption": mean_ad, "haar_band_hi": band_hi,
                "adopted": bool(mean_ad > band_hi),
                "mean_participation_ratio": mean_sp, "per_layer": per_layer}
            print(f"[{a.arm}] {switch:20s} {opt_name:12s} s{seed} loss={final_loss:.4f} "
                  f"adoption={mean_ad:.4f} (haar hi {band_hi:.4f}) "
                  f"participation={mean_sp:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()
            M.s3_sync_incremental(out)

    at.check_dual_basis(["adoption_angle", "emergent_leverage_participation_ratio"])

    def mean_over_seeds(pred, key):
        v = [c[key] for c in cells.values() if pred(c)]
        return float(np.mean(v)) if v else None

    if a.arm == "M-10":
        adopted = sorted({f"{c['switch']}/{c['optimizer']}" for c in cells.values()
                          if c["adopted"]})
        gates.record("m10_a_switch_gets_the_block_adopted", bool(adopted),
                     adopted_cells=adopted,
                     adoption_by_cell={k: v["mean_adoption"] for k, v in cells.items()},
                     haar_band_hi={k: v["haar_band_hi"] for k, v in cells.items()},
                     note=("the refuter fires when nothing is adopted: the network ignores the "
                           "sanctioned block whatever the switch"))
        # The denominator ladder is the arm's own mechanism claim, so it gets its own reading.
        by_opt = {op: mean_over_seeds(lambda c, o=op: c["optimizer"] == o, "mean_adoption")
                  for op in ("adam", "block_adam", "scalar_adam")}
        gates.record("m10_adoption_rises_as_the_denominator_coarsens",
                     bool(by_opt["adam"] is not None
                          and by_opt["scalar_adam"] >= by_opt["adam"]),
                     adoption_by_optimizer=by_opt, block_size=a.block_size,
                     note=("block-Adam interpolates between ExactAdam and ScalarDenomAdam; if the "
                           "denominator's shape is what selects coordinates, coarsening it should "
                           "make a dictated block easier to adopt"))
    else:
        sp_by_opt = {op: mean_over_seeds(lambda c, o=op: c["optimizer"] == o,
                                         "mean_participation_ratio")
                     for op in ("adam", "sgd", "block_adam", "scalar_adam")}
        # A HIGHER participation ratio is FLATTER. The equivariant twin should be flatter than
        # Adam; the refuter fires if it is just as spiky.
        # A DIVERGED ARM HAS NO COMPARABLE SUBSPACE. Removing the per-coordinate denominator
        # changes the update scale by orders of magnitude, so scalar_adam at Adam's learning rate
        # can blow up — and the emergent subspace of a diverged run is the geometry of divergence,
        # not of training. Arms whose final loss exceeds a multiple of the reference's are recorded
        # DIVERGED and excluded from the ladder, with the losses on the row (defect M-D32).
        ref_loss = float(np.mean([c["final_loss"] for c in cells.values()
                                  if c["optimizer"] == "adam"]))
        loss_by_opt = {op: float(np.mean([c["final_loss"] for c in cells.values()
                                          if c["optimizer"] == op]))
                       for op in ("adam", "sgd", "block_adam", "scalar_adam")}
        diverged = [op for op, v in loss_by_opt.items() if v > DIVERGENCE_MULTIPLE * ref_loss]
        gates.record("m11_all_optimizer_arms_converged", not diverged,
                     diverged=diverged, final_loss_by_optimizer=loss_by_opt,
                     reference_loss=ref_loss, multiple=DIVERGENCE_MULTIPLE,
                     outcome=None if not diverged else M.VOID_INSTRUMENT,
                     note=("an arm that did not train has no comparable emergent subspace; the "
                           "ladder below is read over the converged arms only and names the rest"))
        equivariant_flat = bool(sp_by_opt["sgd"] > sp_by_opt["adam"]
                                and "sgd" not in diverged and "adam" not in diverged)
        gates.record("m11_equivariant_twin_is_flatter_than_adam", equivariant_flat,
                     participation_by_optimizer=sp_by_opt, diverged=diverged,
                     converged_ladder={op: sp_by_opt[op] for op in sp_by_opt
                                       if op not in diverged},
                     note=("SGD's update IS the gradient, so it has no preferred frame to imprint. "
                           "The refuter fires when this is False: spiky structure in the "
                           "equivariant twin would mean the grid theory is in trouble"))
        gates.record("m11_adam_is_spiky_at_all",
                     bool(sp_by_opt["adam"] < SPIKY_FRACTION * a.d_model),
                     participation_by_optimizer=sp_by_opt,
                     flat_participation_ratio=float(a.d_model),
                     threshold=SPIKY_FRACTION * a.d_model,
                     note=("the premise: the participation ratio's FLAT value is n (=d_model), "
                           "not 1, so 'spiky' means well below n. If Adam is already flat here "
                           "there is nothing for the twin to be flatter than"))
        sur.add(arm=a.arm, expected_ref=expected_ref, cell="sgd_vs_adam",
                statistic="emergent_leverage_participation_ratio",
                s=M.surprise(sp_by_opt["sgd"], sp_by_opt["adam"],
                             [sp_by_opt["block_adam"], sp_by_opt["scalar_adam"]]))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{a.arm}_result.json",
                 {"arm": a.arm, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "model": {"d_model": a.d_model, "n_layers": a.n_layers,
                                           "n_heads": a.n_heads, "vocab": len(tok),
                                           "seq_len": a.seq_len},
                  "steps": a.steps, "batch": a.batch, "lr": a.lr, "sgd_lr": a.sgd_lr,
                  "block_size": a.block_size, "n_snapshots": a.n_snapshots,
                  "seeds": seeds, "grid": [list(g) for g in grid], "optimizer": optinfo,
                  "sanctioned_block": "the first 32 coordinate axes", "cells": cells,
                  "scope": ("small models, switches on them; the question is about what training "
                            "does, not about scale")})
    print(f"{a.arm.replace('-', '')}_OK " + json.dumps(
        {"rows": info["n_rows"],
         "cells": {k: {"adoption": v["mean_adoption"], "pr": v["mean_participation_ratio"]}
                   for k, v in cells.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
