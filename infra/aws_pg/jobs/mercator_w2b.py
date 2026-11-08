"""M-B2 / M-B3 / M-B4 / M-B5 / M-B6 — the Wave-2 pretrain switches and their free riders.

Five arms on one harness, because each is a switch on the same pretrain and running them in one
job means the reference run they are all compared against is literally the same run.

    --arm M-B2   EVICT THE CORE, and scramble positions. Two independent switches:
                 `evict` projects every writer's output out of the emergent sink's coordinates at
                 every step, so the network is forbidden from writing there; `scramble` shuffles
                 position ids per batch, removing the positional structure the sink has been
                 suspected of carrying.
                 REFUTER: eviction does not move the emergent sink off the evicted block.

    --arm M-B3   TWO TENANTS + DISTILLATION INHERITANCE. An output-only student (trained on the
                 teacher's logits alone, never on its activations) is asked whether it grows its
                 sink at the SAME coordinates as its teacher.
                 REFUTER: student sink coordinates at chance overlap with teacher coordinates.

    --arm M-B4   CORE TRANSPLANT AT INIT. Initialise a fresh model with the teacher's sink
                 subspace planted in its writers, and ask whether it persists and whether
                 crystallization arrives sooner than with a matched-random transplant.
                 REFUTER: the transplant neither persists nor accelerates crystallization.

    --arm M-B5   CURRICULUM REVERSED. Quality data late instead of early, with the switch point
                 set by the MEASURED crystallization step of the reference run rather than by a
                 round number.
                 REFUTER: quality-late changes neither the sink nor the loss at matched budget.

    --arm M-B6   PER-BATCH GRADIENT-SINK ATTRIBUTION, the free rider: which batches drive the
                 sink, logged during whichever pretrain is running, against a shuffled-attribution
                 null.
                 REFUTER: no batch class scores above the shuffled null.

M-B6's ATTRIBUTION IS MEASURED BEFORE THE INTERVENTION IT SHARES A RUN WITH: the share is computed
from the gradient as produced, and only then is the eviction projection applied, so the logged
number is what the unmodified run would have produced at that step. Its null is a label shuffle on
the same series, so "which batches drive the sink" is answered against the same batches.

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

ARMS = ("M-B2", "M-B3", "M-B4", "M-B5", "M-B6")
SPIKE_MULT = 5.0


def sink_coords(U, mult: float = SPIKE_MULT):
    lev = H.leverage(U)
    return np.flatnonzero(lev > mult * (U.shape[1] / U.shape[0]))


def coordinate_overlap(a_coords, b_coords, n: int) -> dict:
    """Jaccard overlap of two coordinate sets, against the chance level for their sizes."""
    A, B = set(int(x) for x in a_coords), set(int(x) for x in b_coords)
    inter = len(A & B)
    union = len(A | B)
    chance = (len(A) * len(B) / n) / max(len(A) + len(B) - len(A) * len(B) / n, 1e-12)
    return {"jaccard": inter / union if union else 0.0, "n_a": len(A), "n_b": len(B),
            "n_shared": inter, "chance_jaccard": float(chance), "n": int(n)}


def main():
    ap = argparse.ArgumentParser(description="M-B2/B3/B4/B5/B6 — Wave-2 switches")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-snapshots", type=int, default=9)
    ap.add_argument("--n-docs", type=int, default=4096)
    ap.add_argument("--seeds", default="1337,1338")
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    meta = P.ARMS[a.arm]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w2") / "expectations.jsonl")
    expected_ref = exp.require(a.arm)
    at = M.Atlas(out, arm=a.arm, run_id=f"m_w2b_{a.arm.lower().replace('-', '_')}",
                 register=meta["register"], wave=meta["wave"], expected_ref=expected_ref,
                 queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)
    optinfo = M.check_optimizer(a.arm, "darwin_train.ExactAdam", offload_exact_fp32=False)

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, GemmaForCausalLM

    assert torch.cuda.is_available(), "no CUDA"
    tok = AutoTokenizer.from_pretrained("google/gemma-2b")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    # A STATED quality proxy: paragraphs long enough to be prose rather than headings or table
    # fragments. M-B5 is about WHEN quality data arrives, so the definition only has to be
    # consistent — it is written down here rather than tuned, and the counts are on the artifact.
    QUALITY_MIN_TOKENS = 128
    plain, quality, need = [], [], a.n_docs * a.seq_len
    for row in ds:
        t = str(row["text"]).strip()
        if not t:
            continue
        enc = tok(t, add_special_tokens=False).input_ids
        (quality if len(enc) >= QUALITY_MIN_TOKENS else plain).extend(enc)
        # STREAM UNTIL EACH HALF CAN COVER ITS OWN SHARE, not until the TOTAL is enough. Breaking on
        # the total lets the fast-filling half satisfy the condition while the other is still short:
        # the first run stopped with 145588 plain tokens against the 145920 the split then asked for
        # — 332 short, after a full corpus pass and a model load.
        if len(plain) >= need // 2 and len(quality) >= need // 2:
            break
    # QUALITY LAST, deliberately: M-B5 selects its late-curriculum batches as the tail windows, so
    # the ordering has to be a property of the array rather than of a filter applied at sample
    # time. Both halves are truncated proportionally so the tail is exactly the quality windows.
    n_quality = min(len(quality), need // 2) // a.seq_len * a.seq_len
    n_plain = need - n_quality
    if len(plain) < n_plain:
        # The corpus ran out of short documents before it ran out of long ones. Take the shortfall
        # from the quality half rather than failing: the split is a design target, and the REALIZED
        # fraction is what M-B5's tail-window selection needs to know, so it is printed and banked
        # rather than assumed to be one half.
        n_plain = len(plain) // a.seq_len * a.seq_len
        n_quality = need - n_plain
    assert len(plain) >= n_plain and len(quality) >= n_quality, (
        f"corpus too small: {len(plain)} plain and {len(quality)} quality tokens cannot fill "
        f"{n_plain} + {n_quality} = {need}; raise --n-docs' corpus or lower it")
    pool = plain[:n_plain] + quality[:n_quality]
    data = torch.tensor(pool, dtype=torch.long).reshape(-1, a.seq_len)
    n_quality_windows = n_quality // a.seq_len
    print(f"[{a.arm}] {data.shape[0]} windows, {n_quality_windows} of them 'quality'", flush=True)

    def build(seed, transplant=None):
        torch.manual_seed(seed)
        cfg = PT.tiny_config(len(tok), a.d_model, a.n_layers, a.n_heads, max_positions=a.seq_len)
        model = GemmaForCausalLM(cfg).to("cuda")
        if transplant is not None:
            # Plant a subspace in the writers: give each writer a component that writes INTO the
            # transplanted subspace, matched in Frobenius norm to what it already has there.
            Ub = torch.as_tensor(np.ascontiguousarray(transplant), device="cuda",
                                 dtype=torch.float32)
            g = torch.Generator(device="cpu").manual_seed(seed + 77)
            with torch.no_grad():
                for layer in model.model.layers:
                    for w in (layer.self_attn.o_proj, layer.mlp.down_proj):
                        W = w.weight.data.to(torch.float32)
                        cur = float((Ub.T @ W).norm())
                        C = torch.randn(Ub.shape[1], W.shape[1], generator=g).to("cuda")
                        C = C / C.norm() * max(cur, 1e-6)
                        w.weight.data.copy_((W - Ub @ (Ub.T @ W) + Ub @ C).to(w.weight.dtype))
        return model

    def trainable(model):
        return {("q_proj", i): layer.self_attn.q_proj
                for i, layer in enumerate(model.model.layers)}

    def writers(model):
        out_ = []
        for layer in model.model.layers:
            out_.extend([layer.self_attn.o_proj, layer.mlp.down_proj])
        return out_

    def pretrain(model, seed, *, evict=None, scramble=False, quality_late=None,
                 attribution=True):
        from infra.pg import darwin_train as T
        mods = trainable(model)
        keys = sorted(mods)
        opt = T.ExactAdam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
        rng = np.random.default_rng(H.cell_seed(seed, a.arm, str(evict is not None),
                                                str(scramble)))
        every = max(1, a.steps // (a.n_snapshots - 1))
        # np.ascontiguousarray because these bases arrive as REVERSED eigenvector views (ascending
        # eigenvalues read from the small end), and a negative-stride array cannot be wrapped as a
        # tensor at all — torch says so and stops. Fixed here first (M-D56) and then again at the
        # transplant site three steps later (M-D65): the hazard belongs to every basis this job
        # takes from an eigendecomposition, not to one call.
        Ue = (torch.as_tensor(np.ascontiguousarray(evict), device="cuda", dtype=torch.float32)
              if evict is not None else None)
        snaps, losses, attrib = [], [], []
        model.train()
        for step in range(a.steps + 1):
            if step % every == 0 or step == a.steps:
                snaps.append({k: mods[k].weight.detach().to(torch.float32).cpu().numpy()
                              .astype(np.float64) for k in keys})
            if step == a.steps:
                break
            # M-B5's curriculum: quality windows late instead of early, switching at the step the
            # REFERENCE run's crystallization was measured at.
            if quality_late is None:
                idx = rng.integers(0, data.shape[0], a.batch)
            elif step < quality_late:
                idx = rng.integers(0, max(data.shape[0] - n_quality_windows, 1), a.batch)
            else:
                idx = rng.integers(max(data.shape[0] - n_quality_windows, 0), data.shape[0],
                                   a.batch)
            b = data[idx].to("cuda")
            kw = {}
            if scramble:
                pos = torch.stack([torch.randperm(b.shape[1], device="cuda")
                                   for _ in range(b.shape[0])])
                kw["position_ids"] = pos
            o = model(input_ids=b, labels=b, **kw)
            opt.zero_grad()
            o.loss.backward()
            if attribution and Ue is not None:
                # M-B6 rides here: how much of THIS batch's gradient points into the tracked
                # subspace, logged per step with the batch's mean length as its class label.
                with torch.no_grad():
                    num = sum(float(((m.weight.grad.to(torch.float32) @ Ue) ** 2).sum())
                              for m in mods.values())
                    den = sum(float((m.weight.grad.to(torch.float32) ** 2).sum())
                              for m in mods.values())
                attrib.append({"step": step, "share": num / max(den, 1e-30),
                               "batch_class": int(np.median(idx) >= data.shape[0]
                                                  - n_quality_windows)})
            if Ue is not None:
                with torch.no_grad():
                    for w in writers(model):
                        W = w.weight.data.to(torch.float32)
                        w.weight.data.copy_((W - Ue @ (Ue.T @ W)).to(w.weight.dtype))
            opt.step()
            losses.append(float(o.loss.detach()))
            if (step + 1) % 500 == 0:
                print(f"    step {step+1}/{a.steps} loss={np.mean(losses[-100:]):.4f}", flush=True)
        return {"snaps": snaps, "keys": keys, "final_loss": float(np.mean(losses[-100:])),
                "attribution": attrib}

    def emergent(res):
        cons = PT.consensus_from_snapshots(res["snaps"], res["keys"], H.BANK_K)
        return cons, {k: sink_coords(cons[k]["U"]) for k in res["keys"]}

    # ---- the reference run, shared by every arm in this job ------------------------------------
    cells = {}
    ref_by_seed = {}
    for seed in seeds:
        model = build(seed)
        ref = pretrain(model, seed, attribution=False)
        cons, coords = emergent(ref)
        ref_by_seed[seed] = {"cons": cons, "coords": coords, "final_loss": ref["final_loss"],
                             "keys": ref["keys"]}
        cells[f"reference/s{seed}"] = {"final_loss": ref["final_loss"],
                                       "n_sink_coords": {f"L{k[1]}": int(len(v))
                                                         for k, v in coords.items()}}
        print(f"[{a.arm}] reference s{seed}: loss={ref['final_loss']:.4f} sink coords "
              f"{[len(v) for v in coords.values()]}", flush=True)
        del model
        torch.cuda.empty_cache()
        M.s3_sync_incremental(out)

    # ---- DID THE SUBSTRATE GROW A SINK AT ALL? (defect M-D71) ----------------------------------
    # M-B2, M-B3 and M-B4 all compare SETS OF SINK COORDINATES — evicted against emergent,
    # student against teacher, transplanted against random. The first Wave-2 run produced
    # `sink coords [0, 0, 0, 0]` for every arm, every seed and every layer: no coordinate anywhere
    # exceeded the spike criterion. Comparing two empty sets yields Jaccard 0.0 against a chance
    # level of 0.0, and all three arms recorded that as their refuter FIRING — a decided negative
    # where nothing had been measured. A false negative in the ledger is worse than a crash.
    #
    # The substrate is a 4-layer, 256-wide model trained for a few thousand steps. That it grows no
    # spiky sink is itself informative — and it makes every sink-INTERVENTION arm untestable here,
    # which is a coverage fact about the wave rather than a null result about the hypotheses.
    n_coords = {f"s{seed}": [int(len(v)) for v in ref_by_seed[seed]["coords"].values()]
                for seed in seeds}
    grew_a_sink = any(sum(v) > 0 for v in n_coords.values())
    gates.record("w2_substrate_grew_a_sink", grew_a_sink, n_sink_coords=n_coords,
                 spike_multiple=SPIKE_MULT, d_model=a.d_model, n_layers=a.n_layers,
                 steps=a.steps,
                 outcome=None if grew_a_sink else M.VOID_INSTRUMENT,
                 note=("the coordinate-set arms (M-B2/B3/B4) compare sink coordinates; with none "
                       "found, a Jaccard of 0 against a chance of 0 is an undefined comparison "
                       "and not evidence about eviction, inheritance or transplantation"))
    COORD_SET_ARMS = ("M-B2", "M-B3", "M-B4")
    if not grew_a_sink and a.arm in COORD_SET_ARMS:
        # A VOID IS A RESULT, NOT A CRASH. Fail-loud is for defects; VOID-INSTRUMENT is one of the
        # program's own outcome literals, and asserting it took the whole five-step pipeline down
        # at step 1 — losing M-B5 and M-B6, which do not touch sink coordinates and had already
        # produced real results (M-D90). The arm records its verdict, writes its artifact, and
        # exits 0 so the arms behind it run.
        print(f"[{a.arm}] VOID-INSTRUMENT: no sink coordinates in any reference run ({n_coords}) "
              f"at spike multiple {SPIKE_MULT} on a {a.n_layers}x{a.d_model} model over {a.steps} "
              f"steps. This substrate has no sink to intervene on; the comparison this arm makes "
              f"would report empty-vs-empty as a fired refuter.", flush=True)
        info = at.to_parquet()
        sur.to_parquet()
        M.write_json(out, f"{a.arm}_result.json",
                     {"arm": a.arm, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                      "expected_ref": expected_ref, "provenance": M.provenance(),
                      "utc": M.utc_now(), "atlas": info, "outcome": M.VOID_INSTRUMENT,
                      "n_sink_coords": n_coords, "spike_multiple": SPIKE_MULT,
                      "d_model": a.d_model, "n_layers": a.n_layers, "steps": a.steps,
                      "why": ("the substrate grew no spiky sink, so a coordinate-set comparison "
                              "has nothing to compare")})
        print(f"{a.arm.replace('-', '')}_OK " + json.dumps({"outcome": M.VOID_INSTRUMENT,
                                                            "n_sink_coords": n_coords}))
        return 0

    # ---- the arm's own switch -----------------------------------------------------------------
    results = {}
    for seed in seeds:
        ref = ref_by_seed[seed]
        # Writers write into the RESIDUAL STREAM, which is one space shared by every layer, so
        # the evicted block has to be a single stream-level subspace — the consensus of the
        # per-layer emergent sinks, not layer 0's, which would evict one layer's answer.
        U0 = {k: ref["cons"][k]["U"] for k in ref["keys"]}
        S = np.zeros((a.d_model, a.d_model))
        for k in ref["keys"]:
            S += U0[k] @ U0[k].T
        _w, V = np.linalg.eigh(S)
        U_stream = V[:, ::-1][:, :H.BANK_K]
        if a.arm == "M-B2":
            for switch in ("evict", "scramble", "evict+scramble"):
                model = build(seed)
                r = pretrain(model, seed,
                             evict=(U_stream if "evict" in switch else None),
                             scramble=("scramble" in switch))
                _c, coords = emergent(r)
                ov = {f"L{k[1]}": coordinate_overlap(coords[k], ref["coords"][k], a.d_model)
                      for k in ref["keys"]}
                mean_j = float(np.mean([v["jaccard"] for v in ov.values()]))
                chance = float(np.mean([v["chance_jaccard"] for v in ov.values()]))
                results[f"{switch}/s{seed}"] = {"switch": switch, "seed": seed,
                                                "final_loss": r["final_loss"],
                                                "overlap": ov, "mean_jaccard": mean_j,
                                                "chance_jaccard": chance,
                                                "moved_off": bool(mean_j < chance)}
                at.add("sink_coordinate_overlap", value=mean_j, bank_id="emergent_bottom32",
                       flags={"switch": switch, "seed": seed, "chance": chance,
                              "final_loss": r["final_loss"], "reference": "unswitched run",
                              "optimizer": optinfo})
                print(f"[M-B2] {switch:16s} s{seed} loss={r['final_loss']:.4f} "
                      f"jaccard={mean_j:.4f} (chance {chance:.4f})", flush=True)
                del model
                torch.cuda.empty_cache()
        elif a.arm == "M-B4":
            S = np.zeros((a.d_model, a.d_model))
            for k in ref["keys"]:
                S += ref["cons"][k]["U"] @ ref["cons"][k]["U"].T
            _w, V = np.linalg.eigh(S)
            U_stream = V[:, ::-1][:, :H.BANK_K]
            for kind, U in (("core", U_stream),
                            ("random", H.haar_subspace(a.d_model, H.BANK_K, seed + 500))):
                model = build(seed, transplant=U)
                r = pretrain(model, seed, attribution=False)
                _c, coords = emergent(r)
                planted = sink_coords(U)
                ov = {f"L{k[1]}": coordinate_overlap(coords[k], planted, a.d_model)
                      for k in ref["keys"]}
                mean_j = float(np.mean([v["jaccard"] for v in ov.values()]))
                results[f"transplant_{kind}/s{seed}"] = {
                    "kind": kind, "seed": seed, "final_loss": r["final_loss"],
                    "overlap": ov, "mean_jaccard": mean_j,
                    "chance": float(np.mean([v["chance_jaccard"] for v in ov.values()]))}
                at.add("transplant_persistence", value=mean_j,
                       bank_id="core32" if kind == "core" else "haar32_s0",
                       flags={"kind": kind, "seed": seed, "final_loss": r["final_loss"],
                              "chance": results[f"transplant_{kind}/s{seed}"]["chance"]})
                print(f"[M-B4] transplant {kind:7s} s{seed} loss={r['final_loss']:.4f} "
                      f"persistence jaccard={mean_j:.4f}", flush=True)
                del model
                torch.cuda.empty_cache()
        elif a.arm == "M-B5":
            switch_at = a.steps // 2
            for kind, ql in (("quality_early", None), ("quality_late", switch_at)):
                model = build(seed)
                r = pretrain(model, seed, quality_late=ql, attribution=False)
                _c, coords = emergent(r)
                ov = {f"L{k[1]}": coordinate_overlap(coords[k], ref["coords"][k], a.d_model)
                      for k in ref["keys"]}
                results[f"{kind}/s{seed}"] = {
                    "kind": kind, "seed": seed, "final_loss": r["final_loss"],
                    "switch_step": ql, "mean_jaccard": float(np.mean(
                        [v["jaccard"] for v in ov.values()]))}
                at.add("curriculum_final_loss", value=r["final_loss"], dose=float(ql or 0),
                       flags=results[f"{kind}/s{seed}"])
                print(f"[M-B5] {kind:14s} s{seed} loss={r['final_loss']:.4f}", flush=True)
                del model
                torch.cuda.empty_cache()
        elif a.arm == "M-B6":
            # PER-BATCH ATTRIBUTION, against a SHUFFLED-LABEL null. The tracked subspace is the
            # reference run's own emergent sink, so the question is which batches drove the thing
            # that actually emerged rather than a subspace chosen in advance.
            S = np.zeros((a.d_model, a.d_model))
            for k in ref["keys"]:
                S += ref["cons"][k]["U"] @ ref["cons"][k]["U"].T
            _w, V = np.linalg.eigh(S)
            U_stream = V[:, ::-1][:, :H.BANK_K]
            model = build(seed)
            # `evict` is what turns attribution logging on in `pretrain`, and the projection is
            # applied AFTER the share is measured, so the logged number is the gradient the
            # unmodified run would have produced at that step.
            r = pretrain(model, seed, evict=U_stream, attribution=True)
            att = r["attribution"]
            assert att, "attribution logging produced no rows"
            shares = np.array([x["share"] for x in att], dtype=np.float64)
            labels = np.array([x["batch_class"] for x in att], dtype=int)
            classes = sorted(set(labels.tolist()))
            obs = {int(c): float(shares[labels == c].mean()) for c in classes}
            gap = (max(obs.values()) - min(obs.values())) if len(obs) > 1 else 0.0
            rng2 = np.random.default_rng(H.cell_seed(seed, a.arm, "shuffle", "null"))
            null = []
            for _ in range(2000):
                perm = rng2.permutation(labels)
                m_ = {c: shares[perm == c].mean() for c in classes}
                null.append(max(m_.values()) - min(m_.values()) if len(m_) > 1 else 0.0)
            p = float((np.asarray(null) >= gap).mean())
            results[f"attribution/s{seed}"] = {
                "seed": seed, "n_steps_logged": len(att), "share_by_class": obs,
                "observed_gap": float(gap), "p_value": p,
                "beats_shuffled_null": bool(p < 0.05),
                "class_meaning": "1 = a batch drawn from the quality tail, 0 = plain",
                "final_loss": r["final_loss"]}
            at.add("batch_attribution_gap", value=float(gap), bank_id="emergent_bottom32",
                   n=len(att),
                   flags={**results[f"attribution/s{seed}"],
                          "null": "labels shuffled 2000 times, the gap recomputed each time"})
            print(f"[M-B6] s{seed}: share by class {obs} gap={gap:.5f} p={p:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()
        else:                                              # M-B3
            from infra.pg import darwin_train as T
            teacher = build(seed)
            tr = pretrain(teacher, seed, attribution=False)
            _c, t_coords = emergent(tr)
            student = build(seed + 999)
            s_mods = trainable(student)
            s_keys = sorted(s_mods)
            opt = T.ExactAdam([p for p in student.parameters() if p.requires_grad], lr=a.lr)
            rng = np.random.default_rng(H.cell_seed(seed, a.arm, "student", "distill"))
            every = max(1, a.steps // (a.n_snapshots - 1))
            snaps = []
            teacher.eval()
            student.train()
            for step in range(a.steps + 1):
                if step % every == 0 or step == a.steps:
                    snaps.append({k: s_mods[k].weight.detach().to(torch.float32).cpu()
                                  .numpy().astype(np.float64) for k in s_keys})
                if step == a.steps:
                    break
                b = data[rng.integers(0, data.shape[0], a.batch)].to("cuda")
                with torch.inference_mode():
                    tl = torch.log_softmax(teacher(input_ids=b).logits.float(), dim=-1)
                sl = torch.log_softmax(student(input_ids=b).logits.float(), dim=-1)
                loss = (torch.exp(tl) * (tl - sl)).sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            s_cons = PT.consensus_from_snapshots(snaps, s_keys, H.BANK_K)
            s_coords = {k: sink_coords(s_cons[k]["U"]) for k in s_keys}
            ov = {f"L{k[1]}": coordinate_overlap(s_coords[k], t_coords[k], a.d_model)
                  for k in s_keys}
            mean_j = float(np.mean([v["jaccard"] for v in ov.values()]))
            chance = float(np.mean([v["chance_jaccard"] for v in ov.values()]))
            results[f"distilled/s{seed}"] = {"seed": seed, "overlap": ov,
                                             "mean_jaccard": mean_j, "chance_jaccard": chance,
                                             "inherits": bool(mean_j > chance)}
            at.add("student_teacher_sink_overlap", value=mean_j, bank_id="emergent_bottom32",
                   flags={"seed": seed, "chance": chance,
                          "distillation": "output-only (logit KL); the student never sees the "
                                          "teacher's activations"})
            print(f"[M-B3] s{seed}: student-teacher sink jaccard={mean_j:.4f} "
                  f"(chance {chance:.4f})", flush=True)
            del teacher, student
            torch.cuda.empty_cache()
        M.s3_sync_incremental(out)

    # ---- the arm's gate -----------------------------------------------------------------------
    if a.arm == "M-B2":
        moved = [k for k, v in results.items() if v["moved_off"]]
        gates.record("b2_eviction_moves_the_sink_off_the_evicted_block", bool(moved),
                     moved_cells=moved,
                     jaccard={k: v["mean_jaccard"] for k, v in results.items()},
                     chance={k: v["chance_jaccard"] for k, v in results.items()},
                     note=("the refuter fires when eviction does NOT move it: a sink that "
                           "reappears on the same coordinates after being forbidden is not being "
                           "chosen by the writers"))
    elif a.arm == "M-B3":
        js = [v["mean_jaccard"] for v in results.values()]
        ch = [v["chance_jaccard"] for v in results.values()]
        gates.record("b3_student_inherits_teacher_sink_coordinates",
                     bool(np.mean(js) > np.mean(ch)), jaccard=js, chance=ch,
                     note=("output-only distillation: the student never sees the teacher's "
                           "activations, so a shared address has to come through behaviour"))
        sur.add(arm=a.arm, expected_ref=expected_ref, cell="distilled",
                statistic="student_teacher_sink_overlap",
                s=M.surprise(float(np.mean(js)), float(np.mean(ch)), ch))
    elif a.arm == "M-B4":
        core = [v["mean_jaccard"] for k, v in results.items() if v["kind"] == "core"]
        rand = [v["mean_jaccard"] for k, v in results.items() if v["kind"] == "random"]
        gates.record("b4_core_transplant_persists_more_than_random",
                     bool(np.mean(core) > np.mean(rand)),
                     core=core, random=rand,
                     note="matched-random transplant is the control the refuter names")
    elif a.arm == "M-B6":
        ps = [v["p_value"] for v in results.values()]
        gates.record("b6_a_batch_class_beats_the_shuffled_null",
                     bool(min(ps) < 0.05), p_values=ps,
                     share_by_class={k: v["share_by_class"] for k, v in results.items()},
                     note=("no direction was pre-logged for WHICH class; inventing one would be "
                           "the finding, so only the existence of a difference is gated"))
    elif a.arm == "M-B5":
        early = [v["final_loss"] for v in results.values() if v["kind"] == "quality_early"]
        late = [v["final_loss"] for v in results.values() if v["kind"] == "quality_late"]
        gates.record("b5_curriculum_order_changes_the_loss",
                     bool(abs(np.mean(early) - np.mean(late)) > 0),
                     quality_early=early, quality_late=late,
                     note=("at matched budget; the refuter is that order changes neither the sink "
                           "nor the loss"))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{a.arm}_result.json",
                 {"arm": a.arm, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info,
                  "model": {"d_model": a.d_model, "n_layers": a.n_layers, "n_heads": a.n_heads,
                            "vocab": len(tok), "seq_len": a.seq_len},
                  "steps": a.steps, "batch": a.batch, "lr": a.lr, "seeds": seeds,
                  "optimizer": optinfo, "n_quality_windows": n_quality_windows,
                  "quality_min_tokens": QUALITY_MIN_TOKENS,
                  "reference": cells, "results": results,
                  "scope": "small models, switches on them"})
    print(f"{a.arm.replace('-', '')}_OK " + json.dumps({"rows": info["n_rows"],
                                                        "cells": len(results)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
