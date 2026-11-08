"""PLENUM Arm 2 — the feature layer: does the install/consumption show up in features at all?

This is the row LACUNA left blocked. It terminates as MEASURED or as VOID — never as "blocked"
(§1.4 E1). VOID here is an honest terminal state reached by a NAMED gate, not a failure to try.

Streams (§2.2): gemma-2b text · PG1 text-shell (blank image) · PG1 mm (real images) · π0.5 mm
re-shell · π0 mm re-shell (measured last, gibberish-adjacent, expected VOID-prone — the gates
decide, not an expectation).

Featurizer: the house BatchTopK (`grain_bsf`, m=1) with **freeze_b_dec=True** — REGRAIN showed a
learned b_dec voids the block cone. Two seeds per stream, because "the same feature" is only
definable against a cross-seed identity floor.

Gates, all pre-registered and all able to fail (§2.2):
  G_dead      dead-latent fraction ≤ 0.5                       (`block_quality`)
  G_identity  same-stream seeds agree with EACH OTHER by a real margin over the cross-STREAM match
              — discrimination AND self-consistency, per the G0 amendment
  G_behavior  splicing the SAE reconstruction into the forward preserves the stream's OWN battery
              score within 10pp — the FLENS lesson: gate on behavior, never on variance-EV

No em-random-subspace null anywhere (REGRAIN: it saturates for low-rank high-energy objects), and
R² is never used to rank featurizers.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.plenum_sae --type g6e.4xlarge --no-weights \
      --args "--streams pg1_mm,pg1_text,gemma2b,pi05_mm --sites 9,14" \
      --s3-prefix results/plenum_sae --budget-min 700
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

PPL_ANCHOR = 22.42583
PPL_REL_TOL = 1e-3
N_ACT = 200_000          # activation rows per stream (house SAE training scale)
N_BEHAV = 300            # items for the behavioral gate


def _matched_cos(Da, Db):
    """Mean over features of A of the best cosine with any feature of B (unit-norm decoder cols).

    This is the ONE definition of "the same feature" used everywhere in this arm: as the cross-SEED
    identity floor, as the cross-STREAM comparison, and inside `emergent_frac`. Using one definition
    is what makes the emergent fraction interpretable — a fraction below a floor measured a
    different way would mean nothing."""
    import torch
    A = torch.nn.functional.normalize(Da.T, dim=1)      # (W, d)
    B = torch.nn.functional.normalize(Db.T, dim=1)
    best = torch.empty(A.shape[0])
    for s in range(0, A.shape[0], 2048):
        best[s:s + 2048] = (A[s:s + 2048] @ B.T).max(dim=1).values.cpu()
    return best


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", default="pg1_mm,pg1_text,gemma2b,pi05_mm,pi0_mm")
    ap.add_argument("--sites", default="9,14")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--n-act", type=int, default=N_ACT)
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    streams = [s.strip() for s in a.streams.split(",") if s.strip()]
    sites = [int(s) for s in a.sites.split(",") if s.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs.invariant_core_func import _blank_image
    from infra.pg.plenum_core import score_item_fc as score_item
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_core as PC
    from infra.pg import plenum_units as PU

    assert torch.cuda.is_available(), "plenum_sae needs GPU"
    device = "cuda"
    t0 = time.time()
    rep = {"object": "plenum_sae_v1", "streams": streams, "sites": sites, "seeds": seeds,
           "featurizer": "grain_bsf BatchTopK m=1, freeze_b_dec=True (REGRAIN)",
           "gpu": torch.cuda.get_device_name(0),
           "no_nulls": "no em-random-subspace null (REGRAIN); R² never ranks featurizers"}
    cells, gates_out = [], {}
    rep["cells"], rep["gates"] = cells, gates_out

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    tasks = LS.mm_tasks()
    vqa = tasks["vqav2"]
    items, _h = IP.load_items(vqa, max(N_BEHAV, 600))
    choices = PC.choices_for_task("vqav2", items, 20260728)
    gray = _blank_image(128)

    model, proc = LS.load_pg1(device=device)
    pg1_sd = LS.pg1_baseline_sd(model)
    ppl = IP.ppl_gate(model, proc, device)
    val = float(ppl["ppl"] if isinstance(ppl, dict) else ppl)
    rel = abs(val - PPL_ANCHOR) / PPL_ANCHOR
    rep["P2_ppl_anchor"] = {"ppl": val, "rel_diff": rel, "passes": bool(rel < PPL_REL_TOL)}
    assert rep["P2_ppl_anchor"]["passes"], f"P2 FAIL: {val}"
    layers = IP.decoder_layers(model)
    unit_seeds = {"stitch": 1, "matched_dw": 20260727, "rand_dec": 1337}
    _w()

    STREAM_UNIT = {"pg1_mm": "pg1", "pg1_text": "pg1", "gemma2b": "gemma-2b",
                   "pi05_mm": "pi05", "pi0_mm": "pi0"}

    def capture(stream, site, n_rows):
        """Residual rows entering layer[site] for `stream`, as a (N, d) CPU fp32 matrix."""
        acc, handles = [], []

        def hook(m, inp, o):
            t = (o[0] if isinstance(o, tuple) else o).detach()
            acc.append(t.reshape(-1, t.shape[-1]).to("cpu", torch.float32))
        handles.append(layers[site].register_forward_hook(hook))
        img_real = stream.endswith("_mm")
        with torch.inference_mode():
            for r in items:
                if sum(x.shape[0] for x in acc) >= n_rows:
                    break
                img = vqa.image_fn(r).convert("RGB") if img_real else gray
                # same prompt form the behavioural gate scores on, so the dictionary is
                # trained on the distribution it is gated against
                enc = proc(text=[PC.fc_prompt(vqa.prompt_fn(r))], images=[img],
                           return_tensors="pt").to(device)
                model(**{k: (v.to(torch.bfloat16) if k == "pixel_values" else v)
                         for k, v in enc.items()})
        for h in handles:
            h.remove()
        X = torch.cat(acc, dim=0)[:n_rows]
        assert X.shape[0] > 10_000, f"{stream}@L{site}: only {X.shape[0]} activation rows"
        return X

    def behavioral_gate(site, ae):
        """Splice the SAE reconstruction into the residual at `site` and re-read the battery."""
        native, patched = [], []
        for r, (cands, gold) in list(zip(items, choices))[:N_BEHAV]:
            img = vqa.image_fn(r).convert("RGB")
            s, nt = score_item(model, proc, device, img, PC.fc_prompt(vqa.prompt_fn(r)), cands)
            native.append(int(int((np.asarray(s) / np.asarray(nt, float)).argmax()) == gold))
        aeg = ae.to(device).eval()

        def hook(m, inp, o):
            t = o[0] if isinstance(o, tuple) else o
            sh = t.shape
            x = t.reshape(-1, sh[-1]).float()
            rec = aeg.decode(aeg.encode(x, use_threshold=True)).reshape(sh).to(t.dtype)
            return (rec,) + tuple(o[1:]) if isinstance(o, tuple) else rec
        h = layers[site].register_forward_hook(hook)
        for r, (cands, gold) in list(zip(items, choices))[:N_BEHAV]:
            img = vqa.image_fn(r).convert("RGB")
            s, nt = score_item(model, proc, device, img, PC.fc_prompt(vqa.prompt_fn(r)), cands)
            patched.append(int(int((np.asarray(s) / np.asarray(nt, float)).argmax()) == gold))
        h.remove()
        aeg.to("cpu")
        return float(np.mean(native)), float(np.mean(patched))

    dicts = {}
    for stream in streams:
        uid = STREAM_UNIT[stream]
        if uid == "gemma-2b":
            # gemma-2b is a different architecture from the PG1 shell, so `install_unit` cannot load
            # it and an earlier version of this job SKIPPED it — leaving the §2.2 pre-install feature
            # stream unmeasured. It loads natively as a causal LM, and its residual stream is
            # captured from its own decoder layers. The behavioural gate becomes text-native
            # (perplexity under reconstruction patching) because the mm battery does not apply to a
            # unit with no vision path; the FLENS rule is "gate on behaviour", not "gate on the mm
            # battery specifically".
            from transformers import AutoModelForCausalLM, AutoTokenizer
            gm = AutoModelForCausalLM.from_pretrained(
                LS.GEMMA1_REPO, torch_dtype=torch.bfloat16, device_map=device,
                attn_implementation="eager").eval()
            gtok = AutoTokenizer.from_pretrained(LS.GEMMA1_REPO)
            glayers = gm.model.layers
            from infra.pg import text_eval as TE
            for site in sites:
                acc, handles = [], []

                def _h(m, inp, o):
                    t = (o[0] if isinstance(o, tuple) else o).detach()
                    acc.append(t.reshape(-1, t.shape[-1]).to("cpu", torch.float32))
                handles.append(glayers[site].register_forward_hook(_h))
                # STIMULUS. The first version fed this stream `vqa.prompt_fn(r)` — the same short
                # VQA question strings the mm streams use — and ran out of items at 6101 rows,
                # under the frozen 10k minimum. Reusing the mm stimulus was wrong on its own
                # terms anyway: §2.2 names this stream "`gemma-2b` text", and a text stream's
                # residuals should be harvested from text. Wikitext windows give 2048 rows each,
                # so the row floor is met with an order of magnitude of headroom, and the
                # activations are the ones the stream is defined over.
                from datasets import load_dataset
                _ds = load_dataset(TE.WIKITEXT_REPO, TE.WIKITEXT_CONFIG, split="train")
                _txt = "\n\n".join(t for t in _ds["text"] if t.strip())
                _ids = gtok(_txt, return_tensors="pt", add_special_tokens=False).input_ids[0]
                _win = 2048
                with torch.inference_mode():
                    for _st in range(0, _ids.size(0) - 1, _win):
                        if sum(x.shape[0] for x in acc) >= a.n_act:
                            break
                        _chunk = _ids[_st:_st + _win]
                        if _chunk.size(0) < 2:
                            break
                        gm(input_ids=_chunk.unsqueeze(0).to(device))
                for h in handles:
                    h.remove()
                X = torch.cat(acc, dim=0)[: a.n_act]
                assert X.shape[0] > 10_000, f"gemma-2b@L{site}: only {X.shape[0]} rows"
                ppl_native = TE.clean_text_perplexity(gm, gtok, device, max_tokens=20_000)["ppl"]
                ae_first = None
                for sd_seed in seeds:
                    ae, _cfg = GB.train_block_sae(X, m=1, seed=sd_seed, device=device,
                                                  steps=a.steps, freeze_b_dec=True)
                    if ae_first is None:
                        ae_first = ae
                    q = GB.block_quality(ae, X, device)
                    dicts[(stream, site, sd_seed)] = ae.decoder.weight.detach().cpu()
                    cells.append({"stream": stream, "site": site, "seed": sd_seed,
                                  "dead_latent_frac": q["dead_latent_frac"], "fvu": q["fvu"],
                                  "mean_block_l0": q["mean_block_l0"]})
                    _w()
                aeg = ae_first.to(device).eval()

                def _patch(m, inp, o):
                    t = o[0] if isinstance(o, tuple) else o
                    sh = t.shape
                    rec = aeg.decode(aeg.encode(t.reshape(-1, sh[-1]).float(),
                                                use_threshold=True)).reshape(sh).to(t.dtype)
                    return (rec,) + tuple(o[1:]) if isinstance(o, tuple) else rec
                ph = glayers[site].register_forward_hook(_patch)
                ppl_patched = TE.clean_text_perplexity(gm, gtok, device, max_tokens=20_000)["ppl"]
                ph.remove()
                aeg.to("cpu")
                same = float(_matched_cos(dicts[(stream, site, seeds[0])],
                                          dicts[(stream, site, seeds[1])]).mean())
                other = [k for k in dicts if k[1] == site and k[0] != stream]
                cross = (float(_matched_cos(dicts[(stream, site, seeds[0])],
                                            dicts[other[0]]).mean()) if other else 0.0)
                # text-native behavioural gate: relative perplexity change under patching
                rel = abs(ppl_patched - ppl_native) / max(ppl_native, 1e-9)
                g = PC.sae_gates(dead_frac=min(c["dead_latent_frac"] for c in cells
                                               if c["stream"] == stream and c["site"] == site),
                                 same_seed_cos=same, cross_stream_cos=cross,
                                 behav_native=0.0, behav_patched=rel)
                g["behaviour_metric"] = ("relative wikitext-ppl change under reconstruction "
                                         "patching (text-native: this unit has no vision path)")
                g["ppl_native"], g["ppl_patched"] = ppl_native, ppl_patched
                gates_out[f"{stream}@L{site}"] = g
                print(f"[SAE] GATES {stream}@L{site}: {g['note']}", flush=True)
                _w()
            del gm
            torch.cuda.empty_cache()
            continue
        PU.install_unit(uid, model, proc, pg1_sd, device, unit_seeds)
        for site in sites:
            X = capture(stream, site, a.n_act)
            print(f"[SAE] {stream}@L{site}: X={tuple(X.shape)} t={time.time()-t0:.0f}s", flush=True)
            ae_first = None
            for sd_seed in seeds:
                ae, cfg = GB.train_block_sae(X, m=1, seed=sd_seed, device=device,
                                             steps=a.steps, freeze_b_dec=True)
                if ae_first is None:
                    ae_first = ae            # the behavioural gate reads the FIRST seed's dictionary
                q = GB.block_quality(ae, X, device)
                dicts[(stream, site, sd_seed)] = ae.decoder.weight.detach().cpu()
                cells.append({"stream": stream, "site": site, "seed": sd_seed,
                              "dead_latent_frac": q["dead_latent_frac"], "fvu": q["fvu"],
                              "mean_block_l0": q["mean_block_l0"],
                              "elapsed_s": round(time.time() - t0, 1)})
                print(f"[SAE] {stream}@L{site} s{sd_seed}: dead={q['dead_latent_frac']:.3f} "
                      f"fvu={q['fvu']:.4f}", flush=True)
                _w()
            # gates for this (stream, site)
            same = float(_matched_cos(dicts[(stream, site, seeds[0])],
                                      dicts[(stream, site, seeds[1])]).mean())
            other = [k for k in dicts if k[1] == site and k[0] != stream]
            cross = (float(_matched_cos(dicts[(stream, site, seeds[0])],
                                        dicts[other[0]]).mean()) if other else 0.0)
            nat, pat = behavioral_gate(site, ae_first)
            g = PC.sae_gates(dead_frac=min(c["dead_latent_frac"] for c in cells
                                           if c["stream"] == stream and c["site"] == site),
                             same_seed_cos=same, cross_stream_cos=cross,
                             behav_native=nat, behav_patched=pat)
            g["cross_stream_ref"] = (other[0][0] if other else None)
            gates_out[f"{stream}@L{site}"] = g
            print(f"[SAE] GATES {stream}@L{site}: {g['note']}", flush=True)
            _w()

    # ---- emergent_frac across hops, against the cross-seed identity floor -------------------
    for site in sites:
        for parent, child in (("pg1_text", "pg1_mm"), ("pg1_mm", "pi05_mm"), ("pg1_mm", "pi0_mm")):
            kp, kc = (parent, site, seeds[0]), (child, site, seeds[0])
            if kp not in dicts or kc not in dicts:
                continue
            gp = gates_out.get(f"{parent}@L{site}", {})
            gc = gates_out.get(f"{child}@L{site}", {})
            if not (gp.get("all_pass") and gc.get("all_pass")):
                rep.setdefault("emergent", {})[f"{parent}->{child}@L{site}"] = {
                    "state": "VOID", "gate": (gc.get("void_gate") or gp.get("void_gate")),
                    "note": "a stream whose gates failed cannot supply a feature verdict (§2.2)"}
                continue
            floor = float(_matched_cos(dicts[(child, site, seeds[0])],
                                       dicts[(child, site, seeds[1])]).mean())
            best = _matched_cos(dicts[kc], dicts[kp]).numpy()
            rep.setdefault("emergent", {})[f"{parent}->{child}@L{site}"] = PC.emergent_frac(
                best, identity_floor=floor)
            _w()

    rep["elapsed_s"] = round(time.time() - t0, 1)
    _w()
    print(f"PLENUM_SAE_OK cells={len(cells)} gates={len(gates_out)}", flush=True)


if __name__ == "__main__":
    main()
