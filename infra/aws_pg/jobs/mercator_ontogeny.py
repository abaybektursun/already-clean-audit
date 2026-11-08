"""M-A4 and M-12 stage b — the two second clocks, read against W0-9's crystallization curve.

Both arms ask the same shape of question: **something else about this model also forms during
training — does it form before, with, or after the consensus subspace crystallizes?** They differ
only in what that something else is, so they share a job, a checkpoint grid, a crystallization
curve and one definition of "when".

    --arm M-A4   ACTIVATION TRAFFIC. Project the FINAL window's consensus bank onto each earlier
                 checkpoint's residual stream. Does high activation traffic in sink coordinates
                 predate crystallization?
                 REFUTER: sink-coordinate activation traffic does not predate crystallization.

    --arm M-12b  INDUCTION HEADS. Per-head prefix-matching score on [R, R] sequences at each
                 checkpoint. Does the induction bump land before or after crystallization?
                 REFUTER (this stage's share of M-12's): the two clocks are inseparable at the
                 resolution the checkpoint grid has.

ONE RULE FOR "WHEN", FOR BOTH CLOCKS. `mercator_induction.onset_step` is the half-rise crossing of
a curve's own range, applied identically to the crystallization curve and to whichever second curve
the arm is about. Two clocks read by two definitions of onset would produce a separation that is an
artifact of the definitions, and `two_clock_overlay` reports the separation together with the
COARSER of the two grid resolutions, so "the clocks coincide" cannot be claimed at a resolution the
checkpoint grid does not have.

THE MACHINERY IS W0-9's, IMPORTED. `checkpoint_revisions`, `pick_frames`, `delta_terms`,
`window_banks` come from `darwin_w0_ext` unchanged, so the crystallization curve here is the SAME
estimand W0-9 measured rather than a re-derivation that could differ for reasons nobody tracked.

THE CONTROL FOR M-A4 is a matched-random subspace at every checkpoint, plus the structured
alternative: the coordinates carrying the most activation energy at the FINAL checkpoint. A rise in
"energy inside a 32-dim subspace" is only about the SINK if a random 32-dim subspace does not rise
the same way, and only about the CONSENSUS if the plain top-energy coordinates do not either.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import mercator_arch as AR
from infra.pg import mercator_core as M
from infra.pg import mercator_induction as IND
from infra.pg import mercator_program as P

ARMS = {"M-A4": "M-A4", "M-12b": "M-12"}
CRYSTALLIZE_FRAC = 0.5           # the half-rise rule, shared by both clocks
N_HAAR = 3


def residual_energy(model, layers_idx, input_ids, banks, device):
    """Energy fraction of the residual stream inside each bank, per layer, on fixed inputs.

    The site is the input to `self_attn.q_proj` — the same residual site the banks are keyed to
    everywhere else in this program. Measured as a fraction so that a checkpoint whose activations
    are simply larger overall does not read as more traffic in the subspace.
    """
    import torch
    assert hasattr(model, "model") and hasattr(model.model, "layers"), (
        f"{type(model).__name__} does not expose model.layers; the residual site cannot be "
        f"located without knowing the decoder stack")
    dec = model.model.layers
    buf = {}

    def mk(L):
        def pre_hook(_m, args):
            buf[L] = args[0].detach().to(torch.float32)
        return pre_hook

    handles = [dec[L].self_attn.q_proj.register_forward_pre_hook(mk(L)) for L in layers_idx]
    with torch.inference_mode():
        model(input_ids=input_ids.to(device))
    for h in handles:
        h.remove()
    out = {}
    for L in layers_idx:
        h = buf[L].reshape(-1, buf[L].shape[-1])
        tot = float((h * h).sum())
        assert tot > 0, f"layer {L} residual is identically zero"
        for name, U in banks[L].items():
            Ut = torch.as_tensor(U, device=h.device, dtype=torch.float32)
            out[(L, name)] = float((h @ Ut).pow(2).sum()) / tot
    return out


def one_trajectory(revs):
    """Keep only the branches belonging to a SINGLE training trajectory, and say which.

    `checkpoint_revisions` accepts every branch carrying `stepN` and sorts by N. OLMo-2 publishes
    two families — `stage1-stepN` (the pretraining run) and `stage2-ingredient1-stepN` (a separate
    mid-training anneal whose step numbering RESTARTS) — so the pooled list interleaves two
    incomparable series on one axis. M-A4's frames came back as stage1 at 0 and 300, stage2 at
    1000/2000/4000, stage1 at 10000, stage2 at 23852, stage1 from 60000 on: the "collapse" at
    1e4 and 6e4 in the traffic figure is the picker alternating between two different runs, and
    every clock computed on that axis timed a mixture (M-D94).

    The trajectory kept is the one with the most revisions, which is the main pretraining run for
    every suite this program uses; the choice and the drop count are returned so the artifact
    records them rather than leaving a reader to infer the axis.
    """
    import collections
    import re as _re
    groups = collections.defaultdict(list)
    for step, branch in revs:
        # everything before the `stepN` token names the family: "stage1-", "stage2-ingredient1-"
        groups[_re.split(r"step\d+", branch)[0]][:] = groups[_re.split(r"step\d+", branch)[0]] + \
            [(step, branch)]
    best = max(groups, key=lambda k: len(groups[k]))
    kept = sorted(groups[best])
    return kept, {"trajectory": best or "(unprefixed)", "n_kept": len(kept),
                  "n_dropped": len(revs) - len(kept),
                  "families": {k: len(v) for k, v in sorted(groups.items())}}


def refine_frames(revs, chosen, lo: float, hi: float, n_extra: int):
    """Add `n_extra` checkpoints inside [lo, hi], for resolving a crossing the coarse grid brackets.

    THE FIRST PASS LOCATES, THE SECOND RESOLVES. A log grid over 1.9M steps is 248074 steps wide at
    the top, and M-A4's crystallization clock crossed its half-rise somewhere inside one such
    interval — so the measured 188345-step separation from the induction clock was SMALLER than the
    grid step that produced it, and the frozen separability gate correctly recorded VOID-INSTRUMENT
    (defect M-D53). The answer to a grid too coarse to resolve a crossing is a finer grid over the
    bracket it already established, not a softer gate: the gate is left exactly as it was and is
    asked again on frames that can actually answer it.
    """
    import numpy as np
    steps = np.array([s for s, _ in revs], dtype=np.float64)
    used = {revs.index(c) for c in chosen}
    out = list(chosen)
    for target in np.linspace(lo, hi, n_extra + 2)[1:-1]:
        i = int(np.argmin(np.abs(steps - target)))
        if i not in used:
            used.add(i)
            out.append(revs[i])
    return sorted(out, key=lambda r: r[0])


def log_frames(revs, n_frames: int):
    """`n_frames` checkpoints spaced logarithmically in STEP NUMBER, first and last included.

    `pick_frames` spaces evenly over the revision INDEX, and OLMo's revisions are roughly linear in
    step — so an 8-frame even grid over 1.9M steps puts its second frame at 12000 and everything
    a clock might time has already happened before it. Both of M-A4's and M-12b's second clocks
    saturated by frame 2 on that grid: sink traffic 0.016 -> 0.113 and prefix-matching 0.036 ->
    0.928, with no resolution inside the interval where it happened (defect M-D33). Logarithmic
    spacing puts most frames where the change is.
    """
    import numpy as np
    assert len(revs) >= 2, f"suite has {len(revs)} step revisions, need >= 2"
    steps = np.array([s for s, _ in revs], dtype=np.float64)
    lo = max(steps[steps > 0].min(), 1.0) if (steps > 0).any() else 1.0
    targets = np.unique(np.concatenate([[steps.min()],
                                        np.geomspace(lo, steps.max(), n_frames - 1)]))
    chosen, used = [], set()
    for t in targets:
        i = int(np.argmin(np.abs(steps - t)))
        if i not in used:
            used.add(i)
            chosen.append(revs[i])
    return sorted(chosen, key=lambda r: r[0])


def coordinate_energy(model, layers_idx, input_ids, device):
    """Per-coordinate mean squared residual activation, per layer, on the same fixed inputs.

    Used to build the STRUCTURED control: the 32 coordinates that carry the most activation energy
    at the FINAL checkpoint. If sink-coordinate traffic rises before crystallization but so does
    plain top-energy-coordinate traffic, the arm has measured "activations grow", not "the sink
    fills up" — and a Haar draw cannot tell those apart because a Haar subspace is not big anywhere.
    """
    import torch
    dec = model.model.layers
    buf = {}

    def mk(L):
        def pre_hook(_m, args):
            buf[L] = args[0].detach().to(torch.float32)
        return pre_hook

    handles = [dec[L].self_attn.q_proj.register_forward_pre_hook(mk(L)) for L in layers_idx]
    with torch.inference_mode():
        model(input_ids=input_ids.to(device))
    for h in handles:
        h.remove()
    return {L: buf[L].reshape(-1, buf[L].shape[-1]).pow(2).mean(dim=0).cpu().numpy()
            for L in layers_idx}


def head_prefix_scores(model, input_ids, seq_len, device):
    """(n_layers, n_heads) prefix-matching scores, averaged over the [R, R] batch."""
    import torch
    with torch.inference_mode():
        o = model(input_ids=input_ids.to(device), output_attentions=True)
    per_layer = []
    for att in o.attentions:                       # (B, H, T, T)
        a = att.to(torch.float32).cpu().numpy()
        per_layer.append(np.stack([IND.prefix_matching_score(a[b], seq_len, offset=0)
                                   for b in range(a.shape[0])]).mean(axis=0))
    return np.stack(per_layer)


def main():
    ap = argparse.ArgumentParser(description="M-A4 / M-12b — second clocks vs crystallization")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--suite", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--refine-bracket", default="",
                    help="'lo,hi' step bracket to add --n-refine extra frames inside; used to "
                         "resolve a crossing the coarse grid only brackets (M-D53)")
    ap.add_argument("--n-refine", type=int, default=4)
    ap.add_argument("--frame-spacing", default="log", choices=("even", "log"),
                    help=("log spaces frames by STEP NUMBER, which is what a clock needs: the "
                          "first run used even index spacing and both second clocks had already "
                          "saturated by frame 2 (defect M-D33)"))
    ap.add_argument("--types", default="q_proj")
    ap.add_argument("--layers", default="4,8,12")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--n-seq", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    arm = ARMS[a.arm]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers_idx = [int(x) for x in a.layers.split(",") if x.strip()]
    keys = [(t, L) for t in types for L in layers_idx]
    meta = P.ARMS[arm]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(arm)
    at = M.Atlas(out, arm=arm, run_id=f"m_ontogeny_{a.arm.lower().replace('-', '_')}",
                 register=meta["register"], wave=meta["wave"], expected_ref=expected_ref,
                 queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from infra.pg import darwin_w0 as DW
    from infra.aws_pg.jobs.invariant_core import _delete_cache
    from infra.aws_pg.jobs.darwin_w0_ext import (SUITE_FALLBACKS, checkpoint_revisions,
                                                 delta_terms, pick_frames, window_banks)

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    repo, revs = a.suite, checkpoint_revisions(api, a.suite)
    revs, traj = one_trajectory(revs)
    print(f"[{a.arm}] trajectory {traj['trajectory']!r}: kept {traj['n_kept']} revisions, "
          f"dropped {traj['n_dropped']} from other families {traj['families']}", flush=True)
    _traj_info = traj
    for alt in SUITE_FALLBACKS:
        if len(revs) >= 2 * a.window:
            break
        print(f"[{a.arm}] {repo}: {len(revs)} step revisions — trying {alt}", flush=True)
        repo, revs = alt, checkpoint_revisions(api, alt)
    assert len(revs) >= a.window + 1, (
        f"no suite exposes >= {a.window + 1} step revisions (tried "
        f"{[a.suite] + list(SUITE_FALLBACKS)}) — this arm is suite-absent")
    # THE AXIS IS PART OF THE INSTRUMENT. Recorded as a gate so a reader can see which trajectory
    # every clock in this run was timed on, and how many revisions were dropped to get one.
    gates.record("ontogeny_frames_are_one_trajectory", True, **_traj_info,
                 note=("OLMo-2 publishes stage1 and stage2-ingredient1 branches whose step "
                       "numbering restarts; pooling them puts two incomparable series on one "
                       "axis and every clock then times a mixture (M-D94)"))
    frames = (pick_frames(revs, a.n_frames) if a.frame_spacing == "even"
              else log_frames(revs, a.n_frames))
    if a.refine_bracket:
        lo, hi = (float(x) for x in a.refine_bracket.split(","))
        n_before = len(frames)
        frames = refine_frames(revs, frames, lo, hi, a.n_refine)
        print(f"[{a.arm}] refined [{lo:.0f},{hi:.0f}] with {len(frames) - n_before} extra frames",
              flush=True)
    sha = api.model_info(repo, revision=frames[0][1]).sha
    print(f"[{a.arm}] {repo}: {len(revs)} revisions, {len(frames)} frames "
          f"{[s for s, _ in frames]}", flush=True)

    # ---- clock B: crystallization, W0-9's own object ------------------------------------------
    terms, _fro2 = delta_terms(repo, frames, types, layers_idx, _delete_cache)
    wins = window_banks(terms, keys, a.window, H.BANK_K)
    final = wins[-1]
    # THE REFERENCE WINDOW IS NOT A MEASUREMENT. Agreement is computed against the FINAL window, so
    # the final window's own value is 1.0 by definition — and including it pins the half-rise
    # target near 1.0, which no real window reaches, so the crystallization clock would read "at
    # the last step" for every suite by construction (defect M-D26). It is banked as a row, marked
    # as the reference, and excluded from the curve the onset is read from.
    cryst_steps, cryst_vals = [], []
    for w in wins:
        mid = (w["lo_step"] + w["hi_step"]) / 2.0
        agree = float(np.mean([DW.subspace_agreement(w["banks"][k], final["banks"][k])["mean_cos2"]
                               for k in keys]))
        is_reference = w is final
        if not is_reference:
            cryst_steps.append(mid)
            cryst_vals.append(agree)
        at.add("ontogeny_subspace_agreement", value=agree, dose=float(mid),
               bank_id="bottom32_window", incumbent_id="final_window", model_id=repo,
               model_sha=sha,
               flags={"lo_step": w["lo_step"], "hi_step": w["hi_step"], "window": a.window,
                      "keys": [f"{t}.{L}" for t, L in keys], "suite": repo,
                      "is_reference_window": is_reference,
                      "in_onset_curve": not is_reference,
                      "overlap": "mean cos^2 of principal angles, averaged over cells"})
        print(f"[{a.arm}] crystallization @{mid:.0f}: mean cos2 = {agree:.4f}"
              + ("   (reference window, excluded from the onset curve)" if is_reference else ""),
              flush=True)

    assert len(cryst_steps) >= 2, (
        f"only {len(cryst_steps)} non-reference windows; an onset needs at least two. Increase "
        f"--n-frames or decrease --window")

    # ---- clock A: whatever this arm is about --------------------------------------------------
    tok = AutoTokenizer.from_pretrained(repo, revision=frames[-1][1])
    d_model = None
    second_steps, second_vals, per_frame = [], [], {}
    control_series = {}

    if arm == "M-A4":
        # The banks to project: the FINAL window's consensus bottom-32, plus the controls.
        forbid = [i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
                  if i is not None]
        text_ids = torch.as_tensor(IND.repeated_random_tokens(
            len(tok), a.seq_len, a.n_seq, a.seed, forbid=forbid))
        d_model = int(final["banks"][keys[0]].shape[0])
        # The structured control needs the final checkpoint's coordinate energies, so it is
        # measured first and the whole battery is fixed before any frame is scored.
        fin = AutoModelForCausalLM.from_pretrained(
            repo, revision=frames[-1][1], torch_dtype=torch.float32,
            attn_implementation="eager").to("cuda").eval()
        coord_e = coordinate_energy(fin, layers_idx, text_ids, "cuda")
        del fin
        torch.cuda.empty_cache()
        banks = {}
        for t, L in keys:
            b = {"core32": final["banks"][(t, L)]}
            top = np.argsort(coord_e[L])[::-1][:H.BANK_K]
            E = np.zeros((d_model, H.BANK_K), dtype=np.float64)
            E[top, np.arange(H.BANK_K)] = 1.0
            b["topenergy32"] = E
            for s in range(N_HAAR):
                R = AR.random_rotation(d_model, seed=1000 + s)
                b[f"haar32_s{s}"] = np.ascontiguousarray(R[:, :H.BANK_K].numpy())
            banks[L] = b
        control_names = ["topenergy32"] + [f"haar32_s{s}" for s in range(N_HAAR)]
        print(f"[{a.arm}] structured control: top-{H.BANK_K} energy coordinates fixed at the "
              f"final checkpoint", flush=True)
    else:
        forbid = [i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
                  if i is not None]
        text_ids = torch.as_tensor(IND.repeated_random_tokens(
            len(tok), a.seq_len, a.n_seq, a.seed, forbid=forbid))
        banks, control_names = None, []

    for step, branch in frames:
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=branch, torch_dtype=torch.float32,
            attn_implementation="eager").to("cuda").eval()
        if arm == "M-A4":
            e = residual_energy(model, layers_idx, text_ids, banks, "cuda")
            core = float(np.mean([e[(L, "core32")] for L in layers_idx]))
            per_frame[step] = {"core32": core}
            for cn in control_names:
                per_frame[step][cn] = float(np.mean([e[(L, cn)] for L in layers_idx]))
                control_series.setdefault(cn, []).append(per_frame[step][cn])
            for (L, name), v in e.items():
                at.add("sink_activation_traffic", value=v, layer=L, dose=float(step),
                       bank_id=name, model_id=repo, model_sha=sha,
                       flags={"step": step, "branch": branch, "suite": repo,
                              "site": "q_proj input (residual)",
                              "definition": "||h U||^2 / ||h||^2 on fixed [R,R] token inputs"})
            second_steps.append(float(step))
            second_vals.append(core)
            print(f"[{a.arm}] step {step}: core traffic {core:.5f}  controls "
                  + " ".join(f"{cn}={per_frame[step][cn]:.5f}" for cn in control_names),
                  flush=True)
        else:
            sc = head_prefix_scores(model, text_ids, a.seq_len, "cuda")
            heads = IND.induction_heads(sc)
            top = float(sc.max())
            per_frame[step] = {"score_max": top, "score_mean": float(sc.mean()),
                               "n_heads_selected": heads["n_heads_selected"],
                               "sweep": heads["sweep"]}
            second_steps.append(float(step))
            second_vals.append(top)
            at.add("induction_prefix_matching_max", value=top, dose=float(step), model_id=repo,
                   model_sha=sha,
                   flags={"step": step, "branch": branch, "suite": repo,
                          "n_heads_selected": heads["n_heads_selected"],
                          "n_heads_total": heads["n_heads_total"],
                          "score_mean": heads["score_mean"], "sweep": heads["sweep"],
                          "seq_len": a.seq_len, "n_seq": a.n_seq})
            at.add("induction_head_count", value=float(heads["n_heads_selected"]),
                   dose=float(step), model_id=repo, model_sha=sha,
                   flags={"step": step, "threshold": heads["threshold"], "sweep": heads["sweep"]})
            print(f"[{a.arm}] step {step}: max prefix-match {top:.4f}, "
                  f"{heads['n_heads_selected']} heads over threshold", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---- the overlay --------------------------------------------------------------------------
    overlay = IND.two_clock_overlay((second_steps, second_vals), (cryst_steps, cryst_vals))
    at.add("two_clock_separation_steps", value=overlay["separation_steps"], model_id=repo,
           model_sha=sha,
           flags={**overlay, "clock_a": ("sink activation traffic" if arm == "M-A4"
                                         else "induction prefix matching"),
                  "clock_b": "consensus-subspace crystallization",
                  "rule": f"half-rise (frac={CRYSTALLIZE_FRAC}) applied identically to both"})
    separable = bool(abs(overlay["separation_in_grid_units"]) >= 1.0)
    gates.record("ontogeny_clocks_are_separable", separable, **overlay,
                 outcome=None if separable else M.VOID_INSTRUMENT,
                 note=("separation smaller than one grid step is not a measurement of "
                       "simultaneity, it is the resolution of the checkpoint grid"))
    predates = bool(overlay["separation_steps"] > 0)
    gates.record("ontogeny_clock_a_predates_crystallization", predates and separable,
                 separation_steps=overlay["separation_steps"],
                 separation_in_grid_units=overlay["separation_in_grid_units"],
                 note=("positive separation means clock A reaches half-rise BEFORE the "
                       "crystallization curve does"))

    if arm == "M-A4":
        haar_names = [f"haar32_s{s}" for s in range(N_HAAR)]
        haar_last = [control_series[cn][-1] for cn in haar_names]
        core_last = second_vals[-1]
        gates.record("a4_core_traffic_exceeds_matched_random", core_last > max(haar_last),
                     core=core_last,
                     controls={cn: control_series[cn][-1] for cn in control_names},
                     note=("energy inside SOME 32-dim subspace rises trivially; the claim needs "
                           "the consensus bank to carry more than a random one"))
        # THE CLOCK COMPARISON, RUN ON THE STRUCTURED CONTROL TOO. If the top-energy coordinates'
        # traffic also predates crystallization by the same margin, the arm has timed "activations
        # grow", not "the sink fills up".
        ctrl_overlay = IND.two_clock_overlay(
            (second_steps, control_series["topenergy32"]), (cryst_steps, cryst_vals))
        at.add("two_clock_separation_steps", value=ctrl_overlay["separation_steps"],
               bank_id="topenergy32", model_id=repo, model_sha=sha,
               flags={**ctrl_overlay, "clock_a": "top-energy-coordinate activation traffic",
                      "role": "structured control for the timing claim"})
        gates.record("a4_timing_is_not_shared_with_top_energy_coordinates",
                     bool(overlay["separation_steps"] - ctrl_overlay["separation_steps"]
                          >= overlay["grid_resolution"]),
                     core_separation=overlay["separation_steps"],
                     topenergy_separation=ctrl_overlay["separation_steps"],
                     grid_resolution=overlay["grid_resolution"],
                     note=("the sink's head start must exceed the plain top-energy "
                           "coordinates' by at least one grid step to be about the sink"))
        sur.add(arm=arm, expected_ref=expected_ref, cell=f"step={int(second_steps[-1])}",
                statistic="sink_activation_traffic/core32",
                s=M.surprise(core_last, float(np.mean(haar_last)), haar_last))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{a.arm}_result.json",
                 {"arm": arm, "stage": a.arm, "queue_item": meta["queue_item"],
                  "refuter": meta["refuter"], "expected_ref": expected_ref,
                  "provenance": M.provenance(), "utc": M.utc_now(), "atlas": info,
                  "suite": repo, "suite_sha": sha, "n_step_revisions": len(revs),
                  "frames": [{"step": s, "branch": b} for s, b in frames],
                  "types": list(types), "layers": layers_idx, "window": a.window,
                  "d_model": d_model,
                  "crystallization": {"steps": cryst_steps, "values": cryst_vals},
                  "clock_a": {"steps": second_steps, "values": second_vals},
                  "controls": control_series, "per_frame": per_frame,
                  "two_clock_overlay": overlay, "clocks_separable": separable})
    print(f"{a.arm}_OK " + json.dumps({"rows": info["n_rows"],
                                       "separation_steps": overlay["separation_steps"],
                                       "grid_units": overlay["separation_in_grid_units"],
                                       "separable": separable}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
