"""M-9b and M-C4 — the universality cells on foreign bases (queue items 9 and C4).

Everything the program knows about the sink it knows about one decoder family. These two arms take
the machinery to bases the program did not build its banks on, using each base's OWN census banks
from the re-census — a foreign base read with PG1's banks would be a random subspace and would
answer nothing.

    --arm M-9b   THREE CELLS on one foreign base, each with its refuter pre-named:
                   ablation      project the bank out of what attention READS at every layer;
                                 REFUTER: damage within the matched-random band
                   activation    the residual stream's energy share inside the bank;
                                 REFUTER: energy at chance (k/n)
                   addressable   can a bank-confined bias move behaviour at all;
                                 REFUTER: the bank is no more addressable than a twin

    --arm M-C4   THE RESONANCE FINGERPRINT. For each census base, the per-bank resonant energy
                 profile — how much a short optimized input can excite each bank — and then the
                 question the arm exists for: are profiles more similar WITHIN a family than
                 ACROSS families?
                 REFUTER: profiles no more stable within family than across.

BOTH ARMS REQUIRE THE RE-CENSUS to have banked the base they run on. A base without banks is
recorded COVERAGE-IMPOSSIBLE with the reason, not skipped: which foreign bases could be censused
at all is part of the universality answer.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARMS = {"M-9b": "M-9b", "M-C4": "M-C4"}
RECENSUS_BANKS = ("core32", "anti32", "complement32", "haar32_s0", "haar32_s1", "haar32_s2")
# THE TWO SIGMA TWINS ARE THIS ARM'S JOB TO BUILD, and not building them is why M-9b aborted on
# `check_controls_per_statistic` after its banks loaded correctly (M-D101). The re-census banks six
# of the eight standing banks and says so in its own result blob: "the two sigma twins need each
# family's own base matrix and are added by the arm that consumes them". This is the consuming arm,
# it loads the base model anyway, and it simply never held up that end.
#
# The omission is not cosmetic. `topsigma32` is the STRUCTURED control — big somewhere by
# construction — and the house rule exists because a Haar twin is big nowhere, so a battery of
# core + anti + complement + three Haar draws is missing the control that actually competes.
SIGMA_BANKS = ("topsigma32", "bottomsigma32")
BANKS = RECENSUS_BANKS + SIGMA_BANKS
# The re-census banks these; the bank file naming is `mercator_recensus.bank_name`.
FOREIGN_BASES = ("meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-3B", "google/gemma-2b")
# THE BANKS PREFIX IS AN ARGUMENT, NOT A CONSTANT, AND THE LAUNCHER IS WHY. `mercator_launch.sh`
# refuses to relaunch into a prefix that already holds a STATUS, so a dependency that has to be
# retried lands under a NEW prefix every time — recensus, recensus2, … recensus6. A consumer that
# spells its producer's prefix as a constant therefore goes stale BY CONSTRUCTION at the first
# retry, and it fails in the worst possible way: `aws s3 sync` of an empty prefix succeeds, zero
# files arrive, and the arm records a truthful-looking COVERAGE-IMPOSSIBLE naming a dependency that
# was in fact being produced two prefixes over. M-7, M-9b and M-C4 all recorded that verdict against
# `results/mercator_recensus` while `results/mercator_recensus6` was banking the very cells they
# wanted (M-D99).
RECENSUS_S3 = "results/mercator_recensus/out/banks"
N_PROMPTS = 32


class InputProject:
    """Project a bank out of q_proj's input at every layer — FUNC's `_InputProject`, written for a
    foreign decoder whose layer container this job resolves rather than assumes."""

    def __init__(self, layers, basis_by_layer, device, dtype):
        import torch
        self.layers = layers
        self.Q = {L: torch.as_tensor(U, device=device, dtype=torch.float32)
                  for L, U in basis_by_layer.items()}
        self.dtype = dtype
        self._h = []

    def _mk(self, L):
        import torch

        def pre_hook(_m, args):
            x = args[0]
            U = self.Q[L]
            z = x.to(torch.float32)
            return ((z - (z @ U) @ U.T).to(x.dtype),) + tuple(args[1:])
        return pre_hook

    def __enter__(self):
        for L in sorted(self.Q):
            self._h.append(self.layers[L].self_attn.q_proj
                           .register_forward_pre_hook(self._mk(L)))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []
        return False


def main():
    ap = argparse.ArgumentParser(description="M-9b / M-C4 — foreign-base universality cells")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--bases", default=",".join(FOREIGN_BASES))
    ap.add_argument("--banks-s3", default=RECENSUS_S3,
                    help="comma-list of re-census bank prefixes, synced in the order given")
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--resonance-steps", type=int, default=150)
    ap.add_argument("--resonance-len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bases = [b.strip() for b in a.bases.split(",") if b.strip()]
    arm = ARMS[a.arm]
    meta = P.ARMS[arm]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(arm)
    at = M.Atlas(out, arm=arm, run_id=f"m_foreign_{a.arm.lower().replace('-', '_')}",
                 register=meta["register"], wave=meta["wave"], expected_ref=expected_ref,
                 queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    import gc
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    bank_dir = H.cache_dir(out, "recensus_banks")
    bank_dir.mkdir(parents=True, exist_ok=True)
    bank_index, bank_origin, complete_bases = M.load_recensus_banks(a.banks_s3, bank_dir)
    have = sorted(bank_index)
    print(f"[{a.arm}] {len(have)} re-census bank files available; complete bases "
          f"{sorted(complete_bases)}", flush=True)
    # A BASE WHOSE CENSUS DID NOT FINISH IS NOT A BASE. The reaper kills the census instance
    # outright while `out/` is being streamed every 30 s, so half-written banks reach S3 looking
    # exactly like finished ones; `recensus_partial.json` is the only completeness signal.
    skipped_incomplete = [b for b in bases if b not in complete_bases]
    bases = [b for b in bases if b in complete_bases]
    if skipped_incomplete:
        print(f"[{a.arm}] SKIP (census did not complete): {skipped_incomplete}", flush=True)
    gates.record("foreign_bases_with_a_completed_census", bool(bases),
                 complete=sorted(complete_bases), requested_but_incomplete=skipped_incomplete,
                 note=("a base is usable only if its census RETURNED; banks streamed from a reaped "
                       "instance are indistinguishable from finished ones on the file listing"))

    from infra.aws_pg.jobs.mercator_recensus import bank_name

    def bank_path(base, t, L, bid):
        # THE RE-CENSUS'S OWN NAMING, imported rather than re-spelled: a job that guesses the file
        # name finds nothing and reports coverage-impossible on a base that was in fact banked.
        # A name that no prefix supplied resolves to a path under `bank_dir` that does not exist,
        # which is what the callers' `.exists()` checks already expect.
        nm = bank_name(base, t, L, bid)
        return bank_index.get(nm, bank_dir / nm)

    def load_banks(base, n_layers):
        """Every layer that has a full battery banked for this base. A partial battery is a
        partial control, so a layer missing any bank is dropped and the count is recorded."""
        got, missing = {}, []
        for L in range(n_layers):
            # only the SIX file-backed banks are looked for on disk; the two sigma twins are
            # derived from the base matrix below, once the model is in memory.
            paths = {b: bank_path(base, "q_proj", L, b) for b in RECENSUS_BANKS}
            if all(p.exists() for p in paths.values()):
                got[L] = {b: np.load(paths[b]) for b in RECENSUS_BANKS}
            else:
                missing.append(L)
        return got, missing

    def add_sigma_twins(banks, model):
        """Derive `topsigma32` / `bottomsigma32` per layer from this family's own q_proj.

        `H.sigma_subspaces` is the house derivation (DARWIN W0-0 gated it against HUMBOLDT's banked
        `bottomsigma32`), reused rather than re-spelled so a foreign base's structured control means
        the same thing as PG1's. These checkpoints are natively bf16, so reading the weights off the
        loaded model is the checkpoint's own precision and not a downgrade; the fp64 cast happens
        inside `sigma_subspaces` before the SVD.
        """
        for L, banks_L in banks.items():
            W0 = model.model.layers[L].self_attn.q_proj.weight.detach().to(
                torch.float64).cpu().numpy()
            bot, top, sigma = H.sigma_subspaces(W0, H.BANK_K)
            H.orthonormal_check(top)
            H.orthonormal_check(bot)
            banks_L["topsigma32"], banks_L["bottomsigma32"] = top, bot
            del W0
        return banks

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())

    def ppl(model, tok, n_seq, window):
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        tot, n = 0.0, 0
        for i in range(min(n_seq, ids.shape[0] // window)):
            b = ids[i * window:(i + 1) * window].unsqueeze(0).to("cuda")
            with torch.inference_mode():
                lg = model(input_ids=b).logits[:, :-1].to(torch.float32)
            tot += float(torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1), reduction="sum"))
            n += int(b[:, 1:].numel())
        assert n > 0, "no evaluation windows"
        return float(np.exp(tot / n))

    results = {}
    for base in bases:
        cfg = AutoConfig.from_pretrained(base)
        n_layers = int(getattr(cfg, "num_hidden_layers", 0))
        banks, missing = load_banks(base, n_layers)
        if not banks:
            results[base] = {"outcome": M.COVERAGE_IMPOSSIBLE,
                             "why": "no re-census banks for this base",
                             "n_layers": n_layers}
            at.add("foreign_banks_available", value=0.0, model_id=base,
                   flags={**results[base], "outcome": M.COVERAGE_IMPOSSIBLE})
            print(f"[{a.arm}] {base}: no banks — coverage-impossible", flush=True)
            continue
        sha = api.model_info(base).sha
        tok = AutoTokenizer.from_pretrained(base, revision=sha)
        model = AutoModelForCausalLM.from_pretrained(
            base, revision=sha, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="eager").eval()
        # the battery is completed HERE, from this family's own base matrix — see SIGMA_BANKS
        banks = add_sigma_twins(banks, model)
        print(f"[{a.arm}] {base}: battery = {len(BANKS)} banks over {len(banks)} layers "
              f"({len(RECENSUS_BANKS)} banked + {len(SIGMA_BANKS)} derived here)", flush=True)
        layers = model.model.layers
        d_model = int(banks[next(iter(banks))]["core32"].shape[0])
        base_ppl = ppl(model, tok, a.n_seq, a.window)
        at.add("foreign_banks_available", value=float(len(banks)), model_id=base, model_sha=sha,
               flags={"n_layers": n_layers, "layers_missing_a_bank": missing,
                      "d_model": d_model, "base_ppl": base_ppl})
        print(f"[{a.arm}] {base}: {len(banks)}/{n_layers} layers banked, d={d_model}, "
              f"ppl={base_ppl:.4f}", flush=True)

        if arm == "M-9b":
            cell = {"base_ppl": base_ppl, "d_model": d_model, "n_layers_banked": len(banks)}
            # (a) ABLATION — is the bank load-bearing for what attention reads?
            dmg = {}
            for b in BANKS:
                with InputProject(layers, {L: banks[L][b] for L in banks}, "cuda",
                                  torch.bfloat16):
                    p = ppl(model, tok, a.n_seq, a.window)
                dmg[b] = p / base_ppl
                at.add("foreign_ablation_damage", value=dmg[b], bank_id=b, model_id=base,
                       model_sha=sha, corpus_id=f"wikitext2:window{a.window}",
                       flags={"base_ppl": base_ppl, "ablated_ppl": p,
                              "definition": "ppl with the bank projected out of q_proj's input, "
                                            "relative to intact"})
                print(f"[M-9b]   ablate {b:14s} ppl={p:9.4f} damage={dmg[b]:.4f}", flush=True)
            twins = [dmg[b] for b in ("haar32_s0", "haar32_s1", "haar32_s2")]
            cell["ablation"] = {"damage": dmg, "twin_band": [min(twins), max(twins)],
                                "core_outside_band": bool(dmg["core32"] > max(twins))}
            # (b) ACTIVATION SHARE — does the residual stream actually live there?
            buf = {}

            def mk(L):
                def hook(_m, args):
                    buf[L] = args[0].detach().to(torch.float32)
                return hook

            hs = [layers[L].self_attn.q_proj.register_forward_pre_hook(mk(L)) for L in banks]
            ids = tok(text[:20000], return_tensors="pt",
                      add_special_tokens=False).input_ids[:, :a.window].to("cuda")
            with torch.inference_mode():
                model(input_ids=ids)
            for h in hs:
                h.remove()
            share = {}
            for b in BANKS:
                v = []
                for L in banks:
                    hmat = buf[L].reshape(-1, buf[L].shape[-1])
                    U = torch.as_tensor(banks[L][b], device="cuda", dtype=torch.float32)
                    v.append(float((hmat @ U).pow(2).sum()) / float((hmat ** 2).sum()))
                share[b] = float(np.mean(v))
                at.add("foreign_activation_share", value=share[b], bank_id=b, model_id=base,
                       model_sha=sha,
                       flags={"chance": H.BANK_K / d_model, "n_layers": len(banks)})
            cell["activation"] = {"share": share, "chance": H.BANK_K / d_model,
                                  "core_above_chance": bool(share["core32"]
                                                            > H.BANK_K / d_model)}
            print(f"[M-9b]   activation share core={share['core32']:.5f} "
                  f"haar={share['haar32_s0']:.5f} chance={H.BANK_K / d_model:.5f}", flush=True)
            # (c) ADDRESSABILITY — can a bank-confined bias move the output at all?
            addr = {}
            for b in BANKS:
                U0 = torch.as_tensor(banks[min(banks)][b], device="cuda", dtype=torch.float32)
                c = torch.randn(U0.shape[1], generator=torch.Generator().manual_seed(a.seed),
                                device="cpu").to("cuda")
                vec = (U0 @ c)
                vec = vec / vec.norm()
                scale = float(buf[min(banks)].norm(dim=-1).mean())

                def bias_hook(_m, args, v=vec * scale):
                    return (args[0] + v.to(args[0].dtype),) + tuple(args[1:])

                h = layers[min(banks)].self_attn.q_proj.register_forward_pre_hook(bias_hook)
                p = ppl(model, tok, a.n_seq, a.window)
                h.remove()
                addr[b] = p / base_ppl
                at.add("foreign_addressability", value=addr[b], bank_id=b, model_id=base,
                       model_sha=sha,
                       flags={"definition": ("ppl change from a unit-direction bias inside the "
                                             "bank at the first banked layer, scaled to the "
                                             "stream's own norm"), "scale": scale})
            at_twins = [addr[b] for b in ("haar32_s0", "haar32_s1", "haar32_s2")]
            cell["addressability"] = {"effect": addr,
                                      "twin_band": [min(at_twins), max(at_twins)],
                                      "core_outside_band": bool(addr["core32"] > max(at_twins))}
            print(f"[M-9b]   addressability core={addr['core32']:.4f} "
                  f"twin band [{min(at_twins):.4f},{max(at_twins):.4f}]", flush=True)
            results[base] = cell
        else:
            # M-C4: the resonance fingerprint. A short optimized soft prefix is pushed to excite
            # each bank; the profile is the per-bank achieved energy at a FROZEN budget, so the
            # comparison across bases is at matched effort rather than at matched success.
            emb = model.get_input_embeddings()
            profile = {}
            for b in BANKS:
                L0 = min(banks)
                U0 = torch.as_tensor(banks[L0][b], device="cuda", dtype=torch.float32)
                g = torch.Generator(device="cpu").manual_seed(H.cell_seed(a.seed, arm, base, b))
                z = (torch.randn(1, a.resonance_len, emb.weight.shape[1], generator=g)
                     .to("cuda").to(torch.float32) * 0.01).requires_grad_(True)
                opt = torch.optim.Adam([z], lr=1e-2)
                buf2 = {}

                def hook(_m, args):
                    buf2["h"] = args[0]

                h = layers[L0].self_attn.q_proj.register_forward_pre_hook(hook)
                for _ in range(a.resonance_steps):
                    opt.zero_grad()
                    model(inputs_embeds=z.to(emb.weight.dtype))
                    hm = buf2["h"].to(torch.float32)
                    energy = (hm @ U0).pow(2).sum() / hm.pow(2).sum().clamp(min=1e-12)
                    (-energy).backward()
                    opt.step()
                    with torch.no_grad():
                        z.clamp_(-1.0, 1.0)
                h.remove()
                profile[b] = float(energy.detach())
                at.add("resonance_energy", value=profile[b], bank_id=b, model_id=base,
                       model_sha=sha,
                       flags={"steps": a.resonance_steps, "prefix_len": a.resonance_len,
                              "layer": L0, "chance": H.BANK_K / d_model,
                              "budget": "frozen across bases so profiles are at matched effort"})
            results[base] = {"profile": profile, "d_model": d_model, "base_ppl": base_ppl,
                             "chance": H.BANK_K / d_model}
            print(f"[M-C4] {base}: profile " +
                  " ".join(f"{b.replace('32','')}={profile[b]:.4f}" for b in BANKS), flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        M.s3_sync_incremental(out)

    got = {k: v for k, v in results.items() if v.get("outcome") is None}
    if arm == "M-9b":
        gates.record("m9b_any_foreign_base_had_banks", bool(got), bases=sorted(got),
                     outcome=None if got else M.COVERAGE_IMPOSSIBLE)
        if got:
            at.check_controls_per_statistic(["foreign_ablation_damage",
                                             "foreign_activation_share",
                                             "foreign_addressability"])
            for cellname, key in (("ablation", "core_outside_band"),
                                  ("activation", "core_above_chance"),
                                  ("addressability", "core_outside_band")):
                passing = [b for b, v in got.items() if v[cellname][key]]
                gates.record(f"m9b_{cellname}_transfers", bool(passing), passing_bases=passing,
                             by_base={b: v[cellname] for b, v in got.items()},
                             note=f"the {cellname} cell's refuter, per base")
            b0 = next(iter(got))
            sur.add(arm=arm, expected_ref=expected_ref, cell=b0,
                    statistic="foreign_ablation_damage/core32",
                    s=M.surprise(got[b0]["ablation"]["damage"]["core32"],
                                 float(np.mean([got[b0]["ablation"]["damage"][t]
                                                for t in ("haar32_s0", "haar32_s1", "haar32_s2")])),
                                 [got[b0]["ablation"]["damage"][t]
                                  for t in ("haar32_s0", "haar32_s1", "haar32_s2")]))
    else:
        gates.record("c4_any_base_produced_a_profile", bool(got), bases=sorted(got),
                     outcome=None if got else M.COVERAGE_IMPOSSIBLE)
        if len(got) >= 2:
            names = sorted(got)
            vecs = {b: np.array([got[b]["profile"][k] for k in BANKS]) for b in names}

            def fam(b):
                return b.split("/")[0]

            sims, within, across = {}, [], []
            for i, x in enumerate(names):
                for y in names[i + 1:]:
                    c = float(np.dot(vecs[x], vecs[y])
                              / (np.linalg.norm(vecs[x]) * np.linalg.norm(vecs[y])))
                    sims[f"{x}|{y}"] = c
                    (within if fam(x) == fam(y) else across).append(c)
                    at.add("resonance_profile_similarity", value=c, model_id=x,
                           flags={"other": y, "same_family": fam(x) == fam(y),
                                  "definition": "cosine between the per-bank resonance profiles"})
            stable = bool(within and across and np.mean(within) > np.mean(across))
            gates.record("c4_profiles_more_stable_within_family", stable,
                         within=within, across=across, similarities=sims,
                         n_within=len(within), n_across=len(across),
                         outcome=None if (within and across) else M.COVERAGE_IMPOSSIBLE,
                         note=("the refuter fires when this is False; with no within-family pair "
                               "available the comparison cannot be made at all and that is "
                               "recorded rather than reported as a null"))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{a.arm}_result.json",
                 {"arm": arm, "stage": a.arm, "queue_item": meta["queue_item"],
                  "refuter": meta["refuter"], "expected_ref": expected_ref,
                  "provenance": M.provenance(), "utc": M.utc_now(), "atlas": info,
                  "bases": bases, "banks_source": a.banks_s3, "n_bank_files": len(have),
                  "banks_by_prefix": {k: bank_origin[k] for k in have},
                  "n_seq": a.n_seq, "window": a.window, "results": results})
    print(f"{a.arm.replace('-', '')}_OK " + json.dumps(
        {"rows": info["n_rows"], "bases_with_banks": sorted(got)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
