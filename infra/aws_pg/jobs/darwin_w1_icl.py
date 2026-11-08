"""DARWIN W1-A / W1-B / W1-C — the in-context cluster. GPU.

Three arms that all ask the same underlying question in activation space — is the core subspace a
CHANNEL, something you can write into and read out of — and therefore share one harness so that a
difference between them can never be a difference of plumbing.

  W1-A  Context burning (§4/W1-A, product 3.3). Train a per-layer bias confined to a 32-dim bank,
        b_l = U_l c_l, to make the 0-shot model reproduce the 16-shot model's next-token
        distribution on 256 UNLABELED queries. Then score 0-shot+burned against the 0/4/16-shot
        ladder. REFUTER: the in-core burn is no better than its own twin burns.

  W1-B  ICL amplifier (§4/W1-B, product 4.7). Take the measured context shift at 4 shots,
        h_4 - h_0, project it onto a bank, and SCALE it by g. Does amplifying the core-projected
        component alone move 4-shot toward 16-shot? REFUTER: twin gains do the same.

  W1-C  Intercom capacity (§4/W1-C, product 4.4). Write a random +-1 message into the bank at
        layer 0, read it back with a linear probe at layer 17, and price the channel: recovered
        bits per token against amplitude, against the perplexity the write costs.

DESIGN NOTES that are load-bearing, not decoration:

* READOUT MODEL. §4 names gemma-2b. The decoder under PG1 IS gemma-2b and every bank in the
  program is keyed to that decoder's cells, so the arms run on PG1 with the house blank-image
  forward. Using a separately-loaded gemma-2b would put the treatment and its banks in different
  coordinate frames — the exact confound §7 exists to prevent. `readout_model` is on every row.

* NO LABEL LEAKAGE, stated as a data split rather than as a promise. MMLU `dev` + `validation`
  supply the 16 exemplars per subject; the burn's training queries come from what is LEFT of
  `validation` after those 16; every score is on `test`. The burn objective never sees a gold
  answer — it matches the 16-shot model's own distribution, which is what "unlabeled" means here.

* THE MATCHED INCUMBENT. §4/W1-A names "a 576-param unconstrained-bias incumbent". 576 = 18 layers
  x 32 coefficients, i.e. the parameter count of the confined bias. A bias that is unconstrained
  AND has 576 parameters is a bias free to take any value on 32 coordinates per layer, so the
  incumbent `freecoord32` learns a bias on 32 randomly chosen residual COORDINATES per layer. It
  is matched in capacity and free of the bank geometry, which is the comparison the arm needs.

* DOSE. §3 wants a dose ladder. The burn is trained once per bank at unit strength and then
  EVALUATED at lam in {0.25, 0.5, 1, 2, 4} — a scale ladder on a trained vector, which is a dose
  in the sense §3 means (a monotone knob whose safe range the arm has to show) and which costs
  five evals rather than five trainings.

FAIL LOUD: no try/except, no silent fallback.

  pg run infra.aws_pg.jobs.darwin_w1_icl --args "--arm W1-A" \\
      --type g6e.2xlarge,g6e.4xlarge,g6e.8xlarge,g6e.xlarge --no-weights ...
"""
import argparse
import hashlib
import json
import pathlib

from infra.pg import darwin_core as D
from infra.pg import humboldt_core as H

SITE = "q_proj_in"                       # the residual stream entering attention; H.SITE["q_proj"]
BANK_TYPE = "q_proj"                     # banks are keyed by matrix type; q_proj is the resid site
LAM_LADDER = (0.25, 0.5, 1.0, 2.0, 4.0)  # W1-A dose (>=4, log-spaced ratio 2)
GAIN_LADDER = (1.0, 1.5, 2.0, 4.0)       # W1-B, verbatim from §4/W1-B
AMP_LADDER = (0.25, 0.5, 1.0, 2.0, 4.0)  # W1-C message amplitude
SHOT_LADDER = (0, 4, 16)                 # §4/W1-A's reference ladder

# THE POSITIVE CONTROL FOR W1-A AND W1-B, named before it was needed and then needed.
# Both arms are defined against the 16-shot model: W1-A trains a bias to REPRODUCE it, W1-B asks
# whether amplifying a context shift MOVES 4-shot TOWARD it. If 16-shot is not actually better
# than 0-shot on this readout, there is no target to reach and no gap to close, and every
# "gap_closed" number is a ratio with a meaningless denominator. The first full run measured
# 0-shot 0.3056 / 4-shot 0.2963 / 16-shot 0.2500 at n = 108 — the ladder runs BACKWARDS. The gate
# is therefore evaluated and RECORDED before the burn, and its failure makes the arm
# VOID-INSTRUMENT rather than a refutation (feedback_signals_that_cannot_fail).
ICL_GATE_MIN_LIFT = 0.0                  # acc(16) - acc(0) must exceed its own paired CI low end
READOUT_LAYER = 17                       # W1-C readout; the last decoder layer (N_LAYERS = 18)
AUDIT_READ_LAYER = 1                     # W1-C positive control: one block downstream of the write
AUDIT_MIN_LIFT = 0.05                    # ...where recovery must beat its own MEASURED null
CHANNEL_MIN_BITS = 0.5                   # the L0->L17 ladder must recover at least this, or the
                                         # bits-per-ppl comparison is a ratio of noise to noise

# WHY THE GATE IS A LIFT AND NOT A LEVEL. The first version of this control required bit accuracy
# >= 0.9 one block downstream, on my assumption that the residual stream carries a written message
# "essentially untouched" over one block. That assumption is wrong: a decoder block adds a whole
# attention and MLP output to the residual, which is real signal competing with the message. The
# corrected write measured 0.8068 (9.35 of 32 bits) and the 0.9 gate refused it — a gate failing a
# working instrument.
#
# Rather than move a constant after seeing the number, which is the post-unblind rule swap this
# program forbids, the gate now compares against a null it MEASURES: the identical probe run with
# amplitude 0, where no message is written at all. That null is chance by construction, the gate
# is "the write lands above its own null", and it discriminates exactly what the control exists to
# discriminate — a channel that was never opened (the 0.50000-flat first run) from one that is
# open and lossy. Both numbers are recorded.
RIDGE = 1e-2                             # W1-C probe regularizer, frozen

SPEC_EXPECTATIONS = {
    "W1-A": ("the in-core burn is no better than its own twin burns",
             "a core-confined bias trained to match 16-shot activations beats complement/haar-"
             "confined biases of equal size and the 576-param unconstrained incumbent"),
    "W1-B": ("gain on the core channel moves 4-shot no closer to 16-shot than twin gains do",
             "scaling the core-projected context shift moves 4-shot toward 16-shot"),
    "W1-C": ("the core channel carries no more bits per unit perplexity cost than twin channels",
             "an amplitude-a core message is recoverable at layer 17 at a bits/ppl-cost rate "
             "above the twin channels"),
}


# ---------------------------------------------------------------------------
# MMLU prompt pools. Built once and shared by W1-A and W1-B so the two arms cannot drift apart in
# how a "4-shot prompt" is spelled.
# ---------------------------------------------------------------------------
def mmlu_pools(n_shots_max: int, seed: int, n_train_queries: int, max_subjects: int = 0):
    """(exemplars_by_subject, test_by_subject, train_queries, report).

    exemplars come from dev + validation, in that order; train queries are what validation has
    LEFT after the exemplars are taken; test is untouched. A subject without `n_shots_max`
    exemplars is dropped from the shot ladder and COUNTED (§12: never silent shrinkage).
    """
    import random
    from collections import defaultdict
    from datasets import load_dataset
    from infra.pg import text_eval as TE

    ds = load_dataset(TE.MMLU_REPO, "all")
    pool = defaultdict(list)
    for r in ds["dev"]:
        pool[r["subject"]].append(r)
    n_dev = {s: len(v) for s, v in pool.items()}
    for r in ds["validation"]:
        pool[r["subject"]].append(r)
    test_by_subj = defaultdict(list)
    for r in ds["test"]:
        test_by_subj[r["subject"]].append(r)

    # Subject truncation (smoke) happens HERE and nowhere else. Trimming the subject set after
    # the training queries are drawn would leave queries whose subject has no exemplar prefix —
    # a KeyError on the smoke path only, i.e. exactly the class of bug a smoke run exists to find.
    eligible = [s for s in sorted(pool) if len(pool[s]) >= n_shots_max]
    dropped = [s for s in sorted(pool) if len(pool[s]) < n_shots_max]
    if max_subjects:
        eligible = eligible[:max_subjects]
    exemplars, leftover = {}, []
    for subj in eligible:
        exemplars[subj] = pool[subj][:n_shots_max]
        leftover.extend(pool[subj][n_shots_max:])
    assert exemplars, f"no subject has {n_shots_max} exemplars"
    random.Random(seed).shuffle(leftover)
    train_queries = leftover[:n_train_queries]
    assert len(train_queries) == n_train_queries, (
        f"only {len(leftover)} leftover validation items, need {n_train_queries}")
    report = {"n_subjects": len(exemplars), "subjects_dropped": dropped,
              "min_dev_per_subject": min(n_dev.values()),
              "n_leftover": len(leftover), "n_train_queries": len(train_queries),
              "max_subjects": max_subjects or None,
              "split_rule": "exemplars = dev + validation (first n_shots_max); train queries = "
                            "validation leftover; every score is on test"}
    return exemplars, test_by_subj, train_queries, report


def prompt_for(subject, exemplars, n_shots, item=None):
    """The prompt string: subject header + n_shots answered exemplars + (optionally) the query."""
    from infra.pg import text_eval as TE
    p = TE._subject_header(subject)
    for s in exemplars[subject][:n_shots]:
        p += TE._q_block(s["question"], s["choices"], s["answer"])
    if item is not None:
        p += TE._q_block(item["question"], item["choices"])
    return p


# ---------------------------------------------------------------------------
# The three interventions. Each is a context manager over the SAME site, so a difference between
# arms is a difference of what is written, never of where.
# ---------------------------------------------------------------------------
class ConfinedBias:
    """W1-A's treatment: add b_l = scale * (U_l @ c_l) at every layer's site.

    `c` is the only thing that trains. When `coords` is given instead of a basis, the bias lives on
    those residual coordinates directly — that is the `freecoord32` incumbent, same parameter
    count, no bank geometry.
    """

    def __init__(self, layers, device, *, basis_by_layer=None, coords_by_layer=None, scale=1.0):
        import torch
        assert (basis_by_layer is None) != (coords_by_layer is None), \
            "ConfinedBias takes exactly one of basis_by_layer / coords_by_layer"
        self.layers = layers
        self.scale = float(scale)
        self.coords = None
        if basis_by_layer is not None:
            self.U = {L: torch.as_tensor(U, device=device, dtype=torch.float32)
                      for L, U in basis_by_layer.items()}
            self.keys = sorted(self.U)
            k = next(iter(self.U.values())).shape[1]
        else:
            self.U = None
            self.coords = {L: torch.as_tensor(c, device=device, dtype=torch.long)
                           for L, c in coords_by_layer.items()}
            self.keys = sorted(self.coords)
            k = next(iter(self.coords.values())).shape[0]
        self.k = int(k)
        self.c = {L: torch.zeros(self.k, device=device, dtype=torch.float32, requires_grad=True)
                  for L in self.keys}
        self._handles = []

    @property
    def n_params(self) -> int:
        return len(self.keys) * self.k

    def parameters(self):
        return [self.c[L] for L in self.keys]

    def vector(self, L):
        if self.U is not None:
            return self.U[L] @ self.c[L]
        import torch
        v = torch.zeros(self.layers[L].self_attn.q_proj.weight.shape[1],
                        device=self.c[L].device, dtype=torch.float32)
        return v.index_add(0, self.coords[L], self.c[L])

    def _mk(self, L):
        def pre_hook(_m, args):
            x = args[0]
            return (x + (self.scale * self.vector(L)).to(x.dtype),) + tuple(args[1:])
        return pre_hook

    def __enter__(self):
        for L in self.keys:
            self._handles.append(
                self.layers[L].self_attn.q_proj.register_forward_pre_hook(self._mk(L)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


class ContextAmplifier:
    """W1-B's treatment: at the query-block positions, add (g-1) * U Uᵀ (x - h0).

    `h0` is the SAME query block's activation under the 0-shot prompt, captured beforehand, so
    (x - h0) is literally the context shift §4/W1-B names — not a proxy for it. Positions are set
    per item because the query block sits at a different offset in every prompt.
    """

    def __init__(self, layers, device, basis_by_layer, gain: float):
        import torch
        self.layers = layers
        self.P = {L: torch.as_tensor(U, device=device, dtype=torch.float32)
                  for L, U in basis_by_layer.items()}
        self.gain = float(gain)
        self.h0 = {}
        self._win = None
        self._handles = []

    def set_item(self, h0_by_layer, start: int, length: int):
        import torch
        self.h0 = {L: torch.as_tensor(h, dtype=torch.float32) for L, h in h0_by_layer.items()}
        self._win = (int(start), int(length))

    def _mk(self, L):
        def pre_hook(_m, args):
            x = args[0]
            assert x.shape[0] == 1, f"ContextAmplifier requires batch 1, got {x.shape[0]}"
            s, n = self._win
            assert s + n <= x.shape[1], f"window [{s},{s+n}) outside sequence {x.shape[1]}"
            U = self.P[L]
            h0 = self.h0[L].to(x.device)
            blk = x[0, s:s + n, :].to(U.dtype)
            shift = blk - h0
            add = ((shift @ U) @ U.T) * (self.gain - 1.0)
            y = x.clone()
            y[0, s:s + n, :] = (blk + add).to(x.dtype)
            return (y,) + tuple(args[1:])
        return pre_hook

    def __enter__(self):
        for L in sorted(self.P):
            self._handles.append(
                self.layers[L].self_attn.q_proj.register_forward_pre_hook(self._mk(L)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


class Intercom:
    """W1-C's channel: write a per-token +-1 message into the RESIDUAL STREAM at layer 0, read it
    at `layer_read`.

    THE HOOK SITE IS THE DECODER LAYER, NOT q_proj, and that is the whole instrument. A
    forward_pre_hook on `layer.self_attn.q_proj` sees `input_layernorm(hidden_states)` — the
    ATTENTION BRANCH's input, which is consumed by that one projection and then discarded. Writing
    there perturbs layer 0's queries and never enters the residual stream, so nothing can survive
    to layer 17 no matter how large the amplitude. The first run of this arm did exactly that and
    measured bit accuracy 0.500 at every amplitude in every bank — flat chance, which is the
    signature of a channel that was never opened rather than one that is merely lossy. Hooking the
    decoder layer itself, whose `args[0]` IS `hidden_states`, writes into the residual stream.

    (`humboldt_act.BankInject` hooks q_proj deliberately and correctly: W1-H asks what an
    attention-input perturbation does to attention. Different question, different site.)

    The message is drawn per forward from a seeded generator and RECORDED alongside the readout
    activation, so the probe is fit on exactly the messages that were sent. Amplitude `a` is the
    norm of the added vector per token — the message is normalized by sqrt(k) so `a` means the
    same thing for every bank.
    """

    def __init__(self, layers, device, U_write, layer_read: int, amp: float, seed: int):
        import torch
        self.layers = layers
        self.U = torch.as_tensor(U_write, device=device, dtype=torch.float32)
        self.k = int(self.U.shape[1])
        self.amp = float(amp)
        self.layer_read = int(layer_read)
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.sent, self.read, self.host_norm = [], [], []
        self._handles = []

    def _write(self, _m, args):
        import torch
        x = args[0]
        b, t, _ = x.shape
        m = (torch.randint(0, 2, (b, t, self.k), generator=self.gen).float() * 2.0 - 1.0)
        self.sent.append(m.numpy().reshape(-1, self.k).copy())
        # The amplitude is ABSOLUTE, so what it means depends on the stream it is added to. Gemma
        # scales embeddings by sqrt(d_model), and the row is unreadable without that number.
        self.host_norm.append(float(x.detach().float().norm(dim=-1).mean()))
        add = (m.to(x.device) @ self.U.T) * (self.amp / (self.k ** 0.5))
        return (x + add.to(x.dtype),) + tuple(args[1:])

    def _readout(self, _m, args):
        x = args[0]
        self.read.append(x.detach().float().cpu().numpy().reshape(-1, x.shape[-1]).copy())

    def __enter__(self):
        assert self.layer_read > 0, "read layer must be downstream of the write at layer 0"
        self._handles.append(self.layers[0].register_forward_pre_hook(self._write))
        self._handles.append(
            self.layers[self.layer_read].register_forward_pre_hook(self._readout))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


# ---------------------------------------------------------------------------
def banks_by_layer(bank_dir, bank_id, layers):
    from infra.pg import humboldt_weights as W
    return {L: W.load_bank(bank_dir, BANK_TYPE, L, bank_id) for L in layers}


def freecoords_by_layer(n_dim, k, layers, seed):
    import numpy as np
    out = {}
    for L in layers:
        rng = np.random.default_rng(H.cell_seed(seed, "freecoord32", L))
        out[L] = np.sort(rng.choice(n_dim, size=k, replace=False))
    return out


def paired_acc_ci(treat_items, null_items, n_boot=2000, seed=1337):
    """Paired bootstrap of an ACCURACY difference over the same items, in the same order.

    Not `invariant_core_func._paired_contrast_ci`: that one is perplexity-specific — it consumes
    (W,2) [nll_sum, n_tok] arrays and exponentiates a token-weighted mean, which is meaningless on
    0/1 correctness. Resampling the ITEM index once and applying it to both arms is what makes this
    paired; the arms are scored on identical items, which is asserted rather than assumed.
    """
    import numpy as np
    t = np.asarray(treat_items, dtype=np.float64)
    u = np.asarray(null_items, dtype=np.float64)
    assert t.shape == u.shape, f"paired arms have different item counts: {t.shape} vs {u.shape}"
    n = t.shape[0]
    point = float(t.mean() - u.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = t[idx].mean() - u[idx].mean()
    return {"contrast": point, "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)),
            "p_le0": float(np.mean(boots <= 0.0)), "n": int(n)}


def icl_ladder_gate(ladder, seed):
    """Is the 16-shot model actually better than the 0-shot model on this readout?

    Returns the paired lift with its CI and a `valid` flag. Both W1-A and W1-B are DEFINED against
    the 16-shot model, so a ladder that does not rise makes their central quantities ratios with a
    meaningless denominator. Recording the gate turns that from an unexplained null into a stated
    instrument limit.
    """
    r = paired_acc_ci(ladder[max(SHOT_LADDER)]["per_item"], ladder[0]["per_item"], seed=seed)
    return {"lift": r["contrast"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n": r["n"],
            "valid": bool(r["ci_lo"] > ICL_GATE_MIN_LIFT),
            "rule": f"acc({max(SHOT_LADDER)}-shot) - acc(0-shot) CI low end > "
                    f"{ICL_GATE_MIN_LIFT}"}


def choice_letter_ids(tok):
    from infra.pg import text_eval as TE
    ids = []
    for L in TE.CHOICES:
        e = tok.encode(" " + L, add_special_tokens=False)
        assert len(e) == 1, f"' {L}' -> {len(e)} tokens"
        ids.append(e[0])
    assert len(set(ids)) == 4, f"choice-letter ids collide: {ids}"
    return ids


def score_mmlu(model, processor, device, exemplars, test_by_subj, n_shots, n_per_subject,
               batch_size, letter_ids, gray_img, ctx=None):
    """MMLU accuracy under an optional intervention. Returns per-item correctness for pairing.

    Prompt construction is `prompt_for`, i.e. identical to the burn's training prompts and to
    every other cell of this module — the intervention is the only thing that varies.
    """
    import torch
    tok = processor.tokenizer
    orig = tok.padding_side
    tok.padding_side = "left"
    per_item, per_hash, subjects = [], [], []
    from contextlib import nullcontext
    with (ctx if ctx is not None else nullcontext()):
        for subj in sorted(exemplars):
            rows = test_by_subj[subj][:n_per_subject]
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                prompts = [prompt_for(subj, exemplars, n_shots, r) for r in batch]
                inp = processor(text=prompts, images=[gray_img] * len(batch),
                                return_tensors="pt", padding="longest").to(device)
                with torch.inference_mode():
                    last = model(input_ids=inp["input_ids"],
                                 attention_mask=inp["attention_mask"],
                                 pixel_values=inp["pixel_values"].to(torch.bfloat16),
                                 logits_to_keep=1).logits[:, -1, :]
                preds = last[:, letter_ids].argmax(dim=-1).tolist()
                for r, p in zip(batch, preds):
                    per_item.append(int(p == int(r["answer"])))
                    per_hash.append(hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12])
                    subjects.append(subj)
    tok.padding_side = orig
    assert per_item, "score_mmlu produced no items"
    return {"acc": sum(per_item) / len(per_item), "per_item": per_item, "per_hash": per_hash,
            "n": len(per_item), "n_shots": n_shots, "subjects": subjects}


# ---------------------------------------------------------------------------
# W1-A
# ---------------------------------------------------------------------------
def teacher_logprobs(model, processor, device, exemplars, queries, n_shots, gray_img, batch_size):
    """The 16-shot next-token log-distribution on each training query — the burn's target.

    Computed once, kept on CPU in fp16. No gold answers are read; the target IS the model's own
    behaviour with the context present, which is what the burn is asked to internalize.
    """
    import torch
    tok = processor.tokenizer
    orig = tok.padding_side
    tok.padding_side = "left"
    out = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        prompts = [prompt_for(q["subject"], exemplars, n_shots, q) for q in batch]
        inp = processor(text=prompts, images=[gray_img] * len(batch),
                        return_tensors="pt", padding="longest").to(device)
        with torch.inference_mode():
            last = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                         pixel_values=inp["pixel_values"].to(torch.bfloat16),
                         logits_to_keep=1).logits[:, -1, :]
        out.append(torch.log_softmax(last.float(), dim=-1).to(torch.float16).cpu())
    tok.padding_side = orig
    return torch.cat(out, dim=0)


def train_burn(model, processor, device, bias, exemplars, queries, targets, gray_img,
               *, steps, batch_size, lr, log_every=10):
    """Fit `bias.c` so the 0-shot forward reproduces the teacher's next-token distribution.

    Loss = KL(teacher || student) at the answer position. The model is frozen; the only tensors
    with gradients are the 576 coefficients.
    """
    import torch
    tok = processor.tokenizer
    orig = tok.padding_side
    tok.padding_side = "left"
    opt = torch.optim.Adam(bias.parameters(), lr=lr)
    n = len(queries)
    hist = []
    with bias:
        for step in range(steps):
            lo = (step * batch_size) % n
            idx = [(lo + j) % n for j in range(batch_size)]
            batch = [queries[i] for i in idx]
            prompts = [prompt_for(q["subject"], exemplars, 0, q) for q in batch]
            inp = processor(text=prompts, images=[gray_img] * len(batch),
                            return_tensors="pt", padding="longest").to(device)
            logits = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                           pixel_values=inp["pixel_values"].to(torch.bfloat16),
                           logits_to_keep=1).logits[:, -1, :]
            student = torch.log_softmax(logits.float(), dim=-1)
            teacher = targets[idx].to(device).float()
            loss = torch.sum(torch.exp(teacher) * (teacher - student), dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            hist.append(float(loss.detach()))
            if (step + 1) % log_every == 0:
                print(f"    burn step {step+1}/{steps} kl={hist[-1]:.5f}", flush=True)
    tok.padding_side = orig
    return hist


def run_w1_a(model, processor, layers, bank_dir, out, atlas, gates, a, base_sha):
    import numpy as np
    import torch
    from PIL import Image
    from infra.aws_pg.jobs.invariant_core_func import PG1_REPO

    D.check_dose_grid(list(LAM_LADDER), arm="W1-A", min_doses=2 if a.smoke else 4)
    gray_img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    letter_ids = choice_letter_ids(processor.tokenizer)
    n_shots_max = max(SHOT_LADDER)
    exemplars, test_by_subj, queries, pools = mmlu_pools(
        n_shots_max, a.seed, a.n_train_queries, max_subjects=a.max_subjects)
    print(f"[W1-A] pools: {json.dumps(pools)}", flush=True)

    ladder = {}
    for ns in SHOT_LADDER:
        r = score_mmlu(model, processor, "cuda", exemplars, test_by_subj, ns, a.n_per_subject,
                       a.batch_size, letter_ids, gray_img)
        ladder[ns] = r
        atlas.add("mmlu_acc", value=r["acc"], bank_id="none", dose=float(ns),
                  incumbent_id="shot_ladder", n=r["n"], model_id=PG1_REPO, model_sha=base_sha,
                  flags={"n_shots": ns, "role": "reference ladder", "readout_model": PG1_REPO,
                         "site": SITE, "pools": pools})
        print(f"[W1-A] {ns}-shot acc={r['acc']:.4f} (n={r['n']})", flush=True)
    icl = icl_ladder_gate(ladder, a.seed)
    atlas.add("icl_ladder_lift", value=icl["lift"], bank_id="none", dose=float(max(SHOT_LADDER)),
              incumbent_id="zero_shot", n=icl["n"], ci_lo=icl["ci_lo"], ci_hi=icl["ci_hi"],
              model_id=PG1_REPO, model_sha=base_sha,
              flags={"acc_0shot": ladder[0]["acc"], "acc_4shot": ladder[4]["acc"],
                     "acc_16shot": ladder[16]["acc"], "valid": icl["valid"],
                     "outcome": "ICL-LADDER-OK" if icl["valid"] else "VOID-INSTRUMENT",
                     "why": "W1-A trains a bias to reproduce the 16-shot model; if 16-shot is not "
                            "better than 0-shot there is no target to reach"})
    gates.record("W1A_icl_ladder_valid", icl["valid"], **icl)
    print(f"[W1-A] ICL ladder lift {icl['lift']:+.4f} CI[{icl['ci_lo']:+.4f},{icl['ci_hi']:+.4f}] "
          f"-> {'OK' if icl['valid'] else 'VOID-INSTRUMENT'}", flush=True)
    D.s3_sync(out, a.s3_prefix)

    targets = teacher_logprobs(model, processor, "cuda", exemplars, queries, n_shots_max,
                               gray_img, a.batch_size)
    bank_layers = list(range(len(layers)))
    n_dim = layers[0].self_attn.q_proj.weight.shape[1]
    seen, results = set(), {}
    channels = [(bid, "bank") for bid in D.bank_ids_for(BANK_TYPE)] + [("freecoord32", "coords")]
    for bid, kind in channels:
        if kind == "bank":
            bias = ConfinedBias(layers, "cuda",
                                basis_by_layer=banks_by_layer(bank_dir, bid, bank_layers))
        else:
            bias = ConfinedBias(layers, "cuda",
                                coords_by_layer=freecoords_by_layer(n_dim, H.BANK_K, bank_layers,
                                                                    a.seed))
        hist = train_burn(model, processor, "cuda", bias, exemplars, queries, targets, gray_img,
                          steps=a.burn_steps, batch_size=a.burn_batch, lr=a.burn_lr)
        atlas.add("burn_final_kl", value=hist[-1], bank_id=bid, dose=1.0,
                  incumbent_id="freecoord32", model_id=PG1_REPO, model_sha=base_sha,
                  flags={"kl_history": hist, "n_params": bias.n_params, "steps": a.burn_steps,
                         "lr": a.burn_lr, "kind": kind, "readout_model": PG1_REPO,
                         "objective": "KL(16-shot teacher || 0-shot+bias student), unlabeled"})
        best = None
        for lam in LAM_LADDER:
            bias.scale = float(lam)
            with bias:
                r = score_mmlu(model, processor, "cuda", exemplars, test_by_subj, 0,
                               a.n_per_subject, a.batch_size, letter_ids, gray_img)
            ci = paired_acc_ci(r["per_item"], ladder[0]["per_item"], seed=a.seed)
            atlas.add("mmlu_acc_burned", value=r["acc"], bank_id=bid, dose=float(lam),
                      incumbent_id="freecoord32", n=r["n"], ci_lo=ci["ci_lo"], ci_hi=ci["ci_hi"],
                      model_id=PG1_REPO, model_sha=base_sha,
                      flags={"lam": lam, "delta_vs_0shot": r["acc"] - ladder[0]["acc"],
                             "acc_0shot": ladder[0]["acc"], "acc_4shot": ladder[4]["acc"],
                             "acc_16shot": ladder[16]["acc"], "kind": kind,
                             "gap_closed_4": ((r["acc"] - ladder[0]["acc"]) /
                                              (ladder[4]["acc"] - ladder[0]["acc"]))
                                             if ladder[4]["acc"] != ladder[0]["acc"] else None,
                             "gap_closed_16": ((r["acc"] - ladder[0]["acc"]) /
                                               (ladder[16]["acc"] - ladder[0]["acc"]))
                                              if ladder[16]["acc"] != ladder[0]["acc"] else None,
                             "n_params": bias.n_params, "readout_model": PG1_REPO,
                             "site": SITE, "icl_ladder_valid": icl["valid"],
                             "readout_verdict": ("measured" if icl["valid"]
                                                 else "VOID-INSTRUMENT: the 16-shot target is not "
                                                      "above 0-shot on this readout")})
            if best is None or r["acc"] > best[1]:
                best = (lam, r["acc"])
            print(f"[W1-A] {bid} lam={lam}: acc={r['acc']:.4f} "
                  f"(0-shot {ladder[0]['acc']:.4f})", flush=True)
        results[bid] = {"best_lam": best[0], "best_acc": best[1], "final_kl": hist[-1]}
        seen.add(bid)
        D.s3_sync(out, a.s3_prefix)

    D.check_controls(seen, arm="W1-A")
    core_best = results["core32"]["best_acc"]
    twin_best = max(results[t]["best_acc"] for t in H.TWIN_IDS if t in results)
    gates.record("W1A_burn_battery_complete", True, banks=sorted(seen),
                 core_best=core_best, twin_best=twin_best,
                 incumbent_best=results["freecoord32"]["best_acc"],
                 refuter_fires=bool(core_best <= twin_best))
    return {"ladder": {k: v["acc"] for k, v in ladder.items()}, "results": results,
            "core_best": core_best, "twin_best": twin_best, "pools": pools,
            "icl_ladder": icl,
            "verdict": ("refuter_fires" if (icl["valid"] and core_best <= twin_best)
                        else "measured" if icl["valid"] else "VOID-INSTRUMENT"),
            "refuter_fires": bool(icl["valid"] and core_best <= twin_best)}


# ---------------------------------------------------------------------------
# W1-B
# ---------------------------------------------------------------------------
def run_w1_b(model, processor, layers, bank_dir, out, atlas, gates, a, base_sha):
    import numpy as np
    import torch
    from PIL import Image
    from infra.pg import humboldt_act as ACT
    from infra.aws_pg.jobs.invariant_core_func import PG1_REPO, _locate_text_block
    from infra.pg import text_eval as TE

    D.check_dose_grid(list(GAIN_LADDER), arm="W1-B", min_doses=2 if a.smoke else 4,
                      log_spaced=False)
    gray_img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    tok = processor.tokenizer
    letter_ids = choice_letter_ids(tok)
    exemplars, test_by_subj, _, pools = mmlu_pools(
        max(SHOT_LADDER), a.seed, a.n_train_queries, max_subjects=a.max_subjects)
    bank_layers = list(range(len(layers)))

    items = []
    for subj in sorted(exemplars):
        for r in test_by_subj[subj][:a.n_per_subject_b]:
            items.append((subj, r))
    print(f"[W1-B] {len(items)} items x {len(GAIN_LADDER)} gains x banks", flush=True)

    def one(subj, item, n_shots, ctx=None):
        """Batch-1 scored forward; returns (correct, input_ids, query-block offset/length)."""
        from contextlib import nullcontext
        text = prompt_for(subj, exemplars, n_shots, item)
        block = TE._q_block(item["question"], item["choices"])
        inp = processor(text=[text], images=[gray_img], return_tensors="pt").to("cuda")
        win_ids = torch.as_tensor(tok.encode(block, add_special_tokens=False))
        start = _locate_text_block(inp["input_ids"][0], win_ids)
        with (ctx if ctx is not None else nullcontext()), torch.inference_mode():
            last = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                         pixel_values=inp["pixel_values"].to(torch.bfloat16),
                         logits_to_keep=1).logits[:, -1, :]
        pred = int(last[0, letter_ids].argmax())
        return int(pred == int(item["answer"])), start, int(win_ids.shape[0])

    # 0-shot activations over the query block, per item — the h0 the amplifier subtracts.
    h0_by_item, base = {}, {0: [], 4: [], 16: []}
    for i, (subj, item) in enumerate(items):
        text0 = prompt_for(subj, exemplars, 0, item)
        block = TE._q_block(item["question"], item["choices"])
        inp = processor(text=[text0], images=[gray_img], return_tensors="pt").to("cuda")
        win_ids = torch.as_tensor(tok.encode(block, add_special_tokens=False))
        s0 = _locate_text_block(inp["input_ids"][0], win_ids)
        n0 = int(win_ids.shape[0])
        slicer = ACT.SiteSlice(layers, "cuda", sites=(SITE,))
        with slicer, torch.inference_mode():
            slicer.set_window(s0, n0)
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"].to(torch.bfloat16), logits_to_keep=1)
        h0_by_item[i] = {L: slicer.buf[(L, SITE)] for L in bank_layers}
        for ns in (0, 4, 16):
            ok, _, _ = one(subj, item, ns)
            base[ns].append(ok)
    for ns in (0, 4, 16):
        acc = sum(base[ns]) / len(base[ns])
        atlas.add("mmlu_acc", value=acc, bank_id="none", dose=float(ns),
                  incumbent_id="shot_ladder", n=len(base[ns]), model_id=PG1_REPO,
                  model_sha=base_sha,
                  flags={"n_shots": ns, "role": "reference ladder", "batch": 1,
                         "readout_model": PG1_REPO, "site": SITE})
        print(f"[W1-B] {ns}-shot acc={acc:.4f}", flush=True)
    D.s3_sync(out, a.s3_prefix)

    gap = (sum(base[16]) - sum(base[4])) / len(base[4])
    icl = icl_ladder_gate({k: {"acc": sum(v) / len(v), "per_item": v} for k, v in base.items()},
                          a.seed)
    atlas.add("icl_ladder_lift", value=icl["lift"], bank_id="none", dose=float(max(SHOT_LADDER)),
              incumbent_id="zero_shot", n=icl["n"], ci_lo=icl["ci_lo"], ci_hi=icl["ci_hi"],
              model_id=PG1_REPO, model_sha=base_sha,
              flags={"valid": icl["valid"], "gap_4_to_16": gap,
                     "outcome": "ICL-LADDER-OK" if icl["valid"] else "VOID-INSTRUMENT",
                     "why": "W1-B asks whether a gain moves 4-shot TOWARD 16-shot; a 16-shot that "
                            "is not above 4-shot makes the gap, and every gap_closed, meaningless"})
    gates.record("W1B_icl_ladder_valid", icl["valid"], **icl, gap_4_to_16=gap)
    print(f"[W1-B] ICL ladder lift {icl['lift']:+.4f} -> "
          f"{'OK' if icl['valid'] else 'VOID-INSTRUMENT'}", flush=True)
    seen, results = set(), {}
    for bid in D.bank_ids_for(BANK_TYPE):
        Us = banks_by_layer(bank_dir, bid, bank_layers)
        for g in GAIN_LADDER:
            amp = ContextAmplifier(layers, "cuda", Us, g)
            per_item = []
            with amp:
                for i, (subj, item) in enumerate(items):
                    text = prompt_for(subj, exemplars, 4, item)
                    block = TE._q_block(item["question"], item["choices"])
                    inp = processor(text=[text], images=[gray_img], return_tensors="pt").to("cuda")
                    win_ids = torch.as_tensor(tok.encode(block, add_special_tokens=False))
                    s4 = _locate_text_block(inp["input_ids"][0], win_ids)
                    n4 = int(win_ids.shape[0])
                    assert n4 == h0_by_item[i][0].shape[0], (
                        f"query block length differs across shot counts: {n4} vs "
                        f"{h0_by_item[i][0].shape[0]} — the shift would not be elementwise")
                    amp.set_item(h0_by_item[i], s4, n4)
                    with torch.inference_mode():
                        last = model(input_ids=inp["input_ids"],
                                     attention_mask=inp["attention_mask"],
                                     pixel_values=inp["pixel_values"].to(torch.bfloat16),
                                     logits_to_keep=1).logits[:, -1, :]
                    per_item.append(int(int(last[0, letter_ids].argmax()) == int(item["answer"])))
            acc = sum(per_item) / len(per_item)
            ci = paired_acc_ci(per_item, base[4], seed=a.seed)
            closed = ((acc - sum(base[4]) / len(base[4])) / gap) if gap else None
            atlas.add("mmlu_acc_amplified", value=acc, bank_id=bid, dose=float(g),
                      incumbent_id="gain_1", n=len(per_item), ci_lo=ci["ci_lo"],
                      ci_hi=ci["ci_hi"], model_id=PG1_REPO, model_sha=base_sha,
                      flags={"gain": g, "acc_4shot": sum(base[4]) / len(base[4]),
                             "acc_16shot": sum(base[16]) / len(base[16]),
                             "gap_4_to_16": gap, "gap_closed": closed,
                             "readout_model": PG1_REPO, "site": SITE,
                             "shift": "h_4shot - h_0shot, projected on the bank, scaled (g-1)"})
            results.setdefault(bid, {})[g] = acc
            print(f"[W1-B] {bid} g={g}: acc={acc:.4f} closed={closed}", flush=True)
        seen.add(bid)
        D.s3_sync(out, a.s3_prefix)

    D.check_controls(seen, arm="W1-B")
    core_best = max(results["core32"].values())
    twin_best = max(max(results[t].values()) for t in H.TWIN_IDS if t in results)
    gates.record("W1B_amplifier_battery_complete", True, banks=sorted(seen),
                 core_best=core_best, twin_best=twin_best,
                 refuter_fires=bool(core_best <= twin_best))
    return {"base": {k: sum(v) / len(v) for k, v in base.items()}, "results": results,
            "core_best": core_best, "twin_best": twin_best, "icl_ladder": icl,
            "verdict": ("refuter_fires" if (icl["valid"] and core_best <= twin_best)
                        else "measured" if icl["valid"] else "VOID-INSTRUMENT"),
            "refuter_fires": bool(icl["valid"] and core_best <= twin_best)}


# ---------------------------------------------------------------------------
# W1-C
# ---------------------------------------------------------------------------
def probe_bits(sent, read, ridge, n_fit):
    """Least-squares probe h -> m on the fit half; per-bit sign accuracy and bits on the rest.

    bits/token = sum_i (1 - H2(acc_i)) — the standard binary-symmetric-channel accounting, one
    independent bit per message coordinate.
    """
    import numpy as np
    Xf, Yf = read[:n_fit], sent[:n_fit]
    Xh, Yh = read[n_fit:], sent[n_fit:]
    assert len(Xh) > 0, "no held-out tokens for the intercom probe"
    Xf1 = np.concatenate([Xf, np.ones((Xf.shape[0], 1))], axis=1)
    Xh1 = np.concatenate([Xh, np.ones((Xh.shape[0], 1))], axis=1)
    A = Xf1.T @ Xf1 + ridge * np.eye(Xf1.shape[1])
    Wp = np.linalg.solve(A, Xf1.T @ Yf)
    pred = np.sign(Xh1 @ Wp)
    pred[pred == 0] = 1.0
    acc = (pred == Yh).mean(axis=0)
    p = np.clip(acc, 1e-6, 1 - 1e-6)
    h2 = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return {"per_bit_acc": acc.tolist(), "mean_bit_acc": float(acc.mean()),
            "bits_per_token": float(np.sum(1.0 - h2)), "n_fit": int(Xf.shape[0]),
            "n_held": int(Xh.shape[0])}


def run_w1_c(model, processor, layers, bank_dir, out, atlas, gates, a, base_sha):
    import numpy as np
    from infra.aws_pg.jobs.invariant_core_func import PG1_REPO, _ppl_blankimg

    D.check_dose_grid(list(AMP_LADDER), arm="W1-C", min_doses=2 if a.smoke else 4)
    clean = _ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)
    from infra.pg import humboldt_weights as WW
    atlas.add("ppl_clean", value=float(clean["ppl"]), bank_id="none", dose=0.0,
              incumbent_id="none", model_id=PG1_REPO, model_sha=base_sha,
              flags={"n_seq": a.n_seq, "role": "intercom reference", "readout_model": PG1_REPO})
    print(f"[W1-C] clean ppl={clean['ppl']:.5f}", flush=True)

    # POSITIVE CONTROL, run first and gated on. A message written at layer 0 and read one layer
    # later must be almost perfectly recoverable — the residual stream carries it essentially
    # untouched over one block. If it is not, the WRITE is not landing, and every downstream
    # amplitude curve would be a measurement of nothing dressed as a null
    # (feedback_signals_that_cannot_fail).
    audit_U = WW.load_bank(bank_dir, BANK_TYPE, 0, "core32")
    audit_n = max(2, a.n_seq // 4)

    def audit_at(amp, tag):
        ic = Intercom(layers, "cuda", audit_U, AUDIT_READ_LAYER, amp,
                      H.cell_seed(a.seed, "audit", int(amp * 1000)))
        with ic:
            _ppl_blankimg(model, processor, "cuda", n_seq=audit_n)
        sent = np.concatenate(ic.sent, axis=0)
        read = np.concatenate(ic.read, axis=0)
        pr = probe_bits(sent, read, RIDGE, int(sent.shape[0] * 0.5))
        hn = float(np.mean(ic.host_norm))
        print(f"[W1-C] audit {tag} (amp={amp}): bit_acc={pr['mean_bit_acc']:.4f} "
              f"bits={pr['bits_per_token']:.2f} host_norm={hn:.2f}", flush=True)
        return pr, hn

    a_pr, host_norm = audit_at(max(AMP_LADDER), "signal")
    a_null, _ = audit_at(0.0, "null")       # no message written: chance by construction
    lift = a_pr["mean_bit_acc"] - a_null["mean_bit_acc"]
    write_lands = bool(lift > AUDIT_MIN_LIFT)
    atlas.add("intercom_write_audit", value=a_pr["mean_bit_acc"], bank_id="core32",
              dose=float(max(AMP_LADDER)), incumbent_id="amplitude_zero_null", n=a_pr["n_held"],
              model_id=PG1_REPO, model_sha=base_sha,
              flags={"read_layer": AUDIT_READ_LAYER, "write_layer": 0,
                     "bits_per_token": a_pr["bits_per_token"],
                     "null_bit_acc": a_null["mean_bit_acc"],
                     "null_bits_per_token": a_null["bits_per_token"],
                     "lift": lift, "threshold_lift": AUDIT_MIN_LIFT,
                     "outcome": "WRITE-LANDS" if write_lands else "VOID-INSTRUMENT",
                     "why": "a message written into the residual stream at layer 0 must be "
                            "recoverable one layer later ABOVE the amplitude-0 null; if it is "
                            "not, every amplitude curve below is a measurement of nothing"})
    gates.record("W1C_write_lands", write_lands, mean_bit_acc=a_pr["mean_bit_acc"],
                 null_bit_acc=a_null["mean_bit_acc"], lift=lift,
                 threshold_lift=AUDIT_MIN_LIFT, read_layer=AUDIT_READ_LAYER)
    assert write_lands, (
        f"W1-C write audit failed: bit accuracy {a_pr['mean_bit_acc']:.4f} against an "
        f"amplitude-0 null of {a_null['mean_bit_acc']:.4f} (lift {lift:.4f} <= "
        f"{AUDIT_MIN_LIFT}). The channel is not open; running the amplitude ladder would "
        f"produce a null that means nothing.")

    seen, results = set(), {}
    for bid in D.bank_ids_for(BANK_TYPE):
        U0 = WW.load_bank(bank_dir, BANK_TYPE, 0, bid)
        for amp in AMP_LADDER:
            ic = Intercom(layers, "cuda", U0, READOUT_LAYER, amp,
                          H.cell_seed(a.seed, bid, int(amp * 1000)))
            with ic:
                r = _ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)
            sent = np.concatenate(ic.sent, axis=0)
            read = np.concatenate(ic.read, axis=0)
            assert sent.shape[0] == read.shape[0], (
                f"intercom write/read token counts differ: {sent.shape[0]} vs {read.shape[0]}")
            n_fit = int(sent.shape[0] * 0.5)
            pr = probe_bits(sent, read, RIDGE, n_fit)
            dppl = float(r["ppl"]) - float(clean["ppl"])
            atlas.add("intercom_bits_per_token", value=pr["bits_per_token"], bank_id=bid,
                      dose=float(amp), incumbent_id="twin_channel", n=pr["n_held"],
                      model_id=PG1_REPO, model_sha=base_sha,
                      flags={"amp": amp, "mean_bit_acc": pr["mean_bit_acc"], "ppl": float(r["ppl"]),
                             "ppl_clean": float(clean["ppl"]), "delta_ppl": dppl,
                             "bits_per_ppl": (pr["bits_per_token"] / dppl) if dppl > 0 else None,
                             "write_layer": 0, "read_layer": READOUT_LAYER, "ridge": RIDGE,
                             "n_fit": pr["n_fit"], "readout_model": PG1_REPO, "site": SITE})
            atlas.add("intercom_delta_ppl", value=dppl, bank_id=bid, dose=float(amp),
                      incumbent_id="twin_channel", model_id=PG1_REPO, model_sha=base_sha,
                      flags={"amp": amp, "bits_per_token": pr["bits_per_token"]})
            results.setdefault(bid, {})[amp] = {"bits": pr["bits_per_token"], "dppl": dppl,
                                                "bit_acc": pr["mean_bit_acc"]}
            print(f"[W1-C] {bid} a={amp}: bits={pr['bits_per_token']:.3f} "
                  f"acc={pr['mean_bit_acc']:.4f} dppl={dppl:+.4f}", flush=True)
        seen.add(bid)
        D.s3_sync(out, a.s3_prefix)

    D.check_controls(seen, arm="W1-C")
    def rate(bid):
        vals = [v["bits"] / v["dppl"] for v in results[bid].values() if v["dppl"] > 0]
        return max(vals) if vals else 0.0
    core_rate = rate("core32")
    twin_rate = max(rate(t) for t in H.TWIN_IDS if t in results)

    # THE DEPTH GATE. The write audit proves the channel is open AT LAYER 1. That says nothing
    # about layer 17, and the ladder here measured bit accuracy 0.500 with 0.00 bits at every
    # amplitude in every bank. When no bank carries any bits, `bits/delta_ppl` is a ratio of noise
    # to noise, and "core <= twin" on those ratios is not a refutation of anything. The arm
    # therefore reports VOID-NO-CHANNEL-AT-DEPTH and declines the comparison, recording the host
    # residual norm so a reader can see what fraction of the stream the amplitude ladder actually
    # was.
    max_bits = max(v["bits"] for bank in results.values() for v in bank.values())
    channel_at_depth = bool(max_bits > CHANNEL_MIN_BITS)
    atlas.add("channel_at_depth", value=max_bits, bank_id="none",
              dose=float(max(AMP_LADDER)), incumbent_id="none", model_id=PG1_REPO,
              model_sha=base_sha,
              flags={"read_layer": READOUT_LAYER, "threshold_bits": CHANNEL_MIN_BITS,
                     "host_residual_norm": host_norm,
                     "amplitude_as_fraction_of_host": {str(x): x / host_norm
                                                       for x in AMP_LADDER},
                     "outcome": ("CHANNEL-AT-DEPTH" if channel_at_depth
                                 else "VOID-NO-CHANNEL-AT-DEPTH"),
                     "why": "with 0 bits recovered anywhere, bits-per-ppl is a ratio of noise to "
                            "noise and the core-vs-twin comparison is undefined"})
    gates.record("W1C_channel_at_depth", channel_at_depth, max_bits=max_bits,
                 threshold_bits=CHANNEL_MIN_BITS, read_layer=READOUT_LAYER,
                 host_residual_norm=host_norm)
    gates.record("W1C_channel_battery_complete", True, banks=sorted(seen),
                 core_bits_per_ppl=core_rate, twin_bits_per_ppl=twin_rate,
                 channel_at_depth=channel_at_depth,
                 refuter_fires=bool(channel_at_depth and core_rate <= twin_rate))
    print(f"[W1-C] depth gate: max bits {max_bits:.3f} over all banks/amplitudes -> "
          f"{'CHANNEL' if channel_at_depth else 'VOID-NO-CHANNEL-AT-DEPTH'}; host residual norm "
          f"{host_norm:.2f}", flush=True)
    return {"clean_ppl": float(clean["ppl"]), "results": results,
            "core_bits_per_ppl": core_rate, "twin_bits_per_ppl": twin_rate,
            "max_bits": max_bits, "channel_at_depth": channel_at_depth,
            "host_residual_norm": host_norm,
            "verdict": ("refuter_fires" if (channel_at_depth and core_rate <= twin_rate)
                        else "measured" if channel_at_depth else "VOID-NO-CHANNEL-AT-DEPTH"),
            "refuter_fires": bool(channel_at_depth and core_rate <= twin_rate)}


RUNNERS = {"W1-A": run_w1_a, "W1-B": run_w1_b, "W1-C": run_w1_c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--n-per-subject", type=int, default=8)      # W1-A/W1-C scored items/subject
    ap.add_argument("--n-per-subject-b", type=int, default=3)    # W1-B is batch-1; keep it small
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-train-queries", type=int, default=256)  # §4/W1-A: "256 unlabeled queries"
    ap.add_argument("--burn-steps", type=int, default=200)
    ap.add_argument("--burn-batch", type=int, default=4)
    ap.add_argument("--burn-lr", type=float, default=3e-3)
    ap.add_argument("--n-seq", type=int, default=24)             # W1-C ppl windows
    ap.add_argument("--max-subjects", type=int, default=0,
                    help="0 = all MMLU subjects; smoke sets 2. Applied inside mmlu_pools so the "
                         "training queries can never reference a subject with no exemplars.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.run_id = a.run_id or f"darwin_{a.arm.lower().replace('-', '_')}"
    a.s3_prefix = a.s3_prefix or f"{D.PROGRAM}/{a.run_id}"
    if a.smoke:
        a.n_per_subject, a.n_per_subject_b, a.n_seq = 1, 1, 3
        # 2 subjects x (dev 5 + validation ~11) leaves only a handful of items after the 16
        # exemplars are taken, so the smoke's query budget has to fit what 2 subjects can spare.
        # The full run draws from all 57 subjects and has ~900 spare validation items for 256.
        a.n_train_queries, a.burn_steps, a.max_subjects, a.burn_batch = 2, 3, 2, 2

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    from infra.aws_pg.jobs.invariant_core_func import (PG1_REPO, _decoder_layers, _load_full_pg1)

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        refuter, direction = SPEC_EXPECTATIONS[a.arm]
        exp.log(a.arm, refuter=refuter, direction=direction, utc=utc)
    ref = exp.require(a.arm)

    base_sha = HfApi().model_info(PG1_REPO).sha
    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))
    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="ATLAS", wave="W1",
                    expected_ref=ref)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "ATLAS",
                "wave": "W1", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "base_repo": PG1_REPO, "base_sha": base_sha, "readout_model": PG1_REPO,
                "readout_model_note": "PG1's decoder IS gemma-2b and every bank is keyed to it; a "
                                      "separate gemma-2b load would put treatment and banks in "
                                      "different coordinate frames",
                "site": SITE, "bank_type": BANK_TYPE, "shot_ladder": list(SHOT_LADDER),
                "lam_ladder": list(LAM_LADDER), "gain_ladder": list(GAIN_LADDER),
                "amp_ladder": list(AMP_LADDER), "read_layer": READOUT_LAYER,
                "n_per_subject": a.n_per_subject, "n_train_queries": a.n_train_queries,
                "max_subjects": a.max_subjects or None,
                "burn_steps": a.burn_steps, "burn_lr": a.burn_lr,
                "expected_ref": ref, "started_utc": utc, "seed": a.seed}
    D.write_json(out, "manifest.json", manifest)

    model = _load_full_pg1(base_sha, "cuda")
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)
    layers = _decoder_layers(model)

    result = RUNNERS[a.arm](model, processor, layers, bank_dir, out, atlas, gates, a, base_sha)

    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, f"{a.arm.lower().replace('-', '_')}_summary.json",
                 {"manifest": manifest, "atlas": rep, "gates": gates.gates, "result": result,
                  "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_{a.arm.replace('-', '_')}_OK rows={rep['n_rows']} "
          f"{json.dumps(result, default=str)[:300]}")


if __name__ == "__main__":
    main()
