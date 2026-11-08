"""The screen instrument: every MLP neuron read on random-token sequences whose second pass is a repeat of the first, a shuffle of it, its
reverse, a low-loss natural passage, or fresh ids (the control A), with the per-position loss of every input logged. Written against
infra/neurons; the job shape is neurons_job. Replaces the sequence and accumulator part of infra/pg/mechanics.py and
infra/aws_pg/jobs/mechanics_screen.py for these readings (the head scores, families and contexts of the mechanics screen stay there).

Items: {"n_seq", "half", "kinds": subset of KINDS with "A", "vocab": {"ids": [...]} or {"n_top": 1000} (banks.sampling_ids on the tokenizer
and corpus), "natural_pool": [global starts of passages of `half` tokens inside one harvest window], "early": 16}.
Sequences: x = V[default_rng(0).integers(0, |V|, (n_seq, half))] (mechanics' repeated set); pass 2 per kind: repeat x; shuffled x with its
positions permuted by default_rng(14).permutation(half); reversed x[:, ::-1]; natural the n_seq pool passages with the lowest mean loss read
alone after BOS, in ascending order; A rows V[default_rng(1).integers(0, |V|, (n_seq, 2 half))] (mechanics' set A, both passes fresh).
Every row is [BOS] + pass 1 + pass 2; sequence position p is model index p + 1.
Bank (screen.npz): per kind D [L, W] = mean a over pass 2 minus mean a over pass 1, D_early (pass 2 cut to its first `early` positions), loss
[2 half - 1] (mean next-token loss per position), mean_loss_pass2; the natural selection (pool losses, the chosen starts); V.
Gates:
  known_activation  max over layers of |act(g) u - the down_proj input| / max |down_proj input| on one batch <= 1e-2; failing input u from the
                    next layer. Identity: 0 for a forward against itself.
  identity          logits with every tap attached equal the logits without (torch.equal); failing input one neuron's input moved by 5 (write.added).
  sequences         repeat equals pass 1, reversed its flip, shuffled the same ids in other places on every row, and the natural rows' 2-gram repeat
                    share (pass-2 positions whose 2-gram occurs earlier in the row) at most 0.1; failing input the repeat rows read as natural.
  selection         the chosen passages' mean loss below the pool's median; failing input the n_seq highest-loss passages.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F

from infra.aws_pg.jobs import neurons_job as NJ
from infra.neurons import banks as B
from infra.neurons import capture as C
from infra.neurons import rules as R
from infra.neurons import write as Wr

INSTRUMENT = "screen"
KINDS = ("repeat", "shuffled", "reversed", "natural", "A")
SEEDS = {"rep": 0, "A": 1, "shuffle": 14}
KNOWN_TOL, SHARE_MAX = 1e-2, 0.1


def check_items(items: dict):
    assert items["instrument"] == INSTRUMENT and set(items["kinds"]) <= set(KINDS) and "A" in items["kinds"], items.get("kinds")
    assert ("ids" in items["vocab"]) != ("n_top" in items["vocab"]), "the vocabulary is given as ids or as the n_top rule, one of them"
    assert "natural" not in items["kinds"] or len(items["natural_pool"]) >= 2 * items["n_seq"], "the pool needs twice n_seq passages"
    assert all(int(t) % items["hw"] + items["half"] <= items["hw"] for t in items.get("natural_pool", [])), "a pool passage crosses its window"


def repeat_share(row, half: int) -> float:
    """Share of pass-2 positions whose 2-gram (previous id, id) occurs earlier in the row."""
    seen, hits = set(), 0
    for i in range(1, len(row)):
        g = (int(row[i - 1]), int(row[i]))
        hits += i >= half + 1 and g in seen
        seen.add(g)
    return hits / (len(row) - half - 1)


def sequences(V, n: int, half: int) -> tuple:
    """x [n, half], the second passes repeat, reversed and shuffled, and the A rows [n, 2 half]."""
    x = V[np.random.default_rng(SEEDS["rep"]).integers(0, V.size, (n, half))]
    second = {"repeat": x, "reversed": x[:, ::-1], "shuffled": x[:, np.random.default_rng(SEEDS["shuffle"]).permutation(half)]}
    return x, second, V[np.random.default_rng(SEEDS["A"]).integers(0, V.size, (n, 2 * half))]


class Job(NJ.Run):
    def rows_loss(self, rows: np.ndarray, tap=None) -> tuple:
        """Per-position next-token loss [n, S - 1] of [BOS] + row, in batches; tap(layer, a) sees every down_proj input."""
        bos, out = self.items["bos"], []
        for b0 in range(0, len(rows), self.batch):
            x = torch.as_tensor(np.c_[np.full(len(rows[b0:b0 + self.batch]), bos), rows[b0:b0 + self.batch]], device=self.device)
            with C.mlp_tap(self.ad.layers, tap or (lambda l, a: None)):
                lg = C.forward(self.model, x, True).logits.float()
            out.append(F.cross_entropy(lg[:, 1:-1].transpose(1, 2), x[:, 2:], reduction="none").cpu().numpy())
        return np.concatenate(out)

    def vocab(self) -> np.ndarray:
        if "ids" in self.items["vocab"]:
            return np.asarray(self.items["vocab"]["ids"], np.int64)
        V, rec = B.sampling_ids(self.tok.added_tokens_decoder, len(self.tok), self.tokens, self.items["vocab"]["n_top"])
        self.m["vocab"] = rec
        return V

    def gates_first(self, x):
        layers, keep, err, err_f = self.ad.layers, {}, [], []
        hs = [l.mlp.gate_proj.register_forward_hook(lambda m, a, o, i=i: keep.__setitem__(("g", i), o.float())) for i, l in enumerate(layers)]
        hs += [l.mlp.up_proj.register_forward_hook(lambda m, a, o, i=i: keep.__setitem__(("u", i), o.float())) for i, l in enumerate(layers)]
        with C.mlp_tap(layers, lambda l, a: keep.__setitem__(("a", l), a.float())):
            tapped = C.forward(self.model, x, True).logits
        for h in hs:
            h.remove()
        for l in range(len(layers)):
            a_m, rec = keep["a", l], layers[l].mlp.act_fn(keep["g", l])
            err.append(float((rec * keep["u", l] - a_m).abs().max() / a_m.abs().max()))
            err_f.append(float((rec * keep["u", (l + 1) % len(layers)] - a_m).abs().max() / a_m.abs().max()))
        self.gate("known_activation", R.gate(max(err), max(err) <= KNOWN_TOL, min(err_f), min(err_f) > KNOWN_TOL, "u from the next layer", 0.0, None, ["metrics.json"]))
        plain = C.forward(self.model, x, True).logits
        with Wr.added(layers, 1, 3, 5.0):
            moved = C.forward(self.model, x, True).logits
        self.gate("identity", R.gate(float((tapped - plain).abs().max()), torch.equal(tapped, plain), float((moved - plain).abs().max()),
                                     not torch.equal(moved, plain), "one neuron's input moved by 5", 0.0, None, ["metrics.json"]))

    def run(self):
        it, n, half = self.items, self.items["n_seq"], self.items["half"]
        V = self.vocab()
        x, second, A = sequences(V, n, half)
        if "natural" in it["kinds"]:
            pool = np.asarray(it["natural_pool"], np.int64)
            psg = np.stack([self.tokens[t:t + half] for t in pool]).astype(np.int64)
            pl = self.rows_loss(psg).mean(1)
            order = np.argsort(pl, kind="stable")
            second["natural"] = psg[order[:n]]
            self.gate("selection", R.gate(float(pl[order[:n]].mean()), float(pl[order[:n]].mean()) < float(np.median(pl)), float(pl[order[-n:]].mean()),
                                          not float(pl[order[-n:]].mean()) < float(np.median(pl)), "the n_seq highest-loss passages", None, "a selection", ["screen.npz"]))
            self.m["natural"] = {"pool_loss_median": float(np.median(pl)), "chosen_loss_mean": float(pl[order[:n]].mean())}
        rows = {k: np.c_[x, second[k]] for k in it["kinds"] if k != "A"}
        rows["A"] = A
        self.gates_first(torch.as_tensor(np.c_[np.full(min(n, self.batch), it["bos"]), rows["A"][:self.batch]], device=self.device))
        ok = (np.array_equal(rows["repeat"][:, half:], x) if "repeat" in rows else True) and \
             (np.array_equal(rows["reversed"][:, half:], x[:, ::-1]) if "reversed" in rows else True) and \
             (all(sorted(r[half:]) == sorted(r[:half]) and not np.array_equal(r[half:], r[:half]) for r in rows["shuffled"]) if "shuffled" in rows else True)
        share = float(np.mean([repeat_share(r, half) for r in rows["natural"]])) if "natural" in rows else 0.0
        share_f = float(np.mean([repeat_share(r, half) for r in rows["repeat"]])) if "repeat" in rows else 1.0
        self.gate("sequences", R.gate(share, ok and share <= SHARE_MAX, share_f, share_f > SHARE_MAX, "the repeat rows read as natural", None,
                                      "a construction check", ["screen.npz"]))
        L, W, out = len(self.ad.layers), self.ad.width, {"V": V}
        for k, r in rows.items():
            S = torch.zeros(L, 2 * half, W, device=self.device)

            def acc(l, a):
                S[l] += a[:, 1:].float().sum(0)
            loss = self.rows_loss(r, acc)
            m = (S / len(r)).double().cpu().numpy()
            out[f"{k}__D"] = m[:, half:].mean(1) - m[:, :half].mean(1)
            out[f"{k}__D_early"] = m[:, half:half + it.get("early", 16)].mean(1) - m[:, :half].mean(1)
            out[f"{k}__loss"] = loss.mean(0)
            self.m["cases"][k] = {"n_rows": len(r), "mean_loss_pass1": float(loss[:, :half - 1].mean()), "mean_loss_pass2": float(loss[:, half - 1:].mean())}
            self.write()
        self.save("screen", **out)
        self.done()


def dry_checks(model, tokens, offsets, items) -> list:
    """The sequences against infra/pg/mechanics' (the repeated set, set A and the shuffled second pass), exactly."""
    from infra.pg import mechanics as M
    V = np.asarray(items["vocab"]["ids"], np.int64)
    x, second, A = sequences(V, items["n_seq"], items["half"])
    ref = M.sequences(V, items["n_seq"], items["half"], n_narrow=V.size)
    assert np.array_equal(np.c_[x, x], ref["rep"]) and np.array_equal(A, ref["A"])
    assert all(np.array_equal(np.r_[r[:items["half"]], s_], M.shuffled_pass2(np.r_[r, r], items["half"])) for r, s_ in zip(x, second["shuffled"]))
    return ["EQUAL the repeated set, set A and the shuffled second pass = infra/pg/mechanics' sequences and shuffled_pass2, exactly"]


def dry_items(model, tokens, offsets, d: dict) -> dict:
    """32 rows of 2 x 24 ids from the fake corpus's vocabulary, every kind, a pool of 64 passages of 24 tokens inside their windows."""
    rng = np.random.default_rng(2)
    starts = [int(t) for t in rng.choice(len(tokens) - 32, 400, replace=False) if t % d["hw"] + 24 <= d["hw"]][:64]
    return {"n_seq": 32, "half": 24, "kinds": list(KINDS), "vocab": {"ids": list(range(3, d["vocab"]))}, "natural_pool": starts, "early": 8}


if __name__ == "__main__":
    NJ.main(sys.modules[__name__])
