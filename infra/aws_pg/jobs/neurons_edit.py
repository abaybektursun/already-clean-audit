"""The edit instrument: a rank-one edit of a set of down_proj columns along a unit direction u (W[:, cols] <- W[:, cols] - s u u^T W[:, cols]),
read as the change in next-token loss on whole windows, with an edit's strength optionally matched to another edit's Frobenius norm. Written
against infra/neurons (write.edit_columns, write.strength_matching); the job shape is neurons_job. Replaces the edit path of
infra/aws_pg/jobs/alwayson_edit.py and infra/pg/alwayson_edit.py (the column edit and the strength match; the generation flags stay there).

Items: {"windows": [global starts of whole harvest windows], "edits": [{"edit", "layer", "cols": [...], "direction": {"kind": "random", "seed"}
or {"kind": "mean_of_columns"}, "strength": s} or {..., "match": <edit id>} (the strength at which this edit's Frobenius norm equals that one's)]}.
Per edit: the per-window mean loss with and without the edit, their difference (edited minus clean), mean over windows with a bootstrap over
windows (rules.boot_matrix, seed 0); the Frobenius norm of the edit; the strength used.
Bank: edit.npz (clean [n_windows], per edit <edit>__loss [n_windows], <edit>__u [hidden]); metrics.json one record per edit.
Gates:
  identity    a strength-0 edit leaves the logits equal (torch.equal); failing input a strength-1 edit (must change them).
  restore     after every edit the weight equals the original bit for bit; failing input the edited weight against the original (must differ).
  frobenius   every matched edit's norm equals its target's to 1e-5 (relative); failing input the matched edit at the target's strength.
"""
from __future__ import annotations

import contextlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

from infra.aws_pg.jobs import neurons_job as NJ
from infra.neurons import rules as R
from infra.neurons import write as Wr

INSTRUMENT = "edit"
FROB_TOL = 1e-5


def check_items(items: dict):
    assert items["instrument"] == INSTRUMENT and items["windows"] and items["edits"], "edit items need windows and edits"
    ids = [e["edit"] for e in items["edits"]]
    assert len(set(ids)) == len(ids) and all(("strength" in e) != ("match" in e) for e in items["edits"]), "each edit has a strength or a match"
    assert all(e["match"] in ids and e["match"] != e["edit"] for e in items["edits"] if "match" in e), "a match names another edit"
    assert all(int(t) % items["hw"] == 0 for t in items["windows"]), "a window starts at a harvest window boundary"


def direction(W: np.ndarray, e: dict) -> np.ndarray:
    if e["direction"]["kind"] == "random":
        u = np.random.default_rng(e["direction"]["seed"]).normal(size=W.shape[0])
    else:
        assert e["direction"]["kind"] == "mean_of_columns", e["direction"]
        u = W[:, e["cols"]].astype(np.float64).mean(1)
    return (u / np.linalg.norm(u)).astype(np.float32)


@contextlib.contextmanager
def edited(module, E: np.ndarray):
    """module.weight replaced by E (cast to its dtype) inside the block, the original tensor put back on exit."""
    orig = module.weight.data
    module.weight.data = torch.as_tensor(E, device=orig.device).to(orig.dtype)
    yield orig
    module.weight.data = orig


class Job(NJ.Run):
    def window_loss(self) -> np.ndarray:
        hw, bos, out = self.items["hw"], self.items["bos"], []
        for b0 in range(0, len(self.items["windows"]), self.batch):
            ws = self.items["windows"][b0:b0 + self.batch]
            x = torch.as_tensor(np.stack([np.r_[bos, self.tokens[t:t + hw]] for t in ws]).astype(np.int64), device=self.device)
            lg = self.model(input_ids=x, use_cache=False).logits.float()
            out.append(F.cross_entropy(lg[:, :-1].transpose(1, 2), x[:, 1:], reduction="none").mean(1).cpu().numpy())
        return np.concatenate(out)

    def run(self):
        clean, out, recs = self.window_loss(), {}, {}
        x = torch.as_tensor(np.r_[self.items["bos"], self.tokens[:self.items["hw"]]][None].astype(np.int64), device=self.device)
        base = self.model(input_ids=x, use_cache=False).logits
        by = {e["edit"]: e for e in self.items["edits"]}
        restored, differ, id_eq, id_fail = True, True, True, True
        for e in self.items["edits"]:
            mod = self.ad.layers[e["layer"]].mlp.down_proj
            W = mod.weight.data.float().cpu().numpy()
            u = direction(W, e)
            p = u.astype(np.float64) @ W.astype(np.float64)
            # matched: the strength whose Frobenius norm s ||u^T W[:, cols]|| equals the target edit's (write.strength_matching when both share W)
            s = recs[e["match"]]["frobenius"] / float(np.sqrt((p[e["cols"]] ** 2).sum())) if "match" in e else float(e["strength"])
            E = Wr.edit_columns(W, e["cols"], u, s)
            with edited(mod, Wr.edit_columns(W, e["cols"], u, 0.0)):
                id_eq &= torch.equal(self.model(input_ids=x, use_cache=False).logits, base)
            with edited(mod, Wr.edit_columns(W, e["cols"], u, 1.0)):
                id_fail &= not torch.equal(self.model(input_ids=x, use_cache=False).logits, base)
            with edited(mod, E) as orig:
                loss = self.window_loss()
                differ &= not torch.equal(mod.weight.data, orig)
            restored &= torch.equal(mod.weight.data, torch.as_tensor(W).to(mod.weight.dtype))
            frob = float(abs(s) * np.sqrt((p[e["cols"]] ** 2).sum()))
            d = R.boot_matrix(loss - clean, np.mean)
            recs[e["edit"]] = {"layer": e["layer"], "n_cols": len(e["cols"]), "strength": s, "frobenius": frob, "delta_loss": d, "u_p": p}
            out[f"{e['edit']}__loss"], out[f"{e['edit']}__u"] = loss, u
            self.m["cases"][e["edit"]] = {k: v for k, v in recs[e["edit"]].items() if k != "u_p"}
            self.write()
        gap, gap_f = 0.0, 0.0
        for e in self.items["edits"]:
            if "match" in e:
                t, r = recs[e["match"]], recs[e["edit"]]
                gap = max(gap, abs(r["frobenius"] - t["frobenius"]) / t["frobenius"])
                f_raw = abs(t["strength"]) * float(np.sqrt((r["u_p"][by[e["edit"]]["cols"]] ** 2).sum()))
                gap_f = max(gap_f, abs(f_raw - t["frobenius"]) / t["frobenius"])
        art = ["edit.npz"]
        self.gate("identity", R.gate(float(not id_eq), id_eq, float(id_fail), id_fail, "a strength-1 edit", 0.0, None, art))
        self.gate("restore", R.gate(float(not restored), restored, float(differ), differ, "the edited weight against the original", 0.0, None, art))
        self.gate("frobenius", R.gate(gap, gap <= FROB_TOL, gap_f, gap_f > FROB_TOL, "the matched edit at its target's strength", 0.0, None, art))
        self.save("edit", clean=clean, **out)
        self.done()


def dry_checks(model, tokens, offsets, items) -> list:
    """write.edit_columns and strength_matching against infra/pg/alwayson_edit's, exactly, on the tiny model's down_proj."""
    from studies.reproductions.alwayson.code.pg import alwayson_edit as AE  # moved there by the consolidation of 2026-11-04 (branch consolidation/docs); merge that branch first
    W = list(model.model.layers)[1].mlp.down_proj.weight.data.float().numpy()
    u = direction(W, {"direction": {"kind": "random", "seed": 3}})
    assert np.array_equal(Wr.edit_columns(W, [1, 5, 9], u, 0.7), AE.edit_columns(W, [1, 5, 9], u, 0.7))
    assert Wr.strength_matching(W, [2, 3], [1, 5, 9], u, 0.7) == AE.strength_matching(W, [2, 3], [1, 5, 9], u, 0.7)
    return ["EQUAL edit_columns and strength_matching = infra/pg/alwayson_edit's, exactly"]


def dry_items(model, tokens, offsets, d: dict) -> dict:
    """Eight whole windows; a random-direction edit of 12 columns at layer 1, the shared-direction edit of the same columns, and a 4-column edit at
    layer 2 matched to the first edit's norm."""
    rng = np.random.default_rng(3)
    cols = sorted(rng.choice(128, 12, replace=False).tolist())
    return {"windows": [i * d["hw"] for i in range(8)], "edits": [
        {"edit": "e_random", "layer": 1, "cols": cols, "direction": {"kind": "random", "seed": 0}, "strength": 1.0},
        {"edit": "e_shared", "layer": 1, "cols": cols, "direction": {"kind": "mean_of_columns"}, "strength": 1.0},
        {"edit": "e_matched", "layer": 2, "cols": cols[:4], "direction": {"kind": "random", "seed": 0}, "match": "e_random"}]}


if __name__ == "__main__":
    NJ.main(sys.modules[__name__])
