"""The ablation instrument: for each case (a pole, its cluster J and another cluster O of the same pole) the fall of the pole on J when a set of
upstream neurons is written: own (J's input set), cross (O's input set, read on J), random (matched to cross in |ā| neuron by neuron) and the
identity arm, under every written value (mean, zero, median, resample 0 to 3) at every window position or at the marked token only. Written
against infra/neurons; the job shape is neurons_job. Replaces the arm loops of reuse_ablate, reuse_ablate2, reuse_birth, reuse_generic,
reuse_sharedinputs, instruments_e1, instruments_e4 and instruments_e5 (marked-token writes).

Items: {"background": [global positions], "m": 50, "frac": 0.01, "values": [...], "at": ["window", "marked"],
        "cases": [{"case", "pole", "J": [global positions], "O": [global positions]}]}.
Input sets: top-m of the summed attribution over the side's contexts with the outlier set excluded (sets.top_m_excluded); random: matched to
cross, outside both sets and the outlier set (sets.random_matched). I = fall(cross) / fall(own); cancelled when own < 2 |random| or fewer than 3
contexts have a_P > 0 (write.cancelled), each under the mean value written at every window position.
Bank: background.npz (abar, outliers, the resample indices); falls/<case>.npz (tok, arm [n], a_before [N], a_after [n, N], set_layer,
set_neuron [n, m], -1 padding); metrics.json with each case's falls, I and cancel flag.
Gates:
  mean            X.mean(0) = ā; failing input the half mean.
  identity        max |fall| of the identity arm (every layer below L hooked, nothing written) <= 1e-3 over contexts with a_P > 0; failing input the
                  own arm read as identity (must exceed 1e-3).
  exclusion       members of the outlier set in any written set, 0; failing input the unexcluded own and cross sets' members (must be > 0).
  positive_control  median own fall (mean, window) >= the floor (0.25; items may set "floor" for a tiny model); failing input random in place of
                  own (must read under the floor). Identity: the median identity fall.
  cancel_share    share of cases cancelled < 0.40; failing input own and random swapped (must reach 0.40).
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from infra.aws_pg.jobs import neurons_job as NJ
from infra.neurons import capture as C
from infra.neurons import rules as R
from infra.neurons import sets as S
from infra.neurons import write as Wr

INSTRUMENT = "ablation"
ID_TOL, FLOOR, CANCEL_MAX = 1e-3, 0.25, 0.40
ARMS = ("identity", "own", "cross", "random")


def check_items(items: dict):
    assert items["instrument"] == INSTRUMENT and items["background"] and items["cases"], "ablation items need background positions and cases"
    assert set(items["values"]) <= set(Wr.VALUES) and "mean" in items["values"] and set(items["at"]) <= {"window", "marked"} and "window" in items["at"]
    for c in items["cases"]:
        C.parse_pole(c["pole"])
        assert len(c["J"]) >= 3 and len(c["O"]) >= 3 and not set(c["J"]) & set(c["O"]), (c["case"], "J and O need 3 contexts each and share none")


class Job(NJ.Run):
    def run(self):
        n_up = len(self.ad.layers) - 1
        X = C.background_a(self.ad, self.model, np.asarray(self.items["background"]), n_up, **self.kw)
        abar = X.mean(0)
        tab, idx = Wr.tables(X, abar)
        mc = Wr.mean_check(X, abar)
        self.gate("mean", R.gate(mc["max_abs"], mc["pass"], mc["failing_input_max_abs"], mc["failing_input_failed"], mc["failing_input"], 0.0, None,
                                 ["background.npz"]))
        mask = S.outliers(abar, n_up, self.items.get("frac", S.FRAC))["mask"]
        self.save("background", abar=abar.cpu().numpy(), outliers=mask.cpu().numpy(), resample_index=idx)
        m, W, rows, n_in, n_in_fail, id_max, id_fail = self.items.get("m", 50), self.ad.width, [], 0, 0, 0.0, 0.0
        for case in self.items["cases"]:
            unit = C.pole_unit(case["pole"])
            L = unit["layer"]
            score = {s: C.direct_A(self.ad, C.capture(self.ad, self.model, np.asarray(case[s]), L, {L}, **self.kw), unit).sum(0) for s in ("J", "O")}
            sets = {"identity": [], "own": S.top_m_excluded(score["J"], m, W, mask), "cross": S.top_m_excluded(score["O"], m, W, mask)}
            sets["random"] = S.random_matched(sets["cross"], sets["own"], abar[:L], mask)
            n_in += sum(int(S.in_set(sets[x], mask).sum()) for x in ("own", "cross", "random"))
            n_in_fail += sum(int(S.in_set(S.top_m(score[s], m, W), mask).sum()) for s in ("J", "O"))
            J = np.asarray(case["J"])
            before = Wr.readout(self.model, J, unit, None, **self.kw)["a"]
            names, after = [], []
            for arm in ARMS:
                for v in (["mean"] if arm == "identity" else self.items["values"]):
                    for at in (["window"] if arm == "identity" else self.items["at"]):
                        w = Wr.written(sets[arm], v, tab, L, self.device, identity=arm == "identity")
                        names.append(f"{arm}__{v}__{at}")
                        after.append(Wr.readout(self.model, J, unit, w, at=at, **self.kw)["a"])
            f = {n: Wr.fall(before, a) for n, a in zip(names, after)}
            p = before > 0
            id_max = max(id_max, float(np.abs((before[p] - after[0][p]) / before[p]).max()))
            own = after[names.index("own__mean__window")]
            id_fail = max(id_fail, float(np.abs((before[p] - own[p]) / before[p]).max()))
            fo, fc, fr = (f[f"{x}__mean__window"] for x in ("own", "cross", "random"))
            rec = {"pole": case["pole"], "neuron": case["pole"][:-1], "falls": {n: x["mean"] for n, x in f.items()}, "n_pos": fo["n_pos"],
                   "I": fc["mean"] / fo["mean"] if fo["mean"] != 0 else float("nan"), "cancelled": Wr.cancelled(fo["mean"], fr["mean"], fo["n_pos"]),
                   "cancelled_swapped": Wr.cancelled(fr["mean"], fo["mean"], fo["n_pos"])}
            pad = lambda S_: np.array([[l for l, _ in S_] + [-1] * (m - len(S_)), [k for _, k in S_] + [-1] * (m - len(S_))])
            lay = np.stack([pad(sets[n.split("__")[0]]) for n in names])
            rec["path"] = self.save(f"falls/{case['case']}", tok=J, arm=np.array(names), a_before=before, a_after=np.stack(after),
                                    set_layer=lay[:, 0], set_neuron=lay[:, 1])
            self.m["cases"][case["case"]] = rec
            rows.append(rec)
            self.write()
        art = ["falls/*.npz"]
        self.gate("identity", R.gate(id_max, id_max <= ID_TOL, id_fail, id_fail > ID_TOL, "the own arm read as identity", id_max, None, art))
        self.gate("exclusion", R.gate(n_in, n_in == 0, n_in_fail, n_in_fail > 0, "the unexcluded own and cross sets' members in the outlier set", 0, None, art))
        floor = self.items.get("floor", FLOOR)
        med = lambda k: float(np.median([r["falls"][k] for r in rows]))
        self.gate("positive_control", R.gate(med("own__mean__window"), med("own__mean__window") >= floor, med("random__mean__window"),
                                             med("random__mean__window") < floor, "random in place of own", med("identity__mean__window"), None, art, floor=floor))
        sh, shf = float(np.mean([r["cancelled"] for r in rows])), float(np.mean([r["cancelled_swapped"] for r in rows]))
        self.gate("cancel_share", R.gate(sh, sh < CANCEL_MAX, shf, shf >= CANCEL_MAX, "own and random swapped", None, "the rule compares own and random", art))
        self.done()


def dry_checks(model, tokens, offsets, items) -> list:
    """write.readout against reuse_ablate.readout (every window position) and instruments_e5.readout_marked (the marked token), exactly."""
    from infra.pg import instruments_e5 as E5
    from infra.pg import reuse_ablate as RA
    kw = dict(tokens=tokens, length=None, bos=items["bos"], batch=8, hw=items["hw"], device="cpu")
    case = items["cases"][0]
    unit = C.pole_unit(case["pole"])
    abar = C.background_a(C.Adapter(model), model, items["background"], unit["layer"], **kw).mean(0)
    S_ = [(0, 3), (1, 7)]
    for v in ("mean", "zero"):
        w = RA.ablation_dict(S_, v, abar, unit["layer"], "cpu")
        a, b = Wr.readout(model, case["J"], unit, w, **kw), RA.readout(model, case["J"], unit, w, **kw)
        c, d = Wr.readout(model, case["J"], unit, w, at="marked", **kw), E5.readout_marked(model, case["J"], unit, w, **kw)
        assert all(np.array_equal(a[k], b[k]) and np.array_equal(c[k], d[k]) for k in ("a", "g", "u")), v
    return ["EQUAL readout under a write = reuse_ablate.readout (window) and instruments_e5.readout_marked (marked token), exactly, mean and zero"]


def dry_items(model, tokens, offsets, d: dict) -> dict:
    """Eight + poles at layers 2 and 3, each read on its own 48 strongest positions of the fake corpus (as the real cases are a pole's strongest
    passages), split alternately into J and O; m 24; every value at both positions."""
    rng = np.random.default_rng(1)
    ad = C.Adapter(model)
    every = np.array([t for t in range(len(tokens)) if t % d["hw"]])
    cap = C.capture(ad, model, every, 3, {2, 3}, tokens=tokens, length=None, bos=d["bos"], batch=64, hw=d["hw"], device="cpu")
    cases = []
    for i, n in enumerate(rng.choice(ad.width, 8, replace=False)):
        L = 2 + i % 2
        a = (ad.layers[L].mlp.act_fn(cap["g"][L][:, n]) * cap["u"][L][:, n]).numpy()
        top = every[np.argsort(-a, kind="stable")[:48]].tolist()
        cases.append({"case": f"c{i}", "pole": C.pole_id(L, int(n), 0), "J": top[0::2], "O": top[1::2]})
    bg = [int(t) for t in rng.choice(every, 64, replace=False)]
    return {"background": bg, "m": 24, "frac": 0.01, "values": list(Wr.VALUES), "at": ["window", "marked"], "floor": FLOOR, "cases": cases}


if __name__ == "__main__":
    NJ.main(sys.modules[__name__])
