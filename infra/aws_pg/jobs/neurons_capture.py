"""The capture instrument: for each case (a pole and its contexts) the per-context attribution A to every upstream neuron, its coefficient c, the
top-m sets with and without the outlier exclusion, and the background ā with the outlier set. Written against infra/neurons; the job shape is
neurons_job. Replaces the capture steps of the reuse, unite and instruments jobs (unite_route.unit_bank's capture, reuse_birth.part_a's
capture, instruments_e1's and instruments_e4's captures).

Items: {"background": [global positions], "m": 50, "frac": 0.01, "cases": [{"case", "pole", "contexts": [global positions]}]}.
Bank: background.npz (abar [n_up, W], outliers mask); cases/<case>.npz (tok, a_P, A_sum [L W], D, D_unexcluded, D_in_outliers, c_D and a_D
[N, m], top_idx/top_A/top_abar); metrics.json with one record per case.
Gates (each with its failing input; identity values banked):
  mean          X.mean(0) = ā to 1e-5 (1 + |ā|); failing input the mean over the first half of the background positions.
  outliers      round(frac n) members, each at or above every non-member in |ā|; failing input the members of a permuted ā.
  known_activation  per case max |a_P(capture) - a_P(full window forward, down_proj input)| / max |a_P| <= 1e-2; failing input the reference
                rolled by one context. Identity: the capture against itself, 0.
  completeness  max over cases of the g error <= 1e-2 (bf16 forward, fp32 sums); failing input the columns from the next layer.
  coefficient   max over cases of |A - a c| / max |A| <= 1e-5; failing input c with the next layer's loads.
  exclusion     members of the outlier set in the excluded sets, 0; failing input the unexcluded sets' members in it (must be > 0).
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

INSTRUMENT = "capture"
KNOWN_TOL = COMPLETE_TOL = 1e-2


def check_items(items: dict):
    assert items["instrument"] == INSTRUMENT and items["background"] and items["cases"], "capture items need background positions and cases"
    assert len({c["case"] for c in items["cases"]}) == len(items["cases"]), "case ids repeat"
    for c in items["cases"]:
        C.parse_pole(c["pole"])
        assert len(c["contexts"]) >= 3 and all(int(t) % items["hw"] != 0 for t in c["contexts"]), (c["case"], "contexts at window position 0")


def own_a(job, toks, L: int) -> torch.Tensor:
    """The reference: the model's own down_proj input at layer L at each marked token, one plain forward per row (no batching, no stop)."""
    out = []
    for t in toks:
        x, pos = C.pack(job.tokens, [t], job.items.get("length"), job.items.get("bos"), job.items["hw"], job.device)
        keep = {}
        with C.mlp_tap(job.ad.layers, lambda l, a: keep.__setitem__(l, a[0, int(pos[0])].float()) if l == L else None):
            C.forward(job.model, x, False)
        out.append(keep[L])
    return torch.stack(out)


class Job(NJ.Run):
    def background(self):
        n_up = len(self.ad.layers) - 1
        X = C.background_a(self.ad, self.model, np.asarray(self.items["background"]), n_up, **self.kw)
        abar = X.mean(0)
        mc = Wr.mean_check(X, abar)
        self.gate("mean", R.gate(mc["max_abs"], mc["pass"], mc["failing_input_max_abs"], mc["failing_input_failed"], mc["failing_input"], 0.0, None,
                                 ["background.npz"]))
        o = S.outliers(abar, n_up, self.items.get("frac", S.FRAC))
        perm = abar.reshape(-1)[torch.randperm(abar.numel(), generator=torch.Generator().manual_seed(0))].reshape(abar.shape)
        top = lambda mask: bool(abar.abs()[mask].min() >= abar.abs()[~mask].max())
        self.gate("outliers", R.gate(o["count"], o["count"] == round(o["frac"] * o["n_entries"]) and top(o["mask"]), top(S.outliers(perm, n_up, o["frac"])["mask"]),
                                     not top(S.outliers(perm, n_up, o["frac"])["mask"]), "the members of a permuted ā", None, "a count", ["background.npz"]))
        self.save("background", abar=abar.cpu().numpy(), outliers=o["mask"].cpu().numpy())
        return abar, o["mask"]

    def run(self):
        abar, mask = self.background()
        m, rec = self.items.get("m", 50), {"known": [], "known_fail": [], "complete": [], "complete_fail": [], "c": [], "c_fail": [], "in": 0, "in_fail": 0}
        for case in self.items["cases"]:
            unit = C.pole_unit(case["pole"])
            L, n = unit["layer"], unit["terms"][0]["n"]
            toks = np.asarray(case["contexts"])
            cap = C.capture(self.ad, self.model, toks, L, {L}, **self.kw)
            a_P = unit["terms"][0]["sign"] * self.ad.layers[L].mlp.act_fn(cap["g"][L][:, n]) * cap["u"][L][:, n]
            ref = unit["terms"][0]["sign"] * own_a(self, toks, L)[:, n]
            scale = float(ref.abs().max())
            rec["known"].append(float((a_P - ref).abs().max()) / scale)
            rec["known_fail"].append(float((a_P - ref.roll(1)).abs().max()) / scale)
            ok, bad = C.completeness(self.ad, cap, L, n), C.completeness(self.ad, cap, L, n, shift=1)
            rec["complete"].append(float(ok["err_g"].max()))
            rec["complete_fail"].append(float(bad["err_g"].max()))
            A = C.direct_A(self.ad, cap, unit)
            c, cr = C.coefficient(self.ad, cap, unit, A)
            rec["c"].append(cr["value"])
            rec["c_fail"].append(cr["failing_input"])
            score = A.sum(0)
            D, D0 = S.top_m_excluded(score, m, self.ad.width, mask), S.top_m(score, m, self.ad.width)
            rec["in"] += int(S.in_set(D, mask).sum())
            rec["in_fail"] += int(S.in_set(D0, mask).sum())
            idx = torch.tensor([l * self.ad.width + k for l, k in D])
            a_all = torch.cat([cap["a"][l] for l in range(L)], 1)
            tb = S.top_bank(score, abar, self.ad.width)
            path = self.save(f"cases/{case['case']}", tok=toks, a_P=a_P.cpu().numpy(), A_sum=score.cpu().numpy(), D=np.array(D), D_unexcluded=np.array(D0),
                             D_in_outliers=S.in_set(D0, mask), c_D=c[:, idx].cpu().numpy(), a_D=a_all[:, idx].cpu().numpy(),
                             top_idx=tb["idx"], top_A=tb["A"], top_abar=tb["abar"])
            self.m["cases"][case["case"]] = {"pole": case["pole"], "n": len(toks), "path": path, "known": rec["known"][-1], "complete": rec["complete"][-1],
                                             "n_unexcluded_in_outliers": int(S.in_set(D0, mask).sum())}
            self.write()
        art = ["cases/*.npz"]
        self.gate("known_activation", R.gate(max(rec["known"]), max(rec["known"]) <= KNOWN_TOL, min(rec["known_fail"]), min(rec["known_fail"]) > KNOWN_TOL,
                                             "the reference rolled by one context", 0.0, None, art))
        self.gate("completeness", R.gate(max(rec["complete"]), max(rec["complete"]) <= COMPLETE_TOL, min(rec["complete_fail"]),
                                         min(rec["complete_fail"]) > COMPLETE_TOL, "the columns from the next layer", None, "no identity input for a sum", art))
        self.gate("coefficient", R.gate(max(rec["c"]), max(rec["c"]) <= C.C_TOL, min(rec["c_fail"]), min(rec["c_fail"]) > C.C_TOL,
                                        "c with the next layer's loads", None, "c is defined by A itself", art))
        self.gate("exclusion", R.gate(rec["in"], rec["in"] == 0, rec["in_fail"], rec["in_fail"] > 0, "the unexcluded sets' members in the outlier set",
                                      0, None, art))
        self.done()


def dry_checks(model, tokens, offsets, items) -> list:
    """The library against the modules it replaces, on the tiny model: instruments_adapter's capture, direct_A, c and completeness, exactly."""
    from infra.pg import instruments_adapter as AD
    ad, old = C.Adapter(model), AD.Adapter(model)
    rows = AD.Rows(model, old)
    kw = dict(tokens=tokens, length=None, bos=items["bos"], batch=8, hw=items["hw"], device="cpu")
    for case in items["cases"]:
        unit = C.pole_unit(case["pole"])
        L, n = unit["layer"], unit["terms"][0]["n"]
        a, b = C.capture(ad, model, case["contexts"], L, {L}, **kw), AD.capture(old, model, case["contexts"], L, {L}, **kw)
        assert all(torch.equal(a[k][l], b[k][l]) for k in ("a", "g", "u", "s", "r", "attn", "mlp") for l in b[k]) and torch.equal(a["emb"], b["emb"]), case
        A, A_old = C.direct_A(ad, a, unit), AD.direct_A(old, rows, b, unit, ad.width)
        assert torch.equal(A, A_old) and torch.equal(C.coefficient(ad, a, unit, A)[0], AD.coefficients_c(old, rows, b, unit, ad.width, A_old)[0]), case
        for sh in (0, 1):
            x, y = C.completeness(ad, a, L, n, sh), AD.completeness(old, rows, b, L, n, sh)
            assert np.array_equal(x["err_g"], y["err_g"]) and np.array_equal(x["err_u"], y["err_u"]), (case, sh)
    X, X_old = C.background_a(ad, model, items["background"], 3, **kw), AD.capture_background(old, model, items["background"], 3, **kw)
    assert torch.equal(X, X_old)
    return [f"EQUAL capture, direct_A, c, completeness (and its failing input) and the background capture = instruments_adapter's, exactly, "
            f"on {len(items['cases'])} cases"]


def dry_items(model, tokens, offsets, d: dict) -> dict:
    """Background 64 positions, three cases at layers 2 and 3, 12 contexts each, m 40 of the tiny model's 3 x 128 upstream neurons."""
    rng = np.random.default_rng(0)
    pos = lambda k: [int(t) for t in rng.choice(len(tokens), 4 * k, replace=False) if t % d["hw"]][:k]
    return {"background": pos(64), "m": 40, "frac": 0.01, "cases": [{"case": f"c{i}", "pole": p, "contexts": pos(12)}
                                                                      for i, p in enumerate(("L03N5+", "L03N77-", "L02N9+"))]}


if __name__ == "__main__":
    NJ.main(sys.modules[__name__])
