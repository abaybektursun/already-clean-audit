"""Manual session: known-number startup and fixed-prefix probability comparisons.

The existing six operations are unchanged. `inspect` compares clean and steered
next-token distributions on exactly the same tokenized prefix, including a pulse
at an earlier absolute position. Candidate strings are reported with their token
ids; only single-token candidates have a probability (no first-token proxy).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import pathlib
import subprocess
import sys
import time

import torch

from infra.pg import probe_server as S

S.FIELDS["inspect"] = (("model", "prompt"), {
    "pushes": [], "positions": "last", "layer_override": None, "candidates": [],
})


class Session(S.ProbeServer):
    @torch.no_grad()
    def op_inspect(self, r):
        model, tok, _ = self.get(r["model"])
        ids = self.ids(r["model"], r["prompt"])
        x = torch.tensor([ids], device=model.device)
        vecs, resolved = self.pushes(r["model"], r["pushes"], r["layer_override"])
        hook = S.Push(model, vecs, S.position_rule(r["positions"], len(ids), len(S.C.bos_prefix(tok))))

        def forward(h):
            with h if h is not None else contextlib.nullcontext():
                return model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1].float()

        clean = forward(None)
        # Measure the actual add at each hook, including bf16 rounding. Captures
        # bracket Push on the same forward; preceding interventions are shared.
        before, after, hs = {}, {}, []
        tower = S.R.text_tower(model)
        for l in hook.delta:
            hs.append(tower.layers[l].register_forward_hook(
                lambda m, a, o, l=l: before.__setitem__(l, S.R.hidden_of(o).float().clone())))
        try:
            with hook:
                for l in hook.delta:
                    hs.append(tower.layers[l].register_forward_hook(
                        lambda m, a, o, l=l: after.__setitem__(l, S.R.hidden_of(o).float().clone())))
                pushed = forward(None)
        finally:
            for h in hs:
                h.remove()
        realized = []
        mask = hook.positions(torch.arange(len(ids), device=model.device))
        for l, expected in hook.delta.items():
            delta = (after[l] - before[l])[:, mask]
            if float(expected.norm()) > 0 and delta.numel():
                cos = torch.nn.functional.cosine_similarity(delta, expected, dim=-1)
                ratio = delta.norm(dim=-1) / expected.norm()
                realized.append({"layer": l, "n_positions": int(mask.sum()),
                                 "cos_min": float(cos.min()), "ratio_min": float(ratio.min()),
                                 "ratio_max": float(ratio.max()), "wrong_sign_cos_max": float((-cos).max())})
        pc, pp = clean.softmax(-1), pushed.softmax(-1)
        candidates = []
        for text in r["candidates"]:
            ts = tok(text, add_special_tokens=False).input_ids
            row = {"text": text, "ids": ts}
            if len(ts) == 1:
                j = ts[0]
                row.update(p_clean=float(pc[j]), p_pushed=float(pp[j]),
                           logit_delta=float(pushed[j] - clean[j]),
                           rank_clean=int((clean > clean[j]).sum()) + 1,
                           rank_pushed=int((pushed > pushed[j]).sum()) + 1)
            candidates.append(row)
        return {"prompt_ids": ids, "prompt_tokens": len(ids), "pushes": resolved, "realized": realized,
                "positions_pushed": hook.first, "top5_clean": S.top5(clean, tok),
                "top5_pushed": S.top5(pushed, tok), "candidates": candidates,
                "max_abs_logit_delta": float((pushed - clean).abs().max()),
                "total_variation": float((pc - pp).abs().sum() / 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--provisional", action="store_true", help="Dated manual-session deviation; preserve a failed calibration and label all work provisional")
    a = ap.parse_args()
    resp = pathlib.Path(a.responses)
    resp.mkdir(parents=True, exist_ok=True)
    status = {"status": "starting", "started_utc": time.strftime("%FT%TZ", time.gmtime())}

    def progress(stage):
        status["stage"] = stage
        S.write_json(resp / "session_status.json", status)

    assert torch.cuda.is_available()
    torch.use_deterministic_algorithms(True)
    progress("tiny_model_tests")
    if importlib.util.find_spec("pytest") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "--quiet"], check=True)
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_probe_server.py",
                           "tests/test_probe_session.py"], capture_output=True, text=True)
    S.write_json(resp / "tests.json", {"exit_code": test.returncode, "stdout": test.stdout, "stderr": test.stderr})
    assert test.returncode == 0, test.stdout + test.stderr
    progress("load_Q4B")
    models, meta = S.load(["Q4B"], {})
    corpus = pathlib.Path("/workspace/probe_corpus")
    progress("known_corpus")
    subprocess.run(["aws", "s3", "sync", a.corpus, str(corpus), "--only-show-errors"], check=True)
    progress("known_number")
    from infra.aws_pg.jobs.neuronshow import known_reader
    known = known_reader(models["Q4B"][0], models["Q4B"][1], S.P.load(), str(corpus), resp)()
    S.write_json(resp / "known.json", known)
    calibrated = known["pass"] and not known["failing_input"]["pass"]
    status["known_calibration_pass"] = calibrated
    assert calibrated or a.provisional, known
    progress("load_G4_O7")
    rest, m = S.load(["G4", "O7"], {})
    models.update(rest)
    meta.update(m)
    import transformers
    meta.update(device_name=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__)
    srv = Session(models)
    rep = srv.startup(resp, meta)
    assert rep["pass"], rep
    # Real-model identity and deliberately nonidentity paths for the added readout.
    checks = {}
    for key in models:
        base = {"op": "inspect", "model": key, "prompt": S.IDENTITY_PROMPT,
                "candidates": [" the", "the", " sky"]}
        pole = srv.identity_poles[key]
        zero = srv.handle({**base, "pushes": [{"pole": pole, "dose": 0}]})
        one = srv.handle({**base, "pushes": [{"pole": pole, "dose": 1}]})
        checks[key] = {"zero_max_abs_delta": zero["max_abs_logit_delta"],
                       "one_max_abs_delta": one["max_abs_logit_delta"], "realized": one["realized"]}
        assert zero["max_abs_logit_delta"] == 0 and one["max_abs_logit_delta"] > 0
        assert one["realized"] and all(v["cos_min"] >= 0.95 and 0.9 <= v["ratio_min"] <= v["ratio_max"] <= 1.1
                                       and v["wrong_sign_cos_max"] < 0 for v in one["realized"])
    S.write_json(resp / "inspect_checks.json", checks)
    status["status"] = "ready" if calibrated else "ready_provisional"
    progress("serve")
    srv.serve(a.requests, a.responses)


if __name__ == "__main__":
    main()
