"""m13: the checks every study ran. Studies as rows, the plan's nine checks as columns; each cell a status (passed,
failed, not applicable, not recorded), a detail sentence with the recorded values, and the file and key (or line) that
records it.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.M13_DATA = {...}. Prints `path sha256` for every input. Every value in a cell's detail is read from its file here;
every cited text line is asserted to contain the phrase it is cited for. The failing-input column is recounted from the
gate records (every record that carries a registered failing input).

    python3 extract.py
"""
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # walkthrough/<picture> -> already-clean -> studies -> repository root
ID = HERE.name.upper()
MAX_BYTES = 300_000
SOURCES = []
NX = "studies/next"
OCT = "studies/already-clean/results"

CHECKS = [
    ("known_number", "reproduces a number published or measured before"),
    ("same_weights_twice", "runs the same weights twice and gets the same result"),
    ("failing_inputs", "gives every check a registered input that must fail, and the input fails"),
    ("positive_control", "runs a positive control, and the control orders correctly"),
    ("chance_level_control", "measures a random baseline and reads the result against it"),
    ("predictions_before_data", "writes its predictions before the data exist"),
    ("rules_on_one_half", "sets its thresholds on a development split and reads the held-out split once"),
    ("set_aside_before_held_out", "sets itself aside, when a check fails, before it reads its held-out split"),
    ("second_model_rederives", "has an independent reviewer model re-derive the numbers from the files"),
]


def src(rel):
    """Open a repository file by its path from the root, record its sha256 once, print `path sha256`."""
    p = ROOT / rel
    b = p.read_bytes()
    assert not b.startswith(b"version https://git-lfs"), f"{rel} is an LFS pointer: git lfs pull --include={rel}"
    if rel not in [s["path"] for s in SOURCES]:
        h = hashlib.sha256(b).hexdigest()
        SOURCES.append({"path": rel, "sha256": h})
        print(rel, h)
    return p


def compare(name, recomputed, banked, where, tol):
    """Store a recomputed summary beside the banked one; print both and whether they differ beyond rounding."""
    same = abs(recomputed - banked) <= tol
    print(f"CHECK {name}: recomputed {recomputed} banked {banked} ({where}) {'same' if same else 'DIFFERS'}")
    return {"recomputed": recomputed, "banked": banked, "banked_at": where, "same_within": tol, "same": same}


def write(data, notes):
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    data["meta"] = {"commit": commit, "sources": SOURCES,
                    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "notes": notes}
    out = HERE / "data.js"
    out.write_text(f"window.{ID}_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size <= MAX_BYTES, f"{out} is {size:,} bytes"
    print(f"wrote {out.relative_to(ROOT)}: {size:,} bytes")


def J(rel):
    return json.loads(src(rel).read_text())


def line(rel, n, phrase):
    """A cited text line: assert it holds the phrase, return the reference."""
    text = src(rel).read_text().splitlines()[n - 1]
    assert phrase in text, f"{rel}:{n} does not hold {phrase!r}"
    return f"line {n}"


def cell(status, detail, path, key):
    assert status in ("passed", "failed", "not applicable", "not recorded"), status
    return {"status": status, "detail": detail, "source": path, "key": key}


def ok(flag):
    return "passed" if flag else "failed"


def walk(x, path=""):
    if isinstance(x, dict):
        if "failing_input" in x or "failing_input_pass" in x:
            yield path, x
        for k, v in x.items():
            yield from walk(v, f"{path}.{k}" if path else k)


def fi_failed(x):
    if "failing_input" not in x:
        return x["failing_input_pass"] is False
    fi = x["failing_input"]
    return fi["failed"] is True if "failed" in fi else fi["pass"] is False


def failing_tally(files):
    """Recount: every record with a registered failing input, and whether that input failed as it must."""
    n = failed = 0
    did_not_fail = []
    for f in files:
        for p, x in walk(J(f)):
            n += 1
            if fi_failed(x):
                failed += 1
            else:
                did_not_fail.append(f"{f}: {p}")
    return n, failed, did_not_fail


def failing_cell(files, key="every record with a failing_input field"):
    n, failed, bad = failing_tally(files)
    detail = f"{failed} of {n} registered failing inputs fail as they must"
    if bad:
        detail += "; did not fail: " + ", ".join(bad)
    return cell(ok(failed == n), detail, " ; ".join(files), key), {"records": n, "failed_as_required": failed}


def main():
    rows, tallies = {}, {}

    # ------------------------------------------------------------------ October: groups and births in 300 units
    R = f"{OCT}/RESULTS.md"
    rows["october"] = {
        "name": "Clusters and their onsets in 600 poles of 300 neurons (19 to 25 October)",
        "known_number": cell("not recorded", "No reproduction of an earlier number is recorded in the repository's results "
                             "files; the study's bank is outside the repository.", R, "—"),
        "same_weights_twice": cell("not recorded", "No identity run of the same weights is recorded in the repository's "
                                   "results files.", R, "—"),
        "failing_inputs": cell("not recorded", "The record says the harvest checks' failing inputs were recorded, not asserted; "
                               "no per-check record is in the repository.", R,
                               line(R, 144, "G0/G1 failing inputs are recorded, not asserted")),
        "positive_control": cell("passed", "The judge's control (each side's main group against its own record) scored 0.910 "
                                 "on 39 tuning sides against the threshold 0.8.", R,
                                 line(R, 89, "Judge validity gate: a_ctrl 0.910 on 39 poles (pass; threshold 0.8)")),
        "chance_level_control": cell("not recorded", "The hard-decoy control could not be measured (0.798 against 0.80); no "
                                     "other chance-level control is recorded.", R,
                                     line(R, 149, "both below 0.80, so the row is reported as not measurable")),
        "predictions_before_data": cell("passed", "The predictions entry was committed 2026-10-20T03:13:32Z, before the "
                                        "unblinding of 2026-10-24.", R,
                                        line(R, 94, "PREDICTIONS entry: heading 2026-10-20T02:58:11+00:00, committed "
                                                    "2026-10-20T03:13:32+00:00")),
        "rules_on_one_half": cell("failed", "The rules were set on the tuning sides, but the unblinding of the 600 sides ran "
                                  "seven times: its 'filled once' guard was bypassed six times, once per re-run.", R,
                                  line(R, 144, "was bypassed six times, once per re-run")),
        "set_aside_before_held_out": cell("not applicable", "The checks recorded before the read of the 600 sides passed "
                                          "(the judge's control and the explanation threshold).", R,
                                          line(R, 94, "θ_s = 0.3153 (gate pass). Judge validity gate: pass.")),
        "second_model_rederives": cell("not recorded", "Opus checkers read each wave and the closeout (reads/CHECKERS.md, "
                                       "reads/closeout_W5.md); the results file says its numbers are copied from the "
                                       "bank, not re-derived.", R, line(R, 5, "nothing below is re-derived")),
    }
    src(f"{OCT}/reads/CHECKERS.md")
    src(f"{OCT}/reads/closeout_W5.md")

    # ------------------------------------------------------------------ M1: the three data mixes
    m1 = f"{NX}/m1"
    v = J(f"{m1}/verdict_m1.json")
    k0 = J(f"{m1}/banks/k0-01/k0/m1/metrics.json")["K0"]
    m1_files = [f"{m1}/verdict_m1.json", f"{m1}/banks/k0-01/k0/m1/metrics.json"] + \
               [f"{m1}/banks/{h}/m1/metrics.json" for h in ("start_a", "start_b", "G", "R", "M")]
    fc, tallies["m1"] = failing_cell(m1_files)
    fc["detail"] += ("; four checks have no banked failing-input value (version check, input hashes, corpus hash, the "
                     "room assertion; READ_m1.md section 4)")
    fc["source"] += f" ; {m1}/READ_m1.md"
    fc["key"] += "; READ_m1.md " + line(f"{m1}/READ_m1.md", 248, "Not made to fail, and no failing-input value banked in the run:")
    sp = v["specificity"]
    rows["m1"] = {
        "name": "The three data mixes (5 November)",
        "known_number": cell(ok(k0["values"]["pass"]), f"{k0['values']['n']:,} banked activations of the end-of-stage-1 "
                             f"snapshot recomputed; share bit-identical {k0['values']['exact_share']}.",
                             f"{m1}/banks/k0-01/k0/m1/metrics.json", "K0.values"),
        "same_weights_twice": cell(ok(v["G_ID"]["pass"]), f"The start harvested twice: band differences {v['G_ID']['I']}; "
                                   f"share of sides whose counts differ {v['G_ID']['changed_pole_share']}.",
                                   f"{m1}/verdict_m1.json", "G_ID"),
        "failing_inputs": fc,
        "positive_control": cell(ok(v["PC_NLL"]["pass"]), "On the corpus's code-and-math windows the 65 % mix predicts "
                                 f"better than the no-code mix: {v['PC_NLL']['cm_M_minus_G'][0]:.3f} nats per token "
                                 f"(range {v['PC_NLL']['cm_M_minus_G'][1]:.3f} to {v['PC_NLL']['cm_M_minus_G'][2]:.3f}).",
                                 f"{m1}/verdict_m1.json", "PC_NLL"),
        "chance_level_control": cell("passed", "1,000 random sets of 5 other directories: median |contrast| "
                                     f"{min(x['random_abs_median'] for x in sp):.4f} to {max(x['random_abs_median'] for x in sp):.4f} "
                                     f"by band, against the code-and-math contrast {min(x['cm'] for x in sp):.4f} to "
                                     f"{max(x['cm'] for x in sp):.4f}.", f"{m1}/verdict_m1.json", "specificity"),
        "predictions_before_data": cell("passed", "The author's prior, dated 2026-11-05, logged before any data.",
                                        f"{NX}/07_PICKS/M1_SPEC.md",
                                        line(f"{NX}/07_PICKS/M1_SPEC.md", 151, "logged before any data (2026-11-05)")),
        "rules_on_one_half": cell("not applicable", "No halves: the bar was frozen on stage-1 reference snapshots before the "
                                  "mixes were read.", f"{m1}/thresholds.json", "tau"),
        "set_aside_before_held_out": cell("not applicable", "Every check passed (verdict gates: all five harvests pass).",
                                          f"{m1}/verdict_m1.json", "gates"),
        "second_model_rederives": cell("passed", "An independent Opus read recomputed the verdict from the banked counts; "
                                       "the headline reproduces exactly.", f"{m1}/READ_m1.md",
                                       line(f"{m1}/READ_m1.md", 3, "The headline reproduces exactly")),
    }
    assert all(g["pass"] for g in v["gates"].values())

    # ------------------------------------------------------------------ the three birth studies that share a pipeline
    def birth_study(s, bank, name, spec, prior_line, prior_phrase, read, read_line, read_phrase):
        t = J(f"{NX}/{s}/thresholds.json")
        vv = J(f"{NX}/{s}/verdict_{s}.json")
        m = J(bank)
        g = m["gates"]
        fcell, tallies[s] = failing_cell([bank, f"{NX}/{s}/thresholds.json", f"{NX}/{s}/verdict_{s}.json"])
        return t, vv, g, fcell, {
            "name": name,
            "predictions_before_data": cell("passed", f"The author's prior, {prior_phrase}.", f"{NX}/07_PICKS/{spec}",
                                            line(f"{NX}/07_PICKS/{spec}", prior_line, prior_phrase)),
            "second_model_rederives": cell("passed", "An independent Opus read recomputed the verdict from the banked "
                                           f"records: '{read_phrase}'.", f"{NX}/{s}/{read}",
                                           line(f"{NX}/{s}/{read}", read_line, read_phrase)),
        }

    # M2: naming the carrier one snapshot early
    b = f"{NX}/m2/banks/score-06/m2/metrics.json"
    t, vv, g, fcell, row = birth_study("m2", b, "Naming the cluster’s own pole at the checkpoint before the onset (5 to 6 November)",
                                       "M2_SPEC.md", 210, "logged before any rank exists (2026-11-05)", "READ_m2.md", 10,
                                       "The headline reproduces exactly")
    fl = t["floor_freeze"]["B"]
    row |= {
        "known_number": cell(ok(g["K0"]["pass"] and g["t15_G_EX"]["pass"]), f"{g['K0']['n']} banked activations (share "
                             f"exact {g['K0']['exact_share']}) and {g['t15_G_EX']['n']:,} banked group activations at the "
                             f"last snapshot (share exact {g['t15_G_EX']['exact_share']}) reproduced.", b, "gates.K0, gates.t15_G_EX"),
        "same_weights_twice": cell(ok(g["I_ID"]["pass"]), f"{g['I_ID']['n']:,} values recomputed, share exact "
                                   f"{g['I_ID']['exact_share']}; carriers moved in or out of the top ten: "
                                   f"{g['I_ID']['carrier_top10_flips_B']}.", b, "gates.I_ID"),
        "failing_inputs": fcell,
        "positive_control": cell(ok(min(vv["test"]["end_B"][0], vv["test"]["end_A"][0]) >= t["ctrl_min"]),
                                 "The construction control: at the last snapshot, where the groups are defined, the carrier "
                                 f"is in the top ten in {vv['test']['end_B'][3]} of {vv['test']['n_usable']} held-out births "
                                 f"by the registered score and {vv['test']['end_A'][3]} by the second score (threshold share "
                                 f"{t['ctrl_min']}); its opposite side in {vv['test']['end_opposite_B'][3]}.",
                                 f"{NX}/m2/verdict_m2.json", "test.end_B, test.end_A, test.end_opposite_B; thresholds.json ctrl_min"),
        "chance_level_control": cell("passed", f"Random sides matched on layer and activation: share in the top ten "
                                     f"{fl['mean']:.5f} (97.5th percentile {fl['q975']:.4f}) on the rule-setting half, "
                                     f"against chance 10/22,016 = {t['chance']:.5f}; held out {vv['test']['floor_B']['mean']:.5f}.",
                                     f"{NX}/m2/thresholds.json", "chance, floor_freeze.B; verdict_m2.json test.floor_B"),
        "rules_on_one_half": cell("passed", f"Rules set on the '{t['discovery']['half']}' half (even unit index); the "
                                  f"'{vv['test']['half']}' half ({vv['test']['n_usable']} births) read by the verdict.",
                                  f"{NX}/m2/thresholds.json", "discovery.half; verdict_m2.json test.half"),
        "set_aside_before_held_out": cell("not applicable", "No check failed (verdict gates pass; the cross-device check is "
                                          "a report, not a check).", f"{NX}/m2/verdict_m2.json", "gates"),
    }
    assert vv["gates"]["pass"] and t["gates"]["pass"]
    rows["m2"] = row

    # N12: own rise, the distinct-token comparison, the gradient readings in 16-bit
    b = f"{NX}/n12/banks/score-02/n12/metrics.json"
    t, vv, g, fcell, row = birth_study("n12", b, "Does the cluster’s own pole rise at the onset, or do the other poles fall; attribution in bfloat16 (6 November)",
                                       "N12_SPEC.md", 200, "logged before any test-half N12 quantity exists (2026-11-06)",
                                       "READ_n12.md", 9, "The headline reproduces exactly")
    vf, vt = t["validation_freeze"], vv["test"]["validation"]
    fu = t["floor_u_freeze"]
    row |= {
        "known_number": cell(ok(g["K0"]["pass"]), f"The earlier study's score for {len(g['K0']['units'])} units reproduced "
                             f"(largest difference {g['K0']['max_abs_dB']}) and {g['K0']['activations']['n']} banked "
                             f"activations (share exact {g['K0']['activations']['exact_share']}).", b, "gates.K0"),
        "same_weights_twice": cell(ok(g["I_ID"]["pass"] and g["I_GRAD"]["pass"]), "The same weights run twice: activations "
                                   f"(share exact {g['I_ID']['exact_share']}) and gradients agree.", b, "gates.I_ID, gates.I_GRAD"),
        "failing_inputs": fcell,
        "positive_control": cell(ok(vf["pass"] and vt["pass"]), "The gradient prediction against the switch-off measurement "
                                 f"(the positive control of the gradient readings): rank agreement {vf['rho']:.3f} on the "
                                 f"rule-setting half and {vt['rho']:.3f} held out, threshold {vf['rho_min']}.",
                                 f"{NX}/n12/thresholds.json ; {NX}/07_PICKS/N12_SPEC.md",
                                 "validation_freeze; verdict_n12.json test.validation; N12_SPEC.md " +
                                 line(f"{NX}/07_PICKS/N12_SPEC.md", 388, "the positive control of the gradient readouts")),
        "chance_level_control": cell("passed", f"Random comparison selections: mean share {fu['mean']:.3f} (2.5th percentile "
                                     f"{fu['q025']:.3f}, {fu['reps']} replicates); the shuffled prediction's 97.5th "
                                     f"percentile {vf['shuffle_q975']:.3f}.", f"{NX}/n12/thresholds.json",
                                     "floor_u_freeze, validation_freeze.shuffle_q975"),
        "rules_on_one_half": cell("passed", f"Rules set on the '{t['discovery']['half']}' half; the '{vv['test']['half']}' "
                                  "half read by the verdict.", f"{NX}/n12/thresholds.json", "discovery.half; verdict_n12.json test.half"),
        "set_aside_before_held_out": cell("failed", "The gradient check failed on the rule-setting half "
                                          f"({vf['rho']:.3f} against {vf['rho_min']}); by the spec's rule the held-out half "
                                          f"was still read and the {sum(1 for x in vv['labels'].values() if x.get('label') == 'DOES NOT COUNT')} gradient "
                                          "readings were set aside afterwards (reported as exploratory).",
                                          f"{NX}/07_PICKS/N12_SPEC.md",
                                          line(f"{NX}/07_PICKS/N12_SPEC.md", 395, "otherwise every gradient label DOES NOT "
                                                                                  "COUNT") + "; verdict_n12.json labels.H_signal"),
    }
    rows["n12"] = row

    # N2V: the switch-off measurement in 32-bit
    b = f"{NX}/n2v/banks/score-02/n2v/metrics.json"
    t, vv, g, fcell, row = birth_study("n2v", b, "Zero-ablation measured in float32 (6 to 8 November)", "N2V_SPEC.md", 188,
                                       "logged before any N2V number exists (2026-11-06)", "READ_n2v.md", 11,
                                       "The verdict stands under my recount")
    wv = vv["test"]["window_jvp"]["validation"]
    row |= {
        "known_number": cell(ok(g["K_A"]["pass"]), f"{g['K_A']['n_exact']} of {g['K_A']['n_pairs']} banked switch-off "
                             "values of the 16-bit study reproduced exactly.", b, "gates.K_A"),
        "same_weights_twice": cell(ok(g["I_FWD"]["pass"] and g["I_JVP"]["pass"]), f"Forward rows {g['I_FWD']['n_bitwise']} "
                                   f"of {g['I_FWD']['n']} and derivative rows {g['I_JVP']['n_bitwise']} of {g['I_JVP']['n']} "
                                   "bitwise equal when run twice.", b, "gates.I_FWD, gates.I_JVP"),
        "failing_inputs": fcell,
        "positive_control": cell("passed", "The spec's positive control is the finite-difference check on a small model; its "
                                 "planted input passes in the offline check.", f"{NX}/07_PICKS/N2V_OFFLINE_GATE.md ; "
                                 f"{NX}/07_PICKS/N2V_SPEC.md", "N2V_OFFLINE_GATE.md " +
                                 line(f"{NX}/07_PICKS/N2V_OFFLINE_GATE.md", 31, "| G-FD |") + "; N2V_SPEC.md " +
                                 line(f"{NX}/07_PICKS/N2V_SPEC.md", 294, "G-FD on the tiny model")),
        "chance_level_control": cell("passed", f"1,000 shuffles of the prediction: 97.5th percentile of rank agreement "
                                     f"{wv['shuffle_q975']:.3f} held out, against the measured {wv['rho']:.3f}.",
                                     f"{NX}/n2v/verdict_n2v.json", "test.window_jvp.validation.shuffle_q975"),
        "rules_on_one_half": cell("passed", f"Rules set on the rule-setting half ({t['freeze']['n_pairs']} pairs); "
                                  f"{vv['test']['n_pairs']} held-out pairs read by the verdict.", f"{NX}/n2v/thresholds.json",
                                  "freeze; verdict_n2v.json test"),
        "set_aside_before_held_out": cell("not applicable", "No check failed.", f"{NX}/n2v/verdict_n2v.json", "gates"),
    }
    assert vv["gates"]["pass"]
    rows["n2v"] = row

    # N1: the carrier's weight change at the birth
    b = f"{NX}/n1/banks/weights-03/n1/metrics.json"
    t, vv, g, fcell, row = birth_study("n1", b, "The relative weight update of the cluster’s neuron at the onset (6 to 8 November)", "N1_SPEC.md", 149,
                                       "logged before any weight statistic exists (2026-11-06)", "READ_n1.md", 3,
                                       "The verdict's label stands under my recount")
    rep = J(b)["reports"]
    kb = [rep[k] for k in rep if k.endswith("_K_B")]
    pcs = [g[k] for k in g if k.endswith("_PC")]
    fi_, fo = t["floor_in_freeze"], t["floor_out_freeze"]
    row |= {
        "known_number": cell(ok(all(x["pass"] for x in kb)), f"Column norms recomputed against the bank at {len(kb)} "
                             f"snapshots; largest relative difference {max(x['max_rel'] for x in kb):.1e}.", b, "reports.tNN_K_B"),
        "same_weights_twice": cell(ok(g["I_ID"]["pass"]), "The same snapshot read twice: norms bitwise equal, its change "
                                   "against itself zero.", b, "gates.I_ID"),
        "failing_inputs": fcell,
        "positive_control": cell(ok(all(x["pass"] for x in pcs)), f"A planted change is recovered at all {len(pcs)} "
                                 f"snapshots (smallest recovery {min(min(x['in'], x['out']) for x in pcs):.2f}, threshold "
                                 f"{pcs[0]['min']}).", b, "gates.tNN_PC"),
        "chance_level_control": cell("passed", f"Matched random comparisons, 1,000 replicates: input rows mean "
                                     f"{fi_['mean']:.3f} (range {fi_['q025']:.3f} to {fi_['q975']:.3f}), output column mean "
                                     f"{fo['mean']:.3f} (range {fo['q025']:.3f} to {fo['q975']:.3f}).",
                                     f"{NX}/n1/thresholds.json", "floor_in_freeze, floor_out_freeze"),
        "rules_on_one_half": cell("passed", "Rules set on the rule-setting half; the held-out half read by the verdict.",
                                  f"{NX}/n1/thresholds.json", "discovery; verdict_n1.json test"),
        "set_aside_before_held_out": cell("not applicable", "No check failed.", f"{NX}/n1/verdict_n1.json", "gates"),
    }
    assert vv["gates"]["pass"]
    rows["n1"] = row

    # ------------------------------------------------------------------ N13: where the mathematics gain lives
    n13 = f"{NX}/n13"
    v = J(f"{n13}/verdict_n13.json")
    th = J(f"{n13}/thresholds.json")
    r1 = J(f"{n13}/s3/s3gate-01/r1_rank/metrics.json")["stages"]
    gates = v["gates"]
    n_v = len(gates)
    n_failed_v = sum(x["failed"] is True for x in gates.values())
    n_t, f_t, bad_t = failing_tally([f"{n13}/thresholds.json", f"{n13}/s3/s3gate-01/r1_rank/metrics.json"])
    tallies["n13"] = {"records": n_v + n_t, "failed_as_required": n_failed_v + f_t}
    gk = gates["G-KNOWN"]
    rows["n13"] = {
        "name": "Which part of the weight delta gives RL-Zero-Math its MATH-500 gain (5 to 8 November)",
        "known_number": cell(ok(gk["pass"]), f"The published score of the base model on competition mathematics reproduced: "
                             f"{gk['value']:.2f} (its failing input {gk['failing_value']:.2f}).", f"{n13}/verdict_n13.json",
                             "gates.G-KNOWN"),
        "same_weights_twice": cell(ok(gates["G-ID"]["pass"]), "The same weights under two seed sets: mathematics model "
                                   f"{th['L']:.1f} and {th['L_prime']:.1f}, base {th['B']:.1f} and {th['B_prime']:.1f}, "
                                   f"within the bound {th['G-ID']['bound']:.2f}.", f"{n13}/thresholds.json", "G-ID"),
        "failing_inputs": cell(ok(n_failed_v == n_v and f_t == n_t), f"{n_failed_v} of {n_v} verdict checks and {f_t} of "
                               f"{n_t} threshold and ranking checks fail on their registered failing inputs.",
                               f"{n13}/verdict_n13.json ; {n13}/thresholds.json ; {n13}/s3/s3gate-01/r1_rank/metrics.json",
                               "gates.*.failed; every record with a failing_input field"),
        "positive_control": cell(ok(r1["PC-RANK"]["pass"] and th["G-GATE"]["pass"]), "Planted changed units recovered by the "
                                 f"ranking: {r1['PC-RANK']['recovery']}; the gain {th['G']:.1f} points against the bound "
                                 f"{th['G-GATE']['bound']:.2f}.", f"{n13}/s3/s3gate-01/r1_rank/metrics.json",
                                 "stages.PC-RANK; thresholds.json G-GATE"),
        "chance_level_control": cell("passed", f"The seed baseline: one reading's noise sigma {th['sigma']:.3f} points from "
                                     f"two seed sets of the same weights; a difference must exceed {th['delta']:.2f}.",
                                     f"{n13}/thresholds.json", "sigma, delta"),
        "predictions_before_data": cell("passed", "The author's prior, logged before any data (2026-11-05).",
                                        f"{NX}/07_PICKS/N13_SPEC.md",
                                        line(f"{NX}/07_PICKS/N13_SPEC.md", 253, "logged before any data (2026-11-05)")),
        "rules_on_one_half": cell("not applicable", "No halves: the thresholds were frozen on the four reference "
                                  "conditions before the other conditions ran; the 1,319 word problems are a held-out "
                                  "benchmark.", f"{n13}/thresholds.json", "sigma, delta, G-GATE"),
        "set_aside_before_held_out": cell("not applicable", f"No check failed ({v['verdict']}; {n_v} of {n_v} checks pass).",
                                          f"{n13}/verdict_n13.json", "gates, verdict"),
        "second_model_rederives": cell("passed", "An independent Opus read re-graded every banked answer and recomputed the "
                                       "verdict; 465 of 465 verdict fields agree.", f"{n13}/READ_head.md",
                                       line(f"{n13}/READ_head.md", 14, "465 of 465 verdict fields agree")),
    }
    assert all(x["pass"] for x in gates.values())
    src(f"{n13}/READ_n13.md")

    # ------------------------------------------------------------------ N6: does an older side cost more to edit
    n6 = f"{NX}/n6"
    v6 = J(f"{n6}/verdict_n6.json")
    t6 = J(f"{n6}/thresholds.json")
    b6 = f"{n6}/banks/L2a-01/metrics.json"
    g6 = J(b6)["gates"]
    fc6, tallies["n6"] = failing_cell([b6])
    n_own_failed = sum(1 for x in g6.values() if x["pass"] is False)
    n_fi_passed = sum(1 for x in g6.values() if not fi_failed(x))
    tallies["n6"]["check_failed_gates"] = compare("N6 gates failed or whose failing input passed", n_own_failed + n_fi_passed,
                                                  len(v6["failed_gates"]), "verdict_n6.json failed_gates", 0)
    pc = g6["G_PC"]
    AC = f"{NX}/20_ABSTRACT_CHECK.md"
    rows["n6"] = {
        "name": "Does ablating the write of a pole with an earlier onset cost more loss (6 to 8 November)",
        "known_number": cell(ok(g6["K1"]["pass"] and g6["K2"]["pass"]), f"A banked activation fall reproduced "
                             f"({g6['K1']['value']:.6f} against {g6['K1']['target']:.6f}) and the banked corpus loss "
                             f"({g6['K2']['value']:.6f} against {g6['K2']['target']:.6f}).", b6, "gates.K1, gates.K2"),
        "same_weights_twice": cell(ok(g6["G_ID"]["pass"]), f"The same weights run twice: largest difference "
                                   f"{g6['G_ID']['value_max_abs']}.", b6, "gates.G_ID"),
        "failing_inputs": fc6,
        "positive_control": cell(ok(pc["pass"]), f"The own-direction edit cost more than the random edit in "
                                 f"{pc['n_col_gt_rand']} of {pc['n_layers']} control layers, {pc['need']} needed; with the "
                                 f"labels swapped it read {pc['failing_input_swapped']} of {pc['n_layers']}.", b6, "gates.G_PC"),
        "chance_level_control": cell("not recorded", "No chance-level control is registered; the random-direction edit is a "
                                     "matched comparison, recorded under the positive control.", b6, "—"),
        "predictions_before_data": cell("passed", "The author's prior, logged before any N6 number exists (2026-11-06).",
                                        f"{NX}/07_PICKS/N6_SPEC.md",
                                        line(f"{NX}/07_PICKS/N6_SPEC.md", 175, "logged before any N6 number exists (2026-11-06)")),
        "rules_on_one_half": cell("passed", f"Rules set on the rule-setting half only ({t6['discovery']['n']} pairs); the "
                                  f"held-out half was read {v6['n_test_read']} times.", f"{n6}/thresholds.json",
                                  "what, discovery.n; verdict_n6.json n_test_read"),
        "set_aside_before_held_out": cell(ok(not v6["G_POWER"]["pass"] and v6["n_test_read"] == 0), "The power check failed "
                                          f"on the rule-setting half (rounding floor {v6['G_POWER']['value']:.3g} against half "
                                          f"the median pair difference {v6['G_POWER']['bound']:.3g}, ratio "
                                          f"{v6['G_POWER']['ratio']:.3f}); the held-out half was never read "
                                          f"(n_test_read {v6['n_test_read']}).", f"{n6}/verdict_n6.json", "G_POWER, n_test_read"),
        "second_model_rederives": cell("passed", "The abstract check recounted the study's rows from edits.jsonl (31 of 31 "
                                       "equal); the rounding floor was compared with its file, not recounted.", AC,
                                       line(AC, 382, "**Answer: 31 items; 31 EQUAL, 0 ROUNDED, 0 MISMATCH.**") + "; " +
                                       line(AC, 399, "G-POWER τ_s 1.5096e-6")),
    }

    # ------------------------------------------------------------------ N22: does the model hold a variable (go/no-go)
    n22 = f"{NX}/n22"
    b22 = f"{n22}/banks/l1-01/n22/metrics.json"
    g22 = J(b22)["gates"]
    t22 = J(f"{n22}/thresholds.json")["points"]["15"]
    v22 = J(f"{n22}/verdict_n22_t15.json")["15"]
    fc22, tallies["n22"] = failing_cell([b22])
    n_items = len(J(f"{n22}/items.json")["items"])
    rows["n22"] = {
        "name": "Does the model hold a variable in arithmetic: the go/no-go (6 November)",
        "known_number": cell(ok(g22["t15_K0"]["pass"]), f"The banked corpus loss reproduced: {g22['t15_K0']['nll']:.6f} "
                             f"against {g22['t15_K0']['bank']:.6f}.", b22, "gates.t15_K0"),
        "same_weights_twice": cell(ok(g22["t15_I_GRAD"]["pass"] and g22["t15_I_GRAD"]["bit_identical"]), "The same input "
                                   "run twice gives a bit-identical Jacobian (largest relative difference "
                                   f"{g22['t15_I_GRAD']['max_rel']}).", b22, "gates.t15_I_GRAD"),
        "failing_inputs": fc22,
        "positive_control": cell("not applicable", "The interchange positive control never ran: the go/no-go stopped the "
                                 f"study ({t22['go_nogo']['n_answered']} of {n_items:,} items answered; no edit ran).",
                                 f"{n22}/thresholds.json", "points.15.go_nogo; verdict_n22_t15.json 15.slot"),
        "chance_level_control": cell("not applicable", "No reading ran past the go/no-go.", f"{n22}/thresholds.json",
                                     "points.15.go_nogo"),
        "predictions_before_data": cell("passed", "The author's prior, logged before any model output exists (2026-11-06).",
                                        f"{NX}/07_PICKS/N22_SPEC.md",
                                        line(f"{NX}/07_PICKS/N22_SPEC.md", 199, "logged before any model output exists")),
        "rules_on_one_half": cell("passed", "Thresholds frozen on the rule-setting halves only.", f"{n22}/thresholds.json",
                                  "what"),
        "set_aside_before_held_out": cell(ok(t22["go_nogo"]["go"] is False and v22["slot"].startswith("NOT_RUN")),
                                          f"The go/no-go failed ({t22['go_nogo']['n_answered']} of {n_items:,} items answered: "
                                          f"{v22['n_answered']['freeze']} and {v22['n_answered']['test']} by half); no edit "
                                          "ran.", f"{n22}/verdict_n22_t15.json",
                                          "15.slot; thresholds.json points.15.go_nogo"),
        "second_model_rederives": cell("not recorded", "The study has no independent read; the head's decision is a journal "
                                       "entry.", AC, line(AC, 175, "`n22/` has no READ file")),
    }
    assert "freeze halves only" in J(f"{n22}/thresholds.json")["what"]

    order = ["october", "m1", "m2", "n12", "n2v", "n1", "n13", "n6", "n22"]
    table = [{"study": s, **rows[s]} for s in order]
    for r in table:
        assert set(r) == {"study", "name"} | {c for c, _ in CHECKS}, r["study"]
    counts = {c: {st: sum(r[c]["status"] == st for r in table) for st in ("passed", "failed", "not applicable", "not recorded")}
              for c, _ in CHECKS}
    gaps = [f"{r['study']}: {c}" for r in table for c, _ in CHECKS if r[c]["status"] == "not recorded"]
    for s, x in tallies.items():
        print(f"NOTE failing inputs {s}: {x['failed_as_required']} of {x['records']}")
    print("NOTE not recorded:", gaps)
    print("PLAN N6's held-out half never read, n_test_read 0 -> studies[n6].set_aside_before_held_out")
    print("PLAN the studies x checks table (M13) -> studies[*]")

    data = {"checks": [{"id": c, "name": n} for c, n in CHECKS], "studies": table, "status_counts": counts,
            "failing_input_recount": tallies, "not_recorded": gaps}
    notes = {
        "study_ids": "Row id -> study folder: october = studies/already-clean (results/RESULTS.md); m1, m2, n12, n2v, n1, n13, "
                     "n6, n22 = studies/next/<id>/ (the study codes stay out of the prose; they are folder names).",
        "status": "passed / failed / not applicable / not recorded. 'not recorded' means no file in the repository records "
                  "the check for that study; such cells are listed in not_recorded.",
        "detail": "One sentence per cell; every number in it is read from the cell's source file by extract.py.",
        "source": "Repository path(s) of the file(s) that record the cell; several paths are joined by ' ; '.",
        "key": "The JSON key(s) or the text line in the source; 'line N' references are asserted to contain the cited phrase.",
        "failing_inputs": "Recounted: every gate record in the named files that carries a registered failing input "
                          "(failing_input or failing_input_pass); passed only if every one of them failed as it must.",
        "chance_level_control": "Read as: a chance-level reference (random sides, random directory sets, shuffled "
                                "predictions, a seed baseline) was measured and banked, and the result was read against "
                                "it; the cell gives the reference's value.",
        "set_aside_before_held_out": "'not applicable' where no check failed before the held-out half; the failed cell (the "
                                     "16-bit gradient study) followed its own spec, which read the held-out half and then "
                                     "set the gradient readings aside.",
        "second_model_rederives": "An independent reader's file (READ_<study>.md, an Opus agent in its own worktree) or the "
                                  "abstract check (20_ABSTRACT_CHECK.md).",
        "failing_input_recount": "Per study: records with a registered failing input, and how many failed as required.",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
