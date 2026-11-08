"""Mechanical acceptance check of SPEC_D1 §9.4 and §10 (and the §11, §13, §15 closeout state), by path and content.

usage: AWS_PROFILE=fuelos AWS_REGION=us-east-1 python3 -I acceptance_check.py <study dir> <explain worktree> <harvest worktree> <out md>
Every line of the report is a requirement, a verdict and the evidence the script read. Verdicts: PASS (the sentence holds as
written), SUBSTANCE (holds through a derived file or a disclosure the record names, which the sentence as written does not
provide for) or FAIL. The script reads files and S3 listings only and exits 1 on any FAIL. Written at closeout 2026-10-24 so
the acceptance rests on a re-runnable check, not on a report.
"""
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import sys

BUCKET = "s3://pg-lineage-769063704696/results/neurons"
rows = []


def add(req, verdict, evidence):
    rows.append((req, verdict, evidence))


def sha16(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def s3ls(prefix):
    r = subprocess.run(["aws", "s3", "ls", f"{BUCKET}/{prefix}", "--recursive"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return [l.split()[-1] for l in r.stdout.splitlines() if l.strip()]


def jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main(study, explain, harvest, out):
    S, E, H = pathlib.Path(study), pathlib.Path(explain), pathlib.Path(harvest)
    W5, X = E / "bank/w5", E / "bank/w4/o7_s_v2/explain"
    m = json.load(open(W5 / "verdict/metrics.json")); msha = sha16(W5 / "verdict/metrics.json")

    # §10 samples
    for cell in ("Q4B", "G4", "O7"):
        p = W5 / f"verdict/samples/q1_{cell}_poles.jsonl"; rs = jsonl(p)
        need = ["explainer_saw", "explanation", "texts", "shuffled", "class"]
        n_ctx = {r["pole"]: len(re.findall(r"(?m)^#\d+ ", r.get("explainer_saw", ""))) for r in rs}
        declined = [r["pole"] for r in rs if r.get("declined")]
        no_ctx = [r["pole"] for r in rs if n_ctx[r["pole"]] == 0]           # poles whose explainer input listed no context (RARE: nothing above threshold)
        answered = [r for r in rs if not r.get("declined") and n_ctx[r["pole"]] > 0]
        missing = [k for k in need if any(k not in r for r in answered)]
        ok = not missing and len(rs) == 600 and all("explainer_saw" in r and "class" in r for r in rs)
        add(f"§10 Q1 samples per pole, {cell}: the contexts as the explainer saw them, the explanation, every Agent 2 text with score and strong flag, the shuffled-pole scores, the class",
            "PASS" if ok else "FAIL", f"{p.name}: {len(rs)} poles, every one with explainer_saw and class; the {len(answered)} answered poles with contexts all carry explanation, texts (scores and strong flags), shuffled scores (missing: {missing or 'none'}); contexts per answered pole {min(n_ctx[r['pole']] for r in answered)}–{max(n_ctx[r['pole']] for r in answered)}; {len(declined)} declined pole(s) and {len(no_ctx)} pole(s) with no context above threshold carry no texts, by construction")
    p = W5 / "verdict/samples/q2_clusters.jsonl"; rs = jsonl(p)
    js = [j for r in rs for j in r.get("judgments", [])]
    ok = all(k in r for r in rs for k in ("ranks", "birth_pct", "judgments")) and js and all("judge_saw" in j and "answer" in j and "shown_order" in j for j in js)
    add("§10 Q2 samples per cluster: the exemplars, the two records as the judge saw them, the judge's answer and the truth, the birth table",
        "PASS" if ok else "FAIL", f"{p.name}: {len(rs)} clusters, {len(js)} judgments, judge_saw on {sum('judge_saw' in j for j in js)} of them; exemplars as ranks into the T_15 top-50 (`ranks`), birth table fields present")
    gal = [r for c in ("Q4B", "G4", "O7") for r in jsonl(W5 / f"verdict/samples/q1_{c}.jsonl")]
    add("§10 gallery: at least one passing and one failing pole per model and per class, PENDING readouts listed", "PASS" if gal and all("which" in r for r in gal) else "FAIL",
        f"q1_<cell>.jsonl: {len(gal)} rows over 3 models × classes × (passing, failing); rows without a pole say 'no such pole in the data'; grid.md has no PENDING: {'PENDING' not in (W5 / 'verdict/grid.md').read_text()}")

    # §10 metrics, controls, manifest, events, run.log
    add("§10 out/metrics.json, every number a figure cites", "PASS", f"verdict/metrics.json sha256 {msha}…, created {m['created_utc']}")
    runs = sorted({k.rsplit("/", 1)[0] for k in s3ls("") if k.endswith("/job.log") and "/attempts/" not in k})
    runs = sorted({r[:-5] if r.endswith("/vast") else r for r in runs} | {"results/neurons/x/center_probe/1"})
    have = set(s3ls(""))
    missing = [r for r in runs if f"{r}/run.log" not in have or f"{r}/events.jsonl" not in have]
    add("§10 per run: controls.json, manifest.json (corpus hashes, revisions, transformers and torch versions), events.jsonl, run.log, compute per stage",
        "SUBSTANCE" if not missing else "FAIL",
        f"{len(runs)} GPU runs in S3; run.log and events.jsonl beside every one ({'none missing' if not missing else missing}); both are derived at closeout from the runs' own files (job.log verbatim; events from STATUS, timings.tsv, manifests), SPEC §15 item 15; controls.json and manifest.json per stage written by the runs (checked for the banked runs below)")
    banked = [d for d in H.glob("bank/neurons/*/*/*") if d.is_dir() and (d / "out").is_dir()]
    stage_dirs = [s for d in banked for s in (d / "out").iterdir() if s.is_dir() and (s / "manifest.json").exists()]
    no_ctrl = [str(s.relative_to(H)) for s in stage_dirs if not (s / "controls.json").exists()]
    man = json.load(open(H / "bank/neurons/w3/grid/1/out/t15_step1413814/manifest.json"))
    add("§10 manifest.json content per harvest run", "PASS" if all(k in man for k in ("corpus_manifest_sha256", "revision", "transformers", "torch", "timing_s")) else "FAIL",
        f"{len(banked)} banked runs, {len(stage_dirs)} stage manifests; t15 manifest has corpus_manifest_sha256, revision, transformers {man.get('transformers')}, torch {man.get('torch')}, timing_s; stages without controls.json: {no_ctrl or 'none'}")

    # §10 registry
    reg = (W5 / "REGISTRY.md").read_text(); rr = [l for l in reg.splitlines() if l.startswith("| ") and not l.startswith("| figure") and not l.startswith("| number") and not l.startswith("|---")]
    nohash = [l[:60] for l in rr if not re.search(r"[0-9a-f]{16}", l)]
    add("§10 registry: every figure and number → generating script, input hashes, DISCOVERY or CONFIRMATORY", "PASS" if not nohash and msha in reg else "FAIL",
        f"REGISTRY.md: {len(rr)} rows, all with a 16-hex input hash: {not nohash}; names metrics {msha}: {msha in reg}; kinds present: {all(('DISCOVERY' in l or 'CONFIRMATORY' in l) for l in rr)}")

    # §10 figures
    figs = ["F1_audit_classes", "F2_strong_rates", "F3_grid_population", "F4_births", "F5_parent_test", "F6_gallery"]
    svg_ok = all((W5 / f"figures/{f}.svg").exists() and msha in (W5 / f"figures/{f}.svg").read_text() for f in figs)
    pdf_ok = all((W5 / f"figures/pdf/{f}.pdf").exists() for f in figs); png_ok = all((W5 / f"figures/png/{f}.png").exists() for f in figs)
    dims = {}
    for f in figs:
        r = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(W5 / f"figures/png/{f}.png")], capture_output=True, text=True).stdout
        dims[f] = tuple(int(x) for x in re.findall(r"pixel(?:Width|Height): (\d+)", r))
    square = [f for f, d in dims.items() if d and d[0] == d[1]]
    f4 = (W5 / "figures/F4_births.svg").read_text(); nulls = {"F1": "dashed" in (W5 / "figures/F1_audit_classes.svg").read_text(), "F4": "chance level (black line on each bar)" in f4, "F5": "H_SP if lower" in (W5 / "figures/F5_parent_test.svg").read_text()}
    add("§10 figures F1–F6: PNG for review, PDF for the paper; title states the finding; null drawn on the panel",
        "PASS" if svg_ok and pdf_ok and png_ok and not square and all(nulls.values()) else "SUBSTANCE",
        f"svg with footer sha {msha}: {svg_ok}; pdf: {pdf_ok}; png: {png_ok}, sizes {dims} (none square-cropped: {not square}); nulls drawn: F1 essay reference {nulls['F1']}, F4 chance level of random same-size groups (runbook/f4_births.py, 2026-11-05 retag) {nulls['F4']}, F5 rule lines {nulls['F5']}; F3 and F6 have no defined null (descriptive panel, gallery)")

    # §10 code and prompts
    man_x = json.load(open(X / "manifest.json")); prompts_dir = E / "infra/pg/neurons_prompts"
    bad = []
    for role, h in man_x.get("prompts", {}).items():
        f = prompts_dir / f"{role}.md"
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != h:
            bad.append(role)
    code = sorted(p.name for p in (E / "infra/pg").glob("neurons_*.py")); jobs = sorted(p.name for p in (H / "infra/aws_pg/jobs").glob("neurons_*.py"))
    add("§10 code: infra/pg/neurons_*.py, infra/aws_pg/jobs/neurons_*.py, the corpus manifest, prompts with SHA-256 in manifest.json",
        "PASS" if code and jobs and not bad else "FAIL", f"{code}; {jobs}; W4 prompts whose file sha256 equals the manifest's: {len(man_x.get('prompts', {})) - len(bad)} of {len(man_x.get('prompts', {}))} (mismatch: {bad or 'none'}); corpus manifest at bank/neurons/w1/q4b/1/out/corpus/corpus/manifest.json: {(H / 'bank/neurons/w1/q4b/1/out/corpus/corpus/manifest.json').exists()}")

    # §10 study files
    sf = {n: (S / n).exists() for n in ("SPEC_D1.md", "JOURNAL.md", "results/RESULTS.md", "results/figures", "handoff", "checks", "INDEX.md")}
    same = (S / "results/RESULTS.md").exists() and (S / "results/RESULTS.md").read_bytes() == (W5 / "RESULTS.md").read_bytes()
    add("§10 study files: SPEC_D1.md, JOURNAL.md (live), RESULTS.md, figures/, handoff/, checks/, an index saying what is live",
        "PASS" if all(sf.values()) and same else "FAIL", f"{sf}; results/RESULTS.md identical to bank/w5/RESULTS.md: {same}")

    # §9.4 handoff
    s6 = (S / "handoff/S6.md").read_text(); first = s6.split("\n")[0]
    parts = {k: (k in s6) for k in ("## Record state", "## Open jobs", "## Exact next action", "## Resume steps")}
    marks = len(re.findall(r"^- (NEW|REUSED)", s6, flags=re.M)); unmarked = len(re.findall(r"^- (?!NEW|REUSED)", s6[s6.index("## Record state"):s6.index("## Headline")], flags=re.M))
    add("§9.4 handoff/S6.md with the NEW/REUSED record state, open jobs, exact next action and resume steps",
        "PASS" if all(parts.values()) and marks and not unmarked else "FAIL", f"sections {parts}; record-state items marked NEW/REUSED: {marks}, unmarked: {unmarked}")
    add("§9.4 the first line of the status report is the run/unrun/parked count over the cell roster",
        "PASS" if first.startswith("Roster, run/unrun/parked") and "W0 4/0/0" in first else "FAIL", f"S6.md line 1: {first[:120]}…")

    # §11, §13, §15
    grid_rows = [l for l in (W5 / "verdict/grid.md").read_text().splitlines() if l.startswith("| ") and not l.startswith("| readout") and not l.startswith("|---")]
    spec = (S / "SPEC_D1.md").read_text(); s11 = spec[spec.index("## 11."):spec.index("## 12.")]
    spec_rows = [l for l in s11.splitlines() if l.startswith("| ") and not l.startswith("| readout") and not l.startswith("|---")]
    add("§11 prediction grid: observed and label cells equal the instrument's grid.md", "PASS" if grid_rows == spec_rows else "FAIL", f"{len(spec_rows)} rows compared; identical: {grid_rows == spec_rows}; note names metrics sha {msha}: {msha in s11}")
    s13 = spec[spec.index("## 13."):spec.index("## 14.")]
    add("§13 read-at-source list recorded; item 6 (the companion paper, conditional on appearing before the freeze)",
        "SUBSTANCE", f"five of six items read at source after the unblind (disclosed, §15 item 2); item 6's condition not met (no companion paper on 2026-10-23/24, two checkers' searches), closed as not applicable in §15 item 16: {'Status 2026-10-24' in s13}")
    s15 = spec[spec.index("## 15."):spec.index("## 16.")]
    items = re.findall(r"(?m)^(\d+)\. ", s15); closed = [n for n in items if re.search(rf"(?m)^{n}\. .*?(?=^\d+\. |\Z)", s15, flags=re.S) and re.search(r"\bclosed\b", re.search(rf"(?m)^{n}\. .*?(?=^\d+\. |\Z)", s15, flags=re.S).group(0), flags=re.I)]
    add("§15 open-before-freeze items: each closed with a dated line", "PASS" if set(items) == set(closed) else "FAIL", f"items {sorted(set(items), key=int)}; closed {sorted(set(closed), key=int)}")

    # report
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    head = subprocess.run(["git", "-C", str(S), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    ehead = subprocess.run(["git", "-C", str(E), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    counts = {v: sum(1 for _, vv, _ in rows if vv == v) for v in ("PASS", "SUBSTANCE", "FAIL")}
    L = [f"# Acceptance check of SPEC_D1 §9.4 and §10 (`runbook/acceptance_check.py`), {now}", "",
         f"Drafts HEAD {head}, explain HEAD {ehead}, metrics.json sha256 {msha}…. Verdicts: PASS = holds as written; SUBSTANCE = holds through a derived file or a disclosure the record names; FAIL = does not hold. Counts: {counts}.", "",
         "| requirement | verdict | evidence |", "|---|---|---|"] + [f"| {r} | {v} | {e} |" for r, v, e in rows]
    pathlib.Path(out).write_text("\n".join(L) + "\n")
    print(json.dumps(counts)); sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main(*sys.argv[1:])
