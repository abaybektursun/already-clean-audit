"""conserved-core Tier 2 — per-subject policy-erasure structure (CPU re-analysis of PCR banks).

Re-derives the per-item MMLU subject labels by replaying mmlu_cloze's sorted(test_by_subj) iteration over
cais/mmlu (n_per_subject=None, full test), aligns to the PCR banks via per_item_hash (sha1(question[:64])
[:12]) — asserted, FAIL LOUD — and tests whether the deployed-policy content erasure is UNIFORM or
STRUCTURED across subjects (per-subject cloze−floor gap vs the frozen spatial proxy, §3.2). CPU only,
no GPU, no model — runs on an r-family instance (Standard quota).

  pg run infra.aws_pg.jobs.policy_subject_structure \
      --args="--banks-s3 s3://pg-lineage-769063704696/results/pcr_full/out/policy_content_readout_banks.json" \
      --no-weights --type r7i.xlarge --s3-prefix results/policy_subject_structure --budget-min 30
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
from collections import defaultdict

from infra.pg import subject_structure as SS


def _derive_subjects_and_hashes(n_per_subject=None):
    """Replay mmlu_cloze's per-item iteration EXACTLY: sorted(test_by_subj), dataset row order within a
    subject. Returns (subjects, hashes) aligned position-for-position with the PCR banks."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all")
    test = ds["test"]
    test_by_subj = defaultdict(list)
    for r in test:
        test_by_subj[r["subject"]].append(r)
    subjects, hashes = [], []
    for subj in sorted(test_by_subj):
        rows = test_by_subj[subj]
        if n_per_subject:
            rows = rows[:n_per_subject]
        for r in rows:
            subjects.append(subj)
            hashes.append(hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12])
    return subjects, hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--banks-s3", required=True, help="s3 uri of policy_content_readout_banks.json")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                       # cais/mmlu metadata download needs online + token

    local_banks = out / "banks.json"
    subprocess.run(["aws", "s3", "cp", a.banks_s3, str(local_banks), "--region", "us-east-1"], check=True)
    banks = json.loads(local_banks.read_text())
    arms = list(banks)
    ref_hash = banks[arms[0]]["per_item_hash"]
    for nm in arms:
        assert banks[nm]["per_item_hash"] == ref_hash, f"{nm}: per_item_hash differs across arms"
    n_items = len(ref_hash)
    print(f"[t2] banks: arms={arms} n_items={n_items}", flush=True)

    subjects, hashes = _derive_subjects_and_hashes(n_per_subject=None)
    assert len(hashes) == n_items, f"replayed {len(hashes)} items vs banks {n_items}"
    # ALIGNMENT GATE (fail loud): the replayed hash sequence must match the banks exactly.
    n_match = sum(int(h == g) for h, g in zip(hashes, ref_hash))
    assert hashes == ref_hash, \
        f"subject re-derivation misaligned: {n_match}/{n_items} hashes match (order/version drift)"
    print(f"[t2] alignment OK: {n_items}/{n_items} per_item_hash match", flush=True)

    floor = banks["floor"]["cloze_per_item"]
    rep = {"object": "policy_subject_structure_v1", "n_items": n_items, "arms": arms,
           "proxy_keywords": list(SS.SPATIAL_KEYWORDS), "banks_s3": a.banks_s3}

    # per-subject cloze + mcq for every arm (context).
    rep["per_subject_cloze"] = {nm: SS.per_subject_accuracy(subjects, banks[nm]["cloze_per_item"]) for nm in arms}
    rep["per_subject_mcq"] = {nm: SS.per_subject_accuracy(subjects, banks[nm]["mcq_per_item"]) for nm in arms}

    # the §3.2 readout: per-subject cloze−floor gap vs the frozen spatial proxy, for each policy.
    rep["gap_vs_proxy"] = {}
    for pol in ("pi0", "pi05"):
        res = SS.gap_proxy_spearman(subjects, banks[pol]["cloze_per_item"], floor, n_boot=2000, seed=1337)
        rep["gap_vs_proxy"][pol] = res
        print(f"[t2] {pol}: cloze−floor gap vs spatial-proxy ρ={res['rho']:+.3f} "
              f"CI[{res['lo']:+.3f},{res['hi']:+.3f}] structured={res['structured']} "
              f"(spatial={res['n_spatial']} verbal={res['n_verbal']})", flush=True)

    (out / "policy_subject_structure.json").write_text(json.dumps(rep, indent=2))
    print("=" * 78, flush=True)
    for pol in ("pi0", "pi05"):
        r = rep["gap_vs_proxy"][pol]
        print(f"POLICY_SUBJECT_STRUCTURE {pol}: {'STRUCTURED' if r['structured'] else 'UNIFORM'} "
              f"ρ={r['rho']:+.3f} CI[{r['lo']:+.3f},{r['hi']:+.3f}]", flush=True)
    print(f"POLICY_SUBJECT_STRUCTURE_OK n_items={n_items}", flush=True)


if __name__ == "__main__":
    main()
