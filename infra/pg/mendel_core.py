"""MENDEL — shared plumbing: artifact provenance, the atlas writer, the decision statistics.

Three jobs, kept in one module because every arm needs all three and splitting them would only
add import ceremony:

  1. PROVENANCE. `Manifest` fetches a banked artifact from S3 (or reads it from disk), hashes it,
     and records uri + local path + sha256 + bytes. §0's provenance rule says every reference
     number is a POINTER recomputed at run time and cited with artifact + SHA; a manifest entry IS
     that citation, produced mechanically so it cannot drift from what was actually read.

  2. THE ATLAS. `Atlas` extends HUMBOLDT's append-only writer with the four columns MENDEL's rows
     need beyond the house schema — `wave`, `hop_id`, `control_id`, `queue_item`. `control_id` is
     the load-bearing one: §5 requires the structured control, the haar null and the shuffle test
     on every flagship statistic, and `check_control_trinity` can only enforce that if each row
     says which member of the trinity it is.

  3. THE STATISTICS. Paired bootstrap CIs, label-permutation tests, and the two decision helpers
     the frozen rules in §2 are written in terms of (`clears_at_k_hops`, `exceeds_band`). Pure
     numpy, no torch, no S3 — so the whole decision layer is unit-testable on the control plane,
     which is what §9's weight-free dry-run gate checks.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess

from infra.pg import humboldt_core as H

BUCKET = os.environ.get("PG_BUCKET", "pg-lineage-769063704696")
RESULTS_S3 = f"s3://{BUCKET}/results"
REGION = "us-east-1"

# The control plane needs an explicit profile; the instance role must NOT pass one (there is no
# ~/.aws/config on the box). Presence of the credentials file is the discriminator.
_PROFILE = ["--profile", "fuelos"] if pathlib.Path("~/.aws/credentials").expanduser().exists() else []

# §5's control trinity, named so a row can declare which member it is. TREATMENT is the measured
# quantity itself; the three others are the controls every flagship statistic must carry.
TREATMENT = "treatment"
STRUCTURED = "structured"        # traffic- / energy-matched — never haar alone (standing rule)
HAAR = "haar"                    # the seed/rotation null band
SHUFFLE = "shuffle"              # hop-label or faculty-label permutation
TRINITY = (STRUCTURED, HAAR, SHUFFLE)
CONTROL_IDS = (TREATMENT,) + TRINITY

ATLAS_COLUMNS = H.ATLAS_COLUMNS + ("wave", "hop_id", "control_id", "queue_item")

PROVENANCE_FILE = pathlib.Path(__file__).resolve().parent / "mendel_provenance.json"
PROGRAM = "mendel"


def provenance() -> dict:
    """Run provenance for every MENDEL row — the program's OWN stamp, never another's.

    `pg run` ships only the `infra/` tree, so the instance has no .git and a job must read a
    stamped file. That file must be MENDEL's: darwin, humboldt, mercator and copernicus
    provenance files are all committed in this same tree, and `humboldt_core.provenance()` returns
    the FIRST one it finds — so a MENDEL job that skipped stamping would silently label every atlas
    row with HUMBOLDT's sha from a different branch on a different day, and the rows would look
    perfectly well-provenanced while pointing at the wrong code.

    The `program` key is the guard: a stamp belonging to another program is refused rather than
    inherited. On the control plane git answers directly and no stamp is needed.
    """
    if PROVENANCE_FILE.exists():
        rec = json.loads(PROVENANCE_FILE.read_text())
        assert rec.get("program") == PROGRAM, (
            f"{PROVENANCE_FILE.name} belongs to program {rec.get('program')!r}, not {PROGRAM!r} — "
            f"run scripts/mendel_stamp.py before launching")
        return rec
    root = pathlib.Path(__file__).resolve().parents[2]
    assert (root / ".git").exists(), (
        f"no {PROVENANCE_FILE.name} and no .git — this is an instance that was launched without "
        f"stamping; every atlas row would carry another program's sha")
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                       check=True, capture_output=True, text=True)
    d = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                       check=True, capture_output=True, text=True)
    return {"program": PROGRAM, "commit_sha": r.stdout.strip(),
            "dirty": bool(d.stdout.strip()), "source": "git (control plane)"}


def commit_sha() -> str:
    return provenance()["commit_sha"]


# ---------------------------------------------------------------------------
# 1. Provenance
# ---------------------------------------------------------------------------
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def s3_get(uri: str, dest, *, recursive: bool = False) -> pathlib.Path:
    """Fetch one S3 object (or prefix) to `dest`, idempotently.

    Idempotent because every arm re-reads the same banks and re-downloading a 1 GB library per arm
    is pure waste; `aws s3 sync` handles the prefix case and `cp` skips nothing, so the object case
    checks for the file first. A missing object must fail loud — a silently absent input is how an
    arm reports a number computed on less data than it claims.
    """
    d = pathlib.Path(dest)
    assert uri.startswith("s3://"), f"not an s3 uri: {uri!r}"
    if recursive:
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["aws", "s3", "sync", uri.rstrip("/") + "/", str(d),
                        "--region", REGION, *_PROFILE], check=True)
        return d
    d.parent.mkdir(parents=True, exist_ok=True)
    if not d.exists():
        subprocess.run(["aws", "s3", "cp", uri, str(d), "--region", REGION, "--quiet", *_PROFILE],
                       check=True)
    assert d.exists() and d.stat().st_size > 0, f"empty or missing after fetch: {uri} -> {d}"
    return d


class Manifest:
    """The reuse ledger, produced rather than transcribed.

    Every artifact an arm reads goes through `require()`, which fetches it, hashes it and records
    the row. `write()` renders `out/manifest.json`. §13's ledger is re-verified at freeze by
    running this over the whole input set — a line that cannot be fetched and hashed is not a
    verified line.
    """

    def __init__(self, cache_root, out_dir):
        self.cache = pathlib.Path(cache_root)
        self.out = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.entries = {}

    def require(self, name: str, uri: str, *, recursive: bool = False) -> pathlib.Path:
        assert name not in self.entries, f"manifest key collision: {name!r}"
        local = self.cache / name
        p = s3_get(uri, local, recursive=recursive) if uri.startswith("s3://") else pathlib.Path(uri)
        assert p.exists(), f"{name}: {uri} did not resolve to an existing path"
        if p.is_dir():
            files = sorted(q for q in p.rglob("*") if q.is_file())
            assert files, f"{name}: {uri} fetched an empty directory"
            digest = hashlib.sha256()
            for q in files:
                digest.update(q.relative_to(p).as_posix().encode())
                digest.update(sha256_file(q).encode())
            entry = {"uri": uri, "local": str(p), "kind": "prefix", "n_files": len(files),
                     "bytes": sum(q.stat().st_size for q in files),
                     "sha256": digest.hexdigest()}
        else:
            entry = {"uri": uri, "local": str(p), "kind": "object",
                     "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        self.entries[name] = entry
        print(f"[manifest] {name}: {entry['sha256'][:12]} {entry['bytes']}B {uri}", flush=True)
        return p

    def write(self) -> dict:
        doc = {"provenance": provenance(), "bucket": BUCKET, "entries": self.entries}
        (self.out / "manifest.json").write_text(json.dumps(doc, indent=2, sort_keys=True))
        return doc


def write_metrics(out_dir, doc: dict) -> pathlib.Path:
    """out/metrics.json — the ONLY surface a status check may poll (house compute policy: job.log
    is multi-MB sync spam and repeated fetches once filled the control plane's disk).

    Written last, so its presence means the arm reached its end. Anything a monitor needs to decide
    'done / failed / still running' belongs in here, not in the log.
    """
    p = pathlib.Path(out_dir) / "metrics.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"provenance": provenance(), **H._jsonable(doc)},
                            indent=2, sort_keys=True))
    return p


# ---------------------------------------------------------------------------
# 2. The atlas
# ---------------------------------------------------------------------------
class Atlas(H.Atlas):
    """Append-only atlas writer with MENDEL's four extra columns.

    `wave` and `queue_item` are constructor-bound (an arm runs in exactly one wave and maps to
    exactly one queue item); `hop_id` and `control_id` are per-call, because one arm writes many
    hops and every statistic writes its treatment row beside its control rows.
    """

    def __init__(self, out_dir, arm: str, run_id: str, *, wave: str, queue_item: str,
                 tag: str = H.DISCOVERY):
        from infra.pg.mendel_program import WAVES
        assert wave in WAVES, f"unknown wave {wave!r} (expected one of {WAVES})"
        assert str(queue_item).strip(), f"{arm}: queue_item may not be empty"
        super().__init__(out_dir, arm=arm, run_id=run_id, tag=tag)
        # Re-stamp: H.Atlas.__init__ set self.sha from HUMBOLDT's committed provenance file, which
        # on an instance is a different program's sha on a different branch.
        self.sha = commit_sha()
        self.wave = wave
        self.queue_item = str(queue_item)
        self.stat_controls = {}        # statistic -> set(control_id), for check_control_trinity

    def add(self, statistic: str, *, value=None, vector_ref=None, layer=None, type=None,
            bank_id=None, model_id=None, model_sha=None, n=None, ci_lo=None, ci_hi=None,
            seed=None, corpus_id=None, item_id=None, token_class=None, flags=None, tag=None,
            hop_id=None, control_id=TREATMENT):
        assert control_id in CONTROL_IDS, \
            f"{statistic}: unknown control_id {control_id!r} (expected one of {CONTROL_IDS})"
        assert (value is None) != (vector_ref is None), \
            f"{statistic}: exactly one of value / vector_ref (got {value!r} / {vector_ref!r})"
        if value is not None:
            value = float(value)
            assert value == value, f"{statistic}: NaN value (record it as a flag, not a silent NaN)"
        self.stat_controls.setdefault(statistic, set()).add(control_id)
        row = {"arm": self.arm, "model_id": model_id, "model_sha": model_sha,
               "layer": None if layer is None else int(layer), "type": type, "bank_id": bank_id,
               "statistic": statistic, "value": value, "vector_ref": vector_ref,
               "n": None if n is None else int(n),
               "ci_lo": None if ci_lo is None else float(ci_lo),
               "ci_hi": None if ci_hi is None else float(ci_hi),
               "seed": None if seed is None else int(seed), "corpus_id": corpus_id,
               "item_id": item_id, "token_class": token_class,
               "flags": json.dumps(H._jsonable(flags or {}), sort_keys=True),
               "run_id": self.run_id, "commit_sha": self.sha, "tag": tag or self.tag,
               "wave": self.wave, "hop_id": hop_id, "control_id": control_id,
               "queue_item": self.queue_item}
        assert set(row) == set(ATLAS_COLUMNS), \
            f"schema drift: {sorted(set(row) ^ set(ATLAS_COLUMNS))}"
        self._fh.write(json.dumps(row) + "\n")
        self.n_rows += 1

    def add_ci(self, statistic: str, values, **kw):
        seed = kw.pop("seed", 1337)
        s = H.mean_ci(values, seed=seed)
        flags = {**(kw.pop("flags", None) or {}), "ci_method": s["ci_method"],
                 "n_boot": s["n_boot"]}
        self.add(statistic, value=s["mean"], n=s["n"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"],
                 seed=seed, flags=flags, **kw)
        return s

    def check_control_trinity(self, statistics):
        """§5 made mechanical: every flagship statistic carries all three controls beside it.

        A statistic that was never written is as much a failure as one written without its
        controls — silent absence is how a control requirement degrades into an uncontrolled
        number. Returns the per-statistic table so the caller can bank it.
        """
        report = {}
        missing = {}
        for s in statistics:
            seen = self.stat_controls.get(s, set())
            report[s] = sorted(seen)
            gap = [c for c in TRINITY if c not in seen]
            if TREATMENT not in seen:
                gap = ["treatment(absent)"] + gap
            if gap:
                missing[s] = gap
        assert not missing, (
            f"{self.arm}: §5 control trinity incomplete — {json.dumps(missing, sort_keys=True)}")
        return report

    def to_parquet(self):
        import pandas as pd
        self.close()
        df = pd.read_json(self.path, lines=True)
        if df.empty:
            return {"n_rows": 0, "path": str(self.path), "parquet": None,
                    "note": "atlas empty: the arm recorded an outcome before adding any row"}
        assert list(df.columns) == list(ATLAS_COLUMNS), \
            f"parquet column order drift: {list(df.columns)}"
        for c in ("model_id", "model_sha", "type", "bank_id", "vector_ref", "corpus_id",
                  "item_id", "token_class", "flags", "run_id", "commit_sha", "tag", "arm",
                  "statistic", "wave", "hop_id", "control_id", "queue_item"):
            df[c] = df[c].astype("object").where(df[c].notna(), None)
        p = self.out / "atlas.parquet"
        df.to_parquet(p, index=False)
        return {"path": str(p), "n_rows": int(len(df)),
                "n_vectors": len(list(self.vec.glob("*.npy")))}


# ---------------------------------------------------------------------------
# 3. Decision statistics
# ---------------------------------------------------------------------------
N_BOOT = 2000            # §7: bootstrap CIs on every mean, >= 2000 resamples


def paired_delta_ci(treatment, control, *, n_boot: int = N_BOOT, seed: int = 1337) -> dict:
    """Bootstrap CI on the PAIRED difference mean(treatment - control).

    Paired on purpose. The cells of a hop (type x layer) differ enormously in how predictable
    their address is, and an unpaired comparison of two means over the same cells throws that
    shared variation into the noise term — which is exactly the regime where an effect the size of
    the ones §11 predicts would be lost. The pairing is by position, so callers must pass the two
    vectors in the same cell order; the assert below is the only thing standing between a
    mis-zipped pair and a confident wrong CI, so it is not optional.
    """
    import numpy as np
    t = np.asarray(treatment, dtype=np.float64).ravel()
    c = np.asarray(control, dtype=np.float64).ravel()
    assert t.shape == c.shape, f"paired vectors differ in length: {t.shape} vs {c.shape}"
    assert t.size > 0, "paired_delta_ci on an empty sample"
    assert np.isfinite(t).all() and np.isfinite(c).all(), "paired_delta_ci saw non-finite values"
    d = t - c
    s = H.mean_ci(d, n_boot=n_boot, seed=seed)
    return {"delta_mean": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "n": s["n"],
            "ci_method": s["ci_method"], "n_boot": s["n_boot"], "seed": int(seed),
            "clears_zero": bool(s["ci_lo"] > 0.0),
            "treatment_mean": float(t.mean()), "control_mean": float(c.mean())}


def permutation_pvalue_multi(values, labels, *, n_perm: int = 10000, seed: int = 1337) -> dict:
    """Label-permutation test for MORE THAN TWO groups (§5's shuffle test, k-group form).

    `permutation_pvalue` is structurally two-group: its statistic is a difference of group means
    and it enumerates with comb(n, n_a). Handed three labels it asserts rather than guessing, which
    is right — but an arm with three groups then has no shuffle member at all, and MN-13 (three
    faculty hops) hit exactly that, after computing its entire 810-cell grid.

    The statistic here is the between-group sum of squares,  sum_g n_g (mean_g - grand_mean)^2,
    which is the natural k-group generalisation: it is large when the labels carry information
    about the value and zero when every group has the same mean. It is one-sided by construction —
    the between-group SS cannot be negative, and "more separated than chance" is the only
    direction that means anything — so unlike the two-group test there is no `two_sided` knob to
    get wrong.

    Sampled only. Exact enumeration over k-group assignments is multinomial and is not attempted
    for any size this program produces; the method is RECORDED either way so a reader never has to
    guess which they were given.
    """
    import numpy as np
    v = np.asarray(values, dtype=np.float64).ravel()
    lab = np.asarray(labels).ravel()
    assert v.shape == lab.shape, f"values/labels length mismatch {v.shape} vs {lab.shape}"
    uniq = sorted(set(lab.tolist()))
    assert len(uniq) >= 2, f"permutation_pvalue_multi needs at least two groups, got {uniq}"
    masks = [lab == g for g in uniq]
    assert all(m.sum() > 0 for m in masks), "every group must be non-empty"

    def stat(order):
        grand = v.mean()
        w = v[order]
        out, i = 0.0, 0
        for m in masks:
            n = int(m.sum())
            out += n * (float(w[i:i + n].mean()) - grand) ** 2
            i += n
        return out

    ident = np.concatenate([np.nonzero(m)[0] for m in masks])
    obs = stat(ident)
    rng = np.random.default_rng(seed)
    draws = np.array([stat(rng.permutation(v.size)) for _ in range(n_perm)])
    # +1 top and bottom, as in the two-group test: the observed labelling is itself a permutation.
    p = float(((draws >= obs).sum() + 1) / (draws.size + 1))
    return {"observed": float(obs), "p_value": p, "n_draws": int(draws.size),
            "method": "sampled", "statistic": "between_group_sum_of_squares",
            "two_sided": False, "seed": int(seed), "groups": uniq,
            "group_means": {str(g): float(v[m].mean()) for g, m in zip(uniq, masks)}}


def permutation_pvalue(values, labels, *, n_perm: int = 10000, seed: int = 1337,
                       two_sided: bool = True) -> dict:
    """Label-permutation test on the difference of group means (§5's shuffle test).

    Exhaustive enumeration when the number of distinct label assignments is small enough to
    enumerate exactly (§7: "permutation tests where n permits exact enumeration"), sampled
    otherwise. Which regime ran is RECORDED — a sampled p-value and an exact one are different
    objects and a reader must not have to guess which they were given.
    """
    import itertools
    import math

    import numpy as np
    v = np.asarray(values, dtype=np.float64).ravel()
    lab = np.asarray(labels).ravel()
    assert v.shape == lab.shape, f"values/labels length mismatch {v.shape} vs {lab.shape}"
    uniq = sorted(set(lab.tolist()))
    assert len(uniq) == 2, f"permutation_pvalue expects two groups, got {uniq}"
    a_mask = lab == uniq[0]
    n_a = int(a_mask.sum())
    assert 0 < n_a < v.size, "permutation_pvalue needs both groups non-empty"

    def stat(mask):
        return float(v[mask].mean() - v[~mask].mean())

    obs = stat(a_mask)
    n_total = math.comb(v.size, n_a)
    if n_total <= n_perm:
        draws = np.array([stat(np.isin(np.arange(v.size), c))
                          for c in itertools.combinations(range(v.size), n_a)])
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        idx = np.argsort(rng.random((n_perm, v.size)), axis=1)[:, :n_a]
        m = np.zeros((n_perm, v.size), dtype=bool)
        np.put_along_axis(m, idx, True, axis=1)
        draws = (v[None, :] * m).sum(1) / n_a - (v[None, :] * ~m).sum(1) / (v.size - n_a)
        method = "sampled"
    extreme = np.abs(draws) >= abs(obs) if two_sided else draws >= obs
    # +1 in numerator and denominator: the observed assignment is itself a permutation, so a
    # p-value of exactly 0 is not attainable and reporting one would overstate the evidence.
    p = float((extreme.sum() + 1) / (draws.size + 1))
    return {"observed": obs, "p_value": p, "n_draws": int(draws.size), "method": method,
            "two_sided": bool(two_sided), "seed": int(seed), "groups": uniq}


def clears_at_k_hops(per_hop: dict, *, k: int = 2) -> dict:
    """MN-1/MN-2's frozen rule shape: 'clears with CI clear of 0 at >= k independent hops'.

    `per_hop` maps hop_id -> a `paired_delta_ci` result. Returns the hops that clear, the count,
    and the boolean the frozen rule reads. Kept here rather than inlined in each arm so the two
    arms that share this rule shape cannot drift apart in how they apply it.
    """
    clearing = sorted(h for h, r in per_hop.items() if r["clears_zero"])
    return {"clearing_hops": clearing, "n_clearing": len(clearing), "k_required": int(k),
            "clears": len(clearing) >= k, "n_hops": len(per_hop)}


def exceeds_band(value: float, band_lo: float, band_hi: float) -> dict:
    """Is a measured fraction above its control BAND (not merely above the control mean)?

    §5 forbids haar-alone comparisons; the structured control is a band because it is drawn with a
    seed and its own spread is part of what 'no effect' looks like. Reporting 'above the mean'
    where the rule says 'above the band' is the single easiest way to manufacture an effect, so
    the band edges travel with the answer.
    """
    return {"value": float(value), "band_lo": float(band_lo), "band_hi": float(band_hi),
            "above_band": bool(value > band_hi), "below_band": bool(value < band_lo),
            "in_band": bool(band_lo <= value <= band_hi)}


def bh_fdr(pvalues, q: float = 0.10) -> dict:
    """Benjamini-Hochberg at q (§7: q=0.10 within each confirmatory family).

    Returns the rejection mask in the INPUT order plus the threshold, so a caller can attach the
    decision to the right row without re-sorting and mis-aligning it.
    """
    import numpy as np
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    assert p.size > 0, "bh_fdr on an empty family"
    assert ((p >= 0) & (p <= 1)).all(), f"p-values outside [0,1]: {p}"
    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, p.size + 1)
    crit = q * ranks / p.size
    passing = p[order] <= crit
    n_reject = int(np.max(np.nonzero(passing)[0]) + 1) if passing.any() else 0
    thresh = float(crit[n_reject - 1]) if n_reject else 0.0
    rej = np.zeros(p.size, dtype=bool)
    rej[order[:n_reject]] = True
    return {"reject": rej.tolist(), "n_reject": n_reject, "threshold": thresh, "q": float(q),
            "n": int(p.size)}


# ---------------------------------------------------------------------------
# The energy-matched control, with its reachability checked
# ---------------------------------------------------------------------------
ENERGY_MATCH_TOL = 0.03


def energy_matched_control(act, k: int, e_target: float, seed: int, *, tol: float = ENERGY_MATCH_TOL):
    """A k-dim subspace matched on removed activation energy — AND a verdict on whether it matched.

    `conserved_core_probes.energy_matched_subspace` returns its CLOSEST attempt when it cannot hit
    the target, and says so only through the returned energy. That is a silent failure with teeth:
    a "matched" control that actually captures 350x the target energy is not a control, and because
    write fraction rises with captured energy it biases the contrast in favour of whatever the sink
    is being compared against.

    The failure is structural, not a tuning problem. A k-dim subspace of an n-dim space cannot
    capture arbitrarily little activation energy — the floor is set by the flattest directions
    available — so a sink capturing LESS than that floor has no matched control at all. Measured on
    MENDEL's own substrate, 39.7% of MN-6's cells were in that regime (q_proj L0's sink captures
    4.5e-5 of the energy; the closest reachable 32-dim subspace captures 1.6e-2).

    Returns (Q, info) where info carries the achieved energy, the relative error and `matched`.
    Callers must EXCLUDE or separately report unmatched cells; this function refuses to hide them.
    """
    import numpy as np
    e2 = np.asarray(act, dtype=np.float64) ** 2
    assert e2.ndim == 1 and (e2 >= 0).all() and e2.sum() > 0, "not a positive energy profile"
    order = np.argsort(-e2)
    return energy_matched_control_gram(np.eye(e2.size)[:, order], e2[order], k, e_target, seed,
                                       tol=tol, basis="diagonal")


def energy_matched_control_gram(evecs, evals, k: int, e_target: float, seed: int, *,
                                tol: float = ENERGY_MATCH_TOL, basis: str = "gram"):
    """The same control built on a REAL activation eigenbasis rather than the coordinate diagonal.

    G4b banks `act_gram_evecs`/`act_gram_evals` (top 512 per site) precisely so this control stops
    being coordinate-aligned; its own metrics name MN-2/MN-13 as the consumers. The diagonal
    version above is what MN-2 and MN-6 shipped with, and it makes the control axis-aligned, which
    is a real restriction: it can only ever select whole coordinates, so it cannot match a sink
    that lies obliquely to the coordinate frame.

    Both paths return the same (Q, info) contract and both report whether the match held.
    """
    import numpy as np
    from infra.pg import conserved_core_probes as CCP
    V = np.asarray(evecs, dtype=np.float64)
    e = np.asarray(evals, dtype=np.float64)
    assert V.ndim == 2 and e.ndim == 1 and V.shape[1] == e.size, (V.shape, e.shape)
    assert (e >= 0).all() and e.sum() > 0, "not a positive energy spectrum"
    assert (np.diff(e) <= 1e-9 * e.max()).all(), "eigenvalues must be DESCENDING for the m-ladder"
    Q, got = CCP.energy_matched_subspace(V, e, e_target, k, seed, tol=tol)
    rel = abs(got - e_target) / e_target if e_target > 0 else float("inf")
    return Q, {"e_target": float(e_target), "e_achieved": float(got), "rel_error": float(rel),
               "tol": float(tol), "matched": bool(rel <= tol), "k": int(k), "n": int(V.shape[0]),
               "basis": basis, "n_basis": int(e.size)}


def load_activation_eigenbank(vec_dir, t: str, layer: int):
    """G4b's banked activation eigensystem for one site, with its two halves ALIGNED.

    G4b banks the eigenVECTORS truncated to its top_k (512 of 2048) and the eigenVALUES in FULL —
    `mendel_g4b_activation_gram.py:150` truncates V, line 151 does not truncate w. Every consumer
    must therefore cut the spectrum to the banked width before using the pair, and two of them did
    not: MN-13 and MN-7 each died with `index 512 is out of bounds for axis 1 with size 512`. MN-7
    died after a 48 GB weight sync and 400 completed training steps, on the first site it touched.
    I fixed MN-13 and did not grep for the same pattern, which is why MN-7 then repeated it.

    Centralised so a third consumer cannot. Returns (V, e, trace_share) where trace_share is the
    fraction of the activation trace the banked span holds — worth carrying, because everything
    measured against this basis is measured inside that span.

    The descending convention is ASSERTED rather than imposed: the truncation already discarded
    the eigenvectors a reordering would need, so there is nothing to reorder with.
    """
    import numpy as np
    d = pathlib.Path(vec_dir)
    V = np.load(d / f"act_gram_evecs__{t}__L{layer}.npy").astype(np.float64)
    e_full = np.load(d / f"act_gram_evals__{t}__L{layer}.npy").astype(np.float64)
    assert V.ndim == 2 and e_full.ndim == 1, (V.shape, e_full.shape)
    assert e_full.size >= V.shape[1], (
        f"{t}/L{layer}: {e_full.size} eigenvalues for {V.shape[1]} eigenvectors — fewer values "
        f"than vectors means the pair cannot be aligned at all")
    assert (np.diff(e_full) <= 1e-9 * max(float(e_full.max()), 1.0)).all(), (
        f"{t}/L{layer}: eigenvalues are not DESCENDING, so the banked top-k vectors are not the "
        f"top-k of this spectrum and the truncation cannot be aligned")
    e = e_full[: V.shape[1]]
    return V, e, float(e.sum() / e_full.sum())


def sink_energy_share(evecs, evals, U) -> float:
    """tr(Uᵀ Σ U) / tr(Σ) for Σ = V diag(evals) Vᵀ — the activation energy the sink removes.

    Computed inside the banked top-512 span on BOTH sides, so the target and whatever the control
    achieves are measured against the same Σ. G4b's truncation keeps a median 94% of the energy;
    using the truncated trace in the denominator and the full one in the numerator would be an
    inconsistency that shows up as a systematic match error, so it is not done.
    """
    import numpy as np
    V = np.asarray(evecs, dtype=np.float64)
    e = np.asarray(evals, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    assert V.shape[0] == U.shape[0], (V.shape, U.shape)
    return float((e * ((V.T @ U) ** 2).sum(axis=1)).sum() / e.sum())


def energy_match_report(infos) -> dict:
    """Pooled match quality over a set of cells, for the gate that must accompany every use."""
    import numpy as np
    rel = np.asarray([i["rel_error"] for i in infos], dtype=np.float64)
    matched = [bool(i["matched"]) for i in infos]
    return {"n_cells": len(infos), "n_matched": int(sum(matched)),
            "fraction_matched": float(np.mean(matched)) if infos else 0.0,
            "median_rel_error": float(np.median(rel)), "max_rel_error": float(rel.max()),
            "tol": infos[0]["tol"] if infos else None}
