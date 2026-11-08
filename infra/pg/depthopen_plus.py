"""DEPTHOPEN R5 — LIBERO-plus, the perturbation benchmark with headroom.

WHY THIS BENCHMARK. SPEC §0: standard LIBERO sits at 97-100 %, where a 0.9-point difference needs about
4,700 episodes per arm. The place a depth-using policy could show as better is a benchmark with room, and
LIBERO-plus is one: its own README reports π₀ at 13.8 %, π₀-Fast at 65.1 % and OpenVLA-OFT at 56.4 % on the
camera-viewpoint dimension. No MolmoAct number exists, which is why §5 measures the released checkpoint's
row first.

WHAT IT IS, MEASURED FROM THE REPOSITORY at the pinned commit rather than from its paper: 10,030 tasks over
four suites, each labelled with one of seven perturbation categories and a difficulty level.
`Camera Viewpoints` has 1,599 tasks and `Objects Layout` 1,525 — the two dimensions §3 R5 evaluates.

HOW IT IS INSTALLED. LIBERO-plus IS the `libero` package ("simply replace the originally installed LIBERO
repository with our repository"), so it is installed over the DEPTHANCHOR environment with `--no-deps`:
robosuite, mujoco and bddl stay the versions the released checkpoint was evaluated under, and the
benchmark's own `extra_requirements.txt` — which pins numpy 1.22, transformers 4.21 and opencv 4.6 — is
NOT installed, because it would replace the stack the model runs on. What the environment actually needs
beyond that is discovered by an import probe and banked.

The join between the task list and the perturbation labels is on the task NAME, not on the file's `id`
field: the benchmark builds its task order from `libero_task_map`, and a positional id would silently
mislabel a whole stratum if that order ever differed.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

LP_URL = "https://github.com/sylvestf/LIBERO-plus.git"
LP_SHA = "4976dc30028e805ff8094b55501d532c48fec182"   # recon, 2026-09-09
LP_ASSETS_REPO = "Sylvest/LIBERO-plus"
LP_ASSETS_FILE = "assets.zip"
S3_ASSETS = "s3://pg-lineage-769063704696/depthopen/liberoplus_assets/v1"
CLASSIFICATION = "libero/libero/benchmark/task_classification.json"
CAMERA = "Camera Viewpoints"
LAYOUT = "Objects Layout"
DIMENSIONS = (CAMERA, LAYOUT)
# module -> package for the imports LIBERO-plus adds on top of LIBERO
PACKAGE_FOR_MODULE = {"wand": "wand", "skimage": "scikit-image", "easydict": "easydict",
                      "hydra": "hydra-core", "cv2": "opencv-python-headless", "thop": "thop",
                      "pxr": "usd-core", "robomimic": "robomimic", "gym": "gym"}
NEVER_INSTALL = {"torch", "torchvision", "numpy", "transformers", "robosuite", "mujoco", "bddl"}


def _stable_key(text: str) -> int:
    """A process-independent integer for a label, for seeding. `hash()` is salted per process."""
    import hashlib
    return int(hashlib.sha256(str(text).encode()).hexdigest()[:8], 16)


def _run(cmd: Sequence[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), check=True, **kw)


def clone(work: pathlib.Path) -> pathlib.Path:
    root = work / "LIBERO-plus"
    if not (root / ".git").is_dir():
        _run(["git", "clone", "--quiet", LP_URL, str(root)])
    _run(["git", "-C", str(root), "checkout", "--quiet", LP_SHA])
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != LP_SHA:
        raise RuntimeError(f"LIBERO-plus checkout is {head}, wanted {LP_SHA}")
    return root


def fetch_assets(root: pathlib.Path, token: Optional[str] = None,
                 mirror: str = S3_ASSETS) -> Dict[str, object]:
    """`assets.zip` (6.4 GB) unzipped into `libero/libero/`, as the benchmark's README instructs.

    The perturbed scenes reference textures and objects that only exist in this archive; without it a
    camera-viewpoint task can still load while a background-texture task cannot, which would look like a
    model failure rather than a missing file.

    The Hub copy comes down at about 5 MB/s, which is 20 minutes every time a new instance starts, so the
    unzipped tree is mirrored to S3 on the first fetch and synced in-region afterwards — the same thing
    DEPTHANCHOR does for the base LIBERO assets.
    """
    import shutil
    target = root / "libero" / "libero"
    marker = target / "assets"
    before = sum(1 for _ in marker.rglob("*")) if marker.is_dir() else 0
    have_tar = subprocess.run(["aws", "s3", "ls", f"{mirror}/assets.tar", "--region", "us-east-1"],
                              capture_output=True, text=True).returncode == 0
    have_tree = subprocess.run(["aws", "s3", "ls", f"{mirror}/DONE", "--region", "us-east-1"],
                               capture_output=True, text=True).returncode == 0
    if have_tar:
        # ONE object, not 448,802: the loose-object mirror is correct but `aws s3 sync` of a tree that
        # size spends tens of minutes on tiny GETs, and every R5 run pays it
        tar = root / "assets_mirror.tar"
        _run(["aws", "s3", "cp", f"{mirror}/assets.tar", str(tar), "--region", "us-east-1",
              "--only-show-errors"])
        marker.mkdir(parents=True, exist_ok=True)
        _run(["tar", "xf", str(tar), "-C", str(marker)])
        tar.unlink()
        after = sum(1 for _ in marker.rglob("*"))
        print(f"[assets] from the S3 tarball: {before} -> {after} files", flush=True)
        return {"source": "s3_tarball", "mirror": mirror, "assets_dir": str(marker),
                "files_before": before, "files_after": after}
    if have_tree:
        _run(["aws", "s3", "sync", f"{mirror}/tree", str(marker), "--region", "us-east-1",
              "--only-show-errors"])
        after = sum(1 for _ in marker.rglob("*")) if marker.is_dir() else 0
        print(f"[assets] from the S3 object mirror: {before} -> {after} files", flush=True)
        _upload_tarball(marker, mirror, root)
        return {"source": "s3_mirror", "mirror": mirror, "assets_dir": str(marker),
                "files_before": before, "files_after": after, "tarball_uploaded": True}

    from huggingface_hub import hf_hub_download
    zip_path = pathlib.Path(hf_hub_download(LP_ASSETS_REPO, LP_ASSETS_FILE, repo_type="dataset",
                                            token=token))
    listing = subprocess.run(["unzip", "-l", str(zip_path)], capture_output=True, text=True)
    head = [l.split()[-1] for l in listing.stdout.splitlines()[3:23] if l.strip() and l.split()[-1]]
    print(f"[assets] zip {zip_path.stat().st_size / 1e9:.2f} GB; first entries: {head[:6]}", flush=True)
    # The archive's internal layout is NOT assumed: attempt 2 unzipped it straight into
    # `libero/libero` and the asset tree did not appear there, so the S3 mirror captured only the 402 MB
    # base tree. Extract to a scratch directory, FIND the tree, then merge it into place.
    extract = root / "_assets_zip"
    if extract.exists():
        shutil.rmtree(extract)
    extract.mkdir(parents=True)
    _run(["unzip", "-q", "-o", str(zip_path), "-d", str(extract)])
    src = _locate_assets_tree(extract)
    print(f"[assets] tree located at {src}", flush=True)
    shutil.copytree(src, marker, dirs_exist_ok=True)
    shutil.rmtree(extract)
    after = sum(1 for _ in marker.rglob("*")) if marker.is_dir() else 0
    if after <= before:
        raise RuntimeError(f"assets.zip added nothing to {marker}: {before} -> {after} files")
    _run(["aws", "s3", "sync", str(marker), f"{mirror}/tree", "--region", "us-east-1",
          "--only-show-errors"])
    _upload_tarball(marker, mirror, root)
    done = root / "ASSETS_DONE"
    done.write_text(json.dumps({"files": after, "zip_bytes": zip_path.stat().st_size,
                                "zip_first_entries": head[:6]}))
    _run(["aws", "s3", "cp", str(done), f"{mirror}/DONE", "--region", "us-east-1"])
    print(f"[assets] mirrored {after} files to {mirror}", flush=True)
    return {"source": "huggingface_then_mirrored", "zip": str(zip_path),
            "zip_bytes": zip_path.stat().st_size, "zip_first_entries": head[:6],
            "located_at": str(src), "mirror": mirror,
            "assets_dir": str(marker), "files_before": before, "files_after": after}


def _upload_tarball(marker: pathlib.Path, mirror: str, root: pathlib.Path) -> None:
    """One tar of the asset tree beside the loose-object mirror, because the tree is 448k files."""
    tar = root / "assets_upload.tar"
    _run(["tar", "cf", str(tar), "-C", str(marker), "."])
    _run(["aws", "s3", "cp", str(tar), f"{mirror}/assets.tar", "--region", "us-east-1",
          "--only-show-errors"])
    print(f"[assets] tarball {tar.stat().st_size / 1e9:.2f} GB -> {mirror}/assets.tar", flush=True)
    tar.unlink()


def _locate_assets_tree(extract: pathlib.Path) -> pathlib.Path:
    """The directory inside the extracted archive that IS the asset tree.

    Identified by content rather than by name: LIBERO's asset tree carries `scenes/` and the object
    directories, so the deepest directory holding `scenes` is the tree wherever the archive put it.
    """
    candidates = [p.parent for p in extract.rglob("scenes") if p.is_dir()]
    if not candidates:
        listing = sorted(str(p.relative_to(extract)) for p in extract.rglob("*"))[:20]
        raise FileNotFoundError(f"no directory containing `scenes` inside the archive; saw {listing}")
    # prefer one that is itself called assets, else the shallowest candidate
    named = [c for c in candidates if c.name == "assets"]
    return (named or sorted(candidates, key=lambda p: len(p.parts)))[0]


def write_config(root: pathlib.Path) -> Dict[str, object]:
    """`~/.libero/config.yaml`, pointing at this clone. The package reads it at import."""
    lib = root / "libero" / "libero"
    cfg_dir = pathlib.Path.home() / ".libero"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    # LIBERO reads these five paths at import and some helpers stat them; `datasets` holds the
    # demonstrations, which nothing here needs, so it is created empty rather than left dangling
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    for required in (lib / "bddl_files", lib / "init_files"):
        if not required.is_dir():
            raise FileNotFoundError(f"the LIBERO-plus clone has no {required}")
    payload = {
        "benchmark_root": str(lib),
        "bddl_files": str(lib / "bddl_files"),
        "init_states": str(lib / "init_files"),
        "datasets": str(root / "datasets"),
        "assets": str(lib / "assets"),
    }
    (cfg_dir / "config.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in payload.items()) + "\n")
    return {"path": str(cfg_dir / "config.yaml"), **payload}


APT_PACKAGES = "libmagickwand-dev libexpat1 libfontconfig1-dev libpython3-stdlib unzip"


def install_system_packages() -> Dict[str, object]:
    """The system libraries LIBERO-plus's own README installs.

    `wand` is a binding, not a library: `libero/libero/envs/env_wrapper.py` imports it at module scope for
    the background-texture perturbations, so without ImageMagick's shared library NO environment loads at
    all. Attempt 2 died exactly there, after a 36-minute asset download. `DPkg::Lock::Timeout` because the
    DLAMI runs unattended-upgrades at boot.
    """
    cmd = (f"apt-get -o DPkg::Lock::Timeout=900 update -y && DEBIAN_FRONTEND=noninteractive "
           f"apt-get -o DPkg::Lock::Timeout=900 install -y {APT_PACKAGES}")
    r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"apt failed ({r.returncode}): {r.stdout[-800:]}{r.stderr[-800:]}")
    probe = subprocess.run([sys.executable, "-c", "import wand.api; print('WAND_OK')"],
                           capture_output=True, text=True)
    return {"packages": APT_PACKAGES, "wand_importable": "WAND_OK" in probe.stdout,
            "wand_error": None if "WAND_OK" in probe.stdout else probe.stderr[-300:]}


def adopt_base_assets(root: pathlib.Path) -> Dict[str, object]:
    """Copy the base LIBERO asset tree out of the installed `libero` package into the clone.

    `hf-libero` has to be uninstalled — two distributions cannot both own the `libero` import name, and
    the site-packages copy wins — but DEPTHANCHOR's mirrored base assets live inside it. They are copied
    into the clone first, and `assets.zip` is then unzipped over the result, so the clone's tree is a
    superset of both rather than whichever one happened to be installed last.
    """
    import shutil
    try:
        import libero as installed
    except ImportError:
        return {"copied": False, "why": "no `libero` package was installed"}
    src = pathlib.Path(installed.__file__).resolve().parent / "libero" / "assets"
    dst = root / "libero" / "libero" / "assets"
    if not src.is_dir():
        return {"copied": False, "why": f"{src} is not a directory"}
    before = sum(1 for _ in dst.rglob("*")) if dst.is_dir() else 0
    shutil.copytree(src, dst, dirs_exist_ok=True)
    after = sum(1 for _ in dst.rglob("*"))
    return {"copied": True, "from": str(src), "to": str(dst),
            "files_before": before, "files_after": after}


def install(work: pathlib.Path, *, token: Optional[str] = None,
            max_rounds: int = 15) -> Dict[str, object]:
    """LIBERO-plus over the DEPTHANCHOR environment, with the import chain walked for what it adds.

    Order matters and attempt 1 got it wrong: installing the clone editable while `hf-libero` still owns
    `site-packages/libero` leaves `import libero` resolving to the OLD package, and the import probe then
    passes while testing the wrong code. So: adopt the base assets, uninstall `hf-libero` (its
    dependencies — robosuite, bddl, mujoco — stay, which is the point of installing it first), install the
    clone, and put the clone at the head of `sys.path` for this process and of `PYTHONPATH` for children.
    """
    apt = install_system_packages()
    root = clone(work)
    adopted = adopt_base_assets(root)
    uninstalled = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "hf-libero"],
                                 capture_output=True, text=True)
    _run([sys.executable, "-m", "pip", "install", "-q", "-e", str(root), "--no-deps"])
    assets = fetch_assets(root, token=token)
    cfg = write_config(root)
    # children of this process must see the clone first too, or the probe tests the wrong package
    os.environ["PYTHONPATH"] = f"{root}:" + os.environ.get("PYTHONPATH", "")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    rounds: List[Dict[str, object]] = []
    # `libero` is a NAMESPACE package in this repository — the top-level directory has no `__init__.py`,
    # so `libero.__file__` is None and only `__path__` says where it came from
    probe_code = ("import libero, libero.libero.benchmark as b;"
                  "d=b.get_benchmark_dict();"
                  "print('IMPORT_OK', list(libero.__path__), len(d))")
    for _ in range(max_rounds):
        probe = subprocess.run([sys.executable, "-c", probe_code], capture_output=True, text=True)
        if probe.returncode == 0:
            rounds.append({"round": len(rounds), "result": probe.stdout.strip()})
            break
        m = re.search(r"No module named '([^']+)'", probe.stderr)
        if not m:
            raise RuntimeError(f"LIBERO-plus import failed for a reason that is not a missing module:\n"
                               f"{probe.stderr[-3000:]}")
        module = m.group(1)
        top = module.split(".")[0]
        if top in NEVER_INSTALL:
            raise RuntimeError(f"LIBERO-plus wants {module}, which would replace the stack the model "
                               f"runs on; refusing")
        package = PACKAGE_FOR_MODULE.get(module, PACKAGE_FOR_MODULE.get(top, top))
        rounds.append({"round": len(rounds), "missing": module, "installing": package})
        print(f"[lp-install] missing {module} -> pip install {package}", flush=True)
        _run([sys.executable, "-m", "pip", "install", "-q", package])
    else:
        raise RuntimeError(f"LIBERO-plus still not importable after {max_rounds} rounds: {rounds[-4:]}")
    import importlib
    for mod in [m for m in list(sys.modules) if m == "libero" or m.startswith("libero.")]:
        sys.modules.pop(mod, None)
    importlib.invalidate_caches()
    import libero                                     # noqa: F401  (the replaced package)
    libero_file = str(libero.__file__ or "") or str(list(getattr(libero, "__path__", []) or []))
    if str(root.resolve()) not in libero_file:
        raise RuntimeError(f"`libero` still resolves to {libero_file}, not to the LIBERO-plus clone; "
                           f"pip uninstall said: {uninstalled.stdout[-300:]}{uninstalled.stderr[-300:]}")
    probe_says = next((r.get("result") for r in reversed(rounds) if r.get("result")), "")
    if str(root.resolve()) not in str(probe_says):
        raise RuntimeError(f"the import probe passed against the wrong package: {probe_says}")
    return {"root": str(root), "sha": LP_SHA, "libero_file": libero_file,
            "apt": apt, "adopted_base_assets": adopted,
            "hf_libero_uninstalled": bool(uninstalled.returncode == 0),
            "assets": assets, "config": cfg, "install_rounds": rounds}


def enable_init_state_loading() -> Dict[str, object]:
    """Let LIBERO-plus load its own initial-state files under torch >= 2.6.

    `Benchmark.get_task_init_states` calls `torch.load(path)`, and since torch 2.6 that defaults to
    `weights_only=True`, which refuses the numpy globals inside those pickles:
    "Unsupported global: GLOBAL numpy.core.multiarray._reconstruct". The files are numpy arrays of joint
    positions from the repository this study cloned at a pinned commit.

    Two things are done, in this order. First the documented remedy: allowlist the numpy globals, which
    keeps `weights_only=True` semantics — no arbitrary code execution, just numpy arrays. Then, because a
    pickle may name a global the list misses, `get_task_init_states` itself is wrapped to retry with
    `weights_only=False`, which is scoped to that one method and to files from the pinned clone.
    """
    import numpy as np
    import torch
    allowed = []
    for name in ("core.multiarray._reconstruct", "ndarray", "dtype"):
        obj = np
        for part in name.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            allowed.append(obj)
    for extra in ("Float64DType", "Float32DType", "Int64DType", "UInt8DType", "BoolDType"):
        obj = getattr(getattr(np, "dtypes", None), extra, None)
        if obj is not None:
            allowed.append(obj)
    torch.serialization.add_safe_globals(allowed)

    from libero.libero import benchmark as lb
    original = lb.Benchmark.get_task_init_states
    if getattr(original, "_depthopen_wrapped", False):
        return {"allowlisted": [getattr(a, "__name__", str(a)) for a in allowed], "wrapped": "already"}

    def patched(self, i, _original=original):
        try:
            return _original(self, i)
        except Exception:
            loader = torch.load
            def trusting(*args, **kwargs):
                kwargs["weights_only"] = False
                return loader(*args, **kwargs)
            torch.load = trusting
            try:
                return _original(self, i)
            finally:
                torch.load = loader

    patched._depthopen_wrapped = True
    lb.Benchmark.get_task_init_states = patched
    return {"allowlisted": [getattr(a, "__name__", str(a)) for a in allowed],
            "wrapped": "libero.libero.benchmark.Benchmark.get_task_init_states"}


# ------------------------------------------------------------------ the task table
def classification(root: pathlib.Path) -> Dict[str, Dict[str, object]]:
    """(suite, task name) -> category and difficulty level, from the repository's own file."""
    payload = json.loads((root / CLASSIFICATION).read_text())
    out: Dict[str, Dict[str, object]] = {}
    for suite, rows in payload.items():
        for row in rows:
            out[f"{suite}/{row['name']}"] = {"suite": suite, "name": row["name"],
                                             "category": row.get("category"),
                                             "level": row.get("difficulty_level"),
                                             "id": row.get("id")}
    return out


def task_table(suites: Sequence[str], labels: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """Every task the installed benchmark exposes, with its index, joined to its label by name."""
    from libero.libero import benchmark as lb
    rows: List[Dict[str, object]] = []
    unlabelled: List[str] = []
    for suite in suites:
        bench = lb.get_benchmark_dict()[suite]()
        names = bench.get_task_names()
        for i, name in enumerate(names):
            label = labels.get(f"{suite}/{name}")
            if label is None:
                unlabelled.append(f"{suite}/{name}")
                continue
            rows.append({"suite": suite, "task_id": i, "name": name,
                         "category": label["category"], "level": label["level"],
                         "language": bench.get_task(i).language})
    return {"rows": rows, "n_tasks": len(rows), "n_unlabelled": len(unlabelled),
            "unlabelled_sample": unlabelled[:5],
            "by_category": _counts(r["category"] for r in rows),
            "by_level": _counts(r["level"] for r in rows)}


def _counts(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def sample_dimension(rows: Sequence[Dict[str, object]], *, category: str, n: int,
                     seed: int) -> Dict[str, object]:
    """A fixed stratified sample: proportional over the levels this category actually has, drawn once.

    Proportional rather than equal-per-level because the benchmark's levels are not equally populated,
    and an equal draw would over-weight the rarest level in the headline rate.
    """
    import numpy as np
    pool = [r for r in rows if r["category"] == category]
    if not pool:
        raise RuntimeError(f"no tasks in category {category!r}")
    levels = sorted({str(r["level"]) for r in pool})
    # stable per-category offset; `hash()` on a str is salted per process and would redraw the sample
    rng = np.random.default_rng([seed, _stable_key(category)])
    per_level = {}
    picked: List[Dict[str, object]] = []
    for level in levels:
        sel = [r for r in pool if str(r["level"]) == level]
        want = int(round(n * len(sel) / len(pool)))
        want = min(max(want, 1), len(sel))
        idx = rng.choice(len(sel), size=want, replace=False)
        take = [sel[int(i)] for i in sorted(idx)]
        per_level[level] = {"available": len(sel), "picked": len(take)}
        picked.extend(take)
    # rounding can leave the total a couple off n; trim or top up deterministically
    if len(picked) > n:
        drop = rng.choice(len(picked), size=len(picked) - n, replace=False)
        keep = [r for i, r in enumerate(picked) if i not in set(int(x) for x in drop)]
        picked = keep
    elif len(picked) < n:
        rest = [r for r in pool if r not in picked]
        add = rng.choice(len(rest), size=min(n - len(picked), len(rest)), replace=False)
        picked.extend(rest[int(i)] for i in sorted(add))
    picked = sorted(picked, key=lambda r: (r["suite"], r["task_id"]))
    return {"category": category, "n": len(picked), "seed": seed,
            "per_level": per_level, "by_suite": _counts(r["suite"] for r in picked),
            "by_level_final": _counts(r["level"] for r in picked), "tasks": picked}
