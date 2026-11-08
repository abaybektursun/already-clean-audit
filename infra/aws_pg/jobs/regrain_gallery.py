"""REGRAIN F-C4 + F-E2 (GPU) — the qualitative galleries, each paired with its matched null.

The paper's two-resolution reading (§3.5): a unit carries a concept's PRESENCE in its code norm and
the concept's INTERNAL GEOMETRY in the principal coordinates of its contributions. Rendered on PG1's
own stream at L0, where the image tokens are a 16×16 patch grid:

  F-C4  presence:  per-patch |z_j| for each top handle atom, over N VQA images  (DESCRIPTIVE)
  F-E2  geometry:  per-patch hue = the first three contribution-PCA coordinates of the B*-union,
                   mapped to RGB — the paper's Fig. 9 grammar                    (DESCRIPTIVE)

Each geometry panel ships with its **matched-noise null** (App. E design): the SAME trained
featurizer run on coordinate-shuffled activations, which preserve every coordinate's marginal and
destroy all cross-coordinate structure. A gallery is never evidence alone; these are labelled
DESCRIPTIVE and are cited as such.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_gallery --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/regrain_gallery --budget-min 90 \
      --args "--cdisc-s3 s3://…/regrain_c_disc/out"
"""
import argparse
import json
import os
import pathlib

N_IMAGES = 10
N_ATOMS = 8
GRID = 16               # PG1 pt-224: 224/14 = 16 → 256 image tokens


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cdisc-s3", required=True)
    ap.add_argument("--n-images", type=int, default=N_IMAGES)
    ap.add_argument("--n-atoms", type=int, default=N_ATOMS)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.aws_pg.jobs._inherit_shared import vqa_forward_batches
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC

    RS.install_pg()
    cd = json.loads(RS.fetch(f"{a.cdisc_s3}/c_disc.json", "/tmp/regrain/c_disc.json").read_text())
    bstar = [int(x) for x in cd["bstar"]]
    med_s, L = int(cd["median_seed"]), int(cd["L"])
    singles = cd["per_seed"][str(med_s)]["singles"]
    top_atoms = [int(x) for x in sorted(singles, key=lambda k: -singles[k])][: a.n_atoms]
    rep = {"object": "regrain_gallery_v1", "label": "DESCRIPTIVE", "L": L,
           "bstar_len": len(bstar), "top_atoms": top_atoms, "grid": GRID}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    nm = RS.dict_bank_name("primary", 1, med_s)
    src = f"{RS.GRAIN_DICTS_L0}/{nm}" if med_s == 0 else f"{a.cdisc_s3}/{nm}"
    ae, _c, _q, _m = GB.load_dict_bank(RS.fetch(src, f"/tmp/regrain/{nm}"))
    ae = ae.to("cuda").eval()
    Db = RC.decoder_matrix(ae)[:, bstar].float().to("cuda")

    sha, _ = RS.pinned_pg1_sha("/tmp/regrain")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    vqa, _coco = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json",
                             "/tmp/regrain/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift — halt"
    split = RS.grain_split(hashes, name="grain_vqa")
    # VQA content hashes include the QUESTION, so consecutive discovery items often share an image.
    # Dedupe on the resized image's bytes so the gallery shows n_images DISTINCT images.
    import hashlib as _hl
    items, seen = [], set()
    for i in split["disc_idx"]:
        im = vqa.image_fn(rows[i]).convert("RGB").resize((224, 224))
        h = _hl.sha1(im.tobytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        items.append(rows[i])
        if len(items) >= a.n_images:
            break
    assert len(items) == a.n_images, f"only {len(items)} distinct images in the discovery half"
    rep["n_distinct_images"] = len(items)
    image_id = model.config.image_token_index

    # capture the entering-L0 residual, keep the image-token rows, per image
    store, labels = [], []
    h = layers[L].register_forward_pre_hook(RS.make_row_capture(store, IP.D), with_kwargs=True)
    with torch.no_grad():
        for inp in vqa_forward_batches(proc, vqa, items, "cuda", batch_size=1):
            model(**inp)
            labels.append((inp["input_ids"].reshape(-1) == image_id).cpu())
    h.remove()
    assert len(store) == len(items), f"{len(store)} captures vs {len(items)} images"

    patches, imgs = [], []
    for i, (rowsi, mask) in enumerate(zip(store, labels)):
        sel = rowsi[mask]
        assert sel.shape[0] == GRID * GRID, f"image {i}: {sel.shape[0]} image tokens != {GRID**2}"
        patches.append(sel.float())
        imgs.append(vqa.image_fn(items[i]).convert("RGB").resize((224, 224)))
    Xp = torch.cat(patches, 0).to("cuda")                       # (n_img·256, d)

    with torch.inference_mode():
        f = ae.encode(Xp, use_threshold=True)                   # presence
        contrib = f[:, bstar] @ Db.T                            # (n, d) B*-union contributions
        C = contrib.double()
        C = C - C.mean(0)
        _U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        coords = (C @ Vh[:3].T).cpu().numpy()                   # (n, 3) top-3 PCA coordinates
        # matched-noise null: the SAME featurizer on coordinate-shuffled rows
        Xn = RC.shuffled_null_rows(Xp.cpu(), seed=77).to("cuda")
        fn = ae.encode(Xn, use_threshold=True)
        cn = (fn[:, bstar] @ Db.T).double()
        cn = cn - cn.mean(0)
        Sn = torch.linalg.svdvals(cn)

    ev = (S ** 2 / max(1, C.shape[0])).cpu()
    evn = (Sn ** 2 / max(1, cn.shape[0])).cpu()
    rep["contribution_pca"] = {
        "rank_triplet": RC.rank_triplet(ev), "matched_noise_null_rank_triplet": RC.rank_triplet(evn),
        "top3_variance_share": float(ev[:3].sum() / ev.sum()),
        "top3_variance_share_null": float(evn[:3].sum() / evn.sum())}
    _w()

    # ---- Is a handle atom a POSITION detector or a CONTENT detector? ----
    # A[i, p] = |z_j| for image i at patch p. Decompose the total variance of A into the part
    # explained by the patch-position mean (position_share) and by the per-image mean
    # (image_share). A pure positional atom fires the same way on every image: position_share → 1.
    # Also report the mean pairwise Pearson r between images' 256-vectors.
    fa_g = f.abs().cpu().numpy()
    n_img = len(items)
    pos = {}
    for at in top_atoms:
        A = fa_g[:, at].reshape(n_img, GRID * GRID)
        tot = float(A.var())
        if tot <= 0:
            continue
        pos_share = float(A.mean(0).var() / tot)
        img_share = float(A.mean(1).var() / tot)
        Ac = A - A.mean(1, keepdims=True)
        sd = Ac.std(1, keepdims=True)
        Z = Ac / np.where(sd > 0, sd, 1.0)
        R = (Z @ Z.T) / (GRID * GRID)
        iu = np.triu_indices(n_img, 1)
        pos[str(at)] = {"position_share": pos_share, "image_share": img_share,
                        "mean_pairwise_r_across_images": float(R[iu].mean()),
                        "single_atom_effect": singles[str(at)]}
    rep["positional_vs_content"] = {
        "n_images": n_img, "per_atom": pos,
        "reading": ("position_share = Var_p(mean_i |z|) / Var(|z|): the fraction of an atom's "
                    "firing variance explained by patch POSITION alone. A content detector fires "
                    "differently on different images (low position_share, low cross-image r); a "
                    "positional atom fires the same way on every image (both -> 1).")}
    _w()

    fa = f.cpu().numpy()
    n_img = len(items)

    # ---------------- F-C4: presence heatmaps, one row per atom ----------------
    fig, axs = plt.subplots(a.n_atoms + 1, n_img, figsize=(1.5 * n_img, 1.5 * (a.n_atoms + 1)))
    for j in range(n_img):
        axs[0, j].imshow(imgs[j])
        axs[0, j].set_xticks([]); axs[0, j].set_yticks([])
        if j == 0:
            axs[0, j].set_ylabel("image", fontsize=6)
    for r, at in enumerate(top_atoms, start=1):
        v = fa[:, at].reshape(n_img, GRID, GRID)
        vmax = float(np.abs(v).max()) or 1.0
        for j in range(n_img):
            axs[r, j].imshow(imgs[j], alpha=0.55)
            axs[r, j].imshow(v[j], cmap="inferno", alpha=0.65, vmin=0, vmax=vmax,
                             extent=(0, 224, 224, 0))
            axs[r, j].set_xticks([]); axs[r, j].set_yticks([])
        axs[r, 0].set_ylabel(f"atom {at}\neff {singles[str(at)]:+.3f}", fontsize=5.5)
    fig.suptitle("F-C4 · presence: per-patch |z_j| of the top handle atoms @ L0  (DESCRIPTIVE)",
                 fontsize=9)
    fig.text(0.005, 0.002, f"regrain_gallery · L{L} · |B*|={len(bstar)} · DESCRIPTIVE",
             fontsize=5, color="#999999")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"regrain_F_C4_presence_gallery.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------------- F-E2: two-resolution — hue = contribution-PCA coordinates ----------------
    c = coords.reshape(n_img, GRID, GRID, 3)
    lo, hi = np.percentile(coords, 2, axis=0), np.percentile(coords, 98, axis=0)
    rgb = np.clip((c - lo) / np.maximum(hi - lo, 1e-9), 0, 1)
    norm = np.linalg.norm(fa[:, bstar], axis=1).reshape(n_img, GRID, GRID)
    fig, axs = plt.subplots(3, n_img, figsize=(1.6 * n_img, 5.2))
    for j in range(n_img):
        axs[0, j].imshow(imgs[j])
        axs[1, j].imshow(norm[j], cmap="magma")
        axs[2, j].imshow(rgb[j])
        for r in range(3):
            axs[r, j].set_xticks([]); axs[r, j].set_yticks([])
    axs[0, 0].set_ylabel("image", fontsize=7)
    axs[1, 0].set_ylabel("presence\n‖z_B*‖", fontsize=7)
    axs[2, 0].set_ylabel("geometry\nPCA₁₋₃ → RGB", fontsize=7)
    t = rep["contribution_pca"]
    fig.suptitle("F-E2 · two-resolution reading of the B*-union  ·  top-3 PCA variance share "
                 "%.3f vs matched-noise null %.3f  (DESCRIPTIVE)"
                 % (t["top3_variance_share"], t["top3_variance_share_null"]), fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"regrain_F_E2_two_resolution.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    print("REGRAIN_GALLERY_OK", json.dumps(rep["contribution_pca"], default=str))
    # green written LAST: a crash after this point must not leave a green
    # artifact -- the drivers gate on `green` (regrain_drive*.sh wait_for_green).
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
