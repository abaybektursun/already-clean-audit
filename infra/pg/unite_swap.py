"""UNITE Arm 2 (studies/unite/PREREG.md §5, §6): swap weights between a parent and a child of one fine-tune hop and score absolute wikitext NLL.

Text-tower parameters only (blocks, embedding, final norm, head), by name in the loaded model. Windows and scoring are SINKPATCH's B3 calls (`sinkpatch.window_ids`,
`window_targets`, `forward_all`, `nll_of`), batch 1, scored windows 5..58. A neuron is (layer, i): MolmoAct2's `mlp.ff_proj` is cat[up; gate] (rows i and W + i) and `mlp.ff_out` holds
its output column i (`family_resolvers.qwen3_to_molmo2`). Fail loud; heavy imports inside functions.
"""
from __future__ import annotations

import numpy as np

from infra.pg import sinkpatch as SP

N_FIT = 5
K_FORWARD_LAST = (1, 3, 4, 6, 12, 24, 36)
K_FORWARD_FIRST = (1, 3, 6, 12, 24, 36)
K_REVERSE = (1, 3, 6, 12, 24, 36)
MASS_TOL = 0.05
PROJ, OUT = "mlp.ff_proj.weight", "mlp.ff_out.weight"


def text_params(member: dict) -> dict:
    """{name: parameter} of the text tower: blocks, embedding, final norm, head."""
    an = member["an"]
    roots = (an["layers_path"] + ".", an["norm_path"] + ".", an["embed_path"] + ".", an["lm_head_path"] + ".")
    out = {n: p for n, p in member["model"].named_parameters() if n.startswith(roots)}
    assert out, "no text parameters found"
    return out


def block(an: dict, i: int) -> str:
    return f"{an['layers_path']}.{i}."


def names_of(params: dict, an: dict, *, blocks=(), head=False, norm=False, embed=False) -> list:
    pre = tuple(block(an, i) for i in blocks)
    if head:
        pre += (an["lm_head_path"] + ".",)
    if norm:
        pre += (an["norm_path"] + ".",)
    if embed:
        pre += (an["embed_path"] + ".",)
    return [n for n in params if n.startswith(pre)] if pre else []


def forward_arms(params: dict, an: dict, n_layers: int) -> dict:
    """The 16 forward arms of one hop: last k, first k, last 4 plus head, head only, final norm only. Plus the gates' k = 0 and 'all text'."""
    arms = {f"last_{k}": names_of(params, an, blocks=range(n_layers - k, n_layers)) for k in K_FORWARD_LAST}
    arms |= {f"first_{k}": names_of(params, an, blocks=range(k)) for k in K_FORWARD_FIRST}
    arms["last_4_head"] = names_of(params, an, blocks=range(n_layers - 4, n_layers), head=True)
    arms["head"] = names_of(params, an, head=True)
    arms["final_norm"] = names_of(params, an, norm=True)
    return arms


def reverse_arms(params: dict, an: dict, n_layers: int) -> dict:
    arms = {f"rev_last_{k}": names_of(params, an, blocks=range(n_layers - k, n_layers)) for k in K_REVERSE}
    arms["rev_head"] = names_of(params, an, head=True)
    return arms


def copy_in(params: dict, source: dict, names: list) -> None:
    import torch
    with torch.no_grad():
        for n in names:
            assert params[n].shape == source[n].shape, (n, params[n].shape, source[n].shape)
            params[n].copy_(source[n].to(device=params[n].device, dtype=params[n].dtype))


def clone(params: dict) -> dict:
    return {n: p.detach().clone() for n, p in params.items()}


def mass(child: dict, parent: dict, names: list) -> float:
    return float(sum((child[n].float() - parent[n].float()).pow(2).sum() for n in names))


# ------------------------------------------------------------------------------------------ windows
def prepared_windows(member: dict, text: str, window: int = 2048, max_tokens: int = 120_000) -> list:
    tok, fwd = member["tok"], member["fwd"]
    wins = SP.window_ids(tok, text, window=window, max_tokens=max_tokens)
    assert len(wins) > N_FIT, len(wins)
    out = []
    for w, chunk in list(enumerate(wins))[N_FIT:]:
        win_text = tok.decode(chunk.tolist())
        enc, lo, hi, tg, off = SP.window_targets(fwd, win_text)
        out.append({"window": w, "enc": enc, "lo": lo, "hi": hi, "tg": tg, "text_tail": win_text[-300:], "last_gold": tok.decode([int(tg[-1])])})
    return out


def score(member: dict, wins: list) -> list:
    """Per window: [mean NLL, n, top id at the last scored row, its probability]."""
    rows = []
    for w in wins:
        logits = SP.forward_all(member["fwd"], w["enc"])
        loss, n, top, p = SP.nll_of(logits, w["lo"], w["hi"], w["tg"])
        rows.append([loss / n, n, top, p])
    return rows


def window_samples(rows: list, wins: list, tok) -> list:
    """The best-scored and worst-scored window of an arm: the text, the model's top token, the gold."""
    order = sorted(range(len(rows)), key=lambda i: rows[i][0])
    return [{"selection": tag, "window": wins[i]["window"], "prompt_tail": wins[i]["text_tail"], "nll_per_token": rows[i][0], "top_token": tok.decode([rows[i][2]]),
             "top_p": rows[i][3], "gold": wins[i]["last_gold"], "correct": tok.decode([rows[i][2]]) == wins[i]["last_gold"]} for i, tag in ((order[0], "best-scored window"), (order[-1], "worst-scored window"))]


def top1(member: dict, prompts: list, v_shared: int) -> list:
    return [(lambda lg: (int(lg[:v_shared].argmax()), float(lg[:v_shared].softmax(-1).max())))(SP.forward_last(member["fwd"], SP.encode1(member["fwd"], p))) for p in prompts]


# ------------------------------------------------------------------------------------------ the neuron basis
def neuron_mass(child: dict, parent: dict, an: dict, n_layers: int):
    """Per layer and neuron: ||d up||^2 + ||d gate||^2 (input) and ||d down||^2 (output), as NumPy arrays [L, W]."""
    inp, outp = [], []
    for l in range(n_layers):
        dp = (child[block(an, l) + PROJ].float() - parent[block(an, l) + PROJ].float())
        do = (child[block(an, l) + OUT].float() - parent[block(an, l) + OUT].float())
        W = dp.shape[0] // 2
        assert do.shape[1] == W, (dp.shape, do.shape)
        inp.append((dp[:W].pow(2).sum(1) + dp[W:].pow(2).sum(1)).cpu().numpy())
        outp.append(do.pow(2).sum(0).cpu().numpy())
    return np.stack(inp), np.stack(outp)


def mass_table(m_in, m_out, ks) -> dict:
    """Before any forward. For each k: the top k by total mass, and the smallest set of least-changed neurons whose mass is within 5% of it; none → reported and skipped."""
    total = m_in + m_out
    flat = total.reshape(-1)
    order = np.argsort(-flat, kind="stable")
    asc = order[::-1]
    cum = np.cumsum(flat[asc].astype(np.float64))
    out = {"n_neurons": int(flat.size), "total_mass": float(flat.sum(dtype=np.float64)), "k": {}}
    for k in ks:
        top = order[:k]
        m = float(flat[top].sum(dtype=np.float64))
        n_tail = int(np.searchsorted(cum, (1 - MASS_TOL) * m))
        ok = n_tail < flat.size and abs(cum[n_tail] - m) <= MASS_TOL * m and n_tail + 1 + k <= flat.size
        rec = {"k": k, "top_mass": m, "top_share_of_total": m / out["total_mass"], "tail_exists": bool(ok), "top": top, "tail": asc[:n_tail + 1] if ok else None}
        if ok:
            rec.update({"tail_n": n_tail + 1, "tail_mass": float(cum[n_tail])})
        out["k"][k] = rec
    return out


def restore_neurons(params: dict, source: dict, an: dict, flat_idx, W: int, side: str) -> None:
    """Copy the source's rows and/or columns for neurons in `flat_idx` (layer * W + i). side: input, output or both."""
    import torch
    assert side in ("input", "output", "both"), side
    by = {}
    for x in np.asarray(flat_idx):
        by.setdefault(int(x) // W, []).append(int(x) % W)
    with torch.no_grad():
        for l, ids in by.items():
            i = torch.tensor(ids, device=params[block(an, l) + PROJ].device)
            if side in ("input", "both"):
                r = torch.cat([i, i + W])
                params[block(an, l) + PROJ][r] = source[block(an, l) + PROJ][r].to(params[block(an, l) + PROJ].dtype)
            if side in ("output", "both"):
                params[block(an, l) + OUT][:, i] = source[block(an, l) + OUT][:, i].to(params[block(an, l) + OUT].dtype)


def sv_plan(child: dict, parent: dict, an: dict, n_layers: int) -> dict:
    """SVD of parent - child per MLP matrix; every singular component ranked by energy. A control restores the largest components up to a matched mass, on both sides."""
    import torch
    comps, svd = [], {}
    for l in range(n_layers):
        for name in (PROJ, OUT):
            key = block(an, l) + name
            u, s, vh = torch.linalg.svd(parent[key].float() - child[key].float(), full_matrices=False)
            svd[key] = (u.cpu(), s.cpu(), vh.cpu())         # host memory: resident models and clones already hold most of the GPU
            comps += [(float(x) ** 2, key, i) for i, x in enumerate(s.cpu().numpy())]
            del u, s, vh
    comps.sort(key=lambda c: -c[0])
    return {"svd": svd, "comps": comps}


def restore_sv(params: dict, plan: dict, target_mass: float) -> dict:
    """Add the largest singular components of (parent - child) until their energy reaches 95% of the target mass. Returns the realized mass and whether it lies within 5%."""
    import torch
    got, chosen = 0.0, {}
    for e, key, i in plan["comps"]:
        if got >= (1 - MASS_TOL) * target_mass:
            break
        chosen.setdefault(key, []).append(i)
        got += e
    with torch.no_grad():
        for key, ids in chosen.items():
            u, s, vh = plan["svd"][key]
            j = torch.tensor(ids)
            dev = params[key].device
            delta = (u[:, j].to(dev) * s[j].to(dev)) @ vh[j].to(dev)
            params[key].copy_((params[key].float() + delta).to(params[key].dtype))
    return {"realized_mass": got, "n_components": sum(map(len, chosen.values())), "matched": abs(got - target_mass) <= MASS_TOL * target_mass}


def mlp_names(params: dict) -> list:
    return [n for n in params if n.endswith((PROJ, OUT)) or ".ff_norm." in n]


def attn_names(params: dict) -> list:
    return [n for n in params if ".self_attn." in n or ".attn_norm." in n]
