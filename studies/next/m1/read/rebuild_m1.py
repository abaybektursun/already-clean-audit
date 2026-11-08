"""Rebuild of counts.npz from raw positions, three ways (READ_m1 task item 7), and the samples (item 8).

(a) the 640 grid poles of every harvest, from samples/contexts.jsonl (doc -> directory);
(b) two whole layers of M and G from the raw S3 topk files (tok -> doc via doc_offsets -> directory);
(c) the same comparison on planted wrong inputs, which must fail.
Written blind to infra/pg/m1.py. Numpy only.  Run: python -I rebuild_m1.py  ->  read/rebuild.json, read/samples.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/Users/abaybektursun/projects/interp_research-m1read/studies/next/m1/read')
from common import (BANKS, CM_DIRS, HARV, OFFSETS, OUT, S3DIR, doc_dirs, load_counts, manifest,  # noqa: E402
                    parse_pole)

m = manifest()
ddir = doc_dirs(m)
off = np.load(OFFSETS, allow_pickle=False)
assert off.shape == (m['n_docs'] + 1,) and off[0] == 0 and off[-1] == m['tokenizers']['olmo3']['n_tokens']
assert (np.bincount(ddir, weights=np.diff(off), minlength=50) == [s['olmo3_tokens_selected'] for s in m['shards']]).all()
assert off[m['bands']['2']['doc_end']] == m['bands']['2']['tok_end']['olmo3'] == 20570743
cm_set = set(m['dirs'].index(d) for d in CM_DIRS)

# band rule check: within each shard, band 1 docs start at running count < 20 % of the shard budget, band 2 < 40 %
budget = m['tokens_per_shard']
L = np.diff(off)
band_rule_ok = True
starts = {b: 0 for b in '123'}
pos = {b: m['bands'][b]['doc_start'] for b in '123'}
for i, s in enumerate(m['shards']):
    run = 0
    for b in '123':
        n = s['n_docs_selected'][b]
        docs = np.arange(pos[b], pos[b] + n)
        pos[b] += n
        st = run + np.concatenate([[0], np.cumsum(L[docs])[:-1]]) if n else np.array([])
        lim_lo = {'1': -1, '2': 0.2 * budget, '3': 0.4 * budget}[b]
        lim_hi = {'1': 0.2 * budget, '2': 0.4 * budget, '3': np.inf}[b]
        if n and not ((st >= lim_lo).all() and (st < lim_hi).all()):
            band_rule_ok = False
        run += L[docs].sum()
# token share per directory over the harvested prefix (bands 1-2)
tok_per_dir_prefix = np.bincount(ddir[:m['bands']['2']['doc_end']], weights=L[:m['bands']['2']['doc_end']], minlength=50)


def tok_to_dir(tok, side='right', dshift=0, dirshift=0):
    d = np.searchsorted(off, tok, side=side) - 1 + dshift
    d = np.clip(d, 0, len(ddir) - 1)
    return (ddir[d] + dirshift) % 50


def counts_from_dirs(dirs, valid):
    """dirs, valid: [..., 50] -> counts [..., 50 directories]"""
    flat = dirs.reshape(-1, dirs.shape[-1])
    v = valid.reshape(-1, dirs.shape[-1])
    out = np.zeros((flat.shape[0], 50), dtype=np.int64)
    rows = np.repeat(np.arange(flat.shape[0]), flat.shape[1])
    np.add.at(out, (rows[v.ravel()], flat.ravel()[v.ravel()]), 1)
    return out.reshape(dirs.shape[:-1] + (50,))


res = {'mapping': {'tokens_per_dir_equal_manifest': True, 'band_rule_ok': band_rule_ok,
                   'tok_per_dir_prefix': tok_per_dir_prefix.astype(int).tolist(),
                   'cm_token_share_prefix': float(tok_per_dir_prefix[list(cm_set)].sum() / tok_per_dir_prefix.sum())}}

# ---- (a) grid poles from contexts.jsonl ----
res['a'] = {}
ctx = {}
for k in HARV:
    c, f = load_counts(k, m)
    rows = [json.loads(line) for line in open(f'{BANKS}/{k}/samples/contexts.jsonl')]
    ctx[k] = rows
    n_eq = n_doc_ok = n_slots = 0
    plant = {'dir+1': 0, 'doc+1': 0, 'sign_swap': 0}
    for r in rows:
        l, n, s = parse_pole(r['pole'])
        toks = np.array([e['tok'] for e in r['topk']])
        docs = np.array([e['doc'] for e in r['topk']])
        n_slots += len(toks)
        n_doc_ok += int((np.searchsorted(off, toks, side='right') - 1 == docs).sum())
        rb = np.bincount(ddir[docs], minlength=50)
        n_eq += int((rb == c[l, n, s]).all() and len(toks) == f[l, n, s])
        plant['dir+1'] += int((np.bincount((ddir[docs] + 1) % 50, minlength=50) == c[l, n, s]).all())
        plant['doc+1'] += int((np.bincount(ddir[np.minimum(docs + 1, len(ddir) - 1)], minlength=50) == c[l, n, s]).all())
        plant['sign_swap'] += int((rb == c[l, n, 1 - s]).all())
    res['a'][k] = {'n_poles': len(rows), 'n_poles_equal': n_eq, 'n_slots': n_slots, 'n_doc_field_equal_offsets': n_doc_ok,
                   'planted_n_poles_equal': plant}
    print('a', k, res['a'][k])

# ---- (b) two whole layers of M and G from the raw topk ----
layers = sorted(np.random.default_rng(0).choice(32, size=2, replace=False).tolist())
res['b'] = {'layers': layers, 'rule': 'sorted(np.random.default_rng(0).choice(32, size=2, replace=False))'}
for k in ('M', 'G'):
    c, f = load_counts(k, m)
    for Lr in layers:
        z = np.load(f'{S3DIR}/{k}/layer_{Lr:02d}.npz', allow_pickle=False)
        tok = z['tok']
        assert tok.shape == (11008, 2, 50)
        valid = tok >= 0
        assert tok[valid].max() < 20570743
        t = np.where(valid, tok, 0)
        cb = counts_from_dirs(tok_to_dir(t), valid)
        filled_eq = bool((valid.sum(axis=2) == f[Lr]).all())
        eq = (cb == c[Lr]).all(axis=2)
        planted = {}
        for name, kw in {'dir+1': dict(dirshift=1), 'doc+1': dict(dshift=1), 'doc_left_boundary': dict(side='left')}.items():
            planted[name] = int((counts_from_dirs(tok_to_dir(t, **kw), valid) == c[Lr]).all(axis=2).sum())
        planted['sign_swap'] = int((cb[:, ::-1] == c[Lr]).all(axis=2).sum())
        # the contexts' top-50 for grid poles in this layer equal the raw tok set
        ctx_eq = ctx_n = 0
        for r in ctx[k]:
            l, n, s = parse_pole(r['pole'])
            if l == Lr:
                ctx_n += 1
                ctx_eq += int(sorted(e['tok'] for e in r['topk']) == sorted(tok[n, s][valid[n, s]].tolist()))
        # within-pole duplicate positions (a slot counted twice)
        dup = int(sum(len(set(tok[n, s][valid[n, s]].tolist())) != valid[n, s].sum() for n in range(11008) for s in range(2)))
        res['b'][f'{k}_L{Lr}'] = {'n_poles': int(eq.size), 'n_poles_equal': int(eq.sum()), 'filled_equal': filled_eq,
                                  'n_empty_slots': int((~valid).sum()), 'planted_n_poles_equal': planted,
                                  'grid_poles_in_layer': ctx_n, 'grid_poles_contexts_equal_raw': ctx_eq,
                                  'n_poles_with_duplicate_positions': dup}
        print('b', k, Lr, res['b'][f'{k}_L{Lr}'])

json.dump(res, open(f'{OUT}/rebuild.json', 'w'), indent=1)

# ---- samples: one slot counted as code+math and one counted as other, M and G ----
rs = np.random.default_rng(7)
samples = []
for k in ('M', 'G'):
    slots = [(r['pole'], e) for r in ctx[k] for e in r['topk']]
    cmk = [x for x in slots if ddir[x[1]['doc']] in cm_set]
    oth = [x for x in slots if ddir[x[1]['doc']] not in cm_set]
    for lab, pool in (('code+math', cmk), ('other', oth)):
        for j in rs.choice(len(pool), size=3, replace=False):
            p, e = pool[int(j)]
            samples.append({'harvest': k, 'counted_as': lab, 'pole': p, 'rank': e['rank'], 'tok': e['tok'], 'doc': e['doc'],
                            'dir': m['dirs'][ddir[e['doc']]], 'a': e['a'], 'before': e['before'], 'token': e['token'],
                            'after': e['after']})
json.dump(samples, open(f'{OUT}/samples.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(res['mapping'])[:300])
