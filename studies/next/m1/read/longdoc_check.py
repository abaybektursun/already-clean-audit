"""The long FineMath document: how much of Delta_MG and of PC_NLL rests on it, and the per-directory split of Delta_MG.
Run: python -I longdoc_check.py  ->  read/longdoc.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/Users/abaybektursun/projects/interp_research-m1read/studies/next/m1/read')
from common import BANKS, CM_DIRS, OFFSETS, OUT, S3DIR, doc_dirs, manifest  # noqa: E402

m = manifest()
ddir = doc_dirs(m)
off = np.load(OFFSETS, allow_pickle=False)
L = np.diff(off)
fm = m['dirs'].index('data/finemath-3plus')
docs_fm = np.flatnonzero(ddir[:m['bands']['2']['doc_end']] == fm)
big = int(docs_fm[np.argmax(L[docs_fm])])
out = {'long_doc': big, 'long_doc_tokens': int(L[big]), 'long_doc_tok_range': [int(off[big]), int(off[big + 1])],
       'finemath_tokens_bands12_without_long_doc': int(L[docs_fm].sum() - L[big])}
R = json.load(open(f'{OUT}/read_numbers.json'))
out['dir_delta_MG_cm_dirs'] = {d: R['specific']['dir_delta_MG'][d] for d in CM_DIRS}
cm_idx = [m['dirs'].index(d) for d in CM_DIRS]
# layers 20 and 26 from the raw topk: share of slots in the long doc, M and G, and Delta_MG with / without it
for Lr in (20, 26):
    r = {}
    for k in ('M', 'G'):
        tok = np.load(f'{S3DIR}/{k}/layer_{Lr:02d}.npz', allow_pickle=False)['tok']
        doc = np.searchsorted(off, tok, side='right') - 1
        cmslot = np.isin(ddir[doc], cm_idx)
        inbig = doc == big
        r[k] = {'cm_share': float(cmslot.mean()), 'long_doc_share': float(inbig.mean()),
                'cm_share_without_long_doc': float((cmslot & ~inbig).mean()),
                'share_poles_with_any_slot_in_long_doc': float(inbig.any(axis=2).mean())}
    r['Delta_MG'] = r['M']['cm_share'] - r['G']['cm_share']
    r['Delta_MG_without_long_doc'] = r['M']['cm_share_without_long_doc'] - r['G']['cm_share_without_long_doc']
    r['Delta_MG_long_doc_only'] = r['M']['long_doc_share'] - r['G']['long_doc_share']
    out[f'layer_{Lr}'] = r
# grid-pole slots in the long doc, with text (M)
rows = [json.loads(x) for x in open(f'{BANKS}/M/samples/contexts.jsonl')]
hits = [(r['pole'], e) for r in rows for e in r['topk'] if e['doc'] == big]
out['M_grid_slots_in_long_doc'] = len(hits)
out['M_grid_slot_examples_in_long_doc'] = [{'pole': p, 'rank': e['rank'], 'tok': e['tok'], 'before': e['before'][-160:],
                                            'token': e['token'], 'after': e['after'][:80]} for p, e in hits[:2]]
# PC_NLL: windows by document, and a document-cluster bootstrap (cluster = the document holding the window's start)
nll = {k: json.load(open(f'{BANKS}/{k}/m1/metrics.json'))['pc_nll'] for k in ('M', 'G')}
win = nll['M']['windows']
dc = np.array(nll['M']['nll']['cm']) - np.array(nll['G']['nll']['cm'])
dw = np.array(nll['M']['nll']['web']) - np.array(nll['G']['nll']['web'])
start_doc = {g: np.searchsorted(off, np.array(win[g]) * 2048, side='right') - 1 for g in ('cm', 'web')}
in_big = np.array([(w * 2048 < off[big + 1]) and ((w + 1) * 2048 > off[big]) for w in win['cm']])
out['pc_cm_windows_in_long_doc'] = int(in_big.sum())
out['pc_d_cm_without_long_doc_windows'] = float(dc[~in_big].mean())
cl = {g: {d: np.flatnonzero(start_doc[g] == d) for d in np.unique(start_doc[g])} for g in ('cm', 'web')}
out['pc_n_clusters'] = {g: len(cl[g]) for g in cl}
rng = np.random.default_rng(5)
bc, bd = [], []
kc, kw = list(cl['cm'].values()), list(cl['web'].values())
for _ in range(2000):
    ic = np.concatenate([kc[i] for i in rng.integers(0, len(kc), len(kc))])
    iw = np.concatenate([kw[i] for i in rng.integers(0, len(kw), len(kw))])
    bc.append(dc[ic].mean())
    bd.append(dc[ic].mean() - dw[iw].mean())
out['pc_cluster_boot'] = {'d_cm_ci': np.percentile(bc, [2.5, 97.5]).tolist(), 'd_dd_ci': np.percentile(bd, [2.5, 97.5]).tolist()}
json.dump(out, open(f'{OUT}/longdoc.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
