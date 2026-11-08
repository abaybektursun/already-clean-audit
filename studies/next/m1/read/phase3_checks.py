"""Phase 3 of the independent read: does the frozen code do what the spec says, and do its recorded gates hold?

1. each harvest's recorded registry copy (`same`, `nll_band`) against infra/pg/m1.py SAME and NLL_BAND at HEAD;
2. every recorded instance-gate failing input (G-SHARD, G-CFG, G-CAST, G0) and G-FIT on its failing inputs;
3. the PC_NLL windows rebuilt with my own candidate rule and the spec's draw (default_rng(0)), and how many
   distinct documents the 32 + 32 windows come from;
4. the verdict's bootstrap intervals rebuilt with the code's RNG scheme (written here, not imported);
5. token_share_per_dir against my own count.
Run: python -I phase3_checks.py  ->  read/phase3.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/Users/abaybektursun/projects/interp_research-m1read/studies/next/m1/read')
sys.path.insert(0, '/Users/abaybektursun/projects/interp_research-m1read')
from common import BANDS, BANKS, CM_DIRS, HARV, OFFSETS, OUT, WT, doc_dirs, load_counts, manifest  # noqa: E402
from infra.pg import m1 as M1  # noqa: E402  (registry constants only)

m = manifest()
ddir = doc_dirs(m)
off = np.load(OFFSETS, allow_pickle=False)
V = json.load(open(f'{WT}/studies/next/m1/verdict_m1.json'))
res = {}

# 1. registry copies
res['registry'] = {}
for k in HARV:
    rec = json.load(open(f'{BANKS}/{k}/m1/metrics.json'))
    res['registry'][k] = {'same_equals_m1_SAME': rec['same'] == M1.SAME,
                          'nll_band_equals': tuple(rec['nll_band']) == tuple(M1.NLL_BAND)}

# 2. recorded gates and failing inputs
res['gates'] = {}
for k in HARV:
    rec = json.load(open(f'{BANKS}/{k}/m1/metrics.json'))
    met = json.load(open(f'{BANKS}/{k}/metrics.json'))
    ctl = json.load(open(f'{BANKS}/{k}/controls.json'))
    man = json.load(open(f'{BANKS}/{k}/manifest.json'))
    g = {'gates_pass': rec['gates_pass'],
         'G_SHARD': [rec['download']['G_SHARD']['pass'], rec['download']['G_SHARD']['failing_input']['pass']],
         'G_REV_commit_equals_registered': rec['download']['commit'] == M1.BRANCHES[rec['branch']]['commit'],
         'G_CFG': [rec['config']['G_CFG']['pass'], rec['config']['failing_input']['pass']],
         'G_CAST': [rec['G_CAST']['pass'], rec['G_CAST']['failing_input']['pass']],
         'G0': [ctl['G0']['pass'], ctl['G0']['failing_input']['pass'], ctl['G0']['failing_input_2']['pass']],
         'device': man['device_name']}
    # G-FIT, my own rule, on the real record and on the registered failing inputs
    nll = ctl['nll_corpus_first_120k']['mean_nll']

    def fit(met_, nll_, man_):
        same_diff = [s for s in M1.SAME if man_.get('max_tokens_arg' if s == 'max_tokens' else s) != M1.SAME[s]]
        ok = (met_['complete'] is True and met_['G0_pass'] is True and met_['dead_poles'] == 0
              and met_['n_tokens_scored'] == M1.SAME['n_tokens_scored'] and 2.40 <= nll_ <= 3.20 and not same_diff)
        return bool(ok), same_diff
    g['G_FIT'] = fit(met, nll, man)[0]
    g['G_FIT_nll'] = nll
    g['G_FIT_fail_complete_false'] = fit(dict(met, complete=False), nll, man)[0]
    g['G_FIT_fail_nll_12.223'] = fit(met, 12.223, man)[0]
    g['G_FIT_fail_batch_16'] = fit(met, nll, dict(man, batch=16))[0]
    res['gates'][k] = g

# 3. PC windows rebuilt
W, NW = 2048, 10044
grp = {d: ('cm' if d in CM_DIRS else ('web' if d.startswith('data/common_crawl-') else 'other')) for d in m['dirs']}
gdoc = np.array([grp[m['dirs'][i]] for i in ddir])
first = np.searchsorted(off, np.arange(NW) * W, side='right') - 1
last = np.searchsorted(off, np.arange(NW) * W + W - 1, side='right') - 1
cand = {'cm': [], 'web': []}
for w in range(NW):
    gs = set(gdoc[first[w]:last[w] + 1].tolist())
    if len(gs) == 1 and next(iter(gs)) in cand:
        cand[next(iter(gs))].append(w)
rec = json.load(open(f'{BANKS}/M/m1/metrics.json'))['pc_nll']['windows']
pcw = {}
for g in ('cm', 'web'):
    drawn = sorted(int(x) for x in np.random.default_rng(0).choice(cand[g], size=32, replace=False))
    docs = sorted(set(int(d) for w in rec[g] for d in range(first[w], last[w] + 1)))
    pcw[g] = {'n_candidates_mine': len(cand[g]), 'n_candidates_recorded': rec[f'n_candidates_{g}'],
              'draw_equals_recorded': drawn == rec[g],
              'n_distinct_docs_in_32_windows': len(docs),
              'n_distinct_dirs': len(set(int(ddir[d]) for d in docs)),
              'dirs': sorted(set(m['dirs'][int(ddir[d])] for d in docs)),
              'windows_per_doc_max': int(max(sum(1 for w in rec[g] if first[w] <= d <= last[w]) for d in docs))}
res['pc_windows'] = pcw
# windows whose NLL is identical in a harvest
nllM = json.load(open(f'{BANKS}/M/m1/metrics.json'))['pc_nll']['nll']['cm']
vals, cts = np.unique(np.round(nllM, 9), return_counts=True)
res['pc_windows']['cm_windows_with_repeated_nll_M'] = [
    {'nll': float(v), 'windows': [rec['cm'][i] for i in range(32) if round(nllM[i], 9) == v],
     'docs': sorted(set(int(d) for i in range(32) if round(nllM[i], 9) == v for d in range(first[rec['cm'][i]], last[rec['cm'][i]] + 1)))}
    for v, c in zip(vals, cts) if c > 1]
nll_all = {k: json.load(open(f'{BANKS}/{k}/m1/metrics.json'))['pc_nll']['nll'] for k in HARV}
res['pc_windows']['per_window_cm_M_minus_G'] = (np.array(nll_all['M']['cm']) - np.array(nll_all['G']['cm'])).tolist()
res['pc_windows']['per_window_web_M_minus_G'] = (np.array(nll_all['M']['web']) - np.array(nll_all['G']['web'])).tolist()


# 4. the verdict's intervals with the code's RNG scheme: a fresh default_rng(0) per (contrast, band) call,
#    one integers(0, n, n) draw per resample, np.quantile linear
def boot_code(v):
    rng = np.random.default_rng(0)
    b = np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(2000)])
    lo, hi = np.quantile(b, [0.025, 0.975])
    return float(v.mean()), float(lo), float(hi)


cm_idx = np.array([m['dirs'].index(d) for d in CM_DIRS])
sh = {}
for k in ('G', 'R', 'M'):
    c, f = load_counts(k, m)
    sh[k] = c[..., cm_idx].sum(axis=3, dtype=np.int64) / f.astype(np.float64)
rows = []
for name, (x, y), vfield in (('MG', ('M', 'G'), None), ('RG', ('R', 'G'), 'R_minus_G'), ('MR', ('M', 'R'), 'M_minus_R')):
    for bi, (a, b) in enumerate(BANDS):
        mine = boot_code((sh[x][a:b] - sh[y][a:b]).mean(axis=2).ravel())
        theirs = ([V['lead'][bi]['mean'], V['lead'][bi]['lo'], V['lead'][bi]['hi']] if vfield is None
                  else V['between'][bi][vfield])
        rows.append({'contrast': name, 'band': bi, 'mine_code_rng': mine, 'verdict': theirs,
                     'max_abs_diff': float(max(abs(p - q) for p, q in zip(mine, theirs)))})
res['bootstrap_code_rng'] = rows
# PC_NLL with the code's RNG scheme
mc = np.array(nll_all['M']['cm']) - np.array(nll_all['G']['cm'])
mw = np.array(nll_all['M']['web']) - np.array(nll_all['G']['web'])
rng = np.random.default_rng(0)
bc, bd = [], []
for _ in range(2000):
    i, j = rng.integers(0, 32, 32), rng.integers(0, 32, 32)
    bc.append(mc[i].mean())
    bd.append(mc[i].mean() - mw[j].mean())
res['pc_code_rng'] = {'cm': [float(mc.mean())] + np.quantile(bc, [0.025, 0.975]).tolist(),
                      'dd': [float(mc.mean() - mw.mean())] + np.quantile(bd, [0.025, 0.975]).tolist(),
                      'verdict_cm': V['PC_NLL']['cm_M_minus_G'], 'verdict_dd': V['PC_NLL']['cm_minus_web']}

# 5. token share per directory over the windowed range [0, 20,570,112)
nt = 10044 * 2048
lens = np.diff(np.minimum(off, nt))
mine_tok = np.bincount(ddir, weights=lens, minlength=50).astype(np.int64)
rec_tok = np.array(json.load(open(f'{BANKS}/M/m1/metrics.json'))['token_share_per_dir'])
res['token_share_per_dir'] = {'equal': bool((mine_tok == rec_tok).all()), 'sum': int(mine_tok.sum()),
                              'cm_share': float(mine_tok[cm_idx].sum() / mine_tok.sum()),
                              'min': int(mine_tok.min()), 'max': int(mine_tok.max())}

# t15 poles with fewer than 50 filled slots (the tau pair t14c-t15)
c15, f15 = load_counts('t15', m)
res['t15_poles_filled_lt_50'] = int((f15 < 50).sum())
json.dump(res, open(f'{OUT}/phase3.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('bootstrap_code_rng', 'pc_windows')}, indent=0)[:3000])
print(json.dumps({g: {kk: vv for kk, vv in res['pc_windows'][g].items()} for g in ('cm', 'web')}))
print(res['pc_windows']['cm_windows_with_repeated_nll_M'])
print('max boot diff (code rng)', max(r['max_abs_diff'] for r in rows))
