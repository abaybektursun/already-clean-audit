"""Phase 1 of the independent read of M1: every headline number from the banked count files.

Written blind: without infra/pg/m1.py, the job files, verdict_m1.json, JOURNAL.md or raw_examples_m1.md.
Definitions are from studies/next/07_PICKS/M1_SPEC.md §0.5, §1.3, §3, §5. Numpy only.
Run: python -I read_m1.py  ->  read/read_numbers.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/Users/abaybektursun/projects/interp_research-m1read/studies/next/m1/read')
from common import (BANDS, BANKS, CM_DIRS, HARV, OUT, REFS, load_counts, manifest, parse_pole)  # noqa: E402

NB = 2000          # bootstrap resamples (spec §0.5)
SEED = 20261105    # my own seed; the spec's default_rng(0) is not reproduced on purpose
NSETS = 1000       # random directory sets (spec §3.6)

m = manifest()
cm_idx = np.array([m['dirs'].index(d) for d in CM_DIRS])
non_cm = np.array([i for i in range(50) if i not in set(cm_idx.tolist())])
assert len(non_cm) == 45

keys = HARV + REFS
cnt, share = {}, {}
for k in keys:
    c, f = load_counts(k, m)
    cnt[k] = (c, f)
    share[k] = c[..., cm_idx].sum(axis=3, dtype=np.int64) / f.astype(np.float64)   # [32, 11008, 2]


def C(k):
    return [float(share[k][a:b].mean()) for a, b in BANDS]


Cs = {k: C(k) for k in keys}

# ---- bootstrap over neurons, the two poles of a neuron resampled together ----
rng = np.random.default_rng(SEED)
CONTRASTS = {'MG': ('M', 'G'), 'RG': ('R', 'G'), 'MR': ('M', 'R'), 'I': ('start_b', 'start_a'),
             'G_minus_start': ('G', 'start_a'), 'R_minus_start': ('R', 'start_a'),
             'M_minus_start': ('M', 'start_a'), 'Gid_fail_t15_minus_start_a': ('t15', 'start_a')}


def boot_band(pairs, a, b):
    """pairs: list of (X, Y). Returns point means and 95 % percentile intervals, band [a, b)."""
    X = np.stack([(share[x][a:b] - share[y][a:b]).mean(axis=2).ravel() for x, y in pairs])   # [K, neurons]
    N = X.shape[1]
    means = []
    for _ in range(NB // 20):
        idx = rng.integers(0, N, size=(20, N))
        means.append(X[:, idx].mean(axis=2))
    means = np.concatenate(means, axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5], axis=1)
    return X.mean(axis=1), lo, hi, N


contr = {name: {'d': [], 'lo': [], 'hi': []} for name in CONTRASTS}
for a, b in BANDS:
    pairs = list(CONTRASTS.values())
    d, lo, hi, N = boot_band(pairs, a, b)
    assert N == 88064
    for i, name in enumerate(CONTRASTS):
        contr[name]['d'].append(float(d[i]))
        contr[name]['lo'].append(float(lo[i]))
        contr[name]['hi'].append(float(hi[i]))

# identity pair: share of poles whose counts differ at all
ca, fa = cnt['start_a']
cb, fb = cnt['start_b']
diff_pole = (ca != cb).any(axis=3) | (fa != fb)
ident = {'share_poles_counts_differ_all': float(diff_pole.mean()),
         'n_poles_counts_differ': int(diff_pole.sum()),
         'share_poles_counts_differ_by_band': [float(diff_pole[a:b].mean()) for a, b in BANDS],
         'counts_arrays_identical': bool((ca == cb).all() and (fa == fb).all())}

# ---- references: tau, eps_dev, kappa, bar ----
R_ = {k: np.array(Cs[k]) for k in REFS}
tau = np.max(np.stack([np.abs(R_['t13'] - R_['t12']), np.abs(R_['t14b'] - R_['t13']),
                       np.abs(R_['t15'] - R_['t14c'])]), axis=0)
eps_dev = np.abs(R_['t14c'] - R_['t14b'])
kappa = np.zeros(4)
kappa_pair = [None] * 4
for i, x in enumerate(REFS):
    for y in REFS[i + 1:]:
        v = np.abs(R_[x] - R_[y])
        for bi in range(4):
            if v[bi] > kappa[bi]:
                kappa[bi], kappa_pair[bi] = v[bi], f'{x}-{y}'
I_ = np.array(contr['I']['d'])
bar = np.max(np.stack([tau, eps_dev, np.abs(I_)]), axis=0)


# ---- labels (§1.3) ----
def label_lead(d, lo, hi, bar_b, kap_b):
    if d > kap_b:
        return 'ABOVE_CEILING'
    if lo > bar_b:
        return 'SUPPORTED'
    if hi < bar_b:
        return 'REFUTED'
    return 'UNDECIDED'


def combine(labels):
    if all(x == 'SUPPORTED' for x in labels):
        return 'SUPPORTED'
    if any(x == 'ABOVE_CEILING' for x in labels):
        return 'ABOVE_CEILING'
    if any(x == 'REFUTED' for x in labels):
        return 'REFUTED'
    return 'UNDECIDED'


def label_between(rg_lo, rg_hi, mr_lo, mr_hi, bar_b):
    if rg_lo > bar_b and mr_lo > bar_b:
        return 'SUPPORTED'
    if rg_hi < -bar_b or mr_hi < -bar_b:
        return 'REFUTED'
    return 'UNDECIDED'


lead = [label_lead(contr['MG']['d'][i], contr['MG']['lo'][i], contr['MG']['hi'][i], bar[i], kappa[i]) for i in range(4)]
H_lead = combine(lead)
between = [label_between(contr['RG']['lo'][i], contr['RG']['hi'][i], contr['MR']['lo'][i], contr['MR']['hi'][i], bar[i])
           for i in range(4)]
# §1.3: SUPPORTED needs H_lead in every band and R between in every band; REFUTED if H_lead refuted or R outside
# in any band. §5 step 6 says "per band where the lead is SUPPORTED"; both readings are computed.
if H_lead == 'REFUTED':
    H_order_13 = 'REFUTED'
elif any(x == 'REFUTED' for x in between):
    H_order_13 = 'REFUTED'
elif H_lead == 'SUPPORTED' and all(x == 'SUPPORTED' for x in between):
    H_order_13 = 'SUPPORTED'
else:
    H_order_13 = 'UNDECIDED'
between_where_lead = [between[i] if lead[i] == 'SUPPORTED' else 'NOT_LABELLED' for i in range(4)]

# G-ID (§4): max_b |I(b)| <= min_b tau(b) / 4; failing input start_b := t15
gid_line = float(tau.min() / 4)
gid = {'max_abs_I': float(np.abs(I_).max()), 'line': gid_line, 'pass': bool(np.abs(I_).max() <= gid_line),
       'failing_input_t15_vs_start_a': contr['Gid_fail_t15_minus_start_a']['d'],
       'failing_input_max_abs': float(np.abs(contr['Gid_fail_t15_minus_start_a']['d']).max())}
gid['failing_input_pass'] = bool(gid['failing_input_max_abs'] <= gid_line)

# ---- H_specific (§3.6) ----
cM, fM = cnt['M']
cG, fG = cnt['G']
dir_delta = np.zeros((4, 50))
for bi, (a, b) in enumerate(BANDS):
    sM = cM[a:b].astype(np.float64) / fM[a:b, ..., None]
    sG = cG[a:b].astype(np.float64) / fG[a:b, ..., None]
    dir_delta[bi] = (sM - sG).reshape(-1, 50).mean(axis=0)
rs = np.random.default_rng(SEED + 1)
sets = np.stack([rs.choice(non_cm, size=5, replace=False) for _ in range(NSETS)])
spec = {'per_band': []}
for bi in range(4):
    d_cm = float(dir_delta[bi, cm_idx].sum())
    assert abs(d_cm - contr['MG']['d'][bi]) < 1e-12
    r = np.abs(dir_delta[bi][sets].sum(axis=1))
    p99, med = float(np.percentile(r, 99)), float(np.median(r))
    lab = 'SUPPORTED' if d_cm > p99 else ('REFUTED' if d_cm <= med else 'UNDECIDED')
    spec['per_band'].append({'d_cm': d_cm, 'p99_abs_random': p99, 'median_abs_random': med,
                             'max_abs_random': float(r.max()), 'mean_signed_random': float(dir_delta[bi][sets].sum(axis=1).mean()),
                             'label': lab})
spec['H_specific'] = combine([x['label'] for x in spec['per_band']])
spec['dir_delta_MG'] = {m['dirs'][i]: [float(dir_delta[bi, i]) for bi in range(4)] for i in range(50)}

# ---- PC_NLL (§3.5) ----
pc = {}
for k in HARV:
    mm = json.load(open(f'{BANKS}/{k}/m1/metrics.json'))
    pc[k] = mm['pc_nll']
w0 = pc['start_a']['windows']
windows_identical = all(pc[k]['windows']['cm'] == w0['cm'] and pc[k]['windows']['web'] == w0['web'] for k in HARV)
nll = {k: {g: np.array(pc[k]['nll'][g], dtype=np.float64) for g in ('cm', 'web')} for k in HARV}
for k in HARV:
    assert len(nll[k]['cm']) == 32 and len(nll[k]['web']) == 32, k
rp = np.random.default_rng(SEED + 2)
ic = rp.integers(0, 32, size=(NB, 32))
iw = rp.integers(0, 32, size=(NB, 32))


def pc_eval(X, Y):
    dc = nll[X]['cm'] - nll[Y]['cm']
    dw = nll[X]['web'] - nll[Y]['web']
    bc = dc[ic].mean(axis=1)
    bw = dw[iw].mean(axis=1)
    bd = bc - bw
    return {'d_cm': float(dc.mean()), 'd_cm_ci': [float(x) for x in np.percentile(bc, [2.5, 97.5])],
            'd_web': float(dw.mean()), 'd_web_ci': [float(x) for x in np.percentile(bw, [2.5, 97.5])],
            'd_dd': float(dc.mean() - dw.mean()), 'd_dd_ci': [float(x) for x in np.percentile(bd, [2.5, 97.5])]}


idn = np.concatenate([nll['start_b']['cm'] - nll['start_a']['cm'], nll['start_b']['web'] - nll['start_a']['web']])
f_thr = max(0.01, abs(float(idn.mean())))


def fires(e):
    return bool(e['d_cm_ci'][1] < -f_thr and e['d_dd_ci'][1] < -f_thr and abs(e['d_cm']) < 2.0)


pcr = {'windows_identical_all_five': windows_identical,
       'n_cm': len(w0['cm']), 'n_web': len(w0['web']),
       'identity_mean_diff_64': float(idn.mean()), 'identity_max_abs_diff_64': float(np.abs(idn).max()),
       'f': f_thr,
       'MG': pc_eval('M', 'G'), 'GM_swapped': pc_eval('G', 'M'), 'RG': pc_eval('R', 'G'),
       'mean_nll': {k: {g: float(nll[k][g].mean()) for g in ('cm', 'web')} for k in HARV}}
pcr['fires'] = fires(pcr['MG'])
pcr['swapped_fires'] = fires(pcr['GM_swapped'])

# ---- room on the references (§3.4 repeated on all poles; bar without |I|) ----
bar_ref = np.maximum(tau, eps_dev)
order = REFS
room = {'pairs': [], 'n_SUPPORTED': 0, 'n_REFUTED': 0, 'n_UNDECIDED': 0, 'n_ABOVE_CEILING': 0}
pairs_room = []
for i in range(len(order) - 1):
    pairs_room += [(order[i + 1], order[i]), (order[i], order[i + 1])]   # (as M, as G)
for bi, (a, b) in enumerate(BANDS):
    d, lo, hi, _ = boot_band(pairs_room, a, b)
    for j, (x, y) in enumerate(pairs_room):
        lab = label_lead(d[j], lo[j], hi[j], bar_ref[bi], kappa[bi])
        room['n_' + lab] += 1
        room['pairs'].append({'M': x, 'G': y, 'band': bi, 'd': float(d[j]), 'lo': float(lo[j]), 'hi': float(hi[j]), 'label': lab})
room['tau_lt_kappa_all_bands'] = bool((tau < kappa).all())

# ---- C on the 640 grid poles at t13, to check against spec §3.4 (0.1704, 0.1353, 0.0566, 0.0530) ----
grid = [parse_pole(json.loads(line)['pole']) for line in open(f'{BANKS}/M/samples/contexts.jsonl')]
gl, gn, gs = (np.array(x) for x in zip(*grid))
c640 = {}
for k in ('t13', 't15', 'start_a', 'G', 'M'):
    v = share[k][gl, gn, gs]
    c640[k] = [float(v[(gl >= a) & (gl < b)].mean()) for a, b in BANDS]

# ---- per-pole distribution, exploratory ----
dist = {}
for k in ('start_a', 'G', 'R', 'M'):
    dist[k] = {'share_poles_cm_share_0': [float((share[k][a:b] == 0).mean()) for a, b in BANDS],
               'share_poles_cm_share_1': [float((share[k][a:b] == 1).mean()) for a, b in BANDS]}
dMG = share['M'] - share['G']
dist['MG_per_pole'] = {'share_pos': [float((dMG[a:b] > 0).mean()) for a, b in BANDS],
                       'share_neg': [float((dMG[a:b] < 0).mean()) for a, b in BANDS],
                       'share_zero': [float((dMG[a:b] == 0).mean()) for a, b in BANDS],
                       'sd_neuron_mean': [float(dMG[a:b].mean(axis=2).std()) for a, b in BANDS]}

out = {'phase': 1, 'blind': 'written before reading verdict_m1.json, infra/pg/m1.py, the job files, JOURNAL.md, raw_examples_m1.md',
       'seed': SEED, 'n_boot': NB, 'interval': 'percentile 2.5 / 97.5 of bootstrap means (numpy linear)',
       'C': Cs, 'contrasts': contr, 'identity': ident,
       'tau': tau.tolist(), 'eps_dev': eps_dev.tolist(), 'kappa': kappa.tolist(), 'kappa_pair': kappa_pair,
       'bar': bar.tolist(), 'bar_argmax': [['tau', 'eps_dev', 'I'][int(i)] for i in np.argmax(np.stack([tau, eps_dev, np.abs(I_)]), axis=0)],
       'lead_labels': lead, 'H_lead': H_lead,
       'between_labels_all_bands': between, 'between_labels_where_lead_supported': between_where_lead,
       'H_order': H_order_13, 'G_ID': gid, 'specific': spec, 'pc_nll': pcr, 'room_refs': room,
       'C_640_grid': c640, 'per_pole': dist}
out['headline'] = f"H_lead {H_lead}; H_order {H_order_13}; H_specific {spec['H_specific']}"
json.dump(out, open(f'{OUT}/read_numbers.json', 'w'), indent=1)
print(out['headline'])
