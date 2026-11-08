"""Phase 2 of the independent read: verdict_m1.json and thresholds.json against read/read_numbers.json, field by field.
Also the freeze room check on all 30 ordered reference pairs with my own bootstrap.
Run: python -I compare_m1.py  ->  read/compare.json, read/compare.txt
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/Users/abaybektursun/projects/interp_research-m1read/studies/next/m1/read')
from common import BANDS, CM_DIRS, OUT, REFS, WT, load_counts, manifest  # noqa: E402

V = json.load(open(f'{WT}/studies/next/m1/verdict_m1.json'))
T = json.load(open(f'{WT}/studies/next/m1/thresholds.json'))
R = json.load(open(f'{OUT}/read_numbers.json'))
rows = []


def cmp(field, mine, theirs, kind='exact'):
    if isinstance(mine, str) or isinstance(theirs, str) or isinstance(mine, bool):
        rows.append({'field': field, 'mine': mine, 'verdict': theirs, 'diff': None, 'kind': kind, 'match': mine == theirs})
        return
    d = float(mine) - float(theirs)
    tol = 1e-12 if kind == 'exact' else None
    rows.append({'field': field, 'mine': float(mine), 'verdict': float(theirs), 'diff': d, 'kind': kind,
                 'match': (abs(d) <= tol) if tol is not None else None})


names = {0: 'b0 (L0-7)', 1: 'b1 (L8-15)', 2: 'b2 (L16-23)', 3: 'b3 (L24-31)'}
for k in ('start_a', 'start_b', 'G', 'R', 'M'):
    for b in range(4):
        cmp(f'C[{k}] {names[b]}', R['C'][k][b], V['exploratory']['C'][k][b])
for k in REFS:
    for b in range(4):
        cmp(f'C[{k}] {names[b]}', R['C'][k][b], T['C_refs'][k][b])
for f in ('tau', 'eps_dev', 'kappa'):
    for b in range(4):
        cmp(f'{f} {names[b]}', R[f][b], T[f][b])
for b in range(4):
    cmp(f'bar {names[b]}', R['bar'][b], V['bar'][b])
    cmp(f'I {names[b]}', R['contrasts']['I']['d'][b], V['G_ID']['I'][b])
cmp('G-ID line', R['G_ID']['line'], V['G_ID']['line'])
cmp('G-ID pass', R['G_ID']['pass'], V['G_ID']['pass'])
cmp('G-ID changed pole share', R['identity']['share_poles_counts_differ_all'], V['G_ID']['changed_pole_share'])
cmp('G-ID failing input passes', R['G_ID']['failing_input_pass'], V['G_ID']['failing_input_pass'])
for b in range(4):
    L = V['lead'][b]
    cmp(f'Delta_MG mean {names[b]}', R['contrasts']['MG']['d'][b], L['mean'])
    cmp(f'Delta_MG lo {names[b]}', R['contrasts']['MG']['lo'][b], L['lo'], 'bootstrap')
    cmp(f'Delta_MG hi {names[b]}', R['contrasts']['MG']['hi'][b], L['hi'], 'bootstrap')
    cmp(f'lead bar {names[b]}', R['bar'][b], L['bar'])
    cmp(f'lead ceiling {names[b]}', R['kappa'][b], L['ceiling'])
    cmp(f'lead label {names[b]}', R['lead_labels'][b], L['label'])
for b in range(4):
    Bt = V['between'][b]
    for nm, key in (('Delta_RG', 'R_minus_G'), ('Delta_MR', 'M_minus_R')):
        c = R['contrasts'][nm.split('_')[1]]
        cmp(f'{nm} mean {names[b]}', c['d'][b], Bt[key][0])
        cmp(f'{nm} lo {names[b]}', c['lo'][b], Bt[key][1], 'bootstrap')
        cmp(f'{nm} hi {names[b]}', c['hi'][b], Bt[key][2], 'bootstrap')
    cmp(f'between label {names[b]}', R['between_labels_all_bands'][b], Bt['label'])
P = V['PC_NLL']
cmp('PC_NLL d_cm mean', R['pc_nll']['MG']['d_cm'], P['cm_M_minus_G'][0])
cmp('PC_NLL d_cm lo', R['pc_nll']['MG']['d_cm_ci'][0], P['cm_M_minus_G'][1], 'bootstrap')
cmp('PC_NLL d_cm hi', R['pc_nll']['MG']['d_cm_ci'][1], P['cm_M_minus_G'][2], 'bootstrap')
cmp('PC_NLL d_cm-d_web mean', R['pc_nll']['MG']['d_dd'], P['cm_minus_web'][0])
cmp('PC_NLL d_cm-d_web lo', R['pc_nll']['MG']['d_dd_ci'][0], P['cm_minus_web'][1], 'bootstrap')
cmp('PC_NLL d_cm-d_web hi', R['pc_nll']['MG']['d_dd_ci'][1], P['cm_minus_web'][2], 'bootstrap')
cmp('PC_NLL identity NLL diff', R['pc_nll']['identity_mean_diff_64'], P['identity_nll_diff'])
cmp('PC_NLL floor f', R['pc_nll']['f'], P['floor'])
cmp('PC_NLL fires', R['pc_nll']['fires'], P['pass'])
cmp('PC_NLL swapped fires', R['pc_nll']['swapped_fires'], P['failing_input_pass'])
for b in range(4):
    S, mine = V['specificity'][b], R['specific']['per_band'][b]
    cmp(f'H_specific cm {names[b]}', mine['d_cm'], S['cm'])
    cmp(f'H_specific random q99 {names[b]}', mine['p99_abs_random'], S['random_abs_q99'], 'own random sets')
    cmp(f'H_specific random median {names[b]}', mine['median_abs_random'], S['random_abs_median'], 'own random sets')
    cmp(f'H_specific label {names[b]}', mine['label'], S['label'])
for k in ('G', 'R', 'M'):
    c = R['contrasts'][f'{k}_minus_start']
    for b in range(4):
        sh = V['exploratory']['shift_from_start'][k][b]
        cmp(f'shift {k}-start_a mean {names[b]}', c['d'][b], sh[0])
        cmp(f'shift {k}-start_a lo {names[b]}', c['lo'][b], sh[1], 'bootstrap')
        cmp(f'shift {k}-start_a hi {names[b]}', c['hi'][b], sh[2], 'bootstrap')
for h in ('H_lead', 'H_order'):
    cmp(h, R[h], V[h])
cmp('H_specific', R['specific']['H_specific'], V['H_specific'])
cmp('headline', R['headline'], V['headline'])
for k in ('start_a', 'start_b', 'G', 'R', 'M'):
    mm = json.load(open(f'{WT}/studies/next/m1/banks/{k}/m1/metrics.json'))
    cmp(f'G-FIT nll {k} (m1/metrics.json fit)', mm['fit']['nll'], V['gates'][k]['fit']['nll'])

# ---- room on all 30 ordered reference pairs ----
m = manifest()
cm_idx = np.array([m['dirs'].index(d) for d in CM_DIRS])
share = {}
for k in REFS:
    c, f = load_counts(k, m)
    share[k] = c[..., cm_idx].sum(axis=3, dtype=np.int64) / f.astype(np.float64)
bar_ref = np.maximum(np.array(R['tau']), np.array(R['eps_dev']))
kap = np.array(R['kappa'])
rng = np.random.default_rng(99)
pairs = [(p['as_G'], p['as_M']) for p in T['room']['pairs']]
lab_mine = {p: [None] * 4 for p in pairs}
for bi, (a, b) in enumerate(BANDS):
    X = np.stack([(share[mm][a:b] - share[gg][a:b]).mean(axis=2).ravel() for gg, mm in pairs])
    N = X.shape[1]
    means = np.concatenate([X[:, rng.integers(0, N, size=(20, N))].mean(axis=2) for _ in range(100)], axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5], axis=1)
    d = X.mean(axis=1)
    for j, p in enumerate(pairs):
        if d[j] > kap[bi]:
            lab = 'ABOVE_CEILING'
        elif lo[j] > bar_ref[bi]:
            lab = 'SUPPORTED'
        elif hi[j] < bar_ref[bi]:
            lab = 'REFUTED'
        else:
            lab = 'UNDECIDED'
        lab_mine[p][bi] = (lab, float(d[j]), float(lo[j]), float(hi[j]))
room_rows = []
n_same = 0
for p in T['room']['pairs']:
    key = (p['as_G'], p['as_M'])
    for bi in range(4):
        same = lab_mine[key][bi][0] == p['labels'][bi]
        n_same += same
        room_rows.append({'as_G': key[0], 'as_M': key[1], 'band': bi, 'mine': lab_mine[key][bi][0], 'thresholds': p['labels'][bi],
                          'd': lab_mine[key][bi][1], 'lo': lab_mine[key][bi][2], 'hi': lab_mine[key][bi][3], 'bar': float(bar_ref[bi])})
counts_mine = {}
for r in room_rows:
    counts_mine[r['mine']] = counts_mine.get(r['mine'], 0) + 1
out = {'rows': rows, 'n_fields': len(rows),
       'n_exact_match': sum(1 for r in rows if r['match'] is True),
       'n_exact_mismatch': sum(1 for r in rows if r['match'] is False),
       'max_abs_diff_exact_numeric': max(abs(r['diff']) for r in rows if r['kind'] == 'exact' and r['diff'] is not None),
       'max_abs_diff_bootstrap': max(abs(r['diff']) for r in rows if r['kind'] == 'bootstrap'),
       'max_abs_diff_random_sets': max(abs(r['diff']) for r in rows if r['kind'] == 'own random sets'),
       'room_30_pairs': {'n_labels': len(room_rows), 'n_same_label': int(n_same), 'label_counts_mine': counts_mine,
                         'differing': [r for r in room_rows if r['mine'] != r['thresholds']], 'rows': room_rows}}
json.dump(out, open(f'{OUT}/compare.json', 'w'), indent=1)
with open(f'{OUT}/compare.txt', 'w') as fh:
    for r in rows:
        if r['diff'] is None:
            fh.write(f"{r['field']:<44} mine={r['mine']!s:<14} verdict={r['verdict']!s:<14} match={r['match']}\n")
        else:
            fh.write(f"{r['field']:<44} mine={r['mine']:+.10f} verdict={r['verdict']:+.10f} diff={r['diff']:+.2e} ({r['kind']})\n")
print({k: v for k, v in out.items() if k not in ('rows', 'room_30_pairs')})
print({k: v for k, v in out['room_30_pairs'].items() if k != 'rows'})
