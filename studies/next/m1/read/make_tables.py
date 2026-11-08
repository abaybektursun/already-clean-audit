"""Markdown tables for READ_m1.md, generated from read_numbers.json, compare.json, verdict_m1.json, thresholds.json.
Run: python -I make_tables.py > tables.md
"""
import json

WT = '/Users/abaybektursun/projects/interp_research-m1read'
R = json.load(open(f'{WT}/studies/next/m1/read/read_numbers.json'))
V = json.load(open(f'{WT}/studies/next/m1/verdict_m1.json'))
T = json.load(open(f'{WT}/studies/next/m1/thresholds.json'))
B = ['b0 L0-7', 'b1 L8-15', 'b2 L16-23', 'b3 L24-31']


def f(x, n=5):
    return f'{x:+.{n}f}'


def dmax(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


print('### C per checkpoint (fraction of a pole\'s filled top-50 slots)\n')
print('| checkpoint | ' + ' | '.join(B) + ' | max \\|mine − verdict\\| | verdict field |')
print('|---|' + '---|' * 6)
for k in ['start_a', 'start_b', 'G', 'R', 'M']:
    print(f'| {k} | ' + ' | '.join(f'{x:.5f}' for x in R['C'][k]) + f" | {dmax(R['C'][k], V['exploratory']['C'][k]):.1e} | `verdict_m1.json` exploratory.C |")
for k in ['t00', 't12', 't13', 't14b', 't14c', 't15']:
    print(f'| {k} | ' + ' | '.join(f'{x:.5f}' for x in R['C'][k]) + f" | {dmax(R['C'][k], T['C_refs'][k]):.1e} | `thresholds.json` C_refs |")

print('\n### Thresholds and the identity pair\n')
print('| quantity | ' + ' | '.join(B) + ' | max \\|mine − verdict\\| | verdict field |')
print('|---|' + '---|' * 6)
for k, src, val in (('tau', 'thresholds.json tau', T['tau']), ('eps_dev', 'thresholds.json eps_dev', T['eps_dev']),
                    ('kappa', 'thresholds.json kappa', T['kappa']), ('bar', 'verdict_m1.json bar', V['bar'])):
    print(f'| {k} | ' + ' | '.join(f'{x:.6f}' for x in R[k]) + f' | {dmax(R[k], val):.1e} | `{src}` |')
print('| I = C(start_b) − C(start_a) | ' + ' | '.join(f'{x:.6f}' for x in R['contrasts']['I']['d']) + f" | {dmax(R['contrasts']['I']['d'], V['G_ID']['I']):.1e} | `verdict_m1.json` G_ID.I |")
fi = R['contrasts']['Gid_fail_t15_minus_start_a']['d']
print('| G-ID failing input: C(t15) − C(start_a) | ' + ' | '.join(f'{x:+.5f}' for x in fi) + ' | (verdict stores only its pass flag) | `verdict_m1.json` G_ID.failing_input_pass |')
print(f"\nκ pair per band (mine): {', '.join(R['kappa_pair'])}. bar is set by τ in every band.\n")

print('### Band contrasts, 95 % bootstrap over neurons (2,000 resamples)\n')
print('Mine uses `default_rng(20261105)`; the verdict uses `default_rng(0)`. Means are deterministic.\n')
print('| contrast | band | mine mean | verdict mean | mine [lo, hi] | verdict [lo, hi] | max \\|diff\\| of bounds |')
print('|---|---|---|---|---|---|---|')
for nm, key in (('Δ_MG', 'MG'), ('Δ_RG', 'RG'), ('Δ_MR', 'MR')):
    for b in range(4):
        c = R['contrasts'][key]
        if key == 'MG':
            vm, vl, vh = V['lead'][b]['mean'], V['lead'][b]['lo'], V['lead'][b]['hi']
        else:
            vm, vl, vh = V['between'][b]['R_minus_G' if key == 'RG' else 'M_minus_R']
        print(f"| {nm} | {B[b]} | {f(c['d'][b])} | {f(vm)} | [{f(c['lo'][b])}, {f(c['hi'][b])}] | [{f(vl)}, {f(vh)}] | {max(abs(c['lo'][b]-vl), abs(c['hi'][b]-vh)):.1e} |")
for k in ('G', 'R', 'M'):
    for b in range(4):
        c = R['contrasts'][f'{k}_minus_start']
        vm, vl, vh = V['exploratory']['shift_from_start'][k][b]
        print(f"| {k} − start_a | {B[b]} | {f(c['d'][b])} | {f(vm)} | [{f(c['lo'][b])}, {f(c['hi'][b])}] | [{f(vl)}, {f(vh)}] | {max(abs(c['lo'][b]-vl), abs(c['hi'][b]-vh)):.1e} |")

print('\n### Labels\n')
print('| rule | ' + ' | '.join(B) + ' | combined (mine) | combined (verdict) |')
print('|---|' + '---|' * 6)
print('| H_lead (§1.3) | ' + ' | '.join(R['lead_labels']) + f" | {R['H_lead']} | {V['H_lead']} |")
print('| verdict lead labels | ' + ' | '.join(x['label'] for x in V['lead']) + ' | | |')
print('| H_order, R between (§1.3) | ' + ' | '.join(R['between_labels_all_bands']) + f" | {R['H_order']} | {V['H_order']} |")
print('| verdict between labels | ' + ' | '.join(x['label'] for x in V['between']) + ' | | |')
sp = R['specific']['per_band']
print('| H_specific (§3.6) | ' + ' | '.join(x['label'] for x in sp) + f" | {R['specific']['H_specific']} | {V['H_specific']} |")
print('| verdict specificity labels | ' + ' | '.join(x['label'] for x in V['specificity']) + ' | | |')

print('\n### H_specific numbers (my own 1,000 random sets of 5 of the 45 other directories)\n')
print('| band | Δ_MG code+math (mine = verdict) | mine p99 \\|random\\| | verdict p99 | mine median | verdict median | mine max \\|random\\| |')
print('|---|---|---|---|---|---|---|')
for b in range(4):
    print(f"| {B[b]} | {f(sp[b]['d_cm'])} | {sp[b]['p99_abs_random']:.5f} | {V['specificity'][b]['random_abs_q99']:.5f} | {sp[b]['median_abs_random']:.5f} | {V['specificity'][b]['random_abs_median']:.5f} | {sp[b]['max_abs_random']:.5f} |")

P, VP = R['pc_nll'], V['PC_NLL']
print('\n### PC_NLL (nats per token; 32 code+math windows, 32 web windows)\n')
print('| quantity | mine | verdict | diff |')
print('|---|---|---|---|')
print(f"| Δcm = mean(NLL_M − NLL_G), cm windows | {P['MG']['d_cm']:+.4f} [{P['MG']['d_cm_ci'][0]:+.4f}, {P['MG']['d_cm_ci'][1]:+.4f}] | {VP['cm_M_minus_G'][0]:+.4f} [{VP['cm_M_minus_G'][1]:+.4f}, {VP['cm_M_minus_G'][2]:+.4f}] | mean {P['MG']['d_cm']-VP['cm_M_minus_G'][0]:.1e}; bounds ≤ {max(abs(P['MG']['d_cm_ci'][0]-VP['cm_M_minus_G'][1]), abs(P['MG']['d_cm_ci'][1]-VP['cm_M_minus_G'][2])):.1e} |")
print(f"| Δweb = mean(NLL_M − NLL_G), web windows | {P['MG']['d_web']:+.4f} [{P['MG']['d_web_ci'][0]:+.4f}, {P['MG']['d_web_ci'][1]:+.4f}] | not stored | |")
print(f"| Δcm − Δweb | {P['MG']['d_dd']:+.4f} [{P['MG']['d_dd_ci'][0]:+.4f}, {P['MG']['d_dd_ci'][1]:+.4f}] | {VP['cm_minus_web'][0]:+.4f} [{VP['cm_minus_web'][1]:+.4f}, {VP['cm_minus_web'][2]:+.4f}] | mean {P['MG']['d_dd']-VP['cm_minus_web'][0]:.1e}; bounds ≤ {max(abs(P['MG']['d_dd_ci'][0]-VP['cm_minus_web'][1]), abs(P['MG']['d_dd_ci'][1]-VP['cm_minus_web'][2])):.1e} |")
print(f"| identity NLL difference, mean over 64 windows (start_b − start_a) | {P['identity_mean_diff_64']:.1e} (max \\|per-window\\| {P['identity_max_abs_diff_64']:.1e}) | {VP['identity_nll_diff']:.1e} | 0 |")
print(f"| floor f | {P['f']} | {VP['floor']} | 0 |")
print(f"| fires | {P['fires']} | {VP['pass']} | |")
print(f"| failing input (G and M swapped) fires | {P['swapped_fires']} (Δcm {P['GM_swapped']['d_cm']:+.4f}, upper bound {P['GM_swapped']['d_cm_ci'][1]:+.4f}) | {VP['failing_input_pass']} | |")
print(f"| window lists identical across the five harvests | {P['windows_identical_all_five']} | | |")
mn = P['mean_nll']
print('\nMean NLL per harvest (mine, `read_numbers.json` pc_nll.mean_nll): ' + '; '.join(f"{k} cm {mn[k]['cm']:.4f}, web {mn[k]['web']:.4f}" for k in mn) + '.')
