"""The frozen verdict code (infra/pg/m1.py at HEAD) on two planted bank sets built from symlinks:
(1) G missing -> must give DOES NOT COUNT (missing); (2) G and M swapped -> PC_NLL must fail and the headline must
read it. Run: python -I verdict_planted.py  ->  read/verdict_planted.json
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, '/Users/abaybektursun/projects/interp_research-m1read')
from infra.pg import m1 as M1  # noqa: E402

WT = '/Users/abaybektursun/projects/interp_research-m1read'
BANKS = pathlib.Path(WT + '/studies/next/m1/banks')
TMP = pathlib.Path('/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/m1read_planted')
th = json.loads((pathlib.Path(WT) / 'studies/next/m1/thresholds.json').read_text())
plans = {'G_missing': {'start_a': 'start_a', 'start_b': 'start_b', 'R': 'R', 'M': 'M', 'refs': 'refs'},
         'G_M_swapped': {'start_a': 'start_a', 'start_b': 'start_b', 'R': 'R', 'G': 'M', 'M': 'G', 'refs': 'refs'}}
out = {}
for name, links in plans.items():
    d = TMP / name
    d.mkdir(parents=True, exist_ok=False)
    for dst, src in links.items():
        os.symlink(BANKS / src, d / dst)
    v = M1.verdict(d, th)
    out[name] = {'headline': v['headline'],
                 'PC_NLL_pass': v.get('PC_NLL', {}).get('pass'),
                 'lead_labels': [x['label'] for x in v.get('lead', [])],
                 'H_lead': v.get('H_lead'), 'H_order': v.get('H_order'), 'H_specific': v.get('H_specific')}
    print(name, out[name])
json.dump(out, open(WT + '/studies/next/m1/read/verdict_planted.json', 'w'), indent=1)
