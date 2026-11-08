"""Shared paths and loaders for the independent read of M1. Numpy only. Written blind to infra/pg/m1.py."""
import json

import numpy as np

WT = '/Users/abaybektursun/projects/interp_research-m1read'
SP = '/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad'
BANKS = WT + '/studies/next/m1/banks'
OUT = WT + '/studies/next/m1/read'
MANIFEST = SP + '/m1/corpus_manifest.json'
S3DIR = SP + '/m1read_s3'
OFFSETS = S3DIR + '/doc_offsets/doc_offsets.npy'

HARV = ['start_a', 'start_b', 'G', 'R', 'M']
REFS = ['t00', 't12', 't13', 't14b', 't14c', 't15']
BANDS = [(0, 8), (8, 16), (16, 24), (24, 32)]
CM_DIRS = ['data/finemath-3plus', 'data/stack_edu-CSharp', 'data/stack_edu-JavaScript',
           'data/stack_edu-Python', 'data/stack_edu-SQL']
SIGN = {'+': 0, '-': 1}


def manifest():
    return json.load(open(MANIFEST))


def doc_dirs(m):
    """Directory index of every document: band 1 of every shard in shard order, then band 2, then band 3."""
    sh = m['shards']
    assert [s['dir'] for s in sh] == m['dirs']
    out = []
    for b in ('1', '2', '3'):
        assert sum(s['n_docs_selected'][b] for s in sh) == m['bands'][b]['doc_end'] - m['bands'][b]['doc_start']
        for i, s in enumerate(sh):
            out += [i] * s['n_docs_selected'][b]
    out = np.array(out, dtype=np.int64)
    assert len(out) == m['n_docs']
    return out


def counts_path(k):
    return f'{BANKS}/refs/{k}/m1/counts.npz' if k in REFS else f'{BANKS}/{k}/m1/counts.npz'


def load_counts(k, m):
    z = np.load(counts_path(k), allow_pickle=False)
    c, f = z['counts'], z['filled']
    assert c.shape == (32, 11008, 2, 50) and f.shape == (32, 11008, 2)
    assert list(z['dirs']) == m['dirs'], k
    cm = np.array([d in CM_DIRS for d in m['dirs']])
    assert (z['cm_mask'] == cm).all(), k
    assert (c.sum(axis=3, dtype=np.int64) == f).all(), k
    assert f.min() > 0, k
    return c, f


def parse_pole(p):
    return int(p[1:3]), int(p[4:-1]), SIGN[p[-1]]
