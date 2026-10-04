#!/usr/bin/env python3
"""Error of each configuration against local brightness.

The image is divided into blocks; for each block the error energy relative to
the block's own reference energy is computed and tabulated against the block's
brightness relative to the image mean.

  python brightness_check.py --ref out/sz8192_cpu --test out/sz8192_cpu,out/sz8192_tpu-v5e --tags ffbp/fp32,ffbp/bf16_mm,ffbp/fp32_fast,ffbp/f16 --out results/sizes/brightness8192.json
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--ref', required=True)
ap.add_argument('--test', required=True)
ap.add_argument('--tags', required=True)
ap.add_argument('--block', type=int, default=256)
ap.add_argument('--out', required=True)
a = ap.parse_args()
ref = np.load(f'{a.ref}/bp_numba_fp64.npy')
n, b = ref.shape[0], a.block
m = n // b
gain = None
for cand in ('ffbp_fp64', 'ffbp_fp32'):
    p = f'{a.ref}/{cand}.npy'
    if os.path.exists(p) and gain is None:
        src = np.load(p).astype(np.complex128)
        gain = complex(np.vdot(src, ref) / np.vdot(src, src).real)
        del src
E = (np.abs(ref) ** 2).reshape(m, b, m, b).sum((1, 3))
bright = 10 * np.log10(E / E.mean())
edges = [-100, -15, -10, -5, 0, 5, 10, 100]
out = dict(n=n, block=b, edges=edges, blocks=[int(((bright >= lo) & (bright < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])], rows=[])
for d in a.test.split(','):
    meta = json.load(open(f'{d}/timing.json')) if os.path.exists(f'{d}/timing.json') else {}
    label = meta.get('_meta', {}).get('label', os.path.basename(d))
    for tag in a.tags.split(','):
        p = f"{d}/{tag.replace('/', '_')}.npy"
        if not os.path.exists(p):
            continue
        t = np.load(p)
        if tag.startswith('ffbp'):
            t = t * gain
        e = (np.abs(t - ref) ** 2).reshape(m, b, m, b).sum((1, 3))
        rel = 10 * np.log10(e / E)
        x, y = bright.ravel(), rel.ravel()
        slope, icpt = np.polyfit(x, y, 1)
        row = dict(label=label, tag=tag, whole_db=float(10 * np.log10(e.sum() / E.sum())), slope_db_per_db=float(slope), corr=float(np.corrcoef(x, y)[0, 1]),
                   density_spread_db=float(np.percentile(10 * np.log10(e.ravel()), 90) - np.percentile(10 * np.log10(e.ravel()), 10)),
                   bins=[float(np.median(rel[(bright >= lo) & (bright < hi)])) if ((bright >= lo) & (bright < hi)).any() else None for lo, hi in zip(edges[:-1], edges[1:])])
        out['rows'].append(row)
        print({k: (round(v, 3) if isinstance(v, float) else ([None if q is None else round(q, 1) for q in v] if isinstance(v, list) else v)) for k, v in row.items()}, flush=True)
        del t
print('blocks per bin', out['blocks'], 'brightness spread p10..p90 dB', np.percentile(bright, 10), np.percentile(bright, 90))
json.dump(out, open(a.out, 'w'), indent=1)
