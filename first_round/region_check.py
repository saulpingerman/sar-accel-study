#!/usr/bin/env python3
"""Local error in one ground region at two image sizes.

The 1024-pixel image of the size series covers the central 256 m of the
4096-pixel image at the same pixel spacing. This script compares the error of
each precision configuration in that region within the larger image with the
error over the smaller image, and reports how bright the region is relative
to its surroundings.

  python region_check.py --dir /data/out --out results/sizes/region_check.json
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
D = a.dir
r4 = np.load(f'{D}/sz4096_cpu/bp_numba_fp64.npy')
r1 = np.load(f'{D}/sz1024_cpu/bp_numba_fp64.npy')
c = slice(1536, 2560)
E4, Ec, E1 = np.sum(np.abs(r4) ** 2), np.sum(np.abs(r4[c, c]) ** 2), np.sum(np.abs(r1) ** 2)
out = dict(center_vs_mean_db=float(10 * np.log10(16 * Ec / E4)), rows=[])
p = np.abs(r4) ** 2
for h in (181, 256, 362):
    k = int(h / 0.25)
    cc = slice(2048 - k, 2048 + k)
    out[f'energy_square_{h}m_over_center'] = float(p[cc, cc].sum() / Ec)


def gain(d):
    s, r = np.load(f'{d}/ffbp_fp64.npy'), np.load(f'{d}/bp_numba_fp64.npy')
    s = s.astype(np.complex128)
    return np.vdot(s, r) / np.vdot(s, s).real


g4, g1 = gain(f'{D}/sz4096_cpu'), gain(f'{D}/sz1024_cpu')
for tag in ('ffbp_fp32', 'ffbp_f16', 'ffbp_bf16_mm', 'ffbp_bf16', 'ffbp_f8_mm'):
    t4, t1 = np.load(f'{D}/sz4096_cpu/{tag}.npy') * g4, np.load(f'{D}/sz1024_cpu/{tag}.npy') * g1
    e4, e1 = np.abs(t4 - r4) ** 2, np.abs(t1 - r1) ** 2
    row = dict(tag=tag.replace('_', '/', 1), err4096_db=float(10 * np.log10(e4.sum() / E4)), err4096_center_db=float(10 * np.log10(e4[c, c].sum() / Ec)),
               err1024_db=float(10 * np.log10(e1.sum() / E1)), density_center_over_mean=float(e4[c, c].mean() / e4.mean()))
    out['rows'].append(row)
    print(row, flush=True)
json.dump(out, open(a.out, 'w'), indent=1)
