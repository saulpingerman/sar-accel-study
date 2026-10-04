#!/usr/bin/env python3
"""Float32 evaluation of the second-level range differences of the factorized algorithm.

At levels after the first, the device computes the range difference between a
parent tile center and each child center in float32 (the `phases` function in
sarbench/ffbp.py). This script evaluates the same expression in float32 and in
float64 for the antenna path of the measured collection and for the first-level
tile sizes of the image-size series, and reports the phase error and the image
error it implies (10 log10 of the mean squared phase error in radians).

  python geometry_check.py --data sz1024.npz --out results/sizes/geometry_check.json
"""
import argparse
import json

import numpy as np

C0 = 299792458.0
ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
d_ = np.load(a.data)
ant = d_['ant'].astype(np.float64)
ant = ant[::max(1, len(ant) // 512)]                     # the result does not depend on the pulse density
f0 = float(d_['fmin']) + 0.5 * float(d_['df']) * d_['S'].shape[1]
k0 = 2.0 * f0 / C0
r0 = np.linalg.norm(ant, axis=1)
u = ant / r0[:, None]


def ddr(u, r0, ref, d, f):
    u, r0, ref, d = u.astype(f), r0.astype(f), ref.astype(f), d.astype(f)
    uc = u[:, 0] * ref[0] + u[:, 1] * ref[1] + u[:, 2] * ref[2]
    wn = r0 * np.sqrt(f(1) + ((ref * ref).sum() - f(2) * r0 * uc) / (r0 * r0))
    ud = d[:, 0:1] * u[None, :, 0] + d[:, 1:2] * u[None, :, 1] + d[:, 2:3] * u[None, :, 2]
    wd = r0[None, :] * ud - (d * ref[None, :]).sum(1)[:, None]
    num = (d * d).sum(1)[:, None] - f(2) * wd
    return num / (np.sqrt(wn[None, :] * wn[None, :] + num) + wn[None, :])


out = dict(rows=[], range_km=float(r0.mean() / 1e3), f0=f0)
for side, tile, split in ((4096, 128.0, 4), (8192, 256.0, 8), (16384, 512.0, 8)):
    n1 = int(round(side * 0.25 / tile))                   # first-level tiles per axis
    cs = tile / split                                     # child spacing
    off = (np.arange(split) - (split - 1) / 2.0) * cs
    d = np.array([[x, y, 0.0] for x in off for y in off])
    cen = (np.arange(n1) - (n1 - 1) / 2.0) * tile
    e2 = []
    for x in cen:
        for y in cen:
            ref = np.array([x, y, 0.0])
            exact = ddr(u, r0, ref, d, np.float64)
            # the exact value from float64 inputs, against float32 arithmetic on float32 inputs
            err = (ddr(u, r0, ref, d, np.float32).astype(np.float64) - exact) * k0 * 2.0 * np.pi
            e2.append(np.mean(err ** 2))
    row = dict(side=side, first_level_tile_m=tile, rms_phase_rad=float(np.sqrt(np.mean(e2))), implied_error_db=float(10 * np.log10(np.mean(e2))))
    out['rows'].append(row)
    print(row)
json.dump(out, open(a.out, 'w'), indent=1)
