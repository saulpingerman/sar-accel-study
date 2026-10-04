#!/usr/bin/env python3
"""Moving-target refocus of the ship in the Panama image.

A target moving at constant velocity v is imaged exactly by backprojecting with the antenna track shifted by -v t,
which places it at its position at the aperture centre and focuses it. The ship is assumed to follow the channel
axis; its speed s (positive toward bearing B, negative opposite) is swept and the sharpness of the tile is recorded.

  python ship_refocus.py --data /data/panama.npz --out /data/ship --speeds -8:8:0.5 --bearing 158
"""
import argparse
import json
import os
import time

import numpy as np

import v2_prep
from sarbench import bp, cpu_ref

C = 299792458.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--speeds', default='-8:8:0.5')
    ap.add_argument('--bearing', type=float, default=158.0, help='channel bearing, degrees clockwise from north')
    ap.add_argument('--heading', default='0.7968,-0.5947', help='slant-plane unit heading per metre of ground travel along the bearing: e1, e2 components')
    ap.add_argument('--tile', default='5300,8372,2450,3218', help='i0,i1,j0,j1 of the tile')
    ap.add_argument('--oversample', type=int, default=8)
    ap.add_argument('--vsat', type=float, default=7680.0, help='platform speed used for the pulse times (m/s)')
    ap.add_argument('--duration', type=float, default=3.3748, help='collect duration from the SICD (s), used as a check')
    ap.add_argument('--save', default='', help='speeds whose tiles are saved, comma separated')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    col, S, grid = v2_prep.load(a.data)
    P, K = S.shape
    e1, e2, spx, spy, nx, ny = grid['e1'], grid['e2'], grid['spx'], grid['spy'], grid['nx'], grid['ny']
    from scipy.signal.windows import taylor
    wp = taylor(P, nbar=4, sll=35.0, norm=False).astype(np.float32)
    wk = taylor(K, nbar=4, sll=35.0, norm=False).astype(np.float32)
    nfft = 1 << int(np.ceil(np.log2(a.oversample * K)))
    dr = bp.range_bin(col.df, nfft)
    t0 = time.time()
    Sw = ((S * wp[:, None]) * wk[None, :]).astype(np.complex64)
    del S
    freqs = col.freqs.astype(np.float64)
    # pulse times from the track length and the platform speed, centred on the aperture
    ant = col.ant.astype(np.float64)
    step = np.linalg.norm(np.diff(ant, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(step)]) / a.vsat
    t -= t[-1] / 2.0
    print(f'aperture time {t[-1] - t[0]:.3f} s (SICD {a.duration:.3f} s)', flush=True)
    h1, h2 = (float(x) for x in a.heading.split(','))
    head = h1 * np.asarray(e1) + h2 * np.asarray(e2)                        # slant-plane velocity direction, per m/s of ground speed
    i0, i1, j0, j1 = (int(x) for x in a.tile.split(','))
    X, Y = np.meshgrid((np.arange(i0, i1) - nx / 2.0) * spx, (np.arange(j0, j1) - ny / 2.0) * spy, indexing='ij')
    pos = X.ravel()[:, None] * np.asarray(e1)[None, :] + Y.ravel()[:, None] * np.asarray(e2)[None, :]
    px, py, pz = (np.ascontiguousarray(pos[:, c]) for c in range(3))
    del X, Y, pos
    lo, hi, st = (float(x) for x in a.speeds.split(':'))
    speeds = np.round(np.arange(lo, hi + st / 2, st), 3)
    save = {float(x) for x in a.save.split(',') if x}
    rows = []
    for s in speeds:
        ts = time.time()
        ant_s = ant - s * head[None, :] * t[:, None]
        u, r0 = bp.host_geometry(ant_s)
        # the data are referenced to |ant_p|, the kernel to |ant_s|; the difference is the moving target's extra phase
        delta = r0 - np.linalg.norm(ant, axis=1)
        ramp = np.exp(1j * (4.0 * np.pi / C) * np.outer(delta, freqs)).astype(np.complex64)
        rc = cpu_ref.range_compress(Sw * ramp, nfft).astype(np.complex64)
        del ramp
        img = cpu_ref.bp(rc, u, r0, px, py, pz, dr, col.fref).reshape(i1 - i0, j1 - j0)
        del rc
        I = np.abs(img) ** 2
        top = np.sort(I.ravel())[-2000:]                                      # the ship: its brightest pixels
        pk = np.unravel_index(np.argmax(I), I.shape)
        row = dict(speed=float(s), sharpness=float(np.sum(I * I)), peak=float(I.max()), top2000=float(top.sum()),
                   peak_i=int(pk[0] + i0), peak_j=int(pk[1] + j0), seconds=time.time() - ts)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if float(s) in save or abs(s) < 1e-9:
            np.save(f'{a.out}/tile_{s:+.2f}.npy', img.astype(np.complex64))
        json.dump(dict(tile=[i0, i1, j0, j1], bearing=a.bearing, heading=[h1, h2], rows=rows), open(f'{a.out}/sweep_{a.bearing:.0f}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
