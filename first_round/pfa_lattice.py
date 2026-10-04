#!/usr/bin/env python3
"""Polar format accuracy across a wide scene.

Polar format assumes planar wavefronts, so its focus degrades with distance
from the scene centre at a rate set by range and wavelength. A 9 x 9 lattice
of point targets spanning an N-pixel scene is imaged by polar format and by
float64 backprojection chips, and the peak loss, broadening and sidelobe level
of every target are reported against radius.

  python pfa_lattice.py --N 4096 --r0 600e3 --out results/lat/pfa_4096_space.json
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=4096)
    ap.add_argument('--r0', type=float, default=600e3)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import jax
    jax.config.update('jax_enable_x64', True)
    from sarbench import sim, pfa, cpu_ref, metrics as M
    N, res = a.N, 0.3
    scene = N * res / (1.25 * np.sqrt(2.0))
    spacing = scene / N
    col = sim.make_collect(res=res, scene=scene, r0=a.r0)
    g = np.linspace(-0.45, 0.45, 9) * scene
    X, Y = np.meshgrid(g, g, indexing='ij')
    pos = np.stack([X, Y, np.zeros_like(X)], axis=-1).reshape(-1, 3)
    S = sim.simulate(col, pos, np.ones(len(pos), np.complex128))
    geo = pfa.pfa_geometry(col)
    nf = 1 << int(np.ceil(np.log2(1.5 * N)))
    dx, dy = pfa.pixel_spacing(geo, nf, nf)
    rows = []
    imgs = {}
    imgs['pfaczt'] = np.asarray(pfa.make_pfa_czt('fp64', col.K, col.Np, geo['nkx'], geo['nky'], nf, nf)(*pfa.prepare_czt('fp64', S, geo)))
    imgs['pfa'] = np.asarray(pfa.make_pfa('fp64', geo['nkx'], geo['nky'], nf, nf)(*pfa.prepare('fp64', S, geo)))
    for name, img in imgs.items():
        per = []
        for (x, y, _) in pos:
            # polar format places a target near its ground position; search a window for the peak
            r0_, c0_ = int(round(nf / 2 + x / dx)), int(round(nf / 2 + y / dy))
            w = 40
            sub = np.abs(img[r0_ - w:r0_ + w, c0_ - w:c0_ + w])
            k = np.unravel_index(np.argmax(sub), sub.shape)
            r, c = r0_ - w + k[0], c0_ - w + k[1]
            rep = M.irf_report(img, (r, c), (dx, dy), half=24, up=8)
            per.append(dict(radius_m=float(np.hypot(x, y)), peak=abs(rep['peak']), shift_m=[float((rep['pos'][0] - nf / 2) * dx - x), float((rep['pos'][1] - nf / 2) * dy - y)],
                            res0=rep['ax0']['irw'], res1=rep['ax1']['irw'], pslr=max(rep['ax0']['pslr_db'], rep['ax1']['pslr_db'])))
        cen = min(per, key=lambda p: p['radius_m'])
        for p in per:
            p['peak_db'] = float(20 * np.log10(p['peak'] / cen['peak']))
            p['broadening'] = float(max(p['res0'] / cen['res0'], p['res1'] / cen['res1']))
        per.sort(key=lambda p: p['radius_m'])
        row = dict(algo=name, N=N, r0=a.r0, scene_m=scene, dx=dx, dy=dy, targets=per,
                   worst_peak_db=min(p['peak_db'] for p in per), worst_broadening=max(p['broadening'] for p in per),
                   worst_pslr_db=max(p['pslr'] for p in per), centre_pslr_db=cen['pslr'],
                   worst_shift_m=float(max(np.hypot(*p['shift_m']) for p in per)))
        rows.append(row)
        print(name, f"N={N} r0={a.r0 / 1e3:.0f} km scene {scene:.0f} m: peak loss to {row['worst_peak_db']:.2f} dB, broadening to {row['worst_broadening']:.3f}, "
              f"PSLR centre {row['centre_pslr_db']:.1f} worst {row['worst_pslr_db']:.1f} dB, position shift to {row['worst_shift_m']:.2f} m", flush=True)
        for p in per[::10]:
            print(f"   r={p['radius_m']:6.0f} m peak {p['peak_db']:+.2f} dB broadening {p['broadening']:.3f} pslr {p['pslr']:.1f}")
    json.dump(rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
