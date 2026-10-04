#!/usr/bin/env python3
"""Compare the image formed at each precision setting with the float64 image
of the same collection.

Two measures are taken per setting: the energy of the difference from the
float64 image (the error), and a coherence map between the two images, the
quantity a coherent change detector computes. Here the two inputs are the
same collection formed with different arithmetic, so any loss of coherence
is change introduced by the arithmetic alone.

  python same_image.py --images lock_cpu.npz --ref-tag bp/numba_fp64 --win 9 --out same_lock.json \
      --figs figs/real --prefix real --zoom 560,1040,320 --range-db 45
  python same_image.py --images hdr6_cpu.npz --data hdr6.npz --win 13 --out same_hdr6.json --figs figs/hdr6 --prefix hdr6 --range-db 70
"""
import argparse
import json
import os

import numpy as np

from sarbench import metrics as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--ref', help='file holding the float64 images, if not --images')
    ap.add_argument('--ref-tag', default='bp/fp64')
    ap.add_argument('--data', help='scene file of the dynamic-range scene; adds per-band results')
    ap.add_argument('--win', type=int, default=11)
    ap.add_argument('--out', required=True)
    ap.add_argument('--figs')
    ap.add_argument('--prefix', default='')
    ap.add_argument('--zoom', default='', help='row,col,size of a full-resolution window for the figures')
    ap.add_argument('--range-db', type=float, default=45.0)
    ap.add_argument('--down', type=int, default=2)
    a = ap.parse_args()
    z = np.load(a.images)
    zr = np.load(a.ref) if a.ref else z
    meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}
    whichs = [w for w in ('p1', 'p2') if f'{a.ref_tag}/{w}' in zr.files]
    refs = {w: zr[f'{a.ref_tag}/{w}'] for w in whichs}
    tags = sorted({k.rsplit('/', 1)[0] for k in z.files if k.startswith(('bp/', 'ffbp')) and k.endswith('/p1')})
    gains = {}

    def gain(tag, w):
        """Backprojection shares the reference's normalization. The factorized algorithm differs by a constant,
        fitted once per collection on its most precise image, so amplitude errors of a setting remain errors."""
        fam = tag.split('/')[0]
        if fam == 'bp':
            return 1.0
        if (fam, w) not in gains:
            for src in (zr, z):
                for pol in ('fp64', 'fp32'):
                    k = f'{fam}/{pol}/{w}'
                    if k in src.files and (fam, w) not in gains:
                        gains[(fam, w)] = np.vdot(src[k], refs[w]) / np.vdot(src[k], src[k])
        return gains[(fam, w)]

    bands = None
    if a.data:
        from sarbench import sim
        d = np.load(a.data)
        if 'strata' in d.files:
            sim.STRATA_DB = tuple(float(v) for v in d['strata'])
        n, sp, scene = meta['n'], meta['spacing'], float(d['scene'])
        ax = (np.arange(n) - n / 2) * sp
        X, Y = np.meshgrid(ax, ax, indexing='ij')
        edges, _ = sim.hdr_layout(scene)
        # keep clear of the two reflectors at x = -0.41 scene and of the band edges
        inside = (X > -0.38 * scene) & (X < 0.43 * scene) & (Y > edges[0] + 2) & (Y < edges[-1] - 2)
        for e in edges[1:-1]:
            inside &= np.abs(Y - e) >= a.win * sp
        stratum = sim.hdr_stratum(Y, scene)
        bands = [(db_, inside & (stratum == b)) for b, db_ in enumerate(sim.STRATA_DB)]

    rows = []
    for tag in tags:
        row = dict(tag=tag)
        for w in whichs:
            if f'{tag}/{w}' not in z.files:
                continue
            t = z[f'{tag}/{w}']
            if not np.isfinite(t).all():
                row['nonfinite'] = True
                continue
            t = t * gain(tag, w)
            c = M.coherence(t, refs[w], a.win)
            e = dict(err_db=float(M.error_db(t, refs[w])), coh_mean=float(c.mean()), coh_p01=float(np.percentile(c, 1)), coh_min=float(c.min()),
                     below_099=float((c < 0.99).mean()), below_09=float((c < 0.9).mean()))
            if bands and w == 'p1':
                e['bands'] = [dict(db=db_, coh=float(c[m].mean()), coh_p01=float(np.percentile(c[m], 1)),
                                   err_db=float(M.error_db(t[m], refs[w][m]))) for db_, m in bands]
            row[w] = e
        rows.append(row)
        if 'p1' in row:
            e = row['p1']
            print(f"{tag:22s} err {e['err_db']:7.1f} dB  coherence with float64: mean {e['coh_mean']:.5f} p01 {e['coh_p01']:.4f} min {e['coh_min']:.3f} "
                  f"below 0.99 {100 * e['below_099']:.2f}%" + ('  bands ' + ' '.join(f"{b['coh']:.4f}" for b in e['bands']) if 'bands' in e else ''), flush=True)
    json.dump(dict(win=a.win, ref=a.ref_tag, rows=rows, meta=meta), open(a.out, 'w'), indent=1)

    if a.figs:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(a.figs, exist_ok=True)
        r1 = refs['p1']
        pk = np.percentile(np.abs(r1), 99.9)
        pre = a.prefix + '_' if a.prefix else ''
        if a.zoom:
            i, j, s = (int(v) for v in a.zoom.split(','))
            zs = (slice(i, i + s), slice(j, j + s))
        else:
            zs = (slice(None), slice(None))
        down = 1 if a.zoom else a.down

        def save(name, arr, lo, hi, cmap='gray', dn=down):
            if dn > 1:
                n0, n1 = (arr.shape[0] // dn) * dn, (arr.shape[1] // dn) * dn
                arr = arr[:n0, :n1].reshape(n0 // dn, dn, n1 // dn, dn).mean((1, 3))
            plt.imsave(f'{a.figs}/{pre}{name}.png', np.clip(arr, lo, hi).T[::-1], vmin=lo, vmax=hi, cmap=cmap)

        def db(x):
            return 20 * np.log10(np.abs(x) / pk + 1e-12)

        save('img_ref', db(r1[zs]), -a.range_db, 0)
        if a.zoom:
            save('full_ref', db(r1), -a.range_db - 5, 0, dn=2)
        for tag in tags:
            if f'{tag}/p1' not in z.files or not np.isfinite(z[f'{tag}/p1']).all():
                continue
            name = tag.replace('/', '_')
            t = z[f'{tag}/p1'] * gain(tag, 'p1')
            c = M.coherence(t, r1, a.win)
            save(f'img_{name}', db(t[zs]), -a.range_db, 0)
            save(f'err_{name}', db((t - r1)[zs]), -90, -20, cmap='magma')
            save(f'coh_{name}', c[zs], 0, 1)                                              # white: coherent with float64
            save(f'loss_{name}', np.log10(1.0 - np.minimum(c[zs], 1.0) + 1e-7), -5, 0, cmap='magma')   # 1 - coherence, log scale
            if a.zoom:
                save(f'fullcoh_{name}', c, 0, 1, dn=4)
        print('wrote', len(os.listdir(a.figs)), 'files to', a.figs)


if __name__ == '__main__':
    main()
