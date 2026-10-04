#!/usr/bin/env python3
"""Image panels for the simulated scenes, one set per precision setting, all
on fixed display scales so settings can be compared with each other.

  python figs.py --images q_air_cpu.npz --out figs/air

Per setting: the formed image (pass 1), its difference from the float64 image
(amplified, fixed scale), the change map between the two passes, and the
response to one point target. One scale factor is used per algorithm, never
one per image.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sarbench import metrics as M


def save(path, arr, vmin, vmax, cmap='gray', down=2):
    if down > 1:  # block mean
        n0, n1 = (arr.shape[0] // down) * down, (arr.shape[1] // down) * down
        arr = arr[:n0, :n1].reshape(n0 // down, down, n1 // down, down).mean((1, 3))
    plt.imsave(path, np.clip(arr, vmin, vmax).T[::-1], vmin=vmin, vmax=vmax, cmap=cmap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tags', default='bp/fp64,bp/fp32,bp/fp32_naive,bp/bf16_store,bp/bf16_arith,bp/bf16_acc,bp/bf16_acc_seq,bp/bf16_all,'
                                      'ffbp/fp32,ffbp/bf16_mm,ffbp/bf16,ffbp/f16,ffbp/f8_mm,ffbp/f4_mm')
    ap.add_argument('--win', type=int, default=11)
    ap.add_argument('--range-db', type=float, default=40.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    z = np.load(a.images)
    ref1, ref2 = z['bp/fp64/p1'], z['bp/fp64/p2']
    peak = np.percentile(np.abs(ref1), 99.5)
    gains = {}

    def gain(tag):
        fam = tag.split('/')[0]
        if fam == 'bp':
            return 1.0
        if fam not in gains:
            src = z[f'{fam}/fp64/p1'] if f'{fam}/fp64/p1' in z.files else z[f'{fam}/fp32/p1']
            gains[fam] = np.vdot(src, ref1) / np.vdot(src, src)
        return gains[fam]

    def db(x):
        return 20 * np.log10(np.abs(x) / peak + 1e-12)

    save(f'{a.out}/img_pass2.png', db(ref2), -a.range_db, 0)
    for tag in a.tags.split(','):
        if f'{tag}/p1' not in z.files:
            continue
        name = tag.replace('/', '_')
        g = gain(tag)
        p1, p2 = z[f'{tag}/p1'] * g, z[f'{tag}/p2'] * g
        save(f'{a.out}/img_{name}.png', db(p1), -a.range_db, 0)
        save(f'{a.out}/err_{name}.png', db(p1 - ref1), -90, -20, cmap='magma')
        save(f'{a.out}/ccd_{name}.png', M.coherence(p1, p2, a.win), 0, 1)
        pts = z[f'{tag}/pts'] * g
        rp = z['bp/fp64/pts']
        n = pts.shape[0]
        r, c = np.unravel_index(np.argmax(np.abs(rp[n // 2 - 40:n // 2 + 40, n // 2 - 40:n // 2 + 40])), (80, 80))
        r, c = r + n // 2 - 40, c + n // 2 - 40
        up = M._upsample(pts[r - 32:r + 32, c - 32:c + 32], 4)
        upr = M._upsample(rp[r - 32:r + 32, c - 32:c + 32], 4)
        # every point-target panel is scaled to the float64 peak, so a weak or smeared response looks weak
        save(f'{a.out}/irf_{name}.png', 20 * np.log10(np.abs(up) / np.abs(upr).max() + 1e-9), -60, 0, down=1)
    print('wrote', len(os.listdir(a.out)), 'files to', a.out)


if __name__ == '__main__':
    main()
