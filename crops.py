#!/usr/bin/env python3
"""Cut named regions out of the reference and chosen test images of a scene, with the local coherence map of
each test image, for the zoom figures.

  python crops.py --ref out/panama/ref --tests out/panama/gpu-l4/bp_cuda_f16.npy,out/panama/tpu-v6e/ffbp_fp32_fast.npy \
      --crops "locks:3000,4000,256,256;ship:..." --out results/fastsar/figs/panama_crops.npz
"""
import argparse
import os

import numpy as np
from scipy.ndimage import uniform_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--tests', required=True)
    ap.add_argument('--crops', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--win', type=int, default=5)
    ap.add_argument('--margin', type=int, default=8)
    a = ap.parse_args()
    ref = np.load(f'{a.ref}/bp_numba_fp64.npy', mmap_mode='r')
    boxes = []
    for c in [c for c in a.crops.split(';') if c]:
        name, box = c.split(':')
        boxes.append((name, *[int(v) for v in box.split(',')]))
    m = a.margin
    out = {}
    for name, i0, j0, h, w in boxes:
        r = np.array(ref[i0 - m:i0 + h + m, j0 - m:j0 + w + m])
        out[f'{name}/ref'] = r[m:-m, m:-m].astype(np.complex64)
        pr = uniform_filter(np.abs(r) ** 2, a.win)
        for path in a.tests.split(','):
            label = os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)[:-4]
            t = np.array(np.load(path, mmap_mode='r')[i0 - m:i0 + h + m, j0 - m:j0 + w + m]).astype(np.complex128)
            g = float(np.sqrt(np.vdot(r, r).real / np.vdot(t, t).real))   # amplitude match; phase screens leave it alone
            t = t * g
            x = t * np.conj(r)
            num = uniform_filter(x.real, a.win) + 1j * uniform_filter(x.imag, a.win)
            coh = np.abs(num) / np.sqrt(np.maximum(uniform_filter(np.abs(t) ** 2, a.win) * pr, 1e-300))
            out[f'{name}/{label}'] = t[m:-m, m:-m].astype(np.complex64)
            out[f'{name}/{label}/coh'] = coh[m:-m, m:-m].astype(np.float32)
            out[f'{name}/{label}/phase'] = np.angle(num)[m:-m, m:-m].astype(np.float32)
    np.savez_compressed(a.out, **out)
    print('saved', len(out), 'arrays to', a.out)


if __name__ == '__main__':
    main()
