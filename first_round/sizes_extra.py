#!/usr/bin/env python3
"""Two further comparisons for the image-size series, per test image:

  local_err_db   error against the float64 reference after one complex factor has been
                 fitted per block of --block pixels, which removes any difference that is
                 smooth over the image and leaves what a coherence window would see
  vs_fp32_db     energy of the difference from the factorized float32 image formed on the
                 same device (the CPU image for emulated rows), which isolates the arithmetic
                 from the approximations of the factorized algorithm

Also written: statistics of the blockwise factor of the factorized float32 image (rms phase
in degrees and rms amplitude deviation), which describe the smooth difference itself.

  python sizes_extra.py --ref out/sz8192_cpu --test out/sz8192_cpu,out/sz8192_tpu-v5e --out results/sizes/extra8192.json
"""
import argparse
import glob
import json
import os

import numpy as np


def blocks(x, b):
    n = (x.shape[0] // b) * b
    return x[:n, :n].reshape(n // b, b, n // b, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--block', type=int, default=256)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ref = np.load(f'{a.ref}/bp_numba_fp64.npy')
    n = ref.shape[0]
    rb = blocks(ref, a.block)
    e_ref = float(np.sum(np.abs(rb) ** 2))
    out = dict(n=n, block=a.block, rows=[])
    for d in a.test.split(','):
        meta = json.load(open(f'{d}/timing.json')) if os.path.exists(f'{d}/timing.json') else {}
        label = meta.get('_meta', {}).get('label', os.path.basename(d))
        base = np.load(f'{d}/ffbp_fp32.npy') if os.path.exists(f'{d}/ffbp_fp32.npy') else None
        for path in sorted(glob.glob(f'{d}/*.npy')):
            name = os.path.basename(path)[:-4]
            if name == 'bp_numba_fp64':
                continue
            t = np.load(path)
            tb = blocks(t, a.block)
            g = np.einsum('iajb,iajb->ij', np.conj(tb), rb) / np.maximum(np.einsum('iajb,iajb->ij', np.conj(tb), tb).real, 1e-300)
            res = float(np.sum(np.abs(tb * g[:, None, :, None] - rb) ** 2))
            row = dict(label=label, tag=name.replace('_', '/', 1), n=n, local_err_db=float(10 * np.log10(res / e_ref + 1e-300)))
            w = np.einsum('iajb->ij', np.abs(rb) ** 2)
            g0 = np.sum(g * w) / np.sum(w)
            rel = g / g0
            row['factor_phase_rms_deg'] = float(np.degrees(np.sqrt(np.sum(w * np.angle(rel) ** 2) / np.sum(w))))
            row['factor_amp_rms'] = float(np.sqrt(np.sum(w * (np.abs(rel) - 1) ** 2) / np.sum(w)))
            if base is not None and name.startswith('ffbp') and name != 'ffbp_fp32':
                row['vs_fp32_db'] = float(10 * np.log10(np.sum(np.abs(t - base) ** 2) / np.sum(np.abs(base) ** 2) + 1e-300))
            out['rows'].append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            del t, tb
        del base
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
