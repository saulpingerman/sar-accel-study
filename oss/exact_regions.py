"""FastSAR's exact backprojection (ExactFormer, public API, default cubic interpolation) on the three Panama regions
(lock, port, ship; 512 by 512 pixels of the vendor grid) on the CPU: the warm time and the error of each region
against the reference, and the full-image estimate that scales the lock region's time by the number of pixels (the
method of the open-source estimates).
   python -I exact_regions.py <panama .npz> <reference .npy> <out .json>"""
import sys, json, time, os, subprocess
import numpy as np
import fastsar
d = np.load(sys.argv[1]); ref = np.load(sys.argv[2], mmap_mode='r')
S, ant, fmin, df = d['S'], d['ant'], float(d['fmin']), float(d['df'])
nx, ny, spx, spy, e1, e2 = int(d['nx']), int(d['ny']), float(d['spx']), float(d['spy']), np.asarray(d['e1'], float), np.asarray(d['e2'], float)
crops = dict(locks=(3400, 6700), port=(9500, 1150), ships=(6400, 2650)); h = 512
out = dict(fastsar=subprocess.run(['git', '-C', os.path.dirname(fastsar.__path__[0]), 'log', '-1', '--format=%H'], capture_output=True, text=True).stdout.strip(),
           api="ExactFormer(..., 512, 512, center=<region center>)", regions={})
for n, (i, j) in crops.items():
    c = (i + h // 2 - nx / 2) * spx * e1 + (j + h // 2 - ny / 2) * spy * e2       # pixel (i, j) lies at (i - nx/2) spx e1 + (j - ny/2) spy e2
    f = fastsar.ExactFormer(ant, fmin, df, S.shape[1], h, h, spx, spy, e1, e2, backend='cpu', center=c)
    f(S)
    t0 = time.perf_counter(); img = f(S); t = time.perf_counter() - t0
    r = np.asarray(ref[i:i + h, j:j + h]).astype(np.complex128); a = img.astype(np.complex128)
    g = np.vdot(a, r) / np.vdot(a, a)
    e = float(10 * np.log10(np.sum(np.abs(g * a - r) ** 2) / np.sum(np.abs(r) ** 2)))
    out['regions'][n] = dict(seconds=t, error_db=e)
    print(n, f'{t:.2f} s, {e:.2f} dB', flush=True)
out['full_image_estimate_s_from_lock'] = out['regions']['locks']['seconds'] * nx * ny / h ** 2
json.dump(out, open(sys.argv[3], 'w'), indent=1)
