"""FastSAR's polar format (release record, CPU) on the three Panama regions: amplitude and log-amplitude correlation
with the reference and the mean 5 by 5 coherence, as for the open-source polar formats (their images needed
registration; FastSAR's is on the reference grid).
   python -I pfa_region_scores.py <image .npy> <reference .npy> <out .json>"""
import sys, json
import numpy as np
from scipy.ndimage import uniform_filter
img = np.load(sys.argv[1], mmap_mode='r'); ref = np.load(sys.argv[2], mmap_mode='r')
crops = dict(locks=(3400, 6700), port=(9500, 1150), ships=(6400, 2650)); h = 512
out = {}
for n, (i, j) in crops.items():
    a = np.asarray(img[i:i + h, j:j + h]).astype(np.complex128); r = np.asarray(ref[i:i + h, j:j + h]).astype(np.complex128)
    def box(x):
        return uniform_filter(x.real, 5) + 1j * uniform_filter(x.imag, 5) if np.iscomplexobj(x) else uniform_filter(x, 5)
    coh = np.abs(box(a * np.conj(r))) / np.sqrt(box(np.abs(a) ** 2) * box(np.abs(r) ** 2) + 1e-300)
    la, lr = np.log10(np.abs(a) + 1e-12), np.log10(np.abs(r) + 1e-12)
    out[n] = dict(amp_corr=float(np.corrcoef(np.abs(a).ravel(), np.abs(r).ravel())[0, 1]), log_amp_corr=float(np.corrcoef(la.ravel(), lr.ravel())[0, 1]),
                  coherence_5x5_mean=float(coh.mean()))
    print(n, out[n], flush=True)
json.dump(out, open(sys.argv[3], 'w'), indent=1)
