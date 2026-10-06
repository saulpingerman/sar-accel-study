"""RITSAR's backprojection (CPU, float64 NumPy) on three Panama regions of our pixel grid, against our float64
reference and our kernel images.   python -I ritsar_panama.py <work dir>"""
import sys, json, time, importlib, gc
import numpy as np
from scipy.signal.windows import taylor
W = sys.argv[1]
C = 299792458.0
d = np.load(f'{W}/panama.npz')
S = d['S'].astype(np.complex128); ant = d['ant'].astype(np.float64)
fmin, df = float(d['fmin']), float(d['df']); nx, ny = int(d['nx']), int(d['ny']); spx, spy = float(d['spx']), float(d['spy'])
e1, e2 = d['e1'].astype(np.float64), d['e2'].astype(np.float64)
P, K = S.shape
S *= taylor(P, nbar=4, sll=35.0, norm=False)[:, None] * taylor(K, nbar=4, sll=35.0, norm=False)[None, :]
k_r = 4 * np.pi * (fmin + np.arange(K) * df) / C
S /= np.abs(k_r)[None, :]                                  # RITSAR multiplies by |k_r|; divided out so the weighting matches
crops = dict(locks=(3400, 6700), port=(9500, 1150), ships=(6400, 2650)); h = 512
pix, idx = [], {}
for n, (i0, j0) in crops.items():
    I, J = np.meshgrid(np.arange(i0, i0 + h), np.arange(j0, j0 + h), indexing='ij')
    x, y = (I - nx / 2.0) * spx, (J - ny / 2.0) * spy
    idx[n] = (len(pix) and sum(p.shape[0] for p in pix)) or 0
    pix.append(x.ravel()[:, None] * e1 + y.ravel()[:, None] * e2)
pix = np.concatenate(pix)                                    # [3 * 512 * 512, 3]
platform = dict(nsamples=K, npulses=P, k_r=k_r, pos=ant, delta_r=C / (2 * K * df))
img_plane = dict(u=np.arange(pix.shape[0]), v=np.arange(1), pixel_locs=pix.T)
rm = ant[P // 2]; k_c = k_r[K // 2]
undo = np.exp(-1j * k_c * (np.linalg.norm(rm) - np.linalg.norm(pix - rm[None, :], axis=1)))   # RITSAR's final mid-pulse phase
mm = {n: np.load(f'{W}/{f}', mmap_mode='r') for n, f in (('reference (float64 exact BP)', 'ref.npy'), ('CPU C++ kernels', 'cpp.npy'),
      ('L4 CUDA float32', 'cuda32.npy'), ('L4 CUDA float16', 'cuda16.npy'), ('TPU v6e three-pass', 'v6e3.npy'), ('TPU v6e single-pass', 'v6e1.npy'))}
def crop(a, n):
    i0, j0 = crops[n]; return np.asarray(a[i0:i0 + h, j0:j0 + h]).astype(np.complex128)
def err(test, ref):
    a = np.vdot(test.ravel(), ref.ravel()) / np.vdot(test.ravel(), test.ravel())
    return float(10 * np.log10(np.sum(np.abs(a * test - ref) ** 2) / np.sum(np.abs(ref) ** 2)))
out = dict(pixels=int(pix.shape[0]), pulses=P, samples=K, runs={})
saved = {}
for variant, mod, ups in (('RITSAR as published (Python 3 port), 6x', 'RITSAR_py3', 6), ('RITSAR as published (Python 3 port), 16x', 'RITSAR_py3', 16),
                           ('RITSAR with exact FFT range axis, 16x', 'RITSAR_axisfix', 16)):
    sys.path.insert(0, f'{W}/{mod}')
    for k in [k for k in sys.modules if k.startswith('ritsar')]:
        del sys.modules[k]
    from ritsar import imgTools
    t = time.perf_counter()
    img = imgTools.backprojection(S, platform, img_plane, taylor=0, upsample=ups, prnt=False).ravel() * undo
    secs = time.perf_counter() - t
    sys.path.pop(0); gc.collect()
    rec = dict(seconds=secs, seconds_per_pulse_per_mpixel=secs / P / (pix.shape[0] / 1e6), errors={})
    for n in crops:
        r = img[idx[n]:idx[n] + h * h].reshape(h, h)
        saved[f'{variant}/{n}'] = r.astype(np.complex64)
        rec['errors'][n] = {name: err(crop(a, n), r) for name, a in mm.items()}
    out['runs'][variant] = rec
    print(variant, json.dumps(rec, indent=1), flush=True)
    del img; gc.collect()
# ours against each other on the same crops, for scale
out['ours_vs_reference'] = {n: {name: err(crop(a, n), crop(mm['reference (float64 exact BP)'], n)) for name, a in mm.items() if not name.startswith('reference')} for n in crops}
out['full_image_estimate_hours_6x'] = out['runs']['RITSAR as published (Python 3 port), 6x']['seconds_per_pulse_per_mpixel'] * P * (nx * ny / 1e6) / 3600
json.dump(out, open(f'{W}/ritsar_panama.json', 'w'), indent=1)
np.savez_compressed(f'{W}/ritsar_panama_crops.npz', **{k.replace(' ', '_'): v for k, v in saved.items()})
print(json.dumps(out['ours_vs_reference'], indent=1)); print('full image estimate (h, 6x):', out['full_image_estimate_hours_6x'])
