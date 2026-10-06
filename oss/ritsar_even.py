"""RITSAR (range axis fixed) against our float64 backprojection on identical inputs: the Panama phase history with
its last frequency sample dropped (an even sample count), the same FFT length and linear interpolation."""
import sys, json, time, gc
import numpy as np
from scipy.signal.windows import taylor
W = sys.argv[1]
sys.path.insert(0, f'{W}/RITSAR_axisfix'); sys.path.insert(0, W)
from ritsar import imgTools
import cpu_ref
C = 299792458.0
d = np.load(f'{W}/panama.npz')
S = d['S'][:, :-1].astype(np.complex128); ant = d['ant'].astype(np.float64)
fmin, df = float(d['fmin']), float(d['df']); nx, ny, spx, spy = int(d['nx']), int(d['ny']), float(d['spx']), float(d['spy'])
e1, e2 = d['e1'].astype(np.float64), d['e2'].astype(np.float64)
P, K = S.shape
S *= taylor(P, nbar=4, sll=35.0, norm=False)[:, None] * taylor(K, nbar=4, sll=35.0, norm=False)[None, :]
k_r = 4 * np.pi * (fmin + np.arange(K) * df) / C
crops = dict(locks=(3400, 6700), port=(9500, 1150), ships=(6400, 2650)); h = 512
pix = np.concatenate([(((I - nx / 2.0) * spx).ravel()[:, None] * e1 + ((J - ny / 2.0) * spy).ravel()[:, None] * e2)
                      for I, J in (np.meshgrid(np.arange(i0, i0 + h), np.arange(j0, j0 + h), indexing='ij') for (i0, j0) in crops.values())])
ups = 8
N_fft = 2 ** (int(np.log2(K * ups)) + 1)
t = time.perf_counter()
rit = imgTools.backprojection(S / np.abs(k_r)[None, :], dict(nsamples=K, npulses=P, k_r=k_r, pos=ant, delta_r=C / (2 * K * df)),
                              dict(u=np.arange(len(pix)), v=np.arange(1), pixel_locs=pix.T), taylor=0, upsample=ups, prnt=False).ravel()
t_rit = time.perf_counter() - t
rm = ant[P // 2]; rit *= np.exp(-1j * k_r[K // 2] * (np.linalg.norm(rm) - np.linalg.norm(pix - rm[None, :], axis=1)))
gc.collect()
# ours, float64, same FFT length and linear interpolation
t = time.perf_counter()
r0 = np.linalg.norm(ant, axis=1); u = ant / r0[:, None]
ref = np.zeros(len(pix), complex)
for p0 in range(0, P, 1024):
    sl = slice(p0, min(P, p0 + 1024))
    rc = cpu_ref.range_compress(S[sl], N_fft)
    ref += cpu_ref.bp(rc, u[sl], r0[sl], pix[:, 0].copy(), pix[:, 1].copy(), pix[:, 2].copy(), C / (2 * df * N_fft), fmin + (K // 2) * df)
t_ref = time.perf_counter() - t
out = dict(pulses=P, samples=K, N_fft=N_fft, ritsar_seconds=t_rit, ours_numba_seconds=t_ref, errors={})
for k, n in enumerate(crops):
    a = rit[k * h * h:(k + 1) * h * h]; b = ref[k * h * h:(k + 1) * h * h]
    g = np.vdot(a, b) / np.vdot(a, a)
    out['errors'][n] = float(10 * np.log10(np.sum(np.abs(g * a - b) ** 2) / np.sum(np.abs(b) ** 2)))
print(json.dumps(out, indent=1)); json.dump(out, open(f'{W}/ritsar_even.json', 'w'), indent=1)
