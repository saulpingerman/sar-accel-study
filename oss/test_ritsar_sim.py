"""RITSAR backprojection (CPU, float64 NumPy) against our float64 reference and our C++ kernels, simulated scene."""
import os, sys, time
sys.path.insert(0, os.environ.get('RITSAR', 'RITSAR_py3'))          # a RITSAR checkout
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.signal.windows import taylor
from ritsar import imgTools
from dev import sim, cpu_ref, bp as bpmod
C = 299792458.0
rng = np.random.default_rng(1)
col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
pos = np.stack([rng.uniform(-25, 25, 40), rng.uniform(-25, 25, 40), np.zeros(40)], 1)
amp = rng.standard_normal(40) + 1j * rng.standard_normal(40)
S = sim.simulate_brute(col, pos, amp).astype(np.complex128)
P, K = S.shape
wp = taylor(P, nbar=4, sll=35, norm=False); wk = taylor(K, nbar=4, sll=35, norm=False)
Sw = S * wp[:, None] * wk[None, :]
nx = ny = 128; spx = spy = 0.5
e1, e2 = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
X, Y = np.meshgrid((np.arange(nx) - nx / 2.0) * spx, (np.arange(ny) - ny / 2.0) * spy, indexing='ij')
pix = X.ravel()[:, None] * e1 + Y.ravel()[:, None] * e2                    # [nx*ny, 3], i-major
# ours: float64 exact backprojection, 16x oversampled profiles
nfft = 1 << int(np.ceil(np.log2(16 * K)))
rc = cpu_ref.range_compress(Sw, nfft)
r0 = np.linalg.norm(col.ant, axis=1); u = col.ant / r0[:, None]
fref = col.fmin + (K // 2) * col.df
t = time.perf_counter()
ref = cpu_ref.bp(rc, u, r0, pix[:, 0].copy(), pix[:, 1].copy(), pix[:, 2].copy(), bpmod.range_bin(col.df, nfft), fref).reshape(nx, ny)
t_ref = time.perf_counter() - t
# RITSAR: same windowed phase history; its |k_r| filter divided out, its window disabled (taylor=0)
k_r = 4 * np.pi * (col.fmin + np.arange(K) * col.df) / C
platform = dict(nsamples=K, npulses=P, k_r=k_r, pos=col.ant, delta_r=C / (2 * K * col.df))
img_plane = dict(u=np.arange(ny), v=np.arange(nx), pixel_locs=pix.T)
for ups in (6, 16):
    t = time.perf_counter()
    rit = imgTools.backprojection(Sw / np.abs(k_r)[None, :], platform, img_plane, taylor=0, upsample=ups, prnt=False)[::-1, :]
    t_rit = time.perf_counter() - t
    for name, img in (('as is', rit), ('conj', np.conj(rit))):
        a = np.vdot(img.ravel(), ref.ravel()) / np.vdot(img.ravel(), img.ravel())
        e = 10 * np.log10(np.sum(np.abs(a * img - ref) ** 2) / np.sum(np.abs(ref) ** 2))
        print(f'RITSAR BP upsample {ups} ({name}) vs our float64 reference: {e:.1f} dB   time {t_rit:.1f} s (ours {t_ref:.1f} s)')
