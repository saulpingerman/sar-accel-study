"""FastSAR against torchbp on one simulated spotlight scene (broadside track along y, looking along +x).

    python -I bench_gpu.py <image side n> <range r0 (m)> <device cpu|cuda> <out.json> [fastsar_root]

Every image is scored against a float64 exact backprojection evaluated on sampled pixels (a 64 x 64 window around each
of five targets plus 8,000 random pixels), after one complex gain fit.
"""
import os, sys, time, json
import numpy as np, torch, torchbp
from scipy.signal.windows import taylor
n, r0_, dev, out = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
root = sys.argv[5] if len(sys.argv) > 5 else '../FastSAR'
sys.path.insert(0, root)
import fastsar
from fastsar import sim
from fastsar.ffbp import C
res = 0.5
rng = np.random.default_rng(7)
col = sim.make_collect(res=res, scene=n * res, r0=r0_)
half = n * res / 2
tg = np.stack([rng.uniform(-0.8 * half, 0.8 * half, 60), rng.uniform(-0.8 * half, 0.8 * half, 60), np.zeros(60)], 1)
amp = rng.standard_normal(60) + 1j * rng.standard_normal(60)
t = time.perf_counter()
S = np.zeros((col.ant.shape[0], col.K), np.complex64)
for k_ in range(len(amp)):                                     # one target at a time (memory)
    S += sim.simulate_brute(col, tg[k_:k_ + 1], amp[k_:k_ + 1]).astype(np.complex64)
t_sim = time.perf_counter() - t
P, K = S.shape
print(f'n {n}, range {r0_:.0f} m: pulses {P}, samples {K}, simulated in {t_sim:.0f} s', flush=True)
Sw = (S * taylor(P, nbar=4, sll=35, norm=False)[:, None].astype(np.float32) * taylor(K, nbar=4, sll=35, norm=False)[None, :].astype(np.float32)).astype(np.complex64)
del S
e1, e2 = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]); en = np.cross(e1, e2)
spx = spy = res
# sampled pixels: windows around five targets and random pixels; index (i, j) with x_i = (i - n/2) spx along e1
sel = []
for k in range(5):
    ci = int(round(tg[k] @ e1 / spx + n / 2)); cj = int(round(tg[k] @ e2 / spy + n / 2))
    I, J = np.meshgrid(np.arange(ci - 32, ci + 32), np.arange(cj - 32, cj + 32), indexing='ij'); sel.append(np.stack([I.ravel(), J.ravel()], 1))
sel.append(rng.integers(0, n, (8000, 2)))
sel = np.clip(np.concatenate(sel), 0, n - 1)
pix = ((sel[:, 0] - n / 2.0) * spx)[:, None] * e1 + ((sel[:, 1] - n / 2.0) * spy)[:, None] * e2
# float64 reference: exact backprojection, 16x oversampled range profiles, on the sampled pixels only
def range_compress(Sx, nfft):
    Kx = Sx.shape[1]; h = Kx // 2
    pad = np.zeros((Sx.shape[0], nfft), np.complex128); pad[:, :Kx - h] = Sx[:, h:]; pad[:, nfft - h:] = Sx[:, :h]
    return np.fft.fftshift(np.fft.ifft(pad, axis=1), axes=1)
nfft = 1 << int(np.ceil(np.log2(16 * K)))
dr = C / (2 * col.df * nfft)
fref = col.fmin + (K // 2) * col.df
r0 = np.linalg.norm(col.ant, axis=1)
def ref_bp():
    out_ = np.zeros(len(pix), np.complex128)
    for p0 in range(0, P, 256):
        rc = range_compress(Sw[p0:p0 + 256].astype(np.complex128), nfft)
        for k in range(rc.shape[0]):
            a = col.ant[p0 + k]
            dR = np.linalg.norm(pix - a, axis=1) - r0[p0 + k]
            tt = dR / dr + nfft // 2
            i0 = np.floor(tt).astype(int); w = tt - i0; ok = (i0 >= 0) & (i0 < nfft - 1); i0 = np.clip(i0, 0, nfft - 2)
            v = rc[k, i0] * (1 - w) + rc[k, i0 + 1] * w
            out_ += np.where(ok, v * np.exp(1j * 4 * np.pi * fref / C * dR), 0)
    return out_
t = time.perf_counter(); ref = ref_bp(); t_ref = time.perf_counter() - t
print(f'reference on {len(pix)} pixels in {t_ref:.0f} s', flush=True)
def score(img):
    img = np.asarray(img).reshape(n, n)
    v = img[sel[:, 0], sel[:, 1]].astype(np.complex128)
    a = np.vdot(v, ref) / np.vdot(v, v)
    return float(10 * np.log10(np.sum(np.abs(a * v - ref) ** 2) / np.sum(np.abs(ref) ** 2)))
def timed(fn, sync, reps=2):
    fn(); sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); r = fn(); sync(); ts.append(time.perf_counter() - t)
    return r, min(ts)
results = dict(n=n, range_m=r0_, pulses=P, samples=K, device=dev, methods={})
sync = (lambda: torch.cuda.synchronize()) if dev == 'cuda' else (lambda: None)
# torchbp: data re-referenced from the scene center to one common range Rbar, positions in the image-plane frame
Rbar = float(r0.mean())
f = col.fmin + np.arange(K) * col.df
nfft_tb = 1 << int(np.ceil(np.log2(4 * K)))                    # 4x oversampled profiles for torchbp's linear interpolation
dr_tb = C / (2 * col.df * nfft_tb)
rc_tb = np.empty((P, nfft_tb), np.complex64)
for p0 in range(0, P, 512):
    blk = Sw[p0:p0 + 512].astype(np.complex128) * np.exp(-1j * 4 * np.pi * f[None, :] * (r0[p0:p0 + 512] - Rbar)[:, None] / C)
    rc_tb[p0:p0 + 512] = range_compress(blk, nfft_tb)
zup = np.cross(e2, e1)
posf = np.stack([col.ant @ e2, col.ant @ e1, col.ant @ zup], 1)
ONLY = os.environ.get('ONLY', 'torch_exact,torch_ffbp,fastsar').split(',')
data_t = torch.tensor(rc_tb, device=dev) if ('torch_exact' in ONLY or 'torch_ffbp' in ONLY) else None; pos_t = torch.tensor(posf, dtype=torch.float32, device=dev)
grid = {"x": (-n / 2.0 * spy, n / 2.0 * spy), "y": (-n / 2.0 * spx, n / 2.0 * spx), "nx": n, "ny": n}
d0 = -Rbar + (nfft_tb // 2) * dr_tb
ramp_t = torch.tensor(np.exp(-1j * 4 * np.pi * f[None, :] * (r0 - Rbar)[:, None] / C).astype(np.complex64), device=dev)   # per-collection setup
def rc_torch(S_host, blk=1024):
    out_ = torch.empty((P, nfft_tb), dtype=torch.complex64, device=dev)
    h = K // 2
    for p0 in range(0, P, blk):                                  # in pulse blocks (memory)
        St = torch.tensor(S_host[p0:p0 + blk], device=dev) * ramp_t[p0:p0 + blk]
        pad = torch.zeros((St.shape[0], nfft_tb), dtype=torch.complex64, device=dev)
        pad[:, :K - h] = St[:, h:]; pad[:, nfft_tb - h:] = St[:, :h]
        out_[p0:p0 + blk] = torch.fft.fftshift(torch.fft.ifft(pad, dim=1), dim=1)
    return out_
if 'torch_exact' in ONLY:
  img, secs = timed(lambda: torchbp.ops.backprojection_cart_2d(data_t, grid, fref, dr_tb, pos_t, d0=d0), sync, reps=1)
  img = img.cpu(); torch.cuda.empty_cache() if dev == 'cuda' else None
  img2, secs_e2e = timed(lambda: torchbp.ops.backprojection_cart_2d(rc_torch(Sw), grid, fref, dr_tb, pos_t, d0=d0).cpu(), sync, reps=1)
  results['methods']['torchbp exact backprojection (cartesian)'] = dict(seconds=secs_e2e, kernel_seconds=secs, error_db=score(img.cpu().numpy().reshape(n, n).T),
                                                                           error_db_gpu_range_compression=score(img2.numpy().reshape(n, n).T))
  del img2
  print('torchbp cart', results['methods']['torchbp exact backprojection (cartesian)'], flush=True); del img
# torchbp fast factorized backprojection: polar grid about the aperture-center ground point, then its polar_to_cart
try:
    if 'torch_ffbp' not in ONLY:
        raise StopIteration
    c = posf.mean(0); org = np.array([c[0], c[1], 0.0])          # polar origin: the ground point below the aperture center
    pos_p = torch.tensor(posf - org, dtype=torch.float32, device=dev)
    cx = -org[0]; cy = -org[1]                                    # scene center relative to the polar origin
    gr = float(np.hypot(cx, cy)); rmin = gr - 0.75 * n * spx; rmax = gr + 0.75 * n * spx
    ang = float(np.arctan2(cy, cx)); w = 0.75 * n * spx / gr
    nr = int(1.5 * n * spx / (res / 2)); nth = int(2 * w * gr / (res / 2))           # polar grid sampled at res/2 (2x oversampled), torchbp's best setting here
    gpol = {"r": (rmin, rmax), "theta": (np.sin(-w), np.sin(w)), "nr": nr, "ntheta": nth}
    gcart = {"x": (-n / 2.0 * spy - org[0], n / 2.0 * spy - org[0]), "y": (-n / 2.0 * spx - org[1], n / 2.0 * spx - org[1]), "nx": n, "ny": n}
    def run_ffbp(e2e=False):
        ip = torchbp.ops.ffbp(rc_torch(Sw) if e2e else data_t, gpol, fref, dr_tb, pos_p, stages=8, d0=d0, dealias=True, grid_oversample=2.0)
        return torchbp.ops.polar_to_cart(ip.reshape(1, *ip.shape[-2:]), torch.tensor([[0.0, 0.0, float(posf[:, 2].mean())]], device=dev), gpol, gcart, fref, ang, method=("lanczos", 6))
    img, secs = timed(run_ffbp, sync, reps=1)
    img = img.cpu(); torch.cuda.empty_cache() if dev == 'cuda' else None
    _, secs_e2e = timed(lambda: run_ffbp(True).cpu(), sync, reps=1)
    results['methods']['torchbp fast factorized backprojection (+ polar_to_cart)'] = dict(seconds=secs_e2e, kernel_seconds=secs, error_db=score(img.cpu().numpy().reshape(n, n).T), polar_grid=[nr, nth])
    ipe, secs_e = timed(lambda: torchbp.ops.polar_to_cart(torchbp.ops.backprojection_polar_2d(data_t, gpol, fref, dr_tb, pos_p, d0=d0, dealias=True).reshape(1, nr, nth), torch.tensor([[0.0, 0.0, float(posf[:, 2].mean())]], device=dev), gpol, gcart, fref, ang, method=("lanczos", 6)), sync, reps=1)
    results['methods']['torchbp exact backprojection (polar + polar_to_cart)'] = dict(seconds=secs_e, error_db=score(ipe.cpu().numpy().reshape(n, n).T))
    print('torchbp polar exact', results['methods']['torchbp exact backprojection (polar + polar_to_cart)'], flush=True); del ipe
    print('torchbp ffbp', results['methods']['torchbp fast factorized backprojection (+ polar_to_cart)'], flush=True); del img
except StopIteration:
    pass
except Exception as e:
    results['methods']['torchbp fast factorized backprojection (+ polar_to_cart)'] = dict(error=f'{type(e).__name__}: {str(e)[:300]}')
    print('torchbp ffbp failed', type(e).__name__, str(e)[:300], flush=True)
del data_t
# FastSAR
for backend, prec, plan in ([] if 'fastsar' not in ONLY else [('cuda', 'float32', (32, 3)), ('cuda', 'float16', (32, 3)), ('cuda', 'float32', ('auto', 3))] if dev == 'cuda' else [('cpu', 'float32', (32, 3)), ('cpu', 'float32', ('auto', 3))]):
    try:
        g = dict(S=Sw, ant=col.ant, fmin=col.fmin, df=col.df, nx=n, ny=n, spx=spx, spy=spy, e1=e1, e2=e2)
        torch.cuda.empty_cache() if dev == 'cuda' else None
        if prec == 'float16' and plan[0] != 32:
            continue
        t = time.perf_counter()
        former = fastsar.ImageFormer(col.ant, col.fmin, col.df, Sw.shape[1], n, n, spx, spy, e1, e2, backend=backend, precision=prec, window=False, T=plan[0], levels=plan[1])
        t_setup = time.perf_counter() - t
        img, secs = timed(lambda: former(Sw), lambda: None, reps=2)
        key = f'FastSAR {backend} {prec} (T={former.T}{" auto" if plan[0] == "auto" else ""}, {plan[1]} levels)'
        results['methods'][key] = dict(seconds=secs, setup_seconds=t_setup, error_db=score(img))
        print(key, results['methods'][key], flush=True); del img
    except Exception as e:
        results['methods'][f'FastSAR {backend} {prec} (T={plan[0]}, {plan[1]} levels)'] = dict(error=f'{type(e).__name__}: {str(e)[:300]}')
        print('fastsar failed', backend, prec, type(e).__name__, str(e)[:300], flush=True)
json.dump(results, open(out, 'w'), indent=1)
