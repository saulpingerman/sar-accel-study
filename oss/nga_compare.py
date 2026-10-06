"""Score NGA's bpBasic images against our float64 reference (sim: computed here; Panama: the study's reference)."""
import sys, json, glob, numpy as np
from scipy.signal.windows import taylor
W = sys.argv[1]
sys.path.insert(0, f'{W}/FastSAR')
C = 299792458.0
def err(test, ref):
    a = np.vdot(test.ravel(), ref.ravel()) / np.vdot(test.ravel(), test.ravel())
    return float(10 * np.log10(np.sum(np.abs(a * test - ref) ** 2) / np.sum(np.abs(ref) ** 2)))
def read(name, nfft, N):
    return np.fromfile(f'{W}/{name}_nga_{nfft}.f32', np.float32).reshape(N, 2) @ np.array([1, 1j])
out = {}
# simulated scene: our float64 exact backprojection, 16x profiles, computed in NumPy
g = np.load(f'{W}/sim_geom.npz'); S = g['S']; ant = g['ant']; fmin, df, K = float(g['fmin']), float(g['df']), int(g['K'])
P = S.shape[0]
Sw = S * taylor(P, nbar=4, sll=35.0, norm=False)[:, None] * taylor(K, nbar=4, sll=35.0, norm=False)[None, :]
pix = np.fromfile(f'{W}/sim_pix.f64').reshape(-1, 3)
nf = 1 << int(np.ceil(np.log2(16 * K))); h = K // 2
pad = np.zeros((P, nf), complex); pad[:, :K - h] = Sw[:, h:]; pad[:, nf - h:] = Sw[:, :h]
rc = np.fft.fftshift(np.fft.ifft(pad, axis=1), axes=1); dr = C / (2 * df * nf); fref = fmin + h * df
ref = np.zeros(len(pix), complex)
for p in range(P):
    dR = np.linalg.norm(pix - ant[p], axis=1) - np.linalg.norm(ant[p]); t = dR / dr + nf // 2
    i0 = np.floor(t).astype(int); w = t - i0
    ref += (rc[p, i0] * (1 - w) + rc[p, i0 + 1] * w) * np.exp(1j * 4 * np.pi * fref / C * dR)
for fn in sorted(glob.glob(f'{W}/sim_nga_*.f32')):
    nfft = int(fn.split('_')[-1].split('.')[0]); img = read('sim', nfft, len(pix))
    out[f'sim, Nfft {nfft or "default"}'] = dict(error_db=err(img, ref), conj_error_db=err(np.conj(img), ref))
# Panama: the study's reference and kernel images, three 512 x 512 regions
names = {'reference (float64 exact BP)': 'ref.npy', 'CPU C++ kernels': 'cpp.npy', 'L4 CUDA float32': 'cuda32.npy', 'TPU v6e three-pass': 'v6e3.npy'}
mm = {k: np.load(f'{W}/{v}', mmap_mode='r') for k, v in names.items()}
crops = (('locks', 3400, 6700), ('port', 9500, 1150), ('ships', 6400, 2650))
for fn in sorted(glob.glob(f'{W}/panama_nga_*.f32')):
    nfft = int(fn.split('_')[-1].split('.')[0]); img = read('panama', nfft, 3 * 512 * 512)
    secs = float(open(fn.replace('.f32', '.txt')).read())
    rec = dict(seconds=secs, errors={})
    for k, (n, i0, j0) in enumerate(crops):
        r = img[k * 512 * 512:(k + 1) * 512 * 512].reshape(512, 512)
        rec['errors'][n] = {nm: err(np.asarray(a[i0:i0 + 512, j0:j0 + 512]).astype(complex), r) for nm, a in mm.items()}
    out[f'panama, Nfft {nfft or "default"}'] = rec
json.dump(out, open(f'{W}/nga_results.json', 'w'), indent=1); print(json.dumps(out, indent=1))
