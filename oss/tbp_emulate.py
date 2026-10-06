import sys, numpy as np, torch
sys.argv = ['x', sys.argv[1], '/dev/null']
import os; os.environ['DEV'] = 'cpu'; os.environ['QUICK'] = '1'
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torchbp_panama.py')).read().split("for os_ in")[0]
exec(src)
data, nfft, base = rc_gpu(2)
dr = C / (2 * df * nfft); d0 = -Rbar + (nfft // 2 + base) * dr
rc = data.numpy()
(i0, j0), = crops.values()
I, J = np.meshgrid(np.arange(i0, i0 + h), np.arange(j0, j0 + h), indexing='ij')
X = ((J - ny / 2.0) * spy).ravel(); Y = ((I - nx / 2.0) * spx).ravel()
ref = np.asarray(ref_img[i0:i0 + h, j0:j0 + h])
for dt in (np.float64, np.float32):
    acc = np.zeros(len(X), np.complex128)
    x, y = X.astype(dt), Y.astype(dt)
    for p in range(P):
        px, py, pz = (dt(v) for v in posf[p])
        d = np.sqrt((x - px) ** 2 + (y - py) ** 2 + pz * pz)
        sx = (d + dt(d0)) / dt(dr)
        i = np.floor(sx).astype(np.int64); f = (sx - i).astype(np.float64)
        v = rc[p, i] * (1 - f) + rc[p, i + 1] * f
        acc += v * np.exp(1j * np.pi * (4 * fref / C * d.astype(np.float64) if dt is np.float64 else (dt(4 * fref / C) * d).astype(np.float64)))
    print(dt.__name__, 'emulation of torchbp formula vs reference:', round(err(acc.reshape(h, h), ref), 1), 'dB', flush=True)
