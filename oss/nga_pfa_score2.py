"""NGA pfa_mem image against the reference on the three Panama regions, as the vendor image of Appendix F: each
region registered by amplitude cross-correlation (applied in the pfa image's own pixel coordinates), then amplitude
correlation, log-amplitude correlation and the mean 5 by 5 coherence.   python -I score2.py <dir> <cphd> <out json>"""
import sys, json
import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter
from sarpy.io.phase_history.converter import open_phase_history
W, cphd, out = sys.argv[1:4]
g = np.array(open(f'{W}/pfa_grid.txt').read().split(), float)
secs, Nu, Nv = g[0], int(g[1]), int(g[2]); fpn, ipn, arp = g[3:6], g[6:9], g[9:12]; kv, ku, K, P = g[12:14], g[14:16], int(g[16]), int(g[17])
d = (-arp @ ipn) / (fpn @ ipn); ipx = arp + d * fpn; ipx /= np.linalg.norm(ipx); ipy = np.cross(ipx, ipn)
dv = 1.0 / (Nv * (kv[1] - kv[0]) / (K - 1)); du = 1.0 / (Nu * (ku[1] - ku[0]) / (P - 1))
img = np.memmap(f'{W}/pfa_img.c64', np.complex64, 'r', shape=(Nv, Nu))
r = open_phase_history(cphd)
tx, rcv, srp = (r.read_pvp_variable(n, 0) for n in ('TxPos', 'RcvPos', 'SRPPos'))
apc = 0.5 * (tx + rcv) - srp[0]; up = srp[0] / np.linalg.norm(srp[0]); mid = apc[len(apc) // 2]
los_h = mid - (mid @ up) * up; yhat = -los_h / np.linalg.norm(los_h); xhat = np.cross(yhat, up); R = np.stack([xhat, yhat, up])
pz = np.load(f'{W}/panama.npz'); nx, ny, spx, spy, e1, e2 = int(pz['nx']), int(pz['ny']), float(pz['spx']), float(pz['spy']), pz['e1'], pz['e2']
ref = np.load(f'{W}/ref.npy', mmap_mode='r')
crops = dict(locks=(3400, 6700), port=(9500, 1150), ships=(6400, 2650)); h = 512
sv, su = -1, 1                                               # orientation found by the first scoring pass


def coords(I, J):
    pe = ((((I - nx / 2.0) * spx)[..., None] * e1 + ((J - ny / 2.0) * spy)[..., None] * e2)) @ R
    return Nv // 2 + sv * (pe @ ipx) / dv, Nu // 2 + su * (pe @ ipy) / du


def shift_of(a, b):
    A, B = np.abs(a) - np.abs(a).mean(), np.abs(b) - np.abs(b).mean()
    c = np.abs(np.fft.ifft2(np.fft.fft2(B) * np.conj(np.fft.fft2(A)))); k = np.unravel_index(np.argmax(c), c.shape); n = a.shape[0]
    return [((kk + n // 2) % n - n // 2) for kk in k], float(c.max() / np.sqrt((A ** 2).sum() * (B ** 2).sum()))


def coh5(a, b):
    f = lambda x: uniform_filter(x.real, 5) + 1j * uniform_filter(x.imag, 5)
    return np.abs(f(a * np.conj(b))) / np.sqrt(uniform_filter(np.abs(a) ** 2, 5) * uniform_filter(np.abs(b) ** 2, 5) + 1e-30)


res = dict(pfa_seconds=secs, regions={})
for name, (i0, j0) in crops.items():
    I, J = np.meshgrid(np.arange(i0, i0 + h, dtype=float), np.arange(j0, j0 + h, dtype=float), indexing='ij')
    b = np.asarray(ref[i0:i0 + h, j0:j0 + h]).astype(np.complex128)
    oi = oj = 0.0
    hist = []
    for it in range(6):
        v, u = coords(I + oi, J + oj)
        v0, u0 = int(v.min()) - 20, int(u.min()) - 20
        blk = np.nan_to_num(np.asarray(img[v0:int(v.max()) + 20, u0:int(u.max()) + 20]).astype(np.complex128))
        # spectrum of the block moved to zero frequency before interpolation (its power-weighted circular centroid)
        F = np.abs(np.fft.fft2(blk)) ** 2
        cen = [np.angle(np.sum(F * np.exp(2j * np.pi * np.fft.fftfreq(F.shape[ax])[(slice(None), None) if ax == 0 else (None, slice(None))]))) / (2 * np.pi) for ax in (0, 1)]
        yy, xx = np.meshgrid(np.arange(blk.shape[0]), np.arange(blk.shape[1]), indexing='ij')
        blk = blk * np.exp(-2j * np.pi * (cen[0] * yy + cen[1] * xx))
        a = map_coordinates(blk.real, [v - v0, u - u0], order=3) + 1j * map_coordinates(blk.imag, [v - v0, u - u0], order=3)
        (si, sj), cc = shift_of(a, b)
        hist.append(([oi, oj], [si, sj], round(cc, 3)))
        if si == 0 and sj == 0:
            break
        # move the sampling grid by the measured shift, in whichever direction raises the correlation
        best = None
        for sgn in (1, -1):
            v2, u2 = coords(I + oi + sgn * si, J + oj + sgn * sj)
            a2 = map_coordinates(blk.real, [v2 - v0, u2 - u0], order=1, mode='nearest') + 1j * map_coordinates(blk.imag, [v2 - v0, u2 - u0], order=1, mode='nearest')
            c2 = float(np.corrcoef(np.abs(a2).ravel(), np.abs(b).ravel())[0, 1])
            if best is None or c2 > best[0]:
                best = (c2, sgn)
        oi += best[1] * si; oj += best[1] * sj
    la, lb = np.log10(np.abs(a) + 1e-12), np.log10(np.abs(b) + 1e-12)
    # remaining linear phase between the two images (their spectral centering) fitted and removed before coherence
    from scipy.optimize import minimize
    Jm, Im = np.meshgrid(np.arange(h) - h / 2, np.arange(h) - h / 2)
    xc = a * np.conj(b)
    fq = lambda q: -np.abs(np.sum(xc * np.exp(-1j * (q[0] * Jm + q[1] * Im))))
    q0 = [np.angle(np.sum(xc[:, 1:] * np.conj(xc[:, :-1]))), np.angle(np.sum(xc[1:] * np.conj(xc[:-1])))]
    q = minimize(fq, q0, method='Nelder-Mead', options=dict(xatol=1e-8, fatol=1e-10)).x
    a = a * np.exp(-1j * (q[0] * Jm + q[1] * Im))
    res['regions'][name] = dict(offset_ref_px=[oi, oj], amp_corr=float(np.corrcoef(np.abs(a).ravel(), np.abs(b).ravel())[0, 1]),
                                log_amp_corr=float(np.corrcoef(la.ravel(), lb.ravel())[0, 1]), coherence_5x5_mean=float(coh5(a, b).mean()), history=hist)
    print(name, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in res['regions'][name].items() if k != 'history'}, flush=True)
json.dump(res, open(out, "w"), indent=1, default=float)
