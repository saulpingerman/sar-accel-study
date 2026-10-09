"""Diagnostic of the ISCE3 error on one real collection: one small patch at region 1, ISCE3 at two range
oversamplings, against the float64 reference with three range models (midpoint as in modes_common.reference, exact
bistatic from TxPos and RcvPos, and ISCE3's own transmit-position-plus-velocity model).
    python -I modes_diag.py <cphd> <sicd> <out dir> diag cpu      (MODE=spot for a spotlight)"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modes_common as mc  # noqa: E402
import numba as nb, isce3, scipy.fft  # noqa: E402

cphd, sicd, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
SPOT = os.environ.get('MODE', 'strip') == 'spot'
fx, ant, meta, sm, ap = mc.load(cphd, sicd, stripmap=not SPOT)
P, K = fx['S'].shape
regs = mc.regions(sm, meta)
C = mc.C
S, f0, df, delta = fx['S'], fx['fmin'], fx['df'], fx['ref']
R, o = meta['R'], meta['origin']
tx = np.asarray(meta['tx']) @ R + o; rcv = np.asarray(meta['rcv']) @ R + o
ttx = np.asarray(meta['tx_time'], np.float64)
dtm = float(np.mean(np.diff(ttx)))
txv = np.gradient(tx, ttx, axis=0)
fref = f0 + (K // 2) * df; wvl = C / fref; h = K // 2
epoch = isce3.core.DateTime(2000, 1, 1)
t_rel = ttx - ttx[0]
tau = np.arange(P) * dtm
orbit = isce3.core.Orbit([isce3.core.StateVector(epoch + isce3.core.TimeDelta(float(tau[k])), tx[k], txv[k]) for k in range(P)], epoch)
srp_e = mc.io.local_to_ecf(np.asarray(regs[1]['local']), meta)
right = np.cross(txv[P // 2], tx[P // 2] / np.linalg.norm(tx[P // 2])) @ (srp_e - tx[P // 2]) > 0
side = isce3.core.LookSide.Right if right else isce3.core.LookSide.Left
fD = float(np.mean(2.0 / wvl * ap['sp'] * np.linalg.norm(txv, axis=1)))
dop = isce3.core.LUT2d(np.array([0.0, 2e7]), np.array([-1e3, 1e3 + t_rel[-1]]), np.full((2, 2), fD))
ds = 1e-3 if SPOT else 1.0 / float(sm.Grid.Col.ImpRespBW)
rec = dict(pulses=P, samples=K, ds=ds, fD=fD, pulse_time_deviation_s=float(np.max(np.abs(t_rel - tau))))
# local-frame tx, rcv, and the PVP-free checks of the inputs
txl, rcvl = np.asarray(meta['tx'], np.float64), np.asarray(meta['rcv'], np.float64)
rec['ant_minus_mid_max_m'] = float(np.abs(ant - 0.5 * (txl + rcvl)).max())
tr = np.asarray(meta['rcv_time'], np.float64) - ttx
rec['rcv_model_err_max_m'] = float(np.linalg.norm(rcv - (tx + txv * tr[:, None]), axis=1).max())
rec['rcv_model_err_p99_m'] = float(np.quantile(np.linalg.norm(rcv - (tx + txv * tr[:, None]), axis=1), 0.99))
print(rec, flush=True)

n = int(os.environ.get('DIAG_N', 48))
r = regs[1]
x0 = mc.io.local_to_ecf(np.asarray(r['local']), meta)
ell = isce3.core.Ellipsoid()
dem = isce3.geometry.DEMInterpolator(float(ell.xyz_to_lon_lat(x0)[2]))
tc, rc = isce3.geometry.geo2rdr_bracket(x0, orbit, dop, wvl, side)
pc, vc = orbit.interpolate(tc)
dt_out = float(sm.Grid.Col.SS) / (np.linalg.norm(vc) * np.linalg.norm(x0) / np.linalg.norm(pc))
dr_out = float(sm.Grid.Row.SS)
og = isce3.product.RadarGridParameters(tc - (n // 2) * dt_out, wvl, 1.0 / dt_out, rc - (n // 2) * dr_out, dr_out, side, n, n, epoch)
out_geom = isce3.container.RadarGeometry(og, orbit, dop)
X = np.zeros((n * n, 3)); lo = np.zeros(n * n, np.int64); hi = np.zeros(n * n, np.int64)
for j in range(n):
    for i in range(n):
        x = np.asarray(isce3.geometry.rdr2geo_bracket(og.sensing_start + j / og.prf, og.starting_range + i * dr_out, orbit, side, fD, wvl, dem)).ravel()
        tt, rr = isce3.geometry.geo2rdr_bracket(x, orbit, dop, wvl, side)
        p_, v_ = orbit.interpolate(tt)
        cpi = wvl * rr * (np.linalg.norm(p_) / np.linalg.norm(x)) / (2 * ds) / np.linalg.norm(v_)
        X[j * n + i] = x
        lo[j * n + i] = max(int(np.floor((tt - 0.5 * cpi) / dtm)), 0)
        hi[j * n + i] = min(int(np.ceil((tt + 0.5 * cpi) / dtm)), P)
pts = mc.io.ecf_to_local(X, meta)


@nb.njit(parallel=True, cache=True)
def accum(out, prof, drr, nf2, fref4, txl, rcvl, txvl, trl, ref, p0, pts, lo, hi, model):
    n_, m = prof.shape[0], pts.shape[0]
    for i in nb.prange(m):
        acc = 0j
        for q in range(n_):
            p = p0 + q
            if p < lo[i] or p >= hi[i]:
                continue
            a0 = pts[i, 0] - txl[p, 0]; a1 = pts[i, 1] - txl[p, 1]; a2 = pts[i, 2] - txl[p, 2]
            rt = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
            if model == 0:      # midpoint
                b0 = pts[i, 0] - 0.5 * (txl[p, 0] + rcvl[p, 0]); b1 = pts[i, 1] - 0.5 * (txl[p, 1] + rcvl[p, 1]); b2 = pts[i, 2] - 0.5 * (txl[p, 2] + rcvl[p, 2])
                rr = np.sqrt(b0 * b0 + b1 * b1 + b2 * b2)
            elif model == 1:    # exact bistatic
                b0 = pts[i, 0] - rcvl[p, 0]; b1 = pts[i, 1] - rcvl[p, 1]; b2 = pts[i, 2] - rcvl[p, 2]
                rr = 0.5 * (rt + np.sqrt(b0 * b0 + b1 * b1 + b2 * b2))
            else:               # ISCE3: tau = 2 (r.v - c|r|) / (|v|^2 - c^2), r = x - tx
                vv = txvl[p, 0] ** 2 + txvl[p, 1] ** 2 + txvl[p, 2] ** 2
                rv = a0 * txvl[p, 0] + a1 * txvl[p, 1] + a2 * txvl[p, 2]
                cc = 299792458.0
                rr = 0.5 * cc * 2.0 * (rv - cc * rt) / (vv - cc * cc)
            dR = rr - ref[p]
            tt = dR / drr + nf2
            k = int(np.floor(tt))
            if k < 0 or k + 1 >= prof.shape[1]:
                continue
            a = tt - k
            acc += (prof[q, k] * (1.0 - a) + prof[q, k + 1] * a) * np.exp(1j * fref4 * dR)
        out[i] += acc


def reference(model, up=16, block=256):
    nf = 1 << int(np.ceil(np.log2(up * K))); drr = C / (2 * df * nf)
    txvl = txv @ R.T
    outv = np.zeros(len(pts), np.complex128)
    for p0 in range(0, P, block):
        sl = slice(p0, min(P, p0 + block)); nn = sl.stop - sl.start
        if hi.max() <= p0 or lo.min() >= sl.stop:
            continue
        pad = np.zeros((nn, nf), np.complex128); pad[:, :K - h] = S[sl, h:]; pad[:, nf - h:] = S[sl, :h]
        prof = scipy.fft.fftshift(scipy.fft.ifft(pad, axis=1, workers=-1), axes=1)
        accum(outv, prof, drr, nf // 2, 4 * np.pi * fref / C, txl, rcvl, txvl, tr, np.asarray(delta, np.float64), p0, pts, lo, hi, model)
    return outv


t = time.perf_counter()
refs = {nm: reference(m) for m, nm in enumerate(('mid', 'bistatic', 'isce_model'))}
refs['mc'] = mc.reference(fx, ant, pts, lo=lo, hi=hi)
rec['reference_seconds'] = time.perf_counter() - t
for a in refs:
    for b in refs:
        if a < b:
            rec[f'{a}_vs_{b}_db'] = mc.err_db(refs[a], refs[b])
print({k: v for k, v in rec.items() if k.endswith('_db')}, flush=True)
for os_ in (2, 4):
    nfft = 1 << int(np.ceil(np.log2(os_ * K))); dr = C / (2 * df * nfft)
    dmin = float(delta.min()); off = (delta - dmin) / dr; m_int = np.floor(off).astype(int); frac = off - m_int
    width = nfft + int(m_int.max()) + 2
    r0_buf = dmin - (nfft // 2) * dr
    buf = np.zeros((P, width), np.complex64)
    q = (np.arange(K) - h).astype(np.float64)
    for p0 in range(0, P, 512):
        sl = slice(p0, min(P, p0 + 512)); nn = sl.stop - sl.start
        Sb = S[sl] * np.exp(-2j * np.pi * q[None, :] * frac[sl, None] / nfft).astype(np.complex64)
        pad = np.zeros((nn, nfft), np.complex64); pad[:, :K - h] = Sb[:, h:]; pad[:, nfft - h:] = Sb[:, :h]
        prof = scipy.fft.fftshift(scipy.fft.ifft(pad, axis=1, workers=-1), axes=1)
        prof *= np.exp(-1j * 4 * np.pi * fref * delta[sl, None] / C).astype(np.complex64)
        for i in range(nn):
            k = sl.start + i; buf[k, m_int[k]:m_int[k] + nfft] = prof[i]
    in_grid = isce3.product.RadarGridParameters(0.0, wvl, 1.0 / dtm, r0_buf, dr, side, P, width, epoch)
    in_geom = isce3.container.RadarGeometry(in_grid, orbit, dop)
    kern = isce3.core.TabulatedKernelF32(isce3.core.KnabKernel(8.0, K / nfft), 4096)
    img = np.zeros((n, n), np.complex64)
    isce3.focus.backproject(img, out_geom, buf, in_geom, dem, fref, ds, kern, 'nodelay')
    for nm, rf in refs.items():
        rec[f'isce3_os{os_}_vs_{nm}_db'] = mc.err_db(img.ravel(), rf)
    np.savez_compressed(f'{out}/diag_os{os_}.npz', img=img, **{f'ref_{k}': v.reshape(n, n) for k, v in refs.items()})
    del buf
    print({k: v for k, v in rec.items() if k.startswith(f'isce3_os{os_}')}, flush=True)
json.dump(rec, open(f'{out}/diag.json', 'w'), indent=1, default=float)
