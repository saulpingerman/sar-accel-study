"""ISCE3's backprojection (isce3.focus.backproject) on the Umbra Panama CPHD, against our float64 exact
backprojection evaluated at the same 3D targets.

    python isce3_compare.py <cphd> <out.json> <L> <W> [x_local y_local] [ref_oversample] [isce_os]

ISCE3 forms an L x W patch of its zero-Doppler radar grid (0.3 m spacing in range and about 0.3 m on the ground in
azimuth) centered on the point (x_local, y_local) of our image plane (default: the scene reference point), on a
constant-height DEM at the reference point's height. Its targets are recovered with ISCE3's own rdr2geo, and our
reference is evaluated at exactly those ECEF points, so the two images are compared pixel for pixel.

Inputs ISCE3 needs and CPHD does not hold directly, and how they are made:
- range-compressed echoes on one absolute range axis: each pulse's profile (referenced to the scene point in CPHD)
  is shifted to its absolute delay, the integer part by placement in a wider buffer and the fractional part by a
  phase ramp before the transform, and given the carrier phase of that delay;
- an orbit sampled uniformly in time: the pulses are not uniform in time (deviation up to 4.7 ms on Panama), so the
  orbit's time axis is the pulse index scaled by the mean interval, with each pulse's own transmit position and
  velocity at its knot. ISCE3 evaluates pulses exactly at the knots.
"""
import os, sys, json, time
import numpy as np, scipy.fft
from scipy.signal.windows import taylor
import isce3
from sarpy.io.phase_history.converter import open_phase_history
C = 299792458.0
cphd, out_json, L, W = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
xl, yl = (float(sys.argv[5]), float(sys.argv[6])) if len(sys.argv) > 6 else (0.0, 0.0)
ref_os = int(sys.argv[7]) if len(sys.argv) > 7 else 16
isce_os = int(sys.argv[8]) if len(sys.argv) > 8 else 2
import os
BISTATIC = os.environ.get('REF_BISTATIC', '0') == '1'
t_all = time.perf_counter()
r = open_phase_history(cphd); m = r.cphd_meta
P, K = m.Data.Channels[0].NumVectors, m.Data.Channels[0].NumSamples
tx, rcv, srp = (r.read_pvp_variable(n, 0) for n in ('TxPos', 'RcvPos', 'SRPPos'))
txv = r.read_pvp_variable('TxVel', 0); ttx = r.read_pvp_variable('TxTime', 0)
f0, df = float(r.read_pvp_variable('SC0', 0).mean()), float(r.read_pvp_variable('SCSS', 0).mean())
S = r.read_chip((0, P), (0, K), index=0).astype(np.complex64)
if m.Global.SGN > 0:
    S = np.conj(S)
srp0 = srp[0]
S *= (taylor(P, nbar=4, sll=35.0, norm=False)[:, None] * taylor(K, nbar=4, sll=35.0, norm=False)[None, :]).astype(np.float32)
fref = f0 + (K // 2) * df; wvl = C / fref; h = K // 2
# our local frame, exactly as v2_prep builds it (antenna = transmit/receive midpoint, origin at the reference point)
apc = 0.5 * (tx + rcv) - srp0
up = srp0 / np.linalg.norm(srp0); mid = apc[P // 2]
los_h = mid - (mid @ up) * up; yhat = -los_h / np.linalg.norm(los_h); xhat = np.cross(yhat, up)
Rm = np.stack([xhat, yhat, up]); ant = apc @ Rm.T
info = dict(pulses=P, samples=K, fref=fref, reference_bistatic=BISTATIC)
# ---- ISCE3 inputs
dtm = float(np.mean(np.diff(ttx)))
epoch = isce3.core.DateTime(2023, 7, 18, 2, 30, 32)
svs = [isce3.core.StateVector(epoch + isce3.core.TimeDelta(k * dtm), tx[k], txv[k]) for k in range(P)]
orbit = isce3.core.Orbit(svs, epoch)
right = np.cross(txv[P // 2], tx[P // 2] / np.linalg.norm(tx[P // 2])) @ (srp0 - tx[P // 2]) > 0
side = isce3.core.LookSide.Right if right else isce3.core.LookSide.Left
info['look_side'] = 'right' if right else 'left'
dop0 = isce3.core.LUT2d()
# Doppler centroid of the reference point at mid-aperture: the collection is squinted, so zero Doppler can fall
# outside the aperture; both geometries use this constant centroid
pm, vm = tx[P // 2], txv[P // 2]
fD = float(2.0 / wvl * vm @ (srp0 - pm) / np.linalg.norm(srp0 - pm))
info['doppler_centroid_hz'] = fD
info['squint_deg'] = float(np.degrees(np.arcsin(vm @ (srp0 - pm) / np.linalg.norm(srp0 - pm) / np.linalg.norm(vm))))
dop = isce3.core.LUT2d(np.array([0.0, 2e6]), np.array([-1e3, 1e3 + P * dtm]), np.full((2, 2), fD))
delta = 0.5 * (np.linalg.norm(tx - srp0, axis=1) + np.linalg.norm(rcv - srp0, axis=1))   # one-way equivalent delay of the reference point
nfft = 1 << int(np.ceil(np.log2(isce_os * K))); dr = C / (2 * df * nfft)
dmin = float(delta.min()); off = (delta - dmin) / dr; m_int = np.floor(off).astype(int); frac = off - m_int
width = nfft + int(m_int.max()) + 2
r0_buf = dmin - (nfft // 2) * dr
t = time.perf_counter()
buf = np.zeros((P, width), np.complex64)
q = (np.arange(K) - h).astype(np.float64)
for p0 in range(0, P, 512):
    sl = slice(p0, min(P, p0 + 512)); n = sl.stop - sl.start
    Sb = S[sl] * np.exp(-2j * np.pi * q[None, :] * frac[sl, None] / nfft).astype(np.complex64)
    pad = np.zeros((n, nfft), np.complex64); pad[:, :K - h] = Sb[:, h:]; pad[:, nfft - h:] = Sb[:, :h]
    prof = scipy.fft.fftshift(scipy.fft.ifft(pad, axis=1, workers=-1), axes=1)
    prof *= np.exp(-1j * 4 * np.pi * fref * delta[sl, None] / C).astype(np.complex64)
    for i in range(n):
        k = sl.start + i; buf[k, m_int[k]:m_int[k] + nfft] = prof[i]
info['buffer'] = dict(width=width, range_spacing=dr, seconds=time.perf_counter() - t)
in_grid = isce3.product.RadarGridParameters(0.0, wvl, 1.0 / dtm, r0_buf, dr, side, P, width, epoch)
in_geom = isce3.container.RadarGeometry(in_grid, orbit, dop)
# output grid: L x W around the chosen image-plane point, on a constant-height DEM through it
target0 = srp0 + Rm.T @ np.array([xl, yl, 0.0])
ell = isce3.core.Ellipsoid(); llh0 = ell.xyz_to_lon_lat(target0)
dem = isce3.geometry.DEMInterpolator(float(llh0[2]))
tc, rc = isce3.geometry.geo2rdr_bracket(target0, orbit, dop, wvl, side)
pc, vc = orbit.interpolate(tc)
dt_out = 0.3 / (np.linalg.norm(vc) * np.linalg.norm(target0) / np.linalg.norm(pc))
out_grid = isce3.product.RadarGridParameters(tc - (L // 2) * dt_out, wvl, 1.0 / dt_out, rc - (W // 2) * 0.3, 0.3, side, L, W, epoch)
out_geom = isce3.container.RadarGeometry(out_grid, orbit, dop)
kern = isce3.core.TabulatedKernelF32(isce3.core.KnabKernel(8.0, K / nfft), 4096)
out = np.zeros((L, W), np.complex64)
t = time.perf_counter()
bp_fn = isce3.focus.backproject
if os.environ.get('ISCE_CUDA') == '1':
    import isce3.cuda.focus
    bp_fn = isce3.cuda.focus.backproject
    bp_fn(np.zeros((8, 8), np.complex64), isce3.container.RadarGeometry(isce3.product.RadarGridParameters(out_grid.sensing_start, wvl, out_grid.prf, out_grid.starting_range, 0.3, side, 8, 8, epoch), orbit, dop), buf, in_geom, dem, fref, 1e-3, kern, 'nodelay')   # warm-up (context, transfer)
    t = time.perf_counter()
ok = bp_fn(out, out_geom, buf, in_geom, dem, fref, 1e-3, kern, 'nodelay')
info['device'] = 'cuda' if os.environ.get('ISCE_CUDA') == '1' else 'cpu'
info['isce3'] = dict(seconds=time.perf_counter() - t, converged=bool(ok))
del buf
# ISCE3's targets
X = np.zeros((L, W, 3))
for j in range(L):
    for i in range(W):
        X[j, i] = np.asarray(isce3.geometry.rdr2geo_bracket(out_grid.sensing_start + j / out_grid.prf, out_grid.starting_range + i * 0.3, orbit, side, fD, wvl, dem)).ravel()
pix = (X.reshape(-1, 3) - srp0) @ Rm.T
# ---- our float64 exact backprojection at the same points (as the study's reference: midpoint antenna, linear
#      interpolation of ref_os-times oversampled profiles)
t = time.perf_counter()
nf = 1 << int(np.ceil(np.log2(ref_os * K))); drr = C / (2 * df * nf)
ref = np.zeros(len(pix), np.complex128); r0a = np.linalg.norm(ant, axis=1)
for p0 in range(0, P, 256):
    sl = slice(p0, min(P, p0 + 256)); n = sl.stop - sl.start
    pad = np.zeros((n, nf), np.complex128); pad[:, :K - h] = S[sl, h:]; pad[:, nf - h:] = S[sl, :h]
    prof = scipy.fft.fftshift(scipy.fft.ifft(pad, axis=1, workers=-1), axes=1)
    for i in range(n):
        k = sl.start + i
        if BISTATIC:      # transmit and receive positions separately, as the CPHD defines them
            Xe = srp0 + pix @ Rm
            dR = 0.5 * (np.linalg.norm(Xe - tx[k], axis=1) + np.linalg.norm(Xe - rcv[k], axis=1)) - delta[k]
        else:
            dR = np.linalg.norm(pix - ant[k], axis=1) - r0a[k]
        tt = dR / drr + nf // 2
        i0 = np.floor(tt).astype(int); w = tt - i0
        ref += (prof[i, i0] * (1 - w) + prof[i, i0 + 1] * w) * np.exp(1j * 4 * np.pi * fref / C * dR)
info['reference_seconds'] = time.perf_counter() - t
img = out.reshape(-1).astype(np.complex128)
def err(a, b):
    g = np.vdot(a, b) / np.vdot(a, a)
    return float(10 * np.log10(np.sum(np.abs(g * a - b) ** 2) / np.sum(np.abs(b) ** 2))), complex(g)
e, g = err(img, ref); ec, _ = err(np.conj(img), ref)
bright = np.abs(ref) > np.percentile(np.abs(ref), 90)
d = img[bright] * np.conj(ref[bright])
info.update(error_db=e, conj_error_db=ec, gain=[abs(g), float(np.angle(g))],
            amp_corr=float(np.corrcoef(np.abs(img), np.abs(ref))[0, 1]),
            phase_std_bright_deg=float(np.degrees(np.std(np.angle(d * np.exp(-1j * np.angle(d.sum())))))),
            patch=dict(L=L, W=W, center_local=[xl, yl], height=float(llh0[2])), seconds_total=time.perf_counter() - t_all)
print(json.dumps(info, indent=1)); json.dump(info, open(out_json, 'w'), indent=1)
np.savez_compressed(out_json.replace('.json', '.npz'), isce3=out, ref=ref.reshape(L, W).astype(np.complex64), xyz=X)
