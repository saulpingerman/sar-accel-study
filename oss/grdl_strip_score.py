"""GRDL stripmap images (range-Doppler, FFBP) against the float64 reference on the three regions of the other-modes
comparison. GRDL processes the full Doppler band, the reference the vendor's processed band, so the images differ in
azimuth resolution and are compared as the polar-format images of Panama were: each region of the reference (FastSAR's
pixels of that region) is mapped into the GRDL image through GRDL's own grid geometry (range-Doppler: zero-Doppler
pulse index and slant range about the reference range at the center column; FFBP: cross-range and range axes about its
reference point), the block is basebanded and interpolated, registered by amplitude cross-correlation, and scored by
amplitude and log-amplitude correlation, by the same after 4 by 4 multilooking of both images, and by mean 5 by 5 coherence.
    python -I grdl_strip_score.py <cphd> <sicd> <modes dir of the collection> <rda|ffbp> <image: local .npy or gs:// .npy> <out json>"""
import os, sys, json, subprocess
import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modes_common as mc  # noqa: E402

cphd, sicd, md, alg, src, out = sys.argv[1:7]
rec = json.load(open(f'{md}/grdl_{alg}.json')); shape = tuple(rec['image_shape']); at = rec['algorithm_attrs']


class Image:
    def __init__(self, path):
        self.path = path
        if path.startswith('gs://'):
            hdr = subprocess.run(['gcloud', 'storage', 'cat', '-r', '0-4095', path], capture_output=True, check=True).stdout
            self.off = 10 + int.from_bytes(hdr[8:10], 'little')
        else:
            self.mm = np.load(path, mmap_mode='r')

    def block(self, r0, r1, c0, c1):
        r0, r1, c0, c1 = max(r0, 0), min(r1, shape[0]), max(c0, 0), min(c1, shape[1])
        if self.path.startswith('gs://'):
            a, b = self.off + r0 * shape[1] * 8, self.off + r1 * shape[1] * 8 - 1
            raw = subprocess.run(['gcloud', 'storage', 'cat', '-r', f'{a}-{b}', self.path], capture_output=True, check=True).stdout
            blk = np.frombuffer(raw, np.complex64).reshape(r1 - r0, shape[1])[:, c0:c1]
        else:
            blk = self.mm[r0:r1, c0:c1]
        return np.nan_to_num(np.asarray(blk).astype(np.complex128)), (r0, c0)


img = Image(src)
fx, ant, meta, sm, ap = mc.load(cphd, sicd, stripmap=True)
P = len(ant)
R, o = meta['R'], meta['origin']
apc_e = ant @ R + o
vel = np.gradient(apc_e, axis=0)
dr = at['_delta_r']; r0c = at['_r0_center']
lo_trim = (at['_npulses_orig'] - at['_npulses']) // 2 if alg == 'rda' else 0


CORNERS = os.environ.get('CORNERS')          # 'k0,k1,k2,k3': which reported corner is pixel (0,0), (0,N1), (N0,N1), (N0,0)


def coords(pts_local):
    x = mc.io.local_to_ecf(pts_local, meta)
    if CORNERS:
        # GRDL's own geolocation: the four reported corner points (lat, lon), bilinear over the image
        cp = np.asarray(rec['grid']['corner_points'], np.float64)[[int(c) for c in CORNERS.split(',')]]
        ll = np.stack([np.degrees(np.arcsin(x[:, 2] / np.linalg.norm(x, axis=1))), np.degrees(np.arctan2(x[:, 1], x[:, 0]))], 1)
        # invert the bilinear map by Newton iterations from the image center
        uv = np.full((len(x), 2), 0.5)
        for _ in range(20):
            u, w = uv[:, 0:1], uv[:, 1:2]
            f = (1 - u) * (1 - w) * cp[0] + (1 - u) * w * cp[1] + u * w * cp[2] + u * (1 - w) * cp[3] - ll
            du = -(1 - w) * cp[0] - w * cp[1] + w * cp[2] + (1 - w) * cp[3]
            dw = -(1 - u) * cp[0] + (1 - u) * cp[1] + u * cp[2] - u * cp[3]
            det = du[:, 0] * dw[:, 1] - du[:, 1] * dw[:, 0]
            uv[:, 0] -= (f[:, 0] * dw[:, 1] - f[:, 1] * dw[:, 0]) / det
            uv[:, 1] -= (du[:, 0] * f[:, 1] - du[:, 1] * f[:, 0]) / det
        return uv[:, 0] * shape[0], uv[:, 1] * shape[1]
    if alg == 'rda':
        # zero-Doppler pulse: (x - a_p) . v_p = 0, monotonic in p; fractional index by bisection on a coarse search
        g = lambda p: np.einsum('ij,ij->i', x - apc_e[p], vel[p])
        lo, hi = np.zeros(len(x), np.int64), np.full(len(x), P - 1, np.int64)
        s_lo = np.sign(g(lo))
        while (hi - lo > 1).any():
            mid = (lo + hi) // 2
            same = np.sign(g(mid)) == s_lo
            lo = np.where(same, mid, lo); hi = np.where(same, hi, mid)
        gl, gh = g(lo), g(hi)
        pz = lo + gl / (gl - gh)
        a = apc_e[lo] + (pz - lo)[:, None] * (apc_e[hi] - apc_e[lo])
        r = np.linalg.norm(x - a, axis=1)
        return pz - lo_trim, shape[1] / 2 + (r - r0c) / dr
    srp = np.asarray(at['_ref_srp']); ur, ux = np.asarray(at['_u_range']), np.asarray(at['_u_cross_range'])
    return shape[0] / 2 + (x - srp) @ ux / at['_xr_spacing'], shape[1] / 2 - (x - srp) @ ur / dr


def shift_of(a, b):
    A, B = np.abs(a) - np.abs(a).mean(), np.abs(b) - np.abs(b).mean()
    c = np.abs(np.fft.ifft2(np.fft.fft2(B) * np.conj(np.fft.fft2(A)))); k = np.unravel_index(np.argmax(c), c.shape); n = a.shape[0]
    return [((kk + n // 2) % n - n // 2) for kk in k], float(c.max() / np.sqrt((A ** 2).sum() * (B ** 2).sum()))


def coh5(a, b):
    f = lambda z: uniform_filter(z.real, 5) + 1j * uniform_filter(z.imag, 5)
    return np.abs(f(a * np.conj(b))) / np.sqrt(uniform_filter(np.abs(a) ** 2, 5) * uniform_filter(np.abs(b) ** 2, 5) + 1e-30)


def ml(a, k=4):
    p = np.abs(a) ** 2; n = (p.shape[0] // k) * k
    return p[:n, :n].reshape(n // k, k, n // k, k).mean((1, 3))


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


res = dict(algorithm=alg, seconds=rec['seconds'], regions={})
for k in range(3):
    z = np.load(f'{md}/fastsar_cpu_r{k}.npz')
    b, pts = z['ref'].astype(np.complex128), z['pts']; n = b.shape[0]
    e1 = (pts[n] - pts[0]); e2 = (pts[1] - pts[0])            # grid steps along the region's two axes
    oi = oj = 0.0; hist = []
    for it in range(8):
        P_ = pts + oi * e1 + oj * e2
        v, u = coords(P_)
        blk, (v0, u0) = img.block(int(v.min()) - 40, int(v.max()) + 40, int(u.min()) - 40, int(u.max()) + 40)
        F = np.abs(np.fft.fft2(blk)) ** 2
        cen = [np.angle(np.sum(F * np.exp(2j * np.pi * np.fft.fftfreq(F.shape[ax])[(slice(None), None) if ax == 0 else (None, slice(None))]))) / (2 * np.pi) for ax in (0, 1)]
        yy, xx = np.meshgrid(np.arange(blk.shape[0]), np.arange(blk.shape[1]), indexing='ij')
        blk = blk * np.exp(-2j * np.pi * (cen[0] * yy + cen[1] * xx))
        a = (map_coordinates(blk.real, [v - v0, u - u0], order=3) + 1j * map_coordinates(blk.imag, [v - v0, u - u0], order=3)).reshape(n, n)
        (si, sj), cc = shift_of(a, b)
        hist.append(([oi, oj], [si, sj], round(cc, 3)))
        if si == 0 and sj == 0:
            break
        best = None
        for sg in (1, -1):
            v2, u2 = coords(pts + (oi + sg * si) * e1 + (oj + sg * sj) * e2)
            a2 = (map_coordinates(blk.real, [v2 - v0, u2 - u0], order=1, mode='nearest') + 1j * map_coordinates(blk.imag, [v2 - v0, u2 - u0], order=1, mode='nearest')).reshape(n, n)
            c2 = corr(np.abs(a2), np.abs(b))
            if best is None or c2 > best[0]:
                best = (c2, sg)
        oi += best[1] * si; oj += best[1] * sj
    xc = a * np.conj(b)
    Jm, Im = np.meshgrid(np.arange(n) - n / 2, np.arange(n) - n / 2)
    q = [np.angle(np.sum(xc[:, 1:] * np.conj(xc[:, :-1]))), np.angle(np.sum(xc[1:] * np.conj(xc[:-1])))]
    a = a * np.exp(-1j * (q[0] * Jm + q[1] * Im))
    la, lb = np.log10(np.abs(a) + 1e-12), np.log10(np.abs(b) + 1e-12)
    res['regions'][f'r{k}'] = dict(offset_px=[oi, oj], amp_corr=corr(np.abs(a), np.abs(b)), log_amp_corr=corr(la, lb),
                                   amp_corr_ml4=corr(ml(a), ml(b)), log_amp_corr_ml4=corr(np.log10(ml(a) + 1e-24), np.log10(ml(b) + 1e-24)),
                                   coherence_5x5_mean=float(coh5(a, b).mean()), history=hist)
    print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in res['regions'][f'r{k}'].items() if kk != 'history'}, flush=True)
json.dump(res, open(out, 'w'), indent=1, default=float)
