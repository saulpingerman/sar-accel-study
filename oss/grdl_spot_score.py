"""GRDL spotlight images (polar format, FFBP) of a modes-comparison collection against the float64 reference on the three
regions of modes_strip.py, scored as on Panama (grdl_score.py): each region's reference points are mapped into the GRDL
image through the geometry of its grid (image center at the scene reference point), the block is basebanded and
interpolated, registered by amplitude cross-correlation, and compared after removal of a linear phase and the best gain.
The reference points are in modes_common's local frame, whose origin load() moves to the center of the vendor footprint;
that origin is the mean of the SICD's corner points on the ground.
    python -I grdl_spot_score.py <cphd url or path> <sicd> <dir with fastsar_cpu_r{0,1,2}.npz> <grdl json> <image .npy, local or gs://> <out json>"""
import sys, json, subprocess, itertools
import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter
from scipy.optimize import minimize

cphd, sicd, W, gj, src, out = sys.argv[1:7]
rec = json.load(open(gj)); alg = rec['algorithm']; shape = tuple(rec['image_shape'])


class Image:
    """Row blocks of a complex64 .npy, from a local file or by byte range from a bucket."""
    def __init__(self, path):
        self.path = path
        if path.startswith('gs://'):
            hdr = subprocess.run(['gcloud', 'storage', 'cat', '-r', '0-4095', path], capture_output=True, check=True).stdout
            hl = int.from_bytes(hdr[8:10], 'little'); self.off = 10 + hl
        else:
            self.mm = np.load(path, mmap_mode='r')

    def rows(self, r0, r1, c0, c1):
        r0, r1 = max(r0, 0), min(r1, shape[0]); c0, c1 = max(c0, 0), min(c1, shape[1])
        if r1 <= r0 or c1 <= c0:
            return None, (r0, c0)
        if self.path.startswith('gs://'):
            a, b = self.off + r0 * shape[1] * 8, self.off + r1 * shape[1] * 8 - 1
            raw = subprocess.run(['gcloud', 'storage', 'cat', '-r', f'{a}-{b}', self.path], capture_output=True, check=True).stdout
            blk = np.frombuffer(raw, np.complex64).reshape(r1 - r0, shape[1])[:, c0:c1]
        else:
            blk = self.mm[r0:r1, c0:c1]
        return np.nan_to_num(np.asarray(blk).astype(np.complex128)), (r0, c0)


img = Image(src)


def pvp(u):
    """TxPos, RcvPos, SRPPos of a CPHD, from a URL by byte ranges (PVP block only) or a local file."""
    import re, urllib.request, xml.etree.ElementTree as ET
    def rd(a, b):
        if u.startswith('http'):
            return urllib.request.urlopen(urllib.request.Request(u, headers={'Range': f'bytes={a}-{b - 1}'})).read()
        with open(u, 'rb') as f:
            f.seek(a)
            return f.read(b - a)
    kv = dict(re.findall(r'^([A-Z_]+) := (.*)$', rd(0, 4096).decode('latin1'), re.M))
    xo, xs, po = int(kv['XML_BLOCK_BYTE_OFFSET']), int(kv['XML_BLOCK_SIZE']), int(kv['PVP_BLOCK_BYTE_OFFSET'])
    x = ET.fromstring(re.sub(r'\sxmlns="[^"]+"', '', rd(xo, xo + xs).decode(), count=1))
    nv, nb = int(x.find('.//Data/Channel/NumVectors').text), int(x.find('.//Data/NumBytesPVP').text)
    raw = np.frombuffer(rd(po, po + nv * nb), '>f8').reshape(nv, nb // 8)
    return [raw[:, int(x.find(f'.//PVP/{n}/Offset').text):][:, :3].copy() for n in ('TxPos', 'RcvPos', 'SRPPos')]


tx, rcv, srp = pvp(cphd)
apc = 0.5 * (tx + rcv); mid = len(apc) // 2
s0 = srp[mid]; up = s0 / np.linalg.norm(s0); m = apc[mid] - s0
los_h = m - (m @ up) * up; yhat = -los_h / np.linalg.norm(los_h); xhat = np.cross(yhat, up); R = np.stack([xhat, yhat, up])
from sarpy.io.complex.converter import open_complex
if sicd.endswith('.xml'):
    from sarpy.io.complex.sicd_elements.SICD import SICDType
    sm = SICDType.from_xml_string(open(sicd).read())
else:
    sm = open_complex(sicd).sicd_meta
nr, nc = int(sm.ImageData.NumRows), int(sm.ImageData.NumCols)
cor = np.asarray(sm.project_image_to_ground(np.array([[0, 0], [0, nc - 1], [nr - 1, 0], [nr - 1, nc - 1]], float), projection_type='HAE',
                                            hae0=float(sm.GeoData.SCP.LLH.HAE)))
shift = cor.mean(0) - s0                                             # local origin of the reference points, relative to the SRP
regs = {f'r{k}': np.load(f'{W}/fastsar_cpu_r{k}.npz') for k in range(3)}
h = regs['r0']['ref'].shape[0]

# GRDL grid: unit vectors and spacings of its two axes, image center at the scene reference point
los = -m / np.linalg.norm(m)
vel = apc[mid + 1] - apc[mid - 1]; vel /= np.linalg.norm(vel)
u_rg = los; u_az = vel - (vel @ u_rg) * u_rg; u_az /= np.linalg.norm(u_az)
if alg == 'pfa':
    g = rec['grid']; sp_rg, sp_az = g['rg_ss'], g['az_ss']
else:
    a = rec['algorithm_attrs']; u_rg = -np.array(a['_u_range']); u_az = np.array(a['_u_cross_range'])
    sp_rg, sp_az = a['_delta_r'], a['_xr_spacing']


def coords(P_, hyp):
    """Reference points [..., 3] (local frame) to GRDL image (row, column) under orientation hypothesis hyp."""
    swap, sa, sr = hyp
    pe = P_ @ R + shift
    az, rg = sa * (pe @ u_az) / sp_az, sr * (pe @ u_rg) / sp_rg
    return (shape[0] / 2 + rg, shape[1] / 2 + az) if swap else (shape[0] / 2 + az, shape[1] / 2 + rg)


def shift_of(a, b):
    A, B = np.abs(a) - np.abs(a).mean(), np.abs(b) - np.abs(b).mean()
    c = np.abs(np.fft.ifft2(np.fft.fft2(B) * np.conj(np.fft.fft2(A)))); k = np.unravel_index(np.argmax(c), c.shape); n = a.shape[0]
    return [((kk + n // 2) % n - n // 2) for kk in k], float(c.max() / np.sqrt((A ** 2).sum() * (B ** 2).sum()))


def coh5(a, b):
    f = lambda x: uniform_filter(x.real, 5) + 1j * uniform_filter(x.imag, 5)
    return np.abs(f(a * np.conj(b))) / np.sqrt(uniform_filter(np.abs(a) ** 2, 5) * uniform_filter(np.abs(b) ** 2, 5) + 1e-30)


def sample(P_, hyp, pad=40):
    v, u = coords(P_, hyp)
    blk, (v0, u0) = img.rows(int(np.floor(v.min())) - pad, int(np.ceil(v.max())) + pad, int(np.floor(u.min())) - pad, int(np.ceil(u.max())) + pad)
    if blk is None or blk.shape[0] < 8 or blk.shape[1] < 8:
        return None
    F = np.abs(np.fft.fft2(blk)) ** 2
    cen = [np.angle(np.sum(F * np.exp(2j * np.pi * np.fft.fftfreq(F.shape[ax])[(slice(None), None) if ax == 0 else (None, slice(None))]))) / (2 * np.pi) for ax in (0, 1)]
    yy, xx = np.meshgrid(np.arange(blk.shape[0]), np.arange(blk.shape[1]), indexing='ij')
    blk = blk * np.exp(-2j * np.pi * (cen[0] * yy + cen[1] * xx))
    return map_coordinates(blk.real, [v - v0, u - u0], order=3, cval=0.0) + 1j * map_coordinates(blk.imag, [v - v0, u - u0], order=3, cval=0.0)


def pts_of(z, oi=0.0, oj=0.0):
    """The region's points [h, h, 3], moved by (oi, oj) of its own pixels."""
    P_ = z['pts'].reshape(h, h, 3)
    di, dj = P_[1, 0] - P_[0, 0], P_[0, 1] - P_[0, 0]
    return P_ + oi * di + oj * dj


# orientation from region 1 (the scene center)
z = regs['r1']; b = z['ref'].astype(np.complex128)
tries = []
for hyp in itertools.product((False, True), (1, -1), (1, -1)):
    a = sample(pts_of(z), hyp)
    if a is None:
        tries.append((hyp, None)); continue
    (si, sj), cc = shift_of(a, b)
    tries.append((hyp, cc))
    print('orientation', hyp, round(cc, 3), (si, sj), flush=True)
hyp = max((t for t in tries if t[1] is not None), key=lambda t: t[1])[0]
res = dict(algorithm=alg, seconds=rec['seconds'], orientation=dict(swap=hyp[0], az_sign=hyp[1], rg_sign=hyp[2]),
           orientation_scores=[[list(t[0]), t[1]] for t in tries], regions={})
for name, z in regs.items():
    b = z['ref'].astype(np.complex128)
    oi = oj = 0.0; hist = []
    for it in range(8):
        a = sample(pts_of(z, oi, oj), hyp)
        (si, sj), cc = shift_of(a, b)
        hist.append(([oi, oj], [si, sj], round(cc, 3)))
        if si == 0 and sj == 0:
            break
        best = None
        for sgn in (1, -1):
            a2 = sample(pts_of(z, oi + sgn * si, oj + sgn * sj), hyp)
            c2 = float(np.corrcoef(np.abs(a2).ravel(), np.abs(b).ravel())[0, 1])
            if best is None or c2 > best[0]:
                best = (c2, sgn)
        oi += best[1] * si; oj += best[1] * sj
    Jm, Im = np.meshgrid(np.arange(h) - h / 2, np.arange(h) - h / 2)
    xc = a * np.conj(b)
    fq = lambda q: -np.abs(np.sum(xc * np.exp(-1j * (q[0] * Jm + q[1] * Im))))
    q0 = [np.angle(np.sum(xc[:, 1:] * np.conj(xc[:, :-1]))), np.angle(np.sum(xc[1:] * np.conj(xc[:-1])))]
    q = minimize(fq, q0, method='Nelder-Mead', options=dict(xatol=1e-8, fatol=1e-10)).x
    a = a * np.exp(-1j * (q[0] * Jm + q[1] * Im))
    g_ = np.vdot(a, b) / np.vdot(a, a)
    err = 10 * np.log10(np.sum(np.abs(b - g_ * a) ** 2) / np.sum(np.abs(b) ** 2))
    la, lb = np.log10(np.abs(a) + 1e-12), np.log10(np.abs(b) + 1e-12)
    res['regions'][name] = dict(offset_ref_px=[oi, oj], amp_corr=float(np.corrcoef(np.abs(a).ravel(), np.abs(b).ravel())[0, 1]),
                                log_amp_corr=float(np.corrcoef(la.ravel(), lb.ravel())[0, 1]), coherence_5x5_mean=float(coh5(a, b).mean()),
                                error_db_after_gain=float(err), history=hist)
    print(name, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in res['regions'][name].items() if k != 'history'}, flush=True)
json.dump(res, open(out, 'w'), indent=1, default=float)
