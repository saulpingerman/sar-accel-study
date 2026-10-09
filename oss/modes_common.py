"""Shared pieces of the other-modes comparison (stripmap and further spotlight collections): every implementation gets
the same input, read by FastSAR's CPHD reader (which corrects the phase sign of Capella's files) and multiplied by a
Taylor window (nbar 4, 35 dB) along frequency, and along pulses as well for spotlight collections; each is scored
against a float64 exact backprojection evaluated at its own output pixels with its own processed aperture.
The reference interpolates range profiles oversampled 16 times linearly, as the study's reference does."""
import os, sys, json
import numpy as np
from scipy.signal.windows import taylor
sys.path.insert(0, os.environ.get('FASTSAR', os.path.join(os.path.dirname(__file__), '..', '..', 'FastSAR')))
from fastsar import io  # noqa: E402

C = 299792458.0


def load(cphd, sicd, stripmap):
    """-> fx (FastSAR input), meta, sicd metadata and the processed-aperture description."""
    from sarpy.io.complex.converter import open_complex
    from sarpy.io.phase_history.converter import open_phase_history
    col, meta = io.read_cphd(cphd, meta=True)
    S = col['S']; P, K = S.shape
    wk = taylor(K, nbar=4, sll=35.0, norm=False).astype(np.float32)
    wp = taylor(P, nbar=4, sll=35.0, norm=False).astype(np.float32) if not stripmap else np.ones(P, np.float32)
    for p0 in range(0, P, 1024):            # in place, row blocks (the largest collections are tens of GB)
        S[p0:p0 + 1024] *= wp[p0:p0 + 1024, None] * wk[None, :]
    if sicd.endswith('.xml'):                 # the SICD's metadata alone (ICEYE publishes it beside the image)
        from sarpy.io.complex.sicd_elements.SICD import SICDType
        sm = SICDType.from_xml_string(open(sicd).read())
    else:
        sm = open_complex(sicd).sicd_meta
    if not stripmap:
        # spotlight: move the phase reference (and the local origin) to the center of the vendor image's footprint, so
        # that a grid centered on the origin covers it; ICEYE's receive window starts at the scene reference point
        rows, cols = int(sm.ImageData.NumRows), int(sm.ImageData.NumCols)
        R_, C_ = np.meshgrid([0, rows - 1], [0, cols - 1], indexing='ij')
        c = io.sicd_points(sm, R_.ravel(), C_.ravel(), meta).mean(0)
        tx, rcv = np.asarray(meta['tx']) - c, np.asarray(meta['rcv']) - c
        ref_new = 0.5 * (np.linalg.norm(tx, axis=1) + np.linalg.norm(rcv, axis=1))
        dref = np.asarray(meta['ref'], np.float64) - ref_new
        f = col['fmin'] + col['df'] * np.arange(K)
        for p0 in range(0, P, 1024):
            sl = slice(p0, min(P, p0 + 1024))
            S[sl] *= np.exp(-4j * np.pi * f[None, :] / C * dref[sl, None]).astype(np.complex64)
        col['ant'] = np.asarray(col['ant'], np.float64) - c
        meta['origin'] = meta['origin'] + c @ meta['R']
        meta['tx'], meta['rcv'], meta['ref'] = tx, rcv, ref_new
        meta['notes'].append(f'phase reference moved {np.linalg.norm(c):.1f} m to the center of the vendor footprint')
    lo, hi = meta.get('pulses', (0, P))
    srp = io.ecf_to_local(open_phase_history(cphd).read_pvp_variable('SRPPos', 0)[lo:hi], meta)   # (after any move)
    ant = np.asarray(col['ant'], np.float64)
    lam = C / (col['fmin'] + K / 2 * col['df'])
    d = np.gradient(ant, axis=0); d /= np.linalg.norm(d, axis=1, keepdims=True)
    sp = ((srp - ant) * d).sum(1) / np.linalg.norm(srp - ant, axis=1)
    ap = dict(d=d, sp=sp, dsin=float(sm.Grid.Col.ImpRespBW) * lam / 2, lam=lam)
    fx = dict(S=S, fmin=float(col['fmin']), df=float(col['df']), ref=np.asarray(meta['ref'], np.float64), band=(col['fmin'], col['fmin'] + K * col['df']))
    return fx, ant, meta, sm, ap


def beam_fn(ant, ap):
    d, sp, dsin = ap['d'], ap['sp'], ap['dsin']

    def beam(idx, pts):
        w = np.asarray(pts)[None] - ant[idx][:, None]
        return ((w * d[idx][:, None]).sum(-1) / np.linalg.norm(w, axis=-1) - sp[idx][:, None]) / (dsin / 2)
    return beam


def hann(t):
    t = np.clip(np.asarray(t, np.float64), -1, 1)
    return 0.5 * (1 + np.cos(np.pi * t))


def regions(sm, meta, n=512, frac=(0.25, 0.5, 0.75)):
    """Three region centers on the vendor's grid, at fractions of its azimuth extent and mid-range, moved to the
    brightest-structured 512 by 512 block within a quarter-image neighborhood (amplitude variance), as local points."""
    from sarpy.io.complex.converter import open_complex
    rows, cols = int(sm.ImageData.NumRows), int(sm.ImageData.NumCols)
    out = []
    for f in frac:
        r, c = rows // 2, int(f * cols)
        out.append(dict(row=r, col=c, local=io.sicd_points(sm, np.array([r]), np.array([c]), meta)[0].tolist()))
    return out


try:
    import numba as nb

    @nb.njit(parallel=True, fastmath=False, cache=True)
    def _accum(out, prof, drr, nf2, fref4, ant, ref, p0, pts, lo, hi, wkind, d, sp, dsin2):
        # out[i] += sum over pulses p in this block with lo[i] <= p < hi[i] of w(p, i) * profile_p(dR) exp(j fref4 dR)
        n, m = prof.shape[0], pts.shape[0]
        for i in nb.prange(m):
            x0, x1, x2 = pts[i, 0], pts[i, 1], pts[i, 2]
            acc = 0j
            for q in range(n):
                p = p0 + q
                if p < lo[i] or p >= hi[i]:
                    continue
                w0, w1, w2 = x0 - ant[p, 0], x1 - ant[p, 1], x2 - ant[p, 2]
                rr = np.sqrt(w0 * w0 + w1 * w1 + w2 * w2)
                wt = 1.0
                if wkind == 1:
                    u = ((w0 * d[p, 0] + w1 * d[p, 1] + w2 * d[p, 2]) / rr - sp[p]) / dsin2
                    if u <= -1.0 or u >= 1.0:
                        continue
                    wt = 0.5 * (1.0 + np.cos(np.pi * u))
                dR = rr - ref[p]
                tt = dR / drr + nf2
                k = int(np.floor(tt))
                if k < 0 or k + 1 >= prof.shape[1]:
                    continue
                a = tt - k
                v = prof[q, k] * (1.0 - a) + prof[q, k + 1] * a
                acc += wt * v * np.exp(1j * fref4 * dR)
            out[i] += acc

    @nb.njit(parallel=True, fastmath=False, cache=True)
    def _accum_rx(out, prof, drr, nf2, fref4, tx, txv, ref, p0, pts, lo, hi):
        # as _accum with uniform weights, but with ISCE3's range model: the receiver at transmit position plus velocity
        # times the pixel's own two-way delay, tau = 2 (r.v - c|r|) / (|v|^2 - c^2), r = x - tx (isce3 bistaticDelay)
        n, m = prof.shape[0], pts.shape[0]
        c = 299792458.0
        for i in nb.prange(m):
            acc = 0j
            for q in range(n):
                p = p0 + q
                if p < lo[i] or p >= hi[i]:
                    continue
                w0, w1, w2 = pts[i, 0] - tx[p, 0], pts[i, 1] - tx[p, 1], pts[i, 2] - tx[p, 2]
                rt = np.sqrt(w0 * w0 + w1 * w1 + w2 * w2)
                vv = txv[p, 0] * txv[p, 0] + txv[p, 1] * txv[p, 1] + txv[p, 2] * txv[p, 2]
                rv = w0 * txv[p, 0] + w1 * txv[p, 1] + w2 * txv[p, 2]
                dR = c * (rv - c * rt) / (vv - c * c) - ref[p]
                tt = dR / drr + nf2
                k = int(np.floor(tt))
                if k < 0 or k + 1 >= prof.shape[1]:
                    continue
                a = tt - k
                acc += (prof[q, k] * (1.0 - a) + prof[q, k + 1] * a) * np.exp(1j * fref4 * dR)
            out[i] += acc
except ImportError:
    _accum = _accum_rx = None


def cached_reference(fx, ant, pts, **kw):
    """reference(fx, ant, pts, **kw), kept in REF_CACHE (a directory) under a hash of everything it depends on: the
    points, the antenna path, the pulse ranges, the model options and a fingerprint of the samples. The float64
    reference of a collection region does not change between runs, so later runs read it instead of recomputing."""
    import hashlib
    d = os.environ.get('REF_CACHE')
    h = hashlib.sha1()
    S = fx['S']
    for a in (pts, ant, fx['ref'], np.asarray([fx['fmin'], fx['df']]), np.asarray(S.shape), S[::997, ::97]):
        h.update(np.ascontiguousarray(a).tobytes())
    for k in sorted(kw):
        v = kw[k]
        if isinstance(v, dict):
            v = [v[q] for q in sorted(v)]
        h.update(repr(k).encode())
        for a in (v if isinstance(v, (list, tuple)) else [v]):
            h.update(np.ascontiguousarray(np.asarray(a, dtype=object if np.isscalar(a) else None)).tobytes() if not np.isscalar(a) else repr(a).encode())
    path = None if not d else os.path.join(d, h.hexdigest() + '.npy')
    if path and os.path.exists(path):
        return np.load(path)
    r = reference(fx, ant, pts, **kw)
    if path:
        os.makedirs(d, exist_ok=True)
        np.save(path, r)
    return r


def reference(fx, ant, pts, lo=None, hi=None, beam_ap=None, up=16, block=256, rx=None):
    """float64 exact backprojection at local points pts [m, 3]. Aperture: per-point pulse interval [lo, hi) (uniform
    weight), or, with beam_ap (the dict of load), the Hann window over |u| < 1 of the processed azimuth bandwidth.
    rx=(tx, txv) (local frame) uses ISCE3's range model, the receiver at transmit position plus velocity times each
    pixel's own delay, in place of the per-pulse antenna position ant."""
    import scipy.fft
    S, f0, df, ref = fx['S'], fx['fmin'], fx['df'], fx['ref']
    P, K = S.shape; h = K // 2
    nf = 1 << int(np.ceil(np.log2(up * K))); drr = C / (2 * df * nf)
    fref = f0 + h * df
    pts = np.ascontiguousarray(pts, np.float64); m = len(pts)
    lo = np.zeros(m, np.int64) if lo is None else np.asarray(lo, np.int64)
    hi = np.full(m, P, np.int64) if hi is None else np.asarray(hi, np.int64)
    if beam_ap is not None:
        d, sp, dsin2, wkind = beam_ap['d'], beam_ap['sp'], beam_ap['dsin'] / 2, 1
    else:
        d, sp, dsin2, wkind = np.zeros((P, 3)), np.zeros(P), 1.0, 0
    out = np.zeros(m, np.complex128)
    for p0 in range(0, P, block):
        sl = slice(p0, min(P, p0 + block)); n = sl.stop - sl.start
        if hi.max() <= p0 or lo.min() >= sl.stop:
            continue
        pad = np.zeros((n, nf), np.complex128); pad[:, :K - h] = S[sl, h:]; pad[:, nf - h:] = S[sl, :h]
        prof = scipy.fft.fftshift(scipy.fft.ifft(pad, axis=1, workers=-1), axes=1)
        if rx is not None:
            _accum_rx(out, prof, drr, nf // 2, 4 * np.pi * fref / C, rx[0], rx[1], ref, p0, pts, lo, hi)
        else:
            _accum(out, prof, drr, nf // 2, 4 * np.pi * fref / C, ant, ref, p0, pts, lo, hi, wkind, d, sp, dsin2)
    return out


def err_db(a, b):
    a, b = np.asarray(a, np.complex128).ravel(), np.asarray(b, np.complex128).ravel()
    g = np.vdot(a, b) / np.vdot(a, a)
    return float(10 * np.log10(np.sum(np.abs(g * a - b) ** 2) / np.sum(np.abs(b) ** 2)))
