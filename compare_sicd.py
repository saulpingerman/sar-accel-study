#!/usr/bin/env python3
"""Compare the float64 exact-backprojection reference with the vendor's complex image (SICD) of the same collection,
on the vendor's pixel grid (which the reference was formed on).

  python compare_sicd.py --sicd x_SICD.nitf --ref bp_numba_fp64.npy --npz scene.npz --out sicd_<scene>.json [--crops ...]

Reports the registration offset between the two images (from the cross-correlation of block-averaged amplitude,
refined to a fraction of a pixel), the Pearson correlation of log amplitude after alignment, the 5 by 5 coherence
(mean, 0.1 percentile, fraction below 0.99 and 0.9) after a Fourier sub-pixel shift, with the conjugation convention
that gives the higher coherence, and the −3 dB impulse response widths of the brightest isolated point in each given
crop. Optional crops are saved for a figure.
"""
import argparse, json, time

import numpy as np
from scipy.ndimage import uniform_filter


def coherence(a, b, win=5):
    """|<a b*>| / sqrt(<|a|^2><|b|^2>) over win x win windows."""
    ab = uniform_filter(a.real * b.real + a.imag * b.imag, win) + 1j * uniform_filter(a.imag * b.real - a.real * b.imag, win)
    aa = uniform_filter(np.abs(a) ** 2, win)
    bb = uniform_filter(np.abs(b) ** 2, win)
    return np.abs(ab) / np.sqrt(np.maximum(aa * bb, 1e-30))


def block_mean(x, f):
    n0, n1 = (x.shape[0] // f) * f, (x.shape[1] // f) * f
    return x[:n0, :n1].reshape(n0 // f, f, n1 // f, f).mean(axis=(1, 3))


def fourier_shift(x, dy, dx):
    """Shift the complex image by (dy, dx) pixels (positive moves content toward larger indices)."""
    X = np.fft.fft2(x)
    ky = np.fft.fftfreq(x.shape[0])[:, None]
    kx = np.fft.fftfreq(x.shape[1])[None, :]
    return np.fft.ifft2(X * np.exp(-2j * np.pi * (ky * dy + kx * dx))).astype(np.complex64)


def ipr_widths(amp, i, j, half=24):
    """-3 dB widths (pixels) of the peak at (i, j) along the two axes, by linear interpolation of the amplitude."""
    out = []
    for axis in (0, 1):
        line = amp[i - half:i + half + 1, j] if axis == 0 else amp[i, j - half:j + half + 1]
        pk = line[half]
        thr = pk / np.sqrt(2.0)
        lo = half
        while lo > 0 and line[lo] > thr:
            lo -= 1
        hi = half
        while hi < 2 * half and line[hi] > thr:
            hi += 1
        f_lo = (line[lo + 1] - thr) / max(line[lo + 1] - line[lo], 1e-30)
        f_hi = (line[hi - 1] - thr) / max(line[hi - 1] - line[hi], 1e-30)
        out.append(float((hi - 1 + f_hi) - (lo + 1 - f_lo)))
    return out


def local_shift(a, b):
    """Shift (dy, dx) by which b must move to match a, from the cross-correlation of log amplitude with parabolic refinement."""
    la, lb = np.log1p(np.abs(a)), np.log1p(np.abs(b))
    la = la - la.mean(); lb = lb - lb.mean()
    cc = np.fft.ifft2(np.fft.fft2(la) * np.conj(np.fft.fft2(lb))).real
    pk = np.unravel_index(np.argmax(cc), cc.shape)
    out = []
    for ax, (k, n) in enumerate(zip(pk, cc.shape)):
        if ax == 0:
            y0, y1, y2 = cc[(k - 1) % n, pk[1]], cc[k, pk[1]], cc[(k + 1) % n, pk[1]]
        else:
            y0, y1, y2 = cc[pk[0], (k - 1) % n], cc[pk[0], k], cc[pk[0], (k + 1) % n]
        d = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / d if d != 0 else 0.0
        out.append((k if k <= n // 2 else k - n) + frac)
    peak = float(cc.max() / np.sqrt((la * la).sum() * (lb * lb).sum()))
    return out, peak


def deramp(a, b):
    """Remove the linear phase difference between a and b (the peak of the cross-spectrum of a b*), returning b
    multiplied by the ramp and the ramp in cycles per pixel."""
    X = np.fft.fft2(a * np.conj(b))
    pk = np.unravel_index(np.argmax(np.abs(X)), X.shape)
    ky = (pk[0] if pk[0] <= X.shape[0] // 2 else pk[0] - X.shape[0]) / X.shape[0]
    kx = (pk[1] if pk[1] <= X.shape[1] // 2 else pk[1] - X.shape[1]) / X.shape[1]
    iy = np.arange(a.shape[0])[:, None]
    ix = np.arange(a.shape[1])[None, :]
    return (b * np.exp(2j * np.pi * (ky * iy + kx * ix))).astype(np.complex64), [float(ky), float(kx)]


def window_grid(R, Vb, info, origin, model, win=512, step=1024):
    """Windows across the aligned images: measured offset of the vendor image, the model displacement, and the
    coherence and log-amplitude correlation after local alignment and phase-ramp removal."""
    out = []
    n0, n1 = R.shape
    mg = 64
    for i0 in range(step // 2, n0 - win, step):
        for j0 in range(step // 2, n1 - win, step):
            a = R[i0:i0 + win, j0:j0 + win]
            if i0 - mg < 0 or j0 - mg < 0 or i0 + win + mg > n0 or j0 + win + mg > n1:
                continue
            bw = Vb[i0 - mg:i0 + win + mg, j0 - mg:j0 + win + mg]
            (sy, sx), peak = local_shift(a, bw[mg:-mg, mg:-mg])
            if peak < 0.05 or abs(sy) > mg - 4 or abs(sx) > mg - 4:
                out.append(dict(i=i0 + win // 2 + origin[0], j=j0 + win // 2 + origin[1], peak=peak, valid=False))
                continue
            bs = fourier_shift(bw, sy, sx)[mg:-mg, mg:-mg]
            bd, ramp = deramp(a, bs)
            coh = coherence(a, bd)[8:-8, 8:-8]
            la, lb = np.log(np.abs(a) + 1e-12), np.log(np.abs(bs) + 1e-12)
            corr = float(np.corrcoef(la.ravel(), lb.ravel())[0, 1])
            rec = dict(i=i0 + win // 2 + origin[0], j=j0 + win // 2 + origin[1], peak=peak, valid=True,
                       measured=[-sy, -sx], ramp=ramp, coh_mean=float(coh.mean()), coh_median=float(np.median(coh)), logamp_corr=corr)
            if model is not None:
                dx, dy = model(rec['i'], rec['j'])
                rec['model'] = [float(dx), float(dy)]
            out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sicd', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--crops', default='', help='name:i0,j0,h,w;... regions (our image indices) saved for figures')
    ap.add_argument('--crops-out', default='')
    a = ap.parse_args()
    t0 = time.time()
    from sarpy.io.complex.converter import open_complex
    r = open_complex(a.sicd)
    sm = r.sicd_meta
    info = json.loads(str(np.load(a.npz, allow_pickle=True)['info']))
    range_is_row = info['sicd']['range_is_row']
    V = r[:, :].astype(np.complex64)                                          # [rows, cols]
    if range_is_row:
        V = np.ascontiguousarray(V.T)                                       # -> [azimuth, range] like our images
    ref = np.load(a.ref).astype(np.complex64)
    print('sicd', V.shape, 'ref', ref.shape, f'{time.time() - t0:.0f}s', flush=True)
    assert V.shape == ref.shape, (V.shape, ref.shape)
    res = dict(scene=info.get('name', ''), shape=list(ref.shape), range_is_row=bool(range_is_row),
               sicd_weighting=[str(getattr(sm.Grid.Row.WgtType, 'WindowName', '')) if sm.Grid.Row.WgtType else '', str(getattr(sm.Grid.Col.WgtType, 'WindowName', '')) if sm.Grid.Col.WgtType else ''],
               sicd_image_formation=str(getattr(getattr(sm, 'ImageFormation', None), 'ImageFormAlgo', '')))
    model = None
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import prep
        from dev import pfa2
        col, _, grid = prep.load(a.npz)
        nx, ny, spx, spy = grid['nx'], grid['ny'], grid['spx'], grid['spy']
        dist = pfa2.distortion(col, nx, ny, spx, spy, e1=grid['e1'], e2=grid['e2'])
        powers, cx, cy, sx, sy = dist['powers'], dist['cx'], dist['cy'], dist['sx'], dist['sy']

        def model(i, j):
            x, y = (np.asarray(i, np.float64) - nx / 2.0) * spx, (np.asarray(j, np.float64) - ny / 2.0) * spy
            Bv = np.stack([(x / sx) ** p_ * (y / sy) ** q_ for p_, q_ in powers], -1)
            return (Bv @ cx) / spx, (Bv @ cy) / spy
        res['model_max_shift_px'] = dist['max_shift_px']
    except Exception as e:                                                   # the comparison stands without the model
        res['model_error'] = f'{type(e).__name__}: {e}'
    res.update(compare(V, ref, info, a.crops, a.crops_out, model))
    res['seconds'] = time.time() - t0
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'points'}, indent=1))
    print(json.dumps(res['points'], indent=1))


def compare(V, ref, info, crops_spec='', crops_out='', model=None):
    """V, ref complex64 [azimuth, range] on the same grid -> dict of registration, amplitude and coherence statistics.
    model: optional function (i, j) -> (dx, dy) pixels, the displacement a polar-format image is expected to show."""
    res = {}
    # 1. registration: cross-correlation of block-averaged amplitude (log), coarse then fine
    f = 4
    A = np.log1p(block_mean(np.abs(ref), f))
    Bm = np.log1p(block_mean(np.abs(V), f))
    A -= A.mean(); Bm -= Bm.mean()
    cc = np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(Bm))).real
    pk = np.unravel_index(np.argmax(cc), cc.shape)
    sh = [int(pk[0]) if pk[0] <= cc.shape[0] // 2 else int(pk[0]) - cc.shape[0], int(pk[1]) if pk[1] <= cc.shape[1] // 2 else int(pk[1]) - cc.shape[1]]
    coarse = [sh[0] * f, sh[1] * f]
    # fine: full-resolution amplitude cross-correlation on a central window after the coarse shift
    n0, n1 = ref.shape
    h = min(2048, n0 // 2 - abs(coarse[0]) - 8), min(2048, n1 // 2 - abs(coarse[1]) - 8)
    c0, c1 = n0 // 2, n1 // 2
    Aw = np.abs(ref[c0 - h[0]:c0 + h[0], c1 - h[1]:c1 + h[1]])
    Bw = np.abs(V[c0 - h[0] - coarse[0]:c0 + h[0] - coarse[0], c1 - h[1] - coarse[1]:c1 + h[1] - coarse[1]])
    Aw = np.log1p(Aw); Bw = np.log1p(Bw); Aw -= Aw.mean(); Bw -= Bw.mean()
    cc = np.fft.ifft2(np.fft.fft2(Aw) * np.conj(np.fft.fft2(Bw))).real
    pk = np.unravel_index(np.argmax(cc), cc.shape)
    p0 = int(pk[0]) if pk[0] <= cc.shape[0] // 2 else int(pk[0]) - cc.shape[0]
    p1 = int(pk[1]) if pk[1] <= cc.shape[1] // 2 else int(pk[1]) - cc.shape[1]
    # parabolic sub-pixel refinement
    def sub(c, i, j, ax):
        if ax == 0:
            y0, y1, y2 = c[(i - 1) % c.shape[0], j], c[i, j], c[(i + 1) % c.shape[0], j]
        else:
            y0, y1, y2 = c[i, (j - 1) % c.shape[1]], c[i, j], c[i, (j + 1) % c.shape[1]]
        d = y0 - 2 * y1 + y2
        return float(0.5 * (y0 - y2) / d) if d != 0 else 0.0
    d0, d1 = sub(cc, pk[0], pk[1], 0), sub(cc, pk[0], pk[1], 1)
    shift = [coarse[0] + p0 + d0, coarse[1] + p1 + d1]                    # ref = sicd moved by this many pixels
    res['shift_parabolic'] = list(shift)
    # the conjugation convention and the linear phase ramp (spectral centering) from the cross-spectrum of a central
    # window, applied to the whole vendor image before the coherence refinement
    i0c, j0c = int(np.round(shift[0])), int(np.round(shift[1]))
    hc = min(1024, min(c0, n0 - c0, c1, n1 - c1) - max(abs(i0c), abs(j0c)) - 8)
    Rc = ref[c0 - hc:c0 + hc, c1 - hc:c1 + hc]
    Vc = V[c0 - hc - i0c:c0 + hc - i0c, c1 - hc - j0c:c1 + hc - j0c]
    cand = []
    for conj in (False, True):
        X = np.abs(np.fft.fft2(Rc * np.conj(np.conj(Vc) if conj else Vc)))
        pk = np.unravel_index(np.argmax(X), X.shape)
        ky = (pk[0] if pk[0] <= X.shape[0] // 2 else pk[0] - X.shape[0]) / X.shape[0]
        kx = (pk[1] if pk[1] <= X.shape[1] // 2 else pk[1] - X.shape[1]) / X.shape[1]
        cand.append((float(X.max() / X.mean()), conj, ky, kx))
    _, use_conj, ky, kx = max(cand)
    res['conjugate'] = bool(use_conj)
    res['phase_ramp_cycles_per_px'] = [float(ky), float(kx)]
    res['ramp_peak_to_mean'] = [c[0] for c in cand]
    iy = np.arange(n0, dtype=np.float64)[:, None]
    ix = np.arange(n1, dtype=np.float64)[None, :]
    if use_conj:
        V = np.conj(V)
    V = (V * np.exp(2j * np.pi * (ky * iy + kx * ix))).astype(np.complex64)
    del iy, ix
    print('conjugate', use_conj, 'ramp', ky, kx, flush=True)
    # refine the fractional part by maximizing the mean coherence on a central window
    avail = min(c0, n0 - c0, c1, n1 - c1) - max(abs(i0c), abs(j0c)) - 8
    hw = min(512, avail * 4 // 5)
    mg = hw // 4
    Rw = ref[c0 - hw:c0 + hw, c1 - hw:c1 + hw]
    Vw = V[c0 - hw - mg - i0c:c0 + hw + mg - i0c, c1 - hw - mg - j0c:c1 + hw + mg - j0c]

    def score(fy, fx, conj):
        Vs = fourier_shift(Vw, fy, fx)[mg:-mg, mg:-mg]
        return float(coherence(Rw, np.conj(Vs) if conj else Vs)[8:-8, 8:-8].mean())
    best = None
    for conj in (False,):
        fy, fx = shift[0] - i0c, shift[1] - j0c
        for step in (0.1, 0.02, 0.005):
            cand = [(fy + a_ * step, fx + b_ * step) for a_ in range(-2, 3) for b_ in range(-2, 3)]
            sc = [(score(y_, x_, conj), y_, x_) for y_, x_ in cand]
            _, fy, fx = max(sc)
        v = score(fy, fx, conj)
        if best is None or v > best[0]:
            best = (v, conj, fy, fx)
    shift = [i0c + best[2], j0c + best[3]]
    res['shift_pixels'] = shift
    res['refine_window_coherence'] = best[0]
    res['shift_m'] = [shift[0] * info['spx'], shift[1] * info['spy']]
    print('shift (pixels)', shift, flush=True)
    # 2. align the vendor image to ours: Fourier shift by the fractional part, integer part by slicing
    i0, j0 = int(np.round(shift[0])), int(np.round(shift[1]))
    fy, fx = shift[0] - i0, shift[1] - j0
    Vs = fourier_shift(V, fy, fx) if (abs(fy) > 1e-3 or abs(fx) > 1e-3) else V
    # overlapping region after the integer shift
    ya, yb = max(0, i0), min(n0, n0 + i0)
    xa, xb = max(0, j0), min(n1, n1 + j0)
    R = ref[ya:yb, xa:xb]
    Vv = Vs[ya - i0:yb - i0, xa - j0:xb - j0]
    del V, Vs
    # 3. amplitude correlation (log) and intensity scale
    la, lb = np.log(np.abs(R) + 1e-12), np.log(np.abs(Vv) + 1e-12)
    m = (np.abs(R) > 0) & (np.abs(Vv) > 0)
    res['log_amplitude_correlation'] = float(np.corrcoef(la[m].ravel()[::7], lb[m].ravel()[::7])[0, 1])
    res['gain_db'] = float(20 * np.log10(np.sqrt((np.abs(Vv) ** 2).mean() / (np.abs(R) ** 2).mean())))
    # 4. coherence (convention and ramp already applied)
    coh = coherence(R, Vv)
    res['coherence'] = dict(coh_mean=float(coh.mean()), coh_p001=float(np.percentile(coh, 0.1)), coh_p01=float(np.percentile(coh, 1)),
                            coh_median=float(np.median(coh)), below_099=float((coh < 0.99).mean()), below_09=float((coh < 0.9).mean()))
    print('coherence', res['coherence'], flush=True)
    Vb = Vv
    # 5. impulse response widths at the brightest point of each crop
    crops = []
    for c in [c for c in crops_spec.split(';') if c]:
        name, box = c.split(':')
        ci, cj, hh, ww = [int(v) for v in box.split(',')]
        ci -= ya; cj -= xa
        sub_r = np.abs(R[ci:ci + hh, cj:cj + ww])
        if sub_r.size == 0:
            continue
        pi, pj = np.unravel_index(np.argmax(sub_r), sub_r.shape)
        pi += ci; pj += cj
        if 30 < pi < R.shape[0] - 30 and 30 < pj < R.shape[1] - 30:
            w_ref = ipr_widths(np.abs(R), pi, pj)
            sub_v = np.abs(Vb[pi - 3:pi + 4, pj - 3:pj + 4])
            qi, qj = np.unravel_index(np.argmax(sub_v), sub_v.shape)
            w_v = ipr_widths(np.abs(Vb), pi - 3 + qi, pj - 3 + qj)
            crops.append(dict(name=name, peak=[int(pi + ya), int(pj + xa)], ipr_ref_px=w_ref, ipr_sicd_px=w_v,
                              ipr_ref_m=[w_ref[0] * info['spx'], w_ref[1] * info['spy']], ipr_sicd_m=[w_v[0] * info['spx'], w_v[1] * info['spy']]))
    res['points'] = crops
    res['windows'] = window_grid(R, Vb, info, (ya, xa), model)
    if model is not None:
        # the vendor image resampled through the model (V_true(p) = V(p + d(p))), then compared globally
        from scipy.ndimage import map_coordinates
        Vw = np.empty_like(Vb)
        bs = 1024
        ii_all = np.arange(R.shape[0]) + ya
        jj_all = np.arange(R.shape[1]) + xa
        for b0 in range(0, R.shape[0], bs):
            b1 = min(R.shape[0], b0 + bs)
            I, J = np.meshgrid(ii_all[b0:b1].astype(np.float64), jj_all.astype(np.float64), indexing='ij')
            dx, dy = model(I, J)
            ci = (I - ya) + dx
            cj = (J - xa) + dy
            lo_i, hi_i = max(0, int(np.floor(ci.min())) - 3), min(Vb.shape[0], int(np.ceil(ci.max())) + 4)
            sub = Vb[lo_i:hi_i]
            Vw[b0:b1] = (map_coordinates(sub.real, [ci - lo_i, cj], order=3, mode='nearest') +
                         1j * map_coordinates(sub.imag, [ci - lo_i, cj], order=3, mode='nearest')).astype(np.complex64)
        cohw = coherence(R, Vw)
        edge = 16
        c_in = cohw[edge:-edge, edge:-edge]
        la_, lb_ = np.log(np.abs(R[edge:-edge, edge:-edge]) + 1e-12), np.log(np.abs(Vw[edge:-edge, edge:-edge]) + 1e-12)
        res['warped'] = dict(coh_mean=float(c_in.mean()), coh_median=float(np.median(c_in)), coh_p01=float(np.percentile(c_in, 1)),
                             below_09=float((c_in < 0.9).mean()), below_05=float((c_in < 0.5).mean()),
                             log_amplitude_correlation=float(np.corrcoef(la_.ravel()[::7], lb_.ravel()[::7])[0, 1]))
        print('warped', res['warped'], flush=True)
        # the windows again on the warped image: residual offsets should vanish
        res['windows_warped'] = window_grid(R, Vw, info, (ya, xa), None)
        Vb, coh = Vw, cohw
    if crops_spec and crops_out:
        saved = {}
        for c in [c for c in crops_spec.split(';') if c]:
            name, box = c.split(':')
            ci, cj, hh, ww = [int(v) for v in box.split(',')]
            ci -= ya; cj -= xa
            saved[f'{name}/ref'] = R[ci:ci + hh, cj:cj + ww]
            saved[f'{name}/sicd'] = Vb[ci:ci + hh, cj:cj + ww]
            saved[f'{name}/coh'] = coh[ci:ci + hh, cj:cj + ww].astype(np.float32)
        np.savez_compressed(crops_out, **saved)
    return res


if __name__ == '__main__':
    main()
