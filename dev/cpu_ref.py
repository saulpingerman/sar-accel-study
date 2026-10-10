"""Multithreaded CPU implementations (Numba + SciPy FFT).

XLA's CPU backend runs most element-wise work on one thread, so the JAX code
understates what a CPU can do. These kernels mirror the JAX algorithms and use
every core, and are what the CPU cost figures are based on.

Numba promotes float32 mixed with a Python literal to float64, so every
constant is passed in as a scalar of the working type.
"""
import numpy as np
import scipy.fft
from numba import njit, prange


def range_compress(S, nfft):
    K = S.shape[1]
    h = K // 2
    pad = np.zeros((S.shape[0], nfft), S.dtype)
    pad[:, :K - h] = S[:, h:]
    pad[:, nfft - h:] = S[:, :h]
    return scipy.fft.fftshift(scipy.fft.ifft(pad, axis=1, workers=-1), axes=1)


@njit(parallel=True, fastmath=True, cache=True)
def _bp(rc_re, rc_im, u, r0, px, py, pz, inv_dr, half, kcyc, twopi, one, two, zero, top, blk):
    P = px.size
    Np, nfft = rc_re.shape
    o_re = np.zeros(P, rc_re.dtype)
    o_im = np.zeros(P, rc_re.dtype)
    nblk = (P + blk - 1) // blk
    for b in prange(nblk):
        lo = b * blk
        hi = min(P, lo + blk)
        for p in range(Np):
            u0, u1, u2 = u[p, 0], u[p, 1], u[p, 2]
            ir = one / r0[p]
            for i in range(lo, hi):
                x, y, z = px[i], py[i], pz[i]
                q = (x * x + y * y + z * z) * ir - two * (u0 * x + u1 * y + u2 * z)
                dR = q / (one + np.sqrt(one + q * ir))
                t = dR * inv_dr + half
                fl = np.floor(t)
                ok = one if (fl >= 0 and fl <= top) else zero
                i0 = min(max(np.int32(fl), 0), nfft - 2)
                w = t - fl
                vr = rc_re[p, i0] + w * (rc_re[p, i0 + 1] - rc_re[p, i0])
                vi = rc_im[p, i0] + w * (rc_im[p, i0 + 1] - rc_im[p, i0])
                cyc = dR * kcyc
                ang = (cyc - np.rint(cyc)) * twopi
                cs, sn = np.cos(ang) * ok, np.sin(ang) * ok
                o_re[i] += vr * cs - vi * sn
                o_im[i] += vr * sn + vi * cs
    return o_re, o_im


def bp(rc, u, r0, px, py, pz, dr, fc, blk=4096):
    """Backprojection of range profiles `rc` (complex64 or complex128)."""
    f = np.float32 if rc.dtype == np.complex64 else np.float64
    C = 299792458.0
    re, im = _bp(np.ascontiguousarray(rc.real), np.ascontiguousarray(rc.imag),
                 u.astype(f), r0.astype(f), px.astype(f), py.astype(f), pz.astype(f),
                 f(1.0 / dr), f(rc.shape[1] // 2), f(2.0 * fc / C), f(2 * np.pi), f(1), f(2), f(0), f(rc.shape[1] - 2), blk)
    return re + 1j * im


@njit(parallel=True, fastmath=True, cache=True)
def _resample_rows(d, a, b, M, taps, pi, pi_over, halfc, tiny):
    """out[r, m] = windowed-sinc sample of row d[r] at index a[r] + b[r] * m."""
    R, L = d.shape
    out = np.zeros((R, M), d.dtype)
    h = taps // 2 - 1
    for r in prange(R):
        for m in range(M):
            idx = a[r] + b[r] * m
            fl = np.floor(idx)
            base = int(fl) - h
            if base < 0 or base + taps - 1 > L - 1:
                continue
            fr = idx - fl
            acc = d[r, 0] * tiny
            ws = tiny
            for j in range(taps):
                x = fr - (j - h)
                px_ = pi * x
                s = np.sin(px_) / px_ if abs(px_) > tiny else tiny / tiny
                wgt = s * (halfc + halfc * np.cos(x * pi_over))
                acc += d[r, base + j] * wgt
                ws += wgt
            out[r, m] = acc / ws
    return out


def pfa(S, geo, nfx, nfy, win, taps=8):
    """Polar format; S complex64 or complex128, coefficients cast to match."""
    f = np.float32 if S.dtype == np.complex64 else np.float64
    nkx, nky = geo['nkx'], geo['nky']
    k = (f(np.pi), f(np.pi / (taps // 2)), f(0.5), f(1e-12))
    a = _resample_rows(S, geo['alpha'].astype(f), geo['beta'].astype(f), nkx, taps, *k)
    gam = geo['gam'].astype(f)
    off = (geo['delta'] - geo['gam'] * (nky // 2)).astype(f)
    b = _resample_rows(np.ascontiguousarray(a.T), off, gam, nky, taps, *k)
    G = np.zeros((nfx, nfy), S.dtype)
    py0 = (nfy - nky) // 2
    G[:nkx, py0:py0 + nky] = b * win.astype(f)
    G = scipy.fft.ifftshift(G, axes=1)
    img = scipy.fft.ifft(scipy.fft.fft(G, axis=1, workers=-1), axis=0, workers=-1)
    return scipy.fft.fftshift(img)
