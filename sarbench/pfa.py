"""Polar format algorithm in JAX with the same precision policies as bp.py.

Two separable windowed-sinc resampling passes take the polar phase history to
a rectangular wavenumber grid, followed by a 2-D FFT. JAX has no bfloat16
complex type, so the FFT always runs in complex64 (complex128 for fp64) and
the low-precision policies apply to storage and resampling only.
"""
import math

import numpy as np
import jax
import jax.numpy as jnp

from .bp import POLICIES

C = 299792458.0


def pfa_geometry(col, taps=8, grid_from=None):
    """Float64 host-side resampling coefficients.

    Assumes the ground-plane tangent of the look angle is linear in pulse
    index (true for a straight track); the fit residual is asserted. Pass
    `grid_from` (another collect's geometry dict) to reuse its wavenumber grid
    so a repeat-pass pair lands on identical pixels and support.
    """
    u = col.ant / np.linalg.norm(col.ant, axis=1)[:, None]
    ux = -u[:, 0]
    t = u[:, 1] / ux
    p = np.arange(col.Np)
    dt, t0 = np.polyfit(p, t, 1)
    assert np.abs(t - (t0 + dt * p)).max() < 1e-3 * abs(dt), 'look angle not linear in pulse'
    m = taps // 2
    if grid_from is None:
        fa, fb = col.fmin + m * col.df, col.fmin + (col.K - 1 - m) * col.df
        kx_min = 4 * np.pi * fa / C * ux.max()
        kx_max = 4 * np.pi * fb / C * ux.min()
        nkx, nky = col.K, col.Np
        dkx = (kx_max - kx_min) / (nkx - 1)
        dky = kx_min * abs(dt) * (col.Np - 1 - 2 * m) / nky
    else:
        kx_min, dkx, dky, nkx, nky = (grid_from[k] for k in ('kx_min', 'dkx', 'dky', 'nkx', 'nky'))
    kx = kx_min + dkx * np.arange(nkx)
    scale = C / (4 * np.pi * ux * col.df)
    return dict(alpha=kx_min * scale - col.fmin / col.df,   # [Np] freq index at kx_min
                beta=dkx * scale,                           # [Np] freq index step per kx sample
                gam=dky / (dt * kx),                        # [nkx] pulse index step per ky sample
                delta=-t0 / dt,                             # pulse index at ky = 0
                kx_min=kx_min, dkx=dkx, dky=dky, nkx=nkx, nky=nky)


def pixel_spacing(geo, nfx, nfy):
    return 2 * np.pi / (nfx * geo['dkx']), 2 * np.pi / (nfy * geo['dky'])


def _resample(d_re, d_im, idx, taps, ar):
    """Windowed-sinc resampling along axis 1. d_*: [R, L], idx: [R, M] (geom type)."""
    L = d_re.shape[1]
    g = idx.dtype
    i0f = jnp.floor(idx)
    fr = idx - i0f
    base = i0f.astype(jnp.int32) - (taps // 2 - 1)
    ok = ((base >= 0) & (base + taps - 1 <= L - 1)).astype(ar)
    o_re = jnp.zeros(idx.shape, ar)
    o_im = jnp.zeros(idx.shape, ar)
    wsum = jnp.zeros(idx.shape, g)
    for j in range(taps):
        d = fr - (j - (taps // 2 - 1))
        wgt = jnp.sinc(d) * (0.5 + 0.5 * jnp.cos(d * jnp.asarray(math.pi / (taps // 2), g)))
        ii = jnp.clip(base + j, 0, L - 1)
        wa = wgt.astype(ar)
        o_re = o_re + jnp.take_along_axis(d_re, ii, axis=1).astype(ar) * wa
        o_im = o_im + jnp.take_along_axis(d_im, ii, axis=1).astype(ar) * wa
        wsum = wsum + wgt
    inv = (1.0 / wsum).astype(ar) * ok
    return o_re * inv, o_im * inv


def make_pfa(policy, nkx, nky, nfx, nfy, taps=8):
    """Build pfa(S_re, S_im, alpha, beta, gam, delta, win) -> complex image [nfx, nfy].

    S_*: [Np, K] in the storage type; alpha, beta: [Np]; gam: [nkx]; delta: scalar;
    win: [nkx, nky] taper applied on the rectangular grid.
    """
    p = POLICIES[policy]
    g, ar = jnp.dtype(p['geom']), jnp.dtype(p['arith'])
    f = jnp.float64 if p['acc'] == 'float64' else jnp.float32
    py0 = (nfy - nky) // 2

    def pfa(S_re, S_im, alpha, beta, gam, delta, win):
        m = jnp.arange(nkx, dtype=jnp.int32).astype(g)
        n = (jnp.arange(nky, dtype=jnp.int32) - nky // 2).astype(g)
        idx1 = alpha.astype(g)[:, None] + m[None, :] * beta.astype(g)[:, None]
        a_re, a_im = _resample(S_re, S_im, idx1, taps, ar)              # [Np, nkx]
        idx2 = n[None, :] * gam.astype(g)[:, None] + delta.astype(g)
        b_re, b_im = _resample(a_re.T, a_im.T, idx2, taps, ar)          # [nkx, nky]
        G = lax_complex(b_re.astype(f) * win, b_im.astype(f) * win)
        G = jnp.pad(G, ((0, nfx - nkx), (py0, nfy - nky - py0)))
        G = jnp.fft.ifftshift(G, axes=1)
        img = jnp.fft.ifft(jnp.fft.fft(G, axis=1), axis=0)
        return jnp.fft.fftshift(img)

    return jax.jit(pfa)


def lax_complex(re, im):
    return jax.lax.complex(re, im)


def prepare(policy, S, geo):
    p = POLICIES[policy]
    st, g = jnp.dtype(p['store']), jnp.dtype(p['geom'])
    h = np.float64 if p['store'] == 'float64' else np.float32
    hg = np.float64 if p['geom'] == 'float64' else np.float32
    f = np.float64 if p['acc'] == 'float64' else np.float32
    from scipy.signal.windows import taylor
    win = (taylor(geo['nkx'], nbar=4, sll=35, norm=False)[:, None] *
           taylor(geo['nky'], nbar=4, sll=35, norm=False)[None, :]).astype(f)
    return (jnp.asarray(S.real.astype(h)).astype(st), jnp.asarray(S.imag.astype(h)).astype(st),
            jnp.asarray(geo['alpha'].astype(hg)).astype(g), jnp.asarray(geo['beta'].astype(hg)).astype(g),
            jnp.asarray(geo['gam'].astype(hg)).astype(g), jnp.asarray(hg(geo['delta'])).astype(g),
            jnp.asarray(win))


# ------------------------------------------------- matrix-multiply variant

MM_POLICIES = {
    # resampling policy, matmul operand type
    'fp32':       ('fp32', 'float32'),
    'bf16_mm':    ('fp32', 'bfloat16'),
    'bf16_arith': ('bf16_arith', 'bfloat16'),
    'f16_mm':     ('fp32', 'float16'),
}


def dft_matrices(geo, nx, ny, sx, sy):
    """Float64 DFT matrices that put the image on an nx x ny grid with pixel
    spacing (sx, sy). With sx, sy = pixel_spacing(geo, nx, ny) the result is
    identical to the FFT path. Stacked as [2 nx, nkx] and [nky, 2 ny]."""
    x = (np.arange(nx) - nx // 2) * sx
    y = (np.arange(ny) - ny // 2) * sy
    ph_x = np.outer(x, geo['dkx'] * np.arange(geo['nkx']))
    ph_y = -np.outer(geo['dky'] * (np.arange(geo['nky']) - geo['nky'] // 2), y)
    X = np.concatenate([np.cos(ph_x), np.sin(ph_x)], axis=0)
    Y = np.concatenate([np.cos(ph_y), np.sin(ph_y)], axis=1)
    return X, Y


def make_pfa_mm(policy, nkx, nky, nx, ny, taps=8):
    """Polar format with the 2-D FFT replaced by two dense matrix products.

    This costs O(N^3) multiply-adds instead of O(N^2 log N), but they are the
    one operation a TPU matrix unit (or GPU tensor cores) runs at full speed in
    16-bit floats. Products accumulate into float32.
    pfa_mm(S_re, S_im, alpha, beta, gam, delta, win, X, Y) -> (img_re, img_im).
    """
    rpol, mm = MM_POLICIES[policy]
    p = POLICIES[rpol]
    g, ar, mm = jnp.dtype(p['geom']), jnp.dtype(p['arith']), jnp.dtype(mm)
    f = jnp.float32

    def pfa_mm(S_re, S_im, alpha, beta, gam, delta, win, X, Y):
        m = jnp.arange(nkx, dtype=jnp.int32).astype(g)
        n = (jnp.arange(nky, dtype=jnp.int32) - nky // 2).astype(g)
        idx1 = alpha.astype(g)[:, None] + m[None, :] * beta.astype(g)[:, None]
        a_re, a_im = _resample(S_re, S_im, idx1, taps, ar)
        idx2 = n[None, :] * gam.astype(g)[:, None] + delta.astype(g)
        b_re, b_im = _resample(a_re.T, a_im.T, idx2, taps, ar)
        # operands are scaled to unit peak before the 16-bit cast so IEEE half cannot overflow
        G = jnp.concatenate([b_re.astype(f) * win, b_im.astype(f) * win], axis=1)              # [nkx, 2 nky]
        s1 = jnp.max(jnp.abs(G))
        T = jnp.matmul(X, (G / s1).astype(mm), preferred_element_type=f)                       # [2 nx, 2 nky]
        t_re = T[:nx, :nky] - T[nx:, nky:]
        t_im = T[:nx, nky:] + T[nx:, :nky]
        T2 = jnp.concatenate([t_re, t_im], axis=0)                                             # [2 nx, nky]
        s2 = jnp.max(jnp.abs(T2))
        U = jnp.matmul((T2 / s2).astype(mm), Y, preferred_element_type=f)                      # [2 nx, 2 ny]
        s = s1 * s2 / nx                                                                       # 1/nx matches ifft
        return (U[:nx, :ny] - U[nx:, ny:]) * s, (U[:nx, ny:] + U[nx:, :ny]) * s

    return jax.jit(pfa_mm)


def prepare_mm(policy, S, geo, nx, ny, sx, sy):
    rpol, mm = MM_POLICIES[policy]
    X, Y = dft_matrices(geo, nx, ny, sx, sy)
    return prepare(rpol, S, geo) + (jnp.asarray(X.astype(np.float32)).astype(mm),
                                    jnp.asarray(Y.astype(np.float32)).astype(mm))


# ------------------------------------------------------ chirp-z variant

def _next_pow2(n):
    return 1 << int(np.ceil(np.log2(n)))


def make_pfa_czt(policy, K, Np, nkx, nky, nfx, nfy):
    """Polar format with both resampling passes done by chirp-z transforms.

    Windowed-sinc resampling needs a lookup per tap, which some accelerators
    execute slowly. A chirp-z transform evaluates the same band-limited
    interpolation on a scaled grid using only FFTs and element-wise products.
    The profile is tapered outside the scene so the periodic extension that
    trigonometric interpolation implies does not ring from the band edges.
    pfa_czt(S, alpha, eps_r, delta, eps_a, win) -> complex image [nfx, nfy];
    eps_r = beta - 1 per pulse, eps_a = gam - 1 per kx row.
    """
    x64 = POLICIES[policy]['acc'] == 'float64'
    f = jnp.float64 if x64 else jnp.float32
    two_pi = 2.0 * math.pi
    py0 = (nfy - nky) // 2

    def cis(cyc):
        ang = (cyc - jnp.round(cyc)) * two_pi
        return lax_complex(jnp.cos(ang), jnp.sin(ang))

    def chirp(idx, eps, n, sign):
        """exp(sign j pi (1 + eps) idx^2 / n) for integer idx [M] and eps [R]."""
        i2 = idx * idx
        base = (i2 % (2 * n)).astype(f) / (2 * n)            # exact part
        extra = eps[:, None] * (i2.astype(f) / (2 * n))
        return cis(sign * (base[None, :] + extra))

    def taper(n):
        r = jnp.abs(jnp.arange(n, dtype=jnp.int32) - n // 2).astype(f) / n
        return jnp.where(r <= 0.36, 1.0, jnp.where(r >= 0.50, 0.0, 0.5 + 0.5 * jnp.cos((r - 0.36) * (math.pi / 0.14)))).astype(f)

    def czt(x, shift, eps, n_out, j0):
        """Rows of x [R, n] are spectra; return their trigonometric interpolation
        at fractional index shift + (j0 + j)(1 + eps), j = 0..n_out-1."""
        n = x.shape[1]
        L = _next_pow2(n + n_out)
        r = jnp.arange(n, dtype=jnp.int32) - n // 2
        g = jnp.fft.fftshift(jnp.fft.ifft(x, axis=1), axes=1) * taper(n)[None, :]
        pre = g * cis(-(shift[:, None] * r.astype(f)[None, :]) / n) * chirp(r, eps, n, -1.0)
        xin = jnp.concatenate([pre[:, n // 2:], jnp.zeros((x.shape[0], L - n), pre.dtype), pre[:, :n // 2]], axis=1)
        hi = j0 + n_out - 1 + n // 2
        i = jnp.arange(L, dtype=jnp.int32)
        y = jnp.fft.ifft(jnp.fft.fft(xin, axis=1) * jnp.fft.fft(chirp(jnp.where(i <= hi, i, i - L), eps, n, 1.0), axis=1), axis=1)
        j = jnp.arange(n_out, dtype=jnp.int32) + j0
        y = y[:, :n_out] if j0 == 0 else jnp.concatenate([y[:, L + j0:], y[:, :n_out + j0]], axis=1)
        return y * chirp(j, eps, n, -1.0)

    def pfa_czt(S, alpha, eps_r, delta, eps_a, win):
        a = czt(S, alpha.astype(f), eps_r.astype(f), nkx, 0)                                   # [Np, nkx]
        b = czt(a.T, jnp.broadcast_to(delta.astype(f), (nkx,)), eps_a.astype(f), nky, -(nky // 2))  # [nkx, nky]
        G = jnp.pad(b * win, ((0, nfx - nkx), (py0, nfy - nky - py0)))
        G = jnp.fft.ifftshift(G, axes=1)
        return jnp.fft.fftshift(jnp.fft.ifft(jnp.fft.fft(G, axis=1), axis=0))

    return jax.jit(pfa_czt)


def prepare_czt(policy, S, geo):
    x64 = POLICIES[policy]['acc'] == 'float64'
    f, c = (np.float64, np.complex128) if x64 else (np.float32, np.complex64)
    from scipy.signal.windows import taylor
    win = (taylor(geo['nkx'], nbar=4, sll=35, norm=False)[:, None] *
           taylor(geo['nky'], nbar=4, sll=35, norm=False)[None, :]).astype(f)
    return (jnp.asarray(S.astype(c)), jnp.asarray(geo['alpha'].astype(f)), jnp.asarray((geo['beta'] - 1.0).astype(f)),
            jnp.asarray(f(geo['delta'])), jnp.asarray((geo['gam'] - 1.0).astype(f)), jnp.asarray(win))
