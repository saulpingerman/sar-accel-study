"""Backprojection in JAX with an explicit precision policy.

The kernel is written in real arithmetic (separate re/im planes) so that any
float type, including bfloat16 which has no complex counterpart, can be used
for each stage independently:

  geom   differential range, range-bin index, phase
  store  range-compressed profiles held on the device
  arith  interpolation and phase rotation
  acc    coherent sum over pulses

Per-pulse scalars (unit line of sight, standoff range) are computed in float64
on the host. Per-pixel work happens on the device in the policy's types.
"""
import math

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

C = 299792458.0

POLICIES = {
    'fp64':        dict(geom='float64', store='float64', arith='float64', acc='float64'),
    'fp32':        dict(geom='float32', store='float32', arith='float32', acc='float32'),
    'fp32_naive':  dict(geom='float32', store='float32', arith='float32', acc='float32', naive=True),
    'bf16_store':  dict(geom='float32', store='bfloat16', arith='float32', acc='float32'),
    'bf16_arith':  dict(geom='float32', store='bfloat16', arith='bfloat16', acc='float32'),
    'bf16_acc':    dict(geom='float32', store='bfloat16', arith='bfloat16', acc='bfloat16'),
    'bf16_all':    dict(geom='bfloat16', store='bfloat16', arith='bfloat16', acc='bfloat16'),
    'f16_arith':   dict(geom='float32', store='float16', arith='float16', acc='float32'),
    'f16_acc':     dict(geom='float32', store='float16', arith='float16', acc='float16'),
}


def needs_x64(policy):
    return any(v == 'float64' for v in POLICIES[policy].values() if isinstance(v, str))


def host_geometry(ant):
    """Float64 per-pulse scalars: unit vector to the radar and standoff range."""
    r0 = np.linalg.norm(ant, axis=1)
    return ant / r0[:, None], r0


def range_bin(col_df, nfft):
    return C / (2.0 * col_df * nfft)


def range_compress(S, nfft):
    """[Np, K] frequency samples -> [Np, nfft] range profiles, dR = 0 at nfft//2.

    The spectrum is centred on sample K//2 so the profile is baseband about
    `Collect.fref`, which keeps linear interpolation error low.
    """
    K = S.shape[1]
    h = K // 2
    z = jnp.zeros((S.shape[0], nfft - K), S.dtype)
    pad = jnp.concatenate([S[:, h:], z, S[:, :h]], axis=1)
    return jnp.fft.fftshift(jnp.fft.ifft(pad, axis=1), axes=1)


def _tree_sum(x):
    """Pairwise sum over axis 0 (length a power of two); order is explicit so
    low-precision results do not depend on a backend's reduction order."""
    while x.shape[0] > 1:
        x = x[0::2] + x[1::2]
    return x[0]


def make_bp(policy, nfft, dr, fc, chunk):
    """Build bp(rc_re, rc_im, u, r0, ant, px, py, pz) -> (img_re, img_im).

    rc_*: [nchunk, chunk, nfft]; u, ant: [nchunk, chunk, 3]; r0: [nchunk, chunk];
    px, py, pz: [P]. `chunk` must be a power of two.
    """
    assert chunk & (chunk - 1) == 0
    p = POLICIES[policy]
    g, ar, ac = (jnp.dtype(p[k]) for k in ('geom', 'arith', 'acc'))
    naive = p.get('naive', False)
    inv_dr, half, kcyc = 1.0 / dr, float(nfft // 2), 2.0 * fc / C

    def bp(rc_re, rc_im, u, r0, ant, px, py, pz):
        px, py, pz = px.astype(g), py.astype(g), pz.astype(g)
        xx = px * px + py * py + pz * pz

        def body(carry, xs):
            rre, rim, u_, r0_, a_ = xs
            if naive:
                dx = px[None, :] - a_[:, 0:1]
                dy = py[None, :] - a_[:, 1:2]
                dz = pz[None, :] - a_[:, 2:3]
                dR = jnp.sqrt(dx * dx + dy * dy + dz * dz) - r0_[:, None]
            else:
                ux = u_[:, 0:1] * px[None, :] + u_[:, 1:2] * py[None, :] + u_[:, 2:3] * pz[None, :]
                q = xx[None, :] / r0_[:, None] - 2 * ux
                dR = q / (1 + jnp.sqrt(1 + q / r0_[:, None]))
            t = dR * jnp.asarray(inv_dr, g) + jnp.asarray(half, g)
            i0f = jnp.floor(t)
            ok = ((i0f >= 0) & (i0f <= nfft - 2)).astype(ar)
            w = (t - i0f).astype(ar)
            i0 = jnp.clip(i0f.astype(jnp.int32), 0, nfft - 2)
            cyc = dR * jnp.asarray(kcyc, g)
            cyc = cyc - jnp.round(cyc)
            ang = cyc * jnp.asarray(2.0 * math.pi, g)
            cs = jnp.cos(ang).astype(ar) * ok
            sn = jnp.sin(ang).astype(ar) * ok
            a_re = jnp.take_along_axis(rre, i0, axis=1).astype(ar)
            b_re = jnp.take_along_axis(rre, i0 + 1, axis=1).astype(ar)
            a_im = jnp.take_along_axis(rim, i0, axis=1).astype(ar)
            b_im = jnp.take_along_axis(rim, i0 + 1, axis=1).astype(ar)
            v_re = a_re + w * (b_re - a_re)
            v_im = a_im + w * (b_im - a_im)
            o_re = (v_re * cs - v_im * sn).astype(ac)
            o_im = (v_re * sn + v_im * cs).astype(ac)
            return (carry[0] + _tree_sum(o_re), carry[1] + _tree_sum(o_im)), None

        z = jnp.zeros(px.shape, ac)
        (ire, iim), _ = lax.scan(body, (z, z), (rc_re, rc_im, u, r0, ant))
        return ire, iim

    return jax.jit(bp)


def chunk_pulses(arrs, chunk):
    """Zero-pad the pulse axis to a multiple of `chunk` and fold it to
    [nchunk, chunk, ...]. Padded pulses carry zero signal and unit range."""
    n = arrs[0].shape[0]
    pad = (-n) % chunk
    out = []
    for a in arrs:
        if pad:
            fill = np.zeros((pad,) + a.shape[1:], a.dtype)
            a = np.concatenate([a, fill], axis=0)
        out.append(a.reshape((-1, chunk) + a.shape[1:]))
    return out


def prepare(policy, S, ant, nfft, chunk, window=None):
    """Host-side packing for `make_bp`: range compression runs on the device in
    complex64 (complex128 for fp64 policies), then profiles are cast to the
    storage type."""
    p = POLICIES[policy]
    st, g = jnp.dtype(p['store']), jnp.dtype(p['geom'])
    cdt = np.complex128 if p['store'] == 'float64' else np.complex64
    if window is not None:
        S = S * window
    rc = jax.jit(range_compress, static_argnums=1)(jnp.asarray(S.astype(cdt)), nfft)
    u, r0 = host_geometry(ant)
    r0p = r0.copy()
    rre, rim = np.asarray(rc.real), np.asarray(rc.imag)
    rre, rim, u, ant_c = chunk_pulses([rre, rim, u, ant], chunk)
    (r0c,) = chunk_pulses([r0p], chunk)
    r0c = np.where(r0c == 0, 1.0, r0c)
    return (jnp.asarray(rre).astype(st), jnp.asarray(rim).astype(st),
            jnp.asarray(u.astype(np.float64) if g == jnp.float64 else u.astype(np.float32)).astype(g),
            jnp.asarray(r0c if g == jnp.float64 else r0c.astype(np.float32)).astype(g),
            jnp.asarray(ant_c if g == jnp.float64 else ant_c.astype(np.float32)).astype(g))


def matched_filter(S, col, px, py, pz):
    """Exact float64 matched filter on the host, the oracle for small images."""
    u = np.stack([px, py, pz], axis=1)
    img = np.zeros(len(px), np.complex128)
    f = col.freqs
    r0 = np.linalg.norm(col.ant, axis=1)
    for p in range(col.Np):
        d = u - col.ant[p]
        dR = np.sqrt(np.einsum('nk,nk->n', d, d)) - r0[p]
        img += np.exp(1j * (4.0 * np.pi / C) * dR[:, None] * f[None, :]) @ S[p]
    return img
