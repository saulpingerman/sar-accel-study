"""Fused phase rotation and frequency-axis decimation for the GPU (Pallas, Triton backend).

Same algorithm as pallas_ffbp.py, tiled for a GPU: each program takes pb pulses and one block of kob output columns,
loads the input window it needs (win columns of the padded phase history), rotates it with the direct ramp (the
GPU's special-function units make the sines and cosines cheap), and multiplies it by the banded block of the
decimation matrix on the tensor cores. The phase history is read from HBM about once per child (windows overlap by
the filter length only) and the decimated result written once.

    y_re, y_im = fused_rotate_dec_k_gpu(S_re, S_im, c0, slope, band, kc, policy)   each [P, Ko_pad]
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl

TWO_PI = 2.0 * math.pi


def band_blocks_gpu(Fk, D, kob=16, win=None, pad_align=16):
    """Decimation matrix Fk [K, Ko] cut into [nb * win, kob] blocks on the padded column index (window widths a power
    of two for Triton). The blocks are small because the window and the filter block share the SM's shared memory
    (99 KB on the L4): kob = 16 outputs span 6 * 16 = 96 input columns, a 256-column window with the filter length.
    Returns dict(blocks, Kpad, pl_pad, Ko_pad, kob, win, stride, nb, Ko, K)."""
    K, Ko = Fk.shape
    nz = np.nonzero(Fk[:, Ko // 2])[0]
    L = int(nz[-1] - nz[0] + 1)
    first0 = int(np.nonzero(Fk[:, 0])[0][0]) if np.any(Fk[:, 0]) else 0
    pl_pad = pad_align * int(math.ceil(max(0, L - first0) / pad_align))
    stride = D * kob
    need = stride + L + pl_pad
    if win is None:
        win = 1 << int(math.ceil(math.log2(need)))
    assert win >= need, (win, need)
    nb = -(-Ko // kob)
    Ko_pad = nb * kob
    Kpad = int(pad_align * math.ceil((pl_pad + K + win) / pad_align))
    Fpad = np.zeros((Kpad, Ko_pad), np.float64)
    Fpad[pl_pad:pl_pad + K, :Ko] = Fk
    blocks = np.zeros((nb, win, kob), np.float32)
    for b in range(nb):
        r0 = b * stride
        blocks[b] = Fpad[r0:r0 + win, b * kob:(b + 1) * kob]
        cols = Fpad[:, b * kob:(b + 1) * kob]
        rows = np.nonzero(np.abs(cols).sum(1))[0]
        assert rows.size == 0 or (rows.min() >= r0 and rows.max() < r0 + win), (b, rows.min(), rows.max(), r0, win)
    return dict(blocks=jnp.asarray(blocks.reshape(nb * win, kob)), Kpad=Kpad, pl_pad=pl_pad, Ko_pad=Ko_pad, kob=kob, win=win,
                stride=stride, nb=nb, Ko=Ko, K=K)


def pad_columns_gpu(x, band):
    K = x.shape[-1]
    return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(band['pl_pad'], band['Kpad'] - band['pl_pad'] - K)])


def fused_rotate_dec_k_gpu(S_re, S_im, c0, slope, band, kc, mm_dtype=jnp.float32, precision=None, pb=8, interpret=False):
    """S_re, S_im [P, Kpad] float32 (padded); c0, slope [P]; band from band_blocks_gpu; kc the ramp centre (unpadded).
    mm_dtype: operand type of the banded product (float32 or float16); precision: None (tensor-core default,
    TF32 for float32 operands) or 'highest' (full float32). Returns (y_re, y_im) [P, Ko_pad] float32."""
    P, Kpad = S_re.shape
    assert Kpad == band['Kpad'] and P % pb == 0, (P, pb, Kpad, band['Kpad'])
    nb, win, kob, stride, pl_pad = band['nb'], band['win'], band['kob'], band['stride'], band['pl_pad']
    Ko_pad = band['Ko_pad']
    fblocks = band['blocks'].astype(mm_dtype)
    prec = lax.Precision.HIGHEST if precision == 'highest' else None

    def kernel(sre_ref, sim_ref, c0_ref, sl_ref, f_ref, ore_ref, oim_ref):
        i = pl.program_id(0)
        b = pl.program_id(1)
        r0 = b * stride
        rows = pl.ds(i * pb, pb)
        cols = pl.ds(r0, win)
        sr = sre_ref[rows, cols]                                         # [pb, win]
        si = sim_ref[rows, cols]
        c0b = c0_ref[...]                                                # [pb, 1]
        slb = sl_ref[...]
        k = (r0 - pl_pad - kc).astype(jnp.float32) + lax.broadcasted_iota(jnp.float32, (1, win), 1)
        cyc = c0b + slb * k
        ang = (cyc - jnp.floor(cyc + 0.5)) * TWO_PI
        cs, sn = jnp.cos(ang), jnp.sin(ang)
        zr = (sr * cs - si * sn).astype(mm_dtype)
        zi = (sr * sn + si * cs).astype(mm_dtype)
        F = f_ref[pl.ds(b * win, win), :]                                # [win, kob]
        ore_ref[...] = jnp.dot(zr, F, precision=prec, preferred_element_type=jnp.float32)
        oim_ref[...] = jnp.dot(zi, F, precision=prec, preferred_element_type=jnp.float32)

    grid = (P // pb, nb)
    in_specs = [pl.BlockSpec(memory_space=pl.ANY), pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec((pb, 1), lambda i, b: (i, 0)), pl.BlockSpec((pb, 1), lambda i, b: (i, 0)),
                pl.BlockSpec(memory_space=pl.ANY)]
    out_specs = [pl.BlockSpec((pb, kob), lambda i, b: (i, b))] * 2
    call = pl.pallas_call(kernel, grid=grid, in_specs=in_specs, out_specs=out_specs,
                          out_shape=[jax.ShapeDtypeStruct((P, Ko_pad), jnp.float32)] * 2, interpret=interpret)
    return call(S_re, S_im, c0.reshape(P, 1), slope.reshape(P, 1), fblocks)


def fused_rotate_dec_k2_gpu(S_re, S_im, c0, slope, band, kc, mm_dtype=jnp.float32, precision=None, pb=16, interpret=False):
    """Several children per window load (the GPU counterpart of pallas_ffbp.fused_rotate_dec_k2).
    S_re, S_im [N, P, Kpad]; c0, slope [N, nc, P]; returns (y_re, y_im) [N, nc, P, Ko_pad] float32."""
    N, P, Kpad = S_re.shape
    nc = c0.shape[1]
    assert Kpad == band['Kpad'] and P % pb == 0 and c0.shape == (N, nc, P), (S_re.shape, c0.shape, pb)
    nb, win, kob, stride, pl_pad = band['nb'], band['win'], band['kob'], band['stride'], band['pl_pad']
    Ko_pad = band['Ko_pad']
    fblocks = band['blocks'].astype(mm_dtype)
    prec = lax.Precision.HIGHEST if precision == 'highest' else None

    def kernel(sre_ref, sim_ref, c0_ref, sl_ref, f_ref, ore_ref, oim_ref):
        n = pl.program_id(0)
        i = pl.program_id(1)
        b = pl.program_id(2)
        r0 = b * stride
        rows = pl.ds(i * pb, pb)
        cols = pl.ds(r0, win)
        sr = sre_ref[n, rows, cols]                                      # [pb, win]
        si = sim_ref[n, rows, cols]
        k = (r0 - pl_pad - kc).astype(jnp.float32) + lax.broadcasted_iota(jnp.float32, (1, win), 1)
        F = f_ref[pl.ds(b * win, win), :]                                # [win, kob]
        for c in range(nc):
            c0b = c0_ref[c]                                              # [pb, 1]
            slb = sl_ref[c]
            cyc = c0b + slb * k
            ang = (cyc - jnp.floor(cyc + 0.5)) * TWO_PI
            cs, sn = jnp.cos(ang), jnp.sin(ang)
            zr = (sr * cs - si * sn).astype(mm_dtype)
            zi = (sr * sn + si * cs).astype(mm_dtype)
            ore_ref[c] = jnp.dot(zr, F, precision=prec, preferred_element_type=jnp.float32)
            oim_ref[c] = jnp.dot(zi, F, precision=prec, preferred_element_type=jnp.float32)

    grid = (N, P // pb, nb)
    in_specs = [pl.BlockSpec(memory_space=pl.ANY), pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec((None, nc, pb, 1), lambda n, i, b: (n, 0, i, 0)), pl.BlockSpec((None, nc, pb, 1), lambda n, i, b: (n, 0, i, 0)),
                pl.BlockSpec(memory_space=pl.ANY)]
    out_specs = [pl.BlockSpec((None, nc, pb, kob), lambda n, i, b: (n, 0, i, b))] * 2
    call = pl.pallas_call(kernel, grid=grid, in_specs=in_specs, out_specs=out_specs,
                          out_shape=[jax.ShapeDtypeStruct((N, nc, P, Ko_pad), jnp.float32)] * 2, interpret=interpret)
    return call(S_re, S_im, c0.reshape(N, nc, P, 1), slope.reshape(N, nc, P, 1), fblocks)


def _cat(parts):
    while len(parts) > 1:
        parts = [jnp.concatenate(parts[i:i + 2], axis=1) if i + 1 < len(parts) else parts[i] for i in range(0, len(parts), 2)]
    return parts[0]


def fused_final_gpu(tre, tim, gxT, gyT, qv, mm_dtype=jnp.float32, precision=None, pg=8, interpret=False):
    """GPU counterpart of pallas_ffbp.fused_final: one program per tile, ramps built in registers/shared memory, the
    [T, T] products accumulated on the tensor cores. tre, tim [B, Pf, Qpad]; gxT, gyT [B, T, Pf]; qv [1, Qpad]."""
    B, Pf, Qpad = tre.shape
    T = gxT.shape[1]
    assert Pf % pg == 0 and gxT.shape == (B, T, Pf), (tre.shape, gxT.shape)
    ng = Pf // pg
    prec = lax.Precision.HIGHEST if precision == 'highest' else None

    def nt(a, b):
        return lax.dot_general(a.astype(mm_dtype), b.astype(mm_dtype), (((1,), (1,)), ((), ())), precision=prec,
                               preferred_element_type=jnp.float32)

    def kernel(tre_ref, tim_ref, gx_ref, gy_ref, qv_ref, ore_ref, oim_ref):
        q = qv_ref[...]
        rr = ri = ir = ii = jnp.zeros((T, T), jnp.float32)
        for g in range(ng):
            Ar, Ai, Br, Bi = [], [], [], []
            for p in range(g * pg, (g + 1) * pg):
                gx = gx_ref[:, p:p + 1]
                gy = gy_ref[:, p:p + 1]
                ax = -(gx * q)
                ax = (ax - jnp.floor(ax + 0.5)) * TWO_PI
                ay = -(gy * q)
                ay = (ay - jnp.floor(ay + 0.5)) * TWO_PI
                cx, sx = jnp.cos(ax), jnp.sin(ax)
                cy, sy = jnp.cos(ay), jnp.sin(ay)
                dr = tre_ref[p:p + 1, :]
                di = tim_ref[p:p + 1, :]
                Ar.append(dr * cx - di * sx)
                Ai.append(dr * sx + di * cx)
                Br.append(cy)
                Bi.append(sy)
            Ar, Ai, Br, Bi = (_cat(v) for v in (Ar, Ai, Br, Bi))          # Triton concatenates two at a time
            rr = rr + nt(Ar, Br)
            ii = ii + nt(Ai, Bi)
            ri = ri + nt(Ar, Bi)
            ir = ir + nt(Ai, Br)
        ore_ref[...] = rr - ii
        oim_ref[...] = ri + ir

    in_specs = [pl.BlockSpec((None, Pf, Qpad), lambda b: (b, 0, 0)), pl.BlockSpec((None, Pf, Qpad), lambda b: (b, 0, 0)),
                pl.BlockSpec((None, T, Pf), lambda b: (b, 0, 0)), pl.BlockSpec((None, T, Pf), lambda b: (b, 0, 0)),
                pl.BlockSpec((1, Qpad), lambda b: (0, 0))]
    out_specs = [pl.BlockSpec((None, T, T), lambda b: (b, 0, 0))] * 2
    call = pl.pallas_call(kernel, grid=(B,), in_specs=in_specs, out_specs=out_specs,
                          out_shape=[jax.ShapeDtypeStruct((B, T, T), jnp.float32)] * 2, interpret=interpret)
    return call(tre, tim, gxT, gyT, qv)
