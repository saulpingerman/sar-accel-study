"""Achievable unit rates of a TPU core for the operations the factorized kernels use, measured in the same framework
(JAX/XLA and Pallas), for the roofline bounds of the kernels appendix.

    python ubench_tpu.py --out ubench.json

Measures: HBM read+write bandwidth; the vector unit on the kernel's own rotation mix (table product and complex
multiply) with data resident in vector memory; the matrix units at the kernel's product shapes; sine and cosine
throughput in XLA and inside a kernel; and a float32 copy inside a kernel (vector load/store rate).
"""
import argparse, json, os, time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def timeit(fn, *args, reps=5):
    out = fn(*args); jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); out = fn(*args); jax.block_until_ready(out); ts.append(time.perf_counter() - t)
    return min(ts)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='ubench.json'); ap.add_argument('--smoke', action='store_true', help='tiny sizes, interpret mode (CPU check)'); a = ap.parse_args()
    INTERP = a.smoke
    res = dict(device=str(jax.devices()[0]))
    k = jax.random.PRNGKey(0)
    # 1. HBM: read and write 2 GB of float32
    n = (512 * 1024 * 1024) if not a.smoke else (1 << 20)
    x = jax.random.normal(k, (n,), jnp.float32)
    f = jax.jit(lambda v: v * 1.0001 + 1.0)
    t = timeit(f, x)
    res['hbm_copy_GBps'] = 2 * 4 * n / t / 1e9
    print(f"HBM read+write: {res['hbm_copy_GBps']:.0f} GB/s", flush=True)
    # 2. XLA elementwise complex rotation (memory-bound in XLA: 4 inputs, 2 outputs)
    m = (64 * 1024 * 1024) if not a.smoke else (1 << 20)
    ar, ai, cr, ci = (jax.random.normal(jax.random.fold_in(k, i), (m,), jnp.float32) for i in range(4))
    g = jax.jit(lambda ar, ai, cr, ci: (ar * cr - ai * ci, ar * ci + ai * cr))
    t = timeit(g, ar, ai, cr, ci)
    res['xla_rotate_Gelem_s'] = m / t / 1e9
    print(f"XLA complex rotate: {res['xla_rotate_Gelem_s']:.1f} G elements/s ({6 * 4 * m / t / 1e9:.0f} GB/s)", flush=True)
    del x, ar, ai, cr, ci
    # 3. Pallas: the level kernel's rotation mix on resident data. One block [pb, win] of data and tables in VMEM; the
    #    grid repeats the same block index so no DMA happens after the first step; each step rotates the block through
    #    nch chunks exactly as the level kernel does (coarse lane broadcast, table product, complex multiply, store).
    pb, chunk, nch, reps_in = 256, 128, 14, (64 if not a.smoke else 2)
    win = chunk * nch
    sr = jax.random.normal(k, (pb, win), jnp.float32); si = jax.random.normal(jax.random.fold_in(k, 9), (pb, win), jnp.float32)
    fc = jnp.cos(jax.random.uniform(k, (pb, chunk))); fs = jnp.sin(jax.random.uniform(k, (pb, chunk)))
    cc = jnp.cos(jax.random.uniform(k, (pb, 128))); cs = jnp.sin(jax.random.uniform(k, (pb, 128)))

    def rot_kernel(sr_ref, si_ref, fc_ref, fs_ref, cc_ref, cs_ref, or_ref, oi_ref):
        fcos, fsin = fc_ref[...], fs_ref[...]
        ccos_all, csin_all = cc_ref[...], cs_ref[...]
        for m_ in range(nch):
            ccos, csin = ccos_all[:, m_:m_ + 1], csin_all[:, m_:m_ + 1]
            c = ccos * fcos - csin * fsin
            s_ = csin * fcos + ccos * fsin
            lo = m_ * chunk
            xr, xi = sr_ref[:, lo:lo + chunk], si_ref[:, lo:lo + chunk]
            or_ref[:, lo:lo + chunk] = xr * c - xi * s_
            oi_ref[:, lo:lo + chunk] = xr * s_ + xi * c

    same = lambda i: (0, 0)
    call = pl.pallas_call(rot_kernel, grid=(reps_in,),
                          in_specs=[pl.BlockSpec((pb, win), same)] * 2 + [pl.BlockSpec((pb, chunk), same)] * 2 + [pl.BlockSpec((pb, 128), same)] * 2,
                          out_specs=[pl.BlockSpec((pb, win), same)] * 2,
                          out_shape=[jax.ShapeDtypeStruct((pb, win), jnp.float32)] * 2,
                          compiler_params=None if INTERP else pltpu.CompilerParams(vmem_limit_bytes=64 << 20), interpret=INTERP)
    fk = jax.jit(call)
    t = timeit(fk, sr, si, fc, fs, cc, cs)
    res['pallas_rotate_Gelem_s'] = reps_in * pb * win / t / 1e9
    print(f"Pallas rotation mix on resident data: {res['pallas_rotate_Gelem_s']:.0f} G elements/s", flush=True)

    # 3b. the same with the three-pass bfloat16 split of the result (as the three-pass level kernel does)
    def rot_split_kernel(sr_ref, si_ref, fc_ref, fs_ref, cc_ref, cs_ref, hr_ref, lr_ref, hi_ref, li_ref):
        fcos, fsin = fc_ref[...], fs_ref[...]
        ccos_all, csin_all = cc_ref[...], cs_ref[...]
        for m_ in range(nch):
            ccos, csin = ccos_all[:, m_:m_ + 1], csin_all[:, m_:m_ + 1]
            c = ccos * fcos - csin * fsin
            s_ = csin * fcos + ccos * fsin
            lo = m_ * chunk
            xr, xi = sr_ref[:, lo:lo + chunk], si_ref[:, lo:lo + chunk]
            zr = xr * c - xi * s_
            zi = xr * s_ + xi * c
            h = zr.astype(jnp.bfloat16); hr_ref[:, lo:lo + chunk] = h; lr_ref[:, lo:lo + chunk] = (zr - h.astype(jnp.float32)).astype(jnp.bfloat16)
            h = zi.astype(jnp.bfloat16); hi_ref[:, lo:lo + chunk] = h; li_ref[:, lo:lo + chunk] = (zi - h.astype(jnp.float32)).astype(jnp.bfloat16)

    call = pl.pallas_call(rot_split_kernel, grid=(reps_in,),
                          in_specs=[pl.BlockSpec((pb, win), same)] * 2 + [pl.BlockSpec((pb, chunk), same)] * 2 + [pl.BlockSpec((pb, 128), same)] * 2,
                          out_specs=[pl.BlockSpec((pb, win), same)] * 4,
                          out_shape=[jax.ShapeDtypeStruct((pb, win), jnp.bfloat16)] * 4,
                          compiler_params=None if INTERP else pltpu.CompilerParams(vmem_limit_bytes=64 << 20), interpret=INTERP)
    t = timeit(jax.jit(call), sr, si, fc, fs, cc, cs)
    res['pallas_rotate_split_Gelem_s'] = reps_in * pb * win / t / 1e9
    print(f"Pallas rotation mix with bf16 hi/lo split: {res['pallas_rotate_split_Gelem_s']:.0f} G elements/s", flush=True)

    # 3c. plain copy inside a kernel (vector load/store rate), same block
    def copy_kernel(sr_ref, si_ref, or_ref, oi_ref):
        for m_ in range(nch):
            lo = m_ * chunk
            or_ref[:, lo:lo + chunk] = sr_ref[:, lo:lo + chunk]
            oi_ref[:, lo:lo + chunk] = si_ref[:, lo:lo + chunk]
    call = pl.pallas_call(copy_kernel, grid=(reps_in,), in_specs=[pl.BlockSpec((pb, win), same)] * 2, out_specs=[pl.BlockSpec((pb, win), same)] * 2,
                          out_shape=[jax.ShapeDtypeStruct((pb, win), jnp.float32)] * 2, compiler_params=None if INTERP else pltpu.CompilerParams(vmem_limit_bytes=64 << 20), interpret=INTERP)
    t = timeit(jax.jit(call), sr, si)
    res['pallas_copy_Gelem_s'] = reps_in * pb * win / t / 1e9
    print(f"Pallas copy on resident data: {res['pallas_copy_Gelem_s']:.0f} G elements/s", flush=True)

    # 4. sines and cosines: XLA on 64 M elements, and inside a kernel on a resident block
    ang = jax.random.uniform(k, (m,), jnp.float32, -3.0, 3.0)
    h = jax.jit(lambda v: (jnp.cos(v), jnp.sin(v)))
    t = timeit(h, ang)
    res['xla_sincos_Gpairs_s'] = m / t / 1e9
    print(f"XLA cos+sin: {res['xla_sincos_Gpairs_s']:.1f} G pairs/s", flush=True)

    def trig_kernel(a_ref, c_ref, s_ref):
        for m_ in range(nch):
            lo = m_ * chunk
            v = a_ref[:, lo:lo + chunk]
            c_ref[:, lo:lo + chunk] = jnp.cos(v)
            s_ref[:, lo:lo + chunk] = jnp.sin(v)
    angb = jax.random.uniform(k, (pb, win), jnp.float32, -3.0, 3.0)
    call = pl.pallas_call(trig_kernel, grid=(reps_in,), in_specs=[pl.BlockSpec((pb, win), same)], out_specs=[pl.BlockSpec((pb, win), same)] * 2,
                          out_shape=[jax.ShapeDtypeStruct((pb, win), jnp.float32)] * 2, compiler_params=None if INTERP else pltpu.CompilerParams(vmem_limit_bytes=64 << 20), interpret=INTERP)
    t = timeit(jax.jit(call), angb)
    res['pallas_sincos_Gpairs_s'] = reps_in * pb * win / t / 1e9
    print(f"Pallas cos+sin on resident data: {res['pallas_sincos_Gpairs_s']:.1f} G pairs/s", flush=True)

    # 5. matrix units: bf16 products at the kernels' shapes, operands resident in HBM (XLA), float32 accumulation
    def mxu(Mr, K, Nc, batch=1):
        A = jax.random.normal(k, (batch, Mr, K), jnp.float32).astype(jnp.bfloat16)
        Bm = jax.random.normal(jax.random.fold_in(k, 3), (batch, K, Nc), jnp.float32).astype(jnp.bfloat16)
        fm = jax.jit(lambda A, B: jnp.einsum('bmk,bkn->bmn', A, B, preferred_element_type=jnp.float32))
        t = timeit(fm, A, Bm)
        return 2.0 * batch * Mr * K * Nc / t / 1e12
    sc = 1 if not a.smoke else 8
    res['mxu_256x1792x256_x64_TFLOPs'] = mxu(256, 1792, 256, 64 // sc)        # level kernel's product, 64 steps
    res['mxu_4096x1792x256_TFLOPs'] = mxu(4096 // sc, 1792, 256, 1)           # the same with a tall M
    res['mxu_64x10112x64_x420_TFLOPs'] = mxu(64, 10112, 64, 420 // sc)        # final stage, one tile per product
    res['mxu_256x10112x256_x105_TFLOPs'] = mxu(256, 10112, 256, 105 // sc)    # final stage, four tiles stacked
    res['mxu_4096x4096x4096_TFLOPs'] = mxu(4096 // sc, 4096 // sc, 4096 // sc, 1)   # a large square product
    for kk, v in res.items():
        if kk.startswith('mxu'):
            print(f"{kk}: {v:.0f} TFLOP/s", flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
