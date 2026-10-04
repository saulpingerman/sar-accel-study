#!/usr/bin/env python3
"""Primitive throughput on the current JAX device: element-wise math, FFT,
lookups, and matrix products at several shapes and float types. The numbers
decide which algorithm formulation suits the hardware."""
import json
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

out = sys.argv[1] if len(sys.argv) > 1 else None
label = sys.argv[2] if len(sys.argv) > 2 else jax.default_backend()
rng = np.random.default_rng(0)


def bench(name, fn, args, work, reps=3):
    try:
        f = jax.jit(fn)
        t = time.perf_counter(); jax.block_until_ready(f(*args)); first = time.perf_counter() - t
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); jax.block_until_ready(f(*args)); ts.append(time.perf_counter() - t)
        rec = dict(label=label, name=name, first_s=first, run_s=min(ts), work=work, rate=work / min(ts))
        print(f'{name:34s} {min(ts) * 1e3:10.2f} ms  {work / min(ts):.3e} /s  (first {first:.1f}s)', flush=True)
    except Exception as e:
        rec = dict(label=label, name=name, error=f'{type(e).__name__}: {str(e)[:300]}')
        print(f'{name:34s} ERROR {rec["error"][:200]}', flush=True)
    if out:
        with open(out, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')


n = 1 << 26
x32 = jnp.asarray(rng.standard_normal(n, np.float32))
for dt in ['float32', 'bfloat16', 'float16']:
    x = x32.astype(dt)
    bench(f'elementwise mul-add {dt}', lambda v: v * v + v, (x,), n)
    bench(f'elementwise sin+cos {dt}', lambda v: jnp.sin(v) + jnp.cos(v), (x,), n)
    bench(f'elementwise sqrt {dt}', lambda v: jnp.sqrt(jnp.abs(v)), (x,), n)
    bench(f'reduce sum {dt}', lambda v: v.reshape(64, -1).sum(0), (x,), n)

z = jnp.asarray((rng.standard_normal((4096, 8192), np.float32) + 1j * rng.standard_normal((4096, 8192), np.float32)))
bench('fft 1d batched 4096x8192 c64', lambda v: jnp.fft.fft(v, axis=1), (z,), z.size)
bench('fft 1d batched axis0 4096x8192', lambda v: jnp.fft.fft(v, axis=0), (z,), z.size)
z2 = z[:, :4096]
bench('fft 2d 4096x4096 c64', lambda v: jnp.fft.fft2(v), (z2,), z2.size)
bench('complex mul c64', lambda v: v * v, (z,), z.size)

tab = jnp.asarray(rng.standard_normal((64, 8192), np.float32))
idx = jnp.asarray(rng.integers(0, 8192, (64, 1 << 18)).astype(np.int32))
bench('gather take_along_axis', lambda t, i: jnp.take_along_axis(t, i, axis=1), (tab, idx), idx.size)
idx_s = jnp.sort(idx, axis=1)
bench('gather sorted indices', lambda t, i: jnp.take_along_axis(t, i, axis=1), (tab, idx_s), idx.size)
flat = jnp.asarray(rng.standard_normal(8192, np.float32))
idx1 = idx[0]
bench('gather 1d take', lambda t, i: t[i], (flat, idx1), idx1.size)
bench('gather via one-hot matmul bf16', lambda t, i: jnp.matmul(jax.nn.one_hot(i, 8192, dtype=jnp.bfloat16), t.astype(jnp.bfloat16),
                                                                 preferred_element_type=jnp.float32), (flat, idx1[:1 << 16]), 1 << 16)
# hat-function interpolation weights, no lookup: [M, L] weights then a product
tq = jnp.asarray(rng.uniform(0, 255, 4096).astype(np.float32))
row = jnp.asarray(rng.standard_normal((256, 256), np.float32))
bench('hat-weight interp 256 tiles', lambda t, r: jnp.einsum('ml,bl->bm', jnp.maximum(0, 1 - jnp.abs(t[:, None] - jnp.arange(256, dtype=jnp.float32)[None, :])), r),
      (tq, row), 256 * 4096)

for dt in ['float32', 'bfloat16', 'float16']:
    a = jnp.asarray(rng.standard_normal((8192, 8192), np.float32)).astype(dt)
    bench(f'matmul 8192^2 {dt}', lambda m: jnp.matmul(m, m, preferred_element_type=jnp.float32), (a,), 2.0 * 8192 ** 3)
    b = jnp.asarray(rng.standard_normal((1024, 64, 4096), np.float32)).astype(dt)
    bench(f'batched matmul 1024x(64x4096x64) {dt}', lambda m: jnp.matmul(m, jnp.swapaxes(m, 1, 2), preferred_element_type=jnp.float32),
          (b,), 2.0 * 1024 * 64 * 4096 * 64)
    c = jnp.asarray(rng.standard_normal((4096, 4096), np.float32)).astype(dt)
    d = jnp.asarray(rng.standard_normal((4096, 128), np.float32)).astype(dt)
    bench(f'matmul 4096x4096 @ 4096x128 {dt}', lambda m, v: jnp.matmul(m, v, preferred_element_type=jnp.float32), (c, d), 2.0 * 4096 * 4096 * 128)

# does a float32 matmul keep float32 accuracy on this device?
a = rng.standard_normal((2048, 2048)).astype(np.float32)
ref = a.astype(np.float64) @ a.astype(np.float64)
for dt in ['float32', 'bfloat16']:
    got = np.asarray(jax.jit(lambda m: jnp.matmul(m, m, preferred_element_type=jnp.float32))(jnp.asarray(a).astype(dt)))
    err = 10 * np.log10(np.sum((got - ref) ** 2) / np.sum(ref ** 2))
    print(f'matmul accuracy {dt}: error {err:.1f} dB', flush=True)
    if out:
        with open(out, 'a') as fh:
            fh.write(json.dumps(dict(label=label, name=f'matmul accuracy {dt}', err_db=float(err))) + '\n')
