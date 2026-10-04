#!/usr/bin/env python3
"""Follow-up primitive probe: matmul precision modes, windowed lookups, and the
batched product shapes used by the factorized algorithm."""
import sys, time, json
import numpy as np
import jax, jax.numpy as jnp
rng = np.random.default_rng(0)

def bench(name, fn, args, work, reps=3):
    try:
        f = jax.jit(fn)
        t = time.perf_counter(); jax.block_until_ready(f(*args)); first = time.perf_counter() - t
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); jax.block_until_ready(f(*args)); ts.append(time.perf_counter() - t)
        print(f'{name:44s} {min(ts)*1e3:10.2f} ms  {work/min(ts):.3e} /s  (first {first:.1f}s)', flush=True)
    except Exception as e:
        print(f'{name:44s} ERROR {type(e).__name__}: {str(e)[:200]}', flush=True)

a = rng.standard_normal((2048, 2048)).astype(np.float32)
ref = a.astype(np.float64) @ a.astype(np.float64)
big = jnp.asarray(rng.standard_normal((8192, 8192), np.float32))
for prec in [None, 'high', 'highest']:
    got = np.asarray(jax.jit(lambda m: jnp.matmul(m, m, precision=prec))(jnp.asarray(a)))
    print(f'matmul float32 precision={prec}: error {10*np.log10(np.sum((got-ref)**2)/np.sum(ref**2)):.1f} dB', flush=True)
    bench(f'matmul 8192^2 float32 precision={prec}', lambda m: jnp.matmul(m, m, precision=prec), (big,), 2.0 * 8192 ** 3)

# contiguous-window lookups: one index fetches a run of elements
tab = jnp.asarray(rng.standard_normal((4096, 8192), np.float32))
for w in [16, 128]:
    starts = jnp.asarray(rng.integers(0, 8192 - w, (4096, 4096 // w)).astype(np.int32))
    def win(t, s, w=w):
        return jax.vmap(lambda row, ss: jax.vmap(lambda i: jax.lax.dynamic_slice(row, (i,), (w,)))(ss))(t, s)
    bench(f'windowed lookup, run length {w}', win, (tab, starts), 4096 * 4096)

# factorized-algorithm shapes
for dt in ['float32', 'bfloat16']:
    h = jnp.asarray(rng.standard_normal((8, 4096, 4096), np.float32)).astype(dt)
    f = jnp.asarray(rng.standard_normal((4096, 1024), np.float32)).astype(dt)
    bench(f'[8,4096,4096]@[4096,1024] {dt}', lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.float32), (h, f), 2.0 * 8 * 4096 * 4096 * 1024)
    h2 = jnp.asarray(rng.standard_normal((512, 1024, 1024), np.float32)).astype(dt)
    f2 = jnp.asarray(rng.standard_normal((1024, 128), np.float32)).astype(dt)
    bench(f'[512,1024,1024]@[1024,128] {dt}', lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.float32), (h2, f2), 2.0 * 512 * 1024 * 1024 * 128)
    bench(f'einsum cpk,pq->cqk [512,1024,128]x[1024,128] {dt}', lambda x, y: jnp.einsum('cpk,pq->cqk', x, y, preferred_element_type=jnp.float32),
          (h2[:, :, :128], f2), 2.0 * 512 * 1024 * 128 * 128)
    A = jnp.asarray(rng.standard_normal((256, 64, 16384), np.float32)).astype(dt)
    bench(f'tile product [256,64,16384]x[256,16384,64] {dt}', lambda x: jnp.matmul(x, jnp.swapaxes(x, 1, 2), preferred_element_type=jnp.float32),
          (A,), 2.0 * 256 * 64 * 16384 * 64)

# element-wise complex rotation of a block, the recentring step
for dt in ['float32', 'bfloat16']:
    hr = jnp.asarray(rng.standard_normal((8, 4096, 4096), np.float32)).astype(dt)
    d = jnp.asarray(rng.standard_normal((8, 4096), np.float32))
    def rot(hr_, d_, dt=dt):
        k = jnp.arange(4096, dtype=jnp.float32)
        cyc = d_[:, :, None] * 0.37 + k[None, None, :] * (d_[:, :, None] * 1e-3)
        cyc = cyc - jnp.round(cyc)
        c, s = jnp.cos(6.2831853 * cyc).astype(dt), jnp.sin(6.2831853 * cyc).astype(dt)
        return hr_ * c - hr_ * s, hr_ * s + hr_ * c
    bench(f'recentre rotate [8,4096,4096] {dt}', rot, (hr, d), 8.0 * 4096 * 4096)
