#!/usr/bin/env python3
"""Sustained end-to-end throughput.

Forms `--count` consecutive images, cycling four phase histories held in host
memory. Each iteration is timed as a whole and includes what a production
loop would do per collection: copy the phase history to the device, compute
the per-collection geometry on the host and copy it over, form the image, and
copy the complex image back to the host. A second loop times the device
computation alone for comparison.

  python e2e.py --label tpu-v5e --N 4096 --count 200 --policies fp32_fast,fp32 --out results/e2e_tpu-v5e.jsonl
"""
import argparse
import json
import time

import numpy as np


def summarize(ts, skip):
    t = np.asarray(ts[skip:])
    return dict(mean_s=float(t.mean()), p05_s=float(np.percentile(t, 5)), p50_s=float(np.percentile(t, 50)),
                p95_s=float(np.percentile(t, 95)), first_s=float(ts[0]), n=int(t.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--N', type=int, default=4096)
    ap.add_argument('--count', type=int, default=200)
    ap.add_argument('--skip', type=int, default=20)
    ap.add_argument('--policies', default='fp32_fast')
    ap.add_argument('--T', type=int, default=32)
    ap.add_argument('--levels', type=int, default=3)
    ap.add_argument('--block', type=int, default=0, help='apply the decimation filters as banded products with this block size (0 = dense)')
    ap.add_argument('--cuda', default='', help='also run the CUDA exact kernel with these profile types, e.g. fp32,f16')
    ap.add_argument('--instance', default='')
    ap.add_argument('--transfer-probe', action='store_true')
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from sarbench import sim, bp, ffbp

    N = a.N
    res = 0.3
    scene = N * res / (1.25 * np.sqrt(2.0))
    col = sim.make_collect(res=res, scene=scene)
    spacing = scene / N
    rng = np.random.default_rng(0)
    hist = [(rng.standard_normal((col.Np, col.K), np.float32) + 1j * rng.standard_normal((col.Np, col.K), np.float32))
            for _ in range(4)]
    base = dict(label=a.label, instance=a.instance, device=str(jax.devices()[0]), N=N, count=a.count)

    def emit(rec):
        rec = {**base, **rec}
        print(json.dumps(rec), flush=True)
        with open(a.out, 'a') as f:
            f.write(json.dumps(rec) + '\n')

    levels = ffbp.default_levels(N, col.K, a.T, a.levels)
    plan = ffbp.make_plan(col, N, spacing, levels, a.T, block=a.block)
    for pol in [p for p in a.policies.split(',') if p]:
        try:
            p = ffbp.POLICIES[pol]
            fn = ffbp.make_ffbp(pol, plan)
            static = ffbp.device_arrays(pol, plan, ffbp.collection_arrays(plan, col.ant))

            @jax.jit
            def prep(S):
                s = jnp.max(jnp.abs(S))
                return (S.real / s).astype(p['ew'])[None], (S.imag / s).astype(p['ew'])[None], s

            @jax.jit
            def finish(re, im, s):
                # two float32 planes; a complex64 array leaves the TPU far more slowly than the same bytes as floats
                return jnp.stack([re * s, im * s], axis=-1)

            def per_collection():
                """Host geometry for this collection, copied to the device; filters stay on the device."""
                coll = ffbp.collection_arrays(plan, col.ant)
                h = np.float32
                arrs = dict(levels=[dict(e) for e in static['levels']], final=dict(static['final']))
                arrs['levels'][0]['cyc0'] = jnp.asarray(coll['levels'][0]['cyc0'].astype(h))
                arrs['levels'][0]['slope'] = jnp.asarray(coll['levels'][0]['slope'].astype(h))
                for i in range(1, len(arrs['levels'])):
                    arrs['levels'][i]['u'] = jnp.asarray(coll['levels'][i]['u'].astype(h))
                    arrs['levels'][i]['r0'] = jnp.asarray(coll['levels'][i]['r0'].astype(h))
                arrs['final']['u'] = jnp.asarray(coll['final']['u'].astype(h))
                arrs['final']['r0'] = jnp.asarray(coll['final']['r0'].astype(h))
                return arrs

            def one(i):
                arrs = per_collection()
                hre, him, s = prep(jnp.asarray(hist[i % 4]))
                re, im = fn(hre, him, arrs)
                return np.ascontiguousarray(np.asarray(finish(re, im, s))).view(np.complex64)[..., 0]

            img = one(0)                      # compile
            assert img.dtype == np.complex64 and img.shape == (N, N)
            ts = []
            for i in range(a.count):
                t = time.perf_counter()
                img = one(i)
                ts.append(time.perf_counter() - t)
            hre, him, s = prep(jnp.asarray(hist[0]))
            arrs = per_collection()
            jax.block_until_ready(fn(hre, him, arrs))
            cs = []
            for i in range(max(a.count // 4, 10)):
                t = time.perf_counter()
                jax.block_until_ready(fn(hre, him, arrs))
                cs.append(time.perf_counter() - t)
            # transfers alone
            t = time.perf_counter(); x = jnp.asarray(hist[1]); jax.block_until_ready(x); up = time.perf_counter() - t
            y = finish(*fn(hre, him, arrs), s); jax.block_until_ready(y)
            t = time.perf_counter(); np.asarray(y); down = time.perf_counter() - t
            emit(dict(algo='ffbp', policy=pol, levels=str(levels), T=a.T, block=a.block, end_to_end=summarize(ts, a.skip),
                      compute_only=summarize(cs, 2), upload_s=up, download_s=down,
                      per_image_s=[round(v, 4) for v in ts]))
        except Exception as e:
            emit(dict(algo='ffbp', policy=pol, error=f'{type(e).__name__}: {str(e)[:300]}'))

    if a.transfer_probe:
        # device-to-host copy of an N x N complex image in several forms
        x = jnp.asarray(hist[0][:N, :N] if hist[0].shape[0] >= N else np.resize(hist[0], (N, N)))
        forms = {
            'complex64': lambda v: v,
            'float32 x2': lambda v: jnp.stack([v.real, v.imag]),
            'float16 x2': lambda v: jnp.stack([v.real, v.imag]).astype(jnp.float16),
            'bfloat16 x2': lambda v: jnp.stack([v.real, v.imag]).astype(jnp.bfloat16),
            'float32 magnitude': lambda v: jnp.abs(v),
            'uint8 magnitude': lambda v: jnp.clip(jnp.abs(v) * 40.0, 0, 255).astype(jnp.uint8),
        }
        for name, fn_ in forms.items():
            try:
                y = jax.jit(fn_)(x)
                jax.block_until_ready(y)
                np.asarray(y)
                ts = []
                for _ in range(5):
                    y = jax.jit(fn_)(x + 1)
                    jax.block_until_ready(y)
                    t = time.perf_counter(); h = np.asarray(y); ts.append(time.perf_counter() - t)
                emit(dict(algo='transfer', policy=name, bytes=int(h.nbytes), down_s=min(ts), mb_per_s=h.nbytes / 1e6 / min(ts)))
            except Exception as e:
                emit(dict(algo='transfer', policy=name, error=f'{type(e).__name__}: {str(e)[:200]}'))
        for name, v in (('upload complex64', hist[0][:N, :N].astype(np.complex64)),
                        ('upload float32 planes', np.ascontiguousarray(hist[0][:N, :N].astype(np.complex64)).view(np.float32))):
            ts = []
            for _ in range(5):
                t = time.perf_counter(); y = jnp.asarray(v); jax.block_until_ready(y); ts.append(time.perf_counter() - t)
            emit(dict(algo='transfer', policy=name, bytes=int(v.nbytes), up_s=min(ts), mb_per_s=v.nbytes / 1e6 / min(ts)))
        for nb in (1 << 20, 1 << 24, 1 << 27):
            v = np.zeros(nb // 4, np.float32)
            ts = []
            for _ in range(5):
                t = time.perf_counter(); y = jnp.asarray(v + 1); jax.block_until_ready(y + 0); ts.append(time.perf_counter() - t)
            emit(dict(algo='transfer', policy=f'upload {nb >> 20} MB float32', up_s=min(ts), mb_per_s=nb / 1e6 / min(ts)))

    if a.cuda:
        import cupy as cp
        from sarbench import gpu_ref
        nfft = 1 << int(np.ceil(np.log2(8 * col.K)))
        dr = bp.range_bin(col.df, nfft)
        px, py, pz = (cp.asarray(v.astype(np.float32)) for v in sim.ground_grid(N, spacing))
        for st in a.cuda.split(','):
            try:
                def one(i):
                    u, r0 = bp.host_geometry(col.ant)
                    rc = gpu_ref.range_compress(cp.asarray(hist[i % 4]), nfft)
                    re, im = gpu_ref.bp(gpu_ref.pack(rc, st), cp.asarray(u.astype(np.float32)), cp.asarray(r0.astype(np.float32)),
                                        px, py, pz, nfft, dr, col.fref, st)
                    out = cp.asnumpy(re) + 1j * cp.asnumpy(im)
                    del rc
                    return out
                one(0)
                ts = []
                for i in range(a.count):
                    t = time.perf_counter()
                    one(i)
                    ts.append(time.perf_counter() - t)
                emit(dict(algo='bp_cuda', policy=st, end_to_end=summarize(ts, min(a.skip, a.count // 4)),
                          per_image_s=[round(v, 4) for v in ts]))
            except Exception as e:
                emit(dict(algo='bp_cuda', policy=st, error=f'{type(e).__name__}: {str(e)[:300]}'))


if __name__ == '__main__':
    main()
