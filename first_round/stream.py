#!/usr/bin/env python3
"""Sustained throughput for a stream of independent images.

A queue of phase histories is processed continuously. Stages overlap: while
one batch is being copied back to the host by a worker thread, the next is
already being formed and the one after that is being copied in. Reported is
images per second over the whole run, for several batch sizes and for two
output products (the full complex image, or an 8-bit detected image made on
the device).

  python stream.py --label tpu-v5e --sizes 2048,4096 --batches 1,2,4 --out results/v2/stream_tpu-v5e.jsonl
"""
import argparse
import json
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--instance', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sizes', default='2048,4096')
    ap.add_argument('--batches', default='1,2,4')
    ap.add_argument('--policies', default='fp32_fast')
    ap.add_argument('--products', default='complex64,uint8')
    ap.add_argument('--images', type=int, default=64, help='images per configuration, at least')
    ap.add_argument('--min-seconds', type=float, default=20.0)
    ap.add_argument('--depth', type=int, default=3, help='batches in flight')
    ap.add_argument('--geom', default='space', choices=['space', 'air'])
    ap.add_argument('--T', type=int, default=32)
    ap.add_argument('--levels', type=int, default=3)
    ap.add_argument('--groups', type=int, default=0, help='bounded-memory mode of the factorized algorithm (one image per call only)')
    ap.add_argument('--nhist', type=int, default=4, help='distinct phase histories held in host memory')
    ap.add_argument('--block', type=int, default=0, help='apply the decimation filters as banded products with this block size (0 = dense)')
    ap.add_argument('--cuda', default='', help='also stream the CUDA exact kernel with these profile types')
    ap.add_argument('--pfa', default='', help="also stream polar format: 'czt' (chirp-z) and/or 'sinc' (windowed-sinc lookups)")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from sarbench import sim, bp, ffbp

    base = dict(label=a.label, instance=a.instance, device=str(jax.devices()[0]), geom=a.geom, depth=a.depth)

    def emit(rec):
        rec = {**base, **rec}
        print(json.dumps(rec), flush=True)
        with open(a.out, 'a') as f:
            f.write(json.dumps(rec) + '\n')

    def drive(submit, B, nbatch_min):
        """Run the pipelined loop; `submit(i)` returns a zero-argument function that fetches batch i to the host."""
        pool = ThreadPoolExecutor(2)
        inflight = deque()
        for i in range(2):                                 # warm the pipeline
            pool.submit(submit(i)).result()
        t0 = time.perf_counter()
        n, lat = 0, []
        while n < nbatch_min or time.perf_counter() - t0 < a.min_seconds:
            ts = time.perf_counter()
            inflight.append((ts, pool.submit(submit(n))))
            n += 1
            if len(inflight) >= a.depth:
                t_in, fut = inflight.popleft()
                fut.result()
                lat.append(time.perf_counter() - t_in)
        while inflight:
            t_in, fut = inflight.popleft()
            fut.result()
            lat.append(time.perf_counter() - t_in)
        el = time.perf_counter() - t0
        pool.shutdown()
        return dict(images=n * B, seconds=el, images_per_s=n * B / el, s_per_image=el / (n * B),
                    latency_p50_s=float(np.percentile(lat, 50)), latency_p95_s=float(np.percentile(lat, 95)))

    rng = np.random.default_rng(0)
    for N in [int(v) for v in a.sizes.split(',')]:
        res = 0.3
        scene = N * res / (1.25 * np.sqrt(2.0))
        col = sim.make_collect(res=res, scene=scene, r0=600e3 if a.geom == 'space' else 10e3)
        spacing = scene / N
        hist = [(rng.standard_normal((col.Np, col.K), np.float32) + 1j * rng.standard_normal((col.Np, col.K), np.float32))
                for _ in range(a.nhist)]
        levels = ffbp.default_levels(N, col.K, a.T, a.levels)
        plan = ffbp.make_plan(col, N, spacing, levels, a.T, block=a.block)
        for kind in [k for k in a.pfa.split(',') if k]:
            from sarbench import pfa
            geo0 = pfa.pfa_geometry(col)
            nf = 1 << int(np.ceil(np.log2(1.5 * N)))
            lo = (nf - N) // 2
            f32 = np.float32
            from scipy.signal.windows import taylor
            win = jnp.asarray((taylor(geo0['nkx'], nbar=4, sll=35, norm=False)[:, None] *
                               taylor(geo0['nky'], nbar=4, sll=35, norm=False)[None, :]).astype(f32))
            if kind == 'czt':
                core = pfa.make_pfa_czt('fp32', col.K, col.Np, geo0['nkx'], geo0['nky'], nf, nf)
            else:
                core = pfa.make_pfa('fp32', geo0['nkx'], geo0['nky'], nf, nf)
            for product in a.products.split(','):
                try:
                    @jax.jit
                    def run(planes, c1, c2, c3, c4, w):
                        # the phase history arrives as float planes; the image leaves as float planes
                        # or as an 8-bit detected image, cropped to the N x N scene
                        if kind == 'czt':
                            img = core(jax.lax.complex(planes[..., 0], planes[..., 1]), c1, c2, c3, c4, w)
                        else:
                            img = core(planes[..., 0], planes[..., 1], c1, c2, c3, c4, w)
                        img = img[lo:lo + N, lo:lo + N]
                        if product == 'uint8':
                            pw = img.real * img.real + img.imag * img.imag
                            db = 10.0 * jnp.log10(pw / jnp.max(pw) + 1e-12)
                            return jnp.clip((db + 50.0) * (255.0 / 50.0), 0, 255).astype(jnp.uint8)
                        return jnp.stack([img.real, img.imag], axis=-1)

                    def submit(i):
                        geo = pfa.pfa_geometry(col)                    # per-collection resampling coefficients, float64 on the host
                        if kind == 'czt':
                            co = (geo['alpha'], geo['beta'] - 1.0, np.asarray(geo['delta']), geo['gam'] - 1.0)
                        else:
                            co = (geo['alpha'], geo['beta'], geo['gam'], np.asarray(geo['delta']))
                        planes = jnp.asarray(hist[i % a.nhist].view(f32).reshape(col.Np, col.K, 2))
                        y = run(planes, *(jnp.asarray(v.astype(f32)) for v in co), win)
                        return lambda: (np.asarray(y) if product == 'uint8' else np.ascontiguousarray(np.asarray(y)).view(np.complex64)[..., 0])

                    out = submit(0)()
                    assert out.shape == (N, N), out.shape
                    r = drive(submit, 1, a.images)
                    emit(dict(algo='pfaczt' if kind == 'czt' else 'pfa', policy='fp32', N=N, batch=1, product=product, nfft=nf, **r))
                except Exception as e:
                    emit(dict(algo='pfaczt' if kind == 'czt' else 'pfa', policy='fp32', N=N, batch=1, product=product,
                              error=f'{type(e).__name__}: {str(e)[:300]}'))

        for pol in [p for p in a.policies.split(',') if p]:
            p = ffbp.POLICIES[pol]
            static = ffbp.device_arrays(pol, plan, ffbp.collection_arrays(plan, col.ant))
            for B in [int(v) for v in a.batches.split(',')]:
                for product in a.products.split(','):
                    try:
                        if a.groups:
                            assert B == 1
                            single = ffbp.make_ffbp(pol, plan, groups=a.groups)

                            def fn(hre, him, arrs):
                                lv_ = []
                                for i_, e_ in enumerate(arrs['levels']):
                                    e_ = dict(e_)
                                    for k_ in (('cyc0', 'slope') if i_ == 0 else ('u', 'r0')):
                                        e_[k_] = e_[k_][0]
                                    lv_.append(e_)
                                fin_ = dict(arrs['final'], u=arrs['final']['u'][0], r0=arrs['final']['r0'][0])
                                re_, im_ = single(hre[0], him[0], dict(levels=lv_, final=fin_))
                                return re_[None], im_[None]
                        else:
                            fn = ffbp.make_ffbp_batched(pol, plan, B)

                        @jax.jit
                        def prep(S):                                   # [B, Np, K] complex64
                            s = jnp.max(jnp.abs(S), axis=(1, 2), keepdims=True)
                            return (S.real / s).astype(p['ew'])[:, None], (S.imag / s).astype(p['ew'])[:, None], s

                        @jax.jit
                        def finish(re, im, s):
                            s = s.reshape(-1, 1, 1)
                            if product == 'uint8':                     # detected image, 50 dB range below each image's peak
                                pw = re * re + im * im
                                db = 10.0 * jnp.log10(pw / jnp.max(pw, axis=(1, 2), keepdims=True) + 1e-12)
                                return jnp.clip((db + 50.0) * (255.0 / 50.0), 0, 255).astype(jnp.uint8)
                            # two float32 planes, viewed as complex on the host. A complex64 array
                            # leaves the TPU about 60 times more slowly than the same bytes as floats
                            return jnp.stack([re * s, im * s], axis=-1)

                        def fetch(y):
                            h = np.asarray(y)
                            return h if product == 'uint8' else np.ascontiguousarray(h).view(np.complex64)[..., 0]

                        def submit(i):
                            colls = [ffbp.collection_arrays(plan, col.ant) for _ in range(B)]   # one flight path per image
                            arrs = ffbp.stack_collections(pol, plan, colls, static)
                            S = jnp.asarray(np.stack([hist[(i * B + j) % a.nhist] for j in range(B)]))
                            hre, him, s = prep(S)
                            del S
                            if a.groups:                               # hold as few device arrays as possible at the largest size
                                re_, im_ = fn(hre, him, arrs)
                                del hre, him
                                y = finish(re_, im_, s)
                                del re_, im_
                            else:
                                y = finish(*fn(hre, him, arrs), s)                               # dispatched, not waited for
                            return lambda: fetch(y)

                        out = submit(0)()
                        assert out.shape == (B, N, N), out.shape
                        if product == 'complex64' and not a.groups:        # the batched program must reproduce the single-image one
                            one = ffbp.make_ffbp(pol, plan)
                            hre1, him1 = ffbp.prepare(pol, hist[0])
                            r1, i1 = one(hre1, him1, ffbp.device_arrays(pol, plan, ffbp.collection_arrays(plan, col.ant)))
                            ref1 = (np.asarray(r1) + 1j * np.asarray(i1)) * np.abs(hist[0]).max()
                            check_db = float(10 * np.log10(np.sum(np.abs(out[0] - ref1) ** 2) / np.sum(np.abs(ref1) ** 2) + 1e-30))
                            del one, r1, i1, ref1
                        else:
                            check_db = None
                        r = drive(submit, B, math.ceil(a.images / B))
                        emit(dict(algo='ffbp', policy=pol, N=N, batch=B, product=product, levels=str(levels), T=a.T, block=a.block, groups=a.groups, batched_vs_single_db=check_db, **r))
                    except Exception as e:
                        emit(dict(algo='ffbp', policy=pol, N=N, batch=B, product=product, error=f'{type(e).__name__}: {str(e)[:300]}'))

        if a.cuda:
            import cupy as cp
            from sarbench import gpu_ref
            nfft = 1 << int(np.ceil(np.log2(8 * col.K)))
            dr = bp.range_bin(col.df, nfft)
            px, py, pz = (cp.asarray(v.astype(np.float32)) for v in sim.ground_grid(N, spacing))
            for st in a.cuda.split(','):
                for product in a.products.split(','):
                    try:
                        def submit(i):
                            u, r0 = bp.host_geometry(col.ant)
                            rc = gpu_ref.range_compress(cp.asarray(hist[i % a.nhist]), nfft)
                            re, im = gpu_ref.bp(gpu_ref.pack(rc, st), cp.asarray(u.astype(np.float32)), cp.asarray(r0.astype(np.float32)),
                                                px, py, pz, nfft, dr, col.fref, st)
                            del rc
                            if product == 'uint8':
                                pw = re * re + im * im
                                y = cp.clip((10.0 * cp.log10(pw / pw.max() + 1e-12) + 50.0) * (255.0 / 50.0), 0, 255).astype(cp.uint8)
                                return lambda: cp.asnumpy(y)
                            y = cp.stack([re, im], axis=-1)
                            return lambda: np.ascontiguousarray(cp.asnumpy(y)).view(np.complex64)[..., 0]
                        submit(0)()
                        r = drive(submit, 1, a.images // 2)
                        emit(dict(algo='bp_cuda', policy=st, N=N, batch=1, product=product, **r))
                    except Exception as e:
                        emit(dict(algo='bp_cuda', policy=st, N=N, batch=1, product=product, error=f'{type(e).__name__}: {str(e)[:300]}'))
                    cp.get_default_memory_pool().free_all_blocks()


if __name__ == '__main__':
    main()
