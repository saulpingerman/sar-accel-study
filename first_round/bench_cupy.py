#!/usr/bin/env python3
"""Time (and optionally quality-check) the hand-written CUDA backprojection.

  python bench_cupy.py --label gpu-l4 --sizes 1024,2048,4096 --out results/l4_cuda.jsonl
  python bench_cupy.py --label gpu-l4 --quality data.npz --images out.npz --n 768
"""
import argparse
import json
import time

import numpy as np
import cupy as cp

from sarbench import sim, gpu_ref
from sarbench.bp import host_geometry, range_bin


def next_pow2(n):
    return 1 << int(np.ceil(np.log2(n)))


def sync():
    cp.cuda.Stream.null.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--out')
    ap.add_argument('--sizes', default='1024,2048,4096')
    ap.add_argument('--stores', default='fp32,f16,bf16')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--quality')
    ap.add_argument('--images')
    ap.add_argument('--n', type=int, default=768)
    ap.add_argument('--spacing', type=float, default=0.2)
    a = ap.parse_args()
    dev = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()

    if a.quality:
        d = np.load(a.quality)
        out = {}
        K = int(d['K'])
        nfft = next_pow2(8 * K)
        dr = range_bin(float(d['df']), nfft)
        fref = float(d['fmin']) + (K // 2) * float(d['df'])
        px, py, pz = (cp.asarray(v.astype(np.float32)) for v in sim.ground_grid(a.n, a.spacing))
        for w in ('p1', 'p2', 'pts'):
            ant = d[f'ant_{w}']
            win = sim.taylor_2d(ant.shape[0], K)
            u, r0 = host_geometry(ant)
            rc = gpu_ref.range_compress(cp.asarray((d[f'S_{w}'] * win).astype(np.complex64)), nfft)
            for st in a.stores.split(','):
                re, im = gpu_ref.bp(gpu_ref.pack(rc, st), cp.asarray(u.astype(np.float32)), cp.asarray(r0.astype(np.float32)),
                                    px, py, pz, nfft, dr, fref, st)
                out[f'bp/cuda_{st}/{w}'] = (cp.asnumpy(re) + 1j * cp.asnumpy(im)).reshape(a.n, a.n).astype(np.complex64)
        np.savez(a.images, meta=json.dumps(dict(device=dev, backend='cupy')), **out)
        print('saved', a.images, len(out))
        return

    rng = np.random.default_rng(0)
    for N in [int(s) for s in a.sizes.split(',')]:
        res = 0.3
        scene = N * res / (1.25 * np.sqrt(2.0))
        col = sim.make_collect(res=res, scene=scene)
        Np, K = col.Np, col.K
        nfft = next_pow2(8 * K)
        dr = range_bin(col.df, nfft)
        S = cp.asarray(rng.standard_normal((Np, K), np.float32) + 1j * rng.standard_normal((Np, K), np.float32))
        u, r0 = host_geometry(col.ant)
        u, r0 = cp.asarray(u.astype(np.float32)), cp.asarray(r0.astype(np.float32))
        px, py, pz = (cp.asarray(v.astype(np.float32)) for v in sim.ground_grid(N, scene / N))
        ts = []
        for _ in range(a.reps + 1):
            t = time.perf_counter(); rc = gpu_ref.range_compress(S, nfft); sync(); ts.append(time.perf_counter() - t)
        rc_s = min(ts[1:])
        for st in a.stores.split(','):
            try:
                packed = gpu_ref.pack(rc, st)
                t = time.perf_counter(); gpu_ref.bp(packed, u, r0, px, py, pz, nfft, dr, col.fref, st); sync()
                first = time.perf_counter() - t
                ts = []
                for _ in range(a.reps):
                    t = time.perf_counter(); gpu_ref.bp(packed, u, r0, px, py, pz, nfft, dr, col.fref, st); sync()
                    ts.append(time.perf_counter() - t)
                work = float(N) * N * Np
                rec = dict(label=a.label, backend='cupy', device=dev, algo='bp_cuda', policy=st, N=N, Np=Np, K=K,
                           nfft=nfft, first_s=first, run_s=min(ts), rc_s=rc_s, work=work, rate=work / min(ts))
                del packed
            except Exception as e:
                rec = dict(label=a.label, backend='cupy', device=dev, algo='bp_cuda', policy=st, N=N,
                           error=f'{type(e).__name__}: {str(e)[:300]}')
            print(json.dumps(rec), flush=True)
            if a.out:
                with open(a.out, 'a') as f:
                    f.write(json.dumps(rec) + '\n')
        del rc, S
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == '__main__':
    main()
