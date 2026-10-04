#!/usr/bin/env python3
"""Throughput benchmark for SAR image formation on whatever device JAX sees.

Each configuration is compiled, run once, then timed over `--reps` warm runs.
Timing is data independent, so the phase history is random; geometry is real.
One JSON line per configuration is appended to `--out`.

  python bench.py --label tpu-v5e --sizes 1024,2048,4096 --out results/tpu-v5e.jsonl
"""
import argparse
import json
import platform
import time

import numpy as np


def next_pow2(n):
    return 1 << int(np.ceil(np.log2(n)))


def timed(fn, args, reps):
    import jax
    t = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - t
    runs = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        runs.append(time.perf_counter() - t)
    leaf = jax.tree_util.tree_leaves(fn(*args))[0]
    return first, runs, str(leaf.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ffbp-blocks', default='0', help='block sizes for banded decimation; 0 = dense matrices')
    ap.add_argument('--ffbp-groups', type=int, default=0, help='carry the image through in this many groups of first-level tiles (bounded memory)')
    ap.add_argument('--ffbp-left', action='store_true', help='apply the pulse-axis filter from the left')
    ap.add_argument('--algos', default='ffbp,bp,pfa,pfaczt,pfamm,rc,matmul')
    ap.add_argument('--sizes', default='1024,2048,4096')
    ap.add_argument('--policies', default='fp32,bf16_store,bf16_arith,bf16_acc,f16_arith,fp64')
    ap.add_argument('--ffbp-configs', default='64:2', help='tile size:levels, comma separated')
    ap.add_argument('--ffbp-policies', default='fp32,fp32_fast,bf16_mm,bf16,f16_mm,f16,fp64')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--budget', type=int, default=1 << 26, help='pulse-pixel elements per scan step')
    ap.add_argument('--tmax', type=float, default=180.0, help='skip configs predicted slower than this')
    ap.add_argument('--x64', action='store_true', help='enable float64 (not available on TPU)')
    ap.add_argument('--numba', action='store_true', help='also time the multithreaded CPU kernels')
    a = ap.parse_args()

    import jax
    jax.config.update('jax_enable_x64', a.x64)
    import jax.numpy as jnp
    from sarbench import sim, bp, pfa, ffbp

    dev = str(jax.devices()[0])
    base = dict(label=a.label, backend=jax.default_backend(), device=dev, n_devices=len(jax.devices()),
                jax=jax.__version__, host=platform.processor() or platform.machine())
    algos = a.algos.split(',')
    sizes = [int(s) for s in a.sizes.split(',')]
    policies = [p for p in a.policies.split(',') if a.x64 or not bp.needs_x64(p)]
    rate = {}

    def emit(rec):
        rec = {**base, **rec}
        print(json.dumps(rec), flush=True)
        with open(a.out, 'a') as f:
            f.write(json.dumps(rec) + '\n')

    def run(algo, policy, N, work, build):
        key = (algo, policy)
        if key in rate and work / rate[key] > a.tmax:
            emit(dict(algo=algo, policy=policy, N=N, skipped=f'predicted {work / rate[key]:.0f}s'))
            return
        try:
            fn, args, extra = build()
            first, runs, out_dtype = timed(fn, args, a.reps)
            best = min(runs)
            rate[key] = work / best
            emit(dict(algo=algo, policy=policy, N=N, first_s=first, run_s=best,
                      run_med_s=float(np.median(runs)), work=work, rate=work / best, out_dtype=out_dtype, **extra))
        except Exception as e:  # keep going: an unsupported type on one device is a result
            emit(dict(algo=algo, policy=policy, N=N, error=f'{type(e).__name__}: {str(e)[:400]}'))

    if 'matmul' in algos:
        n = 8192
        for dt in ['float32', 'bfloat16', 'float16']:
            def build(dt=dt):
                x = jnp.asarray(np.random.default_rng(0).standard_normal((n, n), np.float32)).astype(dt)
                return jax.jit(lambda m: jnp.matmul(m, m, preferred_element_type=jnp.float32)), (x,), {}
            run('matmul', dt, n, 2.0 * n ** 3, build)

    rng = np.random.default_rng(0)
    for N in sizes:
        res = 0.3
        col = sim.make_collect(res=res, scene=N * res / (1.25 * np.sqrt(2.0)))
        Np, K = col.Np, col.K
        nfft = next_pow2(8 * K)
        dr = bp.range_bin(col.df, nfft)
        chunk = max(1, 1 << int(np.floor(np.log2(max(1, a.budget // (N * N))))))
        S = (rng.standard_normal((Np, K), np.float32) + 1j * rng.standard_normal((Np, K), np.float32))
        spacing = N * res / (1.25 * np.sqrt(2.0)) / N
        px, py, pz = sim.ground_grid(N, spacing) if ('bp' in algos or a.numba) else (None, None, None)

        if 'rc' in algos:
            def build():
                return jax.jit(bp.range_compress, static_argnums=1), (jnp.asarray(S), nfft), dict(Np=Np, K=K, nfft=nfft)
            run('rc', 'fp32', N, float(Np) * nfft, build)

        for cfg in a.ffbp_configs.split(',') if 'ffbp' in algos else []:
          for block in [int(v) for v in a.ffbp_blocks.split(',')]:
            T, nlev = (int(v) for v in cfg.split(':'))
            name = ('ffbp' if cfg == '64:2' else f'ffbp_T{T}L{nlev}') + ('nofilter' if block < 0 else f'b{block}' if block else '') + ('left' if a.ffbp_left else '')
            levels = ffbp.default_levels(N, K, T, nlev)
            t0 = time.perf_counter()
            plan = ffbp.make_plan(col, N, spacing, levels, T, block=block)
            plan_s = time.perf_counter() - t0
            host = []                                  # per-collection host work, timed on its own
            for _ in range(3):
                t0 = time.perf_counter()
                coll = ffbp.collection_arrays(plan, col.ant)
                host.append(time.perf_counter() - t0)
            host_s = min(host)
            for pol in a.ffbp_policies.split(','):
                if ffbp.needs_x64(pol) and not a.x64:
                    continue
                def build(pol=pol, plan=plan, levels=levels, plan_s=plan_s, coll=coll, host_s=host_s, block=block):
                    hre, him = ffbp.prepare(pol, S)
                    return (ffbp.make_ffbp(pol, plan, a.budget, left=a.ffbp_left, groups=a.ffbp_groups), (hre, him, ffbp.device_arrays(pol, plan, coll)),
                            dict(Np=Np, K=K, plan_s=plan_s, host_s=host_s, levels=str(levels), T=T, block=block, groups=a.ffbp_groups))
                run(name, pol, N, float(N) * N * Np, build)

        if 'bp' in algos:
            for pol in policies:
                def build(pol=pol):
                    args = bp.prepare(pol, S, col.ant, nfft, chunk)
                    g = jnp.dtype(bp.POLICIES[pol]['geom'])
                    h = np.float64 if g == jnp.float64 else np.float32
                    pix = tuple(jnp.asarray(v.astype(h)) for v in (px, py, pz))
                    return bp.make_bp(pol, nfft, dr, col.fref, chunk), args + pix, dict(Np=Np, K=K, nfft=nfft, chunk=chunk)
                run('bp', pol, N, float(N) * N * Np, build)

        if 'pfa' in algos:
            geo = pfa.pfa_geometry(col)
            nf = next_pow2(int(1.5 * N))
            for pol in policies:
                if pol in ('bf16_acc', 'f16_acc', 'fp32_naive', 'bf16_all'):
                    continue  # identical to the *_arith / fp32 path in this algorithm
                def build(pol=pol):
                    return pfa.make_pfa(pol, geo['nkx'], geo['nky'], nf, nf), pfa.prepare(pol, S, geo), dict(Np=Np, K=K, nfft=nf)
                run('pfa', pol, N, float(Np) * K, build)

        if 'pfaczt' in algos:
            geo = pfa.pfa_geometry(col)
            nf = next_pow2(int(1.5 * N))
            for pol in ['fp32'] + (['fp64'] if a.x64 else []):
                def build(pol=pol):
                    return (pfa.make_pfa_czt(pol, K, Np, geo['nkx'], geo['nky'], nf, nf),
                            pfa.prepare_czt(pol, S, geo), dict(Np=Np, K=K, nfft=nf))
                run('pfaczt', pol, N, float(Np) * K, build)

        if 'pfamm' in algos:
            geo = pfa.pfa_geometry(col)
            nf = next_pow2(int(1.5 * N))
            sx, sy = pfa.pixel_spacing(geo, nf, nf)
            for pol in pfa.MM_POLICIES:
                def build(pol=pol):
                    return (pfa.make_pfa_mm(pol, geo['nkx'], geo['nky'], nf, nf),
                            pfa.prepare_mm(pol, S, geo, nf, nf, sx, sy), dict(Np=Np, K=K, nfft=nf))
                run('pfamm', pol, N, float(Np) * K, build)

        if a.numba:
            from sarbench import cpu_ref
            u, r0 = bp.host_geometry(col.ant)
            for name, cdt in [('fp32', np.complex64)] + ([('fp64', np.complex128)] if a.x64 else []):
                Sx = S.astype(cdt)
                key = ('bp_numba', name)
                work = float(N) * N * Np
                if key in rate and work / rate[key] > a.tmax:
                    emit(dict(algo='bp_numba', policy=name, N=N, skipped='predicted too slow'))
                else:
                    t = time.perf_counter(); rc = cpu_ref.range_compress(Sx, nfft); t_rc = time.perf_counter() - t
                    cpu_ref.bp(rc[:2], u[:2], r0[:2], px[:64], py[:64], pz[:64], dr, col.fref)  # compile
                    runs = []
                    for _ in range(max(1, a.reps - 1)):
                        t = time.perf_counter(); cpu_ref.bp(rc, u, r0, px, py, pz, dr, col.fref); runs.append(time.perf_counter() - t)
                    rate[key] = work / min(runs)
                    emit(dict(algo='bp_numba', policy=name, N=N, run_s=min(runs), rc_s=t_rc, work=work,
                              rate=work / min(runs), Np=Np, K=K, nfft=nfft))
                geo = pfa.pfa_geometry(col)
                nf = next_pow2(int(1.5 * N))
                win = np.ones((geo['nkx'], geo['nky']))
                cpu_ref.pfa(Sx, geo, nf, nf, win)  # compile + warm
                runs = []
                for _ in range(a.reps):
                    t = time.perf_counter(); cpu_ref.pfa(Sx, geo, nf, nf, win); runs.append(time.perf_counter() - t)
                emit(dict(algo='pfa_numba', policy=name, N=N, run_s=min(runs), work=float(Np) * K,
                          rate=float(Np) * K / min(runs), Np=Np, K=K, nfft=nf))


if __name__ == '__main__':
    main()
