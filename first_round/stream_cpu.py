#!/usr/bin/env python3
"""Sustained throughput on a CPU instance with several worker processes.

A stream of independent images can be divided among processes, each pinned to
its own cores. This driver starts W workers, lets each compile, releases them
together, and reports images per second over all workers.

  python stream_cpu.py --label cpu-c4d16 --algo ffbp --N 4096 --workers 1,2,4,8 --out results/v2/stream_cpu.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np


def cores():
    """Physical cores as lists of logical CPUs."""
    seen, out = set(), []
    for c in sorted(os.sched_getaffinity(0)):
        sib = open(f'/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list').read().strip()
        if sib not in seen:
            seen.add(sib)
            ids = []
            for part in sib.split(','):
                lo, _, hi = part.partition('-')
                ids += list(range(int(lo), int(hi or lo) + 1))
            out.append(ids)
    return out


def worker(a):
    cpus = [int(c) for c in a.cpus.split(',')]
    os.sched_setaffinity(0, cpus)
    os.environ['NUMBA_NUM_THREADS'] = str(len(cpus))
    os.environ['OMP_NUM_THREADS'] = str(len(cpus))
    import jax
    import jax.numpy as jnp
    from sarbench import sim, bp, ffbp, pfa
    N = a.N
    res = 0.3
    scene = N * res / (1.25 * np.sqrt(2.0))
    col = sim.make_collect(res=res, scene=scene, r0=600e3)
    spacing = scene / N
    rng = np.random.default_rng(int(a.cpus.split(',')[0]))
    S = (rng.standard_normal((col.Np, col.K), np.float32) + 1j * rng.standard_normal((col.Np, col.K), np.float32))
    if a.algo == 'ffbp':
        levels = ffbp.default_levels(N, col.K, 32, 3)
        plan = ffbp.make_plan(col, N, spacing, levels, 32, block=a.block)
        fn = ffbp.make_ffbp('fp32_fast', plan, groups=a.groups)

        def one():
            coll = ffbp.collection_arrays(plan, col.ant)
            hre, him = ffbp.prepare('fp32_fast', S)
            re, im = fn(hre, him, ffbp.device_arrays('fp32_fast', plan, coll))
            return np.asarray(re), np.asarray(im)
    elif a.algo == 'pfaczt':
        geo0 = pfa.pfa_geometry(col)
        nf = 1 << int(np.ceil(np.log2(1.5 * N)))
        lo = (nf - N) // 2
        core = pfa.make_pfa_czt('fp32', col.K, col.Np, geo0['nkx'], geo0['nky'], nf, nf)

        def one():
            return np.asarray(core(*pfa.prepare_czt('fp32', S, pfa.pfa_geometry(col)))[lo:lo + N, lo:lo + N])
    elif a.algo == 'pfa_numba':
        from sarbench import cpu_ref
        geo0 = pfa.pfa_geometry(col)
        nf = 1 << int(np.ceil(np.log2(1.5 * N)))
        lo = (nf - N) // 2
        win = sim.taylor_2d(geo0['nkx'], geo0['nky']).astype(np.float32)

        def one():
            return cpu_ref.pfa(S, pfa.pfa_geometry(col), nf, nf, win)[lo:lo + N, lo:lo + N]
    elif a.algo == 'bp_numba':
        from sarbench import cpu_ref
        nfft = 1 << int(np.ceil(np.log2(8 * col.K)))
        px, py, pz = sim.ground_grid(N, spacing)

        def one():
            u, r0 = bp.host_geometry(col.ant)
            return cpu_ref.bp(cpu_ref.range_compress(S, nfft), u, r0, px, py, pz, bp.range_bin(col.df, nfft), col.fref)
    one()                                              # compile and warm
    open(a.sync + f'/ready_{a.id}', 'w').write('1')
    while not os.path.exists(a.sync + '/go'):
        time.sleep(0.01)
    ts = []
    for _ in range(a.n):
        t = time.time()
        one()
        ts.append(time.time() - t)
    json.dump(dict(end=time.time(), per_image_s=ts), open(a.sync + f'/done_{a.id}', 'w'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='cpu')
    ap.add_argument('--algo', default='ffbp')
    ap.add_argument('--N', type=int, default=4096)
    ap.add_argument('--workers', default='1,2,4,8')
    ap.add_argument('--n', type=int, default=3, help='images per worker')
    ap.add_argument('--out')
    ap.add_argument('--block', type=int, default=0)
    ap.add_argument('--groups', type=int, default=0)
    ap.add_argument('--cpus'); ap.add_argument('--sync'); ap.add_argument('--id')
    a = ap.parse_args()
    if a.cpus:
        return worker(a)
    cs = cores()
    for W in [int(w) for w in a.workers.split(',')]:
        if W > len(cs):
            continue
        sync = tempfile.mkdtemp()
        per = len(cs) // W
        procs = []
        for i in range(W):
            cpus = ','.join(str(c) for core in cs[i * per:(i + 1) * per] for c in core)
            procs.append(subprocess.Popen([sys.executable, __file__, '--algo', a.algo, '--N', str(a.N), '--n', str(a.n), '--block', str(a.block), '--groups', str(a.groups),
                                           '--cpus', cpus, '--sync', sync, '--id', str(i)],
                                          stdout=subprocess.DEVNULL, stderr=open(f'{sync}/err_{i}', 'w')))

        def wait_for(kind):
            while not all(os.path.exists(f'{sync}/{kind}_{i}') for i in range(W)):
                if any(p.poll() not in (None, 0) for p in procs):
                    return False
                time.sleep(0.05)
            return True

        ok = wait_for('ready')
        t0 = time.time()
        open(sync + '/go', 'w').write('1')
        ok = ok and wait_for('done')
        time.sleep(0.5)
        for p in procs:                                   # a worker that has written its result may still hang in teardown
            if p.poll() is None:
                p.kill()
        if not ok:
            rec = dict(label=a.label, algo=a.algo, N=a.N, workers=W, error=' | '.join(open(f'{sync}/err_{i}').read()[-200:] for i in range(W))[-600:])
        else:
            d = [json.load(open(f'{sync}/done_{i}')) for i in range(W)]
            el = max(x['end'] for x in d) - t0
            rec = dict(label=a.label, algo=a.algo, policy='fp32', N=a.N, block=a.block, groups=a.groups, workers=W, threads_per_worker=2 * per, images=W * a.n, seconds=el,
                       images_per_s=W * a.n / el, s_per_image=el / (W * a.n),
                       worker_image_s=float(np.mean([t for x in d for t in x['per_image_s']])))
        print(json.dumps(rec), flush=True)
        if a.out:
            with open(a.out, 'a') as f:
                f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
