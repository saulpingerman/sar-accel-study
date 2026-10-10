#!/usr/bin/env python3
"""Sustained throughput of the factorized algorithm on the CPU instance with the stream divided among W
worker processes pinned to disjoint physical cores, on a prepared collection at its native grid.

  python cpu_stream.py --data /data/panama.npz --workers 1,2,4 --n 1 --filt conv --trig direct --out results/fastsar/cpu_stream.jsonl
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
    """Lists of logical CPUs that share a physical core."""
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


def util_sample():
    v = [float(x) for x in open('/proc/stat').readline().split()[1:8]]
    return sum(v), v[3] + v[4]


def worker(a):
    cpus = [int(c) for c in a.cpus.split(',')]
    os.sched_setaffinity(0, cpus)
    os.environ['OMP_NUM_THREADS'] = str(len(cpus))
    os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + f' --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={len(cpus)}'
    import jax
    from scipy.signal.windows import taylor
    import prep
    from dev import ffbp2
    col, S, grid = prep.load(a.data)
    P, K = S.shape
    Sw = S * taylor(P, nbar=4, sll=35.0, norm=False).astype(np.float32)[:, None] * taylor(K, nbar=4, sll=35.0, norm=False).astype(np.float32)[None, :]
    plan = ffbp2.make_plan(col, grid['nx'], grid['ny'], grid['spx'], grid['spy'], T=a.T, nlev=a.levels, pmax=a.pmax, e1=grid['e1'], e2=grid['e2'])
    fn = ffbp2.make_ffbp(a.policy, plan, a.filt, trig=a.trig)
    static = ffbp2.static_arrays(a.policy, plan, a.filt)

    def one():
        arrs = ffbp2.device_arrays(a.policy, plan, ffbp2.collection_arrays(plan, col.ant), static)
        hre, him, scale = ffbp2.prepare(a.policy, Sw)
        re, im = fn(hre, him, arrs)
        return np.asarray(re), np.asarray(im)

    one()
    open(f'{a.sync}/ready_{a.id}', 'w').write('1')
    while not os.path.exists(f'{a.sync}/go'):
        time.sleep(0.05)
    t = time.perf_counter()
    for _ in range(a.n):
        one()
    open(f'{a.sync}/done_{a.id}', 'w').write(str(time.perf_counter() - t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--label', default='cpu-c4d16')
    ap.add_argument('--workers', default='1,2,4')
    ap.add_argument('--n', type=int, default=1)
    ap.add_argument('--policy', default='fp32')
    ap.add_argument('--filt', default='conv')
    ap.add_argument('--trig', default='direct')
    ap.add_argument('--T', type=int, default=32)
    ap.add_argument('--levels', type=int, default=3)
    ap.add_argument('--pmax', type=float, default=0.4)
    ap.add_argument('--out')
    ap.add_argument('--cpus'); ap.add_argument('--sync'); ap.add_argument('--id', type=int)
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
            procs.append(subprocess.Popen([sys.executable, __file__, '--data', a.data, '--n', str(a.n), '--policy', a.policy, '--filt', a.filt, '--trig', a.trig,
                                           '--T', str(a.T), '--levels', str(a.levels), '--pmax', str(a.pmax), '--cpus', cpus, '--sync', sync, '--id', str(i)],
                                          stdout=subprocess.DEVNULL, stderr=open(f'{sync}/err_{i}', 'w')))

        def wait_for(kind):
            while not all(os.path.exists(f'{sync}/{kind}_{i}') for i in range(W)):
                if any(p.poll() not in (None, 0) for p in procs):
                    return False
                time.sleep(0.05)
            return True

        ok = wait_for('ready')
        tot0, idle0 = util_sample()
        t0 = time.time()
        open(sync + '/go', 'w').write('1')
        ok = ok and wait_for('done')
        el = time.time() - t0
        tot1, idle1 = util_sample()
        for p in procs:
            p.kill()
        if not ok:
            print('worker failure:', open(f'{sync}/err_0').read()[-600:], flush=True)
            continue
        rec = dict(label=a.label, policy=a.policy, filt=a.filt, trig=a.trig, workers=W, threads_per_worker=2 * per, images=W * a.n, seconds=el,
                   images_per_s=W * a.n / el, s_per_image=el / (W * a.n), cpu_util=1.0 - (idle1 - idle0) / (tot1 - tot0),
                   worker_image_s=float(np.mean([float(open(f'{sync}/done_{i}').read()) for i in range(W)])) / a.n)
        print(json.dumps(rec), flush=True)
        if a.out:
            with open(a.out, 'a') as f:
                f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
