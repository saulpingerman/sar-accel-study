"""Exact against factorized backprojection by image size, on one device, with the released FastSAR's public API:
square grids of n by n pixels at the scene center of the Umbra Panama collection (the vendor's spacings and plane,
all pulses), each former warm (the better of two calls after the first), host memory to host memory.
   python -I crossover.py <panama .npz> <backend cpu|cuda> <out .json> [sizes]"""
import sys, json, time, os, subprocess
import numpy as np
import fastsar
d = np.load(sys.argv[1]); backend = sys.argv[2]
S, ant, fmin, df = d['S'], d['ant'], float(d['fmin']), float(d['df'])
spx, spy, e1, e2 = float(d['spx']), float(d['spy']), d['e1'], d['e2']
sizes = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else '1024,2048,4096,8192').split(',')]
out = dict(fastsar=subprocess.run(['git', '-C', os.path.dirname(fastsar.__path__[0]), 'log', '-1', '--format=%H'], capture_output=True, text=True).stdout.strip(),
           backend=backend, pulses=int(S.shape[0]), sizes={})


def warm(f):
    f(S)
    ts = []
    for _ in range(2):
        t = time.perf_counter(); f(S); ts.append(time.perf_counter() - t)
    return min(ts)
for n in sizes:
    r = {}
    r['exact_cubic_s'] = warm(fastsar.ExactFormer(ant, fmin, df, S.shape[1], n, n, spx, spy, e1, e2, backend=backend))
    r['factorized_s'] = warm(fastsar.ImageFormer(ant, fmin, df, S.shape[1], n, n, spx=spx, spy=spy, e1=e1, e2=e2, backend=backend))
    out['sizes'][n] = r
    print(n, r, flush=True)
json.dump(out, open(sys.argv[3], 'w'), indent=1)
