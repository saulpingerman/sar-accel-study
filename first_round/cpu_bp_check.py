#!/usr/bin/env python3
"""Time the Numba backprojection alone and report whether SVML is in use."""
import sys, time, numpy as np, numba
from sarbench import sim, cpu_ref
from sarbench.bp import host_geometry, range_bin
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
print('numba', numba.__version__, 'threads', numba.get_num_threads(), 'svml', numba.config.USING_SVML)
res = 0.3; scene = N * res / (1.25 * np.sqrt(2.0)); col = sim.make_collect(res=res, scene=scene)
nfft = 1 << int(np.ceil(np.log2(8 * col.K))); dr = range_bin(col.df, nfft)
rng = np.random.default_rng(0)
S = (rng.standard_normal((col.Np, col.K), np.float32) + 1j * rng.standard_normal((col.Np, col.K), np.float32))
rc = cpu_ref.range_compress(S, nfft); u, r0 = host_geometry(col.ant); px, py, pz = sim.ground_grid(N, scene / N)
cpu_ref.bp(rc[:2], u[:2], r0[:2], px[:64], py[:64], pz[:64], dr, col.fref)
for blk in (1024, 4096, 16384):
    t = time.perf_counter(); out = cpu_ref.bp(rc, u, r0, px, py, pz, dr, col.fref, blk); dt = time.perf_counter() - t
    print(f'N={N} blk={blk} {dt:.2f}s rate {N*N*col.Np/dt:.3e}', flush=True)
