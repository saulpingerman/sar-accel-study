#!/usr/bin/env python3
"""Split the factorized algorithm's host planning time into filter design
(reusable for a given imaging mode) and geometry (needed for every collection)."""
import sys, time, numpy as np
from sarbench import sim, ffbp
for N in (1024, 2048, 4096, 8192):
    res = 0.3; scene = N * res / (1.25 * np.sqrt(2.0)); col = sim.make_collect(res=res, scene=scene)
    levels = ffbp.default_levels(N, col.K, 32, 3)
    calls = []
    orig = ffbp.decimator
    def timed(*a, **k):
        t = time.perf_counter(); out = orig(*a, **k); calls.append(time.perf_counter() - t); return out
    ffbp.decimator = timed
    t = time.perf_counter(); plan = ffbp.make_plan(col, N, scene / N, levels, 32); total = time.perf_counter() - t
    ffbp.decimator = orig
    print(f'N={N} plan {total:.3f}s  filters {sum(calls):.3f}s  geometry {total - sum(calls):.3f}s', flush=True)
