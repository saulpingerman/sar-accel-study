import numpy as np, jax
from dev import sim, ffbp
rng = np.random.default_rng(0)
N = 1024
res = 0.3
col = sim.make_collect(res=res, scene=N * res / (1.25 * np.sqrt(2.0)))
# odd pulse count to exercise uneven blocks
S = (rng.standard_normal((col.Np, col.K), np.float32) + 1j * rng.standard_normal((col.Np, col.K), np.float32))
spacing = res / (1.25 * np.sqrt(2.0))
levels = ffbp.default_levels(N, col.K, 32, 3)
plan = ffbp.make_plan(col, N, spacing, levels, 32)
coll = ffbp.collection_arrays(plan, col.ant)
for pol in ('fp32', 'bf16_mm', 'f16', 'fp32_fast'):
    hre, him = ffbp.prepare(pol, S)
    arrs = ffbp.device_arrays(pol, plan, coll)
    a = ffbp.make_ffbp(pol, plan)(hre, him, arrs)
    for G, bud in ((16, 1 << 26), (16, 300000), (64, 300000), (4, 100000)):
        b = ffbp.make_ffbp(pol, plan, bud, groups=G)(hre, him, arrs)
        a_, b_ = np.asarray(a[0]) + 1j * np.asarray(a[1]), np.asarray(b[0]) + 1j * np.asarray(b[1])
        print(pol, 'groups', G, 'budget', bud, 'blocks', -(-(col.Np * col.K) // bud), 'difference dB', round(float(10 * np.log10(np.sum(np.abs(a_ - b_) ** 2) / np.sum(np.abs(a_) ** 2) + 1e-30)), 1), flush=True)
