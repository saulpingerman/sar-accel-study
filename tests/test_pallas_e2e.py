"""End-to-end check: the factorized image with the pallas filter equals the dense-filter image on a small simulated
scene (CPU, Pallas interpret mode). Skips the kernel's interpret flag plumbing by monkeypatching.

  uv run --with jax --with numpy --with scipy python tests/test_pallas_e2e.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')
from dev import sim, ffbp2
from dev import pallas_ffbp, pallas_ffbp_gpu

# run the kernels in interpret mode on the CPU (the CPU path selects the GPU-tiled kernel)
_orig = pallas_ffbp.fused_rotate_dec_k
pallas_ffbp.fused_rotate_dec_k = lambda *a, **k: _orig(*a, **{**k, 'interpret': True})
_orig2 = pallas_ffbp.fused_rotate_dec_k2
pallas_ffbp.fused_rotate_dec_k2 = lambda *a, **k: _orig2(*a, **{**k, 'interpret': True})
_orig_g = pallas_ffbp_gpu.fused_rotate_dec_k_gpu
pallas_ffbp_gpu.fused_rotate_dec_k_gpu = lambda *a, **k: _orig_g(*a, **{**k, 'interpret': True})
_orig_g2 = pallas_ffbp_gpu.fused_rotate_dec_k2_gpu
pallas_ffbp_gpu.fused_rotate_dec_k2_gpu = lambda *a, **k: _orig_g2(*a, **{**k, 'interpret': True})
_orig_gf = pallas_ffbp_gpu.fused_final_gpu
pallas_ffbp_gpu.fused_final_gpu = lambda *a, **k: _orig_gf(*a, **{**k, 'interpret': True})

rng = np.random.default_rng(1)
col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
pos = np.stack([rng.uniform(-25, 25, 40), rng.uniform(-25, 25, 40), np.zeros(40)], 1)
amp = rng.standard_normal(40) + 1j * rng.standard_normal(40)
S = sim.simulate_brute(col, pos, amp).astype(np.complex64)
P, K = S.shape
n = 128
import os as _os
nlev = int(_os.environ.get('NLEV', '2'))
plan = ffbp2.make_plan(col, n, n, 0.5, 0.5, T=16, nlev=nlev, pmax=0.4)
print('levels', nlev)
print('P K', P, K, 'levels', [(l['P'], l['K'], l['Po'], l['Ko'], l['Dk'], l['Dp'], l['C']) for l in plan['levels']])
coll = ffbp2.collection_arrays(plan, col.ant)
hre, him, _ = ffbp2.prepare('fp32_fast', S)
out = {}
for filt, kw in (('dense', {}), ('pallas', dict(pallas_pb=8, pallas_chunk=256)), ('pallas2', dict(pallas_pb=16, pallas_nc=4, pallas_ng=2))):
    static = ffbp2.static_arrays('fp32_fast', plan, filt)
    arrs = ffbp2.device_arrays('fp32_fast', plan, coll, static)
    fn = ffbp2.make_ffbp('fp32_fast', plan, filt, 1 << 24, 'split', **kw)
    re, im = fn(hre, him, arrs)
    out[filt] = np.asarray(re) + 1j * np.asarray(im)
    print(filt, 'done', out[filt].shape, float(np.abs(out[filt]).max()))
for k in ('pallas', 'pallas2'):
    d = out[k] - out['dense']
    print(k, 'vs dense: %.1f dB' % (10 * np.log10(np.sum(np.abs(d) ** 2) / np.sum(np.abs(out['dense']) ** 2))))
