"""Toy-size check of the GPU-tiled fused kernel against the dense product (interpret mode, CPU)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')
from dev.pallas_ffbp_gpu import band_blocks_gpu, pad_columns_gpu, fused_rotate_dec_k_gpu
rng = np.random.default_rng(0)
P, K, D, L = 32, 1536, 6, 44
Ko = (K + D - 1) // D
taps = np.kaiser(L, 8.0) * np.sinc((np.arange(L) - (L - 1) / 2) / D) / D
pl_ = L // 2 - D // 2
Fk = np.zeros((K, Ko))
for j in range(Ko):
    for r in range(L):
        i = D * j + r - pl_
        if 0 <= i < K:
            Fk[i, j] = taps[r]
S = (rng.standard_normal((P, K)) + 1j * rng.standard_normal((P, K))).astype(np.complex64)
c0 = rng.uniform(-1e4, 1e4, P).astype(np.float32); slope = rng.uniform(-0.3, 0.3, P).astype(np.float32)
kc = (K - 1) / 2.0
cyc = c0[:, None].astype(np.float64) + (np.arange(K) - kc)[None, :] * slope[:, None].astype(np.float64)
Yref = (S.astype(np.complex128) * np.exp(1j * (cyc - np.round(cyc)) * 2 * np.pi)) @ Fk
band = band_blocks_gpu(Fk, D)
print('win', band['win'], 'nb', band['nb'], 'Kpad', band['Kpad'])
Sre, Sim = pad_columns_gpu(jnp.asarray(S.real), band), pad_columns_gpu(jnp.asarray(S.imag), band)
for mm, prec in ((jnp.float32, 'highest'), (jnp.float16, None)):
    yr, yi = fused_rotate_dec_k_gpu(Sre, Sim, jnp.asarray(c0), jnp.asarray(slope), band, kc, mm_dtype=mm, precision=prec, pb=16, interpret=True)
    Y = np.asarray(yr)[:, :Ko] + 1j * np.asarray(yi)[:, :Ko]
    print(mm.__name__ if hasattr(mm,'__name__') else mm, prec, 'error %.1f dB' % (10 * np.log10(np.sum(np.abs(Y - Yref) ** 2) / np.sum(np.abs(Yref) ** 2))))
