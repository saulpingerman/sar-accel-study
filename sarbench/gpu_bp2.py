"""Exact backprojection on a GPU, second version (CuPy RawKernel).

Differences from `gpu_ref.py`:

  * Pulses are range compressed and backprojected a chunk at a time, so the
    device holds the range profiles of one chunk and not of the collection.
  * The range to each pixel is taken relative to the centre of a small image
    tile. The range difference between the tile centre and the scene centre is
    computed in float64 once per tile and pulse; the remainder, between the
    pixel and its tile centre, is at most a few tens of metres and float32
    resolves it to well under a thousandth of a wavelength. The earlier kernel
    took the whole difference from the scene centre in float32 and lost
    accuracy on scenes more than about a kilometre across.
  * The image is a rectangle with separate pixel spacings; pixel coordinates
    are generated in the kernel.
"""
import numpy as np
import cupy as cp

C = 299792458.0

_SRC = r'''
#include <cuda_fp16.h>
#if STORE == 16
  typedef __half sample_t;
  #define LOAD(v) __half2float(v)
#else
  typedef float sample_t;
  #define LOAD(v) (v)
#endif
extern "C" __global__
void bp2(const sample_t* rc, const int* tint, const float* tfrac, const float* cfrac, const float* w,
         float* out_re, float* out_im,
         int npc, int nfft, int nx, int ny, int tile, int nty, float spx, float spy, float inv_dr, float kcyc)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nx * ny) return;
    const int i = (int)(idx / ny), j = (int)(idx % ny);
    const int ti = i / tile, tj = j / tile;
    const size_t base = (size_t)(ti * nty + tj) * npc;
    const float dx = ((float)(i - ti * tile) - 0.5f * (float)(tile - 1)) * spx;
    const float dy = ((float)(j - tj * tile) - 0.5f * (float)(tile - 1)) * spy;
    const float dd = dx * dx + dy * dy;
    float are = out_re[idx], aim = out_im[idx];
    for (int p = 0; p < npc; ++p) {
        const float* wp = w + 4 * (base + p);
        const float ir = wp[3];
        const float q = dd * ir - 2.0f * (wp[0] * dx + wp[1] * dy);
        const float ddr = q / (1.0f + sqrtf(1.0f + q * ir));
        const float t = tfrac[base + p] + ddr * inv_dr;
        const float fl = floorf(t);
        const int i0 = tint[base + p] + (int)fl;
        if (i0 < 0 || i0 > nfft - 2) continue;
        const float wg = t - fl;
        const sample_t* s = rc + 2 * ((size_t)p * nfft + i0);
        const float a_re = LOAD(s[0]), a_im = LOAD(s[1]);
        const float vr = a_re + wg * (LOAD(s[2]) - a_re);
        const float vi = a_im + wg * (LOAD(s[3]) - a_im);
        float cyc = cfrac[base + p] + ddr * kcyc;
        cyc -= rintf(cyc);
        float sn, cs;
        sincospif(2.0f * cyc, &sn, &cs);
        are += vr * cs - vi * sn;
        aim += vr * sn + vi * cs;
    }
    out_re[idx] = are;
    out_im[idx] = aim;
}
'''

_kernels = {}


def _kernel(store):
    if store not in _kernels:
        _kernels[store] = cp.RawKernel(_SRC, 'bp2', options=(f"-DSTORE={16 if store == 'f16' else 32}", '--std=c++14'))
    return _kernels[store]


def _compress(S, nfft):
    """[n, K] complex64 on the GPU -> [n, nfft] range profiles with zero range difference at nfft // 2."""
    n, K = S.shape
    h = K // 2
    pad = cp.zeros((n, nfft), cp.complex64)
    pad[:, :K - h] = S[:, h:]
    pad[:, nfft - h:] = S[:, :h]
    return cp.fft.fftshift(cp.fft.ifft(pad, axis=1), axes=1)


def bp2(S, ant, fmin, df, nx, ny, spx, spy, store='fp32', oversample=8, chunk=256, tile=32, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0)):
    """S: [P, K] complex64 phase history on the GPU (already weighted). ant: [P, 3] float64 antenna positions on
    the host, scene-centred. Returns (re, im), each [nx, ny] float32 on the GPU, for pixel (i, j) at
    (i - nx / 2) spx e1 + (j - ny / 2) spy e2, with e1 and e2 the orthonormal basis of the image plane."""
    P, K = S.shape
    nfft = 1 << int(np.ceil(np.log2(oversample * K)))
    dr = C / (2.0 * df * nfft)
    fc = fmin + (K // 2) * df
    inv_dr, kcyc, half = 1.0 / dr, 2.0 * fc / C, nfft // 2
    ntx, nty = -(-nx // tile), -(-ny // tile)
    tcx = (cp.arange(ntx, dtype=cp.float64) * tile + 0.5 * (tile - 1) - nx / 2.0) * spx
    tcy = (cp.arange(nty, dtype=cp.float64) * tile + 0.5 * (tile - 1) - ny / 2.0) * spy
    e1d, e2d = cp.asarray(np.asarray(e1, np.float64)), cp.asarray(np.asarray(e2, np.float64))
    tc = cp.repeat(tcx, nty)[:, None] * e1d[None, :] + cp.tile(tcy, ntx)[:, None] * e2d[None, :]   # [Nt, 3]
    out_re = cp.zeros(nx * ny, cp.float32)
    out_im = cp.zeros(nx * ny, cp.float32)
    a_all = cp.asarray(ant, cp.float64)
    block = 256
    kern = _kernel(store)
    for p0 in range(0, P, chunk):
        a = a_all[p0:p0 + chunk]
        npc = a.shape[0]
        rc = _compress(S[p0:p0 + chunk], nfft).view(cp.float32)
        if store == 'f16':
            rc = rc.astype(cp.float16)
        rc = cp.ascontiguousarray(rc)
        # float64 per tile and pulse: range of the tile centre less that of the scene centre, unit vector, 1 / range
        wv = a[None, :, :] - tc[:, None, :]                                                    # [Nt, npc, 3]
        rcn = cp.sqrt((wv * wv).sum(2))
        basev = rcn - cp.sqrt((a * a).sum(1))[None, :]
        bt = basev * inv_dr + half
        ti = cp.floor(bt)
        cyc = basev * kcyc
        w4 = cp.stack([(wv @ e1d) / rcn, (wv @ e2d) / rcn, cp.zeros_like(rcn), 1.0 / rcn], axis=2).astype(cp.float32)
        tint = cp.ascontiguousarray(ti.astype(cp.int32))
        tfrac = cp.ascontiguousarray((bt - ti).astype(cp.float32))
        cfrac = cp.ascontiguousarray((cyc - cp.rint(cyc)).astype(cp.float32))
        del wv, rcn, basev, bt, ti, cyc
        kern(((nx * ny + block - 1) // block,), (block,),
             (rc, tint, tfrac, cfrac, cp.ascontiguousarray(w4), out_re, out_im,
              np.int32(npc), np.int32(nfft), np.int32(nx), np.int32(ny), np.int32(tile), np.int32(nty),
              np.float32(spx), np.float32(spy), np.float32(inv_dr), np.float32(kcyc)))
        del rc, tint, tfrac, cfrac, w4
        cp.get_default_memory_pool().free_all_blocks()
    return out_re.reshape(nx, ny), out_im.reshape(nx, ny)
