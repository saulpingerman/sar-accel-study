"""Hand-written CUDA backprojection (CuPy RawKernel).

XLA materialises a pulse-by-pixel block for every stage of the JAX kernel.
The conventional GPU design instead gives each pixel a thread that walks all
pulses and keeps its running sum in registers, which is what this does. Range
profiles can be held as float32, IEEE half, or bfloat16; arithmetic is float32
because GPU scalar units gain nothing from 16-bit math.
"""
import numpy as np
import cupy as cp

C = 299792458.0

_SRC = r'''
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#if STORE == 16
  typedef __half sample_t;
  #define LOAD(v) __half2float(v)
#elif STORE == 17
  typedef __nv_bfloat16 sample_t;
  #define LOAD(v) __bfloat162float(v)
#else
  typedef float sample_t;
  #define LOAD(v) (v)
#endif
extern "C" __global__
void bp(const sample_t* rc, const float* u, const float* r0,
        const float* px, const float* py, const float* pz,
        float* out_re, float* out_im,
        int npulse, int nfft, int npix, float inv_dr, float half, float kcyc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npix) return;
    const float x = px[i], y = py[i], z = pz[i];
    const float xx = x * x + y * y + z * z;
    float are = 0.0f, aim = 0.0f;
    for (int p = 0; p < npulse; ++p) {
        const float ir = 1.0f / r0[p];
        const float q = xx * ir - 2.0f * (u[3 * p] * x + u[3 * p + 1] * y + u[3 * p + 2] * z);
        const float dR = q / (1.0f + sqrtf(1.0f + q * ir));
        const float t = dR * inv_dr + half;
        const float fl = floorf(t);
        const int i0 = (int)fl;
        if (i0 < 0 || i0 > nfft - 2) continue;
        const float w = t - fl;
        const sample_t* s = rc + 2 * ((size_t)p * nfft + i0);
        const float a_re = LOAD(s[0]), a_im = LOAD(s[1]);
        const float vr = a_re + w * (LOAD(s[2]) - a_re);
        const float vi = a_im + w * (LOAD(s[3]) - a_im);
        float cyc = dR * kcyc;
        cyc -= rintf(cyc);
        float sn, cs;
        sincospif(2.0f * cyc, &sn, &cs);
        are += vr * cs - vi * sn;
        aim += vr * sn + vi * cs;
    }
    out_re[i] = are;
    out_im[i] = aim;
}
'''

_kernels = {}
STORE = {'fp32': 32, 'f16': 16, 'bf16': 17}


def _kernel(store):
    if store not in _kernels:
        _kernels[store] = cp.RawKernel(_SRC, 'bp', options=(f'-DSTORE={STORE[store]}', '--std=c++14'))
    return _kernels[store]


def range_compress(S, nfft, block=1024):
    """[Np, K] complex64 on the GPU -> [Np, nfft] profiles, dR = 0 at nfft // 2.
    Done a block of pulses at a time so only one padded copy is alive."""
    Np, K = S.shape
    h = K // 2
    out = cp.empty((Np, nfft), cp.complex64)
    for a in range(0, Np, block):
        pad = cp.zeros((min(block, Np - a), nfft), cp.complex64)
        pad[:, :K - h] = S[a:a + block, h:]
        pad[:, nfft - h:] = S[a:a + block, :h]
        out[a:a + block] = cp.fft.fftshift(cp.fft.ifft(pad, axis=1), axes=1)
        del pad
    cp.get_default_memory_pool().free_all_blocks()
    return out


def pack(rc, store):
    """Interleaved re/im in the storage type. bfloat16 is packed on the host."""
    v = rc.view(cp.float32)
    if store == 'fp32':
        return cp.ascontiguousarray(v)
    if store == 'f16':
        return v.astype(cp.float16)
    import ml_dtypes
    return cp.asarray(cp.asnumpy(v).astype(ml_dtypes.bfloat16).view(np.uint16))


def bp(rc_packed, u, r0, px, py, pz, nfft, dr, fc, store='fp32'):
    npix, npulse = px.size, r0.size
    out_re = cp.empty(npix, cp.float32)
    out_im = cp.empty(npix, cp.float32)
    block = 256
    _kernel(store)(((npix + block - 1) // block,), (block,),
                   (rc_packed, u, r0, px, py, pz, out_re, out_im,
                    np.int32(npulse), np.int32(nfft), np.int32(npix),
                    np.float32(1.0 / dr), np.float32(nfft // 2), np.float32(2.0 * fc / C)))
    return out_re, out_im
