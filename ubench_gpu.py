"""Achievable unit rates of the GPU for the operations the CUDA factorized kernels use (CuPy RawKernel), for the
roofline bounds of the kernels appendix.   python ubench_gpu.py --out ubench_gpu.json

Measures: HBM read+write bandwidth; L2 read bandwidth; the level kernel's inner mix (geometric-sequence rotation and
the 44-tap filter) on shared-memory-resident data; the final kernel's inner mix (4 by 2 pixel accumulation from
shared-memory tables); sincospif throughput; and coalesced against strided global stores.
"""
import argparse, json, time

import numpy as np
import cupy as cp

SRC = r'''
extern "C" __global__ void fir_mix(const float* __restrict__ seed, float* __restrict__ out, int iters)
{
    // 256 threads: 32 pulses x 8 column groups, window of 141 floats per pulse (two planes), as in rot_fir_k
    __shared__ float zr[32 * 141], zi[32 * 141], tp[44];
    const int tid = threadIdx.x, pp = tid % 32, jt = tid / 32;
    for (int i = tid; i < 32 * 141; i += 256) { zr[i] = seed[i % 1024]; zi[i] = seed[(i * 7) % 1024]; }
    for (int r = tid; r < 44; r += 256) tp[r] = 0.01f * r;
    __syncthreads();
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    for (int it = 0; it < iters; ++it) {
        // rotation of this thread's samples by a geometric sequence (one sincos, then complex multiplies)
        float cyc = 0.001f * (float)(it + tid); cyc -= rintf(cyc);
        float wr, wi, sr, si; sincospif(2.0f * cyc, &wi, &wr); sincospif(0.002f, &si, &sr);
        for (int i = jt; i < 141; i += 8) {
            const float vr = zr[pp * 141 + i], vi = zi[pp * 141 + i];
            zr[pp * 141 + i] = vr * wr - vi * wi; zi[pp * 141 + i] = vr * wi + vi * wr;
            const float nr = wr * sr - wi * si; wi = wr * si + wi * sr; wr = nr;
        }
        __syncthreads();
        // two adjacent outputs of the 44-tap filter
        const float* row_r = zr + pp * 141 + 2 * jt * 6; const float* row_i = zi + pp * 141 + 2 * jt * 6;
        float a0r = 0.f, a0i = 0.f, a1r = 0.f, a1i = 0.f;
        for (int r = 0; r < 6; ++r) { const float t = tp[r]; a0r = fmaf(t, row_r[r], a0r); a0i = fmaf(t, row_i[r], a0i); }
        for (int r = 6; r < 44; ++r) { const float vr = row_r[r], vi = row_i[r]; const float t0 = tp[r], t1 = tp[r - 6];
            a0r = fmaf(t0, vr, a0r); a0i = fmaf(t0, vi, a0i); a1r = fmaf(t1, vr, a1r); a1i = fmaf(t1, vi, a1i); }
        for (int r = 44; r < 50; ++r) { const float t = tp[r - 6]; a1r = fmaf(t, row_r[r], a1r); a1i = fmaf(t, row_i[r], a1i); }
        acc0 += a0r; acc1 += a0i; acc2 += a1r; acc3 += a1i;
        __syncthreads();
    }
    out[blockIdx.x * 256 + tid] = acc0 + acc1 + acc2 + acc3;
}

extern "C" __global__ void final_mix(const float2* __restrict__ seed, float* __restrict__ out, int iters)
{
    // 128 threads, each a 4 x 2 pixel block, tables A2 [32][79], B2 [32][79] in shared memory, as in final_tile
    __shared__ float2 A2[32 * 79], B2[32 * 79];
    const int tid = threadIdx.x, tx = tid % 8, ty = tid / 8;
    for (int i = tid; i < 32 * 79; i += 128) { A2[i] = seed[i % 1024]; B2[i] = seed[(i * 3) % 1024]; }
    __syncthreads();
    float acc_r[4][2], acc_i[4][2];
    for (int u = 0; u < 4; ++u) { acc_r[u][0] = acc_r[u][1] = acc_i[u][0] = acc_i[u][1] = 0.f; }
    for (int it = 0; it < iters; ++it) {
        const float2* a0 = A2 + (4 * tx) * 79; const float2* b0 = B2 + (2 * ty) * 79;
        #pragma unroll 2
        for (int q = 0; q < 79; ++q) {
            const float2 y0 = b0[q], y1 = b0[79 + q];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float2 x = a0[u * 79 + q];
                acc_r[u][0] = fmaf(x.x, y0.x, acc_r[u][0]); acc_r[u][0] = fmaf(-x.y, y0.y, acc_r[u][0]); acc_i[u][0] = fmaf(x.x, y0.y, acc_i[u][0]); acc_i[u][0] = fmaf(x.y, y0.x, acc_i[u][0]);
                acc_r[u][1] = fmaf(x.x, y1.x, acc_r[u][1]); acc_r[u][1] = fmaf(-x.y, y1.y, acc_r[u][1]); acc_i[u][1] = fmaf(x.x, y1.y, acc_i[u][1]); acc_i[u][1] = fmaf(x.y, y1.x, acc_i[u][1]);
            }
        }
        __syncthreads();
    }
    float s = 0.f; for (int u = 0; u < 4; ++u) s += acc_r[u][0] + acc_r[u][1] + acc_i[u][0] + acc_i[u][1];
    out[blockIdx.x * 128 + tid] = s;
}

extern "C" __global__ void sincos_rate(float* __restrict__ out, int iters)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.f, x = 0.0001f * tid;
    for (int it = 0; it < iters; ++it) { float s, c; sincospif(x, &s, &c); acc += s + c; x += 0.37f; x -= rintf(x); }
    out[tid] = acc;
}

extern "C" __global__ void store_coalesced(float* __restrict__ out, int rows, int cols)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x; const int i = blockIdx.y;
    if (j < cols) out[(size_t)i * cols + j] = 1.0f;
}
extern "C" __global__ void store_strided(float* __restrict__ out, int rows, int cols)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x; const int j = blockIdx.y;   // consecutive threads = consecutive rows
    if (i < rows) out[(size_t)i * cols + j] = 1.0f;
}
'''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='ubench_gpu.json'); a = ap.parse_args()
    mod = cp.RawModule(code=SRC, options=('--std=c++14', '--use_fast_math'))
    res = dict(device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
    dev = cp.cuda.Device()
    nsm = dev.attributes['MultiProcessorCount']
    res['sms'] = nsm

    def timeit(fn, reps=5):
        fn(); cp.cuda.Stream.null.synchronize()
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); fn(); cp.cuda.Stream.null.synchronize(); ts.append(time.perf_counter() - t)
        return min(ts)
    # HBM read + write
    x = cp.random.standard_normal(256 * 1024 * 1024, dtype=cp.float32)
    t = timeit(lambda: cp.multiply(x, 1.0001, out=x))
    res['hbm_copy_GBps'] = 2 * 4 * x.size / t / 1e9
    print(f"HBM read+write: {res['hbm_copy_GBps']:.0f} GB/s", flush=True)
    # L2 read bandwidth: re-read a 16 MB array many times (fits L2 on the L4, 48 MB)
    y = cp.random.standard_normal(4 * 1024 * 1024, dtype=cp.float32); s_ = cp.zeros((), cp.float32)
    t = timeit(lambda: [cp.sum(y) for _ in range(20)])
    res['l2_read_GBps'] = 20 * 4 * y.size / t / 1e9
    print(f"L2 re-read (sum of 16 MB x 20): {res['l2_read_GBps']:.0f} GB/s", flush=True)
    del x, y
    # level kernel mix on resident shared memory
    seed = cp.random.standard_normal(1024, dtype=cp.float32)
    blocks = nsm * 8
    out = cp.zeros(blocks * 256, cp.float32)
    iters = 400
    k = mod.get_function('fir_mix')
    t = timeit(lambda: k((blocks,), (256,), (seed, out, np.int32(iters))))
    elems = blocks * iters * 32 * 141                     # rotated samples
    outs = blocks * iters * 256 * 2                       # filter outputs
    res['fir_mix_Grot_s'] = elems / t / 1e9
    res['fir_mix_Gout_s'] = outs / t / 1e9
    print(f"level mix (resident): {res['fir_mix_Grot_s']:.1f} G rotated samples/s, {res['fir_mix_Gout_s']:.1f} G filter outputs/s", flush=True)
    # final kernel mix
    seed2 = cp.random.standard_normal(2048, dtype=cp.float32).view(cp.complex64).view(cp.float32).reshape(-1, 2)
    seed2 = cp.ascontiguousarray(seed2).view(cp.float32).reshape(-1)
    k = mod.get_function('final_mix')
    out2 = cp.zeros(blocks * 128, cp.float32)
    iters2 = 200
    t = timeit(lambda: k((blocks,), (128,), (seed2, out2, np.int32(iters2))))
    fmas = blocks * iters2 * 128 * 79 * 32
    res['final_mix_TFMA_s'] = fmas / t / 1e12
    print(f"final mix (resident): {res['final_mix_TFMA_s']:.1f} T FMA/s ({2 * res['final_mix_TFMA_s']:.1f} TFLOP/s)", flush=True)
    # sincospif throughput
    k = mod.get_function('sincos_rate')
    out3 = cp.zeros(blocks * 256, cp.float32)
    t = timeit(lambda: k((blocks,), (256,), (out3, np.int32(2000))))
    res['sincospif_G_s'] = blocks * 256 * 2000 / t / 1e9
    print(f"sincospif: {res['sincospif_G_s']:.0f} G pairs/s", flush=True)
    # coalesced vs strided stores of a [15186, 2408] float32 array
    rows, cols = 15186, 2408
    o = cp.zeros((rows, cols), cp.float32)
    kc, ks = mod.get_function('store_coalesced'), mod.get_function('store_strided')
    t = timeit(lambda: kc(((cols + 255) // 256, rows), (256,), (o, np.int32(rows), np.int32(cols))))
    res['store_coalesced_GBps'] = 4 * rows * cols / t / 1e9
    t = timeit(lambda: ks(((rows + 255) // 256, cols), (256,), (o, np.int32(rows), np.int32(cols))))
    res['store_strided_GBps'] = 4 * rows * cols / t / 1e9
    print(f"stores: coalesced {res['store_coalesced_GBps']:.0f} GB/s, strided {res['store_strided_GBps']:.0f} GB/s", flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
