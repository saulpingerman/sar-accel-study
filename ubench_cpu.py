"""Achievable unit rates of the CPU for the operations the C++ factorized kernels use, for the bounds of the kernels
appendix.   python ubench_cpu.py --out ubench_cpu.json

Measures, with the same compiler flags and OpenMP as the kernels: memory read+write bandwidth; the frequency-filter
mix (one tap pass over a cache-resident output row, as rot_fir_k's inner loop); the pulse-filter mix (weighted row
sums on resident rows); the final-product mix (the four-row complex product on resident tables); the rotation
recurrence (complex multiply per sample on resident data); sincos throughput.
"""
import argparse, ctypes, hashlib, json, os, subprocess, time

import numpy as np

SRC = r'''
#include <cmath>
#include <cstring>
#include <vector>
#include <omp.h>
extern "C" {
double mem_copy(float* a, const float* b, long n, int reps) {
    double t0 = omp_get_wtime();
    for (int r = 0; r < reps; ++r) {
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; ++i) a[i] = b[i] * 1.0001f + 1.0f;
    }
    return omp_get_wtime() - t0;
}
// one FIR tap pass: o[j] += t * a[j] over Ko outputs, two planes, repeated as the kernel does (L taps, D polyphase branches)
double fir_mix(int Ko, int L, int iters, float* sink) {
    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        std::vector<float> ur(Ko + 64, 0.5f), ui(Ko + 64, 0.25f), o1(Ko, 0.f), o2(Ko, 0.f);
        for (int it = 0; it < iters; ++it) {
            for (int r = 0; r < L; ++r) {
                const float t = 0.01f * (r + 1);
                const float* a = ur.data() + (r % 8);
                const float* b = ui.data() + (r % 8);
                float* p1 = o1.data(); float* p2 = o2.data();
                #pragma omp simd
                for (int j = 0; j < Ko; ++j) { p1[j] += t * a[j]; p2[j] += t * b[j]; }
            }
            ur[it % Ko] += 1e-6f;
        }
        sink[omp_get_thread_num()] = o1[Ko / 2] + o2[Ko / 3];
    }
    return omp_get_wtime() - t0;
}
// the final product exactly as the kernel runs it: four output rows (two vectors wide) against a chunk of N inner
// terms with tables resident in the second-level cache, named accumulators, complex products as FMA pairs
typedef float v16 __attribute__((vector_size(64)));
static inline v16 bc(float x) { return v16{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x}; }
double final_mix(int T, int N, int iters, float* sink) {
    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        std::vector<v16> Ar((size_t)N * 2), Ai((size_t)N * 2), Br((size_t)N * 2), Bi((size_t)N * 2);
        for (size_t k = 0; k < Ar.size(); ++k) { Ar[k] = bc(0.3f + 1e-4f * (k % 7)); Ai[k] = bc(0.2f); Br[k] = bc(0.1f); Bi[k] = bc(0.7f - 1e-4f * (k % 5)); }
        const float* Arf = reinterpret_cast<const float*>(Ar.data()); const float* Aif = reinterpret_cast<const float*>(Ai.data());
        v16 acc = bc(0.f);
        for (int it = 0; it < iters; ++it) {
            for (int i0 = 0; i0 < T; i0 += 4) {
                const v16 z0 = bc(0.f);
                v16 p00 = z0, p01 = z0, p10 = z0, p11 = z0, p20 = z0, p21 = z0, p30 = z0, p31 = z0;
                v16 s00 = z0, s01 = z0, s10 = z0, s11 = z0, s20 = z0, s21 = z0, s30 = z0, s31 = z0;
                for (int n = 0; n < N; ++n) {
                    const v16 br0 = Br[(size_t)n * 2], br1 = Br[(size_t)n * 2 + 1], bi0 = Bi[(size_t)n * 2], bi1 = Bi[(size_t)n * 2 + 1];
                    const float* arow = Arf + (size_t)n * T + i0; const float* airow = Aif + (size_t)n * T + i0;
                    v16 xr, xi;
                    xr = bc(arow[0]); xi = bc(airow[0]); p00 += xr * br0; p00 -= xi * bi0; s00 += xr * bi0; s00 += xi * br0; p01 += xr * br1; p01 -= xi * bi1; s01 += xr * bi1; s01 += xi * br1;
                    xr = bc(arow[1]); xi = bc(airow[1]); p10 += xr * br0; p10 -= xi * bi0; s10 += xr * bi0; s10 += xi * br0; p11 += xr * br1; p11 -= xi * bi1; s11 += xr * bi1; s11 += xi * br1;
                    xr = bc(arow[2]); xi = bc(airow[2]); p20 += xr * br0; p20 -= xi * bi0; s20 += xr * bi0; s20 += xi * br0; p21 += xr * br1; p21 -= xi * bi1; s21 += xr * bi1; s21 += xi * br1;
                    xr = bc(arow[3]); xi = bc(airow[3]); p30 += xr * br0; p30 -= xi * bi0; s30 += xr * bi0; s30 += xi * br0; p31 += xr * br1; p31 -= xi * bi1; s31 += xr * bi1; s31 += xi * br1;
                }
                acc += p00 + p01 + p10 + p11 + p20 + p21 + p30 + p31 + s00 + s01 + s10 + s11 + s20 + s21 + s30 + s31;
            }
            Ar[it % Ar.size()] += bc(1e-6f);
        }
        sink[omp_get_thread_num()] = acc[0] + acc[5];
    }
    return omp_get_wtime() - t0;
}
// rotation recurrence over 16 lanes on resident data
double rot_mix(int n, int iters, float* sink) {
    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        std::vector<float> xr(n, 0.5f), xi(n, 0.25f), ur(n), ui(n);
        float wr[16], wi[16]; for (int l = 0; l < 16; ++l) { wr[l] = 1.f; wi[l] = 0.f; }
        const float sr = 0.001f, sc = 0.9999995f;
        for (int it = 0; it < iters; ++it) {
            for (int tb = 0; tb < n; tb += 16) {
                #pragma omp simd
                for (int l = 0; l < 16; ++l) {
                    const float vr = xr[tb + l], vi = xi[tb + l];
                    ur[tb + l] = vr * wr[l] - vi * wi[l]; ui[tb + l] = vr * wi[l] + vi * wr[l];
                    const float nr = wr[l] * sc - wi[l] * sr; wi[l] = wr[l] * sr + wi[l] * sc; wr[l] = nr;
                }
            }
            xr[it % n] += 1e-6f;
        }
        sink[omp_get_thread_num()] = ur[1] + ui[2];
    }
    return omp_get_wtime() - t0;
}
double sincos_rate(long n, float* sink) {
    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        double acc = 0.0;
        #pragma omp for schedule(static)
        for (long i = 0; i < n; ++i) { double s, c; sincos(1e-3 * (double)i, &s, &c); acc += s + c; }
        sink[omp_get_thread_num()] = (float)acc;
    }
    return omp_get_wtime() - t0;
}
int nthreads(void) { return omp_get_max_threads(); }
}
'''


def build():
    d = os.path.join(os.path.expanduser('~'), '.cache', 'sarbench')
    os.makedirs(d, exist_ok=True)
    tag = hashlib.sha1(SRC.encode()).hexdigest()[:12]
    so = os.path.join(d, f'libubench_cpu_{tag}.so')
    if not os.path.exists(so):
        src = os.path.join(d, f'ubench_cpu_{tag}.cpp')
        open(src, 'w').write(SRC)
        flags = os.environ.get('FFBP_CPU_FLAGS', '-O3 -march=native -mprefer-vector-width=512 -funroll-loops')
        subprocess.run([os.environ.get('CXX', 'g++')] + flags.split() + ['-fopenmp', '-shared', '-fPIC', '-std=c++17', src, '-o', so], check=True)
    L = ctypes.CDLL(so)
    f32 = np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')
    L.mem_copy.restype = ctypes.c_double; L.mem_copy.argtypes = [f32, f32, ctypes.c_long, ctypes.c_int]
    L.fir_mix.restype = ctypes.c_double; L.fir_mix.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, f32]
    L.final_mix.restype = ctypes.c_double; L.final_mix.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, f32]
    L.rot_mix.restype = ctypes.c_double; L.rot_mix.argtypes = [ctypes.c_int, ctypes.c_int, f32]
    L.sincos_rate.restype = ctypes.c_double; L.sincos_rate.argtypes = [ctypes.c_long, f32]
    L.nthreads.restype = ctypes.c_int
    return L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='ubench_cpu.json'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    L = build()
    nt = L.nthreads()
    res = dict(threads=nt, cpu=open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].split(':', 1)[1].strip() if 'model name' in open('/proc/cpuinfo').read() else '')
    sink = np.zeros(max(nt, 1) * 16, np.float32)
    sc = 64 if a.smoke else 1
    # memory: read + write 2 GB
    n = (512 * 1024 * 1024) // sc
    x = np.ones(n, np.float32); y = np.empty(n, np.float32)
    L.mem_copy(y, x, n, 1)
    t = min(L.mem_copy(y, x, n, 1) for _ in range(3))
    res['hbm_copy_GBps'] = 8.0 * n / t / 1e9
    print(f"memory read+write: {res['hbm_copy_GBps']:.0f} GB/s", flush=True)
    del x, y
    # frequency filter mix: Ko = 2408 outputs, 44 taps
    Ko, Lt, iters = 2408, 44, 2000 // sc
    L.fir_mix(Ko, Lt, 10, sink)
    t = min(L.fir_mix(Ko, Lt, iters, sink) for _ in range(3))
    res['fir_mix_Gout_s'] = nt * iters * Ko / t / 1e9                      # filter outputs (complex) per second, all threads
    res['fir_mix_GFMA_s'] = nt * iters * Ko * Lt * 2 / t / 1e9
    print(f"frequency filter mix (resident): {res['fir_mix_Gout_s']:.1f} G outputs/s, {res['fir_mix_GFMA_s']:.0f} GFMA/s", flush=True)
    # final product mix: T = 32, N = 6320
    T, N, it2 = 32, 237, 10000 // sc                                 # one chunk of the kernel (3 pulses of 79 samples), repeated
    L.final_mix(T, N, 4, sink)
    t = min(L.final_mix(T, N, it2, sink) for _ in range(3))
    res['final_mix_GFMA_s'] = nt * it2 * float(T) * T * N * 4 / t / 1e9
    print(f"final product mix (resident): {res['final_mix_GFMA_s']:.0f} GFMA/s", flush=True)
    # rotation recurrence
    nr, it3 = 16384, 2000 // sc
    L.rot_mix(nr, 10, sink)
    t = min(L.rot_mix(nr, it3, sink) for _ in range(3))
    res['rot_mix_Gelem_s'] = nt * it3 * nr / t / 1e9
    print(f"rotation recurrence (resident): {res['rot_mix_Gelem_s']:.1f} G samples/s", flush=True)
    # sincos
    ns = (200_000_000 if not a.smoke else 2_000_000)
    t = L.sincos_rate(ns, sink)
    res['sincos_G_s'] = ns / t / 1e9
    print(f"sincos (double): {res['sincos_G_s']:.2f} G pairs/s", flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
