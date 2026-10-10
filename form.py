#!/usr/bin/env python3
"""Second round: form images of one prepared Umbra collection on its native grid and time them.

  ref     float64 exact backprojection on the CPU (the reference image)
  ffbp    factorized backprojection on the current JAX device, for the listed precision policies and
          filter forms; device time and, with --stream, sustained throughput
  cuda    exact backprojection on the GPU with tile-referenced geometry

Every image is written as <outdir>/<tag>.npy (complex64; the reference complex128), and the timings to
<outdir>/timing.json. --crop nx,ny forms a centred sub-image, for tests.

  python form.py ref  --data scene.npz --outdir out/ref
  python form.py ffbp --data scene.npz --outdir out/tpu --label tpu-v5e --policies fp32_fast,fp32_high --filters dense,conv --stream 6
  python form.py cuda --data scene.npz --outdir out/l4 --label gpu-l4 --stores fp32,f16 --stream 4
"""
import argparse
import json
import os
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class Monitor:
    """Samples host CPU utilisation, and GPU power when nvidia-smi is present, while a timed section runs."""

    def __init__(self, gpu=False):
        self.gpu, self.stop, self.util, self.watts = gpu, False, [], []

    @staticmethod
    def _stat():
        v = [float(x) for x in open('/proc/stat').readline().split()[1:8]]
        return sum(v), v[3] + v[4]

    def _run(self):
        tot0, idle0 = self._stat()
        while not self.stop:
            time.sleep(0.5)
            tot, idle = self._stat()
            if tot > tot0:
                self.util.append(1.0 - (idle - idle0) / (tot - tot0))
            tot0, idle0 = tot, idle
            if self.gpu:
                try:
                    o = subprocess.run(['nvidia-smi', '--query-gpu=power.draw,utilization.gpu', '--format=csv,noheader,nounits'],
                                       capture_output=True, text=True, timeout=5).stdout.strip().split(',')
                    self.watts.append((float(o[0]), float(o[1])))
                except Exception:
                    pass

    def __enter__(self):
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self.stop = True
        self.th.join()

    def result(self):
        out = dict(cpu_util=float(np.mean(self.util)) if self.util else None, ncpu=os.cpu_count())
        if self.watts:
            w = np.array(self.watts)
            out.update(gpu_watts_mean=float(w[:, 0].mean()), gpu_watts_max=float(w[:, 0].max()), gpu_util_mean=float(w[:, 1].mean()))
        return out


def drive(submit, n_min, min_seconds, depth):
    """Pipelined loop: `submit(i)` starts image i and returns a function that fetches it to the host."""
    pool = ThreadPoolExecutor(2)
    inflight = deque()
    pool.submit(submit(0)).result()                         # warm the pipeline
    t0 = time.perf_counter()
    n = 0
    while n < n_min or time.perf_counter() - t0 < min_seconds:
        inflight.append(pool.submit(submit(n)))
        n += 1
        if len(inflight) >= depth:
            inflight.popleft().result()
    while inflight:
        inflight.popleft().result()
    el = time.perf_counter() - t0
    pool.shutdown()
    return dict(images=n, seconds=el, images_per_s=n / el, s_per_image=el / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['ref', 'ffbp', 'cuda', 'pfa', 'ffbpcuda', 'ffbpcpu'])
    ap.add_argument('--resample', default='dense', help='pfa: pulse resampling as dense matmul or taps gather')
    ap.add_argument('--orient', default='1,-1', help='pfa: exponent signs along y (range) and x (azimuth)')
    ap.add_argument('--correct', action='store_true', help='pfa: resample the image at the apparent positions of the planar-wavefront model')
    ap.add_argument('--oversample', type=int, default=2, help='pfa correction: oversampling of the image before resampling')
    ap.add_argument('--taps', type=int, default=6, help='pfa correction: taps per axis of the resampling kernel')
    ap.add_argument('--ptaps', type=int, default=16, help='pfa: taps of the pulse resampling kernel')
    ap.add_argument('--pkaiser', type=float, default=8.0, help='pfa: Kaiser beta of the pulse resampling kernel')
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--label', default='cpu')
    ap.add_argument('--crop', default='')
    ap.add_argument('--policies', default='fp32')
    ap.add_argument('--filters', default='dense')
    ap.add_argument('--stores', default='fp32')
    ap.add_argument('--T', type=int, default=32)
    ap.add_argument('--levels', type=int, default=3)
    ap.add_argument('--pmax', type=float, default=0.4)
    ap.add_argument('--budget', type=int, default=1 << 26)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--stream', type=int, default=0, help='also measure sustained throughput over at least this many images')
    ap.add_argument('--stream-seconds', type=float, default=45.0)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--x64', action='store_true')
    ap.add_argument('--nosave', action='store_true')
    ap.add_argument('--trig', default='split', choices=['split', 'direct'])
    ap.add_argument('--pallas-pb', type=int, default=128, help='pallas filter: pulses per kernel block')
    ap.add_argument('--pallas-chunk', type=int, default=512, help='pallas filter: lanes per rotation chunk')
    ap.add_argument('--pallas-nc', type=int, default=8, help='pallas2: children per parent load')
    ap.add_argument('--pallas-ng', type=int, default=8, help='pallas2: first-level tiles per group')
    ap.add_argument('--no-pallas-final', action='store_true', help='pallas2: keep the XLA final stage')
    ap.add_argument('--cuda-final', default='fp32', help='ffbpcuda configurations, comma list: fp32 (float32 throughout), f16tc (float32 with the tensor-core float16 final), f16 (float16 storage of the phase history and every intermediate, float32 accumulation, tensor-core final)')
    ap.add_argument('--pallas-final', type=int, default=2, help='pallas2 final stage: 0 XLA, 1 direct-trig kernel, 2 recurrence kernel, 3 recurrence with four tiles per step')
    ap.add_argument('--pallas-gen', type=int, default=3, help='pallas2 level kernel generation: 2 or 3 (3: coarse tables precomputed, pulse decimation fused)')
    ap.add_argument('--pad', type=int, default=0, help='pad pulses and samples with zeros to multiples of this (TPU matrix-unit alignment)')
    ap.add_argument('--tile', type=int, default=32, help='geometry tile of the CUDA kernel, pixels')
    ap.add_argument('--bp-oversample', type=int, default=8, help='cuda: range-profile oversampling')
    ap.add_argument('--chunk', type=int, default=256, help='pulses per chunk in the CUDA kernel')
    a = ap.parse_args()
    import prep
    from dev import sim
    os.makedirs(a.outdir, exist_ok=True)
    col, S, grid = prep.load(a.data)
    nx, ny, spx, spy = grid['nx'], grid['ny'], grid['spx'], grid['spy']
    e1, e2 = np.asarray(grid['e1'], np.float64), np.asarray(grid['e2'], np.float64)
    if a.crop:
        nx, ny = (int(v) for v in a.crop.split(','))
    P, K = S.shape
    from scipy.signal.windows import taylor
    pad_tag = ''
    if a.pad:
        from dev.sim import Collect
        P2, K2 = -(-P // a.pad) * a.pad, -(-K // a.pad) * a.pad
        S2 = np.zeros((P2, K2), S.dtype)
        S2[:P, :K] = S
        ant2 = np.concatenate([col.ant, col.ant[-1] + (np.arange(1, P2 - P + 1)[:, None]) * (col.ant[-1] - col.ant[-2])])
        col = Collect(fmin=col.fmin, df=col.df, K=K2, ant=ant2, res=col.res)
        S, P, K = S2, P2, K2
        pad_tag = f'_pad{a.pad}'
    wp = taylor(P, nbar=4, sll=35.0, norm=False).astype(np.float32)
    wk = taylor(K, nbar=4, sll=35.0, norm=False).astype(np.float32)
    if a.pad:                                  # the window belongs to the real samples only
        P0, K0 = grid['info'].get('vectors', P), grid['info'].get('samples', K)
        wp = np.concatenate([taylor(P0, nbar=4, sll=35.0, norm=False).astype(np.float32), np.zeros(P - P0, np.float32)])
        wk = np.concatenate([taylor(K0, nbar=4, sll=35.0, norm=False).astype(np.float32), np.zeros(K - K0, np.float32)])
    tpath = f'{a.outdir}/timing.json'
    timing = json.load(open(tpath)) if os.path.exists(tpath) else {}
    timing['_meta'] = dict(label=a.label, nx=nx, ny=ny, spx=spx, spy=spy, pulses=int(P), samples=int(K), crop=a.crop, name=grid['info'].get('name', ''),
                           e1=e1.tolist(), e2=e2.tolist())

    def done(tag, img, **t):
        if img is not None and not a.nosave:
            np.save(f"{a.outdir}/{tag.replace('/', '_')}.npy", img)
        timing[tag] = t
        json.dump(timing, open(tpath, 'w'), indent=1)
        print(tag, 'ok', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items() if not isinstance(v, dict)}, flush=True)

    if a.mode == 'ref':
        from dev import cpu_ref, bp
        nfft = 1 << int(np.ceil(np.log2(int(os.environ.get('REF_OVERSAMPLE', '8')) * K)))
        dr = bp.range_bin(col.df, nfft)
        u, r0 = bp.host_geometry(col.ant)
        t = time.perf_counter()
        Sw = S.astype(np.complex128)
        Sw *= wp[:, None].astype(np.float64)
        Sw *= wk[None, :].astype(np.float64)
        rc = cpu_ref.range_compress(Sw, nfft)
        del Sw
        X, Y = np.meshgrid((np.arange(nx) - nx / 2.0) * spx, (np.arange(ny) - ny / 2.0) * spy, indexing='ij')
        pos = X.ravel()[:, None] * e1[None, :] + Y.ravel()[:, None] * e2[None, :]
        del X, Y
        img = cpu_ref.bp(rc, u, r0, np.ascontiguousarray(pos[:, 0]), np.ascontiguousarray(pos[:, 1]), np.ascontiguousarray(pos[:, 2]), dr, col.fref).reshape(nx, ny)
        done('bp/numba_fp64', img, run_s=time.perf_counter() - t)
        return

    if a.mode == 'ffbpcuda':
        # factorized backprojection with the CUDA kernels (float32), timed like the other device paths
        import cupy as cp
        from dev import ffbp2, ffbp_cuda
        t = time.perf_counter()
        plan = ffbp2.make_plan(col, nx, ny, spx, spy, T=a.T, nlev=a.levels, pmax=a.pmax, e1=e1, e2=e2)
        print('plan', [(l['sx'], l['sy'], l['Dk'], l['Dp'], l['Ko'], l['Po']) for l in plan['levels']], f'{time.perf_counter() - t:.1f}s', flush=True)
        wpd, wkd = cp.asarray(wp), cp.asarray(wk)
        for fmode in a.cuda_final.split(','):
          store = 'f16' if fmode == 'f16' else 'fp32'
          fin = 'f16tc' if fmode in ('f16tc', 'f16') else 'fp32'
          tag = f'ffbp/{fmode}_cuda' + pad_tag
          try:
              def host():
                  return ffbp2.collection_arrays(plan, col.ant)

              def run_one(coll, Sd=None, fin=fin, store=store):
                  """Sd: the weighted phase history on the device; uploaded here when None (the pipelined loop)."""
                  form = ffbp_cuda.make_ffbp_cuda(plan, coll, final_mode=fin, store=store)
                  if Sd is None:
                      Sd = cp.asarray(S) * wpd[:, None] * wkd[None, :]
                  return form(Sd, ng=a.pallas_ng)
              # timed host memory to host memory: the phase history uploaded and the image downloaded inside the clock
              t = time.perf_counter()
              coll = host()
              out = cp.asnumpy(run_one(coll))
              first = time.perf_counter() - t
              runs, hosts = [], []
              with Monitor(gpu=True) as mon:
                  for _ in range(a.reps):
                      del out
                      cp.get_default_memory_pool().free_all_blocks()
                      t = time.perf_counter()
                      coll = host()
                      hosts.append(time.perf_counter() - t)
                      out = cp.asnumpy(run_one(coll))
                      runs.append(time.perf_counter() - t)
              img = np.asarray(out).astype(np.complex64)
              del out
              cp.get_default_memory_pool().free_all_blocks()
              rec = dict(first_s=first, run_s=min(runs) if runs else first, host_s=min(hosts) if hosts else None, filt='cuda', monitor=mon.result())
              if a.stream:
                  def submit(i):
                      def work():
                          o = run_one(coll)
                          res = cp.asnumpy(o)
                          del o
                          cp.get_default_memory_pool().free_all_blocks()
                          return res
                      return work
                  with Monitor(gpu=True) as mon:
                      rec['stream'] = drive(submit, a.stream, a.stream_seconds, 1)
                  rec['stream']['monitor'] = mon.result()
              done(tag, img, **rec)
              del img
          except Exception as e:
              cp.get_default_memory_pool().free_all_blocks()
              print(tag, 'FAILED', type(e).__name__, str(e)[:300] + (' ... ' + str(e)[-1500:] if len(str(e)) > 300 else ''), flush=True)
        return

    if a.mode == 'ffbpcpu':
        # factorized backprojection with the C++/OpenMP kernels (float32), timed like the other paths; the stream is
        # sequential, one image after another, since the kernels already occupy every core
        from dev import ffbp2, ffbp_cpu
        t = time.perf_counter()
        plan = ffbp2.make_plan(col, nx, ny, spx, spy, T=a.T, nlev=a.levels, pmax=a.pmax, e1=e1, e2=e2)
        print('plan', [(l['sx'], l['sy'], l['Dk'], l['Dp'], l['Ko'], l['Po']) for l in plan['levels']], f'{time.perf_counter() - t:.1f}s', 'threads', ffbp_cpu.lib().ffbp_cpu_threads(), flush=True)
        tag = 'ffbp/fp32_cpp' + pad_tag
        try:
            Sw = (S * wp[:, None] * wk[None, :]).astype(np.complex64)

            def run_one(coll):
                return ffbp_cpu.make_ffbp_cpu(plan, coll)(Sw, ng=a.pallas_ng)
            t = time.perf_counter()
            coll = ffbp2.collection_arrays(plan, col.ant)
            out = run_one(coll)
            first = time.perf_counter() - t
            runs, hosts = [], []
            ffbp_cpu.PROFILE.clear(); ffbp_cpu.PROFILE['on'] = True
            with Monitor() as mon:
                for _ in range(a.reps):
                    t = time.perf_counter()
                    coll = ffbp2.collection_arrays(plan, col.ant)
                    hosts.append(time.perf_counter() - t)
                    out = run_one(coll)
                    runs.append(time.perf_counter() - t)
            prof = {k: v / max(1, a.reps) for k, v in ffbp_cpu.PROFILE.items() if k != 'on'}
            ffbp_cpu.PROFILE['on'] = False
            img = out.astype(np.complex64)
            rec = dict(first_s=first, run_s=min(runs) if runs else first, host_s=min(hosts) if hosts else None, filt='cpp', monitor=mon.result(),
                       stages=prof, threads=ffbp_cpu.lib().ffbp_cpu_threads())
            if a.stream:
                def submit(i):
                    return lambda: run_one(coll)
                with Monitor() as mon:
                    rec['stream'] = drive(submit, a.stream, a.stream_seconds, 1)
                rec['stream']['monitor'] = mon.result()
            done(tag, img, **rec)
        except Exception as e:
            print(tag, 'FAILED', type(e).__name__, str(e)[:300] + (' ... ' + str(e)[-1500:] if len(str(e)) > 300 else ''), flush=True)
        return

    if a.mode == 'cuda':
        import cupy as cp
        from dev import gpu_bp2
        wpd, wkd = cp.asarray(wp), cp.asarray(wk)
        for st in a.stores.split(','):
            def run_one():
                Sd = cp.asarray(S)
                Sd = Sd * wpd[:, None] * wkd[None, :]
                re, im = gpu_bp2.bp2(Sd, col.ant, col.fmin, col.df, nx, ny, spx, spy, store=st, oversample=a.bp_oversample, chunk=a.chunk, tile=a.tile, e1=e1, e2=e2)
                del Sd
                return re, im
            try:
                t = time.perf_counter()                       # host memory to host memory
                re, im = run_one()
                re, im = cp.asnumpy(re), cp.asnumpy(im)
                first = time.perf_counter() - t
                runs = []
                with Monitor(gpu=True) as mon:
                    for _ in range(a.reps):
                        del re, im
                        cp.get_default_memory_pool().free_all_blocks()
                        t = time.perf_counter()
                        re, im = run_one()
                        re, im = cp.asnumpy(re), cp.asnumpy(im)
                        runs.append(time.perf_counter() - t)
                img = (re + 1j * im).astype(np.complex64)
                del re, im
                cp.get_default_memory_pool().free_all_blocks()
                rec = dict(first_s=first, run_s=min(runs) if runs else first, monitor=mon.result())
                if a.stream:
                    def submit(i):
                        def work():
                            re, im = run_one()
                            out = cp.asnumpy(re), cp.asnumpy(im)
                            del re, im
                            cp.get_default_memory_pool().free_all_blocks()
                            return out
                        return work
                    with Monitor(gpu=True) as mon:
                        rec['stream'] = drive(submit, a.stream, a.stream_seconds, 1)
                    rec['stream']['monitor'] = mon.result()
                done(f'bp/cuda_{st}' + ('' if a.bp_oversample == 8 else f'_ov{a.bp_oversample}'), img, **rec)
                del img
            except Exception as e:
                cp.get_default_memory_pool().free_all_blocks()
                print('cuda', st, 'FAILED', type(e).__name__, str(e)[:300] + (' ... ' + str(e)[-1500:] if len(str(e)) > 300 else ''), flush=True)
        return

    import jax
    jax.config.update('jax_enable_x64', a.x64)
    import jax.numpy as jnp
    from dev import ffbp2
    gpu = jax.default_backend() == 'gpu'
    timing['_meta']['device'] = str(jax.devices()[0])
    if a.mode == 'pfa':
        from dev import pfa2
        t = time.perf_counter()
        # polar format forms the whole illuminated scene; a crop is cut from the full native grid afterwards
        fx, fy = grid['nx'] or nx, grid['ny'] or ny
        geo = pfa2.geometry(col, fx, fy, spx, spy, taps=a.ptaps, e1=e1, e2=e2, kaiser=a.pkaiser)
        geo_s = time.perf_counter() - t
        print('pfa grid', {k: geo[k] for k in ('nkr', 'nka', 'nfy', 'nfx', 'oversample_r', 'oversample_a')}, f'{geo_s:.1f}s', flush=True)
        timing['_meta'].update(pfa=dict(nkr=geo['nkr'], nka=geo['nka'], nfy=geo['nfy'], nfx=geo['nfx'], geo_s=geo_s))
        orient = tuple(int(v) for v in a.orient.split(','))
        dist = None
        if a.correct:
            t = time.perf_counter()
            dist = pfa2.distortion(col, fx, fy, spx, spy, e1, e2)
            print('pfa distortion model: max shift (az, rg) px', [round(v, 2) for v in dist['max_shift_px']], 'fit rms px', [round(v, 4) for v in dist['rms_px']], f'{time.perf_counter() - t:.1f}s', flush=True)
            timing['_meta'].setdefault('pfa', {}).update(max_shift_px=list(dist['max_shift_px']), fit_rms_px=list(dist['rms_px']))
        # Backprojection sums the samples with uniform weights, which in the wavenumber plane is a density of
        # 1 / k (polar Jacobian); polar format interpolates the spectrum and weights it uniformly. The frequency
        # samples are weighted by f_c / f_k so that the two algorithms apply the same spectral weighting.
        f_k = col.fmin + np.arange(K) * col.df
        wk_pfa = (wk * (col.fmin + (K // 2) * col.df) / f_k).astype(np.float32)
        wpd, wkd = jnp.asarray(wp), jnp.asarray(wk_pfa)
        for pol in a.policies.split(','):
            mm = jnp.bfloat16 if pol in ('bf16_mm', 'f16') and jax.default_backend() != 'gpu' else (jnp.float16 if pol == 'f16' else None)
            prec = {'fp32': jax.lax.Precision.HIGHEST, 'fp32_high': jax.lax.Precision.HIGH}.get(pol)
            tag = f'pfa/{pol}' + ('' if a.resample == 'dense' else f'_{a.resample}') + ('_corr' if a.correct else '') + ('' if a.ptaps == 16 else f'_p{a.ptaps}')
            try:
                # a crop of the native grid is centred like the cropped reference, whose count is even
                off = ((fx / 2.0 - fx // 2) * spx, (fy / 2.0 - fy // 2) * spy) if (fx, fy) != (nx, ny) else (0.0, 0.0)
                cropbox = ((fx - nx) // 2, (fy - ny) // 2, nx, ny) if (fx, fy) != (nx, ny) else None
                fn = pfa2.make_pfa(geo, fx, fy, spx, spy, a.resample, mm, prec, orient, off, dist=dist, oversample=a.oversample, crop=cropbox, taps=a.taps)
                W, alpha, eps_r, shift, eps_a = pfa2.arrays(geo, P, a.resample)

                # the collection is moved as two float32 planes; complex64 arrays take a slow path to the TPU
                Sr_h, Si_h = np.ascontiguousarray(S.real), np.ascontiguousarray(S.imag)

                @jax.jit
                def prep(sr, si):
                    w = wpd[:, None] * wkd[None, :]
                    return sr * w, si * w

                def run_one():
                    re, im = prep(jnp.asarray(Sr_h), jnp.asarray(Si_h))
                    return fn(re, im, W, alpha, eps_r, shift, eps_a)

                planes = jax.jit(lambda z: jnp.stack([z.real, z.imag], axis=-1))   # complex64 leaves a TPU slowly; two planes do not

                def to_host(z):
                    return np.ascontiguousarray(np.asarray(planes(z))).view(np.complex64)[..., 0]

                t = time.perf_counter()                       # host memory to host memory
                out = to_host(run_one())
                first = time.perf_counter() - t
                runs = []
                with Monitor(gpu=gpu) as mon:
                    for _ in range(a.reps):
                        del out
                        t = time.perf_counter()
                        out = to_host(run_one())
                        runs.append(time.perf_counter() - t)
                out = out.astype(np.complex64)
                if (fx, fy) != (nx, ny) and not a.correct:
                    i0, j0 = (fx - nx) // 2, (fy - ny) // 2
                    out = out[i0:i0 + nx, j0:j0 + ny]
                rec = dict(first_s=first, run_s=min(runs) if runs else first, resample=a.resample, monitor=mon.result())
                if a.stream:
                    def submit(i):
                        y = planes(run_one())
                        return lambda: np.ascontiguousarray(np.asarray(y)).view(np.complex64)[..., 0]
                    with Monitor(gpu=gpu) as mon:
                        rec['stream'] = drive(submit, a.stream, a.stream_seconds, a.depth)
                    rec['stream']['monitor'] = mon.result()
                done(tag, out, **rec)
                del out
            except Exception as e:
                print(tag, 'FAILED', type(e).__name__, str(e)[:300] + (' ... ' + str(e)[-1500:] if len(str(e)) > 300 else ''), flush=True)
        return
    t = time.perf_counter()
    plan = ffbp2.make_plan(col, nx, ny, spx, spy, T=a.T, nlev=a.levels, pmax=a.pmax, e1=e1, e2=e2)
    plan_s = time.perf_counter() - t
    desc = [(l['sx'], l['sy'], l['Dk'], l['Dp'], l['Ko'], l['Po']) for l in plan['levels']]
    print('plan', desc, 'padded', plan['Nx'], plan['Ny'], f'{plan_s:.1f}s', flush=True)
    timing['_meta'].update(plan=str(desc), plan_s=plan_s, padded=[plan['Nx'], plan['Ny']], T=a.T, levels=a.levels, pmax=a.pmax)
    Sw = S * wp[:, None] * wk[None, :]
    for filt in a.filters.split(','):
        for pol in [p for p in a.policies.split(',') if p and (a.x64 or not ffbp2.needs_x64(p))]:
            tag = f'ffbp/{pol}' + ('' if filt == 'dense' else f'_{filt}') + ('' if a.trig == 'split' else '_direct') + pad_tag + ({0: '_xlafinal', 1: '_final1', 2: '', 3: '_final3'}[0 if a.no_pallas_final else a.pallas_final] if filt == 'pallas2' else '') + ('_gen2' if filt == 'pallas2' and a.pallas_gen == 2 else '')
            try:
                fn = ffbp2.make_ffbp(pol, plan, filt, a.budget, a.trig, pallas_pb=a.pallas_pb, pallas_chunk=a.pallas_chunk, pallas_nc=a.pallas_nc, pallas_ng=a.pallas_ng, pallas_final=0 if a.no_pallas_final else a.pallas_final, pallas_gen=a.pallas_gen)
                static = ffbp2.static_arrays(pol, plan, filt)

                def host():
                    return ffbp2.device_arrays(pol, plan, ffbp2.collection_arrays(plan, col.ant), static)

                ew = ffbp2.POLICIES[pol]['ew']
                wpd, wkd = jnp.asarray(wp), jnp.asarray(wk)
                Sr_h, Si_h = np.ascontiguousarray(S.real), np.ascontiguousarray(S.imag)

                @jax.jit
                def prep(sr, si):
                    w = wpd[:, None] * wkd[None, :]
                    sr, si = sr * w, si * w
                    s = jnp.sqrt(jnp.max(sr * sr + si * si))
                    return (sr / s).astype(ew), (si / s).astype(ew), s

                @jax.jit
                def finish(re, im, s):
                    return jnp.stack([re * s, im * s], axis=-1)

                def submit(i):
                    """One image from the phase history in host memory; the returned call brings it to host memory."""
                    arrs = host()
                    hre, him, s = prep(jnp.asarray(Sr_h), jnp.asarray(Si_h))
                    re, im = fn(hre, him, arrs)
                    del hre, him
                    y = finish(re, im, s)
                    del re, im
                    return lambda: np.ascontiguousarray(np.asarray(y)).view(np.complex64)[..., 0]

                x64 = ffbp2.needs_x64(pol)
                if x64:                          # float64 policies: the phase history resident (the planes trick is float32)
                    hre, him, scale = ffbp2.prepare(pol, Sw)

                    def once():
                        r_, i_ = fn(hre, him, host())
                        return ((np.asarray(r_) + 1j * np.asarray(i_)) * scale).astype(np.complex128)
                else:
                    def once():                  # host memory to host memory
                        return submit(0)()
                t = time.perf_counter()
                img = once()
                first = time.perf_counter() - t
                runs, hosts = [], []
                with Monitor(gpu=gpu) as mon:
                    for _ in range(a.reps):
                        del img
                        t = time.perf_counter()
                        arrs = host()
                        jax.block_until_ready(arrs['levels'][0]['c0'])
                        hosts.append(time.perf_counter() - t)
                        del arrs
                        t = time.perf_counter()
                        img = once()
                        runs.append(time.perf_counter() - t)
                if x64:
                    del hre, him
                rec = dict(first_s=first, run_s=min(runs) if runs else first, host_s=min(hosts) if hosts else None, filt=filt,
                           resident=x64, monitor=mon.result())
                if a.stream and not x64:
                    out = submit(0)()
                    assert out.shape == (nx, ny), out.shape
                    del out
                    with Monitor(gpu=gpu) as mon:
                        rec['stream'] = drive(submit, a.stream, a.stream_seconds, a.depth)
                    rec['stream']['monitor'] = mon.result()
                done(tag, img, **rec)
                del img, static
            except Exception as e:
                print(tag, 'FAILED', type(e).__name__, str(e)[:300] + (' ... ' + str(e)[-1500:] if len(str(e)) > 300 else ''), flush=True)


if __name__ == '__main__':
    main()
