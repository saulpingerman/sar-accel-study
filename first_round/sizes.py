#!/usr/bin/env python3
"""Image-size series on the measured data.

One Umbra collection is formed at several image sizes, at each precision
setting, and every image is compared with float64 backprojection of the same
phase history on the same grid.

  form     form images at the settings available on this device; one .npy per
           image in --outdir, with timings in timing.json
  analyze  error and coherence of every image in the test directories against
           the float64 reference image

  python sizes.py form --data sz4096.npz --n 4096 --spacing 0.25 --outdir out/sz4096_cpu --ref --x64 \
      --ffbp-policies fp64,fp32,f16,bf16_mm,bf16,f8_mm,f4_mm
  python sizes.py analyze --ref out/sz4096_cpu --test out/sz4096_cpu,out/sz4096_tpu-v5e --win 9 --out results/sizes/sz4096.json
"""
import argparse
import glob
import json
import os
import time

import numpy as np


def form(a):
    import jax
    jax.config.update('jax_enable_x64', a.x64)
    import jax.numpy as jnp
    from sarbench import sim, bp, ffbp
    import realdata
    os.makedirs(a.outdir, exist_ok=True)
    col, S = realdata.load(a.data)
    S *= sim.taylor_2d(col.Np, col.K)
    if not (a.ref or a.x64):
        S = S.astype(np.complex64)                 # host memory: the float32 settings never need the float64 copy
    K = col.K
    nfft = 1 << int(np.ceil(np.log2(a.oversample * K)))
    dr = bp.range_bin(col.df, nfft)
    px, py, pz = sim.ground_grid(a.n, a.spacing)
    tpath = f'{a.outdir}/timing.json'
    timing = json.load(open(tpath)) if os.path.exists(tpath) else {}
    timing['_meta'] = dict(n=a.n, spacing=a.spacing, pulses=int(col.Np), samples=int(K), nfft=nfft, device=str(jax.devices()[0]), label=a.label)

    def done(tag, img, **t):
        np.save(f"{a.outdir}/{tag.replace('/', '_')}.npy", img)
        timing[tag] = t
        json.dump(timing, open(tpath, 'w'), indent=1)
        print(tag, 'ok', {k: round(v, 3) for k, v in t.items()}, flush=True)

    if a.ref:
        from sarbench import cpu_ref
        u, r0 = bp.host_geometry(col.ant)
        t = time.perf_counter()
        rc = cpu_ref.range_compress(S.astype(np.complex128), nfft)
        img = cpu_ref.bp(rc, u, r0, px, py, pz, dr, col.fref).reshape(a.n, a.n)
        del rc
        done('bp/numba_fp64', img, run_s=time.perf_counter() - t)
        del img

    for pol in [p for p in a.bp_policies.split(',') if p]:
        try:
            g = jnp.dtype(bp.POLICIES[pol]['geom'])
            h = np.float64 if g == jnp.float64 else np.float32
            pix = tuple(jnp.asarray(v.astype(h)) for v in (px, py, pz))
            for chunk, tag in [(a.chunk, pol)] + ([(1, pol + '_seq')] if pol.endswith('_acc') else []):
                fn = bp.make_bp(pol, nfft, dr, col.fref, chunk)
                args = bp.prepare(pol, S, col.ant, nfft, chunk)
                t = time.perf_counter()
                re, im = fn(*args, *pix)
                host = re.dtype if re.dtype in (jnp.float32, jnp.float64) else jnp.float32
                img = (np.asarray(re.astype(host)) + 1j * np.asarray(im.astype(host))).reshape(a.n, a.n).astype(np.complex64)
                done(f'bp/{tag}', img, first_s=time.perf_counter() - t)
        except Exception as e:
            print('bp', pol, 'FAILED', type(e).__name__, str(e)[:300], flush=True)

    if a.cuda:
        import cupy as cp
        from sarbench import gpu_ref
        u, r0 = bp.host_geometry(col.ant)
        for st in a.cuda.split(','):
            try:
                t = time.perf_counter()
                Sd = cp.asarray(S.astype(np.complex64))
                rc = gpu_ref.range_compress(Sd, nfft)
                del Sd
                cp.get_default_memory_pool().free_all_blocks()
                re, im = gpu_ref.bp(gpu_ref.pack(rc, st), cp.asarray(u.astype(np.float32)), cp.asarray(r0.astype(np.float32)),
                                    *(cp.asarray(v.astype(np.float32)) for v in (px, py, pz)), nfft, dr, col.fref, st)
                img = (cp.asnumpy(re) + 1j * cp.asnumpy(im)).reshape(a.n, a.n).astype(np.complex64)
                del rc, re, im
                cp.get_default_memory_pool().free_all_blocks()
                done(f'bp/cuda_{st}', img, run_s=time.perf_counter() - t)
            except Exception as e:
                Sd = rc = re = im = None           # drop the device arrays before the pool can release them
                cp.get_default_memory_pool().free_all_blocks()
                print('cuda', st, 'FAILED', type(e).__name__, str(e)[:300], flush=True)

    pols = [p for p in a.ffbp_policies.split(',') if p and (a.x64 or not ffbp.needs_x64(p))]
    if pols:
        levels = ffbp.default_levels(a.n, K, a.T, a.levels)
        t = time.perf_counter()
        plan = ffbp.make_plan(col, a.n, a.spacing, levels, a.T)
        plan_s = time.perf_counter() - t
        coll = ffbp.collection_arrays(plan, col.ant)
        print('ffbp levels', levels, 'passbands', [(round(l['pass_k'], 2), round(l['pass_p'], 2)) for l in plan['levels']], f'plan {plan_s:.1f}s', flush=True)
        timing['_meta'].update(levels=str(levels), plan_s=plan_s, groups=a.groups)
        scale = np.abs(S).max()
        for pol in pols:
            try:
                fn = ffbp.make_ffbp(pol, plan, a.budget, groups=a.groups)
                hre, him = ffbp.prepare(pol, S)
                arrs = ffbp.device_arrays(pol, plan, coll)
                t = time.perf_counter()
                re, im = fn(hre, him, arrs)
                jax.block_until_ready(re)
                first = time.perf_counter() - t
                runs = []
                for _ in range(a.reps):
                    t = time.perf_counter()
                    re, im = fn(hre, him, arrs)
                    jax.block_until_ready(re)
                    runs.append(time.perf_counter() - t)
                img = (np.asarray(re) + 1j * np.asarray(im)) * scale
                del re, im, hre, him, arrs
                done(f'ffbp/{pol}', img if ffbp.needs_x64(pol) else img.astype(np.complex64), first_s=first, **(dict(run_s=min(runs)) if runs else {}))
                del img
            except Exception as e:
                print('ffbp', pol, 'FAILED', type(e).__name__, str(e)[:300], flush=True)


def analyze(a):
    from scipy.ndimage import uniform_filter
    ref = np.load(f'{a.ref}/bp_numba_fp64.npy')
    n = ref.shape[0]
    pr = uniform_filter(np.abs(ref) ** 2, a.win)
    gain_dir = a.gain or a.ref
    gain = None
    for cand in ('ffbp_fp64', 'ffbp_fp32'):
        p = f'{gain_dir}/{cand}.npy'
        if os.path.exists(p) and gain is None:
            src = np.load(p)
            # float64 accumulation: a single-precision sum over 10^8 pixels is off by about 1%
            src = src.astype(np.complex128)
            gain = complex(np.vdot(src, ref) / np.vdot(src, src).real)
            del src
    rows = []
    figs = a.figs
    if figs:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(figs, exist_ok=True)
        pk = np.percentile(np.abs(ref[::4, ::4]), 99.9)
        dn = max(1, n // 1024)

        def save(name, arr, lo, hi, cmap='gray'):
            m = (arr.shape[0] // dn) * dn
            arr = arr[:m, :m].reshape(m // dn, dn, m // dn, dn).mean((1, 3))
            plt.imsave(f'{figs}/{name}.png', np.clip(arr, lo, hi).T[::-1], vmin=lo, vmax=hi, cmap=cmap)

        save(f'sz{n}_img_ref', 20 * np.log10(np.abs(ref) / pk + 1e-12), -50, 0)
    for d in a.test.split(','):
        meta = json.load(open(f'{d}/timing.json')) if os.path.exists(f'{d}/timing.json') else {}
        label = meta.get('_meta', {}).get('label', os.path.basename(d))
        for path in sorted(glob.glob(f'{d}/*.npy')):
            name = os.path.basename(path)[:-4]
            if name == 'bp_numba_fp64':
                continue
            tag = name.replace('_', '/', 1)
            t = np.load(path)
            if not np.isfinite(t).all():
                rows.append(dict(label=label, tag=tag, nonfinite=True))
                continue
            if tag.startswith('ffbp'):
                t = t * gain
            x = t * np.conj(ref)
            num = uniform_filter(x.real, a.win) + 1j * uniform_filter(x.imag, a.win)
            del x
            c = np.abs(num) / np.sqrt(np.maximum(uniform_filter(np.abs(t) ** 2, a.win) * pr, 1e-300))
            del num
            row = dict(label=label, tag=tag, n=n, err_db=float(10 * np.log10(np.sum(np.abs(t - ref) ** 2) / np.sum(np.abs(ref) ** 2) + 1e-300)),
                       coh_mean=float(c.mean()), coh_p01=float(np.percentile(c[::2, ::2], 1)), coh_min=float(c.min()),
                       below_099=float((c < 0.99).mean()), timing=meta.get(tag, {}))
            rows.append(row)
            print(f"n={n} {label:10s} {tag:20s} err {row['err_db']:7.1f} dB  coherence mean {row['coh_mean']:.5f} p01 {row['coh_p01']:.4f} min {row['coh_min']:.3f} "
                  f"below 0.99 {100 * row['below_099']:.2f}%  time {row['timing']}", flush=True)
            if figs and a.fig_tags and tag in a.fig_tags.split(','):
                save(f"sz{n}_coh_{label}_{name}", c, 0, 1)
                save(f"sz{n}_loss_{label}_{name}", np.log10(1.0 - np.minimum(c, 1.0) + 1e-7), -5, 0, 'magma')
            del t, c
    json.dump(dict(n=n, win=a.win, rows=rows), open(a.out, 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('form')
    f.add_argument('--data', required=True)
    f.add_argument('--n', type=int, required=True)
    f.add_argument('--spacing', type=float, default=0.25)
    f.add_argument('--outdir', required=True)
    f.add_argument('--label', default='cpu')
    f.add_argument('--ref', action='store_true', help='float64 backprojection (Numba)')
    f.add_argument('--bp-policies', default='')
    f.add_argument('--ffbp-policies', default='')
    f.add_argument('--cuda', default='')
    f.add_argument('--groups', type=int, default=0)
    f.add_argument('--T', type=int, default=32)
    f.add_argument('--levels', type=int, default=3)
    f.add_argument('--chunk', type=int, default=8)
    f.add_argument('--budget', type=int, default=1 << 26)
    f.add_argument('--oversample', type=int, default=8)
    f.add_argument('--reps', type=int, default=2)
    f.add_argument('--x64', action='store_true')
    z = sub.add_parser('analyze')
    z.add_argument('--ref', required=True)
    z.add_argument('--test', required=True)
    z.add_argument('--gain', help='directory holding the factorized float64 or float32 image used for the constant factor')
    z.add_argument('--win', type=int, default=9)
    z.add_argument('--out', required=True)
    z.add_argument('--figs')
    z.add_argument('--fig-tags', default='')
    a = ap.parse_args()
    dict(form=form, analyze=analyze)[a.cmd](a)


if __name__ == '__main__':
    main()
