#!/usr/bin/env python3
"""Image quality at the sizes that were timed.

A noise-free 9 x 9 lattice of point targets spanning the image is formed at
N = 4096 or 8192 with the exact tile plan used for timing, on whatever device
JAX sees, and compared with float64 references computed on the CPU: a direct
matched filter on small chips at nine targets (centre, edges, corners) and
float64 backprojection of the whole image.

  make     simulate the phase history (CPU, float64)
  ref      float64 backprojection image and matched-filter chips (CPU, Numba)
  form     form the image at each policy on this device; keep chips and the
           whole-image error against the reference image
  analyze  per-target error against the matched filter, peak loss, sidelobes
"""
import argparse
import json
import time

import numpy as np

C = 299792458.0
NINE = [(4, 4), (0, 4), (8, 4), (4, 0), (4, 8), (0, 0), (0, 8), (8, 0), (8, 8)]
HALF = 32          # chips kept around every lattice target are 64 x 64
MF = 12            # matched-filter chips are 24 x 24


def geometry(N, r0=10e3):
    from sarbench import sim
    res = 0.3
    scene = N * res / (1.25 * np.sqrt(2.0))
    return sim.make_collect(res=res, scene=scene, r0=r0), scene / N


def range_of(d):
    return float(d['r0']) if 'r0' in d.files else 10e3


def lattice(N, spacing):
    """81 unit targets on a 9 x 9 lattice out to 0.45 of the image side, placed off the pixel grid."""
    g = np.linspace(-0.45, 0.45, 9) * N * spacing
    X, Y = np.meshgrid(g, g, indexing='ij')
    jit = np.random.default_rng(1).uniform(-0.5, 0.5, (9, 9, 2)) * spacing
    pos = np.stack([X + jit[..., 0], Y + jit[..., 1], np.zeros((9, 9))], axis=-1).reshape(-1, 3)
    return pos


def pix(pos, N, spacing):
    return np.rint(pos[:, :2] / spacing + N / 2).astype(int)


def make(a):
    from sarbench import sim
    col, spacing = geometry(a.N, a.r0)
    pos = lattice(a.N, spacing)
    t = time.time()
    S = sim.simulate(col, pos, np.ones(len(pos), np.complex128)) * sim.taylor_2d(col.Np, col.K)
    np.savez(a.out, S=S.astype(np.complex64), N=a.N, pos=pos, spacing=spacing, r0=a.r0)
    print('simulated', S.shape, f'{time.time() - t:.0f}s')


def ref(a):
    from numba import njit, prange
    from sarbench import bp, cpu_ref, sim

    @njit(parallel=True, fastmath=True)
    def matched(Sre, Sim, ant, r0, f0, df, px, py):
        n, P, K = px.size, Sre.shape[0], Sre.shape[1]
        ore, oim = np.zeros(n), np.zeros(n)
        for i in prange(n):
            sr, si = 0.0, 0.0
            for p in range(P):
                dx, dy = px[i] - ant[p, 0], py[i] - ant[p, 1]
                dR = np.sqrt(dx * dx + dy * dy + ant[p, 2] * ant[p, 2]) - r0[p]
                a0, a1 = 4.0 * np.pi * f0 * dR / 299792458.0, 4.0 * np.pi * df * dR / 299792458.0
                cr, ci = np.cos(a0), np.sin(a0)
                wr, wi = np.cos(a1), np.sin(a1)
                for k in range(K):
                    sr += Sre[p, k] * cr - Sim[p, k] * ci
                    si += Sre[p, k] * ci + Sim[p, k] * cr
                    cr, ci = cr * wr - ci * wi, cr * wi + ci * wr
            ore[i], oim[i] = sr, si
        return ore, oim

    d = np.load(a.data)
    N, spacing = int(d['N']), float(d['spacing'])
    col, _ = geometry(N, range_of(d))
    S = d['S'].astype(np.complex128)
    out = {}
    cen = pix(d['pos'], N, spacing).reshape(9, 9, 2)
    r0 = np.linalg.norm(col.ant, axis=1)
    t = time.time()
    for j, (ia, ib) in enumerate(NINE):
        r, c = cen[ia, ib]
        R, Cc = np.meshgrid(r + np.arange(-MF, MF), c + np.arange(-MF, MF), indexing='ij')
        ox, oy = (R.ravel() - N / 2) * spacing, (Cc.ravel() - N / 2) * spacing
        re, im = matched(np.ascontiguousarray(S.real), np.ascontiguousarray(S.imag), col.ant, r0, col.fmin, col.df, ox, oy)
        out[f'mf/{j}'] = (re + 1j * im).reshape(2 * MF, 2 * MF)
        print('matched filter chip', j, f'{time.time() - t:.0f}s', flush=True)
    if not a.no_image:
        px, py, pz = sim.ground_grid(N, spacing)
        nfft = 1 << int(np.ceil(np.log2(8 * col.K)))
        u, r0 = bp.host_geometry(col.ant)
        t = time.time()
        img = cpu_ref.bp(cpu_ref.range_compress(S, nfft), u, r0, px, py, pz, bp.range_bin(col.df, nfft), col.fref).reshape(N, N)
        print(f'float64 backprojection {time.time() - t:.0f}s', flush=True)
        np.save(a.out.replace('.npz', '_img.npy'), img.astype(np.complex64))
        out['chips/bp_fp64'] = chips(img, cen)
    np.savez(a.out, **out)


def chips(img, cen):
    return np.stack([img[r - HALF:r + HALF, c - HALF:c + HALF] for r, c in cen.reshape(-1, 2)]).astype(np.complex64)


def form(a):
    import jax
    jax.config.update('jax_enable_x64', False)
    import jax.numpy as jnp
    from sarbench import bp, ffbp, sim
    d = np.load(a.data)
    N, spacing = int(d['N']), float(d['spacing'])
    col, _ = geometry(N, range_of(d))
    S = d['S']
    cen = pix(d['pos'], N, spacing).reshape(9, 9, 2)
    refimg = np.load(a.ref_image) if a.ref_image else None
    out, info = {}, {}

    def keep(tag, img):
        out[f'chips/{tag}'] = chips(img, cen)
        e = dict(finite=bool(np.isfinite(img).all()))
        if refimg is not None and e['finite']:
            g = np.vdot(img, refimg) / np.sum(np.abs(img).astype(np.float64) ** 2)      # float64 accumulation
            e['err_db'] = float(10 * np.log10(np.sum(np.abs(img * g - refimg) ** 2) / np.sum(np.abs(refimg) ** 2)))
        info[tag] = e
        print(tag, e, flush=True)

    levels = ffbp.default_levels(N, col.K, a.T, a.levels)
    plan = ffbp.make_plan(col, N, spacing, levels, a.T, block=a.block)
    coll = ffbp.collection_arrays(plan, col.ant)
    info['levels'] = str(levels)
    info['block'] = a.block
    for pol in a.policies.split(','):
        try:
            fn = ffbp.make_ffbp(pol, plan)
            scale = np.abs(S).max()
            hre, him = ffbp.prepare(pol, S)
            re, im = fn(hre, him, ffbp.device_arrays(pol, plan, coll))
            keep(f'ffbp/{pol}', (np.asarray(re) + 1j * np.asarray(im)) * scale)
        except Exception as e:
            info[f'ffbp/{pol}'] = dict(error=f'{type(e).__name__}: {str(e)[:300]}')
            print('ffbp', pol, 'FAILED', info[f'ffbp/{pol}'], flush=True)
    if a.cuda:
        import cupy as cp
        from sarbench import gpu_ref
        px, py, pz = (cp.asarray(v.astype(np.float32)) for v in sim.ground_grid(N, spacing))
        nfft = 1 << int(np.ceil(np.log2(8 * col.K)))
        u, r0 = bp.host_geometry(col.ant)
        rc = gpu_ref.range_compress(cp.asarray(S.astype(np.complex64)), nfft)
        for st in ('fp32', 'f16'):
            re, im = gpu_ref.bp(gpu_ref.pack(rc, st), cp.asarray(u.astype(np.float32)), cp.asarray(r0.astype(np.float32)),
                                px, py, pz, nfft, bp.range_bin(col.df, nfft), col.fref, st)
            keep(f'bp/cuda_{st}', (cp.asnumpy(re) + 1j * cp.asnumpy(im)).reshape(N, N))
    np.savez(a.out, info=json.dumps(dict(device=str(jax.devices()[0]), N=N, **info)), **out)
    print('saved', a.out)


def analyze(a):
    from sarbench import metrics as M
    d = np.load(a.data)
    N, spacing = int(d['N']), float(d['spacing'])
    r = np.load(a.ref)
    rows = []
    radius = np.linalg.norm(d['pos'][:, :2], axis=1).reshape(9, 9)
    for path in a.chips.split(','):
        z = np.load(path)
        info = json.loads(str(z['info'])) if 'info' in z.files else {}
        for key in [k for k in z.files if k.startswith('chips/')]:
            tag = key[6:]
            ch = z[key].reshape(9, 9, 2 * HALF, 2 * HALF)
            if not np.isfinite(ch).all():
                rows.append(dict(device=info.get('device', 'cpu'), tag=tag, nonfinite=True))
                continue
            # one complex gain per image, fitted on the nine matched-filter chips together
            t = np.stack([ch[ia, ib][HALF - MF:HALF + MF, HALF - MF:HALF + MF] for ia, ib in NINE])
            m = np.stack([r[f'mf/{j}'] for j in range(9)])
            g = np.vdot(t, m) / np.sum(np.abs(t).astype(np.float64) ** 2)
            per = []
            for j, (ia, ib) in enumerate(NINE):
                e = 10 * np.log10(np.sum(np.abs(t[j] * g - m[j]) ** 2) / np.sum(np.abs(m[j]) ** 2))
                pk = 20 * np.log10(np.abs(t[j] * g).max() / np.abs(m[j]).max())
                per.append(dict(radius_m=float(radius[ia, ib]), err_db=float(e), peak_db=float(pk)))
            pslr = []
            for ia in range(9):
                for ib in range(9):
                    rep = M.irf_report(ch[ia, ib], (HALF, HALF), (spacing, spacing), half=24, up=8)
                    pslr.append(max(rep['ax0']['pslr_db'], rep['ax1']['pslr_db']))
            row = dict(r0=range_of(d), device=info.get('device', 'cpu'), source=path.split('/')[-1].replace('.npz', '').split('_', 1)[-1], tag=tag, N=N, targets=per, worst_err_db=max(p['err_db'] for p in per),
                       centre_err_db=per[0]['err_db'], corner_err_db=float(np.mean([p['err_db'] for p in per[5:]])),
                       peak_db_range=[min(p['peak_db'] for p in per), max(p['peak_db'] for p in per)],
                       pslr_worst_db=float(max(pslr)), pslr_median_db=float(np.median(pslr)),
                       image_err_db=info.get(tag, {}).get('err_db'))
            rows.append(row)
            print(f"{row['device'][:10]:10s} {tag:20s} N={N} matched filter: centre {row['centre_err_db']:6.1f} corners {row['corner_err_db']:6.1f} worst {row['worst_err_db']:6.1f} dB | "
                  f"peak {row['peak_db_range'][0]:+.2f}..{row['peak_db_range'][1]:+.2f} dB | PSLR worst {row['pslr_worst_db']:.1f} | image vs fp64 BP {row['image_err_db']}", flush=True)
    json.dump(rows, open(a.out, 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('make')
    m.add_argument('--N', type=int, required=True)
    m.add_argument('--out', required=True)
    m.add_argument('--r0', type=float, default=10e3)
    r = sub.add_parser('ref')
    r.add_argument('--data', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--no-image', action='store_true')
    f = sub.add_parser('form')
    f.add_argument('--data', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--ref-image')
    f.add_argument('--policies', default='fp32,fp32_high,fp32_fast,bf16_mm,bf16,bf16_mm_l1f32,f16_mm,f16')
    f.add_argument('--T', type=int, default=32)
    f.add_argument('--levels', type=int, default=3)
    f.add_argument('--block', type=int, default=0)
    f.add_argument('--cuda', action='store_true')
    z = sub.add_parser('analyze')
    z.add_argument('--data', required=True)
    z.add_argument('--ref', required=True)
    z.add_argument('--chips', required=True, help='comma-separated chip files (the ref file may be included for bp_fp64)')
    z.add_argument('--out', required=True)
    a = ap.parse_args()
    dict(make=make, ref=ref, form=form, analyze=analyze)[a.cmd](a)


if __name__ == '__main__':
    main()
