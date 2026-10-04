#!/usr/bin/env python3
"""Measured-data test on Umbra open-data CPHD phase histories.

  prep     read a CPHD file, keep a patch around the scene reference point by
           low-pass filtering and decimating both axes (float64, host), and
           save the phase history with the antenna track in a local frame.
  form     form the patch with exact backprojection and the factorized
           algorithm at each precision policy available on the device.
  analyze  compare every image with float64 backprojection, and score the
           repeat-pass coherence map of each policy against the float64 map.

The CPHD signal model (FX domain, SGN = -1) is exp(-j 2 pi f dTOA) with dTOA
the two-way delay relative to the scene reference point, which matches the
convention of sarbench with dR = c dTOA / 2 and the antenna taken at the
midpoint of the transmit and receive positions.
"""
import argparse
import json
import time

import numpy as np

C = 299792458.0


def enu_basis(p):
    """East, north, up unit vectors at ECEF point p (spherical up is adequate for a local image plane)."""
    up = p / np.linalg.norm(p)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    return np.stack([east, np.cross(up, east), up])


def prep(a):
    from sarpy.io.phase_history.converter import open_phase_history
    from sarbench import ffbp
    r = open_phase_history(a.cphd)
    m = r.cphd_meta
    ch = m.Data.Channels[0]
    P, K = ch.NumVectors, ch.NumSamples
    assert m.Global.DomainType == 'FX' and m.Channel.Parameters[0].SRPFixed
    sgn = m.Global.SGN
    tx, rcv, srp = (r.read_pvp_variable(n, 0) for n in ('TxPos', 'RcvPos', 'SRPPos'))
    sc0, scss = r.read_pvp_variable('SC0', 0), r.read_pvp_variable('SCSS', 0)
    f0, df = float(sc0.mean()), float(scss.mean())
    drift = max(np.ptp(sc0), np.ptp(scss) * K)                 # Hz; the grid is treated as fixed
    R = enu_basis(srp[0])
    ant = (0.5 * (tx + rcv) - srp[0]) @ R.T                    # local east, north, up, origin at the reference point
    t = time.time()
    S = r.read_chip((0, P), (0, K), index=0).astype(np.complex128)
    if sgn > 0:
        S = np.conj(S)
    off = np.array([float(v) for v in a.center.split(',')] + [0.0])
    if np.any(off != 0):                                       # move the reference point before filtering (same sign as rereference)
        a2 = ant - off[None, :]
        d = np.linalg.norm(a2, axis=1) - np.linalg.norm(ant, axis=1)
        fk = f0 + df * np.arange(K)
        for i in range(0, P, 1024):
            S[i:i + 1024] *= np.exp(1j * (4.0 * np.pi / C) * d[i:i + 1024, None] * fk[None, :])
        ant = a2
    half = a.patch / np.sqrt(2.0)
    u = ant / np.linalg.norm(ant, axis=1)[:, None]
    du = np.linalg.norm(np.diff(u, axis=0), axis=1).max()
    pass_k = half / (C / (2.0 * df * a.dec) / 2.0)
    pass_p = 2.0 * (2.0 * (f0 + K * df) / C) * du * half * a.dec
    if a.dec > 1:
        Fk, mk = ffbp.decimator(K, a.dec, pass_k)
        Fp, mp = ffbp.decimator(P, a.dec, pass_p)
        S = (Fp.T @ (S @ Fk))
        pidx = a.dec * (np.arange(Fp.shape[1]) - mp) + (a.dec - 1) / 2.0
        ant = ffbp._positions(ant, pidx)
        f0 = f0 + ((a.dec - 1) / 2.0 - mk * a.dec) * df
        df = df * a.dec
    rng = np.linalg.norm(ant, axis=1)
    info = dict(cphd=a.cphd, vectors=int(P), samples=int(K), dec=a.dec, pass_k=float(pass_k), pass_p=float(pass_p),
                grid_drift_hz=float(drift), center=a.center, range_km=[float(rng.min() / 1e3), float(rng.max() / 1e3)],
                graze_deg=float(np.degrees(np.arcsin(ant[len(ant) // 2, 2] / rng[len(ant) // 2]))),
                aperture_deg=float(np.degrees(np.arccos(np.clip(u[0] @ u[-1], -1, 1)))),
                bandwidth_hz=float(df * S.shape[1]), prep_s=time.time() - t)
    print(json.dumps(info))
    np.savez(a.out, S=S, ant=ant, fmin=f0, df=df, info=json.dumps(info))


def load(path):
    from sarbench.sim import Collect
    d = np.load(path)
    S = d['S']
    return Collect(fmin=float(d['fmin']), df=float(d['df']), K=S.shape[1], ant=d['ant'], res=0.5), S


def rereference(col, S, shift):
    """Move the image origin by `shift` (metres, local frame): returns the
    phase history referenced to the new origin and the antenna track seen from it."""
    from sarbench.sim import Collect
    a2 = col.ant - shift[None, :]
    d = np.linalg.norm(a2, axis=1) - np.linalg.norm(col.ant, axis=1)
    # the data carry exp(-jk(|x - a| - |a|)); referencing them to the new origin needs exp(-jk(|x - a| - |a - shift|))
    S2 = S * np.exp(1j * (4.0 * np.pi / C) * d[:, None] * col.freqs[None, :])
    return Collect(col.fmin, col.df, col.K, a2, col.res), S2


def form(a):
    import jax
    jax.config.update('jax_enable_x64', a.x64)
    import jax.numpy as jnp
    from sarbench import sim, bp, ffbp

    out, timing = {}, {}
    shift = np.array([float(v) for v in a.shift.split(',')])
    px, py, pz = sim.ground_grid(a.n, a.spacing)
    passes = {}
    for w, path in zip(('p1', 'p2'), a.data.split(',')):
        col, S = load(path)
        off = np.array([float(v) for v in a.center.split(',')] + [0.0])
        if w == 'p2':
            off = off + np.array([shift[0], shift[1], 0.0])
        if np.any(off != 0):
            col, S = rereference(col, S, off)
        passes[w] = (col, S * sim.taylor_2d(col.Np, col.K))
    for w, (col, S) in passes.items():
        K = col.K
        nfft = 1 << int(np.ceil(np.log2(8 * K)))
        dr = bp.range_bin(col.df, nfft)
        for pol in [p for p in a.policies.split(',') if p and (a.x64 or not bp.needs_x64(p))]:
            g = jnp.dtype(bp.POLICIES[pol]['geom'])
            h = np.float64 if g == jnp.float64 else np.float32
            pix = tuple(jnp.asarray(v.astype(h)) for v in (px, py, pz))
            for chunk, tag in [(a.chunk, pol)] + ([(1, pol + '_seq')] if pol.endswith('_acc') else []):
                try:
                    fn = bp.make_bp(pol, nfft, dr, col.fref, chunk)
                    args = bp.prepare(pol, S, col.ant, nfft, chunk)
                    t = time.perf_counter()
                    re, im = fn(*args, *pix)
                    host = re.dtype if re.dtype in (jnp.float32, jnp.float64) else jnp.float32
                    out[f'bp/{tag}/{w}'] = (np.asarray(re.astype(host)) + 1j * np.asarray(im.astype(host))).reshape(a.n, a.n)
                    timing[f'bp/{tag}/{w}'] = time.perf_counter() - t
                    print('bp', tag, w, 'ok', flush=True)
                except Exception as e:
                    print('bp', tag, w, 'FAILED', type(e).__name__, str(e)[:300], flush=True)
        if a.numba:
            from sarbench import cpu_ref
            u, r0 = bp.host_geometry(col.ant)
            t = time.perf_counter()
            rc = cpu_ref.range_compress(S.astype(np.complex128), nfft)
            out[f'bp/numba_fp64/{w}'] = cpu_ref.bp(rc, u, r0, px, py, pz, dr, col.fref).reshape(a.n, a.n)
            timing[f'bp/numba_fp64/{w}'] = time.perf_counter() - t
            print('numba fp64', w, 'ok', flush=True)
        if a.cuda:
            import cupy as cp
            from sarbench import gpu_ref
            u, r0 = bp.host_geometry(col.ant)
            rc = gpu_ref.range_compress(cp.asarray(S.astype(np.complex64)), nfft)
            for st in ('fp32', 'f16', 'bf16'):
                re, im = gpu_ref.bp(gpu_ref.pack(rc, st), cp.asarray(u.astype(np.float32)), cp.asarray(r0.astype(np.float32)),
                                    *(cp.asarray(v.astype(np.float32)) for v in (px, py, pz)), nfft, dr, col.fref, st)
                out[f'bp/cuda_{st}/{w}'] = (cp.asnumpy(re) + 1j * cp.asnumpy(im)).reshape(a.n, a.n).astype(np.complex64)
            del rc
            cp.get_default_memory_pool().free_all_blocks()
            print('cuda', w, 'ok', flush=True)
        if a.ffbp_T:
            levels = ffbp.default_levels(a.n, K, a.ffbp_T, a.ffbp_levels)
            plan = ffbp.make_plan(col, a.n, a.spacing, levels, a.ffbp_T, block=a.block)
            coll = ffbp.collection_arrays(plan, col.ant)
            if w == 'p1':
                print('ffbp levels', levels, 'passbands', [(round(l['pass_k'], 2), round(l['pass_p'], 2)) for l in plan['levels']], flush=True)
            for pol in [p for p in a.ffbp_policies.split(',') if p and (a.x64 or not ffbp.needs_x64(p))]:
                try:
                    fn = ffbp.make_ffbp(pol, plan)
                    scale = np.abs(S).max()
                    hre, him = ffbp.prepare(pol, S)
                    t = time.perf_counter()
                    re, im = fn(hre, him, ffbp.device_arrays(pol, plan, coll))
                    out[f'ffbp/{pol}/{w}'] = (np.asarray(re) + 1j * np.asarray(im)) * scale
                    timing[f'ffbp/{pol}/{w}'] = time.perf_counter() - t
                    print('ffbp', pol, w, 'ok', flush=True)
                except Exception as e:
                    print('ffbp', pol, w, 'FAILED', type(e).__name__, str(e)[:300], flush=True)
    meta = dict(n=a.n, spacing=a.spacing, backend=jax.default_backend(), device=str(jax.devices()[0]), timing=timing,
                shift=list(shift), center=a.center)
    np.savez(a.out, meta=json.dumps(meta), **out)
    print('saved', a.out, len(out))


def register(a):
    """Shift of pass 2 relative to pass 1 from the peak of the intensity cross-correlation (pixels, sub-pixel by a parabola)."""
    z = np.load(a.images)
    i1, i2 = np.abs(z[f'{a.tag}/p1']) ** 2, np.abs(z[f'{a.tag}/p2']) ** 2
    i1, i2 = np.log(i1 + i1.mean() * 1e-3), np.log(i2 + i2.mean() * 1e-3)
    f = np.fft.ifft2(np.fft.fft2(i1 - i1.mean()) * np.conj(np.fft.fft2(i2 - i2.mean()))).real
    k = np.unravel_index(np.argmax(f), f.shape)
    out = []
    for ax, kk in enumerate(k):
        idx = [list(k), list(k), list(k)]
        idx[0][ax], idx[2][ax] = (kk - 1) % f.shape[ax], (kk + 1) % f.shape[ax]
        y0, y1, y2 = (f[tuple(i)] for i in idx)
        sub = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        s = kk + sub
        out.append(s - f.shape[ax] if s > f.shape[ax] / 2 else s)
    meta = json.loads(str(z['meta']))
    print(json.dumps(dict(shift_px=out, shift_m=[-o * meta['spacing'] for o in out], peak=float(f.max() / np.sqrt((i1.var() * i2.var())) / f.size))))


def analyze(a):
    from scipy.ndimage import uniform_filter
    from sarbench import metrics as M
    z = np.load(a.images)
    zr = np.load(a.ref) if a.ref else z
    r1, r2 = zr[f'{a.ref_tag}/p1'], zr[f'{a.ref_tag}/p2']
    win = a.win
    g_ref = M.coherence(r1, r2, win)
    # brightness strata from the reference image: local mean power in deciles
    pw = uniform_filter(np.abs(r1) ** 2, win)
    edges = np.quantile(pw, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, pw) - 1, 0, 9)
    rows = []
    tags = sorted({k.rsplit('/', 1)[0] for k in z.files if k.startswith(('bp/', 'ffbp/'))})
    gains = {}

    def family_gain(tag):
        """One complex scale per algorithm, so amplitude differences between settings stay visible.
        Backprojection shares the reference's normalisation exactly. The factorized algorithm has a
        different constant, taken from its most precise image available."""
        fam = tag.split('/')[0]
        if fam == 'bp':
            return 1.0
        if fam not in gains:
            for src in (zr, z):
                for pol in ('fp64', 'fp32'):
                    k = f'{fam}/{pol}/p1'
                    if k in src.files and fam not in gains:
                        gains[fam] = np.vdot(src[k], r1) / np.vdot(src[k], src[k])
        return gains[fam]

    for tag in tags:
        if f'{tag}/p2' not in z.files:
            continue
        t1, t2 = z[f'{tag}/p1'], z[f'{tag}/p2']
        if not (np.isfinite(t1).all() and np.isfinite(t2).all()):
            rows.append(dict(tag=tag, nonfinite=True))
            continue
        gain = family_gain(tag)
        fit = np.vdot(t1, r1) / np.vdot(t1, t1)
        g = M.coherence(t1, t2, win)
        c1 = M.coherence(t1, r1, win)
        d = g - g_ref
        row = dict(tag=tag, err_db=float(M.error_db(t1 * gain, r1)), amp_db=float(20 * np.log10(abs(fit / gain) + 1e-30)),
                   coh_vs_ref_mean=float(c1.mean()), coh_vs_ref_p01=float(np.percentile(c1, 1)),
                   ccd_mean=float(g.mean()), ccd_ref_mean=float(g_ref.mean()),
                   ccd_diff_mean=float(d.mean()), ccd_diff_rms=float(np.sqrt((d ** 2).mean())), ccd_diff_max=float(np.abs(d).max()),
                   deciles=[dict(power_db=float(10 * np.log10(pw[dec == i].mean() / pw.mean())), ccd_ref=float(g_ref[dec == i].mean()),
                                 ccd_diff=float(d[dec == i].mean()), ccd_diff_rms=float(np.sqrt((d[dec == i] ** 2).mean())),
                                 coh_vs_ref=float(c1[dec == i].mean())) for i in range(10)])
        # agreement of a thresholded change map with the float64 one
        for thr in (0.3, 0.5):
            row[f'flip_{thr}'] = float(((g < thr) != (g_ref < thr)).mean())
        rows.append(row)
        print(f"{tag:20s} err {row['err_db']:7.1f} dB  coh-vs-ref {row['coh_vs_ref_mean']:.5f} (p01 {row['coh_vs_ref_p01']:.4f})  "
              f"ccd mean {row['ccd_mean']:.4f} ref {row['ccd_ref_mean']:.4f} diff rms {row['ccd_diff_rms']:.4f} max {row['ccd_diff_max']:.3f}  "
              f"dark-decile diff {row['deciles'][0]['ccd_diff']:+.4f}  flips@0.5 {row['flip_0.5']:.5f}", flush=True)
    hist, _ = np.histogram(g_ref, bins=20, range=(0, 1))
    json.dump(dict(win=win, ref=a.ref_tag, ccd_ref_hist=[int(v) for v in hist], rows=rows,
                   meta=json.loads(str(z['meta']))), open(a.out, 'w'), indent=1)
    if a.figs:
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(a.figs, exist_ok=True)

        def save(name, arr, lo, hi, down=1, cmap='gray'):
            if down > 1:
                n = (arr.shape[0] // down) * down
                arr = arr[:n, :n].reshape(n // down, down, n // down, down).mean((1, 3))
            plt.imsave(f'{a.figs}/{name}.png', np.clip(arr, lo, hi).T[::-1], vmin=lo, vmax=hi, cmap=cmap)

        def db(x):
            return 20 * np.log10(np.abs(x) / pk + 1e-12)

        pk = np.percentile(np.abs(r1), 99.9)
        # zoom window: the 320 x 320 block holding the most bright pixels, so structures rather than open ground
        bright = (np.abs(r1) > np.percentile(np.abs(r1), 99)).astype(float)
        z0, best = (0, 0), -1
        for i in range(0, r1.shape[0] - 320 + 1, 80):
            for j in range(0, r1.shape[1] - 320 + 1, 80):
                c = bright[i:i + 320, j:j + 320].sum()
                if c > best:
                    z0, best = (i, j), c
        zs = (slice(z0[0], z0[0] + 320), slice(z0[1], z0[1] + 320))
        json.dump(dict(zoom=[int(z0[0]), int(z0[1])], peak=float(pk)), open(f'{a.figs}/zoom.json', 'w'))
        save('real_img', db(r1), -50, 0, 2)
        save('real_img_pass2', db(r2 * (np.vdot(r2, r2).real / np.vdot(r1, r1).real) ** -0.5), -50, 0, 2)
        save('real_ccd_ref', g_ref, 0, 1, 4)
        save('real_zoom_ref', db(r1[zs]), -45, 0)
        save('real_zoomccd_ref', g_ref[zs], 0, 1)
        for tag in a.fig_tags.split(','):
            if f'{tag}/p1' not in z.files:
                continue
            name = tag.replace('/', '_')
            t1, t2 = z[f'{tag}/p1'], z[f'{tag}/p2']
            t1 = t1 * family_gain(tag)
            g = M.coherence(t1, t2, win)
            save(f'real_zoom_{name}', db(t1[zs]), -45, 0)
            save(f'real_err_{name}', db((t1 - r1)[zs]), -90, -20, cmap='magma')
            save(f'real_ccd_{name}', g, 0, 1, 4)
            save(f'real_ccddiff_{name}', np.abs(g - g_ref), 0, 0.2, 4, cmap='magma')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prep')
    p.add_argument('--cphd', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--dec', type=int, default=4)
    p.add_argument('--patch', type=float, default=640.0, help='side of the ground patch to keep unaliased, metres')
    p.add_argument('--center', default='0,0', help='patch centre east,north of the scene reference point, metres')
    f = sub.add_parser('form')
    f.add_argument('--data', required=True, help='pass-1 and pass-2 files, comma separated')
    f.add_argument('--out', required=True)
    f.add_argument('--n', type=int, default=2048)
    f.add_argument('--spacing', type=float, default=0.3)
    f.add_argument('--center', default='0,0', help='patch centre east,north of the reference point, metres')
    f.add_argument('--shift', default='0,0', help='registration shift applied to pass 2, metres east,north')
    f.add_argument('--policies', default='fp32,fp32_naive,bf16_store,bf16_arith,bf16_acc,bf16_all')
    f.add_argument('--ffbp-policies', default='fp64,fp32,fp32_fast,bf16_mm,bf16_mm_l1f32,bf16,f16_mm,f16,f8_mm,f4_mm')
    f.add_argument('--ffbp-T', type=int, default=32)
    f.add_argument('--block', type=int, default=0)
    f.add_argument('--ffbp-levels', type=int, default=3)
    f.add_argument('--chunk', type=int, default=8)
    f.add_argument('--x64', action='store_true')
    f.add_argument('--numba', action='store_true')
    f.add_argument('--cuda', action='store_true')
    g = sub.add_parser('register')
    g.add_argument('--images', required=True)
    g.add_argument('--tag', default='bp/numba_fp64')
    z = sub.add_parser('analyze')
    z.add_argument('--images', required=True)
    z.add_argument('--ref')
    z.add_argument('--ref-tag', default='bp/numba_fp64')
    z.add_argument('--out', required=True)
    z.add_argument('--win', type=int, default=9)
    z.add_argument('--figs')
    z.add_argument('--fig-tags', default='bp/fp32,bp/fp32_naive,bp/bf16_store,bp/bf16_arith,bp/bf16_acc,bp/bf16_acc_seq,bp/bf16_all,ffbp/fp32,ffbp/bf16_mm,ffbp/bf16,ffbp/bf16_mm_l1f32,ffbp/f16,ffbp/f8_mm,ffbp/f4_mm')
    a = ap.parse_args()
    dict(prep=prep, form=form, register=register, analyze=analyze)[a.cmd](a)


if __name__ == '__main__':
    main()
