#!/usr/bin/env python3
"""Precision study: does forming SAR images below float64 damage them?

  make-data   float64 phase histories for a repeat-pass clutter pair with a
              known change mask, plus an isolated point-target scene (CPU).
  form        images for every algorithm and precision policy the device
              supports. Run on the CPU first with --x64 for the reference.
  analyze     error, coherence against the float64 image, impulse response,
              noise floor and coherent change detection for each policy.
"""
import argparse
import json
import time

import numpy as np

WHICH = ('p1', 'p2', 'pts')


def next_pow2(n):
    return 1 << int(np.ceil(np.log2(n)))


def load_collect(d, tag):
    from sarbench.sim import Collect
    return Collect(fmin=float(d['fmin']), df=float(d['df']), K=int(d['K']),
                   ant=d[f'ant_{tag}'], res=float(d['res']))


def make_data(a):
    from sarbench import sim
    rng = np.random.default_rng(a.seed)
    r0 = dict(air=10e3, space=600e3)[a.geom]
    if a.K:  # pick the scene size that gives exactly K samples and K pulses
        a.scene = a.K * a.res / (1.25 * np.sqrt(2.0)) * (1 - 1e-9)
    c1 = sim.make_collect(res=a.res, scene=a.scene, r0=r0)
    c2 = sim.make_collect(res=a.res, scene=a.scene, r0=r0, offset=(0.0, 2.0, 3.0))
    assert not a.K or (c1.K == a.K and c1.Np == a.K), (c1.K, c1.Np)
    if a.hdr:
        if a.strata:
            sim.STRATA_DB = tuple(float(v) for v in a.strata.split(','))
        pos, a1, a2, full_power = sim.clutter_pair_hdr(a.scene, a.res, rng)
    else:
        pos, a1, a2 = sim.clutter_pair(a.scene, a.res, rng)

    # the NUFFT synthesis must agree with the direct sum
    sub = rng.choice(len(a1), 200, replace=False)
    small = sim.Collect(c1.fmin, c1.df, c1.K, c1.ant[:4], c1.res)
    err = np.abs(sim.simulate(small, pos[sub], a1[sub]) - sim.simulate_brute(small, pos[sub], a1[sub])).max()
    assert err < 1e-8 * np.sqrt(len(sub)), f'NUFFT mismatch {err}'

    t = time.time()
    S1, S2 = sim.simulate(c1, pos, a1), sim.simulate(c2, pos, a2)
    # noise relative to the mean signal power, or (hdr) to the power of a uniformly bright scene
    sigma = np.sqrt((full_power if a.hdr else np.mean(np.abs(S1) ** 2)) / 10 ** (a.cnr_db / 10))
    S1, S2 = sim.add_noise(S1, sigma, rng), sim.add_noise(S2, sigma, rng)
    ppos, pamp = sim.point_scene(a.scene)
    Sp = sim.simulate(c1, ppos, pamp)
    print(f'simulated {len(a1)} scatterers x {c1.Np} pulses x {c1.K} samples in {time.time() - t:.0f}s, '
          f'NUFFT check {err:.1e}')
    np.savez(a.out, fmin=c1.fmin, df=c1.df, K=c1.K, res=a.res, scene=a.scene, geom=a.geom,
             ant_p1=c1.ant, ant_p2=c2.ant, ant_pts=c1.ant, S_p1=S1, S_p2=S2, S_pts=Sp, ppos=ppos, hdr=a.hdr, strata=np.asarray(sim.STRATA_DB))


def form(a):
    import jax
    jax.config.update('jax_enable_x64', a.x64)
    import jax.numpy as jnp
    from sarbench import sim, bp, pfa, ffbp

    d = np.load(a.data)
    out = {}
    timing = {}
    policies = [p for p in a.policies.split(',') if a.x64 or not bp.needs_x64(p)]
    px, py, pz = sim.ground_grid(a.n, a.spacing)
    cols = {w: load_collect(d, w) for w in WHICH}
    K = cols['p1'].K
    nfft = next_pow2(8 * K)
    dr = bp.range_bin(cols['p1'].df, nfft)
    win = sim.taylor_2d(cols['p1'].Np, K)
    geo1 = pfa.pfa_geometry(cols['p1'])
    geos = dict(p1=geo1, pts=geo1, p2=pfa.pfa_geometry(cols['p2'], grid_from=geo1))
    nf = next_pow2(int(1.5 * max(K, cols['p1'].Np)))
    # polar-format images cover the whole unambiguous extent; keep the scene only
    dx, dy = pfa.pixel_spacing(geo1, nf, nf)
    hx, hy = int(0.45 * float(d['scene']) / dx), int(0.45 * float(d['scene']) / dy)
    crop = (slice(nf // 2 - hx, nf // 2 + hx), slice(nf // 2 - hy, nf // 2 + hy))

    algos = a.algos.split(',')
    for pol in policies:
        g = jnp.dtype(bp.POLICIES[pol]['geom'])
        h = np.float64 if g == jnp.float64 else np.float32
        pix = tuple(jnp.asarray(v.astype(h)) for v in (px, py, pz))
        for chunk, tag in ([(a.chunk, pol)] + ([(1, pol + '_seq')] if pol.endswith('_acc') else [])) if 'bp' in algos else []:
            try:
                fn = bp.make_bp(pol, nfft, dr, cols['p1'].fref, chunk)
                for w in WHICH:
                    args = bp.prepare(pol, d[f'S_{w}'], cols[w].ant, nfft, chunk, window=win)
                    t = time.perf_counter()
                    re, im = fn(*args, *pix)
                    host = re.dtype if re.dtype in (jnp.float32, jnp.float64) else jnp.float32
                    re, im = np.asarray(re.astype(host)), np.asarray(im.astype(host))
                    timing[f'bp/{tag}/{w}'] = time.perf_counter() - t
                    out[f'bp/{tag}/{w}'] = (re + 1j * im).reshape(a.n, a.n)
                print('bp', tag, 'ok', flush=True)
            except Exception as e:
                print('bp', tag, 'FAILED', type(e).__name__, str(e)[:300], flush=True)
        if pol in ('bf16_acc', 'f16_acc', 'fp32_naive', 'bf16_all') or 'pfa' not in algos:
            continue
        try:
            fn = pfa.make_pfa(pol, geo1['nkx'], geo1['nky'], nf, nf)
            for w in WHICH:
                out[f'pfa/{pol}/{w}'] = np.asarray(fn(*pfa.prepare(pol, d[f'S_{w}'], geos[w]))[crop])
            print('pfa', pol, 'ok', flush=True)
        except Exception as e:
            print('pfa', pol, 'FAILED', type(e).__name__, str(e)[:300], flush=True)

    sx, sy = pfa.pixel_spacing(geo1, nf, nf)
    for pol in pfa.MM_POLICIES if 'pfamm' in algos else []:
        try:
            fn = pfa.make_pfa_mm(pol, geo1['nkx'], geo1['nky'], nf, nf)
            for w in WHICH:
                re, im = fn(*pfa.prepare_mm(pol, d[f'S_{w}'], geos[w], nf, nf, sx, sy))
                out[f'pfamm/{pol}/{w}'] = (np.asarray(re[crop]) + 1j * np.asarray(im[crop])).astype(np.complex64)
            print('pfamm', pol, 'ok', flush=True)
        except Exception as e:
            print('pfamm', pol, 'FAILED', type(e).__name__, str(e)[:300], flush=True)

    for pol in ('fp64', 'fp32') if 'pfaczt' in algos else ():
        if pol == 'fp64' and not a.x64:
            continue
        try:
            fn = pfa.make_pfa_czt(pol, K, cols['p1'].Np, geo1['nkx'], geo1['nky'], nf, nf)
            for w in WHICH:
                out[f'pfaczt/{pol}/{w}'] = np.asarray(fn(*pfa.prepare_czt(pol, d[f'S_{w}'], geos[w]))[crop])
            print('pfaczt', pol, 'ok', flush=True)
        except Exception as e:
            print('pfaczt', pol, 'FAILED', type(e).__name__, str(e)[:300], flush=True)

    # several tile configurations: "4:2,6:8@32;4:2,4:4,3:3@16" = levels@tile size
    for ci, cfg in enumerate(a.ffbp_levels.split(';') if (a.ffbp_levels and 'ffbp' in algos) else []):
        lv, T = cfg.split('@') if '@' in cfg else (cfg, a.ffbp_T)
        levels = [tuple(int(v) for v in t.split(':')) for t in lv.split(',')]
        name = 'ffbp' if ci == 0 else f'ffbpT{T}L{len(levels)}'
        plan = ffbp.make_plan(cols['p1'], a.n, a.spacing, levels, int(T), block=a.block)
        colls = {w: ffbp.collection_arrays(plan, cols[w].ant) for w in WHICH}
        print(name, 'passbands', [(round(l['pass_k'], 2), round(l['pass_p'], 2)) for l in plan['levels']], flush=True)
        for pol in ffbp.POLICIES if ci == 0 else ('fp32', 'bf16_mm'):
            if ffbp.needs_x64(pol) and not a.x64:
                continue
            try:
                fn = ffbp.make_ffbp(pol, plan)
                for w in WHICH:
                    hre, him = ffbp.prepare(pol, d[f'S_{w}'], window=win)
                    t = time.perf_counter()
                    re, im = fn(hre, him, ffbp.device_arrays(pol, plan, colls[w]))
                    assert re.dtype == (jnp.float64 if ffbp.needs_x64(pol) else jnp.float32), re.dtype
                    re, im = np.asarray(re), np.asarray(im)
                    timing[f'{name}/{pol}/{w}'] = time.perf_counter() - t
                    out[f'{name}/{pol}/{w}'] = re + 1j * im
                print(name, pol, 'ok', flush=True)
            except Exception as e:
                print(name, pol, 'FAILED', type(e).__name__, str(e)[:300], flush=True)

    if a.numba:
        from sarbench import cpu_ref
        for w in WHICH:
            S = (d[f'S_{w}'] * win).astype(np.complex64)
            u, r0 = bp.host_geometry(cols[w].ant)
            out[f'bp/numba_fp32/{w}'] = cpu_ref.bp(cpu_ref.range_compress(S, nfft), u, r0, px, py, pz, dr,
                                                    cols['p1'].fref).reshape(a.n, a.n)
            from scipy.signal.windows import taylor
            wg = taylor(geo1['nkx'], nbar=4, sll=35, norm=False)[:, None] * taylor(geo1['nky'], nbar=4, sll=35, norm=False)[None, :]
            out[f'pfa/numba_fp32/{w}'] = cpu_ref.pfa(d[f'S_{w}'].astype(np.complex64), geos[w], nf, nf, wg)[crop]
        print('numba ok', flush=True)

    if a.oracle:
        # exact matched filter on small chips around two targets
        col = cols['pts']
        Sw = d['S_pts'] * win
        for i in (0, 1):
            x0, y0 = d['ppos'][i, 0], d['ppos'][i, 1]
            r, c = int(round(x0 / a.spacing + a.n / 2)), int(round(y0 / a.spacing + a.n / 2))
            ax = np.arange(-12, 12)
            R, Cc = np.meshgrid(r + ax, c + ax, indexing='ij')
            ox, oy = (R.ravel() - a.n / 2) * a.spacing, (Cc.ravel() - a.n / 2) * a.spacing
            out[f'oracle/{i}'] = bp.matched_filter(Sw, col, ox, oy, np.zeros_like(ox)).reshape(24, 24)
            out[f'oracle_rc/{i}'] = np.array([r, c])

    meta = dict(n=a.n, spacing=a.spacing, nfft=nfft, nf=nf, chunk=a.chunk, backend=jax.default_backend(),
                device=str(jax.devices()[0]), pfa_dx=float(dx), pfa_dy=float(dy), pfa_hx=hx, pfa_hy=hy, timing=timing)
    np.savez(a.out, meta=json.dumps(meta), **out)
    print('saved', a.out, len(out), 'images')


def analyze(a):
    from sarbench import sim, metrics as M
    d = np.load(a.data)
    ref = np.load(a.ref)
    test = np.load(a.test)
    meta = json.loads(str(ref['meta']))
    n, sp = meta['n'], meta['spacing']
    scene = float(d['scene'])
    report = dict(ref=json.loads(str(ref['meta']))['device'], test=json.loads(str(test['meta']))['device'], rows=[])

    grids = {}
    ax = (np.arange(n) - n / 2) * sp
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    def odd(v):
        return max(3, int(round(v)) | 1)

    grids['bp'] = dict(mask=sim.change_mask(X, Y, scene), thin=sim.change_mask(X, Y, scene, 'track'), spacing=(sp, sp), win=(odd(a.win_m / sp),) * 2,
                       tgt=[(x / sp + n / 2, y / sp + n / 2) for x, y, _ in d['ppos']], crop=None)
    dx, dy, hx, hy = meta['pfa_dx'], meta['pfa_dy'], meta['pfa_hx'], meta['pfa_hy']
    Xp, Yp = np.meshgrid((np.arange(2 * hx) - hx) * dx, (np.arange(2 * hy) - hy) * dy, indexing='ij')
    grids['pfa'] = dict(mask=sim.change_mask(Xp, Yp, scene), thin=sim.change_mask(Xp, Yp, scene, 'track'), spacing=(dx, dy), crop=None,
                        win=(odd(a.win_m / dx), odd(a.win_m / dy)),
                        tgt=[(x / dx + hx, y / dy + hy) for x, y, _ in d['ppos'] if abs(x) < 0.4 * scene and abs(y) < 0.4 * scene])
    grids['pfamm'] = grids['pfa']
    grids['pfaczt'] = grids['pfa']
    grids['ffbp'] = grids['bp']
    report['windows_px'] = {k: v['win'] for k, v in grids.items()}

    def get(z, key, algo):
        img = z[key]
        c = grids['bp' if algo.startswith('ffbp') else algo]['crop']
        return img if c is None else img[c]

    gains = {}
    tags = sorted({k.rsplit('/', 1)[0] for k in test.files if k.startswith(('bp/', 'pfa', 'ffbp'))})
    for tag in tags:
        algo, pol = tag.split('/')
        G = grids['bp' if algo.startswith('ffbp') else algo]
        win = G['win']
        base = 'bp' if algo.startswith('ffbp') else dict(pfamm='pfa', pfaczt='pfa').get(algo, algo)
        r64 = {w: get(ref, f'{base}/fp64/{w}', algo) for w in WHICH}
        t = {w: get(test, f'{tag}/{w}', algo) for w in WHICH}
        row = dict(algo=algo, policy=pol)
        if algo.startswith('ffbp') and np.isfinite(t['p1']).all():
            # the factorized algorithm is normalised differently from backprojection. One complex
            # gain per algorithm, from its most precise image, so amplitude differences between
            # precision settings are not fitted away
            if algo not in gains:
                src = ref[f'{algo}/fp64/p1'] if f'{algo}/fp64/p1' in ref.files else test[f'{algo}/fp32/p1']
                gains[algo] = np.vdot(src, r64['p1']) / np.vdot(src, src)
            t = {w: v * gains[algo] for w, v in t.items()}
        if not all(np.isfinite(t[w]).all() for w in WHICH):
            row['nonfinite'] = True
            report['rows'].append(row)
            continue
        row['err_db'] = M.error_db(t['p1'], r64['p1'])
        row['coh_vs_fp64_global'] = float(M.global_coherence(t['p1'], r64['p1']))
        cmap = M.coherence(t['p1'], r64['p1'], win)
        row['coh_vs_fp64_mean'] = float(cmap.mean())
        row['coh_vs_fp64_p01'] = float(np.percentile(cmap, 1))
        row['phase_rms_deg'] = float(M.phase_rms_deg(t['p1'], r64['p1']))
        # change detection: same policy for both passes, and mixed (fp64 archive vs this policy)
        thin = None if ('hdr' in d.files and bool(d['hdr'])) else G['thin']
        _, row['ccd'] = M.ccd_report(t['p1'], t['p2'], G['mask'], win, thin=thin)
        _, row['ccd_mixed'] = M.ccd_report(r64['p1'], t['p2'], G['mask'], win, thin=thin)
        g_t = M.coherence(t['p1'], t['p2'], win)
        g_r = M.coherence(r64['p1'], r64['p2'], win)
        row['ccd_map_mean_abs_diff'] = float(np.abs(g_t - g_r).mean())
        row['ccd_map_max_abs_diff'] = float(np.abs(g_t - g_r).max())
        # impulse response on the isolated targets
        irf = []
        for rc in G['tgt']:
            try:
                m = M.irf_report(t['pts'], rc, G['spacing'])
                mr = M.irf_report(r64['pts'], rc, G['spacing'])
                irf.append(dict(ax0=m['ax0'], ax1=m['ax1'],
                                peak_db=float(20 * np.log10(abs(m['peak']) / abs(mr['peak']))),
                                peak_phase_deg=float(np.rad2deg(np.angle(m['peak'] * np.conj(mr['peak'])))),
                                shift_px=float(np.hypot(m['pos'][0] - mr['pos'][0], m['pos'][1] - mr['pos'][1]))))
            except Exception as e:
                irf.append(dict(error=str(e)[:100]))
        row['irf'] = irf
        row['floor_db'] = M.floor_db(t['pts'], G['tgt'])
        report['rows'].append(row)
        print(f"{tag:22s} err {row['err_db']:7.1f} dB  coh {row['coh_vs_fp64_mean']:.5f}  "
              f"ccd auc {row['ccd']['auc']:.4f} unch {row['ccd']['coh_unchanged']:.3f} "
              f"floor {row['floor_db']:.1f} dB", flush=True)

    if 'oracle/0' in ref.files:
        for i in (0, 1):
            r, c = ref[f'oracle_rc/{i}']
            chip = ref['bp/fp64/pts'][r - 12:r + 12, c - 12:c + 12]
            report[f'bp_fp64_vs_matched_filter_db_{i}'] = M.error_db(chip, ref[f'oracle/{i}'], fit_gain=True)
            if 'ffbp/fp64/pts' in ref.files:
                chip = ref['ffbp/fp64/pts'][r - 12:r + 12, c - 12:c + 12]
                report[f'ffbp_fp64_vs_matched_filter_db_{i}'] = M.error_db(chip, ref[f'oracle/{i}'], fit_gain=True)
                print('ffbp fp64 vs exact matched filter:', report[f'ffbp_fp64_vs_matched_filter_db_{i}'], 'dB')
            print('bp fp64 vs exact matched filter:', report[f'bp_fp64_vs_matched_filter_db_{i}'], 'dB')
    with open(a.out, 'w') as f:
        json.dump(report, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('make-data')
    m.add_argument('--out', required=True)
    m.add_argument('--geom', default='air', choices=['air', 'space'])
    m.add_argument('--res', type=float, default=0.3)
    m.add_argument('--scene', type=float, default=150.0)
    m.add_argument('--cnr-db', type=float, default=20.0)
    m.add_argument('--seed', type=int, default=0)
    m.add_argument('--K', type=int, default=0, help='force K samples and K pulses (overrides --scene)')
    m.add_argument('--strata', default='', help='clutter bands of the --hdr scene in dB, e.g. 0,-10,-20,-30,-40,-50')
    m.add_argument('--hdr', action='store_true', help='high-dynamic-range scene with partial changes (see sim.clutter_pair_hdr)')
    f = sub.add_parser('form')
    f.add_argument('--data', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--policies', default='fp64,fp32,fp32_naive,bf16_store,bf16_arith,bf16_acc,bf16_all,f16_arith,f16_acc')
    f.add_argument('--n', type=int, default=640)
    f.add_argument('--spacing', type=float, default=0.2)
    f.add_argument('--chunk', type=int, default=8)
    f.add_argument('--x64', action='store_true')
    f.add_argument('--oracle', action='store_true')
    f.add_argument('--algos', default='bp,pfa,pfamm,pfaczt,ffbp')
    f.add_argument('--numba', action='store_true', help='also form images with the multithreaded CPU kernels')
    f.add_argument('--ffbp-levels', default='', help='e.g. 4:2,6:8 = children per axis:decimation, per level')
    f.add_argument('--ffbp-T', type=int, default=32)
    f.add_argument('--block', type=int, default=0)
    z = sub.add_parser('analyze')
    z.add_argument('--data', required=True)
    z.add_argument('--ref', required=True)
    z.add_argument('--test', required=True)
    z.add_argument('--out', required=True)
    z.add_argument('--win-m', type=float, default=2.2, help='coherence window, metres on the ground')
    a = ap.parse_args()
    dict(**{'make-data': make_data}, form=form, analyze=analyze)[a.cmd](a)


if __name__ == '__main__':
    main()
