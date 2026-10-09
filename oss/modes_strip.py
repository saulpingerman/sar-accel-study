"""Stripmap comparison on one Capella collection: FastSAR's mosaic over the whole vendor footprint (timed) and ISCE3's
backprojection on three regions, each scored against the float64 reference at its own pixels with its own processed
aperture (FastSAR: Hann window over the vendor's processed azimuth bandwidth, per pixel; ISCE3: its uniform coherent
processing interval for the azimuth resolution ds = 1 / ImpRespBW, replicated from its source).
    python -I modes_strip.py <cphd> <sicd> <out dir> fastsar <cpu|cuda>
    python -I modes_strip.py <cphd> <sicd> <out dir> isce3 <cpu|cuda>
The mosaic is timed on its second call (a warm device, compilation excluded, as everywhere in the study).
With MODE=spot the collection is a spotlight: full aperture for every pixel, Taylor window along pulses as well,
FastSAR's whole-image ImageFormer on a grid centered on the scene reference point (timed on its second call, as in
the study's protocol), and ISCE3 with ds = 1 mm, i.e. every pulse, as on Panama. ISCE_OS sets ISCE3's range
oversampling (default 2).
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modes_common as mc  # noqa: E402

cphd, sicd, out, what, dev = sys.argv[1:6]
# FastSAR's memory fallbacks (MemoryWarning), recorded in each result: tables show full-speed runs only
import warnings
FALLBACKS = []
_show = warnings.showwarning


def _record(message, category, *a, **k):
    if category.__name__ == 'MemoryWarning':
        FALLBACKS.append(str(message))
    _show(message, category, *a, **k)


warnings.showwarning = _record
warnings.filterwarnings('always', message='.*full speed needs.*')
os.makedirs(out, exist_ok=True)
t_load = time.perf_counter()
SPOT = os.environ.get('MODE', 'strip') == 'spot'
fx, ant, meta, sm, ap = mc.load(cphd, sicd, stripmap=not SPOT)
P, K = fx['S'].shape
rec = dict(collection=os.path.basename(cphd), pulses=P, samples=K, load_seconds=time.perf_counter() - t_load,
           sicd=dict(rows=int(sm.ImageData.NumRows), cols=int(sm.ImageData.NumCols), row_ss=float(sm.Grid.Row.SS), col_ss=float(sm.Grid.Col.SS),
                     col_imp_resp_bw=float(sm.Grid.Col.ImpRespBW), col_imp_resp_wid=float(sm.Grid.Col.ImpRespWid)))
regs = mc.regions(sm, meta)
n = 512

if what == 'fastsar':
    from fastsar import patches
    beam = mc.beam_fn(ant, ap)
    # output grid: horizontal along-track and cross-track axes at the vendor's azimuth and ground-range spacing, covering
    # the vendor image's footprint
    rows, cols = rec['sicd']['rows'], rec['sicd']['cols']
    R_, C_ = np.meshgrid([0, rows - 1], [0, cols - 1], indexing='ij')
    corners = mc.io.sicd_points(sm, R_.ravel(), C_.ravel(), meta)
    d = ap['d'][P // 2].copy(); d[2] = 0; e1 = d / np.linalg.norm(d); e2 = np.cross([0, 0, 1.0], e1)
    graze = np.radians(float(sm.SCPCOA.GrazeAng))
    spx, spy = rec['sicd']['col_ss'], rec['sicd']['row_ss'] / np.cos(graze)
    if SPOT:          # vendors oversample spotlight images (Capella: 0.14 m azimuth pixels); FastSAR's grid is 0.8 resolution
        spx = max(spx, 0.8 * float(sm.Grid.Col.ImpRespWid))
        spy = max(spy, 0.8 * float(sm.Grid.Row.ImpRespWid) / np.cos(graze))
    a1, a2 = corners @ e1, corners @ e2
    if SPOT:          # whole-image grid centered on the scene reference point (the local origin), as form_image uses
        nx, ny = 2 * int(np.ceil(np.abs(a1).max() / spx)), 2 * int(np.ceil(np.abs(a2).max() / spy))
        origin = -(nx / 2.0) * spx * e1 - (ny / 2.0) * spy * e2
    else:
        origin = a1.min() * e1 + a2.min() * e2 + np.array([0, 0, corners[:, 2].mean()])
        nx, ny = int(np.ceil((a1.max() - a1.min()) / spx)) + 1, int(np.ceil((a2.max() - a2.min()) / spy)) + 1
    rec['grid'] = dict(nx=nx, ny=ny, spx=spx, spy=spy, e1=e1.tolist(), e2=e2.tolist(), origin=origin.tolist(), backend=dev)
    print('grid', nx, ny, f'{nx * ny / 1e6:.1f} Mpx', flush=True)
    info = []
    ps = int(os.environ.get('MOSAIC_PATCH', 1024))
    rec['grid']['patch'] = ps
    t = time.perf_counter()
    if SPOT and dev == 'cuda' and os.environ.get('MODES_SPOT_MOSAIC'):   # a mosaic of range-gated patches over the full
        # aperture (the whole-image former below streams a history larger than the GPU through its first level)
        img = patches.form_mosaic(fx, ant, origin, nx, ny, spx, spy, e1, e2, patch=(ps, ps), backend=dev, info=info)
    elif SPOT:
        from fastsar import ImageFormer
        former = ImageFormer(ant, fx['fmin'], fx['df'], K, nx, ny, spx, spy, e1, e2, backend=dev, window=False,   # the data are already windowed
                             ref=fx['ref'])          # referenced to the bistatic half path, not |ant|
        img = former(fx['S'])
        rec['first_call_seconds'] = time.perf_counter() - t
        img = None                      # the second call's image replaces it (memory)
        # timed host memory to host memory (the study's rule): the phase history from host memory, the image back to it
        t = time.perf_counter()
        img = former(fx["S"])
    else:
        if os.environ.get('MODES_PROFILE'):      # diagnostics: where the mosaic's time goes
            import cProfile, pstats
            os.environ['FASTSAR_TIMING'] = '1'
            pr = cProfile.Profile(); pr.enable()
        # timed warm, as every time in the study (compilation and one-time setup excluded): a first mosaic, then the
        # timed one, host memory to host memory
        img = patches.form_mosaic(fx, ant, origin, nx, ny, spx, spy, e1, e2, patch=(ps, ps), beam=beam, awin=mc.hann, backend=dev)
        rec['first_call_seconds'] = time.perf_counter() - t
        print(f'first mosaic {rec["first_call_seconds"]:.1f} s', flush=True)
        img = None
        t = time.perf_counter()
        img = patches.form_mosaic(fx, ant, origin, nx, ny, spx, spy, e1, e2, patch=(ps, ps), beam=beam, awin=mc.hann, backend=dev, info=info)
        if os.environ.get('MODES_PROFILE'):
            pr.disable(); pstats.Stats(pr).sort_stats('cumulative').print_stats(30)
            print(patches.report_timing()); sys.stdout.flush()
    if 'seconds' not in rec:
        rec['seconds'] = time.perf_counter() - t
    rec['patches'] = len(info)
    print(f'mosaic {rec["seconds"]:.1f} s, {len(info)} patches', flush=True)
    if os.environ.get('MODES_SAVE_IMAGE', '1') == '1':      # the full image (gigabytes); the region crops are kept regardless
        np.save(f'{out}/fastsar_{dev}_mosaic.npy', img)
    rec['memory_fallbacks'] = FALLBACKS; rec['full_speed'] = not FALLBACKS
    json.dump(rec, open(f'{out}/fastsar_{dev}.json', 'w'), indent=1, default=float)
    rec['regions'] = {}
    for k, r in enumerate(regs):
        c = np.asarray(r['local'])
        i0, j0 = int(round((c - origin) @ e1 / spx)) - n // 2, int(round((c - origin) @ e2 / spy)) - n // 2
        I, J = np.meshgrid(np.arange(i0, i0 + n), np.arange(j0, j0 + n), indexing='ij')
        pts = origin + (I.ravel() * spx)[:, None] * e1 + (J.ravel() * spy)[:, None] * e2
        pts[:, 2] = origin[2]
        crop = np.asarray(img[i0:i0 + n, j0:j0 + n])
        t = time.perf_counter()
        ref = mc.cached_reference(fx, ant, pts, beam_ap=None if SPOT else ap)
        e = mc.err_db(crop.ravel(), ref)
        rec['regions'][f'r{k}'] = dict(i0=i0, j0=j0, error_db=e, reference_seconds=time.perf_counter() - t, center=r)
        np.savez_compressed(f'{out}/fastsar_{dev}_r{k}.npz', img=crop, ref=ref.reshape(n, n).astype(np.complex64), pts=pts)
        print(f'region {k}: error {e:.2f} dB', flush=True)
    rec['memory_fallbacks'] = FALLBACKS; rec['full_speed'] = not FALLBACKS
    json.dump(rec, open(f'{out}/fastsar_{dev}.json', 'w'), indent=1, default=float)

else:
    import isce3, scipy.fft
    C = mc.C
    S, f0, df, delta = fx['S'], fx['fmin'], fx['df'], fx['ref']
    R, o = meta['R'], meta['origin']
    tx = np.asarray(meta['tx']) @ R + o; rcv = np.asarray(meta['rcv']) @ R + o
    ttx = np.asarray(meta['tx_time'], np.float64)
    dtm = float(np.mean(np.diff(ttx)))
    txv = np.gradient(tx, ttx, axis=0)
    fref = f0 + (K // 2) * df; wvl = C / fref; h = K // 2
    epoch = isce3.core.DateTime(2000, 1, 1)
    t_rel = ttx - ttx[0]
    # ISCE3's input grid places pulse k at time k dt (one PRF) and requires uniform state vectors. The pulse times are
    # not exactly uniform (PRF changes and dropped pulses), so the orbit is given on ISCE3's own time axis: one state
    # vector per pulse at k dt with that pulse's transmit position. Pulse k then sits exactly where ISCE3 assumes it
    # is. The velocity is the true one (d position / d time), since ISCE3 uses it to place the antenna at receive time.
    tau = np.arange(P) * dtm
    rec['pulse_time_deviation_s'] = float(np.max(np.abs(t_rel - tau)))
    orbit = isce3.core.Orbit([isce3.core.StateVector(epoch + isce3.core.TimeDelta(float(tau[k])), tx[k], txv[k]) for k in range(P)], epoch)
    srp_e = mc.io.local_to_ecf(np.asarray(regs[1]['local']), meta)
    right = np.cross(txv[P // 2], tx[P // 2] / np.linalg.norm(tx[P // 2])) @ (srp_e - tx[P // 2]) > 0
    side = isce3.core.LookSide.Right if right else isce3.core.LookSide.Left
    # Doppler centroid: the mean over pulses of the Doppler of each pulse's own scene reference point
    fD = float(np.mean(2.0 / wvl * ap['sp'] * np.linalg.norm(txv, axis=1)))
    dop = isce3.core.LUT2d(np.array([0.0, 2e7]), np.array([-1e3, 1e3 + t_rel[-1]]), np.full((2, 2), fD))
    ds = 1e-3 if SPOT else 1.0 / rec['sicd']['col_imp_resp_bw']
    rec.update(doppler_centroid_hz=fD, ds=ds, look_side='right' if right else 'left', device=dev)
    # range-compressed buffer on an absolute range axis (as in the Panama comparison)
    isce_os = int(os.environ.get('ISCE_OS', '2'))
    nfft = 1 << int(np.ceil(np.log2(isce_os * K))); dr = C / (2 * df * nfft)
    dmin = float(delta.min()); off = (delta - dmin) / dr; m_int = np.floor(off).astype(int); frac = off - m_int
    width = nfft + int(m_int.max()) + 2
    r0_buf = dmin - (nfft // 2) * dr
    t = time.perf_counter()
    buf = np.zeros((P, width), np.complex64)
    q = (np.arange(K) - h).astype(np.float64)
    for p0 in range(0, P, 512):
        sl = slice(p0, min(P, p0 + 512)); nn = sl.stop - sl.start
        Sb = S[sl] * np.exp(-2j * np.pi * q[None, :] * frac[sl, None] / nfft).astype(np.complex64)
        pad = np.zeros((nn, nfft), np.complex64); pad[:, :K - h] = Sb[:, h:]; pad[:, nfft - h:] = Sb[:, :h]
        prof = scipy.fft.fftshift(scipy.fft.ifft(pad, axis=1, workers=-1), axes=1)
        prof *= np.exp(-1j * 4 * np.pi * fref * delta[sl, None] / C).astype(np.complex64)
        for i in range(nn):
            k = sl.start + i; buf[k, m_int[k]:m_int[k] + nfft] = prof[i]
    rec['buffer_seconds'] = time.perf_counter() - t
    in_grid = isce3.product.RadarGridParameters(0.0, wvl, 1.0 / dtm, r0_buf, dr, side, P, width, epoch)
    in_geom = isce3.container.RadarGeometry(in_grid, orbit, dop)
    kern = isce3.core.TabulatedKernelF32(isce3.core.KnabKernel(8.0, K / nfft), 4096)
    ell = isce3.core.Ellipsoid()
    bp_fn = isce3.focus.backproject
    if dev == 'cuda':
        import isce3.cuda.focus
        bp_fn = isce3.cuda.focus.backproject
    rec['regions'] = {}
    for k, r in enumerate(regs):
        x0 = mc.io.local_to_ecf(np.asarray(r['local']), meta)
        dem = isce3.geometry.DEMInterpolator(float(ell.xyz_to_lon_lat(x0)[2]))
        tc, rc = isce3.geometry.geo2rdr_bracket(x0, orbit, dop, wvl, side)
        pc, vc = orbit.interpolate(tc)
        dt_out = rec['sicd']['col_ss'] / (np.linalg.norm(vc) * np.linalg.norm(x0) / np.linalg.norm(pc))
        dr_out = rec['sicd']['row_ss']
        og = isce3.product.RadarGridParameters(tc - (n // 2) * dt_out, wvl, 1.0 / dt_out, rc - (n // 2) * dr_out, dr_out, side, n, n, epoch)
        out_geom = isce3.container.RadarGeometry(og, orbit, dop)
        img = np.zeros((n, n), np.complex64)
        if dev == 'cuda' and k == 0:
            bp_fn(np.zeros((8, 8), np.complex64), isce3.container.RadarGeometry(isce3.product.RadarGridParameters(og.sensing_start, wvl, og.prf, og.starting_range, dr_out, side, 8, 8, epoch), orbit, dop), buf, in_geom, dem, fref, ds, kern, 'nodelay')
        t = time.perf_counter()
        ok = bp_fn(img, out_geom, buf, in_geom, dem, fref, ds, kern, 'nodelay')
        sec = time.perf_counter() - t
        # ISCE3's targets and its coherent processing interval for each (Backproject.cpp)
        X = np.zeros((n * n, 3)); lo = np.zeros(n * n, np.int64); hi = np.zeros(n * n, np.int64)
        for j in range(n):
            for i in range(n):
                x = np.asarray(isce3.geometry.rdr2geo_bracket(og.sensing_start + j / og.prf, og.starting_range + i * dr_out, orbit, side, fD, wvl, dem)).ravel()
                tt, rr = isce3.geometry.geo2rdr_bracket(x, orbit, dop, wvl, side)
                p_, v_ = orbit.interpolate(tt)
                cpi = wvl * rr * (np.linalg.norm(p_) / np.linalg.norm(x)) / (2 * ds) / np.linalg.norm(v_)
                X[j * n + i] = x
                lo[j * n + i] = max(int(np.floor((tt - 0.5 * cpi) / dtm)), 0)
                hi[j * n + i] = min(int(np.ceil((tt + 0.5 * cpi) / dtm)), P)
        pts = mc.io.ecf_to_local(X, meta)
        t = time.perf_counter()
        # the reference with ISCE3's own range model (receiver at transmit position plus velocity times the pixel's
        # delay); the error against the per-pulse receive position of the CPHD, as FastSAR uses, is kept beside it
        ref = mc.cached_reference(fx, ant, pts, lo=lo, hi=hi, rx=(np.asarray(meta['tx'], np.float64), txv @ R.T))
        ref_fixed = mc.cached_reference(fx, ant, pts, lo=lo, hi=hi)
        e = mc.err_db(img.ravel(), ref); ec = mc.err_db(np.conj(img.ravel()), ref)
        rec['regions'][f'r{k}'] = dict(seconds=sec, converged=bool(ok), error_db=e, conj_error_db=ec, cpi_pulses=float(np.median(hi - lo)),
                                       error_fixed_receiver_db=mc.err_db(img.ravel(), ref_fixed),
                                       reference_seconds=time.perf_counter() - t, center=r)
        np.savez_compressed(f'{out}/isce3_{dev}_r{k}.npz', img=img, ref=ref.reshape(n, n).astype(np.complex64), pts=pts)
        print(f'region {k}: {sec:.1f} s, error {e:.2f} dB (conj {ec:.2f}; fixed receiver {rec["regions"][f"r{k}"]["error_fixed_receiver_db"]:.2f}), CPI {np.median(hi - lo):.0f} pulses', flush=True)
    rec['full_image_estimate_seconds'] = float(np.mean([v['seconds'] for v in rec['regions'].values()]) * rec['sicd']['rows'] * rec['sicd']['cols'] / (n * n))
    json.dump(rec, open(f'{out}/isce3_{dev}.json', 'w'), indent=1, default=float)
