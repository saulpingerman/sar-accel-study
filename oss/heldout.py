"""Held-out collections: FastSAR (as installed, e.g. 0.1.0 from PyPI) on CPHD files of satellites, sites, modes and
polarizations not used in its development, on the CPU.

   python heldout.py <cphd> <vendor complex image (SICD .nitf or Capella SLC .tif) or none> <label> <out dir>

1. Fast against exact: ImageFormer and ExactFormer on the same 1024 by 1024 grid (horizontal plane at the local
   origin, spacing 0.6 times the slant-range resolution), all pulses; the difference after a fitted complex gain.
2. Vendor pixels: exact backprojection (fastsar.backproject) at the vendor's own pixel positions (io.sicd_points) of
   a 256 by 256 window at the center and one off center, with the pulses whose look direction lies within the
   vendor's processed azimuth band of the window center; amplitude correlation with the vendor's image, for the
   phase sign read_cphd chose and for the opposite one.
3. One call: form_cphd on the whole collection with default settings: time, mode, notes, warnings, a quicklook.
"""
import json, os, sys, time, warnings
import numpy as np
import fastsar
from fastsar import io

C = 299792458.0
cphd, vendor, label, outd = sys.argv[1:5]
vendor = None if vendor == 'none' else vendor
os.makedirs(outd, exist_ok=True)
res = dict(label=label, fastsar=fastsar.__version__, fastsar_file=fastsar.__file__, cphd=os.path.basename(cphd))


def log(*a):
    print(label, *a, flush=True)


def gain_err(a, b):        # error of a against b (dB) after a fitted complex gain
    a, b = a.astype(np.complex128).ravel(), b.astype(np.complex128).ravel()
    g = np.vdot(a, b) / np.vdot(a, a)
    return float(10 * np.log10(np.sum(np.abs(g * a - b) ** 2) / np.sum(np.abs(b) ** 2)))


t = time.perf_counter()
col, meta = io.read_cphd(cphd, meta=True)
S, ant, f0, df = col['S'], np.asarray(col['ant'], np.float64), float(col['fmin']), float(col['df'])
P, K = S.shape
res['read'] = dict(seconds=time.perf_counter() - t, pulses=int(P), samples=int(K), notes=meta.get('notes'),
                   mode=str(meta.get('mode')), collector=str(meta.get('collector')), polarization=str(meta.get('polarization')),
                   fc_ghz=(f0 + K / 2 * df) / 1e9, bandwidth_mhz=K * df / 1e6)
log('read', res['read'])
ref = meta.get('ref')

# 1. fast against exact on a 1024 by 1024 grid at the local origin
rres = C / (2 * K * df)
sp = 0.6 * rres
n = 1024
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    while n > 128:                    # the largest grid the collection's range ambiguity allows
        try:
            fastsar.ExactFormer(ant, f0, df, K, n, n, sp, sp, backend='cpu', ref=ref)
            break
        except ValueError:
            n //= 2
    try:
        t = time.perf_counter()
        fi = fastsar.ImageFormer(ant, f0, df, K, n, n, spx=sp, spy=sp, backend='cpu', ref=ref)
        a = fi(S); t_f = time.perf_counter() - t
        t = time.perf_counter()
        ex = fastsar.ExactFormer(ant, f0, df, K, n, n, sp, sp, backend='cpu', ref=ref)
        b = ex(S); t_e = time.perf_counter() - t
        res['fast_vs_exact'] = dict(n=n, spacing_m=sp, error_db=gain_err(a, b), factorized_s=t_f, exact_s=t_e,
                                    peak_to_mean=float(np.abs(b).max() / np.abs(b).mean()), T=int(fi.T),
                                    predicted_error_db=fi.predicted_error_db)
        del a, b, fi, ex
    except Exception as e:
        res['fast_vs_exact'] = dict(error=repr(e)[:500])
    res.setdefault('fast_vs_exact', {})['warnings'] = [str(x.message)[:300] for x in w if 'deprecated' not in str(x.message)]
log('fast vs exact', res['fast_vs_exact'])

# 2. the vendor's pixels
if vendor:
    try:
        from sarpy.io.complex.converter import open_complex
        rd = open_complex(vendor)
        sm = rd.sicd_meta
        res['vendor'] = dict(algorithm=str(sm.ImageFormation.ImageFormAlgo), grid=str(sm.Grid.Type), rows=int(sm.ImageData.NumRows),
                             cols=int(sm.ImageData.NumCols), windows={})
        lam = C / (f0 + K / 2 * df)
        dsin = float(sm.Grid.Col.ImpRespBW) * lam / 2            # the processed spread of the sine of the look angle
        d = np.gradient(ant, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        from sarpy.io.phase_history.converter import open_phase_history
        lo_, hi_ = meta.get('pulses', (0, P))
        srp = io.ecf_to_local(open_phase_history(cphd).read_pvp_variable('SRPPos', 0)[lo_:hi_], meta)
        ws = srp - ant
        look_srp = (ws * d).sum(1) / np.linalg.norm(ws, axis=1)      # the beam center (the SRP) seen from each pulse
        m = min(256, int(sm.ImageData.NumRows) // 2, int(sm.ImageData.NumCols) // 2)
        for name, fr, fc in (('center', 0.5, 0.5), ('off', 0.35, 0.65)):
            r0, c0 = int(fr * sm.ImageData.NumRows) - m // 2, int(fc * sm.ImageData.NumCols) - m // 2
            rr, cc = np.meshgrid(np.arange(r0, r0 + m), np.arange(c0, c0 + m), indexing='ij')
            pts = io.sicd_points(sm, rr, cc, meta)
            v = np.asarray(rd[r0:r0 + m, c0:c0 + m])
            wv = pts[m // 2, m // 2][None] - ant           # the window center seen from each pulse
            look = (wv * d).sum(1) / np.linalg.norm(wv, axis=1)
            sel = np.nonzero(np.abs(look - look_srp) <= dsin / 2)[0]     # within the processed band around the beam center
            if sel.size < 16:
                sel = np.arange(P)
            out = dict(pulses=int(sel.size))
            for sgn in ('as read', 'conjugated'):
                Sx = S[sel] if sgn == 'as read' else np.conjugate(S[sel])
                img = fastsar.backproject(Sx, ant[sel], f0, df, pts, ref=None if ref is None else ref[sel], backend='cpu')
                del Sx
                aa, av = np.abs(img).ravel(), np.abs(v).ravel()
                out[sgn] = dict(amp_corr=float(np.corrcoef(aa, av)[0, 1]),
                                logamp_corr=float(np.corrcoef(np.log10(aa + 1e-9 * aa.max()), np.log10(av + 1e-9 * av.max()))[0, 1]),
                                peak_to_mean=float(aa.max() / aa.mean()))
            res['vendor']['windows'][name] = out
            log('vendor', name, out)
        del rd
    except Exception as e:
        res['vendor'] = dict(error=repr(e)[:800])
        log('vendor failed', res['vendor'])
del S, col

# 3. one call on the whole collection
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    try:
        t = time.perf_counter()
        out = fastsar.form_cphd(cphd, backend='cpu')
        el = time.perf_counter() - t
        img = out['image']
        a = np.abs(img)
        res['form_cphd'] = dict(seconds=el, mode=out['mode'], shape=list(img.shape), spx=out['spx'], spy=out['spy'],
                                notes=out.get('notes'), finite=bool(np.isfinite(img).all()),
                                peak_to_mean=float(a.max() / a.mean()), p999_to_median=float(np.percentile(a, 99.9) / np.median(a)))
        dn = max(1, max(img.shape) // 1500)
        q = (a[:a.shape[0] // dn * dn, :a.shape[1] // dn * dn] ** 2).reshape(a.shape[0] // dn, dn, a.shape[1] // dn, dn).mean((1, 3))
        qd = 10 * np.log10(q / np.percentile(q, 99.5) + 1e-12)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.imsave(f'{outd}/{label}_quicklook.png', np.clip(qd, -40, 0).T, cmap='gray', vmin=-40, vmax=0, origin='lower')
        del out, img, a
    except Exception as e:
        res['form_cphd'] = dict(error=repr(e)[:800])
    res['form_cphd']['warnings'] = [str(x.message)[:300] for x in w if 'deprecated' not in str(x.message)]
log('form_cphd', res['form_cphd'])
json.dump(res, open(f'{outd}/{label}.json', 'w'), indent=1, default=str)
