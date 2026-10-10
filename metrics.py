#!/usr/bin/env python3
"""Second round: compare every image of a scene with its float64 reference.

For each test image:
  err_db        energy of the difference from the reference over the energy of the reference, after one
                complex factor per algorithm family (fitted on the float32 image of that family and device; in the
                release records the float32-class image: CUDA or C++ float32, TPU three-pass)
  err_fit_db    the same after a factor fitted to the image itself, which removes any overall gain error
  gain_db       amplitude of that fitted factor relative to the family factor (overall amplitude change)
  coherence     over a sliding window (default 5 x 5): mean, 1st and 0.1th percentile, minimum, and the
                fractions of pixels below 0.99 and 0.9
  amplitude     |20 log10(|test| / |ref|)| over the brighter half of the pixels: median, 99th and 99.9th
                percentile, maximum
  phase         |phase difference| in degrees over the same pixels: the same percentiles
  max_diff_db   largest pixel difference relative to the brightest reference pixel
  bins          median block error against block intensity (64-pixel blocks), and the fitted slope

--crops "name:i0,j0,h,w;..." saves those regions of the reference, of each test image, and of its coherence
map to <figs>/<scene>_crops.npz for figures.

  python metrics.py --ref out/ref --test out/tpu-v5e,out/gpu-l4 --out results/fastsar/panama.json
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.ndimage import uniform_filter


def lsgain(src, ref):
    s = src.astype(np.complex128)
    return complex(np.vdot(s, ref) / np.vdot(s, s).real)


def ampgain(src, ref):
    """Real gain matching the energies, for images whose phase differs from the reference by a smooth screen."""
    s = src.astype(np.complex128)
    return float(np.sqrt(np.vdot(ref, ref).real / np.vdot(s, s).real))


def phase_screen(x, b, nx, ny):
    """Quadratic phase screen fitted to the cross product x = test * conj(ref) over b-pixel blocks."""
    mx, my = nx // b, ny // b
    c = x[:mx * b, :my * b].reshape(mx, b, my, b).sum((1, 3))
    w = np.abs(c)
    ang = np.angle(c)
    # unwrap along both axes from the centre block
    ang = np.unwrap(np.unwrap(ang, axis=0), axis=1)
    ang -= ang[mx // 2, my // 2]
    u = (np.arange(mx) * b + b / 2.0 - nx / 2.0) / (nx / 2.0)
    v = (np.arange(my) * b + b / 2.0 - ny / 2.0) / (ny / 2.0)
    U, V = np.meshgrid(u, v, indexing='ij')
    A = np.stack([U * U, V * V, U * V, U, V, np.ones_like(U)], axis=-1).reshape(-1, 6)
    sw = np.sqrt(w.ravel())
    coef = np.linalg.lstsq(A * sw[:, None], ang.ravel() * sw, rcond=None)[0]
    fit = (A @ coef).reshape(mx, my)
    resid = np.sqrt(np.average((ang - fit) ** 2, weights=w))
    ui = (np.arange(nx) - nx / 2.0) / (nx / 2.0)
    vi = (np.arange(ny) - ny / 2.0) / (ny / 2.0)
    Ui, Vi = np.meshgrid(ui, vi, indexing='ij')
    screen = coef[0] * Ui * Ui + coef[1] * Vi * Vi + coef[2] * Ui * Vi + coef[3] * Ui + coef[4] * Vi + coef[5]
    return screen, coef, float(resid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--win', type=int, default=5)
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--figs')
    ap.add_argument('--crops', default='')
    ap.add_argument('--scene', default='scene')
    a = ap.parse_args()
    ref = np.load(f'{a.ref}/bp_numba_fp64.npy')
    nx, ny = ref.shape
    pw = np.abs(ref) ** 2
    e_ref = float(pw.sum())
    pr = uniform_filter(pw, a.win)
    bright = pw >= np.median(pw)
    peak = float(np.sqrt(pw.max()))
    b = a.block
    mx, my = nx // b, ny // b
    Eb = pw[:mx * b, :my * b].reshape(mx, b, my, b).sum((1, 3))
    bri = 10 * np.log10(Eb / Eb.mean())
    edges = [-100, -15, -10, -5, 0, 5, 10, 100]
    crops = []
    for c in [c for c in a.crops.split(';') if c]:
        name, box = c.split(':')
        crops.append((name, *[int(v) for v in box.split(',')]))
    saved = {f'{name}/ref': ref[i0:i0 + h, j0:j0 + w].astype(np.complex64) for name, i0, j0, h, w in crops}
    out = dict(scene=a.scene, nx=nx, ny=ny, win=a.win, block=b, edges=edges,
               blocks=[int(((bri >= lo) & (bri < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])],
               dyn_range_db=dict(p01=float(np.percentile(bri, 1)), p99=float(np.percentile(bri, 99))), rows=[])
    if a.figs:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(a.figs, exist_ok=True)
        dn = max(1, max(nx, ny) // 1400)
        ov = pw[:(nx // dn) * dn, :(ny // dn) * dn].reshape(nx // dn, dn, ny // dn, dn).mean((1, 3))
        ovdb = 10 * np.log10(ov / np.percentile(ov, 99.9) + 1e-12)
        plt.imsave(f'{a.figs}/{a.scene}_overview.png', np.clip(ovdb, -45, 0), vmin=-45, vmax=0, cmap='gray')
        np.save(f'{a.figs}/{a.scene}_overview_db.npy', ovdb.astype(np.float32))
    for d in a.test.split(','):
        meta = json.load(open(f'{d}/timing.json')) if os.path.exists(f'{d}/timing.json') else {}
        label = meta.get('_meta', {}).get('label', os.path.basename(d))
        fam = {}
        for path in sorted(glob.glob(f'{d}/*.npy')):
            name = os.path.basename(path)[:-4]
            if name == 'bp_numba_fp64':
                continue
            tag = name.replace('_', '/', 1)
            t = np.load(path)
            if t.shape != ref.shape or not np.isfinite(t).all():
                out['rows'].append(dict(label=label, tag=tag, bad=True))
                continue
            g_own = ampgain(t, ref) if tag.startswith('pfa') else lsgain(t, ref)
            if tag.startswith(('ffbp', 'pfa')):
                family = name.split('_')[0]
                suffix = ''.join('_' + s for s in name.split('_')[2:] if s in ('conv', 'taps'))
                src = f'{d}/{family}_fp32{suffix}.npy'
                if family == 'ffbp' and not os.path.exists(src):       # release records: the float32-class image of the device
                    src = next((f'{d}/ffbp_{t}.npy' for t in ('fp32_cuda', 'fp32_cpp', 'fp32_high_pallas2_direct') if os.path.exists(f'{d}/ffbp_{t}.npy')), src)
                if (family, suffix) not in fam:
                    fam[family, suffix] = ((ampgain if family == 'pfa' else lsgain)(np.load(src), ref)) if os.path.exists(src) else g_own
                g_fam = fam[family, suffix]
            else:
                g_fam = g_own                  # exact backprojection: its range compression is normalised differently
            tf = t.astype(np.complex128) * g_fam
            diff = np.abs(tf - ref) ** 2
            row = dict(label=label, tag=tag, err_db=float(10 * np.log10(diff.sum() / e_ref + 1e-300)),
                       err_fit_db=float(10 * np.log10(np.sum(np.abs(t.astype(np.complex128) * g_own - ref) ** 2) / e_ref + 1e-300)),
                       gain_db=float(20 * np.log10(abs(g_own) / abs(g_fam))),
                       max_diff_db=float(10 * np.log10(diff.max()) - 20 * np.log10(peak)))
            x = tf * np.conj(ref)
            # error after removing a quadratic phase screen (polar format's planar-wavefront term); near zero
            # change for the other algorithms
            screen, coef, resid = phase_screen(x, b, nx, ny)
            row['screen'] = dict(coef=[float(c) for c in coef], resid_rad=resid,
                                 err_db=float(10 * np.log10(np.sum(np.abs(tf * np.exp(-1j * screen) - ref) ** 2) / e_ref + 1e-300)))
            del screen
            amp_err = np.abs(tf) - np.sqrt(pw)
            row['amp_err_db'] = float(10 * np.log10(np.sum(amp_err * amp_err) / e_ref + 1e-300))
            del amp_err
            num = uniform_filter(x.real, a.win) + 1j * uniform_filter(x.imag, a.win)
            coh = np.abs(num) / np.sqrt(np.maximum(uniform_filter(np.abs(tf) ** 2, a.win) * pr, 1e-300))
            del num
            row.update(coh_mean=float(coh.mean()), coh_p01=float(np.percentile(coh, 1)), coh_p001=float(np.percentile(coh, 0.1)),
                       coh_min=float(coh.min()), below_099=float((coh < 0.99).mean()), below_09=float((coh < 0.9).mean()))
            amp = np.abs(20 * np.log10(np.maximum(np.abs(tf[bright]), 1e-300) / np.sqrt(pw[bright])))
            ph = np.abs(np.degrees(np.angle(x[bright])))
            del x
            for key, v in (('amp_db', amp), ('phase_deg', ph)):
                row[key] = dict(p50=float(np.percentile(v, 50)), p99=float(np.percentile(v, 99)), p999=float(np.percentile(v, 99.9)), max=float(v.max()))
            del amp, ph
            eb = diff[:mx * b, :my * b].reshape(mx, b, my, b).sum((1, 3))
            rel = 10 * np.log10(eb / Eb + 1e-300)
            slope = np.polyfit(bri.ravel(), rel.ravel(), 1)[0]
            row['bins'] = [float(np.median(rel[(bri >= lo) & (bri < hi)])) if ((bri >= lo) & (bri < hi)).any() else None for lo, hi in zip(edges[:-1], edges[1:])]
            row['slope'] = float(slope)
            row['corr'] = float(np.corrcoef(bri.ravel(), rel.ravel())[0, 1])
            # error by distance from the scene centre (largest normalised axis offset), for algorithms whose
            # approximations grow outward
            rr = np.maximum(np.abs((np.arange(mx) * b + b / 2.0 - nx / 2.0) / (nx / 2.0))[:, None],
                            np.abs((np.arange(my) * b + b / 2.0 - ny / 2.0) / (ny / 2.0))[None, :])
            row['rings'] = [float(10 * np.log10(eb[(rr >= lo) & (rr < hi)].sum() / max(Eb[(rr >= lo) & (rr < hi)].sum(), 1e-300)))
                            for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))]
            row['timing'] = meta.get(tag, {})
            out['rows'].append(row)
            print(f"{label:9s} {tag:22s} err {row['err_db']:6.1f} fit {row['err_fit_db']:6.1f} amp {row['amp_err_db']:6.1f} screen {row['screen']['err_db']:6.1f} gain {row['gain_db']:+.3f} dB | coh mean {row['coh_mean']:.5f} p01 {row['coh_p01']:.4f} "
                  f"min {row['coh_min']:.3f} <0.99 {100 * row['below_099']:.3f}% | amp p99 {row['amp_db']['p99']:.3f} max {row['amp_db']['max']:.2f} dB | "
                  f"phase p99 {row['phase_deg']['p99']:.2f} max {row['phase_deg']['max']:.1f} deg | maxdiff {row['max_diff_db']:.1f} dB", flush=True)
            for cname, i0, j0, h, w in crops:
                saved[f'{cname}/{label}/{name}'] = tf[i0:i0 + h, j0:j0 + w].astype(np.complex64)
                saved[f'{cname}/{label}/{name}/coh'] = coh[i0:i0 + h, j0:j0 + w].astype(np.float32)
            del t, tf, diff, coh
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    if crops and a.figs:
        np.savez_compressed(f'{a.figs}/{a.scene}_crops.npz', **saved)


if __name__ == '__main__':
    main()
