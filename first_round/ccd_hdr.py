#!/usr/bin/env python3
"""Change-detection sensitivity on the high-dynamic-range scene.

The scene has reflectivity strata at 0, -10, -20 and -30 dB, patches whose
pass-to-pass coherence is 0.9, 0.7 or 0.5 against 0.98 elsewhere, a fully
changed 1 m strip, and noise 40 dB below a uniformly bright scene. For each
precision configuration and stratum this reports how far the coherence over
unchanged ground moves from the float64 value, and how well each change class
is detected.

  python quality.py make-data --hdr --cnr-db 40 --K 1024 --out hdr.npz
  python quality.py form --data hdr.npz --out hdr_cpu.npz --n 768 --x64 --ffbp-levels "2:1,3:3,4:4@32" ...
  python ccd_hdr.py --data hdr.npz --images hdr_cpu.npz --out results/quality/hdr_cpu.json
"""
import argparse
import json

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

from sarbench import sim, metrics as M


def pd_at_pfa(score_changed, score_unchanged, pfa):
    thr = np.quantile(score_unchanged, 1.0 - pfa)
    return float((score_changed > thr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--ref', help='file holding the float64 images, if not --images')
    ap.add_argument('--out', required=True)
    ap.add_argument('--win', type=int, default=11)
    ap.add_argument('--win-strip', type=int, default=5)
    a = ap.parse_args()
    d = np.load(a.data)
    if 'strata' in d.files:
        sim.STRATA_DB = tuple(float(v) for v in d['strata'])
    z = np.load(a.images)
    zr = np.load(a.ref) if a.ref else z
    meta = json.loads(str(zr['meta']))
    n, sp, scene = meta['n'], meta['spacing'], float(d['scene'])
    ax = (np.arange(n) - n / 2) * sp
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    stratum = sim.hdr_stratum(Y, scene)
    cls = sim.hdr_change_class(X, Y, scene)
    edges, _ = sim.hdr_layout(scene)
    # the two bright reflectors sit at x = -0.41 scene; keep them out of the strata statistics
    inside = (X > -0.38 * scene) & (X < 0.43 * scene) & (Y > edges[0] + 2) & (Y < edges[-1] - 2)
    # unchanged ground: away from every changed pixel by one window, and away from stratum edges
    k = np.ones((a.win, a.win), bool)
    edge_band = np.zeros_like(inside)
    for e in edges[1:-1]:
        edge_band |= np.abs(Y - e) < a.win * sp
    unchanged = inside & ~binary_dilation(cls >= 0, k) & ~edge_band
    nstr, ncls = len(sim.STRATA_DB), len(sim.CLASSES)
    core = {c: binary_erosion(cls == c, np.ones((7, 7), bool)) for c in range(ncls)}   # patches: 20 px, keep the middle
    strip = cls == ncls                                                               # 5 px wide, scored with the small window

    ref1, ref2 = zr['bp/fp64/p1'], zr['bp/fp64/p2']
    g_ref = M.coherence(ref1, ref2, a.win)
    g_ref_s = M.coherence(ref1, ref2, a.win_strip)
    looks = a.win * a.win * (sp / float(d['res'])) ** 2            # rough count of independent looks per window
    tags = sorted({key.rsplit('/', 1)[0] for key in z.files if key.startswith(('bp/', 'ffbp'))})
    rows = []
    for tag in tags:
        p1, p2 = z[f'{tag}/p1'], z[f'{tag}/p2']
        if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            rows.append(dict(tag=tag, nonfinite=True))
            continue
        g = M.coherence(p1, p2, a.win)
        gs = M.coherence(p1, p2, a.win_strip)
        row = dict(tag=tag, strata=[])
        for b in range(nstr):
            un = unchanged & (stratum == b)
            diff = g[un] - g_ref[un]
            neff = un.sum() / (a.win * a.win)                        # windows are the independent units
            e = dict(db=sim.STRATA_DB[b], coh=float(g[un].mean()), coh_ref=float(g_ref[un].mean()),
                     dcoh=float(diff.mean()), dcoh_se=float(diff.std() / np.sqrt(max(neff, 1.0))),
                     err_db=float(M.error_db(p1[un] * (np.vdot(p1[un], ref1[un]) / np.vdot(p1[un], p1[un])), ref1[un])),
                     classes=[])
            for c in range(ncls):
                ch = core[c] & (stratum == b)
                e['classes'].append(dict(gamma=sim.CLASSES[c], coh=float(g[ch].mean()), coh_ref=float(g_ref[ch].mean()),
                                         auc=float(M.auc(np.concatenate([1 - g[ch], 1 - g[un]]),
                                                         np.concatenate([np.ones(ch.sum(), bool), np.zeros(un.sum(), bool)]))),
                                         auc_ref=float(M.auc(np.concatenate([1 - g_ref[ch], 1 - g_ref[un]]),
                                                             np.concatenate([np.ones(ch.sum(), bool), np.zeros(un.sum(), bool)]))),
                                         pd=pd_at_pfa(1 - g[ch], 1 - g[un], 1e-3), pd_ref=pd_at_pfa(1 - g_ref[ch], 1 - g_ref[un], 1e-3)))
            st = strip & (stratum == b) & inside
            e['strip'] = dict(auc=float(M.auc(np.concatenate([1 - gs[st], 1 - gs[un]]),
                                              np.concatenate([np.ones(st.sum(), bool), np.zeros(un.sum(), bool)]))),
                              auc_ref=float(M.auc(np.concatenate([1 - g_ref_s[st], 1 - g_ref_s[un]]),
                                                  np.concatenate([np.ones(st.sum(), bool), np.zeros(un.sum(), bool)]))),
                              pd=pd_at_pfa(1 - gs[st], 1 - gs[un], 1e-3), pd_ref=pd_at_pfa(1 - g_ref_s[st], 1 - g_ref_s[un], 1e-3))
            row['strata'].append(e)
        rows.append(row)
        print(f"{tag:22s} " + '  '.join(f"{e['db']:+.0f}dB dcoh {e['dcoh']:+.4f}+-{e['dcoh_se']:.4f} err {e['err_db']:6.1f} "
                                        f"auc.9 {e['classes'][0]['auc']:.3f}({e['classes'][0]['auc_ref']:.3f}) strip {e['strip']['auc']:.3f}"
                                        for e in row['strata']), flush=True)
    json.dump(dict(win=a.win, win_strip=a.win_strip, looks=looks, rows=rows), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
