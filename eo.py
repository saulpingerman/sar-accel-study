#!/usr/bin/env python3
"""Validation of the Panama regions against Sentinel-2: each region as the geocoded float64 reference (north up, 1 m
grid) beside the Sentinel-2 true-color image of the same 1.2 km window, with the 512 by 512 crop outlined.

  uv run --with rasterio --with numpy --with matplotlib python eo.py --ref <panama_ref.npy>
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rasterio.warp import transform as rtransform
from geo_panama import Geo

R = 'results/fastsar'
REGIONS = {'locks': (3400, 6700), 'port': (9500, 1150), 'ships': (6400, 2650), 'corner': (11400, 8000)}
TITLES = {'locks': 'Cocolí Locks', 'port': 'Port of Balboa', 'ships': 'ship in the channel', 'corner': 'vegetation'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', default='paper/fig/eo_panama.pdf')
    ap.add_argument('--half', type=float, default=600.0, help='half-width of each window (m)')
    a = ap.parse_args()
    geo = Geo(f'{R}/panama_sicd.xml', 12207, 8808)
    ref = np.load(a.ref, mmap_mode='r')
    s2 = {k: np.load(f'{R}/figs/s2_{k}.npz') for k in ('S2B_20230723', 'S2A_20240223')}
    half = a.half
    fig, axes = plt.subplots(4, 3, figsize=(9.6, 13.2))
    for row, (name, (i0, j0)) in zip(axes, REGIONS.items()):
        lat, lon = geo.to_ground(i0 + 256, j0 + 256)
        (cx,), (cy,) = rtransform('EPSG:4326', 'EPSG:32617', [lon], [lat])
        ii = np.array([i0, i0 + 512, i0 + 512, i0, i0])
        jj = np.array([j0, j0, j0 + 512, j0 + 512, j0])
        la2, lo2 = geo.to_ground(ii, jj)
        fx, fy = rtransform('EPSG:4326', 'EPSG:32617', list(lo2), list(la2))
        # the reference, geocoded on a 1 m north-up grid: the SAR pixel nearest to every map point
        g = np.arange(-half, half, 1.0)
        X, Y = np.meshgrid(cx + g, cy - g)
        lon_g, lat_g = rtransform('EPSG:32617', 'EPSG:4326', X.ravel(), Y.ravel())
        pi, pj = geo.to_pixel(np.array(lat_g), np.array(lon_g))
        pi, pj = np.round(pi).astype(int), np.round(pj).astype(int)
        ok = (pi >= 0) & (pi < 12207) & (pj >= 0) & (pj < 8808)
        lo_i, hi_i = max(pi[ok].min(), 0), min(pi[ok].max() + 1, 12207)
        blk = np.abs(np.asarray(ref[lo_i:hi_i])) ** 2
        p = np.full(X.size, np.nan)
        p[ok] = blk[pi[ok] - lo_i, pj[ok]]
        p = p.reshape(X.shape)
        top = np.nanpercentile(p, 99.9)
        db = 10 * np.log10(np.clip(p / top, 1e-6, None))
        ax = row[0]
        ax.imshow(np.clip(db, -45, 0), cmap='gray', vmin=-45, vmax=0, extent=[cx - half, cx + half, cy - half, cy + half])
        ax.set_title(f'{TITLES[name]}: SAR, float64 reference', fontsize=9)
        for ax, key, label in ((row[1], 'S2B_20230723', 'Sentinel-2, 2023-07-23'), (row[2], 'S2A_20240223', 'Sentinel-2, 2024-02-23')):
            d = s2[key]
            img = np.moveaxis(d['img'], 0, -1)
            ta, tb, tc, td, te, tf = d['transform']
            H, W = img.shape[:2]
            ax.imshow(img, extent=[tc, tc + ta * W, tf + te * H, tf])
            ax.set_title(f'{TITLES[name]}: {label}', fontsize=9)
        for ax in row:
            ax.plot(fx, fy, '-', color='#ff5a36', lw=1.2)
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].plot([cx - half + 60, cx - half + 260], [cy - half + 60] * 2, 'w-', lw=3)
        row[0].text(cx - half + 160, cy - half + 90, '200 m', color='w', ha='center', fontsize=7)
    fig.subplots_adjust(wspace=0.04, hspace=0.12, left=0.02, right=0.98, top=0.97, bottom=0.01)
    fig.savefig(a.out, dpi=200)
    print('written', a.out)




def channel_series(out='paper/fig/channel_panama.pdf'):
    """The channel target through time: the vendor's geocoded products (GEC) of six Umbra passes and the Sentinel-2 image
    of 2023-07-23, all warped to one north-up 1 m grid (results/fastsar/figs/gec_channel_windows.npz, made with
    rasterio WarpedVRT from the open catalog)."""
    d = np.load(f'{R}/figs/gec_channel_windows.npz')
    ext = d['extent']
    fx, fy = d['fx'], d['fy']
    order = [('2023-07-18-02-30-32', 'Umbra, 2023-07-18 02:30 UTC (this study)'), ('2023-07-21-15-13-56', 'Umbra, 2023-07-21 15:14'),
             ('2023-07-22-15-00-27', 'Umbra, 2023-07-22 15:00'), ('2023-07-23-14-47-14', 'Umbra, 2023-07-23 14:47'),
             ('S2B_20230723', 'Sentinel-2, 2023-07-23 15:51'), ('2023-07-25-02-30-32', 'Umbra, 2023-07-25 02:30')]
    s2 = np.load(f'{R}/figs/s2_S2B_20230723.npz')
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6))
    for ax, (key, title) in zip(axes.ravel(), order):
        if key.startswith('S2'):
            img = np.moveaxis(s2['img'], 0, -1)
            ta, tb, tc, td, te, tf = s2['transform']
            H, W = img.shape[:2]
            ax.imshow(img, extent=[tc, tc + ta * W, tf + te * H, tf])
        else:
            img = d[key].astype(np.float32)
            ax.imshow(img, cmap='gray', vmin=0, vmax=np.percentile(img, 99.7), extent=ext)
        ax.plot(fx, fy, '-', color='#ff5a36', lw=1.0)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8.5)
    axes[0, 0].plot([ext[0] + 60, ext[0] + 260], [ext[2] + 60] * 2, 'w-', lw=3)
    axes[0, 0].text(ext[0] + 160, ext[2] + 90, '200 m', color='w', ha='center', fontsize=7)
    fig.subplots_adjust(wspace=0.04, hspace=0.14, left=0.02, right=0.98, top=0.95, bottom=0.02)
    fig.savefig(out, dpi=200)
    print('written', out)


if __name__ == '__main__':
    main()
    channel_series()
