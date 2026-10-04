#!/usr/bin/env python3
"""Figure: the ship before and after moving-target compensation (tiles from ship_refocus.py).

  uv run --with rasterio --with numpy --with matplotlib --with scipy python v2_ship_fig.py --tiles <dir>
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rasterio.warp import transform as rtransform
from geo_panama import Geo

R = 'results/v2r'


def geocode(tile, i0, j0, geo, cx, cy, half, res=1.0):
    g = np.arange(-half, half, res)
    X, Y = np.meshgrid(cx + g, cy - g)
    lon, lat = rtransform('EPSG:32617', 'EPSG:4326', X.ravel(), Y.ravel())
    pi, pj = geo.to_pixel(np.array(lat), np.array(lon))
    pi, pj = np.round(pi).astype(int) - i0, np.round(pj).astype(int) - j0
    ok = (pi >= 0) & (pi < tile.shape[0]) & (pj >= 0) & (pj < tile.shape[1])
    out = np.full(X.size, np.nan)
    out[ok] = tile[pi[ok], pj[ok]]
    return out.reshape(X.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles', required=True)
    ap.add_argument('--out', default='report/v2/fig/ship_panama.pdf')
    a = ap.parse_args()
    geo = Geo(f'{R}/panama_sicd.xml', 12207, 8808)
    i0, i1, j0, j1 = json.load(open(f'{R}/ship/sweep_158_coarse.json'))['tile']
    if a.tiles == 'results':                                   # the stored sub-tiles (i 6300 to 7500)
        t0 = np.load(f'{R}/ship/tiles/tile_+0.00.npy'); t3 = np.load(f'{R}/ship/tiles/tile_+3.00.npy'); i0 += 1000
    else:
        t0 = np.load(f'{a.tiles}/tile_+0.00.npy'); t3 = np.load(f'{a.tiles}/best/tile_+3.00.npy')
    P0, P3 = np.abs(t0) ** 2, np.abs(t3) ** 2
    top = np.percentile(P3, 99.99)
    # window centred between the two positions of the ship
    pk0 = np.unravel_index(np.argmax(P0), P0.shape)
    pk3 = np.unravel_index(np.argmax(P3), P3.shape)
    ci, cj = (pk0[0] + pk3[0]) / 2 + i0, (pk0[1] + pk3[1]) / 2 + j0
    lat, lon = geo.to_ground(ci, cj)
    (cx,), (cy,) = rtransform('EPSG:4326', 'EPSG:32617', [lon], [lat])
    half = 380
    s2 = np.load(f'{R}/figs/s2_S2A_20240223.npz')
    img = np.moveaxis(s2['img'], 0, -1)
    ta, tb, tc, td, te, tf = s2['transform']
    H, W = img.shape[:2]
    fig = plt.figure(figsize=(9.6, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.78], hspace=0.16, wspace=0.12)
    ext = [cx - half, cx + half, cy - half, cy + half]
    for k, (P, title) in enumerate(((P0, 'as formed'), (P3, 'compensated, 3.0 m/s along the channel'))):
        ax = fig.add_subplot(gs[0, k])
        db = 10 * np.log10(np.clip(geocode(P, i0, j0, geo, cx, cy, half) / top, 1e-6, None))
        ax.imshow(np.clip(db, -45, 0), cmap='gray', vmin=-45, vmax=0, extent=ext)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(img, extent=[tc, tc + ta * W, tf + te * H, tf])
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title('Sentinel-2, 2024-02-23', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    # the ship's positions on the optical image
    for pk, mk, lab in ((pk0, 'x', 'as formed'), (pk3, '+', 'compensated')):
        la, lo = geo.to_ground(pk[0] + i0, pk[1] + j0)
        (x,), (y,) = rtransform('EPSG:4326', 'EPSG:32617', [lo], [la])
        ax.plot(x, y, mk, color='#ff5a36', ms=10, mew=2, label=lab)
    # heading arrow at the compensated position: 3.0 m/s toward 158 degrees, drawn 150 m long
    la, lo = geo.to_ground(pk3[0] + i0, pk3[1] + j0)
    (x3,), (y3,) = rtransform('EPSG:4326', 'EPSG:32617', [lo], [la])
    br = np.radians(158.0)
    ax.annotate('', xy=(x3 + 150 * np.sin(br), y3 + 150 * np.cos(br)), xytext=(x3, y3),
                arrowprops=dict(arrowstyle='-|>', color='#ff5a36', lw=1.8, mutation_scale=12))
    ax.text(x3 + 165 * np.sin(br) + 15, y3 + 165 * np.cos(br), '3.0 m/s\ntoward 158\u00b0', color='#ff5a36', fontsize=7, va='center')
    # the displacement, from the stationary-scene position to the compensated one
    la, lo = geo.to_ground(pk0[0] + i0, pk0[1] + j0)
    (x0,), (y0,) = rtransform('EPSG:4326', 'EPSG:32617', [lo], [la])
    ax.annotate('', xy=(x3, y3), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color='white', lw=1.0, ls='--', mutation_scale=9))
    ax.text((x0 + x3) / 2 + 10, (y0 + y3) / 2 + 25, '194 m', color='white', fontsize=7)
    ax.legend(fontsize=7, loc='lower left', framealpha=0.8)
    for ax in fig.axes[:1]:
        ax.plot([ext[0] + 40, ext[0] + 240], [ext[2] + 40] * 2, 'w-', lw=3)
        ax.text(ext[0] + 140, ext[2] + 65, '200 m', color='w', ha='center', fontsize=7)
    # pixel-scale windows around the peak, image coordinates, nearest neighbour
    win, up = 160, 3
    for k, (P, pk, title) in enumerate(((P0, pk0, 'as formed, 160 by 160 pixels'), (P3, pk3, 'compensated, 160 by 160 pixels'))):
        ax = fig.add_subplot(gs[1, k])
        a0 = int(np.clip(pk[0] - win // 2, 0, P.shape[0] - win)); b0 = int(np.clip(pk[1] - win // 2, 0, P.shape[1] - win))
        w = P[a0:a0 + win, b0:b0 + win]
        db = 10 * np.log10(np.clip(w / top, 1e-6, None))
        ax.imshow(np.kron(np.clip(db, -45, 0).T, np.ones((up, up))), cmap='gray', vmin=-45, vmax=0, origin='lower', interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            bar = up * 20.0 / 0.4129
            ax.plot([2 * up, 2 * up + bar], [2 * up, 2 * up], 'w-', lw=2)
            ax.text(2 * up + bar / 2, 5 * up, '20 m', color='w', fontsize=7, ha='center')
    ax = fig.add_subplot(gs[1, 2])
    for f, st in (('sweep_158_coarse.json', 'o-'), ('sweep_158_fine.json', '.-')):
        rows = json.load(open(f'{R}/ship/{f}'))['rows']
        ax.plot([r['speed'] for r in rows], [r['sharpness'] for r in rows], st, color='k', ms=3, lw=0.8)
    rows = json.load(open(f'{R}/ship/sweep_148_coarse.json'))['rows']
    ax.plot([r['speed'] for r in rows], [r['sharpness'] for r in rows], 's--', color='0.5', ms=3, lw=0.8, label='heading 148$^\\circ$')
    ax.set_yscale('log'); ax.set_xlabel('speed along the channel toward 158$^\\circ$ (m/s)', fontsize=8); ax.set_ylabel('sharpness (sum of intensity squared)', fontsize=8, labelpad=1)
    ax.yaxis.set_label_position('right'); ax.yaxis.tick_right()
    ax.tick_params(labelsize=7); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.savefig(a.out, dpi=200, bbox_inches='tight')
    print('written', a.out, 'shift px', pk3[0] - pk0[0], pk3[1] - pk0[1], 'peak gain dB', 10 * np.log10(P3.max() / P0.max()))


if __name__ == '__main__':
    main()
