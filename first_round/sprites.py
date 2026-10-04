#!/usr/bin/env python3
"""Pack per-setting panels into one sheet per view, with an index, for the comparison page.

  python sprites.py --dir results/real/figs --prefix real --views zoom,err,ccd,ccddiff --tags ref,bp_fp32,... --out results/sheets
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--prefix', required=True, help='file prefix before the view name ("real" for real_zoom_x.png, "" for img_x.png)')
    ap.add_argument('--views', required=True)
    ap.add_argument('--tags', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cols', type=int, default=5)
    ap.add_argument('--size', type=int, default=0, help='resample tiles to this many pixels (0 = keep)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    index = dict(name=a.name, cols=a.cols, views={})
    pre = a.prefix + '_' if a.prefix else ''
    for view in a.views.split(','):
        tiles, tags = [], []
        for tag in a.tags.split(','):
            p = f'{a.dir}/{pre}{view}_{tag}.png'
            if not os.path.exists(p):
                continue
            im = mpimg.imread(p)[..., :3]
            if a.size and im.shape[0] != a.size:
                k = im.shape[0] // a.size
                if k >= 1:
                    n = a.size * k
                    im = im[:n, :n].reshape(a.size, k, a.size, k, 3).mean((1, 3))
            tiles.append(im)
            tags.append(tag)
        if not tiles:
            continue
        t = tiles[0].shape[0]
        rows = -(-len(tiles) // a.cols)
        sheet = np.zeros((rows * t, a.cols * t, 3), np.float32)
        for i, im in enumerate(tiles):
            r, c = divmod(i, a.cols)
            sheet[r * t:(r + 1) * t, c * t:(c + 1) * t] = im[:t, :t]
        fn = f'{a.name}_{view}.png'
        mpimg.imsave(f'{a.out}/{fn}', np.clip(sheet, 0, 1))
        index['views'][view] = dict(file=fn, tags=tags, tile=int(t), rows=rows)
        print(view, len(tiles), 'tiles of', t, 'px ->', fn, os.path.getsize(f'{a.out}/{fn}') // 1024, 'KB')
    json.dump(index, open(f'{a.out}/{a.name}.json', 'w'))


if __name__ == '__main__':
    main()
