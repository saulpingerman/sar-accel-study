#!/usr/bin/env python3
"""Figures for the paper from results/fastsar (release records of FastSAR and the historical files kept there).

  python paper/figs.py
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'paper')
from build import PRICE, DEV, SCENE, config_name, rows_of, load_metrics  # noqa: E402

R = 'results/fastsar'
FIG = 'paper/fig'
COL = {'tpu-v5e': '#1b6ca8', 'tpu-v6e': '#0b3d91', 'gpu-l4': '#2e8b57', 'cpu-c4d16': '#b5651d'}
MARK = {'bp': 'o', 'ffbp': 's', 'pfa': '^'}


def teaser(m):
    """The key figure: cost against error for every selected configuration; only the Pareto front is labeled."""
    rows = rows_of(m, 'panama')
    timing = {lab: json.load(open(f'{R}/timing/panama_{lab}.json')) for lab in ('gpu-l4', 'tpu-v6e', 'tpu-v5e', 'cpu-c4d16')}
    # (label, timing tag, metrics tag, filled, short name = key in pareto_regions_v3.json)
    record = [('gpu-l4', 'ffbp/f16_cuda', 'ffbp/f16_cuda', True, 'L4 float16'), ('gpu-l4', 'ffbp/fp32_cuda', 'ffbp/fp32_cuda', True, 'L4 float32'),
              ('gpu-l4', 'pfa/fp32_taps_corr', 'pfa/fp32_taps_corr', True, 'L4 polar format'), ('gpu-l4', 'bp/cubic', 'bp/cubic', True, 'FastSAR exact, L4'),
              ('tpu-v6e', 'ffbp/fp32_fast_pallas2_direct', 'ffbp/fp32_fast_pallas2_direct', True, 'v6e single-pass'), ('tpu-v6e', 'ffbp/fp32_high_pallas2_direct', 'ffbp/fp32_high_pallas2_direct', True, 'v6e three-pass'),
              ('tpu-v6e', 'pfa/fp32_taps_corr', 'pfa/fp32_taps_corr', True, 'v6e polar format'),
              ('tpu-v5e', 'ffbp/fp32_fast_pallas2_direct', 'ffbp/fp32_fast_pallas2_direct', True, 'v5e single-pass'), ('tpu-v5e', 'ffbp/fp32_high_pallas2_direct', 'ffbp/fp32_high_pallas2_direct', True, 'v5e three-pass'),
              ('tpu-v5e', 'pfa/fp32_taps_corr', 'pfa/fp32_taps_corr', True, 'v5e polar format'),
              ('cpu-c4d16', 'ffbp/fp32_cpp', 'ffbp/fp32_cpp', True, 'CPU float32'), ('cpu-c4d16', 'pfa/fp32_taps_corr', 'pfa/fp32_taps_corr', True, 'CPU polar format'),
              ('cpu-c4d16', 'bp/cubic', 'bp/cubic', True, 'FastSAR exact, CPU')]
    pts = []
    for lab, ttag, mtag, filled, name in record:
        t = timing[lab].get(ttag)
        r = rows.get((lab, mtag))
        if not t or not r:
            continue
        st = t.get('stream') or {}
        per = st.get('s_per_image') or t.get('run_s')
        usd = PRICE[lab] / 3600.0 * per * 1000
        pts.append(dict(lab=lab, alg=ttag.split('/')[0], filled=filled, name=name, x=usd, y=r['err_db'], mtag=mtag))
    # errors over the three Panama regions for every point (the open-source implementations were scored there only)
    # FastSAR: errors over the three regions against the float64 computation at 64 times (truth64); the open-source
    # codes: against the reference (pareto_regions_v3.json), whose own error is far below theirs
    reg = json.load(open(f'{R}/pareto_regions_v3.json'))['configurations']
    tr = json.load(open(f'{R}/truth64/score_truth64.json'))
    E = tr['_energy']
    tkey = {'L4 float16': 'gpu-l4_ffbp_f16_cuda', 'L4 float32': 'gpu-l4_ffbp_fp32_cuda', 'L4 polar format': 'gpu-l4_pfa_fp32_taps_corr',
            'FastSAR exact, L4': 'gpu-l4_bp_cubic8', 'FastSAR exact, CPU': 'cpu-c4d16_bp_cubic8',
            'v6e single-pass': 'tpu-v6e_ffbp_fp32_fast_pallas2_direct', 'v6e three-pass': 'tpu-v6e_ffbp_fp32_high_pallas2_direct',
            'v6e polar format': 'tpu-v6e_pfa_fp32_taps_corr', 'v5e single-pass': 'tpu-v5e_ffbp_fp32_fast_pallas2_direct',
            'v5e three-pass': 'tpu-v5e_ffbp_fp32_high_pallas2_direct', 'v5e polar format': 'tpu-v5e_pfa_fp32_taps_corr',
            'CPU float32': 'cpu-c4d16_ffbp_fp32_cpp', 'CPU polar format': 'cpu-c4d16_pfa_fp32_taps_corr'}
    pts = [p for p in pts if tkey.get(p['name']) in tr]
    for p in pts:
        v = tr[tkey[p['name']]]
        p['y'] = float(10 * np.log10(sum(10 ** (v[r] / 10) * E[r] for r in E) / sum(E.values())))
    oss = json.load(open(f'{R}/oss_times.json'))
    for name, lab, alg, secs, key, gray in oss:
        pts.append(dict(lab=lab, alg=alg, filled=True, name=name, x=PRICE[lab] / 3600.0 * secs * 1000, y=reg[key]['pooled_db'], oss=gray))
    for p in pts:
        p['front'] = not any(q is not p and q['x'] <= p['x'] and q['y'] <= p['y'] + 0.05 and (q['x'] < p['x'] or q['y'] < p['y'] - 0.05) for q in pts)
    front = sorted((p for p in pts if p['front']), key=lambda p: p['x'])
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    # dominated region, lightly shaded above and to the right of the front
    fx, fy = [], []
    for k, p in enumerate(front):
        if k:
            fx.append(p['x']); fy.append(front[k - 1]['y'])
        fx.append(p['x']); fy.append(p['y'])
    fx.append(2e5); fy.append(front[-1]['y'])
    ax.fill_between(fx, fy, 10, step=None, color='0.93', zorder=0)
    ax.plot(fx, fy, color='0.4', lw=1.1, zorder=2)
    top = -20.0
    for p in pts:
        if p.get('oss'):
            y = min(p['y'], top)
            ax.scatter(p['x'], y, s=46, marker='X' if p['y'] > top else 'D', c='0.55', edgecolor='0.25', linewidth=0.7, zorder=3)
            if p['y'] > top:
                ax.annotate(p['name'] + '\n(no image)', (p['x'], y), xytext=(0, -7), textcoords='offset points', fontsize=6.5, color='0.25', ha='center', va='top')
            else:
                ax.annotate(p['name'], (p['x'], y), xytext=(-7, 0), textcoords='offset points', fontsize=6.5, color='0.25', ha='right', va='center')
            continue
        kw = dict(marker=MARK[p['alg']], linewidth=0.9, edgecolor=COL[p['lab']], zorder=3)
        if p['front']:
            ax.scatter(p['x'], p['y'], s=70, c=COL[p['lab']] if p['filled'] else 'white', **kw)
        else:
            ax.scatter(p['x'], p['y'], s=36, c=COL[p['lab']] if p['filled'] else 'white', alpha=0.35, **kw)
    # labels for the Pareto front, in the empty margin left of and below the front, with thin leader lines
    place = {'L4 polar format': (0.9, -34.5, 'left'), 'L4 float16': (0.52, -55.5, 'right'), 'L4 float32': (0.52, -64.5, 'right'),
             'FastSAR exact, L4': (5.0, -84.0, 'left')}
    names = {'FastSAR exact, L4': 'L4 exact BP'}
    for p in front:
        if p['name'] in place:
            x, y, ha = place[p['name']]
            ax.annotate(names.get(p['name'], p['name']), (p['x'], p['y']), xytext=(x, y), textcoords='data', fontsize=7.5,
                        color=COL[p['lab']], ha=ha, va='center', fontweight='medium',
                        arrowprops=dict(arrowstyle='-', color=COL[p['lab']], lw=0.6, shrinkA=1, shrinkB=4))
    # worst FastSAR configuration of each regime of visible change, on the three-region axis (pareto_regions.json)
    def regime(r):
        return 3 if r['coh_p001'] < 0.99 else (2 if r['coh_p001'] < 0.999 else 1)
    worst = {}
    for p in pts:
        if p.get('oss') is None and p.get('mtag'):
            k = regime(rows[(p['lab'], p['mtag'])])
            worst[k] = max(worst.get(k, -1e9), p['y'])
    lines_ = [(worst[k], t, v) for k, t, v in ((1, 'regime 1: no visible change', 'top'), (2, 'regime 2: coherence loss beside bright returns', 'bottom'),
                                              (3, 'regime 3: striped coherence loss', 'bottom')) if k in worst]
    for y, txt, va in lines_:
        ax.text(1.8e5, y - 0.6 if va == 'top' else y + 0.4, txt, fontsize=7, color='0.3', ha='right', va=va)
        ax.axhline(y, color='0.75', lw=0.5, ls=':', zorder=1)
    ax.text(0.13, -28.0, 'FastSAR\n(color = device)', fontsize=8.5, fontweight='bold', color='#2e5e3e', ha='left', va='center')
    ax.text(1.5e3, -23.5, 'open-source implementations (gray)', fontsize=8.5, fontweight='bold', color='0.3', ha='center', va='bottom')
    ax.set_xscale('log')
    ax.set_xlim(0.1, 2e5)
    ax.set_ylim(-86, top + 1)
    ax.set_xlabel('Cost per 1000 Panama images (US dollars, on-demand us-central1)')
    ax.set_ylabel('Error over three regions relative to the float64 image (dB)')
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(facecolor=COL[k], edgecolor=COL[k], label='FastSAR, ' + ('GPU L4' if k == 'gpu-l4' else DEV[k])) for k in COL]      # color = device (any marker)
    h += [Line2D([], [], marker=MARK[a], color='w', markerfacecolor='#2e8b57', markeredgecolor='#2e8b57', markersize=7, label=f'FastSAR {n}') for a, n in (('bp', 'exact BP'), ('ffbp', 'factorized BP'), ('pfa', 'polar format'))]
    h += [Line2D([], [], marker='D', color='w', markerfacecolor='0.55', markeredgecolor='0.25', markersize=6, label='open source'), Line2D([], [], marker='X', color='w', markerfacecolor='0.55', markeredgecolor='0.25', markersize=7, label='open source, no image')]
    h += [Line2D([], [], color='0.4', lw=1.1, label='Pareto front of FastSAR')]
    ax.legend(handles=h, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=4, framealpha=0.95)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{FIG}/teaser.pdf')
    plt.close(fig)


def db(x):
    return 20 * np.log10(np.abs(x) + 1e-12)


def sicd_fig():
    """Three Panama regions: the reference, the vendor image as delivered, the vendor image resampled through the
    displacement model, and the coherence of the last with the reference."""
    p, q = f'{R}/sicd/sicd4_panama_crops.npz', f'{R}/sicd/sicd_panama_raw_crops.npz'
    if not (os.path.exists(p) and os.path.exists(q)):
        return
    d, e = np.load(p), np.load(q)
    regions = [n for n in ('port', 'locks', 'ships') if f'{n}/ref' in d and f'{n}/raw' in e]
    if not regions:
        return
    fig, axes = plt.subplots(len(regions), 4, figsize=(7.2, 1.95 * len(regions)), squeeze=False)
    titles = ('reference', 'vendor image as delivered', 'vendor image, resampled', 'coherence (0 to 1)')
    for i, n in enumerate(regions):
        a, raw, b, c = d[f'{n}/ref'], e[f'{n}/raw'], d[f'{n}/sicd'], d[f'{n}/coh']
        ref_db = 20 * np.log10(np.abs(a) + 1e-12); top = np.percentile(ref_db, 99.9)
        for j, img in enumerate((a, raw, b)):
            g = 1.0 if j == 0 else float(np.sqrt((np.abs(img) ** 2).mean() / (np.abs(a) ** 2).mean()))
            db = 20 * np.log10(np.abs(img) / g + 1e-12)
            axes[i, j].imshow(np.clip(db.T, top - 45, top), cmap='gray', vmin=top - 45, vmax=top, origin='lower', interpolation='nearest')
        axes[i, 3].imshow(c.T, cmap='viridis', vmin=0, vmax=1, origin='lower', interpolation='nearest')
        axes[i, 3].set_xlabel(f'mean {c.mean():.2f}', fontsize=6.5, labelpad=1)
        axes[i, 0].set_ylabel({'port': 'Port', 'locks': 'Locks', 'ships': 'Ship'}[n], fontsize=8)
        for j, ax in enumerate(axes[i]):
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=7.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(f'{FIG}/sicd_panama.pdf')
    plt.close(fig)


def zoom(scene, picks, names):
    """Zoom crops: rows = regions, columns = reference and chosen test images (amplitude), then coherence maps."""
    p = f'{R}/figs/{scene}_crops.npz'
    if not os.path.exists(p):
        return
    d = np.load(p)
    regions = [n for n in names if f'{n}/ref' in d]
    cols = [('ref', 'float64 exact BP')] + picks
    nr, nc = len(regions), len(cols)
    fig, axes = plt.subplots(nr * 2, nc, figsize=(0.95 * nc, 0.95 * nr * 2 + 1.1))
    axes = np.atleast_2d(axes)
    im_amp = None
    for i, reg in enumerate(regions):
        ref = d[f'{reg}/ref']
        top = np.percentile(db(ref), 99.9)
        for j, (key, title) in enumerate(cols):
            k = f'{reg}/{key}' if key != 'ref' else f'{reg}/ref'
            if k not in d:
                axes[2 * i, j].axis('off'); axes[2 * i + 1, j].axis('off')
                continue
            img = d[k]
            im_amp = axes[2 * i, j].imshow(np.clip(db(img) - top, -45, 0).T, cmap='gray', vmin=-45, vmax=0, origin='lower')
            axes[2 * i, j].set_title(title if i == 0 else '', fontsize=5.5)
            axes[2 * i, j].set_xticks([]); axes[2 * i, j].set_yticks([])
            if j == 0:
                axes[2 * i, j].set_ylabel({'ships': 'ship', 'corner': 'vegetation'}.get(reg, reg), fontsize=7)
                axes[2 * i + 1, j].axis('off')
                bar = 100.0 / 0.4129                                  # 100 m along azimuth (x axis of the display)
                axes[2 * i, j].plot([20, 20 + bar], [20, 20], color='white', lw=2)
                axes[2 * i, j].text(20 + bar / 2, 36, '100 m', color='white', fontsize=5, ha='center')
            else:
                coh = d[f'{k}/coh']
                im = axes[2 * i + 1, j].imshow(coh.T, cmap='Reds_r', vmin=0.99, vmax=1.0, origin='lower')
                axes[2 * i + 1, j].set_xticks([]); axes[2 * i + 1, j].set_yticks([])
                axes[2 * i + 1, j].set_xlabel(f'{coh.mean():.4f}', fontsize=5.5, labelpad=1)
    fig.subplots_adjust(wspace=0.04, hspace=0.12, bottom=0.07)
    cax1 = fig.add_axes([0.13, 0.035, 0.32, 0.012])
    cb1 = fig.colorbar(im_amp, cax=cax1, orientation='horizontal', ticks=[-45, -30, -15, 0])
    cb1.set_label('amplitude (dB re brightest 0.1%)', fontsize=6)
    cb1.ax.tick_params(labelsize=5)
    cax2 = fig.add_axes([0.55, 0.035, 0.32, 0.012])
    cb2 = fig.colorbar(im, cax=cax2, orientation='horizontal', extend='min', ticks=[0.99, 0.995, 1.0])
    cb2.set_label('5 x 5 coherence with the reference', fontsize=6)
    cb2.ax.tick_params(labelsize=5)
    fig.savefig(f'{FIG}/zoom_{scene}.pdf', bbox_inches='tight')
    plt.close(fig)


def pixel_zoom(scene, picks, names, win=128, up=4):
    """Pixel-scale windows (win by win pixels, nearest-neighbour, each pixel an up by up block) around the brightest
    feature of each region: amplitude rows and coherence rows as in zoom()."""
    from scipy.ndimage import uniform_filter
    p = f'{R}/figs/{scene}_crops.npz'
    if not os.path.exists(p):
        return
    d = np.load(p)
    regions = [n for n in names if f'{n}/ref' in d]
    cols = [('ref', 'float64 exact BP')] + picks
    nr, nc = len(regions), len(cols)
    fig, axes = plt.subplots(nr * 2, nc, figsize=(1.08 * nc, 1.08 * nr * 2 + 1.1))
    axes = np.atleast_2d(axes)
    im_amp = im = None
    for i, reg in enumerate(regions):
        ref = d[f'{reg}/ref']
        if reg == 'corner':
            ci, cj = ref.shape[0] // 2, ref.shape[1] // 2
        else:
            sm = uniform_filter(np.abs(ref) ** 2, 48)
            ci, cj = np.unravel_index(np.argmax(sm), sm.shape)
        i0 = int(np.clip(ci - win // 2, 0, ref.shape[0] - win))
        j0 = int(np.clip(cj - win // 2, 0, ref.shape[1] - win))
        sl = (slice(i0, i0 + win), slice(j0, j0 + win))
        top = np.percentile(db(ref[sl]), 99.9)
        for j, (key, title) in enumerate(cols):
            k = f'{reg}/{key}' if key != 'ref' else f'{reg}/ref'
            if k not in d:
                axes[2 * i, j].axis('off'); axes[2 * i + 1, j].axis('off')
                continue
            img = np.kron(np.clip(db(d[k][sl]) - top, -45, 0).T, np.ones((up, up)))
            im_amp = axes[2 * i, j].imshow(img, cmap='gray', vmin=-45, vmax=0, origin='lower', interpolation='nearest')
            axes[2 * i, j].set_title(title if i == 0 else '', fontsize=5.5)
            axes[2 * i, j].set_xticks([]); axes[2 * i, j].set_yticks([])
            if j == 0:
                axes[2 * i, j].set_ylabel({'ships': 'ship', 'corner': 'vegetation'}.get(reg, reg), fontsize=7)
                axes[2 * i + 1, j].axis('off')
                bar = up * 10.0 / 0.4129                                  # 10 m along azimuth
                axes[2 * i, j].plot([2 * up, 2 * up + bar], [2 * up, 2 * up], color='white', lw=2)
                axes[2 * i, j].text(2 * up + bar / 2, 5 * up, '10 m', color='white', fontsize=5, ha='center')
            else:
                coh = np.kron(d[f'{k}/coh'][sl].T, np.ones((up, up)))
                im = axes[2 * i + 1, j].imshow(coh, cmap='Reds_r', vmin=0.99, vmax=1.0, origin='lower', interpolation='nearest')
                axes[2 * i + 1, j].set_xticks([]); axes[2 * i + 1, j].set_yticks([])
                axes[2 * i + 1, j].set_xlabel(f'{d[f"{k}/coh"][sl].mean():.4f}', fontsize=5.5, labelpad=1)
    fig.subplots_adjust(wspace=0.04, hspace=0.12, bottom=0.09)
    cax1 = fig.add_axes([0.13, 0.045, 0.32, 0.012])
    cb1 = fig.colorbar(im_amp, cax=cax1, orientation='horizontal', ticks=[-45, -30, -15, 0])
    cb1.set_label('amplitude (dB re brightest 0.1%)', fontsize=6)
    cb1.ax.tick_params(labelsize=5)
    cax2 = fig.add_axes([0.55, 0.045, 0.32, 0.012])
    cb2 = fig.colorbar(im, cax=cax2, orientation='horizontal', extend='min', ticks=[0.99, 0.995, 1.0])
    cb2.set_label('5 x 5 coherence with the reference', fontsize=6)
    cb2.ax.tick_params(labelsize=5)
    fig.savefig(f'{FIG}/pixels_{scene}.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)


def profile():
    files = sorted(glob.glob(f'{R}/logs/profile_*.json'))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(8, 3.2))
    names = ['ramp_direct', 'ramp_split', 'rotate_direct', 'rotate_split', 'dec_k_dense', 'dec_k_conv', 'dec_k_banded', 'dec_p_dense', 'dec_p_conv', 'dec_p_banded']
    w = 0.8 / max(1, len(files))
    for i, f in enumerate(files):
        d = json.load(open(f))
        st = {(s['policy'], s['stage']): s['seconds'] for s in d['stages']}
        vals = [st.get(('fp32_fast', n)) or st.get(('f16', n)) for n in names]
        ax.bar(np.arange(len(names)) + i * w, [v * 1e3 if v else 0 for v in vals], w, label=DEV.get(d['label'], d['label']), color=COL.get(d['label']))
    ax.set_xticks(np.arange(len(names)) + 0.4 - w / 2)
    ax.set_xticklabels([n.replace('_', ' ') for n in names], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('ms per first-level tile')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y', which='both')
    fig.tight_layout()
    fig.savefig(f'{FIG}/profile.pdf')
    plt.close(fig)


def scenes():
    """The three reference images, downsampled, with the Panama zoom regions marked and a 1 km bar."""
    from matplotlib.patches import Rectangle
    specs = [('panama', 'Panama Canal', 8, (0.4129, 0.3556), {'locks': (3400, 6700), 'port': (9500, 1150), 'ships': (6400, 2650), 'corner': (11400, 8000)}),
             ('melbourne', 'Melbourne', 7, (0.4223, 0.2983), {}), ('iowa', 'Iowa farmland', 6, (0.5166, 0.3278), {})]
    fig = plt.figure(figsize=(10, 10.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[6.2, 3.4], hspace=0.08, wspace=0.06)
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, (s, title, dn, (spx, spy), boxes) in zip(axes, specs):
        p = f'{R}/figs/{s}_overview_db.npy'
        if not os.path.exists(p):
            ax.axis('off'); continue
        ov = np.load(p)
        ax.imshow(np.clip(ov, -45, 0).T, cmap='gray', vmin=-45, vmax=0, origin='lower', aspect=spy / spx)
        for name, (i0, j0) in boxes.items():
            ax.add_patch(Rectangle((i0 / dn, j0 / dn), 512 / dn, 512 / dn, fill=False, color='#ffd400', lw=1.8))
            label = {'ships': 'ship', 'corner': 'vegetation'}.get(name, name)
            # label beside the box, to the left for boxes near the right edge, on a dark pad so it reads on any background
            right = i0 / dn + 512 / dn + 40 + 9 * len(label) < ov.shape[0]
            tx = i0 / dn + 512 / dn + 10 if right else i0 / dn - 10
            ax.text(tx, j0 / dn + 256 / dn, label, color='#ffd400', fontsize=13, fontweight='bold', va='center', ha='left' if right else 'right',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.6, edgecolor='none'))
        bar = 1000.0 / (spx * dn)
        ax.plot([ov.shape[0] * 0.05, ov.shape[0] * 0.05 + bar], [ov.shape[1] * 0.05] * 2, color='white', lw=3)
        ax.text(ov.shape[0] * 0.05 + bar / 2, ov.shape[1] * 0.05 + ov.shape[1] * 0.02, '1 km', color='white', ha='center', fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel('azimuth', fontsize=9); ax.set_ylabel('range', fontsize=9)
    fig.savefig(f'{FIG}/scenes.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)


def rings(m):
    """Error by distance from the scene center for polar format and the float32 factorized image, three scenes."""
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = [0.125, 0.375, 0.625, 0.875]
    style = {'panama': '-', 'melbourne': '--', 'iowa': ':'}
    for s in SCENE:
        rows = rows_of(m, s)
        for (lab, tag), r in rows.items():
            if tag == 'pfa/fp32_taps_corr' and r.get('rings') and lab == 'gpu-l4':
                ax.plot(x, r['rings'], marker='o', color='#c0392b', linestyle=style[s], label=f'{SCENE[s]}, polar format')
            if tag == 'ffbp/fp32_cuda' and r.get('rings') and lab == 'gpu-l4':
                ax.plot(x, r['rings'], marker='s', color='#2c3e50', linestyle=style[s], label=f'{SCENE[s]}, factorized, float32')
    ax.set_xlabel('Distance from the scene center (fraction of the half-width)')
    ax.set_ylabel('Error relative to float64 (dB)')
    ax.set_xticks(x)
    ax.legend(fontsize=6.5, ncol=2, loc='center left')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{FIG}/rings.pdf')
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    m = load_metrics()
    if 'panama' in m:
        teaser(m)
        rings(m)
    scenes()
    profile()
    sicd_fig()
    zoom('panama', [('gpu-l4/bp_cubic', 'L4 exact BP'), ('gpu-l4/ffbp_f16_cuda', 'L4 factorized, fp16'), ('tpu-v6e/ffbp_fp32_fast_pallas2_direct', 'TPU v6e, 1 pass'), ('tpu-v6e/ffbp_fp32_high_pallas2_direct', 'TPU v6e, 3 passes'), ('gpu-l4/pfa_fp32_taps_corr', 'polar format')],
         ['locks', 'port', 'ships', 'corner'])
    pixel_zoom('panama', [('gpu-l4/ffbp_f16_cuda', 'L4 factorized, fp16'), ('tpu-v6e/ffbp_fp32_fast_pallas2_direct', 'TPU v6e, 1 pass'), ('tpu-v6e/ffbp_fp32_high_pallas2_direct', 'TPU v6e, 3 passes'), ('gpu-l4/pfa_fp32_taps_corr', 'polar format')],
               ['locks', 'port', 'ships'])
    print('figures written')
