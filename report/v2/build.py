#!/usr/bin/env python3
"""Build the second-round paper: tables and numbers from results/v2r, spliced into paper.src.tex, then pdflatex.

  python report/v2/build.py            (from the repository root; INTERIM=1 marks missing numbers as running)
"""
import glob
import json
import os
import re
import subprocess
import sys

R = 'results/v2r'
OUT = 'report/v2'
PRICE = {'tpu-v5e': 1.200, 'tpu-v6e': 2.700, 'gpu-l4': 0.707, 'cpu-c4d16': 0.964}
DEV = {'tpu-v5e': 'TPU v5e', 'tpu-v6e': 'TPU v6e', 'gpu-l4': 'L4', 'cpu-c4d16': 'CPU'}
SCENE = {'panama': 'Panama Canal', 'melbourne': 'Melbourne', 'iowa': 'Iowa farmland'}
num = {}
gen = {}


def fmt(v, d=1):
    return '--' if v is None else f'{v:.{d}f}'


def db(v, d=1):
    """A signed decibel value for a table cell, with a true minus sign."""
    return f'${v:.{d}f}$'


def load_metrics():
    """One record per scene; per-device files (<scene>_<label>_metrics.json) are merged."""
    m = {}
    for f in sorted(glob.glob(f'{R}/*_metrics.json')):
        s = os.path.basename(f).split('_')[0]
        d = json.load(open(f))
        if s in m:
            m[s]['rows'] += d['rows']
        else:
            m[s] = d
    return m


STALE = {('tpu-v5e', 'pfa/fp32'), ('tpu-v5e', 'pfa/fp32_fast'), ('tpu-v5e', 'pfa/fp32_corr')}


def rows_of(m, scene):
    """Metrics rows by (label, tag); rows from images of superseded runs are dropped."""
    out = {}
    for r in m.get(scene, {}).get('rows', []):
        if r.get('bad') or (r['label'], r['tag']) in STALE:
            continue
        out[(r['label'], r['tag'])] = r
    return out


def config_name(tag, lab=''):
    """Human name of a timed configuration: (algorithm, arithmetic). The JAX precision policies mean different
    things per device: the default precision of float32 operands is a single bfloat16 pass on a TPU and full
    float32 on the L4 (TF32 disabled)."""
    alg, rest = tag.split('/', 1)
    parts = rest.split('_')
    a = {'bp': 'Exact BP', 'ffbp': 'Factorized BP', 'pfa': 'Polar format'}[alg]
    if alg == 'bp':
        store = 'float16' if 'f16' in parts else 'float32'
        return a, f'{store} profiles' + (', 16x oversampled' if 'ov16' in parts else '')
    pol = parts[0] + ('_' + parts[1] if len(parts) > 1 and parts[1] in ('fast', 'high', 'mm') else '')
    words = {'conv': 'conv. filters', 'taps': 'tap sums' if alg == 'ffbp' else 'gather', 'direct': 'direct ramps', 'pad256': 'padded',
             'pallas': 'fused level kernel (v1)', 'pallas2': 'fused kernels', 'xlafinal': 'XLA final stage', 'final1': 'direct-trig final', 'cuda': 'CUDA kernels', 'cpp': 'C++ kernels'}
    extra = [words[x] for x in parts if x in words and not (x == 'direct' and 'pallas2' in parts)]   # the kernels build their own ramps
    tpu = lab.startswith('tpu')
    if alg == 'pfa':
        base = {'fp32': 'float32', 'fp32_fast': 'single-pass product' if tpu else 'TF32 product', 'f16': 'float16 product'}.get(pol, pol)
    else:
        base = {'fp64': 'float64', 'fp32': 'six-pass products' if tpu else 'float32', 'fp32_high': 'three-pass products',
                'fp32_fast': 'single-pass products' if tpu else 'TF32 products',
                'bf16_mm': 'single-pass, bfloat16 storage', 'f16': ('float16 throughout' if 'cuda' in parts else 'float16 products'), 'f16tc': 'float32, float16 final stage'}.get(pol, pol)
    return a, base + (', ' + ', '.join(extra) if extra else '')


def gen_power(num):
    """Energy per Panama image: measured on the L4, bounded from design power on the TPUs, estimated on the CPU."""
    rows = [('L4', 'Factorized BP, float16 throughout, CUDA kernels', 'measured', num.get('pw.l4.ffbp.w', ''), num.get('c.panama.gpu-l4.ffbp.f16_conv.s', ''), num.get('pw.l4.ffbp.kj', '')),
            ('L4', 'Factorized BP, float32, CUDA kernels', 'measured', num.get('pw.l4.ffbp32.w', ''), num.get('c.panama.gpu-l4.ffbp.fp32_conv.s', ''), num.get('pw.l4.ffbp32.kj', '')),
            ('L4', 'Exact BP, float16 profiles', 'measured', num.get('pw.l4.bp.w', ''), num.get('c.panama.gpu-l4.bp.cuda_f16.s', ''), num.get('pw.l4.bp.kj', '')),
            ('TPU v5e', 'Factorized BP, single-pass, fused kernels', 'design-power bound', '120 to 200', num.get('c.panama.tpu-v5e.ffbp.fp32_fast_direct.s', ''), f"{num.get('pw.v5e.lo', '')} to {num.get('pw.v5e.hi', '')}"),
            ('TPU v5e', 'Factorized BP, three-pass, fused kernels', 'design-power bound', '120 to 200', num.get('c.panama.tpu-v5e.ffbp.fp32_high_direct.s', ''), f"{num.get('pw.v5e.3.lo', '')} to {num.get('pw.v5e.3.hi', '')}"),
            ('TPU v6e', 'Factorized BP, single-pass, fused kernels', 'design-power bound', '200 to 350', num.get('c.panama.tpu-v6e.ffbp.fp32_fast_direct.s', ''), f"{num.get('pw.v6e.lo', '')} to {num.get('pw.v6e.hi', '')}"),
            ('TPU v6e', 'Factorized BP, three-pass, fused kernels', 'design-power bound', '200 to 350', num.get('c.panama.tpu-v6e.ffbp.fp32_high_direct.s', ''), f"{num.get('pw.v6e.3.lo', '')} to {num.get('pw.v6e.3.hi', '')}"),
            ('CPU', 'Factorized BP, float32, C++ kernels', 'pro-rata estimate', f"25 at {num.get('pw.cpu.occ', '')}\\%", num.get('c.panama.cpu-c4d16.ffbp.fp32_cpp.s', ''), num.get('pw.cpu.kj', ''))]
    out = [r'\begin{table}[tb]', r'\centering',
           r'\caption{Energy per Panama image for the processor alone. The L4 draw is the mean of \texttt{nvidia-smi} samples during the pipelined loop; the TPU figures are bounds from the design-power figures third parties quote~\cite{introl2025,gpuadvisor2025}, since Google publishes none, assuming the chip ran at that power throughout; the CPU figure is the pro-rata share of the 8 cores (25~W of the 400~W package) times their occupancy, the busy fraction of the 16 hardware threads doubled and capped at one. Hosts, memory and the other components of each instance are excluded.}',
           r'\label{tab:power}', r'\small', r'\setlength{\tabcolsep}{4pt}', r'\resizebox{\textwidth}{!}{\begin{tabular}{lllrrr}', r'\toprule',
           r'Device & Configuration & Basis & Power (W) & Time (s) & Energy (kJ) \\', r'\midrule']
    out += [' & '.join(r) + r' \\' for r in rows]
    out += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    return out


KERNEL_ROWS = {
    'tpu': [('baseline', 'ffbp/fp32_fast_direct', 'ffbp/fp32_high_direct', 'XLA: ramps materialized, dense decimation products, XLA final stage'),
            ('pallas', 'ffbp/fp32_fast_pallas_direct', 'ffbp/fp32_high_pallas_direct', 'Fused level-0 kernel, one child per load'),
            ('pallas2', 'ffbp/fp32_fast_pallas2_direct', 'ffbp/fp32_high_pallas2_direct', 'Fused level kernels, children share a load, XLA final stage'),
            ('pallas3', 'ffbp/fp32_fast_pallas2_direct_xlafinal', 'ffbp/fp32_high_pallas2_direct_xlafinal', 'Same, device level batched over parents'),
            ('pallas3', 'ffbp/fp32_fast_pallas2_direct_final1', 'ffbp/fp32_high_pallas2_direct_final1', 'Plus fused final stage, per-sample sines'),
            ('pallas3', 'ffbp/fp32_fast_pallas2_direct', 'ffbp/fp32_high_pallas2_direct', 'Plus fused final stage, ramps by recurrence'),
            ('pallas4', 'ffbp/fp32_fast_pallas2_direct', 'ffbp/fp32_high_pallas2_direct', 'Plus clipped windows, no halo at levels 1 and 2, precomputed fine tables'),
            ('pallas5', 'ffbp/fp32_fast_pallas2_direct', 'ffbp/fp32_high_pallas2_direct', 'Fine tables by doubling instead (rejected)'),
            ('pallas6', 'ffbp/fp32_fast_pallas2_direct_final2', 'ffbp/fp32_high_pallas2_direct_final2', 'Coarse tables precomputed too, pulse decimation fused into the kernel at levels 1 and 2'),
            ('pallas6', 'ffbp/fp32_fast_pallas2_direct', 'ffbp/fp32_high_pallas2_direct', 'Four tiles per step in the final stage instead (3\\% faster on the v6e at three-pass precision only; not kept)'),
            ('pallas7', 'ffbp/fp32_fast_pallas2_direct', 'ffbp/fp32_high_pallas2_direct', 'Coarse-table sines over the used lanes only, padded afterwards (rejected)'),
            ('pallas8', 'ffbp/fp32_fast_pallas2_direct', 'ffbp/fp32_high_pallas2_direct', 'Generation-3 level kernel with the one-tile recurrence final (final build)')],
    'gpu': [('baseline', 'ffbp/f16_conv', 'ffbp/fp32_conv', 'XLA: ramps materialized, cuDNN convolutions, XLA final stage'),
            ('pallas', 'ffbp/f16_pallas', 'ffbp/fp32_pallas', 'Triton level-0 kernel (Pallas), 16-column windows'),
            ('cuda', None, 'ffbp/fp32_cuda', 'CUDA: shared-memory FIR levels, one pixel per thread final'),
            ('cuda2', None, 'ffbp/fp32_cuda', 'Final stage 2 by 2 pixels per thread'),
            ('cuda3', 'ffbp/f16tc_cuda', 'ffbp/fp32_cuda', 'Pulse-major filter threads (conflict-free reads, scattered loads and stores), tiled pulse filter, 4 by 2 final'),
            ('cuda4', 'ffbp/f16tc_cuda', 'ffbp/fp32_cuda', 'Coalesced window loads, children as in-place phase steps; float16 tensor-core final'),
            ('cuda5', 'ffbp/f16tc_cuda', 'ffbp/fp32_cuda', 'Rotation as a geometric sequence per thread; interleaved complex tables in the final'),
            ('cuda6', 'ffbp/f16tc_cuda', 'ffbp/fp32_cuda', 'Output tiles staged in shared memory for coalesced stores (float32 build of record)'),
            ('cuda7', 'ffbp/f16_cuda', 'ffbp/fp32_cuda', 'Float16 phase history, intermediates and final-stage operands, float32 accumulation in the filters (float16 build of record)')],
    'cpu': [('baseline', None, 'ffbp/fp32_conv_direct', 'JAX program: conv. filters, direct ramps, one process'),
            ('cpp', None, 'ffbp/fp32_cpp', 'C++ with OpenMP: polyphase filter vectorized along the output columns (stride-D gathers)'),
            ('cpp2', None, 'ffbp/fp32_cpp', 'Children in the vector lanes (no gathers), input rows read once in the pulse filter, final tables by two recurrences'),
            ('cpp3', None, 'ffbp/fp32_cpp', 'Named accumulators, complex products as FMA pairs, 512-bit vectors, reused intermediates'),
            ('cpp4', None, 'ffbp/fp32_cpp', 'Final-stage tables built and consumed three pulses at a time (second-level cache); final build')]}


def gen_kernels():
    """Panama device time of every build of the factorized algorithm on each accelerator (results/v2r/kernels.json)."""
    p = f'{R}/kernels.json'
    if not os.path.exists(p):
        return []
    k = json.load(open(p))
    base = {}
    for f in glob.glob(f'{R}/timing/panama_*.json'):
        lab = os.path.basename(f)[7:-5]
        base[lab] = json.load(open(f))
    out = [r'\begin{table}[tb]', r'\centering',
           r'\caption{Device time per Panama image (s) of each build of the factorized algorithm, two precisions per accelerator and float32 on the CPU: sixteen-bit means single-pass products on the TPUs and, on the L4, float16 products (JAX program), the float32 build with the float16 tensor-core final stage (CUDA builds four to six) or float16 storage throughout (build seven); float32 class means three-pass products on the TPUs and float32 on the L4. Each row adds to the one above; the builds of record in Table~\ref{tab:cost} are marked; the float32-class entry in the last L4 row is a rerun of the float32 configuration of the sixth build.}',
           r'\label{tab:kernels}', r'\small', r'\resizebox{\textwidth}{!}{\begin{tabular}{llrr}', r'\toprule', r'Device & Build & 16-bit (s) & float32 class (s) \\', r'\midrule']
    for lab in ('tpu-v6e', 'tpu-v5e', 'gpu-l4', 'cpu-c4d16'):
        rows = KERNEL_ROWS['tpu' if lab.startswith('tpu') else ('cpu' if lab.startswith('cpu') else 'gpu')]
        first = True
        for build, t1, t2, name in rows:
            src = base.get(lab, {}) if build == 'baseline' else k.get(lab, {}).get(build, {})
            if build == 'baseline' and lab == 'gpu-l4' and 'ffbp/fp32_conv' in k.get(lab, {}).get('cuda5', {}):
                src = k[lab]['cuda5']                         # the JAX rows re-timed on the CUDA builds' instance
            v1 = (src.get(t1) or {}).get('run_s') if t1 else None
            v2 = (src.get(t2) or {}).get('run_s') if t2 else None
            if v1 is None and v2 is None:
                continue
            out.append(f"{DEV[lab] if first else ''} & {name} & {fmt(v1, 2) if v1 is not None else '--'} & {fmt(v2, 2) if v2 is not None else '--'} \\\\")
            first = False
        out.append(r'\midrule')
    out[-1] = r'\bottomrule'
    out += [r'\end{tabular}}', r'\end{table}']
    return out


def gen_bounds():
    """Measured stage time against the bound from the measured unit rates (results/v2r/bounds.json)."""
    p = f'{R}/bounds.json'
    if not os.path.exists(p):
        return []
    b = json.load(open(p))
    out = [r'\begin{table}[tb]', r'\centering',
           r'\caption{Each stage of the final kernel builds against the lower bound set by the unit that binds it, Panama image, single-pass or float32 arithmetic. The bounds divide the inherent work of the stage by rates measured on the same device with microbenchmarks in the same framework (vector unit on the rotation mix of the kernel with data resident on chip, high-bandwidth memory copy, matrix units at the product shapes of the kernel, sines in XLA; on the L4 the inner mixes of the kernels on resident shared memory and the memory copy rate). A ratio near one means the stage runs at the rate of that unit; the last column divides the measured time by the sum of all the bounds of the stage, the time it would take if none of its units overlapped.}',
           r'\label{tab:bounds}', r'\small', r'\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrr}', r'\toprule',
           r'Device & Stage & Binding unit & Bound (s) & Measured (s) & Ratio & To the sum \\', r'\midrule']
    for lab in ('tpu-v6e', 'tpu-v5e', 'gpu-l4', 'cpu-c4d16'):
        rows = b.get(lab)
        if not rows:
            continue
        first = True
        for r in rows:
            if lab.startswith('tpu'):
                units = {'bound_vpu': 'vector unit', 'bound_hbm': 'memory', 'bound_trig': 'sines (XLA)', 'bound_mxu': 'matrix units'}
            else:
                units = {'bound_comp': 'arithmetic', 'bound_mem': 'memory'}
            k = max(units, key=lambda kk: r.get(kk, 0))
            ratio = r['measured'] / r['bound'] if r['bound'] else float('nan')
            tosum = f"{r['measured'] / r['bound_sum']:.2f}" if r.get('bound_sum') else '--'
            out.append(f"{DEV[lab] if first else ''} & {r['stage']} & {units[k]} & {r['bound']:.2f} & {r['measured']:.2f} & {ratio:.1f} & {tosum} \\\\")
            first = False
        out.append(r'\midrule')
    out[-1] = r'\bottomrule'
    out += [r'\end{tabular}}', r'\end{table}']
    return out


def gen_sicd(num):
    """The float64 reference against the vendor's SICD image, per collection (results/v2r/sicd/sicd4_<scene>.json)."""
    rows = []
    for s_ in SCENE:
        f = f'{R}/sicd/sicd4_{s_}.json'
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        w = [x for x in r['windows'] if x.get('valid') and 'model' in x]
        if not w:
            continue
        import numpy as _np
        meas = _np.array([x['measured'] for x in w]); mod = _np.array([x['model'] for x in w])
        rms = _np.sqrt(((meas - mod) ** 2).mean(0))
        ww = [x for x in r.get('windows_warped', []) if x.get('valid')]
        resid = _np.sqrt((_np.array([x['measured'] for x in ww]) ** 2).mean(0)) if ww else [float('nan')] * 2
        wa = r.get('warped', {})
        wcoh = float(_np.median([x['coh_mean'] for x in ww])) if ww else float('nan')
        wlog = float(_np.median([x['logamp_corr'] for x in ww])) if ww else float('nan')
        key = s_
        num[f'sicd.{key}.n'] = str(len(w))
        num[f'sicd.{key}.meas.az'], num[f'sicd.{key}.meas.rg'] = f"{_np.abs(meas[:, 0]).max():.1f}", f"{_np.abs(meas[:, 1]).max():.1f}"
        num[f'sicd.{key}.model.az'], num[f'sicd.{key}.model.rg'] = f"{_np.abs(mod[:, 0]).max():.1f}", f"{_np.abs(mod[:, 1]).max():.1f}"
        num[f'sicd.{key}.rms.az'], num[f'sicd.{key}.rms.rg'] = f"{rms[0]:.2f}", f"{rms[1]:.2f}"
        num[f'sicd.{key}.coh'] = f"{wa.get('coh_mean', float('nan')):.2f}"
        num[f'sicd.{key}.cohmed'] = f"{wa.get('coh_median', float('nan')):.3f}"
        num[f'sicd.{key}.wcoh'] = f"{wcoh:.2f}"
        num[f'sicd.{key}.wlog'] = f"{wlog:.2f}"
        num[f'sicd.{key}.logamp'] = f"{wa.get('log_amplitude_correlation', float('nan')):.2f}"
        num[f'sicd.{key}.ramp.az'], num[f'sicd.{key}.ramp.rg'] = f"{r['phase_ramp_cycles_per_px'][0]:.3f}", f"{r['phase_ramp_cycles_per_px'][1]:.3f}"
        num[f'sicd.{key}.shift.az'], num[f'sicd.{key}.shift.rg'] = f"{r['shift_pixels'][0]:.1f}", f"{r['shift_pixels'][1]:.1f}"
        rows.append(f"{SCENE[s_]} & {len(w)} & {_np.abs(meas[:, 0]).max():.1f} / {_np.abs(meas[:, 1]).max():.1f} & {_np.abs(mod[:, 0]).max():.1f} / {_np.abs(mod[:, 1]).max():.1f} & {rms[0]:.2f} / {rms[1]:.2f} & {resid[0]:.2f} / {resid[1]:.2f} & {wa.get('coh_mean', float('nan')):.2f} & {wcoh:.2f} & {wlog:.2f} \\\\")
    if not rows:
        return []
    return [r'\begin{table}[tb]', r'\centering',
            r'\caption{The reference (float64 exact backprojection) against the vendor\textquoteright s complex image (SICD) of the same collection, which Umbra forms by polar format. Columns: the number of 512 by 512 windows, centered on a grid with 1024-pixel spacing, with enough contrast for an amplitude cross-correlation; the largest displacement of the vendor image relative to the reference over the windows (azimuth / range, pixels); the largest displacement the planar-wavefront model of Appendix~\ref{app:pfa} predicts at the same windows; the root-mean-square difference between measured and predicted displacement; the root-mean-square residual displacement after the vendor image is resampled through the model; and, after that resampling and a local re-alignment and phase-ramp removal in each window, the 5 by 5 coherence of the two images (mean over all pixels of the scene; median of the window means) and the median over the windows of the correlation of their log amplitudes.}',
            r'\label{tab:sicd}', r'\small', r'\setlength{\tabcolsep}{4pt}', r'\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrrrr}', r'\toprule',
            r'Collection & Windows & Measured (px) & Model (px) & RMS difference (px) & Residual after resampling (px) & Coherence, scene mean & window median & Log-amplitude correlation \\', r'\midrule'] + rows + [r'\bottomrule', r'\end{tabular}}', r'\end{table}']


def gen_pfaedge():
    names = {'pfa/fp32_taps_corr': '6-tap kernel, 2 times oversampled (selected)', 'pfa/fp32_taps_corr_t4': '4-tap kernel',
             'pfa/fp32_taps_corr_t8': '8-tap kernel', 'pfa/fp32_taps_corr_ov3': '3 times oversampled',
             'pfa/fp32_taps_corr_p32': '32-tap pulse interpolator, Kaiser 12', 'pfa/fp32_taps_corr_p48': '48-tap pulse interpolator, Kaiser 12'}
    rows = {}
    for f in (f'{R}/edge_panama.json', f'{R}/ptaps_panama.json'):
        if os.path.exists(f):
            for r in json.load(open(f))['rows']:
                if not r.get('bad'):
                    rows[r['tag']] = r
    out = []
    for tag, name in names.items():
        r = rows.get(tag)
        if r:
            out.append(f"{name} & {db(r['err_db'])} & " + ' & '.join(db(v) for v in r['rings']) + r' \\')
    return out


def main():
    os.makedirs(f'{OUT}/gen', exist_ok=True)
    m = load_metrics()
    # ---- scenes table
    info = {}
    for s in SCENE:
        p = f'{R}/{s}.info'
        if os.path.exists(p):
            t = open(p).read()
            info[s] = json.loads(t[t.index('{'):])
    lines = [r'\begin{table}[tb]', r'\centering', r'\caption{The three collections and their native image grids. Spacings are azimuth by range in the slant plane; squint is the Doppler cone angle less $90^\circ$ at the aperture center; range is the slant range to the scene center at the start of the aperture.}',
             r'\label{tab:scenes}', r'\small', r'\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrrr}', r'\toprule',
             r'Scene & Image (az $\times$ rg) & Spacing (m) & Pulses $\times$ samples & Bandwidth (MHz) & Graze & Squint & Range (km) \\', r'\midrule']
    for s, d in info.items():
        lines.append(f"{SCENE[s]} & {d['nx']:,} $\\times$ {d['ny']:,} & {d['spx']:.3f} $\\times$ {d['spy']:.3f} & {d['vectors']:,} $\\times$ {d['samples']:,} & {d['bandwidth_hz'] / 1e6:.0f} & {d['graze_deg']:.1f}$^\\circ$ & {'$' + f"{d['squint_deg']:.1f}" + '$' if d['squint_deg'] < 0 else f"{d['squint_deg']:.1f}"}$^\\circ$ & {d['range_km'][0]:.0f} \\\\")
        num[f'sc.{s}.nx'] = f"{d['nx']:,}"
        num[f'sc.{s}.ny'] = f"{d['ny']:,}"
        num[f'sc.{s}.lookups'] = f"{d['nx'] * d['ny'] * d['vectors'] / 1e12:.1f}"
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    gen['scenes'] = '\n'.join(lines)
    # ---- platforms
    gen['platforms'] = '\n'.join([r'\begin{table}[tb]', r'\centering', r'\caption{Instances, the arithmetic timed on each (Section~\ref{sec:devices} and Appendix~\ref{app:formats}) and on-demand prices (US dollars per hour, us-central1).}', r'\label{tab:platforms}',
                                  r'\small', r'\resizebox{\textwidth}{!}{\begin{tabular}{lllr}', r'\toprule', r'Label & Instance & Arithmetic timed & Price \\', r'\midrule',
                                  r'TPU v5e & 1 chip (v5litepod-1), 24 vCPU host & bfloat16 single-, three- and six-pass products; float32 or bfloat16 data & 1.200 \\',
                                  r'TPU v6e & 1 chip (v6e-1), 44 vCPU host & bfloat16 single-, three- and six-pass products; float32 or bfloat16 data & 2.700 \\',
                                  r'L4 & g2-standard-4, 1 Nvidia L4 (24 GB) & float16, TF32 and float32 products; float16 or float32 data & 0.707 \\',
                                  r'CPU & c4d-highmem-16, AMD EPYC 9B45, 8 cores, 126 GB & float32 and float64 & 0.964 \\',
                                  r'\bottomrule', r'\end{tabular}}', r'\end{table}'])
    # ---- arithmetic table (static text)
    gen['arith'] = '\n'.join([r'\begin{table}[tb]', r'\centering', r'\caption{Floating-point type of each step in the timed configurations. Geometry means tile centers, ranges and phase-ramp coefficients; ramps means the sines and cosines themselves; products means the operands of the matrix products or convolutions, always accumulated in float32 or better.}',
                              r'\label{tab:arith}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}', r'\resizebox{\textwidth}{!}{\begin{tabular}{llllll}', r'\toprule',
                              r'Device & Algorithm, label & Data in memory & Geometry & Ramps & Products \\', r'\midrule',
                              r'L4 & Exact BP, float32 profiles & float32 profiles & float64 per tile & float32 & float32 sums \\',
                              r'L4 & Exact BP, float16 profiles & float16 profiles & float64 per tile & float32 & float32 sums \\',
                              r'L4 & Factorized, float32, CUDA kernels & float32 & float64 & float32 & float32 fused multiply-adds (levels and final) \\',
                              r'L4 & Factorized, float16 throughout, CUDA kernels & float16 & float64 & float32 & float16 data into float32 fused multiply-adds in the levels; float16 operands on the tensor cores in the final stage \\',
                              r'L4 & Factorized, float16 products & float16 & float64 (2 levels), float32 & float32 & float16 operands, tensor cores \\',
                              r'L4 & Factorized, float32 & float32 & float64 (2 levels), float32 & float32 & float32 (highest setting) \\',
                              r'L4 & Factorized, TF32 products & float32 & float64 (2 levels), float32 & float32 & TF32 operands (default setting) \\',
                              r'TPU & Factorized, single-pass, fused kernels & float32 & float64 (2 levels), float32 & float32 & bfloat16 operands, 1 pass (levels and final) \\',
                              r'TPU & Factorized, three-pass, fused kernels & float32 & float64 (2 levels), float32 & float32 & 3 bfloat16 passes (levels and final) \\',
                              r'TPU & Factorized, single-pass & float32 & float64 (2 levels), float32 & float32 & bfloat16 operands, 1 pass \\',
                              r'TPU & Factorized, three-pass & float32 & float64 (2 levels), float32 & float32 & 3 bfloat16 passes \\',
                              r'TPU & Factorized, six-pass & float32 & float64 (2 levels), float32 & float32 & 6 bfloat16 passes \\',
                              r'TPU & Factorized, bfloat16 storage & bfloat16 & float64 (2 levels), float32 & float32 & bfloat16 operands, 1 pass \\',
                              r'TPU & Polar format, float32 & float32 & float64 & float32 & resampling product 6 passes; float32 FFTs \\',
                              r'TPU & Polar format, single-pass product & float32 & float64 & float32 & resampling product 1 pass; float32 FFTs \\',
                              r'L4, CPU & Polar format, float32 & float32 & float64 & float32 & float32 products and FFTs \\',
                              r'CPU & Factorized, float32, C++ kernels & float32 & float64 & float32 & float32 fused multiply-adds (levels and final) \\',
                              r'CPU & Factorized, float32 & float32 & float64 (2 levels), float32 & float32 & float32 \\',
                              r'CPU & Factorized, float64 & float64 & float64 & float64 & float64 \\',
                              r'\bottomrule', r'\end{tabular}}', r'\end{table}'])
    # ---- cost table from the timing files: per scene, per device, the configurations of record
    timing = {}
    for f in glob.glob(f'{R}/timing/*.json'):
        s_, lab = os.path.basename(f)[:-5].split('_', 1)
        timing[s_, lab] = json.load(open(f))

    def t_of(s_, lab, tag, key='run_s'):
        t = timing.get((s_, lab), {}).get(tag) or {}
        return t.get(key)

    def tim(s_, lab, tag):
        t = timing.get((s_, lab), {}).get(tag)
        if not t:
            return None, None, None
        st = t.get('stream') or {}
        mon = dict(st.get('monitor') or {})                      # the pipelined loop's monitor, matching the time the cost uses
        for k, v in (t.get('monitor') or {}).items():           # the single run's monitor where the loop had none
            mon.setdefault(k, v)
        return t.get('run_s'), st.get('s_per_image'), mon

    TPU_W = {'tpu-v5e': (120, 200), 'tpu-v6e': (200, 350)}                 # published third-party design-power figures, a band

    def occupancy(mon, util, threads):
        """Fraction of the 8 cores busy: the monitor counts hardware threads (16), so a run pinned to one thread per
        core reports at most one half."""
        u = util if util is not None else ((mon or {}).get('cpu_util') or 1.0)
        n = (mon or {}).get('ncpu') or 16
        return min(1.0, u * n / 8.0)

    def kwh(lab, per, mon, util=None, threads=None):
        """kWh per 1000 images of the processor alone: measured board power on the L4, the design-power band on the TPUs,
        a pro-rata share of the package at the measured utilization on the CPU (util overrides the monitor's, for the
        multi-process loop). Returns a string."""
        if per is None:
            return ''
        if lab == 'gpu-l4':
            w = (mon or {}).get('gpu_watts_mean')
            return f'{w * per * 1000 / 3.6e6:.2f}' if w else ''
        if lab in TPU_W:
            lo, hi = TPU_W[lab]
            return f'{lo * per * 1000 / 3.6e6:.2f} to {hi * per * 1000 / 3.6e6:.2f}'
        if lab == 'cpu-c4d16':
            w = 400 * 8 / 128 * occupancy(mon, util, threads)
            return f'{w * per * 1000 / 3.6e6:.2f}'
        return ''

    BEST = {('gpu-l4', 'ffbp/fp32_cuda'), ('gpu-l4', 'ffbp/f16_cuda'), ('tpu-v5e', 'ffbp/fp32_high_pallas2_direct'), ('tpu-v5e', 'ffbp/fp32_fast_pallas2_direct'),
            ('tpu-v6e', 'ffbp/fp32_high_pallas2_direct'), ('tpu-v6e', 'ffbp/fp32_fast_pallas2_direct'), ('cpu-c4d16', 'ffbp/fp32_cpp')}

    def bold(c):
        c = c.strip()
        if not c:
            return ''
        if c.startswith('$') and c.endswith('$'):
            return '$\\mathbf{' + c[1:-1] + '}$'
        return '\\textbf{' + c + '}'

    def num_of(c):
        """The number a cell compares by: first number in the cell (the lower end of a range), None when blank."""
        m_ = re.search(r'-?\d+(?:\.\d+)?', c.replace(',', '').replace('$', '').replace('\\mathbf', ''))
        return float(m_.group(0)) if m_ else None

    def bold_best(rows, cols, tol=1e-9):
        """rows: list of cell lists; cols: {index: 'min' | 'max'}; bolds the best cell(s) of each column in place (ties within tol share)."""
        for j, sense in cols.items():
            vals = [num_of(r[j]) for r in rows]
            cand = [v for v in vals if v is not None]
            if not cand:
                continue
            best = min(cand) if sense == 'min' else max(cand)
            for r, v in zip(rows, vals):
                if v is not None and abs(v - best) <= tol:
                    r[j] = bold(r[j])

    def bold_fastest(rows, col=3):
        """Bold the whole row with the smallest device time (ties share)."""
        vals = [num_of(r[col]) for r in rows]
        best = min(v for v in vals if v is not None)
        for r, v in zip(rows, vals):
            if v is not None and abs(v - best) <= 1e-9:
                r[:] = [bold(c) if c.strip() and 'bullet' not in c else c for c in r]

    def cost_header(label, caption):
        return [r'\begin{table}[tb]', r'\centering', caption, label, r'\small', r'\setlength{\tabcolsep}{4pt}', r'\resizebox{\textwidth}{!}{\begin{tabular}{lllrrrr}', r'\toprule',
                r'Device & Algorithm & Arithmetic & Time (s) & Images/h & \$ per 1000 & kWh per 1000 \\', r'\midrule']

    lines = [r'\begin{table}[tb]', r'\centering',
             r'\caption{Panama Canal: the selected configurations, cheapest first. Bold rows are the Pareto front of Figure~\ref{fig:teaser}, the configurations that no other configuration beats on both cost and error (errors within 0.05~dB counted as equal). Cost and error as defined in Section~\ref{sec:protocol}; an asterisk marks rows timed without a pipelined loop, with cost computed from the device time. Energy is that of the processor alone over the pipelined per-image time that the cost column uses. On the L4 it is measured by \texttt{nvidia-smi}. On the TPUs it is a range from the design-power figures third parties quote (120 to 200~W for the v5e, 200 to 350~W for the v6e~\cite{introl2025,gpuadvisor2025}), since Google publishes none~\cite{google2024trillium}, applied as if the chip ran at that power throughout. On the CPU it is the 8 cores\textquoteright{} pro-rata share of the 400~W package times their occupancy, taken as the busy fraction of the 16 hardware threads doubled and capped at one. Throughput and the other two collections are in Table~\ref{tab:costother}.}',
             r'\label{tab:cost}', r'\small', r'\setlength{\tabcolsep}{4pt}', r'\resizebox{\textwidth}{!}{\begin{tabular}{lllrrrr}', r'\toprule',
             r'Device & Algorithm & Arithmetic & Time (s) & \$ per 1000 & kWh per 1000 & Error (dB) \\', r'\midrule']
    panama_rows = []
    lines_other = cost_header(r'\label{tab:costother}', r'\caption{Melbourne and Iowa: device time, throughput and cost for the selected configurations, as in Table~\ref{tab:cost}; bold marks the fastest configuration on each collection. }')
    # candidates per selected row; the fastest timed form is used (ties broken toward a pipelined record)
    record_c = [('gpu-l4', ['bp/cuda_f16']), ('gpu-l4', ['bp/cuda_fp32']), ('gpu-l4', ['ffbp/f16_cuda', 'ffbp/f16_conv']),
                ('gpu-l4', ['ffbp/fp32_cuda', 'ffbp/fp32_conv', 'ffbp/fp32']), ('gpu-l4', ['pfa/fp32_taps_corr']),
                ('tpu-v5e', ['ffbp/fp32_fast_pallas2_direct', 'ffbp/fp32_fast_direct']), ('tpu-v5e', ['ffbp/fp32_high_pallas2_direct', 'ffbp/fp32_high_direct', 'ffbp/fp32_high']), ('tpu-v5e', ['pfa/fp32_taps_corr']),
                ('tpu-v6e', ['ffbp/fp32_fast_pallas2_direct', 'ffbp/fp32_fast_direct']), ('tpu-v6e', ['ffbp/fp32_high_pallas2_direct', 'ffbp/fp32_high_direct', 'ffbp/fp32_high']), ('tpu-v6e', ['pfa/fp32_corr']),
                ('cpu-c4d16', ['ffbp/fp32_cpp', 'ffbp/fp32_conv_direct']), ('cpu-c4d16', ['ffbp/fp64_conv_direct']), ('cpu-c4d16', ['pfa/fp32_taps_corr'])]

    def pick(s_, lab, cands):
        best = None
        for tag in cands:
            run = t_of(s_, lab, tag)
            if run is None:
                continue
            piped = bool((timing.get((s_, lab), {}).get(tag) or {}).get('stream'))
            if best is None or (piped, -run) > (best[1], -best[2]):
                best = (tag, piped, run)
        return best and best[0]

    record = record_c
    cpu_stream = {}
    for f in glob.glob(f'{R}/timing/*_cpu_stream.jsonl'):
        s_ = os.path.basename(f).split('_')[0]
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if recs:
            best = min(recs, key=lambda r: r['s_per_image'])
            cpu_stream[s_] = best
            num[f'cs.{s_}.workers'] = str(best['workers'])
            num[f'cs.{s_}.s'] = f"{best['s_per_image']:.0f}"
            num[f'cs.{s_}.gain'] = f"{recs[0]['s_per_image'] / best['s_per_image']:.2f}"
            num[f'cs.{s_}.util1'] = f"{100 * recs[0]['cpu_util']:.0f}"
            num[f'cs.{s_}.utilbest'] = f"{100 * best['cpu_util']:.0f}"
    for s_ in SCENE:
        if not any(k[0] == s_ for k in timing):
            continue
        first = True
        out_lines = lines if s_ == 'panama' else lines_other
        for lab, cands in record:
            tag = pick(s_, lab, cands)
            if tag is None:
                continue
            run, sp, mon = tim(s_, lab, tag)
            thr = (timing.get((s_, lab), {}).get(tag) or {}).get('threads')
            util = None
            if lab == 'cpu-c4d16' and tag.startswith('ffbp/fp32') and 'cpp' not in tag and s_ in cpu_stream:
                sp = cpu_stream[s_]['s_per_image']
                util = cpu_stream[s_]['cpu_util']
            per = sp if sp else run
            star = '' if sp else '*'
            usd = PRICE[lab] / 3600.0 * per * 1000
            a, arith = config_name(tag, lab)
            if s_ == 'panama':
                panama_rows.append(dict(lab=lab, tag=tag, a=a, arith=arith, run=run, per=per, star=star, usd=usd, kwh=kwh(lab, per, mon, util, thr)))
            else:
                if first:
                    out_lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{SCENE[s_]}}}}} \\\\")
                cells = [DEV[lab], a, arith, fmt(run, 1), f'{3600.0 / per:,.0f}{star}', f'{usd:.2f}', kwh(lab, per, mon, util, thr)]
                out_lines.append(cells)
            key = f"c.{s_}.{lab}.{tag.replace('/', '.')}"
            for key in [key] + [f"c.{s_}.{lab}.{c_.replace('/', '.')}" for c_ in cands]:
                num[key + '.s'] = fmt(run, 1)
                num[key + '.usd'] = f'{usd:.2f}'
                num[key + '.iph'] = f'{3600.0 / per:,.0f}'
                num[key + '.sps'] = fmt(per, 1)
                num[key + '.form'] = config_name(tag, lab)[1]
            for key in [f"c.{s_}.{lab}.{tag.replace('/', '.')}"] + [f"c.{s_}.{lab}.{c_.replace('/', '.')}" for c_ in cands]:
                if mon and mon.get('gpu_watts_mean'):
                    num[key + '.w'] = f"{mon['gpu_watts_mean']:.0f}"
                if mon and mon.get('cpu_util') is not None:
                    num[key + '.util'] = f"{100 * mon['cpu_util']:.0f}"
            first = False
        # bold the best value in each numeric column of this collection (the Panama rows are handled per group below)
        scene_cells = [c for c in out_lines if isinstance(c, list)]
        if scene_cells:
            bold_fastest(scene_cells)
        out_lines[:] = [' & '.join(c) + ' \\\\' if isinstance(c, list) else c for c in out_lines]
        out_lines.append(r'\midrule')
    # metrics for each timed Panama row: the saved image of the same arithmetic (filter and ramp forms do not change it)
    prow = rows_of(m, 'panama')

    def met(lab, tag):
        cands = [tag]
        for suf in ('_direct', '_conv', '_taps'):
            cands += [c.replace(suf, '') for c in list(cands) if suf in c]
        for c in cands:
            if (lab, c) in prow:
                return prow[(lab, c)]
        return None

    TIERS = [('A', 'No visible change: amplitude and coherence maps match the reference'),
             ('B', 'Amplitude unchanged; coherence loss beside bright returns'),
             ('C', 'Amplitude unchanged; striped coherence loss'),
             ('D', 'Visibly displaced')]

    def tier(r):
        if r['amp_db']['p99'] > 3.0:
            return 'D'
        if r['coh_p001'] < 0.99:
            return 'C'
        if r['coh_p001'] < 0.999:
            return 'B'
        return 'A'

    for d_ in panama_rows:
        r = met(d_['lab'], d_['tag'])
        d_['err'] = r['err_db'] if r else None
        d_['tier'] = tier(r) if r else '?'
        d_['pareto'] = False
    pts = [d_ for d_ in panama_rows if d_['err'] is not None]
    for d_ in pts:
        d_['pareto'] = not any(o is not d_ and o['usd'] <= d_['usd'] and o['err'] <= d_['err'] + 0.05 and (o['usd'] < d_['usd'] or o['err'] < d_['err'] - 0.05) for o in pts)
    num['pareto.n'] = str(sum(d_['pareto'] for d_ in pts))
    for d_ in panama_rows:                                                   # energy per thousand images, by role
        num[f"kwh.{d_['lab']}.{d_['tag'].replace('/', '.')}"] = d_['kwh']
    for role, lab, tag in (('l4.fp32', 'gpu-l4', 'ffbp/fp32_cuda'), ('l4.f16', 'gpu-l4', 'ffbp/f16_cuda'), ('v6e.high', 'tpu-v6e', 'ffbp/fp32_high_pallas2_direct'),
                           ('v5e.high', 'tpu-v5e', 'ffbp/fp32_high_pallas2_direct'), ('v6e.fast', 'tpu-v6e', 'ffbp/fp32_fast_pallas2_direct'), ('v5e.fast', 'tpu-v5e', 'ffbp/fp32_fast_pallas2_direct'),
                           ('cpu.fp32', 'cpu-c4d16', 'ffbp/fp32_cpp'), ('l4.pfa', 'gpu-l4', 'pfa/fp32_taps_corr')):
        for d_ in panama_rows:
            if d_['lab'] == lab and d_['tag'] == tag:
                num[f'kwh.{role}'] = d_['kwh']
    num['pareto.list'] = '; '.join(f"{DEV[d_['lab']]} {d_['a'].lower()}, {d_['arith']}" for d_ in sorted((d_ for d_ in pts if d_['pareto']), key=lambda d_: d_['usd']))
    for code, name in TIERS:                                             # the regimes of visible change, used by the text
        grp = sorted((d_ for d_ in panama_rows if d_['tier'] == code), key=lambda d_: d_['usd'])
        if grp:
            num[f'tier.{code}.cheapest'] = f"{DEV[grp[0]['lab']]} {grp[0]['a'].lower()}, {grp[0]['arith']}"
            num[f'tier.{code}.cheapest.usd'] = f"{grp[0]['usd']:.2f}"
    # one flat table, cheapest first; the Pareto front in bold
    for d_ in sorted(panama_rows, key=lambda d_: d_['usd']):
        e = db(d_['err']) if d_['err'] is not None else ''
        cells = [DEV[d_['lab']], d_['a'], d_['arith'], fmt(d_['run'], 1), f"{d_['usd']:.2f}{d_['star']}", d_['kwh'], e]
        if d_['pareto']:
            cells = [bold(c) if c.strip() else c for c in cells]
        lines.append(' & '.join(cells) + ' \\\\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']                   # the Panama table is in a resizebox
    lines_other[-1] = r'\bottomrule'
    lines_other += [r'\end{tabular}}', r'\end{table}']
    gen['cost'] = '\n'.join(lines)
    gen['costother'] = '\n'.join(lines_other)
    gen['pfaedge'] = '\n'.join(gen_pfaedge())
    gen['sicd'] = '\n'.join(gen_sicd(num))
    gen['kernels'] = '\n'.join(gen_kernels())
    gen['bounds'] = '\n'.join(gen_bounds())
    # every timed configuration, for the appendix and the numbers
    for (s_, lab), d in timing.items():
        for tag, t in d.items():
            if tag == '_meta':
                continue
            key = f"t.{s_}.{lab}.{tag.replace('/', '.')}"
            num[key + '.s'] = fmt(t.get('run_s'), 1)
            st = t.get('stream') or {}
            if st:
                num[key + '.sps'] = fmt(st.get('s_per_image'), 2)
            mon = t.get('monitor') or {}
            if mon.get('gpu_watts_mean'):
                num[key + '.w'] = f"{mon['gpu_watts_mean']:.0f}"
    # ---- precision table (Panama, all configurations of record + variants)
    lines = [r'\begin{table}[tb]', r'\centering', r'\caption{Precision of the Panama Canal images against the float64 exact backprojection: energy of the difference (dB), largest pixel difference relative to the brightest pixel (dB), minimum 5 by 5 coherence (the mean and 0.1 percentile are in Appendix~\ref{app:stats}), amplitude ratio (99th percentile and maximum, dB) and phase difference (99th percentile and maximum, degrees) over the brighter half of the pixels. The two 16 times oversampled rows share the reference interpolation and read better than their estimated true error of about $-69$~dB (Section~\ref{sec:prec}). The L4 float32 row is the dense-filter image; the convolution form applies the same filter coefficients. The ramp form does not alter the image; TPU rows are labeled with the form whose image was saved.}',
             r'\label{tab:prec}', r'\small', r'\setlength{\tabcolsep}{4pt}', r'\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrr}', r'\toprule',
             r'Device & Configuration & Error & Max & Coh. min & Amp. p99 & max & Phase p99 & max \\', r'\midrule']
    rows = rows_of(m, 'panama')
    prec_order = [('gpu-l4', 'bp/cuda_fp32'), ('gpu-l4', 'bp/cuda_f16'), ('gpu-l4', 'bp/cuda_fp32_ov16'), ('gpu-l4', 'bp/cuda_f16_ov16'),
                  ('gpu-l4', 'ffbp/fp32_cuda'), ('gpu-l4', 'ffbp/f16_cuda'), ('gpu-l4', 'ffbp/f16tc_cuda'),
                  ('gpu-l4', 'ffbp/fp32'), ('gpu-l4', 'ffbp/fp32_fast'), ('gpu-l4', 'ffbp/f16_conv'),
                  ('tpu-v6e', 'ffbp/fp32_high_pallas2_direct'), ('tpu-v6e', 'ffbp/fp32_fast_pallas2_direct'), ('tpu-v5e', 'ffbp/fp32_high_pallas2_direct'), ('tpu-v5e', 'ffbp/fp32_fast_pallas2_direct'),
                  ('tpu-v6e', 'ffbp/fp32'), ('tpu-v6e', 'ffbp/fp32_high_direct'), ('tpu-v6e', 'ffbp/fp32_fast'), ('tpu-v6e', 'ffbp/bf16_mm'),
                  ('tpu-v5e', 'ffbp/fp32_high_direct'), ('tpu-v5e', 'ffbp/fp32_fast'), ('cpu-c4d16', 'ffbp/fp64_conv_direct'), ('cpu-c4d16', 'ffbp/fp32_cpp'), ('cpu-c4d16', 'ffbp/fp32_conv_direct'),
                  ('tpu-v6e', 'pfa/fp32_corr'), ('gpu-l4', 'pfa/fp32_corr'), ('cpu-c4d16', 'pfa/fp32_taps_corr')]
    for lab, tag in prec_order:
        r = rows.get((lab, tag))
        if r is None:
            continue
        a, arith = config_name(tag, lab)
        lines.append(f"{DEV[lab]} & {a}, {arith} & {db(r['err_db'])} & {db(r['max_diff_db'])} & {r['coh_min']:.3f} & {r['amp_db']['p99']:.2f} & {r['amp_db']['max']:.1f} & {r['phase_deg']['p99']:.1f} & {r['phase_deg']['max']:.0f} \\\\")
    for (lab, tag), r in sorted(rows.items()):
        if True:
            key = f"p.{lab}.{tag.replace('/', '.')}"
            num[key + '.err'] = f"{r['err_db']:.1f}"
            num[key + '.max'] = f"{r['max_diff_db']:.1f}"
            num[key + '.coh'] = f"{r['coh_mean']:.4f}"
            num[key + '.cohmin'] = f"{r['coh_min']:.3f}"
            num[key + '.amp99'] = f"{r['amp_db']['p99']:.2f}"
            num[key + '.ampmax'] = f"{r['amp_db']['max']:.1f}"
            num[key + '.ph99'] = f"{r['phase_deg']['p99']:.1f}"
            num[key + '.phmax'] = f"{r['phase_deg']['max']:.0f}"
            if r.get('rings'):
                num[key + '.rings'] = ', '.join(f'{v:.1f}' for v in r['rings'])
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    gen['prec'] = '\n'.join(lines)
    # ---- full statistics appendix: every scene; rows are the union of timed and measured configurations
    lines = []
    for s in SCENE:
        rows = rows_of(m, s)
        tags = set(rows)
        for (s_, lab), d in timing.items():
            if s_ == s:
                tags |= {(lab, t) for t in d if t != '_meta' and (lab, t) not in STALE}
        tags = {(lab, t) for lab, t in tags if not (t.startswith('pfa') and 'corr' not in t)}   # polar format before its final resampling is an intermediate, not a configuration
        if not tags:
            continue
        lines += [r'\begin{landscape}', r'\begin{table}[p]', r'\centering', f'\\caption{{Every configuration timed or measured, {SCENE[s]}: device time in seconds, with the pipelined seconds per image in parentheses where a loop was run; error and largest pixel difference (dB), 5 by 5 coherence statistics, amplitude and phase statistics over the brighter half of the pixels, and the error in four rings by distance from the scene center (quarters of the half-width, dB). A dash marks a configuration that was timed without saving its image.}}',
                  f'\\label{{tab:full{s}}}', r'\tiny', r'\setlength{\tabcolsep}{2.5pt}', r'\resizebox{\linewidth}{!}{\begin{tabular}{llrrrrrrrrrrl}', r'\toprule',
                  r'Device & Configuration & Time & Error & Max & Coh mean & Coh p0.1 & Coh min & Amp p99 & Amp max & Ph p99 & Ph max & Rings \\\\', r'\midrule']
        for lab in ('gpu-l4', 'tpu-v5e', 'tpu-v6e', 'cpu-c4d16'):
            for (l, tag) in sorted(tags):
                if l != lab:
                    continue
                a, arith = config_name(tag, lab)
                t = timing.get((s, lab), {}).get(tag) or {}
                run = t.get('run_s')
                st = t.get('stream') or {}
                tcell = fmt(run, 1) + (f" ({st['s_per_image']:.1f})" if st.get('s_per_image') else '')
                r = rows.get((l, tag))
                if r is None:
                    lines.append(f"{DEV[lab]} & {a}, {arith} & {tcell} & -- & -- & -- & -- & -- & -- & -- & -- & -- & \\\\")
                    continue
                rings = ', '.join(f'${v:.0f}$' for v in r['rings']) if r.get('rings') else ''
                lines.append(f"{DEV[lab]} & {a}, {arith} & {tcell} & {db(r['err_db'])} & {db(r['max_diff_db'])} & {r['coh_mean']:.4f} & {r['coh_p001']:.3f} & {r['coh_min']:.3f} & {r['amp_db']['p99']:.2f} & {r['amp_db']['max']:.1f} & {r['phase_deg']['p99']:.1f} & {r['phase_deg']['max']:.0f} & {rings} \\\\")
        lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}', r'\end{landscape}', '']
    gen['fullstats'] = '\n'.join(lines)
    # ---- derived numbers
    KTAG = {('tpu-v6e', 'ffbp/fp32_high_direct'): 'ffbp/fp32_high_pallas2_direct', ('tpu-v6e', 'ffbp/fp32_fast_direct'): 'ffbp/fp32_fast_pallas2_direct',
            ('tpu-v5e', 'ffbp/fp32_high_direct'): 'ffbp/fp32_high_pallas2_direct', ('tpu-v5e', 'ffbp/fp32_fast_direct'): 'ffbp/fp32_fast_pallas2_direct',
            ('gpu-l4', 'ffbp/fp32_conv'): 'ffbp/fp32_cuda', ('gpu-l4', 'ffbp/f16_conv'): 'ffbp/f16_cuda', ('cpu-c4d16', 'ffbp/fp32_conv_direct'): 'ffbp/fp32_cpp'}

    def ktag(lab, tag, s_='panama'):
        """The kernel build's tag for a role when it was timed on that scene, else the XLA tag."""
        k = KTAG.get((lab, tag))
        return k if k and timing.get((s_, lab), {}).get(k) else tag

    def t_of(s_, lab, tag, key='run_s'):
        t = timing.get((s_, lab), {}).get(ktag(lab, tag, s_)) or {}
        return t.get(key)

    def ratio(a, b_, d=1):
        return None if (a is None or b_ is None or b_ == 0) else f'{a / b_:.{d}f}'

    # three-pass against single-pass, both with direct ramps
    v = ratio(t_of('panama', 'tpu-v6e', 'ffbp/fp32_high_direct'), t_of('panama', 'tpu-v6e', 'ffbp/fp32_fast_direct'))
    if v: num['f16.ratio.v6e'] = v
    v = ratio(t_of('panama', 'tpu-v5e', 'ffbp/fp32_high_direct'), t_of('panama', 'tpu-v5e', 'ffbp/fp32_fast_direct'))
    if v: num['f16.ratio.v5e'] = v
    def t_raw(s_, lab, tag):
        return (timing.get((s_, lab), {}).get(tag) or {}).get('run_s')

    v = ratio(t_raw('panama', 'gpu-l4', 'ffbp/fp32'), t_raw('panama', 'gpu-l4', 'ffbp/fp32_conv'))
    if v: num['fair.l4.convgain32'] = v
    v = ratio(t_raw('panama', 'cpu-c4d16', 'ffbp/fp64_conv_direct'), t_raw('panama', 'cpu-c4d16', 'ffbp/fp32_conv_direct'))
    if v: num['fair.cpu.speedup'] = v
    v = ratio(t_raw('panama', 'gpu-l4', 'ffbp/f16'), t_raw('panama', 'gpu-l4', 'ffbp/f16_conv'), 2)
    if v: num['fair.l4.convgain'] = v
    # the kernel builds against the XLA program, per device (Panama, float32 class and sixteen-bit)
    kj = json.load(open(f'{R}/kernels.json')) if os.path.exists(f'{R}/kernels.json') else {}
    same = kj.get('gpu-l4', {}).get('cuda5', {})              # the L4's JAX rows timed on the kernel build's own instance (fifth build's session)

    def t_base(lab, tag):
        return (same.get(tag) or {}).get('run_s') if lab == 'gpu-l4' and tag in same else t_raw('panama', lab, tag)

    for lab, a_, b_, key in (('tpu-v6e', 'ffbp/fp32_high_direct', 'ffbp/fp32_high_pallas2_direct', 'v6e.3'), ('tpu-v6e', 'ffbp/fp32_fast_direct', 'ffbp/fp32_fast_pallas2_direct', 'v6e.1'),
                             ('tpu-v5e', 'ffbp/fp32_high_direct', 'ffbp/fp32_high_pallas2_direct', 'v5e.3'), ('tpu-v5e', 'ffbp/fp32_fast_direct', 'ffbp/fp32_fast_pallas2_direct', 'v5e.1'),
                             ('gpu-l4', 'ffbp/fp32_conv', 'ffbp/fp32_cuda', 'l4.32'), ('gpu-l4', 'ffbp/f16_conv', 'ffbp/f16_cuda', 'l4.16'),
                             ('cpu-c4d16', 'ffbp/fp32_conv_direct', 'ffbp/fp32_cpp', 'cpu.32')):
        v = ratio(t_base(lab, a_), t_raw('panama', lab, b_))
        if v:
            num[f'kgain.{key}'] = v
            num[f'kbase.{key}.s'] = fmt(t_base(lab, a_), 1)
            num[f'knew.{key}.s'] = fmt(t_raw('panama', lab, b_), 1)
    num['l4.f16tc.s'] = fmt(t_raw('panama', 'gpu-l4', 'ffbp/f16tc_cuda'), 1)
    for i_, bld in enumerate(('cpp', 'cpp2', 'cpp3', 'cpp4'), 1):
        v_ = (kj.get('cpu-c4d16', {}).get(bld, {}).get('ffbp/fp32_cpp') or {}).get('run_s')
        if v_:
            num[f'kcpu.{i_}.s'] = fmt(v_, 1)
    t32, t16 = t_raw('panama', 'gpu-l4', 'ffbp/fp32_cuda'), t_raw('panama', 'gpu-l4', 'ffbp/f16_cuda')
    if t32 and t16:
        num['l4.f16.gain.s'] = fmt(t32 - t16, 1)
    v = ratio(t_of('panama', 'gpu-l4', 'ffbp/f16_conv'), t_of('panama', 'tpu-v6e', 'ffbp/fp32_fast_direct'))
    if v: num['concl.v6e.vs.l4'] = v
    v = ratio(t_of('panama', 'gpu-l4', 'ffbp/fp32_conv'), t_of('panama', 'tpu-v6e', 'ffbp/fp32_high_direct'))
    if v: num['concl.v6e.vs.l4.fp32'] = v
    num['concl.v6e.price.vs.l4'] = f"{PRICE['tpu-v6e'] / PRICE['gpu-l4']:.1f}"
    try:
        _r = float(num['c.panama.cpu-c4d16.ffbp.fp32_conv_direct.usd']) / float(num['c.panama.gpu-l4.ffbp.fp32.usd'])
        num['concl.cpu.vs.l4'] = f"{_r:.1f}" if _r < 10 else f"{_r:.0f}"
    except KeyError:
        pass
    # power: L4 measured watts times device time; TPU and CPU bounds from design figures
    for lab, key, lo, hi in (('tpu-v5e', 'v5e', 120, 200), ('tpu-v6e', 'v6e', 200, 350)):
        t = t_of('panama', lab, 'ffbp/fp32_high_direct')
        if t:
            num[f'pw.{key}.3.lo'] = f'{lo * t / 1000:.1f}'
            num[f'pw.{key}.3.hi'] = f'{hi * t / 1000:.1f}'
    for tag, key in (('ffbp/f16_conv', 'ffbp'), ('bp/cuda_f16', 'bp'), ('ffbp/fp32_conv', 'ffbp32')):
        _, _, w = tim('panama', 'gpu-l4', ktag('gpu-l4', tag))
        t = t_of('panama', 'gpu-l4', tag)
        if w and w.get('gpu_watts_mean') and t:
            num[f'pw.l4.{key}.kj'] = f"{w['gpu_watts_mean'] * t / 1000:.2f}"
            num[f'pw.l4.{key}.w'] = f"{w['gpu_watts_mean']:.0f}"
    t = t_of('panama', 'tpu-v5e', 'ffbp/fp32_fast_direct')
    if t:
        num['pw.v5e.lo'], num['pw.v5e.hi'] = f'{120 * t / 1000:.1f}', f'{200 * t / 1000:.1f}'
    t = t_of('panama', 'tpu-v6e', 'ffbp/fp32_fast_direct')
    if t:
        num['pw.v6e.lo'], num['pw.v6e.hi'] = f'{200 * t / 1000:.1f}', f'{350 * t / 1000:.1f}'
    t, mon = t_of('panama', 'cpu-c4d16', 'ffbp/fp32_cpp'), (t_of('panama', 'cpu-c4d16', 'ffbp/fp32_cpp', 'stream') or {}).get('monitor')
    if t and mon:
        occ = min(1.0, (mon.get('cpu_util') or 1.0) * (mon.get('ncpu') or 16) / 8.0)
        num['pw.cpu.kj'] = f"{400 * 8 / 128 * occ * t / 1000:.2f}"
        num['pw.cpu.occ'] = f"{100 * occ:.0f}"
    # reference cost: 48 vCPU n2-highmem-48 at 2.3 USD/h
    rt = {}
    for f in glob.glob(f'{R}/timing/*_ref.json'):
        s_ = os.path.basename(f).split('_')[0]
        rt[s_] = json.load(open(f)).get('bp/numba_fp64', {}).get('run_s')
    if rt.get('panama'):
        num['ref.panama.s'] = f"{rt['panama']:.0f}"
        num['ref.panama.usd'] = f"{2.30 / 3600 * rt['panama']:.2f}"
    # profile numbers
    for f in glob.glob(f'{R}/logs/profile_*.json'):
        d = json.load(open(f))
        lab = d['label'].replace('tpu-', '').replace('gpu-', '')
        pol = 'fp32_fast' if 'tpu' in d['label'] else 'f16'
        st = {(x['policy'], x['stage']): x['seconds'] for x in d['stages']}
        g = lambda n: st.get((pol, n)) or st.get(('fp32_fast', n)) or st.get(('bf16_mm', n))
        for short, name in (('rot', 'rotate_direct'), ('dk', 'dec_k_dense'), ('dkconv', 'dec_k_conv'), ('dkband', 'dec_k_banded'), ('dp', 'dec_p_dense'), ('dpconv', 'dec_p_conv'), ('dpband', 'dec_p_banded')):
            if g(name):
                num[f'prof.{lab}.{short}'] = f'{1e3 * g(name):.1f}'
        if g('dec_k_dense') and g('dec_p_dense'):
            num[f'prof.{lab}.mm'] = f"{1e3 * (g('dec_k_dense') + g('dec_p_dense')):.1f}"
            flops = 2.0 * 2 * d['P'] * d['K'] * d['Ko'] + 2.0 * 2 * d['Ko'] * d['P'] * d['Po']
            num[f'prof.{lab}.tflops'] = f"{flops / (g('dec_k_dense') + g('dec_p_dense')) / 1e12:.0f}"
        if g('rotate_direct'):
            num[f'prof.{lab}.rotgbs'] = f"{(2 * 4 * d['P'] * d['K'] + 2 * 2 * 4 * d['P'] * d['K']) / g('rotate_direct') / 1e9:.0f}"
    # polar format numbers
    for s_, d in info.items():
        if s_ == 'panama':
            num['pfa.P'] = f"{d['vectors']:,}"
            num['data.griddrift'] = f"{d['grid_drift_hz'] / d['df']:.2f}"
    for (s_, lab), d in timing.items():
        pf = d.get('_meta', {}).get('pfa')
        if s_ == 'panama' and pf:
            num['pfa.nfx'], num['pfa.nfy'] = f"{pf['nfx']:,}", f"{pf['nfy']:,}"
    # cost of the polar-format correction per device
    for lab, base_tag, corr_tag, key in (('gpu-l4', 'pfa/fp32_taps', 'pfa/fp32_taps_corr', 'l4'), ('cpu-c4d16', 'pfa/fp32_taps', 'pfa/fp32_taps_corr', 'cpu'), ('tpu-v6e', 'pfa/fp32', 'pfa/fp32_corr', 'v6e'), ('tpu-v5e', 'pfa/fp32_taps', 'pfa/fp32_taps_corr', 'v5e')):
        a_, b_ = t_of('panama', lab, base_tag), t_of('panama', lab, corr_tag)
        if a_ and b_:
            num[f'pfa.corr.{key}.add'] = f'{b_ - a_:.1f}'
            num[f'pfa.corr.{key}.factor'] = f'{b_ / a_:.0f}'
    num['pfa.corner.shift'] = '10'
    def usd_of(s_, lab, tag):
        run, sp, _ = tim(s_, lab, tag)
        per = sp or run
        return None if per is None else PRICE[lab] / 3600 * per * 1000

    u = usd_of('panama', 'gpu-l4', 'ffbp/fp32_fast_conv')
    if u:
        num['c.l4.tf32.usd'] = f'{u:.2f}'
    for s_ in ('panama', 'melbourne', 'iowa'):
        for key, lab, tag in (('l4.tf32', 'gpu-l4', 'ffbp/fp32_fast_conv'), ('l4.fp32', 'gpu-l4', 'ffbp/fp32_conv'), ('l4.f16', 'gpu-l4', 'ffbp/f16_conv'),
                              ('v6e.high', 'tpu-v6e', 'ffbp/fp32_high_direct'), ('v5e.high', 'tpu-v5e', 'ffbp/fp32_high_direct'),
                              ('v6e.fast', 'tpu-v6e', 'ffbp/fp32_fast_direct'), ('v5e.fast', 'tpu-v5e', 'ffbp/fp32_fast_direct')):
            u = usd_of(s_, lab, ktag(lab, tag, s_))
            if u:
                num[f'c.{s_}.{key}.usd'] = f'{u:.2f}'
    # exact backprojection's time against the factorized images, and the float16 builds' loss
    for tag, key, ref_ in (('bp/cuda_f16', 'bp.time.vs.ffbp16', 'ffbp/fp32_cuda'), ('bp/cuda_fp32', 'bp.time.vs.ffbp32', 'ffbp/fp32_cuda')):
        v = ratio(t_raw('panama', 'gpu-l4', tag), t_raw('panama', 'gpu-l4', ref_), 0)
        if v: num[key] = v
    # the same over Panama and Melbourne (the two collections where exact backprojection is not more accurate)
    tr, cr = [], []
    for s_ in ('panama', 'melbourne'):
        for tag in ('bp/cuda_f16', 'bp/cuda_fp32'):
            a_, b_ = t_raw(s_, 'gpu-l4', tag), t_raw(s_, 'gpu-l4', 'ffbp/fp32_cuda')
            if a_ and b_:
                tr.append(a_ / b_)
            u1, u2 = usd_of(s_, 'gpu-l4', tag), usd_of(s_, 'gpu-l4', 'ffbp/fp32_cuda')
            if u1 and u2:
                cr.append(u1 / u2)
    if tr:
        num['bp.time.pm.lo'], num['bp.time.pm.hi'] = f'{min(tr):.0f}', f'{max(tr):.0f}'
    if cr:
        num['bp.cost.pm.lo'], num['bp.cost.pm.hi'] = f'{min(cr):.0f}', f'{max(cr):.0f}'
    ri = rows_of(m, 'iowa')
    if ('gpu-l4', 'ffbp/fp32_cuda') in ri and ('gpu-l4', 'bp/cuda_fp32') in ri and ('gpu-l4', 'bp/cuda_f16') in ri:
        d1 = ri[('gpu-l4', 'ffbp/fp32_cuda')]['err_db'] - ri[('gpu-l4', 'bp/cuda_f16')]['err_db']
        d2 = ri[('gpu-l4', 'ffbp/fp32_cuda')]['err_db'] - ri[('gpu-l4', 'bp/cuda_fp32')]['err_db']
        num['bp.iowa.gain.lo'], num['bp.iowa.gain.hi'] = f'{min(d1, d2):.1f}', f'{max(d1, d2):.1f}'
    pr_ = rows_of(m, 'panama')
    if ('gpu-l4', 'ffbp/f16_cuda') in pr_ and ('gpu-l4', 'ffbp/fp32_cuda') in pr_:
        num['l4.f16.loss'] = f"{pr_[('gpu-l4', 'ffbp/f16_cuda')]['err_db'] - pr_[('gpu-l4', 'ffbp/fp32_cuda')]['err_db']:.1f}"
    if ('gpu-l4', 'ffbp/f16tc_cuda') in pr_ and ('gpu-l4', 'ffbp/fp32_cuda') in pr_:
        num['l4.f16tc.loss'] = f"{pr_[('gpu-l4', 'ffbp/f16tc_cuda')]['err_db'] - pr_[('gpu-l4', 'ffbp/fp32_cuda')]['err_db']:.1f}"
    # single-pass penalties, polar-format time ratio, price fraction, TPU correction slowdown
    pr0 = rows_of(m, 'panama')
    def err_(lab, tag):
        r = pr0.get((lab, tag)); return None if r is None else r['err_db']
    e_sp, e_3, e_l4f16 = err_('tpu-v6e', 'ffbp/fp32_fast_pallas2_direct'), err_('tpu-v6e', 'ffbp/fp32_high_pallas2_direct'), err_('gpu-l4', 'ffbp/f16_cuda')
    if e_sp is not None and e_3 is not None:
        num['sp.vs.3pass.db'] = f'{e_sp - e_3:.0f}'
    if e_sp is not None and e_l4f16 is not None:
        num['sp.vs.l4f16.db'] = f'{e_sp - e_l4f16:.1f}'
    v = ratio(t_raw('panama', 'gpu-l4', 'ffbp/fp32_cuda'), t_raw('panama', 'gpu-l4', 'pfa/fp32_taps_corr'))
    if v: num['pfa.time.vs.ffbp.l4'] = v
    v = ratio(t_raw('panama', 'cpu-c4d16', 'ffbp/fp32_cpp'), t_raw('panama', 'cpu-c4d16', 'pfa/fp32_taps_corr'))
    if v: num['pfa.time.vs.ffbp.cpu'] = v
    num['concl.l4.price.frac'] = f"{PRICE['gpu-l4'] / PRICE['tpu-v6e']:.2f}"
    v = ratio(t_raw('panama', 'tpu-v6e', 'pfa/fp32_corr'), t_raw('panama', 'gpu-l4', 'pfa/fp32_taps_corr'), 0)
    if v: num['pfa.tpu.vs.l4'] = v
    v = ratio(t_raw('panama', 'tpu-v5e', 'pfa/fp32_taps_corr'), t_raw('panama', 'gpu-l4', 'pfa/fp32_taps_corr'), 0)
    if v: num['pfa.v5e.vs.l4'] = v
    # polar format against the factorized float32 image on the L4 (now, and with the JAX program) and on the CPU
    up, uf = usd_of('panama', 'gpu-l4', 'pfa/fp32_taps_corr'), usd_of('panama', 'gpu-l4', 'ffbp/fp32_cuda')
    if up and uf:
        num['pfa.vs.ffbp.l4'] = f'{uf / up:.1f}'
    uf0 = usd_of('panama', 'gpu-l4', 'ffbp/fp32_conv')
    if up and uf0:
        num['pfa.vs.ffbp.l4.before'] = f'{uf0 / up:.0f}'
    try:
        num['pfa.vs.ffbp.cpu'] = f"{float(num['c.panama.cpu-c4d16.ffbp.fp32_conv_direct.usd']) / float(num['c.panama.cpu-c4d16.pfa.fp32_taps_corr.usd']):.0f}"
    except (KeyError, ValueError):
        pass
    ub, uf = usd_of('panama', 'gpu-l4', 'bp/cuda_f16'), usd_of('panama', 'gpu-l4', 'ffbp/fp32_cuda')
    if ub and uf:
        num['bp.vs.ffbp32.lo'] = f'{ub / uf:.0f}'
    # exact backprojection against the factorized images of the L4
    for tag, key, ref_ in (('bp/cuda_f16', 'bp.vs.ffbp16', 'ffbp/f16_cuda'), ('bp/cuda_fp32', 'bp.vs.ffbp32', 'ffbp/fp32_cuda')):
        ub, uf = usd_of('panama', 'gpu-l4', tag), usd_of('panama', 'gpu-l4', ref_)
        if ub and uf:
            num[key] = f'{ub / uf:.0f}'
    # the L4's saving from float16, and the XLA-program spread of the three accelerators
    u16, u32 = usd_of('panama', 'gpu-l4', 'ffbp/f16_cuda'), usd_of('panama', 'gpu-l4', 'ffbp/fp32_cuda')
    if u16 and u32:
        num['l4.f16.saving.pct'] = f'{100 * (1 - u16 / u32):.0f}'
    for lab, key in (('tpu-v6e', 'v6e'), ('tpu-v5e', 'v5e')):
        u1, u3 = usd_of('panama', lab, 'ffbp/fp32_fast_pallas2_direct'), usd_of('panama', lab, 'ffbp/fp32_high_pallas2_direct')
        if u1 and u3:
            num[f'sp.saving.{key}.pct'] = f'{100 * (1 - u1 / u3):.0f}'
    us0 = [usd_of('panama', lab, tag) for lab, tag in (('gpu-l4', 'ffbp/fp32_conv'), ('tpu-v5e', 'ffbp/fp32_high_direct'), ('tpu-v6e', 'ffbp/fp32_high_direct'))]
    if all(us0):
        num['spread0.panama.fp32.pct'] = f'{100 * (max(us0) / min(us0) - 1):.0f}'
    # spread of the three accelerators at float32 accuracy, per scene
    for s_ in ('panama', 'melbourne', 'iowa'):
        us = [usd_of(s_, lab, ktag(lab, tag, s_)) for lab, tag in (('gpu-l4', 'ffbp/fp32_conv'), ('tpu-v5e', 'ffbp/fp32_high_direct'), ('tpu-v6e', 'ffbp/fp32_high_direct'))]
        if all(us):
            num[f'spread.{s_}.fp32.pct'] = f'{100 * (max(us) / min(us) - 1):.0f}'
        us = [usd_of(s_, lab, ktag(lab, tag, s_)) for lab, tag in (('gpu-l4', 'ffbp/f16_conv'), ('tpu-v5e', 'ffbp/fp32_fast_direct'), ('tpu-v6e', 'ffbp/fp32_fast_direct'))]
        if all(us):
            num[f'spread.{s_}.16.pct'] = f'{100 * (max(us) / min(us) - 1):.0f}'

    r = rows_of(m, 'panama').get(('gpu-l4', 'pfa/fp32_corr'))
    if r:
        num['pfa.corr.screen.gain'] = f"{r['err_db'] - r['screen']['err_db']:.1f}"

    for (s_, lab), d in timing.items():
        pf = d.get('_meta', {}).get('pfa') or {}
        if pf.get('max_shift_px'):
            num[f'pfa.shift.{s_}.az'] = f"{pf['max_shift_px'][0]:.0f}"
            num[f'pfa.shift.{s_}.rg'] = f"{pf['max_shift_px'][1]:.0f}"
            if s_ in info:
                num[f'pfa.shift.{s_}.azm'] = f"{pf['max_shift_px'][0] * info[s_]['spx']:.1f}"
                num[f'pfa.shift.{s_}.rgm'] = f"{pf['max_shift_px'][1] * info[s_]['spy']:.1f}"
    # cost of the 16x exact rows
    for tag, key in (('bp/cuda_f16_ov16', 'f16'), ('bp/cuda_fp32_ov16', 'fp32')):
        t = t_of('panama', 'gpu-l4', tag)
        if t:
            num[f'c16.{key}.usd'] = f"{PRICE['gpu-l4'] / 3600 * t * 1000:.1f}"
    # matched-accuracy time ratio v6e three-pass vs L4 float16
    v = ratio(t_of('panama', 'gpu-l4', 'ffbp/f16_conv'), t_of('panama', 'tpu-v6e', 'ffbp/fp32_high_direct'))
    if v: num['concl.v6e.vs.l4.matched'] = v
    # worst-case phase extreme of single-pass products over the three scenes
    worst = max((rows_of(m, s_).get(('tpu-v6e', 'ffbp/fp32_fast')) or {'phase_deg': {'max': 0}})['phase_deg']['max'] for s_ in SCENE)
    num['sp.phmax.worst'] = f'{worst:.0f}'
    # float32 vs float64 on the CPU
    a_, b_ = rows_of(m, 'panama').get(('cpu-c4d16', 'ffbp/fp32_conv_direct')), rows_of(m, 'panama').get(('cpu-c4d16', 'ffbp/fp64_conv_direct'))
    if a_ and b_:
        num['cpu.f32vsf64.db'] = f"{a_['err_db'] - b_['err_db']:.1f}"
    c_ = rows_of(m, 'panama').get(('cpu-c4d16', 'ffbp/fp32_cpp'))
    if c_ and b_:
        num['cpu.cppvsf64.db'] = f"{c_['err_db'] - b_['err_db']:.1f}"
    # corrected polar format on the other scenes
    for s_ in ('melbourne', 'iowa'):
        r = rows_of(m, s_).get(('gpu-l4', 'pfa/fp32_corr'))
        if r:
            num[f'p.{s_}.pfacorr.err'] = f"{r['err_db']:.1f}"
    cc = f'{R}/pfa_center_crop.json'
    if os.path.exists(cc):
        d = json.load(open(cc))
        num['pfa.center.coh'] = f"{d['coh_mean']:.3f}"
        num['pfa.center.quad'] = f"{d['quad_rad']:.0f}"
        num['pfa.center.err'] = f"{d['err_corr_db']:.0f}"
    ov = f'{R}/overrides.json'
    if os.path.exists(ov):
        num.update(json.load(open(ov)))
    gen['power'] = '\n'.join(gen_power(num))                  # after every token exists
    for k, v in gen.items():
        open(f'{OUT}/gen/{k}.tex', 'w').write(v + '\n')
    json.dump(num, open(f'{OUT}/numbers.json', 'w'), indent=1, sort_keys=True)
    # ---- splice
    tex = open(f'{OUT}/paper.src.tex').read()
    for k, v in gen.items():
        pat = re.compile(r'(% BEGIN GEN ' + re.escape(k) + r'\n).*?(% END GEN ' + re.escape(k) + r'\n)', re.S)
        tex = pat.sub(lambda mm: mm.group(1) + v + '\n' + mm.group(2), tex)
    missing = []

    def sub(mm):
        if mm.group(1) not in num:
            missing.append(mm.group(1))
            return r'\textit{[running]}' if os.environ.get('INTERIM') else '??'
        v = num[mm.group(1)]
        if re.fullmatch(r'-\d+(\.\d+)?', v):                       # a negative number in prose gets a mathematical minus
            return f'${v}$'
        return v

    tex = re.sub(r'@@([^@\s]+)@@', sub, tex)
    open(f'{OUT}/paper.tex', 'w').write(tex)
    print('missing numbers:', sorted(set(missing)))
    for _ in range(2):
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'paper.tex'], cwd=OUT, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:])
        sys.exit(1)
    print([l for l in r.stdout.splitlines() if 'Output written' in l][:1])


if __name__ == '__main__':
    main()
