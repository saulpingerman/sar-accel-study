"""(Release records, results/fastsar.) Error of each configuration of the cost figure on the three Panama regions (lock, port, ship), per region and
pooled (energy of the differences over energy of the reference, each region with its own fitted gain), from the saved
images; the open-source implementations are pooled from their per-region records with the reference energies.
   python -I region_errors.py <reference .npy> <tmp dir> <bucket prefix> <records dir> <out json>"""
import sys, os, json, subprocess
import numpy as np
ref_path, tmp, B, rec, out = sys.argv[1:6]
ref = np.load(ref_path, mmap_mode='r')
crops = dict(locks=(3400, 6700), port=(9500, 1150), ships=(6400, 2650)); h = 512
E = {n: float(np.sum(np.abs(np.asarray(ref[i:i + h, j:j + h]).astype(np.complex128)) ** 2)) for n, (i, j) in crops.items()}


def pooled(errs):            # per-region dB -> pooled dB, weighted by the reference energy of each region
    return float(10 * np.log10(sum(10 ** (errs[n] / 10) * E[n] for n in crops) / sum(E.values())))


def err(a, b):
    a, b = a.astype(np.complex128).ravel(), b.astype(np.complex128).ravel(); g = np.vdot(a, b) / np.vdot(a, a)
    gr = g if np.iscomplexobj(a) else g
    return float(10 * np.log10(np.sum(np.abs(g * a - b) ** 2) / np.sum(np.abs(b) ** 2)))


images = {   # name in the figure: path under the bucket's v3/out/panama/ (release records of FastSAR)
    'L4 float16': 'gpu-l4/ffbp_f16_cuda.npy', 'L4 float32': 'gpu-l4/ffbp_fp32_cuda.npy', 'L4 polar format': 'gpu-l4/pfa_fp32_taps_corr.npy',
    'v6e single-pass': 'tpu-v6e/ffbp_fp32_fast_pallas2_direct.npy', 'v6e three-pass': 'tpu-v6e/ffbp_fp32_high_pallas2_direct.npy',
    'v6e polar format': 'tpu-v6e/pfa_fp32_taps_corr.npy',
    'v5e single-pass': 'tpu-v5e/ffbp_fp32_fast_pallas2_direct.npy', 'v5e three-pass': 'tpu-v5e/ffbp_fp32_high_pallas2_direct.npy',
    'v5e polar format': 'tpu-v5e/pfa_fp32_taps_corr.npy',
    'CPU float32': 'cpu-c4d16/ffbp_fp32_cpp.npy', 'CPU polar format': 'cpu-c4d16/pfa_fp32_taps_corr.npy',
    'FastSAR exact, L4': 'gpu-l4/bp_cubic.npy', 'FastSAR exact, CPU': 'cpu-c4d16/bp_cubic.npy',
    'FastSAR exact linear, L4': 'gpu-l4/bp_linear.npy'}
res = dict(reference_energy=E, configurations={})
for name, rel in images.items():
    f = f'{tmp}/img.npy'
    subprocess.run(['gcloud', 'storage', 'cp', f'{B}/{rel}', f], check=True, capture_output=True)  # B = gs://.../v3/out/panama
    a = np.load(f, mmap_mode='r')
    errs = {n: err(np.asarray(a[i:i + h, j:j + h]), np.asarray(ref[i:i + h, j:j + h])) for n, (i, j) in crops.items()}
    del a; os.remove(f)
    res['configurations'][name] = dict(image=rel, regions=errs, pooled_db=pooled(errs))
    print(name, {k: round(v, 2) for k, v in errs.items()}, round(pooled(errs), 2), flush=True)
# the open-source implementations, from their records
L = lambda f: json.load(open(f'{rec}/{f}'))
rp = L('ritsar_panama.json')['runs']['RITSAR as published (Python 3 port), 16x']['errors']
r16 = {n: rp[n]['reference (float64 exact BP)'] for n in crops}; res['configurations']['RITSAR, 16x'] = dict(regions=r16, pooled_db=pooled(r16))
ng = L('nga_results.json')
for key, nm in (('panama, Nfft default', 'bpBasic, 2^17'), ('panama, Nfft 262144', 'bpBasic, 2^18')):
    e = {n: ng[key]['errors'][n]['reference (float64 exact BP)'] for n in crops}; res['configurations'][nm] = dict(regions=e, pooled_db=pooled(e))
ir = L('isce3_ramp_removed.json')
for pre, nm in (('cpu_', 'ISCE3, CPU'), ('gpu_', 'ISCE3, CUDA')):
    e = [ir[pre + p]['error_after_ramp_db'] for p in ('center', 'locks', 'port')]
    res['configurations'][nm] = dict(patches=dict(zip(('center', 'locks', 'port'), e)), pooled_db=float(10 * np.log10(np.mean([10 ** (x / 10) for x in e]))),
                                     note='three 256 by 256 pixel patches of its own grid (center, lock, port), equal weights, after ramp removal')
res['configurations']['torchbp exact'] = dict(pooled_db=0.0, note='0.0 dB on every region (torchbp_panama.json)')
res['configurations']['GRDL FFBP'] = json.load(open(f'{rec}/../pareto_regions.json'))['configurations']['GRDL FFBP']   # its smeared image, as scored in grdl_ffbp_scores.json
json.dump(res, open(out, 'w'), indent=1)
