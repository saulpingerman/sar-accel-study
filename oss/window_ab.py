"""Does the moving GPU window of range profiles cost speed? A fifth of the 2025 Capella stripmap (small enough that its
profiles fit on the L4) is formed with all profiles resident and with the window that the full image uses, alternating,
on the same GPU; times, window moves and the image difference are reported.
    python -I window_ab.py <cphd> <sicd> <out dir>"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modes_common as mc  # noqa: E402
from fastsar import patches  # noqa: E402
import cupy as cp  # noqa: E402

cphd, sicd, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
fx, ant, meta, sm, ap = mc.load(cphd, sicd, stripmap=True)
P, K = fx['S'].shape
rows, cols = int(sm.ImageData.NumRows), int(sm.ImageData.NumCols)
R_, C_ = np.meshgrid([0, rows - 1], [0, cols - 1], indexing='ij')
corners = mc.io.sicd_points(sm, R_.ravel(), C_.ravel(), meta)
d = ap['d'][P // 2].copy(); d[2] = 0; e1 = d / np.linalg.norm(d); e2 = np.cross([0, 0, 1.0], e1)
graze = np.radians(float(sm.SCPCOA.GrazeAng))
spx, spy = float(sm.Grid.Col.SS), float(sm.Grid.Row.SS) / np.cos(graze)
a1, a2 = corners @ e1, corners @ e2
origin = a1.min() * e1 + a2.min() * e2 + np.array([0, 0, corners[:, 2].mean()])
nx, ny = int(np.ceil((a1.max() - a1.min()) / spx)) + 1, int(np.ceil((a2.max() - a2.min()) / spy)) + 1
ps = 1024
# the window the full image uses: a quarter of the free GPU memory
K2 = patches._padding(fx)[3]
row = K2 * 8
rows_full = int(0.25 * cp.cuda.Device().mem_info[0] // row)
# a fifth of the strip along track, at its middle, with the pulses that serve it
sx = (nx // 5) // ps * ps
o2 = origin + ((nx - sx) // 2) * spx * e1
beam_all = mc.beam_fn(ant, ap)
q = o2 + (np.array([0.0, sx]) * spx)[:, None] * e1 + (ny / 2 * spy) * e2
lo, hi = patches.beam_span(beam_all, P, q, 1.0)
lo, hi = max(0, lo - 2000), min(P, hi + 2000)
fxs = dict(fx, S=fx['S'][lo:hi], ref=fx['ref'][lo:hi])
ants = ant[lo:hi]
aps = dict(ap, d=ap['d'][lo:hi], sp=ap['sp'][lo:hi])
beam = mc.beam_fn(ants, aps)
Qbytes = (hi - lo) * row
rec = dict(pulses=int(hi - lo), grid=[int(sx), int(ny)], patches=int((sx // ps) * -(-ny // ps)), profiles_gb=Qbytes / 1e9,
           free_gb=cp.cuda.Device().mem_info[0] / 1e9, window_rows=rows_full, gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
print(rec, flush=True)
assert Qbytes < 0.4 * cp.cuda.Device().mem_info[0], 'the fifth of the strip does not fit resident; use a smaller part'


def run(window):
    if window:
        os.environ['FASTSAR_PROFILE_WINDOW_ROWS'] = str(rows_full)
    else:
        os.environ.pop('FASTSAR_PROFILE_WINDOW_ROWS', None)
    cp.cuda.Device().synchronize()
    t = time.perf_counter()
    img = patches.form_mosaic(fxs, ants, o2, sx, ny, spx, spy, e1, e2, patch=(ps, ps), beam=beam, awin=mc.hann, backend='cuda')
    return time.perf_counter() - t, img


run(False)                                    # warm up: kernel compilation, plans
times = {'resident': [], 'window': []}
imgs = {}
for k in range(2):
    for name in ('resident', 'window'):
        t, img = run(name == 'window')
        times[name].append(t)
        imgs[name] = img
        print(f'{name}: {t:.1f} s', flush=True)
diff = float(10 * np.log10(np.sum(np.abs(imgs['window'] - imgs['resident']) ** 2) / np.sum(np.abs(imgs['resident']) ** 2)))
rec.update(times=times, image_difference_db=diff)
print(f'window vs resident image: {diff:.1f} dB; resident {min(times["resident"]):.1f} s, window {min(times["window"]):.1f} s', flush=True)
json.dump(rec, open(f'{out}/window_ab.json', 'w'), indent=1)
