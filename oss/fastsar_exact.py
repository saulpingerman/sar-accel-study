"""FastSAR's exact backprojection (fastsar.backproject, cpu backend) on the Panama collection: the three 512 by 512
pixel regions against the float64 reference, the lock region timed alone, and the full image timed.
   python -I fastsar_exact.py <work dir> <out json> [full]"""
import os, sys, json, time
import numpy as np
import fastsar
W, out = sys.argv[1:3]
d = np.load(f'{W}/panama.npz')
S, ant, fmin, df = d['S'], d['ant'], float(d['fmin']), float(d['df'])
nx, ny, spx, spy, e1, e2 = int(d['nx']), int(d['ny']), float(d['spx']), float(d['spy']), d['e1'], d['e2']
ref = np.load(f'{W}/ref.npy', mmap_mode='r')
crops = dict(locks=(3400, 6700), port=(9500, 1150), ships=(6400, 2650)); h = 512
X = fastsar.plane_points(nx, ny, spx, spy, e1, e2)


def err(a, b):
    a, b = a.astype(np.complex128).ravel(), b.astype(np.complex128).ravel(); g = np.vdot(a, b) / np.vdot(a, a)
    return float(10 * np.log10(np.sum(np.abs(g * a - b) ** 2) / np.sum(np.abs(b) ** 2)))


res = dict(fastsar=fastsar.__version__, upsample=8, regions={})
for name, (i0, j0) in crops.items():
    t = time.perf_counter(); c = time.process_time()
    img = fastsar.backproject(S, ant, fmin, df, X[i0:i0 + h, j0:j0 + h], backend=os.environ.get('BACKEND', 'cpu'), chunk=1024)
    wall, cpu = time.perf_counter() - t, time.process_time() - c
    res['regions'][name] = dict(error_db=err(img, np.asarray(ref[i0:i0 + h, j0:j0 + h])), seconds=wall, cpu_seconds=cpu, threads_used=cpu / wall)
    print(name, res['regions'][name], flush=True)
lock = res['regions']['locks']['seconds']
res['full_image_estimate_hours_from_lock'] = lock * nx * ny / h ** 2 / 3600
if len(sys.argv) > 3:
    t = time.perf_counter(); c = time.process_time()
    img = fastsar.backproject(S, ant, fmin, df, X, backend=os.environ.get('BACKEND', 'cpu'), chunk=1024)
    wall, cpu = time.perf_counter() - t, time.process_time() - c
    res['full_image'] = dict(seconds=wall, cpu_seconds=cpu, threads_used=cpu / wall,
                             errors={n: err(img[i0:i0 + h, j0:j0 + h], np.asarray(ref[i0:i0 + h, j0:j0 + h])) for n, (i0, j0) in crops.items()})
    print('full image', res['full_image'], flush=True)
json.dump(res, open(out, 'w'), indent=1)
