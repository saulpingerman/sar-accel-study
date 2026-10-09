"""FastSAR's exact backprojection (fastsar.ExactFormer) on the full Panama image, as recorded for the paper: the
former built once, the image formed twice and the second call timed from the phase history in host memory to the
image in host memory; errors over the whole image and on the lock, port and ship regions against the float64
reference; the image saved for the precision statistics.
   python -I exact_record.py <panama.npz> <reference .npy> <out json> <backend> <interp> [image .npy]"""
import sys, json, time, platform, subprocess
import numpy as np
import fastsar

data, refp, out, backend, interp = sys.argv[1:6]
img_out = sys.argv[6] if len(sys.argv) > 6 else None
d = np.load(data)
S, ant, fmin, df = d['S'], d['ant'], float(d['fmin']), float(d['df'])
nx, ny, spx, spy, e1, e2 = int(d['nx']), int(d['ny']), float(d['spx']), float(d['spy']), d['e1'], d['e2']
ref = np.load(refp, mmap_mode='r')
crops = dict(locks=(3400, 6700), port=(9500, 1150), ships=(6400, 2650)); h = 512


def err(a, b):
    a, b = a.astype(np.complex128).ravel(), b.astype(np.complex128).ravel(); g = np.vdot(a, b) / np.vdot(a, a)
    return float(10 * np.log10(np.sum(np.abs(g * a - b) ** 2) / np.sum(np.abs(b) ** 2)))


t = time.perf_counter()
former = fastsar.ExactFormer(ant, fmin, df, S.shape[1], nx, ny, spx, spy, e1, e2, backend=backend, interp=interp)
setup = time.perf_counter() - t
t = time.perf_counter(); img = former(S); first = time.perf_counter() - t
t = time.perf_counter(); img = former(S); secs = time.perf_counter() - t
res = dict(fastsar=subprocess.run(['git', '-C', fastsar.__path__[0] + '/..', 'log', '-1', '--format=%H'], capture_output=True, text=True).stdout.strip(),
           backend=backend, interp=interp, upsample=former.upsample, host=platform.node(), setup_seconds=setup, first_call_seconds=first,
           full_image=dict(seconds=secs, whole_image_db=err(img, np.asarray(ref)),
                           errors={n: err(img[i:i + h, j:j + h], np.asarray(ref[i:i + h, j:j + h])) for n, (i, j) in crops.items()}))
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(out, 'w'), indent=1)
if img_out:
    np.save(img_out, img)
