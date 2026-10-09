"""GRDL (github.com/GEOINT/grdl) stripmap algorithms on a Capella stripmap CPHD, run as published (default parameters):
range-Doppler (RangeDopplerAlgorithm), subaperture polar format (StripmapPFA) and fast factorized backprojection
(FastBackProjection). Like every implementation compared, it receives the phase history multiplied by the Taylor window
(nbar 4, 35 dB) along frequency. GRDL keeps the CPHD's declared phase sign: run on the 2021 Capella stripmap collection,
its range-Doppler image focuses with the declared sign (+1) and not with the sign FastSAR's reader uses (-1), so GRDL's
convention already accounts for Capella's files. The image, timing and the small numeric attributes of each algorithm object
are saved.   python -I grdl_strip.py <cphd> <out dir> rda|smpfa|ffbp"""
import sys, os, time, json
import numpy as np
from scipy.signal.windows import taylor
from grdl.IO.sar.cphd import CPHDReader
from grdl.image_processing.sar.image_formation import RangeDopplerAlgorithm, StripmapPFA, FastBackProjection

cphd, out, which = sys.argv[1:4]
os.makedirs(out, exist_ok=True)


def small_attrs(obj):
    d = {}
    for k, v in vars(obj).items():
        try:
            a = np.asarray(v)
        except Exception:
            continue
        if a.dtype.kind in 'fiub' and a.size <= 64:
            d[k] = a.tolist()
    return d


t0 = time.perf_counter()
with CPHDReader(cphd) as r:
    meta = r.metadata
    sig = r.read_full()
t_read = time.perf_counter() - t0
if sig.dtype.names:
    sig = sig['real'].astype(np.float32) + 1j * sig['imag'].astype(np.float32)
P, K = sig.shape
sig = (sig * taylor(K, nbar=4, sll=35.0, norm=False)[None, :]).astype(np.complex64)
rec = dict(algorithm=which, read_seconds=t_read, signal_shape=[P, K], phase_sgn=meta.global_params.phase_sgn)
t0 = time.perf_counter()
if which == 'rda':
    alg = RangeDopplerAlgorithm(meta)
elif which == 'smpfa':
    alg = StripmapPFA(meta)
else:
    alg = FastBackProjection(meta)
img = alg.form_image(sig, None)
rec['seconds'] = time.perf_counter() - t0
try:
    rec['grid'] = {k: (np.asarray(v).tolist() if np.asarray(v).dtype.kind in 'fiub' else str(v)) for k, v in alg.get_output_grid().items()}
except Exception as e:
    rec['grid_error'] = repr(e)
rec['algorithm_attrs'] = small_attrs(alg)
rec['image_shape'] = list(img.shape)
np.save(f'{out}/grdl_{which}.npy', img.astype(np.complex64))
json.dump(rec, open(f'{out}/grdl_{which}.json', 'w'), indent=1, default=str)
print(which, 'formed', img.shape, f"{rec['seconds']:.1f} s (read {t_read:.1f} s)", flush=True)
