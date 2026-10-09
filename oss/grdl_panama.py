"""GRDL (github.com/GEOINT/grdl) polar format and fast factorized backprojection on the Umbra Panama CPHD, run as
published (default parameters, no weighting of its own). Like every other implementation compared, it receives the
reference's input: the phase history multiplied by the same separable Taylor window (nbar 4, 35 dB). Each algorithm forms the whole scene on its own grid; the image, the
timing and every small numeric attribute of the algorithm object (grid vectors, spacings, reference point) are saved
for scoring against the reference by registration.   python -I grdl_panama.py <cphd> <out dir> pfa|ffbp"""
import sys, os, time, json
import numpy as np
from scipy.signal.windows import taylor
from grdl.IO.sar.cphd import CPHDReader
from grdl.image_processing.sar.image_formation import PolarFormatAlgorithm, FastBackProjection, PolarGrid, CollectionGeometry

cphd, out, which = sys.argv[1:4]
os.makedirs(out, exist_ok=True)


def small_attrs(obj):
    """Numeric attributes with at most 64 elements, for reconstructing the output grid."""
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
sig = (sig * (taylor(P, nbar=4, sll=35.0, norm=False)[:, None] * taylor(K, nbar=4, sll=35.0, norm=False)[None, :])).astype(np.complex64)
rec = dict(algorithm=which, read_seconds=t_read, signal_shape=list(sig.shape), signal_dtype=str(sig.dtype))
t0 = time.perf_counter()
if which == 'pfa':
    geom = CollectionGeometry(meta, slant=True)
    grid = PolarGrid(geom)
    sgn = int(os.environ.get('GRDL_SGN') or meta.global_params.phase_sgn)   # Capella's files declare +1 but follow -1
    rec['phase_sgn'] = sgn
    alg = PolarFormatAlgorithm(grid=grid, phase_sgn=sgn)
    img = alg.form_image(sig, geom)
    rec['grid'] = {k: (np.asarray(v).tolist() if np.asarray(v).dtype.kind in 'fiub' else str(v)) for k, v in alg.get_output_grid().items()}
    rec['geometry_attrs'] = small_attrs(geom)
    rec['polar_grid_attrs'] = small_attrs(grid)
else:
    if os.environ.get('GRDL_SGN'):
        meta.global_params.phase_sgn = int(os.environ['GRDL_SGN'])
    alg = FastBackProjection(meta)
    img = alg.form_image(sig)
    rec['grid'] = {k: (np.asarray(v).tolist() if np.asarray(v).dtype.kind in 'fiub' else str(v)) for k, v in alg.get_output_grid().items()}
rec['seconds'] = time.perf_counter() - t0
rec['algorithm_attrs'] = small_attrs(alg)
rec['image_shape'] = list(img.shape)
np.save(f'{out}/grdl_{which}.npy', img.astype(np.complex64))
json.dump(rec, open(f'{out}/grdl_{which}.json', 'w'), indent=1)
print(which, 'formed', img.shape, f"{rec['seconds']:.1f} s (read {t_read:.1f} s)")
