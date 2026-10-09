"""ISCE3's region images of the modes comparison scored as on Panama: after removal of a fitted linear phase ramp (rad
per pixel along each axis) and the best complex gain, against the float64 reference with ISCE3's own aperture.
    python -I modes_isce3_ramp.py <dir or gs:// prefix with isce3_<dev>_r{0,1,2}.npz> <dev> <out json>"""
import sys, json, subprocess, io
import numpy as np
from scipy.optimize import minimize

src, dev, out = sys.argv[1:4]


def load(k):
    p = f'{src.rstrip("/")}/isce3_{dev}_r{k}.npz'
    if p.startswith('gs://'):
        return np.load(io.BytesIO(subprocess.run(['gcloud', 'storage', 'cat', p], capture_output=True, check=True).stdout))
    return np.load(p)


res = {}
for k in range(3):
    z = load(k); a, b = z['img'].astype(np.complex128), z['ref'].astype(np.complex128)
    n0, n1 = a.shape
    I, J = np.meshgrid(np.arange(n0) - n0 / 2, np.arange(n1) - n1 / 2, indexing='ij')
    xc = a * np.conj(b)
    q0 = [np.angle(np.sum(xc[1:] * np.conj(xc[:-1]))), np.angle(np.sum(xc[:, 1:] * np.conj(xc[:, :-1])))]
    q = minimize(lambda q: -np.abs(np.sum(xc * np.exp(-1j * (q[0] * I + q[1] * J)))), q0, method='Nelder-Mead',
                 options=dict(xatol=1e-9, fatol=1e-12)).x
    err = lambda x: float(10 * np.log10(np.sum(np.abs(b - (np.vdot(x, b) / np.vdot(x, x)) * x) ** 2) / np.sum(np.abs(b) ** 2)))
    res[f'r{k}'] = dict(ramp_axis0=float(q[0]), ramp_axis1=float(q[1]), error_db=err(a),
                        error_after_ramp_db=err(a * np.exp(-1j * (q[0] * I + q[1] * J))))
    print(k, {x: round(v, 5) for x, v in res[f'r{k}'].items()}, flush=True)
json.dump(res, open(out, 'w'), indent=1)
