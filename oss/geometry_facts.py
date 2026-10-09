"""Geometry facts of the released FastSAR for the three Umbra collections, from its public modules (no device): the
factorized plan the formers choose (final tile T from the predicted error, and per level the divisors of the image
sides and the decimations of the samples and pulses; the CUDA float16 former always uses T = 32), and polar
format's FFT sizes and largest planar-wavefront displacement over the image.
   python -I geometry_facts.py <out .json> <scene .npz> ..."""
import sys, json, os, subprocess
import zipfile
import numpy as np
import fastsar
from fastsar import api, ffbp2, pfa2
out = dict(fastsar=subprocess.run(['git', '-C', os.path.dirname(fastsar.__path__[0]), 'log', '-1', '--format=%H'], capture_output=True, text=True).stdout.strip())
for path in sys.argv[2:]:
    d = np.load(path)
    ant, fmin, df = np.asarray(d['ant'], np.float64), float(d['fmin']), float(d['df'])
    with zipfile.ZipFile(path) as z, z.open('S.npy') as fh:      # the shape from the array header, without loading it
        ver = np.lib.format.read_magic(fh)
        P, K = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(fh)[0]
    nx, ny, spx, spy, e1, e2 = int(d['nx']), int(d['ny']), float(d['spx']), float(d['spy']), np.asarray(d['e1'], float), np.asarray(d['e2'], float)
    s = os.path.basename(path).split('.')[0]
    col = api.Collect(fmin=fmin, df=df, K=K, ant=ant, res=0.5)
    T, err = ffbp2.choose_T(ant, fmin + K * df, nx, ny, spx, spy, e1, e2, -40.0)
    r = dict(P=int(P), K=int(K), T=int(T), predicted_error_db=float(err))
    for t in sorted({int(T), 32}):
        pl = ffbp2.make_plan(col, nx, ny, spx, spy, T=t, nlev=3, pmax=0.4, e1=e1, e2=e2)
        r[f'levels_T{t}'] = [dict(sx=l['sx'], sy=l['sy'], Dk=l['Dk'], Dp=l['Dp'], K=l['K'], P=l['P'], Ko=l['Ko'], Po=l['Po']) for l in pl['levels']]
    g = pfa2.geometry(col, nx, ny, spx, spy, e1=e1, e2=e2, guard=300.0)
    dist = pfa2.distortion(col, nx, ny, spx, spy, e1, e2)
    r['pfa'] = dict(nfx=int(g['nfx']), nfy=int(g['nfy']), nkr=int(g['nkr']), nka=int(g['nka']),
                    max_shift_px=list(dist['max_shift_px']), max_shift_m=[dist['max_shift_px'][0] * spx, dist['max_shift_px'][1] * spy])
    out[s] = r
    print(s, json.dumps(r, default=str), flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1, default=str)
