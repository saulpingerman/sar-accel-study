"""Per-stage device time of the CUDA factorized image on one scene.   python profile_cuda.py --data /tmp/v2/panama.npz"""
import argparse, json, time
import numpy as np, cupy as cp
import v2_prep
from sarbench import ffbp2, ffbp_cuda
ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--ng', type=int, default=8); ap.add_argument('--out', default='profile_cuda.json')
a = ap.parse_args()
from scipy.signal.windows import taylor
col, S, grid = v2_prep.load(a.data)
nx, ny, spx, spy = grid['nx'], grid['ny'], grid['spx'], grid['spy']
P, K = S.shape
wp = taylor(P, nbar=4, sll=35.0, norm=False).astype(np.float32); wk = taylor(K, nbar=4, sll=35.0, norm=False).astype(np.float32)
plan = ffbp2.make_plan(col, nx, ny, spx, spy, T=32, nlev=3, pmax=0.4, e1=np.asarray(grid['e1']), e2=np.asarray(grid['e2']))
coll = ffbp2.collection_arrays(plan, col.ant)
form = ffbp_cuda.make_ffbp_cuda(plan, coll)
Sd = cp.asarray(S) * cp.asarray(wp)[:, None] * cp.asarray(wk)[None, :]
out = form(Sd, ng=a.ng); cp.cuda.Stream.null.synchronize(); del out
t = time.perf_counter(); out = form(Sd, ng=a.ng); cp.cuda.Stream.null.synchronize(); total = time.perf_counter() - t; del out
ffbp_cuda.PROFILE.clear(); ffbp_cuda.PROFILE['on'] = True
t = time.perf_counter(); out = form(Sd, ng=a.ng); cp.cuda.Stream.null.synchronize(); ptotal = time.perf_counter() - t
prof = {k: v for k, v in ffbp_cuda.PROFILE.items() if k != 'on'}
print('image', round(total, 3), 's (profiled run', round(ptotal, 3), 's); stages:', {k: round(v, 3) for k, v in sorted(prof.items(), key=lambda kv: -kv[1])}, 'unaccounted', round(ptotal - sum(prof.values()), 3), flush=True)
json.dump(dict(image_s=total, profiled_s=ptotal, stages=prof), open(a.out, 'w'), indent=1)
