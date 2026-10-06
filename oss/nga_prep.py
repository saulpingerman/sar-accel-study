"""Write inputs for NGA's bpBasic.m (Octave): a simulated scene and three Panama regions, as raw binary files."""
import sys, json, numpy as np
from scipy.signal.windows import taylor
W = sys.argv[1]
sys.path.insert(0, f'{W}/FastSAR')
from fastsar import sim
C = 299792458.0
def write_case(name, S, ant, fmin, df, pix):
    P, K = S.shape
    Sw = (S * taylor(P, nbar=4, sll=35.0, norm=False)[:, None] * taylor(K, nbar=4, sll=35.0, norm=False)[None, :]).astype(np.complex64)
    Sw.view(np.float32).tofile(f'{W}/{name}_ph.f32')                       # [P][K][re, im] -> Octave [2K, P]
    np.asarray(ant, np.float64).tofile(f'{W}/{name}_ant.f64')               # [P][3]
    np.linalg.norm(ant, axis=1).astype(np.float64).tofile(f'{W}/{name}_r0.f64')
    np.asarray(pix, np.float64).tofile(f'{W}/{name}_pix.f64')               # [N][3]
    json.dump(dict(P=P, K=K, minF=float(fmin), deltaF=float(df), N=int(pix.shape[0])), open(f'{W}/{name}_meta.json', 'w'))
# simulated scene (as in the RITSAR check): 128 x 128 pixels at 0.5 m, broadside track along y at 5 km
rng = np.random.default_rng(1)
col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
tg = np.stack([rng.uniform(-25, 25, 40), rng.uniform(-25, 25, 40), np.zeros(40)], 1)
amp = rng.standard_normal(40) + 1j * rng.standard_normal(40)
S = sim.simulate_brute(col, tg, amp)
e1, e2 = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
X, Y = np.meshgrid((np.arange(128) - 64.0) * 0.5, (np.arange(128) - 64.0) * 0.5, indexing='ij')
write_case('sim', S, col.ant, col.fmin, col.df, X.ravel()[:, None] * e1 + Y.ravel()[:, None] * e2)
np.savez(f'{W}/sim_geom.npz', ant=col.ant, fmin=col.fmin, df=col.df, K=col.K, S=S)
# Panama: the three regions of the RITSAR comparison, on our pixel grid
d = np.load(f'{W}/panama.npz')
nx, ny, spx, spy = int(d['nx']), int(d['ny']), float(d['spx']), float(d['spy'])
pix = []
for (i0, j0) in ((3400, 6700), (9500, 1150), (6400, 2650)):
    I, J = np.meshgrid(np.arange(i0, i0 + 512), np.arange(j0, j0 + 512), indexing='ij')
    pix.append(((I - nx / 2.0) * spx).ravel()[:, None] * d['e1'] + ((J - ny / 2.0) * spy).ravel()[:, None] * d['e2'])
write_case('panama', d['S'], d['ant'], float(d['fmin']), float(d['df']), np.concatenate(pix))
