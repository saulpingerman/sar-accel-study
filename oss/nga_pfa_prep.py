"""Inputs for the NGA polar format (pfa_mem) from the Panama CPHD: the phase history with the reference's Taylor
windows, as complex64 [pulses, samples] (which Octave reads as samples x pulses, one pulse per column), and the
per-pulse TxPos, RcvPos, SRPPos (ECF), SC0 and SCSS.   python -I nga_pfa_prep.py <cphd> <work dir>"""
import sys, json
import numpy as np
from scipy.signal.windows import taylor
from sarpy.io.phase_history.converter import open_phase_history
cphd, W = sys.argv[1:3]
r = open_phase_history(cphd); m = r.cphd_meta
P, K = m.Data.Channels[0].NumVectors, m.Data.Channels[0].NumSamples
S = r.read_chip((0, P), (0, K), index=0).astype(np.complex64)
assert m.Global.SGN < 0
S *= (taylor(P, nbar=4, sll=35.0, norm=False)[:, None] * taylor(K, nbar=4, sll=35.0, norm=False)[None, :]).astype(np.float32)
S.tofile(f'{W}/pfa_ph.c64')
nb = np.concatenate([r.read_pvp_variable(n, 0).reshape(P, -1) for n in ('TxPos', 'RcvPos', 'SRPPos', 'SC0', 'SCSS')], 1).astype(np.float64)
np.asfortranarray(nb).T.tofile(f'{W}/pfa_nb.f64')        # column-major [P, 11] for Octave
json.dump(dict(P=P, K=K), open(f'{W}/pfa_dims.json', 'w'))
print('pulses', P, 'samples', K, flush=True)
