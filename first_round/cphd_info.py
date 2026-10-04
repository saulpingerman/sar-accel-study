#!/usr/bin/env python3
"""Print what is needed to process a CPHD file: array sizes, domain, sign
convention, frequency grid, and whether per-vector parameters are fixed."""
import sys
import numpy as np
from sarpy.io.phase_history.converter import open_phase_history

for path in sys.argv[1:]:
    r = open_phase_history(path)
    m = r.cphd_meta
    ch = m.Data.Channels[0]
    print('==', path)
    print('vectors', ch.NumVectors, 'samples', ch.NumSamples, 'format', m.Data.SignalArrayFormat,
          'domain', m.Global.DomainType, 'SGN', m.Global.SGN)
    p = m.Channel.Parameters[0]
    print('FXFixed', p.FXFixed, 'TOAFixed', p.TOAFixed, 'SRPFixed', p.SRPFixed, 'FxC', p.FxC, 'FxBW', p.FxBW)
    for name in ('SC0', 'SCSS', 'FX1', 'FX2', 'TOA1', 'TOA2', 'aFDOP', 'aFRR1', 'aFRR2', 'AmpSF', 'TxTime'):
        try:
            v = r.read_pvp_variable(name, 0)
            print(f'  {name:6s} first {v[0]!r} last {v[-1]!r} min {np.min(v)!r} max {np.max(v)!r}')
        except Exception as e:
            print(f'  {name}: {type(e).__name__} {e}')
    tx, rcv, srp = (r.read_pvp_variable(n, 0) for n in ('TxPos', 'RcvPos', 'SRPPos'))
    print('  SRP first', srp[0], 'spread', np.ptp(srp, axis=0))
    a = 0.5 * (tx + rcv) - srp
    rng = np.linalg.norm(a, axis=1)
    print('  range to SRP: %.1f to %.1f km; tx-rcv separation %.2f m; track length %.1f km' %
          (rng.min() / 1e3, rng.max() / 1e3, np.linalg.norm(tx - rcv, axis=1).mean(), np.linalg.norm(a[-1] - a[0]) / 1e3))
    u = a / rng[:, None]
    print('  aperture angle %.3f deg' % np.degrees(np.arccos(np.clip(u[0] @ u[-1], -1, 1))))
    blk = r.read_chip((ch.NumVectors // 2, ch.NumVectors // 2 + 2), (0, 8), index=0) if hasattr(r, 'read_chip') else None
    print('  sample', None if blk is None else (blk.dtype, blk.shape, blk.ravel()[:3]))
