#!/usr/bin/env python3
"""Prepare one Umbra collection for the second round: the full phase history, the antenna path in a local
frame at the scene reference point, and the vendor's native image grid.

The local frame has x along track (cross range), y along ground range away from the radar, and z up, with
the origin at the scene reference point. The image grid is the vendor's own (SICD): its pixel counts and
spacings, in its image plane, whose azimuth (e1) and range (e2) unit vectors are stored in the local frame.
Umbra's products are slant-plane images, so e2 is the line of sight at the aperture centre and e1 is in the
plane of the line of sight and the velocity; both are ground-plane axes when ImagePlane is GROUND.

  python v2_prep.py --cphd x_CPHD.cphd --sicd x_SICD.nitf --out scene.npz
"""
import argparse
import json
import time

import numpy as np

C = 299792458.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cphd', required=True)
    ap.add_argument('--sicd', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', default='')
    a = ap.parse_args()
    from sarpy.io.phase_history.converter import open_phase_history
    from sarpy.io.complex.converter import open_complex
    r = open_phase_history(a.cphd)
    m = r.cphd_meta
    ch = m.Data.Channels[0]
    P, K = ch.NumVectors, ch.NumSamples
    assert m.Global.DomainType == 'FX' and m.Channel.Parameters[0].SRPFixed
    tx, rcv, srp = (r.read_pvp_variable(n, 0) for n in ('TxPos', 'RcvPos', 'SRPPos'))
    sc0, scss = r.read_pvp_variable('SC0', 0), r.read_pvp_variable('SCSS', 0)
    f0, df = float(sc0.mean()), float(scss.mean())
    t = time.time()
    S = r.read_chip((0, P), (0, K), index=0).astype(np.complex64)
    if m.Global.SGN > 0:
        S = np.conj(S)
    apc = 0.5 * (tx + rcv) - srp[0]
    # Some products carry pulses without a valid position or with all-zero samples at the ends of the
    # aperture (27 and 43 of Melbourne's 15,991); they are trimmed, and any interior invalid position is
    # interpolated from its neighbours.
    bad = ~np.isfinite(apc).all(1) | ~np.isfinite(S).all(1) | (np.abs(S).sum(1) == 0)
    good = np.nonzero(~bad)[0]
    lo, hi = int(good[0]), int(good[-1]) + 1
    S, apc, bad = S[lo:hi], apc[lo:hi], bad[lo:hi]
    nfix = int(bad.sum())
    if nfix:
        idx = np.arange(len(apc))
        for c in range(3):
            apc[bad, c] = np.interp(idx[bad], idx[~bad], apc[~bad, c])
        S[~np.isfinite(S)] = 0
    trimmed = dict(leading=lo, trailing=int(P - hi), interior_fixed=nfix)
    P = len(apc)
    up = srp[0] / np.linalg.norm(srp[0])
    mid = apc[P // 2]
    los_h = mid - (mid @ up) * up                           # horizontal direction from the scene to the radar
    yhat = -los_h / np.linalg.norm(los_h)
    xhat = np.cross(yhat, up)
    R = np.stack([xhat, yhat, up])
    ant = apc @ R.T
    rng = np.linalg.norm(ant, axis=1)
    graze = float(np.arcsin(ant[P // 2, 2] / rng[P // 2]))
    u = ant / rng[:, None]
    # the vendor's image grid
    sm = open_complex(a.sicd).sicd_meta
    rows, cols = int(sm.ImageData.NumRows), int(sm.ImageData.NumCols)
    row_ss, col_ss = float(sm.Grid.Row.SS), float(sm.Grid.Col.SS)
    row_u = np.array(sm.Grid.Row.UVectECF.get_array()) @ R.T
    range_is_row = abs(row_u[1]) > abs(row_u[0])
    n_rg, ss_rg = (rows, row_ss) if range_is_row else (cols, col_ss)
    n_az, ss_az = (cols, col_ss) if range_is_row else (rows, row_ss)
    plane = str(sm.Grid.ImagePlane)
    spy = ss_rg
    col_u = np.array(sm.Grid.Col.UVectECF.get_array()) @ R.T
    e2, e1 = (row_u, col_u) if range_is_row else (col_u, row_u)
    e1, e2 = e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)
    assert abs(e1 @ e2) < 1e-6
    los = -ant[P // 2] / rng[P // 2]
    print('image plane: e1', e1.round(4), 'e2', e2.round(4), 'e2.los', round(float(e2 @ los), 6), 'twist', float(getattr(sm.SCPCOA, 'TwistAng', 0.0)))
    info = dict(name=a.name, cphd=a.cphd.split('/')[-1], vectors=int(P), samples=int(K), f0=f0, df=df, trimmed=trimmed,
                grid_drift_hz=float(max(np.ptp(sc0), np.ptp(scss) * K)), bandwidth_hz=float(df * K),
                range_km=[float(rng.min() / 1e3), float(rng.max() / 1e3)], graze_deg=float(np.degrees(graze)),
                aperture_deg=float(np.degrees(np.arccos(np.clip(u[0] @ u[-1], -1, 1)))),
                sicd=dict(rows=rows, cols=cols, row_ss=row_ss, col_ss=col_ss, plane=plane, type=str(sm.Grid.Type), range_is_row=bool(range_is_row)),
                nx=int(n_az), ny=int(n_rg), spx=float(ss_az), spy=float(spy), e1=[float(v) for v in e1], e2=[float(v) for v in e2],
                twist_deg=float(getattr(sm.SCPCOA, 'TwistAng', 0.0)), squint_deg=float(getattr(sm.SCPCOA, 'DopplerConeAng', 90.0)) - 90.0,
                extent_m=[float(n_az * ss_az), float(n_rg * spy)],
                unambiguous_range_m=float(C / (2.0 * df)), read_s=time.time() - t)
    print(json.dumps(info, indent=1))
    np.savez(a.out, S=S, ant=ant, fmin=f0, df=df, nx=n_az, ny=n_rg, spx=ss_az, spy=spy, e1=e1, e2=e2, info=json.dumps(info))


def load(path):
    """-> (collection, S complex64 [P, K], grid dict)"""
    from sarbench.sim import Collect
    d = np.load(path)
    S = d['S']
    col = Collect(fmin=float(d['fmin']), df=float(d['df']), K=S.shape[1], ant=d['ant'], res=0.5)
    gx, gy = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    if 'nx' not in d:                      # a first-round file (decimated patch): the grid comes from the command line
        return col, S.astype(np.complex64), dict(nx=0, ny=0, spx=0.25, spy=0.25, e1=gx, e2=gy, info={})
    return col, S, dict(nx=int(d['nx']), ny=int(d['ny']), spx=float(d['spx']), spy=float(d['spy']),
                        e1=d['e1'] if 'e1' in d else gx, e2=d['e2'] if 'e2' in d else gy, info=json.loads(str(d['info'])))


if __name__ == '__main__':
    main()
