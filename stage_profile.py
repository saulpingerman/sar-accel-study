#!/usr/bin/env python3
"""Where the time goes in the factorized algorithm on this device: the first level's stages timed one at a
time on the native problem (one first-level tile of the real collection), each form of each stage.

  python stage_profile.py --data /tmp/v2/panama.npz --label tpu-v5e --out /tmp/v2/profile_tpu-v5e.json
"""
import argparse
import json
import math
import time

import numpy as np


def timed(fn, *args, reps=3):
    import jax
    out = fn(*args)
    jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t)
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--label', default='dev')
    ap.add_argument('--out', required=True)
    ap.add_argument('--policies', default='fp32_fast,bf16_mm,fp32')
    ap.add_argument('--T', type=int, default=32)
    ap.add_argument('--levels', type=int, default=3)
    ap.add_argument('--pmax', type=float, default=0.4)
    a = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from jax import lax
    import prep
    from dev import ffbp2
    col, S, grid = prep.load(a.data)
    P, K = S.shape
    plan = ffbp2.make_plan(col, grid['nx'], grid['ny'], grid['spx'], grid['spy'], T=a.T, nlev=a.levels, pmax=a.pmax, e1=grid['e1'], e2=grid['e2'])
    lv = plan['levels'][0]
    Ko, Po, Dk, Dp, Cn = lv['Ko'], lv['Po'], lv['Dk'], lv['Dp'], lv['C']
    res = dict(label=a.label, device=str(jax.devices()[0]), P=P, K=K, Ko=Ko, Po=Po, Dk=Dk, Dp=Dp, children=Cn,
               fir_k_taps=lv['fir_k']['L'] if lv['fir_k'] else 0, fir_p_taps=lv['fir_p']['L'] if lv['fir_p'] else 0, stages=[])
    print(json.dumps({k: v for k, v in res.items() if k != 'stages'}), flush=True)
    coll = ffbp2.collection_arrays(plan, col.ant)
    for pol in a.policies.split(','):
        p = ffbp2.POLICIES[pol]
        ew, mm, prec = jnp.dtype(p['ew']), jnp.dtype(p['mm']), p['prec']
        f = jnp.float32
        hre, him, _ = ffbp2.prepare(pol, S)
        c0 = jnp.asarray(np.asarray(coll['levels'][0]['c0']).reshape(-1, P)[:1].astype(np.float32))        # one child's phases [1, P]
        sl = jnp.asarray(np.asarray(coll['levels'][0]['slope']).reshape(-1, P)[:1].astype(np.float32))
        two_pi = 2.0 * math.pi

        def ramp_direct(c0, sl, n, centre):
            kk = jnp.arange(n, dtype=jnp.int32).astype(f) - float(centre)
            cyc = c0[..., None] + kk * sl[..., None]
            ang = (cyc - jnp.round(cyc)) * two_pi
            return jnp.cos(ang).astype(ew), jnp.sin(ang).astype(ew)

        def ramp_split(c0, sl, n, centre):
            Bq = int(math.ceil(math.sqrt(n)))
            nh = -(-n // Bq)
            kh = jnp.arange(nh, dtype=jnp.int32).astype(f) * float(Bq) - float(centre)
            ch = c0[..., None] + kh * sl[..., None]
            ah = (ch - jnp.round(ch)) * two_pi
            cl = jnp.arange(Bq, dtype=jnp.int32).astype(f) * sl[..., None]
            al = (cl - jnp.round(cl)) * two_pi
            chh, shh, cll, sll = jnp.cos(ah), jnp.sin(ah), jnp.cos(al), jnp.sin(al)
            lead = c0.shape
            cs = (chh[..., :, None] * cll[..., None, :] - shh[..., :, None] * sll[..., None, :]).reshape(lead + (nh * Bq,))[..., :n]
            sn = (chh[..., :, None] * sll[..., None, :] + shh[..., :, None] * cll[..., None, :]).reshape(lead + (nh * Bq,))[..., :n]
            return cs.astype(ew), sn.astype(ew)

        def rotate_with(rampf):
            def rot(pre, pim, c0, sl):
                cs, sn = rampf(c0, sl, K, (K - 1) / 2.0)
                return jnp.concatenate([pre[None] * cs - pim[None] * sn, pre[None] * sn + pim[None] * cs], axis=0)
            return rot

        rows = []
        for name, fn, args in (('ramp_direct', jax.jit(lambda c, s: ramp_direct(c, s, K, (K - 1) / 2.0)), (c0, sl)),
                               ('ramp_split', jax.jit(lambda c, s: ramp_split(c, s, K, (K - 1) / 2.0)), (c0, sl)),
                               ('rotate_direct', jax.jit(rotate_with(ramp_direct)), (hre, him, c0, sl)),
                               ('rotate_split', jax.jit(rotate_with(ramp_split)), (hre, him, c0, sl))):
            try:
                rows.append((name, timed(fn, *args)))
            except Exception as e:
                rows.append((name, None))
                print(pol, name, 'FAILED', str(e)[:200], flush=True)
        # decimation stages on a rotated pair [2, P, K]
        x = jax.jit(rotate_with(ramp_split))(hre, him, c0, sl)
        static = ffbp2.static_arrays(pol, plan, 'dense')
        static_c = ffbp2.static_arrays(pol, plan, 'conv')
        la, lac = static['levels'][0], static_c['levels'][0]

        def mmul(u, v):
            return jnp.matmul(u.astype(mm), v, precision=prec, preferred_element_type=f)

        def dk_dense(x):
            return mmul(x, la['Fk'])

        def dp_dense(y):
            return jnp.swapaxes(mmul(jnp.swapaxes(y, 1, 2), la['Fp']), 1, 2)

        fk, fp = lv['fir_k'], lv['fir_p']

        def dk_conv(x):
            xi = x.reshape((-1, 1, 1, K)).astype(mm)
            y = lax.conv_general_dilated(xi, lac['kk'].reshape(1, 1, 1, fk['L']), (1, Dk), ((0, 0), (fk['pl'], fk['pr'])), precision=prec, preferred_element_type=f)
            return y[:, 0, 0, :Ko].reshape(2, P, Ko)

        def dp_conv(y):
            yi = y[:, None].astype(mm)
            z = lax.conv_general_dilated(yi, lac['kp'].reshape(1, 1, fp['L'], 1), (Dp, 1), ((fp['pl'], fp['pr']), (0, 0)), precision=prec, preferred_element_type=f)
            return z[:, 0, :Po, :]

        # block-banded form: outputs in blocks of b, each from its own window of the input
        def banded(x, F, D, Lk, pl, n_out, b=256):
            """x [..., n_in] @ banded F [n_in, n_out] as a batched product over output blocks of b."""
            nb = -(-n_out // b)
            win = b * D + 2 * Lk
            xp = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(Lk, nb * b * D + win)])
            idx = (jnp.arange(nb) * b * D)[:, None] + jnp.arange(win)[None, :]
            xs = xp[..., idx]                                                              # [..., nb, win]
            xs = jnp.moveaxis(xs, -2, 0)                                                    # [nb, ..., win]
            Fp_ = np.pad(np.asarray(F), ((Lk, nb * b * D + win), (0, nb * b - n_out)))
            Fb = jnp.asarray(np.stack([Fp_[j * b * D:j * b * D + win, j * b:(j + 1) * b] for j in range(nb)]))     # [nb, win, b]
            y = jnp.einsum('n...w,nwb->...nb', xs.astype(mm), Fb.astype(mm), precision=prec, preferred_element_type=f)
            return y.reshape(y.shape[:-2] + (nb * b,))[..., :n_out]

        def dk_banded(x):
            return banded(x, la['Fk'], Dk, fk['L'], fk['pl'], Ko)

        def dp_banded(y):
            return jnp.swapaxes(banded(jnp.swapaxes(y, 1, 2), la['Fp'], Dp, fp['L'], fp['pl'], Po), 1, 2)

        y = None
        for name, fn in (('dec_k_dense', dk_dense), ('dec_k_conv', dk_conv), ('dec_k_banded', dk_banded)):
            try:
                jf = jax.jit(fn)
                t = timed(jf, x)
                out = jf(x)
                if y is None:
                    y = out
                    err = 0.0
                else:
                    err = float(10 * np.log10(np.sum(np.abs(np.asarray(out, np.float64) - np.asarray(y, np.float64)) ** 2) / np.sum(np.abs(np.asarray(y, np.float64)) ** 2) + 1e-300))
                rows.append((name, t, err))
                del out
            except Exception as e:
                rows.append((name, None))
                print(pol, name, 'FAILED', str(e)[:300], flush=True)
        z = None
        for name, fn in (('dec_p_dense', dp_dense), ('dec_p_conv', dp_conv), ('dec_p_banded', dp_banded)):
            try:
                jf = jax.jit(fn)
                t = timed(jf, y)
                out = jf(y)
                if z is None:
                    z = out
                    err = 0.0
                else:
                    err = float(10 * np.log10(np.sum(np.abs(np.asarray(out, np.float64) - np.asarray(z, np.float64)) ** 2) / np.sum(np.abs(np.asarray(z, np.float64)) ** 2) + 1e-300))
                rows.append((name, t, err))
                del out
            except Exception as e:
                rows.append((name, None))
                print(pol, name, 'FAILED', str(e)[:300], flush=True)
        del x, y, z
        for r in rows:
            res['stages'].append(dict(policy=pol, stage=r[0], seconds=r[1], rel_err_db=(r[2] if len(r) > 2 else None)))
            print(pol, r[0], None if r[1] is None else f'{r[1] * 1e3:.1f} ms', '' if len(r) < 3 else f'{r[2]:.1f} dB', flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
