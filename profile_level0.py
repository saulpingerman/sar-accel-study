"""Per-stage timing of the first factorization level on the accelerator: the XLA rotation, the dense (or convolution)
decimation, the pulse decimation, the fused Pallas kernel, and the whole first-level tile, for one scene and policy.

    python profile_level0.py --data /tmp/v2/panama.npz --policy fp32_fast --pb 128 --chunk 512
"""
import argparse, json, time

import numpy as np

import v2_prep
from sarbench import ffbp2


def timeit(fn, *args, reps=5):
    import jax
    out = fn(*args); jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); out = fn(*args); jax.block_until_ready(out); ts.append(time.perf_counter() - t)
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--policy', default='fp32_fast')
    ap.add_argument('--pb', type=int, default=128)
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--out', default='profile.json')
    ap.add_argument('--filters', default='', help='comma list; default dense,pallas on a TPU')
    ap.add_argument('--nc', type=int, default=8)
    ap.add_argument('--ng', type=int, default=8)
    ap.add_argument('--gen', type=int, default=3)
    a = ap.parse_args()
    import jax, jax.numpy as jnp
    from jax import lax
    from scipy.signal.windows import taylor
    col, S, grid = v2_prep.load(a.data)
    nx, ny, spx, spy = grid['nx'], grid['ny'], grid['spx'], grid['spy']
    e1, e2 = np.asarray(grid['e1'], np.float64), np.asarray(grid['e2'], np.float64)
    P, K = S.shape
    wp = taylor(P, nbar=4, sll=35.0, norm=False).astype(np.float32)
    wk = taylor(K, nbar=4, sll=35.0, norm=False).astype(np.float32)
    Sw = S * wp[:, None] * wk[None, :]
    plan = ffbp2.make_plan(col, nx, ny, spx, spy, T=32, nlev=3, pmax=0.4, e1=e1, e2=e2)
    lv = plan['levels'][0]
    print('level 0', {k: lv[k] for k in ('P', 'K', 'Dk', 'Dp', 'Po', 'Ko', 'sx', 'sy')}, flush=True)
    pol = a.policy
    p = ffbp2.POLICIES[pol]
    mm, prec = jnp.dtype(p['mm']), p['prec']
    f = jnp.float32
    hre, him, scale = ffbp2.prepare(pol, Sw)
    res = dict(level0={k: int(lv[k]) for k in ('P', 'K', 'Dk', 'Dp', 'Po', 'Ko')}, policy=pol, device=str(jax.devices()[0]))
    platform = jax.devices()[0].platform
    filts = a.filters.split(',') if a.filters else (['dense', 'pallas'] if platform == 'tpu' else ['conv', 'pallas'])
    for filt in filts:
        static = ffbp2.static_arrays(pol, plan, filt)
        arrs = ffbp2.device_arrays(pol, plan, ffbp2.collection_arrays(plan, col.ant), static)
        la = arrs['levels'][0]
        G = lv['sx'] * lv['sy']
        for trig in (['direct'] if filt == 'pallas' else ['split', 'direct']):
            fn = ffbp2.make_ffbp(pol, plan, filt, 1 << 26, trig, pallas_pb=a.pb, pallas_chunk=a.chunk, pallas_nc=a.nc, pallas_ng=a.ng, pallas_gen=a.gen)
            # whole image through the production path
            t_all = timeit(lambda: fn(hre, him, arrs), reps=2)
            res[f'{filt}_{trig}_image_s'] = t_all
            print(filt, trig, 'image', round(t_all, 3), flush=True)
            # one first-level tile, stage by stage (times G for the image)
            st = fn.stages
            children, final, device_phases, levels_ = st['children'], st['final'], st['device_phases'], st['levels']
            HL = st['host_levels']
            stage_t = {}
            la0, lv0 = arrs['levels'][0], levels_[0]
            g = 0
            f0 = jax.jit(lambda hre, him, c0, sl: children(hre, him, c0, sl, la0, dict(lv0, C=1)))
            stage_t['L0'] = timeit(f0, hre, him, la0['c0'][0, g][None], la0['slope'][0, g][None], reps=3)
            A, Bm = f0(hre, him, la0['c0'][0, g][None], la0['slope'][0, g][None])
            for i in range(1, len(levels_)):
                la_i, lvl = arrs['levels'][i], levels_[i]
                if i < HL:
                    fi = jax.jit(lambda a, b, la=la_i, lvl=lvl: children(a[0], b[0], la['c0'][g], la['slope'][g], la, lvl))
                    stage_t[f'L{i}'] = timeit(fi, A, Bm, reps=3)
                    A, Bm = fi(A, Bm)
                else:
                    Gn = st['G']
                    refs = la_i['ref'].reshape(Gn, A.shape[0], 3)[g]

                    def fi(a, b, refs, la=la_i, lvl=lvl):
                        def per_parent(x):
                            c0, sl = device_phases(la, x[2], lvl)
                            return children(x[0], x[1], c0, sl, la, lvl)
                        a, b = lax.map(per_parent, (a, b, refs))
                        return a.reshape((-1,) + a.shape[2:]), b.reshape((-1,) + b.shape[2:])
                    fi = jax.jit(fi)
                    stage_t[f'L{i}'] = timeit(fi, A, Bm, refs, reps=3)
                    A, Bm = fi(A, Bm, refs)
            fa = arrs['final']
            cen = fa['cen'].reshape(st['G'], -1, 3)[g]
            ff = jax.jit(lambda a, b, cen: final(a, b, dict(fa, cen=cen)))
            stage_t['final'] = timeit(ff, A, Bm, cen, reps=3)
            stage_t['final_tiles'] = int(A.shape[0])
            stage_t['final_PQ'] = [int(v) for v in A.shape[1:]]
            res[f'{filt}_{trig}_tile_stages'] = stage_t
            tot = sum(v for k, v in stage_t.items() if k.startswith('L') or k == 'final')
            print(filt, trig, 'tile stages', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stage_t.items()},
                  'sum x G =', round(tot * st['G'], 3), flush=True)
            if filt == 'pallas2':
                # final stage alone on the real tiles of this first-level tile, each available form
                for mode in (0, 1, 2, 3):
                    if mode >= 2 and platform != 'tpu':
                        continue
                    fnm = ffbp2.make_ffbp(pol, plan, 'pallas2', 1 << 26, trig, pallas_pb=a.pb, pallas_chunk=a.chunk, pallas_nc=a.nc, pallas_ng=a.ng, pallas_final=mode, pallas_gen=a.gen)
                    ffm = jax.jit(lambda a_, b_, cen_, fnm=fnm: fnm.stages['final'](a_, b_, dict(fa, cen=cen_)))
                    tfm = timeit(ffm, A, Bm, cen, reps=3)
                    res[f'final_mode{mode}_s'] = tfm
                    print(f'final stage mode {mode} ({"XLA" if mode == 0 else "direct-trig kernel" if mode == 1 else "recurrence kernel" if mode == 2 else "recurrence, 4 tiles per step"}): {tfm*1e3:.2f} ms per {A.shape[0]} tiles = {tfm/A.shape[0]*1e6:.2f} us per tile', flush=True)
                ngr = st['ng']
                gs = jnp.arange(0, ngr, dtype=jnp.int32)
                t_grp = timeit(lambda: st['one_group'](hre, him, arrs, gs), reps=2)
                res['pallas2_group_s'] = t_grp
                print('pallas2 one group of', ngr, 'tiles', round(t_grp, 3), 'x', st['G'] // ngr, '=', round(t_grp * st['G'] / ngr, 3), flush=True)
                # level-0 children as a group of ng (the production call) and its pieces
                f0g = jax.jit(lambda hre, him, c0, sl: children(hre, him, c0, sl, la0, dict(lv0, C=ngr)))
                t0g = timeit(f0g, hre, him, la0['c0'][0, gs], la0['slope'][0, gs], reps=3)
                res['pallas2_L0_group_s'] = t0g
                print('pallas2 L0 children for', ngr, 'tiles', round(t0g, 4), flush=True)
                # the bare kernel at each level's shapes
                if platform == 'tpu':
                    from sarbench.pallas_ffbp import fused_rotate_dec_k2, pad_columns
                else:
                    from sarbench.pallas_ffbp_gpu import fused_rotate_dec_k2_gpu, pad_columns_gpu as pad_columns
                    gprec = 'highest' if ffbp2.POLICIES[pol]['prec'] == lax.Precision.HIGHEST else None

                    def fused_rotate_dec_k2(A, B, c, s_, band, kc, pb, passes):
                        return fused_rotate_dec_k2_gpu(A, B, c, s_, band, kc, mm_dtype=mm, precision=gprec, pb=pb)
                for li, lvl in enumerate(levels_):
                    if lvl['Dk'] <= 1:
                        continue
                    band = st['bands'][(lvl['P'], lvl['K'], lvl['Dk'])]
                    Pl, Kl, Cl = lvl['P'], lvl['K'], lvl['C']
                    pbl = ffbp2._pulse_block(Pl, a.pb) if platform == 'tpu' else a.pb; Pp = -(-Pl // pbl) * pbl
                    ncl = min(a.nc, Cl if li else ngr)
                    npass = 1 if ffbp2.POLICIES[pol]['prec'] is None else 3
                    xr = pad_columns(jnp.zeros((1, Pp, Kl), jnp.float32), band); xi = xr
                    c0z = jnp.zeros((1, ncl, Pp), jnp.float32) + 0.1; slz = jnp.zeros((1, ncl, Pp), jnp.float32) + 1e-3
                    if platform == 'tpu' and a.gen >= 3:
                        from sarbench.pallas_ffbp import fused_rotate_dec_k3
                        fk = jax.jit(lambda A, B, c, s_, pbl=pbl, band=band, npass=npass: fused_rotate_dec_k3(A, B, c, s_, band, (Kl - 1) / 2.0, pb=pbl, passes=npass))
                    else:
                        fk = jax.jit(lambda A, B, c, s_, pbl=pbl, band=band, npass=npass: fused_rotate_dec_k2(A, B, c, s_, band, (Kl - 1) / 2.0, pb=pbl, passes=npass))
                    try:
                        tk = timeit(fk, xr, xi, c0z, slz, reps=3)
                        res[f'pallas2_kernel_L{li}_nc{ncl}_s'] = tk
                        n_par = 1 if li == 0 else (st['G'] * (1 if li == 1 else levels_[1]['C']))
                        calls = n_par * -(-Cl // ncl) if li else st['G'] // ngr
                        print(f'kernel L{li}: P {Pl} K {Kl} Kpad {band["Kpad"]} win {band["win"]} nb {band["nb"]} pb {pbl} nc {ncl}: {tk*1e3:.3f} ms per call x {calls} calls per image = {tk*calls:.3f} s', flush=True)
                    except Exception as e:
                        print(f'kernel L{li} FAILED', type(e).__name__, str(e)[:200], flush=True)
        if filt == 'pallas2':
            continue
        # the pieces of one first-level child
        c0 = la['c0'][0, 0][None]; sl = la['slope'][0, 0][None]
        two_pi = 2 * np.pi
        kc = (K - 1) / 2.0

        @jax.jit
        def rot_direct(hre, him, c0, sl):
            kk = jnp.arange(K, dtype=jnp.int32).astype(f) - kc
            cyc = c0[..., None] + kk * sl[..., None]
            ang = (cyc - jnp.round(cyc)) * two_pi
            cs, sn = jnp.cos(ang), jnp.sin(ang)
            return hre[None] * cs - him[None] * sn, hre[None] * sn + him[None] * cs

        t_rot = timeit(rot_direct, hre, him, c0, sl)
        res[f'{filt}_rotate_child_s'] = t_rot
        if filt == 'dense':
            Fk = la['Fk']

            @jax.jit
            def dec(zr, zi):
                return (jnp.matmul(zr.astype(mm), Fk, precision=prec, preferred_element_type=f),
                        jnp.matmul(zi.astype(mm), Fk, precision=prec, preferred_element_type=f))
            zr, zi = rot_direct(hre, him, c0, sl)
            t_dec = timeit(dec, zr, zi)
            res['dense_dec_k_child_s'] = t_dec
            Fp = la['Fp']

            @jax.jit
            def decp(y):
                return jnp.swapaxes(jnp.matmul(jnp.swapaxes(y, 1, 2).astype(mm), Fp, precision=prec, preferred_element_type=f), 1, 2)
            y = jnp.concatenate(dec(zr, zi), 0)
            t_decp = timeit(decp, y)
            res['dense_dec_p_child_s'] = t_decp
            print('dense child: rotate', round(t_rot, 4), 'dec_k', round(t_dec, 4), 'dec_p', round(t_decp, 4), 'x', G, 'children =',
                  round(G * (t_rot + t_dec + t_decp), 3), flush=True)
        else:
            if platform == 'tpu':
                from sarbench.pallas_ffbp import band_blocks, pad_columns, fused_rotate_dec_k
                band = band_blocks(np.asarray(lv['Fk']), lv['Dk'])
                npass = 1 if prec is None else 3
                Pp = -(-P // a.pb) * a.pb
                pre_p = pad_columns(jnp.pad(hre.astype(f), ((0, Pp - P), (0, 0))), band)
                pim_p = pad_columns(jnp.pad(him.astype(f), ((0, Pp - P), (0, 0))), band)
                c0p = jnp.pad(c0[0].astype(f), (0, Pp - P)); slp = jnp.pad(sl[0].astype(f), (0, Pp - P))
                for pb in sorted({a.pb, 64, 128}):
                    for chunk in sorted({a.chunk, 512, 1024, 2048}):
                        if band['Kpad'] % chunk:
                            continue
                        Pp2 = -(-P // pb) * pb
                        if Pp2 != Pp:
                            continue
                        fused = jax.jit(lambda A, B, c, s, pb=pb, chunk=chunk: fused_rotate_dec_k(A, B, c, s, band, kc, pb=pb, chunk=chunk, passes=npass))
                        try:
                            t_f = timeit(fused, pre_p, pim_p, c0p, slp)
                            res[f'pallas_fused_child_pb{pb}_c{chunk}_s'] = t_f
                            print('fused child pb', pb, 'chunk', chunk, round(t_f, 4), 'x', G, '=', round(G * t_f, 3), flush=True)
                        except Exception as e:
                            print('fused pb', pb, 'chunk', chunk, 'FAILED', type(e).__name__, str(e)[:120], flush=True)
                # a 64-child batch through lax.map, as the production path does it
                c0s = jnp.stack([c0p] * 8); sls = jnp.stack([slp] * 8)
                fmap = jax.jit(lambda c, s: lax.map(lambda cs: fused_rotate_dec_k(pre_p, pim_p, cs[0], cs[1], band, kc, pb=a.pb, chunk=a.chunk, passes=npass), (c, s)))
                t_m = timeit(fmap, c0s, sls, reps=3)
                res['pallas_fused_8children_map_s'] = t_m
                print('fused 8 children via map', round(t_m, 4), 'per child', round(t_m / 8, 4), flush=True)
            else:
                from sarbench.pallas_ffbp_gpu import band_blocks_gpu, pad_columns_gpu, fused_rotate_dec_k_gpu
                band = band_blocks_gpu(np.asarray(lv['Fk']), lv['Dk'])
                gpu_prec = 'highest' if prec is not None and prec == lax.Precision.HIGHEST else None
                for pb in (16, 32):
                    Pp = -(-P // pb) * pb
                    pre_p = pad_columns_gpu(jnp.pad(hre.astype(f), ((0, Pp - P), (0, 0))), band)
                    pim_p = pad_columns_gpu(jnp.pad(him.astype(f), ((0, Pp - P), (0, 0))), band)
                    c0p = jnp.pad(c0[0].astype(f), (0, Pp - P)); slp = jnp.pad(sl[0].astype(f), (0, Pp - P))
                    fused = jax.jit(lambda A, B, c, s, pb=pb: fused_rotate_dec_k_gpu(A, B, c, s, band, kc, mm_dtype=mm, precision=gpu_prec, pb=pb))
                    try:
                        t_f = timeit(fused, pre_p, pim_p, c0p, slp)
                        res[f'pallas_fused_child_pb{pb}_s'] = t_f
                        print('fused child pb', pb, round(t_f, 4), 'x', G, '=', round(G * t_f, 3), flush=True)
                    except Exception as e:
                        print('fused pb', pb, 'FAILED', type(e).__name__, str(e)[:200], flush=True)
    bytes_child = 2 * 4 * P * K
    res['child_bytes'] = bytes_child
    k = f"{'dense' if platform == 'tpu' else 'conv'}_rotate_child_s"
    if k in res:
        print('one child reads', round(bytes_child / 1e9, 3), 'GB; rotate-only effective bandwidth (24 B/sample)', round(3 * bytes_child / res[k] / 1e9, 1), 'GB/s', flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
