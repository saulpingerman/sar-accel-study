"""Roofline-style bounds of each stage of the factorized kernels on each accelerator, from the inherent work of the
Panama plan, the measured unit rates (ubench_*.json) and the measured stage times (stage profiles).

    python bounds.py            -> prints the table and writes results/v2r/bounds.json (read by build.py)

Work counts come from the plan of record (Panama: three levels (8,8,6,4),(8,7,7,9),(6,5,5,6), final tiles 80 x 79)
and the kernels' block geometry as printed by profile_level0.py (window widths, padded pulse counts).
"""
import json, os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'v2r')
PROF = os.path.join(R, 'profiles')

# Panama plan: level (children per parent, P, K, Po, Ko, Dk, Dp) and the parents per image at each level
LEVELS = [dict(C=64, P=15186, K=14399, Po=3805, Ko=2408, Dk=6, Dp=4, parents=1),
          dict(C=56, P=3805, K=2408, Po=431, Ko=352, Dk=7, Dp=9, parents=64),
          dict(C=30, P=431, K=352, Po=80, Ko=79, Dk=5, Dp=6, parents=3584)]
FINAL = dict(tiles=107520, Pf=80, Qf=79, T=32)
# kernel geometry of the TPU build of record (profile logs): padded pulses, window columns, blocks per row, children per call
TPU_GEOM = [dict(Pp=15360, win=1792, nb=10, nc=8, kob=256, Kpad=16896, fused_p=False), dict(Pp=3840, win=2560, nb=1, nc=8, kob=384, Kpad=2560, fused_p=True), dict(Pp=512, win=512, nb=1, nc=30, kob=128, Kpad=512, fused_p=True)]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def tpu_bounds(lab, ub, prof):
    """Per-stage bounds for a TPU: vector-unit rate on the kernel's rotation mix, HBM, matrix units, sines in XLA."""
    out = []
    rot_rate = ub['pallas_rotate_Gelem_s'] * 1e9
    hbm = ub['hbm_copy_GBps'] * 1e9
    trig = ub['xla_sincos_Gpairs_s'] * 1e9
    st = prof['pallas2_direct_tile_stages'] if 'pallas2_direct_tile_stages' in prof else prof['pallas2_split_tile_stages']
    G = 64
    for i, (lv, g) in enumerate(zip(LEVELS, TPU_GEOM)):
        children = lv['C'] * lv['parents']
        rotated = children * g['Pp'] * g['nb'] * g['win']                                   # samples the kernel rotates
        minimal = children * lv['P'] * lv['K']                                                # samples that must be rotated
        # bytes: parent read per group of nc children (twice where the window has a halo), decimated children written once
        groups = children / g['nc']
        halo = 2 if g['nb'] > 1 else 1
        read_b = groups * g['Pp'] * g['Kpad'] * 8 * halo
        write_b = children * lv['Po'] * lv['Ko'] * 8
        if not g['fused_p']:                                                                  # level 0: the frequency-decimated children pass through memory
            write_b += 2 * children * lv['P'] * lv['Ko'] * 8
        tables_b = 2 * children * g['Pp'] * (128 + 128 * g['nb']) * 8                          # fine and coarse tables, written by XLA and read by the kernel
        trig_n = children * g['Pp'] * (128 + 128 * g['nb'])                                   # sines and cosines per child and pulse (fine table, coarse table per block)
        mxu = children * g['Pp'] * g['nb'] * g['win'] * g['kob'] * 2 * 2 + children * lv['Po'] * g['Pp'] * lv['Ko'] * 2 * 2
        mxu_rate = ub['mxu_4096x4096x4096_TFLOPs'] * 1e12
        b_vpu, b_hbm, b_trig, b_mxu = rotated / rot_rate, (read_b + write_b + tables_b) / hbm, trig_n / trig, mxu / mxu_rate
        measured = (prof['pallas2_L0_group_s'] * G / 16 if i == 0 else st[f'L{i}'] * G)
        out.append(dict(stage=f'level {i}', rotated=rotated, minimal=minimal, bound_vpu=b_vpu, bound_hbm=b_hbm, bound_trig=b_trig, bound_mxu=b_mxu,
                        bound=max(b_vpu, b_hbm, b_trig, b_mxu), bound_sum=b_vpu + b_hbm + b_trig + b_mxu, measured=measured))
    # final stage: tables built by recurrence (6 vector ops per table entry, two tables of T x Pf Qf per tile), one bf16 product per tile
    n = FINAL['Pf'] * FINAL['Qf']
    tab_elems = FINAL['tiles'] * 2 * FINAL['T'] * n * 2                                      # real and imaginary of two tables
    vpu_rate_elems = rot_rate * 2                                                              # the rotation mix is about 12 ops per element; a table entry is about 6
    mxu = FINAL['tiles'] * 2 * (2 * FINAL['T']) * n * (2 * FINAL['T'])
    mxu_rate = ub['mxu_64x10112x64_x420_TFLOPs'] * 1e12
    read_b = FINAL['tiles'] * n * 8
    b_mxu, b_hbm = mxu / mxu_rate, read_b / hbm
    # the vector work of the final stage (two complex recurrences per tile row, four stores per step) was not isolated
    # by a microbenchmark; the matrix bound is the one reported, and the text attributes the remainder to the tables
    out.append(dict(stage='final', rotated=tab_elems, minimal=tab_elems, bound_vpu=0.0, bound_hbm=b_hbm, bound_trig=0.0, bound_mxu=b_mxu,
                    bound=max(b_mxu, b_hbm), bound_sum=None, measured=st['final'] * G))
    return out


def gpu_bounds(ub, prof):
    out = []
    outs = sum(lv['C'] * lv['parents'] * lv['P'] * lv['Ko'] for lv in LEVELS)                 # filter outputs along frequency
    write_b = outs * 8
    read_b = sum(lv['C'] * lv['parents'] / 8 * lv['P'] * lv['K'] * 8 * 1.46 for lv in LEVELS)  # windows loaded once per 8 children, 1.46 halo
    st = prof['stages']
    b_comp = outs / (ub['fir_mix_Gout_s'] * 1e9)
    b_mem = (write_b + read_b) / (ub['hbm_copy_GBps'] * 1e9)
    out.append(dict(stage='levels (rotation and frequency filter)', bound_comp=b_comp, bound_mem=b_mem, bound=max(b_comp, b_mem), bound_sum=b_comp + b_mem, measured=st['rot_fir_k']))
    pouts = sum(lv['C'] * lv['parents'] * lv['Po'] * lv['Ko'] for lv in LEVELS)
    b_mem = (outs * 8 + pouts * 8) / (ub['hbm_copy_GBps'] * 1e9)
    out.append(dict(stage='pulse filter', bound_comp=0.0, bound_mem=b_mem, bound=b_mem, bound_sum=b_mem, measured=st['fir_p']))
    n = FINAL['Pf'] * FINAL['Qf']
    fmas = FINAL['tiles'] * FINAL['T'] * FINAL['T'] * n * 4
    b_comp = fmas / (ub['final_mix_TFMA_s'] * 1e12)
    b_mem = FINAL['tiles'] * n * 8 / (ub['hbm_copy_GBps'] * 1e9)
    out.append(dict(stage='final (float32)', bound_comp=b_comp, bound_mem=b_mem, bound=b_comp, bound_sum=b_comp + b_mem, measured=st['final_tile']))
    return out


def main():
    res = {}
    for lab in ('tpu-v6e', 'tpu-v5e'):
        ub = load(os.path.join(PROF, f'ubench_{lab}.json'))
        prof = load(os.path.join(PROF, f'profile8_fp32_fast_{lab}.json'))
        if ub and prof:
            res[lab] = tpu_bounds(lab, ub, prof)
    ub = load(os.path.join(PROF, 'ubench_gpu-l4.json'))
    prof = load(os.path.join(PROF, 'profile_cuda6_gpu-l4.json'))
    if ub and prof:
        res['gpu-l4'] = gpu_bounds(ub, prof)
    for lab, rows in res.items():
        print(lab)
        for r in rows:
            parts = {k: round(v, 3) for k, v in r.items() if k.startswith('bound') and isinstance(v, float)}
            print(f"  {r['stage']:40s} measured {r['measured']:.3f} s  bound {r['bound']:.3f} s  ratio {r['measured'] / r['bound']:.2f}  {parts}")
    json.dump(res, open(os.path.join(R, 'bounds.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
