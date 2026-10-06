import sys, json
src = open(sys.argv[1]).read()
out = {}
for r in (1000.0, 4000.0, 16000.0, 64000.0, 256000.0):
    code = src.replace("n, r0_, dev, out = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]", f"n, r0_, dev, out = 256, {r}, 'cpu', '/dev/null'")
    code = code.split("if 'torch_exact' in ONLY:")[0]
    g = {}
    exec(code, g)
    fs = g['fastsar']; sc = g['score']
    gg = dict(S=g['Sw'], ant=g['col'].ant, fmin=g['col'].fmin, df=g['col'].df, nx=256, ny=256, spx=0.5, spy=0.5, e1=g['e1'], e2=g['e2'])
    tb = g['torchbp'].ops.backprojection_cart_2d(g['data_t'], g['grid'], g['fref'], g['dr_tb'], g['pos_t'], d0=g['d0']).numpy()
    f = fs.ImageFormer(gg['ant'], gg['fmin'], gg['df'], gg['S'].shape[1], 256, 256, 0.5, 0.5, gg['e1'], gg['e2'], backend='cpu', window=False)
    out[r] = dict(fastsar_T32=sc(fs.form_image(**gg, backend='cpu', window=False, T=32)), fastsar_auto=sc(f(gg['S'])), fastsar_auto_T=f.T,
                  torchbp_exact=sc(tb.reshape(256, 256).T))
    print(r, {k: round(v, 1) for k, v in out[r].items()}, flush=True)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
