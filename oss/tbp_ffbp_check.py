import os, sys, numpy as np, torch
os.environ['DEV'] = 'cpu'; os.environ['QUICK'] = '1'
sys.argv = ['x', sys.argv[1], '/dev/null']
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torchbp_panama.py')).read().split("for os_ in")[0]
exec(src)
import torchbp
data, nfft, base = rc_gpu(2)
dr = C / (2 * df * nfft); d0 = -Rbar + (nfft // 2 + base) * dr
(i0, j0), = crops.values()
g = grid(i0, j0, h, h)
c = posf.mean(0); org = np.array([c[0], c[1], 0.0])
pos_p = torch.tensor(posf - org, dtype=torch.float32)
xc, yc = 0.5 * (g['x'][0] + g['x'][1]) - org[0], 0.5 * (g['y'][0] + g['y'][1]) - org[1]
gr = float(np.hypot(xc, yc)); ext = 0.75 * h * max(spx, spy); ang = float(np.arctan2(yc, xc)); w = ext / gr
nr = int(2 * ext / 0.25); nth = int(2 * w * gr / 0.25)
gcart = {"x": (g["x"][0] - org[0], g["x"][1] - org[0]), "y": (g["y"][0] - org[1], g["y"][1] - org[1]), "nx": h, "ny": h}
ref = np.asarray(ref_img[i0:i0 + h, j0:j0 + h])
print('polar origin distance to region (m)', gr, 'angle', ang, 'platform height above origin plane', posf[:, 2].mean())
for name, th, rot in (('offset theta, no rotation', (np.sin(ang - w), np.sin(ang + w)), 0.0), ('symmetric theta, rotation', (np.sin(-w), np.sin(w)), ang)):
    gpol = {"r": (gr - ext, gr + ext), "theta": th, "nr": nr, "ntheta": nth}
    ip = torchbp.ops.ffbp(data, gpol, fref, dr, pos_p, stages=6, d0=d0, dealias=True, grid_oversample=2.0)
    ex = torchbp.ops.backprojection_polar_2d(data, gpol, fref, dr, pos_p, d0=d0, dealias=True)
    for lab, im in (('ffbp', ip), ('exact polar', ex)):
        img = torchbp.ops.polar_to_cart(im.reshape(1, *im.shape[-2:]), torch.tensor([[0.0, 0.0, float(posf[:, 2].mean())]]), gpol, gcart, fref, rot, method=("lanczos", 6))
        a = img.numpy().reshape(h, h).T
        print(name, lab, 'nan in polar', int(torch.isnan(im).sum()), 'nan in cart', int(np.isnan(a).sum()), 'error', round(err(np.nan_to_num(a), ref), 2), flush=True)
