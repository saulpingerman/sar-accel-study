"""torchbp on the Umbra Panama collection: exact backprojection of the three 512 by 512 pixel regions against the
float64 reference images of the study, the full image timed end to end on the GPU, and torchbp's factorized
backprojection attempted on the full image.   python -I torchbp_panama.py <work dir> <out json>"""
import os, sys, time, json
import numpy as np, torch, torchbp
from scipy.signal.windows import taylor
W, out = sys.argv[1], sys.argv[2]
C = 299792458.0
d = np.load(f'{W}/panama.npz')
S0 = d['S']; ant = d['ant'].astype(np.float64)
fmin, df = float(d['fmin']), float(d['df'])
nx, ny, spx, spy = int(d['nx']), int(d['ny']), float(d['spx']), float(d['spy'])
e1, e2 = d['e1'].astype(np.float64), d['e2'].astype(np.float64)
P, K = S0.shape
wp = taylor(P, nbar=4, sll=35.0, norm=False).astype(np.float32); wk = taylor(K, nbar=4, sll=35.0, norm=False).astype(np.float32)
ref_img = np.load(f'{W}/ref.npy', mmap_mode='r')
crops = dict(locks=(3400, 6700), port=(9500, 1150), ships=(6400, 2650)); h = 512
if os.environ.get('QUICK') == '1':
    crops = dict(locks=(3400 + 192, 6700 + 192)); h = 128
dev = os.environ.get('DEV', 'cuda')
QUICK = os.environ.get('QUICK') == '1'
sync = torch.cuda.synchronize if dev == 'cuda' else (lambda: None)
r0 = np.linalg.norm(ant, axis=1); Rbar = float(r0.mean())
fref = fmin + (K // 2) * df
zup = np.cross(e2, e1)
posf = np.stack([ant @ e2, ant @ e1, ant @ zup], 1)
pos_t = torch.tensor(posf, dtype=torch.float32, device=dev)
f_t = torch.tensor(fmin + np.arange(K) * df, dtype=torch.float64, device=dev)
res = dict(pulses=P, samples=K, pixels=[nx, ny], gpu=torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu', methods={})


def rc_gpu(os_):
    """Taylor-windowed phase history range compressed on the GPU onto one absolute delay axis (torchbp takes a
    single offset d0 for all pulses): each pulse is referenced to the common range Rbar, its profile circularly
    re-centered by the integer part m_p of (|a_p| - Rbar)/dr (the carrier phase kept) and placed at offset m_p in
    a buffer that spans the 12 km over which the range to the scene center changes. -> (buffer, nfft, base)."""
    nfft = 1 << int(np.ceil(np.log2(os_ * K)))
    dr = C / (2 * df * nfft)
    m = np.round((r0 - Rbar) / dr).astype(np.int64)
    base = -int(m.min())
    width = nfft + int(m.max() - m.min()) + 1
    out_ = torch.zeros((P, width), dtype=torch.complex64, device=dev)
    hh = K // 2
    q = torch.arange(K, dtype=torch.float64, device=dev) - hh
    for p0 in range(0, P, 256):
        sl = slice(p0, min(P, p0 + 256))
        blk = torch.tensor(S0[sl] * wp[sl, None] * wk[None, :], device=dev).to(torch.complex128)
        blk = blk * torch.exp(-1j * 4 * np.pi * f_t[None, :] * torch.tensor(r0[sl] - Rbar, device=dev)[:, None] / C)
        blk = blk * torch.exp(2j * np.pi * q[None, :] * torch.tensor(m[sl], dtype=torch.float64, device=dev)[:, None] / nfft)
        pad = torch.zeros((blk.shape[0], nfft), dtype=torch.complex128, device=dev)
        pad[:, :K - hh] = blk[:, hh:]; pad[:, nfft - hh:] = blk[:, :hh]
        rc = torch.fft.fftshift(torch.fft.ifft(pad, dim=1), dim=1).to(torch.complex64)
        for k in range(rc.shape[0]):
            o = base + int(m[p0 + k])
            out_[p0 + k, o:o + nfft] = rc[k]
    return out_, nfft, base


def grid(i0, j0, mi, mj):
    """torchbp's cartesian grid (x along e2 = range, y along e1 = azimuth) for image rows i0.. and columns j0.."""
    return {"x": ((j0 - ny / 2.0) * spy, (j0 + mj - ny / 2.0) * spy), "y": ((i0 - nx / 2.0) * spx, (i0 + mi - nx / 2.0) * spx), "nx": mj, "ny": mi}


def err(a, b):
    a, b = a.astype(np.complex128).ravel(), b.astype(np.complex128).ravel()
    g = np.vdot(a, b) / np.vdot(a, a)
    return float(10 * np.log10(np.sum(np.abs(g * a - b) ** 2) / np.sum(np.abs(b) ** 2)))


for os_ in (2,):
    sync(); t = time.perf_counter()
    data, nfft, base = rc_gpu(os_)
    sync(); t_rc = time.perf_counter() - t
    dr = C / (2 * df * nfft); d0 = -Rbar + (nfft // 2 + base) * dr
    e = {}
    for n_, (i0, j0) in crops.items():
        img = torchbp.ops.backprojection_cart_2d(data, grid(i0, j0, h, h), fref, dr, pos_t, d0=d0)
        img = img.cpu().numpy().reshape(h, h).T
        e[n_] = err(img, np.asarray(ref_img[i0:i0 + h, j0:j0 + h]))
    key = f'torchbp exact, {os_}x profiles'
    res['methods'][key] = dict(errors=e, range_compression_seconds=t_rc)
    print(key, e, flush=True)
    if os_ == 2 and not QUICK and os.environ.get('REGIONS_FFBP') != '1':
        # the full image, end to end on the GPU (range compression included), timed after a warm-up
        g = grid(0, 0, nx, ny)
        try:
            torchbp.ops.backprojection_cart_2d(data, grid(0, 0, 256, 256), fref, dr, pos_t, d0=d0); sync()
            del data; (torch.cuda.empty_cache() if dev == 'cuda' else None)
            t = time.perf_counter()
            data, _, _ = rc_gpu(os_)
            full = torchbp.ops.backprojection_cart_2d(data, g, fref, dr, pos_t, d0=d0)
            sync(); res['methods'][key]['full_image_seconds'] = time.perf_counter() - t
            del full; (torch.cuda.empty_cache() if dev == 'cuda' else None)
        except Exception as ex:
            res['methods'][key]['full_image_error'] = f'{type(ex).__name__}: {str(ex)[:300]}'
        print(res['methods'][key], flush=True)
        # torchbp's factorized backprojection on the full image: polar grid about the ground point below the aperture
        # center, then polar_to_cart onto the image grid (the recipe validated on simulated scenes)
        try:
            c = posf.mean(0); org = np.array([c[0], c[1], 0.0])
            pos_p = torch.tensor(posf - org, dtype=torch.float32, device=dev)
            cx, cy = -org[0], -org[1]
            gr = float(np.hypot(cx, cy)); ext = 0.75 * max(nx * spx, ny * spy)
            ang = float(np.arctan2(cy, cx)); w = ext / gr
            res_m = 0.5
            nr = int(2 * ext / (res_m / 2)); nth = int(2 * w * gr / (res_m / 2))
            gpol = {"r": (gr - ext, gr + ext), "theta": (np.sin(-w), np.sin(w)), "nr": nr, "ntheta": nth}
            gcart = {"x": (g["x"][0] - org[0], g["x"][1] - org[0]), "y": (g["y"][0] - org[1], g["y"][1] - org[1]), "nx": ny, "ny": nx}
            (torch.cuda.empty_cache() if dev == 'cuda' else None); sync(); t = time.perf_counter()
            ip = torchbp.ops.ffbp(data, gpol, fref, dr, pos_p, stages=8, d0=d0, dealias=True, grid_oversample=2.0)
            img = torchbp.ops.polar_to_cart(ip.reshape(1, *ip.shape[-2:]), torch.tensor([[0.0, 0.0, float(posf[:, 2].mean())]], device=dev), gpol, gcart, fref, ang, method=("lanczos", 6))
            sync(); tf = time.perf_counter() - t
            img = img.cpu().numpy().reshape(ny, nx).T
            res['methods']['torchbp factorized, 2x profiles'] = dict(seconds_after_range_compression=tf, polar_grid=[nr, nth],
                errors={n_: err(img[i0:i0 + h, j0:j0 + h], np.asarray(ref_img[i0:i0 + h, j0:j0 + h])) for n_, (i0, j0) in crops.items()})
        except Exception as ex:
            res['methods']['torchbp factorized, 2x profiles'] = dict(error=f'{type(ex).__name__}: {str(ex)[:300]}')
        print(res['methods']['torchbp factorized, 2x profiles'], flush=True)
    if os.environ.get('REGIONS_FFBP') == '1':
        # torchbp's factorized backprojection on each region: a polar grid about the ground point below the aperture
        # center covering the region, then polar_to_cart onto the region's pixels
        c = posf.mean(0); org = np.array([c[0], c[1], 0.0])
        pos_p = torch.tensor(posf - org, dtype=torch.float32, device=dev)
        for n_, (i0, j0) in crops.items():
            try:
                g = grid(i0, j0, h, h)
                xc, yc = 0.5 * (g['x'][0] + g['x'][1]) - org[0], 0.5 * (g['y'][0] + g['y'][1]) - org[1]
                gr = float(np.hypot(xc, yc)); ext = 0.75 * h * max(spx, spy); ang = float(np.arctan2(yc, xc)); w = ext / gr
                nr = int(2 * ext / 0.25); nth = int(2 * w * gr / 0.25)
                gpol = {"r": (gr - ext, gr + ext), "theta": (np.sin(ang - w), np.sin(ang + w)), "nr": nr, "ntheta": nth}
                gcart = {"x": (g["x"][0] - org[0], g["x"][1] - org[0]), "y": (g["y"][0] - org[1], g["y"][1] - org[1]), "nx": h, "ny": h}
                sync(); t = time.perf_counter()
                ip = torchbp.ops.ffbp(data, gpol, fref, dr, pos_p, stages=6, d0=d0, dealias=True, grid_oversample=2.0)
                img = torchbp.ops.polar_to_cart(ip.reshape(1, *ip.shape[-2:]), torch.tensor([[0.0, 0.0, float(posf[:, 2].mean())]], device=dev), gpol, gcart, fref, 0.0, method=("lanczos", 6))
                sync(); tf = time.perf_counter() - t
                img = img.cpu().numpy().reshape(h, h).T
                res['methods'].setdefault('torchbp factorized, regions', {})[n_] = dict(error=err(img, np.asarray(ref_img[i0:i0 + h, j0:j0 + h])), seconds=tf)
            except Exception as ex:
                res['methods'].setdefault('torchbp factorized, regions', {})[n_] = dict(failed=f'{type(ex).__name__}: {str(ex)[:300]}')
            print(n_, res['methods']['torchbp factorized, regions'][n_], flush=True)
    del data; (torch.cuda.empty_cache() if dev == 'cuda' else None)
json.dump(res, open(out, 'w'), indent=1)
