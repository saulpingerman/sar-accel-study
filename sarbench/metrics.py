"""Image-quality and change-detection metrics (NumPy, host side)."""
import numpy as np
from scipy.ndimage import uniform_filter, binary_dilation, binary_erosion


def coherence(f, g, win=7):
    """Sample coherence magnitude over a boxcar; `win` is a size or (rows, cols)."""
    x = f * np.conj(g)
    num = uniform_filter(x.real, win) + 1j * uniform_filter(x.imag, win)
    den = np.sqrt(uniform_filter(np.abs(f) ** 2, win) * uniform_filter(np.abs(g) ** 2, win))
    return np.abs(num) / np.maximum(den, 1e-300)


def error_db(test, ref, fit_gain=False):
    """Energy of (test - ref) relative to ref, dB. With `fit_gain` a single
    complex scale is removed first (for comparing differently normalised methods)."""
    if fit_gain:
        test = test * (np.vdot(test, ref) / np.vdot(test, test))
    return 10 * np.log10(np.sum(np.abs(test - ref) ** 2) / np.sum(np.abs(ref) ** 2) + 1e-300)


def global_coherence(test, ref):
    return np.abs(np.vdot(ref, test)) / np.sqrt(np.vdot(ref, ref).real * np.vdot(test, test).real + 1e-300)


def phase_rms_deg(test, ref):
    """Power-weighted RMS phase difference, degrees."""
    x = test * np.conj(ref)
    w = np.abs(ref) ** 2
    return np.rad2deg(np.sqrt(np.sum(w * np.angle(x) ** 2) / np.sum(w)))


def auc(score, truth):
    """Area under the ROC curve; `score` high means detected."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = truth.sum()
    n0 = truth.size - n1
    return (r[truth].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def ccd_report(img1, img2, changed, win=7, thresh=0.5, thin=None):
    """Coherent change detection between two co-registered complex images.

    `changed` is the pixel truth mask. Pixels within one window of a mask edge
    are left out of both classes, which removes any feature narrower than the
    window. `thin` is the mask of such a feature; it is scored on its own,
    without erosion, against the same unchanged class.
    """
    gam = coherence(img1, img2, win)
    k = np.ones(win if isinstance(win, tuple) else (win, win), bool)
    ch = binary_erosion(changed, k)
    un = ~binary_dilation(changed, k)
    use = ch | un
    extra = {}
    if thin is not None:
        u2 = thin | un
        extra = dict(auc_thin=float(auc(1.0 - gam[u2], thin[u2])), coh_thin=float(gam[thin].mean()),
                     pd_thin_at_thresh=float((gam[thin] < thresh).mean()))
    return gam, dict(
        **extra,
        coh_unchanged=float(gam[un].mean()),
        coh_changed=float(gam[ch].mean()),
        auc=float(auc(1.0 - gam[use], ch[use])),
        pd_at_thresh=float((gam[ch] < thresh).mean()),
        pfa_at_thresh=float((gam[un] < thresh).mean()))


def _upsample(chip, up):
    """Band-limited interpolation of a complex chip by zero-padding its spectrum.

    A ground-plane image carries the radar carrier, so its spectrum sits at an
    arbitrary offset and can straddle the edge of the FFT. Each axis is
    rotated to put the energy centroid at zero frequency before padding; that
    only applies a linear phase and leaves magnitudes unchanged.
    """
    n0, n1 = chip.shape
    F = np.fft.fft2(chip)
    e = np.abs(F) ** 2
    for ax, n in ((0, n0), (1, n1)):
        prof = e.sum(1 - ax)
        k = np.angle(np.sum(prof * np.exp(2j * np.pi * np.arange(n) / n))) * n / (2 * np.pi)
        F = np.roll(F, -int(np.rint(k)), axis=ax)
    F = np.fft.fftshift(F)
    P = np.zeros((n0 * up, n1 * up), complex)
    a, b = (n0 * up - n0) // 2, (n1 * up - n1) // 2
    P[a:a + n0, b:b + n1] = F
    return np.fft.ifft2(np.fft.ifftshift(P)) * up * up


def _cut_metrics(c, spacing):
    """-3 dB width, PSLR and ISLR of a 1-D power cut with its peak inside."""
    p = np.abs(c) ** 2
    k = int(np.argmax(p))
    pk = p[k]
    half = pk / 2.0
    i = k
    while i > 0 and p[i] > half:
        i -= 1
    left = i + (half - p[i]) / (p[i + 1] - p[i])
    j = k
    while j < len(p) - 1 and p[j] > half:
        j += 1
    right = j - (half - p[j]) / (p[j - 1] - p[j])
    lo = k
    while lo > 0 and p[lo - 1] < p[lo]:
        lo -= 1
    hi = k
    while hi < len(p) - 1 and p[hi + 1] < p[hi]:
        hi += 1
    side = np.concatenate([p[:lo], p[hi + 1:]])
    return dict(irw=float((right - left) * spacing),
                pslr_db=float(10 * np.log10(side.max() / pk)),
                islr_db=float(10 * np.log10(side.sum() / p[lo:hi + 1].sum())))


def irf_report(img, rc, spacing, half=24, up=16):
    """Impulse-response metrics for the target nearest pixel `rc` = (row, col).

    `spacing` = (axis-0, axis-1) pixel spacing in metres. Returns per-axis
    resolution, PSLR and ISLR plus the complex peak and its sub-pixel position.
    """
    r, c = int(round(rc[0])), int(round(rc[1]))
    chip = img[r - half:r + half, c - half:c + half]
    u = _upsample(chip, up)
    k0, k1 = np.unravel_index(np.argmax(np.abs(u)), u.shape)
    out = dict(peak=complex(u[k0, k1]),
               pos=(r - half + k0 / up, c - half + k1 / up))
    out['ax0'] = _cut_metrics(u[:, k1], spacing[0] / up)
    out['ax1'] = _cut_metrics(u[k0, :], spacing[1] / up)
    return out


def floor_db(img, targets_rc, guard=6):
    """Median power away from every target's row and column, relative to the
    brightest pixel, dB. Off-axis regions hold no sidelobe energy from a
    separable taper, so a raised floor exposes arithmetic noise."""
    p = np.abs(img) ** 2
    keep = np.ones(img.shape, bool)
    for r, c in targets_rc:
        r, c = int(round(r)), int(round(c))
        keep[max(r - guard, 0):r + guard + 1, :] = False
        keep[:, max(c - guard, 0):c + guard + 1] = False
    return float(10 * np.log10(np.median(p[keep]) / p.max() + 1e-300))
