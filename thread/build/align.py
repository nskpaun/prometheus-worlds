"""Making a repainted picture sit on the picture it was made from. numpy/scipy only.
register(): the answer moved, block by block, onto the given picture (the picture model slides things by a few pixels;
            paint is laid on a mesh by the given picture's geometry, so every slide is paint off its surface)."""
import numpy as np
from scipy import ndimage
GREY = np.float32([0.299, 0.587, 0.114])
def _feat(a, s):
    g = ndimage.gaussian_filter(a @ GREY, s); return np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))
def flow(given, ans, block=160, reach=40, s=2.0):
    """Per block (half-overlapping) the shift (dy, dx) that lays ans on given, and how sure it is. -> (ys, xs, d[.., 2], conf)"""
    A, Bf = _feat(given, s), _feat(ans, s); h, w = A.shape; step = block // 2
    ys = np.arange(0, max(h - block, 0) + 1, step); xs = np.arange(0, max(w - block, 0) + 1, step)
    win = np.outer(np.hanning(block), np.hanning(block)); d = np.zeros((len(ys), len(xs), 2)); conf = np.zeros((len(ys), len(xs)))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            a = A[y:y + block, x:x + block]; b = Bf[y:y + block, x:x + block]; a = (a - a.mean()) * win; b = (b - b.mean()) * win
            na, nb = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
            if na < 1e-6 or nb < 1e-6: continue
            c = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape) / (na * nb)      # peak at the shift of b that gives a
            c = np.fft.fftshift(c); m = block // 2; sub = c[m - reach:m + reach + 1, m - reach:m + reach + 1]
            py, px = np.unravel_index(int(sub.argmax()), sub.shape); pk = sub[py, px]
            def fine(v, k, n):      # parabola through the peak and its two neighbours
                if k <= 0 or k >= n - 1: return 0.0
                l, c0, r = v[k - 1], v[k], v[k + 1]; den = l - 2 * c0 + r
                return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (l - r) / den, -0.5, 0.5))
            d[i, j] = (py - reach + fine(sub[:, px], py, sub.shape[0]), px - reach + fine(sub[py], px, sub.shape[1]))
            conf[i, j] = max(0.0, pk) * (0 < py < 2 * reach and 0 < px < 2 * reach)
    return ys + block / 2, xs + block / 2, d, conf
def warp(img, tot):
    h, w = tot.shape[:2]; yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return np.stack([ndimage.map_coordinates(img[..., ch], [yy - tot[..., 0], xx - tot[..., 1]], order=1, mode="nearest") for ch in range(img.shape[2])], -1)
def register(given, ans, rounds=2, want_flow=False):
    """ans (float HxWx3, same size as given) warped onto given. -> (warped, median shift px, 95th percentile shift px)"""
    h, w = given.shape[:2]; out = ans; tot = np.zeros((h, w, 2), np.float32); stats = None
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    for r in range(rounds):
        ys, xs, d, conf = flow(given, out, block=(192, 128)[min(r, 1)])
        wgt = conf ** 2 + 1e-4; f = []
        for ch in range(2):      # a smooth field: sure blocks carry the unsure ones
            num = ndimage.gaussian_filter(d[..., ch] * wgt, 1.0, mode="nearest"); den = ndimage.gaussian_filter(wgt, 1.0, mode="nearest")
            g = num / den
            f.append(ndimage.map_coordinates(g, [np.clip((yy - ys[0]) / (ys[1] - ys[0]), 0, len(ys) - 1), np.clip((xx - xs[0]) / (xs[1] - xs[0]), 0, len(xs) - 1)], order=1, mode="nearest"))
        if stats is None: m = np.hypot(d[..., 0], d[..., 1])[conf > 0.3]; stats = (float(np.median(m)) if m.size else 0.0, float(np.percentile(m, 95)) if m.size else 0.0)
        tot[..., 0] += f[0]; tot[..., 1] += f[1]
        # given(y, x) ~ ans(y - dy, x - dx): c peaks where b shifted BY d matches a
        out = np.stack([ndimage.map_coordinates(ans[..., ch], [yy - tot[..., 0], xx - tot[..., 1]], order=1, mode="nearest") for ch in range(3)], -1)
    return (out, stats[0], stats[1], tot) if want_flow else (out, stats[0], stats[1])
