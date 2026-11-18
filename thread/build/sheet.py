"""The view sheet of one sculpted kind: several straight-on views of its mesh, all at ONE scale (pixels a metre), packed
into one picture that the picture model repaints in one call, so every side is painted together and agrees. thread/.venv, CPU."""
import math, numpy as np
from scipy import ndimage
import geom

def cand_views(attach, small):
    """Directions a walker can see a thing from: (name, az, el, words)."""
    ring = lambda el, tag, words: [(f"{tag}{az:03d}", az, el, words.format(side=s)) for az, s in
                                   ((0, "front"), (90, "left side"), (180, "back"), (270, "right side"), (45, "front left"), (135, "back left"), (225, "back right"), (315, "front right"))]
    if attach == "wall": return [("front", 0, 5, "front"), ("fl", 50, 12, "front left"), ("fr", -50, 12, "front right"), ("up", 0, 60, "from above"), ("dn", 0, -45, "from below")]
    if attach == "ceiling": return ring(-12, "s", "{side}, from slightly below") + [("below", 0, -75, "from below")] + ring(-45, "b", "{side}, from below")
    return ring(8, "s", "{side}") + [c for c in ring(42, "h", "{side}, seen from above") if c[1] % 90 == 0]

def choose_views(V, sc, P, N, cands, first, n, area_w=None, min_gain=0.03):
    """Greedy cover: starting from the views in `first`, add the candidate that adds most well-seen surface, until n views or the gain is small.
    -> list of (name, View(256 px), words), coverage after each"""
    best = np.zeros(len(P), np.float32); out, cov = [], []
    def seen(v):
        vis, cosv = v.visible(sc, P, N); return np.where(vis, np.clip(cosv, 0, 1), 0).astype(np.float32)
    score = lambda b: float(np.sqrt(b).mean())                # sqrt: a side seen at a slant still counts, seen head-on counts most
    for nm, v, words in first:
        best = np.maximum(best, seen(v)); out.append((nm, v, words)); cov.append(score(best))
    S = {c[0]: seen(geom.View(geom.framed_half(V, c[1], c[2]), c[1], c[2], 64)) for c in cands}
    while len(out) < n:
        g = {k: score(np.maximum(best, s)) - score(best) for k, s in S.items()}; k = max(g, key=g.get)
        if g[k] < min_gain: break
        c = next(c for c in cands if c[0] == k); out.append((k, geom.View(geom.framed_half(V, c[1], c[2]), c[1], c[2], 64), c[3])); best = np.maximum(best, S.pop(k)); cov.append(score(best))
    return out, cov

def make_tiles(sc, views, half, ppm, pad):
    """Each view rendered at ppm pixels a metre (a square view 2*half wide), cut to its object's box + pad. -> tiles: dict(name, words, view, R, box (x0, y0, x1, y1) in the view's pixels)"""
    px = int(math.ceil(2 * half * ppm / 2) * 2); tiles = []
    for nm, v, words in views:
        vw = geom.View(half, v.az, v.el, px, v.k, v.flip); R = vw.render(sc)
        if not R["hit"].any(): continue
        x0, y0, x1, y1 = geom.box_of(R["hit"]); x0, y0, x1, y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, px), min(y1 + pad, px)
        tiles.append(dict(name=nm, words=words, view=vw, R=R, box=(int(x0), int(y0), int(x1), int(y1))))
    return tiles

def pack(sizes, gap, lo=0.6, hi=1.7):
    """Shelf packing of (w, h) boxes into the smallest rectangle whose shape is between lo and hi wide-over-tall. -> positions [(x, y)], W, H"""
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1]); best = None; wmax = max(s[0] for s in sizes)
    for Wt in np.unique(np.concatenate([[wmax], np.linspace(wmax, sum(s[0] for s in sizes) + gap * len(sizes), 40)])):
        pos = [None] * len(sizes); x = y = rowh = 0; W = 0; roww = {}
        for i in order:
            w, h = sizes[i]
            if x > 0 and x + w > Wt: x = 0; y += rowh + gap; rowh = 0
            pos[i] = (x, y); x += w + gap; rowh = max(rowh, h); W = max(W, x - gap); roww[y] = x - gap
        H = y + rowh; a = W / H; pen = W * H * (1 + 3 * max(0, lo - a, a - hi)) * (1 + 2 * (1 - min(roww.values()) / W))          # full rows: Klein keeps a grid, and splits or moves a lone small picture
        if best is None or pen < best[0]: best = (pen, pos, W, H)
    return best[1], int(best[2]), int(best[3])

def layout(tiles, gap, mult=16, lo=0.6, hi=1.7):
    """Tiles placed on a sheet whose sides are multiples of `mult` (rows centred). Sets tile['at'] = (x, y) of the tile's box corner on the sheet. -> W, H"""
    sizes = [(t["box"][2] - t["box"][0], t["box"][3] - t["box"][1]) for t in tiles]; pos, W, H = pack(sizes, gap, lo, hi)
    W2 = int(math.ceil((W + 2 * gap) / mult) * mult); H2 = int(math.ceil((H + 2 * gap) / mult) * mult); ox, oy = (W2 - W) // 2, (H2 - H) // 2
    rows = {}
    for t, p, s in zip(tiles, pos, sizes): rows.setdefault(p[1], []).append((t, p, s))
    for y, r in rows.items():
        rw = max(p[0] + s[0] for _, p, s in r); rh = max(s[1] for _, _, s in r); sx = (W - rw) // 2
        for t, p, s in r: t["at"] = (int(ox + p[0] + sx), int(oy + y + (rh - s[1]) // 2))
    return W2, H2

def compose(tiles, W, H, back, imgs):
    """imgs[i]: the i-th tile's whole view picture (float RGB, view px). -> the sheet"""
    S = np.empty((H, W, 3), np.float32); S[:] = np.asarray(back, np.float32)
    for t, im in zip(tiles, imgs):
        x0, y0, x1, y1 = t["box"]; ax, ay = t["at"]; S[ay:ay + y1 - y0, ax:ax + x1 - x0] = im[y0:y1, x0:x1]
    return S

def cut(tile, sheet, k=1.0, fill=0.5):
    """The tile's part of a sheet (the sheet is k times the layout's size) put back into its view's own frame at k times the view's px. -> picture, valid mask"""
    x0, y0, x1, y1 = tile["box"]; ax, ay = tile["at"]; px = int(round(tile["view"].px * k)); out = np.full((px, px, 3), fill, np.float32); ok = np.zeros((px, px), bool)
    X0, Y0, X1, Y1 = (int(round(v * k)) for v in (x0, y0, x1, y1)); AX, AY = int(round(ax * k)), int(round(ay * k)); w = min(X1 - X0, sheet.shape[1] - AX, px - X0); h = min(Y1 - Y0, sheet.shape[0] - AY, px - Y0)
    out[Y0:Y0 + h, X0:X0 + w] = sheet[AY:AY + h, AX:AX + w]; ok[Y0:Y0 + h, X0:X0 + w] = True
    return out, ok

def fit_scale(sc, views, half, pixels, probe=192):
    """Pixels a metre at which the packed sheet has about `pixels` pixels (boxes measured on small renders, then scaled)."""
    p0 = probe / (2 * half); t0 = make_tiles(sc, views, half, p0, pad=0); best = p0
    for ppm in np.geomspace(p0 * 40, p0 * 0.3, 80):
        k = ppm / p0; pad = max(4, int(0.02 * 2 * half * ppm)); gap = max(8, int(0.03 * 2 * half * ppm))
        sizes = [((t["box"][2] - t["box"][0]) * k + 2 * pad + 2, (t["box"][3] - t["box"][1]) * k + 2 * pad + 2) for t in t0]; pos, W, H = pack(sizes, gap)
        if (math.ceil((W + 2 * gap) / 16) * 16) * (math.ceil((H + 2 * gap) / 16) * 16) <= pixels: return float(ppm)
    return float(ppm)
