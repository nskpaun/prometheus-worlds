"""Plain forms made without the shape model, in milliseconds: real geometry (separate books, frame bars, legs, folds), not boxes.
  make(form, w, h, d, seed) -> (V, F, part, names)   V in the thing's own frame: x across -w/2..w/2, y up 0..h, z back -d/2 .. front +d/2
part: one small integer per face (names[part] says what it is: "top", "leg", "book 3", "frame", "picture", "flame" ...), so
whoever paints can give each part its own material or colour. Every mesh fills its box exactly (its top is the box top, so
things placed "on" it rest on it).
Forms: rug frame board books crate cylinder table bottle jar bowl pot candle cloth bookcase drawers rings, and lamp (a
stand-in for a light that could not be sculpted)."""
import math, zlib, random
import numpy as np

def _box(lo, hi):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    V = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    F = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
    return np.array(V, float), np.array(F)

def _lathe(profile, n=24, sx=1.0, sz=1.0):
    """Revolve [(radius, y), ...] (bottom to top; radius 0 closes an end) about the y axis; sx, sz stretch it to an oval."""
    V, F = [], []; rings = []
    for r, y in profile:
        if r <= 1e-9: rings.append([len(V)]); V.append((0.0, y, 0.0))
        else:
            rings.append(list(range(len(V), len(V) + n)))
            V += [(r * math.cos(2 * math.pi * k / n) * sx, y, r * math.sin(2 * math.pi * k / n) * sz) for k in range(n)]
    for a, b in zip(rings, rings[1:]):
        for k in range(n):
            k2 = (k + 1) % n
            if len(a) == 1 and len(b) == 1: continue
            if len(a) == 1: F.append((a[0], b[k2], b[k]))
            elif len(b) == 1: F.append((a[k], a[k2], b[0]))
            else: F += [(a[k], a[k2], b[k2]), (a[k], b[k2], b[k])]
    return np.array(V, float), np.array(F)[:, ::-1]

class Mesh:
    def __init__(self): self.V, self.F, self.P, self.names = [], [], [], []
    def add(self, VF, name):
        V, F = VF
        if name not in self.names: self.names.append(name)
        n = sum(len(v) for v in self.V); self.V.append(V); self.F.append(F + n); self.P.append(np.full(len(F), self.names.index(name)))
    def done(self): return np.concatenate(self.V), np.concatenate(self.F), np.concatenate(self.P), self.names

def _fit(V, w, h, d):
    """Stretch a mesh so that it fills the box exactly."""
    lo, hi = V.min(0), V.max(0); s = np.array([w, h, d]) / np.maximum(hi - lo, 1e-9)
    return (V - lo) * s + [-w / 2, 0, -d / 2]

def rug(w, h, d, rnd):
    m = Mesh(); b = min(0.12, 0.08 * min(w, d))
    m.add(_box((-w / 2, 0, -d / 2), (w / 2, h * 0.6, d / 2)), "border"); m.add(_box((-w / 2 + b, 0, -d / 2 + b), (w / 2 - b, h, d / 2 - b)), "field")
    return m.done()

def frame(w, h, d, rnd):
    m = Mesh(); b = min(max(0.09 * min(w, h), 0.02), 0.09, 0.3 * min(w, h))
    for lo, hi in (((-w / 2, 0, -d / 2), (w / 2, b, d / 2)), ((-w / 2, h - b, -d / 2), (w / 2, h, d / 2)), ((-w / 2, b, -d / 2), (-w / 2 + b, h - b, d / 2)), ((w / 2 - b, b, -d / 2), (w / 2, h - b, d / 2))): m.add(_box(lo, hi), "frame")
    m.add(_box((-w / 2 + b, b, -d / 2), (w / 2 - b, h - b, -d / 2 + 0.45 * d)), "picture")
    return m.done()

def board(w, h, d, rnd):
    m = Mesh()
    if d <= 0.09 or h > 2.5 * d: m.add(_box((-w / 2, 0, -d / 2), (w / 2, h, d / 2)), "panel"); return m.done()      # a sign or a panel flat on the wall
    t = min(h, 0.035); m.add(_box((-w / 2, h - t, -d / 2), (w / 2, h, d / 2)), "top")
    if h - t > 0.03:      # brackets under a shelf
        g = min(h - t, 0.8 * d); bw = 0.025
        for x in (-w / 2 + 0.12 * w, w / 2 - 0.12 * w - bw):
            m.add(_box((x, h - t - g, -d / 2), (x + bw, h - t, -d / 2 + bw)), "bracket"); m.add(_box((x, h - t - bw, -d / 2), (x + bw, h - t, -d / 2 + 0.85 * d)), "bracket")
    return m.done()

def books(w, h, d, rnd, name="", **kw):
    m = Mesh()
    if any(k in name.lower() for k in ("scroll", "roll", "chart", "blueprint", "plan")):      # rolled sheets lying in a small heap
        r = max(0.012, min(0.035, 0.5 * h, 0.2 * d)); rows = max(1, int(h / (2 * r))); k = 0
        for row in range(rows):
            cnt = max(1, int(d / (2 * r)) - row)
            for j in range(cnt):
                L = w * rnd.uniform(0.75, 1.0); V, F = _lathe([(0, -L / 2), (r, -L / 2), (r, L / 2), (0, L / 2)], 10); zc = -cnt * r + r + 2 * r * j
                m.add((np.stack([V[:, 1] + rnd.uniform(-0.02, 0.02) * w, row * 1.73 * r + r + V[:, 0], zc + V[:, 2]], 1), F[:, ::-1]), f"scroll {k % 4}"); k += 1
        V, F, P, names = m.done(); return _fit(V, w, h, d), F, P, names
    if w >= 1.25 * h:      # a row, spines to the front; the tallest book is the box's height
        x = -w / 2; k = 0; tall = rnd.randrange(0, 3)
        while x < w / 2 - 0.012:
            t = min(rnd.uniform(0.022, 0.055) * max(1.0, h / 0.25), w / 2 - x)
            if w / 2 - x - t < 0.02: t = w / 2 - x
            bh = h if k == tall else h * rnd.uniform(0.72, 0.98); bd = d * rnd.uniform(0.78, 1.0)
            m.add(_box((x, 0, -d / 2), (x + t * 0.94, bh, -d / 2 + bd)), f"book {k % 8}"); x += t; k += 1
    else:                  # a stack, lying flat, each a little askew
        y = 0.0; k = 0
        while y < h - 1e-6:
            t = min(rnd.uniform(0.025, 0.06) * max(1.0, h / 0.2), h - y)
            if h - y - t < 0.02: t = h - y
            sw, sd = (1.0, 1.0) if k == 0 else (rnd.uniform(0.75, 0.97), rnd.uniform(0.75, 0.97)); a = rnd.uniform(-0.12, 0.12)
            V, F = _box((-w * sw / 2, y, -d * sd / 2), (w * sw / 2, y + t * 0.96, d * sd / 2))
            if k:      # turned a little, kept inside the box
                cs, sn = math.cos(a), math.sin(a); X, Z = V[:, 0] * cs - V[:, 2] * sn, V[:, 0] * sn + V[:, 2] * cs
                s = min(1.0, (w / 2) / np.abs(X).max(), (d / 2) / np.abs(Z).max()); V[:, 0], V[:, 2] = X * s, Z * s
            m.add((V, F), f"book {k % 8}"); y += t; k += 1
    return m.done()

def crate(w, h, d, rnd, name="", **kw):
    m = Mesh()
    if any(k in name.lower() for k in ("chest", "trunk", "coffer", "footlocker")) and max(w, d) <= 1.6 and h >= 0.25:      # a chest: a body, a lid that overhangs, iron bands over both, a lock plate, end handles
        lid = 0.26 * h; o = 0.012; bw = max(0.03, 0.05 * w); t = 0.006
        m.add(_box((-w / 2 + o, 0, -d / 2 + o), (w / 2 - o, h - lid, d / 2 - o)), "body"); m.add(_box((-w / 2, h - lid, -d / 2), (w / 2, h - 0.25 * lid, d / 2)), "lid"); m.add(_box((-w / 2 + 0.03, h - 0.25 * lid, -d / 2 + 0.03), (w / 2 - 0.03, h, d / 2 - 0.03)), "lid")
        for x in (-0.3 * w, 0.3 * w):
            m.add(_box((x - bw / 2, 0, -d / 2 + o - t), (x + bw / 2, h - lid, d / 2 - o + t)), "band"); m.add(_box((x - bw / 2, h - lid, -d / 2 - t), (x + bw / 2, h - 0.25 * lid + t, d / 2 + t)), "band")
        m.add(_box((-0.05 * w, h - lid - 0.09 * h, d / 2 - o), (0.05 * w, h - lid + 0.03 * h, d / 2 - o + 0.015)), "lock")
        for sx in (-1, 1): x0 = sx * (w / 2 - o) - (0.02 if sx < 0 else 0); m.add(_box((x0, 0.5 * h, -0.12 * d), (x0 + 0.02, 0.5 * h + 0.025, 0.12 * d)), "handle")
        return m.done()
    if max(w, d) > 1.6 and h < 0.7:      # a platform or a step: a slab with a lip and a plinth course
        lip = min(0.05, 0.25 * h)
        m.add(_box((-w / 2 + 0.04, 0, -d / 2 + 0.04), (w / 2 - 0.04, h - lip, d / 2 - 0.04)), "side"); m.add(_box((-w / 2, h - lip, -d / 2), (w / 2, h, d / 2)), "top")
        return m.done()
    e = min(max(0.07 * min(w, h, d), 0.012), 0.05); i = 0.4 * e
    m.add(_box((-w / 2 + i, i, -d / 2 + i), (w / 2 - i, h - i, d / 2 - i)), "panel")
    for sx in (-1, 1):
        for sz in (-1, 1):
            x0, z0 = (sx * w / 2 - (e if sx > 0 else 0)), (sz * d / 2 - (e if sz > 0 else 0)); m.add(_box((x0, 0, z0), (x0 + e, h, z0 + e)), "batten")
    for y0 in (0, h - e):
        for sz in (-1, 1): z0 = sz * d / 2 - (e if sz > 0 else 0); m.add(_box((-w / 2 + e, y0, z0), (w / 2 - e, y0 + e, z0 + e)), "batten")
        for sx in (-1, 1): x0 = sx * w / 2 - (e if sx > 0 else 0); m.add(_box((x0, y0, -d / 2 + e), (x0 + e, y0 + e, d / 2 - e)), "batten")
    return m.done()

def cylinder(w, h, d, rnd):
    m = Mesh(); r = 0.5
    if h > 3 * max(w, d):          # a pillar or a pipe: a shaft with a base and a cap
        b = min(0.08 * h, 0.5 * max(w, d))
        prof = [(0, 0), (r, 0), (r, b * 0.6), (r * 0.86, b), (r * 0.8, h - b), (r * 0.9, h - b * 0.5), (r, h - b * 0.5), (r, h), (0, h)]; m.add(_lathe(prof), "shaft")
    elif h < 0.5 * max(w, d):      # a round platform or a lid: a disc with a lip
        lip = min(0.05, 0.3 * h); m.add(_lathe([(0, 0), (r * 0.96, 0), (r * 0.96, h - lip), (r, h - lip), (r, h), (0, h)], 40), "top")
    else:                          # a barrel, a drum, a can: straight sides, a rim and two raised hoops
        k = 0.965; rib = 0.02 * h
        prof = [(0, 0), (r, 0), (r, rib), (r * k, rib)]
        for y in (h / 3, 2 * h / 3): prof += [(r * k, y - rib / 2), (r, y - rib / 2), (r, y + rib / 2), (r * k, y + rib / 2)]
        prof += [(r * k, h - rib), (r, h - rib), (r, h), (r * 0.93, h), (r * 0.93, h - rib * 0.8), (0, h - rib * 0.8)]
        m.add(_lathe(prof), "body")
    V, F, P, names = m.done(); V[:, 0] *= w; V[:, 2] *= d; return V, F, P, names

def table(w, h, d, rnd, name="", **kw):
    if "stool" in name.lower() and max(w, d) < 0.7 and h < 0.9:      # a round seat on three or four splayed turned legs with a ring stretcher
        m = Mesh(); t = min(0.045, 0.1 * h); n = 3 if rnd.random() < 0.5 else 4; R = 0.5 * min(w, d); ph = rnd.uniform(0, 6.28)
        m.add(_lathe([(0, h - t), (R * 0.94, h - t), (R, h - t * 0.6), (R, h - t * 0.2), (R * 0.96, h), (0, h)], 24), "top"); lr = max(0.014, 0.045 * R * 2)
        for k in range(n):
            a_ = ph + 2 * math.pi * k / n; top_ = np.array([0.55 * R * math.cos(a_), h - t, 0.55 * R * math.sin(a_)]); foot = np.array([(R - lr) * math.cos(a_), 0, (R - lr) * math.sin(a_)])
            V, F = _lathe([(0, 0), (lr * 0.7, 0), (lr, 0.5), (lr * 0.9, 1.0), (0, 1.0)], 8); P = foot + (top_ - foot) * V[:, 1:2] + np.stack([V[:, 0], 0 * V[:, 0], V[:, 2]], 1); m.add((P, F), "leg")
        V, F = _lathe([(R * 0.72 - lr * 0.5, 0.3 * h - lr * 0.5), (R * 0.72 + lr * 0.5, 0.3 * h - lr * 0.5), (R * 0.72 + lr * 0.5, 0.3 * h + lr * 0.5), (R * 0.72 - lr * 0.5, 0.3 * h + lr * 0.5), (R * 0.72 - lr * 0.5, 0.3 * h - lr * 0.5)], 24); m.add((V, F), "rail")
        return m.done()
    m = Mesh(); t = min(max(0.05 * h, 0.025), 0.06); L = min(max(0.06 * min(w, d), 0.03), 0.09); o = min(0.04, 0.1 * min(w, d))
    m.add(_box((-w / 2, h - t, -d / 2), (w / 2, h, d / 2)), "top")
    for sx in (-1, 1):
        for sz in (-1, 1):
            x0 = sx * (w / 2 - o) - (L if sx > 0 else 0); z0 = sz * (d / 2 - o) - (L if sz > 0 else 0); m.add(_box((x0, 0, z0), (x0 + L, h - t, z0 + L)), "leg")
    a = min(0.1, 0.18 * h); th = 0.02       # the rails under the top
    for sz in (-1, 1): z0 = sz * (d / 2 - o - L / 2) - th / 2; m.add(_box((-w / 2 + o + L, h - t - a, z0), (w / 2 - o - L, h - t, z0 + th)), "rail")
    for sx in (-1, 1): x0 = sx * (w / 2 - o - L / 2) - th / 2; m.add(_box((x0, h - t - a, -d / 2 + o + L), (x0 + th, h - t, d / 2 - o - L)), "rail")
    if w > 1.2 and h > 0.6: m.add(_box((-w / 2 + o + L, 0.18 * h, -d / 2 + o + L), (w / 2 - o - L, 0.18 * h + 0.025, d / 2 - o - L)), "shelf")     # a bench's lower shelf
    elif h > 0.35:           # a stool or small table: stretchers between the legs
        y = 0.3 * h
        for sz in (-1, 1): z0 = sz * (d / 2 - o - L / 2) - th / 2; m.add(_box((-w / 2 + o + L, y, z0), (w / 2 - o - L, y + th, z0 + th)), "rail")
    return m.done()

def _round(profile, names_at, w, h, d):
    """A lathe profile in a unit box (radius 0.5, height 1), its faces named by height bands [(y from, name)]."""
    V, F = _lathe(profile); V = _fit(V, w, h, d); yc = V[F].mean(1)[:, 1] / h
    names = [n for _, n in names_at]; P = np.zeros(len(F), int)
    for k, (y, _) in enumerate(names_at): P[yc >= y] = k
    return V, F, P, names

def bottle(w, h, d, rnd):
    s = rnd.uniform(0.5, 0.62); n = rnd.uniform(0.14, 0.2)
    return _round([(0, 0), (0.46, 0), (0.5, 0.03), (0.5, s), (0.4, s + 0.1), (n, s + 0.2), (n, 0.93), (n + 0.04, 0.94), (n + 0.04, 0.98), (n * 0.8, 1.0), (0, 1.0)], [(0, "glass"), (0.93, "stopper")], w, h, d)
def jar(w, h, d, rnd):
    return _round([(0, 0), (0.47, 0), (0.5, 0.04), (0.5, 0.72), (0.42, 0.82), (0.42, 0.86), (0.46, 0.86), (0.46, 0.98), (0.42, 1.0), (0, 1.0)], [(0, "body"), (0.86, "lid")], w, h, d)
def bowl(w, h, d, rnd):
    return _round([(0, 0), (0.22, 0), (0.24, 0.1), (0.36, 0.5), (0.47, 0.9), (0.5, 1.0), (0.46, 1.0), (0.4, 0.75), (0.25, 0.4), (0, 0.32)], [(0, "body")], w, h, d)
def pot(w, h, d, rnd):
    b = rnd.uniform(0.3, 0.45)
    return _round([(0, 0), (0.28, 0), (0.3, 0.03), (0.44, b - 0.12), (0.5, b), (0.46, b + 0.2), (0.34, 0.82), (0.33, 0.9), (0.4, 1.0), (0.34, 1.0), (0.28, 0.9), (0.28, 0.8), (0, 0.78)], [(0, "body")], w, h, d)
def candle(w, h, d, rnd):
    c = rnd.uniform(0.2, 0.28)
    return _round([(0, 0), (0.5, 0), (0.5, 0.03), (0.2, 0.06), (0.12, 0.12), (0.12, 0.3), (0.42, 0.34), (0.42, 0.37), (c, 0.37), (c, 0.86), (0.03, 0.87), (0.03, 0.9), (0.1, 0.93), (0.06, 0.97), (0, 1.0)], [(0, "holder"), (0.37, "wax"), (0.865, "wick"), (0.9, "flame")], w, h, d)

def cloth(w, h, d, rnd, gap=0.0, **kw):
    """A heavy curtain: a real sheet with thickness (two skins joined at the edges, so it is closed from every side), gathered at a
    pole into deep folds that open and wander towards the hem; `gap` (0..0.8 of the width) draws it apart as two gathered
    stacks either side of a window, with nothing but the pole between them. No slits: a curtain is one piece or two whole ones."""
    m = Mesh(); rod = min(0.035, 0.2 * d); small = h < 1.2 or w < 0.5
    V, F = _lathe([(0, -w / 2 - 0.06), (rod * 0.9, -w / 2 - 0.06), (rod * 0.9, -w / 2 - 0.02), (rod / 2, -w / 2 - 0.02), (rod / 2, w / 2 + 0.02), (rod * 0.9, w / 2 + 0.02), (rod * 0.9, w / 2 + 0.06), (0, w / 2 + 0.06)], 10)
    V = np.stack([V[:, 1] * (w / (w + 0.12)), h - rod + V[:, 0] * 0.999, V[:, 2]], 1); m.add((V, F[:, ::-1]), "rod")
    gap = min(max(gap, 0.0), 0.8) * w
    spans = [(-w / 2, -gap / 2), (gap / 2, w / 2)] if gap > 0.15 else [(-w / 2, w / 2)]
    th = 0.012 if not small else 0.006; top = h - 2 * rod; amp0 = max(0.01, d / 2 - th - 0.004)
    for si, (x0, x1) in enumerate(spans):
        L = x1 - x0; full = 1.0 + (1.6 if gap > 0.15 else 0.0)                 # cloth drawn aside is gathered tighter
        folds = max(2, round(L * full / (0.22 if not small else 0.12))); nu, nv = folds * 10 + 1, 22; ph = rnd.uniform(0, 6.28); ph2 = rnd.uniform(0, 6.28)
        u, v = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv))        # v = 0 at the pole
        wander = 0.35 * np.sin(2.3 * v + ph2) * v + 0.25 * np.sin(2 * math.pi * 1.5 * u + ph2) * v      # the folds drift and merge on the way down
        fold = np.sin(2 * math.pi * folds * u + ph + wander * 4)
        fold = np.sign(fold) * np.abs(fold) ** 0.75                              # rounder pleats, sharper valleys
        z = amp0 * (0.5 + 0.5 * v ** 0.7) * fold + amp0 * 0.25 * v * np.sin(2 * math.pi * (folds / 3.1) * u + ph2)
        z = np.clip(z, -amp0, amp0)
        hem = top * (1 - v) + 0.012 * v * np.sin(2 * math.pi * folds * u + ph)  # the hem rises and falls with the folds
        x = x0 + L * u
        front = np.stack([x, hem, z + th / 2], -1).reshape(-1, 3); back = np.stack([x, hem, z - th / 2], -1).reshape(-1, 3)
        idx = np.arange(nu * nv).reshape(nv, nu); a_, b_, c_, e_ = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
        Ff = np.concatenate([np.stack([a_, e_, c_], 1), np.stack([a_, c_, b_], 1)]); n = nu * nv
        rim = np.concatenate([idx[0, :], idx[:, -1][1:], idx[-1, ::-1][1:], idx[::-1, 0][1:]])       # the edge loop, closed by a strip
        r0, r1 = rim[:-1], rim[1:]; Fe = np.concatenate([np.stack([r0, r1, r1 + n], 1), np.stack([r0, r1 + n, r0 + n], 1)])
        m.add((np.concatenate([front, back]), np.concatenate([Ff, Ff[:, ::-1] + n, Fe])), "cloth")
        if not small:                                                           # rings on the pole, one to a pleat
            for k in range(folds):
                rx = x0 + L * (k + 0.5) / folds
                Vr, Fr = _lathe([(rod * 0.75, -0.006), (rod * 1.25, -0.006), (rod * 1.25, 0.006), (rod * 0.75, 0.006), (rod * 0.75, -0.006)], 10)
                m.add((np.stack([rx + Vr[:, 1], h - rod + Vr[:, 0], Vr[:, 2]], 1), Fr[:, ::-1]), "ring")
    V, F, P, names = m.done()
    lo, hi = V.min(0), V.max(0); V = (V - [0, lo[1], 0]) * [w / (hi[0] - lo[0]), h / (hi[1] - lo[1]), 1.0]       # fills its box in width and height; depth stays real
    return V, F, P, names

def bookcase(w, h, d, rnd, name="", **kw):
    """A bookcase or storage rack, made complete: sides, back, top, plinth, shelves a real spacing apart, and on EVERY shelf
    rows of separate books (some leaning, some lying in a stack, gaps), or boxes, tins and jars for a storage rack."""
    m = Mesh(); goods = not any(k in name.lower() for k in ("book", "librar", "tome", "ledger", "archive", "file")) and any(k in name.lower() for k in ("rack", "storage", "shelv", "stock", "pantry", "larder", "tool", "tyre", "tire", "part", "goods", "store"))
    t = min(0.035, 0.12 * w); pl = min(0.09, 0.06 * h) if not goods else 0.0; open_back = goods
    if goods:      # a steel rack: four posts and thin shelves
        for sx in (-1, 1):
            for sz in (-1, 1): x0 = sx * w / 2 - (t if sx > 0 else 0); z0 = sz * d / 2 - (t if sz > 0 else 0); m.add(_box((x0, 0, z0), (x0 + t, h, z0 + t)), "post")
    else:
        m.add(_box((-w / 2, 0, -d / 2), (-w / 2 + t, h, d / 2)), "side"); m.add(_box((w / 2 - t, 0, -d / 2), (w / 2, h, d / 2)), "side")
        m.add(_box((-w / 2 + t, 0, -d / 2), (w / 2 - t, h, -d / 2 + 0.012)), "back"); m.add(_box((-w / 2 + t, 0, -d / 2 + 0.012), (w / 2 - t, pl, d / 2 - 0.015)), "plinth")
        m.add(_box((-w / 2 - 0.015, h - t, -d / 2), (w / 2 + 0.015, h, d / 2 + 0.02)), "top")
    n = max(2, round((h - pl - t) / (0.34 if not goods else 0.45))); pitch = (h - pl - t) / n; st = 0.022; k = 0
    for s in range(n):
        y = pl + s * pitch
        m.add(_box((-w / 2 + (0 if goods else t), y, -d / 2 + (0 if goods else 0.012)), (w / 2 - (0 if goods else t), y + st, d / 2 - (0 if goods else 0.01))), "shelf")
        y += st; clear = pitch - st - 0.025; x = -w / 2 + t + 0.006; x1 = w / 2 - t - 0.006; zb = -d / 2 + (0.03 if goods else 0.02); dep = d - (0.07 if goods else 0.05)
        if goods and s == n - 1 and h > 1.7: clear = min(clear, 0.3)
        while x < x1 - 0.03:
            r = rnd.random()
            if goods:
                bw = min(rnd.uniform(0.16, 0.42), x1 - x); bh = clear * rnd.uniform(0.45, 0.95); bd = dep * rnd.uniform(0.7, 1.0)
                if bw < 0.1: break
                if r < 0.55: m.add(_box((x, y, zb + (dep - bd)), (x + bw, y + bh, zb + dep)), f"box {k % 5}")
                elif r < 0.85:      # a row of tins or jars
                    rr = min(bw, 0.16) / 2; cnt = max(1, int(bw / (2 * rr + 0.01)))
                    for j in range(cnt):
                        Vt, Ft = _lathe([(0, 0), (rr, 0), (rr, min(bh, 0.3) * 0.92), (rr * 0.8, min(bh, 0.3)), (0, min(bh, 0.3))], 12); m.add((Vt + [x + rr + j * (2 * rr + 0.01), y, zb + dep - rr - 0.01], Ft), f"tin {(k + j) % 4}")
                x += bw + rnd.uniform(0.02, 0.09); k += 1; continue
            if r < 0.07 and x1 - x > 0.2: x += rnd.uniform(0.05, 0.16); continue          # a gap
            if r < 0.17 and x1 - x > 0.3:      # a stack lying flat
                sw = min(rnd.uniform(0.2, 0.26), x1 - x); yy = y
                for j in range(rnd.randrange(2, 5)):
                    bt = rnd.uniform(0.025, 0.05)
                    if yy + bt > y + clear: break
                    f = rnd.uniform(0.82, 1.0); m.add(_box((x + (1 - f) * sw / 2, yy, zb + dep - min(dep, 0.19) * f), (x + sw - (1 - f) * sw / 2, yy + bt * 0.95, zb + dep - 0.004)), f"book {k % 8}"); yy += bt; k += 1
                x += sw + 0.012; continue
            run = rnd.randrange(3, 9); hh = clear * rnd.uniform(0.7, 0.97); dd = min(dep, 0.22) * rnd.uniform(0.75, 1.0)       # a run of one set, then single odd ones
            for j in range(run):
                bt = rnd.uniform(0.02, 0.05)
                if x + bt > x1: x = x1; break
                bh = hh * rnd.uniform(0.96, 1.0) if rnd.random() < 0.6 else clear * rnd.uniform(0.6, 0.98); bd = dd if rnd.random() < 0.6 else min(dep, 0.22) * rnd.uniform(0.7, 1.0)
                V, F = _box((x, y, zb + dep - bd - 0.004 - rnd.uniform(0, 0.012)), (x + bt * 0.94, y + bh, zb + dep - 0.004))
                if j == run - 1 and rnd.random() < 0.3 and x + bt + 0.06 < x1:      # the last of a run leans on its neighbours
                    a = rnd.uniform(0.1, 0.22); c, s_ = math.cos(a), math.sin(a); px, py = V[:, 0] - x, V[:, 1] - y; V[:, 0] = x + px * c + py * s_ * 0.0 - 0.0 + py * math.tan(a) * 0.0; V[:, 0] = x + px * c - py * s_ + bh * s_; V[:, 1] = y + px * s_ + py * c; V[:, 1] -= V[:, 1].min() - y; x += bh * s_
                m.add((V, F), f"book {k % 8}"); x += bt; k += 1
    return m.done()

def drawers(w, h, d, rnd, name="", **kw):
    """A closed case with fronts: a chest of drawers, a tool chest, a cupboard or locker with doors, a counter. Fronts stand proud, with handles."""
    m = Mesh(); nm = name.lower(); t = 0.02; pl = min(0.08, 0.08 * h); proud = 0.014
    m.add(_box((-w / 2, pl, -d / 2), (w / 2, h - t, d / 2 - proud)), "case"); m.add(_box((-w / 2 - 0.012, h - t, -d / 2), (w / 2 + 0.012, h, d / 2 + 0.006)), "top")
    if any(k in nm for k in ("tool", "trolley", "cart")) and h < 1.3:
        for sx in (-1, 1):
            for sz in (-1, 1): V, F = _lathe([(0, -0.02), (pl * 0.5, -0.02), (pl * 0.5, 0.02), (0, 0.02)], 12); m.add((np.stack([sx * (w / 2 - 0.08) + V[:, 1], pl * 0.5 + V[:, 0], sz * (d / 2 - 0.08) + V[:, 2]], 1), F[:, ::-1]), "wheel")
    else: m.add(_box((-w / 2 + 0.02, 0, -d / 2 + 0.02), (w / 2 - 0.02, pl, d / 2 - 0.04)), "plinth")
    doors = h > 1.3 or any(k in nm for k in ("cupboard", "wardrobe", "locker", "cabinet", "armoire", "fridge", "counter", "bar"))
    y0, y1 = pl + 0.015, h - t - 0.015; zf = d / 2 - proud; g = 0.008
    def handle(cx, cy, vertical):
        hw, hh = (0.012, 0.07) if vertical else (min(0.09, 0.14 * w), 0.012)
        m.add(_box((cx - hw, cy - hh, zf + proud), (cx + hw, cy + hh, zf + proud + 0.022)), "handle")
    if doors:
        nd = 1 if w < 0.6 else 2 if w < 1.5 else 3; dw = (w - 0.03) / nd; ndr = 1 if (h > 1.6 and "wardrobe" in nm) or h < 1.0 and rnd.random() < 0.5 else 0
        yd = y0 + (0.2 * (y1 - y0) if ndr and h > 1.6 else 0.0)
        if ndr and h > 1.6: m.add(_box((-w / 2 + 0.015 + g, y0, zf), (w / 2 - 0.015 - g, yd - g, zf + proud)), "drawer"); handle(0, (y0 + yd) / 2, False)
        for k in range(nd):
            x0 = -w / 2 + 0.015 + k * dw; m.add(_box((x0 + g, yd, zf), (x0 + dw - g, y1, zf + proud)), "door"); m.add(_box((x0 + g + 0.05, yd + 0.05, zf + proud), (x0 + dw - g - 0.05, y1 - 0.05, zf + proud + 0.004)), "door panel")
            handle(x0 + (dw - 0.05 if k % 2 == 0 and nd > 1 else 0.05), (yd + y1) / 2, True)
    else:
        n = max(2, round((y1 - y0) / (0.2 if "tool" not in nm else 0.11))); cols = 1 if w < 1.3 else 2; cw = (w - 0.03) / cols; ys = np.linspace(y0, y1, n + 1)
        for c in range(cols):
            for k in range(n):
                x0 = -w / 2 + 0.015 + c * cw; m.add(_box((x0 + g, ys[k] + g / 2, zf), (x0 + cw - g, ys[k + 1] - g / 2, zf + proud)), "drawer"); handle(x0 + cw / 2, (ys[k] + ys[k + 1]) / 2, False)
    return m.done()

def rings(w, h, d, rnd, **kw):
    """Stacked tyres, or a coil of rope or hose: rings lying on one another, each a little off the one below."""
    m = Mesh(); R = 0.5; n = max(1, round(h / (0.32 * max(w, d)))) if h > 0.35 * max(w, d) else 1; t = 1.0 / n
    for k in range(n):
        r = min(t / 2, 0.19) ; off = (rnd.uniform(-0.03, 0.03), rnd.uniform(-0.03, 0.03)) if k else (0, 0)
        prof = [(R - 0.03 - r + r * math.cos(a) * 1.0, t / 2 + (t / 2) * math.sin(a)) for a in np.linspace(-math.pi, math.pi, 13)]
        V, F = _lathe(prof, 28); V = V * [0.94, 1, 0.94] + [off[0], k * t, off[1]]; m.add((V, F), "tyre" if n > 1 or h > 0.15 else "coil")
    V, F, P, names = m.done(); return _fit(V, w, h, d), F, P, names

def lamp(w, h, d, rnd):
    """A stand-in for a light whose own shape could not be sculpted: what holds it ("housing") and the part that shines ("glow")."""
    m = Mesh()
    if h > 1.5 * max(w, d):          # a standing lamp: a foot and a post, a shade over the top third
        V, F = _lathe([(0, 0), (0.42, 0), (0.42, 0.02), (0.06, 0.04), (0.04, 0.68), (0, 0.68)]); m.add((V * [w, h, d], F), "housing")
        V, F = _lathe([(0, 0.68), (0.5, 0.68), (0.36, 1.0), (0, 1.0)]); m.add((V * [w, h, d], F), "glow")
    else:                            # a fitting on a ceiling or a wall, or a low lamp: a housing above, a shining face below it
        t = 0.4 * h; e = min(0.02, 0.15 * min(w, d))
        m.add(_box((-w / 2, h - t, -d / 2), (w / 2, h, d / 2)), "housing"); m.add(_box((-w / 2 + e, 0, -d / 2 + e), (w / 2 - e, h - t, d / 2 - e)), "glow")
    return m.done()

FORMS = {"lamp": lamp, "rug": rug, "frame": frame, "board": board, "books": books, "crate": crate, "cylinder": cylinder, "table": table, "bottle": bottle, "jar": jar, "bowl": bowl, "pot": pot, "candle": candle, "cloth": cloth, "bookcase": bookcase, "drawers": drawers, "rings": rings}

def make(form, w, h, d, seed="", **opts):
    """opts: "name" (the thing's name: a rack is filled with goods, a tool chest has wheels), "gap" (a curtain drawn apart)."""
    rnd = random.Random(zlib.crc32(str(seed).encode())); f = FORMS[form]
    V, F, P, names = f(float(w), float(h), float(d), rnd, **opts) if form in ("cloth", "bookcase", "drawers", "rings", "books", "table", "crate") else f(float(w), float(h), float(d), rnd)
    return V.astype(np.float64), F.astype(np.int32), P.astype(np.int16), names

def to_room(V, x, y0, z, yaw):
    """Own frame -> room metres, as llm_boxes.to_world turns a piece: its front (+z) faces (sin yaw, 0, cos yaw)."""
    cs, sn = math.cos(yaw), math.sin(yaw)
    return np.stack([x + cs * V[:, 0] + sn * V[:, 2], y0 + V[:, 1], z - sn * V[:, 0] + cs * V[:, 2]], 1)

if __name__ == "__main__":      # self-check: every form fills its box and (closed ones) has outward faces
    import trimesh
    for f, (w, h, d) in {"rug": (2.4, 0.02, 1.6), "frame": (0.8, 0.6, 0.04), "board": (1.0, 0.2, 0.22), "books": (0.4, 0.25, 0.18), "crate": (0.6, 0.5, 0.4), "cylinder": (0.6, 0.9, 0.6),
                         "table": (1.5, 0.9, 0.7), "bottle": (0.08, 0.3, 0.08), "jar": (0.12, 0.16, 0.12), "bowl": (0.3, 0.12, 0.3), "pot": (0.3, 0.45, 0.3), "candle": (0.1, 0.3, 0.1), "cloth": (2.0, 2.4, 0.15)}.items():
        V, F, P, names = make(f, w, h, d, f); ext = V.max(0) - V.min(0); t = trimesh.Trimesh(V, F, process=False)
        print(f"{f:9s} {len(V):5d} v {len(F):5d} f  box {np.round(ext, 3)}  volume {t.volume:+.4f}  parts {names}")
