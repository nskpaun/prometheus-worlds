"""The architecture kit's mesh builder and the primitives every kit piece is made of. numpy only.
Frame: room metres as boxes.json (x right, y up, z to the back wall). Everything is real size.

A Builder collects PARTS. A part is one piece of architecture (one skirting run, one window frame, one beam) with
  piece    what it is ("skirting", "beam", "window.frame", ...)
  group    its material group ("wall", "floor", "ceiling", "trim", "timber", "stone", "metal", "glass", "door", "concrete", "emitter")
  V, F, UV triangles; vertices are NOT shared between flat faces, so every face is flat and has its own UVs
  UV       in METRES (true scale, not 0..1): a flat mapping per face. With grain=True u runs along the piece's length
           (wood grain, the run of a moulding), otherwise u is the horizontal direction on a wall and (x, z) on a level face,
           anchored to the room so neighbouring faces of one material line up.
Normals are flat, from the winding; every primitive is told which way is out."""
import numpy as np

Y = np.array([0.0, 1.0, 0.0])

def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v); return v / n if n > 1e-12 else v

class Builder:
    def __init__(s): s.parts = []; s.cur = None
    def part(s, piece, group, grain=False, **meta):
        s.cur = dict(piece=piece, group=group, grain=grain, V=[], UV=[], F=[], n=0, **meta); s.parts.append(s.cur); return s.cur

    # ---- the one way triangles get in: a flat convex polygon, told which way is out
    def poly(s, P, out, grain=None, uv=None):
        P = np.asarray(P, float); k = len(P)
        if k < 3: return
        n = np.zeros(3)
        for i in range(1, k - 1): n += np.cross(P[i] - P[0], P[i + 1] - P[0])
        if np.linalg.norm(n) < 1e-12: return
        n = unit(n)
        if out is not None and n @ np.asarray(out, float) < 0: P = P[::-1]; n = -n; uv = None if uv is None else np.asarray(uv)[::-1]
        if uv is None:
            g = grain if grain is not None else None
            if g is not None:
                eu = np.asarray(g, float) - (np.asarray(g, float) @ n) * n
                if np.linalg.norm(eu) < 1e-6: g = None
            if g is None: eu = np.array([1.0, 0, 0]) if abs(n[1]) > 0.9 else np.cross(Y, n)
            eu = unit(eu); ev = np.cross(n, eu); uv = np.stack([P @ eu, P @ ev], 1)
        c = s.cur; b = c["n"]; c["V"].append(P); c["UV"].append(np.asarray(uv, float)); c["n"] += k
        c["F"].append(np.array([[b, b + i, b + i + 1] for i in range(1, k - 1)]))

    def box(s, c, R, h, skip=(), grain=None):
        """A box: centre c, axes R (3 unit rows), half sizes h. skip: faces left out, e.g. "-0" = the face at minus axis 0."""
        c = np.asarray(c, float); R = np.asarray(R, float); h = np.asarray(h, float)
        if grain is True: grain = R[int(np.argmax(h))]
        for a in range(3):
            b1, b2 = (a + 1) % 3, (a + 2) % 3
            for sg in (-1, 1):
                if f"{'-' if sg < 0 else '+'}{a}" in skip: continue
                o = c + sg * h[a] * R[a]
                s.poly([o - h[b1] * R[b1] - h[b2] * R[b2], o + h[b1] * R[b1] - h[b2] * R[b2], o + h[b1] * R[b1] + h[b2] * R[b2], o - h[b1] * R[b1] + h[b2] * R[b2]], sg * R[a], grain)

    def abox(s, lo, hi, skip=(), grain=None):
        """An axis-aligned box from corner lo to corner hi."""
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi); s.box((lo + hi) / 2, np.eye(3), (hi - lo) / 2, skip, grain)

    def prism(s, poly, o, ea, eb, ec, length, caps=(True, True), grain=True):
        """A convex 2D section poly [(a, b)] in the plane (ea, eb) at o, pushed `length` along ec."""
        o, ea, eb, ec = (np.asarray(v, float) for v in (o, ea, eb, ec)); poly = np.asarray(poly, float); k = len(poly)
        P0 = o + poly[:, :1] * ea + poly[:, 1:] * eb; P1 = P0 + length * ec; cen = P0.mean(0)
        for i in range(k):
            j = (i + 1) % k; mid = (P0[i] + P0[j]) / 2
            s.poly([P0[i], P0[j], P1[j], P1[i]], mid - cen, ec if grain else None)
        if caps[0]: s.poly(P0, -ec)
        if caps[1]: s.poly(P1, ec)

    def cyl(s, a, b, r, n=16, caps=(True, True), r2=None):
        """A round bar from point a to point b (r2: the radius at b, for a taper)."""
        a, b = np.asarray(a, float), np.asarray(b, float); ax = unit(b - a); t = np.cross(ax, [0, 1.0, 0])
        if np.linalg.norm(t) < 1e-6: t = np.cross(ax, [1.0, 0, 0])
        t = unit(t); w = np.cross(ax, t); r2 = r if r2 is None else r2
        an = np.arange(n) / n * 2 * np.pi; ring = np.cos(an)[:, None] * t + np.sin(an)[:, None] * w
        P0, P1 = a + r * ring, b + r2 * ring; L = np.linalg.norm(b - a)
        for i in range(n):
            j = (i + 1) % n
            s.poly([P0[i], P0[j], P1[j], P1[i]], ring[i] + ring[j], uv=[[i / n * 2 * np.pi * r, 0], [(i + 1) / n * 2 * np.pi * r, 0], [(i + 1) / n * 2 * np.pi * r, L], [i / n * 2 * np.pi * r, L]])
        if caps[0]: s.poly(P0, -ax)
        if caps[1]: s.poly(P1, ax)

    def lathe(s, prof, c, n=24, caps=(False, True), axis=Y):
        """Turn a profile [(radius, height)] round a vertical line through c (heights measured from c)."""
        c = np.asarray(c, float); prof = np.asarray(prof, float); an = np.arange(n + 1) / n * 2 * np.pi
        ring = np.stack([np.cos(an), 0 * an, np.sin(an)], 1); ln = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))])
        for k in range(len(prof) - 1):
            (r0, y0), (r1, y1) = prof[k], prof[k + 1]; dr, dy = r1 - r0, y1 - y0
            for i in range(n):
                P = [c + r0 * ring[i] + y0 * axis, c + r0 * ring[i + 1] + y0 * axis, c + r1 * ring[i + 1] + y1 * axis, c + r1 * ring[i] + y1 * axis]
                out = (ring[i] + ring[i + 1]) * dy - axis * dr * 2
                rm = max(r0, r1); s.poly(P, out if np.linalg.norm(out) > 1e-9 else axis, uv=[[an[i] * rm, ln[k]], [an[i + 1] * rm, ln[k]], [an[i + 1] * rm, ln[k + 1]], [an[i] * rm, ln[k + 1]]])
        if caps[0]: s.poly(c + prof[0, 0] * ring[:-1] + prof[0, 1] * axis, -axis)
        if caps[1]: s.poly(c + prof[-1, 0] * ring[:-1] + prof[-1, 1] * axis, axis)

    def sweep(s, path, prof, o, ea, eb, ec, side=1, closed=False, caps=True):
        """Run a moulding along a path with mitred corners.
        path [(a, b)] lies in the plane (ea, eb) at o. prof [(off, c)]: off = distance to the LEFT of the direction of travel
        (times side), c = distance along ec. The profile is listed so that its outside is at (dc, -doff): from its far, low
        end round to the path. Open runs get flat end caps (the profile closed back onto off = 0)."""
        o, ea, eb, ec = (np.asarray(v, float) for v in (o, ea, eb, ec)); path = np.asarray(path, float); prof = np.asarray(prof, float); n = len(path)
        keep = [0]
        for i in range(1, n):
            if np.abs(path[i] - path[keep[-1]]).sum() > 1e-7: keep.append(i)
        if closed and len(keep) > 1 and np.abs(path[keep[-1]] - path[keep[0]]).sum() < 1e-7: keep.pop()
        path = path[keep]; n = len(path)
        if n < 2 or (closed and n < 3): return
        d = np.diff(np.concatenate([path, path[:1]]) if closed else path, axis=0); L = np.linalg.norm(d, axis=1)
        d = d / L[:, None]; left = np.stack([-d[:, 1], d[:, 0]], 1) * side; m = np.zeros((n, 2))
        for i in range(n):
            a = left[i - 1] if (closed or i > 0) else None; b = left[i] if (closed or i < n - 1) else None
            if a is None: m[i] = b
            elif b is None: m[i] = a
            else: m[i] = (a + b) / max(1 + a @ b, 0.2)
        nseg = n if closed else n - 1; run = np.concatenate([[0], np.cumsum(L)]); pl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))])
        def pt(i, k): q = path[i % n] + prof[k, 0] * m[i % n]; return o + q[0] * ea + q[1] * eb + prof[k, 1] * ec
        for i in range(nseg):
            n3 = left[i, 0] * ea + left[i, 1] * eb
            for k in range(len(prof) - 1):
                doff, dc = prof[k + 1] - prof[k]
                s.poly([pt(i, k), pt(i + 1, k), pt(i + 1, k + 1), pt(i, k + 1)], dc * n3 - doff * ec, uv=[[run[i], pl[k]], [run[i + 1], pl[k]], [run[i + 1], pl[k + 1]], [run[i], pl[k + 1]]])
        if caps and not closed:
            for i, sg in ((0, -1), (n - 1, 1)):
                dd = d[0] if i == 0 else d[-1]; out = sg * (dd[0] * ea + dd[1] * eb)
                base = [o + path[i][0] * ea + path[i][1] * eb + prof[0, 1] * ec] + [pt(i, k) for k in range(len(prof))] + [o + path[i][0] * ea + path[i][1] * eb + prof[-1, 1] * ec]
                for k in range(1, len(base) - 1): s.poly([base[0], base[k], base[k + 1]], out)

    def result(s):
        out = []
        for c in s.parts:
            if not c["n"]: continue
            F, b = [], 0
            r = {k: v for k, v in c.items() if k not in ("V", "UV", "F", "n")}
            r["V"] = np.concatenate(c["V"]).astype(np.float32); r["UV"] = np.concatenate(c["UV"]).astype(np.float32); r["F"] = np.concatenate(c["F"]).astype(np.uint32)
            out.append(r)
        return out

# ---------------------------------------------------------------- outlines of openings, in a wall's (u, v): u along the wall, v up
def outline(u0, u1, v0, v1, head="flat", rise=None, n=10):
    """The edge of an opening, counter-clockwise from its bottom-left, and the corner fans that fill the wall between the
    opening and its bounding rectangle: [(corner, chain of edge points)]. head: flat | round | segmental | pointed | circle."""
    w = u1 - u0; um = (u0 + u1) / 2
    if head == "flat": return np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]), []
    if head == "circle":
        r = min(w, v1 - v0) / 2; c = np.array([um, (v0 + v1) / 2]); a = -np.pi / 2 + np.arange(4 * n) / (4 * n) * 2 * np.pi
        pts = c + r * np.stack([np.cos(a), np.sin(a)], 1); fans = []
        for q, cor in enumerate([[c[0] + r, c[1] - r], [c[0] + r, c[1] + r], [c[0] - r, c[1] + r], [c[0] - r, c[1] - r]]):
            ch = [pts[(q * n + i) % (4 * n)] for i in range(n + 1)]; fans.append((np.array(cor), np.array(ch)))
        return pts, fans
    if head == "round": rise = w / 2
    if head == "pointed": rise = rise or 0.8 * w
    if head == "segmental": rise = rise or 0.18 * w
    rise = min(rise, v1 - v0 - 0.05); vs = v1 - rise
    if rise > w / 2 + 1e-6 or head == "pointed":
        R = (rise * rise + w * w / 4) / w; at = np.arccos(np.clip((um - (u1 - R)) / R, -1, 1)); t = np.linspace(0, at, n + 1)
        right = np.stack([u1 - R + R * np.cos(t), vs + R * np.sin(t)], 1); left = np.stack([u0 + R - R * np.cos(t[::-1]), vs + R * np.sin(t[::-1])], 1)
    else:
        rho = (rise * rise + w * w / 4) / (2 * rise); k = rho - rise; a0 = np.arcsin(np.clip(k / rho, -1, 1)); t = np.linspace(a0, np.pi / 2, n + 1)
        right = np.stack([um + rho * np.cos(t), vs - k + rho * np.sin(t)], 1); left = np.stack([um - rho * np.cos(t[::-1]), vs - k + rho * np.sin(t[::-1])], 1)
    pts = np.concatenate([[[u0, v0], [u1, v0]], right, left[1:]])
    return pts, [(np.array([u1, v1]), right), (np.array([u0, v1]), left)]

def span_v(pts, u):
    """The lowest and highest v of the outline at this u."""
    vs = []
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        if abs(b[0] - a[0]) < 1e-9:
            if abs(a[0] - u) < 1e-9: vs += [a[1], b[1]]
        elif min(a[0], b[0]) - 1e-9 <= u <= max(a[0], b[0]) + 1e-9: vs.append(a[1] + (u - a[0]) / (b[0] - a[0]) * (b[1] - a[1]))
    return (min(vs), max(vs)) if vs else (0.0, 0.0)

def span_u(pts, v): return span_v(pts[:, ::-1], v)

def shrink(pts, k, about=None):
    """The outline scaled by k about its bounding box centre (keeping its foot where it is when about = 'foot')."""
    lo, hi = pts.min(0), pts.max(0); c = (lo + hi) / 2
    if about == "foot": c = np.array([c[0], lo[1]])
    return c + (pts - c) * k

def inset(pts, d):
    """A convex outline moved inwards by d metres (mitred)."""
    n = len(pts); out = np.zeros_like(pts); cen = pts.mean(0)
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]; d0, d1 = unit(b - a), unit(c - b)
        n0, n1 = np.array([-d0[1], d0[0]]), np.array([-d1[1], d1[0]])
        if n0 @ (cen - b) < 0: n0 = -n0
        if n1 @ (cen - b) < 0: n1 = -n1
        out[i] = b + d * (n0 + n1) / max(1 + n0 @ n1, 0.3)
    return out
