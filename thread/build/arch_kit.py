"""The architecture kit. Every piece is a real mesh with depth, sized in metres, built on arch_mesh.Builder.
Nothing here knows a sentence or a style name: pieces take sizes and a few shape words; arch.py chooses and places them.

A Frame is a flat surface seen from inside the room: P(u, v, d) = o + u*right + v*up + d*n, with n pointing into the room
(d < 0 is inside the wall). The four walls, the ceiling and every straight stretch of the room's plan are Frames."""
import numpy as np
from arch_mesh import Builder, Y, unit, outline, span_v, span_u, inset

class Frame:
    def __init__(s, name, o, right, n, L, H, up=Y):
        s.name, s.o, s.right, s.n, s.L, s.H, s.up = name, np.asarray(o, float), np.asarray(right, float), np.asarray(n, float), float(L), float(H), np.asarray(up, float)
    def P(s, u, v, d=0.0): return s.o + u * s.right + v * s.up + d * s.n
    def R(s): return np.stack([s.right, s.up, s.n])
    def box(s, b, u0, u1, v0, v1, d0, d1, skip=(), grain=None):
        """A box on the frame: u0..u1 along, v0..v1 up, d0..d1 out. grain: 0|1|2 = the axis the wood runs along."""
        g = s.R()[grain] if isinstance(grain, int) and not isinstance(grain, bool) else grain
        b.box(s.P((u0 + u1) / 2, (v0 + v1) / 2, (d0 + d1) / 2), s.R(), [abs(u1 - u0) / 2, abs(v1 - v0) / 2, abs(d1 - d0) / 2], skip, g)
    def uvd(s, p): p = np.asarray(p, float) - s.o; return p @ s.right, p @ s.up, p @ s.n

WALLS = ("z1", "x1", "z0", "x0")     # going round the room; each wall's right end meets the next one's left end
def room_frames(W, H, D):
    X, Z = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    F = dict(z1=Frame("z1", [0, 0, D], X, -Z, W, H), x1=Frame("x1", [W, 0, D], -Z, -X, D, H), z0=Frame("z0", [W, 0, 0], -X, Z, W, H), x0=Frame("x0", [0, 0, 0], Z, X, D, H))
    F["ceil"] = Frame("ceil", [0, H, 0], X, -Y, W, D, up=Z); F["floor"] = Frame("floor", [0, 0, 0], X, Y, W, D, up=Z)
    return F

# ---------------------------------------------------------------- moulding profiles [(off, height)], listed far-low end first (see Builder.sweep)
def arc(c, r, a0, a1, n=6):
    a = np.linspace(np.radians(a0), np.radians(a1), n + 1); return [(c[0] + r * np.cos(t), c[1] + r * np.sin(t)) for t in a]

def prof_skirting(kind, h, t):
    if kind == "plinth": return [(t, 0), (t, h - 1.4 * t), (0.25 * t, h), (0, h)]                       # a stone plinth course with a weathered splay
    if kind == "ogee":   return [(t, 0), (t, 0.55 * h), (0.88 * t, 0.58 * h), (0.88 * t, 0.62 * h)] + arc((0.88 * t, 0.62 * h + 0.3 * t), 0.3 * t, -90, -180, 3)[1:] + arc((0.3 * t, 0.62 * h + 0.3 * t), 0.28 * t, 0, 90, 3)[1:] + [(0.3 * t, h), (0, h)]
    if kind == "cove":   return [(h - h * np.cos(a), h - h * np.sin(a)) for a in np.linspace(np.pi / 2, 0, 8)]   # floor-to-wall coving, radius h
    if kind == "kick":   return [(t, 0), (t, 0.35 * h), (0.3 * t, h), (0, h)]                              # a canted metal kick strip
    return [(t, 0), (t, h - 0.008), (t - 0.008, h), (0, h)]                                                # plain board, eased edge

def prof_cornice(kind, h, p):
    """From the wall (0, -h) out to the ceiling (p, 0); heights are below the ceiling."""
    if kind == "cove":    return [(0, -h), (0.06 * p, -h)] + [(0.06 * p + 0.94 * p * (1 - np.cos(a)), -h + h * np.sin(a)) for a in np.linspace(0, np.pi / 2, 8)][1:]
    if kind == "classic": return ([(0, -h), (0.10 * p, -h), (0.10 * p, -0.90 * h)] + [(0.10 * p + 0.45 * p * (1 - np.cos(a)), -0.90 * h + 0.45 * h * np.sin(a)) for a in np.linspace(0, np.pi / 2, 6)][1:]
                                  + [(0.62 * p, -0.45 * h), (0.62 * p, -0.38 * h)] + arc((0.62 * p, -0.2 * h), 0.18 * h, -90, 0, 4)[1:] + [(p, -0.2 * h), (p, 0)])
    if kind == "square":  return [(0, -h), (p, -h), (p, 0)]
    return [(0, -h), (p, -0.25 * h), (p, 0)]                                                               # a plain splay

def prof_rail(kind, y, h, t):
    if kind == "square": return [(0, y - h / 2), (t, y - h / 2), (t, y + h / 2), (0, y + h / 2)]
    return [(0, y - h / 2), (0.45 * t, y - h / 2), (0.45 * t, y - 0.3 * h)] + arc((0.45 * t, y), 0.3 * h, -90, 90, 6)[1:] + [(0.45 * t, y + h / 2), (0, y + h / 2)]

def prof_casing(kind, w, t):
    """Round an opening: off runs away from the opening's edge, the second number is how far it stands off the wall."""
    if kind == "stone":  return [(w, 0), (w, t), (0.045, t), (0, t - 0.045)]
    if kind == "ogee":   return [(w, 0), (w, t), (0.82 * w, t), (0.74 * w, 0.75 * t), (0.3 * w, 0.6 * t), (0.22 * w, 0.75 * t), (0.1 * w, 0.75 * t), (0.04 * w, 0.55 * t), (0, 0.55 * t), (0, 0)]
    if kind == "metal":  return [(w, 0), (w, t), (0, t), (0, 0)]
    return [(w, 0), (w, t), (0.01, t), (0, t - 0.01), (0, 0)]

# ---------------------------------------------------------------- the plan: the room's outline with wall bumps, cut where openings reach the floor
def plan_runs(fr, bumps, gaps=None, level=None):
    """The room's inner outline as polylines of (x, z), going round z1, x1, z0, x0.
    bumps: [dict(wall, u0, u1, d, y0, y1)] stand out of a wall (chimney breast, pier, pilaster); a moulding at height
    `level` goes round those that reach it. gaps: {wall: [(u0, u1)]} are left out (doors, alcoves, tall openings).
    -> (runs, closed): closed is True when there is one unbroken loop."""
    gaps = gaps or {}; runs = [[]]
    def P(w, u, d=0.0): p = fr[w].P(u, 0, d); return (float(p[0]), float(p[2]))
    for w in WALLS:
        L = fr[w].L; cuts = sorted((max(0, a), min(L, b)) for a, b in gaps.get(w, []) if b > a)
        bs = sorted((b for b in bumps if b["wall"] == w and (level is None or b.get("y0", 0) - 1e-6 <= level <= b.get("y1", 1e9) + 1e-6)), key=lambda b: b["u0"])
        segs = []; u = 0.0                                     # (u0, u1, depth) flat stretches and the bump faces
        for b_ in bs:
            if b_["u0"] > u + 1e-6: segs.append((u, b_["u0"], 0.0))
            segs.append((b_["u0"], b_["u1"], b_["d"])); u = b_["u1"]
        if u < L - 1e-6: segs.append((u, L, 0.0))
        for a, c, d in segs:
            if d > 0:
                runs[-1] += [P(w, a), P(w, a, d), P(w, c, d), P(w, c)] if a > 1e-6 and c < L - 1e-6 else [P(w, a, d), P(w, c, d)]
                continue
            x = a
            for g0, g1 in cuts:
                if g1 <= x or g0 >= c: continue
                if g0 > x + 1e-6: runs[-1] += [P(w, x), P(w, g0)]
                runs.append([]); x = max(x, g1)
            if x < c - 1e-6: runs[-1] += [P(w, x), P(w, c)]
    closed = len(runs) == 1
    if not closed and runs[-1] and runs[0]: runs[0] = runs[-1] + runs[0]; runs.pop()
    out = []
    for r in runs:
        q = []
        for p in r:
            if not q or abs(p[0] - q[-1][0]) + abs(p[1] - q[-1][1]) > 1e-6: q.append(p)
        if closed and len(q) > 2 and abs(q[0][0] - q[-1][0]) + abs(q[0][1] - q[-1][1]) < 1e-6: q.pop()
        k = 1                                                  # drop points in the middle of a straight stretch
        while k < len(q) - 1:
            a, b_, c = np.array(q[k - 1]), np.array(q[k]), np.array(q[k + 1])
            if abs((b_ - a)[0] * (c - b_)[1] - (b_ - a)[1] * (c - b_)[0]) < 1e-9 and (b_ - a) @ (c - b_) > 0: q.pop(k)
            else: k += 1
        if len(q) >= 2: out.append(q)
    return out, closed

def mould(b, runs, closed, prof):
    """A moulding swept round the plan (profile heights are absolute)."""
    for r in runs: b.sweep(r, prof, [0, 0, 0], [1, 0, 0], [0, 0, 1], Y, side=-1, closed=closed)

def run_frames(runs, closed, H):
    """Each straight stretch of the plan as a Frame (for panelling)."""
    out = []
    for r in runs:
        pts = r + [r[0]] if closed else r
        for a, c in zip(pts[:-1], pts[1:]):
            a3, c3 = np.array([a[0], 0, a[1]]), np.array([c[0], 0, c[1]]); L = np.linalg.norm(c3 - a3)
            if L < 1e-6: continue
            rt = (c3 - a3) / L; out.append(Frame("run", a3, rt, np.cross(Y, rt), L, H))      # going round the room this way, right x up points in
    return out

# ---------------------------------------------------------------- a flat surface with openings cut in it
def surface(b, f, L, H, opens, v_lo=0.0):
    """The frame's rectangle 0..L x v_lo..H minus the openings' outlines (convex, inside their bounding boxes)."""
    us = sorted({0.0, L} | {float(x) for o in opens for x in (o["u0"], o["u1"]) if 0 < x < L}); vs = sorted({v_lo, H} | {float(x) for o in opens for x in (o["v0"], o["v1"]) if v_lo < x < H})
    for i in range(len(us) - 1):
        for j in range(len(vs) - 1):
            um, vm = (us[i] + us[i + 1]) / 2, (vs[j] + vs[j + 1]) / 2
            if any(o["u0"] < um < o["u1"] and o["v0"] < vm < o["v1"] for o in opens): continue
            b.poly([f.P(us[i], vs[j]), f.P(us[i + 1], vs[j]), f.P(us[i + 1], vs[j + 1]), f.P(us[i], vs[j + 1])], f.n)
    for o in opens:
        for cor, ch in o["fans"]:
            for k in range(len(ch) - 1): b.poly([f.P(*cor), f.P(*ch[k]), f.P(*ch[k + 1])], f.n)

def make_opening(kind, wall, u0, u1, v0, v1, head="flat", rise=None, **kw):
    pts, fans = outline(u0, u1, v0, v1, head, rise)
    if head == "circle": u0, u1, v0, v1 = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()
    return dict(kind=kind, wall=wall, u0=float(u0), u1=float(u1), v0=float(v0), v1=float(v1), head=head, pts=pts, fans=fans, **kw)

def fan(b, f, pts, d, out): b.poly([f.P(p[0], p[1], d) for p in pts], out)

def panel_field(b, f, u0, u1, v0, v1, d0, nx, ny, stile, rail, t, bevel=0.03, flat=False, grain=True):
    """Frame-and-panel joinery standing on the plane d0: stiles and rails t proud, each panel raised and fielded (or flat, set back)."""
    cw = (u1 - u0 - stile * (nx + 1)) / nx; ch = (v1 - v0 - rail * (ny + 1)) / ny
    if cw <= 0.02 or ch <= 0.02: f.box(b, u0, u1, v0, v1, d0, d0 + t, skip=("-2",)); return
    for i in range(nx + 1):
        a = u0 + i * (cw + stile); f.box(b, a, a + stile, v0, v1, d0, d0 + t, skip=("-2",), grain=1 if grain else None)
    for j in range(ny + 1):
        c = v0 + j * (ch + rail)
        for i in range(nx):
            a = u0 + stile + i * (cw + stile); f.box(b, a, a + cw, c, c + rail, d0, d0 + t, skip=("-2", "-0", "+0"), grain=0 if grain else None)
    for i in range(nx):
        for j in range(ny):
            a, c = u0 + stile + i * (cw + stile), v0 + rail + j * (ch + rail); g = f.up if grain else None
            if flat: b.poly([f.P(a, c, d0 + 0.004), f.P(a + cw, c, d0 + 0.004), f.P(a + cw, c + ch, d0 + 0.004), f.P(a, c + ch, d0 + 0.004)], f.n, g); continue
            bv = min(bevel, cw / 3, ch / 3); o4 = [(a, c), (a + cw, c), (a + cw, c + ch), (a, c + ch)]; i4 = [(a + bv, c + bv), (a + cw - bv, c + bv), (a + cw - bv, c + ch - bv), (a + bv, c + ch - bv)]
            dl, dh = d0 + 0.2 * t, d0 + 0.75 * t
            for k in range(4):
                p, q, r, s_ = o4[k], o4[(k + 1) % 4], i4[(k + 1) % 4], i4[k]
                b.poly([f.P(*p, dl), f.P(*q, dl), f.P(*r, dh), f.P(*s_, dh)], f.n, g)
            b.poly([f.P(*p, dh) for p in i4], f.n, g)

def bars(b, f, pts, d, nx, ny, bw, bd):
    """Glazing bars across an outline at depth d: nx upright, ny level, each clipped to the outline."""
    lo, hi = pts.min(0), pts.max(0)
    for i in range(1, nx + 1):
        u = lo[0] + (hi[0] - lo[0]) * i / (nx + 1); a, c = span_v(pts, u - bw / 2); a2, c2 = span_v(pts, u + bw / 2)
        f.box(b, u - bw / 2, u + bw / 2, max(a, a2), min(c, c2), d, d + bd, skip=("-2",), grain=1)
    for j in range(1, ny + 1):
        v = lo[1] + (hi[1] - lo[1]) * j / (ny + 1); a, c = span_u(pts, v - bw / 2); a2, c2 = span_u(pts, v + bw / 2)
        f.box(b, max(a, a2), min(c, c2), v - bw / 2, v + bw / 2, d, d + bd * 0.9, skip=("-2",), grain=0)

def door_leaf(b, f, o, d, kind, mat):
    """The closed leaf filling opening o, its face at depth d (inside the wall)."""
    u0, u1, v0, v1 = o["u0"], o["u1"], o["v0"], o["v1"]; pts = o["inner"]; w = u1 - u0; vs = min(p[1] for p in pts[2:]) if o["head"] != "flat" else v1   # spring line
    b.part("door.leaf", mat["door"], grain=True, wall=o["wall"], style=kind); fan(b, f, pts, d, f.n)
    if kind == "panel":                                   # four or six panels, raised and fielded
        b.part("door.joinery", mat["door"], grain=True, wall=o["wall"])
        st = 0.11 if w < 1.3 else 0.13; leaves = 1 if w < 1.25 else 2
        for k in range(leaves):
            a, c = u0 + k * w / leaves + 0.004, u0 + (k + 1) * w / leaves - 0.004
            hts = [v0 + 0.004, v0 + 0.2 + 0.28 * (vs - v0), v0 + 0.12 + 0.62 * (vs - v0), vs - 0.004]
            for j in range(3): panel_field(b, f, a, c, hts[j] - (0.0 if j == 0 else st / 2), hts[j + 1] + (0 if j == 2 else st / 2), d, 2 if (c - a) > 0.62 else 1, 1, st, st, 0.035)
        b.part("door.handle", mat["metal"], wall=o["wall"])
        hu = u1 - 0.07 if leaves == 1 else u0 + w / 2 - 0.05
        b.cyl(f.P(hu, v0 + 1.0, d + 0.035), f.P(hu, v0 + 1.0, d + 0.09), 0.012, 10); b.lathe([(0.0, -0.03), (0.022, -0.02), (0.03, 0.0), (0.022, 0.02), (0.0, 0.03)], f.P(hu, v0 + 1.0, d + 0.1), 12, caps=(False, False))
    elif kind == "plank":                                 # ledged boards, strap hinges, a ring pull
        b.part("door.boards", mat["door"], grain=True, wall=o["wall"]); n = max(3, int(round(w / 0.17))); bw_ = w / n
        for i in range(n):
            a = u0 + i * bw_; lo_, hi_ = span_v(pts, a + 0.004); lo2, hi2 = span_v(pts, a + bw_ - 0.004)
            f.box(b, a + 0.003, a + bw_ - 0.003, v0, min(hi_, hi2), d, d + 0.022 + 0.006 * (i % 2), skip=("-2",), grain=1)
        b.part("door.ironwork", mat["metal"], wall=o["wall"])
        for hv in (v0 + 0.32, v0 + (vs - v0) * 0.82):
            f.box(b, u0 + 0.01, u0 + 0.72 * w, hv - 0.03, hv + 0.03, d + 0.028, d + 0.038, skip=("-2",))
            for k in range(5): b.cyl(f.P(u0 + 0.06 + k * 0.15 * w, hv, d + 0.038), f.P(u0 + 0.06 + k * 0.15 * w, hv, d + 0.046), 0.012, 8)
        c = f.P(u1 - 0.13, v0 + 1.02, d + 0.03); f.box(b, u1 - 0.17, u1 - 0.09, v0 + 0.96, v0 + 1.12, d + 0.028, d + 0.036, skip=("-2",))
        an = np.linspace(0, 2 * np.pi, 13)
        for k in range(12): b.cyl(c + 0.055 * (np.cos(an[k]) * f.right - (1 - np.sin(an[k])) * f.up * 0.0 + np.sin(an[k]) * f.up) - 0.03 * f.up + 0.02 * f.n, c + 0.055 * (np.cos(an[k + 1]) * f.right + np.sin(an[k + 1]) * f.up) - 0.03 * f.up + 0.02 * f.n, 0.007, 6, caps=(False, False))
    elif kind == "sectional":                             # an overhead door: ribbed sections, side tracks, a header box
        b.part("door.sections", mat["door"], wall=o["wall"]); n = max(3, int(round((v1 - v0) / 0.55))); sh = (v1 - v0) / n
        for j in range(n):
            c = v0 + j * sh; panel_field(b, f, u0 + 0.01, u1 - 0.01, c + 0.006, c + sh - 0.006, d, max(2, int(round(w / 0.9))), 1, 0.07, 0.07, 0.03, bevel=0.02, flat=False, grain=False)
        b.part("door.tracks", mat["metal"], wall=o["wall"])
        for a in (u0 + 0.0, u1 - 0.06): f.box(b, a, a + 0.06, v0, v1, d + 0.03, d + 0.09, skip=("-2",))
        f.box(b, u0, u1, v0, v0 + 0.035, d + 0.0, d + 0.05, skip=("-2",))
        for a in (u0 + w * 0.3, u0 + w * 0.7): f.box(b, a - 0.09, a + 0.09, v0 + 0.5 * sh - 0.02, v0 + 0.5 * sh + 0.02, d + 0.03, d + 0.07, skip=("-2",))
    elif kind == "blast":                                 # two sliding leaves that meet on a stepped seam
        b.part("door.plates", mat["door"], wall=o["wall"]); um = (u0 + u1) / 2
        for k, (a, c) in enumerate(((u0 + 0.01, um - 0.012), (um + 0.012, u1 - 0.01))):
            f.box(b, a, c, v0 + 0.01, v1 - 0.01, d, d + 0.04, skip=("-2",))
            panel_field(b, f, a + 0.06, c - 0.06, v0 + 0.12, v1 - 0.12, d + 0.04, 1, 3, 0.05, 0.05, 0.03, bevel=0.04, grain=False)
        b.part("door.lock", mat["metal"], wall=o["wall"]); f.box(b, um - 0.11, um + 0.11, v0 + 0.95, v0 + 1.25, d + 0.04, d + 0.11, skip=("-2",))
        f.box(b, um - 0.012, um + 0.012, v0, v1, d + 0.0, d + 0.02, skip=("-2",))
    else:                                                 # flush slab with a lever handle
        b.part("door.handle", mat["metal"], wall=o["wall"]); hu = u1 - 0.08
        b.cyl(f.P(hu, v0 + 1.02, d), f.P(hu, v0 + 1.02, d + 0.05), 0.011, 10); b.cyl(f.P(hu, v0 + 1.02, d + 0.045), f.P(hu - 0.12, v0 + 1.02, d + 0.045), 0.009, 10)
        f.box(b, hu - 0.025, hu + 0.025, v0 + 0.94, v0 + 1.1, d, d + 0.006, skip=("-2",))

def opening_parts(b, f, o, mat, emitters):
    """Everything in and round one opening: reveal, frame, glass and bars or door leaf or niche back, sill, casing, threshold."""
    kind, pts, T = o["kind"], o["pts"], o.get("T", 0.25); wall = o["wall"]; glazed = kind in ("window", "skylight", "glazed_door")
    dg = o.get("set", 0.6) * T if glazed else (o.get("depth", 0.3) if kind in ("niche", "alcove") else min(0.12, 0.5 * T))       # how deep the glass, back or leaf sits
    sp = o.get("splay", 0.0); inner = inset(pts, sp) if sp else pts.copy(); o["inner"] = inner; n = len(pts); cen = pts.mean(0)
    b.part(kind + ".reveal", mat["reveal"], wall=wall)
    for i in range(n):
        j = (i + 1) % n
        if kind in ("door", "glazed_door", "alcove") and abs(pts[i][1] - o["v0"]) < 1e-6 and abs(pts[j][1] - o["v0"]) < 1e-6 and f.name not in ("ceil",):
            b.part(kind + ".threshold", mat.get("threshold", mat["floor"]), wall=wall); b.poly([f.P(*pts[i]), f.P(*pts[j]), f.P(*inner[j], -dg), f.P(*inner[i], -dg)], f.up); b.part(kind + ".reveal", mat["reveal"], wall=wall); continue
        mid = (pts[i] + pts[j]) / 2; out2 = cen - mid; b.poly([f.P(*pts[i]), f.P(*pts[j]), f.P(*inner[j], -dg), f.P(*inner[i], -dg)], out2[0] * f.right + out2[1] * f.up)
    if kind in ("niche", "alcove"):
        b.part(kind + ".back", mat["reveal"], wall=wall); fan(b, f, inner, -dg, f.n)
        for sv in o.get("shelves", []):
            b.part(kind + ".shelf", mat["shelf"], grain=True, wall=wall); a, c = span_u(inner, sv); f.box(b, a, c, sv - 0.02, sv + 0.02, -dg, -0.01, skip=("-2",), grain=0)
    elif glazed:
        fw, fd = o.get("frame_w", 0.05), o.get("frame_d", 0.06); fo = f.P(0, 0, -dg)
        b.part(kind + ".frame", mat["frame"], grain=True, wall=wall); b.sweep(inner, [(fw, 0), (fw, fd), (0, fd)], fo, f.right, f.up, f.n, side=1, closed=True)
        gl = inset(inner, fw); b.part(kind + ".glass", mat["glass"], wall=wall); fan(b, f, gl, -dg + 0.01, f.n)
        nx, ny = o.get("bars", (0, 0))
        if nx or ny: b.part(kind + ".bars", mat["frame"], grain=True, wall=wall); bars(b, f, gl, -dg + 0.01, nx, ny, o.get("bar_w", 0.03), fd * 0.8)
        if o.get("meeting"):                               # a sash window's meeting rail
            a, c = span_u(gl, (o["v0"] + o["v1"]) / 2); f.box(b, a, c, (o["v0"] + o["v1"]) / 2 - 0.025, (o["v0"] + o["v1"]) / 2 + 0.025, -dg + 0.01, -dg + fd + 0.015, skip=("-2",), grain=0)
        if kind == "glazed_door":
            b.part(kind + ".rail", mat["frame"], grain=True, wall=wall); a, c = span_u(gl, o["v0"] + 0.2); f.box(b, a, c, o["v0"] + fw, o["v0"] + 0.24, -dg + 0.01, -dg + fd, skip=("-2",), grain=0)
            b.part(kind + ".handle", mat["metal"], wall=wall); um = (o["u0"] + o["u1"]) / 2; b.cyl(f.P(um - 0.05, o["v0"] + 1.0, -dg + fd), f.P(um - 0.05, o["v0"] + 1.0, -dg + fd + 0.06), 0.01, 8)
        P = np.array([f.P(p[0], p[1], -dg - 0.02) for p in inner]); area = 0.5 * abs(sum(inner[i][0] * inner[(i + 1) % n][1] - inner[(i + 1) % n][0] * inner[i][1] for i in range(n)))
        lo, hi = inner.min(0), inner.max(0)
        emitters.append(dict(kind="skylight" if kind == "skylight" else "window", surface=wall, poly=P.tolist(), normal=f.n.tolist(), area_m2=float(area),
                             rect=[f.P(lo[0], hi[1], -dg - 0.02).tolist(), f.P(hi[0], hi[1], -dg - 0.02).tolist(), f.P(hi[0], lo[1], -dg - 0.02).tolist(), f.P(lo[0], lo[1], -dg - 0.02).tolist()]))
        if o.get("sill") and kind == "window":
            sw = o["sill"]; b.part("window.sill", mat["sill"], grain=True, wall=wall); f.box(b, o["u0"] - 0.05, o["u1"] + 0.05, o["v0"] - 0.035, o["v0"] + 0.004, -dg + 0.0, sw, grain=0)
            if o.get("apron"): f.box(b, o["u0"] - 0.02, o["u1"] + 0.02, o["v0"] - 0.035 - 0.09, o["v0"] - 0.035, 0, 0.018, skip=("-2",), grain=0)
    elif kind == "door":
        door_leaf(b, f, o, -dg, o.get("leaf", "panel"), mat)
    cs = o.get("casing")
    if cs:
        b.part(kind + ".casing", mat["casing"], grain=True, wall=wall); pr = prof_casing(cs[0], cs[1], cs[2])
        if kind == "window" and o.get("sill") or kind in ("door", "glazed_door", "alcove"): b.sweep(np.concatenate([pts[1:], pts[:1]]), pr, f.o, f.right, f.up, f.n, side=-1, closed=False)
        else: b.sweep(pts, pr, f.o, f.right, f.up, f.n, side=-1, closed=True)
    if o.get("voussoirs"):                                 # dressed stone blocks round the opening, alternately long and short
        b.part(kind + ".stonework", mat["casing"], wall=wall); t = o["voussoirs"]; q = np.concatenate([pts[1:], pts[:1]]) if o["head"] != "circle" else np.concatenate([pts, pts[:1]])
        seg = np.linalg.norm(np.diff(q, axis=0), axis=1); tot = seg.sum(); k = max(3, int(round(tot / 0.26))); cum = np.concatenate([[0], np.cumsum(seg)])
        segn = []
        for i in range(len(seg)):
            dr = unit(q[i + 1] - q[i]); nr = np.array([dr[1], -dr[0]]); segn.append(nr if nr @ ((q[i] + q[i + 1]) / 2 - cen) > 0 else -nr)
        def at(s_): i = int(np.clip(np.searchsorted(cum, s_, side="right") - 1, 0, len(seg) - 1)); return q[i] + (s_ - cum[i]) / seg[i] * (q[i + 1] - q[i]), segn[i]
        for i in range(k):
            s0, s1 = tot * i / k + 0.004, tot * (i + 1) / k - 0.004; ln = o.get("block", 0.2) * (1.0 if i % 2 else 0.7); chain = [at(s0)]
            for j in range(1, len(seg)):
                if s0 + 1e-4 < cum[j] < s1 - 1e-4: m = (segn[j - 1] + segn[j]) / max(1 + segn[j - 1] @ segn[j], 0.3); chain.append((q[j], m))
            chain.append(at(s1))
            for (p0, n0), (p1, n1) in zip(chain[:-1], chain[1:]):
                P4 = [p0, p1, p1 + ln * n1, p0 + ln * n0]; top = [f.P(p[0], p[1], t) for p in P4]; bot = [f.P(p[0], p[1], 0) for p in P4]; mid = np.mean(bot, 0)
                b.poly(top, f.n); b.poly([bot[2], bot[3], top[3], top[2]], (bot[2] + bot[3]) / 2 - mid); b.poly([bot[0], bot[1], top[1], top[0]], (bot[0] + bot[1]) / 2 - mid)
            for (p, n_), q2, sg in ((chain[0], chain[1][0], -1), (chain[-1], chain[-2][0], 1)):
                e0, e1 = f.P(p[0], p[1], 0), f.P(*(p + ln * n_), 0); dv = (p - q2); b.poly([e0, e1, e1 + t * f.n, e0 + t * f.n], dv[0] * f.right + dv[1] * f.up)
        if kind == "niche":                                # a projecting sill stone under a niche
            f.box(b, o["u0"] - 0.1, o["u1"] + 0.1, o["v0"] - 0.09, o["v0"], 0, 0.07, skip=("-2",))

# ---------------------------------------------------------------- beams, corbels, braces, columns
def beam(b, a, c, w, h, kind="timber"):
    """A beam hanging under the line a -> c (its top), w wide and h deep. kind: timber (chamfered) | concrete | steel (an I) | rib (canted sides)."""
    a, c = np.asarray(a, float), np.asarray(c, float); ax = unit(c - a); L = np.linalg.norm(c - a); sd = unit(np.cross(Y, ax))
    if kind == "steel":
        tf, tw = max(0.012, 0.06 * h), max(0.008, 0.05 * w); m = (a + c) / 2; R = np.stack([sd, Y, ax])
        b.box(m - Y * tf / 2, R, [w / 2, tf / 2, L / 2]); b.box(m - Y * (h - tf / 2), R, [w / 2, tf / 2, L / 2]); b.box(m - Y * h / 2, R, [tw / 2, h / 2 - tf, L / 2], skip=("-1", "+1"))
        return
    ch = 0.0 if kind == "concrete" else (0.12 * min(w, h) if kind == "timber" else 0.3 * w)
    sec = [(-w / 2, 0), (w / 2, 0), (w / 2, -h + ch), (w / 2 - ch, -h), (-w / 2 + ch, -h), (-w / 2, -h + ch)] if ch else [(-w / 2, 0), (w / 2, 0), (w / 2, -h), (-w / 2, -h)]
    b.prism(sec, a, sd, Y, ax, L, grain=True)

def corbel(b, f, u, v, w, h, proj):
    """A bracket on frame f under height v at u: w wide, h high, standing proj off the wall."""
    b.prism([(0, 0), (proj, 0), (proj, -0.3 * h), (0.55 * proj, -0.78 * h), (0, -h)], f.P(u - w / 2, v), f.n, f.up, f.right, w, grain=False)

def brace(b, p0, p1, w, t):
    """A straight timber from p0 to p1 (a knee brace, a strut)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float); ax = unit(p1 - p0); sd = unit(np.cross(ax, Y) if abs(ax[1]) < 0.99 else np.array([1.0, 0, 0])); up = np.cross(sd, ax)
    b.box((p0 + p1) / 2, np.stack([ax, up, sd]), [np.linalg.norm(p1 - p0) / 2, w / 2, t / 2], grain=ax)

def column(b, x, z, y0, y1, r, kind="classical", ax=None):
    c = np.array([x, y0, z], float); h = y1 - y0
    if kind == "classical":
        b.abox(c + [-1.3 * r, 0, -1.3 * r], c + [1.3 * r, 0.3 * r, 1.3 * r])
        b.lathe([(1.25 * r, 0.3 * r)] + arc((1.12 * r, 0.42 * r), 0.12 * r, -90, 90, 4) + [(1.05 * r, 0.56 * r)] + arc((1.05 * r, 0.66 * r), 0.1 * r, -90, 90, 4) + [(r, 0.8 * r), (0.86 * r, h - 0.75 * r), (0.9 * r, h - 0.7 * r), (0.9 * r, h - 0.62 * r), (0.86 * r, h - 0.6 * r)]
                + arc((0.86 * r, h - 0.3 * r), 0.3 * r, -90, 0, 5) + [(1.2 * r, h - 0.26 * r)], c, 24, caps=(False, False))
        b.abox(c + [-1.25 * r, h - 0.26 * r, -1.25 * r], c + [1.25 * r, h, 1.25 * r])
    elif kind == "post":                                   # a square timber post with stopped chamfers on a stone pad
        b.abox(c + [-0.8 * r - 0.04, 0, -0.8 * r - 0.04], c + [0.8 * r + 0.04, 0.12, 0.8 * r + 0.04]); s = 0.8 * r; ch = 0.2 * s
        b.prism([(-s, -s + ch), (-s + ch, -s), (s - ch, -s), (s, -s + ch), (s, s - ch), (s - ch, s), (-s + ch, s), (-s, s - ch)], c + [0, 0.12, 0], [1, 0, 0], [0, 0, 1], Y, h - 0.12 - 0.3)
        b.abox(c + [-s, h - 0.3, -s], c + [s, h, s])
    elif kind == "steel":                                  # an H column on a base plate
        s = r; tf = 0.1 * s; b.abox(c + [-1.4 * s, 0, -1.4 * s], c + [1.4 * s, 0.02, 1.4 * s])
        for sg in (-1, 1): b.abox(c + [-s, 0.02, sg * s - tf * (sg > 0)], c + [s, h, sg * s + tf * (sg < 0)])
        b.abox(c + [-tf / 2, 0.02, -s + tf], c + [tf / 2, h, s - tf])
        for sg in (-1, 1): b.abox(c + [sg * 1.1 * s - 0.012, 0.02, sg * 1.1 * s - 0.012], c + [sg * 1.1 * s + 0.012, 0.05, sg * 1.1 * s + 0.012])
    elif kind == "iron":                                   # a slim cast-iron column with a base and a flared head
        b.lathe([(2.2 * r, 0), (2.2 * r, 0.06), (1.5 * r, 0.1), (1.5 * r, 0.3), (1.15 * r, 0.36), (r, 0.4), (0.9 * r, h - 0.45), (1.2 * r, h - 0.4), (1.2 * r, h - 0.36), (r, h - 0.33), (1.3 * r, h - 0.12), (2.3 * r, h - 0.03), (2.3 * r, h)], c, 16, caps=(False, False))
    else:                                                  # concrete: square with chamfered corners
        s = r; ch = 0.12 * s
        b.prism([(-s, -s + ch), (-s + ch, -s), (s - ch, -s), (s, -s + ch), (s, s - ch), (s - ch, s), (-s + ch, s), (-s, s - ch)], c, [1, 0, 0], [0, 0, 1], Y, h, grain=False)

# ---------------------------------------------------------------- services: pipes, ducts, light fittings
def ball(b, c, r, n=10): b.lathe([(r * np.sin(t), -r * np.cos(t)) for t in np.linspace(0.0, np.pi, 7)], c, n, caps=(False, False))

def pipe(b, pts, r, clip_to=None, every=1.4):
    """A pipe through the points: straight lengths, round elbows, collars by the joints, clips standing it off the nearest surface.
    clip_to: a unit vector towards the wall or ceiling the clips fix to (and how far: its length)."""
    pts = [np.asarray(p, float) for p in pts]
    for i in range(len(pts) - 1):
        a, c = pts[i], pts[i + 1]; ax = unit(c - a); L = np.linalg.norm(c - a); b.cyl(a, c, r, 12, caps=(i == 0, i == len(pts) - 2))
        for e, sg in ((a, 1), (c, -1)):
            if L > 6 * r: b.cyl(e + sg * ax * 1.4 * r, e + sg * ax * 2.6 * r, 1.35 * r, 12)
        if clip_to is not None and L > 0.5:
            cl = np.asarray(clip_to, float); k = max(1, int(L / every))
            for j in range(k):
                p = a + ax * L * (j + 0.5) / k; d = unit(cl); R = np.stack([ax, d, np.cross(ax, d)])
                b.box(p, R, [0.02, 1.25 * r, 1.25 * r]); b.box(p + cl / 2 + d * r * 0.6, R, [0.008, np.linalg.norm(cl) / 2 - 0.6 * r + 1e-4, 0.008])
    for p in pts[1:-1]: ball(b, p, 1.12 * r)

def duct(b, a, c, w, h, hang=0.0, every=1.2):
    """A rectangular duct from a to c (its centre line, level), flanged every `every` metres, on drop rods `hang` long from above."""
    a, c = np.asarray(a, float), np.asarray(c, float); ax = unit(c - a); L = np.linalg.norm(c - a); sd = unit(np.cross(Y, ax)); R = np.stack([sd, Y, ax])
    b.box((a + c) / 2, R, [w / 2, h / 2, L / 2]); k = max(1, int(round(L / every)))
    for i in range(k + 1): b.box(a + ax * L * i / k, R, [w / 2 + 0.025, h / 2 + 0.025, 0.012])
    if hang > 0:
        for i in range(0, k, 2):
            p = a + ax * L * (i + 0.5) / k; b.box(p - Y * (h / 2 + 0.015), R, [w / 2 + 0.06, 0.015, 0.02])
            for sg in (-1, 1): b.box(p + sd * sg * (w / 2 + 0.045) + Y * (hang - h / 2) / 2, R, [0.006, (hang + h / 2) / 2 + 0.015, 0.006])
    return [(a + ax * L * (i + 0.5) / k, sd) for i in range(1, k, 2)]     # where grilles can go

def grille(b, p, sd, ax, w, l, down=True):
    R = np.stack([sd, Y, ax]); b.box(p, R, [w / 2, 0.012, l / 2])
    for i in range(6): b.box(p + ax * (i - 2.5) * l / 7 - Y * 0.014, np.stack([sd, unit(Y + 0.6 * ax), unit(ax - 0.6 * Y)]), [w / 2 - 0.02, 0.003, 0.02])

def round_duct(b, a, c, r, hang=0.0, every=1.5):
    a, c = np.asarray(a, float), np.asarray(c, float); ax = unit(c - a); L = np.linalg.norm(c - a); b.cyl(a, c, r, 20); k = max(1, int(round(L / every)))
    for i in range(k + 1): p = a + ax * L * i / k; b.cyl(p - ax * 0.015, p + ax * 0.015, r + 0.02, 20)
    if hang > 0:
        for i in range(k): p = a + ax * L * (i + 0.5) / k; b.box(p + Y * (hang + r) / 2, np.stack([unit(np.cross(Y, ax)), Y, ax]), [0.006, (hang + r) / 2, 0.012])

def fitting(b, c, ax, L, w, drop, mat, emitters, colour="cool"):
    """A linear light: a housing L long hanging `drop` below the point c on the ceiling, a glowing diffuser underneath."""
    c = np.asarray(c, float); ax = unit(ax); sd = unit(np.cross(Y, ax)); R = np.stack([sd, Y, ax]); hh = 0.05
    b.part("light.housing", mat["metal"], piece_of="light"); b.box(c - Y * (drop + hh / 2), R, [w / 2, hh / 2, L / 2])
    if drop > 0.02:
        for sg in (-1, 1): b.box(c + ax * sg * L * 0.38 - Y * drop / 2, R, [0.004, drop / 2, 0.004])
    q = c - Y * (drop + hh + 0.001); P = [q - sd * w * 0.42 - ax * L * 0.48, q + sd * w * 0.42 - ax * L * 0.48, q + sd * w * 0.42 + ax * L * 0.48, q - sd * w * 0.42 + ax * L * 0.48]
    b.part("light.diffuser", mat["emitter"], piece_of="light"); b.poly(P, -Y)
    emitters.append(dict(kind="fitting", surface="ceil", poly=[p.tolist() for p in P], rect=[p.tolist() for p in P], normal=[0, -1.0, 0], area_m2=float(L * 0.96 * w * 0.84), colour=colour))

# ---------------------------------------------------------------- floor pieces
def platform(b, c, half, h, mat, steps_side=None, round_=False, lip=0.035, inset_=0.03, step_w=1.4):
    """A raised floor with a proper edge: a top slab that oversails a recessed body by `lip`, and steps up on one side.
    c = centre on the floor (x, z), half = half sizes (x, z). steps_side: (dx, dz) unit, the side the steps go on."""
    cx, cz = c; hx, hz = half; tt = min(0.05, h * 0.4)
    if round_:
        r = min(hx, hz); b.part("platform.body", mat["platform"]); b.lathe([(r - inset_, 0), (r - inset_, h - tt)], [cx, 0, cz], 40, caps=(False, False))
        b.part("platform.top", mat["platform_top"]); b.lathe([(r - inset_, h - tt), (r + lip - 0.01, h - tt), (r + lip, h - tt + 0.01), (r + lip, h - 0.01), (r + lip - 0.01, h)], [cx, 0, cz], 40, caps=(False, True))
        n = int(np.ceil(h / 0.19)) - 1
        for k in range(n):                                 # ring steps all round
            y1 = h * (k + 1) / (n + 1); rr = r + lip + 0.3 * (n - k)
            b.part("platform.step", mat["platform_top"]); b.lathe([(rr - 0.02, 0 if k == 0 else h * k / (n + 1)), (rr - 0.02, y1 - 0.04), (rr, y1 - 0.04), (rr, y1 - 0.008), (rr - 0.008, y1)], [cx, 0, cz], 40, caps=(False, False))
            b.poly([[cx + rr * np.cos(t), y1, cz + rr * np.sin(t)] for t in np.linspace(0, 2 * np.pi, 41)[:-1]], Y)
        return
    b.part("platform.body", mat["platform"]); b.abox([cx - hx + inset_, 0, cz - hz + inset_], [cx + hx - inset_, h - tt, cz + hz - inset_], skip=("-1", "+1"))
    b.part("platform.top", mat["platform_top"]); ch = 0.008; x0, x1, z0, z1 = cx - hx - lip, cx + hx + lip, cz - hz - lip, cz + hz + lip
    b.abox([x0, h - tt, z0], [x1, h - ch, z1], skip=("+1",)); b.abox([x0 + ch, h - ch, z0 + ch], [x1 - ch, h, z1 - ch], skip=("-1",))
    for (p, q, out) in (((x0, z0), (x1, z0), (0, -1)), ((x1, z0), (x1, z1), (1, 0)), ((x1, z1), (x0, z1), (0, 1)), ((x0, z1), (x0, z0), (-1, 0))):
        e = np.array([out[0], 0, out[1]], float) * ch
        b.poly([[p[0], h - ch, p[1]], [q[0], h - ch, q[1]], np.array([q[0], h, q[1]]) - e - np.array([np.sign(q[0] - cx) * ch * (out[0] == 0), 0, np.sign(q[1] - cz) * ch * (out[1] == 0)]), np.array([p[0], h, p[1]]) - e - np.array([np.sign(p[0] - cx) * ch * (out[0] == 0), 0, np.sign(p[1] - cz) * ch * (out[1] == 0)])], [out[0], 0.5, out[1]])
    n = int(np.ceil(h / 0.19)) - 1
    if steps_side is not None and n > 0:
        dx, dz = steps_side; d3 = np.array([dx, 0, dz], float); s3 = np.array([-dz, 0, dx], float); edge = np.array([cx, 0, cz]) + d3 * ((hx if dx else hz) + lip); sw = min(step_w, 2 * (hz if dx else hx) - 0.4) / 2
        for k in range(n):
            y1 = h * (k + 1) / (n + 1); y0 = 0.0; far = 0.3 * (n - k); R = np.stack([s3, Y, d3])
            b.part("platform.step", mat["platform"]); b.box(edge + d3 * (far - 0.02) / 2 + Y * (y0 + y1 - 0.04) / 2, R, [sw - 0.02, (y1 - 0.04 - y0) / 2, (far - 0.02) / 2], skip=("-1", "+1", "-2"))
            b.part("platform.tread", mat["platform_top"]); b.box(edge + d3 * far / 2 + Y * (y1 - 0.02), R, [sw, 0.02, far / 2], skip=("-2",))

def hearth(b, f, u0, u1, proj, mat, h=0.06):
    b.part("hearth", mat["stone"], wall=f.name); ch = 0.015
    b.prism([(0, 0), (proj, 0), (proj, h - ch), (proj - ch, h), (0, h)], f.P(u0, 0), f.n, f.up, f.right, u1 - u0, grain=False)

def shelving(b, f, u0, u1, v0, v1, depth, mat, d0=0.0):
    """Built-in shelves on frame f from depth d0 out to d0 + depth: plinth, uprights, shelves, a back, a small cornice."""
    b.part("shelving.carcass", mat["shelf"], grain=True, wall=f.name); n = max(1, int(np.ceil((u1 - u0) / 0.95))); bw = (u1 - u0) / n; t = 0.03
    f.box(b, u0, u1, v0, v0 + 0.1, d0, d0 + depth - 0.02, skip=("-2",), grain=0)
    for i in range(n + 1): a = u0 + i * bw - (t if i == n else 0 if i == 0 else t / 2); f.box(b, a, a + t, v0 + 0.1, v1 - 0.06, d0, d0 + depth, skip=("-2",), grain=1)
    f.box(b, u0 - 0.02, u1 + 0.02, v1 - 0.06, v1, d0, d0 + depth + 0.025, skip=("-2",), grain=0)
    b.part("shelving.back", mat["shelf"], grain=True, wall=f.name); b.poly([f.P(u0, v0 + 0.1, d0 + 0.012), f.P(u1, v0 + 0.1, d0 + 0.012), f.P(u1, v1 - 0.06, d0 + 0.012), f.P(u0, v1 - 0.06, d0 + 0.012)], f.n, f.up)
    b.part("shelving.shelves", mat["shelf"], grain=True, wall=f.name); k = max(2, int(round((v1 - v0 - 0.16) / 0.36)))
    for j in range(k + 1):
        v = v0 + 0.1 + (v1 - v0 - 0.16 - 0.025) * j / k
        for i in range(n): f.box(b, u0 + i * bw + t / 2, u0 + (i + 1) * bw - t / 2, v, v + 0.025, d0, d0 + depth - 0.015, skip=("-2", "-0", "+0"), grain=0)

def rose(b, x, z, H, r):
    """A plaster ceiling rose."""
    b.lathe([(0.0, -0.055), (0.12 * r, -0.05), (0.2 * r, -0.03), (0.36 * r, -0.036), (0.42 * r, -0.02), (0.7 * r, -0.024), (0.76 * r, -0.012), (0.94 * r, -0.016), (r, 0.0)], [x, H, z], 32, caps=(False, False))
