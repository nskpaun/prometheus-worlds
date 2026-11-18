"""Which painted view a surface point takes its paint from: the one rule, used by consistent.py (which lays the paint)
and by evals/density.py and evals/reference.py (which score it), so the quick scores cannot drift from the build.

A painted view is (camera, depth, surface ids, silhouette edges, the band round shapes, the bands round moving things):
views() makes that list from a layout. owners() ranks the views for every point:
  - per surface the views are ranked by how much of it they show (rank(): 0 = most; under 400 px = never);
  - a shape seen side-on (cosine under C.GRAZE) or through the band round another shape's outline is taken only where
    no other view sees the point; a view behind a shape's surface never;
  - next to a thing that moves away the picture holds its spill, its edge and its shadow. Within LIFT_NEAR px of its
    outline the picture is never the room's paint. From there out to LIFT_HALO px it is taken only where no other
    view sees the point (WORLD_GAPS=0: never, as it was before): a bookshelf behind such a thing was otherwise left to
    be filled with a smear of its own nearest paint;
  - the point must really be seen: same surface at that pixel (or at one of its 8 neighbours) and nothing in front.
A world painted to a plan (detail.py) has views left out (None in the list: never a source) and detail views after the
ring's (Views.detail says how many). A detail view is a closer picture of walls and floor a walker comes near, begun
from the ring's paint. It does not compete by rank. After the ring's views have been ranked, each detail view in turn
takes the points
  - that lie inside its picture and not in the outer DETAIL_EDGE of it (there it is faded into the paint around it),
  - that it really sees, within DETAIL_HEAD of head-on, clear of the band round a shape's outline and round a moving thing
    (and of what the picture was not begun from paint: the caller adds that to the view's band, detail.blank),
  - that are not ceiling, sky, an outdoor backdrop or a thing that moves away,
  - and on which it lays over DETAIL_GAIN times the pixels of the owner so far: it never makes anything coarser.
The floor round a moving thing is painted with its shadow. shade() finds, from the paint itself, which of that floor is
darker than the floor further off; only that is thrown away (WORLD_GAPS=0: all the floor within the thing's height of
it, between 0.4 and 2 m, which was a sixth of everything a walker saw close up in a room with a large globe).
"""
import os, numpy as np
from scipy import ndimage
OFFS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
LIFT_HALO = 81      # px, odd: the band round a moving thing (40 px) where a picture is the last choice for room paint
LIFT_NEAR = 33      # px, odd: the band (16 px, as round any shape) where it is no choice at all
NEVER, NONE = 1e6, 1e5
GAPS = os.environ.get("WORLD_GAPS", "1") != "0"      # the rules that leave less surface without paint; 0 = as before them
SHADE_CELL = 0.05   # metres: the grid the floor round a moving thing is looked at on
SHADE_RING = 1.0    # metres of floor beyond the band that say how bright the floor is without a shadow
SHADE_DROP = 0.82   # floor darker than this share of that brightness is shadow

DETAIL_EDGE = 0.05  # share of a detail picture's width and height, at each edge, that it never owns
DETAIL_HEAD = 0.5   # least cosine between a surface's normal and the direction to a detail view's camera
DETAIL_GAIN = 1.2   # a detail view takes a point only where it lays this many times the owner's pixels per metre

class Views(list):
    """The painted views, by view number. .detail: how many at the end are detail views. .shadow: shade()'s answer
    where the caller keeps it with them (evals/density.py)."""
    detail = 0; shadow = None

def big(m, px): return ndimage.maximum_filter(m, int(px), mode="nearest")

def normal(p, P):
    """Unit normals of a box or cylinder primitive at points P on its surface (a box's: along the face's axis, either way)."""
    o = P - np.array(p["c"], float); n = np.zeros(P.shape)
    if p["shape"] == "cyl":
        cap = np.abs(o[:, 1]) > p["hy"] - 1e-4
        n[cap, 1] = 1; n[~cap] = o[~cap] * [1, 0, 1]
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    cs, sn = np.cos(p.get("yaw", 0.0)), np.sin(p.get("yaw", 0.0))
    l = np.stack([cs * o[:, 0] - sn * o[:, 2], o[:, 1], sn * o[:, 0] + cs * o[:, 2]], 1)
    k = np.argmax(np.abs(l) / np.array(p["h"], float), 1); e = np.eye(3)[k]       # the face the point lies on
    return np.stack([cs * e[:, 0] + sn * e[:, 2], e[:, 1], -sn * e[:, 0] + cs * e[:, 2]], 1)

def off(p, P):
    """How far points P are from the surface of a box or cylinder primitive (roughly: enough to tell which one a point is on)."""
    o = P - np.array(p["c"], float)
    if p["shape"] == "cyl": return np.abs(np.maximum(np.hypot(o[:, 0], o[:, 2]) - p["r"], np.abs(o[:, 1]) - p["hy"]))
    cs, sn = np.cos(p.get("yaw", 0.0)), np.sin(p.get("yaw", 0.0))
    l = np.stack([cs * o[:, 0] - sn * o[:, 2], o[:, 1], sn * o[:, 0] + cs * o[:, 2]], 1)
    return np.abs((np.abs(l) - np.array(p["h"], float)).max(1))

def normals(C, P, sid, n=None):
    """A normal for every point: n where it has one (a ray-cast shape's, facing whoever looked), a room face's axis, and
    for a point on an object made of boxes and cylinders the normal of the one it lies on. Only a shape's has a side:
    the others may point either way."""
    out = np.zeros(P.shape) if n is None else np.array(n, float)
    for f in range(6): out[sid == f, f % 3] = 1
    need = (sid >= 6) & ~np.abs(out).sum(-1).astype(bool)
    for s in np.unique(sid[need]):
        prims = [p for p in C.OBJ_PRIMS[s - 6] if p["shape"] != "mesh"]
        if not prims: continue
        m = need & (sid == s); Q = P[m]; best = np.full(len(Q), np.inf); nn = np.zeros(Q.shape)
        for p in prims:
            d = off(p, Q); k = d < best; best[k] = d[k]; nn[k] = normal(p, Q[k])
        out[m] = nn
    return out

def laid(B, cam, P, n):
    """Pixels a picture from cam lays on a square metre of surface at points P with normals n, but for the square of
    its focal length (which every painted view shares): |n.v| / z^3, v from the eye to the point and z its depth."""
    eye, r, u, f = B.basis(cam); v = P - eye
    return np.abs((v * n).sum(-1)) / np.maximum(v @ f, 1e-6) ** 3

def rank(count):
    """surface x view pixel counts -> each view's place for each surface (0 = shows most of it; NEVER under 400 px)."""
    r = np.argsort(np.argsort(-count, axis=1), axis=1).astype(float); r[count < 400] = NEVER
    return r

def view(cam, t, sid, halo, lifted):
    """One painted view as owners() wants it. halo: the band round shapes (C.halo). lifted: the objects that move away."""
    edge = np.zeros(sid.shape, bool)
    for dy, dx in OFFS: edge |= sid != np.roll(sid, (dy, dx), (0, 1))
    on = np.isin(sid, [6 + oi for oi in lifted]) if len(lifted) else None
    # the objects' own pixels count too: a point just past an outline rounds onto the object, then borrows from its edge
    return (cam, t, sid, edge, halo, None if on is None else big(on, LIFT_HALO), None if on is None else big(on, LIFT_NEAR))

def shade_bands(C, lifted):
    """[(lo, hi, metres around)] for the moving things that stand on or just above the floor: where their shadow may lie."""
    return [(lo, hi, float(np.clip(hi[1], 0.4, 2.0))) for lo, hi in (C.bounds(C.OBJ_PRIMS[oi]) for oi in lifted) if lo[1] < 0.6]

def near(P, lo, hi):
    """Level distance from points P to the box lo..hi."""
    return np.hypot(np.maximum(np.maximum(lo[0] - P[..., 0], P[..., 0] - hi[0]), 0), np.maximum(np.maximum(lo[2] - P[..., 2], P[..., 2] - hi[2]), 0))

def owners(C, src, rk, P, sid, n, lifted=(), shadow=None, spot=False, every=None):
    """For points P (any shape) on surfaces sid with normals n (used on shapes only; may be None when there are none):
    own   the painted view each takes its paint from, -1 where none
    yi, xi the pixel of that view to read (moved to a neighbour where the pixel itself shows another surface)
    at    with spot: the exact place (x, y) between that view's pixels, unless moved
    lost  points a painted view shows only next to a moving thing, or that lie in its shadow
    lifted: the things taken out of the room (empty: they are in it and nothing is avoided). shadow: shade()'s answer,
    None for the whole band. every: a dict that is given, per ring view, its score at each point (the lowest wins;
    NEVER where it does not show the point), for a planner that asks what happens when a view is left out."""
    B = C.B; H, W = src[0][2].shape; N = len(src); NS = rk.shape[0]; nd = getattr(src, "detail", 0)
    mesh = np.isin(np.arange(NS), [6 + oi for oi in C.SHAPES]); on = mesh[sid] if mesh.any() else None
    best = np.full(sid.shape, NONE); own = np.full(sid.shape, -1); oy = np.zeros(sid.shape, int); ox = np.zeros(sid.shape, int)
    at = np.zeros(sid.shape + (2,), np.float32) if spot else None
    lost = np.zeros(sid.shape, bool)
    # the views most likely to own these points first: once a point has its best-ranked source, lower-ranked views skip
    # it, which is most of the work saved
    def shown(v, want, xi, yi, dist):
        """Where view v really shows the wanted points, and the pixel of it to read (yi, xi are changed in place)."""
        cam, st, ssid, sedge = src[v][:4]
        # room faces away from any silhouette in the source: same surface, depth within a tolerance that grows with distance
        plain = want & (sid < 6) & ~sedge[yi, xi]
        ok = plain & (ssid[yi, xi] == sid) & (np.abs(st[yi, xi] - dist) < np.maximum(0.05, 0.01 * dist))
        # objects and silhouettes: exact shadow ray (comparing against the source's depth at a rounded pixel gave staircase
        # edges), then take the colour from the nearest source pixel that shows the same surface
        ex = np.nonzero(want & ~plain); moved = None
        if len(ex[0]):
            ey, ex_x, s = yi[ex], xi[ex], sid[ex]
            vis = B.visible(cam[0], P[ex]); miss = vis & (ssid[ey, ex_x] != s)
            for dy, dx in OFFS:
                if not miss.any(): break
                # clip: entries moved by an earlier offset can already sit on the picture's last row or column
                m = miss & (ssid[np.clip(ey + dy, 0, H - 1), np.clip(ex_x + dx, 0, W - 1)] == s)
                ey[m] += dy; ex_x[m] += dx; miss &= ~m
            ok[ex] = vis & ~miss
            moved = (ey != yi[ex]) | (ex_x != xi[ex])
            yi[ex] = ey; xi[ex] = ex_x
        return ok, ex, moved
    def place(v, ok, ex, moved, px, py, xi, yi):
        sp = np.stack([np.clip(np.nan_to_num(px), 0, W - 1), np.clip(np.nan_to_num(py), 0, H - 1)], -1)
        if len(ex[0]): e = sp[ex]; e[moved] = np.stack([xi[ex], yi[ex]], -1)[moved]; sp[ex] = e
        at[ok] = sp[ok]
    for v in np.argsort((np.minimum(rk, N) * np.bincount(sid.ravel(), minlength=NS)[:, None]).sum(0)):
        if v >= N - nd or src[v] is None: continue                   # a detail view (below); a view that was not painted
        cam, st, ssid, sedge, shalo, sfar, snear = src[v]
        score = rk[sid, v]
        if on is not None and on.any():
            # a picture that sees a curved shape side-on smears it, and its outline rarely matches the shape to the pixel
            to = np.array(cam[0]) - P[on]; cos = (to * n[on]).sum(1) / np.maximum(np.linalg.norm(to, axis=1), 1e-9)
            score[on] += N * (cos < C.GRAZE)
            # a view behind the surface is no source at all: through a gap or a single-sheet part it would lend the other side's paint
            score[on] = np.where(cos <= 0, NEVER, score[on])
        with np.errstate(all="ignore"):      # a point level with the painted camera has no place in its picture: dropped just below
            px, py, dist, z = B.project(cam, P)
            xi, yi = np.round(px).astype(int), np.round(py).astype(int)
        inside = (z > 0.05) & (xi >= 1) & (xi < W - 1) & (yi >= 1) & (yi < H - 1)
        xi, yi = np.clip(xi, 0, W - 1), np.clip(yi, 0, H - 1)
        # the band around a shape's outline often holds the shape's own paint: the wall or floor seen there is taken
        # from another view where one exists, or the car's outline ends up stamped on the wall behind it
        if on is not None: score = score + N * shalo[yi, xi]
        if len(lifted):
            by = sfar[yi, xi]; lost |= by & inside
            if GAPS: score = np.where(snear[yi, xi], NEVER, score + 4 * N * by)
            else: score = np.where(by, NEVER, score)
        want = ((score < best) if every is None else (score < NONE)) & inside      # only where this view would become the source
        ok, ex, moved = shown(v, want, xi, yi, dist)
        if every is not None: every[int(v)] = np.where(ok, score, NEVER); ok &= score < best
        best = np.where(ok, score, best); own[ok] = v; oy[ok] = yi[ok]; ox[ok] = xi[ok]
        if spot: place(v, ok, ex, moved, px, py, xi, yi)
    if nd:
        nf = normals(C, P, sid, n); dens = np.zeros(sid.shape)
        for v in np.unique(own[own >= 0]): m = own == v; dens[m] = laid(B, src[v][0], P[m], nf[m])
        free = (sid != 4) & ~np.isin(sid, [6 + oi for oi in C.LIFT] + (list(C.BACKDROP) if C.OUT else []))
        e0, e1 = DETAIL_EDGE * W, DETAIL_EDGE * H
        for v in range(N - nd, N):
            cam, _, _, _, shalo, sfar, _ = src[v]
            to = np.array(cam[0]) - P; cos = (to * nf).sum(-1) / np.maximum(np.linalg.norm(to, axis=-1), 1e-9)
            d = laid(B, cam, P, nf)
            want = free & (np.where(mesh[sid], cos, np.abs(cos)) >= DETAIL_HEAD) & (d > DETAIL_GAIN ** 2 * dens)
            if not want.any(): continue
            with np.errstate(all="ignore"):
                px, py, dist, z = B.project(cam, P)
                xi, yi = np.round(px).astype(int), np.round(py).astype(int)
            want &= (z > 0.05) & (px >= e0) & (px <= W - 1 - e0) & (py >= e1) & (py <= H - 1 - e1)
            xi, yi = np.clip(xi, 0, W - 1), np.clip(yi, 0, H - 1)
            want &= ~shalo[yi, xi]
            if sfar is not None: want &= ~sfar[yi, xi]
            ok, ex, moved = shown(v, want, xi, yi, dist)
            own[ok] = v; oy[ok] = yi[ok]; ox[ok] = xi[ok]; dens[ok] = d[ok]
            if spot: place(v, ok, ex, moved, px, py, xi, yi)
    if len(lifted):      # the floor round a moving thing holds its painted shadow and is filled in instead
        by = np.zeros(sid.shape, bool)
        if shadow is not None: by = (sid == 1) & shadow(P)
        else:
            for lo, hi, b in shade_bands(C, lifted): by |= (sid == 1) & (near(P, lo, hi) < b)
        own[by] = -1; lost |= by
    return own, oy, ox, at, lost

def shade(C, src, rk, lifted, paint):
    """The floor that lies in a moving thing's painted shadow, as a function of points -> bool; None when no moving
    thing stands on the floor (or with WORLD_GAPS=0: owners() then drops the whole band).
    paint: the painted pictures at the working size, one array (H, W, 3) per view, or a function view -> array.
    Per thing: the floor in its band and SHADE_RING beyond is laid out on a grid and given the paint it would take.
    The ring's median brightness is the floor's own. In the band, floor that is darker than SHADE_DROP of it (after a
    15 cm blur, so a dark floorboard seam is not a shadow) is shadow if it is joined to the thing's footprint, with
    10 cm more around. Too little paint in the ring to tell (under 200 cells): the whole band, as before."""
    bands = shade_bands(C, lifted)
    if not GAPS or not bands: return None
    get = paint if callable(paint) else (lambda v: paint[v]); maps = []
    for lo, hi, b in bands:
        reach = b + SHADE_RING; lo_r, hi_r = np.array(C.B.ROOM[0], float), np.array(C.B.ROOM[1], float)
        x0, x1 = max(lo[0] - reach, lo_r[0]), min(hi[0] + reach, hi_r[0]); z0, z1 = max(lo[2] - reach, lo_r[2]), min(hi[2] + reach, hi_r[2])
        nx, nz = max(1, int(np.ceil((x1 - x0) / SHADE_CELL))), max(1, int(np.ceil((z1 - z0) / SHADE_CELL)))
        gx, gz = np.meshgrid(x0 + (np.arange(nx) + 0.5) * SHADE_CELL, z0 + (np.arange(nz) + 0.5) * SHADE_CELL, indexing="ij")
        P = np.stack([gx, np.zeros_like(gx), gz], -1); sid = np.ones(gx.shape, int); up = np.zeros(P.shape); up[..., 1] = 1
        own, oy, ox, _, _ = owners(C, src, rk, P, sid, up, lifted, shadow=lambda Q: np.zeros(Q.shape[:-1], bool))
        lum = np.zeros(gx.shape)
        for v in np.unique(own[own >= 0]):
            m = own == v; lum[m] = get(v)[oy[m], ox[m]].astype(float) @ [0.299, 0.587, 0.114]
        have = own >= 0; d = near(P, lo, hi); band = d < b
        other = np.zeros(gx.shape, bool)
        for lo2, hi2, b2 in bands: other |= near(P, lo2, hi2) < b2
        ring = have & ~other
        if ring.sum() < 200: maps.append((x0, z0, band, lo, hi, b)); continue
        level = float(np.median(lum[ring])); k = 0.15 / SHADE_CELL
        soft = ndimage.gaussian_filter(lum * have, k, mode="constant") / np.maximum(ndimage.gaussian_filter(have.astype(float), k, mode="constant"), 1e-6)
        dark = band & (~have | (soft < SHADE_DROP * level))
        # only what is joined to the thing (its footprint and the rim the pictures cannot be trusted on) is its shadow
        lab, _ = ndimage.label(dark | (d < 0.1)); seeds = np.unique(lab[d < 0.1]); joined = np.isin(lab, seeds[seeds > 0])
        dark = ndimage.binary_dilation(dark & joined, iterations=2) & band
        maps.append((x0, z0, dark, lo, hi, b))
    def shadow(P):
        out = np.zeros(P.shape[:-1], bool)
        for x0, z0, dark, lo, hi, b in maps:
            ix = ((P[..., 0] - x0) / SHADE_CELL).astype(int); iz = ((P[..., 2] - z0) / SHADE_CELL).astype(int)
            ok = (ix >= 0) & (ix < dark.shape[0]) & (iz >= 0) & (iz < dark.shape[1]) & (near(P, lo, hi) < b)
            out |= ok & dark[np.clip(ix, 0, dark.shape[0] - 1), np.clip(iz, 0, dark.shape[1] - 1)]
        return out
    shadow.maps = maps
    return shadow

COVER_WANT = 4      # training views that should show every surface a walker can come close to
COVER_NEAR = 5.0    # metres: a view from further off than this does not count (nor does a walker look that far)
COVER_MOST = 8      # views added at most: each takes training steps from the rest (12 cost the garage's walls 8% of their sharpness)

def sees(B, cam, Q, reach=COVER_NEAR):
    """Which of the points Q a camera (eye, look) at the working size shows: in its picture, within reach, nothing in front."""
    with np.errstate(all="ignore"): px, py, dist, z = B.project(cam, Q)
    ok = (z > 0.05) & (px >= 1) & (px < B.W - 1) & (py >= 1) & (py < B.H - 1) & (dist < reach)
    if ok.any(): ok[ok] = B.visible(cam[0], Q[ok])
    return ok

def walker(C, lifted=(), normals=False):
    """What someone walking the room comes close to. He stands at every free spot of a 1 m grid at eye height and looks
    eight ways, level and 35 degrees down: -> (those views, the surface points he sees within COVER_NEAR on and over
    the ground to walk on, one per 20 cm cell and not the ceiling, their surface ids, their normals or None).
    lifted: the things the room is walked without."""
    B = C.B; keep = [j for j, oi in enumerate(C.PRIM_OBJ) if oi not in lifted]; ids = [C.PRIM_OBJ[j] for j in keep]
    full, was = B.PRIMS, (B.W, B.H)
    clear = lambda e: 0.5 < e[0] < C.RW - 0.5 and 0.5 < e[2] < C.RD - 0.5 and not any(B.inside(e, p, 0.3) for p in C.KEEP_OUT)
    spots = [e for e in (np.array([x, C.EYE, z]) for x in np.arange(0.75, C.RW - 0.5, 1.0) for z in np.arange(0.75, C.RD - 0.5, 1.0)) if clear(e)]
    walk = []
    for e in spots:
        for i in range(8):
            a = np.pi * i / 4
            for down in (0.0, 35.0):
                p = np.radians(down); walk.append((tuple(e), tuple(e + [np.cos(a) * np.cos(p), -np.sin(p), np.sin(a) * np.cos(p)])))
    B.PRIMS = [full[j] for j in keep]; B.W, B.H = 64, 48; pts = [np.zeros((0, 7))]
    try:
        for cam in walk:
            if normals: t, P, sid, nrm = B.trace(cam, ids, True)
            else: (t, P, sid), nrm = B.trace(cam, ids), np.zeros((B.H, B.W, 3))
            m = (t < COVER_NEAR) & (sid != 4) & (P[..., 0] > 0) & (P[..., 0] < C.RW) & (P[..., 2] > 0) & (P[..., 2] < C.RD)      # outdoors: the ground to walk on, not the land beyond
            pts.append(np.concatenate([P[m], sid[m][:, None], nrm[m]], 1))
    finally: B.PRIMS = full; B.W, B.H = was
    pts = np.concatenate(pts); _, first = np.unique(np.concatenate([np.floor(pts[:, :3] / 0.2), pts[:, 3:4]], 1).astype(np.int64), axis=0, return_index=True)
    pts = pts[np.sort(first)]
    return walk, pts[:, :3], pts[:, 3].astype(int), pts[:, 4:] if normals else None

def cover(C, cams, lifted=()):
    """Training views to add so that no surface a walker comes close to is left to two or three views: [(eye, look)].
    They cost no painting. Surfaces that few training views show (the floor a platform hides from every ring) are
    covered thinly by the trainer whatever they are filled with, and the background shows through from close by.
    What walker() sees should be shown by COVER_WANT of `cams` from within COVER_NEAR. While points fall short, the
    walker's downward view that shows the most missing looks is added (each once), up to COVER_MOST and until one adds
    under 3% of what was missing at the start. (Counting views at any distance was tried: it mended the floor but not
    the cars' sides.) Measured on the garage: its floor beside the platform was shown by 3 views at any distance
    (median; 40% of it by 2 or fewer) against 8 for the rest of the floor, and the trainer's background showed there.
    lifted: the things the trained room is without. Nothing with WORLD_GAPS=0."""
    if not GAPS: return []
    B = C.B; walk, Q, _, _ = walker(C, lifted)
    if not len(Q): return []
    full = B.PRIMS; B.PRIMS = [full[j] for j, oi in enumerate(C.PRIM_OBJ) if oi not in lifted]
    try:
        need = np.full(len(Q), float(COVER_WANT))
        for cam in cams: need -= sees(B, cam, Q)
        need = np.maximum(need, 0); short = need > 0; start = need.sum(); out = []
        if not short.any(): return []
        Qs = Q[short]; need = need[short]
        shows = np.stack([sees(B, cam, Qs) for cam in walk[1::2]])       # the views looking down show the ground at the walker's feet too
        while len(out) < COVER_MOST:
            gain = shows @ need; k = int(gain.argmax())
            if gain[k] < 0.03 * start: break
            out.append(walk[1::2][k]); need = np.maximum(need - shows[k], 0); shows[k] = False      # each view once
        return out
    finally: B.PRIMS = full

def lookdown(C, views, rk, n, lifted=()):
    """Where the n views painted after the ring should stand and look, chosen by what the ring leaves without paint:
    [(eye, look)], fewer than n (or none) when nothing more is gained. views, rk: the ring's views (view(), rank()).
    The ring looks across the room, so it never sees the floor that a thing in the middle hides from the far side and
    that the near cameras look over. Candidates stand on a 1 m grid at the height of the views they replace
    (min(RH - 0.3, EYE + 0.6)), clear of things, looking eight ways, 35 and 55 degrees down. One by one the candidate is
    taken that shows the most of walker()'s points that no view owns yet (a shape's surface only from in front, not
    side-on). A view that would have something nearer than 0.6 m across a twentieth of its picture is passed over."""
    B = C.B; _, Q, qs, qn = walker(C, lifted, True)
    if not len(Q) or n <= 0: return []
    un = owners(C, views, rk, Q, qs, qn, lifted, shadow=lambda P: np.zeros(P.shape[:-1], bool))[0] < 0      # before any paint: no shadow is known
    Q, qs, qn = Q[un], qs[un], qn[un]; on = np.isin(qs, [6 + oi for oi in C.SHAPES])
    h = min(C.RH - 0.3, C.EYE + 0.6)
    clear = lambda e: 0.5 < e[0] < C.RW - 0.5 and 0.5 < e[2] < C.RD - 0.5 and not any(B.inside(e, p, 0.3) for p in C.KEEP_OUT) and not C.crowded(e)
    cand = [((x, h, z), (x + np.cos(a) * np.cos(p), h - np.sin(p), z + np.sin(a) * np.cos(p))) for x in np.arange(0.75, C.RW - 0.5, 1.0) for z in np.arange(0.75, C.RD - 0.5, 1.0)
            if clear((x, h, z)) for a in np.pi * np.arange(8) / 4 for p in np.radians([35.0, 55.0])]
    if not cand or not len(Q): return []
    def shows(cam):
        ok = sees(B, cam, Q, np.inf)
        if on.any():
            to = np.array(cam[0]) - Q[on]; ok[on] &= (to * qn[on]).sum(1) >= C.GRAZE * np.linalg.norm(to, axis=1)
        return ok
    show = np.stack([shows(cam) for cam in cand]); left = np.ones(len(Q), bool); out = []; was = (B.W, B.H)
    while len(out) < n:
        gain = show @ left.astype(float); k = int(gain.argmax())
        if gain[k] < max(1, 0.01 * len(Q)): break
        B.W, B.H = 64, 48
        try: t = B.trace(cand[k])[0]
        finally: B.W, B.H = was
        if np.percentile(t, 5) >= 0.6: out.append(cand[k]); left &= ~show[k]
        show[k] = False
    return out
