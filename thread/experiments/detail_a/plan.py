"""Plan a detail pass: a few extra close, square-on views of the walls, floor and large objects, to be painted after the
ring from the paint already there, each owning what it sees. Planning only: nothing is painted, no GPU.
  plan.py WORLD --budget N [--target R[,R..]] [--paint WxH] [--ring-paint WxH] [--no-down] [--table]
  (run from thread/ with .venv/bin/python; WORLD: a folder name under worlds/ or a path; needs boxes.json, depth/, gen/)
Prints one JSON object: "views", at most N, each with "cam" [eye, look] in room metres exactly as chisel_ring.py's
B.CAMS (and gen_c/cams.json), "fov" (degrees side to side, blockout's), "size", what it would own and "paint_seconds";
and "score": the geometry score (score.py, evals/density.py's maths) before and after.
How: every wall, the floor and each side of each large object is tiled with square-on cameras at the distance where the
picture gives R painted pixels per screen pixel to a walker (R = --target, default 0.5): standing 0.8 m away for a wall or
an object, looking 35 degrees down from eye height for the floor. Tiles are then picked one at a time by how much soft
screen area each removes, summed over the planner's own walker cameras (probes(): a grid of standing spots, spots along
the walls, spots by the objects), not over the 24 eval cameras the score is reported on.
"Soft area removed" per screen pixel is the rise in min(r, 1). Several targets (--target 0.5,0.25) offer tiles at
each distance; the picks then go to the furthest, widest ones (doubling a low r over a large area gains the most).
--paint: the detail pictures' size (default 768x576). --ring-paint, --no-down: predict as if the ring had been painted
at another size, or without its downward views; these are how the extra pictures are paid for (NOTES.md).
--table: the score for budgets 2, 4, 6, 8 in one run (the picks are nested: the first N of a longer plan).
"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score as Sc
K = Sc.K
NEAR, PITCH = 0.8, 35.0       # the walker: metres from a wall or object; degrees looked down at the floor (as evals/cams.py)
SEC, REF = 65.0, 768 * 576    # measured: seconds to paint one 768x576 picture; time goes with the pixel count
LAP, COARSE, FINE = 0.9, 96, 512     # tiles overlap by a tenth; picture widths the tiles are rendered at to choose, and to score

def ahead(C, e, d):
    """Metres from e along d to the first wall or object."""
    B = C.B; d = np.array(d, float)[None] / np.linalg.norm(d)
    return float(min([B.slab(e, d, *B.ROOM)[1][0]] + [B.hit(np.array(e, float), d, p)[0] for p in B.PRIMS]))

def clear(C, e, m=0.3):
    return m < e[0] < C.RW - m and m < e[2] < C.RD - m and 0.2 < e[1] and not any(C.B.inside(e, p, 0.2) for p in C.KEEP_OUT)

def probes(C):
    """Where a walker looks from, as [(name, kind, eye, look)]: the cameras tiles are chosen by."""
    c = np.array([C.RW / 2, C.EYE, C.RD / 2]); out = []
    turn = lambda a: np.array([np.cos(np.radians(a)), 0, np.sin(np.radians(a))])
    def add(e, d): out.append((f"p{len(out)}", "probe", e, e + d))
    step = max(1.0, np.sqrt(C.RW * C.RD / 24))                      # about 24 standing spots, whatever the room's size
    for x in np.arange(step / 2, C.RW, step):
        for z in np.arange(step / 2, C.RD, step):
            e = np.array([x, C.EYE, z])
            if not clear(C, e, 0.4): continue
            for a in (45, 135, 225, 315): add(e, turn(a))
            to = np.degrees(np.arctan2(c[2] - z, c[0] - x)); add(e, turn(to) * np.cos(np.radians(PITCH)) - [0, np.sin(np.radians(PITCH)), 0])
    for at, n, along, length in walls(C):                           # walking along the walls (outdoors: the edges of the ground)
        for i, s in enumerate(np.arange(0.75, length, 1.5)):
            e = at + s * along + NEAR * n; e[1] = C.EYE
            if not clear(C, e, 0.4): continue
            e = e + n * max(0.0, NEAR - ahead(C, e, -n))
            if clear(C, e, 0.4): add(e, -n); add(e, -n * np.cos(np.pi / 4) + along * np.sin(np.pi / 4) * (1 if i % 2 else -1))
    for oi, lo, hi in things(C):                                    # standing by each side of each large object
        for n, face, wide in sides(lo, hi):
            e = face + n * NEAR; e[1] = C.EYE; aim = face.copy(); aim[1] = min(C.EYE, 0.25 * lo[1] + 0.75 * hi[1])
            if clear(C, e, 0.4): add(e, aim - e)
    return out

def walls(C):
    """(a corner on the floor, the inward normal, the direction along, the length) of each wall."""
    v = lambda *a: np.array(a, float)
    return [(v(0, 0, 0), v(1, 0, 0), v(0, 0, 1), C.RD), (v(C.RW, 0, 0), v(-1, 0, 0), v(0, 0, 1), C.RD),
            (v(0, 0, 0), v(0, 0, 1), v(1, 0, 0), C.RW), (v(0, 0, C.RD), v(0, 0, -1), v(1, 0, 0), C.RW)]

def things(C):
    """Objects worth a view of their own: (index, bounds), not the ones that move away (the splat is trained without them)."""
    out = []
    for oi in range(len(C.objs)):
        lo, hi = C.bounds(C.OBJ_PRIMS[oi])
        if oi not in C.LIFT and (hi[1] - lo[1]) * max(hi[0] - lo[0], hi[2] - lo[2]) >= 0.5: out.append((oi, lo, hi))
    return out

def sides(lo, hi):
    """The four upright sides of a bounding box: (outward normal, the middle of the side, its width)."""
    m = (lo + hi) / 2
    return [(np.array(n, float), m + np.array(n, float) * (hi - m), w) for n, w in (([1, 0, 0], hi[2] - lo[2]), ([-1, 0, 0], hi[2] - lo[2]), ([0, 0, 1], hi[0] - lo[0]), ([0, 0, -1], hi[0] - lo[0]))]

def tiles(C, f, target):
    """Every candidate view: (kind, eye, look, metres to its surface). f: the detail pictures' focal length in pixels."""
    B = C.B; out = []; t = np.tan(np.radians(B.FOV) / 2)
    need = Sc.FE / NEAR                                             # screen pixels per metre of a wall 0.8 m away, seen square-on
    d = float(np.clip(f / (target * need), 0.5, 0.45 * min(C.RW, C.RD))); w = 2 * d * t
    def square_on(kind, p, n):
        """A camera d metres off the surface point p along its normal n, moved back if something stands in between."""
        e = p + d * n
        for _ in range(2): e = e + n * max(0.0, d - ahead(C, e, -n))
        if clear(C, e) and e[1] < C.RH - 0.2: out.append((kind, e, e - n, d))
    if not C.OUT:                                                   # outdoors the "walls" are a backdrop 20 m off: nobody stands by it
        for k, (at, n, along, length) in zip((0, 3, 2, 5), walls(C)):
            y = float(np.clip(C.EYE, 0.75 * w / 2, max(0.75 * w / 2, C.RH - 0.75 * w / 2))); cnt = int(np.ceil(length / (LAP * w)))
            for i in range(cnt): square_on(Sc.NAMES[k], at + along * length * (i + 0.5) / cnt + [0, y, 0], n)
    for oi, lo, hi in things(C):
        y = min(C.EYE, 0.25 * lo[1] + 0.75 * hi[1])
        for n, face, wide in sides(lo, hi):
            cnt = max(1, int(np.ceil(wide / (LAP * w)))); along = np.array([n[2], 0, n[0]])
            for i in range(cnt):
                p = face + along * wide * ((i + 0.5) / cnt - 0.5); p[1] = y; square_on(C.objs[oi]["name"], p, n)
    # the floor from above: the walker sees it most thinly where he looks, PITCH down from eye height
    s = np.sin(np.radians(PITCH)); need = Sc.FE * s ** 1.5 / C.EYE
    h = float(np.clip(f / (target * need), 1.2, C.RH - 0.3)); wx, wz = 2 * h * t, 1.5 * h * t
    nx, nz = int(np.ceil(C.RW / (LAP * wx))), int(np.ceil(C.RD / (LAP * wz)))
    for i in range(nx):
        for j in range(nz):
            e = np.array([C.RW * (i + 0.5) / nx, h, C.RD * (j + 0.5) / nz])
            # blockout cannot look straight down (basis needs a level part): 1 in 100 off, the picture's width along x
            if not any(B.inside(e, p, 0.2) for p in B.PRIMS): out.append(("floor", e, e + [0, -1, 0.01], h))
    return out

def owns(C, V, f):
    """What a detail view shows square-on enough to own, as its picture's blockout says: per surface, the share of the
    picture and the box of room metres it covers there."""
    B = C.B; h, w = V["sid"].shape
    with Sc.size(B, w, h): eye, d = B.rays(V["cam"])
    P = eye + d * V["t"][..., None]; out = []
    ys, xs = np.mgrid[:h, :w]; inner = (xs >= Sc.SEAM * w) & (xs < w - Sc.SEAM * w) & (ys >= Sc.SEAM * h) & (ys < h - Sc.SEAM * h)
    if V["lift"] is not None: inner &= ~V["lift"]
    for s in np.unique(V["sid"]):
        m = inner & (V["sid"] == s) & ~((s < 6) & V["halo"])
        if m.mean() < 0.02 or (C.OUT and s == 4) or (s >= 6 and s - 6 in C.LIFT): continue
        out.append({"surface": Sc.NAMES[s] if s < 6 else C.objs[s - 6]["name"], "picture_share": round(float(m.mean()), 3),
                    "from": [round(float(v), 2) for v in P[m].min(0)], "to": [round(float(v), 2) for v in P[m].max(0)]})
    return sorted(out, key=lambda o: -o["picture_share"])

def plan(world_dir, budget, target=0.5, paint=(768, 576), ring_paint=None, no_down=False, table=False):
    t0 = time.time(); C, world, src, rank, lifted, ring = Sc.load(world_dir, ring_paint, no_down); B = C.B
    f_ring, f = Sc.focal(B, ring[0]), Sc.focal(B, paint[0]); secs = SEC * paint[0] * paint[1] / REF
    # the walker's screens, coarse (one ray per 20 x 20 pixels), and the 24 fixed eval cameras, with the ring's r on them
    cams = K.layout_cams(world); S, E = Sc.samples(C, src, rank, lifted, f_ring, (probes(C), 20), (cams, Sc.STEP))
    cur = np.minimum(S["r"], 1); cand = []
    for kind, e, look, d in [t for R in (target if isinstance(target, (list, tuple)) else [target]) for t in tiles(C, f, R)]:
        cam = (tuple(map(float, e)), tuple(map(float, look)))
        idx, r, _ = Sc.take(C, Sc.view(C, cam, lifted, COARSE), f, S, 3 * d)       # a first cut: its own surface lies within 1.3 d
        cand.append((kind, cam, d, idx, np.minimum(r, 1)))
    picks = []
    for _ in range(min(max(budget, 8 if table else 0), len(cand))):
        gain = [float(np.maximum(r - cur[idx], 0).sum()) if k not in picks else -1.0 for k, (_, _, _, idx, r) in enumerate(cand)]
        k = int(np.argmax(gain))                                    # ties go to the first: walls, then objects, then floor
        if gain[k] <= 0: break
        picks.append(k); idx, r = cand[k][3:]; cur[idx] = np.maximum(cur[idx], r)
    # the score, on the eval cameras, with the picks rendered finer
    views = [Sc.view(C, cand[k][1], lifted, FINE) for k in picks]; takes = [Sc.take(C, V, f, E) for V in views]
    after = lambda n: Sc.stats(cams, E, *Sc.apply(E, takes[:n]))[0]
    before, _ = Sc.stats(cams, E, E["r"], E["worst"])
    out = {"world": os.path.basename(world), "budget": budget, "target": target, "ring": {"paint": ring, "views": C.N, "down_views_dropped": bool(no_down)},
           "views": [{"name": f"detail_{i:02d}", "kind": cand[k][0], "cam": [list(map(lambda v: round(v, 3), cand[k][1][0])), list(map(lambda v: round(v, 3), cand[k][1][1]))],
                      "fov": B.FOV, "size": list(paint), "metres_to_surface": round(cand[k][2], 2), "owns": owns(C, views[i], f), "paint_seconds": round(secs, 1)}
                     for i, k in enumerate(picks[:budget])],
           "paint_seconds": round(secs * min(budget, len(picks)), 1), "candidates": len(cand),
           "score": {"before": before, "after": after(budget)}}
    if table: out["table"] = {str(n): after(n) for n in (2, 4, 6, 8)}
    out["seconds"] = round(time.time() - t0, 2)
    return out

if __name__ == "__main__":
    a = sys.argv[2:]
    val = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    wh = lambda s: [int(v) for v in s.split("x")] if s else None
    print(json.dumps(plan(sys.argv[1], int(val("--budget", 4)), [float(v) for v in val("--target", "0.5").split(",")], wh(val("--paint", "768x576")), wh(val("--ring-paint")), "--no-down" in a, "--table" in a)))
