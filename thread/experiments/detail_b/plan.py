"""Plan a detail pass: a few extra close pictures of what a walking person gets near. Planning only: no GPU, no pictures read.
  plan.py WORLD --budget N [--pay down|shrink|none] [--detail WxH] [--step 8]
WORLD: a folder name under worlds/ or a path (needs boxes.json and depth/, as evals/density.py). Prints one JSON object:
  views    at most N extra views, in the order to paint them: "cam" = [eye, look] in room metres, exactly what
           chisel_ring.py keeps in B.CAMS; "fov" (side to side, blockout's), "paint" size, "seconds" to paint,
           "owns" = the parts of surfaces it would own, "gain" = its share of the softness removed, "pixels_used" =
           the share of its pixels that land on what it owns
  pay      how the pictures are paid for (see NOTES.md): the ring's new size, ring views dropped, seconds before and after
  walk     softness over the walk positions that the views were chosen for, before and after
  before, after   the geometry score on the 24 eval cameras (score.py), which the choice never looked at. --step: one
           ray per so many screen pixels; 8 agrees with evals/density.py's 4 to 0.001 on the cached worlds and is quicker
How views are chosen: people stand on a grid of free spots and look four ways level and two ways down; each screen
pixel's softness is how many octaves its paint is coarser than the screen (0 to 3; 3 = no paint). Candidate pictures
are ordinary photos from the same kind of spot: eye height, twelve headings, looking a little down (a wall with its
floor) or well down (the floor ahead). The one that removes the most softness is taken, the paint is updated, and again
until the budget is spent. A candidate owns only what score.py's rule lets it own.
"""
import sys, os, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import score as S
K = S.K
CELL = 0.15                                    # metres: screen pixels are gathered into surface patches this size
BINS = 32 * 2 ** (np.arange(14) / 2)           # screen pixels per metre a patch is seen at, in half octaves (32 .. 2900)
CAP = 3.0                                      # octaves: softer than this counts the same, as does no paint at all
LADDER = [(768, 576), (704, 528), (640, 480), (576, 432), (512, 384)]     # 4:3 sizes in steps of 64 the ring may be painted at
WALK = 25                                      # a walker's screen is traced every 25th pixel (64 x 36 rays)
PITCH = (-10, -40)                             # degrees a candidate looks down: wall with floor; the floor ahead
G = {}                                         # what forked workers read

def aim(eye, yaw, pitch):
    a, p = np.radians(yaw), np.radians(pitch)
    return tuple(float(v) for v in np.array(eye) + [np.cos(a) * np.cos(p), np.sin(p), np.sin(a) * np.cos(p)])

def spots(C, most):
    """Free places to stand at eye height, on a grid fine enough to give about `most` (never finer than 0.5 m)."""
    clear = lambda e: 0.4 < e[0] < C.RW - 0.4 and 0.4 < e[2] < C.RD - 0.4 and not any(C.B.inside(e, p, 0.3) for p in C.KEEP_OUT)
    d = max(0.5, float(np.sqrt(C.RW * C.RD / most)))
    xs, zs = [(np.arange(n) + 0.5) * w / n for n, w in ((max(1, round(C.RW / d)), C.RW), (max(1, round(C.RD / d)), C.RD))]
    return [np.array([x, C.EYE, z]) for x in xs for z in zs if clear((x, C.EYE, z))]

def pay(C, n, mode, detail):
    """Keep the painting time what it was: (ring size, ring views dropped, the arithmetic).
    shrink: every ring view is painted at the largest smaller size that frees enough seconds.
    down:   the views that look down from above are dropped first (the floor ahead is what detail views paint), every
            other one before the rest; if that frees too little the ring is shrunk as well.
    none:   nothing is paid; the build gets longer."""
    was = C.N * S.SECONDS; cost = n * S.paint_seconds(detail)
    downs = list(range(C.RING, C.N)); drop = (downs[1::2] + downs[0::2])[:n] if mode == "down" else []
    kept = C.N - len(drop)
    size = next((s for s in LADDER if mode == "none" or kept * S.paint_seconds(s) + cost <= was + 1e-6), LADDER[-1])
    now = kept * S.paint_seconds(size) + cost
    return size, sorted(drop), {"mode": mode, "ring_views": kept, "ring_paint": list(size), "ring_seconds_each": round(S.paint_seconds(size), 1),
                                "dropped_views": sorted(drop), "detail_views": n, "detail_seconds_each": round(S.paint_seconds(detail), 1),
                                "paint_seconds_before": round(was), "paint_seconds_after": round(now), "extra_seconds": round(now - was)}

def _walk(i):
    """One walker's screen: per pixel that a detail view could own, the patch key, point, normal, surface, ring paint
    and screen pixels per metre; and the softness of the pixels no detail view can own."""
    C, src, rank, lifted, k = G["C"], G["src"], G["rank"], G["lifted"], G["k"]; B = C.B; cam = G["walk"][i]
    P, sid, n = S.seen(C, cam, lifted, WALK); own = S.owner(C, src, rank, lifted, P, sid, n)
    q = np.zeros(sid.shape)
    for v in np.unique(own[own >= 0]):
        m = own == v; q[m] = S.per_metre(B, src[v][0], S.focal(B, S.BASE) * k, P[m], n[m])
    e = S.per_metre(B, cam, S.FE, P, n)
    free = S.may_own(C, lifted, sid) & ~S.hollow(C, lifted, P, sid, np.zeros(sid.shape, int)).astype(bool)
    sky = (sid == 4) if C.OUT else np.zeros(sid.shape, bool); rest = ~free & ~sky
    cell = np.floor(P[free] / CELL).astype(int); face = sid[free] < 6
    cell[face, sid[free][face] % 3] = 0                        # a point on a room face lies a hair either side of it: one patch, not two
    key = np.concatenate([sid[free][:, None], cell, np.where(face[:, None], 0, np.round(n[free]).astype(int) @ [[1], [3], [9]])], 1)
    return key, P[free], n[free], q[free], e[free], soft(q[rest], e[rest]), int(sky.sum())

def soft(q, e):
    """Octaves that paint at q pixels per metre is coarser than a screen showing e pixels per metre there (0 .. CAP)."""
    with np.errstate(divide="ignore"): return np.clip(np.log2(e / np.maximum(q, 1e-9)), 0, CAP)

def patches(pool):
    """Every walker pixel gathered into surface patches: (point, normal, surface, ring pixels per metre, weight per
    screen-density bin), and what cannot change: (softness of fixed pixels, pixels in all)."""
    res = pool.map(_walk, range(len(G["walk"])), chunksize=4)
    key, P, n, q, e = (np.concatenate([r[j] for r in res]) for j in range(5)); fixed = np.concatenate([r[5] for r in res])
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); inv = inv.ravel(); J = len(first)
    b = np.clip(np.round(2 * np.log2(np.maximum(e, 1) / 32)), 0, len(BINS) - 1).astype(int)
    Wt = np.zeros((J, len(BINS))); np.add.at(Wt, (inv, b), 1)
    cnt = np.bincount(inv, minlength=J); q0 = np.bincount(inv, q, J) / cnt
    return P[first], n[first], key[first, 0], q0, Wt, (float(fixed.sum()), float((fixed > 0).sum()), len(key) + len(fixed))

def _render(cam): return S.render(G["C"], G["lifted"], cam)

def _cands(job):
    """The candidate pictures from one spot: [(camera, patch indices it would show well, pixels per metre there)].
    What is hidden depends on the spot alone, so it is tested once for all its headings."""
    si, Pp, npp = job; C, detail = G["C"], G["detail"]; B = C.B; eye = G["spots"][si]; out = []
    to = eye - Pp; dist = np.linalg.norm(to, axis=1); cos = (to * npp).sum(1) / np.maximum(dist, 1e-9)
    near = np.nonzero((cos >= S.COS) & (dist > 0.3))[0]
    near = near[B.visible(eye, Pp[near])]                      # every primitive, lifted things too: the painter paints them
    if not len(near): return out
    for yaw in np.arange(12) * 30.0 + 15 * (si % 2):
        for pitch in PITCH:
            cam = (tuple(map(float, eye)), aim(eye, yaw, pitch))
            with S.sized(B, 16, 12):                           # a thumbnail of the picture: is it a photo anyone would take?
                t, _, sid = B.trace(cam, C.PRIM_OBJ)
            share = np.bincount(sid.ravel(), minlength=6) / sid.size; share[1] = 0
            if t.min() < 0.6 or t[5:7, 7:9].mean() < 1.2 or share.max() > 0.85: continue      # something in its face; one wall or thing filling it
            with np.errstate(all="ignore"), S.sized(B, 1000, 750):
                px, py, _, z = B.project(cam, Pp[near])
            m = S.EDGE * 1000; ok = (z > 0.05) & (px >= m) & (px < 1000 - m) & (py >= m) & (py < 750 - m)
            if ok.sum() < 20: continue
            j = near[ok]; out.append((cam, j, S.per_metre(B, cam, S.focal(B, detail), Pp[j], npp[j])))
    return out

def choose(cands, q, Wt, n):
    """Greedy, lazily: a candidate's gain can only fall as paint improves, so one is re-counted only when it comes up
    as the best. -> [(candidate index, gain, patches it takes, its pixels per metre there)]; q becomes the paint after
    all of them."""
    def gain(c):
        _, j, qc = cands[c]; new = np.where(qc >= S.GAIN * q[j], qc, q[j])
        return float((Wt[j] * (soft(q[j][:, None], BINS) - soft(new[:, None], BINS))).sum())
    g = np.array([gain(c) for c in range(len(cands))]); fresh = np.zeros(len(cands), bool); picked = []
    while len(picked) < n and len(cands):
        c = int(np.argmax(g))                                  # ties go to the lowest index: the same plan every run
        if g[c] <= 0: break
        if not fresh[c]: g[c] = gain(c); fresh[c] = True; continue
        _, j, qc = cands[c]; take = qc >= S.GAIN * q[j]; q[j[take]] = qc[take]
        picked.append((c, g[c], j[take], qc[take])); g[c] = -1; fresh[:] = False
    return picked

def owns(C, sid, P, j, qj):
    """The patches j a view owns (when it is painted), as parts of surfaces: name, area (a patch counts as CELL x CELL:
    right on walls and floor, too much on a shape), the box they lie in (room metres) and its painted pixels per metre."""
    names = ["wall x0", "floor", "wall z0", "wall x1", "ceiling", "wall z1"] + [o["name"] for o in C.objs]; out = []
    for s in np.unique(sid[j]):
        k = j[sid[j] == s]; qk = qj[sid[j] == s]
        if len(k) * CELL ** 2 < 0.1: continue                  # a sliver: not worth naming
        out.append({"surface": names[s], "m2": round(len(k) * CELL ** 2, 2), "from": np.round(P[k].min(0), 2).tolist(), "to": np.round(P[k].max(0), 2).tolist(),
                    "px_per_m": int(np.median(qk))})
    return sorted(out, key=lambda o: -o["m2"])

def used(B, cam, P):
    """The share of a picture that the patches at P cover, counted on a 32 x 24 grid of the picture."""
    with np.errstate(all="ignore"), S.sized(B, 32, 24):
        px, py, _, _ = B.project(cam, P)
    return round(len(np.unique(np.round(py).astype(int) * 32 + np.round(px).astype(int))) / (32 * 24), 2)

def plan(world_dir, budget, mode="down", detail=S.BASE, step=8):
    import multiprocessing as mp
    t0 = time.time(); C, world = K.layout(world_dir); B = C.B
    size, drop, bill = pay(C, budget, mode, detail)
    src, count, lifted = S.sources(C, world)
    stand = spots(C, 60)
    walk = [(tuple(map(float, e)), aim(e, 90 * i + 45 * (si % 2), 0)) for si, e in enumerate(stand) for i in range(4)]
    walk += [(tuple(map(float, e)), aim(e, 180 * i + 45 * (si % 4), -30)) for si, e in enumerate(stand) for i in range(2)]
    G.update(C=C, src=src, rank=S.ranks(count, drop), lifted=lifted, k=size[0] / S.BASE[0], walk=walk, spots=stand, detail=detail)
    # one set of forked workers for everything before the score: each builds its ray-casting scenes once
    pool = mp.get_context("fork").Pool(max(1, min(8, (os.cpu_count() or 4) - 2)))
    P, n, sid, q, Wt, (fixed, fixed_soft, pixels) = patches(pool)
    cands = [c for cs in pool.map(_cands, [(si, P, n) for si in range(len(stand))], chunksize=1) for c in cs]
    def walked(q):      # mean softness in octaves and the share of pixels that are soft at all (r < 1)
        s = soft(q[:, None], BINS); return {"mean_octaves_soft": round(float(((Wt * s).sum() + fixed) / pixels), 3), "soft_share": round(float(((Wt * (s > 0)).sum() + fixed_soft) / pixels), 4)}
    before = walked(q); total = float((Wt * soft(q[:, None], BINS)).sum()); views = []
    # of all the pixels the ring paints, the share that ends up on a surface a walker sees (the rest: overlap between
    # views, the ceiling, the backdrop). Rough, and too high where there are shapes: a patch counts as CELL x CELL
    ring_used = float((q ** 2).sum() * CELL ** 2 / ((C.N - len(drop)) * size[0] * size[1]))
    for c, g, j, qj in choose(cands, q, Wt, budget):
        cam = cands[c][0]; d = np.array(cam[1]) - cam[0]
        views.append({"cam": [list(cam[0]), list(cam[1])], "eye": list(cam[0]), "look": list(cam[1]), "heading_deg": round(float(np.degrees(np.arctan2(d[2], d[0]))) % 360, 1),
                      "pitch_deg": round(float(np.degrees(np.arcsin(d[1]))), 1), "fov": B.FOV, "paint": list(detail), "seconds": round(S.paint_seconds(detail), 1),
                      "gain": round(g / max(total, 1e-9), 4), "pixels_used": used(B, cam, P[j]), "owns": owns(C, sid, P, j, qj)})
    out = {"world": os.path.basename(world), "budget": budget, "views": views, "ring": list(size), "detail": list(detail), "drop": drop, "pay": bill,
           "walk": {"spots": len(stand), "screens": len(walk), "patches": len(P), "candidates": len(cands), "ring_pixels_used": round(ring_used, 2), "before": before, "after": walked(q)}}
    if len(views) < budget: bill["note"] = f"only {len(views)} views were worth painting; the ring was still shrunk for {budget}"
    dsrc = pool.map(_render, [(tuple(v["eye"]), tuple(v["look"])) for v in views], chunksize=1); pool.close()
    sc = S.score(world, out, step, (C, world, src, count, lifted, dsrc))
    out["before"], out["after"] = ({k: v for k, v in sc[s].items() if k != "cams"} for s in ("before", "after"))
    out["cams"] = {k: [sc["before"]["cams"][k], sc["after"]["cams"][k]] for k in sc["before"]["cams"]}
    out["seconds"] = round(time.time() - t0, 2)
    return out

if __name__ == "__main__":
    a = sys.argv[1:]; opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    print(json.dumps(plan(a[0], int(opt("--budget", 4)), opt("--pay", "down"), tuple(int(v) for v in opt("--detail", "768x576").split("x")), int(opt("--step", 8)))))
