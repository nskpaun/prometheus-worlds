"""Plan a detail pass for a prompt world and say what it would do to the geometry score. No GPU, nothing is painted.
  plan.py WORLD --budget N [--paint-seconds 1020] [--secs 65] [--big-floor] [--keep-ring]
WORLD: a folder name under worlds/ or a path (needs boxes.json, depth/, gen/views.json). Prints one JSON object:
  views   at most N extra pictures: "eye" and "look" in room metres, the pair chisel_ring.py keeps in B.CAMS; "fov" in
          degrees side to side, as blockout.FOV; "size" in pixels; "seconds" to paint; "owns": what it would own, each
          with its share of the picture and its box in room metres.
  pay     how the ring is cut back so that ring + detail fits in --paint-seconds: which views are left out, what size
          the rest are painted at, and the sum.
  score   evals/density.py's numbers on the 24 eval cameras, "before" (the world as painted) and "after".
How it chooses. Candidates: each wall cut into 1..6 tiles of its lower 2.6 m seen square-on; the floor cut into grids
and seen from above; each large thing from each side it can be stood at. A walker is stood on a grid of free spots
looking eight ways (not the eval cameras: the plan must not be fitted to the test). For every way of paying (leave out
the k ring views that own least, paint the rest at the largest size that then fits) candidates are taken greedily by
gain per second, gain being the rise in min(r, 1) summed over what the walker sees. The best way of paying wins; ways
within 1% of it give way to the one that leaves out fewest views.
--big-floor also offers each floor tile at 1024x768's pixel count (160 s). --keep-ring pays by size only.
"""
import sys, os, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import score as Z
K = Z.K
BAND, TILE, MIN_SIDE = 2.6, 6, 1.5      # metres of wall height a tile takes; most tiles along one wall; smallest floor tile side
RING_SIZES = [(64 * j, 48 * j) for j in (12, 11, 10, 9, 8)]      # 768x576 down to 512x384: nothing smaller, the painter was never run there
WALK = 32                                # a walker's picture is traced every 32nd screen pixel (50 x 28)

def size_for(aspect, px):
    """A picture of about px pixels and this shape, both sides multiples of 16 as the painter needs."""
    h = max(16, round(np.sqrt(px / aspect) / 16) * 16); return [max(16, round(h * aspect / 16) * 16), h]

def candidates(C, big_floor=False):
    """Every detail view on offer: {"name", "kind", "eye", "look", "fov", "size", "seconds"}."""
    B = C.B; RW, RH, RD = C.RW, C.RH, C.RD; px = Z.BASE[0] * Z.BASE[1]; grow = 1 / (1 - 2 * Z.EDGE); out = []
    top = (8.0 if C.OUT else RH - 0.3)       # outdoors a picture from 8 m up is an ordinary drone photo
    clear = lambda e: 0.3 < e[0] < RW - 0.3 and 0.3 < e[2] < RD - 0.3 and 0.3 < e[1] <= top + 1e-6 and not any(B.inside(e, p, 0.25) for p in C.KEEP_OUT)
    def stand(at, n3, w, h, aspect, dmax):
        """The nearest clear spot along n3 from which a w x h patch at `at` fills the picture inside its rim: (eye, fov)."""
        for fov in (65.0, 72.0, 80.0):       # 65 is the angle every ring view is painted at; wider only when the room is too small
            th = np.tan(np.radians(fov) / 2); d = max(w * grow / 2 / th, h * grow / 2 / (th / aspect))
            e = np.array(at, float) + np.array(n3, float) * d
            if d <= dmax and clear(e): return e, fov
        return None
    def add(name, kind, got, look, size, secs):
        if got: out.append({"name": name, "kind": kind, "eye": [round(float(v), 3) for v in got[0]], "look": [round(float(v), 3) for v in look], "fov": got[1], "size": size, "seconds": round(secs, 1)})
    if not C.OUT:                            # outdoors the sides are a backdrop 20 m off: nobody gets near it
        hb = min(RH, BAND)
        for name, n3, ax, fix, length, far in (("x0", (1, 0, 0), 2, 0.0, RD, RW), ("x1", (-1, 0, 0), 2, RW, RD, RW), ("z0", (0, 0, 1), 0, 0.0, RW, RD), ("z1", (0, 0, -1), 0, RD, RW, RD)):
            for k in range(1, TILE + 1):
                w = length / k
                if k > 1 and w < MIN_SIDE: break
                size = size_for(float(np.clip(w / hb, 0.75, 2.0)), px)
                for i in range(k):
                    at = np.array([fix, hb / 2, fix]); at[ax] = (i + 0.5) * w
                    add(f"wall_{name}_{i + 1}of{k}", "wall", stand(at, n3, w, hb, size[0] / size[1], far - 0.3), at, size, Z.seconds(size))
    for kx in range(1, 6):
        for kz in range(1, 6):
            tw, td = RW / kx, RD / kz; lng, sht = max(tw, td), min(tw, td)
            if (kx * kz > 1 and sht < MIN_SIDE) or lng > 2.2 * sht or kx * kz > 12: continue      # more than 12 tiles is more than any budget here buys
            for scale in (1.0, 1024 * 768 / px) if big_floor else (1.0,):
                size = size_for(float(np.clip(lng / sht, 1.0, 2.0)), px * scale)
                for ix in range(kx):
                    for iz in range(kz):
                        at = np.array([(ix + 0.5) * tw, 0.0, (iz + 0.5) * td])
                        # straight down has no "up" for blockout.basis: the look point is nudged 2 cm so the picture's long side lies along the tile's
                        look = at + ([0, 0, 0.02] if tw >= td else [0.02, 0, 0])
                        add(f"floor_{kx}x{kz}_{ix * kz + iz + 1}" + ("_big" if scale > 1 else ""), "floor", stand(at, (0, 1, 0), lng, sht, size[0] / size[1], top), look, size, Z.seconds(size))
    for oi, o in enumerate(C.objs):
        if oi in C.LIFT: continue
        lo, hi = C.bounds(C.OBJ_PRIMS[oi]); m = (lo + hi) / 2; hgt = hi[1] - lo[1]
        for side, n3, half, wide, far in (("x+", (1, 0, 0), hi[0] - m[0], hi[2] - lo[2], RW - hi[0]), ("x-", (-1, 0, 0), hi[0] - m[0], hi[2] - lo[2], lo[0]),
                                          ("z+", (0, 0, 1), hi[2] - m[2], hi[0] - lo[0], RD - hi[2]), ("z-", (0, 0, -1), hi[2] - m[2], hi[0] - lo[0], lo[2])):
            if max(wide, hgt) < 0.7: continue
            size = size_for(float(np.clip(wide / max(hgt, 0.1), 0.75, 2.0)), px)
            at = m + np.array(n3, float) * half; y = min(C.EYE, max(m[1], 0.9))      # level at tall things, a little down at low ones
            got = stand([at[0], y, at[2]], n3, 1.1 * wide, 1.1 * hgt, size[0] / size[1], far - 0.3)
            add(f"obj{oi}_{o['name'].replace(' ', '-')}_{side}", "object", got, at, size, Z.seconds(size))
    return out

def walk_cams(C):
    """Where a walking person looks from: a grid of free standing spots no further apart than 2.5 m and reaching to
    0.8 m from the walls, six level directions and two looking 35 degrees down at each."""
    B = C.B; step = float(np.clip(min(C.RW, C.RD) / 4, 1.0, 2.5)); out = []
    line = lambda L: np.linspace(0.8, L - 0.8, max(2, int(round((L - 1.6) / step)) + 1))
    for ix, x in enumerate(line(C.RW)):
        for iz, z in enumerate(line(C.RD)):
            e = np.array([x, C.EYE, z])
            if any(B.inside(e, p, 0.3) for p in C.KEEP_OUT): continue
            for j in range(8):
                a = np.radians(60 * j + 20 * ((ix + iz) % 3)) if j < 6 else np.radians(180 * j + 45 * ((ix + 2 * iz) % 4))
                d = np.array([np.cos(a), 0, np.sin(a)]) if j < 6 else np.array([np.cos(a) * np.cos(np.radians(35)), -np.sin(np.radians(35)), np.sin(a) * np.cos(np.radians(35))])
                out.append((tuple(e), tuple(e + d)))
    return out

def _walk(i):
    """One walker picture: the points it sees, every ring view's score and r there, and how the screen's pixels lie on them."""
    C, src, lifted = Z.G["C"], Z.G["src"], Z.G["lifted"]; B = C.B; cam = Z.G["walk"][i]
    P, sid, n = Z.seen(C, cam, lifted, WALK); S = Z.scores(C, src, Z.G["rank"], lifted, P, sid, n)
    dead = Z.shade(C, lifted, P, sid); sky = (sid == 4) if C.OUT else np.zeros(sid.shape, bool); ok = ~sky.ravel()
    P, sid, n, dead, S = P.reshape(-1, 3)[ok], sid.ravel()[ok], n.reshape(-1, 3)[ok], dead.ravel()[ok], S.reshape(len(src), -1)[:, ok]      # the sky is nobody's to paint
    T = Z.tangents(n); Je = Z.spread(B, cam, Z.FE, P, T); fo = Z.focal(Z.BASE, B.FOV); R = np.zeros(S.shape, np.float32)
    for v in range(len(src)):
        m = S[v] < 1e5
        if m.any(): R[v, m] = Z.mix(Z.spread(B, src[v][0], fo, P[m], T[m]), Je[m])[0]
    return P, sid, n.astype(np.float32), dead, S, R, Je.astype(np.float32)

def _cand(c):
    """One candidate: its small surface map, the walker points it would own and its r at each."""
    C, lifted = Z.G["C"], Z.G["lifted"]; P, sid, n, dead, Je = Z.G["pts"]; d = Z.look_through(C, Z.G["cands"][c], lifted)
    j = np.nonzero(Z.owned(C, d, lifted, P, sid, n, dead))[0]
    r = Z.mix(Z.spread(C.B, (d["eye"], d["look"]), Z.focal(d["size"], d["fov"]), P[j], Z.tangents(n[j].astype(float))), Je[j])[0] if len(j) else np.zeros(0)
    return d, j.astype(np.int32), r.astype(np.float32)

def demand(C, src, rank, lifted, cands):
    """What the walker sees, gathered: S, R (views x points), dead, and per candidate (point indices, r)."""
    Z.G.update(C=C, src=src, rank=rank, lifted=lifted, cands=cands, walk=walk_cams(C))
    res = Z.pool_map(_walk, len(Z.G["walk"]))
    P, sid, n, dead = (np.concatenate([r[j] for r in res]) for j in range(4)); S, R = (np.concatenate([r[j] for r in res], 1) for j in (4, 5))
    Z.G["pts"] = (P, sid, n, dead, np.concatenate([r[6] for r in res]))
    got = Z.pool_map(_cand, len(cands)); cands[:] = [g[0] for g in got]
    return S, R, dead, [g[1:] for g in got]

def ring_r(S, R, dead, keep, size):
    """r at each walker point from the ring views in `keep` painted at `size`, and which view owns it."""
    keep = np.asarray(keep); j = S[keep].argmin(0); own = keep[j]; ok = (S[own, np.arange(S.shape[1])] < 1e5) & ~dead
    return np.where(ok, R[own, np.arange(S.shape[1])] * size[0] / Z.BASE[0], 0.0), np.where(ok, own, -1)

def greedy(base, hit, cands, n, allow):
    """Up to n candidates within `allow` seconds, each the best gain per second given those already taken.
    -> (their indices, r at each walker point afterwards)."""
    use = lambda r: np.minimum(r, 1.0)       # paint finer than the screen shows nothing more
    cur, det, took, left = base.copy(), np.zeros(len(base)), [], allow
    while len(took) < n:
        best = (1e-4 * len(base) / Z.SECS, None)     # a view must lift a ten-thousandth of what is seen to be worth its minute
        for c, (j, v) in enumerate(hit):
            if c in took or cands[c]["seconds"] > left + 1e-6 or not len(j): continue
            gain = float((use(np.maximum(det[j], v)) - use(cur[j])).sum()) / cands[c]["seconds"]
            if gain > best[0]: best = (gain, c)
        if best[1] is None: break
        c = best[1]; j, v = hit[c]; det[j] = np.maximum(det[j], v); cur[j] = det[j]; took.append(c); left -= cands[c]["seconds"]
    return took, cur

def choose(C, S, R, dead, hit, cands, n, budget, secs, keep_ring=False):
    """The best way to pay and the views it buys: (kept ring views, ring size, chosen candidates, table of what was tried)."""
    N = S.shape[0]; _, own = ring_r(S, R, dead, range(N), Z.BASE)
    share = np.bincount(own[own >= 0], minlength=N).astype(float); share[0] = np.inf      # view 0 is painted first and sets the look: never dropped
    order = [int(v) for v in np.argsort(share, kind="stable")]; tried = []
    want = min([c["seconds"] for c in cands] or [0]) * n
    for k in range(0, 1 if keep_ring else N - 5):
        keep = sorted(order[k:])
        size = next((s for s in RING_SIZES if len(keep) * Z.seconds(s, secs) <= budget - want), None)
        if size is None: continue
        base, _ = ring_r(S, R, dead, keep, size); ring_s = len(keep) * Z.seconds(size, secs)
        took, cur = greedy(base, hit, cands, n, budget - ring_s)
        tried.append({"dropped": sorted(order[:k]), "size": list(size), "ring_seconds": round(ring_s, 1), "took": took,
                      "walker_mean": round(float(np.minimum(cur, 1).mean()), 4), "walker_median": round(float(np.median(cur)), 3), "keep": keep})
    if not tried: return list(range(N)), Z.BASE, [], tried
    top = max(t["walker_mean"] for t in tried)
    pick = min((t for t in tried if t["walker_mean"] >= 0.99 * top), key=lambda t: (len(t["dropped"]), -t["size"][0]))
    return pick["keep"], tuple(pick["size"]), pick["took"], tried

def owns(C, d, lifted):
    """What a detail view would own, looked at through its own picture: [{"surface", "share" of the picture, "box"}]."""
    P, sid, n = Z.seen(C, (d["eye"], d["look"]), [], size=d["sid"].shape[::-1], fov=d["fov"])
    P, sid, n = P.reshape(-1, 3), sid.ravel(), n.reshape(-1, 3)
    m = Z.owned(C, d, lifted, P, sid, n, Z.shade(C, lifted, P, sid)); out = []
    for s in np.unique(sid[m]):
        k = m & (sid == s)
        if k.mean() < 0.01: continue
        out.append({"surface": Z.FACES[s] if s < 6 else C.objs[s - 6]["name"], "id": int(s), "share": round(float(k.mean()), 3),
                    "box": [[round(float(v), 2) for v in P[k].min(0)], [round(float(v), 2) for v in P[k].max(0)]]})
    return sorted(out, key=lambda o: -o["share"])

def brief(s): return {"near_median_r": s["near"]["median_r"], "soft_share_r_lt_1": s["near"]["r_lt_1"], "r_lt_0.5": s["near"]["r_lt_0.5"], "unpainted": s["near"]["unpainted"],
                      "near_median_worst": s["near"]["median_worst"], "worst_close_cam": s["worst_close_cam"], "close_median_r": s["close"]["median_r"],
                      "floor_median_r": s["floor"]["median_r"], "centre_median_r": s["centre"]["median_r"]}

def plan(world_dir, n, budget=1020.0, secs=Z.SECS, big_floor=False, keep_ring=False, with_cams=False, ring=None):
    """ring: (chisel_ring, world folder, sources, eval cameras) already loaded, for a ring other than the painted one (study.py)."""
    t0 = time.time(); C, world = ring[:2] if ring else K.layout(world_dir); src, rank, lifted = ring[2] if ring else Z.sources(C, world)
    try: painted = json.load(open(f"{world}/gen/views.json"))["paint"]
    except (OSError, KeyError, ValueError): painted = list(Z.BASE)
    cands = candidates(C, big_floor) if n > 0 else []
    for c in cands: c["seconds"] = round(c["seconds"] * secs / Z.SECS, 1)
    S, R, dead, hit = demand(C, src, rank, lifted, cands)
    keep, size, took, tried = choose(C, S, R, dead, hit, cands, n, budget, secs, keep_ring)
    views = [cands[c] for c in took]
    before, after = Z.evaluate(C, world, src, rank, lifted, [(list(range(C.N)), painted, []), (keep, size, views)], ring[3] if ring else None)
    out_views = [{"name": d["name"], "kind": d["kind"], "eye": d["eye"], "look": d["look"], "fov": d["fov"], "size": d["size"], "seconds": d["seconds"], "owns": owns(C, d, lifted)} for d in views]
    ring_s = len(keep) * Z.seconds(size, secs); det_s = sum(d["seconds"] for d in views)
    out = {"world": os.path.basename(world), "budget": n, "views": out_views,
           "pay": {"paint_seconds_allowed": budget, "as_painted": {"views": C.N, "size": list(painted), "seconds": round(C.N * Z.seconds(painted, secs), 1)},
                   "ring": {"views_kept": keep, "views_left_out": [v for v in range(C.N) if v not in keep], "size": list(size), "seconds": round(ring_s, 1)},
                   "detail_seconds": round(det_s, 1), "total_seconds": round(ring_s + det_s, 1)},
           "score": {"before": brief(before), "after": brief(after)},
           "walker": {"cameras": len(Z.G["walk"]), "candidates": len(cands), "ways_to_pay_tried": [{k: v for k, v in t.items() if k != "keep"} for t in tried]}}
    if with_cams: out["cams"] = {"before": before["cams"], "after": after["cams"]}
    out["seconds"] = round(time.time() - t0, 2)
    return out

if __name__ == "__main__":
    a = sys.argv[2:]; val = lambda flag, d: type(d)(a[a.index(flag) + 1]) if flag in a else d
    print(json.dumps(plan(sys.argv[1], val("--budget", 4), val("--paint-seconds", 1020.0), val("--secs", Z.SECS), "--big-floor" in a, "--keep-ring" in a, "--cams" in a)))
