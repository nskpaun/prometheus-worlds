"""The geometry score (evals/density.py: painted pixels per screen pixel, r) with planned detail views as extra owners.
  score.py WORLD PLAN.json [--step 4]     PLAN.json: what plan.py printed. Prints {"before", "after"} as one JSON line.
The maths is copied from evals/density.py (sources, normal, seen, owner, spread, ratio), not imported: that file is
being changed, and its owner() had to be opened up. The eval cameras themselves come from evals/cams.py, so the same
24 are judged. Nothing is written, no picture is read, no GPU.
What a detail view owns (the rule consistent.py would need, see NOTES.md): a point it really sees, by the same tests a
ring view must pass, and also
  - not within EDGE of its picture's border (room to fade into the ring paint),
  - seen at least COS head-on (a detail view exists to be head-on; a ring view is taken at any angle),
  - not the ceiling, the sky or an outdoor backdrop, nor a thing that is lifted out of the splat,
  - where it has at least GAIN times the painted pixels per metre of whoever owns the point now.
Detail views are tried in the plan's order, so an earlier one keeps what a later one does not clearly beat.
Paying for the pictures: "ring" in the plan is the size the ring is painted at instead of 768x576 and "drop" the ring
views left out; both are applied to the "after" side. "before" is the world as it was painted.
"""
import sys, os, json, time, contextlib, numpy as np
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); TH = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TH, "evals"))
import cams as K
SCREEN, VFOV = (1600, 900), 65.0               # the viewer's picture, as evals/density.py
FE = SCREEN[1] / 2 / np.tan(np.radians(VFOV) / 2)
OFFS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
EDGE, COS, GAIN = 0.04, 0.3, 1.25              # share of the picture's width; cosine; pixels per metre, times
BASE = (768, 576)                              # the size every timing and the cached pictures are for
SECONDS = 65.0                                 # one picture at BASE, measured

def paint_seconds(size):
    """Seconds to paint one picture. Taken as proportional to its pixels, as the task states; only 768x576 (65 s) and
    1024x768 (160 s, which is more than proportional) were measured, so smaller sizes are an estimate."""
    return SECONDS * size[0] * size[1] / (BASE[0] * BASE[1])

def focal(B, size): return size[0] / 2 / np.tan(np.radians(B.FOV) / 2)

@contextlib.contextmanager
def sized(B, w, h, fov=None):
    """blockout renders at the size it is told in its globals: set them, and put them back."""
    old = B.W, B.H, B.FOV; B.W, B.H = w, h
    if fov: B.FOV = fov
    try: yield
    finally: B.W, B.H, B.FOV = old

def band(sid, C, lifted):
    """(silhouette edges, the band round shapes, the band round lifted things) of one painted view's surface ids."""
    big = lambda m, px: ndimage.maximum_filter(m, px, mode="nearest")
    edge = np.zeros(sid.shape, bool)
    for dy, dx in OFFS: edge |= sid != np.roll(sid, (dy, dx), (0, 1))
    on = np.isin(sid, [6 + oi for oi in {*C.SHAPES, *(C.SLIM if C.NEW else ())}])
    return edge, big(on, C.HALO) & ~on, big(np.isin(sid, [6 + oi for oi in lifted]), 81) if lifted else None

def sources(C, world):
    """As evals/density.py: per painted view (camera, depth, surface ids, edges, bands), how many pixels of each
    surface each view shows, the lifted objects."""
    NS = 6 + len(C.objs)
    try: lifted = [i for i in json.load(open(f"{world}/gen_c/lifted.json")) if type(i) is int]
    except (OSError, ValueError): lifted = [] if os.path.isdir(f"{world}/gen_c") else list(C.LIFT)
    src = []
    for i in range(C.N):
        t, sid, _ = C.surfaces(i); src.append((C.B.CAMS[str(i)], t, sid, *band(sid, C, lifted)))
    return src, np.stack([np.bincount(s[2].ravel(), minlength=NS) for s in src], 1).astype(float), lifted

def ranks(count, drop=()):
    """Each view's rank for each surface, by how much of it the view shows (consistent.py's rule). drop: ring views
    that are not painted at all: their rank is "never"."""
    count = count.copy(); count[:, list(drop)] = 0
    rank = np.argsort(np.argsort(-count, axis=1), axis=1).astype(float); rank[count < 400] = 1e6
    return rank

def detail_sources(C, views, lifted):
    """The same for planned views, which have no saved depth yet: rendered here as chisel_ring.py prep would."""
    global _JOBS; _JOBS = [(C, lifted, (tuple(v["eye"]), tuple(v["look"]))) for v in views]
    return _pool(_render, len(views))

def render(C, lifted, cam):
    t, _, sid, _ = C.B.trace(cam, C.PRIM_OBJ, True)
    return (cam, t, sid, *band(sid, C, lifted))

def _render(i): return render(*_JOBS[i])

def _pool(fn, n):
    """fn(0..n-1) in forked workers, which share the depth maps without copying, as consistent.py's do."""
    import multiprocessing as mp
    k = max(1, min(8, (os.cpu_count() or 4) - 2, n))
    if k < 2: return [fn(i) for i in range(n)]
    with mp.get_context("fork").Pool(k) as pool: return pool.map(fn, range(n), chunksize=1)

def normal(p, P):
    """Unit normals of a box or cylinder primitive at points P on its surface."""
    o = P - np.array(p["c"], float); n = np.zeros(P.shape)
    if p["shape"] == "cyl":
        cap = np.abs(o[:, 1]) > p["hy"] - 1e-4
        n[cap, 1] = 1; n[~cap] = o[~cap] * [1, 0, 1]
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    cs, sn = np.cos(p.get("yaw", 0.0)), np.sin(p.get("yaw", 0.0))
    l = np.stack([cs * o[:, 0] - sn * o[:, 2], o[:, 1], sn * o[:, 0] + cs * o[:, 2]], 1)
    k = np.argmax(np.abs(l) / np.array(p["h"], float), 1); e = np.eye(3)[k]       # the face the point lies on
    return np.stack([cs * e[:, 0] + sn * e[:, 2], e[:, 1], -sn * e[:, 0] + cs * e[:, 2]], 1)

def seen(C, cam, lifted, step=4):
    """What a screen camera sees of the blockout the splat was trained on: per traced pixel the point, the surface id
    and the surface normal (turned towards the camera)."""
    B = C.B; keep = [j for j, oi in enumerate(C.PRIM_OBJ) if oi not in lifted]; full = B.PRIMS
    B.PRIMS = [full[j] for j in keep]
    try:
        with sized(B, SCREEN[0] // step, SCREEN[1] // step, 2 * np.degrees(np.arctan(np.tan(np.radians(VFOV) / 2) * SCREEN[0] / SCREEN[1]))):
            t, P, pid, nrm = B.trace(cam, keep, True)
    finally: B.PRIMS = full
    n = nrm.astype(float); sid = pid.copy()
    for f in range(6): n[pid == f, f % 3] = 1
    for j in keep:
        m = pid == 6 + j; sid[m] = 6 + C.PRIM_OBJ[j]
        if m.any() and full[j]["shape"] != "mesh": n[m] = normal(full[j], P[m])
    n *= np.sign(((np.array(cam[0]) - P) * n).sum(-1, keepdims=True))            # a room face's normal points inwards
    return P, sid, n

def shows(C, s, P, sid, want, m=1):
    """Where painted view s = (camera, depth, ids, edges, bands) really shows the points P: consistent.py's tests.
    m: pixels kept clear of the picture's border. -> (ok, in the band round a shape, in the band round a lifted thing)"""
    B = C.B; cam, st, ssid, sedge, shalo, slift = s; H, W = ssid.shape
    with np.errstate(all="ignore"), sized(B, W, H):
        px, py, dist, z = B.project(cam, P)
        xi, yi = np.round(px).astype(int), np.round(py).astype(int)
    inside = (z > 0.05) & (xi >= m) & (xi < W - m) & (yi >= m) & (yi < H - m)
    xi, yi = np.clip(xi, 0, W - 1), np.clip(yi, 0, H - 1)
    want = want & inside
    plain = want & (sid < 6) & ~sedge[yi, xi]
    ok = plain & (ssid[yi, xi] == sid) & (np.abs(st[yi, xi] - dist) < np.maximum(0.05, 0.01 * dist))
    ex = np.nonzero(want & ~plain)
    if len(ex[0]):
        ey, ex_x, k = yi[ex], xi[ex], sid[ex]
        vis = B.visible(cam[0], P[ex]); miss = vis & (ssid[ey, ex_x] != k)
        for dy, dx in OFFS:
            if not miss.any(): break
            mm = miss & (ssid[np.clip(ey + dy, 0, H - 1), np.clip(ex_x + dx, 0, W - 1)] == k)
            ey[mm] += dy; ex_x[mm] += dx; miss &= ~mm
        ok[ex] = vis & ~miss
    return ok, shalo[yi, xi], slift[yi, xi] if slift is not None else np.zeros(sid.shape, bool)

def owner(C, src, rank, lifted, P, sid, n):
    """The ring view each point takes its paint from, -1 where none: consistent.py's render(), without the colours."""
    N = len(src); NS = rank.shape[0]
    mesh = np.isin(np.arange(NS), [6 + oi for oi in C.SHAPES]); on = mesh[sid]
    best = np.full(sid.shape, 1e5); own = np.full(sid.shape, -1)
    for v in np.argsort((np.minimum(rank, N) * np.bincount(sid.ravel(), minlength=NS)[:, None]).sum(0)):
        if (rank[:, v] >= 1e6).all(): continue
        score = rank[sid, v]
        if on.any():
            to = np.array(src[v][0][0]) - P[on]; cos = (to * n[on]).sum(1) / np.maximum(np.linalg.norm(to, axis=1), 1e-9)
            score[on] += N * (cos < C.GRAZE); score[on] = np.where(cos <= 0, 1e6, score[on])
        # the two bands change the score before it is compared, so they are looked up first (cheap) and the tests after
        with np.errstate(all="ignore"), sized(C.B, *src[v][2].shape[::-1]):
            px, py, _, _ = C.B.project(src[v][0], P)
        yi, xi = (np.clip(np.nan_to_num(np.round(a)), 0, lim - 1).astype(int) for a, lim in ((py, src[v][2].shape[0]), (px, src[v][2].shape[1])))
        if mesh.any(): score = score + N * src[v][4][yi, xi]
        if lifted: score = np.where(src[v][5][yi, xi], 1e6, score)
        ok, _, _ = shows(C, src[v], P, sid, score < best)
        best = np.where(ok, score, best); own[ok] = v
    return hollow(C, lifted, P, sid, own)

def hollow(C, lifted, P, sid, own):
    """Points that are filled in, not painted, whoever shows them: the floor round a lifted thing, the sky."""
    for lo, hi in (C.bounds(C.OBJ_PRIMS[oi]) for oi in lifted):
        if lo[1] < 0.6:
            gap = np.hypot(np.maximum(np.maximum(lo[0] - P[..., 0], P[..., 0] - hi[0]), 0), np.maximum(np.maximum(lo[2] - P[..., 2], P[..., 2] - hi[2]), 0))
            own[(sid == 1) & (gap < float(np.clip(hi[1], 0.4, 2.0)))] = -1
    if C.OUT: own[sid == 4] = -1
    return own

def spread(B, cam, focal, P, T):
    """How a picture's pixels lie on the surface at points P: the 2x2 matrix of pixels per metre along the two surface
    directions T. cam: (eye, look); focal in pixels."""
    eye, r, u, f = B.basis(cam); v = P - eye; z = (v @ f)[..., None, None]
    a = np.stack([v @ r, v @ u], -1)[..., :, None]
    along = lambda d: np.einsum("...ik,i->...k", T, d)
    Tc = np.stack([along(r), along(u)], -2)
    return focal * (Tc * z - a * along(f)[..., None, :]) / z ** 2

def tangents(n):
    ax = np.eye(3)[np.argmin(np.abs(n), -1)]
    t1 = np.cross(n, ax); t1 /= np.linalg.norm(t1, axis=-1, keepdims=True)
    return np.stack([t1, np.cross(n, t1)], -1)

def per_metre(B, cam, focal, P, n):
    """Pixels per metre of surface (the square root of pixels per m2) that a picture or the screen has at points P."""
    J = spread(B, cam, focal, P, tangents(n))
    return np.sqrt(np.abs(J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]))

def ratio(B, P, n, src_cam, fo, cam, fe):
    """(r, worst) at points P with normals n: painted pixels per screen pixel, by area and along the thinnest direction."""
    T = tangents(n); Jo, Je = spread(B, src_cam, fo, P, T), spread(B, cam, fe, P, T)
    de = Je[..., 0, 0] * Je[..., 1, 1] - Je[..., 0, 1] * Je[..., 1, 0]
    de = np.where(np.abs(de) < 1e-12, 1e-12, de)[..., None, None]
    M = Jo @ (np.stack([np.stack([Je[..., 1, 1], -Je[..., 0, 1]], -1), np.stack([-Je[..., 1, 0], Je[..., 0, 0]], -1)], -2) / de)
    det = np.abs(M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]); q = (M ** 2).sum((-1, -2))
    return np.sqrt(det), np.sqrt(np.maximum(q - np.sqrt(np.maximum(q * q - 4 * det * det, 0)), 0) / 2)

def may_own(C, lifted, sid):
    """Surfaces a detail view is allowed to own at all."""
    no = [4] + (list(C.BACKDROP) if C.OUT else []) + [6 + oi for oi in lifted]
    return ~np.isin(sid, no)

def take_over(C, dsrc, fo, lifted, P, sid, n, cam, r, worst, own, first):
    """Let the detail views own what they clearly improve, in order. r, worst, own are changed in place; a detail
    view k is written to own as first + k."""
    B = C.B; free = may_own(C, lifted, sid); mesh = np.isin(sid, [6 + oi for oi in C.SHAPES])
    for k, s in enumerate(dsrc):
        to = np.array(s[0][0]) - P; cos = (to * n).sum(-1) / np.maximum(np.linalg.norm(to, axis=-1), 1e-9)
        ok, halo, lift = shows(C, s, P, sid, free & (cos >= COS), int(round(EDGE * s[2].shape[1])))
        ok &= ~lift & ~(halo & ~mesh)            # the band round a shape may hold the shape's paint: not this view's wall
        if not ok.any(): continue
        rk, wk = ratio(B, P[ok], n[ok], s[0], fo, cam, FE)
        better = rk >= GAIN * r[ok]; at = tuple(a[better] for a in np.nonzero(ok))
        r[at], worst[at], own[at] = rk[better], wk[better], first + k
    own[:] = hollow(C, lifted, P, sid, own)
    r[own < 0] = 0; worst[own < 0] = 0

def stats(pools, kinds):
    cat = lambda j: np.concatenate([p[j] for k in kinds for p in pools.get(k, [])])
    r, w, none, det = (cat(j) for j in range(4))
    return {"median_r": round(float(np.median(r)), 3), "r_lt_1": round(float((r < 1).mean()), 4), "r_lt_0.5": round(float((r < 0.5).mean()), 4),
            "median_worst": round(float(np.median(w)), 3), "worst_lt_1": round(float((w < 1).mean()), 4), "unpainted": round(float(none.mean()), 4),
            "detail_owned": round(float(det.mean()), 4)}

def summary(pools, per):
    close = {k: v for k, v in per.items() if k.startswith("close_")}; worst = min(close, key=close.get)
    near = stats(pools, ("close", "floor"))
    return {"near_r_median": near["median_r"], "soft_share": near["r_lt_1"], "worst_close": [worst, close[worst]],
            "near": near, "centre": stats(pools, ("centre",)), "cams": per}

def _one(job):
    """One eval camera: (name, kind, before, after), each side (r, worst, unpainted, owned by a detail view) off the sky."""
    C, src, rank, rank2, lifted, dsrc, sizes, step, name, kind, eye, look = job
    B = C.B; cam = (tuple(eye), tuple(look)); P, sid, n = seen(C, cam, lifted, step)
    sky = (sid == 4) if C.OUT else np.zeros(sid.shape, bool); sides = []
    def ring(rk):
        own = owner(C, src, rk, lifted, P, sid, n); r, worst = np.zeros(sid.shape), np.zeros(sid.shape)
        for v in np.unique(own[own >= 0]):
            m = own == v; r[m], worst[m] = ratio(B, P[m], n[m], src[v][0], focal(B, BASE), cam, FE)
        return own, r, worst
    own, r, worst = ring(rank); sides.append((r[~sky], worst[~sky], own[~sky] < 0, own[~sky] >= len(src)))
    if rank2 is not rank: own, r, worst = ring(rank2)
    k = sizes[0][0] / BASE[0]; own, r, worst = own.copy(), r * k, worst * k      # a smaller picture of the same view: fewer pixels everywhere alike
    if dsrc: take_over(C, dsrc, focal(B, sizes[1]), lifted, P, sid, n, cam, r, worst, own, len(src))
    sides.append((r[~sky], worst[~sky], own[~sky] < 0, own[~sky] >= len(src)))
    return name, kind, sides[0], sides[1]

def score(world_dir, plan, step=4, ready=None):
    """{"before", "after"} for a plan = {"views": [{"eye", "look"}], "ring": [w, h], "detail": [w, h], "drop": [ring view indices]}.
    ready: (C, world, src, count, lifted, the plan's views rendered) when the caller has them already."""
    C, world, src, count, lifted, dsrc = ready or (lambda C, w: (C, w, *sources(C, w), None))(*K.layout(world_dir))
    rank = ranks(count); rank2 = ranks(count, plan["drop"]) if plan.get("drop") else rank
    if dsrc is None: dsrc = detail_sources(C, plan["views"], lifted)
    sizes = (tuple(plan.get("ring", BASE)), tuple(plan.get("detail", BASE)))
    global _JOBS; _JOBS = [(C, src, rank, rank2, lifted, dsrc, sizes, step, *c) for c in K.layout_cams(world)]; res = _pool(_job, len(_JOBS))
    out = {}
    for side, j in (("before", 2), ("after", 3)):
        pools, per = {}, {}
        for r in res:
            pools.setdefault(r[1], []).append(r[j]); per[r[0]] = round(float(np.median(r[j][0])), 3) if len(r[j][0]) else None
        out[side] = summary(pools, per)
    return out

_JOBS = []
def _job(i): return _one(_JOBS[i])

if __name__ == "__main__":
    a = sys.argv[1:]; t0 = time.time(); plan = json.load(open(a[1]))
    out = score(a[0], plan, int(a[a.index("--step") + 1]) if "--step" in a else 4)
    print(json.dumps({"world": plan.get("world"), **out, "seconds": round(time.time() - t0, 2)}))
