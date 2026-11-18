"""The geometry score of evals/density.py, with three things made variable: which painted views exist, how large they
are painted, and extra "detail" views that own what they see head-on. No pictures are read, no GPU.
  score.py WORLD        the world as it was painted: prints the same near numbers as evals/density.py WORLD (a check)
The maths (sources, normal, seen, spread, ratio) is copied from evals/density.py as it stood on 2026-11-16, because that
folder is being edited; only evals/cams.py is imported, for the 24 eval cameras. What differs:
- scores() is density.owner() kept per view instead of reduced to the winner, so a world with some views left out is the
  lowest score among those kept. (Leaving views out never changes the order of the rest: a score is the view's rank for
  the surface plus whole multiples of the view count.)
- owned() says where a detail view would be the owner. The rule proposed for consistent.py: a detail view owns a point
  when the point is inside its picture and off the outer EDGE of it, is seen from within HEAD of head-on, is really
  visible, and is not ceiling, backdrop, a thing that moves away, the floor in such a thing's shadow, or a wall or floor
  point in the band round a shape's or a moving thing's outline. Detail views come before every ring view; between two
  detail views the one with more painted pixels there wins.
r is painted pixels per screen pixel, as in density.py. It counts pixels, not whether the painter fills them with detail.
"""
import sys, os, json, time, contextlib, numpy as np
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); TH = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TH, "evals"))
import cams as K
SCREEN, VFOV, STEP = (1600, 900), 65.0, 4      # the viewer's picture; one ray per STEP x STEP screen pixels
FE = SCREEN[1] / 2 / np.tan(np.radians(VFOV) / 2)
OFFS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
BASE, SECS = (768, 576), 65.0                  # what a ring view is painted at today, and what that takes
GROW = 1.565                                   # 1024x768 was measured at about 160 s: time grows faster than the pixel count
HEAD, EDGE, MAP = 0.5, 0.06, 128               # cosine a detail view must see a point within; the share of its picture left as a rim; width of its small surface map
FACES = ["x0", "floor", "z0", "x1", "ceiling", "z1"]

def seconds(size, secs=SECS):
    """Paint time of one picture. Above today's size by the one larger measurement; below it by pixel count, which is a
    guess (nothing smaller was timed) and leaves out whatever a picture costs regardless of size."""
    px = size[0] * size[1] / (BASE[0] * BASE[1])
    return secs * (px ** GROW if px > 1 else px)

@contextlib.contextmanager
def shot(B, w, h, fov):
    """blockout.py draws at one size and angle held in the module: lend it another for a moment."""
    old = B.W, B.H, B.FOV; B.W, B.H, B.FOV = int(w), int(h), float(fov)
    try: yield
    finally: B.W, B.H, B.FOV = old

def sources(C, world, fresh=False):
    """As density.sources. fresh: trace the painted views here instead of reading depth/NN.npz (for a ring that was never prepared)."""
    B = C.B; NS = 6 + len(C.objs)
    try: lifted = [i for i in json.load(open(f"{world}/gen_c/lifted.json")) if type(i) is int]
    except (OSError, ValueError): lifted = [] if os.path.isdir(f"{world}/gen_c") else list(C.LIFT)
    big = lambda m, px: ndimage.maximum_filter(m, px, mode="nearest")
    shaped = [6 + oi for oi in {*C.SHAPES, *(C.SLIM if C.NEW else ())}]
    src = []
    for i in range(C.N):
        if fresh: t, _, sid, _ = B.trace(B.CAMS[str(i)], C.PRIM_OBJ, True)
        else: t, sid, _ = C.surfaces(i)
        edge = np.zeros(sid.shape, bool)
        for dy, dx in OFFS: edge |= sid != np.roll(sid, (dy, dx), (0, 1))
        on = np.isin(sid, shaped)
        src.append((B.CAMS[str(i)], t, sid, edge, big(on, C.HALO) & ~on, big(np.isin(sid, [6 + oi for oi in lifted]), 81) if lifted else None))
    count = np.stack([np.bincount(s[2].ravel(), minlength=NS) for s in src], 1).astype(float)
    rank = np.argsort(np.argsort(-count, axis=1), axis=1).astype(float); rank[count < 400] = 1e6
    return src, rank, lifted

def normal(p, P):
    """Unit normals of a box or cylinder primitive at points P on its surface."""
    o = P - np.array(p["c"], float); n = np.zeros(P.shape)
    if p["shape"] == "cyl":
        cap = np.abs(o[:, 1]) > p["hy"] - 1e-4
        n[cap, 1] = 1; n[~cap] = o[~cap] * [1, 0, 1]
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    cs, sn = np.cos(p.get("yaw", 0.0)), np.sin(p.get("yaw", 0.0))
    l = np.stack([cs * o[:, 0] - sn * o[:, 2], o[:, 1], sn * o[:, 0] + cs * o[:, 2]], 1)
    k = np.argmax(np.abs(l) / np.array(p["h"], float), 1); e = np.eye(3)[k]
    return np.stack([cs * e[:, 0] + sn * e[:, 2], e[:, 1], -sn * e[:, 0] + cs * e[:, 2]], 1)

def seen(C, cam, lifted, step=STEP, size=None, fov=None):
    """As density.seen: what a camera sees of the blockout the splat was trained on: per traced pixel the point, the
    surface id and the normal. By default a screen camera traced every `step` pixels; size and fov (side to side) give
    any other picture, which is how a detail view is looked through."""
    B = C.B; keep = [j for j, oi in enumerate(C.PRIM_OBJ) if oi not in lifted]; full = B.PRIMS
    w, h = size or (SCREEN[0] // step, SCREEN[1] // step)
    fov = fov or 2 * np.degrees(np.arctan(np.tan(np.radians(VFOV) / 2) * SCREEN[0] / SCREEN[1]))
    B.PRIMS = [full[j] for j in keep]
    try:
        with shot(B, w, h, fov): t, P, pid, nrm = B.trace(cam, keep, True)
    finally: B.PRIMS = full
    n = nrm.astype(float); sid = pid.copy()
    for f in range(6): n[pid == f, f % 3] = 1
    for j in keep:
        m = pid == 6 + j; sid[m] = 6 + C.PRIM_OBJ[j]
        if m.any() and full[j]["shape"] != "mesh": n[m] = normal(full[j], P[m])
    return P, sid, n

def shade(C, lifted, P, sid):
    """Floor under and round a lifted thing: it holds the thing's painted shadow and is filled in, whoever painted it."""
    m = np.zeros(sid.shape, bool)
    for lo, hi in (C.bounds(C.OBJ_PRIMS[oi]) for oi in lifted):
        if lo[1] < 0.6:
            gap = np.hypot(np.maximum(np.maximum(lo[0] - P[..., 0], P[..., 0] - hi[0]), 0), np.maximum(np.maximum(lo[2] - P[..., 2], P[..., 2] - hi[2]), 0))
            m |= (sid == 1) & (gap < float(np.clip(hi[1], 0.4, 2.0)))
    return m

def scores(C, src, rank, lifted, P, sid, n):
    """Per painted view, the score consistent.py's render() gives it at each point (the lowest wins); 1e6 where the view
    cannot be the source. density.owner() with every view's answer kept."""
    B = C.B; H, W = src[0][2].shape; N = len(src); NS = rank.shape[0]
    mesh = np.isin(np.arange(NS), [6 + oi for oi in C.SHAPES]); on = mesh[sid]
    S = np.full((N,) + sid.shape, 1e6, np.float32)
    for v in range(N):
        cam, st, ssid, sedge, shalo, slift = src[v]
        score = rank[sid, v]
        if on.any():
            to = np.array(cam[0]) - P[on]; cos = (to * n[on]).sum(1) / np.maximum(np.linalg.norm(to, axis=1), 1e-9)
            score[on] += N * (cos < C.GRAZE); score[on] = np.where(cos <= 0, 1e6, score[on])
        with np.errstate(all="ignore"):
            px, py, dist, z = B.project(cam, P)
            xi, yi = np.round(px).astype(int), np.round(py).astype(int)
        inside = (z > 0.05) & (xi >= 1) & (xi < W - 1) & (yi >= 1) & (yi < H - 1)
        xi, yi = np.clip(xi, 0, W - 1), np.clip(yi, 0, H - 1)
        if mesh.any(): score = score + N * shalo[yi, xi]
        if lifted: score = np.where(slift[yi, xi], 1e6, score)
        want = (score < 1e5) & inside
        plain = want & (sid < 6) & ~sedge[yi, xi]
        ok = plain & (ssid[yi, xi] == sid) & (np.abs(st[yi, xi] - dist) < np.maximum(0.05, 0.01 * dist))
        ex = np.nonzero(want & ~plain)
        if len(ex[0]):
            ey, ex_x, s = yi[ex], xi[ex], sid[ex]
            vis = B.visible(cam[0], P[ex]); miss = vis & (ssid[ey, ex_x] != s)
            for dy, dx in OFFS:
                if not miss.any(): break
                m = miss & (ssid[np.clip(ey + dy, 0, H - 1), np.clip(ex_x + dx, 0, W - 1)] == s)
                ey[m] += dy; ex_x[m] += dx; miss &= ~m
            ok[ex] = vis & ~miss
        S[v][ok] = score[ok]
    return S

def ring_owner(C, S, keep, dead):
    """The painted view each point takes its paint from when only the views in `keep` exist; -1 where none.
    dead: points that are filled in whatever was painted (shade, and the sky outdoors)."""
    keep = np.asarray(keep); Sk = S[keep]; j = Sk.argmin(0)
    own = np.where(np.take_along_axis(Sk, j[None], 0)[0] < 1e5, keep[j], -1); own[dead] = -1
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
    """Two unit directions lying in the surface at each normal n: (.., 3, 2)."""
    ax = np.eye(3)[np.argmin(np.abs(n), -1)]
    t1 = np.cross(n, ax); t1 /= np.linalg.norm(t1, axis=-1, keepdims=True); return np.stack([t1, np.cross(n, t1)], -1)

def mix(Jo, Je):
    """(r, worst) from how the painted picture's pixels (Jo) and the screen's (Je) lie on the surface."""
    de = Je[..., 0, 0] * Je[..., 1, 1] - Je[..., 0, 1] * Je[..., 1, 0]
    de = np.where(np.abs(de) < 1e-12, 1e-12, de)[..., None, None]
    M = Jo @ (np.stack([np.stack([Je[..., 1, 1], -Je[..., 0, 1]], -1), np.stack([-Je[..., 1, 0], Je[..., 0, 0]], -1)], -2) / de)
    det = np.abs(M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]); q = (M ** 2).sum((-1, -2))
    return np.sqrt(det), np.sqrt(np.maximum(q - np.sqrt(np.maximum(q * q - 4 * det * det, 0)), 0) / 2)

def ratio(B, P, n, src_cam, fo, cam, fe=FE):
    """(r, worst) at points P with normals n: painted pixels per screen pixel, by area and along the thinnest direction.
    (density.ratio, split in two so the screen's half can be worked out once and kept.)"""
    T = tangents(n); return mix(spread(B, src_cam, fo, P, T), spread(B, cam, fe, P, T))

def focal(size, fov): return size[0] / 2 / np.tan(np.radians(fov) / 2)

def look_through(C, d, lifted):
    """A detail view's own small picture of the blockout as it would be painted (moving things in place): surface ids,
    and the two bands its paint is not trusted in. Stored on the view; plain arrays, so it survives a trip to a worker."""
    B = C.B; w = MAP; h = max(8, round(MAP * d["size"][1] / d["size"][0]))
    with shot(B, w, h, d["fov"]): _, _, sid = B.trace((d["eye"], d["look"]), C.PRIM_OBJ)
    big = lambda m, px: ndimage.maximum_filter(m, int(np.ceil(px * w / 1024)) | 1, mode="nearest")
    on = np.isin(sid, [6 + oi for oi in {*C.SHAPES, *(C.SLIM if C.NEW else ())}])
    d["sid"] = sid.astype(np.int16); d["halo"] = big(on, C.HALO) & ~on
    d["lift"] = big(np.isin(sid, [6 + oi for oi in lifted]), 81) if lifted else np.zeros(sid.shape, bool)
    return d

def owned(C, d, lifted, P, sid, n, dead):
    """Where detail view d would be the owner (the rule at the top of this file). P (m, 3), sid (m,), n (m, 3)."""
    B = C.B; eye = np.array(d["eye"], float); h, w = d["sid"].shape
    m = (sid != 4) & ~dead & ~np.isin(sid, [6 + oi for oi in lifted])
    if C.OUT: m &= ~np.isin(sid, C.BACKDROP)
    to = eye - P; cos = (to * n).sum(-1) / np.maximum(np.linalg.norm(to, axis=-1), 1e-9)
    m &= np.where(sid < 6, np.abs(cos), cos) >= HEAD            # a room face's normal is stored unsigned
    with shot(B, w, h, d["fov"]), np.errstate(all="ignore"): px, py, _, z = B.project((d["eye"], d["look"]), P)
    nx, ny = (px + 0.5) / w, (py + 0.5) / h
    m &= (z > 0.05) & (nx >= EDGE) & (nx <= 1 - EDGE) & (ny >= EDGE) & (ny <= 1 - EDGE)
    xi, yi = np.clip(np.nan_to_num(px).round().astype(int), 0, w - 1), np.clip(np.nan_to_num(py).round().astype(int), 0, h - 1)
    m &= ~d["lift"][yi, xi] & ~(d["halo"][yi, xi] & (sid < 6))
    j = np.nonzero(m)[0]
    if len(j): m[j] = B.visible(eye, P[j])
    return m

def field(C, src, lifted, cam, S, P, sid, n, keep, size, details):
    """(r, worst, owner) over one screen camera's traced pixels. keep: the ring views that exist; size: what they are
    painted at; details: detail views (owner index = len(src) + its place in the list)."""
    B = C.B; dead = shade(C, lifted, P, sid)
    if C.OUT: dead |= sid == 4
    own = ring_owner(C, S, keep, dead); r, worst = np.zeros(sid.shape), np.zeros(sid.shape); fo = focal(size, B.FOV)
    for v in np.unique(own[own >= 0]):
        m = own == v; r[m], worst[m] = ratio(B, P[m], n[m], src[v][0], fo, cam)
    if details:
        Pf, sf, nf, df = P.reshape(-1, 3), sid.ravel(), n.reshape(-1, 3), dead.ravel()
        rd, wd, od = np.zeros(sf.shape), np.zeros(sf.shape), np.full(sf.shape, -1)
        for k, d in enumerate(details):
            j = np.nonzero(owned(C, d, lifted, Pf, sf, nf, df))[0]
            if not len(j): continue
            a, b = ratio(B, Pf[j], nf[j], (d["eye"], d["look"]), focal(d["size"], d["fov"]), cam)
            up = a > rd[j]; j = j[up]; rd[j], wd[j], od[j] = a[up], b[up], len(src) + k
        got = (od >= 0).reshape(sid.shape)
        r[got], worst[got], own[got] = rd.reshape(sid.shape)[got], wd.reshape(sid.shape)[got], od.reshape(sid.shape)[got]
    return r, worst, own

def summary(rows):
    """density.py's numbers from [(name, kind, r, worst, unpainted)], each already without the sky."""
    cat = lambda kinds, j: np.concatenate([x[j] for x in rows if x[1] in kinds])
    def stats(kinds):
        r, w, none = (cat(kinds, j) for j in (2, 3, 4))
        return {"median_r": round(float(np.median(r)), 3), "r_lt_1": round(float((r < 1).mean()), 4), "r_lt_0.5": round(float((r < 0.5).mean()), 4),
                "median_worst": round(float(np.median(w)), 3), "worst_lt_1": round(float((w < 1).mean()), 4), "unpainted": round(float(none.mean()), 4)}
    per = {x[0]: round(float(np.median(x[2])), 3) if len(x[2]) else None for x in rows}
    close = {k: v for k, v in per.items() if k.startswith("close_") and v is not None}; low = min(close, key=close.get)
    return {"near": stats(("close", "floor")), "centre": stats(("centre",)), "close": stats(("close",)), "floor": stats(("floor",)),
            "worst_close_cam": [low, close[low]], "cams": per}

G = {}      # what the forked workers read: set before the pool is made
def _eval(i):
    C, src, lifted = G["C"], G["src"], G["lifted"]; name, kind, eye, look = G["cams"][i]; cam = (tuple(eye), tuple(look))
    P, sid, n = seen(C, cam, lifted); S = scores(C, src, G["rank"], lifted, P, sid, n); out = []
    sky = (sid == 4) if C.OUT else np.zeros(sid.shape, bool)
    for keep, size, details in G["variants"]:
        r, worst, own = field(C, src, lifted, cam, S, P, sid, n, keep, size, details)
        out.append((name, kind, r[~sky].astype(np.float32), worst[~sky].astype(np.float32), own[~sky] < 0))
    return out

def pool_map(fn, n):
    import multiprocessing as mp
    k = max(1, min(8, (os.cpu_count() or 4) - 2, n))
    if k == 1: return [fn(i) for i in range(n)]
    with mp.get_context("fork").Pool(k) as pool: return pool.map(fn, range(n), chunksize=1)

def evaluate(C, world, src, rank, lifted, variants, cams=None):
    """The score on the 24 eval cameras for each variant (keep, size, details): a list of summary()."""
    G.update(C=C, src=src, rank=rank, lifted=lifted, variants=variants, cams=cams or K.layout_cams(world))
    res = pool_map(_eval, len(G["cams"]))
    return [summary([row[k] for row in res]) for k in range(len(variants))]

if __name__ == "__main__":
    t0 = time.time(); C, world = K.layout(sys.argv[1]); src, rank, lifted = sources(C, world)
    try: size = json.load(open(f"{world}/gen/views.json"))["paint"]
    except (OSError, KeyError, ValueError): size = list(BASE)
    s = evaluate(C, world, src, rank, lifted, [(list(range(C.N)), size, [])])[0]
    print(json.dumps({"world": os.path.basename(world), "paint": size, "near": s["near"], "worst_close_cam": s["worst_close_cam"], "cams": s["cams"], "seconds": round(time.time() - t0, 2)}))
