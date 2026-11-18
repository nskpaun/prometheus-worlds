"""The geometry score of evals/density.py, with extra "detail" views allowed to own what they see. No pictures, no GPU.
  score.py WORLD [--paint WxH] [--no-down]     prints the score of the world as it is painted now (one JSON line)
sources, normal, seen and owner are copied from evals/density.py (16 Nov 2026) so that edits there do not move these
numbers; ratio is split in two (the screen's half is worked out once per point, the painted view's half per view), which
gives the same r. Checked: with no detail views the near numbers equal density.py's on ex-tower, garage-shapes, ex-blossom.
New here: view() and take(). A detail view owns a point when
  the point lies in its picture, away from the outer SEAM of it (left for blending into the paint around);
  its blockout render shows that surface there at that depth (so not what stands behind an object);
  it sees the surface within 60 degrees of head-on (HEAD), a shape's surface from its outer side;
  the point is not in the band round a shape's outline (for a wall or floor) or round a thing that moves away;
  and it brings more painted pixels per screen pixel than the owner so far (so it never makes anything coarser).
--no-down scores the world as if the 4 to 6 downward ring views had not been painted (their time can pay for detail views).
"""
import sys, os, json, time, contextlib, multiprocessing as mp, numpy as np
from PIL import Image
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); TH = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TH, "evals"))
import cams as K                                # read only: the 24 fixed cameras and the world loader
SCREEN, VFOV, STEP = (1600, 900), 65.0, 4       # the viewer's picture; one ray per STEP x STEP screen pixels
FE = SCREEN[1] / 2 / np.tan(np.radians(VFOV) / 2)
OFFS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
SEAM, HEAD = 0.04, 0.5      # share of a detail picture's width and height left unowned at each edge; least cosine to its camera
NAMES = ["wall x0", "floor", "wall z0", "wall x1", "ceiling", "wall z1"]
big = lambda m, px: ndimage.maximum_filter(m, int(px) | 1, mode="nearest")

@contextlib.contextmanager
def size(B, w, h):
    """blockout renders and projects at one global size: borrow it."""
    old = (B.W, B.H); B.W, B.H = w, h
    try: yield
    finally: B.W, B.H = old

def focal(B, width): return width / 2 / np.tan(np.radians(B.FOV) / 2)      # blockout's angle runs side to side

# ---- copied from evals/density.py ----
def sources(C, world, drop=()):
    """What consistent.py knows of the painted views; drop: views treated as never painted."""
    B = C.B; NS = 6 + len(C.objs)
    try: lifted = [i for i in json.load(open(f"{world}/gen_c/lifted.json")) if type(i) is int]
    except (OSError, ValueError): lifted = [] if os.path.isdir(f"{world}/gen_c") else list(C.LIFT)
    shaped = [6 + oi for oi in {*C.SHAPES, *(C.SLIM if C.NEW else ())}]
    src = []
    for i in range(C.N):
        t, sid, _ = C.surfaces(i); edge = np.zeros(sid.shape, bool)
        for dy, dx in OFFS: edge |= sid != np.roll(sid, (dy, dx), (0, 1))
        on = np.isin(sid, shaped)
        src.append((B.CAMS[str(i)], t, sid, edge, big(on, C.HALO) & ~on, big(np.isin(sid, [6 + oi for oi in lifted]), 81) if lifted else None))
    count = np.stack([np.bincount(s[2].ravel(), minlength=NS) for s in src], 1).astype(float)
    count[:, list(drop)] = 0
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

def seen(C, cam, lifted, step=STEP):
    """What a screen camera sees of the blockout the splat was trained on: point, surface id, normal per traced pixel."""
    B = C.B; keep = [j for j, oi in enumerate(C.PRIM_OBJ) if oi not in lifted]
    full, fov = B.PRIMS, B.FOV
    B.PRIMS = [full[j] for j in keep]
    B.FOV = 2 * np.degrees(np.arctan(np.tan(np.radians(VFOV) / 2) * SCREEN[0] / SCREEN[1]))
    try:
        with size(B, SCREEN[0] // step, SCREEN[1] // step): t, P, pid, nrm = B.trace(cam, keep, True)
    finally: B.PRIMS = full; B.FOV = fov
    n = nrm.astype(float); sid = pid.copy()
    for f in range(6): n[pid == f, f % 3] = 1
    for j in keep:
        m = pid == 6 + j; sid[m] = 6 + C.PRIM_OBJ[j]
        if m.any() and full[j]["shape"] != "mesh": n[m] = normal(full[j], P[m])
    return P, sid, n

def owner(C, src, rank, lifted, P, sid, n):
    """The painted view each point takes its paint from, -1 where none: consistent.py's render(), without the colours."""
    B = C.B; H, W = src[0][2].shape; N = len(src); NS = rank.shape[0]
    mesh = np.isin(np.arange(NS), [6 + oi for oi in C.SHAPES]); on = mesh[sid]
    best = np.full(sid.shape, 1e5); own = np.full(sid.shape, -1)
    for v in np.argsort((np.minimum(rank, N) * np.bincount(sid.ravel(), minlength=NS)[:, None]).sum(0)):
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
        want = (score < best) & inside
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
        best = np.where(ok, score, best); own[ok] = v
    own[shade(C, lifted, P, sid)] = -1      # the floor round a lifted thing holds its painted shadow and is filled in instead
    if C.OUT: own[sid == 4] = -1
    return own

def shade(C, lifted, P, sid):
    m = np.zeros(sid.shape, bool)
    for lo, hi in (C.bounds(C.OBJ_PRIMS[oi]) for oi in lifted):
        if lo[1] < 0.6:
            gap = np.hypot(np.maximum(np.maximum(lo[0] - P[..., 0], P[..., 0] - hi[0]), 0), np.maximum(np.maximum(lo[2] - P[..., 2], P[..., 2] - hi[2]), 0))
            m |= (sid == 1) & (gap < float(np.clip(hi[1], 0.4, 2.0)))
    return m

def spread(B, cam, f, P, T):
    """How a picture's pixels lie on the surface at points P: the 2x2 matrix of pixels per metre along the two surface
    directions T. cam: (eye, look); f: focal length in pixels."""
    eye, r, u, fw = B.basis(cam); v = P - eye; z = (v @ fw)[..., None, None]
    a = np.stack([v @ r, v @ u], -1)[..., :, None]
    along = lambda d: np.einsum("...ik,i->...k", T, d)
    Tc = np.stack([along(r), along(u)], -2)
    return f * (Tc * z - a * along(fw)[..., None, :]) / z ** 2

def screen(B, P, n, cam):
    """The screen's half of density.py's ratio(): the surface directions T at each point and the inverse of the screen's
    pixels-per-metre matrix there."""
    ax = np.eye(3)[np.argmin(np.abs(n), -1)]
    t1 = np.cross(n, ax); t1 /= np.linalg.norm(t1, axis=-1, keepdims=True); T = np.stack([t1, np.cross(n, t1)], -1)
    Je = spread(B, cam, FE, P, T)
    de = Je[..., 0, 0] * Je[..., 1, 1] - Je[..., 0, 1] * Je[..., 1, 0]
    de = np.where(np.abs(de) < 1e-12, 1e-12, de)[..., None, None]
    return T, np.stack([np.stack([Je[..., 1, 1], -Je[..., 0, 1]], -1), np.stack([-Je[..., 1, 0], Je[..., 0, 0]], -1)], -2) / de

def dens(B, cam, f, P, T, Jinv):
    """(r, worst): painted pixels per screen pixel from the painted view `cam`, by area and along the thinnest direction."""
    M = spread(B, cam, f, P, T) @ Jinv
    det = np.abs(M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]); q = (M ** 2).sum((-1, -2))
    return np.sqrt(det), np.sqrt(np.maximum(q - np.sqrt(np.maximum(q * q - 4 * det * det, 0)), 0) / 2)

# ---- new: points to judge by, and detail views ----
def samples(C, src, rank, lifted, f_ring, *sets):
    """Per set (cameras [(name, kind, eye, look)], step): every traced screen pixel of those cameras as one flat table,
    with its ring owner and r. Owners are found camera by camera, as density.py does (the order views are tried in can
    settle a tie), on several cores: forked workers share the painted views' depth maps without copying, as in
    consistent.py. All sets go through one pool: each worker builds its ray-casting scenes once."""
    global JOB
    todo = [(c, step) for cams, step in sets for c in cams]
    JOB = (C, src, rank, lifted, f_ring, todo); n = max(1, min(8, (os.cpu_count() or 4) - 2, len(todo)))
    if n > 1:
        with mp.get_context("fork").Pool(n) as pool: got = pool.map(one, range(len(todo)), chunksize=1)
    else: got = [one(i) for i in range(len(todo))]
    out = []
    for cams, _ in sets:
        mine, got = got[:len(cams)], got[len(cams):]
        S = dict(zip(("P", "sid", "n", "T", "Jinv", "own", "r", "worst"), (np.concatenate(a) for a in zip(*mine))))
        S["cam"] = np.repeat(np.arange(len(cams)), [len(g[1]) for g in mine])
        S["sky"] = (S["sid"] == 4) if C.OUT else np.zeros(len(S["sid"]), bool)
        S["hold"] = S["sky"] | shade(C, lifted, S["P"], S["sid"])      # never given to a detail view
        out.append(S)
    return out

def one(i):
    """samples() for one camera."""
    C, src, rank, lifted, f_ring, todo = JOB; B = C.B; (_, _, eye, look), step = todo[i]
    cam = (tuple(eye), tuple(look)); P, sid, n = (a.reshape(-1, *a.shape[2:]) for a in seen(C, cam, lifted, step))
    T, Jinv = screen(B, P, n, cam); own = owner(C, src, rank, lifted, P, sid, n)
    r, worst = np.zeros(len(sid)), np.zeros(len(sid))
    for v in np.unique(own[own >= 0]):
        m = own == v; r[m], worst[m] = dens(B, src[v][0], f_ring, P[m], T[m], Jinv[m])
    return P, sid, n, T, Jinv, own, r, worst

def view(C, cam, lifted, w):
    """A detail view's blockout render at width w (4:3): what take() needs to know of it."""
    B = C.B; h = w * 3 // 4
    with size(B, w, h): t, _, sid = B.trace(cam, C.PRIM_OBJ)
    k = w / B.W; on = np.isin(sid, [6 + oi for oi in {*C.SHAPES, *(C.SLIM if C.NEW else ())}])
    return {"cam": cam, "t": t, "sid": sid, "halo": big(on, C.HALO * k) & ~on,
            "lift": big(np.isin(sid, [6 + oi for oi in lifted]), 81 * k) if lifted else None}

def take(C, V, f, S, near=None):
    """Where detail view V (painted with focal length f) may own the points of S: (indices, r, worst).
    near: only points within this many metres of its camera are looked at (a quick first cut for planning)."""
    B = C.B; eye = np.array(V["cam"][0]); h, w = V["sid"].shape
    idx = np.nonzero(~S["hold"] & (((S["P"] - eye) ** 2).sum(1) < near * near if near else True))[0]
    P, sid, n = S["P"][idx], S["sid"][idx], S["n"][idx]
    with size(B, w, h), np.errstate(all="ignore"):
        px, py, dist, z = B.project(V["cam"], P)
        xi, yi = np.round(px).astype(int), np.round(py).astype(int)
    ok = (z > 0.05) & (xi >= SEAM * w) & (xi < w - SEAM * w) & (yi >= SEAM * h) & (yi < h - SEAM * h)
    idx, P, sid, n, xi, yi, dist = (a[ok] for a in (idx, P, sid, n, xi, yi, dist))
    cos = ((eye - P) * n).sum(1) / np.maximum(dist, 1e-9)
    ok = (V["sid"][yi, xi] == sid) & (np.abs(V["t"][yi, xi] - dist) < np.maximum(0.05, 0.01 * dist)) & (np.where(sid < 6, np.abs(cos), cos) >= HEAD)
    ok &= ~((sid < 6) & V["halo"][yi, xi])
    if V["lift"] is not None: ok &= ~V["lift"][yi, xi]
    idx = idx[ok]
    r, worst = dens(B, V["cam"], f, S["P"][idx], S["T"][idx], S["Jinv"][idx])
    return idx, r, worst

def apply(S, takes):
    """S with detail views owning what they improve; takes: take()'s answer per view, in order. -> (r, worst, view or -1)"""
    r, worst, by = S["r"].copy(), S["worst"].copy(), np.full(len(S["r"]), -1)
    for k, (idx, rk, wk) in enumerate(takes):
        m = rk > r[idx]; r[idx[m]], worst[idx[m]], by[idx[m]] = rk[m], wk[m], k
    return r, worst, by

def stats(cams, S, r, worst, by=None):
    """density.py's "near" numbers (close and floor cameras together), the worst close camera, and each camera's median."""
    kind = np.array([c[1] for c in cams])[S["cam"]]; live = ~S["sky"]
    per = {c[0]: round(float(np.median(r[live & (S["cam"] == i)])), 3) for i, c in enumerate(cams) if (live & (S["cam"] == i)).any()}
    m = live & np.isin(kind, ("close", "floor")); close = {k: v for k, v in per.items() if k.startswith("close_")}
    low = min(close, key=lambda k: (close[k], k))
    out = {"near_r": round(float(np.median(r[m])), 3), "soft_share": round(float((r[m] < 1).mean()), 4), "r_lt_0.5": round(float((r[m] < 0.5).mean()), 4),
           "near_worst": round(float(np.median(worst[m])), 3), "unpainted": round(float(((S["own"] < 0) & (True if by is None else by < 0))[m].mean()), 4),
           "worst_close": [low, close[low]], "centre_r": round(float(np.median(r[live & (kind == "centre")])), 3)}
    if by is not None: out["detail_share"] = round(float((by[m] >= 0).mean()), 4)
    return out, per

def load(world_dir, paint=None, no_down=False):
    """(chisel_ring for the world, its folder, sources, rank, lifted, the ring pictures' size)."""
    C, world = K.layout(world_dir)
    src, rank, lifted = sources(C, world, range(C.RING, C.N) if no_down else ())
    if paint is None:
        try: paint = list(Image.open(f"{world}/gen/images/00.png").size)
        except OSError: paint = json.load(open(f"{world}/gen/views.json"))["paint"]
    return C, world, src, rank, lifted, list(paint)

if __name__ == "__main__":
    a = sys.argv[2:]; t0 = time.time()
    C, world, src, rank, lifted, paint = load(sys.argv[1], [int(v) for v in a[a.index("--paint") + 1].split("x")] if "--paint" in a else None, "--no-down" in a)
    cams = K.layout_cams(world); S, = samples(C, src, rank, lifted, focal(C.B, paint[0]), (cams, STEP))
    st, per = stats(cams, S, S["r"], S["worst"])
    print(json.dumps({"world": os.path.basename(world), "paint": paint, **st, "cams": per, "seconds": round(time.time() - t0, 2)}))
