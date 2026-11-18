"""One scorer for every detail-pass plan, so the three planners are judged by the same rule. No pictures, no GPU.
  judge.py WORLD [TAG=PLAN.json ...]     (run from thread/ with .venv/bin/python; PLAN.json: what a plan.py printed,
                                          from detail_a, detail_b, detail_c or pick.py here; TAG names it in the output)
Prints one JSON object: "baseline" (the world as painted) and one entry per plan with
  eval   evals/density.py's numbers over the 24 eval cameras (near = close + floor), as that file stands today: medians
         and shares over painted pixels, the unpainted share beside them; "median_r0" is the median with no paint
         counted as 0, which is how that file stood when the three planners copied it (their reports use it)
  walk   the same over a dense set of standing spots the planners never saw (walk_cams), all pixels and those within
         2.5 m of the eye. "sharp" is the mean of min(r, 1) with no paint counted as 0: one number that cannot be
         raised by leaving surfaces unpainted, which a median over painted pixels can
  steps  the share of neighbouring screen pixels on one surface whose r differs by more than 1.5 times: where a seam
         between a coarse and a fine owner would show
  eff    near median of min(r, training-view r): what today's training cameras could pass on to the splat
  paint_seconds, and per view "checks": is the camera in free space, what fills its picture, how much of the picture
  it may own by the rule below, how many times finer than the ring it is there, and what a painter would make of it.
The maths (sources, normal, seen, the owner tests, spread, ratio, train_ratio) is copied from evals/density.py as it
stood on 16 Nov 2026 late evening: that folder is being edited. Only evals/cams.py is imported.
The one ownership rule every plan is held to (none of the three planners' own): a detail view owns a point when
  the point is in its picture and not in the outer EDGE of it (room to fade into the paint around),
  it really sees it (an exact shadow ray, moving things in place as the painter sees them),
  within HEAD of head-on,
  it is not ceiling, sky, an outdoor backdrop, a thing that moves away, the floor in such a thing's shadow, nor a wall
  or floor point in the band round a shape's outline or round a moving thing,
  and it brings more painted pixels per screen pixel than the owner so far (it never makes anything coarser).
Leaving ring views out: the rest keep their order (a view's score is its rank plus whole multiples of the view count).
"""
import sys, os, json, time, contextlib, multiprocessing as mp, numpy as np
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); TH = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TH, "evals"))
import cams as K
SCREEN, VFOV = (1600, 900), 65.0
FE = SCREEN[1] / 2 / np.tan(np.radians(VFOV) / 2)
SFOV = 2 * np.degrees(np.arctan(np.tan(np.radians(VFOV) / 2) * SCREEN[0] / SCREEN[1]))     # blockout's angle runs side to side
OFFS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
BASE, SECS, GROW = (768, 576), 65.0, 1.565     # the ring's size and seconds a view (the brief's figure); 1024x768 took 160 s
EDGE, HEAD, MAP = 0.05, 0.5, 256               # rim of a detail picture left unowned; least cosine; width its blockout is looked at
NEARBY = 2.5                                   # metres: "up close" in the walk set
BINS = np.linspace(-7, 3, 1001)                # log2 r, for medians over the walk set without keeping every pixel
FACES = ["x0", "floor", "z0", "x1", "ceiling", "z1"]
G = {}                                         # what forked workers read

def seconds(size, secs=SECS):
    """Seconds to paint one picture: by pixel count below today's size (a guess: nothing smaller was timed), by the one
    larger measurement above it."""
    px = size[0] * size[1] / (BASE[0] * BASE[1]); return secs * (px ** GROW if px > 1 else px)

def focal(size, fov): return size[0] / 2 / np.tan(np.radians(fov) / 2)

@contextlib.contextmanager
def shot(B, w, h, fov):
    """blockout draws at one size and angle held in the module: lend it another for a moment."""
    old = B.W, B.H, B.FOV; B.W, B.H, B.FOV = int(w), int(h), float(fov)
    try: yield
    finally: B.W, B.H, B.FOV = old

# ---- copied from evals/density.py ----
def sources(C, world):
    B = C.B; NS = 6 + len(C.objs)
    try: lifted = [i for i in json.load(open(f"{world}/gen_c/lifted.json")) if type(i) is int]
    except (OSError, ValueError): lifted = [] if os.path.isdir(f"{world}/gen_c") else list(C.LIFT)
    big = lambda m, px: ndimage.maximum_filter(m, px, mode="nearest")
    shaped = [6 + oi for oi in {*C.SHAPES, *(C.SLIM if C.NEW else ())}]
    src = []
    for i in range(C.N):
        t, sid, _ = C.surfaces(i); edge = np.zeros(sid.shape, bool)
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

def seen(C, cam, lifted, w, h, fov):
    """What a camera sees of the blockout without the things in `lifted`: per pixel the point, surface id and normal."""
    B = C.B; keep = [j for j, oi in enumerate(C.PRIM_OBJ) if oi not in lifted]; full = B.PRIMS
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

def scores(C, src, rank, lifted, P, sid, n):
    """density.owner() with every ring view's answer kept: per view its score at each point (the lowest wins), 1e6
    where it cannot be the source. P (m, 3)."""
    B = C.B; H, W = src[0][2].shape; N = len(src); NS = rank.shape[0]
    mesh = np.isin(np.arange(NS), [6 + oi for oi in C.SHAPES]); on = mesh[sid]
    S = np.full((N, len(sid)), 1e6, np.float32)
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
        ex = np.nonzero(want & ~plain)[0]
        if len(ex):
            ey, ex_x, s = yi[ex], xi[ex], sid[ex]
            vis = B.visible(cam[0], P[ex]); miss = vis & (ssid[ey, ex_x] != s)
            for dy, dx in OFFS:
                if not miss.any(): break
                m = miss & (ssid[np.clip(ey + dy, 0, H - 1), np.clip(ex_x + dx, 0, W - 1)] == s)
                ey[m] += dy; ex_x[m] += dx; miss &= ~m
            ok[ex] = vis & ~miss
        S[v, ok] = score[ok]
    return S

def shade(C, lifted, P, sid):
    """Floor under and round a lifted thing: it holds the thing's painted shadow and is filled in, whoever painted it."""
    m = np.zeros(sid.shape, bool)
    for lo, hi in (C.bounds(C.OBJ_PRIMS[oi]) for oi in lifted):
        if lo[1] < 0.6:
            gap = np.hypot(np.maximum(np.maximum(lo[0] - P[..., 0], P[..., 0] - hi[0]), 0), np.maximum(np.maximum(lo[2] - P[..., 2], P[..., 2] - hi[2]), 0))
            m |= (sid == 1) & (gap < float(np.clip(hi[1], 0.4, 2.0)))
    return m

def spread(B, cam, f, P, T):
    """How a picture's pixels lie on the surface at points P: pixels per metre along the two surface directions T."""
    eye, r, u, fw = B.basis(cam); v = P - eye; z = (v @ fw)[..., None, None]
    a = np.stack([v @ r, v @ u], -1)[..., :, None]
    along = lambda d: np.einsum("...ik,i->...k", T, d)
    Tc = np.stack([along(r), along(u)], -2)
    return f * (Tc * z - a * along(fw)[..., None, :]) / z ** 2

def tangents(n):
    ax = np.eye(3)[np.argmin(np.abs(n), -1)]
    t1 = np.cross(n, ax); t1 /= np.linalg.norm(t1, axis=-1, keepdims=True); return np.stack([t1, np.cross(n, t1)], -1)

def mix(Jo, Je):
    """(r, worst) from how the painted picture's pixels (Jo) and the screen's (Je) lie on the surface."""
    de = Je[..., 0, 0] * Je[..., 1, 1] - Je[..., 0, 1] * Je[..., 1, 0]
    de = np.where(np.abs(de) < 1e-12, 1e-12, de)[..., None, None]
    M = Jo @ (np.stack([np.stack([Je[..., 1, 1], -Je[..., 0, 1]], -1), np.stack([-Je[..., 1, 0], Je[..., 0, 0]], -1)], -2) / de)
    det = np.abs(M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]); q = (M ** 2).sum((-1, -2))
    return np.sqrt(det), np.sqrt(np.maximum(q - np.sqrt(np.maximum(q * q - 4 * det * det, 0)), 0) / 2)

def per_metre(B, cam, f, P, n):
    """Pixels per metre of surface a picture lays at points P (the square root of pixels per m2)."""
    J = spread(B, cam, f, P, tangents(n)); return np.sqrt(np.abs(J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]))

def train_ratio(B, P, n, sid, tcams, ft, cam, fe):
    """Training-view pixels per screen pixel: the finest of the training cameras that face the point and hold it in frame."""
    P = P.astype(np.float32); n = n.astype(np.float32); best = np.zeros(len(P), np.float32)
    lay = lambda c: (lambda eye, r, u, fw: (P - np.float32(eye), np.float32(r), np.float32(u), np.float32(fw)))(*B.basis(c))
    for c in tcams:
        v, r, u, fw = lay(c); z = v @ fw; nv = -(v * n).sum(1)
        with np.errstate(all="ignore"):
            ok = (z > 0.05) & (np.abs(v @ r) * ft < z * B.W / 2) & (np.abs(v @ u) * ft < z * B.H / 2) & ((nv > 0) | (sid < 6))
            best = np.maximum(best, np.where(ok, ft * ft * np.abs(nv) / z ** 3, 0))
    v, r, u, fw = lay(cam)
    return np.sqrt(best / np.maximum(fe * fe * np.abs((v * n).sum(1)) / (v @ fw) ** 3, 1e-9))

# ---- detail views ----
def look_through(C, d, lifted):
    """A detail view's own small picture of the blockout as it would be painted (moving things in place): the bands its
    paint is not trusted in."""
    B = C.B; w = MAP; h = max(8, round(MAP * d["size"][1] / d["size"][0])); cam = (tuple(d["eye"]), tuple(d["look"]))
    with shot(B, w, h, d["fov"]): t, _, sid = B.trace(cam, C.PRIM_OBJ)
    big = lambda m, px: ndimage.maximum_filter(m, int(np.ceil(px * w / 1024)) | 1, mode="nearest")
    on = np.isin(sid, [6 + oi for oi in {*C.SHAPES, *(C.SLIM if C.NEW else ())}])
    return dict(d, cam=cam, t=t, sid=sid, halo=big(on, C.HALO) & ~on, lift=big(np.isin(sid, [6 + oi for oi in lifted]), 81) if lifted else np.zeros(sid.shape, bool))

def owned(C, d, lifted, P, sid, n, dead):
    """Where detail view d may own points (all of the rule but "finer than the owner so far"). P (m, 3)."""
    B = C.B; eye = np.array(d["eye"], float); h, w = d["sid"].shape
    m = (sid != 4) & ~dead & ~np.isin(sid, [6 + oi for oi in lifted])
    if C.OUT: m &= ~np.isin(sid, C.BACKDROP)
    to = eye - P; cos = (to * n).sum(-1) / np.maximum(np.linalg.norm(to, axis=-1), 1e-9)
    m &= np.where(sid < 6, np.abs(cos), cos) >= HEAD            # a room face's normal is stored unsigned
    with shot(B, w, h, d["fov"]), np.errstate(all="ignore"): px, py, _, z = B.project(d["cam"], P)
    nx, ny = (px + 0.5) / w, (py + 0.5) / h
    m &= (z > 0.05) & (nx >= EDGE) & (nx <= 1 - EDGE) & (ny >= EDGE) & (ny <= 1 - EDGE)
    xi, yi = np.clip(np.nan_to_num(px).round().astype(int), 0, w - 1), np.clip(np.nan_to_num(py).round().astype(int), 0, h - 1)
    m &= ~d["lift"][yi, xi] & ~(d["halo"][yi, xi] & (sid < 6))
    j = np.nonzero(m)[0]
    if len(j): m[j] = B.visible(eye, P[j])
    return m

def walk_cams(C):
    """A walker the planners never saw: free standing spots on a 0.75 m grid (coarser in a large place: about 80 spots),
    to within 0.5 m of the walls, each looking six ways level and two ways 35 degrees down, turned from spot to spot."""
    B = C.B; d = max(0.75, float(np.sqrt(C.RW * C.RD / 80))); out = []
    xs, zs = [0.5 + (np.arange(int((L - 1.0) / d) + 1) + 0.37) * (L - 1.0) / (int((L - 1.0) / d) + 1) for L in (C.RW, C.RD)]
    for i, (x, z) in enumerate((x, z) for x in xs for z in zs):
        e = np.array([x, C.EYE, z])
        if any(B.inside(e, p, 0.3) for p in C.KEEP_OUT): continue
        for j in range(8):
            a = np.radians(137.5 * i + 60 * j) if j < 6 else np.radians(137.5 * i + 30 + 180 * j); p = 0.0 if j < 6 else np.radians(35)
            out.append((f"w{i}_{j}", "walk", e, e + [np.cos(a) * np.cos(p), -np.sin(p), np.sin(a) * np.cos(p)], 8, False))
    return out

def _cam(i):
    """One screen camera under every variant. Eval cameras give their pixels back; walk cameras a histogram."""
    C, src, rank, lifted, tcams = (G[k] for k in ("C", "src", "rank", "lifted", "tcams")); B = C.B; N = len(src)
    name, kind, eye, look, step, full = G["cams"][i]; cam = (tuple(eye), tuple(look))
    P, sid, n = seen(C, cam, lifted, SCREEN[0] // step, SCREEN[1] // step, SFOV); shape = sid.shape
    P, sid, n = P.reshape(-1, 3), sid.ravel(), n.reshape(-1, 3); ar = np.arange(len(sid))
    sky = (sid == 4) if C.OUT else np.zeros(sid.shape, bool); dead = shade(C, lifted, P, sid) | sky
    S = scores(C, src, rank, lifted, P, sid, n); T = tangents(n); Je = spread(B, cam, FE, P, T); fo = focal(BASE, B.FOV)
    R, Wo = np.zeros(S.shape, np.float32), np.zeros(S.shape, np.float32)
    for v in range(N):
        m = S[v] < 1e5
        if m.any(): R[v, m], Wo[v, m] = mix(spread(B, src[v][0], fo, P[m], T[m]), Je[m])
    close = np.linalg.norm(P - eye, axis=1) < NEARBY
    tr = train_ratio(B, P, n, sid, tcams, B.W / 2 / np.tan(np.radians(B.FOV) / 2), cam, FE) if full and tcams else None
    out = []
    for keep, size, details in G["variants"]:
        keep = np.asarray(keep); Sk = S[keep]; j = Sk.argmin(0); own = np.where(Sk[j, ar] < 1e5, keep[j], -1); own[dead] = -1
        k = size[0] / BASE[0]; r = np.where(own >= 0, R[own, ar] * k, 0.0); worst = np.where(own >= 0, Wo[own, ar] * k, 0.0)
        for di, d in enumerate(details):
            j = np.nonzero(owned(C, d, lifted, P, sid, n, dead))[0]
            if not len(j): continue
            a, b = mix(spread(B, d["cam"], focal(d["size"], d["fov"]), P[j], T[j]), Je[j]); up = a > r[j]; j = j[up]
            r[j], worst[j], own[j] = a[up], b[up], N + di
        gr, gs, go = r.reshape(shape), sid.reshape(shape), own.reshape(shape); steps = pairs = 0
        for a, b in ((np.s_[:, 1:], np.s_[:, :-1]), (np.s_[1:], np.s_[:-1])):
            same = (gs[a] == gs[b]) & (go[a] >= 0) & (go[b] >= 0); hi, lo = np.maximum(gr[a], gr[b]), np.minimum(gr[a], gr[b])
            pairs += int(same.sum()); steps += int((same & (hi > 1.5 * lo)).sum())
        if full:      # as density.py pools them: every other traced pixel, off the sky
            g = lambda a: a.reshape(shape + a.shape[1:])[::2, ::2][~sky.reshape(shape)[::2, ::2]]
            out.append({"r": g(r).astype(np.float32), "worst": g(worst).astype(np.float32), "none": g(own < 0), "det": g(own >= N), "tr": g(tr) if tr is not None else None,
                        "med": float(np.median(r[~sky])) if (~sky).any() else None, "steps": steps, "pairs": pairs, "sharp": float(np.minimum(r[~sky], 1).sum()), "live": int((~sky).sum())})
        else:
            row = {"steps": steps, "pairs": pairs}
            for key, m in (("all", ~sky), ("near", ~sky & close)):
                p = m & (own >= 0); x = r[p]
                row[key] = (np.histogram(np.log2(np.maximum(x, 1e-9)), BINS)[0], np.array([m.sum(), (m & (own < 0)).sum(), (x < 1).sum(), (x < 0.5).sum(), np.minimum(x, 1).sum(), (own[m] >= N).sum(), (x < 2.0 ** BINS[0]).sum()], float))
            out.append(row)
    return name, kind, out

def pool_map(fn, n):
    k = max(1, min(8, (os.cpu_count() or 4) - 2, n))
    if k == 1: return [fn(i) for i in range(n)]
    with mp.get_context("fork").Pool(k) as pool: return pool.map(fn, range(n), chunksize=2)

def check(C, d, src, rank, lifted, keep, ring):
    """Would this view be paintable and useful? Its camera's place, what fills its picture (as the painter sees it,
    moving things in place), the share of the picture it may own, and how many times finer than the ring it is there."""
    B = C.B; eye = np.array(d["eye"], float); h, w = d["sid"].shape; fwd = np.array(d["look"], float) - eye; fwd /= np.linalg.norm(fwd); flags = []
    if not (0.3 < eye[0] < C.RW - 0.3 and 0.3 < eye[2] < C.RD - 0.3 and eye[1] >= 0.3 and (C.OUT or eye[1] <= C.RH - 0.2)): flags.append("camera outside the room or against a wall, floor or ceiling")
    if any(B.inside(eye, p, 0.25) for p in C.KEEP_OUT): flags.append("camera inside or against an object")
    if C.NEW and C.crowded(eye): flags.append("camera by a post or in an open frame (chisel_ring's crowded())")
    P, sid, n = seen(C, d["cam"], [], w, h, d["fov"]); P, sid, n = P.reshape(-1, 3), sid.ravel(), n.reshape(-1, 3); t = np.linalg.norm(P - eye, axis=1)
    pitch = float(np.degrees(np.arcsin(fwd[1]))); share = np.bincount(sid, minlength=6 + len(C.objs)) / len(sid)
    inv = 1 / np.maximum(t, 0.3); lo, hi = np.percentile(inv, [1, 99]); depth_range = float(hi / max(lo, 1e-9))
    if t.min() < 0.5: flags.append(f"something {t.min():.2f} m from the lens")
    if pitch < -75: flags.append("looks straight down: not a photo anyone takes, and blockout.basis has no 'up' there")
    if eye[1] > C.EYE + 0.61: flags.append(f"camera {eye[1]:.1f} m up" + (": a drone picture" if eye[1] > 3 else ""))
    if share.max() > 0.9: flags.append(f"one surface fills {share.max():.0%} of the picture: nothing to recognise the room by")
    if depth_range < 1.3 and not C.OUT: flags.append(f"flat depth map (near to far only {depth_range:.2f} times): prep()'s stretch would turn it into a bowl")
    if d["fov"] != B.FOV: flags.append(f"angle {d['fov']:g}, not the ring's {B.FOV:g}")
    if d["size"][0] * 3 != d["size"][1] * 4: flags.append(f"picture {d['size'][0]}x{d['size'][1]} is not 4:3: consistent.py reads every picture as 1024x768")
    ok = owned(C, d, lifted, P, sid, n, shade(C, lifted, P, sid) | ((sid == 4) if C.OUT else False))
    S = scores(C, src, rank, lifted, P, sid, n)[np.asarray(keep)]; j = S.argmin(0); own = np.where(S[j, np.arange(len(sid))] < 1e5, np.asarray(keep)[j], -1)
    qd = per_metre(B, d["cam"], focal(d["size"], d["fov"]), P, n); qr = np.zeros(len(sid))
    for v in np.unique(own[own >= 0]): m = own == v; qr[m] = per_metre(B, src[v][0], focal(ring, B.FOV), P[m], n[m])
    finer = ok & (qd > qr); had = finer & (qr > 0); names = FACES + [o["name"] for o in C.objs]
    mine = {names[s]: round(float((finer & (sid == s)).mean()), 3) for s in np.unique(sid[finer]) if (finer & (sid == s)).mean() >= 0.01}
    claim = {str(o.get("surface", "")).replace("wall ", ""): o.get("picture_share", o.get("share")) for o in d.get("owns", [])}
    wrong = sorted(k for k, v in claim.items() if k not in mine and (v is None or v >= 0.03))
    if wrong: flags.append("claims to own " + ", ".join(wrong) + ", which this rule gives it under 1% of its picture of")
    return {"eye": [round(float(v), 2) for v in eye], "pitch": round(pitch, 1), "nearest_m": round(float(t.min()), 2), "depth_range": round(depth_range, 2),
            "fills": {names[s]: round(float(share[s]), 2) for s in np.argsort(-share)[:3] if share[s] >= 0.05}, "owns": mine, "own_share": round(float(finer.mean()), 3),
            "times_finer": round(float(np.median(qd[had] / qr[had])), 2) if had.any() else None, "new_paint": round(float((finer & (qr == 0)).mean()), 3),
            "px_per_m": int(np.median(qd[finer])) if finer.any() else 0, "seconds": round(seconds(d["size"]), 1), "flags": flags}

def norm(d, N, ring_n):
    """Any of the planners' JSON as {"views": [{"eye", "look", "fov", "size", "owns"}], "drop": [...], "ring": [w, h]}."""
    if "ring" in d and isinstance(d["ring"], dict):      # detail_a
        return {"views": [{"eye": v["cam"][0], "look": v["cam"][1], "fov": v["fov"], "size": v["size"], "owns": v["owns"]} for v in d["views"]],
                "drop": list(range(ring_n, N)) if d["ring"]["down_views_dropped"] else [], "ring": d["ring"]["paint"]}
    if "pay" in d and "ring" in d["pay"]:                # detail_c
        return {"views": [{k: v[k] for k in ("eye", "look", "fov", "size", "owns")} for v in d["views"]], "drop": d["pay"]["ring"]["views_left_out"], "ring": d["pay"]["ring"]["size"]}
    return {"views": [{"eye": v["eye"], "look": v["look"], "fov": v["fov"], "size": v.get("paint", v.get("size")), "owns": v.get("owns", [])} for v in d["views"]], "drop": d.get("drop", []), "ring": d["ring"]}   # detail_b, pick.py

def med(h, low):
    """Median of log-binned r: h the histogram, low the count below the first bin."""
    c = np.cumsum(h) + low; t = c[-1] / 2
    return round(float(2.0 ** BINS[min(int(np.searchsorted(c, t)) + 1, len(BINS) - 1)]), 3) if c[-1] else None

def judge(world_dir, plans):
    """plans: [(tag, plan as norm() gives it)]. -> the report."""
    t0 = time.time(); C, world = K.layout(world_dir); B = C.B; src, rank, lifted = sources(C, world); N = C.N
    try: tcams = [(tuple(e), tuple(l)) for e, l in json.load(open(f"{world}/gen_c/cams.json"))]
    except OSError: tcams = []
    variants = [(list(range(N)), BASE, [])]
    for tag, p in plans: variants.append(([v for v in range(N) if v not in p["drop"]], tuple(p["ring"]), [look_through(C, v, lifted) for v in p["views"]]))
    ev = [(nm, kind, e, l, 4, True) for nm, kind, e, l in K.layout_cams(world)]; wk = walk_cams(C)
    G.update(C=C, src=src, rank=rank, lifted=lifted, tcams=tcams, variants=variants, cams=ev + wk)
    res = pool_map(_cam, len(G["cams"])); out = {"world": os.path.basename(world), "room": [C.RW, C.RH, C.RD], "outdoor": bool(C.OUT), "ring_views": N, "walk_cameras": len(wk)}
    for vi, (keep, size, details) in enumerate(variants):
        E = [(nm, kind, o[vi]) for nm, kind, o in res[:len(ev)]]; Wk = [o[vi] for _, _, o in res[len(ev):]]
        def stats(kinds):
            rows = [x for _, kind, x in E if kind in kinds]; cat = lambda k: np.concatenate([x[k] for x in rows])
            r, w, none = cat("r"), cat("worst"), cat("none"); p = ~none; s = {"median_r": round(float(np.median(r[p])), 3), "r_lt_1": round(float((r[p] < 1).mean()), 4), "r_lt_0.5": round(float((r[p] < 0.5).mean()), 4),
                 "median_worst": round(float(np.median(w[p])), 3), "unpainted": round(float(none.mean()), 4), "sharp": round(float(np.minimum(r, 1).mean()), 4), "median_r0": round(float(np.median(r)), 3), "detail_owned": round(float(cat("det").mean()), 4)}
            if tcams: s["median_eff_r"] = round(float(np.median(np.minimum(r, cat("tr"))[p])), 3)
            return s
        close = {nm: round(x["med"], 3) for nm, kind, x in E if kind == "close"}; low = min(close, key=lambda k: (close[k], k))
        walk = {}
        for key in ("all", "near"):
            h = sum(x[key][0] for x in Wk); c = sum(x[key][1] for x in Wk); painted = c[0] - c[1]
            walk[key] = {"median_r": med(h, c[6]), "r_lt_1": round(c[2] / painted, 4), "r_lt_0.5": round(c[3] / painted, 4), "unpainted": round(c[1] / c[0], 4), "sharp": round(c[4] / c[0], 4), "detail_owned": round(c[5] / c[0], 4)}
        row = {"eval": {"near": stats(("close", "floor")), "close": stats(("close",)), "floor": stats(("floor",)), "centre": stats(("centre",)), "worst_close": [low, close[low]]}, "walk": walk,
               "steps": {"eval": round(sum(x["steps"] for _, _, x in E) / max(1, sum(x["pairs"] for _, _, x in E)), 4), "walk": round(sum(x["steps"] for x in Wk) / max(1, sum(x["pairs"] for x in Wk)), 4)},
               "ring": {"views": len(keep), "left_out": [v for v in range(N) if v not in keep], "size": list(size)},
               "paint_seconds": round(len(keep) * seconds(size) + sum(seconds(d["size"]) for d in details), 1)}
        if vi: row["views"] = [check(C, d, src, rank, lifted, keep, size) for d in details]; out.setdefault("plans", {})[plans[vi - 1][0]] = row
        else: out["baseline"] = row
    out["seconds"] = round(time.time() - t0, 1)
    return out

if __name__ == "__main__":
    C, world = K.layout(sys.argv[1])
    plans = [(a.split("=", 1)[0], norm(json.load(open(a.split("=", 1)[1])), C.N, C.RING)) for a in sys.argv[2:]]
    print(json.dumps(judge(sys.argv[1], plans)))
