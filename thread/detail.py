"""What to paint when there is time for only so many pictures, and how a detail picture is made to fit the room.

plan(): chisel_ring.py prep calls it with the number of ordinary pictures there is time for (pipeline.py works that out
from the time limit and what the earlier steps took). Painting is nearly all of a build, so the number of pictures is
the time budget. Two things are chosen, by one measure: how much of what a walker's screen could show the paint
delivers ("sharp": the mean over his screen pixels of COVER for having paint at all plus the rest for min(painted
pixels per screen pixel, 1); no paint counts 0).
  - Which of the usual views (the ring and the downward ones) are left out: the one whose loss costs the walker least,
    one at a time, counted again after each. The first few cost next to nothing: neighbours stand in for each other.
    View 0 always stays: it is painted first and sets the look.
  - Which detail views are added: ordinary photos from free standing spots, at eye height looking a little down (a wall
    with its floor) or well down (the floor ahead), and from 3 m up looking steeply down (the ground nearly square-on).
    One at a time, the one that adds most, given what is painted so far. One is passed over if something stands within
    0.6 m of its lens or one wall or thing fills it.
Every split of the time between usual and detail views is tried and the best taken. What a detail view may own is
owner.py's rule; the planner uses the same limits (head-on, rim, finer than the owner) on patches of surface.
Unless told to be strict, the plan leaves out only the usual views a walker does not miss (together under FREE of his
"sharp"): when the time is too short for the rest, they are painted all the same, no detail view is, and the build runs
over. Measured on the tower room, whose shapes and moving things leave time for 8.8 pictures: with its two unmissed
views left out the close cameras' pixels without paint stay 3.3%; with five left out they are 4.5%, seven 6.9%, nine 13.5%.

pictures(): the painted pictures as the paint is laid from them. A detail picture is settled into the room first: its
colours at scales above about 40 pixels are replaced by those of the paint it covers (the ring's, and earlier detail
pictures'), and towards its rim it fades into that paint. So a drift in tone or light cannot show as an edge on the
wall; only the fine detail is the detail picture's own.
"""
import os, numpy as np
from PIL import Image
from scipy import ndimage
import owner as O

SCREEN, VFOV = (1600, 900), 65.0               # the viewer's picture (viewer/main.js; as evals/density.py)
FE = SCREEN[1] / 2 / np.tan(np.radians(VFOV) / 2)
SFOV = 2 * np.degrees(np.arctan(np.tan(np.radians(VFOV) / 2) * SCREEN[0] / SCREEN[1]))      # blockout's angle runs side to side
CELL = 0.15                                    # metres: a walker's screen pixels are gathered into surface patches this size
BINS = 32 * 2 ** (np.arange(14) / 2)           # screen pixels per metre a patch is seen at, in half octaves (32 .. 2900)
SPOTS = 40                                     # about how many places the walker stands, and detail views are offered from
HIGH, STEEP = 3.0, -60.0                       # metres up and degrees down of the raised views (indoors no nearer the ceiling than 0.4 m)
PITCH = (-10.0, -40.0)                         # degrees down from eye height: a wall with its floor; the floor ahead
# What paint is worth beside fine paint. Tried at 0 on the tower room with time for 8.8 pictures: 6 usual views were kept
# and 6 detail views added, the close and floor cameras' pixels without paint went from 3% to 17%, and one wall seen
# from 0.8 m was a smear from end to end. Surface without paint is filled with a blur of its surroundings; coarse paint
# is still the room
COVER = 0.5
FREE = 0.001                                   # share of the walker's "sharp" that leaving usual views out may cost when not strict
RING_LEAST = 4                                 # usual views that are painted whatever the time
TONE = 20                                      # px (at the working size): the blur that splits a detail picture's tone from its detail

def aim(eye, yaw, pitch):
    a, p = np.radians(yaw), np.radians(pitch)
    return tuple(float(v) for v in np.array(eye) + [np.cos(a) * np.cos(p), np.sin(p), np.sin(a) * np.cos(p)])

def spots(C, most=SPOTS):
    """Free places to stand at eye height, on a grid fine enough to give about `most` (never finer than 0.5 m)."""
    clear = lambda e: 0.4 < e[0] < C.RW - 0.4 and 0.4 < e[2] < C.RD - 0.4 and not any(C.B.inside(e, p, 0.3) for p in C.KEEP_OUT)
    d = max(0.5, float(np.sqrt(C.RW * C.RD / most)))
    xs, zs = [(np.arange(n) + 0.5) * w / n for n, w in ((max(1, round(C.RW / d)), C.RW), (max(1, round(C.RD / d)), C.RD))]
    return [np.array([x, C.EYE, z]) for x in xs for z in zs if clear((x, C.EYE, z))]

def seen(C, cam, w, h, fov):
    """What a camera sees of the blockout, everything in place, at w x h and the angle fov: point, surface id (6 + object)
    and normal per pixel."""
    B = C.B; was = B.W, B.H, B.FOV; B.W, B.H, B.FOV = int(w), int(h), float(fov)
    try: t, P, sid, nrm = B.trace(cam, C.PRIM_OBJ, True)
    finally: B.W, B.H, B.FOV = was
    return P, sid, O.normals(C, P, sid, nrm)

def per_metre(B, cam, focal, P, n):
    """Pixels per metre of surface (the square root of pixels per m2) that a picture with that focal length lays at P."""
    return focal * np.sqrt(O.laid(B, cam, P, n))

def walk(C):
    """The walker's screens gathered into patches of surface: (point, normal, surface id, weight per BINS of the screen
    pixels per metre each is seen at). He stands on every spot and looks four ways level and two ways 30 degrees down."""
    B = C.B; keys, Ps, ns, es = [], [], [], []
    for si, e in enumerate(spots(C)):
        looks = [aim(e, 90 * i + 45 * (si % 2), 0) for i in range(4)] + [aim(e, 180 * i + 45 * (si % 4), -30) for i in range(2)]
        for look in looks:
            cam = (tuple(map(float, e)), look); P, sid, n = seen(C, cam, 64, 36, SFOV)
            live = (sid != 4) if C.OUT else np.ones(sid.shape, bool)             # the sky outdoors is not painted
            P, sid, n = P[live], sid[live], n[live]
            cell = np.floor(P / CELL).astype(np.int64); face = sid < 6
            cell[face, sid[face] % 3] = 0                                        # a point on a room face lies a hair either side of it
            side = np.where(face[:, None], 0, np.round(n).astype(np.int64) @ [[1], [3], [9]])
            keys.append(np.concatenate([sid[:, None], cell, side], 1)); Ps.append(P); ns.append(n); es.append(per_metre(B, cam, FE, P, n))
    key, P, n, e = (np.concatenate(a) for a in (keys, Ps, ns, es))
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
    b = np.clip(np.round(2 * np.log2(np.maximum(e, 1) / 32)), 0, len(BINS) - 1).astype(int)
    Wt = np.zeros((len(first), len(BINS))); np.add.at(Wt, (inv, b), 1)
    return P[first], n[first], key[first, 0].astype(int), Wt

def worth(Wt, q):
    """Per patch: what the walker's screen pixels of it are worth with paint at q pixels a metre (0: none). A pixel with
    paint counts COVER, and the rest by min(painted pixels per screen pixel, 1)."""
    return (Wt * np.where(q[:, None] > 0, COVER + (1 - COVER) * np.minimum(q[:, None] / BINS, 1), 0)).sum(1)

def leave_out(S, Q, Wt):
    """The order in which the usual views are missed least: [(view, what the walker loses with it)], view 0 never.
    S: view x patch scores (owner.owners' `every`: the lowest owns; NEVER where the view does not show the patch).
    Q: view x patch painted pixels per metre."""
    S = S.copy(); N, J = S.shape; ar = np.arange(J); out = []
    for _ in range(N - 1):
        own = S.argmin(0); have = S[own, ar] < O.NONE; now = np.where(have, worth(Wt, np.where(have, Q[own, ar], 0)), 0); best = None
        for v in range(1, N):
            cols = np.nonzero(have & (own == v))[0]
            if (S[v] >= O.NEVER).all(): continue                         # already left out
            sub = S[:, cols].copy(); sub[v] = O.NEVER; nxt = sub.argmin(0); ok = sub[nxt, np.arange(len(cols))] < O.NONE
            loss = float((now[cols] - np.where(ok, worth(Wt[cols], np.where(ok, Q[nxt, cols], 0)), 0)).sum())
            if best is None or loss < best[1]: best = (v, loss)
        S[best[0]] = O.NEVER; out.append((best[0], best[1] / max(Wt.sum(), 1)))
    return out

def candidates(C, P, n, sid, focal):
    """The detail views on offer: (cameras, and as three flat arrays over all of them: which camera, which patch it
    would show well enough to own, its painted pixels per metre there)."""
    B = C.B; stand = spots(C); up = HIGH if C.OUT else min(HIGH, C.RH - 0.4)
    raised = [e + [0, up - C.EYE, 0] for e in stand if up >= C.EYE + 0.5 and not any(B.inside(e + [0, up - C.EYE, 0], p, 0.3) for p in C.KEEP_OUT)]
    free = (sid != 4) & ~np.isin(sid, [6 + oi for oi in C.LIFT] + (list(C.BACKDROP) if C.OUT else []))
    shape = np.isin(sid, [6 + oi for oi in C.SHAPES]); cams, cj, cq, cc = [], [], [], []; was = B.W, B.H
    for si, eye in enumerate(stand + raised):
        to = eye - P; dist = np.linalg.norm(to, axis=1); cos = (to * n).sum(1) / np.maximum(dist, 1e-9)
        near = np.nonzero(free & (np.where(shape, cos, np.abs(cos)) >= O.DETAIL_HEAD) & (dist > 0.3))[0]
        near = near[B.visible(eye, P[near])]                             # what is hidden depends on the spot alone
        if not len(near): continue
        for yaw in np.arange(12) * 30.0 + 15 * (si % 2):
            for pitch in (PITCH if si < len(stand) else (STEEP,)):
                cam = (tuple(map(float, eye)), aim(eye, yaw, pitch))
                B.W, B.H = 16, 12                                        # a thumbnail: is it a photo anyone would take?
                try: t, _, s = B.trace(cam, C.PRIM_OBJ)
                finally: B.W, B.H = was
                share = np.bincount(s.ravel(), minlength=6) / s.size; share[1] = 0
                if t.min() < 0.6 or t[5:7, 7:9].mean() < 1.2 or share.max() > 0.85: continue
                with np.errstate(all="ignore"): px, py, _, z = B.project(cam, P[near])
                e0, e1 = O.DETAIL_EDGE * B.W, O.DETAIL_EDGE * B.H
                ok = (z > 0.05) & (px >= e0) & (px <= B.W - 1 - e0) & (py >= e1) & (py <= B.H - 1 - e1)
                if ok.sum() < 20: continue
                j = near[ok]; cc.append(np.full(len(j), len(cams))); cj.append(j); cq.append(per_metre(B, cam, focal, P[j], n[j])); cams.append(cam)
    if not cams: return [], np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return cams, np.concatenate(cc), np.concatenate(cj), np.concatenate(cq)

def add(q, Wt, cands, most):
    """Up to `most` detail views, each the one that adds most given the paint so far (q, pixels per metre by patch;
    changed in place). -> [camera index]"""
    cams, cc, cj, cq = cands; picked = []
    if not len(cams) or most <= 0: return picked
    mine = worth(Wt[cj], cq); live = np.ones(len(cams), bool)
    for _ in range(most):
        take = (cq > O.DETAIL_GAIN * q[cj]) & (q[cj] > 0)               # only what is painted already: a detail picture is that paint, finer
        gain = np.bincount(cc, weights=np.where(take, mine - worth(Wt, q)[cj], 0), minlength=len(cams)); gain[~live] = -1
        c = int(gain.argmax())                                           # ties go to the lowest index: the same plan every run
        if gain[c] <= 1e-4 * Wt.sum(): break
        m = (cc == c) & take; q[cj[m]] = cq[m]; picked.append(c); live[c] = False
    return picked

def plan(C, traced, budget, most=6, cost=1.0, strict=False):
    """{"drop": the usual views to leave out, "views": [[eye, look], ...] the detail views to paint, in order,
    "sharp": what the walker gets, as painted in full / as planned, "over": how many ordinary pictures' time the plan
    takes beyond the budget (0 when it fits)}.
    traced: view -> B.trace(cam, C.PRIM_OBJ, True) for the usual views. budget: the time there is, in ordinary
    pictures. cost: a detail picture's time as a share of an ordinary one's. strict: hold to the budget whatever it
    costs (down to RING_LEAST usual views); otherwise only views the walker does not miss are left out."""
    B = C.B; N = C.N; lifted = list(C.LIFT)
    shaped = [6 + oi for oi in {*C.SHAPES, *(C.SLIM if C.NEW else ())}]; views = O.Views()
    for i in range(N):
        t, _, sid, _ = traced[i]; on = np.isin(sid, shaped); views.append(O.view(B.CAMS[str(i)], t, sid, O.big(on, C.HALO) & ~on, lifted))
    rank = O.rank(np.stack([np.bincount(v[2].ravel(), minlength=6 + len(C.objs)) for v in views], 1).astype(float))
    try: pw = int(os.environ.get("WORLD_PAINT", "").split("x")[0])
    except ValueError: pw = B.W
    focal = pw / 2 / np.tan(np.radians(B.FOV) / 2)
    P, n, sid, Wt = walk(C); J = len(P); none = lambda Q: np.zeros(Q.shape[:-1], bool)      # before any paint no shadow is known
    every = {}; O.owners(C, views, rank, P, sid, n, lifted, none, every=every)
    mover = np.isin(sid, [6 + oi for oi in lifted])
    if mover.any():      # the moving things themselves: painted in place, by the rule for a room they are in
        ev = {}; O.owners(C, views, rank, P[mover], sid[mover], n[mover], (), none, every=ev)
        for v in ev: every[v][mover] = ev[v]
    S = np.stack([every.get(v, np.full(J, O.NEVER)) for v in range(N)]); Q = np.stack([per_metre(B, views[v][0], focal, P, n) for v in range(N)])
    gone = leave_out(S, Q, Wt); cands = candidates(C, P, n, sid, focal); ar = np.arange(J); total = max(Wt.sum(), 1)
    def ring(k):      # the paint with the k least-missed views left out
        Sk = S.copy(); Sk[[v for v, _ in gone[:k]]] = O.NEVER; own = Sk.argmin(0)
        return np.where(Sk[own, ar] < O.NONE, Q[own, ar], 0.0)
    full = float(worth(Wt, ring(0)).sum() / total); best = None
    free, lost = 0, 0.0                                                  # the views that can go without the walker missing them
    for _, l in gone:
        if lost + l > FREE: break
        lost += l; free += 1
    least = min(N, RING_LEAST) if strict else N - free
    for keep in range(max(least, min(N, int(budget))), least - 1, -1):   # too little time for `least`: they are painted all the same
        q = ring(N - keep); d = max(0, min(most, int((budget - keep) / cost + 1e-9))); picked = add(q, Wt, cands, d)
        got = float(worth(Wt, q).sum() / total)
        if best is None or got > best[0] + 1e-4: best = (got, keep, picked)
    got, keep, picked = best
    return {"drop": sorted(int(v) for v, _ in gone[:N - keep]), "views": [[list(cands[0][c][0]), list(cands[0][c][1])] for c in picked],
            "sharp": [round(full, 4), round(got, 4)], "over": round(max(0.0, keep + len(picked) * cost - budget), 2),
            "missed": [[int(v), round(float(l), 5)] for v, l in gone]}

def blank(C, folder, i):
    """Where detail picture i (folder/NN.png) is not to be trusted: what it was not begun from paint (the painter left
    its flat filling there; chisel_ring saves the mask as begun/NN.png beside the pictures' folder), with 4 px around.
    All of it is trusted when there is no mask: it was painted afresh."""
    f = f"{os.path.dirname(os.path.normpath(folder))}/begun/{i:02d}.png"
    if not os.path.exists(f): return np.zeros((C.B.H, C.B.W), bool)
    return O.big(np.asarray(Image.open(f).convert("L").resize((C.B.W, C.B.H), Image.NEAREST)) < 128, 9)

def pictures(C, views, rank, folder):
    """Every painted picture at the working size, as the paint is laid from it: a list by view number, None for a view
    that was left out. views, rank: as owner.owners() takes them. folder: where NN.png are."""
    B = C.B; W, H = B.W, B.H; paint = []
    for i in range(len(views)):
        if views[i] is None: paint.append(None); continue
        im = Image.open(f"{folder}/{i:02d}.png").convert("RGB")
        if im.size != (W, H): im = im.resize((W, H), Image.LANCZOS)      # pictures may be painted smaller than the working size
        paint.append(np.asarray(im, np.float32))
    nd = getattr(views, "detail", 0)
    for v in range(len(views) - nd, len(views)):      # each settled into the paint it covers: the ring's and earlier detail pictures'
        t, P, sid, nrm = B.trace(views[v][0], C.PRIM_OBJ, True)
        under = O.Views(views[:v]); under.detail = v - (len(views) - nd)
        own, _, _, at, _ = O.owners(C, under, rank, P, sid, nrm, spot=True)
        paint[v] = settle(paint[v], *lay(paint, own, at))
    return paint

def lay(paint, own, at):
    """The colour each point takes from its owner, read between that picture's pixels: (picture, where there is any)."""
    col = np.zeros(own.shape + (3,), np.float32)
    for u in np.unique(own[own >= 0]):
        m = own == u
        for ch in range(3): col[m, ch] = ndimage.map_coordinates(paint[u][..., ch], [at[m][:, 1], at[m][:, 0]], order=1, mode="nearest")
    return col, own >= 0

def settle(raw, under, have):
    """A detail picture made to sit in the paint it covers (under, where `have`): its tone (what a TONE px blur keeps)
    becomes that paint's, and from 2 x DETAIL_EDGE in from its border out to DETAIL_EDGE it fades into that paint."""
    H, W = have.shape; k = TONE * W / 1024; soft = lambda a: ndimage.gaussian_filter(a, k, mode="nearest")
    w = soft(have.astype(np.float32)); diff = (under - raw) * have[..., None]
    out = raw + np.stack([soft(diff[..., ch]) for ch in range(3)], -1) / np.maximum(w, 1e-3)[..., None] * np.minimum(1, 4 * w)[..., None]
    yy, xx = np.mgrid[:H, :W]; e = O.DETAIL_EDGE
    a = np.clip(np.minimum(np.minimum(xx, W - 1 - xx) / W, np.minimum(yy, H - 1 - yy) / H) / e - 1, 0, 1); a = (a * a * (3 - 2 * a))[..., None]
    return np.clip(np.where(have[..., None], under * (1 - a) + out * a, out), 0, 255).astype(np.float32)
