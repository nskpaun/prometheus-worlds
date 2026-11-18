"""How blurry will this world look, worked out in seconds from its geometry alone: no pictures are read, no GPU.
  evals/density.py WORLD [--paint WxH]      (WORLD: a folder name under worlds/ or a path; needs boxes.json, depth/, gen/)
For each camera of evals/cams.py, at the viewer's size (1600x900, 65 degrees top to bottom as viewer/main.js; every 4th
pixel is traced), find the blockout surface point each pixel sees and the painted view that point takes its paint from,
by the rule consistent.py lays the paint by (owner.py, the one copy of it). The rules: per surface the views are ranked
by how much of it they show; shapes are not taken from side-on or from behind; the band round a shape's outline and
round a thing that moves away is avoided; the floor in such a thing's painted shadow is left out (for that the painted
pictures are read; nothing else reads them); the point must really be seen. Then
  r = painted pixels per screen pixel = sqrt(painted pixels per m2 of the surface there / screen pixels per m2)
which is the task's sqrt((fo^2 cos_o / do^2) / (fe^2 cos_e / de^2)) with the stretch towards a picture's edges counted
on both sides. Below 1 the paint is coarser than the screen and looks soft. "worst" is the same along the direction the
paint is thinnest: a floor painted from far and low has plenty of pixels across and few along.
A point no painted view shows (filled with a blurred or flat colour) has no r: its share is "unpainted", and the medians
and shares are over the painted pixels only. The sky outdoors is left out.
The line gives, per kind of camera and for close and floor together ("near"): the median r, the share of pixels with
r < 1 and r < 0.5, the same for worst, and the share with no paint; "blurry_share" (r < 1, near); each camera's median r
(there, no paint counts as 0, as on its map).
The paint is not the only limit. "median_train_r" is the same ratio for the 1024x768 training views (the finest one
that faces the point; WORLD/gen_c/cams.json): the splat never sees the paint finer than that. "median_eff_r" takes the
lesser of the two per pixel. A paint size that lifts median_r far above median_eff_r buys nothing by itself. A third
limit is not worked out here: the splats' own size (median_axis_cm in splat_scores.py).
--paint WxH asks "what if the views had been painted this size"; otherwise the size of gen/images/00.png is used.
Prints one JSON line; writes a map per camera to evals/work/WORLD/density/NAME.png: dark red r <= 0.25, orange 0.5,
yellow 1, green 2, blue 4 and up, magenta no paint, grey sky.
Not modelled: consistent.py enlarges the pictures to 1024x768 and its training views sample them once more; whether an
indoor ceiling ends up one flat colour; what the trainer then makes of it.
"""
import sys, os, json, time, numpy as np
from PIL import Image
from scipy import ndimage
EV = os.path.dirname(os.path.abspath(__file__)); TH = os.path.dirname(EV)
sys.path[:0] = [EV, TH]
import cams as K, owner as O, detail as DT
SCREEN, VFOV, STEP = (1600, 900), 65.0, 4      # the viewer's picture; one ray per STEP x STEP screen pixels
STOPS = np.array([[-2, 150, 0, 0], [-1, 255, 90, 0], [0, 255, 230, 0], [1, 60, 200, 60], [2, 40, 90, 220]], float)   # log2 r -> colour

Views = O.Views      # the painted views as owner.py wants them, with .shadow (owner.shade's answer) and .paint (the pictures, once read)

def sources(C, world):
    """What consistent.py knows of the painted views (owner.view per view), the rank of each view for each surface, and
    the lifted objects. The floor in a lifted thing's shadow is found from the paint, so then the pictures are read."""
    B = C.B; NS = 6 + len(C.objs)
    try: lifted = [i for i in json.load(open(f"{world}/gen_c/lifted.json")) if type(i) is int]     # what the splat was trained without
    except (OSError, ValueError): lifted = [] if os.path.isdir(f"{world}/gen_c") else list(C.LIFT)
    # the band as C.halo makes it, with a filter 100 times quicker than PIL's (checked equal on these worlds)
    shaped = [6 + oi for oi in {*C.SHAPES, *(C.SLIM if C.NEW else ())}]
    src = Views(); src.detail = C.ND; src.paint = None; counts = []
    for i in range(C.N + C.ND):      # a world painted to a plan: usual views left out (None), detail views after them
        if i in C.SKIP: src.append(None); counts.append(np.zeros(NS, int)); continue
        t, sid, _ = C.surfaces(i); on = np.isin(sid, shaped)
        src.append(O.view(B.CAMS[str(i)], t, sid, (O.big(on, C.HALO) & ~on) | (DT.blank(C, f"{world}/gen/images", i) if i >= C.N else False), lifted))
        counts.append(np.bincount(sid.ravel(), minlength=NS) if i < C.N else np.zeros(NS, int))
    rank = O.rank(np.stack(counts, 1).astype(float))
    if lifted and O.shade_bands(C, lifted):
        got = {}
        def paint(v):      # as consistent.py reads a picture: enlarged to its working size
            if C.ND and src.paint is None: src.paint = DT.pictures(C, src, rank, f"{world}/gen/images")      # detail pictures are settled first
            if src.paint is not None: return src.paint[v]
            if v not in got:
                im = Image.open(f"{world}/gen/images/{v:02d}.png").convert("RGB")
                got[v] = np.asarray(im.resize((B.W, B.H), Image.LANCZOS) if im.size != (B.W, B.H) else im, np.float32)
            return got[v]
        try: src.shadow = O.shade(C, src, rank, lifted, paint)
        except OSError: pass                                         # a world not yet painted: the whole band counts as shadow
    return src, rank, lifted

normal = O.normal      # unit normals of a box or cylinder primitive at points on its surface

def seen(C, cam, lifted, size=None):
    """What a screen camera sees of the blockout the splat was trained on: per traced pixel the point, the surface id
    (as consistent.py's: 0..5 room faces, 6 + object) and the surface normal. size (w, h): a picture of the trainer's
    own angle at that size instead (evals/reference.py)."""
    B = C.B; keep = [j for j, oi in enumerate(C.PRIM_OBJ) if oi not in lifted]
    full, was = B.PRIMS, (B.W, B.H, B.FOV)
    B.PRIMS = [full[j] for j in keep]; B.W, B.H = size or (SCREEN[0] // STEP, SCREEN[1] // STEP)
    if not size: B.FOV = 2 * np.degrees(np.arctan(np.tan(np.radians(VFOV) / 2) * SCREEN[0] / SCREEN[1]))      # blockout's angle runs side to side
    try: t, P, pid, nrm = B.trace(cam, keep, True)
    finally: B.PRIMS = full; B.W, B.H, B.FOV = was
    n = nrm.astype(float); sid = pid.copy()
    for f in range(6): n[pid == f, f % 3] = 1
    for j in keep:
        m = pid == 6 + j; sid[m] = 6 + C.PRIM_OBJ[j]
        if m.any() and full[j]["shape"] != "mesh": n[m] = normal(full[j], P[m])
    return P, sid, n

def owner(C, src, rank, lifted, P, sid, n, where=False):
    """The painted view each point takes its paint from, -1 where none: owner.owners(), the rule consistent.py lays the
    paint by. where: also the place in that view (x, y in its pixels, between pixels unless borrowed from a neighbour)."""
    own, _, _, at, _ = O.owners(C, src, rank, P, sid, n, lifted, getattr(src, "shadow", None), spot=where)
    if C.OUT: own[sid == 4] = -1
    return (own, at) if where else own

def spread(B, cam, focal, P, T):
    """How a picture's pixels lie on the surface at points P: the 2x2 matrix of pixels per metre along the two surface
    directions T. cam: (eye, look); focal in pixels."""
    eye, r, u, f = B.basis(cam); v = P - eye; z = (v @ f)[..., None, None]
    a = np.stack([v @ r, v @ u], -1)[..., :, None]                       # (.., 2, 1)
    along = lambda d: np.einsum("...ik,i->...k", T, d)
    Tc = np.stack([along(r), along(u)], -2)                              # (.., 2 picture axes, 2 surface directions)
    return focal * (Tc * z - a * along(f)[..., None, :]) / z ** 2

def ratio(B, P, n, src_cam, fo, cam, fe):
    """(r, worst) at points P with normals n: painted pixels per screen pixel, by area and along the thinnest direction."""
    ax = np.eye(3)[np.argmin(np.abs(n), -1)]
    t1 = np.cross(n, ax); t1 /= np.linalg.norm(t1, axis=-1, keepdims=True); T = np.stack([t1, np.cross(n, t1)], -1)   # (.., 3, 2)
    Jo, Je = spread(B, src_cam, fo, P, T), spread(B, cam, fe, P, T)
    de = Je[..., 0, 0] * Je[..., 1, 1] - Je[..., 0, 1] * Je[..., 1, 0]
    de = np.where(np.abs(de) < 1e-12, 1e-12, de)[..., None, None]
    M = Jo @ (np.stack([np.stack([Je[..., 1, 1], -Je[..., 0, 1]], -1), np.stack([-Je[..., 1, 0], Je[..., 0, 0]], -1)], -2) / de)
    det = np.abs(M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]); q = (M ** 2).sum((-1, -2))
    return np.sqrt(det), np.sqrt(np.maximum(q - np.sqrt(np.maximum(q * q - 4 * det * det, 0)), 0) / 2)

def train_ratio(B, P, n, sid, tcams, ft, cam, fe):
    """Training-view pixels per screen pixel at points P: the finest of the training cameras that face the point and
    hold it in frame. A picture lays f^2 |n.v| / z^3 pixels on a square metre (v from the eye to the point, z its depth).
    Nothing checks that the point is not hidden from the training camera, so this is the most it can be."""
    P = P.astype(np.float32); n = n.astype(np.float32); best = np.zeros(len(P), np.float32)
    lay = lambda c, f: (lambda eye, r, u, fw: (P - np.float32(eye), np.float32(r), np.float32(u), np.float32(fw)))(*B.basis(c))
    for c in tcams:
        v, r, u, fw = lay(c, ft); z = v @ fw; nv = -(v * n).sum(1)
        with np.errstate(all="ignore"):
            ok = (z > 0.05) & (np.abs(v @ r) * ft < z * B.W / 2) & (np.abs(v @ u) * ft < z * B.H / 2) & ((nv > 0) | (sid < 6))
            best = np.maximum(best, np.where(ok, ft * ft * np.abs(nv) / z ** 3, 0))
    v, r, u, fw = lay(cam, fe)
    return np.sqrt(best / np.maximum(fe * fe * np.abs((v * n).sum(1)) / (v @ fw) ** 3, 1e-9))

def colours(r, own, sky):
    x = np.log2(np.maximum(r, 1e-6))
    img = np.stack([np.interp(x, STOPS[:, 0], STOPS[:, k]) for k in (1, 2, 3)], -1)
    img[own < 0] = (200, 0, 200); img[sky] = (150, 150, 150)
    return Image.fromarray(img.astype(np.uint8)).resize((SCREEN[0] // 2, SCREEN[1] // 2), Image.NEAREST)

def density(world_dir, paint=None, maps=True, train_cams=None):
    t0 = time.time(); C, world = K.layout(world_dir); B = C.B
    src, rank, lifted = sources(C, world)
    if paint is None:
        try: paint = list(Image.open(f"{world}/gen/images/00.png").size)
        except OSError: paint = json.load(open(f"{world}/gen/views.json"))["paint"]
    fo = paint[0] / 2 / np.tan(np.radians(B.FOV) / 2); fe = SCREEN[1] / 2 / np.tan(np.radians(VFOV) / 2)
    out = os.path.join(EV, "work", os.path.basename(world), "density")
    if maps: os.makedirs(out, exist_ok=True)
    try: tcams = [(tuple(e), tuple(l)) for e, l in json.load(open(train_cams or f"{world}/gen_c/cams.json"))]
    except OSError: tcams = []                                           # a world not yet through the agree step
    ft = B.W / 2 / np.tan(np.radians(B.FOV) / 2)                         # the training views' focal length
    per, pools = {}, {}
    for name, kind, eye, look in K.layout_cams(world):
        cam = (tuple(eye), tuple(look)); P, sid, n = seen(C, cam, lifted); own = owner(C, src, rank, lifted, P, sid, n)
        r, worst = np.zeros(sid.shape), np.zeros(sid.shape)
        for v in np.unique(own[own >= 0]):
            m = own == v; r[m], worst[m] = ratio(B, P[m], n[m], src[v][0], fo, cam, fe)
        sky = (sid == 4) if C.OUT else np.zeros(sid.shape, bool)
        if maps: colours(r, own, sky).save(f"{out}/{name}.png")
        per[name] = round(float(np.median(r[~sky])), 3) if (~sky).any() else None      # as the map shows it: no paint is 0
        k = ~sky[::2, ::2]; h = lambda a: a[::2, ::2][k]                # every other traced pixel: plenty for medians
        tr = train_ratio(B, h(P), h(n), h(sid), tcams, ft, cam, fe) if tcams else np.full(int(k.sum()), np.inf)
        pools.setdefault(kind, []).append((h(r), h(worst), h(own) < 0, tr))
    cat = lambda kinds, j: np.concatenate([p[j] for k in kinds for p in pools.get(k, [])])
    med = lambda a: round(float(np.median(a)), 3) if len(a) else None
    def stats(kinds):
        """Over the pixels that have paint; the share that has none is given beside them, not mixed in as r = 0."""
        r, w, none, tr = (cat(kinds, j) for j in range(4)); r, w, t = r[~none], w[~none], tr[~none]
        share = lambda a, lim: round(float((a < lim).mean()), 4) if len(a) else None
        return {"median_r": med(r), "r_lt_1": share(r, 1), "r_lt_0.5": share(r, 0.5),
                "median_worst": med(w), "worst_lt_1": share(w, 1), "worst_lt_0.5": share(w, 0.5), "unpainted": round(float(none.mean()), 4),
                # what else caps detail: the training views' own pixels on the surface, and the lesser of the two per pixel
                "median_train_r": med(tr) if tcams else None, "median_eff_r": med(np.minimum(r, t)) if tcams else None}
    near = stats(("close", "floor"))
    return {"world": os.path.basename(world), "paint": list(paint), "screen": list(SCREEN), "kinds": {k: stats((k,)) for k in pools},
            "near": near, "blurry_share": near["r_lt_1"], "cams": per, "maps": out if maps else None, "seconds": round(time.time() - t0, 2)}

if __name__ == "__main__":
    a = sys.argv[2:]
    print(json.dumps(density(sys.argv[1], [int(v) for v in a[a.index("--paint") + 1].split("x")] if "--paint" in a else None)))
