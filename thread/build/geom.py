"""One sculpted kind's mesh for the paint: its geometry, straight-on views of it, and pictures projected onto it.
Paint is stored on the kind's mesh as load() returns it (centred, upright, front on +z), per surface sample: a face index
and barycentric coordinates. shapes.place() keeps vertex and face order, so the same samples land on every placed copy.
thread/.venv, CPU."""
import math
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree as _cKD
class cKDTree(_cKD):
    """scipy's tree, asked with every core"""
    def query(self, x, k=1, **kw):
        kw.setdefault("workers", -1); return super().query(x, k=k, **kw)

MAX_FACES = 60000
FILL = 0.58          # half-width of a view over the mesh's bounding diagonal

# ---------------------------------------------------------------- mesh (as shapes.load reads it, with the folder as a parameter: the two must agree, or paint lands on the wrong faces)
def load(shapes_dir, kind):
    import trimesh, open3d as o3d
    m = trimesh.load(f"{shapes_dir}/{kind}.glb", force="mesh")
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
    if len(F) > MAX_FACES:
        o = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)).simplify_quadric_decimation(MAX_FACES)
        V, F = np.asarray(o.vertices), np.asarray(o.triangles)
    parts = trimesh.graph.connected_components(trimesh.Trimesh(V, F, process=False).face_adjacency, nodes=np.arange(len(F)))
    if len(parts) > 1:
        big = max(len(c) for c in parts); low = V[:, 1].min() + 0.05 * (V[:, 1].max() - V[:, 1].min())
        F = F[np.sort(np.concatenate([c for c in parts if len(c) >= 0.05 * big or V[F[c]][..., 1].max() > low]))]
    used = np.unique(F); V = V[used]; F = np.searchsorted(used, F)
    return V - (V.min(0) + V.max(0)) / 2, F

def scene_of(V, F):
    import open3d as o3d
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.core.Tensor(np.ascontiguousarray(V, np.float32)), o3d.core.Tensor(np.ascontiguousarray(F, np.uint32)))
    return s

def cast(scene, o, d):
    import open3d as o3d
    rays = np.concatenate([o, d], -1).astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))

# ---------------------------------------------------------------- cameras
def axes(az, el):
    """Right, up, forward of a camera az degrees round the object from its front (+z), el degrees above."""
    a, e = math.radians(az), math.radians(el)
    f = -np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
    r = np.cross(f, [0.0, 1.0, 0.0]); r /= np.linalg.norm(r)
    return r, np.cross(r, f), f

class View:
    """A view of a mesh centred on the origin. k = camera distance over half (None: orthographic). The picture plane through
    the origin is 2*half wide. flip mirrors left-right."""
    def __init__(s, half, az, el, px, k=None, flip=False):
        s.half, s.az, s.el, s.px, s.k, s.flip = half, az, el, px, k, flip
        s.r, s.u, s.f = axes(az, el); s.dist = (k or 4.0) * half; s.eye = -s.f * s.dist
    def rays(s):
        g = (np.arange(s.px) + 0.5) / s.px * 2 - 1
        X, Y = np.meshgrid(g * s.half * (-1 if s.flip else 1), -g * s.half)
        p = X[..., None] * s.r + Y[..., None] * s.u
        if s.k is None: return p - s.f * s.dist, np.broadcast_to(s.f, p.shape)
        d = p - s.eye; return np.broadcast_to(s.eye, p.shape), d / np.linalg.norm(d, axis=-1, keepdims=True)
    def render(s, scene):
        """-> dict t (inf where empty), fid, uv, n, and Q the hit points"""
        o, d = s.rays(); a = cast(scene, o, d)
        t = a["t_hit"].numpy(); fid = a["primitive_ids"].numpy().astype(np.int64); fid[~np.isfinite(t)] = -1
        Q = o + d * np.where(np.isfinite(t), t, 0)[..., None]
        return {"t": t, "fid": fid, "uv": a["primitive_uvs"].numpy(), "n": a["primitive_normals"].numpy(), "Q": Q, "hit": fid >= 0}
    def project(s, P):
        """points -> column, row (float px)"""
        z = (P - s.eye) @ s.f
        sc = 1.0 if s.k is None else s.dist / np.maximum(z, 1e-6)
        x = (P @ s.r) * sc * (-1 if s.flip else 1); y = (P @ s.u) * sc
        return (x / s.half + 1) / 2 * s.px - 0.5, (-y / s.half + 1) / 2 * s.px - 0.5
    def visible(s, scene, P, N):
        """is each surface point seen by this view (not hidden by the mesh)?  also the cosine to the view"""
        if s.k is None: d = np.broadcast_to(-s.f, P.shape).copy()
        else: d = s.eye - P; d /= np.linalg.norm(d, axis=-1, keepdims=True)
        eps = 2e-3 * s.half
        a = cast(scene, P + d * eps + N * np.sign((N * d).sum(-1, keepdims=True)) * eps, d)
        return ~np.isfinite(a["t_hit"].numpy()), np.abs((N * d).sum(-1))
    def spec(s): return {"half": s.half, "az": s.az, "el": s.el, "px": s.px, "k": s.k, "flip": s.flip}

# ---------------------------------------------------------------- pictures
def backdrop(im):
    h, w = im.shape[:2]; m = max(4, h // 25)
    b = np.median(np.concatenate([im[:m].reshape(-1, 3), im[-m:].reshape(-1, 3), im[:, :m].reshape(-1, 3), im[:, -m:].reshape(-1, 3)]), 0)
    k = np.clip(im @ b / max(float(b @ b), 1e-6), 0.45, 1.3)
    return np.linalg.norm(im - k[..., None] * b, axis=-1), k, b

def shade(R, view, mode, back, albedo=None):
    """A render on a plain backdrop. mode: clay (grey, lit from the camera's upper left), depth (near light, far dark),
    claydepth (both), paint (albedo with a soft shade), flat (albedo as it is)."""
    r, u, f = view.r, view.u, view.f; hit = R["hit"]; n = R["n"]
    l = -0.6 * f + 0.6 * u - 0.4 * r; l /= np.linalg.norm(l)
    n = np.where((n @ f)[..., None] > 0, -n, n)
    lam = np.clip(n @ l, 0, 1); head = np.clip(-(n @ f), 0, 1)
    z = (R["Q"] - view.eye) @ f; zh = z[hit]
    dn = np.zeros_like(z)
    if hit.any(): dn = np.clip((z - zh.min()) / max(zh.max() - zh.min(), 1e-6), 0, 1)
    img = np.empty(hit.shape + (3,), np.float32); img[:] = np.asarray(back, np.float32)
    if mode == "clay": g = 0.28 + 0.62 * lam + 0.10 * head; img[hit] = (0.82 * g[hit])[:, None]
    elif mode == "depth": g = 0.95 - 0.75 * dn; img[hit] = g[hit][:, None]
    elif mode == "claydepth": g = (0.35 + 0.55 * lam + 0.10 * head) * (1.0 - 0.45 * dn); img[hit] = (0.9 * g[hit])[:, None]
    elif mode == "paint": g = 0.62 + 0.30 * lam + 0.08 * head; img[hit] = albedo[hit] * g[hit][:, None]
    elif mode == "flat": img[hit] = albedo[hit]
    return np.clip(img, 0, 1)

def to_img(a): return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
def from_img(p, size=None):
    im = Image.open(p).convert("RGB")
    if size and im.size != (size, size): im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255

def outline(mask): return mask ^ ndimage.binary_erosion(mask)
def iou(a, b): return float((a & b).sum() / max((a | b).sum(), 1))
def edge_gap(a, b):
    ea, eb = outline(a), outline(b)
    if not ea.any() or not eb.any(): return float("inf")
    return float(ndimage.distance_transform_edt(~eb)[ea].mean())
def stray(paint, mesh):
    """How far the painted outline strays from the mesh outline (both bool, same size)."""
    px = mesh.shape[0]
    return {"iou": round(iou(paint, mesh), 3), "edge_px": round((edge_gap(paint, mesh) + edge_gap(mesh, paint)) / 2 * 512 / px, 2),
            "spill": round(float((paint & ~mesh).sum() / max(mesh.sum(), 1)), 3), "bare": round(float((mesh & ~paint).sum() / max(mesh.sum(), 1)), 3)}

def box_of(m):
    ys, xs = np.nonzero(m); return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
def box_mask(m, px=96):
    if not m.any(): return np.zeros((px, px), bool)
    x0, y0, x1, y1 = box_of(m); c = m[y0:y1, x0:x1]; s = max(c.shape); pad = np.zeros((s, s), bool)
    oy, ox = (s - c.shape[0]) // 2, (s - c.shape[1]) // 2; pad[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return np.asarray(Image.fromarray(pad).resize((px, px), Image.NEAREST))

def box_affine(src_mask, dst_mask):
    """Scale (x, y) and shift taking src mask's bounding box onto dst's: (sx, sy, tx, ty), dst = s * src + t."""
    a, b = box_of(src_mask), box_of(dst_mask)
    sx, sy = (b[2] - b[0]) / max(a[2] - a[0], 1), (b[3] - b[1]) / max(a[3] - a[1], 1)
    return sx, sy, b[0] - sx * a[0], b[1] - sy * a[1]

def warp_to(img, aff, shape, order=1):
    """Resample img (src) into dst pixels under dst = s * src + t."""
    sx, sy, tx, ty = aff; h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (xx - tx) / sx, (yy - ty) / sy
    if img.ndim == 2: return ndimage.map_coordinates(img.astype(np.float32), [cy, cx], order=order, mode="nearest")
    return np.stack([ndimage.map_coordinates(img[..., c], [cy, cx], order=order, mode="nearest") for c in range(img.shape[2])], -1)

# ---------------------------------------------------------------- the hero picture
def fit_picture(scene, half, alpha):
    """Where the object's own picture was taken from: the view (orthographic or with a lens, and mirrored or not) whose
    silhouette matches the cut-out best. -> View spec at the picture's size class, IoU after box alignment."""
    want = box_mask(alpha); best = (-1.0,)
    for flip in (False, True):
        for k in (None, 5.0, 3.0):
            for az in range(-80, 81, 5):
                for el in range(0, 46, 5):
                    R = View(half, az, el, 96, k, flip).render(scene)
                    s = iou(want, box_mask(R["hit"]))
                    if s > best[0]: best = (s, az, el, k, flip)
    _, az, el, k, flip = best
    # refine: finer angles and lenses, at twice the size
    want = box_mask(alpha, 160); best = (iou(want, box_mask(View(half, az, el, 160, k, flip).render(scene)["hit"], 160)), az, el, k, flip)
    for k2 in (None, 8.0, 5.0, 4.0, 3.0, 2.5, 2.0):
        for a2 in range(az - 6, az + 7, 2):
            for e2 in range(max(el - 6, 0), el + 7, 2):
                s = iou(want, box_mask(View(half, a2, e2, 160, k2, flip).render(scene)["hit"], 160))
                if s > best[0]: best = (s, a2, e2, k2, flip)
    return {"az": float(best[1]), "el": float(best[2]), "k": best[3], "flip": bool(best[4]), "box_iou": round(best[0], 3)}

# ---------------------------------------------------------------- samples and projection
def sample(V, F, n):
    import trimesh
    m = trimesh.Trimesh(V, F, process=False)
    P, fi = trimesh.sample.sample_surface(m, n, seed=0)
    bary = trimesh.triangles.points_to_barycentric(m.triangles[fi], P)
    return np.asarray(P), np.asarray(fi), np.asarray(bary), np.asarray(m.face_normals[fi])

def bilin(img, x, y):
    h, w = img.shape[:2]; c = [np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
    if img.ndim == 2: return ndimage.map_coordinates(img.astype(np.float32), c, order=1)
    return np.stack([ndimage.map_coordinates(img[..., ch], c, order=1) for ch in range(img.shape[2])], 1)

def project_paint(view, scene, P, N, rgb, ok, power=2):
    """One painted picture (float RGB at the view's px, already aligned to the view) onto the samples.
    ok: bool mask of pixels whose paint may be used. -> colours (S, 3), weights (S,)"""
    x, y = view.project(P); vis, cosv = view.visible(scene, P, N)
    inb = (x >= 0) & (x <= view.px - 1) & (y >= 0) & (y <= view.px - 1)
    w = np.where(vis & inb, cosv ** power, 0.0) * (bilin(ok.astype(np.float32), x, y) > 0.99)
    return bilin(rgb, x, y), w

def align(fg, mesh):
    """A painted outline that is the mesh's outline a little shifted or scaled is put back on it. -> (affine or None, mask aligned)"""
    if not fg.any(): return None, fg
    a = box_affine(fg, mesh)
    if not (0.4 < a[0] < 2.5 and 0.4 < a[1] < 2.5) or abs(a[0] / a[1] - 1) > 0.15: return None, fg
    fa = warp_to(fg.astype(np.float32), a, fg.shape) > 0.5
    return (a, fa) if iou(fa, mesh) > iou(fg, mesh) + 0.005 else (None, fg)

def despill(img, back):
    """Take the backdrop's colour cast out of a picture (glass and chrome show the backdrop): the part of each pixel's
    colour that points the backdrop's way from grey is removed."""
    b = np.asarray(back, np.float32); d = b - b.mean(); d /= max(float(np.linalg.norm(d)), 1e-6)
    c = img - img.mean(-1, keepdims=True); p = np.clip(c @ d, 0, None)
    return np.clip(img - p[..., None] * d, 0, 1)

def framed_half(V, az, el, fill=0.80):
    """Half-width of an orthographic view in which the object fills `fill` of the frame (Klein re-frames a small object)."""
    r, u, f = axes(az, el); return float(max(np.abs(V @ r).max(), np.abs(V @ u).max()) / fill)

def depth_edges(R, view, grow):
    """Pixels near a jump in depth inside the object (one part in front of another): paint a few pixels off lands on the wrong part there."""
    z = np.where(R["hit"], (R["Q"] - view.eye) @ view.f, 0.0); j = 0.03 * view.half
    e = np.zeros(z.shape, bool)
    e[:, 1:] |= np.abs(np.diff(z, axis=1)) > j; e[1:, :] |= np.abs(np.diff(z, axis=0)) > j
    e &= ndimage.binary_erosion(R["hit"], iterations=2)
    return ndimage.binary_dilation(e, iterations=grow)
