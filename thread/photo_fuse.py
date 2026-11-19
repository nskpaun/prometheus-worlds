"""A world from photos without training: every photo's SHARP piece (photo_pieces.py) placed and merged into one splat.
  photo_fuse.py WORLD            many photos: needs WORLD/txt (solved cameras), WORLD/dense/images, WORLD/pieces
  photo_fuse.py WORLD --one      one photo: WORLD/pieces holds a single piece
Writes WORLD/scene.ply, WORLD/scene.json (poses, up, start view, focal, width, metre 1) and WORLD/surface.ply, the fused
surface with the photos' colours, which mesh_photos.py --surface turns into the mesh and collider. thread/.venv, CPU, about 10 GB.

The frame is metres, as SHARP guesses them: the solved cameras' frame scaled and centred on the cameras; for one photo
the camera sits at the origin looking down -z with y up.

Many photos, in order:
 1 place   each piece goes where its photo was taken, its depth fitted to the solved points seen in that photo; splats
           strung between a thing and what is behind it (depth edges) are dropped
 2 agree   all pieces' depth is fused into one surface, and each piece is bent smoothly onto it (twice)
 3 colour  brightness and colour per photo from the points photos share, then a smooth correction per photo toward the
           colour the photos agree on for each small cell of space
 4 doubt   a photo's content in a cell is dropped when most other photos that see a surface there show a clearly different
           colour (a foot that stood there for one photo, a misplaced photo), or when two or more look straight through it
 5 own     each cell goes to one photo: the nearest, most face-on one that covers it, smoothed so owners form patches
 6 thin    flat, evenly coloured blocks of a photo's grid become single discs lying on the surface. With --flat every other
           splat is squashed along the surface's normal (SHARP's splats are long along their photo's line of sight); off by
           default, because on the two rooms tried it changed little and left dark specks
 7 patch   where the surface is left with no splat (behind silhouettes, steep sides), discs from the surface in its own colour
OPTIONS  --cell M (0.04)  --pool T (0.02, 0 = off)  --ghost T (0.35)  --big F (4)  --flat F (most a splat may reach along the normal, in grid cells; 0 = off, the default)  --no-patch"""
import sys, os, json, time, numpy as np
N = 768                                           # SHARP's grid
CV = np.diag([1.0, -1.0, -1.0])                    # SHARP's camera axes (y down, z forward) -> ours (y up, looking down -z)
SH0 = 0.28209479177387814
T0 = time.time()
def say(*x): print("[%4.0fs]" % (time.time() - T0), *x, flush=True)

# ---------------------------------------------------------------- small pieces, tested in tests/test_photo_fuse.py
def pack(k):
    """Three small signed integers per row -> one sortable int64 key."""
    k = np.asarray(k, np.int64); return (k[:, 0] + 2 ** 20) << 42 | (k[:, 1] + 2 ** 20) << 21 | (k[:, 2] + 2 ** 20)
def unpack(key):
    return np.stack([(key >> 42) - 2 ** 20, ((key >> 21) & (2 ** 21 - 1)) - 2 ** 20, (key & (2 ** 21 - 1)) - 2 ** 20], 1)

def fit_inverse_depth(zs, zt, rounds=4):
    """Robust a, b with 1/zt = a/zs + b: a piece's own depth zs against the true depth zt at the solved points. Returns (a, b, median error)."""
    A = np.stack([1 / zs, np.ones(len(zs))], 1); keep = np.ones(len(zs), bool)
    for _ in range(rounds):
        co = np.linalg.lstsq(A[keep], 1 / zt[keep], rcond=None)[0]; res = np.abs(A @ co - 1 / zt) * zt; keep = res < max(0.05, 2.5 * np.median(res[keep]))
    return float(co[0]), float(co[1]), float(np.median(res))

def photo_gains(rows, n_photos, pull=3.0, limit=0.5):
    """rows: (point id, photo index, log r, log g, log b), one per sighting of a solved point. Returns a per-photo colour
    multiplier (n_photos, 3) that makes photos agree: log colour = photo + point, with a weak pull of every photo to no change."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    pid = np.unique(rows[:, 0], return_inverse=True)[1]; ii = rows[:, 1].astype(int); m = len(rows); npid = pid.max() + 1
    M = coo_matrix((np.r_[np.ones(2 * m), np.full(n_photos, pull)], (np.r_[np.arange(m), np.arange(m), m + np.arange(n_photos)], np.r_[ii, n_photos + pid, np.arange(n_photos)])), shape=(m + n_photos, n_photos + npid)).tocsr()
    G = np.stack([lsqr(M, np.r_[rows[:, 2 + c], np.zeros(n_photos)])[0][:n_photos] for c in range(3)], 1); G -= np.median(G, 0)
    return np.exp(-np.clip(G, -limit, limit)).astype(np.float32)

def pick_owner(ci, photo, cost, n_cells):
    """The photo with the lowest cost in each cell (-1 where a cell has no candidate). ci: cell index per candidate row."""
    order = np.lexsort((cost, ci)); first = np.r_[True, ci[order][1:] != ci[order][:-1]]; own = np.full(n_cells, -1); own[ci[order][first]] = photo[order][first]; return own

def settle_owners(cells, ci, photo, cost, smooth=0.5, rounds=4):
    """Owners per cell, where an owner the neighbouring cells do not share costs extra, so owners form patches."""
    own = pick_owner(ci, photo, cost, len(cells)); ck = unpack(cells)
    offs = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)])
    for _ in range(rounds if smooth > 0 else 0):
        same = np.zeros(len(ci)); tot = np.zeros(len(ci))
        for o in offs:
            nk = pack(ck[ci] + o); j = np.searchsorted(cells, nk); j[j >= len(cells)] = 0; has = cells[j] == nk
            same += has & (own[j] == photo); tot += has
        own = pick_owner(ci, photo, cost * (1 + smooth * (1 - same / np.maximum(tot, 1))), len(cells))
    return own

def blocks(x, b):
    """A per-splat array in grid order (2 layers, N rows, N columns) -> (N/b, N/b, 2*b*b, ...): every b x b block with both layers."""
    m = N // b; return x.reshape((2, m, b, m, b) + x.shape[1:]).transpose(1, 3, 0, 2, 4, *range(5, 4 + x.ndim)).reshape((m, m, 2 * b * b) + x.shape[1:])
def unblocks(flat, b):
    """A per-block flag (N/b, N/b) -> per-splat flag in grid order."""
    m = N // b; return np.repeat(flat[:, :, None], 2 * b * b, 2).reshape(m, m, 2, b, b).transpose(2, 0, 3, 1, 4).reshape(-1)

def discs_from_normals(n):
    """Quaternions (w x y z) turning +z onto each unit normal."""
    from scipy.spatial.transform import Rotation
    ax = np.cross([0, 0, 1.0], n); s = np.linalg.norm(ax, axis=1); ang = np.arctan2(s, n[:, 2])
    ax = np.where(s[:, None] > 1e-9, ax / np.maximum(s, 1e-9)[:, None], [1.0, 0, 0])        # a normal along -z turns about x
    return Rotation.from_rotvec(ax * ang[:, None]).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)

def flatten(quat, scale, normal, thick):
    """Splats squashed along a direction: quat (w x y z), scale and unit normal per splat; `thick` is the most a splat may reach
    along its normal. SHARP's splats are long along their own photo's line of sight, which shows as streaks from any other angle;
    squashed onto the surface they lie on, they look the same from the photo and right from elsewhere. Returns (quat, scale)."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix().astype(np.float64); n = normal.astype(np.float64)
    reach = np.sqrt(np.einsum("ni,nij,nj->n", n, np.einsum("nij,nj,nkj->nik", R, scale.astype(np.float64) ** 2, R), n))        # the splat's size along the normal
    k = np.minimum(1.0, thick / np.maximum(reach, 1e-12)); do = k < 0.999
    if not do.any(): return quat, scale
    A = np.eye(3)[None] - (1 - k[do])[:, None, None] * n[do][:, :, None] * n[do][:, None, :]
    M = A @ R[do] * scale[do][:, None, :].astype(np.float64); w, V = np.linalg.eigh(M @ M.transpose(0, 2, 1)); V[:, :, 0] *= np.sign(np.linalg.det(V))[:, None]
    quat, scale = quat.copy(), scale.copy(); quat[do] = Rotation.from_matrix(V).as_quat()[:, [3, 0, 1, 2]]; scale[do] = np.sqrt(np.maximum(w, 1e-14)); return quat, scale

def backdrop(rgb):
    """What the viewer shows behind the splats, where no photo looked: the place's middle colour, darkened."""
    return [round(float(x) * 0.55, 4) for x in np.median(rgb[:: max(1, len(rgb) // 200000)], 0)]

def write_ply(path, P, rgb, opacity, scale, quat):
    """A plain 3D Gaussian splat file: positions, colour (as the constant harmonic), opacity and scale as logits and logs, rotation w x y z."""
    n = len(P); o = np.clip(opacity, 1e-4, 1 - 1e-4)
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    E = np.concatenate([P, (rgb - 0.5) / SH0, np.log(o / (1 - o))[:, None], np.log(scale), quat], 1).astype("<f4")
    with open(path, "wb") as fh:
        fh.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n + "".join("property float %s\n" % k for k in names) + "end_header\n").encode()); fh.write(E.tobytes())
    return n

def lines(path):
    with open(path) as fh: return [l for l in fh if l[0] != "#"]

def read_colmap(txt):
    """Cameras and photos from COLMAP text files: ({id: (w, h, f)}, [photo dicts sorted by name])."""
    from scipy.spatial.transform import Rotation
    cams = {l.split()[0]: (int(l.split()[2]), int(l.split()[3]), float(l.split()[4])) for l in lines(f"{txt}/cameras.txt")}
    P3 = {int(l.split()[0]): np.array(l.split()[1:4], float) for l in lines(f"{txt}/points3D.txt")}
    L = lines(f"{txt}/images.txt"); ims = []
    for h, b in zip(L[::2], L[1::2]):
        t = h.split(); w, x, y, z = map(float, t[1:5]); R = Rotation.from_quat([x, y, z, w]).as_matrix(); tv = np.array(t[5:8], float)
        p = b.split(); ids = np.array(p[2::3], int); xy = np.array(list(zip(p[0::3], p[1::3])), float).reshape(-1, 2); k = ids >= 0
        ims.append(dict(name=t[9], cam=cams[t[8]], R=R, C=-R.T @ tv, X=np.array([P3[i] for i in ids[k]]).reshape(-1, 3), xy=xy[k], ids=ids[k]))
    return sorted(ims, key=lambda i: i["name"])

GY, GX = (np.tile(g.ravel(), 2) for g in np.mgrid[0:N, 0:N])          # grid row and column of every splat of a piece

def skirts(z, ok, jump=0.04, band=0.012):
    """Splats hanging between a thing and what is behind it. At a depth edge SHARP strings splats from the near surface back to
    the far one; from the photo's own view they hide behind the thing, from the side they are fins. A splat is one when the
    depth around its grid cell jumps by more than `jump` and it lies clear of both the near and the far side."""
    from scipy import ndimage as ndi
    near = depth_image(z, ok); far = np.full((N, N), -np.inf, np.float32); np.maximum.at(far, (GY[ok], GX[ok]), z[ok])
    near = ndi.minimum_filter(near, 3, mode="nearest"); far = ndi.maximum_filter(far, 3, mode="nearest"); a, b = near[GY, GX], far[GY, GX]
    return np.isfinite(a) & np.isfinite(b) & (b > a * (1 + jump)) & (z > a * (1 + band)) & (z < b * (1 - band))

def depth_image(z, ok):
    """Nearest depth per grid cell (inf where none) from a piece's splat depths."""
    dm = np.full((N, N), np.inf, np.float32); np.minimum.at(dm, (GY[ok], GX[ok]), z[ok]); return dm

# ---------------------------------------------------------------- one photo
def one(d):
    from scipy.spatial.transform import Rotation
    import open3d as o3d
    f = sorted(os.listdir(f"{d}/pieces"))[0]; g = np.load(f"{d}/pieces/{f}"); W, H, fpx = g["size"]
    P = g["xyz"] @ CV; q = (Rotation.from_matrix(CV) * Rotation.from_quat(g["quat"][:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    n = write_ply(f"{d}/scene.ply", P, g["rgb"], g["opacity"], g["scale"], q)
    surface_from([dict(Rc=CV, C=np.zeros(3), z=g["xyz"][:, 2], ok=g["opacity"] > 0.5, rgb=g["rgb"], W=W, H=H, f=fpx)], f"{d}/surface.ply")
    scene = {"up": [0, 1, 0], "center": [0, 0, 0], "poses": [np.eye(4).tolist()], "eye": [0, 0, 0], "forward": [0, 0, -1], "splats": n, "focal": float(fpx), "width": int(W), "metre": 1.0, "backdrop": backdrop(g["rgb"])}
    with open(f"{d}/scene.json", "w") as fh: json.dump(scene, fh)
    say("one photo:", n, "splats")

def surface_from(pcs, path=None, res=512):
    """All pieces' depth fused into one surface (TSDF). Each piece: Rc (camera axes -> world), C, z, ok, W, H, f, and rgb for colours."""
    import open3d as o3d
    samp = np.concatenate([(np.stack([(GX[::400] + 0.5) / N * p["W"] - p["W"] / 2, (GY[::400] + 0.5) / N * p["H"] - p["H"] / 2, np.full(len(GX[::400]), p["f"])], 1) / p["f"] * p["z"][::400, None])[p["ok"][::400]] @ p["Rc"].T + p["C"] for p in pcs])
    lo, hi = np.percentile(samp, [1, 99], 0); ctr = (lo + hi) / 2; ext = float((hi - lo).max()) * 1.3; vox = ext / res
    vol = o3d.pipelines.integration.UniformTSDFVolume(length=ext, resolution=res, sdf_trunc=4 * vox, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8, origin=ctr - ext / 2)
    for p in pcs:
        dm = depth_image(p["z"], p["ok"]); dm[~np.isfinite(dm)] = 0; col = np.full((N, N, 3), 128, np.uint8)
        if p.get("rgb") is not None: col[GY[p["ok"]], GX[p["ok"]]] = (np.clip(p["rgb"][p["ok"]], 0, 1) * 255).astype(np.uint8)
        E = np.eye(4); E[:3, :3] = p["Rc"].T; E[:3, 3] = -p["Rc"].T @ p["C"]
        K = o3d.camera.PinholeCameraIntrinsic(N, N, p["f"] * N / p["W"], p["f"] * N / p["H"], N / 2, N / 2)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.ascontiguousarray(col)), o3d.geometry.Image(dm), depth_scale=1.0, depth_trunc=float(ext), convert_rgb_to_intensity=False)
        vol.integrate(rgbd, K, E)
    m = vol.extract_triangle_mesh()
    if path: o3d.io.write_triangle_mesh(path, m)
    return m

# ---------------------------------------------------------------- many photos
def many(d, CELL=0.04, POOL=0.02, GHOST=0.35, BIG=4.0, SMOOTH=0.5, ROUNDS=2, PATCH=True, FLAT=0.0):
    import open3d as o3d
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from scipy import ndimage as ndi
    ims = read_colmap(f"{d}/txt"); ims = [im for im in ims if os.path.exists(f"{d}/pieces/{os.path.splitext(im['name'])[0]}.npz")]; n_im = len(ims)
    load = lambda im: np.load(f"{d}/pieces/{os.path.splitext(im['name'])[0]}.npz")

    # ---- 1 place: fit each piece's depth to its solved points; the median scale makes the frame metres
    fits = []
    for im in ims:
        g = load(im); W, H, f = g["size"]; z = g["xyz"][:, 2].astype(np.float64); dm = depth_image(z, g["opacity"] > 0.5)
        zt = (im["X"] - im["C"]) @ im["R"].T[:, 2] if len(im["X"]) else np.zeros(0)
        zs = dm[np.clip((im["xy"][:, 1] / H * N).astype(int), 0, N - 1), np.clip((im["xy"][:, 0] / W * N).astype(int), 0, N - 1)] if len(zt) else np.zeros(0); ok = (zt > 0) & np.isfinite(zs)
        fits.append((fit_inverse_depth(zs[ok], zt[ok]) if ok.sum() >= 30 else None, int(ok.sum()), float(np.median(zt[ok] / zs[ok])) if ok.any() else np.nan))
    k = 1.0 / np.nanmedian([f[2] for f in fits]); c0 = np.mean([im["C"] for im in ims], 0)        # solved units -> metres; the cameras' middle is the origin
    pcs = []; n_skirt = 0
    for im, (fit, npts, _) in zip(ims, fits):
        if fit is None: say(im["name"], "left out: only", npts, "solved points"); continue
        g = load(im); W, H, f = g["size"]; z = g["xyz"][:, 2].astype(np.float64); a, b, err = fit
        op = np.where(skirts(g["xyz"][:, 2], g["opacity"] > 0.5), 0, g["opacity"]).astype(np.float32); n_skirt += int((op == 0).sum())
        ratio = (1 / np.maximum(a / z + b, 1e-6)) * k / z; Rc = im["R"].T
        pcs.append(dict(name=im["name"], im=im, Rc=Rc, C=(im["C"] - c0) * k, ray=(g["xyz"] @ Rc.T).astype(np.float32), ratio=ratio.astype(np.float32), z=z.astype(np.float32), op=op, W=W, H=H, f=f, npts=npts, err=err))
    say("placed", len(pcs), "of", n_im, "pieces; median fit error %.1f%%;" % (100 * np.median([p["err"] for p in pcs])), "%.1f%% of splats hang at depth edges and are dropped" % (100 * n_skirt / (len(pcs) * 2 * N * N)))
    world = lambda p: p["C"].astype(np.float32) + p["ray"] * p["ratio"][:, None]
    for p in pcs: p["foot"] = p["W"] / N / p["f"]              # a grid cell's width at depth 1

    # ---- 3a brightness and colour per photo
    rows = []
    for i, p in enumerate(pcs):
        im = p["im"]; W, H = int(p["W"]), int(p["H"]); px = np.asarray(Image.open(f"{d}/dense/images/{im['name']}").convert("RGB").resize((W // 4, H // 4), Image.BOX), float)
        c = px[np.clip((im["xy"][:, 1] / 4).astype(int), 0, H // 4 - 1), np.clip((im["xy"][:, 0] / 4).astype(int), 0, W // 4 - 1)]; ok = (c.min(1) > 20) & (c.max(1) < 235)
        rows.append(np.c_[im["ids"][ok], np.full(ok.sum(), i), np.log(c[ok])])
    gain = photo_gains(np.concatenate(rows), len(pcs)); say("brightness differs by %.0f%% between photos" % (100 * np.log(gain).mean(1).std()))

    # ---- 2 agree: one surface, each piece bent smoothly onto it
    def surface(colour=False, path=None):
        return surface_from([dict(Rc=p["Rc"], C=p["C"], z=p["z"] * p["ratio"], ok=p["op"] > 0.5, W=p["W"], H=p["H"], f=p["f"], rgb=np.clip(load(p["im"])["rgb"] * gain[i], 0, 1) if colour else None) for i, p in enumerate(pcs)], path)
    def cast(scene, p):
        ln = np.linalg.norm(p["ray"], axis=1, keepdims=True); dirs = p["ray"] / ln
        hit = scene.cast_rays(o3d.core.Tensor(np.c_[np.tile(p["C"].astype(np.float32), (len(dirs), 1)), dirs])); nrm = hit["primitive_normals"].numpy(); cs = (nrm * dirs).sum(1)
        return hit["t_hit"].numpy() / ln[:, 0], np.abs(cs), nrm * -np.sign(cs)[:, None]        # the ratio that puts each splat on the surface, how face-on, the normal toward the camera
    def scene_of(mesh):
        s = o3d.t.geometry.RaycastingScene(); s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh)); return s
    B = 24; nb = N // B; blk = (GY // B) * nb + GX // B
    for r in range(ROUNDS):
        sc = scene_of(surface()); moved = []
        for p in pcs:
            want, _, _ = cast(sc, p); rel = want / p["ratio"]; ok = np.isfinite(rel) & (rel > 0.85) & (rel < 1.18) & (p["op"] > 0.5)
            order = np.lexsort((rel[ok], blk[ok])); kk = blk[ok][order]; st = np.flatnonzero(np.r_[True, kk[1:] != kk[:-1]]); en = np.r_[st[1:], len(kk)]
            fld = np.full(nb * nb, np.nan); big = (en - st) >= 40; fld[kk[st][big]] = rel[ok][order][((st + en) // 2)[big]]; fld = fld.reshape(nb, nb)
            if np.isnan(fld).all(): continue
            fld = ndi.gaussian_filter(fld[tuple(ndi.distance_transform_edt(np.isnan(fld), return_distances=False, return_indices=True))], 1.0)
            up = ndi.zoom(fld, B, order=1)[GY, GX].astype(np.float32); moved.append(np.median(np.abs(up - 1))); p["ratio"] = p["ratio"] * up
        say("agree, round %d: pieces moved by a median of %.1f%%" % (r + 1, 100 * np.median(moved)))
    mesh = surface(colour=True, path=f"{d}/surface.ply"); sc = scene_of(mesh); say("surface:", len(mesh.triangles), "triangles")

    # ---- what each piece has in each cell
    rK, rP, rS, rA, rC, rX = [], [], [], [], [], []
    for i, p in enumerate(pcs):
        g = load(p["im"]); Pw = world(p); zz = p["z"] * p["ratio"]
        # how far each splat is from the agreed surface: from the nearest point of it, not the first one along the photo's ray, or a stray
        # bump in front (a foot one photo saw, a fin behind a tap) would cast a hole onto the true surface behind it
        near = sc.compute_closest_points(o3d.core.Tensor(Pw)); nrm = near["primitive_normals"].numpy(); off = np.linalg.norm(near["points"].numpy() - Pw, axis=1) / np.maximum(zz, 1e-6)
        dirs = p["ray"] / np.linalg.norm(p["ray"], axis=1, keepdims=True); cs = (nrm * dirs).sum(1); cosn = np.abs(cs); nrm = nrm * -np.sign(cs)[:, None]
        edge = np.maximum(np.abs(GX / (N - 1) * 2 - 1), np.abs(GY / (N - 1) * 2 - 1)); big = g["scale"].max(1) * p["ratio"] / (zz * p["foot"])
        good = (p["op"] > 0.1) & np.isfinite(off) & (off < 0.025) & (big < BIG)            # on the agreed surface and not stretched: the rest is floaters and depth edges
        p.update(good=good, key=pack(np.floor(Pw / CELL)), Pw=Pw, zz=zz, nrm=nrm.astype(np.float16), dm=depth_image(zz, good))
        lc = np.log(np.clip(g["rgb"] * gain[i], 0.02, 1)); M = N // 8; acc = np.zeros((M * M, 4)); np.add.at(acc, ((GY // 8) * M + GX // 8)[good], np.c_[lc[good], np.ones(int(good.sum()))])
        p["look"] = np.where(acc[:, 3:] > 8, acc[:, :3] / np.maximum(acc[:, 3:], 1), np.nan).reshape(M, M, 3).astype(np.float32)       # what the photo shows, coarsely (log colour, 8 x 8 grid cells)
        score = zz / np.clip(cosn, 0.25, 1) * (1 + 1.5 * edge ** 4)
        kk = p["key"][good]; order = np.argsort(kk); ks = kk[order]; st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]) if len(ks) else np.zeros(0, int); cnt = np.diff(np.r_[st, len(ks)])
        if not len(st): rK.append(ks[:0]); rP.append(np.zeros(0, int)); rS.append(np.zeros(0)); rA.append(np.zeros(0)); rC.append(np.zeros((0, 3))); rX.append(np.zeros((0, 3))); p["rows"] = (good.copy(), order, st, cnt); continue
        rK.append(ks[st]); rP.append(np.full(len(st), i)); rS.append(np.add.reduceat(score[good][order], st) / cnt); rA.append(np.add.reduceat((zz[good][order] * p["foot"]) ** 2, st))
        rC.append(np.add.reduceat(np.log(np.clip(g["rgb"] * gain[i], 0.02, 1))[good][order], st, axis=0) / cnt[:, None]); rX.append(np.add.reduceat(Pw[good][order].astype(np.float64), st, axis=0) / cnt[:, None]); p["rows"] = (good.copy(), order, st, cnt)
    K_, P_, S_, A_, C_, X_ = (np.concatenate(x) for x in (rK, rP, rS, rA, rC, rX)); cells, ci = np.unique(K_, return_inverse=True); start = np.cumsum([0] + [len(x) for x in rK])
    ncand = np.bincount(ci, minlength=len(cells)); st_ = np.flatnonzero(np.r_[True, np.sort(ci)[1:] != np.sort(ci)[:-1]]); en_ = np.r_[st_[1:], len(ci)]; cons = np.zeros((len(cells), 3))
    for c in range(3):
        oc = np.lexsort((C_[:, c], ci)); cons[ci[oc][st_], c] = C_[oc, c][(st_ + en_) // 2]            # the agreed colour of a cell: the median over the photos that see it

    # ---- 4 doubt
    def seen_through(X, photo=None, colour=None):
        """For each point: how many photos look straight through it to something clearly behind, and how many see a surface there.
        With `colour` (log colour per point) also how many of the photos that see a surface about there show a clearly different colour, and how many the same."""
        viol = np.zeros(len(X), np.int16); agree = np.zeros(len(X), np.int16); other = np.zeros(len(X), np.int16); same = np.zeros(len(X), np.int16)
        for j, pj in enumerate(pcs):
            x = (X - pj["C"]) @ pj["Rc"]; z = x[:, 2]; zc = np.maximum(z, 1e-6); gu = np.floor((x[:, 0] / zc * pj["f"] + pj["W"] / 2) / pj["W"] * N); gv = np.floor((x[:, 1] / zc * pj["f"] + pj["H"] / 2) / pj["H"] * N)
            ins = (z > 0.05) & (gu >= 2) & (gu < N - 2) & (gv >= 2) & (gv < N - 2); ref = np.full(len(z), np.inf)
            if photo is not None: ins &= photo != j
            ref[ins] = pj["dm"][gv[ins].astype(int), gu[ins].astype(int)]
            mar = np.maximum(0.03, 0.025 * z); fin = np.isfinite(ref); viol += ins & fin & (z < ref - mar); agree += ins & fin & (np.abs(z - ref) <= mar)
            if colour is not None:
                there = np.flatnonzero(ins & fin & (np.abs(z - ref) <= 2 * mar)); diff = np.abs(pj["look"][gv[there].astype(int) // 8, gu[there].astype(int) // 8] - colour[there]).mean(1)
                other[there[diff > GHOST]] += 1; same[there[diff <= GHOST]] += 1
        return (viol, agree, other, same) if colour is not None else (viol, agree)
    viol, agree, other, same = seen_through(X_, P_, C_)
    ghost = (other >= 2) & (other > same); carved = (viol >= 2) & (viol > agree)
    say("doubt: %.1f%% of cells by colour, %.1f%% seen through" % (100 * ghost.mean(), 100 * carved.mean())); ghost |= carved
    for i, p in enumerate(pcs):
        sl = slice(start[i], start[i + 1]); gk = np.sort(K_[sl][ghost[sl]]); p["doubt"] = float(ghost[sl].mean()) if start[i + 1] > start[i] else 0.0
        if len(gk): j = np.searchsorted(gk, p["key"]); j[j >= len(gk)] = 0; p["good"] &= gk[j] != p["key"]
    worst = sorted(pcs, key=lambda p: -p["doubt"])[:5]; say("most doubted photos:", [(p["name"], round(p["doubt"], 2), p["npts"]) for p in worst])

    # ---- 3b a colour correction per photo toward the agreed colour, smooth over a few centimetres: fine enough that two photos meet in the
    # same shade where one's patch ends and the other's begins (gloss differs from photo to photo), coarse enough to leave texture alone
    Bc = 8; nc = N // Bc; cblk = (GY // Bc) * nc + GX // Bc
    for i, p in enumerate(pcs):
        good, order, st, cnt = p.pop("rows"); sl = slice(start[i], start[i + 1]); p["cgain"] = None
        if not len(st): continue
        per = np.repeat(np.where(ghost[sl, None] | (ncand[ci[sl], None] < 2), 0, cons[ci[sl]] - C_[sl]), cnt, axis=0); bk = cblk[good][order]; fld = np.zeros((nc * nc, 3)); num = np.zeros(nc * nc); np.add.at(fld, bk, per); np.add.at(num, bk, 1)
        has = (num > 6).reshape(nc, nc)
        if not has.any(): continue
        fld = (fld / np.maximum(num, 1)[:, None]).reshape(nc, nc, 3)[tuple(ndi.distance_transform_edt(~has, return_distances=False, return_indices=True))]
        p["cgain"] = np.exp(np.clip(np.stack([ndi.zoom(ndi.gaussian_filter(fld[:, :, c], 1.5), Bc, order=1) for c in range(3)], -1), -0.6, 0.6)).astype(np.float32)

    # ---- 5 own
    A_ = np.where(ghost, 0, A_); amax = np.zeros(len(cells)); np.maximum.at(amax, ci, A_); ok = (A_ >= 0.6 * amax[ci]) & ~ghost     # an owner must cover most of what any photo covers there
    own = settle_owners(cells, ci[ok], P_[ok], S_[ok], SMOOTH, rounds=8); say("own:", len(cells), "cells of %g cm," % (CELL * 100), len(np.unique(own[own >= 0])), "photos own something")
    fine = 0.012; fkey = lambda X: pack(np.floor(X / fine)); occ = np.zeros(0, np.int64); near = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)])
    grown = lambda o: np.unique(np.concatenate([pack(unpack(o) + dd) for dd in near]))
    for i, p in enumerate(pcs):
        j = np.searchsorted(cells, p["key"]); j[j >= len(cells)] = 0; p["keep"] = p["good"] & (cells[j] == p["key"]) & (own[j] == i); occ = np.union1d(occ, fkey(p["Pw"][p["keep"]]))
    n1 = sum(int(p["keep"].sum()) for p in pcs); occ = grown(occ)
    for i in np.argsort([np.median(p["zz"]) for p in pcs]):          # gaps the owners left: the nearest photo with something there fills them
        p = pcs[i]; cand = np.flatnonzero(p["good"] & ~p["keep"]); fk = fkey(p["Pw"][cand]); j = np.searchsorted(occ, fk); j[j >= len(occ)] = 0; add = cand[occ[j] != fk]; p["keep"][add] = True
        if len(add): occ = np.union1d(occ, grown(np.unique(fkey(p["Pw"][add]))))
    say("kept", n1, "splats, and", sum(int(p["keep"].sum()) for p in pcs) - n1, "more in gaps")

    # ---- 6 thin, and write
    out = [[] for _ in range(5)]; add = lambda *x: [o.append(v) for o, v in zip(out, x)]
    for i, p in enumerate(pcs):
        keep = p["keep"]
        if not keep.any(): continue
        g = load(p["im"]); rgb = np.clip(g["rgb"] * gain[i] * (p["cgain"][GY, GX] if p["cgain"] is not None else 1), 0, 1).astype(np.float32); sz = g["scale"] * p["ratio"][:, None]
        q = (Rotation.from_matrix(p["Rc"]) * Rotation.from_quat(g["quat"][:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
        if POOL > 0:
            nrm = p["nrm"].astype(np.float32); fp = p["zz"] * p["foot"]
            for b, tol in ((4, POOL * 0.6), (2, POOL)):          # 4 x 4 cells where very even, else 2 x 2
                cb, zb, nm = blocks(rgb, b), blocks(p["zz"], b), blocks(nrm, b).mean(2); nl = np.linalg.norm(nm, axis=-1)
                flat = blocks(keep, b).all(2) & (cb.std(2).max(-1) < tol) & ((zb.max(2) - zb.min(2)) < 0.03 * zb.mean(2)) & (nl > 0.97)
                if not flat.any(): continue
                sg = blocks(fp, b).mean(2)[flat] * b * 0.75
                add(blocks(p["Pw"], b).mean(2)[flat], cb.mean(2)[flat], np.full(int(flat.sum()), 0.97, np.float32), np.stack([sg, sg, sg * 0.12], 1), discs_from_normals((nm / np.maximum(nl, 1e-6)[..., None])[flat]))
                keep = keep & ~unblocks(flat, b)
        qk, sk = flatten(q[keep], sz[keep], p["nrm"][keep].astype(np.float32), FLAT * p["zz"][keep] * p["foot"]) if FLAT > 0 else (q[keep], sz[keep])
        add(p["Pw"][keep], rgb[keep], g["opacity"][keep], sk, qk)
    # ---- 7 patch: where the surface has no splat at all (behind silhouettes, steep sides), discs from the surface itself, in its averaged colour
    n_patch = 0
    if PATCH:
        mesh.compute_vertex_normals(); V = np.asarray(mesh.vertices); T = np.asarray(mesh.triangles); edge = float(np.median(np.linalg.norm(V[T[::50, 0]] - V[T[::50, 1]], axis=1)))
        fk = fkey(V); j = np.searchsorted(occ, fk); j[j >= len(occ)] = 0; free = np.flatnonzero(occ[j] != fk)
        if len(free):
            vi, ag = seen_through(V[free]); free = free[~((vi >= 2) & (vi > ag)) & (ag >= 1)]          # not in space photos look through, and at least one photo sees a surface there
            sg = np.full(len(free), 0.7 * edge, np.float32); n_patch = len(free)
            add(V[free].astype(np.float32), np.asarray(mesh.vertex_colors)[free].astype(np.float32), np.full(len(free), 0.9, np.float32), np.stack([sg, sg, sg * 0.15], 1), discs_from_normals(np.asarray(mesh.vertex_normals)[free]))
    say("patched", n_patch, "places the photos' own splats left open")
    P, rgb, op, sz, q = (np.concatenate(x) for x in out); n = write_ply(f"{d}/scene.ply", P, rgb, op, sz, q)
    poses = np.array([np.r_[np.c_[p["Rc"] @ CV, p["C"]], [[0, 0, 0, 1]]] for p in pcs]); up = poses[:, :3, 1].mean(0); up /= np.linalg.norm(up)
    s = int(np.argmin(np.linalg.norm(poses[:, :3, 3] - np.median(poses[:, :3, 3], 0), axis=1)))                       # start in the middle of where the photos were taken
    scene = {"up": up.tolist(), "center": poses[:, :3, 3].mean(0).tolist(), "poses": poses.tolist(), "eye": poses[s, :3, 3].tolist(), "forward": (-poses[s, :3, 2]).tolist(),
             "splats": n, "focal": float(pcs[s]["f"]), "width": int(pcs[s]["W"]), "metre": 1.0, "placed": len(pcs), "backdrop": backdrop(rgb)}
    with open(f"{d}/scene.json", "w") as fh: json.dump(scene, fh)
    say("wrote", n, "splats,", round(os.path.getsize(f"{d}/scene.ply") / 1e6), "MB")

if __name__ == "__main__":
    a = sys.argv; opt = lambda k, v: type(v)(a[a.index(k) + 1]) if k in a else v
    if "--one" in a: one(a[1])
    else: many(a[1], CELL=opt("--cell", 0.04), POOL=opt("--pool", 0.02), GHOST=opt("--ghost", 0.35), BIG=opt("--big", 4.0), SMOOTH=opt("--smooth", 0.5), PATCH="--no-patch" not in a, FLAT=opt("--flat", 0.0))
