"""Scores read straight from a trained splat: does it sit on the blockout, are its splats small.  A few seconds, no GPU.
splat_scores.py PLY MESH_GLB [--metre FILE]   -> one JSON line        splat_scores.py --selftest PLY MESH_GLB
blockout_mesh.py writes the mesh in the splat file's own frame, so the two need no lining up; the only number needed is
"metre" (file units per metre) from the NAME_metre.json beside the mesh. on_mesh_share is the proof: if the two did not
line up it would be near 0, and "aligned" turns false.
  stray_share / stray15_share   opacity-weighted share of splats more than 5 cm / 15 cm from any mesh surface
  large_share / big_share       opacity-weighted share whose longest axis is over 10 cm / 25 cm. Mostly round discs on
                                plain walls and ground: it says how coarse the splats are, not how stretched
  needle_share                  opacity-weighted share that is a needle: longest axis over 4 times the middle one, and
                                over 3 cm (shorter ones are too small to show as streaks)
  median_axis_cm                the median longest axis
A splat's axis is its width at half strength (2.355 x its sigma): detail finer than that is smeared away.
Outdoors the mesh is only the ground and the objects, yet the splat also holds a painted backdrop 20 m out and a sky
20 m up. Those are told apart by place (outside the ground slab, or nearer the far shell than the mesh), counted as
backdrop_share and left out of every score above: nobody walks up to them, so centimetres mean nothing there. They get
one loose check of their own, backdrop_stray_share: more than 50 cm from the shell."""
import sys, os, json, numpy as np, trimesh, open3d as o3d

STRAY, STRAY15, LONG, BIG, FAR_STRAY, NEEDLE = 0.05, 0.15, 0.10, 0.25, 0.5, 0.03     # metres
THIN = 4.0        # a needle's longest axis is over this many times its middle one
MARGIN, APRON = 20.0, 6.0     # outdoors: the shell lies 20 m beyond the play area (chisel_ring.py), the ground slab 6 m (blockout_mesh.py)
FWHM = 2.355

def read_ply(path):
    """Positions, opacity 0..1 and the longest and middle sigma (file units, as two columns) of a 3DGS PLY."""
    f = open(path, "rb"); names = []
    while True:
        l = f.readline().split()
        if not l: raise SystemExit(f"{path}: not a PLY")
        if l[0] == b"element": n = int(l[2])
        if l[0] == b"property": names.append(l[2].decode())
        if l[0] == b"end_header": break
    v = np.fromfile(f, np.dtype([(k, "<f4") for k in names]), n)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    return xyz, 1 / (1 + np.exp(-v["opacity"].astype(np.float64))), np.exp(np.sort([v["scale_0"], v["scale_1"], v["scale_2"]], 0)[:0:-1].T)

def distance(verts, faces, pts):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.core.Tensor(np.asarray(verts, np.float32)), o3d.core.Tensor(np.asarray(faces, np.uint32)))
    return s.compute_distance(o3d.core.Tensor(np.asarray(pts, np.float32))).numpy()

def split(xyz, mesh, meta):
    """Metres from each splat to the mesh, and which splats are the far backdrop (with their metres to the shell or mesh)."""
    d = distance(mesh.vertices, mesh.faces, xyz) / meta["metre"]; far = np.zeros(len(xyz), bool)
    if meta.get("setting") != "outdoor" or "fromRoom" not in meta: return d, far, d[:0]
    M = np.array(meta["fromRoom"]).reshape(4, 4); p = (xyz - M[:3, 3]) @ np.linalg.inv(M[:3, :3]).T      # room metres
    W, _, D = meta["room"]; lo, hi = np.array([-MARGIN, 0, -MARGIN]), np.array([W + MARGIN, MARGIN, D + MARGIN])
    shell = trimesh.creation.box(extents=hi - lo); shell.apply_translation((lo + hi) / 2)
    ds = distance(shell.vertices, shell.faces, p)
    off = np.max([-p[:, 0], p[:, 0] - W, -p[:, 2], p[:, 2] - D], 0)        # metres outside the play area
    far = (off > APRON) | (ds + 0.01 < d)       # the shell's floor is the mesh's ground: a tie there is not backdrop
    return d, far, np.minimum(ds, d)[far]

def score_arrays(xyz, op, sigma, mesh, meta):
    sigma = np.asarray(sigma); sigma = sigma if sigma.ndim == 2 else np.stack([sigma, sigma], 1)       # one column: round splats
    d, far, df = split(xyz, mesh, meta); axis = FWHM * sigma[:, 0] / meta["metre"]; thin = sigma[:, 0] > THIN * sigma[:, 1]; out = {"splats": len(xyz)}
    if meta.get("setting") == "outdoor":
        out.update(backdrop_share=float(far.mean()), backdrop_stray_share=float(op[far] @ (df > FAR_STRAY) / max(op[far].sum(), 1e-9)))
    d, axis, thin, w = d[~far], axis[~far], thin[~far], op[~far]; w = w / max(w.sum(), 1e-9)
    out.update(stray_share=float(w @ (d > STRAY)), stray15_share=float(w @ (d > STRAY15)), on_mesh_share=float((d <= STRAY15).mean()),
               median_off_cm=float(np.median(d) * 100), large_share=float(w @ (axis > LONG)), big_share=float(w @ (axis > BIG)),
               needle_share=float(w @ (thin & (axis > NEEDLE))),
               median_axis_cm=float(np.median(axis) * 100))
    out["aligned"] = out["on_mesh_share"] > 0.5       # a splat in another frame than the mesh would leave almost nothing on it
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}

def load(ply, glb, metre=None):
    meta = json.load(open(metre or glb.replace("_mesh.glb", "_metre.json")))
    return (*read_ply(ply), trimesh.load(glb, force="mesh"), meta)

def score(ply, glb, metre=None):
    return score_arrays(*load(ply, glb, metre))

def selftest(ply, glb, metre=None):
    """Add 5% extra splats floating in the air (and, apart, 5% large round ones and 5% needles): the shares must rise by about that much."""
    xyz, op, sigma, mesh, meta = load(ply, glb, metre); m = meta["metre"]; base = score_arrays(xyz, op, sigma, mesh, meta)
    d, far, _ = split(xyz, mesh, meta); near = np.flatnonzero(~far); n = len(near) // 20; rng = np.random.default_rng(0)
    # floaters: points in the box around the splats that lie on the mesh, kept when 0.3 to 2 m from every surface
    lo, hi = np.percentile(xyz[near[d[near] < STRAY]], [1, 99], axis=0); air = rng.uniform(lo, hi, (40 * n, 3))
    da = distance(mesh.vertices, mesh.faces, air) / m; air = air[(da > 0.3) & (da < 2)][:n]
    pick = rng.choice(near, n, replace=False); k = op[pick].sum() / op[near].sum()          # weight added / weight there
    a = score_arrays(np.r_[xyz, air], np.r_[op, op[pick]], np.r_[sigma, sigma[pick]], mesh, meta)
    b = score_arrays(np.r_[xyz, xyz[pick]], np.r_[op, op[pick]], np.r_[sigma, np.full((n, 2), 0.2 * m / FWHM)], mesh, meta)
    c = score_arrays(np.r_[xyz, xyz[pick]], np.r_[op, op[pick]], np.r_[sigma, np.tile([0.08 * m / FWHM, 0.01 * m / FWHM], (n, 1))], mesh, meta)
    want = lambda key, add: round(float((base[key] + add) / (1 + k)), 4)                    # shares are of the new, larger total
    out = {"added": len(air), "stray": [base["stray_share"], a["stray_share"], want("stray_share", k)],
           "stray15": [base["stray15_share"], a["stray15_share"], want("stray15_share", k)],
           "large": [base["large_share"], b["large_share"], want("large_share", k)],
           "big": [base["big_share"], b["big_share"], want("big_share", 0)],                # 20 cm splats are large, not big
           "needle": [base["needle_share"], c["needle_share"], want("needle_share", k)],     # 8 cm by 1 cm: needles...
           "needle_round": [base["needle_share"], b["needle_share"], want("needle_share", 0)],    # ...and round ones are not
           "large_needle": [base["large_share"], c["large_share"], want("large_share", 0)]}       # nor are 8 cm needles large
    print(json.dumps({"selftest [before, after, expected]": out}))
    assert len(air) == n and all(abs(v[1] - v[2]) < 0.002 for v in out.values() if isinstance(v, list))

if __name__ == "__main__":
    a = sys.argv[1:]; test = a[:1] == ["--selftest"]; a = a[test:]
    metre = a.pop(a.index("--metre") + 1) if "--metre" in a else None
    a = [x for x in a if x != "--metre"]
    if len(a) != 2: raise SystemExit(__doc__)
    selftest(*a, metre) if test else print(json.dumps(score(*a, metre)))
