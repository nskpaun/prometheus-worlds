"""What blender_finish.py left in WORLD_DIR/blender -> the rest of the world's files (thread/.venv; a few seconds):
  blender_export.py WORLD_DIR
  light0.png ..  the lightmaps: the baked light, on the second picture coordinates of scene.glb's meshes (each says which)
  env.jpg        a panorama from where the visitor stands, for reflections
  mesh.glb       everything in one colour a thing (the viewer's plain Mesh view)
  collider.glb   every solid shape, to walk against
  scene.json     where things are: all files are in metres, y up, the floor at y = 0 (Blender's x, z, -y)
scene.glb itself (the meshes with their textures) is written by Blender."""
import sys, os, json, numpy as np, trimesh
from PIL import Image

def y_up(v): return np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1).astype(np.float32)

def calm(px):
    """A lightmap with the path tracer's speckle taken out: single bright or dark texels replaced by what surrounds them,
    then the faintest blur. Light changes slowly across a surface, so nothing true is lost."""
    from scipy import ndimage
    out = np.empty_like(px)
    for c in range(3): out[..., c] = ndimage.gaussian_filter(ndimage.median_filter(px[..., c], size=5).astype(np.float32), 0.8).round().clip(0, 255)
    return out

def main(d):
    w = os.path.join(d, "blender"); g = np.load(os.path.join(w, "geo.npz"))
    with open(os.path.join(w, "meta.json")) as fh: meta = json.load(fh)
    maps = []
    for k in range(int(meta.get("lightmaps", 1))):
        Image.fromarray(calm(np.load(os.path.join(w, f"light{k}.npy")))).save(os.path.join(d, f"light{k}.png"), optimize=False, compress_level=4); maps.append(f"light{k}.png")
    Image.fromarray(np.load(os.path.join(w, "env.npy"))).save(os.path.join(d, "env.jpg"), quality=92)
    V, T = y_up(g["PV"]), g["PT"]; flat = V[T].reshape(-1, 3); col = (np.clip(g["PC"], 0, 1) * 255).astype(np.uint8)
    vc = np.concatenate([np.repeat(col, 3, 0), np.full((len(flat), 1), 255, np.uint8)], 1)
    trimesh.Trimesh(flat, np.arange(len(flat)).reshape(-1, 3), vertex_colors=vc, process=False).export(os.path.join(d, "mesh.glb"))
    trimesh.Trimesh(y_up(g["CV"]), g["CT"], process=True).export(os.path.join(d, "collider.glb"))
    lo, hi = np.array(meta["lo"]), np.array(meta["hi"]); e, f = meta["eye"], meta["forward"]; room = meta["room"]
    scene = {"model": "scene.glb", "lightmaps": maps, "env": "env.jpg", "light_max": meta.get("light_max", 8.0), "mesh": "mesh.glb", "collider": "collider.glb", "up": [0, 1, 0], "floor": 0.0, "metre": 1.0,
             "centre": [0.0, 0.0] if room else [float((lo[0] + hi[0]) / 2), float(-(lo[1] + hi[1]) / 2)], "eye": [e[0], e[2], -e[1]], "forward": [f[0], f[2], -f[1]],
             "backdrop": meta["backdrop"], "setting": meta["setting"], **({"room": [room[0], room[2], room[1]]} if room else {})}
    with open(os.path.join(d, "scene.json"), "w") as fh: json.dump(scene, fh, indent=1)
    print(f"{len(T)} faces; scene.glb {os.path.getsize(os.path.join(d, 'scene.glb')) / 1e6:.1f} MB, {len(maps)} lightmaps", flush=True)

if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]))
