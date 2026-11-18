"""Export a world built by the direct method for Unreal Engine, or anything else that reads glTF: the real geometry of the room and of
every object, a baked texture for every piece, and its lights. thread/.venv, CPU only, about 6 GB.
  export.py WORLD_DIR [OUT_DIR [NAME]]   -> OUT_DIR (default WORLD_DIR/export)/   (NAME: the world's name, for the files)
      NAME.glb           the level to light in Unreal: base colour textures (and roughness and metal on sculpted things), glowing lamps,
                         fittings and windows, and the lamps, fittings and sun as lights (KHR_lights_punctual)
      NAME_baked.glb     the same geometry and UVs with the splat's own traced light baked into unlit textures: it looks like the
                         viewer the moment it is opened, with no lights to set up (KHR_materials_unlit)
      README.txt         how to bring them into Unreal
Every piece is unwrapped on its own (xatlas) and its texels are coloured by the very calls the assembler colours its discs with,
so the export shows the same materials and paint as the splat. Frame: glTF's (metres, +Y up, right-handed) with the middle of the
room's floor at the origin, as the viewer shows it. Unreal's importer turns it into centimetres, Z up."""
import os, sys, io, re, json, time, struct, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image

PX_ROOM, PX_OBJ, PX_SAID = 160, 320, 480          # texels per metre: the room's surfaces, the things in it, the things the sentence names
MAX_TEX = 4096
CHUNK = 2_000_000
# what each of the paint's material classes (0 plain, 1 glass, 2 metal, 3 glow, 4 gloss) is as (roughness, metallic)
MAT_RM = {0: (0.75, 0.0), 1: (0.06, 0.0), 2: (0.25, 1.0), 3: (0.5, 0.0), 4: (0.3, 0.0)}
CD_PER_FLUX = {"lamp": 30.0, "candle": 8.0, "fire": 120.0, "screen": 25.0, "downlight": 90.0}   # the light's flux -> candela
SUN_LUX = {"day": 60000.0, "dusk": 6000.0, "night": 0.3}
say = lambda t: print(t, flush=True)
def stage(t): print("STAGE: " + t, flush=True)

# ---------------------------------------------------------------- frames
def frame(room):
    """room metres -> glTF metres: z mirrored (the viewer's handedness), the middle of the floor at the origin"""
    W, D = float(room[0]), float(room[2])
    pos = lambda P: np.c_[np.asarray(P, float)[:, 0] - W / 2, np.asarray(P, float)[:, 1], D / 2 - np.asarray(P, float)[:, 2]]
    vec = lambda N: np.asarray(N, float) * [1, 1, -1]
    return pos, vec

def quat_from_to(a, b):
    """unit quaternion (x, y, z, w) turning unit vector a onto unit vector b"""
    a = np.asarray(a, float) / np.linalg.norm(a); b = np.asarray(b, float) / np.linalg.norm(b); c = np.cross(a, b); d = float(a @ b)
    if d < -0.999999:
        ax = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0]); ax /= np.linalg.norm(ax); return [*ax.tolist(), 0.0]
    q = np.array([*c, 1 + d]); return (q / np.linalg.norm(q)).tolist()

# ---------------------------------------------------------------- unwrap and raster
_atlas = {}
def unwrap(V, F, px):
    """-> vmap (new vertex -> old), faces (new indices), uv (0..1, v down), width, height. One atlas, at most MAX_TEX a side.
    Copies of one sculpted shape share their faces: the atlas of the first is reused."""
    import xatlas
    key = (hashlib.md5(F.tobytes()).hexdigest(), len(V), px)
    if key in _atlas: return _atlas[key]
    tpu = float(px)
    for _ in range(6):
        a = xatlas.Atlas(); a.add_mesh(np.ascontiguousarray(V, np.float32), np.ascontiguousarray(F, np.uint32))
        po = xatlas.PackOptions(); po.padding = 2; po.bilinear = True; po.texels_per_unit = tpu; po.resolution = 0; po.blockAlign = True
        a.generate(xatlas.ChartOptions(), po)
        if max(a.width, a.height) <= MAX_TEX and a.atlas_count == 1: break
        tpu *= 0.95 * MAX_TEX / max(a.width, a.height, 1)
    vmap, idx, uv = a[0]
    _atlas[key] = (vmap.astype(np.int64), idx.astype(np.int64), uv.astype(np.float64), int(a.width), int(a.height), tpu); return _atlas[key]

def raster(T, W, H):
    """T (f, 3, 2) triangles in texel units -> (tri, bary) per texel (tri -1 where none). Texels a little outside a triangle are
    taken too (their barycentrics clamped), then the exact pass overwrites: no dark seams where charts end."""
    tri = np.full(W * H, -1, np.int64); bary = np.zeros((W * H, 3), np.float32)
    a, b, c = T[:, 0], T[:, 1], T[:, 2]; den = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]); ok = np.abs(den) > 1e-12
    x0 = np.clip(np.floor(T[..., 0].min(1) - 1.5), 0, W - 1).astype(np.int64); x1 = np.clip(np.ceil(T[..., 0].max(1) + 0.5), 0, W - 1).astype(np.int64)
    y0 = np.clip(np.floor(T[..., 1].min(1) - 1.5), 0, H - 1).astype(np.int64); y1 = np.clip(np.ceil(T[..., 1].max(1) + 0.5), 0, H - 1).astype(np.int64)
    wx, wy = x1 - x0 + 1, y1 - y0 + 1; n = np.where(ok, wx * wy, 0); ids = np.nonzero(n)[0]
    elen = np.stack([np.linalg.norm(T[:, (i + 1) % 3] - T[:, (i + 2) % 3], axis=1) for i in range(3)], 1)          # the edge opposite each corner
    for tol in (0.75, 0.0):
        s = 0
        while s < len(ids):
            e = s; tot = 0
            while e < len(ids) and tot < 8_000_000: tot += n[ids[e]]; e += 1
            t = ids[s:e]; cnt = n[t]; ti = np.repeat(t, cnt); loc = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            px = x0[ti] + loc % wx[ti]; py = y0[ti] + loc // wx[ti]; cx, cy = px + 0.5, py + 0.5
            l1 = ((b[ti, 0] - cx) * (c[ti, 1] - cy) - (b[ti, 1] - cy) * (c[ti, 0] - cx)) / den[ti]
            l2 = ((c[ti, 0] - cx) * (a[ti, 1] - cy) - (c[ti, 1] - cy) * (a[ti, 0] - cx)) / den[ti]; L = np.stack([l1, l2, 1 - l1 - l2], 1)
            # the weight of corner i falls to 0 on the edge opposite it: distance outside that edge in texels = -l * |den| / edge
            dist = -L * (np.abs(den[ti])[:, None] / np.maximum(elen[ti], 1e-9))
            inside = (dist <= tol).all(1)
            px, py, ti, L = px[inside], py[inside], ti[inside], np.clip(L[inside], 0, None); L /= np.maximum(L.sum(1, keepdims=True), 1e-9)
            k = py * W + px; tri[k] = ti; bary[k] = L.astype(np.float32); s = e
    return tri, bary

def fill(img, have):
    """every texel no triangle covers takes its nearest covered texel's value (gutters, so mipmaps do not bleed in black)"""
    from scipy import ndimage
    if have.all() or not have.any(): return img
    _, (iy, ix) = ndimage.distance_transform_edt(~have, return_indices=True); return img[iy, ix]

def jpeg(img, q=90):
    b = io.BytesIO(); Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(b, "JPEG", quality=q, subsampling=0); return b.getvalue()

# ---------------------------------------------------------------- the piece's own colours, as the assembler lays them
def piece_colour(p, M, P, N, UV, step):
    """-> rgb (n, 3) 0..255, mat (n,) uint8 or None. The same calls assemble.build colours its discs with."""
    import edge_discs
    n = len(P); mat = None
    if p.get("emit") is not None: return np.broadcast_to(np.float32(p["emit"]), (n, 3)).copy(), None
    out = np.empty((n, 3), np.float32)
    if "xyz" in p:
        from scipy.spatial import cKDTree
        if "o_tree" not in p: p["o_tree"] = cKDTree(p["xyz"])
        mat = np.empty(n, np.uint8)
        for s in range(0, n, CHUNK):
            sl = slice(s, s + CHUNK); out[sl] = edge_discs.colour(P[sl], N[sl], p["xyz"], p["nrm"], p["rgb"], tree=p["o_tree"]); mat[sl] = p["mat"][p["o_tree"].query(P[sl], workers=4)[1]]
        return out, mat
    for s in range(0, n, CHUNK):
        sl = slice(s, s + CHUNK)
        if p.get("fn"): out[sl] = p["fn"](P[sl], N[sl], step)
        elif p["flat"]: out[sl] = M.colour(p["key"], UV[sl, 0], UV[sl, 1], step, p["tint"], p["seed"], flip_v_height=True if p.get("wall_v") else None)
        else:
            ax = np.abs(N[sl]).argmax(1); u = np.where(ax == 0, P[sl, 2], P[sl, 0]); v = np.where(ax == 1, P[sl, 2], P[sl, 1]); out[sl] = M.colour(p["key"], u, v, step, p["tint"], p["seed"])
    return out, None

def material_of(p, light):
    """-> (roughness, metallic, emissive strength or 0) for a piece with no per-texel class"""
    if p.get("emit") is not None: return 0.5, 0.0, (1.0 if p["name"].startswith("shell:") and np.mean(p["emit"]) < 240 else 4.0)
    if p.get("mcls"):
        cls, g = p["mcls"]
        if cls == light.MIRROR: return 0.02, 1.0, 0.0
        return float(np.clip(1 - 0.85 * g, 0.05, 0.9)), (1.0 if cls == light.METAL else 0.0), 0.0
    g = float((p.get("gloss") or (0.0, 0))[0]); return float(np.clip(0.8 - 0.6 * g, 0.15, 0.85)), 0.0, 0.0

# ---------------------------------------------------------------- the splat's lit colour (the baked file)
class Lit:
    """the written splat's discs (both files), their colours and facings, in room metres"""
    C0 = 0.28209479177387814
    def __init__(self, world, scene):
        from scipy.spatial import cKDTree
        A = np.asarray(scene["fromRoom"], float).reshape(4, 4); Ai = np.linalg.inv(A); xyz, rgb, nrm = [], [], []
        for f in ("world.ply", "world_sh.ply"):
            path = f"{world}/{f}"
            if not os.path.exists(path): continue
            with open(path, "rb") as fh:
                head = b""
                while not head.endswith(b"end_header\n"): head += fh.readline()
                names = [l.split()[-1].decode() for l in head.split(b"\n") if l.startswith(b"property")]; n = int(re.search(rb"element vertex (\d+)", head).group(1))
                a = np.fromfile(fh, "<f4", n * len(names)).reshape(n, len(names))
            col = {k: i for i, k in enumerate(names)}; p = a[:, [col["x"], col["y"], col["z"]]].astype(np.float64)
            xyz.append(p @ Ai[:3, :3].T + Ai[:3, 3]); rgb.append(np.clip(0.5 + self.C0 * a[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]], 0, 1) * 255)
            w, x, y, z = (a[:, col[f"rot_{i}"]] for i in range(4)); q = np.sqrt(w * w + x * x + y * y + z * z) + 1e-12; w, x, y, z = w / q, x / q, y / q, z / q
            nv = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1); nv = nv @ np.linalg.inv(A[:3, :3]).T; nrm.append(nv / np.linalg.norm(nv, axis=1, keepdims=True))
        self.ok = bool(xyz)
        if self.ok: self.xyz = np.concatenate(xyz); self.rgb = np.concatenate(rgb).astype(np.float32); self.nrm = np.concatenate(nrm).astype(np.float32); self.tree = cKDTree(self.xyz)
    def at(self, P, N, alb, step):
        out = alb * 0.6
        for s in range(0, len(P), CHUNK):
            sl = slice(s, s + CHUNK); dd, jj = self.tree.query(P[sl], k=8, distance_upper_bound=max(4 * step, 0.03), workers=8); have = np.isfinite(dd); jj = np.minimum(jj, len(self.xyz) - 1)
            agree = ((self.nrm[jj] * N[sl][:, None, :]).sum(-1) > 0.35) & have; w = agree / (np.where(have, dd, 1.0) + 1e-3) ** 2; ws = w.sum(1); got = ws > 0
            c = (self.rgb[jj] * w[..., None]).sum(1) / np.maximum(ws, 1e-12)[:, None]; o = out[sl]; o[got] = c[got]; out[sl] = o
        return out

# ---------------------------------------------------------------- glTF
class GLB:
    def __init__(self, unlit=False):
        self.bin = bytearray(); self.unlit = unlit
        self.j = {"asset": {"version": "2.0", "generator": "Prometheus Worlds export"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"name": "World", "children": []}],
                  "meshes": [], "materials": [], "textures": [], "images": [], "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
                  "accessors": [], "bufferViews": [], "buffers": []}
        self.lights = []
    def view(self, data, target=None):
        while len(self.bin) % 4: self.bin.append(0)
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target: v["target"] = target
        self.bin += data; self.j["bufferViews"].append(v); return len(self.j["bufferViews"]) - 1
    def acc(self, a, ctype, typ, target, minmax=False):
        a = np.ascontiguousarray(a); d = {"bufferView": self.view(a.tobytes(), target), "componentType": ctype, "count": len(a), "type": typ}
        if minmax: d["min"] = a.min(0).tolist(); d["max"] = a.max(0).tolist()
        self.j["accessors"].append(d); return len(self.j["accessors"]) - 1
    def texture(self, data):
        self.j["images"].append({"bufferView": self.view(data), "mimeType": "image/jpeg"}); self.j["textures"].append({"sampler": 0, "source": len(self.j["images"]) - 1}); return len(self.j["textures"]) - 1
    def material(self, name, base, rm=None, rough=0.8, metal=0.0, emit=0.0):
        pbr = {"baseColorTexture": {"index": self.texture(base)}, "roughnessFactor": 1.0 if rm else rough, "metallicFactor": 1.0 if rm else metal}
        if rm: pbr["metallicRoughnessTexture"] = {"index": self.texture(rm)}
        m = {"name": name, "pbrMetallicRoughness": pbr}
        if self.unlit: m["extensions"] = {"KHR_materials_unlit": {}}; pbr["metallicFactor"] = 0.0; pbr["roughnessFactor"] = 1.0; pbr.pop("metallicRoughnessTexture", None)
        elif emit:
            m["emissiveTexture"] = {"index": pbr["baseColorTexture"]["index"]}; m["emissiveFactor"] = [1.0, 1.0, 1.0]
            if emit != 1.0: m["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": float(emit)}}
        self.j["materials"].append(m); return len(self.j["materials"]) - 1
    def mesh(self, name, P, N, UV, I, mat, parent):
        a = {"POSITION": self.acc(P.astype(np.float32), 5126, "VEC3", 34962, True), "NORMAL": self.acc(N.astype(np.float32), 5126, "VEC3", 34962), "TEXCOORD_0": self.acc(UV.astype(np.float32), 5126, "VEC2", 34962)}
        self.j["meshes"].append({"name": name, "primitives": [{"attributes": a, "indices": self.acc(I.astype(np.uint32).reshape(-1), 5125, "SCALAR", 34963), "material": mat}]})
        return self.node({"name": name, "mesh": len(self.j["meshes"]) - 1}, parent)
    def node(self, d, parent=0):
        self.j["nodes"].append(d); k = len(self.j["nodes"]) - 1; self.j["nodes"][parent].setdefault("children", []).append(k); return k
    def light(self, name, kind, colour, intensity, pos, direction=None, cone=None):
        l = {"name": name, "type": kind, "color": [float(c) for c in colour], "intensity": float(intensity)}
        if kind == "spot": l["spot"] = {"innerConeAngle": float(cone[0]), "outerConeAngle": float(cone[1])}
        self.lights.append(l); d = {"name": name, "translation": [float(x) for x in pos], "extensions": {"KHR_lights_punctual": {"light": len(self.lights) - 1}}}
        if direction is not None: d["rotation"] = quat_from_to([0, 0, -1], direction)
        return self.node(d, self.lights_node)
    def save(self, path):
        j = self.j; used = []
        if self.lights: j.setdefault("extensions", {})["KHR_lights_punctual"] = {"lights": self.lights}; used.append("KHR_lights_punctual")
        if any("KHR_materials_unlit" in (m.get("extensions") or {}) for m in j["materials"]): used.append("KHR_materials_unlit")
        if any("KHR_materials_emissive_strength" in (m.get("extensions") or {}) for m in j["materials"]): used.append("KHR_materials_emissive_strength")
        if used: j["extensionsUsed"] = used
        while len(self.bin) % 4: self.bin.append(0)
        j["buffers"] = [{"byteLength": len(self.bin)}]; js = json.dumps(j, separators=(",", ":")).encode()
        while len(js) % 4: js += b" "
        with open(path + ".part", "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(self.bin))); f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942)); f.write(self.bin)
        os.replace(path + ".part", path)

# ---------------------------------------------------------------- lights
def lights_of(Lv, pos, vec):
    """the light's own emitters for this level (as assemble.build finds them) -> [(name, type, colour 0..1, intensity, position, direction, cone)]"""
    import light
    b = Lv["b"]; blockers = [x for x in Lv["lm"] if x["blocker"]]; remap = np.full(len(Lv["lm"]), -1, np.int64); remap[[i for i, x in enumerate(Lv["lm"]) if x["blocker"]]] = np.arange(len(blockers))
    lobj = [dict(o, mesh=int(remap[o["mesh"]]) if o["mesh"] is not None else -1) for o in Lv["lobj"]]; ops, fits = [], []
    for e in Lv["S"]["emitters"]:
        r = np.asarray(e["rect"], float); d = {"kind": e["kind"], "c": r.mean(0), "u": (r[1] - r[0]) / 2, "v": (r[0] - r[3]) / 2, "n": np.asarray(e["normal"], float)}; (ops if e["kind"] in ("window", "skylight") else fits).append(d)
    mood = Lv["mood"]; out = []
    for k, e in enumerate(light.find_emitters([o for o in lobj if o["mesh"] >= 0], ops, fits, b["request"], blockers, mood)):
        col = np.asarray(e.get("col", (1, 1, 1)), float); col = col / max(float(col.max()), 1e-6); cls = e.get("cls", "")
        if e["type"] == "sun":
            out.append(("Moon" if mood["time"] == "night" else "Sun", "directional", col, SUN_LUX.get(mood["time"], 20000.0), pos(np.asarray([[b_ / 2 for b_ in Lv["room"]]]))[0], vec(-np.asarray(e["dir"], float)), None))
        elif e["type"] == "mesh":
            V = np.asarray(blockers[e["mesh"]]["V"], float); c = (V.min(0) + V.max(0)) / 2
            out.append((f"{e.get('name') or cls} light", "point", col, CD_PER_FLUX.get(cls, 30.0) * e["flux"], pos(c[None])[0], None, None))
        elif cls != "opening":           # windows give their light as glowing glass and the sun; fires, screens and fittings as spots
            out.append((f"{e.get('name') or cls} light", "spot", col, CD_PER_FLUX.get(cls, 60.0) * e["flux"], pos(np.asarray(e["c"], float)[None])[0], vec(np.asarray(e["n"], float)), (0.6, 1.25)))
    return out

README = """{name}: exported from Prometheus Worlds

{name}.glb
    The level to light in Unreal: real geometry, base colour textures (roughness and metal on sculpted
    things), glowing lamps and windows, and {n_lights} lights (lamps, fittings{sun}).
{name}_baked.glb
    The same level with the viewer's traced light baked into unlit textures. It looks like the viewer as
    soon as it is placed and needs no lights. {baked_note}

Bringing it into Unreal Engine 5 (5.3 or later). These steps are written from Unreal's documentation and
have not yet been tried in Unreal itself; the files were checked in Blender.
  1. File > Import Into Level..., then pick {name}.glb. This places every piece where it belongs and makes
     the lights. Dragging the file into the Content Browser brings in the meshes and materials only.
  2. The file is in metres, +Y up; Unreal turns it into centimetres, Z up. The middle of the room's floor
     is at the origin.
  3. The lights are in real units (candela; lux for the sun). Indoors, add a Post Process Volume with
     Infinite Extent and set its exposure to Manual, or narrow auto exposure's range, until it looks right.
  4. The baked file: the same steps. Its materials are unlit, so set exposure to Manual at 0 to keep the
     colours as the viewer shows them.
  5. Collision: on each Static Mesh, set Collision Complexity to Use Complex Collision As Simple, or let
     Unreal generate simple collision.

{pieces} pieces, {tris} triangles, {texels:.0f} million texels of texture.
"""

# ---------------------------------------------------------------- the export
def export(world, out=None, name=None):
    import assemble, light
    t00 = time.time(); world = os.path.abspath(world); out = out or f"{world}/export"; os.makedirs(out, exist_ok=True)
    scene = json.load(open(f"{world}/scene.json")); name = re.sub(r"[^A-Za-z0-9_-]+", "_", name or os.path.basename(world)).strip("_") or "world"
    for f in os.listdir(out):            # an earlier export of this world (perhaps under another name) goes
        if f.endswith((".glb", ".glb.part")): os.remove(os.path.join(out, f))
    stage("reading the level")
    Lv = assemble.level(world); M = Lv["M"]; pos, vec = frame(Lv["room"])
    lit = Lit(world, scene) if os.path.exists(f"{world}/world.ply") else None
    say(f"{len(Lv['pieces'])} pieces" + ("" if lit and lit.ok else "; no splat found: the baked file is left out"))
    G, B = GLB(), (GLB(unlit=True) if lit and lit.ok else None)
    groups = {}
    for g in (G, B):
        if g is None: continue
        g.room_node = g.node({"name": "Room"}); g.obj_node = g.node({"name": "Objects"}); g.lights_node = g.node({"name": "Lights"})
    stage("baking the textures")
    tris = 0; texels = 0; t0 = time.time()
    for k, p in enumerate(Lv["pieces"]):
        V = np.asarray(p["V"], np.float64); F = np.asarray(p["F"], np.int64)
        n3 = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); ar = np.linalg.norm(n3, axis=1) / 2; keep = ar > 1e-10; F, n3, ar = F[keep], n3[keep], ar[keep]
        if not len(F): continue
        fN = n3 / (2 * ar)[:, None]
        px = PX_ROOM if p["name"].startswith("shell:") else (PX_SAID if p.get("said") else PX_OBJ)
        if p.get("emit") is not None: px = min(px, 32)          # one colour: a few texels are enough
        vmap, I, uv, W, H, tpu = unwrap(V, F, px)
        P0 = V[vmap]; T = uv[I] * [W, H]
        tri, bary = raster(T, W, H); have = tri >= 0; t_ = tri[have]; b_ = bary[have].astype(np.float64)
        P = (P0[I[t_]] * b_[..., None]).sum(1)
        if p["flat"]: N = fN[t_]
        else:           # smooth normals on a sculpted or curved thing (what the paint was laid by); flat faces keep their own
            vn = np.zeros_like(V); np.add.at(vn, F.reshape(-1), np.repeat(n3, 3, axis=0)); vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
            N = (vn[vmap][I[t_]] * b_[..., None]).sum(1); N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        UV = None
        if p["flat"]:
            UVf = np.asarray(p["UV"], np.float64)[F]; ua, ub = UVf[:, 1] - UVf[:, 0], UVf[:, 2] - UVf[:, 0]; uar = np.abs(ua[:, 0] * ub[:, 1] - ua[:, 1] * ub[:, 0]) / 2
            bad = (uar < 0.25 * ar) | (uar > 4 * ar)             # a face whose UV is not true scale: its own plane, as the assembler lays it
            UV = (UVf[t_] * b_[..., None]).sum(1); bt = bad[t_]
            if bt.any():
                e1 = V[F[t_[bt], 1]] - V[F[t_[bt], 0]]; e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(fN[t_[bt]], e1); UV[bt] = np.stack([(P[bt] * e1).sum(1), (P[bt] * e2).sum(1)], 1)
        step = 1.0 / tpu
        rgb, mat = piece_colour(p, M, P, N, UV, step)
        img = np.zeros((H * W, 3), np.float32); img[have] = rgb; img = fill(img.reshape(H, W, 3), have.reshape(H, W))
        rough, metal, emit = material_of(p, light); rm = None
        if mat is not None:
            r = np.zeros((H * W, 3), np.float32); look = np.array([MAT_RM.get(m, MAT_RM[0]) for m in range(256)], np.float32)
            r[have, 1] = look[mat, 0] * 255; r[have, 2] = look[mat, 1] * 255; rm = jpeg(fill(r.reshape(H, W, 3), have.reshape(H, W)))
        # the mesh in glTF's frame: z mirrored, so the winding turns too
        Pv = pos(P0); Iv = I[:, [0, 2, 1]]
        if p["flat"]:
            acc = np.zeros_like(P0); np.add.at(acc, I.reshape(-1), np.repeat(n3, 3, axis=0)); Nv = vec(acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12))
        else: Nv = vec(vn[vmap])
        baked = _baked(lit, P, N, rgb, step, have, H, W) if B is not None else None
        nm = re.sub(r"[^A-Za-z0-9_ :-]+", "", p["name"]).replace("shell:", "") or f"piece {k}"; nm = f"{nm} {k:03d}"
        for g in (G, B):
            if g is None: continue
            if p["name"].startswith("shell:"): parent = g.room_node
            else:
                key = (id(g), p.get("obj"))
                if key not in groups: groups[key] = g.node({"name": re.sub(r"[^A-Za-z0-9_ -]+", "", p["name"].split(":")[0]) or "thing"}, g.obj_node)
                parent = groups[key]
            if g is G: mi = g.material(nm, jpeg(img), rm, rough, metal, emit)
            else: mi = g.material(nm, jpeg(baked))
            g.mesh(nm, Pv, Nv, uv, Iv, mi, parent)
        tris += len(F); texels += W * H
        if k % 20 == 0: say(f"  {k + 1} of {len(Lv['pieces'])} pieces, {time.time() - t0:.0f} s")
    stage("adding the lights")
    L = lights_of(Lv, pos, vec)
    for (ln, kind, col, inten, p_, d_, cone) in L:
        G.light(ln, kind, col, inten, p_, d_, cone)
    stage("writing the files")
    G.save(f"{out}/{name}.glb")
    if B is not None: B.save(f"{out}/{name}_baked.glb")
    with open(f"{out}/README.txt", "w") as f:
        f.write(README.format(name=name, n_lights=len(L), sun=", the sun" if any(x[1] == "directional" for x in L) else "", pieces=len(Lv["pieces"]), tris=tris, texels=texels / 1e6,
                              baked_note="" if B is not None else "(Not made: this world has no splat file.)"))
    sizes = {f: round(os.path.getsize(f"{out}/{f}") / 1e6, 1) for f in os.listdir(out) if f.endswith(".glb")}
    info = {"pieces": len(Lv["pieces"]), "triangles": tris, "texels_M": round(texels / 1e6, 1), "lights": len(L), "files_mb": sizes, "seconds": round(time.time() - t00, 1)}
    json.dump(info, open(f"{out}/export.json", "w"), indent=1)
    say(f"exported {len(Lv['pieces'])} pieces, {tris // 1000}k triangles, {texels / 1e6:.0f} M texels and {len(L)} lights to {out} in {time.time() - t00:.0f} s")
    stage("done"); return info

def _baked(lit, P, N, rgb, step, have, H, W):
    img = np.zeros((H * W, 3), np.float32); img[have] = lit.at(P, N, rgb, step); return fill(img.reshape(H, W, 3), have.reshape(H, W))

if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, sys.argv[3] if len(sys.argv) > 3 else None)
