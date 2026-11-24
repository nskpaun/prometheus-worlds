"""The scene put together in Blender, made ready to walk in (run by pipeline.py, inside Blender, with no window):
  blender -b WORLD_DIR/blender/scene.blend --python blender_finish.py -- WORLD_DIR [SIZE] [SAMPLES]
What a thing looks like and how it is lit are kept apart, as a game engine keeps them, so textures stay as sharp as they
were made and only the light is baked:
  - a thing that came with its own texture (the level's pieces) keeps it;
  - things with no texture of their own (what the model built from shapes) are joined and their colour and roughness
    baked into one picture each for the room and for the rest;
  - the LIGHT (lamps, sky, sun, shadows, light bounced between surfaces; Cycles) goes into two lightmaps SIZE pixels
    square (default 4096), one for the room and one for the things, on a second set of picture coordinates, and for
    dense sculpted meshes into their corners;
  - a panorama from where the visitor stands, for the reflections the viewer adds on floors, metal and glass.
Brightness is set here too: everything is written at one exposure found from the baked light itself, so a scene lit ten
times too weakly or too strongly still comes out well exposed.
Writes WORLD_DIR/scene.glb (Blender's own glTF export: the meshes, their textures, the lightmap coordinates and corner
light) and, into WORLD_DIR/blender, light0.npy, light1.npy and env.npy (the lightmaps and the panorama, 8 bit sRGB, 1.0 = LM_MAX
times a lit white surface), geo.npz (the solid shapes to walk against, and a plainly coloured copy of everything) and
meta.json. blender_export.py turns the rest into the world's files."""
import sys, os, json, math, time
import bpy, bmesh
import numpy as np
from mathutils import Vector

LM_MAX = 8.0            # the brightest light the pictures can hold, in units of the exposure's white
DENSE = 4000            # faces from which a mesh is lit at its corners instead of through the lightmap (a sculpted thing: too many small faces to lay out)
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)

def gpu():
    sc = bpy.context.scene; sc.render.engine = "CYCLES"
    try:
        p = bpy.context.preferences.addons["cycles"].preferences; p.compute_device_type = "METAL"; p.get_devices()
        for dv in p.devices: dv.use = dv.type == "METAL"
        sc.cycles.device = "GPU"; print("Cycles on", [dv.name for dv in p.devices if dv.use], flush=True)
    except Exception as e: print("Cycles stays on the CPU:", e)

def flat_copy(objs, name):
    """One mesh object holding all of `objs` as they stand in the world, modifiers applied, materials kept."""
    dg = bpy.context.evaluated_depsgraph_get(); bm = bmesh.new(); mats = []
    for o in objs:
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg)); me.transform(o.matrix_world)
        if o.matrix_world.determinant() < 0: me.flip_normals()
        slots = [s.material for s in o.material_slots] or [None]; remap = []
        for m in slots:
            if m not in mats: mats.append(m)
            remap.append(mats.index(m))
        tmp = bmesh.new(); tmp.from_mesh(me)
        for f in tmp.faces: f.material_index = remap[min(f.material_index, len(remap) - 1)]
        tmp.to_mesh(me); tmp.free(); bm.from_mesh(me); bpy.data.meshes.remove(me)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    out = bpy.data.meshes.new(name); bm.to_mesh(out); bm.free()
    plain = bpy.data.materials.get("plain") or bpy.data.materials.new("plain")
    try: plain.use_nodes = True
    except Exception: pass
    for m in mats: out.materials.append(m or plain)
    o = bpy.data.objects.new(name, out); bpy.context.scene.collection.objects.link(o); return o

def drop_unseen(o, room, t=0.12):
    """Take the outer faces of the room's shell off the mesh: nobody sees them, and they would take half the picture."""
    if not room: return
    W, D, H = room; bm = bmesh.new(); bm.from_mesh(o.data); e = 0.005
    x0, y0, x1, y1 = bpy.context.scene.get("room_box") or (-W / 2, -D / 2, W / 2, D / 2)
    top = H + float(bpy.context.scene.get("roof_rise", 0.0))
    gone = [f for f in bm.faces if (c := f.calc_center_median()) and (c.x < x0 - t + e or c.x > x1 + t - e or c.y < y0 - t + e or c.y > y1 + t - e or c.z < -e or c.z > top + e)]
    bmesh.ops.delete(bm, geom=gone, context="FACES"); bm.to_mesh(o.data); bm.free()

def only(objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]

def unwrap(objs, layer=None, margin=0.0015):
    """Lay the faces of these objects out together on one square, into the named coordinate layer (made if missing)."""
    for o in objs:
        if layer:
            uv = o.data.uv_layers.get(layer) or o.data.uv_layers.new(name=layer, do_init=False); o.data.uv_layers.active = uv
    only(objs); bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=margin, area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")

def target(objs, img):
    """Point every material of these objects at the picture a bake is to fill. -> the nodes added, to take out again"""
    added = []
    for m in {s.material for o in objs for s in o.material_slots if s.material}:
        nt = m.node_tree; n = nt.nodes.new("ShaderNodeTexImage"); n.image = img; nt.nodes.active = n; added.append((nt, n))
    return added

def untarget(added):
    for nt, n in added: nt.nodes.remove(n)

def pixels(img):
    px = np.empty(img.size[0] * img.size[1] * 4, np.float32); img.pixels.foreach_get(px); return px.reshape(img.size[1], img.size[0], 4)

def bake_look(o, size):
    """A joined mesh of untextured things -> one material with its colour and roughness as pictures on its first coordinates."""
    sc = bpy.context.scene; unwrap([o]); only([o]); sc.render.bake.margin = 8; made = []
    for kind, flt, space in (("DIFFUSE", {"COLOR"}, "sRGB"), ("ROUGHNESS", None, "Non-Color")):
        img = bpy.data.images.new(f"{o.name}_{kind.lower()}", size, size, alpha=False); img.colorspace_settings.name = space
        added = target([o], img)
        if flt: bpy.ops.object.bake(type=kind, pass_filter=flt, margin=8, use_clear=True)
        else: bpy.ops.object.bake(type=kind, margin=8, use_clear=True)
        untarget(added); img.pack(); made.append(img)
    m = bpy.data.materials.new(o.name)
    try: m.use_nodes = True
    except Exception: pass
    nt = m.node_tree; b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    for img, key in zip(made, ("Base Color", "Roughness")):
        n = nt.nodes.new("ShaderNodeTexImage"); n.image = img; nt.links.new(n.outputs["Color"], b.inputs[key])
    o.data.materials.clear(); o.data.materials.append(m)
    for p in o.data.polygons: p.material_index = 0

def has_look(o):
    """Does this object carry its own texture: coordinates, and a picture feeding every material's colour?"""
    if not o.data.uv_layers or not o.material_slots: return False
    for s in o.material_slots:
        b = next((n for n in s.material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if s.material and s.material.node_tree else None
        if b is None or not b.inputs["Base Color"].is_linked or not any(n.type == "TEX_IMAGE" for n in s.material.node_tree.nodes): return False
        if any(n.type == "TEX_COORD" for n in s.material.node_tree.nodes): return False       # laid on by position (the kit's swatches), not by coordinates
    return True

def relief(objs, strength=1.5, made=None):
    """A normal map for each picture that colours these objects, from the picture itself: its fine light and dark read as
    small bumps (the grain of plaster, boards, stone), the broad changes left out. The glTF carries it, and the viewer
    shades it as if lit from above; Cycles bakes on as before. -> number of maps made"""
    made = {} if made is None else made
    for m in {s.material for o in objs for s in o.material_slots if s.material and s.material.node_tree}:
        nt = m.node_tree; b = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if b is None or b.inputs["Normal"].is_linked or not b.inputs["Base Color"].is_linked: continue
        src = b.inputs["Base Color"].links[0].from_node
        if src.type != "TEX_IMAGE" or src.image is None or src.image.size[0] < 64: continue
        key = src.image.name
        if key not in made:
            px = pixels(src.image)[..., :3] @ LUM; h = px.shape[0]; w = px.shape[1]
            k = max(1, int(round(min(h, w) / 512)))                                 # work at about 512 px, then the map is scaled up by Blender's sampling
            small = px[::k, ::k].astype(np.float32)
            def blur(a, r):
                c = np.cumsum(np.pad(a, ((r + 1, r), (0, 0)), mode="wrap"), 0); a = (c[2 * r + 1:] - c[:-2 * r - 1]) / (2 * r + 1)
                c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r)), mode="wrap"), 1); return (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)
            hgt = blur(small, 1) - blur(small, 12); hgt /= max(float(np.percentile(np.abs(hgt), 98)), 1e-4)       # fine detail only, about -1..1
            gx = (np.roll(hgt, -1, 1) - np.roll(hgt, 1, 1)) * 0.5 * strength / 8; gy = (np.roll(hgt, -1, 0) - np.roll(hgt, 1, 0)) * 0.5 * strength / 8
            n = np.stack([-gx, -gy, np.ones_like(gx)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
            img = bpy.data.images.new(key + "_relief", n.shape[1], n.shape[0], alpha=False); img.colorspace_settings.name = "Non-Color"
            img.pixels.foreach_set(np.concatenate([n * 0.5 + 0.5, np.ones(n.shape[:2] + (1,), np.float32)], -1).astype(np.float32).ravel()); img.pack(); made[key] = img
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = made[key]; t.interpolation = "Linear"
        if src.inputs["Vector"].is_linked: nt.links.new(src.inputs["Vector"].links[0].from_socket, t.inputs["Vector"])
        nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 1.0; nt.links.new(t.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    return len(made)

def bake_light(sets, dense, size, samples):
    """The light alone (no surface colour): onto a lightmap for each set of objects in `sets`, into the corners of the
    `dense` ones. -> the lightmaps' pixels (linear, alpha where a face is), one per set."""
    sc = bpy.context.scene; c = sc.cycles; c.samples = samples; c.use_denoising = True; c.max_bounces = 6; c.diffuse_bounces = 4; c.sample_clamp_indirect = 8.0
    try: c.denoiser = "OPENIMAGEDENOISE"
    except Exception: pass
    out = []
    for k, flat in enumerate(sets):
        for o in flat:
            if o.data.users > 1: o.data = o.data.copy()                    # copies of one mesh are each lit where they stand
        unwrap(flat, "light", margin=0.004); img = bpy.data.images.new(f"lightmap{k}", size, size, alpha=True, float_buffer=True); img.colorspace_settings.name = "Linear Rec.709"
        added = target(flat, img); only(flat); sc.render.bake.margin = 8
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={"DIRECT", "INDIRECT"}, margin=8, use_clear=True); untarget(added); out.append(pixels(img)); bpy.data.images.remove(img)
        for o in flat: o.data.uv_layers.active = o.data.uv_layers[0]; o["lm"] = k + 1
    for o in dense:
        col = o.data.color_attributes.get("Col") or o.data.color_attributes.new("Col", "FLOAT_COLOR", "POINT"); o.data.color_attributes.active_color = col
        try: o.data.color_attributes.render_color_index = list(o.data.color_attributes).index(col)
        except Exception: pass
    if dense: only(dense); bpy.ops.object.bake(type="DIFFUSE", pass_filter={"DIRECT", "INDIRECT"}, target="VERTEX_COLORS")
    return out

def corner_light(o):
    a = o.data.color_attributes["Col"]; v = np.empty(len(a.data) * 4, np.float32); a.data.foreach_get("color", v); return v.reshape(-1, 4)

def encode(x, gain):
    """Linear light -> 8 bit sRGB in which 1.0 stands for LM_MAX times the exposure's white."""
    x = (np.nan_to_num(x, nan=0.0, posinf=LM_MAX * 100, neginf=0.0).clip(0, None) * (gain / LM_MAX)).clip(0, 1)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-8), 1 / 2.4) - 0.055)
    return (x * 255 + 0.5).astype(np.uint8)

def panorama(eye, path, samples=48, width=1024):
    """What is seen all round from a point, as Cycles has it: an equirectangular picture, linear, upright, its middle
    looking along +x (which is where the viewer's lookup has its middle)."""
    sc = bpy.context.scene; cam = bpy.data.objects.new("pano", bpy.data.cameras.new("pano")); sc.collection.objects.link(cam); cam.data.type = "PANO"
    try: cam.data.panorama_type = "EQUIRECTANGULAR"
    except Exception: cam.data.cycles.panorama_type = "EQUIRECTANGULAR"
    cam.location = eye; cam.rotation_euler = (math.radians(90), 0, math.radians(-90)); sc.camera = cam
    rd = sc.render; rd.resolution_x, rd.resolution_y, rd.resolution_percentage = width, width // 2, 100; rd.image_settings.file_format = "OPEN_EXR"; rd.filepath = path; sc.cycles.samples = samples
    bpy.ops.render.render(write_still=True); img = bpy.data.images.load(path); px = pixels(img)[..., :3].copy(); bpy.data.images.remove(img); bpy.data.objects.remove(cam); return px

def write_exr(path, a, flip=True):
    """An H x W x 3 float array (row 0 at the top when flip, else Blender's bottom-up order) -> a half-float EXR."""
    a = np.ascontiguousarray(a[::-1] if flip else a, np.float32); H, W = a.shape[:2]
    img = bpy.data.images.new(os.path.basename(path), W, H, alpha=False, float_buffer=True)
    img.pixels.foreach_set(np.concatenate([a, np.ones((H, W, 1), np.float32)], -1).ravel())
    sc = bpy.context.scene; s = sc.render.image_settings; old = (s.file_format, s.color_depth, s.exr_codec)
    s.file_format = "OPEN_EXR"; s.color_depth = "16"; s.exr_codec = "ZIP"
    img.save_render(path, scene=sc); s.file_format, s.color_depth, s.exr_codec = old; bpy.data.images.remove(img)

def calm_hdr(x):
    """The path tracer's speckle out of a float lightmap: each texel the median of its 3 x 3 neighbourhood, where that
    neighbourhood is lit (the black between patches is left alone). Blender's Python has numpy but not scipy."""
    lum = x @ LUM; H, W = lum.shape; pad = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="edge")
    stack = np.stack([pad[i:i + H, j:j + W] for i in range(3) for j in range(3)], 0)
    med = np.median(stack, axis=0); return np.where((lum > 0)[..., None], med, x)

def probe_points(room, lo, hi, eye):
    """Where the reflection probes go: the start, then the middles of the four quarters of the floor (a small room
    gets two), all at eye height; outdoors, the start and the middle."""
    pts = [list(eye)]
    if room:
        W, D = hi.x - lo.x, hi.y - lo.y; n = 4 if W * D > 30 else 2
        cand = [(lo.x + W * fx, lo.y + D * fy) for fx, fy in ((0.28, 0.28), (0.72, 0.72), (0.72, 0.28), (0.28, 0.72))][:n]
    else: cand = [(0.0, 0.0)]
    for x, y in cand:
        if all(math.hypot(x - p[0], y - p[1]) > 1.2 for p in pts): pts.append([x, y, 1.5])
    return pts[:4]

def stand(things, room, lo, hi):
    """Where to put the visitor: a clear spot of floor far from the middle of the things, looking at them."""
    rb = bpy.context.scene.get("room_box")
    if room and rb: x0, x1, y0, y1 = rb[0] + 0.5, rb[2] - 0.5, rb[1] + 0.5, rb[3] - 0.5
    elif room: x0, x1, y0, y1 = -room[0] / 2 + 0.5, room[0] / 2 - 0.5, -room[1] / 2 + 0.5, room[1] / 2 - 0.5
    else:                                                    # outdoors: round the things, not out at the edge of the ground
        x0, x1, y0, y1 = (min(a.x for a, b in things) - 2, max(b.x for a, b in things) + 2, min(a.y for a, b in things) - 2, max(b.y for a, b in things) + 2) if things else (-3, 3, -3, 3)
        x0, x1, y0, y1 = max(x0, lo.x + 0.5), min(x1, hi.x - 0.5), max(y0, lo.y + 0.5), min(y1, hi.y - 0.5)
    mid = sum((Vector(((a.x + b.x) / 2, (a.y + b.y) / 2, 0)) for a, b in things), Vector()) / max(len(things), 1); best = None
    for x in np.linspace(x0, x1, 15):
        for y in np.linspace(y0, y1, 15):
            clear = min([max(a.x - x, x - b.x, a.y - y, y - b.y) for a, b in things if a.z < 1.8 and b.z > 0.05] or [9.0])
            if clear < 0.3: continue
            score = min(clear, 0.8) * 2 + (Vector((x, y, 0)) - mid).length
            if best is None or score > best[0]: best = (score, x, y)
    x, y = (best[1], best[2]) if best else ((x0 + x1) / 2, (y0 + y1) / 2); f = Vector((mid.x - x, mid.y - y, 0))
    f = f.normalized() if f.length > 0.05 else Vector((0, 1, 0)); return [float(x), float(y), 1.6], [f.x, f.y, -0.12]

def plain_colour(o):
    """One colour for an object, for the viewer's plain Mesh view: its picture's middle tone, or its material's own colour."""
    for s in o.material_slots:
        m = s.material
        if m is None: continue
        b = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if m.node_tree else None
        img = b.inputs["Base Color"].links[0].from_node.image if b is not None and b.inputs["Base Color"].is_linked and getattr(b.inputs["Base Color"].links[0].from_node, "image", None) else None
        if img is not None and img.size[0] > 0:
            px = pixels(img); st = max(1, px.shape[0] // 24); c = px[::st, ::st, :3].reshape(-1, 3).mean(0)        # Blender holds a picture's pixels in linear light
            return [float(v) for v in np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.maximum(c, 1e-8), 1 / 2.4) - 0.055)]
        return [float(v) for v in m.diffuse_color[:3]]
    return [0.6, 0.6, 0.6]

def arrays(o, most=None):
    """An object's triangles as it stands in the world: corner positions and triangles. most: simplified to about that many
    faces first (the collider and the plain copy need the shape, not every face of it)."""
    dm = None
    if most and len(o.data.polygons) > most: dm = o.modifiers.new("lighter", "DECIMATE"); dm.ratio = most / len(o.data.polygons)
    dg = bpy.context.evaluated_depsgraph_get(); me = bpy.data.meshes.new_from_object(o.evaluated_get(dg)); me.transform(o.matrix_world); me.calc_loop_triangles()
    if dm: o.modifiers.remove(dm)
    co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co); tri = np.empty(len(me.loop_triangles) * 3, np.int32); me.loop_triangles.foreach_get("vertices", tri)
    bpy.data.meshes.remove(me); return co.reshape(-1, 3), tri.reshape(-1, 3)

def main(d, size=4096, samples=64):
    t0 = t = time.time(); sc = bpy.context.scene; gpu(); work = os.path.join(d, "blender"); room = list(sc["room"]) if sc.get("room") else None
    meshes = [o for o in sc.objects if o.type == "MESH" and len(o.data.polygons) and not o.hide_render]
    seen = [o for o in meshes if not o.get("collide_only")]
    if not seen: raise SystemExit("The scene has no shapes in it.")
    tops = []
    for o in sc.objects:                                                    # the things, for choosing where to stand
        if o.parent is None and not o.get("room_part") and o.type in ("MESH", "EMPTY"):
            fam = [o] + list(o.children_recursive); pts = [p.matrix_world @ Vector(c) for p in fam if p.type == "MESH" for c in p.bound_box]
            if pts: tops.append((Vector([min(p[i] for p in pts) for i in range(3)]), Vector([max(p[i] for p in pts) for i in range(3)])))
    solid = flat_copy(meshes, "solid"); CV, CT = arrays(solid, most=150000 if sc.get("pbr") else None); bpy.data.objects.remove(solid)
    # 1. what has no texture of its own: joined (the room's part and the rest apart), its colour and roughness baked to pictures
    glassy = [o for o in seen if o.get("glow") or any(sl.material and sl.material.get("glass") for sl in o.material_slots)]   # real glass and glowing bulbs: exported as they are, no lightmap
    seen = [o for o in seen if o not in glassy]
    keep = [o for o in seen if has_look(o)]; bare = [o for o in seen if o not in keep]; show = list(keep)
    for name, objs in (("built_room", [o for o in bare if o.get("room_part")]), ("built_things", [o for o in bare if not o.get("room_part")])):
        if not objs: continue
        j = flat_copy(objs, name)
        if name == "built_room": drop_unseen(j, room, float(sc.get("shell", 0.12))); j["room_part"] = True
        bake_look(j, size // 2); show.append(j)
    for o in bare + [o for o in meshes if o.get("collide_only")]: bpy.data.objects.remove(o)        # the joined meshes stand for them from here on
    for o in [o for o in sc.objects if o.type == "EMPTY" and not o.children]: bpy.data.objects.remove(o)
    print(f"{len(keep)} textured pieces kept, {len(bare)} plain ones joined and their look baked, in {time.time() - t:.0f} s", flush=True); t = time.time()
    # 2. the light
    dense = [o for o in show if len(o.data.polygons) >= DENSE and o.name not in ("built_room", "built_things")]; flat = [o for o in show if o not in dense]
    sets = [x for x in ([o for o in flat if o.get("room_part")], [o for o in flat if not o.get("room_part")]) if x]      # the room's large plain surfaces on a lightmap of their own, so the many small things do not crowd them out
    lms = bake_light(sets, dense, size, samples); print(f"light baked onto {len(flat)} pieces ({len(sets)} lightmaps) and into the corners of {len(dense)} dense ones in {time.time() - t:.0f} s", flush=True); t = time.time()
    lit = [(lm[..., :3] @ LUM)[lm[..., 3] > 0.5][::5] for lm in lms[:1]] + [(corner_light(o)[:, :3] @ LUM)[::5] for o in dense]; lum = np.concatenate(lit) if lit else np.zeros(1, np.float32); lum = lum[np.isfinite(lum)]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import blender_direct as DIR
    skyp = list(sc["sky"]) if sc.get("sky") else None
    hour = sc.get("hour") or DIR.hour_of(d) or ("day" if not skyp or skyp[3] >= 0.5 else "dusk" if skyp[3] >= 0.1 else "night")
    gain = DIR.exposure(lum, hour, float(sc.get("exposure_ev", 0.0))) if len(lum) else 1.0    # by day a mid-toned surface comes out a middle grey; at dusk and night the lamps' pools set it
    wb = DIR.white_balance([lm[..., :3][lm[..., 3] > 0.5] for lm in lms[:1]] + [corner_light(o)[:, :3] for o in dense])    # as a camera balanced to the room's own light: most of a lamp's colour cast taken out
    for k, lm in enumerate(lms): np.save(os.path.join(work, f"light{k}.npy"), encode(lm[::-1, :, :3] * wb, gain))   # Blender's pictures start at the bottom row
    hdr = bool(sc.get("pbr"))
    if hdr:                                                                  # the visuals-first worlds: the lightmaps in full range, as half-float EXR
        for k, lm in enumerate(lms): write_exr(os.path.join(work, f"light{k}.exr"), calm_hdr(np.nan_to_num(lm[:, :, :3]).clip(0, None) * wb * gain), flip=False)
    for o in dense:
        a = o.data.color_attributes["Col"]; v = corner_light(o); v[:, :3] = (np.nan_to_num(v[:, :3]).clip(0, None) * wb * (gain / LM_MAX)).clip(0, 1); v[:, 3] = 1; a.data.foreach_set("color", v.ravel())
    if os.environ.get("BLENDER_RELIEF") != "0":
        try: print(f"relief: {relief([o for o in keep if o.get('room_part')])} normal maps for the room's surfaces (after the bake: the lightmaps stay smooth)", flush=True)
        except Exception as e: print("relief left out:", e, flush=True)
    # 3. where to stand, and what is seen from there
    allV = np.concatenate([np.array([o.matrix_world @ Vector(c) for c in o.bound_box], np.float32) for o in show]); lo, hi = Vector(allV.min(0).tolist()), Vector(allV.max(0).tolist())
    eye, fwd = stand(tops, room, lo, hi)
    if sc.get("start"): eye = [float(v) for v in sc["start"]]; fwd = list((Vector((0, 0, 1.0)) - Vector(eye)).normalized()) if Vector(eye).length > 0.1 else [0.0, 1.0, -0.12]
    pano = panorama(eye, os.path.join(work, "pano.exr")); np.save(os.path.join(work, "env.npy"), encode(pano[::-1] * wb, gain)); print(f"panorama in {time.time() - t:.0f} s", flush=True); t = time.time()
    probes = []
    if hdr:                                                                  # reflection probes: HDR panoramas at points spread over the place, each with the box it corrects against
        for k, p in enumerate(probe_points(room, lo, hi, eye)):
            px = panorama(p, os.path.join(work, f"probe{k}_raw.exr"), samples=32, width=512)
            write_exr(os.path.join(work, f"probe{k}.exr"), (px * wb * gain), flip=False); probes.append({"at": list(p), "file": f"probe{k}.exr"})     # Blender's rows, bottom first: the file comes out upright
        if sc.get("sky_hdr") and os.path.exists(os.path.join(d, "pbr", "env_view.npy")):
            view = np.load(os.path.join(d, "pbr", "env_view.npy")).astype(np.float32) * wb * gain; write_exr(os.path.join(work, "sky.exr"), view, flip=True)    # row 0 the zenith
        print(f"{len(probes)} reflection probes and the HDR sky in {time.time() - t:.0f} s", flush=True); t = time.time()
    # the painted sky round the place (blender_direct.py), at the same exposure, for the viewer to show behind the windows
    sky_k = None
    if sc.get("sky_picture") and os.path.exists(sc["sky_picture"]):
        img = bpy.data.images.load(sc["sky_picture"]); img.colorspace_settings.name = "sRGB"; px = pixels(img)[::-1, :, :3] * float(sc.get("sky_strength", 1.0)) * float(sc.get("sky_seen", 1.0)) * wb * gain; bpy.data.images.remove(img)
        px *= min(1.0, 0.8 / max(float(np.median(px[: px.shape[0] // 2] @ LUM)), 1e-6))     # its sky's middle tone just under white: a view, not a glare
        sky_k = float(max(np.percentile(px @ LUM, 99.5), 1e-3)); x = (px / sky_k).clip(0, 1)
        np.save(os.path.join(work, "sky.npy"), (np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-8), 1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8))
    elif os.path.exists(os.path.join(work, "sky.npy")): os.remove(os.path.join(work, "sky.npy"))
    # 4. the plain copy (one colour a thing) and the world's own file
    PV, PT, PC = [], [], []; n = 0
    for o in show:
        V, T = arrays(o, most=12000 if sc.get("pbr") else None); PV.append(V); PT.append(T + n); n += len(V); PC.append(np.tile(np.array(plain_colour(o), np.float32), (len(T), 1)))
    np.savez(os.path.join(work, "geo.npz"), CV=CV, CT=CT, PV=np.concatenate(PV), PT=np.concatenate(PT), PC=np.concatenate(PC))
    for o in show:
        if o.parent is not None: w = o.matrix_world.copy(); o.parent = None; o.matrix_world = w
    only(show)
    bpy.ops.export_scene.gltf(filepath=os.path.join(d, "scene.glb"), export_format="GLB", use_selection=True, export_apply=True, export_yup=True, export_texcoords=True, export_normals=True,
                              export_vertex_color="ACTIVE", export_active_vertex_color_when_no_material=True, export_all_vertex_colors=False, export_image_format="JPEG", export_jpeg_quality=88,
                              export_extras=True, export_lights=False, export_cameras=False, export_animations=False)
    sky = list(sc["sky"]) if sc.get("sky") else None
    def tone(c, k):                                                          # a sky colour as the exposure shows it
        x = np.array(c[:3], np.float32) * k * gain; x = x * (1 + x / 16.0) / (1 + x)
        return [round(float(v), 3) for v in np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-8), 1 / 2.4) - 0.055).clip(0, 1)]
    meta = {"room": room, "setting": "indoor" if room else "outdoor", "lo": list(lo), "hi": list(hi), "eye": eye, "forward": fwd, "gain": gain, "light_max": LM_MAX, "lightmaps": len(lms),
            "backdrop": tone(sky, sky[3]) if sky else [0.02, 0.02, 0.03], "things": len(tops), "hour": hour, "sky_k": sky_k, "white_balance": [round(float(x), 3) for x in wb], "hdr": hdr, "probes": probes, "room_box": list(sc["room_box"]) if sc.get("room_box") else None, "fog": list(sc["fog"]) if sc.get("fog") else None, "faces": int(sum(len(t) for t in PT)), "pieces": len(show), "size": size, "samples": samples}
    json.dump(meta, open(os.path.join(work, "meta.json"), "w"), indent=1)
    print(f"exposure gain {gain:.3g}, {len(tops)} things, {meta['faces']} faces in {len(show)} pieces; written in {time.time() - t:.0f} s ({time.time() - t0:.0f} s in all)", flush=True)

if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:]; main(a[0], *(int(x) for x in a[1:3]))
