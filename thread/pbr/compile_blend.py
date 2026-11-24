"""The scene description -> a Blender scene, in headless Blender (RFC pillar 3; replaces the numpy room and level builders).
  blender -b --factory-startup --python compile_blend.py -- WORLD_DIR
Builds, from WORLD_DIR/pbr/scene_ir.json:
  - indoors, the room of any outline: walls with real thickness, openings cut through them (flat, round or pointed
    heads), each window with its reveal, frame, mullions, a pane of real glass and a Cycles light portal; the door with
    its leaf; floor, skirting, an optional wainscot; the ceiling flat, beamed, coffered, vaulted or pitched (gables filled);
    every solid edge bevelled (bevel_mm) with weighted normals, so edges catch the light instead of ending razor-sharp;
  - outdoors, the ground: a 40 m sheet shaped by layered noise, kept flat where the visitor starts and along a path to
    the middle, its three materials mixed by height, slope, the path and patches, then baked to one set of maps; water
    where there is a water level; the foliage scattered from the CC0 library (Poisson discs, kept off the path, varied
    in turn and size);
  - the things: hero 3D models and library models brought in once per kind and set at their places, fitted to their
    planned size; cloth draped by Blender's cloth simulation (curtains hang in folds, cloths settle over what is under
    them, rugs lie flat with a little give); the rest plain bevelled forms;
  - every material as a full PBR node set (colour, normal, roughness, metallic) on picture coordinates in metres;
  - lamps with colour by temperature and a soft size, a glowing bulb for each, the sun (or moon) as a sun lamp, the
    HDR sky as the world;
and saves WORLD_DIR/blender/scene.blend, ready for art direction (blender_direct.py) and the bake (blender_finish.py)."""
import sys, os, math, json, random
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix, Euler

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ir

TRIM_W = 0.07          # frame and casing width
HERO_FACES = int(os.environ.get("HERO_FACES", "60000"))   # most faces a model keeps
START = (0.0, -5.0)    # where the visitor starts outdoors (plan.py tells the model the same)
SC = None

# ---------------------------------------------------------------------------------------------------- helpers
def kelvin_rgb(t):
    import blender_direct as D
    return D.kelvin(t)

def link(o, col=None):
    (col or bpy.context.scene.collection).objects.link(o); return o

def mesh_obj(name, verts, faces, mat=None):
    me = bpy.data.meshes.new(name); me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces]); me.update()
    o = link(bpy.data.objects.new(name, me))
    if mat: o.data.materials.append(mat)
    return o

def from_bm(name, bm, mat=None):
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); o = link(bpy.data.objects.new(name, me))
    if mat: o.data.materials.append(mat)
    return o

def box(name, size, at, mat=None, rot=0.0):
    """A box `size` (x, y, z) whose underside's middle is at `at`, turned `rot` radians about z."""
    bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0); bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    bmesh.ops.translate(bm, vec=Vector((0, 0, size[2] / 2)), verts=bm.verts)
    o = from_bm(name, bm, mat); o.location = Vector(at); o.rotation_euler = (0, 0, rot); return o

def uv_metres(o, scale=1.0):
    """Picture coordinates in metres over `scale` (one tile = scale metres), laid by each face's main direction in the
    world, so every surface shows its material at true size and the glTF carries them."""
    me = o.data; bm = bmesh.new(); bm.from_mesh(me); uv = bm.loops.layers.uv.verify(); M = o.matrix_world; R = M.to_3x3()
    for f in bm.faces:
        n = (R @ f.normal); ax = max(range(3), key=lambda i: abs(n[i]))
        for l in f.loops:
            p = M @ l.vert.co
            u, v = ((p.y, p.z) if ax == 0 else (p.x, p.z) if ax == 1 else (p.x, p.y))
            if ax == 0 and n[0] < 0: u = -u
            if ax == 1 and n[1] > 0: u = -u
            l[uv].uv = (u / scale, v / scale)
    bm.to_mesh(me); bm.free()

def bevel(o, mm, angle=40):
    """Rounded edges (three segments) and normals weighted by face size: the small rims that catch light on real things."""
    if mm <= 0: return
    b = o.modifiers.new("bevel", "BEVEL"); b.width = mm / 1000.0; b.segments = 3; b.limit_method = "ANGLE"; b.angle_limit = math.radians(angle)
    try: b.harden_normals = True
    except Exception: pass
    w = o.modifiers.new("weighted", "WEIGHTED_NORMAL"); w.keep_sharp = True

def apply_mods(o):
    bpy.ops.object.select_all(action="DESELECT"); o.select_set(True); bpy.context.view_layer.objects.active = o
    for m in list(o.modifiers):
        try: bpy.ops.object.modifier_apply(modifier=m.name)
        except Exception: o.modifiers.remove(m)

def material(m, cache={}):
    """A PBRMaterial -> a Principled material: its maps on the first picture coordinates (colour sRGB, the rest
    non-colour, normal through a normal-map node), or its plain values where a map is missing; glass is real glass."""
    if m.name in cache: return cache[m.name]
    mt = bpy.data.materials.new(m.name)
    try: mt.use_nodes = True
    except Exception: pass
    nt = mt.node_tree; b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = (*[c ** 2.2 for c in m.rgb[:3]], 1.0); b.inputs["Roughness"].default_value = m.rough; b.inputs["Metallic"].default_value = m.metal
    if m.transmission > 0.5:
        for k in ("Transmission Weight", "Transmission"):
            if k in b.inputs: b.inputs[k].default_value = m.transmission
        b.inputs["IOR"].default_value = m.ior; b.inputs["Roughness"].default_value = 0.02; mt["glass"] = True
        try: mt.surface_render_method = "BLENDED"
        except Exception: pass
        cache[m.name] = mt; return mt
    def tex(path, space):
        if not path or not os.path.exists(path): return None
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(path, check_existing=True); t.image.colorspace_settings.name = space; return t
    a = tex(m.albedo, "sRGB")
    if a: nt.links.new(a.outputs["Color"], b.inputs["Base Color"])
    r = tex(m.roughness, "Non-Color")
    if r: nt.links.new(r.outputs["Color"], b.inputs["Roughness"])
    mm = tex(m.metallic, "Non-Color")
    if mm and m.metal > 0.05: nt.links.new(mm.outputs["Color"], b.inputs["Metallic"])
    n = tex(m.normal, "Non-Color")
    if n:
        nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 1.0; nt.links.new(n.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    mt["scale_m"] = m.scale_m; cache[m.name] = mt; return mt

def solid(o, room=True, mm=None):
    o["room_part"] = bool(room)
    if mm: bevel(o, mm)
    return o

# ---------------------------------------------------------------------------------------------------- indoors
def edges(fp):
    n = len(fp); return [(Vector((*fp[i], 0)), Vector((*fp[(i + 1) % n], 0))) for i in range(n)]

def opening_outline(op, segs=16):
    """The outline of an opening in the wall's own plane (u along, v up), its head flat, round or pointed. -> [(u, v)]"""
    w, h, s = op.width_m, op.height_m, op.sill_m; u0, u1 = op.u_centre - w / 2, op.u_centre + w / 2
    if op.arch == "round" and h > w / 2:
        r = w / 2; spring = s + h - r
        return [(u0, s), (u1, s), (u1, spring)] + [(op.u_centre + r * math.cos(math.pi * k / segs), spring + r * math.sin(math.pi * k / segs)) for k in range(1, segs)] + [(u0, spring)]
    if op.arch == "pointed" and h > w * 0.6:
        r = w * 0.8; cR, cL = u1 - r, u0 + r                                    # two arcs of radius r, each centred on the far side
        aR = math.acos(max(-1.0, min(1.0, (op.u_centre - cR) / r))); rise = r * math.sin(aR); spring = s + h - rise
        right = [(cR + r * math.cos(t), spring + r * math.sin(t)) for t in np.linspace(0, aR, segs // 2 + 1)[1:]]
        left = [(cL + r * math.cos(t), spring + r * math.sin(t)) for t in np.linspace(math.pi - aR, math.pi, segs // 2 + 1)[1:-1]]
        return [(u0, s), (u1, s), (u1, spring)] + right + left + [(u0, spring)]
    return [(u0, s), (u1, s), (u1, s + h), (u0, s + h)]

def prism(name, outline, a, b, depth, inset=0.0):
    """A solid through a wall: `outline` in the wall's (u, v) plane, from a to b along its edge, `depth` through it
    centred on the wall's middle line (thickness T outward from the inner face, so the middle is T/2 out)."""
    d = (b - a).normalized(); out = Vector((d.y, -d.x, 0)); bm = bmesh.new(); front, back = [], []
    for u, v in outline:
        p = a + d * u + Vector((0, 0, v))
        front.append(bm.verts.new(p - out * inset)); back.append(bm.verts.new(p + out * (depth - inset)))
    bm.faces.new(front); bm.faces.new(back[::-1])
    n = len(outline)
    for i in range(n): bm.faces.new((front[i], back[i], back[(i + 1) % n], front[(i + 1) % n]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces); return from_bm(name, bm)

def wall_slab(i, a, b, T, H, mat):
    """One wall: a slab from the inner face (the outline's edge) T outward, as high as the room; ends mitred to meet its neighbours."""
    d = (b - a).normalized(); out = Vector((d.y, -d.x, 0)); L = (b - a).length
    o = box(f"wall {i:02d}", (L + 2 * T, T, H), (0, 0, 0), mat); o.location = (a + b) / 2 + out * (T / 2); o.rotation_euler = (0, 0, math.atan2(d.y, d.x))
    bpy.context.view_layer.update(); return o

def cut(o, cutter):
    """A hole through `o` where `cutter` is. The boolean leaves an empty material slot and the picture coordinates as a
    plain attribute: both are cleared (coordinates are laid again later, in metres)."""
    keep = [m.name for m in o.modifiers]
    for n in keep: apply_one(o, n)
    m = o.modifiers.new("cut", "BOOLEAN"); m.operation = "DIFFERENCE"; m.object = cutter
    try: m.solver = "EXACT"
    except Exception: pass
    apply_one(o, "cut"); bpy.data.objects.remove(cutter)
    while o.data.uv_layers: o.data.uv_layers.remove(o.data.uv_layers[0])
    for a in [a for a in o.data.attributes if a.data_type == "FLOAT2" and not a.name.startswith(".")]: o.data.attributes.remove(a)
    for k in reversed(range(len(o.material_slots))):
        if o.material_slots[k].material is None:
            o.active_material_index = k; bpy.context.view_layer.objects.active = o; bpy.ops.object.material_slot_remove()
    for p in o.data.polygons: p.material_index = 0

def apply_one(o, name):
    bpy.ops.object.select_all(action="DESELECT"); o.select_set(True); bpy.context.view_layer.objects.active = o
    try: bpy.ops.object.modifier_apply(modifier=name)
    except Exception: o.modifiers.remove(o.modifiers[name])

def frame_around(name, outline, a, b, T, mat, width=TRIM_W, proud=0.025):
    """The casing round an opening, on the room side: the outline offset outward by `width`, a ring standing `proud` off the wall."""
    d = (b - a).normalized(); out = Vector((d.y, -d.x, 0)); ring = []
    cu = sum(u for u, _ in outline) / len(outline); cv = sum(v for _, v in outline) / len(outline)
    def grow(p, k):
        u, v = p; du, dv = u - cu, v - cv; n = math.hypot(du, dv) or 1.0
        return (u + du / n * k * 1.4, v + dv / n * k) if v > outline[0][1] + 1e-3 else (u + (1 if du > 0 else -1) * k, v)
    inner = list(outline); outer = [grow(p, width) for p in inner]; bm = bmesh.new()
    def V(p, o_): return bm.verts.new(a + d * p[0] + Vector((0, 0, p[1])) - out * o_)
    vi0 = [V(p, 0.0) for p in inner]; vo0 = [V(p, 0.0) for p in outer]; vi1 = [V(p, proud) for p in inner]; vo1 = [V(p, proud) for p in outer]
    n = len(inner); skip_bottom = outline[0][1] < 0.01                         # a door's casing has no bottom
    for i in range(n):
        j = (i + 1) % n
        if skip_bottom and i == 0: continue
        bm.faces.new((vi1[i], vi1[j], vo1[j], vo1[i])); bm.faces.new((vo0[i], vo0[j], vo1[j], vo1[i])); bm.faces.new((vi0[j], vi0[i], vi1[i], vi1[j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces); return from_bm(name, bm, mat)

def window(i, op, a, b, T, mats):
    d = (b - a).normalized(); out = Vector((d.y, -d.x, 0)); outline = opening_outline(op); made = []
    made.append(frame_around(f"window casing {i}", outline, a, b, T, mats["trim"]))
    # the frame in the middle of the wall's depth, with its mullions, and the glass
    mid = T * 0.55; cols, rows = op.mullions; w, h = op.width_m, op.height_m
    u0 = op.u_centre - w / 2; bars = []
    for k in range(1, cols): bars.append(("v", u0 + w * k / cols))
    for k in range(1, rows): bars.append(("h", op.sill_m + h * k / rows))
    for k, (kind, at) in enumerate(bars):
        if kind == "v": o = box(f"mullion {i}.{k}", (0.035, 0.05, h), (0, 0, 0), mats["trim"]); o.location = a + d * at + out * mid + Vector((0, 0, op.sill_m))
        else: o = box(f"transom {i}.{k}", (w, 0.05, 0.035), (0, 0, 0), mats["trim"]); o.location = a + d * op.u_centre + out * mid + Vector((0, 0, at - 0.0175))
        o.rotation_euler = (0, 0, math.atan2(d.y, d.x)); made.append(o)
    glass = prism(f"window glass {i}", outline, a, b, 0.008, inset=-mid); glass.data.materials.append(mats["glass"]); glass["glass"] = True; made.append(glass)
    sill = box(f"window sill {i}", (w + 0.12, 0.16, 0.035), (0, 0, 0), mats["trim"]); sill.location = a + d * op.u_centre - out * 0.04 + Vector((0, 0, op.sill_m - 0.035)); sill.rotation_euler = (0, 0, math.atan2(d.y, d.x)); made.append(sill)
    # a light portal over the opening, outside: Cycles sends its sky samples through the window
    L = bpy.data.lights.new(f"portal {i}", "AREA"); L.shape = "RECTANGLE"; L.size = w; L.size_y = h
    try: L.cycles.is_portal = True
    except Exception: pass
    po = link(bpy.data.objects.new(f"portal {i}", L)); po.location = a + d * op.u_centre + out * (T + 0.02) + Vector((0, 0, op.sill_m + h / 2))
    po.rotation_euler = (out * -1).to_track_quat("-Z", "Z").to_euler()
    for o in made:
        if o.type == "MESH": solid(o, True, 2.0)
    return made

def door(i, op, a, b, T, mats):
    d = (b - a).normalized(); out = Vector((d.y, -d.x, 0)); outline = opening_outline(op); made = [frame_around(f"door casing {i}", outline, a, b, T, mats["trim"], 0.08)]
    leaf = box(f"door leaf {i}", (op.width_m - 0.01, 0.045, op.height_m - 0.01), (0, 0, 0), mats["trim"]); leaf.location = a + d * op.u_centre + out * (T * 0.5); leaf.rotation_euler = (0, 0, math.atan2(d.y, d.x)); made.append(leaf)
    for k, (du, dv) in enumerate(((0.0, 0.25), (0.0, 0.62))):                    # two raised panels
        p = box(f"door panel {i}.{k}", (op.width_m * 0.66, 0.015, op.height_m * 0.3), (0, 0, 0), mats["trim"]); p.location = a + d * op.u_centre + out * (T * 0.5 - 0.03) + Vector((0, 0, op.height_m * dv))
        p.rotation_euler = (0, 0, math.atan2(d.y, d.x)); made.append(p)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.025, depth=0.09); hd = bpy.context.object; hd.name = f"door handle {i}"; hd.rotation_euler = (math.pi / 2, 0, math.atan2(d.y, d.x))
    hd.location = a + d * (op.u_centre + op.width_m / 2 - 0.09) + out * (T * 0.5 - 0.06) + Vector((0, 0, 1.0)); hd.data.materials.append(mats.get("brass") or mats["trim"]); made.append(hd)
    for o in made: solid(o, True, 2.0)
    return made

def ceiling(room, mats, lo, hi):
    """The ceiling and what it needs above the walls: gables for a vault or a pitch."""
    H = room.height_m; fp = room.footprint; made = []
    W, D = hi.x - lo.x, hi.y - lo.y; along_x = W >= D; span = D if along_x else W; rise = min(span * 0.35, 1.6)
    if room.ceiling in ("vaulted", "pitched"):
        segs = 24 if room.ceiling == "vaulted" else 2; bm = bmesh.new(); L = W if along_x else D; rows = []
        for k in range(segs + 1):
            t = k / segs; s = -span / 2 + span * t
            z = H + (rise * math.sin(math.pi * t) if room.ceiling == "vaulted" else rise * (1 - abs(2 * t - 1)))
            rows.append((s, z))
        vs = []
        for s, z in rows:
            if along_x: vs.append((bm.verts.new((lo.x, (lo.y + hi.y) / 2 + s, z)), bm.verts.new((hi.x, (lo.y + hi.y) / 2 + s, z))))
            else: vs.append((bm.verts.new(((lo.x + hi.x) / 2 + s, hi.y, z)), bm.verts.new(((lo.x + hi.x) / 2 + s, lo.y, z))))
        for k in range(segs): bm.faces.new((vs[k][0], vs[k][1], vs[k + 1][1], vs[k + 1][0]))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces); bmesh.ops.reverse_faces(bm, faces=bm.faces)
        v = from_bm("ceiling vault", bm, mats["ceiling"]); bmesh_solidify(v, 0.12); made.append(v)
        for end in (0, 1):                                                       # the gable at each end, from the wall's top to the curve
            bm = bmesh.new(); pts = []
            for s, z in rows:
                pts.append(((lo.x if end == 0 else hi.x, (lo.y + hi.y) / 2 + s, z)) if along_x else (((lo.x + hi.x) / 2 + s, lo.y if end == 0 else hi.y, z)))
            base = [(p[0], p[1], H) for p in pts]; vv = [bm.verts.new(p) for p in pts]; vb = [bm.verts.new(p) for p in base]
            for k in range(len(pts) - 1): bm.faces.new((vb[k], vb[k + 1], vv[k + 1], vv[k]))
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4); g = from_bm(f"gable {end}", bm, mats["wall"]); bmesh_solidify(g, 0.25); made.append(g)
        if room.ceiling == "pitched":                                            # rafters
            n = max(2, int((W if along_x else D) / 0.9))
            for k in range(n + 1):
                t = k / n; x = lo.x + (hi.x - lo.x) * t if along_x else (lo.x + hi.x) / 2; y = (lo.y + hi.y) / 2 if along_x else lo.y + (hi.y - lo.y) * t
                for side in (-1, 1):
                    ang = math.atan2(rise, span / 2); r = box(f"rafter {k}{side}", (0.12, math.hypot(rise, span / 2), 0.18), (0, 0, 0), mats["trim"])
                    r.location = Vector((x, y + side * span / 4, H + rise / 2 - 0.09)) if along_x else Vector((x + side * span / 4, y, H + rise / 2 - 0.09))
                    r.rotation_euler = (side * ang, 0, 0) if along_x else (0, -side * ang, math.pi / 2); made.append(r)
    else:
        bm = bmesh.new(); vs = [bm.verts.new((x, y, H)) for x, y in fp]; bm.faces.new(vs[::-1]); made.append(from_bm("ceiling", bm, mats["ceiling"]))
        if room.ceiling in ("beamed", "coffered"):
            n = max(2, int((W if not along_x else D) / 0.9)); dirs = [0] if room.ceiling == "beamed" else [0, 1]
            for dd in dirs:
                across_x = (dd == 0) == along_x; length = W if across_x else D; count = max(2, int((D if across_x else W) / (0.9 if room.ceiling == "beamed" else 1.2)))
                for k in range(1, count):
                    t = k / count; b_ = box(f"beam {dd}.{k}", (length if across_x else 0.16, 0.16 if across_x else length, 0.2), (0, 0, 0), mats["trim"])
                    b_.location = Vector(((lo.x + hi.x) / 2, lo.y + (hi.y - lo.y) * t, H - 0.2)) if across_x else Vector((lo.x + (hi.x - lo.x) * t, (lo.y + hi.y) / 2, H - 0.2)); made.append(b_)
    for o in made: solid(o, True, room.bevel_mm)
    bpy.context.scene["roof_rise"] = float(rise if room.ceiling in ("vaulted", "pitched") else 0.0)
    return made

def bmesh_solidify(o, t):
    m = o.modifiers.new("thick", "SOLIDIFY"); m.thickness = t; m.offset = 1.0

def build_room(s, mats):
    room = s.room; fp = room.footprint; H = room.height_m; T = room.thickness_m
    xs = [p[0] for p in fp]; ys = [p[1] for p in fp]; lo, hi = Vector((min(xs), min(ys), 0)), Vector((max(xs), max(ys), H))
    bm = bmesh.new(); vs = [bm.verts.new((x, y, 0)) for x, y in fp]; bm.faces.new(vs); fl = from_bm("floor", bm, mats["floor"]); solid(fl)
    walls = []; E = edges(fp)
    for i, (a, b) in enumerate(E):
        w = wall_slab(i, a, b, T, H, mats["wall"]); solid(w); walls.append(w)
    # openings
    for k, op in enumerate(room.openings):
        a, b = E[op.wall_index % len(E)]; c = prism(f"cutter {k}", opening_outline(op), a, b, T + 0.4, inset=0.2)
        cut(walls[op.wall_index % len(E)], c)
        (window if op.kind != "door" else door)(k, op, a, b, T, mats)
    # mitred corners: the slabs overlap by T at each end; a convex outline needs nothing more
    # skirting and wainscot
    for i, (a, b) in enumerate(E):
        d = (b - a).normalized(); out = Vector((d.y, -d.x, 0)); L = (b - a).length; rot = math.atan2(d.y, d.x)
        gaps = sorted([(op.u_centre - op.width_m / 2 - 0.08, op.u_centre + op.width_m / 2 + 0.08) for op in room.openings if op.wall_index % len(E) == i and (op.kind == "door" or op.sill_m < max(room.skirting_m, room.panelling_m) + 0.05)])
        runs, u = [], 0.0
        for g0, g1 in gaps: runs.append((u, max(u, g0))); u = max(u, g1)
        runs.append((u, L))
        for r0, r1 in runs:
            if r1 - r0 < 0.05: continue
            sk = box(f"skirting {i}", (r1 - r0, 0.022, room.skirting_m), (0, 0, 0), mats["trim"]); sk.location = a + d * ((r0 + r1) / 2) - out * 0.011; sk.rotation_euler = (0, 0, rot); solid(sk, True, 2.0)
            if room.panelling_m > 0.3:
                pn = box(f"wainscot {i}", (r1 - r0, 0.015, room.panelling_m), (0, 0, 0), mats["trim"]); pn.location = a + d * ((r0 + r1) / 2) - out * 0.0075; pn.rotation_euler = (0, 0, rot); solid(pn, True, 1.0)
                rail = box(f"dado rail {i}", (r1 - r0, 0.035, 0.045), (0, 0, 0), mats["trim"]); rail.location = a + d * ((r0 + r1) / 2) - out * 0.0175 + Vector((0, 0, room.panelling_m)); rail.rotation_euler = (0, 0, rot); solid(rail, True, 3.0)
                n = max(1, int((r1 - r0) / 0.6))
                for k in range(n):                                               # raised panels on the wainscot
                    u0 = r0 + (r1 - r0) * (k + 0.12) / n; u1 = r0 + (r1 - r0) * (k + 0.88) / n
                    pp = box(f"panel {i}.{k}", (u1 - u0, 0.012, room.panelling_m * 0.62), (0, 0, 0), mats["trim"]); pp.location = a + d * ((u0 + u1) / 2) - out * 0.021 + Vector((0, 0, room.panelling_m * 0.19)); pp.rotation_euler = (0, 0, rot); solid(pp, True, 2.5)
        cr = box(f"cornice {i}", (L, 0.06, 0.08), (0, 0, 0), mats["trim"]); cr.location = a + d * (L / 2) - out * 0.03 + Vector((0, 0, H - 0.08)); cr.rotation_euler = (0, 0, rot); solid(cr, True, 6.0)
    ceiling(room, mats, lo, hi)
    for w in walls: bevel(w, room.bevel_mm)
    sc = bpy.context.scene; sc["room"] = [float(hi.x - lo.x), float(hi.y - lo.y), float(H)]; sc["shell"] = float(T + 0.3); sc["room_box"] = [float(lo.x), float(lo.y), float(hi.x), float(hi.y)]
    return lo, hi

# ---------------------------------------------------------------------------------------------------- outdoors
def fbm(x, y, octaves=5, seed=0):
    rng = np.random.default_rng(seed); out = np.zeros_like(x); amp = 1.0; f = 1.0 / 14.0
    for _ in range(octaves):
        ph = rng.uniform(0, 100, 4); out += amp * (np.sin(x * f + ph[0]) * np.cos(y * f * 1.3 + ph[1]) + 0.5 * np.sin((x + y) * f * 0.7 + ph[2]) * np.cos((x - y) * f * 0.9 + ph[3]))
        amp *= 0.5; f *= 2.03
    return out / 1.9

def build_terrain(s, mats):
    t = s.terrain; S = t.size_m; N = 220; xs = np.linspace(-S / 2, S / 2, N); X, Y = np.meshgrid(xs, xs)
    start = np.array(START); h = fbm(X, Y) * t.relief_m
    # flat where the visitor starts and along a path to the middle (a gentle S); patches for the accent layer
    path_pts = np.array([[start[0] + math.sin(k / 10 * math.pi) * 1.5, start[1] + (0 - start[1]) * k / 10] for k in range(11)])
    dpath = np.min(np.hypot(X[..., None] - path_pts[:, 0], Y[..., None] - path_pts[:, 1]), -1)
    dstart = np.hypot(X - start[0], Y - start[1]); flat = np.clip(np.minimum(dpath / 3.0, dstart / t.clear_radius_m), 0, 1); flat = flat * flat * (3 - 2 * flat)
    h = h * flat                                                                  # level 0 where flat
    for a in s.assets:                                                            # flat under what stands on it
        x, y = a.position[:2]; r = max(a.size_m[0], a.size_m[1]) / 2 + 0.8; k = np.clip(np.hypot(X - x, Y - y) / r - 0.6, 0, 1); h = h * k
    bm = bmesh.new(); V = [[bm.verts.new((X[i, j], Y[i, j], h[i, j])) for j in range(N)] for i in range(N)]
    uvl = bm.loops.layers.uv.new("UVMap")
    for i in range(N - 1):
        for j in range(N - 1):
            f = bm.faces.new((V[i][j], V[i][j + 1], V[i + 1][j + 1], V[i + 1][j]))
            for l, (ii, jj) in zip(f.loops, ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))): l[uvl].uv = (jj / (N - 1), ii / (N - 1))
    g = from_bm("ground", bm); g["room_part"] = True
    for p in g.data.polygons: p.use_smooth = True
    # the three layers, mixed by a mask, baked to one map set for the glTF
    mask = np.zeros((N, N, 3), np.float32); mask[..., 0] = np.clip(1 - dpath / 1.2, 0, 1)
    slope = np.hypot(*np.gradient(h, xs[1] - xs[0])); mask[..., 1] = np.clip((fbm(X * 2.3, Y * 2.3, 3, 7) - 0.25) * 2.5, 0, 1) * (1 - mask[..., 0]) + np.clip(slope - 0.6, 0, 1)
    mi = bpy.data.images.new("ground mask", N, N, alpha=False, float_buffer=True); mi.colorspace_settings.name = "Non-Color"
    mi.pixels.foreach_set(np.concatenate([mask, np.ones((N, N, 1), np.float32)], -1).ravel()); mi.pack()
    mixed = mix_ground(s.materials, mi, S); g.data.materials.append(mixed)          # the description's materials (their maps), not Blender's
    bake_ground(g, S); bpy.context.scene["ground"] = float(S); bpy.context.scene["room"] = None
    if t.water_level_m is not None:
        w = box("water", (S, S, 0.02), (0, 0, t.water_level_m - 0.02), water_material()); w["room_part"] = True
    return h, X, Y, dpath

def mix_ground(mats, mask_img, S):
    """One material mixing ground, path and accent by the mask (red: the path, green: the patches), each at its own tile size."""
    mt = bpy.data.materials.new("ground mixed")
    try: mt.use_nodes = True
    except Exception: pass
    nt = mt.node_tree; b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"); uv = nt.nodes.new("ShaderNodeTexCoord")
    mk = nt.nodes.new("ShaderNodeTexImage"); mk.image = mask_img; mk.interpolation = "Cubic"; nt.links.new(uv.outputs["UV"], mk.inputs["Vector"])
    sep = nt.nodes.new("ShaderNodeSeparateColor"); nt.links.new(mk.outputs["Color"], sep.inputs["Color"])
    def layer(m, socket_kind):
        path = {"color": m.albedo, "rough": m.roughness, "normal": m.normal}[socket_kind]
        if not path or not os.path.exists(path):
            if socket_kind == "color": c = nt.nodes.new("ShaderNodeRGB"); c.outputs[0].default_value = (*[v ** 2.2 for v in m.rgb], 1); return c.outputs[0]
            if socket_kind == "rough": v = nt.nodes.new("ShaderNodeValue"); v.outputs[0].default_value = m.rough; return v.outputs[0]
            c = nt.nodes.new("ShaderNodeRGB"); c.outputs[0].default_value = (0.5, 0.5, 1, 1); return c.outputs[0]
        tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = bpy.data.images.load(path, check_existing=True); tx.image.colorspace_settings.name = "sRGB" if socket_kind == "color" else "Non-Color"
        mp = nt.nodes.new("ShaderNodeMapping"); k = S / max(m.scale_m, 0.1); mp.inputs["Scale"].default_value = (k, k, 1); nt.links.new(uv.outputs["UV"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], tx.inputs["Vector"])
        return tx.outputs["Color"]
    def mixed(kind):
        a, p, c = (layer(mats[n], kind) for n in ("ground", "path", "accent"))
        m1 = nt.nodes.new("ShaderNodeMix"); m1.data_type = "RGBA"; nt.links.new(sep.outputs[1], m1.inputs["Factor"]); nt.links.new(a, m1.inputs[6]); nt.links.new(c, m1.inputs[7])
        m2 = nt.nodes.new("ShaderNodeMix"); m2.data_type = "RGBA"; nt.links.new(sep.outputs[0], m2.inputs["Factor"]); nt.links.new(m1.outputs[2], m2.inputs[6]); nt.links.new(p, m2.inputs[7])
        return m2.outputs[2]
    nt.links.new(mixed("color"), b.inputs["Base Color"]); nt.links.new(mixed("rough"), b.inputs["Roughness"])
    nm = nt.nodes.new("ShaderNodeNormalMap"); nt.links.new(mixed("normal"), nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    return mt

def bake_ground(g, S, size=4096):
    """The mixed ground baked to three pictures on its coordinates (colour, roughness, normal), so the glTF and the
    viewer carry exactly what Cycles shows. About a centimetre a pixel over 40 m."""
    sc = bpy.context.scene; sc.render.engine = "CYCLES"; sc.cycles.samples = 4; made = {}
    for kind, space, bt in (("color", "sRGB", "DIFFUSE"), ("rough", "Non-Color", "ROUGHNESS"), ("normal", "Non-Color", "NORMAL")):
        img = bpy.data.images.new(f"ground_{kind}", size, size, alpha=False); img.colorspace_settings.name = space
        nt = g.data.materials[0].node_tree; n = nt.nodes.new("ShaderNodeTexImage"); n.image = img; nt.nodes.active = n
        bpy.ops.object.select_all(action="DESELECT"); g.select_set(True); bpy.context.view_layer.objects.active = g
        if bt == "DIFFUSE": bpy.ops.object.bake(type=bt, pass_filter={"COLOR"}, margin=4)
        else: bpy.ops.object.bake(type=bt, margin=4)
        nt.nodes.remove(n); img.pack(); made[kind] = img
    mt = bpy.data.materials.new("ground")
    try: mt.use_nodes = True
    except Exception: pass
    nt = mt.node_tree; b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    for kind, sock in (("color", "Base Color"), ("rough", "Roughness")):
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = made[kind]; nt.links.new(t.outputs["Color"], b.inputs[sock])
    t = nt.nodes.new("ShaderNodeTexImage"); t.image = made["normal"]; nm = nt.nodes.new("ShaderNodeNormalMap"); nt.links.new(t.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    g.data.materials.clear(); g.data.materials.append(mt)

def water_material():
    mt = bpy.data.materials.new("water")
    try: mt.use_nodes = True
    except Exception: pass
    b = next(n for n in mt.node_tree.nodes if n.type == "BSDF_PRINCIPLED"); b.inputs["Base Color"].default_value = (0.02, 0.05, 0.06, 1); b.inputs["Roughness"].default_value = 0.04
    for k in ("Transmission Weight", "Transmission"):
        if k in b.inputs: b.inputs[k].default_value = 0.6
    b.inputs["IOR"].default_value = 1.33; return mt

def scatter(s, h, X, Y, dpath, cache):
    """Each kind of foliage from the CC0 library, set by Poisson discs over the ground: off the path, out of the visitor's
    clear circle and away from the things, each one turned and sized a little differently, standing on the ground."""
    import kitbash
    rng = random.Random(5); S = s.terrain.size_m; N = X.shape[0]; placed = []
    def ground_z(x, y):
        j = int(round((x + S / 2) / S * (N - 1))); i = int(round((y + S / 2) / S * (N - 1))); return float(h[min(max(i, 0), N - 1), min(max(j, 0), N - 1)])
    def free(x, y, r):
        if math.hypot(x - START[0], y - START[1]) < s.terrain.clear_radius_m + r: return False
        i = int(round((y + S / 2) / S * (N - 1))); j = int(round((x + S / 2) / S * (N - 1)))
        if dpath[min(max(i, 0), N - 1), min(max(j, 0), N - 1)] < 1.4 + r: return False
        if any(math.hypot(x - a.position[0], y - a.position[1]) < r + max(a.size_m[:2]) / 2 + 0.3 for a in s.assets): return False
        return all(math.hypot(x - px, y - py) > r + pr for px, py, pr in placed)
    for f in s.terrain.foliage:
        m = kitbash.match(f["what"], None, least=1.0) or kitbash.match("tree" if f["height_m"] > 3 else "plant", None, least=0.5)
        if not m: continue
        src = import_kind(m["path"], f"foliage {f['what']}", cache)
        if src is None: continue
        r = max(0.4, f["height_m"] * (0.3 if f["height_m"] > 3 else 0.5)); n = 0
        for _ in range(f["count"] * 30):
            if n >= f["count"]: break
            big = f["height_m"] > 3; R = S / 2 - 1.0 if big else min(S / 2 - 1.0, 14.0)
            x, y = rng.uniform(-R, R), rng.uniform(-R, R)
            if not free(x, y, r): continue
            k = rng.uniform(0.8, 1.2); inst = instance(src, f"{f['what']} {n}", [x, y, ground_z(x, y)], rng.uniform(0, 360), None, height=f["height_m"] * k)
            placed.append((x, y, r)); n += 1
    print(f"scattered {len(placed)} plants of {len(s.terrain.foliage)} kinds", flush=True)

# ---------------------------------------------------------------------------------------------------- things
def import_kind(path, name, cache):
    """A model file brought in once, joined into one object at the origin (its underside's middle), hidden: copies are
    linked duplicates of it. -> the source object, or None"""
    if path in cache: return cache[path]
    if not path or not os.path.exists(path): return None
    before = set(bpy.data.objects.keys())
    try: bpy.ops.import_scene.gltf(filepath=path)
    except Exception as e: print("could not bring in", path, e, flush=True); cache[path] = None; return None
    new = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before]; meshes = [o for o in new if o.type == "MESH"]
    if not meshes: cache[path] = None; return None
    bpy.context.view_layer.update()
    for o in meshes:                                                              # bake in their place in the hierarchy, then join
        w = o.matrix_world.copy(); o.parent = None; o.matrix_world = w
    for o in new:
        if o.type != "MESH": bpy.data.objects.remove(o)
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes: o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1: bpy.ops.object.join()
    src = bpy.context.view_layer.objects.active; bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    pts = [Vector(v.co) for v in src.data.vertices]; lo = Vector([min(p[i] for p in pts) for i in range(3)]); hi = Vector([max(p[i] for p in pts) for i in range(3)])
    src.data.transform(Matrix.Translation(-Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z)))); src.location = (0, 0, 0)
    n = len(src.data.polygons)
    if n > HERO_FACES:                                                            # TRELLIS.2 gives ~200K faces: a third is plenty at room scale, and loads far faster
        d = src.modifiers.new("lighter", "DECIMATE"); d.ratio = HERO_FACES / n; apply_one(src, "lighter")
    src.name = name; src["extent"] = list(hi - lo); src.hide_render = True; src.hide_viewport = True; cache[path] = src
    return src

def instance(src, name, at, yaw_deg, size=None, height=None):
    """A linked copy of a source, fitted: to `size` (x, y, z) keeping its proportions (turned a quarter if that fits
    better), or to `height`."""
    o = src.copy(); o.name = name; o.hide_render = False; o.hide_viewport = False; link(o); e = Vector(src["extent"]); rot = 0.0
    if size:
        w, d, h = size
        if e.x > 0 and e.y > 0 and (abs(math.log((e.y / e.x) / max(d / max(w, 1e-3), 1e-3))) > abs(math.log((e.x / e.y) / max(d / max(w, 1e-3), 1e-3)))): rot = math.pi / 2; e = Vector((e.y, e.x, e.z))
        ks = [v for v in (w / max(e.x, 1e-3), d / max(e.y, 1e-3), h / max(e.z, 1e-3))]
        k = sorted(ks)[1] if h > 0.05 else min(ks)                               # the middle one: a chair keeps its height and its seat's width
        k = min(k, 1.15 * min(ks) if min(ks) > 0 else k)
        o.scale = (k, k, k)
    elif height: k = height / max(e.z, 1e-3); o.scale = (k, k, k)
    o.location = Vector(at); o.rotation_euler = (0, 0, math.radians(yaw_deg) + rot); return o

def plain(a, mat):
    """A bevelled form for a thing nothing better was found for: a drum for round things, a box for the rest."""
    w, d, h = a.size_m; round_ = any(k in a.name.lower() for k in ("round", "barrel", "drum", "pot", "vase", "column", "pillar", "bin", "stool", "lamp"))
    if round_:
        bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=0.5, depth=1.0, location=(0, 0, 0.5)); o = bpy.context.object
        bpy.ops.object.transform_apply(location=True); o.scale = (w, d, h); bpy.ops.object.transform_apply(scale=True)
    else: o = box(a.name, (w, d, h), (0, 0, 0))
    o.name = a.name; o.data.materials.append(mat); bevel(o, 6.0); o.location = Vector(a.position); o.rotation_euler = (0, 0, math.radians(a.yaw_deg)); return o

def cloth(a, mat, colliders):
    """Draped cloth. A curtain: a pleated sheet pinned along its top, settled by the cloth simulation. A rug: a flat sheet
    with thickness. Anything else (a tablecloth, a throw): a sheet let fall onto what is under it."""
    w, d, h = a.size_m; low = a.name.lower(); x, y, z = a.position; yaw = math.radians(a.yaw_deg)
    if any(k in low for k in ("rug", "carpet", "mat")):
        o = box(a.name, (w, d, 0.012), (x, y, z + 0.001), mat, yaw); bevel(o, 5.0); return o
    curtain = any(k in low for k in ("curtain", "drape", "banner", "tapestry", "flag"))
    nx, ny = (max(8, int(w / 0.03)), max(8, int(h / 0.05))) if curtain else (40, 40)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=nx, y_subdivisions=ny, size=1.0); o = bpy.context.object; o.name = a.name
    if curtain:
        o.scale = (w * 1.35, h, 1); bpy.ops.object.transform_apply(scale=True)
        for v in o.data.vertices: v.co = Vector((v.co.x, math.sin(v.co.x / 0.06 * math.pi) * 0.035, v.co.y + h / 2))   # pleats, upright, its bottom at 0
        grp = o.vertex_groups.new(name="pin"); grp.add([v.index for v in o.data.vertices if v.co.z > h - 0.02], 1.0, "REPLACE")
        o.location = (x, y, z); o.rotation_euler = (0, 0, yaw)
    else:
        top = z + h; o.scale = (w * 1.3, d * 1.3, 1); bpy.ops.object.transform_apply(scale=True); o.location = (x, y, top + 0.05); o.rotation_euler = (0, 0, yaw)
    o.data.materials.append(mat)
    c = o.modifiers.new("cloth", "CLOTH"); c.settings.quality = 6; c.settings.mass = 0.25; c.collision_settings.use_self_collision = False; c.collision_settings.distance_min = 0.004
    if curtain: c.settings.vertex_group_mass = "pin"
    for k in colliders:
        if k.type == "MESH" and k.name != o.name and (k.location - o.location).length < 3.0 and "collision" not in [m.type.lower() for m in k.modifiers]: k.modifiers.new("collision", "COLLISION")
    sc = bpy.context.scene; sc.frame_start = 1; sc.frame_end = 40
    for f in range(1, 41): sc.frame_set(f)
    bpy.ops.object.select_all(action="DESELECT"); o.select_set(True); bpy.context.view_layer.objects.active = o
    try: bpy.ops.object.modifier_apply(modifier="cloth")
    except Exception as e: print("cloth kept unsettled:", e, flush=True)
    sz = o.modifiers.new("thick", "SOLIDIFY"); sz.thickness = 0.004; apply_mods(o)
    for p in o.data.polygons: p.use_smooth = True
    sc.frame_set(1); return o

def ground_things(s, objs):
    """Each thing resting on another (or on the ground outdoors) lowered or raised onto what is under its middle."""
    sc = bpy.context.scene; dg = bpy.context.evaluated_depsgraph_get()
    for a, o in objs:
        if a.on_surface not in ("on", "floor") or o is None: continue
        bpy.context.view_layer.update(); bb = [o.matrix_world @ Vector(c) for c in o.bound_box]; lo = min(p.z for p in bb)
        mid = Vector((o.location.x, o.location.y, lo + 0.02)); o.hide_set(True) if hasattr(o, "hide_set") else None
        hit, loc, *_ = sc.ray_cast(dg, mid + Vector((0, 0, 0.3)), Vector((0, 0, -1)))
        o.hide_set(False) if hasattr(o, "hide_set") else None
        if hit and abs(loc.z - lo) < 0.6: o.location.z += loc.z - lo

def lights(s):
    for L in s.lights:
        kind = {"point": "POINT", "spot": "SPOT", "area": "AREA"}.get(L.kind, "POINT"); d = bpy.data.lights.new(L.id, kind); d.energy = float(L.power_w)
        try: d.use_temperature = True; d.temperature = float(L.kelvin)
        except Exception: d.color = kelvin_rgb(L.kelvin)
        if kind == "AREA": d.size = L.radius_m * 2
        else: d.shadow_soft_size = L.radius_m
        o = link(bpy.data.objects.new(L.id, d)); o.location = Vector(L.position); o["of"] = L.of
        bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=0.025)                     # the bulb, glowing
        bulb = from_bm(L.id + " bulb", bm); bulb.location = o.location; em = bpy.data.materials.new(L.id + " glow")
        try: em.use_nodes = True
        except Exception: pass
        b = next(n for n in em.node_tree.nodes if n.type == "BSDF_PRINCIPLED"); col = kelvin_rgb(L.kelvin)
        b.inputs["Emission Color"].default_value = (*col, 1); b.inputs["Emission Strength"].default_value = 20.0; b.inputs["Base Color"].default_value = (*col, 1)
        bulb.data.materials.append(em); bulb["glow"] = True; bulb.visible_shadow = False
    a = s.atmosphere; sd = Vector(a.sun_dir).normalized()
    if a.sun_w > 0 and sd.z > -0.05:
        d = bpy.data.lights.new("sun", "SUN"); d.energy = a.sun_w; d.angle = math.radians(0.53 if a.time_of_day != "night" else 0.5)
        k = {"day": 5800, "golden_hour": 3400, "dusk": 2600, "night": 7800}.get(a.time_of_day, 5800)
        try: d.use_temperature = True; d.temperature = k
        except Exception: d.color = kelvin_rgb(k)
        o = link(bpy.data.objects.new("sun", d)); o.rotation_euler = (-sd).to_track_quat("-Z", "Y").to_euler(); o.location = sd * 30

def world(s, d):
    p = os.path.join(d, "pbr", "env.npy"); sc = bpy.context.scene; w = bpy.data.worlds.new("World"); sc.world = w
    try: w.use_nodes = True
    except Exception: pass
    nt = w.node_tree; bg = next(n for n in nt.nodes if n.type == "BACKGROUND"); out = next(n for n in nt.nodes if n.type == "OUTPUT_WORLD")
    if os.path.exists(p):
        a = np.load(p).astype(np.float32); H, W = a.shape[:2]; img = bpy.data.images.new("env", W, H, alpha=False, float_buffer=True)
        img.pixels.foreach_set(np.concatenate([a[::-1], np.ones((H, W, 1), np.float32)], -1).ravel())          # Blender's rows start at the bottom
        exr = os.path.join(d, "pbr", "env.exr"); img.filepath_raw = exr; img.file_format = "OPEN_EXR"; img.save(); img.pack()
        env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = img; nt.links.new(env.outputs["Color"], bg.inputs["Color"]); bg.inputs["Strength"].default_value = 1.0
        sc["sky_hdr"] = exr
    else: bg.inputs["Color"].default_value = (0.4, 0.55, 0.8, 1); bg.inputs["Strength"].default_value = 0.3
    sc["hour"] = s.atmosphere.time_of_day; sc["exposure_ev"] = float(s.atmosphere.exposure_ev)
    sc["fog"] = [float(s.atmosphere.fog_density), float(s.atmosphere.fog_height_m), *[float(c) for c in s.atmosphere.fog_color]]
    sc["sky"] = [0.4, 0.55, 0.8, 1.0]

def main(d):
    for o in list(bpy.data.objects): bpy.data.objects.remove(o)
    s = ir.load(os.path.join(d, "pbr", "scene_ir.json")); random.seed(3)
    mats = {k: material(m) for k, m in s.materials.items()}
    if "trim" not in mats: mats["trim"] = material(ir.PBRMaterial(name="trim", prompt="", rgb=[0.85, 0.83, 0.8], rough=0.45))
    if "glass" not in mats: mats["glass"] = material(ir.PBRMaterial(name="glass", prompt="", transmission=0.95, ior=1.52))
    mats["brass"] = material(ir.PBRMaterial(name="brass", prompt="", rgb=[0.85, 0.66, 0.35], rough=0.28, metal=1.0))
    if s.room: build_room(s, mats); terrain = None
    else: terrain = build_terrain(s, mats)
    cache = {}; objs = []; cloths = []
    for a in s.assets:
        o = None
        if a.source in ("hero_3d", "kitbash", "foliage_card") and a.mesh_path:
            src = import_kind(a.mesh_path, f"src {a.kind}", cache)
            if src is not None: o = instance(src, a.name, a.position, a.yaw_deg, a.size_m)
        if o is None and a.source == "cloth_sim": cloths.append(a); continue
        if o is None:
            m = s.materials.get(a.material) or s.materials.get(a.kind); o = plain(a, material(m) if m else mats["trim"])
        o["thing"] = True; o["kind"] = a.kind; objs.append((a, o))
    ground_things(s, objs)
    colliders = [o for _, o in objs] + [o for o in bpy.data.objects if o.get("room_part")]
    for a in cloths:
        m = s.materials.get(a.material) or s.materials.get(a.kind); o = cloth(a, material(m) if m else mats["trim"], colliders); o["thing"] = True; objs.append((a, o))
    for o in bpy.data.objects:
        if o.type == "MESH" and "collision" in [m.type.lower() for m in o.modifiers]:
            for m in [m for m in o.modifiers if m.type == "COLLISION"]: o.modifiers.remove(m)
    if terrain is not None: scatter(s, *terrain, cache)
    for o in list(bpy.data.objects):
        if o.type == "MESH" and o.get("room_part") and o.data.materials and o.data.materials[0].get("scale_m"): uv_metres(o, float(o.data.materials[0]["scale_m"]))
        elif o.type == "MESH" and o.get("thing") and not o.data.uv_layers and o.data.materials and o.data.materials[0].get("scale_m"): uv_metres(o, float(o.data.materials[0]["scale_m"]))
    for o in list(bpy.data.objects):
        if o.type == "MESH" and o.modifiers and o.data.users == 1: apply_mods(o)
    # the hidden source of each kind stays (its copies share its mesh); the bake and export only take what renders
    lights(s); world(s, d)
    sc = bpy.context.scene; sc["setting"] = s.setting; sc["pbr"] = True
    if s.setting == "outdoor": sc["start"] = [START[0], START[1], 1.6]                 # outdoors the visitor starts where the plan kept clear for them
    os.makedirs(os.path.join(d, "blender"), exist_ok=True); bpy.ops.wm.save_as_mainfile(filepath=os.path.join(d, "blender", "scene.blend"))
    print(f"scene compiled: {sum(1 for o in bpy.data.objects if o.type == 'MESH')} meshes, {sum(1 for o in bpy.data.objects if o.type == 'LIGHT')} lights", flush=True)

if __name__ == "__main__":
    main(os.path.abspath(sys.argv[sys.argv.index("--") + 1]))
