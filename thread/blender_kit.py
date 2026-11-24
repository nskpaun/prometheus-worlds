"""Helpers the world model may call from the code it sends to Blender (`from kit import *`): blender_host.py loads this
file as the module `kit`. They do the things a language model most often gets wrong in Blender's own API (materials, mesh
making without a window, a room with openings), and all of them share one convention:
  metres, z up, the floor at z = 0, and `at` is where a thing STANDS: the middle of its underside.
Anything else is still plain bpy / bmesh / mathutils, which the model may use freely beside these.
check() lists what looks wrong in the scene (things in the air, under the floor, outside the room, inside each other);
blender_world.py shows it to the model after every step."""
import math
import bpy, bmesh
from mathutils import Matrix, Vector

__all__ = ["mat", "box", "cyl", "cone", "sphere", "lathe", "extrude", "tube", "room", "ground", "sky", "lamp", "group", "place", "delete", "check", "things"]

def _rgb(c):
    if isinstance(c, str):
        c = c.lstrip("#"); c = [int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]        # a hex colour is sRGB; Blender wants linear
    return (float(c[0]), float(c[1]), float(c[2]), 1.0)

def mat(name, color="#808080", roughness=0.6, metallic=0.0, emission=0.0, glass=False):
    """A material, made once per name. color: "#rrggbb" or (r, g, b) linear. emission: strength of its own glow (a lamp
    shade or screen: 2 to 20). glass=True: clear glass."""
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name)
        try: m.use_nodes = True
        except Exception: pass
    b = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if b is None: return m
    def put(key, v):
        if key in b.inputs: b.inputs[key].default_value = v
    put("Base Color", _rgb(color)); put("Roughness", float(roughness)); put("Metallic", float(metallic))
    if emission: put("Emission Color", _rgb(color)); put("Emission Strength", float(emission))
    if glass: put("Transmission Weight", 1.0); put("Roughness", 0.02); put("IOR", 1.45)
    m.diffuse_color = _rgb(color); m["glass"] = bool(glass)
    return m

def _mat(m):
    return m if not isinstance(m, str) else bpy.data.materials.get(m) or mat(m)

def _rot(rot):
    if isinstance(rot, (int, float)): return (0.0, 0.0, math.radians(rot))
    return tuple(math.radians(float(a)) for a in rot)

def _unique(name):
    if name not in bpy.data.objects: return name
    k = 2
    while f"{name}_{k}" in bpy.data.objects: k += 1
    return f"{name}_{k}"

def _finish(name, bm, at, rot, material, smooth=False):
    me = bpy.data.meshes.new(name); bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:]); bm.to_mesh(me); bm.free()
    if smooth:
        for p in me.polygons: p.use_smooth = True
        try: me.set_sharp_from_angle(angle=math.radians(40))
        except Exception: pass
    if material is not None: me.materials.append(_mat(material))
    o = bpy.data.objects.new(_unique(name), me); bpy.context.scene.collection.objects.link(o)
    o.location = Vector(at); o.rotation_euler = _rot(rot)
    return o

def box(name, size, at=(0, 0, 0), material=None, rot=0, bevel=0.0):
    """A box size = (x, y, z) standing at `at`. rot: degrees about z, or (x, y, z) degrees. bevel: rounded edges, in metres."""
    sx, sy, sz = (max(float(v), 1e-4) for v in size); bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.transform(bm, matrix=Matrix.Translation((0, 0, sz / 2)) @ Matrix.Diagonal((sx, sy, sz, 1)), verts=bm.verts[:])
    b = min(float(bevel), 0.45 * min(sx, sy, sz))
    if b > 0: bmesh.ops.bevel(bm, geom=bm.edges[:], offset=b, segments=3, affect="EDGES", profile=0.5)
    return _finish(name, bm, at, rot, material, smooth=b > 0)

def cone(name, radius_bottom, radius_top, height, at=(0, 0, 0), material=None, rot=0, segments=32):
    """A cone or tapered cylinder standing at `at` (radius_top 0 makes a point)."""
    h = max(float(height), 1e-4); bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=int(segments), radius1=max(float(radius_bottom), 0), radius2=max(float(radius_top), 0), depth=h)
    bmesh.ops.translate(bm, vec=(0, 0, h / 2), verts=bm.verts[:])
    return _finish(name, bm, at, rot, material, smooth=True)

def cyl(name, radius, height, at=(0, 0, 0), material=None, rot=0, segments=32):
    """An upright cylinder standing at `at`. To lay it down give rot=(90, 0, 0) or (0, 90, 0)."""
    return cone(name, radius, radius, height, at, material, rot, segments)

def sphere(name, radius, at=(0, 0, 0), material=None, scale=(1, 1, 1)):
    """A ball resting on `at` (its lowest point is there). scale squashes it into an ellipsoid."""
    r = max(float(radius), 1e-4); bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=r)
    bmesh.ops.transform(bm, matrix=Matrix.Translation((0, 0, r * scale[2])) @ Matrix.Diagonal((scale[0], scale[1], scale[2], 1)), verts=bm.verts[:])
    return _finish(name, bm, at, 0, material, smooth=True)

def lathe(name, profile, at=(0, 0, 0), material=None, rot=0, segments=32):
    """A turned shape (vase, bottle, bowl, lamp base, column): profile is a list of (radius, z) points from bottom to top,
    spun about the upright axis."""
    bm = bmesh.new(); vs = [bm.verts.new((max(float(r), 0.0), 0.0, float(z))) for r, z in profile]
    es = [bm.edges.new((a, b)) for a, b in zip(vs, vs[1:])]
    bmesh.ops.spin(bm, geom=vs + es, cent=(0, 0, 0), axis=(0, 0, 1), angle=2 * math.pi, steps=int(segments), use_duplicate=False)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-5)
    bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
    return _finish(name, bm, at, rot, material, smooth=True)

def extrude(name, outline, height, at=(0, 0, 0), material=None, rot=0):
    """A flat outline, a list of (x, y) points going round once, raised to `height`: any prism (an L-shaped counter, a
    star, a gable when laid on its side)."""
    bm = bmesh.new(); f = bm.faces.new([bm.verts.new((float(x), float(y), 0.0)) for x, y in outline])
    top = bmesh.ops.extrude_face_region(bm, geom=[f])
    bmesh.ops.translate(bm, vec=(0, 0, max(float(height), 1e-4)), verts=[g for g in top["geom"] if isinstance(g, bmesh.types.BMVert)])
    return _finish(name, bm, at, rot, material)

def tube(name, points, radius, material=None):
    """A round bar or pipe through a list of (x, y, z) points in the world (a lamp arm, a rail, a chair leg at a slant)."""
    cu = bpy.data.curves.new(name, "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = float(radius); cu.bevel_resolution = 4; cu.use_fill_caps = True
    sp = cu.splines.new("POLY"); sp.points.add(len(points) - 1)
    for p, q in zip(sp.points, points): p.co = (float(q[0]), float(q[1]), float(q[2]), 1.0)
    tmp = bpy.data.objects.new(name + "_curve", cu); bpy.context.scene.collection.objects.link(tmp)
    me = bpy.data.meshes.new_from_object(tmp.evaluated_get(bpy.context.evaluated_depsgraph_get())); bpy.data.objects.remove(tmp); bpy.data.curves.remove(cu)
    for p in me.polygons: p.use_smooth = True
    if material is not None: me.materials.append(_mat(material))
    o = bpy.data.objects.new(_unique(name), me); bpy.context.scene.collection.objects.link(o); return o

WALLS = {"north": ((0, 1), (1, 0)), "south": ((0, -1), (1, 0)), "east": ((1, 0), (0, 1)), "west": ((-1, 0), (0, 1))}      # outward normal, direction along the wall

def room(*a, **k):
    if a and isinstance(a[0], str): a = a[1:]              # a name given first, as every other helper takes one
    return _room(*a, **k)

def _room(width, depth, height, wall="wall", floor="floor", ceiling="ceiling", doors=(), windows=(), thickness=0.12):
    """The room: a floor width (x) by depth (y) centred on the origin, four walls and a ceiling, with openings.
    doors:   [(wall, offset, width, height)]        wall is "north" (+y), "south" (-y), "east" (+x) or "west" (-x);
    windows: [(wall, offset, sill, width, height)]  offset is the opening's middle, in metres along the wall from its middle.
    A door is closed with a door leaf; a window is left open to the sky (with an unseen pane that stops walking through).
    wall, floor, ceiling: materials or their names."""
    W, D, H, T = float(width), float(depth), float(height), float(thickness); sc = bpy.context.scene
    sc["room"] = [W, D, H]; made = []
    get = lambda m: m if not isinstance(m, str) else bpy.data.materials.get(m) or mat(m, {"floor": "#8b6f4e", "ceiling": "#f2f0ea"}.get(m, "#d9d4c7"))
    made.append(box("floor", (W + 2 * T, D + 2 * T, 0.1), (0, 0, -0.1), get(floor))); made.append(box("ceiling", (W + 2 * T, D + 2 * T, 0.1), (0, 0, H), get(ceiling)))
    for side, (n, t) in WALLS.items():
        L = W if t[0] else D; half = (D if t[0] else W) / 2 + T / 2; holes = []
        for d in doors:
            if d[0] == side: holes.append((float(d[1]) - float(d[2]) / 2, float(d[1]) + float(d[2]) / 2, 0.0, min(float(d[3]), H - 0.05), "door"))
        for w in windows:
            if len(w) == 4: w = (w[0], w[1], 0.9, w[2], w[3])       # no sill given: a usual one
            if w[0] == side: holes.append((float(w[1]) - float(w[3]) / 2, float(w[1]) + float(w[3]) / 2, float(w[2]), min(float(w[2]) + float(w[4]), H - 0.05), "window"))
        holes = [(max(a, -L / 2 + 0.05), min(b, L / 2 - 0.05), z0, z1, k) for a, b, z0, z1, k in holes]; holes = [h for h in holes if h[1] - h[0] > 0.1 and h[3] - h[2] > 0.1]
        xs = sorted({-L / 2 - T, L / 2 + T, *[h[0] for h in holes], *[h[1] for h in holes]}); zs = sorted({0.0, H, *[h[2] for h in holes], *[h[3] for h in holes]})
        def put(name, a, b, z0, z1, m, thick=T, out=0.0):
            mid = (a + b) / 2; c = (n[0] * (half + out) + t[0] * mid, n[1] * (half + out) + t[1] * mid, z0)
            size = (b - a, thick, z1 - z0) if t[0] else (thick, b - a, z1 - z0)
            o = box(name, size, c, m); made.append(o); return o
        for a, b in zip(xs, xs[1:]):
            for z0, z1 in zip(zs, zs[1:]):
                if any(h[0] <= (a + b) / 2 <= h[1] and h[2] <= (z0 + z1) / 2 <= h[3] for h in holes): continue
                put(f"wall_{side}", a, b, z0, z1, get(wall))
        for a, b, z0, z1, kind in holes:
            if kind == "door":
                put(f"door_{side}", a, b, z0, z1, mat("door", "#6b4a2f", 0.5), thick=0.05)
                k = cyl(f"door_{side}_knob", 0.03, 0.05, (0, 0, 0), mat("brass", "#b08d57", 0.3, 1.0), rot=(90, 0, 0) if t[0] else (0, 90, 0))
                k.location = (n[0] * (half - 0.06) + t[0] * (b - 0.12), n[1] * (half - 0.06) + t[1] * (b - 0.12), 1.0); made.append(k)
            else:
                f = mat("window_frame", "#f4f2ec", 0.5); w = 0.05
                for fa, fb, fz0, fz1 in ((a, b, z0, z0 + w), (a, b, z1 - w, z1), (a, a + w, z0, z1), (b - w, b, z0, z1), ((a + b) / 2 - w / 2, (a + b) / 2 + w / 2, z0, z1)):
                    put(f"window_{side}_frame", fa, fb, fz0, fz1, f, thick=T * 0.6)
                put(f"window_{side}_pane", a, b, z0, z1, None, thick=0.02)["collide_only"] = True
    for o in made: o["room_part"] = True
    return made

def ground(size=30.0, material="ground", *more):
    """Outdoors instead of room(): flat ground `size` metres across, its top at z = 0."""
    if isinstance(size, str): size, material = (material if isinstance(material, (int, float)) else 30.0), (more[0] if more else "ground")     # a name given first
    m = material if not isinstance(material, str) else bpy.data.materials.get(material) or mat(material, "#5f7a45", 0.9)
    bpy.context.scene["ground"] = float(size); o = box("ground", (size, size, 0.2), (0, 0, -0.2), m); o["room_part"] = True; return o

def sky(color="#9cc4ee", strength=1.0, sun=3.0, sun_from=(-1, -1.2, 2.0), sun_color="#fff1dc"):
    """Daylight: the colour and strength of the sky, and a sun of strength `sun` (0 for none, 2 to 5 for a clear day)
    shining from the direction `sun_from`. Indoors it comes in by the windows. Night: a dark colour and sun=0."""
    w = bpy.context.scene.world or bpy.data.worlds.new("World"); bpy.context.scene.world = w
    try: w.use_nodes = True
    except Exception: pass
    bg = next((n for n in w.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg: bg.inputs["Color"].default_value = _rgb(color); bg.inputs["Strength"].default_value = float(strength)
    bpy.context.scene["sky"] = list(_rgb(color)[:3]) + [float(strength)]
    old = bpy.data.objects.get("sun")
    if old: bpy.data.objects.remove(old)
    if sun > 0:
        d = bpy.data.lights.new("sun", "SUN"); d.energy = float(sun); d.color = _rgb(sun_color)[:3]; d.angle = math.radians(2)
        o = bpy.data.objects.new("sun", d); bpy.context.scene.collection.objects.link(o)
        o.rotation_euler = Vector(sun_from).normalized().to_track_quat("Z", "Y").to_euler(); o.location = Vector(sun_from).normalized() * 20

def lamp(name, at, watts=60.0, color="#ffe2b8", kind="point", size=0.1, aim=(0, 0, -1)):
    """A light at the point `at` (not a standing point: the place the light itself is). kind: "point", "spot" or "area"
    (area and spot shine along `aim`). A room lamp is 40 to 150 watts; make its shade or bulb from shapes with emission."""
    d = bpy.data.lights.new(name, {"point": "POINT", "spot": "SPOT", "area": "AREA"}.get(str(kind).lower(), "POINT"))
    d.energy = float(watts); d.color = _rgb(color)[:3]
    if d.type == "AREA": d.size = float(size)
    else: d.shadow_soft_size = float(size)
    o = bpy.data.objects.new(_unique(name), d); bpy.context.scene.collection.objects.link(o); o.location = Vector(at)
    if d.type != "POINT": o.rotation_euler = Vector(aim).normalized().to_track_quat("-Z", "Y").to_euler()
    return o

def _obj(x):
    o = x if not isinstance(x, str) else bpy.data.objects.get(x)
    if o is None: raise ValueError(f"there is no object named {x!r}: things() lists the names")
    return o

def _corners(o):
    bpy.context.view_layer.update()
    return [o.matrix_world @ Vector(c) for c in o.bound_box] if o.type == "MESH" else [o.matrix_world.translation.copy()]

def _bounds(objs):
    pts = [p for o in objs for p in _corners(o)]
    return Vector([min(p[i] for p in pts) for i in range(3)]), Vector([max(p[i] for p in pts) for i in range(3)])

def _family(o): return [o] + [c for k in o.children for c in _family(k)]

def group(name, parts):
    """Several shapes as ONE thing (a chair from its seat, back and legs), so it can be placed and turned whole and is
    checked as one. parts: the objects or their names. Returns the thing; its standing point is the middle of its underside."""
    parts = [_obj(p) for p in parts]; lo, hi = _bounds(parts)
    e = bpy.data.objects.new(_unique(name), None); bpy.context.scene.collection.objects.link(e); e.empty_display_size = 0.1
    e.location = ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z); bpy.context.view_layer.update()
    for p in parts:
        w = p.matrix_world.copy(); p.parent = e; p.matrix_parent_inverse = e.matrix_world.inverted(); p.matrix_world = w
    return e

def place(thing, at=None, rot=None):
    """Stand a thing (an object, a group, or its name) at `at` and turn it to `rot` degrees about the upright axis."""
    o = _obj(thing)
    if rot is not None: o.rotation_euler = _rot(rot); bpy.context.view_layer.update()
    if at is not None:
        lo, hi = _bounds([p for p in _family(o) if p.type == "MESH"] or [o])
        o.location += Vector(at) - Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    bpy.context.view_layer.update(); return o

def delete(thing):
    """Remove a thing and all its parts."""
    for p in _family(_obj(thing)): bpy.data.objects.remove(p)

def _tops():
    """Each thing of the scene: a top-level object that is no part of the room, with its bounds."""
    out = []
    for o in bpy.context.scene.objects:
        if o.parent is not None or o.get("room_part") or o.type in ("LIGHT", "CAMERA"): continue
        fam = [p for p in _family(o) if p.type == "MESH"]
        if fam: out.append((o, *_bounds(fam)))
    return out

def things():
    """One line per thing: name, size, where it stands."""
    return "\n".join(f"{o.name}: {hi.x - lo.x:.2f} x {hi.y - lo.y:.2f} x {hi.z - lo.z:.2f} m, stands at ({(lo.x + hi.x) / 2:.2f}, {(lo.y + hi.y) / 2:.2f}, {lo.z:.2f})" for o, lo, hi in _tops())

def check(limit=12):
    """What looks wrong, as lines of text ("" when nothing does)."""
    sc = bpy.context.scene; rm = sc.get("room"); tops = _tops(); bad = []
    if not rm and not sc.get("ground"): bad.append("there is no room yet: call room(...) first (or ground(...) for a place outdoors)")
    lights = [o for o in sc.objects if o.type == "LIGHT"]
    if tops and not lights and not sc.get("sky"): bad.append("there is no light yet: add lamp(...) or sky(...)")
    for o, lo, hi in tops:
        size = hi - lo; n = o.name
        if max(size) > 40: bad.append(f"{n} is {max(size):.1f} m long: far too big"); continue
        if rm:
            over = max(abs(lo.x) - rm[0] / 2, abs(hi.x) - rm[0] / 2, abs(lo.y) - rm[1] / 2, abs(hi.y) - rm[1] / 2)
            if over > 0.03: bad.append(f"{n} sticks {over:.2f} m out through a wall (the room is {rm[0]:g} x {rm[1]:g} m, centred on 0, 0)")
            if hi.z > rm[2] + 0.03: bad.append(f"{n} goes {hi.z - rm[2]:.2f} m up through the ceiling (the room is {rm[2]:g} m high)")
        if lo.z < -0.03: bad.append(f"{n} is {-lo.z:.2f} m under the floor")
        elif lo.z > 0.03:
            held = rm and (hi.z > rm[2] - 0.03 or min(rm[0] / 2 - abs(lo.x), rm[0] / 2 - abs(hi.x), rm[1] / 2 - abs(lo.y), rm[1] / 2 - abs(hi.y)) < 0.1)     # hangs from the ceiling or on a wall
            under = any(p is not o and abs(phi.z - lo.z) < 0.03 and plo.x < (lo.x + hi.x) / 2 < phi.x and plo.y < (lo.y + hi.y) / 2 < phi.y for p, plo, phi in tops)
            inside = any(p is not o and plo.z < lo.z < phi.z and plo.x < (lo.x + hi.x) / 2 < phi.x and plo.y < (lo.y + hi.y) / 2 < phi.y for p, plo, phi in tops)       # on a shelf inside something
            if not (held or under or inside): bad.append(f"{n} hangs in the air: its underside is at z = {lo.z:.2f} with nothing under it")
    for i, (a, alo, ahi) in enumerate(tops):
        for b, blo, bhi in tops[i + 1:]:
            ov = [min(ahi[k], bhi[k]) - max(alo[k], blo[k]) for k in range(3)]
            if min(ov) <= 0.02: continue
            va = (ahi - alo).x * (ahi - alo).y * (ahi - alo).z; vb = (bhi - blo).x * (bhi - blo).y * (bhi - blo).z; small = min(va, vb)
            if small > 1e-5 and ov[0] * ov[1] * ov[2] > 0.4 * small and min(va, vb) / max(va, vb) > 0.15: bad.append(f"{a.name} and {b.name} are inside each other")
    return "\n".join(bad[:limit] + ([f"... and {len(bad) - limit} more"] if len(bad) > limit else []))

room.__doc__ = _room.__doc__

def _usage(name):
    """A helper's arguments and what it is for, to put beside an error in a call of it."""
    import inspect
    f = {"room": _room}.get(name) or globals().get(name)
    return f"{name}{inspect.signature(f)}\n{inspect.getdoc(f)}" if callable(f) else ""

_before, _props = set(), {}
def _begin():
    """Note what is in the scene before a step of the model's code runs."""
    global _before, _props; _before = {o.name for o in bpy.data.objects}; sc = bpy.context.scene
    _props = {k: (list(sc[k]) if hasattr(sc[k], "__len__") else sc[k]) for k in ("room", "ground", "sky") if k in sc.keys()}

def _rollback():
    """Take out what a failed step had made so far, so that running the mended step again does not make it twice."""
    for o in [o for o in bpy.data.objects if o.name not in _before]: bpy.data.objects.remove(o)
    sc = bpy.context.scene
    for k in ("room", "ground", "sky"):
        if k in _props: sc[k] = _props[k]
        elif k in sc.keys(): del sc[k]

# ---- what blender_world.py does with a thing once the model has built it at the origin: make it one thing, bring it to
# ---- the planned size if it is far off, and stand it where the plan says

def _size(o):
    lo, hi = _bounds([p for p in _family(o) if p.type == "MESH"]); return hi - lo

def _collect(name):
    """Everything the step made, as one thing called `name`. -> its size (x, y, z)."""
    new = [o for o in bpy.data.objects if o.name not in _before and o.parent is None]
    if not any(p.type == "MESH" and len(p.data.polygons) for o in new for p in _family(o)): raise ValueError("the code ran but made no shapes")
    old = bpy.data.objects.get(name)
    if old is not None and old.name in _before: delete(old)
    if len(new) == 1 and new[0].type == "EMPTY": e = new[0]
    else:
        for o in new:
            if o.name == name: o.name = name + "_part"
        e = group(name, new)
    e.name = name; e["thing"] = True
    lo, hi = _bounds([p for p in _family(e) if p.type == "MESH"])                # stand it on the origin whatever the code did
    e.location -= Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z)); bpy.context.view_layer.update()
    return [round(v, 3) for v in _size(e)]

def _fit(name, want, low=0.8, high=1.25):
    """Bring a thing to the planned size along any axis where it is off by more than a fifth (a chair built 2 m wide),
    and under the ceiling whatever the plan said. -> the axes changed."""
    e = _obj(name); have = _size(e); changed = []; rm = bpy.context.scene.get("room"); want = list(want)
    if rm: want[2] = min(want[2], rm[2] - 0.03)
    for i, ax in enumerate("xyz"):
        if want[i] > 0.004 and have[i] > 1e-4 and (not low <= have[i] / want[i] <= high or (i == 2 and rm and have[i] > rm[2] - 0.03)): e.scale[i] *= want[i] / have[i]; changed.append(ax)
    bpy.context.view_layer.update(); return changed

def _levels(base):
    """The heights of a thing's flat upward surfaces big enough to stand something on: a table has one, a bookcase one per shelf."""
    zs = []
    for p in _family(base):
        if p.type != "MESH": continue
        R = p.matrix_world.to_3x3()
        for f in p.data.polygons:
            if f.area > 0.01 and (R @ f.normal).normalized().z > 0.9: zs.append((p.matrix_world @ f.center).z)
    out = []
    for z in sorted(zs):
        if not out or z - out[-1] > 0.04: out.append(z)
    return out

FACE = {"north": 0, "south": 180, "east": -90, "west": 90}          # the turn that brings a thing's front (-y) to face into the room from that wall

def _clear(e, lo, hi, others):
    """Is the box lo..hi free of the other things that stand at its level? Flat things (rugs) never block."""
    for o, a, b in others:
        if o is e or b.z - a.z < 0.06 or hi.z - lo.z < 0.06: continue
        if min(hi.x, b.x) - max(lo.x, a.x) > 0.04 and min(hi.y, b.y) - max(lo.y, a.y) > 0.04 and min(hi.z, b.z) - max(lo.z, a.z) > 0.04: return False
    return True

def _put(name, where):
    """Stand a thing where the plan says. where is one of
      {"wall": side, "offset": m}                       on the floor, its back against that wall
      {"x": m, "y": m, "rot": degrees}                  on the floor, free standing
      {"on": other thing, "dx": m, "dy": m, "rot": d}   on top of another thing
      {"hang": side, "offset": m, "height": m}          on a wall, its underside `height` above the floor
      {"ceiling": true, "x": m, "y": m}                 hanging from the ceiling
    A place already taken is left for the nearest free one. -> a line saying where it went."""
    e = _obj(name); sc = bpy.context.scene; rm = sc.get("room"); W, D, H = (rm[0], rm[1], rm[2]) if rm else (sc.get("ground", 30.0), sc.get("ground", 30.0), 1e9)
    num = lambda k, d=0.0: float(where.get(k, d) or 0) if isinstance(where.get(k, d), (int, float)) else d
    side = where.get("wall") or where.get("hang"); side = side if side in FACE else None
    base = bpy.data.objects.get(str(where.get("on"))) if where.get("on") else None
    others = [t for t in _tops() if t[0] is not e]
    def try_at(x, y, z, rot):
        place(e, (x, y, z), rot); s = _size(e)
        if rm: x = min(max(x, -W / 2 + s.x / 2), W / 2 - s.x / 2); y = min(max(y, -D / 2 + s.y / 2), D / 2 - s.y / 2); z = min(z, max(0.0, H - s.z))
        place(e, (x, y, z)); lo, hi = _bounds([p for p in _family(e) if p.type == "MESH"]); return (x, y, z), _clear(e, lo, hi, others)
    if base is not None and base is not e:
        lo, hi = _bounds([p for p in _family(base) if p.type == "MESH"]); s = _size(e); rot = num("rot")
        rx, ry = max(0.0, (hi.x - lo.x - s.x) / 2), max(0.0, (hi.y - lo.y - s.y) / 2)
        x = (lo.x + hi.x) / 2 + min(max(num("dx"), -rx), rx); y = (lo.y + hi.y) / 2 + min(max(num("dy"), -ry), ry)
        top = hi.z; lv = _levels(base); H = rm[2] if rm else 1e9
        if hi.z - lo.z > 1.3 and len(lv) > 2 or top + s.z > H - 0.02:          # shelves, or no room on top: a shelf, a different one each time, from eye height down
            gaps = [(z, (lv[i + 1] if i + 1 < len(lv) else H) - z) for i, z in enumerate(lv) if z < min(top, H) - 0.1 and z < 1.9]
            gaps = [g for g in gaps if g[1] > 0.12] or [(lv[0] if lv else lo.z, 0.3)]; k = int(base.get("stood", 0)); base["stood"] = k + 1
            top, gap = sorted(gaps, key=lambda g: -g[0])[k % len(gaps)]
            fit = min(1.0, (gap - 0.03) / max(s.z, 1e-4), 0.9 * (hi.x - lo.x) / max(s.x, 1e-4), 0.8 * (hi.y - lo.y) / max(s.y, 1e-4))
            if fit < 1.0: e.scale = [v * fit for v in e.scale]; bpy.context.view_layer.update(); s = _size(e)
            rx, ry = max(0.0, (hi.x - lo.x - s.x) / 2 - 0.03), max(0.0, (hi.y - lo.y - s.y) / 2 - 0.02)
            x = (lo.x + hi.x) / 2 + min(max(num("dx"), -rx), rx); y = (lo.y + hi.y) / 2 + min(max(num("dy"), -ry), ry)
            place(e, (x, y, top), rot); return f"{name} stands on a shelf of {base.name}"
        place(e, (x, y, hi.z), rot); lo2, hi2 = _bounds([p for p in _family(e) if p.type == "MESH"])
        if not _clear(e, lo2, hi2, [t for t in others if t[0] is not base]):          # taken: the free spot on the base nearest to it
            spots = sorted(((lo.x + s.x / 2 + i * (hi.x - lo.x - s.x) / 6, lo.y + s.y / 2 + j * (hi.y - lo.y - s.y) / 6) for i in range(7) for j in range(7)), key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
            for px, py in spots:
                place(e, (px, py, hi.z)); a, b = _bounds([p for p in _family(e) if p.type == "MESH"])
                if _clear(e, a, b, [t for t in others if t[0] is not base]): break
            else: place(e, (x, y, hi.z))
        return f"{name} stands on {base.name}"
    if where.get("ceiling") and rm:
        place(e, (0, 0, 0), num("rot")); s = _size(e); try_at(num("x"), num("y"), H - s.z, num("rot")); return f"{name} hangs from the ceiling"
    if side and rm:
        rot = FACE[side]; n, t = WALLS[side]; place(e, (0, 0, 0), rot); s = _size(e); deep = s.y if t[0] else s.x; half = (D if t[0] else W) / 2 - deep / 2; L = W if t[0] else D
        z = num("height", 1.2) if where.get("hang") else 0.0; off = num("offset"); best = None
        for step in sorted(range(-40, 41), key=abs):                               # along the wall from the planned spot, 15 cm at a time
            o = off + step * 0.15
            if abs(o) > L / 2: continue
            p, ok = try_at(n[0] * half + t[0] * o, n[1] * half + t[1] * o, z, rot)
            if best is None: best = p
            if ok: best = p; break
        place(e, best); return f"{name} is {'on' if where.get('hang') else 'against'} the {side} wall"
    x, y, rot = num("x"), num("y"), num("rot"); best = None
    for r in range(0, 16):                                                         # rings outwards from the planned spot
        ring = [(x, y)] if r == 0 else [(x + r * 0.2 * math.cos(a), y + r * 0.2 * math.sin(a)) for a in [k * math.pi / 6 for k in range(12)]]
        for px, py in ring:
            p, ok = try_at(px, py, 0.0, rot)
            if best is None: best = p
            if ok: best = p; break
        else: continue
        break
    place(e, best); return f"{name} stands at ({best[0]:.2f}, {best[1]:.2f})"

def _glow(name, watts, color="#ffe2b8"):
    """The light of a lamp the model built: just under a ceiling lamp, just over any other, so its shade does not swallow it."""
    e = _obj(name); lo, hi = _bounds([p for p in _family(e) if p.type == "MESH"]); rm = bpy.context.scene.get("room")
    up = rm and hi.z > rm[2] - 0.05; z = lo.z - 0.06 if up else min(hi.z + 0.06, (rm[2] - 0.05) if rm else 1e9)
    lamp(name + "_light", ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, z), watts, color, size=0.08)

# ---- a room made by the direct method's steps, brought in as its exported level (build/export.py), and the plain
# ---- furniture in it replaced by what the model builds

def _import_level(path, room):
    """The level's glTF -> the scene: the room's pieces marked as the room, every object a thing of its own (carrying the
    number of its layout object), the lamps kept as lights. room = (width, depth, height) in Blender's frame. -> counts"""
    sc = bpy.context.scene; before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o.name not in before]; bpy.context.view_layer.update()
    top = {o.name.split(".")[0]: o for o in new if o.type == "EMPTY" and o.parent is not None and o.parent.parent is None}
    def loose(o):
        w = o.matrix_world.copy(); o.parent = None; o.matrix_world = w
    n_room = n_things = n_lights = n_glass = 0
    for o in list(top.get("Room").children_recursive if top.get("Room") else []):
        loose(o); o["room_part"] = True; n_room += 1
        if o.type == "MESH" and o.name.lower().replace(" ", "").startswith("windowglass"): o["collide_only"] = True; n_glass += 1      # the level's pane glows to stand for daylight; here real sky shines through the opening
    for o in list(top.get("Objects").children if top.get("Objects") else []):
        loose(o); o["thing"] = True; n_things += 1
    for o in list(top.get("Lights").children_recursive if top.get("Lights") else []):
        if o.type == "LIGHT": loose(o); n_lights += 1
    for o in [o for o in new if o.type == "EMPTY" and not o.children and not o.get("thing")]: bpy.data.objects.remove(o)
    for o in [o for o in bpy.data.objects if o.type == "EMPTY" and o.name not in before and not o.children and not o.get("thing")]: bpy.data.objects.remove(o)
    sc["room"] = [float(v) for v in room]; sc["shell"] = 0.6
    w = sc.world or bpy.data.worlds.new("World"); sc.world = w                    # a closed room: no light but its own lamps and windows
    try: w.use_nodes = True
    except Exception: pass
    bg = next((n for n in w.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg: bg.inputs["Strength"].default_value = 0.0
    return {"room": n_room, "things": n_things, "lights": n_lights, "windows": n_glass}

def _texmat(name, path, px_per_m, rgb, cls=""):
    """A material from a swatch picture of true scale (px_per_m), laid on every side of a shape at that scale. rgb (0..255 sRGB)
    is its plain colour, used where the picture cannot be."""
    m = mat(name, "#%02x%02x%02x" % tuple(int(min(max(v, 0), 255)) for v in rgb), {"metal": 0.35, "wood": 0.5, "stone": 0.7, "glass": 0.1}.get(cls, 0.65), 1.0 if cls == "metal" else 0.0)
    b = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if b is None or not path: return m
    try: img = bpy.data.images.load(path, check_existing=True)
    except Exception: return m
    nt = m.node_tree; tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img; tex.projection = "BOX"; tex.projection_blend = 0.15
    try: tex.extension = "MIRROR"                                               # a swatch is not made to tile: mirrored, its edges meet
    except Exception: pass
    co = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping"); k = float(px_per_m) / max(img.size[0], 1)
    mp.inputs["Scale"].default_value = (k, k, k); nt.links.new(co.outputs["Object"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], b.inputs["Base Color"]); return m

def _drop_object(index):
    """Take out the level's own version of a layout object (the model has built another)."""
    for o in [o for o in bpy.data.objects if o.get("thing") and o.get("object") == index]: delete(o)

def _copy(name, new):
    """A second one of a thing, parts and all. -> its name"""
    src = _obj(name); twin = {}
    for o in _family(src):
        c = o.copy(); bpy.context.scene.collection.objects.link(c); twin[o] = c
    for o, c in twin.items():
        if o.parent in twin: c.parent = twin[o.parent]; c.matrix_parent_inverse = o.matrix_parent_inverse.copy()
    twin[src].name = new; bpy.context.view_layer.update(); return twin[src].name

def _exact(name, size):
    """Bring a thing to exactly this size (x, y, z): it stands in for a planned box that other things were placed around."""
    e = _obj(name); have = _size(e)
    for i in range(3):
        if have[i] > 1e-4 and size[i] > 0.004: e.scale[i] *= size[i] / have[i]
    bpy.context.view_layer.update(); return [round(v, 3) for v in _size(e)]

def _gpu():
    """Cycles on the Mac's GPU, for the pictures the model is shown of what it built."""
    sc = bpy.context.scene; sc.render.engine = "CYCLES"
    try:
        p = bpy.context.preferences.addons["cycles"].preferences; p.compute_device_type = "METAL"; p.get_devices()
        for dv in p.devices: dv.use = dv.type == "METAL"
        sc.cycles.device = "GPU"
    except Exception: pass

def _snap(name, path, px=448):
    """A picture of one thing alone, from its front and a little above and aside, on white, evenly lit: for the model to look at."""
    e = _obj(name); fam = set(_family(e)); sc = bpy.context.scene; lo, hi = _bounds([p for p in fam if p.type == "MESH"]); c = (lo + hi) / 2; r = max((hi - lo).length / 2, 0.05)
    was = [(o, o.hide_render) for o in sc.objects]
    for o in sc.objects: o.hide_render = o not in fam
    cam = bpy.data.objects.new("snap_cam", bpy.data.cameras.new("snap_cam")); sun = bpy.data.objects.new("snap_sun", bpy.data.lights.new("snap_sun", "SUN")); sun.data.energy = 2.5; sun.data.angle = 0.3
    for o in (cam, sun): sc.collection.objects.link(o)
    cam.location = c + Vector((-0.6, -1.0, 0.55)).normalized() * r * 3.1; cam.rotation_euler = (c - cam.location).to_track_quat("-Z", "Y").to_euler(); cam.data.lens = 50
    sun.rotation_euler = Vector((-0.3, -0.6, 1.0)).normalized().to_track_quat("Z", "Y").to_euler()
    w = sc.world or bpy.data.worlds.new("World"); sc.world = w
    try: w.use_nodes = True
    except Exception: pass
    bg = next((n for n in w.node_tree.nodes if n.type == "BACKGROUND"), None); old = (tuple(bg.inputs["Color"].default_value), bg.inputs["Strength"].default_value) if bg else None
    if bg: bg.inputs["Color"].default_value = (1, 1, 1, 1); bg.inputs["Strength"].default_value = 1.0
    rd = sc.render; keep = (sc.camera, rd.resolution_x, rd.resolution_y, rd.resolution_percentage, rd.filepath, rd.image_settings.file_format, sc.cycles.samples, sc.cycles.use_denoising)
    try:
        sc.camera = cam; rd.resolution_x = rd.resolution_y = int(px); rd.resolution_percentage = 100; rd.filepath = path; rd.image_settings.file_format = "PNG"; sc.cycles.samples = 12; sc.cycles.use_denoising = True
        bpy.ops.render.render(write_still=True)
    finally:
        sc.camera, rd.resolution_x, rd.resolution_y, rd.resolution_percentage, rd.filepath, rd.image_settings.file_format, sc.cycles.samples, sc.cycles.use_denoising = keep
        if bg: bg.inputs["Color"].default_value = old[0]; bg.inputs["Strength"].default_value = old[1]
        for o, h in was:
            try: o.hide_render = h
            except ReferenceError: pass
        for o in (cam, sun): d = o.data; bpy.data.objects.remove(o); (bpy.data.cameras if isinstance(d, bpy.types.Camera) else bpy.data.lights).remove(d)
    return path
