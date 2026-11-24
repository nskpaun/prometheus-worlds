"""Art direction, between putting a place together in Blender and baking its light (pipeline.py runs it on both routes).
Two things the RFC "visuals first" (docs/rfc-visuals-first.md, pillars 2 and 4) asks for, done with what is on this Mac:

  1. A real sky. Until now a window showed one flat colour. Here the picture model (FLUX.2 Klein) paints the view out of
     this building at its hour as a wide panorama; it is laid round the scene as Blender's world, so the sky's own colours
     come in by the windows and light the bake, and the viewer shows the same picture behind the windows.
  2. A look before the bake. The scene is rendered from a few places (Cycles, a few seconds); the world-plan model, with
     its picture tower, looks at them beside the sentence and an inventory of things and lamps, and names what is wrong:
     a thing that should not be there (a wall's plaster turned into a slab), a thing hanging in the air, lamps that make a
     night room look like noon. It answers in a small fixed set of mends, which code applies within bounds: take a thing
     out, set it down on what is under it, scale it, change a lamp, change the exposure. Two rounds at most.

  blender_direct.py WORLD_DIR ["SENTENCE"]             (thread/.venv) the whole step; WORLD_DIR/blender/scene.blend is changed in place
Run inside Blender by the above:
  blender -b scene.blend --python blender_direct.py -- sky  WORLD_DIR PICTURE.png HOUR
  blender -b scene.blend --python blender_direct.py -- look WORLD_DIR OUT_DIR
  blender -b scene.blend --python blender_direct.py -- mend WORLD_DIR MENDS.json
  blender -b scene.blend --python blender_direct.py -- inv  WORLD_DIR OUT.json
BLENDER_SKY=0 leaves the sky as it was; BLENDER_DIRECT=0 skips the look (0, 1 or 2 rounds; default 2)."""
import sys, os, json, math, time

HERE = os.path.dirname(os.path.abspath(__file__))
HOURS = ("day", "dusk", "night")
# how bright the sky is laid in at each hour (Blender's world strength for an average sky pixel of 1), and how high its sun stands
SKY_STRENGTH = {"day": 1.6, "dusk": 0.4, "night": 0.04}
SKY_SEEN = {"day": 0.15, "dusk": 0.3, "night": 1.0}     # the sky as the eye sees it through a window, against the light it casts: a photographer's window gel, so the view is not burnt white
HOUR_WORDS = {"day": "on a clear bright day, soft white clouds", "dusk": "at sunset, warm orange and pink sky low on the horizon, the first lights coming on",
              "night": "at night, deep blue night sky, a few lit windows and street lamps glowing warm in the distance"}

def exposure(lum, hour, ev=0.0):
    """The one exposure for a scene from the light falling on its surfaces (`lum`: samples of a lit white surface's
    brightness). By day a middling surface comes out a middle grey, as before. At dusk and at night the brighter patches
    (the pools under lamps) set it, so the rest is allowed to fall dark, as it would to the eye; never brighter than by day.
    ev: the art direction's own correction, in stops."""
    import numpy as np
    lum = np.asarray(lum, np.float64); lum = lum[np.isfinite(lum) & (lum > 0)]
    if not len(lum): return 1.0
    day = (0.62 / max(float(np.percentile(lum, 55)), 1e-6)) ** 0.9
    if hour == "night": g = float(np.clip(1.25 / max(float(np.percentile(lum, 96)), 1e-6), day * 0.35, day))      # never more than 1.5 stops under the day's: dark, still legible
    elif hour == "dusk": g = float(np.clip(0.9 / max(float(np.percentile(lum, 88)), 1e-6), day * 0.55, day))
    else: g = day
    return float(np.clip(g * 2.0 ** max(-3.0, min(3.0, ev)), 1e-3, 1e4))

def white_balance(samples, keep=0.4):
    """Per-channel factors that take (1 - keep) of the average colour cast of the light out, as a camera's white balance
    does, without changing its brightness: a room lit by tungsten lamps stays warm but no longer all orange."""
    import numpy as np
    x = np.concatenate([np.asarray(a, np.float64).reshape(-1, 3) for a in samples if len(a)]) if any(len(a) for a in samples) else np.ones((1, 3))
    x = x[np.isfinite(x).all(1)]; m = x.mean(0) if len(x) else np.ones(3); m = np.maximum(m, 1e-6)
    y = float(m @ [0.2126, 0.7152, 0.0722]); f = (y / m) ** (1 - keep); f /= float(f @ [0.2126, 0.7152, 0.0722]) or 1.0
    return f.astype(np.float32)

# ------------------------------------------------------------------------------------------------ inside Blender
def _bl():
    import bpy, bmesh
    from mathutils import Vector
    return bpy, bmesh, Vector

def family(o): return [o] + list(o.children_recursive)

def tops():
    """The things of the scene: parentless objects that are not part of the room, with their bounds."""
    bpy, _, Vector = _bl(); out = []
    for o in bpy.context.scene.objects:
        if o.parent is None and not o.get("room_part") and o.type in ("MESH", "EMPTY") and not o.get("collide_only"):
            pts = [p.matrix_world @ Vector(c) for p in family(o) if p.type == "MESH" for c in p.bound_box]
            if pts: out.append((o, Vector([min(p[i] for p in pts) for i in range(3)]), Vector([max(p[i] for p in pts) for i in range(3)])))
    return out

def below(o, lo, hi):
    """How far the underside of a thing is above whatever is under its middle (floor, table, shelf). None: nothing under it."""
    bpy, _, Vector = _bl(); sc = bpy.context.scene; dg = bpy.context.evaluated_depsgraph_get(); mine = set(family(o))
    best = None
    for fx, fy in ((0.5, 0.5), (0.3, 0.3), (0.7, 0.7), (0.3, 0.7), (0.7, 0.3)):
        p = Vector((lo.x + (hi.x - lo.x) * fx, lo.y + (hi.y - lo.y) * fy, lo.z + 0.002)); z0 = p.z
        for _ in range(8):                                                       # step past the thing's own faces
            hit, loc, _n, _i, ob, _m = sc.ray_cast(dg, p, Vector((0, 0, -1)))
            if not hit: break
            if ob in mine or ob.get("collide_only"): p = loc - Vector((0, 0, 0.001)); continue
            gap = z0 - 0.002 - loc.z; best = gap if best is None else min(best, gap); break
    return best

def near_wall(lo, hi, room, t=0.12):
    if not room: return False
    W, D = room[0] / 2, room[1] / 2
    return lo.x < -W + t or hi.x > W - t or lo.y < -D + t or hi.y > D - t

def hex_of(c): return "#%02x%02x%02x" % tuple(int(max(0, min(1, v)) ** (1 / 2.2) * 255 + 0.5) for v in c[:3])

def inventory():
    """What the model is shown besides the pictures: each thing (name, size, where, how far its underside hangs over what
    is under it) and each lamp."""
    bpy, _, Vector = _bl(); sc = bpy.context.scene; room = list(sc["room"]) if sc.get("room") else None; things = []
    for o, lo, hi in tops():
        gap = below(o, lo, hi); size = hi - lo
        t = {"name": o.name, "size_m": [round(size.x, 2), round(size.y, 2), round(size.z, 2)], "middle": [round((lo.x + hi.x) / 2, 2), round((lo.y + hi.y) / 2, 2)], "bottom_z": round(lo.z, 2)}
        if gap is not None and gap > 0.06 and not near_wall(lo, hi, room) and not (room and hi.z > room[2] - 0.1): t["hangs_m"] = round(gap, 2)     # not on a wall, not from the ceiling: in the air
        things.append(t)
    lamps = [{"name": o.name, "kind": o.data.type.lower(), "watts": round(o.data.energy, 1), "colour": hex_of(o.data.color), "at": [round(v, 2) for v in o.location], "on": not o.hide_render}
             for o in sc.objects if o.type == "LIGHT" and o.data.type != "SUN"]
    sun = next((o for o in sc.objects if o.type == "LIGHT" and o.data.type == "SUN"), None)
    return {"room_m": room, "things": things, "lamps": lamps, "sun": round(sun.data.energy, 2) if sun else 0, "exposure_ev": float(sc.get("exposure_ev", 0.0)), "hour": sc.get("hour"),
            "sky_strength": float(sc["sky"][3]) if sc.get("sky") else None}

def cycles(samples, w, h):
    bpy, _, _ = _bl(); sys.path.insert(0, HERE); import blender_finish as F
    F.gpu(); sc = bpy.context.scene; c = sc.cycles; c.samples = samples; c.use_denoising = True; c.max_bounces = 4; c.diffuse_bounces = 3; c.sample_clamp_indirect = 8.0
    try: c.denoiser = "OPENIMAGEDENOISE"
    except Exception: pass
    rd = sc.render; rd.resolution_x, rd.resolution_y, rd.resolution_percentage = w, h, 100; rd.image_settings.file_format = "OPEN_EXR"
    sc.view_settings.view_transform = "Standard"; sc.view_settings.look = "None"; sc.view_settings.exposure = 0

def views():
    """Where to look from: the start (as blender_finish chooses it), then from each corner across the room, and one low and
    close on the busiest spot. -> [(name, eye, target)]"""
    bpy, _, Vector = _bl(); sys.path.insert(0, HERE); import blender_finish as F
    sc = bpy.context.scene; room = list(sc["room"]) if sc.get("room") else None; tt = tops(); boxes = [(lo, hi) for _, lo, hi in tt]
    allp = [p for _, lo, hi in tt for p in (lo, hi)] or [Vector((-3, -3, 0)), Vector((3, 3, 2))]
    lo = Vector([min(p[i] for p in allp) for i in range(3)]); hi = Vector([max(p[i] for p in allp) for i in range(3)])
    eye, fwd = F.stand(boxes, room, lo, hi); out = [("start", Vector(eye), Vector(eye) + Vector(fwd) * 3)]
    W, D = (room[0] / 2 - 0.45, room[1] / 2 - 0.45) if room else ((hi.x - lo.x) / 2 + 1.5, (hi.y - lo.y) / 2 + 1.5)
    cx, cy = (0.0, 0.0) if room else ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2)
    for k, (sx, sy) in enumerate(((1, 1), (-1, -1), (1, -1))):
        for f in (1.0, 0.85, 0.7, 0.55, 0.4):                                     # stepped in from the corner until no thing is in the way of the eye
            e = Vector((cx + sx * W * f, cy + sy * D * f, 1.6))
            if all(max(a.x - e.x, e.x - b.x, a.y - e.y, e.y - b.y) > 0.35 or b.z < 1.2 or a.z > 2.0 for a, b in boxes): break
        out.append((f"corner{k + 1}", e, Vector((cx - sx * W * 0.3, cy - sy * D * 0.3, 0.9))))
    if tt:
        big = max(tt, key=lambda x: (x[2] - x[1]).length); c = (big[1] + big[2]) / 2; d = Vector((cx - c.x, cy - c.y, 0))
        d = d.normalized() if d.length > 0.1 else Vector((0, -1, 0)); r = max((big[2] - big[1]).length * 0.9, 1.2)
        out.append(("close", Vector((c.x + d.x * r, c.y + d.y * r, 1.25)), Vector((c.x, c.y, (big[1].z + big[2].z) / 2))))
    return out

def look(d, out_dir, samples=24, w=768, h=432):
    """Render each view (linear, saved as .npy) and the inventory; the caller tones and judges them."""
    bpy, _, Vector = _bl(); sc = bpy.context.scene; os.makedirs(out_dir, exist_ok=True); cycles(samples, w, h)
    for o in sc.objects:
        if o.get("collide_only"): o.hide_render = True                           # as the bake will have it: the level's glowing window pane is not there, the sky is
    cam = bpy.data.objects.new("look", bpy.data.cameras.new("look")); sc.collection.objects.link(cam); cam.data.lens_unit = "FOV"; cam.data.angle = math.radians(72); sc.camera = cam
    made = []
    for name, e, t in views():
        cam.location = e; cam.rotation_euler = (t - e).to_track_quat("-Z", "Y").to_euler(); sc.render.filepath = os.path.join(out_dir, name + ".exr")
        bpy.ops.render.render(write_still=True); img = bpy.data.images.load(sc.render.filepath); import numpy as np
        px = np.empty(img.size[0] * img.size[1] * 4, np.float32); img.pixels.foreach_get(px); bpy.data.images.remove(img)
        np.save(os.path.join(out_dir, name + ".npy"), px.reshape(h, w, 4)[::-1, :, :3].astype(np.float16)); os.remove(sc.render.filepath); made.append(name)
    bpy.data.objects.remove(cam)
    json.dump({"views": made, **inventory()}, open(os.path.join(out_dir, "inventory.json"), "w"), indent=1)

def set_sky(d, picture, hour):
    """The painted panorama (equirectangular, its middle along +x as the viewer has it) as the world round the scene,
    as bright on average as the plain sky it replaces. The sun stays as it was."""
    bpy, _, _ = _bl(); import numpy as np
    sc = bpy.context.scene; w = sc.world or bpy.data.worlds.new("World"); sc.world = w
    try: w.use_nodes = True
    except Exception: pass
    nt = w.node_tree; bg = next((n for n in nt.nodes if n.type == "BACKGROUND"), None) or nt.nodes.new("ShaderNodeBackground")
    out = next(n for n in nt.nodes if n.type == "OUTPUT_WORLD")
    for n in [n for n in nt.nodes if n.type in ("TEX_ENVIRONMENT", "MAPPING", "TEX_COORD")]: nt.nodes.remove(n)
    img = bpy.data.images.load(picture, check_existing=False); img.colorspace_settings.name = "sRGB"; img.pack()
    px = np.empty(img.size[0] * img.size[1] * 4, np.float32); img.pixels.foreach_get(px); mean = float((px.reshape(-1, 4)[:, :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)).mean())
    old = list(sc["sky"]) if sc.get("sky") else None
    want = (old[3] * max(float(np.dot(old[:3], [0.2126, 0.7152, 0.0722])), 1e-4) / 0.5) if old and old[3] > 0 else SKY_STRENGTH.get(hour, 1.0)   # the old sky's light, in units of a mid-grey sky
    k = float(np.clip(want * 0.5 / max(mean, 1e-4), 0.005, 50))
    for n in [n for n in nt.nodes if n.type in ("LIGHT_PATH", "MIX_SHADER") or (n.type == "BACKGROUND" and n != bg)]: nt.nodes.remove(n)
    env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = img; env.projection = "EQUIRECTANGULAR"
    nt.links.new(env.outputs["Color"], bg.inputs["Color"]); bg.inputs["Strength"].default_value = k
    seen = nt.nodes.new("ShaderNodeBackground"); nt.links.new(env.outputs["Color"], seen.inputs["Color"]); seen.inputs["Strength"].default_value = k * SKY_SEEN.get(hour, 1.0)
    lp = nt.nodes.new("ShaderNodeLightPath"); mix = nt.nodes.new("ShaderNodeMixShader")       # what lights the room: the full sky; what the camera sees of it: the dimmer one
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"]); nt.links.new(bg.outputs["Background"], mix.inputs[1]); nt.links.new(seen.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"]); sc["sky_seen"] = SKY_SEEN.get(hour, 1.0)
    sc["sky_picture"] = os.path.abspath(picture); sc["sky_strength"] = k; sc["hour"] = hour
    print(f"sky: {os.path.basename(picture)} at strength {k:.3g} ({hour})", flush=True)

def lights_named(sc, key):
    key = str(key).strip().lower()
    import re
    exact = [o for o in sc.objects if o.type == "LIGHT" and re.fullmatch(re.escape(key) + r"(\.\d{3})?", o.name.lower())]      # a name and Blender's copies of it (".001")
    return exact or [o for o in sc.objects if o.type == "LIGHT" and o.data.type != "SUN" and (o.name.lower().startswith(key) or key in o.name.lower())]

def glow_near(o, k):
    """Scale the glow of what shows a lamp (its shade, bulb or diffuser): emissive materials on meshes close to the light."""
    bpy, _, Vector = _bl(); done = set()
    for m in bpy.context.scene.objects:
        if m.type != "MESH": continue
        c = sum((m.matrix_world @ Vector(v) for v in m.bound_box), Vector()) / 8
        if (c - o.location).length > 0.45: continue
        for s in m.material_slots:
            mt = s.material
            if not mt or mt.name in done or not mt.node_tree: continue
            b = next((n for n in mt.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if b is not None and "Emission Strength" in b.inputs and b.inputs["Emission Strength"].default_value > 0: b.inputs["Emission Strength"].default_value *= k; done.add(mt.name)
            for n in mt.node_tree.nodes:
                if n.type == "EMISSION": n.inputs["Strength"].default_value *= k; done.add(mt.name)

def kelvin(t):
    """A colour temperature -> a lamp's linear RGB as a camera balanced for 4000 K sees it (as film is balanced for indoor
    light), so 2700 K reads as warm cream rather than deep orange and 6500 K as a cool white. Brightest channel 1."""
    a, b = _kelvin(t), _kelvin(4000)
    c = [x / max(y, 1e-4) for x, y in zip(a, b)]; m = max(c); return tuple(x / m for x in c)

def _kelvin(t):
    """Tanner Helland's fit of a black body's colour, in linear RGB."""
    t = max(1000.0, min(12000.0, float(t))) / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    return tuple((max(0.0, min(255.0, v)) / 255.0) ** 2.2 for v in (r, g, b))

def mend(d, path):
    """Apply the mends the model chose, each within bounds. -> what was done, one line each"""
    bpy, _, Vector = _bl(); sc = bpy.context.scene; done = []; touched = set()
    things = {o.name.lower(): (o, lo, hi) for o, lo, hi in tops()}
    for f in json.load(open(path)):
        op = f.get("op"); name = str(f.get("thing") or f.get("lamp") or "").strip(); t = things.get(name.lower())
        try:
            if op == "remove" and t:
                o, lo, hi = t
                for L in [L for L in sc.objects if L.type == "LIGHT" and all(lo[i] - 0.1 <= L.location[i] <= hi[i] + 0.1 for i in range(3))]: bpy.data.objects.remove(L)
                for p in reversed(family(o)): bpy.data.objects.remove(p)
                things.pop(name.lower()); done.append(f"took out {name}")
            elif op == "ground" and t:
                o, lo, hi = t; gap = below(o, lo, hi); room = list(sc["room"]) if sc.get("room") else None
                if near_wall(lo, hi, room) and min(hi.x - lo.x, hi.y - lo.y) < 0.2 and lo.z > 0.3: done.append(f"left {name} on its wall"); continue    # a picture, a rack, a shelf: hung, not dropped
                if gap is not None and 0.02 < gap < 3.0:
                    o.location.z -= gap
                    for L in [L for L in sc.objects if L.type == "LIGHT" and all(lo[i] - 0.1 <= L.location[i] <= hi[i] + 0.1 for i in range(3))]: L.location.z -= gap
                    done.append(f"set {name} down {gap:.2f} m")
            elif op == "scale" and t:
                o, lo, hi = t; k = max(0.4, min(2.0, float(f.get("by", 1))))
                if abs(k - 1) > 0.05:
                    piv = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z)); o.scale = o.scale * k; o.location = piv + (o.location - piv) * k; done.append(f"scaled {name} by {k:.2f}")
            elif op == "lamp":
                ls = [L for L in lights_named(sc, name) if L.name not in touched]; k = max(0.0, min(5.0, float(f.get("watts_times", 1))))
                for L in ls:                                                     # each lamp changed once a round, and never past 5 times (or under nothing of) the watts it came with
                    touched.add(L.name); w0 = float(L.get("watts_at_start", L.data.energy)); L["watts_at_start"] = w0
                    new = min(L.data.energy * k, 5.0 * w0); kk = new / L.data.energy if L.data.energy > 0 else 0.0
                    L.data.energy = new; glow_near(L, kk)
                    if k == 0: L.hide_render = True
                    if f.get("kelvin"): L.data.color = kelvin(f["kelvin"])
                if ls: done.append(f"lamps {name!r} ({len(ls)}): watts x{k:.2f}" + (f", {int(f['kelvin'])} K" if f.get("kelvin") else ""))
            elif op == "exposure":
                ev = max(-2.0, min(2.0, float(f.get("ev", 0)))); sc["exposure_ev"] = float(max(-1.0, min(1.0, sc.get("exposure_ev", 0.0) + ev))); done.append(f"exposure {ev:+.1f} EV")
        except Exception as e: done.append(f"could not {op} {name}: {e}")
    print("\n".join("MEND " + x for x in done), flush=True)

def blender_main(a):
    import bpy
    cmd, d = a[0], a[1]
    if cmd == "sky": set_sky(d, a[2], a[3])
    elif cmd == "look": look(d, a[2])
    elif cmd == "inv": json.dump({**inventory(), "windows": sum(1 for o in bpy.data.objects if o.type == "MESH" and "window" in o.name.lower().replace(" ", ""))}, open(a[2], "w"), indent=1)
    elif cmd == "mend": mend(d, a[2])
    else: raise SystemExit(f"unknown: {cmd}")
    if cmd in ("sky", "mend"): bpy.ops.wm.save_mainfile()

# ------------------------------------------------------------------------------------------------ outside Blender (thread/.venv)
CRITIC = """You are the art director of a 3D place made from this description:
"{sentence}"
It is {hour}. The light it should have: {light}

The picture shows the place from {n} points of view (labelled), as it will look after the light is baked. Below is an
inventory: every thing (name, size in metres as width, depth, height, its middle, its underside's height; "hangs_m" means
its underside is that far above whatever is under it, in the air) and every lamp.
{inventory}

Judge it as a film's art director would, by what you SEE and by the inventory:
- Things that do not belong: a big flat slab or blank panel that is really a wall's or floor's material made into an object
  (named after a material: plaster, paint, wood floor), a second copy of something that should be one, a pot floating
  above its plant. Take them out.
- Things hanging in the air that should stand on something: set them down. Lamps on the ceiling and things on walls are fine.
- A thing far too big or small for what it is: scale it (0.5 to 2).
- The light. Does it match the hour and the light described? At night a room is lit only by its lamps: warm pools of
  light and dark corners, not even white light everywhere; ceiling strip lights a description does not ask for should be
  dimmed or off (watts_times 0 to 0.3), the lamps it names made the main light (watts_times 1.5 to 3, warm: 2400 to 3000 K).
  By day, light from the windows should lead. A scene that is too dark or washed out overall: exposure, in stops.
Mends you may give (use names exactly as in the inventory; for lamps a name or the start of several lamps' names):
  {{"op": "remove", "thing": NAME}}
  {{"op": "ground", "thing": NAME}}
  {{"op": "scale", "thing": NAME, "by": NUMBER}}
  {{"op": "lamp", "lamp": NAME, "watts_times": NUMBER, "kelvin": NUMBER or null}}
  {{"op": "exposure", "ev": NUMBER}}
Answer with JSON only: {{"score": 1 to 10 (how good it looks now), "faults": ["short sentence", ...], "mends": [at most 8 mends]}}
If it already looks right, give no mends."""

def tone(x, gain):
    import numpy as np
    x = np.nan_to_num(x, nan=0.0).clip(0, None) * gain; x = x * (1 + x / 16.0) / (1 + x)
    return (np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-8), 1 / 2.4) - 0.055).clip(0, 1) * 255 + 0.5).astype(np.uint8)

def load_linear(p):
    import numpy as np
    return np.load(p).astype(np.float32)

def sheet(look_dir, hour, ev):
    """The rendered views, toned at the exposure the bake will use, labelled, on one picture. -> path"""
    import numpy as np
    from PIL import Image, ImageDraw
    inv = json.load(open(os.path.join(look_dir, "inventory.json"))); ims = [(v, load_linear(os.path.join(look_dir, v + ".npy"))) for v in inv["views"]]
    lum = np.concatenate([(x[::4, ::4] @ np.array([0.2126, 0.7152, 0.0722], np.float32)).ravel() for _, x in ims]) / 0.45     # a surface's light, roughly: what it shows over a typical colour
    g = exposure(lum, hour, ev); tiles = []; wb = white_balance([x[::4, ::4] for _, x in ims])
    for v, x in ims:
        x = x * wb
        im = Image.fromarray(tone(x, g)); dr = ImageDraw.Draw(im); dr.rectangle((0, 0, 9 * len(v) + 14, 22), fill=(0, 0, 0)); dr.text((6, 5), v, fill=(255, 255, 255)); tiles.append(im)
    w, h = tiles[0].size; cols = 2; rows = (len(tiles) + 1) // 2; s = Image.new("RGB", (w * cols, h * rows), (20, 20, 20))
    for i, t in enumerate(tiles): s.paste(t, ((i % cols) * w, (i // cols) * h))
    out = os.path.join(look_dir, "sheet.jpg"); s.save(out, quality=88); return out, inv

def hour_of(d):
    try: h = json.load(open(os.path.join(d, "materials", "style.json"))).get("time")
    except (OSError, ValueError): h = None
    return h if h in HOURS else None

def sky_job(d, sentence, hour, work):
    """The picture job for the panorama: a wide view of what is outside, at the hour, with a lot of sky."""
    try: st = json.load(open(os.path.join(d, "materials", "style.json")))
    except (OSError, ValueError): st = {}
    import re
    kind = re.split(r"\s+(?:with|where|that|which|in which)\b|[,;(]", st.get("building") or "", maxsplit=1)[0].strip()      # "19th-century masonry townhouse", not its floors and beams
    seen = (f"the neighbourhood outside a {kind}: what its windows look out on (a street, gardens, a town, fields or hills, whichever suits it)" if kind else f"the far surroundings of this place: {sentence[:240]}")
    prompt = (f"A seamless wide 360 degree panoramic photograph taken OUTDOORS at eye level: {seen}, {HOUR_WORDS.get(hour, HOUR_WORDS['day'])}. "
              "The horizon low across the picture, the upper half open sky. Buildings, trees or hills in the middle distance and far away, nothing close to the camera, no interior, no window frames, no walls or floor around the camera. Photographic, natural colour, no people, no text.")
    return {"name": "sky", "out": os.path.join(work, "sky_raw.png"), "w": 2048, "h": 512, "seed": 11, "prompt": prompt}

def equirect(raw, out, hour):
    """A wide picture (2048x512, 90 degrees tall, horizon low) -> a 2048x1024 equirectangular panorama: the picture as the
    band from 30 degrees below the horizon to 60 above, its two ends blended so it closes round, the sky above carried up to
    the zenith and the ground below carried down, both softened."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    a = np.asarray(Image.open(raw).convert("RGB").resize((2048, 512), Image.LANCZOS), np.float32); W = a.shape[1]; b = 160
    t = np.linspace(0, 1, b, dtype=np.float32)[None, :, None]; a[:, :b] = a[:, :b] * t + a[:, -b:] * (1 - t)            # left edge fades from the right edge's colours: no seam
    a = np.concatenate([a[:, :W - b]], 1); a = np.asarray(Image.fromarray(a.clip(0, 255).astype(np.uint8)).resize((2048, 512), Image.LANCZOS), np.float32)
    H = 1024; top = int(H * (90 - 60) / 180); bot = top + 512; pano = np.zeros((H, 2048, 3), np.float32); pano[top:bot] = a
    zen = a[:24].mean((0, 1)); up = ndimage.gaussian_filter(a[:6].mean(0), (24, 0))
    for r in range(top):                                                         # above the band: the band's top, blurring into one zenith colour
        f = (top - r) / top; row = up * (1 - f ** 0.7) + zen * f ** 0.7; pano[r] = ndimage.gaussian_filter1d(row, 4 + 60 * f, axis=0, mode="wrap")
    gnd = a[-8:].mean(0); gm = a[-40:].mean((0, 1))
    for r in range(bot, H):
        f = (r - bot) / (H - bot); pano[r] = ndimage.gaussian_filter1d(gnd * (1 - f) + gm * f, 4 + 50 * f, axis=0, mode="wrap") * (1 - 0.35 * f)
    Image.fromarray(pano.clip(0, 255).astype(np.uint8)).save(out); return out

def main(d, said=""):
    import subprocess
    sys.path.insert(0, os.path.join(HERE, "build")); import chat, tools as C, blender_world as BW
    d = os.path.abspath(d); work = os.path.join(d, "blender"); blend = os.path.join(work, "scene.blend"); app = BW.find_blender(); t0 = time.time()
    sentence = said
    try: st = json.load(open(os.path.join(d, "materials", "style.json"))); sentence = sentence or st.get("request", "").split("\n\nThe same place")[0]; light = st.get("light", {}).get("words", "")
    except (OSError, ValueError): st = {}; light = ""
    def blender(*args):
        r = subprocess.run(BW.local_only([app, "-b", blend, "--python", os.path.abspath(__file__), "--"] + [str(x) for x in args]), capture_output=True, text=True)
        if r.returncode: raise SystemExit("Blender stopped during art direction: " + (r.stdout + r.stderr)[-600:])
        return r.stdout
    os.makedirs(os.path.join(work, "direct"), exist_ok=True); inv_p = os.path.join(work, "direct", "start.json")
    blender("inv", d, inv_p); inv0 = json.load(open(inv_p))
    sk = inv0.get("sky_strength"); hour = hour_of(d) or inv0.get("hour") or ("day" if sk is None or sk >= 0.5 else "dusk" if sk >= 0.1 else "night")
    # 1. the sky
    if os.environ.get("BLENDER_SKY") != "0" and (inv0.get("windows") or not inv0.get("room_m")):
        job = sky_job(d, sentence, hour, work); jp = os.path.join(work, "sky_jobs.json")
        if not os.path.exists(os.path.join(work, "sky.png")):
            json.dump([job], open(jp, "w"), indent=1); C.klein(jp); equirect(job["out"], os.path.join(work, "sky.png"), hour)
        print(blender("sky", d, os.path.join(work, "sky.png"), hour).strip().splitlines()[-1:] , flush=True)
    else: print("no windows: the sky is left as it was", flush=True)
    # 2. look and mend
    rounds = int(os.environ.get("BLENDER_DIRECT", "2")); log = []
    if rounds > 0:
        with chat.Chat(vision=True) as model:
            for r in range(1, rounds + 1):
                ld = os.path.join(work, "direct", f"r{r}"); blender("look", d, ld)
                inv = json.load(open(os.path.join(ld, "inventory.json"))); pic, _ = sheet(ld, hour, inv.get("exposure_ev", 0.0))
                text = CRITIC.format(sentence=sentence, hour=hour, light=light or "as the description says", n=len(inv["views"]),
                                     inventory=json.dumps({k: inv[k] for k in ("room_m", "things", "lamps", "sun", "exposure_ev")}, separators=(",", ":")))
                j = None
                for tries, extra in enumerate(("", "\n\nAnswer with the JSON object only, nothing before or after it.")):
                    a = model.say([{"role": "user", "content": [{"type": "image", "path": pic}, {"type": "text", "text": text + extra}]}], max_tokens=1400, temperature=0.0)
                    open(os.path.join(ld, f"answer{tries}.txt"), "w").write(a["text"])
                    try: j = C.json_in(a["text"]); break
                    except (ValueError, json.JSONDecodeError): pass
                if not isinstance(j, dict): print("the art director's answer was not JSON twice; left as it is", flush=True); break
                mends = [m for m in (j.get("mends") or []) if isinstance(m, dict)][:8]
                if any(m.get("op") == "lamp" for m in mends):                   # the exposure follows the lamps by itself: with lamp changes, only half of an exposure change, at most 3/4 stop
                    for m in mends:
                        if m.get("op") == "exposure":
                            try: m["ev"] = max(-0.75, min(0.75, float(m.get("ev", 0)) / 2))
                            except (TypeError, ValueError): m["ev"] = 0
                print(f"round {r}: score {j.get('score')}; " + "; ".join(str(x) for x in (j.get("faults") or [])[:6]), flush=True)
                log.append({"round": r, "score": j.get("score"), "faults": j.get("faults"), "mends": mends, "s": a["s"]})
                if not mends: break
                mp = os.path.join(ld, "mends.json"); json.dump(mends, open(mp, "w"), indent=1)
                print("\n".join(l[5:] for l in blender("mend", d, mp).splitlines() if l.startswith("MEND ")), flush=True)
    json.dump({"hour": hour, "rounds": log, "s": round(time.time() - t0, 1)}, open(os.path.join(work, "direct", "log.json"), "w"), indent=1)
    print(f"art direction took {time.time() - t0:.0f} s", flush=True)

if __name__ == "__main__":
    if "--" in sys.argv: blender_main(sys.argv[sys.argv.index("--") + 1:])
    else: main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
