"""A sentence -> the scene description (ir.SceneIR), for both kinds of place (RFC pillars 1 and 5).
  plan.py indoor  WORLD_DIR "SENTENCE"   after the layout step (build/layout.py has written WORLD_DIR/layout.json)
  plan.py outdoor WORLD_DIR "SENTENCE"   the model plans the ground, what grows on it and what stands on it
Indoors the layout's placing is kept (chairs round tables, things on things, clear floor, windows the sentence asks
for); one more answer of the world-plan model (Qwen3.5 35B-A3B) gives what a box of furniture cannot: the room's
outline (a bay, an alcove, an apse, cut corners), its ceiling, wainscot, every surface's material as words for the
material model, each lamp's colour temperature and power, the time of day and the sky. Each thing is then given its
source: a few named and large kinds become hero 3D models, things the CC0 library holds are taken from it, cloth is
draped, the rest are plain bevelled forms. Writes WORLD_DIR/pbr/scene_ir.json."""
import sys, os, re, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); THREAD = os.path.dirname(HERE)
sys.path.insert(0, THREAD); sys.path.insert(0, os.path.join(THREAD, "build")); sys.path.insert(0, HERE)
import ir, kitbash

HEROES = int(os.environ.get("WORLD_HEROES", "6"))           # kinds made by the 3D model: each costs minutes
LAMP = re.compile(r"\b(lamp|lamps|light|lights|lantern|chandelier|sconce|candle|candles|pendant|bulb|torch|fire|fireplace|brazier)\b")
NOT_LAMP = re.compile(r"\b(lighthouse|lighter|light switch|candle holder|lampshade only)\b")
CLOTH = re.compile(r"\b(curtain|curtains|drape|drapes|rug|carpet|mat|throw|blanket|tablecloth|towel|banner|flag|tapestry|sheet)\b")
PLANT = re.compile(r"\b(plant|fern|tree|bush|shrub|flower|flowers|grass|hedge|ivy|palm|cactus|succulent|bonsai|sapling)\b")
HOURS = ("day", "golden_hour", "dusk", "night")
WALLISH = re.compile(r"\b(wall|walls|fence|fencing|hedge|railing|balustrade|kerb|curb|steps|paving|parapet|terrace)\b")
SUN = {"day": (52, 4.5, 5800), "golden_hour": (12, 3.2, 3400), "dusk": (3, 1.2, 2600), "night": (38, 0.06, 7800)}     # height (degrees), strength, colour (K; night: the moon)

ARCHITECT = """You are the production designer of a 3D place that will be built and walked through. The place:
"{sentence}"
{more}
Its furniture is already placed in a room {W:.1f} m wide (x), {D:.1f} m deep (y) and {H:.1f} m high, with these openings: {openings}.
The things in it: {things}.

Decide what the furniture cannot: answer with JSON only, in this shape:
{{"mood": "a few words on the feel",
 "shape": {{"kind": "rect" | "bay" | "alcove" | "apse" | "chamfer", "wall": "back" | "front" | "left" | "right", "width": metres, "depth": metres}},
 "ceiling": "flat" | "beamed" | "vaulted" | "coffered" | "pitched",
 "panelling_m": 0 to 1.3 (a wainscot this high, 0 for none),
 "window_arch": "flat" | "round" | "pointed", "mullions": [panes across, panes up],
 "materials": {{"wall": M, "floor": M, "ceiling": M, "trim": M}},
 "lamps": {{"name of a lamp-like thing in the list": {{"kelvin": 1800 to 6500, "watts": 5 to 200}}}},
 "atmosphere": {{"time_of_day": "day" | "golden_hour" | "dusk" | "night", "outside": "what the windows look out on, one sentence", "fog_density": 0 to 0.03, "fog_color": "#hex"}}}}
where each M is {{"prompt": "the surface as a materials artist would name it: material, finish, colour, wear", "scale_m": metres one tile covers (planks 1.5, tiles 0.6, plaster 2), "rgb": "#hex", "rough": 0 to 1, "metal": 0 to 1}}.
The outline adds to the room, it never cuts into it: a bay or an apse bulges out of one wall (best round a window), an alcove is a recess in a wall, "chamfer" cuts the four corners only where none of the things stands; "rect" keeps it a box. Pick what the place really would have: a cathedral is vaulted, a barn is pitched and beamed, a modern flat is flat.
A real room has character in its build: unless the description says modern, minimal or bare, give it at least two of a wainscot (panelling_m 0.8 to 1.2), a beamed or coffered ceiling, a bay, alcove or apse, and divided window panes. Keep the walls a middle tone or lighter (not near black), even at night, so that lamplight has something to fall on; let the dark come from the light, not the paint. The time of day is the one the description gives (night if it says night, evening is dusk). Candles and fires are 1800 to 2000 K, old bulbs 2400 to 2700, modern warm LEDs 3000, daylight lamps 5000 and over."""

OUTDOOR = """You are the production designer of an outdoor 3D place that will be built and walked through. The place:
"{sentence}"
{more}
The ground is {S:.0f} m square, its middle at (0, 0); x runs right, y away from the visitor, who starts at (0, -5) looking at the middle. The place itself has its real size in the middle of the ground (a courtyard or a plaza 10 to 20 m across, a clearing 15 m, a street 8 m wide), and what lies beyond it is the surroundings: keep the things of the place close together, as they really are. Answer with JSON only:
{{"mood": "a few words",
 "terrain": {{"relief_m": 0 to 4, "water_level_m": null or a height for a pond or shore,
   "layers": {{"ground": M, "path": M, "accent": M}},
   "foliage": [{{"what": "a kind of tree, bush, grass or flower", "count": 1 to 60, "height_m": metres}}]}},
 "things": [{{"name": "...", "what": "how it looks: material, colour, age", "size": [width, depth, height], "at": [x, y], "yaw": degrees, "said": true if the description names it, "lamp": null or {{"kelvin": 1800 to 6500, "watts": 5 to 400}}}}],
 "atmosphere": {{"time_of_day": "day" | "golden_hour" | "dusk" | "night", "outside": "the far horizon all round: hills, a city, the sea, forest", "fog_density": 0 to 0.03, "fog_color": "#hex"}}}}
where each M is {{"prompt": "the surface as a materials artist names it", "scale_m": metres a tile covers, "rgb": "#hex", "rough": 0 to 1, "metal": 0}}.
Give 12 to 30 things; everything the description names, in real sizes; keep a 2.5 m circle round (0, -5) and a way from it to the middle clear; if the place has a wall or fence round it, the visitor starts inside it. Trees and plants go in "foliage", unless they stand in a pot or tub: then the pot with its plant is a thing. Walls, fences, hedges, steps, kerbs and paved areas are things too: give each straight run its own entry ("low stone wall, north side"), its length as the width, and "plain": true, with "what" saying its material; it is built as a clean form in that material. Name copies the same ("wooden bench", not "bench 1", "bench 2"). Only what the description asks for or what such a place really has: no vents, pipes or signs unless asked."""

def hexrgb(h, d=(0.6, 0.6, 0.6)):
    try: h = h.strip().lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    except Exception: return list(d)

def num(v, d, lo=None, hi=None):
    try: v = float(v)
    except (TypeError, ValueError): return d
    if not math.isfinite(v): return d
    if lo is not None: v = max(lo, v)
    if hi is not None: v = min(hi, v)
    return v

def mat(name, m, fallback):
    m = m if isinstance(m, dict) else {}
    return ir.PBRMaterial(name=name, prompt=str(m.get("prompt") or fallback)[:200], scale_m=num(m.get("scale_m"), 1.5, 0.2, 6), rgb=hexrgb(m.get("rgb", "")), rough=num(m.get("rough"), 0.6, 0.02, 1), metal=num(m.get("metal"), 0.0, 0, 1))

def ask(prompt, tokens=2600):
    import chat, tools as C
    with chat.Chat() as c:
        for t in range(2):
            a = c.say([{"role": "user", "content": prompt + ("" if not t else "\nAnswer with the JSON object only.")}], max_tokens=tokens, temperature=0.2 if not t else 0.0)
            try: return C.json_in(a["text"])
            except (ValueError, json.JSONDecodeError): continue
    return {}

def sun_of(hour, toward=(0.0, 1.0)):
    """Where the sun (or by night the moon) is: `toward` is the way from the room out through its windows, so its light comes in."""
    h, w, k = SUN.get(hour, SUN["day"]); a = math.radians(h); x, y = toward; n = math.hypot(x, y) or 1.0
    x, y = x / n + 0.45 * -y / n, y / n + 0.45 * x / n; n = math.hypot(x, y)
    return [x / n * math.cos(a), y / n * math.cos(a), math.sin(a)], w, k

# ---------------------------------------------------------------------------------------------- indoors
WALL_EDGE = {"front": 0, "right": 1, "back": 2, "left": 3}

def footprint(W, D, shape, things):
    """The outline, anticlockwise from above, starting at the front-left corner: the room's rectangle, with a bay, an
    alcove or an apse added on one wall, or its corners cut where nothing stands. -> (points, the index of each wall's edge)"""
    x0, x1, y0, y1 = -W / 2, W / 2, -D / 2, D / 2; kind = shape.get("kind", "rect"); wall = shape.get("wall", "back")
    w = num(shape.get("width"), min(W, D) * 0.45, 1.2, (W if wall in ("front", "back") else D) - 0.8); dp = num(shape.get("depth"), 0.9, 0.5, 2.5)
    corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    if kind == "chamfer":
        c = 0.7
        free = lambda cx, cy: not any(abs(t.position[0] - cx) < c + 0.3 + t.size_m[0] / 2 and abs(t.position[1] - cy) < c + 0.3 + t.size_m[1] / 2 for t in things if t.on_surface in ("floor", "on"))
        if all(free(*p) for p in corners):
            pts = [[x0 + c, y0], [x1 - c, y0], [x1, y0 + c], [x1, y1 - c], [x1 - c, y1], [x0 + c, y1], [x0, y1 - c], [x0, y0 + c]]
            return pts, {"front": 0, "right": 2, "back": 4, "left": 6}
        kind = "rect"
    if kind not in ("bay", "alcove", "apse"): return corners, dict(WALL_EDGE)
    e = WALL_EDGE.get(wall, 2); a, b = corners[e], corners[(e + 1) % 4]
    L = math.dist(a, b); dx, dy = (b[0] - a[0]) / L, (b[1] - a[1]) / L; ox, oy = dy, -dx          # along the wall, and outward
    s, t = (L - w) / 2, (L + w) / 2; p = lambda u, o: [a[0] + dx * u + ox * o, a[1] + dy * u + oy * o]
    if kind == "bay": add = [p(s, 0), p(s + dp * 0.6, dp), p(t - dp * 0.6, dp), p(t, 0)]
    elif kind == "alcove": add = [p(s, 0), p(s, dp), p(t, dp), p(t, 0)]
    else: add = [p(s, 0)] + [p((s + t) / 2 - math.cos(q) * w / 2, math.sin(q) * min(dp, w / 2)) for q in [math.pi * k / 8 for k in range(1, 8)]] + [p(t, 0)]
    pts = corners[:e + 1] + add + corners[e + 1:]; idx = {}
    for name, k in WALL_EDGE.items(): idx[name] = k if k < e else (k + len(add) if k > e else k)
    return pts, idx

def on_outline(pts, W, D, wall, along):
    """An opening the layout put on a wall of its rectangle (`along`: metres from the wall's start, x for front and back,
    depth for left and right) -> (edge of the outline, metres along it). One that falls in the mouth of a bay or apse goes
    on the middle of the far face that replaced that stretch of wall."""
    q = {"front": (along - W / 2, -D / 2), "back": (along - W / 2, D / 2), "left": (-W / 2, along - D / 2), "right": (W / 2, along - D / 2)}.get(wall, (0.0, D / 2))
    n = len(pts); best = None
    for e in range(n):
        a, b = pts[e], pts[(e + 1) % n]; L = math.dist(a, b) or 1e-9; t = ((q[0] - a[0]) * (b[0] - a[0]) + (q[1] - a[1]) * (b[1] - a[1])) / L
        dd = math.dist(q, (a[0] + (b[0] - a[0]) * t / L, a[1] + (b[1] - a[1]) * t / L)) if 0 <= t <= L else 1e9
        if dd < 1e-3: return e, t
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2); far = math.dist(q, mid)
        if best is None or far < best[0]: best = (far, e, L / 2)
    return best[1], best[2]

def indoor(d, sentence):
    L = json.load(open(os.path.join(d, "layout.json"))); W, H, D = L["room"]
    more = ""
    if os.path.exists(os.path.join(d, "blender", "enriched.txt")): more = "In full: " + open(os.path.join(d, "blender", "enriched.txt")).read().strip()[:1400]
    assets = []; lamps = []
    for i, o in enumerate(L["objects"]):
        c = o["prims"][0]["c"]; w, h, dp = (float(v) for v in o["size"]); name = o["name"]; low = name.lower()
        a = ir.AssetInstance(id=f"o{i:03d}", name=name, source="", prompt=(name + (", " + o["look"] if o.get("look") else ""))[:220], position=[c[0] - W / 2, c[2] - D / 2, c[1] - h / 2],
                             yaw_deg=180.0 - math.degrees(float(o.get("yaw") or 0.0)), size_m=[w, dp, h], on_surface={"floor": "floor", "on": "on", "wall": "wall", "ceiling": "ceiling"}.get(o.get("attach"), "floor"),
                             kind=o.get("kind") or re.sub(r"\W+", "_", low), said=bool(o.get("said")))
        assets.append(a)
        if LAMP.search(low) and not NOT_LAMP.search(low): lamps.append(a)
    listing = "; ".join(f"{a.name} ({a.size_m[0]:.2f} x {a.size_m[1]:.2f} x {a.size_m[2]:.2f} m, {a.on_surface})" for a in assets[:60])
    ops = ", ".join(f"{o['kind']} on the {o['wall']} wall {o['width']:.1f} m wide" for o in L.get("openings", [])) or "none"
    j = ask(ARCHITECT.format(sentence=sentence, more=more, W=W, D=D, H=H, openings=ops, things=listing))
    pts, edge = footprint(W, D, j.get("shape") if isinstance(j.get("shape"), dict) else {}, assets)
    arch = j.get("window_arch") if j.get("window_arch") in ("flat", "round", "pointed") else "flat"
    mul = j.get("mullions") if isinstance(j.get("mullions"), list) and len(j["mullions"]) == 2 else [2, 2]
    openings = []
    for o in L.get("openings", []):
        e, u = on_outline(pts, W, D, o["wall"], float(o.get("along", 1.0)))
        openings.append(ir.Opening(kind=o["kind"], wall_index=e, u_centre=u, sill_m=float(o.get("sill", 0.0)), width_m=float(o.get("width", 1.0)),
                                   height_m=float(o.get("height", 1.5)), arch=arch if o["kind"] == "window" else "flat", mullions=[int(num(mul[0], 2, 1, 6)), int(num(mul[1], 2, 1, 6))] if o["kind"] == "window" else [1, 1]))
    M = j.get("materials") if isinstance(j.get("materials"), dict) else {}
    mats = {k: mat(k, M.get(k), f) for k, f in (("wall", "painted plaster wall, matte, off-white"), ("floor", "oak floorboards, satin finish"), ("ceiling", "smooth white painted plaster ceiling"), ("trim", "painted wooden trim, satin, white"))}
    mats["glass"] = ir.PBRMaterial(name="glass", prompt="clear window glass", rgb=[0.95, 0.97, 0.97], rough=0.03, transmission=0.95, ior=1.52)
    room = ir.RoomShell(footprint=[[round(x, 4), round(y, 4)] for x, y in pts], height_m=H, ceiling=j.get("ceiling") if j.get("ceiling") in ("flat", "beamed", "vaulted", "coffered", "pitched") else "flat",
                        openings=openings, panelling_m=num(j.get("panelling_m"), 0.0, 0.0, min(1.3, H / 2)))
    atmo = atmosphere(j.get("atmosphere"), L.get("mood", {}).get("time"), next(((o["wall"]) for o in L.get("openings", []) if o["kind"] == "window"), "back"), indoor=True)
    lights = lamp_lights(lamps, j.get("lamps") if isinstance(j.get("lamps"), dict) else {}, atmo.time_of_day)
    sources(assets)
    return ir.SceneIR(sentence=sentence, setting="indoor", mood=str(j.get("mood") or "")[:120], room=room, materials=mats, assets=assets, lights=lights, atmosphere=atmo)

def atmosphere(a, hour, window_wall="back", indoor=True):
    a = a if isinstance(a, dict) else {}; t = a.get("time_of_day") if a.get("time_of_day") in HOURS else ({"day": "day", "dusk": "dusk", "night": "night"}.get(hour, "day"))
    out = {"back": (0, 1), "front": (0, -1), "left": (-1, 0), "right": (1, 0)}.get(window_wall, (0, 1)) if indoor else (1.0, -0.35)   # outdoors from the side of the start view: raking shadows across it, not the sun in the visitor's eyes
    sd, sw, _ = sun_of(t, out)
    return ir.Atmosphere(time_of_day=t, sky_prompt=str(a.get("outside") or "")[:300], sun_dir=sd, sun_w=sw, fog_density=num(a.get("fog_density"), 0.004 if indoor else 0.01, 0, 0.03),
                         fog_color=hexrgb(a.get("fog_color", ""), (0.78, 0.82, 0.9)))

def lamp_lights(lamps, spec, hour):
    out = []
    for a in lamps:
        s = next((v for k, v in spec.items() if isinstance(v, dict) and (k.lower() == a.name.lower() or k.lower() in a.name.lower())), {})
        fire = re.search(r"candle|fire|torch|brazier", a.name.lower()); k = num(s.get("kelvin"), 1900 if fire else 2700, 1500, 7000); w = num(s.get("watts"), 25 if fire else 60, 2, 400)
        x, y, z = a.position; hgt = a.size_m[2]
        z = z - 0.08 if a.on_surface == "ceiling" else z + hgt * (0.55 if fire else 0.85)
        out.append(ir.PhysicalLight(id=a.id + "_light", kind="point", position=[x, y, z], kelvin=k, power_w=w, radius_m=0.05 if fire else 0.09, of=a.id))
    return out

def sources(assets):
    """Who makes each thing. Cloth is draped; the CC0 library gives what it holds well; the HEROES most telling kinds
    left (named by the sentence first, then the largest) are made by the 3D model; the rest are plain bevelled forms."""
    kinds = {}
    for a in assets: kinds.setdefault(a.kind, []).append(a)
    left = []
    for k, group in kinds.items():
        a = group[0]; low = a.name.lower()
        if CLOTH.search(low): src = "cloth_sim"
        else:
            m = kitbash.match(a.name, a.size_m)
            big = max(a.size_m) >= 0.5 and a.said
            if m and (m["score"] >= 6.0 or (m["score"] >= 3.0 and not big)): src = "kitbash"
            else: src = ""; left.append((not a.said, -a.size_m[0] * a.size_m[1] * a.size_m[2] * len(group) ** 0.5, k))
            if m and src == "kitbash":
                for g in group: g.mesh_path = m["path"]
        for g in group: g.source = src
    left.sort()
    for n, (_, _, k) in enumerate(left):
        src = "hero_3d" if n < HEROES or kinds[k][0].said else ("foliage_card" if PLANT.search(kinds[k][0].name.lower()) else "plain")   # what the sentence names is always made properly
        if src != "hero_3d":
            m = kitbash.match(kinds[k][0].name, kinds[k][0].size_m, least=4.0 if not kinds[k][0].said else 3.0)      # a filler thing keeps its plain form unless the library has the real thing
            if m: src = "kitbash"; [setattr(g, "mesh_path", m["path"]) for g in kinds[k]]
        for g in kinds[k]: g.source = src

# ---------------------------------------------------------------------------------------------- outdoors
def outdoor(d, sentence):
    S = 40.0; more = ""
    if os.path.exists(os.path.join(d, "blender", "enriched.txt")): more = "In full: " + open(os.path.join(d, "blender", "enriched.txt")).read().strip()[:1400]
    j = ask(OUTDOOR.format(sentence=sentence, more=more, S=S), tokens=3600)
    t = j.get("terrain") if isinstance(j.get("terrain"), dict) else {}; lay = t.get("layers") if isinstance(t.get("layers"), dict) else {}
    mats = {k: mat(k, lay.get(k), f) for k, f in (("ground", "short meadow grass over soil"), ("path", "packed earth path with small stones"), ("accent", "moss and fallen leaves"))}
    fol = [{"what": str(f.get("what"))[:80], "count": int(num(f.get("count"), 8, 1, 80)), "height_m": num(f.get("height_m"), 2.0, 0.1, 30)} for f in (t.get("foliage") or []) if isinstance(f, dict) and f.get("what")]
    wet = re.search(r"\b(lake|pond|pool|river|stream|sea|ocean|shore|beach|harbour|harbor|bay|marsh|swamp|lagoon|canal)\b", sentence.lower())       # a fountain's water is its own; open water only when asked for
    terrain = ir.OutdoorTerrain(size_m=S, relief_m=num(t.get("relief_m"), 1.2, 0, 4), foliage=fol, water_level_m=None if (t.get("water_level_m") is None or not wet) else num(t.get("water_level_m"), None, -2, 3))
    assets, lamps = [], []
    for i, th in enumerate([x for x in (j.get("things") or []) if isinstance(x, dict) and x.get("name")][:40]):
        sz = th.get("size") if isinstance(th.get("size"), list) and len(th["size"]) == 3 else [1, 1, 1]; at = th.get("at") if isinstance(th.get("at"), list) and len(th["at"]) == 2 else [0, 0]
        a = ir.AssetInstance(id=f"t{i:03d}", name=str(th["name"])[:40], source="", prompt=(str(th["name"]) + ", " + str(th.get("what") or ""))[:220],
                             position=[num(at[0], 0, -S / 2 + 1, S / 2 - 1), num(at[1], 0, -S / 2 + 1, S / 2 - 1), 0.0], yaw_deg=num(th.get("yaw"), 0.0),
                             size_m=[num(v, 1.0, 0.05, 30) for v in sz], kind=re.sub(r"(_?\d+)+$", "", re.sub(r"\W+", "_", str(th["name"]).lower()).strip("_")) or "thing", said=bool(th.get("said")))   # "bench 3" is a bench: copies share one model
        if th.get("plain") or WALLISH.search(a.name.lower()):                 # a wall, a fence, steps: a clean form in its own material, never thicker than a wall is
            if WALLISH.search(a.name.lower()) and not re.search(r"\b(step|steps|paving|paved|terrace|platform)\b", a.name.lower()):
                w, dp, h = a.size_m; a.size_m = [max(w, dp), min(min(w, dp), 0.45), h]
                a.yaw_deg = 90.0 if abs(a.position[0]) > abs(a.position[1]) else 0.0      # a run on an east or west side goes north-south: decided by where it stands
            a.source = "plain"; a.material = a.kind
            mats.setdefault(a.kind, ir.PBRMaterial(name=a.kind, prompt=str(th.get("what") or th["name"])[:200], scale_m=1.5))
        assets.append(a)
        if isinstance(th.get("lamp"), dict): lamps.append((a, th["lamp"]))
    keep = {a.id for a in assets if a.source == "plain"}
    sources([a for a in assets if a.id not in keep])
    atmo = atmosphere(j.get("atmosphere"), None, indoor=False)
    lights = lamp_lights([a for a, _ in lamps], {a.name: s for a, s in lamps}, atmo.time_of_day)
    return ir.SceneIR(sentence=sentence, setting="outdoor", mood=str(j.get("mood") or "")[:120], terrain=terrain, materials=mats, assets=assets, lights=lights, atmosphere=atmo)

def main(kind, d, sentence):
    s = indoor(d, sentence) if kind == "indoor" else outdoor(d, sentence)
    os.makedirs(os.path.join(d, "pbr"), exist_ok=True); ir.save(s, os.path.join(d, "pbr", "scene_ir.json"))
    n = {}
    for a in s.assets: n[a.source] = n.get(a.source, 0) + 1
    print(f"planned a {s.setting} place: {len(s.assets)} things ({', '.join(f'{v} {k}' for k, v in sorted(n.items()))}), {len(s.lights)} lamps, {s.atmosphere.time_of_day}" + (f", a {s.room.ceiling} ceiling, {len(s.room.footprint)}-sided outline" if s.room else ""), flush=True)

if __name__ == "__main__":
    main(sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3])
