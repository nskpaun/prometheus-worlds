"""A place built in Blender by the local world-plan model, through the Blender MCP.
  blender_world.py own WORLD_DIR "SENTENCE" ["STYLE"]    the model plans the whole place and builds every thing (thread/.venv)
  blender_world.py assemble WORLD_DIR                    a room made by the direct method's steps: its level goes into Blender
                                                         and the model rebuilds the plain furniture
  blender_world.py enrich WORLD_DIR "SENTENCE" ["STYLE"] the sentence filled out into a fuller description of the same place
Three processes: Blender with no window, serving the MCP addon (blender_host.py); the MCP server (ahujasid's "MCP for
Blender", a pinned package), which this file talks to as any MCP client would; and the model (build/chat.py), which is
shown the MCP's tools and answers with calls to them. Each step of the model's code is run, the scene is checked
(blender_kit.py check), and what the code printed, any error and what the check found go back to the model. It ends when
the model says it is done, or when BLENDER_MINUTES (default 10) are up. The scene is left in WORLD_DIR/blender/scene.blend.
Nothing leaves this Mac: the MCP package can send prompts, code and pictures to its makers and reach online asset and
generation services, so its telemetry is switched off AND both of its processes run with the network closed to all but
this Mac (sandbox-exec). Of its tools the model is given only the two that work on the scene."""
import sys, os, re, json, math, time, shutil, socket, subprocess, asyncio
HERE = os.path.dirname(os.path.abspath(__file__)); BUILD = os.path.join(HERE, "build")
sys.path.insert(0, BUILD)
import chat

TOOLS = ("execute_blender_code", "get_scene_info")
LOCAL_ONLY = '(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))(allow network-outbound (remote unix-socket))'
STEP_TOKENS = 1800        # a reply that runs past this is not run
LOOKS = os.environ.get("BLENDER_LOOK") != "0"        # the model is shown a picture of each thing it builds, and mends what looks wrong

ENRICH = """You are a set designer. Someone gives a short description of a place that will be built in 3D and walked through. Write the fuller description a builder needs, in plain prose, 150 to 220 words, no lists and no headings.

Keep everything the short description names, exactly as asked: never drop, swap or contradict a thing, a position, a time of day or a mood it gives.
Then add what it leaves out, choosing one clear character for the place and holding to it:
- whose place it is and what is done there, shown by what lies about;
- the room itself: its size and shape, where the door and windows are, what the walls, floor and ceiling are made of and their colours;
- each main piece of furniture: its material, colour, age and a telling detail of how it is made;
- the small things on the surfaces, on the walls and on the floor that make it lived in, and where each one is;
- the light: the time of day, each lamp and where it stands, how warm or cool, where the shadows fall.
Furnish it as a place of its kind really is: a shop or cafe has seats and tables for many, a kitchen has its appliances, a workshop its tools and stock. Say how many of each.
Give every colour and material by its plain name, chosen to suit THIS place. Only things that can be seen and built: no smells, sounds, people, animals, flicker, or remarks on what it all means."""

def enriched(model, sentence, style):
    """The sentence made into a fuller description of the same place (ENRICH). "" when the model's answer cannot be used."""
    a = model.say([{"role": "system", "content": ENRICH}, {"role": "user", "content": sentence + (f"\nIts look: {style}" if style else "")}], max_tokens=600, temperature=0.5)
    text = " ".join(a["text"].split())
    return text if 60 <= len(text.split()) <= 400 else ""

PLAN = """You plan a 3D place that will be built in Blender and walked through. Reply with ONE JSON object and nothing else.

{"setting": "indoor" or "outdoor",
 "room": {"width": metres along x, "depth": metres along y, "height": metres},          (indoor only; a home room is 3 to 7 m, 2.5 to 3 m high)
 "doors": [{"wall": "south", "offset": 1.2, "width": 0.9, "height": 2.05}],
 "windows": [{"wall": "north", "offset": 0.0, "sill": 0.9, "width": 1.6, "height": 1.3}],
 "wall": "#hex", "floor": "#hex", "ceiling": "#hex",                                    (outdoors: "ground": "#hex")
 "sky": {"color": "#hex", "strength": 0.0 to 1.5, "sun": 0 to 5},                       (day: strength 1, sun 3. evening: strength 0.15, sun 0. night: 0.03)
 "materials": [{"name": "oak", "color": "#8a5a2b", "roughness": 0.5, "metallic": 0.0}],  (6 to 12, the palette of the place)
 "things": [{"name": "desk", "what": "...", "size": [width, depth, height], "where": {...}, "light": 0}]}

Walls are "north" (+y), "south" (-y), "east" (+x), "west" (-x). An offset is in metres along the wall from its middle. The room is centred on x = 0, y = 0.
"where" is ONE of:
  {"wall": "west", "offset": 0.5}                    stands on the floor with its back against that wall
  {"x": 0.5, "y": -0.8, "rot": 30}                    stands free on the floor; rot in degrees, 0 = its front faces south
  {"on": "desk", "dx": 0.3, "dy": 0.0, "rot": 0}     stands on top of another thing, dx dy from that thing's middle
  {"hang": "north", "offset": 1.0, "height": 1.4}    hangs on a wall (picture, shelf, clock, curtain, mirror); height = its underside above the floor
  {"ceiling": true, "x": 0, "y": 0}                  hangs from the ceiling
Outdoors only the "x, y, rot" and "on" forms exist.

Rules
- 18 to 28 things: every thing the description names, then what such a place really has, down to the small things that make it lived in (books, cups, plants, pictures, cushions, a rug, curtains, a waste bin).
- "name" is short, lowercase, unique, with underscores. "what" is one or two sentences that say what it is MADE OF, part by part, with materials from the palette: "a writing desk: a thick oak top with rounded edges, four square tapered legs, a drawer with a brass knob under the top".
- "size" is the true size in metres, width (along its front) x depth x height. A rug or a picture is thin in one of them.
- Things must not overlap: keep doors and windows clear, leave room to walk, spread things round the room. A thing listed "on" another comes after it in the list.
- Doors and windows are NOT things: they go in "doors" and "windows". Nor are their parts, or a part of another thing (a door knob, a window sill, a rug's tassels, a lamp's shade). A bookcase's books and a bed's pillows are part of that thing: say so in its "what", do not list them apart.
- Every lamp is a thing with "light": its watts (40 to 150). Give the place at least one, unless it is outdoors by day. Other things have "light": 0."""

THING = """You build ONE thing in Blender by calling execute_blender_code once. The thing is made of several simple shapes, as the real thing is made of parts.

Where to build it: standing on the origin. Its middle is at x = 0, y = 0; its underside is at z = 0; z is up; its FRONT faces -y (so its back is at +y); metres.
You do not place it in the room: that is done for you afterwards. Do not make a room, floor, walls or lights.

Helpers (already imported). `at` is always where a shape STANDS: the middle of its underside.
mat(name, color="#rrggbb", roughness=0.6, metallic=0.0, emission=0.0, glass=False)   a material; the place's own materials can be used by name, as "oak"
box(name, (size_x, size_y, size_z), at, material, rot=0, bevel=0.0)       rot: degrees about z, or (x, y, z) degrees; bevel rounds the edges, in metres
cyl(name, radius, height, at, material, rot=0)                            upright; rot=(90, 0, 0) lays it along y, rot=(0, 90, 0) along x
cone(name, radius_bottom, radius_top, height, at, material, rot=0)
sphere(name, radius, at, material, scale=(1, 1, 1))                       rests on `at`
lathe(name, [(radius, z), ...], at, material)                             a turned shape from its profile: vase, bottle, bowl, lamp base, pot
extrude(name, [(x, y), ...], height, at, material, rot=0)                 a flat outline raised to a height
tube(name, [(x, y, z), ...], radius, material)                            a round bar through points
You may also use bpy, bmesh, mathutils and math directly, but no bpy.ops.

Rules
- 6 to 40 shapes. Use loops for repeated parts (legs, slats, books, leaves, drawers). Give neighbouring parts slightly different colours where the real thing has them (books, cushions).
- Keep to the size asked for: it is width (x) by depth (y) by height (z).
- Every part must touch another part: nothing floats.
- Never name a variable after a helper (box, cyl, cone, sphere, lathe, extrude, tube, mat).
- Under 60 lines. No comments longer than a few words. Print nothing.

An example, for "side_table: a small round oak table with a lower shelf, three turned legs, and a white mug on it", size 0.5 x 0.5 x 0.62:

<tool_call>
<function=execute_blender_code>
<parameter=code>
import math
oak = mat("oak", "#8a5a2b", 0.5)
cyl("top", 0.25, 0.03, (0, 0, 0.56), oak)
cyl("shelf", 0.17, 0.02, (0, 0, 0.2), oak)
for i in range(3):
    a = math.radians(90 + i * 120)
    lathe(f"leg_{i}", [(0.012, 0), (0.02, 0.1), (0.014, 0.3), (0.022, 0.5), (0.018, 0.56)], (0.18 * math.cos(a), 0.18 * math.sin(a), 0), oak)
white = mat("white_glaze", "#f1efe8", 0.25)
lathe("mug", [(0.03, 0), (0.035, 0.005), (0.036, 0.09), (0.032, 0.09), (0.03, 0.01), (0, 0.01)], (0.08, -0.05, 0.59), white)
tube("mug_handle", [(0.115, -0.05, 0.66), (0.14, -0.05, 0.655), (0.14, -0.05, 0.625), (0.115, -0.05, 0.615)], 0.005, white)
</parameter>
</function>
</tool_call>

Another, for "armchair: a deep brown leather armchair, a thick seat cushion on a low base, a padded back, two padded arms, four short dark wooden feet", size 0.85 x 0.85 x 0.9:

<tool_call>
<function=execute_blender_code>
<parameter=code>
leather = mat("brown_leather", "#6b4226", 0.55)
dark = mat("dark_wood", "#3b2a1c", 0.5)
box("base", (0.85, 0.8, 0.26), (0, 0.0, 0.1), leather, bevel=0.03)
box("seat_cushion", (0.6, 0.6, 0.13), (0, -0.07, 0.36), leather, bevel=0.05)
box("back", (0.85, 0.2, 0.54), (0, 0.325, 0.36), leather, bevel=0.06)
for sx in (-1, 1):
    box(f"arm_{sx}", (0.125, 0.65, 0.27), (sx * 0.3625, -0.075, 0.36), leather, bevel=0.05)
    for sy in (-1, 1):
        cone(f"foot_{sx}_{sy}", 0.03, 0.04, 0.1, (sx * 0.36, sy * 0.34, 0), dark)
</parameter>
</function>
</tool_call>

Another, for "bookcase: a tall oak bookcase, two sides, a back panel, six shelves, the upper five full of books of many colours and heights", size 1.0 x 0.3 x 2.0:

<tool_call>
<function=execute_blender_code>
<parameter=code>
import random
random.seed(3)
for sx in (-1, 1):
    box(f"side_{sx}", (0.03, 0.3, 2.0), (sx * 0.485, 0, 0), "oak")
box("back_panel", (0.94, 0.015, 2.0), (0, 0.1425, 0), "oak")
for i in range(6):
    z = 0.05 + i * 0.38
    box(f"shelf_{i}", (0.94, 0.28, 0.025), (0, -0.005, z), "oak")
    x = -0.46
    while i < 5 and x < 0.4:
        w, h, col = random.uniform(0.02, 0.05), random.uniform(0.2, 0.3), random.choice(["#7a2e2e", "#2e4a7a", "#3f6b45", "#c9a24b", "#5a3d6b", "#d8d2c4"])
        box(f"book_{i}", (w, random.uniform(0.16, 0.22), h), (x + w / 2, 0.02, z + 0.025), mat("book_" + col[1:], col, 0.7))
        x += w + 0.003
</parameter>
</function>
</tool_call>

Work out where each part goes before writing it: a part's `at` z is the height of whatever it rests on, and parts that meet must share a face."""

WHERE_KEYS = ("wall", "hang", "on", "ceiling", "x", "y", "rot", "offset", "height", "dx", "dy")
SIDES = ("north", "south", "east", "west")

def read_plan(text):
    """The model's plan, made safe to build from: numbers where numbers belong, sizes within reason, unique names, every
    thing with a place, and things that stand on others after them. Raises ValueError when it cannot be used at all."""
    j = json.loads(text[text.index("{"):text.rindex("}") + 1])
    def num(v, d, lo, hi):
        try: return min(max(float(v), lo), hi)
        except (TypeError, ValueError): return d
    hexc = lambda v, d: v if isinstance(v, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", v) else d
    out = {"setting": "outdoor" if j.get("setting") == "outdoor" else "indoor", "doors": [], "windows": [], "materials": [], "things": []}
    indoor = out["setting"] == "indoor"; r = j.get("room") if isinstance(j.get("room"), dict) else {}
    if indoor: out["room"] = W, D, H = [num(r.get("width"), 5, 2.5, 20), num(r.get("depth"), 4, 2.5, 20), num(r.get("height"), 2.7, 2.2, 8)]
    for o in (j.get("doors") or []) if indoor else []:
        if isinstance(o, dict) and o.get("wall") in SIDES: out["doors"].append((o["wall"], num(o.get("offset"), 0, -10, 10), num(o.get("width"), 0.9, 0.6, 3), num(o.get("height"), 2.05, 1.8, H - 0.1)))
    for o in (j.get("windows") or []) if indoor else []:
        if isinstance(o, dict) and o.get("wall") in SIDES: out["windows"].append((o["wall"], num(o.get("offset"), 0, -10, 10), num(o.get("sill"), 0.9, 0, H - 0.6), num(o.get("width"), 1.4, 0.4, 8), num(o.get("height"), 1.2, 0.3, H)))
    for k, d in (("wall", "#d9d4c7"), ("floor", "#8b6f4e"), ("ceiling", "#f2f0ea"), ("ground", "#5f7a45")): out[k] = hexc(j.get(k), d)
    sk = j.get("sky") if isinstance(j.get("sky"), dict) else None
    out["sky"] = {"color": hexc(sk.get("color"), "#9cc4ee"), "strength": num(sk.get("strength"), 1, 0, 3), "sun": num(sk.get("sun"), 0, 0, 8)} if sk else ({"color": "#9cc4ee", "strength": 1.0, "sun": 3.0} if not indoor or out["windows"] else None)
    for m in j.get("materials") or []:
        if isinstance(m, dict) and isinstance(m.get("name"), str): out["materials"].append({"name": re.sub(r"\W+", "_", m["name"].strip().lower())[:30], "color": hexc(m.get("color"), "#808080"), "roughness": num(m.get("roughness"), 0.6, 0, 1), "metallic": num(m.get("metallic"), 0, 0, 1)})
    seen = set()
    for t in j.get("things") or []:
        if not isinstance(t, dict) or not str(t.get("what") or t.get("name") or "").strip(): continue
        if re.fullmatch(r"(the_)?(window|door|wall|floor|ceiling|room)s?(_(\d+|knob|handle|frame|sill|pane|glass))?", re.sub(r"\W+", "_", str(t.get("name") or "").strip().lower())): continue      # the shell is made from the plan, not built as a thing
        name = re.sub(r"\W+", "_", str(t.get("name") or t["what"]).strip().lower()).strip("_")[:30] or "thing"; base, k = name, 2
        while name in seen: name, k = f"{base}_{k}", k + 1
        seen.add(name); sz = t.get("size") if isinstance(t.get("size"), list) and len(t["size"]) == 3 else [0.5, 0.5, 0.5]
        w = {k: v for k, v in (t.get("where") if isinstance(t.get("where"), dict) else {}).items() if k in WHERE_KEYS}
        if not indoor: w = {k: v for k, v in w.items() if k in ("on", "x", "y", "rot", "dx", "dy")}
        for k in ("wall", "hang"):
            if k in w and w[k] not in SIDES: del w[k]
        out["things"].append({"name": name, "what": str(t.get("what") or name).strip()[:400], "size": [num(v, 0.5, 0.005, 12) for v in sz], "where": w, "light": num(t.get("light"), 0, 0, 2000)})
    if not out["things"]: raise ValueError("the plan lists no things")
    names = {t["name"] for t in out["things"]}; done, order, rest = set(), [], list(out["things"])
    for t in rest:
        if t["where"].get("on") not in names or t["where"].get("on") == t["name"]: t["where"].pop("on", None)
    while rest:                                              # what stands on a thing comes after it; a ring of such is cut
        ready = [t for t in rest if t["where"].get("on") is None or t["where"]["on"] in done] or [dict(rest[0], where={})]
        if ready[0] not in rest: rest[0] = ready[0]
        for t in ready: order.append(t); done.add(t["name"]); rest.remove(t)
    out["things"] = order; return out

def room_code(plan):
    """The code that makes the shell of the place from the plan: its palette, then the room (or ground) and the sky."""
    c = [f"mat({m['name']!r}, {m['color']!r}, {m['roughness']}, {m['metallic']})" for m in plan["materials"]]
    if plan["setting"] == "indoor":
        c += [f"mat('wall', {plan['wall']!r}, 0.85)", f"mat('floor', {plan['floor']!r}, 0.55)", f"mat('ceiling', {plan['ceiling']!r}, 0.9)"]
        W, D, H = plan["room"]; c.append(f"room({W}, {D}, {H}, wall='wall', floor='floor', ceiling='ceiling', doors={plan['doors']!r}, windows={plan['windows']!r})")
    else:                                                    # ground as far as the things reach and 8 m more: a larger one would take the baked picture from them
        reach = max([max(abs(float(t["where"].get(k) or 0)) for k in ("x", "y")) + max(t["size"][:2]) / 2 for t in plan["things"] if isinstance(t["where"].get("x", 0), (int, float)) and isinstance(t["where"].get("y", 0), (int, float))] or [4.0])
        c += [f"mat('ground', {plan['ground']!r}, 0.9)", f"ground({min(60.0, 2 * reach + 16):.1f}, 'ground')"]
    if plan["sky"]: c.append(f"sky({plan['sky']['color']!r}, {plan['sky']['strength']}, {plan['sky']['sun']})")
    return "\n".join(c)

def thing_ask(t, plan):
    w, d, h = t["size"]; pal = ", ".join(f"{m['name']} {m['color']}" for m in plan["materials"])
    return f"Build this thing.\nname: {t['name']}\nwhat: {t['what']}\nsize: {w:g} wide (x) by {d:g} deep (y) by {h:g} high (z), in metres\nMaterials of the place, usable by name: {pal or 'none yet: make your own with mat()'}"

def find_blender():
    for p in (os.environ.get("BLENDER"), "/Applications/Blender.app/Contents/MacOS/Blender", os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender"), shutil.which("blender")):
        if p and os.path.exists(p): return p
    raise SystemExit("Blender is not installed. Get it from blender.org (or: brew install --cask blender), then try again.")

def local_only(cmd):
    """The command with the network closed to everything but this Mac."""
    return ["/usr/bin/sandbox-exec", "-p", LOCAL_ONLY] + cmd if os.path.exists("/usr/bin/sandbox-exec") else cmd

def addon_path():
    import blender_mcp
    return os.path.join(os.path.dirname(blender_mcp.__file__), "bundled", "addon.py")

def free_port():
    with socket.socket() as s: s.bind(("127.0.0.1", 0)); return s.getsockname()[1]

class Host:
    """Blender with no window, serving the MCP addon on a port of this Mac."""
    def __init__(self, log):
        self.port = free_port(); self.log = open(log, "w")
        self.p = subprocess.Popen(local_only([find_blender(), "-b", "--factory-startup", "--python", os.path.join(HERE, "blender_host.py"), "--", addon_path(), str(self.port)]),
                                  stdout=self.log, stderr=subprocess.STDOUT, env={**os.environ, "BLENDERMCP_NO_UPDATE_CHECK": "1"})
        t = time.time()
        while "HOST READY" not in open(log).read():
            if self.p.poll() is not None or time.time() - t > 90: self.close(); raise SystemExit("Blender could not be started with the MCP addon (see blender/host.log in the world's folder).")
            time.sleep(0.3)
    def close(self):
        if self.p.poll() is None:
            self.p.terminate()
            try: self.p.wait(timeout=10)
            except subprocess.TimeoutExpired: self.p.kill()
        self.log.close()

def tool_specs(tools):
    """The MCP's tools as the model's chat template wants them. `user_prompt` is the MCP's telemetry field: left out."""
    out = []
    for t in tools:
        if t.name not in TOOLS: continue
        schema = json.loads(json.dumps(t.inputSchema)); schema.get("properties", {}).pop("user_prompt", None)
        schema["required"] = [r for r in schema.get("required", []) if r != "user_prompt"]
        out.append({"type": "function", "function": {"name": t.name, "description": (t.description or "").strip().split("\n\n")[0], "parameters": schema}})
    return out

def wrap(code):
    """A step of the model's code as it is sent: the helpers in reach, and whatever it made taken out again if it fails."""
    return ("from kit import *\nimport kit as _kit\n_kit._begin()\n_src = " + repr(code) + "\ntry:\n    exec(compile(_src, '<step>', 'exec'), globals())\n"
            "except BaseException:\n    _kit._rollback()\n    raise\n")

HELPERS = ("mat", "box", "cyl", "cone", "sphere", "lathe", "extrude", "tube", "room", "ground", "sky", "lamp", "group", "place", "delete", "check", "things")

def explain(text, code):
    """An error as the model can act on it: its last lines, the line of the step's code it points at, and the usual cause
    where one is known."""
    text = text.replace("\\n", "\n").replace('\\"', '"').strip().rstrip('"}').strip()
    lines = code.split("\n"); at = [int(n) for n in re.findall(r'File "<step>", line (\d+)', text)]
    last = [l for l in text.split("\n") if l.strip()][-1] if text else "Error"
    out = "Error: " + last[-400:] + (f"\nAt line {at[-1]} of your code: {lines[at[-1] - 1].strip()}" if at and 0 < at[-1] <= len(lines) else "")
    if "is not callable" in last:
        hid = sorted({h for h in HELPERS if re.search(rf"^\s*(?:for\s+)?{h}\s*(?:=[^=]|,|\s+in\s)", code, re.M)})
        out += "\nA helper's name was used for a variable" + (f" ({', '.join(hid)})" if hid else "") + ", which hides the helper: give variables other names (the_lamp, desk_box)."
    return out

def coerce(args, spec):
    """The model writes every argument as text: turn each into the type the tool's schema asks for."""
    props = spec["function"]["parameters"].get("properties", {}); out = {}
    for k, v in args.items():
        if k not in props: continue
        kind = props[k].get("type") or next((a.get("type") for a in props[k].get("anyOf", []) if a.get("type") != "null"), "string")
        try: out[k] = v if kind == "string" else json.loads(v)
        except ValueError: out[k] = v
    return out

LOOK = """This is a picture of a 3D model that is meant to be: {what}
It is built from a few simple shapes in plain colours, and that is fine: do not fault it for being blocky, or for lacking fine detail, texture, pattern or decoration.
Look at it as a carpenter would. Is it put together like the real thing: every main part there (a chair's seat, back and legs; a table's top; a lamp's shade), each part where it belongs and joined to the rest, the proportions believable?
Reply with ONE line and nothing else, either
GOOD
or
BAD: what is wrong, in one or two sentences that name the part and say where it should be or how big."""

async def make(model, specs, run, name, ask, what, note, snap=None):
    """One thing built by the model: its call run, an error sent back (three tries), then, where a picture can be taken
    (snap: a path), the model LOOKS at what it built and, if it says what is wrong with the shape, builds it once more
    with that in mind. -> (built?, seconds the model spent)"""
    msgs = [{"role": "system", "content": THING}, {"role": "user", "content": ask}]; spent = 0.0; looked = False; have = False; tries = 0
    while tries < 4:
        a = model.say(msgs, max_tokens=STEP_TOKENS, temperature=0.3 + 0.2 * tries, tools=specs); spent += a["s"]; tries += 1; calls = chat.tool_calls(a["text"])
        msgs.append({"role": "assistant", "content": a["text"].strip()}); code = next((c[1].get("code", "") for c in calls if c[0] == "execute_blender_code"), None)
        if a["finish"] == "length" or not code: fb = "That was cut off for length, so nothing was run: build it again from fewer, simpler shapes, under 60 lines." if a["finish"] == "length" else "Reply with one execute_blender_code call that builds the thing."
        else:
            if have: await run(f"import kit\nkit.delete({name!r})"); have = False      # the one it is mending
            out, bad = await run(wrap(code) + f"print('SIZE' + __import__('json').dumps(_kit._collect({name!r})))")
            if not bad and "SIZE" in out:
                have = True; note(thing=name, attempt=tries, code=code, size=json.loads(out[out.index("SIZE") + 4:].splitlines()[0]), s=a["s"], tokens=a["gen_tokens"])
                if not snap or looked or tries >= 4: break
                looked = True; _, bad = await run(f"import kit\nkit._snap({name!r}, {snap!r})")
                if bad or not os.path.exists(snap): break
                v = model.say([{"role": "user", "content": [{"type": "image", "path": snap}, {"type": "text", "text": LOOK.format(what=what)}]}], max_tokens=120, temperature=0.0); spent += v["s"]
                verdict = " ".join(v["text"].split()); note(thing=name, looked=verdict, s=v["s"])
                fault = re.match(r"\W*BAD\W+(.{12,})", verdict, re.I | re.S)
                if not fault: break                                              # GOOD, or an answer that names no fault
                verdict = fault.group(1)
                msgs.append({"role": "tool", "content": f"It ran. A picture of what you built was then looked at, and this is wrong with its shape: {verdict}\nBuild the whole thing again with that put right. Think where each part must be so that the parts join."}); continue
            if not bad: await run("import kit\nkit._rollback()")
            fb = explain(out, code); used = [h for h in HELPERS if re.search(rf"\b{h}\(", fb)]
            if used and "TypeError" in fb: fb += "\nIts arguments are: " + (await run(f"import kit\nprint(kit._usage({used[0]!r}))"))[0]
            fb += "\nNothing was kept. Send the whole code again, mended."
        note(thing=name, attempt=tries, code=code, error=fb, s=a["s"], tokens=a["gen_tokens"]); msgs.append({"role": "tool" if calls else "user", "content": fb})
    return have, spent

async def build(d, sentence, style, minutes):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    work = os.path.join(d, "blender"); os.makedirs(work, exist_ok=True); host = Host(os.path.join(work, "host.log")); record = open(os.path.join(work, "steps.jsonl"), "w")
    def note(**k): record.write(json.dumps(k) + "\n"); record.flush()
    env = {**os.environ, "BLENDER_HOST": "127.0.0.1", "BLENDER_PORT": str(host.port), "DISABLE_TELEMETRY": "true", "BLENDERMCP_NO_UPDATE_CHECK": "1"}
    server = local_only([sys.executable, "-m", "blender_mcp.server"])
    try:
        with open(os.path.join(work, "mcp.log"), "w") as errlog:
            async with stdio_client(StdioServerParameters(command=server[0], args=server[1:], env=env), errlog=errlog) as (r, w):
                async with ClientSession(r, w) as mcp:
                    await mcp.initialize(); specs = tool_specs((await mcp.list_tools()).tools); by_name = {s["function"]["name"]: s for s in specs}
                    if "execute_blender_code" not in by_name: raise SystemExit("The Blender MCP did not offer its code tool.")
                    async def run(code):
                        x = await mcp.call_tool("execute_blender_code", {"code": code}); out = "\n".join(c.text for c in x.content if c.type == "text")
                        return out.replace("Code executed successfully:", "").strip(), bool(x.isError) or out.lstrip().startswith("Error")
                    async def kit(expr):                      # one of the kit's own functions, its answer read back
                        out, bad = await run(f"import kit, json\nprint('KIT' + json.dumps({expr}))")
                        if bad or "KIT" not in out: raise RuntimeError(out[-300:])
                        return json.loads(out[out.index("KIT") + 3:])
                    await run("import bpy\nfor o in list(bpy.data.objects): bpy.data.objects.remove(o)")           # Blender's own cube, lamp and camera
                    t0 = time.time(); left = lambda: minutes * 60 - (time.time() - t0); built = []
                    with chat.Chat(vision=LOOKS) as model:
                        print(f"model loaded in {model.load_s} s; Blender is on port {host.port}", flush=True)
                        more = "" if os.environ.get("BLENDER_ENRICH") == "0" else enriched(model, sentence, style)
                        if more:
                            with open(os.path.join(work, "enriched.txt"), "w") as fh: fh.write(more + "\n")
                            print("the description, filled out: " + more, flush=True)
                        ask = f"The place: {sentence}" + (f"\nIts look: {style}" if style else "") + (f"\n\nThe same place described in full, to plan from. Where the two differ, the first is right:\n{more}" if more else ""); plan = None
                        for attempt in range(3):
                            a = model.say([{"role": "system", "content": PLAN}, {"role": "user", "content": ask}], max_tokens=5000, temperature=0.3 + 0.2 * attempt); note(plan=a["text"], s=a["s"])
                            try: plan = read_plan(a["text"]); break
                            except ValueError as e: print(f"the plan could not be read ({str(e)[:60]}): asking again", flush=True)
                        if plan is None: raise SystemExit("The model did not write a plan that could be read.")
                        json.dump(plan, open(os.path.join(work, "plan.json"), "w"), indent=1)
                        out, bad = await run("from kit import *\n" + room_code(plan))
                        if bad: raise SystemExit("The room could not be made: " + out[-300:])
                        print(f"plan: {plan['setting']}, " + (f"a room {plan['room'][0]:g} x {plan['room'][1]:g} x {plan['room'][2]:g} m, " if plan["setting"] == "indoor" else "") + f"{len(plan['things'])} things ({a['s']:.0f} s)", flush=True)
                        for i, t in enumerate(plan["things"]):
                            if left() <= 0: print(f"out of time: {len(plan['things']) - i} things left unbuilt", flush=True); break
                            ok, secs = await make(model, specs, run, t["name"], thing_ask(t, plan), f"{t['name'].replace('_', ' ')}: {t['what']}", note, snap=os.path.join(work, "look.png") if LOOKS else None)
                            if not ok: print(f"{i + 1}/{len(plan['things'])} {t['name']}: could not be built, left out", flush=True); continue
                            changed = await kit(f"kit._fit({t['name']!r}, {t['size']!r})"); spot = await kit(f"kit._put({t['name']!r}, {t['where']!r})")
                            if t["light"] > 0: await run(f"import kit\nkit._glow({t['name']!r}, {t['light']})")
                            built.append(t["name"]); print(f"{i + 1}/{len(plan['things'])} {spot}" + (f" (stretched along {', '.join(changed)} to the planned size)" if changed else "") + f"  [{secs:.0f} s]", flush=True)
                    if len(built) < 2: raise SystemExit("The model did not manage to build anything in Blender.")
                    lit = (await run("import bpy\nprint(any(o.type == 'LIGHT' and o.name != 'sun' for o in bpy.data.objects) or bpy.context.scene.get('sky', [0, 0, 0, 0])[3] > 0.5)"))[0]
                    if "True" not in lit: await run("from kit import *\nr = bpy.context.scene.get('room')\nif r: lamp('fill_light', (0, 0, r[2] - 0.25), 120)\nelse: sky()")         # never a place in the dark
                    problems = (await run("from kit import *\nprint(check())"))[0]
                    if problems: print("the check still finds:\n" + problems, flush=True)
                    out, bad = await run(f"import bpy\nbpy.ops.wm.save_as_mainfile(filepath={os.path.join(work, 'scene.blend')!r})")
                    if bad or not os.path.exists(os.path.join(work, "scene.blend")): raise SystemExit("Blender could not save the scene: " + out[-300:])
                    print(f"{len(built)} of {len(plan['things'])} things built in {(time.time() - t0) / 60:.1f} minutes", flush=True)
    finally: host.close(); record.close()

DAYLIGHT = {"day": ("#bcd7f5", 1.6, 4.0, 0.75), "dusk": ("#e8a878", 0.35, 0.9, 0.16), "night": ("#101a33", 0.03, 0.0, 0.5)}      # sky colour, its strength, the sun's, how high the sun stands
DAYLIGHT_SUN = {"day": "#fff1dc", "dusk": "#ffb070"}
REBUILT = ("table", "bookcase", "drawers", "crate", "cylinder")      # the plain forms the model builds anew: furniture, not flat or small things

FORM_WORDS = {"table": "a table", "bookcase": "a set of shelves, with every shelf FILLED by what it holds (for books: full rows of books of many heights, widths and colours, as in the bookcase example)", "drawers": "a cabinet with drawers or doors", "crate": "a box-shaped thing", "cylinder": "a round upright thing"}

def candidates(boxes, plan, surfaces):
    """The plain kinds of a planned room that the model is to build in Blender, most telling first: furniture forms at least
    0.35 m in some direction, the ones the sentence names before the rest, then by size. -> [{kind, name, look, form, size
    (x, y, z in Blender's frame), objects [index...], mats [{name, words, texture, px_per_m, rgb, cls, parts}]}]"""
    out = []; objs = boxes["objects"]
    for kind, pl in (plan.get("plain") or {}).items():
        idx = [i for i in pl.get("objects", []) if i < len(objs)]
        if pl.get("form") not in REBUILT or not idx: continue
        o = objs[idx[0]]; w, h, dp = (float(v) for v in o["size"])
        if max(w, h, dp) < 0.35 or o.get("part_of") is not None: continue
        mats = {}
        for part, m in ((surfaces.get("parts") or {}).get(kind, {}).get("parts") or {}).items():
            if m.get("mode") == "emit" or not m.get("material"): continue
            key = re.sub(r"\W+", "_", m["material"].lower()).strip("_")[:28] or part
            mats.setdefault(key, {"name": key, "words": m["material"], "texture": m.get("texture") if m.get("mode") == "swatch" else None, "px_per_m": m.get("px_per_m", 512.0), "rgb": m.get("rgb") or [128, 128, 128], "cls": m.get("cls") or "", "parts": []})["parts"].append(part)
        out.append({"kind": kind, "name": o["name"], "look": o.get("look") or "", "form": pl["form"], "size": [w, dp, h], "objects": idx, "mats": list(mats.values()), "said": any(objs[i].get("said") for i in idx)})
    out.sort(key=lambda c: (not c["said"], -c["size"][0] * c["size"][1] * c["size"][2] * len(c["objects"]) ** 0.5)); return out

def spot(o, room):
    """Where a layout object stands in Blender's frame: the middle of its underside, and its turn in degrees for a thing
    built with its front to -y. The layout's frame is x across, y up, z deep, a yaw of 0 facing +z."""
    c, h = o["prims"][0]["c"], o["size"][1]
    return [c[0] - room[0] / 2, c[2] - room[2] / 2, c[1] - h / 2], 180.0 - math.degrees(float(o.get("yaw") or 0.0))

def rebuild_ask(c):
    w, d, h = c["size"]; n = len(c["objects"])
    mats = "\n".join(f"  {m['name']}: {m['words']} (for its {', '.join(m['parts'])})" for m in c["mats"])
    return (f"Build this thing.\nname: {c['name']}\nwhat: {c['name']}" + (f", {c['look']}" if c["look"] else "") + f". In shape it is {FORM_WORDS.get(c['form'], 'a ' + c['form'])}.\n"
            f"size: {w:g} wide (x) by {d:g} deep (y) by {h:g} high (z), in metres. Fill that box exactly: other things stand on it and beside it.\n"
            + (f"Its own materials are ready made: use them by these names for the thing itself.\n{mats}\nWhat it HOLDS (books, bottles, boxes, folded cloth) is not made of these: give those their own colours with mat(), several different ones, and keep each row lower than the gap to the shelf above." if mats else "Make its materials with mat()."))

async def assemble(d, minutes):
    """A room made by the direct method's steps (its level exported to WORLD_DIR/blender/level/level.glb) -> the scene in
    Blender: the level brought in, then each plain piece of furniture built anew by the model through the MCP, to the
    size and place the layout gave it, in the materials the style gave it. What the model cannot build in three tries, or
    in the time, keeps the level's own plain shape."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    work = os.path.join(d, "blender"); os.makedirs(work, exist_ok=True)
    boxes = json.load(open(os.path.join(d, "boxes.json"))); plan = json.load(open(os.path.join(d, "shapes", "plan.json"))); surfaces = json.load(open(os.path.join(d, "materials", "surfaces.json")))
    room = [float(v) for v in boxes["room"]]; todo = candidates(boxes, plan, surfaces)
    host = Host(os.path.join(work, "host.log")); record = open(os.path.join(work, "steps.jsonl"), "w")
    def note(**k): record.write(json.dumps(k) + "\n"); record.flush()
    env = {**os.environ, "BLENDER_HOST": "127.0.0.1", "BLENDER_PORT": str(host.port), "DISABLE_TELEMETRY": "true", "BLENDERMCP_NO_UPDATE_CHECK": "1"}
    server = local_only([sys.executable, "-m", "blender_mcp.server"])
    try:
        with open(os.path.join(work, "mcp.log"), "w") as errlog:
            async with stdio_client(StdioServerParameters(command=server[0], args=server[1:], env=env), errlog=errlog) as (r, w):
                async with ClientSession(r, w) as mcp:
                    await mcp.initialize(); specs = tool_specs((await mcp.list_tools()).tools)
                    async def run(code):
                        x = await mcp.call_tool("execute_blender_code", {"code": code}); out = "\n".join(c.text for c in x.content if c.type == "text")
                        return out.replace("Code executed successfully:", "").strip(), bool(x.isError) or out.lstrip().startswith("Error")
                    async def kit(expr):
                        out, bad = await run(f"import kit, json\nprint('KIT' + json.dumps({expr}))")
                        if bad or "KIT" not in out: raise RuntimeError(out[-300:])
                        return json.loads(out[out.index("KIT") + 3:])
                    await run("import bpy\nfor o in list(bpy.data.objects): bpy.data.objects.remove(o)")
                    got = await kit(f"kit._import_level({os.path.join(work, 'level', 'level.glb')!r}, {[room[0], room[2], room[1]]!r})")
                    print(f"the level is in Blender: {got['room']} pieces of room, {got['things']} things, {got['lights']} lights; {len(todo)} kinds of furniture for the model to build", flush=True)
                    if got.get("windows"):                                   # daylight for the hour the style guide gave, the sun let in by the first window
                        try: hour = json.load(open(os.path.join(d, "materials", "style.json"))).get("time")
                        except (OSError, ValueError): hour = None
                        wall = next((o.get("wall") for o in boxes.get("openings") or [] if o.get("kind") == "window"), "back")
                        out_ = {"back": (0, 1), "front": (0, -1), "left": (-1, 0), "right": (1, 0)}.get(wall, (0, 1)); side = (-out_[1], out_[0])
                        colour, strength, sun, up = DAYLIGHT.get(hour, DAYLIGHT["day"]); sun_from = (out_[0] + 0.45 * side[0], out_[1] + 0.45 * side[1], up)
                        await run(f"from kit import *\nsky({colour!r}, {strength}, {sun}, {sun_from!r}, {DAYLIGHT_SUN.get(hour, '#fff1dc')!r})")
                    t0 = time.time(); left = lambda: minutes * 60 - (time.time() - t0); built = 0
                    if todo:
                        with chat.Chat(vision=LOOKS) as model:
                            for k, c in enumerate(todo):
                                if left() <= 0: print(f"out of time: {len(todo) - k} kinds keep their plain shape", flush=True); break
                                for m in c["mats"]: await run(f"import kit\nkit._texmat({m['name']!r}, {os.path.join(d, 'materials', m['texture']) if m['texture'] else ''!r}, {m['px_per_m']}, {m['rgb']!r}, {m['cls']!r})")
                                name = "built_" + re.sub(r"\W+", "_", c["kind"])
                                ok, secs = await make(model, specs, run, name, rebuild_ask(c), c["name"] + (f", {c['look']}" if c["look"] else ""), note, snap=os.path.join(work, "look.png") if LOOKS else None)
                                if not ok: print(f"{k + 1}/{len(todo)} {c['name']}: could not be built, keeps its plain shape", flush=True); continue
                                await kit(f"kit._exact({name!r}, {c['size']!r})")
                                for j, i in enumerate(c["objects"]):                # one built, the others of its kind copied from it
                                    at, rot = spot(boxes["objects"][i], room); one = name if j == 0 else await kit(f"kit._copy({name!r}, {name + '_' + str(j)!r})")
                                    await run(f"import kit\nkit._drop_object({i})\ne = kit.place({one!r}, {at!r}, {rot})\ne['object'] = {i}")
                                built += 1; print(f"{k + 1}/{len(todo)} {c['name']}" + (f" x {len(c['objects'])}" if len(c["objects"]) > 1 else "") + f" built  [{secs:.0f} s]", flush=True)
                    out, bad = await run(f"import bpy\nbpy.ops.wm.save_as_mainfile(filepath={os.path.join(work, 'scene.blend')!r})")
                    if bad or not os.path.exists(os.path.join(work, "scene.blend")): raise SystemExit("Blender could not save the scene: " + out[-300:])
                    print(f"{built} of {len(todo)} kinds of furniture built by the model in {(time.time() - t0) / 60:.1f} minutes", flush=True)
    finally: host.close(); record.close()

def enrich(d, sentence, style=""):
    """The filled-out description alone -> WORLD_DIR/blender/enriched.txt (not written when the model's answer cannot be used)."""
    os.makedirs(os.path.join(d, "blender"), exist_ok=True)
    with chat.Chat() as model: more = enriched(model, sentence, style)
    if more:
        with open(os.path.join(d, "blender", "enriched.txt"), "w") as fh: fh.write(more + "\n")
    print("the description, filled out: " + more if more else "the description is used as it is", flush=True)

if __name__ == "__main__":
    a = sys.argv[1:]; d = os.path.abspath(a[1]); minutes = float(os.environ.get("BLENDER_MINUTES", "9"))
    if a[0] == "enrich": enrich(d, *a[2:4])
    elif a[0] == "own": asyncio.run(build(d, a[2], a[3] if len(a) > 3 else "", minutes))
    elif a[0] == "assemble": asyncio.run(assemble(d, float(os.environ.get("BLENDER_MINUTES", "4"))))
    else: raise SystemExit("blender_world.py enrich|own|assemble WORLD_DIR ...")
