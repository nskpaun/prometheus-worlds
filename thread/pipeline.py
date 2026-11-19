"""End-to-end pipelines, run from this folder.
  pipeline.py photos WORLD_DIR "Name"                     WORLD_DIR/raw/* photos -> splat + mesh + collider
  pipeline.py prompt WORLD_DIR "Name" "SENTENCE" ["STYLE"]   a sentence -> a world
Each step prints a 'STAGE: ...' line. On success the world is added to scenes.json.
A sentence is built by the direct method (build/, see docs/direct-method.md): a rich layout, real geometry for the room,
materials and object paint from a style guide, traced light, and the splat written straight from all of it. Its Standard
build takes about 20 minutes; Full spends more on shapes and paint. Running it again on the same folder goes on where it
stopped. Switches (environment):
  WORLD_METHOD=ring     the earlier method instead: boxes painted from a ring of views, then a trained splat (it needs a
                        STYLE). An outdoor place is always built that way: the direct method only builds rooms yet.
  WORLD_VIEWS           the app's Detail choice: 16 and over is Full, anything else Standard (ring method: how many views)
  WORLDS_ROOT           a folder that holds worlds/ and scenes.json, instead of this one (for trials that must not touch
                        the worlds already made; server.py reads it too)
The ring method is held to WORLD_MINUTES (default 20; 0 for no limit): painting is nearly all of it, so once the layout
and the shapes are made, the time that is left decides how many pictures are painted (paint_views).
"""
import sys, os, re, json, subprocess, shutil, glob, socket, time, hashlib, resource, threading, ctypes
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.environ.get("WORLDS_ROOT") or HERE)          # holds worlds/ and scenes.json; the server serves both from here
PY = os.path.join(HERE, ".venv/bin/python")                 # splat training, meshing
PY_IMG = os.path.join(HERE, "../spike/.venv/bin/python")    # image and depth models (mflux)

def stage(t): print("STAGE: " + t, flush=True)
def run(cmd, env=None, quiet=False, soft=False, watch=None):      # soft: a failure is reported and the pipeline carries on
    r = subprocess.Popen(cmd, cwd=HERE, env={**os.environ, **(env or {})}, stdout=subprocess.DEVNULL if quiet else None, stderr=subprocess.STDOUT if not quiet else subprocess.DEVNULL)
    if watch: watch.follow(r)
    r.wait()
    if r.returncode and soft: print(f"Skipped a step that failed: {' '.join(map(str, cmd[:3]))}", flush=True)
    elif r.returncode: raise SystemExit(f"Step failed: {' '.join(map(str, cmd[:3]))}")
    return not r.returncode

class Memory:
    """The most memory any one process under a step held, as macOS counts it against the machine (its footprint, which
    includes what the GPU holds for it: a picture model's resident size is about half of that). Read from the system's
    own record of each process's highest footprint, every 2 s while the step runs, for the step's process and all below it."""
    def __init__(self):
        self.peak = 0
        try: self.lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        except OSError: self.lib = None
    def of(self, pid):
        buf = (ctypes.c_uint64 * 64)()                    # rusage_info_v4: a 16-byte id, then 64-bit counters; the 29th is the highest footprint so far
        return buf[2 + 28] if self.lib and self.lib.proc_pid_rusage(int(pid), 4, ctypes.byref(buf)) == 0 else 0
    def look(self, top):
        try: rows = [l.split() for l in subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True, text=True, timeout=10).stdout.splitlines()]
        except (OSError, subprocess.SubprocessError): rows = []
        kids = {}
        for a in rows:
            if len(a) == 2: kids.setdefault(a[1], []).append(a[0])
        todo = [str(top)]
        while todo:
            q = todo.pop(); self.peak = max(self.peak, self.of(q)); todo += kids.get(q, [])
    def follow(self, proc):
        def loop():
            while proc.poll() is None: self.look(proc.pid); time.sleep(2)
        if self.lib: threading.Thread(target=loop, daemon=True).start()
    def gb(self):
        """The step's peak in GB; without the system's record, the largest resident size of any process so far."""
        return round((self.peak or resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 2**30, 1)

def url(d):
    """A world folder's address on the server. It must lie under ROOT, or the viewer could not fetch its files."""
    rel = os.path.relpath(d, ROOT)
    if rel.startswith(".."): raise SystemExit(f"The world folder must be inside {ROOT} (set WORLDS_ROOT to build somewhere else).")
    return "/" + rel

def register(entry):
    path = os.path.join(ROOT, "scenes.json")
    try: scenes = json.load(open(path))
    except OSError: scenes = []
    scenes = [s for s in scenes if s["name"] != entry["name"]] + [entry]
    json.dump(scenes, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)

def photos(d, name):
    """One photo: its SHARP piece is the world. Eight or more: cameras are solved, every photo's piece is placed and the pieces
    merged (photo_fuse.py). PHOTOS_METHOD=train builds the earlier way instead, by training a splat on the photos."""
    rel = url(d)
    raw = sorted(glob.glob(f"{d}/raw/*")); one = len(raw) == 1; train = os.environ.get("PHOTOS_METHOD") == "train"
    if not one and len(raw) < 8: raise SystemExit(f"Pick one photo, or at least 8 (got {len(raw)})")
    if one and train: raise SystemExit("One photo cannot be trained on: it needs the default method")
    stage(f"converting {len(raw)} photo" + ("" if one else "s"))
    os.makedirs(f"{d}/images", exist_ok=True)
    for i, f in enumerate(raw):       # at most 1920 across; a smaller photo keeps its size (sips -Z would enlarge it)
        side = [int(x) for x in re.findall(r"pixel(?:Width|Height): (\d+)", subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", f], capture_output=True, text=True).stdout)]
        run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "92"] + (["-Z", "1920"] if not side or max(side) > 1920 else []) + [f, "--out", f"{d}/images/{i:03d}.jpg"], quiet=True)
    for p in ("db.db", "sparse", "dense", "txt", "pieces", "matches.npz", "pairs.txt"):
        shutil.rmtree(f"{d}/{p}", ignore_errors=True) if os.path.isdir(f"{d}/{p}") else (os.path.exists(f"{d}/{p}") and os.remove(f"{d}/{p}"))
    n_reg = 1
    if not one:
        stage("solving camera positions")
        # learned features find matches on plain walls and tiles where SIFT does not; if that step fails, COLMAP's own SIFT is used
        learned = not train and run([PY_IMG, "photo_match.py", f"{d}/images", f"{d}/matches.npz"], {"PYTHONDONTWRITEBYTECODE": "1"}, soft=True)
        run([PY, "sfm.py", "solve", d] + ([f"{d}/matches.npz"] if learned else []), quiet=True)      # COLMAP, through pycolmap: camera positions, undistorted images
        os.makedirs(f"{d}/dense/sparse/0")
        for b in glob.glob(f"{d}/dense/sparse/*.bin"): shutil.move(b, f"{d}/dense/sparse/0/")
        os.makedirs(f"{d}/txt"); run([PY, "sfm.py", "totxt", f"{d}/dense/sparse/0", f"{d}/txt"], quiet=True)
        n_reg = len([l for l in open(f"{d}/txt/images.txt") if l[0] != "#"]) // 2
        print(f"registered {n_reg} of {len(raw)} photos", flush=True)
        if n_reg < 5: raise SystemExit(f"Only {n_reg} photos could be placed: the photos need more overlap")
    if train:
        stage(f"training the splat ({n_reg} photos placed)")
        run([PY, "train_scene.py", f"{d}/dense", f"{d}/scene", "7000"])
        shutil.copy(f"{d}/scene_scene.json", f"{d}/scene.json")
        stage("estimating depth for each photo")
        run([PY_IMG, "photo_depth.py", f"{d}/dense/images", f"{d}/mono"])
        stage("building the mesh and collider")
        run([PY, "mesh_photos.py", d, f"{d}/mesh"])
    else:
        stage("building the scene from the photo" if one else f"building a piece of the scene from each photo ({n_reg} placed)")
        run([PY_IMG, "photo_pieces.py"] + ([f"{d}/images", f"{d}/pieces"] if one else [f"{d}/dense/images", f"{d}/pieces", "--colmap", f"{d}/txt"]), {"PYTHONDONTWRITEBYTECODE": "1"})
        if not one: stage("merging the pieces into one scene")
        run([PY, "photo_fuse.py", d] + (["--one"] if one else []))
        stage("building the mesh and collider")       # one photo may look well down or up, so its floor may lean far from the camera's up
        run([PY, "mesh_photos.py", d, f"{d}/mesh", "--surface", f"{d}/surface.ply"] + (["--tilt", "75"] if one else []))
    s = json.load(open(f"{d}/scene.json"))
    entry = {"name": name, "ply": f"{rel}/scene.ply", "mesh": f"{rel}/mesh.glb", "collider": f"{rel}/mesh_collider.glb", "up": s["up"], "center": s["center"],
             "eye": s["eye"], "forward": s["forward"], "note": ("1 photo" if one else f"{n_reg} photos" if n_reg == len(raw) else f"{n_reg} of {len(raw)} photos placed") + f", {s['splats'] // 1000}k splats", "kind": "photos"}
    entry.update({k: s[k] for k in ("focal", "width", "metre", "backdrop") if k in s})
    if not train:       # these worlds are in metres, so the floor and up found from the surface can be used (the trained ones' scale is not sure enough: see mesh_photos.py)
        entry["method"] = "pieces"; meta = json.load(open(f"{d}/mesh_meta.json"))
        if "up" in meta and "floor" in meta: entry["up"], entry["floor"] = meta["up"], meta["floor"]
    register(entry)
    stage("done")

def prompt_entry(d, name, layout, style, room="gen_c"):
    """The scenes.json entry of a world built from a sentence, whose splat and mesh are in the folder `room` of d."""
    rel = url(d) + "/" + room
    s, m = json.load(open(f"{d}/{room}/{room}_scene.json")), json.load(open(f"{d}/{room}/{room}_metre.json"))
    entry = {"name": name, "ply": f"{rel}/{room}.ply", "mesh": f"{rel}/{room}_mesh.glb", "collider": f"{rel}/{room}_mesh_collider.glb", "up": s["up"], "center": s["center"],
             "eye": s["eye"], "forward": s["forward"], "metre": m["metre"], "room": m["room"], "flat": True, "lamp": 5, "note": f"invented: {layout[:60]}", "kind": "prompt"}
    entry.update({k: m[k] for k in ("centre", "floor") if k in m})          # exact room placement for the viewer
    entry.update({k: s[k] for k in ("focal", "width") if k in s})           # lets the viewer match the trainer's blur
    entry.update({k: m[k] for k in ("up", "eye", "forward") if k in m})     # the room's exact up; a start view that looks at the scene, not at a post
    entry["style"] = style
    # only worlds that have them: where room metres lie in the splat, outdoors and its sky, wind, mood, effects
    entry.update({k: m[k] for k in ("fromRoom", "setting", "sky", "surround", "wind", "mood", "effects") if k in m})
    # the things that move; the files of those with a splat of their own become paths the viewer can fetch
    if m.get("movers"): entry["movers"] = [{**v, **{k: f"{rel}/{v[k]}" for k in ("ply", "mesh", "collider") if k in v}} for v in m["movers"]]
    return entry

PAINT_S = 56.0      # seconds to paint one picture at 768x576 in 8 steps on an M4 Pro (measured 55.8)
def paint_seconds(size, steps=8):
    """Seconds to paint one picture of that size. Measured: 55.8 s at 768x576, 42.9 s at 640x480 (smaller pictures do
    not pay: 0.69 of the pixels for 0.77 of the time), about 148 s at 1024x768 (160 s before the painter's 12 s a view of
    mask filters went). A step is 6.85 s of the 56, the rest is per picture."""
    px = size[0] * size[1] / (768 * 576)
    return PAINT_S * (px ** 1.69 if px > 1 else 0.3 + 0.7 * px) * (1 + 6.85 * steps) / (1 + 6.85 * 8)

def paint_views(d, spent, size, steps=8, limit=None):
    """How many ordinary pictures there is time to paint: what is left of the limit after `spent` seconds (the layout
    and the shapes) and after what follows the painting, over the seconds a picture takes. None for no limit.
    What follows, measured on worlds of 114 to 136 training views: the blockout and the plan 16 s; making the views
    agree 60 to 80 s, and 60 more for the second set when things move away; training 75 s, and again for that set;
    the mesh, the moving things and loading the painter 10 s."""
    if limit is None:
        try: limit = 60 * float(os.environ.get("WORLD_MINUTES", "20"))
        except ValueError: limit = 1200.0
    if limit <= 0: return None
    try:
        with open(f"{d}/boxes.json") as h: w = json.load(h)
        n = len(w["objects"])
        moving = any(isinstance(m, dict) and type(m.get("i")) is int and 0 <= m["i"] < n and m.get("type") in ("swing", "spin", "bob", "path") for m in w.get("motions") or [])
    except (OSError, ValueError, KeyError, TypeError): moving = False
    after = 16 + 85 + 75 + 10 + (60 + 75 if moving else 0)
    return max(0.0, (limit - spent - after) / paint_seconds(size, steps))

def prompt_ring(d, name, layout, style):
    """The ring-view method: boxes (and a few sculpted shapes) painted from a ring of views, then a trained splat."""
    os.makedirs(d, exist_ok=True); t0 = time.time(); url(d)
    n_views = int(os.environ.get("WORLD_VIEWS", "12"))
    # the full detail level paints at 1024x768; the others at 768x576, which is about twice as fast per view
    env = {"WORLD_DIR": os.path.relpath(d, HERE), "WORLD_VIEWS": str(n_views),
           "WORLD_PAINT": os.environ.get("WORLD_PAINT", "1024x768" if n_views >= 16 else "768x576")}
    # objects get real 3D shapes unless WORLD_SHAPES is 0 (or not a number) or the shape model is not installed. Their plan
    # needs the layout model too, so both questions are asked in one sitting; a plan made for this layout and style is kept
    try: shaped = int(float(os.environ.get("WORLD_SHAPES", "6"))) > 0
    except (ValueError, OverflowError): shaped = False
    hy3d = os.environ.get("HY3D_DIR") or os.path.join(HERE, "../ext/Hunyuan3D-Swift")
    if shaped and not (os.path.exists(f"{hy3d}/.build/release/hy3d") and os.path.isdir(f"{hy3d}/weights/shape-small")):
        print(f"The 3D shape model is not installed ({os.path.normpath(hy3d)}): objects stay boxes.", flush=True); shaped = False
    need_layout = not os.path.exists(f"{d}/boxes.json")           # a re-run keeps its layout: asking the model again could change it
    try:
        p, w = json.load(open(f"{d}/shapes/plan.json")), json.load(open(f"{d}/boxes.json"))
        layout_sum = hashlib.md5(json.dumps([w.get("room"), w.get("objects")], sort_keys=True).encode()).hexdigest()     # as shapes.stamp
        planned = p.get("style") == style and p.get("layout") == layout_sum
    except (OSError, ValueError, AttributeError): planned = False
    need_plan = shaped and not planned
    if need_layout or need_plan:
        if need_layout: stage("planning the layout")
        if need_layout: run([PY, "llm_boxes.py", layout, f"{d}/boxes.json"])    # the layout model runs in its own process and leaves memory when done
        if need_plan:
            stage("choosing which objects get real shapes")
            run([PY, "shapes.py", "plan", style], env, soft=True)
    if shaped:      # each step loads one model, skips what exists, and on any failure leaves the objects as boxes
        stage("making shapes for the objects")
        t = time.time()
        run([PY_IMG, "shapes.py", "pictures"], env, soft=True)
        run([PY, "shapes.py", "build"], env, soft=True)        # also after a failure above: it uses the pictures that were made
        print(f"shapes took {time.time() - t:.0f} s", flush=True)
    # the time that is left decides what is painted (chisel_ring prep plans it). The full detail level's 1024x768 takes
    # 2.6 times as long a picture: when under 8 of them would fit, the pictures are 768x576 like the other levels'
    size = lambda: [int(v) for v in env["WORLD_PAINT"].split("x")]; steps = 6 if n_views <= 8 else 8
    try:
        views = paint_views(d, time.time() - t0, size(), steps)
        if views is not None and views < 8 and "WORLD_PAINT" not in os.environ and size()[0] > 768:
            env["WORLD_PAINT"] = "768x576"; views = paint_views(d, time.time() - t0, size(), steps)
    except (ValueError, IndexError): views = None
    if views is not None and "WORLD_PAINT_VIEWS" not in os.environ:
        env["WORLD_PAINT_VIEWS"] = f"{views:.2f}"
        print(f"{time.time() - t0:.0f} s so far: time to paint {views:.1f} pictures of {env['WORLD_PAINT']}", flush=True)
    stage("rendering the blockout")
    run([PY, "chisel_ring.py", "prep"], env)
    stage("painting views")
    run([PY_IMG, "chisel_ring.py", "gen", "gen", style], env)   # skips views already painted for this layout, view count, size and style
    stage("making the views agree")
    run([PY, "consistent.py", "gen"], env)
    stage("training the splat")
    # painted rooms have no view-dependent shine, so low-order colour is enough: the file is about 2.4x smaller
    train = lambda n, soft=False: run([PY, "train_scene.py", f"{d}/{n}", f"{d}/{n}/{n}", "10000", "--sh", "1"], soft=soft)
    train("gen_c")
    for old in glob.glob(f"{d}/gen_c/mover_*") + glob.glob(f"{d}/gen_c/movers.json") + glob.glob(f"{d}/gen_m/gen_m*"): os.remove(old)   # an earlier build's
    try: lifted = json.load(open(f"{d}/gen_c/lifted.json"))
    except (OSError, ValueError): lifted = []
    room = "gen_c"
    if lifted:
        # things that move away from their spot: gen_c was trained without them, gen_m is the same views with them. Each
        # is cut out of gen_m's splat. If that fails, gen_m becomes the world, where they are in place and stay still
        stage("lifting out the moving things")
        if not (train("gen_m", True) and run([PY, "movers.py", "gen_c", "gen_m"], env, soft=True)):
            if os.path.exists(f"{d}/gen_m/gen_m.ply"): room = "gen_m"; print("The moving things could not be lifted out: they stay still in this world.", flush=True)
            else:
                try: objs = json.load(open(f"{d}/boxes.json"))["objects"]; names = ", ".join(str(objs[i].get("name", i)) for i in lifted)
                except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError): names = f"{len(lifted)} things"
                print(f"The moving things could not be made: this world is built without them. Missing: {names}.", flush=True)
    stage("building the mesh and collider")
    run([PY, "blockout_mesh.py", room], env)
    register(prompt_entry(d, name, layout, style, room))
    stage("done")

# ---------------------------------------------------------------- the direct method (build/)
BUILD = os.path.join(HERE, "build")
KEPT = ("layout.json", "layout.raw.json", "boxes.json", "shapes", "materials", "paint", "world.ply", "world_sh.ply", "collider.glb", "scene.json", "build.json", "steps.json", "export", "export.log")

def world_entry(d, name, sentence, style):
    """The scenes.json entry of a world built by the direct method: what assemble.py wrote, with the files as addresses."""
    rel = url(d); s = json.load(open(f"{d}/scene.json"))
    return {"name": name, **s, **{k: f"{rel}/{s[k]}" for k in ("ply", "ply2", "mesh", "collider") if k in s},
            "note": f"built: {sentence[:60]}", "kind": "prompt", "style": style or sentence, "method": "direct"}

def has_shapes():
    hy3d = os.environ.get("HY3D_DIR") or os.path.join(HERE, "../ext/Hunyuan3D-Swift")
    return os.path.exists(f"{hy3d}/.build/release/hy3d") and os.path.isdir(f"{hy3d}/weights/shape-small")

def prompt_direct(d, name, sentence, style=""):
    os.makedirs(d, exist_ok=True); url(d); t00 = time.time()
    quality = "full" if int(os.environ.get("WORLD_VIEWS", "12")) >= 16 else "standard"
    env = {"WORLD_DIR": d, "WORLD_QUALITY": quality, "PYTHONDONTWRITEBYTECODE": "1"}
    sys.path.insert(0, BUILD); import where; q = where.QUALITY[quality]
    # steps.json: what this folder is a build of, and which steps are done. The same sentence, style and quality: go on after
    # the last finished step. Anything else: the folder's earlier build is cleared away first
    mine = {"sentence": sentence, "style": style, "quality": quality}
    def clear(keep=()):
        for f in KEPT:
            if f in keep: continue
            if os.path.isdir(f"{d}/{f}"): shutil.rmtree(f"{d}/{f}")
            elif os.path.exists(f"{d}/{f}"): os.remove(f"{d}/{f}")
    try: rec = json.load(open(f"{d}/steps.json"))
    except (OSError, ValueError): rec = {}
    if {k: rec.get(k) for k in mine} != mine or not isinstance(rec.get("done"), dict): clear(); rec = {**mine, "done": {}}
    def save(): json.dump(rec, open(f"{d}/steps.json.tmp", "w"), indent=1); os.replace(f"{d}/steps.json.tmp", f"{d}/steps.json")
    def step(key, cmd):
        """One process of the build, skipped if an earlier run finished it. Its seconds and its own peak memory go on record."""
        if key in rec["done"]: return
        t = time.time(); mem = Memory(); run(cmd, env, watch=mem)
        rec["done"][key] = {"s": round(time.time() - t, 1), "peak_gb": mem.gb()}; save()
    b = lambda f: os.path.join(BUILD, f)
    save()
    if rec.get("outdoor"): return prompt_ring(d, name, sentence, style or sentence)       # found to be outdoors on an earlier run
    stage("planning the world")
    step("layout", [PY, b("layout.py"), sentence, f"{d}/layout.json"])        # the world-plan model runs in its own process and leaves memory when done
    step("plan", [PY, b("sculpt.py"), "plan", f"{d}/layout.json", str(q["sculpt_s"])])
    try: outdoor = json.load(open(f"{d}/boxes.json")).get("setting") == "outdoor"
    except (OSError, ValueError): outdoor = False
    if outdoor:         # only when the choice was left to the plan: with Indoors chosen the layout never answers outdoor
        print("This is an outdoor place, which the direct method cannot build yet: building it with the ring-view method.", flush=True)
        clear(keep=("steps.json",)); rec.update(outdoor=True, done={}); save()
        return prompt_ring(d, name, sentence, style or sentence)
    stage("sculpting the objects")
    step("pictures", [PY_IMG, b("sculpt.py"), "pictures"])
    step("meshes", [PY, b("sculpt.py"), "meshes"])
    step("place", [PY, b("sculpt.py"), "place"])
    stage("writing the style guide")
    step("style", [PY, b("style.py"), sentence] + ([style] if style else []))
    stage("making the materials")
    step("materials", [PY, b("materials.py"), str(q["swatches"])])
    stage("painting the objects")
    step("paint_prep", [PY, b("paint.py"), "prep"])
    step("paint_ref", [PY, b("paint.py"), "ref"])            # clears earlier answers: never run again once the pictures below exist
    step("paint_pictures", [PY_IMG, b("klein_paint.py"), f"{d}/paint/jobs.json"])
    if q["paint_maps"]:
        step("paint_maps", [PY, b("paint.py"), "maps"])
        step("paint_map_pictures", [PY_IMG, b("klein.py"), f"{d}/paint/mapjobs.json"])
    step("paint_bake", [PY, b("paint.py"), "bake"])
    stage("laying the splat and tracing the light")
    step("assemble", [PY, b("assemble.py")])
    register(world_entry(d, name, sentence, style))
    rec["total_s"] = round(sum(v["s"] for v in rec["done"].values()), 1); rec["peak_gb"] = max(v["peak_gb"] for v in rec["done"].values()); save()
    print(f"built in {(time.time() - t00) / 60:.1f} minutes ({rec['total_s'] / 60:.1f} of work over all runs); the largest process held {rec['peak_gb']} GB of memory", flush=True)
    stage("done")

def export(d, name):
    """A world built by the direct method as glTF files for Unreal Engine (build/export.py) -> WORLD_DIR/export/."""
    if not os.path.exists(f"{d}/build.json"): raise SystemExit("Only worlds built from a sentence by the direct method can be exported for Unreal yet")
    run([PY, os.path.join(BUILD, "export.py"), d, f"{d}/export", name], {"PYTHONDONTWRITEBYTECODE": "1"})

def prompt(d, name, sentence, style=""):
    asked = os.environ.get("WORLD_METHOD", "").strip().lower() == "ring"; outdoor = os.environ.get("WORLD_SETTING") == "outdoor"
    if not (asked or outdoor or not has_shapes()): return prompt_direct(d, name, sentence, style)
    if asked: pass
    elif outdoor: print("An outdoor place: built with the ring-view method (the direct method only builds rooms yet).", flush=True)
    else: print("The 3D shape model is not installed (run ./setup.sh to add it): built with the ring-view method, where objects stay painted boxes.", flush=True)
    return prompt_ring(d, name, sentence, style or sentence)

if __name__ == "__main__":
    {"photos": photos, "prompt": prompt, "export": export}[sys.argv[1]](os.path.abspath(sys.argv[2]), *sys.argv[3:])
