"""End-to-end pipelines, run from this folder.
  pipeline.py photos WORLD_DIR "Name"                    WORLD_DIR/raw/* photos -> splat + mesh + collider
  pipeline.py prompt WORLD_DIR "Name" "LAYOUT" "STYLE"   sentence + style -> blockout -> pictures -> splat
Each step prints a 'STAGE: ...' line. On success the world is added to scenes.json.
A prompt build is held to WORLD_MINUTES (default 20; 0 for no limit): painting is nearly all of it, so once the layout
and the shapes are made, the time that is left decides how many pictures are painted (paint_views).
"""
import sys, os, re, json, subprocess, shutil, glob, socket, time, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv/bin/python")                 # splat training, meshing
PY_IMG = os.path.join(HERE, "../spike/.venv/bin/python")    # image and depth models (mflux)

def stage(t): print("STAGE: " + t, flush=True)
def run(cmd, env=None, quiet=False, soft=False):      # soft: a failure is reported and the pipeline carries on
    r = subprocess.run(cmd, cwd=HERE, env={**os.environ, **(env or {})}, stdout=subprocess.DEVNULL if quiet else None, stderr=subprocess.STDOUT if not quiet else subprocess.DEVNULL)
    if r.returncode and soft: print(f"Skipped a step that failed: {' '.join(map(str, cmd[:3]))}", flush=True)
    elif r.returncode: raise SystemExit(f"Step failed: {' '.join(map(str, cmd[:3]))}")
    return not r.returncode

def register(entry):
    path = os.path.join(HERE, "scenes.json")
    scenes = [s for s in json.load(open(path)) if s["name"] != entry["name"]] + [entry]
    json.dump(scenes, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)

def photos(d, name):
    rel = "/" + os.path.relpath(d, HERE)
    raw = sorted(glob.glob(f"{d}/raw/*"))
    if len(raw) < 8: raise SystemExit(f"Need at least 8 photos, got {len(raw)}")
    stage(f"converting {len(raw)} photos")
    os.makedirs(f"{d}/images", exist_ok=True)
    for i, f in enumerate(raw):
        run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "92", "-Z", "1920", f, "--out", f"{d}/images/{i:03d}.jpg"], quiet=True)
    stage("solving camera positions")
    for p in ("db.db", "sparse", "dense", "txt"):
        shutil.rmtree(f"{d}/{p}", ignore_errors=True) if os.path.isdir(f"{d}/{p}") else (os.path.exists(f"{d}/{p}") and os.remove(f"{d}/{p}"))
    run([PY, "sfm.py", "solve", d], quiet=True)      # COLMAP, through pycolmap: features, matching, camera positions, undistorted images
    os.makedirs(f"{d}/dense/sparse/0")
    for b in glob.glob(f"{d}/dense/sparse/*.bin"): shutil.move(b, f"{d}/dense/sparse/0/")
    os.makedirs(f"{d}/txt"); run([PY, "sfm.py", "totxt", f"{d}/dense/sparse/0", f"{d}/txt"], quiet=True)
    n_reg = len([l for l in open(f"{d}/txt/images.txt") if l[0] != "#"]) // 2
    print(f"registered {n_reg} of {len(raw)} photos", flush=True)
    if n_reg < 5: raise SystemExit(f"Only {n_reg} photos could be placed: the photos need more overlap")
    stage(f"training the splat ({n_reg} photos placed)")
    run([PY, "train_scene.py", f"{d}/dense", f"{d}/scene", "7000"])
    shutil.copy(f"{d}/scene_scene.json", f"{d}/scene.json")
    stage("estimating depth for each photo")
    run([PY_IMG, "photo_depth.py", f"{d}/dense/images", f"{d}/mono"])
    stage("building the mesh and collider")
    run([PY, "mesh_photos.py", d, f"{d}/mesh"])
    s = json.load(open(f"{d}/scene.json"))
    entry = {"name": name, "ply": f"{rel}/scene.ply", "mesh": f"{rel}/mesh.glb", "collider": f"{rel}/mesh_collider.glb", "up": s["up"], "center": s["center"],
             "eye": s["eye"], "forward": s["forward"], "note": f"{n_reg} photos, {s['splats'] // 1000}k splats", "kind": "photos"}
    entry.update({k: s[k] for k in ("focal", "width") if k in s})
    # mesh_meta.json also holds estimates of scale, floor and up; they are kept on disk but not used yet, because the
    # scale disagrees with the viewer's own estimate by about 20% and one real measurement is needed to say which is right
    register(entry)
    stage("done")

def prompt_entry(d, name, layout, style, room="gen_c"):
    """The scenes.json entry of a world built from a sentence, whose splat and mesh are in the folder `room` of d."""
    rel = "/" + os.path.relpath(d, HERE) + "/" + room
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

def prompt(d, name, layout, style):
    os.makedirs(d, exist_ok=True); t0 = time.time()
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

if __name__ == "__main__":
    {"photos": photos, "prompt": prompt}[sys.argv[1]](os.path.abspath(sys.argv[2]), *sys.argv[3:])
