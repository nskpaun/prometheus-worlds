"""24 fixed cameras for judging a prompt world, the same every run, placed from its layout (boxes.json).
  evals/cams.py WORLD            prints [{"name", "kind": "centre"|"close"|"floor"|"far", "pose"}] as JSON
  evals/cams.py WORLD --check    rebuilds every training pose in WORLD/gen_c/gen_c_scene.json from the layout cameras in
                                 WORLD/gen_c/cams.json and prints the largest difference
WORLD is a folder name under worlds/ or a path. Needs WORLD/boxes.json and WORLD/gen_c/cams.json; nothing is written.
8 "centre": the free spot nearest the middle, eye height, looking level eight ways.
12 "close": 0.8 m from each wall and from the largest objects, looking straight at it and turned 45 degrees along it.
            From an object that is 0.8 m level across the floor from its box, at eye height: 1.3 to 1.7 m from a low
            thing's surface. close_x0, close_x1, close_z0, close_z1 are the four walls straight on.
            Outdoors there are no walls: the same spots at the edges of the ground to walk on look out over the far
            backdrop, 20 m and more away. Those are named far_... and are of kind "far", so they do not pass for close
            views; only the object views are "close" there, and more objects are visited.
4 "floor": off-centre, eye height, pitched 35 degrees down across open floor.
"pose" is camera-to-world as in gen_c_scene.json "poses" (what train_scene.py passes to render_from_pose): columns right,
up, back (the camera looks along minus its third column), then the position. From a layout camera (eye, look) in room
metres, with r, u, f = blockout.basis and S = the z mirror consistent.py writes its cameras with:
  rotation = [S(-r), S(u), -S(f)],  position = (S(eye) - m) * k
m = the mean of all training cameras' mirrored eyes, k = 1 / the largest |coordinate| among them once centred: the
trainer centres and scales its cameras like that. Measured with --check: the 125, 114 and 128 training poses of
ex-tower, garage-shapes and ex-blossom all come out within 6e-7.
"""
import sys, os, re, json, importlib, contextlib, numpy as np
EV = os.path.dirname(os.path.abspath(__file__)); TH = os.path.dirname(EV)
S = np.diag([1.0, 1, -1])
NEAR, PITCH = 0.8, 35.0      # metres a close camera stands from what it looks at; degrees a floor camera looks down

def layout(world):
    """(chisel_ring loaded for this world, the world's folder). chisel_ring holds one world at a time: asking for
    another loads it afresh. Its cameras depend on the view count the pictures were painted for, as in pipeline.py."""
    world = os.path.abspath(world if os.path.isdir(world) else os.path.join(TH, "worlds", world))
    try: views = str(json.load(open(f"{world}/gen/views.json"))["views"])
    except (OSError, ValueError, KeyError): views = "12"
    C = sys.modules.get("chisel_ring")
    if C and os.path.abspath(C.D) == world and str(C.RING) == views: return C, world
    old = {k: os.environ.get(k) for k in ("WORLD_DIR", "WORLD_VIEWS")}
    os.environ.update(WORLD_DIR=world, WORLD_VIEWS=views)
    for p in (TH, os.path.join(TH, "../spike")):
        if p not in sys.path: sys.path.insert(0, p)
    try:
        with contextlib.redirect_stdout(sys.stderr):      # its notes must not end up in our JSON
            C = importlib.reload(C) if C else importlib.import_module("chisel_ring")
    finally:                                              # the caller's environment is left as it was
        for k, v in old.items(): os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return C, world

def layout_cams(world):
    """The cameras in room metres: [(name, kind, eye, look)]."""
    C, _ = layout(world); B = C.B; c = np.array([C.RW / 2, C.EYE, C.RD / 2])
    # a splat is trained without the things that move away, so nothing is placed by them. Shapes count as their boxes
    # (C.KEEP_OUT, as for the painted cameras): no mesh is ray-cast here
    solid = [p for p, oi in zip(C.KEEP_OUT, (oi for oi, o in enumerate(C.objs) for _ in o["prims"])) if oi not in C.LIFT]
    clear = lambda e: 0.4 < e[0] < C.RW - 0.4 and 0.4 < e[2] < C.RD - 0.4 and not any(B.inside(e, p, 0.3) for p in C.KEEP_OUT)
    steps = sorted(((dx, dz) for dx in np.arange(-40, 41) / 10 for dz in np.arange(-40, 41) / 10), key=lambda s: (round(s[0] ** 2 + s[1] ** 2, 6), s))
    def spot(want):
        """The clear place nearest to `want`, at its height (on a 10 cm grid; `want` itself if there is none within 4 m)."""
        return next((e for e in (np.array(want, float) + [dx, 0, dz] for dx, dz in steps) if clear(e)), np.array(want, float))
    def ahead(e, d):
        """Metres from e along d to the first wall or object."""
        d = np.array(d, float)[None] / np.linalg.norm(d)
        return float(min([B.slab(e, d, *B.ROOM)[1][0]] + [B.hit(e, d, p)[0] for p in solid]))
    turn = lambda d, deg: np.array([np.cos(np.radians(deg)) * d[0] - np.sin(np.radians(deg)) * d[2], d[1], np.sin(np.radians(deg)) * d[0] + np.cos(np.radians(deg)) * d[2]])
    out = []
    mid = spot(c)
    for i in range(8): out.append((f"centre_{i}", "centre", mid, mid + turn([1, 0, 0], 45 * i)))
    # close: (name, eye, direction) per wall and per object, largest first
    walls = []
    for name, at, n, length in (("x0", [0, C.RD / 2], [1, 0], C.RD), ("x1", [C.RW, C.RD / 2], [-1, 0], C.RD), ("z0", [C.RW / 2, 0], [0, 1], C.RW), ("z1", [C.RW / 2, C.RD], [0, -1], C.RW)):
        n3, along = np.array([n[0], 0, n[1]], float), np.array([n[1], 0, n[0]], float); first = np.array([at[0], C.EYE, at[1]]) + NEAR * n3
        for slide in [k * sg for k in np.arange(0, length / 2 - 0.5, 0.5) for sg in (1, -1)]:     # along the wall until a place is free
            e = first + slide * along
            if not clear(e): continue
            e = e + n3 * max(0.0, NEAR - ahead(e, -n3))       # something stands against the wall: back off to 0.8 m from that
            if clear(e): break
        else: e = spot(first)
        walls.append((name, e, -n3))
    things = []
    for oi, o in enumerate(C.objs):
        if oi in C.LIFT: continue
        lo, hi = C.bounds(o["prims"]); m = (lo + hi) / 2; aim_y = min(C.EYE, 0.25 * lo[1] + 0.75 * hi[1])      # level at tall things, down at low ones
        sides = []
        for n3, half, wide in (([1, 0, 0], hi[0] - m[0], hi[2] - lo[2]), ([-1, 0, 0], hi[0] - m[0], hi[2] - lo[2]), ([0, 0, 1], hi[2] - m[2], hi[0] - lo[0]), ([0, 0, -1], hi[2] - m[2], hi[0] - lo[0])):
            n3 = np.array(n3, float); face = np.array([m[0], aim_y, m[2]]) + n3 * half; e = face + n3 * NEAR; e[1] = C.EYE
            sides.append((not clear(e), -round(wide, 3), round(float(np.hypot(e[0] - c[0], e[2] - c[2])), 3), len(sides), e, face))
        blocked, _, _, _, e, face = min(sides, key=lambda s: s[:4])      # its widest side that can be stood at, nearest the middle
        if blocked: e = spot(e)
        if np.hypot(*(face - e)[[0, 2]]) < 0.2: continue                 # standing on top of it: no level direction to look in
        things.append((-(hi[1] - lo[1]) * max(hi[0] - lo[0], hi[2] - lo[2]), oi, "obj%d-%s" % (oi, re.sub(r"[^a-z0-9]+", "-", o["name"].lower()).strip("-")), e, face - e))
    # a thing against a wall is often that wall's own view: once is enough
    same = lambda a, b: np.linalg.norm(a[1] - b[1]) < 0.5 and a[2][[0, 2]] @ b[2][[0, 2]] > np.cos(np.radians(30)) * np.linalg.norm(a[2][[0, 2]]) * np.linalg.norm(b[2][[0, 2]])
    things = [t[2:] for t in sorted(things, key=lambda t: t[:2]) if not any(same(t[2:], w) for w in walls)]
    n_obj = 4 if C.OUT else 2
    close = walls + ([] if C.OUT else [(n + "_45", e, turn(d, 45)) for n, e, d in walls])
    close += things[:n_obj] + [(n + "_45", e, turn(d, 45)) for n, e, d in things[:n_obj]]
    # too few objects: the walls again, turned the other way (and outdoors both ways), then further objects
    close += [(n + "_-45", e, turn(d, -45)) for n, e, d in walls] + ([(n + "_45", e, turn(d, 45)) for n, e, d in walls] if C.OUT else []) + things[n_obj:]
    for n, e, d in close[:12]:
        kind = "close" if n.startswith("obj") or not C.OUT else "far"
        out.append((f"{kind}_{n}", kind, e, e + d))
    # floor: from four spots off the middle, turned (from facing the middle) until the centre of the picture is open floor
    down = np.array([np.cos(np.radians(PITCH)), -np.sin(np.radians(PITCH)), 0]); reach = C.EYE / np.sin(np.radians(PITCH))
    for i, (sx, sz) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        e = spot(c + [sx * 0.22 * C.RW, 0, sz * 0.22 * C.RD]); to = np.degrees(np.arctan2(c[2] - e[2], c[0] - e[0]))
        for a in (0, 45, -45, 90, -90, 135, -135, 180):
            d = turn(down, to + a); p = e + d * reach
            if ahead(e, d) > reach - 0.01 and 0 < p[0] < C.RW and 0 < p[2] < C.RD: break
        else: d = turn(down, to)
        out.append((f"floor_{i}", "floor", e, e + d))
    return out

def placing(cams_json):
    """(m, k): where the trainer puts a set of layout cameras (see the top of this file)."""
    eyes = np.array([S @ np.array(c[0], float) for c in json.load(open(cams_json))])
    m = eyes.mean(0); return m, 1 / np.abs(eyes - m).max()

def pose(B, cam, m, k):
    """A layout camera (eye, look) as the trainer's camera-to-world matrix."""
    eye, r, u, f = B.basis(cam); M = np.eye(4)
    M[:3, :3] = np.stack([S @ -r, S @ u, -(S @ f)], 1); M[:3, 3] = (S @ eye - m) * k
    return M

def eval_cams(world_dir, cams_json=None):
    """The 24 cameras as poses for a splat trained on the layout cameras listed in cams_json (default
    world_dir/gen_c/cams.json, which consistent.py writes). Train on other cameras and the splat lies elsewhere: pass
    that training set's cams.json."""
    C, world = layout(world_dir); m, k = placing(cams_json or f"{world}/gen_c/cams.json")
    return [{"name": n, "kind": kind, "pose": pose(C.B, (e, l), m, k).tolist()} for n, kind, e, l in layout_cams(world)]

def check(world_dir):
    """Largest difference between the training poses the trainer saved and the ones worked out here from the layout."""
    C, world = layout(world_dir); m, k = placing(f"{world}/gen_c/cams.json")
    mine = np.array([pose(C.B, cam, m, k) for cam in json.load(open(f"{world}/gen_c/cams.json"))])
    saved = np.array(json.load(open(f"{world}/gen_c/gen_c_scene.json"))["poses"])
    return {"world": os.path.basename(world), "training_poses": len(saved), "max_pose_error": float(np.abs(mine - saved).max())}

if __name__ == "__main__":
    print(json.dumps(check(sys.argv[1]) if "--check" in sys.argv[2:] else eval_cams(sys.argv[1])))
