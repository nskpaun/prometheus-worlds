"""Redo the last steps of a prompt build on a finished world's painted views: agree (consistent.py), train, mesh, and
render fixed cameras. About 3 minutes instead of the 20 a full build takes; the image model is never run.
  evals/refinish.py WORLD [--label L] [--poses FILE] [--skip-agree] [--views-from L2] [--train "ARGS"]     (run from thread/)
WORLD is a folder name under worlds/ or a path. Everything is written to evals/work/WORLD-L/ (the world is only read):
gen_c/ (training views, gen_c.ply, gen_c_scene.json, gen_c_mesh.glb), renders/NAME.png, poses.json, agree.log, train.log.
--poses FILE     cameras to render ([{"name", "pose"}], poses as in gen_c_scene.json); default evals/cams.py's eval_cams
--skip-agree     keep this work folder's training views and only train again
--views-from L2  train on the views of evals/work/WORLD-L2 instead of making them (trainer-only experiments, own label).
                 The agree step gives the same views bit for bit every run, so unless consistent.py or its environment
                 changed this is the same as a whole refinish, a minute or two sooner
--train "ARGS"   extra train_scene.py arguments, e.g. "--set refine_every=200"
Environment variables (WORLD_LIFT and the like) pass through to every step. Prints one JSON line.
The trainer centres and scales the world by the cameras it is given, so a run that trains on other cameras ends in
another frame. Nothing here is taken from the finished world's frame: the eval cameras are placed from this run's own
training cameras (gen_c/cams.json), and the mesh for the splat scores is built again from this run's own training
(blockout_mesh.py), without the things this run lifted out. "pose_error" is the proof: the largest difference between
the training poses the trainer saved and the ones worked out from the layout cameras; None when their number differs.
Only the room is trained, so the agree step is told to leave out the set with the moving things (WORLD_ROOM_ONLY).
The step count is fixed at the build's 10000. Shorter runs were tried and are no stand-in: at 5000 and 3000 steps the
same change to the trainer turned stray_share from 12% worse into 3% and 25% better (docs/evaluation.md).
"""
import sys, os, re, json, time, shutil, shlex, argparse, subprocess
EV = os.path.dirname(os.path.abspath(__file__)); TH = os.path.dirname(EV); WORK = os.path.join(EV, "work")
PY = os.path.join(TH, ".venv/bin/python")
INPUTS = ("boxes.json", "lookdown.json", "detail.json", "gen", "shapes/plan.json", "shapes/placed")      # all that consistent.py and blockout_mesh.py read

def plain(label):
    """A label becomes part of a folder name that is deleted and written: only a plain name will do."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", label or "") or label in (".", ".."): raise SystemExit(f"label {label!r}: letters, digits, dot, dash and underscore only")
    return label

def timed(cmd, log, env=None):
    """Run cmd from thread/ under /usr/bin/time -l, output to the file log. {"s": wall seconds, "peak_mb": peak memory
    footprint and "rss_mb": most resident, both of that one process; "tree_mb": the most resident memory of it and
    all the processes it started, added up each second (pages shared between workers count once each: an upper bound)}."""
    tf = log + ".time"; tree = 0
    p = subprocess.Popen(["/usr/bin/time", "-l", "-o", tf, *cmd], cwd=TH, env={**os.environ, **(env or {})}, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    while p.poll() is None:
        try: tree = max(tree, sum(int(l.split()[1]) for l in subprocess.run(["ps", "-axo", "pgid=,rss="], capture_output=True, text=True).stdout.splitlines() if int(l.split()[0]) == p.pid))
        except (ValueError, IndexError, OSError): pass
        time.sleep(1)
    if p.returncode: raise SystemExit(f"Step failed ({' '.join(map(str, cmd[1:3]))}); see {log}\n" + open(log).read()[-1500:])
    t = open(tf).read(); os.remove(tf)
    num = lambda pat: float(re.search(pat, t).group(1))
    return {"s": round(num(r"([\d.]+) real"), 1), "peak_mb": round(num(r"(\d+)\s+peak memory footprint") / 2**20),
            "rss_mb": round(num(r"(\d+)\s+maximum resident set size") / 2**20), "tree_mb": round(tree / 1024)}

def refinish(world, label="base", poses=None, skip_agree=False, views_from=None, train_args=()):
    t0 = time.time()
    world = os.path.abspath(world if os.path.isdir(world) else os.path.join(TH, "worlds", world))
    name = os.path.basename(world)
    work = os.path.join(WORK, f"{name}-{plain(label)}"); src = os.path.join(WORK, f"{name}-{plain(views_from)}") if views_from else work
    # the work folder is emptied below: it must be a folder of its own under evals/work, wherever links lead
    rw, rwork = os.path.realpath(world), os.path.realpath(work)
    if os.path.dirname(rwork) != os.path.realpath(WORK) or os.path.commonpath([rw, rwork]) in (rw, rwork): raise SystemExit(f"{work} is not a folder of its own under evals/work")
    stages = {}
    v = json.load(open(f"{world}/gen/views.json"))
    # as pipeline.py's prompt(): the view count the pictures were painted for decides where the cameras stand
    env = {"WORLD_DIR": os.path.relpath(work, TH), "WORLD_VIEWS": str(v["views"]), "WORLD_PAINT": "%dx%d" % tuple(v["paint"])}
    if not (skip_agree or views_from): shutil.rmtree(work, ignore_errors=True)       # an earlier run with this label
    if not os.path.exists(f"{work}/boxes.json"):
        for f in INPUTS:
            a, b = os.path.join(world, f), os.path.join(work, f)
            if not os.path.exists(a): continue        # a world without shapes
            os.makedirs(os.path.dirname(b), exist_ok=True)
            shutil.copytree(a, b) if os.path.isdir(a) else shutil.copy(a, b)
        stages["copy"] = {"s": round(time.time() - t0, 1)}
    if not (skip_agree or views_from):
        stages["agree"] = timed([PY, "consistent.py", "gen"], f"{work}/agree.log", {**env, "WORLD_ROOM_ONLY": "1"})
    if not os.path.isdir(f"{src}/gen_c/images"): raise SystemExit(f"no training views in {src}/gen_c: run without --skip-agree first")
    os.makedirs(f"{work}/gen_c", exist_ok=True)
    for f in ("cams.json", "lifted.json", "sky.json"):       # what the mesh step reads beside the splat: the views' own
        if src != work and os.path.exists(f"{src}/gen_c/{f}"): shutil.copy(f"{src}/gen_c/{f}", f"{work}/gen_c/{f}")
    sys.path.insert(0, EV)
    import numpy as np, cams as K
    if poses is None: cams = K.eval_cams(world, f"{work}/gen_c/cams.json")      # placed for this run's training cameras
    else: cams = json.load(open(poses))
    json.dump(cams, open(f"{work}/poses.json", "w"))
    shutil.rmtree(f"{work}/renders", ignore_errors=True)
    for old in ("gen_c_mesh.glb", "gen_c_mesh_collider.glb", "gen_c_metre.json"):
        if os.path.exists(f"{work}/gen_c/{old}"): os.remove(f"{work}/gen_c/{old}")
    stages["train"] = timed([PY, "train_scene.py", f"{src}/gen_c", f"{work}/gen_c/gen_c", "10000", "--sh", "1", *train_args,
                             "--render-poses", f"{work}/poses.json", "--render-out", f"{work}/renders"], f"{work}/train.log")
    s = json.load(open(f"{work}/gen_c/gen_c_scene.json"))
    C, _ = K.layout(world); m, k = K.placing(f"{work}/gen_c/cams.json")
    mine = np.array([K.pose(C.B, cam, m, k) for cam in json.load(open(f"{work}/gen_c/cams.json"))])
    err = float(np.abs(mine - np.array(s["poses"])).max()) if mine.shape == np.shape(s["poses"]) else None
    out = {"world": name, "label": label, "work": work, "ply": f"{work}/gen_c/gen_c.ply", "scene": f"{work}/gen_c/gen_c_scene.json",
           "views": f"{src}/gen_c/images", "renders": f"{work}/renders", "paint": f"{world}/gen/images", "mesh": None, "pose_error": err,
           "splats": s["splats"], "train_views": len(s["poses"]), "rendered": len(os.listdir(f"{work}/renders"))}
    try:    # the blockout in this splat's own frame; it refuses when the splat was trained on other cameras than cams.json lists
        stages["mesh"] = timed([PY, "blockout_mesh.py", "gen_c"], f"{work}/mesh.log", env); out["mesh"] = f"{work}/gen_c/gen_c_mesh.glb"
    except SystemExit as e: out["mesh_error"] = str(e)[-400:]
    out.update(stages=stages, total_s=round(time.time() - t0, 1))
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("world"); p.add_argument("--label", default="base"); p.add_argument("--poses")
    p.add_argument("--skip-agree", action="store_true"); p.add_argument("--views-from"); p.add_argument("--train", default="")
    a = p.parse_args()
    print(json.dumps(refinish(a.world, a.label, a.poses, a.skip_agree, a.views_from, shlex.split(a.train))))
