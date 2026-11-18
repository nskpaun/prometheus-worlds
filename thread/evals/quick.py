"""Quick checks of how a prompt world looks, at three speeds. Run from thread/. See docs/evaluation.md.
  evals/quick.py geometry [WORLD...]              seconds: how coarse the paint is on screen (density.py), no GPU
  evals/quick.py refinish [WORLD...] --label L    3 to 4 min a world: agree + train again on the world's painted views
        [--skip-agree] [--views-from L2] [--train "ARGS"]     (as refinish.py), render the 24 fixed cameras, score
  evals/quick.py compare LABEL_A LABEL_B          each score's change from A to B per world, marked better or worse
  evals/quick.py full --label L [NAME...]         ~20 min a world: builds the sentences in evals/prompts.json from nothing
WORLD is a folder name under worlds/ or a path; left out: ex-tower garage-shapes ex-blossom. The worlds are only read.
refinish and full keep one line per label and world in evals/work/results.jsonl (a rerun replaces its line) and write
evals/work/WORLD-L/sheet.jpg (every camera's render with its sharpness) and crops.jpg (the close views at full size,
each beside the paint it should show). Environment variables pass through to the pipeline steps, so an experiment is:
change the code or set a variable, run refinish with a new label, compare it with the baseline label.
A trainer-only experiment needs no new views: --views-from baseline (the agree step is the same bit for bit every run).
"""
import sys, os, json, time, glob, shlex, hashlib, argparse
EV = os.path.dirname(os.path.abspath(__file__)); TH = os.path.dirname(EV)
sys.path[:0] = [EV, TH]
from refinish import plain
WORLDS = ["ex-tower", "garage-shapes", "ex-blossom"]
RESULTS = os.path.join(EV, "work", "results.jsonl")
WALLS = ("close_x0", "close_x1", "close_z0", "close_z1")
# every score compare shows: (which way is better: +1 up, -1 down, 0 not judged; a change counts only beyond this share
# of the value AND beyond this much). The limits are at least twice the largest difference measured between three
# trainings of the same views (docs/evaluation.md). Times vary with what else the machine is doing, and memory with
# whether the agree step ran: both are shown, not judged
SCORES = {"near_r": (1, 0.02, 0), "near_worst": (1, 0.02, 0), "near_soft": (-1, 0.02, 0), "unpainted": (-1, 0.02, 0.001),
          "near_train_r": (0, 0, 0), "near_eff_r": (1, 0.02, 0), "wall_r": (1, 0.02, 0),
          "sharp_wall": (1, 0.06, 0), "sharp_close": (1, 0.02, 0), "sharp_floor": (1, 0.02, 0), "sharp_centre": (1, 0.02, 0), "sharp_far": (1, 0.02, 0),
          "kept_wall": (0, 0, 0), "detail_kept": (0, 0, 0),             # sharpness again, over the paint's own: shown, not counted twice
          "ref_corr": (1, 0, 0.01), "ref_err": (-1, 0.05, 0.001), "bg_share": (-1, 0, 0.002),
          "stray_share": (-1, 0.07, 0.001), "stray15_share": (-1, 0.05, 0.003), "needle_share": (-1, 0.05, 0.001), "large_share": (-1, 0.05, 0.001),
          "big_share": (-1, 0.1, 0.001), "median_axis_cm": (-1, 0.03, 0), "median_off_cm": (-1, 0.03, 0), "backdrop_stray_share": (-1, 0.05, 0.001),
          "splats": (0, 0, 0), "agree_s": (0, 0, 0), "train_s": (0, 0, 0), "total_s": (0, 0, 0), "peak_mb": (0, 0, 0), "tree_mb": (0, 0, 0)}
SHARP = ("sharp_wall", "sharp_close", "sharp_floor", "sharp_centre", "sharp_far")

def world_dir(world): return os.path.abspath(world if os.path.isdir(world) else os.path.join(TH, "worlds", world))

def kept(world, what, make, extra=""):
    """make()'s answer for this world, worked out once: kept in evals/work/WORLD/WHAT.json until the world's layout,
    paint or cameras or the code that reads them changes."""
    w = world_dir(world)
    files = [f"{EV}/{f}.py" for f in ("density", "cams", "reference")] + [f"{TH}/chisel_ring.py", f"{TH}/owner.py", f"{TH}/detail.py", f"{TH}/../spike/blockout.py"] + \
            [f"{w}/{f}" for f in ("boxes.json", "lookdown.json", "detail.json", "gen/views.json", "gen/images", "gen_c/cams.json", "gen_c/lifted.json", "shapes/placed")]
    key = [w, extra] + [os.path.getmtime(f) if os.path.exists(f) else None for f in files]
    path = os.path.join(EV, "work", os.path.basename(w), what + ".json")
    try:
        d = json.load(open(path))
        if d["key"] == key: return d["value"]
    except (OSError, ValueError, KeyError): pass
    value = make(); os.makedirs(os.path.dirname(path), exist_ok=True); json.dump({"key": key, "value": value}, open(path, "w"))
    return value

def geometry(world, train_cams=None, fresh=True):
    """The density scores under the names results.jsonl uses. train_cams: the cams.json of the training views to count
    (default the world's own). fresh=False: take an earlier answer for the same world, code and cameras."""
    from density import density
    def make():
        d = density(world, train_cams=train_cams); n = d["near"]; walls = [d["cams"][c] for c in WALLS if d["cams"].get(c) is not None]
        return {"near_r": n["median_r"], "near_worst": n["median_worst"], "near_soft": n["r_lt_0.5"], "unpainted": n["unpainted"],
                "near_train_r": n["median_train_r"], "near_eff_r": n["median_eff_r"], "wall_r": round(sum(walls) / len(walls), 3) if walls else None,
                "centre_r": d["kinds"]["centre"]["median_r"], "geometry_s": d["seconds"]}
    if fresh: return make()
    return kept(world, "geometry", make, hashlib.md5(open(train_cams, "rb").read()).hexdigest() if train_cams else "")

def reference(world):
    """The folder of this world's reference pictures (reference.py), made once."""
    import reference as R
    return kept(world, "reference", lambda: R.reference(world))["reference"]

def scored(rec, world, ply, mesh, renders, paint, poses, out, train_cams=None):
    """Add every score to rec. mesh None: no blockout fits this splat, so it gets no splat scores."""
    from splat_scores import score as splat
    from image_scores import score as image
    rec.update(geometry(world, train_cams, fresh=False))
    if mesh:
        s = splat(ply, mesh)
        if not s["aligned"]:      # distances to a mesh that lies elsewhere mean nothing: they are not recorded
            print(f"{rec['world']}: the splat does not lie on the mesh (on_mesh_share {s['on_mesh_share']}): no splat scores", file=sys.stderr)
            s = {k: s[k] for k in ("splats", "on_mesh_share", "aligned")}
        rec.update(s)
    i = image(renders, paint, poses, f"{out}/sheet.jpg", reference(world), f"{out}/crops.jpg")
    rec.update({"sharp_" + k: v for k, v in i["sharp"].items()}, paint_sharp=i.get("paint"), worst=i["worst"], sheet=f"{out}/sheet.jpg", crops=f"{out}/crops.jpg",
               **{k: i.get(k) for k in ("detail_kept", "kept_wall", "ref_corr", "ref_err", "bg_share", "bg_max")},
               views={n: [v[k] for k in ("sharp", "kept", "corr", "err", "bg")] for n, v in i["views"].items()})      # per view: sharp, kept, corr, err, bg
    return rec

def save(rec):
    """One line per tier, label and world: a rerun takes the place of its earlier line."""
    rec["when"] = time.strftime("%Y-%m-%d %H:%M"); same = lambda r: all(r.get(k) == rec.get(k) for k in ("tier", "label", "world"))
    old = [l for l in open(RESULTS) if l.strip() and not same(json.loads(l))] if os.path.exists(RESULTS) else []
    with open(RESULTS + ".tmp", "w") as f: f.write("".join(old) + json.dumps(rec) + "\n")
    os.replace(RESULTS + ".tmp", RESULTS)
    return rec

def refinish(world, label, **how):
    from refinish import refinish as again
    t0 = time.time(); r = again(world, plain(label), **how); st = r["stages"]
    rec = {"tier": "refinish", "label": label, "world": r["world"], "work": r["work"], "train_views": r["train_views"], "rendered": r["rendered"],
           "agree_s": st.get("agree", {}).get("s"), "train_s": st["train"]["s"], "pose_error": r["pose_error"],
           "peak_mb": max(v["peak_mb"] for v in st.values() if "peak_mb" in v),      # the largest single process
           "tree_mb": max(v["tree_mb"] for v in st.values() if "tree_mb" in v)}      # the largest step, all its processes added up
    if r["pose_error"] is None or r["pose_error"] > 1e-4:
        # the eval cameras were placed for the cameras gen_c/cams.json lists; the trainer was given others
        rec["failed"] = f"the trained poses do not match gen_c/cams.json (pose_error {r['pose_error']}): the renders are not of the eval cameras"
        print(f"{r['world']}: {rec['failed']}: not scored", file=sys.stderr)
    else:
        if not r["mesh"]: print(f"{r['world']}: no mesh could be built for this splat, so no splat scores: {r.get('mesh_error')}", file=sys.stderr)
        scored(rec, world, r["ply"], r["mesh"], r["renders"], r["paint"], f"{r['work']}/poses.json", r["work"], f"{r['work']}/gen_c/cams.json")
    rec["total_s"] = round(time.time() - t0, 1)      # scoring included
    return save(rec)

def full(label, only=()):
    """Build each sentence of prompts.json with pipeline.py into evals/work/full-LABEL/NAME and score it. Slow: the image
    model paints every view. The world is not added to scenes.json, and the room's training also renders the cameras."""
    import pipeline, cams
    out = []; plain(label)
    for p in json.load(open(os.path.join(EV, "prompts.json"))):
        if only and p["name"] not in only: continue
        d = os.path.join(EV, "work", f"full-{label}", plain(p["name"])); work = os.path.join(EV, "work", f"{p['name']}-{label}")
        os.makedirs(work, exist_ok=True); marks = []; was = (pipeline.run, pipeline.register, pipeline.stage); run = was[0]
        def run_with_renders(cmd, *a, **k):
            # the views to train on exist by now, so the fixed cameras can be placed; the room's training renders them
            if cmd[1] == "train_scene.py" and cmd[2].endswith("/gen_c"):
                json.dump(cams.eval_cams(d), open(f"{work}/poses.json", "w"))
                cmd = [*cmd, "--render-poses", f"{work}/poses.json", "--render-out", f"{work}/renders"]
            return run(cmd, *a, **k)
        keep = {k: os.environ.get(k) for k in ("WORLD_SETTING",)}
        if p.get("setting"): os.environ["WORLD_SETTING"] = p["setting"]
        pipeline.run, pipeline.register, pipeline.stage = run_with_renders, lambda entry: None, lambda t: (marks.append((t, time.time())), print("STAGE: " + t, flush=True))
        t0 = time.time(); rec = {"tier": "full", "label": label, "world": p["name"], "work": work}
        try:      # whatever goes wrong with one sentence, building or scoring, is written down and the next one is built
            pipeline.prompt_ring(d, p["name"], p["layout"], p["style"])      # this tier scores the ring method's views and trained splat
            rec.update(total_s=round(time.time() - t0, 1), stages_s={a[0]: round(b[1] - a[1], 1) for a, b in zip(marks, marks[1:])})
            mesh = sorted(glob.glob(f"{d}/gen_?/gen_?_mesh.glb"))
            if not mesh or not os.path.isdir(f"{work}/renders"): rec["failed"] = "no mesh or no renders"
            else: scored(rec, d, f"{d}/gen_c/gen_c.ply", mesh[0], f"{work}/renders", f"{d}/gen/images", f"{work}/poses.json", work)
        except (Exception, SystemExit) as e: rec["failed"] = f"{type(e).__name__}: {e}"[:500]
        finally:
            pipeline.run, pipeline.register, pipeline.stage = was
            for k, v in keep.items(): os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        rec.setdefault("total_s", round(time.time() - t0, 1))
        print(json.dumps(save(rec)), flush=True); out.append(rec)
    return out

def compare(a, b):
    """Rows of (world, score, a, b, change, verdict) from the line of each label and world in results.jsonl.
    A sharper picture is only called better while it still shows the paint: with ref_corr or ref_err worse, or more
    background showing, the verdict is "check" (look at crops.jpg). A run that failed gets one row saying so."""
    last = {}
    for l in open(RESULTS):
        r = json.loads(l); last[r["label"], r["world"]] = r
    rows = []
    for w in dict.fromkeys(w for l, w in last if l == a):
        if (b, w) not in last: continue
        A, B = last[a, w], last[b, w]; mine = []
        if A.get("failed") or B.get("failed"):
            rows.append((w, "failed", None, None, None, "; ".join(f"{l}: {r['failed']}" for l, r in ((a, A), (b, B)) if r.get("failed")))); continue
        for k, (way, rel, floor) in SCORES.items():
            x, y = A.get(k), B.get(k)
            if x is None or y is None: continue
            same = abs(y - x) <= max(rel * abs(x), floor)
            mine.append([w, k, x, y, y - x, "" if way == 0 else "same" if same else "better" if (y - x) * way > 0 else "worse"])
        if any(r[5] == "worse" for r in mine if r[1] in ("ref_corr", "ref_err", "bg_share")):
            for r in mine:
                if r[1] in SHARP and r[5] == "better": r[5] = "check"
        for n in WALLS:      # the views the scores above are means of, one by one
            x, y = A.get("views", {}).get(n), B.get("views", {}).get(n)
            if x and y: mine.append([w, "  " + n, x[0], y[0], y[0] - x[0], ""])
        rows += [tuple(r) for r in mine]
    return rows

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tier", choices=("geometry", "refinish", "compare", "full")); p.add_argument("names", nargs="*")
    p.add_argument("--label"); p.add_argument("--skip-agree", action="store_true"); p.add_argument("--views-from"); p.add_argument("--train", default="")
    a = p.parse_args()
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    if a.tier == "geometry":
        for w in a.names or WORLDS: print(json.dumps({"world": os.path.basename(os.path.normpath(w)), **geometry(w)}), flush=True)
    elif a.tier == "compare":
        if len(a.names) != 2: p.error("compare LABEL_A LABEL_B")
        rows = compare(*a.names); w0 = None
        print(f"{'':24}{a.names[0]:>12}{a.names[1]:>12}{'change':>10}{'':>8}   (limits per score: SCORES in quick.py; times and memory are not judged)")
        for w, k, x, y, d, verdict in rows:
            if w != w0: print(w); w0 = w
            if x is None: print(f"  {k:22}{verdict}"); continue
            print(f"  {k:22}{x:12.4g}{y:12.4g}{d:+10.3g}{(f'{100 * d / abs(x):+.1f}%' if x else ''):>8}   {verdict}")
        if not rows: print("no world has a line under both labels")
    elif not a.label: p.error("--label is needed")
    elif a.tier == "refinish":
        for w in a.names or WORLDS:
            print(json.dumps(refinish(w, a.label, skip_agree=a.skip_agree, views_from=a.views_from, train_args=shlex.split(a.train))), flush=True)
    else: full(a.label, a.names)
