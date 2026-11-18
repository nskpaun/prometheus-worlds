"""Where things are for the builder: the repository's folders and its two Pythons. Importing this also lets the
modules here import the ones in thread/ (llm_boxes, shapes).
In this package a "world" is always the path of the world's folder. What a build keeps in it:
  layout.json, layout.raw.json, boxes.json, shapes/      the layout and every object's mesh (layout.py, sculpt.py)
  materials/{concept.png, style.json, surfaces.json, tex/, work/}   the style guide and every material (style.py, materials.py)
  paint/KIND/                                            each sculpted kind painted on its own mesh (paint.py)
  world.ply, world_sh.ply, collider.glb, scene.json, build.json, steps.json  the splat, its collider and its record (assemble.py)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); THREAD = os.path.dirname(HERE); REPO = os.path.dirname(THREAD)
PY = os.path.join(THREAD, ".venv/bin/python")               # numpy, scipy, Open3D, trimesh: everything but the models
PY_IMG = os.path.join(REPO, "spike/.venv/bin/python")       # MLX: the text model and the picture models
if THREAD not in sys.path: sys.path.append(THREAD)

def mat_dir(world):
    d = os.path.join(world, "materials"); os.makedirs(f"{d}/tex", exist_ok=True); return d
def mat_work(world, *sub):
    d = os.path.join(world, "materials", "work", *sub); os.makedirs(d, exist_ok=True); return d
def paint_dir(world, kind=None):
    d = os.path.join(world, "paint", kind) if kind else os.path.join(world, "paint"); os.makedirs(d, exist_ok=True); return d

# The two builds. "standard" is the cut that fits about 20 minutes on an M4 Pro; "full" spends more where it shows:
# more sculpting time (fewer small kinds fall back to plain forms), finer paint sheets, material maps of the painted
# sheets (glass and lamp glass seen in the paint, not only read from words), more part swatches.
QUALITY = {"standard": dict(sculpt_s=250, swatches=10, sheet_mp=(0.5, 0.75, 1.1), paint_maps=False),
           "full": dict(sculpt_s=330, swatches=14, sheet_mp=(0.60, 1.00, 1.60), paint_maps=True)}
def quality(): return QUALITY["full" if os.environ.get("WORLD_QUALITY") == "full" else "standard"]
