"""Native-PBR 3D models for the hero things of a place (RFC pillar 3): TRELLIS.2-4B (Microsoft, MIT) through its Apple
Silicon port (ext/trellis-mac), which gives a mesh with baked colour, metallic and roughness maps in one go: no repaint.
  ../../spike/.venv/bin/python heroes.py pictures WORLD_DIR    one clean picture of each hero kind, alone, on a plain backdrop
                                                               (Z-Image Turbo), checked against its planned proportions, cut out
  ../../ext/trellis-mac/.venv/bin/python heroes.py meshes WORLD_DIR
                                                               TRELLIS.2 on each cut-out, the pipeline loaded once for all of them
  ../.venv/bin/python heroes.py fallback WORLD_DIR             a kind TRELLIS.2 could not make: Hunyuan3D shape, or the library, or plain
Writes WORLD_DIR/pbr/heroes/KIND.glb and the mesh paths into scene_ir.json."""
import sys, os, json, time, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); THREAD = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ir

TRELLIS = os.path.abspath(os.path.join(THREAD, "..", "ext", "trellis-mac"))
PX = 768

def kinds(s):
    out = {}
    for a in s.assets:
        if a.source == "hero_3d": out.setdefault(a.kind, a)
    return out

def outline_want(w, d, h):
    return (h + 0.3 * d) / max(0.75 * (w + d), 1e-3)

def pictures(d):
    s = ir.load(os.path.join(d, "pbr", "scene_ir.json")); out = os.path.join(d, "pbr", "heroes"); os.makedirs(out, exist_ok=True)
    todo = {k: a for k, a in kinds(s).items() if not os.path.exists(os.path.join(out, f"{k}.rgba.png"))}
    if not todo: return
    sys.path.insert(0, THREAD); os.environ.setdefault("WORLD_DIR", d)
    import shapes as SH
    os.environ["HF_HUB_OFFLINE"] = "1"
    import mlx.core as mx
    mx.set_cache_limit(2 * 2**30)
    from mflux.models.z_image import ZImageTurbo
    model = ZImageTurbo(quantize=8); t0 = time.time()
    for k, a in todo.items():
        w, dp, h = a.size_m; want = outline_want(w, dp, h); coloured = bool(SH.BLUISH.search(a.prompt)); back = "bright green" if coloured else "bright blue"
        prop = SH.proportions(np.array([w, h, dp])) or ""
        best = None
        for seed, whole in ((7, False), (8, False), (9, True), (10, True)):
            if whole and best and best[0][0]: break
            extra = f". A piece {round(h * 100)} cm tall: it stands on its own legs or base; show the whole {a.name} from the side, legs and all" if whole else ""
            p = (f"Product photo of {a.prompt}{', ' + prop if prop else ''}{extra}. One single whole object, centred, entirely in frame with a margin around it, "
                 f"{'seen from eye level at a slight angle' if whole else 'seen from the front three-quarter view, slightly from above'}, isolated on a plain flat solid {back} background, no shadow, no floor, even soft lighting.")
            img = model.generate_image(seed=seed, prompt=p, num_inference_steps=9, width=PX, height=PX)
            px = np.asarray(img.image.convert("RGB"), np.float32) / 255; fg, rim = SH.framing(px)
            res, _ = SH.backdrop(px); ys, xs = np.nonzero(res > 0.16)
            got = (np.percentile(ys, 99.5) - np.percentile(ys, 0.5) + 1) / (np.percentile(xs, 99.5) - np.percentile(xs, 0.5) + 1) if len(ys) > 50 else 0
            built = got >= 0.75 * want; score = (built and rim < 0.03, built, -rim)
            if 0.02 < fg < 0.8 and rim <= 0.10 and (best is None or score > best[0]): best = (score, img)
        if not best: print(f"{k}: no clean picture, left to the fallback", flush=True); continue
        best[1].image.save(os.path.join(out, f"{k}.png"))
        if not SH.cut_out(os.path.join(out, f"{k}.png"), os.path.join(out, f"{k}.rgba.png")): print(f"{k}: could not be cut out", flush=True)
        else: print(f"hero picture: {k}", flush=True)
    print(f"{len(todo)} hero pictures in {time.time() - t0:.0f} s", flush=True)

def meshes(d):
    """TRELLIS.2 for every cut-out, one pipeline load for all: the port's own generate.py is run for each, with its
    pipeline loader made to hand back the one already loaded."""
    s = ir.load(os.path.join(d, "pbr", "scene_ir.json")); out = os.path.join(d, "pbr", "heroes")
    todo = [k for k in kinds(s) if os.path.exists(os.path.join(out, f"{k}.rgba.png")) and not os.path.exists(os.path.join(out, f"{k}.glb"))]
    if not todo: return
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    sys.path.insert(0, TRELLIS); sys.path.insert(0, os.path.join(TRELLIS, "TRELLIS.2")); os.chdir(TRELLIS)
    import generate as G
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline as P
    import trellis2.pipelines.rembg as RB
    for n in [n for n in dir(RB) if not n.startswith("_") and isinstance(getattr(RB, n), type)]:
        setattr(RB, n, lambda *a, **k: None)        # our cut-outs carry their own transparency: the background remover (a gated download) is never needed
    loaded = {}; orig = P.from_pretrained.__func__
    def once(cls, *a, **k):
        if "p" not in loaded: loaded["p"] = orig(cls, *a, **k)
        return loaded["p"]
    P.from_pretrained = classmethod(once)
    for k in todo:
        t = time.time(); sys.argv = ["generate.py", os.path.join(out, f"{k}.rgba.png"), "--output", os.path.join(out, k), "--pipeline-type", os.environ.get("TRELLIS_PIPELINE", "512"),
                                     "--texture-size", os.environ.get("TRELLIS_TEX", "1024"), "--seed", "42"]
        try: G.main()
        except SystemExit as e:
            if e.code: print(f"{k}: TRELLIS.2 stopped ({e.code})", flush=True)
        except Exception as e: print(f"{k}: TRELLIS.2 failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        print(f"hero mesh: {k} {'made' if os.path.exists(os.path.join(out, f'{k}.glb')) else 'NOT made'} in {time.time() - t:.0f} s", flush=True)

def fallback(d):
    """Hero kinds left without a TRELLIS.2 mesh: the CC0 library if it has anything like it, else a plain form."""
    import kitbash
    s = ir.load(os.path.join(d, "pbr", "scene_ir.json")); out = os.path.join(d, "pbr", "heroes"); n = 0
    for a in s.assets:
        if a.source != "hero_3d": continue
        glb = os.path.join(out, f"{a.kind}.glb")
        if os.path.exists(glb): a.mesh_path = glb; continue
        m = kitbash.match(a.name, a.size_m, least=1.0)
        if m: a.source, a.mesh_path = "kitbash", m["path"]
        else: a.source = "plain"                                          # materials.py, which runs after, gives it a material of its own
        n += 1
    ir.save(s, os.path.join(d, "pbr", "scene_ir.json"))
    if n: print(f"{n} hero things without a 3D model: taken from the library or made plain", flush=True)

if __name__ == "__main__":
    {"pictures": pictures, "meshes": meshes, "fallback": fallback}[sys.argv[1]](os.path.abspath(sys.argv[2]))
