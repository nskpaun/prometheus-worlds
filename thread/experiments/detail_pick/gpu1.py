"""First GPU experiment of PLAN.md: does painting a detail view again add detail, and is it still the same wall?
  (cd thread; WORLD_DIR=evals/work/dp/ex-tower WORLD_VIEWS=12 ../spike/.venv/bin/python experiments/detail_pick/gpu1.py OUT VARIANT...)
The world copy must have a detail.json with one view (number N) and its depth map. VARIANT is NAME=rim,start,size[,view]:
rim 1/0 (keep the outer 12% as the ring painted it), start = step of 8 to begin at from the ring's paint (0: from noise),
size WxH, view = which view to paint (default the detail view N; a ring view is painted as the build paints it).
Writes OUT/P0.png (the ring's paint seen from the detail view), OUT/NAME.png and OUT/times.json. One model load."""
import sys, os, json, time, numpy as np
from PIL import Image
out = sys.argv[1]; variants = sys.argv[2:]; sys.argv = [sys.argv[0]]
import chisel_ring as C
from mflux.models.z_image.variants.controlnet.control_types import ControlSpec, ControlType
from mflux.models.z_image.variants.controlnet.z_image_turbo_controlnet import ZImageTurboControlnet
B = C.B; os.makedirs(out, exist_ok=True); style = json.load(open(f"{C.D}/gen/views.json"))["style"]
C.speed_patches(); C.painter_patches(); t = time.time(); model = ZImageTurboControlnet(quantize=8); times = {"load": round(time.time() - t, 1)}
done = {}
for i in range(C.N):
    tt, sid, _ = C.surfaces(i)
    done[i] = (np.asarray(Image.open(f"{C.D}/gen/images/{i:02d}.png").convert("RGB").resize((B.W, B.H), Image.LANCZOS), np.float32), tt, sid, C.halo(sid, C.SLIM))
for v in variants:
    name, spec = v.split("=", 1); spec = spec.split(","); rim, start = spec[0] == "1", int(spec[1]); pw, ph = map(int, spec[2].split("x")); i = int(spec[3]) if len(spec) > 3 else C.N
    os.environ["WORLD_DETAIL_RIM"] = "1" if rim else "0"
    t = time.time(); got = C.carry_over(i, {k: d for k, d in done.items() if k < i}); tc = time.time() - t
    C.KEEP.clear(); C.KEEP.update(ref=got[0], mask=got[1])
    if i >= C.N: got[2].save(f"{out}/P0.png")
    controls = [ControlSpec(type=ControlType.depth, image_path=f"{C.D}/depth/{i:02d}.png", strength=0.9)]
    t = time.time()
    if start: img = C.paint_from(model, got[2], start, 7, C.view_prompt(i, style), controls, 8, pw, ph)
    else: img = model.generate_image(seed=7, prompt=C.view_prompt(i, style), controlnet_strength=1.0, num_inference_steps=8, width=pw, height=ph, controls=controls)
    img.save(f"{out}/{name}.png"); times[name] = {"carry": round(tc, 1), "paint": round(time.time() - t, 1), "spec": spec}
    print(name, times[name], flush=True); json.dump(times, open(f"{out}/times.json", "w"))
