"""The picture model for object paint: FLUX.2 Klein 4B edits through mflux, one process for a whole job list (8 bit,
guidance 1), each job one reference picture and a start picture. setup.sh downloads the weights; nothing goes to the network.
  ../../spike/.venv/bin/python klein_paint.py JOBS.json
JOBS.json: [{"name", "w", "h", "prompt", "ref", "out", "seed", "init", "steps", "strength", "sigma0"}]. A job whose out
exists is skipped. Appends JOBS.json.times.jsonl.
THIS FILE REACHES INTO mflux 0.24.0 (runtime/paint.lock pins it): it replaces the edit model's starting latents, lifts its
limit on the reference picture's size for the start picture, and overrides one value of the scheduler's noise schedule.
The asserts below fail loudly if a later mflux changes those shapes; move this file and the pin together."""
import sys, os, json, time
os.environ["HF_HUB_OFFLINE"] = "1"
jobs = [j for j in json.load(open(sys.argv[1])) if not os.path.exists(j["out"])]
if not jobs: sys.exit()
import mlx.core as mx
mx.set_cache_limit(2 * 2**30)
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2KleinEdit
log = open(sys.argv[1] + ".times.jsonl", "a")
def note(d): log.write(json.dumps(d) + "\n"); log.flush()
t0 = time.time()
try: model = Flux2KleinEdit(model_config=ModelConfig.flux2_klein_4b(), quantize=8); mx.eval(model.parameters())
except Exception as e: sys.exit(f"The picture model (FLUX.2 Klein) could not be loaded ({type(e).__name__}: {str(e)[:200]}). If it is not downloaded, open the app again to finish setup, or run ./setup.sh.")
note({"load_s": round(time.time() - t0, 1)})
from PIL import Image
# A job with "init" (a picture of the answer's own size) starts from that picture under noise instead of from pure noise (SDEdit): "strength" of the
# "steps" are skipped, so what is where in the picture (each view's place, camera and outline) is given, and the model only repaints it. mflux's edit
# variant skips the steps for image_strength but still starts from pure noise; the start is mixed here.
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers as _H
from mflux.models.common.config.config import Config as _Cfg
_orig = _H.prepare_generation_latents; CUR = {}
def _start(*, seed, height, width):
    lat, ids, lh, lw = _orig(seed=seed, height=height, width=width)
    if CUR.get("init"):
        import mflux.models.flux2.variants.edit.flux2_klein_edit_helpers as _M
        keep = _M.MAX_REFERENCE_AREA; _M.MAX_REFERENCE_AREA = 10**9          # the start picture is the answer's own size, however large
        try: x0, _ = _H.prepare_reference_image_conditioning(vae=model.vae, tiling_config=model.tiling_config, image_paths=[CUR["init"]], batch_size=1)
        finally: _M.MAX_REFERENCE_AREA = keep
        assert x0.shape == lat.shape, (x0.shape, lat.shape); s = CUR["sigma"]
        lat = s * lat + (1 - s) * x0.astype(lat.dtype)
    return lat, ids, lh, lw
_H.prepare_generation_latents = staticmethod(_start)
# "sigma0": how much noise the start picture is put under (1 = pure noise, nothing held). mflux's own first kept step of 4 sits near 0.9 to 0.93, which
# also freezes the start picture's colours; a higher value is written into the schedule itself, so the model is told the noise level it really gets.
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler as _S
def _ovr(arr, scale):
    if CUR.get("sigma0") and CUR.get("k0") is not None:
        import numpy as _np
        a = _np.array(arr); a[CUR["k0"]] = CUR["sigma0"] * scale; return mx.array(a)
    return arr
_S.sigmas = property(lambda self: _ovr(self._sigmas, 1.0)); _S.timesteps = property(lambda self: _ovr(self._timesteps, self.num_train_timesteps))
for n, j in enumerate(jobs):
    mx.reset_peak_memory(); t = time.time()
    steps = j.get("steps", 4); st = j.get("strength") if j.get("init") else None; CUR.clear()
    if st:
        cfg = lambda: _Cfg(model_config=ModelConfig.flux2_klein_4b(), num_inference_steps=steps, height=j["h"], width=j["w"], guidance=1.0, image_path=j["ref"], image_strength=st, scheduler="flow_match_euler_discrete")
        c = cfg()
        if j.get("sigma0"): CUR.update(sigma0=float(j["sigma0"]), k0=c.init_time_step); c = cfg()
        CUR.update(init=j["init"], sigma=float(c.scheduler.sigmas[c.init_time_step]))
    img = model.generate_image(seed=j.get("seed", 7), prompt=j["prompt"], num_inference_steps=steps, guidance=1.0, width=j["w"], height=j["h"], image_paths=[j["ref"]], image_strength=st)
    os.makedirs(os.path.dirname(j["out"]), exist_ok=True); tmp = j["out"] + ".tmp.png"; img.save(tmp, overwrite=True); os.replace(tmp, j["out"])
    note({"name": j["name"], "sigma": CUR.get("sigma"), "w": j["w"], "h": j["h"], "s": round(time.time() - t, 1), "mlx_peak_gb": round(mx.get_peak_memory() / 2**30, 2)})
    print(f"painted {n + 1} of {len(jobs)}: {j['name']} ({time.time() - t:.0f} s)", flush=True)
note({"total_s": round(time.time() - t0, 1)})
