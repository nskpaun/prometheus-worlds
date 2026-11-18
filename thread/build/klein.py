"""The picture model for materials: FLUX.2 Klein 4B through mflux, one process for a whole job list (8 bit, 4 steps,
guidance 1). setup.sh downloads the weights; at run time nothing goes to the network.
  ../../spike/.venv/bin/python klein.py JOBS.json
JOBS.json: [{"name", "out", "w", "h", "prompt", "seed", "ref" (optional: ONE reference picture; without it the picture
is made from the words alone)}]. Jobs whose out exists are skipped. Appends JOBS.json.times.jsonl."""
import sys, os, json, time
os.environ["HF_HUB_OFFLINE"] = "1"
jobs = [j for j in json.load(open(sys.argv[1])) if not os.path.exists(j["out"])]
if not jobs: sys.exit()
import mlx.core as mx
mx.set_cache_limit(2 * 2**30)
from mflux.models.common.config import ModelConfig
log = open(sys.argv[1] + ".times.jsonl", "a")
def note(d): log.write(json.dumps(d) + "\n"); log.flush()
t0 = time.time(); done = 0
def run(j, model):
    kw = dict(seed=j.get("seed", 7), prompt=j["prompt"], num_inference_steps=4, guidance=1.0, width=j["w"], height=j["h"])
    if j.get("ref"): kw["image_paths"] = [j["ref"]]
    return model.generate_image(**kw)
for kind in ("ref", "t2i"):          # pictures made from a reference first (one model), then from words alone (the other); each loaded only if needed, never both at once
    sel = [j for j in jobs if bool(j.get("ref")) == (kind == "ref")]
    if not sel: continue
    t = time.time()
    if kind == "ref":
        from mflux.models.flux2.variants import Flux2KleinEdit as Cls
    else:
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein as Cls
    model = None; mx.clear_cache()
    try: model = Cls(model_config=ModelConfig.flux2_klein_4b(), quantize=8); mx.eval(model.parameters())
    except Exception as e: sys.exit(f"The picture model (FLUX.2 Klein) could not be loaded ({type(e).__name__}: {str(e)[:200]}). If it is not downloaded, open the app again to finish setup, or run ./setup.sh.")
    note({"load": kind, "load_s": round(time.time() - t, 1)})
    for j in sel:
        mx.reset_peak_memory(); t = time.time(); img = run(j, model); done += 1
        os.makedirs(os.path.dirname(j["out"]), exist_ok=True); tmp = j["out"] + ".tmp.png"; img.save(tmp, overwrite=True); os.replace(tmp, j["out"])
        note({"name": j["name"], "w": j["w"], "h": j["h"], "kind": kind, "s": round(time.time() - t, 1), "mlx_peak_gb": round(mx.get_peak_memory() / 2**30, 2)})
        print(f"picture {done} of {len(jobs)}: {j['name']} ({time.time() - t:.0f} s)", flush=True)
note({"total_s": round(time.time() - t0, 1)})
