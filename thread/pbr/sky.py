"""The 360 degree sky and far surroundings as a true HDR panorama (RFC pillar 2).
  ../../spike/.venv/bin/python sky.py WORLD_DIR        (the picture model's environment; one process, gone when done)
The picture: FLUX.2 Klein base 9B with a 360 degree panorama LoRA (crafiq/flux-2-klein-9b-360-panorama-lora, Apache 2.0),
2048 x 1024 equirectangular, its two ends blended so it closes round. The lift to HDR: the picture is made linear, the
sky lifted to the brightness of a real sky at that hour, and the sun (or the moon) put back as a disc of its physical
radiance where the scene's sun is, so that Cycles casts real sunlight from it and the viewer's bloom catches it. Written
as WORLD_DIR/pbr/env.npy (float32, linear Rec.709, H x W x 3, row 0 the zenith, column 1024 looking along +x as Blender's
and the viewer's lookups both have it; no sun disc: Blender's sun lamp is the sun) and env_view.npy (the same with the
disc, for the viewer to show), and env_ldr.png (the painted picture)."""
import sys, os, json, math, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ir

LORA = ("crafiq/flux-2-klein-9b-360-panorama-lora", None)
W, H = 2048, 1024
SKY_LIFT = {"day": 0.38, "golden_hour": 0.22, "dusk": 0.08, "night": 0.004}           # a mid-sky pixel's radiance, in the units of Blender's sun strength: a clear sky lights a surface about a quarter as much as the sun
SUN_RAD = {"day": 3.0e3, "golden_hour": 2.0e3, "dusk": 800.0, "night": 60.0}          # the disc as the viewer shows it: bright enough to glow under bloom, not to white out the screen (the sun lamp does the lighting)
WORDS = {"day": "clear bright daylight, a few soft white clouds, sunlit", "golden_hour": "golden hour, low warm sun, long shadows, glowing clouds",
         "dusk": "dusk, the sun just set, orange and violet sky at the horizon, deep blue above, the first lights coming on",
         "night": "night, deep blue night sky with stars and a bright moon, warm lit windows and street lamps in the distance"}

def lora_file():
    """The LoRA's weights from the local cache (setup.sh fetched them; nothing goes to the network here)."""
    from huggingface_hub import snapshot_download
    d = snapshot_download(LORA[0], local_files_only=True)
    return next(os.path.join(d, x) for x in sorted(os.listdir(d)) if x.endswith(".safetensors"))

def paint(prompt, seed=21):
    os.environ["HF_HUB_OFFLINE"] = "1"
    import mlx.core as mx
    mx.set_cache_limit(2 * 2**30)
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    m = Flux2Klein(model_config=ModelConfig.flux2_klein_base_9b(), quantize=8, lora_paths=[lora_file()], lora_scales=[1.0])
    img = m.generate_image(seed=seed, prompt=prompt, num_inference_steps=int(os.environ.get("SKY_STEPS", "20")), width=1536, height=768, guidance=4.0)
    from PIL import Image
    return np.asarray(img.image.convert("RGB").resize((W, H), Image.LANCZOS), np.float32) / 255.0      # painted at 1536 x 768 (a third of the time), used at 2048 x 1024

def seam(a, band=96):
    """Blend across the wrap: the picture rolled half way round so the seam is in the middle, the seam band replaced by a
    cross-fade of what lies either side of it, and rolled back."""
    r = np.roll(a, W // 2, axis=1); c = W // 2; L = r[:, c - band:c]; R = r[:, c:c + band]
    t = np.linspace(0, 1, 2 * band, dtype=np.float32)[None, :, None]
    left = np.concatenate([L, L[:, ::-1]], 1); right = np.concatenate([R[:, ::-1], R], 1)
    r[:, c - band:c + band] = left * (1 - t) + right * t
    return np.roll(r, -(W // 2), axis=1)

def linear(x): return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def direction(r, c):
    """Pixel -> unit direction in Blender's frame (z up; column W/2 looks along +x; columns grow clockwise from above, as
    Blender's equirectangular lookup has it: u = -atan2(y, x) / 2pi + 0.5)."""
    lon = (0.5 - (c + 0.5) / W) * 2 * math.pi; lat = (0.5 - (r + 0.5) / H) * math.pi
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat) * np.ones_like(lon)], -1)

def lift(ldr, hour, sun_dir, disc_too=False):
    """LDR picture -> HDR radiance: the sky (everything above the horizon) scaled to a real sky's brightness at that hour,
    the ground below kept as lit by it, a glow round the sun, and (disc_too, for the picture the viewer shows) the sun or
    moon itself as a disc of physical radiance. The world Blender lights with has no disc: its sun lamp is the sun."""
    lin = linear(ldr); rr, cc = np.mgrid[0:H, 0:W]; dirs = direction(rr, cc)
    up = np.clip(dirs[..., 2], 0, 1)[..., None]; above = dirs[..., 2] > 0
    mid = float(np.median((lin[above] @ [0.2126, 0.7152, 0.0722]))) or 1e-3
    k = SKY_LIFT.get(hour, 1.0) / mid
    hdr = lin * k * (0.75 + 0.5 * up)                                                     # a real sky is brighter overhead and near the sun than a photo shows
    s = np.asarray(sun_dir, np.float64); s /= np.linalg.norm(s) or 1.0
    cosang = dirs @ s; ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    disc = ang < (0.9 if hour != "night" else 0.7); glow = np.exp(-ang / (6.0 if hour in ("golden_hour", "dusk") else 3.0))
    tint = {"day": [1.0, 0.97, 0.92], "golden_hour": [1.0, 0.78, 0.5], "dusk": [1.0, 0.55, 0.3], "night": [0.75, 0.85, 1.0]}.get(hour, [1, 1, 1])
    if s[2] > -0.05:
        hdr += (glow[..., None] * SKY_LIFT.get(hour, 1.0) * 4.0 * np.array(tint))
        if disc_too: hdr[disc] = SUN_RAD.get(hour, 1e4) * np.array(tint)
    return hdr.astype(np.float32)

def prompt_for(s):
    a = s.atmosphere; seen = a.sky_prompt or ("open landscape" if s.setting == "outdoor" else "a quiet street with buildings and trees")
    lead = "Equirectangular 360 panorama"
    if s.setting == "indoor": return f"{lead}, outdoors at eye level, {seen}, {WORDS.get(a.time_of_day, WORDS['day'])}, everything at a distance, nothing close to the camera, photographic, natural colour"
    return f"{lead}, {seen}, the far horizon all round, {WORDS.get(a.time_of_day, WORDS['day'])}, the ground near the camera plain and open, photographic, natural colour"

def main(d):
    s = ir.load(os.path.join(d, "pbr", "scene_ir.json")); out = os.path.join(d, "pbr", "env.npy")
    if os.path.exists(out): return print("the sky is already painted")
    t = time.time(); raw = os.path.join(d, "pbr", "env_ldr.png")
    if os.path.exists(raw):
        from PIL import Image; ldr = np.asarray(Image.open(raw).convert("RGB"), np.float32) / 255.0
    else:
        ldr = seam(paint(prompt_for(s)))
        from PIL import Image; Image.fromarray((ldr * 255 + 0.5).astype(np.uint8)).save(raw)
    hdr = lift(ldr, s.atmosphere.time_of_day, s.atmosphere.sun_dir); np.save(out, hdr)
    np.save(os.path.join(d, "pbr", "env_view.npy"), lift(ldr, s.atmosphere.time_of_day, s.atmosphere.sun_dir, disc_too=True))
    s.atmosphere.env_exr = os.path.join(d, "pbr", "env.exr"); ir.save(s, os.path.join(d, "pbr", "scene_ir.json"))
    print(f"sky painted and lifted to HDR in {time.time() - t:.0f} s (peak radiance {hdr.max():.3g})", flush=True)

if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]))
