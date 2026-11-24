"""Full PBR material sets for every surface (RFC pillar 2): colour, normal, roughness, metallic and height maps that tile.
  ../../spike/.venv/bin/python materials.py pictures WORLD_DIR     the picture model paints each surface as a seamless tile (FLUX.2 Klein 4B)
  ../../ext/ubisoft-laforge-chord/.venv/bin/python materials.py maps WORLD_DIR
                                                                   CHORD (Ubisoft La Forge, SIGGRAPH Asia 2025) splits each tile into its
                                                                   base colour, normal, roughness and metallic maps; the height map is
                                                                   integrated from the normal map (Frankot-Chellappa)
Each process loads its model once for all surfaces and leaves memory when done. Writes WORLD_DIR/pbr/materials/NAME_*.png
and the paths into scene_ir.json. A plain thing (no 3D model, no library model) gets a material of its own here too."""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ir

PX = 1024
CHORD = os.path.abspath(os.path.join(HERE, "..", "..", "ext", "ubisoft-laforge-chord"))

def plan_materials(s):
    """Every material the scene needs a picture for: the room's (or the ground's) and one for each kind of plain thing."""
    kinds = {}
    for a in s.assets:
        if a.source in ("plain", "cloth_sim") and a.kind not in s.materials and a.kind not in kinds:
            what = f"the fabric of {a.prompt}, woven textile, soft folds" if a.source == "cloth_sim" else f"the surface material of {a.prompt}"
            kinds[a.kind] = ir.PBRMaterial(name=a.kind, prompt=what, scale_m=0.6 if a.source == "cloth_sim" else 1.0, rough=0.85 if a.source == "cloth_sim" else 0.6)
        if a.source in ("plain", "cloth_sim"): a.material = a.kind
    s.materials.update(kinds)
    return [m for m in s.materials.values() if m.transmission < 0.5]

def tileable(a, band=128):
    """A picture made to tile. Rolled half way, its edges meet in a cross through the middle; along each arm of the cross
    the picture is blended with a copy shifted a quarter along that arm only (so the copy's own seam lies away from it),
    and in the middle with a copy shifted both ways. Rolled back. Works on any channels."""
    h, w = a.shape[:2]; r = np.roll(np.roll(a, h // 2, 0), w // 2, 1)
    sy = np.roll(r, h // 4, 0); sx = np.roll(r, w // 4, 1); sb = np.roll(sy, w // 4, 1)
    yy, xx = np.mgrid[0:h, 0:w]
    st = lambda d: (lambda t: t * t * (3 - 2 * t))(np.clip(d / band, 0, 1)).astype(np.float32)
    ty, tx = st(np.abs(yy - h // 2)), st(np.abs(xx - w // 2))
    if a.ndim == 3: ty, tx = ty[..., None], tx[..., None]
    out = r * ty * tx + sy * (1 - ty) * tx + sx * ty * (1 - tx) + sb * (1 - ty) * (1 - tx)
    return np.roll(np.roll(out, -(h // 2), 0), -(w // 2), 1)

def pictures(d):
    s = ir.load(os.path.join(d, "pbr", "scene_ir.json")); todo = plan_materials(s); ir.save(s, os.path.join(d, "pbr", "scene_ir.json"))
    out = os.path.join(d, "pbr", "materials"); os.makedirs(out, exist_ok=True)
    todo = [m for m in todo if not os.path.exists(os.path.join(out, f"{m.name}_tile.png"))]
    if not todo: return
    os.environ["HF_HUB_OFFLINE"] = "1"
    import mlx.core as mx
    mx.set_cache_limit(2 * 2**30)
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    from PIL import Image
    model = Flux2Klein(model_config=ModelConfig.flux2_klein_4b(), quantize=8); t0 = time.time()
    for m in todo:
        p = (f"Seamless tileable texture, a flat surface seen straight on from above, filling the whole frame edge to edge: {m.prompt}. "
             f"About {m.scale_m:g} m across. Even soft diffuse light, no shadows, no perspective, no objects, no border, photographic, physically based material reference.")
        img = np.asarray(model.generate_image(seed=3, prompt=p, num_inference_steps=4, width=PX, height=PX, guidance=1.0).image.convert("RGB"), np.float32)
        Image.fromarray(tileable(img).clip(0, 255).astype(np.uint8)).save(os.path.join(out, f"{m.name}_tile.png"))
        print(f"material picture: {m.name}", flush=True)
    print(f"{len(todo)} material pictures in {time.time() - t0:.0f} s", flush=True)

def height_from_normal(n):
    """Frankot-Chellappa: the height field whose slopes best match a tangent-space normal map (periodic, so it tiles). -> 0..1"""
    nz = np.clip(n[..., 2], 0.15, 1.0); p = -n[..., 0] / nz; q = n[..., 1] / nz           # OpenGL normal: +y up the picture
    h, w = p.shape; wy, wx = np.meshgrid(np.fft.fftfreq(h) * 2 * np.pi, np.fft.fftfreq(w) * 2 * np.pi, indexing="ij")
    den = wx ** 2 + wy ** 2; den[0, 0] = 1.0
    Z = (-1j * wx * np.fft.fft2(p) - 1j * wy * np.fft.fft2(q)) / den; Z[0, 0] = 0          # p, q: the height's slope along the picture's columns and rows
    z = np.real(np.fft.ifft2(Z)); lo, hi = np.percentile(z, [1, 99]); return np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)

def maps(d):
    s = ir.load(os.path.join(d, "pbr", "scene_ir.json")); out = os.path.join(d, "pbr", "materials")
    todo = [m for m in s.materials.values() if m.transmission < 0.5 and os.path.exists(os.path.join(out, f"{m.name}_tile.png"))]
    need = [m for m in todo if not os.path.exists(os.path.join(out, f"{m.name}_normal.png"))]
    if need:
        sys.path.insert(0, CHORD)
        import torch
        from omegaconf import OmegaConf
        from torchvision.transforms import v2
        from huggingface_hub import hf_hub_download
        from chord import ChordModel
        from chord.io import read_image, load_torch_file
        from PIL import Image
        os.environ["HF_HUB_OFFLINE"] = "1"; dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); t0 = time.time()
        model = ChordModel(OmegaConf.load(os.path.join(CHORD, "config", "chord.yaml")))
        model.load_state_dict(load_torch_file(hf_hub_download(repo_id="Ubisoft/ubisoft-laforge-chord", filename="chord_v1.safetensors"))); model.eval().to(dev)
        for m in need:
            x = v2.Resize(size=(1024, 1024), antialias=True)(read_image(os.path.join(out, f"{m.name}_tile.png"))).unsqueeze(0).to(dev)
            with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16): o = model(x)      # the model's weights are bfloat16
            got = {}
            for k, v in o.items():                                       # basecolor and normal are B x 3 x H x W; roughness and metalness B x H x W
                if not torch.is_tensor(v): continue
                v = v[0].float().clamp(0, 1).cpu().numpy(); got[k] = v.transpose(1, 2, 0) if v.ndim == 3 else v[..., None]
            col = got.get("basecolor"); nor = got.get("normal"); rou = got.get("roughness"); met = got.get("metalness", got.get("metallic"))
            if col is None or nor is None: print(f"CHORD gave no maps for {m.name}: its picture is used as colour alone", flush=True); continue
            save = lambda a, suffix: Image.fromarray((np.clip(tileable(a), 0, 1) * 255 + 0.5).astype(np.uint8).squeeze()).save(os.path.join(out, f"{m.name}_{suffix}.png"))
            save(col[..., :3], "albedo"); save(nor[..., :3], "normal")
            if rou is not None: save(rou[..., 0], "roughness")
            if met is not None: save(met[..., 0], "metallic")
            save(height_from_normal(nor[..., :3] * 2 - 1), "height")
            print(f"PBR maps: {m.name}", flush=True)
        print(f"{len(need)} materials split into PBR maps in {time.time() - t0:.0f} s", flush=True)
    for m in todo:
        for k in ("albedo", "normal", "roughness", "metallic", "height"):
            p = os.path.join(out, f"{m.name}_{k}.png"); setattr(m, k, p if os.path.exists(p) else "")
        if not m.albedo: m.albedo = os.path.join(out, f"{m.name}_tile.png")
    ir.save(s, os.path.join(d, "pbr", "scene_ir.json"))

if __name__ == "__main__":
    {"pictures": pictures, "maps": maps}[sys.argv[1]](os.path.abspath(sys.argv[2]))
