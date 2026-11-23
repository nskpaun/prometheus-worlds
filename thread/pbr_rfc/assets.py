"""Three-tier PBR asset, material, and 360 HDRI synthesis (RFC prototype).

Replaces:
  - diffuse-only FLUX swatches (thread/build/materials.py)
  - single-view Hunyuan3D 2.0 mini + 4-step sheet repainting (shapes.py + paint.py)
  - flat constant-color window quads (SKYC[tm])

Each function runs in its own isolated subprocess in pipeline.py so only ONE large
open model sits in unified memory at a time (peak <= 32 GB on a 64 GB Mac).
"""
from pathlib import Path
from .schema import AssetInstance, Atmosphere, PBRMaterial, SceneIR


# ---------------------------------------------------------------------------
# 1. Seamless PBR material sets (Albedo + Normal + Roughness + Metallic + Height)
# ---------------------------------------------------------------------------

def synthesize_pbr_materials(scene: SceneIR, out_dir: Path) -> None:
    """Generate tileable PBR texture maps for every architectural & terrain surface.

    Target open local model (~14 GB): CHORD / MatFuse / Material-Anything-class PBR
    synthesizer (or FLUX albedo + Lotus/Marigold intrinsic normal+roughness+depth).
    """
    mat_dir = out_dir / "materials"
    mat_dir.mkdir(parents=True, exist_ok=True)

    # Pseudocode for local MLX/MPS PBR model loaded once for the whole batch:
    #   model = load_local_pbr_generator("chord-pbr-1k", device="mps")
    for key, mat in scene.materials.items():
        if mat.transmission > 0.5:
            continue  # clear glass uses analytical IOR + subtle procedural smudge roughness
        stem = mat_dir / key
        # maps = model.generate_seamless(
        #     prompt=f"{mat.prompt}, seamless orthographic tile, {mat.scale_m:g}m square, even diffuse light",
        #     resolution=1024,
        #     channels=("albedo", "normal", "roughness", "metallic", "height"),
        # )
        # save_pbr_maps(stem, maps)
        mat.albedo = f"{stem}_albedo.png"
        mat.normal = f"{stem}_normal.png"
        mat.roughness = f"{stem}_roughness.png"
        mat.metallic = f"{stem}_metallic.png"
        mat.height = f"{stem}_height.png"


# ---------------------------------------------------------------------------
# 2. 360-degree HDRI sky & exterior vista (drives IBL + real window views)
# ---------------------------------------------------------------------------

def synthesize_hdri_environment(atmo: Atmosphere, out_dir: Path) -> str:
    """Generate a seamless 2:1 equirectangular panorama and lift it to 32-bit HDR (.exr).

    Solves two current visual failures at once:
      - Indoor windows look out onto a real sky/city/forest vista with parallax instead of flat quads.
      - Outdoor worlds get physically consistent sky lighting + horizon matching the height fog.
    """
    env_path = out_dir / "env.exr"
    # 1. Equirectangular panorama diffusion (~16 GB, e.g. FLUX-pano / SDXL-360 LoRA, circular padding).
    #    ldr_pano = pano_model.generate(
    #        prompt=f"360 equirectangular HDRI panorama, {atmo.sky_prompt}, {atmo.time_of_day}",
    #        width=4096, height=2048, seamless_x=True,
    #    )
    # 2. Inverse tone-mapping (e.g. DiffusionLight / local HDR-lift CNN) recovers true sun/sky
    #    dynamic range (EV +8..+14 on the solar disc) so Cycles casts real directional sunlight.
    #    hdr_pano = lift_ldr_to_hdr_exr(ldr_pano, time_of_day=atmo.time_of_day)
    #    write_exr(env_path, hdr_pano)
    atmo.env_exr = str(env_path)
    return atmo.env_exr


# ---------------------------------------------------------------------------
# 3. Full-scale native-PBR 3D hero assets + local CC0 micro-kitbashing
# ---------------------------------------------------------------------------

def resolve_assets(scene: SceneIR, out_dir: Path, kitbash_root: Path) -> None:
    """Populate AssetInstance.mesh_path with crisp PBR glTFs.

    Budget split:
      - hero_3d (6-10 unique kinds): Full-scale open native-3D PBR model (e.g. TRELLIS-2 or
        Hunyuan3D 2.1+ full, ~24-32 GB). Outputs clean UV-unwrapped mesh + multi-view-consistent
        albedo/roughness/metallic/normal maps directly — no clay-sheet img2img or KD-tree patching.
      - kitbash / foliage_card: Curated local CC0 library (Poly Haven / AmbientCG trim, leaf-card
        trees, books, glassware, stones) matched via local embedding index and tint-matched to palette.
      - cloth_sim: Procedural high-res mesh draped in Blender (curtains, rugs, table runners).
    """
    hero_dir = out_dir / "heroes"
    hero_dir.mkdir(parents=True, exist_ok=True)

    heroes = [a for a in scene.assets if a.source == "hero_3d"]
    if heroes:
        # Load full-scale native-3D PBR model ONCE, generate all hero kinds, then exit process:
        #   img_gen = load_local_flux(device="mps")        # 1024px clean studio reference
        #   gen_3d  = load_native_pbr_3d("trellis-2-pbr")  # geometry + native UV PBR maps
        for asset in heroes:
            glb = hero_dir / f"{asset.id}.glb"
            # ref = img_gen(f"{asset.prompt}, isolated 3/4 studio photograph, neutral backdrop")
            # gen_3d.generate_pbr_glb(ref, out_path=glb, target_faces=35_000, tex_res=2048)
            asset.mesh_path = str(glb)

    for asset in scene.assets:
        if asset.source in ("kitbash", "foliage_card"):
            # Local cosine search over pre-indexed CC0 library (0 GB GPU, milliseconds):
            #   asset.mesh_path = str(match_cc0_asset(kitbash_root, asset.prompt, asset.size_m))
            asset.mesh_path = str(kitbash_root / f"{asset.id}.glb")
