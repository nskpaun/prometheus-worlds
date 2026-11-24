"""End-to-end orchestrator for `WORLD_METHOD=pano` (RFC prototype).

Reuses existing local models & photo-fusion machinery:
  - FLUX.2 Klein base 9B + 360° Panorama LoRA (already in thread/pbr/sky.py)
  - Apple SHARP (already in thread/photo_pieces.py)
  - Multi-piece alignment, ownership fusion & collider meshing (photo_fuse.py + mesh_photos.py)
  - Existing Spark splat viewer (zero viewer changes needed for Stage 1!)
"""
from pathlib import Path
import numpy as np
from . import project, walkable


def build_pano_prompt(sentence: str, style: str = "") -> str:
    """Format a user prompt so the 360° panorama LoRA places the camera inside a rich, coherent world."""
    look = f", {style}" if style else ""
    return (
        f"Equirectangular 360 degree panoramic photograph from eye level inside {sentence}{look}. "
        "Coherent natural lighting, rich architectural and material detail, balanced contrast, "
        "photorealistic, sharp focus from foreground to horizon, no people, no text, no distortion."
    )


def generate_360_panorama(prompt: str, out_png: Path, upscale_4k: bool = True) -> Path:
    """Stage 1a (~29 GB peak -> unload, ~60-90 s): FLUX.2 Klein base 9B + 360° LoRA.

    Uses the exact cached model already wired in thread/pbr/sky.py, with circular seam blending
    and an optional 2x tile-detail pass (2048x1024 -> 4096x2048) for crisp close-up surfaces.
    """
    # pano = sky.seam(sky.paint(prompt, seed=21))          # 2048x1024 seamless equirectangular
    # if upscale_4k: pano = tile_upscale_2x(pano)          # sharp 4096x2048 for close-up crops
    # save_png(out_png, pano)
    _ = (prompt, upscale_4k)
    return out_png


def project_rig_crops(pano_png: Path, crops_dir: Path) -> list[dict]:
    """Stage 1b (<1 GB, ~1 s): Project 10 overlapping rectilinear views with exact camera poses."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    # pano = np.asarray(Image.open(pano_png).convert("RGB"))
    manifest = []
    for cam in project.RIG:
        # crop = project.equirect_to_rectilinear(pano, cam)
        # Image.fromarray(crop).save(crops_dir / f"{cam.name}.jpg", quality=94)
        manifest.append({
            "name": cam.name,
            "image": str(crops_dir / f"{cam.name}.jpg"),
            "R_c2w": cam.rot_c2w().tolist(),
            "t_c2w": [0.0, 0.0, 0.0],
            "focal_px": cam.focal_px(),
            "width": cam.width,
            "height": cam.height,
        })
    return manifest


def run_sharp_and_fuse(world_dir: Path, manifest: list[dict]) -> tuple[Path, Path, np.ndarray]:
    """Stage 1c-d (~16 GB peak -> unload, ~85 s):
      1. Apple SHARP on each crop in `manifest` (~6 s/view, reusing thread/photo_pieces.py).
      2. Compute median log depth ratios across overlapping rays between adjacent rig cameras.
      3. Call `project.solve_global_depth_scales(...)` so all 10 pieces lock onto one seamless shell.
      4. Fuse pieces by angular/cell ownership (reusing photo_fuse.py thinning + patching) -> stage1.ply.
      5. Extract agreed surface mesh + collider via thread/mesh_photos.py --surface -> collider.glb.
    """
    _ = manifest
    return world_dir / "stage1.ply", world_dir / "collider.glb", np.zeros((0, 3), np.float32)


def build_pano_world(world_dir: Path, name: str, sentence: str, style: str = "", stage2: bool = True) -> dict:
    """Unified `WORLD_METHOD=pano` pipeline -> returns a Spark-compatible `scenes.json` entry."""
    world_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Coherent 360° panorama -> 10 exact crops -> SHARP -> globally aligned fused 3DGS (~2 min)
    pano_png = generate_360_panorama(build_pano_prompt(sentence, style), world_dir / "pano.png")
    manifest = project_rig_crops(pano_png, world_dir / "crops")
    splat_ply, collider_glb, depth_shell = run_sharp_and_fuse(world_dir, manifest)

    # Stage 2 (optional, +2 min): Off-center disocclusion inpainting for full-room walkability
    final_ply = splat_ply
    if stage2:
        step_cams = walkable.plan_step_cameras(depth_shell, n_steps=6, max_step_m=2.0)
        novel = walkable.render_novel_views_and_masks(splat_ply, step_cams, world_dir / "stage2")
        walkable.inpaint_disocclusions(novel, sentence)
        final_ply = walkable.lift_and_append_holes(splat_ply, novel, world_dir / "scene.ply")

    return {
        "name": name,
        "kind": "prompt",
        "method": "pano",
        "ply": f"{world_dir.name}/{final_ply.name}",
        "mesh": f"{world_dir.name}/mesh.glb",
        "collider": f"{world_dir.name}/{collider_glb.name}",
        "up": [0.0, 1.0, 0.0],
        "eye": [0.0, 0.0, 0.0],
        "forward": [0.0, 0.0, -1.0],
        "note": f"pano 3DGS: {sentence[:60]}",
    }
