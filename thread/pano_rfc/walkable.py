"""Stage 2 walkability: off-center disocclusion rendering, masked inpainting, and SHARP gap-fill.

Stage 1 gives a seamless 360° 3DGS core around the origin. Stepping 1.5–2.2 m away reveals
occluded regions behind foreground furniture/pillars. This module fills ONLY those disocclusions
without modifying the coherent Stage 1 geometry.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class StepCamera:
    name: str
    position: tuple[float, float, float]    # world metres (+Y up)
    look_at: tuple[float, float, float]
    fov_deg: float = 75.0
    width: int = 1024
    height: int = 768


def plan_step_cameras(depth_shell_xyz: np.ndarray, n_steps: int = 6, max_step_m: float = 2.0) -> list[StepCamera]:
    """Pick off-center eye positions inside the clear walkable volume of the Stage 1 depth shell.

    Each camera steps 45% of the free distance along evenly spaced azimuths (clamped to max_step_m)
    and looks back across foreground objects to expose their occluded rear/side surfaces.
    """
    cams: list[StepCamera] = []
    # Eye-band points near camera height (|y| < 0.6 m) measure free horizontal radius per direction.
    eye_band = depth_shell_xyz[np.abs(depth_shell_xyz[:, 1]) < 0.6]
    az_pts = np.arctan2(eye_band[:, 0], -eye_band[:, 2]) if len(eye_band) else np.zeros(1)
    r_pts = np.hypot(eye_band[:, 0], eye_band[:, 2]) if len(eye_band) else np.full(1, 3.0)

    for k in range(n_steps):
        az = -np.pi + 2.0 * np.pi * k / n_steps
        sector = np.abs((az_pts - az + np.pi) % (2.0 * np.pi) - np.pi) < np.radians(20.0)
        free_r = float(np.percentile(r_pts[sector], 10)) if sector.any() else 2.5
        step = min(max_step_m, max(0.6, 0.45 * (free_r - 0.5)))
        pos = (step * float(np.sin(az)), 0.0, -step * float(np.cos(az)))
        # Look slightly past the origin so foreground obstacles between origin and wall reveal their backs.
        target = (-0.5 * pos[0], -0.1, -0.5 * pos[2])
        cams.append(StepCamera(name=f"step_{k}", position=pos, look_at=target))
    return cams


def render_novel_views_and_masks(splat_ply: Path, cams: list[StepCamera], out_dir: Path) -> list[dict]:
    """Render each off-center StepCamera from the Stage 1 splat -> RGB + binary disocclusion mask.

    A pixel is marked as a disocclusion hole where accumulated alpha < 0.85 or depth gradient tears.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    views = []
    for cam in cams:
        # In production: rasterize Stage 1 splat (Metal/gsplat/Spark headless) -> rgb.png, alpha.npy, depth.npy.
        # hole_mask = dilate(alpha < 0.85, px=6)
        views.append({
            "cam": cam,
            "rgb": str(out_dir / f"{cam.name}_rgb.png"),
            "mask": str(out_dir / f"{cam.name}_hole_mask.png"),
            "depth": str(out_dir / f"{cam.name}_depth.npy"),
            "inpainted": str(out_dir / f"{cam.name}_filled.png"),
        })
    return views


def inpaint_disocclusions(views: list[dict], sentence: str) -> None:
    """Inpaint ONLY masked disocclusion holes while locking existing Stage 1 pixels.

    Target open local models (~15 GB, unloaded after batch):
      - FLUX.2 Klein 4B masked inpainting (already cached on Mac), OR
      - Difix3D+ (1-step diffusion restoration specifically trained for novel-view 3DGS cleanup).
    """
    _ = (views, sentence)
    # model = load_local_inpaint_or_difix3d()
    # for v in views:
    #     if hole_fraction(v["mask"]) < 0.015: continue
    #     filled = model.inpaint(v["rgb"], mask=v["mask"], prompt=sentence, lock_unmasked=True)
    #     save_png(v["inpainted"], filled)


def lift_and_append_holes(splat_ply: Path, views: list[dict], out_ply: Path) -> Path:
    """Run Apple SHARP on each inpainted off-center view, fit depth scale to the unmasked pixels,
    and append ONLY the newly revealed hole-mask Gaussians into the fused Stage 1 splat.
    """
    # For each view:
    #   1. piece = sharp_predict(v["inpainted"], focal=v["cam"].focal_px())
    #   2. scale = median(existing_depth[~mask] / piece_depth[~mask])   # exact metric lock
    #   3. keep = piece[mask & ~depth_edge(piece)]                      # never overwrite Stage 1 core
    #   4. transform `keep` to world frame via v["cam"] and merge into `out_ply`
    _ = (splat_ply, views)
    return out_ply
