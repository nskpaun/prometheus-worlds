"""Exact 360° equirectangular -> overlapping rectilinear camera rig + global depth-scale alignment.

All Stage 1 views share one camera origin and exact mathematical rotations, so adjacent SHARP
pieces agree pixel-for-pixel in their overlap regions (no COLMAP, no seams, no blank ceilings).
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class RigCam:
    name: str
    yaw_deg: float      # clockwise around +Y (0 = +Z forward)
    pitch_deg: float    # +up, -down
    fov_deg: float = 95.0
    width: int = 1024
    height: int = 1024

    def rot_c2w(self) -> np.ndarray:
        """3x3 camera-to-world rotation matrix (+X right, +Y up, -Z forward -> world +Y up)."""
        cy, sy = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        cp, sp = math.cos(math.radians(self.pitch_deg)), math.sin(math.radians(self.pitch_deg))
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float64)
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float64)
        return ry @ rx

    def focal_px(self) -> float:
        return 0.5 * self.width / math.tan(math.radians(self.fov_deg * 0.5))


# 10 overlapping cameras cover the full sphere: 6-view ring + 2 pitched up + 2 pitched down.
RIG: tuple[RigCam, ...] = (
    *(RigCam(f"ring_{k}", yaw_deg=k * 60.0, pitch_deg=0.0) for k in range(6)),
    RigCam("up_0", yaw_deg=0.0, pitch_deg=52.0, fov_deg=105.0),
    RigCam("up_1", yaw_deg=180.0, pitch_deg=52.0, fov_deg=105.0),
    RigCam("down_0", yaw_deg=30.0, pitch_deg=-52.0, fov_deg=105.0),
    RigCam("down_1", yaw_deg=210.0, pitch_deg=-52.0, fov_deg=105.0),
)


def equirect_to_rectilinear(pano: np.ndarray, cam: RigCam) -> np.ndarray:
    """Bilinearly project an HxWx3 equirectangular panorama into one rectilinear camera crop."""
    H, W = pano.shape[:2]
    f = cam.focal_px()
    xs = (np.arange(cam.width, dtype=np.float64) + 0.5 - cam.width * 0.5) / f
    ys = -(np.arange(cam.height, dtype=np.float64) + 0.5 - cam.height * 0.5) / f
    gx, gy = np.meshgrid(xs, ys)
    dirs_c = np.stack([gx, gy, -np.ones_like(gx)], axis=-1)
    dirs_c /= np.linalg.norm(dirs_c, axis=-1, keepdims=True)
    dirs_w = dirs_c @ cam.rot_c2w().T

    lon = np.arctan2(dirs_w[..., 0], -dirs_w[..., 2])            # [-pi, pi]
    lat = np.arcsin(np.clip(dirs_w[..., 1], -1.0, 1.0))          # [-pi/2, pi/2]
    u = ((lon / (2.0 * math.pi)) + 0.5) * W - 0.5
    v = (0.5 - (lat / math.pi)) * H - 0.5

    x0 = np.floor(u).astype(np.int32) % W
    x1 = (x0 + 1) % W
    y0 = np.clip(np.floor(v).astype(np.int32), 0, H - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    fx = (u - np.floor(u))[..., None].astype(np.float32)
    fy = (v - np.floor(v))[..., None].astype(np.float32)

    top = pano[y0, x0] * (1.0 - fx) + pano[y0, x1] * fx
    bot = pano[y1, x0] * (1.0 - fx) + pano[y1, x1] * fx
    return np.clip(top * (1.0 - fy) + bot * fy, 0, 255).astype(pano.dtype)


def solve_global_depth_scales(pair_log_ratios: list[tuple[int, int, float, float]], n_views: int) -> np.ndarray:
    """Solve per-view positive depth scales s_i (with mean log-scale = 0) so overlapping SHARP
    pieces agree globally: log(s_i) - log(s_j) ~= median(log(d_j / d_i)) weighted by overlap count.
    """
    A = np.zeros((n_views, n_views), dtype=np.float64)
    b = np.zeros(n_views, dtype=np.float64)
    for i, j, log_r, weight in pair_log_ratios:
        A[i, i] += weight
        A[j, j] += weight
        A[i, j] -= weight
        A[j, i] -= weight
        b[i] += weight * log_r
        b[j] -= weight * log_r
    # Anchor gauge freedom: sum(log s_i) = 0
    aug_A = np.vstack([A, np.ones((1, n_views))])
    aug_b = np.append(b, 0.0)
    log_s, *_ = np.linalg.lstsq(aug_A, aug_b, rcond=None)
    return np.exp(log_s)
