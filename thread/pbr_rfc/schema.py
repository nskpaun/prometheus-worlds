"""Scene IR for the visuals-first PBR pipeline (replaces box-only boxes.json).

One representation covers both indoors (non-box rooms, vaults, recessed openings)
and outdoors (displaced terrain, instanced alpha-card foliage, 360 HDRI sky).
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PBRMaterial:
    name: str
    prompt: str
    scale_m: float = 1.0                    # physical world tile size in metres
    albedo: str = ""                        # sRGB PNG/KTX2
    normal: str = ""                        # tangent-space normal map (OpenGL +Y)
    roughness: str = ""                     # linear roughness (micro-surface variation)
    metallic: str = ""                      # linear metallic mask
    height: str = ""                        # displacement / parallax occlusion
    detail_normal_scale: float = 0.25       # tiled micro-grain so surfaces hold at 30 cm
    transmission: float = 0.0               # >0 for real refractive window/bottle glass
    ior: float = 1.5


@dataclass
class Opening:
    kind: Literal["window", "archway", "skylight", "door"]
    wall_index: int                         # index into RoomShell.footprint edges
    u_centre: float                         # metres along wall edge
    sill_m: float
    width_m: float
    height_m: float
    depth_m: float = 0.32                   # real wall thickness / splayed reveal
    arch: Literal["flat", "round", "pointed"] = "flat"
    mullions: tuple[int, int] = (1, 2)
    glazed: bool = True


@dataclass
class RoomShell:
    """Non-box indoor architecture compiled in Blender with beveled trim & real reveals."""
    footprint_xz: list[tuple[float, float]] # arbitrary polygon: L-shape, octagon, alcove, etc.
    height_m: float
    ceiling: Literal["flat", "vaulted", "coffered", "pitched", "beamed", "glazed"] = "beamed"
    bevel_mm: float = 4.0                   # never razor 90-degree edges: bevels catch specular rims
    openings: list[Opening] = field(default_factory=list)
    wall_mat: str = "wall"
    floor_mat: str = "floor"
    ceiling_mat: str = "ceiling"
    trim_mat: str = "trim"


@dataclass
class OutdoorTerrain:
    """Walkable 0..30 m PBR foreground that blends smoothly into the 360 HDRI horizon."""
    size_m: float = 36.0
    relief_m: float = 2.2
    layers: list[str] = field(default_factory=lambda: ["ground_base", "path", "accent"])
    foliage_species: list[str] = field(default_factory=list)  # branching + alpha-card trees/shrubs
    water_level_m: float | None = None


@dataclass
class AssetInstance:
    id: str
    name: str
    source: Literal["hero_3d", "kitbash", "cloth_sim", "foliage_card"]
    prompt: str                             # hero_3d generation prompt or kitbash query
    position: tuple[float, float, float]
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size_m: tuple[float, float, float] = (1.0, 1.0, 1.0)
    on_surface: str = "floor"               # floor | wall | ceiling | <parent AssetInstance.id>
    material_override: str | None = None
    mesh_path: str = ""                     # filled by assets.py (.glb with native PBR maps)


@dataclass
class PhysicalLight:
    id: str
    kind: Literal["sun", "area", "point", "spot", "portal"]
    position: tuple[float, float, float]
    direction: tuple[float, float, float] = (0.0, -1.0, 0.0)
    kelvin: int = 3200                      # warm candle/fire ~2200K vs cool moonlight ~7500K
    power_w: float = 60.0
    radius_m: float = 0.08                  # physical emitter size -> realistic soft penumbras
    volumetric_scatter: float = 0.0         # god rays / dust shaft strength


@dataclass
class Atmosphere:
    time_of_day: Literal["day", "golden_hour", "dusk", "night"] = "golden_hour"
    sky_prompt: str = ""                    # drives 360 HDRI panorama outside windows & outdoors
    env_exr: str = ""                       # generated 32-bit equirectangular HDRI
    fog_density: float = 0.008
    fog_height_m: float = 4.0
    fog_color: tuple[float, float, float] = (0.78, 0.82, 0.90)
    exposure_ev: float = 0.0


@dataclass
class SceneIR:
    sentence: str
    setting: Literal["indoor", "outdoor"]
    mood: str
    room: RoomShell | None = None
    terrain: OutdoorTerrain | None = None
    materials: dict[str, PBRMaterial] = field(default_factory=dict)
    assets: list[AssetInstance] = field(default_factory=list)
    lights: list[PhysicalLight] = field(default_factory=list)
    atmosphere: Atmosphere = field(default_factory=Atmosphere)
    camera_eye: tuple[float, float, float] = (0.0, 1.65, 2.5)
    camera_target: tuple[float, float, float] = (0.0, 1.2, 0.0)


@dataclass
class WorldBundle:
    """Outputs written for the real-time PBR viewer AND zero-cost Unreal/Godot/Blender import."""
    scene_glb: str                          # PBR meshes + UV2 lightmap coords + KTX2 textures
    collider_glb: str                       # simplified physics mesh for Rapier
    env_exr: str                            # 360 HDRI sky dome + image-based lighting
    lightmap_exr: str                       # Cycles + OIDN baked HDR indirect GI
    probes_json: str                        # parallax-corrected cubemap reflection probes
    meta_json: str                          # exposure, fog, start camera, audio mood
