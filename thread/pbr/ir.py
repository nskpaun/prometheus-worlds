"""The scene description of the visuals-first pipeline (WORLD_METHOD=pbr; docs/pbr-method.md, pillar 1).
One form for indoors (a room of any outline, its ceiling, real openings) and outdoors (shaped ground, scattered
foliage, a 360 degree sky). Written as WORLD_DIR/pbr/scene_ir.json; read by every later step, Blender included.
Frame: metres, Blender's (x right, y back, z up), the floor's middle at the origin."""
from dataclasses import dataclass, field, asdict, fields
import json

@dataclass
class PBRMaterial:
    name: str
    prompt: str                             # what the surface is, for the picture model ("wide hand-scraped dark oak floorboards")
    scale_m: float = 1.0                    # how many metres one tile of it covers
    albedo: str = ""                        # paths filled by materials.py: sRGB colour
    normal: str = ""                        # tangent-space normal (OpenGL, +Y)
    roughness: str = ""
    metallic: str = ""
    height: str = ""                        # for bump in Blender and parallax nowhere else
    rgb: list = field(default_factory=lambda: [0.6, 0.6, 0.6])     # a plain colour where no picture is (0..1 sRGB)
    rough: float = 0.6
    metal: float = 0.0
    transmission: float = 0.0               # > 0.5: real glass
    ior: float = 1.5

@dataclass
class Opening:
    kind: str                               # window | door | archway | skylight
    wall_index: int                         # edge of the footprint: from point i to point i+1
    u_centre: float                         # metres along that edge from its start
    sill_m: float
    width_m: float
    height_m: float
    arch: str = "flat"                      # flat | round | pointed
    mullions: list = field(default_factory=lambda: [1, 2])      # panes across, panes up
    glazed: bool = True

@dataclass
class RoomShell:
    footprint: list                         # [[x, y], ...] anticlockwise from above, any shape (L, alcove, octagon)
    height_m: float
    ceiling: str = "flat"                   # flat | beamed | vaulted | coffered | pitched
    thickness_m: float = 0.25
    bevel_mm: float = 4.0
    openings: list = field(default_factory=list)
    wall_mat: str = "wall"
    floor_mat: str = "floor"
    ceiling_mat: str = "ceiling"
    trim_mat: str = "trim"
    skirting_m: float = 0.12
    panelling_m: float = 0.0                # a wainscot this high, of trim_mat; 0 for none

@dataclass
class OutdoorTerrain:
    size_m: float = 40.0
    relief_m: float = 1.5
    layers: list = field(default_factory=lambda: ["ground", "path", "accent"])     # materials: base, worn path, patches
    foliage: list = field(default_factory=list)       # [{"what": "pine tree", "count": 12, "height_m": 9}]
    water_level_m: float = None
    clear_radius_m: float = 2.5             # kept open round the start (0, -5), for walking

@dataclass
class AssetInstance:
    id: str
    name: str
    source: str                             # hero_3d | kitbash | cloth_sim | foliage_card | plain
    prompt: str
    position: list                          # the middle of its underside
    yaw_deg: float = 0.0
    size_m: list = field(default_factory=lambda: [1.0, 1.0, 1.0])    # width (x), depth (y), height (z) before turning
    on_surface: str = "floor"               # floor | wall | ceiling | on (resting on what is under it)
    kind: str = ""                          # copies share a kind and one mesh
    material: str = ""                      # a plain thing's material name
    mesh_path: str = ""
    said: bool = False

@dataclass
class PhysicalLight:
    id: str
    kind: str                               # sun | point | spot | area
    position: list
    direction: list = field(default_factory=lambda: [0.0, 0.0, -1.0])
    kelvin: float = 3000.0
    power_w: float = 60.0
    radius_m: float = 0.08
    of: str = ""                            # the asset it belongs to (its glowing shade dims with it)

@dataclass
class Atmosphere:
    time_of_day: str = "day"                # day | golden_hour | dusk | night
    sky_prompt: str = ""
    env_exr: str = ""
    sun_dir: list = field(default_factory=lambda: [-0.5, -0.6, 0.6])        # towards the sun
    sun_w: float = 4.0
    fog_density: float = 0.0
    fog_height_m: float = 4.0
    fog_color: list = field(default_factory=lambda: [0.78, 0.82, 0.90])
    exposure_ev: float = 0.0

@dataclass
class SceneIR:
    sentence: str
    setting: str                            # indoor | outdoor
    mood: str = ""
    room: RoomShell = None
    terrain: OutdoorTerrain = None
    materials: dict = field(default_factory=dict)
    assets: list = field(default_factory=list)
    lights: list = field(default_factory=list)
    atmosphere: Atmosphere = field(default_factory=Atmosphere)
    camera_eye: list = field(default_factory=lambda: [0.0, -2.0, 1.6])
    camera_target: list = field(default_factory=lambda: [0.0, 0.0, 1.0])

def _build(cls, d):
    if d is None: return None
    names = {f.name for f in fields(cls)}; return cls(**{k: v for k, v in d.items() if k in names})

def load(path):
    with open(path) as fh: d = json.load(fh)
    room = d.get("room"); terrain = d.get("terrain")
    if room: room = {**room, "openings": [_build(Opening, o) for o in room.get("openings", [])]}
    return SceneIR(sentence=d["sentence"], setting=d["setting"], mood=d.get("mood", ""), room=_build(RoomShell, room), terrain=_build(OutdoorTerrain, terrain),
                   materials={k: _build(PBRMaterial, v) for k, v in d.get("materials", {}).items()}, assets=[_build(AssetInstance, a) for a in d.get("assets", [])],
                   lights=[_build(PhysicalLight, l) for l in d.get("lights", [])], atmosphere=_build(Atmosphere, d.get("atmosphere") or {}),
                   camera_eye=d.get("camera_eye", [0, -2, 1.6]), camera_target=d.get("camera_target", [0, 0, 1]))

def save(s, path):
    import os
    tmp = path + ".tmp"
    with open(tmp, "w") as fh: json.dump(asdict(s), fh, indent=1)
    os.replace(tmp, path)
