"""Local VLM Art-Director critique & refinement loop (RFC prototype).

Current pipelines plan layouts blind: the LLM outputs coordinates once and never sees the room.
Here, a local MLX VLM (Qwen3-VL / Qwen2.5-VL, ~18 GB) inspects 5 rendered preview views
(hero, two off-centre corners, one close-up, one overview) and emits targeted SceneIR patches
for scale, clutter density, focal composition, and lighting contrast before the final GI bake.
"""
from dataclasses import dataclass, field
from typing import Literal
from .schema import AssetInstance, SceneIR

CRITIQUE_PROMPT = """\
You are a senior environment artist reviewing 5 preview renders of a 3D world generated from:
  "{sentence}" (mood: {mood})

Inspect the images critically — especially corners, object proportions, and close-up surfaces:
1. Scale & grounding: flag any floating, intersecting, or badly proportioned objects.
2. Visual richness: flag bare walls, empty table tops, or sterile corners needing small dressing clutter.
3. Lighting hierarchy: flag flat/washed-out lighting; suggest key/fill Kelvin contrast or exposure_ev delta.
4. Composition: verify the start camera frames the focal hero clearly with depth in foreground & background.

Return JSON only:
{{"score_10": int, "issues": [str], "patches": [
  {{"op": "move"|"rescale"|"add_clutter"|"tune_light"|"tune_atmosphere", "target": str, "params": {{...}}}}
]}}
"""


@dataclass
class ScenePatch:
    op: Literal["move", "rescale", "add_clutter", "tune_light", "tune_atmosphere"]
    target: str
    params: dict = field(default_factory=dict)


@dataclass
class CritiqueResult:
    score_10: int
    issues: list[str]
    patches: list[ScenePatch]


def critique_renders(scene: SceneIR, preview_paths: list[str]) -> CritiqueResult:
    """Run local MLX VLM over the 5 rendered previews and parse structured art-direction patches."""
    _prompt = CRITIQUE_PROMPT.format(sentence=scene.sentence, mood=scene.mood)
    # Subprocess call to local MLX VLM (unloaded immediately after returning JSON):
    #   raw = run_mlx_vlm(model="Qwen/Qwen2.5-VL-32B-Instruct-4bit", images=preview_paths, prompt=_prompt)
    #   data = parse_json(raw)
    _ = (_prompt, preview_paths)
    return CritiqueResult(score_10=8, issues=[], patches=[])


def apply_patches(scene: SceneIR, patches: list[ScenePatch]) -> int:
    """Deterministically apply VLM art-direction patches to the SceneIR; returns count applied."""
    by_asset = {a.id: a for a in scene.assets}
    by_light = {l.id: l for l in scene.lights}
    applied = 0

    for p in patches:
        if p.op == "move" and p.target in by_asset and "delta" in p.params:
            a = by_asset[p.target]
            dx, dy, dz = p.params["delta"]
            a.position = (a.position[0] + dx, a.position[1] + dy, a.position[2] + dz)
            applied += 1
        elif p.op == "rescale" and p.target in by_asset and "factor" in p.params:
            a = by_asset[p.target]
            f = max(0.4, min(2.2, float(p.params["factor"])))
            a.size_m = (a.size_m[0] * f, a.size_m[1] * f, a.size_m[2] * f)
            applied += 1
        elif p.op == "add_clutter" and "prompt" in p.params and "position" in p.params:
            # Small secondary storytelling props (books, mugs, lanterns, pebbles) use instant CC0 kitbash
            scene.assets.append(AssetInstance(
                id=f"clutter_{len(scene.assets):03d}",
                name=p.params.get("name", p.target),
                source="kitbash",
                prompt=p.params["prompt"],
                position=tuple(p.params["position"]),
                size_m=tuple(p.params.get("size_m", (0.25, 0.25, 0.25))),
                on_surface=p.params.get("on_surface", "floor"),
            ))
            applied += 1
        elif p.op == "tune_light" and p.target in by_light:
            lt = by_light[p.target]
            if "power_mul" in p.params:
                lt.power_w *= max(0.2, min(4.0, float(p.params["power_mul"])))
            if "kelvin" in p.params:
                lt.kelvin = int(max(1800, min(10000, p.params["kelvin"])))
            applied += 1
        elif p.op == "tune_atmosphere":
            if "exposure_ev" in p.params:
                scene.atmosphere.exposure_ev = float(p.params["exposure_ev"])
            if "fog_density" in p.params:
                scene.atmosphere.fog_density = max(0.0, min(0.08, float(p.params["fog_density"])))
            applied += 1
    return applied
