# RFC: Visuals-First Rebuild for Generated Worlds (`WORLD_METHOD=pbr`)

> **Status:** Draft RFC — illustrative architecture + non-runnable prototype (`< 1000` LOC).  
> **Scope:** Sentence-to-world (both indoor and outdoor). Photo worlds keep the SHARP splat pipeline.  
> **Rules kept:** One 64 GB Apple Silicon Mac (`<= 48 GB` peak), open local models only, zero cloud/APIs.

## 1. Why the current worlds hit a visual ceiling

Prometheus Worlds currently has two sentence-to-world pipelines, and both bottleneck visual quality at the representation layer:

1. **Ring-view trained splats (`chisel_ring.py` -> `consistent.py` -> `msplat`)** — used for outdoors and the README showcases. Sequential 2D ControlNet views are not 3D-consistent: walls stretch 1 painted pixel over 7–12 screen pixels up close (`docs/evaluation.md`), ceilings go blank, floors smear, and foliage/sky collect floaters and streaks.
2. **Direct-method flat discs (`thread/build/`)** — already moved to real geometry, true-scale materials, and Embree light tracing, **but throws away crisp triangles and textures by sampling everything back into 6–7M flat Gaussian discs (`world.ply`, 300–800 MB, ~30 fps)**. Keeping the Spark splat format as the target strips away normal/roughness/metallic maps, real-time specular/reflections, real refractive glass, crisp silhouettes, and post-processing.

Upstream of rendering, four more choices make scenes read as "toy" up close:
- **Boxy rooms & unbeveled primitives:** `arch.py` restricts rooms to rectangular boxes from 9 keyword-matched templates; `forms.py` builds sharp 90° boxes and 24-segment lathes with no bevels, moldings, cloth drape, or branching foliage.
- **Single-view mini shapes + sheet repainting:** `Hunyuan3D 2.0 mini` (`shape-small`) + 4-step FLUX Klein contact-sheet img2img (`paint.py`) leaves seams, smeared back faces, and word-guessed roughness/metal classes.
- **No PBR maps or exterior world:** Surfaces are diffuse-only RGB; window glass is a flat constant-color emissive quad (`SKYC[tm]`); `viewer/main.js` uses raw `WebGLRenderer({ antialias: false })` with no bloom, GTAO, volumetrics, or tone-mapped HDR pipeline.
- **Blind layout:** The LLM writes coordinates once and never sees what it built.

---

## 2. Core Architectural Bet

**Stop using Gaussian splats to represent synthetic worlds.**

- **Photos -> World** keeps **SHARP + Spark splats** (splats excel at capturing reality).
- **Sentence -> World** outputs a **real-time PBR glTF scene** (`scene.glb`) paired with **Cycles GPU path-traced HDR lightmaps** (denoised with OIDN), **parallax-corrected reflection probes**, an **HDRI 360° sky/exterior environment** (`env.exr`), and a **real-time PBR viewer** (Three.js WebGPU / Metal-backed post-processing).

```
Sentence
  │
  ▼
1. Plan (Qwen3.5 35B-A3B, MLX)
   Non-box architectural footprint OR outdoor terrain + Hero assets + Dressing + Lighting mood
  │
  ▼
2. Materials & Sky (Sequential, unloaded after use)
   ├─ Seamless PBR maps (Albedo + Normal + Roughness + Metallic + Height)
   ├─ 360° HDRI sky & exterior vista (seen through real glass + drives IBL)
   └─ Full-scale native-3D PBR hero assets (TRELLIS-2 / Hunyuan3D 2.1+ full)
      + local CC0 micro-kitbash library (trim, foliage cards, small clutter)
  │
  ▼
3. Scene Compilation (Headless Blender `bpy` + Geometry Nodes)
   Beveled non-box architecture, cloth drape, alpha-card foliage, UV2 lightmap pack
  │
  ▼
4. VLM Art-Director Loop (1–2 passes, local Qwen3-VL / Qwen2.5-VL)
   Render 5 off-center / close-up previews -> critique scale, clutter, framing, lighting -> patch IR
  │
  ▼
5. Film-Grade GI Bake (Blender Cycles Metal GPU + OpenImageDenoise)
   HDR directional lightmaps (UV2) + cubemap reflection probes + irradiance grid (15–30 s)
  │
  ▼
6. Real-Time PBR Viewer (WebGPU / Metal, 60–120 fps)
   PBR meshes + baked HDR GI + real glass/reflections + GTAO + AgX + bloom + volumetric fog
```

---

## 3. The 6 Visual Pillars (mapped to code in this RFC)

| Pillar | What replaces what | Prototype file |
|---|---|---|
| **1. Rich Scene IR** | Replaces box-only `boxes.json` with polygon/vaulted/multi-level footprints, outdoor heightfields, full PBR specs, and physical lights | `thread/pbr_rfc/schema.py` |
| **2. Full PBR Assets & Sky** | Replaces flat RGB swatches + `shape-small` + `klein_paint.py` with native PBR material synthesis, full-scale PBR 3D hero models, 360° HDRI sky, and CC0 micro-kitbashing | `thread/pbr_rfc/assets.py` |
| **3. Headless Blender Compiler + Cycles/OIDN Bake** | Replaces numpy `arch_mesh.py`/`forms.py`/`assemble.py`/`light.py`/`export.py` with beveled Geometry Nodes, cloth sim, alpha-card foliage, Metal GPU Cycles lightmaps, and native glTF export | `thread/pbr_rfc/blender_scene.py` |
| **4. VLM Art-Director Loop** | Replaces blind single-shot layout with visual inspection of 5 rendered views + automated scale/clutter/lighting patches | `thread/pbr_rfc/art_director.py` |
| **5. Unified Indoor + Outdoor Pipeline** | Replaces `prompt_direct` + `prompt_ring` split with one sequential `< 48 GB` orchestrator (`WORLD_METHOD=pbr`) | `thread/pbr_rfc/pipeline.py` |
| **6. Cinematic PBR Viewer** | Replaces disc-only Spark rendering for synthetic scenes with Three.js WebGPU PBR, AgX tone mapping, bloom, GTAO, SSR/probes, and height fog | `thread/viewer/pbr_viewer_rfc.js` |

### Why outdoors becomes dramatically better too
Instead of painting a 20 m flat backdrop box around a rectangle, outdoors uses the **same PBR pipeline**:
- **Far field (`30 m -> inf`):** Generated 360° HDRI panorama (`env.exr`) drives both the sky/distant mountains and physically consistent image-based lighting.
- **Walkable foreground (`0 -> 30 m`):** Displaced heightfield terrain with splat-mapped PBR ground layers (snow, cobble, moss, dirt), instanced **branching + alpha-card trees** (sunlight filters through leaves), and PBR buildings/props blended seamlessly into the horizon via height fog.

---

## 4. Model, Memory & Time Budget (64 GB M4 Pro)

Every large model runs in its own subprocess and exits before the next loads, keeping peak footprint under the 48 GB rule in `CONTRIBUTING.md`:

| Stage | Local Open Model / Engine | Licence | Peak Footprint | Est. Time |
|---|---|---|---:|---:|
| Plan + Style | Qwen3.5 35B-A3B (MLX 4-bit) | Apache-2.0 | ~22 GB | ~45 s |
| 360° HDRI Sky | Equirectangular FLUX / SDXL-pano + HDR inverse-TM | Apache-2.0 / Open | ~16 GB | ~30 s |
| PBR Materials | Seamless PBR generator (e.g. CHORD / MatFuse / Material-Anything) | Open (TBD pin) | ~14 GB | ~2 min |
| Hero 3D Assets (6–10) | Full-scale native-PBR 3D (TRELLIS-2 / Hunyuan3D 2.1+ full) | MIT / Open (verify) | ~24–32 GB | ~4–6 min |
| Scene Compile + Preview | Headless Blender 4.x (`bpy`, Metal GPU) | GPL-3.0 (external CLI) | ~6 GB | ~30 s |
| VLM Critique (1–2x) | Qwen2.5-VL / Qwen3-VL (MLX 4-bit) | Apache-2.0 | ~18 GB | ~40 s |
| Final GI + Probe Bake | Blender Cycles (Metal GPU + OIDN, 128 spp) | GPL-3.0 (external CLI) | ~8 GB | ~45 s |
| **Total** | **Peak <= 32 GB** (well below 48 GB cap) | | **<= 32 GB** | **~9–11 min** |

*Bonus:* Because `scene.glb` + lightmaps + `collider.glb` are native PBR meshes (~40–90 MB vs. 300–800 MB `.ply`), **Unreal / Godot / Blender export becomes zero-cost** — the viewer already renders the exact engine-ready scene.

---

## 5. Open Decisions Before Implementation

1. **Viewer runtime:** Three.js WebGPU inside the existing WKWebView (keeps stitching + Rapier + Web Audio in one codebase) vs. native Swift **RealityKit / Metal 3** (hardware ray-traced reflections on M3/M4, zero WebGL/WebGPU quirks). This RFC illustrates Three.js WebGPU for lowest migration friction.
2. **Hero 3D model choice & licence:** Prefer an MIT/Apache-2.0 native-PBR 3D model (e.g., TRELLIS / Step1X-3D family) over territory-restricted weights so every Mac user gets full quality out of the box.
3. **CC0 micro-kitbash pack:** Bundle a ~500 MB curated CC0 pack (Poly Haven / AmbientCG-derived moldings, leaf cards, books, tableware, stones) so every scene gets dense micro-geometry without burning 3D-gen budget on small clutter.
4. **Stitching:** Doorways between `WORLD_METHOD=pbr` rooms become clean CSG boolean cuts in Blender / Three.js rather than SDF splat-erasure boxes, and cross-room light portals can be rebaked in seconds.

## 6. Phased Rollout

- **Phase 1 (Biggest immediate jump):** Keep existing `layout.py` + `sculpt.py`, replace `assemble.py`/`light.py` disc output with headless Blender Cycles+OIDN baked `scene.glb` + `pbr_viewer_rfc.js`.
- **Phase 2:** Upgrade to full PBR material maps (normal/roughness/metallic), beveled Geometry Nodes architecture, and 360° HDRI exteriors.
- **Phase 3:** Swap `Hunyuan3D mini + klein_paint.py` for full native-PBR 3D assets + CC0 micro-kitbashing + VLM art-director loop.
- **Phase 4:** Retire `prompt_ring` once outdoor terrain + alpha-card foliage + HDRI horizon beats it visually across all eval prompts.
