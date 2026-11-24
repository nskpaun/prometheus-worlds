# The visuals-first method: a sentence to a real-time PBR world

The method from the "visuals first" proposal ([nskpaun/prometheus-worlds#5](https://github.com/nskpaun/prometheus-worlds/pull/5)), built: a sentence becomes a glTF scene of real PBR meshes, lit by Cycles-baked HDR lightmaps and reflection probes under a 360 degree HDR sky, shown by the viewer with a film pass. No splats; photos-to-world keeps SHARP and splats. It runs for both indoor and outdoor sentences, on one Mac, with open local models only and nothing sent off the Mac.

It is the default for sentences (`WORLD_METHOD=pbr`, both the From a prompt and the Built in Blender tabs). `WORLD_METHOD=direct`, `ring` or `blender` gives the earlier methods.

```
thread/.venv/bin/python thread/pipeline.py pbr WORLD_DIR "Name" "SENTENCE"
```

## The steps

| Step | File | Model | What it does |
|---|---|---|---|
| Enrich | `blender_world.py enrich` | Qwen3.5 35B-A3B | The sentence filled out into a fuller description of the same place |
| Layout | `build/layout.py` | Qwen3.5 35B-A3B | Indoors: the things, their sizes and places, chairs round their tables, the openings (unchanged from the direct method). Also says whether the place is outdoors |
| Scene | `pbr/plan.py` | Qwen3.5 35B-A3B | The scene description (`pbr/ir.py`, pillar 1). Indoors: the room's outline (a bay, an alcove, an apse or cut corners added to the layout's box), its ceiling (flat, beamed, coffered, vaulted, pitched), wainscot, window heads and glazing bars, every surface's material in words and true tile size, each lamp's colour temperature and power, the hour and what the windows look out on. Outdoors: the ground's relief and three ground materials, the kinds of tree, bush and grass and how many, the things and where they stand, the horizon. Then who makes each thing: cloth is draped; the CC0 library gives what it holds well; the six most telling kinds left become hero 3D models; the rest are plain bevelled forms |
| Hero things | `pbr/heroes.py` | Z-Image Turbo, TRELLIS.2-4B | A clean picture of each hero kind, checked against its planned proportions (a table painted as just its top is asked again), cut out, then TRELLIS.2 through its Apple Silicon port (`ext/trellis-mac`), which returns a mesh with baked colour, metallic and roughness maps. The pipeline is loaded once for all of them. A kind it cannot make falls back to the library, then to a plain form |
| Materials | `pbr/materials.py` | FLUX.2 Klein 4B, CHORD | Each surface painted as a seamless tile, then split by CHORD (Ubisoft La Forge, SIGGRAPH Asia 2025) into base colour, normal, roughness and metallic maps; height integrated from the normal map. All made to tile (pillar 2) |
| Sky | `pbr/sky.py` | FLUX.2 Klein base 9B + a 360 degree LoRA | An equirectangular panorama of what lies outside at that hour, its ends blended so it closes round, lifted to HDR: made linear, the sky set to a real sky's radiance against the sun (about a quarter of the sun's light on a surface), a glow round the sun. The world Blender lights with has no sun disc (the sun lamp is the sun); the viewer's copy has one, so bloom catches it |
| Compile | `pbr/compile_blend.py` | Blender (Cycles) | The scene in headless Blender (pillar 3). Indoors: walls with thickness on any outline; openings cut through them with flat, round or pointed heads; windows with casing, mullions and transoms, sill, a pane of real glass and a Cycles light portal; the door with its leaf and panels; skirting, cornice, an optional wainscot with raised panels; the ceiling type, with gables for a vault or pitch and beams or rafters; every edge bevelled with weighted normals. Outdoors: 40 m of ground shaped by layered noise, flat where the visitor starts and along a winding path, its three materials mixed by the path and patches and baked to one map set; water at its level; foliage from the library scattered by Poisson discs. Things: hero and library models brought in once a kind and fitted to their planned size; cloth settled by the cloth simulation (curtains hang in pleats, cloths fall onto what is under them); plain forms bevelled. Every material a full PBR node set on coordinates in metres. Lamps with temperature, soft size and a glowing bulb; the sun as a sun lamp; the HDR sky as the world |
| Art direction | `blender_direct.py` | Qwen3-VL 32B | Five Cycles previews (the start, three corners, close on the largest thing) judged against the sentence and an inventory; up to eight mends a round, two rounds: take out, set down, scale, relight, exposure, add clutter from the library onto a named surface, fog (pillar 4) |
| Bake | `blender_finish.py` | Cycles, OpenImageDenoise | The light only, 128 samples, into two 4096 px HDR lightmaps (half-float EXR, speckle filtered) and the corners of dense meshes; white balance and exposure by the hour; two to four reflection probes (HDR panoramas) with the room's box; the viewer's HDR sky; glass left out of the lightmaps |
| Files | `blender_export.py` | | `scene.glb` (PBR meshes, normal maps, glass with KHR_materials_transmission, lightmap coordinates), `light0.exr`, `light1.exr`, `probe*.exr`, `sky.exr`, `mesh.glb`, `collider.glb`, `scene.json` |

Each large model runs in its own process and leaves memory before the next one loads.

## In the viewer (pillar 6)

High dynamic range throughout: the EXR lightmaps on the second coordinates, each piece lit by its nearest reflection probe looked up with box projection (the panorama treated as the room's walls, so reflections line up as you walk), real glass, the HDR sky behind the windows and all round outdoors, height fog as bright as the place's own light, then the film pass: 4x multisampling, contact shadows (GTAO), bloom, Neutral tone mapping. `?post=0` turns the film pass off.

The proposal names a WebGPU renderer; like its own prototype, this uses WebGL2 with a half-float buffer, which the vendored three.js already has.

## Measured on the M4 Pro (64 GB)

| Step | Time | Largest process |
|---|---|---|
| Enrich, layout, scene | about 2 min | 22 GB (Qwen3.5 35B) |
| Hero pictures (768 px, up to four tries) | about 100 s a kind | 12 GB |
| TRELLIS.2 meshes | 70 to 350 s a kind (about 3 min on average), the model loaded once | 30 GB |
| Material pictures and CHORD | about 10 s a material | 14 GB |
| Sky (20 steps, 1536 x 768) | about 10 min | 18 GB |
| Compile in Blender | 1 to 5 min (cloth and the ground bake are most of it) | 6 GB |
| Art direction, two rounds | about 3 min | 20 GB (Qwen3-VL 32B) |
| Bake and files | 2 to 12 min | 8 GB |

A night lounge with nine hero kinds came to about an hour of work; the courtyard (four hero kinds) to 35 minutes. That is well over the 20-minute budget the earlier methods kept to: the hero models and the sky are most of it. `WORLD_HEROES` (default 6, plus every kind the sentence names) and `SKY_STEPS` (default 20) trade time for quality.

## Limits

- Slow, as above.
- The plan decides a lot: an architect answer that plays safe gives a plain box room, and the outdoor planner can still put things oddly. Walls, fences and the like are always built plain and oriented by where they stand; copies numbered by the model ("bench 1") are made one kind.
- The library's matches are by words and proportions only; a filler thing takes a library model only on a good match (hazard signs, weapons, rust are kept out unless asked for).
- The lightmaps hold no direction, so normal maps show only in reflections and the from-above shading of relief.
- Art direction sees five small renders and can be wrong; it cannot add large things or rebuild a bad model.
- Open water only where the sentence asks for a lake, pond, river, sea or shore.

## Setup

`./setup.sh` fetches all of it (`--no-pbr` skips it): Qwen3-VL 32B, FLUX.2 Klein base 9B and the panorama LoRA, the Poly Haven CC0 library (521 models, 1K textures, about 4 GB, `ext/cc0-kitbash`), CHORD in its own environment (`ext/ubisoft-laforge-chord`, patched to run on the Mac's GPU, with Stable Diffusion 2.1's configs from `sd2-community`), and the TRELLIS.2 port in its own (`ext/trellis-mac`, which needs Xcode's Metal Toolchain for its texture baker). CHORD and FLUX.2 Klein base 9B are gated: ask for access on their Hugging Face pages and sign in (`spike/.venv/bin/hf auth login`) first. Without one of them the method still runs: materials keep their colour pictures, skies stay plain, hero things come from the library.

Licences: TRELLIS.2 and its port MIT; Qwen3-VL Apache 2.0; the panorama LoRA Apache 2.0; FLUX.2 Klein base 9B under Black Forest Labs' licence; CHORD under the Ubisoft Machine Learning Licence (research only); Poly Haven's models CC0.
