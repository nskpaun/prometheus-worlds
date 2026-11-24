# RFC: Panorama-First Walkable 3DGS Worlds (`WORLD_METHOD=pano`)

> **Status:** Draft RFC — illustrative architecture + `< 500` LOC prototype.  
> **Scope:** Sentence-to-world (both indoor and outdoor) prioritizing **stunning visual quality** and **2–5 minute builds**.  
> **Rules kept:** One 64 GB Apple Silicon Mac (`<= 30 GB` peak), open local models only, zero cloud/APIs.

## 1. Why we are pivoting away from object-by-object scene assembly

After implementing and running `WORLD_METHOD=pbr` (`6ef00a4`) end-to-end on real prompts (`Mid Century Lounge`, `Bakery`), we hit a structural wall:

- **86 minutes for an incoherent room:** A cozy pastel bakery prompt took **5,157 s** (`bakery-ab62ed`), planned a **12 m × 20 m** (240 m²) warehouse, filled 63 of 132 slots with mismatched Poly Haven props (garden watering cans, Victorian streetlamps), shrunk counters leaving jars floating in midair, and bathed the empty floor in flat yellow light.
- **Object-by-object assembly cannot produce art direction:** A local 35B LLM writing JSON coordinates and gluing 50–130 unrelated assets (TRELLIS.2 meshes + Poly Haven scans + plain beveled boxes + procedural shells) cannot match the composition, lighting hierarchy, clutter density, and style harmony of a single diffusion image.
- **The contrast is already on our own disk:** Single-image / photo-derived worlds (`docs/office-splat.jpg` via SHARP; `pbr/env_ldr.png` via FLUX.2 Klein 9B + 360° LoRA) look 10× more believable and cohesive in **~30–60 seconds** than an 86-minute assembled Blender level.

## 2. Core Bet: Holistic 360° Diffusion -> Multi-View SHARP -> Disocclusion Fill

Instead of generating 100 objects separately and hoping an LLM places them tastefully, **let the diffusion model compose the entire world in one coherent 360° equirectangular pass**, then lift it to a walkable 3D Gaussian Splat in three stages:

```
Sentence
  │
  ▼  Stage 1: Coherent 360° Core (~2 min, ZERO new model downloads!)
  ├─ 1a. FLUX.2 Klein base 9B + 360° Panorama LoRA -> seamless equirectangular pano (2048x1024 -> 4K detail pass)
  ├─ 1b. Exact rectilinear projection -> 10 overlapping perspective crops (6 ring + 2 up + 2 down, 95° FOV)
  │      Exact known poses from one origin -> 100% pixel-consistent across views (no COLMAP, no seams, no blank ceilings)
  ├─ 1c. Apple SHARP per crop (~6 s/view) -> global closed-form depth-scale alignment on overlapping rays
  └─ 1d. Spherical ownership fusion (reusing photo_fuse.py logic) + depth collider mesh (mesh_photos.py)
  │
  ▼  Stage 2: True Walkability (+2-3 min)
  ├─ 2a. Place 4-6 off-center virtual cameras (stepped 1.5-2.2 m away from origin into clear floor space)
  ├─ 2b. Render Stage 1 splat from each off-center camera -> exact RGB + disocclusion hole mask
  ├─ 2c. Inpaint ONLY masked holes (FLUX.2 Klein inpaint or 1-step Difix3D+ restoration), locking existing pixels
  └─ 2d. Lift ONLY newly revealed pixels via SHARP (anchored to surrounding splat depth) -> append to world splat
  │
  ▼  Stage 3: Crispness, Atmosphere & Multi-Room
  ├─ 4K tile-detail upscale of the 360° panorama before cropping (sharp walls & tabletops at 0.8 m)
  ├─ Subtle viewer atmosphere (height fog, drifting particles, Web Audio mood from the sentence)
  └─ Doorway-conditioned panorama chaining for multi-room stitched worlds
```

## 3. Why this solves every failure of the earlier `chisel_ring` splat method

The old ring-view splat method failed because it painted 12 independent perspective views sequentially with a 2D depth ControlNet — neighbouring views disagreed, ceilings stayed blank, sky collected black floaters, and walls stretched 1 pixel over 7–12 screen pixels.

| Old `chisel_ring` failure | How `WORLD_METHOD=pano` eliminates it |
|---|---|
| Neighbouring views hallucinate different rooms | All 10 Stage 1 crops come from **one coherent 360° panorama** — pixel-exact agreement |
| Blank ceilings & floating sky smudges | Pitched-up (`+50°`) and pitched-down (`-50°`) crops cover zenith and floor seamlessly |
| Trained splat needles & haze (`msplat`) | **Feed-forward SHARP pieces** + closed-form overlap scale alignment (zero splat training) |
| Mismatched furniture / giant empty rooms | One diffusion pass designs proportions, clutter, lighting, reflections, and architecture holistically |
| 30–86 minute build times | **~2 min** (Stage 1) to **~5 min** (Stage 1 + Stage 2) |

## 4. Model, Memory & Time Budget (64 GB M4 Pro)

Every Stage 1 model is **already downloaded and pinned by `setup.sh`**:

| Stage | Local Open Model | Licence | Peak RAM | Est. Time |
|---|---|---|---:|---:|
| 360° Panorama | FLUX.2 Klein base 9B + `crafiq/flux-2-klein-9b-360-panorama-lora` | Apache-2.0 | ~29 GB | ~60–90 s |
| Rectilinear crops | Pure numpy (`project.py`) | MIT | < 1 GB | ~1 s |
| 3DGS pieces (×10) | Apple SHARP (`ext/ml-sharp`) | Apple ML Research (non-commercial)* | ~16 GB | ~60 s |
| Overlap align & fuse | `project.py` + `photo_fuse.py` + `mesh_photos.py` | MIT | ~10 GB | ~25 s |
| Stage 2 inpaint (×6) | FLUX.2 Klein 4B inpaint *or* Difix3D+ (optional) | Apache-2.0 | ~15 GB | ~90 s |
| Stage 2 SHARP fill (×6) | Apple SHARP (masked pixels only) | Apple ML Research* | ~16 GB | ~40 s |
| **Total** | **Peak <= 30 GB** (well below 48 GB cap) | | **<= 30 GB** | **~2.5–5 min** |

*\*Licence note (per `CONTRIBUTING.md`):* SHARP weights are non-commercial research only, same as `photos-method.md`. For commercial/fully-permissive builds, the same 10-crop rig can swap SHARP for an Apache-2.0/MIT depth+3DGS lifter (e.g., Depth Pro + surfel lifting or DepthSplat / VGGT-GS).

## 5. Honest Tradeoffs & Rollout

- **What `pano` excels at:** Photorealistic or stylized **visual worlds** (indoors *and* outdoors — Brutalist cliff villas, Kyoto tea rooms, cozy bakeries, snow villages, sci-fi cockpits) built in minutes, with rich lighting, reflections, clutter, and 60 fps Spark playback.
- **What `pano` does NOT replace:** Editable game-engine glTF export (`pbr` / `direct`) or arbitrary 100 m open-world traversal. Walkability is a rich **room / courtyard bubble** (~4–8 m across after Stage 2, expandable via multi-room doorway chaining in Stage 3).
- **Phased rollout:**
  1. **Ship Stage 1 first** behind `WORLD_METHOD=pano` — zero new downloads, immediate visual benchmark on 6 indoor + outdoor prompts.
  2. **Add Stage 2 disocclusion pass** (off-center renders + masked Klein/Difix3D+ inpaint + masked SHARP lift) to double/triple the comfortable walking radius.
  3. **Add 4K detail pass + doorway chaining** once Stage 2 passes off-centre visual evals.
