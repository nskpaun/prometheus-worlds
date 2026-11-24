# The direct method: a sentence to a world with no splat training

This is how a world is built from a sentence since November 2026. The code is in `thread/build/`; `thread/pipeline.py` runs it (`prompt_direct`). The earlier way, boxes painted from a ring of views and a trained splat, is still there: see [the switch](#switches) below.

The idea: make real geometry for everything, give every surface its own true-scale material and every object its own paint, light the lot with traced rays, and write the splat straight from the result as millions of small flat discs. Nothing is trained, so nothing is hazy where no camera looked, and a wall is as sharp from 80 cm as from the middle of the room.

## The steps

Times are for the Standard build of a 5 x 6 m room on an M4 Pro. One large model is in memory at a time. The largest process is the object-picture model at about 30 GB of memory footprint (what macOS counts against the machine, GPU memory included; its resident size is about half of that); the text model holds about 21 GB, the material and paint model 12 to 15 GB and the assembler 7 to 8 GB.

| Step | What happens | Code | Model | About |
|---|---|---|---|---|
| Plan | The sentence becomes a list of 30 to 45 things, each with a true size, a look, a form and a relation (on the floor against a wall, on another thing, on a wall, from the ceiling). A second short answer says what does not belong and how full the place is, and gives pools of small things. Code turns relations into positions that hold, repairs what is wrong, and fills tops, walls and floor by rule. Tables people sit at are spread over the floor and the free-standing chairs dealt out round them, facing them (four to a table, two to a small one, one to a desk): written as rows of free chairs, a cafe came out a lecture hall. Several copies of a thing the sentence puts "in the middle" keep their own spread instead of piling on the centre point. | `layout.py`, `prompts.py`, `chat.py` | Qwen3.5 35B-A3B | 80 s |
| Sculpt | Up to 11 kinds of thing get a real mesh: one picture each, cut out, turned into a shape, placed and re-seated on the real surface of what it stands on. A picture whose object is far flatter than its planned box (under three quarters of the height-to-width it should show from that view) is painted again, asking for the whole thing from the side on its legs or base: asked for "a round oak table, satin waxed finish", the picture model paints a tabletop, and the 3D model makes a disc. The rest are plain forms made by code in milliseconds: tables, bookcases with separate books, crates, bottles, curtains with folds, rugs, framed pictures. | `sculpt.py`, `forms.py`, `../shapes.py` | Z-Image Turbo, Hunyuan3D | 4 min |
| Room | The room itself is real geometry: walls with depth, windows with frames, glass and bars, doors, skirting, beams, a chimney breast behind a fireplace. A style and the openings are chosen from the words of the sentence by rule. | `arch.py`, `arch_kit.py`, `arch_mesh.py` | none | under 1 s |
| Style guide | One concept picture is painted from the sentence and its colours are measured. The text model then writes a guide: a colour scheme, a palette, how worn the place is, what the walls, floor and ceiling are made of. In the same sitting it names a material for every part of every plain form, says what each flat thing shows (a rug's pattern, a painting's subject), and rewrites each sculpted kind's look in the guide's colours. | `style.py` | FLUX.2 Klein 4B, Qwen3.5 35B-A3B | 70 s |
| Materials | Each wall, the floor and the ceiling is laid for this room at 512, 384 and 192 pixels a metre from one close-up photograph of its material whose bricks, boards or flags were drawn first, so each can be cut out and laid in real courses. Wear is painted from the room's own geometry: grime rising from the floor, soot above a fire, a worn path from the door. Every kit group and every plain-form part gets a swatch; flat things get their content. | `materials.py`, `surfaces.py`, `mats.py`, `klein.py` | FLUX.2 Klein 4B | 6 min |
| Paint | Each sculpted kind is painted on its own mesh: several straight-on views of it on one sheet, repainted in one call so all sides agree, registered back onto the mesh, unseen sides mirrored or filled. Each painted point gets a material class (plain, glass, metal, glow, gloss). | `paint.py`, `sheet.py`, `geom.py`, `align.py`, `paint_class.py`, `klein_paint.py` | FLUX.2 Klein 4B | 3 to 6 min |
| Assemble | Discs are laid on everything: a true-scale lattice on flat faces, thrown darts fitted to edges on free shapes, finer where a walker's eye can come close, inside a budget of 5.6 to 6.9 million. | `assemble.py`, `painted.py`, `edge_discs.py` | none | 40 s |
| Light | Lamps, candles, a fire found in its hollow, sun or moon through the windows with the shadows of the glazing bars, three bounces, contact shade, one exposure for the time of day, shine on glossy things, flat mirrors that are real mirrors. Baked into the discs. | `light.py`, `light_base.py`, `mirror.py` | none | 25 s |
| Write | `world.ply`, and `world_sh.ply` for the glossy discs only (they carry the harmonics that make shine move with the eye), `collider.glb` and `scene.json`. | `light.py`, `assemble.py` | none | 1 s |

The world's folder keeps every stage's files (`layout.json`, `boxes.json`, `shapes/`, `materials/`, `paint/`, `steps.json`, `build.json`). `steps.json` lists the finished steps with their seconds and the largest process's memory.

## Switches

- **Detail** in the dialog (`WORLD_VIEWS`): Standard is the build above. Full (16) plans more sculpting time, paints object sheets with about twice the pixels, has the picture model mark glass and lamp glass on each painted sheet, and makes up to 14 part swatches instead of 10. It takes about 25 to 30 minutes.
- `WORLD_METHOD=ring` builds with the earlier method. Set it where the server or `pipeline.py` is started; it needs a style as well as a layout.
- An outdoor place is always built by the earlier method: choose Outdoors in the dialog, or leave the choice on Auto and let the plan say so (the direct build then stops after its plan, one to two and a half minutes in, and the earlier method starts afresh with its own 20 minutes). With Indoors chosen (`WORLD_SETTING=indoor`) a room is built whatever the plan says.
- Without the 3D shape model (`./setup.sh --no-shapes`, or a setup that could not build it) every sentence is built by the earlier method, and the build's log says so.
- `WORLDS_ROOT=/some/folder` makes the server and the pipeline keep `worlds/` and `scenes.json` there instead of in `thread/`, for trials that must not touch your worlds.

From the command line, into a scratch folder:

```bash
mkdir -p /tmp/trial/worlds
cd thread && WORLDS_ROOT=/tmp/trial .venv/bin/python pipeline.py prompt /tmp/trial/worlds/study "Study" \
  "A wizard's tower study at night: a big desk in the middle with a brass globe on it, tall bookshelves along the back wall, a fireplace on the right wall" \
  "dark oak, old stone, deep blue velvet"
WORLDS_ROOT=/tmp/trial .venv/bin/python server.py 8765      # then open http://127.0.0.1:8765/viewer/index.html
```

Run the same `pipeline.py` line again after a stop and it goes on after the last finished step. A different sentence, style or detail level in the same folder starts again from nothing.

## What works in the viewer

Seen working on trial builds: walking with the collider, the splat, mesh and collider views, and night mode. The mesh view shows the collider, since there is no separate blockout mesh: the room's real geometry and every object, large sculpted ones as their convex hulls (a car is a rounded hull).

Written into the world's entry the same way as for any invented room, but not yet tried on a direct world: the tour, dropping balls, the sound's mood, the small drifting effects (dust, sparks), sway, and stitching with doorways.

## Exporting for Unreal Engine

**Export for Unreal…** in the viewer, or `pipeline.py export WORLD_DIR NAME`, writes the world's `export/` folder in about 3 to 4 minutes (CPU only, about 10 GB):

- `NAME.glb`: the level to light in Unreal. Each piece is a real mesh: the room's own geometry, every plain-form part and every sculpted thing. Each has its own texture, unwrapped with xatlas and coloured by the same calls the assembler colours its discs with, so the export shows the splat's own materials and paint: 160 texels a metre on the room, 320 on things, 480 on the things the sentence names, at most 4096 a side. Sculpted things get roughness and metal from the paint's material classes (glass, bare metal, gloss). Lamps, fittings and windows glow, and the light's own emitters become glTF lights: lamps as point lights, fittings, fires and screens as spots, the sun as a directional light, in real units.
- `NAME_baked.glb`: the same meshes and UVs with the splat's traced light baked into unlit textures. It looks like the viewer at once and needs no lights.
- `README.txt`: the import steps (File > Import Into Level keeps the placement and the lights).

The frame is glTF's: metres, +Y up, the middle of the room's floor at the origin, laid out as the viewer shows it. Checked by importing both files in Blender 5.3 and rendering from the viewer's own test spots; not yet imported into Unreal itself. Worlds built from photos or by the ring method cannot be exported yet: there are no real meshes to export.

## Known limits

- **Rooms only.** One box-shaped room, 3 to about 16 m across. The architecture kit has no pieces for caves or open ground, which is why outdoor places go to the earlier method.
- **Almost nothing moves.** Every object is baked into one splat together with its light and shadow. A swing, a spin or a travelling thing asked for in the sentence stands still. Only sway is passed to the viewer (curtains, plants bend in place), and it has not been looked at closely: glossy discs in the second file do not bend with the rest, and shadows stay where they were baked.
- **Light is baked for the sentence's time of day.** Night mode still works, but it darkens a world that is already lit; a world built "at night" does not become day.
- **Measured on few worlds.** The rules and constants were tuned on a tower study and a garage and checked on a handful of other sentences. An unusual place can come out odd: the style is chosen by word rules, and a sentence with no known style word gets a plain modern room.
- **A sentence that names many large things takes longer than 20 minutes.** Named things are always sculpted and painted, whatever the time plan says.
- **A kind whose picture, cut-out or shape fails is left out**, with what stands on it. `boxes.json` lists it under `dropped`; `shapes/KIND.failed` says why. A light is the exception: it gets a plain stand-in of its size (a housing and a shining part), so the room keeps its light. Long thin things, such as a lighting track, are the ones whose picture most often fails.
- **The app cannot go on with a stopped build.** Each build from the dialog gets a new folder, so going on after the last finished step only works from the command line, and a failed or cancelled build leaves its folder under `worlds/` until you delete it.
- **English only.** The word rules (what is a lamp, a fire, a vehicle, which style a place has) know English words.
- **Two named things whose real shapes intersect** are only reported (in `job.log`), not moved.
- **Sculpted shapes are still one picture's worth.** Copies of a kind share a shape; of a counted thing ("5 cars") at most three shapes are made and recoloured.
- **Shine is modest.** A reflection is one baked ray a disc, so a car shows a soft picture of the room, not a sharp one. On a pale car no chrome is claimed, because it cannot be told from the body by colour.
- **A flat mirror costs up to 600,000 discs** and its image hangs outside the room's box, so a stitched neighbour behind that wall would stand inside it.
- **Frame rate.** The disc budget is set for about 30 frames a second at 1024 x 768 on an M4 Pro in the widest view of a 14 x 12 m hall, and that is what was measured on a 12 x 10 m and a 14 x 12 m garage in headless Chrome: 26 to 37 frames a second looking across the whole room, 55 to 60 close to a wall or an object. Worlds from the earlier method and from photos run at 60. A slower Mac draws the same discs more slowly; there is no setting yet.
- **Files are large.** About 250 to 400 MB for `world.ply` and 40 to 480 MB for `world_sh.ply` (most where many glossy things stand, such as cars).
- **`klein_paint.py` reaches into mflux 0.24.0** (it starts the edit model from a picture under noise). It asserts the shapes it relies on and must be moved together with the pin in `runtime/paint.lock`.
- **Not verified:** a build from the app's own window (the trial build was started through the same job API the dialog posts to, and looked at in headless Chrome), stitching a direct world to another, and a build on a Mac that has never downloaded the models.
