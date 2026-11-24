# Built in Blender: a sentence to a mesh world, through the Blender MCP

The third way to make a world. You describe a place; it is put together in Blender, where the world-plan model on this Mac (Qwen3.5 35B-A3B) builds furniture shape by shape by calling the tools of the [Blender MCP](https://github.com/ahujasid/blender-mcp); Cycles bakes the light; the viewer shows the result as a mesh, with no splat.

Choose **New world…**, then **Built in Blender**. It needs [Blender](https://www.blender.org) installed (developed with 5.3; `brew install --cask blender` also works). `BLENDER` names another copy.

```
thread/.venv/bin/python thread/pipeline.py blender WORLD_DIR "Name" "SENTENCE" ["STYLE"]
```

There are two routes. **Rich** is for rooms and builds on the direct method ([direct-method.md](direct-method.md)). **Quick** is for everything else, and for rooms when asked (`BLENDER_PLAN=own`, or Detail: Quick in the app): the model plans and builds all of it itself.

## Rich: the direct method's room, assembled and lit in Blender

| Step | File | What happens | Time (study, M4 Pro) |
|---|---|---|---|
| Enrich | `blender_world.py enrich` | The sentence is filled out into about 200 words on the same place: its room, each piece's material and colour, the small things, the light. The layout and the style guide are then asked from both, the sentence having the last word | 14 s |
| Plan, sculpt, style, materials, paint | `build/` | The direct method's own steps, unchanged: the layout with its openings and filler, a picture and a Hunyuan3D mesh for each thing that is not a plain form, the style guide, material swatches, paint on the sculpted things | 16 min |
| Level | `build/export.py` | The room's geometry (skirting, panelling, door, ceiling), every object and the lamps written as one textured glTF: the file the Unreal export makes, at fewer texels | 110 s |
| Furniture | `blender_world.py assemble` | The level is brought into Blender. Then each plain piece of furniture (tables, bookcases, cabinets, box and drum shapes over 35 cm) is built anew by the model through the MCP, in the materials the style gave that piece, brought to exactly its planned box, and put in place of the plain shape. Copies of a kind are built once | 10 to 35 s a kind |
| Look | `blender_kit.py` `_snap` | Blender renders each built piece alone; the model looks at the picture and, if it names a fault in the shape, builds the piece once more with that in mind | 5 s, and a rebuild |
| Light | `blender_finish.py` | Only the light is baked, so textures stay as sharp as they were made: Cycles puts lamps, sky, sun, shadows and bounced light into two 4096 px lightmaps (the room, the things) on a second set of picture coordinates, and into the corners of dense sculpted meshes. What the model built from plain shapes gets its colour and roughness baked to a picture first. A panorama is rendered where the visitor stands. One exposure for all of it, found from the baked light | 100 s |
| Files | `blender_finish.py`, `blender_export.py` | `scene.glb` (Blender's own glTF export: meshes, textures, lightmap coordinates, corner light), `light0.png`, `light1.png` (speckle filtered out), `env.jpg`, `mesh.glb`, `collider.glb`, `scene.json` | 10 s |

The study takes about 21 minutes; the largest process (the picture model) held 30 GB. Running the same command again goes on after the last finished step.

What the model cannot build in three tries, or in `BLENDER_MINUTES` (default 4), keeps the level's plain shape, so a build never ends with a hole.

### Windows and daylight

The layout plans windows and keeps the wall clear for them; `build/level.py` now hands them to the room builder, which before cut windows only by its own style rules (so a window the sentence asked for was often missing, in the sentence mode too). In Blender the level's glowing pane is taken out and real sky shines through the opening, with a sun for the hour the style guide gives: high and white by day, low and orange at dusk, none at night. A wall given to a window can leave no room for the door, which is then left out.

### How it is shown

The viewer draws these worlds as lit meshes, not splats: each mesh's own texture times the lightmap (or its corner light), plus reflections of the panorama by the surface's roughness and metal, dimmed where little light falls. Polished floors, leather and brass catch the light as you move. The viewer adds no lights of its own.

## Quick: the model plans and builds everything

| Step | What happens | Time |
|---|---|---|
| Enrich | As above | 14 s |
| Plan | One JSON plan: the room with doors and windows (or open ground), a palette, the sky, and 18 to 28 things, each with its parts described, a size and a place | 45 s |
| Shell | `blender_kit.py` `room`: floor, ceiling, walls cut round the openings, door leaves, window frames | under 1 s |
| Things | One thing per turn, built at the origin from simple shapes, looked at and mended as above | 5 to 30 s each |
| Size and place | `_fit`, `_put`: each thing brought to its planned size where it is off by more than a fifth, then stood where the plan says, or at the nearest free spot; a thing "on" a bookcase goes on a shelf | under 1 s |
| Light, files | As above | 1 min |

About 10 minutes. Plain colours, no sculpted shapes: it reads as a clean model of the place rather than the place.

## How the model is helped

A local model of this size cannot be left to drive Blender freely. What was tried first, and what replaced it:

- **A free loop** ("build the place step by step, here are the tools") failed. The model wrote the whole room in one 140-line step, which failed near its end; held to short steps it sent the same failing code four times; later it repeated one finished step until time ran out. Positions were wrong throughout.
- **A plan, then one thing per turn** works: every planned thing was built, with no errors, in 4 to 6 minutes. Each turn is a fresh, short conversation with one small job.
- **The model never places anything.** It builds each thing at the origin; where it goes is the plan's business and is checked by code.
- **A kit of helpers** (`blender_kit.py`: `box`, `cyl`, `lathe`, `extrude`, `tube`, `mat`…) is loaded into Blender for the model's code. They hide the parts of Blender's API a model most often gets wrong and share one rule: `at` is where a shape stands, the middle of its underside. Plain `bpy` and `bmesh` stay available.
- **Worked examples** in the prompt (a side table, an armchair, a bookcase with books) did more for the shapes than any rule.
- **It sees what it built.** Qwen3.5's weights include a picture tower that the text loader leaves unused; loaded through mlx-vlm (`chat.Chat(vision=True)`) the same model judges a render in 3 to 6 seconds. Shown an early armchair it answered "a chaotic stack of flat panels and blocks with no clear seat, backrest, or armrests". One look and one rebuild per thing; `BLENDER_LOOK=0` turns it off.
- **A failed step leaves nothing behind**, so the mended code does not build things twice.

## Nothing leaves the Mac

The Blender MCP package is made for online use: it collects prompts, code and screenshots unless told not to, checks for updates, and has tools that reach asset libraries and generation services. Here:

- its telemetry is switched off (`DISABLE_TELEMETRY`), and its update check too;
- Blender and the MCP server both run under `sandbox-exec` with the network closed to everything but this Mac, so nothing can be sent whatever the package tries (`tests/test_blender_world.py` checks that a request to the internet fails inside it);
- the model is offered only two of its tools, `execute_blender_code` and `get_scene_info`.

Blender runs with no window. The MCP's addon refuses to start that way, because Blender runs no timers without a window; `blender_host.py` starts the addon's server by hand and empties its queue of commands itself.

The packages are pinned with hashes: `mcp-for-blender` 2.3.1 (MIT) in `runtime/thread.lock`, `mlx-vlm` 0.8.0 (MIT) in `runtime/paint.lock`.

## Limits

- Rich takes about 21 minutes, a little over the 20-minute budget, and the model builds only the plain furniture in it; the sculpted things and the room come from the direct method.
- The model's own shapes are simple. In materials given to the piece they read well as shelves, cabinets and tables; what stands on the shelves is blocky.
- The one look checks shape only, from one side, and is not always right.
- Through a window there is only a flat sky colour, and the reflections come from one panorama taken at the starting point, so they are a little off elsewhere in the room.
- A room whose walls are full can lose its door to a window.
- Quick, outdoors: trees and plants are beyond simple shapes.
- No moving things, no sound and no export for Unreal yet.
