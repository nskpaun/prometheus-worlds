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
| Art direction | `blender_direct.py` | The sky painted and laid round the place; then the place rendered from five spots and looked at by the model, which takes out what does not belong, sets down what hangs in the air and relights it for its hour (below) | about 1.5 min |
| Light | `blender_finish.py` | Only the light is baked, so textures stay as sharp as they were made: Cycles puts lamps, sky, sun, shadows and bounced light into two 4096 px lightmaps (the room, the things) on a second set of picture coordinates, and into the corners of dense sculpted meshes. What the model built from plain shapes gets its colour and roughness baked to a picture first. A panorama is rendered where the visitor stands. One exposure for all of it, found from the baked light | 100 s |
| Files | `blender_finish.py`, `blender_export.py` | `scene.glb` (Blender's own glTF export: meshes, textures, lightmap coordinates, corner light), `light0.png`, `light1.png` (speckle filtered out), `env.jpg`, `mesh.glb`, `collider.glb`, `scene.json` | 10 s |

The study takes about 23 minutes; the largest process (the picture model) held 30 GB. Running the same command again goes on after the last finished step.

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
| Art direction, light, files | As above | 2.5 min |

About 10 minutes. Plain colours, no sculpted shapes: it reads as a clean model of the place rather than the place.

## Art direction: a sky, a look, and a film camera

Taken from the "visuals first" proposal ([nskpaun/prometheus-worlds#5](https://github.com/nskpaun/prometheus-worlds/pull/5)), with what is on this Mac: its sky (pillar 2), its art-director loop (pillar 4), and its cinematic viewer (pillar 6). It runs between putting the place together and baking its light, on both routes.

**The sky.** Where the place has windows, or is outdoors, the picture model (FLUX.2 Klein, 30 s) paints the view outside: the neighbourhood of the kind of building the style guide names, at its hour. It is painted as a wide 2048 x 512 picture with a low horizon and laid as the band of an equirectangular panorama from 30 degrees below the horizon to 60 above; its ends are blended so it closes round, the sky above is carried up to the zenith and the ground below softened. In Blender it becomes the world, as bright on average as the plain sky colour it replaces, so its colours come in by the windows and light the bake. The camera sees it about 2.5 stops dimmer by day (1.7 at dusk), and the bake caps its middle tone just under white, than the light it casts (a photographer's window gel), so the view stays visible instead of burning out white. The viewer shows the same picture behind the windows (`sky.jpg`). `BLENDER_SKY=0` leaves the plain sky.

**A look before the bake.** The place is rendered from where the visitor starts, from three corners and close to its largest thing (Cycles, 24 samples, about a second each once Metal's kernels are compiled), toned at the exposure the bake will use. The world-plan model, with its picture tower, sees the five views beside the sentence, the light the style guide describes, and an inventory: each thing's name, size and place, how far it hangs over what is under it, and each lamp. It answers with a score, its faults and up to eight mends from a fixed set, which code applies within bounds:

| Mend | What code does |
|---|---|
| `remove` a thing | takes it out with its parts and any lamp inside it |
| `ground` a thing | lowers it onto what is under its middle (rays cast down); a thin thing on a wall stays where it hangs |
| `scale` a thing | 0.4 to 2 times, about the middle of its underside |
| `lamp` (a name, or the start of several) | watts 0 to 5 times, its glowing shade with it; optionally a colour temperature |
| `exposure` | -2 to +2 stops a round, -1 to +1 in all; halved when the same round changes lamps, since the exposure follows the lamps by itself |

Two rounds at most; a round with no mends ends it. On the night lounge of the test set it took out two 2.4 m slabs the layout had made of "textured beige plaster" (the walls' material), turned off four ceiling strip lights the sentence never asked for, and made the brass floor lamp the light of the room at 2700 K. `BLENDER_DIRECT=0` skips the look, `1` gives one round.

**Exposure by the hour.** By day a middling surface comes out middle grey, as before. At dusk and night the brightest patches set it (the pools under the lamps at the 96th percentile by night), so the rest falls dark as it does to the eye, never more than 1.5 stops under the daytime exposure. The art direction's stops are added on top.

**White balance.** A camera balanced for tungsten light shows a 2700 K lamp as warm cream, not orange. The bake measures the average colour of the light it baked and takes 60 % of that cast out of the lightmaps, the corner light, the panorama and the sky. Lamp colours chosen by temperature are given as a camera balanced for 4000 K sees them.

**Relief.** After the light is baked, each picture that colours the room's surfaces (walls, floor, ceiling, panelling, skirting) gets a normal map made from its own fine detail: the grain of plaster, the joints of boards. The lightmaps hold no direction, so the viewer shades that relief as if lit from above. `BLENDER_RELIEF=0` leaves it out.

**The viewer's film.** Worlds built in Blender are drawn in high dynamic range into a 4x multisampled buffer, then: contact shadows at a small scale (GTAO, where the lightmaps are too coarse), a glow round lamps, bright panes and sunlit patches (bloom), and Neutral tone mapping (AgX was tried and greys the colours out). `?post=0` on the viewer's address turns it off, `?tm=agx|aces|neutral&ev=…` picks another curve and exposure. Splat worlds are drawn as before.

Not taken from the proposal, for now: replacing the direct method's numpy geometry with Blender Geometry Nodes, full PBR material sets, TRELLIS-2 heroes and a CC0 kitbash library (new downloads, and the layout is the bigger fault), terrain for outdoors, and a WebGPU viewer.

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

- Rich takes about 23 minutes with art direction, over the 20-minute budget, and the model builds only the plain furniture in it; the sculpted things and the room come from the direct method.
- The model's own shapes are simple. In materials given to the piece they read well as shelves, cabinets and tables; what stands on the shelves is blocky.
- The one look checks shape only, from one side, and is not always right.
- The sky through a window is a picture at infinity: no parallax with nearby buildings. Reflections come from one panorama taken at the starting point, so they are a little off elsewhere in the room.
- The art director judges five small renders. It can mistake things (it has called a lamp that was already off "on"), and it can only take out, set down, scale and relight: it cannot add what is missing or mend a badly painted sculpted thing.
- A room whose walls are full can lose its door to a window.
- Quick, outdoors: trees and plants are beyond simple shapes.
- No moving things, no sound and no export for Unreal yet.
