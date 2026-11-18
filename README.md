# Prometheus Worlds

**The frontier of locally generated worlds.**

How good a world can you generate on a 64 GB Mac? This project wants to find out. It is open source, and contributions are welcome.

**[Join the Discord](https://discord.gg/MgucVj7sJ)** · [Website](https://prometheus-worlds.vercel.app)

![Snowy Village Square, built from one sentence](docs/snow-village.webp)

| | |
|---|---|
| ![Alchemist's Tower Study](docs/tower-study.webp) | ![Cherry Blossom Garden](docs/blossom-garden.webp) |
| Alchemist's Tower Study | Cherry Blossom Garden |
| ![Palm Conservatory](docs/conservatory.webp) | ![The office, rebuilt from 20 phone photos](docs/office-splat.jpg) |
| Palm Conservatory | The office, rebuilt from 20 phone photos |

Every world above came from a sentence or a handful of photos, built on one Mac with no cloud service.

## What it is

Prometheus Worlds is an open source world builder. It does four things:

- **Sentence to world.** Generate a world from a description.
- **Photos to world.** Rebuild a real room from photos.
- **Stitching.** Put worlds side by side and walk from one to the next.
- **Export.** Take a world into a game engine. Unreal Engine first, as glTF, which Godot and Blender read too.

A world is four things: **visuals** (what you see from anywhere you stand), **colliders** (what you can't walk through), **sound** (what you hear, and where from) and **motion** (what moves without you).

## Status

| | Visuals | Colliders | Sound | Motion |
|---|---|---|---|---|
| Sentence to world | Indoors: clean geometry and light, object paint rough up close; outdoors rough | Works | Works | Indoors only sway; outdoors works, rigid |
| Photos to world | Rough, needs 60+ photos | Works, some holes | Missing | Missing |
| Stitching | Works | Works | Untested | Untested |
| Export | Sentence worlds (direct method): geometry, textures and lights as glTF; checked in Blender, not yet in Unreal | Unreal makes them from the meshes | Missing | Missing |

Every "rough" and "missing" is a place to help. See [CONTRIBUTING.md](CONTRIBUTING.md).

## The rules

- One machine, 64 GB of memory max.
- Open models only.
- No accounts, no cloud, no analytics, no external APIs.

These are what make it a challenge. A change that breaks one of them will not be merged.

## Requirements

- An Apple Silicon Mac with 64 GB of memory. Developed on an M4 Pro.
- macOS 14 or later.
- [Xcode](https://apps.apple.com/app/xcode/id497799835). Nothing else: the app brings its own Python and everything it needs.
- About 90 GB of free disk for tools and model weights.

## Build it

1. Clone the repository.
2. Open `Prometheus Worlds.xcodeproj` in Xcode.
3. Press **Run**.

On first launch the app sets itself up and shows its progress: a self-contained Python, its packages (including COLMAP, as pycolmap), the two text models, the two picture models that worlds from a sentence need and the 3D shape model (about 77 GB of downloads). Python and its packages go inside the project folder and the models into the Hugging Face cache (`~/.cache/huggingface`); nothing else is installed on the system, and every one of these downloads is pinned and checked. That takes a while and needs the internet once. After that, a room built from a sentence needs no internet. Two more models still download the first time they are used, at the revision mflux pins: Depth Pro (the first world from photos) and the Z-Image ControlNet (the first outdoor place, or the first build with the earlier method).

If you set the app up before the direct method arrived, the next launch fetches the three models it added (about 67 GB) before the app opens.

If a step fails, the window shows why. Fix it, quit, and open the app again: setup carries on from where it stopped.

The app runs from your checkout, so keep the folder where it is. Your worlds are saved in `thread/worlds/`.

**From the command line instead:**

```bash
./setup.sh
```

```bash
app/build.sh
```

`setup.sh` needs only Xcode and is safe to run again. `./setup.sh --no-shapes` skips the 3D shape model; worlds from a sentence are then built by the earlier method, with objects as painted boxes. If some of the models are already on the Mac in another Hugging Face cache, `PROMETHEUS_ADOPT=/path/to/hub ./setup.sh` links them in instead of downloading them again. `app/build.sh` builds `app/Prometheus Worlds.app` with Xcode's tools. To run without the app at all:

```bash
thread/.venv/bin/python thread/server.py 8765
```

Then open `http://127.0.0.1:8765/viewer/index.html`.

## Using it

In the app, choose **New world…**, then **From your photos** or **From a prompt**. One world is built at a time, and a panel in the corner shows progress. A world from a sentence takes about 20 minutes at Standard detail, longer if the sentence names many large things.

In the viewer: **W A S D** to move, drag to look, **Q** and **E** for down and up, Shift to go faster. Tick **Tour** for a hands-free glide, **Sound** for music and scene sounds, and **Drop balls** to test the colliders. **Stitch worlds…** puts worlds side by side. **Export for Unreal…** saves the world shown as glTF files for Unreal Engine, with a README beside them on how to import them; **Show in Finder** opens their folder when it is done.

## Models

| Model | Job | Runs through |
|---|---|---|
| Qwen3.5 35B-A3B (MLX 4-bit) | The plan of a world: its things, its style guide, its material choices | mlx-lm (MLX) |
| FLUX.2 Klein 4B (8-bit at load) | Concept picture, materials, object paint | mflux (MLX) |
| Z-Image Turbo | One picture of each object to sculpt | mflux (MLX) |
| Qwen3.5 9B (MLX 4-bit) | Earlier method and outdoor places: layout sentence to a blockout | mlx-lm (MLX) |
| Z-Image-Turbo-Fun-Controlnet-Union 2.1 (`2601-8steps`) | Earlier method and outdoor places: painting views that follow the blockout | mflux (MLX) |
| Depth Pro | Depth for each photo | mflux (MLX) |
| Hunyuan3D 2.0 shape (mini) | One picture of an object to a 3D mesh | [Hunyuan3D-Swift](https://github.com/ZimengXiong/Hunyuan3D-Swift) (MLX) |

COLMAP (through pycolmap), msplat, Open3D, three.js, Spark and Rapier do the rest. Exact versions are pinned in `runtime/*.lock` and `setup.sh`.

**Licences.** This project's code is MIT. The models keep their own licences, as their publishers state them: FLUX.2 Klein 4B is Apache-2.0 (its `LICENSE.md` is downloaded with it), and the Qwen3.5 models and Z-Image Turbo are published as Apache-2.0. Hunyuan3D is the exception: its weights are under the Tencent Hunyuan community licence, which is not an open-source licence and excludes some territories (the European Union, the United Kingdom and South Korea at the time of writing). If that licence does not suit you, run `./setup.sh --no-shapes`: nothing of Hunyuan3D is then downloaded, and sentences are built by the earlier method. Check each model's licence yourself before using this for anything beyond personal use.

## How it works

[docs/how-it-works.md](docs/how-it-works.md) has the detail: the pipelines step by step, moving things and outdoor places, stitching, speed measurements, what was tried and dropped, and the known limits. [docs/direct-method.md](docs/direct-method.md) describes how a sentence becomes a room.

In short: a sentence becomes a plan of 30 to 45 real things in a room; the room, its furniture and its objects are made as real geometry; a style guide drives a true-scale material for every surface and paint for every object; light is traced over all of it; and the splat is written straight from the result, with no training. An outdoor sentence still goes the earlier way: a blockout of boxes and generated shapes, painted views that follow it, and a Gaussian splat trained from them. Photos go through COLMAP, a splat trainer and Depth Pro for the mesh. The viewer is three.js with Spark for splats, Rapier for physics and Web Audio for sound; the three libraries are vendored in `thread/viewer/vendor/`, so there is no Node.js dependency.

## Repository layout

| Path | What |
|---|---|
| `Prometheus Worlds.xcodeproj` | Open this and press Run (generated from `project.yml` by XcodeGen) |
| `setup.sh` | First-run setup: Python, packages, models |
| `runtime/` | Pinned package lists (with hashes) for the two Python environments |
| `app/` | The macOS app: `main.swift`, `Info.plist`, `build.sh`, `make_icon.py` |
| `thread/server.py` | Local server: static files plus the job API |
| `thread/pipeline.py` | The pipelines: photos, and a sentence by either method |
| `thread/build/` | The direct method: layout, shapes, the room's geometry, style guide, materials, object paint, light, the splat writer, and `export.py`, the Unreal export |
| `thread/llm_boxes.py` | Earlier method: layout sentence to blockout |
| `thread/shapes.py` | Real 3D shapes for objects |
| `thread/chisel_ring.py` | Earlier method: painting views (runs in `spike/.venv`) |
| `thread/consistent.py` | Earlier method: making the views agree, training views |
| `thread/movers.py` | Lifting moving things out as their own splats |
| `thread/viewer/` | The viewer page |
| `thread/tests/` | Unit tests |
| `spike/` | The image-model environment and `blockout.py`, the blockout renderer |
| `experiments/` | Tests and studies behind past decisions |
| `ext/` | Third-party code and weights fetched at build time (not in git) |
| `thread/worlds/`, `thread/scenes.json` | Your built worlds (not in git) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: open an issue or a pull request, or fork it and send a link to your fork. Questions and show-and-tell go in the [Discord](https://discord.gg/MgucVj7sJ).

## License

[MIT](LICENSE).
