# Prometheus Worlds

**The frontier of locally generated worlds.**

How good a world can you generate on a 64 GB Mac? This project wants to find out. It is open source, and contributions are welcome.

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
- **Export.** Take a world into a game engine. Godot first.

A world is four things: **visuals** (what you see from anywhere you stand), **colliders** (what you can't walk through), **sound** (what you hear, and where from) and **motion** (what moves without you).

## Status

| | Visuals | Colliders | Sound | Motion |
|---|---|---|---|---|
| Sentence to world | Rough up close | Works | Works | Works, rigid |
| Photos to world | Rough, needs 60+ photos | Works, some holes | Missing | Missing |
| Stitching | Works | Works | Untested | Untested |
| Export | Missing | Missing | Missing | Missing |

Every "rough" and "missing" is a place to help. See [CONTRIBUTING.md](CONTRIBUTING.md).

## The rules

- One machine, 64 GB of memory max.
- Open models only.
- No accounts, no cloud, no analytics, no external APIs.

These are what make it a challenge. A change that breaks one of them will not be merged.

## Requirements

- An Apple Silicon Mac with 64 GB of memory. Developed on an M4 Pro.
- macOS 14 or later, Xcode command line tools (`xcode-select --install`), and [Homebrew](https://brew.sh).
- Tens of GB of free disk for model weights, which download on first use.

## Build it

From the repository root:

```bash
brew install uv colmap ollama libusb node
```

```bash
ollama pull qwen3.5:9b
```

```bash
uv venv --python 3.12 thread/.venv && uv pip install --python thread/.venv/bin/python -r thread/requirements.txt
```

```bash
uv venv --python 3.12 spike/.venv && uv pip install --python spike/.venv/bin/python -r spike/requirements.txt
```

```bash
npm install --prefix thread/viewer
```

Optional, for real object shapes in invented rooms (without it objects are painted boxes):

```bash
git clone https://github.com/ZimengXiong/Hunyuan3D-Swift ext/Hunyuan3D-Swift && (cd ext/Hunyuan3D-Swift && swift build -c release && hf download zimengxiong/hunyuan3d-mlx-shape-small --local-dir weights/shape-small)
```

`swift build` does not produce MLX's Metal shader library unless full Xcode is installed. Without Xcode, take it from the matching Python package (the version must match the `mlx-swift` the port pins, 0.31.1 here):

```bash
uv pip install --target /tmp/mlx0311 mlx==0.31.1 && cp /tmp/mlx0311/mlx/lib/mlx.metallib ext/Hunyuan3D-Swift/.build/release/
```

```bash
app/build.sh
```

That produces `app/Prometheus Worlds.app`. Drag it to the Desktop or Applications and open it.

The app is a thin window around this folder: it starts `thread/server.py` and shows the page. It stores the folder's path when built, so rebuild it if the folder moves. It is signed ad hoc, for this Mac only.

To run without the app:

```bash
thread/.venv/bin/python thread/server.py 8765
```

Then open `http://127.0.0.1:8765/viewer/index.html`.

To redraw the icon, run `thread/.venv/bin/python make_icon.py` inside `app/`, then rebuild.

## Using it

In the app, choose **New world…**, then **From your photos** or **From a prompt**. One world is built at a time, and a panel in the corner shows progress. A prompt world takes about 30 minutes at Standard detail.

In the viewer: **W A S D** to move, drag to look, **Q** and **E** for down and up, Shift to go faster. Tick **Tour** for a hands-free glide, **Sound** for music and scene sounds, and **Drop balls** to test the colliders. **Stitch worlds…** puts worlds side by side.

## Models

| Model | Job | Runs through |
|---|---|---|
| Qwen3.5 9B | Layout sentence to a blockout | Ollama |
| Z-Image Turbo + Z-Image-Turbo-Fun-Controlnet-Union 2.1 (`2601-8steps`) | Painting views that follow the blockout | mflux (MLX) |
| Depth Pro | Depth for each photo | mflux (MLX) |
| Hunyuan3D 2.0 shape (mini), optional | One picture of an object to a 3D mesh | [Hunyuan3D-Swift](https://github.com/ZimengXiong/Hunyuan3D-Swift) (MLX) |

COLMAP, msplat, Open3D, three.js, Spark and Rapier do the rest. Each model has its own licence; check them before using this for anything beyond personal use.

## How it works

[docs/how-it-works.md](docs/how-it-works.md) has the detail: both pipelines step by step, moving things and outdoor places, stitching, speed measurements, what was tried and dropped, and the known limits.

In short: a sentence becomes a blockout of boxes and generated shapes, an image model paints views that follow it, the views are made to agree and re-projected to about 115 training views, and a Gaussian splat is trained from them. Photos go through COLMAP, a splat trainer and Depth Pro for the mesh. The viewer is three.js with Spark for splats, Rapier for physics and Web Audio for sound.

## Repository layout

| Path | What |
|---|---|
| `app/` | The macOS app: `main.swift`, `build.sh`, `make_icon.py` |
| `thread/server.py` | Local server: static files plus the job API |
| `thread/pipeline.py` | The two pipelines |
| `thread/llm_boxes.py` | Layout sentence to blockout |
| `thread/shapes.py` | Real 3D shapes for objects |
| `thread/chisel_ring.py` | Painting views (runs in `spike/.venv`) |
| `thread/consistent.py` | Making the views agree, training views |
| `thread/movers.py` | Lifting moving things out as their own splats |
| `thread/viewer/` | The viewer page |
| `thread/tests/` | Unit tests |
| `spike/` | The image-model environment and `blockout.py`, the blockout renderer |
| `experiments/` | Tests and studies behind past decisions |
| `ext/` | Third-party code and weights fetched at build time (not in git) |
| `thread/worlds/`, `thread/scenes.json` | Your built worlds (not in git) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: open an issue or a pull request, or fork it and send a link to your fork.

## License

[MIT](LICENSE).
