# Contributing

The goal is the most realistic AI-generated world one 64 GB Mac can make. Anything that moves the status table in the [README](README.md) from "missing" to "rough", or from "rough" to "works", is welcome.

## The rules

Every change has to keep these. They are the point of the project.

- **One machine, 64 GB of memory max.** Peak memory for a whole build must stay under 48 GB, leaving room for the system. Run one large model at a time and unload it after use.
- **Open models only.** Weights must be downloadable and runnable locally. Say in your pull request which model you add, its size, and its licence.
- **No accounts, no cloud, no analytics, no external APIs.** Nothing may call out once the models are downloaded.

## Where to start

- **Close-up detail.** Worlds hold from across the room and fall apart up close.
- **Photo worlds that live.** Rooms from photos have no sound and nothing moves.
- **Export to Godot.** Visuals, colliders, sound and motion, all of it.

Issues labelled `good first issue` and `help wanted` are a good place to look.

## Two ways to contribute

1. **Pull request.** Fork, branch, change, open a pull request against `main`.
2. **Fork and tell us.** If you take the project somewhere of your own, open an issue with a link to your fork and what it does better. We will pull in what fits.

## Showing it works

This project is judged by how worlds look, so a pull request that changes what a world looks like, sounds like or how it moves needs evidence:

- **Before and after screenshots from the viewer, taken off-centre.** Walk into a corner or up close. A world that only looks good from the start view is not good.
- **The prompt or photos you used**, so someone else can rebuild it.
- **Build time and peak memory**, if your change touches the pipeline.

Run the unit tests before opening a pull request:

```bash
cd thread && .venv/bin/python -m unittest discover tests
```

CI runs the same tests plus syntax checks on every pull request. It cannot run the models, so it does not show that a build still works; your screenshots do that.

## Code style

- Keep to the style of the file you are in. Most files are compact, with short plain comments.
- Prefer general fixes over special cases. Do not fix one prompt; fix the reason that prompt failed.
- A world built before your change should still load and look the same, unless improving it is the point of the change.
- `thread/chisel_ring.py` runs in `spike/.venv`, which has no scipy, open3d or trimesh. Do not import them there.

## Reporting a problem

Open an issue with the bug template. Include the prompt or the number of photos, the detail level, the last lines of the world's `job.log`, and a screenshot.
