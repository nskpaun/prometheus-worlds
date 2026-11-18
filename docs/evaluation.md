# Checking how a world looks, quickly

A full prompt build takes about 20 minutes, nearly all of it the image model painting views. That is too slow to try an
idea and see whether surfaces got sharper. `thread/evals/` gives three speeds. All commands run from `thread/`, with
`.venv/bin/python`. Nothing here writes to `worlds/` or `scenes.json`; all output goes to `thread/evals/work/`.

## The three speeds

| Command | What it does | Measured (M4 Pro) |
|---|---|---|
| `evals/quick.py geometry [WORLD...]` | From the layout alone: how coarse the paint is on screen. No pictures, no GPU. | about 6 s a world, 18 s for the three |
| `evals/quick.py refinish [WORLD...] --label L` | Keeps a finished world's painted views; redoes "make the views agree", the training and the mesh, renders 24 fixed cameras, scores. | 196 s (garage-shapes), 207 s (ex-tower), 219 s (ex-blossom) the first time; about 20 s less once the world's reference pictures exist |
| `evals/quick.py refinish ... --label L --views-from L2` | Training only, on label L2's views. `--skip-agree` does the same in place. | 95 to 107 s a world |
| `evals/quick.py full --label L [NAME...]` | Builds the eight sentences of `evals/prompts.json` from nothing and scores each. | never run with the real models; expect about 20 min a world |
| `evals/quick.py compare A B` | Table of every score's change from label A to label B, marked better, worse, same or check. | instant |

WORLD is a folder name under `worlds/` or a path. Left out, the three reference worlds are used: `ex-tower` (indoor,
shapes, moving things), `garage-shapes` (large indoor hall), `ex-blossom` (outdoor).

Of a refinish, the agree step takes 68 to 88 s, the training 85 to 96 s, the mesh and the scoring together about 10 s,
the geometry scores 6 s (skipped when the training cameras are the same as last time) and making a world's reference
pictures 20 s (once per world). So the "3 minute" tier is 3.3 to 3.7 minutes the first time and about 3 after that.
Before the agree step was told to leave out the set with the moving things it took 173 s (ex-tower) and 197 s (ex-blossom).

To try an idea: change the code (or set an environment variable; they pass through to the steps), run
`quick.py refinish --label my-idea`, then `quick.py compare baseline my-idea`, then open the two pictures per world.

- **An idea that only touches the trainer needs no new views.** The agree step writes the same views bit for bit
  every run (checked by hash on all three worlds), so `--views-from baseline` gives exactly what a whole refinish
  would, in under 2 minutes. `--train "ARGS"` hands extra arguments to `train_scene.py`, for example
  `--train "--set refine_every=200"`.
- **The step count is fixed at the build's 10000 on purpose.** Shorter runs were tried on garage-shapes (5000 steps in
  41 s, 3000 in 24 s) and are no stand-in. The same trainer change read as stray_share 12% worse at 10000 steps, 3%
  better at 5000 and 25% better at 3000; a mild softening of the views that costs sharp_close 5.6% at 10000 steps cost
  only 2.2% and 0.8% at the shorter ones.
- **A label is a plain name** (letters, digits, dot, dash, underscore): it becomes a folder that is emptied and rewritten.

`evals/work/results.jsonl` holds one line per label and world (a rerun replaces its line): every score, seconds per
step, memory. Each run also writes two pictures to `evals/work/WORLD-L/`:

- `sheet.jpg`: the 24 renders, each with its sharpness. Thumbnails of 400 px, so blur that is plain in the app is mild here.
- `crops.jpg`: the middle 512x384 of each close and floor render at full size, each beside the same part of its
  reference (the paint that view should show; green = no paint there). Judge blur and damage here. The viewer on a
  retina screen still magnifies about 1.8 times more than these renders (1412 px focal length against 804).

## The cameras

`evals/cams.py` places the same 24 cameras every run, from the layout: 8 at the middle of the room looking level all
round ("centre"), 12 "close", 4 looking down across the floor ("floor").

- **Indoors** the close cameras stand 0.8 m from each wall, straight on (`close_x0`, `close_x1`, `close_z0`, `close_z1`:
  the "wall" views, where the blur complaint lives) and turned 45 degrees, and at the two largest objects.
- **At an object** the camera is 0.8 m level across the floor from its box, at eye height: 1.3 to 1.7 m from a low
  thing's surface.
- **Outdoors** there are no walls. The cameras at the edges of the ground look out over the backdrop 20 m away, so
  they are named `far_...` and scored as their own kind; only the object views count as close. ex-blossom has 6 of each.

## The scores

Geometry (`density.py`), from where the cameras and surfaces are. No splat is needed.

| Score | Meaning | Better |
|---|---|---|
| `near_r` | Painted pixels per screen pixel on what the close and floor cameras see (median over the pixels that have paint). 1 = the paint is as fine as the screen; 0.2 = one painted pixel is stretched over 5 screen pixels. | higher |
| `wall_r` | The same for the four wall views (mean of their medians). | higher |
| `near_worst` | The same along the direction the paint is thinnest: the smear on a surface painted from side-on. | higher |
| `near_soft` | Share of those pixels with less than half a painted pixel each. | lower |
| `unpainted` | Share no painted view shows (filled in from the paint around it). Not mixed into the medians above. | lower |
| `near_train_r` | Training-view pixels per screen pixel: the 1024x768 views the splat is trained on are a second limit. An upper bound (hidden surfaces are not excluded). | |
| `near_eff_r` | The lesser of paint and training views, per pixel. A paint size that lifts `near_r` above this buys nothing by itself. | higher |

The screen is taken as 1600x900 at one pixel per point; on a retina screen every ratio halves.

Pictures (`image_scores.py`), from the renders of the trained splat:

| Score | Meaning | Better |
|---|---|---|
| `sharp_wall` | Mean sharpness of the four straight-on wall views. Indoors only. | higher |
| `sharp_close`, `sharp_floor`, `sharp_centre`, `sharp_far` | Mean sharpness of that kind of camera. Sharpness is fine detail as a share of coarse detail: a painted picture scores about 0.5, a badly blurred one 0.15. | higher |
| `ref_corr` | Does the render show the same shapes as its reference (`reference.py`: the paint laid on the blockout, seen from that camera). 1 = yes. Blur leaves it alone; speckle, streaks, tears and misplaced paint pull it down. | higher |
| `ref_err` | Colour difference from the reference after a slight blur (0 to 1). Holes and wrong colours. | lower |
| `bg_share`, `bg_max` | Share of pixels showing the trainer's magenta background, mean and worst view. It shows wherever the splat has a gap from that camera (the magenta patches on the garage's cars and floor). | lower |
| `detail_kept`, `kept_wall` | Sharpness of a render over the sharpness of its own reference, mean over the close views, and over the walls. 1 = as crisp as the paint enlarged to that view. Shown, not judged: in the refinish tier it is `sharp_close` on another scale. | |
| `worst` | The three least sharp views. | |

Splat (`splat_scores.py`), from the splat file and the blockout mesh built for that same training:

| Score | Meaning | Better |
|---|---|---|
| `stray_share` | Share of splats (weighted by how solid they are) more than 5 cm off every surface. Mostly surface thickness. | lower |
| `stray15_share` | The same beyond 15 cm: the floaters. | lower |
| `needle_share` | Share that is a needle: longest axis over 4 times the middle one and over 3 cm. The streak artefact. | lower |
| `large_share`, `big_share` | Share whose longest axis is over 10 cm, over 25 cm. Mostly round discs on plain surfaces: coarseness, not stretch. | lower |
| `median_axis_cm`, `median_off_cm` | The median longest axis; the median distance from the mesh. | lower |
| `backdrop_stray_share` | Outdoors only: backdrop and sky splats more than 50 cm from where they belong. | lower |
| `aligned`, `pose_error` | Proof that splat, mesh and eval cameras are in one frame. If either fails the affected scores are not recorded. | |

### Why sharpness is never read alone

Sharpness counts any fine detail, so damage scores as detail. Measured on baseline renders (`image_scores.py --selftest`):

| ex-tower `close_z0` | sharpness | ref_corr | ref_err |
|---|---|---|---|
| as rendered | 0.145 | 0.954 | 0.015 |
| blurred 2 px | 0.118 | 0.959 | 0.015 |
| speckle added | 0.988 | 0.364 | 0.064 |
| 150 streaks added | 0.475 | 0.552 | 0.061 |
| a hole (background showing) | 0.145 | 0.901 | 0.070, bg 0.063 |

So compare calls a sharpness rise "better" only while `ref_corr`, `ref_err` and `bg_share` are not worse; otherwise it
says "check", and the crops decide. Likewise a splat with smaller or fewer splats scores better on the size and stray
shares even if it now has gaps: the gaps show as `bg_share` and `ref_err`.

## Baseline (label `baseline`, 2026-11-16, paint 768x576)

| | ex-tower | garage-shapes | ex-blossom |
|---|---|---|---|
| near_r | 0.218 | 0.122 | 0.239 |
| wall_r | 0.142 | 0.081 | n/a |
| near_worst | 0.169 | 0.088 | 0.137 |
| near_soft | 0.869 | 0.879 | 0.756 |
| unpainted | 0.212 | 0.026 | 0.053 |
| near_train_r | 0.889 | 0.586 | 0.953 |
| near_eff_r | 0.218 | 0.122 | 0.239 |
| sharp_wall | 0.174 | 0.155 | n/a |
| sharp_close | 0.268 | 0.249 | 0.322 |
| sharp_floor | 0.380 | 0.355 | 0.394 |
| sharp_centre | 0.449 | 0.581 | 0.518 |
| sharp_far | n/a | n/a | 0.424 |
| ref_corr | 0.979 | 0.901 | 0.966 |
| ref_err | 0.0093 | 0.0193 | 0.0127 |
| bg_share / bg_max | 0 / 0 | 0.0005 / 0.0054 | 0.0006 / 0.0104 |
| detail_kept / kept_wall | 1.03 / 1.08 | 1.01 / 1.27 | 0.95 / n/a |
| stray_share | 0.111 | 0.202 | 0.178 |
| stray15_share | 0.0035 | 0.0010 | 0.0011 |
| needle_share | 0.115 | 0.179 | 0.142 |
| large_share | 0.074 | 0.206 | 0.287 |
| big_share | 0.0022 | 0.0084 | 0.0397 |
| median_axis_cm | 4.69 | 6.22 | 5.77 |
| median_off_cm | 1.97 | 2.57 | 2.03 |
| splats | 918,677 | 879,181 | 1,054,299 |
| agree / train / all, seconds | 68 / 96 / 207 | 73 / 85 / 196 | 88 / 94 / 219 |
| memory: largest process / whole step | 7.0 / 9.6 GB | 6.5 / 9.4 GB | 8.5 / 10.5 GB |

What the baseline says about the known problem:

- **The blur on close walls is the paint, not the trainer.** `detail_kept` is 0.95 to 1.03: the splat shows the paint
  about as crisply as the paint itself enlarged to that view (see `crops.jpg`). A wall seen from 0.8 m gets one painted
  pixel per 7 to 12 screen pixels (`wall_r` 0.14 and 0.08).
- **The training views are not yet the limit**: `near_train_r` is 4 to 5 times `near_r`. The splats themselves are
  close to it: the median splat is 5 to 6 cm long, and a painted pixel on a wall seen from 0.8 m is about 1 cm.
- **The streaky ground outdoors is also in the reference**: it comes from laying far, low paint on the ground, before
  any training.

## How much is noise

The agree step is exact: the same views bit for bit. Training is not. Three trainings of the same views per world
(labels `baseline`, `noise`, `noise2`) differed by at most:

| Score | Largest range over three runs | compare's limit |
|---|---|---|
| sharp_close, sharp_floor, sharp_centre, sharp_far | 0.7% | 2% |
| sharp_wall | 2.8% (garage; 0.7% in ex-tower) | 6% |
| one view's sharpness | 0.011 (the garage's plain wall, 7%); 0.005 elsewhere | shown, not judged |
| ref_corr | 0.005 | 0.01 |
| ref_err | 2.2% (0.0002) | 5% and 0.001 |
| bg_share | 0.0003 (bg_max 0.0025) | 0.002 |
| stray_share | 2.2% here, 3.3% in an earlier set of three | 7% |
| stray15_share | 0.0013 (37% of its value in ex-tower) | 0.003 |
| needle_share, large_share | 1.2% | 5% |
| big_share | 4.5% (0.0003) | 10% and 0.001 |
| median_axis_cm, median_off_cm | 0.5% | 3% |
| backdrop_stray_share | 1.4% | 5% |
| splats | 0.3% | not judged |
| training seconds | 9% here; 21% seen with other jobs on the machine | not judged |
| geometry scores | none | 2% |

`compare baseline noise`, `baseline noise2` and `noise noise2` mark nothing better or worse. A change smaller than
the limit can still be real: compare prints the percentage, so the same small move in all three worlds is visible.
A softening known to be real, training views shrunk to 640 px wide and enlarged again, moved the old median
sharp_close by 5.6% in the garage.

## What the scores cannot tell you

- **Whether the world matches the sentence, and whether it looks good.** No score reads the picture's content. A person
  has to look at the pictures, and for the full tier walk the world.
- **Detail finer than the paint has nothing to be checked against.** The reference is the paint enlarged. A method
  that invents finer detail raises sharpness and may lower `ref_corr`; compare then says "check" and the crops decide.
- **The geometry scores and the reference read the finished world's paint and layout**, by the same rule the build
  lays the paint by: `thread/owner.py` is the one copy, used by `consistent.py`, `density.py` and `reference.py`. Change
  the rule there and all three follow. `WORLD_GAPS=0` gives the rules as they were at the baseline (the agree step then
  writes the baseline's views bit for bit: `experiments/gaps/agree_check.sh`). `near_train_r` follows the run's own
  training cameras.
- **A world painted to a plan** (`detail.json` beside `boxes.json`, written by `chisel_ring.py prep` when `pipeline.py`
  hands it the time there is to paint; `thread/detail.py`) has usual views that were left out and detail views after
  them. `owner.py` holds the rule for what a detail view owns, so the geometry scores and the reference follow the
  build. `refinish` copies the plan with the pictures. To try a plan on a finished world without painting its usual
  views again: `experiments/detail_pick/run.sh LABEL "WORLD=BUDGET"` (it paints only the detail views, in a copy).
- **A detail picture lowers `sharp_*` where it removes streaks.** In the garage the plain wall's `close_x0` fell from
  0.160 to 0.078 because a white seam across it went; the crops, `ref_err` and the splat scores say which it is.
- **`unpainted` is what no painted view owns, whatever it is filled with.** The build never leaves it bare: floor and
  walls continue the paint around the gap, an object's unseen part takes the object's own paint, the backdrop above
  its paint fades into the sky. Whether a fill reads as surface is for the crops to say (green in a reference = filled).
- **A plain wall scores low on sharpness** with nothing wrong (the garage's concrete wall, 0.15); its own reference is as plain, so its kept is 0.82.
- **`bg_share` only sees dark magenta.** A gap in front of other splats is not seen. Paint that is itself magenta is not counted.
- **The refinish tier cannot judge anything upstream of the paint**: layout, shapes, how many views are painted, their
  size, the painter's prompt. Those need the full tier, which has never been run with the real models.
- **The renders are 1024x768 from the trainer**, not the viewer's own drawing; the moving things are not in them (only
  the room is trained, and the agree step is told to skip the set with the moving things).
- **Memory of the whole step** is resident memory added over all its processes each second. Pages the workers share
  are counted once each, so it is an upper bound.
- `stray_share` at 5 cm is mostly splats sitting a few centimetres behind a wall, not floaters; use `stray15_share` for those.
