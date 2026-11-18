# Unpainted surface in prompt worlds: what it is, what was changed, what was not reached

2026-11-17. All commands from `thread/`. Labels in `evals/work/results.jsonl`: `baseline`, `gaps1` to `gaps7`.
`gaps7` is the code as left on by default.

## Result in one table

`unpainted` = share of the close and floor eval cameras' pixels that no painted view owns (geometry tier).

| | ex-tower | garage-shapes | ex-blossom |
|---|---|---|---|
| unpainted, baseline | 0.212 | 0.026 | 0.053 |
| unpainted, default now (`gaps7`) | **0.032** | **0.026** | **0.048** |
| unpainted with the look-down planner (`gaps5`, painted; OFF by default) | 0.018 | 0.004 | 0.048 by geometry, not painted |
| near_r baseline -> gaps7 | 0.218 -> 0.267 | 0.122 -> 0.122 | 0.239 -> 0.239 |
| sharp_close | 0.268 -> 0.282 | 0.249 -> 0.247 | 0.322 -> 0.324 |
| sharp_floor | 0.380 -> 0.420 | 0.355 -> 0.370 | 0.394 -> 0.395 |
| sharp_wall | 0.174 -> 0.177 | 0.155 -> 0.154 | n/a |
| ref_corr | 0.979 -> 0.980 | 0.901 -> 0.909 | 0.966 -> 0.967 |
| ref_err | 0.0093 -> 0.0091 | 0.0193 -> 0.0188 | 0.0127 -> 0.0120 |
| bg_share / bg_max | 0 / 0 -> 0 / 0 | 0.0005 / 0.0054 -> 0.0001 / 0.0014 | 0.0006 / 0.0104 -> 0.0006 / 0.0120 |
| stray_share | 0.111 -> 0.106 | 0.202 -> 0.219 (**worse**, +8.8%, limit 7%) | 0.178 -> 0.178 |
| median_off_cm | 1.97 -> 1.91 | 2.57 -> 2.68 (**worse**, +4.1%, limit 3%) | 2.03 -> 2.08 |
| training views | 125 -> 132 | 114 -> 122 | 128 -> 136 |
| agree step, room only, s | 68.4 -> 60.7 | 73.3 -> 53.2 | 87.5 -> 77.0 |

**The target (under 2% on all three) is NOT met by the default.** ex-tower went from 21.2% to 3.2%; garage-shapes
and ex-blossom are where they were. The only thing found that takes the garage and the tower under 2% is painting
other views (the planner below), and that was rejected on the pictures. Nothing found takes ex-blossom under 2%.

## What "unpainted" really was

The task's premise was that unpainted surface is left bare and shows magenta. It is not left bare: `consistent.py`
already filled every such pixel in every training view (floor and walls from the nearest paint, blurred; an object's
unseen part from its own paint; the backdrop above its paint fading to sky). Magenta in a density map and green in a
reference mean "no painted view owns this", not "nothing there". Real magenta (the trainer's background through the
splat) was only in the garage: 0.05% of pixels, 0.5% of `floor_2`.

Why each world had it (`experiments/gaps/diagnose.py`, share of all near pixels, baseline rules):

| cause | ex-tower | garage-shapes | ex-blossom |
|---|---|---|---|
| **shade**: floor within its height (0.4 to 2 m) of a thing that moves away, thrown out by rule | 0.165 | 0 | 0.005 |
| **lift**: seen only within 40 px of a moving thing's outline, thrown out by rule | 0.033 | 0 | 0.004 |
| **hidden**: in some picture's frame, but something stands in front in every one | 0.013 | 0.026 | 0.024 |
| **frame**: in no picture's frame | 0.001 | 0 | 0.020 |
| behind a shape's surface, too few pixels, outline mismatch | 0.0001 | 0.0002 | 0.0005 |

- ex-tower: a 1.6 m wide, 1.8 m tall globe spins in the middle of a 6 x 8 m room. The shade rule discarded all floor
  within 1.8 m of it: a 5.2 x 5.1 m patch. Not a camera problem, not grazing angle, not distance: the owner rule.
- garage-shapes: all of it is occlusion. The platform and car in the middle hide the floor on each side from the far
  half of the ring, and the near cameras look over it (floor 1.65%); the rest is wall and car sides behind cars.
- ex-blossom: the backdrop above the top edge of every picture (2.0%; the eval camera looks 32 degrees up, the
  painted views 22), and the far sides and undersides of the two tree crowns (1.7%).

After the changes: ex-tower hidden 1.3%, lift 1.1%, shade 0.75%, frame 0.1%; garage hidden 2.6%; blossom hidden
2.4%, frame 2.0%, shade 0.15%, lift 0.1%.

## What was changed (on by default; `WORLD_GAPS=0` turns all of it off)

1. **One owner rule, in `thread/owner.py`.** `consistent.py` and `evals/density.py` each had a copy; both now call
   `owner.owners()`. With `WORLD_GAPS=0` the agree step writes the baseline's training views bit for bit on all
   three worlds (`experiments/gaps/agree_check.sh`: 125, 114, 128 views, same md5), so the move changed nothing.
2. **The floor round a moving thing keeps its paint unless the paint shows a shadow** (`owner.shade`). The floor in
   the band and 1 m beyond is laid out on a 5 cm grid with the paint it would take; floor darker than 0.82 of the
   ring's median brightness (after a 15 cm blur) and joined to the thing is shadow, with 10 cm more around. Only that
   is discarded. ex-tower: 16.5% of near pixels -> 0.75%. The geometry tier now reads the painted pictures for
   worlds with such a thing (the only place it does); it still takes 6 s.
3. **The band round a moving thing's outline is graded.** 16 px (as for any shape): never the room's paint. 16 to
   40 px: taken only where no other view sees the point. ex-tower: 3.3% -> 1.1%. The bookshelf behind the globe was
   a smear of its own nearest paint and is now rows of books (`close_z1` sharpness 0.206 -> 0.213, with the planner
   0.257).
4. **Every gap in a floor or wall continues the paint around it with grain** (`mend`, until now only where a thing
   was lifted out), not a 24 cm blur of the nearest cell. Tried alone as `gaps1`: garage bg_share 0.0005 -> 0.0003,
   inside the noise. It does not hurt and is kept, but it is not what removes the magenta.
5. **Training views are added where few look** (`owner.cover`, costs no painting). The garage's unpainted floor was
   shown by 3 training views (median; 40% of it by 2 or fewer) against 8 for painted floor: the splat was thin there
   whatever the views showed. A walker (1 m grid, eight headings, level and 35 degrees down) lists what he comes
   within 5 m of; while points are shown by fewer than 4 views from within 5 m, his downward view that helps most is
   added, at most 8, stopping under 3% gain. Adds 7 / 8 / 8 views. Garage floor views: bg 0.0054 -> under 0.001.

Time: the planner aside, nothing is painted that was not painted before. In the agree step the additions cost about
3 s (cover) + under 1 s (shade) + 7 or 8 more views to render; replacing PIL's 33 px maximum filter with scipy's for
the band round shapes (0.58 s a view, 16 to 18 views, same pixels) takes back 9 to 10 s. Measured, room only:
68.4 -> 60.7, 73.3 -> 53.2, 87.5 -> 77.0 s. Measured, both sets as the pipeline runs it, ex-tower: 116.2 s now
against 108.2 s with `WORLD_GAPS=0` on the same code, which itself is about 9 s under the code before (16 views x
0.58 s): net about -1 s. Training is 10000 steps whatever the view count (95.9 -> 78.3 s here; the machine was
quieter). No added build time.

## What the sheets show (`evals/work/WORLD-gaps7/sheet.jpg`, `crops.jpg`)

- **ex-tower.** The floor is now the painted floor: dark boards and stone, real texture, not the soft brown fill.
  It is a patchwork: straight seams where one view's paint meets another's in a different tone, in `floor_0`,
  `floor_1`, `floor_2`. That is the paint's own inconsistency, hidden before by the fill. A chair the painter put
  under the desk (not in the layout) lies flat on the floor in `close_obj0-desk`; the old rule hid it by accident.
  The remaining fill (a rounded patch at the globe's foot in `floor_0`, `floor_1`) blends in. No magenta.
- **garage-shapes.** The filled floor beside the platform (`floor_0`, `floor_2`) is a soft, blotchy stretch with a
  few faint pink flecks: a smear, not a plausible floor, but no longer holed with magenta. The cars' wall-side
  corners still show magenta (`close_obj0-car-corner-fl` 0.14%).
- **ex-blossom.** Unchanged to the eye. `far_z1` and `far_x0_-45` still show blurred blocks where the backdrop has
  no paint.

## Tried and rejected

- **Choosing the look-down painted views by coverage (option b), `owner.lookdown`, OFF unless `WORLD_LOOKDOWN=plan`.**
  Same number of views, placed one by one where the ring leaves most of the walker's points unowned. By geometry
  and after painting only those views in copies (`experiments/gaps/paint_lookdown.sh`, 63 to 69 s a view, as
  before) and a refinish (`gaps5`): ex-tower unpainted 0.018, near_r 0.265, sharp_wall +7.2%; garage unpainted
  0.004, near_r 0.141 (+15.6%), ref_corr +0.016, but sharp_floor -3.0% and stray_share +21.5% (both "worse").
  **Rejected on the pictures:** three of the garage's six planned views show open floor, and the painter stood a car
  on it that is not in the layout. It lies flat on the floor in `evals/work/garage-shapes-gaps5/sheet.jpg`
  (`floor_0`, `floor_2`, `centre_4`). Adding "The floor is plain and unbroken." to those prompts (`gaps6`) turned
  the cars into ghosts, not into floor. ex-tower got a flat hatch in `floor_3`. ex-blossom gains nothing (0.0475 ->
  0.0479 by geometry) and was not painted. The planner, its file (`lookdown.json`) and its tests stay in the code.
- **12 added training views** (`gaps2`): garage bg 0 in every view and ref_corr +0.026, but sharp_wall -8.1%
  (worse), stray_share +19%; blossom got 12 views of ground outside the walking area for nothing, stray_share +11%.
- **Counting training views at any distance** (`gaps3`): mended the floor, not the cars (`close_obj1` bg 1.1%),
  stray_share +24%.
- **A neutral trainer background (option d): not done.** The viewer draws its own background (`#111`), so magenta
  never shows in the app; it is the eval's marker for a gap. Changing it would blind `bg_share` and fix nothing.

## Not done, not verified

- No full build was run. The default was checked by refinish only (the world's own pictures) and by running the
  agree step both ways on an ex-tower copy; `chisel_ring.py prep` writes the baseline's 18 garage depth maps
  unchanged (test).
- garage-shapes `stray_share` +8.8% and `median_off_cm` +4.1% are over the noise limits. `stray15_share` (floaters)
  is unchanged at 0.001. Not looked into further.
- ex-blossom's 4.8% needs paint the present views cannot give: pictures that look higher up the backdrop, and views
  of the tree crowns from outside and below.
- The seams in ex-tower's floor and the invented chair are paint, upstream of this work.
- The shade test was only seen at work on two worlds with such things (ex-tower, ex-blossom); the rest is tests.

## Files

- `thread/owner.py` (new): `owners`, `view`, `rank`, `shade`, `walker`, `cover`, `lookdown`.
- `thread/consistent.py`: uses them; grain in every gap; added training views before the upward ones.
- `thread/chisel_ring.py`: `plan_lookdown` (off by default), reads `lookdown.json` if a world has one.
- `thread/evals/density.py`, `reference.py`, `quick.py`, `refinish.py`: the shared rule; `owner.py` and
  `lookdown.json` in the cache key and the copied inputs.
- `thread/tests/test_owner.py` (new, 8 tests), `thread/tests/test_layout.py` (garage depth test extended).
  `.venv/bin/python -m unittest tests.test_layout tests.test_owner`: 52 tests pass.
- `experiments/gaps/`: `diagnose.py`, `lookdown.py`, `agree_check.sh`, `paint_lookdown.sh`.
- Painted copies: `evals/work/ld/garage-shapes` (second paint), `evals/work/ld1/` (first paint, both worlds).
