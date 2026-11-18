# Detail pass, plan A: square-on tiles per surface

Planning only. Nothing here was painted or trained; every number is the geometry score (painted pixels per screen
pixel), not a look at a picture.

    cd thread
    .venv/bin/python experiments/detail_a/plan.py worlds/ex-tower --budget 6      # the plan and its predicted score, JSON
    .venv/bin/python experiments/detail_a/plan.py worlds/ex-tower --budget 8 --table   # also the score at 2, 4, 6, 8
    .venv/bin/python experiments/detail_a/score.py worlds/ex-tower                # the score as painted now

`score.py` is `evals/density.py`'s maths, copied (so edits there do not move these numbers), plus detail views as owners.
With no detail views it gives density.py's near numbers on all three worlds. `plan.py` picks the views.

## Approach

1. Tile every wall (indoors), each side of each large object, and the floor with square-on cameras. The distance is
   the one at which a 768x576 picture gives `--target` painted pixels per screen pixel to a walker: 0.8 m from a wall
   or object, or looking 35 degrees down at the floor from eye height. Default target 0.5: 1.37 m from a wall, a
   picture 1.74 x 1.31 m; the floor from 5.9 m up, or as high as the ceiling allows (0.3 m under it).
2. A camera with something between it and its wall backs off until that thing is 1.37 m away, so the tile becomes a
   view of the bookshelf, not of the wall behind it. Cameras inside objects or out of the room are dropped.
3. Pick tiles one at a time by the soft screen area each removes (the rise in min(r, 1), summed over screen pixels).
   The screens are the planner's own walker cameras (a grid of standing spots, spots 0.8 m along the walls, spots by
   each object side: about 130 to 140 cameras here), not the 24 eval cameras the result is reported on.
4. A detail view owns a point if the point is in its picture but not in the outer 4% (left for blending), its blockout
   shows that surface there, it sees it within 60 degrees of head-on, the point is not in the band round a shape or
   round a thing that moves away, and it brings more pixels than the ring owner. It can only raise r.

## Predicted score (cached worlds, ring as painted at 768x576, detail pictures 768x576, target 0.5)

near = close + floor eval cameras. soft = share with r < 1. worst close = the close camera with the lowest median r.

| world | N | near_r median | soft (r<1) | r < 0.5 | worst close camera | near pixels owned by detail |
|---|---|---|---|---|---|---|
| ex-tower | 0 | 0.163 | 99.7% | 89.7% | obj0-desk 0.000 | |
| | 2 | 0.190 | 99.0% | 84.4% | obj0-desk 0.000 | 7% |
| | 4 | 0.190 | 97.4% | 82.7% | obj0-desk 0.000 | 9% |
| | 6 | 0.190 | 96.3% | 81.9% | obj0-desk 0.000 | 10% |
| | 8 | 0.262 | 96.2% | 76.2% | obj0-desk 0.000 | 20% |
| garage-shapes | 0 | 0.118 | 98.6% | 88.1% | x0 0.074 | |
| | 2 | 0.126 | 97.9% | 85.1% | x1 0.074 | 3% |
| | 4 | 0.146 | 97.6% | 80.6% | x1 0.074 | 8% |
| | 6 | 0.153 | 96.2% | 77.8% | x1 0.076 | 11% |
| | 8 | 0.192 | 95.1% | 72.2% | x1 0.076 | 19% |
| ex-blossom | 0 | 0.255 | 98.2% | 72.9% | obj2-wooden-bench_45 0.089 | |
| | 2 | 0.293 | 96.8% | 70.6% | obj2-wooden-bench_45 0.089 | 10% |
| | 4 | 0.336 | 96.1% | 69.2% | z1 0.181 | 24% |
| | 6 | 0.341 | 95.8% | 68.5% | z1 0.181 | 25% |
| | 8 | 0.374 | 95.1% | 64.7% | z1 0.182 | 30% |

Picks at N=8: ex-tower 7 wall tiles and the bookshelf; garage-shapes 8 wall tiles; ex-blossom 6 floor tiles from 5.7 m
and one side of each cherry tree.

**The honest reading: this does not fix the problem.** Eight extra pictures move the soft share by 3 to 4 points and
the median r from 0.12-0.26 to 0.19-0.37. The worst close camera does not move at all: the picks go where the most
screen area is, not to the worst spot, and ex-tower's worst (the desk) is ground that is unpainted by rule (the floor
within 1.8 m of the globe, which moves away, is filled in).

Why, in plain numbers: a screen 0.8 m from a wall shows 883 screen pixels per metre. A 768-pixel picture at r = 1
therefore covers 0.87 x 0.65 m of wall, and at r = 0.5 it covers 1.74 x 1.31 m. The band of wall at eye height alone
(1.3 m tall) is 36 m2 in ex-tower and 57 m2 in garage-shapes: 16 and 25 pictures at r = 0.5, 64 and 100 at r = 1.
The painter makes about 6,800 pixels a second, so a whole 20 minutes of nothing but painting is 8 megapixels; the
tower's eye band at r = 1 needs 28. Close-up sharpness on walls cannot come from painted pixels at this speed. It has
to come from pixels that cost less than 65 s per 0.44 megapixel (an upscaler run on each owner picture or on each
wall's texture, or a tiling texture per surface), with a detail pass at most seeding them.
The floor is different: it needs only about 200 pixels per metre (seen from 2.6 m, at a slant), so one picture covers
10 m2 at r = 1 if the ceiling lets the camera get 2.95 m up. Outdoors that is why the picks are floor tiles.

Other targets, N=8 (near_r / soft / r<0.5): target 1: tower 0.192 / 96.0% / 81.3%, garage 0.138 / 93.2% / 79.8%,
blossom 0.390 / 94.0% / 64.2%. Target 0.25: tower 0.253 / 93.5% / 77.2%, garage 0.250 / 94.9% / 80.5%. (These two rows
were run before a last speed change to the planner's first cut; the default rows above are from the final code.)
Offering several distances at once always picks the furthest (camera 2.7 m off the wall, close to what a ring view is).

## Paying for it inside 20 minutes

Measured figures given to this job: 65 s per 768x576 picture, time in proportion to pixels; training 75 s; a build
already takes about 20 minutes. The ring is 12 level + 4 downward views in a room under 60 m2 (ex-tower: 16 x 65 =
1040 s) and 12 + 6 otherwise (garage-shapes, ex-blossom: 18 x 65 = 1170 s). Ring + training = 1115 s or 1245 s of the
1200 s, before layout, shapes, depth maps and consistent.py (not timed by me). So there is no slack: every detail
picture (65 s) must be taken from the ring. N = 8 unpaid is +520 s, a build of about 28 minutes.

| how | frees | buys | predicted near_r / soft / r<0.5 (today -> with the trade) |
|---|---|---|---|
| A. drop the downward views | 4 x 65 = 260 s (tower); 6 x 65 = 390 s | 4; 6 pictures | tower 0.163 -> 0.185 / 97.4% / 83.1%; garage 0.118 -> 0.135 / 97.1% / 80.2%; blossom 0.255 -> 0.325 / 96.2% / 70.7% |
| B. ring at 640x480 (45.1 s a view) | 16 x 19.9 = 318 s; 18 x 19.9 = 358 s | 4; 5 pictures | tower 0.163 -> 0.158 / 97.5% / 86.6%; garage 0.118 -> 0.127 / 97.6% / 82.2%; blossom 0.255 -> 0.309 / 96.6% / 79.3% |
| A + B (12 views at 640x480 = 541 s) | 499 s; 629 s | 7; 8 (of 9) pictures | tower 0.222 / 96.5% / 81.4%; garage 0.121 / 96.3% / 79.8%; blossom 0.321 / 95.9% / 77.2% |

Commands: `--no-down --budget 4` (6), `--ring-paint 640x480 --budget 4` (5), both with `--budget 7` (8).
Only A is a clear, small net gain. Shrinking the ring gives back most of what the tiles add (the centre cameras' median
r falls from 0.46 to 0.39 in ex-tower) because the tiles own only 10-30% of what a walker sees. A costs paint coverage:
in garage-shapes the unpainted share of near pixels goes 2.6% -> 4.6% even with two floor tiles picked, and the tops of
low furniture lose their views; the score does not see what that does to the look.

## What could go wrong when these are painted

- **Seams.** A tile's edge crosses the middle of a wall. consistent.py picks one owner per pixel with a hard edge, so
  any difference in tone or pattern shows as a rectangle. The plan leaves the outer 4% of each tile unowned for a
  cross-fade, and tiles overlap by a tenth, but no blending exists yet. Along a seam the paint also jumps from r = 0.5
  to r = 0.15: a sharp patch in a soft wall may look worse than an evenly soft wall.
- **Drift from the existing look.** carry_over() gives a later view the pixels earlier views show, as "keep". For a
  detail view every pixel is already shown, so keeping them all paints nothing, and keeping none starts a new wall.
  Something in between is needed (keep a border, or start from the enlarged old paint with partial noise); whether
  mflux's Z-Image ControlNet pipeline can start from an image was not checked. One seed and one prompt per tile also
  means neighbouring tiles each invent their own bricks.
- **Views the painter is not used to.** A wall filling the picture square-on from 1.4 m, and a floor from straight
  above (outdoors from 5.7 m, a drone picture), are not "interior photo" compositions; the model may add furniture,
  frames, a horizon or a vignette. The prompt ("In view: ..." or "A bare stretch of wall and floor.") says nothing
  of a close-up. Lighting painted into a top-down floor (a pool of light) will not match the ring's.
- **A flat depth map.** prep() stretches each view's inverse depth between its 1st and 99th percentile. A square-on
  wall has almost no depth range (1.37 m at the middle, 1.6 m at the corners), so the stretch turns it into a full
  black-to-white bowl and the painter is told the wall is curved. Detail views need a fixed depth scale.
- **Objects in front of walls.** A tile aimed at a wall that also catches a chair paints the chair from close and
  owns it where it is seen head-on (the plan lists this under "owns"). The painter rarely keeps a shape inside its
  outline, so the wall just round it must not be taken from that tile; score.py leaves that band with the ring, which
  makes a ring of old paint round every object inside a tile. Backed-off tiles (the bookshelf) are views of the
  object, and their wall share is what shows past its sides.
- **Training may not keep the detail.** consistent.py re-renders at 1024x768 from about 125 cameras, few of them
  within 1.5 m of a wall; a splat only holds detail some training view shows. The detail cameras themselves would be
  added as training views (they are painted views), but more close training views are probably needed. Not modelled.

## What would have to change to paint and use these views

`thread/chisel_ring.py`
- The module-level camera block (the two loops filling `B.CAMS`, then `N = len(B.CAMS)`): append the plan's cameras
  after the downward views and keep their index range (say `DETAIL`). They cannot be planned at import as the ring is:
  the plan needs the ring's depth maps, so it would be read from a file written between the two painting passes.
- `prep()`: a fixed depth scale for detail views (above); it already skips them for the start view (`int(k) < RING`).
- `view_prompt()`: words for a close view of one surface; name the surface, not "A bare stretch of wall and floor".
- `carry_over()`: today the first view to show a point wins and is kept. Detail views need the opposite: the old paint
  as a guide, not as fixed pixels, with only a border kept (and earlier detail tiles kept where they overlap).
- `gen()`: paint the detail views after the ring (they are last in `todo` already if appended), let `KEEP` carry the
  new kind of guide, and put the plan into `meta` (views.json) so a changed plan repaints only the tiles.
- `surfaces()`, `halo()`, `mend()`: unchanged.

`thread/consistent.py`
- The `count` / `rank` lines after the `src` loop: rank is by how many pixels of a surface a view shows. Detail views
  must come first for what they see head-on, whatever the count (for example rank -1 for them).
- `render()`: for a detail view, the head-on test for every surface (today only shapes are tested, against
  `C.GRAZE`; walls and boxes need their normals here), the inner margin in place of the 1-pixel `inside` test, and a
  cross-fade over that margin where it now writes `col[ok] = simg[...]` for a single winner. A tone match of each
  tile to the ring paint under it would go here too.
- The target cameras (`tgt = [src_cams[str(i)] for i in range(C.N)]` and the `add()` blocks): the detail cameras
  become training views by themselves; add close views of the tiled surfaces so the splat is trained where the
  detail is.
- `write()`: unchanged.

Also `thread/pipeline.py` (run the planner between the two painting passes; not read closely by me),
`evals/cams.py` `layout()` and `evals/density.py` `sources()` / `owner()` (so the score counts the new owners).

## Limits of the prediction, and what was not verified

- The score counts painted pixels, not whether the painter fills them with real detail. A close picture of a plain
  wall may hold no more detail than the ring's; then r doubles and nothing looks sharper.
- It assumes a detail view owns everything its rule allows, with no loss at seams, and that training keeps it.
- Chosen tiles are scored on a 512x384 blockout render (the ring's maps are 1024x768); candidates are compared on a
  96x72 one and on walker screens traced one ray per 20x20 pixels. The pick order can differ slightly from an exact one.
- The picks maximise total soft area, so the worst camera is not targeted; a pick rule for the worst spot was not tried.
- Only a wall's eye-height row is tiled; only upright sides of objects (their tops are seen by floor tiles).
- Baseline: I measured near_r 0.163 / 0.118 / 0.255 and soft share 99.7% / 98.6% / 98.2%, lower than the "about
  0.25 and about 70% soft" in the brief (70% is near the r < 0.5 share of ex-blossom, 72.9%).
- Speed: 4.1 to 5.4 s per run (its own "seconds") on nine runs when this Mac was quieter; 5.6 to 11.5 s while the load
  average was 25 to 90 from other jobs (evals/density.py alone then took 7.9 s against its usual 5.1 s). So under 10 s
  holds on a machine that is not overloaded, and not always otherwise. Output is the same on repeated runs (checked on
  the three worlds, budget 8).
- Not run: any painting, any training, any world besides these three, `--paint` sizes other than one 512x384 smoke test.
