# Detail pass, planning half: walker-driven greedy choice (detail_b)

Nothing here paints, trains or reads a picture. It plans extra views and predicts the geometry score.

    cd thread
    .venv/bin/python experiments/detail_b/plan.py ex-tower --budget 4       # the plan and its predicted score, JSON, 4 to 6 s
    .venv/bin/python experiments/detail_b/score.py ex-tower PLAN.json       # the score of a saved plan, at density.py's ray step
    .venv/bin/python experiments/detail_b/table.py                          # every number below (about 5 minutes)

## The short answer

A detail pass paid for inside the present painting time lifts the close-up score a little and cannot fix it.
Predicted, with the pictures paid for by dropping the look-down views (`--pay down`, the default):

| world | pictures | near_r median | soft share (r<1) | r<0.5 share | worst close camera |
|---|---|---|---|---|---|
| ex-tower (6x8 m, 16 views) | before | 0.163 | 0.997 | 0.896 | 0.000 desk |
| | 2 | 0.197 | 0.984 | 0.859 | 0.000 desk |
| | 4 | 0.215 | 0.967 | 0.817 | 0.000 desk |
| | 6 | 0.229 | 0.964 | 0.809 | 0.000 desk |
| | 8 | 0.238 | 0.949 | 0.804 | 0.000 desk |
| garage-shapes (12x10 m, 18 views) | before | 0.118 | 0.986 | 0.881 | 0.074 x0 |
| | 2 | 0.135 | 0.983 | 0.866 | 0.088 z0 |
| | 4 | 0.189 | 0.973 | 0.843 | 0.088 z0 |
| | 6 | 0.203 | 0.961 | 0.840 | 0.088 z0 |
| | 8 | 0.189 | 0.956 | 0.850 | 0.131 x1 |
| ex-blossom (14x12 m outdoor, 18 views) | before | 0.255 | 0.982 | 0.729 | 0.089 bench_45 |
| | 2 | 0.270 | 0.974 | 0.716 | 0.091 bench_45 |
| | 4 | 0.291 | 0.960 | 0.704 | 0.091 bench_45 |
| | 6 | 0.314 | 0.967 | 0.698 | 0.170 z1 |
| | 8 | 0.310 | 0.970 | 0.808 | 0.141 z1 |

"before" equals `evals/density.py` on the same worlds (0.163 / 0.118 / 0.255). All rows are over the close and floor
eval cameras, which the planner never looks at. Full lines, with the other two ways of paying, are in `work/table.txt`.

The best paid plans are 4 pictures on ex-tower and 6 on the two large worlds: exactly the number of look-down views
there are to swap. Past that the ring has to be painted smaller, which takes back what the extra pictures give (8 is no
better than 6, and worse on the garage).

Unpaid (`--pay none`, the build gets 65 s longer per picture) 8 pictures give 0.280 / 0.212 / 0.317: the ceiling of
this idea at this picture size is about 1.3 to 1.8 times the present r, not the 4 to 8 times that r = 1 needs.

Why: time buys pixels, and there are not enough. A 768x576 picture that is as sharp as the screen for a viewer 0.8 m
away (882 pixels per metre) covers 0.87 x 0.65 m of wall. Half as sharp, 2.3 m2. The tower's walls up to 2.5 m and its
floor are about 118 m2, so r = 0.5 everywhere from 0.8 m is about 50 pictures. Eight pictures can make a few stretches
sharper or the whole room slightly sharper; the greedy choice ends up between the two (in the tower it picks views 2
to 3 m from a wall that give 145 to 260 pixels per metre, where the ring's median is 0.163 x 882 = 144).

What the pass does do: the ring wastes most of what it paints. About 0.3 to 0.45 of the ring's pixels end up on a
surface a walker sees (the rest is overlap between views, ceiling, backdrop; a rough figure, `ring_pixels_used`),
against mostly 0.5 to 0.75 of a detail picture's (`pixels_used`; lowest 0.26).

## Approach

1. Walkers. Free spots on a grid (38 to 48 a world, 0.4 m off walls, 0.3 m off things), each looking four ways level
   and two ways 30 degrees down, on the viewer's screen (1600x900, 65 degrees), traced coarsely. 228 to 288 screens,
   none of them an eval camera.
2. Softness of a screen pixel: octaves that its paint is coarser than the screen, log2(screen px per metre / painted px
   per metre), between 0 and 3; no paint is 3. The ring's owner per pixel comes from the same rules as density.py.
   Counting "soft or not" alone (r < 1) would be blind here: nearly nothing a few pictures can do crosses r = 1.
3. Pixels are gathered into 15 cm patches of surface, each with a histogram of how finely the walkers' screens see it.
4. Candidates: from the same spots, eye height, twelve headings, pitched 10 degrees down (a wall with its floor) or 40
   degrees down (the floor ahead): 850 to 950 of them. Dropped if anything is nearer than 0.6 m, the middle of the
   picture is nearer than 1.2 m, or one wall or one thing fills over 85% of it. Same 65 degree lens and 4:3 shape as
   every other painted view, so nothing in blockout changes.
5. Greedy: take the candidate that removes the most softness, update the paint, repeat. A candidate owns a patch only
   by the rule in score.py: really seen, at least 30% head-on, 4% clear of the picture's border, not ceiling, sky,
   backdrop or a lifted thing, and at least 1.25 times the pixels per metre of the present owner.
6. Score: density.py's maths, copied into score.py, with the planned views rendered at full size and tried after the
   ring by the same visibility tests consistent.py uses.

Indoors (tower, garage) every view picked is a 10-degree one, mostly square-on to a wall from 2 to 3 m: it owns the
wall, the floor strip at its foot and whatever stands against it. Outdoors (blossom) every view picked is a 40-degree
one: the ground ahead is all there is to own near a walker.

## Paying for it inside 20 minutes

Measured figures given for this task: one picture at 768x576 is 65 s; training is about 75 s.

    today      16 views x 65 s = 1040 s (17.3 min)   rooms under 60 m2: 12 ring + 4 looking down
               18 views x 65 s = 1170 s (19.5 min)   larger rooms:      12 ring + 6 looking down
    down, N=4  12 ring x 65 + 4 detail x 65        = 1040 s   (tower: no change)
    down, N=6  12 ring x 65 + 6 detail x 65        = 1170 s   (large rooms: no change)
    down, N=6  tower: 12 ring x 45.1 (640x480) + 6 x 65 =  932 s   (only 4 to drop: the ring shrinks for the rest)
    down, N=8  tower: 12 ring x 36.6 (576x432) + 8 x 65 =  959 s;  large rooms: 12 x 45.1 (640x480) + 8 x 65 = 1062 s
    shrink     all ring views at 704x528 / 640x480 / 576x432 / 512x384 for N = 2 / 4 / 6 / 8 (54.6 / 45.1 / 36.6 / 28.9 s each)
    none       +65 s per picture: N=4 is +4.3 min, N=8 is +8.7 min

`plan.py` prints this as `pay` (`paint_seconds_before`, `paint_seconds_after`, never more than before unless `none`).
Other costs of the pass, not painting: this planner 4 to 6 s, N more depth maps in prep, N more sources in
consistent.py (not measured; small next to 65 s).

Not checked: that a smaller picture is quicker in proportion to its pixels. Only 768x576 and 1024x768 (160 s, 2.5
times the time for 1.8 times the pixels) were measured, so shrinking may save more than assumed; nor how well the
painter paints at 512x384. `down` at N up to 4 (6) needs neither assumption, which is another reason it is the default.

What `down` gives up: the look-down views are what paint the middle of the floor and the tops of low furniture. The
score sees a small price (unpainted share on the garage 0.026 -> 0.039 at N=6; the tower and blossom barely move), but
the eval cameras do not look at table tops from above, so the true price may be larger.

## What the prediction cannot tell

- It counts painted pixels, not detail. A painter that is handed blurry paint and returns it enlarged scores the same
  as one that paints wood grain. An upscaler run on the ring pictures would double r in this score for a few seconds.
- "Before" and "after" both ignore what consistent.py and the trainer then do (density.py says the same). In
  particular the splat only keeps detail that some training view sees at that size: see the last section.
- ex-tower's worst close camera stays 0.000: the desk camera looks at floor within 1.8 m of the lifted globe, which
  is filled in, not painted, whoever shows it. No painted view can change that.
- Outdoors, most of what the close cameras see is ground far off and the backdrop; detail views own 4 to 17% of those
  pixels, so blossom moves least.
- The owned areas in `owns` are exact on walls and floor and too large on shapes (a patch counts as 15 x 15 cm).
- Candidates keep clear of things by 0.3 m but do not use chisel_ring's `crowded()` rule for posts and open frames.

## What could go wrong when these are painted

- **The painter keeps the blur.** carry_over() hands earlier paint to the painter as pixels to keep (mask = 1). A
  detail view is almost wholly covered by ring paint, so as the code stands it would get that paint back, enlarged,
  and the pass would gain nothing real. The carried paint has to become a start, not a constraint: keep only the border
  band, and begin the inside from the enlarged ring paint with noise added (image-to-image). mflux's
  `ZImageTurboControlnet.generate_image` starts from pure noise (`ZImageLatentCreator.create_noise`) and takes no
  start picture, so this is a new in-memory patch like `painter_patches`. Whether the 8-step distilled model gives
  detail from a half-noised start is untested.
- **Drift.** Free the inside entirely and the painter makes a new wall: other bricks, other shelves, different light.
  The seam then shows where the detail view's ownership ends, and things half inside it (a bookshelf) change mid-way.
- **Seams.** Ownership ends 4% inside the picture's border and where the detail view stops being 1.25 times better,
  which is a ragged line across a wall. It needs a blend band (the border margin is there for it) and the colours
  matched to the ring paint (mean and contrast per surface) before blending.
- **Bare pictures.** A square-on wall has a nearly flat depth map; prep() stretches its contrast between the 1st and
  99th percentile, so tiny depth differences become strong edges the painter may follow. The 10-degree pitch and the
  85% rule keep floor or furniture in every candidate, but the prompt can still be "A bare stretch of wall and floor".
  Straight-down floor pictures are not among the candidates for this reason. The 40-degree ones the outdoor plan is
  made of show nothing but ground ("An open stretch of ground."): the ring's look-down views are like that already,
  but these are nearer, with less in them to tell the painter what the ground is.
- **Things in front of walls.** The wall behind a shape is not taken inside the band round the shape (score.py does
  this as consistent.py does), but a close view shows a large thing large: the painter may repaint it in another
  style than the ring did. The band is a fixed 16 px, so at close range it is narrower in metres.
  Lifted things are painted in these views too and never owned; the 81 px band round them is kept out.
- **Shapes from the side.** A detail view may own part of a shape while a ring view owns the rest. The plan lets it
  (30% head-on); for single objects it may be better to give a shape wholly to one view.

## What would have to change to paint and use these views

`chisel_ring.py`
- The camera list after `N = len(B.CAMS)`: append the plan's `cam` entries as `B.CAMS[str(N + k)]`, with a separate
  count (N must keep meaning "ring and look-down": consistent.py's targets, `start.json` and evals/cams.py go by it).
  The plan needs the ring's depth files, so the order is prep, plan.py, prep again for the new views.
- `prep()`: already loops over all of `B.CAMS`; only the dropped look-down views must be skipped, and the contrast
  stretch looked at for near-flat views.
- `gen()`: `todo` and the loop run over `range(N)`; the views-per-size `pw, ph` is one value for all (the ring may
  now be smaller than the detail views); `meta` must record the plan so stale pictures are cleared.
- `carry_over()` and `painter_patches()` / `KEEP`: the new mode above (keep the border, start the inside from the
  enlarged paint). "The first view wins" is the wrong rule for a detail view.
- `view_prompt()`: works as is; perhaps say "close view".

`consistent.py`
- The `src` loop (`for i in range(C.N)`) and `count` / `rank`: detail views must be loaded but kept out of the rank,
  or a close view becomes the "best view" of a whole wall by pixel count.
- `render()`: after the ring's loop, try the detail views in plan order with the rule in `score.take_over()`; it
  needs the present owner's pixels per metre (`score.per_metre()`), and a blend weight near the border instead of a
  hard switch.
- `tgt`: add training cameras at and around each detail view, and nearer. Today the grid of standing views stops
  0.2 of the room's width from a wall (1.2 m in the tower); paint finer than the training views show is averaged away. Not checked: what msplat's
  downscale schedule leaves of 1024x768 training pictures.
- `evals/density.py` `sources()` / `owner()`: the same rule, so the measured score follows (score.py is that code).
