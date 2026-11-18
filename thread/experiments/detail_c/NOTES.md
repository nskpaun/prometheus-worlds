# Detail pass, approach C: re-budget the whole painting stage

Planning only. Nothing here was painted or trained; every number is the geometry score of `evals/density.py`
(painted pixels per screen pixel, r), which counts pixels and not whether the painter fills them with real detail.

Files: `plan.py` (the planner), `score.py` (density.py's maths copied, plus detail views as owners; `score.py WORLD`
reproduces density.py's per-camera numbers exactly on the three worlds), `study.py` (the sweeps below, a few minutes),
`work/study.json`, `work/study.txt`, `work/plan_WORLD_N.json`.

## The approach

1. Candidates: each wall's lower 2.6 m cut into 1 to 6 tiles and seen square-on; the floor cut into grids and seen from
   straight above (from under the ceiling indoors, from up to 8 m outdoors); each large still object from each side
   that can be stood at. All at today's pixel count (about 442k, so about 65 s), the picture's shape matching the tile.
2. A walker stands on a grid of free spots looking eight ways (not the eval cameras, so the plan is not fitted to the
   test). Gain = rise in min(r, 1) over what the walker sees.
3. Paying: for k = 0, 1, 2 ... the k ring views that own least are left out (never view 0, which sets the look) and the
   rest are painted at the largest of 768x576 ... 512x384 that leaves room for N detail views in 1020 s. For each way
   candidates are taken greedily by gain per second. The best way wins; ways within 1% give way to fewer views left out.
4. Proposed ownership rule (what `score.py owned()` predicts with): a detail view owns a point that is inside its
   picture and off the outer 6%, seen within 60 degrees of square-on, truly visible, and not ceiling, backdrop, a moving
   thing, the floor in a moving thing's shadow, or wall/floor in the band round a shape's or moving thing's outline.
   Detail views come before all ring views; between two, the one with more pixels there.

## What the numbers say

Who owns anything today (share of what the ~125 training cameras see): every painted view owns something, but
3 of 16 (ex-tower), 4 of 18 (garage-shapes) and 6 of 18 (ex-blossom) own under 2% each. Leaving out the 4 least-owning
views costs almost nothing (near median r 0.163 -> 0.160, 0.118 -> 0.117, 0.255 -> 0.244) and frees 260 s.
The four to six downward views were painted with 74-93% of their pixels carried over (job.log).

The ring alone, other sizes (near median r / soft share): 1024x768 for every view 0.218 / 0.977 (tower), 0.157 / 0.959
(garage), 0.341 / 0.935 (blossom) but 2560-2880 s; 640x480 0.136, 0.098, 0.213; 512x384 0.109, 0.079, 0.170.
The Quick level's 8-view ring alone: 0.153, 0.121, 0.238 (about 670-780 s).

Plans (`plan.py WORLD --budget N`, 1020 s of painting). "soft" is the share of near pixels with r < 1.

| world | N | near median r | soft (r<1) | r<0.5 | unpainted | worst close camera | ring kept | seconds |
|---|---|---|---|---|---|---|---|---|
| ex-tower | before | 0.163 | 0.9973 | 0.897 | 0.212 | desk 0.0 | 16 at 768x576 | 1040 |
| | 2 | 0.196 | 0.9922 | 0.876 | 0.218 | desk 0.0 | 13 at 768x576 | 978 |
| | 4 | 0.198 | 0.9607 | 0.849 | 0.215 | desk 0.0 | 11 at 768x576 | 978 |
| | 6 | 0.213 | 0.9446 | 0.809 | 0.221 | desk 0.0 | 9 at 768x576 | 977 |
| | 8 | 0.234 | 0.9349 | 0.788 | 0.217 | desk 0.0 | 9 at 704x528 | 1018 |
| garage-shapes | before | 0.118 | 0.9856 | 0.881 | 0.026 | x0 0.074 | 18 at 768x576 | 1170 |
| | 2 | 0.163 | 0.9789 | 0.864 | 0.029 | x1 0.074 | 13 at 768x576 | 978 |
| | 4 | 0.207 | 0.9696 | 0.845 | 0.052 | car-corner-fr_45 0.098 | 11 at 768x576 | 981 |
| | 6 | 0.207 | 0.9559 | 0.817 | 0.052 | car-corner-fr_45 0.176 | 9 at 768x576 | 983 |
| | 8 | 0.207 | 0.9510 | 0.815 | 0.040 | car-corner-fr 0.164 | 9 at 704x528 | 1015 |
| ex-blossom | before | 0.255 | 0.9824 | 0.729 | 0.079 | bench_45 0.089 | 18 at 768x576 | 1170 |
| | 2 | 0.265 | 0.9726 | 0.723 | 0.097 | bench_45 0.103 | 13 at 768x576 | 971 |
| | 4 | 0.314 | 0.9637 | 0.706 | 0.110 | z1 0.149 | 11 at 768x576 | 967 |
| | 6 | 0.326 | 0.9552 | 0.710 | 0.109 | z1 0.136 | 11 at 704x528 | 986 |
| | 8 | 0.327 | 0.9506 | 0.766 | 0.109 | z1 0.124 | 11 at 640x480 | 1012 |

Read plainly:
- The detail pass lifts near median r by 1.3 to 1.75 times and takes 3 to 6 points off the soft share. It does not make
  the world sharp: 93-95% of near pixels still have under one painted pixel per screen pixel.
- It cannot, at this painting speed. 1020 s paints 6.9 million pixels. Spread once, square-on, over the floor and the
  walls' lower 2.6 m that is 240, 172 and 203 pixels per metre, r = 0.27, 0.20 and 0.23 from 0.8 m (the screen shows
  883 pixels per metre there). r = 1 on those surfaces needs 13 to 26 times the pixels. The plans above already sit
  near that ceiling on the close cameras (garage close median 0.202 against 0.195).
- Best split: 6 detail views with the 7 to 9 least-owning ring views left out is the knee. From 6 to 8 the ring has to
  shrink as well, and what detail views do not cover gets worse (blossom r<0.5 0.710 -> 0.766, centre 0.442 -> 0.404).
  Beyond 8 the outdoor world loses (N=12: a close camera with no paint at all).
- Leaving views out beats shrinking all of them (N=4: tower 0.198 against 0.180, garage 0.207 against 0.154, blossom
  0.314 against 0.287). Shrinking alone cannot pay for more than 8 detail views (16-view world) or 6 (18-view worlds) at all.
- The 8-view ring plus 5 detail views lands in the same place (tower 0.239, garage 0.207, blossom 0.326): no better.
- A larger picture for the floor never wins: at 160 s, 1024x768 paints 4.9k pixels a second against 6.8k at 768x576,
  and the planner offered it (`--big-floor`) chose it in no case. If 1024x768 really takes nearer 110 s the two are
  level; that was not tried.
- Indoors the planner spends on walls first (the close cameras look at walls from 0.8 m; the floor is seen from 2 m
  or more and is already the less soft surface); outdoors only on the ground.
- ex-tower's worst close camera stays at 0.0: the desk stands in the band round the globe, which moves away, so its
  paint is refused whoever painted it. Only a view painted of the room without the moving things would fix that. Not planned here.

## Time

Painting today, at the 65 s a view the task gives: 16 views 1040 s, 18 views 1170 s (job.log sums: tower 1031 s,
garage 947 s, blossom 1091 s). The plan's painting is ring kept x 65 s x (size / 768x576 in pixels) + detail views at
about 63-67 s each, held to 1020 s: e.g. tower N=6: 9 x 65 + 392 = 977 s; blossom N=6: 11 x 54.6 + 385 = 986 s.
So a build's painting gets no longer (18-view worlds get 2.5 minutes shorter). The rest of the 20 minutes was not
measured here; ex-tower's log shows 346 s of shapes before painting, so that build is over 20 minutes with or without this.
Assumed, not measured: time below 768x576 falls with pixel count; a picture of another shape with the same pixel count
costs the same; consistent.py takes no longer with about the same number of source views.

## What could go wrong when these are painted

- The hand-over of the look. `carry_over` gives a view the earlier paint as pixels to keep. For a detail view that paint
  is ring paint enlarged 2 to 4 times: kept as it is, the view reproduces the blur and the pixel count above means
  nothing. It has to be given as a guide to repaint, not as pixels to keep (mask 0 with the picture still supplied, or a
  start from the enlarged paint at partial strength). Whether this ControlNet then holds the look is the first thing to test.
- Drift. If the guide is weak the wall becomes another wall. Seen from the room's middle it is the ring's paint in
  the training views that look across, and the detail view's in those that stand close.
- Seams. The rule switches owner at a hard line 6% inside the detail picture, and at 2.6 m up a taller wall (ex-tower's
  walls are 4 m). Colour or brightness steps will show unless the detail picture is matched to the ring paint
  (mean and spread per surface) and blended across the rim; `render()` takes one source per pixel today.
- Flat depth. A wall seen square-on or a floor from above has almost no depth range. `prep()` stretches each map between
  its own 1st and 99th percentile, so the faint fall-off towards the corners becomes a full black-to-white bowl and the
  painter is told to paint a dome. These views need the ring's range, not their own.
- Unnatural views. The model has only been run on room views at 65 degrees. A square-on wall with no floor line and an
  indoor floor from 2.3 m straight down are unusual photos; some plans use 72 or 80 degrees because the room is small.
  The prompt still says "interior photo ... In view: ...". Pictures of 928x464 or 576x768 were never tried either.
- Objects in front of walls. A wall tile that sees a shape owns its near face too (bookshelf, fireplace: 3-5% of the
  picture), and the wall in the band round its outline goes back to the ring: a second seam, hugging the object.
- Fewer ring views. Leaving out up to 9 changes what is carried over between the rest, and raises the unpainted
  share by 0.5 to 3 points. The score says nothing about whether the room still looks like one room.
- The score's own limits: r counts pixels; the 6.9 million are assumed to be new detail. Detail views are not added as
  training cameras, and whether 75 s of training keeps finer paint was not tested.

## What would have to change

`chisel_ring.py`
- the camera table (the two loops filling `B.CAMS`, and `N`): accept views to leave out and append the detail views,
  each with its own size and angle (blockout has one `W, H, FOV` for all; `score.py shot()` shows the swap).
- `prep()`: trace each detail view at its own size and angle; fixed depth range for them; keep them out of `start.json`.
- `surfaces()`, `halo()`, `ground_words()`, `view_prompt()`: per-view size (the bands are in pixels of a 1024-wide
  picture); wording for a close wall and for a floor from above.
- `carry_over()` and `mend()`: project into each earlier view with that view's size; return the paint as a guide.
- `painter_patches()` (`encode_controls`, `KEEP`): a way to pass the guide without marking it kept.
- `gen()`: size per view instead of one `WORLD_PAINT`, `remember()` at each view's size, the plan in `views.json` so
  stale pictures are found, detail views last.

`consistent.py`
- the `src` loop: trace and keep each picture at its own size (today every picture is resized to 1024x768).
- `count` / `rank`: detail views first where the rule holds.
- `tgt`: starts from the painted cameras; leave detail cameras out or add them at the common size.
- `render()`: `B.project` per view size, the rim and square-on tests, band widths per size, blending at the rim.

`pipeline.py`: run the plan after layout and shapes, pass it to both. `evals/density.py owner()`: the same rule, or it
will score a real build as if the detail views were not there.
