# Detail pass: the plan picked from detail_a, detail_b and detail_c

Nothing here was painted or trained. Every number is the geometry score (painted pixels per screen pixel, r),
worked out by one scorer for all plans (`judge.py`). Whether the painter fills finer pixels with real detail is not
known; the first GPU experiment at the end settles that.

    cd thread
    .venv/bin/python experiments/detail_pick/pick.py ex-tower --budget 6 > plan.json      # the chosen planner, about 5 s
    .venv/bin/python experiments/detail_pick/judge.py ex-tower H6=plan.json                # the common scorer, 6 s + 2 s a plan
    .venv/bin/python experiments/detail_pick/table.py --md experiments/detail_pick/work/judge_*.json

## Verdict

- **Chosen: a merge.** detail_b's views (ordinary photos from standing spots), paid for by leaving out the painted
  views a walker misses least (detail_c's idea, done one view at a time), plus one kind of view taken from
  detail_a/detail_c's floor tiles in a paintable form: from 3 m up looking 60 degrees down. `pick.py` is this planner.
- **One out, one in.** Each detail picture replaces one of today's 16 or 18 pictures. Painting time does not change,
  every picture stays 768x576 at 65 degrees, and no picture is painted at a size that was never timed.
- **Default 6 pictures.** From 8 to 12 the gain per picture falls by a fifth to a third and unpainted area rises.
- **What it buys, by the score.** With 6 pictures the near median r goes 0.218 to 0.273 (ex-tower), 0.122 to 0.192
  (garage-shapes), 0.239 to 0.334 (ex-blossom). Over a dense walk the mean of min(r, 1) goes 0.321 to 0.433, 0.356
  to 0.449, 0.381 to 0.465.
- **What it does not buy.** 93 to 96% of near pixels still have under one painted pixel per screen pixel. All three
  designers found the same wall independently and I agree: the painter lays about 6,800 pixels a second, and a wall
  0.8 m away needs about 880 painted pixels per metre. A detail pass makes close surfaces about 1.3 to 1.6 times
  finer. It cannot make them sharp.

## What was checked, and what did not hold up

Each `plan.py` was run on the three worlds at budgets 4 and 8 (outputs in `work/plans/`). All ran in 3.7 to 5.5 s
wall time and exited cleanly.

**The reports reproduce, with two caveats.**
- Rerun, the planners print the same numbers as their reports for ex-tower and garage-shapes. For ex-blossom they
  now print a baseline of 0.223, not 0.255: `evals/cams.py` changed after they ran (outdoor wall views are now kind
  "far" and no longer count as near).
- `evals/density.py` also changed: medians are now over painted pixels only. Today's baselines are 0.218 / 0.122 /
  0.239, not 0.163 / 0.118 / 0.255. `judge.py` matches today's `density.py` exactly on all three worlds (near
  median, r<1, r<0.5, worst, unpainted), and also gives the old count as "r0" so claims can be compared.

**Claims against the common scorer** (near median r, no paint counted as 0):
- detail_a and detail_c: within 0.004 of their own predictions on every plan.
- detail_b: over-predicts. garage-shapes N=4 claimed 0.189, judged 0.163; N=8 claimed 0.189, judged 0.151;
  ex-blossom N=8 claimed 0.302, judged 0.235. Its rule lets a view own surfaces seen up to 72 degrees off head-on;
  the common rule stops at 60.
- No plan is fitted to the 24 eval cameras: each ranks about the same on the 424 to 472 walk cameras as on the eval
  cameras.

**Cameras.** Every planned camera is inside the room and clear of objects except one: detail_a's eighth ex-blossom
view sits 5.7 m up against the left cherry tree's box. detail_a's "owns" lists also name surfaces its own scoring
would not grant (it lists what is in frame, without its head-on test).

**Time arithmetic.**
- detail_a's default output is unpaid: 4 pictures add 260 s, 8 add 520 s. Its paid form at 8 (drop the look-down
  views, ring at 640x480) is 1062 s on ex-tower against 1040 s today, and judged it gains almost nothing on the
  walk: 0.321 to 0.366 on ex-tower, and garage-shapes gets worse (0.356 to 0.347).
- detail_b is honest up to 4 (6) pictures. Past that it shrinks the ring to 640x480 or 576x432, which was never
  timed or looked at, and judged it loses on ex-blossom (0.381 to 0.368).
- detail_c is honest about seconds but cuts painting to 1020 s, 150 s less than garage-shapes and ex-blossom get
  today, and uses sizes never timed (704x528, 640x480, and 928x464, 672x656, 576x768 for tiles).
- The brief's 65 s a picture is the upper end. The build logs give 64.4 s (ex-tower), 52.6 s (garage-shapes) and
  60.7 s (ex-blossom) a view.
- The 20 minute limit is already broken on two of the three worlds, by file times: ex-tower took 28.5 minutes from
  layout to the last splat and ex-blossom 26.3 (shapes for up to six objects, and a second set of views and a
  second training for the moving things). garage-shapes took 18.9. None of the plans changes that.

**Paintability** (flags from `judge.py`'s check of each view, in `work/judge_WORLD.json`):
- detail_a: every wall tile is one wall filling 93 to 100% of the picture from 1.37 m, with a depth map whose near
  and far differ by only 1.24 times. `prep()` stretches that to full contrast: a bowl. Outdoors its floor views
  look straight down from 5.7 m.
- detail_c: the same flat wall tiles from further back (5 of 8 flagged flat on ex-tower), straight-down floor
  views from 2.6 m indoors and 3.6 to 6.2 m outdoors, and every picture a different shape and some a different
  angle (72, 80 degrees). `consistent.py` and `blockout.py` assume one size and one angle for all pictures.
- detail_b: no flags indoors. Its views are a wall with a strip of floor, or two walls and a corner, from 2 to
  2.6 m. Outdoors every view is bare ground looking 40 degrees down, which the common rule lets it own only 54% of.

**Seams.** "steps" is the share of neighbouring screen pixels on one surface whose r jumps by more than 1.5 times.
Every plan raises it, by 1.4 to 3.2 times (walk set: 0.0096 / 0.0058 / 0.0115 today; 0.010 to 0.026 with 8 pictures).
Tiles have fewer steps than oblique photos (detail_c 0.016 / 0.012 / 0.018; detail_b 0.026 / 0.019 / 0.016).
They fall at each detail picture's rim, where a wall view stops owning the floor in front of it, round shapes'
outlines, and where two detail views meet.

**Outdoors.** All three only ever help the ground: the backdrop is 20 m away. detail_a and detail_c do it best by
the score but with drone pictures; detail_b does it paintably but weakly (+0.007 a picture against +0.017).

## The chosen design

1. **Plan before painting.** The plan needs only the blockout, so it runs inside `prep()`, right after the ring's
   depth maps. The painting environment never has to ray-cast.
2. **Which pictures go.** A coarse walker (30 spots, four looks each) is traced once. The painted view whose loss
   lowers the walker's mean min(r, 1) least is left out, the sum is counted again, and so on. View 0 is never left
   out. The first few cost nothing: ex-tower's first four lose 0.0000 to 0.0013 each, garage-shapes' first six
   -0.0004 to 0.0000, two of ex-blossom's first three are gains (-0.0025, -0.0035), because a view that ranks
   first for a surface is not always the finest one that sees it.
3. **Which pictures come.** Candidates from about 40 free standing spots:
   - eye height, 12 headings, 10 degrees down: a wall with its floor;
   - eye height, 40 degrees down: the floor ahead;
   - 3 m up (indoors no nearer the ceiling than 0.4 m), 60 degrees down: the ground nearly square-on. This is like
     the look-down views the ring already paints, a little steeper.
   A candidate is dropped if anything is nearer than 0.6 m or one wall or thing fills over 85% of it. The pick is
   greedy: most softness removed from what a walker sees, given the ring that is left.
4. **What a detail view owns** (the rule `judge.py` holds every plan to):
   - the point is in its picture, not in the outer 5%;
   - it really sees it, within 60 degrees of head-on;
   - not ceiling, sky, backdrop, a thing that moves away, the floor in such a thing's shadow, nor wall or floor in
     the band round a shape's outline;
   - and it lays more pixels there than the owner so far. It never makes anything coarser.
5. **Keeping it the same room.** A detail picture keeps only the fine part of what it paints. Its colours at scales
   above about 40 pixels are replaced by the ring paint reprojected into the same view, and its rim is cross-faded
   into the ring paint. So a drift in tone or light cannot show as a seam; a drift in pattern still can.

What the planner picks at 6:
- ex-tower: six wall-with-floor views from 2 to 3 m, each owning 6 to 15 m2 of wall at 170 to 215 pixels a metre
  (the ring gives about 95 there), one of them the bookshelf.
- garage-shapes: four wall views at 140 to 170 pixels a metre and two raised floor views of 11 m2 each.
- ex-blossom: six raised ground views of 14 to 15 m2 each at 142 pixels a metre, 3.7 to 8 times the ring's.

## The common-scorer table

A = detail_a (paid: no look-down views, and at 8 a 640x480 ring), B = detail_b (paid its default way),
C = detail_c, "free" = unpaid, H = the chosen plan (`pick.py`). Columns from "claimed" to "sharp" are over the 24
eval cameras' close and floor views; "claimed" is the planner's own near median and "r0" the judge's by the same
count (no paint = 0). "walk sharp" is the mean of min(r, 1) over the walk set, no paint counted as 0, and "per
picture" its rise per detail picture. "flagged" counts views with a paintability or camera flag.

### ex-tower: 6 x 8 m, indoor, 16 painted views, 472 walk cameras

| plan | N | ring kept | paint s | claimed | r0 | near med r | r<1 | r<0.5 | unpainted | worst close | sharp | walk sharp | per picture | walk <2.5 m sharp | walk <2.5 m med r | walk unpainted | steps | flagged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 16 at 768x576 | 1040 |  | 0.163 | 0.218 | 0.997 | 0.869 | 0.212 | 0.000 | 0.222 | 0.321 |  | 0.191 | 0.188 | 0.161 | 0.0096 |  |
| A4 | 4 | 12 at 768x576 | 1040 | 0.185 | 0.184 | 0.266 | 0.969 | 0.786 | 0.219 | 0.000 | 0.259 | 0.362 | +0.0103 | 0.266 | 0.224 | 0.172 | 0.0135 | 4/4 |
| A8 | 8 | 12 at 640x480 | 1062 | 0.240 | 0.238 | 0.311 | 0.958 | 0.733 | 0.207 | 0.000 | 0.284 | 0.366 | +0.0057 | 0.316 | 0.237 | 0.168 | 0.0197 | 5/8 |
| A4free | 4 | 16 at 768x576 | 1300 |  | 0.188 | 0.268 | 0.968 | 0.783 | 0.212 | 0.000 | 0.263 | 0.368 | +0.0118 | 0.268 | 0.224 | 0.161 | 0.0133 | 4/4 |
| A8free | 8 | 16 at 768x576 | 1560 |  | 0.260 | 0.329 | 0.954 | 0.707 | 0.201 | 0.000 | 0.303 | 0.411 | +0.0114 | 0.335 | 0.272 | 0.158 | 0.0169 | 5/8 |
| B4 | 4 | 12 at 768x576 | 1040 | 0.215 | 0.208 | 0.258 | 0.966 | 0.785 | 0.216 | 0.000 | 0.268 | 0.390 | +0.0173 | 0.256 | 0.248 | 0.167 | 0.0165 | 0/4 |
| B8 | 8 | 12 at 576x432 | 959 | 0.238 | 0.229 | 0.265 | 0.945 | 0.793 | 0.198 | 0.000 | 0.291 | 0.408 | +0.0109 | 0.289 | 0.279 | 0.155 | 0.0264 | 0/8 |
| B4free | 4 | 16 at 768x576 | 1300 |  | 0.212 | 0.259 | 0.966 | 0.782 | 0.210 | 0.000 | 0.272 | 0.394 | +0.0183 | 0.258 | 0.248 | 0.159 | 0.0166 | 0/4 |
| B8free | 8 | 16 at 768x576 | 1560 |  | 0.270 | 0.338 | 0.936 | 0.712 | 0.207 | 0.000 | 0.328 | 0.455 | +0.0168 | 0.323 | 0.310 | 0.156 | 0.0234 | 0/8 |
| C4 | 4 | 11 at 768x576 | 978 | 0.198 | 0.198 | 0.239 | 0.949 | 0.806 | 0.215 | 0.000 | 0.264 | 0.395 | +0.0185 | 0.263 | 0.247 | 0.172 | 0.0112 | 4/4 |
| C8 | 8 | 9 at 704x528 | 1017 | 0.234 | 0.238 | 0.305 | 0.916 | 0.728 | 0.216 | 0.000 | 0.316 | 0.444 | +0.0155 | 0.311 | 0.306 | 0.177 | 0.0160 | 8/8 |
| H2 | 2 | 14 at 768x576 | 1040 | 0.199 | 0.199 | 0.233 | 0.981 | 0.841 | 0.212 | 0.000 | 0.246 | 0.364 | +0.0217 | 0.232 | 0.219 | 0.162 | 0.0134 | 0/2 |
| H4 | 4 | 12 at 768x576 | 1040 | 0.222 | 0.222 | 0.256 | 0.962 | 0.802 | 0.200 | 0.000 | 0.273 | 0.399 | +0.0197 | 0.267 | 0.252 | 0.160 | 0.0170 | 0/4 |
| H6 | 6 | 10 at 768x576 | 1040 | 0.225 | 0.225 | 0.273 | 0.932 | 0.750 | 0.198 | 0.000 | 0.301 | 0.433 | +0.0187 | 0.298 | 0.289 | 0.162 | 0.0182 | 0/6 |
| H8 | 8 | 8 at 768x576 | 1040 | 0.252 | 0.253 | 0.316 | 0.907 | 0.711 | 0.201 | 0.000 | 0.333 | 0.456 | +0.0170 | 0.328 | 0.316 | 0.169 | 0.0206 | 0/8 |
| H12 | 12 | 4 at 768x576 | 1040 | 0.244 | 0.250 | 0.349 | 0.901 | 0.685 | 0.229 | 0.000 | 0.339 | 0.469 | +0.0124 | 0.348 | 0.351 | 0.200 | 0.0196 | 3/12 |

### garage-shapes: 12 x 10 m, indoor, 18 painted views, 432 walk cameras

| plan | N | ring kept | paint s | claimed | r0 | near med r | r<1 | r<0.5 | unpainted | worst close | sharp | walk sharp | per picture | walk <2.5 m sharp | walk <2.5 m med r | walk unpainted | steps | flagged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 18 at 768x576 | 1170 |  | 0.118 | 0.122 | 0.985 | 0.878 | 0.026 | 0.074 | 0.212 | 0.356 |  | 0.186 | 0.146 | 0.024 | 0.0058 |  |
| A4 | 4 | 12 at 768x576 | 1040 | 0.129 | 0.127 | 0.137 | 0.976 | 0.820 | 0.049 | 0.074 | 0.233 | 0.361 | +0.0013 | 0.223 | 0.156 | 0.051 | 0.0080 | 4/4 |
| A8 | 8 | 12 at 640x480 | 1062 | 0.121 | 0.118 | 0.127 | 0.963 | 0.795 | 0.043 | 0.064 | 0.241 | 0.347 | -0.0011 | 0.247 | 0.144 | 0.043 | 0.0102 | 8/8 |
| A4free | 4 | 18 at 768x576 | 1430 |  | 0.144 | 0.150 | 0.975 | 0.805 | 0.026 | 0.074 | 0.249 | 0.386 | +0.0073 | 0.246 | 0.166 | 0.024 | 0.0082 | 4/4 |
| A8free | 8 | 18 at 768x576 | 1690 |  | 0.186 | 0.196 | 0.951 | 0.721 | 0.026 | 0.076 | 0.300 | 0.413 | +0.0071 | 0.299 | 0.188 | 0.024 | 0.0104 | 8/8 |
| B4 | 4 | 14 at 768x576 | 1170 | 0.189 | 0.163 | 0.173 | 0.977 | 0.854 | 0.038 | 0.088 | 0.247 | 0.403 | +0.0116 | 0.216 | 0.186 | 0.038 | 0.0127 | 0/4 |
| B8 | 8 | 12 at 640x480 | 1062 | 0.189 | 0.151 | 0.166 | 0.966 | 0.868 | 0.049 | 0.114 | 0.247 | 0.408 | +0.0065 | 0.222 | 0.213 | 0.048 | 0.0187 | 0/8 |
| B4free | 4 | 18 at 768x576 | 1430 |  | 0.155 | 0.161 | 0.970 | 0.846 | 0.022 | 0.088 | 0.261 | 0.414 | +0.0145 | 0.221 | 0.183 | 0.022 | 0.0121 | 0/4 |
| B8free | 8 | 18 at 768x576 | 1690 |  | 0.198 | 0.199 | 0.953 | 0.821 | 0.021 | 0.126 | 0.303 | 0.466 | +0.0137 | 0.267 | 0.238 | 0.020 | 0.0162 | 0/8 |
| C4 | 4 | 11 at 768x576 | 981 | 0.207 | 0.207 | 0.207 | 0.967 | 0.836 | 0.052 | 0.098 | 0.271 | 0.425 | +0.0171 | 0.254 | 0.219 | 0.049 | 0.0096 | 4/4 |
| C8 | 8 | 9 at 704x528 | 1015 | 0.207 | 0.207 | 0.207 | 0.948 | 0.806 | 0.039 | 0.165 | 0.305 | 0.470 | +0.0142 | 0.286 | 0.262 | 0.031 | 0.0122 | 8/8 |
| H2 | 2 | 16 at 768x576 | 1170 | 0.158 | 0.157 | 0.160 | 0.979 | 0.870 | 0.026 | 0.088 | 0.236 | 0.392 | +0.0180 | 0.217 | 0.177 | 0.024 | 0.0087 | 0/2 |
| H4 | 4 | 14 at 768x576 | 1170 | 0.178 | 0.177 | 0.180 | 0.971 | 0.851 | 0.022 | 0.088 | 0.262 | 0.423 | +0.0167 | 0.243 | 0.202 | 0.018 | 0.0112 | 1/4 |
| H6 | 6 | 12 at 768x576 | 1170 | 0.190 | 0.190 | 0.192 | 0.963 | 0.815 | 0.017 | 0.093 | 0.290 | 0.449 | +0.0155 | 0.265 | 0.222 | 0.014 | 0.0139 | 2/6 |
| H8 | 8 | 10 at 768x576 | 1170 | 0.201 | 0.202 | 0.203 | 0.950 | 0.793 | 0.024 | 0.148 | 0.314 | 0.472 | +0.0144 | 0.287 | 0.257 | 0.030 | 0.0176 | 2/8 |
| H12 | 12 | 6 at 768x576 | 1170 | 0.214 | 0.215 | 0.221 | 0.941 | 0.762 | 0.033 | 0.148 | 0.335 | 0.494 | +0.0115 | 0.311 | 0.291 | 0.067 | 0.0176 | 5/12 |

### ex-blossom: 14 x 12 m, outdoor, 18 painted views, 424 walk cameras

| plan | N | ring kept | paint s | claimed | r0 | near med r | r<1 | r<0.5 | unpainted | worst close | sharp | walk sharp | per picture | walk <2.5 m sharp | walk <2.5 m med r | walk unpainted | steps | flagged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 18 at 768x576 | 1170 |  | 0.223 | 0.239 | 0.970 | 0.756 | 0.053 | 0.089 | 0.319 | 0.381 |  | 0.128 | 0.115 | 0.124 | 0.0115 |  |
| A4 | 4 | 12 at 768x576 | 1040 | 0.314 | 0.311 | 0.325 | 0.944 | 0.734 | 0.054 | 0.234 | 0.376 | 0.449 | +0.0170 | 0.194 | 0.247 | 0.126 | 0.0153 | 4/4 |
| A8 | 8 | 12 at 640x480 | 1062 | 0.328 | 0.325 | 0.340 | 0.933 | 0.718 | 0.050 | 0.205 | 0.387 | 0.418 | +0.0046 | 0.222 | 0.261 | 0.120 | 0.0161 | 6/8 |
| A4free | 4 | 18 at 768x576 | 1430 |  | 0.328 | 0.343 | 0.935 | 0.698 | 0.052 | 0.239 | 0.400 | 0.453 | +0.0179 | 0.196 | 0.247 | 0.123 | 0.0160 | 4/4 |
| A8free | 8 | 18 at 768x576 | 1690 |  | 0.366 | 0.384 | 0.921 | 0.642 | 0.050 | 0.239 | 0.431 | 0.475 | +0.0117 | 0.238 | 0.266 | 0.119 | 0.0189 | 6/8 |
| B4 | 4 | 14 at 768x576 | 1170 | 0.270 | 0.268 | 0.289 | 0.942 | 0.713 | 0.055 | 0.080 | 0.349 | 0.408 | +0.0067 | 0.175 | 0.133 | 0.126 | 0.0152 | 4/4 |
| B8 | 8 | 12 at 640x480 | 1062 | 0.302 | 0.235 | 0.254 | 0.968 | 0.810 | 0.056 | 0.100 | 0.301 | 0.368 | -0.0017 | 0.178 | 0.134 | 0.126 | 0.0162 | 8/8 |
| B4free | 4 | 18 at 768x576 | 1430 |  | 0.245 | 0.265 | 0.949 | 0.724 | 0.053 | 0.089 | 0.344 | 0.409 | +0.0069 | 0.172 | 0.133 | 0.124 | 0.0149 | 4/4 |
| B8free | 8 | 18 at 768x576 | 1690 |  | 0.246 | 0.266 | 0.942 | 0.721 | 0.053 | 0.089 | 0.347 | 0.433 | +0.0065 | 0.205 | 0.160 | 0.124 | 0.0183 | 8/8 |
| C4 | 4 | 11 at 768x576 | 967 | 0.323 | 0.325 | 0.344 | 0.935 | 0.692 | 0.061 | 0.243 | 0.396 | 0.427 | +0.0115 | 0.200 | 0.240 | 0.166 | 0.0158 | 4/4 |
| C8 | 8 | 11 at 640x480 | 1012 | 0.355 | 0.356 | 0.374 | 0.914 | 0.682 | 0.061 | 0.203 | 0.407 | 0.409 | +0.0035 | 0.229 | 0.299 | 0.166 | 0.0182 | 8/8 |
| H2 | 2 | 16 at 768x576 | 1170 | 0.226 | 0.225 | 0.242 | 0.961 | 0.747 | 0.054 | 0.089 | 0.326 | 0.417 | +0.0178 | 0.171 | 0.155 | 0.124 | 0.0137 | 2/2 |
| H4 | 4 | 14 at 768x576 | 1170 | 0.314 | 0.315 | 0.331 | 0.945 | 0.731 | 0.055 | 0.250 | 0.371 | 0.445 | +0.0159 | 0.206 | 0.210 | 0.125 | 0.0167 | 4/4 |
| H6 | 6 | 12 at 768x576 | 1170 | 0.316 | 0.318 | 0.334 | 0.936 | 0.721 | 0.055 | 0.248 | 0.377 | 0.465 | +0.0139 | 0.221 | 0.245 | 0.129 | 0.0183 | 6/6 |
| H8 | 8 | 10 at 768x576 | 1170 | 0.333 | 0.334 | 0.354 | 0.934 | 0.681 | 0.060 | 0.231 | 0.397 | 0.477 | +0.0120 | 0.226 | 0.255 | 0.140 | 0.0194 | 8/8 |
| H12 | 12 | 6 at 768x576 | 1170 | 0.377 | 0.382 | 0.424 | 0.896 | 0.600 | 0.112 | 0.171 | 0.421 | 0.481 | +0.0084 | 0.249 | 0.328 | 0.181 | 0.0217 | 12/12 |

Other things the judge measured:
- **Training cameras are not the limit yet.** min(r, training-view r) equals r to within 0.01 on every plan: the
  1024x768 training views already see these surfaces finer than any plan paints them.
- **The centre views gain too** (median r, 6 pictures): 0.488 to 0.582, 0.364 to 0.373, 0.450 to 0.512.
- **ex-tower's worst close camera stays at 0.** The desk camera looks at floor inside the band round the globe,
  which is filled in whoever paints it. No plan can change that.

## Expected gain per picture

Rise in the walk set's mean min(r, 1) per detail picture, chosen plan, painting time unchanged:

| pictures | ex-tower | garage-shapes | ex-blossom |
|---|---|---|---|
| 2 | +0.022 | +0.018 | +0.018 |
| 4 | +0.020 | +0.017 | +0.016 |
| 6 | +0.019 | +0.016 | +0.014 |
| 8 | +0.017 | +0.014 | +0.012 |
| 12 | +0.012 | +0.012 | +0.008 |

Near median r on the eval cameras rises about 0.009 (ex-tower), 0.012 (garage-shapes) and 0.016 (ex-blossom) a
picture up to 6. At 12 only 4 to 6 ring views are left and unpainted area rises (walk set: 0.161 to 0.200,
0.024 to 0.067, 0.124 to 0.181), so 12 is past the point of use.

## Time budget

| step | change | seconds |
|---|---|---|
| planning in `prep()` | new | about 5 (measured 4.7 to 5.2 for `pick.py`, which also scores itself) |
| depth maps | as many as today | 0 |
| painting | 6 out, 6 in, all 768x576 | 0 by the brief's 65 s a picture; not measured |
| making the views agree | 6 more target views and a second pass over each target for the detail owners | not measured; my guess is 20 to 60 on a stage that takes 57 to 195 today |
| training | 6 more of about 125 views | not measured, expected small |

If the agree step's extra time matters, leave out seven views and add six: that frees 65 s for a walker loss of
0.0033 or less.

## Changes needed, function by function

Line numbers move; `consistent.py` was being edited by another job while this was written.

### `thread/chisel_ring.py`
- **Camera block (after `N = len(B.CAMS)`).** Read `depth/detail.json` if there is one: `{"drop": [...], "views":
  [[eye, look], ...]}`. Append the views as `B.CAMS[str(N + k)]`. Keep `N` meaning "ring plus look-down", so no
  existing view is renumbered, and add `SKIP = set(drop)` and `ND = len(views)`. `evals/cams.py layout()` reloads
  this module, so it then sees the same cameras.
- **`prep()`.** After the ring's maps are written, when `WORLD_DETAIL` (a count, default 0) is set: run the planner
  (pick.py's `order()` and view choice, moved into `thread/`), write `depth/detail.json`, add the cameras and render
  their maps in the same loop. Two changes inside the loop:
  - the start view must be a painted one: `int(k) < RING and int(k) not in SKIP`;
  - for detail views, `lo = min(lo, hi / 2.5)` before the stretch, so a near-flat wall stays near-flat. Their
    near-to-far ratio is 1.3 to 2.3 indoors (1.31 for ex-tower's first), 1.88 for raised views.
- **`view_prompt()`.** Works as it is. A raised outdoor view gets "An open stretch of ground." and no "Beyond".
- **`carry_over(i, done)`.** Today every pixel of a detail view would come back as "keep" and nothing would be
  painted. For `i >= N` return three things: the reprojected picture (as now), a keep mask that is only the outer
  12% rim of it, and the whole reprojected picture as the start picture. Save the start picture as
  `images/NN.start.png`; `consistent.py` needs it for the tone match.
- **`painter_patches()`.** Add a module dict `START` beside `KEEP`. Two more in-memory patches, read from mflux's
  source and not run:
  - `ZImageLatentCreator.create_noise`: when `START` is set, return
    `LatentCreator.add_noise_by_interpolation(clean=pack_latents(vae.encode(start)), noise=noise, sigma=sigmas[k])`,
    as `LatentCreator.create_for_txt2img_or_img2img` does for the plain model;
  - `Config.init_time_step`: return `k` when `START` is set, so the loop starts at step k of 8.
  `ZImageTurboControlnet.generate_image` itself starts from pure noise and takes no picture. Only needed if the
  GPU experiment says the rim alone does not hold the look.
- **`gen()`.** `todo` skips `SKIP` and runs on through `N .. N + ND - 1`, after the ring. The meta gets
  `"detail": {"drop": [...], "views": [...]}` and `"painted": N - len(SKIP) + ND`, so pictures painted for another
  plan count as stale. Detail views are remembered in `done` like any other.
- **`surfaces()`, `halo()`, `mend()`, `stand()`, `free()`.** Unchanged.

### `thread/consistent.py`
- **The `src` loop.** Run over `range(C.N + C.ND)`. A left-out view gets no picture and a zero count. For a detail
  view, load `NN.start.png` too and tone-match once: `img += blur(start - img, 40 px)` where the start has paint.
- **`count` / `rank`.** Zero the count columns of left-out and detail views before ranking: a left-out view is
  never a source, and detail views do not compete by rank.
- **`render(k)`.** After the loop over ring views, a second loop over detail views, in plan order, applying the
  ownership rule above:
  - the same projection and visibility tests as a ring view, with a 5% rim instead of the 1 pixel inside test;
  - a head-on test for every surface (cosine at least 0.5), not only shapes. Room faces and boxes need their
    normal; `evals/density.py`'s `normal()` does that;
  - skip the ceiling, the backdrop, lifted things, `shalo` for room faces, `src_lift` when `BARE`;
  - compare pixels per m2 (`f^2 |n.v| / z^3`, as `density.train_ratio`) against the current owner's, kept in an
    array beside `best`; take the pixel only where the detail view lays more;
  - write `col = ring * (1 - w) + detail * w`, with w rising from 0 at the 5% rim to 1 at 10%.
- **`tgt`.** Keep every ring camera as a training view, left-out ones included (they cost no painting), and append
  the detail cameras. No closer training views are needed at these gains.
- **`write()`.** Unchanged.

### Elsewhere
- `thread/pipeline.py`: pass `WORLD_DETAIL` (6 for Standard) in `env`.
- `evals/density.py` `sources()` / `owner()`: the same ownership rule, or the measured score will not see the
  detail views. `judge.py`'s `owned()` is that rule.

## Risks, worst first

1. **The painter may not add detail.** A picture twice as fine counts as twice the pixels, but started from
   enlarged blurry paint the model may hand the blur back. Nothing in the score sees this.
2. **Drift.** With only a rim kept, the wall may come back as a different wall (other panels, other shelves). The
   tone match fixes colour, not pattern. A pattern step then shows at each rim.
3. **Seams.** Resolution steps rise 1.6 to 2.4 times at 6 pictures, and oblique photos give more of them than
   tiles would. The cross-fade hides tone steps; a step from soft to crisp remains visible as a change of sharpness.
4. **Fewer ring views.** With 6 of 16 gone, objects are seen from fewer sides. The score counts pixels, not
   whether a shape is seen side-on, and I did not measure how often shapes fall back to a grazing view.
5. **Near-flat depth maps.** ex-tower's wall views have a near-to-far ratio of 1.3 to 1.5. The `prep()` change
   above is untested.
6. **Raised ground views.** Outdoors all the gain comes from pictures that are 96 to 100% ground from 3 m up.
   The prompt gives the painter little to go on; it may add paths, a horizon or a vignette.
7. **The start-picture patch** is read from source, not run, and untested with the 8-step distilled ControlNet.
8. **Splat size.** Whether 75 s of training keeps paint 2 to 4 times finer is not modelled.
9. **The agree step's extra time** is a guess.

## First GPU experiment

Two pictures, about 130 s of painting plus the model load, on ex-tower's cached pictures. It answers risks 1 and 2.

The view is the plan's first: eye (2.143, 1.7, 3.111), look (2.143, 1.526, 2.126), 768x576. It faces wall z0 from
3 m with a strip of floor, and is 1.96 times finer than the ring there.

- **P0, no GPU:** the ring paint reprojected into this view (`carry_over`'s picture). This is today's wall.
- **P1, rim only:** paint with `KEEP` = the outer 12% of P0, the usual prompt, seed 7, 8 steps.
- **P2, rim and start:** the same, started from P0 at step 3 of 8 (the two patches above).

Measure on the inner 80%, with `evals/image_scores.py`'s sharpness and reference measures, P0 as the reference:
- sharpness of P1 and P2 against P0: detail was added if it is at least 1.5 times P0's;
- agreement of coarse shapes with P0 (detail between an 8 and a 32 pixel blur): the same wall or another one;
- colour difference after a wide blur, before and after the tone match.

Reading it:
- P1 sharp and the same wall: no mflux patch is needed; build the pass as above.
- Only P2 passes: add the start patch. P2 also runs 5 steps instead of 8, so detail pictures may cost about 40 s
  each, which would buy 9 or 10 pictures for the price of 6.
- Neither is sharper: repainting does not add detail, and the pass is dead whatever the planner does. Sharpness
  then has to come from cheaper pixels (an upscaler on each owner picture), which the same score would rate.
- Sharp but another wall: try a wider rim (25%) or an earlier start before giving up.

If it passes, the second experiment is one raised ground view on ex-blossom: eye (4.375, 3.0, 2.571), look
(3.875, 2.134, 2.571), which is risk 6.

## Not verified

- Anything painted or trained; any GPU or model run; any world besides these three.
- That the painter keeps the look from a rim, or accepts a start picture.
- Paint time for anything but 768x576 and 1024x768. The chosen plan needs neither assumption.
- The agree step's and training's extra seconds.
- How shapes fare with fewer ring views.
- `pick.py` imports `detail_b/plan.py` and `detail_b/score.py` and sets two of their constants in memory. It is a
  planner to judge by, not yet a file that can move into `thread/` by itself.
- The walk set and the "sharp" and "steps" measures are mine. They are not part of `evals/`.
- `judge.py`'s copy of the density maths is from `evals/density.py` as it stood late on 16 Nov 2026; that file is
  still being edited.
