"""Checks of the painting plan (detail.py, chisel_ring.py's use of it, owner.py's rule for detail views, pipeline.py's
time budget) on a canned room: a crate on a bare floor. No model is asked, nothing is painted.
  .venv/bin/python -m unittest tests.test_detail        (from the thread folder; about half a minute)
"""
import os, sys, json, shutil, subprocess, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__)); THREAD = os.path.dirname(HERE)

SETUP = """import chisel_ring as C, owner as O, detail as DT, json, numpy as np
B = C.B; NS = 6 + len(C.objs)
def sources(extra=(), skip=()):      # the usual views (some left out) and detail views after them, as consistent.py builds them
    views = O.Views(); views.detail = len(extra); counts = []
    for i, cam in enumerate([B.CAMS[str(i)] for i in range(C.N)] + list(extra)):
        if i in skip: views.append(None); counts.append(np.zeros(NS, int)); continue
        t, P, sid = B.trace(cam, C.PRIM_OBJ); views.append(O.view(cam, t, sid, np.zeros(sid.shape, bool), []))
        counts.append(np.bincount(sid.ravel(), minlength=NS) if i < C.N else np.zeros(NS, int))
    return views, O.rank(np.stack(counts, 1).astype(float))
"""
def run(tmp, code, **env):
    r = subprocess.run([sys.executable, "-c", SETUP + code], cwd=THREAD, capture_output=True, text=True,
                       env={**{k: v for k, v in os.environ.items() if not k.startswith("WORLD_")}, "WORLD_DIR": tmp, "WORLD_VIEWS": "8",
                            **({"WORLD_DETAIL_PLAN": "1"} if "WORLD_PAINT_VIEWS" in env else {}), **env})      # the planner is off by default
    if r.returncode: raise AssertionError(r.stderr[-2000:])
    return json.loads(r.stdout.splitlines()[-1])

class Detail(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        box = {"name": "crate", "prims": [{"shape": "box", "c": [3, 0.5, 3], "h": [0.5, 0.5, 0.5], "yaw": 0.0}]}
        with open(f"{self.tmp}/boxes.json", "w") as h: json.dump({"room": [6.0, 3.0, 6.0], "objects": [box]}, h)
    def tearDown(self): shutil.rmtree(self.tmp, ignore_errors=True)

    def test_filters_are_pils(self):
        # the painter's quick mask filters give PIL's MaxFilter and MinFilter to the pixel
        out = run(self.tmp, """from PIL import Image, ImageFilter
rng = np.random.default_rng(1); bad = 0
for m in (rng.random((96, 128)) > 0.99, rng.random((96, 128)) > 0.3, np.pad(np.ones((40, 50), bool), ((0, 56), (30, 48)))):
    for px in (3, 13, 33, 65):
        im = Image.fromarray((m * 255).astype(np.uint8))
        bad += int((np.asarray(im.filter(ImageFilter.MaxFilter(px))) > 127).__ne__(C.grow(m, px)).sum()) + int((np.asarray(im.filter(ImageFilter.MinFilter(px))) > 127).__ne__(C.shrink(m, px)).sum())
print(json.dumps(bad))""")
        self.assertEqual(out, 0)

    def test_no_budget_no_plan(self):
        # without WORLD_PAINT_VIEWS every usual view is painted, as before
        out = run(self.tmp, "C.prep(); import os; print(json.dumps([C.N, C.ND, sorted(C.SKIP), C.painted(), os.path.exists(C.DETAIL_FILE), len([f for f in os.listdir(C.D + '/depth') if f.endswith('.png')])]))")
        self.assertEqual(out, [12, 0, [], list(range(12)), False, 12])

    def test_plan_holds_to_the_budget(self):
        code = """C.prep(); p = json.load(open(C.DETAIL_FILE)); import os
print(json.dumps([C.N, C.ND, sorted(C.SKIP), C.painted(), C.DETAIL_COST, p["views"], sorted(f for f in os.listdir(C.D + '/depth') if f.endswith('.png')), json.load(open(C.D + '/depth/start.json'))["view"], p["sharp"]]))"""
        for budget in ("7", "4.5", "30"):
            n, nd, skip, painted, cost, views, depth, start, sharp = run(self.tmp, code, WORLD_PAINT_VIEWS=budget, WORLD_DETAIL_REPLAN="1", WORLD_STRICT="1")
            self.assertLessEqual(n - len(skip) + nd * cost, float(budget) + 1e-6)     # the time is not overspent
            self.assertNotIn(0, skip); self.assertLessEqual(nd, 6)                    # the first view sets the look; at most WORLD_DETAIL detail views
            self.assertEqual(painted, [i for i in range(n) if i not in skip] + list(range(n, n + nd)))
            self.assertEqual(depth, [f"{i:02d}.png" for i in painted])                # a depth map for every painted view and no other
            self.assertIn(start, painted); self.assertLess(start, n)                  # the viewer starts in a level view that was painted
            if budget == "7": self.assertGreater(nd, 0); first = (skip, views)      # (held strictly to 7 of 12 pictures the walker gets less than from all 12: sharp 0.699 against 0.712)
            if budget == "4.5": self.assertGreaterEqual(n - len(skip), 4)             # never fewer than four usual views
            if budget == "30": self.assertEqual(n - len(skip) + nd, n + nd); self.assertGreater(nd, 0)
        # the plan on disk is what every later step sees, and the same plan comes out every time
        again = run(self.tmp, "print(json.dumps([sorted(C.SKIP), C.ND, [list(map(list, B.CAMS[str(C.N + k)])) for k in range(C.ND)]]))")
        self.assertEqual(again[1], nd); self.assertEqual(again[2], views)
        skip2, views2 = run(self.tmp, code, WORLD_PAINT_VIEWS="7", WORLD_DETAIL_REPLAN="1", WORLD_STRICT="1")[2::3][:2]
        self.assertEqual((skip2, views2), first)

    def test_no_detail_views_when_asked_for_none(self):
        out = run(self.tmp, "C.prep(); print(json.dumps([C.ND, C.N - len(C.SKIP)]))", WORLD_PAINT_VIEWS="7", WORLD_DETAIL="0", WORLD_STRICT="1")
        self.assertEqual(out, [0, 7])

    def test_planner_is_off_by_default(self):
        # pipeline.py always hands over WORLD_PAINT_VIEWS: without WORLD_DETAIL_PLAN=1 every usual view is painted, as before
        out = run(self.tmp, "C.prep(); import os; print(json.dumps([C.ND, sorted(C.SKIP), os.path.exists(C.DETAIL_FILE)]))", WORLD_PAINT_VIEWS="7", WORLD_DETAIL_PLAN="0")
        self.assertEqual(out, [0, [], False])

    def test_a_painted_world_keeps_its_views(self):
        # pictures painted without a plan: prep must not plan (their cameras stay); painted to a plan: that plan stays
        os.makedirs(f"{self.tmp}/gen/images"); open(f"{self.tmp}/gen/images/00.png", "w").close()
        with open(f"{self.tmp}/gen/views.json", "w") as h: json.dump({"views": 8}, h)
        self.assertEqual(run(self.tmp, "C.prep(); print(json.dumps([C.ND, len(C.SKIP)]))", WORLD_PAINT_VIEWS="7"), [0, 0])
        first = run(self.tmp, "C.prep(); print(json.dumps([C.ND, sorted(C.SKIP)]))", WORLD_PAINT_VIEWS="7", WORLD_DETAIL_REPLAN="1")
        with open(f"{self.tmp}/gen/views.json", "w") as h: json.dump({"views": 8, "detail": {}}, h)
        self.assertEqual(run(self.tmp, "C.prep(); print(json.dumps([C.ND, sorted(C.SKIP)]))", WORLD_PAINT_VIEWS="11"), first)

    def test_what_a_detail_view_owns(self):
        # a detail view 1.5 m from wall x0, looking straight at it and a little down
        out = run(self.tmp, """cam = ((1.5, 1.5, 3.0), (0.0, 1.3, 3.0)); views, rank = sources([cam]); N = C.N
yy, zz = np.meshgrid(np.arange(0.05, 3, 0.05), np.arange(0.05, 6, 0.05), indexing="ij")
P = np.stack([np.zeros_like(yy), yy, zz], -1); sid = np.zeros(yy.shape, int)
own = O.owners(C, views, rank, P, sid, None)[0]; ring = O.owners(C, O.Views(views[:N]), rank, P, sid, None)[0]
with np.errstate(all="ignore"): px, py, _, z = B.project(cam, P)
fx, fy = px / (B.W - 1), py / (B.H - 1); inside = (fx > 0.07) & (fx < 0.93) & (fy > 0.07) & (fy < 0.93); rim = (fx < 0.04) | (fx > 0.96) | (fy < 0.04) | (fy > 0.96)
up = np.stack([zz, np.full_like(zz, 3.0), yy * 2], -1); ceil = O.owners(C, views, rank, up, np.full(yy.shape, 4), None)[0]
fl = np.stack([yy, np.zeros_like(yy), zz], -1); floor = O.owners(C, views, rank, fl, np.ones(yy.shape, int), None)[0]
far = ((5.5, 1.5, 3.0), (0.0, 1.3, 3.0)); v2, r2 = sources([far]); own2 = O.owners(C, v2, r2, P, sid, None)[0]
v3, r3 = sources([cam], skip=(3,)); own3 = O.owners(C, v3, r3, P, sid, None)[0]
print(json.dumps([float((own[inside] == N).mean()), int((own[rim] == N).sum()), int((own == N).sum()), int((ceil == N).sum()), float((own[~inside & ~rim] == ring[~inside & ~rim]).mean()),
                  int((own2 == N).sum()), int((own3 == 3).sum()), int((ring == 3).sum()), int((floor == N).sum()), float((own[own != N] == ring[own != N]).mean())]))""")
        self.assertGreater(out[0], 0.98); self.assertEqual(out[1], 0); self.assertGreater(out[2], 500)   # the wall inside its picture, none of the rim
        self.assertEqual(out[3], 0)                                     # never the ceiling
        self.assertEqual(out[9], 1.0)                                   # everything else keeps the owner the ring gave it
        self.assertEqual(out[5], 0)                                     # from across the room it is coarser than the ring: it takes nothing
        self.assertEqual(out[6], 0)                                     # a view that was left out owns nothing

    def test_settle(self):
        # a detail picture 30 brighter than the paint it covers, with a fine pattern: the tone becomes the paint's, the
        # pattern stays, and at the border it is the paint itself
        out = run(self.tmp, """H, W = 192, 256; yy, xx = np.mgrid[:H, :W]; under = np.full((H, W, 3), 100, np.float32) + (xx[..., None] / W * 40)
fine = (((xx // 2 + yy // 2) % 2) * 16 - 8)[..., None].astype(np.float32); raw = under + 30 + fine; have = np.ones((H, W), bool)
out = DT.settle(raw, under, have); mid = np.s_[60:130, 80:180]
half = have.copy(); half[:, W // 2:] = False; out2 = DT.settle(raw, under, half)
print(json.dumps([float(np.abs(out - under)[mid].mean()), float(np.abs((out - under)[mid] - fine[mid]).mean()), float(np.abs(out[0] - under[0]).max()), float(np.abs(out[:, 0] - under[:, 0]).max()),
                  float(np.abs(out2[mid][:, 80:] - raw[mid][:, 80:]).mean())]))""")
        self.assertLess(abs(out[0] - 8), 1.0); self.assertLess(out[1], 1.0)      # what is left is the pattern (amplitude 8), not the 30
        self.assertEqual(out[2], 0); self.assertEqual(out[3], 0)
        self.assertLess(out[4], 2.0)                                             # far from any paint to match, the picture is left as painted

    def test_leave_out(self):
        # three views of four patches: view 2 shows what view 1 shows, less finely, and owns none of it, so it is missed
        # least; view 0 is never left out
        out = run(self.tmp, """S = np.array([[0, 0, 1e6, 1e6], [1e6, 1, 0, 0], [1e6, 2, 1, 1]], float); Q = np.full((3, 4), 200.0); Q[0] = 50; Q[2] = 100
Wt = np.zeros((4, len(DT.BINS))); Wt[:, 6] = 1
print(json.dumps(DT.leave_out(S, Q, Wt)))""")
        self.assertEqual([v for v, _ in out], [2, 1]); self.assertEqual(out[0][1], 0); self.assertGreater(out[1][1], 0.3)

    def test_time_budget(self):
        sys.path.insert(0, THREAD); import pipeline as p
        a = p.paint_views(self.tmp, 100, [768, 576], limit=1200); b = p.paint_views(self.tmp, 400, [768, 576], limit=1200)
        self.assertAlmostEqual(a, (1200 - 100 - 186) / 56, 2); self.assertAlmostEqual(a - b, 300 / 56, 2)
        self.assertIsNone(p.paint_views(self.tmp, 100, [768, 576], limit=0)); self.assertEqual(p.paint_views(self.tmp, 5000, [768, 576], limit=1200), 0)
        self.assertLess(p.paint_views(self.tmp, 100, [1024, 768], limit=1200), a / 2.5)      # the large size takes over 2.5 times as long
        with open(f"{self.tmp}/boxes.json") as h: w = json.load(h)
        w["motions"] = [{"i": 0, "type": "spin"}]
        with open(f"{self.tmp}/boxes.json", "w") as h: json.dump(w, h)
        self.assertAlmostEqual(a - p.paint_views(self.tmp, 100, [768, 576], limit=1200), 135 / 56, 2)      # a second set of views and a second training

if __name__ == "__main__":
    unittest.main()
