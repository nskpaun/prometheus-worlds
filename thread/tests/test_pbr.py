"""The visuals-first method (thread/pbr): the scene description, room outlines and where openings land on them, the CC0
library's matching, tileable material pictures and height from normals, the sky's directions, seam and HDR lift. No
model is run.
  .venv/bin/python -m unittest tests.test_pbr        (from the thread folder)
"""
import os, sys, json, math, tempfile, unittest
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "pbr")); sys.path.insert(0, os.path.join(HERE, "..", "build"))
import ir, plan as P, kitbash as K, materials as M, sky as S

class Description(unittest.TestCase):
    def test_round_trip(self):
        s = ir.SceneIR(sentence="a study", setting="indoor", room=ir.RoomShell(footprint=[[0, 0], [4, 0], [4, 3], [0, 3]], height_m=2.8, openings=[ir.Opening(kind="window", wall_index=1, u_centre=1.5, sill_m=0.9, width_m=1.2, height_m=1.4)]),
                       materials={"wall": ir.PBRMaterial(name="wall", prompt="plaster")}, assets=[ir.AssetInstance(id="a", name="desk", source="hero_3d", prompt="desk", position=[0, 0, 0])],
                       lights=[ir.PhysicalLight(id="l", kind="point", position=[0, 0, 2])])
        with tempfile.TemporaryDirectory() as t:
            ir.save(s, f"{t}/s.json"); r = ir.load(f"{t}/s.json")
        self.assertEqual(r.room.openings[0].wall_index, 1); self.assertEqual(r.materials["wall"].prompt, "plaster"); self.assertEqual(r.assets[0].source, "hero_3d"); self.assertEqual(r.lights[0].kind, "point")

class Outline(unittest.TestCase):
    def area(self, pts): return 0.5 * sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))

    def test_shapes_only_add_to_the_room(self):
        for kind in ("rect", "bay", "alcove", "apse"):
            pts, _ = P.footprint(6, 8, {"kind": kind, "wall": "back", "width": 2.5, "depth": 1.0}, [])
            self.assertGreaterEqual(self.area(pts), 48 - 1e-6, kind)              # anticlockwise and never smaller than the box
            for x, y in [(-3, -4), (3, -4), (3, 4), (-3, 4)]: self.assertIn([x, y], [list(map(float, p)) for p in pts] if kind == "rect" else [[float(a), float(b)] for a, b in pts])

    def test_chamfer_only_where_the_corners_are_free(self):
        self.assertEqual(len(P.footprint(6, 8, {"kind": "chamfer"}, [])[0]), 8)
        chair = __import__("ir").AssetInstance(id="c", name="chair", source="plain", prompt="", position=[2.7, 3.7, 0], size_m=[0.5, 0.5, 0.9])
        self.assertEqual(len(P.footprint(6, 8, {"kind": "chamfer"}, [chair])[0]), 4)

    def test_openings_land_on_their_wall_and_in_a_bay(self):
        pts, _ = P.footprint(6, 10, {"kind": "bay", "wall": "left", "width": 2.5, "depth": 1.0}, [])
        e, u = P.on_outline(pts, 6, 10, "back", 3.0); a, b = pts[e], pts[(e + 1) % len(pts)]
        self.assertAlmostEqual(a[1], 5.0); self.assertAlmostEqual(b[1], 5.0); self.assertAlmostEqual(u, 3.0)
        e, u = P.on_outline(pts, 6, 10, "left", 5.0); a, b = pts[e], pts[(e + 1) % len(pts)]
        self.assertLess(min(a[0], b[0]), -3.5)                                   # the bay's far face, out past the old wall
        self.assertAlmostEqual(u, math.dist(a, b) / 2, places=5)

    def test_sun_comes_in_by_the_windows(self):
        d, w, k = P.sun_of("golden_hour", (0, 1)); self.assertGreater(d[1], 0.5); self.assertGreater(d[2], 0); self.assertLess(d[2], 0.3)
        self.assertLess(P.sun_of("night")[1], 0.1)

class Library(unittest.TestCase):
    def setUp(self):
        K._INDEX = [{"id": i, "name": n, "categories": c, "tags": t, "description": "", "size": z, "path": i + ".gltf"} for i, n, c, t, z in (
            ("round_wooden_table_01", "Round Wooden Table 01", ["furniture", "table"], ["wooden", "round"], [1.4, 1.4, 1.0]),
            ("ceramic_vase_01", "Ceramic Vase 01", ["decorative", "vases"], ["ceramic", "pottery"], [0.2, 0.2, 0.35]),
            ("brass_candleholders", "Brass Candleholders", ["decorative"], ["brass", "candle"], [0.1, 0.1, 0.25]),
            ("mug_01", "Coffee Mug", ["dishes"], ["cup", "coffee"], [0.1, 0.08, 0.1]))]
        for m in K._INDEX: m["_w"] = (set(K.words(m["name"])), set(K.words(" ".join(m["tags"] + m["categories"]))), set(K.words(m["description"])))
    def tearDown(self): K._INDEX = None

    def test_the_head_word_decides(self):
        self.assertEqual(K.match("round oak table", [0.8, 0.8, 0.75])["id"], "round_wooden_table_01")
        self.assertEqual(K.match("ceramic mug", [0.09, 0.09, 0.1])["id"], "mug_01")
        self.assertEqual(K.match("small blue vase")["id"], "ceramic_vase_01")

    def test_nothing_like_it(self):
        self.assertIsNone(K.match("grand piano", [1.5, 2.0, 1.0]))

class Materials(unittest.TestCase):
    def test_tileable_edges_meet(self):
        r = np.random.default_rng(0); a = r.uniform(0, 1, (256, 256, 3)).astype(np.float32); a[:, :128] += 1.0          # a hard seam across the middle and at the edges
        t = M.tileable(a, band=48)
        self.assertLess(np.abs(t[:, 0] - t[:, -1]).mean(), np.abs(a[:, 0] - a[:, -1]).mean())
        self.assertLess(np.abs(t[0] - t[-1]).mean(), 0.6)

    def test_height_from_a_bump(self):
        yy, xx = np.mgrid[0:128, 0:128]; h = np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / 200.0)
        gy, gx = np.gradient(h); n = np.dstack([-gx * 8, gy * 8, np.ones_like(h)]); n /= np.linalg.norm(n, axis=-1, keepdims=True)
        z = M.height_from_normal(n); self.assertGreater(z[64, 64], z[5, 5]); self.assertGreater(np.corrcoef(z.ravel(), h.ravel())[0, 1], 0.9)

class Sky(unittest.TestCase):
    def test_directions_match_blenders_lookup(self):
        d = S.direction(np.array([0, 512, 512, 512]), np.array([0, 1024, 512, 1536]))
        np.testing.assert_allclose(d[0], [0, 0, 1], atol=0.01); np.testing.assert_allclose(d[1], [1, 0, 0], atol=0.01)
        np.testing.assert_allclose(d[2], [0, 1, 0], atol=0.01); np.testing.assert_allclose(d[3], [0, -1, 0], atol=0.01)

    def test_seam_closes_round(self):
        a = np.zeros((S.H, S.W, 3), np.float32); a[:, : S.W // 2] = 1.0
        b = S.seam(a); self.assertLess(np.abs(b[:, 0] - b[:, -1]).mean(), np.abs(a[:, 0] - a[:, -1]).mean())

    def test_lift_has_the_sun_only_for_the_viewer(self):
        ldr = np.full((S.H, S.W, 3), 0.55, np.float32)
        world, view = S.lift(ldr, "day", [0.5, 0.2, 0.8]), S.lift(ldr, "day", [0.5, 0.2, 0.8], disc_too=True)
        self.assertLess(world.max(), 100); self.assertGreater(view.max(), 1e3)
        self.assertGreater(S.lift(ldr, "day", [0, 0, 1]).mean(), S.lift(ldr, "night", [0, 0, 1]).mean() * 20)

if __name__ == "__main__":
    unittest.main()
