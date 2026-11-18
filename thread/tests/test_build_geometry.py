"""The direct method's geometry and its splat writer (build/): plain forms, the architecture kit, how discs are laid
and how they are written to a splat file. Pure numpy, scipy and trimesh: no model is run, nothing is traced.
  .venv/bin/python -m unittest tests.test_build_geometry        (from the thread folder; a few seconds)
"""
import os, sys, math, tempfile, unittest
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "build"), os.path.join(HERE, "..")]
import forms, arch, assemble, light, light_base, level

SIZES = {"rug": (2.4, 0.02, 1.6), "frame": (0.8, 0.6, 0.04), "board": (1.0, 0.2, 0.22), "books": (0.4, 0.25, 0.18), "crate": (0.6, 0.5, 0.4), "cylinder": (0.6, 0.9, 0.6),
         "table": (1.5, 0.9, 0.7), "bottle": (0.08, 0.3, 0.08), "jar": (0.12, 0.16, 0.12), "bowl": (0.3, 0.12, 0.3), "pot": (0.3, 0.45, 0.3), "candle": (0.1, 0.3, 0.1)}

class PlainForms(unittest.TestCase):
    def test_every_form_is_made_and_fills_its_box(self):
        for f in forms.FORMS:
            w, h, d = SIZES.get(f, (1.0, 1.2, 0.5)); V, F, part, names = forms.make(f, w, h, d, seed=f, name=f)
            with self.subTest(form=f):
                self.assertEqual(len(part), len(F)); self.assertTrue(0 <= part.min() and part.max() < len(names), "every face names its part")
                self.assertTrue(0 <= F.min() and F.max() < len(V)); self.assertTrue(np.isfinite(V).all())
                lo, hi = V.min(0), V.max(0)
                self.assertAlmostEqual(lo[1], 0, places=5, msg="it stands on y = 0"); self.assertAlmostEqual(hi[1], h, places=5, msg="its top is the box top, so things rest on it")
                self.assertLessEqual(hi[0] - lo[0], w + 0.05); self.assertLessEqual(hi[2] - lo[2], d + 0.05)       # a cornice or a handle may stand a few centimetres proud

    def test_the_same_seed_makes_the_same_thing(self):
        a, b, c = (forms.make("books", 0.4, 0.25, 0.18, seed=s)[0] for s in ("x", "x", "y"))
        self.assertTrue(np.array_equal(a, b)); self.assertFalse(a.shape == c.shape and np.array_equal(a, c), "another seed, another row of books")

    def test_a_table_has_named_parts(self):
        names = forms.make("table", 1.5, 0.9, 0.7, seed=1)[3]
        self.assertIn("top", names); self.assertTrue(any(n.startswith("leg") for n in names))

    def test_to_room_turns_the_front_to_the_yaw(self):
        P = forms.to_room(np.array([[0.0, 0.0, 1.0]]), 2.0, 0.5, 3.0, math.pi / 2)        # the front (+z) of a thing turned a quarter faces +x
        self.assertTrue(np.allclose(P, [[3.0, 0.5, 3.0]], atol=1e-9))

def room(request, objects=(), size=(5.0, 3.0, 6.0)):
    return {"request": request, "room": list(size), "objects": list(objects), "openings": []}

class Architecture(unittest.TestCase):
    def test_a_plain_room_is_closed_and_true_to_size(self):
        S = arch.shell_geometry(room("A study with a desk."))
        self.assertEqual(S["room"], [5.0, 3.0, 6.0]); pieces = {m["piece"] for m in S["meshes"]}
        self.assertTrue({"floor", "ceiling", "wall"} <= pieces); self.assertEqual(sum(m["piece"] == "wall" for m in S["meshes"]), 4)
        for m in S["meshes"]:
            self.assertEqual(len(m["UV"]), len(m["V"])); self.assertTrue(0 <= m["F"].min() and m["F"].max() < len(m["V"])); self.assertIn(m["group"], S["materials"] | {"glass": 0, "emitter": 0})
        fl = next(m for m in S["meshes"] if m["piece"] == "floor"); self.assertTrue(np.allclose(fl["V"][:, 1], 0, atol=1e-6))
        self.assertTrue(np.allclose([fl["V"][:, 0].min(), fl["V"][:, 0].max(), fl["V"][:, 2].min(), fl["V"][:, 2].max()], [0, 5, 0, 6], atol=0.6), "the floor spans the room (and may run under the walls)")

    def test_floor_uv_is_in_metres(self):
        fl = next(m for m in arch.shell_geometry(room("A study."))["meshes"] if m["piece"] == "floor"); V, F, UV = fl["V"], fl["F"], fl["UV"]
        area = lambda P: np.linalg.norm(np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]), axis=1).sum() / 2
        self.assertAlmostEqual(area(V), area(np.c_[UV, np.zeros(len(UV))]), delta=0.01 * area(V))

    def test_the_sentence_chooses_the_style_and_the_openings(self):
        a = arch.shell_geometry(room("A wizard's study in a medieval stone tower.")); b = arch.shell_geometry(room("A modern garage with five cars.", size=(14.0, 4.5, 12.0)))
        self.assertNotEqual(a["style"], b["style"])
        self.assertFalse(any(e["kind"] == "window" for e in arch.shell_geometry(room("A windowless cellar full of barrels."))["emitters"]), "a windowless place gets no windows")
        self.assertTrue(any(e["kind"] == "window" for e in arch.shell_geometry(room("A bright kitchen with windows along the left wall."))["emitters"]))

    def test_openings_stay_clear_of_tall_things(self):
        case = {"name": "bookcase", "prims": [{"shape": "box", "c": [2.5, 1.1, 5.8], "h": [2.4, 1.1, 0.2]}], "yaw": 0.0, "kind": "bookcase", "form": "bookcase", "attach": "floor", "wall": "back", "said": True, "look": "", "size": [4.8, 2.2, 0.4]}
        S = arch.shell_geometry(room("A library.", [case]))
        for e in S["emitters"]:
            r = np.asarray(e["rect"], float)
            if e["kind"] == "window" and abs(r[:, 2].mean() - 6.0) < 0.5: self.assertTrue(r[:, 1].min() > 2.2 or r[:, 0].max() < 0.1 or r[:, 0].min() > 4.9, "no window in the back wall behind the bookcase")

class Discs(unittest.TestCase):
    def test_snap_takes_a_spacing_to_the_nearest_level(self):
        s = assemble.snap(np.array([0.0001, 0.002, 0.0031, 0.9]))
        self.assertTrue(np.isin(s, assemble.LEVELS).all()); self.assertEqual(s[0], assemble.LEVELS[0]); self.assertEqual(s[-1], assemble.LEVELS[-1]); self.assertTrue(np.all(np.diff(assemble.LEVELS) > 0))

    def test_split_long_keeps_the_area_and_shortens_every_edge(self):
        V = np.array([[0, 0, 0], [3, 0, 0], [3, 0, 1], [0, 0, 1]], float); F = np.array([[0, 1, 2], [0, 2, 3]]); T3, U = assemble.split_long(V, F, V[:, [0, 2]], 0.4)
        e = np.stack([np.linalg.norm(T3[:, i] - T3[:, (i + 1) % 3], axis=1) for i in range(3)], 1); self.assertLessEqual(e.max(), 0.4 + 1e-9)
        self.assertAlmostEqual(np.linalg.norm(np.cross(T3[:, 1] - T3[:, 0], T3[:, 2] - T3[:, 0]), axis=1).sum() / 2, 3.0, places=6)
        self.assertTrue(np.allclose(U, T3[:, :, [0, 2]]), "UVs are cut with their triangles")

    def test_lattice_lays_one_point_a_cell_on_a_rectangle(self):
        V = np.array([[0, 0, 0], [2, 0, 0], [2, 0, 1], [0, 0, 1]], float); F = np.array([[0, 1, 2], [0, 2, 3]]); T3, U = assemble.split_long(V, F, V[:, [0, 2]], 0.4)
        P, uv, tid = assemble.lattice(T3, U, 0.05); P = np.unique(np.round(P, 6), axis=0)
        self.assertAlmostEqual(len(P) / (41 * 21), 1.0, delta=0.02); self.assertTrue(np.allclose(P[:, 1], 0)); self.assertTrue(P[:, 0].min() >= -1e-6 and P[:, 0].max() <= 2 + 1e-6)
        g = np.round(P[:, [0, 2]] / 0.05); self.assertTrue(np.allclose(g * 0.05, P[:, [0, 2]], atol=1e-6), "every point sits on the shared lattice, so neighbouring pieces line up")

    def test_lattice_gives_a_missed_sliver_its_centroid(self):
        T3 = np.array([[[0.01, 0, 0.01], [0.02, 0, 0.01], [0.01, 0, 0.02]]]); P, uv, tid = assemble.lattice(T3, T3[:, :, [0, 2]], 0.05)
        self.assertEqual(len(P), 1); self.assertTrue(np.allclose(P[0], T3[0].mean(0)))

    def test_poisson_keeps_points_apart_and_leaves_no_holes(self):
        from scipy.spatial import cKDTree
        P = np.c_[np.random.default_rng(0).random((20000, 2)), np.zeros(20000)]; r = 0.03; k = assemble.poisson(P, r, rounds=3); Q = P[k]
        d = cKDTree(Q).query(Q, k=2)[0][:, 1]; self.assertGreaterEqual(d.min(), r * 0.5); self.assertLess(cKDTree(Q).query(P)[0].max(), 2 * r, "every thrown point has a kept one near it")

    def test_quat_turns_z_onto_the_normal(self):
        N = np.array([[0, 0, 1.0], [0, 1, 0], [1, 0, 0], [0, 0, -1], [0.6, 0.0, 0.8]]); q = light_base.quat_z_to(N); w, x, y, z = q.T
        zr = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)       # where each turn takes the z axis
        self.assertTrue(np.allclose(zr, N, atol=1e-6)); self.assertTrue(np.allclose(np.linalg.norm(q, axis=1), 1))

    def discs(self, n=50):
        rng = np.random.default_rng(1); N = np.tile([0.0, 1.0, 0.0], (n, 1))
        d = {"xyz": rng.random((n, 3)) * [4, 3, 6], "normal": N, "sigma": np.full(n, 0.004, np.float32), "thick": np.full(n, 0.0005, np.float32)}
        res = {"rgb": rng.integers(0, 255, (n, 3)).astype(np.uint8), "opacity": np.full(n, 0.97, np.float32), "keep": np.ones(n, bool), "extra": None, "sh": None}
        return d, res, {"m": [2.0, 1.5, -3.0], "k": 1 / 3.0}

    def test_rows_put_discs_in_the_file_frame(self):
        d, res, pl = self.discs(); res["keep"][:10] = False; a, idx = light.rows(d, res, pl)
        self.assertEqual(a.shape, (40, 14)); self.assertTrue(np.array_equal(idx, np.arange(10, 50)))
        self.assertTrue(np.allclose(a[:, :3], (d["xyz"][idx] * [1, 1, -1] - pl["m"]) * pl["k"], atol=1e-6), "z mirrored, the room's middle at the origin, scaled")
        self.assertTrue(np.allclose(a[:, 3:6] * light_base.C0 + 0.5, res["rgb"][idx] / 255, atol=1e-6)); self.assertTrue(np.allclose(1 / (1 + np.exp(-a[:, 6])), 0.97, atol=1e-5))
        self.assertTrue(np.allclose(np.exp(a[:, 7:9]), 0.004 * pl["k"], rtol=1e-5)); self.assertTrue(np.all(a[:, 9] < a[:, 7]), "a disc is flat: thinner than it is wide")

    def test_write_makes_a_splat_file_and_a_second_one_for_shine(self):
        d, res, pl = self.discs(); res["sh"] = {"idx": np.arange(0, 50, 5), "coef": np.random.default_rng(2).random((10, 15, 3)).astype(np.float32)}
        with tempfile.TemporaryDirectory() as t:
            info = light.write(f"{t}/world.ply", d, res, pl); self.assertEqual((info["discs"], info["sh_discs"]), (50, 10))
            def read(p):
                with open(p, "rb") as f: raw = f.read()
                cut = raw.index(b"end_header\n") + 11; head = raw[:cut].decode().split("\n"); cols = [l.split()[-1] for l in head if l.startswith("property")]
                n = int(next(l for l in head if l.startswith("element vertex")).split()[-1]); return cols, np.frombuffer(raw[cut:], "<f4").reshape(n, len(cols))
            c1, a1 = read(f"{t}/world.ply"); c2, a2 = read(f"{t}/world_sh.ply")
            self.assertEqual(c1, light_base.PLY14); self.assertEqual(len(a1), 40); self.assertEqual(len(c2), 59); self.assertEqual(len(a2), 10)
            self.assertEqual(c2[6:51], [f"f_rest_{i}" for i in range(45)]); self.assertTrue(np.isfinite(a1).all() and np.isfinite(a2).all())
            res["sh"] = None; light.write(f"{t}/world.ply", d, res, pl); self.assertFalse(os.path.exists(f"{t}/world_sh.ply"), "a build without shine leaves no stale second file")

class Colours(unittest.TestCase):
    def test_a_colour_word_is_found_and_a_thing_recoloured_to_it(self):
        self.assertEqual(level.colour_word("a small navy-blue car")[0], "navy"); self.assertEqual(level.colour_word("a wooden chair"), (None, None))
        red = np.tile(np.float32([0.8, 0.1, 0.1]), (500, 1)); grey = np.tile(np.float32([0.5, 0.5, 0.5]), (100, 1)); out = level.recolour(np.concatenate([red, grey]), level.COLOURS["blue"])
        self.assertTrue(out[0, 2] > out[0, 0], "the red body turns blue"); self.assertTrue(np.allclose(out[-1], 0.5, atol=1e-3), "what has no colour stays as it is")
        self.assertTrue(np.allclose(level.hsv2rgb(level.rgb2hsv(red[:1])), red[:1], atol=1e-5))

class StartView(unittest.TestCase):
    def test_the_visitor_starts_on_the_floor_looking_at_what_was_named(self):
        box = lambda lo, hi: dict(zip("VF", forms._box(lo, hi)))
        room = box((0, 0, 0), (8, 3, 6)); room["F"] = room["F"][:, ::-1]; plat = box((2.5, 0, 1.5), (5.5, 0.5, 4.5))        # a platform fills the middle of the room
        Lv = {"room": np.array([8.0, 3.0, 6.0]), "b": {"objects": [{"said": True, "prims": [{"c": [4.0, 1.0, 3.0]}]}]}}
        eye, f = assemble.start_view(Lv, light_base.Scene([room, plat]))
        self.assertFalse(2.2 < eye[0] < 5.8 and 1.2 < eye[2] < 4.8, "not standing up on the platform")
        to = np.array([4.0, 0, 3.0]) - eye; to[1] = 0; self.assertGreater(float(f @ to / np.linalg.norm(to)), 0.99, "looking at the named thing")
        self.assertAlmostEqual(float(np.linalg.norm(f)), 1.0, places=5); self.assertEqual(float(f[1]), 0.0)

class StandIn(unittest.TestCase):
    def test_a_lamp_stand_in_has_a_part_that_shines(self):
        for size in ((1.2, 0.12, 0.1), (0.4, 1.6, 0.4)):          # a ceiling fitting, a standing lamp
            V, F, part, names = forms.make("lamp", *size, seed=1); self.assertEqual(sorted(names), ["glow", "housing"])
            self.assertTrue(np.allclose(V.max(0) - V.min(0), size, atol=1e-6)); self.assertTrue((part == names.index("glow")).any())

if __name__ == "__main__": unittest.main()
