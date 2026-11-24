"""Art direction before the bake (blender_direct.py): the exposure for each hour, the white balance, lamp colours, the sky
panorama's shape and the mends applied in Blender. No model is run; the mend test starts Blender (no window, a few
seconds) and is skipped where Blender is not installed.
  .venv/bin/python -m unittest tests.test_blender_direct        (from the thread folder)
"""
import os, sys, json, subprocess, tempfile, unittest
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "build"))
import blender_direct as D, blender_world as W

class Exposure(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(1); self.even = r.uniform(0.3, 0.5, 20000)                       # a room lit all over
        self.pools = np.concatenate([r.uniform(0.01, 0.04, 19000), r.uniform(1.0, 2.0, 1000)])      # one lamp's pool, the rest dim

    def test_day_puts_a_middling_surface_at_middle_grey(self):
        g = D.exposure(self.even, "day"); self.assertAlmostEqual(g * np.percentile(self.even, 55), 0.62, delta=0.1)

    def test_night_is_set_by_the_pools_and_never_brighter_than_day(self):
        day, night = D.exposure(self.pools, "day"), D.exposure(self.pools, "night")
        self.assertLess(night, day); self.assertGreaterEqual(night, day * 0.35 - 1e-9)
        self.assertLessEqual(D.exposure(self.even, "night"), D.exposure(self.even, "day") + 1e-9)

    def test_ev_is_in_stops_and_bounded(self):
        g = D.exposure(self.even, "day"); self.assertAlmostEqual(D.exposure(self.even, "day", 1.0), 2 * g, places=6)
        self.assertAlmostEqual(D.exposure(self.even, "day", 10.0), 8 * g, places=6)

    def test_no_light_gives_a_plain_exposure(self):
        self.assertEqual(D.exposure([], "day"), 1.0); self.assertEqual(D.exposure([0.0, np.nan], "night"), 1.0)

class Colour(unittest.TestCase):
    def test_white_balance_keeps_brightness_and_some_warmth(self):
        f = D.white_balance([np.array([[1.0, 0.45, 0.15]] * 50)]); lum = np.array([0.2126, 0.7152, 0.0722])
        self.assertAlmostEqual(float(f @ lum), 1.0, places=4)
        out = np.array([1.0, 0.45, 0.15]) * f; self.assertGreater(out[0], out[2])               # still warm
        self.assertLess(out[0] / out[2], 1.0 / 0.15)                                             # but less so

    def test_neutral_light_is_left_alone(self):
        np.testing.assert_allclose(D.white_balance([np.ones((10, 3)) * 0.7]), np.ones(3), atol=1e-6)

    def test_lamp_colours_run_warm_to_cool(self):
        warm, mid, cool = D.kelvin(2700), D.kelvin(4000), D.kelvin(6500)
        self.assertEqual(max(warm), 1.0); self.assertGreater(warm[0], warm[2]); self.assertLess(cool[0], cool[2])
        np.testing.assert_allclose(mid, (1, 1, 1), atol=1e-6)

class Sky(unittest.TestCase):
    def test_a_wide_picture_becomes_a_closed_equirectangular_panorama(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as t:
            a = np.zeros((512, 2048, 3), np.uint8); a[:300] = (90, 140, 220); a[300:] = (60, 90, 40); a[:, :1024, 0] = 200     # sky over ground, left half redder
            Image.fromarray(a).save(f"{t}/raw.png"); D.equirect(f"{t}/raw.png", f"{t}/pano.png", "day")
            p = np.asarray(Image.open(f"{t}/pano.png"), np.float32)
            self.assertEqual(p.shape, (1024, 2048, 3))
            self.assertLess(np.abs(p[:, 0] - p[:, -1]).mean(), 40)                               # the two ends meet
            self.assertGreater(p[100, :, 2].mean(), p[950, :, 2].mean())                          # sky above, ground below

    def test_the_sky_prompt_names_the_building_kind_only(self):
        with tempfile.TemporaryDirectory() as t:
            os.makedirs(f"{t}/materials"); json.dump({"building": "19th-century masonry townhouse with high ceilings, wide-plank timber floors"}, open(f"{t}/materials/style.json", "w"))
            p = D.sky_job(t, "", "night", t)["prompt"]
            self.assertIn("19th-century masonry townhouse", p); self.assertNotIn("timber floors", p); self.assertIn("night", p); self.assertIn("OUTDOORS", p)

def blender_here():
    try: return bool(W.find_blender())
    except SystemExit: return False

@unittest.skipUnless(blender_here(), "Blender is not installed")
class Mends(unittest.TestCase):
    def test_mends_are_applied_within_bounds(self):
        app = W.find_blender()
        with tempfile.TemporaryDirectory() as t:
            make = f"""
import bpy, sys
sys.path.insert(0, {os.path.join(HERE, '..')!r})
import blender_kit as K
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
K.box("floor", (6, 6, 0.1), (0, 0, -0.1)); bpy.data.objects["floor"]["room_part"] = True
K.box("table", (1, 1, 0.7), (0, 0, 0)); K.box("vase", (0.2, 0.2, 0.3), (0, 0, 1.2)); K.box("slab", (2.4, 0.1, 2.0), (2, 0, 0))
K.lamp("fitting light", (0, 0, 2.9), 100); K.lamp("fitting light.001", (1, 0, 2.9), 100); K.lamp("floor lamp light", (2, 2, 1.5), 60)
bpy.context.scene["room"] = [6.0, 6.0, 3.0]
bpy.ops.wm.save_as_mainfile(filepath={t + '/scene.blend'!r})"""
            subprocess.run([app, "-b", "--factory-startup", "--python-expr", make], capture_output=True, check=True)
            json.dump([{"op": "remove", "thing": "slab"}, {"op": "ground", "thing": "vase"}, {"op": "lamp", "lamp": "fitting light", "watts_times": 0},
                       {"op": "lamp", "lamp": "floor lamp light", "watts_times": 9, "kelvin": 2700}, {"op": "exposure", "ev": 5}, {"op": "remove", "thing": "nothing here"}], open(f"{t}/m.json", "w"))
            r = subprocess.run([app, "-b", f"{t}/scene.blend", "--python", os.path.join(HERE, "..", "blender_direct.py"), "--", "mend", t, f"{t}/m.json"], capture_output=True, text=True)
            done = [l[5:] for l in r.stdout.splitlines() if l.startswith("MEND ")]
            self.assertIn("took out slab", done); self.assertTrue(any(l.startswith("set vase down 0.50") for l in done), done)
            check = """
import bpy
sc = bpy.context.scene; L = {o.name: o for o in sc.objects if o.type == 'LIGHT'}
print('RESULT', 'slab' in bpy.data.objects, round(bpy.data.objects['vase'].matrix_world.translation.z, 2), L['fitting light'].hide_render, L['fitting light.001'].data.energy,
      round(L['floor lamp light'].data.energy), round(sc['exposure_ev'], 1))"""
            out = subprocess.run([app, "-b", f"{t}/scene.blend", "--python-expr", check], capture_output=True, text=True).stdout
            res = next(l for l in out.splitlines() if l.startswith("RESULT")).split()[1:]
            self.assertEqual(res, ["False", "0.7", "True", "0.0", "300", "1.0"])                 # lamp x5 at most, exposure +1 in all at most

if __name__ == "__main__":
    unittest.main()
