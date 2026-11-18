"""The Unreal export (build/export.py): its frame, its unwrap and texel raster, and the glTF file it writes. No world is
built and no model is run.
  .venv/bin/python -m unittest tests.test_build_export        (from the thread folder; a few seconds)
"""
import os, sys, json, struct, tempfile, unittest
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "build"), os.path.join(HERE, "..")]
import export

def box(w=2.0, h=1.0, d=0.5):
    import trimesh
    m = trimesh.creation.box((w, h, d)); return np.asarray(m.vertices, float) + [w / 2, h / 2, d / 2], np.asarray(m.faces, np.int64)

def read_glb(path):
    with open(path, "rb") as f: b = f.read()
    magic, ver, total = struct.unpack("<III", b[:12]); n, kind = struct.unpack("<II", b[12:20])
    return magic, ver, total, len(b), kind, json.loads(b[20:20 + n])

class Frame(unittest.TestCase):
    def test_room_to_gltf_mirrors_z_and_centres_the_floor(self):
        pos, vec = export.frame([14.0, 4.5, 12.0])
        np.testing.assert_allclose(pos(np.array([[7.0, 0.0, 6.0], [0.0, 2.0, 0.0], [14.0, 0.0, 12.0]])), [[0, 0, 0], [-7, 2, 6], [7, 0, -6]])
        np.testing.assert_allclose(vec(np.array([[0.0, 0.0, 1.0]])), [[0, 0, -1]])

    def test_a_light_node_turns_minus_z_onto_its_direction(self):
        from scipy.spatial.transform import Rotation
        for d in ([0, -1, 0], [1, 0, 0], [0, 0, 1], [0, 0, -1], [0.3, -0.8, 0.5]):
            d = np.asarray(d, float) / np.linalg.norm(d); q = export.quat_from_to([0, 0, -1], d)
            np.testing.assert_allclose(Rotation.from_quat(q).apply([0, 0, -1]), d, atol=1e-9)

class Raster(unittest.TestCase):
    def test_every_texel_inside_a_triangle_is_found_with_its_barycentrics(self):
        W = H = 32; T = np.array([[[2.0, 2.0], [30.0, 4.0], [6.0, 29.0]]]); tri, bary = export.raster(T, W, H)
        ys, xs = np.mgrid[0:H, 0:W]; c = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1); a, b, cc = T[0]
        den = (b[0] - a[0]) * (cc[1] - a[1]) - (b[1] - a[1]) * (cc[0] - a[0]); l1 = ((b[0] - c[:, 0]) * (cc[1] - c[:, 1]) - (b[1] - c[:, 1]) * (cc[0] - c[:, 0])) / den
        l2 = ((cc[0] - c[:, 0]) * (a[1] - c[:, 1]) - (cc[1] - c[:, 1]) * (a[0] - c[:, 0])) / den; inside = (l1 >= 0) & (l2 >= 0) & (1 - l1 - l2 >= 0)
        self.assertTrue((tri[inside] == 0).all(), "no texel inside is missed")
        got = tri >= 0; self.assertLessEqual(got.sum(), inside.sum() * 1.4, "only a thin rim outside is taken")
        np.testing.assert_allclose(bary[got].sum(1), 1, atol=1e-5)
        p = (bary[got][:, :, None] * T[0][None]).sum(1); np.testing.assert_allclose(p[inside[got]], c[inside & got], atol=1e-4)

    def test_fill_spreads_covered_texels_into_the_gutter(self):
        img = np.zeros((4, 4, 3), np.float32); have = np.zeros((4, 4), bool); img[1, 1] = 200; have[1, 1] = True
        out = export.fill(img, have); self.assertTrue((out == 200).all())

class Unwrap(unittest.TestCase):
    def test_a_box_unwraps_into_one_atlas_within_bounds(self):
        V, F = box(); vmap, I, uv, W, H, tpu = export.unwrap(V, F, 64)
        self.assertEqual(len(I), len(F)); self.assertTrue((uv >= 0).all() and (uv <= 1).all()); self.assertLessEqual(max(W, H), export.MAX_TEX)
        np.testing.assert_allclose(np.sort(np.linalg.norm(V[vmap][I] - V[vmap][I][:, [1, 2, 0]], axis=2), 1), np.sort(np.linalg.norm(V[F] - V[F][:, [1, 2, 0]], axis=2), 1), atol=1e-6)

    def test_a_large_piece_is_held_to_the_largest_texture(self):
        V, F = box(60.0, 10.0, 40.0); _, _, _, W, H, tpu = export.unwrap(V, F, 400)
        self.assertLessEqual(max(W, H), export.MAX_TEX); self.assertLess(tpu, 400)

class Gltf(unittest.TestCase):
    def test_the_file_is_valid_glb_that_trimesh_reads_back(self):
        import trimesh
        V, F = box(); vmap, I, uv, W, H, _ = export.unwrap(V, F, 32); pos, vec = export.frame([4.0, 3.0, 4.0])
        G = export.GLB(); G.room_node = G.node({"name": "Room"}); G.lights_node = G.node({"name": "Lights"})
        img = np.full((H, W, 3), 128, np.float32); mi = G.material("crate", export.jpeg(img), export.jpeg(img), emit=0.0)
        P = pos(V[vmap]); N = vec(np.tile([0.0, 1.0, 0.0], (len(P), 1))); G.mesh("crate", P, N, uv, I[:, [0, 2, 1]], mi, G.room_node)
        G.light("lamp light", "point", [1, 0.9, 0.8], 40, [0, 2, 0]); G.light("Sun", "directional", [1, 1, 1], 60000, [0, 3, 0], [0.3, -0.9, 0.1])
        with tempfile.TemporaryDirectory() as t:
            path = os.path.join(t, "x.glb"); G.save(path); magic, ver, total, size, kind, j = read_glb(path)
            self.assertEqual((magic, ver, kind), (0x46546C67, 2, 0x4E4F534A)); self.assertEqual(total, size)
            self.assertEqual(len(j["extensions"]["KHR_lights_punctual"]["lights"]), 2); self.assertIn("KHR_lights_punctual", j["extensionsUsed"])
            self.assertEqual(j["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"], 1)
            for v in j["bufferViews"]: self.assertEqual(v["byteOffset"] % 4, 0)
            s = trimesh.load(path); self.assertEqual(sum(len(g.faces) for g in s.geometry.values()), len(F))
            np.testing.assert_allclose(s.bounds, [P.min(0), P.max(0)], atol=1e-5)

    def test_unlit_materials_carry_no_lighting_maps(self):
        G = export.GLB(unlit=True); img = np.zeros((4, 4, 3), np.float32); G.material("m", export.jpeg(img), export.jpeg(img), emit=4.0); m = G.j["materials"][0]
        self.assertIn("KHR_materials_unlit", m["extensions"]); self.assertNotIn("metallicRoughnessTexture", m["pbrMetallicRoughness"]); self.assertNotIn("emissiveTexture", m)

if __name__ == "__main__":
    unittest.main()
