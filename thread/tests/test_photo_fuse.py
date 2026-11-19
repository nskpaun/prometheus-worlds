"""The merge of photo pieces (photo_fuse.py): its keys, fits, colour gains, owners, grid blocks, discs and the file it writes.
No model is run and no world is built.
  .venv/bin/python -m unittest tests.test_photo_fuse        (from the thread folder; a few seconds)
"""
import os, sys, tempfile, unittest
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import photo_fuse as F

class Keys(unittest.TestCase):
    def test_cells_pack_to_one_number_and_back(self):
        k = np.array([[0, 0, 0], [-5, 7, -900], [1000, -1000, 3], [-1, -1, -1]]); key = F.pack(k)
        np.testing.assert_array_equal(F.unpack(key), k); self.assertEqual(len(np.unique(key)), len(k))

class Fits(unittest.TestCase):
    def test_depth_fit_finds_scale_and_shift_despite_outliers(self):
        r = np.random.default_rng(0); zs = r.uniform(0.8, 3.0, 400); a, b = 1.25, -0.03; zt = 1 / (a / zs + b)
        zt[:40] *= r.uniform(1.4, 2.0, 40)                                      # a tenth of the points are wrong matches
        fa, fb, err = F.fit_inverse_depth(zs, zt); self.assertAlmostEqual(fa, a, 2); self.assertAlmostEqual(fb, b, 2); self.assertLess(err, 0.01)

    def test_gains_bring_photos_of_the_same_points_together(self):
        r = np.random.default_rng(1); true = np.log(r.uniform(0.2, 0.8, (300, 3))); bright = np.array([0.0, 0.25, -0.2, 0.1]); rows = []
        for i, g in enumerate(bright):
            ids = r.choice(300, 200, replace=False); rows.append(np.c_[ids, np.full(200, i), true[ids] + g])
        gain = F.photo_gains(np.concatenate(rows), 4, pull=0.01); fixed = bright + np.log(gain).mean(1)
        self.assertLess(fixed.std(), 0.01, "what is left after the gains is the same for every photo")

class Owners(unittest.TestCase):
    def cells(self, n=12):
        k = np.array([(x, 0, z) for x in range(n) for z in range(n)]); key = F.pack(k); order = np.argsort(key); return k[order], key[order]

    def test_the_cheapest_photo_owns_a_cell(self):
        own = F.pick_owner(np.array([0, 0, 1, 2, 2]), np.array([3, 4, 5, 6, 7]), np.array([2.0, 1.0, 9.0, 1.0, 0.5]), 4)
        np.testing.assert_array_equal(own, [4, 5, 7, -1])

    def test_a_lone_cell_joins_the_patch_around_it(self):
        k, key = self.cells(); n = len(key); ci = np.r_[np.arange(n), np.arange(n)]; photo = np.r_[np.zeros(n, int), np.ones(n, int)]
        cost = np.r_[np.full(n, 1.0), np.full(n, 1.1)]; mid = int(np.flatnonzero((k[:, 0] == 6) & (k[:, 2] == 6))[0]); cost[mid] = 1.2     # photo 0 is a little worse in one cell
        self.assertEqual(F.settle_owners(key, ci, photo, cost, smooth=0)[mid], 1)
        self.assertTrue((F.settle_owners(key, ci, photo, cost, smooth=0.5) == 0).all())

class Grid(unittest.TestCase):
    def test_blocks_hold_both_layers_of_each_square_and_unblock_back(self):
        idx = np.arange(2 * F.N * F.N); b = 4; bl = F.blocks(idx, b); self.assertEqual(bl.shape, (F.N // b, F.N // b, 2 * b * b))
        row, col, lay = (idx % (F.N * F.N)) // F.N, idx % F.N, idx // (F.N * F.N); one = bl[5, 9]
        self.assertTrue((row[one] // b == 5).all() and (col[one] // b == 9).all()); self.assertEqual(sorted(np.bincount(lay[one]).tolist()), [b * b, b * b])
        flag = np.zeros((F.N // b, F.N // b), bool); flag[5, 9] = True; per = F.unblocks(flag, b); np.testing.assert_array_equal(np.sort(np.flatnonzero(per)), np.sort(one))
        vec = np.stack([idx, idx], 1).astype(np.float32); self.assertEqual(F.blocks(vec, 2).shape, (F.N // 2, F.N // 2, 8, 2))

    def test_discs_face_along_their_normals(self):
        from scipy.spatial.transform import Rotation
        n = np.array([[0, 0, 1.0], [0, 0, -1.0], [1, 0, 0], [0.3, -0.8, 0.5]]); n /= np.linalg.norm(n, axis=1, keepdims=True); q = F.discs_from_normals(n)
        np.testing.assert_allclose(Rotation.from_quat(q[:, [1, 2, 3, 0]]).apply([0, 0, 1.0]), n, atol=1e-6)

class Flatten(unittest.TestCase):
    def test_a_splat_long_along_the_normal_is_squashed_and_the_rest_left(self):
        from scipy.spatial.transform import Rotation
        q = np.tile([1.0, 0, 0, 0], (3, 1)); sc = np.array([[0.01, 0.01, 0.2], [0.05, 0.05, 0.001], [0.2, 0.01, 0.01]]); n = np.tile([0, 0, 1.0], (3, 1))
        q2, s2 = F.flatten(q, sc, n, np.full(3, 0.01))
        def reach(q, s, d): R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix(); return np.sqrt(np.einsum("i,nij,nj,nkj,k->n", d, R, s ** 2, R, d))
        np.testing.assert_allclose(reach(q2, s2, np.array([0, 0, 1.0])), [0.01, 0.001, 0.01], atol=1e-6)        # along the normal: capped, or as it was
        np.testing.assert_allclose(reach(q2, s2, np.array([1.0, 0, 0])), sc[:, 0], atol=1e-6)                 # across it: unchanged
        np.testing.assert_allclose(q2[1:], q[1:]); np.testing.assert_allclose(s2[1:], sc[1:])

    def test_a_tilted_splat_keeps_its_spread_across_the_surface(self):
        from scipy.spatial.transform import Rotation
        r = Rotation.from_euler("y", 45, degrees=True); q = r.as_quat()[[3, 0, 1, 2]][None]; sc = np.array([[0.01, 0.01, 0.3]]); n = np.array([[0, 0, 1.0]])
        q2, s2 = F.flatten(q, sc, n, np.array([0.01])); R = Rotation.from_quat(q2[:, [1, 2, 3, 0]]).as_matrix()[0]; cov = R @ np.diag(s2[0] ** 2) @ R.T
        self.assertAlmostEqual(np.sqrt(cov[2, 2]), 0.01, 5); self.assertAlmostEqual(np.sqrt(cov[0, 0]), np.sqrt((0.3 ** 2 + 0.01 ** 2) / 2), 4); self.assertAlmostEqual(np.linalg.det(R), 1, 6)

class Files(unittest.TestCase):
    def test_the_splat_file_reads_back(self):
        P = np.array([[0, 1, 2], [3, 4, 5.0]]); rgb = np.array([[0.5, 0.5, 0.5], [1, 0, 0.25]]); op = np.array([0.5, 0.9]); sc = np.full((2, 3), 0.01); q = np.tile([1.0, 0, 0, 0], (2, 1))
        with tempfile.TemporaryDirectory() as t:
            path = os.path.join(t, "s.ply"); self.assertEqual(F.write_ply(path, P, rgb, op, sc, q), 2)
            with open(path, "rb") as fh: raw = fh.read()
            head, body = raw.split(b"end_header\n"); self.assertIn(b"element vertex 2", head); A = np.frombuffer(body, "<f4").reshape(2, 14)
            np.testing.assert_allclose(A[:, :3], P); np.testing.assert_allclose(A[:, 3:6] * F.SH0 + 0.5, rgb, atol=1e-6)
            np.testing.assert_allclose(1 / (1 + np.exp(-A[:, 6])), op, atol=1e-6); np.testing.assert_allclose(np.exp(A[:, 7:10]), sc, rtol=1e-5)

    def test_solved_cameras_are_read_from_text(self):
        with tempfile.TemporaryDirectory() as t:
            for name, text in (("cameras", "# c\n1 PINHOLE 1920 1440 1400.0 1400.0 960 720\n"), ("points3D", "# p\n7 1.0 2.0 3.0 9 9 9 0.5 1 0\n"),
                               ("images", "# i\n2 1 0 0 0 0.5 0 -2 1 b.jpg\n10 20 7 30 40 -1\n1 1 0 0 0 0 0 0 1 a.jpg\n\n")):
                with open(f"{t}/{name}.txt", "w") as fh: fh.write(text)
            ims = F.read_colmap(t); self.assertEqual([i["name"] for i in ims], ["a.jpg", "b.jpg"]); b = ims[1]
            self.assertEqual(b["cam"], (1920, 1440, 1400.0)); np.testing.assert_allclose(b["C"], [-0.5, 0, 2]); np.testing.assert_allclose(b["X"], [[1, 2, 3]]); np.testing.assert_allclose(b["xy"], [[10, 20]])
            self.assertEqual(len(ims[0]["X"]), 0)

if __name__ == "__main__":
    unittest.main()
