"""Checks of owner.py, the rule for which painted view a surface point takes its paint from, on a canned room: a box
that spins (so it is lifted out) in the middle of a bare floor. No model is asked, nothing is painted.
  .venv/bin/python -m unittest tests.test_owner        (from the thread folder; a few seconds)
"""
import os, sys, json, shutil, subprocess, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__)); THREAD = os.path.dirname(HERE)

SETUP = """import chisel_ring as C, owner as O, json, numpy as np
B = C.B; NS = 6 + len(C.objs); lifted = C.LIFT
traced = [B.trace(B.CAMS[str(i)], C.PRIM_OBJ) for i in range(C.N)]
views = [O.view(B.CAMS[str(i)], t, sid, np.zeros(sid.shape, bool), lifted) for i, (t, P, sid) in enumerate(traced)]
rank = O.rank(np.stack([np.bincount(v[2].ravel(), minlength=NS) for v in views], 1).astype(float))
lo, hi = C.bounds(C.OBJ_PRIMS[0])
def paint(dark):      # every view shows the floor at brightness 200, and 60 where dark(P) says so
    out = []
    for t, P, sid in traced:
        im = np.full(sid.shape + (3,), 200, np.float32); im[(sid == 1) & dark(P)] = 60; out.append(im)
    return out
gx, gz = np.meshgrid(np.arange(0.5, 5.5, 0.05), np.arange(0.5, 5.5, 0.05), indexing="ij")
P = np.stack([gx, np.zeros_like(gx), gz], -1); sid = np.ones(gx.shape, int); d = O.near(P, lo, hi); band = (d < 1.0) & (d > 0)
def own(shadow): return O.owners(C, views, rank, P, sid, None, lifted, shadow)
"""
def run(tmp, code, **env):
    r = subprocess.run([sys.executable, "-c", SETUP + code], cwd=THREAD, capture_output=True, text=True,
                       env={**os.environ, "WORLD_DIR": tmp, "WORLD_VIEWS": "8", **env})
    if r.returncode: raise AssertionError(r.stderr[-2000:])
    return json.loads(r.stdout.splitlines()[-1])

class Owner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        box = {"name": "crate", "prims": [{"shape": "box", "c": [3, 0.5, 3], "h": [0.5, 0.5, 0.5], "yaw": 0.0}]}
        with open(f"{self.tmp}/boxes.json", "w") as h: json.dump({"room": [6.0, 3.0, 6.0], "objects": [box], "motions": [{"i": 0, "type": "spin"}]}, h)
    def tearDown(self): shutil.rmtree(self.tmp, ignore_errors=True)

    def test_whole_band_without_paint(self):
        # no shadow map: all the floor within the thing's height of it is dropped and marked lost; the floor beyond is owned
        out = run(self.tmp, """o, _, _, _, lost = own(None); far = d > 1.3
print(json.dumps([float((o[band] >= 0).mean()), float(lost[band].mean()), float((o[far] >= 0).mean()), O.shade_bands(C, lifted)[0][2]]))""")
        self.assertEqual(out[0], 0); self.assertEqual(out[1], 1); self.assertGreater(out[2], 0.85); self.assertEqual(out[3], 1.0)

    def test_only_the_painted_shadow_is_dropped(self):
        # a dark patch against the +x side is the thing's shadow; one of the same size apart from it is the floor's own
        # pattern; the rest of the band is ordinary floor and keeps its paint
        out = run(self.tmp, """joined = lambda Q: (Q[..., 0] > 3.5) & (Q[..., 0] < 4.1) & (abs(Q[..., 2] - 3) < 0.4)
apart = lambda Q: (Q[..., 0] > 1.7) & (Q[..., 0] < 2.1) & (abs(Q[..., 2] - 3) < 0.2)
sh = O.shade(C, views, rank, lifted, paint(lambda Q: joined(Q) | apart(Q))); o, _, _, _, lost = own(sh)
plain = band & ~joined(P) & ~apart(P) & (d > 0.35)
core = joined(P) & (P[..., 0] > 3.6) & (P[..., 0] < 4.0) & (abs(P[..., 2] - 3) < 0.3)
print(json.dumps([float((o[core] >= 0).mean()), float(lost[core].mean()), float((o[apart(P)] >= 0).mean()), float((o[plain] >= 0).mean()),
                  bool(sh(np.array([[0.2, 0.0, 0.2]]))[0])]))""")
        self.assertEqual(out[0], 0); self.assertEqual(out[1], 1)        # the shadow has no paint and is marked for mending
        self.assertGreater(out[2], 0.9); self.assertGreater(out[3], 0.9)   # the dark patch apart from the thing and the plain floor keep theirs
        self.assertFalse(out[4])                                        # nothing outside the band is ever shadow

    def test_evenly_lit_floor_keeps_its_paint(self):
        out = run(self.tmp, "sh = O.shade(C, views, rank, lifted, paint(lambda Q: np.zeros(Q.shape[:-1], bool))); o = own(sh)[0]; m = band & (d > 0.35)\nprint(json.dumps(float((o[m] >= 0).mean())))")
        self.assertGreater(out, 0.95)

    def test_band_round_a_moving_thing(self):
        # a view is never the room's paint within 16 px of the thing's outline, the last choice out to 40 px, and with
        # WORLD_GAPS=0 no choice anywhere in the 40 px
        code = """v = views[0]; cam, t, s0, edge, halo, far, near = v; one = [v]; rk = O.rank(np.bincount(s0.ravel(), minlength=NS)[:, None].astype(float))
P0 = traced[0][1]; fl = s0 == 1; none = lambda Q: np.zeros(Q.shape[:-1], bool)
o = O.owners(C, one, rk, P0, s0, None, lifted, none)[0]
both = O.owners(C, views, rank, P0, s0, None, lifted, none)[0]
ring = fl & far & ~near
print(json.dumps([float((o[fl & near] >= 0).mean()), float((o[ring] >= 0).mean()), float((o[fl & ~far] >= 0).mean()), float((both[ring] == 0).mean()), int(ring.sum()), O.GAPS]))"""
        new = run(self.tmp, code); old = run(self.tmp, code, WORLD_GAPS="0")
        self.assertTrue(new[5]); self.assertFalse(old[5]); self.assertGreater(new[4], 1000)
        self.assertEqual(new[0], 0); self.assertGreater(new[1], 0.95); self.assertGreater(new[2], 0.95)
        self.assertLess(new[3], 0.2)            # with the other views to choose from, the band is taken from them instead
        self.assertEqual(old[0], 0); self.assertEqual(old[1], 0); self.assertGreater(old[2], 0.95)

    def test_old_rule_when_switched_off(self):
        self.assertIsNone(run(self.tmp, "print(json.dumps(O.shade(C, views, rank, lifted, paint(lambda Q: np.zeros(Q.shape[:-1], bool)))))", WORLD_GAPS="0"))

    def test_hidden_floor_has_no_owner(self):
        # with one view only, the floor behind the crate is in its picture but not seen
        out = run(self.tmp, """one = [views[0]]; rk = O.rank(np.bincount(views[0][2].ravel(), minlength=NS)[:, None].astype(float)); eye = np.array(B.CAMS['0'][0])
away = np.array([3, 0, 3]) - eye; away[1] = 0; away /= np.linalg.norm(away); Q = (np.array([3, 0, 3]) + away * 0.8)[None]
o = O.owners(C, one, rk, Q, np.ones(1, int), None, (), None)[0]; print(json.dumps([int(o[0]), bool(B.visible(eye, Q)[0])]))""")
        self.assertEqual(out, [-1, False])

    def test_training_views_are_added_where_few_look(self):
        # one training view leaves most of what a walker comes close to unseen: views are added, each one of the
        # walker's own downward looks, none twice, at most COVER_MOST. With the walker's views as the training set: none
        out = run(self.tmp, """walk, Q, qs, _ = O.walker(C, lifted); one = [B.CAMS['0']]
add = O.cover(C, one, lifted); short = lambda cams: float(np.maximum(O.COVER_WANT - sum(O.sees(B, c, Q).astype(float) for c in cams), 0).sum())      # looks missing
print(json.dumps([len(add), len({(tuple(e), tuple(l)) for e, l in add}), all((tuple(e), tuple(l)) in set(walk[1::2]) for e, l in add),
                  short(one), short(one + add), len(O.cover(C, walk, lifted)), O.COVER_MOST, len(Q), int((qs == 4).sum()), bool((Q[:, [0, 2]] >= 0).all() and (Q[:, [0, 2]] <= 6).all())]))""")
        self.assertEqual(out[0], out[6]); self.assertEqual(out[1], out[0]); self.assertTrue(out[2])
        self.assertLess(out[4], 0.8 * out[3]); self.assertEqual(out[5], 0)
        self.assertGreater(out[7], 1000); self.assertEqual(out[8], 0); self.assertTrue(out[9])         # the walker's points: no ceiling, inside the room
        self.assertEqual(run(self.tmp, "print(json.dumps(len(O.cover(C, [B.CAMS['0']], lifted))))", WORLD_GAPS="0"), 0)

    def test_downward_views_go_where_the_ring_leaves_no_paint(self):
        # half a ring leaves floor behind the crate and at its own feet unpainted: the planned views stand inside the
        # room at the downward views' height, look down, and leave less of the walker's points without an owner
        out = run(self.tmp, """half = views[:4]; rk = O.rank(np.stack([np.bincount(v[2].ravel(), minlength=NS) for v in half], 1).astype(float))
plan = O.lookdown(C, half, rk, 3, lifted); _, Q, qs, qn = O.walker(C, lifted, True); none = lambda Z: np.zeros(Z.shape[:-1], bool)
def unowned(cams):
    vs = list(half)
    for cam in cams: t, P, sid = B.trace(cam, C.PRIM_OBJ); vs.append(O.view(cam, t, sid, np.zeros(sid.shape, bool), lifted))
    r = O.rank(np.stack([np.bincount(v[2].ravel(), minlength=NS) for v in vs], 1).astype(float))
    return int((O.owners(C, vs, r, Q, qs, qn, lifted, none)[0] < 0).sum())
print(json.dumps([[list(e) + list(l) for e, l in plan], unowned([]), unowned(plan), min(C.RH - 0.3, C.EYE + 0.6), len(O.lookdown(C, half, rk, 0, lifted))]))""")
        plan, before, after, h, none = out
        self.assertEqual(len(plan), 3); self.assertEqual(none, 0)
        for x, y, z, lx, ly, lz in plan: self.assertTrue(0.5 < x < 5.5 and 0.5 < z < 5.5); self.assertAlmostEqual(y, h); self.assertLess(ly, y - 0.5)
        self.assertGreater(before, 100); self.assertLess(after, 0.5 * before)

if __name__ == "__main__": unittest.main()
