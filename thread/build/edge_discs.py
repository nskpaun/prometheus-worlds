"""Discs on a sculpted mesh without fuzz. Darts thrown evenly, each disc 0.75 of its spacing in radius, hang over a sharp
edge, a thin leg, a spoke or a wire by up to twice their radius, and paint taken from the nearest points whichever way
they face gives a table top's rim the colour of its dark underside. Both are fixed here:
  fit(V, F, pts, nrm, sig)            -> pts, nrm, sig, parent: each disc shrunk until its rim lies on the surface; the room it gave up is
                                         filled with small discs on the same face, so nothing gets holes
  colour(pts, nrm, xyz, n_s, rgb)     -> paint from the nearest points THAT FACE THE SAME WAY
thread/.venv, CPU."""
import numpy as np
from scipy.spatial import cKDTree
def scene_of(V, F):
    import open3d as o3d
    s = o3d.t.geometry.RaycastingScene(); s.add_triangles(o3d.core.Tensor(np.ascontiguousarray(V, np.float32)), o3d.core.Tensor(np.ascontiguousarray(F, np.uint32))); return s
def _closest(sc, X):
    import open3d as o3d
    return sc.compute_closest_points(o3d.core.Tensor(np.ascontiguousarray(X, np.float32)))["points"].numpy()
def _tangent(n):
    a = np.cross(n, [0, 1.0, 0]); bad = np.linalg.norm(a, axis=1) < 1e-3; a[bad] = np.cross(n[bad], [1.0, 0, 0]); a /= np.linalg.norm(a, axis=1, keepdims=True); return a, np.cross(n, a)
def overhang(sc, pts, nrm, r, k=6):
    """Does a ring of radius r round each point, in its own plane, leave the surface? -> bool"""
    a, b = _tangent(nrm); out = np.zeros(len(pts), bool); r = np.broadcast_to(np.asarray(r, float), (len(pts),))
    for i in range(k):
        t = 2 * np.pi * (i + 0.5) / k; q = pts + (a * np.cos(t) + b * np.sin(t)) * r[:, None]; out |= np.linalg.norm(_closest(sc, q) - q, axis=1) > 0.35 * r
    return out
def fit(V, F, pts, nrm, sig, reach=1.7, levels=(0.6, 0.36), seed=0, sc=None):
    """Discs (centres pts, normals nrm, radii sig) made to stay on the surface. -> pts, nrm, sig, parent (index of the disc each one came from)"""
    sc = sc or scene_of(V, F); sig = np.asarray(sig, np.float64).copy(); f = np.ones(len(pts)); bad = overhang(sc, pts, nrm, reach * sig)
    for lv in levels:
        f[bad] = lv; i = np.nonzero(bad)[0]
        if not len(i): break
        bad[i] = overhang(sc, pts[i], nrm[i], reach * sig[i] * lv)
    i = np.nonzero(f < 1)[0]; rng = np.random.default_rng(seed); P2, N2, S2, par = [pts], [nrm], [sig * f], [np.arange(len(pts))]
    if len(i):          # the room given up: small discs round the shrunk one, kept only where they land on the surface next to it
        a, b = _tangent(nrm[i]); m = np.where(f[i] > 0.5, 3, 7)
        for j in range(7):
            s = i[m > j]; 
            if not len(s): continue
            aa, bb = a[m > j], b[m > j]; t = rng.uniform(0, 2 * np.pi, len(s)); rr = sig[s] * rng.uniform(0.55, 1.15, len(s)); q = pts[s] + (aa * np.cos(t)[:, None] + bb * np.sin(t)[:, None]) * rr[:, None]
            c = _closest(sc, q); ok = np.linalg.norm(c - q, axis=1) < 0.25 * sig[s]; ok[ok] &= ~overhang(sc, c[ok], nrm[s][ok], reach * sig[s][ok] * f[s][ok])
            P2.append(c[ok]); N2.append(nrm[s][ok]); S2.append(sig[s][ok] * f[s][ok]); par.append(s[ok])
    return np.concatenate(P2), np.concatenate(N2), np.concatenate(S2), np.concatenate(par)
def colour(pts, nrm, xyz, n_s, rgb, tree=None, k=6):
    """Paint for disc centres from an object's painted points (xyz, normals n_s, rgb): the nearest that face the same way, nearest first."""
    tree = tree or cKDTree(xyz); dd, jj = tree.query(pts, k=k, workers=4); agree = np.clip((n_s[jj] * nrm[:, None, :]).sum(-1), 0, 1) ** 2 + 1e-3
    w = agree / (dd + 2e-4) ** 2; return (rgb[jj] * w[..., None]).sum(1) / w.sum(1)[:, None]
def _poisson(P, r, rounds=3, seed=0):
    rng = np.random.default_rng(seed); n = len(P); cell = np.floor(P / r).astype(np.int64); cell -= cell.min(0); ph = (cell[:, 0] % 2) * 4 + (cell[:, 1] % 2) * 2 + cell[:, 2] % 2
    key = (cell[:, 0] * (cell[:, 1].max() + 1) + cell[:, 1]) * (cell[:, 2].max() + 1) + cell[:, 2]; alive = np.ones(n, bool); kept = np.zeros(n, bool)
    for rd in range(rounds):
        order = rng.permutation(n)
        for p in range(8):
            idx = order[(ph[order] == p) & alive[order]]
            if not len(idx): continue
            if kept.any():
                d, _ = cKDTree(P[kept]).query(P[idx], k=1, distance_upper_bound=r, workers=4); alive[idx[np.isfinite(d)]] = False; idx = idx[~np.isfinite(d)]
            _, first = np.unique(key[idx], return_index=True); kept[idx[first]] = True; alive[idx[first]] = False
    return np.nonzero(kept)[0]
