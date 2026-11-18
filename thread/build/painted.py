"""The painted sculpted kinds of a built world, read by the assembler. thread/.venv, CPU.
  paint(world, kind)                 -> dict(P, N, rgb (0..1), how, mat, meta): the painted points on the kind's own mesh (geom.load's frame)
  placed(world, kind, V)             -> dict(xyz, nrm, rgb (0..255), mat, V0, F0): the same points on one placed copy (V = the copy's vertices in
                                        room metres, same vertex order as the kind's mesh, as shapes/placed/NNN.npz has them)
  discs(piece, pts, nrm, sig)        -> pts, nrm, sig, rgb (0..255), mat: dart-thrown discs fitted to the mesh's edges and painted by facing
Material classes (mat, uint8): 0 plain, 1 glass, 2 metal (bright bare metal), 3 glow (flame, bulb: gives light), 4 gloss (car paint, polished wood, leather)."""
import json
import numpy as np
import geom, edge_discs, paint as OP
from scipy.spatial import cKDTree
MATS = OP.MATS
def paint(world, kind):
    d = OP.wd(world, kind); S = np.load(f"{d}/samples.npz"); T = np.load(f"{d}/tex_final.npz")
    return dict(P=S["P"].astype(np.float64), N=S["N"].astype(np.float64), face=S["face"], bary=S["bary"], rgb=np.clip(T["rgb"].astype(np.float32), 0, 1), how=T["how"], mat=T["mat"], meta=json.load(open(f"{d}/meta.json")))
def placed(world, kind, V):
    p = paint(world, kind); Kd = OP.kinds(world)[kind]; V0, F0 = geom.load(Kd["shapes"], kind); V = np.asarray(V, np.float64)
    A = np.linalg.lstsq(np.c_[V0, np.ones(len(V0))], V, rcond=None)[0]; Am = A[:3].T
    xyz = (V[F0[p["face"]]] * p["bary"][:, :, None].astype(np.float64)).sum(1); n = p["N"] @ np.linalg.inv(Am); n /= np.linalg.norm(n, axis=1, keepdims=True)
    if np.linalg.det(Am) < 0: n = -n
    return dict(xyz=xyz, nrm=n, rgb=(p["rgb"] * 255).astype(np.float32), mat=p["mat"], V0=V0, F0=F0, how=p["how"])
def discs(piece, pts, nrm, sig, fit=True):
    """piece: a dict with V, F (the placed mesh), xyz, nrm, rgb, mat (from placed()). pts, nrm, sig: the discs the assembler threw on it.
    -> pts, nrm, sig, rgb (0..255), mat. Colour by the nearest painted points that face the disc's way; discs near an edge shrunk and the room refilled."""
    if "o_tree" not in piece: piece["o_tree"] = cKDTree(piece["xyz"]); piece["o_scene"] = edge_discs.scene_of(piece["V"], piece["F"])
    if fit: pts, nrm, sig, _ = edge_discs.fit(piece["V"], piece["F"], pts, nrm, sig, sc=piece["o_scene"])
    col = edge_discs.colour(pts, nrm, piece["xyz"], piece["nrm"], piece["rgb"], tree=piece["o_tree"]); mat = piece["mat"][piece["o_tree"].query(pts, workers=4)[1]]
    return pts, nrm, sig, col, mat
