"""A flat mirror that is a real mirror. The discs of the mirror's face and of the wall behind it are taken out, and a mirror image
of everything the mirror can show to a walker is hung behind the wall: it has true depth and moves correctly with the eye, in the unchanged viewer.
Only what some eye in the room can see THROUGH the mirror's rectangle is copied (thinned, each disc enlarged to close the gaps), so a mirror costs
a few hundred thousand discs, not a second level.
    extra, keep, info = mirrors(discs, P, N, cls, rgb, opac, keep, extra, cap=450_000)"""
import numpy as np
MIRROR = 6
def mirrors(discs, P, N, cls, rgb, opac, keep, extra=None, cap=600_000, dim=0.88):
    out = {k: [] for k in ("xyz", "sigma", "rgb", "opacity", "normal")}; info = []; dm = np.asarray(discs["mesh"]); n = len(P); sig = np.asarray(discs["sigma"], np.float32); thick = np.asarray(discs["thick"], np.float32)
    lo, hi = np.percentile(P, 0.5, axis=0), np.percentile(P, 99.5, axis=0); rng = np.random.default_rng(11)
    gx, gz = np.meshgrid(np.linspace(lo[0] + 0.4, hi[0] - 0.4, 5), np.linspace(lo[2] + 0.4, hi[2] - 0.4, 5)); eyes = np.concatenate([np.stack([gx.ravel(), np.full(gx.size, h), gz.ravel()], 1) for h in (1.0, 1.65)]).astype(np.float32)
    for k in np.unique(dm[cls == MIRROR]):
        sel = np.nonzero((cls == MIRROR) & (dm == k))[0]
        if len(sel) < 50: continue
        ax = np.abs(N[sel]).argmax(1); sgn = np.sign(N[sel][np.arange(len(sel)), ax]); key = ax * 2 + (sgn < 0); b = np.bincount(key, minlength=6).argmax(); nm = np.zeros(3, np.float32); nm[b // 2] = -1.0 if b % 2 else 1.0     # the face: the facing most of its discs have
        face = sel[key == b]; p0 = np.median(P[face], 0); t1 = np.float32([0, 1, 0]) if abs(nm[1]) < 0.5 else np.float32([1, 0, 0]); t2 = np.cross(nm, t1); q1, q2 = (P[face] - p0) @ t1, (P[face] - p0) @ t2
        h1, h2 = float(np.ptp(q1)) / 2, float(np.ptp(q2)) / 2; p0 = p0 + t1 * (q1.max() + q1.min()) / 2 + t2 * (q2.max() + q2.min()) / 2
        if 4 * h1 * h2 < 0.04: continue
        # the hole: the face itself and whatever lies behind it (the picture's back, the wall, the wall's solid layer)
        s = (P - p0) @ nm; a1, a2 = (P - p0) @ t1, (P - p0) @ t2; hole = (np.abs(a1) < h1) & (np.abs(a2) < h2) & (s < 0.004) & (s > -0.8); hole |= (cls == MIRROR) & (dm == k); keep = keep & ~hole
        # what the mirror can show: seen from a mirrored eye through the rectangle, from its front
        cand = np.nonzero(keep & (s > 0.01) & (thick < 0.002))[0]; cand = cand[rng.random(len(cand)) < 0.5]; seen = np.zeros(len(cand), bool); Pc, Nc = P[cand], N[cand]
        for e in eyes:
            if (e - p0) @ nm < 0.2: continue
            e2 = e - 2 * ((e - p0) @ nm) * nm; d = Pc - e2; den = d @ nm; t = ((p0 - e2) @ nm) / np.where(np.abs(den) < 1e-6, 1e-6, den); Xp = e2 + d * t[:, None] - p0
            seen |= (t > 0) & (t < 1) & (np.abs(Xp @ t1) < h1) & (np.abs(Xp @ t2) < h2) & ((Nc * -d).sum(1) > 0)
        cand = cand[seen]; T = 2.0
        if len(cand) > cap: T *= len(cand) / cap; cand = cand[rng.random(len(cand)) < cap / len(cand)]
        refl = lambda v: v - 2 * (v @ nm)[:, None] * nm; g = np.sqrt(T) * 1.5
        out["xyz"].append(P[cand] - 2 * s[cand, None] * nm); out["normal"].append(refl(N[cand])); out["rgb"].append(rgb[cand].astype(np.float32) / 255 * dim * np.float32([0.97, 1.0, 1.0]))
        out["sigma"].append(np.stack([sig[cand] * g, sig[cand] * g, thick[cand]], 1)); out["opacity"].append(np.minimum(opac[cand], 0.97))
        if extra is not None:                                   # flames and glows are mirrored too
            se = (extra["xyz"] - p0) @ nm; m = se > 0.01
            out["xyz"].append(extra["xyz"][m] - 2 * se[m, None] * nm); out["normal"].append(refl(extra["normal"][m])); out["rgb"].append(extra["rgb"][m] * dim); out["sigma"].append(extra["sigma"][m]); out["opacity"].append(extra["opacity"][m])
        info.append(dict(mesh=int(k), size=[round(2 * h1, 2), round(2 * h2, 2)], normal=nm.tolist(), hole_discs=int(hole.sum()), image_discs=int(len(cand)), thin=round(float(T), 2)))
    if not info: return None, keep, info
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}, keep, info
