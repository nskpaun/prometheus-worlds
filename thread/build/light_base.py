"""The ground the light stands on (light.py is what the assembler calls): colour spaces, the words that make a thing a
lamp, a fire or a screen, the sky and sun of each time of day, and the tracing machinery. CPU only (Embree through open3d).
  receivers  discs are gathered into cells of a few cm x 6 facings; each cell's front-most disc is a receiver (cells under a surface borrow)
  direct     every emitter is cut into lights (a point-like centre + a pool of points on it); per receiver K shadow rays to random pool points
             give a VISIBILITY per light (soft cast shadows from every mesh; a lamp's own mesh does not block it). The cosine and distance
             terms are then worked out per DISC with its own normal, visibility read between cells: shadow edges and round things stay smooth.
  bounce     cosine rays from each receiver; where they land they read direct light x paint from a coarse copy of the level; the same hits are
             reused for 3 bounces. The same rays give openness (ambient occlusion) for a small cool fill.
  meshes     [{"V": (v,3), "F": (f,3), "name": str, "painted": bool}] everything that can block light. painted=True means its discs carry
             painted studio light (object paint): the smooth part is estimated and divided out (delight).
  emitters   see find_emitters: {"type": "mesh"|"rect"|"point"|"sun", ...}
Also the splat file's writer (write_ply) and the turn that lays a disc flat on its surface (quat_z_to)."""
import numpy as np
YW = np.array([0.2126, 0.7152, 0.0722], np.float32); C0 = 0.28209479177
PLY14 = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
AXES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float32)

def s2l(c): c = np.asarray(c, np.float32) / 255; return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)
def l2s(c): c = np.clip(c, 0, 1); return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055).astype(np.float32)
# ================================================================ the sentence: time of day, and what gives light
TIMES = {"night": ("night", "midnight", "moonlit", "moonlight", "moon", "starlit", "stars", "starry", "after dark", "candlelit"),
         "dusk": ("dusk", "sunset", "twilight", "evening", "golden hour", "dawn", "sunrise", "daybreak"),
         "day": ("day", "noon", "morning", "afternoon", "sunny", "sunlit", "bright", "summer", "daylight")}
LAMP = ("lantern", "lamp", "light", "candle", "candelabra", "chandelier", "bulb", "torch", "sconce", "neon", "glowing", "orb", "crystal ball", "streetlight", "spotlight")
FIRE = ("fireplace", "hearth", "stove", "brazier", "forge", "furnace", "campfire", "bonfire", "kiln", "fire pit", "firepit")
SCREEN = ("screen", "monitor", "tv", "television", "console", "hologram", "display panel", "control panel")
COOL = ("neon", "orb", "crystal", "hologram", "bioluminescent", "glowing", "magic")
NOT_LAMP = ("lighthouse", "light switch", "lightning")
C_LAMP, C_CANDLE, C_FIRE, C_COOL, C_PANEL = np.array([1.0, 0.70, 0.40]), np.array([1.0, 0.62, 0.30]), np.array([1.0, 0.45, 0.16]), np.array([0.55, 0.78, 1.0]), np.array([1.0, 0.96, 0.90])
SKY = {"night": (np.array([0.45, 0.62, 1.0]), 0.22), "dusk": (np.array([1.0, 0.62, 0.42]), 1.2), "day": (np.array([0.82, 0.92, 1.0]), 3.0)}      # colour, flux a square metre of opening (a lantern = 1)
SUN = {"night": (np.array([0.55, 0.70, 1.0]), 0.5, 42.0), "dusk": (np.array([1.0, 0.55, 0.25]), 5.0, 14.0), "day": (np.array([1.0, 0.95, 0.85]), 14.0, 50.0)}  # colour, strength, height in degrees
def mood_of(sentence, mood=None):
    """{"time": "night" | "dusk" | "day"} from the layout's mood field if it has one, else from the sentence's words (day if it says nothing)."""
    t = (mood or {}).get("time"); s = " " + " ".join(str(sentence or "").lower().replace(",", " ").replace(".", " ").split()) + " "
    if t not in TIMES:
        t = next((k for k in ("night", "dusk", "day") if any(f" {w} " in s or f" {w}:" in s for w in TIMES[k])), "day")
    return {"time": t}

def _has(words, vocab): return any((w in words) if " " in w else (w in words.split()) for w in vocab)

def find_emitters(objects, openings=None, sentence="", room=None, meshes=None, mood=None):
    """objects: [{"name", "kind", "look", "mesh": index into meshes}] (the layout's things). openings: [{"kind": "window"|"skylight"|"door"|...,
    "c": centre, "u": half-vector across, "v": half-vector up (room metres), "n": unit normal INTO the room}] from the shell's geometry.
    -> emitters. If nothing in the level gives light (no lamp, no fire, no opening) a closed room gets plain ceiling panels on a grid."""
    mood = mood or mood_of(sentence); tm = mood["time"]; em = []; rng = np.random.default_rng(0)
    for k, o in enumerate(objects or []):
        words = " ".join(str(o.get(q) or "") for q in ("name", "kind", "look")).lower().replace("_", " ").replace("-", " ")
        if _has(words, NOT_LAMP): continue
        mi = o.get("mesh", k); V = np.asarray(meshes[mi]["V"], float); lo, hi = V.min(0), V.max(0); ext = hi - lo; c = (lo + hi) / 2
        if _has(words, FIRE) or ("fire" in words.split() and not _has(words, LAMP)):
            ax = 0 if ext[0] < ext[2] else 2; other = 2 - ax; sign = 1.0
            if room is not None: sign = 1.0 if c[ax] < room[ax] / 2 else -1.0            # the opening faces the room
            n = np.zeros(3); n[ax] = sign; cc = c.copy(); cc[ax] = (hi[ax] if sign > 0 else lo[ax]) + sign * 0.04; cc[1] = lo[1] + 0.34 * ext[1]
            u = np.zeros(3); u[other] = 0.26 * ext[other]; v = np.array([0, 0.22 * ext[1], 0.0])
            em.append(dict(type="rect", cls="fire", name=o.get("name"), c=cc, u=u, v=v, n=n, col=C_FIRE, flux=float(0.6 + 0.8 * np.sqrt(4 * np.linalg.norm(u) * np.linalg.norm(v))), mesh=mi))
        elif _has(words, SCREEN):
            ax = int(np.argmin(ext)); n = np.zeros(3); n[ax] = 1.0
            if room is not None: n[ax] = 1.0 if c[ax] < room[ax] / 2 else -1.0
            oth = [a for a in range(3) if a != ax]; u = np.zeros(3); u[oth[0]] = 0.45 * ext[oth[0]]; v = np.zeros(3); v[oth[1]] = 0.45 * ext[oth[1]]; cc = c + n * (ext[ax] / 2 + 0.03)
            em.append(dict(type="rect", cls="screen", name=o.get("name"), c=cc, u=u, v=v, n=n, col=C_COOL, flux=float(0.25 + 0.6 * np.sqrt(4 * np.linalg.norm(u) * np.linalg.norm(v))), mesh=mi))
        elif _has(words, LAMP):
            candle = "candle" in words or "candelabra" in words; size = float(np.linalg.norm(ext))
            col = C_COOL if _has(words, COOL) else C_CANDLE if candle else C_LAMP
            em.append(dict(type="mesh", cls="candle" if candle else "lamp", name=o.get("name"), mesh=mi, col=col, flux=float((0.5 if candle else 2.4) + (0.8 if candle else 3.2) * min(size, 1.5))))      # shines all round: 4x the flux of a flat source for the same light on a facing surface
    for op in openings or []:
        u, v, n = (np.asarray(op[q], float) for q in ("u", "v", "n")); a = 4 * np.linalg.norm(u) * np.linalg.norm(v); col, per = SKY[tm]
        em.append(dict(type="rect", cls="opening", name=op.get("kind", "opening"), c=np.asarray(op["c"], float) + 0.02 * n, u=u, v=v, n=n, col=col, flux=float(per * a), mesh=-1))
    ops = [e for e in em if e["cls"] == "opening"]
    if ops:                                                  # sun or moon: comes in through the biggest opening, slanted, height by the time of day
        big = max(ops, key=lambda e: e["flux"]); col, power, elev = SUN[tm]; n = big["n"]
        if abs(n[1]) > 0.7: az = np.array([0.5, 0, 0.3])                                 # a skylight: steep, a little off straight down
        else: h = np.array([n[0], 0, n[2]]); h /= np.linalg.norm(h); az = h * np.cos(np.radians(28)) + np.cross([0, 1.0, 0], h) * np.sin(np.radians(28))
        el = np.radians(max(elev, 62.0) if abs(n[1]) > 0.7 else elev); to_sun = -(az / np.linalg.norm(az)) * np.cos(el) + np.array([0, np.sin(el), 0])
        em.append(dict(type="sun", cls="sun", name="moon" if tm == "night" else "sun", dir=to_sun / np.linalg.norm(to_sun), col=col, flux=float(power), openings=[dict(c=e["c"], u=e["u"], v=e["v"], n=e["n"]) for e in ops], mesh=-1))
    if not em and room is not None:                          # nothing gives light: recessed downlights in the ceiling, about one a ceiling height (pools on the floor, scallops on the walls)
        R = np.asarray(room, float); step = 1.3 * R[1]; nx, nz = max(1, int(round(R[0] / step))), max(1, int(round(R[2] / step)))
        for i in range(nx):
            for j in range(nz):
                c = np.array([(i + 0.5) * R[0] / nx, R[1] - 0.02, (j + 0.5) * R[2] / nz]); s = 0.09
                em.append(dict(type="rect", cls="downlight", name="downlight", c=c, u=np.array([s, 0, 0.0]), v=np.array([0, 0, s]), n=np.array([0, -1.0, 0]), col=C_PANEL, flux=1.0, beam=7.0, mesh=-1, generated=True))
    return em

# ================================================================ geometry
class Scene:
    def __init__(self, meshes):
        import open3d as o3d
        self.o3d = o3d; self.sc = o3d.t.geometry.RaycastingScene(); self.lut = np.full(len(meshes) + 2, -1, np.int64); self.tris = 0
        for k, m in enumerate(meshes):
            g = self.sc.add_triangles(o3d.core.Tensor(np.ascontiguousarray(m["V"], np.float32)), o3d.core.Tensor(np.ascontiguousarray(m["F"], np.uint32))); self.lut[g] = k; self.tris += len(m["F"])
        self.n_rays = 0
    def cast(self, o, d, normals=False, chunk=4_000_000):
        """-> t (inf for none), mesh index (-1), [unit normal of the hit triangle]"""
        o = np.ascontiguousarray(o, np.float32).reshape(-1, 3); d = np.ascontiguousarray(d, np.float32).reshape(-1, 3); n = len(o); t = np.empty(n, np.float32); mi = np.empty(n, np.int64); N = np.empty((n, 3), np.float32) if normals else None
        for a in range(0, n, chunk):
            r = self.sc.cast_rays(self.o3d.core.Tensor(np.concatenate([o[a:a + chunk], d[a:a + chunk]], 1))); tt = r["t_hit"].numpy(); t[a:a + chunk] = tt
            mi[a:a + chunk] = np.where(np.isfinite(tt), self.lut[np.minimum(r["geometry_ids"].numpy().astype(np.int64), len(self.lut) - 1)], -1)
            if normals: N[a:a + chunk] = r["primitive_normals"].numpy()
        self.n_rays += n; return (t, mi, N) if normals else (t, mi)
    def nearest(self, p, chunk=2_000_000):
        p = np.ascontiguousarray(p, np.float32); mi = np.empty(len(p), np.int64); d = np.empty(len(p), np.float32)
        for a in range(0, len(p), chunk):
            r = self.sc.compute_closest_points(self.o3d.core.Tensor(p[a:a + chunk])); mi[a:a + chunk] = self.lut[np.minimum(r["geometry_ids"].numpy().astype(np.int64), len(self.lut) - 1)]
            d[a:a + chunk] = np.linalg.norm(r["points"].numpy() - p[a:a + chunk], axis=1)
        return mi, d

def hemi(n, count, rng):
    """count cosine-weighted directions about each normal n (m, 3) -> (m, count, 3), stratified in the first number."""
    a = np.where(np.abs(n[:, :1]) < 0.9, np.float32([[1.0, 0, 0]]), np.float32([[0, 1.0, 0]])); t = np.cross(n, a); t /= np.linalg.norm(t, axis=1, keepdims=True); b = np.cross(n, t)
    u1 = ((rng.random((len(n), count), np.float32) + np.arange(count, dtype=np.float32)) / count); u2 = rng.random((len(n), count), np.float32); r = np.sqrt(u1); ph = 2 * np.pi * u2
    return (r * np.cos(ph))[..., None] * t[:, None] + (r * np.sin(ph))[..., None] * b[:, None] + np.sqrt(1 - u1)[..., None] * n[:, None]

# ================================================================ cells
class Cells:
    """Discs gathered into cells (size c) x 6 facings. key -> row by a sorted table; values are read back between cells (8 corners x up to 3 facings)."""
    def __init__(self, P, N, c):
        self.c = c; self.lo = P.min(0) - 2.37 * c; ijk = np.floor((P - self.lo) / c).astype(np.int64); self.dims = ijk.max(0) + 3
        b = self.bucket(N); key = self.key(ijk, b); self.keys, inv = np.unique(key, return_inverse=True); self.n = len(self.keys); self.inv = inv
        # the front-most disc of each cell along its facing
        front = (P * AXES[b]).sum(1); order = np.lexsort((-front, inv)); first = np.ones(len(order), bool); first[1:] = inv[order][1:] != inv[order][:-1]; self.rep = order[first]
        self.b = b[self.rep]; self.count = np.bincount(inv, minlength=self.n).astype(np.float32)
    @staticmethod
    def bucket(N): ax = np.abs(N).argmax(1); return (2 * ax + (N[np.arange(len(N)), ax] < 0)).astype(np.int64)
    def key(self, ijk, b): return ((ijk[..., 0] * self.dims[1] + ijk[..., 1]) * self.dims[2] + ijk[..., 2]) * 6 + b
    def find(self, key):
        j = np.searchsorted(self.keys, key); j[j >= self.n] = self.n - 1; return j, self.keys[j] == key
    def mean(self, vals):
        """Per-disc values (n, k) -> per-cell mean (cells, k)."""
        out = np.zeros((self.n, vals.shape[1]), np.float32)
        for q in range(vals.shape[1]): out[:, q] = np.bincount(self.inv, vals[:, q], self.n) / self.count
        return out
    def read(self, P, N, vals, ok, flat=None):
        """Values (cells, k) read at points P with normals N: trilinear between cells, blended over the facings the normal leans to; cells that are
        not ok (under a surface) are skipped. -> (m, k), found (m,)"""
        f = (P - self.lo) / self.c - 0.5; i0 = np.floor(f).astype(np.int64); fr = (f - i0).astype(np.float32); out = np.zeros((len(P), vals.shape[1]), np.float32); wsum = np.zeros(len(P), np.float32)
        n2 = N * N
        for ax in range(3):
            wb = n2[:, ax]; sel = np.nonzero(wb > 0.03)[0]
            if not len(sel): continue
            b = 2 * ax + (N[sel, ax] < 0)
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        w = wb[sel] * np.where(dx, fr[sel, 0], 1 - fr[sel, 0]) * np.where(dy, fr[sel, 1], 1 - fr[sel, 1]) * np.where(dz, fr[sel, 2], 1 - fr[sel, 2])
                        j, hit = self.find(self.key(i0[sel] + [dx, dy, dz], b)); hit &= ok[j]; w = w * hit
                        out[sel] += w[:, None] * vals[j]; wsum[sel] += w
        good = wsum > 1e-4; out[good] /= wsum[good, None]; return out, good
    def blur(self, vals, ok):
        """One pass of neighbour averaging (27 cells, same facing)."""
        ijk = self.keys // 6; b = self.keys % 6; k_ = ijk % self.dims[2]; j_ = (ijk // self.dims[2]) % self.dims[1]; i_ = ijk // (self.dims[1] * self.dims[2]); I = np.stack([i_, j_, k_], 1)
        acc = np.zeros_like(vals); w = np.zeros(self.n, np.float32)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    wt = 0.5 ** (abs(dx) + abs(dy) + abs(dz)); j, hit = self.find(self.key(I + [dx, dy, dz], b)); hit &= ok[j]; acc += (wt * hit)[:, None] * vals[j]; w += wt * hit
        return acc / np.maximum(w, 1e-6)[:, None]

# ================================================================ lights
def make_lights(em, meshes, rng, max_lights=40):
    """Emitters -> lights: each {c, n (or None), r0 (softness, m), pool (m, 3) points light leaves from, col (3,) = flux x colour, mesh, kind, e (emitter index)}."""
    L = []
    for ei, e in enumerate(em):
        col = np.asarray(e["col"], np.float32); col = col / float(col @ YW)
        if e["type"] == "sun":
            L.append(dict(kind="sun", dir=np.asarray(e["dir"], np.float32), col=col * e["flux"], openings=e.get("openings"), mesh=-1, e=ei, spread=np.radians(e.get("spread_deg", 2.0)))); continue
        if e["type"] == "mesh":
            import trimesh
            m = meshes[e["mesh"]]; tm = trimesh.Trimesh(m["V"], m["F"], process=False); P, fi = trimesh.sample.sample_surface(tm, 96, seed=1); c = np.asarray(m["V"], float).mean(0) * 0 + (np.asarray(m["V"]).min(0) + np.asarray(m["V"]).max(0)) / 2
            ext = np.asarray(m["V"]).max(0) - np.asarray(m["V"]).min(0)
            L.append(dict(kind="iso", c=c.astype(np.float32), n=None, r0=float(max(0.5 * np.linalg.norm(ext) * 0.5, 0.05)), pool=(P + 0.01 * tm.face_normals[fi]).astype(np.float32), col=col * e["flux"], mesh=e["mesh"], e=ei)); continue
        if e["type"] == "point":
            r = float(e.get("r", 0.05)); d = rng.normal(size=(64, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
            L.append(dict(kind="iso", c=np.asarray(e["c"], np.float32), n=None, r0=max(r, 0.04), pool=(np.asarray(e["c"]) + r * d).astype(np.float32), col=col * e["flux"], mesh=e.get("mesh", -1), e=ei)); continue
        # rect: cut into patches no longer than ~0.8 m so each is point-like to what it lights
        u, v, n, c = (np.asarray(e[q], float) for q in ("u", "v", "n", "c")); lu, lv_ = 2 * np.linalg.norm(u), 2 * np.linalg.norm(v); nu, nv = max(1, int(np.ceil(lu / 0.8))), max(1, int(np.ceil(lv_ / 0.8)))
        while nu * nv > 12: nu, nv = max(1, nu - (nu >= nv)), max(1, nv - (nv > nu))
        for i in range(nu):
            for j in range(nv):
                cc = c + u * ((2 * i + 1) / nu - 1) + v * ((2 * j + 1) / nv - 1); q = rng.random((48, 2)) * 2 - 1; pool = cc + q[:, :1] * u / nu + q[:, 1:] * v / nv
                L.append(dict(kind="lamb", beam=float(e.get("beam", 1.0)), c=cc.astype(np.float32), n=n.astype(np.float32), r0=float(max(0.35 * np.sqrt(lu * lv_ / (nu * nv)), 0.04)), pool=pool.astype(np.float32), col=col * e["flux"] / (nu * nv), mesh=e.get("mesh", -1), e=ei))
    if len(L) > max_lights:                                   # keep the strongest
        L.sort(key=lambda l: -float(l["col"] @ YW)); L = L[:max_lights]
    return L

def geom(l, P, N, to=False):
    """Unshadowed irradiance factor (m,) of light l at points P with normals N (the light's colour x flux is applied by the caller).
    to=True: -> (the factor WITHOUT the surface's own cosine, unit direction to the light, cosine)."""
    if l["kind"] == "sun":
        c = N @ l["dir"]; return (np.ones(len(P), np.float32), np.broadcast_to(l["dir"], P.shape), c) if to else np.clip(c, 0, 1)
    v = l["c"] - P; d2 = (v * v).sum(1); d = np.sqrt(d2); om = v / np.maximum(d, 1e-6)[:, None]; cr = (om * N).sum(1)
    if to:
        b = l.get("beam", 1.0); f = 1 / (4 * np.pi * (d2 + l["r0"] ** 2)) if l["kind"] == "iso" else np.clip(-(om @ l["n"]), 0, 1) ** b * (b + 1) / (2 * np.pi * (d2 + l["r0"] ** 2)); return f.astype(np.float32), om, cr
    cr = np.clip(cr, 0, 1)
    if l["kind"] == "iso": return cr / (4 * np.pi * (d2 + l["r0"] ** 2))
    b = l["beam"]; return cr * np.clip(-(v @ l["n"]) / np.maximum(d, 1e-6), 0, 1) ** b * (b + 1) / (2 * np.pi * (d2 + l["r0"] ** 2))      # beam 1 = a flat glowing surface; higher = a narrower cone

def visibility(S, l, P, N, rng, K):
    """Share (m,) of light l that reaches points P (receivers, lifted off their surface along N): K shadow rays each."""
    vis = np.zeros(len(P), np.float32); g = geom(l, P, N); live = np.nonzero(g > 1e-5 * (1 if l["kind"] != "sun" else 1))[0]
    if not len(live): return vis
    O = np.repeat(P[live] + 0.012 * N[live], K, 0)
    if l["kind"] == "sun":
        j = rng.normal(size=(len(O), 3)).astype(np.float32) * l["spread"]; D = l["dir"] + j; D /= np.linalg.norm(D, axis=1, keepdims=True); t, mi = S.cast(O, D)
        if l["openings"]:
            through = np.full(len(O), np.inf, np.float32)
            for op in l["openings"]:
                n = np.asarray(op["n"], np.float32); den = D @ n; tt = ((np.asarray(op["c"], np.float32) - O) @ n) / np.where(np.abs(den) < 1e-6, 1e-6, den); X = O + D * tt[:, None] - np.asarray(op["c"], np.float32)
                u, v = np.asarray(op["u"], np.float32), np.asarray(op["v"], np.float32); inside = (tt > 0) & (den < 0) & (np.abs(X @ u) <= u @ u) & (np.abs(X @ v) <= v @ v); through = np.where(inside, np.minimum(through, tt), through)
            free = np.isfinite(through) & (t > through - 0.05)
        else: free = ~np.isfinite(t)
    else:
        Q = l["pool"][rng.integers(0, len(l["pool"]), len(O))]; D = Q - O; dist = np.linalg.norm(D, axis=1); D /= np.maximum(dist, 1e-6)[:, None]; t, mi = S.cast(O, D)
        free = (t > dist - 0.02) | ((mi == l["mesh"]) & (l["mesh"] >= 0))
    vis[live] = free.reshape(len(live), K).mean(1); return vis

# ================================================================ shine: view-dependent colour as spherical harmonics (3DGS layout, degree 3)
SH_A = np.exp(-np.arange(4) * (np.arange(4) + 1) * 0.22 / 2).astype(np.float32)         # a smooth lobe's weight per degree (about 65 degrees wide: the narrowest 3 degrees can hold without ringing)
SH_DEG = np.array([1] * 3 + [2] * 5 + [3] * 7)
def sh_basis(d):
    """The 15 basis functions of degree 1..3 at unit directions d (m, 3), as the viewer evaluates them."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]; xx, yy, zz = x * x, y * y, z * z
    return np.stack([-0.4886025 * y, 0.4886025 * z, -0.4886025 * x,
                     1.0925484 * x * y, -1.0925484 * y * z, 0.3153916 * (2 * zz - xx - yy), -1.0925484 * x * z, 0.5462742 * (xx - yy),
                     -0.5900436 * y * (3 * xx - yy), 2.8906114 * x * y * z, -0.4570458 * y * (4 * zz - xx - yy), 0.3731763 * z * (2 * zz - 3 * xx - 3 * yy),
                     -0.4570458 * x * (4 * zz - xx - yy), 1.4453057 * z * (xx - yy), -0.5900436 * x * (xx - 3 * yy)], 1).astype(np.float32)
GLOSS = (("chrome", 1.0, 1), ("mirror", 1.0, 1), ("brass", 0.8, 1), ("gold", 0.8, 1), ("silver", 0.9, 1), ("steel", 0.7, 1), ("metal", 0.7, 1), ("copper", 0.8, 1), ("bronze", 0.7, 1), ("iron", 0.35, 1), ("armour", 0.7, 1), ("armor", 0.7, 1),
         ("car", 0.9, 0), ("vehicle", 0.8, 0), ("motorcycle", 0.8, 0), ("glass", 0.9, 0), ("crystal", 0.9, 0), ("polished", 0.7, 0), ("lacquer", 0.7, 0), ("marble", 0.6, 0), ("porcelain", 0.7, 0), ("ceramic", 0.6, 0),
         ("piano", 0.8, 0), ("globe", 0.6, 1), ("leather", 0.3, 0), ("ice", 0.7, 0), ("wet", 0.6, 0), ("plastic", 0.4, 0), ("screen", 0.6, 0), ("console", 0.4, 0), ("trophy", 0.8, 1), ("bottle", 0.8, 0), ("vase", 0.5, 0))
def gloss_of(objects):
    """Per layout object -> (gloss 0..1, metal 0/1) from its name, kind and look words (a materials list from the assembler is better: pass gloss= yourself)."""
    g = []
    for o in objects:
        words = " ".join(str(o.get(q) or "") for q in ("name", "kind", "look")).lower().replace("_", " ").replace("-", " "); hit = [(v, m) for w, v, m in GLOSS if w in words.split() or (" " in w and w in words) or any(t.startswith(w) and len(t) <= len(w) + 1 for t in words.split())]
        g.append(max(hit) if hit else (0.0, 0))
    return g

# ================================================================ the bake
def delight(rgb_lin, N, P, strength):
    """Painted object: divide out the smooth part of its own light: log-brightness fitted against facing (up to second order) and height."""
    y = np.log(np.maximum(rgb_lin @ YW, 2e-3)); h = (P[:, 1] - P[:, 1].min()) / max(float(np.ptp(P[:, 1])), 1e-3); x, yy, z = N.T
    A = np.stack([np.ones_like(x), x, yy, z, x * yy, yy * z, x * z, x * x - z * z, 3 * yy * yy - 1, h], 1).astype(np.float32)
    s = np.random.default_rng(0).choice(len(y), min(len(y), 40000), replace=False); co, *_ = np.linalg.lstsq(A[s], y[s], rcond=None); fit = A @ co; fit -= fit.mean()
    return rgb_lin / np.exp(np.clip(strength * fit, -0.6, 0.6))[:, None]

# ================================================================ splat files
def write_ply(p, a, cols):
    with open(p, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(a) + "".join(f"property float {s}\n" for s in cols) + "end_header\n").encode()); np.ascontiguousarray(a, "<f4").tofile(f)

def quat_z_to(n):
    z = np.array([0, 0, 1.0]); c = n @ z; ax = np.cross(np.broadcast_to(z, n.shape), n); q = np.concatenate([(1 + c)[:, None], ax], 1); q[c < -0.9999] = [0, 1, 0, 0]; return q / np.linalg.norm(q, axis=1, keepdims=True)
