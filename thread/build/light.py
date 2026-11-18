"""The light: believable sources, crisp sun, a filmic curve, contact shade and shine, traced on ALL of a level's geometry and baked into the
discs of a directly written splat. CPU only (Embree through open3d), no model. thread/.venv; uses every core. The cells, lights and
bounce are light_base.py's; how sources LOOK, the sun, the exposure, the curve and the shine are here.

THE CALL THE ASSEMBLER MAKES (one entry point)
    import light
    res = light.light(discs, meshes, objects, openings, fittings, sentence, mood=None, opts=None)
    light.write(out_ply, discs, res, placing)          # OUT.ply (+ OUT_sh.ply: the glossy discs with their harmonics; the viewer loads both)
  discs     {"xyz" (n,3) room metres, "normal" (n,3) unit, facing the air, "rgb" (n,3) uint8 paint taken as albedo, "mesh" (n,) index into meshes
             or -1, "sigma" (n,) disc radius m, "thick" (n,) m, "opacity" (n,)   [what the writer needs]
             optional: "emit" (n,) bool (window glass, lamp diffusers, flames: not lit, they shine), "part" (n,) int + discs["parts"] [str]
             (names such as "candle:flame", "shell:window.glass", "mirror:picture"), "cls" (n,) uint8 material class (see CLASSES; overrides the
             rough classes derived here from words and colour), "gloss" (n,) 0..1}
  meshes    [{"V","F","name","painted"}] everything that blocks light (shell with its openings cut, kit pieces, every object)
  objects   [{"name","kind","look","mesh"}] the layout's things (which mesh is whose)
  openings  [{"kind": "window"|"skylight", "c","u","v","n"}] (n INTO the room); fittings [{"c","u","v","n"}] the kit's own lamps (panels)
  -> res    {"rgb" (n,3) uint8, "opacity" (n,) new opacity (glass is a little see-through), "keep" (n,) bool (discs taken out: the wall behind a
             mirror), "extra" {"xyz","rgb","sigma" (m,3),"normal","opacity"} new discs (flames, embers, glow, mirror images), "sh" {"idx","coef"},
             "info" timings and what was found}

HOW (all rules general: nothing knows a world)
  sources   a lamp shines from a small core inside it (not from its whole skin): its glass glows with a hot centre falling off to orange, its
            frame stays dark, a soft halo sits round it whose strength follows the exposure (strong at night, faint by day). A candle gets a flame.
            A fire is FOUND: the hollow of the fireplace is measured by rays; the light sits inside it (so the hollow glows and the opening throws
            a fan of light), with an ember bed, logs and flame tongues made by rule.
  sun/moon  every disc that can see the sun asks with its own rays (7, over the sun's own half degree): glazing bars and frames cast crisp
            shadows that soften only with distance. The sky's glow comes from the openings as before.
  exposure  one rule per time of day, metered half on the light that falls (so a white room stays white and a dark room dark) and half on
            what is reflected; a filmic curve with a long shoulder on brightness, colour kept and let go towards white only at the top.
  contact   the bounce rays also measure closeness within 22 cm: things sit in a soft dark line where they meet the floor and each other.
  shine     glossy discs (car paint, chrome, glass, brass, polished floors) get (1) a mirror picture of the level baked for an eye standing where a
            walker would look from (one reflection ray a disc into the lit level: chrome shows the windows, the lamps and the floor), (2) degree-3
            harmonics that move with the eye: a lobe about each light's mirror direction, with Fresnel, and a rim sheen.
            Glass is darker, reflective and a little see-through. A flat mirror becomes a real one: the wall behind it is opened and a mirror
            image of what it can show is hung behind (see mirror.py)."""
import os, time
import numpy as np
import light_base as g4
from light_base import YW, C0, s2l, l2s, Scene, Cells, hemi, sh_basis, SH_DEG, AXES, mood_of, write_ply, quat_z_to, PLY14

MATT, GLOSS, METAL, GLASS, LAMPGLASS, FLAME, MIRROR, SKY, PANEL = range(9)
CLASSES = ("matt", "gloss paint", "metal/chrome", "glass", "lamp glass", "flame", "mirror", "sky seen through glass", "light panel")
OPT = dict(receivers=700_000, k_shadow=9, rays=48, bounces=3, reach=1.5, contact=0.22, delight=0.7, max_lights=56, sun_deg=0.6,
           halo=1.0, floor_gloss=None, mirror=True, reflect=True, sh=True, spec=25.0)
# one rule per time of day: key = mean display brightness aimed at; grade = light contrast kept over physical; wb = how far the lights' common
# cast is taken out; amb = cool fill as a share of the mean light
TIME = {"night": dict(key=0.125, grade=0.30, wb=0.2, amb=0.07), "dusk": dict(key=0.27, grade=0.2, wb=0.35, amb=0.05), "day": dict(key=0.47, grade=0.1, wb=0.5, amb=0.04)}
KEY_EVEN = {"night": 0.30, "dusk": 0.36, "day": 0.47}      # the key when the light is even all over (lamps on everywhere): TIME's key is for light in pools
SKY_TOP = {"night": [0.012, 0.022, 0.07], "dusk": [0.20, 0.22, 0.42], "day": [0.36, 0.58, 0.95]}       # linear display colours of the sky seen through glass
SKY_LOW = {"night": [0.035, 0.055, 0.13], "dusk": [0.95, 0.47, 0.20], "day": [0.86, 0.91, 1.0]}
SUN_CAP = {"night": 0.5, "dusk": 5.0, "day": 9.0}

FILM_P = 1.3
def film(x, W=40.0):
    """Linear colour (n, 3) 0..inf -> 0..1 display-linear. A long shoulder on BRIGHTNESS (y / (1 + y), white at W); the colour is kept and let go
    towards white only in the top third, so a lamp-lit wall turns cream, not white, and nothing clips to a flat blob."""
    y = np.maximum(x @ YW, 1e-9); t = np.minimum(y / (1 + y ** FILM_P) ** (1 / FILM_P) * ((1 + W ** FILM_P) ** (1 / FILM_P) / W), 1.0); c = x * (t / y)[..., None]
    s = np.clip((t - 0.6) / 0.4, 0, 1); s = s * s * (3 - 2 * s) * 0.8; c = c + (t[..., None] - c) * s[..., None]
    m = np.maximum(c.max(-1), 1.0); k = m ** -0.55; c = c * k[..., None]; t = t * k           # a colour too strong for the screen: partly dimmed (a red car stays red) ...
    m = c.max(-1); over = m > 1; f = np.where(over, (1 - t) / np.maximum(m - t, 1e-6), 1.0); c = t[..., None] + (c - t[..., None]) * f[..., None]      # ... and the rest let go towards white
    return np.clip(c, 0, 1)

# ================================================================ what things are made of (a rough stand-in for the paint part's classes)
VEHICLE = ("car", "sedan", "coupe", "truck", "van", "vehicle", "automobile", "motorcycle", "motorbike", "scooter", "bus", "tractor", "jeep", "taxi", "limousine", "roadster")
def words_of(o, main_only=False):
    look = str(o.get("look") or "").lower()
    if main_only: look = look.split(" with ")[0]            # "leather with brass tacks" is leather
    return (" ".join(str(o.get(q) or "") for q in ("name", "kind")) + " " + look).lower().replace("_", " ").replace("-", " ")
PAINTED = ("red", "blue", "green", "yellow", "orange", "purple", "pink", "white", "black", "blackened", "painted", "enamel", "enamelled", "lacquered", "navy", "teal", "crimson", "maroon", "cream", "turquoise")
def gloss_words(words):
    ws = words.split(); hit = [(v, m) for w, v, m in g4.GLOSS if w in ws or (" " in w and w in words) or any(t.startswith(w) and len(t) <= len(w) + 1 for t in ws)]
    if any(w in ws or (w + "s") in ws for w in VEHICLE): hit.append((0.9, 0))
    if not hit: return (0.0, 0)
    g, m = max(hit)
    if m and any(w in ws for w in PAINTED) and not any(w in ws for w in ("chrome", "brass", "gold", "silver", "copper", "bronze", "mirror")): m = 0; g = min(g, 0.65)      # "red metal" is painted metal: gloss paint, not a mirror
    return (g, m)

def classes(discs, meshes, objects, alb, floor_gloss=0.0):
    """-> cls (n,) uint8, gloss (n,) float32. From the object's words (its main material, not its details), the part's name, and for
    vehicles the paint itself (body colour = gloss paint, pale grey = chrome, dark or grey above the waist = glass, dark below = rubber)."""
    n = len(alb); cls = np.zeros(n, np.uint8); gl = np.zeros(n, np.float32); dm = discs["mesh"]; P = discs["xyz"]; N = discs["normal"]
    emit = discs.get("emit"); emit = np.zeros(n, bool) if emit is None else emit
    for o in objects or []:
        mi = o.get("mesh", -1)
        if mi is None or mi < 0: continue
        w = words_of(o, True); g, metal = gloss_words(w); sel = np.nonzero(dm == mi)[0]
        if not len(sel): continue
        veh = any(v in w.split() or (v + "s") in w.split() for v in VEHICLE)
        if veh:
            a = alb[sel]; mx = a.max(1); mn = a.min(1); sat = (mx - mn) / np.maximum(mx, 1e-4); h = (P[sel, 1] - P[sel, 1].min()) / max(float(np.ptp(P[sel, 1])), 1e-3)
            c = np.full(len(sel), GLOSS, np.uint8); gg = np.full(len(sel), 0.9, np.float32)
            def vote(m, cell=0.05):                            # the share of a 5 cm neighbourhood that says so
                ijk = np.floor(P[sel] / cell).astype(np.int64); ijk -= ijk.min(0); key = (ijk[:, 0] * (ijk[:, 1].max() + 1) + ijk[:, 1]) * (ijk[:, 2].max() + 1) + ijk[:, 2]; _, inv = np.unique(key, return_inverse=True)
                return (np.bincount(inv, m.astype(np.float32)) / np.bincount(inv))[inv]
            chrome = vote((sat < 0.22) & (mx > 0.25)) > 0.5; c[chrome] = METAL; gg[chrome] = 1.0
            glass = vote((h > 0.55) & (sat < 0.33) & (np.abs(N[sel, 1]) < 0.75)) > 0.5; c[glass] = GLASS; gg[glass] = 0.95       # the cabin band, not the roof
            rub = (vote((mx < 0.045) & (h < 0.4)) > 0.5) & ~glass; c[rub] = MATT; gg[rub] = 0.0          # tyres
            cls[sel] = c; gl[sel] = gg
        elif "mirror" in w.split(): cls[sel] = METAL; gl[sel] = 0.8
        elif g > 0.05:
            gls = any(q in w.split() for q in ("glass", "crystal", "bottle", "ice")); cls[sel] = GLASS if gls else METAL if metal else GLOSS; gl[sel] = g
    parts = discs.get("parts"); part = discs.get("part")
    if parts is not None and part is not None:
        for i, nm in enumerate(parts):
            nm = nm.lower(); leaf = nm.split(":")[-1]; sel = part == i
            if "glass" in leaf: cls[sel] = np.where(emit[sel], SKY, GLASS); gl[sel] = np.maximum(gl[sel], 0.9)
            elif "diffuser" in leaf or ("light" in leaf and "panel" in leaf): cls[sel & emit] = PANEL
            elif "flame" in leaf: cls[sel] = FLAME
            elif leaf == "picture" and "mirror" in nm: cls[sel] = MIRROR; gl[sel] = 1.0
            elif nm.startswith("shell:floor") and floor_gloss and floor_gloss > 0: cls[sel] = GLOSS; gl[sel] = floor_gloss
    if floor_gloss and floor_gloss > 0 and (parts is None or part is None):
        for k, m in enumerate(meshes):
            if str(m.get("name", "")).startswith("shell:floor"): sel = (dm == k) & (N[:, 1] > 0.9); cls[sel] = GLOSS; gl[sel] = floor_gloss
    cls[emit & (cls != SKY) & (cls != PANEL) & (cls != FLAME)] = FLAME      # anything else flagged as shining is taken as a small flame/bulb
    c2 = discs.get("cls")
    if c2 is not None: cls = np.asarray(c2, np.uint8).copy()
    g2 = discs.get("gloss")
    if g2 is not None: gl = np.asarray(g2, np.float32).copy()
    return cls, gl

# ================================================================ sources: where the light really leaves from, and what the thing looks like
def find_hollow(S, mi, meshes, n):
    """The firebox of a fireplace-like mesh, measured: rays from in front, into it. -> dict(uc, w, y0, h, depth, ax, ot, front) or None."""
    from scipy import ndimage
    V = np.asarray(meshes[mi]["V"], float); lo, hi = V.min(0), V.max(0); ext = hi - lo; ax = int(np.abs(n).argmax()); ot = 2 - ax; sg = float(np.sign(n[ax])); front = hi[ax] if sg > 0 else lo[ax]
    nu, nv = 56, 40; us = lo[ot] + (np.arange(nu) + 0.5) / nu * ext[ot]; vs = hi[1] - (np.arange(nv) + 0.5) / nv * ext[1]; U, Vv = np.meshgrid(us, vs)
    O = np.zeros((nu * nv, 3)); O[:, ot] = U.ravel(); O[:, 1] = Vv.ravel(); O[:, ax] = front + 0.3 * sg; D = np.zeros((nu * nv, 3)); D[:, ax] = -sg
    t, hit = S.cast(O, D); d = np.where(np.isfinite(t), t - 0.3, 9.0).reshape(nv, nu); own = (hit == mi).reshape(nv, nu) & (d < ext[ax])
    if own.sum() < 20: return None
    ref = float(np.median(d[own])); deep = d > ref + 0.10; lab, k = ndimage.label(deep); best = None
    for j in range(1, k + 1):
        m = lab == j; ys, xs = np.nonzero(m)
        if ys.min() == 0 or xs.min() == 0 or xs.max() == nu - 1: continue          # open to the outside: not a hollow (the bottom row may be open to the hearth)
        area = m.sum() * (ext[ot] / nu) * (ext[1] / nv)
        if area < 0.03 or vs[ys].mean() > lo[1] + 0.75 * ext[1]: continue
        if best is None or area > best[0]: best = (area, m, ys, xs)
    if best is None: return None
    _, m, ys, xs = best; u0, u1 = us[xs.min()] - 0.5 * ext[ot] / nu, us[xs.max()] + 0.5 * ext[ot] / nu; y1, y0 = vs[ys.min()] + 0.5 * ext[1] / nv, vs[ys.max()] - 0.5 * ext[1] / nv
    depth = float(np.clip(np.median(np.minimum(d[m], ext[ax] + 0.4)) - ref, 0.12, 0.8)); y0 = max(y0, lo[1])
    isl = ndimage.binary_fill_holes(m) & ~m; island = None                    # something sculpted INSIDE the opening (logs, a fire): it will burn
    if isl.sum() * (ext[ot] / nu) * (ext[1] / nv) > 0.008:
        iy, ix = np.nonzero(isl); island = dict(mask=ndimage.binary_dilation(isl), u0=float(lo[ot]), du=float(ext[ot] / nu), v1=float(hi[1]), dv=float(ext[1] / nv), ya=float(vs[iy.max()]), yb=float(vs[iy.min()]), uc=float(us[ix].mean()), w=float(np.ptp(us[ix])))
    return dict(island=island, uc=(u0 + u1) / 2, w=u1 - u0, y0=float(y0), h=float(y1 - y0), depth=depth, ax=ax, ot=ot, sg=sg, front=float(front - sg * ref), area=float(best[0]))

def sources(emitters, meshes, S, discs):
    """The found emitters -> (emitters to trace, models of what each source looks like)."""
    out, models = [], []
    for e in emitters:
        e = dict(e); cls = e.get("cls")
        if e["type"] == "mesh" and cls in ("lamp", "candle"):
            V = np.asarray(meshes[e["mesh"]]["V"], float); lo, hi = V.min(0), V.max(0); ext = hi - lo; c = (lo + hi) / 2; nm = str(e.get("name") or "").lower()
            if cls == "candle": c = np.array([c[0], hi[1] - 0.018, c[2]]); rs = 0.012; flame = 1.0; e["flux"] *= 0.6
            elif any(w in nm for w in ("sconce", "torch", "brazier")): c = c + [0, 0.12 * ext[1], 0]; rs = float(0.2 * np.sort(ext)[1]); flame = 1.5; e["flux"] *= 0.45
            else: rs = float(0.22 * ext.mean()); flame = 0.0; e["flux"] *= 2.5
            out.append(dict(type="point", cls=cls, name=e.get("name"), c=c, r=max(0.6 * rs, 0.02), col=e["col"], flux=e["flux"], mesh=e["mesh"]))
            models.append(dict(kind=cls, c=c, rs=rs, mesh=e["mesh"], col=np.asarray(e["col"], np.float32), flux=float(e["flux"]), flame=flame, size=float(np.linalg.norm(ext)), em=len(out) - 1))
        elif e["type"] == "rect" and cls == "fire":
            n = np.asarray(e["n"], float)
            if e.get("mesh", -1) >= 0:                           # which way the fireplace faces: the side with open room in front of it (a guess from the layout alone went into the wall)
                V = np.asarray(meshes[e["mesh"]]["V"], float); lo, hi = V.min(0), V.max(0); cm = (lo + hi) / 2; ax = int(np.abs(n).argmax()); fr = []
                for sg in (1.0, -1.0):
                    O = np.tile(cm, (5, 1)); O[:, ax] = (hi[ax] if sg > 0 else lo[ax]) + 0.02 * sg; O[:, 1] = lo[1] + np.linspace(0.3, 0.9, 5) * (hi[1] - lo[1]); D = np.zeros((5, 3)); D[:, ax] = sg; t, _ = S.cast(O, D); fr.append(float(np.where(np.isfinite(t), t, 0.0).max()))      # a ray that meets nothing is outside the level
                sg = 1.0 if fr[0] >= fr[1] else -1.0
                if sg != np.sign(n[ax]): n = -n; c0 = np.asarray(e["c"], float).copy(); c0[ax] = (hi[ax] if sg > 0 else lo[ax]) + 0.04 * sg; e["n"] = n; e["c"] = c0
            H = find_hollow(S, e["mesh"], meshes, n) if e.get("mesh", -1) >= 0 else None
            if H is None:
                u, v, c = (np.asarray(e[q], float) for q in ("u", "v", "c")); out.append(e); ax = int(np.abs(n).argmax())
                models.append(dict(kind="fire", hollow=False, H=dict(uc=float(c[2 - ax]), w=float(2 * np.linalg.norm(u)), y0=float(c[1] - np.linalg.norm(v)), h=float(2 * np.linalg.norm(v)), depth=0.0, ax=ax, ot=2 - ax, sg=float(np.sign(n[ax])), front=float(c[ax])),
                                   mesh=e.get("mesh", -1), col=np.asarray(e["col"], np.float32), flux=float(e["flux"]), em=len(out) - 1))
            else:
                first = len(out)
                for q in (-0.27, 0.0, 0.27):                    # the light sits inside the hollow, low: the hollow glows, the opening throws a fan (if a sculpted fire fills the opening: just in front of it)
                    c = np.zeros(3); c[H["ot"]] = H["uc"] + q * H["w"]; c[1] = H["y0"] + min(0.3 * H["h"], 0.16); c[H["ax"]] = H["front"] - H["sg"] * (0.5 * H["depth"] if H["island"] is None else -0.07)
                    out.append(dict(type="point", cls="fire", name=e.get("name"), c=c, r=0.13, col=e["col"], flux=1.3 * e["flux"] / 3, mesh=-1))
                models.append(dict(kind="fire", hollow=True, H=H, mesh=e.get("mesh", -1), col=np.asarray(e["col"], np.float32), flux=float(e["flux"]), em=first))
        else: out.append(e)
    return out, models

def blob(X, c, sig, col, op, nrm=(0, 1, 0)):
    X.append((np.asarray(c, np.float32), np.broadcast_to(np.asarray(sig, np.float32), (3,)).copy(), np.clip(np.asarray(col, np.float32), 0, 1), float(op), np.asarray(nrm, np.float32)))

def flame(X, c, s=1.0, up=(0, 1, 0)):
    """A flame as three soft drops: white-yellow heart, orange coat, a faint red tip. s = 1: a candle (1.2 cm wide)."""
    c = np.asarray(c, float); u = np.asarray(up, float)
    blob(X, c + u * 0.016 * s, [0.0065 * s, 0.0065 * s, 0.017 * s], [1.0, 0.62, 0.16], 0.55, u); blob(X, c + u * 0.030 * s, [0.004 * s, 0.004 * s, 0.012 * s], [1.0, 0.42, 0.08], 0.3, u)
    blob(X, c + u * 0.009 * s, [0.0042 * s, 0.0042 * s, 0.010 * s], [1.0, 0.97, 0.80], 0.96, u)

def halo(X, c, rs, col, strength):
    """Bloom: three wide soft gaussians, warm and more coloured outwards. strength 0..1 follows how far the source is over the exposure."""
    col = np.asarray(col, np.float32); col = col / col.max(); s = float(np.clip(strength, 0, 1))
    if s < 0.03: return
    blob(X, c, 1.0 * rs + 0.012, 0.45 + 0.55 * col, 0.62 * s); blob(X, c, 2.2 * rs + 0.04, 0.2 + 0.8 * col, 0.36 * s); blob(X, c, 5 * rs + 0.10, col * [1, 0.92, 0.8], 0.17 * s); blob(X, c, 11 * rs + 0.25, col * [1, 0.85, 0.65], 0.07 * s)

def fire_extras(X, m, seed=0):
    """Ember bed, logs and flame tongues in a firebox (or against a flat front if no hollow was found), by rule from its measured size."""
    H = m["H"]; rng = np.random.default_rng(seed); ax, ot, sg = H["ax"], H["ot"], H["sg"]; w = H["w"]; dpt = H["depth"]; isl = H.get("island")
    mid = H["front"] - sg * (0.5 * dpt if m["hollow"] else -0.10); half = 0.32 * min(dpt, 0.5) if m["hollow"] else 0.07; y0 = H["y0"] + 0.015; hh = H["h"]
    if isl is not None: mid = H["front"] + sg * 0.035; half = 0.03                 # a sculpted fire stands in the opening: only embers, at its foot
    def at(u, y, d): p = np.zeros(3); p[ot] = H["uc"] + u; p[1] = y; p[ax] = mid + d; return p
    eo = np.zeros(3); eo[ot] = 1.0
    for i in range(3 if isl is None else 0):                  # logs: dark, lying across, lit from under
        y = y0 + 0.045 + 0.035 * (i == 2); d = (-0.5, 0.5, 0.0)[i] * half; L = (0.36, 0.33, 0.28)[i] * w
        blob(X, at((rng.random() - 0.5) * 0.08 * w, y, d), [0.022, 0.022, 0.5 * L], [0.10, 0.055, 0.035], 0.97, eo)
        for q in np.linspace(-0.85, 0.85, 7): blob(X, at(q * 0.5 * L, y - 0.012 + 0.02 * rng.random(), d + 0.012 * sg), [0.011, 0.009, 0.022], [1.0, 0.38 + 0.25 * rng.random(), 0.06], 0.85, eo)
    for _ in range(110):                                      # embers: a bed, hotter towards the middle
        u = (rng.random() - 0.5) * 0.8 * w; d = (rng.random() - 0.5) * 2 * half * 1.25; heat = np.clip(rng.random() ** 1.3 * (1.15 - abs(u) / (0.5 * w)), 0, 1)
        col = (1 - heat) * np.array([0.30, 0.03, 0.01]) + heat * np.array([1.0, 0.50, 0.10]) + (heat > 0.8) * np.array([0, 0.25, 0.3]) * (heat - 0.8) * 5
        blob(X, at(u, y0 + 0.012 + 0.03 * rng.random() * heat, d), [0.01 + 0.014 * rng.random(), 0.01 + 0.014 * rng.random(), 0.007], col, 0.93)
    for k in range(7 if isl is None else 0):                  # tongues
        u = (k / 6 - 0.5) * 0.62 * w + (rng.random() - 0.5) * 0.06 * w; d = (rng.random() - 0.5) * half; ht = float(np.clip((0.3 + 0.38 * rng.random()) * hh * (1.1 - abs(u) / (0.6 * w)), 0.1, 0.5)); sw = (rng.random() - 0.5) * 0.05; n_ = 6
        for j in range(n_):
            f = (j + 0.5) / n_; r = (0.036 * (1 - f) + 0.010) * (0.7 + 0.6 * min(w, 0.8)); col = np.array([1.0, 0.86 - 0.60 * f, 0.36 - 0.34 * f ** 0.6]); op = 0.62 * (1 - f) ** 1.2 + 0.08
            blob(X, at(u + sw * f * f * 4 * (0.5 - (k % 2)), y0 + 0.05 + f * ht, d), [r, r, 0.62 * ht / n_ * 1.6], col, op)

# ================================================================ emitters (kept here so there is ONE call)
def find_emitters(objects, openings, fittings, sentence, meshes, mood):
    em = g4.find_emitters(objects, openings, sentence, None, meshes, mood); tm = mood["time"]
    for e in em:
        if e.get("cls") == "opening": e["c"] = np.asarray(e["c"], float) + 0.18 * np.asarray(e["n"], float)      # the sky's glow leaves from just inside the glazing: frames and bars stand dark against it
        if e.get("cls") == "sun": e["flux"] = min(e["flux"], SUN_CAP[tm])
    for f in fittings or []:
        u, v, nn = (np.asarray(f[q], float) for q in ("u", "v", "n"))
        em.append(dict(type="rect", cls="downlight", name="fitting", c=np.asarray(f["c"], float) + 0.03 * nn, u=u, v=v, n=nn, col=g4.C_PANEL, flux=float(0.5 + 1.2 * np.sqrt(4 * np.linalg.norm(u) * np.linalg.norm(v))), beam=1.5, mesh=-1))
    return em

# ================================================================ the light
def light(discs, meshes, objects, openings=None, fittings=None, sentence="", mood=None, opts=None, emitters=None):
    mood = dict(mood_of(sentence, mood), **{k: v for k, v in (mood or {}).items() if k in ("key",)})
    if emitters is None:
        emitters = find_emitters(objects, openings, fittings, sentence, meshes, mood)
        if not emitters:                                       # nothing gives light: ceiling downlights
            V = np.concatenate([np.asarray(m["V"], float) for m in meshes]); emitters = g4.find_emitters([], None, sentence, V.max(0) - V.min(0), meshes, mood)
    return light_level(discs, meshes, emitters, mood, objects, opts, sentence)

def light_level(discs, meshes, emitters, mood, objects=None, opts=None, sentence=""):
    tm_ = mood["time"]; o = dict(OPT, **TIME[tm_]); o.update({k: v for k, v in mood.items() if k in ("key",)}); o.update(opts or {}); t0 = time.time(); tm = {}; rng = np.random.default_rng(3)
    P = np.ascontiguousarray(discs["xyz"], np.float32).copy(); N = np.ascontiguousarray(discs["normal"], np.float32); n = len(P); meshes = list(meshes); S = Scene(meshes); dmesh = np.asarray(discs["mesh"]).astype(np.int64)
    alb = np.clip(s2l(discs["rgb"]), 0.004, 0.9); emit = discs.get("emit"); emit = np.zeros(n, bool) if emit is None else np.asarray(emit, bool)
    if o["floor_gloss"] is None: o["floor_gloss"] = 0.0       # a polished floor's sheen in degree-3 harmonics is too broad to be seen (tried: 150 MB for nothing); pass floor_gloss to try it
    cls, gls = classes(discs, meshes, objects, alb, o["floor_gloss"]); tm["scene_s"] = round(time.time() - t0, 2)
    # ---- cells and receivers
    t, mi, hn = S.cast(P + 0.001 * N, N, normals=True); under = (t < 0.05) & ((hn * N).sum(1) > 0); P[under] += N[under] * (t[under] + 0.003)[:, None]
    area = sum(float(np.linalg.norm(np.cross(m["V"][m["F"][:, 1]] - m["V"][m["F"][:, 0]], m["V"][m["F"][:, 2]] - m["V"][m["F"][:, 0]]), axis=1).sum()) / 2 for m in meshes)
    c = float(np.clip(np.sqrt(area / o["receivers"]), 0.012, 0.06)); C = Cells(P, N, c); RP = P[C.rep]; RN = N[C.rep]
    t, mi, hn = S.cast(RP + 0.002 * RN, RN, normals=True); ok = ~((t < 0.05) & ((hn * RN).sum(1) > 0))
    tm["cells_s"] = round(time.time() - t0, 2); tm["cell_mm"] = round(c * 1000, 1); tm["receivers"] = int(C.n)
    painted = np.array([bool(m.get("painted")) for m in meshes] + [False])
    for k in np.nonzero(painted[:-1])[0]:
        sel = np.nonzero(dmesh == k)[0]
        if len(sel) > 500: alb[sel] = np.clip(g4.delight(alb[sel], N[sel], P[sel], o["delight"]), 0.004, 0.9)
    # ---- sources and lights
    em, models = sources(emitters, meshes, S, discs); L = g4.make_lights(em, meshes, rng, o["max_lights"]); nL = len(L)
    for m in models:                                           # the inside of a firebox is sooty: flames and embers show against it
        if m["kind"] == "fire" and m["hollow"]:
            H = m["H"]; dpt = (H["front"] - P[:, H["ax"]]) * H["sg"]; ins = (np.abs(P[:, H["ot"]] - H["uc"]) < 0.5 * H["w"] + 0.03) & (P[:, 1] > H["y0"] - 0.03) & (P[:, 1] < H["y0"] + H["h"] + 0.03) & (dpt > 0.03) & (dpt < H["depth"] + 0.25)
            alb[ins] = alb[ins] * 0.22 * np.float32([1.0, 0.9, 0.8])
    ralb = C.mean(alb)
    if L:
        cb = sum(l["col"] for l in L); cb = cb / float(cb @ YW); wbc = np.maximum(cb, 0.2) ** o["wb"]
        for l in L: y0 = float(l["col"] @ YW); l["col"] = l["col"] / wbc; l["col"] = (l["col"] * y0 / float(l["col"] @ YW)).astype(np.float32)
    else: wbc = np.ones(3, np.float32)
    isun = [k for k, l in enumerate(L) if l["kind"] == "sun"]; nos = [k for k in range(nL) if k not in isun]
    vis = np.zeros((C.n, nL), np.float32); live = np.nonzero(ok)[0]
    U = np.stack([g4.geom(l, RP[live], RN[live]) * float(l["col"] @ YW) for l in L], 1) if L else np.zeros((len(live), 0), np.float32); Ut = U.sum(1)
    for k, l in enumerate(L):                                  # shadow rays only where this light is more than 1/250 of what could reach the place (the rest: taken as half seen)
        need = (U[:, k] > 0.004 * Ut) | (l["kind"] == "sun"); vis[live[need], k] = g4.visibility(S, l, RP[live[need]], RN[live[need]], rng, o["k_shadow"]); vis[live[~need], k] = 0.5
    del U
    vis = C.blur(vis, ok) * 0.5 + vis * 0.5
    Ed = np.zeros((C.n, 3), np.float32); Es = np.zeros((C.n, 3), np.float32)
    for k, l in enumerate(L):
        e = (vis[:, k] * g4.geom(l, RP, RN))[:, None] * l["col"]
        if k in isun: Es += e
        else: Ed += e
    # a lamp does not light its own skin from inside (its glass glows by its own rule, its frame is a dark shape against it)
    rmesh = np.full(C.n, -1, np.int64); rmesh[:] = dmesh[C.rep]
    for k, l in enumerate(L):
        if l.get("mesh", -1) >= 0 and l["kind"] == "iso":
            own = rmesh == l["mesh"]; Ed[own] -= 0.97 * (vis[own, k] * g4.geom(l, RP[own], RN[own]))[:, None] * l["col"]
    tm["direct_s"] = round(time.time() - t0, 2); tm["lights"] = nL
    # ---- bounce + openness wide and close
    cc = 3 * c; CC = Cells(RP[live], RN[live], cc); R = o["rays"]; hit_row = np.empty((len(live), R), np.int32); opn = np.zeros(len(live), np.float32); cls_ = np.zeros(len(live), np.float32)
    for a in range(0, len(live), 80000):
        s = slice(a, a + 80000); p, nn = RP[live[s]], RN[live[s]]; d = hemi(nn, R, rng).astype(np.float32); O = np.broadcast_to((p + 0.012 * nn)[:, None], d.shape).reshape(-1, 3); D = d.reshape(-1, 3)
        t, mi, hn = S.cast(O, D, normals=True); fin = np.isfinite(t); X = O + D * np.where(fin, t, 0)[:, None]; hn = hn * -np.sign((hn * D).sum(1, keepdims=True)); X = X + 0.004 * hn
        j, hit = CC.find(CC.key(np.floor((X - CC.lo) / cc).astype(np.int64), Cells.bucket(hn))); hit_row[s] = np.where(hit & fin, j, -1).astype(np.int32).reshape(-1, R)
        opn[s] = np.clip(t / o["reach"], 0, 1).reshape(-1, R).mean(1); cls_[s] = np.clip(t / o["contact"], 0, 1).reshape(-1, R).mean(1)
    calb = CC.mean(ralb[live]); B = calb * CC.mean((Ed + Es)[live]); tot = np.zeros((len(live), 3), np.float32); miss = hit_row < 0
    for bnc in range(o["bounces"]):
        Bm = B.mean(0) * 0.5; g = B[np.maximum(hit_row, 0)]; g[miss] = Bm; Ei = g.mean(1); tot += Ei; B = calb * CC.mean(Ei)
    allc = np.ones(CC.n, bool); ci = CC.blur(CC.mean(tot), allc); co = CC.mean(opn[:, None]); rc, good = CC.read(RP[live], RN[live], np.concatenate([ci, co], 1), allc); rc[~good] = np.concatenate([tot, opn[:, None]], 1)[~good]
    AO = np.zeros((C.n, 1), np.float32); AO[live, 0] = opn; AO = C.blur(C.blur(AO, ok), ok); AC = np.ones((C.n, 1), np.float32); AC[live, 0] = cls_; AC = C.blur(AC, ok)
    Eind = np.zeros((C.n, 3), np.float32); Eind[live] = rc[:, :3] * np.clip(AO[live] / np.maximum(rc[:, 3:4], 1e-3), 0.45, 1.5)
    tm["bounce_s"] = round(time.time() - t0, 2)
    w = C.count * ok; mean_E = float((((Ed + Es + Eind) @ YW) * w).sum() / w.sum()); cbar = sum(l["col"] for l in L) if L else np.ones(3, np.float32); cbar = cbar / max(float(cbar @ YW), 1e-9)
    amb_c = 1 / np.maximum(cbar, 0.25); amb_c = (0.5 * amb_c / (amb_c @ YW) + 0.5).astype(np.float32); A = o["amb"] * mean_E / max(float((AO[:, 0] * w).sum() / w.sum()), 1e-6) if L else 1.0
    def shade(ed, es, ei, ao, ac):
        """All the light on a surface: close things take most of the soft light and part of the direct light away where they meet."""
        ac = np.clip(ac, 0, 1); return (ed + es) * (0.5 + 0.5 * ac) + (ei + A * ao * amb_c) * ac ** 1.7
    # ---- per disc: flat axis-facing discs read the light between cells; everything else works each light out with its own facing
    Edir = np.zeros((n, 3), np.float32); Esoft = np.zeros((n, 3), np.float32); sunv = np.zeros(n, np.float32); flat = np.abs(N).max(1) > 0.9995; lost = 0; aux = np.zeros((n, 2), np.float32)      # aux: openness wide, close
    vf = np.concatenate([Ed, Eind, AO, AC, vis[:, isun] if isun else np.zeros((C.n, 0), np.float32)], 1); i_flat = np.nonzero(flat)[0]
    for a in range(0, len(i_flat), 600_000):
        i1 = i_flat[a:a + 600_000]; r, good = C.read(P[i1], N[i1], vf, ok); bad = np.nonzero(~good)[0]
        if len(bad): r[bad] = vf[C.inv[i1[bad]]]; lost += len(bad)
        Edir[i1] = r[:, :3]; Esoft[i1] = r[:, 3:6] + A * r[:, 6:7] * amb_c; aux[i1] = r[:, 6:8]
        if isun: sunv[i1] = r[:, 8:].max(1)
    i_obj = np.nonzero(~flat)[0]; i_obj = i_obj[np.argsort(dmesh[i_obj], kind="stable")]; vb = np.concatenate([Eind, AO, AC], 1); kept_lights = []      # thing by thing, so a chunk is one place
    for a in range(0, len(i_obj), 250_000):
        i2 = i_obj[a:a + 250_000]; sub = i2[:: max(1, len(i2) // 3000)]; Us = np.array([float((g4.geom(L[k], P[sub], N[sub]) * float(L[k]["col"] @ YW)).mean()) for k in nos]) if nos else np.zeros(0)
        kk = [nos[j] for j in np.nonzero(Us > 0.012 * Us.sum())[0]] if len(Us) else []; gain = float(Us.sum() / max(sum(Us[nos.index(k)] for k in kk), 1e-12)) if kk else 1.0; kept_lights.append(len(kk))      # the lights that matter here; the rest is made up by their share
        cols = kk + isun; vo = np.concatenate([vis[:, cols], vb], 1); r, good = C.read(P[i2], N[i2], vo, ok); bad = np.nonzero(~good)[0]; nc = len(cols)
        if len(bad): r[bad] = vo[C.inv[i2[bad]]]; lost += len(bad)
        ed = np.zeros((len(i2), 3), np.float32); dm2 = dmesh[i2]
        for q, k in enumerate(kk):
            l = L[k]; e = r[:, q] * g4.geom(l, P[i2], N[i2])
            if l.get("mesh", -1) >= 0 and l["kind"] == "iso": e = e * np.where(dm2 == l["mesh"], 0.03, 1.0)
            ed += e[:, None] * l["col"]
        Edir[i2] = ed * gain; Esoft[i2] = r[:, nc:nc + 3] + A * r[:, nc + 3:nc + 4] * amb_c; aux[i2] = r[:, nc + 3:nc + 5]
        if isun: sunv[i2] = r[:, len(kk):nc].max(1)
    tm["discs_s"] = round(time.time() - t0, 2); tm["lights_per_thing"] = round(float(np.mean(kept_lights)), 1) if kept_lights else 0
    # ---- sun and moon: every disc that may see it asks with its own rays (crisp bars; soft only with distance)
    Esun = np.zeros((n, 3), np.float32); sv = np.zeros(n, np.float32)
    for k in isun:
        l = L[k]; d0 = l["dir"].astype(np.float32); cs = N @ d0; cand = np.nonzero((cs > 0.02) & (sunv > 0.002))[0]; tm["sun_discs"] = int(len(cand))
        if not len(cand): continue
        a_ = np.where(abs(d0[1]) < 0.9, np.float32([0, 1, 0]), np.float32([1, 0, 0])); t1 = np.cross(d0, a_); t1 /= np.linalg.norm(t1); t2 = np.cross(d0, t1); rad = np.radians(o["sun_deg"]) / 2
        offs = [(0.0, 0.0)] + [(0.75 * np.cos(q), 0.75 * np.sin(q)) for q in np.arange(6) * np.pi / 3 + 0.3]; acc = np.zeros(len(cand), np.float32); O = P[cand] + 0.012 * N[cand]
        for (ox, oy) in offs:
            D = d0 + rad * (ox * t1 + oy * t2); D = (D / np.linalg.norm(D)).astype(np.float32); t, mi = S.cast(O, np.broadcast_to(D, O.shape)); free = ~np.isfinite(t)
            if l.get("openings"):
                thr = np.zeros(len(O), bool)
                for op in l["openings"]:
                    nn = np.asarray(op["n"], np.float32); den = float(D @ nn)
                    if den > -1e-6: continue
                    oc = np.asarray(op["c"], np.float32); tt = ((oc - O) @ nn) / den; Xo = O + D * tt[:, None] - oc; u, v = np.asarray(op["u"], np.float32), np.asarray(op["v"], np.float32)
                    thr |= (tt > 0) & (np.abs(Xo @ u) <= 1.3 * (u @ u)) & (np.abs(Xo @ v) <= 1.3 * (v @ v))
                free &= thr
            acc += free
        sv[cand] = acc / len(offs); Esun[cand] = (sv[cand] * cs[cand])[:, None] * l["col"]
    tm["sun_s"] = round(time.time() - t0, 2)
    # ---- contact: where a thing meets the floor or another thing, the discs ask again with their own short rays (the cells are too coarse for a stool's leg)
    obj_m = np.array([not str(m.get("name", "")).startswith("shell:") for m in meshes] + [False]); bb = [(np.asarray(m["V"]).min(0) - 0.25, np.asarray(m["V"]).max(0) + 0.25) for k, m in enumerate(meshes) if obj_m[k]]
    nearo = np.zeros(n, bool)
    cand0 = np.nonzero(aux[:, 1] < 0.992)[0]
    for lo_, hi_ in bb:
        q = cand0[~nearo[cand0]]; nearo[q[((P[q] > lo_) & (P[q] < hi_)).all(1)]] = True
    ci_ = np.nonzero(nearo)[0]; tm["contact_discs"] = int(len(ci_))
    if len(ci_):
        K_ = 12; u1 = (np.arange(K_) + 0.5) / K_; ph = np.arange(K_) * 2.399963; rr_ = np.sqrt(u1); loc = np.stack([rr_ * np.cos(ph), rr_ * np.sin(ph), np.sqrt(1 - u1)], 1).astype(np.float32)
        for a in range(0, len(ci_), 400_000):
            ii = ci_[a:a + 400_000]; nn = N[ii]; ax_ = np.where(np.abs(nn[:, :1]) < 0.9, np.float32([[1.0, 0, 0]]), np.float32([[0, 1.0, 0]])); tt_ = np.cross(nn, ax_); tt_ /= np.linalg.norm(tt_, axis=1, keepdims=True); bb_ = np.cross(nn, tt_)
            D = loc[None, :, 0, None] * tt_[:, None] + loc[None, :, 1, None] * bb_[:, None] + loc[None, :, 2, None] * nn[:, None]; O = np.broadcast_to((P[ii] + 0.006 * nn)[:, None], D.shape)
            t, _ = S.cast(O.reshape(-1, 3), D.reshape(-1, 3)); fine = np.clip(t / o["contact"], 0, 1).reshape(len(ii), K_).mean(1); aux[ii, 1] = 0.7 * fine + 0.3 * np.minimum(aux[ii, 1], fine)
    ac = np.clip(aux[:, 1:2], 0, 1); E = (Edir + Esun) * (0.5 + 0.5 * ac) + Esoft * ac ** 1.7; del Edir, Esoft
    tm["contact_s"] = round(time.time() - t0, 2)
    # ---- exposure: one number. The meter reads half the light that falls and half what comes back, and aims at the key of the time of day
    rE = shade(Ed, Es, Eind, AO, AC); ry = np.maximum(rE @ YW, 1e-9); gy = float(np.exp((np.log(ry) * w).sum() / w.sum())); gr = o["grade"]
    con = lambda e: e * (np.clip(np.maximum(e @ YW, 1e-9) / gy, 0.03, 6) ** gr)[:, None]
    uneven = float(np.log((ry * w).sum() / w.sum() / gy)); wk = float(np.clip((uneven - 0.2) / 0.5, 0, 1)); key = o["key"] if "key" in (opts or {}) or "key" in mood else KEY_EVEN[tm_] + (o["key"] - KEY_EVEN[tm_]) * wk
    rl = np.sqrt(ralb * 0.3) * con(rE); lo, hi = 1e-3, 1e4; tm["uneven"] = round(uneven, 3)
    for _ in range(40):
        x = (lo * hi) ** 0.5; y = float(((l2s(film(rl * x)) @ YW) * w).sum() / w.sum())
        if y < key: lo = x
        else: hi = x
    Blin = alb * con(E)                                        # what leaves each disc, before the camera
    # ---- the sources themselves
    X = []; emw = np.zeros(n, np.float32); emc = np.zeros((n, 3), np.float32); emd = np.zeros((n, 3), np.float32); info_src = []
    for m in models:
        col = m["col"] / wbc; col = (col / col.max()).astype(np.float32)
        if m["kind"] in ("lamp", "candle"):
            sel = np.nonzero(dmesh == m["mesh"])[0]; cpt = m["c"].astype(np.float32); rs = m["rs"]; s_src = x * 0.6 * m["flux"] / (4 * np.pi * (rs * rs + (0.5 * rs) ** 2))
            if len(sel):
                d2 = ((P[sel] - cpt) ** 2).sum(1); yy = alb[sel] @ YW; lam = cls[sel] == LAMPGLASS
                if lam.any(): wq = lam.astype(np.float32)                                  # the paint part said which discs are lamp glass
                else: q3, q8 = np.quantile(yy, [0.35, 0.8]); wq = np.clip((yy - q3) / max(q8 - q3, 1e-4), 0, 1)        # else: the lighter part of the thing is its glass
                if m["kind"] == "candle": wq = wq * np.clip(1 - np.sqrt(d2) / 0.07, 0, 1)   # only the top of the wax glows
                Lg = x * 0.6 * m["flux"] / (4 * np.pi * (d2 + (0.5 * rs) ** 2)); emw[sel] = np.maximum(emw[sel], wq * np.clip(Lg / 1.5, 0, 1)); u = np.clip(np.log10(1 + Lg) / 2.0, 0, 1)
                u0, u1 = np.quantile(u, [0.05, 0.99]); v = np.clip((u - u0) / (u1 - u0), 0, 1) if u1 - u0 > 0.08 else np.full(len(u), 0.7, np.float32); v = v[:, None]; lvl = float(np.clip(u1 / 0.75, 0.4, 1.0)); cs = col ** 1.5; cs = cs / cs.max()
                deep = cs ** 2.2 * 0.55; amber = cs; white = 0.4 * cs + 0.6 * np.float32([1.0, 0.93, 0.74]); ramp = np.where(v < 0.55, deep + (amber - deep) * (v / 0.55), amber + (white - amber) * np.clip((v - 0.55) / 0.45, 0, 1) ** 1.5) * lvl
                emd[sel] = ramp * (0.7 + 0.3 * yy[:, None] / max(float(np.quantile(yy, 0.95)), 1e-3)).clip(0, 1.1); emc[sel] = Lg[:, None] * col
            if m["flame"] > 0 or m["kind"] == "candle": flame(X, cpt, m["flame"] if m["flame"] > 0 else 1.0)
            st = float(np.clip(0.33 * np.log10(1 + s_src), 0, 1)) * o["halo"]; halo(X, cpt, rs if m["kind"] != "candle" else 0.02, col, st); info_src.append(f"{m['kind']} core {np.round(cpt, 2).tolist()} glow {st:.2f}")
        elif m["kind"] == "fire":
            H = m["H"]; fire_extras(X, m, seed=int(abs(H["uc"]) * 1000) % 9973); cpt = np.zeros(3); cpt[H["ot"]] = H["uc"]; cpt[1] = H["y0"] + 0.3 * H["h"]; cpt[H["ax"]] = H["front"] - H["sg"] * (0.5 * H["depth"] if m["hollow"] else -0.1)
            st = float(np.clip(0.33 * np.log10(1 + x * m["flux"] / (4 * np.pi * 0.04)), 0, 1)) * o["halo"]; halo(X, cpt, 0.3 * H["w"], col * [1, 0.8, 0.55], 0.8 * st)
            info_src.append(f"fire {'in a hollow %.2f x %.2f, %.2f deep' % (H['w'], H['h'], H['depth']) if m['hollow'] else 'NO HOLLOW FOUND: laid against the front'} glow {st:.2f}" + (" (a sculpted lump stands in the opening: painted as the fire)" if H.get("island") else ""))
            isl = H.get("island")
            if isl is not None:
                sel = np.nonzero(dmesh == m["mesh"])[0]; iu = np.floor((P[sel, H["ot"]] - isl["u0"]) / isl["du"]).astype(int); iv = np.floor((isl["v1"] - P[sel, 1]) / isl["dv"]).astype(int); nvv, nuu = isl["mask"].shape
                inb = (iu >= 0) & (iu < nuu) & (iv >= 0) & (iv < nvv); dpt = (H["front"] - P[sel, H["ax"]]) * H["sg"]; body = np.zeros(len(sel), bool); body[inb] = isl["mask"][iv[inb], iu[inb]]; body &= (dpt > -0.08) & (dpt < H["depth"] - 0.03); sel = sel[body]
                if len(sel):
                    hsh = lambda q: np.modf(np.abs(np.sin((np.floor(P[sel] / q) * np.float32([12.9898, 78.233, 37.719])).sum(1)) * 43758.5453))[0].astype(np.float32)
                    tt = np.clip((P[sel, 1] - isl["ya"]) / max(isl["yb"] - isl["ya"], 0.05) + 0.22 * (hsh(0.03) - 0.5) + 0.25 * np.abs(P[sel, H["ot"]] - isl["uc"]) / max(isl["w"], 0.05), 0, 1)[:, None]
                    c_hot, c_mid, c_tip = np.float32([1.0, 0.84, 0.42]), np.float32([1.0, 0.40, 0.05]), np.float32([0.78, 0.15, 0.02]); fc = np.where(tt < 0.45, c_hot + (c_mid - c_hot) * (tt / 0.45), c_mid + (c_tip - c_mid) * ((tt - 0.45) / 0.55))
                    fc = fc * np.where(hsh(0.012) < 0.10, 0.55, 1.0)[:, None] * (0.85 + 0.3 * hsh(0.006))[:, None]; emw[sel] = 1.0; emd[sel] = np.clip(fc, 0, 1); emc[sel] = fc * 4.0
    lit_lin = Blin * x * (1 - emw[:, None]) + emc * emw[:, None]
    disp = film(Blin * x) * (1 - emw[:, None]) + emd * emw[:, None]
    # sky through glass, light panels, flames flagged by the assembler: they shine with their own picture
    sk = np.nonzero(cls == SKY)[0]; ops = [e for e in emitters if e.get("cls") == "opening"]
    if len(sk):
        hgt = np.full(len(sk), 0.5, np.float32)
        if ops:
            oc = np.array([e["c"] for e in ops], np.float32); j = ((P[sk][:, None] - oc[None]) ** 2).sum(-1).argmin(1); vv = np.array([np.asarray(e["v"], np.float32) for e in ops])[j]
            up = np.abs(vv[:, 1]) > 0.5 * np.linalg.norm(vv, axis=1); hgt = np.where(up, np.clip(0.5 + 0.5 * (P[sk, 1] - oc[j, 1]) / np.maximum(np.abs(vv[:, 1]), 1e-3), 0, 1), 0.8)
        disp[sk] = np.float32(SKY_LOW[tm_]) * (1 - hgt[:, None]) ** 1.5 + np.float32(SKY_TOP[tm_]) * (1 - (1 - hgt[:, None]) ** 1.5)
    pn = np.nonzero(cls == PANEL)[0]
    if len(pn): disp[pn] = film(np.broadcast_to(np.float32([1.0, 0.95, 0.86]) / wbc * np.clip(6.0 * x, 1.5, 40), (len(pn), 3)).copy())
    fm = np.nonzero((cls == FLAME) & emit)[0]
    if len(fm): disp[fm] = np.float32([1.0, 0.86, 0.5])
    tm["sources_s"] = round(time.time() - t0, 2)
    # ---- shine
    opac = np.asarray(discs.get("opacity", np.full(n, 0.97)), np.float32).copy(); keep = np.ones(n, bool); sh = None; gsel = np.nonzero((gls > 0.05) & np.isin(cls, (GLOSS, METAL, GLASS)) & ~emit & (emw < 0.3))[0]
    gsel = gsel[np.argsort(dmesh[gsel] * 4096 + np.floor(P[gsel, 0] / 4).astype(np.int64) % 64 * 64 + np.floor(P[gsel, 2] / 4).astype(np.int64) % 64, kind="stable")]; tm["glossy_discs"] = int(len(gsel)); obj_m = np.array([not str(m.get("name", "")).startswith("shell:") for m in meshes] + [False])
    cen = np.zeros((len(meshes) + 1, 3), np.float32)
    for k, m in enumerate(meshes): V = np.asarray(m["V"]); cen[k] = (V.min(0) + V.max(0)) / 2
    if len(gsel) and (o["reflect"] or o["sh"]):
        # the eye a mirror picture is baked for: where a walker would look from (out from the thing's middle, 1.6 m away, at eye height)
        Pg, Ng = P[gsel], N[gsel]; hdir = (Pg - cen[dmesh[gsel]]) * [1, 0, 1] + 0.6 * Ng * [1, 0, 1]; hl = np.linalg.norm(hdir, axis=1, keepdims=True); hdir = np.where(hl > 1e-3, hdir / np.maximum(hl, 1e-6), np.float32([1, 0, 0]))
        eye = Pg + 1.6 * hdir; eye[:, 1] = 1.6; V_ = eye - Pg; V_ /= np.linalg.norm(V_, axis=1, keepdims=True); nv = (Ng * V_).sum(1); low = nv < 0.12; V_[low] += Ng[low] * (0.12 - nv[low])[:, None]; V_ /= np.linalg.norm(V_, axis=1, keepdims=True); nv = (Ng * V_).sum(1)
        Rr = (2 * nv[:, None] * Ng - V_).astype(np.float32); isobj = obj_m[dmesh[gsel]]; cg = cls[gsel]; g1 = gls[gsel]
        # (1) what ONE reflection ray reads where it lands: the lit level, coarse (the cells the bounce used); the sky if it leaves through a window
        pos_live = np.full(C.n, -1, np.int64); pos_live[live] = np.arange(len(live)); dl = pos_live[C.inv]; okd = (dl >= 0) & ~emit; row = CC.inv[dl[okd]]; Bw = np.where(emw[okd, None] > 0.3, emc[okd] / x, Blin[okd])
        Bcc = np.stack([np.bincount(row, Bw[:, q], CC.n) for q in range(3)], 1) / np.maximum(np.bincount(row, minlength=CC.n), 1)[:, None]; Bmean = Bw.mean(0); del Bw, row, dl, okd
        sky_B = (np.float32(SKY_LOW[tm_]) * 0.5 + np.float32(SKY_TOP[tm_]) * 0.5); sky_B = sky_B / np.maximum(1 - sky_B, 0.05) / x * 1.5      # the sky as bright as it is shown
        O = Pg + 0.012 * Ng; t, mi, hn = S.cast(O, Rr, normals=True); fin = np.isfinite(t); Xh = O + Rr * np.where(fin, t, 0)[:, None]; hn = hn * -np.sign((hn * Rr).sum(1, keepdims=True)); Xh = Xh + 0.004 * hn
        refl, hit = CC.read(Xh, hn.astype(np.float32), Bcc.astype(np.float32), np.ones(CC.n, bool)); refl = np.where((hit & fin)[:, None], refl, Bmean).astype(np.float32); refl[~fin] = sky_B; skyhit = ~fin; del O, Xh, hn      # read BETWEEN the coarse cells: a blocky mirror picture shows on a car's bonnet
        # (2) the lights themselves: a still highlight for that eye (a sharp lobe, worked out, so it is smooth) and harmonics that move with the eye
        SA = np.exp(-np.arange(4) * (np.arange(4) + 1) * 0.06).astype(np.float32); peak = float(sum((2 * l + 1) / (4 * np.pi) * SA[l] for l in (1, 2, 3))); mz = np.float32([1, 1, -1])
        co = np.zeros((len(gsel), 15, 3), np.float32); dc = np.zeros((len(gsel), 3), np.float32); spec = np.zeros((len(gsel), 3), np.float32); avgB = float(Bmean @ YW)
        kap = np.where(cg == METAL, 260.0, np.where(cg == GLASS, 400.0, 30 + 170 * g1 ** 2)).astype(np.float32); f0a = np.where(cg == METAL, 0.7, np.where(cg == GLASS, 0.10, 0.06)).astype(np.float32); used = []
        for a in range(0, len(gsel), 250_000):
            sl = slice(a, a + 250_000); ii = gsel[sl]; Pn, Nn = Pg[sl], Ng[sl]; f0 = f0a[sl]; sub = np.arange(0, len(ii), max(1, len(ii) // 2500))
            Isub = np.array([float((g4.geom(l, Pn[sub], Nn[sub], to=True)[0]).mean()) * float(l["col"] @ YW) for l in L]) if L else np.zeros(0); Isub[isun] = 0; kk = [int(k) for k in np.argsort(-Isub)[:12] if Isub[k] > 0.02 * Isub.max()] + isun; used.append(len(kk))      # the dozen lights that matter to this thing, and the sun
            cols = kk; r, good = C.read(Pn, Nn, vis[:, cols], ok); r[~good] = vis[:, cols][C.inv[ii[~good]]]
            tint = np.where((cg[sl] == METAL)[:, None], 0.35 + 0.65 * alb[ii] / np.maximum(alb[ii].max(1, keepdims=True), 0.05), 1.0).astype(np.float32); cc_ = np.zeros((len(ii), 15, 3), np.float32); d0 = np.zeros((len(ii), 3), np.float32); Is, Os, Cs = [], [], []
            for q, k in enumerate(kk):
                l = L[k]; f, om, cr = g4.geom(l, Pn, Nn, to=True); vk = sv[ii] if k in isun else r[:, q]; F = f0 + (1 - f0) * (1 - np.clip(cr, 0, 1)) ** 5; I = vk * f * np.clip((cr + 0.05) / 0.2, 0, 1) * F
                if l.get("mesh", -1) >= 0: I = I * (dmesh[ii] != l["mesh"])
                if o["reflect"]:                               # the still highlight: a lobe about the mirror direction of the baked eye, as bright as so narrow a lobe makes it
                    ca = (Rr[sl] * om).sum(1); D = np.exp(kap[sl] * (ca - 1)) * kap[sl] / (2 * np.pi); spec[sl] += (I * D * np.where(isobj[sl], 1.0, 0.0))[:, None] * l["col"] * tint
                Is.append((I * g1[sl] * float(l["col"] @ YW)).astype(np.float32)); Os.append(om); Cs.append(cr)
            if o["sh"] and kk:
                Im = np.max(Is, 0)
                for q, k in enumerate(kk):
                    l = L[k]; lv_ = (Is[q] > 0.3 * Im) & (Is[q] > 1e-9)
                    if not lv_.any(): continue
                    rr = 2 * Cs[q][lv_, None] * Nn[lv_] - Os[q][lv_]; c3 = (Is[q][lv_] / float(l["col"] @ YW))[:, None] * l["col"] * tint[lv_]
                    cc_[lv_] += (sh_basis(-rr * mz) * SA[SH_DEG])[:, :, None] * c3[:, None, :]; d0[lv_] += c3 * (1 / (4 * np.pi))
                sc = x * o["spec"] / peak; cc_ *= sc; d0 *= sc
                pk = (np.abs(cc_).sum(1).max(1) * 0.4 + 1e-6); cap = np.where(cg[sl] == METAL, 0.5, np.where(cg[sl] == GLASS, 0.4, 0.12 + 0.2 * g1[sl])); f = cap * np.tanh(pk / cap) / pk; cc_ *= f[:, None, None]; d0 *= f[:, None]
                # rim: surfaces seen at a glancing angle mirror the room more (sin^2 of the angle: degree 0 and 2 about the disc's facing)
                Rm = (np.where(cg[sl] == METAL, 0.06, np.where(cg[sl] == GLASS, 0.22, 0.10)) * g1[sl] * min(x * avgB * 2.2, 0.5)).astype(np.float32) * np.clip(aux[ii, 0], 0.2, 1)
                Y2 = sh_basis(-Nn * mz)[:, 3:8]; cc_[:, 3:8] += (-(2 / 3) * (4 * np.pi / 5) * Rm)[:, None, None] * Y2[:, :, None] * tint[:, None, :]; d0 += (2 / 3) * Rm[:, None] * tint
                co[sl] = cc_; dc[sl] = d0
            del Is, Os, Cs
        tm["shine_lights"] = round(float(np.mean(used)), 1) if used else 0
        if o["reflect"]:
            Fr = 0.04 + 0.96 * (1 - np.clip(nv, 0, 1)) ** 5; g_ = g1[:, None]; base = lit_lin[gsel]; tint = alb[gsel] / np.maximum(alb[gsel].max(1, keepdims=True), 0.05); rx = refl * x; sp = spec * x; ob = isobj[:, None]
            win = np.where((skyhit & isobj)[:, None], rx, 0.0)                                       # a window seen in the paint
            out = base + (Fr[:, None] * 0.28 * g_) * rx * ob + 0.4 * g_ * win + sp                   # gloss paint: the colour stays deep, a clear coat mirrors over it
            mt = cg == METAL; out[mt] = (0.12 * base + g_ * (0.35 + 0.65 * tint) * rx * 0.9 + (1 - g_) * base + sp)[mt]          # metal: mostly what it mirrors, in its own colour
            gs = cg == GLASS; gi = np.nonzero(gs)[0]
            if len(gi):                                        # glass: one dark pane (the middle colour of that thing's glass, not the smears painted on it), and what it mirrors
                am = alb[gsel[gi]]; mid_ = np.zeros_like(am); dmg = dmesh[gsel[gi]]
                for k in np.unique(dmg): q = dmg == k; mid_[q] = np.median(am[q], 0)
                bg = base[gi] / np.maximum(am, 0.004) * mid_; tg = mid_ / np.maximum(mid_.max(1, keepdims=True), 0.05); out[gi] = 0.22 * bg * (0.4 + 0.6 * tg) + (0.05 + 0.8 * Fr[gi, None]) * rx[gi] + 0.6 * win[gi] + sp[gi]
            disp[gsel] = film(out); ya = alb[gsel[gs]] @ YW; opac[gsel[gs]] = np.minimum(opac[gsel[gs]], np.clip(0.92 - 0.4 * np.sqrt(np.median(ya) if len(ya) else 0.1), 0.7, 0.92))      # ... and a hint of what is behind (clear glass more than dark)
        if o["sh"]:
            base_s = l2s(disp[gsel]); base_s = np.clip(base_s + dc * 0.3, 0, 1); disp[gsel] = s2l(base_s * 255); sh = {"idx": gsel, "coef": np.clip(co, -1, 1).astype(np.float16)}
    tm["shine_s"] = round(time.time() - t0, 2)
    rgb = np.clip(l2s(disp) * 255 + 0.5, 0, 255).astype(np.uint8)
    # ---- new discs: flames, embers, glow; and flat mirrors (a mirror image hung behind the wall)
    extra = None
    if X:
        extra = {"xyz": np.stack([b[0] for b in X]), "sigma": np.stack([b[1] for b in X]), "rgb": np.stack([b[2] for b in X]), "opacity": np.float32([b[3] for b in X]), "normal": np.stack([b[4] for b in X])}
    if o["mirror"] and (cls == MIRROR).any() and "sigma" in discs and "thick" in discs:
        import mirror
        extra_m, keep, minfo = mirror.mirrors(discs, P, N, cls, rgb, opac, keep, extra); tm["mirror"] = minfo
        if extra_m is not None: extra = extra_m if extra is None else {k: np.concatenate([extra_m[k], extra[k]]) for k in extra}
    tm["mirror_s"] = round(time.time() - t0, 2)
    tm.update(total_s=round(time.time() - t0, 2), rays_M=round(S.n_rays / 1e6, 1), discs=n, expo=round(float(x), 3), read_fallback=int(lost), time=tm_, key=key, sources=info_src, extra_discs=0 if extra is None else len(extra["xyz"]),
              classes={CLASSES[k]: int((cls == k).sum()) for k in range(9) if (cls == k).any()})
    return {"rgb": rgb, "opacity": opac, "keep": keep, "extra": extra, "sh": sh, "info": tm, "cls": cls, "expo": float(x)}

# ================================================================ writer
def rows(discs, res, placing):
    """The lit discs (+ extras) as splat rows in the file's frame: (mirror_z(p) - m) * k. -> rows (m, 14), index of each kept disc"""
    m = np.asarray(placing["m"], float); k = float(placing["k"]); keep = res["keep"]; idx = np.nonzero(keep)[0]; P = np.asarray(discs["xyz"], np.float64)[idx]; N = np.asarray(discs["normal"], np.float64)[idx]
    a = np.zeros((len(idx), 14), "<f4"); a[:, :3] = (P * [1, 1, -1] - m) * k; a[:, 3:6] = (res["rgb"][idx].astype(np.float32) / 255 - 0.5) / C0; op = np.clip(res["opacity"][idx], 0.01, 0.999); a[:, 6] = np.log(op / (1 - op))
    a[:, 7] = a[:, 8] = np.log(np.asarray(discs["sigma"])[idx] * k); a[:, 9] = np.log(np.asarray(discs["thick"])[idx] * k); a[:, 10:14] = quat_z_to(N * [1, 1, -1])
    ex = res.get("extra")
    if ex is not None:
        b = np.zeros((len(ex["xyz"]), 14), "<f4"); b[:, :3] = (ex["xyz"].astype(np.float64) * [1, 1, -1] - m) * k; b[:, 3:6] = (ex["rgb"] - 0.5) / C0; op = np.clip(ex["opacity"], 0.004, 0.999); b[:, 6] = np.log(op / (1 - op))
        b[:, 7:10] = np.log(np.maximum(ex["sigma"], 1e-5) * k); b[:, 10:14] = quat_z_to(ex["normal"].astype(np.float64) * [1, 1, -1]); a = np.concatenate([a, b])
    return a, idx

def write(out, discs, res, placing):
    a, idx = rows(discs, res, placing); cols = PLY14; info = {}; sh = res.get("sh")
    for f in (out, out[:-4] + "_sh.ply"):
        if os.path.exists(f): os.remove(f)
    if sh is not None and len(sh["idx"]):
        row = np.full(len(res["keep"]) + 1, -1, np.int64); row[idx] = np.arange(len(idx)); r = row[sh["idx"]]; good = r >= 0; r = r[good]; co = sh["coef"][good].astype(np.float32)
        b = np.zeros((len(r), 59), "<f4"); b[:, :6] = a[r, :6]; b[:, 6:51] = co.transpose(0, 2, 1).reshape(len(r), 45); b[:, 51:] = a[r, 6:]; cols2 = cols[:6] + [f"f_rest_{i}" for i in range(45)] + cols[6:]
        rest = np.ones(len(a), bool); rest[r] = False; write_ply(out, a[rest], cols); write_ply(out[:-4] + "_sh.ply", b, cols2); info.update(sh_discs=len(r), sh_mb=round(os.path.getsize(out[:-4] + "_sh.ply") / 1e6))
    else: write_ply(out, a, cols)
    info.update(discs=len(a), mb=round(os.path.getsize(out) / 1e6)); return info
