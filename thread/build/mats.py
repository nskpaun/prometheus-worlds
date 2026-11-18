"""The materials of a built world, read by the assembler. numpy + PIL + scipy only; no model.
    M = mats.Mats(world)                             # reads WORLD/materials/{surfaces.json, style.json, tex/}
  SURFACES (walls, floor, ceiling: laid for this room, registered to it)
    rgb = M.colour("x1", u, v, spacing, flip_v_height=True)      u, v = the shell mesh's own UV in metres
    rgb = M.surface_at("x1", P, spacing)                         or straight from room points P (n, 3)
  ARCHITECTURE PIECES (every other shell mesh)
    rgb = M.arch(mesh, P, N, spacing, seed)          mesh = the shell's record (group, piece); P, N room points and normals (n, 3)
        group wall / floor / ceiling pieces (reveals, a chimney breast, a pier): the wall's own material by HEIGHT, so courses and the lower/upper split carry
        across from the wall beside them; trim groups: their swatch (a stone/plaster group in a room of that stuff = the wall's own material); wood runs along the piece.
        Every piece takes the wall's wear by height (dirt rising from the floor, smoke under the ceiling), so trim does not sit clean on a worn wall.
  PLAIN FORMS (forms.py's things: every PART its own material, flat things their content)
    Pt = M.plain(kind, o, V, F, part, names)         o = the layout object, V (n, 3) its placed mesh in room metres, F faces, part per face, names
    rgb, emit = Pt.colour(part_name, P, N, spacing)  P, N room points and normals on that part
    Pt.info(part_name) -> {cls, material, finish, mirror, emit}          cls "mirror": a reflective class for the light (not painted)
  M.info("x1") / M.info("arch:metal") -> {cls, finish, material}        for the light's sheen rules;  M.style = style.json
Colours are float 0..255, the material's own colour evenly lit. `spacing` (metres between discs) picks a pre-shrunk copy so nothing sparkles."""
import os, json, math, zlib
import numpy as np
from PIL import Image
GREY = np.float32([0.299, 0.587, 0.114])
WEAR = {"pristine": 0.25, "light": 0.55, "worn": 1.0, "heavy": 1.5}
def _sd(*a): return zlib.crc32("|".join(str(x) for x in a).encode()) % (2 ** 31)

class Tex:
    """A true-scale picture read at metric coordinates; pre-shrunk (half-octave steps) to the disc spacing. mode: mirror (tiles without a seam) | clamp."""
    def __init__(s, img, ppm, mode="mirror"):
        s.img = np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img, np.uint8); s.ppm = float(ppm); s.mips = {}; s.mode = mode; s._tone = None; s._det = None
    def level(s, spacing):
        want = min(s.ppm, 1.15 / max(spacing, 1e-4)); lvl = int(np.clip(np.floor(np.log2(s.ppm / want) * 2 + 0.5), 0, 14))
        if lvl not in s.mips:
            f = 2 ** (lvl / 2); h, w = s.img.shape[:2]; s.mips[lvl] = (np.asarray(Image.fromarray(s.img).resize((max(2, int(w / f)), max(2, int(h / f))), Image.BOX)) if lvl else s.img, s.ppm / f)
        return s.mips[lvl]
    def at(s, u, v, spacing):
        """Colours at metres (u right, v down from the picture's top-left)."""
        im, ppm = s.level(spacing); h, w = im.shape[:2]; x = np.asarray(u, np.float64) * ppm; y = np.asarray(v, np.float64) * ppm
        if s.mode == "mirror":
            x = x % (2 * w); y = y % (2 * h); x = np.where(x >= w, 2 * w - 1e-6 - x, x); y = np.where(y >= h, 2 * h - 1e-6 - y, y)
        return im[np.clip(y.astype(int), 0, h - 1), np.clip(x.astype(int), 0, w - 1)].astype(np.float32)
    def at01(s, a, b, spacing):
        """Colours at shares of the picture (a across 0..1, b down 0..1)."""
        im, ppm = s.level(spacing); h, w = im.shape[:2]; return im[np.clip((np.asarray(b) * h).astype(int), 0, h - 1), np.clip((np.asarray(a) * w).astype(int), 0, w - 1)].astype(np.float32)
    def tone(s):
        if s._tone is None: s._tone = np.median(s.img[::7, ::7].reshape(-1, 3), 0).astype(np.float32)
        return s._tone
    def detail_spacing(s, levels):
        """The coarsest disc spacing (one of levels, metres) that still loses less than a just-visible share of this picture's contrast."""
        if s._det is None:
            h, w = s.img.shape[:2]; n = min(h, w, 1024); a = s.img[(h - n) // 2:(h + n) // 2, (w - n) // 2:(w + n) // 2].astype(np.float32) @ GREY; im = Image.fromarray(a.astype(np.uint8)); thr = 0.012 * (float(a.mean()) + 30); out = levels[0]
            for g in levels:
                f = g * s.ppm
                if f > 1.2:
                    lo = np.asarray(im.resize((max(1, round(n / f)), max(1, round(n / f))), Image.BOX).resize((n, n), Image.BICUBIC), np.float32)
                    if np.sqrt(((lo - a) ** 2).mean()) >= thr: break
                out = g
            s._det = float(out)
        return s._det

def _hash3(P, seed):
    """Smooth-ish noise 0..1 from room points (cheap, continuous): for wear on pieces."""
    a = np.sin(P[:, 0] * 2.1 + seed) * np.cos(P[:, 2] * 1.7 - seed * 0.7) + 0.5 * np.sin(P[:, 1] * 5.3 + P[:, 0] * 3.1 + seed * 1.3) + 0.25 * np.sin((P[:, 0] + P[:, 2]) * 11.0 + seed)
    return np.clip(0.5 + a / 3.5, 0, 1).astype(np.float32)

class Mats:
    def __init__(s, world):
        s.dir = os.path.join(world, "materials"); s.S = json.load(open(f"{s.dir}/surfaces.json")); s.style = json.load(open(f"{s.dir}/style.json")); s.tex = {}; s.room = np.float32(s.S["room"])
        for n, r in s.S["surfaces"].items(): s.tex[n] = Tex(Image.open(f"{s.dir}/{r['texture']}"), r["px_per_m"], "clamp")
        for g, r in s.S["groups"].items(): s.tex["g:" + g] = Tex(Image.open(f"{s.dir}/{r['texture']}"), r["px_per_m"])
        for k in ("wall_lower", "wall_upper", "floor", "ceiling"):
            p = f"{s.dir}/tex/sheet_{k}.png"
            if os.path.exists(p): s.tex["sheet:" + k] = Tex(Image.open(p), 512.0)
        s._files = {}; s.wear = WEAR.get(s.style.get("wear"), 0.55); s.night = s.style.get("time") == "night"
        w0 = next((r for r in s.S["surfaces"].values() if r["kind"] == "wall"), None); s.split = float(w0.get("split_m") or 0) if w0 else 0.0
    def file(s, rel, ppm, mode="mirror"):
        if rel not in s._files: s._files[rel] = Tex(Image.open(f"{s.dir}/{rel}"), ppm, mode)
        return s._files[rel]
    # ---- surfaces
    def colour(s, key, u, v, spacing, tint=None, seed=0, flip_v_height=None):
        T = s.tex.get(key) or s.tex.get("g:plaster") or next(iter(s.tex.values())); r = s.S["surfaces"].get(key)
        if r is not None:                                   # a registered surface: the shell's own UV -> its texel
            x = np.asarray(u, np.float64) - r["u0"]; y = (r["size_m"][1] - (np.asarray(v, np.float64) - r["v0"])) if r["v_up"] else (np.asarray(v, np.float64) - r["v0"]); c = T.at(x, y, spacing)
        else:
            rng = np.random.default_rng(int(seed) * 7919 + 13); c = T.at(np.asarray(u) + rng.uniform(0, 7), np.asarray(v) + rng.uniform(0, 7), spacing)
        if tint is not None: c = c * np.clip((np.float32(tint) + 6) / (T.tone() + 6), 0.45, 2.2)
        return np.clip(c, 0, 255)
    def surface_at(s, key, P, spacing):
        r = s.S["surfaces"][key]; uv = np.c_[np.asarray(P, np.float64), np.ones(len(P))] @ np.asarray(r["uv_fit"]).T; return s.colour(key, uv[:, 0], uv[:, 1], spacing)
    def spec(s, id_, fallback="g:plaster"):
        """(texture key, tint or None, emit) for "arch:GROUP", without the height rules."""
        g = id_.split(":", 1)[-1]
        return ("g:" + g if "g:" + g in s.tex else fallback), None, False
    def info(s, id_):
        if id_ in s.S["surfaces"]: z = s.S["surfaces"][id_]["zones"]; return {"cls": "surface", "material": z[0]["material"], "finish": z[0].get("finish"), "zones": z}
        g = s.S["groups"].get(id_.split(":", 1)[-1]) or {}; return {"cls": g.get("cls"), "material": g.get("material"), "finish": g.get("finish") or ("satin" if g.get("cls") in ("metal", "wood") else "matte")}
    # ---- architecture pieces
    def height_wear(s, P, seed=0):
        """The wall's wear by height, for any piece that stands on or against it: dirt rising from the floor, smoke gathered under the ceiling at night."""
        y = P[:, 1]; n = _hash3(P, seed % 97); g = 1 + s.wear * 0.07 * (n - 0.5) - s.wear * 0.16 * np.exp(-np.maximum(y, 0) / (0.22 + 0.12 * n)) * (0.6 + 0.4 * n)
        if s.night: g -= s.wear * 0.14 * np.exp(-np.maximum(s.room[1] - y, 0) / (0.35 + 0.15 * n)) * (0.6 + 0.4 * n)
        return np.clip(g, 0.4, 1.2)[:, None]
    def arch(s, mesh, P, N, spacing, seed=0):
        P = np.asarray(P, np.float32); N = np.asarray(N, np.float32); g = mesh["group"]; rec = s.S["groups"].get(g) or {}; ax = np.abs(N).argmax(1); rng = np.random.default_rng(int(seed) * 7919 + 13); ou, ov = rng.uniform(0, 7, 2)
        hu = np.where(ax == 0, P[:, 2], P[:, 0])                                   # along the piece, level
        if g in ("wall", "floor", "ceiling") and rec.get("mode") == "sheet":
            if g == "wall":
                H = float(s.room[1]); up = P[:, 1]; low = up < s.split if s.split > 0 else np.zeros(len(P), bool); c = np.empty((len(P), 3), np.float32)
                TL = s.tex.get("sheet:wall_lower"); TU = s.tex.get("sheet:wall_upper") or s.tex["g:wall"]
                flat = ax == 1; hv = np.where(flat, P[:, 2], 0.0)                 # a level face of a wall piece (a sill, a head): laid flat in the same stuff
                if low.any() and TL is not None: c[low] = TL.at(hu[low] + ou, np.where(flat[low], hv[low], s.split - up[low]), spacing)
                elif low.any(): c[low] = TU.at(hu[low] + ou, H - up[low], spacing)
                hi = ~low
                if hi.any(): c[hi] = TU.at(hu[hi] + ou, np.where(flat[hi], hv[hi], H - up[hi]), spacing)
            else: c = s.tex["g:" + g].at(P[:, 0] + ou, P[:, 2] + ov, spacing)
            return np.clip(c * s.height_wear(P, seed), 0, 255)
        T = s.tex.get("g:" + g) or s.tex.get("g:trim") or next(iter(s.tex.values())); V = np.asarray(mesh["V"], np.float32); ext = V.max(0) - V.min(0)
        u = hu; v = np.where(ax == 1, P[:, 2], P[:, 1])
        if rec.get("grain"):                                                       # wood runs along the piece's longer side
            long_ax = int(ext.argmax()); along = P[:, long_ax]; other = np.where(ax == (long_ax + 1) % 3, P[:, (long_ax + 2) % 3], P[:, (long_ax + 1) % 3]); u, v = along, other
        c = T.at(u + ou, v + ov, spacing)
        return np.clip(c * s.height_wear(P, seed), 0, 255)
    # ---- plain forms
    def plain(s, kind, o, V, F=None, part=None, names=None): return Painter(s, kind, o, V, F, part, names)

BOOKS = [(120, 30, 28), (30, 44, 86), (36, 70, 48), (92, 58, 30), (60, 26, 30), (140, 110, 60), (48, 40, 62), (160, 140, 105), (70, 22, 20), (24, 60, 66)]
class Painter:
    """One placed plain form. Own frame: x across (-w/2..w/2), y up (0..h), z towards its front (+d/2), as the layout part's y_proc.make."""
    def __init__(s, M, kind, o, V, F=None, part=None, names=None):
        s.M = M; s.kind = kind; s.rec = (M.S.get("parts") or {}).get(kind) or {"form": o.get("form"), "parts": {}, "content": {"type": "none"}}; s.form = s.rec.get("form") or o.get("form"); s.o = o
        V = np.asarray(V, np.float64); s.yaw = float(o.get("yaw") or 0.0); lo, hi = V.min(0), V.max(0); s.c = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]); Vo = s.own(V); s.lo, s.hi = Vo.min(0), Vo.max(0); s.size = s.hi - s.lo
        s.seed = _sd(kind, o.get("name"), np.round(s.c, 2).tolist()); s.pb = {}; s.books = None; s.names = [str(n) for n in names] if names is not None else ["body"]
        if F is not None and part is not None:
            F = np.asarray(F); part = np.asarray(part)
            for k, nm in enumerate(s.names):
                idx = np.unique(F[part == k])
                if len(idx): s.pb[nm] = (Vo[idx].min(0), Vo[idx].max(0))
            if s.form == "books":
                from scipy.sparse import coo_matrix
                from scipy.sparse.csgraph import connected_components
                e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]]]); n, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))), directed=False)
                used = np.unique(lab[np.unique(F)]); s.books = np.array([np.r_[Vo[lab == l].min(0), Vo[lab == l].max(0)] for l in used])
    def own(s, P):
        d = np.asarray(P, np.float64) - s.c; cs, sn = math.cos(s.yaw), math.sin(s.yaw); return np.stack([cs * d[:, 0] - sn * d[:, 2], d[:, 1], sn * d[:, 0] + cs * d[:, 2]], 1)
    def own_n(s, N):
        N = np.asarray(N, np.float64); cs, sn = math.cos(s.yaw), math.sin(s.yaw); return np.stack([cs * N[:, 0] - sn * N[:, 2], N[:, 1], sn * N[:, 0] + cs * N[:, 2]], 1)
    def prec(s, part):
        P = s.rec.get("parts") or {}; return P.get(part) or P.get(part.split()[0]) or (P.get("cover") if part.startswith("book") else None) or (next(iter(P.values())) if P else {"mode": "flat", "rgb": [128, 124, 118], "cls": "other", "material": ""})
    def info(s, part):
        r = s.prec(part); c = s.rec.get("content") or {}; mirror = c.get("type") == "mirror" and part == "picture"
        return {"cls": "mirror" if mirror else r.get("cls"), "material": "silvered mirror glass" if mirror else r.get("material"), "mirror": mirror, "emit": r.get("mode") == "emit",
                "finish": "gloss" if mirror or r.get("cls") in ("glass",) else "satin" if r.get("cls") in ("metal", "leather", "ceramic", "wax") else "matte"}
    def swatch(s, part, Po, No, spacing):
        r = s.prec(part)
        if r.get("mode") != "swatch": return np.broadcast_to(np.float32(r.get("rgb", [128, 124, 118])), (len(Po), 3)).copy()
        T = s.M.file(r["texture"], r["px_per_m"]); ax = np.abs(No).argmax(1); u = np.where(ax == 0, Po[:, 2], Po[:, 0]); v = np.where(ax == 1, Po[:, 2], Po[:, 1])
        if r.get("cls") in ("wood", "fabric", "rope", "wicker") and part in s.pb:
            ext = s.pb[part][1] - s.pb[part][0]; la = int(ext.argmax())
            if r.get("cls") == "fabric": la = 0 if s.form == "rug" else 1          # a hanging cloth's pile and weave run up and down
            along = Po[:, la]; other = np.where(ax == (la + 1) % 3, Po[:, (la + 2) % 3], Po[:, (la + 1) % 3]); u, v = (along, other) if r.get("cls") != "fabric" else (other, along)
        rng = np.random.default_rng(s.seed + _sd(part)); c = T.at(u + rng.uniform(0, 5), v + rng.uniform(0, 5), spacing)
        return c * np.clip((np.float32(r["rgb"]) + 6) / (T.tone() + 6), 0.5, 2.0)
    def colour(s, part, P, N, spacing):
        """-> (rgb (n, 3) float 0..255, emit bool)"""
        r = s.prec(part); Po = s.own(P); No = s.own_n(N); c = s.rec.get("content") or {"type": "none"}; ty = c.get("type")
        if r.get("mode") == "emit": return np.broadcast_to(np.float32(r["rgb"]), (len(Po), 3)).copy(), True
        w, h, d = s.size; x01 = (Po[:, 0] - s.lo[0]) / max(w, 1e-6); y01 = (Po[:, 1] - s.lo[1]) / max(h, 1e-6); z01 = (Po[:, 2] - s.lo[2]) / max(d, 1e-6)
        if s.form == "rug" and ty == "pattern" and c.get("texture"):
            T = s.M.file(c["texture"], c["px"][0] / max(c["size_m"][0], 1e-6), "clamp"); col = T.at01(x01, 1 - z01, spacing)
            fr = s.prec("field")
            if fr.get("mode") == "swatch":                # the wool's own pile over the woven picture (a large rug's picture is coarser than its wool)
                W = s.M.file(fr["texture"], fr["px_per_m"]); g = W.at(Po[:, 0] + 3.1, Po[:, 2] + 1.7, spacing) @ GREY; col = col * np.clip(g / max(float(W.tone() @ GREY), 1.0), 0.8, 1.2)[:, None]
            side = np.abs(No[:, 1]) < 0.5; col[side] *= 0.6; return np.clip(col, 0, 255), False
        if s.form == "frame" and part == "picture":
            if ty == "mirror": return np.broadcast_to(np.float32(c.get("rgb", [200, 205, 208])), (len(Po), 3)).copy(), False
            if ty == "picture" and c.get("texture") and "picture" in s.pb:
                lo, hi = s.pb["picture"]; T = s.M.file(c["texture"], c["px"][0] / max(c["size_m"][0], 1e-6), "clamp")
                return T.at01(1 - (Po[:, 0] - lo[0]) / max(hi[0] - lo[0], 1e-6), 1 - (Po[:, 1] - lo[1]) / max(hi[1] - lo[1], 1e-6), spacing), False          # 1 - x: the room frame is left-handed, so a thing's own +x is on the left of someone facing its front
        if s.form == "cloth" and part == "cloth":
            if ty == "picture" and c.get("texture"):
                T = s.M.file(c["texture"], c["px"][0] / max(c["size_m"][0], 1e-6), "clamp"); return T.at01(1 - x01, 1 - y01, spacing), False
            col = s.swatch(part, Po, No, spacing)
            if ty == "trim":                                # a woven braid above the hem: two narrow lines and a patterned band between them
                yb = Po[:, 1] - s.lo[1]; band = (yb > 0.07) & (yb < 0.15); line = ((yb > 0.05) & (yb < 0.062)) | ((yb > 0.158) & (yb < 0.17))
                weave = 0.75 + 0.25 * np.sign(np.sin(Po[:, 0] * 180) * np.sin((yb - 0.07) * 180)); g = (col @ GREY) / max(float(np.median(col @ GREY)), 1.0); t = np.float32(c.get("rgb", [190, 150, 60]))
                col[band] = (t * weave[band, None] * np.clip(g[band, None], 0.7, 1.3)); col[line] = t * 0.85 * np.clip(g[line, None], 0.7, 1.3)
            return np.clip(col, 0, 255), False
        if s.form == "books" and s.books is not None and len(s.books): return s.book_colour(Po, No, spacing), False
        col = s.swatch(part, Po, No, spacing)
        if ty == "label" and c.get("texture") and part in ("panel", "side", "body", "top") and s.form in ("crate", "board", "cylinder"):
            T = s.M.file(c["texture"], 1024.0, "clamp"); lw = min(0.62 * w, 0.5); lh = lw * 0.375; cy = s.lo[1] + (0.55 * h if s.form != "board" else 0.5 * h)
            if s.form == "board" and h < lh: lw = h / 0.375 * 0.8; lh = lw * 0.375
            a = 1 - (Po[:, 0] + lw / 2) / lw; b = 1 - (Po[:, 1] - (cy - lh / 2)) / lh; ok = (No[:, 2] > 0.6) & (a > 0) & (a < 1) & (b > 0) & (b < 1)
            if ok.any(): al = T.at01(a[ok], b[ok], spacing)[:, :1] / 255 * 0.82; col[ok] = col[ok] * (1 - al) + np.float32([28, 26, 24]) * al
        return np.clip(col, 0, 255), False
    def book_colour(s, Po, No, spacing):
        B = s.books; cen = (B[:, :3] + B[:, 3:]) / 2; half = (B[:, 3:] - B[:, :3]) / 2 + 1e-4; dd = (np.abs(Po[:, None, :] - cen[None]) / half[None]).max(-1); bi = dd.argmin(1); lo, hi = B[bi, :3], B[bi, 3:]; ext = hi - lo
        pal = [np.float32(c_) for c_ in ((s.rec.get("content") or {}).get("colours") or [])]; pal = [np.clip(p * (0.55 if p @ GREY > 110 else 0.85), 20, 200) for p in pal] + [np.float32(b_) for b_ in BOOKS]
        rng = np.random.default_rng(s.seed); pick = rng.integers(0, len(pal), len(B)); base = np.stack(pal)[pick][bi] * rng.uniform(0.8, 1.15, len(B))[bi, None]
        standing = ext[:, 1] >= ext[:, 0]; t_ax = np.where(standing, 0, 1); l_ax = np.where(standing, 1, 0); ar = np.arange(len(Po))
        sp = (Po[ar, l_ax] - lo[ar, l_ax]) / np.maximum(ext[ar, l_ax], 1e-6); tp = (Po[ar, t_ax] - lo[ar, t_ax]) / np.maximum(ext[ar, t_ax], 1e-6); nax = np.abs(No).argmax(1)
        r = s.prec("cover"); lea = np.ones(len(Po), np.float32)
        if r.get("mode") == "swatch": T = s.M.file(r["texture"], r["px_per_m"]); g = T.at(Po[:, 0] * 3 + bi * 0.37, Po[:, 1] * 3 + bi * 0.61, spacing / 3) @ GREY; lea = np.clip(g / max(float(T.tone() @ GREY), 1.0), 0.6, 1.4)
        col = base * lea[:, None]; spine = (nax == 2) & (No[:, 2] > 0); cover = nax == t_ax; pages = ~spine & ~cover
        gold = np.float32([186, 150, 72]); hb = rng.integers(0, 4, len(B))[bi]; gb = np.zeros(len(Po), bool)
        for f0 in (0.1, 0.9): gb |= np.abs(sp - f0) < 0.008
        gb |= (hb > 0) & ((np.abs(sp - 0.22) < 0.006) | (np.abs(sp - 0.78) < 0.006)); gb |= (hb > 1) & (np.abs(sp - 0.5) < 0.005)
        lab = (sp > 0.62) & (sp < 0.76) & (tp > 0.12) & (tp < 0.88) & (hb != 2); labc = np.clip(base * 0.45 + 12, 0, 255); title = lab & (np.sin(sp * 420) > 0.2) & (tp > 0.22) & (tp < 0.78)
        raised = np.zeros(len(Po), np.float32)
        for f0 in (0.3, 0.42, 0.56): raised += np.exp(-((sp - f0) / 0.006) ** 2)
        c2 = col * (1 + 0.18 * raised[:, None] * (hb[:, None] % 2)); c2[lab] = labc[lab] * lea[lab, None]; c2[gb | title] = gold * np.clip(lea[gb | title, None], 0.8, 1.2); col[spine] = c2[spine]
        edge = (tp < 0.06) | (tp > 0.94); pg = np.float32([214, 200, 168]) * (0.88 + 0.12 * np.sin(Po[:, t_ax[0] if len(t_ax) else 0] * 2600))[:, None] * np.clip(lea[:, None], 0.9, 1.1); col[pages & ~edge] = pg[pages & ~edge]
        return np.clip(col, 0, 255)
