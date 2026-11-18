"""Laying. Every wall, the floor and the ceiling of a layout's room as one true-scale texture, laid from photographed pieces.
CPU only (numpy, PIL, scipy). Nothing here knows a world.
  THE IDEA  the picture model photographs each material ONCE, close up, at 512 px a metre: a 2 m square whose pieces (blocks, bricks, boards, flags, tiles) were
            drawn first, so every piece's place in the photograph is known. The photograph is cut into its pieces (a LIBRARY of real photographed
            units, each with its own joints) and the room's faces are LAID from that library on a plan drawn for the real room: courses that run
            round every corner at one height, whole units from each corner, tiles and panels fitted between corners, piers and openings.
            So a face is sharp everywhere (the swatch's own 512 px a metre), nothing is mirror-tiled, and no two rooms share a picture.
            A material without pieces (plaster, paint, poured concrete) is spread from its photograph in overlapping patches taken at random
            places (no repeat to see), and what makes it a big surface is added after: see WEAR.
  ZONES     a wall has one treatment or two (style.json walls.split_m): lower below, upper above. The split snaps to a kit rail if one is near.
  PATTERN   a repeated motif (stencil, wallpaper) is ONE photographed motif stamped on a half-drop lattice at its true repeat, the same heights on every wall.
  WEAR      large-scale variation as paint, from the room's own geometry: grime rising from the floor, scuffs above the skirting, soot above a fire,
            runs under window sills, damp in corners, darkening under the ceiling by candle light; on floors a worn path from each door, dirt along
            the walls, ash by a hearth, tyre marks from the widest door to each large thing, stains. Strength by style.json wear.
"""
import os, sys, json, math, zlib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import tools as C
SW_M, SW_PPM = 2.0, 512                     # a material's photograph: metres square, px a metre
PPM = {"wall": 512, "floor": 384, "ceil": 192}
MAX_PX = 2.6e7                              # most pixels in one stored texture
WEAR_PPM = 48
WEAR = {"pristine": 0.25, "light": 0.55, "worn": 1.0, "heavy": 1.5}
WALLS = ("x0", "x1", "z0", "z1")
INTO = {"x0": (1, 0, 0), "x1": (-1, 0, 0), "z0": (0, 0, 1), "z1": (0, 0, -1)}
LINED = ("plank", "board", "floorboard", "slat", "strip")
COURSED = ("block", "stone", "ashlar", "flagstone", "flag", "paver")
def sd(*a): return zlib.crc32("|".join(str(x) for x in a).encode()) % (2 ** 31)

# ---------------------------------------------------------------- the room's faces from the shell
def faces(S, layout):
    """-> {name: face}. A face: kind wall|floor|ceil, w, h (metres), A (2x4: room xyz1 -> the shell's own uv), u0, v0 (uv of the texture's origin),
    rects: kit pieces that touch it [{piece, group, box (x0, y0, x1, y1) in texture metres, depth}], things: layout objects that stand against / on it,
    holes: a mask at 32 px a metre of where the face is cut away (openings).  Texture metres: x right, y DOWN from the top-left."""
    room = np.array(S["room"], float); F = {}
    for m in S["meshes"]:
        sname = m.get("surface")
        if m["piece"] not in ("wall", "floor", "ceiling") or not sname: continue
        V = np.asarray(m["V"], float); UV = np.asarray(m["UV"], float); A = np.linalg.lstsq(np.c_[V, np.ones(len(V))], UV, rcond=None)[0].T
        kind = {"wall": "wall", "floor": "floor", "ceiling": "ceil"}[m["piece"]]
        if kind == "wall": w = room[2] if sname[0] == "x" else room[0]; h = room[1]
        else: w, h = room[0], room[2]
        c = np.array([[0, 0, 0], [room[0], 0, 0], [0, 0, room[2]], [room[0], 0, room[2]], [0, room[1], 0], [room[0], room[1], room[2]]], float); uvc = np.c_[c, np.ones(len(c))] @ A.T
        f = dict(name=sname, kind=kind, w=float(w), h=float(h), A=A, u0=float(np.round(uvc[:, 0].min(), 6)), v0=float(np.round(uvc[:, 1].min(), 6)), rects=[], things=[])
        hp = 32; hm = Image.new("L", (int(round(w * hp)), int(round(h * hp))), 0); dr = ImageDraw.Draw(hm); xy = to_tex(f, V) * hp
        for t in np.asarray(m["F"]): dr.polygon([tuple(xy[i]) for i in t], fill=255)
        f["solid"] = np.asarray(hm) > 127; F[sname] = f
    for m in S["meshes"]:
        if m["piece"] in ("wall", "floor", "ceiling") and m.get("surface"): continue
        V0 = np.asarray(m["V"], float)
        for n, f in F.items():
            dv = plane_d(f, V0, room); V = V0[dv < (0.6 if f["kind"] == "wall" else 0.45)]          # a piece that runs round the room (skirting, a rail) is taken wall by wall: only what is near this face
            if len(V) < 3 or dv.min() > 0.03: continue
            lo, hi = V.min(0), V.max(0); d0, d1 = dist(f, lo, hi, room)
            xy = to_tex(f, np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]))
            b = [float(np.clip(xy[:, 0].min(), 0, f["w"])), float(np.clip(xy[:, 1].min(), 0, f["h"])), float(np.clip(xy[:, 0].max(), 0, f["w"])), float(np.clip(xy[:, 1].max(), 0, f["h"]))]
            if b[2] - b[0] > 0.01 and b[3] - b[1] > 0.005: f["rects"].append({"piece": m["piece"], "group": m["group"], "box": b, "depth": float(d1)})
    claimed = set(S.get("claimed") or [])
    for i, o in enumerate(layout.get("objects") or []):
        if i in claimed or not o.get("prims"): continue
        lo, hi = obj_bounds(o)
        for n, f in F.items():
            d0, d1 = dist(f, lo, hi, room)
            if d0 > (0.3 if f["kind"] == "wall" else 0.05): continue
            xy = to_tex(f, np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]))
            f["things"].append({"i": i, "name": o.get("name", ""), "kind": o.get("kind", ""), "box": [float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())], "height": float(hi[1] - lo[1]), "depth": float(d1 - d0)})
    return F
def obj_bounds(o):
    lo, hi = np.full(3, 1e9), np.full(3, -1e9)
    for p in o["prims"]:
        c = np.array(p["c"], float)
        if p.get("shape") == "cyl": h = np.array([p["r"], p["hy"], p["r"]], float)
        else:
            h = np.array(p["h"], float); yaw = float(p.get("yaw") or 0); cs, sn = abs(math.cos(yaw)), abs(math.sin(yaw)); h = np.array([cs * h[0] + sn * h[2], h[1], sn * h[0] + cs * h[2]])
        lo = np.minimum(lo, c - h); hi = np.maximum(hi, c + h)
    return lo, hi
def plane_d(f, V, room):
    n = f["name"]
    return {"x0": V[:, 0], "x1": room[0] - V[:, 0], "z0": V[:, 2], "z1": room[2] - V[:, 2]}.get(n, V[:, 1] if f["kind"] == "floor" else room[1] - V[:, 1])
def dist(f, lo, hi, room):
    """Nearest and farthest distance of a box from a face's plane."""
    n = f["name"]
    if n == "x0": return max(lo[0], 0.0), hi[0]
    if n == "x1": return max(room[0] - hi[0], 0.0), room[0] - lo[0]
    if n == "z0": return max(lo[2], 0.0), hi[2]
    if n == "z1": return max(room[2] - hi[2], 0.0), room[2] - lo[2]
    if f["kind"] == "floor": return max(lo[1], 0.0), hi[1]
    return max(room[1] - hi[1], 0.0), room[1] - lo[1]
def to_tex(f, P):
    """Room points -> texture metres (x right, y down) on face f."""
    uv = np.c_[np.asarray(P, float), np.ones(len(P))] @ f["A"].T
    x = uv[:, 0] - f["u0"]; y = (f["h"] - (uv[:, 1] - f["v0"])) if f["kind"] == "wall" else (uv[:, 1] - f["v0"])
    return np.stack([x, y], 1)
def rails(f):
    """Kit pieces that run the whole width of a wall as a thin level band: [(y top, y bottom, piece)] in texture metres."""
    return sorted((r["box"][1], r["box"][3], r["piece"]) for r in f["rects"] if r["box"][2] - r["box"][0] > 0.8 * f["w"] and r["box"][3] - r["box"][1] < 0.45)
def breaks(f):
    """Places along a wall where a fitted layout (tiles, panels, boards) starts again: corners, pier and breast edges, opening jambs. -> sorted metres"""
    b = {0.0, f["w"]}
    for r in f["rects"]:
        x0, y0, x1, y1 = r["box"]
        if y1 - y0 > 0.7 * f["h"] and x1 - x0 < 0.6 * f["w"]: b.update((x0, x1))
    s = f["solid"]; col = (~s).sum(0) > 0.25 * s.shape[0]; e = np.flatnonzero(np.diff(col.astype(int))) + 1
    b.update(float(x) / 32 for x in e); b = sorted(b); out = [b[0]]
    for x in b[1:]:
        if x - out[-1] > 0.25: out.append(x)
    out[-1] = f["w"]; return out

# ---------------------------------------------------------------- plans: cells in metres (x0, y0, x1, y1), y down
def unit_of(z):
    u = str(z.get("unit") or "none").lower().rstrip("s"); cm = sorted(float(v) for v in (z.get("unit_cm") or [0, 0])[:2])
    if u == "none" or not cm or cm[-1] <= 0: return "none", 0.0, 0.0
    a, b = cm[0] / 100, cm[1] / 100
    if a <= 0: a = b
    if u in LINED: return "plank", float(np.clip(a, 0.07, 0.4)), float(np.clip(b, 0.6, 4.0))
    if u == "brick": return "brick", float(np.clip(a, 0.05, 0.12)) + 0.01, float(np.clip(b, 0.15, 0.4)) + 0.01
    if u in COURSED: return ("block" if u in ("block", "stone", "ashlar") else "flag"), float(np.clip(a, 0.15, 1.2)), float(np.clip(b, a, 1.6))
    if u in ("tile", "panel"): return u, float(np.clip(a, 0.05, 1.5)), float(np.clip(b, 0.05, 2.4))
    if u == "slab": return ("slab" if a >= 1.0 else "tile"), float(np.clip(a, 0.2, 4.0)), float(np.clip(b, 0.2, 4.0))
    return "tile", float(np.clip(a, 0.05, 1.5)), float(np.clip(b, 0.05, 2.4))
def course_edges(h, ch, seed, vary):
    """Course edges from the bottom (0) up to h: whole courses from the floor, heights varied by `vary`."""
    rng = np.random.default_rng(seed); e = [0.0]
    while e[-1] < h - 1e-6: e.append(e[-1] + ch * (1 + vary * rng.uniform(-1, 1)))
    if len(e) > 2 and h - e[-2] < 0.4 * ch: e.pop(-2)
    e[-1] = h; return e
def cells_coursed(w, h, ch, ul, seed, vary=0.12, lv=0.35, bond=0.5, k0=0):
    """Level courses from the bottom; in each, units of varied length from a whole unit at the left, alternate courses set off by `bond`."""
    rng = np.random.default_rng(seed); out = []; e = course_edges(h, ch, seed, vary)
    for k in range(len(e) - 1):
        y0, y1 = h - e[k + 1], h - e[k]; x = -ul * bond if (k + k0) % 2 else 0.0; row = []
        while x < w - 1e-6:
            ln = ul * (1 + lv * rng.uniform(-1, 1)); row.append([max(x, 0.0), min(x + ln, w)]); x += ln
        if len(row) > 1 and row[-1][1] - row[-1][0] < 0.3 * ul: row[-2][1] = row[-1][1]; row.pop()
        out += [(a, y0, b, y1) for a, b in row if b - a > 1e-4]
    return out
def cells_grid(w, h, uw, uh, segs=None, from_bottom=True):
    """A grid fitted to each segment along x (whole units between its ends); rows whole from the bottom."""
    segs = segs or [0.0, w]; out = []; ny = max(1, int(round(h / uh))); ys = np.linspace(0, h, ny + 1) if abs(h / ny - uh) < 0.25 * uh else None
    if ys is None:
        e = list(np.arange(0, h, uh)) + [h]; ys = np.array([h - v for v in e][::-1]) if from_bottom else np.array(e)
    for a, b in zip(segs, segs[1:]):
        n = max(1, int(round((b - a) / uw))); xs = np.linspace(a, b, n + 1)
        out += [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for i in range(n) for j in range(len(ys) - 1)]
    return out
def cells_planks(w, h, uw, ul, seed, along_x, segs=None):
    """Boards side by side, butt joints at random lengths (never longer than the photographed board)."""
    rng = np.random.default_rng(seed); ul = min(ul, 0.92 * SW_M); out = []
    if along_x:
        n = max(1, int(round(h / uw))); ys = np.linspace(0, h, n + 1)
        for k in range(n):
            x = -rng.random() * ul
            while x < w:
                ln = ul * rng.uniform(0.55, 1.0); out.append((max(x, 0.0), ys[k], min(x + ln, w), ys[k + 1])); x += ln
    else:
        segs = segs or [0.0, w]
        for a, b in zip(segs, segs[1:]):
            n = max(1, int(round((b - a) / uw))); xs = np.linspace(a, b, n + 1)
            for k in range(n):
                if h <= ul: out.append((xs[k], 0.0, xs[k + 1], h)); continue
                y = h + rng.random() * ul * 0.5
                while y > 0:
                    ln = ul * rng.uniform(0.55, 1.0); out.append((xs[k], max(y - ln, 0.0), xs[k + 1], min(y, h))); y -= ln
    return [c for c in out if c[2] - c[0] > 1e-4 and c[3] - c[1] > 1e-4]
def plan_cells(z, w, h, kind, seed, segs=None, k0=0):
    """The laying plan of a zone w x h metres. -> (unit kind, cells, joint width m)"""
    u, a, b = unit_of(z)
    if u in ("none", "slab"): return u, [], 0.0
    if u == "brick": return u, cells_coursed(w, h, a, b, seed, vary=0.0, lv=0.0, k0=k0), 0.01
    if u == "block": return u, cells_coursed(w, h, a, b, seed, vary=0.14, lv=0.35, k0=k0), 0.008
    if u == "flag": return u, cells_coursed(w, h, a, b, seed, vary=0.2, lv=0.35, bond=0.37, k0=k0), 0.01
    if u == "plank": return u, cells_planks(w, h, a, b, seed, along_x=(w >= h) if kind != "wall" else False, segs=segs), 0.003
    if kind == "wall":
        if u == "panel": a, b = min(a, b), max(a, b)                      # a wall panel stands upright
        return u, cells_grid(w, h, a, b if u == "panel" else a, segs), 0.004 if u == "tile" else 0.006
    return u, cells_grid(w, h, a, b if abs(a - b) > 0.02 else a, None, from_bottom=False), 0.004

# ---------------------------------------------------------------- a material's photograph: the drawing Klein keeps, and the library cut from it
JOINT = {"brick": "pale", "block": "pale", "flag": "dark", "plank": "dark", "tile": "dark", "panel": "dark"}
def swatch_plan(z, kind, seed):
    """-> (unit, cells in the swatch (metres), reference picture (float, SW px), words about the drawing)"""
    n = int(SW_M * SW_PPM); rgb = np.float32(z["rgb"]); u, cells, jw = plan_cells(z, SW_M, SW_M, "floor" if kind != "wall" or u_lined(z) else "wall", seed)
    if u == "plank": cells = cells_grid(SW_M, SW_M, SW_M, unit_of(z)[1])           # whole boards, the full 2 m long, lying along x
    ref = np.ones((n, n, 3), np.float32) * rgb; rng = np.random.default_rng(seed + 1)
    if not cells: return u, [], ref, ""
    line = np.zeros((n, n), np.float32); lw = max(5, int(round(jw * SW_PPM)))          # drawn a little wider than life (the reference is half size): Klein keeps what it can see
    for x0, y0, x1, y1 in cells:
        a, b, c, d = [int(round(v * SW_PPM)) for v in (x0, y0, x1, y1)]; ref[b:d, a:c] *= rng.uniform(0.86, 1.14)
        line[b:d, max(a - lw // 2, 0):a + (lw + 1) // 2] = 1; line[b:d, c - lw // 2:c + (lw + 1) // 2] = 1; line[max(b - lw // 2, 0):b + (lw + 1) // 2, a:c] = 1; line[d - lw // 2:d + (lw + 1) // 2, a:c] = 1
    pale = JOINT.get(u) == "pale" and float(rgb @ C.GREY) < 150
    jc = np.clip(0.5 * rgb + 0.5 * np.float32([205, 198, 185]), 0, 255) if pale else rgb * 0.3
    ref = ref * (1 - line[..., None]) + jc * line[..., None]
    nm = {"brick": "brick", "block": "stone block", "flag": "flagstone", "plank": "board", "tile": "tile", "panel": "panel"}[u]; cnt = len({round(c[1], 3) for c in cells})
    if u == "plank": w = f" The picture shows {len(cells)} {nm}s lying side by side, each running the full width of the picture from left to right, each about {unit_of(z)[1] * 100:.0f} cm wide."
    elif u in ("brick", "block", "flag"): w = f" The {nm}s are laid in {cnt} level courses, each course about {unit_of(z)[1] * 100:.0f} cm high."
    else:
        cols = len({round(c[0], 3) for c in cells}); cw, chh = cells[0][2] - cells[0][0], cells[0][3] - cells[0][1]
        w = f" It is made of flat rectangular {nm}s butted edge to edge in a regular grid: {cols} {nm}s side by side across the picture and {cnt} rows of them from top to bottom, each {nm} about {cw * 100:.0f} cm wide and {chh * 100:.0f} cm high."
        if u == "panel": w += " The grain of every panel runs upright, from the bottom to the top."
    w += f" The thin {'pale' if pale else 'dark'} lines in the given picture are the real joints: keep every joint exactly where it is, straight, and add no other joints."
    return u, cells, ref, w
def u_lined(z): return unit_of(z)[0] == "plank"
BARE = "Nothing but the bare surface: no furniture, no shelves, no pictures, no doors, no windows, no lamps, no skirting, no objects, no people, no text, no cast shadows, no vignette."
def swatch_words(z, kind, place, extra):
    what = {"wall": "wall", "floor": "floor", "ceil": "ceiling"}.get(kind, kind)
    return (f"A close-up photograph taken straight on of part of the bare {what} of {place}: {z['material']}. The picture shows an area only {SW_M:g} metres wide and {SW_M:g} metres high, flat and head-on with no perspective, so grain, pores and joints are fine and small.{extra} "
            f"The same material and the same scale from edge to edge, evenly lit with soft flat shadowless light, no glare, no reflections, no dark corners. Keep the colours of the given picture. Sharp focus, crisp real material detail: grain, pores, chips, fine scratches, wear. {BARE}")
def even(a, rgb, sig, pull=0.7):
    """Evenly lit and on its guide colour: everything coarser than sig px is flattened, the middle colour pulled `pull` of the way to rgb."""
    lo = C.g3(a, sig, "reflect"); med = np.median(a[::4, ::4].reshape(-1, 3), 0); a = a * np.clip((med + 2) / (lo + 2), 0.4, 2.5)
    if rgb is not None: a = a * np.clip(((np.float32(rgb) + 4) / (med + 4)) ** pull, 0.4, 2.5)
    return np.clip(a, 0, 255)
def _peaks(p, min_gap, rel=0.3):
    """Places of the joints along a profile of edge strength: local maxima at least min_gap apart and at least `rel` of the strong ones."""
    p = ndimage.gaussian_filter1d(p.astype(np.float64), 2.0); base = np.median(p); top = np.percentile(p, 99) - base
    if top <= 1e-6: return []
    mx = (p == ndimage.maximum_filter1d(p, int(max(3, min_gap)) | 1)) & (p - base > rel * top); return [int(i) for i in np.flatnonzero(mx)]
def measure_units(sw, unit, uh, ul):
    """Klein keeps the SCALE of the drawn plan but lays the pieces its own way (courses shifted, a brick more or less), so the pieces are found in the
    photograph itself: level joints from the edge strength summed along x, then in each course the upright joints from that course's own edges.
    -> [(x0, y0, x1, y1) px] whole pieces only (cut along joint centres, so each carries half a joint all round)."""
    g = ndimage.gaussian_filter(sw @ C.GREY, 1.2); n = g.shape[0]; ey = np.abs(ndimage.sobel(g, 0)); ex = np.abs(ndimage.sobel(g, 1))
    def ends(pk, size):          # the photograph's own edge is a joint too when a whole piece fits between it and the first joint found (large pieces: a 2 m photograph holds only a few)
        pk = list(pk)
        if not pk: return [0, n] if 0.7 * size < n < 1.4 * size else []
        if 0.7 * size < pk[0] < 1.4 * size: pk = [0] + pk
        if 0.7 * size < n - pk[-1] < 1.4 * size: pk = pk + [n]
        return pk
    rows = ends(_peaks(ey.mean(1), 0.55 * uh * SW_PPM), uh * SW_PPM)
    if len(rows) < 2: return []
    out = []
    for y0, y1 in zip(rows, rows[1:]):
        if not (0.5 * uh * SW_PPM < y1 - y0 < 1.8 * uh * SW_PPM): continue
        m = max(2, (y1 - y0) // 6)
        if unit == "plank": out.append((0, y0, n, y1)); continue
        cols = ends(_peaks(ex[y0 + m:y1 - m].mean(0), 0.45 * ul * SW_PPM, 0.35), ul * SW_PPM)
        for x0, x1 in zip(cols, cols[1:]):
            if 0.45 * ul * SW_PPM < x1 - x0 < 2.2 * ul * SW_PPM: out.append((x0, y0, x1, y1))
    return out
class Library:
    """A photographed material: the swatch, and its pieces found in it and cut out."""
    def __init__(s, img, z, kind, seed):
        s.z = z; s.unit, s.cells, _, _ = swatch_plan(z, kind, seed); a = np.asarray(img, np.float32)
        if a.shape[0] != int(SW_M * SW_PPM): a = C.resize(a, int(SW_M * SW_PPM), int(SW_M * SW_PPM))
        s.sw = even(a, z["rgb"], 0.3 * SW_PPM); s.units = []; s.tone = np.median(s.sw[::4, ::4].reshape(-1, 3), 0); s.kept = None; s.loose = False; s.boxes = []
        if s.cells:
            _, uh, ul = unit_of(z)
            if s.unit in ("tile", "panel"): uh, ul = s.cells[0][3] - s.cells[0][1], s.cells[0][2] - s.cells[0][0]
            s.boxes = measure_units(s.sw, s.unit, uh, ul); want = len(s.cells); s.kept = round(len(s.boxes) / max(want, 1), 2)          # share of the drawn pieces found whole in the photograph
            for x0, y0, x1, y1 in s.boxes:
                u_ = s.sw[y0:y1, x0:x1]; s.units.append(np.clip(u_ * ((s.tone + 4) / (np.median(u_.reshape(-1, 3), 0) + 4)) ** 0.6, 0, 255))          # each keeps 40% of its own tone: a few pieces laid many times must not chequer
            if len(s.units) < max(3, 0.3 * want): s.units = []; s.loose = True          # too few found: pieces are cut at random places and their joints drawn (lay)
    def spread(s, shape, ppm, seed, patch_m=0.7):
        """The swatch spread over (h, w) at ppm with no repeat to see: overlapping patches from random places, blended so contrast is kept."""
        h, w = shape; k = ppm / SW_PPM; sw = s.sw if abs(k - 1) < 0.02 else C.resize(s.sw, round(s.sw.shape[1] * k), round(s.sw.shape[0] * k)); n = sw.shape[0]
        p = int(min(patch_m * ppm, 0.6 * n)); st = max(8, int(p * 0.62)); rng = np.random.default_rng(seed); mean = sw.reshape(-1, 3).mean(0)
        acc = np.zeros((h, w, 3), np.float32); w1 = np.zeros((h, w), np.float32); w2 = np.zeros((h, w), np.float32)
        r = np.clip(np.minimum(np.arange(p) + 0.5, p - np.arange(p) - 0.5) / (0.5 * (p - st) + 1e-6), 0, 1); r = (r * r * (3 - 2 * r)).astype(np.float32); win = np.outer(r, r) + 1e-4
        for y in range(-p // 3, h, st):
            for x in range(-p // 3, w, st):
                sy, sx = rng.integers(0, n - p), rng.integers(0, n - p); pt = sw[sy:sy + p, sx:sx + p]
                if rng.random() < 0.5: pt = pt[:, ::-1]
                ya, xa = max(y, 0), max(x, 0); yb, xb = min(y + p, h), min(x + p, w)
                if yb <= ya or xb <= xa: continue
                ww = win[ya - y:yb - y, xa - x:xb - x]; acc[ya:yb, xa:xb] += (pt[ya - y:yb - y, xa - x:xb - x] - mean) * ww[..., None]; w1[ya:yb, xa:xb] += ww; w2[ya:yb, xa:xb] += ww * ww
        return np.clip(mean + acc / np.sqrt(np.maximum(w2, 1e-8))[..., None], 0, 255)
    def lay(s, cells, shape, ppm, seed):
        """Cells (metres) laid with photographed units at ppm. Planks are cut to length from a photographed board; other units are fitted (stretch under a quarter)."""
        h, w = shape; out = np.zeros((h, w, 3), np.float32); rng = np.random.default_rng(seed); N = len(s.units); k = ppm / SW_PPM
        asp = np.array([u.shape[1] / u.shape[0] for u in s.units]) if N else None; last = -1; n = s.sw.shape[0]; jl = max(1, int(round(0.003 * ppm)))
        for x0, y0, x1, y1 in cells:
            a, b, c, d = [int(round(v * ppm)) for v in (x0, y0, x1, y1)]; a, b = max(a, 0), max(b, 0); c, d = min(c, w), min(d, h); tw, th = c - a, d - b
            if tw < 1 or th < 1: continue
            if not N:                                                      # no cut-out units: a piece cut from the photograph at a random place, its edge drawn
                upright = s.unit == "panel" and th > tw; cw_, ch_ = (th, tw) if upright else (tw, th); sw_, sh_ = int(min(n, max(4, round(cw_ / k)))), int(min(n, max(4, round(ch_ / k))))
                ox, oy = int(rng.integers(0, n - sw_ + 1)), int(rng.integers(0, n - sh_ + 1)); u = C.resize(s.sw[oy:oy + sh_, ox:ox + sw_], cw_, ch_)
                if upright: u = np.rot90(u)
                u = u.copy() * rng.uniform(0.93, 1.07); u[:jl] *= 0.5; u[-jl:] *= 0.5; u[:, :jl] *= 0.5; u[:, -jl:] *= 0.5; out[b:d, a:c] = u; continue
            if s.unit == "plank":
                upright = th > tw; L, Wd = (th, tw) if upright else (tw, th); u = s.units[int(rng.integers(N))]; need = int(min(u.shape[1], round(L / k)))
                o = int(rng.integers(0, u.shape[1] - need + 1)); u = u[:, o:o + need]
                if rng.random() < 0.5: u = u[::-1, ::-1]
                u = C.resize(u, L, Wd)
                if upright: u = np.rot90(u)
            else:
                ta = tw / th; cand = rng.integers(0, N, 5); cand = cand[cand != last] if (cand != last).any() else cand; j = int(cand[np.abs(np.log(asp[cand] / ta)).argmin()]); last = j; u = s.units[j]
                if u.shape[1] / u.shape[0] > 1.3 * ta:                     # much longer than the cell: its middle is cut, both ends (with their joints) kept
                    keep = int(u.shape[0] * ta * 1.15); u = np.concatenate([u[:, :keep // 2], u[:, -(keep - keep // 2):]], 1)
                if rng.random() < 0.5: u = u[:, ::-1]
                if rng.random() < 0.5 and s.unit != "panel": u = u[::-1]
                u = C.resize(u, tw, th)
            out[b:d, a:c] = u * rng.uniform(0.97, 1.03)
        return out
def fake_swatch(z, kind, seed):
    """A stand-in for Klein's photograph (the drawn plan with noise): for testing the laying without the model."""
    u, cells, ref, _ = swatch_plan(z, kind, seed); n = ref.shape[0]; return np.clip(ref * (1 + 0.08 * C.noise((n, n), 40, seed)[..., None] + 0.05 * np.random.default_rng(seed).standard_normal((n, n, 1))), 0, 255)

# ---------------------------------------------------------------- pattern: one photographed motif on a half-drop lattice
def motif_words(z):
    p = z["pattern"]; return (f"One single {p['motif']}, painted flat in the middle of a plain flat wall, seen straight on. Only this one motif, centred, about half as wide as the picture, with clean crisp edges; "
                              f"the rest of the picture is the plain flat background colour of the given picture, evenly lit, no shadow, no frame, no border, no other marks, no text.")
def motif_alpha(img, ground):
    """A motif picture -> (rgb, alpha): how far each pixel is from the picture's own ground colour (its border's middle colour)."""
    a = np.asarray(img, np.float32); e = max(4, a.shape[0] // 16); g = np.median(np.concatenate([a[:e].reshape(-1, 3), a[-e:].reshape(-1, 3), a[:, :e].reshape(-1, 3), a[:, -e:].reshape(-1, 3)]), 0)
    d = np.linalg.norm(a - g, axis=-1); thr = max(18.0, float(np.percentile(d, 60)) * 2.5); al = np.clip((d - thr) / (1.2 * thr), 0, 1); al = ndimage.gaussian_filter(al, 0.7)
    ys, xs = np.nonzero(al > 0.3)
    if len(ys) < 50: return None, None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1; return a[y0:y1, x0:x1], al[y0:y1, x0:x1]
def stamp(tex, z, motif, ppm, y_top_m, seed, wear=1.0):
    """The motif on a half-drop lattice at its true repeat; rows at the same heights on every wall (y_top_m: metres from the room's top to this picture's top)."""
    rgb, al = motif
    if rgb is None: return tex
    rep = z["pattern"]["repeat_cm"] / 100; size = 0.36 * rep; h, w = tex.shape[:2]; k = size * ppm / max(rgb.shape[:2]); mw, mh = max(2, int(round(rgb.shape[1] * k))), max(2, int(round(rgb.shape[0] * k)))
    m = C.resize(rgb, mw, mh); a = np.asarray(Image.fromarray((al * 255).astype(np.uint8)).resize((mw, mh), Image.LANCZOS), np.float32) / 255
    m = m * np.clip((np.float32(z["pattern"]["rgb"]) + 4) / (np.sum(m * a[..., None], (0, 1)) / max(a.sum(), 1) + 4), 0.5, 2.0) ** 0.6
    worn = np.clip(0.9 - wear * 0.28 * (C.noise((h // 4 + 1, w // 4 + 1), 0.25 * ppm / 4, seed) + 0.6), 0.25, 1.0); worn = np.asarray(Image.fromarray(worn).resize((w, h), Image.BILINEAR))
    nx = int(w / (rep * ppm)) + 2; off = (w - (nx - 1) * rep * ppm) / 2; row = 0; y = rep * 0.5 - (y_top_m % (rep * 0.5))
    r0 = int(round(y_top_m // (rep * 0.5)))
    while y * ppm < h + mh:
        for i in range(-1, nx + 1):
            cx = off + (i + (0.5 if (row + r0) % 2 else 0.0)) * rep * ppm; x0 = int(round(cx - mw / 2)); y0 = int(round(y * ppm - mh / 2)); xa, ya = max(x0, 0), max(y0, 0); xb, yb = min(x0 + mw, w), min(y0 + mh, h)
            if xb <= xa or yb <= ya: continue
            aa = (a[ya - y0:yb - y0, xa - x0:xb - x0] * worn[ya:yb, xa:xb])[..., None]; tex[ya:yb, xa:xb] = tex[ya:yb, xa:xb] * (1 - aa) + m[ya - y0:yb - y0, xa - x0:xb - x0] * aa
        y += rep * 0.5; row += 1
    return tex

# ---------------------------------------------------------------- wear: large-scale variation as paint, from the room's own geometry
def is_fire(t): return any(k in (t["name"] + " " + t["kind"]).lower() for k in ("fireplace", "hearth", "stove", "furnace", "forge", "brazier", "campfire", "oven"))
def wall_wear(f, style, seed):
    """-> (gain (h, w) at WEAR_PPM, warm (h, w): how much soot / grime colour), for a wall."""
    s = WEAR.get(style.get("wear"), 0.55); p = WEAR_PPM; h, w = int(round(f["h"] * p)), int(round(f["w"] * p)); yy, xx = np.mgrid[:h, :w].astype(np.float32) / p; up = f["h"] - yy
    n1 = C.noise((h, w), 1.6 * p, seed); n2 = C.noise((h, w), 0.35 * p, seed + 1); n3 = C.noise((h, w), 0.08 * p, seed + 2, 2)
    g = 1 + s * (0.07 * n1 + 0.035 * n2)                                           # the wall is never one tone
    sk = max([r["box"][3] for r in f["rects"] if r["box"][3] > f["h"] - 0.02 and r["box"][2] - r["box"][0] > 0.5 * f["w"]] + [f["h"]]); sk_h = f["h"] - min([r["box"][1] for r in f["rects"] if r["box"][3] > f["h"] - 0.02 and r["box"][2] - r["box"][0] > 0.5 * f["w"]] + [f["h"]])
    grime = np.exp(-np.maximum(up - sk_h, 0) / np.maximum(0.22 + 0.12 * n1, 0.06)) * (0.6 + 0.4 * n2); g *= 1 - s * 0.16 * np.clip(grime, 0, 1)             # dirt rising from the floor
    band = np.exp(-((up - sk_h - 0.5) / 0.3) ** 2) * np.clip(ndimage.gaussian_filter1d(n3, 0.4 * p, axis=1) * 3 + 0.2, 0, 1); g *= 1 - s * 0.06 * band   # scuffs where things rub
    cor = np.exp(-np.minimum(xx, f["w"] - xx) / 0.18) * (0.5 + 0.5 * np.clip(up / f["h"], 0, 1) * 0 + 0.5 * n2); g *= 1 - s * 0.07 * np.clip(cor, 0, 1)       # corners gather dirt
    warm = np.zeros((h, w), np.float32)
    if style.get("time") == "night" or any(is_fire(t) for t in f["things"]):         # candle and fire smoke gathers under the ceiling
        top = np.exp(-yy / np.maximum(0.35 + 0.15 * n1, 0.08)) * (0.6 + 0.4 * n2); g *= 1 - s * 0.14 * top; warm += 0.5 * s * top
    for t in f["things"]:
        if not is_fire(t): continue
        x0, y0, x1, y1 = t["box"]; cx = (x0 + x1) / 2; hw = (x1 - x0) / 2; rise = np.clip((y0 + 0.25 * (y1 - y0) - yy), 0, None)            # soot: from the fire's mouth upward, widening and thinning
        spread = hw * 0.55 + 0.22 * rise; plume = np.exp(-((xx - cx + 0.25 * n1 * rise) / spread) ** 2) * np.exp(-rise / np.maximum(1.1 + 0.4 * n2, 0.3)) * (rise > 0) * (0.7 + 0.5 * n2 + 0.3 * n3)
        plume = np.clip(plume, 0, 1); g *= 1 - np.clip(s * 0.6, 0.25, 0.75) * plume; warm += plume
    sol = f["solid"]; hole = np.asarray(Image.fromarray((~sol * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), np.float32) / 255
    if hole.max() > 0.5:                                                               # runs under sills: a streaked stain below each opening that does not reach the floor
        below = np.zeros((h, w), np.float32); acc = np.zeros(w, np.float32)
        for y in range(h):
            acc = np.where(hole[y] > 0.5, 1.0, acc * (1 - 1 / (0.55 * p))); below[y] = acc * (hole[y] < 0.5)
        streak = np.clip(ndimage.gaussian_filter1d(C.noise((h, w), 0.05 * p, seed + 5, 1), 0.5 * p, axis=0) * 4 + 0.35, 0, 1); floor_open = hole[-2] > 0.5
        below[:, floor_open] = 0; g *= 1 - s * 0.13 * below * streak
    return np.clip(g, 0.3, 1.25), np.clip(warm, 0, 1)
def floor_wear(f, style, F, seed):
    s = WEAR.get(style.get("wear"), 0.55); p = WEAR_PPM; h, w = int(round(f["h"] * p)), int(round(f["w"] * p)); yy, xx = np.mgrid[:h, :w].astype(np.float32) / p
    n1 = C.noise((h, w), 2.0 * p, seed); n2 = C.noise((h, w), 0.4 * p, seed + 1); n3 = C.noise((h, w), 0.06 * p, seed + 2, 2); g = 1 + s * (0.06 * n1 + 0.04 * n2); warm = np.zeros((h, w), np.float32)
    edge = np.minimum(np.minimum(xx, f["w"] - xx), np.minimum(yy, f["h"] - yy)); g *= 1 - s * 0.13 * np.exp(-edge / np.maximum(0.16 + 0.08 * n2, 0.04)) * (0.6 + 0.4 * n1)                    # dirt along the walls
    marks = " ".join(style.get("floor_marks") or []).lower(); doors = floor_doors(f, F); big = [t for t in f["things"] if (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1]) > 2.0 and t["height"] > 0.5]
    cx, cy = f["w"] / 2, f["h"] / 2
    def seg_d(ax, ay, bx, by, bend=0.0):
        t = np.clip(((xx - ax) * (bx - ax) + (yy - ay) * (by - ay)) / max((bx - ax) ** 2 + (by - ay) ** 2, 1e-6), 0, 1); px, py = ax + t * (bx - ax), ay + t * (by - ay)
        nx_, ny_ = -(by - ay), (bx - ax); L = math.hypot(nx_, ny_) + 1e-6; px += bend * np.sin(t * math.pi) * nx_ / L; py += bend * np.sin(t * math.pi) * ny_ / L; return np.hypot(xx - px, yy - py), t
    for (dx, dy, dw) in doors:                                                         # a path worn from each door towards the middle: a little paler and duller
        d, t = seg_d(dx, dy, cx, cy); g *= 1 + s * 0.07 * np.exp(-(d / np.maximum(0.45 + 0.2 * n1, 0.15)) ** 2) * (1 - 0.6 * t) * (0.6 + 0.4 * n2)
    if any(k in marks for k in ("tyre", "tire", "skid", "wheel")) and doors:
        dx, dy, dw = max(doors, key=lambda q: q[2]); rng = np.random.default_rng(seed)
        for t_ in (big or [{"box": [cx - 1, cy - 1, cx + 1, cy + 1]}]):
            bx, by = (t_["box"][0] + t_["box"][2]) / 2, (t_["box"][1] + t_["box"][3]) / 2; bend = rng.uniform(-1.2, 1.2)
            for side in (-0.72, 0.72):                                                 # two tracks a car's width apart, 20 cm wide, broken and fading
                ox, oy = (by - dy), -(bx - dx); L = math.hypot(ox, oy) + 1e-6; ox, oy = ox / L * side, oy / L * side
                d, t = seg_d(dx + ox, dy + oy, bx + ox, by + oy, bend); tr = np.exp(-(d / 0.085) ** 4) * np.clip(0.55 + 0.9 * n2 + 0.5 * n3, 0, 1) * np.clip(1.2 - t, 0.2, 1) * np.clip(0.5 + 2.0 * np.sin(t * 9 + side) ** 2, 0, 1)
                g *= 1 - np.clip(0.2 + 0.25 * s, 0, 0.5) * tr
    if any(k in marks for k in ("oil", "stain", "spill", "drip", "wax", "grease")):
        rng = np.random.default_rng(seed + 9); spots = [( (t_["box"][0] + t_["box"][2]) / 2 + rng.uniform(-0.6, 0.6), (t_["box"][1] + t_["box"][3]) / 2 + rng.uniform(-0.8, 0.8), rng.uniform(0.05, 0.14)) for t_ in (big or f["things"])[:6]]
        for sx, sy, r in spots:
            d = np.hypot(xx - sx, yy - sy) + 0.12 * n2 * r * 4; blot = np.clip(1.4 - d / r, 0, 1) ** 1.5; g *= 1 - np.clip(0.1 + 0.12 * s, 0, 0.3) * blot; warm += 0.3 * blot
    for t in f["things"]:
        if not is_fire(t): continue
        x0, y0, x1, y1 = t["box"]; d = np.hypot(np.clip(np.maximum(x0 - xx, xx - x1), 0, None), np.clip(np.maximum(y0 - yy, yy - y1), 0, None)); ash = np.exp(-d / np.maximum(0.3 + 0.1 * n2, 0.08)) * (0.6 + 0.5 * n2 + 0.3 * n3)
        g *= 1 - np.clip(0.2 * s + 0.05, 0, 0.35) * np.clip(ash, 0, 1); warm += 0.4 * np.clip(ash, 0, 1)
    return np.clip(g, 0.35, 1.25), np.clip(warm, 0, 1)
def floor_doors(f, F):
    """Where the walls' openings reach the floor: [(x, y on the floor texture, width m)]."""
    out = []
    for n in WALLS:
        if n not in F: continue
        wf = F[n]; s = wf["solid"]; row = ~s[-2]; lab, k = ndimage.label(row)
        for i in range(1, k + 1):
            xs = np.flatnonzero(lab == i); u0, u1 = xs.min() / 32, (xs.max() + 1) / 32
            if u1 - u0 < 0.5: continue
            Ainv = np.linalg.pinv(np.c_[wf["A"][:, :3]]); uv = np.array([(u0 + u1) / 2 + wf["u0"], wf["v0"]]); room_pt = wall_point(wf, (u0 + u1) / 2, F)
            xy = to_tex(f, room_pt[None])[0]; out.append((float(xy[0]), float(xy[1]), float(u1 - u0)))
    return out
def wall_point(wf, x_tex, F):
    """A wall's texture x (metres) at the floor -> room point."""
    A = wf["A"]; u = x_tex + wf["u0"]; M = np.array([A[0, [0, 2]], [1.0, 0.0] if wf["name"][0] == "x" else [0.0, 1.0]]); n = wf["name"]
    fl = F.get("floor"); W = fl["w"] if fl else 0; D = fl["h"] if fl else 0; fixed = {"x0": 0.0, "x1": W, "z0": 0.0, "z1": D}[n]
    sol = np.linalg.solve(M, np.array([u - A[0, 3], fixed])); return np.array([sol[0], 0.0, sol[1]])
def apply_wear(tex, g, warm, ppm, soot=(0.42, 0.36, 0.3)):
    """Wear multiplied into a texture, in bands (the texture may be very large)."""
    h, w = tex.shape[:2]; G = Image.fromarray(g); Wm = Image.fromarray(warm); step = 1024
    for y in range(0, h, step):
        y1 = min(y + step, h); box = (0, y * g.shape[0] / h, g.shape[1], y1 * g.shape[0] / h)
        gg = np.asarray(G.resize((w, y1 - y), Image.BICUBIC, box=box), np.float32); ww = np.asarray(Wm.resize((w, y1 - y), Image.BICUBIC, box=box), np.float32)[..., None]
        t = tex[y:y1] * gg[..., None]; lum = (t @ C.GREY)[..., None]; tex[y:y1] = t * (1 - 0.45 * ww) + 0.45 * ww * lum * np.float32(soot) * 2.2
    return tex

# ---------------------------------------------------------------- a face laid
def zones_of(f, style):
    """A wall's zones from the bottom: [(y0, y1 in texture metres (down), zone dict, key)], and the split's cover: True if a kit rail hides the line."""
    w = style["walls"]; sp = float(w.get("split_m") or 0)
    if sp <= 0 or sp >= f["h"] - 0.3: return [(0.0, f["h"], w["upper"], "wall_upper")], True, 0.0
    ysp = f["h"] - sp; covered = False
    for y0, y1, piece in rails(f):
        if y1 > f["h"] - 0.4 or y0 < 0.4: continue
        if abs((y0 + y1) / 2 - ysp) < 0.8: ysp = (y0 + y1) / 2; covered = True; break
    return [(ysp, f["h"], w["lower"], "wall_lower"), (0.0, ysp, w["upper"], "wall_upper")], covered, f["h"] - ysp
def lay_zone(lib, z, w, h, kind, ppm, seed, segs=None, y_top=0.0, motif=None, wear=1.0, k0=0):
    H, W = int(round(h * ppm)), int(round(w * ppm)); u, cells, jw = plan_cells(z, w, h, kind, seed, segs, k0)
    if cells and (lib.units or lib.loose): tex = lib.lay(cells, (H, W), ppm, seed + 3)
    else:
        tex = lib.spread((H, W), ppm, seed + 3)
        if u == "slab":                                                               # cast in large pieces: fine saw-cut joints, a whole number across the face
            _, a, b = unit_of(z); nx, ny = max(1, int(round(w / a))), max(1, int(round(h / b))); rng = np.random.default_rng(seed); lw = max(1, int(round(0.005 * ppm)))
            xs = np.round(np.linspace(0, W, nx + 1)).astype(int); ys = np.round(np.linspace(0, H, ny + 1)).astype(int)
            for i in range(nx):
                for j in range(ny): tex[ys[j]:ys[j + 1], xs[i]:xs[i + 1]] *= rng.uniform(0.965, 1.035)
            for x in xs[1:-1]: tex[:, x - lw:x + lw] *= 0.62
            for y in ys[1:-1]: tex[y - lw:y + lw] *= 0.62
    if z.get("pattern") and motif is not None: tex = stamp(tex, z, motif, ppm, y_top, seed + 7, wear)
    return tex
def lay_face(f, style, libs, motifs, F, seed=0):
    """-> (texture float (h, w, 3), record). libs: {key: Library}, motifs: {key: (rgb, alpha)}."""
    kind = f["kind"]; ppm = float(min(PPM[kind], (MAX_PX / (f["w"] * f["h"])) ** 0.5)); H, W = int(round(f["h"] * ppm)), int(round(f["w"] * ppm)); rec = {"px_per_m": ppm, "zones": []}
    wl = WEAR.get(style.get("wear"), 0.55)
    if kind == "wall":
        zs, covered, split = zones_of(f, style); tex = np.zeros((H, W, 3), np.float32); segs = breaks(f); k0 = WALLS.index(f["name"]) if f["name"] in WALLS else 0
        for y0, y1, z, key in zs:
            a, b = int(round(y0 * ppm)), int(round(y1 * ppm)); t = lay_zone(libs[key], z, f["w"], y1 - y0, "wall", ppm, sd(key, 1), segs, y_top=y0, motif=motifs.get(key), wear=wl, k0=k0)       # the same seed on every wall: courses meet at the corners
            if unit_of(z)[0] in ("none", "slab", "plank"): t = lay_zone(libs[key], z, f["w"], y1 - y0, "wall", ppm, sd(key, f["name"]), segs, y_top=y0, motif=motifs.get(key), wear=wl)
            tex[a:b] = t[:b - a]; rec["zones"].append({"from_floor_m": [round(f["h"] - y1, 3), round(f["h"] - y0, 3)], "key": key, "material": z["material"], "finish": z.get("finish")})
        if len(zs) > 1 and not covered:                                                # the meeting line is a painted band where no rail hides it
            bw = 0.07; c = np.float32(band_colour(style)); a, b = int(round((zs[0][0] - bw / 2) * ppm)), int(round((zs[0][0] + bw / 2) * ppm))
            tex[a:b] = libs["wall_upper"].spread((b - a, W), ppm, 5) * np.clip((c + 4) / (libs["wall_upper"].tone + 4), 0.3, 3.0)
        g, warm = wall_wear(f, style, sd("wear", f["name"])); rec["split_m"] = round(split, 3); rec["split_hidden_by_rail"] = bool(covered)
    else:
        key = "floor" if kind == "floor" else "ceiling"; z = style[key]; tex = lay_zone(libs[key], z, f["w"], f["h"], kind, ppm, sd(key, 2)); rec["zones"].append({"key": key, "material": z["material"], "finish": z.get("finish")})
        if kind == "floor": g, warm = floor_wear(f, style, F, sd("wear", "floor"))
        else:
            p = WEAR_PPM; hh, ww = int(round(f["h"] * p)), int(round(f["w"] * p)); g = 1 + wl * (0.06 * C.noise((hh, ww), 1.8 * p, 4) + 0.03 * C.noise((hh, ww), 0.3 * p, 5)); warm = np.zeros((hh, ww), np.float32)
            if style.get("time") == "night": warm += 0.25 * wl * (0.5 + 0.5 * C.noise((hh, ww), 1.2 * p, 6)); g *= 1 - 0.1 * wl
            g = g.astype(np.float32)
    tex = apply_wear(tex, g.astype(np.float32), warm.astype(np.float32), ppm); return np.clip(tex, 0, 255), rec
def band_colour(style):
    """The accent: a pattern's own colour, else the palette's most colourful entry that is neither wall colour."""
    import colorsys
    for z in (style["walls"]["upper"], style["walls"]["lower"]):
        if z.get("pattern"): return z["pattern"]["rgb"]
    best, bs = style["materials"]["trim"]["rgb"], 0.0; wc = [np.float32(style["walls"][k]["rgb"]) for k in ("upper", "lower")]
    for p in style.get("palette") or []:
        c = np.float32(p["rgb"]); h, sat, v = colorsys.rgb_to_hsv(*(c / 255)); sc = sat * (0.4 + v)
        if min(float(np.abs(c - w).sum()) for w in wc) > 90 and sc > bs: best, bs = [float(x) for x in c], sc
    return best
