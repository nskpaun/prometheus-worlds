"""Object paint: every sculpted kind of a world painted on its OWN mesh from ONE picture of the picture model: a sheet of
several straight-on views of the mesh at one scale, so all sides are painted together and agree. The model STARTS from the
sheet under noise (so each view keeps its place, camera and outline) and is given the kind's own picture as what the thing
looks like. A thing longer than 2.5 m has its views on three sheets.
  paint.py prep    samples at the paint's own density, the kind's own picture fitted and projected (first coat)   (CPU, 4 workers)
  paint.py ref     views chosen by cover, the start sheets, the job list paint/jobs.json                           (CPU, 4 workers)
  (klein_paint.py paint/jobs.json     ../../spike/.venv: the picture model)
  paint.py maps    full quality only: the job list paint/mapjobs.json that turns each painted sheet into a material map (then klein.py on it)
  paint.py bake    answers registered, each view accepted softly, laid on the samples, unseen sides filled, material classes  (CPU, 4 workers)
Environment: WORLD_DIR, WORLD_QUALITY (where.quality). Out per kind, WORLD_DIR/paint/KIND/: samples.npz (P, face, bary, N on
geom.load's mesh), tex_final.npz (rgb, how, mat), meta.json, sheet.json, bake.json.
  how: 0 the painted view anchored by the kind's own picture, 3 another painted view, 1 mirrored from the other side, 5 filled in the mesh's own space.
  mat: per point material class, see MATS.
`ref` clears a kind's earlier answers: after the picture model has run, go on from `bake`, not from `ref`.
thread/.venv; nothing here knows a world or a kind."""
import sys, os, json, time, math, zlib
import numpy as np
import where
import geom, align, sheet as SH, paint_class
from geom import cKDTree
from scipy import ndimage
from PIL import Image
LUM = np.float32([0.299, 0.587, 0.114])
MATS = ["plain", "glass", "metal", "glow", "gloss"]          # mat codes 0..4: matt stuff; clear or dark glass; bright bare metal (chrome, brass, steel); a thing that gives light (flame, bulb, lamp glass); shiny paint, polished wood, leather
TEXEL, TEXEL_SAID = 0.0030, 0.0022          # the paint's pixel on the object, metres: no finer than the level lays discs
MP_MIN, MP_MAX, MP_BIG = where.quality()["sheet_mp"]  # sheet megapixels: every kind; a kind as large as furniture; a thing over 2.5 m or one the sentence names that is over 1 m
PROMPT = ("The same {name} as in the reference picture ({look}), photographed {n} times on a plain {bw} background, each time from a different side. "
          "Every one is a sharp, clean, realistic studio photograph of that same {name}: the same colours, materials and details as in the reference picture, on every side. "
          "Soft even light from all sides. Plain flat {bw} background. No shadows on the ground, no floor, no text. Photorealistic, sharp focus, fine real material detail.")
BIG_M, N_BIG = 2.5, 3                        # a thing longer than BIG_M metres has its views on N_BIG sheets
SIGMA0 = 0.96                               # the noise the start sheet is put under. Measured on the car: 0.93 freezes the guessed colours (red wheels, red windows), 0.975 lets Klein move the cameras
STEPS, STRENGTH = 4, 0.25                   # Klein starts from the sheet itself under noise and skips the first step: where each view is, its camera and its outline are given

def kinds(world):
    """Every sculpted kind of the world's layout -> {kind: dict(shapes, look, name, objects, said, attach, size)}"""
    sh = f"{world}/shapes"; plan = json.load(open(f"{sh}/plan.json")); b = json.load(open(f"{world}/boxes.json")); out = {}
    for k, v in plan["kinds"].items():
        os_ = [b["objects"][i] for i in v["objects"]]; o = os_[0]
        out[k] = dict(shapes=sh, look=v["look"], name=k.replace("_", " "), objects=list(v["objects"]), said=any(x.get("said") for x in os_), attach=o.get("attach") or "floor", size=o.get("size") or [1, 1, 1], request=b.get("request", ""))
    return out
def wd(world, kind): return where.paint_dir(world, kind)

def budget(Kd, metres):
    """Sheet megapixels and view count from how big the thing is and whether the sentence names it."""
    big = max(metres); said = Kd["said"]
    mp = MP_BIG if (big > 2.5 or (said and big > 1.0)) else MP_MAX if (big > 0.7 or said) else MP_MIN
    return mp, (1 if Kd["attach"] == "wall" and min(metres) < 0.12 * big else 4 if mp < MP_BIG else 6)

# ---------------------------------------------------------------- prep
def prep_one(a):
    world, kind = a; t0 = time.time(); Kd = kinds(world)[kind]; d = wd(world, kind)
    V, F = geom.load(Kd["shapes"], kind); sc = geom.scene_of(V, F); diag = float(np.linalg.norm(V.max(0) - V.min(0))); half = geom.FILL * diag
    Vp = np.load(f"{Kd['shapes']}/placed/{Kd['objects'][0]:03d}.npz")["V"].astype(float); M = np.linalg.lstsq(np.c_[V, np.ones(len(V))], Vp, rcond=None)[0]; A = M[:3].T; scale = float(abs(np.linalg.det(A)) ** (1 / 3))
    metres = (np.ptp(V, 0) * scale).tolist(); mp, nv = budget(Kd, metres); texel = TEXEL_SAID if Kd["said"] else TEXEL
    import trimesh
    area = float(trimesh.Trimesh(V, F, process=False).area) * scale ** 2; n = int(np.clip(area / texel ** 2, 250_000, 2_500_000)); P, fi, bary, N = geom.sample(V, F, n)
    rgba = np.asarray(Image.open(f"{Kd['shapes']}/{kind}.rgba.png"), np.float32) / 255; hpx = rgba.shape[0]; hero = geom.from_img(f"{Kd['shapes']}/{kind}.png")
    if hero.shape[0] != hpx: hero = np.asarray(Image.fromarray((hero * 255).astype(np.uint8)).resize((hpx, hpx), Image.LANCZOS), np.float32) / 255
    alpha = rgba[..., 3] > 0.5; back = geom.backdrop(hero)[2]; fit = geom.fit_picture(sc, half, alpha); t_fit = time.time() - t0
    v0 = geom.View(half, fit["az"], fit["el"], hpx, fit["k"], fit["flip"]); R0 = v0.render(sc); aff = geom.box_affine(alpha, R0["hit"])
    hero_w = geom.despill(geom.warp_to(hero, aff, (hpx, hpx)), back); alpha_w = geom.warp_to(alpha.astype(np.float32), aff, (hpx, hpx)) > 0.5
    ok = ndimage.binary_erosion(alpha_w & R0["hit"], iterations=max(2, hpx // 200)) & ~geom.depth_edges(R0, v0, max(3, hpx // 85))
    c0, w0 = geom.project_paint(v0, sc, P, N, hero_w, ok)
    np.savez_compressed(f"{d}/samples.npz", P=P.astype(np.float32), face=fi.astype(np.int32), bary=bary.astype(np.float32), N=N.astype(np.float32))
    np.savez_compressed(f"{d}/hero.npz", rgb=c0.astype(np.float16), w=w0.astype(np.float16)); geom.to_img(hero_w).save(f"{d}/hero_view.png"); Image.fromarray(ok).save(f"{d}/hero_ok.png")
    meta = dict(kind=kind, look=Kd["look"], name=Kd["name"], said=Kd["said"], attach=Kd["attach"], half=half, diag=diag, scale=scale, metres=metres, area_m2=area, samples=n, mp=mp, n_views=nv, fit=fit, hero_px=hpx,
                hero_seen=round(float((w0 > 0.05).mean()), 3), hero_stray=geom.stray(alpha_w, R0["hit"]), A=A.tolist(), t=M[3].tolist(), s_prep=round(time.time() - t0, 1))
    json.dump(meta, open(f"{d}/meta.json", "w"), indent=1); return kind, "prep", meta["s_prep"], "samples", n, "own picture seen", meta["hero_seen"], "fit", fit["box_iou"]

def ctx(world, kind):
    Kd = kinds(world)[kind]; d = wd(world, kind); meta = json.load(open(f"{d}/meta.json")); V, F = geom.load(Kd["shapes"], kind); S = np.load(f"{d}/samples.npz")
    return Kd, d, meta, V, F, geom.scene_of(V, F), S["P"].astype(np.float64), S["N"].astype(np.float64)

# ---------------------------------------------------------------- the reference sheet
def tile_views(S):
    return [dict(t, view=geom.View(**t["spec"])) for t in S["tiles"]]

def main_colour(px):
    """The colour most of a thing has, as it looks lit (px: (n, 3) pixels of its own picture). A coloured thing: the commonest hue's lit pixels (a red car
    is red, whatever its glass, chrome and tyres add up to). A thing without colour: its middle greys."""
    x = px[np.random.default_rng(0).choice(len(px), min(len(px), 40000), replace=False)].astype(np.float32); h, s_, v = paint_class.hsv(x); lum = x @ np.float32([0.299, 0.587, 0.114]); col = (s_ > 0.3) & (v > 0.15)
    if col.mean() > 0.25:
        hist = np.bincount((h[col] / 10).astype(int) % 36, weights=(s_ * v)[col], minlength=36); pk = (np.argmax(np.convolve(np.r_[hist[-1:], hist, hist[:1]], [1, 2, 1], "valid")) + 0.5) * 10
        m = col & (np.abs((h - pk + 180) % 360 - 180) < 20); lo, hi = np.percentile(lum[m], [25, 65]); return x[m & (lum >= lo) & (lum <= hi)].mean(0)
    lo, hi = np.percentile(lum, [40, 80]); return x[(lum >= lo) & (lum <= hi)].mean(0)

def ref_one(a):
    world, kind = a; t0 = time.time(); Kd, d, meta, V, F, sc, P, N = ctx(world, kind); half, fit = meta["half"], meta["fit"]; H = np.load(f"{d}/hero.npz"); hw = H["w"].astype(np.float32); hrgb = H["rgb"].astype(np.float32)
    sub = np.random.default_rng(0).choice(len(P), min(len(P), 120_000), replace=False); hv = geom.View(half, fit["az"], fit["el"], 64, fit["k"], fit["flip"])
    trust = meta["fit"]["box_iou"] > 0.75 and meta["hero_stray"]["iou"] > 0.75          # the kind's own picture really is a picture of this mesh
    first = [("own", hv, "")] if trust else []
    meta["mp"], meta["n_views"] = budget(Kd, meta["metres"]); ns = N_BIG if max(meta["metres"]) > BIG_M else 1; meta["n_sheets"] = ns; json.dump(meta, open(f"{d}/meta.json", "w"), indent=1)
    views, cov = SH.choose_views(V, sc, P[sub], N[sub], SH.cand_views(Kd["attach"], max(meta["metres"]) < 0.5), first, meta["n_views"], min_gain=0.004)
    good = hw > 0.25
    hero = geom.from_img(f"{Kd['shapes']}/{kind}.png"); al = np.asarray(Image.open(f"{Kd['shapes']}/{kind}.rgba.png").resize((hero.shape[1], hero.shape[0])), np.float32)[..., 3] > 127
    hfg = geom.despill(hero, geom.backdrop(hero)[2])[ndimage.binary_erosion(al, iterations=4)]; dom = main_colour(hfg) if len(hfg) > 500 else np.float32([0.5, 0.5, 0.5]); lum = float(np.median(hfg @ [0.299, 0.587, 0.114])) if len(hfg) else 0.5
    back, bw = (([0.74, 0.74, 0.74], "light grey") if lum < 0.45 else ([0.24, 0.24, 0.24], "dark grey"))
    # the first coat on every point: the kind's own picture where it saw the surface, and its mirror image. Everything else is a GUESS Klein will keep the
    # broad colour of, so the guess is the plainest true thing: the thing's own main colour, softly lit so its form reads
    # (a filled-in guess of nearest colours puts window, tyre and soot colours where they do not belong, and Klein paints them as damage).
    cs = np.random.default_rng(1).choice(len(P), min(len(P), 250_000), replace=False); Pc, Nc = P[cs], N[cs]; coat = np.empty((len(cs), 3), np.float32); coat[:] = np.maximum(dom, 0.13)
    if not (trust and good.sum() > 500): trust = False; good = np.zeros(len(P), bool)
    if trust:          # a side's guess: the main colour of what the own picture saw FACING THAT WAY (a mantel's top is wood though the fireplace is black), if it saw enough of it
        seen = hw > 0.04; ax = np.abs(N).argmax(1); face = ax * 2 + (np.take_along_axis(N, ax[:, None], 1)[:, 0] > 0); fc = face[cs]
        for b_ in range(6):
            m = seen & (face == b_)
            if m.sum() > max(800, 0.03 * seen.sum()):          # only if it is another stuff altogether (another hue, or colour against no colour): the same paint in more light is not
                c_ = main_colour(hrgb[m]); (h1, s1, v1), (h2, s2, v2) = [[float(z[0]) for z in paint_class.hsv(np.float32(x)[None])] for x in (dom, c_)]
                if (s1 > 0.25) != (s2 > 0.25) or (s1 > 0.25 and abs((h1 - h2 + 180) % 360 - 180) > 25) or (s1 <= 0.25 and abs(v1 - v2) > 0.3): coat[fc == b_] = np.maximum(c_, 0.13)
        nbc = cKDTree(Pc).query(Pc, k=16)[1]; coat = coat[nbc].mean(1)          # no hard line where the facing changes
    np.savez_compressed(f"{d}/coat.npz", rgb=coat.astype(np.float16), idx=cs); tree = cKDTree(P); tc = cKDTree(Pc)
    hs = 512; Image.fromarray((hero * 255).astype(np.uint8)).resize((hs, hs), Image.LANCZOS).save(f"{d}/ref.png")          # what the thing looks like: its own picture, small (time is tokens)
    for f in os.listdir(d):
        if f.startswith(("out_", "mat_", "ref_full_")): os.remove(f"{d}/{f}")
    J = dict(back=back, bw=bw, cov=[round(c, 3) for c in cov], trust=bool(trust), dom=[float(x) for x in dom], sheets=[]); jobs = []; texel = TEXEL_SAID if Kd["said"] else TEXEL
    for si in range(ns):          # a very large thing gets its views on several sheets, or its paint's pixel would be a centimetre
        vs = views[si::ns]
        if not vs: continue
        ppm = min(SH.fit_scale(sc, vs, half, meta["mp"] * 1e6), meta["scale"] / texel)        # no finer than the level can show
        ppm = max(ppm, SH.fit_scale(sc, vs, half, MP_MIN * 1e6))                              # a small thing still gets a picture large enough for Klein to draw in
        tiles = SH.make_tiles(sc, vs, half, ppm, pad=max(6, int(0.025 * 2 * half * ppm))); W, Hh = SH.layout(tiles, gap=max(10, int(0.035 * 2 * half * ppm))); imgs = []
        for t in tiles:
            R, vw = t["R"], t["view"]; hit = R["hit"]; Q = R["Q"][hit]; j = tree.query(Q)[1]; alb = np.zeros(hit.shape + (3,), np.float32); g = np.zeros(hit.shape, np.float32); g[hit] = good[j] * (t["name"] == "own")          # the own picture's paint only in its own view: seen from elsewhere it is a slanted, broken patch that Klein keeps as damage
            g = ndimage.gaussian_filter(ndimage.binary_opening(g > 0.5, iterations=2).astype(np.float32), 2.0)[..., None]; alb[hit] = hrgb[j]; img = geom.shade(R, vw, "flat", back, alb); alb[hit] = coat[tc.query(Q)[1]]
            cl = geom.shade(R, vw, "clay", back); sh = np.where(hit[..., None], np.clip(alb * (0.55 + 1.1 * cl), 0, 1), cl)          # everything not seen by the own picture: the form lit plainly in the thing's own main colour
            imgs.append(np.where(hit[..., None], img * g + sh * (1 - g), img))
        geom.to_img(SH.compose(tiles, W, Hh, back, imgs)).save(f"{d}/ref_full_{si}.png")
        J["sheets"].append(dict(W=W, H=Hh, ppm=ppm, mm_px=round(1000 * meta["scale"] / ppm, 2), tiles=[dict(name=t["name"], words=t["words"], box=t["box"], at=t["at"], spec=t["view"].spec()) for t in tiles]))
        jobs.append(dict(name=f"{kind}#{si}", kind=kind, out=f"{d}/out_{si}.png", ref=f"{d}/ref.png", init=f"{d}/ref_full_{si}.png", steps=STEPS, strength=STRENGTH, sigma0=SIGMA0, w=W, h=Hh, seed=7,
                         prompt=PROMPT.format(name=Kd["name"], look=Kd["look"], n=len(tiles), bw=bw)))
    J["mm_px"] = max(s_["mm_px"] for s_ in J["sheets"]); json.dump(J, open(f"{d}/sheet.json", "w"), indent=1)
    return kind, {"jobs": jobs}, f"{len(jobs)} sheet(s) " + " ".join(f"{s_['W']}x{s_['H']}" for s_ in J["sheets"]) + f" {J['mm_px']} mm/px views {[t['name'] for s_ in J['sheets'] for t in s_['tiles']]} cover {J['cov'][-1]} {round(time.time() - t0, 1)} s"

MAT_PROMPT = ("Turn this picture into a flat material map. Keep every object exactly where it is, with exactly its shape and outline. Paint every piece of glass pure blue. "
              "Paint every piece of bare shiny metal (chrome, brass, gold, polished steel) pure yellow. Paint every flame and every glowing lamp bulb pure red. Paint everything else of the objects pure black. "
              "Make the background pure white. Flat colours only: no shading, no highlights, no reflections, no outlines, no text.")
MAT_MP = 0.16
def maps(world):
    """A second, small Klein job for the kinds whose words say they have glass or a light: each painted sheet turned into a material map."""
    jobs = []
    for kind, Kd in kinds(world).items():
        d = wd(world, kind); A = paint_class.allowed(Kd)
        if not os.path.exists(f"{d}/sheet.json") or not (A["glass"] or A["glow"]): continue
        for si in range(len(json.load(open(f"{d}/sheet.json"))["sheets"])):
            if not os.path.exists(f"{d}/out_{si}.png"): continue
            im = Image.open(f"{d}/out_{si}.png").convert("RGB"); k = min(1.0, (MAT_MP * 1e6 / (im.width * im.height)) ** 0.5); w, h = int(round(im.width * k / 16) * 16), int(round(im.height * k / 16) * 16)
            im.resize((w, h), Image.LANCZOS).save(f"{d}/mat_ref_{si}.png")
            if os.path.exists(f"{d}/mat_{si}.png"): os.remove(f"{d}/mat_{si}.png")
            jobs.append(dict(name=f"{kind}#{si}/mat", out=f"{d}/mat_{si}.png", ref=f"{d}/mat_ref_{si}.png", w=w, h=h, seed=7, prompt=MAT_PROMPT))
    json.dump(jobs, open(f"{where.paint_dir(world)}/mapjobs.json", "w"), indent=1); print("material maps to make:", len(jobs), flush=True)

# ---------------------------------------------------------------- bake
def symmetric(P, sub, diag):
    """Is the mesh its own mirror image left to right (x -> -x)? -> share of points whose mirror image lies on the surface"""
    q = P[sub[:20000]] * [-1, 1, 1]; return float((cKDTree(P[sub]).query(q)[0] < 0.012 * diag).mean())

def fill_space(P, N, rgb, got, diag, seed=0, base=None, grey_grain=False, clip=3.0):
    """Paint for points no view saw, made in the mesh's own space. Smooth part: the colour of the nearest painted points that face the same way.
    Fine part: real texture carried over in patches: every 5%-of-the-thing cell of unseen surface takes the fine detail (colour minus its smooth part) of
    one painted place that faces the same way, point for point at the same offsets, so the grain is the material's own and hangs together (no streaks,
    no salt and pepper)."""
    out = rgb.astype(np.float32).copy(); q = np.nonzero(~got)[0]; sa = np.nonzero(got)[0]
    if not len(q) or len(sa) < 50: return out
    X = np.concatenate([P, N * 0.25 * diag], 1); rng = np.random.default_rng(seed); st = sa if len(sa) <= 60_000 else rng.choice(sa, 60_000, replace=False)
    t6 = cKDTree(X[st]); k = min(12, len(st)); cs = out[st]
    def smooth(idx):
        res = np.empty((len(idx), 3), np.float32)
        for a in range(0, len(idx), 300_000):
            dd, jj = t6.query(X[idx[a:a + 300_000]], k=k); w = 1 / (dd + 0.01 * diag) ** 2; res[a:a + 300_000] = (cs[jj] * w[..., None]).sum(1) / w.sum(1)[:, None]
        return res
    grain = out[sa] - smooth(sa); base = smooth(q) if base is None else np.broadcast_to(np.asarray(base, np.float32), (len(q), 3)); t3 = cKDTree(P[sa]); L = 0.05 * diag
    if grey_grain: grain = np.repeat(grain.mean(1, keepdims=True), 3, 1)          # light and dark only: the grain never brings another part's colour
    lim = clip * float(np.median(np.abs(grain))) + 1e-4; grain = np.clip(grain, -lim, lim)          # a smooth thing stays smooth: its few busy places (a grille, a rim) do not become grain
    cell = np.floor(P[q] / L).astype(np.int64); cell -= cell.min(0); key = (cell[:, 0] * (cell[:, 1].max() + 1) + cell[:, 1]) * (cell[:, 2].max() + 1) + cell[:, 2]
    side = np.abs(N[q]).argmax(1) * 2 + (np.take_along_axis(N[q], np.abs(N[q]).argmax(1)[:, None], 1)[:, 0] > 0); key = key * 6 + side          # one patch a cell and facing
    u, inv = np.unique(key, return_inverse=True); cnt = np.bincount(inv).astype(np.float64); cen = np.stack([np.bincount(inv, weights=X[q][:, c]) / cnt for c in range(6)], 1)
    _, js = t6.query(cen, k=min(3, len(st))); js = js.reshape(len(u), -1); src = P[st][js[np.arange(len(u)), rng.integers(0, js.shape[1], len(u))]]
    d3, j3 = t3.query(src[inv] + (P[q] - cen[inv, :3])); w = np.clip(1 - d3 / (0.012 * diag), 0, 1)[:, None]
    out[q] = np.clip(base + grain[j3] * w, 0, 1); return out

def bake_one(a):
    world, kind = a; t0 = time.time(); Kd, d, meta, V, F, sc, P, N = ctx(world, kind)
    J = json.load(open(f"{d}/sheet.json")); back = np.asarray(J["back"], np.float32); st = {"slid_px": [], "tiles": {}}
    H = np.load(f"{d}/hero.npz"); cols, wts, tags, mcols, gains = [], [], [], [], []; S = len(P); any_mat = False
    for si, SJ in enumerate(J["sheets"]):
        if not os.path.exists(f"{d}/out_{si}.png"): continue
        W, Hh = SJ["W"], SJ["H"]; out = geom.from_img(f"{d}/out_{si}.png")
        if out.shape[:2] != (Hh, W): out = np.asarray(geom.to_img(out).resize((W, Hh), Image.LANCZOS), np.float32) / 255
        ref = geom.from_img(f"{d}/ref_full_{si}.png"); reg, sm, s95, flow = align.register(ref, out, want_flow=True); st["slid_px"].append([round(sm, 1), round(s95, 1)]); mk = None
        if os.path.exists(f"{d}/mat_{si}.png"): mk = align.warp(np.asarray(Image.open(f"{d}/mat_{si}.png").convert("RGB").resize((W, Hh), Image.BILINEAR), np.float32) / 255, flow); any_mat = True
        for tj in tile_views(SJ):
            vw = tj["view"]; R = vw.render(sc); mesh = R["hit"]; img, valid = SH.cut(tj, reg, fill=float(back[0])); raw, _ = SH.cut(tj, out, fill=float(back[0])); x0, y0, x1, y1 = tj["box"]; px = vw.px
            # what Klein drew as the object: everything that is not its backdrop (the backdrop's colour taken from the tile's own rim; a shadow is still backdrop)
            crop = img[y0:y1, x0:x1]; res, kk, b = geom.backdrop(crop); fgc = ndimage.binary_opening(res > 0.10); fg = np.zeros(mesh.shape, bool); fg[y0:y1, x0:x1] = fgc
            fgf = ndimage.binary_fill_holes(fg); mf = ndimage.binary_fill_holes(mesh); iou = geom.iou(fgf, mf); size = max(x1 - x0, y1 - y0)
            aff, fa = geom.align(fgf, mf)          # Klein drew the thing a little smaller or larger than the mesh: its outline's box is put back on the mesh's
            if aff is not None and geom.iou(fa, mf) > iou + 0.02:
                img = geom.warp_to(img, aff, mesh.shape); fg = geom.warp_to(fg.astype(np.float32), aff, mesh.shape) > 0.5; fgf = fa; st.setdefault("refit", []).append(tj["name"]); iou = geom.iou(fa, mf)
                if mk is not None: mk_t = geom.warp_to(SH.cut(tj, mk, fill=1.0)[0], aff, mesh.shape)
            else: aff = None
            gr = ndimage.gaussian_filter(img.mean(-1), 1.2); e = np.hypot(ndimage.sobel(gr, 0), ndimage.sobel(gr, 1)); ol = geom.outline(mesh); strong = e > np.percentile(e[y0:y1, x0:x1], 88)
            edge = float((ndimage.distance_transform_edt(~strong)[ol] <= max(3, 0.006 * size)).mean())          # the share of the mesh's outline that lies on a strong edge Klein drew
            q = float(max(np.clip((iou - 0.72) / 0.16, 0, 1), np.clip((edge - 0.45) / 0.25, 0, 1)))             # soft, and either sign will do: a grey thing on grey cannot be cut from its backdrop, a wire cage has no fill
            # backdrop showing inside the mesh's outline (Klein drew the thing smaller there): every backdrop pixel joined to the outside is not paint
            lab, n = ndimage.label(~fg); outside = np.unique(lab[~ndimage.binary_dilation(mesh, iterations=2)]); spill = np.isin(lab, outside[outside > 0]) & mesh
            if iou < 0.85 or spill.sum() > 0.12 * mesh.sum(): spill[:] = False          # the cut is not to be trusted here
            er = max(2, int(round(0.004 * px)) + 1); ok = ndimage.binary_erosion(mesh, iterations=er) & ~geom.depth_edges(R, vw, er) & ~ndimage.binary_dilation(spill, iterations=er + 1) & valid
            st["tiles"][tj["name"]] = dict(iou=round(iou, 3), edge=round(float(edge), 2), q=round(q, 2), spill=round(float(spill.sum() / max(mesh.sum(), 1)), 3))
            if q > 0:
                # tone: every view is brought to the level of the sheet it started from (the same guess everywhere), by the middle colour of what it shows, so a
                # view Klein lit from above is not a paler car than the one it lit from the side
                if ok.sum() <= 200: gains.append(None)
                else:
                    gain = float(np.clip(np.median(SH.cut(tj, ref, fill=float(back[0]))[0][ok] @ LUM) / max(float(np.median(img[ok] @ LUM)), 1e-3), 0.6, 1.7)); gains.append(gain); st["tiles"][tj["name"]]["gain"] = round(gain, 2)          # brightness only: a gain per colour turns tyres and chrome the body's colour
                c, w = geom.project_paint(vw, sc, P, N, img, ok, power=2); cols.append(c.astype(np.float32)); wts.append((w * q * (1.6 if tj["name"] == "own" else 1.0)).astype(np.float32)); tags.append(0 if tj["name"] == "own" else 3)          # the view the own picture anchors has the most say where it looks straight on
                mcols.append(geom.project_paint(vw, sc, P, N, mk_t if aff is not None else SH.cut(tj, mk, fill=1.0)[0], ok, power=2)[0].astype(np.float32) if mk is not None else np.zeros((S, 3), np.float32))
    mk = True if any_mat and cols else None
    gv = [g for g in gains if g]; gm = float(np.median(gv)) if gv else 1.0          # ... relative to the kind's middle view, so the whole thing is not made lighter or darker by it
    for i, g in enumerate(gains):
        if g: cols[i] = np.clip(cols[i] * (1 + 0.75 * (float(np.clip(g / gm, 0.75, 1.33)) - 1)), 0, 1)
    if not cols: cols, wts, tags = [H["rgb"].astype(np.float32)], [H["w"].astype(np.float32)], [0]; st["note"] = "no view kept: the kind's own picture only"
    C, Wt = np.stack(cols), np.stack(wts).astype(np.float32); sub = np.random.default_rng(0).choice(S, min(S, 120_000), replace=False)
    nb = cKDTree(P).query(P, k=8)[1]
    for i in range(len(Wt)): smw = Wt[i][nb]; Wt[i] = np.where(Wt[i] > 0, smw.mean(1) * (smw > 0).mean(1) ** 2, 0.0)          # a view's say fades toward the edge of what it saw: no hard seams
    w4 = Wt.astype(np.float64) ** 4; tot = w4.sum(0); got = Wt.max(0) > 0.02
    rgb = np.zeros((S, 3), np.float32); rgb[got] = ((w4[:, got, None] * C[:, got]).sum(0) / np.maximum(tot[got, None], 1e-30)).astype(np.float32)
    mask = None
    if mk is not None and len(mcols) == len(cols):          # each point takes the map of the view that painted it most; a mirrored or filled point takes its source's class below
        mask = np.take_along_axis(np.stack(mcols), Wt.argmax(0)[None, :, None], 0)[0]; mask[~got] = 0
    how = np.where(got, np.asarray(tags, np.uint8)[Wt.argmax(0)], 5).astype(np.uint8); st["seen"] = round(float(got.mean()), 3)
    # sides no view saw: the mirror image's paint if the thing is its own mirror image, then paint made in the mesh's own space
    sym = symmetric(P, sub, meta["diag"]); st["mirror_share"] = round(sym, 2)
    if sym > 0.8 and (~got).any() and got.any():
        qi = np.nonzero(~got)[0]; dm, im = cKDTree(P[got]).query(P[qi] * [-1, 1, 1]); use = dm < 0.01 * meta["diag"]; src = np.nonzero(got)[0][im[use]]; rgb[qi[use]] = rgb[src]; how[qi[use]] = 1
        if mask is not None: mask[qi[use]] = mask[src]
    done = how != 5; st["mirrored"] = round(float((how == 1).mean()), 3); st["filled"] = round(float((~done).mean()), 3)
    rgb = fill_space(P, N, rgb, done, meta["diag"], grey_grain=True)
    mat, mst = paint_class.classify(Kd, meta, P, N, rgb, how, mask); st["mat"] = mst
    if mask is not None and (~done).any() and done.any():          # a filled point has no map of its own: it takes its nearest painted neighbour's class if that is the thing's body (metal, gloss), else plain
        sd = np.nonzero(done)[0]; sd = sd[:: max(1, len(sd) // 100_000)]; nm = mat[sd][cKDTree(P[sd]).query(P[~done])[1]]; mat[~done] = np.where(np.isin(nm, (2, 4)), nm, 0)
    np.savez_compressed(f"{d}/tex_final.npz", rgb=rgb.astype(np.float16), how=how, mat=mat)
    st["s_bake"] = round(time.time() - t0, 1); json.dump(st, open(f"{d}/bake.json", "w"), indent=1)
    return kind, "bake", f"seen {st['seen']}, mirrored {st['mirrored']}, filled {st['filled']}, {st['s_bake']} s"

def pool(fn, world):
    from multiprocessing import get_context
    ks = list(kinds(world)); t0 = time.time(); res = []
    if not ks: return res
    with get_context("spawn").Pool(min(4, len(ks))) as p:
        for r in p.imap_unordered(fn, [(world, k) for k in ks]): res.append(r); print(*[x for x in r if not isinstance(x, dict)], flush=True)
    print(f"{fn.__name__.split('_')[0]}: {len(ks)} kinds, {time.time() - t0:.0f} s", flush=True); return res

if __name__ == "__main__":
    cmd = sys.argv[1]; world = os.path.abspath(os.environ["WORLD_DIR"])
    if cmd == "prep": pool(prep_one, world)
    elif cmd == "ref":
        res = pool(ref_one, world); order = list(kinds(world)); jobs = [j for r in sorted(res, key=lambda r: order.index(r[0])) for j in r[1]["jobs"]]
        json.dump(jobs, open(f"{where.paint_dir(world)}/jobs.json", "w"), indent=1); print(f"{len(jobs)} sheets to paint, {sum(j['w'] * j['h'] for j in jobs) / 1e6:.1f} megapixels", flush=True)
    elif cmd == "maps": maps(world)
    elif cmd == "bake": pool(bake_one, world)
