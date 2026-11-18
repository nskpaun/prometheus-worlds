"""The assembler: one built world -> one directly written splat, with no training. Everything made before it comes together here:
  layout     boxes.json, shapes/plan.json, shapes/placed/NNN.npz (layout.py, sculpt.py) and the room's own geometry (arch.py)
  materials  mats.Mats: every surface laid for this room, every kit piece, every plain-form part and flat thing's content
  objects    painted.placed / painted.discs: every sculpted kind's paint on its own mesh, discs fitted to edges, a material class per disc
  light      light.light / light.write, LAST, over everything: sources, sun with bar shadows, exposure, contact, shine, real mirrors
The splat is written as flat discs: a lattice on flat faces, thrown darts on free meshes, as fine as the nearest place a
walker's eye can be, inside one budget of discs (a frame-rate limit).
  assemble.py     -> WORLD_DIR/world.ply (+ world_sh.ply: the glossy discs with their harmonics), collider.glb, scene.json (what
                     pipeline.py registers), build.json (timings, counts)
thread/.venv, CPU only (Embree uses every core, 6 to 9 GB). Nothing here knows a world."""
import os, sys, json, time
import numpy as np
import where, level as C, mats, painted, light
import light_base
SIG, OSIG, NEAR, OBJ_MIN, EYE = 0.56, 0.75, 0.8, 0.003, 1.7
SAID, SAID_MIN = 0.6, 0.002          # a thing the sentence names is laid finer (its paint is 2.2 mm)
MIRROR_DISCS = 600_000                                                   # what the light part hangs behind each flat mirror (its cap)
LEVELS = np.array([2.0 * 2 ** (k / 4) for k in range(21)]) / 1000
snap = lambda s: LEVELS[np.clip(np.round(4 * np.log2(np.maximum(s, 1e-6) / LEVELS[0])).astype(int), 0, len(LEVELS) - 1)]
SKYC = {"night": [26, 36, 70], "dusk": [235, 150, 105], "day": [214, 229, 246]}
BOOKS = [(8, 0.55, 0.38), (215, 0.5, 0.32), (130, 0.4, 0.28), (40, 0.45, 0.55), (350, 0.5, 0.3), (30, 0.3, 0.2), (265, 0.3, 0.3), (48, 0.25, 0.75)]
T = {}; NOPAINT = []
def tick(k, t0): T[k] = round(T.get(k, 0) + time.time() - t0, 1)

def tri_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); a = np.linalg.norm(n, axis=1); return n / np.maximum(a, 1e-12)[:, None], a / 2

def split_long(V, F, UV, max_edge):
    """Longest-edge bisection until no edge is longer than max_edge (keeps a thin strip's width, unlike 1 -> 4). -> tris (f, 3, 3), uv (f, 3, 2)"""
    T3 = V[F].astype(np.float64); U = UV[F].astype(np.float64)
    for _ in range(40):
        e = np.stack([np.linalg.norm(T3[:, i] - T3[:, (i + 1) % 3], axis=1) for i in range(3)], 1); k = e.argmax(1); big = e.max(1) > max_edge
        if not big.any(): break
        idx = np.nonzero(big)[0]; kk = k[idx]; a, b, c = kk, (kk + 1) % 3, (kk + 2) % 3; r = np.arange(len(idx))
        def cut(X):
            A, B, Cc = X[idx][r, a], X[idx][r, b], X[idx][r, c]; M = (A + B) / 2; return np.stack([A, M, Cc], 1), np.stack([M, B, Cc], 1)
        t1, t2 = cut(T3); u1, u2 = cut(U); T3 = np.concatenate([T3[~big], t1, t2]); U = np.concatenate([U[~big], u1, u2])
    return T3, U

# ---------------------------------------------------------------- the level as pieces
def level(world):
    b = C.layout(world); S = C.shell(world, b); P = C.plan(world); mood = light_base.mood_of(b["request"], b.get("mood")); tm = mood["time"]; M = mats.Mats(world)
    wall_cls = S["materials"].get("wall", {}).get("cls", "")
    pieces, lm, lobj = [], [], []          # pieces to lay discs on; meshes for the light (blockers); layout-object records for the light's emitter and gloss rules
    def add_lm(V, F, name, painted, blocker=True):
        lm.append({"V": np.ascontiguousarray(V, np.float32), "F": np.ascontiguousarray(F, np.int32), "name": name, "painted": painted, "blocker": blocker}); return len(lm) - 1
    for k, m in enumerate(S["meshes"]):
        g = m["group"]; V, F, UV = np.asarray(m["V"], np.float64), np.asarray(m["F"]), np.asarray(m["UV"], np.float64); emit = None; tint = None; wall_v = False; cls = S["materials"].get(g, {}).get("cls", "")
        if g == "glass": emit = np.float32(SKYC[tm]); key = None
        elif g == "emitter": emit = np.float32([255, 244, 222]); key = None
        elif g in ("wall", "floor", "ceiling"):
            surf = m.get("surface")
            if m["piece"] == "wall" and surf in M.tex: key, wall_v = surf, True
            elif m["piece"] in ("floor", "ceiling") and surf in M.tex: key = surf
            else: key = "g:" + g if ("g:" + g) in M.tex else {"wall": "x0", "floor": "floor", "ceiling": "ceil"}[g]
        else: key = "g:" + g
        fn = None
        if emit is None and not (m["piece"] in ("wall", "floor", "ceiling") and m.get("surface") in M.tex): fn = (lambda P_, N_, sp_, m=m, k=k: M.arch(m, P_, N_, sp_, k))          # every kit piece by the materials' rules
        li = add_lm(V, F, "shell:" + m["piece"], False, blocker=emit is None)
        pieces.append(dict(name="shell:" + m["piece"], flat=True, V=V, F=F, UV=UV, key=key, tint=tint, emit=emit, wall_v=wall_v, lm=li, seed=k, gloss=(0.0, 0), fn=fn))
    plain = {i: (kind, pl["form"]) for kind, pl in (P.get("plain") or {}).items() for i in pl["objects"]}; sculpt = {i: kind for kind, v in P["kinds"].items() for i in v["objects"]}
    for i, o in enumerate(b["objects"]):
        if i in S["claimed"]: continue
        z = C.placed(world, i); V = np.asarray(z["V"], np.float64); F = C.face_air(V, np.asarray(z["F"])); rec = dict(o, mesh=None)
        nw = o["name"].lower().replace("-", " ").split()
        if o.get("form") in FLAT_FORMS: rec = dict(rec, name="flat thing", kind="", look="")          # a picture, a sign, a rug or a cloth gives no light whatever its words say (a "fire map", a "lamp poster")
        elif max(o.get("got") or o.get("size") or [1]) < 0.3 and o.get("attach") not in ("wall", "ceiling") and not any(w.startswith("candle") for w in nw) and light_base._has(" ".join(str(o.get(q) or "") for q in ("name", "kind", "look")).lower().replace("_", " ").replace("-", " "), light_base.LAMP):
            rec = dict(rec, name="small thing", kind="", look="")          # a hand torch or a spare bulb lying about is not switched on (the light rule would give it a room lamp's glow)
        elif (o.get("got") or o.get("size") or [1, 1, 1])[1] < 0.2 and light_base._has(" ".join(nw), light_base.FIRE): rec = dict(rec, name="floor slab", kind="", look="")          # a hearth stone is the slab a fire stands on, not the fire
        elif "fire" in nw and nw[-1] != "fire": rec = dict(rec, name=" ".join(w for w in nw if w != "fire") or "thing", kind="", look=" ".join(w for w in str(o.get("look") or "").split() if w.lower() != "fire"))        # "fire" before another noun (extinguisher, blanket, door, alarm) is not a fire
        if o.get("part_of") is not None: rec = dict(name="cord", kind="cord", look="", mesh=None)        # a hanging thing's cord is not the lamp
        if i in sculpt:
            kind = sculpt[i]
            try: pc = painted.placed(world, kind, V)
            except Exception as e:          # a kind whose paint is missing: built anyway, in one plain colour, and said so
                print(f"{kind} has no paint ({type(e).__name__}): built in one plain colour", flush=True); NOPAINT.append(kind); li = add_lm(V, F, o["name"], False); rec["mesh"] = li
                pieces.append(dict(name=o["name"], flat=False, V=V, F=F, lm=li, seed=i, key=None, tint=None, emit=None, gloss=(0.0, 0), obj=i, fn=lambda P_, N_, sp_: np.broadcast_to(np.float32([150, 145, 138]), P_.shape))); lobj.append(rec); continue
            rgb = pc["rgb"] / 255.0
            kl = P["kinds"][kind]["look"]; ol = (P.get("objects") or {}).get(str(i), {}).get("look") or o.get("look", "")
            cw0, _ = colour_word(kl.split(",", 1)[-1]); cw1, tgt = colour_word(ol.split(",", 1)[-1])
            if cw1 and cw1 != cw0 and P["kinds"][kind]["objects"][0] != i: rgb = recolour(np.clip(rgb, 0, 1), tgt, pc["mat"])          # a copy of a shape whose own look names another colour
            li = add_lm(V, F, o["name"], True); rec["mesh"] = li
            pieces.append(dict(name=o["name"], flat=False, V=V, F=F, lm=li, seed=i, xyz=pc["xyz"], nrm=pc["nrm"], rgb=(np.clip(rgb, 0, 1) * 255).astype(np.float32), mat=pc["mat"], gloss=light_base.gloss_of([o])[0], obj=i, said=bool(o.get("said"))))
        else:
            kind, form = plain.get(i, (o.get("kind"), o.get("form"))); part = np.asarray(z["part"]) if "part" in z else np.zeros(len(F), int); names = [str(n) for n in z["names"]] if "names" in z else ["body"]
            li = add_lm(V, F, o["name"], False); rec["mesh"] = li; Pt = M.plain(kind, o, V, F, part, names)          # every part its own material, flat things their content
            bk = [p for p, nm in enumerate(names) if nm.split()[0] == "book"] if form != "books" else []; Pb = None
            if bk: Pb = book_painter(M, kind, o, V, F[np.isin(part, bk)], Pt)          # a bookcase's books: mats paints spines only on its own "books" form
            for p, nm in enumerate(names):
                Fp = F[part == p]
                if not len(Fp): continue
                inf = Pt.info(nm); emit = inf["emit"]; key = None; tint = np.float32([255, 200, 120]) if emit else None
                pc = dict(name=f"{o['name']}:{nm}", flat=False, V=V, F=Fp, lm=li, seed=i * 31 + p, key=key, tint=tint, emit=np.float32(tint) if emit else None, gloss=light_base.gloss_of([o])[0], obj=i,
                          fn=None if emit else (lambda P_, N_, sp_, Pt=Pt, nm=nm: Pt.colour(nm, P_, N_, sp_)[0]), mcls=PLAIN_CLS.get(str(inf.get("cls")), None))
                if Pb is not None and p in bk: pc["fn"] = (lambda P_, N_, sp_, Pb=Pb: Pb.colour("cover", P_, N_, sp_)[0]); pc["mcls"] = None
                # a boxy part (a rug, a frame, a board, a crate's panels: every face square to the room) is laid like the kit's flat pieces, on a true-scale lattice: crisp edges, no fuzz
                Tp = V[Fp]; nn, ar = tri_normals(V, Fp); ax = np.abs(nn).argmax(1)
                if float(ar[np.abs(nn).max(1) > 0.999].sum()) > 0.98 * float(ar.sum()) and ar.sum() > 1e-6:
                    uu = np.where((ax == 0)[:, None], Tp[:, :, 2], Tp[:, :, 0]); vv = np.where((ax == 1)[:, None], Tp[:, :, 2], Tp[:, :, 1])
                    pc.update(flat=True, V=Tp.reshape(-1, 3), F=np.arange(3 * len(Fp)).reshape(-1, 3), UV=np.stack([uu, vv], -1).reshape(-1, 2), wall_v=False)
                pieces.append(pc)
        lobj.append(rec)
    return dict(b=b, S=S, mood=mood, M=M, pieces=pieces, lm=lm, lobj=lobj, room=np.array(S["room"], float))

# what the materials say a plain-form part is made of -> the light's (class, gloss)
PLAIN_CLS = {"mirror": (light.MIRROR, 1.0), "glass": (light.GLASS, 0.9), "metal": (light.METAL, 0.55), "ceramic": (light.GLOSS, 0.5), "leather": (light.GLOSS, 0.3)}
FLAT_FORMS = ("frame", "board", "rug", "cloth", "books")
MORE = {"copper": (22, 0.7, 0.62), "bronze": (30, 0.6, 0.45), "brass": (42, 0.65, 0.7), "ivory": (48, 0.2, 0.9), "beige": (40, 0.25, 0.8), "tan": (32, 0.4, 0.7)}
def colour_word(text):
    for w in str(text).lower().replace(",", " ").replace("-", " ").split():
        if w in C.COLOURS: return w, C.COLOURS[w]
        if w in MORE: return w, MORE[w]
    return None, None
def recolour(rgb, target, mat=None):
    """level.recolour (the dominant saturated hue turns into the target), and for a thing whose body has no hue (cream, white, black, grey):
    the commonest colour of its paint and gloss points turns into the target, keeping its light and shade. rgb (n, 3) 0..1."""
    hsv = C.rgb2hsv(rgb); body = np.ones(len(rgb), bool) if mat is None else ~np.isin(mat, (1, 2, 3))
    if (hsv[body, 1] > 0.28).mean() > 0.35: return C.recolour(rgb, target)
    q = (np.clip(rgb[body], 0, 0.999) * 12).astype(int); key = (q[:, 0] * 12 + q[:, 1]) * 12 + q[:, 2]; top = np.bincount(key, minlength=1728).argmax(); dom = np.median(rgb[body][key == top], 0)
    d = np.linalg.norm(rgb - dom, axis=1); w = np.clip((0.30 - d) / 0.12, 0, 1) * body; lum = rgb @ np.float32([0.299, 0.587, 0.114]); l0 = max(float(dom @ np.float32([0.299, 0.587, 0.114])), 0.05)
    t = C.hsv2rgb(np.float32([[target[0], target[1], target[2]]]))[0]; new = np.clip(t[None] * (lum / l0)[:, None], 0, 1)
    return (rgb * (1 - w[:, None]) + new * w[:, None]).astype(np.float32)
def book_painter(M, kind, o, V, Fb, Pt):
    """mats' spine painter (its "books" form) on the books standing in a bookcase: each connected block of faces is one book."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    Pb = M.plain(kind, o, V); Pb.form = "books"; Pb.rec = dict(Pb.rec, parts=dict(Pb.rec.get("parts") or {})); Pb.rec["parts"]["cover"] = Pt.prec("book 0"); Vo = Pb.own(V)
    e = np.concatenate([Fb[:, [0, 1]], Fb[:, [1, 2]]]); n, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))), directed=False); used = np.unique(lab[np.unique(Fb)])
    lo = np.full((n, 3), np.inf); hi = np.full((n, 3), -np.inf); np.minimum.at(lo, lab, Vo); np.maximum.at(hi, lab, Vo); Pb.books = np.c_[lo[used], hi[used]]; return Pb

def start_view(Lv, Sc):
    """Where a visitor stands first and which way they look: a free spot on the floor itself (0.3 m clear of every surface, not
    shut inside a thing, not up on a platform or a table), a few metres back from the things the sentence names (else from
    the middle of the room) with a clear view towards them, looking level at them. -> eye, forward"""
    import open3d as o3d
    R = Lv["room"]; b = Lv["b"]; e0 = np.array([R[0] / 2, min(EYE, R[1] - 0.3), R[2] / 2])
    def bad(E):
        E = np.asarray(E, np.float32).reshape(-1, 3); d = Sc.sc.compute_distance(o3d.core.Tensor(E)).numpy()
        D = np.float32([[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0]])
        t = np.stack([Sc.cast(E, np.tile(dd, (len(E), 1)))[0] for dd in D], 1)          # shut in on every side within a metre and a roof just overhead: inside a thing
        return (d < 0.3) | ((t[:, :4] < 1.0).all(1) & (t[:, 4] < 0.6)) | (E[:, 0] < 0.4) | (E[:, 0] > R[0] - 0.4) | (E[:, 2] < 0.4) | (E[:, 2] > R[2] - 0.4)
    xs, zs = np.arange(0.4, R[0] - 0.4 + 1e-6, 0.2), np.arange(0.4, R[2] - 0.4 + 1e-6, 0.2)
    cand = np.array([[x, e0[1], z] for x in xs for z in zs] or [e0]); ok = ~bad(cand)
    down = Sc.cast(cand.astype(np.float32), np.tile(np.float32([0, -1, 0]), (len(cand), 1)))[0]; floor = ok & (np.abs(down - e0[1]) < 0.12)
    if floor.any(): ok = floor          # a room whose whole floor is covered (a deck, a stage): any free spot then
    said = [np.array(o["prims"][0]["c"], float) for o in b["objects"] if o.get("said")]; f = None
    aim = np.mean(said, 0) if said else np.array([R[0] / 2, 0, R[2] / 2])
    if ok.any():
        # the best spot: about 0.4 of the room's longer side away from what is looked at, and open in front (the mean free way over a fan
        # of five rays towards it, counted to 6 m): a spot behind a pillar or with its nose against a cupboard scores low
        C = cand[ok]; to = aim[None] - C; to[:, 1] = 0; dist = np.linalg.norm(to, axis=1); u = to / np.maximum(dist, 1e-6)[:, None]; free = np.zeros(len(C))
        for a in (-0.5, -0.25, 0.0, 0.25, 0.5):
            dd = np.stack([u[:, 0] * np.cos(a) - u[:, 2] * np.sin(a), np.zeros(len(C)), u[:, 0] * np.sin(a) + u[:, 2] * np.cos(a)], 1).astype(np.float32)
            t = Sc.cast(C.astype(np.float32), dd)[0]; free += np.minimum(np.where(np.isfinite(t), t, 6.0), 6.0) / 5
        want = min(4.5, 0.4 * max(R[0], R[2])); eye = C[int(np.argmax(free - 0.8 * np.abs(dist - want)))]
    else: eye = e0
    f = aim - eye; f[1] = 0
    if np.linalg.norm(f) < 0.5: f = None
    if f is None:
        D = np.array([[np.cos(a), 0, np.sin(a)] for a in np.arange(16) * np.pi / 8], np.float32); t = Sc.cast(np.tile(eye.astype(np.float32), (16, 1)), D)[0]; f = D[int(np.argmax(np.where(np.isfinite(t), t, 0)))].astype(float)
    return eye, f / np.linalg.norm(f)

def walker_eyes(Lv, eye):
    R = Lv["room"]; xs = np.arange(0.3, R[0] - 0.3 + 1e-6, 0.4); zs = np.arange(0.3, R[2] - 0.3 + 1e-6, 0.4); X, Z = np.meshgrid(xs, zs); E = np.stack([X.ravel(), np.full(X.size, EYE), Z.ravel()], 1); ok = np.ones(len(E), bool)
    for p in Lv["pieces"]:
        if p["flat"] and p.get("obj") is None and not p["name"].startswith(("shell:platform", "shell:column", "shell:pier")): continue
        lo, hi = p["V"][np.unique(p["F"])].min(0), p["V"][np.unique(p["F"])].max(0)
        if lo[1] > EYE + 0.1 or hi[1] < 0.25: continue
        g = 0.2 if hi[1] > EYE - 0.5 else -0.1; ok &= ~((E[:, 0] > lo[0] - g) & (E[:, 0] < hi[0] + g) & (E[:, 2] > lo[2] - g) & (E[:, 2] < hi[2] + g))
    return np.concatenate([E[ok], np.asarray(eye, float)[None]])

def nearest_eye(Sc, E, P, N, chunk=5_000_000):
    """Metres from each point to the nearest eye that sees it from the front (inf if none does)."""
    best = np.full(len(P), np.inf, np.float32); N = N.astype(np.float32); Q = (P + N * 0.012).astype(np.float32); per = max(1, chunk // max(len(P), 1))
    for a in range(0, len(E), per):
        e = E[a:a + per].astype(np.float32); v = Q[None] - e[:, None]; dist = np.linalg.norm(v, axis=-1); d = v / dist[..., None]; front = (-(d * N[None]).sum(-1)) > 0.03
        ii, jj = np.nonzero(front & (dist < best[None]))
        if not len(ii): continue
        t, _ = Sc.cast(e[ii], d[ii, jj]); seen = t > dist[ii, jj] - 0.03; np.minimum.at(best, jj[seen], dist[ii, jj][seen])
    return best

def poisson(P, r, rounds=2, seed=0):
    from scipy.spatial import cKDTree
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

def lattice(T3, U, g):
    """Lattice points (spacing g in UV metres, a lattice shared by every triangle) inside triangles T3 (f, 3, 3) with UV U (f, 3, 2). A triangle the
    lattice misses gets its centroid. -> points (n, 3), uv (n, 2), triangle index (n,)"""
    lo = np.ceil(U.min(1) / g - 1e-9).astype(np.int64); hi = np.floor(U.max(1) / g + 1e-9).astype(np.int64); nu = np.maximum(hi[:, 0] - lo[:, 0] + 1, 0); nv = np.maximum(hi[:, 1] - lo[:, 1] + 1, 0); cnt = nu * nv
    Ps, Us, Ts = [], [], []; hit = np.zeros(len(T3), bool); order = np.argsort(-cnt); a = 0
    while a < len(order):                                   # chunks of about 4 million candidates
        c = np.cumsum(cnt[order[a:]]); bnd = int(np.searchsorted(c, 4_000_000)) + 1; sel = order[a:a + bnd]; a += bnd; sel = sel[cnt[sel] > 0]
        if not len(sel): continue
        tid = np.repeat(sel, cnt[sel]); off = np.arange(len(tid)) - np.repeat(np.cumsum(cnt[sel]) - cnt[sel], cnt[sel]); iu = off % np.repeat(nu[sel], cnt[sel]) + lo[tid, 0]; iv = off // np.repeat(nu[sel], cnt[sel]) + lo[tid, 1]
        q = np.stack([iu, iv], 1) * g; A, B, Cc = U[tid, 0], U[tid, 1], U[tid, 2]; v0, v1, v2 = B - A, Cc - A, q - A; den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]; den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den; w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den; w0 = 1 - w1 - w2; e = -1e-7; ins = (w0 >= e) & (w1 >= e) & (w2 > e) & (w0 + 0 * w1 < 1 + 1e-7)
        # a point on a shared edge is taken by one side only: ties broken by the half-open rule w1 >= 0, w2 > 0 is not exact for every edge; duplicates there are harmless (same colour, same place)
        tid, w0, w1, w2, q = tid[ins], w0[ins], w1[ins], w2[ins], q[ins]; hit[tid] = True
        Ps.append(T3[tid, 0] * w0[:, None] + T3[tid, 1] * w1[:, None] + T3[tid, 2] * w2[:, None]); Us.append(q); Ts.append(tid)
    miss = np.nonzero(~hit)[0]
    if len(miss): Ps.append(T3[miss].mean(1)); Us.append(U[miss].mean(1)); Ts.append(miss)
    return np.concatenate(Ps), np.concatenate(Us), np.concatenate(Ts)

VEH_GLOSS = (0.3, 0.1)   # a vehicle's paint: gloss, and matte paint
BUDGET_MAX = 6.9e6       # a frame-rate limit, measured in the viewer (1024 x 768) on an M4 Pro: the widest view of a 14 x 12 m hall draws 7.4 million discs at 29 to 33 frames a second, 7.8 million at 28 to 29
FIT_GROW = 1.35          # discs the edge fitting adds on a sculpted mesh (measured)
def build(world, budget=None):
    import trimesh
    from scipy.spatial import cKDTree
    say = lambda t: print(t, flush=True)
    t00 = time.time(); t0 = time.time(); Lv = level(world); b, M, room = Lv["b"], Lv["M"], Lv["room"]; W = world; tick("level_s", t0)
    # ---- the frame the splat is written in (z mirrored, the room's middle at the origin, its longer half-side 1), and where a visitor starts
    t0 = time.time(); k_ = 1.0 / (0.5 * max(room[0], room[2])); m_ = np.array([room[0] / 2, 1.5, -room[2] / 2]); placing = {"m": m_.tolist(), "k": k_}
    Sc = light_base.Scene([x for x in Lv["lm"] if x["blocker"]]); eye0, fwd0 = start_view(Lv, Sc); E = walker_eyes(Lv, eye0); tick("cams_s", t0)
    info = {"eyes": len(E), "pieces": len(Lv["pieces"]), "meshes": len(Lv["lm"]), "tris": int(sum(len(x["F"]) for x in Lv["lm"]))}
    say(f"{info['pieces']} pieces, {info['tris'] // 1000}k triangles: laying discs")
    # ---- what every patch needs
    t0 = time.time(); flats, frees = [], []
    for p in Lv["pieces"]:
        if p["flat"]:
            T3, U = split_long(p["V"], p["F"], p["UV"], 0.4); n = np.cross(T3[:, 1] - T3[:, 0], T3[:, 2] - T3[:, 0]); ar = np.linalg.norm(n, axis=1) / 2; ok = ar > 1e-9; T3, U, n, ar = T3[ok], U[ok], n[ok], ar[ok]; n /= (2 * ar)[:, None]
            edge = np.stack([np.linalg.norm(T3[:, i] - T3[:, (i + 1) % 3], axis=1) for i in range(3)], 1).max(1); ua, ub = U[:, 1] - U[:, 0], U[:, 2] - U[:, 0]; uar = np.abs(ua[:, 0] * ub[:, 1] - ua[:, 1] * ub[:, 0]) / 2
            bad = (uar < 0.25 * ar) | (uar > 4 * ar)             # a face whose UV is not true scale: laid in its own plane instead
            if bad.any():
                e1 = T3[bad, 1] - T3[bad, 0]; e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(n[bad], e1); U[bad] = np.stack([(T3[bad] * e1[:, None]).sum(-1), (T3[bad] * e2[:, None]).sum(-1)], -1)
            flats.append(dict(p=p, T3=T3, U=U, n=n, ar=ar, hmin=2 * ar / edge))
        else:
            mesh = trimesh.Trimesh(p["V"], p["F"], process=False); area = float(mesh.area)
            if area < 1e-8: continue
            frees.append(dict(p=p, mesh=mesh, area=area))
    cen = np.concatenate([f["T3"].mean(1) for f in flats]); cn = np.concatenate([f["n"] for f in flats]); d = nearest_eye(Sc, E, cen, cn); a = 0
    for f in flats: f["d"] = d[a:a + len(f["ar"])]; a += len(f["ar"])
    for f in frees:
        n = int(np.clip(f["area"] / 0.03 ** 2 * 1.5, 200, 60000)); Pq, fi = trimesh.sample.sample_surface(f["mesh"], n, seed=1); Nq = f["mesh"].face_normals[fi]
        c = (f["p"]["V"].min(0) + f["p"]["V"].max(0)) / 2; near = E[np.linalg.norm(E - c, axis=1) < 4.0 + np.linalg.norm(f["p"]["V"].max(0) - f["p"]["V"].min(0)) / 2]
        ee = near if len(near) else E; d1 = nearest_eye(Sc, ee, Pq, Nq); d2 = np.full(len(Pq), np.inf, np.float32) if f["mesh"].is_watertight else nearest_eye(Sc, ee, Pq, -Nq)       # a sheet (cloth, a rug) is seen from either side: its discs face the side that is looked at
        f["q"] = Pq; f["qd"] = np.minimum(d1, d2); f["qflip"] = d2 < d1
    tick("need_s", t0); info["seen_share_flat"] = round(float(np.isfinite(d).mean()), 3)
    # ---- one finest spacing from the budget
    t0 = time.time()
    for f in flats:          # a plain surface (plaster, glass) needs no fine discs: up to 4x coarser where its picture has nothing finer to show
        T_ = M.tex.get(f["p"]["key"]) if f["p"]["key"] else None; f["det"] = LEVELS[-1] if f["p"]["emit"] is not None else (T_.detail_spacing(LEVELS) if (T_ is not None and not f["p"].get("fn")) else LEVELS[0])
    def want_flat(f, s0):
        view = s0 * np.where(np.isfinite(f["d"]), np.clip(f["d"] / NEAR, 1, 4), 4.0); s = np.maximum(view, np.minimum(f["det"], 4 * view))
        return np.maximum(LEVELS[0], np.minimum(snap(s), snap(0.7 * f["hmin"])))
    def want_free(f, s0):
        base = max(SAID_MIN, SAID * s0) if f["p"].get("said") else max(OBJ_MIN, s0)
        return snap(base * np.where(np.isfinite(f["qd"]), np.clip(f["qd"] / NEAR, 1, 3), 3.0))
    n_back = sum(float(f["ar"][f["hmin"] > 0.06].sum()) for f in flats) / 0.04 ** 2
    def count(s0): return sum(float((f["ar"] / want_flat(f, s0) ** 2).sum()) for f in flats) + sum(float((f["area"] / len(f["qd"]) / want_free(f, s0) ** 2).sum()) * (FIT_GROW if "xyz" in f["p"] else 1.05) for f in frees) + n_back
    if not budget:          # 5.6 million discs for a room of 260 m2 of flat surface (a 5 x 6 m study); a larger level gets more, up to BUDGET_MAX
        budget = int(np.clip(5.6e6 * (sum(float(f["ar"].sum()) for f in flats) / 260.0) ** 0.5, 5.6e6, BUDGET_MAX))
    n_mirror = len({p["lm"] for p in Lv["pieces"] if p.get("mcls") and p["mcls"][0] == light.MIRROR}); info["budget"] = int(budget); info["mirrors"] = n_mirror
    budget = max(budget - MIRROR_DISCS * n_mirror, 0.6 * budget); lo_, hi_ = 0.002, 0.03
    for _ in range(14):
        mid = (lo_ * hi_) ** 0.5
        if count(mid) > budget: lo_ = mid
        else: hi_ = mid
    s0 = hi_; info["s0_mm"] = round(s0 * 1000, 2); tick("solve_s", t0)
    # ---- discs
    t0 = time.time(); P, Nn, Cc, Sp, Th, Op, Mi, Em, Gl, Pi, Mt = [], [], [], [], [], [], [], [], [], [], []; CUR = [0]; pidx = {id(p): k for k, p in enumerate(Lv["pieces"])}
    def put(p, n, c, sp, th, op, lmi, em, gl, mat=None):
        P.append(p.astype(np.float32)); Nn.append(n.astype(np.float32)); Cc.append(np.clip(c, 0, 255).astype(np.float32)); Sp.append(np.asarray(sp, np.float32)); Th.append(np.full(len(p), th, np.float32)); Op.append(np.full(len(p), op, np.float32))
        Mi.append(np.full(len(p), lmi, np.int32)); Pi.append(np.full(len(p), CUR[0], np.int32)); Mt.append(np.full(len(p), 255, np.uint8) if mat is None else np.asarray(mat, np.uint8)); Em.append(np.full(len(p), em, bool)); Gl.append(np.full(len(p), gl, bool))
    n_flat = n_free = n_bk = 0
    for f in flats:
        p = f["p"]; CUR[0] = pidx[id(p)]; s = want_flat(f, s0); has_back = f["hmin"] > 0.06
        for g in list(np.unique(s)) + [-1.0]:
            back = g < 0; step = 0.04 if back else g; sel = np.nonzero(has_back if back else (s == g))[0]
            if not len(sel) or (back and p["emit"] is not None): continue
            pts, uv, tid = lattice(f["T3"][sel], f["U"][sel], step); nrm = f["n"][sel][tid]
            if back:          # a solid coarse layer behind big faces, left out where it would poke through the other side of a thin piece
                t, _ = Sc.cast(pts - nrm * 0.002, -nrm); keep = t > 0.04; pts, uv, nrm = pts[keep], uv[keep], nrm[keep]
                if not len(pts): continue
            if p["emit"] is not None: col = np.broadcast_to(p["emit"], pts.shape)
            elif p.get("fn"): col = p["fn"](pts, nrm, step)
            else: col = M.colour(p["key"], uv[:, 0], uv[:, 1], step, p["tint"], p["seed"], flip_v_height=True if p["wall_v"] else None)
            put(pts - (nrm * 0.012 if back else 0), nrm, col, np.full(len(pts), SIG * step), 0.004 if back else 0.0005, 0.995 if back else 0.97, p["lm"], p["emit"] is not None, p["gloss"][0] > 0.05 and not back)
            if back: n_bk += len(pts)
            else: n_flat += len(pts)
    for f in frees:
        p = f["p"]; CUR[0] = pidx[id(p)]; mesh = f["mesh"]; sq = want_free(f, s0); gmin = float(sq.min()); n = int(min(5e6, f["area"] / (0.42 * gmin) ** 2)); Pd, fi = trimesh.sample.sample_surface(mesh, n, seed=0); Nd = mesh.face_normals[fi]
        _, j = cKDTree(f["q"]).query(Pd, k=1, workers=4); g = sq[j]; seen = np.isfinite(f["qd"])[j]; g = np.where(seen, g, np.minimum(g, snap(np.array([0.012]))[0])); live = np.ones(len(Pd), bool)
        if seen.any() and (~seen).any():
            # an unseen point within 3 cm of a seen one: if it faces the other way it is the back of a thin part (left out on a closed mesh);
            # if not, it is the flank of a fold or an edge that a walker WILL see from some other spot: laid as finely as its seen neighbour
            # (leaving these out makes holes in curtains seen off their axis)
            si = np.nonzero(seen)[0][::2]; us = np.nonzero(~seen)[0]; dd, jn = cKDTree(Pd[si]).query(Pd[us], k=1, distance_upper_bound=0.03, workers=4); near = np.isfinite(dd); sj = si[np.minimum(jn, len(si) - 1)]
            back = near & ((Nd[us] * Nd[sj]).sum(1) < -0.3); side = near & ~back
            if mesh.is_watertight: live[us[back]] = False
            g[us[side]] = g[sj[side]]; seen[us[side]] = True
        keep, sp = [], []
        for gv in np.unique(g):
            for cls in (True, False):
                idx = np.nonzero((g == gv) & (seen == cls) & live)[0]
                if not len(idx): continue
                idx = idx[:: max(1, int((gv / gmin) ** 2 * 0.8))]; k2 = idx[poisson(Pd[idx], 0.8 * gv, rounds=3)]; keep.append(k2); sp.append(np.full(len(k2), gv))
        if not keep: continue
        keep, sp = np.concatenate(keep), np.concatenate(sp); pts, nrm = Pd[keep], Nd[keep] * np.where(f["qflip"][j[keep]], -1.0, 1.0)[:, None]
        sig = OSIG * sp; mat = None
        if "xyz" in p:
            pts, nrm, sig, col, mat = painted.discs(p, pts, nrm, sig)          # a sculpted thing: discs fitted to its edges, paint from its own painted points that face the same way
            # its discs face the way its facets do, and a sculpted skin is finely dented: lit with shine, every disc then catches its own highlight (white
            # speckle on car paint). Each disc takes the mean facing of its neighbours within 4 cm that face its way (an edge stays an edge).
            dd, jj = cKDTree(pts).query(pts, k=16, distance_upper_bound=0.04, workers=4); nn = np.concatenate([nrm, np.zeros((1, 3))])[jj]; w = ((nn * nrm[:, None, :]).sum(-1) > 0.6)[..., None]
            ns = (nn * w).sum(1); ln = np.linalg.norm(ns, axis=1, keepdims=True); nrm = np.where(ln > 1e-6, ns / np.maximum(ln, 1e-9), nrm)
        elif p.get("emit") is not None: col = np.broadcast_to(p["emit"], pts.shape)
        elif p.get("fn"):
            col = np.empty((len(pts), 3), np.float32)
            for gv in np.unique(sp): q = sp == gv; col[q] = p["fn"](pts[q], nrm[q], gv)
        else:
            ax = np.abs(nrm).argmax(1); u = np.where(ax == 0, pts[:, 2], pts[:, 0]); v = np.where(ax == 1, pts[:, 2], pts[:, 1]); col = np.empty((len(pts), 3), np.float32)
            for gv in np.unique(sp): q = sp == gv; col[q] = M.colour(p["key"], u[q], v[q], gv, p["tint"], p["seed"])
        put(pts, nrm, col, sig, 0.0006, 0.97, p["lm"], p.get("emit") is not None, p["gloss"][0] > 0.05, mat); n_free += len(pts)
    P, Nn, Cc, Sp, Th, Op, Mi, Em, Pi, Mt = (np.concatenate(x) for x in (P, Nn, Cc, Sp, Th, Op, Mi, Em, Pi, Mt)); n = len(P); tick("discs_s", t0)
    info.update(discs=n, flat_discs=n_flat, object_discs=n_free, backing_discs=n_bk)
    # ---- real light, last, over everything (the light's one call), then its writer
    say(f"{n / 1e6:.1f} million discs laid: tracing the light")
    t0 = time.time(); rgb = Cc.astype(np.uint8); out = f"{W}/world.ply"
    blockers = [x for x in Lv["lm"] if x["blocker"]]; remap = np.full(len(Lv["lm"]), -1, np.int64); remap[[i for i, x in enumerate(Lv["lm"]) if x["blocker"]]] = np.arange(len(blockers))
    lobj = [dict(o, mesh=int(remap[o["mesh"]]) if o["mesh"] is not None else -1) for o in Lv["lobj"]]; ops, fits = [], []
    for e in Lv["S"]["emitters"]:
        r = np.asarray(e["rect"], float); d = {"kind": e["kind"], "c": r.mean(0), "u": (r[1] - r[0]) / 2, "v": (r[0] - r[3]) / 2, "n": np.asarray(e["normal"], float)}; (ops if e["kind"] in ("window", "skylight") else fits).append(d)
    discs = {"xyz": P, "normal": Nn, "rgb": rgb, "mesh": remap[Mi], "emit": Em, "sigma": Sp, "thick": Th, "opacity": Op, "part": Pi, "parts": [p["name"] for p in Lv["pieces"]]}
    # material classes: the light's own rough guess (words, part names, a vehicle's colours), then what the other stages KNOW laid over it:
    # the materials' class of each plain-form part, the paint's class of each disc of a sculpted thing (a hint: it only ever adds shine)
    cls, gls = light.classes(discs, blockers, lobj, np.clip(light.s2l(rgb), 0.004, 0.9), 0.0); veh = np.zeros(n, bool); lampy = np.zeros(n, bool)
    for o in lobj:
        if o["mesh"] < 0: continue
        w = light.words_of(o, True).split()
        if any(v in w or (v + "s") in w for v in light.VEHICLE): veh |= discs["mesh"] == o["mesh"]
        if any(q in " ".join(w) for q in ("lantern", "lamp", "light", "sconce", "chandelier", "bulb", "torch")): lampy |= discs["mesh"] == o["mesh"]
    for k, p in enumerate(Lv["pieces"]):
        if p.get("mcls"): sel = (Pi == k) & ~Em; cls[sel] = p["mcls"][0]; gls[sel] = np.maximum(gls[sel], p["mcls"][1])
    has = Mt != 255; c01 = rgb.astype(np.float32) / 255
    for o in lobj:          # a painted vehicle: the light tells body, chrome and glass apart by colour alone, which turns a cream or a black car into chrome and glass.
        if o["mesh"] < 0: continue          # Here: everything is gloss paint; glass where the paint saw glass; chrome only where it saw metal AND the colour is a bright grey unlike the body; dark low parts are rubber
        w = light.words_of(o, True).split(); sel = np.nonzero((discs["mesh"] == o["mesh"]) & has)[0]
        if not len(sel) or not any(v in w or (v + "s") in w for v in light.VEHICLE): continue
        a = c01[sel]; q = (np.clip(a, 0, 0.999) * 10).astype(int); key = (q[:, 0] * 10 + q[:, 1]) * 10 + q[:, 2]; dom = np.median(a[key == np.bincount(key, minlength=1000).argmax()], 0); d = np.linalg.norm(a - dom, axis=1)
        mx = a.max(1); sat = (mx - a.min(1)) / np.maximum(mx, 1e-3); h = (P[sel, 1] - P[sel, 1].min()) / max(float(np.ptp(P[sel, 1])), 1e-3); m = Mt[sel]
        c = np.full(len(sel), light.GLOSS, np.uint8); g = np.full(len(sel), VEH_GLOSS[1] if ("matte" in w or "matt" in w) else VEH_GLOSS[0], np.float32)          # low: on a sculpted body one baked reflection ray a disc shows the windows as blue smears, not as a reflection
        pale = colour_word(o.get("look") or "")[0] in ("white", "cream", "silver", "grey", "gray", "ivory", "beige")          # on a pale car chrome cannot be told from the body by colour: none is claimed
        chrome = (m == 2) & (sat < 0.15) & (mx > 0.45) & (d > 0.2) & (not pale); c[chrome] = light.METAL; g[chrome] = 0.9
        rub = (mx < 0.17) & (h < 0.45); c[rub] = light.MATT; g[rub] = 0.0
        cls[sel] = c; gls[sel] = g
    glass = has & (Mt == 1); cls[glass] = light.GLASS; gls[glass] = np.maximum(gls[glass], 0.92)
    metal = has & (Mt == 2) & ~veh & (cls == light.MATT); cls[metal] = light.METAL; gls[metal] = 0.8          # on a vehicle the paint's "metal" spills over the body: there the light's own chrome-by-colour rule stands
    glossy = has & (Mt == 4) & (cls == light.MATT); cls[glossy] = light.GLOSS; gls[glossy] = 0.45
    glow = has & (Mt == 3) & lampy; cls[glow] = light.LAMPGLASS
    discs["cls"] = cls; discs["gloss"] = gls
    res = light.light(discs, blockers, lobj, ops, fits, b["request"], b.get("mood"), {}); info["light"] = res["info"]
    tick("light_s", t0); t0 = time.time(); wi = light.write(out, discs, res, placing); info.update(write=wi); tick("write_s", t0)
    # ---- collider + scene entry
    t0 = time.time(); parts = []
    for x in Lv["lm"]:
        if not x["blocker"]: continue
        mm = trimesh.Trimesh(x["V"], x["F"], process=False)
        if not x["name"].startswith("shell:") and len(x["F"]) > 400:
            try: mm = mm.convex_hull
            except Exception: pass
        parts.append(mm)
    col = trimesh.util.concatenate(parts); col.vertices = (np.asarray(col.vertices) * [1, 1, -1] - m_) * k_; col.faces = col.faces[:, ::-1]; col.export(f"{W}/collider.glb")
    # what the viewer needs to show it (pipeline.py adds the name and the files' addresses). fromRoom: room metres -> the file's units, row by row
    scene = {"ply": "world.ply", "mesh": "collider.glb", "collider": "collider.glb", "up": [0.0, 1.0, 0.0], "center": [0.0, 0.0, 0.0],
             "eye": ((eye0 * [1, 1, -1] - m_) * k_).tolist(), "forward": (fwd0 * [1, 1, -1]).tolist(), "metre": k_, "room": room.tolist(), "flat": True, "lamp": 5,
             "centre": [0.0, 0.0], "floor": float(-m_[1] * k_), "focal": 803.679015484155, "width": 1024,
             "fromRoom": [k_, 0, 0, -m_[0] * k_, 0, k_, 0, -m_[1] * k_, 0, 0, -k_, -m_[2] * k_, 0, 0, 0, 1], "discs": int(wi.get("discs", n))}
    if os.path.exists(out[:-4] + "_sh.ply"): scene["ply2"] = "world_sh.ply"
    scene.update({k: b[k] for k, kind in (("mood", dict), ("effects", list)) if isinstance(b.get(k), kind) and b[k]})
    # what sways (a curtain, a plant) is bent inside the splat by the viewer, about the foot of its real mesh's box. Things that swing, spin or travel
    # would need a splat of their own and a world lit without them: they stay still here
    sway = []
    for mo in b.get("motions") or []:
        i = mo.get("i") if isinstance(mo, dict) else None
        if type(i) is not int or not 0 <= i < len(b["objects"]) or mo.get("type") != "sway" or i in Lv["S"]["claimed"]: continue
        try: V = C.placed(world, i)["V"]
        except OSError: continue
        lo, hi = V.min(0), V.max(0); sway.append({"i": i, "name": b["objects"][i]["name"], "type": "sway", "render": "skin", "box": {"c": ((lo + hi) / 2).tolist(), "h": ((hi - lo) / 2).tolist(), "yaw": 0.0}, **{k: v for k, v in mo.items() if k not in ("i", "type")}})
    if sway: scene["movers"] = sway
    json.dump(scene, open(f"{W}/scene.json.tmp", "w"), indent=1); os.replace(f"{W}/scene.json.tmp", f"{W}/scene.json")
    tick("scene_s", t0); info["no_paint"] = sorted(set(NOPAINT)); info["seconds"] = dict(T, total_s=round(time.time() - t00, 1))
    import resource; info["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)
    json.dump(info, open(f"{W}/build.json", "w"), indent=1)
    say(f"{info['discs'] / 1e6:.1f} million discs, {info['seconds']['total_s']:.0f} s, {info['peak_rss_gb']} GB"); return info

if __name__ == "__main__":
    build(os.path.abspath(os.environ["WORLD_DIR"]))
