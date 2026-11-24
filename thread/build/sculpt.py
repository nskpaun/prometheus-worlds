"""Every object of a layout gets a real mesh at its pose, within a time budget.
  sculpt.py plan LAYOUT.json [SECONDS]   which kinds are sculpted, at which quality, within SECONDS (default 330) -> boxes.json, shapes/plan.json
  sculpt.py pictures                     one picture per sculpted kind (Z-Image Turbo; ../../spike/.venv)
  sculpt.py meshes                       cut-outs and Hunyuan3D, one batch
  sculpt.py place                        plain forms, placing, re-seating, cords (CPU, a few seconds)
  sculpt.py verify                       prints what does not hold on the placed meshes (a report; changes nothing)
Environment: WORLD_DIR (the world's folder), HY3D_DIR as shapes.py. The folder ends up as shapes.py leaves one: boxes.json,
shapes/plan.json (kinds -> look, objects), shapes/KIND.png, KIND.rgba.png, KIND.glb, shapes/placed/NNN.npz (V, F in room
metres; plain forms also "part", "names"). plan.json adds: per kind "quality" ("full" | "mid" | "fast"), and "plain":
{kind: {"form", "objects"}} for what forms.py makes. Every step skips what exists.
Re-seating: once a thing's real mesh stands in the room, what rests "on" it is moved down (or up) onto the mesh's own
surface, on a level patch large enough for it, so support holds on the sculpted shape, not only on its box.
A kind whose picture, cut-out or shape fails (shapes/KIND.failed says why) is left out of the world with what stands on
it, and boxes.json "dropped" says so."""
import sys, os, json, math, time, subprocess, re
import numpy as np
D = os.path.abspath(os.environ["WORLD_DIR"]); os.environ["WORLD_DIR"] = D        # shapes.py reads it when imported
import where
import shapes as SH
S = f"{D}/shapes"
# seconds per kind, picture + mesh, measured on an M4 Pro
NO_SHAPE = "it is left out of the world (a light gets a plain stand-in)"
QUALITY = {"full": {"px": 512, "pic_steps": 4, "steps": 10, "octree": 256, "cost": 38.0},      # what the sentence names, when it is large
           "mid": {"px": 384, "pic_steps": 4, "steps": 8, "octree": 192, "cost": 30.0},        # named but small, or added and large
           "fast": {"px": 384, "pic_steps": 4, "steps": 8, "octree": 128, "cost": 22.0}}       # added small props
LOAD_S = 12.0        # loading the picture model once

def box_of(o):
    _, lo, hi = SH.frame(o); return lo, hi

def plan(layout_path, seconds=330.0):
    w = json.load(open(layout_path)); objs = w["objects"]; os.makedirs(f"{S}/placed", exist_ok=True)
    kinds = {}
    for i, o in enumerate(objs): kinds.setdefault(o["kind"], []).append(i)
    sc = [k for k, ix in kinds.items() if objs[ix[0]]["form"] == "sculpted"]
    size = lambda k: max(objs[kinds[k][0]]["size"])
    # what the sentence names first, then what is seen most: size times how many
    rank = sorted(sc, key=lambda k: (not any(objs[i]["said"] for i in kinds[k]), -size(k) * math.sqrt(len(kinds[k]))))
    spent, chosen, cut = LOAD_S, {}, []
    for k in rank:
        said = any(objs[i]["said"] for i in kinds[k]); big = size(k) >= 0.7; q = "full" if said and big else "mid" if said or big else "fast"
        if spent + QUALITY[q]["cost"] > seconds and not said: q = "fast"
        if spent + QUALITY[q]["cost"] > seconds and not said: cut.append(k); continue
        spent += QUALITY[q]["cost"]; chosen[k] = q
    # what there is no time to sculpt is left out, with what stands on it
    gone = {i for k in cut for i in kinds[k]}
    for _ in range(5): gone |= {i for i, o in enumerate(objs) if o.get("on") in gone}
    keep = [i for i in range(len(objs)) if i not in gone]; new = {old: n for n, old in enumerate(keep)}
    for i in gone: w.setdefault("dropped", []).append({"name": objs[i]["name"], "why": "no time left to sculpt this kind", "said": objs[i]["said"]})
    w["objects"] = [dict(objs[i], **({"on": new[objs[i]["on"]]} if "on" in objs[i] else {})) for i in keep]
    if w.get("motions"): w["motions"] = [dict(m, i=new[m["i"]]) for m in w["motions"] if m["i"] in new]
    objs = w["objects"]; kinds = {}
    for i, o in enumerate(objs): kinds.setdefault(o["kind"], []).append(i)
    def look1(o):
        ws = o["name"].split(); art = "an" if o["name"][:1] in "aeiou" else "a"; return re.sub(r"\s+", " ", f"{art} {o['name']}, {o['look']}" if o["look"] else f"{art} {o['name']}").strip(" ,.")[:120]
    def look(k):
        o = objs[kinds[k][0]]; lo, hi = box_of(o); p = SH.proportions(hi - lo)
        ws = o["name"].split(); last = ws[-1]; last = last[:-3] + "f" if last.endswith("ves") else SH.re.sub(r"(?<=[^s])s$", "", last) if len(last) > 3 else last      # one of them: "bookshelves" -> "bookshelf"
        name = " ".join(ws[:-1] + [last]); art = "an" if name[:1] in "aeiou" else "a"
        return re.sub(r"\s+", " ", f"{art} {name}, {o['look']}" if o["look"] else f"{art} {name}").strip(" ,.")[:120]
    P = {"style": w.get("request", "")[:80], "layout": None, "seconds": seconds, "planned_s": round(spent, 1),
         "kinds": {k: {"look": look(k), "objects": kinds[k], "quality": q} for k, q in chosen.items()},
         "plain": {k: {"form": objs[ix[0]]["form"], "objects": ix} for k, ix in kinds.items() if objs[ix[0]]["form"] != "sculpted"}, "cut": cut}
    P["objects"] = {str(i): {"kind": k, "look": look1(objs[i])} for k, v in P["kinds"].items() for i in v["objects"]}
    import layout; w["counts"] = layout.summary({**w, "unsolved": w.get("unsolved", [])})
    SH.save_json(w, f"{D}/boxes.json"); SH.save_json(P, f"{S}/plan.json")
    n = lambda q: sum(v == q for v in chosen.values())
    print(f"{len(objs)} objects, {len(kinds)} kinds: {n('full')} sculpted in full, {n('mid')} mid, {n('fast')} fast, {len(P['plain'])} plain; {len(cut)} kinds ({len(gone)} objects) cut for time; planned {spent:.0f} of {seconds:.0f} s")

def outline_want(w, h, d, flat=False):
    """How tall against how wide a thing of this box looks in its picture: from the front for a thing on a wall, else from
    the front three-quarter view a little from above (its top seen as a band of about a third of its depth)."""
    return h / max(w, 1e-3) if flat else (h + 0.3 * d) / max(0.75 * (w + d), 1e-3)

def outline_of(px):
    """The object's outline in a picture on a plain backdrop: its height over its width. None when there is no object."""
    res, _ = SH.backdrop(px); ys, xs = np.nonzero(res > 0.16)
    if len(ys) < 50: return None
    return float((np.percentile(ys, 99.5) - np.percentile(ys, 0.5) + 1) / (np.percentile(xs, 99.5) - np.percentile(xs, 0.5) + 1))

def pictures():
    os.environ["HF_HUB_OFFLINE"] = "1"                  # the weights are on disk already (setup.sh); never call out
    import mlx.core as mx
    mx.set_cache_limit(2 * 2**30)
    w = json.load(open(f"{D}/boxes.json")); objs = w["objects"]; P = json.load(open(f"{S}/plan.json"))
    todo = [k for k in P["kinds"] if not os.path.exists(f"{S}/{k}.png") and not os.path.exists(f"{S}/{k}.failed")]
    if not todo: return
    from mflux.models.z_image import ZImageTurbo
    t = time.time()
    try: model = ZImageTurbo(quantize=8)
    except Exception as e: sys.exit(f"The object-picture model (Z-Image Turbo) could not be loaded ({type(e).__name__}: {str(e)[:200]}). If it is not downloaded, open the app again to finish setup, or run ./setup.sh.")
    print("picture model loaded", round(time.time() - t, 1), "s", flush=True); SH.note_time("_model", "picture_load", time.time() - t)
    for k in todo:
        v = P["kinds"][k]; q = QUALITY[v["quality"]]; o = objs[v["objects"][0]]; lo, hi = box_of(o); prop = SH.proportions(hi - lo); t = time.time(); look = v["look"]
        ws_, hs_, ds_ = o["size"]
        if not prop and hs_ >= 1.6 * max(ws_, ds_): prop = f"tall and narrow, standing upright, about {hs_ / max(ws_, ds_):.1f} times as tall as it is wide"       # else a bookcase comes back as a low cabinet
        coloured = bool(SH.BLUISH.search(look) or SH.GREENISH.search(look)); best = None
        view = "seen from straight in front" if o["attach"] == "wall" else "seen from the front three-quarter view, slightly from above"
        want = outline_want(ws_, hs_, ds_, o["attach"] == "wall")
        whole = f"A piece of furniture {round(hs_ * 100)} cm tall: it stands on its own legs or base; show the whole {o['name']} from the side, legs and all"
        tries = ((7, "bright blue", False), (7, "bright green", False)) if coloured else ((7, "bright blue", False), (8, "bright blue", False))
        tries += ((9, "bright green" if coloured else "bright blue", True), (10, "bright green" if coloured else "bright blue", True))
        for seed, colour, insist in tries:
            if insist and best and best[0][0]: break                     # the plain wording gave a whole object of the right build: done
            prompt = (f"Product photo of {look}{', ' + prop if prop else ''}{'. ' + whole if insist else ''}. One single whole object, centred, entirely in frame with a margin around it, {'seen from eye level at a slight angle' if insist else view}, "
                      f"isolated on a plain flat solid {colour} background, no shadow, no floor, even soft lighting.")
            img = model.generate_image(seed=seed, prompt=prompt, num_inference_steps=q["pic_steps"], width=q["px"], height=q["px"])
            px_ = np.asarray(img.image.convert("RGB"), np.float32) / 255; fg, rim = SH.framing(px_); got = outline_of(px_)
            built = got is not None and got >= 0.75 * want                  # a table pictured as only its top is a flat disc: far flatter than its box
            if got is not None and not built: print(k, f"pictured {got:.2f} as tall as wide where its box looks {want:.2f}: too flat, asked again", flush=True)
            score = (built and rim < 0.03, built, fg if coloured else -rim)
            if 0.02 < fg < 0.8 and rim <= 0.10 and (best is None or score > best[0]): best = (score, img)
            if best and best[0][0] and not coloured: break
        if not best: SH.give_up(k, "could not be pictured whole on a plain backdrop", NO_SHAPE); continue
        best[1].image.save(f"{S}/{k}.tmp.png"); os.replace(f"{S}/{k}.tmp.png", f"{S}/{k}.png"); SH.note_time(k, "picture", time.time() - t); print(k, v["quality"], "picture", round(time.time() - t, 1), "s", flush=True)

def height_map(V, F, lo, hi, cell=0.015):
    """Heights of a mesh seen from above over the floor rectangle lo..hi -> (xs, zs, heights with -inf where nothing is)."""
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.core.Tensor(np.ascontiguousarray(V, np.float32)), o3d.core.Tensor(np.ascontiguousarray(F, np.uint32)))
    xs, zs = np.arange(lo[0] + cell / 2, hi[0], cell), np.arange(lo[1] + cell / 2, hi[1], cell); gx, gz = np.meshgrid(xs, zs, indexing="ij"); top = float(V[:, 1].max()) + 1.0
    rays = np.stack([gx, np.full_like(gx, top), gz, np.zeros_like(gx), -np.ones_like(gx), np.zeros_like(gx)], -1).astype(np.float32)
    t = sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return xs, zs, np.where(np.isfinite(t), top - t, -np.inf)

def reseat(objs, meshes, i, cell=0.015):
    """Move object i (its mesh and its box) onto the real surface of the thing it rests on. -> a line saying what was done."""
    from scipy import ndimage
    o = objs[i]; p = o["on"]; Vp, Fp = meshes[p][:2]; V = meshes[i][0]
    lo, hi = V.min(0), V.max(0); ex, ez = (hi[0] - lo[0]) / 2, (hi[2] - lo[2]) / 2; cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    plo, phi = Vp.min(0), Vp.max(0); xs, zs, Hm = height_map(Vp, Fp, (plo[0], plo[2]), (phi[0], phi[2]), cell)
    kx, kz = max(1, int(round(2 * ex / cell))), max(1, int(round(2 * ez / cell)))
    big = np.where(np.isfinite(Hm), Hm, -1e9); mx = ndimage.maximum_filter(big, (kx, kz), mode="constant", cval=-1e9); mn = ndimage.minimum_filter(big, (kx, kz), mode="constant", cval=-1e9)
    # a level patch: the whole footprint lies on the mesh and its heights differ by under 1.5 cm
    level = (mx - mn < 0.015) & (mn > -1e8); occ = np.zeros_like(level)
    for j, q in enumerate(objs):       # the patches other things on this parent already have
        if j != i and q.get("on") == p and q.get("seated"):
            a, b = meshes[j][0].min(0), meshes[j][0].max(0)
            occ[np.ix_((xs > a[0] - ex - 0.015) & (xs < b[0] + ex + 0.015), (zs > a[2] - ez - 0.015) & (zs < b[2] + ez + 0.015))] = True
    gx, gz = np.meshgrid(xs, zs, indexing="ij"); cand = level & ~occ; how = "level patch"
    if cand.any():
        topmost = mx[cand].max(); cand &= mx > topmost - 0.03                    # the highest level patch that holds it: the desk top, the mantel shelf
        k = np.argmin(np.where(cand, np.hypot(gx - cx, gz - cz), np.inf)); nx, nz, ny = gx.flat[k], gz.flat[k], mx.flat[k]
    else:       # nowhere level (a lumpy top): it stays where it is, resting on the highest point under it
        m = (np.abs(gx - cx) < ex) & (np.abs(gz - cz) < ez) & np.isfinite(Hm); how = "highest point under it"
        if not m.any(): return f'{o["name"]}: nothing of "{objs[p]["name"]}" is under it; left on its box top'
        nx, nz, ny = cx, cz, float(np.percentile(Hm[m], 97))
    d = np.array([nx - cx, ny - lo[1], nz - cz]); meshes[i] = (V + d, *meshes[i][1:]); o["seated"] = True
    for q in o["prims"]: q["c"] = [float(a + b) for a, b in zip(q["c"], d)]
    return f'{o["name"]} on {objs[p]["name"]}: moved {math.hypot(d[0], d[2]):.2f} m sideways, {d[1]:+.3f} m in height ({how})'

def decks(objs, meshes, room, forms):
    """What stands on a plain platform must have its whole foot on the deck (a car's wheels): the platform is regrown under the real mesh."""
    log = []
    for i, o in enumerate(objs):
        p = o.get("on")
        if p is None or i not in meshes or p not in meshes or objs[p]["form"] not in ("crate", "cylinder") or objs[p]["attach"] != "floor": continue
        if max(objs[p]["size"][0], objs[p]["size"][2]) < 1.6 or o["size"][0] * o["size"][2] < 0.5: continue      # platforms and the large things they show, not a jar on a crate
        V = meshes[i][0]; lo, hi = V.min(0), V.max(0); foot = V[V[:, 1] < lo[1] + 0.12 * (hi[1] - lo[1])]; f0, f1 = foot.min(0), foot.max(0)
        P_ = meshes[p][0]; p0, p1 = P_.min(0), P_.max(0); m = 0.12
        if f0[0] >= p0[0] + m and f1[0] <= p1[0] - m and f0[2] >= p0[2] + m and f1[2] <= p1[2] - m: continue
        n0 = np.array([min(p0[0], f0[0] - 0.25), min(p0[2], f0[2] - 0.25)]); n1 = np.array([max(p1[0], f1[0] + 0.25), max(p1[2], f1[2] + 0.25)])
        n0 = np.maximum(n0, 0.05); n1 = np.minimum(n1, [room[0] - 0.05, room[2] - 0.05]); q = objs[p]; hs = q["size"][1]; c = [(n0[0] + n1[0]) / 2, q["prims"][0]["c"][1], (n0[1] + n1[1]) / 2]
        yawq = abs(math.cos(q["yaw"])) > 0.5; ws, ds = (n1[0] - n0[0], n1[1] - n0[1]) if yawq else (n1[1] - n0[1], n1[0] - n0[0])
        if q["form"] == "cylinder": ws = ds = max(ws, ds) * 1.2
        q["size"] = [round(float(ws), 3), hs, round(float(ds), 3)]; q["prims"][0]["c"] = [float(v) for v in c]
        if q["prims"][0]["shape"] == "box": q["prims"][0]["h"] = [ws / 2, hs / 2, ds / 2]
        else: q["prims"][0]["r"] = ws / 2
        Vn, Fn, part, names = forms.make(q["form"], ws, hs, ds, seed=f"{q['name']}/{p}", name=q["name"]); meshes[p] = (forms.to_room(Vn, c[0], c[1] - hs / 2, c[2], q["yaw"]).astype(np.float32), Fn, part, names)
        log.append(f'{q["name"]}: regrown to {ws:.2f} x {ds:.2f} m so that the whole foot of {o["name"]} stands on its deck')
    return log

def untangle(objs, meshes, room):
    """Collisions on the REAL meshes (boxes cannot see a handlebar inside a car door). Two things collide when surface points
    of one lie inside the other, 2 cm deep or more. The lesser one (added before named, small before large) is moved to the
    nearest spot where its mesh is free of every other mesh; if there is none within 1.5 m (0.9 m along a wall, 0.3 m on a top) it is left out.
    -> (indices to leave out, log)"""
    import open3d as o3d
    T = o3d.core.Tensor; ids = sorted(meshes); sc, pts, bb = {}, {}, {}
    rng = np.random.default_rng(0)
    def prep(i):
        V, F = np.ascontiguousarray(meshes[i][0], np.float32), np.ascontiguousarray(meshes[i][1], np.uint32)
        s_ = o3d.t.geometry.RaycastingScene(); s_.add_triangles(T(V), T(F)); sc[i] = s_
        c = V[F].mean(1); P = np.concatenate([V, c]); pts[i] = P[rng.choice(len(P), min(len(P), 6000), replace=False)]; bb[i] = [V.min(0), V.max(0)]
    for i in ids: prep(i)
    off = {i: np.zeros(3, np.float32) for i in ids}
    kids = {i: [j for j in ids if objs[j].get("on") == i or objs[j].get("part_of") == i] for i in ids}
    def family(i):
        out = [i]
        for j in kids[i]: out += family(j)
        return out
    def related(i, j): return objs[i].get("on") == j or objs[j].get("on") == i or objs[i].get("part_of") == j or objs[j].get("part_of") == i
    def deep(i, j, di=0, dj=0):
        """How many of i's sampled surface points lie 2 cm or more inside j (i shifted by di, j by dj)."""
        P = pts[i] + off[i] + di - (off[j] + dj); lo, hi = bb[j]
        m = np.all((P > lo + 0.02) & (P < hi - 0.02), 1)
        if not m.any(): return 0
        d = sc[j].compute_signed_distance(T(np.ascontiguousarray(P[m], np.float32)), nsamples=3).numpy(); return int((d < -0.02).sum())
    def hit(i, j, di=0, dj=0):
        a0, a1 = bb[i][0] + off[i] + di, bb[i][1] + off[i] + di; b0, b1 = bb[j][0] + off[j] + dj, bb[j][1] + off[j] + dj
        if not np.all((np.minimum(a1, b1) - np.maximum(a0, b0)) > 0.02): return 0
        return deep(i, j, di, dj) + deep(j, i, dj, di)
    rug = lambda i: objs[i]["form"] == "rug"
    rank = lambda i: (objs[i]["said"], not objs[i].get("fill"), float(np.prod(bb[i][1] - bb[i][0])))       # the lesser of two moves
    W, H, D = room; log, gone = [], set()
    pairs = [(i, j) for a, i in enumerate(ids) for j in ids[a + 1:] if not rug(i) and not rug(j) and not related(i, j)]
    for i, j in pairs:
        if i in gone or j in gone: continue
        n = hit(i, j)
        if n < 6: continue
        # an "on" thing is moved with what it stands on only if that is the mover; a thing on a table that collides moves alone on its top
        m, k = (i, j) if rank(i) < rank(j) else (j, i); fam = family(m); o = objs[m]; best = None
        others = [x for x in ids if x not in fam and x not in gone and not rug(x)]
        def free(dv):
            for f in fam:
                a0, a1 = bb[f][0] + off[f] + dv, bb[f][1] + off[f] + dv
                if a0[0] < -0.01 or a0[2] < -0.01 or a1[0] > W + 0.01 or a1[2] > D + 0.01: return False
                for x in others:
                    if related(f, x): continue
                    if hit(f, x, di=dv) >= 6: return False
            return True
        if o["attach"] == "floor": dirs = [(math.cos(a), math.sin(a)) for a in np.arange(16) * math.pi / 8]
        elif o["attach"] in ("wall",): dirs = [(1, 0), (-1, 0)] if o.get("wall") in ("back", "front") else [(0, 1), (0, -1)]
        elif o["attach"] == "on": dirs = [(math.cos(a), math.sin(a)) for a in np.arange(8) * math.pi / 4]
        else: dirs = [(math.cos(a), math.sin(a)) for a in np.arange(8) * math.pi / 4]
        reach = 1.5 if o["attach"] == "floor" else 0.9 if o["attach"] != "on" else 0.3
        for r in np.arange(0.05, reach + 1e-6, 0.05):
            for dx, dz in dirs:
                dv = np.array([dx * r, 0, dz * r], np.float32)
                if o["attach"] == "on":      # it must stay on the top it stands on
                    p = o["on"]; a0, a1 = bb[m][0] + off[m] + dv, bb[m][1] + off[m] + dv; p0, p1 = bb[p][0] + off[p], bb[p][1] + off[p]
                    if a0[0] < p0[0] or a1[0] > p1[0] or a0[2] < p0[2] or a1[2] > p1[2]: continue
                if free(dv): best = dv; break
            if best is not None: break
        if best is not None:
            for f in fam: off[f] = off[f] + best
            log.append(f'{o["name"]} stood {n} sample points deep inside {objs[k]["name"]} (real meshes): moved {float(np.hypot(best[0], best[2])):.2f} m clear')
        elif not o["said"]:
            gone.update(fam); log.append(f'{o["name"]} stood inside {objs[k]["name"]} (real meshes) and no free place was within {reach} m: left out' + (f' with {len(fam) - 1} things on it' if len(fam) > 1 else ""))
        else: log.append(f'UNRESOLVED: {o["name"]} and {objs[k]["name"]} (both named) intersect as meshes')
    for i in ids:
        if i in gone or not off[i].any(): continue
        meshes[i] = (meshes[i][0] + off[i], *meshes[i][1:])
        for q in objs[i]["prims"]: q["c"] = [float(a + b) for a, b in zip(q["c"], off[i])]
    return gone, log

def meshes():
    """Cut-outs and Hunyuan3D for every pictured kind, one after another."""
    P = json.load(open(f"{S}/plan.json")); hy = f"{SH.HY3D}/.build/release/hy3d"
    for k, v in P["kinds"].items():
        if os.path.exists(f"{S}/{k}.glb") or not os.path.exists(f"{S}/{k}.png") or os.path.exists(f"{S}/{k}.failed"): continue
        q = QUALITY[v["quality"]]; t = time.time()
        if not SH.cut_out(f"{S}/{k}.png", f"{S}/{k}.rgba.png"): SH.give_up(k, "could not be cut out of its picture", NO_SHAPE); continue
        from PIL import Image
        al = np.asarray(Image.open(f"{S}/{k}.rgba.png").convert("RGBA"))[..., 3] > 128; ys, xs = np.nonzero(al)
        solid = al.sum() / max(1, (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) if len(ys) else 1.0
        if solid < 0.42 and q["octree"] < 256: q = dict(q, octree=256, steps=max(q["steps"], 8)); print(k, f"is thin or leggy (fills {solid:.0%} of its outline box): fine grid", flush=True)
        r = subprocess.run(["/usr/bin/time", "-p", hy, "shape", f"{S}/{k}.rgba.png", "-o", f"{S}/{k}.tmp.glb", "--weights", f"{SH.HY3D}/weights/shape-small", "--steps", str(q["steps"]), "--octree", str(q["octree"]), "--seed", "0"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
        if r.returncode or not os.path.exists(f"{S}/{k}.tmp.glb"): SH.give_up(k, "could not be turned into a shape", NO_SHAPE); continue
        os.replace(f"{S}/{k}.tmp.glb", f"{S}/{k}.glb"); m = re.search(r"real\s+([\d.]+)", r.stderr or ""); SH.note_time(k, "mesh", float(m.group(1)) if m else time.time() - t)
        print(k, v["quality"], "mesh", m.group(1) if m else "?", "s", flush=True)

def build():
    import trimesh, forms
    w = json.load(open(f"{D}/boxes.json")); objs, room = w["objects"], w["room"]; P = json.load(open(f"{S}/plan.json")); os.makedirs(f"{S}/placed", exist_ok=True); log = []
    n_real = next((i for i, o in enumerate(objs) if o.get("part_of") is not None), len(objs))      # built before: the cords made then are made again
    del objs[n_real:]; P["plain"].pop("cord", None)
    if w.get("motions"): w["motions"] = [m for m in w["motions"] if m["i"] < n_real]
    t0 = time.time(); meshes = {}; aim = None
    for k, v in P["kinds"].items():          # sculpted kinds: the repo's own placing (facing, mirror, one scale), then held inside the box
        if not os.path.exists(f"{S}/{k}.glb"): continue
        try: V, F = SH.load(k)
        except Exception as e: print(k, "has an unusable shape:", e); continue
        for i in v["objects"]:
            Wv, G = SH.place(V, F, objs, i, room, aim, v["objects"]); o = objs[i]; lo, hi = box_of(o); yaw = SH.frame(o)[0]
            # the box in room axes (quarter turns only)
            c = np.array(o["prims"][0]["c"]); hw = np.array(o["size"]) / 2; e = np.array([hw[0], hw[1], hw[2]] if abs(math.cos(yaw)) > 0.5 else [hw[2], hw[1], hw[0]])
            a, b = Wv.min(0), Wv.max(0); ext = np.maximum(b - a, 1e-6); k, km = int(np.argmax(e)), int(np.argmax(ext))
            if k != km and np.sort(ext)[-1] > 2.5 * np.sort(ext)[-2] and np.sort(e)[-1] > 2.5 * np.sort(e)[-2]:
                # a long thing whose shape came out standing while its box lies (a quill, a wrench, a rolled carpet), or the other way: turned a quarter turn
                mid = (a + b) / 2; Q = Wv - mid; Q[:, [k, km]] = np.stack([Q[:, km], -Q[:, k]], 1); Wv = Q + mid; a, b = Wv.min(0), Wv.max(0); ext = np.maximum(b - a, 1e-6)
            # its longest side is the box's; the other sides may stand out of the box a little (more for what stands free on the floor: a mop's head is wider than its handle)
            room_for = 1.12 * 2 * e
            if o["attach"] in ("floor", "ceiling"):
                for ax in (0, 2):
                    if 2 * e[ax] < 0.35: room_for[ax] = 2 * e[ax] + 0.2          # a thin thing standing free: its head or foot may be wider
            room_for[k] = 2 * e[k] * 1.02
            s = float(np.min(room_for / ext)) if np.min(room_for / ext) < 1 or ext[k] < 0.9 * 2 * e[k] else 1.0
            s = min(s, 2 * e[k] * 1.02 / ext[k])
            # the point that stays put: the foot for what stands, the top for what hangs, the wall face for what is fixed to a wall
            pin = np.array([(a[0] + b[0]) / 2, a[1], (a[2] + b[2]) / 2]); goal = np.array([c[0], c[1] - e[1], c[2]])
            if o["attach"] == "ceiling": pin[1] = b[1]; goal[1] = c[1] + e[1]
            if o["attach"] == "wall":
                n = {"back": (2, 1), "front": (2, -1), "left": (0, -1), "right": (0, 1)}[o["wall"]]; ax, sg = n
                pin = np.array([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]); goal = c.copy(); pin[ax] = b[ax] if sg > 0 else a[ax]; goal[ax] = c[ax] + sg * e[ax]
            # one scale, except where the shape fills under three quarters of its box along an axis (a bookcase that came out squat):
            # that axis alone is stretched, by a quarter at most, which furniture bears and which keeps sizes true to the layout
            fill = ext * s / (2 * e); s3 = s * np.where(fill < 0.75, np.minimum(1.25, 0.9 / np.maximum(fill, 1e-6)), 1.0)
            if o["attach"] == "floor" and fill[1] < 0.93: s3[1] = s * min(1.4, 1.0 / fill[1])          # a named height holds: "tall" bookshelves asked at 2.2 m are 2.2 m
            # the other sides too: a side that fills under nine tenths of its box is stretched (to 1.3 at most), one that stands out over 4% is drawn in.
            # The shape model draws a desk squarer than a desk is; a third of stretch on one axis is not seen on furniture, and never reaches round things, which fill their boxes
            now = ext * s3 / (2 * e); s3 = s3 * np.where(now < 0.9, np.minimum(1.3, 0.97 / np.maximum(now, 1e-6)), np.where(now > 1.04, 1.04 / now, 1.0))
            Wv = (Wv - pin) * s3 + goal
            for ax, L in ((0, room[0]), (2, room[2])): Wv[:, ax] += max(0.0, -Wv[:, ax].min()) - max(0.0, Wv[:, ax].max() - L)
            Wv[:, 1] -= max(0.0, Wv[:, 1].max() - room[1])
            meshes[i] = (Wv.astype(np.float32), G)
    if P["kinds"] and not meshes:
        raise SystemExit(f"None of the {len(P['kinds'])} kinds of object could be sculpted, so the shape model is not working. Run ./setup.sh again; shapes/*.failed in the world's folder say what went wrong with each.")
    # a light whose shape could not be made gets a plain stand-in of its size (a housing and a shining part), so the room keeps
    # its light; anything else without a shape is left out below, with what stands on it
    import light_base
    for k, v in list(P["kinds"].items()):
        if any(i in meshes for i in v["objects"]): continue
        words = " ".join(str(objs[v["objects"][0]].get(q) or "") for q in ("name", "kind", "look")).lower().replace("_", " ").replace("-", " ")
        if not light_base._has(words, light_base.LAMP) or light_base._has(words, light_base.NOT_LAMP + light_base.FIRE): continue
        del P["kinds"][k]; P["plain"][k] = {"form": "lamp", "objects": v["objects"], "stand_in": True}
        for i in v["objects"]: objs[i]["form"] = "lamp"
        print(f"{k}: its shape could not be made; a plain lamp of its size stands in for it", flush=True)
    for k, v in P["plain"].items():          # plain forms: made to the box, each copy with its own seed
        for i in v["objects"]:
            o = objs[i]; ws, hs, ds = o["size"]; c = o["prims"][0]["c"]
            V, F, part, names = forms.make(v["form"], ws, hs, ds, seed=f"{o['name']}/{i}", name=o["name"], **(o.get("opts") or {}))
            meshes[i] = (forms.to_room(V, c[0], c[1] - hs / 2, c[2], o["yaw"]).astype(np.float32), F, part, names)
    def leave_out(gone, why):
        """Objects that cannot be in the world are taken out, with every index that points past them moved up. -> objs, meshes"""
        for _ in range(5): gone |= {i for i, o in enumerate(objs) if o.get("on") in gone or o.get("part_of") in gone}
        keep = [i for i in range(len(objs)) if i not in gone]; new = {old: n for n, old in enumerate(keep)}
        for i in sorted(gone): w.setdefault("dropped", []).append({"name": objs[i]["name"], "why": why, "said": objs[i]["said"]})
        w["objects"] = [dict(objs[i], **({"on": new[objs[i]["on"]]} if "on" in objs[i] else {})) for i in keep]
        if w.get("motions"): w["motions"] = [dict(m, i=new[m["i"]]) for m in w["motions"] if m["i"] in new]
        for grp in ("kinds", "plain"):
            for k in list(P[grp]):
                P[grp][k]["objects"] = [new[i] for i in P[grp][k]["objects"] if i in new]
                if not P[grp][k]["objects"]: del P[grp][k]
        return w["objects"], {new[i]: m for i, m in meshes.items() if i in new}
    missing = [objs[i]["name"] for i in range(len(objs)) if i not in meshes]
    if missing: objs, meshes = leave_out({i for i in range(len(objs)) if i not in meshes}, "its shape could not be made")
    def depth(i):
        n = 0
        while "on" in objs[i] and n < 9: i = objs[i]["on"]; n += 1
        return n
    for i in sorted((i for i in meshes if "on" in objs[i] and objs[i]["on"] in meshes), key=lambda i: (depth(i), -objs[i]["size"][0] * objs[i]["size"][2])):
        p = objs[i]["on"]
        if objs[p].get("moved") is not None:        # what it rests on was itself moved: go with it first
            d = np.array(objs[p]["moved"]); meshes[i] = (meshes[i][0] + d, *meshes[i][1:])
            for q in objs[i]["prims"]: q["c"] = [float(a + b) for a, b in zip(q["c"], d)]
        before = meshes[i][0].mean(0); log.append(reseat(objs, meshes, i)); objs[i]["moved"] = (meshes[i][0].mean(0) - before + (np.array(objs[p]["moved"]) if objs[p].get("moved") is not None else 0)).tolist()
    for o in objs: o.pop("moved", None); o.pop("seated", None)
    log += decks(objs, meshes, room, forms)
    gone, ulog = untangle(objs, meshes, room); log += ulog
    if gone: objs, meshes = leave_out(set(gone), "its real shape stood inside another thing and no free place was near")
    # cords: what hangs below the ceiling gets a real cord up to it
    n0 = len(objs)
    for i in range(n0):
        o = objs[i]
        if o["attach"] != "ceiling" or i not in meshes or o.get("part_of") is not None: continue
        V = meshes[i][0]; top = float(V[:, 1].max()); gap = room[1] - top
        if gap < 0.03: continue
        hi_part = V[V[:, 1] > top - 0.02]; x, z = float(hi_part[:, 0].mean()), float(hi_part[:, 2].mean()); r = 0.012
        Vc, Fc, part, names = forms.make("cylinder", 2 * r, gap + 0.02, 2 * r, "cord"); j = len(objs)
        objs.append({"name": f"{o['name']} cord", "prims": [{"shape": "cyl", "c": [x, top - 0.02 + (gap + 0.02) / 2, z], "r": r, "hy": (gap + 0.02) / 2}], "yaw": 0.0, "kind": "cord", "form": "cylinder", "attach": "ceiling",
                     "said": False, "look": "dark iron chain or cord", "size": [2 * r, round(gap + 0.02, 3), 2 * r], "part_of": i})
        meshes[j] = (forms.to_room(Vc, x, top - 0.02, z, 0.0).astype(np.float32), Fc, part, ["cord"]); P["plain"].setdefault("cord", {"form": "cylinder", "objects": []})["objects"].append(j)
        for m in list(w.get("motions") or []):
            if m["i"] == i and m["type"] == "swing": w["motions"].append(dict(m, i=j))
    for f in os.listdir(f"{S}/placed"): os.remove(f"{S}/placed/{f}")
    for i, m in meshes.items():
        extra = {"part": m[2], "names": np.array(m[3])} if len(m) > 2 else {}
        np.savez(f"{S}/placed/{i:03d}.npz", V=m[0].astype(np.float32), F=np.asarray(m[1], np.int32), **extra)
    # what is true now: boxes follow the meshes that were re-seated; the plan carries the layout's stamp like the repo's
    import layout
    for i, o in enumerate(objs):      # the size each thing really has now, beside the size the layout asked for
        if i in meshes: o["got"] = [round(float(v), 3) for v in (meshes[i][0].max(0) - meshes[i][0].min(0))]
    w["reseated"] = log; w["counts"] = layout.summary({**w, "unsolved": w.get("unsolved", [])}); SH.save_json(w, f"{D}/boxes.json")
    old = P.get("objects", {}); P["layout"] = SH.stamp(w); P["objects"] = {str(i): {"kind": k, "look": (lambda o: re.sub(r"\s+", " ", f"a {o['name']}, {o['look']}").strip(" ,.")[:120])(objs[i])} for k, v in P["kinds"].items() for i in v["objects"] if i in meshes}
    SH.save_json(P, f"{S}/plan.json"); SH.note_time("_all", "place", time.time() - t0)
    faces = sum(len(m[1]) for m in meshes.values())
    print("counts:", json.dumps(w["counts"]))
    for i, o in enumerate(objs):
        if o["form"] == "sculpted" and "got" in o:
            yawq = abs(math.cos(o["yaw"])) > 0.5; g = o["got"] if yawq else [o["got"][2], o["got"][1], o["got"][0]]
            if o["said"] or abs(g[1] / o["size"][1] - 1) > 0.1: print(f'   size held? {o["name"]}: asked {o["size"]}, real mesh {g}')
    print(f"{len(meshes)} of {len(objs)} objects have a mesh ({faces // 1000}k faces); left out for want of a shape: {missing}; placing, plain forms and re-seating {time.time() - t0:.1f} s")
    for l in log: print("  ", l)

def verify():
    """Do the placed MESHES hold? each thing 'on' another touches it (gap under 1 cm) and no two meshes' boxes overlap by more than 3 cm."""
    w = json.load(open(f"{D}/boxes.json")); objs = w["objects"]; M = {}
    for i in range(len(objs)):
        f = f"{S}/placed/{i:03d}.npz"
        if os.path.exists(f): z = np.load(f); M[i] = (z["V"], z["F"])
    bad = []
    for i, o in enumerate(objs):
        if "on" in o and i in M and o["on"] in M:
            V = M[i][0]; lo, hi = V.min(0), V.max(0); xs, zs, Hm = height_map(*M[o["on"]], (lo[0], lo[2]), (hi[0], hi[2]), cell=max(0.002, min(0.015, (hi[0] - lo[0]) / 4, (hi[2] - lo[2]) / 4)))
            if not np.isfinite(Hm).any(): bad.append(f'{o["name"]} has nothing of {objs[o["on"]]["name"]} under it'); continue
            gap = lo[1] - Hm[np.isfinite(Hm)].max()
            if abs(gap) > 0.012: bad.append(f'{o["name"]} is {gap * 100:+.1f} cm off the surface of {objs[o["on"]]["name"]}')
    ids = sorted(M); rug = lambda i: objs[i]["form"] == "rug"; near = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if rug(i) or rug(j) or objs[i].get("on") == j or objs[j].get("on") == i or objs[i].get("part_of") == j or objs[j].get("part_of") == i: continue
            l1, h1, l2, h2 = M[i][0].min(0), M[i][0].max(0), M[j][0].min(0), M[j][0].max(0); ov = np.minimum(h1, h2) - np.maximum(l1, l2)
            if (ov > 0.03).all(): near.append((i, j))
    import open3d as o3d
    T = o3d.core.Tensor; rng = np.random.default_rng(1); sc = {}
    def inside(i, j):
        if j not in sc: sc[j] = o3d.t.geometry.RaycastingScene(); sc[j].add_triangles(T(np.ascontiguousarray(M[j][0], np.float32)), T(np.ascontiguousarray(M[j][1], np.uint32)))
        P = np.concatenate([M[i][0], M[i][0][M[i][1]].mean(1)]); P = P[rng.choice(len(P), min(len(P), 6000), replace=False)]
        m = np.all((P > M[j][0].min(0) + 0.02) & (P < M[j][0].max(0) - 0.02), 1)
        return int((sc[j].compute_signed_distance(T(np.ascontiguousarray(P[m], np.float32)), nsamples=3).numpy() < -0.02).sum()) if m.any() else 0
    for i, j in near:
        n = inside(i, j) + inside(j, i)
        if n >= 6: bad.append(f'{objs[i]["name"]} and {objs[j]["name"]} intersect as real meshes ({n} sample points 2 cm deep)')
    print(f"{len(M)} meshes checked ({len(near)} pairs with overlapping bounds tested mesh against mesh): {len(bad)} faults"); [print("  ", b) for b in bad]

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "plan": plan(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 330.0)
    elif c == "pictures": pictures()
    elif c == "meshes": meshes()
    elif c == "place": build()
    elif c == "verify": verify()
