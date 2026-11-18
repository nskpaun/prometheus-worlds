"""The materials of one world: every surface, every architecture group and every plain-form part. thread/.venv.
  materials.py [MAX_SWATCHES]     the picture job list (CPU, 1 s), the picture model on it (klein.py, one load), then faces laid,
                                  groups, parts, contents (CPU) -> WORLD_DIR/materials/{tex/*.png, surfaces.json, report.json}
Needs the world's layout and placed meshes (sculpt.py) and materials/style.json with the part materials (style.py). The room
is arch.shell_geometry on the true bounds of the meshes (level.shell). Then the assembler reads it all through mats.Mats.
What is made (all true scale, evenly lit, the material's own colour; light comes last, in light.py):
  SURFACES  x0 x1 z0 z1 floor ceil       laid for THIS room from photographed pieces (surfaces.py), 512 / 384 / 192 px a metre
  GROUPS    one 2 m x 0.75 m swatch per architecture material group at 512 px a metre (grain along the long side). A stone or plaster group in a room whose
            walls are that same kind of stuff IS the wall's own material (same tone), so quoins, surrounds and string courses stop reading as pale toy blocks.
            "wall" / "floor" / "ceiling" groups (reveals, a chimney breast, a pier, a threshold) are the surface's own material.
  PARTS     every part of every plain form gets its OWN material (a curtain's rod is metal, its cloth velvet): one 1 m swatch per distinct material.
  CONTENT   things that are flat by nature carry a picture made for them: a rug its woven pattern, a framed picture / map its image, a tapestry its
            weaving, a crate or sign its stencilled label, a row of books its spines; a mirror is class "mirror" (for the light), not a picture.
"""
import os, sys, json, time, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import where, level as LV, tools as C
import surfaces as S5
CLS = "wood, stone, metal, fabric, glass, ceramic, paper, leather, plastic, plaster, concrete, wax, rope, wicker, plant, other"
SW_PX, SW_M, MAX_SW = 512, 1.0, 14
GRP_M = (2.0, 0.75)
SKIP_GROUPS = ("glass", "emitter")
BARE = "Nothing but the bare surface: no objects, no furniture, no frame, no border, no people, no text, no cast shadows."

# ---------------------------------------------------------------- the level
def level(world):
    """-> dict(b layout, S shell, P plan, placed(i) -> {V, F, part, names})"""
    return dict(b=LV.layout(world), S=LV.shell(world), P=LV.plan(world), placed=lambda i: LV.placed(world, i))
def plain_items(Lv):
    """Every plain-form kind with its parts: [{kind, form, name, look, size, parts, objects}]"""
    b, S, P = Lv["b"], Lv["S"], Lv["P"]; out = []; claimed = set(S.get("claimed") or [])
    for kind, pl in (P.get("plain") or {}).items():
        objs = [i for i in pl["objects"] if i not in claimed]
        if not objs: continue
        i = objs[0]; o = b["objects"][i]
        try: names = [str(n) for n in Lv["placed"](i).get("names", ["body"])]
        except Exception: names = ["body"]
        if pl["form"] == "books": names = ["cover", "pages"]
        out.append({"kind": kind, "form": pl["form"], "name": o["name"], "look": o.get("look", ""), "size": o.get("size"), "parts": names, "objects": objs, "wall": o.get("wall")})
    return out
def sculpt_items(Lv): return [{"kind": k, "look": v.get("look", "")} for k, v in (Lv["P"].get("kinds") or {}).items()]

# ---------------------------------------------------------------- assign: the text model names every part's material and every flat thing's content
def assign_prompts(style, Lv, req=None):
    """-> [the parts question, the looks question] (the second only if there are sculpted kinds). style None: the guide is an earlier answer in the same model process ({{GUIDE}})."""
    it = plain_items(Lv); sc = sculpt_items(Lv); g = {k: style[k] for k in ("place", "character", "scheme", "time", "palette", "wear_words", "materials") if k in style} if style else None
    head = f'A 3D place is being built from this request: "{req or style["request"]}"\nIts STYLE GUIDE: {json.dumps(g) if g else "{{GUIDE}}"}\n'
    lines = "\n".join(f'{k}. {x["name"]} ({x["look"]}), about {" x ".join(f"{v:g}" for v in (x["size"] or []))} m; parts: {", ".join(x["parts"])}' for k, x in enumerate(it))
    p1 = (head + f'These simple things in it are real geometry made of named parts:\n{lines}\n'
            'For EACH numbered thing give every one of ITS OWN listed parts a material. A thing is what its name says it is (a wooden crate is wood, a copper pot is copper, wax is wax, a plant is green): choose from the style guide the species, finish, wear and '
            'the colour of whatever is dyed, painted, woven or upholstered, so the place holds together. '
            '"material" is a phrase a photographer would write for that bare surface seen close up, with colour, finish and wear (for example "dark waxed walnut", "black wrought iron", "midnight blue silk velvet", "tarnished gilt wood", "cream church-candle wax", "aged brass"); '
            f'"cls" is one of {CLS}; "rgb" its own colour in daylight [r, g, b] 0 to 255 (never pure, nothing darker than [40, 36, 34]). Parts made of the same stuff use exactly the same phrase. A flame is "flame".\n'
            'Then "content": what a flat thing shows, because flat things are never blank. A rug: {"type": "pattern", "words": its woven design, its colours from the palette, its border}. '
            'A framed thing: {"type": "picture", "words": exactly what the painting, map, chart or print shows and its colours} or {"type": "mirror"} if it is a mirror. '
            'A hanging tapestry, banner, flag or poster: {"type": "picture", "words": what is woven or printed on it}. A curtain or drape: {"type": "trim", "words": the colour of its braid}. '
            'A crate, box, barrel, sign or board that would carry lettering: {"type": "label", "words": a short stencilled label of at most 12 letters}. A row or stack of books: {"type": "books", "words": the colours of the bindings}. Anything else: {"type": "none"}.\n'
            'Answer with JSON only: {"things": [{"n": 0, "parts": {"PART": {"material": "...", "cls": "...", "rgb": [r, g, b]}}, "content": {"type": "...", "words": "..."}}, ...]} with one entry for every number, each with exactly its own listed parts.')
    if not sc: return [p1 + COMPACT]
    sl = "\n".join(f'{k}. {x["look"]}' for k, x in enumerate(sc))
    p2 = (head + f'These sculpted things in it will each be painted by an artist who sees only one line of words:\n{sl}\n'
            'Rewrite each line as ONE specific phrase in the style guide\'s materials and colours: say what the thing is and what its surfaces are (paint colour and finish, metal, glass, wood species, cloth, wear). '
            'A thing stays what it is and keeps its natural colours (leaves are green, a flame is fire, tyres are black rubber, chrome is chrome); several of the same thing may differ in colour. No actions, no surroundings.\n'
            'Answer with JSON only: {"looks": [{"n": 0, "look": "..."}, ...]} with one entry for every number.')
    return [q + COMPACT for q in (p1, p2)]
COMPACT = " Write the JSON compactly: no indentation, no line breaks inside a thing."      # about twice as many things fit under the token limit
WORDS = (("iron", "metal"), ("brass", "metal"), ("steel", "metal"), ("copper", "metal"), ("metal", "metal"), ("gold", "metal"), ("silver", "metal"), ("chrome", "metal"), ("stone", "stone"), ("granite", "stone"), ("marble", "stone"),
         ("velvet", "fabric"), ("wool", "fabric"), ("fabric", "fabric"), ("cloth", "fabric"), ("woven", "fabric"), ("linen", "fabric"), ("leather", "leather"), ("glass", "glass"), ("wax", "wax"), ("candle", "wax"), ("wicker", "wicker"), ("basket", "wicker"),
         ("paper", "paper"), ("parchment", "paper"), ("ceramic", "ceramic"), ("clay", "ceramic"), ("terracotta", "ceramic"), ("plastic", "plastic"), ("concrete", "concrete"), ("rope", "rope"), ("wood", "wood"), ("oak", "wood"), ("pine", "wood"), ("plank", "wood"))
PART_CLS = {"rod": "metal", "holder": "metal", "wick": "rope", "wax": "wax", "glass": "glass", "stopper": "wood", "bracket": "metal", "lid": "metal", "frame": "wood", "picture": "paper", "cloth": "fabric", "field": "fabric", "border": "fabric", "cover": "leather", "pages": "paper"}
def guess(x, part, style):
    """A part's material without the model: the part's own nature, else the thing's words, in the style guide's materials."""
    by = lambda t: next((c for w, c in WORDS if w in t.lower()), None); own = by(x["name"]) or by(x["look"]) or "wood"          # the name says what it is; the look only if the name does not
    cls = PART_CLS.get(part) or own
    if part in ("top", "leg", "rail", "shelf", "panel", "batten", "side", "body", "shaft", "book", "cord"): cls = own
    m = style["materials"]; src = {"wood": m["trim"], "metal": m["metal"], "stone": m["stone"], "fabric": m["fabric"], "leather": m["leather"], "concrete": m["concrete"], "plaster": m["plaster"]}.get(cls)
    if src: return {"material": src["material"], "cls": cls, "rgb": src["rgb"]}
    d = {"glass": ("dark green bottle glass", (40, 60, 50)), "wax": ("cream candle wax", (225, 215, 190)), "paper": ("aged cream paper", (205, 190, 160)), "wicker": ("woven willow wicker", (150, 115, 70)), "ceramic": ("glazed stoneware", (120, 100, 85)), "rope": ("hemp rope", (150, 130, 95))}.get(cls, ("painted wood", (110, 100, 90)))
    return {"material": d[0], "cls": cls, "rgb": list(d[1])}
def read_assign(txts, style, Lv):
    it = plain_items(Lv); sc = sculpt_items(Lv); d = {}
    for txt in ([txts] if isinstance(txts, str) else txts):
        try: d.update(C.json_in(txt))
        except Exception as e: print("assign not read:", e)
    got = {}
    for t in d.get("things") or []:
        try: got[int(t["n"])] = t
        except Exception: pass
    out = {"plain": {}, "looks": {}, "missing": []}
    for k, x in enumerate(it):
        g = got.get(k) or {}; gp = g.get("parts") if isinstance(g.get("parts"), dict) else {}; parts = {}
        if not g: out["missing"].append(x["kind"])
        for p in x["parts"]:
            q = gp.get(p) or gp.get(p.split()[0]) or (gp.get("book") if x["form"] == "books" else None)
            if isinstance(q, dict) and q.get("material"):
                cls = str(q.get("cls") or "other").lower(); parts[p] = {"material": str(q["material"]).strip().lower(), "cls": cls if cls in CLS else "other", "rgb": C.tame(C.rgb_ok(q.get("rgb")), vmax=0.9 if cls in ("wax", "paper") else 0.8)}
            else: parts[p] = guess(x, p, style)
            if "flame" in p or p == "glow" or "flame" in parts[p]["material"]: parts[p] = {"material": "flame", "cls": "emit", "rgb": [255, 200, 120]}
        c = g.get("content") if isinstance(g.get("content"), dict) else {}; ty = str(c.get("type") or "none").lower(); words = str(c.get("words") or "")
        want = {"rug": "pattern", "frame": "picture", "books": "books"}.get(x["form"])
        if x["form"] == "frame" and (ty == "mirror" or "mirror" in x["name"].lower()): ty = "mirror"
        elif want: ty = want
        elif x["form"] == "cloth": ty = ty if ty in ("picture", "trim") else ("picture" if any(k in x["name"].lower() for k in ("tapestr", "banner", "flag", "poster")) else "trim")
        elif ty not in ("label", "none"): ty = "none"
        if ty in ("pattern", "picture") and not words: words = x["look"]
        out["plain"][x["kind"]] = {"form": x["form"], "name": x["name"], "look": x["look"], "size": x["size"], "objects": x["objects"], "parts": parts, "content": {"type": ty, "words": words}}
    lk = {}
    for t in d.get("looks") or []:
        try: lk[int(t["n"])] = str(t["look"])
        except Exception: pass
    for k, x in enumerate(sc): out["looks"][x["kind"]] = lk.get(k) or x["look"]
    return out
def assign(world, answer):
    """answer: (prompts, texts, seconds) of the model, asked by style.py in the same process as the guide."""
    t0 = time.time(); style = json.load(open(f"{where.mat_dir(world)}/style.json")); Lv = level(world); p, txt, tl = answer[0], answer[1], {"s": answer[2]}
    a = read_assign(txt, style, Lv); a["llm"] = tl; a["seconds"] = round(time.time() - t0, 1)
    wk = where.mat_work(world); json.dump(a, open(f"{wk}/assign.json", "w"), indent=1); open(f"{wk}/assign_prompt.txt", "w").write("\n\n".join(q + "\n\n" + t for q, t in zip(p, txt)))
    style["objects"] = {"looks": a["looks"], "plain": {k: {"parts": v["parts"], "content": v["content"]} for k, v in a["plain"].items()}}; json.dump(style, open(f"{where.mat_dir(world)}/style.json", "w"), indent=1)
    return a

# ---------------------------------------------------------------- what the architecture groups are
GKEY = {"timber": "timber", "trim": "trim", "stone": "stone", "metal": "metal", "door": "door", "plaster": "plaster", "concrete": "concrete", "platform": "stone", "shelf": "trim", "frame": "trim", "casing": "trim", "sill": "stone"}
ZCLS = {"masonry": "stone", "brick": "brick", "timber_boards": "wood", "panelling": "wood", "tiles": "ceramic", "painted_plaster": "plaster", "patterned_plaster": "plaster", "wallpaper": "plaster", "concrete": "concrete", "painted_concrete": "concrete", "metal_panels": "metal"}
def groups_of(style, S):
    """-> {group: {material, rgb, cls, from: None | surface key whose own material it is}}"""
    out = {}; wz = style["walls"]; low = wz["lower"] if wz.get("split_m") else wz["upper"]; wall_cls = ZCLS.get(low["treatment"], "plaster")
    for g, m in sorted(S["materials"].items()):
        if g in SKIP_GROUPS: continue
        cls = m.get("cls", "")
        if g in ("wall", "floor", "ceiling"): out[g] = {"material": None, "rgb": None, "cls": cls, "from": {"wall": "wall_lower" if wz.get("split_m") else "wall_upper", "floor": "floor", "ceiling": "ceiling"}[g]}; continue
        key = GKEY.get(g) or {"wood": "trim", "stone": "stone", "metal": "metal", "concrete": "concrete", "plaster": "plaster"}.get(cls, "trim"); sm = style["materials"][key]
        r = {"material": sm["material"], "rgb": sm["rgb"], "cls": cls or key, "from": None, "style_key": key}
        if cls in ("stone", "plaster", "concrete") and cls == wall_cls: r["from"] = "wall_lower" if wz.get("split_m") else "wall_upper"; r["material"] = low["material"] + " (the wall's own)"
        out[g] = r
    return out

# ---------------------------------------------------------------- the job list
def surf_keys(style):
    w = style["walls"]; k = [("wall_upper", w["upper"], "wall"), ("floor", style["floor"], "floor"), ("ceiling", style["ceiling"], "ceil")]
    if w.get("split_m"): k.insert(0, ("wall_lower", w["lower"], "wall"))
    return k
def surface_words(z, fallback):
    """A surface's material without what is built separately (beams, trusses, joists) and without its motif: only the flat stuff itself."""
    BUILT = r"\b(beams?|truss\w*|joists?|rafters?|girders?|ducts?|pipes?)\b"
    m = re.sub(r"(,|\b(with|and|between|under|over)\b)\s*[\w\- ]*?" + BUILT + r"[^,]*", "", z["material"], flags=re.I).strip(" ,")
    if z.get("pattern"):          # the motif is stamped at its true repeat afterwards: the photograph is the plain ground
        mw = "|".join(re.escape(w_) for w_ in z["pattern"]["motif"].lower().split() if len(w_) > 4)
        m = re.sub(r",?\s*\b(with|and|in|showing|bearing)\s+[^,]*\b(motifs?|patterns?|patterned|stencil\w*|stars?|printed|damask|stripes?" + ("|" + mw if mw else "") + r")\b[^,]*", "", m, flags=re.I).strip(" ,") + ", plain and unpatterned"
    if len(m) < 6 or re.search(BUILT, m, flags=re.I): m = f"{fallback}, {C.colour_word(z['rgb'])}"
    return dict(z, material=m)
def flat_ref(path, rgb, w, h):
    Image.fromarray(np.full((max(64, h // 2 // 16 * 16), max(64, w // 2 // 16 * 16), 3), np.clip(rgb, 0, 255), np.float32).astype(np.uint8)).save(path); return path
def pal_words(style): return ", ".join(p["name"] for p in style.get("palette", [])[:6])
def jobs(world):
    style = json.load(open(f"{where.mat_dir(world)}/style.json")); Lv = level(world); wk = where.mat_work(world); os.makedirs(f"{wk}/ref", exist_ok=True); os.makedirs(f"{wk}/ans", exist_ok=True); J = []; place = style.get("place") or C.place_of(style["request"])
    A = json.load(open(f"{wk}/assign.json")) if os.path.exists(f"{wk}/assign.json") else read_assign([], style, Lv)
    for key, z, kind in surf_keys(style):
        z = surface_words(z, Lv["S"]["materials"].get({"wall": "wall", "floor": "floor", "ceil": "ceiling"}[kind], {}).get("words") or "plain surface")
        u, cells, ref, words = S5.swatch_plan(z, kind, S5.sd(key)); n = ref.shape[0]; C.u8(ref).resize((n // 2, n // 2), Image.LANCZOS).save(f"{wk}/ref/{key}.png")
        J.append({"name": key, "ref": f"{wk}/ref/{key}.png", "out": f"{wk}/ans/{key}.png", "w": n, "h": n, "seed": 7, "prompt": S5.swatch_words(z, kind, place, words)})
        if z.get("pattern"):
            J.append({"name": f"motif_{key}", "ref": flat_ref(f"{wk}/ref/motif_{key}.png", z["rgb"], 512, 512), "out": f"{wk}/ans/motif_{key}.png", "w": 512, "h": 512, "seed": 7, "prompt": S5.motif_words(z)})
    G = groups_of(style, Lv["S"])
    for g, r in G.items():
        if r["from"]: continue
        w, h = int(GRP_M[0] * 512), int(GRP_M[1] * 512); grain = " The grain runs from left to right along the whole picture; one plain length with no joints and no edges." if r["cls"] == "wood" else " One plain unbroken face that fills the frame, no joints."
        J.append({"name": f"group_{g}", "ref": flat_ref(f"{wk}/ref/group_{g}.png", r["rgb"], w, h), "out": f"{wk}/ans/group_{g}.png", "w": w, "h": h, "seed": 7, "prompt":
                  f"A close-up photograph taken straight on of a flat surface of {r['material']}. The picture shows an area {GRP_M[0]:g} metres wide and {GRP_M[1]:g} metres high, so grain and pores are fine and small.{grain} "
                  f"Flat, no perspective, evenly lit with soft flat light, no highlights, no reflections. Keep the colour of the given picture. Sharp focus, crisp real material detail, a little honest wear. {BARE}"})
    # part swatches: one per distinct material, the most used first; a material past the cap wears the nearest swatch of its class, tinted
    mats = {}
    for kind, x in A["plain"].items():
        for p, q in x["parts"].items():
            if q["cls"] == "emit": continue
            sz = x.get("size") or [0.3, 0.3, 0.3]; area = float(np.prod(sorted(sz)[-2:])) * len(x["objects"]) * (0.3 if p in ("rod", "wick", "stopper", "bracket", "lid", "rail") else 1.0)
            m = mats.setdefault(q["material"], {"cls": q["cls"], "rgb": [], "area": 0.0}); m["rgb"].append(q["rgb"]); m["area"] += area
    order = sorted(mats, key=lambda k: -mats[k]["area"])[:MAX_SW]; sw = {}
    for k, mname in enumerate(order):
        m = mats[mname]; rgb = np.median(np.float32(m["rgb"]), 0); grain = " The grain runs from left to right along the whole picture; one plain length with no joints and no edges." if m["cls"] == "wood" else (" The cloth hangs flat and smooth, its weave or pile running straight, no folds, no creases." if m["cls"] == "fabric" else " One plain unbroken face that fills the frame.")
        J.append({"name": f"sw_{k}", "ref": flat_ref(f"{wk}/ref/sw_{k}.png", rgb, SW_PX, SW_PX), "out": f"{wk}/ans/sw_{k}.png", "w": SW_PX, "h": SW_PX, "seed": 7, "prompt":
                  f"A close-up photograph taken straight on of a flat surface of {mname}. The picture shows an area only {SW_M:g} metre wide and {SW_M:g} metre high, so the grain, weave and pores are fine and small.{grain} "
                  f"The material fills the whole frame from edge to edge, flat, no perspective, evenly lit with soft flat light, no highlights, no reflections of a room. Keep the colour of the given picture. Sharp focus, crisp real material detail. {BARE}"})
        sw[mname] = {"k": k, "cls": m["cls"], "rgb": [float(v) for v in rgb]}
    # contents
    cont = {}
    for kind, x in A["plain"].items():
        c = x["content"]; ty = c["type"]; sz = x.get("size") or [1, 1, 1]
        if ty == "pattern":                       # a rug, seen from straight above: width x depth
            w_m, h_m = sz[0], sz[2]; ppm = float(min(512, (1.05e6 / (w_m * h_m)) ** 0.5)); w, h = C.r16(w_m * ppm), C.r16(h_m * ppm)
            J.append({"name": f"content_{kind}", "out": f"{wk}/ans/content_{kind}.png", "w": w, "h": h, "seed": 11, "prompt":
                      f"A flat-lay photograph taken from straight above of one whole hand-woven rug lying flat: {c['words']}. An intricate woven design with a patterned field and a contrasting ornamental border all the way round, in {pal_words(style)}; soft worn wool pile, slightly faded. "
                      "The rug fills the entire picture from edge to edge, its border touching all four edges of the frame; perfectly flat, no perspective, evenly lit, no floor around it, no fringe, no shadows, no objects on it, no text."})
            cont[kind] = {"type": ty, "size_m": [w_m, h_m], "px": [w, h]}
        elif ty == "picture":
            w_m, h_m = sz[0], sz[1]; ppm = float(min(640, (0.75e6 / (w_m * h_m)) ** 0.5)); w, h = C.r16(w_m * ppm), C.r16(h_m * ppm); woven = x["form"] == "cloth"
            J.append({"name": f"content_{kind}", "out": f"{wk}/ans/content_{kind}.png", "w": w, "h": h, "seed": 11, "prompt":
                      (f"A flat scan of one whole antique woven tapestry hanging flat: {c['words']}. Rich woven wool, visible weave, a woven border all the way round, in {pal_words(style)}. " if woven else f"A flat museum scan of {c['words']}. Finely detailed, slightly aged. ")
                      + "It fills the entire picture from edge to edge, seen exactly head-on with no perspective, evenly lit, no frame, no wall, no table, no shadows, no hands."})
            cont[kind] = {"type": ty, "size_m": [w_m, h_m], "px": [w, h]}
        else: cont[kind] = {"type": ty, "words": c.get("words", "")}
    json.dump(J, open(f"{wk}/jobs.json", "w"), indent=1); json.dump({"groups": G, "swatches": sw, "mats": {k: {"cls": v["cls"], "area": v["area"]} for k, v in mats.items()}, "contents": cont}, open(f"{wk}/plan.json", "w"), indent=1)
    print("material pictures to make:", len(J), "megapixels", round(sum(j["w"] * j["h"] for j in J) / 1e6, 2), "| surfaces", len(surf_keys(style)), "groups", sum(1 for r in G.values() if not r["from"]), "part swatches", len(sw), "of", len(mats), "contents", sum(1 for c in cont.values() if c["type"] in ("pattern", "picture")))
    return J

# ---------------------------------------------------------------- finish
def label_picture(words, rgb):
    """A stencilled label: dark paint letters with a ruled box, worn. -> RGBA-like (h, w) alpha 0..1 and its colour"""
    t = re.sub(r"[^A-Za-z0-9 .&-]", "", words).upper()[:12].strip() or "No 7"
    try: font = ImageFont.load_default(size=96)
    except Exception: font = ImageFont.load_default()
    im = Image.new("L", (1024, 384), 0); dr = ImageDraw.Draw(im); bb = dr.textbbox((0, 0), t, font=font); tw, th = bb[2] - bb[0], bb[3] - bb[1]; s = min(1.0, 860 / max(tw, 1))
    if s < 1:
        try: font = ImageFont.load_default(size=int(96 * s)); bb = dr.textbbox((0, 0), t, font=font); tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception: pass
    dr.text(((1024 - tw) / 2 - bb[0], (384 - th) / 2 - bb[1]), t, fill=255, font=font); dr.rectangle((40, 40, 984, 344), outline=255, width=10)
    a = np.asarray(im, np.float32) / 255
    for x in range(120, 1024, 150): a[:, x:x + 9] *= 0.0                              # stencil bridges
    a *= np.clip(0.75 + 0.5 * C.noise(a.shape, 60, 3) + 0.25 * C.noise(a.shape, 6, 4, 1), 0, 1); return a
def finish(world):
    t0 = time.time(); out = where.mat_dir(world); wk = where.mat_work(world); style = json.load(open(f"{out}/style.json")); Lv = level(world); S = Lv["S"]; pl = json.load(open(f"{wk}/plan.json")); T = {}
    A = json.load(open(f"{wk}/assign.json")) if os.path.exists(f"{wk}/assign.json") else read_assign([], style, Lv)
    libs, motifs = {}, {}
    for key, z, kind in surf_keys(style):
        p = f"{wk}/ans/{key}.png"; img = C.op(p) if os.path.exists(p) else S5.fake_swatch(z, kind, S5.sd(key)); libs[key] = S5.Library(img, z, kind, S5.sd(key))
        C.u8(libs[key].sw).save(f"{out}/tex/material_{key}.png")
        if z.get("pattern") and os.path.exists(f"{wk}/ans/motif_{key}.png"):
            m = S5.motif_alpha(C.op(f"{wk}/ans/motif_{key}.png"), z["rgb"])
            if m[0] is not None: motifs[key] = m
    if "wall_lower" not in libs: libs["wall_lower"] = libs["wall_upper"]
    F = S5.faces(S, Lv["b"]); surf = {}; t = time.time()
    for n, f in F.items():
        tex, rec = S5.lay_face(f, style, libs, motifs, F); C.u8(tex).save(f"{out}/tex/{n}.png"); A_ = f["A"]
        surf[n] = dict(rec, texture=f"tex/{n}.png", kind=f["kind"], px=[tex.shape[1], tex.shape[0]], size_m=[f["w"], f["h"]], uv_fit=A_.tolist(), u0=f["u0"], v0=f["v0"], v_up=f["kind"] == "wall",
                       tone_rgb=[float(v) for v in np.median(tex[::8, ::8].reshape(-1, 3), 0)], evenly_lit=True)
    T["lay_s"] = round(time.time() - t, 1); t = time.time()
    # groups
    G = pl["groups"]; groups = {}; gw, gh = int(GRP_M[0] * 512), int(GRP_M[1] * 512)
    for g, r in G.items():
        if r["from"]:
            lib = libs[r["from"]]; z = lib.z
            if g in ("wall", "floor", "ceiling"):                # the surface's own material with its own pieces, as a sheet a piece is cut from
                kind = {"wall": "wall", "floor": "floor", "ceiling": "ceil"}[g]; mode = "sheet"
                if g == "wall":                                  # by height, exactly as the walls: the same courses, the same split, the same pattern rows
                    H = float(S["room"][1]); sp = float(next(r_["split_m"] for r_ in surf.values() if r_["kind"] == "wall")); wl = S5.WEAR.get(style.get("wear"), 0.55)
                    up = S5.lay_zone(libs["wall_upper"], style["walls"]["upper"], 4.0, H - sp, "wall", 512, S5.sd("wall_upper", 1), motif=motifs.get("wall_upper"), wear=wl); C.u8(up).save(f"{out}/tex/sheet_wall_upper.png"); tex = up
                    if sp > 0: tex = S5.lay_zone(libs["wall_lower"], style["walls"]["lower"], 4.0, sp, "wall", 512, S5.sd("wall_lower", 1), motif=motifs.get("wall_lower"), wear=wl); C.u8(tex).save(f"{out}/tex/sheet_wall_lower.png")
                else: tex = S5.lay_zone(lib, z, 4.0, 3.0, kind, 512, S5.sd(r["from"], 1))
            else:                                                # dressed pieces of the wall's own stuff: its face without the wall's joints, a touch lighter (dressed, not rough)
                tex = np.clip(group_face(lib, (gh, gw)) * 1.07, 0, 255); mode = "wall's own"
            C.u8(tex).save(f"{out}/tex/group_{g}.png"); groups[g] = {"texture": f"tex/group_{g}.png", "px_per_m": 512.0, "material": r["material"] or z["material"], "cls": r["cls"], "from": r["from"], "mode": mode, "grain": S5.unit_of(z)[0] == "plank", "finish": z.get("finish")}
        else:
            p = f"{wk}/ans/group_{g}.png"
            if os.path.exists(p): a = S5.even(C.op(p), r["rgb"], 0.3 * 512)
            else: a = np.clip(np.float32(r["rgb"]) * (1 + 0.06 * C.noise((gh, gw), 30, 3)[..., None]), 0, 255)
            C.u8(a).save(f"{out}/tex/group_{g}.png"); groups[g] = {"texture": f"tex/group_{g}.png", "px_per_m": 512.0, "material": r["material"], "cls": r["cls"], "from": None, "mode": "swatch", "grain": r["cls"] == "wood", "rgb": r["rgb"]}
    # part swatches
    sw = pl["swatches"]; swr = {}
    for mname, s in sw.items():
        p = f"{wk}/ans/sw_{s['k']}.png"
        if os.path.exists(p): a = S5.even(C.op(p), s["rgb"], 0.2 * SW_PX, pull=0.6); e = int(0.06 * SW_PX); a = a[e:-e, e:-e]
        else: a = np.clip(np.float32(s["rgb"]) * (1 + 0.06 * C.noise((SW_PX, SW_PX), 30, 3)[..., None]), 0, 255)
        C.u8(a).save(f"{out}/tex/sw_{s['k']}.png"); swr[mname] = {"texture": f"tex/sw_{s['k']}.png", "px_per_m": SW_PX / SW_M, "cls": s["cls"], "rgb": s["rgb"]}
    parts = {}
    for kind, x in A["plain"].items():
        pr = {}
        for p, q in x["parts"].items():
            if q["cls"] == "emit": pr[p] = {"mode": "emit", "rgb": q["rgb"], "cls": "emit", "material": "flame"}; continue
            if q["material"] in swr: s = swr[q["material"]]; pr[p] = {"mode": "swatch", "texture": s["texture"], "px_per_m": s["px_per_m"], "rgb": q["rgb"], "cls": q["cls"], "material": q["material"]}
            else:
                cand = [v for v in swr.values() if v["cls"] == q["cls"]] or list(swr.values())
                if cand: s = min(cand, key=lambda v: float(np.abs(np.float32(v["rgb"]) - np.float32(q["rgb"])).sum())); pr[p] = {"mode": "swatch", "texture": s["texture"], "px_per_m": s["px_per_m"], "rgb": q["rgb"], "cls": q["cls"], "material": q["material"], "tinted": True}
                else: pr[p] = {"mode": "flat", "rgb": q["rgb"], "cls": q["cls"], "material": q["material"]}
        c = dict(pl["contents"].get(kind) or {"type": "none"}); cp = f"{wk}/ans/content_{kind}.png"
        if c["type"] in ("pattern", "picture"):
            if os.path.exists(cp):
                im = trim_margin(Image.open(cp).convert("RGB")); im.save(f"{out}/tex/content_{kind}.png"); c["texture"] = f"tex/content_{kind}.png"; c["px"] = list(im.size)
            else: c["type"] = "none"
        if c["type"] == "label":
            a = label_picture(c.get("words", ""), None); Image.fromarray((a * 255).astype(np.uint8)).save(f"{out}/tex/label_{kind}.png"); c["texture"] = f"tex/label_{kind}.png"
        if c["type"] == "mirror": c["rgb"] = [200, 205, 208]; c["cls"] = "mirror"
        if c["type"] == "books": c["colours"] = [p_["rgb"] for p_ in style.get("palette", [])]
        if c["type"] == "trim": c["rgb"] = S5.band_colour(style)
        parts[kind] = {"form": x["form"], "objects": x["objects"], "parts": pr, "content": c}
    T["groups_parts_s"] = round(time.time() - t, 1)
    ts = C.times(f"{wk}/jobs.json"); ks = {k: v for k, v in ts.items() if not k.startswith("_")}
    json.dump({"frame": "room metres as in boxes.json: x across, y up, z deep", "room": [float(v) for v in S["room"]],
               "how_to_read": "a surface's texel for a room point: uv = uv_fit @ (x, y, z, 1); px x = (u - u0) * px_per_m; px y = (size_m[1] - (v - v0)) * px_per_m for walls (v_up), else (v - v0) * px_per_m. mats.Mats does this.",
               "style": "style.json", "surfaces": surf, "groups": groups, "parts": parts, "glass": {"cls": "glass"}}, open(f"{out}/surfaces.json", "w"), indent=1)
    rep = {"kept_plan": {k: {"pieces_found_share": l.kept, "units": len(l.units), "laid": "cut-out pieces" if l.units else ("random cuts, joints drawn" if l.loose else "spread")} for k, l in libs.items()}, "klein_s": ks, "klein_s_sum": round(sum(ks.values()), 1), "klein_loads_s": {k: v for k, v in ts.items() if k.startswith("_")}, "klein_pictures": len(ks), "finish": T, "finish_s": round(time.time() - t0, 1),
           "style_s": style.get("seconds"), "assign_s": A.get("seconds"), "px_per_m": {n: round(s["px_per_m"], 1) for n, s in surf.items()}}
    json.dump(rep, open(f"{out}/report.json", "w"), indent=1); print(json.dumps(rep)); return rep
def trim_margin(im):
    """A picture of a flat thing without the plain margin Klein leaves round it (a white or grey surround): cut to where the picture differs from its corners' colour."""
    a = np.asarray(im, np.float32); h, w = a.shape[:2]; e = max(3, int(0.012 * min(h, w))); c = np.median(np.concatenate([a[:e, :e].reshape(-1, 3), a[:e, -e:].reshape(-1, 3), a[-e:, :e].reshape(-1, 3), a[-e:, -e:].reshape(-1, 3)]), 0)
    d = np.abs(a - c).max(-1) > 34; rows = np.flatnonzero(d.mean(1) > 0.5); cols = np.flatnonzero(d.mean(0) > 0.5)
    if len(rows) < 0.6 * h or len(cols) < 0.6 * w: y0, y1, x0, x1 = 0, h, 0, w          # no plain margin (or the picture itself is the corners' colour)
    else: y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    m = max(2, int(0.006 * min(h, w))); return im.crop((x0 + m, y0 + m, x1 - m, y1 - m))
def group_face(lib, shape):
    """A material's face without its laying joints: spread from the insides of its photographed units (or from the whole swatch if it has none)."""
    if not lib.units: return lib.spread(shape, 512, 9)
    big = [u for u in lib.units if min(u.shape[:2]) > 60]
    if not big: return lib.spread(shape, 512, 9)
    h, w = shape; out = np.zeros((h, w, 3), np.float32); acc = np.zeros((h, w), np.float32); rng = np.random.default_rng(5); mean = lib.tone
    acc2 = np.zeros((h, w), np.float32)
    for _ in range(int(6 * h * w / np.mean([u.shape[0] * u.shape[1] for u in big])) + 8):
        u = big[int(rng.integers(len(big)))]; e = max(6, int(0.12 * min(u.shape[:2]))); u = u[e:-e, e:-e]; ph, pw = u.shape[:2]; y, x = int(rng.integers(-ph // 2, h)), int(rng.integers(-pw // 2, w))
        ry = np.clip(np.minimum(np.arange(ph) + 0.5, ph - np.arange(ph) - 0.5) / (0.3 * ph), 0, 1); rx = np.clip(np.minimum(np.arange(pw) + 0.5, pw - np.arange(pw) - 0.5) / (0.3 * pw), 0, 1); win = np.outer(ry, rx).astype(np.float32) + 1e-4
        ya, xa = max(y, 0), max(x, 0); yb, xb = min(y + ph, h), min(x + pw, w)
        if yb <= ya or xb <= xa: continue
        ww = win[ya - y:yb - y, xa - x:xb - x]; out[ya:yb, xa:xb] += (u[ya - y:yb - y, xa - x:xb - x] - mean) * ww[..., None]; acc[ya:yb, xa:xb] += ww; acc2[ya:yb, xa:xb] += ww * ww
    return np.clip(mean + out / np.sqrt(np.maximum(acc2, 1e-6))[..., None], 0, 255)
def run(world):
    t0 = time.time(); jobs(world); C.klein(f"{where.mat_work(world)}/jobs.json"); r = finish(world); print(f"materials took {time.time() - t0:.0f} s", flush=True); return r
if __name__ == "__main__":
    if len(sys.argv) > 1: MAX_SW = int(sys.argv[1])
    run(os.path.abspath(os.environ["WORLD_DIR"]))
