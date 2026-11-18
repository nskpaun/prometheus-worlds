"""The style guide of a world: from the sentence alone, a concept picture and a written guide (style.json).
  style.py "SENTENCE" ["LOOK"]   -> WORLD_DIR/materials/{concept.png, style.json, work/assign.json}   (thread/.venv; the picture model and the
                         world-plan model each run in their own process, one after the other)
With the layout made, the same model process is asked two more things: a material for every part of every plain form and
what each flat thing shows (materials.py reads the answer), and a look for every sculpted kind in the guide's colours
(written into shapes/plan.json for the paint; the layout's own words are kept there as "look0").
The concept picture is a guide only: its colours are measured and told to the text model; it is never put on geometry.
style.json (every colour is the material's OWN colour in even daylight, sRGB 0..255; the light makes night):
  place, building, character, scheme   words (scheme: the dominant colour, the second colour, the accent)
  time                         day | dusk | night
  light {words, key_rgb, fill_rgb}                      the main light's colour and the fill's (for the light)
  palette [{name, rgb, use}]                            5 to 7 colours everything in the place is drawn from
  wear  pristine | light | worn | heavy ;  wear_words   how used the place is
  walls {split_m, lower ZONE, upper ZONE}               split_m 0: one treatment (upper) floor to ceiling; else lower below, upper above
  floor ZONE, ceiling ZONE, floor_marks [words]         floor_marks e.g. "tyre marks", "oil stains", "ash by the hearth"
  materials {timber, trim, stone, metal, door, plaster, concrete, fabric, leather} -> {material, rgb}       what kit pieces and objects of that kind are made of
  ZONE = {treatment, material, rgb, unit, unit_cm [w, l], finish, pattern null | {motif, rgb, repeat_cm}}
  concept_palette              the colours measured in the concept picture (top / middle / bottom band)"""
import sys, os, json, time
import numpy as np
from PIL import Image
import where, tools as C, chat
TREAT = "masonry, brick, timber_boards, panelling, tiles, painted_plaster, patterned_plaster, wallpaper, concrete, painted_concrete, metal_panels"
UNITS = "plank, flagstone, slab, tile, brick, block, panel, none"
ZONE = '{"treatment": "...", "material": "...", "rgb": [r, g, b], "unit": "...", "unit_cm": [w, l], "finish": "matte|satin|gloss", "pattern": null}'
def concept_words(req, look=""):
    return (f"Interior design photograph for a film set, wide angle, the whole room in view from a corner: {req.split('.')[0]}{'. ' + look.strip(' .') if look else ''}. "
            "Rich, specific, believable materials and a strong colour scheme; walls, floor and ceiling all visible; every lamp lit so the materials' own colours show clearly. Photorealistic, sharp, no people, no text.")
def measure(img, k=4):
    """The main colours of three horizontal bands of a picture: [{band, colours: [{rgb, share, word}]}] (k-means on a small copy)."""
    a = np.asarray(img.convert("RGB").resize((96, 72), Image.BOX), np.float32); out = []; rng = np.random.default_rng(0)
    for band, (y0, y1) in (("top (ceiling, upper walls)", (0, 22)), ("middle (walls, furniture)", (22, 50)), ("bottom (floor)", (50, 72))):
        p = a[y0:y1].reshape(-1, 3); c = p[rng.choice(len(p), k, replace=False)]
        for _ in range(12):
            d = ((p[:, None] - c[None]) ** 2).sum(-1); l = d.argmin(1); c = np.stack([p[l == i].mean(0) if (l == i).any() else c[i] for i in range(k)])
        sh = np.bincount(l, minlength=k) / len(p); o = np.argsort(-sh)
        out.append({"band": band, "colours": [{"rgb": [int(v) for v in c[i]], "share": round(float(sh[i]), 2), "word": C.colour_word(c[i])} for i in o if sh[i] > 0.08]})
    return out
def prompt(req, pal):
    seen = "; ".join(f'{b["band"]}: ' + ", ".join(f'{c["word"]} {c["rgb"]}' for c in b["colours"]) for b in pal)
    return (f'You are the production designer of a 3D place that is built from this request: "{req}"\n'
            f'A concept picture was painted from the request. Its main colours, measured (lamp-lit, so darker and warmer than the materials themselves): {seen}.\n'
            'Write the STYLE GUIDE the builders follow. The place must get its character from MATERIALS and COLOUR: specific, rich and believable, the way a real place of this kind at its best is finished '
            '(a wizard\'s study is opulent: deep colour, gilding, dark carved timber, worn stone, velvet; a modern showroom is designed: crisp colour blocking, polished floor, an accent colour). Never bland grey or beige by default. '
            'First settle the COLOUR SCHEME: one dominant deep or strong colour that the large surfaces carry, a second colour that sets it off, and a metallic or bright accent; the palette and every surface below follow it. '
            'The walls are the largest surface and must carry the scheme (a colour, a pattern, a rich material), never soot, dirt or plain dark brown; wear is added later and is not a colour. Where such a place would have one, give a wall or a floor a pattern.\n'
            'Every "rgb" is the material\'s OWN colour as it looks outdoors in even daylight (the concept picture is much darker than the materials are): 0 to 255, never pure white, never a pure primary, and even the darkest material (ebony, soot-black iron) is no darker than about [45, 40, 38].\n'
            'A "material" is a phrase a photographer would write for the bare flat surface: the species or kind, colour, finish and wear, for example "wide dark fumed oak boards, waxed and worn", '
            '"hand-made oxblood glazed tiles with dark grout", "smooth midnight-blue limewashed plaster", "honed pale limestone ashlar with fine lime joints", "satin white painted blockwork", "polished charcoal epoxy-sealed concrete". Name no object in it.\n'
            f'"treatment" is one of: {TREAT}. "unit" is the repeating piece, one of: {UNITS}; plaster, paint, wallpaper and poured concrete have unit "none". '
            '"unit_cm" is the real [width, length] of ONE piece as a builder buys it: floorboard [14 to 22, 200], flagstone [45, 60], brick [6.5, 21.5], stone block [25 to 35, 40 to 70], wall or floor tile [10 to 60, same], wall panel [50 to 90, 60 to 120], concrete slab [200 to 300, same]; [0, 0] for none.\n'
            '"pattern" is null, or (the "material" phrase then names only the plain ground, not the motif) for wallpaper / patterned plaster / decorated tiles a small repeated motif painted or printed on the surface: {"motif": "what ONE motif is, e.g. a gold six-pointed star", "rgb": [r, g, b], "repeat_cm": 20 to 60}.\n'
            'WALLS: either one treatment from floor to ceiling ("split_m": 0, described in "upper"), or two: "lower" from the floor up to "split_m" metres (0.9 to 2.2) and "upper" above it, as real rooms have (a dado of panelling, tiles, darker paint or stone below; plaster, paper or paint above). Choose what suits the place and gives it the most character.\n'
            '"wear": how used the place is, one of pristine, light, worn, heavy. "floor_marks": marks this kind of place really has on its floor (for example "tyre marks", "oil stains", "ash by the hearth", "candle wax drips", "scuffed walkways"), or [].\n'
            'Beams, trusses, joists, ducts, skirting, frames and furniture are built separately as real 3D pieces: a surface\'s "material" describes only the flat stuff itself.\n'
            '"materials": what the separately built pieces are made of here: "timber" (beams, posts), "trim" (skirting, frames, rails), "stone" (sills, surrounds, hearths, steps), "metal" (fittings, bars, handles, ducts), "door", "plaster" (mouldings), "concrete", "fabric" (curtains, upholstery), "leather".\n'
            '"light": the main light and the fill at the request\'s time of day.\n'
            'Answer with JSON only, exactly this shape:\n'
            '{"place": "...", "building": "what kind of real building this is and how it is built", "character": "one sentence", "scheme": "the dominant colour, the second colour and the accent, in words", "time": "day|dusk|night", '
            '"light": {"words": "...", "key_rgb": [r, g, b], "fill_rgb": [r, g, b]}, '
            '"palette": [{"name": "...", "rgb": [r, g, b], "use": "where it is used"}, ... 5 to 7 entries], '
            '"wear": "...", "wear_words": "one sentence on how and where the place is worn", '
            f'"walls": {{"split_m": 0, "lower": {ZONE}, "upper": {ZONE}}}, "floor": {ZONE}, "ceiling": {ZONE}, "floor_marks": ["..."], '
            '"materials": {"timber": {"material": "...", "rgb": [r, g, b]}, "trim": {...}, "stone": {...}, "metal": {...}, "door": {...}, "plaster": {...}, "concrete": {...}, "fabric": {...}, "leather": {...}}}')
def zone(z, default):
    z = dict(default, **(z or {})) if isinstance(z, dict) else dict(default); z["rgb"] = C.tame(C.rgb_ok(z.get("rgb"))); z["unit"] = str(z.get("unit") or "none").lower().rstrip("s"); cm = z.get("unit_cm") or [0, 0]
    try: z["unit_cm"] = [float(cm[0]), float(cm[1])]
    except Exception: z["unit_cm"] = [0.0, 0.0]
    z["material"] = str(z.get("material") or default["material"]); z["treatment"] = str(z.get("treatment") or "painted_plaster").lower().replace(" ", "_"); z["finish"] = str(z.get("finish") or "matte").lower()
    p = z.get("pattern")
    if isinstance(p, dict) and p.get("motif"):
        try: rc = float(p.get("repeat_cm") or 30)
        except Exception: rc = 30.0
        z["pattern"] = {"motif": str(p["motif"]), "rgb": C.tame(C.rgb_ok(p.get("rgb"), (190, 150, 60)), vmax=0.85), "repeat_cm": float(np.clip(rc, 12, 80))}
    else: z["pattern"] = None
    if z["treatment"] in ("painted_plaster", "patterned_plaster", "wallpaper", "concrete", "painted_concrete") and z["unit"] not in ("slab", "block"): z["unit"] = "none"; z["unit_cm"] = [0.0, 0.0]
    return z
def clean(d, req):
    """The model's answer made safe: every field present, colours tamed, sizes in range."""
    D0 = {"treatment": "painted_plaster", "material": "smooth painted plaster", "rgb": [150, 145, 135], "unit": "none", "unit_cm": [0, 0], "finish": "matte", "pattern": None}
    w = d.get("walls") or {}; out = {"request": req, "place": str(d.get("place") or C.place_of(req)), "building": str(d.get("building") or ""), "character": str(d.get("character") or ""), "scheme": str(d.get("scheme") or "")}
    t = str(d.get("time") or "day").lower(); out["time"] = t if t in ("day", "dusk", "night") else "day"
    L = d.get("light") or {}; out["light"] = {"words": str(L.get("words") or ""), "key_rgb": C.rgb_ok(L.get("key_rgb"), (255, 240, 220)), "fill_rgb": C.rgb_ok(L.get("fill_rgb"), (200, 210, 230))}
    out["palette"] = [{"name": str(p.get("name")), "rgb": C.tame(C.rgb_ok(p.get("rgb")), vmax=0.85), "use": str(p.get("use") or "")} for p in (d.get("palette") or []) if isinstance(p, dict)][:8]
    wr = str(d.get("wear") or "light").lower(); out["wear"] = wr if wr in ("pristine", "light", "worn", "heavy") else "light"; out["wear_words"] = str(d.get("wear_words") or "")
    try: sp = float(w.get("split_m") or 0)
    except Exception: sp = 0.0
    out["walls"] = {"split_m": float(np.clip(sp, 0.6, 2.4)) if sp > 0.3 else 0.0, "lower": zone(w.get("lower"), D0), "upper": zone(w.get("upper"), D0)}
    out["floor"] = zone(d.get("floor"), dict(D0, material="worn floorboards", unit="plank", unit_cm=[18, 200])); out["ceiling"] = zone(d.get("ceiling"), D0)
    out["floor_marks"] = [str(x) for x in (d.get("floor_marks") or []) if isinstance(x, str)][:6]
    M0 = {"timber": ("dark aged oak", (70, 50, 35)), "trim": ("painted wood", (120, 110, 100)), "stone": ("dressed limestone", (150, 142, 128)), "metal": ("dark iron", (60, 60, 62)), "door": ("oak boards", (90, 62, 40)),
          "plaster": ("smooth plaster", (190, 185, 175)), "concrete": ("fair-faced concrete", (140, 140, 138)), "fabric": ("heavy woven cloth", (90, 40, 50)), "leather": ("worn brown leather", (95, 55, 35))}
    m = d.get("materials") or {}; out["materials"] = {}
    for k, (words, rgb) in M0.items():
        x = m.get(k) if isinstance(m.get(k), dict) else {}; out["materials"][k] = {"material": str(x.get("material") or words), "rgb": C.tame(C.rgb_ok(x.get("rgb"), rgb))}
    return out
def concept(world, req, look=""):
    d = where.mat_dir(world); jp = f"{where.mat_work(world)}/concept_jobs.json"; t = time.time()
    if not os.path.exists(f"{d}/concept.png"):
        json.dump([{"name": "concept", "out": f"{d}/concept.png", "w": 1024, "h": 768, "seed": 7, "prompt": concept_words(req, look)}], open(jp, "w"), indent=1); C.klein(jp)
    ts = C.times(jp); return {"concept_wall_s": round(time.time() - t, 1), "concept_klein_s": ts.get("concept"), "concept_load_s": ts.get("_load_t2i")}
def save_guide(world, req, txt, pal, T):
    try: g = clean(C.json_in(txt), req)
    except Exception as e: print("The style guide could not be read: the world is built in plain default materials.", f"({type(e).__name__})", flush=True); g = clean({}, req); g["not_read"] = True
    g["concept_palette"] = pal; g["concept"] = "concept.png"; g["model"] = chat.MODEL; g["seconds"] = T
    json.dump(g, open(f"{where.mat_dir(world)}/style.json", "w"), indent=1); return g
def looks(world, style):
    """The guide's look of every sculpted kind into shapes/plan.json, where the paint reads it. A look that drops the
    thing's own name, or is empty or long, is left as the layout wrote it."""
    p = f"{world}/shapes/plan.json"; P = json.load(open(p)); new = (style.get("objects") or {}).get("looks") or {}; n = 0
    for k, v in P["kinds"].items():
        v.setdefault("look0", v["look"]); l = str(new.get(k) or "").strip(); noun = k.replace("_", " ").split()[-1].rstrip("s")
        if l and noun in l.lower() and len(l) < 220: v["look"] = l; n += 1
    json.dump(P, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p); print(f"{n} of {len(P['kinds'])} sculpted kinds take their look from the style guide", flush=True)
def main(world, req, look=""):
    """The guide, the part materials and the looks are asked in ONE model process. look: the style the user asked for in
    their own words, if they gave one; it goes to the concept picture and the guide, never to the layout."""
    import materials as B
    t0 = time.time(); T = concept(world, req, look); req = req if not look else f"{req.rstrip(' .')}. The look asked for: {look.strip(' .')}."; pal = measure(Image.open(f"{where.mat_dir(world)}/concept.png")); p = prompt(req, pal); wk = where.mat_work(world)
    Lv = B.level(world); prompts = [p] + B.assign_prompts(None, Lv, req)
    txt, tl = chat.ask_many(prompts, max_tokens=4500); T["llm"] = tl; T["total_s"] = round(time.time() - t0, 1); g = save_guide(world, req, txt[0], pal, T); open(f"{wk}/style_prompt.txt", "w").write(p + "\n\n" + txt[0])
    a = B.assign(world, answer=(prompts[1:], txt[1:], tl["s"][1:])); looks(world, json.load(open(f"{where.mat_dir(world)}/style.json")))
    print(f"style guide: {g.get('place')}, {g.get('scheme')}; {len(a['missing'])} plain kinds went by word rules; {T['total_s']} s", flush=True)
if __name__ == "__main__": main(os.path.abspath(os.environ["WORLD_DIR"]), sys.argv[1], *sys.argv[2:3])
