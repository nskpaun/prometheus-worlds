"""The room itself as real geometry: the chooser and shell_geometry(layout). CPU, numpy only, a few tenths of a second.
  import arch; S = arch.shell_geometry(layout, bounds=None)     # layout = boxes.json as a dict; bounds = {object index: (lo, hi)} of the real meshes
  S["meshes"]    [dict(piece, group, surface, V (n,3), F (m,3), UV (n,2) in METRES, grain, wall, ...)]  one per piece
  S["emitters"]  [dict(kind window|skylight|fitting, surface, rect (4 corners), poly, normal (into the room), area_m2, colour)]
  S["materials"] {group: dict(cls wood|plaster|stone|metal|glass|concrete|emitter, words)}  what each group is, for the materials
  S["pieces"]    what was chosen and where (plain records), S["counts"] how many of each
  S["claimed"]   indices of layout objects the architecture builds itself (platforms, windows, doors): every later stage skips them
  S["brief"]     the style and switches the chooser settled on; S["notes"] what it could not do
The chooser has two halves, both general:
  brief(layout)   reads the sentence (style words, window/door/skylight phrases, which wall) and the layout (setting, mood, names)
                  -> a brief: the STYLES entry plus the sentence's switches.
  place           fits the brief to this room: nothing stands where a layout object is; openings go in free wall; beams stop
                  for tall things; fireplaces get a chimney breast and hearth; curtains get a window behind; layout platforms get edges and steps."""
import re, time, copy
import numpy as np
from arch_mesh import Builder, Y, unit
import arch_kit as K
from arch_kit import WALLS

# ---------------------------------------------------------------- styles: sizes in metres. Data, not code: add a style by adding an entry.
MAT0 = dict(wall="wall", reveal="wall", floor="floor", ceiling="ceiling", frame="trim", glass="glass", sill="trim", casing="trim", door="door", metal="metal", shelf="trim",
            stone="stone", platform="platform", platform_top="platform", emitter="emitter", threshold="floor", trim="trim", timber="timber", pier="wall")
STYLES = {
 "medieval": dict(T=0.6, skirting=("plinth", 0.26, 0.07), string=(0.62, 0.09, 0.045), wallplate=(0.14, 0.1), ceiling="beams", beam=("timber", 0.24, 0.30), joist=(0.10, 0.15, 0.55), bay=2.0, corbel=(0.3, 0.34, 0.26), brace=False,
    window=dict(w=0.75, h=1.9, sill=1.0, head="pointed", splay=0.13, bars=(1, 5), frame_w=0.035, voussoirs=0.035, gap=1.1, per_wall=2, total=5, set=0.75),
    door=dict(w=1.15, h=2.25, head="round", leaf="plank", voussoirs=0.035), niche=dict(w=0.5, h=0.75, v=1.2, head="round", depth=0.28, total=3, voussoirs=0.025),
    chimney=dict(d=0.4, pad=0.3), hearth=True,
    mat=dict(MAT0, frame="metal", sill="stone", casing="stone", pier="stone"),
    words=dict(wall="rough coursed stone masonry", floor="worn stone flags or wide oak boards", ceiling="dark oak boards over joists", timber="dark aged oak beams", stone="dressed pale limestone blocks", trim="dark oak", metal="black wrought iron, lead cames", door="ledged oak boards", glass="small leaded diamond panes", platform="stone")),
 "rustic": dict(T=0.25, skirting=("plain", 0.12, 0.02), wallplate=(0.14, 0.1), ceiling="beams", beam=("timber", 0.2, 0.24), joist=(0.07, 0.14, 0.5), bay=1.8, post=(0.16, 0.16), brace=True,
    window=dict(w=0.95, h=1.05, sill=0.95, head="flat", bars=(1, 1), frame_w=0.05, casing=("plain", 0.09, 0.025), sill_d=0.1, gap=1.4, per_wall=2, total=4, apron=True),
    door=dict(w=0.95, h=2.0, head="flat", leaf="plank", casing=("plain", 0.1, 0.025)), chimney=dict(d=0.4, pad=0.3), hearth=True,
    mat=dict(MAT0, pier="timber"),
    words=dict(wall="rough timber boards or logs", floor="wide worn floorboards", ceiling="boards over joists", timber="hewn pine beams", stone="rough fieldstone", trim="plain pine", metal="black iron", door="ledged pine boards", glass="clear old glass", platform="timber")),
 "period": dict(T=0.35, skirting=("ogee", 0.24, 0.028), cornice=("classic", 0.16, 0.16), dado=("rail", 0.95, 0.06, 0.035), picture=("rail", 0.45, 0.045, 0.03), wainscot=dict(h=0.95, cell=0.62, stile=0.085, t=0.022),
    ceiling="coffers", coffer=(0.2, 0.14, 1.7), rose=0.42,
    window=dict(w=1.15, h=2.0, sill=0.7, head="flat", bars=(1, 3), frame_w=0.055, casing=("ogee", 0.12, 0.035), sill_d=0.07, gap=1.0, per_wall=3, total=5, meeting=True, apron=True),
    door=dict(w=0.95, h=2.15, head="flat", leaf="panel", casing=("ogee", 0.12, 0.035)), chimney=dict(d=0.4, pad=0.3), hearth=True, shelving=True,
    mat=dict(MAT0),
    words=dict(wall="painted plaster or wallpaper above panelling", floor="polished parquet", ceiling="white plaster", timber="dark walnut", stone="marble", trim="dark polished walnut joinery", metal="polished brass", door="panelled walnut", glass="clear glass", platform="walnut")),
 "modern": dict(T=0.22, skirting=("plain", 0.07, 0.012), ceiling="tray", tray=(0.55, 0.16), fittings=dict(L=1.2, w=0.09, drop=0.0, lane=2.2, colour="warm"),
    window=dict(w=1.6, h=1.9, sill=0.5, head="flat", bars=(0, 0), frame_w=0.04, gap=0.6, per_wall=3, total=4, sill_d=0.02, metal=True),
    door=dict(w=0.92, h=2.1, head="flat", leaf="flush", casing=("metal", 0.05, 0.012)), column=("round", 0.2),
    mat=dict(MAT0, frame="metal", sill="stone", casing="metal"),
    words=dict(wall="smooth painted plaster", floor="large pale stone tiles or oak boards", ceiling="white plaster", timber="pale oak", stone="pale honed limestone", trim="white painted", metal="dark anodised aluminium", door="flush oak veneer", glass="clear glass", platform="pale stone")),
 "industrial": dict(T=0.25, skirting=("cove", 0.09, 0.09), ceiling="services", beam=("concrete", 0.3, 0.38), bay=3.2, pier=(0.4, 0.16), column=("concrete", 0.2),
    fittings=dict(L=1.5, w=0.14, drop=0.38, lane=2.6, colour="cool"), duct=(0.5, 0.3), pipes=3, sprinkler=True,
    window=dict(w=1.9, h=0.65, sill=-1, head="flat", bars=(3, 0), frame_w=0.04, gap=0.5, per_wall=4, total=8, metal=True, sill_d=0.0),
    door=dict(w=0.95, h=2.1, head="flat", leaf="flush", casing=("metal", 0.06, 0.015)), big_door=dict(w=4.6, h=2.5, leaf="sectional"),
    mat=dict(MAT0, frame="metal", sill="wall", casing="metal", pier="concrete", platform="concrete", platform_top="metal"),
    words=dict(wall="smooth painted concrete or blockwork", floor="polished sealed concrete", ceiling="painted concrete soffit", timber="steel", stone="concrete", trim="grey resin coving", metal="galvanised steel", door="powder-coated steel", glass="clear glass", concrete="fair-faced concrete", platform="fair-faced concrete")),
 "scifi": dict(T=0.3, skirting=("kick", 0.16, 0.05), ceiling="ribs", beam=("rib", 0.34, 0.26), bay=2.4, pier=(0.34, 0.2), wainscot=dict(h=-0.55, cell=1.0, stile=0.07, t=0.03, flat=False, v0=0.2),
    fittings=dict(L=1.4, w=0.12, drop=0.0, lane=3.0, colour="cool"), pipes=2,
    window=dict(w=1.5, h=-0.9, sill=0.55, head="flat", bars=(0, 0), frame_w=0.07, gap=0.3, per_wall=6, total=0, metal=True, casing=("metal", 0.09, 0.03)),
    door=dict(w=1.5, h=2.2, head="flat", leaf="blast", casing=("metal", 0.14, 0.05)),
    mat=dict(MAT0, frame="metal", sill="metal", casing="metal", pier="metal", trim="metal", door="metal", shelf="metal", timber="metal", platform="metal", platform_top="metal"),
    words=dict(wall="grey painted steel panels", floor="polished dark metal deck plates", ceiling="steel panels", timber="steel", stone="steel", trim="steel", metal="brushed grey steel", door="armoured steel", glass="thick clear viewport glass", platform="steel deck")),
 "glasshouse": dict(T=0.3, skirting=("plinth", 0.18, 0.05), ceiling="glazed", beam=("steel", 0.09, 0.14), bay=1.6, pier=(0.1, 0.06), glaze_all=True,
    window=dict(w=1.2, h=-0.3, sill=0.7, head="segmental", bars=(1, 3), frame_w=0.035, gap=0.16, per_wall=99, total=99, metal=True, sill_d=0.08, set=0.3),
    door=dict(w=1.4, h=2.2, head="segmental", leaf="glazed"),
    mat=dict(MAT0, frame="metal", sill="stone", casing="metal", pier="metal", timber="metal"),
    words=dict(wall="low red brick wall, white painted iron above", floor="terracotta and encaustic tiles", ceiling="white painted iron glazing bars", timber="white painted cast iron", stone="pale stone sills", trim="white painted iron", metal="white painted cast iron", door="white painted iron and glass", glass="clear old glass", platform="stone")),
 "natural": dict(T=0.5, ceiling=None, mat=dict(MAT0), words=dict(wall="living rock", floor="rock and earth", ceiling="rock")),
 "outdoor": dict(T=0.35, ceiling=None, door=dict(w=1.5, h=2.05, head="round", leaf="plank", voussoirs=0.03), garden_pier=(0.5, 0.12, 3.0), mat=dict(MAT0, casing="stone", pier="wall", door="door"), words=dict(wall="old brick garden wall", floor="ground", stone="weathered stone coping", door="weathered oak gate", metal="black iron")),
}
LEX = {  # word: weight. What kind of place it is counts three, a period or mood word one.
 "medieval": "wizard:3 wizards:3 tower:3 castle:3 dungeon:3 keep:2 medieval:3 abbey:3 monastery:3 crypt:3 chapel:3 cathedral:3 gothic:2 alchemist:3 sorcerer:3 witch:2 throne:3 knight:2 fortress:3 cloister:3 temple:2 tomb:3 armoury:3 great hall:3 stone hall:3",
 "rustic": "cabin:3 log:2 barn:3 attic:3 cottage:3 tavern:3 inn:2 farmhouse:3 hut:3 lodge:3 stable:3 hayloft:3 shed:3 timber:1 rustic:2 saloon:3 mill:2 treehouse:3 boathouse:3",
 "period": "victorian:3 georgian:3 edwardian:3 study:2 library:3 manor:3 parlour:3 parlor:3 drawing room:3 clockmaker:3 clockmakers:3 ballroom:3 mansion:3 palace:3 museum:2 baroque:3 regency:3 gaslight:2 opera:3 theatre:2 gentlemen:2 walnut:1 antique shop:3 bookshop:3 apothecary:3",
 "modern": "modern:1 penthouse:3 lounge:2 apartment:3 minimalist:2 contemporary:2 office:3 gallery:3 studio:2 scandinavian:2 hotel:3 lobby:3 kitchen:2 bedroom:2 living room:2 luxury:1 cafe:2 showroom:2 gym:2 classroom:2",
 "industrial": "garage:3 workshop:2 warehouse:3 factory:3 hangar:3 laboratory:3 lab:3 depot:3 industrial:3 basement:2 boiler:3 engine room:3 bunker:3 car park:3 parking:3 forge:2 foundry:3",
 "scifi": "starship:3 spaceship:3 destroyer:2 space:2 station:1 cyberpunk:3 futuristic:3 sci-fi:3 alien:2 reactor:3 cockpit:3 command bridge:3 starfighter:3 airlock:3 spacecraft:3",
 "glasshouse": "conservatory:4 greenhouse:4 orangery:4 glasshouse:4 palm house:4",
 "natural": "cave:4 grotto:4 cavern:4 forest:3 jungle:3 burrow:4 nest:3 underwater:3 mine:2 den:1",
}
SIDE = {"back": "z1", "rear": "z1", "far": "z1", "end": "z1", "front": "z0", "left": "x0", "right": "x1"}
FIRE = ("fireplace", "hearth", "stove", "furnace", "forge", "chimney")
PLATFORM = ("platform", "stage", "dais", "podium", "plinth", "pedestal base", "walkway", "catwalk", "steps", "stairs", "terrace")
DRAPES = ("curtain", "drape", "blind", "shutter")

def style_of(text, setting=None):
    if setting == "outdoor": return "outdoor", {}
    t = " " + re.sub(r"[^a-z\- ]+", " ", text.lower().replace("'s", "s")) + " "; sc = {}
    for st, words in LEX.items():
        for wv in re.findall(r"([a-z\- ]+?):(\d)\s?", words):
            n = len(re.findall(r"\b" + re.escape(wv[0].strip()) + r"\b", t))
            if n: sc[st] = sc.get(st, 0) + int(wv[1]) * min(n, 2)
    if not sc: return "modern", sc
    return max(sc, key=lambda k: sc[k]), sc

def wall_named(text, what):
    """The wall a phrase puts a thing on: 'windows along the left wall', 'a round window at the end' -> x0, z1; else None."""
    for m in re.finditer(what, text):
        tail = text[m.end(): m.end() + 60]; tail = re.split(r"[,.;]", tail)[0]; head = re.split(r"[,.;]", text[max(0, m.start() - 50): m.start()])[-1]
        for src in (tail, head):
            k = re.search(r"\b(back|rear|far|end|front|left|right)\b", src)
            if k: return SIDE[k.group(1)]
    return None

def brief(layout):
    """The sentence and the layout -> the brief (a STYLES entry plus switches). Rules only."""
    text = (layout.get("request", "") + " " + layout.get("style", "")).lower(); st, sc = style_of(text, layout.get("setting"))
    B = copy.deepcopy(STYLES[st]); B["style"] = st; B["scores"] = sc; B["night"] = (layout.get("mood") or {}).get("time") in ("night", "evening", "dusk") or bool(re.search(r"\b(night|midnight|moonlit|evening|dusk)\b", text))
    B["clean"] = bool(re.search(r"\b(modern|sleek|minimal|minimalist|clean|luxury|pristine)\b", text)); win = B.get("window")
    if win:
        if re.search(r"\b(windowless|no windows|underground|cellar|dungeon|crypt|vault|bunker|basement)\b", text): B["window"] = win = None
        elif re.search(r"floor-to-ceiling window|wall of (tall |large |big )?windows|whole wall of|glass wall|panoramic window|picture window", text):
            win.update(mode="wall", wall=wall_named(text, r"windows?|glass wall"))
        elif re.search(r"\b(round|circular) window|porthole|rose window|oculus", text):
            win.update(mode="round", head="circle", w=0.9, h=0.9, sill=max(1.3, 0.45 * layout["room"][1]), total=1, per_wall=1, bars=(1, 1), wall=wall_named(text, r"window|porthole|oculus"), voussoirs=0, splay=0, casing=win.get("casing") or ("plain", 0.08, 0.025), sill_d=0)
        else:
            w = wall_named(text, r"\bwindows?\b")
            if w: win.update(wall=w)
            if st == "scifi" and re.search(r"\bwindows?|viewport|vista\b", text): win.update(total=12)
    if re.search(r"skylight|roof ?light|glass roof|glass ceiling|lantern roof|shafts? of (sun|moon)?light from above", text): B["skylight"] = dict(w=1.4, l=2.0)
    if re.search(r"\barch(ed|es|way)?\b", text) and B.get("door"): B["door"]["head"] = "round"
    if re.search(r"\b(pipes?|ducts?|ductwork|conduits?)\b", text): B["pipes"] = max(B.get("pipes", 0), 3)
    if re.search(r"\b(columns?|pillars?|colonnade)\b", text) and not any(re.search(r"column|pillar", o["name"].lower()) for o in layout.get("objects", [])): B["want_columns"] = True
    if re.search(r"\b(beams?|rafters?|joists?|timbered)\b", text) and B.get("ceiling") in ("tray", "coffers", None) and st != "outdoor": B.update(ceiling="beams", beam=("timber", 0.2, 0.26), joist=(0.08, 0.14, 0.5), bay=2.0)
    if re.search(r"\bcoffer", text): B.update(ceiling="coffers", coffer=(0.2, 0.14, 1.7))
    if st == "outdoor": B["enclose"] = bool(re.search(r"\bwalled|courtyard|cloister|brick walls?|garden wall|quadrangle\b", text))
    if st == "industrial" and not re.search(r"garage|hangar|workshop|depot|warehouse|showroom|car|truck|vehicle|loading", text): B.pop("big_door", None)
    return B

# ---------------------------------------------------------------- the layout as boxes to keep clear of
def obj_boxes(layout):
    out = []
    for i, o in enumerate(layout.get("objects", [])):
        lo, hi = np.full(3, 1e9), np.full(3, -1e9); rnd = False
        for p in o.get("prims", []):
            c = np.array(p.get("c", [0, 0, 0]), float)
            if p.get("shape") == "box" or "h" in p:
                h = np.array(p["h"], float); a = p.get("yaw", 0.0) or 0.0; cs, sn = abs(np.cos(a)), abs(np.sin(a)); e = np.array([cs * h[0] + sn * h[2], h[1], sn * h[0] + cs * h[2]])
            elif "r" in p: e = np.array([p["r"], p.get("hy", p["r"]), p["r"]]); rnd = True
            else: continue
            lo, hi = np.minimum(lo, c - e), np.maximum(hi, c + e)
        if lo[0] < 1e8: out.append(dict(i=i, name=o["name"].lower(), lo=lo, hi=hi, round=rnd and len(o["prims"]) == 1, yaw=o.get("yaw", (o["prims"][0].get("yaw") if o.get("prims") else 0) or 0)))
    return out

def cut(iv, cuts):
    for a, c in sorted(cuts):
        nxt = []
        for x, y in iv:
            if c <= x or a >= y: nxt.append((x, y)); continue
            if a > x: nxt.append((x, a))
            if c < y: nxt.append((c, y))
        iv = nxt
    return iv

class Placer:
    def __init__(s, layout, B, bounds=None):
        s.W, s.H, s.D = (float(v) for v in layout["room"]); s.B = B; s.fr = K.room_frames(s.W, s.H, s.D); s.objs = obj_boxes(layout); s.claimed = set()
        for o in s.objs:                                     # the true size of the made object, when the caller has it (the layout's box is often deeper)
            bd = (bounds or {}).get(o["i"], (bounds or {}).get(str(o["i"])))
            if bd is not None: o["lo"], o["hi"] = np.array(bd[0], float), np.array(bd[1], float)
        s.opens = {w: [] for w in list(WALLS) + ["ceil"]}; s.bumps = []; s.pieces = []; s.notes = []; s.long_z = s.D >= s.W; s.lines = []; s.ends = {}; s.shelves = []; s.hearths = []
    def occ(s, f, v0, v1, reach, pad=0.0):
        """Stretches of frame f that layout objects stand in front of (within `reach` of it, between heights v0 and v1)."""
        out = []
        for o in s.objs:
            if o["i"] in s.claimed or o["hi"][1] <= v0 or o["lo"][1] >= v1: continue
            cs = np.array([[x, 0, z] for x in (o["lo"][0], o["hi"][0]) for z in (o["lo"][2], o["hi"][2])]) - f.o; u = cs @ f.right; d = cs @ f.n
            if d.min() < reach and d.max() > -0.05 and u.max() > 0 and u.min() < f.L: out.append((u.min() - pad, u.max() + pad))
        return out
    def taken(s, w, pad=0.15, v0=0.0, v1=1e9):
        return [(o["u0"] - pad, o["u1"] + pad) for o in s.opens[w] if o["v0"] < v1 and o["v1"] > v0] + [(b["u0"] - pad, b["u1"] + pad) for b in s.bumps if b["wall"] == w and b.get("y0", 0) < v1 and b.get("y1", s.H) > v0]
    def free(s, w, v0, v1, reach, pad=0.15, margin=0.35):
        f = s.fr[w]; return cut([(margin, f.L - margin)], s.taken(w, pad, v0, v1) + s.occ(f, v0, v1, reach, pad))
    def hits(s, lo, hi, pad=0.0, skip=None):
        lo, hi = np.minimum(lo, hi) - pad, np.maximum(lo, hi) + pad
        return any((o["i"] not in s.claimed or o.get("platform")) and o["i"] != skip and (o["lo"] < hi).all() and (o["hi"] > lo).all() for o in s.objs)
    def near_wall(s, o, reach=0.3):
        """The wall this object stands against (and its stretch of it), or None."""
        best = None
        for w in WALLS:
            f = s.fr[w]; cs = np.array([[x, 0, z] for x in (o["lo"][0], o["hi"][0]) for z in (o["lo"][2], o["hi"][2])]) - f.o; d = cs @ f.n; u = cs @ f.right
            if d.min() < reach and (best is None or d.min() < best[1]): best = (w, d.min(), max(0, u.min()), min(f.L, u.max()), d.max())
        return best
    def add_open(s, o, **rec): s.opens[o["wall"]].append(o); s.pieces.append(dict(piece=o["kind"], wall=o["wall"], u=[round(o["u0"], 2), round(o["u1"], 2)], v=[round(o["v0"], 2), round(o["v1"], 2)], head=o["head"], **rec))

def window_opening(B, wall, u0, u1, v0, v1, kind="window", head=None, **kw):
    w = B["window"] or {}; d = dict(T=B["T"], splay=w.get("splay", 0), bars=w.get("bars", (0, 0)), frame_w=w.get("frame_w", 0.05), set=w.get("set", 0.6), sill=w.get("sill_d", 0), apron=w.get("apron"), casing=w.get("casing"), voussoirs=w.get("voussoirs"), meeting=w.get("meeting"))
    d.update(kw); return K.make_opening(kind, wall, u0, u1, v0, v1, head or w.get("head", "flat"), **d)

def choose(layout, B, bounds=None, hints=None):
    """Fit the brief to the room: openings, wall bumps, claimed objects. -> Placer with everything decided but not yet built."""
    P = Placer(layout, B, bounds); W, H, D = P.W, P.H, P.D; st = B["style"]
    if st in ("natural",): P.notes.append("a natural space (cave, forest): the kit has no pieces for rock; the shell is left plain"); return P
    if st == "outdoor" and not B.get("enclose"): P.notes.append("outdoors with no walls named: no shell to dress; only platforms and steps are built");
    # 1 layout objects that ARE architecture: platforms and stages (built with edges and steps), windows and doors (cut into the wall)
    for o in P.objs:                                         # a layout "object" that is really the shell (walls, a roof, a ceiling): the kit builds that
        if re.search(r"\b(walls?|ceiling|roof|floor|room)\b", o["name"]) and (o["hi"][0] - o["lo"][0] > 0.8 * W or o["hi"][2] - o["lo"][2] > 0.8 * D): P.claimed.add(o["i"]); P.notes.append(f"the layout's '{o['name']}' is the shell itself: claimed, not kept clear of")
    for o in P.objs:
        nm = o["name"]
        if any(k in nm for k in PLATFORM) and o["lo"][1] < 0.05 and o["hi"][1] <= 1.3 and min(o["hi"][0] - o["lo"][0], o["hi"][2] - o["lo"][2]) > 0.8: P.claimed.add(o["i"]); o["platform"] = True
    if st == "outdoor" and not B.get("enclose"): return P
    for o in P.objs:
        nm = o["name"]; nw = P.near_wall(o, 0.25)
        if nw and re.search(r"\b(window|porthole|viewport)\b", nm) and B.get("window") is not None:
            w, _, a, c, _ = nw; P.claimed.add(o["i"]); wn = B["window"]; v0 = max(o["lo"][1], 0.05) if o["lo"][1] > 0.3 else wn["sill"] if wn["sill"] > 0 else 0.9; v1 = min(max(o["hi"][1], v0 + 0.8), H - 0.2)
            P.add_open(window_opening(B, w, a + 0.02, c - 0.02, v0, v1), why="the layout's window")
        elif nw and re.search(r"\b(door|doorway|gate|hatch|entrance)\b", nm) and B.get("door"):
            w, _, a, c, _ = nw; P.claimed.add(o["i"]); dr = B["door"]; P.add_open(K.make_opening("door", w, a, c, 0, min(max(o["hi"][1], 1.9), H - 0.15), dr["head"], T=B["T"], leaf=dr["leaf"], casing=dr.get("casing"), voussoirs=dr.get("voussoirs")), why="the layout's door")
    # 2 a fireplace gets a chimney breast behind and above it and a hearth slab in front
    for o in P.objs:
        if any(k in o["name"] for k in FIRE) and B.get("chimney"):
            nw = P.near_wall(o, 0.3)
            if not nw: continue
            w, _, a, c, dmax = nw; ch = B["chimney"]; f = P.fr[w]; a2, c2 = max(0.06, a - ch["pad"]), min(f.L - 0.06, c + ch["pad"]); d = float(np.clip(0.55 * dmax, 0.12, ch["d"]))
            P.bumps.append(dict(wall=w, u0=a2, u1=c2, d=d, y0=0, y1=H, what="chimney breast", wrap=True, group=B["mat"]["wall"])); P.pieces.append(dict(piece="chimney breast", wall=w, u=[round(a2, 2), round(c2, 2)], depth=round(d, 2), why=f"behind the layout's {o['name']}"))
            if B.get("hearth"): P.hearths = P.hearths + [(w, max(0.05, a - 0.35), min(f.L - 0.05, c + 0.35), dmax + 0.45)]; P.pieces.append(dict(piece="hearth", wall=w, why=f"in front of the layout's {o['name']}"))
            o["fire"] = True
    win, dr = B.get("window"), B.get("door")
    # 3 curtains, drapes and blinds hang at a window: cut one behind them
    if win is not None:
        for o in P.objs:
            if any(k in o["name"] for k in DRAPES):
                nw = P.near_wall(o, 0.3)
                if not nw: continue
                w, _, a, c, _ = nw; wd = min(c - a - 0.3, max(win["w"], 0.55 * (c - a))); um = (a + c) / 2
                if wd < 0.4: continue
                v0 = max(win["sill"] if win["sill"] > 0 else 0.9, o["lo"][1] + 0.3); v1 = min(o["hi"][1] - 0.25, H - 0.35, v0 + max(win["h"] if win["h"] > 0 else 1.6, 1.4))
                if v1 - v0 > 0.5 and not any(x < um + wd / 2 and y > um - wd / 2 for x, y in P.taken(w)): P.add_open(window_opening(B, w, um - wd / 2, um + wd / 2, v0, v1, head="flat" if win.get("head") == "circle" else None), why=f"behind the layout's {o['name']}")
    for h in hints or []:                                    # openings someone else found (bright window shapes in the paint, say): kept if the wall is free there
        if h.get("wall") not in WALLS or (h["kind"] == "window" and win is None) or any(x < h["u1"] and y > h["u0"] for x, y in P.taken(h["wall"], 0.1)): P.notes.append(f"hint left out: {h}"); continue
        if h["kind"] == "window": P.add_open(window_opening(B, h["wall"], h["u0"], h["u1"], h["v0"], h["v1"], head=h.get("head")), why="hint")
        elif h["kind"] == "door" and dr: P.add_open(K.make_opening("door", h["wall"], h["u0"], h["u1"], 0, h["v1"], h.get("head", dr["head"]), T=B["T"], leaf=dr["leaf"], casing=dr.get("casing"), voussoirs=dr.get("voussoirs")), why="hint")
    # 5 doors: the big one first (a garage door needs the widest free wall), then the way in
    def place_door(spec, kind, order, clear):
        for w in order:
            f = P.fr[w]; wd = min(spec["w"], f.L - 1.2); hd = min(spec["h"], H - 0.25); iv = [(a, c) for a, c in P.free(w, 0, hd, clear, 0.25) if c - a >= wd]
            if not iv: continue
            a, c = max(iv, key=lambda t: t[1] - t[0]); um = (a + c) / 2 if (c - a) < wd + 1.5 or kind == "big" else (a + wd / 2 + 0.3 if abs(a + wd / 2 + 0.3 - f.L / 2) > abs(c - wd / 2 - 0.3 - f.L / 2) else c - wd / 2 - 0.3)
            glazed = spec["leaf"] == "glazed"
            o = K.make_opening("glazed_door" if glazed else "door", w, um - wd / 2, um + wd / 2, 0, hd, spec.get("head", "flat"), T=B["T"], leaf=spec["leaf"], casing=spec.get("casing"), voussoirs=spec.get("voussoirs"), bars=(1, 3) if glazed else (0, 0), frame_w=0.05, set=0.4)
            P.add_open(o, leaf=spec["leaf"], why="the way in" if kind != "big" else "vehicles named: an overhead door"); return w
        P.notes.append(f"no free wall for the {'big ' if kind == 'big' else ''}door"); return None
    has_door = any(o["kind"] == "door" for w in WALLS for o in P.opens[w]); bw = None
    if B.get("big_door"): bw = place_door(B["big_door"], "big", ("z0", "x0", "x1", "z1"), 1.2)
    glass_wall = win.get("wall") if win and win.get("mode") == "wall" else None            # keep the way in off a wall the sentence gives to glass
    if dr and not has_door: order = [w for w in ("z0", "x1", "x0", "z1") if w != bw] + ([bw] if bw else []); place_door(dr, "door", [w for w in order if w != glass_wall] + ([glass_wall] if glass_wall else []), 0.9)
    # 4 bays: where beams cross and piers stand. Along the room's long side.
    bay = B.get("bay"); P.long_z = D >= W; Ll, Ls = (D, W) if P.long_z else (W, D); P.lines = []
    if bay and B.get("ceiling") in ("beams", "services", "ribs", "glazed"):
        n = max(1, int(round(Ll / bay))); P.lines = [Ll * i / n for i in range(1, n)] if n > 1 else []
        if B.get("ceiling") in ("ribs", "glazed"): P.lines = [Ll * (i + 0.5) / n for i in range(n)] if B["ceiling"] == "ribs" else P.lines
    sides = ("x0", "x1") if P.long_z else ("z1", "z0")
    def u_on(w, s_): return {"x0": s_, "x1": D - s_, "z1": s_, "z0": W - s_}[w]
    P.u_on = u_on; pier = B.get("pier") or (B.get("post") and (B["post"][0], B["post"][1]))
    P.ends = {}                                              # (line, wall) -> "pier" | "corbel" | None
    for s_ in P.lines:
        for w in sides:
            u = u_on(w, s_); f = P.fr[w]; kind = None
            if pier:
                pw, pd = pier; clash = any(x < u + pw / 2 and y > u - pw / 2 for x, y in P.taken(w, 0.1) + P.occ(f, 0, H, pd + 0.03, 0.03))
                if not clash: P.bumps.append(dict(wall=w, u0=u - pw / 2, u1=u + pw / 2, d=pd, y0=0, y1=H, what="pier", wrap=B.get("post") is None, group=B["mat"]["pier"])); kind = "pier"
            if kind is None and B.get("corbel"):
                on_bump = any(b["wall"] == w and b["u0"] - 0.05 < u < b["u1"] + 0.05 for b in P.bumps)
                tall = any(x < u + 0.2 and y > u - 0.2 for x, y in P.occ(f, H - B["beam"][2] - B["corbel"][1] - 0.05, H, B["corbel"][2] + 0.05))
                kind = None if (on_bump or tall) else "corbel"
            P.ends[(s_, w)] = kind
    if P.lines: P.pieces.append(dict(piece="bay lines", along="z" if P.long_z else "x", at=[round(v, 2) for v in P.lines], piers=sum(v == "pier" for v in P.ends.values()), corbels=sum(v == "corbel" for v in P.ends.values())))
    if st == "outdoor":                                      # a garden wall: piers at even steps on every side, clear of the gate and of what stands against the wall
        pw, pd, step = B["garden_pier"]
        for w in WALLS:
            f = P.fr[w]; n = max(1, int(round(f.L / step)))
            for i in range(n + 1):
                u = float(np.clip(f.L * i / n, pw / 2 + 0.02, f.L - pw / 2 - 0.02))
                if not any(x < u + pw / 2 and y > u - pw / 2 for x, y in P.taken(w, 0.3) + P.occ(f, 0, 2.4, pd + 0.03, 0.03)): P.bumps.append(dict(wall=w, u0=u - pw / 2, u1=u + pw / 2, d=pd, y0=0, y1=2.4, what="pier", wrap=False, group=B["mat"]["pier"]))
        P.pieces.append(dict(piece="garden wall piers", count=len(P.bumps))); return P
    if B.get("ceiling") == "glazed":                         # a glass roof: a light in every bay, each side of the ridge
        ls = [0] + P.lines + [Ll]
        for i in range(len(ls) - 1):
            for t0, t1 in ((0.3, Ls / 2 - 0.12), (Ls / 2 + 0.12, Ls - 0.3)):
                a, c = ls[i] + 0.1, ls[i + 1] - 0.1; (x0, x1, z0, z1) = (t0, t1, a, c) if P.long_z else (a, c, t0, t1)
                if t1 - t0 > 0.5 and c - a > 0.5: P.opens["ceil"].append(K.make_opening("skylight", "ceil", x0, x1, z0, z1, "flat", T=0.14, set=0.7, splay=0, bars=(1, 2) if P.long_z else (2, 1), frame_w=0.035))
        P.pieces.append(dict(piece="glass roof lights", count=len(P.opens["ceil"]), why="a glass house"))
    # 6 windows
    if win is not None:
        mode = win.get("mode"); named = win.get("wall")
        if mode == "wall" or B.get("glaze_all"):
            walls = list(WALLS) if B.get("glaze_all") else [named or min(WALLS, key=lambda w: (len(P.opens[w]) * 10 + sum(c - a for a, c in P.occ(P.fr[w], 0, H, 0.4)) / P.fr[w].L - 0.01 * P.fr[w].L))]
            for w in walls:
                f = P.fr[w]; v0 = win["sill"] if B.get("glaze_all") else 0.06; v1 = H - (0.3 if B.get("glaze_all") else 0.12)
                for a, c in cut([(0.12, f.L - 0.12)], P.taken(w, 0.12)):
                    if c - a < 0.8: continue
                    if B.get("glaze_all"):
                        k = max(1, int(round((c - a) / (win["w"] + win["gap"])))); cw = (c - a - win["gap"] * (k - 1)) / k
                        for i in range(k): x = a + i * (cw + win["gap"]); P.add_open(window_opening(B, w, x, x + cw, v0, v1), why="a glass house: glazed all round")
                    else: P.add_open(window_opening(B, w, a, c, v0, v1, bars=(max(1, int(round((c - a) / 1.4)) - 1), 0), sill=0, casing=None, apron=None, meeting=None, frame_w=0.06, bar_w=0.06), why="a whole wall of glass" + (" (wall named in the sentence)" if named else " (the freest wall)"))
        elif win.get("total", 0) > 0:
            left = win["total"] - sum(o["kind"] == "window" for w in WALLS for o in P.opens[w]); order = ([named] if named else []) + [w for w in ("z1", "x0", "x1", "z0") if w != named]
            if mode == "round" and not named: order = ["z1", "z0", "x0", "x1"]
            for w in order:
                if left <= 0: break
                f = P.fr[w]; hh = win["h"] if win["h"] > 0 else H + win["h"] - (win["sill"] if win["sill"] > 0 else 0); hh = min(hh, H - 0.5)
                v0 = win["sill"] if win["sill"] > 0 else H - hh - 0.35 - (B["beam"][2] if B.get("ceiling") == "services" else 0); v0 = max(0.3, min(v0, H - hh - 0.3)); v1 = v0 + hh
                per = min(win["per_wall"] - sum(o["kind"] == "window" for o in P.opens[w]), left)
                if mode == "round":
                    iv = [(a, c) for a, c in P.free(w, v0, v1, 0.3) if c - a >= win["w"]]
                    if iv: a, c = min(iv, key=lambda t: abs((t[0] + t[1]) / 2 - f.L / 2)); um = float(np.clip(f.L / 2, a + win["w"] / 2, c - win["w"] / 2)); P.add_open(window_opening(B, w, um - win["w"] / 2, um + win["w"] / 2, v0, v1, head="circle"), why="a round window named in the sentence"); left -= 1
                    continue
                for a, c in P.free(w, v0, v1, 0.3, 0.3):
                    k = min(int((c - a + win["gap"]) / (win["w"] + win["gap"])), per)
                    if k <= 0: continue
                    step = (c - a) / k
                    for i in range(k): um = a + step * (i + 0.5); P.add_open(window_opening(B, w, um - win["w"] / 2, um + win["w"] / 2, v0, v1), why="free wall" + (" on the wall the sentence names" if w == named else ""))
                    per -= k; left -= k
                    if per <= 0 or left <= 0: break
    # 7 niches in what wall is left (stone rooms), shelving in the alcoves beside a chimney breast (period rooms)
    nc = B.get("niche")
    if nc:
        left = nc["total"]
        for w in ("x1", "x0", "z1", "z0"):
            for a, c in P.free(w, nc["v"] - 0.1, nc["v"] + nc["h"] + 0.1, 0.25, 0.45):
                if left > 0 and c - a >= nc["w"] + 0.3:
                    um = (a + c) / 2; P.add_open(K.make_opening("niche", w, um - nc["w"] / 2, um + nc["w"] / 2, nc["v"], nc["v"] + nc["h"], nc["head"], T=B["T"], depth=nc["depth"], voussoirs=nc.get("voussoirs")), why="free wall"); left -= 1
    P.shelves = []
    if B.get("shelving"):
        for b_ in [b for b in P.bumps if b["what"] == "chimney breast"]:
            for a, c in P.free(b_["wall"], 0, 2.2, b_["d"] + 0.05, 0.0, 0.03):
                if abs(c - b_["u0"]) < 0.02 and c - a >= 0.6: P.shelves.append((b_["wall"], max(a, c - 1.8), c, b_["d"] - 0.04))
                if abs(a - b_["u1"]) < 0.02 and c - a >= 0.6: P.shelves.append((b_["wall"], a, min(c, a + 1.8), b_["d"] - 0.04))
        for w, a, c, d in P.shelves: P.pieces.append(dict(piece="built-in shelving", wall=w, u=[round(a, 2), round(c, 2)], depth=round(d, 2), why="the alcove beside the chimney breast"))
    # 8 a skylight over the most open floor, between the beams
    sk = B.get("skylight")
    if sk and st != "outdoor":
        cells = []; ls = [0] + P.lines + [Ll]
        for i in range(len(ls) - 1):
            a, c = ls[i] + 0.35, ls[i + 1] - 0.35; ln = min(sk["l"], c - a); wd = min(sk["w"], Ls - 1.5)
            if ln < 0.6: continue
            m = (a + c) / 2; cells.append((abs(m - Ll / 2), m, ln, wd))
        if not P.lines: cells = [(0, Ll / 2, min(sk["l"], Ll - 1.5), min(sk["w"], Ls - 1.5))]
        if cells:
            _, m, ln, wd = min(cells); (x0, x1, z0, z1) = (Ls / 2 - wd / 2, Ls / 2 + wd / 2, m - ln / 2, m + ln / 2) if P.long_z else (m - ln / 2, m + ln / 2, Ls / 2 - wd / 2, Ls / 2 + wd / 2)
            o = K.make_opening("skylight", "ceil", x0, x1, z0, z1, "flat", T=0.45, set=0.85, splay=0, bars=(1, 2), frame_w=0.05); P.add_open(o, why="a skylight named in the sentence")
    return P

# ---------------------------------------------------------------- build
def build(P, layout):
    B = P.B; W, H, D = P.W, P.H, P.D; st = B["style"]; mat = B["mat"]; fr = P.fr; b = Builder(); em = []; outdoor = st == "outdoor"; Hw = H if not outdoor else 2.4
    shell = not outdoor or B.get("enclose")
    if outdoor:
        for f in (fr[w] for w in WALLS): f.H = Hw
        for bm in P.bumps: bm["y1"] = Hw
    # the six sides, walls with their openings cut out
    b.part("floor", mat["floor"], surface="floor"); K.surface(b, fr["floor"], W, D, [])
    if not outdoor: b.part("ceiling", mat["ceiling"], surface="ceil"); K.surface(b, fr["ceil"], W, D, P.opens["ceil"])
    if shell:
        for w in WALLS: b.part("wall", mat["wall"], surface=w, wall=w); K.surface(b, fr[w], fr[w].L, Hw, P.opens[w])
        for w in list(WALLS) + ["ceil"]:
            for o in P.opens[w]: K.opening_parts(b, fr[w], o, mat, em)
        for bm in P.bumps:                                   # chimney breasts, piers
            f = fr[bm["wall"]]; b.part(bm["what"], bm["group"], surface=bm["wall"], wall=bm["wall"], grain=bm["group"] == "timber")
            if outdoor:                                      # a garden wall's pier: through the wall, a little taller, with its own cap stone
                T = B["T"]; f.box(b, bm["u0"], bm["u1"], 0, Hw + 0.2, -T - bm["d"], bm["d"], skip=("-1",)); b.part("pier cap", mat["stone"], wall=bm["wall"])
                b.prism([(-T - bm["d"] - 0.05, 0), (bm["d"] + 0.05, 0), (bm["d"] + 0.05, 0.05), (-T / 2, 0.14), (-T - bm["d"] - 0.05, 0.05)], f.P(bm["u0"] - 0.05, Hw + 0.2), f.n, f.up, f.right, bm["u1"] - bm["u0"] + 0.1, grain=False); continue
            f.box(b, bm["u0"], bm["u1"], bm["y0"], bm["y1"], 0, bm["d"], skip=("-2", "-1") + (("+1",) if bm["y1"] >= Hw - 1e-6 and not outdoor else ()), grain=1 if bm["group"] == "timber" else None)
        # mouldings round the plan
        wrap = [bm for bm in P.bumps if bm.get("wrap")]
        def gaps(v0, v1, pad=0.0): return {w: [(o["u0"] - pad, o["u1"] + pad) for o in P.opens[w] if o["v0"] < v1 and o["v1"] > v0] for w in WALLS}
        def cpad(o): return (o.get("casing") or (0, 0, 0))[1] if not o.get("voussoirs") else 0.0
        sk = B.get("skirting")
        if sk:
            g = {w: [(o["u0"] - cpad(o), o["u1"] + cpad(o)) for o in P.opens[w] if o["v0"] < sk[1]] for w in WALLS}; runs, cl = K.plan_runs(fr, wrap, g, 0.0)
            b.part("coving" if sk[0] == "cove" else "skirting", mat["trim"] if sk[0] not in ("plinth",) else mat["stone"], grain=True, profile=sk[0]); K.mould(b, runs, cl, K.prof_skirting(*sk)); P.pieces.append(dict(piece="floor coving" if sk[0] == "cove" else "skirting", profile=sk[0], h=sk[1], runs=len(runs)))
        for key in ("dado", "picture"):
            r = B.get(key)
            if r and not outdoor:
                y = r[1] if key == "dado" else H - r[1]; g = {w: [(o["u0"] - cpad(o), o["u1"] + cpad(o)) for o in P.opens[w] if o["v0"] < y + r[2] and o["v1"] > y - r[2]] for w in WALLS}
                for w, a, c, d in getattr(P, "shelves", []): g[w].append((a, c)) if y < 2.2 else None
                runs, cl = K.plan_runs(fr, wrap, g, y); b.part(key + " rail", mat["trim"], grain=True); K.mould(b, runs, cl, K.prof_rail("rail", y, r[2], r[3])); P.pieces.append(dict(piece=key + " rail", y=round(y, 2), runs=len(runs)))
        sc_ = B.get("string")                                # a stone string course, a fixed share of the way up the wall
        if sc_ and not outdoor:
            y = round(sc_[0] * H, 2); g = {w: [(o["u0"] - (o.get("block", 0.2) if o.get("voussoirs") else cpad(o)), o["u1"] + (o.get("block", 0.2) if o.get("voussoirs") else cpad(o))) for o in P.opens[w] if o["v0"] - 0.1 < y + sc_[1] and o["v1"] + 0.25 > y - sc_[1]] + P.occ(fr[w], y - sc_[1], y + sc_[1], sc_[2] + 0.03) for w in WALLS}
            runs, cl = K.plan_runs(fr, wrap, g, y); b.part("string course", mat["stone"]); K.mould(b, runs, cl, [(0, y - sc_[1] / 2), (sc_[2], y - sc_[1] / 2 + 0.03), (sc_[2], y + sc_[1] / 2 - 0.02), (0, y + sc_[1] / 2)]); P.pieces.append(dict(piece="string course", y=y, runs=len(runs)))
        wp = B.get("wallplate")                              # a timber plate along the top of the wall, under the joists
        if wp and not outdoor:
            g = gaps(H - wp[0], H); runs, cl = K.plan_runs(fr, wrap, g, H); b.part("wall plate", mat["timber"], grain=True); K.mould(b, runs, cl, [(0, H - wp[0]), (wp[1] - 0.012, H - wp[0]), (wp[1], H - wp[0] + 0.012), (wp[1], H)]); P.pieces.append(dict(piece="wall plate", section=list(wp)))
        co = B.get("cornice")
        if co and not outdoor:
            g = gaps(H - co[1], H); runs, cl = K.plan_runs(fr, wrap, g, H); b.part("cornice", "plaster_trim", profile=co[0]); K.mould(b, runs, cl, [(o_, H + y_) for o_, y_ in K.prof_cornice(*co)]); P.pieces.append(dict(piece="cornice", profile=co[0], h=co[1]))
        if outdoor:                                          # a garden wall's coping
            T = B["T"]; runs, cl = K.plan_runs(fr, wrap, {}, Hw); b.part("coping", mat["stone"]); K.mould(b, runs, cl, [(0.05, Hw), (0.05, Hw + 0.05), (-T / 2, Hw + 0.12), (-T - 0.05, Hw + 0.05), (-T - 0.05, Hw)]); P.pieces.append(dict(piece="coping"))
        # panelling on every straight stretch of plan that is free
        wc = B.get("wainscot")
        if wc and not outdoor:
            hw = wc["h"] if wc["h"] > 0 else H + wc["h"]; v0 = wc.get("v0", (B["skirting"][1] if B.get("skirting") else 0)); g = gaps(v0, hw, 0.0)
            for w in WALLS:
                g[w] = [(a - cpad_, c + cpad_) for (a, c), cpad_ in zip(g[w], [cpad(o) for o in P.opens[w] if o["v0"] < hw and o["v1"] > v0])] + [(a, c) for w2, a, c, d in getattr(P, "shelves", []) if w2 == w]
                g[w] += P.occ(fr[w], v0, hw, 0.06)
            runs, cl = K.plan_runs(fr, wrap, g, 0.0); n = 0
            for f in K.run_frames(runs, cl, H):
                if f.L < 0.3: continue
                nx = max(1, int(round(f.L / wc["cell"]))); b.part("panelling", mat["trim"], grain=mat["trim"] != "metal"); K.panel_field(b, f, 0.0, f.L, v0, hw, 0.0, nx, 1 if hw < 1.4 else 2, wc["stile"], wc["stile"], wc["t"], flat=wc.get("flat", False), grain=mat["trim"] != "metal"); n += nx
            P.pieces.append(dict(piece="wall panelling", h=round(hw, 2), panels=n))
        for w, a, c, d in getattr(P, "shelves", []): K.shelving(b, fr[w], a, c, 0.0, 2.2, d, mat)
        for w, a, c, pr in getattr(P, "hearths", []): K.hearth(b, fr[w], a, c, pr, mat)
    # the ceiling: beams on the bay lines, joists, corbels, braces, coffers, ribs, tray
    ce = B.get("ceiling"); Ll, Ls = (D, W) if P.long_z else (W, D); sides = ("x0", "x1") if P.long_z else ("z1", "z0")
    def pt(s_, t, y): return np.array([t, y, s_]) if P.long_z else np.array([s_, y, t])       # s along the long side, t across
    def depth_at(w, u): return max([bm["d"] for bm in P.bumps if bm["wall"] == w and bm["u0"] - 1e-6 <= u <= bm["u1"] + 1e-6] or [0.0])
    if ce in ("beams", "services", "ribs", "glazed") and not outdoor:
        kind, bw, bh = B["beam"]; nb = 0
        for s_ in P.lines:
            if P.hits(pt(s_ - bw / 2, 0.3, H - bh), pt(s_ + bw / 2, Ls - 0.3, H)): P.notes.append(f"beam at {s_:.2f} left out: something tall stands there"); continue
            b.part("beam" if ce != "ribs" else "ceiling rib", mat["timber"] if kind in ("timber", "steel") else mat["pier"], grain=kind == "timber"); K.beam(b, pt(s_, 0, H), pt(s_, Ls, H), bw, bh, kind); nb += 1
            for w, t_end, sg in ((sides[0], 0.0, 1), (sides[1], Ls, -1)):
                e = P.ends.get((s_, w)); u = P.u_on(w, s_)
                if e == "corbel": cw, chh, cp = B["corbel"]; b.part("corbel", mat["stone"]); K.corbel(b, fr[w], u, H - bh, cw, chh, cp)
                if B.get("brace") and e == "pier":
                    d0 = depth_at(w, u); L_ = min(0.75, 0.3 * Ls, H - bh - 1.9)
                    if L_ > 0.3 and not P.hits(pt(s_ - 0.1, t_end + sg * 0.05, H - bh - L_), pt(s_ + 0.1, t_end + sg * (d0 + L_), H - bh)):
                        b.part("knee brace", mat["timber"], grain=True); K.brace(b, pt(s_, t_end + sg * (d0 - 0.02), H - bh - L_), pt(s_, t_end + sg * (d0 + L_), H - bh + 0.03), 0.11, 0.1)
        P.pieces.append(dict(piece="beams" if ce != "ribs" else "ceiling ribs", kind=kind, section=[bw, bh], count=nb))
        jo = B.get("joist")
        if jo and ce == "beams":
            jw, jh, jsp = jo; n = max(2, int(round(Ls / jsp))); b.part("joist", mat["timber"], grain=True)
            for i in range(n): t = Ls * (i + 0.5) / n; K.beam(b, pt(0, t, H), pt(Ll, t, H), jw, jh, "concrete")
            P.pieces.append(dict(piece="joists", section=[jw, jh], count=n))
        if ce == "glazed":                                   # glass roof: a light in every bay between the ribs, a ridge beam down the middle
            b.part("ridge beam", mat["timber"]); K.beam(b, pt(0, Ls / 2, H), pt(Ll, Ls / 2, H), bw, bh * 1.2, kind)
    if ce == "coffers" and not outdoor:
        cw, chh, cell = B["coffer"]; b.part("coffer beam", "plaster_trim"); nz = max(1, int(round(Ll / cell))); nx = max(1, int(round(Ls / cell))); cnt = 0
        for i in range(1, nz):
            s_ = Ll * i / nz
            if not P.hits(pt(s_ - cw / 2, 0.3, H - chh), pt(s_ + cw / 2, Ls - 0.3, H)): K.beam(b, pt(s_, 0, H), pt(s_, Ls, H), cw, chh, "concrete"); cnt += 1
        for i in range(1, nx):
            t = Ls * i / nx
            if not P.hits(pt(0.3, t - cw / 2, H - chh), pt(Ll - 0.3, t + cw / 2, H)): K.beam(b, pt(0, t, H), pt(Ll, t, H), cw, chh, "concrete"); cnt += 1
        if B.get("rose"):
            b.part("ceiling rose", "plaster_trim")
            for i in range(nz):
                for j in range(nx):
                    if (i + j) % 2 == 0 and min(Ll / nz, Ls / nx) > 1.1: p = pt(Ll * (i + 0.5) / nz, Ls * (j + 0.5) / nx, H); K.rose(b, p[0], p[2], H, min(B["rose"], 0.3 * min(Ll / nz, Ls / nx)))
        P.pieces.append(dict(piece="coffered ceiling", cells=[nz, nx], beams=cnt))
    if ce == "tray" and not outdoor:
        tw, td = B["tray"]; tw = min(tw, 0.14 * Ls); g = {w: [(o["u0"], o["u1"]) for o in P.opens[w] if o["v1"] > H - td] for w in WALLS}; runs, cl = K.plan_runs(fr, [], g, H)
        b.part("ceiling bulkhead", mat["ceiling"]); K.mould(b, runs, cl, [(0, H - td), (tw, H - td), (tw, H - td + 0.05), (tw - 0.08, H - td + 0.05), (tw - 0.08, H)]); P.pieces.append(dict(piece="ceiling bulkhead with light cove", w=round(tw, 2), drop=td))
    # free-standing columns on the bay lines, where the room is wide and the floor is clear
    col = B.get("column")
    if col and not outdoor and (Ls > 8.5 or B.get("want_columns")) and Ls > 5:
        n = 0; ck, cr = col; ck = "classical" if st in ("period", "medieval") else ck; lines = P.lines or [Ll * i / max(2, int(round(Ll / 3.5))) for i in range(1, max(2, int(round(Ll / 3.5))))]
        for s_ in lines:
            for t in (Ls / 3, 2 * Ls / 3):
                p = pt(s_, t, 0)
                if P.hits(p - [cr + 0.45, 0, cr + 0.45], p + [cr + 0.45, H, cr + 0.45]): continue
                top = H - (B["beam"][2] if B.get("beam") and s_ in P.lines else 0); b.part("column", mat["pier"] if ck != "classical" else mat["stone"], kind=ck); K.column(b, p[0], p[2], 0, top, cr, "concrete" if ck == "round" else ck); n += 1
        if n: P.pieces.append(dict(piece="columns", kind=ck, count=n))
    # services: a duct run and pipe runs under the ceiling, sprinkler main, rows of light fittings
    lane_used = []
    if B.get("duct") and not outdoor:
        dw, dh = B["duct"]; bh = B["beam"][2] if B.get("beam") else 0; y = H - bh - 0.08 - dh / 2
        for frac in (0.3, 0.7, 0.22, 0.78):
            t = Ls * frac
            if y - dh / 2 > 2.1 and not P.hits(pt(0.5, t - dw / 2, y - dh / 2), pt(Ll - 0.5, t + dw / 2, H)):
                b.part("duct", mat["metal"]); gr = K.duct(b, pt(0.45, t, y), pt(Ll - 0.45, t, y), dw, dh, hang=H - y)
                b.part("duct grille", mat["metal"])
                for p, sd in gr: K.grille(b, p - Y * (dh / 2 + 0.012), sd, unit(pt(1, 0, 0) - pt(0, 0, 0)), dw * 0.7, 0.5)
                lane_used.append((t - dw / 2 - 0.25, t + dw / 2 + 0.25)); P.pieces.append(dict(piece="duct run", section=[dw, dh], y=round(y, 2), across=round(t, 2))); break
    if B.get("pipes") and not outdoor and shell:
        n = B["pipes"] if not B.get("clean") else max(1, B["pipes"] - 1); w = sides[1] if not any(o["kind"] == "door" for o in P.opens[sides[1]]) else sides[0]; f = fr[w]
        dmax = max([bm["d"] for bm in P.bumps if bm["wall"] == w] or [0]); bh = B["beam"][2] if B.get("beam") else 0; top = max([o["v1"] for o in P.opens[w]] or [0]); b.part("pipe", mat["metal"]); cnt = 0
        for k in range(n):
            r = (0.045, 0.03, 0.022, 0.03)[k % 4]; d = dmax + 0.07 + 0.0; y = H - bh - 0.14 - 0.13 * k
            if y < max(top + 0.08, 2.0): P.notes.append("a pipe left out: no room between the openings and the beams"); continue
            run = [f.P(0.5 + 0.12 * k, y, d), f.P(f.L - 0.25 - 0.1 * k, y, d)]
            lo_ = f.P(f.L - 0.45, 0, 0.0); hi_ = f.P(f.L - 0.1, y, d + 0.1)
            if k < 2 and not P.hits(lo_, hi_): run.append(f.P(f.L - 0.25 - 0.1 * k, 0.0, d))            # down to the floor in the corner
            K.pipe(b, run, r, clip_to=-f.n * d); cnt += 1
        if cnt: P.pieces.append(dict(piece="pipe runs", wall=w, count=cnt))
    if B.get("sprinkler") and not outdoor:
        bh = B["beam"][2] if B.get("beam") else 0; y = H - bh - 0.06; t = Ls * 0.5
        if any(a < t < c for a, c in lane_used): t = Ls * 0.56
        if y > 2.1 and not P.hits(pt(0.3, t - 0.05, y - 0.1), pt(Ll - 0.3, t + 0.05, H)):
            b.part("sprinkler main", mat["metal"]); K.pipe(b, [pt(0.3, t, y), pt(Ll - 0.3, t, y)], 0.02, clip_to=Y * (H - y), every=2.4); b.part("sprinkler head", mat["metal"])
            for i in range(max(2, int(Ll / 3))): p = pt(Ll * (i + 0.5) / max(2, int(Ll / 3)), t, y); b.cyl(p - Y * 0.02, p - Y * 0.07, 0.008, 8); b.lathe([(0.0, -0.1), (0.03, -0.085), (0.0, -0.07)], p, 10, caps=(False, False))
            lane_used.append((t - 0.3, t + 0.3)); P.pieces.append(dict(piece="sprinkler main", y=round(y, 2)))
    ft = B.get("fittings")
    if ft and not outdoor:
        ls = [0] + P.lines + [Ll]; nl = max(1, int(round(Ls / ft["lane"]))); n = 0; inset_ = (B["tray"][0] + 0.3) if ce == "tray" else 0.0
        if not P.lines: k = max(1, int(round(Ll / 3.0))); ls = [Ll * i / k for i in range(k + 1)]
        for i in range(len(ls) - 1):
            m = (ls[i] + ls[i + 1]) / 2
            for j in range(nl):
                t = inset_ + (Ls - 2 * inset_) * (j + 0.5) / nl
                if any(a < t < c for a, c in lane_used): continue
                if any(o["u0"] - 0.2 < (t if P.long_z else m) < o["u1"] + 0.2 and o["v0"] - 0.2 < (m if P.long_z else t) < o["v1"] + 0.2 for o in P.opens["ceil"]): continue
                L_ = min(ft["L"], ls[i + 1] - ls[i] - 0.6)
                if L_ < 0.5 or P.hits(pt(m - L_ / 2, t - 0.1, H - ft["drop"] - 0.1), pt(m + L_ / 2, t + 0.1, H)): continue
                K.fitting(b, pt(m, t, H), pt(1, 0, 0) - pt(0, 0, 0), L_, ft["w"], ft["drop"], mat, em, ft["colour"]); n += 1
        if n: P.pieces.append(dict(piece="linear light fittings", count=n, drop=ft["drop"]))
    # platforms and stages the layout names: proper edges, steps on the side with the most clear floor
    for o in P.objs:
        if not o.get("platform"): continue
        c = ((o["lo"][0] + o["hi"][0]) / 2, (o["lo"][2] + o["hi"][2]) / 2); half = ((o["hi"][0] - o["lo"][0]) / 2, (o["hi"][2] - o["lo"][2]) / 2); h = float(o["hi"][1]); best = None
        for dx, dz in ((0, -1), (-1, 0), (1, 0), (0, 1)):
            n = int(np.ceil(h / 0.19)) - 1; reach = 0.3 * n + 0.6; e = np.array([c[0] + dx * half[0], 0, c[1] + dz * half[1]]); q = e + np.array([dx, 0, dz]) * reach; s3 = np.array([-dz, 0, dx]) * 0.7
            if min(q[0], q[2]) < 0.2 or q[0] > W - 0.2 or q[2] > D - 0.2: continue
            if not P.hits(np.minimum(e - s3, q + s3) + [0, 0.02, 0], np.maximum(e - s3, q + s3) + [0, 1.8, 0], skip=o["i"]): best = (dx, dz); break
        K.platform(b, c, half, h, mat, best, round_=o["round"]); P.pieces.append(dict(piece="platform", at=[round(c[0], 2), round(c[1], 2)], size=[round(2 * half[0], 2), round(2 * half[1], 2)], h=round(h, 2), steps=int(np.ceil(h / 0.19)) - 1 if (best or o["round"]) else 0, round=o["round"], why=f"the layout's {o['name']}"))
    return b, em

CLS = dict(wall="plaster", floor="floor", ceiling="plaster", trim="wood", timber="wood", stone="stone", metal="metal", glass="glass", door="wood", concrete="concrete", emitter="emitter", plaster_trim="plaster", platform="stone")
def shell_geometry(layout, brief_=None, hints=None, bounds=None):
    """layout (boxes.json dict) -> the room's shell as real geometry. See the top of this file for what comes back.
    bounds: {object index: (lo, hi)} the true bounds of the made objects, when known (the layout's boxes are used otherwise).
    hints: [dict(kind 'window'|'door', wall, u0, u1, v0, v1)] openings found elsewhere (for example bright patches in the paint)."""
    t0 = time.time(); B = brief_ or brief(layout)
    P = choose(layout, B, bounds, hints)
    b, em = build(P, layout); meshes = b.result()
    for e in em: e["colour"] = e.get("colour") or ("moon" if B.get("night") else "day")
    words = B.get("words", {}); groups = sorted({m["group"] for m in meshes})
    def cls_of(g, wds):
        if g in ("glass", "emitter"): return g
        for c, keys in (("metal", "steel iron alumin brass metal lead copper"), ("concrete", "concrete resin blockwork"), ("stone", "stone marble limestone brick masonry flags rock tiles terracotta"), ("wood", "oak pine walnut timber board wood joinery parquet veneer"), ("plaster", "plaster paint wallpaper")):
            if any(k in wds for k in keys.split()): return c
        return CLS.get(g, "plaster")
    mats = {g: dict(cls=cls_of(g, words.get(g, "")), words=words.get(g, "")) for g in groups}
    if "plaster_trim" in mats: mats["plaster_trim"] = dict(cls="plaster", words="white moulded plaster")
    counts = {}
    for m in meshes: counts[m["piece"]] = counts.get(m["piece"], 0) + 1
    return dict(meshes=meshes, emitters=em, materials=mats, pieces=P.pieces, counts=counts, claimed=sorted(P.claimed), notes=P.notes, room=[P.W, P.H, P.D],
                brief={k: v for k, v in B.items() if k not in ("mat", "words")}, style=B["style"], seconds=time.time() - t0,
                triangles=int(sum(len(m["F"]) for m in meshes)))

def merged(S, by="group"):
    """The same geometry as a few big meshes, one per material group (or per 'piece'): {key: dict(V, F, UV)}."""
    out = {}
    for m in S["meshes"]:
        g = out.setdefault(m[by], dict(V=[], F=[], UV=[], n=0)); g["V"].append(m["V"]); g["UV"].append(m["UV"]); g["F"].append(m["F"] + g["n"]); g["n"] += len(m["V"])
    return {k: dict(V=np.concatenate(g["V"]), F=np.concatenate(g["F"]), UV=np.concatenate(g["UV"])) for k, g in out.items()}
