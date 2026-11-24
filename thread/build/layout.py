"""The layout of a world built from a sentence: 30 to 45 real things, checked, repaired and placed.
  layout.py "SENTENCE" OUT.json     ask the world-plan model (chat.py; asked again once if the list has faults) and place
                                    (thread/.venv; the model runs in its own process; placing is CPU, under a second)
What the model writes is a list of things, each with a true size, a form (sculpted or one of the plain forms of forms.py)
and a relation: on the floor against a wall / free / in a corner, ON another thing, fixed to a WALL, hung from the CEILING,
beside / in front of / under another thing. solve() turns relations into positions so that they HOLD:
  - a thing "on" another rests on its top, inside its top; too large for it, it is shrunk (or a plain platform is grown);
  - a "wall" thing whose foot is at the floor is a floor thing against that wall; flat wall things are turned flat;
  - nothing intersects (every position is the free one nearest to where the model put it); what cannot fit is dropped
    if the model added it, shrunk if the sentence named it;
  - doors and windows are not objects: they are "openings", and keep floor and wall clear;
  - the floor stays walkable in one piece (added floor things are dropped until it is);
  - hanging things clear heads (2 m) or the thing under them.
A second, short answer of the model (the sense check) says what does not belong and how full the place is, and gives
pools of small things; fill() spreads those over tops, walls and floor by rule, in proportion to the room.
The file keeps the form llm_boxes.py writes (request, room, objects[{name, prims, yaw}], motions, effects, mood, rules)
and adds, per object: "kind", "form", "attach" (floor|on|wall|ceiling), "on" (index of the thing it rests on), "wall",
"said", "look", "size" [w, h, d]; and at the top: "openings", "repairs", "dropped", "rich": 1.
"""
import sys, os, json, math, re, time, copy, zlib, random
import numpy as np
import where
import llm_boxes as LB
from prompts import PLAIN, FEELS

WALLS = ("back", "front", "left", "right")
WALL_YAW = {"back": math.pi, "front": 0.0, "left": math.pi / 2, "right": -math.pi / 2}
MAX_SCULPT = 11                                       # sculpted kinds a 20 minute build can carry (shape and paint)
VARIANT_SHAPES = 3                                    # distinct shapes among the copies of a thing the sentence counts ("5 cars")
FLAT = ("frame", "cloth", "window", "door")           # lie flat on a wall: their thinnest side is their depth
GROW = ("crate", "cylinder", "rug")                   # plain platforms: they grow to hold what the sentence puts on them
RUG_M2 = 12.0                                         # ... and never more than this: no added rug is a carpet for a hall
RUG_SHARE = 0.22                                      # an added rug covers at most this share of the floor
BODY, CELL = 0.3, 0.1                                 # a walker's radius; the grid walkability is judged on
HEAD = 2.0                                            # hanging things stay above this where one can walk
WHERE = set("left right front back rear middle centre center central corner inner outer upper lower north south east west near far row main side".split())
clamp, num = LB.clamp, LB.num

def words_of(name):
    ws = [LB.stem(w) for w in re.findall(r"[a-z]+", str(name).lower())]
    return [w for w in ws if len(w) > 2 and w not in WHERE] or ws

def read(raw):
    """The model's JSON -> (room [W, H, D], things, notes). Nothing here may fail on a badly written thing."""
    notes = []
    r = raw.get("room") if isinstance(raw.get("room"), dict) else {}
    room = [clamp(num(r.get("width"), 5), 3, 24), clamp(num(r.get("height"), 3), 2.4, 9), clamp(num(r.get("depth"), 5), 3, 24)]
    if not r:      # no room was written: one that holds the floor things four times over, between a study and a hall
        fl = sum(num(p["size"][0], 1) * num(p["size"][2], 1) * num(p.get("count"), 1) for p in raw.get("things", []) if isinstance(p, dict) and isinstance(p.get("size"), list) and len(p["size"]) == 3 and isinstance(p.get("place"), dict) and ("floor" in p["place"] or "corner" in p["place"]) and p.get("form") != "rug")
        side = clamp(math.sqrt(4 * fl), 4, 16); room = [round(side * 1.1, 1), 3.2 if side < 8 else 4.5, round(side * 0.9, 1)]; notes.append(f"no room size was written: {room[0]} x {room[2]} m from what stands on the floor")
    out = []; seen = {}
    for p in raw.get("things") if isinstance(raw.get("things"), list) else []:
        if not isinstance(p, dict) or not isinstance(p.get("name"), str): continue
        if p.get("form") not in ("window", "door") and STRUCT.search(p["name"].lower()): notes.append(f'"{p["name"]}" is part of the room itself, not a thing: left out'); continue
        z = p.get("size"); z = [num(v, None) for v in z] if isinstance(z, (list, tuple)) else []
        z = [v for v in z if v is not None and v > 0]
        if len(z) == 2: z = [z[0], z[1], z[0]]
        if len(z) != 3: z = [0.4, 0.4, 0.4]; notes.append(f'"{p["name"]}" had no usable size: 0.4 m')
        if max(z) < 0.03: notes.append(f'"{p["name"]}" is {max(z) * 1000:.0f} mm across: too small to be a thing, left out'); continue
        key = (p["name"].strip().lower(), json.dumps(p.get("place"), sort_keys=True)); seen[key] = seen.get(key, 0) + 1; seen[key[0]] = seen.get(key[0], 0) + 1
        if seen[key] > 1 and "on" not in (p.get("place") or {}): notes.append(f'"{p["name"]}" was listed twice in the same place: once is kept'); continue
        if seen[key[0]] > 3 and p.get("said") is not True: continue                  # a name the list keeps repeating: three entries are kept
        if (p.get("form") == "rug" and p.get("said") is not True or z[1] < 0.15) and z[0] * z[2] > 0.5 * room[0] * room[2]: notes.append(f'"{p["name"]}" covers the floor: it is the floor, not a thing'); continue
        form = p.get("form") if isinstance(p.get("form"), str) else "sculpted"
        form = form.strip().lower()
        if form not in PLAIN and form not in ("window", "door", "sculpted"): form = "sculpted"
        pl = p.get("place") if isinstance(p.get("place"), dict) else {}
        m = p.get("motion"); m = m.get("type") if isinstance(m, dict) else m
        n = int(clamp(num(p.get("count"), 1), 1, 12 if p.get("said") is True else 6))
        if len(out) >= 45 and p.get("said") is not True: continue          # a list that ran on: the first 45 entries are the layout
        if isinstance(pl.get("corner"), str) and pl["corner"].strip().lower() == "each": n = 1      # "each" already makes the four
        out.append({"name": re.sub(r"\s+", " ", p["name"].replace("_", " ")).strip()[:30], "form": form, "w": z[0], "h": z[1], "d": z[2], "said": p.get("said") is True,
                    "look": re.sub(r"\s+", " ", str(p.get("look") or "")).strip()[:80], "motion": m if m in ("sway", "swing", "spin", "bob") else None, "place": pl, "count": n, "n": p.get("n")})
    merged = []
    for t in out:
        m = next((u for u in merged if "ceiling" in u["place"] and "ceiling" in t["place"] and u["name"] == t["name"] and not u["said"] and not t["said"]), None)
        if m: m["count"] += t["count"]
        else: merged.append(t)
    for t in merged:
        if "ceiling" in t["place"] and not t["said"] and t["count"] > 1: t["count"] = int(clamp(t["count"], 1, max(2, round(room[0] * room[2] / 14))))
    return room, merged, notes

STRUCT = re.compile(r"\b(walls?|floor|ceiling|chimney|beams?|rafters?|skyline|vista|view)$")
HEAD_FORMS = (("table", "stool bench table workbench desk trestle"), ("drawers", "cabinet locker dresser wardrobe cupboard counter sideboard drawers fridge safe"),
              ("bookcase", "bookcase bookshelf bookshelves shelving shelves rack"), ("cylinder", "barrel drum pipe tank pillar pedestal log keg column plinth filter"),
              ("crate", "box crate trunk toolbox kit carton package chest case set"), ("bottle", "bottle vial flask decanter phial"), ("jar", "jar mug cup tin beaker tumbler glass"),
              ("bowl", "bowl dish plate basket tray mortar"), ("pot", "pot vase urn bucket kettle cauldron pail"), ("candle", "candle candlestick candlesticks candles"),
              ("books", "book books tome ledger scroll scrolls papers journal manuscript"), ("rings", "tyre tire tyres tires coil hose rope"),
              ("frame", "picture painting mirror chart poster map print"), ("cloth", "curtain curtains drape drapes banner tapestry"), ("rug", "rug mat carpet"))
def plain_for(t):
    """The plain form that describes a thing written as sculpted, by its head noun (None: it needs a shape of its own)."""
    head = re.findall(r"[a-z]+", t["name"].lower())[-1:]; pl = t["place"]
    for form, words in HEAD_FORMS:
        if head and (head[0] in words.split() or LB.stem(head[0]) in [LB.stem(v) for v in words.split()]):
            if form == "bookcase" and ("wall" in pl and "floor" not in pl or t["d"] < 0.22 or t["h"] < 0.7): return "board" if "wall" in pl else None
            if form == "frame" and "wall" not in pl: return None
            if form == "cloth" and "wall" not in pl: return None
            if form == "rug" and ("wall" in pl or "on" in pl): return None
            if form == "crate" and head[0] in ("chest", "case") and re.search(r"tool|drawer|display|glass", t["name"].lower()): return "drawers"
            if form == "table" and (t["h"] < 0.25 or min(t["w"], t["d"]) < 0.2): return None
            return form
    return None

def sense(things, check, notes, req=""):
    """The second pass: what the art director dropped goes, with what stood on it. -> the pools of filler the check named."""
    c = check if isinstance(check, dict) else {}
    gone = {int(v) for v in c.get("drop", []) if isinstance(v, (int, float)) or isinstance(v, str) and v.isdigit()} if isinstance(c.get("drop"), list) else set()
    spoken = set(LB.stems(req)); gone = {n for n in gone if not any(t["n"] == n and any(LB.same(w, v) for w in words_of(t["name"])[-1:] for v in spoken) for t in things)}      # what the sentence itself speaks of is never dropped
    added = [t for t in things if not t["said"] and t["form"] not in ("window", "door")]
    if len([t for t in added if t["n"] in gone]) > 0.45 * max(1, len(added)): notes.append(f"the sense check wanted {len([t for t in added if t['n'] in gone])} of {len(added)} added things dropped: that is not a check, it was not followed"); gone = set()
    out = [t for t in things if not (t["n"] in gone and not t["said"] and t["form"] not in ("window", "door"))]
    for t in things:
        if t not in out: notes.append(f'"{t["name"]}" does not belong in this place (sense check): dropped')
    for _ in range(3):      # what stood on, beside or under a dropped thing goes with it
        names = [u["name"].lower() for u in out]; keep = []
        for t in out:
            rel = next((t["place"][k] for k in ("on", "beside", "front_of", "under") if isinstance(t["place"].get(k), str)), None)
            if rel is not None and not t["said"] and find(rel, [u for u in out if u is not t]) is None and find(rel, things) is not None: notes.append(f'"{t["name"]}" stood by a dropped thing: dropped'); continue
            keep.append(t)
        out = keep
    def pool(key, forms, lo, hi, fallback):
        rows = []
        for p in c.get(key) if isinstance(c.get(key), list) else []:
            if not isinstance(p, dict) or not isinstance(p.get("name"), str) or not isinstance(p.get("size"), list) or len(p["size"]) != 3: continue
            z = [num(v, 0) for v in p["size"]]
            if min(z) <= 0 or STRUCT.search(p["name"].lower()) or any(words_of(t["name"])[-1:] == words_of(p["name"])[-1:] for t in things if t["said"]) or any(words_of(q["name"])[-1:] == words_of(p["name"])[-1:] for q in rows): continue
            if p.get("form") in ("sculpted", "window", "door"): continue
            f = p.get("form") if p.get("form") in forms else plain_for({"name": p["name"], "place": {"wall": 1} if key == "walls" else {}, "w": z[0], "h": z[1], "d": z[2]})
            if f not in forms: f = fallback
            k = min(1.0, hi / max(z)) * max(1.0, lo / max(z)); rows.append({"name": re.sub(r"\s+", " ", p["name"].replace("_", " ")).strip()[:30], "look": re.sub(r"\s+", " ", str(p.get("look") or "")).strip()[:80], "form": f, "size": [round(v * k, 3) for v in z]})
        return rows
    pools = {"props": pool("props", ("books", "bottle", "jar", "bowl", "pot", "candle", "crate", "cylinder", "rings"), 0.06, 0.4, "jar"),
             "storage": pool("storage", ("crate", "cylinder", "drawers", "bookcase", "table", "pot", "rings", "bowl"), 0.35, 1.8, "crate"),
             "walls": pool("walls", ("frame", "board", "cloth"), 0.3, 1.6, "frame")}
    # a check that named nothing: the list's own added plain things serve
    own = lambda ok: [{"name": t["name"], "look": t["look"], "form": t["form"], "size": [t["w"], t["h"], t["d"]]} for t in out if not t["said"] and t["form"] in PLAIN and ok(t)]
    pools["props"] = pools["props"] or own(lambda t: "on" in t["place"] and max(t["w"], t["h"], t["d"]) <= 0.4)
    pools["storage"] = pools["storage"] or own(lambda t: "floor" in t["place"] and t["form"] not in ("rug", "frame", "cloth", "board") and 0.35 <= max(t["w"], t["h"], t["d"]) <= 1.8)
    pools["walls"] = pools["walls"] or own(lambda t: "wall" in t["place"] and t["form"] in ("frame", "board"))
    return out, pools, int(clamp(num(c.get("fullness"), 3), 1, 5))

def budget(req, things, signature, notes):
    """Sculpting and painting are the slow steps: what a plain form describes is made plain, the copies of a counted thing
    share VARIANT_SHAPES shapes, and of the rest the MAX_SCULPT kinds that matter most keep a shape of their own."""
    for t in things:
        if t["form"] != "sculpted": continue
        f = plain_for(t)
        if f and (not t["said"] or f in ("table",) and re.search(r"stool", t["name"].lower()) or f in ("bookcase", "cloth", "rug", "frame")): notes.append(f'"{t["name"]}" was to be sculpted; a plain {f} describes it: made plain'); t["form"] = f
        elif not t["said"] and max(t["w"], t["h"], t["d"]) < 0.13: t["cut"] = "too small to be worth a sculpted shape"
    # the copies of a counted thing: "5 antique cars" written as five different cars
    for n, head, words, text in LB.named(req):
        if n < 3: continue
        grp = [t for t in things if t["said"] and t["form"] == "sculpted" and t["count"] == 1 and not t.get("shape_of")]
        hit = [t for t in grp if any(LB.same(head, w) for w in words_of(t["name"]))]
        if len(hit) >= n: grp = hit
        if len(grp) < n: continue
        base = max(grp, key=lambda t: t["w"] * t["h"] * t["d"]); size = np.array([base["w"], base["h"], base["d"]])
        grp = [t for t in grp if np.all(np.abs(np.array([t["w"], t["h"], t["d"]]) / size - 1) < 0.3)][:n]
        if len(grp) < n or len({t["name"] for t in grp}) <= 1: continue
        for k, t in enumerate(grp):
            t["group"] = text
            if k >= VARIANT_SHAPES: t["shape_of"] = grp[k % VARIANT_SHAPES]["name"]; t["w"], t["h"], t["d"] = (grp[k % VARIANT_SHAPES][a] for a in "whd")
        if len(grp) > VARIANT_SHAPES: notes.append(f'{n} "{text}": {VARIANT_SHAPES} different shapes ({", ".join(t["name"] for t in grp[:VARIANT_SHAPES])}); the others repeat one of them in their own colour')
    sig = {w for s_ in signature for w in words_of(s_)}
    names = {}
    for t in things:
        if t["form"] == "sculpted" and not t.get("cut") and not t.get("shape_of"): names.setdefault(t["name"].lower(), []).append(t)
    def score(ts):
        t = ts[0]; big = max(t["w"], t["h"], t["d"]); n = sum(u["count"] for u in ts)
        return (not any(u["said"] for u in ts), -(big * math.sqrt(n) * (1.5 if set(words_of(t["name"])) & sig else 1.0) * (0.6 if "on" in t["place"] and big < 0.25 else 1.0)))
    rank = sorted(names.values(), key=score)
    for ts in rank[MAX_SCULPT:]:
        if any(u["said"] for u in ts): continue
        for u in ts: u["cut"] = f"only {MAX_SCULPT} kinds can be sculpted in the time; this one matters least"
    out = []
    for t in things:
        if t.get("cut"): notes.append(f'"{t["name"]}": {t["cut"]}: left out'); continue
        out.append(t)
    names = [u["name"].lower() for u in out]
    return [t for t in out if t["said"] or not any(isinstance(t["place"].get(k), str) and find(t["place"][k], out) is None and find(t["place"][k], things) is not None for k in ("on", "beside", "front_of", "under"))]

def loose_json(txt):
    """The answer as JSON; one cut off in mid-list (the model ran out of tokens) keeps the things it finished."""
    try: return LB.llm.json_in(txt)
    except Exception: pass
    rows = []
    for line in txt.splitlines():      # the things written one object a line with nothing around them
        line = line.strip().rstrip(",")
        if line.startswith("{") and line.endswith("}"):
            try: o = json.loads(line)
            except Exception: continue
            if isinstance(o, dict) and "name" in o: rows.append(o)
    if len(rows) >= 3 and '"things"' not in txt: return {"things": rows}
    a = txt.index("{"); k = txt.rfind("}")
    for _ in range(60):
        for tail in ("]}", "}]}", "]}]}"):
            try: return json.loads(txt[a:k + 1] + tail)
            except Exception: pass
        k = txt.rfind("}", a, k)
        if k < 0: break
    raise ValueError("not JSON")

def find(name, things):
    """The thing a relation names: the exact name, else the one sharing most words."""
    if not isinstance(name, str): return None
    key = name.replace("_", " ").strip().lower()
    for t in things:
        if t["name"].lower() == key: return t
    ws = set(words_of(key)); best = max(things, key=lambda t: len(ws & set(words_of(t["name"]))), default=None)
    return best if best is not None and ws & set(words_of(best["name"])) else None

class Room:
    def __init__(self, room):
        self.W, self.H, self.D = room; self.boxes = []            # (x0, x1, y0, y1, z0, z1, owner)
    def add(self, t): self.boxes.append((*box(t), t))
    def block(self, b, owner): self.boxes.append((*b, owner))
    def remove(self, t): self.boxes = [b for b in self.boxes if b[6] is not t]
    def free(self, cx, y0, cz, ex, h, ez, gap=0.04, skip=(), wall_gap=0.0):
        """Which candidate boxes (centres cx, cz; bottoms y0; half sizes ex, ez; height h) are inside the room and touch nothing."""
        cx, y0, cz = (np.asarray(v, float) for v in np.broadcast_arrays(cx, y0, cz))
        ok = (cx - ex >= wall_gap - 1e-6) & (cx + ex <= self.W - wall_gap + 1e-6) & (cz - ez >= wall_gap - 1e-6) & (cz + ez <= self.D - wall_gap + 1e-6) & (y0 >= -1e-6) & (y0 + h <= self.H + 1e-6)
        for x0, x1, a0, a1, z0, z1, o in self.boxes:
            if any(o is s for s in skip) or (isinstance(o, str) and o in skip): continue
            ok &= ~((cx - ex < x1 + gap) & (cx + ex > x0 - gap) & (y0 < a1 - 2e-3) & (y0 + h > a0 + 2e-3) & (cz - ez < z1 + gap) & (cz + ez > z0 - gap))
        return ok

def half(t):      # half sizes on the floor of a thing turned by quarter turns
    return (t["w"] / 2, t["d"] / 2) if abs(math.cos(t["yaw"])) > 0.5 else (t["d"] / 2, t["w"] / 2)
def box(t):
    ex, ez = half(t); return (t["x"] - ex, t["x"] + ex, t["y0"], t["y0"] + t["h"], t["z"] - ez, t["z"] + ez)
def wall_len(R, wall): return R.W if wall in ("back", "front") else R.D
def on_wall(R, wall, a, d):
    """Centre (x, z) of something d deep whose back is on that wall, a metres along it."""
    return {"back": (a, R.D - d / 2), "front": (a, d / 2), "left": (d / 2, a), "right": (R.W - d / 2, a)}[wall]

def nearest(R, t, cx, y0, cz, cost, **kw):
    """Put t at the free candidate of least cost -> True, or False when none is free."""
    ex, ez = half(t); ok = R.free(cx, y0, cz, ex, t["h"], ez, **kw)
    if not ok.any(): return False
    cost = np.where(ok, cost, np.inf); k = int(np.argmin(cost))
    cx, y0, cz = (np.broadcast_to(v, cost.shape) for v in (cx, y0, cz))
    t["x"], t["y0"], t["z"] = float(cx[k]), float(y0[k]), float(cz[k]); return True

def put_wall(R, t, wall, a, others=True):
    """A floor thing against a wall, as near `a` metres along as is free; else on another wall. -> True if placed."""
    for w in [wall] + ([v for v in WALLS if v != wall] if others else []):
        L = wall_len(R, w); t["yaw"] = WALL_YAW[w]; t["y0"] = 0.0
        if t["w"] > L - 0.1: continue
        aa = np.arange(t["w"] / 2 + 0.02, L - t["w"] / 2 - 0.02 + 1e-6, 0.05)
        if not len(aa): continue
        xz = np.array([on_wall(R, w, v, t["d"]) for v in aa])
        if nearest(R, t, xz[:, 0], 0.0, xz[:, 1], np.abs(aa - (a if w == wall else L / 2))): t["wall"] = w; return True
    return False

def put_free(R, t, x, z, gap=0.08, wall_gap=0.05, reach=None):
    """A floor thing at the free spot nearest (x, z)."""
    t["y0"] = 0.0; ex, ez = half(t)
    gx, gz = np.meshgrid(np.arange(ex + wall_gap, R.W - ex - wall_gap + 1e-6, 0.1), np.arange(ez + wall_gap, R.D - ez - wall_gap + 1e-6, 0.1))
    if not gx.size: return False
    gx, gz = np.append(gx.ravel(), clamp(x, ex + wall_gap, max(ex + wall_gap, R.W - ex - wall_gap))), np.append(gz.ravel(), clamp(z, ez + wall_gap, max(ez + wall_gap, R.D - ez - wall_gap)))
    cost = np.hypot(gx - x, gz - z)
    if reach is not None: cost = np.where(cost <= reach, cost, np.inf)
    return nearest(R, t, gx, 0.0, gz, cost, gap=gap, wall_gap=wall_gap) and math.isfinite(float(np.hypot(t["x"] - x, t["z"] - z)))

def walk(R, things):
    """-> (share of the floor a walker can stand on, share of that in the largest joined region)."""
    from scipy import ndimage
    nx, nz = int(R.W / CELL), int(R.D / CELL)
    xs, zs = (np.arange(nx) + 0.5) * CELL, (np.arange(nz) + 0.5) * CELL
    free = np.ones((nx, nz), bool); free[(xs < BODY) | (xs > R.W - BODY)] = False; free[:, (zs < BODY) | (zs > R.D - BODY)] = False
    for t in things:
        if t.get("mode") not in ("floor",) or t["h"] < 0.12: continue
        x0, x1, _, _, z0, z1 = box(t)
        free[np.ix_((xs > x0 - BODY) & (xs < x1 + BODY), (zs > z0 - BODY) & (zs < z1 + BODY))] = False
    n = int(free.sum())
    if not n: return 0.0, 0.0
    lab, k = ndimage.label(free); big = int(np.bincount(lab.ravel())[1:].max())
    return n / (nx * nz), big / n

SIDE = re.compile(r"\b(?:on|along|against|at|by|to|in) the (left|right|back|front|rear|far)(?: |-)?(?:hand )?(wall|side|end)?\b")
def told(req, things, say):
    """Where the SENTENCE puts a thing wins over where the model put it: "a fireplace on the right wall" is on the right
    wall, "a desk in the middle" is in the middle. Read clause by clause; only things the clause names, and only those
    that stand on the floor or are fixed to a wall (what is "on" or "beside" another thing follows that thing)."""
    for clause in re.split(r"[,;:.]| and (?=an? |the |\d)", req.lower()):
        m = SIDE.search(clause); mid = re.search(r"\bin the (middle|centre|center)\b", clause)
        if not m and not mid: continue
        cut = (m or mid).start(); head = set(LB.stems(clause[:cut])) - {LB.stem(w) for w in LB.STOPS | LB.VAGUE}       # the thing is named before the place
        many = {}
        for t in things: many[t["name"].strip().lower()] = many.get(t["name"].strip().lower(), 0) + 1
        for t in things:
            pl = t["place"]; ws = [w for w in words_of(t["name"]) if w in head]
            if mid and not m and many[t["name"].strip().lower()] > 1: continue          # "tables in the middle": several things keep their own spread, not one pile on the centre point
            if not t["said"] or not ws or any(k in pl for k in ("on", "beside", "front_of", "under", "ceiling")): continue
            if not LB.same(words_of(t["name"])[-1], ws[-1]) and len(ws) < 2: continue                                 # the clause must name the thing itself, not its material
            if m:
                wall = {"rear": "back", "far": "back"}.get(m.group(1), m.group(1)); now = next((pl[k] for k in ("wall", "floor") if pl.get(k) in WALLS), None)
                if now != wall:
                    say(f'"{t["name"]}": the sentence says the {wall} wall, the layout said {now or "elsewhere"}: moved to the {wall} wall')
                    t["place"] = {**{k: v for k, v in pl.items() if k not in ("floor", "wall", "corner", "at")}, ("wall" if "wall" in pl and "floor" not in pl else "floor"): wall, "along": 0.5 if now is None else pl.get("along", 0.5)}
            elif pl.get("floor") != "free" or max(abs(num(LB.pair(pl.get("at"))[0], 0.5) - 0.5), abs(num(LB.pair(pl.get("at"))[1], 0.5) - 0.5)) > 0.15:
                say(f'"{t["name"]}": the sentence says the middle: moved there'); t["place"] = {"floor": "free", "at": [0.5, 0.5]}

def solve(room, things, notes):
    """Places every thing (adds x, z, y0, yaw, mode, parent). -> (placed things in order, openings, dropped names)."""
    R = Room(room); W, H, D = room; dropped, openings = [], []
    say = notes.append
    # ---- copies
    full = []
    for t in things:
        pl = t["place"]; n = t.pop("count")
        if isinstance(pl.get("corner"), str) and pl["corner"].strip().lower() == "each" or pl.get("floor") == "each":
            full += [dict(t, place={"floor": "corner", "corner": c}, copy=k) for k, c in enumerate(("back left", "back right", "front left", "front right"))]
        else: full += [dict(t, copy=k, copies=n) for k in range(n)]
    things = full
    for i, t in enumerate(things): t["id"] = i; t["kids"] = []; t["parent"] = None; t["yaw"] = 0.0
    # ---- what kind of placement each has
    for t in things:
        pl, f = t["place"], t["form"]
        if f in FLAT or (f == "board" and "wall" in pl):       # flat on a wall: the thinnest side is the depth
            dims = [t["w"], t["h"], t["d"]]; k = int(np.argmin(dims))
            if f in FLAT and k != 2:
                rest = [v for j, v in enumerate(dims) if j != k]; t["w"], t["h"], t["d"] = rest[0], rest[1], dims[k]; say(f'"{t["name"]}": size turned so that it lies flat on its wall')
            t["curtain"] = f == "cloth" and bool(re.search(r"curtain|drape", t["name"].lower())) and t["h"] >= 1.2
            t["d"] = clamp(t["d"], 0.03, 0.08) if f == "frame" else 0.28 if t["curtain"] else clamp(t["d"], 0.06, 0.2) if f == "cloth" else min(t["d"], 0.45) if f == "board" else 0.06
        if f == "rug":
            a, b, c = sorted([t["w"], t["h"], t["d"]]); t["w"], t["h"], t["d"] = (c, 0.02, b) if t["w"] >= t["d"] else (b, 0.02, c)
        corner = pl.get("corner") if isinstance(pl.get("corner"), str) else pl.get("floor") if isinstance(pl.get("floor"), str) and " " in pl["floor"] else None
        wall = next((pl[k] for k in ("wall", "floor") if isinstance(pl.get(k), str) and pl[k] in WALLS), None)
        rel = next((k for k in ("on", "beside", "front_of", "under") if isinstance(pl.get(k), str)), None)
        if f in ("window", "door"): t["mode"] = "opening"; t["wall"] = wall
        elif f == "rug": t["mode"] = "rug"
        elif rel == "on": t["mode"] = "on"
        elif rel in ("beside", "front_of"): t["mode"] = "floor"; t["how"] = rel
        elif rel == "under": t["mode"] = "floor"; t["how"] = "beside"
        elif "ceiling" in pl: t["mode"] = "ceiling"
        elif "wall" in pl and wall:
            foot = num(pl.get("height"), 1.5) - t["h"] / 2
            if f not in FLAT and f != "board" and (foot < 0.25 or t["h"] > 0.7 * H or t["w"] * t["h"] * t["d"] > 0.12):       # ... or too heavy to hang on a wall       # a fireplace "on the wall" with its foot at the floor stands on the floor
                t["mode"] = "floor"; t["how"] = "wall"; say(f'"{t["name"]}" was fixed to the wall, but its foot is at the floor or it is too large to hang: it stands on the floor against that wall')
            else: t["mode"] = "mount"
            t["wall"] = wall
        elif corner: t["mode"] = "floor"; t["how"] = "corner"; t["corner"] = corner.lower()
        elif wall: t["mode"] = "floor"; t["how"] = "wall"; t["wall"] = wall
        else: t["mode"] = "floor"; t["how"] = "free"
        if rel:
            t["parent"] = next((u for u in things if u["id"] == pl.get("_pid") and u.get("copy", 0) == 0), None) if "_pid" in pl else find(pl[rel], [u for u in things if u is not t and u["name"] != t["name"]])
            if t["parent"] is None:
                say(f'"{t["name"]}" is {rel} "{pl[rel]}", which is not in the list: it stands on the floor'); t["mode"] = "floor" if t["mode"] != "rug" else "rug"; t["how"] = "free"
    for t in things:
        if t["mode"] == "on" and t["parent"]["mode"] in ("rug", "opening", "ceiling"): t["mode"] = "floor"; t["how"] = "beside"
    for t in things:      # a relation loop (a on b on a) is cut
        seen, u = set(), t
        while u is not None and u["parent"] is not None and u["mode"] == "on":
            if u["id"] in seen: t["mode"] = "floor"; t["how"] = "free"; t["parent"] = None; break
            seen.add(u["id"]); u = u["parent"]
    for t in things:
        if t["mode"] == "on": t["parent"]["kids"].append(t)
    for t in things:      # the furniture the sentence itself names is worth sculpting, even where a plain table would do
        if t["form"] == "table" and t["said"]: t["form"] = "sculpted"
    for t in things:      # a low, wide thing that carries others is a platform: made plain, with a true flat top, not sculpted
        if t["form"] == "sculpted" and t["mode"] == "floor" and t["kids"] and t["h"] <= 0.6 and t["h"] < 0.4 * min(t["w"], t["d"]):
            t["form"] = "crate"; say(f'"{t["name"]}" is low, wide and carries things: made as a plain platform')
    for t in things:      # a wall shelf that holds things is a shelf: deep enough to hold them, and thin
        if t["form"] == "board" and t["mode"] == "mount" and t["kids"] and (t["d"] < 0.18 or t["h"] > 0.25):
            was = (t["h"], t["d"]); t["d"] = clamp(max(t["d"], t["h"] if t["h"] < 0.45 else 0.0), 0.2, 0.4); t["h"] = min(was[0], 0.16) if was[0] <= 0.25 else 0.16
            say(f'"{t["name"]}" holds things but was written {was[1]:.2f} m deep and {was[0]:.2f} m high: now a shelf {t["d"]:.2f} m deep')
    # ---- sizes that cannot be
    for t in things:
        m = t["mode"]; was = (t["w"], t["h"], t["d"])
        if m == "floor":
            k = min(1.0, 0.9 * W / max(t["w"], 1e-3), 0.9 * D / max(t["d"], 1e-3), (H - 0.02) / t["h"])
            if not t["said"]: k = min(k, math.sqrt(0.2 * W * D / (t["w"] * t["d"])))          # nothing the sentence did not ask for takes a fifth of the floor
            if k < 1: t["w"], t["h"], t["d"] = t["w"] * k, t["h"] * k, t["d"] * k
        if m == "mount":
            if t["d"] > t["w"]: t["w"], t["d"] = t["d"], t["w"]          # the long side runs along the wall
            if t["d"] > 0.5: k = 0.5 / t["d"]; t["w"], t["h"], t["d"] = t["w"] * k, t["h"] * k, 0.5
            k = min(1.0, 0.8 * wall_len(R, t["wall"]) / t["w"], (H - 0.2) / t["h"]); t["w"], t["h"] = t["w"] * k, t["h"] * k
        if m == "ceiling":
            k = min(1.0, (H - 1.0) / t["h"], 0.5 * W / t["w"], 0.5 * D / t["d"]); t["w"], t["h"], t["d"] = t["w"] * k, t["h"] * k, t["d"] * k
        if m == "rug":
            t["w"], t["d"] = min(t["w"], W - 0.2), min(t["d"], D - 0.2)
            if not t["said"]: k = min(1.0, math.sqrt(min(RUG_M2, RUG_SHARE * W * D) / (t["w"] * t["d"]))); t["w"], t["d"] = t["w"] * k, t["d"] * k
        if max(abs(a / b - 1) for a, b in zip((t["w"], t["h"], t["d"]), was)) > 0.1 and m != "rug":
            say(f'"{t["name"]}": {was[0]:.2f} x {was[1]:.2f} x {was[2]:.2f} m cannot be in this room: now {t["w"]:.2f} x {t["h"]:.2f} x {t["d"]:.2f}')
    def fit_on(t):       # a thing on another must fit its top; children first, so a platform grown for one is known to those below
        p = t["parent"]; inset = 0.0 if p["form"] == "board" else 0.03
        top = lambda: ((p["w"] - 2 * inset) * (0.71 if p["form"] == "cylinder" else 1), (p["d"] - 2 * inset) * (0.71 if p["form"] == "cylinder" else 1))
        s = max(min(top()[0] / t["w"], top()[1] / t["d"]), min(top()[0] / t["d"], top()[1] / t["w"]))
        low = p["h"] <= 0.6 and p["h"] < 0.4 * min(p["w"], p["d"])          # low and wide: a platform, a stage, a plinth, whatever it is called
        if s < 1 and (p["form"] in GROW or low) and p["mode"] in ("floor", "rug") and (t["said"] or not p["said"]):
            k = 1.41 if p["form"] == "cylinder" else 1
            nw, nd = max(p["w"], (t["w"] + 0.5) * k), max(p["d"], (t["d"] + 0.5) * k)
            if nw <= 0.8 * W and nd <= 0.8 * D:
                say(f'"{p["name"]}" ({p["w"]:.1f} x {p["d"]:.1f} m) was too small to hold "{t["name"]}" ({t["w"]:.1f} x {t["d"]:.1f} m): grown to {nw:.1f} x {nd:.1f} m'); p["w"], p["d"] = nw, nd; s = 1
        if s < 0.9 and t["said"] and p["mode"] == "floor" and max(t["w"], t["d"]) > max(p["w"], p["d"]):
            # whatever stands on the floor and carries a named thing longer than itself is a platform, whatever shape and height it was
            # written with: the named thing keeps its true size and the platform is made plain, low and large enough to hold it
            nw, nd = max(p["w"], t["w"] + 0.5), max(p["d"], t["d"] + 0.5)
            if nw <= 0.8 * W and nd <= 0.8 * D:
                say(f'"{p["name"]}" ({p["w"]:.1f} x {p["h"]:.1f} x {p["d"]:.1f} m) is to hold "{t["name"]}" ({t["w"]:.1f} x {t["d"]:.1f} m), which is larger than it: made as a plain platform {nw:.1f} x {nd:.1f} m, {min(p["h"], 0.6):.1f} m high')
                p["w"], p["d"], p["h"], p["form"] = nw, nd, min(p["h"], 0.6), "crate"; s = 1
        s = min(s, (H - 0.05 - p["h"] - (0 if p["mode"] != "mount" else 1.0)) / t["h"]) if t["h"] > 0 else s
        if s < 1:
            s = max(s, 0.02)
            if s < 0.9: say(f'"{t["name"]}" ({t["w"]:.2f} x {t["h"]:.2f} x {t["d"]:.2f} m) could not rest on "{p["name"]}" ({p["w"]:.2f} x {p["d"]:.2f} m top): shrunk to {s * t["w"]:.2f} x {s * t["h"]:.2f} x {s * t["d"]:.2f}')
            def scale(u, k):
                u["w"], u["h"], u["d"] = u["w"] * k, u["h"] * k, u["d"] * k
                for v in u["kids"]: scale(v, k)
            scale(t, s * 0.98)
    def depth(t):
        n = 0
        while t["parent"] is not None and t["mode"] == "on": t = t["parent"]; n += 1
        return n
    for t in [t for t in things if t["mode"] == "on"]:      # nothing the sentence did not ask for goes on top of a thing taller than a person
        p = t["parent"]
        if p["h"] > 1.7 and p["mode"] == "floor" and not t["said"]:
            t["skip"] = f'it was put on top of "{p["name"]}", which is {p["h"]:.1f} m tall'; p["kids"].remove(t)
    for p in things:      # a top carries what it really would: about one thing to 0.15 m2, the named ones first
        ks = [k for k in p["kids"] if not k.get("skip")]; cap = int(p["w"] * p["d"] / 0.15) + 2
        for k in sorted(ks, key=lambda k: (not k["said"], k["id"]))[cap:]:
            if not k["said"]: k["skip"] = f'"{p["name"]}" already carries {cap} things'; p["kids"].remove(k)
    for t in sorted([t for t in things if t["mode"] == "on" and not t.get("skip")], key=depth): fit_on(t)
    # ---- order: what the sentence names first, then large before small
    pr = lambda t: (not t["said"], bool(t.get("extra")), -(t["w"] * t["d"] * max(t["h"], 0.3)), t["id"])
    def drop(t, why):
        if t.get("gone"): return
        t["gone"] = True; dropped.append({"name": t["name"], "why": why, "said": t["said"], **({"extra": 1} if t.get("extra") else {})}); R.remove(t)
        if t["parent"] is not None and t in t["parent"]["kids"]: t["parent"]["kids"].remove(t)
        for k in list(t["kids"]): drop(k, f'what it stood on ("{t["name"]}") was dropped')
    for t in things:
        if t.get("skip"): t["kids"] and None; drop(t, t["skip"])
    # ---- doors and windows: openings, kept clear
    for t in sorted([t for t in things if t["mode"] == "opening"], key=pr):
        wall = t["wall"] or max(WALLS, key=lambda w: wall_len(R, w)); L = wall_len(R, wall); door = t["form"] == "door"
        t["w"] = min(t["w"], L - 0.6 if t["said"] else 0.6 * L); t["h"] = min(t["h"], H - 0.15)
        a = clamp(num(t["place"].get("along"), 0.5), 0, 1) * L
        y0 = 0.0 if door else clamp(num(t["place"].get("height"), 1.5) - t["h"] / 2, min(0.3, H - t["h"] - 0.1), H - t["h"] - 0.1)
        t["yaw"] = WALL_YAW[wall]; t["d"] = 0.25
        aa = np.arange(t["w"] / 2 + 0.25, L - t["w"] / 2 - 0.25 + 1e-6, 0.05)
        xz = np.array([on_wall(R, wall, v, 0.25) for v in aa]) if len(aa) else np.zeros((0, 2))
        if not len(aa) or not nearest(R, t, xz[:, 0], y0, xz[:, 1], np.abs(aa - a), gap=0.15): drop(t, "no free stretch of wall"); continue
        R.add(t); ex, ez = half(t)
        o = {"kind": t["form"], "name": t["name"], "wall": wall, "along": round(float(t["x"] if wall in ("back", "front") else t["z"]), 3), "width": round(t["w"], 3), "height": round(t["h"], 3), "sill": round(y0, 3), "look": t["look"], "said": t["said"]}
        openings.append(o)
        if door:       # the floor in front of a door stays clear
            n = {"back": (0, -1), "front": (0, 1), "left": (1, 0), "right": (-1, 0)}[wall]; dx, dz = n[0] * 0.9, n[1] * 0.9
            R.block((min(t["x"] - ex, t["x"] - ex + dx), max(t["x"] + ex, t["x"] + ex + dx), 0.0, HEAD, min(t["z"] - ez, t["z"] - ez + dz), max(t["z"] + ez, t["z"] + ez + dz)), "doorway")
    # ---- the floor
    def put(t):
        how = t.get("how", "free"); pl = t["place"]
        if how == "wall":
            L = wall_len(R, t["wall"]); a = clamp(num(pl.get("along"), 0.5), 0, 1) * L
            n = t.get("copies", 1)
            if n > 1: a = clamp(a, n * (t["w"] + 0.04) / 2, max(n * (t["w"] + 0.04) / 2, L - n * (t["w"] + 0.04) / 2)) + (t["copy"] - (n - 1) / 2) * (t["w"] + 0.04)       # a row, side by side
            if t.get("extra"): return put_wall(R, t, t["wall"], a, others=False) and abs((t["x"] if t["wall"] in ("back", "front") else t["z"]) - a) < 0.5
            return put_wall(R, t, t["wall"], a)
        if how == "corner":
            c = t["corner"]; back, left = "back" in c, "left" in c
            if not ("left" in c or "right" in c): left = t["copy"] % 2 == 0
            if not ("back" in c or "front" in c): back = True
            t["yaw"] = math.pi if back else 0.0; ex, ez = half(t); m = 0.12 if max(ex, ez) < 0.6 else 0.35
            return put_free(R, t, ex + m if left else W - ex - m, D - ez - m if back else ez + m, wall_gap=0.04)
        if how in ("beside", "front_of") and t["parent"] is not None:
            p = t["parent"]
            if p.get("gone") or "x" not in p: return put_free(R, t, W / 2, D / 2)
            if p["mode"] == "on" or p["mode"] == "mount" or p["mode"] == "ceiling":       # beside a lamp that hangs: under it, near
                return put_free(R, t, p["x"], p["z"])
            pex, pez = half(p); cs, sn = math.cos(p["yaw"]), math.sin(p["yaw"]); f = (sn, cs); side = (cs, -sn)       # the parent's front and its right
            if how == "front_of":
                g = p["d"] / 2 + t["d"] / 2 + 0.25; t["yaw"] = p["yaw"] + math.pi if not p.get("wall") else p["yaw"]
                return put_free(R, t, p["x"] + f[0] * g, p["z"] + f[1] * g, reach=2.5) or put_free(R, t, p["x"], p["z"])
            if p.get("wall") and p.get("how") == "wall":      # beside a thing on a wall: on the same wall, next to it
                L = wall_len(R, p["wall"]); a = p["x"] if p["wall"] in ("back", "front") else p["z"]; g = p["w"] / 2 + t["w"] / 2 + 0.25
                best = None
                for s in (1, -1):
                    c = dict(t); c["yaw"] = p["yaw"]
                    if put_wall(R, c, p["wall"], a + s * g, others=False):
                        e = abs((c["x"] if p["wall"] in ("back", "front") else c["z"]) - (a + s * g))
                        if best is None or e < best[0]: best = (e, c)
                if best and best[0] < 1.5: t.update({k: best[1][k] for k in ("x", "z", "y0", "yaw", "wall")}); return True
            t["yaw"] = p["yaw"]; g = p["w"] / 2 + t["w"] / 2 + 0.2; best = None
            for s in (1, -1):
                c = dict(t)
                if put_free(R, c, p["x"] + side[0] * g * s, p["z"] + side[1] * g * s, reach=2.5):
                    e = math.hypot(c["x"] - (p["x"] + side[0] * g * s), c["z"] - (p["z"] + side[1] * g * s))
                    if best is None or e < best[0]: best = (e, c)
            if best: t.update({k: best[1][k] for k in ("x", "z", "y0")}); return True
            return put_free(R, t, p["x"], p["z"])
        u, v = LB.pair(pl.get("at")); n = t.get("copies", 1); u, v = num(u, 0.5), num(v, 0.5)
        if u > 1 or v > 1: u, v = u / W, v / D          # written in metres, not as fractions
        x, z = clamp(u, 0, 1) * W, clamp(v, 0, 1) * D
        if n > 1: x += (t["copy"] - (n - 1) / 2) * (t["w"] + 0.35)
        return put_free(R, t, x, z)
    floor = sorted([t for t in things if t["mode"] == "floor"], key=pr)
    # walls and corners before the open floor, and a thing after the thing it stands beside
    order = [t for t in floor if t["said"]] + [t for t in floor if not t["said"] and not t.get("extra") and t.get("how") in ("wall", "corner")] + [t for t in floor if not t["said"] and not t.get("extra") and t.get("how") not in ("wall", "corner")] + [t for t in floor if t.get("extra")]
    waiting = list(order)
    for _ in range(4):
        later = []
        for t in waiting:
            p = t["parent"]
            if t.get("how") in ("beside", "front_of") and p is not None and not p.get("gone") and "x" not in p and p["mode"] == "floor": later.append(t); continue
            ok = put(t)
            for _ in range(3):
                if ok or not t["said"]: break
                t["w"], t["h"], t["d"] = t["w"] * 0.8, t["h"] * 0.8, t["d"] * 0.8; say(f'"{t["name"]}" did not fit anywhere: made a fifth smaller'); ok = put(t)
                for k in t["kids"]: fit_on(k)
            if ok: R.add(t)
            else: drop(t, "no free place on the floor")
        waiting = later
        if not waiting: break
    for t in waiting: drop(t, "what it stands beside was never placed")
    # ---- walkable in one piece
    alive = lambda: [t for t in things if not t.get("gone") and "x" in t]
    for _ in range(40):
        share, joined = walk(R, alive())
        if joined >= 0.97 and share >= 0.33: break
        cands = [t for t in alive() if t["mode"] == "floor" and not t["said"] and not any(k["said"] for k in t["kids"])]
        if not cands: say(f"the floor is tight: {share:.0%} can be walked, {joined:.0%} of it joined, and only named things are left"); break
        # the added thing whose removal joins the most floor; ties: free-standing before those on walls, small last
        def gain(t):
            rest = [u for u in alive() if u is not t]; s, j = walk(R, rest); return (round(j, 2), round(s, 2), t.get("how") not in ("wall", "corner"), -t["w"] * t["d"])
        t = max(cands, key=gain); drop(t, f"it cut the floor in two or filled it ({share:.0%} walkable, {joined:.0%} joined)")
    # ---- rugs: lie under things, take no space
    for t in [t for t in things if t["mode"] == "rug"]:
        p = t["parent"]; u, v = LB.pair(t["place"].get("at"))
        if p is not None and not p.get("gone") and "x" in p:
            t["yaw"] = p["yaw"] if abs(math.cos(p["yaw"])) > 0.5 else p["yaw"]; x, z = p["x"], p["z"]
            if p["mode"] != "floor": t["yaw"] = 0.0
            else:
                pex, pez = half(p); tex, tez = half(t)
                k = max(1.0, (pex + 0.4) / tex, (pez + 0.4) / tez) if p["h"] > 0.3 else 1.0
                if not t["said"] and (t["w"] * t["d"] * k * k > 1.05 * min(RUG_M2, RUG_SHARE * W * D) or p["form"] in GROW and p["kids"]): drop(t, f'a rug under "{p["name"]}" would cover the floor of the room'); continue
                t["w"], t["d"] = min(t["w"] * k, W - 0.3), min(t["d"] * k, D - 0.3)
        else: x, z = clamp(num(u, 0.5), 0, 1) * W, clamp(num(v, 0.5), 0, 1) * D
        ex, ez = half(t); t["x"], t["z"], t["y0"] = clamp(x, ex + 0.1, max(ex + 0.1, W - ex - 0.1)), clamp(z, ez + 0.1, max(ez + 0.1, D - ez - 0.1)), 0.0
    # ---- on the walls
    for t in sorted([t for t in things if t["mode"] == "mount" and not t.get("gone")], key=pr):
        done = False; skip = ("doorway",) + (tuple(o for *_, o in R.boxes if isinstance(o, dict) and o["form"] == "window") if t["form"] == "cloth" else ())
        for wall in [t["wall"]] + [w for w in WALLS if w != t["wall"]]:
            L = wall_len(R, wall); t["yaw"] = WALL_YAW[wall]; n = t.get("copies", 1)
            a = clamp(num(t["place"].get("along"), 0.5), 0, 1) * L
            if n > 1: a = L * (t["copy"] + 1) / (n + 1) if abs(a - L / 2) < 0.15 * L else a + (t["copy"] - (n - 1) / 2) * max(2.2 * t["w"], 0.8)
            win = None
            if t.get("curtain") and wall == t["wall"]:      # a curtain hangs at a window when its wall has one near, drawn apart to show it; heavy ones reach from ceiling to floor
                near = [o for o in openings if o["kind"] == "window" and o["wall"] == wall and not o.get("curtained") and abs(o["along"] - a) < max(2.5, 0.3 * L)]
                win = min(near, key=lambda o: abs(o["along"] - a), default=None)
                heavy = t["said"] or t["h"] >= 0.6 * H or re.search(r"heavy|thick|velvet|floor", (t["name"] + " " + t["look"]).lower())
                if win: a = win["along"]; t["w"] = min(max(t["w"], win["width"] + 1.0), L - 0.2); top_ = min(H - 0.06, win["sill"] + win["height"] + 0.25)
                else: top_ = H - 0.06 if heavy else min(H - 0.06, num(t["place"].get("height"), 1.5) + t["h"] / 2)
                foot = 0.03 if heavy or not win else max(0.03, win["sill"] - 0.3)
                if heavy: top_ = H - 0.06
                t["h"] = top_ - foot; t["place"] = dict(t["place"], height=foot + t["h"] / 2)
            if t["w"] > L - 0.1: continue
            y = num(t["place"].get("height"), 1.5) - t["h"] / 2
            lo = 0.02 if t["form"] == "cloth" else min(0.3, max(0.02, H - t["h"] - 0.05))
            if t.get("extra") and wall != t["wall"]: break
            aa, yy = np.meshgrid(np.arange(t["w"] / 2 + 0.05, L - t["w"] / 2 - 0.05 + 1e-6, 0.05), np.arange(lo, H - t["h"] - 0.05 + 1e-6, 0.05))
            if not aa.size: continue
            aa, yy = aa.ravel(), yy.ravel(); xz = np.array([on_wall(R, wall, v, t["d"]) for v in aa])
            if nearest(R, t, xz[:, 0], yy, xz[:, 1], np.abs(aa - a) + 2.0 * np.abs(yy - y) + (0 if wall == t["wall"] else 1e3), gap=0.05, skip=skip):
                if wall != t["wall"]: say(f'"{t["name"]}" had no room on the {t["wall"]} wall: it is on the {wall} wall')
                if win is not None and abs((t["x"] if wall in ("back", "front") else t["z"]) - win["along"]) < 0.1: win["curtained"] = True; t["opts"] = {"gap": round(min(0.8, 0.85 * win["width"] / t["w"]), 3)}
                t["wall"] = wall; done = True; break
        if done: R.add(t)
        else: drop(t, "no free stretch of wall")
    # ---- on top of things (parents before what stands on them)
    def top_of(p):
        pex, pez = half(p); k = 0.71 if p["form"] == "cylinder" else 1.0; inset = 0.0 if p["form"] == "board" else 0.03
        return pex * k - inset, pez * k - inset
    for t in sorted([t for t in things if t["mode"] == "on" and not t.get("gone")], key=lambda t: (depth(t), pr(t))):
        p = t["parent"]
        if p.get("gone") or "x" not in p: drop(t, f'what it stood on ("{p["name"]}") was dropped'); continue
        ok = False
        for attempt in range(4):
            tx, tz = top_of(p); best = None
            for turn in (0, 1):
                t["yaw"] = p["yaw"] + turn * math.pi / 2; ex, ez = half(t)
                if ex > tx + 1e-6 or ez > tz + 1e-6: continue
                gx, gz = np.meshgrid(np.linspace(p["x"] - (tx - ex), p["x"] + (tx - ex), max(1, int(2 * (tx - ex) / 0.03) + 1)), np.linspace(p["z"] - (tz - ez), p["z"] + (tz - ez), max(1, int(2 * (tz - ez) / 0.03) + 1)))
                gx, gz = gx.ravel(), gz.ravel(); sib = [k for k in p["kids"] if k is not t and "x" in k and not k.get("gone")]
                if sib:      # spread over the top: as far from what is there already as the top allows (to 0.35 m), then towards the middle
                    dist = np.min([np.hypot(gx - k["x"], gz - k["z"]) for k in sib], 0); cost = -np.minimum(dist, 0.35) + 0.05 * np.hypot(gx - p["x"], gz - p["z"])
                else:        # the first thing: the middle if the sentence put it there, else a third of the way along
                    off = 0.0 if t["said"] else (tx - ex) * 0.5; cs, sn = math.cos(p["yaw"]), math.sin(p["yaw"])
                    cost = np.hypot(gx - (p["x"] + cs * off), gz - (p["z"] - sn * off))
                c = dict(t)
                if nearest(R, c, gx, p["y0"] + p["h"], gz, cost, gap=0.015, skip=(p,)): best = c; break
            if best: t.update({k: best[k] for k in ("x", "z", "y0", "yaw")}); ok = True; break
            if not t["said"] and attempt >= 1: break
            for u in [t] + t["kids"]: u["w"], u["h"], u["d"] = u["w"] * 0.8, u["h"] * 0.8, u["d"] * 0.8
        if ok: R.add(t)
        else: drop(t, f'no room left on "{p["name"]}"')
    # ---- from the ceiling
    for t in sorted([t for t in things if t["mode"] == "ceiling" and not t.get("gone")], key=pr):
        pl = t["place"]; u, v = LB.pair(pl.get("ceiling")); n = t.get("copies", 1); u, v = num(u, 0.5), num(v, 0.5)
        if u > 1 or v > 1: u, v = u / W, v / D
        x, z = clamp(u, 0, 1) * W, clamp(v, 0, 1) * D
        if n > 1:      # copies spread over the ceiling in a grid
            cols = int(math.ceil(math.sqrt(n * W / D))); rows = int(math.ceil(n / cols)); x, z = W * (t["copy"] % cols + 0.5) / cols, D * (t["copy"] // cols + 0.5) / rows
        ex, ez = half(t); drop_m = clamp(num(pl.get("drop"), 0.6), 0.0, H - t["h"])
        gx, gz = np.meshgrid(np.arange(ex + 0.2, W - ex - 0.2 + 1e-6, 0.1), np.arange(ez + 0.2, D - ez - 0.2 + 1e-6, 0.1)); gx, gz = np.append(gx.ravel(), x), np.append(gz.ravel(), z)
        # how high the thing below each spot reaches: over open floor it clears heads, over furniture it clears that
        under = np.zeros(len(gx))
        for x0, x1, a0, a1, z0, z1, o in R.boxes:
            if isinstance(o, dict) and o["mode"] in ("mount", "ceiling", "opening"): continue
            m = (gx - ex < x1 + 0.1) & (gx + ex > x0 - 0.1) & (gz - ez < z1 + 0.1) & (gz + ez > z0 - 0.1); under = np.where(m, np.maximum(under, a1), under)
        need = np.where(under < 0.5, HEAD, under + 0.3); y0 = np.minimum(H - drop_m - t["h"], H - t["h"] - 0.0); y0 = np.maximum(y0, need)
        fits = y0 + t["h"] <= H + 1e-6
        if not fits.any():
            k = (H - HEAD) / t["h"]; t["w"], t["h"], t["d"] = t["w"] * k, t["h"] * k, t["d"] * k; say(f'"{t["name"]}" hung too low for this ceiling: shrunk to {t["h"]:.2f} m high')
            ex, ez = half(t); y0 = np.maximum(H - drop_m - t["h"], need); fits = y0 + t["h"] <= H + 1e-6
        cost = np.where(fits, np.hypot(gx - x, gz - z), np.inf)
        if fits.any() and nearest(R, t, gx, y0, gz, cost, gap=0.1):
            if H - t["y0"] - t["h"] < drop_m - 0.05: say(f'"{t["name"]}" was hung {drop_m:.1f} m down, into heads or the thing under it: now {H - t["y0"] - t["h"]:.1f} m down')
            R.add(t)
        else: drop(t, "no free place under the ceiling")
    out = [t for t in things if not t.get("gone") and "x" in t and t["mode"] != "opening"]
    for o in openings: o.pop("curtained", None)
    return out, openings, dropped, len(things)

def check_(room, placed):
    """What still does not hold after solve(): [text]. Empty is the aim."""
    W, H, D = room; bad = []
    for i, a in enumerate(placed):
        x0, x1, y0, y1, z0, z1 = box(a)
        if x0 < -1e-3 or x1 > W + 1e-3 or z0 < -1e-3 or z1 > D + 1e-3 or y0 < -1e-3 or y1 > H + 1e-3: bad.append(f'"{a["name"]}" pokes out of the room')
        if a["mode"] == "on":
            p = a["parent"]; px0, px1, _, py1, pz0, pz1 = box(p)
            if abs(y0 - py1) > 1e-3: bad.append(f'"{a["name"]}" does not rest on the top of "{p["name"]}"')
            if x0 < px0 - 1e-3 or x1 > px1 + 1e-3 or z0 < pz0 - 1e-3 or z1 > pz1 + 1e-3: bad.append(f'"{a["name"]}" overhangs "{p["name"]}"')
        if a["mode"] == "rug": continue
        for b in placed[i + 1:]:
            if b["mode"] == "rug": continue
            u = box(b)
            if min(x1, u[1]) - max(x0, u[0]) > 5e-3 and min(y1, u[3]) - max(y0, u[2]) > 5e-3 and min(z1, u[5]) - max(z0, u[4]) > 5e-3: bad.append(f'"{a["name"]}" and "{b["name"]}" intersect')
    return bad

def judge(req, raw):
    """Named things the layout lacks, by llm_boxes' own reading of the sentence: [(weight, text)]."""
    things = [t for t in raw.get("things", []) if isinstance(t, dict) and isinstance(t.get("name"), str)]
    lab = [(LB.stems(t["name"]), int(clamp(num(t.get("count"), 1), 1, 40)) * (4 if isinstance(t.get("place"), dict) and t["place"].get("corner") == "each" else 1)) for t in things]
    eff = [LB.stems(e.get("label")) for e in (raw.get("effects") if isinstance(raw.get("effects"), list) else []) if isinstance(e, dict)]
    have = {w for ws, _ in lab for w in ws} | {w for ws in eff for w in ws}; out = []
    for n, head, words, text in LB.named(req):
        made = sum(c for ws, c in lab if any(LB.same(head, w) for w in ws))
        if n >= 3 and made < n:      # "5 cars" written as five different cars under their own names: said things of one size
            said = [t for t in things if t.get("said") is True and isinstance(t.get("size"), list) and len(t["size"]) == 3 and all(isinstance(v, (int, float)) and v > 0 for v in t["size"])]
            ref = next((t for t in said if any(LB.same(head, w) for w in LB.stems(t["name"]))), None) or max(said, key=lambda t: t["size"][0] * t["size"][1] * t["size"][2], default=None)
            if ref: made = max(made, sum(int(clamp(num(t.get("count"), 1), 1, 40)) for t in said if all(abs(a / b - 1) < 0.3 for a, b in zip(t["size"], ref["size"]))))
        if n > 1 and made and made != n and not any(LB.same(head, w) for ws in eff for w in ws): out.append((3 if made < n else 1, f'the description has {n} "{text}" but {made} were listed'))
        elif not any(LB.same(w, v) for w in words for v in have): out.append((2, f'nothing was listed for "{text}"'))
    return out

def kinds(placed):
    """Things that can share one shape: the same words and form, the same proportions (within 15%). Adds t["kind"]."""
    groups = []
    for t in placed:
        ws = words_of(t.get("shape_of") or t["name"]); size = np.array([t["w"], t["h"], t["d"]])
        if t.get("extra") and t["form"] != "sculpted": ws = words_of(t["name"]) + [t["form"]]
        for g in groups:
            r = size / g["size"]
            if g["words"] == ws and g["form"] == t["form"] and np.all(np.abs(r / r.mean() - 1) < 0.15): t["kind"] = g["name"]; break
        else:
            base = "_".join(ws)[:24].strip("_") or "thing"; name, n = base, 2
            while any(g["name"] == name for g in groups): name, n = f"{base}_{n}", n + 1
            groups.append({"name": name, "words": ws, "form": t["form"], "size": size}); t["kind"] = name

def world(req, raw, room, placed, openings, dropped, notes):
    kinds(placed); idx = {t["id"]: i for i, t in enumerate(placed)}; objs, motions = [], []
    for i, t in enumerate(placed):
        c = [t["x"], t["y0"] + t["h"] / 2, t["z"]]
        prim = {"shape": "cyl", "c": c, "r": min(t["w"], t["d"]) / 2, "hy": t["h"] / 2} if t["form"] == "cylinder" else {"shape": "box", "c": c, "h": [t["w"] / 2, t["h"] / 2, t["d"] / 2], "yaw": t["yaw"]}
        o = {"name": t["name"], "prims": [prim], "yaw": t["yaw"], "kind": t["kind"], "form": t["form"], "attach": {"floor": "floor", "rug": "floor", "on": "on", "mount": "wall", "ceiling": "ceiling"}[t["mode"]],
             "said": t["said"], "look": t["look"], "size": [round(t["w"], 3), round(t["h"], 3), round(t["d"], 3)]}
        if t["mode"] == "on": o["on"] = idx[t["parent"]["id"]]
        if t.get("opts"): o["opts"] = t["opts"]
        if t.get("extra"): o["fill"] = t["extra"]
        if t.get("group"): o["group"] = t["group"]
        if t.get("wall") and (t["mode"] == "mount" or t.get("how") == "wall"): o["wall"] = t["wall"]
        objs.append(o)
        m = t["motion"]; H = room[1]
        if m and len(motions) < 40:
            q = LB.motion({"motion": {"type": m}})
            if m == "swing":
                if t["mode"] != "ceiling": q = None
                else:
                    q["pivot"] = H
                    if not LB.hang(q, t["y0"], t["h"], t["d"] / 2, H): q = None
            if m == "sway" and t["form"] not in ("cloth", "sculpted"): q = None
            if m == "bob" and t["mode"] == "floor": q = None
            if q: motions.append({"i": i, "type": q.pop("type"), **{k: v for k, v in q.items() if k not in ("sound", "rope")}, "phase": round(len(motions) * 0.6180339887 % 1, 3), "sound": q["sound"]})
    out = {"request": req, "room": room, "objects": objs}
    eff = [e for e in (LB.effect(e, room) for e in (raw.get("effects") if isinstance(raw.get("effects"), list) else [])) if e][:4]
    m = raw.get("mood")
    if isinstance(m, dict): out["mood"] = {**({"feel": m["feel"]} if m.get("feel") in FEELS else {}), "energy": clamp(num(m.get("energy"), 0.5), 0, 1), **({"time": m["time"]} if m.get("time") in ("day", "dusk", "night") else {})}
    if raw.get("setting") == "outdoor":
        if os.environ.get("WORLD_SETTING") == "indoor": notes.append("the plan called this an outdoor place, but Indoors was chosen: built as a room")
        else: out["setting"] = "outdoor"
    if any(q["type"] == "sway" for q in motions) or any(e["type"] == "drift" for e in eff): out["wind"] = {"strength": 0.2, "from": 270}
    if motions: out["motions"] = motions
    if eff: out["effects"] = eff
    rugs = [{"name": t["name"], "c": [t["x"], t["z"]], "h": list(half(t)), "yaw": 0.0} for t in placed if t["mode"] == "rug"]
    if rugs: out["grounds"] = rugs[:8]
    out.update(openings=openings, repairs=notes, dropped=dropped, rules=2, rich=1)
    return out

def fill(req, room, placed, openings, pools, fullness, n0):
    """What a real place of this fullness has and the list lacks, by rule: small things on every bare top, storage along bare
    stretches of wall, pictures and shelves on bare wall above. Bigger rooms have longer walls and get more. -> extra things."""
    W, H, D = room; rnd = random.Random(zlib.crc32(req.encode())); out = []
    def thing(p, place, k=None, why=""):
        k = k or rnd.uniform(0.88, 1.12); w, h, d = (v * k for v in p["size"])
        out.append({"name": p["name"], "form": p["form"], "w": w, "h": h, "d": d, "said": False, "look": p["look"], "motion": None, "place": place, "count": 1, "n": None, "extra": why}); return n0 + len(out) - 1
    cyc = {k: (rnd.sample(v, len(v)), [0]) for k, v in pools.items()}; used = {}
    per = 2 * (W + D)      # no thing is spread more often than this (it grows with the walls' length): better a barer wall than seven barrels
    most = {"props": 2 + fullness + int(per / 20), "storage": max(2, round(per / 13)) + (fullness >= 4), "walls": max(2, round(per / 16))}
    def nxt(key, ok=lambda p: True):
        items, at = cyc[key]
        for _ in range(len(items)):
            p = items[at[0] % len(items)]; at[0] += 1
            if ok(p) and used.get((key, p["name"]), 0) < most[key]: used[(key, p["name"])] = used.get((key, p["name"]), 0) + 1; return p
        return None
    SURF = set("desk table bench workbench counter chest cabinet sideboard dresser mantel mantelpiece fireplace shelf sill trunk bar altar stand drawers crate barrel".split())
    kids = lambda t: [k for k in t["kids"] if not k.get("gone") and "x" in k]
    # ---- tops
    def dress(t_id, w, d, form, have=0, top=1.0):
        if not pools["props"]: return
        want = clamp(int(w / 0.3), 1, 4) if form == "board" else clamp(int(w * d / (0.34 - 0.05 * fullness)), 1, 3 + fullness)
        if top > 1.45 and form != "board" or form == "cylinder": want = min(want, 2)
        for _ in range(max(0, want - have)):
            p = nxt("props", lambda p: max(p["size"][0], p["size"][2]) <= 0.55 * min(w, d) + 0.04 and (form != "board" or p["size"][1] <= 0.35))
            if p: thing(p, {"on": "", "_pid": t_id}, why="top")
    for t in placed:
        top = t["y0"] + t["h"]; head = re.findall(r"[a-z]+", t["name"].lower())[-1:]
        if t["mode"] not in ("floor", "mount") or t["form"] in ("rug", "frame", "cloth", "bookcase", "rings", "pot", "bowl", "bottle", "jar", "candle", "books"): continue
        if t["form"] == "sculpted" and not (head and head[0] in SURF): continue
        if t["form"] == "cylinder" and not 0.5 <= top <= 1.2 or t["form"] == "crate" and (max(t["w"], t["d"]) > 1.6 or any(k["said"] for k in kids(t))): continue
        if not (0.3 <= top <= (1.9 if t["form"] == "board" else 1.5)) or min(t["w"], t["d"]) < 0.16 or t["form"] == "board" and (t["d"] < 0.15 or t["h"] > 0.3): continue
        dress(t["id"], t["w"], t["d"], t["form"], len(kids(t)), top)
    # ---- along the walls
    def free_runs(L, taken, least):
        ev = sorted((max(0.0, a), min(L, b)) for a, b in taken if b > 0 and a < L); runs, x = [], 0.0
        for a, b in ev:
            if a - x >= least: runs.append([x, a])
            x = max(x, b)
        if L - x >= least: runs.append([x, L])
        return runs, 1 - sum(b - a for a, b in runs) / L
    def span(t, wall):      # (from, to) along the wall, and how far from it
        x0, x1, _, _, z0, z1 = box(t)
        return ((x0, x1), {"back": D - z1, "front": z0}[wall]) if wall in ("back", "front") else ((z0, z1), {"left": x0, "right": W - x1}[wall])
    for wall in WALLS:
        L = wall_len(R_ := Room(room), wall); low, high, under = [(0, 0.3), (L - 0.3, L)], [(0, 0.35), (L - 0.35, L)], []
        for o in openings:
            if o["wall"] != wall: continue
            m = 0.45 if o["kind"] == "door" else 0.1
            high.append((o["along"] - o["width"] / 2 - 0.25, o["along"] + o["width"] / 2 + 0.25))
            if o["kind"] == "door" or o["sill"] < 1.0: low.append((o["along"] - o["width"] / 2 - m, o["along"] + o["width"] / 2 + m))
        for t in placed:
            if t["mode"] == "rug": continue
            (a, b), off = span(t, wall)
            if t["mode"] in ("floor", "on") and off < 1.1 and t["y0"] < 1.0: low.append((a - 0.3, b + 0.3))
            if off < 0.7 and t["y0"] + t["h"] > 1.7 and t["y0"] < 2.3: high.append((a - 0.3, b + 0.3))
            elif off < 1.2 and t["mode"] in ("floor", "on") and t["y0"] + t["h"] > 0.9: under.append((a, b, t["y0"] + t["h"]))
        # storage on the floor: until this share of the wall's length is furnished
        aim = 0.45 + 0.1 * fullness
        for _ in range(int(L / 1.0)):
            runs, share = free_runs(L, low, 0.75)
            if share >= aim or not runs or not pools["storage"]: break
            run = max(runs, key=lambda r: r[1] - r[0]); room_ = run[1] - run[0]
            p = nxt("storage", lambda p: p["size"][0] * 1.12 + 0.1 <= room_ and p["size"][2] <= 0.25 * (D if wall in ("back", "front") else W))
            if not p: break
            k = rnd.uniform(0.88, 1.12); w = p["size"][0] * k; a = run[0] + w / 2 + (rnd.uniform(0.0, min(0.6, room_ - w)) if run[0] > 0.31 else rnd.uniform(0.15, max(0.15, min(1.0, room_ - w))))
            i = thing(p, {"floor": wall, "along": a / L}, k, why="wall storage"); low.append((a - w / 2 - rnd.uniform(0.05, 0.5), a + w / 2 + rnd.uniform(0.05, 0.5)))
            if p["size"][1] * k > 1.7: high.append((a - w / 2 - 0.3, a + w / 2 + 0.3))
            else: under.append((a - w / 2, a + w / 2, p["size"][1] * k + 0.3))
            if p["size"][1] * k > 1.7: pass
            elif p["form"] in ("crate", "drawers", "table", "cylinder") and 0.3 <= p["size"][1] * k <= 1.3 and min(p["size"][0], p["size"][2]) * k >= 0.3: dress(i, w, p["size"][2] * k, p["form"], 0)
            if p["form"] in ("crate", "cylinder", "rings") and room_ - w > 1.2 and rnd.random() < 0.2 * fullness and used.get(("storage", p["name"]), 0) < most["storage"]:      # things of a kind stand in groups
                w2 = p["size"][0] * rnd.uniform(0.8, 1.0); a2 = a + w / 2 + 0.06 + w2 / 2
                if a2 + w2 / 2 < run[1] - 0.1: used[("storage", p["name"])] += 1; thing(p, {"floor": wall, "along": a2 / L}, w2 / p["size"][0], why="wall storage"); low.append((a2 - w2 / 2, a2 + w2 / 2 + 0.3))
        # the wall above: no bare stretch longer than a couple of metres
        aim = 0.35 + 0.1 * fullness; big = clamp(H / 3.2, 1.0, 1.5)
        for _ in range(int(L / 1.2)):
            runs, share = free_runs(L, high, 1.1)
            if share >= aim and max((r[1] - r[0] for r in runs), default=0) < 4.5 - 0.5 * fullness or not runs or not pools["walls"]: break
            run = max(runs, key=lambda r: r[1] - r[0]); room_ = run[1] - run[0]
            p = nxt("walls", lambda p: p["size"][0] * big * 1.1 + 0.3 <= room_)
            if not p: break
            k = big * rnd.uniform(0.9, 1.1); w = p["size"][0] * k; a = (run[0] + run[1]) / 2 if room_ < 2.5 * w + 1.0 else run[0] + 0.5 + w / 2 + rnd.uniform(0, 0.5)
            hgt = min(H - 0.3 - p["size"][1] * k / 2, (1.55 if p["form"] != "board" else 1.45) * (1.0 + 0.5 * (big - 1)) + rnd.uniform(-0.08, 0.08))
            below = max((top for a0, a1, top in under if a0 < a + w / 2 and a1 > a - w / 2), default=0.0); hgt = min(H - 0.15 - p["size"][1] * k / 2, max(hgt, below + 0.35 + p["size"][1] * k / 2))
            i = thing(p, {"wall": wall, "along": a / L, "height": hgt}, k, why="wall"); high.append((a - w / 2 - 0.45, a + w / 2 + 0.45))
            if p["form"] == "board" and p["size"][2] * k >= 0.15 and p["size"][1] * k <= 0.25: dress(i, w, max(0.2, p["size"][2] * k), "board", 0)
    return out

SEAT_TABLE = re.compile(r"\b(table|desk)s?\b"); NOT_DINING = re.compile(r"\b(coffee|side|bedside|console|end|night|dressing|pool|work|drafting|billiard|changing|ironing|picnic|operating|potting)\b")
SEAT = re.compile(r"\bchairs?\b"); NOT_SEAT = re.compile(r"\b(arm|easy|rocking|high|lounge|deck|wheel|swivel|office|throne|recliner)")

def seat(room, things, say):
    """Chairs go round tables, as people would sit: the free-standing tables that are sat at (not coffee or side tables)
    are spread over the floor, the free-standing chairs dealt out among them (four to a table at least 70 cm across,
    two to a smaller one, one to a desk), each set on its own side of its table, and turned to face it after placing
    (see face()). A model that writes "chairs" as a grid of rows would otherwise make a lecture hall of a cafe.
    Chairs beyond what the tables seat are left out. Nothing is done unless there are both."""
    W, _, D = room; free = lambda t: t["place"].get("floor") == "free" and not any(k in t["place"] for k in ("on", "beside", "front_of", "under"))
    tables = [t for t in things if free(t) and SEAT_TABLE.search(t["name"].lower()) and not NOT_DINING.search(t["name"].lower()) and max(t["w"], t["d"]) <= 2.6]
    chairs = [t for t in things if free(t) and SEAT.search(t["name"].lower()) and not NOT_SEAT.search(t["name"].lower())]
    if not tables or not chairs: return things
    for t in [t for t in tables if t.get("count", 1) > 1]:                # "four tables" written once with a count: four tables, each with its own chairs
        k = things.index(t); copies = [dict(t, count=1, place=dict(t["place"])) for _ in range(int(t["count"]))]
        things = things[:k] + copies + things[k + 1:]; i = tables.index(t); tables = tables[:i] + copies + tables[i + 1:]
    for t in [t for t in chairs if t.get("count", 1) > 1]:                # and "twelve chairs" written once: twelve chairs
        k = things.index(t); copies = [dict(t, count=1, place=dict(t["place"])) for _ in range(int(t["count"]))]
        things = things[:k] + copies + things[k + 1:]; i = chairs.index(t); chairs = chairs[:i] + copies + chairs[i + 1:]
    n = len(tables)
    if n > 1:                                                         # spread the tables over the open floor, in rows across the room's longer side
        cols = max(1, min(n, round(math.sqrt(n * W / D)))); rows = math.ceil(n / cols)
        for k, t in enumerate(sorted(tables, key=lambda t: (num(LB.pair(t["place"].get("at"))[1], 0.5), num(LB.pair(t["place"].get("at"))[0], 0.5)))):
            r, c = divmod(k, cols); left = n - r * cols; cc = min(cols, left)
            t["place"] = {**t["place"], "at": [round(0.22 + 0.56 * (c + 0.5) / cc, 3), round(0.25 + 0.5 * (r + 0.5) / rows, 3)]}
    for k, t in enumerate(tables): t["seat_key"] = k + 1
    cap = [1 if "desk" in t["name"].lower() else 4 if min(t["w"], t["d"]) >= 0.7 else 2 for t in tables]
    seats = [(i, j) for j in range(max(cap)) for i in range(n) if j < cap[i]]                  # round the tables in turn: each gets its first chair before any gets a second
    keep = []
    for ch, (i, j) in zip(chairs, seats):
        tb = tables[i]; u, v = (num(x, 0.5) for x in LB.pair(tb["place"].get("at")))
        side = [(0, 1), (0, -1), (1, 0), (-1, 0)][j] if cap[i] != 2 else [(0, 1), (0, -1)][j]
        gx = (tb["w"] / 2 + ch["d"] / 2 + 0.06) / W; gz = (tb["d"] / 2 + ch["d"] / 2 + 0.06) / D
        ch["place"] = {**ch["place"], "at": [round(u + side[0] * gx, 3), round(v + side[1] * gz, 3)]}; ch["seat_of"] = tb["seat_key"]; keep.append(ch)
    extra = [c for c in chairs if c not in keep]
    if extra: say(f"{len(extra)} chairs more than {n} table{'s' if n > 1 else ''} seat{'' if n > 1 else 's'}: left out")
    say(f"{len(keep)} chairs set round {n} table{'s' if n > 1 else ''}, facing them")
    return [t for t in things if t not in extra]

def face(placed):
    """Each chair dealt to a table turned to face it (a yaw of 0 faces +z)."""
    tables = {t["seat_key"]: t for t in placed if t.get("seat_key") and not t.get("gone") and "x" in t and t.get("copy", 0) == 0}
    for t in placed:
        tb = tables.get(t.get("seat_of"))
        if tb is None or "x" not in t: continue
        dx, dz = tb["x"] - t["x"], tb["z"] - t["z"]
        if dx * dx + dz * dz > 1e-4: t["yaw"] = math.atan2(dx, dz)

def place(req, raw, check=None):
    """The model's list (and its own check of it) -> the world. Two passes of the same solver: the list alone, then the list
    with what fill() adds; the list's own things come first both times, so the filler never moves them."""
    room, things, notes = read(raw); told(req, things, notes.append)
    things, pools, fullness = sense(things, check, notes, req)
    things = budget(req, things, raw.get("signature") if isinstance(raw.get("signature"), list) else [], notes)
    things = seat(room, things, notes.append)
    placed, openings, dropped, n0 = solve(room, copy.deepcopy(things), [])
    extra = fill(req, room, placed, openings, pools, fullness, n0)
    placed, openings, dropped, _ = solve(room, things + extra, notes); face(placed)
    dropped = [d for d in dropped if not d.get("extra")]
    w = world(req, raw, room, placed, openings, dropped, notes); w["unsolved"] = check_(room, placed)
    if isinstance(raw.get("place"), str): w["place"] = raw["place"][:160]
    w["fullness"] = fullness; w["counts"] = summary(w)
    return w

def summary(w):
    """THE count, used by every report, sheet and file: objects are the entries of boxes.json "objects" (cords of hanging
    things, added when meshes are placed, are counted apart as "cords"); a kind is a (kind, form) pair."""
    o = [x for x in w["objects"] if x.get("part_of") is None]; by = lambda k, v: sum(x.get(k) == v for x in o)
    ks = {}
    for x in o: ks.setdefault((x["kind"], x["form"]), 0); ks[(x["kind"], x["form"])] += 1
    return {"room": w["room"], "floor_m2": round(w["room"][0] * w["room"][2], 1), "objects": len(o), "cords": len(w["objects"]) - len(o), "kinds": len(ks), "sculpted_kinds": sum(f == "sculpted" for _, f in ks), "plain_kinds": sum(f != "sculpted" for _, f in ks),
            "sculpted_objects": by("form", "sculpted"), "from_the_list": sum(not x.get("fill") for x in o), "filled_by_rule": sum(bool(x.get("fill")) for x in o),
            "on_floor": by("attach", "floor"), "on_things": by("attach", "on"), "on_walls": by("attach", "wall"), "from_ceiling": by("attach", "ceiling"),
            "openings": len(w["openings"]), "repairs": len(w["repairs"]), "dropped": len(w["dropped"]), "dropped_said": sum(d["said"] for d in w["dropped"]), "unsolved": len(w["unsolved"])}

def ask(c, req):
    """Both turns with the model `c` (a chat.Chat): the list (asked again once if a named thing is missing), then its check. -> (raw, check, log)"""
    from prompts import PROMPT, CHECK, POOLS
    note = ""; best = None; log = []
    for k in range(2):
        m = [{"role": "user", "content": PROMPT + req + note}]; x = c.say(m, max_tokens=5600)
        try: raw = loose_json(x["text"])
        except Exception: log.append("the answer was not JSON; asked again"); continue
        fs = judge(req, raw); wgt = sum(n for n, _ in fs); log.append(f"list, try {k + 1}: {x['s']} s, {x['gen_tokens']} tokens; faults: {[f for _, f in fs]}")
        if best is None or wgt < best[0]: best = (wgt, raw, m, x["text"])
        if wgt < 2: break
        note = "\n\nAn earlier list for this description had these faults. Do not repeat them: " + "; ".join(f for _, f in fs[:5]) + "."
    if best is None: raise SystemExit("The world-plan model did not write a usable list of things for this description. Try again, or word it differently.")
    chk = checks(c, best[2] + [{"role": "assistant", "content": best[3]}], log); return best[1], chk, log

def checks(c, m, log):
    """The two short questions after the list: what does not belong (and how full the place is), then what else to spread."""
    from prompts import CHECK, POOLS
    chk = {}
    for q, n in ((CHECK, 500), (POOLS, 1500)):
        y = c.say(m + [{"role": "user", "content": q}], max_tokens=n); log.append(f"{'check' if q is CHECK else 'filler pools'}: {y['s']} s, {y['gen_tokens']} tokens")
        try: chk.update(loose_json(y["text"]))
        except Exception: log.append("  that answer was not JSON: left unused")
        m = m + [{"role": "user", "content": q}, {"role": "assistant", "content": y["text"]}]
    return chk

def show(w, req, raw):
    print(json.dumps(w["counts"]))
    for k in ("repairs", "dropped", "unsolved"):
        for x in w[k]: print(f"  {k}:", x if isinstance(x, str) else f'{x["name"]}: {x["why"]}')
    for n, f in judge(req, raw): print("  fault:", n, f)

if __name__ == "__main__":      # layout.py "SENTENCE" OUT.json
    import chat
    req, out = sys.argv[1], sys.argv[2]; t0 = time.time()
    with chat.Chat() as c: raw, chk, log = ask(c, req); load = c.load_s
    for l in log: print(l)
    t1 = time.time(); w = place(req, raw, chk); json.dump(w, open(out, "w"), indent=1)
    json.dump({"request": req, "raw": raw, "check": chk}, open(out.replace(".json", ".raw.json"), "w"), indent=1)
    show(w, req, raw); print(f"layout took {time.time() - t0:.1f} s (model load {load} s, placing {time.time() - t1:.2f} s)", flush=True)
