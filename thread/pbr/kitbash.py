"""The CC0 kitbash library (RFC pillar 3): Poly Haven's models, kept on this Mac, for the small and middling things of a
place (books, mugs, vases, plants, rocks, crates, lamps, chairs) so the 3D-shape model's minutes go to the few things that
matter most. Every model is CC0 with real-world size, PBR textures and tags.
  kitbash.py fetch [DIR]          download the library (once, at setup; 1K textures; about 3 GB) into DIR (default ext/cc0-kitbash)
  kitbash.py match "NAME" W H D   the best match for a thing of that name and size (metres), for checking by hand
At build time nothing goes to the network: match() reads DIR/index.json."""
import sys, os, re, json, time, math, urllib.request, concurrent.futures as cf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.environ.get("KITBASH_DIR") or os.path.join(HERE, "..", "..", "ext", "cc0-kitbash"))
API = "https://api.polyhaven.com"
UA = {"User-Agent": "prometheus-worlds-setup"}
RES = "1k"

def _get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r: return r.read()
        except Exception:
            if k == tries - 1: raise
            time.sleep(2 * (k + 1))

def fetch(root=ROOT, workers=6):
    """Every Poly Haven model as glTF with 1K textures, and index.json: id, name, categories, tags, size (metres)."""
    os.makedirs(root, exist_ok=True)
    assets = json.loads(_get(f"{API}/assets?t=models")); index = []
    def one(aid):
        meta = assets[aid]; d = os.path.join(root, aid); gl = os.path.join(d, f"{aid}.gltf")
        if not os.path.exists(gl):
            files = json.loads(_get(f"{API}/files/{aid}"))
            g = (files.get("gltf") or {}).get(RES, {}).get("gltf")
            if not g: return None
            os.makedirs(d, exist_ok=True)
            for rel, f in (g.get("include") or {}).items():
                p = os.path.join(d, rel)
                if not os.path.exists(p): os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(_get(f["url"]))
            open(gl + ".tmp", "wb").write(_get(g["url"])); os.replace(gl + ".tmp", gl)
        dims = [v / 1000.0 for v in (meta.get("dimensions") or [])][:3]
        return {"id": aid, "name": meta.get("name", aid), "categories": meta.get("categories", []), "tags": meta.get("tags", []),
                "description": meta.get("description", ""), "size": dims, "path": os.path.relpath(gl, root)}
    with cf.ThreadPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(lambda a: (a, _safe(one, a)), sorted(assets))):
            aid, got = r
            if got: index.append(got)
            if k % 25 == 0: print(f"{k + 1} of {len(assets)} models", flush=True)
    json.dump(index, open(os.path.join(root, "index.json"), "w"), indent=1)
    print(f"{len(index)} CC0 models in {root}", flush=True); return index

def _safe(f, a):
    try: return f(a)
    except Exception as e: print("skipped", a, type(e).__name__, str(e)[:80], flush=True); return None

# ---- matching
STOP = set("a an the of with and in on for to from its small large big tall short old new little pair set some few".split())
SYN = {"mug": ["cup"], "cup": ["mug"], "lamp": ["light", "lantern"], "lantern": ["lamp"], "plant": ["potted", "fern", "succulent"], "pot": ["planter", "vase"],
       "vase": ["pot"], "book": ["books", "encyclopedia"], "books": ["book"], "stool": ["chair"], "rock": ["boulder", "stone"], "stone": ["rock", "boulder"],
       "boulder": ["rock"], "tree": ["sapling"], "bush": ["shrub", "plant"], "shrub": ["bush", "plant"], "crate": ["box"], "box": ["crate"],
       "bottle": ["jug", "jerrycan"], "clock": ["watch"], "frame": ["picture"], "picture": ["frame"], "candle": ["candleholders"], "bin": ["can", "trash"]}

def words(s):
    w = [x for x in re.findall(r"[a-z]+", s.lower().replace("_", " ")) if x not in STOP and len(x) > 1]
    return [x[:-1] if x.endswith("s") and len(x) > 3 and not x.endswith("ss") else x for x in w]

ODD = set(words("hazard hazardous explosive warning military army rifle weapon gun ammo rusted rusty dirty broken damaged toxic flammable radioactive grenade"))
_INDEX = None
def index(root=ROOT):
    global _INDEX
    if _INDEX is None:
        p = os.path.join(root, "index.json"); _INDEX = json.load(open(p)) if os.path.exists(p) else []
        for m in _INDEX: m["_w"] = (set(words(m["name"])), set(words(" ".join(m["tags"] + m["categories"]))), set(words(m["description"])))
    return _INDEX

def match(name, size=None, root=ROOT, least=2.0):
    """The library model that best fits a thing: by its words (the head word counts most: "brass candle holder" is a holder),
    then by how well its own proportions fit the wanted size. -> {id, path (absolute), size, score} or None."""
    want = words(name)
    if not want: return None
    head = want[-1]; best = None
    for m in index(root):
        nm, tg, ds = m["_w"]; s = 0.0
        for w in want:
            alt = {w, *SYN.get(w, [])}
            k = 3.0 if w == head else 1.0
            if alt & nm: s += 2.0 * k
            elif alt & tg: s += 1.2 * k
            elif alt & ds: s += 0.4 * k
        if head not in nm and not ({head, *SYN.get(head, [])} & (nm | tg)): s *= 0.35          # the thing itself, not something that goes with it
        odd = (nm | tg) & ODD
        if odd - set(want): s -= 5.0                                                           # a hazard sign, a rifle, rust: only when asked for
        if size and len(m["size"]) == 3 and all(v > 0 for v in m["size"]) and all(v > 0 for v in size):
            a = sorted(m["size"]); b = sorted(size)
            s -= 1.5 * sum(abs(math.log(x / y)) for x, y in zip([v / a[-1] for v in a], [v / b[-1] for v in b]))      # proportions, not scale: it is scaled to fit
            s -= 0.4 * abs(math.log(a[-1] / b[-1]))
        if best is None or s > best[1]: best = (m, s)
    if best is None or best[1] < least: return None
    m = best[0]; return {"id": m["id"], "path": os.path.join(root, m["path"]), "size": m["size"], "score": round(best[1], 2), "name": m["name"]}

if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "fetch": fetch(os.path.abspath(a[1]) if len(a) > 1 else ROOT)
    elif a and a[0] == "match": print(json.dumps(match(a[1], [float(x) for x in a[2:5]] if len(a) >= 5 else None), indent=1))
    else: print(__doc__)
