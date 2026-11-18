"""A material class per painted point of a sculpted kind, so the light can give sheen and glow where it belongs.
Classes (paint.MATS): 0 plain, 1 glass, 2 metal (bright bare metal), 3 glow (gives light), 4 gloss (shiny paint, polished wood, leather).
Two sources: the words of the kind's look say what CAN be there; the paint's own colours say where. If the picture model
made a material map of the sheet (paint.py maps, full quality) it decides glass, metal and glow inside what the words allow. numpy only."""
import numpy as np
GLASS = ("glass", "window", "windscreen", "windshield", "pane", "panes", "mirror", "crystal", "bottle", "jar", "lens")
VEHICLE = ("car", "sedan", "coupe", "truck", "van", "bus", "jeep", "taxi", "limousine", "automobile", "roadster", "convertible", "wagon", "hatchback", "pickup")
METAL = ("chrome", "steel", "brass", "silver", "gold", "golden", "copper", "bronze", "aluminium", "aluminum", "nickel", "polished metal", "shiny metal", "tin", "pewter")
DULL_METAL = ("iron", "metal", "blackened", "cast", "wrought")
GLOW = ("flame", "flames", "fire", "candle", "bulb", "lamp", "lantern", "sconce", "light", "torch", "neon", "glowing", "lit", "chandelier", "brazier", "fireplace", "screen")
GLOSS = ("polished", "lacquer", "lacquered", "leather", "varnished", "glossy", "shiny", "enamel", "porcelain", "ceramic", "glazed", "plastic", "painted", "mahogany", "piano")
def words_of(Kd):
    return (Kd["name"] + " " + Kd["look"]).lower().replace(",", " ").replace("-", " ").replace("_", " ").split()
def hsv(c):
    mx, mn = c.max(-1), c.min(-1); d = mx - mn; r, g, b = c[..., 0], c[..., 1], c[..., 2]; h = np.zeros_like(mx); m = d > 1e-6
    h = np.where(m & (mx == r), ((g - b) / np.maximum(d, 1e-6)) % 6, h); h = np.where(m & (mx == g) & (mx != r), (b - r) / np.maximum(d, 1e-6) + 2, h); h = np.where(m & (mx == b) & (mx != r) & (mx != g), (r - g) / np.maximum(d, 1e-6) + 4, h)
    return h * 60 % 360, np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0), mx
def allowed(Kd):
    w = words_of(Kd); has = lambda L: any(x in w or (" " in x and x in " ".join(w)) for x in L); veh = has(VEHICLE)
    if any(x in w for x in ("extinguisher", "alarm", "hydrant", "engine", "truck", "escape", "exit", "switch")): w = [x for x in w if x not in ("fire", "light")]          # a fire extinguisher is not a fire
    return dict(glass=has(GLASS) or veh, metal=has(METAL) or veh, dull=has(DULL_METAL), glow=has(GLOW), gloss=has(GLOSS) or veh, vehicle=veh, name_metal=any(x in Kd["name"].split() for x in METAL))
def classify(Kd, meta, P, N, rgb, how, mask=None):
    """-> mat (uint8 per point), shares. mask: optional (S, 3) colours of a Klein material map sampled at the points (blue glass, yellow metal, red glow, black other)."""
    A = allowed(Kd); h, s, v = hsv(np.clip(rgb.astype(np.float32), 0, 1)); mat = np.zeros(len(P), np.uint8)
    if A["gloss"]: mat[:] = 4
    grey_bright = (s < 0.16) & (v > 0.62); warm = ((h < 62) | (h > 345)); golden = (h > 22) & (h < 62) & (s > 0.28) & (v > 0.45)
    if mask is not None:
        mr, mg, mb = mask[:, 0], mask[:, 1], mask[:, 2]
        if A["glass"]: mat[(mb > 0.5) & (mr < 0.35) & (mg < 0.45)] = 1
        if A["name_metal"]: mat[:] = 2
        if A["metal"] or A["dull"]: mat[(mr > 0.55) & (mg > 0.5) & (mb < 0.35)] = 2
        if A["glow"]: mat[(mr > 0.55) & (mg < 0.35) & (mb < 0.35)] = 3
    else:
        if A["name_metal"]: mat[:] = 2                                    # a brass globe, a steel tool: metal all over
        elif A["metal"]: mat[grey_bright | (golden & any(x in words_of(Kd) for x in ("brass", "gold", "golden", "copper", "bronze")))] = 2
        if A["glow"]:
            hot = (warm & (s > 0.35) & (v > 0.86)) | ((v > 0.93) & (s < 0.35)); 
            if 0 < hot.mean() < 0.35: mat[hot] = 3                         # a flame or a bulb is a small bright part; a thing bright all over is not alight
    return mat, {m: round(float((mat == i).mean()), 3) for i, m in enumerate(["plain", "glass", "metal", "glow", "gloss"])}
