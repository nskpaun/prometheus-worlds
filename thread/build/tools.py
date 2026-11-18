"""Shared by the style guide and the materials: small picture helpers, colour words, and the caller of the picture
model. thread/.venv (numpy, PIL, scipy); nothing here knows a world."""
import os, sys, json, subprocess, time, re, colorsys
import numpy as np
from PIL import Image
from scipy import ndimage
import where
GREY = np.float32([0.299, 0.587, 0.114])
def u8(a): return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
def op(p): return np.asarray(Image.open(p).convert("RGB"), np.float32)
def r16(v): return max(64, int(round(v / 16)) * 16)
def g3(a, s, mode="nearest"): return np.stack([ndimage.gaussian_filter(a[..., c], s, mode=mode) for c in range(a.shape[-1])], -1)
def resize(a, w, h, how=Image.LANCZOS): return np.asarray(u8(a).resize((int(w), int(h)), how), np.float32)

def json_in(txt):
    """The model's answer as JSON. An answer cut off by the token limit (a long list of things) is read up to its last
    whole entry: one lost tail must not lose the list."""
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    try: return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
    except ValueError:
        body = txt[txt.index("{"):]
        for cut in reversed([m.start() for m in re.finditer(r"\}\s*,\s*\{", body)]):
            try: d = json.loads(body[:cut + 1] + "]}")
            except ValueError: continue
            print("an answer was cut off: read up to its last whole entry,", sum(len(v) for v in d.values() if isinstance(v, list)), "entries", flush=True); return d
        raise
def klein(jobs_path):
    """Run a job list through the picture model (klein.py): one process, gone from memory when the list is done. -> {name: seconds}"""
    r = subprocess.run([where.PY_IMG, "-B", os.path.join(where.HERE, "klein.py"), jobs_path])
    if r.returncode: raise SystemExit(r.returncode)       # klein.py has said why, in its last line
    return times(jobs_path)
def times(jobs_path):
    ts = {}
    if os.path.exists(jobs_path + ".times.jsonl"):
        for l in open(jobs_path + ".times.jsonl"):
            d = json.loads(l)
            if "name" in d: ts[d["name"]] = d["s"]
            if "load_s" in d: ts["_load_" + d.get("load", "")] = ts.get("_load_" + d.get("load", ""), 0) + d["load_s"]
    return ts

def rgb_ok(v, default=(128, 124, 118)):
    try:
        v = [float(np.clip(x, 0, 255)) for x in v][:3]; assert len(v) == 3; return v
    except Exception: return [float(x) for x in default]
def tame(rgb, vmax=0.78, smax=0.82, vmin=0.17):
    """A model's colour as a real material's own colour in even light: paint, dye and stone are never pure or white-hot."""
    r, g, b = [x / 255 for x in rgb_ok(rgb)]; h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s > 0.5: v = min(v, vmax - 0.12)
    return [255 * x for x in colorsys.hsv_to_rgb(h, min(s, smax), float(np.clip(v, vmin, vmax)))]
def colour_word(rgb):
    r, g, b = [float(v) / 255 for v in rgb]; h, l, s = colorsys.rgb_to_hls(r, g, b)
    tone = "very dark" if l < 0.15 else "dark" if l < 0.32 else "mid-tone" if l < 0.6 else "light" if l < 0.82 else "near white"
    if s < 0.08: hue = "grey"
    elif s < 0.3 and 15 <= h * 360 < 70: hue = "warm grey-beige" if l >= 0.32 else "grey-brown"
    else:
        d = h * 360; hue = ("red" if d < 15 or d >= 345 else "brown" if d < 45 and l < 0.5 else "orange" if d < 45 else "olive-brown" if d < 70 and l < 0.5 else "yellow" if d < 70 else "green" if d < 165 else "teal" if d < 200 else "blue" if d < 260 else "purple" if d < 300 else "pink")
        if s < 0.3: hue = "greyish " + hue
    return f"{tone} {hue}"
def place_of(request):
    p = re.split(r":| with |\.|,", (request or "a room").strip())[0].strip() or "a room"; return p[0].lower() + p[1:]
def noise(shape, scale_px, seed=0, octaves=3):
    """Smooth noise in -1..1 (about), coarsest feature scale_px; made small and enlarged, so it is cheap on huge textures."""
    rng = np.random.default_rng(seed); h, w = shape; out = np.zeros(shape, np.float32); amp = 1.0; tot = 0.0
    for o in range(octaves):
        s = max(2.0, scale_px / 2 ** o); gh, gw = int(h / s) + 3, int(w / s) + 3; g = rng.standard_normal((gh, gw)).astype(np.float32)
        big = np.asarray(Image.fromarray(g).resize((int(gw * s), int(gh * s)), Image.BICUBIC))[:h, :w]
        if big.shape != (h, w): big = np.pad(big, ((0, h - big.shape[0]), (0, w - big.shape[1])), mode="edge")
        out += amp * big; tot += amp; amp *= 0.5
    return out / tot
