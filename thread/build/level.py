"""One built world read back for the stages that follow its layout: the layout, the shape plan, every object's placed
mesh and the room's own geometry (arch.py) on the true bounds of those meshes. Frame: room metres as boxes.json (x
across, y up, z deep). Also the colour words a look may name, and a thing recoloured to one. thread/.venv, CPU only."""
import os, json
import numpy as np

def layout(world): return json.load(open(f"{world}/boxes.json"))
def plan(world):
    p = f"{world}/shapes/plan.json"; return json.load(open(p)) if os.path.exists(p) else {"plain": {}, "kinds": {}}
def placed(world, i):
    z = np.load(f"{world}/shapes/placed/{i:03d}.npz"); return {k: z[k] for k in z.files}

def face_air(V, F):
    """Faces wound so normals point away from the mesh's own inside. Sculpted and plain meshes are closed and already
    outward; this only fixes a wholly inverted mesh, by its signed volume."""
    V = np.asarray(V, np.float64); a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]; vol = float((a * np.cross(b, c)).sum() / 6)
    return F[:, ::-1].copy() if vol < 0 else F

_shell = {}
def shell(world, b=None):
    """The room's geometry for this layout, built on the TRUE bounds of the made objects. Kept per process: the style
    guide, the materials and the assembler each build it once (a few tenths of a second)."""
    import arch
    if world in _shell: return _shell[world]
    b = b or layout(world); bounds = {}
    for i in range(len(b["objects"])):
        try: V = placed(world, i)["V"]; bounds[i] = (V.min(0), V.max(0))
        except OSError: pass
    _shell[world] = arch.shell_geometry(b, bounds=bounds); return _shell[world]

COLOURS = {"red": (0, 0.85, 0.62), "crimson": (350, 0.85, 0.55), "scarlet": (5, 0.9, 0.7), "maroon": (345, 0.7, 0.35), "burgundy": (345, 0.7, 0.35), "orange": (28, 0.9, 0.85), "yellow": (50, 0.85, 0.88),
           "gold": (44, 0.75, 0.8), "cream": (48, 0.25, 0.9), "green": (130, 0.7, 0.45), "olive": (75, 0.6, 0.45), "teal": (178, 0.7, 0.5), "turquoise": (175, 0.7, 0.7), "cyan": (185, 0.7, 0.8), "blue": (220, 0.8, 0.6),
           "navy": (225, 0.8, 0.3), "purple": (275, 0.65, 0.5), "violet": (270, 0.6, 0.6), "pink": (335, 0.45, 0.9), "brown": (25, 0.6, 0.4), "black": (0, 0.0, 0.08), "white": (0, 0.0, 0.92), "grey": (0, 0.0, 0.5),
           "gray": (0, 0.0, 0.5), "silver": (0, 0.0, 0.72)}
def colour_word(text):
    """The first plain colour word of a look -> (word, (hue degrees, saturation, value)) or (None, None)."""
    for w in str(text).lower().replace(",", " ").replace("-", " ").split():
        if w in COLOURS: return w, COLOURS[w]
    return None, None

def rgb2hsv(c):
    c = np.asarray(c, np.float32); mx, mn = c.max(-1), c.min(-1); d = mx - mn; h = np.zeros_like(mx); m = d > 1e-6; r, g, b = c[..., 0], c[..., 1], c[..., 2]
    h = np.where(m & (mx == r), ((g - b) / np.maximum(d, 1e-6)) % 6, h); h = np.where(m & (mx == g) & (mx != r), (b - r) / np.maximum(d, 1e-6) + 2, h); h = np.where(m & (mx == b) & (mx != r) & (mx != g), (r - g) / np.maximum(d, 1e-6) + 4, h)
    return np.stack([h * 60 % 360, np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0), mx], -1)
def hsv2rgb(c):
    h, s, v = c[..., 0] / 60, c[..., 1], c[..., 2]; i = np.floor(h).astype(int) % 6; f = h - np.floor(h); p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return np.stack([np.choose(i, [v, q, p, p, t, v]), np.choose(i, [t, v, v, q, p, p]), np.choose(i, [p, p, t, v, v, q])], -1)

def recolour(rgb, target):
    """A painted thing whose main colour is one hue, given another colour word's (hue, sat, value): only the pixels of its dominant saturated hue
    change (a red car's body turns blue; chrome, tyres and glass stay). rgb (n, 3) 0..1 -> (n, 3)."""
    hsv = rgb2hsv(rgb); sat = hsv[:, 1] > 0.28
    if sat.sum() < 0.03 * len(rgb): return rgb
    hist = np.bincount((hsv[sat, 0] / 10).astype(int) % 36, weights=hsv[sat, 1] * hsv[sat, 2], minlength=36); h0 = (np.argmax(np.convolve(np.r_[hist[-1:], hist, hist[:1]], [1, 2, 1], "valid")) + 0.5) * 10
    dh = (hsv[:, 0] - h0 + 180) % 360 - 180; w = np.clip((38 - np.abs(dh)) / 14, 0, 1) * np.clip((hsv[:, 1] - 0.15) / 0.2, 0, 1); body = (np.abs(dh) < 25) & sat
    th, ts, tv = target; out = hsv.copy(); v0 = float(np.median(hsv[body, 2])) if body.any() else 0.5; s0 = float(np.median(hsv[body, 1])) if body.any() else 0.6
    out[:, 0] = (th + dh) % 360; out[:, 1] = np.clip(hsv[:, 1] * ts / max(s0, 1e-3), 0, 1); out[:, 2] = np.clip(hsv[:, 2] * tv / max(v0, 1e-3), 0, 1)
    return (rgb * (1 - w[:, None]) + hsv2rgb(out) * w[:, None]).astype(np.float32)
