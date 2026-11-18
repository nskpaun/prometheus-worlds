"""How much fine detail the rendered views keep, and whether they still show the paint at all.
image_scores.py RENDER_DIR [--ref REF_DIR] [--paint PAINT_DIR] [--poses FILE] [--sheet OUT.jpg] [--crops OUT.jpg]   -> one JSON line
A render's kind (centre, close, floor, far) is the start of its file name, or comes from FILE (the list evals/cams.py prints).
  sharp_KIND   mean sharpness of that kind's views; sharp_wall: of the four walls seen straight on from 0.8 m
               (close_x0, close_x1, close_z0, close_z1), the views the blur complaint is about. Means, not medians: a
               median over twelve mixed views does not move when only the worst four get better.
Sharpness alone is easy to fool: speckle, streaks and tears all count as detail. So each render is also held against
its reference (REF_DIR/NAME.png from evals/reference.py: the paint laid on the blockout, seen from the same camera),
over the pixels the reference has paint for:
  ref_corr     do the two show the same shapes: correlation of their detail between a 2 px and an 8 px blur, mean over
               the views. Blur leaves it alone; speckle, streaks, tears and misplaced paint pull it down.
  ref_err      colour difference after a 3 px blur (root mean square, 0..1), mean over the views. Holes, wrong colours.
  detail_kept  sharpness of a render / sharpness of its own reference, mean over the close views; kept_wall over the
               four walls. 1 = as crisp as the paint enlarged to that view; above 1 is detail the paint does not have
               (splat edges, or a method that adds some).
  bg_share     share of pixels showing the trainer's magenta background, which is seen wherever the splat has a gap
               from this camera; bg_max is the worst view.
"views" has every view's own numbers; "worst" the three least sharp. --sheet writes all renders side by side with
their sharpness; --crops the middle of each close and floor render at full size beside its reference.
Detail finer than the paint has nothing to be checked against: a rise in sharpness with ref_corr unchanged is safe, a
rise with ref_corr falling needs a look at the crops.
image_scores.py --selftest [RENDER REF]... checks the promises above on real renders (blur, brightness, speckle, streaks, holes)."""
import sys, os, glob, json, argparse, numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter as blur, median_filter, binary_erosion

WIDTH = 768       # every picture is measured at this width, so a 1024 render and a 768 painting are judged alike
KINDS = ("centre", "close", "floor", "far")
WALLS = ("close_x0", "close_x1", "close_z0", "close_z1")
BG = (0.613, 0.0101, 0.3984)      # msplat's TrainingConfig.bg_color, which train_scene.py leaves alone: a dark magenta
GREY = np.float32([0.299, 0.587, 0.114])

def picture(img):
    """PIL image, path or array -> float RGB(A) array 0..1 at WIDTH."""
    if isinstance(img, str): img = Image.open(img)
    if isinstance(img, np.ndarray): img = Image.fromarray(img if img.dtype == np.uint8 else (np.clip(img, 0, 1) * 255).astype(np.uint8))
    if img.mode not in ("RGB", "RGBA"): img = img.convert("RGB")
    if img.width != WIDTH: img = img.resize((WIDTH, round(img.height * WIDTH / img.width)), Image.LANCZOS)
    return np.asarray(img, np.float32) / 255

def sharpness(img, mask=None):
    """Fine detail as a share of coarse detail, 0 (smooth) to about 1 (crisp). img: PIL image, path or array.
    Fine = what lies between a 1 px and a 2 px blur, coarse = between 4 and 8 px. Blurring empties the fine band long
    before the coarse one. Both are differences of blurs, so a brighter or dimmer picture changes them alike and their
    ratio stays. The band starts at a 1 px blur, not at single pixels, and a 3x3 median runs first: a few odd pixels
    do not count as detail. mask: measure there only."""
    g = median_filter(picture(img)[..., :3] @ GREY, 3)
    b1, b2, b4, b8 = (blur(g, s) for s in (1, 2, 4, 8)); m = slice(None) if mask is None else mask
    return float(np.abs(b1 - b2)[m].mean() / (np.abs(b4 - b8)[m].mean() + 0.002))    # the floor keeps a blank picture at 0, not noise / noise

def magenta(a):
    return (a[..., 0] - a[..., 1] > 0.16) & (a[..., 2] - a[..., 1] > 0.16)

def against(render, ref):
    """One render held against its reference: {"kept", "corr", "err", "bg"}; the first three None where the reference
    has paint for under a fifth of the picture."""
    R = picture(render)[..., :3]; out = {"kept": None, "corr": None, "err": None, "bg": float(magenta(R).mean())}
    if ref is None: return out
    F = picture(ref)
    if F.shape[:2] != R.shape[:2] or F.shape[2] < 4: return out
    have = F[..., 3] > 0.5; F = np.where(have[..., None], F[..., :3], R)      # no paint there: nothing to differ from
    out["bg"] = float((magenta(R) & ~(have & magenta(F))).mean())             # paint that is magenta itself is not a gap
    M = binary_erosion(have, iterations=6)                                    # clear of the rim, where the blurs mix both sides
    if M.mean() < 0.2: return out
    a, b = (blur(x @ GREY, 2) - blur(x @ GREY, 8) for x in (R, F)); a, b = a[M] - a[M].mean(), b[M] - b[M].mean()
    d = blur(R, (3, 3, 0)) - blur(F, (3, 3, 0))
    out.update(kept=sharpness(R, M) / max(sharpness(F, M), 1e-6), corr=float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-12)), err=float(np.sqrt((d[M] ** 2).mean())))
    return out

def kind_of(name, kinds):
    return kinds.get(name) or next((k for k in KINDS if name.startswith(k)), "other")

def sheet(paths, labels, out, cols=6, width=2400):
    """Contact sheet: every picture with its label burnt in, at most `width` px wide."""
    cols = min(cols, len(paths)); tw = width // cols; ims = []
    for p, l in zip(paths, labels):
        im = Image.open(p).convert("RGB") if isinstance(p, str) else p
        if im.width != tw: im = im.resize((tw, round(im.height * tw / im.width)), Image.LANCZOS)
        d = ImageDraw.Draw(im); d.rectangle((0, 0, d.textlength(l) + 8, 15), fill=(0, 0, 0)); d.text((4, 2), l, fill=(255, 255, 255)); ims.append(im)
    th = max(im.height for im in ims); big = Image.new("RGB", (tw * cols, th * -(-len(ims) // cols)))
    for i, im in enumerate(ims): big.paste(im, (i % cols * tw, i // cols * th))
    big.save(out, quality=88)

def crops(rows, ref_dir, out, size=(512, 384)):
    """The middle of each close and floor render at full size (the 400 px thumbnails of the sheet hide blur that is plain
    in the viewer), each beside the same part of its reference. Green in a reference: no paint there."""
    ims, labels = [], []
    for kind, n, s, f in rows:
        if kind not in ("close", "floor"): continue
        im = Image.open(f).convert("RGB"); box = ((im.width - size[0]) // 2, (im.height - size[1]) // 2, (im.width + size[0]) // 2, (im.height + size[1]) // 2)
        ims.append(im.crop(box)); labels.append(f"{n}  {s:.3f}")
        rf = os.path.join(ref_dir or "", n + ".png")
        if os.path.exists(rf):
            r = Image.open(rf).convert("RGBA").resize(im.size, Image.BILINEAR); g = Image.new("RGB", im.size, (0, 255, 0)); g.paste(r, mask=r.split()[3])
            ims.append(g.crop(box)); labels.append("its reference (the paint)")
    if ims: sheet(ims, labels, out, cols=4, width=4 * size[0])

def score(render_dir, paint_dir=None, poses=None, sheet_out=None, ref_dir=None, crops_out=None):
    kinds = {c["name"]: c["kind"] for c in json.load(open(poses))} if poses else {}
    files = sorted(glob.glob(os.path.join(render_dir, "*.png")))
    rows = [(kind_of(n, kinds), n, sharpness(f), f) for f in files for n in [os.path.splitext(os.path.basename(f))[0]]]
    rows.sort(key=lambda r: ((KINDS + ("other",)).index(r[0]), WALLS.index(r[1]) if r[1] in WALLS else 9, r[1]))
    mean = lambda v: round(float(np.mean(v)), 4) if len(v) else None
    out = {"renders": len(rows), "sharp": {k: mean([r[2] for r in rows if r[0] == k]) for k in dict.fromkeys(r[0] for r in rows)}}
    out["sharp"]["wall"] = mean([r[2] for r in rows if r[1] in WALLS and r[0] == "close"])
    out["worst"] = [[r[1], round(r[2], 4)] for r in sorted(rows, key=lambda r: r[2])[:3]]      # the views to look at first
    per = {}
    for kind, n, s, f in rows:
        rf = os.path.join(ref_dir or "", n + ".png")
        per[n] = {"kind": kind, "sharp": round(s, 4), **{k: None if v is None else round(v, 4) for k, v in against(f, rf if ref_dir and os.path.exists(rf) else None).items()}}
    got = lambda key, pick=lambda n, v: True: [v[key] for n, v in per.items() if v[key] is not None and pick(n, v)]
    out.update(bg_share=mean(got("bg")), bg_max=round(max(got("bg"), default=0.0), 4), views=per)
    if ref_dir:
        out.update(ref_corr=mean(got("corr")), ref_err=mean(got("err")), ref_views=len(got("corr")),
                   detail_kept=mean(got("kept", lambda n, v: v["kind"] == "close")), kept_wall=mean(got("kept", lambda n, v: n in WALLS and v["kind"] == "close")))
    if paint_dir:      # the painter's own pictures, for scale: what "crisp" is in this world
        paint = [sharpness(f) for f in sorted(glob.glob(os.path.join(paint_dir, "*.png")) + glob.glob(os.path.join(paint_dir, "*.jpg")))]
        out.update(paint=mean(paint), paint_n=len(paint))
    if sheet_out and rows: sheet([r[3] for r in rows], [f"{r[1]}  {r[2]:.3f}" if r[1].startswith(r[0]) else f"{r[0]} {r[1]}  {r[2]:.3f}" for r in rows], sheet_out)
    if crops_out and rows: crops(rows, ref_dir, crops_out)
    return out

def selftest(render, ref):
    """The promises, on one real render and its reference. Prints the numbers and fails loudly if one is broken."""
    a = picture(render)[..., :3]; rng = np.random.default_rng(0); s0 = sharpness(a); r0 = against(a, ref)
    def speckle(a):
        n = blur(rng.standard_normal(a.shape[:2]), 1.5); return np.clip(a + 0.15 * (n / n.std())[..., None], 0, 1)
    def streaks(a):
        a = a.copy()
        for _ in range(150):
            x, y, l = rng.integers(0, a.shape[1] - 60), rng.integers(0, a.shape[0] - 4), rng.integers(30, 120); a[y:y + 3, x:x + l] = rng.random(3)
        return a
    def hole(a):
        a = a.copy(); h, w = a.shape[:2]; a[h // 3:h // 3 + h // 4, w // 3:w // 3 + w // 4] = BG; return a
    def stray(a):
        a = a.copy(); hit = rng.random(a.shape[:2]) < 0.002; a[hit] = rng.integers(0, 2, (int(hit.sum()), 1)); return a
    did = {"blur2": blur(a, (2, 2, 0)), "blur4": blur(a, (4, 4, 0)), "dim": a * 0.7, "bright": np.clip(a * 1.3, 0, 1), "stray_pixels": stray(a),
           "speckle": speckle(a), "streaks": streaks(a), "hole": hole(a)}
    got = {k: {"sharp": sharpness(v), **against(v, ref)} for k, v in did.items()}
    print(json.dumps({"render": render, "plain": {"sharp": round(s0, 4), **{k: round(v, 4) for k, v in r0.items()}},
                      **{k: {j: round(x, 4) for j, x in v.items()} for k, v in got.items()}}))
    assert s0 > got["blur2"]["sharp"] > got["blur4"]["sharp"], "blur must lower sharpness step by step"
    assert all(abs(got[k]["sharp"] / s0 - 1) < 0.1 for k in ("dim", "bright")), "brightness must barely move sharpness"
    assert abs(got["stray_pixels"]["sharp"] / s0 - 1) < 0.05, "a few stray pixels must barely move sharpness"
    assert all(got[k]["corr"] < r0["corr"] - 0.1 and got[k]["err"] > r0["err"] + 0.02 for k in ("speckle", "streaks")), "speckle and streaks must show against the reference"
    assert got["hole"]["err"] > r0["err"] + 0.02 and got["hole"]["bg"] > r0["bg"] + 0.04, "a hole must show as colour error and as background"
    assert abs(got["blur2"]["corr"] - r0["corr"]) < 0.05, "mild blur must leave ref_corr alone"

if __name__ == "__main__":
    if sys.argv[1:2] == ["--selftest"]:
        a = sys.argv[2:] or [x for w, n in (("ex-tower", "centre_2"), ("ex-tower", "close_z0"), ("garage-shapes", "close_z1")) for x in (f"evals/work/{w}-baseline/renders/{n}.png", f"evals/work/{w}/reference/{n}.png")]
        for r, f in zip(a[::2], a[1::2]): selftest(r, f)
        sys.exit()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("renders"); p.add_argument("--paint"); p.add_argument("--poses"); p.add_argument("--sheet"); p.add_argument("--ref"); p.add_argument("--crops")
    a = p.parse_args()
    print(json.dumps(score(a.renders, a.paint, a.poses, a.sheet, a.ref, a.crops)))
