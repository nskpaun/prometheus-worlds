"""One splat piece per photo, from Apple's SHARP (a photo in, about 1.2 million splats out, in one pass; no training).
  photo_pieces.py IMAGES_DIR OUT_DIR [--colmap TXT_DIR]        (run in spike/.venv, which has PyTorch; one model, about 15 GB)
With --colmap, each photo's focal length is the solved one (TXT_DIR/cameras.txt and images.txt, undistorted pinhole
cameras). Without, it is read from the photo's EXIF (35 mm equivalent), else taken as 30 mm.
Writes OUT_DIR/NAME.npz for each photo: xyz (camera frame: x right, y down, z forward, in SHARP's guess of metres),
scale, quat (w x y z), rgb (sRGB, 0..1), opacity, and size = [width, height, focal in pixels]. The splats lie on a
768 x 768 grid over the photo, two per cell: index = layer * 768 * 768 + row * 768 + column.
SHARP's code is in ext/ml-sharp (setup.sh puts it there); SHARP_DIR names another copy."""
import sys, os, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SHARP = os.environ.get("SHARP_DIR") or os.path.join(HERE, "..", "ext", "ml-sharp")
WEIGHTS = "sharp_2572gikvuh.pt"

def colmap_focals(txt):
    """{photo name: focal in pixels} from COLMAP text files."""
    cams = {l.split()[0]: float(l.split()[4]) for l in open(os.path.join(txt, "cameras.txt")) if l[0] != "#"}
    rows = [l.split() for l in open(os.path.join(txt, "images.txt")) if l[0] != "#"][::2]
    return {r[9]: cams[r[8]] for r in rows}

def exif_focal(path, w, h):
    """Focal length in pixels from the 35 mm equivalent in EXIF; 30 mm when the photo does not say."""
    from PIL import Image
    f35 = 30.0
    try:
        ex = Image.open(path).getexif(); sub = ex.get_ifd(0x8769); v = float(sub.get(41989) or 0)      # FocalLengthIn35mmFilm
        if v >= 10: f35 = v
    except Exception: pass
    return f35 * np.hypot(w, h) / np.hypot(36.0, 24.0), f35

def predict(model, im, f, dev, unproject, side=1536):
    """SHARP on one photo (as its own predict_image does): the photo resized to 1536 x 1536, splats back in the photo's camera frame."""
    import torch, torch.nn.functional as F
    with torch.no_grad():
        x = torch.from_numpy(im.copy()).float().to(dev).permute(2, 0, 1) / 255.0; _, h, w = x.shape
        g = model(F.interpolate(x[None], size=(side, side), mode="bilinear", align_corners=True), torch.tensor([f / w]).float().to(dev))
        K = torch.tensor([[f * side / w, 0, side / 2, 0], [0, f * side / h, side / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float().to(dev)
        return unproject(g, torch.eye(4).to(dev), K, (side, side))

def main(src, out, txt=None):
    import torch
    from PIL import Image, ImageOps
    sys.path.insert(0, os.path.join(SHARP, "src"))
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils.gaussians import unproject_gaussians
    from sharp.utils import color_space as cs
    os.makedirs(out, exist_ok=True); dev = "mps" if torch.backends.mps.is_available() else "cpu"; t = time.time()
    w8 = os.path.join(SHARP, WEIGHTS)
    if not os.path.exists(w8): w8 = os.path.join(torch.hub.get_dir(), "checkpoints", WEIGHTS)
    model = create_predictor(PredictorParams()); model.load_state_dict(torch.load(w8, weights_only=True, map_location="cpu")); model.eval().to(dev)
    print("SHARP loaded on", dev, round(time.time() - t, 1), "s", flush=True)
    focal = colmap_focals(txt) if txt else {}
    flat = lambda x: (x.flatten(0, 1) if x.dim() > 2 else x.flatten()).float().cpu().numpy()
    names = [n for n in sorted(os.listdir(src)) if n.lower().endswith((".jpg", ".jpeg", ".png")) and (not focal or n in focal)]
    for n in names:
        t = time.time(); p = os.path.join(src, n); im = np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB")); h, w = im.shape[:2]
        f, note = (focal[n], "solved") if n in focal else exif_focal(p, w, h); note = note if n in focal else "%g mm" % note
        g = predict(model, im, f, dev, unproject_gaussians)
        np.savez(os.path.join(out, os.path.splitext(n)[0] + ".npz"), xyz=flat(g.mean_vectors), scale=flat(g.singular_values), quat=flat(g.quaternions),
                 rgb=flat(cs.linearRGB2sRGB(g.colors)), opacity=flat(g.opacities), size=np.array([w, h, f], np.float64))
        print(n, f"{w}x{h}", "focal %.0f px (%s)" % (f, note), round(time.time() - t, 1), "s", flush=True)
    if dev == "mps": print("peak GB", round(torch.mps.driver_allocated_memory() / 1e9, 1))

if __name__ == "__main__":
    a = sys.argv[1:]; main(a[0], a[1], a[a.index("--colmap") + 1] if "--colmap" in a else None)
