"""What each eval camera should show if the splat kept the paint exactly: the painted views laid on the blockout and
seen from that camera. A yardstick for the renders (image_scores.py), made without the trainer or any model.
  evals/reference.py WORLD [OUT_DIR]      (run from thread/)  -> OUT_DIR/NAME.png, default evals/work/WORLD/reference
Each pixel takes its colour from the painted view that owns that surface point (owner.py, the rule consistent.py
lays the paint by), read between that picture's pixels, so a wall seen from 0.8 m shows the paint smoothly enlarged: the best a
faithful splat could do. Pixels no painted view shows are left clear (alpha 0): consistent.py fills them with a blur,
which is not paint to be kept. Pictures are 768x576, the size image_scores.py measures at.
The world is only read. Prints one JSON line. Not followed: the moving things (they are not in the room's splat either).
"""
import sys, os, json, time, numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
EV = os.path.dirname(os.path.abspath(__file__)); TH = os.path.dirname(EV)
sys.path.insert(0, EV)
import cams as K, density as D
SIZE = (768, 576)

def reference(world_dir, out=None):
    t0 = time.time(); C, world = K.layout(world_dir); B = C.B
    out = out or os.path.join(EV, "work", os.path.basename(world), "reference"); os.makedirs(out, exist_ok=True)
    src, rank, lifted = D.sources(C, world)
    # as consistent.py reads them: enlarged to its working size, detail pictures settled into the ring's paint
    paint = src.paint if getattr(src, "paint", None) is not None else D.DT.pictures(C, src, rank, f"{world}/gen/images")
    shares = {}
    for name, kind, eye, look in K.layout_cams(world):
        P, sid, n = D.seen(C, (tuple(eye), tuple(look)), lifted, SIZE)
        own, at = D.owner(C, src, rank, lifted, P, sid, n, where=True)
        img = np.zeros(sid.shape + (4,), np.float32)
        for v in np.unique(own[own >= 0]):
            m = own == v
            for ch in range(3): img[m, ch] = map_coordinates(paint[v][..., ch], [at[m][:, 1], at[m][:, 0]], order=1, mode="nearest")
        img[..., 3] = 255 * (own >= 0)
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGBA").save(f"{out}/{name}.png")
        shares[name] = round(float((own >= 0).mean()), 3)
    return {"world": os.path.basename(world), "reference": out, "views": len(shares), "painted_share": shares, "seconds": round(time.time() - t0, 1)}

if __name__ == "__main__":
    print(json.dumps(reference(*sys.argv[1:3])))
