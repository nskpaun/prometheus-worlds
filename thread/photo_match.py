"""Matches between every pair of photos from learned features (ALIKED keypoints, LightGlue matching), for the camera solve.
  photo_match.py IMAGES_DIR OUT.npz        (run in spike/.venv, which has PyTorch; a small model, a few minutes for 60 photos)
They find the same spot in two photos on plain walls, ceilings and repeating tile, where the classic features (SIFT)
find too little: of 67 photos of a bathroom SIFT placed 57 and these 62, with three times the points per photo.
OUT.npz holds names, kp_NAME (keypoints in pixels of the photo, x right, y down, pixel centres at .5), pair_names
("A|B") and m_I (rows of [index in A, index in B]) for pair I. sfm.py solve reads it.
LightGlue's code is in ext/LightGlue with its weights in ext/LightGlue/hub (setup.sh puts them there); LIGHTGLUE_DIR
names another copy."""
import sys, os, time, itertools, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
LG = os.environ.get("LIGHTGLUE_DIR") or os.path.join(HERE, "..", "ext", "LightGlue")
SIDE = 1600                # photos are matched at this many pixels across

def keypoints_for(n_photos):
    """Keypoints per photo: matching time grows with the number of pairs, so large sets get fewer."""
    return 4096 if n_photos * (n_photos - 1) // 2 <= 2500 else 2048

def main(src, out):
    import torch
    from PIL import Image
    sys.path.insert(0, LG)
    if os.path.isdir(os.path.join(LG, "hub")): torch.hub.set_dir(os.path.join(LG, "hub"))       # weights from setup, not the network
    from lightglue import ALIKED, LightGlue
    names = sorted(n for n in os.listdir(src) if n.lower().endswith((".jpg", ".jpeg", ".png"))); nkp = keypoints_for(len(names))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ext = ALIKED(max_num_keypoints=nkp, detection_threshold=0.005).eval().to(dev); mat = LightGlue(features="aliked").eval().to(dev); feats = {}; t = time.time()
    with torch.no_grad():
        for n in names:
            im = Image.open(os.path.join(src, n)).convert("RGB"); W, H = im.size; s = SIDE / max(W, H)
            x = torch.from_numpy(np.asarray(im.resize((round(W * s), round(H * s)), Image.LANCZOS)).copy()).float().permute(2, 0, 1)[None] / 255
            f = {k: v.to(dev) for k, v in ext.extract(x.to(dev), resize=None).items()}; f["scale"] = s; feats[n] = f
        print("keypoints:", len(names), "photos,", int(np.median([f["keypoints"].shape[1] for f in feats.values()])), "each,", round(time.time() - t, 1), "s", flush=True); t = time.time()
        pairs = {}
        for a, b in itertools.combinations(names, 2):
            fa, fb = ({k: v for k, v in feats[n].items() if k != "scale"} for n in (a, b)); m = mat({"image0": fa, "image1": fb})["matches"][0].cpu().numpy()
            if len(m) >= 15: pairs[(a, b)] = m.astype(np.uint32)
    print("matched", len(pairs), "of", len(names) * (len(names) - 1) // 2, "pairs,", round(time.time() - t, 1), "s", flush=True)
    np.savez(out, names=np.array(names), pair_names=np.array([f"{a}|{b}" for a, b in pairs]), **{"kp_" + n: (feats[n]["keypoints"][0].cpu().numpy() + 0.5) / feats[n]["scale"] for n in names},
             **{f"m_{i}": m for i, m in enumerate(pairs.values())})

if __name__ == "__main__": main(*sys.argv[1:3])
