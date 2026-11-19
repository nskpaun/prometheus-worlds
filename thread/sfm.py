"""COLMAP steps through pycolmap, so no separate COLMAP install is needed. Run in thread/.venv:
  sfm.py solve WORLD_DIR [MATCHES.npz]   images/ -> db.db, sparse/N (camera positions), then dense/ (undistorted images + model).
                              With MATCHES.npz (photo_match.py) its learned matches are used; without, COLMAP's own SIFT features
  sfm.py tobin MODEL_DIR      cameras/images/points3D .txt -> .bin, in place
  sfm.py totxt MODEL_DIR OUT  .bin -> .txt in OUT"""
import sys, os, glob, shutil
import pycolmap

def solve(d, matches=None):
    reader = pycolmap.ImageReaderOptions(); reader.camera_model = "SIMPLE_RADIAL"; db = f"{d}/db.db"
    if matches:                                          # learned keypoints and matches from photo_match.py: only checked for geometry here
        import numpy as np
        Z = np.load(matches); pycolmap.Database.open(db).close(); pycolmap.import_images(db, f"{d}/images", camera_mode=pycolmap.CameraMode.AUTO, options=reader)
        D = pycolmap.Database.open(db); ids = {im.name: im.image_id for im in D.read_all_images()}
        for n in Z["names"]: D.write_keypoints(ids[str(n)], Z["kp_" + str(n)].astype(np.float32))
        with open(f"{d}/pairs.txt", "w") as fh:
            for i, pn in enumerate(Z["pair_names"]):
                a, b = str(pn).split("|"); D.write_matches(ids[a], ids[b], Z[f"m_{i}"]); fh.write(f"{a} {b}\n")
        D.close(); pycolmap.verify_matches(db, f"{d}/pairs.txt")
    else:
        pycolmap.extract_features(db, f"{d}/images", camera_mode=pycolmap.CameraMode.AUTO, reader_options=reader)
        pycolmap.match_exhaustive(db)
    os.makedirs(f"{d}/sparse", exist_ok=True)
    pycolmap.incremental_mapping(db, f"{d}/images", f"{d}/sparse")
    models = glob.glob(f"{d}/sparse/*/")
    if not models: sys.exit("Could not solve camera positions: the photos need more overlap")
    best = max(models, key=lambda m: os.path.getsize(m + "images.bin"))
    pycolmap.undistort_images(f"{d}/dense", best, f"{d}/images", output_type="COLMAP")

def tobin(m):
    pycolmap.Reconstruction(m).write_binary(m)

def totxt(m, out):
    pycolmap.Reconstruction(m).write_text(out)

if __name__ == "__main__":
    {"solve": solve, "tobin": tobin, "totxt": totxt}[sys.argv[1]](*sys.argv[2:])
