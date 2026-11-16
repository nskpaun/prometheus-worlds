"""COLMAP steps through pycolmap, so no separate COLMAP install is needed. Run in thread/.venv:
  sfm.py solve WORLD_DIR      images/ -> db.db, sparse/N (camera positions), then dense/ (undistorted images + model)
  sfm.py tobin MODEL_DIR      cameras/images/points3D .txt -> .bin, in place
  sfm.py totxt MODEL_DIR OUT  .bin -> .txt in OUT"""
import sys, os, glob, shutil
import pycolmap

def solve(d):
    reader = pycolmap.ImageReaderOptions(); reader.camera_model = "SIMPLE_RADIAL"
    pycolmap.extract_features(f"{d}/db.db", f"{d}/images", camera_mode=pycolmap.CameraMode.AUTO, reader_options=reader)
    pycolmap.match_exhaustive(f"{d}/db.db")
    os.makedirs(f"{d}/sparse", exist_ok=True)
    pycolmap.incremental_mapping(f"{d}/db.db", f"{d}/images", f"{d}/sparse")
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
