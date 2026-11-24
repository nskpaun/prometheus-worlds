#!/bin/zsh
# Sets up everything Prometheus Worlds needs. Needs only Xcode and the internet, once.
# Nothing is installed on the system: Python and every package live in .runtime/, thread/.venv and spike/.venv.
# Safe to run again: each step skips what is already done. The app runs this on first launch.
#   ./setup.sh             everything
#   ./setup.sh --no-shapes skip the 3D shape model (sentences are then built by the earlier ring method, objects stay boxes)
#   ./setup.sh --no-pbr    skip the visuals-first method's models (art director, sky, materials, hero 3D, prop library)
#   ./setup.sh --check     exit 0 if setup is complete and up to date, 1 if not (changes nothing)
#   PROMETHEUS_ADOPT=DIR[:DIR] ./setup.sh   take models already on this Mac from other Hugging Face caches (.../hub folders)
#                          instead of downloading them again: each file is checked against its checksum, then linked in
# The models are kept in the Hugging Face cache ($HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub), the
# 3D shape model in ext/.
set -e
cd "$(dirname "$0")"

# Pinned versions. Every download is checked against its SHA-256 before use.
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261110/cpython-3.12.16%2B20261110-aarch64-apple-darwin-install_only.tar.gz"
PY_SHA="e956e60bc9465d577ef8062059aa05cd3a2665b4b8678a2ed24e432368330b8f"
LLM_REPO="mlx-community/Qwen3.5-9B-4bit"; LLM_REV="8b2b98c00a6b4d291155e4890773ca8f769aee53"
PLAN_REPO="mlx-community/Qwen3.5-35B-A3B-4bit"; PLAN_REV="1e20fd8d42056f870933bf98ca6211024744f7ec"   # as thread/build/chat.py loads it
KLEIN_REPO="black-forest-labs/FLUX.2-klein-4B"; KLEIN_REV="e7b7dc27f91deacad38e78976d1f2b499d76a294"   # the revision mflux 0.24.0 loads: move the two together
ZIMG_REPO="Tongyi-MAI/Z-Image-Turbo"; ZIMG_REV="f332072aa78be7aecdf3ee76d5c247082da564a6"            # likewise the revision mflux 0.24.0 loads
ZIMG_FILES=('vae/*.safetensors' 'vae/*.json' 'transformer/*.safetensors' 'transformer/*.json' 'text_encoder/*.safetensors' 'text_encoder/*.json' 'tokenizer/*')
KLEIN_FILES=('vae/*' 'transformer/*' 'text_encoder/*' 'tokenizer/*' LICENSE.md)                       # what mflux reads; leaves out a second 7.75 GB copy of the transformer
HY_GIT="https://github.com/ZimengXiong/Hunyuan3D-Swift"; HY_REV="292331f4d26ddb80b9dcea6bcb5629ff82f12b82"
# the visuals-first method (thread/pbr, docs/pbr-method.md)
VL_REPO="mlx-community/Qwen3-VL-32B-Instruct-4bit"; VL_REV="6e5644d3ea4b953b5221ffd02339bf897041038a"                 # the art director
KBASE_REPO="black-forest-labs/FLUX.2-klein-base-9B"; KBASE_REV="32773329fbe7e81a90ef971740e8ba4b0364ecf3"           # the revision mflux 0.24.0 loads; the 360 degree sky, with:
PANO_REPO="crafiq/flux-2-klein-9b-360-panorama-lora"; PANO_REV="494ac687accc8b1e8b4e2f78c49e2dfc67c1c009"
CHORD_GIT="https://github.com/ubisoft/ubisoft-laforge-chord"; CHORD_REV="675a29d1fee5e1b256833bd4edabd619be43d0c7"   # PBR materials
CHORD_REPO="Ubisoft/ubisoft-laforge-chord"; CHORD_WREV="2b61fc12f316b4692c2a79651e03340427450735"
SD21_REPO="sd2-community/stable-diffusion-2-1-base"; SD21_REV="4e63672c03103b6c636b8fb4119ba982469b2955"            # CHORD's base: its configs only
TRELLIS_GIT="https://github.com/shivampkumar/trellis-mac"; TRELLIS_REV="d58628f4f5b9c3de8274cb110074154f4b31cef2"   # hero 3D models
TRELLIS_REPO="microsoft/TRELLIS.2-4B"; TRELLIS_WREV="af44b45f2e35a493886929c6d786e563ec68364d"
DINO_REPO="facebook/dinov3-vitl16-pretrain-lvd1689m"; DINO_REV="ea8dc2863c51be0a264bab82070e3e8836b02d51"
TRELLIS1_REPO="microsoft/TRELLIS-image-large"; TRELLIS1_REV="25e0d31ffbebe4b5a97464dd851910efc3002d96"
EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"; EIGEN_SHA="8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72"   # headers only, for the texture baker
HY_REPO="zimengxiong/hunyuan3d-mlx-shape-small"; HY_WREV="b7536809d38ad13fe6a9b7769a41fd5d42e520df"
SHARP_GIT="https://github.com/apple/ml-sharp"; SHARP_REV="aed6527499ef91cba3b54c18d49a870f25947190"      # one photo in, a splat out: what worlds from photos are made of
SHARP_URL="https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"; SHARP_SHA="94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d"
LG_GIT="https://github.com/cvg/LightGlue"; LG_REV="5a650c444718caffc2669dc0094cde908f42dd73"                 # matches between photos, for the camera solve
LG_KP_URL="https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth"; LG_KP_SHA="5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2"
LG_M_URL="https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth"; LG_M_SHA="d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503"
MLX_WHL="https://files.pythonhosted.org/packages/38/29/71fe1f68756f515856e6930973c23245810d4aa3cd22fddd719d86a709dc/mlx-0.31.1-cp312-cp312-macosx_14_0_arm64.whl"
MLX_SHA="8a63b31a398c9519f2bb0c81cf3865d9baca4ff573ffc31ead465d18286184e8"

PY=.runtime/python/bin/python3
stamp() { shasum setup.sh runtime/thread.lock runtime/paint.lock | shasum | cut -c1-16; }
ours() { [[ -x $1/bin/python ]] && grep -q "$PWD/.runtime/python" $1/pyvenv.cfg 2>/dev/null; }      # made from our Python, not some other one
HUB="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
cached() { local d="$HUB/models--${1//\//--}/snapshots/$2"; shift 2; for f; do [[ -e $d/$f ]] || return 1; done; }   # REPO REV FILE...: every file there (follows the links to the weights)
models() {      # the three large models a world built from a sentence cannot do without
  cached $ZIMG_REPO $ZIMG_REV transformer/diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors vae/diffusion_pytorch_model.safetensors text_encoder/model-0000{1,2,3}-of-00003.safetensors tokenizer/tokenizer.json &&
  cached $PLAN_REPO $PLAN_REV config.json tokenizer.json model-0000{1,2,3,4}-of-00004.safetensors &&
  cached $KLEIN_REPO $KLEIN_REV transformer/diffusion_pytorch_model.safetensors vae/diffusion_pytorch_model.safetensors text_encoder/model-0000{1,2}-of-00002.safetensors tokenizer/tokenizer.json
}
if [[ "$1" == "--check" ]]; then
  [[ -f .setup/done && "$(cat .setup/done)" == "$(stamp)" ]] && ours thread/.venv && ours spike/.venv && models && exit 0 || exit 1
fi
step() { print -r -- "STEP: $1"; }
fail() { print -r -- "ERROR: $1"; exit 1; }
fetch() {   # URL FILE SHA256
  curl -fL --retry 3 --progress-bar -o "$2.part" "$1" || fail "Could not download $1. Check the internet connection and try again."
  [[ "$(shasum -a 256 "$2.part" | cut -d' ' -f1)" == "$3" ]] || { rm -f "$2.part"; fail "The download of $1 did not match its checksum."; }
  mv "$2.part" "$2"
}

[[ "$(uname -m)" == "arm64" ]] || fail "Prometheus Worlds needs an Apple Silicon Mac."
xcrun --find swift >/dev/null 2>&1 || fail "Xcode is not installed, or has never been opened. Install it from the App Store, open it once, then try again."

step "Python (self-contained, about 25 MB)"
if [[ ! -x $PY ]]; then
  mkdir -p .runtime && fetch "$PY_URL" .runtime/python.tar.gz $PY_SHA
  rm -rf .runtime/python && tar -xzf .runtime/python.tar.gz -C .runtime && rm .runtime/python.tar.gz
fi
$PY --version

env_from() {   # VENV LOCK
  ours $1 || { rm -rf $1; $PY -m venv $1; }
  $1/bin/python -m pip install --quiet --disable-pip-version-check --require-hashes --only-binary :all: -r $2
}
step "Packages for the pipelines (numpy, Open3D, COLMAP, the splat trainer)"
env_from thread/.venv runtime/thread.lock
step "Packages for the image, layout and photo models (MLX, mflux, mlx-lm, PyTorch)"
env_from spike/.venv runtime/paint.lock

hf() { spike/.venv/bin/python -c "import sys; from huggingface_hub import snapshot_download as s; s(sys.argv[1], revision=sys.argv[2], **({'local_dir': sys.argv[3]} if len(sys.argv) > 3 else {}))" "$@"; }
model() {   # REPO REV [PATTERN...]  into the Hugging Face cache. Files of that revision found in a cache named in PROMETHEUS_ADOPT are linked in, after a checksum, instead of downloaded.
  spike/.venv/bin/python - "$@" <<'E' || fail "Could not download $1. Check the internet connection and try again."
import sys, os, re, hashlib, shutil
from huggingface_hub import constants, snapshot_download
repo, rev, pats = sys.argv[1], sys.argv[2], sys.argv[3:]
hub = constants.HF_HUB_CACHE; name = "models--" + repo.replace("/", "--"); dst = os.path.join(hub, name)
def digest(path, etag):          # a blob is named by its content: SHA-256 for a large file, git's SHA-1 for a small one
    h = hashlib.sha256() if len(etag) == 64 else hashlib.sha1(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        while b := f.read(1 << 24): h.update(b)
    return h.hexdigest()
for src in filter(None, os.environ.get("PROMETHEUS_ADOPT", "").split(":")):
    snap = os.path.join(src, name, "snapshots", rev)
    if not os.path.isdir(snap) or os.path.realpath(src) == os.path.realpath(hub): continue
    n = 0
    for root, _, files in os.walk(snap):
        for f in files:
            p = os.path.join(root, f); out = os.path.join(dst, "snapshots", rev, os.path.relpath(p, snap))
            if os.path.lexists(out) or not os.path.islink(p) or not os.path.isfile(p): continue
            etag = os.path.basename(os.readlink(p)); real = os.path.realpath(p); blob = os.path.join(dst, "blobs", etag)
            if not re.fullmatch("[0-9a-f]{40}|[0-9a-f]{64}", etag) or digest(real, etag) != etag: print("  not taken (checksum):", p); continue
            os.makedirs(os.path.dirname(blob), exist_ok=True); os.makedirs(os.path.dirname(out), exist_ok=True)
            if not os.path.lexists(blob):
                try: os.link(real, blob)                                   # same disk: a second name for the same bytes, the original stays
                except OSError: shutil.copyfile(real, blob + ".part"); os.replace(blob + ".part", blob)
            os.symlink(os.path.relpath(blob, os.path.dirname(out)), out); n += 1
    if n: print(f"  took {n} files of {repo} from {src}")
snapshot_download(repo, revision=rev, allow_patterns=pats or None)          # fetches whatever is still missing; nothing if all is there
E
}
step "Layout model for the ring method (Qwen3.5 9B, about 6 GB)"
hf $LLM_REPO $LLM_REV
step "World-plan model (Qwen3.5 35B-A3B, about 20 GB)"
model $PLAN_REPO $PLAN_REV
step "Material and paint model (FLUX.2 Klein 4B, about 16 GB)"
model $KLEIN_REPO $KLEIN_REV $KLEIN_FILES
step "Object-picture model (Z-Image Turbo, about 31 GB)"
model $ZIMG_REPO $ZIMG_REV $ZIMG_FILES

step "Photo-to-splat model (SHARP, about 2.8 GB)"
( set -e
  S=ext/ml-sharp
  if [[ "$(git -C $S rev-parse HEAD 2>/dev/null)" != $SHARP_REV ]]; then
    rm -rf $S && mkdir -p ext && git clone --quiet $SHARP_GIT $S && git -C $S checkout --quiet $SHARP_REV
  fi
  [ -f $S/sharp_2572gikvuh.pt ] || fetch "$SHARP_URL" $S/sharp_2572gikvuh.pt $SHARP_SHA
) || print "  WARNING: the photo model could not be set up, so worlds cannot be built from photos yet. Run ./setup.sh again to retry."

step "Photo-matching model (ALIKED and LightGlue, about 50 MB)"
( set -e
  L=ext/LightGlue
  if [[ "$(git -C $L rev-parse HEAD 2>/dev/null)" != $LG_REV ]]; then
    rm -rf $L && mkdir -p ext && git clone --quiet $LG_GIT $L && git -C $L checkout --quiet $LG_REV
  fi
  mkdir -p $L/hub/checkpoints      # where the code looks for its weights, under the names it gives them
  [ -f $L/hub/checkpoints/aliked-n16.pth ] || fetch "$LG_KP_URL" $L/hub/checkpoints/aliked-n16.pth $LG_KP_SHA
  [ -f $L/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth ] || fetch "$LG_M_URL" $L/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth $LG_M_SHA
) || print "  WARNING: the photo-matching model could not be set up. Photos are matched the earlier way until it is, which places fewer of them. Run ./setup.sh again to retry."

if [[ "$1" != "--no-shapes" ]]; then
  step "3D shape model (Hunyuan3D, about 4 GB)"
  ( set -e
    H=ext/Hunyuan3D-Swift
    if [[ "$(git -C $H rev-parse HEAD 2>/dev/null)" != $HY_REV ]]; then
      rm -rf $H && git clone --quiet $HY_GIT $H && git -C $H checkout --quiet $HY_REV
    fi
    [ -x $H/.build/release/hy3d ] || (cd $H && swift build -c release)
    # swift build makes MLX's Metal shaders only with full Xcode selected; otherwise take them from the matching MLX wheel
    if [ ! -f $H/.build/release/mlx.metallib ]; then
      fetch "$MLX_WHL" .runtime/mlx.whl $MLX_SHA && unzip -p .runtime/mlx.whl mlx/lib/mlx.metallib > $H/.build/release/mlx.metallib && rm .runtime/mlx.whl
    fi
    [ -f $H/weights/shape-small/model.fp16.safetensors ] || hf $HY_REPO $HY_WREV $H/weights/shape-small
  ) || print "  WARNING: the shape model could not be set up. Sentences are built by the earlier ring method until it is, where objects stay painted boxes. Run ./setup.sh again to retry."
fi

if [[ "$1" != "--no-pbr" && "$2" != "--no-pbr" ]]; then
  step "Art director (Qwen3-VL 32B, about 18 GB)"
  model $VL_REPO $VL_REV
  step "Sky panorama model (FLUX.2 Klein base 9B and a 360 degree LoRA, about 18 GB; needs access granted on Hugging Face)"
  ( model $KBASE_REPO $KBASE_REV && model $PANO_REPO $PANO_REV ) || print "  WARNING: the sky model could not be fetched (ask for access at huggingface.co/$KBASE_REPO, then sign in with: spike/.venv/bin/hf auth login). Skies stay plain until it is."
  step "CC0 prop library (Poly Haven, about 4 GB)"
  thread/.venv/bin/python -B thread/pbr/kitbash.py fetch ext/cc0-kitbash || print "  WARNING: the prop library could not be fetched. Run ./setup.sh again to retry."
  step "PBR material model (CHORD, about 4 GB; needs access granted on Hugging Face)"
  ( set -e
    C=ext/ubisoft-laforge-chord
    if [[ "$(git -C $C rev-parse HEAD 2>/dev/null)" != $CHORD_REV ]]; then rm -rf $C && git clone --quiet $CHORD_GIT $C && git -C $C checkout --quiet $CHORD_REV; fi
    sed -i '' 's/"cuda" if torch.cuda.is_available() else "cpu"/"mps" if torch.backends.mps.is_available() else "cpu"/' $C/chord/module/stable_diffusion.py $C/chord/module/base.py
    sed -i '' 's#model_key = "RedbeardNZ/stable-diffusion-2-1-base"#model_key = "sd2-community/stable-diffusion-2-1-base"#' $C/chord/module/stable_diffusion.py
    [ -x $C/.venv/bin/python ] || .runtime/python/bin/python3 -m venv $C/.venv
    $C/.venv/bin/pip install --quiet torch torchvision && $C/.venv/bin/pip install --quiet -r $C/requirements.txt
    model $SD21_REPO $SD21_REV "*.json" "*.txt" && model $CHORD_REPO $CHORD_WREV
  ) || print "  WARNING: the material model could not be set up (ask for access at huggingface.co/$CHORD_REPO). Materials keep their colour pictures until it is."
  step "Hero 3D model (TRELLIS.2 4B for Apple Silicon, about 16 GB)"
  ( set -e
    T=ext/trellis-mac
    if [[ "$(git -C $T rev-parse HEAD 2>/dev/null)" != $TRELLIS_REV ]]; then rm -rf $T && git clone --quiet $TRELLIS_GIT $T && git -C $T checkout --quiet $TRELLIS_REV; fi
    xcrun -sdk macosx metal --version >/dev/null 2>&1 || xcodebuild -downloadComponent MetalToolchain
    if [ ! -x $T/.venv/bin/python ]; then
      PATH="$PWD/.runtime/python/bin:$PATH" SKIP_METAL=1 bash $T/setup.sh        # the environment, TRELLIS.2 and its patches; the Metal parts below
      # PyTorch's headers need C++20 with this Xcode, and the texture baker needs Eigen's headers
      sed -i '' 's/-std=c++17/-std=c++20/g' $T/deps/mtlbvh/setup.py $T/deps/mtldiffrast/setup.py $T/deps/mtlmesh/setup.py $T/deps/mtlgemm/setup.py $T/deps/trellis2-apple/o-voxel/setup.py
      [ -d ext/eigen-3.4.0 ] || { fetch "$EIGEN_URL" .runtime/eigen.tar.gz $EIGEN_SHA && tar xzf .runtime/eigen.tar.gz -C ext && rm .runtime/eigen.tar.gz; }
      for p in mtlbvh mtldiffrast mtlmesh mtlgemm trellis2-apple/o-voxel; do
        ( cd $T && MACOSX_DEPLOYMENT_TARGET=12.0 CPLUS_INCLUDE_PATH="$PWD/../eigen-3.4.0" VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pip install --quiet --no-build-isolation deps/$p ) || print "  (the Metal part $p did not build: TRELLIS.2 uses its slower fallback)"
      done
    fi
    model $TRELLIS_REPO $TRELLIS_WREV && model $DINO_REPO $DINO_REV && model $TRELLIS1_REPO $TRELLIS1_REV "ckpts/ss_dec_conv3d_16l8_fp16.*"      # TRELLIS.2 borrows the first TRELLIS's structure decoder
  ) || print "  WARNING: the hero 3D model could not be set up. Hero things come from the prop library or are plain until it is."
fi

mkdir -p .setup && stamp > .setup/done
step "Done"
