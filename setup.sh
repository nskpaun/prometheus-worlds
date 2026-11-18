#!/bin/zsh
# Sets up everything Prometheus Worlds needs. Needs only Xcode and the internet, once.
# Nothing is installed on the system: Python and every package live in .runtime/, thread/.venv and spike/.venv.
# Safe to run again: each step skips what is already done. The app runs this on first launch.
#   ./setup.sh             everything
#   ./setup.sh --no-shapes skip the 3D shape model (sentences are then built by the earlier ring method, objects stay boxes)
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
HY_REPO="zimengxiong/hunyuan3d-mlx-shape-small"; HY_WREV="b7536809d38ad13fe6a9b7769a41fd5d42e520df"
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
step "Packages for the image and layout models (MLX, mflux, mlx-lm)"
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

mkdir -p .setup && stamp > .setup/done
step "Done"
