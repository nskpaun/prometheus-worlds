#!/bin/zsh
# Sets up everything Prometheus Worlds needs, inside this folder. Needs only Xcode and the internet, once.
# Nothing is installed on the system: Python and every package live in .runtime/, thread/.venv and spike/.venv.
# Safe to run again: each step skips what is already done. The app runs this on first launch.
#   ./setup.sh             everything
#   ./setup.sh --no-shapes skip the optional 3D shape model (objects stay boxes)
#   ./setup.sh --check     exit 0 if setup is complete and up to date, 1 if not (changes nothing)
set -e
cd "$(dirname "$0")"

# Pinned versions. Every download is checked against its SHA-256 before use.
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261110/cpython-3.12.16%2B20261110-aarch64-apple-darwin-install_only.tar.gz"
PY_SHA="e956e60bc9465d577ef8062059aa05cd3a2665b4b8678a2ed24e432368330b8f"
LLM_REPO="mlx-community/Qwen3.5-9B-4bit"; LLM_REV="8b2b98c00a6b4d291155e4890773ca8f769aee53"
HY_GIT="https://github.com/ZimengXiong/Hunyuan3D-Swift"; HY_REV="292331f4d26ddb80b9dcea6bcb5629ff82f12b82"
HY_REPO="zimengxiong/hunyuan3d-mlx-shape-small"; HY_WREV="b7536809d38ad13fe6a9b7769a41fd5d42e520df"
MLX_WHL="https://files.pythonhosted.org/packages/38/29/71fe1f68756f515856e6930973c23245810d4aa3cd22fddd719d86a709dc/mlx-0.31.1-cp312-cp312-macosx_14_0_arm64.whl"
MLX_SHA="8a63b31a398c9519f2bb0c81cf3865d9baca4ff573ffc31ead465d18286184e8"

PY=.runtime/python/bin/python3
stamp() { shasum setup.sh runtime/thread.lock runtime/paint.lock | shasum | cut -c1-16; }
ours() { [[ -x $1/bin/python ]] && grep -q "$PWD/.runtime/python" $1/pyvenv.cfg 2>/dev/null; }      # made from our Python, not some other one
if [[ "$1" == "--check" ]]; then
  [[ -f .setup/done && "$(cat .setup/done)" == "$(stamp)" ]] && ours thread/.venv && ours spike/.venv && exit 0 || exit 1
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
step "Layout model (Qwen3.5 9B, about 5 GB)"
hf $LLM_REPO $LLM_REV

if [[ "$1" != "--no-shapes" ]]; then
  step "3D shape model (Hunyuan3D, optional, about 4 GB)"
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
  ) || print "  WARNING: the shape model could not be set up. Worlds still build; objects stay painted boxes."
fi

print "The image models (about 20 GB) download the first time you build a world."
mkdir -p .setup && stamp > .setup/done
step "Done"
