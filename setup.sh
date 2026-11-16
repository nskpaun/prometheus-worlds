#!/bin/zsh
# Sets up everything Prometheus Worlds needs on this Mac. Safe to run again: each step skips what is already done.
# The app runs this on first launch; you can also run it yourself from the repository root.
#   ./setup.sh             everything
#   ./setup.sh --no-shapes skip the optional 3D shape model (objects stay boxes)
#   ./setup.sh --check     exit 0 if setup is complete and up to date, 1 if not (changes nothing)
set -e
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
stamp() { shasum setup.sh thread/requirements.txt spike/requirements.txt | shasum | cut -c1-16; }
if [[ "$1" == "--check" ]]; then
  [[ -f .setup/done && "$(cat .setup/done)" == "$(stamp)" && -x thread/.venv/bin/python && -x spike/.venv/bin/python ]] && exit 0 || exit 1
fi
step() { print -r -- "STEP: $1"; }
fail() { print -r -- "ERROR: $1"; exit 1; }

step "Checking for Homebrew"
command -v brew >/dev/null || fail "Homebrew is not installed. Install it from https://brew.sh (it asks for your password), then run this again."

step "Installing tools with Homebrew (uv, colmap, ollama, libusb)"
export HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_ENV_HINTS=1
for f in uv colmap ollama libusb; do      # a tool already on PATH is left alone
  if brew list --versions $f >/dev/null 2>&1 || { [[ $f != libusb ]] && command -v $f >/dev/null; }; then print "  $f: already installed"
  else brew install $f; fi
done

step "Python environment for the pipelines (thread/.venv)"
[ -x thread/.venv/bin/python ] || uv venv --python 3.12 thread/.venv
uv pip install --python thread/.venv/bin/python -r thread/requirements.txt

step "Python environment for the image models (spike/.venv)"
[ -x spike/.venv/bin/python ] || uv venv --python 3.12 spike/.venv
uv pip install --python spike/.venv/bin/python -r spike/requirements.txt

step "Layout model (qwen3.5:9b, about 6 GB)"
started=""
if ! ollama list >/dev/null 2>&1; then ollama serve >/dev/null 2>&1 & started=$!; for i in {1..30}; do ollama list >/dev/null 2>&1 && break; sleep 1; done; fi
ollama list 2>/dev/null | grep -q "^qwen3.5:9b" && print "  already downloaded" || ollama pull qwen3.5:9b
[ -n "$started" ] && kill $started 2>/dev/null || true

if [[ "$1" != "--no-shapes" ]]; then
  step "3D shape model (Hunyuan3D, optional, about 4 GB)"
  ( set -e
    H=ext/Hunyuan3D-Swift
    [ -d $H ] || git clone --depth 1 https://github.com/ZimengXiong/Hunyuan3D-Swift $H
    [ -x $H/.build/release/hy3d ] || (cd $H && swift build -c release)
    # swift build makes MLX's Metal shaders only with full Xcode selected; otherwise take them from the matching Python package
    if [ ! -f $H/.build/release/mlx.metallib ]; then
      tmp=$(mktemp -d); uv pip install --quiet --target "$tmp" mlx==0.31.1 && cp "$tmp/mlx/lib/mlx.metallib" $H/.build/release/; rm -rf "$tmp"
    fi
    [ -f $H/weights/shape-small/model.fp16.safetensors ] || uvx --from huggingface_hub hf download zimengxiong/hunyuan3d-mlx-shape-small --local-dir $H/weights/shape-small
  ) || print "  WARNING: the shape model could not be set up. Worlds still build; objects stay painted boxes."
fi

print "The image models (about 20 GB) download the first time you build a world."
mkdir -p .setup && stamp > .setup/done
step "Done"
