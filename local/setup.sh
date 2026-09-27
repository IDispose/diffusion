#!/usr/bin/env bash
# Build a local environment that can run RFdiffusion, without Colab.
#
# Why this script exists: RFdiffusion needs DGL, and DGL's last release (2.4.0) only
# ships wheels for CPython 3.8-3.12 built against torch 2.4. So we pin Python 3.11 and
# let the bundled graphbolt library decide the torch version.
#
#   ./local/setup.sh            # auto-detect: CUDA wheels on Linux+NVIDIA, else CPU
#   ./local/setup.sh cpu        # force the CPU build
#
# Creates ./RFdiffusion (clone), ./rfdiff-env (venv) and ./RFdiffusion/models (weights).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
ENV_DIR="$ROOT/rfdiff-env"
PY="$ENV_DIR/bin/python"

MODE="${1:-auto}"
if [ "$MODE" = "auto" ]; then
  if [ "$(uname -s)" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    MODE=cuda
  else
    MODE=cpu
  fi
fi
echo "==> build mode: $MODE"

command -v uv >/dev/null 2>&1 || {
  echo "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}

[ -d "$ROOT/RFdiffusion" ] || {
  echo "==> cloning RFdiffusion"
  git clone -q https://github.com/sokrypton/RFdiffusion.git "$ROOT/RFdiffusion"
}

echo "==> creating Python 3.11 venv at $ENV_DIR"
uv venv -q --seed --python 3.11 "$ENV_DIR"
export VIRTUAL_ENV="$ENV_DIR"
# the CUDA DGL wheel is 350 MB; uv's 30 s default HTTP timeout is not enough on a slow link
export UV_HTTP_TIMEOUT=600
pip_install() { uv pip install -q "$@"; }

# DGL first, with --no-deps: its metadata asks for an unpinned torch, which resolves to
# the newest release and breaks the ABI its compiled extensions were built against.
echo "==> installing DGL"
if [ "$MODE" = "cuda" ]; then
  # by direct wheel URL, not `-f .../repo.html`: data.dgl.ai is a plain directory listing
  # with one wheel per (python, torch, cuda) combo, and resolving "dgl==2.4.0+cu124" out of
  # it needs the installer to accept both a local version specifier and the legacy
  # manylinux1 tag. Fall back to the venv's pip if uv will not take the wheel.
  DGL_WHEEL=https://data.dgl.ai/wheels/torch-2.4/cu124/dgl-2.4.0%2Bcu124-cp311-cp311-manylinux1_x86_64.whl
  pip_install --no-deps "$DGL_WHEEL" || {
    echo "    uv could not install the DGL wheel; retrying with pip"
    "$ENV_DIR/bin/python" -m pip install -q --no-deps "$DGL_WHEEL"
  }
else
  pip_install --no-deps "dgl==2.2.0"
fi

# DGL bundles one prebuilt graphbolt library per torch version. Pin torch to the newest
# one present, or `import dgl` dies with "Cannot find DGL C++ graphbolt library".
TORCH_VER="$("$PY" - <<'PYEOF'
import glob, os, sysconfig
libs = glob.glob(os.path.join(sysconfig.get_paths()["purelib"],
                              "dgl", "graphbolt", "libgraphbolt_pytorch_*"))
vers = [os.path.basename(p).split("libgraphbolt_pytorch_")[1].rsplit(".", 1)[0] for p in libs]
if not vers:
    raise SystemExit("no bundled graphbolt library -- wrong dgl wheel?")
print(sorted(vers, key=lambda v: tuple(int(x) for x in v.split(".")))[-1])
PYEOF
)"
echo "==> pinning torch==$TORCH_VER to match DGL's graphbolt build"
if [ "$MODE" = "cuda" ]; then
  pip_install "torch==$TORCH_VER" --index-url https://download.pytorch.org/whl/cu124
else
  pip_install "torch==$TORCH_VER"
fi

# DGL's import chain reaches torchdata.datapipes, yaml, packaging and setuptools.
echo "==> installing DGL runtime deps"
pip_install "numpy<2" "scipy<1.14" networkx requests tqdm psutil pandas pyyaml \
            packaging setuptools pydantic "torchdata==0.8.0"

echo "==> installing RFdiffusion + SE3Transformer"
pip_install hydra-core omegaconf icecream pyrsistent decorator
# --no-deps on e3nn keeps it from resolving its unpinned `torch` requirement and
# churning the pin above, so opt_einsum_fx's own dep on opt_einsum is named here.
pip_install --no-deps "e3nn==0.5.5" opt_einsum_fx
pip_install opt_einsum
pip_install "git+https://github.com/NVIDIA/dllogger#egg=dllogger"
pip_install --no-deps "$ROOT/RFdiffusion/env/SE3Transformer"

if [ "$MODE" = "cpu" ]; then
  echo "==> patching SE3Transformer NVTX markers for a CPU-only torch"
  "$PY" "$ROOT/local/patch_nvtx.py"
fi

echo "==> verifying"
DGLBACKEND=pytorch "$PY" - <<'PYEOF'
import sys, torch, dgl
from dgl.ops import edge_softmax, e_dot_v, copy_e_sum   # SE3Transformer needs all three
import se3_transformer, e3nn, hydra
print(f"    python {sys.version.split()[0]} | torch {torch.__version__} "
      f"| dgl {dgl.__version__} | cuda {torch.cuda.is_available()}")
PYEOF

mkdir -p "$ROOT/RFdiffusion/models"
download() {  # url, dest
  [ -s "$2" ] && { echo "    have $(basename "$2")"; return; }
  echo "    downloading $(basename "$2")"
  curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}
echo "==> downloading RFdiffusion weights (~1.5 GB)"
BASE=http://files.ipd.uw.edu/pub/RFdiffusion
download $BASE/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt          "$ROOT/RFdiffusion/models/Base_ckpt.pt"
download $BASE/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt  "$ROOT/RFdiffusion/models/Complex_base_ckpt.pt"
download $BASE/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt  "$ROOT/RFdiffusion/models/Complex_beta_ckpt.pt"

cat <<MSG

Done. Run a design with:

  ./local/run.sh inference.output_prefix=outputs/test 'contigmap.contigs=[100-100]' diffuser.T=50

The first run spends a few minutes computing the IGSO(3) schedules
(files.ipd.uw.edu/krypton/schedules.zip, the precomputed cache, is 404 now);
they are then cached in ./schedules.
MSG
