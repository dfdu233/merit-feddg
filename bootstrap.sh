#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="smoke"
INCLUDE_GATED=0
INSTALL_SYSTEM=0
SKIP_DOWNLOAD=0
SKIP_TESTS=0
IGNORE_DISK_CHECK=0
TORCH_INDEX=""
MIN_FREE_GB=80

usage() {
  cat <<'USAGE'
Usage:
  ./bootstrap.sh smoke
  ./bootstrap.sh research-2d --include-gated --install-system
  ./bootstrap.sh --profile research-2d --include-gated [options]

Options:
  --profile NAME          smoke, open-small, or research-2d
  --include-gated         Download gated models after access is approved
  --install-system        Install missing apt packages on Debian/Ubuntu
  --torch-index URL       Explicit PyTorch wheel index, e.g. .../whl/cu130
  --min-free-gb N         Required free disk for research-2d (default: 80)
  --ignore-disk-check     Continue below the disk-space threshold
  --skip-download         Install dependencies without downloading assets
  --skip-tests            Skip unit tests and deterministic smoke comparison
  -h, --help              Show this help

HF authentication:
  export HF_TOKEN=hf_...  # or run: .venv/bin/hf auth login
USAGE
}

if [[ $# -gt 0 && "$1" != -* ]]; then
  PROFILE="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:?missing profile}"; shift 2 ;;
    --include-gated) INCLUDE_GATED=1; shift ;;
    --install-system) INSTALL_SYSTEM=1; shift ;;
    --torch-index) TORCH_INDEX="${2:?missing torch index URL}"; shift 2 ;;
    --min-free-gb) MIN_FREE_GB="${2:?missing disk threshold}"; shift 2 ;;
    --ignore-disk-check) IGNORE_DISK_CHECK=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PROFILE" in
  smoke|open-small|research-2d) ;;
  *) echo "Unknown profile: $PROFILE" >&2; exit 2 ;;
esac
if [[ ! "$MIN_FREE_GB" =~ ^[0-9]+$ ]]; then
  echo "--min-free-gb must be a non-negative integer." >&2
  exit 2
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer targets Linux. Use bootstrap.ps1 on Windows." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_EXE="$REPO_ROOT/.venv/bin/python"

if [[ $INSTALL_SYSTEM -eq 1 ]]; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "--install-system currently supports Debian/Ubuntu apt servers only." >&2
    exit 1
  fi
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3 python3-venv python3-pip git git-lfs \
      libgl1 libglib2.0-0 libsm6 libxext6
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3 python3-venv python3-pip git git-lfs \
      libgl1 libglib2.0-0 libsm6 libxext6
  else
    echo "Root privileges or sudo are required for --install-system." >&2
    exit 1
  fi
fi

required_commands=(git "$PYTHON_BIN")
missing_commands=()
for command_name in "${required_commands[@]}"; do
  command -v "$command_name" >/dev/null 2>&1 || missing_commands+=("$command_name")
done

can_create_venv=1
if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  "$PYTHON_BIN" -c "import venv" >/dev/null 2>&1 || can_create_venv=0
else
  can_create_venv=0
fi

if [[ ${#missing_commands[@]} -gt 0 || $can_create_venv -eq 0 ]]; then
  echo "Missing Linux prerequisites. Re-run with --install-system, or install:" >&2
  echo "  python3 python3-venv python3-pip git git-lfs libgl1 libglib2.0-0 libsm6 libxext6" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python 3.10+ is required, found {sys.version.split()[0]}")
PY

if [[ "$PROFILE" == "research-2d" && $IGNORE_DISK_CHECK -ne 1 ]]; then
  free_kb="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')"
  required_kb=$((MIN_FREE_GB * 1024 * 1024))
  if (( free_kb < required_kb )); then
    free_gb=$((free_kb / 1024 / 1024))
    echo "Insufficient free disk: ${free_gb} GiB; require ${MIN_FREE_GB} GiB." >&2
    echo "Use --min-free-gb N to set your threshold or --ignore-disk-check to override." >&2
    exit 1
  fi
fi

if [[ ! -x "$PYTHON_EXE" ]]; then
  "$PYTHON_BIN" -m venv "$REPO_ROOT/.venv"
fi

"$PYTHON_EXE" -m pip install --upgrade pip setuptools wheel

if [[ "$PROFILE" == "smoke" ]]; then
  "$PYTHON_EXE" -m pip install -e "$REPO_ROOT[dev]"
else
  if [[ -n "$TORCH_INDEX" ]]; then
    "$PYTHON_EXE" -m pip install torch torchvision --index-url "$TORCH_INDEX"
  else
    "$PYTHON_EXE" -m pip install torch torchvision
  fi
  "$PYTHON_EXE" -m pip install -e "$REPO_ROOT[research,dev]"
fi

if [[ $INCLUDE_GATED -eq 1 ]]; then
  if ! "$PYTHON_EXE" -c "from huggingface_hub import get_token; raise SystemExit(0 if get_token() else 1)"; then
    echo "Gated download requested, but Hugging Face authentication is missing." >&2
    echo "Export HF_TOKEN or run: $REPO_ROOT/.venv/bin/hf auth login" >&2
    exit 1
  fi
  "$PYTHON_EXE" -m pip install "git+https://github.com/Mahmoodlab/CONCH.git"
fi

export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

"$PYTHON_EXE" -m merit_feddg.cli doctor --root "$REPO_ROOT/artifacts"

if [[ $SKIP_DOWNLOAD -ne 1 ]]; then
  download_args=(--profile "$PROFILE" --root "$REPO_ROOT/artifacts")
  if [[ $INCLUDE_GATED -eq 1 ]]; then
    download_args+=(--include-gated)
  fi
  "$PYTHON_EXE" -m merit_feddg.cli download "${download_args[@]}"
  verify_args=(--profile "$PROFILE" --root "$REPO_ROOT/artifacts")
  if [[ $INCLUDE_GATED -eq 1 ]]; then
    verify_args+=(--include-gated)
  fi
  "$PYTHON_EXE" -m merit_feddg.cli verify-assets "${verify_args[@]}"
fi

if [[ $SKIP_TESTS -ne 1 ]]; then
  "$PYTHON_EXE" -m pytest "$REPO_ROOT/tests"
  "$PYTHON_EXE" -m merit_feddg.cli simulate \
    --config "$REPO_ROOT/configs/smoke.yaml" \
    --output "$REPO_ROOT/runs/smoke"
fi

echo "MERIT-FedDG is ready on Linux. Profile: $PROFILE"
if [[ "$PROFILE" == "research-2d" ]]; then
  echo "Next: edit configs/real_compare.example.yaml and run ./study.sh MANIFEST RUN_NAME"
fi
