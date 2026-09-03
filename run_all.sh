#!/usr/bin/env bash
set -Eeuo pipefail

PRESET="canary"
LIMIT_PER_DOMAIN=""
QUESTIONS_PER_IMAGE=1
MIRROR="cn"
RUN_NAME=""
INSTALL_SYSTEM=0
INCLUDE_GATED=1
FORCE_EXTRACT=0

usage() {
  cat <<'USAGE'
Usage:
  export HF_TOKEN=hf_...
  ./run_all.sh [options]

One command installs the environment, downloads every registered model and dataset,
prepares the public benchmark, extracts model evidence, and runs predicted-router plus
oracle-router comparisons.

Options:
  --preset NAME            canary (8 rows/domain) or paper (all compatible images)
  --limit-per-domain N     Override the preset; 0 means no limit
  --questions-per-image N  Maximum QA rows per unique image; 0 means all (default: 1)
  --mirror MODE            cn, auto, or global (default: cn)
  --run-name NAME          Output name below cache/ and runs/
  --install-system         Install Debian/Ubuntu system prerequisites
  --force-extract          Ignore a reusable evidence cache and rerun all models
  --without-gated          Skip gated assets; real pathology extraction will be unavailable
  -h, --help               Show this help

The generated public domains are deterministic image-level proxy partitions. They test
the mechanism and software path, but are not evidence of cross-hospital generalization.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) PRESET="${2:?missing preset}"; shift 2 ;;
    --limit-per-domain) LIMIT_PER_DOMAIN="${2:?missing limit}"; shift 2 ;;
    --questions-per-image) QUESTIONS_PER_IMAGE="${2:?missing count}"; shift 2 ;;
    --mirror) MIRROR="${2:?missing mirror mode}"; shift 2 ;;
    --run-name) RUN_NAME="${2:?missing run name}"; shift 2 ;;
    --install-system) INSTALL_SYSTEM=1; shift ;;
    --force-extract) FORCE_EXTRACT=1; shift ;;
    --without-gated) INCLUDE_GATED=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PRESET" in
  canary) DEFAULT_LIMIT=8 ;;
  paper) DEFAULT_LIMIT=0 ;;
  *) echo "Unknown preset: $PRESET" >&2; exit 2 ;;
esac
case "$MIRROR" in
  cn|auto|global) ;;
  *) echo "Unknown mirror mode: $MIRROR" >&2; exit 2 ;;
esac
if [[ -z "$LIMIT_PER_DOMAIN" ]]; then
  LIMIT_PER_DOMAIN="$DEFAULT_LIMIT"
fi
if [[ ! "$LIMIT_PER_DOMAIN" =~ ^[0-9]+$ || ! "$QUESTIONS_PER_IMAGE" =~ ^[0-9]+$ ]]; then
  echo "Limits must be non-negative integers." >&2
  exit 2
fi
if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="public-$PRESET"
fi
if [[ $INCLUDE_GATED -eq 1 && -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for the default gated CONCH pathology expert." >&2
  echo "Accept the model terms, export a read token, then rerun this command." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXE="$REPO_ROOT/.venv/bin/python"
ARTIFACTS="$REPO_ROOT/artifacts"
PREPARED="$REPO_ROOT/data/$RUN_NAME"
MANIFEST="$PREPARED/manifest.jsonl"
COMPARE_CONFIG="$PREPARED/compare.yaml"
EVIDENCE="$REPO_ROOT/cache/$RUN_NAME.predicted.jsonl"
ORACLE_EVIDENCE="$REPO_ROOT/cache/$RUN_NAME.oracle.jsonl"
RUN_ROOT="$REPO_ROOT/runs/$RUN_NAME"

bootstrap_args=(--profile research-2d --mirror "$MIRROR")
if [[ $INCLUDE_GATED -eq 1 ]]; then
  bootstrap_args+=(--include-gated)
fi
if [[ $INSTALL_SYSTEM -eq 1 ]]; then
  bootstrap_args+=(--install-system)
fi
"$REPO_ROOT/bootstrap.sh" "${bootstrap_args[@]}"

if [[ $INCLUDE_GATED -ne 1 ]]; then
  echo "The complete configured experiment requires gated CONCH; stopping after public downloads." >&2
  echo "Rerun without --without-gated after access is approved." >&2
  exit 1
fi

"$PYTHON_EXE" -m merit_feddg.cli prepare-public \
  --config "$REPO_ROOT/configs/public_benchmarks.yaml" \
  --artifacts "$ARTIFACTS" \
  --output "$PREPARED" \
  --limit-per-domain "$LIMIT_PER_DOMAIN" \
  --questions-per-image "$QUESTIONS_PER_IMAGE"

"$PYTHON_EXE" -m merit_feddg.cli audit-split \
  --manifest "$MANIFEST" \
  --held-out vqa_rad-target \
  --held-out path_vqa-target \
  --held-out oct_summary-target \
  --held-out slake_xray-target

if [[ $FORCE_EXTRACT -eq 1 || ! -s "$EVIDENCE" || "$MANIFEST" -nt "$EVIDENCE" ]]; then
  "$PYTHON_EXE" -m merit_feddg.cli extract \
    --manifest "$MANIFEST" \
    --config "$REPO_ROOT/configs/real_2d.yaml" \
    --artifacts "$ARTIFACTS" \
    --output "$EVIDENCE"
else
  echo "Reusing newer evidence cache: $EVIDENCE"
fi

"$PYTHON_EXE" -m merit_feddg.cli compare \
  --input "$EVIDENCE" \
  --config "$COMPARE_CONFIG" \
  --output "$RUN_ROOT/predicted-router"

"$PYTHON_EXE" -m merit_feddg.cli oracle-cache \
  --input "$EVIDENCE" \
  --output "$ORACLE_EVIDENCE"

"$PYTHON_EXE" -m merit_feddg.cli compare \
  --input "$ORACLE_EVIDENCE" \
  --config "$COMPARE_CONFIG" \
  --output "$RUN_ROOT/oracle-router"

echo "All configured public experiments completed."
echo "Prepared data: $PREPARED"
echo "Predicted routing: $RUN_ROOT/predicted-router/comparison.md"
echo "Oracle routing: $RUN_ROOT/oracle-router/comparison.md"
echo "Protocol warning: proxy image partitions are not cross-hospital FedDG evidence."
