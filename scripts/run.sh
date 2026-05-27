#!/usr/bin/env bash
set -euo pipefail

CONFIG="config/matrix.yaml"
SHARED_INPUT="assets/shared_input/shared_input.npz"
RESULTS_DIR="results"
POWER_INTERVAL_MS="100"
BACKEND=""
CELL=""
COMPUTE_UNIT=""
ATTENTION="NATIVE"
PRECISION="fp16"
RESOLUTION="512"
DRY_RUN=0
POWER_ENABLED=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-power)
      POWER_ENABLED=0
      shift
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --shared-input)
      SHARED_INPUT="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --power-interval-ms)
      POWER_INTERVAL_MS="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --cell)
      CELL="$2"
      shift 2
      ;;
    --compute-unit)
      COMPUTE_UNIT="$2"
      shift 2
      ;;
    --attention)
      ATTENTION="$2"
      shift 2
      ;;
    --precision)
      PRECISION="$2"
      shift 2
      ;;
    --resolution)
      RESOLUTION="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$RESULTS_DIR/raw"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
POWER_LOG="$RESULTS_DIR/raw/$RUN_ID-powermetrics.plist"

if [[ -n "$CELL" && -n "$BACKEND" ]]; then
  echo "Use either --cell or --backend/--compute-unit, not both." >&2
  exit 2
fi

if [[ -n "$BACKEND" && -z "$COMPUTE_UNIT" ]]; then
  echo "--compute-unit is required when --backend is provided." >&2
  exit 2
fi

if [[ "$POWER_ENABLED" -eq 1 && "$DRY_RUN" -eq 0 && "$(id -u)" -ne 0 ]]; then
  echo "Power sampling requires sudo because powermetrics needs privileged access." >&2
  exit 1
fi

if [[ -n "$CELL" ]]; then
  BENCHMARK_CMD=(
    uv run sdbench run-cell
    --config "$CONFIG"
    --shared-input "$SHARED_INPUT"
    --results-dir "$RESULTS_DIR"
    --cell "$CELL"
  )
elif [[ -n "$BACKEND" ]]; then
  BENCHMARK_CMD=(
    uv run sdbench run-cell
    --config "$CONFIG"
    --shared-input "$SHARED_INPUT"
    --results-dir "$RESULTS_DIR"
    --backend "$BACKEND"
    --compute-unit "$COMPUTE_UNIT"
    --attention "$ATTENTION"
    --precision "$PRECISION"
    --resolution "$RESOLUTION"
  )
else
  BENCHMARK_CMD=(
    uv run sdbench run
    --config "$CONFIG"
    --shared-input "$SHARED_INPUT"
    --results-dir "$RESULTS_DIR"
  )
fi

RUN_CMD=(caffeinate -dimsu "${BENCHMARK_CMD[@]}")
POWER_CMD=(powermetrics --samplers cpu_power,gpu_power,ane_power -i "$POWER_INTERVAL_MS" -f plist -o "$POWER_LOG")

if [[ "$DRY_RUN" -eq 1 ]]; then
  if [[ "$POWER_ENABLED" -eq 1 ]]; then
    printf 'Power command:'
    printf ' %s' "${POWER_CMD[@]}"
    printf '\n'
  else
    echo "Power sampling disabled"
  fi
  printf 'Benchmark command:'
  printf ' %s' "${RUN_CMD[@]}"
  printf '\n'
  exit 0
fi

POWER_PID=""
if [[ "$POWER_ENABLED" -eq 1 ]]; then
  "${POWER_CMD[@]}" &
  POWER_PID=$!
  trap 'if [[ -n "${POWER_PID:-}" ]]; then kill "$POWER_PID" 2>/dev/null || true; fi' EXIT
else
  echo "Power sampling disabled"
fi

"${RUN_CMD[@]}"

if [[ -n "$POWER_PID" ]]; then
  kill "$POWER_PID" 2>/dev/null || true
  wait "$POWER_PID" 2>/dev/null || true   # let powermetrics flush the plist before we read it
  echo "Raw power samples retained at $POWER_LOG"

  if [[ -n "$CELL" ]]; then
    DATA_PATH="$RESULTS_DIR/data/$CELL.jsonl"
  elif [[ -n "$BACKEND" ]]; then
    DATA_PATH=""   # resolved cell id is unknown to the shell for the --backend form
  else
    DATA_PATH="$RESULTS_DIR/data/results.jsonl"
  fi

  if [[ -n "$DATA_PATH" ]]; then
    uv run sdbench power \
      --input "$DATA_PATH" \
      --power-log "$POWER_LOG" \
      --config "$CONFIG" \
      --output-dir "$RESULTS_DIR/tables"
  else
    echo "Power post-processing skipped for --backend form; use --cell or the full matrix." >&2
  fi
fi
