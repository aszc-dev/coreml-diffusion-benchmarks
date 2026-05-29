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
SAMPLER_META="$RESULTS_DIR/raw/$RUN_ID-sampler.meta"

# Propagate to the harness so its environment manifest and the plist filename
# agree on the same run_id (R11.12, R11.14).
export SDBENCH_RUN_ID="$RUN_ID"

# Sidecar: powermetrics version banner + sudo-cache state at run start. Captured
# even on --dry-run so a contributor's bundle is self-describing.
{
  echo "run_id=$RUN_ID"
  echo "config=$CONFIG"
  echo "results_dir=$RESULTS_DIR"
  echo "interval_ms=$POWER_INTERVAL_MS"
  echo "power_enabled=$POWER_ENABLED"
  echo "macos_build=$(sw_vers -buildVersion 2>/dev/null || echo unknown)"
  echo "powermetrics_version=$(powermetrics --help 2>&1 | head -n 1 || echo unknown)"
  if sudo -n true 2>/dev/null; then
    echo "sudo_cached=1"
  else
    echo "sudo_cached=0"
  fi
  echo "started_at_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$SAMPLER_META"

if [[ -n "$CELL" && -n "$BACKEND" ]]; then
  echo "Use either --cell or --backend/--compute-unit, not both." >&2
  exit 2
fi

if [[ -n "$BACKEND" && -z "$COMPUTE_UNIT" ]]; then
  echo "--compute-unit is required when --backend is provided." >&2
  exit 2
fi

if [[ "$POWER_ENABLED" -eq 1 && "$DRY_RUN" -eq 0 && "$(id -u)" -ne 0 ]]; then
  # powermetrics needs root. If we are not root AND sudo has no cached
  # credentials, prompt once: a contributor who doesn't want to type a password
  # can answer "n" and the run proceeds without power. Either way the manifest
  # records whether power was measured (R11.12).
  if sudo -n true 2>/dev/null; then
    : # sudo credentials already cached — nothing to do, sampler will run.
  else
    echo "" >&2
    echo "[sdbench] powermetrics needs sudo for per-engine power sampling (R6.1)." >&2
    echo "[sdbench] You can:" >&2
    echo "          (a) re-run with 'sudo $0 ...' for power figures, or" >&2
    echo "          (b) continue WITHOUT power (latency / size / equivalence still measured)." >&2
    echo "" >&2
    if [[ -t 0 ]]; then
      read -r -p "[sdbench] Authorize sudo for powermetrics now? [y/N] " reply
    else
      reply=""
    fi
    case "$reply" in
      y|Y|yes|YES)
        if ! sudo -v; then
          echo "[sdbench] sudo not granted — continuing WITHOUT power measurement." >&2
          POWER_ENABLED=0
        fi
        ;;
      *)
        echo "[sdbench] Continuing WITHOUT power measurement (gpu_power_w / ane_power_w will be null)." >&2
        POWER_ENABLED=0
        ;;
    esac
    # Refresh the sidecar so contributors reading it see what actually ran.
    echo "power_enabled=$POWER_ENABLED" >> "$SAMPLER_META"
  fi
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
    uv run sdbench run-matrix
    --config "$CONFIG"
    --shared-input "$SHARED_INPUT"
    --results-dir "$RESULTS_DIR"
  )
fi

RUN_CMD=(caffeinate -dimsu "${BENCHMARK_CMD[@]}")
# When the wrapper is launched without `sudo`, powermetrics still needs root.
# We've already authorized `sudo -v` above (interactive prompt or cached creds),
# so re-enter with `sudo -n` to use the cached ticket non-interactively. When the
# wrapper is itself launched via `sudo ./scripts/run.sh`, EUID is 0 and the
# prefix is unnecessary.
if [[ "$(id -u)" -eq 0 ]]; then
  POWER_CMD=(powermetrics --samplers cpu_power,gpu_power,ane_power -i "$POWER_INTERVAL_MS" -f plist -o "$POWER_LOG")
else
  POWER_CMD=(sudo -n powermetrics --samplers cpu_power,gpu_power,ane_power -i "$POWER_INTERVAL_MS" -f plist -o "$POWER_LOG")
fi

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

# A sudo-launched sampler runs as root; sending it SIGTERM as a normal user
# would fail with EPERM. Use the cached sudo ticket for the kill, too.
if [[ "$(id -u)" -eq 0 ]]; then
  KILL_PREFIX=()
else
  KILL_PREFIX=(sudo -n)
fi

POWER_PID=""
if [[ "$POWER_ENABLED" -eq 1 ]]; then
  "${POWER_CMD[@]}" &
  POWER_PID=$!
  trap 'if [[ -n "${POWER_PID:-}" ]]; then "${KILL_PREFIX[@]}" kill "$POWER_PID" 2>/dev/null || true; fi' EXIT
else
  echo "Power sampling disabled"
fi

"${RUN_CMD[@]}"

if [[ -n "$POWER_PID" ]]; then
  "${KILL_PREFIX[@]}" kill "$POWER_PID" 2>/dev/null || true
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
