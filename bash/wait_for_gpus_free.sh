#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="0,1,2,3"
TIMEOUT_SECONDS=1800
INTERVAL_SECONDS=5

usage() {
  cat <<'EOF'
Usage: bash/wait_for_gpus_free.sh [options]

Options:
  --gpus IDS                  GPU ids to check, comma or space separated. Default: 0,1,2,3
  --timeout-seconds SECONDS   Total wait timeout. Default: 1800
  --interval-seconds SECONDS  Probe interval. Default: 5
  -h, --help                  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPU_IDS="$2"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --interval-seconds)
      INTERVAL_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not available; cannot verify GPU cleanup." >&2
  exit 1
fi

IFS=', ' read -r -a GPUS <<< "${GPU_IDS}"
deadline=$((SECONDS + TIMEOUT_SECONDS))

gpu_processes() {
  local gpu="$1"

  nvidia-smi \
    --id="${gpu}" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d'
}

echo "Waiting for GPU(s) ${GPU_IDS} to have no compute processes."

while true; do
  busy=()
  details=()

  for gpu in "${GPUS[@]}"; do
    [[ -z "${gpu}" ]] && continue

    processes="$(gpu_processes "${gpu}")" || {
      echo "Failed to query GPU ${gpu} with nvidia-smi." >&2
      exit 1
    }

    if [[ -n "${processes}" ]]; then
      busy+=("${gpu}")
      while IFS= read -r line; do
        details+=("GPU ${gpu}: ${line}")
      done <<< "${processes}"
    fi
  done

  if [[ ${#busy[@]} -eq 0 ]]; then
    echo "GPU(s) ${GPU_IDS} are free."
    exit 0
  fi

  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for GPU(s) to become free: ${busy[*]}" >&2
    printf '%s\n' "${details[@]}" >&2
    exit 1
  fi

  echo "Still waiting on GPU(s): ${busy[*]}"
  printf '%s\n' "${details[@]}"
  sleep "${INTERVAL_SECONDS}"
done
