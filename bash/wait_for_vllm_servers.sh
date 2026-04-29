#!/usr/bin/env bash
set -euo pipefail

HOST="127.0.0.1"
BASE_PORT=8001
NUM_SERVERS=4
MODEL=""
API_KEY="EMPTY"
TIMEOUT_SECONDS=3600
INTERVAL_SECONDS=10

usage() {
  cat <<'EOF'
Usage: bash/wait_for_vllm_servers.sh [options]

Options:
  --host HOST                 Host to probe. Default: 127.0.0.1
  --base-port PORT            First vLLM port. Default: 8001
  --num-servers N             Number of consecutive ports to probe. Default: 4
  --model MODEL               Expected served model name in /v1/models.
  --api-key KEY               API key for vLLM. Default: EMPTY
  --timeout-seconds SECONDS   Total wait timeout. Default: 3600
  --interval-seconds SECONDS  Probe interval. Default: 10
  -h, --help                  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --base-port)
      BASE_PORT="$2"
      shift 2
      ;;
    --num-servers)
      NUM_SERVERS="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --api-key)
      API_KEY="$2"
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

check_server() {
  local port="$1"
  local url="http://${HOST}:${port}/v1/models"
  local response

  response="$(
    curl -fsS \
      --max-time 5 \
      -H "Authorization: Bearer ${API_KEY}" \
      "${url}" 2>/dev/null
  )" || return 1

  if [[ -n "${MODEL}" ]] && ! grep -Fq "${MODEL}" <<< "${response}"; then
    return 1
  fi
}

deadline=$((SECONDS + TIMEOUT_SECONDS))
echo "Waiting for ${NUM_SERVERS} vLLM server(s) on ${HOST}:${BASE_PORT}-$((BASE_PORT + NUM_SERVERS - 1))"
if [[ -n "${MODEL}" ]]; then
  echo "Expected served model: ${MODEL}"
fi

while true; do
  not_ready=()

  for offset in $(seq 0 $((NUM_SERVERS - 1))); do
    port=$((BASE_PORT + offset))
    if ! check_server "${port}"; then
      not_ready+=("${port}")
    fi
  done

  if [[ ${#not_ready[@]} -eq 0 ]]; then
    echo "All vLLM servers are ready."
    exit 0
  fi

  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for vLLM server(s): ${not_ready[*]}" >&2
    echo "Check tmux session logs, for example: tmux attach -t gpt20b" >&2
    exit 1
  fi

  echo "Still waiting on port(s): ${not_ready[*]}"
  sleep "${INTERVAL_SECONDS}"
done
