#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash train/tree/train_bundles_from_results.sh <experiment> [options]

Options:
  --results-root <dir>   Root directory that contains tuning experiments.
                         Default: train/tree/results
  --bundle-root <dir>    Root directory for exported bundles.
                         Default: train/tree/bundles
  --seed <int>           Random seed for final FIGS training. Default: 0
  --rf-jobs <int>        Unused by FIGS. Kept only for CLI compatibility. Default: 1
  --conda-bin <path>     Path to conda executable.
                         Default: /data1/tianang/anaconda3/condabin/conda
  --conda-env <name>     Conda env used for training.
                         Default on node002: vllm
  --stop-on-error        Stop immediately if one task fails.
  --dry-run              Print the commands without executing training.
  -h, --help             Show this help message.

This script expects:
  <results-root>/<experiment>/<task>/<feature_set>/best_params.json

It will:
  1. Re-train one final FIGS model per best_params.json
  2. Keep the full training artifacts in train/tree/results/<experiment>/...
  3. Export a compact bundle copy to:
     <bundle-root>/<experiment>/<task>/<feature_set>/
EOF
}

RESULTS_ROOT="train/tree/results"
BUNDLE_ROOT="train/tree/bundles"
SEED="0"
RF_JOBS="1"
CONDA_BIN="/data1/tianang/anaconda3/condabin/conda"
CONDA_ENV=""
STOP_ON_ERROR="0"
DRY_RUN="0"
EXPERIMENT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --bundle-root)
      BUNDLE_ROOT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --rf-jobs)
      RF_JOBS="$2"
      shift 2
      ;;
    --conda-bin)
      CONDA_BIN="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --stop-on-error)
      STOP_ON_ERROR="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [ -n "$EXPERIMENT" ]; then
        echo "Only one experiment name may be provided." >&2
        usage >&2
        exit 2
      fi
      EXPERIMENT="$1"
      shift
      ;;
  esac
done

if [ -z "$EXPERIMENT" ]; then
  usage >&2
  exit 2
fi

if [ -z "$CONDA_ENV" ] && [ "$(hostname)" = "node002" ]; then
  CONDA_ENV="vllm"
fi

RESULTS_DIR="${RESULTS_ROOT%/}/$EXPERIMENT"
BUNDLE_EXPERIMENT_DIR="${BUNDLE_ROOT%/}/$EXPERIMENT"

if [ ! -d "$RESULTS_DIR" ]; then
  echo "Results directory not found: $RESULTS_DIR" >&2
  exit 1
fi

mapfile -t PARAMS_FILES < <(find "$RESULTS_DIR" -name best_params.json | sort)
if [ "${#PARAMS_FILES[@]}" -eq 0 ]; then
  echo "No best_params.json files found under: $RESULTS_DIR" >&2
  exit 1
fi

mkdir -p "$BUNDLE_EXPERIMENT_DIR"
INDEX_TSV="$BUNDLE_EXPERIMENT_DIR/bundle_index.tsv"
FAILURES_LOG="$BUNDLE_EXPERIMENT_DIR/failures.log"
printf "task\tfeature_set\tparams_json\ttrain_summary_json\tbundle_best_params_json\tbundle_train_summary_json\tbundle_model_pkl\n" >"$INDEX_TSV"
: >"$FAILURES_LOG"

run_training_command() {
  if [ -n "$CONDA_ENV" ]; then
    "$CONDA_BIN" run -n "$CONDA_ENV" "$@"
  else
    "$@"
  fi
}

status=0

for params_json in "${PARAMS_FILES[@]}"; do
  task="$(basename "$(dirname "$(dirname "$params_json")")")"
  feature_set="$(basename "$(dirname "$params_json")")"
  task_results_dir="$RESULTS_DIR/$task/$feature_set"
  bundle_task_dir="$BUNDLE_EXPERIMENT_DIR/$task/$feature_set"

  mapfile -t metadata_lines < <(
    python -c '
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload.get("dataset_root", ""))
print(payload.get("train_split", "train"))
print(payload.get("valid_split", "valid"))
for path in payload.get("feature_config_paths", []):
    print(path)
' "$params_json"
  )

  data_root="${metadata_lines[0]}"
  train_split="${metadata_lines[1]}"
  valid_split="${metadata_lines[2]}"
  feature_configs=("${metadata_lines[@]:3}")

  cmd=(python train/tree/train_random_forest.py
    --task "$task"
    --params-json "$params_json"
    --data-root "$data_root"
    --train-split "$train_split"
    --valid-split "$valid_split"
    --seed "$SEED"
    --rf-jobs "$RF_JOBS"
  )

  for feature_config in "${feature_configs[@]}"; do
    cmd+=(--feature-config "$feature_config")
  done

  echo "Training bundle for task=$task feature_set=$feature_set"
  if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    continue
  fi

  if ! run_training_command "${cmd[@]}"; then
    echo "$task"$'\t'"$feature_set"$'\t'"$params_json" >>"$FAILURES_LOG"
    status=1
    if [ "$STOP_ON_ERROR" = "1" ]; then
      exit 1
    fi
    continue
  fi

  mkdir -p "$bundle_task_dir"
  cp "$params_json" "$bundle_task_dir/best_params.json"
  cp "$task_results_dir/train_summary.json" "$bundle_task_dir/train_summary.json"
  cp "$task_results_dir/model_bundle.pkl" "$bundle_task_dir/model_bundle.pkl"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$task" \
    "$feature_set" \
    "$params_json" \
    "$task_results_dir/train_summary.json" \
    "$bundle_task_dir/best_params.json" \
    "$bundle_task_dir/train_summary.json" \
    "$bundle_task_dir/model_bundle.pkl" \
    >>"$INDEX_TSV"
done

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run completed for ${#PARAMS_FILES[@]} tasks."
  exit 0
fi

if [ "$status" -ne 0 ]; then
  echo "Some tasks failed. See $FAILURES_LOG" >&2
  exit "$status"
fi

echo "Exported ${#PARAMS_FILES[@]} bundles to $BUNDLE_EXPERIMENT_DIR"
