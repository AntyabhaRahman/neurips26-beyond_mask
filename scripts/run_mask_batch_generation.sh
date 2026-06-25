#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root:
#   bash scripts/run_mask_batch_generation.sh
#   bash scripts/run_mask_batch_generation.sh --num-rows 25
#   bash scripts/run_mask_batch_generation.sh --test --num-rows 5

NUM_ROWS="${NUM_ROWS:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--num-rows)
      NUM_ROWS="$2"
      shift 2
      ;;
    --test)
      TEST_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

MODELS=(
  # "openai/gpt-5.4-mini"
  "anthropic/claude-opus-4-8"
  
  # "openai/another-model"
  # "anthropic/another-model"
)

CONFIGS=(
  "known_facts"
  "continuations"
  "disinformation"
  "doubling_down_known_facts"
  "provided_facts"
  "statistics"
)

ROW_ARGS=(--all)
if [[ -n "$NUM_ROWS" ]]; then
  ROW_ARGS=(--num_rows "$NUM_ROWS")
fi

if [[ "${TEST_RUN:-0}" == "1" ]]; then
  MODELS=("${MODELS[0]}")
  CONFIGS=("${CONFIGS[0]}")
  ROW_ARGS=(--num_rows "${NUM_ROWS:-1}")
fi

COMMON_ARGS=(
  --dataset "cais/MASK"
  --split "test"
  "${ROW_ARGS[@]}"
  --temperature 1.0
  --max_tokens 16384
  --lie_k 1
  --effort_level "high"
  --poll_interval 60
  --timeout_seconds 86400
)

if [[ "${SUMMARY_ONLY:-0}" != "1" ]]; then
  for config in "${CONFIGS[@]}"; do
    uv run python -m bench_setup.dataset_processing \
      "${COMMON_ARGS[@]}" \
      --config "$config" \
      --models "${MODELS[@]}"
  done
fi

if [[ "${RUN_SUMMARY:-1}" == "1" ]]; then
  SUMMARY_ARGS=(--splits full)
  if [[ "${PREPARE_METRICS:-1}" == "1" ]]; then
    SUMMARY_ARGS+=(--prepare-metrics --concurrency-limit "${JUDGE_CONCURRENCY_LIMIT:-50}")
  fi
  uv run python scripts/summarize_mask_results.py "${SUMMARY_ARGS[@]}"
fi
