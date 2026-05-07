#!/usr/bin/env bash
set -euo pipefail


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
PRIMARY_PYTHON="/home/minhae/anaconda3/envs/d1-sd/bin/python"
FALLBACK_PYTHON="/home/minhae/.conda/envs/d1-sd/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${PRIMARY_PYTHON}" ]]; then
    PYTHON_BIN="${PRIMARY_PYTHON}"
  elif [[ -x "${FALLBACK_PYTHON}" ]]; then
    PYTHON_BIN="${FALLBACK_PYTHON}"
  else
    PYTHON_BIN="python"
  fi
fi

# Edit these values as needed.
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Base}"
MODEL_LABEL="${MODEL_PATH##*/}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
TASK="${TASK:-math}"
GEN_LENGTH="${GEN_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/eval_results/${TASK}/baselines}"
SUFFIX="${SUFFIX:-newline_later_early_stop}"
FEW_SHOT="${FEW_SHOT:-0}"
SUBSAMPLE="${SUBSAMPLE:--1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-512}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
PROMPT_STYLE="${PROMPT_STYLE:-raw_long_cot}"
REASONING_PREFILL="${REASONING_PREFILL:-0}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-4096}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${TASK}_$(date +%Y%m%d_%H%M%S).log}"


if [[ ! -f "${ROOT_DIR}/eval.py" ]]; then
  echo "Evaluation entrypoint not found: ${ROOT_DIR}/eval.py" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

declare -a CHECKPOINT_PATH_LIST=()
declare -a RAW_CHECKPOINT_INPUTS=()

append_checkpoint_paths() {
  local input_path="$1"
  local -a discovered_checkpoints=()

  if [[ -z "${input_path}" ]]; then
    CHECKPOINT_PATH_LIST+=("")
    return
  fi

  if [[ -d "${input_path}" ]]; then
    while IFS= read -r checkpoint_dir; do
      discovered_checkpoints+=("${checkpoint_dir}")
    done < <(find "${input_path}" -type d -name 'checkpoint-*' | sort -V)

    if [[ "${#discovered_checkpoints[@]}" -gt 0 ]]; then
      CHECKPOINT_PATH_LIST+=("${discovered_checkpoints[@]}")
      return
    fi
  fi

  CHECKPOINT_PATH_LIST+=("${input_path}")
}

normalized_checkpoint_paths="${CHECKPOINT_PATH//$'\n'/ }"
read -r -a RAW_CHECKPOINT_INPUTS <<< "${normalized_checkpoint_paths}"
if [[ "${#RAW_CHECKPOINT_INPUTS[@]}" -eq 0 ]]; then
  CHECKPOINT_PATH_LIST=("")
else
  for checkpoint_input in "${RAW_CHECKPOINT_INPUTS[@]}"; do
    append_checkpoint_paths "${checkpoint_input}"
  done
fi


echo "Running ${TASK}(test) evaluation"
echo "  MODEL_PATH     : ${MODEL_PATH}"
echo "  CHECKPOINT_PATH: ${CHECKPOINT_PATH:-<none>}"
echo "  NUM_CHECKPOINTS: ${#CHECKPOINT_PATH_LIST[@]}"
echo "  GEN_LENGTH     : ${GEN_LENGTH}"
echo "  BATCH_SIZE     : ${BATCH_SIZE}"
echo "  OUTPUT_DIR     : ${OUTPUT_DIR}"
echo "  SUBSAMPLE      : ${SUBSAMPLE}"
echo "  DIFFUSION_STEPS: ${DIFFUSION_STEPS}"
echo "  BLOCK_LENGTH   : ${BLOCK_LENGTH}"
echo "  PROMPT_STYLE   : ${PROMPT_STYLE}"
echo "  REASONING_PREFILL: ${REASONING_PREFILL}"
echo "  MAX_CONTEXT_LENGTH: ${MAX_CONTEXT_LENGTH}"
echo "  LOG_FILE       : ${LOG_FILE}"
run_eval_for_checkpoint() {
  local checkpoint_path="$1"

  local -a CMD=(
    "${PYTHON_BIN}"
    "${ROOT_DIR}/eval.py"
    --dataset "${TASK}"
    --model_path "${MODEL_PATH}"
    --batch_size "${BATCH_SIZE}"
    --gen_length "${GEN_LENGTH}"
    --few_shot "${FEW_SHOT}"
    --suffix "${SUFFIX}"
    --output_dir "${OUTPUT_DIR}"
    --subsample "${SUBSAMPLE}"
    --diffusion_steps "${DIFFUSION_STEPS}"
    --block_length "${BLOCK_LENGTH}"
    --prompt_style "${PROMPT_STYLE}"
    --max_context_length "${MAX_CONTEXT_LENGTH}"
  )

  if [[ "${REASONING_PREFILL}" == "1" ]]; then
    CMD+=(--add_reasoning)
  else
    CMD+=(--no_add_reasoning)
  fi

  if [[ -n "${checkpoint_path}" ]]; then
    CMD+=(--checkpoint_path "${checkpoint_path}")
  fi

  echo "----------------------------------------"
  echo "[eval] checkpoint_path=${checkpoint_path:-<none>}"
  "${CMD[@]}"
}


for checkpoint_path in "${CHECKPOINT_PATH_LIST[@]}"; do
  run_eval_for_checkpoint "${checkpoint_path}"
done
