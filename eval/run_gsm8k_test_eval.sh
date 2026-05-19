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
# START_INDEX / END_INDEX are 0-based inclusive. Use END_INDEX=-1 to run through the end.
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Base}"
MODEL_LABEL="${MODEL_PATH##*/}"
DATASET_LABEL="${DATASET_LABEL:-Math-NoCoT-format-4096}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/minhae/diffusion/DLM_SFT/checkpoints/Math-CoT-NoCoT-20k-format-4096/LLaDA-8B-Base/ADD_EOS/BS16_math_ff_4096_SFT_tgtnoncot_format_ep8_20260513_164040/checkpoint-752}"
TASK="${TASK:-math}"
GEN_LENGTH="${GEN_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/eval_results/${TASK}/full/${MODEL_LABEL}/${DATASET_LABEL}/SFT_tgtnoncot_format}"
SUFFIX="${SUFFIX:-}"
SUBSAMPLE="${SUBSAMPLE:--1}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-512}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
PROMPT_STYLE="${PROMPT_STYLE:-format}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-4096}"
NEWLINE_LATER="${NEWLINE_LATER:-1}"
EARLYSTOP="${EARLYSTOP:-0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-}"


if [[ ! -f "${ROOT_DIR}/eval.py" ]]; then
  echo "Evaluation entrypoint not found: ${ROOT_DIR}/eval.py" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

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

build_effective_suffix() {
  local effective_suffix="${SUFFIX}"
  if [[ "${NEWLINE_LATER}" == "1" ]]; then
    if [[ -n "${effective_suffix}" ]]; then
      effective_suffix+="_newline_later"
    else
      effective_suffix="newline_later"
    fi
  fi

  if [[ "${EARLYSTOP}" == "1" ]]; then
    if [[ -n "${effective_suffix}" ]]; then
      effective_suffix+="_earlystop"
    else
      effective_suffix="earlystop"
    fi
  fi

  printf '%s\n' "${effective_suffix}"
}

build_index_range_label() {
  if [[ "${START_INDEX}" == "0" && "${END_INDEX}" == "-1" ]]; then
    printf '\n'
    return
  fi

  if [[ "${END_INDEX}" == "-1" ]]; then
    printf 'idx%s-end\n' "${START_INDEX}"
    return
  fi

  printf 'idx%s-%s\n' "${START_INDEX}" "${END_INDEX}"
}

build_runname() {
  local checkpoint_path="$1"
  local effective_suffix="$2"
  local range_label="$3"
  local runname

  if [[ -n "${checkpoint_path}" ]]; then
    runname="$(basename "${checkpoint_path}")"
  else
    runname="${MODEL_LABEL}"
  fi

  if [[ -n "${range_label}" ]]; then
    runname+="_${range_label}"
  fi

  if [[ -n "${effective_suffix}" ]]; then
    runname+="_${effective_suffix}"
  fi

  runname+="_${GEN_LENGTH}_${DIFFUSION_STEPS}"

  printf '%s\n' "${runname}"
}

run_eval_for_checkpoint() {
  local checkpoint_path="$1"
  local effective_suffix
  effective_suffix="$(build_effective_suffix)"
  local range_label
  range_label="$(build_index_range_label)"
  local runname
  runname="$(build_runname "${checkpoint_path}" "${effective_suffix}" "${range_label}")"
  local log_file="${LOG_FILE:-${LOG_DIR}/${runname}_$(date +%Y%m%d_%H%M%S).log}"

  local -a CMD=(
    "${PYTHON_BIN}"
    "${ROOT_DIR}/eval.py"
    --dataset "${TASK}"
    --model_path "${MODEL_PATH}"
    --batch_size "${BATCH_SIZE}"
    --gen_length "${GEN_LENGTH}"
    --suffix "${effective_suffix}"
    --output_dir "${OUTPUT_DIR}"
    --subsample "${SUBSAMPLE}"
    --start_index "${START_INDEX}"
    --end_index "${END_INDEX}"
    --diffusion_steps "${DIFFUSION_STEPS}"
    --block_length "${BLOCK_LENGTH}"
    --prompt_style "${PROMPT_STYLE}"
    --max_context_length "${MAX_CONTEXT_LENGTH}"
  )

  if [[ "${NEWLINE_LATER}" == "1" ]]; then
    CMD+=(--newline_later)
  fi

  if [[ "${EARLYSTOP}" == "1" ]]; then
    CMD+=(--earlystop)
  fi

  if [[ -n "${checkpoint_path}" ]]; then
    CMD+=(--checkpoint_path "${checkpoint_path}")
  fi

  {
    echo "Running ${TASK}(test) evaluation"
    echo "  RUN_NAME       : ${runname}"
    echo "  MODEL_PATH     : ${MODEL_PATH}"
    echo "  CHECKPOINT_PATH: ${checkpoint_path:-<none>}"
    echo "  NUM_CHECKPOINTS: ${#CHECKPOINT_PATH_LIST[@]}"
    echo "  GEN_LENGTH     : ${GEN_LENGTH}"
    echo "  BATCH_SIZE     : ${BATCH_SIZE}"
    echo "  OUTPUT_DIR     : ${OUTPUT_DIR}"
    echo "  SUBSAMPLE      : ${SUBSAMPLE}"
    echo "  START_INDEX    : ${START_INDEX}"
    echo "  END_INDEX      : ${END_INDEX}"
    echo "  DIFFUSION_STEPS: ${DIFFUSION_STEPS}"
    echo "  BLOCK_LENGTH   : ${BLOCK_LENGTH}"
    echo "  PROMPT_STYLE   : ${PROMPT_STYLE}"
    echo "  MAX_CONTEXT_LENGTH: ${MAX_CONTEXT_LENGTH}"
    echo "  NEWLINE_LATER  : ${NEWLINE_LATER}"
    echo "  EARLYSTOP      : ${EARLYSTOP}"
    echo "  LOG_FILE       : ${log_file}"
    echo "----------------------------------------"
    echo "[eval] checkpoint_path=${checkpoint_path:-<none>}"
    echo "[eval] suffix=${effective_suffix:-<none>}"
    echo "[eval] range=${range_label:-full}"
    "${CMD[@]}"
  } 2>&1 | tee -a "${log_file}"
}


for checkpoint_path in "${CHECKPOINT_PATH_LIST[@]}"; do
  run_eval_for_checkpoint "${checkpoint_path}"
done
