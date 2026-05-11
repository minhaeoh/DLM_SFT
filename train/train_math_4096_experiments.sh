#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_DIR="${ROOT_DIR}/train"
TRAIN_SCRIPT="${TRAIN_DIR}/long_cot_train.py"
ENV_FILE="${ROOT_DIR}/.env"
LOCAL_DATASET_PATH="${ROOT_DIR}/datasets/Math-CoT-NoCoT-20k-format-4096"
FALLBACK_DATASET_PATH="/home/minhae/diffusion/diffu-distill/d1-self-distill/dataset/Math-CoT-NoCoT-20k-format-4096"
LOCAL_DATASET_LABEL="${LOCAL_DATASET_PATH%/}"
LOCAL_DATASET_LABEL="${LOCAL_DATASET_LABEL##*/}"
cd "${TRAIN_DIR}"

DEFAULT_ACCELERATE="/home/minhae/anaconda3/envs/d1-sd/bin/accelerate"
DEFAULT_PYTHON="/home/minhae/anaconda3/envs/d1-sd/bin/python"
ACCELERATE_BIN="${ACCELERATE_BIN:-$([[ -x "${DEFAULT_ACCELERATE}" ]] && echo "${DEFAULT_ACCELERATE}" || echo accelerate)}"
PYTHON_BIN="${PYTHON_BIN:-$([[ -x "${DEFAULT_PYTHON}" ]] && echo "${DEFAULT_PYTHON}" || echo python)}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${TRAIN_DIR}/ddp_config_no_deepspeed.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${#CUDA_DEVICE_ARRAY[@]}"
if [[ "${NUM_PROCESSES}" -ne 2 ]]; then
  echo "Expected exactly 2 visible GPUs. Current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

DATASET_PATH="${DATASET_PATH:-$([[ -d "${LOCAL_DATASET_PATH}" ]] && echo "${LOCAL_DATASET_PATH}" || echo "${FALLBACK_DATASET_PATH}")}"
DATASET_LABEL="${DATASET_LABEL:-${LOCAL_DATASET_LABEL}}"
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Base}"
MODEL_LABEL="${MODEL_PATH##*/}"
OUTPUT_ROOT_OVERRIDE="${OUTPUT_ROOT:-}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-${ROOT_DIR}/checkpoints}"
if [[ -n "${OUTPUT_ROOT_OVERRIDE}" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE}"
  OUTPUT_ROOT_IS_EXPLICIT=1
else
  OUTPUT_ROOT="${OUTPUT_ROOT_BASE}/${DATASET_LABEL}/${MODEL_LABEL}"
  OUTPUT_ROOT_IS_EXPLICIT=0
fi
LOG_DIR="${LOG_DIR:-${TRAIN_DIR}/logs/long-CoT}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-long-CoT-train}"
export WANDB_PROJECT

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-8}"
HELDOUT_EVAL_RATIO="${HELDOUT_EVAL_RATIO:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-64}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
SEED="${SEED:-42}"
BF16="${BF16:-True}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
CE_WEIGHT="${CE_WEIGHT:-0.5}"
PROMPT_STYLE="${PROMPT_STYLE:-default}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}"
BASE_MAIN_PROCESS_PORT="${BASE_MAIN_PROCESS_PORT:-29600}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

BATCH_SIZE="$((2 * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "Training entrypoint not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "Accelerate config not found: ${ACCELERATE_CONFIG}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG_FILE="${LOG_DIR}/train_math_4096_experiments_${TIMESTAMP}.log"
exec > >(tee -a "${MASTER_LOG_FILE}") 2>&1

compute_dataset_stats() {
  local dataset_path="$1"
  DATASET_PATH_FOR_STATS="${dataset_path}" "${PYTHON_BIN}" - <<PY
import os
from datasets import load_from_disk
from math import ceil

dataset_path = os.environ["DATASET_PATH_FOR_STATS"]
heldout_ratio = float("${HELDOUT_EVAL_RATIO}")
effective_batch = int("${BATCH_SIZE}")

total_size = len(load_from_disk(dataset_path)["train"])
eval_size = max(int(round(total_size * heldout_ratio)), 1)
eval_size = min(eval_size, total_size - 1)
train_size = total_size - eval_size
effective_batch = max(effective_batch, 1)
steps_per_epoch = ceil(train_size / effective_batch)
half_epoch_save_steps = max(1, round(steps_per_epoch / 2))

print(total_size, eval_size, train_size, steps_per_epoch, half_epoch_save_steps)
PY
}

TOTAL_SIZE="N/A"
EVAL_SIZE="N/A"
TRAIN_SIZE="N/A"
STEPS_PER_EPOCH="N/A"
HALF_EPOCH_SAVE_STEPS="N/A"
if [[ -d "${DATASET_PATH}" ]]; then
  read -r TOTAL_SIZE EVAL_SIZE TRAIN_SIZE STEPS_PER_EPOCH HALF_EPOCH_SAVE_STEPS <<< \
    "$(compute_dataset_stats "${DATASET_PATH}")"
fi

echo "========================================"
echo "long-CoT sequential runs starting"
echo "ROOT_DIR                    : ${ROOT_DIR}"
echo "TRAIN_DIR                   : ${TRAIN_DIR}"
echo "TRAIN_SCRIPT                : ${TRAIN_SCRIPT}"
echo "ACCELERATE_BIN              : ${ACCELERATE_BIN}"
echo "PYTHON_BIN                  : ${PYTHON_BIN}"
echo "ACCELERATE_CONFIG           : ${ACCELERATE_CONFIG}"
echo "CUDA_VISIBLE_DEVICES        : ${CUDA_VISIBLE_DEVICES}"
echo "NUM_PROCESSES               : ${NUM_PROCESSES}"
echo "DATASET_PATH                : ${DATASET_PATH}"
if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "DATASET_PATH_STATUS         : missing at startup; use run_experiment ... --dataset_path /path/to/dataset"
else
  echo "DATASET_PATH_STATUS         : available"
fi
echo "DATASET_LABEL               : ${DATASET_LABEL}"
echo "MODEL_PATH                  : ${MODEL_PATH}"
echo "OUTPUT_ROOT_BASE            : ${OUTPUT_ROOT_BASE}"
echo "OUTPUT_ROOT                 : ${OUTPUT_ROOT}"
echo "OUTPUT_ROOT_IS_EXPLICIT     : ${OUTPUT_ROOT_IS_EXPLICIT}"
echo "LOG_DIR                     : ${LOG_DIR}"
echo "REPORT_TO                   : ${REPORT_TO}"
echo "WANDB_PROJECT               : ${WANDB_PROJECT}"
echo "PER_DEVICE_TRAIN_BATCH_SIZE : ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "PER_DEVICE_EVAL_BATCH_SIZE  : ${PER_DEVICE_EVAL_BATCH_SIZE}"
echo "GRAD_ACCUM_STEPS            : ${GRADIENT_ACCUMULATION_STEPS}"
echo "BATCH_SIZE                  : ${BATCH_SIZE}"
echo "MAX_LENGTH                  : ${MAX_LENGTH}"
echo "NUM_TRAIN_EPOCHS            : ${NUM_TRAIN_EPOCHS}"
echo "HELDOUT_EVAL_RATIO          : ${HELDOUT_EVAL_RATIO}"
echo "LEARNING_RATE               : ${LEARNING_RATE}"
echo "LOSS_CHUNK_SIZE             : ${LOSS_CHUNK_SIZE}"
echo "BF16                        : ${BF16}"
echo "GRADIENT_CHECKPOINTING      : ${GRADIENT_CHECKPOINTING}"
echo "CE_WEIGHT                   : ${CE_WEIGHT}"
echo "PROMPT_STYLE                : ${PROMPT_STYLE}"
echo "TOTAL_SIZE                  : ${TOTAL_SIZE}"
echo "TRAIN_SIZE                  : ${TRAIN_SIZE}"
echo "EVAL_SIZE                   : ${EVAL_SIZE}"
echo "STEPS_PER_EPOCH             : ${STEPS_PER_EPOCH}"
echo "HALF_EPOCH_SAVE_STEPS       : ${HALF_EPOCH_SAVE_STEPS}"
echo "MASTER_LOG_FILE             : ${MASTER_LOG_FILE}"
echo "========================================"

run_experiment() {
  local method="$1"
  local target_source="$2"
  local run_index="$3"
  local prompt_style="${PROMPT_STYLE}"
  local effective_dataset_path="${DATASET_PATH}"
  if [[ $# -ge 4 && "${4}" != --* ]]; then
    prompt_style="$4"
    shift 4
  else
    shift 3
  fi
  local extra_train_args=("$@")
  local filtered_extra_train_args=()
  local normalized_method=""
  local effective_ce_weight="${CE_WEIGHT}"
  local run_method_label=""
  local prompt_style_label=""
  local extra_t_sampling_mode=""
  local extra_t_biased_to_one_strength="2.0"
  local extra_arg_idx=0
  local effective_dataset_label="${DATASET_LABEL}"
  local effective_output_root="${OUTPUT_ROOT}"
  local run_total_size=""
  local run_eval_size=""
  local run_train_size=""
  local run_steps_per_epoch=""
  local run_half_epoch_save_steps=""

  while [[ "${extra_arg_idx}" -lt "${#extra_train_args[@]}" ]]; do
    local extra_arg="${extra_train_args[${extra_arg_idx}]}"
    case "${extra_arg}" in
      --dataset_path=*)
        effective_dataset_path="${extra_arg#*=}"
        ;;
      --dataset_path)
        if [[ $((extra_arg_idx + 1)) -ge "${#extra_train_args[@]}" ]]; then
          echo "--dataset_path requires a value." >&2
          exit 1
        fi
        effective_dataset_path="${extra_train_args[$((extra_arg_idx + 1))]}"
        extra_arg_idx=$((extra_arg_idx + 1))
        ;;
      --t_sampling_mode=*)
        extra_t_sampling_mode="${extra_arg#*=}"
        filtered_extra_train_args+=("${extra_arg}")
        ;;
      --t_sampling_mode)
        if [[ $((extra_arg_idx + 1)) -ge "${#extra_train_args[@]}" ]]; then
          echo "--t_sampling_mode requires a value." >&2
          exit 1
        fi
        extra_t_sampling_mode="${extra_train_args[$((extra_arg_idx + 1))]}"
        filtered_extra_train_args+=("${extra_arg}" "${extra_train_args[$((extra_arg_idx + 1))]}")
        extra_arg_idx=$((extra_arg_idx + 1))
        ;;
      --t_biased_to_one_strength=*)
        extra_t_biased_to_one_strength="${extra_arg#*=}"
        filtered_extra_train_args+=("${extra_arg}")
        ;;
      --t_biased_to_one_strength)
        if [[ $((extra_arg_idx + 1)) -ge "${#extra_train_args[@]}" ]]; then
          echo "--t_biased_to_one_strength requires a value." >&2
          exit 1
        fi
        extra_t_biased_to_one_strength="${extra_train_args[$((extra_arg_idx + 1))]}"
        filtered_extra_train_args+=("${extra_arg}" "${extra_train_args[$((extra_arg_idx + 1))]}")
        extra_arg_idx=$((extra_arg_idx + 1))
        ;;
      *)
        filtered_extra_train_args+=("${extra_arg}")
        ;;
    esac
    extra_arg_idx=$((extra_arg_idx + 1))
  done
  extra_train_args=("${filtered_extra_train_args[@]}")

  if [[ ! -d "${effective_dataset_path}" ]]; then
    echo "Dataset path not found: ${effective_dataset_path}" >&2
    exit 1
  fi

  local normalized_default_dataset_path="${DATASET_PATH%/}"
  local normalized_effective_dataset_path="${effective_dataset_path%/}"
  if [[ "${normalized_effective_dataset_path}" != "${normalized_default_dataset_path}" ]]; then
    effective_dataset_label="${normalized_effective_dataset_path##*/}"
  fi
  if [[ -z "${effective_dataset_label}" ]]; then
    effective_dataset_label="${DATASET_LABEL}"
  fi
  if [[ "${OUTPUT_ROOT_IS_EXPLICIT}" == "1" ]]; then
    effective_output_root="${OUTPUT_ROOT}"
  else
    effective_output_root="${OUTPUT_ROOT_BASE}/${effective_dataset_label}/${MODEL_LABEL}"
  fi

  read -r run_total_size run_eval_size run_train_size run_steps_per_epoch run_half_epoch_save_steps <<< \
    "$(compute_dataset_stats "${effective_dataset_path}")"

  normalized_method="$(printf '%s' "${method}" | tr '[:lower:]-' '[:upper:]_')"
  prompt_style_label="$(printf '%s' "${prompt_style}" | tr '[:upper:]-' '[:lower:]_')"

  case "${normalized_method}" in
    SFT|INP_OH)
      run_method_label="SFT_tgt${target_source}"
      ;;
    *)
      run_method_label="${method}"
      ;;
  esac

  run_method_label="${run_method_label}_${prompt_style_label}"

  case "${extra_t_sampling_mode}" in
    biased_to_one|biased-to-one|biasedtoone|high-bias|highbias)
      run_method_label="${run_method_label}_tbiased_w${extra_t_biased_to_one_strength}"
      ;;
  esac

  local run_name="math_ff_4096_${run_method_label}_ep${NUM_TRAIN_EPOCHS}_${TIMESTAMP}"
  local output_dir="${effective_output_root}/BS${BATCH_SIZE}_${run_name}"
  local run_log_file="${LOG_DIR}/${run_name}.log"
  local main_process_port=$((BASE_MAIN_PROCESS_PORT + run_index))

  if [[ -d "${output_dir}" && "$(find "${output_dir}" -mindepth 1 -maxdepth 1 | head -n 1)" != "" && "${OVERWRITE_OUTPUT_DIR}" != "1" ]]; then
    echo "Output directory already exists and is not empty: ${output_dir}" >&2
    echo "Set OVERWRITE_OUTPUT_DIR=1 if you want to reuse it." >&2
    exit 1
  fi

  mkdir -p "${output_dir}"

  echo
  echo "----------------------------------------"
  echo "Starting run                 : ${run_name}"
  echo "METHOD                       : ${method}"
  echo "RUN_METHOD_LABEL             : ${run_method_label}"
  echo "TARGET_RESPONSE_SOURCE       : ${target_source}"
  echo "PROMPT_STYLE                 : ${prompt_style}"
  echo "DATASET_PATH                 : ${effective_dataset_path}"
  echo "DATASET_LABEL                : ${effective_dataset_label}"
  echo "EFFECTIVE_CE_WEIGHT          : ${effective_ce_weight}"
  echo "TOTAL_SIZE                   : ${run_total_size}"
  echo "TRAIN_SIZE                   : ${run_train_size}"
  echo "EVAL_SIZE                    : ${run_eval_size}"
  echo "STEPS_PER_EPOCH              : ${run_steps_per_epoch}"
  echo "HALF_EPOCH_SAVE_STEPS        : ${run_half_epoch_save_steps}"
  echo "OUTPUT_ROOT                  : ${effective_output_root}"
  echo "OUTPUT_DIR                   : ${output_dir}"
  echo "RUN_LOG_FILE                 : ${run_log_file}"
  echo "MAIN_PROCESS_PORT            : ${main_process_port}"
  if [[ "${#extra_train_args[@]}" -gt 0 ]]; then
    echo "EXTRA_TRAIN_ARGS             : ${extra_train_args[*]}"
  else
    echo "EXTRA_TRAIN_ARGS             : (none)"
  fi
  echo "----------------------------------------"

  local cmd=(
    "${ACCELERATE_BIN}" launch
    --config_file "${ACCELERATE_CONFIG}"
    --num_processes "${NUM_PROCESSES}"
    --main_process_port "${main_process_port}"
    "${TRAIN_SCRIPT}"
    --model_path "${MODEL_PATH}"
    --dataset "${effective_dataset_label}"
    --dataset_path "${effective_dataset_path}"
    --method "${method}"
    --target_response_source "${target_source}"
    --prompt_type "${prompt_style}"
    --output_dir "${output_dir}"
    --run_name "${run_name}"
    --train_split train
    --eval_split heldout
    --heldout_eval_ratio "${HELDOUT_EVAL_RATIO}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --loss_chunk_size "${LOSS_CHUNK_SIZE}"
    --max_length "${MAX_LENGTH}"
    --logging_strategy steps
    --logging_steps "${LOGGING_STEPS}"
    --save_strategy steps
    --save_steps "${run_half_epoch_save_steps}"
    --eval_strategy steps
    --eval_steps "${run_half_epoch_save_steps}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --bf16 "${BF16}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    --remove_unused_columns False
    --ddp_find_unused_parameters False
    --do_eval True
    --report_to "${REPORT_TO}"
    --seed "${SEED}"
    --ce_weight "${effective_ce_weight}"
  )

  if [[ "${OVERWRITE_OUTPUT_DIR}" == "1" ]]; then
    cmd+=(--overwrite_output_dir True)
  fi

  if [[ "${#extra_train_args[@]}" -gt 0 ]]; then
    cmd+=("${extra_train_args[@]}")
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${cmd[@]}" \
    2>&1 | tee -a "${run_log_file}"

  echo "Finished run                 : ${run_name}"
  echo "Saved run log to             : ${run_log_file}"
}

run_experiment "SFT" "noncot" 0 "format"
run_experiment "SFT" "noncot" 0 "answer_first"
run_experiment "SFT" "cot" 0 "format"
run_experiment "SFT" "cot" 0 "answer_first"
run_experiment "SFT" "noncot" 0 "format" --dataset_path "/home/minhae/diffusion/DLM_SFT/datasets/Math-NoCoT-format-4096"
run_experiment "SFT" "noncot" 0 "answer_first" --dataset_path "/home/minhae/diffusion/DLM_SFT/datasets/Math-NoCoT-format-4096"
# run_experiment "SFT" "cot" 1
# run_experiment "SFT" "noncot" 2 \
#   --t_sampling_mode=biased_to_one \
#   --t_biased_to_one_strength=2.0
# run_experiment "SFT" "cot" 3 \
#   --t_sampling_mode=biased_to_one \
#   --t_biased_to_one_strength=2.0
# Example: switch prompt style per run.
# run_experiment "SFT" "cot" 4 "format"
# Example: switch dataset path for one run.
# run_experiment "SFT" "cot" 5 "default" --dataset_path "/path/to/other/dataset"
# run_experiment "SFT" "noncot" 2
# Example: sample continuously over [t_min, t_max] while biasing toward higher t.
# run_experiment "SFT" "noncot" 1 \
#   --t_sampling_mode=uniform \
#   --t_min=0.2 \
#   --t_max=0.9

echo
echo "All long-CoT runs completed."
echo "Master log saved to: ${MASTER_LOG_FILE}"
