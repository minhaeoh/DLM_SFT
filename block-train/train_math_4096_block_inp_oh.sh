#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BLOCK_TRAIN_DIR="${ROOT_DIR}/block-train"
TRAIN_SCRIPT="${BLOCK_TRAIN_DIR}/block_diffusion_train.py"
ENV_FILE="${ROOT_DIR}/.env"
LOCAL_DATASET_PATH="${ROOT_DIR}/datasets/Math-CoT-NoCoT-20k-format-4096"
FALLBACK_DATASET_PATH="/home/minhae/diffusion/diffu-distill/d1-self-distill/dataset/Math-CoT-NoCoT-20k-format-4096"
cd "${BLOCK_TRAIN_DIR}"

DEFAULT_ACCELERATE="/home/minhae/anaconda3/envs/d1-sd/bin/accelerate"
DEFAULT_PYTHON="/home/minhae/anaconda3/envs/d1-sd/bin/python"
ACCELERATE_BIN="${ACCELERATE_BIN:-$([[ -x "${DEFAULT_ACCELERATE}" ]] && echo "${DEFAULT_ACCELERATE}" || echo accelerate)}"
PYTHON_BIN="${PYTHON_BIN:-$([[ -x "${DEFAULT_PYTHON}" ]] && echo "${DEFAULT_PYTHON}" || echo python)}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${ROOT_DIR}/train/ddp_config_no_deepspeed.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${#CUDA_DEVICE_ARRAY[@]}"
if [[ "${NUM_PROCESSES}" -ne 2 ]]; then
  echo "Expected exactly 2 visible GPUs. Current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

DATASET_PATH="${DATASET_PATH:-$([[ -d "${LOCAL_DATASET_PATH}" ]] && echo "${LOCAL_DATASET_PATH}" || echo "${FALLBACK_DATASET_PATH}")}"
DATASET_LABEL="${DATASET_LABEL:-math_long_cot_format_4096}"
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Base}"
MODEL_LABEL="${MODEL_PATH##*/}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/checkpoints/${DATASET_LABEL}/${MODEL_LABEL}/block_train}"
LOG_DIR="${LOG_DIR:-${BLOCK_TRAIN_DIR}/logs/${DATASET_LABEL}}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-block-diffusion-train}"
export WANDB_PROJECT

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
BATCH_SIZE="$((2 * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
MAX_LENGTH="${MAX_LENGTH:-4096}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-8}"
HELDOUT_EVAL_RATIO="${HELDOUT_EVAL_RATIO:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
SEED="${SEED:-42}"
BF16="${BF16:-True}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}"
BASE_MAIN_PROCESS_PORT="${BASE_MAIN_PROCESS_PORT:-29600}"

BLOCK_SIZE="${BLOCK_SIZE:-16}"
MASK_ID="${MASK_ID:-126336}"
T_MIN="${T_MIN:-1e-3}"
T_MAX="${T_MAX:-1.0}"
T_SAMPLING_MODE="${T_SAMPLING_MODE:-uniform}"
T_FIXED="${T_FIXED:-0.9}"
T_BIASED_TO_ONE_STRENGTH="${T_BIASED_TO_ONE_STRENGTH:-2.0}"
T_TWO_POINT_LOW="${T_TWO_POINT_LOW:-0.2}"
T_TWO_POINT_HIGH="${T_TWO_POINT_HIGH:-0.9}"
T_TWO_POINT_HIGH_PROB="${T_TWO_POINT_HIGH_PROB:-0.5}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "Training entrypoint not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "Accelerate config not found: ${ACCELERATE_CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "Dataset path not found: ${DATASET_PATH}" >&2
  echo "Set DATASET_PATH to a valid load_from_disk dataset directory." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG_FILE="${LOG_DIR}/train_math_4096_block_experiments_${TIMESTAMP}.log"
exec > >(tee -a "${MASTER_LOG_FILE}") 2>&1

read -r TOTAL_SIZE EVAL_SIZE TRAIN_SIZE STEPS_PER_EPOCH HALF_EPOCH_SAVE_STEPS <<< "$(
  "${PYTHON_BIN}" - <<PY
from datasets import load_from_disk
from math import ceil

dataset_path = "${DATASET_PATH}"
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
)"

echo "========================================"
echo "block diffusion sequential runs starting"
echo "ROOT_DIR                    : ${ROOT_DIR}"
echo "BLOCK_TRAIN_DIR             : ${BLOCK_TRAIN_DIR}"
echo "TRAIN_SCRIPT                : ${TRAIN_SCRIPT}"
echo "ACCELERATE_BIN              : ${ACCELERATE_BIN}"
echo "PYTHON_BIN                  : ${PYTHON_BIN}"
echo "ACCELERATE_CONFIG           : ${ACCELERATE_CONFIG}"
echo "CUDA_VISIBLE_DEVICES        : ${CUDA_VISIBLE_DEVICES}"
echo "NUM_PROCESSES               : ${NUM_PROCESSES}"
echo "DATASET_PATH                : ${DATASET_PATH}"
echo "MODEL_PATH                  : ${MODEL_PATH}"
echo "OUTPUT_ROOT                 : ${OUTPUT_ROOT}"
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
echo "BLOCK_SIZE                  : ${BLOCK_SIZE}"
echo "T_SAMPLING_MODE             : ${T_SAMPLING_MODE}"
echo "T_MIN                       : ${T_MIN}"
echo "T_MAX                       : ${T_MAX}"
echo "MASK_ID                     : ${MASK_ID}"
echo "BF16                        : ${BF16}"
echo "GRADIENT_CHECKPOINTING      : ${GRADIENT_CHECKPOINTING}"
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
  shift 3
  local extra_train_args=("$@")
  local normalized_method=""
  local effective_block_size="${BLOCK_SIZE}"
  local effective_t_sampling_mode="${T_SAMPLING_MODE}"
  local effective_t_biased_to_one_strength="${T_BIASED_TO_ONE_STRENGTH}"
  local extra_arg_idx=0

  while [[ "${extra_arg_idx}" -lt "${#extra_train_args[@]}" ]]; do
    local extra_arg="${extra_train_args[${extra_arg_idx}]}"
    case "${extra_arg}" in
      --block_size=*)
        effective_block_size="${extra_arg#*=}"
        ;;
      --block_size)
        if [[ $((extra_arg_idx + 1)) -lt "${#extra_train_args[@]}" ]]; then
          effective_block_size="${extra_train_args[$((extra_arg_idx + 1))]}"
          extra_arg_idx=$((extra_arg_idx + 1))
        fi
        ;;
      --t_sampling_mode=*)
        effective_t_sampling_mode="${extra_arg#*=}"
        ;;
      --t_sampling_mode)
        if [[ $((extra_arg_idx + 1)) -lt "${#extra_train_args[@]}" ]]; then
          effective_t_sampling_mode="${extra_train_args[$((extra_arg_idx + 1))]}"
          extra_arg_idx=$((extra_arg_idx + 1))
        fi
        ;;
      --t_biased_to_one_strength=*)
        effective_t_biased_to_one_strength="${extra_arg#*=}"
        ;;
      --t_biased_to_one_strength)
        if [[ $((extra_arg_idx + 1)) -lt "${#extra_train_args[@]}" ]]; then
          effective_t_biased_to_one_strength="${extra_train_args[$((extra_arg_idx + 1))]}"
          extra_arg_idx=$((extra_arg_idx + 1))
        fi
        ;;
    esac
    extra_arg_idx=$((extra_arg_idx + 1))
  done

  normalized_method="$(printf '%s' "${method}" | tr '[:lower:]-' '[:upper:]_')"

  local run_method_label="${normalized_method}_tgt${target_source}_bsz${effective_block_size}"
  case "${effective_t_sampling_mode}" in
    uniform)
      ;;
    biased_to_one|biased-to-one|biasedtoone|high-bias|highbias)
      run_method_label="${run_method_label}_tbiased_w${effective_t_biased_to_one_strength}"
      ;;
    *)
      run_method_label="${run_method_label}_t${effective_t_sampling_mode}"
      ;;
  esac

  local run_name="math_bd_4096_${run_method_label}_ep${NUM_TRAIN_EPOCHS}_${TIMESTAMP}"
  local output_dir="${OUTPUT_ROOT}/BS${BATCH_SIZE}_${run_name}"
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
  echo "EFFECTIVE_BLOCK_SIZE         : ${effective_block_size}"
  echo "EFFECTIVE_T_SAMPLING_MODE    : ${effective_t_sampling_mode}"
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
    --dataset "${DATASET_LABEL}"
    --dataset_path "${DATASET_PATH}"
    --method "${method}"
    --target_response_source "${target_source}"
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
    --max_length "${MAX_LENGTH}"
    --logging_strategy steps
    --logging_steps "${LOGGING_STEPS}"
    --save_strategy steps
    --save_steps "${HALF_EPOCH_SAVE_STEPS}"
    --eval_strategy steps
    --eval_steps "${HALF_EPOCH_SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --bf16 "${BF16}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    --remove_unused_columns False
    --do_eval True
    --report_to "${REPORT_TO}"
    --seed "${SEED}"
    --mask_id "${MASK_ID}"
    --block_size "${BLOCK_SIZE}"
    --t_min "${T_MIN}"
    --t_max "${T_MAX}"
    --t_sampling_mode "${T_SAMPLING_MODE}"
    --t_fixed "${T_FIXED}"
    --t_biased_to_one_strength "${T_BIASED_TO_ONE_STRENGTH}"
    --t_two_point_low "${T_TWO_POINT_LOW}"
    --t_two_point_high "${T_TWO_POINT_HIGH}"
    --t_two_point_high_prob "${T_TWO_POINT_HIGH_PROB}"
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

# The block diffusion trainer currently only supports INP-OH.
run_experiment "INP_OH" "noncot" 0
run_experiment "INP_OH" "cot" 1
# run_experiment "INP_OH" "noncot" 2 \
#   --t_sampling_mode=biased_to_one \
#   --t_biased_to_one_strength=2.0
# run_experiment "INP_OH" "cot" 3 \
#   --t_sampling_mode=biased_to_one \
#   --t_biased_to_one_strength=2.0

echo
echo "All block diffusion runs completed."
echo "Master log saved to: ${MASTER_LOG_FILE}"
