#!/usr/bin/env bash

set -euo pipefail

DEFAULT_REPO_ID="minhaeoh/DLM_SFT"
DEFAULT_BASE_PATH="math_long_cot_format_4096/LLaDA-8B-Base"

repo_id="${HF_REPO_ID:-$DEFAULT_REPO_ID}"
base_path="${HF_BASE_PATH:-$DEFAULT_BASE_PATH}"
repo_type="${HF_REPO_TYPE:-model}"
quiet=0
dry_run=0

usage() {
  cat <<'EOF'
Usage:
  train/upload_dirs_to_hf.sh [options] <dir1> [<dir2> ...]

Uploads each local directory to Hugging Face under:
  <base-path>/<local-directory-name>

Defaults:
  repo_id:   minhaeoh/DLM_SFT
  base_path: math_long_cot_format_4096/LLaDA-8B-Base
  repo_type: model

Options:
  --repo-id <repo_id>      Override Hugging Face repo id.
  --base-path <path>       Override remote base path prefix.
  --repo-type <type>       One of: model, dataset, space.
  --quiet                  Pass --quiet to huggingface-cli upload.
  --dry-run                Print commands without uploading.
  -h, --help               Show this help message.

Examples:
  train/upload_dirs_to_hf.sh \
    eval/eval_results/math/LLaDA-8B-Base/SFT_tgtcot \
    eval/eval_results/math/LLaDA-8B-Base/SFT_tgtnoncot

  train/upload_dirs_to_hf.sh --dry-run checkpoints/my_run
EOF
}

dirs=()
while (($# > 0)); do
  case "$1" in
    --repo-id)
      repo_id="$2"
      shift 2
      ;;
    --base-path)
      base_path="$2"
      shift 2
      ;;
    --repo-type)
      repo_type="$2"
      shift 2
      ;;
    --quiet)
      quiet=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while (($# > 0)); do
        dirs+=("$1")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      dirs+=("$1")
      shift
      ;;
  esac
done

if ((${#dirs[@]} == 0)); then
  echo "At least one directory is required." >&2
  usage >&2
  exit 1
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is not installed or not on PATH." >&2
  exit 1
fi

huggingface-cli whoami >/dev/null

base_path="${base_path%/}"

for dir in "${dirs[@]}"; do
  if [[ ! -d "$dir" ]]; then
    echo "Directory not found: $dir" >&2
    exit 1
  fi
done

for dir in "${dirs[@]}"; do
  dir_name="$(basename "$dir")"
  remote_path="$base_path/$dir_name"
  cmd=(
    huggingface-cli upload
    --repo-type "$repo_type"
    "$repo_id"
    "$dir"
    "$remote_path"
  )

  if ((quiet)); then
    cmd+=(--quiet)
  fi

  echo "Uploading $dir -> $repo_id/$remote_path"

  if ((dry_run)); then
    printf '  %q' "${cmd[@]}"
    printf '\n'
    continue
  fi

  "${cmd[@]}"
done
