"""
Build filtered datasets from a-m-team/AM-DeepSeek-R1-Distilled-1.4M (am_0.5M config).

Steps:
  1. Load am_0.5M via streaming with a custom Features schema (bypasses a schema
     mismatch in the upstream JSONL that causes normal load_dataset to crash).
  2. Extract question / answer (reference_answer) / CoT_response (<think>...</think>
     included) from the two-message format.
  3. Filter to examples where all three fields are non-empty.
  4. Apply format-template length filter: keep only examples whose
     (student_prompt + target_response) token length is <= max_length.
  5. Save the full filtered dataset to output_root.
  6. Shuffle once (fixed seed) then save nested prefix subsets of the sizes
     given by --subset_sizes (default 8k / 16k / 32k / 64k / 128k).
"""

import argparse
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, Features, Value, load_dataset
from transformers import AutoTokenizer

from long_cot_data_utils import (
    _build_math_user_prompt,
    _format_structured_math_response,
    _normalize_prompt_type,
    _render_chat_prompt,
)


DEFAULT_MODEL_PATH = "GSAI-ML/LLaDA-8B-Base"
DEFAULT_SOURCE_DATASET = "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"
DEFAULT_SOURCE_CONFIG = "am_0.5M"
DEFAULT_OUTPUT_ROOT = "/home/minhae/diffusion/DLM_SFT/datasets"
DEFAULT_MAX_LENGTH = 4096
DEFAULT_SUBSET_SIZES = [8_000, 16_000, 32_000, 64_000, 128_000]
DEFAULT_SEED = 42

# Custom features bypasses an upstream schema mismatch where some records have
# a struct in `test_case` while others have a plain string.
_CUSTOM_FEATURES = Features({
    "messages": [
        {
            "role": Value("string"),
            "content": Value("string"),
            "info": {
                "source": Value("string"),
                "reference_answer": Value("string"),
                "test_case": Value("string"),
                "think_content": Value("string"),
                "answer_content": Value("string"),
            },
        }
    ]
})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--source_dataset", type=str, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--source_config", type=str, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--prompt_type", type=str, default="format",
                        help="Prompt template: default | format | answer_first")
    parser.add_argument("--subset_sizes", type=int, nargs="+", default=DEFAULT_SUBSET_SIZES,
                        help="Nested subset sizes to create from the full filtered dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_proc", type=int, default=32)
    return parser.parse_args()


def _clean_str(value) -> str:
    s = str(value or "").strip()
    return "" if s in ("None", "null") else s


def stream_and_extract(source_dataset: str, source_config: str) -> list[dict]:
    ds = load_dataset(
        source_dataset,
        source_config,
        split="train",
        streaming=True,
        features=_CUSTOM_FEATURES,
    )

    records = []
    for ex in ds:
        msgs = ex.get("messages") or []
        if len(msgs) < 2:
            continue
        question = _clean_str(msgs[0].get("content"))
        user_info = msgs[0].get("info") or {}
        answer = _clean_str(user_info.get("reference_answer"))
        cot = _clean_str(msgs[1].get("content"))
        source = _clean_str(user_info.get("source"))

        if question and answer and cot:
            records.append({
                "question": question,
                "answer": answer,
                "CoT_response": cot,
                "data_source": source,
            })

        if len(records) % 50_000 == 0 and len(records) > 0:
            print(f"  collected {len(records):,} valid examples so far...")

    return records


def build_length_filtered_dataset(raw_dataset, tokenizer, prompt_type, max_length, num_proc):
    prompt_type = _normalize_prompt_type(prompt_type)

    def _add_texts(example):
        user_prompt = _build_math_user_prompt(example["question"], mode=1, prompt_type=prompt_type)
        return {
            "student_prompt": _render_chat_prompt(tokenizer, user_prompt),
            "target_response": _format_structured_math_response(
                solution_text=example["CoT_response"],
                answer_text=example["answer"],
                prompt_type=prompt_type,
                tokenizer=tokenizer,
            ),
        }

    def _compute_lengths(batch):
        prompt_ids = tokenizer(batch["student_prompt"], add_special_tokens=False, return_attention_mask=False)["input_ids"]
        response_ids = tokenizer(batch["target_response"], add_special_tokens=False, return_attention_mask=False)["input_ids"]
        return {"total_length": [len(p) + len(r) for p, r in zip(prompt_ids, response_ids)]}

    with_texts = raw_dataset.map(_add_texts, num_proc=num_proc)
    with_lengths = with_texts.map(_compute_lengths, batched=True, num_proc=num_proc,
                                  desc=f"Computing lengths ({prompt_type})")
    filtered = with_lengths.filter(lambda ex: ex["total_length"] <= max_length)
    return filtered.remove_columns(["student_prompt", "target_response", "total_length"])


def save_dataset(dataset, path: Path):
    if path.exists():
        raise FileExistsError(f"Output path already exists: {path}")
    DatasetDict({"train": dataset}).save_to_disk(str(path))


def print_source_breakdown(dataset, label: str, top_n: int = 5):
    counts = Counter(dataset["data_source"])
    top = ", ".join(f"{k}={v}" for k, v in counts.most_common(top_n))
    print(f"  [{label}] data_source top {top_n}: {top}")


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    prompt_type = _normalize_prompt_type(args.prompt_type)

    # dataset name: AM-DeepSeek-R1-CoT-<prompt_type>-<max_length>
    max_k = args.max_length // 1000
    base_name = f"AM-DeepSeek-R1-CoT-{prompt_type}-{max_k}k"

    print(f"Loading {args.source_dataset} ({args.source_config}) via streaming...")
    records = stream_and_extract(args.source_dataset, args.source_config)
    print(f"Valid examples (question + answer + CoT_response): {len(records):,}")

    from datasets import Dataset
    raw_dataset = Dataset.from_list(records)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)

    print(f"\nApplying {prompt_type} template + {args.max_length}-token filter...")
    filtered = build_length_filtered_dataset(raw_dataset, tokenizer, prompt_type, args.max_length, args.num_proc)
    print(f"After filter: {len(filtered):,} / {len(raw_dataset):,} examples")

    # Save full filtered dataset
    full_path = output_root / base_name
    save_dataset(filtered, full_path)
    print(f"\nSaved full dataset ({len(filtered):,}) -> {full_path}")
    print_source_breakdown(filtered, base_name)

    # Build nested subsets by shuffling once and taking prefixes
    shuffled = filtered.shuffle(seed=args.seed)
    total = len(shuffled)

    for size in sorted(args.subset_sizes):
        kept = min(size, total)
        subset = shuffled.select(range(kept))
        size_label = f"{round(size / 1000)}k"
        subset_path = output_root / f"{base_name}-{size_label}"
        save_dataset(subset, subset_path)
        print(f"Saved {size_label} subset ({kept:,}) -> {subset_path}")
        print_source_breakdown(subset, size_label)

    print("\nDone.")


if __name__ == "__main__":
    main()
