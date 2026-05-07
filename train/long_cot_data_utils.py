import random
import re
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk


SYSTEM_PROMPT = """
Please reason step by step, and put your final answer within \boxed{}.
"""

DEFAULT_DATASET_PATH = "dataset/Math-CoT-NoCoT-20k-4096"
DEFAULT_DATASET_NAME = "math_long_cot"
RESPONSE_SOURCE_TO_FIELD = {
    "cot": "CoT_response",
    "noncot": "NonCoT_response",
}
RESPONSE_SOURCE_ALIASES = {
    "cot": "cot",
    "noncot": "noncot",
    "non-cot": "noncot",
    "non_cot": "noncot",
    "nocot": "noncot",
    "no-cot": "noncot",
    "no_cot": "noncot",
}
HELDOUT_SPLIT_NAMES = {"heldout", "eval", "validation", "test"}


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _safe_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _render_chat_prompt(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False) + "\n"
    return f"User: {user_content}\nAssistant:\n"


def _build_teacher_prompt(base_user_prompt: str, response_text: str, tokenizer) -> str:
    privileged_user_prompt = (
        f"{base_user_prompt}\n\n"
        "Here is a reference solution:\n"
        f"{response_text}\n\n"
        "After understanding the reference solution, please try to solve this problem using your own approach below without mentioning the reference solution."
    )
    return _render_chat_prompt(tokenizer, privileged_user_prompt)


def _normalize_response_mode(mode: int, mode_name: str = "mode") -> int:
    normalized_mode = int(mode)
    if normalized_mode not in {1, 2}:
        raise ValueError(f"Unsupported {mode_name} `{mode}`. Expected one of: 1, 2.")
    return normalized_mode


def _normalize_teacher_reference_mode(mode: str) -> str:
    normalized_mode = str(mode or "full").strip().lower()
    if normalized_mode not in {"full", "leave_last_step", "answer_only"}:
        raise ValueError(
            f"Unsupported teacher_reference_mode `{mode}`. Expected one of: full, leave_last_step, answer_only."
        )
    return normalized_mode


def _normalize_response_source(source: str, field_name: str) -> str:
    normalized_source = str(source or "").strip().lower()
    normalized_source = RESPONSE_SOURCE_ALIASES.get(normalized_source, normalized_source)
    if normalized_source not in RESPONSE_SOURCE_TO_FIELD:
        valid = ", ".join(sorted(RESPONSE_SOURCE_TO_FIELD))
        raise ValueError(f"Unsupported {field_name} `{source}`. Expected one of: {valid}.")
    return normalized_source


def _truncate_reasoning_for_teacher(reasoning_text: str) -> str:
    reasoning_text = _safe_str(reasoning_text).strip()
    if not reasoning_text:
        return reasoning_text

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", reasoning_text) if paragraph.strip()]
    if len(paragraphs) >= 2:
        return "\n\n".join(paragraphs[:-1]).strip()

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", reasoning_text) if sentence.strip()]
    if len(sentences) >= 2:
        return " ".join(sentences[:-1]).strip()

    non_empty_lines = [line.strip() for line in reasoning_text.splitlines() if line.strip()]
    if len(non_empty_lines) >= 2:
        return "\n".join(non_empty_lines[:-1]).strip()

    words = reasoning_text.split()
    if len(words) >= 8:
        truncated_word_count = max(int(len(words) * 0.7), len(words) - 3)
        truncated_word_count = max(1, min(truncated_word_count, len(words) - 1))
        return " ".join(words[:truncated_word_count]).strip()

    return reasoning_text


def _build_teacher_reference_response(response_text: str, teacher_reference_mode: str) -> str:
    normalized_mode = _normalize_teacher_reference_mode(teacher_reference_mode)
    response_text = _safe_str(response_text).strip()
    if normalized_mode == "full" or not response_text:
        return response_text

    if normalized_mode == "leave_last_step":
        return _truncate_reasoning_for_teacher(response_text)

    # Raw-response mode no longer extracts a separate answer span.
    return response_text


def _math_response(solution_text: str) -> str:
    return _safe_str(solution_text).strip()


def _format_math_response(raw_response_text: str, mode: int) -> str:
    _normalize_response_mode(mode)
    return _math_response(raw_response_text)


def _build_math_user_prompt(question: str, mode: int) -> str:
    _normalize_response_mode(mode)
    return (
        f"{question}\n\n"
        f"{SYSTEM_PROMPT}"
    )


def _ensure_dataset_dict(dataset_or_dict) -> DatasetDict:
    if isinstance(dataset_or_dict, DatasetDict):
        return dataset_or_dict
    if isinstance(dataset_or_dict, Dataset):
        return DatasetDict({"train": dataset_or_dict})
    raise TypeError(f"Unsupported dataset container type: {type(dataset_or_dict)!r}")


def _split_train_eval(raw_dataset: Dataset, seed: int = 42, eval_ratio: float = 0.01):
    if len(raw_dataset) <= 1:
        return raw_dataset, None

    eval_size = max(int(round(len(raw_dataset) * float(eval_ratio))), 1)
    eval_size = min(eval_size, len(raw_dataset) - 1)
    split_dataset = raw_dataset.train_test_split(test_size=eval_size, seed=seed)
    return split_dataset["train"], split_dataset["test"]


def _load_long_cot_raw_datasets(
    dataset_path: str,
    train_split: str,
    eval_split: Optional[str],
    seed: int,
    heldout_eval_ratio: float,
):
    normalized_dataset_path = str(dataset_path or DEFAULT_DATASET_PATH).strip() or DEFAULT_DATASET_PATH
    dataset_dict = _ensure_dataset_dict(load_from_disk(normalized_dataset_path))
    normalized_train_split = str(train_split or "train").strip()
    if normalized_train_split not in dataset_dict:
        available = ", ".join(sorted(dataset_dict.keys()))
        raise ValueError(
            f"Train split `{train_split}` is unavailable in `{normalized_dataset_path}`. Available splits: {available}."
        )

    train_raw = dataset_dict[normalized_train_split]
    if eval_split is None:
        return train_raw, None

    normalized_eval_split = str(eval_split).strip()
    if normalized_eval_split in dataset_dict:
        return train_raw, dataset_dict[normalized_eval_split]

    if normalized_eval_split.lower() in HELDOUT_SPLIT_NAMES:
        return _split_train_eval(train_raw, seed=seed, eval_ratio=heldout_eval_ratio)

    available = ", ".join(sorted(dataset_dict.keys()))
    raise ValueError(
        f"Eval split `{eval_split}` is unavailable in `{normalized_dataset_path}`. "
        f"Available splits: {available}. Use one of {sorted(HELDOUT_SPLIT_NAMES)} to request an on-the-fly heldout split."
    )


def _select_response_text(example: dict, response_source: str) -> str:
    normalized_source = _normalize_response_source(response_source, field_name="response_source")
    field_name = RESPONSE_SOURCE_TO_FIELD[normalized_source]
    response_text = _safe_str(example.get(field_name)).strip()
    if response_text:
        return response_text
    raise ValueError(f"Example is missing a non-empty `{field_name}` field.")


def _format_example(
    example: dict,
    tokenizer,
    gold_mode: int,
    target_mode: int,
    teacher_reference_mode: str,
    reference_response_source: str,
    target_response_source: str,
) -> dict:
    question = _safe_str(example.get("question")).strip()
    if not question:
        raise ValueError("Every example must contain a non-empty `question` field.")

    reference_raw_response = _select_response_text(example, reference_response_source)
    target_raw_response = _select_response_text(example, target_response_source)

    user_prompt = _build_math_user_prompt(question, mode=gold_mode)
    gold_response = _format_math_response(reference_raw_response, mode=gold_mode)
    target_response = _format_math_response(target_raw_response, mode=target_mode)
    teacher_reference_response = _build_teacher_reference_response(
        gold_response,
        teacher_reference_mode=teacher_reference_mode,
    )

    student_prompt = _render_chat_prompt(tokenizer, user_prompt)
    teacher_prompt = _build_teacher_prompt(user_prompt, teacher_reference_response, tokenizer)
    return {
        "question": question,
        "student_prompt": student_prompt,
        "teacher_prompt": teacher_prompt,
        "teacher_reference_response": teacher_reference_response,
        "gold_response": gold_response,
        "target_response": target_response,
        "inp_par_target_response": target_response,
        "rl_target_response": target_response,
        "tf_target_response": target_response,
        "solution_quality": "full",
        "dataset_name": DEFAULT_DATASET_NAME,
        "reference_response_source": _normalize_response_source(
            reference_response_source,
            field_name="reference_response_source",
        ),
        "target_response_source": _normalize_response_source(
            target_response_source,
            field_name="target_response_source",
        ),
    }


def _format_dataset(
    raw_dataset: Dataset,
    tokenizer,
    gold_mode: int,
    target_mode: int,
    teacher_reference_mode: str,
    reference_response_source: str,
    target_response_source: str,
) -> Dataset:
    columns_to_remove = list(raw_dataset.column_names)
    return raw_dataset.map(
        lambda example: _format_example(
            example,
            tokenizer=tokenizer,
            gold_mode=gold_mode,
            target_mode=target_mode,
            teacher_reference_mode=teacher_reference_mode,
            reference_response_source=reference_response_source,
            target_response_source=target_response_source,
        ),
        remove_columns=columns_to_remove,
    )


def get_distillation_datasets(
    tokenizer,
    dataset_path: str = DEFAULT_DATASET_PATH,
    train_split: str = "train",
    eval_split: Optional[str] = None,
    seed: int = 42,
    gold_mode: int = 1,
    target_mode: int = 1,
    teacher_reference_mode: str = "full",
    reference_response_source: str = "cot",
    target_response_source: str = "cot",
    heldout_eval_ratio: float = 0.01,
):
    gold_mode = _normalize_response_mode(gold_mode, mode_name="gold_mode")
    target_mode = _normalize_response_mode(target_mode, mode_name="target_mode")
    teacher_reference_mode = _normalize_teacher_reference_mode(teacher_reference_mode)
    reference_response_source = _normalize_response_source(
        reference_response_source,
        field_name="reference_response_source",
    )
    target_response_source = _normalize_response_source(
        target_response_source,
        field_name="target_response_source",
    )

    train_raw, eval_raw = _load_long_cot_raw_datasets(
        dataset_path=dataset_path,
        train_split=train_split,
        eval_split=eval_split,
        seed=seed,
        heldout_eval_ratio=heldout_eval_ratio,
    )

    train_dataset = _format_dataset(
        train_raw,
        tokenizer=tokenizer,
        gold_mode=gold_mode,
        target_mode=target_mode,
        teacher_reference_mode=teacher_reference_mode,
        reference_response_source=reference_response_source,
        target_response_source=target_response_source,
    )
    eval_dataset = (
        _format_dataset(
            eval_raw,
            tokenizer=tokenizer,
            gold_mode=gold_mode,
            target_mode=target_mode,
            teacher_reference_mode=teacher_reference_mode,
            reference_response_source=reference_response_source,
            target_response_source=target_response_source,
        )
        if eval_raw is not None
        else None
    )
    return train_dataset, eval_dataset
