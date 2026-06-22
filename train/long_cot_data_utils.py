import random
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk


INSTRUCTION_PROMPT = """
Please reason step by step with the final answer inside \\boxed{}.
"""

DEFAULT_DATASET_PATH = "dataset/Math-CoT-NoCoT-20k-4096"
DEFAULT_DATASET_NAME = "math_long_cot"

# LongCoT responses include a <think>...</think> block; ShortCoT responses are answer-only.
RESPONSE_SOURCE_TO_INCLUDE_THINK = {
    "longcot": True,
    "shortcot": False,
}
RESPONSE_SOURCE_ALIASES = {
    "longcot": "longcot",
    "long-cot": "longcot",
    "long_cot": "longcot",
    "long": "longcot",
    "shortcot": "shortcot",
    "short-cot": "shortcot",
    "short_cot": "shortcot",
    "short": "shortcot",
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


def _normalize_response_source(source: str, field_name: str) -> str:
    normalized_source = str(source or "").strip().lower()
    normalized_source = RESPONSE_SOURCE_ALIASES.get(normalized_source, normalized_source)
    if normalized_source not in RESPONSE_SOURCE_TO_INCLUDE_THINK:
        valid = ", ".join(sorted(RESPONSE_SOURCE_TO_INCLUDE_THINK))
        raise ValueError(f"Unsupported {field_name} `{source}`. Expected one of: {valid}.")
    return normalized_source


def _math_response(solution_text: str) -> str:
    return _safe_str(solution_text).strip()


def _format_structured_math_response(
    think_text: str,
    answer_text: str,
    include_think: bool,
) -> str:
    answer_text = _safe_str(answer_text).strip()
    if not answer_text:
        raise ValueError("Each example requires a non-empty `answer_content` field.")

    answer_block = f"<answer>\n{answer_text}\n</answer>"

    # ShortCoT: answer-only.
    if not include_think:
        return answer_block

    # LongCoT: <think>...</think> followed by the answer block.
    think_text = _math_response(think_text)
    return f"<think>\n{think_text}\n</think>\n{answer_block}"


def _build_math_user_prompt(question: str) -> str:
    return f"Question:\n{question}\n\n{INSTRUCTION_PROMPT.strip()}"


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


def _format_example(
    example: dict,
    tokenizer,
    target_response_source: str,
) -> dict:
    question = _safe_str(example.get("question")).strip()
    if not question:
        raise ValueError("Every example must contain a non-empty `question` field.")

    normalized_source = _normalize_response_source(
        target_response_source,
        field_name="target_response_source",
    )
    include_think = RESPONSE_SOURCE_TO_INCLUDE_THINK[normalized_source]

    think_text = _safe_str(example.get("think_content"))
    answer = _safe_str(example.get("answer_content")).strip()

    user_prompt = _build_math_user_prompt(question)
    target_response = _format_structured_math_response(
        think_text=think_text,
        answer_text=answer,
        include_think=include_think,
    )

    return {
        "question": question,
        "answer": answer,
        "student_prompt": _render_chat_prompt(tokenizer, user_prompt),
        "target_response": target_response,
        "solution_quality": "full",
        "dataset_name": DEFAULT_DATASET_NAME,
        "target_response_source": normalized_source,
    }


def _format_dataset(
    raw_dataset: Dataset,
    tokenizer,
    target_response_source: str,
) -> Dataset:
    columns_to_remove = list(raw_dataset.column_names)
    return raw_dataset.map(
        lambda example: _format_example(
            example,
            tokenizer=tokenizer,
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
    target_response_source: str = "longcot",
    heldout_eval_ratio: float = 0.01,
):
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
        target_response_source=target_response_source,
    )
    eval_dataset = (
        _format_dataset(
            eval_raw,
            tokenizer=tokenizer,
            target_response_source=target_response_source,
        )
        if eval_raw is not None
        else None
    )
    return train_dataset, eval_dataset
