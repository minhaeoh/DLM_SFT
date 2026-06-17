import random
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk


DEFAULT_PROMPT = """
Please reason step by step, and put your final answer within \\boxed{}.
"""

FORMAT_PROMPT = """
Please reason step by step and respond in the following format, with the final answer inside \\boxed{}:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

ANSWER_FIRST_PROMPT = """
Please reason step by step, but respond with the final answer first inside \\boxed{}, followed by the reasoning:

<answer>
...
</answer>
<reasoning>
...
</reasoning>
"""


PROMPT_TYPE_TO_TEXT = {
    "default": DEFAULT_PROMPT,
    "format": FORMAT_PROMPT,
    "answer_first": ANSWER_FIRST_PROMPT,
}
PROMPT_TYPE_ALIASES = {
    "answer-first": "answer_first",
    "answerfirst": "answer_first",
}

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
ANSWER_BLOCK_TARGET_TOKEN_LENGTH = 32
ANSWER_BLOCK_PADDING_CANDIDATES = (
    " ",
    "\n",
    "\t",
    "  ",
    "\n\n",
    "\t\t",
    " \n",
    "\n ",
    " \t",
    "\t ",
    "\n\t",
    "\t\n",
)


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


def _normalize_response_mode(mode: int, mode_name: str = "mode") -> int:
    normalized_mode = int(mode)
    if normalized_mode not in {1, 2}:
        raise ValueError(f"Unsupported {mode_name} `{mode}`. Expected one of: 1, 2.")
    return normalized_mode


def _normalize_response_source(source: str, field_name: str) -> str:
    normalized_source = str(source or "").strip().lower()
    normalized_source = RESPONSE_SOURCE_ALIASES.get(normalized_source, normalized_source)
    if normalized_source not in RESPONSE_SOURCE_TO_FIELD:
        valid = ", ".join(sorted(RESPONSE_SOURCE_TO_FIELD))
        raise ValueError(f"Unsupported {field_name} `{source}`. Expected one of: {valid}.")
    return normalized_source


def _normalize_prompt_type(prompt_type: str) -> str:
    normalized_prompt_type = str(prompt_type or "default").strip().lower()
    normalized_prompt_type = PROMPT_TYPE_ALIASES.get(normalized_prompt_type, normalized_prompt_type)
    if normalized_prompt_type not in PROMPT_TYPE_TO_TEXT:
        valid = ", ".join(sorted(PROMPT_TYPE_TO_TEXT))
        raise ValueError(f"Unsupported prompt_type `{prompt_type}`. Expected one of: {valid}.")
    return normalized_prompt_type


def _math_response(solution_text: str) -> str:
    return _safe_str(solution_text).strip()


def _count_text_tokens(tokenizer, text: str) -> int:
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
    )


def _render_answer_block(answer_text: str, padding: str = "") -> str:
    return f"<answer>\n{answer_text}{padding}\n</answer>"


def _get_answer_block_token_length(tokenizer, answer_text: str, padding: str = "") -> int:
    return _count_text_tokens(tokenizer, _render_answer_block(answer_text, padding=padding))


def _decode_single_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=False)


def _get_answer_block_padding_candidates(tokenizer) -> tuple[str, ...]:
    cached_candidates = getattr(tokenizer, "_answer_block_padding_candidates", None)
    if cached_candidates is not None:
        return cached_candidates

    candidates = []
    seen = set()

    def _register_candidate(candidate: str):
        if not candidate or candidate in seen or not candidate.isspace():
            return
        candidates.append(candidate)
        seen.add(candidate)

    for candidate in ANSWER_BLOCK_PADDING_CANDIDATES:
        _register_candidate(candidate)

    for token_id in range(len(tokenizer)):
        decoded = _decode_single_token(tokenizer, token_id)
        if decoded and decoded.isspace() and len(decoded) <= 8:
            _register_candidate(decoded)

    cached_candidates = tuple(candidates)
    setattr(tokenizer, "_answer_block_padding_candidates", cached_candidates)
    return cached_candidates


def _build_padded_answer_block(tokenizer, answer_text: str, target_token_length: int) -> str:
    padding_candidates = _get_answer_block_padding_candidates(tokenizer)
    padding = ""
    answer_block = _render_answer_block(answer_text)
    answer_block_token_length = _get_answer_block_token_length(tokenizer, answer_text)

    if answer_block_token_length > target_token_length:
        raise ValueError(
            "The `<answer>...</answer>` block already exceeds the requested fixed token length "
            f"({answer_block_token_length} > {target_token_length})."
        )

    while answer_block_token_length < target_token_length:
        remaining_tokens = target_token_length - answer_block_token_length
        next_padding = None

        for candidate in padding_candidates:
            candidate_padding = padding + candidate
            candidate_block = _render_answer_block(answer_text, padding=candidate_padding)
            candidate_length = _get_answer_block_token_length(
                tokenizer,
                answer_text,
                padding=candidate_padding,
            )
            token_increase = candidate_length - answer_block_token_length

            if token_increase <= 0 or token_increase > remaining_tokens:
                continue
            if token_increase == remaining_tokens or token_increase == 1:
                next_padding = candidate_padding
                answer_block = candidate_block
                answer_block_token_length = candidate_length
                break

        if next_padding is None:
            raise ValueError(
                "Failed to pad the `<answer>...</answer>` block to the requested fixed token length "
                f"({target_token_length}) using whitespace-only padding."
            )

        padding = next_padding

    return answer_block


def _is_answer_block_within_token_budget(
    example: dict,
    tokenizer,
    target_token_length: int,
) -> bool:
    answer_text = _safe_str(example.get("answer")).strip()
    if not answer_text:
        return True
    return _get_answer_block_token_length(tokenizer, answer_text) <= target_token_length


def _filter_oversized_answer_block_examples(
    raw_dataset: Dataset,
    tokenizer,
    target_token_length: int,
) -> Dataset:
    original_size = len(raw_dataset)
    filtered_dataset = raw_dataset.filter(
        lambda example: _is_answer_block_within_token_budget(
            example,
            tokenizer=tokenizer,
            target_token_length=target_token_length,
        )
    )
    filtered_count = original_size - len(filtered_dataset)
    if filtered_count > 0:
        print(
            "Filtered out "
            f"{filtered_count} examples whose `<answer>...</answer>` block exceeded "
            f"{target_token_length} tokens."
        )
    return filtered_dataset


def _format_structured_math_response(
    solution_text: str,
    answer_text: str,
    prompt_type: str,
    tokenizer=None,
    answer_block: bool = False,
) -> str:
    normalized_prompt_type = _normalize_prompt_type(prompt_type)
    solution_text = _math_response(solution_text)
    answer_text = _safe_str(answer_text).strip()

    if normalized_prompt_type == "default":
        return solution_text

    if not answer_text:
        raise ValueError(
            f"Prompt type `{normalized_prompt_type}` requires a non-empty `answer` field in each example."
        )

    if normalized_prompt_type == "format":
        return (
            f"<reasoning>\n{solution_text}\n</reasoning>\n"
            f"<answer>\n{answer_text}\n</answer>"
        )

    if normalized_prompt_type == "answer_first":
        if answer_block:
            if tokenizer is None:
                raise ValueError("`answer_block=True` requires a tokenizer.")
            answer_section = _build_padded_answer_block(
                tokenizer=tokenizer,
                answer_text=answer_text,
                target_token_length=ANSWER_BLOCK_TARGET_TOKEN_LENGTH,
            )
            return f"{answer_section}\n<reasoning>\n{solution_text}\n</reasoning>"

        return (
            f"<answer>\n{answer_text}\n</answer>\n"
            f"<reasoning>\n{solution_text}\n</reasoning>"
        )

    raise ValueError(f"Unsupported prompt_type `{prompt_type}`.")


def _build_math_user_prompt(question: str, mode: int, prompt_type: str = "default") -> str:
    _normalize_response_mode(mode)
    normalized_prompt_type = _normalize_prompt_type(prompt_type)
    return f"Question:\n{question}\n\n{PROMPT_TYPE_TO_TEXT[normalized_prompt_type].strip()}"


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
    prompt_type: str,
    target_response_source: str,
    answer_block: bool,
) -> dict:
    question = _safe_str(example.get("question")).strip()
    answer = _safe_str(example.get("answer")).strip()
    if not question:
        raise ValueError("Every example must contain a non-empty `question` field.")

    prompt_type = _normalize_prompt_type(prompt_type)
    target_raw_response = _select_response_text(example, target_response_source)
    user_prompt = _build_math_user_prompt(question, mode=1, prompt_type=prompt_type)
    target_response = _format_structured_math_response(
        solution_text=target_raw_response,
        answer_text=answer,
        prompt_type=prompt_type,
        tokenizer=tokenizer,
        answer_block=answer_block,
    )

    return {
        "question": question,
        "answer": answer,
        "student_prompt": _render_chat_prompt(tokenizer, user_prompt),
        "target_response": target_response,
        "solution_quality": "full",
        "dataset_name": DEFAULT_DATASET_NAME,
        "prompt_type": prompt_type,
        "target_response_source": _normalize_response_source(
            target_response_source,
            field_name="target_response_source",
        ),
    }


def _format_dataset(
    raw_dataset: Dataset,
    tokenizer,
    prompt_type: str,
    target_response_source: str,
    answer_block: bool,
) -> Dataset:
    normalized_prompt_type = _normalize_prompt_type(prompt_type)
    filtered_dataset = raw_dataset
    if answer_block and normalized_prompt_type == "answer_first":
        filtered_dataset = _filter_oversized_answer_block_examples(
            raw_dataset,
            tokenizer=tokenizer,
            target_token_length=ANSWER_BLOCK_TARGET_TOKEN_LENGTH,
        )

    columns_to_remove = list(filtered_dataset.column_names)
    return filtered_dataset.map(
        lambda example: _format_example(
            example,
            tokenizer=tokenizer,
            prompt_type=normalized_prompt_type,
            target_response_source=target_response_source,
            answer_block=answer_block,
        ),
        remove_columns=columns_to_remove,
    )


def get_distillation_datasets(
    tokenizer,
    dataset_path: str = DEFAULT_DATASET_PATH,
    train_split: str = "train",
    eval_split: Optional[str] = None,
    seed: int = 42,
    prompt_type: str = "default",
    target_response_source: str = "cot",
    heldout_eval_ratio: float = 0.01,
    answer_block: bool = False,
):
    prompt_type = _normalize_prompt_type(prompt_type)
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
        prompt_type=prompt_type,
        target_response_source=target_response_source,
        answer_block=answer_block,
    )
    eval_dataset = (
        _format_dataset(
            eval_raw,
            tokenizer=tokenizer,
            prompt_type=prompt_type,
            target_response_source=target_response_source,
            answer_block=answer_block,
        )
        if eval_raw is not None
        else None
    )
    return train_dataset, eval_dataset
