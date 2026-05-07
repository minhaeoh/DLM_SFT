from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


METHOD_ALIASES = {
    "ALL-MASK": "ALL_MASK",
    "ALLMASK": "ALL_MASK",
    "SFT": "ALL_MASK",
    "INP-OH": "INP_OH",
    "INP-OH-AVG-FULL": "INP_OH_PAD",
    "INP-OH-PAD": "INP_OH_PAD",
    "INPOH": "INP_OH",
    "INPOHAVGFULL": "INP_OH_PAD",
    "INPOHPAD": "INP_OH_PAD",
    "INPAINING": "INP",
    "INPAINTING": "INP",
}
VALID_METHODS = {
    "ALL_MASK",
    "INP",
    "INP_OH",
    "INP_OH_PAD",
}
METHOD_METRIC_NAMES = (
    "ALL_MASK",
    "INP",
    "INP_OH",
    "INP_OH_PAD",
)
EXTERNAL_TARGET_METHODS = set()
METHOD_TO_TARGET_PATH_ATTR = {}
T_SAMPLING_MODE_ALIASES = {
    "biased-to-one": "biased_to_one",
    "biasedtoone": "biased_to_one",
    "high-bias": "biased_to_one",
    "highbias": "biased_to_one",
    "two-point": "two_point",
    "twopoint": "two_point",
}
VALID_T_SAMPLING_MODES = {"uniform", "fixed", "two_point", "curriculum", "biased_to_one"}
CE_MASK_MODE_ALIASES = {
    "all": "full",
    "kd": "masked",
    "mask": "masked",
}
VALID_CE_MASK_MODES = {"answer", "full", "masked"}
ONE_HOT_TEACHER_METHODS = {"INP_OH", "INP_OH_PAD"}
PAD_TARGET_RESPONSE_METHODS = {"INP_OH_PAD"}
TOP1_LOGIT_DEBUG_STEPS_PER_EPOCH = 30
TOP1_LOGIT_DEBUG_FILENAME = "debug_top1_logits_first100.jsonl"


def _normalize_method_name(method: str) -> str:
    normalized = str(method or "").strip().upper()
    normalized = METHOD_ALIASES.get(normalized, normalized)
    return normalized


def normalize_training_method(method: str) -> str:
    normalized = _normalize_method_name(method) or "ALL_MASK"
    if normalized not in VALID_METHODS:
        valid = ", ".join(sorted(VALID_METHODS))
        raise ValueError(f"Unsupported method `{method}`. Expected one of: {valid}.")
    return normalized


def normalize_t_sampling_mode(mode: str) -> str:
    normalized = str(mode or "uniform").strip().lower()
    normalized = T_SAMPLING_MODE_ALIASES.get(normalized, normalized)
    if normalized not in VALID_T_SAMPLING_MODES:
        valid = ", ".join(sorted(VALID_T_SAMPLING_MODES))
        raise ValueError(f"Unsupported t_sampling_mode `{mode}`. Expected one of: {valid}.")
    return normalized


def normalize_ce_mask_mode(mode: str) -> str:
    normalized = str(mode or "answer").strip().lower()
    normalized = CE_MASK_MODE_ALIASES.get(normalized, normalized)
    if normalized not in VALID_CE_MASK_MODES:
        valid = ", ".join(sorted(VALID_CE_MASK_MODES))
        raise ValueError(f"Unsupported ce_mask_mode `{mode}`. Expected one of: {valid}.")
    return normalized


def sample_t_biased_to_one(
    batch_size: int,
    device: torch.device | None = None,
    epsilon: float = 0.0,
    strength: float = 2.0,
    t_max: float = 1.0,
) -> torch.Tensor:
    if batch_size <= 0:
        kwargs = {"dtype": torch.float32}
        if device is not None:
            kwargs["device"] = device
        return torch.empty(0, **kwargs)

    epsilon = float(epsilon)
    t_max = float(t_max)
    strength = float(strength)
    if strength <= 0.0:
        raise ValueError("`strength` must be > 0.")
    if epsilon > t_max:
        raise ValueError("`epsilon` must be <= `t_max`.")

    rand_kwargs = {"dtype": torch.float32}
    if device is not None:
        rand_kwargs["device"] = device
    u = torch.rand(batch_size, **rand_kwargs)
    x = u.pow(1.0 / strength)
    return epsilon + (t_max - epsilon) * x


def disable_dropout_in_model(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0


def method_uses_one_hot_teacher(method: str) -> bool:
    return str(method or "").strip().upper() in ONE_HOT_TEACHER_METHODS


def method_uses_padded_target_response(method: str) -> bool:
    return str(method or "").strip().upper() in PAD_TARGET_RESPONSE_METHODS


@dataclass
class BatchMetrics:
    mean_t: float
    avg_mask_ratio: float
    avg_student_mask_ratio: float
    avg_response_length: float
    num_masked_tokens: float
    num_student_masked_tokens: float
    method_fractions: dict[str, float]


class DiffuSelfDistillDataCollator:
    """
    Tokenize prompts and gold responses for method-specific processing inside the trainer.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int,
        mask_id: int,
        t_min: float = 1e-3,
        t_max: float = 1.0,
        t_sampling_mode: str = "uniform",
        t_fixed: float = 0.9,
        t_biased_to_one_strength: float = 2.0,
        t_two_point_low: float = 0.2,
        t_two_point_high: float = 0.9,
        t_two_point_high_prob: float = 0.5,
        t_curriculum_start_min: float = 0.0,
        t_curriculum_start_max: float = 0.4,
        t_curriculum_end_min: float = 0.8,
        t_curriculum_end_max: float = 1.0,
        t_curriculum_total_batches: int = 0,
        method: str = "ALL_MASK",
        dataset_name: str = "",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_id = mask_id
        self.t_min = float(max(t_min, 0.0))
        self.t_max = float(min(max(t_max, 0.0), 1.0))
        self.t_sampling_mode = normalize_t_sampling_mode(t_sampling_mode)
        self.t_fixed = float(min(max(t_fixed, 0.0), 1.0))
        self.t_biased_to_one_strength = float(t_biased_to_one_strength)
        self.t_two_point_low = float(min(max(t_two_point_low, 0.0), 1.0))
        self.t_two_point_high = float(min(max(t_two_point_high, 0.0), 1.0))
        self.t_two_point_high_prob = float(min(max(t_two_point_high_prob, 0.0), 1.0))
        self.t_curriculum_start_min = float(min(max(t_curriculum_start_min, 0.0), 1.0))
        self.t_curriculum_start_max = float(min(max(t_curriculum_start_max, 0.0), 1.0))
        self.t_curriculum_end_min = float(min(max(t_curriculum_end_min, 0.0), 1.0))
        self.t_curriculum_end_max = float(min(max(t_curriculum_end_max, 0.0), 1.0))
        self.t_curriculum_total_batches = max(int(t_curriculum_total_batches), 0)
        self._t_sampling_batch_index = 0
        self._validate_t_sampling_settings()
        self.method = normalize_training_method(method)
        self.dataset_name = str(dataset_name or "").strip().lower()
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            # Keep behavior close to d1 scripts where pad is set to eos when absent.
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must provide pad_token_id or eos_token_id.")
        self.fallback_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else self.pad_token_id
        self.answer_open_tag = "<answer>"
        self.answer_close_tag = "</answer>"
        self.answer_open_tag_ids = self.tokenizer(
            self.answer_open_tag,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        self.answer_close_tag_ids = self.tokenizer(
            self.answer_close_tag,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        self.reasoning_open_tag = "<reasoning>"
        self.reasoning_close_tag = "</reasoning>"
        self.reasoning_open_tag_ids = self.tokenizer(
            self.reasoning_open_tag,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        self.reasoning_close_tag_ids = self.tokenizer(
            self.reasoning_close_tag,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

    def _validate_t_sampling_settings(self):
        if self.t_min > self.t_max:
            raise ValueError(f"`t_min` ({self.t_min}) must be <= `t_max` ({self.t_max}).")
        if self.t_biased_to_one_strength <= 0.0:
            raise ValueError("`t_biased_to_one_strength` must be > 0.")
        if self.t_two_point_low > self.t_two_point_high:
            raise ValueError("`t_two_point_low` must be <= `t_two_point_high`.")
        if self.t_curriculum_start_min > self.t_curriculum_start_max:
            raise ValueError("`t_curriculum_start_min` must be <= `t_curriculum_start_max`.")
        if self.t_curriculum_end_min > self.t_curriculum_end_max:
            raise ValueError("`t_curriculum_end_min` must be <= `t_curriculum_end_max`.")

    def set_t_sampling_total_batches(self, total_batches: int):
        self.t_curriculum_total_batches = max(int(total_batches), 0)

    def reset_t_sampling_state(self):
        self._t_sampling_batch_index = 0

    def _get_curriculum_progress(self) -> float:
        total_batches = max(int(self.t_curriculum_total_batches), 0)
        if total_batches <= 1:
            return 0.0
        return min(float(self._t_sampling_batch_index) / float(total_batches - 1), 1.0)

    def _sample_t_values(self, batch_size: int) -> torch.Tensor:
        if batch_size <= 0:
            return torch.empty(0, dtype=torch.float32)

        if self.t_sampling_mode == "fixed":
            return torch.full((batch_size,), self.t_fixed, dtype=torch.float32)

        if self.t_sampling_mode == "biased_to_one":
            return sample_t_biased_to_one(
                batch_size,
                epsilon=self.t_min,
                strength=self.t_biased_to_one_strength,
                t_max=self.t_max,
            )

        if self.t_sampling_mode == "two_point":
            low_values = torch.full((batch_size,), self.t_two_point_low, dtype=torch.float32)
            high_values = torch.full((batch_size,), self.t_two_point_high, dtype=torch.float32)
            choose_high = torch.rand(batch_size) < self.t_two_point_high_prob
            return torch.where(choose_high, high_values, low_values)

        sample_min = self.t_min
        sample_max = self.t_max
        if self.t_sampling_mode == "curriculum":
            progress = self._get_curriculum_progress()
            sample_min = (1.0 - progress) * self.t_curriculum_start_min + progress * self.t_curriculum_end_min
            sample_max = (1.0 - progress) * self.t_curriculum_start_max + progress * self.t_curriculum_end_max

        if sample_min > sample_max:
            sample_min, sample_max = sample_max, sample_min
        if abs(sample_max - sample_min) <= 1e-8:
            return torch.full((batch_size,), float(sample_min), dtype=torch.float32)
        return torch.empty(batch_size, dtype=torch.float32).uniform_(float(sample_min), float(sample_max))

    def _tokenize_text(self, text: str, truncation: bool = True):
        return self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=truncation,
            max_length=self.max_length,
            return_attention_mask=False,
        )["input_ids"]

    def _tokenize_text_with_offsets(self, text: str, truncation: bool = True):
        if not getattr(self.tokenizer, "is_fast", False):
            return None
        return self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=truncation,
            max_length=self.max_length,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )

    def _truncate_prompt_ids(self, prompt_ids: list[int]) -> list[int]:
        return prompt_ids[: max(self.max_length - 1, 1)]

    def _ensure_non_empty_response(self, response_ids: list[int]) -> list[int]:
        if response_ids:
            return response_ids
        return [self.fallback_token_id]

    @staticmethod
    def _find_subsequence(sequence: list[int], subsequence: list[int], start_idx: int = 0) -> int:
        if not subsequence:
            return -1
        max_start = len(sequence) - len(subsequence)
        for idx in range(max(start_idx, 0), max_start + 1):
            if sequence[idx : idx + len(subsequence)] == subsequence:
                return idx
        return -1

    def _build_tag_mask_from_token_ids(
        self,
        token_ids: list[int],
        open_tag_ids: list[int],
        close_tag_ids: list[int],
    ) -> list[bool]:
        mask = [False] * len(token_ids)
        if not token_ids:
            return mask

        open_start = self._find_subsequence(token_ids, open_tag_ids)
        if open_start < 0:
            return mask

        content_start = open_start + len(open_tag_ids)
        close_start = self._find_subsequence(token_ids, close_tag_ids, start_idx=content_start)
        if close_start < 0 or close_start < content_start:
            return mask

        for idx in range(content_start, close_start):
            mask[idx] = True
        return mask

    def _build_answer_mask_from_token_ids(self, token_ids: list[int]) -> list[bool]:
        return self._build_tag_mask_from_token_ids(
            token_ids,
            open_tag_ids=self.answer_open_tag_ids,
            close_tag_ids=self.answer_close_tag_ids,
        )

    def _build_reasoning_mask_from_token_ids(self, token_ids: list[int]) -> list[bool]:
        return self._build_tag_mask_from_token_ids(
            token_ids,
            open_tag_ids=self.reasoning_open_tag_ids,
            close_tag_ids=self.reasoning_close_tag_ids,
        )

    @staticmethod
    def _find_tag_char_span(text: str, open_tag: str, close_tag: str):
        open_start = text.find(open_tag)
        if open_start < 0:
            return None

        content_start = open_start + len(open_tag)
        content_end = text.find(close_tag, content_start)
        if content_end < 0 or content_end < content_start:
            return None
        return content_start, content_end

    def _find_answer_char_span(self, text: str):
        return self._find_tag_char_span(text, self.answer_open_tag, self.answer_close_tag)

    def _find_reasoning_char_span(self, text: str):
        return self._find_tag_char_span(text, self.reasoning_open_tag, self.reasoning_close_tag)

    @staticmethod
    def _apply_char_span_to_mask(answer_mask: list[bool], offsets, char_span):
        if char_span is None:
            return
        span_start, span_end = char_span
        for idx, (token_start, token_end) in enumerate(offsets):
            if token_end <= token_start:
                continue
            if token_start < span_end and token_end > span_start:
                answer_mask[idx] = True

    def _tokenize_response_text(self, text: str):
        if getattr(self.tokenizer, "is_fast", False):
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=False,
                return_offsets_mapping=True,
            )
            response_ids = encoding["input_ids"]
            answer_mask = [False] * len(response_ids)
            reasoning_mask = [False] * len(response_ids)
            self._apply_char_span_to_mask(
                answer_mask,
                encoding["offset_mapping"],
                self._find_answer_char_span(text),
            )
            self._apply_char_span_to_mask(
                reasoning_mask,
                encoding["offset_mapping"],
                self._find_reasoning_char_span(text),
            )
            return response_ids, answer_mask, reasoning_mask

        response_ids = self._tokenize_text(text)
        return (
            response_ids,
            self._build_answer_mask_from_token_ids(response_ids),
            self._build_reasoning_mask_from_token_ids(response_ids),
        )

    def build_answer_mask_from_response_ids(self, response_ids: torch.Tensor | list[int]) -> list[bool]:
        token_ids = response_ids.tolist() if isinstance(response_ids, torch.Tensor) else list(response_ids)
        if not token_ids:
            return []

        if getattr(self.tokenizer, "is_fast", False):
            response_text = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            encoding = self.tokenizer(
                response_text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_offsets_mapping=True,
            )
            if encoding["input_ids"] == token_ids:
                answer_mask = [False] * len(token_ids)
                answer_span = self._find_answer_char_span(response_text)
                if answer_span is not None:
                    answer_start, answer_end = answer_span
                    for idx, (token_start, token_end) in enumerate(encoding["offset_mapping"]):
                        if token_end <= token_start:
                            continue
                        if token_start < answer_end and token_end > answer_start:
                            answer_mask[idx] = True
                if any(answer_mask):
                    return answer_mask
                return [True] * len(token_ids)

        answer_mask = self._build_answer_mask_from_token_ids(token_ids)
        if any(answer_mask):
            return answer_mask
        return [True] * len(token_ids)

    def _tokenize_response_feature(self, feature: dict[str, Any], field_name: str):
        response_text = feature.get(field_name) or feature.get("gold_response") or feature.get("response") or ""
        response_ids, answer_mask, reasoning_mask = self._tokenize_response_text(response_text)
        response_ids = self._ensure_non_empty_response(response_ids)
        if not answer_mask or not any(answer_mask):
            answer_mask = [True] * len(response_ids)
        if not reasoning_mask:
            reasoning_mask = [False] * len(response_ids)
        return (
            torch.tensor(response_ids, dtype=torch.long),
            torch.tensor(answer_mask, dtype=torch.bool),
            torch.tensor(reasoning_mask, dtype=torch.bool),
        )

    @staticmethod
    def _longest_common_prefix_len(left_ids: list[int], right_ids: list[int]) -> int:
        max_prefix_len = min(len(left_ids), len(right_ids))
        prefix_len = 0
        while prefix_len < max_prefix_len and left_ids[prefix_len] == right_ids[prefix_len]:
            prefix_len += 1
        return prefix_len

    def _extract_visible_reference_text(
        self,
        teacher_prompt_text: str,
        teacher_reference_response: str,
        teacher_prompt_token_limit: int,
    ) -> str:
        teacher_prompt_text = str(teacher_prompt_text or "")
        teacher_reference_response = str(teacher_reference_response or "")
        if teacher_prompt_token_limit <= 0 or not teacher_prompt_text or not teacher_reference_response:
            return ""

        reference_start = teacher_prompt_text.find(teacher_reference_response)
        if reference_start < 0:
            return ""
        reference_end = reference_start + len(teacher_reference_response)

        prompt_encoding = self._tokenize_text_with_offsets(teacher_prompt_text, truncation=True)
        if prompt_encoding is None:
            return ""

        prompt_offsets = prompt_encoding["offset_mapping"][:teacher_prompt_token_limit]
        visible_reference_end = reference_start
        for token_start, token_end in prompt_offsets:
            if token_end <= token_start:
                continue
            if token_start < reference_end and token_end > reference_start:
                visible_reference_end = max(visible_reference_end, min(token_end, reference_end))

        if visible_reference_end <= reference_start:
            return ""

        local_visible_end = visible_reference_end - reference_start
        return teacher_reference_response[:local_visible_end]

    def _build_reference_visible_reasoning_mask(
        self,
        feature: dict[str, Any],
        response_ids: torch.Tensor,
        reasoning_mask: torch.Tensor,
        teacher_prompt_token_limit: int,
    ) -> torch.Tensor:
        response_len = int(response_ids.numel())
        visible_reasoning_mask = torch.zeros(response_len, dtype=torch.bool)
        if response_len <= 0:
            return visible_reasoning_mask

        teacher_reference_response = str(feature.get("teacher_reference_response") or "")
        if not teacher_reference_response:
            return visible_reasoning_mask

        visible_reference_text = self._extract_visible_reference_text(
            teacher_prompt_text=str(feature.get("teacher_prompt") or ""),
            teacher_reference_response=teacher_reference_response,
            teacher_prompt_token_limit=teacher_prompt_token_limit,
        )
        if not visible_reference_text:
            return visible_reasoning_mask

        visible_reasoning_char_span = self._find_reasoning_char_span(visible_reference_text)
        if visible_reasoning_char_span is None:
            return visible_reasoning_mask

        visible_reasoning_text = visible_reference_text[
            visible_reasoning_char_span[0] : visible_reasoning_char_span[1]
        ]
        visible_reasoning_ids = self.tokenizer(
            visible_reasoning_text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        if not visible_reasoning_ids:
            return visible_reasoning_mask

        reasoning_positions = torch.nonzero(reasoning_mask[:response_len], as_tuple=False).flatten()
        if reasoning_positions.numel() == 0:
            return visible_reasoning_mask

        target_reasoning_ids = response_ids.index_select(0, reasoning_positions).tolist()
        visible_prefix_len = self._longest_common_prefix_len(target_reasoning_ids, visible_reasoning_ids)
        if visible_prefix_len <= 0:
            return visible_reasoning_mask

        visible_positions = reasoning_positions[:visible_prefix_len]
        visible_reasoning_mask[visible_positions] = True
        return visible_reasoning_mask

    @staticmethod
    def _trim_char_span(text: str, span: tuple[int, int]) -> tuple[int, int] | None:
        span_start, span_end = span
        span_start = max(int(span_start), 0)
        span_end = min(int(span_end), len(text))
        while span_start < span_end and text[span_start].isspace():
            span_start += 1
        while span_end > span_start and text[span_end - 1].isspace():
            span_end -= 1
        if span_end <= span_start:
            return None
        return span_start, span_end

    def _find_reasoning_sentence_spans(self, text: str) -> list[tuple[int, int]]:
        reasoning_char_span = self._find_reasoning_char_span(text)
        if reasoning_char_span is None:
            return []

        reasoning_start, reasoning_end = reasoning_char_span
        reasoning_text = text[reasoning_start:reasoning_end]
        if not reasoning_text.strip():
            return []

        boundary_positions = {0, len(reasoning_text)}
        boundary_patterns = (
            r"\n\s*\n+",
            r"(?<=[.!?])(?:[\"')\]}]+)?\s+(?=\S)",
            r"(?<=\\\])\s*(?=\S)",
            r"(?<=\$\$)\s*(?=\S)",
            r"\n+(?=\S)",
        )
        for pattern in boundary_patterns:
            for match in re.finditer(pattern, reasoning_text):
                boundary_positions.add(match.end())

        sorted_boundaries = sorted(boundary_positions)
        sentence_spans = []
        for left, right in zip(sorted_boundaries, sorted_boundaries[1:]):
            trimmed_span = self._trim_char_span(reasoning_text, (left, right))
            if trimmed_span is None:
                continue
            local_start, local_end = trimmed_span
            sentence_spans.append((reasoning_start + local_start, reasoning_start + local_end))

        if sentence_spans:
            alnum_spans = [span for span in sentence_spans if any(ch.isalnum() for ch in text[span[0] : span[1]])]
            if alnum_spans:
                return alnum_spans
            return sentence_spans

        trimmed_reasoning_span = self._trim_char_span(text, reasoning_char_span)
        return [trimmed_reasoning_span] if trimmed_reasoning_span is not None else []

    def _build_random_reasoning_sentence_hint_mask(
        self,
        feature: dict[str, Any],
        response_ids: torch.Tensor,
        reasoning_mask: torch.Tensor,
    ) -> torch.Tensor:
        response_len = int(response_ids.numel())
        hint_mask = torch.zeros(response_len, dtype=torch.bool)
        if response_len <= 0 or not getattr(self.tokenizer, "is_fast", False):
            return hint_mask

        response_text = str(feature.get("target_response") or feature.get("response") or "")
        if not response_text:
            return hint_mask

        encoding = self.tokenizer(
            response_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        encoding_ids = encoding["input_ids"][:response_len]
        target_ids = response_ids.tolist()
        if encoding_ids != target_ids:
            return hint_mask

        candidate_spans = []
        reasoning_sentence_spans = self._find_reasoning_sentence_spans(response_text)
        for sentence_span in reasoning_sentence_spans:
            overlaps_visible_token = False
            for token_start, token_end in encoding["offset_mapping"][:response_len]:
                if token_end <= token_start:
                    continue
                if token_start < sentence_span[1] and token_end > sentence_span[0]:
                    overlaps_visible_token = True
                    break
            if overlaps_visible_token:
                candidate_spans.append(sentence_span)
        if not candidate_spans:
            return hint_mask

        chosen_idx = int(torch.randint(0, len(candidate_spans), (1,)).item())
        chosen_span = candidate_spans[chosen_idx]
        hint_mask_list = [False] * response_len
        self._apply_char_span_to_mask(hint_mask_list, encoding["offset_mapping"][:response_len], chosen_span)
        hint_mask = torch.tensor(hint_mask_list, dtype=torch.bool)
        reasoning_mask = reasoning_mask.to(dtype=torch.bool)[:response_len]
        hint_mask = hint_mask & reasoning_mask
        return hint_mask

    @staticmethod
    def _pad_tensor_batch(tensors: list[torch.Tensor], pad_value: int | bool):
        max_len = max(tensor.numel() for tensor in tensors)
        batch = torch.full(
            (len(tensors), max_len),
            pad_value,
            dtype=tensors[0].dtype,
        )
        for idx, tensor in enumerate(tensors):
            batch[idx, : tensor.numel()] = tensor
        return batch

    def _collate_response_batch(self, features: list[dict[str, Any]], field_name: str, prefix: str):
        response_ids_per_example = []
        answer_masks_per_example = []
        reasoning_masks_per_example = []
        response_lengths = []

        for feature in features:
            response_ids, answer_mask, reasoning_mask = self._tokenize_response_feature(feature, field_name)
            response_ids_per_example.append(response_ids)
            answer_masks_per_example.append(answer_mask)
            reasoning_masks_per_example.append(reasoning_mask)
            response_lengths.append(int(response_ids.numel()))

        return {
            f"{prefix}_response_ids": self._pad_tensor_batch(response_ids_per_example, self.pad_token_id),
            f"{prefix}_answer_mask": self._pad_tensor_batch(answer_masks_per_example, False),
            f"{prefix}_reasoning_mask": self._pad_tensor_batch(reasoning_masks_per_example, False),
            f"{prefix}_response_lengths": torch.tensor(response_lengths, dtype=torch.long),
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(features)
        t_values = self._sample_t_values(batch_size)
        self._t_sampling_batch_index += 1

        student_prompt_ids_per_example = []
        teacher_prompt_ids_per_example = []
        student_prompt_lengths = []
        teacher_prompt_lengths = []

        for feature in features:
            student_prompt_ids = self._truncate_prompt_ids(self._tokenize_text(feature["student_prompt"]))
            teacher_prompt_ids = self._truncate_prompt_ids(self._tokenize_text(feature["teacher_prompt"]))

            student_prompt_ids_per_example.append(torch.tensor(student_prompt_ids, dtype=torch.long))
            teacher_prompt_ids_per_example.append(torch.tensor(teacher_prompt_ids, dtype=torch.long))
            student_prompt_lengths.append(len(student_prompt_ids))
            teacher_prompt_lengths.append(len(teacher_prompt_ids))

        student_prompt_max_len = max(seq.numel() for seq in student_prompt_ids_per_example)
        teacher_prompt_max_len = max(seq.numel() for seq in teacher_prompt_ids_per_example)

        student_prompt_ids = torch.full(
            (batch_size, student_prompt_max_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        teacher_prompt_ids = torch.full(
            (batch_size, teacher_prompt_max_len),
            self.pad_token_id,
            dtype=torch.long,
        )

        for i in range(batch_size):
            student_prompt_ids[i, : student_prompt_ids_per_example[i].numel()] = student_prompt_ids_per_example[i]
            teacher_prompt_ids[i, : teacher_prompt_ids_per_example[i].numel()] = teacher_prompt_ids_per_example[i]

        collated = {
            "student_prompt_ids": student_prompt_ids,
            "teacher_prompt_ids": teacher_prompt_ids,
            "student_prompt_lengths": torch.tensor(student_prompt_lengths, dtype=torch.long),
            "teacher_prompt_lengths": torch.tensor(teacher_prompt_lengths, dtype=torch.long),
            "t_values": t_values,
        }
        collated.update(self._collate_response_batch(features, "gold_response", "gold"))
        collated.update(self._collate_response_batch(features, "target_response", "target"))
        collated.update(self._collate_response_batch(features, "inp_par_target_response", "inp_par"))
        collated.update(self._collate_response_batch(features, "rl_target_response", "rl"))
        collated.update(self._collate_response_batch(features, "tf_target_response", "tf"))

        target_reference_visible_reasoning_masks = []
        target_random_hint_masks = []
        for idx, feature in enumerate(features):
            target_response_len = int(collated["target_response_lengths"][idx].item())
            target_response_ids = collated["target_response_ids"][idx, :target_response_len]
            target_reasoning_mask = collated["target_reasoning_mask"][idx, :target_response_len]
            visible_reasoning_mask = self._build_reference_visible_reasoning_mask(
                feature=feature,
                response_ids=target_response_ids,
                reasoning_mask=target_reasoning_mask,
                teacher_prompt_token_limit=int(collated["teacher_prompt_lengths"][idx].item()),
            )
            target_reference_visible_reasoning_masks.append(visible_reasoning_mask)
            target_random_hint_masks.append(
                self._build_random_reasoning_sentence_hint_mask(
                    feature=feature,
                    response_ids=target_response_ids,
                    reasoning_mask=target_reasoning_mask,
                )
            )

        collated["target_reference_visible_reasoning_mask"] = self._pad_tensor_batch(
            target_reference_visible_reasoning_masks,
            False,
        )
        collated["target_random_hint_mask"] = self._pad_tensor_batch(
            target_random_hint_masks,
            False,
        )

        return collated


class DiffuSelfDistillTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = str(getattr(self.args, "dataset", "")).strip().lower()
        self.method_spec = str(getattr(self.args, "method_spec", getattr(self.args, "method", "ALL_MASK")))
        self.method = normalize_training_method(self.method_spec)
        if method_uses_one_hot_teacher(self.method) and float(getattr(self.args, "ce_weight", 0.0)) != 0.0:
            print(
                f"`method={self.method}` already uses masked-token hard-label KD, so forcing `ce_weight` "
                f"from {getattr(self.args, 'ce_weight')} to 0.0."
            )
            self.args.ce_weight = 0.0
        self.ce_mask_mode = normalize_ce_mask_mode(getattr(self.args, "ce_mask_mode", "answer"))
        self.args.ce_mask_mode = self.ce_mask_mode
        self._validate_required_target_paths()
        self._validate_ainp_settings()
        if getattr(self.args, "disable_dropout", False):
            # Teacher and student share weights; removing dropout stabilizes teacher targets.
            disable_dropout_in_model(self.model)
        self.debug_save_examples = max(int(getattr(self.args, "debug_save_examples", 0)), 0)
        self.debug_save_logits_topk = max(int(getattr(self.args, "debug_save_logits_topk", 0)), 1)
        self.debug_save_max_masked_positions = max(
            int(getattr(self.args, "debug_save_max_masked_positions", 0)),
            1,
        )
        self.debug_record_index = 0
        debug_filename = str(getattr(self.args, "debug_save_examples_filename", "debug_training_examples.jsonl")).strip()
        if not debug_filename:
            debug_filename = "debug_training_examples.jsonl"
        self.debug_examples_path = os.path.join(self.args.output_dir, debug_filename)
        self._debug_examples_initialized = False
        self.top1_logit_debug_path = os.path.join(self.args.output_dir, TOP1_LOGIT_DEBUG_FILENAME)
        self._top1_logit_debug_initialized = False
        self._top1_logit_debug_epoch_index = -1
        self._top1_logit_debug_logged_steps_in_epoch = 0
        self._top1_logit_debug_seen_global_steps_in_epoch: set[int] = set()
        self._reset_interval_log_accumulators()

    def _infer_t_sampling_total_batches(self, train_dataloader) -> int:
        configured_total_batches = max(int(getattr(self.data_collator, "t_curriculum_total_batches", 0)), 0)
        if configured_total_batches > 0:
            return configured_total_batches

        if self.args.max_steps and self.args.max_steps > 0:
            return max(int(self.args.max_steps) * max(int(self.args.gradient_accumulation_steps), 1), 1)

        try:
            dataloader_batches = len(train_dataloader)
        except TypeError:
            dataloader_batches = 0
        if dataloader_batches <= 0:
            return 0
        return max(int(math.ceil(float(dataloader_batches) * float(self.args.num_train_epochs))), 1)

    def get_train_dataloader(self):
        train_dataloader = super().get_train_dataloader()
        if hasattr(self.data_collator, "set_t_sampling_total_batches"):
            total_batches = self._infer_t_sampling_total_batches(train_dataloader)
            self.data_collator.set_t_sampling_total_batches(total_batches)
        return train_dataloader

    def train(self, *args, **kwargs):
        if hasattr(self.data_collator, "reset_t_sampling_state"):
            self.data_collator.reset_t_sampling_state()
        return super().train(*args, **kwargs)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        if "student_prompt_ids" not in inputs:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

        loss = loss.mean().detach()
        return loss, None, None

    def _get_required_external_target_methods(self) -> set[str]:
        if self.method in EXTERNAL_TARGET_METHODS:
            return {self.method}
        return set()

    def _validate_required_target_paths(self):
        for method_name in sorted(self._get_required_external_target_methods()):
            path_attr = METHOD_TO_TARGET_PATH_ATTR[method_name]
            configured_path = str(getattr(self.args, path_attr, "")).strip()
            if configured_path:
                continue
            raise ValueError(
                f"`method={method_name}` requires `{path_attr}` to be set for dataset `{self.dataset_name}`."
            )

    def _validate_ainp_settings(self):
        teacher_scale = float(getattr(self.args, "ainp_teacher_mask_ratio_scale", 0.5))
        if not 0.0 <= teacher_scale <= 1.0:
            raise ValueError("`ainp_teacher_mask_ratio_scale` must be in [0, 1].")

        student_t_min = float(getattr(self.args, "ainp_student_t_min", 0.0))
        if not 0.0 <= student_t_min <= 1.0:
            raise ValueError("`ainp_student_t_min` must be in [0, 1].")

        if self.method == "AINP":
            tau = float(getattr(self.args, "distill_temperature", 1.0))
            if abs(tau - 1.0) > 1e-8:
                raise ValueError("`method=AINP` currently expects `distill_temperature=1.0`.")

    def _resolve_ainp_student_t_value(self, t_value: torch.Tensor | float) -> float:
        if getattr(self.args, "ainp_student_full_mask", False):
            return 1.0

        raw_t_value = float(t_value.item()) if isinstance(t_value, torch.Tensor) else float(t_value)
        student_t = max(raw_t_value, self.data_collator.t_min)
        student_t = max(student_t, float(getattr(self.args, "ainp_student_t_min", 0.0)))
        return min(student_t, 1.0)

    def _reset_interval_log_accumulators(self):
        self._interval_log_count = 0
        self._interval_log_sums = {
            "kd_loss": 0.0,
            "ce_loss": 0.0,
            "mean_t": 0.0,
            "avg_mask_ratio": 0.0,
            "avg_student_mask_ratio": 0.0,
            "avg_response_length": 0.0,
            "masked_tokens": 0.0,
            "student_masked_tokens": 0.0,
        }
        for method_name in METHOD_METRIC_NAMES:
            self._interval_log_sums[f"batch_{method_name.lower()}_frac"] = 0.0

    def _accumulate_interval_log_metrics(
        self,
        inputs: dict[str, torch.Tensor],
        kd_loss: torch.Tensor,
        ce_loss: torch.Tensor,
    ):
        metrics = self._collect_batch_metrics(inputs)
        self._interval_log_count += 1
        self._interval_log_sums["kd_loss"] += float(kd_loss.detach().item())
        self._interval_log_sums["ce_loss"] += float(ce_loss.detach().item())
        self._interval_log_sums["mean_t"] += float(metrics.mean_t)
        self._interval_log_sums["avg_mask_ratio"] += float(metrics.avg_mask_ratio)
        self._interval_log_sums["avg_student_mask_ratio"] += float(metrics.avg_student_mask_ratio)
        self._interval_log_sums["avg_response_length"] += float(metrics.avg_response_length)
        self._interval_log_sums["masked_tokens"] += float(metrics.num_masked_tokens)
        self._interval_log_sums["student_masked_tokens"] += float(metrics.num_student_masked_tokens)
        for method_name, fraction in metrics.method_fractions.items():
            self._interval_log_sums[f"batch_{method_name.lower()}_frac"] += float(fraction)

    def log(self, logs: dict[str, float], *args, **kwargs):
        log_payload = dict(logs)
        is_eval_or_test_log = any(
            str(key).startswith("eval_") or str(key).startswith("test_") for key in log_payload.keys()
        )
        if self._interval_log_count > 0 and not is_eval_or_test_log:
            interval_count = float(self._interval_log_count)
            for key, total in self._interval_log_sums.items():
                log_payload[key] = total / interval_count
            self._reset_interval_log_accumulators()
        return super().log(log_payload, *args, **kwargs)

    def _iter_chunk_ranges(self, total_items: int):
        chunk_size = max(int(getattr(self.args, "loss_chunk_size", 0)), 0)
        if chunk_size <= 0 or total_items <= chunk_size:
            yield 0, total_items
            return

        for start in range(0, total_items, chunk_size):
            yield start, min(start + chunk_size, total_items)

    def _decode_token_ids(self, token_ids: torch.Tensor | list[int]) -> str:
        ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        if not ids:
            return ""
        return self.data_collator.tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _ensure_debug_examples_file(self):
        if self._debug_examples_initialized:
            return
        os.makedirs(os.path.dirname(self.debug_examples_path), exist_ok=True)
        if not os.path.exists(self.debug_examples_path):
            with open(self.debug_examples_path, "w", encoding="utf-8"):
                pass
        self._debug_examples_initialized = True

    def _ensure_top1_logit_debug_file(self):
        if self._top1_logit_debug_initialized:
            return
        os.makedirs(os.path.dirname(self.top1_logit_debug_path), exist_ok=True)
        if not os.path.exists(self.top1_logit_debug_path):
            with open(self.top1_logit_debug_path, "w", encoding="utf-8"):
                pass
        self._top1_logit_debug_initialized = True

    def _get_current_epoch_index(self) -> int:
        epoch_value = self.state.epoch
        if epoch_value is None:
            return 0
        return max(int(math.floor(float(epoch_value) + 1e-8)), 0)

    def _reserve_top1_logit_debug_step(self) -> tuple[int, int, int] | None:
        current_step = int(self.state.global_step)
        if current_step < 0:
            return None

        epoch_index = self._get_current_epoch_index()
        if epoch_index != self._top1_logit_debug_epoch_index:
            self._top1_logit_debug_epoch_index = epoch_index
            self._top1_logit_debug_logged_steps_in_epoch = 0
            self._top1_logit_debug_seen_global_steps_in_epoch = set()

        if current_step in self._top1_logit_debug_seen_global_steps_in_epoch:
            return None
        if self._top1_logit_debug_logged_steps_in_epoch >= TOP1_LOGIT_DEBUG_STEPS_PER_EPOCH:
            return None

        epoch_step_index = self._top1_logit_debug_logged_steps_in_epoch
        self._top1_logit_debug_seen_global_steps_in_epoch.add(current_step)
        self._top1_logit_debug_logged_steps_in_epoch += 1
        return current_step, epoch_index, epoch_step_index

    def _masked_top1_probs(self, response_logits: torch.Tensor, masked_positions: torch.Tensor) -> list[float]:
        if masked_positions.numel() == 0:
            return []
        masked_logits = response_logits.index_select(0, masked_positions).detach().float()
        top1_probs = F.softmax(masked_logits, dim=-1).max(dim=-1).values
        return [float(value.item()) for value in top1_probs.cpu()]

    def _masked_top1_logits(self, response_logits: torch.Tensor, masked_positions: torch.Tensor) -> list[float]:
        if masked_positions.numel() == 0:
            return []
        masked_logits = response_logits.index_select(0, masked_positions).detach().float()
        top1_logits = masked_logits.max(dim=-1).values
        return [float(value.item()) for value in top1_logits.cpu()]

    def _token_ids_to_pieces(self, token_ids: list[int]) -> list[str]:
        if not token_ids:
            return []
        token_pieces = self.data_collator.tokenizer.convert_ids_to_tokens([int(token_id) for token_id in token_ids])
        return [str(token_piece) for token_piece in token_pieces]

    def _mark_subsequence_sections(
        self,
        token_sections: list[str],
        token_ids: list[int],
        subsequence_ids: list[int],
        label: str,
    ):
        if not subsequence_ids or not token_ids:
            return

        start_idx = 0
        while start_idx < len(token_ids):
            found_idx = self.data_collator._find_subsequence(token_ids, subsequence_ids, start_idx=start_idx)
            if found_idx < 0:
                return
            end_idx = min(found_idx + len(subsequence_ids), len(token_sections))
            for token_idx in range(found_idx, end_idx):
                token_sections[token_idx] = label
            start_idx = end_idx

    def _build_response_token_sections(self, response_ids: torch.Tensor | list[int]) -> list[str]:
        token_ids = response_ids.tolist() if isinstance(response_ids, torch.Tensor) else list(response_ids)
        if not token_ids:
            return []

        token_sections = ["other"] * len(token_ids)
        reasoning_mask = self.data_collator._build_reasoning_mask_from_token_ids(token_ids)
        answer_mask = self.data_collator._build_answer_mask_from_token_ids(token_ids)

        for token_idx, is_reasoning in enumerate(reasoning_mask):
            if is_reasoning:
                token_sections[token_idx] = "reasoning_content"
        for token_idx, is_answer in enumerate(answer_mask):
            if is_answer:
                token_sections[token_idx] = "answer_content"

        self._mark_subsequence_sections(
            token_sections,
            token_ids,
            self.data_collator.reasoning_open_tag_ids,
            "reasoning_open_tag",
        )
        self._mark_subsequence_sections(
            token_sections,
            token_ids,
            self.data_collator.reasoning_close_tag_ids,
            "reasoning_close_tag",
        )
        self._mark_subsequence_sections(
            token_sections,
            token_ids,
            self.data_collator.answer_open_tag_ids,
            "answer_open_tag",
        )
        self._mark_subsequence_sections(
            token_sections,
            token_ids,
            self.data_collator.answer_close_tag_ids,
            "answer_close_tag",
        )
        return token_sections

    def _build_response_tag_mask(self, response_ids: torch.Tensor) -> torch.Tensor:
        token_sections = self._build_response_token_sections(response_ids)
        if not token_sections:
            return torch.zeros(0, dtype=torch.bool, device=response_ids.device)

        tag_section_names = {
            "reasoning_open_tag",
            "reasoning_close_tag",
            "answer_open_tag",
            "answer_close_tag",
        }
        return torch.tensor(
            [section_name in tag_section_names for section_name in token_sections],
            dtype=torch.bool,
            device=response_ids.device,
        )

    def _build_response_eligible_mask(
        self,
        response_ids: torch.Tensor,
        response_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not getattr(self.args, "unmask_xml_tags", False):
            return None

        if response_len <= 0:
            return torch.zeros(0, dtype=torch.bool, device=device)

        eligible_mask = ~self._build_response_tag_mask(response_ids)
        eligible_mask = eligible_mask.to(device=device, dtype=torch.bool)[:response_len]
        if eligible_mask.numel() < response_len:
            padded_mask = torch.zeros(response_len, dtype=torch.bool, device=device)
            padded_mask[: eligible_mask.numel()] = eligible_mask
            eligible_mask = padded_mask
        return eligible_mask

    def _build_masked_distribution_stats(
        self,
        response_logits: torch.Tensor,
        masked_positions: torch.Tensor,
        target_token_ids: torch.Tensor,
    ):
        if masked_positions.numel() == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=response_logits.device)
            empty_float = torch.empty(0, dtype=torch.float32, device=response_logits.device)
            return {
                "masked_probs": None,
                "top1_probs": empty_float,
                "top1_token_ids": empty_long,
                "target_token_ids": empty_long,
                "target_probs": empty_float,
                "entropies": empty_float,
            }

        masked_logits = response_logits.index_select(0, masked_positions).detach().float()
        masked_probs = F.softmax(masked_logits, dim=-1)
        top1_probs, top1_token_ids = masked_probs.max(dim=-1)
        masked_target_token_ids = target_token_ids.index_select(0, masked_positions).to(
            device=masked_probs.device,
            dtype=torch.long,
        )
        target_probs = masked_probs.gather(dim=-1, index=masked_target_token_ids.unsqueeze(-1)).squeeze(-1)
        entropies = -(masked_probs * masked_probs.clamp_min(1e-12).log()).sum(dim=-1)

        return {
            "masked_probs": masked_probs,
            "top1_probs": top1_probs,
            "top1_token_ids": top1_token_ids,
            "target_token_ids": masked_target_token_ids,
            "target_probs": target_probs,
            "entropies": entropies,
        }

    def _build_soft_sft_teacher_stats(
        self,
        masked_positions: torch.Tensor,
        target_token_ids: torch.Tensor,
    ):
        masked_target_token_ids = target_token_ids.index_select(0, masked_positions).detach().to(dtype=torch.long)
        num_masked_tokens = int(masked_target_token_ids.numel())
        teacher_top1_probs = torch.ones(num_masked_tokens, dtype=torch.float32, device=masked_target_token_ids.device)
        teacher_entropies = torch.zeros(num_masked_tokens, dtype=torch.float32, device=masked_target_token_ids.device)
        return {
            "masked_probs": None,
            "top1_probs": teacher_top1_probs,
            "top1_token_ids": masked_target_token_ids.clone(),
            "target_token_ids": masked_target_token_ids,
            "target_probs": teacher_top1_probs.clone(),
            "entropies": teacher_entropies,
        }

    def _build_one_hot_teacher_stats(
        self,
        target_token_ids: torch.Tensor,
        masked_positions: torch.Tensor,
    ):
        if masked_positions.numel() == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=target_token_ids.device)
            empty_float = torch.empty(0, dtype=torch.float32, device=target_token_ids.device)
            return {
                "masked_probs": None,
                "top1_probs": empty_float,
                "top1_token_ids": empty_long,
                "target_token_ids": empty_long,
                "target_probs": empty_float,
                "entropies": empty_float,
            }

        masked_target_token_ids = target_token_ids.index_select(0, masked_positions).detach().to(dtype=torch.long)
        num_masked_tokens = int(masked_target_token_ids.numel())
        teacher_top1_probs = torch.ones(
            num_masked_tokens,
            dtype=torch.float32,
            device=masked_target_token_ids.device,
        )
        teacher_entropies = torch.zeros(
            num_masked_tokens,
            dtype=torch.float32,
            device=masked_target_token_ids.device,
        )
        return {
            "masked_probs": None,
            "top1_probs": teacher_top1_probs,
            "top1_token_ids": masked_target_token_ids.clone(),
            "target_token_ids": masked_target_token_ids,
            "target_probs": teacher_top1_probs.clone(),
            "entropies": teacher_entropies,
        }

    def _build_topk_entries(
        self,
        masked_probs: torch.Tensor | None,
        top1_token_ids: torch.Tensor,
        top1_probs: torch.Tensor,
        selected_debug_indices: torch.Tensor,
    ) -> list[list[dict[str, Any]]]:
        if selected_debug_indices.numel() == 0:
            return []

        if masked_probs is None:
            selected_top1_token_ids = top1_token_ids.index_select(0, selected_debug_indices).detach().cpu().tolist()
            selected_top1_probs = top1_probs.index_select(0, selected_debug_indices).detach().cpu().tolist()
            selected_top1_token_pieces = self._token_ids_to_pieces(selected_top1_token_ids)
            return [
                [
                    {
                        "token_id": int(token_id),
                        "token_piece": token_piece,
                        "prob": float(token_prob),
                    }
                ]
                for token_id, token_piece, token_prob in zip(
                    selected_top1_token_ids,
                    selected_top1_token_pieces,
                    selected_top1_probs,
                )
            ]

        topk = min(int(self.debug_save_logits_topk), int(masked_probs.shape[-1]))
        selected_probs = masked_probs.index_select(0, selected_debug_indices)
        topk_probs, topk_token_ids = torch.topk(selected_probs, k=topk, dim=-1)
        topk_probs = topk_probs.detach().cpu().tolist()
        topk_token_ids = topk_token_ids.detach().cpu().tolist()

        topk_entries: list[list[dict[str, Any]]] = []
        for token_id_row, token_prob_row in zip(topk_token_ids, topk_probs):
            token_pieces = self._token_ids_to_pieces(token_id_row)
            topk_entries.append(
                [
                    {
                        "token_id": int(token_id),
                        "token_piece": token_piece,
                        "prob": float(token_prob),
                    }
                    for token_id, token_piece, token_prob in zip(token_id_row, token_pieces, token_prob_row)
                ]
            )
        return topk_entries

    def _maybe_dump_debug_examples(
        self,
        inputs: dict[str, torch.Tensor],
        teacher_logits: torch.Tensor | None,
        student_logits: torch.Tensor,
    ):
        if self.debug_save_examples <= 0:
            return
        if not self.is_world_process_zero():
            return
        current_step = int(self.state.global_step)
        if current_step < 0:
            return

        self._ensure_debug_examples_file()

        batch_size = inputs["student_input_ids"].shape[0]
        records: list[dict[str, Any]] = []

        for i in range(batch_size):
            response_len = int(inputs["response_lengths"][i].item())
            if response_len <= 0:
                continue

            selected_method = inputs["selected_methods"][i]
            student_prompt_len = int(inputs["student_prompt_lengths"][i].item())
            teacher_prompt_len = int(inputs["teacher_prompt_lengths"][i].item())
            gold_response_len = int(inputs["gold_response_lengths_used"][i].item())
            student_prompt_ids = inputs["student_input_ids"][i, :student_prompt_len]
            teacher_prompt_ids = inputs["teacher_input_ids"][i, :teacher_prompt_len]
            gold_response_ids = inputs["gold_response_ids"][i, :gold_response_len]
            shared_response_ids = inputs["response_ids"][i, :response_len]
            student_mask_i = inputs["student_mask"][i, :response_len]
            kd_mask_i = inputs["kd_mask"][i, :response_len]
            student_masked_response_ids = inputs["student_input_ids"][
                i, student_prompt_len : student_prompt_len + response_len
            ]
            teacher_masked_response_ids = inputs["teacher_input_ids"][
                i, teacher_prompt_len : teacher_prompt_len + response_len
            ]
            student_response_logits = student_logits[i, student_prompt_len : student_prompt_len + response_len, :]
            student_pred_ids = torch.argmax(student_response_logits, dim=-1)
            student_reconstructed_ids = torch.where(kd_mask_i, student_pred_ids, shared_response_ids)
            kd_positions = torch.nonzero(kd_mask_i, as_tuple=False).flatten()
            kd_loss_sum_i = student_response_logits.new_zeros((), dtype=torch.float32)
            kd_token_count_i = int(kd_positions.numel())
            response_token_sections = self._build_response_token_sections(shared_response_ids)
            masked_positions_list = kd_positions.detach().cpu().tolist()
            masked_position_fractions = [
                float(masked_position / max(response_len - 1, 1)) for masked_position in masked_positions_list
            ]
            masked_sections = [
                response_token_sections[masked_position] if masked_position < len(response_token_sections) else "other"
                for masked_position in masked_positions_list
            ]
            masked_target_token_ids = shared_response_ids.index_select(0, kd_positions)
            if selected_method == "SOFT_SFT":
                gold_target_ids = gold_response_ids[:response_len]
                teacher_pred_ids = gold_target_ids
                teacher_reconstructed_ids = torch.where(kd_mask_i, teacher_pred_ids, shared_response_ids)
                if kd_token_count_i > 0:
                    kd_loss_sum_i, kd_token_count_i = self._compute_soft_sft_kd_loss_sum(
                        student_response_logits=student_response_logits.detach(),
                        target_token_ids=gold_target_ids,
                        kd_positions=kd_positions,
                    )
                teacher_stats = self._build_soft_sft_teacher_stats(
                    masked_positions=kd_positions,
                    target_token_ids=shared_response_ids,
                )
            elif method_uses_one_hot_teacher(selected_method):
                teacher_pred_ids = shared_response_ids
                teacher_reconstructed_ids = shared_response_ids
                if kd_token_count_i > 0:
                    kd_loss_sum_i, kd_token_count_i = self._compute_one_hot_teacher_kd_loss_sum(
                        student_response_logits=student_response_logits.detach(),
                        teacher_target_ids=shared_response_ids,
                        kd_positions=kd_positions,
                    )
                teacher_stats = self._build_one_hot_teacher_stats(
                    target_token_ids=shared_response_ids,
                    masked_positions=kd_positions,
                )
            else:
                if teacher_logits is None:
                    raise RuntimeError(
                        f"Teacher logits are required for method `{selected_method}` but were not computed."
                    )
                teacher_response_logits = teacher_logits[i, teacher_prompt_len : teacher_prompt_len + response_len, :]
                teacher_pred_ids = torch.argmax(teacher_response_logits, dim=-1)
                teacher_reconstructed_ids = torch.where(kd_mask_i, teacher_pred_ids, shared_response_ids)
                if kd_token_count_i > 0:
                    kd_loss_sum_i, kd_token_count_i = self._compute_kd_loss_sum(
                        student_response_logits=student_response_logits.detach(),
                        teacher_response_logits=teacher_response_logits.detach(),
                        kd_positions=kd_positions,
                    )
                teacher_stats = self._build_masked_distribution_stats(
                    response_logits=teacher_response_logits,
                    masked_positions=kd_positions,
                    target_token_ids=shared_response_ids,
                )
            kd_loss_i = float((kd_loss_sum_i / max(kd_token_count_i, 1)).item()) if kd_token_count_i > 0 else 0.0
            student_stats = self._build_masked_distribution_stats(
                response_logits=student_response_logits,
                masked_positions=kd_positions,
                target_token_ids=shared_response_ids,
            )
            teacher_masked_top1_probs = teacher_stats["top1_probs"].detach().cpu().tolist()
            student_masked_top1_probs = student_stats["top1_probs"].detach().cpu().tolist()
            teacher_target_token_probs = teacher_stats["target_probs"].detach().cpu().tolist()
            student_target_token_probs = student_stats["target_probs"].detach().cpu().tolist()
            teacher_masked_entropies = teacher_stats["entropies"].detach().cpu().tolist()
            student_masked_entropies = student_stats["entropies"].detach().cpu().tolist()
            teacher_top1_token_ids = teacher_stats["top1_token_ids"].detach().cpu().tolist()
            student_top1_token_ids = student_stats["top1_token_ids"].detach().cpu().tolist()
            masked_target_token_ids_list = masked_target_token_ids.detach().cpu().tolist()

            teacher_top1_matches_target_tensor = teacher_stats["top1_token_ids"].eq(teacher_stats["target_token_ids"])
            student_top1_matches_target_tensor = student_stats["top1_token_ids"].eq(student_stats["target_token_ids"])
            teacher_student_top1_agreement_tensor = teacher_stats["top1_token_ids"].eq(student_stats["top1_token_ids"])
            teacher_top1_matches_target = [bool(value) for value in teacher_top1_matches_target_tensor.detach().cpu().tolist()]
            student_top1_matches_target = [bool(value) for value in student_top1_matches_target_tensor.detach().cpu().tolist()]
            teacher_student_top1_agreement = [
                bool(value) for value in teacher_student_top1_agreement_tensor.detach().cpu().tolist()
            ]

            teacher_top1_lt_099_section_counts = dict(
                Counter(
                    section
                    for section, top1_prob in zip(masked_sections, teacher_masked_top1_probs)
                    if top1_prob < 0.99
                )
            )
            teacher_target_prob_lt_050_section_counts = dict(
                Counter(
                    section
                    for section, target_prob in zip(masked_sections, teacher_target_token_probs)
                    if target_prob < 0.5
                )
            )

            if kd_token_count_i <= self.debug_save_max_masked_positions:
                selected_debug_indices = torch.arange(kd_token_count_i, device=kd_positions.device)
            else:
                selected_debug_indices = torch.argsort(teacher_stats["target_probs"])[: self.debug_save_max_masked_positions]

            teacher_topk_entries = self._build_topk_entries(
                masked_probs=teacher_stats["masked_probs"],
                top1_token_ids=teacher_stats["top1_token_ids"],
                top1_probs=teacher_stats["top1_probs"],
                selected_debug_indices=selected_debug_indices,
            )
            student_topk_entries = self._build_topk_entries(
                masked_probs=student_stats["masked_probs"],
                top1_token_ids=student_stats["top1_token_ids"],
                top1_probs=student_stats["top1_probs"],
                selected_debug_indices=selected_debug_indices,
            )

            selected_debug_indices_list = selected_debug_indices.detach().cpu().tolist()
            selected_target_token_ids = [masked_target_token_ids_list[masked_idx] for masked_idx in selected_debug_indices_list]
            selected_teacher_top1_token_ids = [teacher_top1_token_ids[masked_idx] for masked_idx in selected_debug_indices_list]
            selected_student_top1_token_ids = [student_top1_token_ids[masked_idx] for masked_idx in selected_debug_indices_list]
            selected_target_token_pieces = self._token_ids_to_pieces(selected_target_token_ids)
            selected_teacher_top1_token_pieces = self._token_ids_to_pieces(selected_teacher_top1_token_ids)
            selected_student_top1_token_pieces = self._token_ids_to_pieces(selected_student_top1_token_ids)

            masked_token_diagnostics = []
            for debug_rank, masked_idx in enumerate(selected_debug_indices_list):
                masked_token_diagnostics.append(
                    {
                        "debug_rank": int(debug_rank),
                        "response_token_index": int(masked_positions_list[masked_idx]),
                        "response_token_fraction": float(masked_position_fractions[masked_idx]),
                        "token_section": masked_sections[masked_idx],
                        "target_token_id": int(selected_target_token_ids[debug_rank]),
                        "target_token_piece": selected_target_token_pieces[debug_rank],
                        "teacher_top1_token_id": int(selected_teacher_top1_token_ids[debug_rank]),
                        "teacher_top1_token_piece": selected_teacher_top1_token_pieces[debug_rank],
                        "teacher_top1_prob": float(teacher_masked_top1_probs[masked_idx]),
                        "teacher_target_token_prob": float(teacher_target_token_probs[masked_idx]),
                        "teacher_entropy": float(teacher_masked_entropies[masked_idx]),
                        "teacher_top1_matches_target": bool(teacher_top1_matches_target[masked_idx]),
                        "teacher_topk": teacher_topk_entries[debug_rank],
                        "student_top1_token_id": int(selected_student_top1_token_ids[debug_rank]),
                        "student_top1_token_piece": selected_student_top1_token_pieces[debug_rank],
                        "student_top1_prob": float(student_masked_top1_probs[masked_idx]),
                        "student_target_token_prob": float(student_target_token_probs[masked_idx]),
                        "student_entropy": float(student_masked_entropies[masked_idx]),
                        "student_top1_matches_target": bool(student_top1_matches_target[masked_idx]),
                        "student_topk": student_topk_entries[debug_rank],
                        "teacher_student_top1_agree": bool(teacher_student_top1_agreement[masked_idx]),
                    }
                )

            records.append(
                {
                    "global_step": current_step,
                    "selected_method": inputs["selected_methods"][i],
                    "t_value": float(inputs["effective_t_values"][i].item()),
                    "raw_t_value": float(inputs["t_values"][i].item()),
                    "effective_t_value": float(inputs["effective_t_values"][i].item()),
                    "response_length": response_len,
                    "num_masked_tokens": int(kd_mask_i.sum().item()),
                    "num_student_masked_tokens": int(student_mask_i.sum().item()),
                    "num_teacher_masked_tokens": int(kd_mask_i.sum().item()),
                    "num_teacher_hint_tokens": int((student_mask_i & ~kd_mask_i).sum().item()),
                    "student_prompt_text": self._decode_token_ids(student_prompt_ids),
                    "teacher_prompt_text": self._decode_token_ids(teacher_prompt_ids),
                    "gold_response_text": self._decode_token_ids(gold_response_ids),
                    "target_response_text": self._decode_token_ids(shared_response_ids),
                    "shared_masked_response_text": self._decode_token_ids(student_masked_response_ids),
                    "teacher_masked_response_text": self._decode_token_ids(teacher_masked_response_ids),
                    "student_reconstructed_response_text": self._decode_token_ids(student_reconstructed_ids),
                    "teacher_reconstructed_response_text": self._decode_token_ids(teacher_reconstructed_ids),
                    "kd_loss": kd_loss_i,
                    "student_masked_top1_probs": student_masked_top1_probs,
                    "teacher_masked_top1_probs": teacher_masked_top1_probs,
                    "masked_response_positions": masked_positions_list,
                    "masked_response_position_fractions": masked_position_fractions,
                    "masked_response_sections": masked_sections,
                    "masked_target_token_ids": masked_target_token_ids_list,
                    "teacher_top1_token_ids": teacher_top1_token_ids,
                    "student_top1_token_ids": student_top1_token_ids,
                    "teacher_target_token_probs": teacher_target_token_probs,
                    "student_target_token_probs": student_target_token_probs,
                    "teacher_masked_entropies": teacher_masked_entropies,
                    "student_masked_entropies": student_masked_entropies,
                    "teacher_top1_matches_target": teacher_top1_matches_target,
                    "student_top1_matches_target": student_top1_matches_target,
                    "teacher_student_top1_agreement": teacher_student_top1_agreement,
                    "masked_section_counts": dict(Counter(masked_sections)),
                    "teacher_top1_lt_099_section_counts": teacher_top1_lt_099_section_counts,
                    "teacher_target_prob_lt_050_section_counts": teacher_target_prob_lt_050_section_counts,
                    "teacher_top1_prob_mean": float(sum(teacher_masked_top1_probs) / max(kd_token_count_i, 1)),
                    "student_top1_prob_mean": float(sum(student_masked_top1_probs) / max(kd_token_count_i, 1)),
                    "teacher_target_prob_mean": float(sum(teacher_target_token_probs) / max(kd_token_count_i, 1)),
                    "student_target_prob_mean": float(sum(student_target_token_probs) / max(kd_token_count_i, 1)),
                    "teacher_entropy_mean": float(sum(teacher_masked_entropies) / max(kd_token_count_i, 1)),
                    "student_entropy_mean": float(sum(student_masked_entropies) / max(kd_token_count_i, 1)),
                    "teacher_top1_matches_target_frac": float(
                        sum(1 for matched in teacher_top1_matches_target if matched) / max(kd_token_count_i, 1)
                    ),
                    "student_top1_matches_target_frac": float(
                        sum(1 for matched in student_top1_matches_target if matched) / max(kd_token_count_i, 1)
                    ),
                    "teacher_student_top1_agreement_frac": float(
                        sum(1 for matched in teacher_student_top1_agreement if matched) / max(kd_token_count_i, 1)
                    ),
                    "masked_token_debug_selection_strategy": "lowest_teacher_target_prob",
                    "masked_token_debug_saved_positions": int(len(masked_token_diagnostics)),
                    "masked_token_diagnostics": masked_token_diagnostics,
                }
            )

        if not records:
            return

        with open(self.debug_examples_path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.debug_record_index += len(records)

    def _maybe_dump_top1_logit_debug(
        self,
        inputs: dict[str, torch.Tensor],
        teacher_logits: torch.Tensor | None,
        student_logits: torch.Tensor,
    ):
        if not self.is_world_process_zero():
            return
        reserved_step = self._reserve_top1_logit_debug_step()
        if reserved_step is None:
            return
        current_step, epoch_index, epoch_step_index = reserved_step

        self._ensure_top1_logit_debug_file()

        batch_size = inputs["student_input_ids"].shape[0]
        records: list[dict[str, Any]] = []

        for i in range(batch_size):
            selected_method = inputs["selected_methods"][i]

            response_len = int(inputs["response_lengths"][i].item())
            if response_len <= 0:
                continue

            student_prompt_len = int(inputs["student_prompt_lengths"][i].item())
            kd_mask_i = inputs["kd_mask"][i, :response_len]
            kd_positions = torch.nonzero(kd_mask_i, as_tuple=False).flatten()
            shared_response_ids = inputs["response_ids"][i, :response_len]
            student_response_logits = student_logits[i, student_prompt_len : student_prompt_len + response_len, :]

            student_top1_token_ids = torch.argmax(student_response_logits, dim=-1)
            student_reconstructed_ids = torch.where(kd_mask_i, student_top1_token_ids, shared_response_ids)
            record = {
                "global_step": current_step,
                "epoch_index": epoch_index,
                "epoch_step_index": epoch_step_index,
                "selected_method": selected_method,
                "t_value": float(inputs["effective_t_values"][i].item()),
                "response_length": response_len,
                "num_masked_tokens": int(kd_positions.numel()),
                "target_response_text": self._decode_token_ids(shared_response_ids),
                "student_generated_text": self._decode_token_ids(student_reconstructed_ids),
                "masked_response_positions": kd_positions.detach().cpu().tolist(),
                "masked_target_token_ids": shared_response_ids.index_select(0, kd_positions).detach().cpu().tolist(),
                "student_top1_token_ids": student_top1_token_ids.index_select(0, kd_positions).detach().cpu().tolist(),
                "student_top1_logits": self._masked_top1_logits(
                    response_logits=student_response_logits,
                    masked_positions=kd_positions,
                ),
            }

            if selected_method == "SOFT_SFT" or method_uses_one_hot_teacher(selected_method):
                records.append(record)
                continue

            if teacher_logits is None:
                raise RuntimeError(
                    f"Teacher logits are required for method `{selected_method}` but were not computed."
                )

            teacher_prompt_len = int(inputs["teacher_prompt_lengths"][i].item())
            teacher_response_logits = teacher_logits[i, teacher_prompt_len : teacher_prompt_len + response_len, :]
            teacher_top1_token_ids = torch.argmax(teacher_response_logits, dim=-1)
            teacher_reconstructed_ids = torch.where(kd_mask_i, teacher_top1_token_ids, shared_response_ids)
            record.update(
                {
                    "teacher_generated_text": self._decode_token_ids(teacher_reconstructed_ids),
                    "teacher_top1_token_ids": teacher_top1_token_ids.index_select(0, kd_positions).detach().cpu().tolist(),
                    "teacher_top1_logits": self._masked_top1_logits(
                        response_logits=teacher_response_logits,
                        masked_positions=kd_positions,
                    ),
                }
            )
            records.append(record)

        if not records:
            return

        with open(self.top1_logit_debug_path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _compute_kd_loss_sum(
        self,
        student_response_logits: torch.Tensor,
        teacher_response_logits: torch.Tensor,
        kd_positions: torch.Tensor,
    ):
        if kd_positions.numel() == 0:
            return student_response_logits.new_zeros((), dtype=torch.float32), 0

        tau = self.args.distill_temperature
        kd_total = student_response_logits.new_zeros((), dtype=torch.float32)
        for start, end in self._iter_chunk_ranges(int(kd_positions.numel())):
            chunk_positions = kd_positions[start:end]
            student_chunk = student_response_logits.index_select(0, chunk_positions)
            teacher_chunk = teacher_response_logits.index_select(0, chunk_positions)
            student_log_probs = F.log_softmax(student_chunk / tau, dim=-1)
            teacher_probs = F.softmax(teacher_chunk / tau, dim=-1)
            kd_total = kd_total + F.kl_div(student_log_probs, teacher_probs, reduction="sum").float()

        return kd_total * (tau**2), int(kd_positions.numel())

    def _compute_soft_sft_kd_loss_sum(
        self,
        student_response_logits: torch.Tensor,
        target_token_ids: torch.Tensor,
        kd_positions: torch.Tensor,
    ):
        if kd_positions.numel() == 0:
            return student_response_logits.new_zeros((), dtype=torch.float32), 0

        tau = self.args.distill_temperature
        kd_total = student_response_logits.new_zeros((), dtype=torch.float32)
        for start, end in self._iter_chunk_ranges(int(kd_positions.numel())):
            chunk_positions = kd_positions[start:end]
            student_chunk = student_response_logits.index_select(0, chunk_positions)
            target_chunk = target_token_ids.index_select(0, chunk_positions).to(
                device=student_response_logits.device,
                dtype=torch.long,
            )
            kd_total = kd_total + F.cross_entropy(
                student_chunk / tau,
                target_chunk,
                reduction="sum",
            ).float()

        return kd_total * (tau**2), int(kd_positions.numel())

    def _compute_one_hot_teacher_kd_loss_sum(
        self,
        student_response_logits: torch.Tensor,
        teacher_target_ids: torch.Tensor,
        kd_positions: torch.Tensor,
    ):
        if kd_positions.numel() == 0:
            return student_response_logits.new_zeros((), dtype=torch.float32), 0

        tau = self.args.distill_temperature
        kd_total = student_response_logits.new_zeros((), dtype=torch.float32)
        for start, end in self._iter_chunk_ranges(int(kd_positions.numel())):
            chunk_positions = kd_positions[start:end]
            student_chunk = student_response_logits.index_select(0, chunk_positions)
            teacher_top1_token_ids = teacher_target_ids.index_select(0, chunk_positions).to(
                device=student_response_logits.device,
                dtype=torch.long,
            )
            kd_total = kd_total + F.cross_entropy(
                student_chunk / tau,
                teacher_top1_token_ids,
                reduction="sum",
            ).float()

        return kd_total * (tau**2), int(kd_positions.numel())

    def _compute_ce_loss_sum(
        self,
        student_response_logits: torch.Tensor,
        targets: torch.Tensor,
        ce_positions: torch.Tensor | None = None,
    ):
        if ce_positions is None:
            token_count = int(targets.numel())
        else:
            token_count = int(ce_positions.numel())

        if token_count == 0:
            return student_response_logits.new_zeros((), dtype=torch.float32), 0

        ce_total = student_response_logits.new_zeros((), dtype=torch.float32)
        for start, end in self._iter_chunk_ranges(token_count):
            if ce_positions is None:
                chunk_logits = student_response_logits[start:end]
                chunk_targets = targets[start:end]
            else:
                chunk_positions = ce_positions[start:end]
                chunk_logits = student_response_logits.index_select(0, chunk_positions)
                chunk_targets = targets.index_select(0, chunk_positions)
            ce_total = ce_total + F.cross_entropy(
                chunk_logits,
                chunk_targets,
                reduction="sum",
            ).float()

        return ce_total, token_count

    def _fit_example_to_max_length(
        self,
        student_prompt_ids: torch.Tensor,
        teacher_prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        *response_token_masks: torch.Tensor | None,
    ):
        max_prompt_len = max(self.data_collator.max_length - 1, 1)
        student_prompt_ids = student_prompt_ids[:max_prompt_len]
        teacher_prompt_ids = teacher_prompt_ids[:max_prompt_len]
        response_budget = self.data_collator.max_length - max(student_prompt_ids.numel(), teacher_prompt_ids.numel())
        response_budget = max(response_budget, 1)

        if response_ids.numel() == 0:
            response_ids = response_ids.new_tensor([self.data_collator.fallback_token_id], dtype=torch.long)
            normalized_masks = []
            for response_token_mask in response_token_masks:
                if response_token_mask is None:
                    normalized_masks.append(None)
                else:
                    normalized_masks.append(torch.zeros(1, dtype=torch.bool, device=response_ids.device))
            response_token_masks = tuple(normalized_masks)

        response_ids = response_ids[:response_budget]
        trimmed_masks = []
        for response_token_mask in response_token_masks:
            if response_token_mask is None:
                trimmed_masks.append(None)
                continue
            trimmed_masks.append(response_token_mask[: response_ids.numel()])
        return student_prompt_ids, teacher_prompt_ids, response_ids, *trimmed_masks

    def _build_kd_mask(
        self,
        method: str,
        response_len: int,
        t_value: torch.Tensor,
        device: torch.device,
        eligible_mask: torch.Tensor | None = None,
    ):
        if response_len <= 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        if eligible_mask is None:
            eligible_mask = torch.ones(response_len, dtype=torch.bool, device=device)
        else:
            eligible_mask = eligible_mask.to(device=device, dtype=torch.bool)[:response_len]
            if eligible_mask.numel() < response_len:
                padded_mask = torch.zeros(response_len, dtype=torch.bool, device=device)
                padded_mask[: eligible_mask.numel()] = eligible_mask
                eligible_mask = padded_mask
        eligible_positions = torch.nonzero(eligible_mask, as_tuple=False).flatten()
        if eligible_positions.numel() == 0:
            return torch.zeros(response_len, dtype=torch.bool, device=device)
        if method in {"ALL_MASK"}:
            return eligible_mask.clone()

        mask = (torch.rand(response_len, device=device) < float(t_value.item())) & eligible_mask
        if not mask.any():
            sampled_idx = eligible_positions[
                torch.randint(0, eligible_positions.numel(), (1,), device=device).item()
            ]
            mask[sampled_idx] = True
        return mask

    @staticmethod
    def _normalize_token_mask(
        response_len: int,
        token_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        normalized_mask = torch.zeros(response_len, dtype=torch.bool, device=device)
        if response_len <= 0 or token_mask is None:
            return normalized_mask

        token_mask = token_mask.to(device=device, dtype=torch.bool)[:response_len]
        normalized_mask[: token_mask.numel()] = token_mask
        return normalized_mask

    def _build_ainp_masks(
        self,
        response_len: int,
        t_value: torch.Tensor,
        device: torch.device,
        response_answer_mask: torch.Tensor,
        eligible_mask: torch.Tensor | None = None,
    ):
        if response_len <= 0:
            empty_mask = torch.zeros(0, dtype=torch.bool, device=device)
            return empty_mask, empty_mask

        normalized_eligible_mask = self._normalize_token_mask(
            response_len=response_len,
            token_mask=eligible_mask,
            device=device,
        )
        if eligible_mask is None:
            normalized_eligible_mask = torch.ones(response_len, dtype=torch.bool, device=device)

        teacher_forced_mask = torch.zeros(response_len, dtype=torch.bool, device=device)
        if getattr(self.args, "ainp_teacher_answer_always_mask", True):
            teacher_forced_mask = self._normalize_token_mask(
                response_len=response_len,
                token_mask=response_answer_mask,
                device=device,
            )
            teacher_forced_mask = teacher_forced_mask & normalized_eligible_mask

        if getattr(self.args, "ainp_student_full_mask", False):
            student_mask = normalized_eligible_mask.clone()
        else:
            student_t = self._resolve_ainp_student_t_value(t_value)
            student_mask = self._build_kd_mask(
                method="INP",
                response_len=response_len,
                t_value=torch.tensor(student_t, device=device),
                device=device,
                eligible_mask=normalized_eligible_mask,
            )
        student_mask = student_mask | teacher_forced_mask

        teacher_scale = float(getattr(self.args, "ainp_teacher_mask_ratio_scale", 0.5))
        if teacher_scale >= 1.0:
            teacher_mask = student_mask.clone()
        else:
            teacher_mask = torch.zeros(response_len, dtype=torch.bool, device=device)
            student_masked_positions = torch.nonzero(student_mask, as_tuple=False).flatten()
            if student_masked_positions.numel() > 0 and teacher_scale > 0.0:
                sampled_teacher_positions = student_masked_positions[
                    torch.rand(student_masked_positions.numel(), device=device) < teacher_scale
                ]
                if sampled_teacher_positions.numel() > 0:
                    teacher_mask[sampled_teacher_positions] = True

        teacher_mask = (teacher_mask | teacher_forced_mask) & student_mask

        if not teacher_mask.any():
            student_masked_positions = torch.nonzero(student_mask, as_tuple=False).flatten()
            if student_masked_positions.numel() == 0:
                eligible_positions = torch.nonzero(normalized_eligible_mask, as_tuple=False).flatten()
                if eligible_positions.numel() == 0:
                    return student_mask, teacher_mask
                sampled_idx = eligible_positions[
                    torch.randint(0, eligible_positions.numel(), (1,), device=device).item()
                ]
                student_mask[sampled_idx] = True
                student_masked_positions = sampled_idx.unsqueeze(0)
            sampled_teacher_idx = student_masked_positions[
                torch.randint(0, student_masked_positions.numel(), (1,), device=device).item()
            ]
            teacher_mask[sampled_teacher_idx] = True

        return student_mask, teacher_mask

    @staticmethod
    def _extract_answer_token_ids(response_ids: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
        if response_ids.numel() == 0:
            return response_ids.new_empty((0,), dtype=torch.long)
        answer_mask = answer_mask.to(device=response_ids.device, dtype=torch.bool)[: response_ids.numel()]
        answer_positions = torch.nonzero(answer_mask, as_tuple=False).flatten()
        if answer_positions.numel() == 0:
            return response_ids.new_empty((0,), dtype=torch.long)
        return response_ids.index_select(0, answer_positions)

    @staticmethod
    def _build_answer_ce_targets(
        response_len: int,
        response_answer_mask: torch.Tensor,
        answer_target_ids: torch.Tensor,
        device: torch.device,
    ):
        ce_target_ids = torch.full((response_len,), -100, dtype=torch.long, device=device)
        ce_mask = torch.zeros((response_len,), dtype=torch.bool, device=device)
        if response_len <= 0 or answer_target_ids.numel() == 0:
            return ce_target_ids, ce_mask

        response_answer_mask = response_answer_mask.to(device=device, dtype=torch.bool)[:response_len]
        answer_positions = torch.nonzero(response_answer_mask, as_tuple=False).flatten()
        if answer_positions.numel() == 0:
            return ce_target_ids, ce_mask

        num_answer_tokens = min(int(answer_positions.numel()), int(answer_target_ids.numel()))
        if num_answer_tokens <= 0:
            return ce_target_ids, ce_mask

        selected_positions = answer_positions[:num_answer_tokens]
        selected_targets = answer_target_ids[:num_answer_tokens].to(device=device, dtype=torch.long)
        ce_target_ids[selected_positions] = selected_targets
        ce_mask[selected_positions] = True
        return ce_target_ids, ce_mask

    @staticmethod
    def _build_full_ce_targets(
        response_ids: torch.Tensor,
        device: torch.device,
    ):
        response_len = int(response_ids.numel())
        ce_target_ids = torch.full((response_len,), -100, dtype=torch.long, device=device)
        ce_mask = torch.zeros((response_len,), dtype=torch.bool, device=device)
        if response_len <= 0:
            return ce_target_ids, ce_mask

        ce_target_ids[:response_len] = response_ids.to(device=device, dtype=torch.long)[:response_len]
        ce_mask[:response_len] = True
        return ce_target_ids, ce_mask

    @staticmethod
    def _build_masked_ce_targets(
        response_ids: torch.Tensor,
        response_kd_mask: torch.Tensor,
        device: torch.device,
    ):
        response_len = int(response_ids.numel())
        ce_target_ids = torch.full((response_len,), -100, dtype=torch.long, device=device)
        ce_mask = torch.zeros((response_len,), dtype=torch.bool, device=device)
        if response_len <= 0:
            return ce_target_ids, ce_mask

        response_kd_mask = response_kd_mask.to(device=device, dtype=torch.bool)[:response_len]
        masked_positions = torch.nonzero(response_kd_mask, as_tuple=False).flatten()
        if masked_positions.numel() == 0:
            return ce_target_ids, ce_mask

        masked_target_ids = response_ids.index_select(0, masked_positions).to(device=device, dtype=torch.long)
        ce_target_ids[masked_positions] = masked_target_ids
        ce_mask[masked_positions] = True
        return ce_target_ids, ce_mask

    def _build_ce_targets(
        self,
        response_ids: torch.Tensor,
        response_len: int,
        response_kd_mask: torch.Tensor,
        response_answer_mask: torch.Tensor,
        answer_target_ids: torch.Tensor,
        device: torch.device,
    ):
        if self.ce_mask_mode == "full":
            return self._build_full_ce_targets(
                response_ids=response_ids[:response_len],
                device=device,
            )
        if self.ce_mask_mode == "masked":
            return self._build_masked_ce_targets(
                response_ids=response_ids[:response_len],
                response_kd_mask=response_kd_mask[:response_len],
                device=device,
            )

        return self._build_answer_ce_targets(
            response_len=response_len,
            response_answer_mask=response_answer_mask,
            answer_target_ids=answer_target_ids,
            device=device,
        )

    def _get_method_response_bundle(self, inputs: dict[str, torch.Tensor], index: int, method: str):
        response_prefix_by_method = {
            "ALL_MASK": "target",
            "INP": "target",
            "INP_OH": "target",
            "INP_OH_PAD": "target",
        }
        response_prefix = response_prefix_by_method[method]
        if method_uses_padded_target_response(method):
            response_ids = inputs[f"{response_prefix}_response_ids"][index]
            answer_mask = inputs[f"{response_prefix}_answer_mask"][index]
            reasoning_mask = inputs[f"{response_prefix}_reasoning_mask"][index]
        else:
            response_len = int(inputs[f"{response_prefix}_response_lengths"][index].item())
            response_ids = inputs[f"{response_prefix}_response_ids"][index, :response_len]
            answer_mask = inputs[f"{response_prefix}_answer_mask"][index, :response_len]
            reasoning_mask = inputs[f"{response_prefix}_reasoning_mask"][index, :response_len]
        return response_ids, answer_mask, reasoning_mask

    def _prepare_batch_inputs(self, model, inputs: dict[str, torch.Tensor]):
        batch_size = inputs["student_prompt_ids"].shape[0]
        device = inputs["student_prompt_ids"].device

        student_sequences = []
        teacher_sequences = []
        gold_response_ids_per_example = []
        response_ids_per_example = []
        ce_target_ids_per_example = []
        student_masks = []
        kd_masks = []
        ce_masks = []
        gold_response_lengths_original = []
        gold_response_lengths_used = []
        response_lengths = []
        student_prompt_lengths = []
        teacher_prompt_lengths = []
        selected_methods = []
        effective_t_values = []

        for i in range(batch_size):
            student_prompt_len = int(inputs["student_prompt_lengths"][i].item())
            teacher_prompt_len = int(inputs["teacher_prompt_lengths"][i].item())
            gold_response_len = int(inputs["gold_response_lengths"][i].item())

            student_prompt_ids = inputs["student_prompt_ids"][i, :student_prompt_len]
            teacher_prompt_ids = inputs["teacher_prompt_ids"][i, :teacher_prompt_len]
            gold_response_ids = inputs["gold_response_ids"][i, :gold_response_len]
            gold_answer_mask = inputs["gold_answer_mask"][i, :gold_response_len]
            gold_reasoning_mask = inputs["gold_reasoning_mask"][i, :gold_response_len]
            gold_response_lengths_original.append(gold_response_len)
            gold_answer_target_ids = self._extract_answer_token_ids(
                gold_response_ids,
                gold_answer_mask[: gold_response_ids.numel()],
            )

            effective_method = self.method
            selected_response_ids, selected_answer_mask, selected_reasoning_mask = self._get_method_response_bundle(
                inputs=inputs,
                index=i,
                method=effective_method,
            )
            selected_reference_visible_reasoning_mask = inputs["target_reference_visible_reasoning_mask"][
                i, : selected_response_ids.numel()
            ]
            selected_random_hint_mask = inputs["target_random_hint_mask"][i, : selected_response_ids.numel()]
            if method_uses_one_hot_teacher(effective_method):
                # One-hot teacher methods do not consume teacher logits, so a privileged
                # teacher prompt should not reduce the remaining response budget.
                teacher_prompt_ids = student_prompt_ids.clone()
            if effective_method == "AINP":
                teacher_prompt_ids = student_prompt_ids.clone()
            (
                student_prompt_ids,
                teacher_prompt_ids,
                response_ids,
                response_answer_mask,
                response_reasoning_mask,
                response_reference_visible_reasoning_mask,
                response_random_hint_mask,
            ) = self._fit_example_to_max_length(
                student_prompt_ids,
                teacher_prompt_ids,
                selected_response_ids,
                selected_answer_mask,
                selected_reasoning_mask,
                selected_reference_visible_reasoning_mask,
                selected_random_hint_mask,
            )

            response_len = int(response_ids.numel())
            gold_response_ids = gold_response_ids[:response_len]
            gold_response_lengths_used.append(int(gold_response_ids.numel()))
            raw_t_value = inputs["t_values"][i]
            response_eligible_mask = self._build_response_eligible_mask(
                response_ids=response_ids,
                response_len=response_len,
                device=device,
            )

            if effective_method == "AINP":
                effective_t_value = self._resolve_ainp_student_t_value(raw_t_value)
                student_mask, kd_mask = self._build_ainp_masks(
                    response_len=response_len,
                    t_value=raw_t_value,
                    device=device,
                    response_answer_mask=response_answer_mask,
                    eligible_mask=response_eligible_mask,
                )
            elif effective_method == "INP_RKD_ACE":
                effective_t_value = float(raw_t_value.item())
                reasoning_eligible_mask = self._normalize_token_mask(
                    response_len=response_len,
                    token_mask=response_reasoning_mask,
                    device=device,
                )
                if response_eligible_mask is not None:
                    reasoning_eligible_mask = reasoning_eligible_mask & response_eligible_mask
                kd_mask = self._build_kd_mask(
                    method="INP",
                    response_len=response_len,
                    t_value=raw_t_value,
                    device=device,
                    eligible_mask=reasoning_eligible_mask,
                )
                student_mask = kd_mask
            elif effective_method == "INP_RKD_ACE_UNSEEN":
                effective_t_value = float(raw_t_value.item())
                reasoning_eligible_mask = self._normalize_token_mask(
                    response_len=response_len,
                    token_mask=response_reasoning_mask,
                    device=device,
                )
                visible_reasoning_mask = self._normalize_token_mask(
                    response_len=response_len,
                    token_mask=response_reference_visible_reasoning_mask,
                    device=device,
                )
                hidden_reasoning_mask = reasoning_eligible_mask & ~visible_reasoning_mask
                if response_eligible_mask is not None:
                    hidden_reasoning_mask = hidden_reasoning_mask & response_eligible_mask
                kd_mask = self._build_kd_mask(
                    method="INP",
                    response_len=response_len,
                    t_value=raw_t_value,
                    device=device,
                    eligible_mask=hidden_reasoning_mask,
                )
                student_mask = kd_mask
            elif effective_method == "INP_PAR_REVERSE_HINT":
                effective_t_value = float(raw_t_value.item())
                hint_visible_mask = self._normalize_token_mask(
                    response_len=response_len,
                    token_mask=response_random_hint_mask,
                    device=device,
                )
                if response_eligible_mask is None:
                    hint_excluded_eligible_mask = ~hint_visible_mask
                else:
                    hint_excluded_eligible_mask = response_eligible_mask & ~hint_visible_mask
                kd_mask = self._build_kd_mask(
                    method="INP",
                    response_len=response_len,
                    t_value=raw_t_value,
                    device=device,
                    eligible_mask=hint_excluded_eligible_mask,
                )
                student_mask = kd_mask
            else:
                effective_t_value = float(raw_t_value.item())
                kd_mask = self._build_kd_mask(
                    method=effective_method,
                    response_len=response_len,
                    t_value=raw_t_value,
                    device=device,
                    eligible_mask=response_eligible_mask,
                )
                student_mask = kd_mask

            if self.args.ce_weight > 0:
                ce_target_ids, ce_mask = self._build_ce_targets(
                    response_ids=response_ids,
                    response_len=response_len,
                    response_kd_mask=kd_mask,
                    response_answer_mask=response_answer_mask,
                    answer_target_ids=gold_answer_target_ids,
                    device=device,
                )
            else:
                ce_target_ids = torch.full((response_len,), -100, dtype=torch.long, device=device)
                ce_mask = torch.zeros((response_len,), dtype=torch.bool, device=device)
            student_masked_response_ids = torch.where(
                student_mask,
                torch.full_like(response_ids, self.data_collator.mask_id),
                response_ids,
            )
            teacher_masked_response_ids = torch.where(
                kd_mask,
                torch.full_like(response_ids, self.data_collator.mask_id),
                response_ids,
            )

            student_sequences.append(torch.cat([student_prompt_ids, student_masked_response_ids], dim=0))
            teacher_sequences.append(torch.cat([teacher_prompt_ids, teacher_masked_response_ids], dim=0))
            gold_response_ids_per_example.append(gold_response_ids)
            response_ids_per_example.append(response_ids)
            ce_target_ids_per_example.append(ce_target_ids)
            student_masks.append(student_mask)
            kd_masks.append(kd_mask)
            ce_masks.append(ce_mask)
            response_lengths.append(response_len)
            student_prompt_lengths.append(int(student_prompt_ids.numel()))
            teacher_prompt_lengths.append(int(teacher_prompt_ids.numel()))
            selected_methods.append(effective_method)
            effective_t_values.append(effective_t_value)

        student_max_len = max(seq.numel() for seq in student_sequences)
        teacher_max_len = max(seq.numel() for seq in teacher_sequences)
        response_max_len = max(resp.numel() for resp in response_ids_per_example)

        student_input_ids = torch.full(
            (batch_size, student_max_len),
            self.data_collator.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        teacher_input_ids = torch.full(
            (batch_size, teacher_max_len),
            self.data_collator.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        student_attention_mask = torch.zeros((batch_size, student_max_len), dtype=torch.long, device=device)
        teacher_attention_mask = torch.zeros((batch_size, teacher_max_len), dtype=torch.long, device=device)
        gold_response_ids = torch.full((batch_size, response_max_len), -100, dtype=torch.long, device=device)
        response_ids = torch.full((batch_size, response_max_len), -100, dtype=torch.long, device=device)
        ce_target_ids = torch.full((batch_size, response_max_len), -100, dtype=torch.long, device=device)
        student_mask = torch.zeros((batch_size, response_max_len), dtype=torch.bool, device=device)
        kd_mask = torch.zeros((batch_size, response_max_len), dtype=torch.bool, device=device)
        ce_mask = torch.zeros((batch_size, response_max_len), dtype=torch.bool, device=device)
        response_mask = torch.zeros((batch_size, response_max_len), dtype=torch.bool, device=device)

        for i in range(batch_size):
            student_input_ids[i, : student_sequences[i].numel()] = student_sequences[i]
            teacher_input_ids[i, : teacher_sequences[i].numel()] = teacher_sequences[i]
            student_attention_mask[i, : student_sequences[i].numel()] = 1
            teacher_attention_mask[i, : teacher_sequences[i].numel()] = 1
            gold_response_ids[i, : gold_response_ids_per_example[i].numel()] = gold_response_ids_per_example[i]
            response_ids[i, : response_ids_per_example[i].numel()] = response_ids_per_example[i]
            ce_target_ids[i, : ce_target_ids_per_example[i].numel()] = ce_target_ids_per_example[i]
            student_mask[i, : student_masks[i].numel()] = student_masks[i]
            kd_mask[i, : kd_masks[i].numel()] = kd_masks[i]
            ce_mask[i, : ce_masks[i].numel()] = ce_masks[i]
            response_mask[i, : response_ids_per_example[i].numel()] = True

        return {
            "student_input_ids": student_input_ids,
            "student_attention_mask": student_attention_mask,
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "gold_response_ids": gold_response_ids,
            "response_ids": response_ids,
            "ce_target_ids": ce_target_ids,
            "response_mask": response_mask,
            "student_mask": student_mask,
            "kd_mask": kd_mask,
            "ce_mask": ce_mask,
            "student_prompt_lengths": torch.tensor(student_prompt_lengths, dtype=torch.long, device=device),
            "teacher_prompt_lengths": torch.tensor(teacher_prompt_lengths, dtype=torch.long, device=device),
            "gold_response_lengths_original": torch.tensor(
                gold_response_lengths_original,
                dtype=torch.long,
                device=device,
            ),
            "gold_response_lengths_used": torch.tensor(
                gold_response_lengths_used,
                dtype=torch.long,
                device=device,
            ),
            "response_lengths": torch.tensor(response_lengths, dtype=torch.long, device=device),
            "selected_methods": selected_methods,
            "t_values": inputs["t_values"],
            "effective_t_values": torch.tensor(effective_t_values, dtype=torch.float32, device=device),
        }

    def _collect_batch_metrics(self, inputs: dict[str, torch.Tensor]) -> BatchMetrics:
        response_lengths = inputs["response_lengths"].float()
        response_lengths_clamped = response_lengths.clamp_min(1.0)
        teacher_mask_ratio = inputs["kd_mask"].float().sum(dim=1) / response_lengths_clamped
        student_mask_ratio = inputs["student_mask"].float().sum(dim=1) / response_lengths_clamped
        selected_methods = inputs.get("selected_methods") or []
        total_selected = max(len(selected_methods), 1)
        method_fractions = {
            method_name: sum(selected_method == method_name for selected_method in selected_methods) / total_selected
            for method_name in METHOD_METRIC_NAMES
        }
        return BatchMetrics(
            mean_t=inputs["t_values"].float().mean().item(),
            avg_mask_ratio=teacher_mask_ratio.mean().item(),
            avg_student_mask_ratio=student_mask_ratio.mean().item(),
            avg_response_length=response_lengths.mean().item(),
            num_masked_tokens=inputs["kd_mask"].float().sum().item(),
            num_student_masked_tokens=inputs["student_mask"].float().sum().item(),
            method_fractions=method_fractions,
        )

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        inputs = self._prepare_batch_inputs(model=model, inputs=inputs)

        teacher_logits = None
        selected_methods = inputs.get("selected_methods") or []
        # One-hot teacher methods use the active target tokens as hard labels, so no teacher forward is needed.
        requires_teacher_logits = any(
            selected_method != "SOFT_SFT" and not method_uses_one_hot_teacher(selected_method)
            for selected_method in selected_methods
        )
        if requires_teacher_logits:
            with torch.no_grad():
                teacher_outputs = model(
                    input_ids=inputs["teacher_input_ids"],
                    attention_mask=inputs["teacher_attention_mask"],
                )
                teacher_logits = teacher_outputs.logits
            del teacher_outputs

        student_outputs = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )
        student_logits = student_outputs.logits
        del student_outputs

        if model.training:
            self._maybe_dump_debug_examples(
                inputs=inputs,
                teacher_logits=teacher_logits,
                student_logits=student_logits,
            )
            self._maybe_dump_top1_logit_debug(
                inputs=inputs,
                teacher_logits=teacher_logits,
                student_logits=student_logits,
            )

        kd_loss_sum = student_logits.new_zeros((), dtype=torch.float32)
        kd_token_count = 0
        ce_loss_sum = student_logits.new_zeros((), dtype=torch.float32)
        ce_token_count = 0

        batch_size = inputs["student_input_ids"].shape[0]
        ce_target_ids = inputs["ce_target_ids"]
        kd_mask = inputs["kd_mask"]
        ce_mask = inputs["ce_mask"]

        for i in range(batch_size):
            response_len = int(inputs["response_lengths"][i].item())
            if response_len <= 0:
                continue

            selected_method = inputs["selected_methods"][i]
            student_prompt_len = int(inputs["student_prompt_lengths"][i].item())
            teacher_prompt_len = int(inputs["teacher_prompt_lengths"][i].item())

            student_response_logits = student_logits[i, student_prompt_len : student_prompt_len + response_len, :]
            kd_mask_i = kd_mask[i, :response_len]

            if kd_mask_i.any():
                kd_positions = torch.nonzero(kd_mask_i, as_tuple=False).flatten()
                if selected_method == "SOFT_SFT":
                    kd_chunk_sum, kd_chunk_count = self._compute_soft_sft_kd_loss_sum(
                        student_response_logits=student_response_logits,
                        target_token_ids=inputs["response_ids"][i, :response_len],
                        kd_positions=kd_positions,
                    )
                elif method_uses_one_hot_teacher(selected_method):
                    kd_chunk_sum, kd_chunk_count = self._compute_one_hot_teacher_kd_loss_sum(
                        student_response_logits=student_response_logits,
                        teacher_target_ids=inputs["response_ids"][i, :response_len],
                        kd_positions=kd_positions,
                    )
                else:
                    if teacher_logits is None:
                        raise RuntimeError(
                            f"Teacher logits are required for method `{selected_method}` but were not computed."
                        )
                    teacher_response_logits = teacher_logits[i, teacher_prompt_len : teacher_prompt_len + response_len, :]
                    kd_chunk_sum, kd_chunk_count = self._compute_kd_loss_sum(
                        student_response_logits=student_response_logits,
                        teacher_response_logits=teacher_response_logits,
                        kd_positions=kd_positions,
                    )
                kd_loss_sum = kd_loss_sum + kd_chunk_sum
                kd_token_count += kd_chunk_count

            if self.args.ce_weight > 0:
                ce_mask_i = ce_mask[i, :response_len]
                if ce_mask_i.any():
                    ce_positions = torch.nonzero(ce_mask_i, as_tuple=False).flatten()
                    ce_chunk_sum, ce_chunk_count = self._compute_ce_loss_sum(
                        student_response_logits=student_response_logits,
                        targets=ce_target_ids[i, :response_len],
                        ce_positions=ce_positions,
                    )
                    ce_loss_sum = ce_loss_sum + ce_chunk_sum
                    ce_token_count += ce_chunk_count

        if kd_token_count > 0:
            kd_loss = kd_loss_sum / kd_token_count
        else:
            kd_loss = student_logits.sum() * 0.0

        if self.args.ce_weight > 0 and ce_token_count > 0:
            ce_loss = ce_loss_sum / ce_token_count
        else:
            ce_loss = student_logits.sum() * 0.0

        loss = self.args.kd_weight * kd_loss + self.args.ce_weight * ce_loss

        if model.training:
            self._accumulate_interval_log_metrics(
                inputs=inputs,
                kd_loss=kd_loss,
                ce_loss=ce_loss,
            )

        if return_outputs:
            return loss, {"student_logits": student_logits}
        return loss
