from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


METHOD_ALIASES = {
    "INP-OH": "SFT",
    "INPOH": "SFT",
    "INP_OH": "SFT",
}
VALID_METHODS = {"SFT"}
T_SAMPLING_MODE_ALIASES = {
    "biased-to-one": "biased_to_one",
    "biasedtoone": "biased_to_one",
    "high-bias": "biased_to_one",
    "highbias": "biased_to_one",
    "two-point": "two_point",
    "twopoint": "two_point",
}
VALID_T_SAMPLING_MODES = {"uniform", "fixed", "two_point", "curriculum", "biased_to_one"}
TOP1_LOGIT_DEBUG_STEPS_PER_EPOCH = 30
TOP1_LOGIT_DEBUG_FILENAME = "debug_top1_logits_first100.jsonl"


def _normalize_method_name(method: str) -> str:
    normalized = str(method or "").strip().upper()
    return METHOD_ALIASES.get(normalized, normalized)


def normalize_training_method(method: str) -> str:
    normalized = _normalize_method_name(method) or "SFT"
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
    Tokenize prompts and target responses for SFT training.
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
        self.dataset_name = str(dataset_name or "").strip().lower()
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must provide pad_token_id or eos_token_id.")
        self.fallback_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else self.pad_token_id

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

    def _tokenize_text(self, text: str):
        return self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=False,
        )["input_ids"]

    def _truncate_prompt_ids(self, prompt_ids: list[int]) -> list[int]:
        return prompt_ids[: max(self.max_length - 1, 1)]

    def _ensure_non_empty_response(self, response_ids: list[int]) -> list[int]:
        if response_ids:
            return response_ids
        return [self.fallback_token_id]

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
        response_lengths = []

        for feature in features:
            response_text = str(feature.get(field_name) or "")
            response_ids = self._ensure_non_empty_response(self._tokenize_text(response_text))
            response_tensor = torch.tensor(response_ids, dtype=torch.long)
            response_ids_per_example.append(response_tensor)
            response_lengths.append(int(response_tensor.numel()))

        return {
            f"{prefix}_response_ids": self._pad_tensor_batch(response_ids_per_example, self.pad_token_id),
            f"{prefix}_response_lengths": torch.tensor(response_lengths, dtype=torch.long),
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(features)
        t_values = self._sample_t_values(batch_size)
        self._t_sampling_batch_index += 1

        student_prompt_ids_per_example = []
        student_prompt_lengths = []

        for feature in features:
            student_prompt_ids = self._truncate_prompt_ids(self._tokenize_text(feature["student_prompt"]))
            prompt_tensor = torch.tensor(student_prompt_ids, dtype=torch.long)
            student_prompt_ids_per_example.append(prompt_tensor)
            student_prompt_lengths.append(int(prompt_tensor.numel()))

        student_prompt_ids = self._pad_tensor_batch(student_prompt_ids_per_example, self.pad_token_id)
        collated = {
            "student_prompt_ids": student_prompt_ids,
            "student_prompt_lengths": torch.tensor(student_prompt_lengths, dtype=torch.long),
            "t_values": t_values,
        }
        collated.update(self._collate_response_batch(features, "target_response", "target"))
        return collated


class DiffuSelfDistillTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = str(getattr(self.args, "dataset", "")).strip().lower()
        self.method = normalize_training_method(getattr(self.args, "method", "SFT"))
        self.args.method = self.method
        if getattr(self.args, "disable_dropout", False):
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
            "batch_sft_frac": 0.0,
        }

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
        self._interval_log_sums["batch_sft_frac"] += 1.0

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

    def _decode_model_input_text(
        self,
        input_ids: torch.Tensor,
        prompt_len: int,
        response_len: int,
    ) -> str:
        total_len = max(int(prompt_len), 0) + max(int(response_len), 0)
        if total_len <= 0:
            return ""
        return self._decode_token_ids(input_ids[:total_len])

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

    def _maybe_dump_debug_examples(
        self,
        inputs: dict[str, torch.Tensor],
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

            student_prompt_len = int(inputs["student_prompt_lengths"][i].item())
            shared_response_ids = inputs["response_ids"][i, :response_len]
            kd_mask_i = inputs["kd_mask"][i, :response_len]
            student_input_text = self._decode_model_input_text(
                inputs["student_input_ids"][i],
                prompt_len=student_prompt_len,
                response_len=response_len,
            )
            student_masked_response_ids = inputs["student_input_ids"][
                i, student_prompt_len : student_prompt_len + response_len
            ]
            student_response_logits = student_logits[i, student_prompt_len : student_prompt_len + response_len, :]
            student_pred_ids = torch.argmax(student_response_logits, dim=-1)
            student_reconstructed_ids = torch.where(kd_mask_i, student_pred_ids, shared_response_ids)
            kd_positions = torch.nonzero(kd_mask_i, as_tuple=False).flatten()
            masked_target_token_ids = shared_response_ids.index_select(0, kd_positions)
            student_masked_top1_probs = self._masked_top1_probs(student_response_logits, kd_positions)
            student_masked_top1_logits = self._masked_top1_logits(student_response_logits, kd_positions)
            student_top1_token_ids = student_pred_ids.index_select(0, kd_positions).detach().cpu().tolist()
            masked_target_token_ids_list = masked_target_token_ids.detach().cpu().tolist()
            if kd_positions.numel() <= self.debug_save_max_masked_positions:
                selected_debug_indices = torch.arange(kd_positions.numel(), device=kd_positions.device)
            else:
                selected_debug_indices = torch.arange(self.debug_save_max_masked_positions, device=kd_positions.device)
            selected_debug_indices_list = selected_debug_indices.detach().cpu().tolist()
            selected_target_token_ids = [masked_target_token_ids_list[idx] for idx in selected_debug_indices_list]
            selected_student_top1_token_ids = [student_top1_token_ids[idx] for idx in selected_debug_indices_list]
            selected_target_token_pieces = self._token_ids_to_pieces(selected_target_token_ids)
            selected_student_top1_token_pieces = self._token_ids_to_pieces(selected_student_top1_token_ids)

            masked_token_diagnostics = []
            for debug_rank, masked_idx in enumerate(selected_debug_indices_list):
                masked_token_diagnostics.append(
                    {
                        "debug_rank": int(debug_rank),
                        "response_token_index": int(kd_positions[masked_idx].item()),
                        "target_token_id": int(selected_target_token_ids[debug_rank]),
                        "target_token_piece": selected_target_token_pieces[debug_rank],
                        "student_top1_token_id": int(selected_student_top1_token_ids[debug_rank]),
                        "student_top1_token_piece": selected_student_top1_token_pieces[debug_rank],
                        "student_top1_prob": float(student_masked_top1_probs[masked_idx]),
                        "student_top1_logit": float(student_masked_top1_logits[masked_idx]),
                    }
                )

            records.append(
                {
                    "global_step": current_step,
                    "selected_method": self.method,
                    "t_value": float(inputs["effective_t_values"][i].item()),
                    "raw_t_value": float(inputs["t_values"][i].item()),
                    "response_length": response_len,
                    "num_masked_tokens": int(kd_mask_i.sum().item()),
                    "input_text": student_input_text,
                    "student_input_text": student_input_text,
                    "student_prompt_text": self._decode_token_ids(
                        inputs["student_input_ids"][i, :student_prompt_len]
                    ),
                    "target_response_text": self._decode_token_ids(shared_response_ids),
                    "shared_masked_response_text": self._decode_token_ids(student_masked_response_ids),
                    "student_reconstructed_response_text": self._decode_token_ids(student_reconstructed_ids),
                    "student_masked_top1_probs": student_masked_top1_probs,
                    "student_masked_top1_logits": student_masked_top1_logits,
                    "masked_response_positions": kd_positions.detach().cpu().tolist(),
                    "masked_target_token_ids": masked_target_token_ids_list,
                    "student_top1_token_ids": student_top1_token_ids,
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
            response_len = int(inputs["response_lengths"][i].item())
            if response_len <= 0:
                continue

            student_prompt_len = int(inputs["student_prompt_lengths"][i].item())
            kd_mask_i = inputs["kd_mask"][i, :response_len]
            kd_positions = torch.nonzero(kd_mask_i, as_tuple=False).flatten()
            shared_response_ids = inputs["response_ids"][i, :response_len]
            student_response_logits = student_logits[i, student_prompt_len : student_prompt_len + response_len, :]
            student_input_text = self._decode_model_input_text(
                inputs["student_input_ids"][i],
                prompt_len=student_prompt_len,
                response_len=response_len,
            )

            student_top1_token_ids = torch.argmax(student_response_logits, dim=-1)
            student_reconstructed_ids = torch.where(kd_mask_i, student_top1_token_ids, shared_response_ids)
            record = {
                "global_step": current_step,
                "epoch_index": epoch_index,
                "epoch_step_index": epoch_step_index,
                "selected_method": self.method,
                "t_value": float(inputs["effective_t_values"][i].item()),
                "response_length": response_len,
                "num_masked_tokens": int(kd_positions.numel()),
                "input_text": student_input_text,
                "student_input_text": student_input_text,
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
            records.append(record)

        if not records:
            return

        with open(self.top1_logit_debug_path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _compute_masked_cross_entropy_sum(
        self,
        student_response_logits: torch.Tensor,
        target_token_ids: torch.Tensor,
        masked_positions: torch.Tensor,
        temperature: float,
    ):
        if masked_positions.numel() == 0:
            return student_response_logits.new_zeros((), dtype=torch.float32), 0

        tau = float(temperature)
        loss_total = student_response_logits.new_zeros((), dtype=torch.float32)
        for start, end in self._iter_chunk_ranges(int(masked_positions.numel())):
            chunk_positions = masked_positions[start:end]
            student_chunk = student_response_logits.index_select(0, chunk_positions)
            target_chunk = target_token_ids.index_select(0, chunk_positions).to(
                device=student_response_logits.device,
                dtype=torch.long,
            )
            loss_total = loss_total + F.cross_entropy(
                student_chunk / tau,
                target_chunk,
                reduction="sum",
            ).float()

        return loss_total * (tau**2), int(masked_positions.numel())

    def _compute_full_ce_loss_sum(
        self,
        student_response_logits: torch.Tensor,
        target_token_ids: torch.Tensor,
    ):
        response_len = int(target_token_ids.numel())
        if response_len <= 0:
            return student_response_logits.new_zeros((), dtype=torch.float32), 0

        loss_total = student_response_logits.new_zeros((), dtype=torch.float32)
        for start, end in self._iter_chunk_ranges(response_len):
            loss_total = loss_total + F.cross_entropy(
                student_response_logits[start:end],
                target_token_ids[start:end].to(
                    device=student_response_logits.device,
                    dtype=torch.long,
                ),
                reduction="sum",
            ).float()

        return loss_total, response_len

    def _fit_example_to_max_length(
        self,
        student_prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ):
        max_prompt_len = max(self.data_collator.max_length - 1, 1)
        student_prompt_ids = student_prompt_ids[:max_prompt_len]
        response_budget = self.data_collator.max_length - int(student_prompt_ids.numel())
        response_budget = max(response_budget, 1)

        if response_ids.numel() == 0:
            response_ids = response_ids.new_tensor([self.data_collator.fallback_token_id], dtype=torch.long)

        response_ids = response_ids[:response_budget]
        return student_prompt_ids, response_ids

    def _build_kd_mask(
        self,
        response_len: int,
        t_value: torch.Tensor,
        device: torch.device,
    ):
        if response_len <= 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        mask = torch.rand(response_len, device=device) < float(t_value.item())
        if not mask.any():
            sampled_idx = int(torch.randint(0, response_len, (1,), device=device).item())
            mask[sampled_idx] = True
        return mask
    def _prepare_batch_inputs(self, model, inputs: dict[str, torch.Tensor]):
        del model
        batch_size = inputs["student_prompt_ids"].shape[0]
        device = inputs["student_prompt_ids"].device

        student_sequences = []
        response_ids_per_example = []
        ce_target_ids_per_example = []
        kd_masks = []
        ce_masks = []
        response_lengths = []
        student_prompt_lengths = []
        effective_t_values = []

        for i in range(batch_size):
            student_prompt_len = int(inputs["student_prompt_lengths"][i].item())
            student_prompt_ids = inputs["student_prompt_ids"][i, :student_prompt_len]
            target_response_len = int(inputs["target_response_lengths"][i].item())
            response_ids = inputs["target_response_ids"][i, :target_response_len]
            student_prompt_ids, response_ids = self._fit_example_to_max_length(
                student_prompt_ids=student_prompt_ids,
                response_ids=response_ids,
            )

            response_len = int(response_ids.numel())
            raw_t_value = inputs["t_values"][i]
            kd_mask = self._build_kd_mask(
                response_len=response_len,
                t_value=raw_t_value,
                device=device,
            )
            student_masked_response_ids = torch.where(
                kd_mask,
                torch.full_like(response_ids, self.data_collator.mask_id),
                response_ids,
            )
            student_sequences.append(torch.cat([student_prompt_ids, student_masked_response_ids], dim=0))
            response_ids_per_example.append(response_ids)
            kd_masks.append(kd_mask)
            response_lengths.append(response_len)
            student_prompt_lengths.append(int(student_prompt_ids.numel()))
            effective_t_values.append(float(raw_t_value.item()))

            if float(getattr(self.args, "ce_weight", 0.0)) > 0.0:
                ce_target_ids_per_example.append(response_ids.clone())
                ce_masks.append(torch.ones(response_len, dtype=torch.bool, device=device))
            else:
                ce_target_ids_per_example.append(torch.full((response_len,), -100, dtype=torch.long, device=device))
                ce_masks.append(torch.zeros(response_len, dtype=torch.bool, device=device))

        student_max_len = max(seq.numel() for seq in student_sequences)
        response_max_len = max(resp.numel() for resp in response_ids_per_example)

        student_input_ids = torch.full(
            (batch_size, student_max_len),
            self.data_collator.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        student_attention_mask = torch.zeros((batch_size, student_max_len), dtype=torch.long, device=device)
        response_ids = torch.full((batch_size, response_max_len), -100, dtype=torch.long, device=device)
        ce_target_ids = torch.full((batch_size, response_max_len), -100, dtype=torch.long, device=device)
        kd_mask = torch.zeros((batch_size, response_max_len), dtype=torch.bool, device=device)
        ce_mask = torch.zeros((batch_size, response_max_len), dtype=torch.bool, device=device)
        response_mask = torch.zeros((batch_size, response_max_len), dtype=torch.bool, device=device)

        for i in range(batch_size):
            student_input_ids[i, : student_sequences[i].numel()] = student_sequences[i]
            student_attention_mask[i, : student_sequences[i].numel()] = 1
            response_ids[i, : response_ids_per_example[i].numel()] = response_ids_per_example[i]
            ce_target_ids[i, : ce_target_ids_per_example[i].numel()] = ce_target_ids_per_example[i]
            kd_mask[i, : kd_masks[i].numel()] = kd_masks[i]
            ce_mask[i, : ce_masks[i].numel()] = ce_masks[i]
            response_mask[i, : response_ids_per_example[i].numel()] = True

        return {
            "student_input_ids": student_input_ids,
            "student_attention_mask": student_attention_mask,
            "response_ids": response_ids,
            "ce_target_ids": ce_target_ids,
            "response_mask": response_mask,
            "kd_mask": kd_mask,
            "ce_mask": ce_mask,
            "student_prompt_lengths": torch.tensor(student_prompt_lengths, dtype=torch.long, device=device),
            "response_lengths": torch.tensor(response_lengths, dtype=torch.long, device=device),
            "selected_methods": [self.method] * batch_size,
            "t_values": inputs["t_values"],
            "effective_t_values": torch.tensor(effective_t_values, dtype=torch.float32, device=device),
        }

    def _collect_batch_metrics(self, inputs: dict[str, torch.Tensor]) -> BatchMetrics:
        response_lengths = inputs["response_lengths"].float()
        response_lengths_clamped = response_lengths.clamp_min(1.0)
        mask_ratio = inputs["kd_mask"].float().sum(dim=1) / response_lengths_clamped
        return BatchMetrics(
            mean_t=inputs["t_values"].float().mean().item(),
            avg_mask_ratio=mask_ratio.mean().item(),
            avg_student_mask_ratio=mask_ratio.mean().item(),
            avg_response_length=response_lengths.mean().item(),
            num_masked_tokens=inputs["kd_mask"].float().sum().item(),
            num_student_masked_tokens=inputs["kd_mask"].float().sum().item(),
            method_fractions={"SFT": 1.0},
        )

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        del num_items_in_batch
        inputs = self._prepare_batch_inputs(model=model, inputs=inputs)

        student_outputs = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )
        student_logits = student_outputs.logits
        del student_outputs

        if model.training:
            self._maybe_dump_debug_examples(
                inputs=inputs,
                student_logits=student_logits,
            )
            self._maybe_dump_top1_logit_debug(
                inputs=inputs,
                student_logits=student_logits,
            )

        kd_loss_sum = student_logits.new_zeros((), dtype=torch.float32)
        kd_token_count = 0
        ce_loss_sum = student_logits.new_zeros((), dtype=torch.float32)
        ce_token_count = 0

        batch_size = inputs["student_input_ids"].shape[0]
        ce_target_ids = inputs["ce_target_ids"]
        kd_mask = inputs["kd_mask"]
        for i in range(batch_size):
            response_len = int(inputs["response_lengths"][i].item())
            if response_len <= 0:
                continue

            student_prompt_len = int(inputs["student_prompt_lengths"][i].item())
            student_response_logits = student_logits[i, student_prompt_len : student_prompt_len + response_len, :]
            kd_mask_i = kd_mask[i, :response_len]
            response_target_ids = inputs["response_ids"][i, :response_len]

            if kd_mask_i.any():
                kd_positions = torch.nonzero(kd_mask_i, as_tuple=False).flatten()
                kd_chunk_sum, kd_chunk_count = self._compute_masked_cross_entropy_sum(
                    student_response_logits=student_response_logits,
                    target_token_ids=response_target_ids,
                    masked_positions=kd_positions,
                    temperature=float(getattr(self.args, "distill_temperature", 1.0)),
                )
                kd_loss_sum = kd_loss_sum + kd_chunk_sum
                kd_token_count += kd_chunk_count

            if float(getattr(self.args, "ce_weight", 0.0)) > 0.0:
                ce_chunk_sum, ce_chunk_count = self._compute_full_ce_loss_sum(
                    student_response_logits=student_response_logits,
                    target_token_ids=ce_target_ids[i, :response_len],
                )
                ce_loss_sum = ce_loss_sum + ce_chunk_sum
                ce_token_count += ce_chunk_count

        if kd_token_count > 0:
            kd_loss = kd_loss_sum / kd_token_count
        else:
            kd_loss = student_logits.sum() * 0.0

        if float(getattr(self.args, "ce_weight", 0.0)) > 0.0 and ce_token_count > 0:
            ce_loss = ce_loss_sum / ce_token_count
        else:
            ce_loss = student_logits.sum() * 0.0

        loss = float(getattr(self.args, "kd_weight", 1.0)) * kd_loss + float(getattr(self.args, "ce_weight", 0.0)) * ce_loss

        if model.training:
            self._accumulate_interval_log_metrics(
                inputs=inputs,
                kd_loss=kd_loss,
                ce_loss=ce_loss,
            )

        if return_outputs:
            return loss, {"student_logits": student_logits}
        return loss
