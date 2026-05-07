from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


METHOD_ALIASES = {
    "INP-OH": "INP_OH",
    "INPOH": "INP_OH",
    "INP_OH": "INP_OH",
}
VALID_METHODS = {"INP_OH"}
T_SAMPLING_MODE_ALIASES = {
    "biased-to-one": "biased_to_one",
    "biasedtoone": "biased_to_one",
    "high-bias": "biased_to_one",
    "highbias": "biased_to_one",
    "two-point": "two_point",
    "twopoint": "two_point",
}
VALID_T_SAMPLING_MODES = {"uniform", "fixed", "two_point", "curriculum", "biased_to_one"}
BLOCK_CONDITIONING_STRATEGIES = {"kv_cache", "blockwise_attention"}


def _normalize_method_name(method: str) -> str:
    normalized = str(method or "").strip().upper()
    normalized = METHOD_ALIASES.get(normalized, normalized)
    return normalized


def normalize_training_method(method: str) -> str:
    normalized = _normalize_method_name(method) or "INP_OH"
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


def infer_block_conditioning_strategy(model) -> str:
    model_config = getattr(model, "config", None)
    model_type = str(getattr(model_config, "model_type", "") or "").strip().lower()
    architectures = [
        str(architecture).strip().lower()
        for architecture in (getattr(model_config, "architectures", None) or [])
    ]

    if model_type == "llada" or any("llada" in architecture for architecture in architectures):
        return "blockwise_attention"
    return "kv_cache"


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
    avg_response_length: float
    avg_blocks_per_example: float
    num_masked_tokens: float


class BlockDiffusionDataCollator:
    """
    Tokenize prompt/response pairs and sample per-block timesteps.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int,
        mask_id: int,
        block_size: int,
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
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_id = mask_id
        self.block_size = int(block_size)
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
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must provide pad_token_id or eos_token_id.")
        self.fallback_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else self.pad_token_id

    def _validate_t_sampling_settings(self):
        if self.block_size <= 0:
            raise ValueError("`block_size` must be >= 1.")
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

    def _sample_t_values(self, sample_count: int) -> torch.Tensor:
        if sample_count <= 0:
            return torch.empty(0, dtype=torch.float32)

        if self.t_sampling_mode == "fixed":
            return torch.full((sample_count,), self.t_fixed, dtype=torch.float32)

        if self.t_sampling_mode == "biased_to_one":
            return sample_t_biased_to_one(
                sample_count,
                epsilon=self.t_min,
                strength=self.t_biased_to_one_strength,
                t_max=self.t_max,
            )

        if self.t_sampling_mode == "two_point":
            low_values = torch.full((sample_count,), self.t_two_point_low, dtype=torch.float32)
            high_values = torch.full((sample_count,), self.t_two_point_high, dtype=torch.float32)
            choose_high = torch.rand(sample_count) < self.t_two_point_high_prob
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
            return torch.full((sample_count,), float(sample_min), dtype=torch.float32)
        return torch.empty(sample_count, dtype=torch.float32).uniform_(float(sample_min), float(sample_max))

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

    def _count_blocks_for_response_length(self, response_len: int) -> int:
        if response_len <= 0:
            return 0
        return (int(response_len) + self.block_size - 1) // self.block_size

    @staticmethod
    def _pad_tensor_batch(tensors: list[torch.Tensor], pad_value: int):
        max_len = max(tensor.numel() for tensor in tensors)
        batch = torch.full(
            (len(tensors), max_len),
            pad_value,
            dtype=tensors[0].dtype,
        )
        for idx, tensor in enumerate(tensors):
            batch[idx, : tensor.numel()] = tensor
        return batch

    def __call__(self, features: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        batch_size = len(features)

        prompt_ids_per_example = []
        response_ids_per_example = []
        prompt_lengths = []
        response_lengths = []
        block_counts = []

        for feature in features:
            student_prompt = str(feature.get("student_prompt") or "").strip()
            if not student_prompt:
                raise ValueError("Every feature must contain a non-empty `student_prompt` field.")
            response_text = feature.get("target_response") or feature.get("response") or feature.get("gold_response") or ""

            prompt_ids = self._truncate_prompt_ids(self._tokenize_text(student_prompt))
            response_ids = self._ensure_non_empty_response(self._tokenize_text(str(response_text)))

            prompt_ids_tensor = torch.tensor(prompt_ids, dtype=torch.long)
            response_ids_tensor = torch.tensor(response_ids, dtype=torch.long)

            prompt_ids_per_example.append(prompt_ids_tensor)
            response_ids_per_example.append(response_ids_tensor)
            prompt_lengths.append(int(prompt_ids_tensor.numel()))
            response_lengths.append(int(response_ids_tensor.numel()))
            effective_response_len = min(
                int(response_ids_tensor.numel()),
                max(self.max_length - int(prompt_ids_tensor.numel()), 1),
            )
            block_counts.append(self._count_blocks_for_response_length(effective_response_len))

        max_block_count = max(block_counts, default=0)
        block_t_values = torch.zeros((batch_size, max_block_count), dtype=torch.float32)
        sampled_t_values = self._sample_t_values(sum(block_counts))
        sample_offset = 0
        for batch_index, block_count in enumerate(block_counts):
            if block_count <= 0:
                continue
            next_offset = sample_offset + block_count
            block_t_values[batch_index, :block_count] = sampled_t_values[sample_offset:next_offset]
            sample_offset = next_offset
        self._t_sampling_batch_index += 1

        return {
            "prompt_ids": self._pad_tensor_batch(prompt_ids_per_example, self.pad_token_id),
            "response_ids": self._pad_tensor_batch(response_ids_per_example, self.pad_token_id),
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "response_lengths": torch.tensor(response_lengths, dtype=torch.long),
            "block_counts": torch.tensor(block_counts, dtype=torch.long),
            "block_t_values": block_t_values,
        }


class BlockDiffusionTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = normalize_training_method(getattr(self.args, "method", "INP-OH"))
        self.block_conditioning_strategy = infer_block_conditioning_strategy(self.model)
        if self.block_conditioning_strategy not in BLOCK_CONDITIONING_STRATEGIES:
            raise ValueError(
                f"Unsupported block conditioning strategy `{self.block_conditioning_strategy}`."
            )
        if (
            getattr(self.args, "gradient_checkpointing", False)
            and self.block_conditioning_strategy == "kv_cache"
        ):
            raise ValueError(
                "block-train currently requires `gradient_checkpointing=False` when using KV-cache block conditioning."
            )
        if int(getattr(self.args, "block_size", 0)) <= 0:
            raise ValueError("`block_size` must be >= 1.")
        if getattr(self.args, "disable_dropout", False):
            disable_dropout_in_model(self.model)
        print(
            f"BlockDiffusionTrainer using `{self.block_conditioning_strategy}` block conditioning "
            f"for model_type={getattr(getattr(self.model, 'config', None), 'model_type', 'unknown')}."
        )
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
        if "prompt_ids" not in inputs:
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
            "masked_ce_loss": 0.0,
            "mean_t": 0.0,
            "avg_mask_ratio": 0.0,
            "avg_response_length": 0.0,
            "avg_blocks_per_example": 0.0,
            "masked_tokens": 0.0,
        }

    def _accumulate_interval_log_metrics(
        self,
        loss: torch.Tensor,
        metrics: BatchMetrics,
    ):
        self._interval_log_count += 1
        self._interval_log_sums["masked_ce_loss"] += float(loss.detach().item())
        self._interval_log_sums["mean_t"] += float(metrics.mean_t)
        self._interval_log_sums["avg_mask_ratio"] += float(metrics.avg_mask_ratio)
        self._interval_log_sums["avg_response_length"] += float(metrics.avg_response_length)
        self._interval_log_sums["avg_blocks_per_example"] += float(metrics.avg_blocks_per_example)
        self._interval_log_sums["masked_tokens"] += float(metrics.num_masked_tokens)

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

    def _fit_example_to_max_length(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ):
        max_prompt_len = max(self.data_collator.max_length - 1, 1)
        prompt_ids = prompt_ids[:max_prompt_len]
        response_budget = self.data_collator.max_length - prompt_ids.numel()
        response_budget = max(response_budget, 1)

        if response_ids.numel() == 0:
            response_ids = response_ids.new_tensor([self.data_collator.fallback_token_id], dtype=torch.long)

        response_ids = response_ids[:response_budget]
        return prompt_ids, response_ids

    def _split_response_into_blocks(self, response_len: int):
        if response_len <= 0:
            return []

        block_size = max(int(getattr(self.args, "block_size", 1)), 1)
        block_ranges = []
        for start in range(0, response_len, block_size):
            end = min(start + block_size, response_len)
            block_ranges.append((start, end))
        return block_ranges

    def _apply_forward_noise(self, block_ids: torch.Tensor, t_value: float):
        block_len = int(block_ids.numel())
        if block_len <= 0:
            empty_mask = torch.zeros(0, dtype=torch.bool, device=block_ids.device)
            return block_ids, empty_mask

        noisy_mask = torch.rand(block_len, device=block_ids.device) < float(t_value)
        if not noisy_mask.any():
            sampled_idx = torch.randint(0, block_len, (1,), device=block_ids.device).item()
            noisy_mask[sampled_idx] = True

        noisy_ids = torch.where(
            noisy_mask,
            torch.full_like(block_ids, self.data_collator.mask_id),
            block_ids,
        )
        return noisy_ids, noisy_mask

    def _build_attention_mask(self, prefix_length: int, current_length: int, device: torch.device):
        total_length = max(int(prefix_length), 0) + max(int(current_length), 0)
        return torch.ones((1, total_length), dtype=torch.long, device=device)

    def _build_block_attention_bias(
        self,
        prompt_length: int,
        block_ranges: list[tuple[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        response_length = block_ranges[-1][1] if block_ranges else 0
        total_length = max(int(prompt_length), 0) + max(int(response_length), 0)
        segment_ids = torch.zeros(total_length, dtype=torch.long, device=device)
        for block_index, (start, end) in enumerate(block_ranges, start=1):
            segment_ids[prompt_length + start : prompt_length + end] = block_index

        # Earlier blocks stay independent from later noisy blocks, while each block
        # can still attend to its clean prefix and its own positions.
        return segment_ids.view(-1, 1) >= segment_ids.view(1, -1)

    def _build_prompt_cache(self, model, prompt_ids: torch.Tensor):
        prompt_len = int(prompt_ids.numel())
        if prompt_len <= 0:
            return None

        with torch.no_grad():
            prompt_outputs = model(
                input_ids=prompt_ids.unsqueeze(0),
                attention_mask=self._build_attention_mask(0, prompt_len, prompt_ids.device),
                use_cache=True,
            )
        past_key_values = getattr(prompt_outputs, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError(
                "Model did not return `past_key_values` with `use_cache=True`. "
                "The block diffusion trainer requires cache support."
            )
        return past_key_values

    def _forward_noisy_block_logits(
        self,
        model,
        noisy_block_ids: torch.Tensor,
        prefix_past_key_values,
        prefix_length: int,
    ):
        block_len = int(noisy_block_ids.numel())
        model_kwargs = {
            "input_ids": noisy_block_ids.unsqueeze(0),
            "attention_mask": self._build_attention_mask(prefix_length, block_len, noisy_block_ids.device),
        }
        if prefix_past_key_values is not None:
            model_kwargs["past_key_values"] = prefix_past_key_values
        noisy_outputs = model(**model_kwargs)
        logits = getattr(noisy_outputs, "logits", None)
        if logits is None:
            raise RuntimeError("Model forward did not return `logits`.")
        return logits[0]

    def _extend_clean_prefix_cache(
        self,
        model,
        prefix_past_key_values,
        prefix_length: int,
        clean_block_ids: torch.Tensor,
    ):
        block_len = int(clean_block_ids.numel())
        model_kwargs = {
            "input_ids": clean_block_ids.unsqueeze(0),
            "attention_mask": self._build_attention_mask(prefix_length, block_len, clean_block_ids.device),
            "use_cache": True,
        }
        if prefix_past_key_values is not None:
            model_kwargs["past_key_values"] = prefix_past_key_values

        with torch.no_grad():
            clean_outputs = model(**model_kwargs)
        next_past_key_values = getattr(clean_outputs, "past_key_values", None)
        if next_past_key_values is None:
            raise RuntimeError(
                "Model did not return `past_key_values` while extending the clean prefix cache."
            )
        return next_past_key_values

    def _compute_block_loss_sum(
        self,
        block_logits: torch.Tensor,
        clean_block_ids: torch.Tensor,
        noisy_mask: torch.Tensor,
    ):
        masked_positions = torch.nonzero(noisy_mask, as_tuple=False).flatten()
        if masked_positions.numel() == 0:
            return block_logits.new_zeros((), dtype=torch.float32), 0

        ce_total = block_logits.new_zeros((), dtype=torch.float32)
        for start, end in self._iter_chunk_ranges(int(masked_positions.numel())):
            chunk_positions = masked_positions[start:end]
            chunk_logits = block_logits.index_select(0, chunk_positions)
            chunk_targets = clean_block_ids.index_select(0, chunk_positions).to(
                device=block_logits.device,
                dtype=torch.long,
            )
            ce_total = ce_total + F.cross_entropy(
                chunk_logits,
                chunk_targets,
                reduction="sum",
            ).float()

        return ce_total, int(masked_positions.numel())

    def _compute_loss_with_blockwise_attention(
        self,
        model,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        block_ranges: list[tuple[int, int]],
        block_t_values: torch.Tensor,
    ):
        noisy_block_ids_per_example = []
        noisy_masks_per_example = []
        example_t_values: list[float] = []

        for block_index, (start, end) in enumerate(block_ranges):
            clean_block_ids = response_ids[start:end]
            if clean_block_ids.numel() <= 0:
                continue

            t_value = float(block_t_values[block_index].item())
            noisy_block_ids, noisy_mask = self._apply_forward_noise(clean_block_ids, t_value)
            noisy_block_ids_per_example.append(noisy_block_ids)
            noisy_masks_per_example.append(noisy_mask)
            example_t_values.append(t_value)

        if not noisy_block_ids_per_example:
            zero = response_ids.sum() * 0.0
            return zero, 0, example_t_values

        noisy_response_ids = torch.cat(noisy_block_ids_per_example, dim=0)
        noisy_response_mask = torch.cat(noisy_masks_per_example, dim=0)
        model_inputs = torch.cat((prompt_ids, noisy_response_ids), dim=0)
        model_outputs = model(
            input_ids=model_inputs.unsqueeze(0),
            attention_mask=self._build_attention_mask(0, model_inputs.numel(), model_inputs.device),
            attention_bias=self._build_block_attention_bias(
                prompt_length=int(prompt_ids.numel()),
                block_ranges=block_ranges,
                device=model_inputs.device,
            ),
        )
        logits = getattr(model_outputs, "logits", None)
        if logits is None:
            raise RuntimeError("Model forward did not return `logits`.")
        response_logits = logits[0, prompt_ids.numel() : prompt_ids.numel() + response_ids.numel()]
        loss_sum, masked_token_count = self._compute_block_loss_sum(
            block_logits=response_logits,
            clean_block_ids=response_ids,
            noisy_mask=noisy_response_mask,
        )
        return loss_sum, masked_token_count, example_t_values

    def _build_batch_metrics(
        self,
        mean_t_values: list[float],
        mask_ratios: list[float],
        response_lengths: list[int],
        block_counts: list[int],
        num_masked_tokens: int,
    ) -> BatchMetrics:
        if not response_lengths:
            return BatchMetrics(
                mean_t=0.0,
                avg_mask_ratio=0.0,
                avg_response_length=0.0,
                avg_blocks_per_example=0.0,
                num_masked_tokens=0.0,
            )

        return BatchMetrics(
            mean_t=float(sum(mean_t_values) / max(len(mean_t_values), 1)),
            avg_mask_ratio=float(sum(mask_ratios) / len(mask_ratios)),
            avg_response_length=float(sum(response_lengths) / len(response_lengths)),
            avg_blocks_per_example=float(sum(block_counts) / len(block_counts)),
            num_masked_tokens=float(num_masked_tokens),
        )

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        batch_size = inputs["prompt_ids"].shape[0]
        total_loss_sum = None
        total_masked_token_count = 0

        mean_t_values: list[float] = []
        mask_ratios: list[float] = []
        response_lengths: list[int] = []
        block_counts: list[int] = []

        for batch_index in range(batch_size):
            prompt_len = int(inputs["prompt_lengths"][batch_index].item())
            response_len = int(inputs["response_lengths"][batch_index].item())

            prompt_ids = inputs["prompt_ids"][batch_index, :prompt_len]
            response_ids = inputs["response_ids"][batch_index, :response_len]
            prompt_ids, response_ids = self._fit_example_to_max_length(prompt_ids, response_ids)
            response_len = int(response_ids.numel())
            if response_len <= 0:
                continue

            block_ranges = self._split_response_into_blocks(response_len)
            expected_block_count = int(inputs["block_counts"][batch_index].item())
            if expected_block_count != len(block_ranges):
                raise RuntimeError(
                    f"Mismatch between collator block count ({expected_block_count}) "
                    f"and trainer block count ({len(block_ranges)}) for response_len={response_len}."
                )
            if not block_ranges:
                continue

            example_masked_token_count = 0
            example_t_values: list[float] = []
            if self.block_conditioning_strategy == "blockwise_attention":
                block_loss_sum, block_masked_token_count, example_t_values = self._compute_loss_with_blockwise_attention(
                    model=model,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    block_ranges=block_ranges,
                    block_t_values=inputs["block_t_values"][batch_index],
                )
                if total_loss_sum is None:
                    total_loss_sum = block_loss_sum
                else:
                    total_loss_sum = total_loss_sum + block_loss_sum
                total_masked_token_count += block_masked_token_count
                example_masked_token_count += block_masked_token_count
            else:
                prefix_past_key_values = self._build_prompt_cache(model, prompt_ids)
                prefix_length = int(prompt_ids.numel())

                for block_index, (start, end) in enumerate(block_ranges):
                    clean_block_ids = response_ids[start:end]
                    if clean_block_ids.numel() <= 0:
                        continue

                    t_value = float(inputs["block_t_values"][batch_index, block_index].item())
                    noisy_block_ids, noisy_mask = self._apply_forward_noise(clean_block_ids, t_value)
                    block_logits = self._forward_noisy_block_logits(
                        model=model,
                        noisy_block_ids=noisy_block_ids,
                        prefix_past_key_values=prefix_past_key_values,
                        prefix_length=prefix_length,
                    )
                    block_loss_sum, block_masked_token_count = self._compute_block_loss_sum(
                        block_logits=block_logits,
                        clean_block_ids=clean_block_ids,
                        noisy_mask=noisy_mask,
                    )
                    if total_loss_sum is None:
                        total_loss_sum = block_loss_sum
                    else:
                        total_loss_sum = total_loss_sum + block_loss_sum
                    total_masked_token_count += block_masked_token_count
                    example_masked_token_count += block_masked_token_count
                    example_t_values.append(t_value)

                    prefix_past_key_values = self._extend_clean_prefix_cache(
                        model=model,
                        prefix_past_key_values=prefix_past_key_values,
                        prefix_length=prefix_length,
                        clean_block_ids=clean_block_ids,
                    )
                    prefix_length += int(clean_block_ids.numel())

            if example_t_values:
                mean_t_values.extend(example_t_values)
                mask_ratios.append(float(example_masked_token_count) / max(response_len, 1))
                response_lengths.append(response_len)
                block_counts.append(len(example_t_values))

        if total_loss_sum is None or total_masked_token_count <= 0:
            loss = inputs["prompt_ids"].sum() * 0.0
        else:
            loss = total_loss_sum / total_masked_token_count

        if model.training:
            self._accumulate_interval_log_metrics(
                loss=loss,
                metrics=self._build_batch_metrics(
                    mean_t_values=mean_t_values,
                    mask_ratios=mask_ratios,
                    response_lengths=response_lengths,
                    block_counts=block_counts,
                    num_masked_tokens=total_masked_token_count,
                ),
            )

        if return_outputs:
            return loss, {"num_masked_tokens": total_masked_token_count}
        return loss
