import argparse
import functools
import importlib
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import transformers.dynamic_module_utils as dynamic_module_utils
import transformers.modeling_rope_utils as rope_utils
import transformers.utils as transformers_utils
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM

from generate import generate
from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from parser_helper import is_equiv, last_boxed_only_string, first_boxed_only_string, remove_boxed


def ensure_dynamic_rope_compat():
    """Backfill newer transformers RoPE decorator on older installs."""
    if hasattr(rope_utils, "dynamic_rope_update"):
        pass
    else:
        def dynamic_rope_update(func):
            @functools.wraps(func)
            def wrapper(self, x, position_ids, *args, **kwargs):
                rope_type = getattr(self, "rope_type", "")
                if (
                    isinstance(rope_type, str)
                    and "dynamic" in rope_type
                    and hasattr(self, "_dynamic_frequency_update")
                ):
                    self._dynamic_frequency_update(position_ids, device=x.device)
                return func(self, x, position_ids, *args, **kwargs)

            return wrapper

        rope_utils.dynamic_rope_update = dynamic_rope_update

    if not hasattr(transformers_utils, "TransformersKwargs"):
        class TransformersKwargs(TypedDict, total=False):
            pass

        transformers_utils.TransformersKwargs = TransformersKwargs

    if not getattr(dynamic_module_utils.get_cached_module_file, "_llada_eval_repo_id_compat", False):
        original_get_cached_module_file = dynamic_module_utils.get_cached_module_file

        def sanitize_module_path(module_path):
            parts = Path(module_path).parts
            sanitized_parts = []
            for index, part in enumerate(parts):
                stem, suffix = os.path.splitext(part) if index == len(parts) - 1 else (part, "")
                sanitized = re.sub(r"[^0-9A-Za-z_]", "_", stem)
                if sanitized and sanitized[0].isdigit():
                    sanitized = f"_{sanitized}"
                sanitized_parts.append((sanitized or "_") + suffix)
            return os.path.join(*sanitized_parts)

        @functools.wraps(original_get_cached_module_file)
        def get_cached_module_file_compat(*args, **kwargs):
            module_path = original_get_cached_module_file(*args, **kwargs)
            sanitized_path = sanitize_module_path(module_path)
            if sanitized_path == module_path:
                return module_path

            modules_cache = Path(dynamic_module_utils.HF_MODULES_CACHE)
            source_file = modules_cache / module_path
            target_file = modules_cache / sanitized_path
            dynamic_module_utils.create_dynamic_module(
                os.path.relpath(target_file.parent, modules_cache)
            )
            for source in source_file.parent.glob("*.py"):
                shutil.copy2(source, target_file.parent / source.name)
            importlib.invalidate_caches()
            return sanitized_path

        get_cached_module_file_compat._llada_eval_repo_id_compat = True
        dynamic_module_utils.get_cached_module_file = get_cached_module_file_compat


ensure_dynamic_rope_compat()


def get_tokenizer_required_size(tokenizer) -> int:
    vocab = tokenizer.get_vocab()
    if vocab:
        return max(vocab.values()) + 1
    return len(tokenizer)


def load_eval_model_and_tokenizer(model_path, device, torch_dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer_required_size = get_tokenizer_required_size(tokenizer)

    model_cls = AutoModelForCausalLM if getattr(config, "model_type", "") == "llada2_moe" else AutoModel
    model = model_cls.from_pretrained(
        model_path,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch_dtype,
    ).to(device)

    if (
        hasattr(model, "get_input_embeddings")
        and model.get_input_embeddings() is not None
        and model.get_input_embeddings().num_embeddings != tokenizer_required_size
        and hasattr(model, "resize_token_embeddings")
    ):
        model.resize_token_embeddings(tokenizer_required_size)

    return tokenizer, model


DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
}


def resolve_mask_id(tokenizer, configured_mask_id: int = -1) -> int:
    if configured_mask_id is not None and int(configured_mask_id) >= 0:
        return int(configured_mask_id)

    tokenizer_mask_id = getattr(tokenizer, "mask_token_id", None)
    if tokenizer_mask_id is not None:
        return int(tokenizer_mask_id)

    for special_token in ("<|mdm_mask|>",):
        token_id = tokenizer.convert_tokens_to_ids(special_token)
        if token_id is not None and token_id >= 0 and token_id != getattr(tokenizer, "unk_token_id", None):
            return int(token_id)

    return 126336


def resolve_stop_token_ids(tokenizer) -> set[int]:
    stop_token_ids: set[int] = set()
    for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        if token_id is not None:
            stop_token_ids.add(int(token_id))

    for special_token in ("<|eot_id|>",):
        token_id = tokenizer.convert_tokens_to_ids(special_token)
        if token_id is not None and token_id >= 0 and token_id != getattr(tokenizer, "unk_token_id", None):
            stop_token_ids.add(int(token_id))

    return stop_token_ids


def decode_generated_texts(
    tokenizer,
    generated_token_ids: torch.Tensor,
    mask_id: int,
    truncate_at_mask: bool = False,
) -> list[str]:
    stop_token_ids = resolve_stop_token_ids(tokenizer)
    decoded_texts = []
    for token_ids in generated_token_ids.tolist():
        cutoff = len(token_ids)
        for idx, token_id in enumerate(token_ids):
            if (truncate_at_mask and token_id == mask_id) or token_id in stop_token_ids:
                cutoff = idx
                break
        decoded_texts.append(
            tokenizer.decode(
                token_ids[:cutoff],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    return decoded_texts


def _extract_number(text, use_last_match=False):
    try:
        return float(text)
    except (TypeError, ValueError):
        numbers = re.findall(r"-?\d+\.?\d*", text or "")
        if numbers:
            index = -1 if use_last_match else 0
            try:
                return float(numbers[index])
            except ValueError:
                return None
    return None


def _truncate_generation_at_stop(
    raw_generation,
    truncate_repeated_newlines: bool = False,
    newline_run_length: int = 10,
):
    text = raw_generation or ""
    eot_marker = "<|eot_id|>"
    if eot_marker in text:
        text = text.split(eot_marker, 1)[0]
    if truncate_repeated_newlines and newline_run_length > 0:
        repeated_newlines = re.compile(r"(?:\n){" + str(newline_run_length) + r",}")
        match = repeated_newlines.search(text)
        if match:
            text = text[:match.start()]
    return text


def extract_gsm8k_answer(
    raw_generation,
    prompt_style: str = "default",
    truncate_repeated_newlines: bool = False,
):
    truncated_generation = _truncate_generation_at_stop(
        raw_generation,
        truncate_repeated_newlines=truncate_repeated_newlines,
    )

    if "\\boxed" in truncated_generation or "\\fbox" in truncated_generation:
        boxed_string = (
            first_boxed_only_string(truncated_generation)
            if prompt_style == "answer_first"
            else last_boxed_only_string(truncated_generation)
        )
        boxed_content = str(remove_boxed(boxed_string or "")).strip()
        if boxed_content and boxed_content != "..." and not re.match(r"^\.+$", boxed_content):
            parsed_answer = _extract_number(boxed_content)
            if parsed_answer is not None:
                return parsed_answer

    answer_match = re.search(r"<answer>(.*?)</answer>", truncated_generation, re.DOTALL)
    if answer_match:
        return _extract_number(answer_match.group(1).strip(), use_last_match=True)

    return None


def compute_gsm8k_accuracy(
    generations,
    prompt_style: str = "default",
    truncate_repeated_newlines: bool = False,
):
    correct = 0
    processed = len(generations)

    for item in generations:
        parsed_answer = extract_gsm8k_answer(
            item.get("generations", ""),
            prompt_style=prompt_style,
            truncate_repeated_newlines=truncate_repeated_newlines,
        )
        is_correct = parsed_answer is not None and parsed_answer == item.get("ground_truth")
        item["parsed_answer"] = parsed_answer
        item["correct"] = is_correct
        if is_correct:
            correct += 1

    accuracy = correct / processed * 100 if processed > 0 else 0.0
    return {"correct": correct, "processed": processed, "accuracy": accuracy}


def extract_math_answer(
    raw_generation,
    prompt_style: str = "default",
    truncate_repeated_newlines: bool = False,
):
    truncated_generation = _truncate_generation_at_stop(
        raw_generation,
        truncate_repeated_newlines=truncate_repeated_newlines,
    )
    parsed_answer = None

    try:
        boxed_string = (
            first_boxed_only_string(truncated_generation)
            if prompt_style == "answer_first"
            else last_boxed_only_string(truncated_generation)
        )
        parsed_answer = remove_boxed(boxed_string)
    except Exception:
        parsed_answer = None

    if not parsed_answer:
        answer_match = re.search(r"<answer>(.*?)</answer>", truncated_generation, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            try:
                boxed_string = (
                    first_boxed_only_string(answer_text)
                    if prompt_style == "answer_first"
                    else last_boxed_only_string(answer_text)
                )
                parsed_answer = remove_boxed(boxed_string)
            except Exception:
                parsed_answer = "<unparsed>"

    return parsed_answer


def compute_math_accuracy(
    generations,
    prompt_style: str = "default",
    truncate_repeated_newlines: bool = False,
):
    correct = 0
    processed = len(generations)

    for item in generations:
        parsed_answer = extract_math_answer(
            item.get("generations", ""),
            prompt_style=prompt_style,
            truncate_repeated_newlines=truncate_repeated_newlines,
        )
        is_correct = parsed_answer is not None and is_equiv(parsed_answer, item.get("ground_truth", ""))
        item["parsed_answer"] = parsed_answer
        item["correct"] = is_correct
        if is_correct:
            correct += 1

    accuracy = correct / processed * 100 if processed > 0 else 0.0
    return {"correct": correct, "processed": processed, "accuracy": accuracy}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def setup_device():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group("nccl")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)

    return torch.device("cpu")


def cleanup_ddp():
    if is_distributed():
        dist.destroy_process_group()


def annotate_generation_results(
    dataset_name,
    generations,
    prompt_style: str = "default",
    truncate_repeated_newlines: bool = False,
):
    if dataset_name == "gsm8k":
        compute_gsm8k_accuracy(
            generations,
            prompt_style=prompt_style,
            truncate_repeated_newlines=truncate_repeated_newlines,
        )
    elif dataset_name == "math":
        compute_math_accuracy(
            generations,
            prompt_style=prompt_style,
            truncate_repeated_newlines=truncate_repeated_newlines,
        )
    else:
        raise ValueError(f"Unsupported dataset `{dataset_name}` in standalone eval package.")


def get_logged_generated_answer(example_result: dict) -> str:
    parsed_value = example_result.get("parsed_answer", None)
    if parsed_value is None:
        return "<unparsed>"

    parsed_text = str(parsed_value)
    if not parsed_text.strip():
        return "<unparsed>"

    return parsed_text


def format_generation_for_log(generation: str, max_blank_lines: int = 6) -> str:
    if generation is None:
        return "<empty>"

    text = str(generation)
    if not text:
        return "''"

    # Drop whitespace-only tail so logs do not get flooded by trailing blank lines.
    text = text.rstrip()
    if not text:
        return "''"

    # Keep the body readable, but summarize pathological blank-line runs.
    pattern = re.compile(r"(?:\n\s*){" + str(max_blank_lines + 1) + r",}")

    def _summarize_blank_lines(match: re.Match) -> str:
        newline_count = match.group(0).count("\n")
        return "\n\n[... {} blank lines omitted ...]\n\n".format(newline_count - 1)

    return pattern.sub(_summarize_blank_lines, text)


def evaluate(
    model,
    tokenizer,
    dataloader,
    dataset_name,
    gen_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    steps=64,
    block_length=32,
    mask_id=126336,
    max_context_length=None,
    prompt_style="default",
    newline_later=False,
    earlystop=False,
):
    model.eval()
    device = next(model.parameters()).device
    total_processed = torch.tensor(0, device=device)
    wall_times = []
    all_generations = []
    warned_context_overflow = False
    warned_context_skip = False

    for batch in tqdm(dataloader, disable=(get_rank() != 0)):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        effective_gen_length = gen_length

        if max_context_length is not None and max_context_length > 0:
            remaining_context = max_context_length - input_ids.shape[1]
            if remaining_context < gen_length:
                effective_gen_length = max(0, (remaining_context // block_length) * block_length)

                if get_rank() == 0 and not warned_context_overflow:
                    print(
                        f"[warn] prompt_length({input_ids.shape[1]}) + gen_length({gen_length}) exceeds "
                        f"max_context_length({max_context_length}). "
                        f"Using reduced gen_length({effective_gen_length}) for this batch."
                    )
                    warned_context_overflow = True

        if effective_gen_length <= 0:
            if get_rank() == 0 and not warned_context_skip:
                print(
                    f"[warn] prompt_length({input_ids.shape[1]}) leaves no room for generation within "
                    f"max_context_length({max_context_length}). Skipping this batch."
                )
                warned_context_skip = True
            continue

        out = generate(
            model,
            input_ids,
            tokenizer,
            steps=steps,
            gen_length=effective_gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking="low_confidence",
            mask_id=mask_id,
            newline_later=newline_later,
            earlystop=earlystop,
        )

        generated_texts = decode_generated_texts(
            tokenizer,
            out[:, -effective_gen_length:],
            mask_id=mask_id,
            truncate_at_mask=earlystop,
        )
        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
            }
            for j in range(len(gt_answers))
        ]
        annotate_generation_results(
            dataset_name,
            example_result,
            prompt_style=prompt_style,
            truncate_repeated_newlines=earlystop,
        )
        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        wall_times.append(time.time() - start_time)

        # Print individual results
        if get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            print("Generation:")
            sample_generation = generated_texts[idx]
            print(format_generation_for_log(sample_generation))
            print("-" * 50)
            print(f"Generated Answer: {get_logged_generated_answer(example_result[idx])}")
            print(f"Ground truth: {gt_answers[idx]}")
            if "correct" in example_result[idx]:
                print(f"Correct: {example_result[idx]['correct']}")

    avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0.0
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            # If we don't drop the last batch, we need to calculate the number of samples per rank.
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


if __name__ == "__main__":
    init_seed(42)

    device = setup_device()
    local_rank = device.index if device.type == "cuda" else 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data1/shared/LLaDA-8B-Instruct/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math"],
        default="gsm8k",
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument(
        "--prompt_style",
        type=str,
        choices=["default", "format", "answer_first"],
        default="default",
        help="Prompt format used to build eval inputs.",
    )
    parser.add_argument(
        "--mask_id",
        type=int,
        default=-1,
        help="Override diffusion mask token id. Negative values auto-resolve from the tokenizer.",
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=None,
        help="Optional total prompt+generation budget used for overflow warnings.",
    )
    parser.add_argument("--newline_later", action="store_true")
    parser.add_argument("--earlystop", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--dont_use_box", action="store_true")
    parser.add_argument(
        "--subsample",
        type=int,
        default=-1,
        help="If > 0, evaluate only this many samples (useful for smoke tests).",
    )
    args = parser.parse_args()

    # args.diffusion_steps = args.gen_length // 2
    num_evals = {"gsm8k": -1, "math": -1}
    dataset_subsample = args.subsample if args.subsample > 0 else num_evals[args.dataset]

    tokenizer, model = load_eval_model_and_tokenizer(
        args.model_path,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    resolved_mask_id = resolve_mask_id(tokenizer, configured_mask_id=args.mask_id)
    if get_rank() == 0:
        print(f"Resolved mask_id: {resolved_mask_id}")
        print(f"Prompt style    : {args.prompt_style}")
        print(f"Newline later   : {args.newline_later}")
        print(f"Early stop      : {args.earlystop}")

    if args.checkpoint_path:
        try:
            from peft import PeftModel
        except ImportError as import_error:
            raise ImportError(
                "Checkpoint evaluation requires `peft` because --checkpoint_path was provided."
            ) from import_error

        model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=torch.bfloat16).to(device)

        if get_world_size() > 1:
            dist.barrier()  # Make sure all processes are ready
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            print(f"Rank {local_rank}: Parameters synchronized")

    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=dataset_subsample,
        prompt_style=args.prompt_style,
    )

    sampler = CustomDistributedSampler(dataset, shuffle=False) if get_world_size() > 1 else None

    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=dataset.collate_fn)

    if len(args.checkpoint_path):
        model_name = args.checkpoint_path.split("/")
        model_name = model_name[-2] + "_" + model_name[-1]
    else:
        model_name = args.model_path.split("/")[-1]
        # model_name = "instruct" if "Instruct" in args.model_path else "base"

    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{args.output_dir}/{args.dataset}_{model_name}_{args.gen_length}_{args.diffusion_steps}_{get_rank()}_generations.json"
    print(f"Saving generations to {filename}")

    metrics = evaluate(
        model,
        tokenizer,
        dataloader,
        dataset_name=args.dataset,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        mask_id=resolved_mask_id,
        max_context_length=args.max_context_length,
        prompt_style=args.prompt_style,
        newline_later=args.newline_later,
        earlystop=args.earlystop,
    )

    if args.dataset == "gsm8k":
        gsm8k_metrics = compute_gsm8k_accuracy(
            metrics["generations"],
            prompt_style=args.prompt_style,
            truncate_repeated_newlines=args.earlystop,
        )
        metrics.update(gsm8k_metrics)
        print(
            f"GSM8K accuracy: {gsm8k_metrics['correct']}/{gsm8k_metrics['processed']} "
            f"({gsm8k_metrics['accuracy']:.2f}%)"
        )
    elif args.dataset == "math":
        math_metrics = compute_math_accuracy(
            metrics["generations"],
            prompt_style=args.prompt_style,
            truncate_repeated_newlines=args.earlystop,
        )
        metrics.update(math_metrics)
        print(
            f"{args.dataset.upper()} accuracy: {math_metrics['correct']}/{math_metrics['processed']} "
            f"({math_metrics['accuracy']:.2f}%)"
        )

    if not args.dont_save:
        saved_metrics = {
            "wall_time": metrics["wall_time"],
            "total_processed": metrics.get("processed", metrics["total_processed"]),
            "correct": metrics.get("correct"),
            "processed": metrics.get("processed"),
            "accuracy": metrics.get("accuracy"),
        }

        with open(filename, "w") as f:
            json.dump(
                {
                    "generations": metrics["generations"],
                    "metrics": saved_metrics,
                    "model_path": args.model_path,
                    "checkpoint_path": args.checkpoint_path,
                    "gen_length": args.gen_length,
                    "diffusion_steps": args.diffusion_steps,
                    "block_length": args.block_length,
                    "prompt_style": args.prompt_style,
                    "newline_later": args.newline_later,
                    "earlystop": args.earlystop,
                },
                f,
                indent=2,
            )

    cleanup_ddp()
