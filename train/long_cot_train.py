import os
import sys
from argparse import ArgumentParser

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser

from long_cot_data_utils import get_distillation_datasets, set_random_seed
from long_cot_config import DiffuSelfDistillConfig
from long_cot_trainer import (
    DiffuSelfDistillDataCollator,
    DiffuSelfDistillTrainer,
    normalize_ce_mask_mode,
    normalize_training_method,
)


def configure_wandb_env(args: DiffuSelfDistillConfig):
    report_to = args.report_to
    if isinstance(report_to, str):
        report_to = [report_to]
    if "wandb" not in (report_to or []):
        return

    os.environ.setdefault("WANDB_PROJECT", "long-cot")
    if getattr(args, "run_name", None):
        os.environ.setdefault("WANDB_NAME", str(args.run_name))


def parse_args() -> DiffuSelfDistillConfig:
    parser = HfArgumentParser(DiffuSelfDistillConfig)
    if len(sys.argv) == 2 and sys.argv[1].endswith((".json", ".yaml", ".yml")):
        config_path = os.path.abspath(sys.argv[1])
        if config_path.endswith(".json"):
            (args,) = parser.parse_json_file(config_path)
        else:
            (args,) = parser.parse_yaml_file(config_path)
        return args

    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    known, remaining = pre_parser.parse_known_args(sys.argv[1:])

    if known.config:
        config_path = os.path.abspath(known.config)
        if config_path.endswith(".json"):
            import json

            with open(config_path, "r", encoding="utf-8") as handle:
                config_dict = json.load(handle)
        else:
            try:
                import yaml
            except ImportError as import_error:
                raise ImportError("`--config` with YAML requires pyyaml installed.") from import_error
            with open(config_path, "r", encoding="utf-8") as handle:
                config_dict = yaml.safe_load(handle)

        cli_from_config = []
        for key, value in (config_dict or {}).items():
            if value is None:
                continue
            cli_from_config.extend([f"--{key}", str(value)])

        (args,) = parser.parse_args_into_dataclasses(args=cli_from_config + remaining)
        return args

    (args,) = parser.parse_args_into_dataclasses()
    return args


def get_torch_dtype(args: DiffuSelfDistillConfig):
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return None


def load_model_and_tokenizer(args: DiffuSelfDistillConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="right",
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer_vocab_size = len(tokenizer)
    if hasattr(config, "vocab_size") and config.vocab_size != tokenizer_vocab_size:
        print(
            f"Adjusting config.vocab_size from {config.vocab_size} to {tokenizer_vocab_size} "
            f"to match tokenizer at {args.model_path}"
        )
        config.vocab_size = tokenizer_vocab_size
    if (
        os.path.isdir(args.model_path)
        and hasattr(config, "embedding_size")
        and getattr(config, "embedding_size") is not None
        and getattr(config, "embedding_size") != tokenizer_vocab_size
    ):
        print(
            f"Adjusting local config.embedding_size from {config.embedding_size} to {tokenizer_vocab_size} "
            f"to match tokenizer at {args.model_path}"
        )
        config.embedding_size = tokenizer_vocab_size

    model_kwargs = {"trust_remote_code": True, "config": config}
    model_dtype = get_torch_dtype(args)
    if model_dtype is not None:
        model_kwargs["torch_dtype"] = model_dtype

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModel.from_pretrained(args.model_path, **model_kwargs)
    if (
        hasattr(model, "get_input_embeddings")
        and model.get_input_embeddings() is not None
        and model.get_input_embeddings().num_embeddings != len(tokenizer)
        and hasattr(model, "resize_token_embeddings")
    ):
        model.resize_token_embeddings(len(tokenizer))
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if args.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as import_error:
            raise ImportError(
                "LoRA requested (`use_lora=True`) but `peft` is not installed."
            ) from import_error

        target_modules = [module.strip() for module in args.lora_target_modules.split(",") if module.strip()]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        if args.bf16:
            model = model.to(torch.bfloat16)

    return tokenizer, model


def main():
    args = parse_args()
    args.method_spec = args.method
    args.method = normalize_training_method(args.method_spec)
    args.ce_mask_mode = normalize_ce_mask_mode(args.ce_mask_mode)
    configure_wandb_env(args)
    set_random_seed(args.seed)

    tokenizer, model = load_model_and_tokenizer(args)

    train_dataset, eval_dataset = get_distillation_datasets(
        tokenizer=tokenizer,
        dataset_path=args.dataset_path,
        train_split=args.train_split,
        eval_split=args.eval_split,
        seed=args.seed,
        gold_mode=args.gold_mode,
        target_mode=args.target_mode,
        teacher_reference_mode=args.teacher_reference_mode,
        reference_response_source=args.reference_response_source,
        target_response_source=args.target_response_source,
        heldout_eval_ratio=args.heldout_eval_ratio,
    )

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if eval_dataset is not None and args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    mask_id = args.mask_id
    if mask_id < 0:
        if tokenizer.mask_token_id is None:
            raise ValueError("mask_id < 0 and tokenizer has no mask_token_id.")
        mask_id = tokenizer.mask_token_id

    data_collator = DiffuSelfDistillDataCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        mask_id=mask_id,
        t_min=args.t_min,
        t_max=args.t_max,
        t_sampling_mode=args.t_sampling_mode,
        t_fixed=args.t_fixed,
        t_biased_to_one_strength=args.t_biased_to_one_strength,
        t_two_point_low=args.t_two_point_low,
        t_two_point_high=args.t_two_point_high,
        t_two_point_high_prob=args.t_two_point_high_prob,
        t_curriculum_start_min=args.t_curriculum_start_min,
        t_curriculum_start_max=args.t_curriculum_start_max,
        t_curriculum_end_min=args.t_curriculum_end_min,
        t_curriculum_end_max=args.t_curriculum_end_max,
        t_curriculum_total_batches=args.t_curriculum_total_batches,
        method=args.method_spec,
        dataset_name=args.dataset,
    )

    trainer = DiffuSelfDistillTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
