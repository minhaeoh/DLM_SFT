from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class DiffuSelfDistillConfig(TrainingArguments):
    """
    Training configuration for long-CoT SFT training.

    This trainer is intentionally narrow:
    - only `method=SFT` is supported
    - the main loss is masked-token cross entropy on the noisy target response
    - optional auxiliary CE, when enabled, is always applied over the full response
    - there is no teacher prompt / teacher forward branch
    """

    model_path: str = field(
        default="relaxe-system-lab/UltraLLaDA",
        metadata={"help": "Base model path or checkpoint path (base/SFT/continued)."},
    )
    dataset: str = field(
        default="math_long_cot",
        metadata={"help": "Metadata label only."},
    )
    dataset_path: str = field(
        default="dataset/Math-CoT-NoCoT-20k-4096",
        metadata={"help": "Dataset path loaded via datasets.load_from_disk()."},
    )
    target_response_source: str = field(
        default="cot",
        metadata={"help": "Response source used as the clean training target: cot | noncot."},
    )
    prompt_type: str = field(
        default="default",
        metadata={"help": "Prompt style appended to each question: default | format | answer_first."},
    )
    answer_block: bool = field(
        default=False,
        metadata={
            "help": "If True with prompt_type=answer_first, pad the <answer>...</answer> block to a fixed 32-token length before <reasoning> starts."
        },
    )
    max_length: int = field(
        default=4096,
        metadata={"help": "Max total length for [prompt; response]."},
    )
    train_split: str = field(
        default="train",
        metadata={"help": "Dataset split used for training where applicable."},
    )
    eval_split: Optional[str] = field(
        default=None,
        metadata={"help": "Optional eval split. If unset, no eval dataset is attached."},
    )
    heldout_eval_ratio: float = field(
        default=0.01,
        metadata={
            "help": "If eval_split is one of heldout/eval/validation/test and the dataset only has train, split this ratio off on the fly."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap for train dataset size (debug/smoke runs)."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap for eval dataset size (debug/smoke runs)."},
    )
    method: str = field(
        default="SFT",
        metadata={
            "help": "Compatibility flag. `SFT` is the canonical method name; legacy INP_OH aliases are still accepted."
        },
    )

    mask_id: int = field(
        default=126336,
        metadata={"help": "Mask token id for diffusion noising."},
    )
    t_min: float = field(
        default=1e-3,
        metadata={"help": "Minimum diffusion timestep for Uniform(t_min, t_max)."},
    )
    t_max: float = field(
        default=1.0,
        metadata={"help": "Maximum diffusion timestep for Uniform(t_min, t_max)."},
    )
    t_sampling_mode: str = field(
        default="uniform",
        metadata={
            "help": "How to sample t values: uniform | fixed | biased_to_one | biased_to_zero | logit_normal | two_point | curriculum."
        },
    )
    t_fixed: float = field(
        default=0.9,
        metadata={"help": "Fixed timestep used when t_sampling_mode=fixed."},
    )
    t_biased_to_one_strength: float = field(
        default=2.0,
        metadata={
            "help": "Strength for t_sampling_mode=biased_to_one. > 1 biases more samples toward higher t."
        },
    )
    t_biased_to_zero_strength: float = field(
        default=2.0,
        metadata={
            "help": "Strength for t_sampling_mode=biased_to_zero. > 1 biases more samples toward lower t."
        },
    )
    t_logit_normal_mean: float = field(
        default=0.0,
        metadata={
            "help": "Mean of the normal distribution in logit space for t_sampling_mode=logit_normal. 0 centers mass at t=0.5."
        },
    )
    t_logit_normal_std: float = field(
        default=1.0,
        metadata={
            "help": "Std of the normal distribution in logit space for t_sampling_mode=logit_normal."
        },
    )
    t_two_point_low: float = field(
        default=0.2,
        metadata={"help": "Low timestep used when t_sampling_mode=two_point."},
    )
    t_two_point_high: float = field(
        default=0.9,
        metadata={"help": "High timestep used when t_sampling_mode=two_point."},
    )
    t_two_point_high_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of sampling t_two_point_high when t_sampling_mode=two_point."},
    )
    t_curriculum_start_min: float = field(
        default=0.0,
        metadata={"help": "Curriculum start range lower bound for t_sampling_mode=curriculum."},
    )
    t_curriculum_start_max: float = field(
        default=0.4,
        metadata={"help": "Curriculum start range upper bound for t_sampling_mode=curriculum."},
    )
    t_curriculum_end_min: float = field(
        default=0.8,
        metadata={"help": "Curriculum end range lower bound for t_sampling_mode=curriculum."},
    )
    t_curriculum_end_max: float = field(
        default=1.0,
        metadata={"help": "Curriculum end range upper bound for t_sampling_mode=curriculum."},
    )
    t_curriculum_total_batches: int = field(
        default=0,
        metadata={
            "help": "Optional total number of batch calls used by curriculum progress. <= 0 lets the trainer infer it."
        },
    )
    distill_temperature: float = field(
        default=1.0,
        metadata={"help": "Distillation temperature (tau)."},
    )
    kd_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for forward-KL distillation loss."},
    )
    ce_weight: float = field(
        default=0.5,
        metadata={"help": "Optional auxiliary CE weight. If <= 0, CE is disabled."},
    )
    loss_chunk_size: int = field(
        default=128,
        metadata={"help": "Number of response tokens processed at once when computing losses."},
    )
    disable_dropout: bool = field(
        default=True,
        metadata={
            "help": "Disable dropout during training for more stable masked-token targets."
        },
    )
    debug_save_examples: int = field(
        default=0,
        metadata={
            "help": "If > 0, save per-batch debug examples with masked inputs and student predictions."
        },
    )
    debug_save_examples_filename: str = field(
        default="debug_training_examples.jsonl",
        metadata={"help": "Filename under output_dir used for saving debug training examples."},
    )
    debug_save_every_steps: int = field(
        default=100,
        metadata={
            "help": "Deprecated compatibility flag. Debug training examples are now appended for every training batch across the full run."
        },
    )
    debug_save_logits_topk: int = field(
        default=5,
        metadata={"help": "Top-k logits/probs to store per masked position in debug example dumps."},
    )
    debug_save_max_masked_positions: int = field(
        default=128,
        metadata={"help": "Maximum number of masked response positions to serialize per saved debug example."},
    )

    use_lora: bool = field(
        default=True,
        metadata={"help": "Enable LoRA fine-tuning."},
    )
    lora_r: int = field(default=128, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        metadata={"help": "Comma-separated target module names for LoRA."},
    )

    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load model with 4-bit quantization (bitsandbytes)."},
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "4-bit quantization type for bitsandbytes."},
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={"help": "Enable nested quantization for 4-bit loading."},
    )

    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Keep custom collator fields."},
    )
