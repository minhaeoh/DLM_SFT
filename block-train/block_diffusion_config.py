from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class BlockDiffusionConfig(TrainingArguments):
    """
    Training configuration for block diffusion with an INP-OH objective.

    This trainer is intentionally narrow:
    - only `method=INP-OH` is supported
    - the only loss is masked-token cross entropy on the active noisy block
    - no teacher/KD branch or auxiliary CE branch is used
    """

    model_path: str = field(
        default="relaxe-system-lab/UltraLLaDA",
        metadata={"help": "Base model path or checkpoint path."},
    )
    dataset: str = field(
        default="math_long_cot",
        metadata={"help": "Metadata label only."},
    )
    dataset_path: str = field(
        default="dataset/Math-CoT-NoCoT-20k-4096",
        metadata={"help": "Dataset path loaded via datasets.load_from_disk()."},
    )
    reference_response_source: str = field(
        default="cot",
        metadata={
            "help": "Kept for dataset formatting compatibility. Not used by the block diffusion trainer itself."
        },
    )
    target_response_source: str = field(
        default="cot",
        metadata={"help": "Response source used as the clean training target: cot | noncot."},
    )
    gold_mode: int = field(
        default=1,
        metadata={"help": "Compatibility flag reused from the long-CoT data formatter."},
    )
    target_mode: int = field(
        default=1,
        metadata={"help": "Compatibility flag reused from the long-CoT data formatter."},
    )
    teacher_reference_mode: str = field(
        default="full",
        metadata={
            "help": "Compatibility flag for dataset formatting. The trainer does not consume teacher prompts."
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
            "help": "If eval_split requests a heldout split and the dataset only has train, split off this ratio."
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
        default="INP-OH",
        metadata={"help": "Compatibility flag. Only INP-OH is supported in block-train."},
    )

    mask_id: int = field(
        default=126336,
        metadata={"help": "Mask token id for the block diffusion forward noise process."},
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
            "help": "How to sample t values: uniform | fixed | biased_to_one | two_point | curriculum."
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
    block_size: int = field(
        default=16,
        metadata={"help": "Size of each contiguous response block used by block diffusion."},
    )
    loss_chunk_size: int = field(
        default=128,
        metadata={"help": "Number of masked block positions processed at once when computing CE loss."},
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Disable dropout so cache-conditioned denoising targets stay stable."},
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
