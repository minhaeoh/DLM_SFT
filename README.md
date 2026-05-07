# DLM_SFT

Minimal standalone package extracted for these two launchers:

- `train/train_math_4096_experiments.sh`
- `eval/run_gsm8k_test_eval.sh`

## Structure

- `train/`
  - long-CoT training entrypoint and helpers
  - `ddp_config_no_deepspeed.yaml`
- `eval/`
  - `gsm8k` / `math` evaluation entrypoint and helpers only

## Path behavior

- Training defaults to `train/ddp_config_no_deepspeed.yaml`.
- Training writes outputs under `train/checkpoints/` and logs under `train/logs/`.
- Evaluation writes outputs under `eval/eval_results/` and logs under `eval/logs/`.
- Training looks for the dataset at `DLM_SFT/dataset/Math-CoT-NoCoT-20k-format-4096` first.
- If that local dataset folder is absent, training falls back to:
  - `/home/minhae/diffusion/diffu-distill/d1-self-distill/dataset/Math-CoT-NoCoT-20k-format-4096`
- Evaluation auto-discovers the latest LoRA checkpoint under:
  - `train/checkpoints/long-CoT/math_format_4096_base/ultrallada`
  - You can still override it with `CHECKPOINT_PATH=...`.

## Notes

- The eval package was trimmed to `gsm8k` and `math` only.
- `gsm8k`, `HuggingFaceH4/MATH-500`, and `EleutherAI/hendrycks_math` are still loaded through Hugging Face datasets.
