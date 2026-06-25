#!/usr/bin/env python3
"""
Reflection Experiments for LLaDA Wrong Reasoning Correction
============================================================

Experiment 1 — Inplace-correction:
  Prompt = [problem] + [correct_prefix] + [MASK × wrong_sentence_len] + [reflection_token]
  Method  = single forward pass; inspect model predictions at MASK positions.
  Question: does the model predict correct tokens where it originally went wrong?

Experiment 2 — Forward-correction:
  Prompt = [problem] + [correct_prefix + wrong_sentence] + [reflection_token]
  Method  = full LLaDA generation after the reflection token.
  Question: does the subsequent generation correct the error and reach the right answer?

Experiment 3 — Context-aware replace:
  Prompt = [problem] + [correct_prefix + wrong_sentence] + [reflection_token]  ← same as Exp 2
  Method  = single forward pass (no masking); for each token in wrong_sentence,
            if model prediction ≠ original token → replace in-place.
  Question: does the reflection context shift the model's token predictions enough
            to self-correct the wrong reasoning step?

Experiment 4 — Mask-then-generate:
  Prompt = [problem] + [correct_prefix + wrong_sentence] + [reflection_token]  ← same as Exp 2/3
  Method  = single forward pass → positions where prediction ≠ original → MASK them →
            run iterative LLaDA denoising to fill ONLY those masked positions.
  Question: does two-stage correction (identify uncertain tokens, then properly
            denoise them with full bidirectional context) produce better fixes than
            direct replacement (Exp 3)?

Usage (two-stage):
  # Stage 1 — run a small eval to obtain wrong examples (see run_reflection_experiments.sh)
  # Stage 2 — run this script on the saved JSON
  python reflection_experiments.py \\
      --model_path GSAI-ML/LLaDA-8B-Base \\
      --checkpoint_path ../checkpoints/.../checkpoint-238 \\
      --eval_results path/to/eval_results.json \\
      --output_dir results/reflection/ \\
      [--split_sentences_from_end 2] \\
      [--reflection_token "wait_think"] \\
      [--max_wrong_examples 20] \\
      [--exp2_gen_length 512]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

# Allow running from the eval/ directory directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from eval import (
    load_eval_model_and_tokenizer,
    resolve_mask_id,
    extract_math_answer,
    compute_math_accuracy,
    _preprocess_math_answer,
)
from generate import generate
from math_scorer import MATHScorer as _math_scorer

# ---------------------------------------------------------------------------
# Reflection token options
# ---------------------------------------------------------------------------
REFLECTION_TOKENS = {
    "wait_think":  "\n\nWait, let me think about this again.\n\n",
    "reconsider":  "\n\nActually, let me reconsider.\n\n",
    "error":       "\n\nI think I made an error. Let me redo this.\n\n",
    "try_again":   "\n\nLet me try again from scratch.\n\n",
}


# ---------------------------------------------------------------------------
# Helpers: split the generation at a heuristic "first wrong sentence" boundary
# ---------------------------------------------------------------------------

def _split_by_lines(text):
    """Return non-empty lines preserving newline-delimited structure."""
    return [ln for ln in text.split("\n") if ln.strip()]


def find_wrong_boundary(
    generation_text: str,
    sentences_from_end: int = 2,
) -> tuple[str, str, str]:
    """
    Heuristic split of *generation_text* into three parts:
      correct_prefix   — reasoning assumed correct
      wrong_sentence   — first wrong step (to mask / prefix-with-reflection)
      remaining        — everything after (typically empty or the old boxed line)

    Strategy:
      • Split by newline into logical "lines" (steps).
      • Find the FIRST line containing \\boxed or \\fbox (the final answer line).
      • The wrong_sentence = the `sentences_from_end` lines just before the boxed line.
      • correct_prefix = everything before wrong_sentence.
      • remaining = boxed line + anything after.

    Falls back to a 70 / 30 character split when no \\boxed is found.
    """
    lines = _split_by_lines(generation_text)

    if not lines:
        mid = len(generation_text) // 2
        return generation_text[:mid], generation_text[mid:mid+1], generation_text[mid+1:]

    # Find first boxed line
    boxed_idx = None
    for i, line in enumerate(lines):
        if r"\boxed" in line or r"\fbox" in line:
            boxed_idx = i
            break

    if boxed_idx is None:
        # No boxed: split at 70%
        split_at = max(1, int(len(lines) * 0.70))
        wrong_end = min(split_at + sentences_from_end, len(lines))
        correct_lines = lines[:split_at]
        wrong_lines   = lines[split_at:wrong_end]
        rest_lines    = lines[wrong_end:]
    else:
        wrong_start = max(0, boxed_idx - sentences_from_end)
        correct_lines = lines[:wrong_start]
        wrong_lines   = lines[wrong_start:boxed_idx]
        rest_lines    = lines[boxed_idx:]

    def rejoin(ls):
        return "\n".join(ls) if ls else ""

    return rejoin(correct_lines), rejoin(wrong_lines), rejoin(rest_lines)


# ---------------------------------------------------------------------------
# Tokenisation utility: find char-offset-based token positions
# ---------------------------------------------------------------------------

def _char_to_token_range(tokenizer, full_text: str, start_char: int, end_char: int):
    """
    Return (tok_start, tok_end) such that full_ids[tok_start:tok_end] corresponds
    to full_text[start_char:end_char], using offset_mapping.
    Returns (None, None) if the tokenizer doesn't support offset_mapping.
    """
    try:
        enc = tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
    except Exception:
        return None, None

    offsets = enc["offset_mapping"][0].tolist()  # list of (s, e) per token
    tok_start, tok_end = None, None
    for i, (s, e) in enumerate(offsets):
        if e == 0 and s == 0:
            continue  # special tokens have (0,0)
        if tok_start is None and e > start_char:
            tok_start = i
        if e <= end_char:
            tok_end = i + 1

    if tok_start is None:
        tok_start = 0
    if tok_end is None or tok_end <= tok_start:
        tok_end = tok_start + 1

    return tok_start, tok_end


# ---------------------------------------------------------------------------
# Experiment 1: Inplace-correction (single forward pass)
# ---------------------------------------------------------------------------

def run_exp1_inplace(
    model,
    tokenizer,
    mask_id: int,
    prompt_text: str,
    correct_prefix: str,
    wrong_sentence: str,
    reflection_token: str,
    device,
) -> dict:
    """
    Build: [prompt][correct_prefix][MASK × wrong_len][reflection_token]
    Run a single model forward pass.
    Return predicted tokens at the MASK positions.
    """
    # Build the full text (without masking) to get consistent tokenisation.
    # We rely on char offsets to find where wrong_sentence lives.
    prefix_text  = prompt_text + correct_prefix
    full_text    = prefix_text + wrong_sentence + reflection_token

    try:
        enc_full = tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        has_offsets = True
    except Exception:
        enc_full = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
        has_offsets = False

    full_ids = enc_full["input_ids"][0].clone()

    if has_offsets:
        offsets = enc_full["offset_mapping"][0].tolist()
        wrong_start_char = len(prefix_text)
        wrong_end_char   = wrong_start_char + len(wrong_sentence)

        mask_positions = []
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue  # special token
            if s < wrong_end_char and e > wrong_start_char:
                mask_positions.append(i)
    else:
        # Fallback: tokenise prefix separately and derive wrong token span
        prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        wrong_ids  = tokenizer(wrong_sentence, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        p_len = len(prefix_ids)
        w_len = len(wrong_ids)
        mask_positions = list(range(p_len, p_len + w_len))

    if not mask_positions:
        return {
            "error": "No mask positions found — wrong_sentence may be empty or tokenisation failed.",
            "wrong_sentence_original": wrong_sentence,
            "predicted_inplace": "",
        }

    # Replace wrong-sentence tokens with mask_id
    masked_ids = full_ids.clone()
    for pos in mask_positions:
        if pos < len(masked_ids):
            masked_ids[pos] = mask_id

    input_tensor = masked_ids.unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            logits = model(input_ids=input_tensor).logits  # (1, seq, vocab)

    # Predicted tokens at MASK positions
    predicted_ids = []
    original_ids  = []
    for pos in mask_positions:
        if pos < logits.shape[1]:
            pred = int(logits[0, pos].argmax().item())
            predicted_ids.append(pred)
            original_ids.append(int(full_ids[pos].item()))

    predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    original_text  = tokenizer.decode(original_ids,  skip_special_tokens=False, clean_up_tokenization_spaces=False)

    # Check if prediction is different from original
    tokens_changed = sum(p != o for p, o in zip(predicted_ids, original_ids))

    return {
        "wrong_sentence_original": wrong_sentence,
        "wrong_token_ids":         original_ids,
        "predicted_inplace":       predicted_text,
        "predicted_token_ids":     predicted_ids,
        "tokens_changed":          tokens_changed,
        "total_mask_tokens":       len(mask_positions),
        "mask_positions":          mask_positions,
        "input_length":            int(input_tensor.shape[1]),
    }


# ---------------------------------------------------------------------------
# Experiment 2: Forward-correction (full LLaDA generation)
# ---------------------------------------------------------------------------

def run_exp2_forward(
    model,
    tokenizer,
    mask_id: int,
    prompt_text: str,
    prefix_incl_wrong: str,
    reflection_token: str,
    gen_length: int,
    diffusion_steps: int,
    block_length: int,
    device,
) -> dict:
    """
    Build: [prompt][prefix_incl_wrong][reflection_token] then generate gen_length tokens.
    Returns generated text and whether it contains a parseable (hopefully correct) answer.
    """
    full_prompt = prompt_text + prefix_incl_wrong + reflection_token
    input_ids = tokenizer(
        full_prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    prompt_length = input_ids.shape[1]

    # Align gen_length to block_length
    gen_length = max(block_length, (gen_length // block_length) * block_length)

    with torch.no_grad():
        out = generate(
            model,
            input_ids,
            tokenizer,
            steps=diffusion_steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=mask_id,
        )

    generated_ids  = out[0, prompt_length:].tolist()
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    # Strip trailing EOS/pad artefacts
    eot_marker = "<|eot_id|>"
    if eot_marker in generated_text:
        generated_text = generated_text.split(eot_marker, 1)[0]

    parsed_answer = extract_math_answer(generated_text)

    return {
        "full_prompt":     full_prompt,
        "prompt_length":   prompt_length,
        "gen_length":      gen_length,
        "generated_text":  generated_text,
        "parsed_answer":   parsed_answer,
    }


# ---------------------------------------------------------------------------
# Experiment 3: Context-aware replace (same input as Exp 2, single forward pass)
# ---------------------------------------------------------------------------

def run_exp3_context_replace(
    model,
    tokenizer,
    mask_id: int,
    prompt_text: str,
    correct_prefix: str,
    wrong_sentence: str,
    reflection_token: str,
    remaining: str,
    device,
) -> dict:
    """
    Input  = [prompt][correct_prefix][wrong_sentence][reflection_token]  ← identical to Exp 2
    Method = single forward pass (no masking); at positions of wrong_sentence,
             replace tokens where model prediction ≠ original token.
    Check  = does the corrected wrong_sentence (+ original remaining) contain
             a parseable answer?

    Key contrast with Exp 1:
      Exp 1 — wrong_sentence is *masked*, model predicts blind.
      Exp 3 — wrong_sentence is *visible*, model predicts given full context
               (including the reflection token to its right).
    """
    prefix_text = prompt_text + correct_prefix
    full_text   = prefix_text + wrong_sentence + reflection_token

    try:
        enc = tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        has_offsets = True
    except Exception:
        enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
        has_offsets = False

    full_ids = enc["input_ids"][0].clone()

    if has_offsets:
        offsets = enc["offset_mapping"][0].tolist()
        wrong_start_char = len(prefix_text)
        wrong_end_char   = wrong_start_char + len(wrong_sentence)
        wrong_positions  = [
            i for i, (s, e) in enumerate(offsets)
            if s < wrong_end_char and e > wrong_start_char and not (s == 0 and e == 0)
        ]
    else:
        prefix_ids      = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        wrong_ids_tmp   = tokenizer(wrong_sentence, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        p_len           = len(prefix_ids)
        wrong_positions = list(range(p_len, p_len + len(wrong_ids_tmp)))

    if not wrong_positions:
        return {
            "error": "No wrong_sentence positions found.",
            "wrong_sentence_original":  wrong_sentence,
            "wrong_sentence_corrected": wrong_sentence,
            "tokens_changed": 0,
        }

    input_tensor = full_ids.unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            logits = model(input_ids=input_tensor).logits  # (1, seq, vocab)

    original_ids  = [int(full_ids[p].item()) for p in wrong_positions if p < logits.shape[1]]
    predicted_ids = [int(logits[0, p].argmax().item()) for p in wrong_positions if p < logits.shape[1]]

    # Replace tokens where prediction differs
    corrected_ids = full_ids.clone()
    tokens_changed = 0
    changed_positions = []
    for i, pos in enumerate(wrong_positions):
        if pos >= logits.shape[1]:
            continue
        if predicted_ids[i] != original_ids[i]:
            corrected_ids[pos] = predicted_ids[i]
            tokens_changed += 1
            changed_positions.append(pos)

    # Decode corrected wrong_sentence
    corrected_wrong_ids = [int(corrected_ids[p].item()) for p in wrong_positions if p < len(corrected_ids)]
    corrected_wrong_text = tokenizer.decode(
        corrected_wrong_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    original_wrong_text = tokenizer.decode(
        original_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    # Try to extract an answer from corrected_wrong + remaining combined
    combined_for_answer = corrected_wrong_text + ("\n" + remaining if remaining else "")
    parsed_answer = extract_math_answer(combined_for_answer)

    return {
        "wrong_sentence_original":   original_wrong_text,
        "wrong_sentence_corrected":  corrected_wrong_text,
        "original_token_ids":        original_ids,
        "predicted_token_ids":       predicted_ids,
        "tokens_changed":            tokens_changed,
        "total_wrong_tokens":        len(wrong_positions),
        "changed_positions":         changed_positions,
        "parsed_answer":             parsed_answer,
        "input_length":              int(input_tensor.shape[1]),
    }


# ---------------------------------------------------------------------------
# Experiment 4: Mask-then-generate (same input as Exp 2/3)
# ---------------------------------------------------------------------------

def run_exp4_mask_then_generate(
    model,
    tokenizer,
    mask_id: int,
    prompt_text: str,
    correct_prefix: str,
    wrong_sentence: str,
    reflection_token: str,
    remaining: str,
    device,
    steps: int = 50,
) -> dict:
    """
    Input  = [prompt][correct_prefix][wrong_sentence][reflection_token]  ← same as Exp 2/3
    Step 1 = single forward pass (no masking) → find positions in wrong_sentence
             where argmax(logits) ≠ original token.
    Step 2 = MASK those positions (set to mask_id).
    Step 3 = iterative LLaDA denoising: repeatedly forward-pass the full sequence,
             pick the most confident masked position, unmask it — until all masked
             positions are filled.

    Key contrast with Exp 3:
      Exp 3 — replace directly with the first forward-pass prediction (one shot).
      Exp 4 — mask first, then let the full diffusion process (with bidirectional
               context that now includes a gap) converge iteratively.
    """
    prefix_text = prompt_text + correct_prefix
    full_text   = prefix_text + wrong_sentence + reflection_token

    try:
        enc = tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        has_offsets = True
    except Exception:
        enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
        has_offsets = False

    full_ids = enc["input_ids"][0].clone()

    # ── Identify wrong_sentence token positions ──────────────────────────
    if has_offsets:
        offsets = enc["offset_mapping"][0].tolist()
        wrong_start_char = len(prefix_text)
        wrong_end_char   = wrong_start_char + len(wrong_sentence)
        wrong_positions  = [
            i for i, (s, e) in enumerate(offsets)
            if s < wrong_end_char and e > wrong_start_char and not (s == 0 and e == 0)
        ]
    else:
        prefix_ids     = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        wrong_ids_tmp  = tokenizer(wrong_sentence, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        p_len          = len(prefix_ids)
        wrong_positions = list(range(p_len, p_len + len(wrong_ids_tmp)))

    if not wrong_positions:
        return {
            "error": "No wrong_sentence positions found.",
            "wrong_sentence_original":  wrong_sentence,
            "wrong_sentence_corrected": wrong_sentence,
            "tokens_masked": 0,
        }

    # ── Step 1: forward pass (no masking) ───────────────────────────────
    x = full_ids.clone().unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            logits = model(input_ids=x).logits

    original_ids  = [int(full_ids[p].item()) for p in wrong_positions if p < logits.shape[1]]
    predicted_ids = [int(logits[0, p].argmax().item()) for p in wrong_positions if p < logits.shape[1]]

    # ── Step 2: mask positions where prediction ≠ original ───────────────
    positions_to_mask = [
        wrong_positions[i]
        for i in range(len(original_ids))
        if predicted_ids[i] != original_ids[i]
    ]
    tokens_masked = len(positions_to_mask)

    if tokens_masked == 0:
        orig_text = tokenizer.decode(original_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return {
            "wrong_sentence_original":  orig_text,
            "wrong_sentence_corrected": orig_text,
            "original_token_ids":       original_ids,
            "tokens_masked":            0,
            "total_wrong_tokens":       len(wrong_positions),
            "denoising_steps_run":      0,
            "parsed_answer":            None,
        }

    for pos in positions_to_mask:
        x[0, pos] = mask_id

    # ── Step 3: iterative denoising on masked positions only ─────────────
    actual_steps = min(steps, tokens_masked)
    base      = tokens_masked // actual_steps
    remainder = tokens_masked % actual_steps
    tokens_per_step = [base + (1 if i < remainder else 0) for i in range(actual_steps)]

    with torch.no_grad():
        for n_transfer in tokens_per_step:
            if n_transfer <= 0:
                continue

            still_masked = [p for p in positions_to_mask if int(x[0, p].item()) == mask_id]
            if not still_masked:
                break

            with torch.autocast(device_type="cuda"):
                step_logits = model(input_ids=x).logits  # (1, seq, vocab)

            probs = F.softmax(step_logits[0], dim=-1)   # (seq, vocab)
            x0    = step_logits[0].argmax(dim=-1)        # (seq,)
            x0_p  = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # (seq,)

            # Rank still-masked positions by confidence
            ranked = sorted(
                [(float(x0_p[p].item()), p, int(x0[p].item())) for p in still_masked],
                reverse=True,
            )
            for _, pos, pred_tok in ranked[:n_transfer]:
                x[0, pos] = pred_tok

    # ── Decode result ────────────────────────────────────────────────────
    result_ids          = x[0].tolist()
    corrected_wrong_ids = [result_ids[p] for p in wrong_positions if p < len(result_ids)]
    corrected_wrong_text = tokenizer.decode(
        corrected_wrong_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    original_wrong_text = tokenizer.decode(
        original_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    combined_for_answer = corrected_wrong_text + ("\n" + remaining if remaining else "")
    parsed_answer = extract_math_answer(combined_for_answer)

    return {
        "wrong_sentence_original":  original_wrong_text,
        "wrong_sentence_corrected": corrected_wrong_text,
        "original_token_ids":       original_ids,
        "tokens_masked":            tokens_masked,
        "total_wrong_tokens":       len(wrong_positions),
        "denoising_steps_run":      actual_steps,
        "parsed_answer":            parsed_answer,
        "input_length":             int(x.shape[1]),
    }


# ---------------------------------------------------------------------------
# Score a parsed answer against the ground truth
# ---------------------------------------------------------------------------

def grade_answer(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False
    return bool(
        _math_scorer.grade(
            _preprocess_math_answer(predicted),
            _preprocess_math_answer(ground_truth),
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reflection correction experiments for LLaDA.")
    parser.add_argument("--model_path",       type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--checkpoint_path",  type=str, default="",
                        help="Path to a LoRA checkpoint directory. Leave empty to use model_path directly (e.g. for Instruct models).")
    parser.add_argument("--eval_results",     type=str, required=True,
                        help="Path to an eval JSON file produced by eval.py (contains 'generations').")
    parser.add_argument("--output_dir",       type=str, default="results/reflection/")
    parser.add_argument("--reflection_token", type=str, default="wait_think",
                        choices=list(REFLECTION_TOKENS.keys()),
                        help="Which reflection phrase to append.")
    parser.add_argument("--custom_reflection_token", type=str, default="",
                        help="Override reflection token with a custom string.")
    parser.add_argument("--split_sentences_from_end", type=int, default=2,
                        help="How many lines before \\boxed to treat as 'first wrong sentence'.")
    parser.add_argument("--max_wrong_examples", type=int, default=20,
                        help="Maximum number of wrong examples to process.")
    parser.add_argument("--exp2_gen_length",  type=int, default=512,
                        help="Generation length (tokens) for Experiment 2.")
    parser.add_argument("--block_length",     type=int, default=32)
    parser.add_argument("--mask_id",          type=int, default=-1)
    parser.add_argument("--run_exp1",         action="store_true", default=True)
    parser.add_argument("--run_exp2",         action="store_true", default=True)
    parser.add_argument("--run_exp3",         action="store_true", default=True)
    parser.add_argument("--run_exp4",         action="store_true", default=True)
    parser.add_argument("--no_exp1",          action="store_true")
    parser.add_argument("--no_exp2",          action="store_true")
    parser.add_argument("--no_exp3",          action="store_true")
    parser.add_argument("--no_exp4",          action="store_true")
    parser.add_argument("--exp4_steps",       type=int, default=50,
                        help="Iterative denoising steps for Experiment 4.")
    args = parser.parse_args()

    if args.no_exp1:
        args.run_exp1 = False
    if args.no_exp2:
        args.run_exp2 = False
    if args.no_exp3:
        args.run_exp3 = False
    if args.no_exp4:
        args.run_exp4 = False

    reflection_token = (
        args.custom_reflection_token
        if args.custom_reflection_token
        else REFLECTION_TOKENS[args.reflection_token]
    )
    print(f"Reflection token: {repr(reflection_token)}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from {args.model_path} ...")
    tokenizer, model = load_eval_model_and_tokenizer(args.model_path, device=device)
    mask_id = resolve_mask_id(tokenizer, configured_mask_id=args.mask_id)
    print(f"mask_id = {mask_id}")

    if args.checkpoint_path:
        print(f"Loading LoRA checkpoint from {args.checkpoint_path} ...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Load eval results and filter wrong examples
    # ------------------------------------------------------------------
    with open(args.eval_results, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    all_generations = eval_data.get("generations", [])
    wrong_examples  = [g for g in all_generations if g.get("correct_include_fallback") is False]
    print(f"Total examples  : {len(all_generations)}")
    print(f"Wrong examples (fallback=False): {len(wrong_examples)}")

    if not wrong_examples:
        print("No wrong examples found. Exiting.")
        return

    wrong_examples = wrong_examples[: args.max_wrong_examples]
    print(f"Processing first {len(wrong_examples)} wrong example(s).")

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    results = []

    for idx, example in enumerate(wrong_examples):
        question      = example.get("question", "")
        ground_truth  = example.get("ground_truth", "")
        prompt_input  = example.get("prompt_input", "")
        generation    = example.get("generations", "")  # already truncated at EOS
        dataset_index = example.get("dataset_index", idx)
        parsed_orig   = example.get("parsed_answer", None)

        print(f"\n{'='*70}")
        print(f"[{idx+1}/{len(wrong_examples)}] dataset_index={dataset_index}")
        print(f"Ground truth : {ground_truth}")
        print(f"Original ans : {parsed_orig}")
        print(f"Generation (first 400 chars):\n{generation[:400]}")

        # ------ split into prefix / wrong_sentence / rest ------
        correct_prefix, wrong_sentence, remaining = find_wrong_boundary(
            generation, sentences_from_end=args.split_sentences_from_end
        )

        if not wrong_sentence.strip():
            print("[warn] wrong_sentence is empty — skipping this example.")
            continue

        print(f"\n--- Split point ---")
        print(f"  correct_prefix (last 200): ...{correct_prefix[-200:]!r}")
        print(f"  wrong_sentence           : {wrong_sentence!r}")
        print(f"  remaining (first 100)    : {remaining[:100]!r}")
        print(f"  reflection_token         : {reflection_token!r}")

        result = {
            "dataset_index":    dataset_index,
            "question":         question,
            "ground_truth":     ground_truth,
            "generation_orig":  generation,
            "parsed_orig":      parsed_orig,
            "correct_prefix":   correct_prefix,
            "wrong_sentence":   wrong_sentence,
            "remaining":        remaining,
            "reflection_token": reflection_token,
            "exp1":             None,
            "exp2":             None,
            "exp3":             None,
            "exp4":             None,
        }

        # ------ Experiment 1: Inplace correction ------
        if args.run_exp1:
            print("\n[Exp 1 — Inplace-correction]")
            try:
                exp1 = run_exp1_inplace(
                    model, tokenizer, mask_id,
                    prompt_text     = prompt_input,
                    correct_prefix  = "\n" + correct_prefix if correct_prefix else "",
                    wrong_sentence  = "\n" + wrong_sentence,
                    reflection_token= reflection_token,
                    device=device,
                )
                result["exp1"] = exp1
                print(f"  Original wrong  : {exp1['wrong_sentence_original']!r}")
                print(f"  Predicted inplace: {exp1['predicted_inplace']!r}")
                print(f"  Tokens changed  : {exp1['tokens_changed']} / {exp1['total_mask_tokens']}")
            except Exception as e:
                print(f"  [ERROR] {e}")
                result["exp1"] = {"error": str(e)}

        # ------ Experiment 2: Forward correction ------
        if args.run_exp2:
            print("\n[Exp 2 — Forward-correction]")
            prefix_incl_wrong = (
                ("\n" + correct_prefix if correct_prefix else "")
                + "\n" + wrong_sentence
            )
            try:
                exp2 = run_exp2_forward(
                    model, tokenizer, mask_id,
                    prompt_text       = prompt_input,
                    prefix_incl_wrong = prefix_incl_wrong,
                    reflection_token  = reflection_token,
                    gen_length        = args.exp2_gen_length,
                    diffusion_steps   = args.exp2_gen_length,
                    block_length      = args.block_length,
                    device=device,
                )
                exp2["correct"] = grade_answer(exp2["parsed_answer"], ground_truth)
                result["exp2"] = exp2
                print(f"  Generated (first 400): {exp2['generated_text'][:400]!r}")
                print(f"  Parsed answer        : {exp2['parsed_answer']!r}")
                print(f"  Correct              : {exp2['correct']}")
            except Exception as e:
                print(f"  [ERROR] {e}")
                result["exp2"] = {"error": str(e)}

        # ------ Experiment 3: Context-aware replace ------
        if args.run_exp3:
            print("\n[Exp 3 — Context-aware replace (same input as Exp 2, forward pass → replace)]")
            try:
                exp3 = run_exp3_context_replace(
                    model, tokenizer, mask_id,
                    prompt_text      = prompt_input,
                    correct_prefix   = "\n" + correct_prefix if correct_prefix else "",
                    wrong_sentence   = "\n" + wrong_sentence,
                    reflection_token = reflection_token,
                    remaining        = remaining,
                    device=device,
                )
                exp3["correct"] = grade_answer(exp3.get("parsed_answer"), ground_truth)
                result["exp3"] = exp3
                print(f"  Original wrong    : {exp3['wrong_sentence_original']!r}")
                print(f"  Corrected sentence: {exp3['wrong_sentence_corrected']!r}")
                print(f"  Tokens changed    : {exp3['tokens_changed']} / {exp3['total_wrong_tokens']}")
                print(f"  Parsed answer     : {exp3.get('parsed_answer')!r}")
                print(f"  Correct           : {exp3['correct']}")
            except Exception as e:
                print(f"  [ERROR] {e}")
                result["exp3"] = {"error": str(e)}

        # ------ Experiment 4: Mask-then-generate ------
        if args.run_exp4:
            print("\n[Exp 4 — Mask-then-generate (forward → mask disagreed → iterative denoise)]")
            try:
                exp4 = run_exp4_mask_then_generate(
                    model, tokenizer, mask_id,
                    prompt_text      = prompt_input,
                    correct_prefix   = "\n" + correct_prefix if correct_prefix else "",
                    wrong_sentence   = "\n" + wrong_sentence,
                    reflection_token = reflection_token,
                    remaining        = remaining,
                    device=device,
                    steps=args.exp4_steps,
                )
                exp4["correct"] = grade_answer(exp4.get("parsed_answer"), ground_truth)
                result["exp4"] = exp4
                print(f"  Original wrong    : {exp4['wrong_sentence_original']!r}")
                print(f"  Corrected sentence: {exp4['wrong_sentence_corrected']!r}")
                print(f"  Tokens masked     : {exp4['tokens_masked']} / {exp4['total_wrong_tokens']}")
                print(f"  Denoise steps     : {exp4.get('denoising_steps_run', 0)}")
                print(f"  Parsed answer     : {exp4.get('parsed_answer')!r}")
                print(f"  Correct           : {exp4['correct']}")
            except Exception as e:
                print(f"  [ERROR] {e}")
                result["exp4"] = {"error": str(e)}

        results.append(result)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"Wrong examples processed: {len(results)}")

    if args.run_exp1:
        inplace_changed = [
            r for r in results
            if r["exp1"] and isinstance(r["exp1"], dict) and r["exp1"].get("tokens_changed", 0) > 0
        ]
        print(f"Exp1 — examples where ≥1 token changed at MASK positions: "
              f"{len(inplace_changed)} / {len(results)}")

    if args.run_exp2:
        forward_correct = [
            r for r in results
            if r["exp2"] and isinstance(r["exp2"], dict) and r["exp2"].get("correct") is True
        ]
        print(f"Exp2 — examples corrected by forward generation: "
              f"{len(forward_correct)} / {len(results)}")

    if args.run_exp3:
        ctx_changed = [
            r for r in results
            if r["exp3"] and isinstance(r["exp3"], dict) and r["exp3"].get("tokens_changed", 0) > 0
        ]
        ctx_correct = [
            r for r in results
            if r["exp3"] and isinstance(r["exp3"], dict) and r["exp3"].get("correct") is True
        ]
        print(f"Exp3 — examples where ≥1 token replaced: "
              f"{len(ctx_changed)} / {len(results)}")
        print(f"Exp3 — examples with correct answer after replace: "
              f"{len(ctx_correct)} / {len(results)}")

    if args.run_exp4:
        exp4_masked = [
            r for r in results
            if r["exp4"] and isinstance(r["exp4"], dict) and r["exp4"].get("tokens_masked", 0) > 0
        ]
        exp4_correct = [
            r for r in results
            if r["exp4"] and isinstance(r["exp4"], dict) and r["exp4"].get("correct") is True
        ]
        print(f"Exp4 — examples where ≥1 token masked+denoised: "
              f"{len(exp4_masked)} / {len(results)}")
        print(f"Exp4 — examples with correct answer after denoise: "
              f"{len(exp4_correct)} / {len(results)}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_label = Path(args.checkpoint_path).name if args.checkpoint_path else Path(args.model_path).name
    out_file = os.path.join(
        args.output_dir,
        f"reflection_{ckpt_label}_{args.reflection_token}.json",
    )
    payload = {
        "checkpoint_path":         args.checkpoint_path,
        "eval_results_source":     args.eval_results,
        "reflection_token":        reflection_token,
        "reflection_token_key":    args.reflection_token,
        "split_sentences_from_end":args.split_sentences_from_end,
        "exp2_gen_length":         args.exp2_gen_length,
        "results":                 results,
        "summary": {
            "wrong_examples_processed": len(results),
            "exp1_tokens_changed_any":  sum(
                1 for r in results
                if r["exp1"] and isinstance(r["exp1"], dict) and r["exp1"].get("tokens_changed", 0) > 0
            ) if args.run_exp1 else None,
            "exp2_correct": sum(
                1 for r in results
                if r["exp2"] and isinstance(r["exp2"], dict) and r["exp2"].get("correct") is True
            ) if args.run_exp2 else None,
            "exp3_tokens_changed_any": sum(
                1 for r in results
                if r["exp3"] and isinstance(r["exp3"], dict) and r["exp3"].get("tokens_changed", 0) > 0
            ) if args.run_exp3 else None,
            "exp3_correct": sum(
                1 for r in results
                if r["exp3"] and isinstance(r["exp3"], dict) and r["exp3"].get("correct") is True
            ) if args.run_exp3 else None,
            "exp4_tokens_masked_any": sum(
                1 for r in results
                if r["exp4"] and isinstance(r["exp4"], dict) and r["exp4"].get("tokens_masked", 0) > 0
            ) if args.run_exp4 else None,
            "exp4_correct": sum(
                1 for r in results
                if r["exp4"] and isinstance(r["exp4"], dict) and r["exp4"].get("correct") is True
            ) if args.run_exp4 else None,
        },
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
