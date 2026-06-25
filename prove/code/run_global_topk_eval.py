"""
Remasking decoding experiments for masked diffusion LMs.

Modes:
  global        — At each step, keep top-k tokens across ALL gen positions globally.
                  Suffers from EOS-token dominance (tokens fill with <|endoftext|>).

  block_global  — Block-wise left-to-right (same as standard), but within each block
                  the selection is over ALL block positions (re-masking allowed).
                  At step t within block b: keep top-num_decided(t) from the full block,
                  re-mask the rest. Previously decided tokens in the block can be evicted.
"""
import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from eval import (
    extract_math_answer,
    extract_math_answer_strict,
    _math_scorer,
    _preprocess_math_answer,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits
    noise = -torch.log(-torch.log(torch.clamp(torch.rand_like(logits), min=1e-20)))
    return logits + temperature * noise


def _make_schedule(gen_length: int, steps: int):
    """Cumulative number of decided tokens at each step (same distribution as generate.py)."""
    base = gen_length // steps
    remainder = gen_length % steps
    schedule, total = [], 0
    for t in range(steps):
        total += base + (1 if t < remainder else 0)
        schedule.append(total)
    return schedule


def generate_global_topk(
    model,
    prompt: torch.Tensor,       # (B, prompt_len)
    gen_length: int,
    steps: int,
    temperature: float,
    mask_id: int,
) -> torch.Tensor:
    """
    Global top-k remasking generation.

    At step t: run forward on current x (with num_decided(t-1) unmasked tokens),
    compute confidence for ALL gen positions, keep top-num_decided(t) globally,
    re-mask everything else. Previously decided tokens can be evicted.

    Returns x of shape (B, prompt_len + gen_length).
    """
    B, prompt_len = prompt.shape
    total_len = prompt_len + gen_length
    device = prompt.device

    # Start: all generation positions are MASK
    x = torch.full((B, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt

    schedule = _make_schedule(gen_length, steps)

    for num_decided in tqdm(schedule, desc="global-topk", leave=False):
        with torch.no_grad():
            logits = model(input_ids=x).logits          # (B, total_len, V)

        logits_noisy = _add_gumbel_noise(logits, temperature)
        x0 = logits_noisy.argmax(dim=-1)               # (B, total_len)

        p = F.softmax(logits.float(), dim=-1)
        confidence = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)  # (B, total_len)

        # Prompt positions must not be selected
        confidence[:, :prompt_len] = float("-inf")

        # Build new x: all gen positions re-masked, then fill top-num_decided
        x_new = torch.full((B, total_len), mask_id, dtype=torch.long, device=device)
        x_new[:, :prompt_len] = prompt

        if num_decided > 0:
            # topk across all gen+prompt positions; prompt already set to -inf
            top_idx = confidence.topk(num_decided, dim=-1).indices  # (B, num_decided)
            x_new.scatter_(1, top_idx, x0.gather(1, top_idx))

        x = x_new

    return x


def generate_block_global_topk(
    model,
    prompt: torch.Tensor,   # (B, prompt_len)
    gen_length: int,
    steps: int,
    temperature: float,
    mask_id: int,
    block_length: int = 32,
) -> torch.Tensor:
    """
    Block-wise global top-k remasking.

    Processes blocks left-to-right (same ordering as standard generate.py).
    Within each block: at each step t, re-evaluate ALL block positions and keep only
    the top-num_decided(t) — previously decided tokens can be re-masked if a
    different set of positions has higher confidence.

    This avoids the EOS-dominance failure of the fully-global variant by anchoring
    to the block structure while still allowing intra-block remasking.
    """
    B, prompt_len = prompt.shape
    total_len = prompt_len + gen_length
    device = prompt.device

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)

    # Start: all generation positions are MASK
    x = torch.full((B, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt

    for block_idx in tqdm(range(num_blocks), desc="blocks", leave=False):
        start = prompt_len + block_idx * block_length
        end   = prompt_len + (block_idx + 1) * block_length

        # Cumulative schedule for this block: how many positions to keep at each step
        block_schedule = _make_schedule(block_length, steps_per_block)

        for num_decided in block_schedule:
            with torch.no_grad():
                logits = model(input_ids=x).logits      # (B, total_len, V)

            # Focus only on current block positions
            block_logits = logits[:, start:end, :]      # (B, block_length, V)

            block_logits_noisy = _add_gumbel_noise(block_logits, temperature)
            x0_block = block_logits_noisy.argmax(dim=-1)  # (B, block_length)

            p_block = F.softmax(block_logits.float(), dim=-1)
            conf_block = torch.gather(
                p_block, -1, x0_block.unsqueeze(-1)
            ).squeeze(-1)                               # (B, block_length)

            # Re-build this block: start all-MASK, fill top-num_decided
            block_new = torch.full(
                (B, block_length), mask_id, dtype=torch.long, device=device
            )
            if num_decided > 0:
                top_idx = conf_block.topk(
                    min(num_decided, block_length), dim=-1
                ).indices                               # (B, num_decided)
                block_new.scatter_(1, top_idx, x0_block.gather(1, top_idx))

            x[:, start:end] = block_new

    return x


def generate_block_global_topk_eos_fix(
    model,
    prompt: torch.Tensor,   # (B, prompt_len)
    gen_length: int,
    steps: int,
    temperature: float,
    mask_id: int,
    eos_id: int,
    block_length: int = 32,
) -> torch.Tensor:
    """
    Block-global top-k with EOS-fix mode switch.

    Within each block, run the same global top-k re-ranking.
    The moment any position is decided as EOS (for a given batch item):
      1. All currently decided (non-MASK) positions in that block are fixed (locked).
      2. Remaining steps in the block only fill MASK positions (standard mask-only).
      3. All subsequent blocks also use standard mask-only filling.
    No early stopping — generation continues through all blocks/steps.

    This prevents the "re-masking cascade" while still allowing re-masking up to
    the point where the model commits to EOS for the first time.
    """
    B, prompt_len = prompt.shape
    total_len = prompt_len + gen_length
    device = prompt.device

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)

    x = torch.full((B, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt

    # fixed[b, p] = True → position p is permanently decided for batch item b
    fixed = torch.zeros(B, total_len, dtype=torch.bool, device=device)
    fixed[:, :prompt_len] = True

    # global_mode[b] = True → still using block-global re-masking for item b
    global_mode = torch.ones(B, dtype=torch.bool, device=device)

    for block_idx in tqdm(range(num_blocks), desc="blocks", leave=False):
        start = prompt_len + block_idx * block_length
        end   = prompt_len + (block_idx + 1) * block_length

        block_schedule = _make_schedule(block_length, steps_per_block)

        for num_decided in block_schedule:
            with torch.no_grad():
                logits = model(input_ids=x).logits          # (B, total_len, V)

            block_logits = logits[:, start:end, :]          # (B, block_len, V)
            block_logits_noisy = _add_gumbel_noise(block_logits, temperature)
            x0_block = block_logits_noisy.argmax(dim=-1)    # (B, block_len)
            p_block = F.softmax(block_logits.float(), dim=-1)
            conf_block = p_block.gather(-1, x0_block.unsqueeze(-1)).squeeze(-1)

            block_fixed = fixed[:, start:end]               # (B, block_len) - view

            block_new = x[:, start:end].clone()

            for b in range(B):
                b_fixed = block_fixed[b]                    # (block_len,) bool
                b_conf  = conf_block[b].clone()
                b_x0    = x0_block[b]

                if global_mode[b]:
                    # ── Block-global: re-mask non-fixed, fill top-k ──────────
                    block_new[b, ~b_fixed] = mask_id        # re-mask non-fixed
                    b_conf[b_fixed] = float("-inf")         # exclude fixed from selection

                    n_fixed = int(b_fixed.sum().item())
                    k = min(max(0, num_decided - n_fixed),
                            int((~b_fixed).sum().item()))
                    if k > 0:
                        top_idx = b_conf.topk(k).indices
                        block_new[b, top_idx] = b_x0[top_idx]

                    # Check if EOS now appears in any decided position
                    newly_decided = block_new[b] != mask_id
                    if (block_new[b][newly_decided] == eos_id).any():
                        # Switch to standard mode: lock all decided positions
                        global_mode[b] = False
                        fixed[b, start:end] |= newly_decided
                else:
                    # ── Standard mask-only: only fill MASK positions ─────────
                    mask_pos = block_new[b] == mask_id      # (block_len,)
                    n_fixed  = int(b_fixed.sum().item())
                    k = min(max(0, num_decided - n_fixed),
                            int(mask_pos.sum().item()))
                    if k > 0:
                        b_conf[~mask_pos] = float("-inf")   # only from MASK
                        top_idx = b_conf.topk(k).indices
                        block_new[b, top_idx] = b_x0[top_idx]

                    # Lock newly decided positions
                    newly_decided = block_new[b] != mask_id
                    fixed[b, start:end] |= newly_decided & ~b_fixed

            x[:, start:end] = block_new

        # End of block: lock everything decided in this block
        fixed[:, start:end] |= (x[:, start:end] != mask_id)

    return x


# ── scoring ───────────────────────────────────────────────────────────────────

def extract_and_grade(generation_text: str, ground_truth: str):
    """Returns (parsed_strict, parsed_fb, correct_strict, correct_fb)."""
    parsed_strict = extract_math_answer_strict(generation_text)
    correct_strict = bool(parsed_strict and _math_scorer.grade(
        _preprocess_math_answer(parsed_strict), _preprocess_math_answer(ground_truth)
    ))

    parsed_fb = extract_math_answer(generation_text)
    correct_fb = bool(parsed_fb and _math_scorer.grade(
        _preprocess_math_answer(parsed_fb), _preprocess_math_answer(ground_truth)
    ))
    return parsed_strict, parsed_fb, correct_strict, correct_fb


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",    type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--reference_json", type=str, required=True,
                    help="Eval JSON from the standard run (to reuse same problems + prompts).")
    ap.add_argument("--gen_length",   type=int, default=1024)
    ap.add_argument("--steps",        type=int, default=1024,
                    help="Denoising steps. steps=gen_length → exactly 1 new token decided per step.")
    ap.add_argument("--temperature",  type=float, default=0.0)
    ap.add_argument("--batch_size",   type=int, default=4)
    ap.add_argument("--output_dir",   type=str, default="prove/results/decoding_eos")
    ap.add_argument("--mask_id",      type=int, default=-1)
    ap.add_argument("--mode",         type=str, default="block_global",
                    choices=["global", "block_global", "eos_fix"],
                    help="global: all gen positions remaskable (EOS dominance). "
                         "block_global: remask within each block. "
                         "eos_fix: block-global until first EOS, then standard mask-only (no early stop).")
    ap.add_argument("--block_length", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load reference eval to get same problems & prompts ────────────────────
    with open(args.reference_json) as f:
        ref_data = json.load(f)
    examples = ref_data["generations"]
    print(f"Loaded {len(examples)} problems from reference eval.")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model {args.model_path} ...")
    from eval import load_eval_model_and_tokenizer, resolve_mask_id
    tokenizer, model = load_eval_model_and_tokenizer(args.model_path, device=device)
    model.eval()

    mask_id = resolve_mask_id(tokenizer, configured_mask_id=args.mask_id)
    print(f"mask_id = {mask_id}")

    # ── Batch-process ─────────────────────────────────────────────────────────
    results = []
    n = len(examples)
    for batch_start in tqdm(range(0, n, args.batch_size), desc="batches"):
        batch = examples[batch_start : batch_start + args.batch_size]

        # Tokenise prompts (reuse pre-built prompt strings from reference eval)
        prompt_strs = [ex["prompt_input"] for ex in batch]
        enc = tokenizer(
            prompt_strs,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        prompt_ids = enc["input_ids"].to(device)    # (B, max_prompt_len)

        # Align gen_length to 32 for cleanliness (block_length compat)
        gen_len = max(32, (args.gen_length // 32) * 32)

        # Generate
        if args.mode == "eos_fix":
            eos_id = getattr(tokenizer, "eos_token_id", None) or mask_id
            out = generate_block_global_topk_eos_fix(
                model=model,
                prompt=prompt_ids,
                gen_length=gen_len,
                steps=args.steps,
                temperature=args.temperature,
                mask_id=mask_id,
                eos_id=eos_id,
                block_length=args.block_length,
            )
        elif args.mode == "block_global":
            out = generate_block_global_topk(
                model=model,
                prompt=prompt_ids,
                gen_length=gen_len,
                steps=args.steps,
                temperature=args.temperature,
                mask_id=mask_id,
                block_length=args.block_length,
            )
        else:
            out = generate_global_topk(
                model=model,
                prompt=prompt_ids,
                gen_length=gen_len,
                steps=args.steps,
                temperature=args.temperature,
                mask_id=mask_id,
            )

        # Decode only the generated portion
        prompt_len = prompt_ids.shape[1]
        for i, ex in enumerate(batch):
            gen_ids = out[i, prompt_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

            parsed_s, parsed_fb, correct, correct_fb = extract_and_grade(gen_text, ex["ground_truth"])

            results.append({
                "dataset_index":              ex["dataset_index"],
                "question":                   ex["question"],
                "ground_truth":               ex["ground_truth"],
                "generations":                gen_text,
                "parsed_answer":              parsed_s,
                "parsed_answer_include_fallback": parsed_fb,
                "correct":                    correct,
                "correct_include_fallback":   correct_fb,
                "ref_correct":                ex.get("correct"),
                "ref_correct_fallback":       ex.get("correct_include_fallback"),
            })

            tqdm.write(
                f"  idx={ex['dataset_index']:3d}  gt={str(ex['ground_truth'])[:15]:15s}"
                f"  parsed={str(parsed_fb)[:15]:15s}  ok={correct}  ok_fb={correct_fb}"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    n_correct    = sum(1 for r in results if r["correct"])
    n_correct_fb = sum(1 for r in results if r["correct_include_fallback"])
    n_ref        = sum(1 for r in results if r["ref_correct_fallback"])

    print(f"\n=== {args.mode} remasking (steps={args.steps}, gen_length={gen_len}, block={args.block_length}) ===")
    print(f"  acc_strict  : {n_correct}/{n} = {n_correct/n*100:.1f}%")
    print(f"  acc_fallback: {n_correct_fb}/{n} = {n_correct_fb/n*100:.1f}%")
    print(f"  ref_fallback: {n_ref}/{n} = {n_ref/n*100:.1f}%  (standard decoding)")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"{args.mode}_{Path(args.model_path).name}_gl{gen_len}_s{args.steps}.json",
    )
    payload = {
        "model_path": args.model_path,
        "gen_length": gen_len,
        "steps":      args.steps,
        "temperature": args.temperature,
        "generations": results,
        "metrics": {
            "acc_strict":   n_correct / n,
            "acc_fallback": n_correct_fb / n,
            "ref_fallback": n_ref / n,
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
