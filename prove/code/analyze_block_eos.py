"""
Block-global top-k EOS dominance analysis.

For each of 4 examples, runs block-wise global top-k and logs per-step
token statistics: how many EOS vs content tokens are in the decided set,
when EOS starts to take over, and a full token-trace per block.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from eval import load_eval_model_and_tokenizer, resolve_mask_id


# ── helpers ───────────────────────────────────────────────────────────────────

def _gumbel_noise(logits, temperature=0.0):
    if temperature == 0.0:
        return logits
    noise = -torch.log(-torch.log(torch.clamp(torch.rand_like(logits), 1e-20)))
    return logits + temperature * noise


def _make_schedule(gen_length, steps):
    base, rem = divmod(gen_length, steps)
    sched, total = [], 0
    for t in range(steps):
        total += base + (1 if t < rem else 0)
        sched.append(total)
    return sched


# ── per-block trace ───────────────────────────────────────────────────────────

def trace_block_global_topk(
    model,
    prompt: torch.Tensor,     # (1, prompt_len) – single example
    gen_length: int,
    steps: int,
    mask_id: int,
    eos_ids: set,
    block_length: int = 32,
    log_every: int = 1,       # log every N blocks
):
    """
    Run block-global top-k on a single example, collecting per-block stats.
    Returns:
        x_final : (1, total_len) final token tensor
        block_logs : list of dicts, one per block
    """
    device = prompt.device
    prompt_len = prompt.shape[1]
    total_len = prompt_len + gen_length

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)

    x = torch.full((1, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt

    block_logs = []

    for b in range(num_blocks):
        start = prompt_len + b * block_length
        end   = prompt_len + (b + 1) * block_length

        schedule = _make_schedule(block_length, steps_per_block)

        step_logs = []  # per-step within block
        first_eos_step = None

        for step_idx, num_decided in enumerate(schedule):
            with torch.no_grad():
                logits = model(input_ids=x).logits      # (1, total_len, V)

            block_logits = logits[:, start:end, :]      # (1, block_len, V)
            x0_block = block_logits.argmax(dim=-1)      # (1, block_len)

            p_block = F.softmax(block_logits.float(), dim=-1)
            conf_block = p_block.gather(-1, x0_block.unsqueeze(-1)).squeeze(-1)

            # Build new block
            block_new = torch.full((1, block_length), mask_id, dtype=torch.long, device=device)
            if num_decided > 0:
                k = min(num_decided, block_length)
                top_idx = conf_block.topk(k, dim=-1).indices
                block_new.scatter_(1, top_idx, x0_block.gather(1, top_idx))

            x[:, start:end] = block_new

            # Count EOS in decided slots
            decided_ids = block_new[0].tolist()
            n_decided = sum(1 for t in decided_ids if t != mask_id)
            n_eos     = sum(1 for t in decided_ids if t in eos_ids)
            top_token_id = int(x0_block[0, conf_block[0].argmax()].item())
            top_conf     = float(conf_block[0].max().item())

            if n_eos > 0 and first_eos_step is None:
                first_eos_step = step_idx

            step_logs.append({
                "step":        step_idx,
                "num_decided": n_decided,
                "n_eos":       n_eos,
                "top_token_id": top_token_id,
                "top_conf":    top_conf,
            })

        # Final block contents after all steps
        final_block = x[0, start:end].tolist()
        n_eos_final = sum(1 for t in final_block if t in eos_ids)
        n_mask_final = final_block.count(mask_id)

        block_logs.append({
            "block_idx":     b,
            "start":         start - prompt_len,
            "end":           end   - prompt_len,
            "first_eos_step": first_eos_step,
            "n_eos_final":   n_eos_final,
            "n_mask_final":  n_mask_final,
            "step_logs":     step_logs,
            "final_ids":     final_block,
        })

    return x, block_logs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",     type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--reference_json", type=str, required=True)
    ap.add_argument("--example_indices", type=int, nargs="+", default=[9, 73, 155, 374])
    ap.add_argument("--gen_length",     type=int, default=1024)
    ap.add_argument("--steps",          type=int, default=1024)
    ap.add_argument("--block_length",   type=int, default=32)
    ap.add_argument("--mask_id",        type=int, default=-1)
    ap.add_argument("--output_json",    type=str, default="prove/results/decoding_eos/eos_analysis.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.reference_json) as f:
        ref = json.load(f)
    examples = {g["dataset_index"]: g for g in ref["generations"]}

    print(f"Loading model {args.model_path} ...")
    tokenizer, model = load_eval_model_and_tokenizer(args.model_path, device=device)
    model.eval()

    mask_id = resolve_mask_id(tokenizer, configured_mask_id=args.mask_id)
    # Collect all special / EOS token ids
    eos_ids = set()
    for attr in ["eos_token_id", "bos_token_id", "pad_token_id"]:
        v = getattr(tokenizer, attr, None)
        if v is not None:
            eos_ids.add(v)
    # also add <|endoftext|> explicitly
    eos_ids.add(tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    eos_ids.discard(None)
    print(f"mask_id={mask_id}  eos_ids={eos_ids}")

    results = []

    for idx in args.example_indices:
        if idx not in examples:
            print(f"idx={idx} not in reference, skip")
            continue

        ex = examples[idx]
        print(f"\n{'='*60}")
        print(f"Example idx={idx}  gt={ex['ground_truth']}")

        enc = tokenizer(
            [ex["prompt_input"]],
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        prompt_ids = enc["input_ids"].to(device)

        gen_len = max(args.block_length, (args.gen_length // args.block_length) * args.block_length)

        x_final, block_logs = trace_block_global_topk(
            model=model,
            prompt=prompt_ids,
            gen_length=gen_len,
            steps=args.steps,
            mask_id=mask_id,
            eos_ids=eos_ids,
            block_length=args.block_length,
        )

        # Decode final generation
        gen_ids  = x_final[0, prompt_ids.shape[1]:].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

        # Print per-block summary
        print(f"\n  Block-by-block EOS summary (gen_len={gen_len}, {len(block_logs)} blocks):")
        print(f"  {'blk':>4}  {'pos':>8}  {'first_eos_step':>14}  {'n_eos_final':>12}  {'n_mask':>7}  final_tokens[:8]")
        for bl in block_logs:
            tok_preview = tokenizer.decode(
                [t for t in bl["final_ids"][:8] if t != mask_id],
                skip_special_tokens=False
            )
            eos_marker = " ← EOS starts" if bl["first_eos_step"] == 0 else ""
            print(
                f"  {bl['block_idx']:>4}  pos{bl['start']:>4}-{bl['end']:>4}"
                f"  first_eos@step={str(bl['first_eos_step']):>4}"
                f"  eos_final={bl['n_eos_final']:>3}/{args.block_length}"
                f"  mask={bl['n_mask_final']:>2}"
                f"  {repr(tok_preview[:40])}"
                f"{eos_marker}"
            )

        # Per-step detail for first 6 blocks
        print(f"\n  === Step-by-step detail: first 6 blocks ===")
        for bl in block_logs[:6]:
            b_idx = bl["block_idx"]
            print(f"\n  Block {b_idx} (pos {bl['start']}-{bl['end']}):")
            print(f"  {'step':>5}  {'decided':>8}  {'n_eos':>6}  {'top_conf':>9}  top_token")
            for s in bl["step_logs"]:
                tok = tokenizer.decode([s["top_token_id"]], skip_special_tokens=False)
                is_eos = s["top_token_id"] in eos_ids
                flag = " ← EOS!" if is_eos else ""
                print(f"  {s['step']:>5}  {s['num_decided']:>8}  {s['n_eos']:>6}  {s['top_conf']:>9.4f}  {repr(tok[:20])}{flag}")

        print(f"\n  gen_text[:300]: {repr(gen_text[:300])}")

        results.append({
            "dataset_index": idx,
            "ground_truth":  ex["ground_truth"],
            "gen_text":      gen_text,
            "block_logs":    block_logs,
        })

    import os
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {args.output_json}")


if __name__ == "__main__":
    main()
