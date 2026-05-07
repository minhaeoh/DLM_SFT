import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    # Create tensor once and modify in-place
    num_transfer_tokens = base.expand(-1, steps).clone()

    # Handle remainder more efficiently
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1

    return num_transfer_tokens.to(torch.int64)


def _get_model_attention_mask_dtype(model):
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def _uses_block_attention(model):
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    return isinstance(model_type, str) and "llada2" in model_type


def _get_single_token_id(tokenizer, text):
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    return token_ids[0] if len(token_ids) == 1 else None


def _resolve_stop_token_ids(tokenizer, pad_token_id):
    stop_token_ids = set()
    for token_id in (getattr(tokenizer, "eos_token_id", None), pad_token_id):
        if token_id is not None and token_id >= 0:
            stop_token_ids.add(int(token_id))

    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_tokens_to_ids):
        for special_token in ("<|eot_id|>",):
            token_id = convert_tokens_to_ids(special_token)
            if token_id is not None and token_id >= 0 and token_id != getattr(tokenizer, "unk_token_id", None):
                stop_token_ids.add(int(token_id))

    return stop_token_ids


def _build_stop_mask(token_ids, stop_token_ids):
    if stop_token_ids is None or stop_token_ids.numel() == 0:
        return torch.zeros_like(token_ids, dtype=torch.bool)

    stop_mask = token_ids.eq(stop_token_ids[0])
    for token_id in stop_token_ids[1:]:
        stop_mask |= token_ids.eq(token_id)
    return stop_mask


def _update_stop_positions(x, prompt_length, stop_token_ids, stop_positions):
    if stop_token_ids is None or stop_token_ids.numel() == 0:
        return stop_positions

    generated_tokens = x[:, prompt_length:]
    stop_mask = _build_stop_mask(generated_tokens, stop_token_ids)
    has_stop = stop_mask.any(dim=1)
    if not has_stop.any():
        return stop_positions

    first_stop_offsets = stop_mask.to(torch.int64).argmax(dim=1)
    first_stop_positions = prompt_length + first_stop_offsets
    return torch.where(has_stop, torch.minimum(stop_positions, first_stop_positions), stop_positions)


def _build_block_attention_mask(input_ids, pad_token_id, dtype):
    batch_size, seq_len = input_ids.shape
    valid_tokens = input_ids.ne(pad_token_id)
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device))
    query_valid = valid_tokens[:, :, None]
    key_valid = valid_tokens[:, None, :]
    allowed = query_valid & key_valid & causal_mask.unsqueeze(0)

    attention_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=dtype, device=input_ids.device)
    attention_mask = attention_mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return attention_mask


@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """
    Optimized version of the generate function.
    """
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", 0)

    use_block_attention = _uses_block_attention(model)
    attention_mask_dtype = _get_model_attention_mask_dtype(model) if use_block_attention else None
    newline_token_id = _get_single_token_id(tokenizer, "\n")

    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        prompt_length = prompt.shape[1]
        total_length = prompt_length + gen_length
        x = torch.full(
            (prompt.shape[0], total_length), mask_id, dtype=torch.long, device=prompt.device
        )
        x[:, :prompt_length] = prompt.clone()

        prompt_index = x.ne(mask_id) & x.ne(pad_token_id)
        sequence_positions = torch.arange(total_length, device=prompt.device)
        stop_token_ids = _resolve_stop_token_ids(tokenizer, pad_token_id)
        stop_token_ids_tensor = (
            torch.tensor(sorted(stop_token_ids), device=prompt.device, dtype=torch.long) if stop_token_ids else None
        )
        stop_positions = torch.full((prompt.shape[0],), total_length, dtype=torch.long, device=prompt.device)
        attention_mask = (
            _build_block_attention_mask(x, pad_token_id=pad_token_id, dtype=attention_mask_dtype)
            if use_block_attention
            else None
        )

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        should_stop_generation = False
        for num_block in tqdm(range(num_blocks), disable=(get_rank() != 0)):
            start_idx = prompt_length + num_block * block_length
            end_idx = prompt_length + (num_block + 1) * block_length

            block_positions = sequence_positions[start_idx:end_idx]
            block_mask_index = (x[:, start_idx:end_idx] == mask_id) & (
                block_positions.unsqueeze(0) < stop_positions.unsqueeze(1)
            )
            if not block_mask_index.any():
                break
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                allowed_mask_index = (x == mask_id) & (sequence_positions.unsqueeze(0) < stop_positions.unsqueeze(1))
                active_rows = allowed_mask_index.any(dim=1)
                if not active_rows.any():
                    should_stop_generation = True
                    break

                active_x = x[active_rows]
                active_prompt_index = prompt_index[active_rows]
                active_mask_index = allowed_mask_index[active_rows]
                active_attention_mask = attention_mask[active_rows] if attention_mask is not None else None

                # Handle classifier-free guidance more efficiently
                if cfg_scale > 0.0:
                    un_x = active_x.clone()
                    un_x[active_prompt_index] = mask_id
                    x_ = torch.cat([active_x, un_x], dim=0)
                    attention_mask_ = (
                        torch.cat([active_attention_mask, active_attention_mask], dim=0)
                        if active_attention_mask is not None
                        else None
                    )

                    # Get logits in a single forward pass
                    logits = model(input_ids=x_, attention_mask=attention_mask_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(input_ids=active_x, attention_mask=active_attention_mask).logits

                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Ensure we don't process tokens beyond the current block
                x0_p[:, end_idx:] = -np.inf

                # Update masked tokens
                x0 = torch.where(active_mask_index, x0, active_x)
                confidence = torch.where(active_mask_index, x0_p, torch.full_like(x0_p, float("-inf")))

                # Select tokens to transfer based on confidence
                active_row_indices = torch.nonzero(active_rows, as_tuple=False).squeeze(1)
                for active_j, row_idx in enumerate(active_row_indices.tolist()):
                    num_tokens = num_transfer_tokens[row_idx, i].item()
                    if num_tokens > 0:
                        if newline_token_id is None:
                            block_confidence = confidence[active_j, start_idx:end_idx]
                            available_tokens = int(torch.isfinite(block_confidence).sum().item())
                            if available_tokens == 0:
                                continue

                            num_tokens = min(num_tokens, available_tokens)
                            _, block_select_indices = torch.topk(block_confidence, k=num_tokens)
                            select_indices = block_select_indices + start_idx
                        else:
                            block_positions = torch.arange(start_idx, end_idx, device=x.device)
                            block_mask = active_mask_index[active_j, start_idx:end_idx]
                            candidate_indices = block_positions[block_mask]

                            if candidate_indices.numel() == 0:
                                continue

                            candidate_token_ids = x0[active_j, candidate_indices]
                            newline_mask = candidate_token_ids.eq(newline_token_id)
                            normal_candidate_indices = candidate_indices[~newline_mask]
                            newline_candidate_indices = candidate_indices[newline_mask]
                            selected_parts = []

                            # Fill non-newline predictions first; only backfill with "\n" when needed.
                            if normal_candidate_indices.numel() > 0:
                                normal_k = min(num_tokens, normal_candidate_indices.numel())
                                _, normal_order = torch.topk(confidence[active_j, normal_candidate_indices], k=normal_k)
                                selected_parts.append(normal_candidate_indices[normal_order])

                            remaining = num_tokens - sum(part.numel() for part in selected_parts)
                            if remaining > 0 and newline_candidate_indices.numel() > 0:
                                newline_k = min(remaining, newline_candidate_indices.numel())
                                _, newline_order = torch.topk(
                                    confidence[active_j, newline_candidate_indices], k=newline_k
                                )
                                selected_parts.append(newline_candidate_indices[newline_order])

                            if not selected_parts:
                                continue

                            select_indices = torch.cat(selected_parts)
                        x[row_idx, select_indices] = x0[active_j, select_indices]

                stop_positions = _update_stop_positions(
                    x,
                    prompt_length=prompt_length,
                    stop_token_ids=stop_token_ids_tensor,
                    stop_positions=stop_positions,
                )
            if should_stop_generation:
                break
        return x
