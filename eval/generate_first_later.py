import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from generate import (
    _build_block_attention_mask,
    _get_model_attention_mask_dtype,
    _get_single_token_id,
    _mask_stopped_tokens,
    _resolve_stop_token_sequences,
    _select_transfer_indices,
    _update_stop_positions,
    _uses_block_attention,
    add_gumbel_noise,
    get_num_transfer_tokens,
    get_rank,
)


def _process_block(
    model,
    x,
    prompt_index,
    attention_mask,
    tokenizer,
    prompt_length,
    start_idx,
    end_idx,
    steps_per_block,
    temperature,
    cfg_scale,
    remasking,
    mask_id,
    newline_later,
    newline_token_id,
    earlystop,
    stop_positions,
    stop_token_sequences_tensor,
    sequence_positions,
    newline_run_length,
):
    block_positions = sequence_positions[start_idx:end_idx]
    block_mask_index = x[:, start_idx:end_idx] == mask_id
    if earlystop:
        block_mask_index &= block_positions.unsqueeze(0) < stop_positions.unsqueeze(1)

    if not block_mask_index.any():
        return x, stop_positions, False, False

    num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
    should_stop_generation = False

    for step_idx in range(steps_per_block):
        allowed_mask_index = x == mask_id
        if earlystop:
            allowed_mask_index &= sequence_positions.unsqueeze(0) < stop_positions.unsqueeze(1)

        active_rows = allowed_mask_index.any(dim=1)
        if not active_rows.any():
            should_stop_generation = True
            break

        active_x = x[active_rows]
        active_prompt_index = prompt_index[active_rows]
        active_mask_index = allowed_mask_index[active_rows]
        active_attention_mask = attention_mask[active_rows] if attention_mask is not None else None

        if cfg_scale > 0.0:
            un_x = active_x.clone()
            un_x[active_prompt_index] = mask_id
            x_ = torch.cat([active_x, un_x], dim=0)
            attention_mask_ = (
                torch.cat([active_attention_mask, active_attention_mask], dim=0)
                if active_attention_mask is not None
                else None
            )

            logits = model(input_ids=x_, attention_mask=attention_mask_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(input_ids=active_x, attention_mask=active_attention_mask).logits

        logits_with_noise = add_gumbel_noise(logits, temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        if remasking == "low_confidence":
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == "random":
            x0_p = torch.rand(x0.shape, device=x0.device)
        else:
            raise NotImplementedError(remasking)

        # Keep token transfers inside the current block even though other masked
        # blocks may still exist in the sequence.
        x0_p[:, :start_idx] = -np.inf
        x0_p[:, end_idx:] = -np.inf

        x0 = torch.where(active_mask_index, x0, active_x)
        confidence = torch.where(active_mask_index, x0_p, torch.full_like(x0_p, float("-inf")))

        active_row_indices = torch.nonzero(active_rows, as_tuple=False).squeeze(1)
        for active_j, row_idx in enumerate(active_row_indices.tolist()):
            select_indices = _select_transfer_indices(
                confidence_row=confidence[active_j],
                active_mask_row=active_mask_index[active_j],
                predicted_token_row=x0[active_j],
                start_idx=start_idx,
                end_idx=end_idx,
                num_tokens=num_transfer_tokens[row_idx, step_idx].item(),
                newline_token_id=newline_token_id,
                newline_later=newline_later,
            )
            if select_indices is None:
                continue
            x[row_idx, select_indices] = x0[active_j, select_indices]

        if earlystop:
            stop_positions = _update_stop_positions(
                x,
                prompt_length=prompt_length,
                stop_token_sequences=stop_token_sequences_tensor,
                stop_positions=stop_positions,
                tokenizer=tokenizer,
                eot_marker="<|eot_id|>",
                newline_token_id=newline_token_id,
                newline_run_length=newline_run_length,
            )
            x = _mask_stopped_tokens(x, stop_positions=stop_positions, mask_id=mask_id)

    return x, stop_positions, True, should_stop_generation


def _has_remaining_later_block_work(x, start_idx, stop_positions, mask_id, sequence_positions):
    if start_idx >= x.shape[1]:
        return False

    remaining_mask_index = x[:, start_idx:] == mask_id
    remaining_positions = sequence_positions[start_idx:]
    remaining_mask_index &= remaining_positions.unsqueeze(0) < stop_positions.unsqueeze(1)
    return bool(remaining_mask_index.any().item())


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
    newline_later=False,
    earlystop=False,
    newline_run_length=10,
):
    """
    Generate with deferred first-block filling.

    Block order is `1, 2, ..., N-1, 0` when multiple blocks exist, so the
    first generation block is filled last. If early stopping makes later blocks
    unnecessary, the loop still falls through to block 0 before returning.
    """
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", 0)

    use_block_attention = _uses_block_attention(model)
    attention_mask_dtype = _get_model_attention_mask_dtype(model) if use_block_attention else None
    newline_token_id = _get_single_token_id(tokenizer, "\n") if (newline_later or earlystop) else None

    with torch.autocast(device_type="cuda"):
        prompt_length = prompt.shape[1]
        total_length = prompt_length + gen_length
        x = torch.full(
            (prompt.shape[0], total_length), mask_id, dtype=torch.long, device=prompt.device
        )
        x[:, :prompt_length] = prompt.clone()

        prompt_index = x.ne(mask_id) & x.ne(pad_token_id)
        sequence_positions = torch.arange(total_length, device=prompt.device)
        stop_token_sequences = _resolve_stop_token_sequences(tokenizer, pad_token_id) if earlystop else []
        stop_token_sequences_tensor = [
            torch.tensor(token_sequence, device=prompt.device, dtype=torch.long)
            for token_sequence in stop_token_sequences
        ]
        stop_positions = torch.full((prompt.shape[0],), total_length, dtype=torch.long, device=prompt.device)
        attention_mask = (
            _build_block_attention_mask(x, pad_token_id=pad_token_id, dtype=attention_mask_dtype)
            if use_block_attention
            else None
        )

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // max(num_blocks, 1))
        should_stop_generation = False
        non_first_blocks = [] if num_blocks <= 1 else list(range(1, num_blocks))
        for num_block in tqdm(non_first_blocks, disable=(get_rank() != 0)):
            start_idx = prompt_length + num_block * block_length
            end_idx = prompt_length + (num_block + 1) * block_length

            x, stop_positions, _, should_stop_generation = _process_block(
                model=model,
                x=x,
                prompt_index=prompt_index,
                attention_mask=attention_mask,
                tokenizer=tokenizer,
                prompt_length=prompt_length,
                start_idx=start_idx,
                end_idx=end_idx,
                steps_per_block=steps_per_block,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=mask_id,
                newline_later=newline_later,
                newline_token_id=newline_token_id,
                earlystop=earlystop,
                stop_positions=stop_positions,
                stop_token_sequences_tensor=stop_token_sequences_tensor,
                sequence_positions=sequence_positions,
                newline_run_length=newline_run_length,
            )
            if should_stop_generation:
                break

            next_start_idx = end_idx
            if earlystop and not _has_remaining_later_block_work(
                x=x,
                start_idx=next_start_idx,
                stop_positions=stop_positions,
                mask_id=mask_id,
                sequence_positions=sequence_positions,
            ):
                break

        if not should_stop_generation:
            start_idx = prompt_length
            end_idx = prompt_length + block_length
            x, stop_positions, _, should_stop_generation = _process_block(
                model=model,
                x=x,
                prompt_index=prompt_index,
                attention_mask=attention_mask,
                tokenizer=tokenizer,
                prompt_length=prompt_length,
                start_idx=start_idx,
                end_idx=end_idx,
                steps_per_block=steps_per_block,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=mask_id,
                newline_later=newline_later,
                newline_token_id=newline_token_id,
                earlystop=earlystop,
                stop_positions=stop_positions,
                stop_token_sequences_tensor=stop_token_sequences_tensor,
                sequence_positions=sequence_positions,
                newline_run_length=newline_run_length,
            )

        return x
