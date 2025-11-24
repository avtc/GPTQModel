# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import torch


def normalize_seq_mask(mask: torch.Tensor | None, seq_len: int | None = None) -> torch.Tensor | None:
    """
    Normalize a variety of HF attention mask formats to a boolean keep-mask [B, S].
    True = keep (attended), False = drop (padding/fully-masked).

    Accepts typical HF forms:
      - [B, S] with 1/0
      - [B, 1, 1, S] 'extended' masks with {0 or positive} keep and {negative large} masked
      - [B, 1, S] (rare)
    """
    if mask is None:
        return None

    m = mask
    # Debug logging
    print(f"[DEBUG] normalize_seq_mask input: shape={mask.shape}, dtype={mask.dtype}")
    print(f"  min={mask.min().item()}, max={mask.max().item()}")
    print(f"  unique values (first 10): {torch.unique(mask.flatten()[:1000])[:10].tolist()}")
    
    # Convert numeric to bool 'keep'
    # Detect bias-style masks (0=keep, negative=mask) vs standard masks (positive=keep)
    if m.dtype != torch.bool:
        if mask.max() <= 0:
            # Bias-style mask: 0 is keep, negative is mask
            print(f"[DEBUG] bias-style mask detected (max <= 0), using >= 0")
            m = (m >= 0)
        else:
            # Standard mask: positive is keep
            print(f"[DEBUG] standard mask detected (max > 0), using > 0")
            m = (m > 0)
        print(f"  after conversion: True count={m.sum().item()}, False count={(~m).sum().item()}")

    # Squeeze broadcast dims to reach [B, S]
    if m.dim() == 4:
        # For 4D masks [B, H, S_q, S_k], check if each query position can attend to anything
        # This handles both causal masks and padding masks
        print(f"[DEBUG] 4D mask detected, reducing with .any(dim=-1)")
        m = m.any(dim=-1)  # [B, H, S_q]
        print(f"  after .any(dim=-1): shape={m.shape}, True count={m.sum().item()}")
        if m.size(1) == 1:
            m = m[:, 0, :]  # [B, S]
    elif m.dim() == 3 and m.size(1) == 1:
        m = m[:, 0, :]  # [B, S]
    elif m.dim() == 2:
        pass  # already [B, S]
    else:
        # Fallback: try to flatten to [B, S] if seq_len is known
        if seq_len is not None and m.dim() > 2 and m.size(-1) == seq_len:
            m = m.reshape(m.size(0), -1)[..., :seq_len]
        else:
            raise ValueError(f"Unsupported attention_mask shape: {tuple(mask.shape)}")
            
    print(f"[DEBUG] normalize_seq_mask output: shape={m.shape}, kept={m.sum().item()}/{m.numel()}")
    return m.to(dtype=torch.bool)


def apply_keep_mask_bt(x: torch.Tensor, keep_mask_bs: torch.Tensor | None) -> torch.Tensor:
    """
    Apply [B, S] keep-mask to a tensor x of shape [B, S, ...].
    Returns a flattened tensor of shape [N_kept, ...] (collapses batch/time on the kept rows).
    If keep_mask is None or x doesn't have [B, S, ...] leading dims, returns x unchanged.
    """
    if keep_mask_bs is None or x.dim() < 2:
        return x

    B, S = x.size(0), x.size(1)
    if keep_mask_bs.shape != (B, S):
        raise AssertionError(f"Mask shape {keep_mask_bs.shape} does not match leading dims {(B, S)} of x={tuple(x.shape)}")

    # Concatenate variable-length selections per batch along the sequence axis:
    kept_rows = [x[b, keep_mask_bs[b]] for b in range(B)]
    if len(kept_rows) == 0:
        return x.new_zeros((0,) + x.shape[2:], dtype=x.dtype, device=x.device)
    return torch.cat(kept_rows, dim=0).contiguous()
