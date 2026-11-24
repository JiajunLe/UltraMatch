"""Inference-only fused kernels for packed sparse coarse matching."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without Triton
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_confidence_rowmax_kernel(
        row_probability,
        column_probability,
        candidate_valid,
        output_value,
        output_position,
        candidates: tl.constexpr,
        block_candidates: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_candidates)
        mask = offsets < candidates
        base = row * candidates + offsets
        row_prob = tl.load(row_probability + base, mask=mask, other=0.0)
        column_prob = tl.load(column_probability + base, mask=mask, other=0.0)
        valid = tl.load(candidate_valid + base, mask=mask, other=0).to(tl.int1)
        confidence = tl.where(mask & valid, row_prob * column_prob, 0.0)
        maximum = tl.max(confidence, axis=0)
        position = tl.argmax(confidence, axis=0)
        tl.store(output_value + row, maximum)
        tl.store(output_position + row, position)


def fused_confidence_rowmax(
    row_probability: torch.Tensor,
    column_probability: torch.Tensor,
    candidate_valid: torch.Tensor,
):
    """Multiply sparse dual-softmax factors and reduce each source row.

    The regular PyTorch expression materializes another packed confidence
    tensor before ``max``. The Triton path performs multiplication, masking,
    row maximum and argmax in one pass and writes only two values per row.
    """
    if row_probability.shape != column_probability.shape:
        raise ValueError("dual-softmax factors must have identical shapes")
    if candidate_valid.shape != row_probability.shape:
        raise ValueError("candidate mask must match dual-softmax factors")

    if not row_probability.is_cuda or not _TRITON_AVAILABLE:
        confidence = (
            row_probability * column_probability * candidate_valid
        )
        return confidence.max(dim=-1)

    row_probability = row_probability.contiguous()
    column_probability = column_probability.contiguous()
    candidate_valid = candidate_valid.contiguous()
    candidates = row_probability.shape[-1]
    rows = row_probability.numel() // candidates
    output_value = torch.empty(
        rows, dtype=row_probability.dtype, device=row_probability.device
    )
    output_position = torch.empty(
        rows, dtype=torch.int64, device=row_probability.device
    )
    block_candidates = triton.next_power_of_2(candidates)
    _fused_confidence_rowmax_kernel[(rows,)](
        row_probability,
        column_probability,
        candidate_valid,
        output_value,
        output_position,
        candidates=candidates,
        block_candidates=block_candidates,
        num_warps=4,
    )
    output_shape = row_probability.shape[:-1]
    return output_value.view(output_shape), output_position.view(output_shape)
