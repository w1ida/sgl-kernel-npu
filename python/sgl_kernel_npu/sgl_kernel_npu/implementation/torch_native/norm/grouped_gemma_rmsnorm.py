"""Independent CPU FP64 reference for grouped Gemma RMSNorm.

The number of weight slices is derived from the weight length, so one shared
weight slice, one slice per branch and any group width are all handled; x may
have more than two dimensions and keeps its shape.
"""

import torch

# Bound the FP64 temporaries. Chunking runs over whole weight periods, which
# the public operator guarantees divide the group count, so a chunk never
# straddles a period boundary.
_PERIOD_CHUNK = 64


def grouped_gemma_rmsnorm(x, weight, group_size, eps):
    width = x.shape[-1] if group_size is None else group_size
    slices = weight.numel() // width
    scale = 1 + weight.detach().cpu().double().reshape(slices, width)
    periods = x.detach().cpu().reshape(-1, slices, width)
    result = torch.empty_like(periods)
    for start in range(0, periods.shape[0], _PERIOD_CHUNK):
        stop = min(periods.shape[0], start + _PERIOD_CHUNK)
        grouped = periods[start:stop].double()
        square_mean = (grouped * grouped).sum(-1, keepdim=True) / width
        normalized = grouped / torch.sqrt(square_mean + eps)
        result[start:stop] = (normalized * scale).to(torch.bfloat16)
    return result.reshape(x.shape).to(x.device)
