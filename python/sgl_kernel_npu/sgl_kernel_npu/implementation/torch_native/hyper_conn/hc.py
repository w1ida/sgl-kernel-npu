"""Independent CPU FP64 HC reference, returned on the input device.

Explicit reference only: host transfers are outside capture. The branch count
and per-branch width come from the operator arguments, so the same reference
serves the production layout and the narrow or eight-branch shapes the public
operator accepts.
"""

import torch

# Bound the FP64 temporaries; each chunk keeps every group's own reduction.
_ROW_CHUNK = 64


def _ref_hc(op, args, hc, hs):
    """CPU FP64, with BF16 rounding only at the public output boundary."""
    tensors = [v.detach().cpu() for v in args if isinstance(v, torch.Tensor)]
    rows = tensors[0].shape[0]
    width = hs if op == "mix" else hc * hs
    result = torch.empty((rows, width), dtype=torch.bfloat16, device="cpu")
    weights = [v.double() for v in (tensors[1:] if op != "combine" else tensors[3:])]
    for start in range(0, rows, _ROW_CHUNK):
        stop = min(rows, start + _ROW_CHUNK)
        x = tensors[0][start:stop].double()
        if op == "mix":
            low = x @ weights[0].T / hc
            hidden = low / (1 + torch.exp(-low))
            gates = 1 / (1 + torch.exp(-(hidden @ weights[1].T)))
            y = (gates.reshape(-1, hc, hs) * x.reshape(-1, hc, hs)).sum(1) / hc
        else:
            residual, normed = [v[start:stop].double() for v in tensors[1:3]]
            logits = normed @ weights[0].T / hc
            gate = 2 / (1 + torch.exp(-logits))
            y = (
                residual.reshape(-1, hc, hs) + gate[:, :, None] * x[:, None, :]
            ).flatten(1)
        result[start:stop] = y.to(torch.bfloat16)
    return result


def mix(x, down, up, hc, hs):
    return _ref_hc("mix", (x, down, up, hc, hs), hc, hs).to(x.device)


def combine(block, residual, normed, weight, hc, hs):
    return _ref_hc("combine", (block, residual, normed, weight, hc, hs), hc, hs).to(
        block.device
    )
