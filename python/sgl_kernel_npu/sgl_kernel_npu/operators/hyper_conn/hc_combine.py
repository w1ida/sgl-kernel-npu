"""Public hyper-connection gated residual combine operator."""

from typing import Optional

import torch
import torch.nn.functional as F

from ..norm.grouped_gemma_rmsnorm import tensors_overlap
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_hc_combine_triton_fn = None
_hc_combine_torch_fn = None

# Unified Buffer sizing caps the per-branch hidden size and the branch count.
MAX_HIDDEN_SIZE = 4096
MAX_HC_COUNT = 8
WIDTH_ALIGNMENT = 16


def _check_inputs(block, residual, normed, weight, hc, hs, out):
    """Reject inputs outside the operator contract before any kernel launch.

    All backends share this contract, so it is checked once here instead of
    being duplicated per backend. Only metadata is inspected; no
    device-to-host data read happens.
    """
    tensors = (block, residual, normed, weight)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise ValueError("hc_combine requires BF16 inputs")
    if residual.device.type != "npu" or any(
        t.device != residual.device for t in tensors
    ):
        raise ValueError("all inputs must be on the same NPU")
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("block, residual, normed and weight must be 2-D")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("block, residual, normed and weight must be contiguous")
    # The branch count divides the projected injection logit, and the kernel
    # multiplies by the reciprocal, so only a power of two stays exact.
    if (
        isinstance(hc, bool)
        or not isinstance(hc, int)
        or hc <= 0
        or hc > MAX_HC_COUNT
        or hc & (hc - 1)
    ):
        raise ValueError("hc must be a power of two in [1, 8]")
    if (
        isinstance(hs, bool)
        or not isinstance(hs, int)
        or hs <= 0
        or hs > MAX_HIDDEN_SIZE
        or hs % WIDTH_ALIGNMENT
    ):
        raise ValueError("hs must be a multiple of 16 up to 4096")
    rows = residual.shape[0]
    total_width = hc * hs
    if residual.shape[1] != total_width:
        raise ValueError("residual must have shape [rows, hc * hs]")
    if normed.shape != (rows, total_width):
        raise ValueError("normed must have shape [rows, hc * hs]")
    if block.shape != (rows, hs):
        raise ValueError("block must have shape [rows, hs]")
    if weight.shape != (hc, total_width):
        raise ValueError("weight must have shape [hc, hc * hs]")
    if any(t.requires_grad for t in tensors):
        raise ValueError("hc_combine is inference only")
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise ValueError("out must be a torch.Tensor")
        if out.dtype != torch.bfloat16:
            raise ValueError("out must be BF16")
        if out.device != residual.device:
            raise ValueError("out must be on the same NPU as residual")
        if out.shape != residual.shape:
            raise ValueError("out must have the same shape as residual")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out.requires_grad:
            raise ValueError("hc_combine is inference only")
        if any(tensors_overlap(out, t) for t in tensors):
            raise ValueError("out must not alias or overlap any input tensor")


def _with_out(result, out):
    """Return ``out`` filled from a backend that allocates its own output."""
    if out is None:
        return result
    out.copy_(result)
    return out


# The native entry point is compiled by the A3-only operator set of an arch22
# build, so Ascend C is declared for arch22 and takes over auto there. Triton
# keeps auto on arch35 and stays selectable by name everywhere.
@register_impl(
    "hyper_conn.hc_combine", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "hyper_conn.hc_combine", "ascendc", supported_generations={Generation.ARCH22}
)
def _hc_combine_ascendc(block, residual, normed, weight, hc, hs, out=None):
    if residual.shape[0] == 0:
        return out if out is not None else torch.empty_like(residual)
    gate_logits = F.linear(normed, weight)
    return torch.ops.npu.hyper_conn_branch_combine(
        block, residual, gate_logits, hc, out
    )


@register_impl(
    "hyper_conn.hc_combine", "auto", supported_generations={Generation.ARCH35}
)
@register_impl(
    "hyper_conn.hc_combine",
    "triton",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _hc_combine_triton(block, residual, normed, weight, hc, hs, out=None):
    global _hc_combine_triton_fn
    if _hc_combine_triton_fn is None:
        from ...implementation.triton.hyper_conn.hc import combine

        _hc_combine_triton_fn = combine
    return _with_out(
        _hc_combine_triton_fn(block, residual, normed, weight, hc, hs), out
    )


@register_impl(
    "hyper_conn.hc_combine",
    "torch_native",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _hc_combine_torch(block, residual, normed, weight, hc, hs, out=None):
    global _hc_combine_torch_fn
    if _hc_combine_torch_fn is None:
        from ...implementation.torch_native.hyper_conn.hc import combine

        _hc_combine_torch_fn = combine
    if residual.shape[0] == 0 and out is not None:
        return out
    return _with_out(_hc_combine_torch_fn(block, residual, normed, weight, hc, hs), out)


@register_operator("hyper_conn.hc_combine")
def hc_combine(
    block: torch.Tensor,
    residual: torch.Tensor,
    normed: torch.Tensor,
    weight: torch.Tensor,
    hc: int,
    hs: int,
    out: Optional[torch.Tensor] = None,
    *,
    impl: str = "auto",
) -> torch.Tensor:
    """Project per-branch gates and inject ``block`` into ``residual``.

    One gate per ``(row, branch)`` is projected from ``normed`` and ``weight``,
    scaled by ``2 * sigmoid(logit / hc)``, and used to scale ``block`` onto
    every branch of ``residual``:

    ``z = linear(normed, weight)``

    ``g[m, c] = 2 * sigmoid(z[m, c] / hc)``

    ``out[m, c, j] = bfloat16(residual[m, c, j] + block[m, j] * g[m, c])``

    ``impl="auto"`` selects Ascend C on arch22 and Triton on arch35.

    Args:
        block (torch.Tensor): BF16 contiguous ``[rows, hs]`` sub-block output
            on an NPU.
        residual (torch.Tensor): BF16 contiguous ``[rows, hc * hs]`` residual
            on block's device.
        normed (torch.Tensor): BF16 contiguous ``[rows, hc * hs]`` normalized
            residual on block's device.
        weight (torch.Tensor): BF16 contiguous ``[hc, hc * hs]`` injection
            projection weight on block's device.
        hc (int): Hyper-connection branch count; a power of two in [1, 8].
        hs (int): Per-branch hidden size; a multiple of 16 up to 4096.
        out (torch.Tensor | None): Optional pre-allocated BF16 contiguous
            output tensor on residual's device with residual's shape, not
            aliasing or overlapping any input tensor. Defaults to None.
        impl (str): Implementation name: "auto", "ascendc", "triton" or
            "torch_native". Defaults to "auto". Keyword-only.

    Returns:
        torch.Tensor: Combined BF16 result with residual's shape and device.
        Returns ``out`` (written in place) when supplied, or a newly allocated
        tensor otherwise. Inputs are not modified; the residual is not updated
        in place.

    On ``ascendc`` the gate projection runs through
    ``torch.nn.functional.linear`` while the combine stage evaluates the gate,
    the branch scaling and the residual addition in float32 and rounds to
    bfloat16 once when writing the output. The backends evaluate sigmoid with
    different transcendentals, so outputs may differ by up to a BF16 ulp
    rather than being bitwise equal.
    """
    _check_inputs(block, residual, normed, weight, hc, hs, out)
    return get_impl_by_name("hyper_conn.hc_combine", impl)(
        block, residual, normed, weight, hc, hs, out
    )
