"""Public grouped Gemma RMSNorm with independent backend implementations."""

import math
from typing import Optional

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_grouped_gemma_rmsnorm_triton_fn = None
_grouped_gemma_rmsnorm_torch_fn = None

# Unified Buffer sizing caps the group width, which must also fill whole
# 16-byte BF16 copies.
MAX_GROUP_SIZE = 4096
GROUP_SIZE_ALIGNMENT = 8


def _check_inputs(x, weight, group_size, eps, out):
    """Reject inputs outside the operator contract before any kernel launch.

    All backends share this contract, so it is checked once here instead of
    being duplicated per backend. Only metadata is inspected; no
    device-to-host data read happens.
    """
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("grouped_gemma_rmsnorm requires BF16 x and weight")
    if x.device.type != "npu" or weight.device != x.device:
        raise ValueError("x and weight must be on the same NPU")
    if x.ndim < 2 or weight.ndim != 1:
        raise ValueError("x must have at least 2 dimensions and weight must be 1-D")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")
    if group_size is None:
        width = x.shape[-1]
    elif isinstance(group_size, bool) or not isinstance(group_size, int):
        raise ValueError("group_size must be an integer or None")
    else:
        width = group_size
    if (
        width <= 0
        or width > MAX_GROUP_SIZE
        or width % GROUP_SIZE_ALIGNMENT
        or x.shape[-1] == 0
        or x.shape[-1] % width
    ):
        raise ValueError(
            "group size must be a multiple of 8 up to 4096 and divide x's last dimension"
        )
    slices, remainder = divmod(weight.numel(), width)
    if remainder or slices == 0:
        raise ValueError("weight length must be a positive multiple of the group size")
    # The weight offset cycles every `slices` groups, so a partial period at the
    # end of the tensor would scale its groups by the wrong slice.
    if (x.numel() // width) % slices:
        raise ValueError(
            "the group count must be a whole number of weight periods "
            "(weight length / group size)"
        )
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise ValueError("eps must be a real number")
    if not math.isfinite(eps) or not (1.17549435e-38 <= eps <= 3.4028235e38):
        raise ValueError("eps must be positive and finite")
    if x.requires_grad or weight.requires_grad:
        raise ValueError("grouped_gemma_rmsnorm is inference only")
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise ValueError("out must be a torch.Tensor")
        if out.dtype != torch.bfloat16:
            raise ValueError("out must be BF16")
        if out.device != x.device:
            raise ValueError("out must be on the same NPU as x")
        if out.shape != x.shape:
            raise ValueError("out must have the same shape as x")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out.requires_grad:
            raise ValueError("grouped_gemma_rmsnorm is inference only")
        if tensors_overlap(out, x) or tensors_overlap(out, weight):
            raise ValueError("out must not alias or overlap x or weight")


def tensors_overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Report alias or byte-range overlap, allowing disjoint slices of a pool.

    Shared by the operators that accept a pre-allocated ``out`` buffer; the
    native entry points repeat the same test on the C++ side.
    """
    if a is b:
        return True
    if a.device != b.device or a.numel() == 0 or b.numel() == 0:
        return False

    def _byte_bounds(t: torch.Tensor) -> tuple[int, int]:
        base = t.data_ptr()
        elem = t.element_size()
        if t.is_contiguous():
            return base, base + t.numel() * elem
        min_elem = 0
        max_elem = 0
        for size, stride in zip(t.shape, t.stride()):
            extent = (size - 1) * stride
            if extent >= 0:
                max_elem += extent
            else:
                min_elem += extent
        return base - (-min_elem) * elem, base + (max_elem + 1) * elem

    a_begin, a_end = _byte_bounds(a)
    b_begin, b_end = _byte_bounds(b)
    return a_begin < b_end and b_begin < a_end


def _with_out(result, out):
    """Return ``out`` filled from a backend that allocates its own output."""
    if out is None:
        return result
    out.copy_(result)
    return out


# The Ascend C entry point is compiled by the A3-only operator set of an
# arch22 build, so it is declared for arch22 and takes over auto there. Triton
# keeps auto on arch35 and stays selectable by name everywhere.
@register_impl(
    "norm.grouped_gemma_rmsnorm", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "norm.grouped_gemma_rmsnorm", "ascendc", supported_generations={Generation.ARCH22}
)
def _grouped_gemma_rmsnorm_ascendc(x, weight, group_size, eps, out=None):
    return torch.ops.npu.hyper_conn_grouped_norm(x, weight, group_size, eps, out)


@register_impl(
    "norm.grouped_gemma_rmsnorm",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "norm.grouped_gemma_rmsnorm",
    "triton",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _grouped_gemma_rmsnorm_triton(x, weight, group_size, eps, out=None):
    global _grouped_gemma_rmsnorm_triton_fn
    if _grouped_gemma_rmsnorm_triton_fn is None:
        from ...implementation.triton.norm.grouped_gemma_rmsnorm import (
            grouped_gemma_rmsnorm,
        )

        _grouped_gemma_rmsnorm_triton_fn = grouped_gemma_rmsnorm
    width = x.shape[-1] if group_size is None else group_size
    return _with_out(_grouped_gemma_rmsnorm_triton_fn(x, weight, width, eps), out)


@register_impl(
    "norm.grouped_gemma_rmsnorm",
    "torch_native",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _grouped_gemma_rmsnorm_torch(x, weight, group_size, eps, out=None):
    global _grouped_gemma_rmsnorm_torch_fn
    if _grouped_gemma_rmsnorm_torch_fn is None:
        from ...implementation.torch_native.norm.grouped_gemma_rmsnorm import (
            grouped_gemma_rmsnorm,
        )

        _grouped_gemma_rmsnorm_torch_fn = grouped_gemma_rmsnorm
    width = x.shape[-1] if group_size is None else group_size
    if x.numel() == 0 and out is not None:
        return out
    return _with_out(_grouped_gemma_rmsnorm_torch_fn(x, weight, width, eps), out)


@register_operator("norm.grouped_gemma_rmsnorm")
def grouped_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_size: Optional[int] = None,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    *,
    impl: str = "auto",
) -> torch.Tensor:
    """RMS-normalize fixed-width groups and scale by float32 ``1 + weight``.

    Groups are the consecutive ``group_size``-long spans of ``x`` in row-major
    order, so a hyper-connection residual of shape ``[rows, branches * width]``
    with ``group_size=width`` normalizes each branch of each row separately.
    Group ``g`` scales by ``1 + weight[(g * group_size) % weight.numel() + j]``,
    so a weight holding one slice per branch gives every branch its own slice,
    and a weight of exactly one group is shared by all groups.

    ``impl="auto"`` selects Ascend C on arch22 and Triton on arch35.

    Args:
        x (torch.Tensor): BF16 contiguous input on an NPU with at least two
            dimensions. Its last dimension must be a multiple of group_size.
        weight (torch.Tensor): BF16 contiguous 1-D weight on x's device. Its
            length must be a positive multiple of group_size, and the number of
            groups in x must be a whole number of weight periods, where one
            period is weight.numel() // group_size groups.
        group_size (int | None): Group width; a multiple of 8 up to 4096. None
            uses x's last dimension, making one group per last-dimension span.
            Defaults to None.
        eps (float): Positive finite stability term added to the mean square.
            Defaults to 1e-6.
        out (torch.Tensor | None): Optional pre-allocated BF16 contiguous output
            tensor on x's device with x's shape, not aliasing or overlapping x
            or weight. Defaults to None.
        impl (str): Implementation name: "auto", "ascendc", "triton" or
            "torch_native". Defaults to "auto". Keyword-only.

    Returns:
        torch.Tensor: Normalized BF16 result with x's shape and device. Returns
        ``out`` (written in place) when supplied, or a newly allocated tensor
        otherwise. Inputs are not modified.

    Normalization accumulates in float32 and rounds the result to BF16 once.
    On ``ascendc`` the block-count lookup keys on the group count, so a
    ``[1, S, D]`` input launches the same blocks as ``[S, D]``. The Ascend C
    reduction is grouped differently from the references, so the backends are
    not bitwise equal; see the API documentation for the tolerance.
    """
    _check_inputs(x, weight, group_size, eps, out)
    impl_fn = get_impl_by_name("norm.grouped_gemma_rmsnorm", impl)
    return impl_fn(x, weight, group_size, eps, out)
