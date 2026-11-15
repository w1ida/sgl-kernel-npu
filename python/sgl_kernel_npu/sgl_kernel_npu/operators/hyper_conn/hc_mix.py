"""Public hyper-connection gated branch mix operator."""

import torch
import torch.nn.functional as F

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_hc_mix_triton_fn = None
_hc_mix_torch_fn = None

# Unified Buffer sizing caps the per-branch hidden size, the low-rank width and
# the branch count.
MAX_HIDDEN_SIZE = 4096
MAX_LOW_RANK = 4096
MAX_HC_COUNT = 8
WIDTH_ALIGNMENT = 16


def _check_inputs(x, down, up, hc, hs):
    """Reject inputs outside the operator contract before any kernel launch.

    All backends share this contract, so it is checked once here instead of
    being duplicated per backend. Only metadata is inspected; no
    device-to-host data read happens.
    """
    if any(t.dtype != torch.bfloat16 for t in (x, down, up)):
        raise ValueError("hc_mix requires BF16 inputs")
    if x.device.type != "npu" or any(t.device != x.device for t in (down, up)):
        raise ValueError("all inputs must be on the same NPU")
    if x.ndim != 2 or down.ndim != 2 or up.ndim != 2:
        raise ValueError("x, down and up must be 2-D")
    if any(not t.is_contiguous() for t in (x, down, up)):
        raise ValueError("x, down and up must be contiguous")
    # The branch count divides both the down-projected activation and the
    # accumulated branch sum, and the kernels multiply by the reciprocal, so
    # only a power of two stays exact.
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
    total_width = hc * hs
    if x.shape[1] != total_width:
        raise ValueError("x must have shape [rows, hc * hs]")
    if down.shape[1] != total_width:
        raise ValueError("down must have shape [low_rank, hc * hs]")
    low_rank = down.shape[0]
    if low_rank <= 0 or low_rank > MAX_LOW_RANK or low_rank % WIDTH_ALIGNMENT:
        raise ValueError("low_rank (down.shape[0]) must be a multiple of 16 up to 4096")
    if up.shape != (total_width, low_rank):
        raise ValueError("up must have shape [hc * hs, low_rank]")
    if any(t.requires_grad for t in (x, down, up)):
        raise ValueError("hc_mix is inference only")


# The native entry points are compiled by the A3-only operator set of an
# arch22 build, so Ascend C is declared for arch22 and takes over auto there.
# Triton keeps auto on arch35 and stays selectable by name everywhere.
@register_impl("hyper_conn.hc_mix", "auto", supported_generations={Generation.ARCH22})
@register_impl(
    "hyper_conn.hc_mix", "ascendc", supported_generations={Generation.ARCH22}
)
def _hc_mix_ascendc(x, down, up, hc, hs):
    if x.shape[0] == 0:
        return x.new_empty((0, hs))
    low = F.linear(x, down)
    activated = torch.ops.npu.hyper_conn_scaled_silu(low, float(hc))
    gate_logits = F.linear(activated, up)
    return torch.ops.npu.hyper_conn_branch_mix(x, gate_logits, hc)


@register_impl("hyper_conn.hc_mix", "auto", supported_generations={Generation.ARCH35})
@register_impl(
    "hyper_conn.hc_mix",
    "triton",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _hc_mix_triton(x, down, up, hc, hs):
    global _hc_mix_triton_fn
    if _hc_mix_triton_fn is None:
        from ...implementation.triton.hyper_conn.hc import mix

        _hc_mix_triton_fn = mix
    return _hc_mix_triton_fn(x, down, up, hc, hs)


@register_impl(
    "hyper_conn.hc_mix",
    "torch_native",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _hc_mix_torch(x, down, up, hc, hs):
    global _hc_mix_torch_fn
    if _hc_mix_torch_fn is None:
        from ...implementation.torch_native.hyper_conn.hc import mix

        _hc_mix_torch_fn = mix
    return _hc_mix_torch_fn(x, down, up, hc, hs)


@register_operator("hyper_conn.hc_mix")
def hc_mix(x, down, up, hc, hs, *, impl="auto"):
    """Compute the low-rank gated average of hyper-connection branches.

    Given a normalized residual ``x`` of shape ``[rows, hc * hs]``, run the
    down projection, divide by ``hc`` and apply SiLU, run the up projection to
    produce per-branch gate logits of shape ``[rows, hc, hs]``, and average the
    sigmoid-gated branches into ``[rows, hs]``:

    ``t = linear(x, down) / hc``

    ``g = linear(bfloat16(t * sigmoid(t)), up)``

    ``out[m, j] = bfloat16(sum_c sigmoid(g[m, c, j]) * x[m, c, j] / hc)``

    ``impl="auto"`` selects Ascend C on arch22 and Triton on arch35.

    Args:
        x (torch.Tensor): BF16 contiguous ``[rows, hc * hs]`` normalized
            residual on an NPU.
        down (torch.Tensor): BF16 contiguous ``[low_rank, hc * hs]``
            down-projection weight on x's device. ``low_rank`` must be a
            multiple of 16 up to 4096.
        up (torch.Tensor): BF16 contiguous ``[hc * hs, low_rank]``
            up-projection weight on x's device. On ``ascendc`` both ND and
            pre-converted ``FRACTAL_NZ`` weights are accepted.
        hc (int): Hyper-connection branch count; a power of two in [1, 8].
        hs (int): Per-branch hidden size; a multiple of 16 up to 4096.
        impl (str): Implementation name: "auto", "ascendc", "triton" or
            "torch_native". Defaults to "auto". Keyword-only.

    Returns:
        torch.Tensor: Newly allocated BF16 result of shape ``[rows, hs]`` on
        x's device. Inputs are not modified.

    On ``ascendc`` the two projections run through
    ``torch.nn.functional.linear`` while the scaled SiLU and the branch
    reduction accumulate in float32 and round to bfloat16 only when writing
    their outputs. The backends evaluate sigmoid with different
    transcendentals, so outputs may differ by up to a BF16 ulp rather than
    being bitwise equal.
    """
    _check_inputs(x, down, up, hc, hs)
    return get_impl_by_name("hyper_conn.hc_mix", impl)(x, down, up, hc, hs)
