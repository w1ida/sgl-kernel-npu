# hc_mix

## Hardware support

| impl | arch22 A3 | arch22 A2 | arch35 |
| --- | --- | --- | --- |
| `ascendc` / `auto` on arch22 | Supported | Supported | Not implemented |
| `triton` / `auto` on arch35 | Supported | Supported | Supported |
| `torch_native` | Supported | Supported | Supported |

arch22 is split because the two members of that family were measured
separately: A3 means the Ascend910_93 series, A2 the Ascend910B series. The
Ascend C kernels are built by `SGL_KERNEL_ENABLE_A3_ONLY_OPS`, which covers
every SoC outside the Ascend950 series, and their block counts are clamped to
the runtime vector core count. An Ascend950 build omits them, so `auto` falls
to Triton there.

## Signature

`auto` → `ascendc` (arch22) / `triton` (arch35).

```python
from sgl_kernel_npu.operators.hyper_conn import hc_mix

def hc_mix(x, down, up, hc, hs, *, impl="auto"):
    ...
```

## Description

HC sigmoid-gated mixing. Contiguous BF16 x[R,10240], down[320,10240], up[10240,320], hc=4, hs=2560. Returns BF16[R,2560], using FP32 projections and SiLU(x@down.T/4). auto selects Ascend C on arch22 and the Triton/native hybrid on arch35.

All tensors are on the same NPU. Inputs are not modified; zero rows return empty output.

Let `h = SiLU(x @ down.T / 4)` and `g = sigmoid(h @ up.T)`. Reshape x and g to `[R,4,2560]`; output `sum(g*x, dim=1)/4`. Intermediates are FP32; only the output rounds to BF16.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Constraints |
| --- | --- | --- | --- | --- |
| `x` | Tensor | [R,10240], BF16 | required | Contiguous NPU input. |
| `down` | Tensor | [320,10240], BF16 | required | Contiguous on x.device. |
| `up` | Tensor | [10240,320], BF16 | required | Contiguous on x.device. |
| `hc / hs` | int | — | required | 4 / 2560. |
| `impl` | str | — | "auto" | Keyword-only: auto, triton, torch_native. |

| Return | Type | Shape | Dtype | Device / layout |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,2560] | BF16 | Input device; contiguous, newly allocated. |

## triton

Default implementation. Checks input metadata before launch. Supports graph capture and replay after in-place input updates.

## torch_native

Independent CPU FP64 reference; returns results to the input device. Use outside graph capture. Select explicitly with `impl="torch_native"`.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import hc_mix

x = torch.randn(1, 10240, device="npu", dtype=torch.bfloat16)
down = torch.zeros(320, 10240, device="npu", dtype=x.dtype)
up = torch.zeros(10240, 320, device="npu", dtype=x.dtype)
y = hc_mix(x, down, up, 4, 2560, impl="auto")
```
