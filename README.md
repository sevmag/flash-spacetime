# flash-spacetime

Fused attention for models that bias query-key pairs by the **spacetime
interval** between them.

Standard attention scores a pair by `q . k`. Neutrino-telescope transformers
(IceMix / DeepIce and descendants) add a term built from the signed
four-distance between two pulses:

    score_ij = q_i . k_j + <q_i, R_ij>,     R_ij = E(x_i, x_j) W^T + b

where `E` sinusoidally embeds the interval. Written eagerly, `R` is an
`[B, L, L, C]` tensor -- quadratic in sequence length *and* carrying a channel
axis. That tensor, not the attention, is what makes the mechanism expensive.

These kernels never build it. `E` is recomputed inside each tile from the
coordinates and `W` is applied on the host in fp32, so the pair term costs
registers instead of memory.

## Numbers

Batch 256 x 768 pulses, C=16, H200:

| | step | peak memory |
|---|---|---|
| eager | 389 ms | 63 GiB |
| fused | **287 ms** | **9.8 GiB** |

1.36x faster, 6.4x smaller. The memory figure is the point: it is what makes
long or uncapped sequences affordable at all.

Accuracy is unchanged where it matters — a full direction-reconstruction
training run with the fused kernel converges to the same place as the eager
one (2.4602 deg against 2.4989 / 2.4673 / 2.4465 for three eager variants of
the same model, all within the run-to-run noise of that setup).

## Use

```python
from flash_spacetime import flash_spacetime_attention

out = flash_spacetime_attention(q, k, v, pos, w, b, key_padding_mask=mask)
```

`flash_spacetime_attention_varlen` takes packed sequences with offsets and
skips padding entirely.

`triton` is an optional dependency: the package imports on a CPU-only machine
and only pulls the toolchain in when a kernel entry point is first touched.
`flash_spacetime.reference` holds the pure-PyTorch definition, which is the
specification the tests validate against rather than a fallback path.

## A warning worth reading before modifying the backward

The softmax backward subtracts a row sum `D` from `dp`. Eager autograd forms
`D` from the same `dp` it then subtracts, so `rowsum(ds)` is identically zero
by construction. An earlier version of this kernel computed `D` from a
separately-rounded quantity instead. The residual was tiny, the forward was
exact in fp32, and at frozen weights the kernel's bf16 gradients were *more*
accurate than eager's — every parity test passed.

It still broke training. The residual had a **fixed direction**, so it
compounded over thousands of Adam steps: a model trained with it tracked the
eager one for eight epochs, then collapsed. A second, subtler version of the
same inconsistency survived a first fix and resurfaced eighteen epochs later.

The property that matters is **self-consistency, not accuracy**. An fp64
ablation settled it: feeding the backward the *exact* `D` made things worse,
while forming it from the same `p` and `dp` that build `ds` cut the gradient
error thirtyfold. The kernel now computes `D = sum(p * dp) / sum(p)` in-kernel.
Note the normalisation — at trained weights the logits can reach O(1e4), the
recomputed `p` of a row then carries a common factor, and an unnormalised
`sum(p * dp)` inherits it.

Two testing lessons came with it, both generic:

- Validate at **trained** checkpoints against an **fp64** truth, never against
  eager in reduced precision. Eager-bf16 was itself 28% off at the weights
  where this kernel measured 6.6%; comparing the two implied a kernel error
  that did not exist.
- Random inputs with random targets cannot see a defect that a trajectory
  integrates. The same probe read R^2 = 0.000 on synthetic targets and 0.32 on
  real ones, because the systematic part averages away when the readout is
  random.

## Hardware

The tile sizes are chosen for **H200-class shared memory** (~228 KB per SM).
On a smaller budget the kernels raise
`triton.runtime.errors.OutOfResources: out of resource: shared memory` at
launch rather than falling back — an A100 MIG slice
(`nvidia_a100_3g.20gb`), for instance, fails the wider channel
configurations while the narrow ones still pass. Making the block sizes
adapt to `torch.cuda.get_device_properties().shared_memory_per_block` is not
done.

## Tests

    pytest tests/          # needs an H200-class CUDA GPU and triton

`test_gpu.py` covers forward and backward against the fp64 reference across
channel widths and dtypes; `test_varlen.py` covers packed sequences including
single-pulse and ragged-boundary events.

## Origin and downstream use

Developed inside [sevmag/graphnet](https://github.com/sevmag/graphnet) on the
[`transformers-flash`](https://github.com/sevmag/graphnet/tree/transformers-flash)
branch, where it is reached as `DeepIce(rel_attention="flash")` through
`Block_rel.forward_flash`. That branch still carries its own copy of these
files; this repository is the canonical one going forward.

Integration tests that need the model -- `test_flash_spacetime.py` and
`test_flash_spacetime_integration.py` -- stay in graphnet, since they import
`DeepIce`, `Attention_rel` and `SpacetimeEncoder`. The two suites here are
the ones that depend on nothing but this package.
