"""Fused attention with a learned spacetime bias on the query-key pairs.

Standard attention scores a pair by `q . k`. Neutrino-telescope models add a
term built from the *four-distance* between the two pulses, so the score is

    q . k  +  <q, R_ij>        with R_ij = E(x_i, x_j) W^T + b

where `E` is a sinusoidal embedding of the signed spacetime interval. Written
eagerly that `R` is an `[B, L, L, C]` tensor -- quadratic in sequence length
*and* carrying a channel axis -- which is what makes the mechanism expensive
rather than the attention itself.

These kernels never materialise it. `E` is rebuilt inside each tile from the
coordinates, and `W` is applied on the host in fp32, so the pair term costs
registers rather than memory. At batch 256 x 768 pulses with C=16 that is
287 ms/step and 9.8 GiB against an eager 389 ms and 63 GiB.

`reference.py` holds the pure-PyTorch definition every test validates
against; it is the specification, not a fallback.
"""

from .reference import (  # noqa: F401
    SINEMB_CLIP,
    SINEMB_INPUT_SCALE,
    SINEMB_N_FREQ,
    TIME_SCALE,
    attention_rel_oracle_inputs,
    float_padding_mask,
    merge_heads,
    pair_mask_bias,
    sinusoidal_frequencies,
    spacetime_attention_reference,
    spacetime_pair_features,
    split_heads,
    valid_row_mask,
)

__all__ = [
    "flash_spacetime_attention",
    "flash_spacetime_attention_varlen",
    "spacetime_attention_reference",
    "spacetime_pair_features",
    "sinusoidal_frequencies",
    "attention_rel_oracle_inputs",
    "pair_mask_bias",
    "float_padding_mask",
    "valid_row_mask",
    "split_heads",
    "merge_heads",
    "TIME_SCALE",
    "SINEMB_INPUT_SCALE",
    "SINEMB_CLIP",
    "SINEMB_N_FREQ",
]


def __getattr__(name: str) -> object:
    """Import the Triton entry points on first use.

    Importing `triton` pulls in a GPU toolchain, so the package stays
    importable on a CPU-only machine for anything that only needs the
    reference implementation or the constants.
    """
    if name in ("flash_spacetime_attention", "flash_spacetime_attention_varlen"):
        from . import forward

        return getattr(forward, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
