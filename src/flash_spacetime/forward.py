"""Triton forward kernel for fused spacetime-bias attention.

Implements the op boundary of `flash_spacetime.spacetime_attention_reference`
as a flash-style kernel: the per-pair feature `R_ij` is never materialised.
The `SpacetimeEncoder` projection is folded out of the kernel entirely:
every use of `R` is a channel contraction, so

- the pre-softmax bias `sum_c qt_c R_ijc` becomes `sum_e u_e E_ije` with the
  host-side GEMM `u = (q * scale) @ W` (the row-constant `qt·b` term shifts
  each softmax row uniformly and is dropped — exact);
- the value-side output `sum_j P_ij R_ij` is accumulated as `A' = sum_j P E_j`
  in-kernel and mapped through `A' @ W^T + b` on the host (`sum_j P = 1` on
  every valid row after flash normalisation).

The kernel therefore only ever builds sin/cos panels of the pair angle
`x_ij = 1024 * clip(sign(I) sqrt(|I|), -4, 4)` in `C_CHUNK`-wide channel
chunks (see `_tiles` for the chunking scheme).

Padding contract: outputs and LSE are written 0 on padding rows; valid rows
match eager exactly (one-sided pairs masked with -inf; the eager both-padding
quirk affects only padding rows and so is not replicated). Heads are padded
in-register to `G_PAD` (next power of two >= H, min 16 for `tl.dot`).

The backward runs the deterministic two-kernel Triton scheme in
`flash_spacetime.backward` (set FLASH_ST_REFERENCE_BWD=1 to fall back to
the exact autograd-through-the-reference path, which materialises `R`).
"""

import os
from typing import Any, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from flash_spacetime.reference import (
    SINEMB_CLIP,
    SINEMB_INPUT_SCALE,
    TIME_SCALE,
    sinusoidal_frequencies,
    spacetime_attention_reference,
)
from flash_spacetime._tiles import (
    _acc3_e_rescaled,
    _acc3_pv,
    _chunk_off,
    _coords4,
    _dot3_e,
    _dot3_qk,
    _e_chunks3,
    _feat_ptrs,
    _load3,
    _load3_scaled,
    _next_pow2,
    _pair_angle,
    _store3,
    _vec_off,
    _zeros3,
)


@triton.jit
def flash_spacetime_fwd_kernel(
    q_ptr, k_ptr, v_ptr,
    u_ptr,  # [B, H, L, C] (u may be dummy when no attn bias)
    feats_ptr,  # [B, L, F_STRIDE]
    seqlen_ptr,  # [B] int32 valid lengths
    o1_ptr, ae_ptr, lse_ptr,  # [B, H, L, C], [B, H, L, C], [B, H, L]
    cu_ptr,  # [B+1] token offsets (PACKED only)
    freq_ptr,  # [F]
    scale,
    L: tl.constexpr,
    H: tl.constexpr,
    FEAT_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    G_PAD: tl.constexpr,
    C_: tl.constexpr,
    C_CHUNK: tl.constexpr,
    F_: tl.constexpr,
    USE_ATTN_BIAS: tl.constexpr,
    USE_ACT_BIAS: tl.constexpr,
    PACKED: tl.constexpr,
    CDTYPE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    TIME_SCALE_C: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
    CLIP: tl.constexpr,
) -> None:
    """One CTA: BLOCK_M query rows of one batch element, all heads."""
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    m0 = pid_m * BLOCK_M

    seqlen = tl.load(seqlen_ptr + b)
    # Whole-block early exit: outputs are zero-initialised on the host, so
    # row blocks entirely inside the padding write nothing at all. Without
    # this, long-L padded batches pay O(L * max_len) dead tile loops.
    if m0 >= seqlen:
        return
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_g = tl.arange(0, G_PAD)
    offs_cc = tl.arange(0, C_CHUNK)
    row_valid = offs_m < seqlen
    head_live = offs_g < H
    if PACKED:
        tok0 = tl.load(cu_ptr + b)
    else:
        tok0 = 0

    # Chunked [M, G, CC] loads; dead heads and padding rows load 0 (their
    # outputs are zero-stored at the end).
    row_off = _chunk_off(b, tok0, offs_m, offs_g, offs_cc, H, L, C_, PACKED)
    row_mask = row_valid[:, None, None] & head_live[None, :, None]
    qt0, qt1, qt2 = _load3_scaled(
        q_ptr, row_off, row_mask, scale, C_, C_CHUNK, CDTYPE
    )
    if USE_ATTN_BIAS:
        u0, u1, u2 = _load3(u_ptr, row_off, row_mask, C_, C_CHUNK, CDTYPE)

    feats_i = _feat_ptrs(feats_ptr, b, tok0, offs_m, FEAT_STRIDE, L, PACKED)
    pix, piy, piz, pit = _coords4(feats_i, row_valid)

    acc0, acc1, acc2 = _zeros3(BLOCK_M, G_PAD, C_CHUNK)
    if USE_ACT_BIAS:
        # Held transposed ([M, CC, G]) to match channel-major E; the
        # epilogue store flips them back once.
        ae0, ae1, ae2 = _zeros3(BLOCK_M, C_CHUNK, G_PAD)
    m_run = tl.full((BLOCK_M, G_PAD), float("-inf"), dtype=tl.float32)
    l_run = tl.zeros((BLOCK_M, G_PAD), dtype=tl.float32)

    # Constexpr loop bound (runtime bounds force the compiler into a
    # shared-memory allocation far past the SM90 budget); the scalar skip
    # discards tiles beyond the event's length.
    for n0 in range(0, L, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        col_valid = offs_n < seqlen
        if n0 < seqlen:
            col_off = _chunk_off(
                b, tok0, offs_n, offs_g, offs_cc, H, L, C_, PACKED
            )
            col_mask = col_valid[:, None, None] & head_live[None, :, None]
            k0, k1, k2 = _load3(k_ptr, col_off, col_mask, C_, C_CHUNK, CDTYPE)

            s = _dot3_qk(
                qt0, qt1, qt2, k0, k1, k2, C_, C_CHUNK, INPUT_PRECISION
            )  # [G, M, N] fp32 accum

            if USE_ATTN_BIAS or USE_ACT_BIAS:
                feats_j = _feat_ptrs(
                    feats_ptr, b, tok0, offs_n, FEAT_STRIDE, L, PACKED
                )
                pjx, pjy, pjz, pjt = _coords4(feats_j, col_valid)
                x = _pair_angle(
                    pix, piy, piz, pit, pjx, pjy, pjz, pjt,
                    TIME_SCALE_C, INPUT_SCALE, CLIP,
                )  # [M, N]
                e0, e1, e2 = _e_chunks3(x, freq_ptr, C_, C_CHUNK, F_, CDTYPE)

            if USE_ATTN_BIAS:
                # sum_e u_e E_e, batched over rows; [M,G,N] -> [G,M,N].
                s += tl.trans(
                    _dot3_e(
                        u0, u1, u2, e0, e1, e2, C_, C_CHUNK, INPUT_PRECISION
                    ),
                    1, 0, 2,
                )

            # One-sided masking only: this CTA's valid rows never see the
            # both-padding case, and padding rows are zero-stored.
            s = tl.where(col_valid[None, None, :], s, float("-inf"))
            s = tl.trans(s, 1, 0, 2)  # [M, G, N]

            m_new = tl.maximum(m_run, tl.max(s, axis=2))
            # Rows with no valid key never occur (diagonal lemma); still,
            # guard exp against -inf - -inf on fully-dead padded rows.
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, :, None])
            p = tl.where(col_valid[None, None, :], p, 0.0)
            alpha = tl.where(
                m_run == float("-inf"), 0.0, tl.exp(m_run - m_safe)
            )
            l_run = l_run * alpha + tl.sum(p, axis=2)
            m_run = m_new
            pb = p.to(CDTYPE)
            pbt = tl.trans(pb, 1, 0, 2)  # [G, M, N]

            v0, v1, v2 = _load3(v_ptr, col_off, col_mask, C_, C_CHUNK, CDTYPE)
            acc0, acc1, acc2 = _acc3_pv(
                acc0, acc1, acc2, alpha, pbt, v0, v1, v2,
                C_, C_CHUNK, INPUT_PRECISION,
            )
            if USE_ACT_BIAS:
                pbt_c = tl.trans(pb, 0, 2, 1)  # [M, N, G]
                ae0, ae1, ae2 = _acc3_e_rescaled(
                    ae0, ae1, ae2, alpha, e0, e1, e2, pbt_c,
                    C_, C_CHUNK, INPUT_PRECISION,
                )

    l_safe = tl.where(l_run == 0.0, 1.0, l_run)
    inv_l = 1.0 / l_safe[:, :, None]
    _store3(
        o1_ptr, row_off, acc0 * inv_l, acc1 * inv_l, acc2 * inv_l,
        row_mask, C_, C_CHUNK,
    )
    if USE_ACT_BIAS:
        _store3(
            ae_ptr, row_off,
            tl.trans(ae0, 0, 2, 1) * inv_l,
            tl.trans(ae1, 0, 2, 1) * inv_l,
            tl.trans(ae2, 0, 2, 1) * inv_l,
            row_mask, C_, C_CHUNK,
        )
    lse = m_run + tl.log(l_safe)
    lse_off = _vec_off(b, tok0, offs_m, offs_g, H, L, PACKED)
    tl.store(
        lse_ptr + lse_off, lse, mask=row_valid[:, None] & head_live[None, :]
    )


def flash_spacetime_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    feats: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    seqlens: Tensor,
    scale: Optional[float] = None,
    use_attn_bias: bool = True,
    use_activation_bias: bool = True,
    block_m: int = 16,
    block_n: Optional[int] = None,
    num_warps: int = 8,
    num_stages: int = 1,
    cu_seqlens: Optional[Tensor] = None,
) -> Tuple[Tensor, ...]:
    """Fused forward. Returns (O [B,H,L,D] with pad rows zeroed, LSE).

    Inputs follow `spacetime_attention_reference`; `seqlens` is the [B]
    int tensor of valid lengths (front-packed padding assumed). With
    `cu_seqlens` ([B+1] token offsets) the tensors are instead PACKED:
    q/k/v [T, H, D], feats [T, F], outputs [T, H, D] — no padding exists
    anywhere and lengths are unbounded (`seqlens` is ignored).
    """
    packed = cu_seqlens is not None
    if cu_seqlens is not None:
        cu = cu_seqlens.to(torch.int32).contiguous()
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        batch = int(seqlens.numel())
        _, heads, dim = q.shape
        # The kernel's loop bound is constexpr; bucketize to the next power
        # of two so at most a handful of specializations ever compile.
        length = _next_pow2(int(seqlens.max()), floor=16)
    else:
        batch, heads, length, dim = q.shape
    c = weight.shape[1]
    if c != dim:
        raise ValueError(f"C (={c}) must equal head_dim (={dim})")
    if c not in (16, 32, 48):
        # Channels are processed in 16-wide chunks because Triton block
        # shapes must be powers of two and the widths in use are not; the
        # second and third chunk compile away when C does not reach them.
        raise ValueError(f"kernel supports C in (16, 32, 48), got {c}")
    scale_value = dim**-0.5 if scale is None else scale

    # fp32 inputs compute in true fp32 (ieee dots) to honour the
    # 2x-eager-error contract; bf16 inputs use bf16 tensor-core math. fp32
    # operands double the tile staging, so the key-block shrinks.
    compute_bf16 = q.dtype != torch.float32
    if block_n is None:
        block_n = 32 if compute_bf16 else 16
    qc, kc, vc = (t.contiguous() for t in (q, k, v))
    featsc = feats[..., :4].to(torch.float32).contiguous()
    freqs = sinusoidal_frequencies(c, q.device)
    seq32 = seqlens.to(torch.int32).contiguous()

    if use_attn_bias:
        u = (
            ((qc.to(torch.float32) * scale_value) @ weight.to(torch.float32))
            .to(qc.dtype)
            .contiguous()
        )
    else:
        u = qc  # dummy pointer, never read

    o1 = torch.zeros_like(qc)
    if packed:
        ae = torch.zeros_like(qc) if use_activation_bias else qc
        lse = torch.zeros(
            qc.shape[0], heads, device=q.device, dtype=torch.float32
        )
    else:
        ae = (
            torch.zeros(
                batch, heads, length, c, device=q.device, dtype=q.dtype
            )
            if use_activation_bias
            else qc  # dummy pointer, never written
        )
        lse = torch.zeros(
            batch, heads, length, device=q.device, dtype=torch.float32
        )

    grid = (triton.cdiv(length, block_m), batch)
    flash_spacetime_fwd_kernel[grid](
        qc, kc, vc, u, featsc, seq32, o1, ae, lse,
        cu if packed else seq32,
        freqs,
        scale_value,
        L=length,
        H=heads,
        FEAT_STRIDE=4,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        G_PAD=_next_pow2(heads),
        C_=c,
        C_CHUNK=16,
        F_=c // 2,
        USE_ATTN_BIAS=use_attn_bias,
        USE_ACT_BIAS=use_activation_bias,
        PACKED=packed,
        CDTYPE=tl.bfloat16 if compute_bf16 else tl.float32,
        INPUT_PRECISION="ieee",
        TIME_SCALE_C=TIME_SCALE,
        INPUT_SCALE=SINEMB_INPUT_SCALE,
        CLIP=SINEMB_CLIP,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    out = o1
    if use_activation_bias:
        o2 = ae.to(torch.float32) @ weight.to(torch.float32).t()
        if bias is not None:
            o2 = o2 + bias.to(torch.float32)
        if packed:
            # Every packed row is a real token; nothing to re-zero.
            out = out + o2.to(out.dtype)
        else:
            # Pad rows must stay exactly zero after the +bias broadcast.
            idx = torch.arange(length, device=q.device)
            valid = (idx.unsqueeze(0) < seqlens.unsqueeze(1))[:, None, :, None]
            out = out + (o2 * valid).to(out.dtype)
    return out, lse


class _FlashSpacetimeAttention(torch.autograd.Function):
    """Fused forward and deterministic Triton backward.

    FLASH_ST_REFERENCE_BWD=1 selects autograd through the eager
    reference instead (exact but materialises R) — the debugging A/B for
    the Triton backward.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        feats: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        seqlens: Tensor,
        scale: Optional[float],
        use_attn_bias: bool,
        use_activation_bias: bool,
        cu_seqlens: Optional[Tensor] = None,
    ) -> Tensor:
        if feats.requires_grad:
            raise ValueError("feats (detector data) must not require grad")
        out, lse = flash_spacetime_forward(
            q, k, v, feats, weight, bias, seqlens,
            scale=scale,
            use_attn_bias=use_attn_bias,
            use_activation_bias=use_activation_bias,
            cu_seqlens=cu_seqlens,
        )
        # The backward forms its own softmax row sums; the output is not
        # needed.
        ctx.save_for_backward(
            q, k, v, feats, weight,
            bias if bias is not None else q.new_empty(0),
            seqlens, lse,
            cu_seqlens if cu_seqlens is not None else q.new_empty(0),
        )
        ctx.has_bias = bias is not None
        ctx.packed = cu_seqlens is not None
        ctx.scale = scale
        ctx.flags = (use_attn_bias, use_activation_bias)
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_out: Tensor
    ) -> Tuple[Optional[Tensor], ...]:
        q, k, v, feats, weight, bias_t, seqlens, lse, cu_t = ctx.saved_tensors
        bias = bias_t if ctx.has_bias else None
        cu_seqlens = cu_t if ctx.packed else None
        use_attn_bias, use_activation_bias = ctx.flags
        if ctx.packed and os.environ.get("FLASH_ST_REFERENCE_BWD") == "1":
            raise RuntimeError(
                "the reference backward supports the padded layout only"
            )
        if os.environ.get("FLASH_ST_REFERENCE_BWD", "0") != "1":
            from flash_spacetime.backward import flash_spacetime_backward

            dq, dk, dv, dw, db = flash_spacetime_backward(
                q, k, v, feats, weight, bias, seqlens, lse, grad_out,
                scale=ctx.scale,
                use_attn_bias=use_attn_bias,
                use_activation_bias=use_activation_bias,
                cu_seqlens=cu_seqlens,
            )
            # With both biases off the projection never enters the graph;
            # autograd's convention for unused parameters is None, not zeros.
            proj_used = use_attn_bias or use_activation_bias
            return (
                dq if q.requires_grad else None,
                dk if k.requires_grad else None,
                dv if v.requires_grad else None,
                None,
                dw if (weight.requires_grad and proj_used) else None,
                (
                    db
                    if (bias is not None and bias.requires_grad and proj_used)
                    else None
                ),
                None, None, None, None, None,
            )
        length = q.shape[2]
        idx = torch.arange(length, device=q.device)
        valid = idx.unsqueeze(0) < seqlens.unsqueeze(1)
        mask = torch.zeros(valid.shape, dtype=torch.float32, device=q.device)
        mask[~valid] = float("-inf")

        # No autocast is active inside a backward, so the eager reference
        # would mix the saved bf16 activations with fp32 parameters; running
        # it under autocast in the saved dtype makes it compute what the
        # eager module computes under the trainer's own autocast.
        with torch.enable_grad(), torch.autocast(
            q.device.type, dtype=q.dtype, enabled=q.dtype != torch.float32
        ):
            qd, kd, vd = (t.detach().requires_grad_(True) for t in (q, k, v))
            wd = weight.detach().requires_grad_(True)
            bd = (
                bias.detach().requires_grad_(True)
                if bias is not None
                else None
            )
            out = spacetime_attention_reference(
                qd, kd, vd, feats.detach(), wd, bd,
                key_padding_mask=mask,
                scale=ctx.scale,
                use_attn_bias=use_attn_bias,
                use_activation_bias=use_activation_bias,
            )
            # The kernel zeroes pad rows; the reference leaves eager garbage
            # there. Blank the upstream grad at pad rows so both agree (the
            # contract guarantees it is zero there anyway).
            g = grad_out * valid[:, None, :, None]
            inputs = [qd, kd, vd, wd] + ([bd] if bd is not None else [])
            grads = torch.autograd.grad(
                out, inputs, grad_outputs=g, allow_unused=True
            )
        dq, dk, dv, dw = grads[:4]
        db = grads[4] if bias is not None else None
        return (
            dq if q.requires_grad else None,
            dk if k.requires_grad else None,
            dv if v.requires_grad else None,
            None,
            dw if weight.requires_grad else None,
            db if (bias is not None and bias.requires_grad) else None,
            None, None, None, None, None,
        )


def flash_spacetime_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    feats: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    seqlens: Tensor,
    scale: Optional[float] = None,
    use_attn_bias: bool = True,
    use_activation_bias: bool = True,
) -> Tensor:
    """Fused spacetime-bias attention (public op).

    Semantics of `spacetime_attention_reference` on valid rows; padding
    rows are zeroed. See the module docstring for the padding contract.
    """
    return _FlashSpacetimeAttention.apply(
        q, k, v, feats, weight, bias, seqlens,
        scale, use_attn_bias, use_activation_bias, None,
    )


def flash_spacetime_attention_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    feats: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    cu_seqlens: Tensor,
    scale: Optional[float] = None,
    use_attn_bias: bool = True,
    use_activation_bias: bool = True,
) -> Tensor:
    """Packed (uncapped) fused spacetime-bias attention.

    q/k/v are [T, H, D] and feats [T, F] with events delimited by
    `cu_seqlens` ([B+1] token offsets, cu[0] = 0, cu[-1] = T) — the layout
    of a jagged NestedTensor's values buffer. No padding exists anywhere,
    so event lengths are unbounded.
    """
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
    return _FlashSpacetimeAttention.apply(
        q, k, v, feats, weight, bias, seqlens,
        scale, use_attn_bias, use_activation_bias, cu_seqlens,
    )
