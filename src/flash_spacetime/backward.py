"""Triton backward kernels for fused spacetime-bias attention.

Deterministic two-kernel scheme with the projection folded to the host on
both sides, mirroring the forward:

- host prologue: `u = (q·scale) @ W`, `w~ = dO @ W`, `dOβ = dO·β` (all
  deterministic GEMMs/reductions);
- row-owned kernel: a first pass over the keys forms the softmax row sum
  `D_i = Σ_j P_ij dP_ij / Σ_j P_ij` from the same P and dP that the second
  pass turns into dS; the second pass writes the dS@k part of dq plus the per-row
  E-accumulators `H = Σ_j dS·E` and `G = Σ_j P·E` and `σ = Σ_j dS`, and
  `D` itself for the column kernel;
- column-owned kernel: recomputes S, P, dS tile-by-tile with that `D` and
  writes dk, dv (exclusive rows, fixed loop order — no atomics);
- host epilogue: `dq = scale·(dqp + [A](H @ W^T + σ⊗β))`,
  `dW = [A] Σ qt⊗H + [V] Σ dO⊗G` (two GEMMs over flattened rows),
  `dβ = [V] Σ_valid dO` (the logit path vanishes analytically).

Bitwise run-to-run determinism is structural: every buffer has exactly one
writer and every reduction is either an in-kernel fixed-order loop or a
cuBLAS call.

`D` is formed in-kernel rather than taken from the forward output through
the identity `D_i = dO_i·O_i`. The identity holds for the exact softmax,
but the forward accumulates with weights rounded to the compute dtype and
the backward recomputes P from the LSE, so a `D` read off the output is
consistent with neither the P it multiplies nor the dP it is subtracted
from, and `Σ_j dS_ij` does not vanish: every rounding on either side leaks
into a per-row term of one sign, which in bf16 training walks the
attention toward one-hot rows. Formed from the very P and dP that form dS,
the row sum of dS is zero identically, whatever the forward stored, and
the backward needs no saved output at all.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from flash_spacetime.reference import (
    SINEMB_CLIP,
    SINEMB_INPUT_SCALE,
    TIME_SCALE,
    sinusoidal_frequencies,
)
from flash_spacetime._tiles import (
    _acc3_dotT,
    _acc3_e,
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
def _bwd_row_operands(
    q_ptr, do_ptr, u_ptr, wt_ptr, lse_ptr, dob_ptr, feats_ptr,
    b, tok0, offs_m, offs_g, offs_cc, seqlen, scale,
    H: tl.constexpr,
    L: tl.constexpr,
    FEAT_STRIDE: tl.constexpr,
    C_: tl.constexpr,
    C_CHUNK: tl.constexpr,
    USE_ATTN_BIAS: tl.constexpr,
    USE_ACT_BIAS: tl.constexpr,
    PACKED: tl.constexpr,
    CDTYPE: tl.constexpr,
) -> tuple:
    """Everything both backward kernels need for one block of query rows:

    offsets and masks, the qt/dO/u/w~ chunk triples (u and w~ falling back
    to qt aliases when their bias term is off — never read then), the
    saved LSE and dO·β, and the rows' interval coordinates. Padding rows
    and dead heads load 0.
    """
    row_valid = offs_m < seqlen
    head_live = offs_g < H
    row_off = _chunk_off(b, tok0, offs_m, offs_g, offs_cc, H, L, C_, PACKED)
    row_mask = row_valid[:, None, None] & head_live[None, :, None]
    qt0, qt1, qt2 = _load3_scaled(
        q_ptr, row_off, row_mask, scale, C_, C_CHUNK, CDTYPE
    )
    do0, do1, do2 = _load3(do_ptr, row_off, row_mask, C_, C_CHUNK, CDTYPE)
    if USE_ATTN_BIAS:
        u0, u1, u2 = _load3(u_ptr, row_off, row_mask, C_, C_CHUNK, CDTYPE)
    else:
        u0, u1, u2 = qt0, qt1, qt2
    if USE_ACT_BIAS:
        wt0, wt1, wt2 = _load3(wt_ptr, row_off, row_mask, C_, C_CHUNK, CDTYPE)
    else:
        wt0, wt1, wt2 = qt0, qt1, qt2
    row_vec = _vec_off(b, tok0, offs_m, offs_g, H, L, PACKED)
    row_vec_mask = row_valid[:, None] & head_live[None, :]
    lse = tl.load(lse_ptr + row_vec, mask=row_vec_mask, other=0.0)
    if USE_ACT_BIAS:
        dob = tl.load(dob_ptr + row_vec, mask=row_vec_mask, other=0.0)
    else:
        dob = lse * 0.0
    feats_i = _feat_ptrs(feats_ptr, b, tok0, offs_m, FEAT_STRIDE, L, PACKED)
    pix, piy, piz, pit = _coords4(feats_i, row_valid)
    return (
        row_off, row_mask, row_vec, row_vec_mask,
        qt0, qt1, qt2, do0, do1, do2, u0, u1, u2, wt0, wt1, wt2,
        lse, dob, pix, piy, piz, pit,
    )


@triton.jit
def _recompute_p_dpr(
    qt0, qt1, qt2,  # [M, G, CC]
    u0, u1, u2,
    do0, do1, do2,
    k0, k1, k2,  # [N, G, CC]
    v0, v1, v2,
    e0, e1, e2,  # [M, CC, N]
    wt0, wt1, wt2,  # w~ chunks [M, G, CC]
    dob,  # [M, G] dO·β
    lse,  # [M, G]
    col_valid,  # [N]
    C_: tl.constexpr,
    C_CHUNK: tl.constexpr,
    USE_ATTN_BIAS: tl.constexpr,
    USE_ACT_BIAS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
) -> tuple:
    """P and raw dP for one (M, N) tile, in [M, G, N] layout (fp32).

    P is exactly zero at masked columns; dP there is whatever the masked
    (zero) operands give and must not be read on its own.

    Annotated with the builtin rather than `typing.Tuple`: Triton's
    frontend parses the return annotation of a jit'd function and
    rejects `typing` constructs.
    """
    s = _dot3_qk(qt0, qt1, qt2, k0, k1, k2, C_, C_CHUNK, INPUT_PRECISION)
    if USE_ATTN_BIAS:
        s += tl.trans(
            _dot3_e(u0, u1, u2, e0, e1, e2, C_, C_CHUNK, INPUT_PRECISION),
            1, 0, 2,
        )
    s = tl.trans(s, 1, 0, 2)  # [M, G, N]

    # P from the saved LSE; exact zeros at masked columns.
    p = tl.exp(s - lse[:, :, None])
    p = tl.where(col_valid[None, None, :], p, 0.0)

    # dPraw = dO·v^T (+ [V] (w~·E + dO·β)).
    dpr = tl.trans(
        _dot3_qk(do0, do1, do2, v0, v1, v2, C_, C_CHUNK, INPUT_PRECISION),
        1, 0, 2,
    )  # [M, G, N]
    if USE_ACT_BIAS:
        dpr += (
            _dot3_e(wt0, wt1, wt2, e0, e1, e2, C_, C_CHUNK, INPUT_PRECISION)
            + dob[:, :, None]
        )
    return p, dpr


@triton.jit
def _col_tiles(
    k_ptr, v_ptr, freq_ptr,
    col_off,  # [N, G, CC] element offsets of the key block
    col_mask,  # [N, G, 1]
    feats_j,  # [N] feature-row pointers of the key block
    col_valid,  # [N]
    pix, piy, piz, pit,  # [M] query positions
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    C_: tl.constexpr,
    C_CHUNK: tl.constexpr,
    F_: tl.constexpr,
    USE_ATTN_BIAS: tl.constexpr,
    USE_ACT_BIAS: tl.constexpr,
    CDTYPE: tl.constexpr,
    TIME_SCALE_C: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
    CLIP: tl.constexpr,
) -> tuple:
    """K, v and pair-feature chunks of one key block for a row block."""
    k0, k1, k2 = _load3(k_ptr, col_off, col_mask, C_, C_CHUNK, CDTYPE)
    v0, v1, v2 = _load3(v_ptr, col_off, col_mask, C_, C_CHUNK, CDTYPE)
    if USE_ATTN_BIAS or USE_ACT_BIAS:
        pjx, pjy, pjz, pjt = _coords4(feats_j, col_valid)
        x = _pair_angle(
            pix, piy, piz, pit, pjx, pjy, pjz, pjt,
            TIME_SCALE_C, INPUT_SCALE, CLIP,
        )
        e0, e1, e2 = _e_chunks3(x, freq_ptr, C_, C_CHUNK, F_, CDTYPE)
    else:
        e0 = tl.zeros((BLOCK_M, C_CHUNK, BLOCK_N), dtype=CDTYPE)
        e1 = e0
        e2 = e0
    return k0, k1, k2, v0, v1, v2, e0, e1, e2


@triton.jit
def flash_spacetime_bwd_cols_kernel(
    q_ptr, k_ptr, v_ptr, u_ptr, do_ptr, wt_ptr,
    feats_ptr, seqlen_ptr, lse_ptr,
    d_ptr,  # [B, H, L] softmax row sums from the row kernel
    dob_ptr, dk_ptr, dv_ptr,
    cu_ptr,  # [B+1] token offsets (PACKED only)
    freq_ptr,
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
    """Column-owned backward: one CTA owns BLOCK_N key rows; writes dk, dv."""
    pid_n = tl.program_id(0)
    b = tl.program_id(1)
    n0 = pid_n * BLOCK_N

    seqlen = tl.load(seqlen_ptr + b)
    if n0 >= seqlen:
        return
    if PACKED:
        tok0 = tl.load(cu_ptr + b)
    else:
        tok0 = 0
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_g = tl.arange(0, G_PAD)
    offs_cc = tl.arange(0, C_CHUNK)
    col_valid = offs_n < seqlen
    head_live = offs_g < H

    col_off = _chunk_off(b, tok0, offs_n, offs_g, offs_cc, H, L, C_, PACKED)
    col_mask = col_valid[:, None, None] & head_live[None, :, None]
    k0, k1, k2 = _load3(k_ptr, col_off, col_mask, C_, C_CHUNK, CDTYPE)
    v0, v1, v2 = _load3(v_ptr, col_off, col_mask, C_, C_CHUNK, CDTYPE)
    feats_j = _feat_ptrs(feats_ptr, b, tok0, offs_n, FEAT_STRIDE, L, PACKED)
    pjx, pjy, pjz, pjt = _coords4(feats_j, col_valid)

    dk0, dk1, dk2 = _zeros3(BLOCK_N, G_PAD, C_CHUNK)
    dv0, dv1, dv2 = _zeros3(BLOCK_N, G_PAD, C_CHUNK)

    for m0 in range(0, L, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        if m0 < seqlen:
            (
                row_off, row_mask, row_vec, row_vec_mask,
                qt0, qt1, qt2, do0, do1, do2, u0, u1, u2, wt0, wt1, wt2,
                lse, dob, pix, piy, piz, pit,
            ) = _bwd_row_operands(
                q_ptr, do_ptr, u_ptr, wt_ptr, lse_ptr, dob_ptr, feats_ptr,
                b, tok0, offs_m, offs_g, offs_cc, seqlen, scale,
                H, L, FEAT_STRIDE, C_, C_CHUNK,
                USE_ATTN_BIAS, USE_ACT_BIAS, PACKED, CDTYPE,
            )
            d_row = tl.load(d_ptr + row_vec, mask=row_vec_mask, other=0.0)

            if USE_ATTN_BIAS or USE_ACT_BIAS:
                x = _pair_angle(
                    pix, piy, piz, pit, pjx, pjy, pjz, pjt,
                    TIME_SCALE_C, INPUT_SCALE, CLIP,
                )
                e0, e1, e2 = _e_chunks3(x, freq_ptr, C_, C_CHUNK, F_, CDTYPE)
            else:
                e0 = tl.zeros((BLOCK_M, C_CHUNK, BLOCK_N), dtype=CDTYPE)
                e1 = e0
                e2 = e0
            p, dpr = _recompute_p_dpr(
                qt0, qt1, qt2, u0, u1, u2, do0, do1, do2,
                k0, k1, k2, v0, v1, v2, e0, e1, e2, wt0, wt1, wt2,
                dob, lse, col_valid,
                C_, C_CHUNK, USE_ATTN_BIAS, USE_ACT_BIAS, INPUT_PRECISION,
            )
            ds = p * (dpr - d_row[:, :, None])
            ds = tl.where(col_valid[None, None, :], ds, 0.0)
            pcd = p.to(CDTYPE)
            dscd = ds.to(CDTYPE)

            # dv_j += P^T dO_i ; dk_j += dS^T qt_i  (per chunk, [N, G, CC]).
            pT = tl.trans(pcd, 1, 2, 0)  # [G, N, M]
            dsT = tl.trans(dscd, 1, 2, 0)
            dv0, dv1, dv2 = _acc3_dotT(
                dv0, dv1, dv2, pT, do0, do1, do2,
                C_, C_CHUNK, INPUT_PRECISION,
            )
            dk0, dk1, dk2 = _acc3_dotT(
                dk0, dk1, dk2, dsT, qt0, qt1, qt2,
                C_, C_CHUNK, INPUT_PRECISION,
            )

    _store3(dk_ptr, col_off, dk0, dk1, dk2, col_mask, C_, C_CHUNK)
    _store3(dv_ptr, col_off, dv0, dv1, dv2, col_mask, C_, C_CHUNK)


@triton.jit
def flash_spacetime_bwd_rows_kernel(
    q_ptr, k_ptr, v_ptr, u_ptr, do_ptr, wt_ptr,
    feats_ptr, seqlen_ptr, lse_ptr,
    d_ptr,  # [B, H, L] softmax row sums, written here
    dob_ptr, dqp_ptr, hacc_ptr, gacc_ptr, sig_ptr,
    cu_ptr,  # [B+1] token offsets (PACKED only)
    freq_ptr,
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
    """Row-owned backward: D, then dq's dS@k part plus H, G, sigma."""
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    m0 = pid_m * BLOCK_M

    seqlen = tl.load(seqlen_ptr + b)
    if m0 >= seqlen:
        return
    if PACKED:
        tok0 = tl.load(cu_ptr + b)
    else:
        tok0 = 0
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_g = tl.arange(0, G_PAD)
    offs_cc = tl.arange(0, C_CHUNK)
    head_live = offs_g < H

    (
        row_off, row_mask, row_vec, row_vec_mask,
        qt0, qt1, qt2, do0, do1, do2, u0, u1, u2, wt0, wt1, wt2,
        lse, dob, pix, piy, piz, pit,
    ) = _bwd_row_operands(
        q_ptr, do_ptr, u_ptr, wt_ptr, lse_ptr, dob_ptr, feats_ptr,
        b, tok0, offs_m, offs_g, offs_cc, seqlen, scale,
        H, L, FEAT_STRIDE, C_, C_CHUNK,
        USE_ATTN_BIAS, USE_ACT_BIAS, PACKED, CDTYPE,
    )

    # First pass: the softmax row sum from the same P and dP that the second
    # pass turns into dS, so that rowsum(dS) is zero identically. It is the
    # P-weighted mean of dP, normalised by the recomputed weights rather
    # than assumed to have unit mass: at large logits the fp32 LSE from the
    # forward and the logits recomputed here differ by order one, which
    # scales every P of a row by the same factor, and the mean is invariant
    # to that where the bare sum is not.
    dsum = tl.zeros((BLOCK_M, G_PAD), dtype=tl.float32)
    psum = tl.zeros((BLOCK_M, G_PAD), dtype=tl.float32)
    for n0 in range(0, L, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        col_valid = offs_n < seqlen
        if n0 < seqlen:
            col_off = _chunk_off(
                b, tok0, offs_n, offs_g, offs_cc, H, L, C_, PACKED
            )
            feats_j = _feat_ptrs(
                feats_ptr, b, tok0, offs_n, FEAT_STRIDE, L, PACKED
            )
            col_mask = col_valid[:, None, None] & head_live[None, :, None]
            k0, k1, k2, v0, v1, v2, e0, e1, e2 = _col_tiles(
                k_ptr, v_ptr, freq_ptr, col_off, col_mask, feats_j, col_valid,
                pix, piy, piz, pit,
                BLOCK_M, BLOCK_N, C_, C_CHUNK, F_,
                USE_ATTN_BIAS, USE_ACT_BIAS, CDTYPE,
                TIME_SCALE_C, INPUT_SCALE, CLIP,
            )
            p, dpr = _recompute_p_dpr(
                qt0, qt1, qt2, u0, u1, u2, do0, do1, do2,
                k0, k1, k2, v0, v1, v2, e0, e1, e2, wt0, wt1, wt2,
                dob, lse, col_valid,
                C_, C_CHUNK, USE_ATTN_BIAS, USE_ACT_BIAS, INPUT_PRECISION,
            )
            dsum += tl.sum(p * dpr, axis=2)
            psum += tl.sum(p, axis=2)
    d_row = tl.where(psum > 0.0, dsum / psum, 0.0)
    tl.store(d_ptr + row_vec, d_row, mask=row_vec_mask)

    dqp0, dqp1, dqp2 = _zeros3(BLOCK_M, G_PAD, C_CHUNK)
    if USE_ATTN_BIAS:
        # Held transposed ([M, CC, G]) to match channel-major E; the
        # epilogue store flips them back once.
        h0, h1, h2 = _zeros3(BLOCK_M, C_CHUNK, G_PAD)
        sig = tl.zeros((BLOCK_M, G_PAD), dtype=tl.float32)
    if USE_ACT_BIAS:
        g0, g1, g2 = _zeros3(BLOCK_M, C_CHUNK, G_PAD)

    for n0 in range(0, L, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        col_valid = offs_n < seqlen
        if n0 < seqlen:
            col_off = _chunk_off(
                b, tok0, offs_n, offs_g, offs_cc, H, L, C_, PACKED
            )
            feats_j = _feat_ptrs(
                feats_ptr, b, tok0, offs_n, FEAT_STRIDE, L, PACKED
            )
            col_mask = col_valid[:, None, None] & head_live[None, :, None]
            k0, k1, k2, v0, v1, v2, e0, e1, e2 = _col_tiles(
                k_ptr, v_ptr, freq_ptr, col_off, col_mask, feats_j, col_valid,
                pix, piy, piz, pit,
                BLOCK_M, BLOCK_N, C_, C_CHUNK, F_,
                USE_ATTN_BIAS, USE_ACT_BIAS, CDTYPE,
                TIME_SCALE_C, INPUT_SCALE, CLIP,
            )
            p, dpr = _recompute_p_dpr(
                qt0, qt1, qt2, u0, u1, u2, do0, do1, do2,
                k0, k1, k2, v0, v1, v2, e0, e1, e2, wt0, wt1, wt2,
                dob, lse, col_valid,
                C_, C_CHUNK, USE_ATTN_BIAS, USE_ACT_BIAS, INPUT_PRECISION,
            )
            ds = p * (dpr - d_row[:, :, None])
            ds = tl.where(col_valid[None, None, :], ds, 0.0)
            pcd = p.to(CDTYPE)
            dscd = ds.to(CDTYPE)
            dsT = tl.trans(dscd, 1, 0, 2)  # [G, M, N]

            # dqp += dS @ k (per chunk).
            dqp0, dqp1, dqp2 = _acc3_dotT(
                dqp0, dqp1, dqp2, dsT, k0, k1, k2,
                C_, C_CHUNK, INPUT_PRECISION,
            )
            if USE_ATTN_BIAS:
                # H += dS·E (batched over rows), sigma += rowsum(dS).
                dst_c = tl.trans(dscd, 0, 2, 1)  # [M, N, G]
                h0, h1, h2 = _acc3_e(
                    h0, h1, h2, e0, e1, e2, dst_c,
                    C_, C_CHUNK, INPUT_PRECISION,
                )
                sig += tl.sum(ds, axis=2)
            if USE_ACT_BIAS:
                pct_c = tl.trans(pcd, 0, 2, 1)
                g0, g1, g2 = _acc3_e(
                    g0, g1, g2, e0, e1, e2, pct_c,
                    C_, C_CHUNK, INPUT_PRECISION,
                )

    _store3(dqp_ptr, row_off, dqp0, dqp1, dqp2, row_mask, C_, C_CHUNK)
    if USE_ATTN_BIAS:
        _store3(
            hacc_ptr, row_off,
            tl.trans(h0, 0, 2, 1),
            tl.trans(h1, 0, 2, 1),
            tl.trans(h2, 0, 2, 1),
            row_mask, C_, C_CHUNK,
        )
        tl.store(sig_ptr + row_vec, sig, mask=row_vec_mask)
    if USE_ACT_BIAS:
        _store3(
            gacc_ptr, row_off,
            tl.trans(g0, 0, 2, 1),
            tl.trans(g1, 0, 2, 1),
            tl.trans(g2, 0, 2, 1),
            row_mask, C_, C_CHUNK,
        )


def flash_spacetime_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    feats: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    seqlens: Tensor,
    lse: Tensor,
    grad_out: Tensor,
    scale: Optional[float] = None,
    use_attn_bias: bool = True,
    use_activation_bias: bool = True,
    block_m: int = 16,
    block_n: int = 16,
    num_warps: int = 8,
    num_stages: int = 1,
    cu_seqlens: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Deterministic backward; returns (dq, dk, dv, dW, db).

    With `cu_seqlens` the tensors are packed [T, H, D] (see the forward);
    every row is a real token, so no upstream-gradient masking exists.
    """
    packed = cu_seqlens is not None
    if cu_seqlens is not None:
        cu = cu_seqlens.to(torch.int32).contiguous()
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        batch = int(seqlens.numel())
        _, heads, dim = q.shape
        length = _next_pow2(int(seqlens.max()), floor=16)
        do = grad_out.contiguous()
    else:
        batch, heads, length, dim = q.shape
    scale_value = dim**-0.5 if scale is None else scale
    compute_bf16 = q.dtype != torch.float32
    fp = torch.float32

    if not packed:
        valid = torch.arange(length, device=q.device).unsqueeze(
            0
        ) < seqlens.unsqueeze(1)
        # The op's pad-row outputs are the constant zero, so upstream
        # grads there are discardable regardless of caller garbage.
        do = (grad_out * valid[:, None, :, None]).contiguous()
    qc, kc, vc = (t.contiguous() for t in (q, k, v))
    featsc = feats[..., :4].to(torch.float32).contiguous()
    freqs = sinusoidal_frequencies(dim, q.device)
    w32 = weight.to(fp)

    # Softmax row sums, written by the row kernel and read by the column
    # kernel; same layout as the LSE.
    dsum = torch.zeros_like(lse)
    u = (
        ((qc.to(fp) * scale_value) @ w32).to(qc.dtype).contiguous()
        if use_attn_bias
        else qc
    )
    if use_activation_bias:
        wt = (do.to(fp) @ w32).to(do.dtype).contiguous()
        dob = (
            do.to(fp) @ bias.to(fp)
            if bias is not None
            else torch.zeros_like(lse)
        )
    else:
        wt = qc
        dob = lse  # dummy pointer, never read
    dob = dob.contiguous()

    dk = torch.zeros_like(kc, dtype=fp)
    dv = torch.zeros_like(vc, dtype=fp)
    dqp = torch.zeros_like(qc, dtype=fp)
    hacc = torch.zeros_like(qc, dtype=fp) if use_attn_bias else dqp
    gacc = torch.zeros_like(qc, dtype=fp) if use_activation_bias else dqp
    sig = torch.zeros_like(lse) if use_attn_bias else lse

    seq32 = seqlens.to(torch.int32).contiguous()
    cu32 = cu if packed else seq32
    common = dict(
        scale=scale_value,
        L=length,
        H=heads,
        FEAT_STRIDE=4,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        G_PAD=_next_pow2(heads),
        C_=dim,
        C_CHUNK=16,
        F_=dim // 2,
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
    flash_spacetime_bwd_rows_kernel[(triton.cdiv(length, block_m), batch)](
        qc, kc, vc, u, do, wt, featsc, seq32, lse,
        dsum, dob, dqp, hacc, gacc, sig, cu32, freqs,
        **common,
    )
    flash_spacetime_bwd_cols_kernel[(triton.cdiv(length, block_n), batch)](
        qc, kc, vc, u, do, wt, featsc, seq32, lse,
        dsum, dob, dk, dv, cu32, freqs,
        **common,
    )

    dq = dqp
    if use_attn_bias:
        dq = dq + hacc @ w32.t()
        if bias is not None:
            dq = dq + sig.unsqueeze(-1) * bias.to(fp)
    dq = dq * scale_value

    dw = torch.zeros_like(w32)
    if use_attn_bias:
        qt_flat = (qc.to(fp) * scale_value).reshape(-1, dim)
        dw = dw + qt_flat.t() @ hacc.reshape(-1, dim)
    if use_activation_bias:
        dw = dw + do.to(fp).reshape(-1, dim).t() @ gacc.reshape(-1, dim)
    # dW_ce = sum dR_c E_e: the row side (qt/dO) carries the dR channel
    # index c, the accumulators carry the E index e.
    db: Optional[Tensor] = None
    if bias is not None:
        if use_activation_bias:
            db = do.to(fp).reshape(-1, dim).sum(0)
        else:
            # The logit path's db vanishes analytically (softmax shift
            # invariance).
            db = torch.zeros(dim, dtype=fp, device=q.device)

    return (
        dq.to(q.dtype),
        dk.to(k.dtype),
        dv.to(v.dtype),
        dw.to(weight.dtype),
        db.to(bias.dtype) if (db is not None and bias is not None) else db,
    )
