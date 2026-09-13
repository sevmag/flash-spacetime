"""Shared Triton building blocks for the fused spacetime-attention kernels.

Every channel-carrying tile lives in up to three `C_CHUNK`-wide chunks:
Triton block shapes must be powers of two, the supported head widths
(16, 32, 48) are covered exactly by 16-wide chunks, and the narrow dot
slices keep shared-memory staging inside the fp32 path's budget. The
`*3` helpers here operate on such chunk triples, with `C_` constexpr
guards folding away every chunk past the actual width at compile time.

Where a chunk does not exist its slot in a triple aliases chunk 0; no
guarded consumer ever reads it, so the alias is dead code after
specialization. It exists only so triples can be passed around whole.

Layout conventions (M = query rows, N = key columns, G = padded heads,
CC = C_CHUNK): row-side operand chunks are [M, G, CC], key-side chunks
[N, G, CC], and pair-feature chunks channel-major [M, CC, N] so the
logit-side dots consume E untransposed and the value-side dots transpose
the much smaller P tile instead.
"""

import triton
import triton.language as tl


@triton.jit
def _pair_angle(
    px, py, pz, pt,  # [M] query-row coordinates
    qx, qy, qz, qt_,  # [N] key-column coordinates
    TIME_SCALE_C: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
    CLIP: tl.constexpr,
) -> tl.tensor:
    """`x_ij` of SpacetimeEncoder for one (M, N) tile, fp32."""
    dx = px[:, None] - qx[None, :]
    dy = py[:, None] - qy[None, :]
    dz = pz[:, None] - qz[None, :]
    dt = (pt[:, None] - qt_[None, :]) * TIME_SCALE_C
    interval = dx * dx + dy * dy + dz * dz - dt * dt
    # sign(I) * sqrt(|I|); sign(0) = 0 is preserved by the where.
    dist = tl.where(interval > 0.0, tl.sqrt(interval), -tl.sqrt(-interval))
    dist = tl.where(interval == 0.0, 0.0, dist)
    return INPUT_SCALE * tl.minimum(tl.maximum(dist, -CLIP), CLIP)


@triton.jit
def _e_chunk(
    x,  # [M, N] pair angle
    freq_ptr,
    BASE: tl.constexpr,
    F_: tl.constexpr,
    C_CHUNK: tl.constexpr,
    CDTYPE: tl.constexpr,
) -> tl.tensor:
    """One channel chunk of E: sin for gc < F, cos above -> [M, CC, N]."""
    gc = BASE + tl.arange(0, C_CHUNK)
    f = tl.load(freq_ptr + gc % F_)
    theta = x[:, None, :] * f[None, :, None]
    return tl.where((gc < F_)[None, :, None], tl.sin(theta), tl.cos(theta)).to(
        CDTYPE
    )


@triton.jit
def _e_chunks3(
    x, freq_ptr,
    C_: tl.constexpr, C_CHUNK: tl.constexpr, F_: tl.constexpr,
    CDTYPE: tl.constexpr,
) -> tuple:
    """All live channel chunks of E for one pair-angle tile."""
    e0 = _e_chunk(x, freq_ptr, 0 * C_CHUNK, F_, C_CHUNK, CDTYPE)
    e1 = e0
    e2 = e0
    if C_ > C_CHUNK:
        e1 = _e_chunk(x, freq_ptr, 1 * C_CHUNK, F_, C_CHUNK, CDTYPE)
    if C_ > 2 * C_CHUNK:
        e2 = _e_chunk(x, freq_ptr, 2 * C_CHUNK, F_, C_CHUNK, CDTYPE)
    return e0, e1, e2


@triton.jit
def _chunk_off(
    b, tok0,
    offs, offs_g, offs_cc,  # [X] event-local indices, [G], [CC]
    H: tl.constexpr, L: tl.constexpr, C_: tl.constexpr,
    PACKED: tl.constexpr,
) -> tl.tensor:
    """[X, G, CC] element offsets of chunk 0 in the q/k/v/o layout."""
    if PACKED:
        # Packed [T, H, C]: rows of event b start at token tok0.
        return (
            (tok0 + offs[:, None, None]) * H + offs_g[None, :, None]
        ) * C_ + offs_cc[None, None, :]
    return (
        (b * H + offs_g[None, :, None]) * L + offs[:, None, None]
    ) * C_ + offs_cc[None, None, :]


@triton.jit
def _vec_off(
    b, tok0, offs, offs_g,
    H: tl.constexpr, L: tl.constexpr, PACKED: tl.constexpr,
) -> tl.tensor:
    """[X, G] element offsets in the per-(row, head) layout (LSE, D, ...)."""
    if PACKED:
        return (tok0 + offs[:, None]) * H + offs_g[None, :]
    return (b * H + offs_g[None, :]) * L + offs[:, None]


@triton.jit
def _feat_ptrs(
    feats_ptr, b, tok0, offs,
    FEAT_STRIDE: tl.constexpr, L: tl.constexpr, PACKED: tl.constexpr,
) -> tl.tensor:
    """[X] pointers to the feature rows of a row or column block."""
    if PACKED:
        return feats_ptr + (tok0 + offs) * FEAT_STRIDE
    return feats_ptr + b * L * FEAT_STRIDE + offs * FEAT_STRIDE


@triton.jit
def _coords4(base, mask) -> tuple:
    """The four interval coordinates (x, y, z, t) of a feature-row block."""
    x = tl.load(base + 0, mask=mask, other=0.0)
    y = tl.load(base + 1, mask=mask, other=0.0)
    z = tl.load(base + 2, mask=mask, other=0.0)
    t = tl.load(base + 3, mask=mask, other=0.0)
    return x, y, z, t


@triton.jit
def _load3(
    ptr, off, mask,
    C_: tl.constexpr, C_CHUNK: tl.constexpr, CDTYPE: tl.constexpr,
) -> tuple:
    """Chunk triple of one operand block; dead lanes load 0."""
    t0 = tl.load(ptr + off + 0 * C_CHUNK, mask=mask, other=0.0).to(CDTYPE)
    t1 = t0
    t2 = t0
    if C_ > C_CHUNK:
        t1 = tl.load(ptr + off + 1 * C_CHUNK, mask=mask, other=0.0).to(CDTYPE)
    if C_ > 2 * C_CHUNK:
        t2 = tl.load(ptr + off + 2 * C_CHUNK, mask=mask, other=0.0).to(CDTYPE)
    return t0, t1, t2


@triton.jit
def _load3_scaled(
    ptr, off, mask, scale,
    C_: tl.constexpr, C_CHUNK: tl.constexpr, CDTYPE: tl.constexpr,
) -> tuple:
    """Chunk triple scaled before the compute-dtype cast (q · scale)."""
    t0 = (tl.load(ptr + off + 0 * C_CHUNK, mask=mask, other=0.0) * scale).to(
        CDTYPE
    )
    t1 = t0
    t2 = t0
    if C_ > C_CHUNK:
        t1 = (
            tl.load(ptr + off + 1 * C_CHUNK, mask=mask, other=0.0) * scale
        ).to(CDTYPE)
    if C_ > 2 * C_CHUNK:
        t2 = (
            tl.load(ptr + off + 2 * C_CHUNK, mask=mask, other=0.0) * scale
        ).to(CDTYPE)
    return t0, t1, t2


@triton.jit
def _store3(
    ptr, off, t0, t1, t2, mask,
    C_: tl.constexpr, C_CHUNK: tl.constexpr,
) -> None:
    """Store the live chunks of a triple."""
    tl.store(ptr + off + 0 * C_CHUNK, t0, mask=mask)
    if C_ > C_CHUNK:
        tl.store(ptr + off + 1 * C_CHUNK, t1, mask=mask)
    if C_ > 2 * C_CHUNK:
        tl.store(ptr + off + 2 * C_CHUNK, t2, mask=mask)


@triton.jit
def _zeros3(D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr) -> tuple:
    """fp32 accumulator triple; dead chunks alias chunk 0 harmlessly."""
    t0 = tl.zeros((D0, D1, D2), dtype=tl.float32)
    return t0, t0, t0


@triton.jit
def _dot3_qk(
    a0, a1, a2,  # [M, G, CC]
    b0, b1, b2,  # [N, G, CC]
    C_: tl.constexpr, C_CHUNK: tl.constexpr, IP: tl.constexpr,
) -> tl.tensor:
    """Per-head channel contraction: sum_c a_c b_c -> [G, M, N] fp32."""
    s = tl.dot(tl.trans(a0, 1, 0, 2), tl.trans(b0, 1, 2, 0), input_precision=IP)
    if C_ > C_CHUNK:
        s += tl.dot(
            tl.trans(a1, 1, 0, 2), tl.trans(b1, 1, 2, 0), input_precision=IP
        )
    if C_ > 2 * C_CHUNK:
        s += tl.dot(
            tl.trans(a2, 1, 0, 2), tl.trans(b2, 1, 2, 0), input_precision=IP
        )
    return s


@triton.jit
def _dot3_e(
    a0, a1, a2,  # [M, G, CC]
    e0, e1, e2,  # [M, CC, N]
    C_: tl.constexpr, C_CHUNK: tl.constexpr, IP: tl.constexpr,
) -> tl.tensor:
    """Row-batched contraction against E: sum_e a_e E_e -> [M, G, N] fp32."""
    s = tl.dot(a0, e0, input_precision=IP)
    if C_ > C_CHUNK:
        s += tl.dot(a1, e1, input_precision=IP)
    if C_ > 2 * C_CHUNK:
        s += tl.dot(a2, e2, input_precision=IP)
    return s


@triton.jit
def _acc3_dotT(
    acc0, acc1, acc2,  # [X, G, CC]
    aT,  # [G, X, Y]
    b0, b1, b2,  # [Y, G, CC]
    C_: tl.constexpr, C_CHUNK: tl.constexpr, IP: tl.constexpr,
) -> tuple:
    """acc_c += aT @ b_c per head, restored to the [X, G, CC] layout."""
    acc0 += tl.trans(
        tl.dot(aT, tl.trans(b0, 1, 0, 2), input_precision=IP), 1, 0, 2
    )
    if C_ > C_CHUNK:
        acc1 += tl.trans(
            tl.dot(aT, tl.trans(b1, 1, 0, 2), input_precision=IP), 1, 0, 2
        )
    if C_ > 2 * C_CHUNK:
        acc2 += tl.trans(
            tl.dot(aT, tl.trans(b2, 1, 0, 2), input_precision=IP), 1, 0, 2
        )
    return acc0, acc1, acc2


@triton.jit
def _acc3_pv(
    acc0, acc1, acc2,  # [M, G, CC]
    alpha,  # [M, G] online-softmax rescale
    pbt,  # [G, M, N]
    v0, v1, v2,  # [N, G, CC]
    C_: tl.constexpr, C_CHUNK: tl.constexpr, IP: tl.constexpr,
) -> tuple:
    """Flash-rescaled P@V accumulation, per chunk."""
    acc0 = acc0 * alpha[:, :, None] + tl.trans(
        tl.dot(pbt, tl.trans(v0, 1, 0, 2), input_precision=IP), 1, 0, 2
    )
    if C_ > C_CHUNK:
        acc1 = acc1 * alpha[:, :, None] + tl.trans(
            tl.dot(pbt, tl.trans(v1, 1, 0, 2), input_precision=IP), 1, 0, 2
        )
    if C_ > 2 * C_CHUNK:
        acc2 = acc2 * alpha[:, :, None] + tl.trans(
            tl.dot(pbt, tl.trans(v2, 1, 0, 2), input_precision=IP), 1, 0, 2
        )
    return acc0, acc1, acc2


@triton.jit
def _acc3_e(
    h0, h1, h2,  # [M, CC, G]
    e0, e1, e2,  # [M, CC, N]
    m,  # [M, N, G]
    C_: tl.constexpr, C_CHUNK: tl.constexpr, IP: tl.constexpr,
) -> tuple:
    """h_c += E_c @ m: per-row E contraction against P or dS."""
    h0 += tl.dot(e0, m, input_precision=IP)
    if C_ > C_CHUNK:
        h1 += tl.dot(e1, m, input_precision=IP)
    if C_ > 2 * C_CHUNK:
        h2 += tl.dot(e2, m, input_precision=IP)
    return h0, h1, h2


@triton.jit
def _acc3_e_rescaled(
    h0, h1, h2,  # [M, CC, G]
    alpha,  # [M, G]
    e0, e1, e2,  # [M, CC, N]
    m,  # [M, N, G]
    C_: tl.constexpr, C_CHUNK: tl.constexpr, IP: tl.constexpr,
) -> tuple:
    """Flash-rescaled variant of `_acc3_e` for the forward's online pass."""
    h0 = h0 * alpha[:, None, :] + tl.dot(e0, m, input_precision=IP)
    if C_ > C_CHUNK:
        h1 = h1 * alpha[:, None, :] + tl.dot(e1, m, input_precision=IP)
    if C_ > 2 * C_CHUNK:
        h2 = h2 * alpha[:, None, :] + tl.dot(e2, m, input_precision=IP)
    return h0, h1, h2


def _next_pow2(n: int, floor: int = 16) -> int:
    p = floor
    while p < n:
        p *= 2
    return p
