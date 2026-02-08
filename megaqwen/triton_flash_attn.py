from __future__ import annotations

import math
from typing import Optional

import torch


# NOTE: This is a minimal flash-attention style kernel for audio tower self-attention.
# It targets the common ASR path:
# - batch size B=1
# - no padding mask (full 30s window)
# - head_dim=64 (Qwen3-ASR-0.6B audio tower: d_model=896, n_heads=14)
#
# We keep it small/contained (KISS) and fall back to PyTorch SDPA for other cases.


def is_triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception:
        return False
    return True


def flash_attn_bhwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute attention(q,k,v) for q/k/v shaped [B, H, T, D] (no mask).

    Returns: [B, H, T, D]
    """
    if q.device.type != "cuda":
        raise ValueError("flash_attn_bhwd expects CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"flash_attn_bhwd only supports fp16/bf16, got {q.dtype}")
    if q.ndim != 4:
        raise ValueError(f"Expected q.ndim=4 [B,H,T,D], got {tuple(q.shape)}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"Shape mismatch q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")

    # Keep it minimal: only support head_dim=64 for now (Qwen3-ASR-0.6B audio tower).
    b, h, t, d = q.shape
    if d != 64:
        raise ValueError(f"Only head_dim=64 supported, got {d}")

    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # Ensure contiguous last dimension for tl.dot.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty_like(q)

    import triton
    import triton.language as tl

    @triton.jit
    def _attn_fwd(
        Q,
        K,
        V,
        O,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qt: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kt: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vt: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_ot: tl.constexpr,
        stride_od: tl.constexpr,
        T: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        OUT_BF16: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        # Query block offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        # Pointers to Q block: [BLOCK_M, D]
        q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < T), other=0.0).to(tl.float32)

        # Online softmax accumulators
        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

        # Loop over keys/values
        for start_n in range(0, T, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
            v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[None, :] * stride_vt + offs_d[:, None] * stride_vd

            k = tl.load(k_ptrs, mask=(offs_n[None, :] < T), other=0.0).to(tl.float32)  # [D, N]
            v = tl.load(v_ptrs, mask=(offs_n[None, :] < T), other=0.0).to(tl.float32)  # [D, N]

            # qk: [M, N]
            qk = tl.dot(q, k) * SCALE
            qk = tl.where(offs_n[None, :] < T, qk, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            alpha = tl.exp(m_i - m_ij)
            acc = acc * alpha[:, None] + tl.dot(p, tl.trans(v))
            l_i = l_i * alpha + l_ij
            m_i = m_ij

        out = acc / l_i[:, None]
        out = out.to(tl.bfloat16) if OUT_BF16 else out.to(tl.float16)

        o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
        tl.store(o_ptrs, out, mask=(offs_m[:, None] < T))

    grid = (triton.cdiv(t, 128), h, b)
    # Tuned for head_dim=64. For T~1500, BLOCK_M=128 works well on many GPUs.
    _attn_fwd[grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        T=t,
        SCALE=scale,
        BLOCK_M=128,
        BLOCK_N=64,
        BLOCK_D=64,
        OUT_BF16=(q.dtype == torch.bfloat16),
        num_warps=4,
    )

    return out
