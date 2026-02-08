/**
 * Fused Decode with __ldg() cached reads.
 *
 * Split layout:
 * - fused_decode_ldg_common.inl : debug/config + qkv/attn helpers
 * - fused_decode_ldg_mlp.inl    : o-proj/postnorm/mlp
 * - fused_decode_ldg_kernel.inl : cooperative decode kernel
 * - fused_decode_ldg_lm.inl     : lm-head kernels
 * - fused_decode_ldg_launch.inl : C API launch wrappers
 */

#include "fused_decode_ldg_common.inl"
#include "fused_decode_ldg_mlp.inl"
#include "fused_decode_ldg_kernel.inl"
#include "fused_decode_ldg_lm.inl"
#include "fused_decode_ldg_launch.inl"
