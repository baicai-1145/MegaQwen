/**
 * Fused Prefill Kernel for Qwen3-0.6B
 *
 * Split layout:
 * - fused_prefill_common.inl   : config/debug/helpers
 * - fused_prefill_kernels.inl  : CUDA kernels
 * - fused_prefill_launch.inl   : launch entrypoints
 */

#include "fused_prefill_common.inl"
#include "fused_prefill_kernels.inl"
#include "fused_prefill_launch.inl"
