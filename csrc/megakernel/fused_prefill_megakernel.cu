/**
 * Fused Prefill Megakernel for Qwen3-0.6B
 *
 * Split layout:
 * - fused_prefill_megakernel_common.inl : config/debug/helpers
 * - fused_prefill_megakernel_core.inl   : core device/kernels
 * - fused_prefill_megakernel_launch.inl : launch entrypoint
 */

#include "fused_prefill_megakernel_common.inl"
#include "fused_prefill_megakernel_core.inl"
#include "fused_prefill_megakernel_launch.inl"
