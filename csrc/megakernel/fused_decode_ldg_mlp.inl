// =============================================================================
// O Projection + Residual + PostNorm + MLP (all with __ldg)
// =============================================================================

__device__ void ldg_o_proj_postnorm_mlp(
    cg::grid_group& grid,
    const __nv_bfloat16* __restrict__ o_weight,
    const __nv_bfloat16* __restrict__ post_norm_weight,
    const __nv_bfloat16* __restrict__ gate_weight,
    const __nv_bfloat16* __restrict__ up_weight,
    const __nv_bfloat16* __restrict__ down_weight,
    const float* __restrict__ attn_out,
    float* __restrict__ g_residual,
    float* __restrict__ g_activations,
    float* __restrict__ g_mlp_intermediate,
    __nv_bfloat16* __restrict__ hidden_out,
    unsigned long long* __restrict__ dbg_stage
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // O Projection + Residual
    int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
    int hid_start = block_id * hid_per_block;
    int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

    unsigned long long t0_oproj = 0;
    if (dbg_stage != nullptr) t0_oproj = clock64();
    for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
        int m = m_base + warp_id;

        if (m < hid_end) {
            const __nv_bfloat16* o_row = o_weight + m * Q_SIZE;

            float sum = 0.0f;
            MEGAQWEN_PRAGMA_UNROLL_N(MEGAQWEN_LDG_UNROLL)
            for (int k = lane_id * 4; k < Q_SIZE; k += WARP_SIZE * 4) {
                uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(o_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum += __bfloat162float(w_ptr[0]) * attn_out[k] +
                       __bfloat162float(w_ptr[1]) * attn_out[k+1] +
                       __bfloat162float(w_ptr[2]) * attn_out[k+2] +
                       __bfloat162float(w_ptr[3]) * attn_out[k+3];
            }

            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0) {
                g_activations[m] = sum + g_residual[m];
            }
        }
    }

    if (dbg_stage != nullptr) {
        unsigned long long t1 = clock64();
        dbg_stage[DBG_OPROJ_RESID] = t1 - t0_oproj;
    }
    {
        unsigned long long t0s = 0;
        if (dbg_stage != nullptr) t0s = clock64();
        grid.sync();
        if (dbg_stage != nullptr) {
            unsigned long long t1 = clock64();
            dbg_stage[DBG_OPROJ_SYNC] = t1 - t0s;
        }
    }

    // Post-attention RMSNorm
    if (block_id == 0) {
        unsigned long long t0_post = 0;
        if (dbg_stage != nullptr) t0_post = clock64();
        __shared__ float smem_reduce[LDG_NUM_WARPS];

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float v = g_activations[i];
            g_residual[i] = v;
            local_sum_sq += v * v;
        }

        local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float w = __bfloat162float(__ldg(post_norm_weight + i));
            g_activations[i] = g_residual[i] * rstd * w;
        }
        if (dbg_stage != nullptr) {
            unsigned long long t1 = clock64();
            dbg_stage[DBG_POSTNORM] = t1 - t0_post;
        }
    }

    {
        unsigned long long t0s = 0;
        if (dbg_stage != nullptr) t0s = clock64();
        grid.sync();
        if (dbg_stage != nullptr) {
            unsigned long long t1 = clock64();
            dbg_stage[DBG_POSTNORM_SYNC] = t1 - t0s;
        }
    }

    // Gate + Up + SiLU
    int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
    int int_start = block_id * int_per_block;
    int int_end = min(int_start + int_per_block, INTERMEDIATE_SIZE);

    unsigned long long t0_gate = 0;
    if (dbg_stage != nullptr) t0_gate = clock64();
    for (int m_base = int_start; m_base < int_end; m_base += LDG_NUM_WARPS) {
        int m = m_base + warp_id;

        if (m < int_end) {
            const __nv_bfloat16* gate_row = gate_weight + m * HIDDEN_SIZE;
            const __nv_bfloat16* up_row = up_weight + m * HIDDEN_SIZE;

            float gate_sum = 0.0f, up_sum = 0.0f;

            MEGAQWEN_PRAGMA_UNROLL_N(MEGAQWEN_LDG_UNROLL)
            for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 g_u2 = __ldg(reinterpret_cast<const uint2*>(gate_row + k));
                uint2 u_u2 = __ldg(reinterpret_cast<const uint2*>(up_row + k));
                __nv_bfloat16* g_ptr = reinterpret_cast<__nv_bfloat16*>(&g_u2);
                __nv_bfloat16* u_ptr = reinterpret_cast<__nv_bfloat16*>(&u_u2);

                gate_sum += __bfloat162float(g_ptr[0]) * g_activations[k] +
                            __bfloat162float(g_ptr[1]) * g_activations[k+1] +
                            __bfloat162float(g_ptr[2]) * g_activations[k+2] +
                            __bfloat162float(g_ptr[3]) * g_activations[k+3];

                up_sum += __bfloat162float(u_ptr[0]) * g_activations[k] +
                          __bfloat162float(u_ptr[1]) * g_activations[k+1] +
                          __bfloat162float(u_ptr[2]) * g_activations[k+2] +
                          __bfloat162float(u_ptr[3]) * g_activations[k+3];
            }

            gate_sum = ldg_warp_reduce_sum(gate_sum);
            up_sum = ldg_warp_reduce_sum(up_sum);

            if (lane_id == 0) {
                g_mlp_intermediate[m] = ldg_silu(gate_sum) * up_sum;
            }
        }
    }

    if (dbg_stage != nullptr) {
        unsigned long long t1 = clock64();
        dbg_stage[DBG_GATE_UP] = t1 - t0_gate;
    }
    {
        unsigned long long t0s = 0;
        if (dbg_stage != nullptr) t0s = clock64();
        grid.sync();
        if (dbg_stage != nullptr) {
            unsigned long long t1 = clock64();
            dbg_stage[DBG_GATE_UP_SYNC] = t1 - t0s;
        }
    }

    // Down projection + residual
    unsigned long long t0_down = 0;
    if (dbg_stage != nullptr) t0_down = clock64();
    for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
        int m = m_base + warp_id;

        if (m < hid_end) {
            const __nv_bfloat16* down_row = down_weight + m * INTERMEDIATE_SIZE;

            float sum = 0.0f;
            MEGAQWEN_PRAGMA_UNROLL_N(MEGAQWEN_LDG_UNROLL)
            for (int k = lane_id * 4; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 4) {
                uint2 d_u2 = __ldg(reinterpret_cast<const uint2*>(down_row + k));
                __nv_bfloat16* d_ptr = reinterpret_cast<__nv_bfloat16*>(&d_u2);

                sum += __bfloat162float(d_ptr[0]) * g_mlp_intermediate[k] +
                       __bfloat162float(d_ptr[1]) * g_mlp_intermediate[k+1] +
                       __bfloat162float(d_ptr[2]) * g_mlp_intermediate[k+2] +
                       __bfloat162float(d_ptr[3]) * g_mlp_intermediate[k+3];
            }

            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0) {
                hidden_out[m] = __float2bfloat16(sum + g_residual[m]);
            }
        }
    }

    if (dbg_stage != nullptr) {
        unsigned long long t1 = clock64();
        dbg_stage[DBG_DOWNPROJ] = t1 - t0_down;
    }
    {
        unsigned long long t0s = 0;
        if (dbg_stage != nullptr) t0s = clock64();
        grid.sync();
        if (dbg_stage != nullptr) {
            unsigned long long t1 = clock64();
            dbg_stage[DBG_DOWNPROJ_SYNC] = t1 - t0s;
        }
    }
}

