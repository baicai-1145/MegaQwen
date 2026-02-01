# MegaQwen Experiment Results

**Model**: Qwen3-0.6B (28 layers, 16 heads, 1024 hidden dim)
**Hardware**: NVIDIA RTX 3090 (24GB, 420W TDP)
**Kernel**: 82 blocks x 256 threads, ~225 grid.sync() per decode step

---

## 1. Framework Comparison

### Throughput

| Framework | TTFT (ms) | Decode tok/s | Peak Memory | Speedup vs HF |
|-----------|-----------|--------------|-------------|---------------|
| HuggingFace | 71 | 81 | 1.4 GB | 1.0x |
| **Megakernel** | **4** | **239** | 2.6 GB | **2.95x** |
| vLLM | 47 | 107 | ~2.0 GB | 1.32x |

### Power & Energy Efficiency

| Framework | Idle (W) | Avg (W) | Peak (W) | tok/s | tok/J | Efficiency vs HF |
|-----------|----------|---------|----------|-------|-------|------------------|
| HuggingFace | 169 | 185 | 190 | 58 | 0.31 | 1.0x |
| **Megakernel** | 172 | 200 | 228 | 157 | **0.78** | **2.48x** |
| vLLM | 171 | 195 | 209 | 107 | 0.55 | 1.74x |

### Key Findings

- Megakernel is **2.95x faster** and **2.48x more energy efficient** than HuggingFace
- TTFT is **17.75x faster** (4ms vs 71ms) due to single fused kernel launch
- Higher peak power (228W) but significantly faster completion = better energy efficiency

---

## 2. Cooperative Kernel vs CUDA Graph

### Synchronization Overhead (Empty Kernels)

| Approach | Time | Per-Op Cost |
|----------|------|-------------|
| Cooperative + 225 grid.sync() | 167.3 us | 0.73 us/sync |
| CUDA graph (225 kernels) | 186.9 us | 0.83 us/kernel |
| 225 regular kernel launches | 347.5 us | 1.54 us/launch |

### Conclusion

**Cooperative kernel wins by 19.7 us** over CUDA graph for pure sync overhead.

But the real benefit of the megakernel isn't sync savings - it's **memory bandwidth savings**:
- Avoids ~340 intermediate global memory writes/reads
- Saves ~2.7 MB memory traffic per token
- Estimated savings: 1000+ us per decode step

Splitting at grid.sync() points would lose these memory benefits.

---

## 3. Optimization Experiments

### Redundant RMSNorm (Implemented)

**Approach**: All 82 blocks compute RMSNorm redundantly instead of only block 0.
**Syncs eliminated**: 56 (2 per layer x 28 layers)

| Position | Original | Optimized | Speedup |
|----------|----------|-----------|---------|
| 1 | 5.655ms | 3.982ms | 1.42x |
| 10 | 5.708ms | 4.006ms | 1.42x |
| 50 | 5.819ms | 4.107ms | 1.42x |
| 100 | 5.936ms | 4.227ms | 1.40x |
| 200 | 6.202ms | 6.883ms | 0.90x |

**Result**: +26.3% throughput (170 -> 215 tok/s) at short sequences. Degrades at long sequences due to L2 cache pressure from KV cache.

**Trade-off**: Best for interactive use (short contexts). Not recommended for long-context workloads.

### Head-Based Work Distribution (NOT VIABLE)

**Approach**: Assign 5 blocks per Q head so QKV + attention can proceed without grid.sync().
**Syncs eliminated**: 28 (1 per layer)

| Position | Original | Optimized | Speedup |
|----------|----------|-----------|---------|
| 1 | 4.023ms | 6.862ms | 0.59x |
| 10 | 4.051ms | 6.892ms | 0.59x |
| 50 | 4.151ms | 6.990ms | 0.59x |

**Result**: -33% throughput (213 -> 142 tok/s). **Not viable.**

**Why it failed**: QKV is memory-bound. Reducing from 82 to 16 working blocks loses parallelism. The memory bandwidth loss far exceeds sync savings.

**Lesson**: Don't sacrifice block utilization to eliminate syncs.

### Fused Phases (Not Implemented)

**Approach**: Fuse adjacent phases (QKV + QK norm + RoPE, O proj + residual + post-attn RMSNorm).
**Expected syncs eliminated**: ~56

Status: Not yet implemented.

---

## 4. Quality Metrics

| Framework | Argmax Match vs HF |
|-----------|-------------------|
| Megakernel | 100% |

Full KL divergence comparison requires exposing logits from the megakernel.

---

## Summary Table

| Metric | Megakernel | vs HuggingFace | vs vLLM |
|--------|------------|----------------|---------|
| Decode tok/s | 239 | **2.95x** | 2.23x |
| TTFT | 4ms | **17.75x** | 11.75x |
| Energy (tok/J) | 0.78 | **2.48x** | 1.42x |
| Memory | 2.6 GB | 1.86x more | 1.30x more |

---

## Running Benchmarks

```bash
# Framework throughput
python experiments/framework_bench/benchmark_suite.py

# Power consumption
python experiments/framework_bench/power_benchmark.py

# Sync overhead analysis
python experiments/sync_overhead.py

# Optimization experiments
python experiments/optimizations/redundant_rmsnorm/benchmark.py
python experiments/optimizations/compare_all.py
```

---

## TODO

- [ ] Convert to GGUF for llama.cpp/Ollama comparison
- [ ] Set up SGLang server benchmark
- [ ] Add TensorRT-LLM benchmark
- [ ] Add ExLlamaV2 benchmark
- [ ] Implement fused phases optimization
- [ ] Expose logits for KL divergence measurement
