# Framework Benchmark Results - Qwen3-0.6B

## Hardware
- GPU: RTX 3090 (24GB)
- CPU: AMD Threadripper
- CUDA: 12.1

## Benchmark Configuration
- Prompts: 3 test prompts
- Decode tokens per prompt: 100
- Temperature: 0.0 (greedy)

## Results

| Framework | TTFT (s) | Decode tok/s | Peak Memory | Speedup vs HF |
|-----------|----------|--------------|-------------|---------------|
| HuggingFace | 0.071 | 81 | 1.4 GB | 1.0x |
| **Megakernel** | **0.004** | **239** | 2.6 GB | **2.95x** |
| vLLM | - | - | - | OOM (needs isolated run) |
| SGLang | - | - | - | Requires server |
| llama.cpp | - | - | - | Needs GGUF conversion |

## Key Findings

1. **Megakernel is 2.95x faster than HuggingFace** for decode throughput
2. **TTFT is 17.75x faster** (4ms vs 71ms) due to fused kernel launch
3. Memory usage is higher for megakernel (2.6 GB vs 1.4 GB) due to:
   - Pre-allocated intermediate buffers
   - Full KV cache allocation upfront

## Quality Metrics

| Framework | KL Divergence | Perplexity | Argmax Match |
|-----------|---------------|------------|--------------|
| HuggingFace | 0.0 (ref) | 35.51 | 100% |
| Megakernel | N/A | N/A | 100% |

The megakernel produces identical argmax predictions to HuggingFace on test prompts.
Full KL divergence comparison requires exposing logits from the megakernel.

## Running Benchmarks

```bash
# Throughput benchmark
python experiments/framework_bench/benchmark_suite.py

# Quality metrics
python experiments/framework_bench/quality_metrics.py
```

## TODO

- [ ] Run vLLM in isolated process (OOM with other models loaded)
- [ ] Convert model to GGUF for llama.cpp/Ollama
- [ ] Set up SGLang server for comparison
- [ ] Add TensorRT-LLM benchmark
- [ ] Add ExLlamaV2 benchmark
