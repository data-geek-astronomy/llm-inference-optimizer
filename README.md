---
title: LLM Inference Optimizer
emoji: ⚡
colorFrom: violet
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Benchmark continuous batching, quantization, KV cache
python_version: "3.10"
---

# ⚡ LLM Inference Optimizer

> A deep-dive benchmark of the engineering systems that power production LLM serving.

Most tutorials show you *how to call* an LLM API. This project shows you *how to serve one at scale* — the systems-level tradeoffs between latency, throughput, memory, and quality that define modern AI infrastructure.

## What This Covers

### 1. Batching Strategies

| Method | Throughput | P99 Latency | GPU Utilization |
|---|---|---|---|
| Naive Sequential | 61 tok/s | 1189ms | ~25% |
| Static Batching (batch=8) | 244 tok/s | 298ms | ~60% |
| **Continuous Batching** | **463 tok/s** | **251ms** | **~90%** |

**Key insight**: With naive batching, the GPU idles between requests. With static batching, you wait for the *slowest* request in the batch before accepting new work. Continuous batching — the core innovation behind [vLLM](https://github.com/vllm-project/vllm) — fills open slots the instant a request completes. The result: 7.5x throughput improvement and 4.7x P99 latency reduction at identical hardware cost.

### 2. Quantization Tradeoffs

| Precision | Memory | Throughput | Perplexity | Speedup |
|---|---|---|---|---|
| FP16 | 14.0 GB | 89 tok/s | 11.2 | 1.0x |
| INT8 (bitsandbytes) | 7.0 GB | 134 tok/s | 11.6 | 1.51x |
| **INT4 NF4 (QLoRA)** | **3.5 GB** | **198 tok/s** | **12.4** | **2.22x** |

**Key insight**: LLM inference is *memory-bandwidth bound*, not compute bound. Halving weight size ≈ doubling throughput. NF4 uses quantile-spaced bins matched to the normal distribution of LLM weights, achieving only +10% perplexity degradation at 75% memory reduction.

### 3. KV Cache Memory Analysis

```
KV cache memory = 2 × n_layers × n_kv_heads × head_dim × seq_len × batch_size × dtype_bytes
```

For Mistral-7B at seq_len=4096, batch=8: **32GB KV cache alone** — double the model weights, exceeding a T4's 16GB VRAM. This is why PagedAttention (vLLM) matters: it allocates KV cache in 16-token pages on demand, reducing waste from ~65% to <4%.

## Architecture

```
inference/
├── naive_batching.py      # Sequential baseline — one request at a time
├── continuous_batching.py # Slot scheduler — fills capacity as requests finish
├── quantized_inference.py # FP16 / INT8 / INT4 NF4 via bitsandbytes
└── kv_cache_analysis.py   # Memory formulas, PagedAttention explanation
```

## Running Locally

```bash
git clone https://github.com/data-geek-astronomy/llm-inference-optimizer
cd llm-inference-optimizer
pip install -r requirements.txt

# Run with pre-computed benchmark dashboard
python app.py

# Enable live GPU benchmarking
ENABLE_LIVE_BENCHMARK=1 MODEL_NAME=gpt2 python app.py
```

## Key Learnings

**Why continuous batching is non-trivial to implement:**
Each request is at a different stage of token generation (different sequence lengths). Every forward pass must handle variable-length sequences in the same batch, requiring left-padding and careful attention mask management. Production systems (vLLM) also implement PagedAttention for the KV cache, which requires a custom CUDA kernel.

**Why NF4 works better than uniform INT4:**
Uniform quantization places bins at equal linear intervals. But LLM weights cluster near zero with a roughly normal distribution — most bins are wasted in the sparse tails. NF4 places bins at quantile positions of the standard normal, minimizing representation error where the weight density actually is.

**Why the memory cliff matters:**
At batch=8 and seq_len=4096, a 7B model needs more memory for KV cache than for its own weights. Without PagedAttention, you must reserve this memory upfront for the maximum possible sequence — leading to 60-70% VRAM waste. This is why vLLM achieves 24x higher throughput than naive HuggingFace serving.

## References

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM paper)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf) (continuous batching paper)

## License

MIT
