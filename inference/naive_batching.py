"""
Naive Batching: process one request at a time, no concurrency.
This is the baseline — every LLM serving system starts here.
"""

import time
import torch
import numpy as np
from dataclasses import dataclass
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class InferenceResult:
    prompt: str
    output: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    tokens_per_second: float


class NaiveBatchingEngine:
    """
    Sequential inference: each request waits for the previous to complete.
    Problems:
    - GPU sits idle between requests
    - No sharing of KV cache computation
    - Latency scales linearly with queue depth
    """

    def __init__(self, model_name: str, device: str = "auto"):
        print(f"[NaiveBatching] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"[NaiveBatching] Model loaded on {self.device}")

    @torch.no_grad()
    def generate_single(self, prompt: str, max_new_tokens: int = 50) -> InferenceResult:
        """Generate for a single prompt, sequentially."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        output_len = output_ids.shape[1] - input_len
        output_text = self.tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )
        tps = (output_len / elapsed_ms) * 1000

        return InferenceResult(
            prompt=prompt,
            output=output_text,
            input_tokens=input_len,
            output_tokens=output_len,
            latency_ms=elapsed_ms,
            tokens_per_second=tps,
        )

    def benchmark(
        self, prompts: List[str], max_new_tokens: int = 50
    ) -> dict:
        """Run prompts sequentially and collect latency statistics."""
        results = []
        for i, prompt in enumerate(prompts):
            result = self.generate_single(prompt, max_new_tokens)
            results.append(result)
            print(f"  [{i+1}/{len(prompts)}] {result.latency_ms:.1f}ms, "
                  f"{result.tokens_per_second:.1f} tok/s")

        latencies = [r.latency_ms for r in results]
        tps_values = [r.tokens_per_second for r in results]
        total_time = sum(latencies)

        return {
            "method": "naive_sequential",
            "n_requests": len(prompts),
            "total_time_ms": total_time,
            "throughput_requests_per_sec": len(prompts) / (total_time / 1000),
            "throughput_tokens_per_sec": sum(r.output_tokens for r in results) / (total_time / 1000),
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "latency_mean_ms": float(np.mean(latencies)),
            "tokens_per_second_mean": float(np.mean(tps_values)),
            "results": results,
        }
