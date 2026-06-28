"""
Continuous Batching: the key innovation behind vLLM and modern LLM serving.

Core insight: with naive batching, requests that finish early leave GPU idle
while waiting for the slowest request in the batch. Continuous batching
inserts new requests as soon as a slot opens — no idle GPU time.

This implementation is a pedagogical version that demonstrates the scheduling
logic without the full PagedAttention KV cache management.
"""

import time
import torch
import numpy as np
import threading
import queue
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class Request:
    id: int
    prompt: str
    max_new_tokens: int
    arrival_time: float = field(default_factory=time.perf_counter)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # State for iterative generation
    input_ids: Optional[torch.Tensor] = None
    generated_ids: List[int] = field(default_factory=list)
    finished: bool = False

    @property
    def waiting_time_ms(self):
        if self.start_time:
            return (self.start_time - self.arrival_time) * 1000
        return None

    @property
    def latency_ms(self):
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    @property
    def tokens_generated(self):
        return len(self.generated_ids)


class ContinuousBatchingEngine:
    """
    Continuous batching scheduler:
    - Maintains a queue of pending requests
    - Groups in-flight requests into batches for each forward pass
    - Immediately inserts new requests when capacity allows
    - No waiting for the slowest request before accepting new ones

    This is how vLLM, TGI, and TensorRT-LLM serve LLMs at scale.
    """

    def __init__(
        self,
        model_name: str,
        max_batch_size: int = 8,
        device: str = "auto",
    ):
        print(f"[ContinuousBatching] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Left-pad for decoder-only models

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_batch_size = max_batch_size
        print(f"[ContinuousBatching] Ready on {self.device}, max_batch={max_batch_size}")

    @torch.no_grad()
    def _forward_batch(self, active_requests: List[Request]) -> List[int]:
        """Single forward pass over a batch of in-flight requests."""
        # Build batch — each request at a different stage of generation
        input_ids_list = []
        for req in active_requests:
            if req.start_time is None:
                # First token: encode the full prompt
                req.start_time = time.perf_counter()
                ids = self.tokenizer.encode(req.prompt, return_tensors="pt")[0]
                req.input_ids = ids
            else:
                # Subsequent tokens: append last generated token
                last_token = torch.tensor([req.generated_ids[-1]])
                req.input_ids = torch.cat([req.input_ids, last_token])
            input_ids_list.append(req.input_ids)

        # Pad to same length for batched forward pass
        max_len = max(ids.shape[0] for ids in input_ids_list)
        padded = []
        attention_masks = []
        for ids in input_ids_list:
            pad_len = max_len - ids.shape[0]
            padded_ids = torch.cat([
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long),
                ids
            ])
            mask = torch.cat([torch.zeros(pad_len), torch.ones(ids.shape[0])]).long()
            padded.append(padded_ids)
            attention_masks.append(mask)

        input_batch = torch.stack(padded).to(self.device)
        mask_batch = torch.stack(attention_masks).to(self.device)

        outputs = self.model(input_ids=input_batch, attention_mask=mask_batch)
        next_token_logits = outputs.logits[:, -1, :]  # [batch, vocab]
        next_tokens = next_token_logits.argmax(dim=-1).tolist()  # greedy
        return next_tokens

    def process_requests(self, requests: List[Request]) -> List[Request]:
        """
        Main continuous batching loop.
        Processes requests in overlapping batches — finished requests
        are replaced immediately rather than waiting for the whole batch.
        """
        pending = list(requests)
        active: List[Request] = []
        completed: List[Request] = []

        total_forward_passes = 0

        while pending or active:
            # Fill up to max_batch_size from the pending queue
            while pending and len(active) < self.max_batch_size:
                active.append(pending.pop(0))

            if not active:
                break

            # One forward pass over all active requests
            next_tokens = self._forward_batch(active)
            total_forward_passes += 1

            # Update each request with its new token
            still_active = []
            for req, token_id in zip(active, next_tokens):
                req.generated_ids.append(token_id)

                is_eos = (token_id == self.tokenizer.eos_token_id)
                is_max = (req.tokens_generated >= req.max_new_tokens)

                if is_eos or is_max:
                    req.end_time = time.perf_counter()
                    req.finished = True
                    completed.append(req)
                    # Key: slot immediately available for next pending request
                else:
                    still_active.append(req)

            active = still_active

        return completed

    def benchmark(self, prompts: List[str], max_new_tokens: int = 50) -> dict:
        requests = [
            Request(id=i, prompt=p, max_new_tokens=max_new_tokens)
            for i, p in enumerate(prompts)
        ]

        wall_start = time.perf_counter()
        completed = self.process_requests(requests)
        wall_elapsed_ms = (time.perf_counter() - wall_start) * 1000

        latencies = [r.latency_ms for r in completed if r.latency_ms]
        total_tokens = sum(r.tokens_generated for r in completed)

        for r in completed:
            text = self.tokenizer.decode(r.generated_ids, skip_special_tokens=True)
            print(f"  [req {r.id}] {r.latency_ms:.1f}ms | '{text[:60]}'")

        return {
            "method": "continuous_batching",
            "n_requests": len(prompts),
            "max_batch_size": self.max_batch_size,
            "total_time_ms": wall_elapsed_ms,
            "throughput_requests_per_sec": len(prompts) / (wall_elapsed_ms / 1000),
            "throughput_tokens_per_sec": total_tokens / (wall_elapsed_ms / 1000),
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "latency_mean_ms": float(np.mean(latencies)),
            "tokens_per_second_mean": total_tokens / (wall_elapsed_ms / 1000),
            "completed": completed,
        }
