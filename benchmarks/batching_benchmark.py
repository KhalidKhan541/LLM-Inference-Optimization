import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    max_batch_size: int = 32
    max_wait_time_ms: float = 100.0
    max_tokens_per_batch: int = 4096
    continuous_batching: bool = True
    priority_scheduling: bool = False
    dynamic_batching: bool = True

@dataclass
class Request:
    id: str
    prompt: str
    max_tokens: int
    priority: int = 0
    arrival_time: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def latency(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.arrival_time
        return 0.0
    
    @property
    def queue_time(self) -> float:
        if self.start_time:
            return self.start_time - self.arrival_time
        return 0.0

@dataclass
class BatchResult:
    batch_size: int
    total_time_s: float
    total_tokens_generated: int
    tokens_per_second: float
    requests_per_second: float
    avg_latency_per_request_ms: float
    avg_queue_time_ms: float
    gpu_utilization_pct: float
    memory_usage_gb: float
    batching_strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class StaticBatcher:
    def __init__(self, config: BatchConfig, inference_fn: Callable):
        self.config = config
        self.inference_fn = inference_fn
    
    def process_batch(self, requests: List[Request]) -> List[Dict[str, Any]]:
        prompts = [r.prompt for r in requests]
        max_tokens = max(r.max_tokens for r in requests)
        
        start_time = time.perf_counter()
        results = self.inference_fn(prompts, max_tokens=max_tokens)
        total_time = time.perf_counter() - start_time
        
        for req in requests:
            req.start_time = start_time
            req.end_time = time.perf_counter()
        
        return results

class DynamicBatcher:
    def __init__(self, config: BatchConfig, inference_fn: Callable):
        self.config = config
        self.inference_fn = inference_fn
        self.pending_requests: deque = deque()
        self._lock = threading.Lock()
    
    def add_request(self, request: Request):
        with self._lock:
            request.arrival_time = time.perf_counter()
            self.pending_requests.append(request)
    
    def get_ready_batch(self) -> Optional[List[Request]]:
        with self._lock:
            if not self.pending_requests:
                return None
            
            batch = []
            current_tokens = 0
            now = time.perf_counter()
            
            while self.pending_requests and len(batch) < self.config.max_batch_size:
                req = self.pending_requests[0]
                
                wait_time = (now - req.arrival_time) * 1000
                if wait_time >= self.config.max_wait_time_ms and batch:
                    break
                
                if current_tokens + req.max_tokens > self.config.max_batch_tokens_per_batch:
                    break
                
                batch.append(self.pending_requests.popleft())
                current_tokens += req.max_tokens
            
            return batch if batch else None
    
    def process_requests(self) -> List[BatchResult]:
        results = []
        
        while True:
            batch = self.get_ready_batch()
            if not batch:
                break
            
            prompts = [r.prompt for r in batch]
            max_tokens = max(r.max_tokens for r in batch)
            
            start_time = time.perf_counter()
            for r in batch:
                r.start_time = start_time
            
            inference_results = self.inference_fn(prompts, max_tokens=max_tokens)
            
            end_time = time.perf_counter()
            for r in batch:
                r.end_time = end_time
            
            total_time = end_time - start_time
            total_tokens = sum(r.get('completion_tokens', 0) for r in inference_results)
            
            batch_result = BatchResult(
                batch_size=len(batch),
                total_time_s=total_time,
                total_tokens_generated=total_tokens,
                tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
                requests_per_second=len(batch) / total_time if total_time > 0 else 0,
                avg_latency_per_request_ms=(total_time / len(batch)) * 1000,
                avg_queue_time_ms=np.mean([r.queue_time * 1000 for r in batch]),
                gpu_utilization_pct=0.0,
                memory_usage_gb=0.0,
                batching_strategy='dynamic'
            )
            results.append(batch_result)
        
        return results

class ContinuousBatcher:
    def __init__(self, config: BatchConfig, inference_fn: Callable):
        self.config = config
        self.inference_fn = inference_fn
        self.pending_requests: deque = deque()
        self.active_batch: List[Request] = []
        self._lock = threading.Lock()
        self._running = False
    
    def add_request(self, request: Request):
        with self._lock:
            request.arrival_time = time.perf_counter()
            self.pending_requests.append(request)
    
    def _can_add_to_batch(self, request: Request, current_tokens: int) -> bool:
        return current_tokens + request.max_tokens <= self.config.max_tokens_per_batch
    
    def process_continuous(self, total_time_s: float = 10.0) -> List[BatchResult]:
        results = []
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < total_time_s:
            with self._lock:
                while (self.pending_requests and 
                       len(self.active_batch) < self.config.max_batch_size):
                    req = self.pending_requests[0]
                    current_tokens = sum(r.max_tokens for r in self.active_batch)
                    
                    if self._can_add_to_batch(req, current_tokens):
                        self.active_batch.append(self.pending_requests.popleft())
                    else:
                        break
            
            if not self.active_batch:
                time.sleep(0.001)
                continue
            
            batch_start = time.perf_counter()
            for req in self.active_batch:
                req.start_time = batch_start
            
            prompts = [r.prompt for r in self.active_batch]
            max_tokens = max(r.max_tokens for r in self.active_batch)
            
            inference_results = self.inference_fn(prompts, max_tokens=max_tokens)
            
            batch_end = time.perf_counter()
            for req in self.active_batch:
                req.end_time = batch_end
            
            total_time = batch_end - batch_start
            total_tokens = sum(r.get('completion_tokens', 0) for r in inference_results)
            
            batch_result = BatchResult(
                batch_size=len(self.active_batch),
                total_time_s=total_time,
                total_tokens_generated=total_tokens,
                tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
                requests_per_second=len(self.active_batch) / total_time if total_time > 0 else 0,
                avg_latency_per_request_ms=(total_time / len(self.active_batch)) * 1000,
                avg_queue_time_ms=np.mean([r.queue_time * 1000 for r in self.active_batch]),
                gpu_utilization_pct=0.0,
                memory_usage_gb=0.0,
                batching_strategy='continuous'
            )
            results.append(batch_result)
            
            self.active_batch = []
        
        return results

class BatchingBenchmark:
    def __init__(self, config: BatchConfig, inference_fn: Callable, output_dir: str = "results"):
        self.config = config
        self.inference_fn = inference_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BatchResult] = []
    
    def benchmark_static_batching(self, prompts: List[str], 
                                 batch_sizes: List[int] = None) -> List[BatchResult]:
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]
        
        results = []
        
        for batch_size in batch_sizes:
            if batch_size > len(prompts):
                batch_size = len(prompts)
            
            batch_prompts = prompts[:batch_size]
            requests = [
                Request(id=f"req_{i}", prompt=p, max_tokens=128)
                for i, p in enumerate(batch_prompts)
            ]
            
            batcher = StaticBatcher(self.config, self.inference_fn)
            
            mem_before = psutil.Process().memory_info().rss / (1024 ** 3)
            gpu_before = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
            
            batcher.process_batch(requests)
            
            mem_after = psutil.Process().memory_info().rss / (1024 ** 3)
            gpu_after = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
            
            total_time = sum(r.latency for r in requests)
            total_tokens = sum(r.max_tokens for r in requests)
            
            result = BatchResult(
                batch_size=batch_size,
                total_time_s=total_time,
                total_tokens_generated=total_tokens,
                tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
                requests_per_second=batch_size / total_time if total_time > 0 else 0,
                avg_latency_per_request_ms=(total_time / batch_size) * 1000,
                avg_queue_time_ms=np.mean([r.queue_time * 1000 for r in requests]),
                gpu_utilization_pct=0.0,
                memory_usage_gb=mem_after - mem_before,
                batching_strategy='static'
            )
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def benchmark_dynamic_batching(self, prompts: List[str], 
                                  num_concurrent: int = 8) -> List[BatchResult]:
        batcher = DynamicBatcher(self.config, self.inference_fn)
        
        threads = []
        for i, prompt in enumerate(prompts[:num_concurrent]):
            req = Request(id=f"req_{i}", prompt=prompt, max_tokens=128)
            thread = threading.Thread(target=batcher.add_request, args=(req,))
            threads.append(thread)
            thread.start()
            time.sleep(0.01)
        
        for thread in threads:
            thread.join()
        
        time.sleep(0.1)
        
        results = batcher.process_requests()
        self.results.extend(results)
        return results
    
    def benchmark_continuous_batching(self, prompts: List[str], 
                                     duration_s: float = 5.0) -> List[BatchResult]:
        batcher = ContinuousBatcher(self.config, self.inference_fn)
        
        def add_requests():
            idx = 0
            start = time.perf_counter()
            while time.perf_counter() - start < duration_s:
                req = Request(
                    id=f"req_{idx}",
                    prompt=prompts[idx % len(prompts)],
                    max_tokens=128
                )
                batcher.add_request(req)
                idx += 1
                time.sleep(0.01)
        
        request_thread = threading.Thread(target=add_requests)
        request_thread.start()
        
        results = batcher.process_continuous(total_time_s=duration_s)
        
        request_thread.join()
        
        self.results.extend(results)
        return results
    
    def run_full_benchmark(self, prompts: List[str]) -> Dict[str, List[BatchResult]]:
        logger.info("Running full batching strategy benchmark")
        
        static_results = self.benchmark_static_batching(prompts)
        dynamic_results = self.benchmark_dynamic_batching(prompts)
        continuous_results = self.benchmark_continuous_batching(prompts)
        
        return {
            'static': static_results,
            'dynamic': dynamic_results,
            'continuous': continuous_results
        }
    
    def save_results(self, filename: str = None):
        if filename is None:
            filename = "batching_strategy_results.json"
        
        filepath = self.output_dir / filename
        
        results_data = {
            'config': {
                'max_batch_size': self.config.max_batch_size,
                'max_wait_time_ms': self.config.max_wait_time_ms,
                'max_tokens_per_batch': self.config.max_tokens_per_batch,
                'continuous_batching': self.config.continuous_batching
            },
            'results': [
                {
                    'batch_size': r.batch_size,
                    'total_time_s': r.total_time_s,
                    'total_tokens_generated': r.total_tokens_generated,
                    'tokens_per_second': r.tokens_per_second,
                    'requests_per_second': r.requests_per_second,
                    'avg_latency_per_request_ms': r.avg_latency_per_request_ms,
                    'avg_queue_time_ms': r.avg_queue_time_ms,
                    'gpu_utilization_pct': r.gpu_utilization_pct,
                    'memory_usage_gb': r.memory_usage_gb,
                    'batching_strategy': r.batching_strategy
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Batching results saved to {filepath}")
        return filepath
    
    def generate_comparison_table(self) -> str:
        header = "| Strategy | Batch Size | Throughput (tok/s) | Requests/s | Avg Latency (ms) | Avg Queue (ms) |"
        separator = "|----------|------------|-------------------|------------|------------------|----------------|"
        
        rows = []
        for r in self.results:
            row = (
                f"| {r.batching_strategy:10} | {r.batch_size:10} | "
                f"{r.tokens_per_second:17.1f} | {r.requests_per_second:10.1f} | "
                f"{r.avg_latency_per_request_ms:16.1f} | {r.avg_queue_time_ms:14.1f} |"
            )
            rows.append(row)
        
        table = "\n".join([header, separator] + rows)
        return table