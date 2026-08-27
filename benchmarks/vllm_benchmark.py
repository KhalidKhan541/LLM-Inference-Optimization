import time
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

from .core import BenchmarkResult, BenchmarkTimer, MetricsCollector

logger = logging.getLogger(__name__)

@dataclass
class VLLMConfig:
    model: str = "meta-llama/Llama-2-7b-hf"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 256
    quantization: Optional[str] = None
    dtype: str = "auto"
    trust_remote_code: bool = False
    disable_log_stats: bool = True

class VLLMBenchmark:
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.metrics = MetricsCollector()
        self.engine = None
    
    def initialize(self):
        try:
            from vllm import LLM, SamplingParams
            self.LLM = LLM
            self.SamplingParams = SamplingParams
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
        
        logger.info(f"Initializing vLLM with model: {self.config.model}")
        
        kwargs = {
            'model': self.config.model,
            'tensor_parallel_size': self.config.tensor_parallel_size,
            'gpu_memory_utilization': self.config.gpu_memory_utilization,
            'max_num_batched_tokens': self.config.max_num_batched_tokens,
            'max_num_seqs': self.config.max_num_seqs,
            'dtype': self.config.dtype,
            'trust_remote_code': self.config.trust_remote_code,
            'disable_log_stats': self.config.disable_log_stats
        }
        
        if self.config.quantization:
            kwargs['quantization'] = self.config.quantization
        
        self.engine = self.LLM(**kwargs)
        logger.info("vLLM engine initialized successfully")
    
    def generate(self, prompt: str, max_tokens: int = 512, 
                temperature: float = 0.0, top_p: float = 1.0,
                **kwargs) -> Dict[str, Any]:
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        sampling_params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        timer = BenchmarkTimer().start()
        outputs = self.engine.generate([prompt], sampling_params)
        timer.stop()
        
        output = outputs[0]
        generated_text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids
        
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(token_ids)
        total_tokens = prompt_tokens + completion_tokens
        
        prefill_time = output.metrics.finished_time if hasattr(output.metrics, 'finished_time') else timer.elapsed
        num_generated_tokens = len(token_ids)
        
        stats = self.metrics.get_memory_stats()
        gpu_stats = self.metrics.get_gpu_stats()
        
        return {
            'generated_text': generated_text,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'latency_s': timer.elapsed,
            'tokens_per_second': num_generated_tokens / timer.elapsed if timer.elapsed > 0 else 0,
            'prefill_tokens_per_second': prompt_tokens / prefill_time if prefill_time > 0 else 0,
            'time_to_first_token_ms': None,
            'memory_stats': stats,
            'gpu_stats': gpu_stats
        }
    
    def generate_batch(self, prompts: List[str], max_tokens: int = 512,
                      temperature: float = 0.0, **kwargs) -> List[Dict[str, Any]]:
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        sampling_params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        timer = BenchmarkTimer().start()
        outputs = self.engine.generate(prompts, sampling_params)
        timer.stop()
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(token_ids)
            
            results.append({
                'generated_text': generated_text,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'latency_s': timer.elapsed / len(prompts),
                'tokens_per_second': completion_tokens / (timer.elapsed / len(prompts))
            })
        
        return results
    
    def benchmark_latency(self, prompt: str, num_iterations: int = 10,
                         max_tokens: int = 128) -> Dict[str, Any]:
        latencies = []
        tokens_per_second_list = []
        memory_samples = []
        
        for i in range(num_iterations):
            mem_before = self.metrics.get_memory_stats()
            
            result = self.generate(prompt, max_tokens=max_tokens)
            
            mem_after = self.metrics.get_memory_stats()
            
            latencies.append(result['latency_s'] * 1000)
            tokens_per_second_list.append(result['tokens_per_second'])
            memory_samples.append({
                'before': mem_before,
                'after': mem_after,
                'delta_gb': mem_after.get('rss_gb', 0) - mem_before.get('rss_gb', 0)
            })
        
        return {
            'backend': 'vllm',
            'model': self.config.model,
            'quantization': self.config.quantization or 'none',
            'latencies_ms': latencies,
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p90_latency_ms': float(np.percentile(latencies, 90)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'tokens_per_second': tokens_per_second_list,
            'mean_tokens_per_second': float(np.mean(tokens_per_second_list)),
            'memory_samples': memory_samples,
            'mean_memory_delta_gb': float(np.mean([s['delta_gb'] for s in memory_samples]))
        }
    
    def benchmark_throughput(self, prompts: List[str], batch_sizes: List[int] = None,
                            max_tokens: int = 128) -> Dict[str, Any]:
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]
        
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(prompts):
                batch_size = len(prompts)
            
            batch_prompts = prompts[:batch_size]
            
            mem_before = self.metrics.get_memory_stats()
            
            timer = BenchmarkTimer().start()
            batch_results = self.generate_batch(batch_prompts, max_tokens=max_tokens)
            timer.stop()
            
            mem_after = self.metrics.get_memory_stats()
            
            total_tokens = sum(r['completion_tokens'] for r in batch_results)
            total_latency = timer.elapsed
            
            results[batch_size] = {
                'batch_size': batch_size,
                'total_time_s': total_latency,
                'total_tokens': total_tokens,
                'tokens_per_second': total_tokens / total_latency if total_latency > 0 else 0,
                'requests_per_second': batch_size / total_latency if total_latency > 0 else 0,
                'avg_latency_per_request_ms': (total_latency / batch_size) * 1000,
                'memory_before': mem_before,
                'memory_after': mem_after,
                'memory_delta_gb': mem_after.get('rss_gb', 0) - mem_before.get('rss_gb', 0)
            }
        
        return results
    
    def benchmark_all(self, prompts: List[str], num_iterations: int = 5) -> Dict[str, Any]:
        logger.info(f"Running comprehensive vLLM benchmark for {self.config.model}")
        
        latency_results = self.benchmark_latency(
            prompts[0], 
            num_iterations=num_iterations,
            max_tokens=128
        )
        
        throughput_results = self.benchmark_throughput(
            prompts, 
            batch_sizes=[1, 2, 4, 8],
            max_tokens=128
        )
        
        memory_stats = self.metrics.get_memory_stats()
        gpu_stats = self.metrics.get_gpu_stats()
        
        return {
            'backend': 'vllm',
            'config': {
                'model': self.config.model,
                'tensor_parallel_size': self.config.tensor_parallel_size,
                'gpu_memory_utilization': self.config.gpu_memory_utilization,
                'quantization': self.config.quantization,
                'dtype': self.config.dtype
            },
            'latency': latency_results,
            'throughput': throughput_results,
            'memory': memory_stats,
            'gpu': gpu_stats
        }