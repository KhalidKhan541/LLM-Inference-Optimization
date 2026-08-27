import time
import json
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

from .core import BenchmarkResult, BenchmarkTimer, MetricsCollector

logger = logging.getLogger(__name__)

@dataclass
class TGIConfig:
    model: str = "meta-llama/Llama-2-7b-hf"
    hostname: str = "localhost"
    port: int = 8080
    max_batch_prefill_tokens: int = 4096
    max_batch_total_tokens: int = 8192
    max_concurrent_requests: int = 64
    max_waiting_tokens: int = 20
    quantize: Optional[str] = None
    dtype: str = "float16"
    sharded: bool = False
    num_shard: int = 1
    use_flash_attention: bool = True

class TGIBenchmark:
    def __init__(self, config: TGIConfig):
        self.config = config
        self.metrics = MetricsCollector()
        self.base_url = f"http://{config.hostname}:{config.port}"
        self.session = requests.Session()
    
    def check_health(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"TGI health check failed: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 512,
                temperature: float = 0.0, top_p: float = 1.0,
                do_sample: bool = False, **kwargs) -> Dict[str, Any]:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                **kwargs
            }
        }
        
        timer = BenchmarkTimer().start()
        response = self.session.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=300
        )
        timer.stop()
        
        if response.status_code != 200:
            raise RuntimeError(f"TGI generation failed: {response.text}")
        
        result = response.json()
        
        stats = self.metrics.get_memory_stats()
        gpu_stats = self.metrics.get_gpu_stats()
        
        generated_text = result.get('generated_text', '')
        token_count = len(generated_text.split()) * 1.3
        
        return {
            'generated_text': generated_text,
            'prompt_tokens': len(prompt.split()) * 1.3,
            'completion_tokens': token_count,
            'total_tokens': len(prompt.split()) * 1.3 + token_count,
            'latency_s': timer.elapsed,
            'tokens_per_second': token_count / timer.elapsed if timer.elapsed > 0 else 0,
            'prefill_tokens_per_second': 0,
            'time_to_first_token_ms': None,
            'memory_stats': stats,
            'gpu_stats': gpu_stats
        }
    
    def generate_stream(self, prompt: str, max_tokens: int = 512,
                       temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                **kwargs
            }
        }
        
        timer = BenchmarkTimer().start()
        response = self.session.post(
            f"{self.base_url}/generate_stream",
            json=payload,
            stream=True,
            timeout=300
        )
        
        first_token_time = None
        generated_text = ""
        token_count = 0
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'token' in data:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        generated_text += data['token']['text']
                        token_count += 1
                except json.JSONDecodeError:
                    continue
        
        timer.stop()
        
        ttft = (first_token_time - timer.start_time) * 1000 if first_token_time else None
        
        stats = self.metrics.get_memory_stats()
        gpu_stats = self.metrics.get_gpu_stats()
        
        return {
            'generated_text': generated_text,
            'prompt_tokens': len(prompt.split()) * 1.3,
            'completion_tokens': token_count,
            'total_tokens': len(prompt.split()) * 1.3 + token_count,
            'latency_s': timer.elapsed,
            'tokens_per_second': token_count / timer.elapsed if timer.elapsed > 0 else 0,
            'time_to_first_token_ms': ttft,
            'memory_stats': stats,
            'gpu_stats': gpu_stats
        }
    
    def generate_batch(self, prompts: List[str], max_tokens: int = 512,
                      temperature: float = 0.0, **kwargs) -> List[Dict[str, Any]]:
        payload = {
            "inputs": prompts,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                **kwargs
            }
        }
        
        timer = BenchmarkTimer().start()
        response = self.session.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=600
        )
        timer.stop()
        
        if response.status_code != 200:
            raise RuntimeError(f"TGI batch generation failed: {response.text}")
        
        results = response.json()
        if not isinstance(results, list):
            results = [results]
        
        batch_results = []
        for i, result in enumerate(results):
            generated_text = result.get('generated_text', '')
            token_count = len(generated_text.split()) * 1.3
            
            batch_results.append({
                'generated_text': generated_text,
                'prompt_tokens': len(prompts[i].split()) * 1.3,
                'completion_tokens': token_count,
                'total_tokens': len(prompts[i].split()) * 1.3 + token_count,
                'latency_s': timer.elapsed / len(prompts),
                'tokens_per_second': token_count / (timer.elapsed / len(prompts))
            })
        
        return batch_results
    
    def benchmark_latency(self, prompt: str, num_iterations: int = 10,
                         max_tokens: int = 128) -> Dict[str, Any]:
        latencies = []
        tokens_per_second_list = []
        memory_samples = []
        ttft_samples = []
        
        for i in range(num_iterations):
            mem_before = self.metrics.get_memory_stats()
            
            result = self.generate_stream(prompt, max_tokens=max_tokens)
            
            mem_after = self.metrics.get_memory_stats()
            
            latencies.append(result['latency_s'] * 1000)
            tokens_per_second_list.append(result['tokens_per_second'])
            if result['time_to_first_token_ms']:
                ttft_samples.append(result['time_to_first_token_ms'])
            memory_samples.append({
                'before': mem_before,
                'after': mem_after,
                'delta_gb': mem_after.get('rss_gb', 0) - mem_before.get('rss_gb', 0)
            })
        
        return {
            'backend': 'tgi',
            'model': self.config.model,
            'quantization': self.config.quantize or 'none',
            'latencies_ms': latencies,
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p90_latency_ms': float(np.percentile(latencies, 90)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'tokens_per_second': tokens_per_second_list,
            'mean_tokens_per_second': float(np.mean(tokens_per_second_list)),
            'time_to_first_token_ms': ttft_samples,
            'mean_ttft_ms': float(np.mean(ttft_samples)) if ttft_samples else None,
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
        logger.info(f"Running comprehensive TGI benchmark for {self.config.model}")
        
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
            'backend': 'tgi',
            'config': {
                'model': self.config.model,
                'hostname': self.config.hostname,
                'port': self.config.port,
                'max_batch_prefill_tokens': self.config.max_batch_prefill_tokens,
                'max_concurrent_requests': self.config.max_concurrent_requests,
                'quantize': self.config.quantize,
                'dtype': self.config.dtype
            },
            'latency': latency_results,
            'throughput': throughput_results,
            'memory': memory_stats,
            'gpu': gpu_stats
        }