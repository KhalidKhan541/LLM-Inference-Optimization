import time
import torch
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

from .core import BenchmarkResult, BenchmarkTimer, MetricsCollector

logger = logging.getLogger(__name__)

@dataclass
class HFConfig:
    model: str = "meta-llama/Llama-2-7b-hf"
    device: str = "auto"
    torch_dtype: str = "float16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    quantization_config: Optional[Dict[str, Any]] = None
    attn_implementation: Optional[str] = None
    trust_remote_code: bool = False
    pad_token_id: Optional[int] = None

class HFBenchmark:
    def __init__(self, config: HFConfig):
        self.config = config
        self.metrics = MetricsCollector()
        self.model = None
        self.tokenizer = None
    
    def initialize(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Initializing HuggingFace model: {self.config.model}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            trust_remote_code=self.config.trust_remote_code
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {
            'pretrained_model_name_or_path': self.config.model,
            'torch_dtype': getattr(torch, self.config.torch_dtype, torch.float16),
            'trust_remote_code': self.config.trust_remote_code
        }
        
        if self.config.device == "auto":
            model_kwargs['device_map'] = "auto"
        else:
            model_kwargs['device_map'] = self.config.device
        
        if self.config.load_in_8bit:
            model_kwargs['load_in_8bit'] = True
        elif self.config.load_in_4bit:
            model_kwargs['load_in_4bit'] = True
            if self.config.quantization_config:
                from transformers import BitsAndBytesConfig
                model_kwargs['quantization_config'] = BitsAndBytesConfig(**self.config.quantization_config)
        
        if self.config.attn_implementation:
            model_kwargs['attn_implementation'] = self.config.attn_implementation
        
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        self.model.eval()
        
        logger.info("HuggingFace model initialized successfully")
    
    def generate(self, prompt: str, max_tokens: int = 512,
                temperature: float = 0.0, top_p: float = 1.0,
                do_sample: bool = False, **kwargs) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if self.config.device != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        prompt_tokens = inputs['input_ids'].shape[1]
        
        timer = BenchmarkTimer().start()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        timer.stop()
        
        generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        completion_tokens = len(generated_ids)
        total_tokens = prompt_tokens + completion_tokens
        
        stats = self.metrics.get_memory_stats()
        gpu_stats = self.metrics.get_gpu_stats()
        
        return {
            'generated_text': generated_text,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'latency_s': timer.elapsed,
            'tokens_per_second': completion_tokens / timer.elapsed if timer.elapsed > 0 else 0,
            'prefill_tokens_per_second': prompt_tokens / timer.elapsed if timer.elapsed > 0 else 0,
            'time_to_first_token_ms': None,
            'memory_stats': stats,
            'gpu_stats': gpu_stats
        }
    
    def generate_with_streaming(self, prompt: str, max_tokens: int = 512,
                               temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
        from transformers import TextIteratorStreamer
        import threading
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.config.device != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = {
            **inputs,
            'max_new_tokens': max_tokens,
            'temperature': temperature if temperature > 0 else None,
            'do_sample': temperature > 0,
            'streamer': streamer,
            'pad_token_id': self.tokenizer.pad_token_id,
            **kwargs
        }
        
        timer = BenchmarkTimer().start()
        
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        generated_text = ""
        token_count = 0
        first_token_time = None
        
        for new_text in streamer:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            generated_text += new_text
            token_count += 1
        
        thread.join()
        timer.stop()
        
        ttft = (first_token_time - timer.start_time) * 1000 if first_token_time else None
        
        stats = self.metrics.get_memory_stats()
        gpu_stats = self.metrics.get_gpu_stats()
        
        return {
            'generated_text': generated_text,
            'prompt_tokens': inputs['input_ids'].shape[1],
            'completion_tokens': token_count,
            'total_tokens': inputs['input_ids'].shape[1] + token_count,
            'latency_s': timer.elapsed,
            'tokens_per_second': token_count / timer.elapsed if timer.elapsed > 0 else 0,
            'time_to_first_token_ms': ttft,
            'memory_stats': stats,
            'gpu_stats': gpu_stats
        }
    
    def benchmark_latency(self, prompt: str, num_iterations: int = 10,
                         max_tokens: int = 128) -> Dict[str, Any]:
        import numpy as np
        
        latencies = []
        tokens_per_second_list = []
        memory_samples = []
        ttft_samples = []
        
        for i in range(num_iterations):
            mem_before = self.metrics.get_memory_stats()
            
            result = self.generate_with_streaming(prompt, max_tokens=max_tokens)
            
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
            'backend': 'huggingface',
            'model': self.config.model,
            'quantization': '8bit' if self.config.load_in_8bit else ('4bit' if self.config.load_in_4bit else 'none'),
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
        import numpy as np
        
        if batch_sizes is None:
            batch_sizes = [1, 2, 4]
        
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(prompts):
                batch_size = len(prompts)
            
            batch_prompts = prompts[:batch_size]
            
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            if self.config.device != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            mem_before = self.metrics.get_memory_stats()
            
            timer = BenchmarkTimer().start()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            timer.stop()
            
            mem_after = self.metrics.get_memory_stats()
            
            total_tokens = sum(outputs.shape[1] - inputs['input_ids'].shape[1] for outputs in [outputs])
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
        import numpy as np
        
        logger.info(f"Running comprehensive HuggingFace benchmark for {self.config.model}")
        
        latency_results = self.benchmark_latency(
            prompts[0], 
            num_iterations=num_iterations,
            max_tokens=128
        )
        
        throughput_results = self.benchmark_throughput(
            prompts, 
            batch_sizes=[1, 2],
            max_tokens=128
        )
        
        memory_stats = self.metrics.get_memory_stats()
        gpu_stats = self.metrics.get_gpu_stats()
        
        return {
            'backend': 'huggingface',
            'config': {
                'model': self.config.model,
                'device': self.config.device,
                'torch_dtype': self.config.torch_dtype,
                'load_in_8bit': self.config.load_in_8bit,
                'load_in_4bit': self.config.load_in_4bit,
                'attn_implementation': self.config.attn_implementation
            },
            'latency': latency_results,
            'throughput': throughput_results,
            'memory': memory_stats,
            'gpu': gpu_stats
        }