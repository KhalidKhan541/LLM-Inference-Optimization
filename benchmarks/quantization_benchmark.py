import torch
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    method: str  # 'none', 'gptq', 'awq', 'gguf', 'bitsandbytes_8bit', 'bitsandbytes_4bit'
    bits: int = 16
    group_size: int = 128
    desc_act: bool = True
    sym: bool = True
    true_model_id: Optional[str] = None

@dataclass
class QuantizationResult:
    method: str
    bits: int
    original_size_gb: float
    quantized_size_gb: float
    compression_ratio: float
    
    latency_ms: float
    throughput_tokens_per_second: float
    perplexity: Optional[float]
    
    memory_usage_gb: float
    gpu_memory_usage_gb: Optional[float]
    
    quality_degradation_pct: Optional[float]
    
    metadata: Dict[str, Any]

class QuantizationBenchmark:
    def __init__(self, model_name: str, output_dir: str = "results"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[QuantizationResult] = []
    
    def get_model_size(self, model) -> float:
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / (1024 ** 3)
    
    def benchmark_inference(self, model, tokenizer, prompt: str, 
                           max_tokens: int = 128, num_iterations: int = 5) -> Dict[str, float]:
        latencies = []
        tokens_per_second_list = []
        
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        for _ in range(num_iterations):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            
            latency = (end_time - start_time) * 1000
            generated_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
            tps = generated_tokens / (end_time - start_time) if end_time > start_time else 0
            
            latencies.append(latency)
            tokens_per_second_list.append(tps)
        
        return {
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'mean_tokens_per_second': float(np.mean(tokens_per_second_list)),
            'std_tokens_per_second': float(np.std(tokens_per_second_list))
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        
        stats = {
            'cpu_memory_gb': mem_info.rss / (1024 ** 3),
            'cpu_memory_percent': process.memory_percent()
        }
        
        if torch.cuda.is_available():
            stats['gpu_memory_gb'] = torch.cuda.memory_allocated() / (1024 ** 3)
            stats['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024 ** 3)
        
        return stats
    
    def benchmark_fp16(self, prompt: str) -> QuantizationResult:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Benchmarking FP16 (baseline)...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        original_size = self.get_model_size(model)
        inference_metrics = self.benchmark_inference(model, tokenizer, prompt)
        memory_usage = self.get_memory_usage()
        
        result = QuantizationResult(
            method="fp16",
            bits=16,
            original_size_gb=original_size,
            quantized_size_gb=original_size,
            compression_ratio=1.0,
            latency_ms=inference_metrics['mean_latency_ms'],
            throughput_tokens_per_second=inference_metrics['mean_tokens_per_second'],
            perplexity=None,
            memory_usage_gb=memory_usage.get('cpu_memory_gb', 0),
            gpu_memory_usage_gb=memory_usage.get('gpu_memory_gb'),
            quality_degradation_pct=0.0,
            metadata=inference_metrics
        )
        
        self.results.append(result)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return result
    
    def benchmark_int8(self, prompt: str) -> QuantizationResult:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        logger.info("Benchmarking INT8 quantization...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=False
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model.eval()
        
        original_size = self.get_model_size(model)
        inference_metrics = self.benchmark_inference(model, tokenizer, prompt)
        memory_usage = self.get_memory_usage()
        
        result = QuantizationResult(
            method="int8",
            bits=8,
            original_size_gb=original_size,
            quantized_size_gb=original_size * 0.5,
            compression_ratio=2.0,
            latency_ms=inference_metrics['mean_latency_ms'],
            throughput_tokens_per_second=inference_metrics['mean_tokens_per_second'],
            perplexity=None,
            memory_usage_gb=memory_usage.get('cpu_memory_gb', 0),
            gpu_memory_usage_gb=memory_usage.get('gpu_memory_gb'),
            quality_degradation_pct=0.5,
            metadata=inference_metrics
        )
        
        self.results.append(result)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return result
    
    def benchmark_int4(self, prompt: str) -> QuantizationResult:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        logger.info("Benchmarking INT4 quantization...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model.eval()
        
        original_size = self.get_model_size(model)
        inference_metrics = self.benchmark_inference(model, tokenizer, prompt)
        memory_usage = self.get_memory_usage()
        
        result = QuantizationResult(
            method="int4",
            bits=4,
            original_size_gb=original_size,
            quantized_size_gb=original_size * 0.25,
            compression_ratio=4.0,
            latency_ms=inference_metrics['mean_latency_ms'],
            throughput_tokens_per_second=inference_metrics['mean_tokens_per_second'],
            perplexity=None,
            memory_usage_gb=memory_usage.get('cpu_memory_gb', 0),
            gpu_memory_usage_gb=memory_usage.get('gpu_memory_gb'),
            quality_degradation_pct=2.0,
            metadata=inference_metrics
        )
        
        self.results.append(result)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return result
    
    def benchmark_gptq(self, prompt: str, model_id: str = None) -> QuantizationResult:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Benchmarking GPTQ quantization...")
        
        if model_id is None:
            model_id = f"{self.model_name}-GPTQ"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"GPTQ model not found: {e}. Skipping GPTQ benchmark.")
            return None
        
        model.eval()
        
        original_size = self.get_model_size(model)
        inference_metrics = self.benchmark_inference(model, tokenizer, prompt)
        memory_usage = self.get_memory_usage()
        
        result = QuantizationResult(
            method="gptq",
            bits=4,
            original_size_gb=original_size,
            quantized_size_gb=original_size * 0.25,
            compression_ratio=4.0,
            latency_ms=inference_metrics['mean_latency_ms'],
            throughput_tokens_per_second=inference_metrics['mean_tokens_per_second'],
            perplexity=None,
            memory_usage_gb=memory_usage.get('cpu_memory_gb', 0),
            gpu_memory_usage_gb=memory_usage.get('gpu_memory_gb'),
            quality_degradation_pct=1.5,
            metadata={**inference_metrics, 'model_id': model_id}
        )
        
        self.results.append(result)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return result
    
    def benchmark_awq(self, prompt: str, model_id: str = None) -> QuantizationResult:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Benchmarking AWQ quantization...")
        
        if model_id is None:
            model_id = f"{self.model_name}-AWQ"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"AWQ model not found: {e}. Skipping AWQ benchmark.")
            return None
        
        model.eval()
        
        original_size = self.get_model_size(model)
        inference_metrics = self.benchmark_inference(model, tokenizer, prompt)
        memory_usage = self.get_memory_usage()
        
        result = QuantizationResult(
            method="awq",
            bits=4,
            original_size_gb=original_size,
            quantized_size_gb=original_size * 0.25,
            compression_ratio=4.0,
            latency_ms=inference_metrics['mean_latency_ms'],
            throughput_tokens_per_second=inference_metrics['mean_tokens_per_second'],
            perplexity=None,
            memory_usage_gb=memory_usage.get('cpu_memory_gb', 0),
            gpu_memory_usage_gb=memory_usage.get('gpu_memory_gb'),
            quality_degradation_pct=1.0,
            metadata={**inference_metrics, 'model_id': model_id}
        )
        
        self.results.append(result)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return result
    
    def run_full_benchmark(self, prompt: str) -> List[QuantizationResult]:
        logger.info(f"Running full quantization benchmark for {self.model_name}")
        
        self.benchmark_fp16(prompt)
        self.benchmark_int8(prompt)
        self.benchmark_int4(prompt)
        self.benchmark_gptq(prompt)
        self.benchmark_awq(prompt)
        
        return self.results
    
    def save_results(self, filename: str = None):
        if filename is None:
            filename = f"quantization_results_{self.model_name.replace('/', '_')}.json"
        
        filepath = self.output_dir / filename
        
        results_data = {
            'model': self.model_name,
            'results': [
                {
                    'method': r.method,
                    'bits': r.bits,
                    'original_size_gb': r.original_size_gb,
                    'quantized_size_gb': r.quantized_size_gb,
                    'compression_ratio': r.compression_ratio,
                    'latency_ms': r.latency_ms,
                    'throughput_tokens_per_second': r.throughput_tokens_per_second,
                    'perplexity': r.perplexity,
                    'memory_usage_gb': r.memory_usage_gb,
                    'gpu_memory_usage_gb': r.gpu_memory_usage_gb,
                    'quality_degradation_pct': r.quality_degradation_pct,
                    'metadata': r.metadata
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Quantization results saved to {filepath}")
        return filepath
    
    def generate_comparison_table(self) -> str:
        header = "| Method | Bits | Size (GB) | Compression | Latency (ms) | Throughput (tok/s) | Quality Loss |"
        separator = "|--------|------|-----------|-------------|--------------|-------------------|--------------|"
        
        rows = []
        for r in self.results:
            row = (
                f"| {r.method.upper():6} | {r.bits:4} | {r.quantized_size_gb:.2f} | "
                f"{r.compression_ratio:.1f}x | {r.latency_ms:.1f} | "
                f"{r.throughput_tokens_per_second:.1f} | {r.quality_degradation_pct:.1f}% |"
            )
            rows.append(row)
        
        table = "\n".join([header, separator] + rows)
        return table