import time
import json
import psutil
import threading
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    backend: str
    model_name: str
    quantization: str
    batch_size: int
    input_tokens: int
    output_tokens: int
    
    total_time_s: float
    prefill_time_s: float
    decode_time_s: float
    
    tokens_per_second: float
    prefill_tokens_per_second: float
    
    latency_ms: float
    time_to_first_token_ms: float
    
    peak_memory_gb: float
    avg_memory_gb: float
    memory_utilization_pct: float
    
    gpu_memory_gb: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    
    throughput_requests_per_second: float = 0.0
    concurrent_requests: int = 1
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class MetricsCollector:
    def __init__(self):
        self._lock = threading.Lock()
        self._gpu_available = False
        self._init_gpu_monitoring()
    
    def _init_gpu_monitoring(self):
        try:
            import torch
            self._gpu_available = torch.cuda.is_available()
        except ImportError:
            self._gpu_available = False
    
    def get_memory_stats(self) -> Dict[str, float]:
        process = psutil.Process()
        mem_info = process.memory_info()
        
        stats = {
            'rss_gb': mem_info.rss / (1024 ** 3),
            'vms_gb': mem_info.vms / (1024 ** 3),
            'percent': process.memory_percent(),
            'available_gb': psutil.virtual_memory().available / (1024 ** 3)
        }
        
        if self._gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    stats['gpu_memory_used_gb'] = torch.cuda.memory_allocated() / (1024 ** 3)
                    stats['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024 ** 3)
                    stats['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            except Exception:
                pass
        
        return stats
    
    def get_gpu_stats(self) -> Optional[Dict[str, float]]:
        if not self._gpu_available:
            return None
        
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'gpu_memory_used_gb': torch.cuda.memory_allocated() / (1024 ** 3),
                    'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024 ** 3),
                    'gpu_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
                }
        except Exception:
            pass
        
        return None

class BenchmarkTimer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.splits = {}
    
    def start(self):
        self.start_time = time.perf_counter()
        return self
    
    def split(self, name: str):
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        self.splits[name] = time.perf_counter()
        return self
    
    def stop(self):
        self.end_time = time.perf_counter()
        return self
    
    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.perf_counter()
        return end - self.start_time
    
    def get_split(self, name: str) -> float:
        if name not in self.splits:
            return 0.0
        prev = self.start_time
        for split_name in self.splits:
            if split_name == name:
                break
            prev = self.splits[split_name]
        return self.splits[name] - prev

class BenchmarkRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = MetricsCollector()
        self.results: List[BenchmarkResult] = []
        self.output_dir = Path(config.get('output_dir', 'results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_single_inference(self, backend, prompt: str, **kwargs) -> BenchmarkResult:
        raise NotImplementedError
    
    def run_throughput_test(self, backend, prompts: List[str], 
                           concurrency: int = 1) -> List[BenchmarkResult]:
        raise NotImplementedError
    
    def run_latency_test(self, backend, prompt: str, 
                        num_iterations: int = 10) -> Dict[str, Any]:
        latencies = []
        memory_samples = []
        
        for i in range(num_iterations):
            mem_before = self.metrics.get_memory_stats()
            
            timer = BenchmarkTimer().start()
            output = backend.generate(prompt, **self.config.get('generation', {}))
            timer.stop()
            
            mem_after = self.metrics.get_memory_stats()
            
            latencies.append(timer.elapsed * 1000)
            memory_samples.append({
                'before': mem_before,
                'after': mem_after,
                'delta_gb': mem_after.get('rss_gb', 0) - mem_before.get('rss_gb', 0)
            })
        
        return {
            'latencies_ms': latencies,
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p90_latency_ms': np.percentile(latencies, 90),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'memory_samples': memory_samples,
            'mean_memory_delta_gb': np.mean([s['delta_gb'] for s in memory_samples])
        }
    
    def save_results(self, filename: str = None):
        if filename is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        results_data = {
            'config': self.config,
            'results': [r.to_dict() for r in self.results],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        filepath = self.output_dir / filename
        with open(filepath, 'r') as f:
            return json.load(f)