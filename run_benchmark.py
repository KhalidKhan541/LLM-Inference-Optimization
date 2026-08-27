import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

from benchmarks.vllm_benchmark import VLLMBenchmark, VLLMConfig
from benchmarks.tgi_benchmark import TGIBenchmark, TGIConfig
from benchmarks.hf_benchmark import HFBenchmark, HFConfig
from benchmarks.quantization_benchmark import QuantizationBenchmark
from benchmarks.batching_benchmark import BatchingBenchmark, BatchConfig
from visualizations.plots import ResultsVisualizer
from visualizations.report import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SAMPLE_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "Write a Python function to sort a list of integers using quicksort.",
    "What are the main differences between supervised and unsupervised learning?",
    "Describe the architecture of a transformer neural network.",
    "How does gradient descent work in optimizing neural networks?",
    "What is the vanishing gradient problem and how can it be solved?",
    "Explain the difference between L1 and L2 regularization.",
    "What are attention mechanisms in neural networks?",
    "How do recurrent neural networks (RNNs) process sequential data?",
    "What is transfer learning and when should it be used?",
    "Explain the concept of backpropagation in neural networks.",
    "What are convolutional neural networks (CNNs) used for?",
    "How does batch normalization improve neural network training?",
    "What is the purpose of dropout in neural networks?",
    "Explain the difference between generative and discriminative models.",
    "What are autoencoders and how are they used?",
    "How do GANs (Generative Adversarial Networks) work?",
    "What is reinforcement learning and how does it differ from supervised learning?",
    "Explain the concept of Q-learning in reinforcement learning.",
    "What are the challenges of training deep neural networks?",
]

def run_backend_comparison(model_name: str, prompts: List[str], 
                          output_dir: str = "results") -> Dict[str, Any]:
    logger.info("Running backend comparison benchmark...")
    
    results = {}
    
    try:
        vllm_config = VLLMConfig(model=model_name)
        vllm_benchmark = VLLMBenchmark(vllm_config)
        vllm_benchmark.initialize()
        results['vllm'] = vllm_benchmark.benchmark_all(prompts)
        del vllm_benchmark
    except Exception as e:
        logger.error(f"vLLM benchmark failed: {e}")
        results['vllm'] = {'error': str(e)}
    
    try:
        tgi_config = TGIConfig(model=model_name)
        tgi_benchmark = TGIBenchmark(tgi_config)
        if tgi_benchmark.check_health():
            results['tgi'] = tgi_benchmark.benchmark_all(prompts)
        else:
            logger.warning("TGI server not available, skipping TGI benchmark")
            results['tgi'] = {'error': 'TGI server not available'}
        del tgi_benchmark
    except Exception as e:
        logger.error(f"TGI benchmark failed: {e}")
        results['tgi'] = {'error': str(e)}
    
    try:
        hf_config = HFConfig(model=model_name)
        hf_benchmark = HFBenchmark(hf_config)
        hf_benchmark.initialize()
        results['huggingface'] = hf_benchmark.benchmark_all(prompts)
        del hf_benchmark
    except Exception as e:
        logger.error(f"HuggingFace benchmark failed: {e}")
        results['huggingface'] = {'error': str(e)}
    
    return results

def run_quantization_analysis(model_name: str, prompt: str,
                             output_dir: str = "results") -> List[Dict]:
    logger.info("Running quantization analysis...")
    
    quant_benchmark = QuantizationBenchmark(model_name, output_dir)
    results = quant_benchmark.run_full_benchmark(prompt)
    quant_benchmark.save_results()
    
    return [
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
            'quality_degradation_pct': r.quality_degradation_pct
        }
        for r in results
    ]

def run_batching_analysis(prompts: List[str], output_dir: str = "results") -> Dict[str, List[Dict]]:
    logger.info("Running batching strategy analysis...")
    
    def dummy_inference(prompts, max_tokens=128):
        import time
        time.sleep(0.01 * len(prompts))
        return [{'completion_tokens': max_tokens} for _ in prompts]
    
    batch_config = BatchConfig(
        max_batch_size=32,
        max_wait_time_ms=100,
        max_tokens_per_batch=4096
    )
    
    batching_benchmark = BatchingBenchmark(batch_config, dummy_inference, output_dir)
    results = batching_benchmark.run_full_benchmark(prompts)
    batching_benchmark.save_results()
    
    return {
        'static': [
            {
                'batch_size': r.batch_size,
                'total_time_s': r.total_time_s,
                'total_tokens_generated': r.total_tokens_generated,
                'tokens_per_second': r.tokens_per_second,
                'requests_per_second': r.requests_per_second,
                'avg_latency_per_request_ms': r.avg_latency_per_request_ms,
                'avg_queue_time_ms': r.avg_queue_time_ms,
                'batching_strategy': r.batching_strategy
            }
            for r in results.get('static', [])
        ],
        'dynamic': [
            {
                'batch_size': r.batch_size,
                'total_time_s': r.total_time_s,
                'total_tokens_generated': r.total_tokens_generated,
                'tokens_per_second': r.tokens_per_second,
                'requests_per_second': r.requests_per_second,
                'avg_latency_per_request_ms': r.avg_latency_per_request_ms,
                'avg_queue_time_ms': r.avg_queue_time_ms,
                'batching_strategy': r.batching_strategy
            }
            for r in results.get('dynamic', [])
        ],
        'continuous': [
            {
                'batch_size': r.batch_size,
                'total_time_s': r.total_time_s,
                'total_tokens_generated': r.total_tokens_generated,
                'tokens_per_second': r.tokens_per_second,
                'requests_per_second': r.requests_per_second,
                'avg_latency_per_request_ms': r.avg_latency_per_request_ms,
                'avg_queue_time_ms': r.avg_queue_time_ms,
                'batching_strategy': r.batching_strategy
            }
            for r in results.get('continuous', [])
        ]
    }

def main():
    parser = argparse.ArgumentParser(description='LLM Inference Optimization Benchmark')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                       help='Model name or path')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--skip-backends', action='store_true',
                       help='Skip backend comparison')
    parser.add_argument('--skip-quantization', action='store_true',
                       help='Skip quantization analysis')
    parser.add_argument('--skip-batching', action='store_true',
                       help='Skip batching analysis')
    parser.add_argument('--num-prompts', type=int, default=20,
                       help='Number of prompts to use')
    parser.add_argument('--prompt-file', type=str, default=None,
                       help='File containing prompts (one per line)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = SAMPLE_PROMPTS[:args.num_prompts]
    
    all_results = {
        'model': args.model,
        'config': {
            'model': args.model,
            'num_prompts': len(prompts),
            'skip_backends': args.skip_backends,
            'skip_quantization': args.skip_quantization,
            'skip_batching': args.skip_batching
        }
    }
    
    if not args.skip_backends:
        all_results['backend_comparison'] = run_backend_comparison(
            args.model, prompts, args.output_dir
        )
    
    if not args.skip_quantization:
        all_results['quantization'] = run_quantization_analysis(
            args.model, prompts[0], args.output_dir
        )
    
    if not args.skip_batching:
        all_results['batching'] = run_batching_analysis(
            prompts, args.output_dir
        )
    
    raw_results_path = output_dir / 'all_results.json'
    with open(raw_results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Raw results saved to {raw_results_path}")
    
    visualizer = ResultsVisualizer(args.output_dir, str(output_dir / 'visualizations'))
    visualizer.generate_all_plots(all_results)
    
    report_gen = ReportGenerator(args.output_dir, args.output_dir)
    report_gen.generate_report(all_results)
    
    logger.info("Benchmark complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Report available at: {output_dir / 'benchmark_report.html'}")

if __name__ == "__main__":
    main()