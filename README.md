# LLM Inference Optimization

A comprehensive benchmarking suite for comparing LLM inference strategies across different serving backends, quantization methods, and batching strategies.

## Overview

This project benchmarks open-weight models (Llama/Mistral class) across:

- **Serving Backends**: vLLM vs TGI vs raw HuggingFace Transformers
- **Quantization Methods**: FP16, INT8, INT4, GPTQ, AWQ
- **Batching Strategies**: Static, Dynamic, Continuous

## Features

- Automated benchmarking with configurable parameters
- Latency profiling (P50, P90, P95, P99)
- Throughput measurement (tokens/second, requests/second)
- Memory usage tracking (CPU and GPU)
- Quantization tradeoff analysis
- Batching strategy comparison
- Professional HTML and Markdown reports
- Visualization generation (charts and graphs)

## Quick Start

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA support (recommended)
- 16GB+ RAM
- 50GB+ disk space for models

### Installation

```bash
# Clone the repository
git clone https://github.com/KhalidKhan541/LLM-Inference-Optimization.git
cd LLM-Inference-Optimization

# Install dependencies
pip install -r requirements.txt

# For vLLM support
pip install vllm

# For quantization support
pip install bitsandbytes auto-gptq autoawq
```

### Running Benchmarks

```bash
# Run full benchmark suite
python run_benchmark.py --model meta-llama/Llama-2-7b-hf

# Run specific benchmarks
python run_benchmark.py --model meta-llama/Llama-2-7b-hf --skip-backends
python run_benchmark.py --model meta-llama/Llama-2-7b-hf --skip-quantization
python run_benchmark.py --model meta-llama/Llama-2-7b-hf --skip-batching

# Use custom prompts
python run_benchmark.py --prompt-file prompts.txt

# Specify output directory
python run_benchmark.py --output-dir ./my_results
```

### Docker Deployment

```bash
# Using Docker Compose
cd docker
docker-compose up benchmark

# Start vLLM server
docker-compose up vllm-server

# Start TGI server
docker-compose up tgi-server
```

## Project Structure

```
llm-inference-optimization/
├── benchmarks/
│   ├── __init__.py
│   ├── core.py                 # Core benchmark utilities
│   ├── vllm_benchmark.py       # vLLM backend benchmark
│   ├── tgi_benchmark.py        # TGI backend benchmark
│   ├── hf_benchmark.py         # HuggingFace backend benchmark
│   ├── quantization_benchmark.py # Quantization analysis
│   └── batching_benchmark.py   # Batching strategy analysis
├── visualizations/
│   ├── __init__.py
│   ├── plots.py                # Chart generation
│   └── report.py               # Report generation
├── configs/
│   └── benchmark_config.yaml   # Benchmark configuration
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── results/                    # Benchmark results (generated)
├── visualizations/             # Charts and graphs (generated)
├── run_benchmark.py            # Main benchmark runner
├── requirements.txt
└── README.md
```

## Backend Comparison

### vLLM

High-throughput serving with:
- Continuous batching
- PagedAttention for efficient memory usage
- Tensor parallelism support
- OpenAI-compatible API

```bash
# Start vLLM server
docker-compose up vllm-server

# Benchmark against vLLM
python run_benchmark.py --model meta-llama/Llama-2-7b-hf
```

### TGI (Text Generation Inference)

Production-ready serving with:
- Dynamic batching
- Quantization support (GPTQ, AWQ, BitsAndBytes)
- Watermarking
- Token streaming

```bash
# Start TGI server
docker-compose up tgi-server
```

### HuggingFace Transformers

Reference implementation with:
- Full control over generation
- Easy customization
- Debugging capabilities

## Quantization Methods

| Method | Bits | Compression | Quality Loss | Best For |
|--------|------|-------------|--------------|----------|
| FP16 | 16 | 1x | None | Baseline |
| INT8 | 8 | 2x | <1% | Balanced |
| INT4 | 4 | 4x | 1-3% | Memory constrained |
| GPTQ | 4 | 4x | 1-2% | Production |
| AWQ | 4 | 4x | <1% | Best quality |

## Batching Strategies

### Static Batching
- Fixed batch size
- Simple implementation
- Predictable latency

### Dynamic Batching
- Adaptive batch sizes
- Better throughput
- Handles variable loads

### Continuous Batching
- Highest throughput
- Best for high concurrency
- Used by vLLM and TGI

## Output

Results are saved to the `results/` directory:

- `all_results.json` - Complete benchmark data
- `benchmark_report.md` - Markdown report
- `benchmark_report.html` - Interactive HTML report

Visualizations are saved to `visualizations/`:

- `latency_comparison.png`
- `throughput_comparison.png`
- `quantization_tradeoffs.png`
- `batching_strategies.png`
- `memory_usage.png`

## Configuration

Edit `configs/benchmark_config.yaml` to customize:

```yaml
models:
  - name: "meta-llama/Llama-2-7b-hf"
    size: "7B"
    context_length: 4096

benchmark:
  num_iterations: 10
  max_tokens: 128
  batch_sizes: [1, 2, 4, 8, 16, 32]
```

## Environment Variables

- `HF_TOKEN` - Hugging Face token for gated models
- `NVIDIA_VISIBLE_DEVICES` - GPU device selection

## Results Format

```json
{
  "model": "meta-llama/Llama-2-7b-hf",
  "backend_comparison": {
    "vllm": {
      "latency": {
        "mean_latency_ms": 45.2,
        "p95_latency_ms": 62.1,
        "mean_tokens_per_second": 85.3
      }
    }
  },
  "quantization": [...],
  "batching": {...}
}
```

## License

MIT License