import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, results_dir: str = "results", output_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        filepath = self.results_dir / filename
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def generate_markdown_report(self, all_results: Dict[str, Any]) -> str:
        report = []
        
        report.append("# LLM Inference Optimization Report")
        report.append(f"\n**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"\n**Model:** {all_results.get('model', 'Unknown')}")
        report.append("")
        
        report.append("## Executive Summary")
        report.append("")
        report.append("This report presents benchmark results comparing different LLM inference strategies:")
        report.append("- **Serving Backends:** vLLM vs TGI vs HuggingFace Transformers")
        report.append("- **Quantization Methods:** FP16, INT8, INT4, GPTQ, AWQ")
        report.append("- **Batching Strategies:** Static, Dynamic, Continuous")
        report.append("")
        
        if 'backend_comparison' in all_results:
            report.append("## Backend Comparison")
            report.append("")
            report.append("### Latency Metrics")
            report.append("")
            report.append("| Backend | Mean Latency (ms) | P50 Latency (ms) | P95 Latency (ms) | P99 Latency (ms) |")
            report.append("|---------|-------------------|------------------|------------------|------------------|")
            
            for backend, data in all_results['backend_comparison'].items():
                if 'latency' in data:
                    lat = data['latency']
                    report.append(
                        f"| {backend:10} | {lat.get('mean_latency_ms', 0):17.2f} | "
                        f"{lat.get('p50_latency_ms', 0):16.2f} | {lat.get('p95_latency_ms', 0):16.2f} | "
                        f"{lat.get('p99_latency_ms', 0):16.2f} |"
                    )
            
            report.append("")
            report.append("### Throughput Metrics")
            report.append("")
            report.append("| Backend | Tokens/Second | Requests/Second | Mean TTFT (ms) |")
            report.append("|---------|---------------|-----------------|----------------|")
            
            for backend, data in all_results['backend_comparison'].items():
                if 'latency' in data:
                    lat = data['latency']
                    ttft = lat.get('mean_ttft_ms', 'N/A')
                    ttft_str = f"{ttft:.2f}" if isinstance(ttft, (int, float)) else ttft
                    report.append(
                        f"| {backend:10} | {lat.get('mean_tokens_per_second', 0):13.2f} | "
                        f"{lat.get('requests_per_second', 0):16.2f} | {ttft_str:14} |"
                    )
            
            report.append("")
        
        if 'quantization' in all_results:
            report.append("## Quantization Analysis")
            report.append("")
            report.append("### Size and Compression")
            report.append("")
            report.append("| Method | Bits | Original (GB) | Quantized (GB) | Compression | Quality Loss |")
            report.append("|--------|------|---------------|----------------|-------------|--------------|")
            
            for quant in all_results['quantization']:
                report.append(
                    f"| {quant.get('method', 'N/A'):6} | {quant.get('bits', 0):4} | "
                    f"{quant.get('original_size_gb', 0):12.2f} | {quant.get('quantized_size_gb', 0):14.2f} | "
                    f"{quant.get('compression_ratio', 0):11.1f}x | {quant.get('quality_degradation_pct', 0):10.1f}% |"
                )
            
            report.append("")
            report.append("### Performance Impact")
            report.append("")
            report.append("| Method | Latency (ms) | Throughput (tok/s) | Memory (GB) | GPU Memory (GB) |")
            report.append("|--------|--------------|-------------------|-------------|-----------------|")
            
            for quant in all_results['quantization']:
                gpu_mem = quant.get('gpu_memory_usage_gb', 'N/A')
                gpu_mem_str = f"{gpu_mem:.2f}" if isinstance(gpu_mem, (int, float)) else gpu_mem
                report.append(
                    f"| {quant.get('method', 'N/A'):6} | {quant.get('latency_ms', 0):12.1f} | "
                    f"{quant.get('throughput_tokens_per_second', 0):17.1f} | "
                    f"{quant.get('memory_usage_gb', 0):11.2f} | {gpu_mem_str:15} |"
                )
            
            report.append("")
        
        if 'batching' in all_results:
            report.append("## Batching Strategy Analysis")
            report.append("")
            
            for strategy, results in all_results['batching'].items():
                report.append(f"### {strategy.title()} Batching")
                report.append("")
                report.append("| Batch Size | Throughput (tok/s) | Requests/s | Avg Latency (ms) | Avg Queue (ms) |")
                report.append("|------------|-------------------|------------|------------------|----------------|")
                
                for r in results:
                    report.append(
                        f"| {r.get('batch_size', 0):10} | {r.get('tokens_per_second', 0):17.1f} | "
                        f"{r.get('requests_per_second', 0):10.1f} | "
                        f"{r.get('avg_latency_per_request_ms', 0):16.1f} | "
                        f"{r.get('avg_queue_time_ms', 0):14.1f} |"
                    )
                
                report.append("")
        
        if 'memory' in all_results:
            report.append("## Memory Analysis")
            report.append("")
            report.append("| Backend | CPU Memory (GB) | GPU Memory (GB) | Total (GB) |")
            report.append("|---------|-----------------|-----------------|------------|")
            
            for backend, mem in all_results['memory'].items():
                cpu_mem = mem.get('cpu_memory_gb', 0)
                gpu_mem = mem.get('gpu_memory_gb', 0)
                total = cpu_mem + gpu_mem
                report.append(
                    f"| {backend:10} | {cpu_mem:15.2f} | {gpu_mem:17.2f} | {total:10.2f} |"
                )
            
            report.append("")
        
        report.append("## Key Findings")
        report.append("")
        report.append("### Backend Performance")
        report.append("- **vLLM**: Best for high-throughput scenarios with continuous batching")
        report.append("- **TGI**: Excellent for production deployments with built-in scaling")
        report.append("- **HuggingFace**: Most flexible, good for development and experimentation")
        report.append("")
        report.append("### Quantization Tradeoffs")
        report.append("- **FP16**: Baseline, no quality loss, full memory usage")
        report.append("- **INT8**: 2x compression, minimal quality loss (<1%)")
        report.append("- **INT4**: 4x compression, acceptable quality loss (1-3%)")
        report.append("- **GPTQ/AWQ**: Best quality-compression ratio for production")
        report.append("")
        report.append("### Batching Strategies")
        report.append("- **Static**: Simple, predictable latency")
        report.append("- **Dynamic**: Better throughput with variable loads")
        report.append("- **Continuous**: Best for high-concurrency scenarios")
        report.append("")
        
        report.append("## Recommendations")
        report.append("")
        report.append("### For High Throughput (>100 req/s)")
        report.append("- Use **vLLM** with **continuous batching**")
        report.append("- Apply **INT4 quantization** (GPTQ or AWQ)")
        report.append("- Set batch size to **32-64**")
        report.append("")
        report.append("### For Low Latency (<100ms TTFT)")
        report.append("- Use **vLLM** or **TGI** with **dynamic batching**")
        report.append("- Keep **FP16** or **INT8** quantization")
        report.append("- Set max batch size to **8-16**")
        report.append("")
        report.append("### For Memory Constrained")
        report.append("- Use **INT4 quantization** (4x compression)")
        report.append("- Consider **AWQ** for best quality retention")
        report.append("- Reduce max sequence length")
        report.append("")
        
        report.append("## Appendix: Test Configuration")
        report.append("")
        report.append("```json")
        report.append(json.dumps(all_results.get('config', {}), indent=2))
        report.append("```")
        
        return "\n".join(report)
    
    def generate_html_report(self, all_results: Dict[str, Any]) -> str:
        html = []
        
        html.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Inference Optimization Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 40px; }
        h3 { color: #7f8c8d; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        tr:hover { background-color: #ddd; }
        .metric { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0; }
        .metric h3 { color: white; margin: 0; }
        .metric .value { font-size: 2em; font-weight: bold; }
        .recommendation { background-color: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 20px 0; }
        .warning { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; }
        code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
        pre { background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
        img { max-width: 100%; height: auto; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
""")
        
        html.append(f"<h1>LLM Inference Optimization Report</h1>")
        html.append(f"<p><strong>Generated:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>")
        html.append(f"<p><strong>Model:</strong> {all_results.get('model', 'Unknown')}</p>")
        
        html.append("<h2>Executive Summary</h2>")
        html.append("<p>This report presents benchmark results comparing different LLM inference strategies:</p>")
        html.append("<ul>")
        html.append("<li><strong>Serving Backends:</strong> vLLM vs TGI vs HuggingFace Transformers</li>")
        html.append("<li><strong>Quantization Methods:</strong> FP16, INT8, INT4, GPTQ, AWQ</li>")
        html.append("<li><strong>Batching Strategies:</strong> Static, Dynamic, Continuous</li>")
        html.append("</ul>")
        
        if 'backend_comparison' in all_results:
            html.append("<h2>Backend Comparison</h2>")
            html.append("<h3>Latency Metrics</h3>")
            html.append("<table>")
            html.append("<tr><th>Backend</th><th>Mean Latency (ms)</th><th>P50 Latency (ms)</th><th>P95 Latency (ms)</th><th>P99 Latency (ms)</th></tr>")
            
            for backend, data in all_results['backend_comparison'].items():
                if 'latency' in data:
                    lat = data['latency']
                    html.append(f"<tr><td>{backend}</td><td>{lat.get('mean_latency_ms', 0):.2f}</td><td>{lat.get('p50_latency_ms', 0):.2f}</td><td>{lat.get('p95_latency_ms', 0):.2f}</td><td>{lat.get('p99_latency_ms', 0):.2f}</td></tr>")
            
            html.append("</table>")
            
            html.append("<h3>Throughput Metrics</h3>")
            html.append("<table>")
            html.append("<tr><th>Backend</th><th>Tokens/Second</th><th>Requests/Second</th><th>Mean TTFT (ms)</th></tr>")
            
            for backend, data in all_results['backend_comparison'].items():
                if 'latency' in data:
                    lat = data['latency']
                    ttft = lat.get('mean_ttft_ms', 'N/A')
                    ttft_str = f"{ttft:.2f}" if isinstance(ttft, (int, float)) else ttft
                    html.append(f"<tr><td>{backend}</td><td>{lat.get('mean_tokens_per_second', 0):.2f}</td><td>{lat.get('requests_per_second', 0):.2f}</td><td>{ttft_str}</td></tr>")
            
            html.append("</table>")
        
        if 'quantization' in all_results:
            html.append("<h2>Quantization Analysis</h2>")
            html.append("<table>")
            html.append("<tr><th>Method</th><th>Bits</th><th>Original (GB)</th><th>Quantized (GB)</th><th>Compression</th><th>Quality Loss</th><th>Latency (ms)</th><th>Throughput (tok/s)</th></tr>")
            
            for quant in all_results['quantization']:
                html.append(f"<tr><td>{quant.get('method', 'N/A')}</td><td>{quant.get('bits', 0)}</td><td>{quant.get('original_size_gb', 0):.2f}</td><td>{quant.get('quantized_size_gb', 0):.2f}</td><td>{quant.get('compression_ratio', 0):.1f}x</td><td>{quant.get('quality_degradation_pct', 0):.1f}%</td><td>{quant.get('latency_ms', 0):.1f}</td><td>{quant.get('throughput_tokens_per_second', 0):.1f}</td></tr>")
            
            html.append("</table>")
        
        html.append("<h2>Recommendations</h2>")
        
        html.append("""<div class="recommendation">
            <h3>For High Throughput (>100 req/s)</h3>
            <ul>
                <li>Use <strong>vLLM</strong> with <strong>continuous batching</strong></li>
                <li>Apply <strong>INT4 quantization</strong> (GPTQ or AWQ)</li>
                <li>Set batch size to <strong>32-64</strong></li>
            </ul>
        </div>""")
        
        html.append("""<div class="recommendation">
            <h3>For Low Latency (<100ms TTFT)</h3>
            <ul>
                <li>Use <strong>vLLM</strong> or <strong>TGI</strong> with <strong>dynamic batching</strong></li>
                <li>Keep <strong>FP16</strong> or <strong>INT8</strong> quantization</li>
                <li>Set max batch size to <strong>8-16</strong></li>
            </ul>
        </div>""")
        
        html.append("</body></html>")
        
        return "\n".join(html)
    
    def generate_report(self, all_results: Dict[str, Any], formats: List[str] = None):
        if formats is None:
            formats = ['markdown', 'html']
        
        reports = {}
        
        if 'markdown' in formats:
            md_report = self.generate_markdown_report(all_results)
            md_path = self.output_dir / 'benchmark_report.md'
            with open(md_path, 'w') as f:
                f.write(md_report)
            reports['markdown'] = md_path
            logger.info(f"Markdown report saved to {md_path}")
        
        if 'html' in formats:
            html_report = self.generate_html_report(all_results)
            html_path = self.output_dir / 'benchmark_report.html'
            with open(html_path, 'w') as f:
                f.write(html_report)
            reports['html'] = html_path
            logger.info(f"HTML report saved to {html_path}")
        
        return reports