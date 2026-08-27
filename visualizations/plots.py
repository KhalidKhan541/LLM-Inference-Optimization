import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

class ResultsVisualizer:
    def __init__(self, results_dir: str = "results", output_dir: str = "visualizations"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'vllm': '#2196F3',
            'tgi': '#4CAF50',
            'huggingface': '#FF9800',
            'fp16': '#9C27B0',
            'int8': '#E91E63',
            'int4': '#00BCD4',
            'gptq': '#FF5722',
            'awq': '#795548',
            'static': '#607D8B',
            'dynamic': '#F44336',
            'continuous': '#3F51B5'
        }
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        filepath = self.results_dir / filename
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def plot_latency_comparison(self, results: Dict[str, List[Dict]], 
                              save_path: Optional[str] = None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        backends = list(results.keys())
        metrics = ['mean_latency_ms', 'p50_latency_ms', 'p95_latency_ms']
        metric_labels = ['Mean Latency', 'P50 Latency', 'P95 Latency']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]
            values = []
            for backend in backends:
                backend_results = results[backend]
                if 'latency' in backend_results and metric in backend_results['latency']:
                    values.append(backend_results['latency'][metric])
                else:
                    values.append(0)
            
            bars = ax.bar(backends, values, color=[self.colors.get(b, '#999999') for b in backends])
            ax.set_title(label, fontsize=14, fontweight='bold')
            ax.set_ylabel('Milliseconds', fontsize=12)
            ax.set_xlabel('Backend', fontsize=12)
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Latency Comparison Across Backends', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'latency_comparison.png', dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_throughput_comparison(self, results: Dict[str, List[Dict]],
                                 save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        backends = list(results.keys())
        throughputs = []
        
        for backend in backends:
            backend_results = results[backend]
            if 'latency' in backend_results:
                throughputs.append(backend_results['latency'].get('mean_tokens_per_second', 0))
            else:
                throughputs.append(0)
        
        bars = ax.bar(backends, throughputs, color=[self.colors.get(b, '#999999') for b in backends])
        ax.set_title('Throughput Comparison (Tokens/Second)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Tokens per Second', fontsize=12)
        ax.set_xlabel('Backend', fontsize=12)
        
        for bar, val in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'throughput_comparison.png', dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_quantization_tradeoffs(self, results: List[Dict],
                                   save_path: Optional[str] = None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        methods = [r['method'] for r in results]
        latencies = [r['latency_ms'] for r in results]
        throughputs = [r['throughput_tokens_per_second'] for r in results]
        compression = [r['compression_ratio'] for r in results]
        quality_loss = [r.get('quality_degradation_pct', 0) for r in results]
        
        ax1 = axes[0]
        bars1 = ax1.bar(methods, latencies, color=[self.colors.get(m, '#999999') for m in methods])
        ax1.set_title('Latency by Quantization', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Latency (ms)', fontsize=12)
        ax1.set_xlabel('Quantization Method', fontsize=12)
        for bar, val in zip(bars1, latencies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        ax2 = axes[1]
        bars2 = ax2.bar(methods, throughputs, color=[self.colors.get(m, '#999999') for m in methods])
        ax2.set_title('Throughput by Quantization', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Tokens/Second', fontsize=12)
        ax2.set_xlabel('Quantization Method', fontsize=12)
        for bar, val in zip(bars2, throughputs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        ax3 = axes[2]
        scatter = ax3.scatter(compression, quality_loss, c=latencies, cmap='RdYlGn_r', s=100)
        for i, method in enumerate(methods):
            ax3.annotate(method, (compression[i], quality_loss[i]), 
                        textcoords="offset points", xytext=(0, 10), ha='center')
        ax3.set_title('Compression vs Quality Loss', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Compression Ratio', fontsize=12)
        ax3.set_ylabel('Quality Degradation (%)', fontsize=12)
        plt.colorbar(scatter, ax=ax3, label='Latency (ms)')
        
        plt.suptitle('Quantization Method Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'quantization_tradeoffs.png', dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_batching_strategies(self, results: Dict[str, List[Dict]],
                               save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        strategies = list(results.keys())
        
        ax1 = axes[0, 0]
        for strategy in strategies:
            strategy_results = results[strategy]
            batch_sizes = [r['batch_size'] for r in strategy_results]
            throughputs = [r['tokens_per_second'] for r in strategy_results]
            ax1.plot(batch_sizes, throughputs, marker='o', label=strategy,
                    color=self.colors.get(strategy, '#999999'), linewidth=2)
        ax1.set_title('Throughput vs Batch Size', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Batch Size', fontsize=12)
        ax1.set_ylabel('Tokens/Second', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        for strategy in strategies:
            strategy_results = results[strategy]
            batch_sizes = [r['batch_size'] for r in strategy_results]
            latencies = [r['avg_latency_per_request_ms'] for r in strategy_results]
            ax2.plot(batch_sizes, latencies, marker='s', label=strategy,
                    color=self.colors.get(strategy, '#999999'), linewidth=2)
        ax2.set_title('Latency vs Batch Size', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Batch Size', fontsize=12)
        ax2.set_ylabel('Avg Latency (ms)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        strategy_names = []
        avg_latencies = []
        for strategy in strategies:
            strategy_results = results[strategy]
            if strategy_results:
                strategy_names.append(strategy)
                avg_latencies.append(np.mean([r['avg_latency_per_request_ms'] for r in strategy_results]))
        
        bars = ax3.bar(strategy_names, avg_latencies, 
                      color=[self.colors.get(s, '#999999') for s in strategy_names])
        ax3.set_title('Average Latency by Strategy', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Avg Latency (ms)', fontsize=12)
        ax3.set_xlabel('Batching Strategy', fontsize=12)
        for bar, val in zip(bars, avg_latencies):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        ax4 = axes[1, 1]
        avg_throughputs = []
        for strategy in strategies:
            strategy_results = results[strategy]
            if strategy_results:
                avg_throughputs.append(np.mean([r['tokens_per_second'] for r in strategy_results]))
        
        bars = ax4.bar(strategy_names, avg_throughputs,
                      color=[self.colors.get(s, '#999999') for s in strategy_names])
        ax4.set_title('Average Throughput by Strategy', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Tokens/Second', fontsize=12)
        ax4.set_xlabel('Batching Strategy', fontsize=12)
        for bar, val in zip(bars, avg_throughputs):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Batching Strategy Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'batching_strategies.png', dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_latency_distribution(self, latencies: Dict[str, List[float]],
                                save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for backend, lat_list in latencies.items():
            if lat_list:
                ax.hist(lat_list, bins=30, alpha=0.5, label=backend,
                       color=self.colors.get(backend, '#999999'))
        
        ax.set_title('Latency Distribution by Backend', fontsize=14, fontweight='bold')
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'latency_distribution.png', dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_memory_usage(self, memory_data: Dict[str, Dict[str, float]],
                         save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        backends = list(memory_data.keys())
        cpu_memory = [memory_data[b].get('cpu_memory_gb', 0) for b in backends]
        gpu_memory = [memory_data[b].get('gpu_memory_gb', 0) for b in backends]
        
        x = np.arange(len(backends))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, cpu_memory, width, label='CPU Memory', color='#2196F3')
        bars2 = ax.bar(x + width/2, gpu_memory, width, label='GPU Memory', color='#FF9800')
        
        ax.set_title('Memory Usage by Backend', fontsize=14, fontweight='bold')
        ax.set_ylabel('Memory (GB)', fontsize=12)
        ax.set_xlabel('Backend', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(backends)
        ax.legend()
        
        for bar, val in zip(bars1, cpu_memory):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, gpu_memory):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'memory_usage.png', dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def generate_all_plots(self, results: Dict[str, Any]):
        if 'backend_comparison' in results:
            self.plot_latency_comparison(results['backend_comparison'])
            self.plot_throughput_comparison(results['backend_comparison'])
        
        if 'quantization' in results:
            self.plot_quantization_tradeoffs(results['quantization'])
        
        if 'batching' in results:
            self.plot_batching_strategies(results['batching'])
        
        if 'latency_distribution' in results:
            self.plot_latency_distribution(results['latency_distribution'])
        
        if 'memory' in results:
            self.plot_memory_usage(results['memory'])
        
        logger.info(f"All plots saved to {self.output_dir}")