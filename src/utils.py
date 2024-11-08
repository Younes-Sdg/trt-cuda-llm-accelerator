import torch
import time
import numpy as np
import matplotlib.pyplot as plt

def measure_inference_perf(model, tokenizer, prompt, max_length=50, num_runs=10, batch_size=1, temperature=0.9):
    """
    Measure the inference performance of the model: average inference time, throughput, latency, memory usage.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    single_input = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = torch.cat([single_input] * batch_size).to(device)
    
    for _ in range(10):
        _ = model.generate(inputs, max_length=max_length,temperature=temperature,do_sample=True)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    start_time = time.time()
    token_lengths = []
    
    for _ in range(num_runs):
        outputs = model.generate(inputs, max_length=max_length)
        token_lengths.append(sum(len(output) for output in outputs) / batch_size)
        
    end_time = time.time()
    
    peak_memory_usage = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    avg_inference_time = (end_time - start_time) / num_runs
    throughput = (np.mean(token_lengths) * batch_size) / avg_inference_time
    latency = avg_inference_time / (np.mean(token_lengths) * batch_size)
    
    return {
        "average_inference_time": avg_inference_time,
        "throughput_tokens_per_second": throughput,
        "latency_per_token": latency,
        "peak_memory_usage_MB": peak_memory_usage / (1024 * 1024),
        "batch_size": batch_size,
        "avg_tokens_per_sequence": np.mean(token_lengths)
    }

def analyze_seq_len(model, tokenizer, base_prompt, max_lengths=[16, 32, 64, 128, 256]):
    """
    Analyze how sequence length affects model performance.
    """
    results = []
    for length in max_lengths:
        metrics = measure_inference_perf(model, tokenizer, base_prompt, max_length=length)
        results.append({'max_length': length, **metrics})
    return results

def analyze_batch_size(model, tokenizer, prompt, batch_sizes=[1, 2, 4, 8, 16]):
    """
    Analyze how batch size affects model performance.
    """
    results = []
    for batch_size in batch_sizes:
        metrics = measure_inference_perf(model, tokenizer, prompt, batch_size=batch_size)
        results.append({'batch_size': batch_size, **metrics})
    return results

def memory_profile(model, tokenizer, prompt):
    """
    Detailed memory profiling during generation.
    """
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    torch.cuda.reset_peak_memory_stats()
    
    memory_trace = []
    def track_memory():
        memory_trace.append({
            'allocated': torch.cuda.memory_allocated() / 1024**2,
            'reserved': torch.cuda.memory_reserved() / 1024**2,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**2
        })
    
    track_memory()
    inputs = tokenizer.encode(prompt, return_tensors="pt").cuda()
    track_memory()
    outputs = model.generate(inputs)
    track_memory()
    
    return memory_trace

def plot_metrics(results, x_key, title_prefix=""):
    """
    Create visualization plots for performance metrics.
    """
    metrics = ['average_inference_time', 'throughput_tokens_per_second', 'latency_per_token', 'peak_memory_usage_MB']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{title_prefix} Performance Analysis', fontsize=16)
    
    for (metric, ax) in zip(metrics, axes.ravel()):
        x = [r[x_key] for r in results]
        y = [r[metric] for r in results]
        
        ax.plot(x, y, marker='o', linewidth=2)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel(x_key.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def run_perf_analysis(model, tokenizer, prompts):
    """
    Run performance analysis on multiple prompts.
    """
    all_results = {}

    for prompt in prompts:
        print(f"\nRunning performance analysis for prompt: {prompt}")
        
        
        print("Measuring inference performance...")
        inference_perf = measure_inference_perf(model, tokenizer, prompt)
        
       
        print("Running sequence length analysis...")
        seq_results = analyze_seq_len(model, tokenizer, prompt)
        plot_metrics(seq_results, 'max_length', "Sequence Length")
        
       
        print("\nRunning batch size analysis...")
        batch_results = analyze_batch_size(model, tokenizer, prompt)
        plot_metrics(batch_results, 'batch_size', "Batch Size")
        
        
        print("\nRunning memory profiling...")
        memory_trace = memory_profile(model, tokenizer, prompt)
        
        
        all_results[prompt] = {
            'inference_perf': inference_perf,  
            'sequence_length_results': seq_results,
            'batch_size_results': batch_results,
            'memory_trace': memory_trace
        }
        
    return all_results


