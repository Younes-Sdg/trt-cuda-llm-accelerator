import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import psutil
import torch.cuda
import numpy as np
from utils import run_perf_analysis
import transformers



transformers.logging.set_verbosity_error()

def get_gpu_info():
    """
    Fetch and print GPU information, including memory usage and utilization.
    """
    gpu_info = torch.cuda.get_device_properties(0)
    mem_total = gpu_info.total_memory / (1024 ** 2)  
    mem_used = torch.cuda.memory_allocated(0) / (1024 ** 2)  
    gpu_name = torch.cuda.get_device_name(0)
    gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
    return gpu_name, mem_used, mem_total, gpu_utilization


def generate_results(model, tokenizer, prompts, num_results=2, max_length=50):
    """
    Generate inference results for a list of prompts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    generated_results = {}
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_results = []
        for _ in range(num_results):
            output = model.generate(inputs['input_ids'], max_length=max_length)
            prompt_results.append(tokenizer.decode(output[0], skip_special_tokens=True))
        generated_results[prompt] = prompt_results
    
    return generated_results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    prompts = []
    while True:
        prompt = input("Enter a prompt for the model (or type 'done' to finish): ")
        if prompt.lower() == 'done':
            if not prompts:
                print("No prompts provided. Exiting...")
                return
            break
        prompts.append(prompt)
    
    try:
        model_path = "../saved_models/gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
        model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        
        
        print("\nGenerating text results...")
        generated_results = generate_results(model, tokenizer, prompts, num_results=2)
        
        for prompt, results in generated_results.items():
            print(f"\nGenerated Results for prompt: {prompt}")
            for i, result in enumerate(results, 1):
                print(f"Result {i}: {result}")
        
        
        print("\nRunning comprehensive performance analysis...")
        perf_analysis_results = run_perf_analysis(model, tokenizer, prompts)
        
        
        for prompt, results in perf_analysis_results.items():
            print(f"\nResults for prompt: {prompt}")
            
            gpu_name, mem_used, mem_total, gpu_utilization = get_gpu_info()
            inference_perf = results['inference_perf']
            
            
            print("\nPerformance Results:")
            print("+---------------------------------+---------------+")
            print("| Metric                          | Value         |")
            print("+---------------------------------+---------------+")
            print(f"| GPU Name                       | {gpu_name[:15]}|")
            print(f"| GPU Memory Used (MB)           | {mem_used:13.2f}|")
            print(f"| GPU Memory Total (MB)          | {mem_total:13.2f}|")
            print(f"| GPU Utilization (%)            | {gpu_utilization:13.2f}|")
            print(f"| Average Inference Time (s)     | {inference_perf['average_inference_time']:13.4f}|")
            print(f"| Throughput (tokens/s)          | {inference_perf['throughput_tokens_per_second']:13.2f}|")
            print(f"| Latency per Token (s)          | {inference_perf['latency_per_token']:13.4f}|")
            print(f"| Peak Memory Usage (MB)         | {inference_perf['peak_memory_usage_MB']:13.2f}|")
            print(f"| Average Tokens per Sequence    | {inference_perf['avg_tokens_per_sequence']:13.2f}|")
            print("+---------------------------------+---------------+")
            
            
            print("\nMemory Trace:")
            for i, mem_point in enumerate(results['memory_trace']):
                print(f"Step {i}: Allocated: {mem_point['allocated']:.2f} MB, "
                      f"Reserved: {mem_point['reserved']:.2f} MB, "
                      f"Max Allocated: {mem_point['max_allocated']:.2f} MB")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()