import tensorrt as trt
import tensorrt_llm
from tensorrt_llm.builder import Engine
import torch
import numpy as np
import os

def load_engine(engine_path):
    """
    Load a TensorRT engine file from the specified path.
    
    Args:
        engine_path (str): Path to the TensorRT engine file.
    
    Returns:
        Engine: Loaded TensorRT engine.
    """
    try:
        engine = Engine.from_dir(engine_path)
        engine_buffer = engine.engine
        if engine_buffer is None:
            raise ValueError("Failed to load engine: engine buffer is empty.")
        print("Engine loaded successfully.")
        return engine
    except Exception as e:
        print(f"Error loading engine: {e}")
        return None

def perform_inference(engine, input_text, max_new_tokens=20):
    """
    Perform inference using the loaded TensorRT-LLM engine.
    
    Args:
        engine (Engine): Loaded TensorRT-LLM engine
        input_text (str): Input text to generate from
        max_new_tokens (int): Maximum number of new tokens to generate
    
    Returns:
        str: Generated text
    """
    # Tokenize the input text
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Prepare the input for the engine
    input_length = input_ids.shape[1]
    
    # Create a context for the engine
    context = engine.create_context()
    
    # Prepare input and output buffers
    # This part might need adjustment based on your specific engine configuration
    input_buffer = input_ids.numpy()
    output_buffer = np.zeros((1, input_length + max_new_tokens), dtype=np.int32)
    
    # Set up bindings (this is a simplified example and might need customization)
    bindings = {
        'input_ids': input_buffer,
        'output_ids': output_buffer
    }
    
    # Run inference
    context.execute(bindings)
    
    # Decode the output
    generated_ids = output_buffer[0][input_length:]
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text

if __name__ == "__main__":
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    engine_path = os.path.join(base_dir, "../saved_models/gpt2_trt_f16_w_int8/trt_engines/fp16/1-gpu")
    
    # Load the engine
    engine = load_engine(engine_path)
    
    if engine:
        # Perform a simple inference
        input_prompt = "Once upon a time,"
        generated_output = perform_inference(engine, input_prompt)
        print("\nInput Prompt:", input_prompt)
        print("Generated Text:", generated_output)