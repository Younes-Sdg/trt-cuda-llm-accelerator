import tensorrt as trt
from tensorrt_llm.builder import Engine
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



############# test ################


if __name__ == "__main__":
    
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    engine_path = os.path.join(base_dir, "../saved_models/gpt2_trt_f16_w_int8/trt_engines/fp16/1-gpu")
    
    engine = load_engine(engine_path)
    if engine:
        print("Engine details:")
        print(f"Engine type: {type(engine)}")
