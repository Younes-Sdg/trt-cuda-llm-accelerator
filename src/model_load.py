from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

def load_and_save_model():

    cache_dir = "saved_models/gpt2"
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    print("Downloading model and tokenizer from HuggingFace...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
    model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache_dir)

    save_model = input("Do you want to save the model and tokenizer? (y/n): ").strip().lower()

    if save_model == "y":
        tokenizer.save_pretrained(cache_dir)
        model.save_pretrained(cache_dir)
        print("Model and tokenizer saved.")
    else:
        print("Model and tokenizer not saved.")

    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Using GPU.")
    else:
        print("Using CPU.")

    print("Model and tokenizer loaded.")
    
    return model, tokenizer

load_and_save_model()
