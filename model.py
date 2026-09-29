"""
RLHF from Scratch on DistilGPT2

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_distilgpt2_tokenizer
from transformers import AutoTokenizer


def load_distilgpt2_tokenizer(model_name="sshleifer/tiny-gpt2"):
    # TODO: load and return the Hugging Face tokenizer for the given model name.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

# Step 2 - load_distilgpt2_model
from transformers import AutoModelForCausalLM


def load_distilgpt2_model(model_name="sshleifer/tiny-gpt2"):
    # TODO: load a causal LM by name and return it in eval mode
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model

# Step 3 - set_pad_token_to_eos
def set_pad_token_to_eos(tokenizer):
    # TODO: assign tokenizer.pad_token = tokenizer.eos_token and return the tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 4 - generate_and_decode
import torch


def generate_and_decode(model, tokenizer, prompt, max_new_tokens=8):
    # TODO: tokenize prompt, generate continuation greedily, decode and return as a string
    prompt_ids = tokenizer.encode([prompt], return_tensors='pt')

    for _ in range(max_new_tokens):
        token = torch.argmax(model.generate(prompt_ids), dim=-1, keepdim=True)
        prompt_ids = torch.cat([prompt_ids, token], dim=-1)

    return tokenizer.decode(prompt_ids[0])

# Step 5 - greedy_decode
import torch

def greedy_decode(logits):
    """Return the argmax token id from a single-row logits vector."""
    # TODO: return the token id with the largest logit as a Python int
    return torch.argmax(logits, dim=-1).item()

# Step 6 - sample_with_temperature
def sample_with_temperature(logits, temperature):
    # TODO: rescale logits by temperature, softmax, and sample one token id
    if temperature <= 0.0:
        temperature = 1.0

    probs = torch.softmax(logits/temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, replacement=True).item()

