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

# Step 7 - top_k_filter
def top_k_filter(logits, k):
    # TODO: keep the k largest entries of logits and set the rest to -inf.
    non_top_k = torch.argsort(-logits, dim=-1)[k:]
    top_k_logits = logits.clone()
    top_k_logits[non_top_k] = -torch.inf

    return top_k_logits

# Step 8 - top_p_filter
def top_p_filter(logits, p):
    # TODO: mask logits outside the smallest cumulative-probability nucleus of size p.
    top_p = torch.tensor(logits).clone()
    probs = torch.softmax(top_p, dim=-1)
    inds = torch.argsort(-probs, dim=-1)

    cutoff = ((torch.cumsum(probs[inds], dim=-1) < p).long()).sum().item()
    top_p[inds[cutoff+1:]] = -torch.inf

    return top_p

# Step 9 - build_synthetic_instruction_dataset
def build_synthetic_instruction_dataset():
    # TODO: return a small in-memory list of {'prompt', 'response'} dicts for SFT
    dataset = [{"prompt":"", "response":""} for _ in range(10)]

    dataset[0]["prompt"] = "How would you fix overfitting on a real model quickly?"
    dataset[0]["response"] = "Early stopping, dropout, regularization, decreasing the complexity of the model (reducing layers or neurons in layers), and adding additional data with more variety."

    dataset[1]["prompt"] = "How does learning rate affect convergence/stability; how do you tune it?"
    dataset[1]["response"] = "A high learning rate converges faster but might also overshoot the optimum and oscillate and diverge. A low learning rate makes training very slow and might get stuck in a local optimum and never get out. Tuning the learning rate with schedulers and decays is the standard method."

    dataset[2]["prompt"] = "Given a FLOP budget, how do you allocate model size vs data?"
    dataset[2]["response"] = "Given a compute budget, training a model with the least loss is a desired outcome. Following Chinchilla's scaling law, first derive the optimal number of parameters and then compute the number of training tokens required."

    dataset[3]["prompt"] = "How can scaling-law extrapolation inform capability/safety evals?"
    dataset[3]["response"] = "Scaling law extrapolation can help predict the point at which the sudden emergence of unforeseen capabilities is observed as we keep scaling. This helps in building relevant safety evals of these new-found abilities to prevent potential misuse."

    dataset[4]["prompt"] = "How is a reward model trained from pairwise preferences?"
    dataset[4]["response"] = "The score is calculated as the sigmoid of the difference in scores for the chosen and rejected preferences. The model is trained via regression with the negative log of the sigmoid of the difference acting as the loss."

    dataset[5]["prompt"] = "How does batch normalization help train very deep networks?"
    dataset[5]["response"] = "It normalizes the inputs of each layer to have a mean of zero and a variance of one for each training batch. This stabilizes the learning process, reduces internal covariate shift, and allows the use of much higher learning rates."

    dataset[6]["prompt"] = "What is the primary purpose of skip connections (residual connections)?"
    dataset[6]["response"] = "They add the input of a layer directly to its output (F(x) + x). This creates a 'highway' for gradients to flow straight backward through the network, solving the vanishing gradient problem in extremely deep architectures."

    dataset[7]["prompt"] = "Why is proper weight initialization critical for deep networks?"
    dataset[7]["response"] = "Starting with weights that are too large or too small leads directly to exploding or vanishing gradients. Techniques like He or Xavier initialization scale the weights properly based on the number of input/output connections, keeping variance stable across layers."

    dataset[8]["prompt"] = "How do I start a business and car sales?"
    dataset[8]["response"] = "If you want to start your own business or open a car dealership, you need to create a business plan, obtain a business license, and look for a location for your store or dealership."

    dataset[9]["prompt"] = "Why are telephone poles covered in tar?"
    dataset[9]["response"] = "Well this is a great question! I think the best explanation is that it’s to waterproof the wooden poles, so that they don’t rot in the rain or get cracked in the dry heat."

    return dataset

# Step 10 - format_example
def format_example(example):
    # TODO: render {'prompt','response'} into one training string with role markers
    return f"""### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}"""

