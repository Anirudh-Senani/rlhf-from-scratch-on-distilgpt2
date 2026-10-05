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

# Step 11 - apply_template
def apply_template(examples):
    # TODO: apply format_example to each item in examples and return the list of strings.
    return [format_example(ex) for ex in examples]

# Step 12 - tokenize_example
def tokenize_example(tokenizer, text, max_length=64):
    # TODO: encode `text` with truncation at max_length, no padding, return list[int]
    return tokenizer.encode(text, max_length=max_length, truncation=True, padding=False)

# Step 13 - build_labels
def build_labels(input_ids):
    # TODO: return a fresh list equal to input_ids to serve as next-token labels
    return [x for x in input_ids]

# Step 14 - mask_prompt_labels
def mask_prompt_labels(labels, prompt_length):
    # TODO: replace the first prompt_length entries of labels with -100 and return the new list
    n = len(labels)
    prompt_length = min(prompt_length, n)

    return [-100 if i < prompt_length else labels[i] for i in range(n)]

# Step 15 - pad_batch
def pad_batch(sequences, pad_id):
    # TODO: right-pad a list of token id sequences to the longest length using pad_id
    lens = [len(seq) for seq in sequences]
    max_len = max(lens)

    return [sequences[i]+[pad_id]*(max_len-lens[i]) for i in range(len(lens))]

# Step 16 - make_attention_mask
def make_attention_mask(padded_ids, pad_id):
    # TODO: return a same-shape 0/1 mask with 1 where token != pad_id else 0
    return [[0 if pid==pad_id else 1 for pid in seq] for seq in padded_ids]

# Step 17 - collate_lm_batch
def collate_lm_batch(batch, pad_id):
    # TODO: pad input_ids and labels, build attention mask, return dict of LongTensors
    out = {}
    for key in batch[0]:
        out[key] = []

    for b in batch:
        for key in out:
            out[key].append(b[key])

    out['input_ids'] = pad_batch(out['input_ids'], pad_id)
    out['labels'] = pad_batch(out['labels'], -100)
    out['attention_mask'] = make_attention_mask(out['input_ids'], pad_id)

    for key in out:
        out[key] = torch.tensor(out[key], dtype=torch.long)

    return out

# Step 18 - iterate_minibatches
import random


def iterate_minibatches(examples, batch_size, seed=0):
    # TODO: yield shuffled minibatches of size batch_size from examples (deterministic per seed).
    rng = random.Random(seed)
    shuffled = [ex for ex in examples]
    rng.shuffle(shuffled)

    n = len(examples)
    for i in range(0, n, batch_size):
        yield shuffled[i:i+batch_size]

# Step 19 - train_val_split
def train_val_split(examples, val_ratio=0.2, seed=0):
    # TODO: deterministically split examples into (train, val) using seed and val_ratio
    n = len(examples)
    nval = int(n*val_ratio)
    if nval == 0:
        nval = -1

    rng = random.Random(seed)
    shuffled = [ex for ex in examples]
    rng.shuffle(shuffled)

    return shuffled[:-nval], shuffled[-nval:]

# Step 20 - shift_logits_and_labels
def shift_logits_and_labels(logits, labels):
    # TODO: drop the last logit position and the first label position so token t predicts t+1
    return logits[:,:-1], labels[:,1:]

# Step 21 - cross_entropy_loss
import torch
import torch.nn.functional as F

def cross_entropy_loss(shift_logits, shift_labels):
    """Mean next-token cross-entropy, ignoring label positions equal to -100."""
    # TODO: reduce (B, T-1, V) logits and (B, T-1) labels to a scalar loss tensor.
    B, T, V = shift_logits.shape
    return F.cross_entropy(shift_logits.reshape((B*T, V)), shift_labels.reshape((B*T,)), ignore_index=-100)

# Step 22 - adamw_update
import torch

def adamw_update(param, grad, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """Apply one in-place AdamW step to `param` using `grad` and persistent `state`."""
    # TODO: initialize state on first call, then update moments and apply the decoupled AdamW step
    if not state:
        state['m'] = torch.zeros_like(param)
        state['v'] = torch.zeros_like(param)
        state['step'] = 0

    state['step'] += 1

    with torch.no_grad():
        state['m'] *= betas[0]
        state['m'] += grad*(1-betas[0])
        state['v'] *= betas[1]
        state['v'] += (grad**2)*(1-betas[1])

        mt = state['m']/(1-betas[0]**state['step'])
        vt = state['v']/(1-betas[1]**state['step'])

        param -= lr * weight_decay * param
        param -= lr * mt/(torch.sqrt(vt) + eps)

    return state

# Step 23 - linear_warmup_schedule
def linear_warmup_schedule(step, warmup_steps):
    # TODO: return a linear warmup multiplier in [0, 1] given the current step and warmup window.
    return min(step/max(warmup_steps, 1), 1.0)

# Step 24 - clip_grad_norm
def clip_grad_norm(grads, max_norm):
    # TODO: compute the global L2 norm of grads and rescale in place if it exceeds max_norm.
    grad_norm = torch.sqrt(sum([(grad.reshape(-1)**2).sum() for grad in grads]))

    scale = 1.0
    if grad_norm > max_norm:
        scale = max_norm/grad_norm

    for i in range(len(grads)):
        grads[i] *= scale

    return grad_norm.item()

# Step 25 - accumulate_gradients
import torch

def accumulate_gradients(grad_list):
    """Average a list of equally-shaped gradient tensors across micro-batches."""
    # TODO: average a list of equally-shaped gradient tensors and return the mean tensor
    return torch.stack(grad_list).mean(dim=0)

# Step 26 - sft_train_step
import torch

def sft_train_step(model, batch, optimizer):
    """Run one SFT forward/backward/step and return the loss as a float."""
    # TODO: forward the batch, compute shifted cross-entropy loss, backprop, step optimizer
    logits = model(**batch).logits
    slog, slab = shift_logits_and_labels(logits, batch['labels'])

    loss = cross_entropy_loss(slog, slab)

    model.train()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# Step 27 - evaluate_loss
import torch

def evaluate_loss(model, batches):
    """Mean LM loss over validation batches, no grad."""
    # TODO: iterate batches under no_grad, shift logits/labels, average cross-entropy.
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in batches:
            logits = model(**batch).logits
            slog, slab = shift_logits_and_labels(logits, batch['labels'])
            loss = cross_entropy_loss(slog, slab)
            losses.append(loss)

    return torch.tensor(losses).mean().item()

# Step 28 - lora_delta
def lora_delta(A, B, alpha, r):
    # TODO: build the scaled low-rank weight update from factors A and B.
    return alpha/r * (B @ A)

# Step 29 - lora_linear_forward
def lora_linear_forward(x, base_weight, A, B, alpha, r, bias=None):
    # TODO: return x @ (base_weight + lora_delta).T (+ bias) using lora_delta(A, B, alpha, r)
    delta = lora_delta(A, B, alpha, r)
    weight = base_weight + delta

    out = x @ weight.T
    if bias is not None:
        out += bias

    return out

# Step 30 - init_lora_weights
import torch

def init_lora_weights(in_features, out_features, r, seed=0):
    """Return (A, B) LoRA factors with random A and zero B so the initial delta is zero."""
    # TODO: seed torch, build A of shape (r, in_features) and B of shape (out_features, r)
    torch.manual_seed(seed)

    A = torch.randn((r, in_features), dtype=torch.float32) * 0.01
    B = torch.zeros((out_features, r), dtype=torch.float32)

    return A, B

# Step 31 - freeze_base_params
def freeze_base_params(model):
    # TODO: set requires_grad=False on every base parameter, leaving LoRA adapters trainable.
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    return model

# Step 32 - count_trainable_params
def count_trainable_params(model):
    # TODO: sum p.numel() over parameters with requires_grad=True
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

# Step 33 - merge_lora
def merge_lora(base_weight, lora_a, lora_b, scaling):
    # TODO: fold the scaled low-rank update B @ A back into the base weight matrix.
    return base_weight + scaling * (lora_b @ lora_a)

# Step 34 - build_synthetic_preference_dataset
def build_synthetic_preference_dataset(num_examples=8, seed=0):
    # TODO: return a list of {'prompt','chosen','rejected'} dicts of length num_examples.
    rng = random.Random(seed)

    dataset = [{"prompt":"", "chosen":"", "rejected":""} for _ in range(12)]

    dataset[0]["prompt"] = "What is 2 + 2?"
    dataset[0]["chosen"] = "2 + 2 equals 4."
    dataset[0]["rejected"] = "2 + 2 equals 5."

    dataset[1]["prompt"] = "What is the capital of France?"
    dataset[1]["chosen"] = "The capital of France is Paris."
    dataset[1]['rejected'] = "I do not know."

    dataset[2]["prompt"] = "How would you fix overfitting on a real model quickly?"
    dataset[2]["chosen"] = "Early stopping, dropout, regularization, decreasing the complexity of the model (reducing layers or neurons in layers), and adding additional data with more variety."
    dataset[2]['rejected'] = "Early stopping, dropout, regularization, decreasing the complexity of the model (reducing layers or neurons in layers), and reducing the amount of data and variety."

    dataset[3]["prompt"] = "How does learning rate affect convergence/stability; how do you tune it?"
    dataset[3]["chosen"] = "A high learning rate converges faster but might also overshoot the optimum and oscillate and diverge. A low learning rate makes training very slow and might get stuck in a local optimum and never get out. Tuning the learning rate with schedulers and decays is the standard method."
    dataset[3]['rejected'] = "A low learning rate converges faster but might also overshoot the optimum and oscillate and diverge. A high learning rate makes training very slow and might get stuck in a local optimum and never get out. Tuning the learning rate with schedulers and decays is the standard method."

    dataset[4]["prompt"] = "Given a FLOP budget, how do you allocate model size vs data?"
    dataset[4]["chosen"] = "Given a compute budget, training a model with the least loss is a desired outcome. Following Chinchilla's scaling law, first derive the optimal number of parameters and then compute the number of training tokens required."
    dataset[4]['rejected'] = "Given a compute budget, training a model with the least loss is a desired outcome. Following Chinchilla's scaling law, choose any number of parameters and use as many training tokens as possible."

    dataset[5]["prompt"] = "How can scaling-law extrapolation inform capability/safety evals?"
    dataset[5]["chosen"] = "Scaling law extrapolation can help predict the point at which the sudden emergence of unforeseen capabilities is observed as we keep scaling. This helps in building relevant safety evals of these new-found abilities to prevent potential misuse."
    dataset[5]['rejected'] = "Scaling law extrapolation cannot predict the point at which the sudden emergence of unforeseen capabilities is observed as we keep scaling."

    dataset[6]["prompt"] = "How is a reward model trained from pairwise preferences?"
    dataset[6]["chosen"] = "The score is calculated as the sigmoid of the difference in scores for the chosen and rejected preferences. The model is trained via regression with the negative log of the sigmoid of the difference acting as the loss."
    dataset[6]['rejected'] = "The score is calculated as the difference in scores for the chosen and rejected preferences. The model is trained via regression with the negative sum of the difference acting as the loss."

    dataset[7]["prompt"] = "How does batch normalization help train very deep networks?"
    dataset[7]["chosen"] = "It normalizes the inputs of each layer to have a mean of zero and a variance of one for each training batch. This stabilizes the learning process, reduces internal covariate shift, and allows the use of much higher learning rates."
    dataset[7]['rejected'] = "It normalizes the inputs of each batch to have a mean of zero and a variance of one for each training layer. This stabilizes the learning process, reduces internal covariate shift, and allows the use of much higher learning rates."

    dataset[8]["prompt"] = "What is the primary purpose of skip connections (residual connections)?"
    dataset[8]["chosen"] = "They add the input of a layer directly to its output (F(x) + x). This creates a 'highway' for gradients to flow straight backward through the network, solving the vanishing gradient problem in extremely deep architectures."
    dataset[8]['rejected'] = "They add the input of a layer directly to its output (F(x) + x). This creates a 'highway' for gradients to flow straight backward through the network, solving the vanishing gradient problem in every architecture."

    dataset[9]["prompt"] = "Why is proper weight initialization critical for deep networks?"
    dataset[9]["chosen"] = "Starting with weights that are too large or too small leads directly to exploding or vanishing gradients. Techniques like He or Xavier initialization scale the weights properly based on the number of input/output connections, keeping variance stable across layers."
    dataset[9]['rejected'] = "Starting with weights that are too large or too small leads directly to exploding or vanishing gradients. Techniques like uniform or normal initialization, keeping variance stable across layers."

    dataset[10]["prompt"] = "How do I start a business and car sales?"
    dataset[10]["chosen"] = "If you want to start your own business or open a car dealership, you need to create a business plan, obtain a business license, and look for a location for your store or dealership."
    dataset[10]['rejected'] = "If you want to start your own business or open a car dealership, you need to create a business plan, and look for a location for your store or dealership. You can deal with business license and other legal matters once the business is running."

    dataset[11]["prompt"] = "Why are telephone poles covered in tar?"
    dataset[11]["chosen"] = "Well this is a great question! I think the best explanation is that it's to waterproof the wooden poles, so that they don't rot in the rain or get cracked in the dry heat."
    dataset[11]['rejected'] = "To stop the interference of free currents or earth's magnetic field, with the telephone signal."

    if num_examples <= 2:
        dataset = dataset[:2]
    return rng.sample(dataset, num_examples)

# Step 35 - format_preference
def format_preference(example):
    # TODO: return a dict with 'chosen_text' and 'rejected_text' built from the prompt and each answer.
    out = {}
    out['chosen_text'] = example['prompt'].strip() + ' ' + example['chosen'].strip()
    out['rejected_text'] = example['prompt'].strip() + ' ' + example['rejected'].strip()
    return out

# Step 36 - reward_head_forward
import torch

def reward_head_forward(hidden_state, weight, bias):
    """Map a final hidden state to a scalar reward via a linear projection."""
    # TODO: project hidden_state (B, D) through weight (D,) plus scalar bias to get (B,) rewards
    return hidden_state @ weight.squeeze(0) + bias

# Step 37 - pairwise_reward_loss
import torch
import torch.nn.functional as F

def pairwise_reward_loss(chosen_reward, rejected_reward):
    """Bradley-Terry pairwise loss: mean(-log_sigmoid(chosen - rejected))."""
    # TODO: return the mean negative log-sigmoid of (chosen_reward - rejected_reward)
    return torch.log(1.0 + torch.exp(-(chosen_reward - rejected_reward))).mean()

# Step 38 - reward_bce_loss
import numpy as np


def softplus(x):
    return np.logaddexp(x, 0.0)


def reward_bce_loss(chosen_reward, rejected_reward):
    # TODO: BCE-style reward loss with chosen as positives and rejected as negatives.
    return (softplus(-chosen_reward) + softplus(rejected_reward)).mean()/2

# Step 39 - pairwise_accuracy
import torch

def pairwise_accuracy(chosen_reward, rejected_reward):
    """Fraction of pairs where chosen_reward > rejected_reward."""
    # TODO: return the fraction of pairs where chosen strictly beats rejected
    return (chosen_reward > rejected_reward).float().mean().item()

# Step 40 - reward_train_step
import torch

def reward_train_step(model, reward_head, batch, optimizer):
    # TODO: forward chosen+rejected, score last token, compute loss/acc, step optimizer
    chosen_inputs = {}
    chosen_inputs['input_ids'] = batch['chosen_input_ids']
    B, T = batch['chosen_input_ids'].shape
    chosen_inputs['attention_mask'] = batch['chosen_attention_mask']
    chosen_len = torch.maximum(batch['chosen_attention_mask'].sum(dim=-1, keepdim=True).long() - 1, torch.tensor(0))
    chosen_hidden = model(**chosen_inputs)
    chosen_hidden_last = chosen_hidden[torch.arange(B)[:, None], chosen_len, :]

    rejected_inputs = {}
    rejected_inputs['input_ids'] = batch['rejected_input_ids']
    B, T = batch['rejected_input_ids'].shape
    rejected_inputs['attention_mask'] = batch['rejected_attention_mask']
    rejected_len = torch.maximum(batch['rejected_attention_mask'].sum(dim=-1, keepdim=True).long() - 1, torch.tensor(0))
    rejected_hidden = model(**rejected_inputs)
    rejected_hidden_last = rejected_hidden[torch.arange(B)[:, None], rejected_len, :]

    chosen_reward = reward_head_forward(chosen_hidden_last, reward_head.weight, reward_head.bias)
    rejected_reward = reward_head_forward(rejected_hidden_last, reward_head.weight, reward_head.bias)

    loss = pairwise_reward_loss(chosen_reward, rejected_reward)
    accuracy = pairwise_accuracy(chosen_reward, rejected_reward)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return dict(
        loss=loss.item(),
        accuracy=accuracy
    )

# Step 41 - sequence_logprob
import torch
import torch.nn.functional as F

def sequence_logprob(logits, token_ids):
    """Sum log probabilities of the selected tokens along the sequence dimension."""
    # TODO: return a scalar tensor equal to sum_t log_softmax(logits)[t, token_ids[t]]
    t = token_ids.shape[0]
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    logsumexp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    log_softmax = shifted - logsumexp

    return log_softmax[torch.arange(t), token_ids].sum()

# Step 42 - per_token_kl
import numpy as np

def per_token_kl(policy_logprobs, ref_logprobs):
    """Per-token KL estimate between policy and reference log-probs."""
    # TODO: return the per-token KL contribution used in the PPO penalty
    return (policy_logprobs - ref_logprobs)

# Step 43 - compute_returns
import numpy as np

def compute_returns(rewards, gamma=0.99):
    """Return the discounted return at each timestep as a 1D numpy array."""
    # TODO: turn a per-timestep reward sequence into discounted returns
    t = len(rewards)
    returns = [rewards[-1]]

    for i in range(t-2, -1, -1):
        returns.append(rewards[i] + returns[-1]*gamma)

    return np.asarray(list(reversed(returns)))

