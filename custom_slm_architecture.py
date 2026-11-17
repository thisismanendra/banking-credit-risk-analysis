# Recovered custom SLM architecture
# Source: LendingClub_Qwen3_Final.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import *
from dataclasses import dataclass

import math

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

SEED = 42

important_columns = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'grade', 'purpose', 'loan_status']

TOKENIZER_NAME = 'Qwen/Qwen3-8B'

sample_text = 'loan_amnt: 10000.0 | int_rate: 9.17 | annual_inc: 48000.0'

config = {'vocab_size': 151669, 'n_layer': 4, 'n_head': 4, 'n_kv_head': 2, 'n_embd': 256, 'intermediate_size': 768, 'block_size': 128, 'dropout': 0.0, 'bias': False}

batch_size = 8

max_iters = 4000

learning_rate = 0.0001

weight_decay = 0.1

test_batch_size = 1

test_seq_length = 32

losses = []

prompt = 'Loan amount: 10000'

required_fields = ['Loan amount', 'bc_util', 'dti', 'data_source', 'disbursement_method']

num_records = 5

generated_records = []

command = 'Get-CimInstance Win32_Process -Filter "ProcessId = 30272" | Select-Object ProcessId, ExecutablePath, CommandLine | Format-List'

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

def apply_rotary_emb(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos + rotated * sin

def build_rope_cache(seq_len, head_dim, device):
    inv_freq = 1.0 / 10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    positions = torch.arange(seq_len, device=device).float()
    frequencies = torch.outer(positions, inv_freq)
    frequencies = frequencies.repeat_interleave(2, dim=-1)
    cos = frequencies.cos()[None, None, :, :]
    sin = frequencies.sin()[None, None, :, :]
    return (cos, sin)

class GroupedQueryAttention(nn.Module):

    def __init__(self, d_model, n_heads, n_kv_groups, head_dim, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.head_dim = head_dim
        self.groups = n_heads // n_kv_groups
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_groups * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_groups * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_groups, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_groups, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        cos, sin = build_rope_cache(T, self.head_dim, x.device)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        attention = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        attention = attention.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        output = attention @ v
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(output)

class SwiGLU(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads, n_kv_groups, d_ff, max_seq_len):
        super().__init__()
        head_dim = d_model // n_heads
        self.attention_norm = RMSNorm(d_model)
        self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_groups, head_dim, max_seq_len)
        self.ffn_norm = RMSNorm(d_model)
        self.feed_forward = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class Qwen3Model(nn.Module):

    def __init__(self, vocab_size, d_model, n_heads, n_kv_groups, n_layers, d_ff, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, n_kv_groups, d_ff, max_seq_len) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.max_seq_len:
            raise ValueError('Sequence length exceeds maximum sequence length')
        x = self.token_embedding(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return (logits, loss)

def get_test_batch(batch_size=8, block_size=128):
    ix = torch.randint(len(test_data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(test_data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(test_data[i + 1:i + block_size + 1].astype(np.int64)) for i in ix])
    return (x.to(device), y.to(device))

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    for _ in range(max_new_tokens):
        idx_cond = input_ids[:, -config['block_size']:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            values, _ = torch.topk(logits, top_k, dim=-1)
            threshold = values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float('-inf'))
        probabilities = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

