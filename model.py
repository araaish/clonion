import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from hyperparameters import *

torch.manual_seed(1337)

class Head(nn.Module):
    """ A single attention head """
    def __init__(self, head_size):
        super().__init__()
        self.q = nn.Linear(num_embed, head_size, bias=False)
        self.k = nn.Linear(num_embed, head_size, bias=False)
        self.v = nn.Linear(num_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        att = (q @ k.transpose(-2, -1) * k.shape[-1]**(-0.5))
        att = att.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v
        return out

class CausalSelfAttention(nn.Module):
    """ A multi-head, causal self-attention layer """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, num_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class MLP(nn.Module):
    """ Linear layer for computation """
    def __init__(self, num_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embed, num_embed * 4),
            nn.ReLU(),
            nn.Linear(num_embed * 4, num_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block based on nanoGPT """
    def __init__(self, num_embed, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(num_embed)
        self.ln2 = nn.LayerNorm(num_embed)
        self.attn = CausalSelfAttention(num_heads, num_embed // num_heads)
        self.mlp = MLP(num_embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_embed)
        self.pos_embedding_table = nn.Embedding(block_size, num_embed)
        self.blocks = nn.Sequential(*[Block(num_embed, num_heads) for _ in range(num_layer)])
        self.ln = nn.LayerNorm(num_embed)
        self.head = nn.Linear(num_embed, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x, targets=None):
        B,T = x.shape
        token_embed = self.token_embedding_table(x)
        pos_embed = self.pos_embedding_table(torch.arange(T, device=device))
        x = token_embed + pos_embed
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx
    
