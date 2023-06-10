from model import *
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameters import *
import numpy as np
import os

torch.manual_seed(1337)

# load data
with open('headlines.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# "poor man's dataloader" - karpathy
data_dir = 'data'
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# instantiate model
model_args = dict(n_layer=num_layer, n_head=num_heads, n_embd=num_embed, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
model_args['vocab_size'] = 50257 # GPT2 vocab size, can change to manual vocab size
config = GPTConfig(**model_args)
model = GPT(config)
m = model.to(device)
# print number of model parameters
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create optimizer
optim = torch.optim.Adam(model.parameters(), lr=lr)

# training loop
for i in range(max_iters):

    # estimate loss
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f'iteration {i} | train loss {losses["train"]} | val loss {losses["val"]}')
    
    # sample a batch of data
    X, Y = get_batch('train')

    # evaluate loss and backprop
    logits, loss = model(X, Y)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

# generate some samples
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# save model
torch.save(model.state_dict(), 'model_weights')