from model import *
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from hyperparameters import *
import pickle
from contextlib import nullcontext
import tiktoken

# train from GPT-2 checkpoint

dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# instantiate model
init_from = 'gpt2'
print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
# initialize from OpenAI GPT-2 weights
model = GPT.from_pretrained(init_from)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
m = model.to(device)

# print number of model parameters
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create optimizer
optim = torch.optim.Adam(model.parameters(), lr=lr)
#scheduler = ExponentialLR(optim, gamma=0.99)

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
    #scheduler.step()

# generate some samples
context = torch.zeros((1,1), dtype=torch.long, device=device)
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# save model
torch.save(model.state_dict(), 'model_weights')