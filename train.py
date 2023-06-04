from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameters import *

torch.manual_seed(1337)

# load data
with open('headlines.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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
model = GPT(vocab_size)
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
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# save model
torch.save(model.state_dict(), 'model_weights')