import torch

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
lr = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
num_embed = 384
num_heads = 6
num_layer = 6
dropout = 0.2
# -----------------

