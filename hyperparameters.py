import torch

# hyperparameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_type = 'cuda'
num_embed = 384
num_heads = 6
num_layer = 6
dropout = 0.2

# training from scratch
# block_size = 256
# batch_size = 64
# max_iters = 5000
# eval_interval = 500
# lr = 3e-4
# eval_iters = 200
# -----------------

# finetuning from pretrained

block_size = 256
batch_size = 16
max_iters = 500
eval_interval = 50
lr = 1e-4
eval_iters = 40
