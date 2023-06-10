import torch

# hyperparameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_type = 'cuda'
num_embed = 768
num_heads = 12
num_layer = 12
dropout = 0.2
bias = True

# training from scratch
# block_size = 256
# batch_size = 64
# max_iters = 5000
# eval_interval = 500
# lr = 3e-4
# eval_iters = 200
# -----------------

# finetuning from pretrained

block_size = 1024
batch_size = 1
max_iters = 500
eval_interval = 50
lr = 1e-4
eval_iters = 40
