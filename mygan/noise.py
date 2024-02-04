import torch


def gen_noise(batch_size, z_dim):
  return torch.randn(batch_size, z_dim)
