import torch
import torch.nn as nn
import numpy as np

# Utility functions for model analysis and training

def count_and_list_parameters(model):
    """Count and display model parameters."""
    total_params = 0
    print(f"{'Layer':50} {'Shape':25} {'# Params'}")
    print('-' * 80)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f"{name:50} {str(tuple(param.shape)):25} {param_count}")
    print('-' * 80)
    print(f"{'Total trainable parameters:':50} {total_params}")
    return total_params

def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def dropout_no_scaling(x, p=0.5):
    """Custom dropout without scaling."""
    mask = (torch.rand_like(x) > p).float()
    return x * mask
