import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):

    def __init__(self, in_features, out_features, rank=8, alpha = 16):

        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Lora Matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform for A, Zeros for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)


class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16, use_dora=False):
        super().__init__()
        self.original_layer = original_layer
        self.use_dora = use_dora
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora = LoRALayer(in_features, out_features, rank, alpha)
        
        if use_dora:
            # DoRA: Magnitude vector 'm'
            # Initialize with the magnitude of the original weight matrix
            self.m = nn.Parameter(original_layer.weight.data.norm(p=2, dim=1, keepdim=True))

    def forward(self, x):
        if not self.use_dora:
            # Standard LoRA logic
            original_out = self.original_layer(x)
            lora_out = (x @ self.lora.lora_A.t()) @ self.lora.lora_B.t()
            return original_out + (lora_out * self.lora.scaling)
        else:
            # DoRA Logic: Reconstruct weight W' = m * (W + BA) / ||W + BA||
            W_orig = self.original_layer.weight
            delta_W = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            W_combined = W_orig + delta_W
            
            # Normalize direction and apply magnitude m
            column_norm = W_combined.norm(p=2, dim=1, keepdim=True)
            W_dora = self.m * (W_combined / column_norm)
            
            return F.linear(x, W_dora, self.original_layer.bias)