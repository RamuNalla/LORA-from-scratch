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
        
        # Check if it's a standard Linear or a GPT2 Conv1D layer
        if hasattr(original_layer, 'in_features'):
            in_features = original_layer.in_features
            out_features = original_layer.out_features
            self.is_conv1d = False
        elif hasattr(original_layer, 'nx'): # For GPT-2 Conv1D
            in_features = original_layer.nx
            out_features = original_layer.nf
            self.is_conv1d = True
        else:
            raise AttributeError("Layer must be nn.Linear or Conv1D")
        
        self.lora = LoRALayer(in_features, out_features, rank, alpha)
        
        if use_dora:
            # For DoRA, we need the norm. Conv1D weights are (nx, nf)
            if self.is_conv1d:
                # Norm along the input dimension (dim=0 for Conv1D)
                self.m = nn.Parameter(original_layer.weight.data.norm(p=2, dim=0, keepdim=True))
            else:
                # Norm along input dimension (dim=1 for Linear)
                self.m = nn.Parameter(original_layer.weight.data.norm(p=2, dim=1, keepdim=True))

    def forward(self, x):
        if not self.use_dora:
            # Standard LoRA logic
            original_out = self.original_layer(x)
            lora_out = (x @ self.lora.lora_A.t()) @ self.lora.lora_B.t()
            return original_out + (lora_out * self.lora.scaling)
        else:
            # DoRA requires weight reconstruction
            W_orig = self.original_layer.weight
            delta_W = (self.lora.lora_B @ self.lora.lora_A).t() if self.is_conv1d else (self.lora.lora_B @ self.lora.lora_A)
            delta_W = delta_W * self.lora.scaling
            
            W_combined = W_orig + delta_W
            
            # Normalize direction and apply magnitude m
            # Conv1D weights are (in, out), Linear are (out, in)
            norm_dim = 0 if self.is_conv1d else 1
            column_norm = W_combined.norm(p=2, dim=norm_dim, keepdim=True)
            W_dora = self.m * (W_combined / column_norm)
            
            if self.is_conv1d:
                # Manual implementation of Conv1D's forward: x @ W + b
                return (x @ W_dora) + self.original_layer.bias
            else:
                return F.linear(x, W_dora, self.original_layer.bias)