from src.lora_layers import LoRALinear

def inject_adapter(model, rank=8, alpha=16, use_dora=False, target_modules=["c_attn"]):
    """
    Freezes the model and replaces target layers with LoRA/DoRA wrappers.
    """
    # 1. Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # 2. Swap layers
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            # Extract parent and child name
            parts = name.split(".")
            parent_name = ".".join(parts[:-1])
            child_name = parts[-1]
            
            parent = dict(model.named_modules())[parent_name]
            
            # Create the adapter
            new_layer = LoRALinear(module, rank=rank, alpha=alpha, use_dora=use_dora)
            
            # Replace
            setattr(parent, child_name, new_layer)
            
    print(f"Injected {'DoRA' if use_dora else 'LoRA'} into: {target_modules}")
    return model