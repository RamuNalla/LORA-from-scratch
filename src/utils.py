def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    percent = 100 * trainable_params / all_param
    print(f"Total Params: {all_param:,}")
    print(f"Trainable Params: {trainable_params:,} ({percent:.4f}%)")
    return {"total": all_param, "trainable": trainable_params, "percent": percent}

    # test