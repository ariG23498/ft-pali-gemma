def freeze_layers(model, not_to_freeze):
    for name, param in model.named_parameters():
        if not_to_freeze in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model
