import torch
import torch.nn as nn
import timm


NUM_CLASSES = 6


def create_model():
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=True,
        num_classes=NUM_CLASSES
    )

    # Freeze the earlier transformer blocks
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last two transformer blocks
    for block in model.blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    # Unfreeze final normalization layer
    for param in model.norm.parameters():
        param.requires_grad = True

    # Unfreeze classification head
    for param in model.head.parameters():
        param.requires_grad = True

    return model


if __name__ == "__main__":
    model = create_model()

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Model: ViT-Tiny")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")