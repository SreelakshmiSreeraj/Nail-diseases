import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

from data_loader import train_loader, val_loader, CLASS_NAMES, NUM_CLASSES
from model import create_model


# =========================
# Configuration
# =========================

DEVICE = torch.device("cpu")

NUM_EPOCHS = 20
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.05

PATIENCE = 5

CHECKPOINT_PATH = "best_nail_vit_tiny.pth"


# =========================
# Model
# =========================

model = create_model()
model = model.to(DEVICE)

print("\nDevice:", DEVICE)
print("Classes:", CLASS_NAMES)


# =========================
# Class weights
# =========================

train_targets = train_loader.dataset.targets

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=np.array(train_targets)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights:")

for class_name, weight in zip(CLASS_NAMES, class_weights):
    print(f"{class_name}: {weight.item():.4f}")


# =========================
# Loss function
# =========================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# =========================
# Optimizer
# =========================

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# =========================
# Learning-rate scheduler
# =========================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# =========================
# Training
# =========================

best_val_loss = float("inf")

epochs_without_improvement = 0


for epoch in range(NUM_EPOCHS):

    # =========================
    # TRAIN
    # =========================

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total


    # =========================
    # VALIDATION
    # =========================

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_running_loss / val_total
    val_accuracy = val_correct / val_total


    # =========================
    # Scheduler
    # =========================

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # =========================
    # Print results
    # =========================

    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy * 100:.2f}%"
    )

    print(
        f"Learning Rate: {current_lr:.6f}"
    )


    # =========================
    # Save best model
    # =========================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

        epochs_without_improvement = 0

        print(
            f"Best model saved "
            f"(Val Loss: {val_loss:.4f})"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )


    # =========================
    # Early stopping
    # =========================

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")

        break


# =========================
# Training complete
# =========================

print("\n" + "=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)

print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best model saved to: {CHECKPOINT_PATH}")