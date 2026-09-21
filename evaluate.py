import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from data_loader import test_loader, CLASS_NAMES
from model import create_model


# =========================
# Configuration
# =========================

DEVICE = torch.device("cpu")
CHECKPOINT_PATH = "best_nail_vit_tiny.pth"


# =========================
# Load model
# =========================

model = create_model()

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# =========================
# Test
# =========================

all_labels = []
all_predictions = []

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# =========================
# Metrics
# =========================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# =========================
# Results
# =========================

print("\n" + "=" * 60)
print("NAIL DISEASE MODEL — TEST RESULTS")
print("=" * 60)

print(f"\nTest samples: {total}")

print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")


# =========================
# Per-class report
# =========================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# =========================
# Confusion matrix
# =========================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\nClass order:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{i} = {name}")

print("\nRows = Actual")
print("Columns = Predicted\n")

print(cm)