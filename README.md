# Nail Disease Classification using ViT-Tiny

A deep learning model for classifying nail images into six different classes using a pretrained Vision Transformer (ViT-Tiny).

## Classes

The model classifies images into the following six categories:

- Acral Lentiginous Melanoma
- Healthy Nail
- Onychogryphosis
- Blue Finger
- Clubbing
- Pitting

## Model

- Architecture: ViT-Tiny (`vit_tiny_patch16_224`)
- Input size: 224 × 224
- Transfer learning: Pretrained ImageNet weights
- Fine-tuning: Last two Transformer blocks, normalization layer, and classification head
- Number of classes: 6
- Optimizer: AdamW
- Loss: Class-weighted Cross Entropy Loss
- Learning rate: 2e-5
- Weight decay: 0.05
- Batch size: 16
- Training epochs: 20
- Early stopping: Patience of 5 epochs

## Dataset

The dataset contains 3,835 nail images distributed across six classes.

The data was divided into:

| Split | Images |
|---|---:|
| Training | 2,683 |
| Validation | 574 |
| Testing | 578 |
| **Total** | **3,835** |

The test set was kept separate for final evaluation.

## Results

The trained model achieved the following results on the test set:

- **Accuracy:** 85.47%
- **Weighted Precision:** 85.63%
- **Weighted Recall:** 85.47%
- **Weighted F1-Score:** 85.51%
- **Macro F1-Score:** 86.42%

### Per-Class F1-Score

| Class | F1-Score |
|---|---:|
| Acral Lentiginous Melanoma | 84.35% |
| Healthy Nail | 96.08% |
| Onychogryphosis | 90.38% |
| Blue Finger | 78.45% |
| Clubbing | 80.17% |
| Pitting | 89.12% |

## Explainability

Grad-CAM is used to visualize image regions that contribute to the model's prediction.

The implementation is adapted for the Vision Transformer architecture by using the final Transformer block and reshaping the patch activations into a spatial feature map.

The visualizations are intended to provide model interpretability and do not represent medical causation or clinical validation.

## Project Structure

```text
naildiseases/
│
├── data_loader.py
├── model.py
├── train.py
├── evaluate.py
├── explain.py
├── check_dataset.py
├── check_duplicates.py
├── create_split.py
├── best_nail_vit_tiny.pth
├── .gitignore
└── README.md
