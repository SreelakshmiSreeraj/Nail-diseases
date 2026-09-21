import cv2
import numpy as np
import torch
import timm

from pathlib import Path
from PIL import Image

from torchvision import transforms
from model import create_model


# =========================
# Configuration
# =========================

DEVICE = torch.device("cpu")

CHECKPOINT_PATH = "best_nail_vit_tiny.pth"

IMAGE_SIZE = 224

CLASS_NAMES = [
    "Acral_Lentiginous_Melanoma",
    "Healthy_Nail",
    "Onychogryphosis",
    "blue_finger",
    "clubbing",
    "pitting"
]

# Change this to any test image you want to explain
IMAGE_PATH = Path(
    "dataset_split/test/Onychogryphosis"
)

OUTPUT_DIR = Path("explainability_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# Find first image
# =========================

image_files = [
    f for f in IMAGE_PATH.iterdir()
    if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
]

if not image_files:
    raise FileNotFoundError(
        f"No images found in {IMAGE_PATH}"
    )

IMAGE_PATH = image_files[0]

print("Image:", IMAGE_PATH)


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
# Transform
# =========================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# Load image
# =========================

original_image = Image.open(
    IMAGE_PATH
).convert("RGB")

input_tensor = transform(
    original_image
).unsqueeze(0)

input_tensor.requires_grad_(True)


# =========================
# Grad-CAM hook
# =========================

activations = []
gradients = []


def forward_hook(module, input, output):
    activations.append(output)


def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])


target_layer = model.blocks[-1].norm1

forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# =========================
# Forward pass
# =========================

output = model(input_tensor)

probabilities = torch.softmax(
    output,
    dim=1
)

predicted_class = output.argmax(
    dim=1
).item()

confidence = probabilities[
    0, predicted_class
].item()


print("\nPrediction:", CLASS_NAMES[predicted_class])
print(f"Confidence: {confidence * 100:.2f}%")


# =========================
# Backward pass
# =========================

model.zero_grad()

score = output[
    0, predicted_class
]

score.backward()


# =========================
# Extract activations
# =========================

activation = activations[0]
gradient = gradients[0]

print("Activation shape:", activation.shape)
print("Gradient shape:", gradient.shape)


# =========================
# Remove CLS token
# =========================

activation = activation[:, 1:, :]
gradient = gradient[:, 1:, :]


# =========================
# Reshape patches
# =========================

num_patches = activation.shape[1]

grid_size = int(
    np.sqrt(num_patches)
)

activation = activation.reshape(
    1,
    grid_size,
    grid_size,
    -1
)

gradient = gradient.reshape(
    1,
    grid_size,
    grid_size,
    -1
)


# =========================
# Grad-CAM
# =========================

weights = gradient.mean(
    dim=(1, 2),
    keepdim=True
)

cam = (
    activation * weights
).sum(dim=-1)

cam = torch.relu(cam)

cam = cam[0].detach().cpu().numpy()

# Normalize
cam -= cam.min()

if cam.max() != 0:
    cam /= cam.max()


# =========================
# Resize heatmap
# =========================

heatmap = cv2.resize(
    cam,
    (IMAGE_SIZE, IMAGE_SIZE)
)

heatmap_uint8 = np.uint8(
    255 * heatmap
)

heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)


# =========================
# Original image
# =========================

original_cv = cv2.cvtColor(
    np.array(original_image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )),
    cv2.COLOR_RGB2BGR
)


# =========================
# Overlay
# =========================

overlay = cv2.addWeighted(
    original_cv,
    0.55,
    heatmap_color,
    0.45,
    0
)


# =========================
# Save results
# =========================

output_path = (
    OUTPUT_DIR /
    f"gradcam_{IMAGE_PATH.stem}.jpg"
)

cv2.imwrite(
    str(output_path),
    overlay
)


# Save original too
original_path = (
    OUTPUT_DIR /
    f"original_{IMAGE_PATH.stem}.jpg"
)

cv2.imwrite(
    str(original_path),
    original_cv
)


# =========================
# Cleanup hooks
# =========================

forward_handle.remove()
backward_handle.remove()


print("\nGrad-CAM saved to:")
print(output_path)