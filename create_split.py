from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

SOURCE_DIR = Path("dataset")
OUTPUT_DIR = Path("dataset_split")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Get classes from the original train folder
classes = sorted([
    folder.name
    for folder in (SOURCE_DIR / "train").iterdir()
    if folder.is_dir()
])

print("Classes found:")
for cls in classes:
    print(f"  - {cls}")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Process each class
for cls in classes:

    images = []

    # Collect images from original train
    train_dir = SOURCE_DIR / "train" / cls

    if train_dir.exists():
        for image in train_dir.iterdir():
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(image)

    # Collect images from original validation
    val_dir = SOURCE_DIR / "validation" / cls

    if val_dir.exists():
        for image in val_dir.iterdir():
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(image)

    print(f"\n{cls}: {len(images)} images")

    # 70% train, 30% temporary
    train_images, temp_images = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    # 15% validation, 15% test
    val_images, test_images = train_test_split(
        temp_images,
        test_size=0.50,
        random_state=42
    )

    # Create class folders
    for split in ["train", "validation", "test"]:
        (OUTPUT_DIR / split / cls).mkdir(
            parents=True,
            exist_ok=True
        )

    # Copy images
    for image in train_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "train" / cls / image.name
        )

    for image in val_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "validation" / cls / image.name
        )

    for image in test_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "test" / cls / image.name
        )

    print(f"  Train:      {len(train_images)}")
    print(f"  Validation: {len(val_images)}")
    print(f"  Test:       {len(test_images)}")


print("\n" + "=" * 50)
print("DATASET SPLIT COMPLETE")
print("=" * 50)