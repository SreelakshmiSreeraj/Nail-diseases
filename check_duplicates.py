from pathlib import Path
import hashlib

DATASET_DIR = Path("dataset")


def get_hash(file_path):
    """Create a unique hash for an image file."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


images = []

for split in ["train", "validation"]:
    split_dir = DATASET_DIR / split

    for class_dir in split_dir.iterdir():
        if class_dir.is_dir():
            for image in class_dir.iterdir():
                if image.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    images.append((split, class_dir.name, image))


print(f"Total images checked: {len(images)}")

# Group images by hash
hashes = {}

for split, class_name, image in images:
    image_hash = get_hash(image)

    if image_hash not in hashes:
        hashes[image_hash] = []

    hashes[image_hash].append((split, class_name, image))


# Find duplicates
duplicates = {
    h: files
    for h, files in hashes.items()
    if len(files) > 1
}

print(f"Unique images: {len(hashes)}")
print(f"Duplicate groups: {len(duplicates)}")

if duplicates:
    print("\nDUPLICATES FOUND:")
    print("=" * 60)

    for i, files in enumerate(duplicates.values(), 1):
        print(f"\nDuplicate group {i}:")

        for split, class_name, image in files:
            print(f"  {split}/{class_name}/{image.name}")

else:
    print("\nNo exact duplicate images found.")