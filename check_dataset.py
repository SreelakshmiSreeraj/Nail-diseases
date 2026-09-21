from pathlib import Path

DATASET_DIR = Path("dataset_split")

for split in ["train", "validation","test"]:
    print(f"\n{split.upper()}")
    print("-" * 40)

    split_dir = DATASET_DIR / split

    total = 0

    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir():
            count = len([
                f for f in class_dir.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
            ])

            print(f"{class_dir.name}: {count}")
            total += count

    print("-" * 40)
    print(f"Total: {total}")