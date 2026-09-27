from pathlib import Path
import random
import shutil

IMAGE_DIR = Path("AIFloodSense/images")
MASK_DIR = Path("AIFloodSense/flood_masks")

OUTPUT_DIR = Path("AIFloodSense/processed")

TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

# ---------------------------------------------------------
# Find matching image/mask pairs
# ---------------------------------------------------------

image_files = sorted(IMAGE_DIR.glob("*.jpg"))

pairs = []

for image_path in image_files:

    mask_path = MASK_DIR / f"{image_path.stem}.png"

    if mask_path.exists():
        pairs.append((image_path, mask_path))

print(f"Total pairs: {len(pairs)}")

# ---------------------------------------------------------
# Shuffle
# ---------------------------------------------------------

random.shuffle(pairs)

split_index = int(len(pairs) * TRAIN_RATIO)

train_pairs = pairs[:split_index]
val_pairs = pairs[split_index:]

print(f"Training pairs   : {len(train_pairs)}")
print(f"Validation pairs : {len(val_pairs)}")

# ---------------------------------------------------------
# Create folders
# ---------------------------------------------------------

train_image_dir = OUTPUT_DIR / "train" / "images"
train_mask_dir = OUTPUT_DIR / "train" / "masks"

val_image_dir = OUTPUT_DIR / "val" / "images"
val_mask_dir = OUTPUT_DIR / "val" / "masks"

for directory in [
    train_image_dir,
    train_mask_dir,
    val_image_dir,
    val_mask_dir
]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Copy training data
# ---------------------------------------------------------

for image_path, mask_path in train_pairs:

    shutil.copy2(
        image_path,
        train_image_dir / image_path.name
    )

    shutil.copy2(
        mask_path,
        train_mask_dir / mask_path.name
    )

# ---------------------------------------------------------
# Copy validation data
# ---------------------------------------------------------

for image_path, mask_path in val_pairs:

    shutil.copy2(
        image_path,
        val_image_dir / image_path.name
    )

    shutil.copy2(
        mask_path,
        val_mask_dir / mask_path.name
    )

print("\nDataset split completed.")

print("\nOutput structure:")
print(OUTPUT_DIR)
print("├── train")
print("│   ├── images")
print("│   └── masks")
print("│")
print("└── val")
print("    ├── images")
print("    └── masks")