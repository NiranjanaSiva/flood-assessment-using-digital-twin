from pathlib import Path
from PIL import Image
import numpy as np

IMAGE_DIR = Path("AIFloodSense/images")
MASK_DIR = Path("AIFloodSense/flood_masks")

images = {p.stem: p for p in IMAGE_DIR.glob("*.jpg")}
masks = {p.stem: p for p in MASK_DIR.glob("*.png")}

print("=" * 60)
print("AIFloodSense Dataset Verification")
print("=" * 60)

print(f"Images found : {len(images)}")
print(f"Masks found  : {len(masks)}")

# ---------------------------------------------------------
# Check matching files
# ---------------------------------------------------------

missing_masks = sorted(set(images) - set(masks))
missing_images = sorted(set(masks) - set(images))

if missing_masks:
    print("\nImages WITHOUT masks:")
    print(missing_masks)

if missing_images:
    print("\nMasks WITHOUT images:")
    print(missing_images)

if not missing_masks and not missing_images:
    print("\n✓ Every image has a matching flood mask.")

# ---------------------------------------------------------
# Check dimensions and flood pixels
# ---------------------------------------------------------

bad_dimensions = []
empty_masks = []
statistics = []

for image_id in sorted(images):

    if image_id not in masks:
        continue

    image_path = images[image_id]
    mask_path = masks[image_id]

    image = Image.open(image_path)
    mask = np.array(Image.open(mask_path))

    width, height = image.size

    if mask.shape != (height, width):
        bad_dimensions.append(
            f"{image_id}: image={image.size}, mask={mask.shape}"
        )

    flood_pixels = np.sum(mask == 255)
    total_pixels = mask.size

    flood_percentage = (flood_pixels / total_pixels) * 100

    statistics.append(
        (image_id, width, height, flood_pixels, flood_percentage)
    )

    if flood_pixels == 0:
        empty_masks.append(image_id)

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Dimension Check")
print("=" * 60)

if not bad_dimensions:
    print("✓ All image and mask dimensions match.")
else:
    print("✗ Dimension mismatches found:")
    for item in bad_dimensions:
        print(item)

print("\n" + "=" * 60)
print("Empty Flood Masks")
print("=" * 60)

print(f"Images with no flood pixels: {len(empty_masks)}")

if empty_masks:
    print("IDs:")
    print(", ".join(empty_masks))

# ---------------------------------------------------------
# Flood statistics
# ---------------------------------------------------------

if statistics:

    percentages = [
        row[4]
        for row in statistics
    ]

    print("\n" + "=" * 60)
    print("Flood Statistics")
    print("=" * 60)

    print(f"Minimum flood area : {min(percentages):.2f}%")
    print(f"Maximum flood area : {max(percentages):.2f}%")
    print(f"Average flood area : {sum(percentages) / len(percentages):.2f}%")

    # Largest flood images
    largest = sorted(
        statistics,
        key=lambda x: x[4],
        reverse=True
    )[:10]

    print("\nTop 10 images by flood area:")

    for image_id, width, height, pixels, percentage in largest:
        print(
            f"{image_id}: "
            f"{percentage:.2f}% flood "
            f"({pixels:,} pixels)"
        )

print("\n" + "=" * 60)
print("Verification complete.")
print("=" * 60)