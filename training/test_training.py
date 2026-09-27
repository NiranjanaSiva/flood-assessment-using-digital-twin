from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation
)


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "nvidia/mit-b2"

IMAGE_DIR = Path("AIFloodSense/processed/train/images")
MASK_DIR = Path("AIFloodSense/processed/train/masks")

IMAGE_SIZE = 512
BATCH_SIZE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# PROCESSOR
# =========================================================

processor = SegformerImageProcessor(
    do_resize=True,
    size={
        "height": IMAGE_SIZE,
        "width": IMAGE_SIZE
    },
    do_normalize=True
)


# =========================================================
# DATASET
# =========================================================

class FloodDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.images = sorted(
            self.image_dir.glob("*.jpg")
        )

        self.images = [
            image
            for image in self.images
            if (
                self.mask_dir /
                f"{image.stem}.png"
            ).exists()
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        mask_path = (
            self.mask_dir /
            f"{image_path.stem}.png"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        # Resize image
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        # Resize mask using nearest-neighbor
        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        mask = np.array(mask)

        # 255 = flood
        # 0   = background
        mask = (
            mask == 255
        ).astype(np.int64)

        encoded = processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = (
            encoded["pixel_values"]
            .squeeze(0)
        )

        labels = torch.tensor(
            mask,
            dtype=torch.long
        )

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


# =========================================================
# DATA LOADER
# =========================================================

dataset = FloodDataset(
    IMAGE_DIR,
    MASK_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("=" * 60)
print("Training Sanity Test")
print("=" * 60)

print("Device:", DEVICE)
print("Dataset size:", len(dataset))


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading model...")

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True
)

model.to(DEVICE)

model.train()


# =========================================================
# GET ONE BATCH
# =========================================================

batch = next(iter(loader))

pixel_values = batch["pixel_values"].to(DEVICE)
labels = batch["labels"].to(DEVICE)

print("\nBatch information:")
print("Pixel values shape:", pixel_values.shape)
print("Labels shape:", labels.shape)

print(
    "Label values:",
    torch.unique(labels).cpu().numpy()
)


# =========================================================
# FORWARD PASS
# =========================================================

print("\nRunning forward pass...")

outputs = model(
    pixel_values=pixel_values,
    labels=labels
)

print("Logits shape:", outputs.logits.shape)
print("Loss:", outputs.loss.item())


# =========================================================
# BACKWARD PASS
# =========================================================

print("\nRunning backward pass...")

outputs.loss.backward()

print("Backward pass successful!")


# =========================================================
# DONE
# =========================================================

print("\n" + "=" * 60)
print("SANITY TEST PASSED")
print("=" * 60)

print("\nThe following pipeline works:")
print("Image")
print("  ↓")
print("Processor")
print("  ↓")
print("SegFormer")
print("  ↓")
print("2-class prediction")
print("  ↓")
print("Flood mask")
print("  ↓")
print("Loss")
print("  ↓")
print("Backward propagation")