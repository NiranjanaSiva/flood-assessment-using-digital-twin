from pathlib import Path
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation
)

# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "nvidia/mit-b2"

DATASET_DIR = Path("AIFloodSense/processed")

TRAIN_IMAGE_DIR = DATASET_DIR / "train" / "images"
TRAIN_MASK_DIR = DATASET_DIR / "train" / "masks"

VAL_IMAGE_DIR = DATASET_DIR / "val" / "images"
VAL_MASK_DIR = DATASET_DIR / "val" / "masks"

OUTPUT_DIR = Path("aifloodsense_flood_model")

NUM_CLASSES = 2

EPOCHS = 1
BATCH_SIZE = 1
IMAGE_SIZE = 512
LEARNING_RATE = 5e-5
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================================
# PRINT DEVICE
# =========================================================

print("=" * 60)
print("AIFloodSense Flood Segmentation Training")
print("=" * 60)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU not available - using CPU")

# =========================================================
# IMAGE PROCESSOR
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

class FloodDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.images = sorted(
            self.image_dir.glob("*.jpg")
        )

        # Keep only images that have masks
        self.images = [
            image_path
            for image_path in self.images
            if (self.mask_dir / f"{image_path.stem}.png").exists()
        ]

        print(
            f"Loaded {len(self.images)} image/mask pairs "
            f"from {self.image_dir}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        mask_path = (
            self.mask_dir /
            f"{image_path.stem}.png"
        )

        # -------------------------------------------------
        # Load image
        # -------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        # -------------------------------------------------
        # Load binary flood mask
        #
        # 0   = background
        # 255 = flood
        # -------------------------------------------------

        mask = Image.open(mask_path).convert("L")

        # Resize image
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        # Resize mask using NEAREST
        # so class labels are not mixed
        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        mask = np.array(mask)

        # Convert:
        #
        # 0   → 0 background
        # 255 → 1 flood
        #
        mask = (mask == 255).astype(np.int64)

        # -------------------------------------------------
        # Process image
        # -------------------------------------------------

        encoded = processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = encoded["pixel_values"].squeeze(0)

        mask = torch.tensor(
            mask,
            dtype=torch.long
        )

        return {
            "pixel_values": pixel_values,
            "labels": mask
        }


# =========================================================
# CREATE DATASETS
# =========================================================

train_dataset = FloodDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR
)

val_dataset = FloodDataset(
    VAL_IMAGE_DIR,
    VAL_MASK_DIR
)

# =========================================================
# DATA LOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading SegFormer model...")

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

model.to(DEVICE)

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# =========================================================
# TRAINING
# =========================================================

best_val_loss = float("inf")

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    # =====================================================
    # TRAIN
    # =====================================================

    model.train()

    train_loss = 0.0

    for batch_index, batch in enumerate(train_loader):

        pixel_values = batch["pixel_values"].to(DEVICE)

        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        if (batch_index + 1) % 20 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_index + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    train_loss /= len(train_loader)

    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for batch in val_loader:

            pixel_values = batch["pixel_values"].to(DEVICE)

            labels = batch["labels"].to(DEVICE)

            outputs = model(
                pixel_values=pixel_values,
                labels=labels
            )

            val_loss += outputs.loss.item()

    val_loss /= len(val_loader)

    print("\n" + "-" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Training Loss   : {train_loss:.4f}"
    )

    print(
        f"Validation Loss : {val_loss:.4f}"
    )

    print("-" * 60 + "\n")

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        print("New best model!")

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        model.save_pretrained(
            OUTPUT_DIR
        )

        processor.save_pretrained(
            OUTPUT_DIR
        )

        print(
            f"Model saved to: {OUTPUT_DIR}"
        )

# =========================================================
# FINISHED
# =========================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation loss: {best_val_loss:.4f}"
)

print(
    f"Model location: {OUTPUT_DIR}"
)