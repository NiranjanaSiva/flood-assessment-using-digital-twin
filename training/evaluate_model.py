from pathlib import Path

import numpy as np
import torch
from PIL import Image

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation
)


# =========================================================
# CONFIG
# =========================================================

MODEL_DIR = Path("aifloodsense_flood_model")

IMAGE_DIR = Path(
    "AIFloodSense/processed/val/images"
)

MASK_DIR = Path(
    "AIFloodSense/processed/val/masks"
)

IMAGE_SIZE = 512

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("AIFloodSense Model Evaluation")
print("=" * 60)

print("Device:", DEVICE)

processor = SegformerImageProcessor.from_pretrained(
    MODEL_DIR
)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)
model.eval()


# =========================================================
# FIND VALIDATION IMAGES
# =========================================================

image_files = sorted(
    IMAGE_DIR.glob("*.jpg")
)

print("Validation images:", len(image_files))


# =========================================================
# METRIC STORAGE
# =========================================================

ious = []
dices = []
precisions = []
recalls = []

total_intersection = 0
total_union = 0
total_predicted = 0
total_actual = 0
total_true_positive = 0


# =========================================================
# PROCESS EACH IMAGE
# =========================================================

for index, image_path in enumerate(image_files):

    mask_path = (
        MASK_DIR /
        f"{image_path.stem}.png"
    )

    if not mask_path.exists():
        print(
            f"Skipping {image_path.name}: "
            "mask not found"
        )
        continue

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    original_size = image.size

    input_image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    # -----------------------------------------------------
    # Prepare input
    # -----------------------------------------------------

    inputs = processor(
        images=input_image,
        return_tensors="pt"
    )

    pixel_values = (
        inputs["pixel_values"]
        .to(DEVICE)
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    with torch.no_grad():

        outputs = model(
            pixel_values=pixel_values
        )

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        )

    prediction = (
        prediction
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # -----------------------------------------------------
    # Resize prediction back to original size
    # -----------------------------------------------------

    prediction_image = Image.fromarray(
        (prediction * 255).astype(np.uint8)
    )

    prediction_image = prediction_image.resize(
        original_size,
        Image.Resampling.NEAREST
    )

    prediction = np.array(
        prediction_image
    )

    prediction = (
        prediction > 127
    ).astype(np.uint8)

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    ground_truth = np.array(
        Image.open(
            mask_path
        ).convert("L")
    )

    ground_truth = (
        ground_truth == 255
    ).astype(np.uint8)

    # -----------------------------------------------------
    # Calculate confusion values
    # -----------------------------------------------------

    pred_flood = prediction == 1
    true_flood = ground_truth == 1

    intersection = np.logical_and(
        pred_flood,
        true_flood
    ).sum()

    union = np.logical_or(
        pred_flood,
        true_flood
    ).sum()

    predicted_pixels = pred_flood.sum()
    actual_pixels = true_flood.sum()

    true_positive = intersection

    false_positive = np.logical_and(
        pred_flood,
        ~true_flood
    ).sum()

    false_negative = np.logical_and(
        ~pred_flood,
        true_flood
    ).sum()

    # -----------------------------------------------------
    # IoU
    # -----------------------------------------------------

    if union > 0:
        iou = intersection / union
    else:
        iou = 1.0

    # -----------------------------------------------------
    # Dice
    # -----------------------------------------------------

    denominator = (
        predicted_pixels +
        actual_pixels
    )

    if denominator > 0:
        dice = (
            2 * intersection
        ) / denominator
    else:
        dice = 1.0

    # -----------------------------------------------------
    # Precision
    # -----------------------------------------------------

    precision_denominator = (
        true_positive +
        false_positive
    )

    if precision_denominator > 0:
        precision = (
            true_positive /
            precision_denominator
        )
    else:
        precision = 1.0

    # -----------------------------------------------------
    # Recall
    # -----------------------------------------------------

    recall_denominator = (
        true_positive +
        false_negative
    )

    if recall_denominator > 0:
        recall = (
            true_positive /
            recall_denominator
        )
    else:
        recall = 1.0

    # -----------------------------------------------------
    # Store metrics
    # -----------------------------------------------------

    ious.append(iou)
    dices.append(dice)
    precisions.append(precision)
    recalls.append(recall)

    total_intersection += intersection
    total_union += union
    total_predicted += predicted_pixels
    total_actual += actual_pixels
    total_true_positive += true_positive

    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    print(
        f"[{index + 1}/{len(image_files)}] "
        f"{image_path.name} | "
        f"IoU: {iou:.4f} | "
        f"Dice: {dice:.4f}"
    )


# =========================================================
# FINAL RESULTS
# =========================================================

print("\n")
print("=" * 60)
print("FINAL VALIDATION RESULTS")
print("=" * 60)

print(
    f"Images evaluated : {len(ious)}"
)

print(
    f"Mean IoU         : "
    f"{np.mean(ious):.4f}"
)

print(
    f"Mean Dice        : "
    f"{np.mean(dices):.4f}"
)

print(
    f"Mean Precision   : "
    f"{np.mean(precisions):.4f}"
)

print(
    f"Mean Recall      : "
    f"{np.mean(recalls):.4f}"
)

print("=" * 60)

print("\nPercentage values:")

print(
    f"Mean IoU       : "
    f"{np.mean(ious) * 100:.2f}%"
)

print(
    f"Mean Dice      : "
    f"{np.mean(dices) * 100:.2f}%"
)

print(
    f"Mean Precision : "
    f"{np.mean(precisions) * 100:.2f}%"
)

print(
    f"Mean Recall    : "
    f"{np.mean(recalls) * 100:.2f}%"
)

print("=" * 60)