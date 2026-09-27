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

IMAGE_PATH = Path("AIFloodSense/images/387.jpg")
GROUND_TRUTH_PATH = Path("AIFloodSense/flood_masks/387.png")

OUTPUT_PATH = Path("training/prediction_387.png")

IMAGE_SIZE = 512

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# LOAD PROCESSOR
# =========================================================

processor = SegformerImageProcessor.from_pretrained(
    MODEL_DIR
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

print("=" * 60)
print("Testing AIFloodSense Flood Model")
print("=" * 60)

print("Device:", DEVICE)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)
model.eval()


# =========================================================
# LOAD IMAGE
# =========================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")

original_size = image.size

print("Original image size:", original_size)


# =========================================================
# PREPARE IMAGE
# =========================================================

input_image = image.resize(
    (IMAGE_SIZE, IMAGE_SIZE),
    Image.Resampling.BILINEAR
)

inputs = processor(
    images=input_image,
    return_tensors="pt"
)

pixel_values = inputs["pixel_values"].to(DEVICE)


# =========================================================
# MODEL PREDICTION
# =========================================================

print("\nRunning prediction...")

with torch.no_grad():

    outputs = model(
        pixel_values=pixel_values
    )

    # Convert logits to class prediction
    prediction = torch.argmax(
        outputs.logits,
        dim=1
    )

prediction = prediction.squeeze(0).cpu().numpy()

print("Prediction shape:", prediction.shape)


# =========================================================
# RESIZE PREDICTION TO ORIGINAL IMAGE SIZE
# =========================================================

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

# Convert back to 0/1
prediction_binary = (
    prediction > 127
).astype(np.uint8)


# =========================================================
# SAVE PREDICTION
# =========================================================

prediction_output = (
    prediction_binary * 255
).astype(np.uint8)

Image.fromarray(
    prediction_output
).save(OUTPUT_PATH)

print(
    "Prediction saved to:",
    OUTPUT_PATH
)


# =========================================================
# LOAD GROUND TRUTH
# =========================================================

ground_truth = np.array(
    Image.open(
        GROUND_TRUTH_PATH
    ).convert("L")
)

ground_truth = (
    ground_truth == 255
).astype(np.uint8)


# =========================================================
# CALCULATE METRICS
# =========================================================

pred = prediction_binary

true = ground_truth


# ---------------------------------------------------------
# Intersection and Union
# ---------------------------------------------------------

intersection = np.logical_and(
    pred == 1,
    true == 1
).sum()

union = np.logical_or(
    pred == 1,
    true == 1
).sum()


# ---------------------------------------------------------
# IoU
# ---------------------------------------------------------

if union == 0:
    iou = 1.0
else:
    iou = intersection / union


# ---------------------------------------------------------
# Dice
# ---------------------------------------------------------

pred_pixels = (
    pred == 1
).sum()

true_pixels = (
    true == 1
).sum()

denominator = pred_pixels + true_pixels

if denominator == 0:
    dice = 1.0
else:
    dice = (
        2 * intersection
    ) / denominator


# ---------------------------------------------------------
# Flood percentages
# ---------------------------------------------------------

total_pixels = pred.size

predicted_percentage = (
    pred_pixels / total_pixels
) * 100

actual_percentage = (
    true_pixels / total_pixels
) * 100


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"Actual flood area     : "
    f"{actual_percentage:.2f}%"
)

print(
    f"Predicted flood area  : "
    f"{predicted_percentage:.2f}%"
)

print(
    f"IoU                   : "
    f"{iou:.4f}"
)

print(
    f"Dice score            : "
    f"{dice:.4f}"
)

print("=" * 60)