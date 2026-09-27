import os
import numpy as np

from PIL import Image

import torch
import torch.nn.functional as F

from mhnet import MHNet


# ============================================================
# SETTINGS
# ============================================================

IMAGE_PATH = "AIFloodSense/images/387.jpg"

MASK_PATH = "AIFloodSense/flood_masks/387.png"

MODEL_PATH = "mhnet_flood_model/mhnet_best.pth"

OUTPUT_PATH = "training/mhnet_prediction_387.png"

IMAGE_SIZE = 256

BASE_CHANNELS = 32

MASK_RATIO = 0.25


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading MHNet...")

model = MHNet(
    in_channels=3,
    base_channels=BASE_CHANNELS,
    mask_ratio=MASK_RATIO
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# The saved file contains model_state_dict

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

# IMPORTANT:
# eval() disables MHNet masking during testing.

model.eval()

print("✓ MHNet loaded")


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")

original_width, original_height = image.size

print(
    "Original image size:",
    (original_width, original_height)
)


# ============================================================
# PREPARE IMAGE
# ============================================================

input_image = image.resize(
    (IMAGE_SIZE, IMAGE_SIZE),
    Image.Resampling.BILINEAR
)

image_array = np.array(
    input_image
).astype(
    np.float32
) / 255.0

image_tensor = torch.from_numpy(
    image_array
)

# HWC -> CHW

image_tensor = image_tensor.permute(
    2,
    0,
    1
)

# Add batch dimension

image_tensor = image_tensor.unsqueeze(
    0
)

image_tensor = image_tensor.to(
    device
)


# ============================================================
# PREDICTION
# ============================================================

print("Running MHNet prediction...")

with torch.no_grad():

    logits = model(
        image_tensor
    )

    probabilities = torch.sigmoid(
        logits
    )

    prediction = (
        probabilities >= 0.5
    ).float()


# ============================================================
# RESIZE PREDICTION TO ORIGINAL IMAGE SIZE
# ============================================================

prediction = F.interpolate(
    prediction,
    size=(
        original_height,
        original_width
    ),
    mode="nearest"
)

prediction = prediction[
    0,
    0
].cpu().numpy()


# ============================================================
# LOAD ACTUAL FLOOD MASK
# ============================================================

actual_mask = Image.open(
    MASK_PATH
).convert("L")

actual_mask = np.array(
    actual_mask
)

actual_mask = (
    actual_mask > 127
).astype(
    np.uint8
)


# ============================================================
# PREDICTED BINARY MASK
# ============================================================

predicted_mask = (
    prediction > 0.5
).astype(
    np.uint8
)


# ============================================================
# FLOOD AREA
# ============================================================

actual_flood_pixels = (
    actual_mask == 1
).sum()

predicted_flood_pixels = (
    predicted_mask == 1
).sum()

total_pixels = actual_mask.size

actual_percentage = (
    actual_flood_pixels
    /
    total_pixels
    *
    100
)

predicted_percentage = (
    predicted_flood_pixels
    /
    total_pixels
    *
    100
)


# ============================================================
# CONFUSION VALUES
# ============================================================

tp = np.logical_and(
    predicted_mask == 1,
    actual_mask == 1
).sum()

fp = np.logical_and(
    predicted_mask == 1,
    actual_mask == 0
).sum()

fn = np.logical_and(
    predicted_mask == 0,
    actual_mask == 1
).sum()


# ============================================================
# IoU
# ============================================================

union = tp + fp + fn

if union > 0:

    iou = tp / union

else:

    iou = 1.0


# ============================================================
# DICE
# ============================================================

dice_denominator = (
    2 * tp
    +
    fp
    +
    fn
)

if dice_denominator > 0:

    dice = (
        2 * tp
        /
        dice_denominator
    )

else:

    dice = 1.0


# ============================================================
# PRECISION
# ============================================================

if tp + fp > 0:

    precision = (
        tp
        /
        (tp + fp)
    )

else:

    precision = 0.0


# ============================================================
# RECALL
# ============================================================

if tp + fn > 0:

    recall = (
        tp
        /
        (tp + fn)
    )

else:

    recall = 0.0


# ============================================================
# SAVE PREDICTION MASK
# ============================================================

prediction_image = (
    predicted_mask * 255
).astype(
    np.uint8
)

Image.fromarray(
    prediction_image
).save(
    OUTPUT_PATH
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)

print("MHNet TEST RESULT - 387.jpg")

print("=" * 60)

print(
    f"Actual flood area    : "
    f"{actual_percentage:.2f}%"
)

print(
    f"Predicted flood area : "
    f"{predicted_percentage:.2f}%"
)

print()

print(
    f"IoU       : "
    f"{iou * 100:.2f}%"
)

print(
    f"Dice      : "
    f"{dice * 100:.2f}%"
)

print(
    f"Precision : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall    : "
    f"{recall * 100:.2f}%"
)

print()

print(
    f"TP: {tp}"
)

print(
    f"FP: {fp}"
)

print(
    f"FN: {fn}"
)

print()

print(
    "Prediction saved to:"
)

print(
    OUTPUT_PATH
)

print("=" * 60)