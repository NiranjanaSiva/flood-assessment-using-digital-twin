import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from mhnet import MHNet

IMAGE_DIR = "AIFloodSense/images"
MASK_DIR = "AIFloodSense/flood_masks"
MODEL_PATH = "mhnet_flood_model/mhnet_best.pth"

IMAGE_SIZE = 256
BASE_CHANNELS = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

model = MHNet(
    in_channels=3,
    base_channels=BASE_CHANNELS
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model.eval()

print("✓ MHNet loaded")

tp = 0
tn = 0
fp = 0
fn = 0

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print("Images found:", len(image_files))

for image_name in image_files:

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    mask_name = os.path.splitext(image_name)[0] + ".png"

    mask_path = os.path.join(
        MASK_DIR,
        mask_name
    )

    if not os.path.exists(mask_path):
        print("Skipping:", image_name, "- mask not found")
        continue

    image = cv2.imread(image_path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original_h, original_w = image.shape[:2]

    image_resized = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image_tensor = (
        torch.from_numpy(
            image_resized
        )
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image_tensor)

        prediction = torch.sigmoid(output)

        prediction = (
            prediction > 0.5
        ).float()

    predicted_mask = prediction[
        0, 0
    ].cpu().numpy().astype(np.uint8)

    actual_mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    actual_mask = cv2.resize(
        actual_mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    actual_mask = (
        actual_mask > 127
    ).astype(np.uint8)

    predicted_mask = (
        predicted_mask > 0
    ).astype(np.uint8)

    tp += np.logical_and(
        predicted_mask == 1,
        actual_mask == 1
    ).sum()

    tn += np.logical_and(
        predicted_mask == 0,
        actual_mask == 0
    ).sum()

    fp += np.logical_and(
        predicted_mask == 1,
        actual_mask == 0
    ).sum()

    fn += np.logical_and(
        predicted_mask == 0,
        actual_mask == 1
    ).sum()

print()
print("=" * 60)
print("MHNet TEST SET CONFUSION MATRIX")
print("=" * 60)

print("TP :", tp)
print("TN :", tn)
print("FP :", fp)
print("FN :", fn)

accuracy = (
    (tp + tn)
    /
    (tp + tn + fp + fn)
) if (tp + tn + fp + fn) > 0 else 0

precision = (
    tp
    /
    (tp + fp)
) if (tp + fp) > 0 else 0

recall = (
    tp
    /
    (tp + fn)
) if (tp + fn) > 0 else 0

f1 = (
    2 * precision * recall
    /
    (precision + recall)
) if (precision + recall) > 0 else 0

iou = (
    tp
    /
    (tp + fp + fn)
) if (tp + fp + fn) > 0 else 0

dice = (
    2 * tp
    /
    (2 * tp + fp + fn)
) if (2 * tp + fp + fn) > 0 else 0

print()
print("Accuracy :", round(accuracy * 100, 2), "%")
print("IoU      :", round(iou * 100, 2), "%")
print("Dice     :", round(dice * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print()
print("                 Predicted")
print("              Background    Flood")
print("Actual")
print("Background   ", tn, "        ", fp)
print("Flood        ", fn, "        ", tp)

confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])

plt.figure(figsize=(7, 6))

plt.imshow(
    confusion_matrix,
    interpolation="nearest"
)

plt.title("MHNet - Test Set Confusion Matrix")

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    [0, 1],
    ["Background", "Flood"]
)

plt.yticks(
    [0, 1],
    ["Background", "Flood"]
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            str(confusion_matrix[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar(
    label="Pixel Count"
)

plt.tight_layout()

plt.show()
