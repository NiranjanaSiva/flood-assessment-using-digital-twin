
import os
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mhnet import MHNet


# ============================================================
# SETTINGS
# ============================================================

IMAGE_DIR = "AIFloodSense/images"
MASK_DIR = "AIFloodSense/flood_masks"

MODEL_DIR = "mhnet_flood_model"

IMAGE_SIZE = 256

BATCH_SIZE = 1

EPOCHS = 50

LEARNING_RATE = 0.001

MOMENTUM = 0.9

WEIGHT_DECAY = 0.0001

MIN_LR = 0.00001

MASK_RATIO = 0.25

BASE_CHANNELS = 32

TRAIN_SIZE = 300

SEED = 42


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
# RANDOM SEED
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# DATASET
# ============================================================

class FloodDataset(Dataset):

    def __init__(
        self,
        image_files,
        image_dir,
        mask_dir
    ):

        self.image_files = image_files

        self.image_dir = image_dir

        self.mask_dir = mask_dir

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, index):

        image_name = self.image_files[index]

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Corresponding flood mask
        # ----------------------------------------------------

        mask_name = (
            os.path.splitext(image_name)[0]
            + ".png"
        )

        mask_path = os.path.join(
            self.mask_dir,
            mask_name
        )

        mask = Image.open(
            mask_path
        ).convert("L")

        # ----------------------------------------------------
        # Resize image
        # ----------------------------------------------------

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        # ----------------------------------------------------
        # Resize mask
        #
        # IMPORTANT:
        # nearest-neighbor is used for segmentation masks.
        # ----------------------------------------------------

        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        # ----------------------------------------------------
        # Convert image to tensor
        # ----------------------------------------------------

        image = np.array(
            image
        ).astype(
            np.float32
        ) / 255.0

        image = torch.from_numpy(
            image
        )

        image = image.permute(
            2,
            0,
            1
        )

        # ----------------------------------------------------
        # Flood mask
        #
        # flood_masks contain:
        #
        # 255 = flood
        # 0   = background
        # ----------------------------------------------------

        mask = np.array(
            mask
        ).astype(
            np.float32
        )

        mask = (
            mask > 127
        ).astype(
            np.float32
        )

        mask = torch.from_numpy(
            mask
        )

        mask = mask.unsqueeze(0)

        return image, mask


# ============================================================
# FIND IMAGE/MASK PAIRS
# ============================================================

image_files = []

for filename in os.listdir(IMAGE_DIR):

    if filename.lower().endswith(".jpg"):

        image_name = os.path.splitext(
            filename
        )[0]

        mask_name = image_name + ".png"

        mask_path = os.path.join(
            MASK_DIR,
            mask_name
        )

        if os.path.exists(mask_path):

            image_files.append(
                filename
            )


image_files.sort()

print(
    "Total matching pairs:",
    len(image_files)
)


# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(
    image_files
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_files = image_files[
    :TRAIN_SIZE
]

val_files = image_files[
    TRAIN_SIZE:
]


print(
    "Training images:",
    len(train_files)
)

print(
    "Validation images:",
    len(val_files)
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = FloodDataset(
    train_files,
    IMAGE_DIR,
    MASK_DIR
)

val_dataset = FloodDataset(
    val_files,
    IMAGE_DIR,
    MASK_DIR
)


# ============================================================
# DATALOADERS
# ============================================================

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


# ============================================================
# MODEL
# ============================================================

model = MHNet(
    in_channels=3,
    base_channels=BASE_CHANNELS,
    mask_ratio=MASK_RATIO
)

model = model.to(device)


# ============================================================
# BCE LOSS
#
# Paper Eq. (6)
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()


# ============================================================
# DICE LOSS
#
# Paper Eq. (7)
# ============================================================

def dice_loss(
    logits,
    targets,
    epsilon=1e-6
):

    probabilities = torch.sigmoid(
        logits
    )

    batch_size = probabilities.shape[0]

    probabilities = probabilities.reshape(
        batch_size,
        -1
    )

    targets = targets.reshape(
        batch_size,
        -1
    )

    intersection = (
        probabilities * targets
    ).sum(dim=1)

    denominator = (
        probabilities.sum(dim=1)
        +
        targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection
        +
        epsilon
    ) / (
        denominator
        +
        epsilon
    )

    return 1.0 - dice.mean()


# ============================================================
# TOTAL LOSS
#
# Paper Eq. (8)
#
# Ltotal = alpha * LBCE + beta * LDice
#
# alpha = 0.5
# beta  = 1.0
# ============================================================

def total_loss(
    logits,
    targets
):

    loss_bce = bce_loss(
        logits,
        targets
    )

    loss_dice = dice_loss(
        logits,
        targets
    )

    loss = (
        0.5 * loss_bce
        +
        1.0 * loss_dice
    )

    return loss


# ============================================================
# OPTIMIZER
#
# Paper:
# SGD
# LR = 0.001
# momentum = 0.9
# weight decay = 0.0001
# ============================================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# COSINE ANNEALING
#
# Paper:
# minimum LR = 0.00001
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=MIN_LR
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits,
    targets
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= 0.5
    ).float()

    targets = targets.float()

    tp = (
        predictions * targets
    ).sum().item()

    fp = (
        predictions * (1 - targets)
    ).sum().item()

    fn = (
        (1 - predictions) * targets
    ).sum().item()

    intersection = tp

    union = tp + fp + fn

    iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return (
        iou,
        precision,
        recall,
        f1
    )


# ============================================================
# TRAINING
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


best_iou = 0.0


for epoch in range(EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss = 0.0

    for batch_index, (
        images,
        masks
    ) in enumerate(train_loader):

        images = images.to(device)

        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass

        outputs = model(
            images
        )

        # Hybrid BCE + Dice

        loss = total_loss(
            outputs,
            masks
        )

        # Backpropagation

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        if (
            batch_index + 1
        ) % 20 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_index + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )


    train_loss /= len(
        train_loader
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    validation_loss = 0.0

    total_iou = 0.0

    total_precision = 0.0

    total_recall = 0.0

    total_f1 = 0.0


    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)

            masks = masks.to(device)

            outputs = model(
                images
            )

            loss = total_loss(
                outputs,
                masks
            )

            validation_loss += loss.item()

            (
                iou,
                precision,
                recall,
                f1
            ) = calculate_metrics(
                outputs,
                masks
            )

            total_iou += iou

            total_precision += precision

            total_recall += recall

            total_f1 += f1


    validation_loss /= len(
        val_loader
    )

    mean_iou = (
        total_iou
        /
        len(val_loader)
    )

    mean_precision = (
        total_precision
        /
        len(val_loader)
    )

    mean_recall = (
        total_recall
        /
        len(val_loader)
    )

    mean_f1 = (
        total_f1
        /
        len(val_loader)
    )


    # ========================================================
    # LEARNING RATE
    # ========================================================

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Training Loss   : {train_loss:.4f}"
    )

    print(
        f"Validation Loss : {validation_loss:.4f}"
    )

    print(
        f"mIoU            : {mean_iou * 100:.2f}%"
    )

    print(
        f"Precision       : {mean_precision * 100:.2f}%"
    )

    print(
        f"Recall          : {mean_recall * 100:.2f}%"
    )

    print(
        f"F1 Score        : {mean_f1 * 100:.2f}%"
    )

    print(
        f"Learning Rate   : {current_lr:.8f}"
    )

    print("=" * 60)
    print()


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if mean_iou > best_iou:

        best_iou = mean_iou

        save_path = os.path.join(
            MODEL_DIR,
            "mhnet_best.pth"
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "best_iou":
                    best_iou,

                "epoch":
                    epoch + 1,

                "base_channels":
                    BASE_CHANNELS,

                "mask_ratio":
                    MASK_RATIO
            },
            save_path
        )

        print(
            "✓ New best model saved:"
        )

        print(
            save_path
        )


print()
print("Training finished.")
print(
    "Best Validation mIoU:",
    f"{best_iou * 100:.2f}%"
)
