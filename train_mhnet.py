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

# Keep this small because your dataset is relatively small
BATCH_SIZE = 2

# Maximum number of epochs
EPOCHS = 50

# Reduced learning rate
LEARNING_RATE = 0.0001

# AdamW regularization
WEIGHT_DECAY = 0.0005

# Minimum learning rate
MIN_LR = 0.000001

# MHNet masking
MASK_RATIO = 0.25

BASE_CHANNELS = 32

# Number of images used for training
TRAIN_SIZE = 300

# Reproducibility
SEED = 42

# Early stopping
PATIENCE = 8


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

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class FloodDataset(Dataset):

    def __init__(
        self,
        image_files,
        image_dir,
        mask_dir,
        augment=False
    ):

        self.image_files = image_files

        self.image_dir = image_dir

        self.mask_dir = mask_dir

        self.augment = augment


    def __len__(self):

        return len(self.image_files)


    def __getitem__(self, index):

        image_name = self.image_files[index]


        # ----------------------------------------------------
        # IMAGE PATH
        # ----------------------------------------------------

        image_path = os.path.join(
            self.image_dir,
            image_name
        )


        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")


        # ----------------------------------------------------
        # CORRESPONDING MASK
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
        # RESIZE IMAGE
        # ----------------------------------------------------

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )


        # ----------------------------------------------------
        # RESIZE MASK
        #
        # IMPORTANT:
        # NEVER use bilinear interpolation for masks.
        # ----------------------------------------------------

        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )


        # ====================================================
        # DATA AUGMENTATION
        # ====================================================
        #
        # VERY IMPORTANT:
        # The SAME transformation is applied to both
        # image and flood mask.
        #
        # Otherwise the image and mask would no longer align.
        # ====================================================

        if self.augment:

            # ------------------------------------------------
            # RANDOM HORIZONTAL FLIP
            # ------------------------------------------------

            if random.random() < 0.5:

                image = image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

                mask = mask.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )


            # ------------------------------------------------
            # RANDOM VERTICAL FLIP
            # ------------------------------------------------

            if random.random() < 0.5:

                image = image.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )

                mask = mask.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )


            # ------------------------------------------------
            # RANDOM 90 DEGREE ROTATION
            # ------------------------------------------------

            rotation = random.randint(
                0,
                3
            )

            if rotation == 1:

                image = image.transpose(
                    Image.Transpose.ROTATE_90
                )

                mask = mask.transpose(
                    Image.Transpose.ROTATE_90
                )

            elif rotation == 2:

                image = image.transpose(
                    Image.Transpose.ROTATE_180
                )

                mask = mask.transpose(
                    Image.Transpose.ROTATE_180
                )

            elif rotation == 3:

                image = image.transpose(
                    Image.Transpose.ROTATE_270
                )

                mask = mask.transpose(
                    Image.Transpose.ROTATE_270
                )


        # ====================================================
        # IMAGE → TENSOR
        # ====================================================

        image = np.array(
            image
        ).astype(
            np.float32
        ) / 255.0


        image = torch.from_numpy(
            image
        )


        # H,W,C → C,H,W

        image = image.permute(
            2,
            0,
            1
        )


        # ====================================================
        # MASK → TENSOR
        # ====================================================

        mask = np.array(
            mask
        ).astype(
            np.float32
        )


        # 255 → 1
        # 0   → 0

        mask = (
            mask > 127
        ).astype(
            np.float32
        )


        mask = torch.from_numpy(
            mask
        )


        # Add channel dimension
        #
        # H,W → 1,H,W

        mask = mask.unsqueeze(0)


        return image, mask


# ============================================================
# FIND IMAGE / MASK PAIRS
# ============================================================

image_files = []


for filename in os.listdir(
    IMAGE_DIR
):

    if filename.lower().endswith(
        ".jpg"
    ):

        image_name = os.path.splitext(
            filename
        )[0]


        mask_name = (
            image_name
            + ".png"
        )


        mask_path = os.path.join(
            MASK_DIR,
            mask_name
        )


        if os.path.exists(
            mask_path
        ):

            image_files.append(
                filename
            )


image_files.sort()


print(
    "Total matching pairs:",
    len(image_files)
)


# ============================================================
# CHECK DATASET SIZE
# ============================================================

if len(image_files) < 2:

    raise RuntimeError(
        "Not enough image/mask pairs found."
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

train_count = min(
    TRAIN_SIZE,
    len(image_files) - 1
)


train_files = image_files[
    :train_count
]


val_files = image_files[
    train_count:
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

# AUGMENTATION ONLY FOR TRAINING

train_dataset = FloodDataset(
    train_files,
    IMAGE_DIR,
    MASK_DIR,
    augment=True
)


# NO AUGMENTATION FOR VALIDATION

val_dataset = FloodDataset(
    val_files,
    IMAGE_DIR,
    MASK_DIR,
    augment=False
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


model = model.to(
    device
)


print()
print("MHNet created.")
print(
    "Base channels:",
    BASE_CHANNELS
)
print(
    "Mask ratio:",
    MASK_RATIO
)


# ============================================================
# BCE LOSS
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()


# ============================================================
# DICE LOSS
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
        probabilities
        * targets
    ).sum(
        dim=1
    )


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


    # BCE + Dice

    loss = (
        0.5 * loss_bce
        +
        1.0 * loss_dice
    )


    return loss


# ============================================================
# OPTIMIZER
# ============================================================
#
# CHANGED:
#
# OLD:
# SGD
# LR = 0.001
#
# NEW:
# AdamW
# LR = 0.0001
# Weight decay = 0.0005
#
# This gives stronger regularization and a smaller update
# step, which is useful when the training set is small.
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================
#
# If validation loss stops improving, reduce learning rate.
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=MIN_LR
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
        predictions
        * targets
    ).sum().item()


    fp = (
        predictions
        * (1 - targets)
    ).sum().item()


    fn = (
        (1 - predictions)
        * targets
    ).sum().item()


    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    intersection = tp

    union = (
        tp
        + fp
        + fn
    )


    if union > 0:

        iou = (
            intersection
            / union
        )

    else:

        iou = 1.0


    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    if (
        tp + fp
        > 0
    ):

        precision = (
            tp
            /
            (tp + fp)
        )

    else:

        precision = 0.0


    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    if (
        tp + fn
        > 0
    ):

        recall = (
            tp
            /
            (tp + fn)
        )

    else:

        recall = 0.0


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if (
        precision + recall
        > 0
    ):

        f1 = (
            2
            * precision
            * recall
            /
            (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0


    return (
        iou,
        precision,
        recall,
        f1
    )


# ============================================================
# MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# BEST MODEL
# ============================================================

best_iou = 0.0


# ============================================================
# EARLY STOPPING
# ============================================================

epochs_without_improvement = 0


# ============================================================
# TRAINING
# ============================================================

for epoch in range(
    EPOCHS
):


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()


    train_loss = 0.0


    for batch_index, (
        images,
        masks
    ) in enumerate(
        train_loader
    ):


        images = images.to(
            device
        )


        masks = masks.to(
            device
        )


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        outputs = model(
            images
        )


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = total_loss(
            outputs,
            masks
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Gradient clipping
        #
        # Helps prevent unusually large updates.
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        # ----------------------------------------------------
        # Update model
        # ----------------------------------------------------

        optimizer.step()


        train_loss += (
            loss.item()
        )


        if (
            batch_index + 1
        ) % 20 == 0:

            print(
                f"Epoch "
                f"[{epoch + 1}/{EPOCHS}] "
                f"Batch "
                f"[{batch_index + 1}/"
                f"{len(train_loader)}] "
                f"Loss: "
                f"{loss.item():.4f}"
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


            images = images.to(
                device
            )


            masks = masks.to(
                device
            )


            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            outputs = model(
                images
            )


            # ------------------------------------------------
            # Validation loss
            # ------------------------------------------------

            loss = total_loss(
                outputs,
                masks
            )


            validation_loss += (
                loss.item()
            )


            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

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


    # ========================================================
    # AVERAGES
    # ========================================================

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

    scheduler.step(
        validation_loss
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        f"Epoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    print(
        f"Training Loss   : "
        f"{train_loss:.4f}"
    )

    print(
        f"Validation Loss : "
        f"{validation_loss:.4f}"
    )

    print(
        f"mIoU            : "
        f"{mean_iou * 100:.2f}%"
    )

    print(
        f"Precision       : "
        f"{mean_precision * 100:.2f}%"
    )

    print(
        f"Recall          : "
        f"{mean_recall * 100:.2f}%"
    )

    print(
        f"F1 Score        : "
        f"{mean_f1 * 100:.2f}%"
    )

    print(
        f"Learning Rate   : "
        f"{current_lr:.8f}"
    )

    print(
        "=" * 60
    )

    print()


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if mean_iou > best_iou:

        best_iou = mean_iou

        epochs_without_improvement = 0


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

        print(
            f"✓ Best mIoU: "
            f"{best_iou * 100:.2f}%"
        )


    else:

        epochs_without_improvement += 1


        print(
            f"No mIoU improvement "
            f"for "
            f"{epochs_without_improvement} "
            f"epoch(s)."
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()

        print(
            "Early stopping triggered."
        )

        print(
            f"Validation mIoU did not "
            f"improve for "
            f"{PATIENCE} epochs."
        )

        break


# ============================================================
# FINISHED
# ============================================================

print()

print(
    "=" * 60
)

print(
    "Training finished."
)

print(
    "Best Validation mIoU:",
    f"{best_iou * 100:.2f}%"
)

print(
    "=" * 60
)
