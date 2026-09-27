import io
import base64
from collections import deque

import numpy as np
import requests

from PIL import Image

import torch
import torch.nn.functional as F

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from mhnet import MHNet


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="MHNet Flood Assessment API"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "mhnet_flood_model/mhnet_best.pth"

BASE_CHANNELS = 32

MASK_RATIO = 0.25

IMAGE_SIZE = 256


# ============================================================
# ELEVATION SETTINGS
# ============================================================

DEM_ROWS = 30

DEM_COLS = 30

IMAGE_WIDTH_DEGREES = 0.020

IMAGE_HEIGHT_DEGREES = 0.015

MAX_FLOW_STEPS = 12

MAX_ELEVATION_RISE = 0.5

OPEN_ELEVATION_URL = (
    "https://api.open-elevation.com/api/v1/lookup"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# ============================================================
# LOAD MHNET
# ============================================================

print(
    "Loading MHNet model..."
)

model = MHNet(
    in_channels=3,
    base_channels=BASE_CHANNELS,
    mask_ratio=MASK_RATIO
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(
    device
)

# IMPORTANT:
# Evaluation mode disables the training-only
# masking mechanism.

model.eval()

print(
    "✓ MHNet loaded successfully"
)


# ============================================================
# ROOT STATUS
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "model": "MHNet",
        "device": str(device),
        "model_path": MODEL_PATH
    }


# ============================================================
# IMAGE -> TENSOR
# ============================================================

def prepare_image(image):

    original_width, original_height = image.size

    resized = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image_array = np.array(
        resized
    ).astype(
        np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        image_array
    )

    # HWC -> CHW

    tensor = tensor.permute(
        2,
        0,
        1
    )

    # Add batch dimension

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        device
    )

    return (
        tensor,
        original_width,
        original_height
    )


# ============================================================
# MHNET FLOOD SEGMENTATION
# ============================================================

def predict_flood_mask(
    image
):

    (
        tensor,
        original_width,
        original_height
    ) = prepare_image(
        image
    )

    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = torch.sigmoid(
            logits
        )

    # --------------------------------------------------------
    # Resize prediction to original image size
    # --------------------------------------------------------

    probabilities = F.interpolate(
        probabilities,
        size=(
            original_height,
            original_width
        ),
        mode="bilinear",
        align_corners=False
    )

    probabilities = probabilities[
        0,
        0
    ]

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    mask = (
        probabilities >= 0.5
    ).cpu().numpy().astype(
        np.uint8
    )

    return mask


# ============================================================
# CREATE RED FLOOD OVERLAY
# ============================================================

def create_red_overlay(
    flood_mask
):

    height, width = flood_mask.shape

    overlay = np.zeros(
        (
            height,
            width,
            4
        ),
        dtype=np.uint8
    )

    # Red

    overlay[:, :, 0] = 255

    # Transparent where there is no flood

    overlay[:, :, 3] = (
        flood_mask * 120
    )

    image = Image.fromarray(
        overlay,
        mode="RGBA"
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )

    return (
        "data:image/png;base64,"
        + encoded
    )


# ============================================================
# CREATE ORANGE PREDICTED OVERLAY
# ============================================================

def create_predicted_overlay(
    prediction_mask,
    current_mask
):

    # Only show NEW predicted areas in orange.

    new_prediction = (
        (prediction_mask == 1)
        &
        (current_mask == 0)
    ).astype(
        np.uint8
    )

    height, width = new_prediction.shape

    overlay = np.zeros(
        (
            height,
            width,
            4
        ),
        dtype=np.uint8
    )

    # Orange = red + green

    overlay[:, :, 0] = 255

    overlay[:, :, 1] = 140

    overlay[:, :, 3] = (
        new_prediction * 120
    )

    image = Image.fromarray(
        overlay,
        mode="RGBA"
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )

    return (
        "data:image/png;base64,"
        + encoded
    )


# ============================================================
# CREATE FLOOD GRID
# ============================================================

def create_flood_grid(
    flood_mask
):

    height, width = flood_mask.shape

    grid = np.zeros(
        (
            DEM_ROWS,
            DEM_COLS
        ),
        dtype=np.uint8
    )

    for row in range(DEM_ROWS):

        y1 = int(
            row * height / DEM_ROWS
        )

        y2 = int(
            (row + 1) * height / DEM_ROWS
        )

        for col in range(DEM_COLS):

            x1 = int(
                col * width / DEM_COLS
            )

            x2 = int(
                (col + 1) * width / DEM_COLS
            )

            cell = flood_mask[
                y1:y2,
                x1:x2
            ]

            if cell.size == 0:
                continue

            flood_ratio = (
                np.mean(cell)
            )

            # Cell is considered flooded if
            # at least 10% of its pixels are flood.

            if flood_ratio >= 0.10:

                grid[
                    row,
                    col
                ] = 1

    return grid


# ============================================================
# CREATE LAT/LON GRID
# ============================================================

def create_coordinate_grid(
    latitude,
    longitude
):

    lat_step = (
        IMAGE_HEIGHT_DEGREES
        /
        DEM_ROWS
    )

    lon_step = (
        IMAGE_WIDTH_DEGREES
        /
        DEM_COLS
    )

    points = []

    for row in range(DEM_ROWS):

        for col in range(DEM_COLS):

            lat = (
                latitude
                -
                IMAGE_HEIGHT_DEGREES / 2
                +
                (row + 0.5) * lat_step
            )

            lon = (
                longitude
                -
                IMAGE_WIDTH_DEGREES / 2
                +
                (col + 0.5) * lon_step
            )

            points.append(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "row": row,
                    "col": col
                }
            )

    return points


# ============================================================
# GET ELEVATIONS
# ============================================================

def get_elevations(
    latitude,
    longitude
):

    points = create_coordinate_grid(
        latitude,
        longitude
    )

    elevation_grid = np.full(
        (
            DEM_ROWS,
            DEM_COLS
        ),
        np.nan,
        dtype=np.float32
    )

    # Open-Elevation accepts batches.
    # Use 100 points at a time.

    batch_size = 100

    for start in range(
        0,
        len(points),
        batch_size
    ):

        batch = points[
            start:start + batch_size
        ]

        locations = []

        for point in batch:

            locations.append(
                {
                    "latitude":
                        point["latitude"],

                    "longitude":
                        point["longitude"]
                }
            )

        try:

            response = requests.post(
                OPEN_ELEVATION_URL,
                json={
                    "locations": locations
                },
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            results = data.get(
                "results",
                []
            )

            for point, result in zip(
                batch,
                results
            ):

                elevation = result.get(
                    "elevation"
                )

                if elevation is not None:

                    elevation_grid[
                        point["row"],
                        point["col"]
                    ] = float(
                        elevation
                    )

        except Exception as error:

            print(
                "Elevation request failed:",
                error
            )

    return elevation_grid


# ============================================================
# FLOOD PROPAGATION
# ============================================================

def predict_flood_spread(
    current_grid,
    elevation_grid
):

    predicted_grid = (
        current_grid.copy()
    )

    queue = deque()

    # --------------------------------------------------------
    # Start from ALL currently flooded cells.
    # --------------------------------------------------------

    for row in range(DEM_ROWS):

        for col in range(DEM_COLS):

            if current_grid[
                row,
                col
            ] == 1:

                queue.append(
                    (
                        row,
                        col,
                        0
                    )
                )


    # 8 neighboring directions

    directions = [

        (-1, -1),
        (-1, 0),
        (-1, 1),

        (0, -1),
        (0, 1),

        (1, -1),
        (1, 0),
        (1, 1)

    ]


    while queue:

        (
            row,
            col,
            steps
        ) = queue.popleft()

        # Maximum propagation distance

        if steps >= MAX_FLOW_STEPS:

            continue

        source_elevation = elevation_grid[
            row,
            col
        ]

        # If source elevation is unavailable,
        # don't propagate from it.

        if np.isnan(
            source_elevation
        ):

            continue


        for dr, dc in directions:

            nr = row + dr

            nc = col + dc

            # Boundary check

            if nr < 0:
                continue

            if nr >= DEM_ROWS:
                continue

            if nc < 0:
                continue

            if nc >= DEM_COLS:
                continue

            # Already predicted/flooded

            if predicted_grid[
                nr,
                nc
            ] == 1:

                continue

            neighbor_elevation = elevation_grid[
                nr,
                nc
            ]

            if np.isnan(
                neighbor_elevation
            ):

                continue

            # ------------------------------------------------
            # Elevation difference
            # ------------------------------------------------

            elevation_difference = (
                neighbor_elevation
                -
                source_elevation
            )

            # ------------------------------------------------
            # Water can move to:
            #
            # 1. lower ground
            # 2. almost level ground
            #
            # within the allowed rise.
            # ------------------------------------------------

            if (
                elevation_difference
                <=
                MAX_ELEVATION_RISE
            ):

                predicted_grid[
                    nr,
                    nc
                ] = 1

                queue.append(
                    (
                        nr,
                        nc,
                        steps + 1
                    )
                )


    return predicted_grid


# ============================================================
# SEGMENT ENDPOINT
# ============================================================

@app.post("/segment")
async def segment(

    file: UploadFile = File(...),

    latitude: float = Form(...),

    longitude: float = Form(...)

):

    # ========================================================
    # READ IMAGE
    # ========================================================

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert(
        "RGB"
    )

    width, height = image.size


    # ========================================================
    # MHNET SEGMENTATION
    # ========================================================

    print(
        f"Processing {file.filename}"
    )

    print(
        "Running MHNet..."
    )

    flood_mask = predict_flood_mask(
        image
    )


    # ========================================================
    # CURRENT FLOOD STATISTICS
    # ========================================================

    flood_pixels = int(
        flood_mask.sum()
    )

    total_pixels = (
        width * height
    )

    flood_percentage = (
        flood_pixels
        /
        total_pixels
        *
        100
    )


    # ========================================================
    # CURRENT FLOOD OVERLAY
    # ========================================================

    overlay = create_red_overlay(
        flood_mask
    )


    # ========================================================
    # CREATE FLOOD GRID
    # ========================================================

    current_grid = create_flood_grid(
        flood_mask
    )


    current_cells = int(
        current_grid.sum()
    )


    # ========================================================
    # GET ELEVATION
    # ========================================================

    print(
        "Getting elevation data..."
    )

    elevation_grid = get_elevations(
        latitude,
        longitude
    )


    valid_elevations = (
        elevation_grid[
            ~np.isnan(
                elevation_grid
            )
        ]
    )


    # ========================================================
    # PROPAGATE FLOOD
    # ========================================================

    predicted_grid = predict_flood_spread(
        current_grid,
        elevation_grid
    )


    predicted_cells = int(
        predicted_grid.sum()
    )


    new_predicted_cells = int(
        (
            predicted_grid
            &
            (current_grid == 0)
        ).sum()
    )


    # ========================================================
    # EXPAND GRID TO IMAGE SIZE
    # ========================================================

    grid_tensor = torch.from_numpy(
        predicted_grid.astype(
            np.float32
        )
    )

    grid_tensor = grid_tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    grid_tensor = F.interpolate(
        grid_tensor,
        size=(
            height,
            width
        ),
        mode="nearest"
    )

    prediction_mask = (
        grid_tensor[
            0,
            0
        ].numpy() > 0.5
    ).astype(
        np.uint8
    )


    # ========================================================
    # PREDICTED OVERLAY
    # ========================================================

    predicted_overlay = (
        create_predicted_overlay(
            prediction_mask,
            flood_mask
        )
    )


    # ========================================================
    # ELEVATION STATISTICS
    # ========================================================

    current_elevations = []

    for row in range(DEM_ROWS):

        for col in range(DEM_COLS):

            if current_grid[
                row,
                col
            ] == 1:

                value = elevation_grid[
                    row,
                    col
                ]

                if not np.isnan(value):

                    current_elevations.append(
                        value
                    )


    if current_elevations:

        mean_flood_elevation = float(
            np.mean(
                current_elevations
            )
        )

    else:

        mean_flood_elevation = None


    if len(valid_elevations) > 0:

        min_elevation = float(
            np.min(
                valid_elevations
            )
        )

        max_elevation = float(
            np.max(
                valid_elevations
            )
        )

        average_elevation = float(
            np.mean(
                valid_elevations
            )
        )

    else:

        min_elevation = None

        max_elevation = None

        average_elevation = None


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "success": True,

        "filename": file.filename,

        "width": width,

        "height": height,

        "center": {

            "latitude": latitude,

            "longitude": longitude

        },

        # ----------------------------------------------
        # MHNet segmentation
        # ----------------------------------------------

        "flood_pixels":
            flood_pixels,

        "flood_percentage":
            flood_percentage,

        "overlay":
            overlay,

        # ----------------------------------------------
        # Elevation prediction
        # ----------------------------------------------

        "predicted_overlay":
            predicted_overlay,

        "prediction_grid":
            predicted_grid.tolist(),

        "current_flood_grid":
            current_grid.tolist(),

        # ----------------------------------------------
        # Elevation information
        # ----------------------------------------------

        "elevation": {

            "available":
                len(valid_elevations) > 0,

            "current_cells":
                current_cells,

            "predicted_cells":
                predicted_cells,

            "new_predicted_cells":
                new_predicted_cells,

            "mean_flood_elevation":
                mean_flood_elevation,

            "min_elevation":
                min_elevation,

            "max_elevation":
                max_elevation,

            "average_elevation":
                average_elevation,

            "elevation_grid":
                np.nan_to_num(
                    elevation_grid,
                    nan=-9999
                ).tolist()

        }

    }

    