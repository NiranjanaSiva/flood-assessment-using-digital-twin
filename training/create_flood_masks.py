from pathlib import Path
from PIL import Image
import numpy as np

MASK_DIR = Path("AIFloodSense/masks")
OUTPUT_DIR = Path("AIFloodSense/flood_masks")

# AIFloodSense flood class
FLOOD_VALUE = 255

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mask_files = sorted(MASK_DIR.glob("*.png"))

print(f"Found {len(mask_files)} masks")

for mask_path in mask_files:

    mask = np.array(Image.open(mask_path))

    # Flood = 1
    # Everything else = 0
    flood_mask = (mask == FLOOD_VALUE).astype(np.uint8)

    # Save as 0/255 PNG for easy visualization
    output_mask = flood_mask * 255

    output_path = OUTPUT_DIR / mask_path.name

    Image.fromarray(output_mask).save(output_path)

print("Done!")
print(f"Flood masks saved to: {OUTPUT_DIR}")