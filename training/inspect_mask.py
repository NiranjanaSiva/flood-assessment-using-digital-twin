from PIL import Image
import numpy as np

image_path = "AIFloodSense/images/387.jpg"
mask_path = "AIFloodSense/masks/387.png"

image = Image.open(image_path).convert("RGB")
mask = np.array(Image.open(mask_path))

print("Image size:", image.size)
print("Mask size:", mask.shape)
print("Mask values:", np.unique(mask))

# Create separate images for each mask class
for value in [0, 170, 255]:
    binary = (mask == value).astype(np.uint8) * 255

    output = Image.fromarray(binary)
    output.save(f"training/mask_{value}.png")

    print(f"Saved training/mask_{value}.png")