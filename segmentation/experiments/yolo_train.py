# %%
import torch

torch.cuda.empty_cache()

# %%
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s-seg.pt")  # load a pretrained model (recommended for training)

# %%
DATASET_DIR = "../datasets/DOOM_yolo_seg"

# %%
model.train(
    data=f"{DATASET_DIR}/vizdoom.yaml",
    epochs=150,
    batch=16,
    device=[0],
    patience=10,
    workers=16,
    # Image augmentation parameters
    hsv_h=0.015,        # Hue augmentation (fraction)
    hsv_s=0.7,          # Saturation augmentation (fraction)
    hsv_v=0.4,          # Value augmentation (fraction)
    degrees=10.0,       # Rotation degrees
    translate=0.1,      # Translation fraction
    scale=0.5,          # Scale augmentation
    shear=0.0,          # Shear degrees
    perspective=0.0001, # Perspective augmentation
    flipud=0.0,         # Vertical flip probability
    fliplr=0.5,         # Horizontal flip probability
    mosaic=1.0,         # Mosaic augmentation probability
    mixup=0.1,          # MixUp augmentation probability
    copy_paste=0.1,     # Copy-paste augmentation probability
    erasing=0.4,        # Random erasing probability
    crop_fraction=1.0,  # Crop fraction for training
)

# %%
model.save(f"{DATASET_DIR}/trained_model.pt")

# %%
