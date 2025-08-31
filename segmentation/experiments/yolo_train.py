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
    hsv_s=0.1,          # Saturation augmentation (fraction)
    hsv_v=0.1,          # Value augmentation (fraction)
    translate=0.1,      # Translation fraction
    scale=0.5,          # Scale augmentation
    shear=0.0,          # Shear degrees
    fliplr=0.5,         # Horizontal flip probability
)

# %%
model.save(f"{DATASET_DIR}/trained_model.pt")

# %%
