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
)

# %%
model.save(f"{DATASET_DIR}/trained_model.pt")

# %%
