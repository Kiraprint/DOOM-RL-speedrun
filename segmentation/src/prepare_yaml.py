import pandas as pd
import yaml

# Load class information
class_df = pd.read_csv(
    "/home/kiraprint/Projects/DOOM-RL-speedrun/segmentation/data/classes.csv"
)
class_colors_df = pd.read_csv(
    "/home/kiraprint/Projects/DOOM-RL-speedrun/segmentation/data/class_colors.csv"
)

# Create the YAML data
yaml_data = {
    "path": "/home/kiraprint/Projects/DOOM-RL-speedrun/segmentation/data/yolo_dataset",
    "train": "images/train",
    "val": "images/val",
    "nc": len(class_df),
    "names": class_df["class_name"].tolist(),
}

# Write to YAML file
with open(
    "/home/kiraprint/Projects/DOOM-RL-speedrun/segmentation/data/vizdoom.yaml", "w"
) as f:
    yaml.dump(yaml_data, f, default_flow_style=False)
