import os
import shutil

import cv2
import numpy as np
import pandas as pd


def load_class_mapping(csv_path):
    """Load class to color mapping from CSV"""
    df = pd.read_csv(csv_path)
    return df


def mask_to_bboxes(mask_path, class_colors_df):
    """Convert segmentation mask to YOLO format bounding boxes"""
    mask = cv2.imread(mask_path)
    bboxes = []

    # Get image dimensions
    height, width = mask.shape[:2]

    # Process each class
    for _, row in class_colors_df.iterrows():
        class_id = row["class_id"]
        color_value = row["gray_scale_value"]

        # Create binary mask for this class
        binary_mask = (mask == color_value).astype(np.uint8)
        if binary_mask.sum() == 0:
            continue

        # Find contours
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Process each contour (object instance)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Convert to YOLO format (x_center, y_center, width, height) normalized
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            norm_width = w / width
            norm_height = h / height

            bboxes.append((class_id, x_center, y_center, norm_width, norm_height))

    return bboxes


def prepare_yolo_dataset(dataset_dir, output_dir, class_colors_csv):
    """Prepare YOLO dataset from Vizdoom segmentation data"""
    # Load class mapping
    class_colors_df = load_class_mapping(class_colors_csv)

    # Create output directories
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

    # Get all RGB images
    rgb_dir = os.path.join(dataset_dir, "rgb")
    labels_dir = os.path.join(dataset_dir, "labels")

    image_files = sorted(os.listdir(rgb_dir))

    # Split into train/val (80/20)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # Process training images
    for img_file in train_files:
        img_path = os.path.join(rgb_dir, img_file)
        img_id = img_file.split(".")[0]

        # Find corresponding label file
        label_files = [f for f in os.listdir(labels_dir) if img_id in f]
        if not label_files:
            continue

        label_path = os.path.join(labels_dir, label_files[0])

        # Convert mask to bboxes
        bboxes = mask_to_bboxes(label_path, class_colors_df)

        # Save image
        shutil.copy(img_path, os.path.join(output_dir, "images", "train", img_file))

        # Save labels
        with open(
            os.path.join(output_dir, "labels", "train", f"{img_id}.txt"), "w"
        ) as f:
            for bbox in bboxes:
                f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")

    # Process validation images
    for img_file in val_files:
        img_path = os.path.join(rgb_dir, img_file)
        img_id = img_file.split(".")[0]

        # Find corresponding label file
        label_files = [f for f in os.listdir(labels_dir) if img_id in f]
        if not label_files:
            continue

        label_path = os.path.join(labels_dir, label_files[0])

        # Convert mask to bboxes
        bboxes = mask_to_bboxes(label_path, class_colors_df)

        # Save image
        shutil.copy(img_path, os.path.join(output_dir, "images", "val", img_file))

        # Save labels
        with open(os.path.join(output_dir, "labels", "val", f"{img_id}.txt"), "w") as f:
            for bbox in bboxes:
                f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
