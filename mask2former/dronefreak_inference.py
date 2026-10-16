#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


MODEL_ID = "dronefreak/uavid-yolo26s-sem"

# Distinct colors for the eight UAVid classes.
PALETTE = np.array([
    [128, 128, 128],  # 0: Clutter
    [128,   0,   0],  # 1: Building
    [128,  64, 128],  # 2: Road
    [  0,   0, 142],  # 3: Static Car
    [  0, 128,   0],  # 4: Tree
    [107, 142,  35],  # 5: Vegetation
    [220,  20,  60],  # 6: Human
    [  0,   0, 230],  # 7: Moving Car
], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--out", default="yolo_output",
                        help="Output directory")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="Inference image size")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check GPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Download weights if not already cached
    print("Loading model...")
    weights = hf_hub_download(
        repo_id=MODEL_ID,
        filename="best.pt",
    )
    model = YOLO(weights)

    print("Model classes:", model.names)

    # Load the input image
    image = Image.open(image_path).convert("RGB")

    print(f"Input image: {image.size[0]} x {image.size[1]}")
    print("Running semantic segmentation...")

    results = model.predict(
        source=np.array(image),
        task="semantic",
        imgsz=args.imgsz,
        device=device,
        verbose=False,
    )

    result = results[0]

    # Extract the per-pixel class-ID map
    if result.semantic_mask is None:
        raise RuntimeError(
            "No semantic mask returned. Check the installed "
            "Ultralytics version and semantic-task support."
        )

    mask = result.semantic_mask.data.cpu().numpy().squeeze()

    if mask.ndim != 2:
        raise RuntimeError(
            f"Expected a 2D class-ID mask, got shape {mask.shape}"
        )

    # Save class IDs without converting them to RGB.
    # Useful for subsequent geospatial processing.
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    Image.fromarray(mask.astype(np.uint8)).save(mask_path)

    # Create a color visualization
    if mask.min() < 0 or mask.max() >= len(PALETTE):
        raise RuntimeError(
            f"Unexpected class IDs: {np.unique(mask)}"
        )

    color_mask = PALETTE[mask.astype(np.int64)]

    color_path = output_dir / f"{image_path.stem}_colors.png"
    Image.fromarray(color_mask).save(color_path)

    # Create a blended overlay
    # Resize the visualization if the model returns a different resolution.
    color_image = Image.fromarray(color_mask)

    if color_image.size != image.size:
        color_image = color_image.resize(
            image.size, Image.Resampling.NEAREST
        )

    overlay = Image.blend(image, color_image, alpha=0.5)
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    overlay.save(overlay_path)

    # Print class statistics
    print("\nPixel statistics:")
    for class_id, class_name in model.names.items():
        count = int(np.count_nonzero(mask == class_id))
        percentage = 100.0 * count / mask.size
        print(
            f"  {class_id}: {class_name:15s} "
            f"{count:10d} pixels ({percentage:5.2f}%)"
        )

    print("\nResults saved:")
    print(f"  Class IDs: {mask_path}")
    print(f"  Colors:    {color_path}")
    print(f"  Overlay:   {overlay_path}")


if __name__ == "__main__":
    main()