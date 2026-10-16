import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def inference_autocast(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM 3 text-prompted segmentation on an image."
    )
    parser.add_argument("image", help="Input image path")
    parser.add_argument(
        "--output",
        default="sam3_output",
        help="Output directory (default: sam3_output)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["road", "building", "car"],
        help="Text prompts to segment",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    print(f"Image: {image_path}")
    print(f"Image size: {image.width} x {image.height}")
    print(f"Prompts: {args.prompts}")

    # Load SAM 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = build_sam3_image_model(device=device)

    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.threshold,
    )

    # Compute image features once; reuse them for each text prompt
    with torch.inference_mode(), inference_autocast(device):
        state = processor.set_image(image)

    # Build a visualization overlay
    overlay = image_np.copy()
    colors = [
        (255, 70, 70),    # red
        (50, 220, 80),    # green
        (50, 130, 255),   # blue
        (255, 210, 40),   # yellow
        (220, 60, 220),   # magenta
        (40, 220, 220),   # cyan
    ]

    combined_masks = []

    for i, prompt in enumerate(args.prompts):
        print(f"\nSegmenting: {prompt!r}")

        with torch.inference_mode(), inference_autocast(device):
            result = processor.set_text_prompt(
                state=state,
                prompt=prompt,
            )

        masks = result["masks"].detach().float().cpu().numpy()
        scores = result["scores"].detach().float().cpu().numpy()
        boxes = result["boxes"].detach().float().cpu().numpy()

        # Expected masks shape: [N, 1, H, W].
        # Remove the singleton channel dimension if present.
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]

        if masks.ndim == 2:
            masks = masks[None, ...]

        # Union all instances for this prompt into one binary mask.
        if len(masks) > 0:
            class_mask = np.any(masks.astype(bool), axis=0)
        else:
            class_mask = np.zeros(
                (image.height, image.width), dtype=bool
            )

        # Save binary mask: 255 = detected, 0 = background.
        mask_path = output_dir / f"{i:02d}_{prompt.replace(' ', '_')}_mask.png"
        Image.fromarray(
            class_mask.astype(np.uint8) * 255
        ).save(mask_path)

        color = colors[i % len(colors)]

        # Blend the class mask into the visualization.
        overlay[class_mask] = (
            0.5 * overlay[class_mask]
            + 0.5 * np.array(color)
        ).astype(np.uint8)

        combined_masks.append(class_mask)

        print(f"  Instances detected: {len(scores)}")
        print(f"  Mask saved: {mask_path}")

        for j, (score, box) in enumerate(zip(scores, boxes)):
            x0, y0, x1, y1 = box
            print(
                f"    #{j + 1}: score={score:.3f}, "
                f"box=({x0:.1f}, {y0:.1f}, "
                f"{x1:.1f}, {y1:.1f})"
            )

    # Save combined overlay
    overlay_path = output_dir / "overlay.png"
    Image.fromarray(overlay).save(overlay_path)

    # Save union of all prompted classes
    if combined_masks:
        union = np.logical_or.reduce(combined_masks)
    else:
        union = np.zeros((image.height, image.width), dtype=bool)

    union_path = output_dir / "all_classes_mask.png"
    Image.fromarray(union.astype(np.uint8) * 255).save(union_path)

    print("\nFinished.")
    print(f"Overlay: {overlay_path}")
    print(f"Combined mask: {union_path}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
