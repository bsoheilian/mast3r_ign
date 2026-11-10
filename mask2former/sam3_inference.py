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


def display_overlay(overlay, prompts, class_masks, colors):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, (image_axes, legend_axes) = plt.subplots(
        1, 2,
        figsize=(12, 8),
        gridspec_kw={"width_ratios": [4, 1]},
        constrained_layout=True,
    )
    image_axes.imshow(overlay)
    image_axes.axis("off")
    legend_axes.axis("off")
    legend_axes.legend(
        handles=[
            Patch(
                facecolor=np.array(colors[index % len(colors)]) / 255,
                label=prompt,
            )
            for index, prompt in enumerate(prompts)
        ],
        loc="upper left",
        title="Classes",
        frameon=False,
    )
    tooltip = image_axes.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.9},
    )
    tooltip.set_visible(False)
    height, width = overlay.shape[:2]

    def on_motion(event):
        if event.inaxes is not image_axes or event.xdata is None or event.ydata is None:
            if tooltip.get_visible():
                tooltip.set_visible(False)
                figure.canvas.draw_idle()
            return

        column = int(np.floor(event.xdata + 0.5))
        row = int(np.floor(event.ydata + 0.5))
        if not (0 <= column < width and 0 <= row < height):
            tooltip.set_visible(False)
            figure.canvas.draw_idle()
            return

        labels = [
            prompt
            for prompt, class_mask in zip(prompts, class_masks)
            if class_mask[row, column]
        ]
        tooltip.xy = (event.xdata, event.ydata)
        tooltip.set_text("\n".join(labels) if labels else "Background")
        tooltip.set_position((12 if column < width / 2 else -12,
                              12 if row >= height / 2 else -12))
        tooltip.set_horizontalalignment("left" if column < width / 2 else "right")
        tooltip.set_verticalalignment("bottom" if row >= height / 2 else "top")
        tooltip.set_visible(True)
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("motion_notify_event", on_motion)
    figure.canvas.mpl_connect("figure_leave_event", on_motion)
    plt.show()


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
        "--output-mode",
        choices=["overlay", "all", "display"],
        default="all",
        help="Save overlay only, save all outputs, or display without saving (default: all)",
    )
    parser.add_argument(
        "-display",
        "--display",
        dest="output_mode",
        action="store_const",
        const="display",
        default=argparse.SUPPRESS,
        help="Show the overlay without saving outputs",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["cars", "trees", "vegetations", "buildings", "trucks", "furnitures", "poles", "buses", "motorcycles", "bicycles", "pedestrians", "crane", "stairs", "water", "chars", "roofs"],
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
    if args.output_mode != "display":
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
        if args.output_mode == "all":
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

        if args.output_mode in ("all", "display"):
            combined_masks.append(class_mask)

        print(f"  Instances detected: {len(scores)}")
        if args.output_mode == "all":
            print(f"  Mask saved: {mask_path}")

        for j, (score, box) in enumerate(zip(scores, boxes)):
            x0, y0, x1, y1 = box
            print(
                f"    #{j + 1}: score={score:.3f}, "
                f"box=({x0:.1f}, {y0:.1f}, "
                f"{x1:.1f}, {y1:.1f})"
            )

    # Save combined overlay
    if args.output_mode == "display":
        display_overlay(overlay, args.prompts, combined_masks, colors)
    else:
        overlay_path = output_dir / "overlay.png"
        Image.fromarray(overlay).save(overlay_path)

    # Save union of all prompted classes
    if args.output_mode == "all":
        if combined_masks:
            union = np.logical_or.reduce(combined_masks)
        else:
            union = np.zeros((image.height, image.width), dtype=bool)

        union_path = output_dir / "all_classes_mask.png"
        Image.fromarray(union.astype(np.uint8) * 255).save(union_path)

    print("\nFinished.")
    if args.output_mode != "display":
        print(f"Overlay: {overlay_path}")
        if args.output_mode == "all":
            print(f"Combined mask: {union_path}")
        print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

# cars trucks buses motorcycles bicycles pedestrians buildings rooftops houses trees bushes vegetation furniture traffic-signs street-lights poles
# given that road areas are already segmented correctly and we need only to segment any other object that can apear over road mask we can maybe limite it to any object that 
# can not be recognized from ground image: 
#cars and any vehicle
#pedestrians
#rooftops , buildings hous 
# vegetation
#poles
#crane grue

#do a first minimal test and validate on the results if it is correct. 
