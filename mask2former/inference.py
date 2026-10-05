import sys
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
)


MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"


def main():

    if len(sys.argv) != 2:
        print("Usage: python inference.py image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    print(f"Image size: {image.size}")

    print("Loading model...")
    processor = AutoImageProcessor.from_pretrained(MODEL)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL
    ).cuda()

    model.eval()

    print("Running inference...")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        k: v.cuda()
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    # H x W tensor containing the semantic class ID
    segmentation = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]]
    )[0]

    print("Segmentation type:", type(segmentation))
    print("Segmentation shape:", segmentation.shape)
    print("Number of classes:", len(model.config.id2label))

    print("\nClasses:")
    for class_id, label in model.config.id2label.items():
        print(f"{class_id:3d}: {label}")

    # Save raw segmentation
    segmentation.cpu().numpy().astype("uint8").tofile(
        "segmentation.raw"
    )

    print("\nSaved segmentation.raw")


    seg = segmentation.cpu().numpy()

    np.random.seed(0)

    num_classes = len(model.config.id2label)

    colors = np.random.randint(
        0, 255,
        size=(num_classes, 3),
        dtype=np.uint8
    )

    mask_rgb = colors[seg]

    original = np.array(image)

    overlay = (
        0.5 * original +
        0.5 * mask_rgb
    ).astype(np.uint8)

    Image.fromarray(overlay).save(
        "segmentation_overlay.jpg"
    )

    # create a mask with markings, curbestone, road mapped to terrain 
    # terrain = {13, 42, 7}  # Replace with the class IDs you want
    # mask = np.isin(seg, list(terrain)).astype(np.uint8)
if __name__ == "__main__":
    main()