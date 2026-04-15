
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

detector = pipeline(
    task="zero-shot-object-detection",
    model="IDEA-Research/grounding-dino-tiny",
    device=0 if device == "cuda" else -1
)

root = os.getcwd()
if not os.path.exists(os.path.join(root, "sam_model")):
    model = AutoModelForMaskGeneration.from_pretrained("facebook/sam-vit-base").to(device)
    model.save_pretrained(os.path.join(root, "sam_model"))
else:
    model = AutoModelForMaskGeneration.from_pretrained(os.path.join(root, "sam_model")).to(device)

if not os.path.exists(os.path.join(root, "sam_processor")):
    processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")
    processor.save_pretrained(os.path.join(root, "sam_processor"))
else:
    processor = AutoProcessor.from_pretrained(os.path.join(root, "sam_processor"))

def load_image(image_str):
    return Image.open(image_str).convert("RGB")


def get_boxes(detections):
    boxes = []
    for det in detections:
        b = det["box"]
        boxes.append([b["xmin"], b["ymin"], b["xmax"], b["ymax"]])
    return [boxes]


def refine_masks(masks):
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1).mean(dim=-1)
    masks = (masks > 0).int().numpy().astype(np.uint8)
    return masks


def detect(image, labels, threshold=0.3):
    labels = [l if l.endswith(".") else l + "." for l in labels]
    results = detector(image, candidate_labels=labels, threshold=threshold)

    detections = []
    for r in results:
        detections.append({
            "score": r["score"],
            "label": r["label"],
            "box": r["box"],
            "mask": None
        })

    return detections


def segment(image, detections):
    boxes = get_boxes(detections)

    inputs = processor(
        images=image,
        input_boxes=boxes,
        return_tensors="pt"
    ).to(device)

    outputs = model(**inputs)

    masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs.original_sizes,
        inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks)

    for det, mask in zip(detections, masks):
        det["mask"] = mask

    return detections


def grounded_segmentation(image, labels, threshold=0.3):
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold)

    if len(detections) == 0:
        return np.array(image), []

    detections = segment(image, detections)

    return np.array(image), detections


def get_best_detection(detections):
    if len(detections) == 0:
        return None

    return max(detections, key=lambda d: d["score"])

def segmentate(images, output):
    if not os.path.exists(output):
        os.makedirs(output)

    labels = ["object"]

    results_json = {}

    for image_name in tqdm(os.listdir(images)):

        if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        full_image_path = os.path.join(images, image_name)

        image_np, detections = grounded_segmentation(
            full_image_path,
            labels=labels,
            threshold=0.3
        )

        if len(detections) == 0:
            print(f"Aucune détection pour {image_name}")
            continue

        best_detection = get_best_detection(detections)

        if (best_detection is None or best_detection["mask"] is None or best_detection["score"] < 0.40):
            print(f"{image_name} ignoré (score {best_detection['score']:.4f})")
            continue


        best_mask = best_detection["mask"]
        best_score = best_detection["score"]

        # Sauvegarde masque
        mask_image = (best_mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_image)

        mask_image_path = os.path.join(
            output,
            f"{os.path.splitext(image_name)[0]}_mask.png"
        )
        mask_image.save(mask_image_path)

