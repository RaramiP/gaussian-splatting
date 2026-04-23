import os
import argparse
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F
from pathlib import Path
from transformers import Sam3Processor, Sam3Model
import matplotlib.pyplot as plt
import requests
import cv2
import numpy as np

def clean_dir(path):
    if os.path.exists(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

def best_score(results):
    scores = results["scores"]
    index = torch.argmax(scores)
    return results["masks"][index], results["scores"][index], results["boxes"][index]

def process_images(base_path, prompt, result_path):
    base_dir = Path(base_path)
    input_dir = base_dir / "input"
    output_dir = base_dir / "masks"
    if result_path:
        result_dir = Path(result_path)
        result_dir.mkdir(parents=True, exist_ok=True)
        clean_dir(result_dir)

    if not input_dir.exists():
        print(f"Erreur : Le dossier {input_dir} n'existe pas.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_dir(output_dir)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation de {device}")

    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]

    if not images:
        print("Aucune image trouvée dans le dossier input.")
        return

    print(f"Traitement de {len(images)} images...")
    for img_name in sorted(images):
        img_path = os.path.abspath(os.path.join(input_dir, img_name))
        print(f"Analyse de : {img_name}")

        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]        
        
        if len(results["masks"]) == 0:
            print("\033[91m", f"Aucune détection pour {img_name}", "\033[0m", sep="")
            continue
        
        mask, score, box = best_score(results)
        mask = mask.detach().cpu().numpy().squeeze()

        mask_image = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_image)

        output_path = os.path.join(output_dir, img_name)
        mask_image.save(output_path)
        print(f"Sauvegardé : {output_path}")

        if result_path is not None:
            new_img = np.array(image)
            new_mask = np.array(mask_image).astype(float) / 255.0

            res_array = new_img * new_mask[:, :, np.newaxis]
            res_array = res_array.astype(np.uint8)

            Image.fromarray(res_array).save(os.path.join(result_dir, img_name))


    print("\nTraitement terminé avec succès.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation automatique avec SAM 2")
    parser.add_argument("-s", "--source", required=True, help="Chemin du dossier racine (contenant 'input')")
    parser.add_argument("-p", "--prompt", required=True, help="Prompt SAM")
    parser.add_argument("--result", required=False, default=None, help="Chemin du dossier de sortie pour le resultat du masquage")

    args = parser.parse_args()
    process_images(args.source, args.prompt, args.result)
