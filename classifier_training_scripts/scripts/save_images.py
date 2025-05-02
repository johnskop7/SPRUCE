#!/usr/bin/env python
import os, yaml , sys
# Add project root to PYTHONPATH so modules like utils, data, models, etc. can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from data.loaders import load_dataloaders
from models import load_classifier_model

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def denormalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485,0.456,0.406], device=img.device).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=img.device).view(3,1,1)
    return (img * std + mean).clamp(0,1)

def save_correct(cfg, device):
    print("Loading data…")
    _, _, test_loader = load_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        image_size=cfg["data"]["image_size"],
        normalized=True,
        dataset_type=cfg["data"]["dataset_type"]
    )

    print("Loading model…")
    model = load_classifier_model(
        arch=cfg["model"]["arch"],
        num_classes=cfg["model"]["num_classes"],
        weights=cfg["model"]["weights"],
        device=device
    )
    model.eval()

    params = cfg["extract"]
    outdir = Path(params["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    saved = 0
    counter = 0

    print("Scanning…")
    for imgs, labels in tqdm(test_loader, desc="batches"):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            logits, _ = model(imgs, return_features=True)
            probs = logits.softmax(1)
        preds = probs.argmax(1)

        correct_idxs = (preds == labels).nonzero(as_tuple=False).flatten()
        for i in correct_idxs:
            if saved >= params["num_images"]:
                break
            lbl  = int(labels[i])
            conf = float(probs[i,lbl])
            if lbl == params["class_number"] and conf >= params["threshold"]:
                img = denormalize(imgs[i].cpu())
                pil = transforms.ToPILImage()(img)
                name = f"{cfg['data']['dataset_type']}_cls{lbl}_{counter:04d}.png"
                pil.save(outdir / name)
                saved += 1
            counter += 1
        if saved >= params["num_images"]:
            break

    print(f"Saved {saved}/{params['num_images']} images to {outdir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        help="Path to correct_images.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_correct(cfg, device)
