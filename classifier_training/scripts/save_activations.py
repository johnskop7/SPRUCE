#!/usr/bin/env python
import os, yaml, pickle,sys
# Add project root to PYTHONPATH so modules like utils, data, models, etc. can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from tqdm import tqdm
from data.loaders import load_dataloaders
from models import load_classifier_model

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def save_activations(cfg, device):
    print("Loading data…")
    train_loader, _, _ = load_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        image_size=cfg["data"]["image_size"],
        normalized=True,
        dataset_type=cfg["data"]["dataset_type"]
    )

    print("Instantiating model…")
    model = load_classifier_model(
        arch=cfg["model"]["arch"],
        num_classes=cfg["model"]["num_classes"],
        weights=cfg["model"]["weights"],
        device=device
    )
    model.eval()

    activations = {}
    print("Extracting…")
    with torch.no_grad():
        for imgs, _ in tqdm(train_loader, desc="batches"):
            imgs = imgs.to(device)
            logits, feats = model(imgs, return_features=True)
            preds = logits.softmax(1).argmax(1).cpu().numpy()
            for feat_np, p in zip(feats.cpu().numpy(), preds):
                activations.setdefault(int(p), []).append(feat_np)

    # stack
    for k in activations:
        activations[k] = np.stack(activations[k])

    out = cfg["extract"]["save_path"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"Saving to {out}")
    with open(out, "wb") as f:
        pickle.dump(activations, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        help="Path to activations.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_activations(cfg, device)
