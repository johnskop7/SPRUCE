import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

from load_dataloaders import load_data
sys.path.append("/home/ubuntu/SPRUCE")
from models.ConvNeXt import CustomConvNeXt
from models.ResNet50 import CustomResNet50
from utils.model_utils import load_classifier_model


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalization: (x*std + mean), clamp to [0,1].
    """
    # These must match your DataLoader’s normalize()
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3,1,1)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3,1,1)
    out = tensor.clone()
    out.mul_(imagenet_std).add_(imagenet_mean)
    return out.clamp(0,1)

def main():
    parser = argparse.ArgumentParser(description="Save correctly classified images")
    # Dataset parameters
    parser.add_argument("--dataset_type", type=str, required=True,
                        help="Dataset name (e.g. cardiomegaly, lung_opacity, oct, dementia)")
    parser.add_argument("--data_dir",      type=Path, required=True,
                        help="Root directory of images")
    # Model parameters
    parser.add_argument("--model_arch",    type=str, choices=['convnext','resnet50'],
                        default='convnext', help="Which CNN to use")
    parser.add_argument("--num_classes",   type=int, default=2,
                        help="Number of classes")
    parser.add_argument("--model_weights", type=str)
    # Output & selection
    parser.add_argument("--output_path",   type=Path, required=True,
                        help="Where to save the result images & logs")
    parser.add_argument("--class_number",  type=int, default=1,
                        help="Which class index to select")
    parser.add_argument("--threshold",     type=float, default=0.5,
                        help="Min confidence to accept a prediction")
    parser.add_argument("--num_images",    type=int, default=100,
                        help="How many images to save")
    # DataLoader
    parser.add_argument("--batch_size",    type=int, default=64,
                        help="Batch size")
    parser.add_argument("--image_size",    type=int, default=256,
                        help="Resize images to this size")

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- load data ---
    train_loader, val_loader, test_loader = load_data(
        data_dir     = args.data_dir,
        batch_size   = args.batch_size,
        image_size   = args.image_size,
        normalized   = True,
        dataset_type = args.dataset_type,

    )
    print(f"Loaded test_loader with {len(test_loader.dataset)} images")

    # --- instantiate model ---
    model = load_classifier_model(args.model_arch,model_weights=args.model_weights,num_classes=args.num_classes)
    model.to(device)
    print(f"Initialized {args.model_arch} model (num_classes={args.num_classes})")

    saved = 0
    idx_counter = 0

    # wrap with tqdm
    from tqdm import tqdm
    for batch in tqdm(test_loader, desc="Scanning images", unit="batch"):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs, _ = model(images, return_features=True)
            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

        # find indices where prediction == label
        matches = (preds == labels).nonzero(as_tuple=False).squeeze()
        if matches.numel() == 0:
            continue
        if matches.dim()==0:
            matches = matches.unsqueeze(0)

        for i in matches:
            if saved >= args.num_images:
                break

            true_lbl = labels[i].item()
            conf     = probs[i, preds[i]].item()
            if true_lbl == args.class_number and conf >= args.threshold:
                # denormalize and save
                img_tensor = denormalize(images[i].cpu())
                pil_img    = transforms.ToPILImage()(img_tensor)
                # filename with dataset name
                fname = f"{args.dataset_type}_cls{true_lbl}_idx{idx_counter}.png"
                outp  = args.output_path / fname
                pil_img.save(outp)
                #print(f"Saved {outp} (conf={conf:.2f})")
                saved += 1
            idx_counter += 1

        if saved >= args.num_images:
            break

    print(f"Done: saved {saved}/{args.num_images} images of class {args.class_number}")

if __name__ == "__main__":
    main()
