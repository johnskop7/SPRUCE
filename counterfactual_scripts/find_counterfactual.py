#!/usr/bin/env python3
import os
import sys
import argparse
import pickle
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from cnn_models import load_cnn_model
from PIL import Image
from matplotlib.animation import FuncAnimation, FFMpegWriter
from utils import (
    convert_tensor_to_png,
    latent_vector_optimization,
    preprocess_for_classifier
)

# allow importing from your project root
sys.path.append("/home/ubuntu/PTI")  # adjust or remove


def save_config(args: argparse.Namespace, output_dir: str) -> None:
    """Save command-line arguments to a config.txt in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.txt")
    with open(config_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    print(f"Configuration saved to {config_path}")


def load_original_image(image_path: str, image_size: int) -> np.ndarray:
    """Load and resize an image to (image_size, image_size)."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img.resize((image_size, image_size)))


def compute_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Compute per‐pixel absolute difference and return a grayscale map."""
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16)).astype(np.uint8)
    return cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)


def tensor_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert a [1,C,H,W] tensor in [0,1] to an HxWxC uint8 NumPy array."""
    t = img_tensor.squeeze(0).detach().cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return arr


def process_counterfactual(
    generator_path: str,
    input_path: str,
    original_images_dir: str,
    model_type: str,
    model_weights: str,
    lr: float,
    epochs: int,
    output_dir: str,
    num_classes: int,
    normalized: bool,
    image_size: int,
    lambda_piece: float,
    lambda_perc: float,
    lambda_latent: float,
    lambda_image: float
) -> None:
    """Compute and save a single counterfactual image & metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_name = os.path.basename(input_path).replace("_cf_feature_vector.pickle", "")
    print(f"\n▶ Processing {image_name}")

    # -- load CNN --
    cnn = load_cnn_model(model_type=model_type,
                         model_weights=model_weights,
                         num_classes=num_classes).to(device)

    # -- load CF vector and pivot latent --
    with open(input_path, "rb") as f:
        data = pickle.load(f)
        cf_x = data["cf_feature_vector"].to(device)
        ws = data["ws"].to(device)

    # -- load original image --
    orig_path = os.path.join(original_images_dir, f"{image_name}.png")
    if not os.path.exists(orig_path):
        print(f"Missing original image, skipping {image_name}")
        return
    orig_np = load_original_image(orig_path, image_size)

    # -- load GAN generator --
    if not os.path.exists(generator_path):
        print(f"Missing generator, skipping {image_name}")
        return
    gan = torch.load(generator_path, map_location=device).to(device)

    # -- optimize latent to match cf_x --
    w_e, _ = latent_vector_optimization(
        generator=gan,
        w_plus_init=ws,
        device=device,
        classifier=cnn,
        cf_feature_vector=cf_x,
        original_image=orig_np,
        epochs=epochs,
        learning_rate=lr,
        lambda_piece=lambda_piece,
        lambda_perc=lambda_perc,
        lambda_latent=lambda_latent,
        lambda_image=lambda_image,
        image_size=image_size,
        normalized=normalized
    )

    # -- synth images --
    x_orig = gan.synthesis(ws, noise_mode="const", force_fp32=True)
    x_orig = (x_orig * 0.5 + 0.5).clamp(0, 1)
    x_cf   = gan.synthesis(w_e,  noise_mode="const", force_fp32=True)
    x_cf   = (x_cf   * 0.5 + 0.5).clamp(0, 1)

    # -- compute metrics --
    inv_np = tensor_to_numpy(x_orig)
    cf_np  = tensor_to_numpy(x_cf)

    l1_norm = np.abs(orig_np.astype(np.float32)/255.0 - cf_np.astype(np.float32)/255.0).mean()

    with torch.no_grad():
        cnn.eval()
        inp = preprocess_for_classifier(x_cf) if normalized else x_cf
        logits, _ = cnn(inp)
        probs = torch.softmax(logits, dim=1)[0]
        pred_prob, pred_cls = probs.max(0)

    # -- write metrics --
    mpath = os.path.join(output_dir, "counterfactual_metrics.txt")
    header = "Image,Normalized_L1,Pred_Prob,Pred_Class\n"
    write_h = not os.path.exists(mpath)
    with open(mpath, "a") as mf:
        if write_h: mf.write(header)
        mf.write(f"{image_name},{l1_norm:.4f},{pred_prob:.4f},{pred_cls.item()}\n")

    # -- save comparison grid --
    comp_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)
    comp_path = os.path.join(comp_dir, f"{image_name}_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    axes[0].imshow(orig_np);      axes[0].set_title("Original");      axes[0].axis("off")
    axes[1].imshow(inv_np);       axes[1].set_title("Inversion");     axes[1].axis("off")
    axes[2].imshow(cf_np);        axes[2].set_title("Counterfactual");axes[2].axis("off")
    plt.savefig(comp_path)
    plt.close()
    print(f"  ✓ Saved comparison to {comp_path}")

    # -- save CF image alone --
    cf_dir = os.path.join(output_dir, "counterfactuals")
    os.makedirs(cf_dir, exist_ok=True)
    cf_path = os.path.join(cf_dir, f"{image_name}_cf.png")
    convert_tensor_to_png(x_cf.squeeze(0).cpu(), filename=cf_path,
                          normalized=False, image_size=image_size)
    print(f"  ✓ Saved CF image to {cf_path}")


def batch_process_counterfactuals(
    checkpoints_dir: str,
    input_dir: str,
    original_images_dir: str,
    model_type: str,
    model_weights: str,
    lr: float,
    epochs: int,
    output_dir: str,
    num_classes: int,
    normalized: bool,
    image_size: int,
    lambda_piece: float,
    lambda_perc: float,
    lambda_latent: float,
    lambda_image: float
) -> None:
    """Run `process_counterfactual` on every fine-tuned GAN in `checkpoints_dir`."""
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(checkpoints_dir):
        if not fname.endswith(".pt"):
            continue
        parts = fname.split("_")
        if len(parts) < 3:
            print(f"Skipping invalid {fname}")
            continue

        name = "_".join(parts[2:])[:-3]
        gan_path = os.path.join(checkpoints_dir, fname)
        cf_pickle = os.path.join(input_dir, f"{name}_cf_feature_vector.pickle")
        if not os.path.exists(cf_pickle):
            print(f"No CF vector for {name}, skipping")
            continue

        process_counterfactual(
            generator_path      = gan_path,
            input_path          = cf_pickle,
            original_images_dir = original_images_dir,
            model_type          = model_type,
            model_weights       = model_weights,
            lr                  = lr,
            epochs              = epochs,
            output_dir          = output_dir,
            num_classes         = num_classes,
            normalized          = normalized,
            image_size          = image_size,
            lambda_piece        = lambda_piece,
            lambda_perc         = lambda_perc,
            lambda_latent       = lambda_latent,
            lambda_image        = lambda_image
        )

    print("\n✅ All counterfactuals generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch‐generate counterfactuals via a central YAML config."
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to config.yaml"
    )
    args = parser.parse_args()

    # 1) load YAML
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 2) unpack
    gan_cfg = cfg["gan"]
    cnn_cfg = cfg["cnn"]
    opt_cfg = cfg["opt"]

    # 3) derive input_dir from piece output
    input_dir = os.path.join(gan_cfg["output_dir"], "feature_vectors")

    batch_process_counterfactuals(
        checkpoints_dir      = gan_cfg["checkpoints_dir"],
        input_dir            = input_dir,
        original_images_dir  = gan_cfg["original_images_dir"],
        model_type           = cnn_cfg["type"],
        model_weights        = cnn_cfg["weights"],
        lr                   = opt_cfg["lr"],
        epochs               = opt_cfg["epochs"],
        output_dir           = gan_cfg["output_dir"],
        num_classes          = cnn_cfg["num_classes"],
        normalized           = cnn_cfg["normalized"],
        image_size           = opt_cfg["image_size"],
        lambda_piece         = opt_cfg["lambda_piece"],
        lambda_perc          = opt_cfg["lambda_perc"],
        lambda_latent        = opt_cfg["lambda_latent"],
        lambda_image         = opt_cfg["lambda_image"]
    )
