import os
import sys
import argparse
import pickle
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from functools import partial
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Adjust or remove this path addition as needed:
sys.path.append("/home/ubuntu/PTI")

# Local imports
from utils import (
    convert_tensor_to_png,
    latent_vector_optimization,
    preprocess_for_classifier,
    update,
    plot_images_difference,
    plot_orig_and_cf
)
from cnn_models import load_cnn_model


def save_config(args: argparse.Namespace, output_dir: str) -> None:
    """
    Save command-line arguments to a config file in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.txt")

    with open(config_path, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    print(f"Configuration saved to {config_path}")


def load_original_image(image_path: str, image_size: int) -> np.ndarray:
    """
    Load, resize, and preprocess an original image from a given path.

    Args:
        image_path (str): Path to the original image file.
        image_size (int): Desired (width, height) for resizing.

    Returns:
        np.ndarray: The loaded and resized image in [0, 255].
    """
    pil_image = Image.open(image_path).convert("RGB")
    pil_image = pil_image.resize((image_size, image_size))
    return np.array(pil_image)


def compute_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Compute absolute difference between two RGB images and return a single-channel map.

    Args:
        img1 (np.ndarray): First image in [0, 255].
        img2 (np.ndarray): Second image in [0, 255].

    Returns:
        np.ndarray: Grayscale image representing per-pixel absolute differences.
    """
    diff_3ch = np.abs(img1.astype(np.int16) - img2.astype(np.int16)).astype(np.uint8)
    diff_gray = cv2.cvtColor(diff_3ch, cv2.COLOR_RGB2GRAY)
    return diff_gray


def tensor_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a single image tensor (C, H, W) in [0,1] to a NumPy array in [0,255].

    Args:
        img_tensor (torch.Tensor): Image tensor of shape (1, C, H, W) or (C, H, W).

    Returns:
        np.ndarray: Corresponding image data in [0, 255], shape (H, W, C).
    """
    # Remove any batch dimension
    img_tensor = img_tensor.squeeze(0).detach().cpu()
    # Permute to [H, W, C] and scale to 0-255
    return (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


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
    """
    Compute and save a counterfactual for a single image using its fine-tuned generator and feature vector.

    1) Load the CNN model.
    2) Load the precomputed counterfactual feature vector (CF) and latent vector (ws).
    3) Run latent vector optimization to produce a counterfactual image.
    4) Compute metrics (e.g., L1 difference, predicted probability).
    5) Save comparison images and the final counterfactual.

    Args:
        generator_path (str): Path to the fine-tuned GAN (.pt) file.
        input_path (str): File path to the pickle file containing 'cf_feature_vector' & 'ws'.
        original_images_dir (str): Directory containing the original images before inversion.
        model_type (str): CNN architecture type (resnet50, convnext).
        model_weights (str): Path to pretrained CNN model weights.
        lr (float): Learning rate for latent vector optimization.
        epochs (int): Number of optimization steps.
        output_dir (str): Directory to save results (images, metrics).
        num_classes (int): Number of classes in the dataset.
        normalized (bool): Whether the CNN expects normalized inputs.
        image_size (int): Spatial resolution for both CNN and GAN.
        lambda_piece (float): Weight for the PIECE-related loss.
        lambda_perc (float): Weight for the perceptual (LPIPS) loss.
        lambda_latent (float): Weight for latent regularization.
        lambda_image (float): Weight for image L1 regularization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract image name from the pickle file name
    image_name = os.path.basename(input_path).replace("_cf_feature_vector.pickle", "")
    print(f"Processing counterfactual for: {image_name}")

    # 1) Load the CNN model
    cnn_model = load_cnn_model(
        model_type=model_type,
        model_weights=model_weights,
        num_classes=num_classes
    ).to(device)

    # 2) Load CF feature vector and latent vector
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
        cf_feature_vector = data['cf_feature_vector']
        ws = data['ws']

    # Load original image
    original_image_path = os.path.join(original_images_dir, f"{image_name}.png")
    if not os.path.exists(original_image_path):
        print(f"Original image not found for {image_name}. Skipping...")
        return
    original_image = load_original_image(original_image_path, image_size)

    # Load fine-tuned GAN
    if not os.path.exists(generator_path):
        print(f"Generator not found for {image_name}. Skipping...")
        return
    print(f"Loading fine-tuned GAN from {generator_path}")
    gan = torch.load(generator_path).cuda()

    # 3) Latent vector optimization
    w_e, images_for_gif = latent_vector_optimization(
        generator=gan,
        w_plus_init=ws,
        device=device,
        classifier=cnn_model,
        cf_feature_vector=cf_feature_vector,
        original_image=original_image,
        epochs=epochs,
        learning_rate=lr,
        lambda_piece=lambda_piece,
        lambda_perc=lambda_perc,
        lambda_latent=lambda_latent,
        lambda_image=lambda_image,
        image_size=image_size,
        normalized=normalized
    )

    # Generate final images from latent vectors
    x_gen = gan.synthesis(w_e, noise_mode='const', force_fp32=True)
    x_gen = (x_gen * 0.5 + 0.5).clamp(0, 1)

    x_orig = gan.synthesis(ws, noise_mode="const", force_fp32=True)
    x_orig = (x_orig * 0.5 + 0.5).clamp(0, 1)

    # 4) Compute metrics
    # Convert images to NumPy [0,255]
    orig_img_np = original_image
    inverted_img_np = tensor_to_numpy(x_orig)
    counterfactual_img_np = tensor_to_numpy(x_gen)

    # L1 difference (normalized) between the original image and the final CF
    original_float = orig_img_np.astype(np.float32) / 255.0
    counterfactual_float = counterfactual_img_np.astype(np.float32) / 255.0
    l1_normalized = np.abs(original_float - counterfactual_float).mean()

    # Classifier prediction on the counterfactual
    with torch.no_grad():
        cnn_model.eval()
        if normalized:
            input_tensor = preprocess_for_classifier(x_gen)
        else:
            input_tensor = x_gen

        logits, features = cnn_model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        predicted_prob, predicted_class = probs.max(dim=1)
        predicted_prob = predicted_prob.item()
        predicted_class = predicted_class.item()

    # Write metrics
    metrics_path = os.path.join(output_dir, "counterfactual_metrics.txt")
    header = "Image,Normalized_L1,Predicted_Probability,Predicted_Class\n"
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a") as f:
        if write_header:
            f.write(header)
        f.write(f"{image_name},{l1_normalized:.4f},{predicted_prob:.4f},{predicted_class}\n")

    # Compute difference images
    black_image = np.zeros_like(orig_img_np)
    diff_orig_vs_inverted_np = compute_difference(orig_img_np, inverted_img_np)
    diff_orig_vs_counterfactual_np = compute_difference(orig_img_np, counterfactual_img_np)

    # 5) Save results
    # Save side-by-side comparison images
    comparison_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    comparison_output_path = os.path.join(comparison_dir, f"{image_name}_comparison.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    images = [
        orig_img_np, inverted_img_np, counterfactual_img_np,
        black_image, diff_orig_vs_inverted_np, diff_orig_vs_counterfactual_np
    ]
    titles = [
        "Original Image",
        "Inverted Original",
        "Counterfactual Image",
        "",
        "Diff: Orig vs Inverted",
        "Diff: Orig vs Counterfactual"
    ]
    for ax, img, title in zip(axes.flat, images, titles):
        if "Diff" in title:
            ax.imshow(img, cmap="magma")
        else:
            ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(comparison_output_path)
    plt.close()
    print(f"Saved comparison figure to {comparison_output_path}")

    # Save just the final CF image
    cf_dir = os.path.join(output_dir, "counterfactuals")
    os.makedirs(cf_dir, exist_ok=True)
    cf_image_path = os.path.join(cf_dir, f"{image_name}_cf.png")
    convert_tensor_to_png(
        x_gen.squeeze(0).detach().cpu(),
        filename=cf_image_path,
        normalized=False,
        image_size=image_size
    )
    print(f"Saved counterfactual image to {cf_image_path}")
    print(f"Finished processing {image_name}.\n")


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
    """
    Iterate over each .pt generator in `checkpoints_dir`. For each generator file:
      1) Parse out an image name.
      2) Locate the corresponding `_cf_feature_vector.pickle` in `input_dir`.
      3) If found, call `process_counterfactual` to produce and save the CF image.

    Args:
        checkpoints_dir (str): Directory containing fine-tuned GAN generator files (.pt).
        input_dir (str): Directory containing the CF feature vector pickles.
        original_images_dir (str): Directory with original images.
        model_type (str): CNN architecture type.
        model_weights (str): Path to pretrained CNN model weights.
        lr (float): Learning rate for latent vector optimization.
        epochs (int): Number of optimization epochs.
        output_dir (str): Where to save images and metrics.
        num_classes (int): Number of classes in the dataset.
        normalized (bool): Whether the CNN expects normalized inputs.
        image_size (int): Spatial size (width, height) for CNN/GAN.
        lambda_piece (float): Weight for PIECE loss.
        lambda_perc (float): Weight for perceptual (LPIPS) loss.
        lambda_latent (float): Weight for latent vector regularization.
        lambda_image (float): Weight for image L1 regularization.
    """
    os.makedirs(os.path.join(output_dir, "counterfactuals"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)

    # Find all fine-tuned GAN generator files
    generator_files = [
        f for f in os.listdir(checkpoints_dir)
        if f.endswith(".pt")
    ]

    for generator_file in generator_files:
        # Example generator filename format: "model_{something}_{image_name}.pt"
        parts = generator_file.split("_")
        if len(parts) < 3:
            print(f"Skipping invalid filename: {generator_file}")
            continue

        # Attempt to reconstruct the image name (e.g. everything after 'model_' plus extra info)
        # If your naming convention differs, adjust this parsing.
        image_name = "_".join(parts[2:])[:-3]  # remove ".pt" at the end
        generator_path = os.path.join(checkpoints_dir, generator_file)

        # Look for the corresponding CF pickle
        pickle_name = f"{image_name}_cf_feature_vector.pickle"
        input_path = os.path.join(input_dir, pickle_name)

        if not os.path.exists(input_path):
            print(f"No CF pickle found for {image_name}, skipping...")
            continue

        # Process that image
        process_counterfactual(
            generator_path=generator_path,
            input_path=input_path,
            original_images_dir=original_images_dir,
            model_type=model_type,
            model_weights=model_weights,
            lr=lr,
            epochs=epochs,
            output_dir=output_dir,
            num_classes=num_classes,
            normalized=normalized,
            image_size=image_size,
            lambda_piece=lambda_piece,
            lambda_perc=lambda_perc,
            lambda_latent=lambda_latent,
            lambda_image=lambda_image
        )

    print("\nAll counterfactuals have been processed!")


def main() -> None:
    """Command-line entry point for batch counterfactual generation using fine-tuned generators."""
    parser = argparse.ArgumentParser(
        description="Batch process counterfactual generation using fine-tuned generators."
    )
    parser.add_argument("--checkpoints_dir", type=str, required=True,
                        help="Directory containing fine-tuned GAN models (.pt).")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the '_cf_feature_vector.pickle' files (from PIECE).")
    parser.add_argument("--original_images_dir", type=str, required=True,
                        help="Directory containing the original images.")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["resnet50", "convnext"], help="CNN model architecture.")
    parser.add_argument("--model_weights", type=str, required=True,
                        help="Path to pretrained CNN model weights.")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for latent vector optimization.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs for latent vector optimization.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save final results.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes in the dataset.")
    parser.add_argument("--normalized", action="store_true",
                        help="Flag indicating if the CNN expects normalized inputs.")
    parser.add_argument("--lambda_piece", type=float, default=1.0,
                        help="Weight for the PIECE loss.")
    parser.add_argument("--lambda_perc", type=float, default=0.1,
                        help="Weight for the perceptual (LPIPS) loss.")
    parser.add_argument("--lambda_latent", type=float, default=1e-4,
                        help="Weight for latent vector regularization.")
    parser.add_argument("--lambda_image", type=float, default=1e-6,
                        help="Weight for image L1 regularization.")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for both the CNN and GAN.")

    args = parser.parse_args()

    # Save the config
    save_config(args, args.output_dir)

    # Perform the batch process
    batch_process_counterfactuals(
        checkpoints_dir=args.checkpoints_dir,
        input_dir=args.input_dir,
        original_images_dir=args.original_images_dir,
        model_type=args.model_type,
        model_weights=args.model_weights,
        lr=args.lr,
        epochs=args.epochs,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        normalized=args.normalized,
        image_size=args.image_size,
        lambda_piece=args.lambda_piece,
        lambda_perc=args.lambda_perc,
        lambda_latent=args.lambda_latent,
        lambda_image=args.lambda_image
    )


if __name__ == "__main__":
    main()
