import os
import pickle
import yaml
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import cv2
import matplotlib.pyplot as plt
import argparse
import warnings
import sys
import torch_utils
from utils.model_utils import load_generators
from utils.io_utils import compute_difference_image
warnings.filterwarnings("ignore")  # Suppress warnings


def save_inverted_image(image, output_path):
    """Saves an image (numpy array) as a PNG file."""
    image_pil = Image.fromarray(image)
    image_pil.save(output_path)
    print(f"Inverted image saved to {output_path}")


def plot_counterfactual_grid(orig, before, after, diff_before, diff_after, save_path=None):
    """
    Top row:    [ Original | Inversion | After PTI ]
    Bottom row: [ Black    | Δ(Inversion) | Δ(After PTI) ]
    Diffs are shown in 'magma' cmap.
    """
    # create figure + axes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # --- Top row ---
    titles_top = ["Original", "Before PTI", "After PTI"]
    for ax, img, title in zip(axes[0], (orig, before, after), titles_top):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    # --- Bottom row ---
    black = np.zeros_like(orig)
    bottom_imgs = [black, diff_before, diff_after]
    titles_bot  = ["", "Diff: Orig vs Before", "Diff: Orig vs After"]
    for ax, img, title in zip(axes[1], bottom_imgs, titles_bot):
        cmap = 'magma' if img is not black else 'gray'
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved comparison grid to {save_path}")
    plt.show()


def process_all_generators(checkpoints_dir, embeddings_dir, old_generator_path,
                           output_dir, original_images_dir):
    """Processes all fine-tuned generators: invert, compare and save results."""
    os.makedirs(output_dir, exist_ok=True)
    comparison_output_dir       = os.path.join(output_dir, "comparisons")
    inverted_images_output_dir  = os.path.join(output_dir, "inverted_images")
    os.makedirs(comparison_output_dir, exist_ok=True)
    os.makedirs(inverted_images_output_dir, exist_ok=True)

    generator_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]

    for generator_file in generator_files:
        parts = generator_file.split("_")
        if len(parts) < 3:
            print(f"Skipping invalid filename: {generator_file}")
            continue

        image_name              = "_".join(parts[2:])[:-3]  # strip ".pt"
        new_generator_path      = os.path.join(checkpoints_dir, generator_file)
        latent_vector_path      = os.path.join(embeddings_dir,image_name, "0.pt")
        original_image_path     = os.path.join(original_images_dir, f"{image_name}.png")
        inverted_image_output   = os.path.join(inverted_images_output_dir, f"{image_name}.png")

        if not os.path.exists(latent_vector_path) or not os.path.exists(original_image_path):
            print(f"Skipping {image_name} - missing latent or original image.")
            continue

        old_G, new_G = load_generators(old_generator_path, new_generator_path)
        if old_G is None or new_G is None:
            continue

        try:
            # Load pivot latent and synthesize
            w_pivot = torch.load(latent_vector_path,
                                 map_location="cuda" if torch.cuda.is_available() else "cpu")
            old_img_tensor = old_G.synthesis(w_pivot, noise_mode='const', force_fp32=True)
            new_img_tensor = new_G.synthesis(w_pivot, noise_mode='const', force_fp32=True)

            # Convert to HWC uint8
            old_image = (old_img_tensor.permute(0,2,3,1) * 127.5 + 128)\
                            .clamp(0,255).to(torch.uint8).cpu().numpy()[0]
            new_image = (new_img_tensor.permute(0,2,3,1) * 127.5 + 128)\
                            .clamp(0,255).to(torch.uint8).cpu().numpy()[0]
            original_image = np.array(Image.open(original_image_path).convert("RGB"))

            # Compute diffs
            old_diff = compute_difference_image(original_image, old_image)
            new_diff = compute_difference_image(original_image, new_image)

            # Save inverted image (new)
            save_inverted_image(new_image, inverted_image_output)

            comparison_path = os.path.join(comparison_output_dir,
                                           f"{image_name}_comparison.png")
            plot_counterfactual_grid(
                orig=original_image,
                before=old_image,
                after=new_image,
                diff_before=old_diff,
                diff_after=new_diff,
                save_path=comparison_path
            )

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process all PTI fine-tuned generators via a config file."
    )
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to your central config.yaml"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as cf:
        cfg = yaml.safe_load(cf)

    gan_cfg = cfg.get('gan', {})
    process_all_generators(
        checkpoints_dir=     gan_cfg['checkpoints_dir'],
        embeddings_dir=      gan_cfg['embeddings_dir'],
        old_generator_path=  gan_cfg['original_weights'],
        output_dir=          gan_cfg['output_dir'],
        original_images_dir= gan_cfg['original_images_dir']
    )
