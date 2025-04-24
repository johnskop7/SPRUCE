import os
import pickle
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import cv2
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings

def load_image(image_path):
    """Loads an image from a file and converts it to RGB format."""
    return np.array(Image.open(image_path).convert("RGB"))

def compute_difference_image(img1, img2):
    """
    Computes the absolute difference between two images (both assumed to be RGB)
    and returns a single-channel (grayscale) difference image so colormaps can apply.
    """
    # 3-channel absolute difference
    diff_3ch = np.abs(img1.astype(np.int16) - img2.astype(np.int16)).astype(np.uint8)
    # Convert to grayscale for colormap
    diff_gray = cv2.cvtColor(diff_3ch, cv2.COLOR_RGB2GRAY)
    return diff_gray

def plot_images(images, titles, save_path=None, cmaps=None):
    """Plots multiple images side by side with their respective titles."""
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))

    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        cmap = cmaps[i] if cmaps else None  # Apply colormap only to specific images
        ax.imshow(img, cmap=cmap)  # If 'img' is 3-channel, cmap is ignored
        ax.axis('off')
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Comparison image saved to {save_path}")
    plt.show()

def save_tensor_as_png(image_tensor, filename):
    """Saves a PyTorch tensor as a PNG image."""
    image_tensor = image_tensor.squeeze(0).detach().cpu()
    if image_tensor.min().item() < 0:
        image_tensor = (image_tensor + 1) / 2.0
    image_tensor = torch.clamp(image_tensor, 0, 1)
    image_pil = TF.to_pil_image(image_tensor)
    image_pil.save(filename)

def save_latent_as_npz(latent_vector_path, npz_path):
    """Converts and saves a PyTorch latent vector tensor to a NumPy .npz file."""
    latent_vector = torch.load(latent_vector_path)
    if isinstance(latent_vector, torch.Tensor):
        np.savez(npz_path, w=latent_vector.detach().cpu().numpy())
        print(f"Latent vector saved to {npz_path}")


def save_inverted_image(image, output_path):
    """Saves an image as a PNG file."""
    image_pil = Image.fromarray(image)
    image_pil.save(output_path)
    print(f"Inverted image saved to {output_path}")



def load_generators(old_generator_path, new_generator_path):
    """Loads the old and new generators."""
    try:
        with open(old_generator_path, 'rb') as f:
            old_G = pickle.load(f)['G_ema']
        with open(new_generator_path, 'rb') as f_new:
            new_G = torch.load(f_new, map_location="cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            old_G, new_G = old_G.cuda(), new_G.cuda()
        return old_G, new_G
    except Exception as e:
        print(f"Error loading generators: {e}")
        return None, None

def process_all_generators(checkpoints_dir, embeddings_dir, old_generator_path, output_dir,
                           original_images_dir):
    """Processes generators and computes differences with original images."""
    os.makedirs(output_dir, exist_ok=True)
    comparison_output_dir = os.path.join(output_dir, "comparisons")
    npz_output_dir = os.path.join(output_dir, "npz")
    inverted_images_output_dir = os.path.join(output_dir, "inverted_images")
    os.makedirs(comparison_output_dir, exist_ok=True)
    os.makedirs(npz_output_dir, exist_ok=True)
    os.makedirs(inverted_images_output_dir, exist_ok=True)


    generator_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]

    for generator_file in generator_files:
        parts = generator_file.split("_")
        if len(parts) < 3:
            print(f"Skipping invalid filename: {generator_file}")
            continue

        image_name = "_".join(parts[2:])[:-3]  # Remove ".pt" extension
        new_generator_path = os.path.join(checkpoints_dir, generator_file)
        latent_vector_path = os.path.join(embeddings_dir, image_name, "0.pt")
        original_image_path = os.path.join(original_images_dir, image_name + ".png")
        npz_output_path = os.path.join(npz_output_dir, f"{image_name}.npz")
        inverted_image_output_path = os.path.join(inverted_images_output_dir, f"{image_name}.png")

        if not os.path.exists(latent_vector_path) or not os.path.exists(original_image_path):
            print(f"Skipping {image_name} - Missing latent vector or original image!")
            continue

        old_G, new_G = load_generators(old_generator_path, new_generator_path)
        if old_G is None or new_G is None:
            continue

        try:
            w_pivot = torch.load(latent_vector_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
            save_latent_as_npz(latent_vector_path, npz_output_path)

            # Synthesize images
            old_image = old_G.synthesis(w_pivot, noise_mode='const', force_fp32=True)
            new_image = new_G.synthesis(w_pivot, noise_mode='const', force_fp32=True)

            # Convert tensor (B, C, H, W) to NumPy (H, W, C)
            old_image = (old_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
            new_image = (new_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
            original_image = load_image(original_image_path)

            # Differences in grayscale so that colormaps can apply
            old_diff = compute_difference_image(original_image, old_image)
            new_diff = compute_difference_image(original_image, new_image)

            # Plot and save
            comparison_output_path = os.path.join(comparison_output_dir, f"{image_name}_comparison.png")
            save_inverted_image(new_image, inverted_image_output_path)
            plot_images(
                [original_image, old_image, old_diff, new_image, new_diff],
                [
                    "Original Image",
                    "Before Pivotal Tuning",
                    "Difference",
                    "After Pivotal Tuning",
                    "Final Difference"
                ],
                save_path=comparison_output_path,
                cmaps=[None, None, 'magma', None, 'magma']
            )

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process all PTI fine-tuned generators.")
    parser.add_argument("--checkpoints_dir", type=str, required=True,
                        help="Path to the directory containing the fine-tuned generators for each image.")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Path to the embeddings directory where the pivotal latent vectors from PTI are stored.")
    parser.add_argument("--original_gan", type=str, required=True,
                        help="Path to the weights of the original GAN model before PTI tuning.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where results (images and latent vectors) will be saved.")
    parser.add_argument("--original_images_dir", type=str, required=True,
                        help="Path to the directory containing original images.")

    args = parser.parse_args()
    process_all_generators(
        checkpoints_dir=args.checkpoints_dir,
        embeddings_dir=args.embeddings_dir,
        old_generator_path=args.original_gan,
        output_dir=args.output_dir,
        original_images_dir=args.original_images_dir
    )
