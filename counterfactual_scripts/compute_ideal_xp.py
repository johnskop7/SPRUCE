import os
import sys
import pickle
import argparse
import numpy as np
import torch

# Local imports
sys.path.append("/home/ubuntu/PTI")  # Adjust or remove as needed
from cnn_models import load_cnn_model
from utils import generate_and_plot_synthetic_image
from piece_utils import (
    precompute_gmms_for_class,
    acquire_feature_probabilities,
    filter_df_of_exceptional_noise,
    modifying_exceptional_features
)


def load_npz_latent_vector(npz_path: str) -> torch.Tensor:
    """
    Load the latent vector from a .npz file and return it as a PyTorch tensor.

    Args:
        npz_path (str): Path to the .npz file containing the latent vector.

    Returns:
        torch.Tensor: The loaded latent vector on CUDA if available.
    """
    latent_data = np.load(npz_path)
    if "w" not in latent_data:
        raise ValueError(f"No 'w' key found in {npz_path}. Check the file format.")
    latent_tensor = torch.tensor(latent_data["w"], dtype=torch.float32).cuda()
    return latent_tensor


def apply_piece_to_image(
        gan_model_path: str,
        latent_vector_path: str,
        model_type: str,
        model_weights: str,
        counterfactual_class: int,
        class_activations: str,
        alpha: float,
        output_pickle_path: str,
        num_classes: int,
        normalized: bool,
        precomputed_gmms: dict,
        n_components: int = 5
) -> int:
    """
    Apply the PIECE algorithm to a single image's latent representation and save the counterfactual feature vector.

    Args:
        gan_model_path (str): Path to the fine-tuned GAN model (.pkl or .pt).
        latent_vector_path (str): Path to the .npz file containing the latent vector.
        model_type (str): Type of CNN model ('resnet50', 'convnext', etc.).
        model_weights (str): Path to the pretrained weights for the CNN model.
        counterfactual_class (int): The class index to generate a counterfactual for.
        class_activations (str): Path to the class activation pickle file (used for GMM).
        alpha (float): Alpha parameter controlling how strictly to shift features.
        output_pickle_path (str): Path to save the computed counterfactual feature vector and ws.
        num_classes (int): Number of classes for the CNN model.
        normalized (bool): Whether images are normalized for the CNN model.
        precomputed_gmms (dict): Precomputed Gaussian Mixture Models for the target class.
        n_components (int): Number of GMM components.

    Returns:
        int: Number of features changed by the PIECE algorithm.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processing {latent_vector_path} with GAN model: {gan_model_path}")

    # 1. Load the CNN model
    cnn_model = load_cnn_model(
        model_type=model_type,
        model_weights=model_weights,
        num_classes=num_classes
    ).to(device)

    # 2. Load the fine-tuned GAN model
    if gan_model_path.endswith('.pkl'):
        with open(gan_model_path, 'rb') as f:
            gan = pickle.load(f)['G_ema'].cuda()
    elif gan_model_path.endswith('.pt'):
        with open(gan_model_path, 'rb') as f:
            gan = torch.load(f, weights_only=False).cuda()
    else:
        raise ValueError("Unsupported GAN file type. Must be .pkl or .pt.")

    # 3. Load the latent vector from .npz
    w_pivot = load_npz_latent_vector(latent_vector_path)

    # 4. Generate the initial synthetic image
    synthetic_image_initial, ws = generate_and_plot_synthetic_image(
        w_pivot,
        cnn_model,
        device,
        gan,
        normalized
    )

    # -------------------------------------------------------
    # 5. Acquire feature probabilities using PRECOMPUTED GMMs
    # -------------------------------------------------------
    df = acquire_feature_probabilities(
        target_class=counterfactual_class,
        cnn=cnn_model,
        original_query_img=synthetic_image_initial,
        alpha=alpha,
        precomputed_gmms=precomputed_gmms,
        model_type=model_type
    )

    # 6. Filter out features that help the classification in the counterfactual class
    df = filter_df_of_exceptional_noise(
        df,
        counterfactual_class,
        cnn_model,
        alpha=alpha,
        model_type=model_type
    )
    df = df.sort_values('Probability of Event')

    # 7. Count features that will be changed
    features_changed = df.shape[0]
    print(f"Features changed: {features_changed}")

    # 8. Get the penultimate-layer feature vector for this image
    _, penultimate_layer_features = cnn_model(
        synthetic_image_initial,
        return_features=True
    )
    query_x = penultimate_layer_features[0]

    # 9. Modify filtered 'exceptional' features by replacing with their expected values
    cf_feature_vector = modifying_exceptional_features(df, counterfactual_class, query_x)
    cf_feature_vector = cf_feature_vector.clone().detach().float().requires_grad_(False)

    # 10. Save the resulting vectors
    with open(output_pickle_path, 'wb') as f:
        pickle.dump({'cf_feature_vector': cf_feature_vector, 'ws': ws}, f)
    print(f"Saved cf_feature_vector and ws to {output_pickle_path}")

    return features_changed


def batch_process_piece(
        checkpoints_dir: str,
        npz_latents_dir: str,
        model_type: str,
        model_weights: str,
        counterfactual_class: int,
        class_activations: str,
        alpha: float,
        output_dir: str,
        num_classes: int,
        normalized: bool,
        n_components: int = 5
) -> None:
    """
    Batch-process the PIECE algorithm for all images in a directory.

    1) Precompute GMMs using the provided class activations.
    2) For each fine-tuned generator (.pt file):
       - Locate the corresponding latent vector.
       - Apply PIECE to generate a counterfactual feature vector.
       - Record how many features changed.

    Args:
        checkpoints_dir (str): Directory containing fine-tuned GAN generator files (.pt).
        npz_latents_dir (str): Directory containing .npz latent vectors.
        model_type (str): Model architecture for CNN (e.g. 'resnet50', 'convnext').
        model_weights (str): Path to the pretrained CNN weights.
        counterfactual_class (int): Class index to use for the counterfactual.
        class_activations (str): Path to the pickle file holding class activations.
        alpha (float): Alpha parameter for PIECE.
        output_dir (str): Directory to save the PIECE outputs.
        num_classes (int): Number of classes in the CNN model.
        normalized (bool): Flag indicating if images are normalized in the CNN.
        n_components (int): Number of GMM components.
    """
    # Make sure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    feature_vectors_dir = os.path.join(output_dir, "feature_vectors")
    txt_results_path = os.path.join(output_dir, "features_changed.txt")
    os.makedirs(feature_vectors_dir, exist_ok=True)

    # ------------------------------
    # 1) Load the class activations
    # ------------------------------
    print(f"Loading class activations from: {class_activations}")
    with open(class_activations, 'rb') as handle:
        dist_data = pickle.load(handle)

    # ------------------------------
    # 2) Precompute GMMs (only once)
    # ------------------------------
    precomputed_gmms = precompute_gmms_for_class(
        dist_data=dist_data,
        class_idx=counterfactual_class,
        model_type=model_type,
        n_components=n_components,
        alpha=alpha
    )

    # Find generator files in checkpoints_dir
    generator_files = [
        f for f in os.listdir(checkpoints_dir)
        if f.endswith(".pt")
    ]

    with open(txt_results_path, "w") as txt_file:
        for generator_file in generator_files:
            parts = generator_file.split("_")
            if len(parts) < 3:
                print(f"Skipping invalid filename: {generator_file}")
                continue

            # Construct image name by removing ".pt"
            image_name = "_".join(parts[2:])[:-3]
            gan_path = os.path.join(checkpoints_dir, generator_file)
            latent_path = os.path.join(npz_latents_dir, f"{image_name}.npz")

            if not os.path.exists(latent_path):
                print(f"Skipping {image_name} - latent vector not found!")
                continue

            # Prepare output path
            output_pickle_path = os.path.join(
                feature_vectors_dir,
                f"{image_name}_cf_feature_vector.pickle"
            )

            # ------------------
            # Apply PIECE
            # ------------------
            features_changed = apply_piece_to_image(
                gan_model_path=gan_path,
                latent_vector_path=latent_path,
                model_type=model_type,
                model_weights=model_weights,
                counterfactual_class=counterfactual_class,
                class_activations=class_activations,
                alpha=alpha,
                output_pickle_path=output_pickle_path,
                num_classes=num_classes,
                normalized=normalized,
                precomputed_gmms=precomputed_gmms,
                n_components=n_components
            )

            # Log how many features were changed
            txt_file.write(f"{image_name}: {features_changed} features changed\n")
            print(f"Processed {image_name}, {features_changed} features changed.")

    print("All images processed. Results saved!")


def main():
    parser = argparse.ArgumentParser(
        description="Use PIECE framework on multiple images and compute the respective counterfactual feature vectors."
    )
    parser.add_argument(
        "--checkpoints_dir", type=str, required=True,
        help="Directory containing fine-tuned GAN generators (PTI)."
    )
    parser.add_argument(
        "--npz_latents_dir", type=str, required=True,
        help="Directory containing .npz latent vectors."
    )
    parser.add_argument(
        "--model_type", type=str, required=True,
        choices=["resnet50", "convnext"],
        help="CNN model architecture."
    )
    parser.add_argument(
        "--model_weights", type=str, required=True,
        help="Path to pretrained CNN model weights."
    )
    parser.add_argument(
        "--counterfactual_class", type=int, required=True,
        help="Index of the counterfactual class."
    )
    parser.add_argument(
        "--class_activations", type=str, required=True,
        help="Path to the class activations .pkl file."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1,
        help="Alpha parameter for PIECE method."
    )
    parser.add_argument(
        "--n_components", type=int, default=5,
        help="Number of GMM components for feature modeling."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save counterfactual feature vectors and results."
    )
    parser.add_argument(
        "--num_classes", type=int, default=2,
        help="Number of classes in the dataset."
    )
    parser.add_argument(
        "--normalized", action="store_true",
        help="Flag indicating if images were normalized during classifier training."
    )

    args = parser.parse_args()

    batch_process_piece(
        checkpoints_dir=args.checkpoints_dir,
        npz_latents_dir=args.npz_latents_dir,
        model_type=args.model_type,
        model_weights=args.model_weights,
        counterfactual_class=args.counterfactual_class,
        class_activations=args.class_activations,
        alpha=args.alpha,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        normalized=args.normalized,
        n_components=args.n_components
    )


if __name__ == "__main__":
    main()
