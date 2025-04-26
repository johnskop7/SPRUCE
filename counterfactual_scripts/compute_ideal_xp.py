#!/usr/bin/env python3
import os
import sys
import pickle
import argparse
import yaml
import numpy as np
import torch

# allow importing from your project root
sys.path.append("/home/ubuntu/PTI")  # adjust or remove as needed

from cnn_models import load_cnn_model
from utils import generate_and_plot_synthetic_image
from piece_utils import (
    precompute_gmms_for_class,
    acquire_feature_probabilities,
    filter_df_of_exceptional_noise,
    modifying_exceptional_features
)


# def load_npz_latent_vector(npz_path: str) -> torch.Tensor:
#     """
#     Load the latent vector from a .npz file and return it as a PyTorch tensor.
#     """
#     data = np.load(npz_path)
#     if "w" not in data:
#         raise ValueError(f"No 'w' key found in {npz_path}")
#     tensor = torch.tensor(data["w"], dtype=torch.float32)
#     if torch.cuda.is_available():
#         tensor = tensor.cuda()
#     return tensor


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
    precomputed_gmms: list,
    n_components: int = 5
) -> int:
    """
    Applies PIECE to one image:
      - loads GAN & latent
      - synthesizes initial image
      - acquires & filters feature probs
      - builds counterfactual feature vector
      - saves cf vector + ws
    Returns number of features changed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n▶ Processing {latent_vector_path} with {gan_model_path}")

    # 1) load classifier
    cnn = load_cnn_model(model_type=model_type,
                         model_weights=model_weights,
                         num_classes=num_classes).to(device)

    # 2) load GAN
    if gan_model_path.endswith(".pkl"):
        with open(gan_model_path, "rb") as f:
            gan = pickle.load(f)["G_ema"].to(device)
    elif gan_model_path.endswith(".pt"):
        gan = torch.load(gan_model_path, map_location=device).to(device)
    else:
        raise ValueError("GAN file must be .pkl or .pt")

    # 3) load pivot latent
    w_pivot = load_npz_latent_vector(latent_vector_path)

    # 4) synth + plot initial
    synth_img, ws = generate_and_plot_synthetic_image(
        w_pivot, cnn, device, gan, normalized
    )

    # 5) feature probabilities
    df = acquire_feature_probabilities(
        target_class=counterfactual_class,
        cnn=cnn,
        original_query_img=synth_img,
        alpha=alpha,
        precomputed_gmms=precomputed_gmms,
        model_type=model_type
    )

    # 6) filter
    df = filter_df_of_exceptional_noise(
        df, counterfactual_class, cnn,
        alpha=alpha, model_type=model_type
    ).sort_values("Probability of Event")

    print(f"  • Features to change: {len(df)}")

    # 7) build cf vector
    _, feats = cnn(synth_img, return_features=True)
    query_x = feats[0]
    cf_x = modifying_exceptional_features(df, counterfactual_class, query_x)
    cf_x = cf_x.clone().detach().float().requires_grad_(False)

    # 8) save
    with open(output_pickle_path, "wb") as f:
        pickle.dump({"cf_feature_vector": cf_x, "ws": ws}, f)
    print(f"  ✓ Saved CF vector → {output_pickle_path}")

    return len(df)


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
    Batch-run PIECE over all fine-tuned GANs in checkpoints_dir.
    Saves per-image CF vectors and a summary txt of features changed.
    """
    os.makedirs(output_dir, exist_ok=True)
    feat_dir = os.path.join(output_dir, "feature_vectors")
    os.makedirs(feat_dir, exist_ok=True)
    summary_txt = os.path.join(output_dir, "features_changed.txt")

    # load class activations & precompute GMMs
    print(f"Loading class activations: {class_activations}")
    with open(class_activations, "rb") as f:
        dist_data = pickle.load(f)

    gmms = precompute_gmms_for_class(
        dist_data=dist_data,
        class_idx=counterfactual_class,
        model_type=model_type,
        n_components=n_components,
        alpha=alpha
    )

    gens = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]

    with open(summary_txt, "w") as log:
        for gf in gens:
            parts = gf.split("_")
            if len(parts) < 3:
                print(f"Skipping invalid name: {gf}")
                continue

            name = "_".join(parts[2:])[:-3]
            gan_path = os.path.join(checkpoints_dir, gf)
            npz_path = os.path.join(npz_latents_dir, f"{name}.npz")
            out_pkl  = os.path.join(feat_dir, f"{name}_cf_feature_vector.pickle")

            if not os.path.exists(npz_path):
                print(f"  ✗ Missing latent .npz for {name}, skipping.")
                continue

            k = apply_piece_to_image(
                gan_model_path=gan_path,
                latent_vector_path=npz_path,
                model_type=model_type,
                model_weights=model_weights,
                counterfactual_class=counterfactual_class,
                class_activations=class_activations,
                alpha=alpha,
                output_pickle_path=out_pkl,
                num_classes=num_classes,
                normalized=normalized,
                precomputed_gmms=gmms,
                n_components=n_components
            )

            log.write(f"{name}: {k} features changed\n")

    print("\n✅ All done. Results in:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-process PIECE on multiple images via a YAML config."
    )
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to your config.yaml"
    )
    args = parser.parse_args()

    # load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    gan_cfg   = cfg["gan"]
    cnn_cfg   = cfg["cnn"]
    piece_cfg = cfg["piece"]

    batch_process_piece(
        checkpoints_dir      = gan_cfg["checkpoints_dir"],
        npz_latents_dir      = gan_cfg["embeddings_dir"],
        model_type           = cnn_cfg["type"],
        model_weights        = cnn_cfg["weights"],
        counterfactual_class = piece_cfg["counterfactual_class"],
        class_activations    = piece_cfg["class_activations"],
        alpha                = piece_cfg["alpha"],
        output_dir           = gan_cfg["output_dir"],
        num_classes          = cnn_cfg["num_classes"],
        normalized           = cnn_cfg["normalized"],
        n_components         = piece_cfg["n_components"]
    )
