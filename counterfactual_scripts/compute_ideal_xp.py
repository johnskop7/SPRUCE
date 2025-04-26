#!/usr/bin/env python3
import os
import sys
import pickle
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
# ignore numpy.core.numeric deprecation
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"
from utils.cnn_models import load_cnn_model
from utils.piece_utils import (
    precompute_gmms_for_class,
    acquire_feature_probabilities,
    filter_df_of_exceptional_noise,
    modifying_exceptional_features,
    load_reconstructed_image
)

sys.path.append("/home/ubuntu/PTI")  # adjust or remove as needed

def apply_piece_to_image(
    gan_model_path: str,
    latent_vector_path: str,
    counterfactual_class: int,
    classifier: nn.Module,
    alpha: float,
    output_pickle_path: str,
    normalized: bool,
    precomputed_gmms: list,
    classifier_type: str,
    device: torch.device
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
    #print(f"\n▶ Processing {latent_vector_path} with {gan_model_path}")

    #Load the finetuned generator
    if gan_model_path.endswith(".pkl"):
        with open(gan_model_path, "rb") as f:
            gan = pickle.load(f)["G_ema"].to(device)
    elif gan_model_path.endswith(".pt"):
        gan = torch.load(gan_model_path, map_location=device,weights_only=False).to(device)
    else:
        raise ValueError("GAN file must be .pkl or .pt")

    #Load the pivot latent vector
    w_pivot = torch.load(latent_vector_path,map_location="cuda" if torch.cuda.is_available() else "cpu",weights_only=False)

    #Load the reconstructed version of the query image and print the respective class probabilities of the classifier
    reconstructed_img = load_reconstructed_image(
        w_pivot, classifier, device, gan, normalized
    )

    #Acquire Exceptional Features for the query image based on the PIECE methodology
    df = acquire_feature_probabilities(
        target_class=counterfactual_class,
        classifier=classifier,
        original_query_img=reconstructed_img,
        alpha=alpha,
        precomputed_gmms=precomputed_gmms,
        classifier_type=classifier_type
    )

    #Filter those Exceptional Features based on the PIECE methodology
    df = filter_df_of_exceptional_noise(
        df, counterfactual_class, classifier,
        alpha=alpha, classifier_type=classifier_type
    ).sort_values("Probability of Event")

    #print(f"  • Features to change: {len(df)}")

    #Change the Exceptional to the Expected and create the counterfactual feature vector
    _, feats = classifier(reconstructed_img, return_features=True)
    query_x = feats[0]
    cf_x = modifying_exceptional_features(df, counterfactual_class, query_x)
    cf_x = cf_x.clone().detach().float().requires_grad_(False)

    #Save the counterfactual feature vector along with the respective pivotal latent vector
    with open(output_pickle_path, "wb") as f:
        pickle.dump({"cf_feature_vector": cf_x, "ws": w_pivot}, f)
    #print(f"  ✓ Saved CF vector → {output_pickle_path}")
    #print("---------------------------------------------------------------------------------")

    return len(df)


def batch_process_piece(
    checkpoints_dir: str,
    embeddings_dir: str,
    classifier_type: str,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)
    feat_dir = os.path.join(output_dir, "feature_vectors")
    os.makedirs(feat_dir, exist_ok=True)
    summary_txt = os.path.join(output_dir, "features_changed.txt")

    # load class activations & precompute GMMs
    print("Loading class activations...")
    with open(class_activations, "rb") as f:
        dist_data = pickle.load(f)

    gmms = precompute_gmms_for_class(
        dist_data=dist_data,
        class_idx=counterfactual_class,
        classifier_type=classifier_type,
        n_components=n_components,
        alpha=alpha
    )

    print("Loading classifier model...")
    classifier = load_cnn_model(classifier_type=classifier_type,
                         model_weights=model_weights,
                         num_classes=num_classes).to(device)

    gens = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]

    with open(summary_txt, "w") as log:
        for gf in tqdm(gens, desc="Computing counterfactual feature vectors for all query images", unit="gen"):
            parts = gf.split("_")
            if len(parts) < 3:
                print(f"Skipping invalid name: {gf}")
                continue

            image_name = "_".join(parts[2:])[:-3]
            gan_path = os.path.join(checkpoints_dir, gf)
            latent_vector_path = os.path.join(embeddings_dir, image_name, "0.pt")
            out_pkl  = os.path.join(feat_dir, f"{image_name}_cf_feature_vector.pickle")

            if not os.path.exists(latent_vector_path):
                print(f"  ✗ Missing latent vector path for {image_name}, skipping.")
                continue

            k = apply_piece_to_image(
                gan_model_path=gan_path,
                latent_vector_path=latent_vector_path,
                classifier=classifier,
                counterfactual_class=counterfactual_class,
                alpha=alpha,
                output_pickle_path=out_pkl,
                normalized=normalized,
                precomputed_gmms=gmms,
                classifier_type=classifier_type,
                device=device
            )

            #log.write(f"{image_name}: {k} features changed\n")

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
        embeddings_dir      = gan_cfg["embeddings_dir"],
        classifier_type      = cnn_cfg["type"],
        model_weights        = cnn_cfg["weights"],
        counterfactual_class = piece_cfg["counterfactual_class"],
        class_activations    = piece_cfg["class_activations"],
        alpha                = piece_cfg["alpha"],
        output_dir           = gan_cfg["output_dir"],
        num_classes          = cnn_cfg["num_classes"],
        normalized           = cnn_cfg["normalized"],
        n_components         = piece_cfg["n_components"]
    )
