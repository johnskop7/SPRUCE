import argparse
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import pickle
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from load_dataloaders import load_data
sys.path.append("/home/ubuntu/SPRUCE")
from models.ConvNeXt import CustomConvNeXt
from models.ResNet50 import CustomResNet50
from utils.model_utils import load_classifier_model

def save_activations(args):
    print("Loading data...")
    train_loader, val_loader, test_loader = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        normalized=True,
        dataset_type=args.dataset_type
    )
    print("Data loaded successfully.")

    # Initialize the model
    print("Loading the trained model's weights...")
    if args.model_arch == 'convnext':
        model = CustomConvNeXt(num_classes=args.num_classes)
    elif args.model_arch == 'resnet50':
        model = CustomResNet50(num_classes=args.num_classes)
    else:
        raise ValueError("Unsupported model architecture specified.")

    model_weights = args.model_weights
    state_dict = torch.load(model_weights, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),weights_only=True)
    model.load_state_dict(state_dict)

    print("Model weights loaded successfully.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("Saving activations...")
    # Dictionary to store activations, initialized empty for each class
    predicted_class_activations = {}

    with torch.no_grad():
        # Wrap train_loader with tqdm for a progress bar
        for inputs, _ in tqdm(train_loader, desc="Processing batches"):
            inputs = inputs.to(device)
            logits, features = model(inputs, return_features=True)
            probs = torch.softmax(logits, dim=1)
            _, predicted_labels = torch.max(probs, dim=1)

            for feature, predicted_label in zip(features, predicted_labels):
                predicted_label_item = predicted_label.item()
                if predicted_label_item not in predicted_class_activations:
                    predicted_class_activations[predicted_label_item] = []
                predicted_class_activations[predicted_label_item].append(feature.cpu().numpy())

    for class_id in predicted_class_activations.keys():
        predicted_class_activations[class_id] = np.stack(predicted_class_activations[class_id])

    save_path = args.save_path
    with open(save_path, 'wb') as handle:
        pickle.dump(predicted_class_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Store classifier's activations")

    # Dataset parameters
    parser.add_argument("--dataset_type", type=str, required=True,
                        help="Type of dataset (cardiomegaly, lung_opacity, oct, dementia)")
    parser.add_argument("--data_dir", type=str, required=True, help="Image directory")

    # Model parameters
    parser.add_argument("--model_arch", type=str, choices=['convnext', 'resnet50'], default='convnext',
                        help="Model architecture to use: 'convnext' or 'resnet50'")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes of the dataset")

    # Model weights' path
    parser.add_argument("--model_weights", type=str, required=True, help="The path where the model's weights are saved")

    # Batch and image size
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")

    # Save path
    parser.add_argument("--save_path", type=str, required=True, help="Path to save activations")

    args = parser.parse_args()

    save_activations(args)
