import os
import logging
import pandas as pd
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from pydicom.pixels import apply_voi_lut

logging.basicConfig(level=logging.INFO)


def load_data(data_dir, batch_size, image_size, normalized=True, dataset_type='oct', num_workers=2):
    """
    Creates and returns train, validation, and test DataLoaders with specified class mappings.

    """

    # Define transformations based on normalization flag
    if normalized:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05),
                shear=5
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        val_test_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        train_transform = train_transforms
        eval_transform = val_test_transforms

    else:
        simple_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        train_transform = simple_transform
        eval_transform = simple_transform

    # Create datasets using ImageFolder for each split
    image_datasets = {}
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        # Use train transforms for training set and eval transforms for others
        transform = train_transform if split == 'train' else eval_transform
        image_datasets[split] = datasets.ImageFolder(split_dir, transform=transform)

    # Define class mapping based on dataset_type
    mapping = None
    if dataset_type == 'cardiomegaly':
        mapping = {'Cardiomegaly': 0, 'Normal': 1}
    elif dataset_type == 'lung_opacity':
        mapping = {'Lung_Opacity': 0, 'Normal': 1}
    # elif dataset_type == 'nodule':
    #     mapping = {'Nodule': 0, 'Normal': 1}
    # elif dataset_type == "mass":
    #     mapping = {'Mass': 0, 'Normal': 1}
    elif dataset_type == 'oct':
        mapping = {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3}
    elif dataset_type == 'dementia':
        mapping = {'Mild Impairment': 0, 'Moderate Impairment':1,'No Impairment': 2, 'Very Mild Impairment':3}
    # elif dataset_type == 'diabetic_retinopathy':
    #     mapping = {  "Healthy" : 0, "Unhealthy" : 1 }
    # elif dataset_type == 'brain_tumor':
    #     mapping = {'glioma': 0, 'meningioma': 1 , 'notumor': 2 , 'pituitary' : 3}
    if mapping is not None:
        for split in image_datasets:
            current_classes = set(image_datasets[split].classes)
            expected_classes = set(mapping.keys())
            if expected_classes.issubset(current_classes):
                image_datasets[split].class_to_idx = mapping
                image_datasets[split].classes = list(mapping.keys())
            else:
                print(f"Warning: Expected classes {expected_classes} not found in {split} split.")

    # Create dataloaders
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        shuffle = True if split == 'train' else False
        dataloaders[split] = DataLoader(image_datasets[split], batch_size=batch_size,
                                        shuffle=shuffle, num_workers=num_workers)

    return dataloaders['train'], dataloaders['val'], dataloaders['test']
