import os
import logging
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from .transforms import get_transforms


def load_dataloaders(
        data_dir: str,
        image_size: int,
        batch_size: int,
        dataset_type: str,
        normalized: bool = True,
        num_workers: int = 2
    ):
    """
    Builds and returns train/validation/test DataLoaders.

    Args:
        data_dir: root directory containing `train/`, `val/`, `test/` subfolders.
        image_size: resize dimension for images.
        batch_size: number of samples per batch.
        dataset_type: name of the dataset for class-to-index mapping.
        normalized: whether to apply normalization in transforms.
        num_workers: number of worker processes for data loading.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create transforms
    train_transform, eval_transform = get_transforms(image_size, normalized)

    # Load datasets for each split
    image_datasets = {}
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        transform = train_transform if split == 'train' else eval_transform
        image_datasets[split] = ImageFolder(split_dir, transform=transform)

    # Define class mappings
    mapping_dict = {
        'cardiomegaly': {'Cardiomegaly': 0, 'Normal': 1},
        'lung_opacity': {'Lung_Opacity': 0, 'Normal': 1},
        'oct': {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3},
        'dementia': {'Mild Impairment': 0,
                     'Moderate Impairment': 1,
                     'No Impairment': 2,
                     'Very Mild Impairment': 3},
        'pleural_effusion': {'Pleural Effusion': 0, 'Normal': 1}
    }
    mapping = mapping_dict.get(dataset_type)

    # Apply mapping to dataset objects
    if mapping:
        for split, ds in image_datasets.items():
            expected = set(mapping.keys())
            found = set(ds.classes)
            if not expected.issubset(found):
                logging.warning(
                    f"Expected classes {expected} not all found in {split} folder of {data_dir}"
                )
            else:
                ds.class_to_idx = mapping
                ds.classes = list(mapping.keys())

    # Build DataLoaders
    loaders = {}
    for split, ds in image_datasets.items():
        shuffle = (split == 'train')
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    return loaders['train'], loaders['val'], loaders['test']