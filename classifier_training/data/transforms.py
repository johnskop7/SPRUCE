from torchvision import transforms


def get_transforms(image_size: int, normalized: bool = True):
    """
    Returns train and evaluation transforms for input images.

    Args:
        image_size: target height and width of the images.
        normalized: whether to apply ImageNet normalization.

    Returns:
        train_transform, eval_transform: torchvision transform pipelines.
    """
    if normalized:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomAffine(degrees=10,
                                     translate=(0.02, 0.02),
                                     scale=(0.95, 1.05),
                                     shear=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        simple = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        train_transform = simple
        eval_transform = simple

    return train_transform, eval_transform