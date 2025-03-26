from torchvision import transforms


def train_test_image_transformations(resize=1024):
    """
    Creates image transformations for training and testing datasets.

    Args:
        resize (int, optional): The size to which images will be resized. Default is 1024.

    Returns:
        tuple: A tuple containing:
            - train_transform (torchvision.transforms.Compose): Transformations for training images.
            - test_transform (torchvision.transforms.Compose): Transformations for testing images.
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop((resize, resize)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((0, 180)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, test_transform
