from torch.utils.data import Dataset
from PIL import Image
import torch

class ImageLoader(Dataset):
    """
    A custom Dataset class for loading images and their corresponding labels.

    Attributes:
        image_path (list of str): List of file paths to images.
        image_class (list of int): List of numeric labels corresponding to each image.
        transform (callable, optional): A function/transform to apply to the images.

    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(item): Retrieves the image and label at the specified index.
        collate_fn(batch): A static method to collate a batch of data.
    """
    def __init__(self, image_path, image_labels, transform=None) -> None:
        """
        Initializes the ImageLoader with image paths, labels, and an optional transform.

        Args:
            image_path (list of str): List of file paths to images.
            image_labels (list of int): List of numeric labels corresponding to each image.
            transform (callable, optional): A function/transform to apply to the images.
        """
        super().__init__()
        self.image_path = image_path
        self.image_class = image_labels
        self.transform = transform

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: The number of images.
        """
        return len(self.image_path)

    def __getitem__(self, item):
        """
        Retrieves the image and label at the specified index.

        Args:
            item (int): The index of the image to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and its label.
        """
        img = Image.open(self.image_path[item]).convert("RGB")
        label = self.image_class[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        """
        A static method to collate a batch of data.

        Args:
            batch (list of tuples): A list of tuples where each tuple contains an image and its label.

        Returns:
            tuple: A tuple containing a batch of images and a batch of labels.
        """
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
