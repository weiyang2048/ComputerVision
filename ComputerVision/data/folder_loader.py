import os
from pathlib import Path

def get_image_paths_and_labels(base_dir):
    """
    Retrieves image file paths and their corresponding labels from a directory structure.

    Args:
        base_dir (str or Path): The base directory containing subdirectories for each class.

    Returns:
        tuple: A tuple containing:
            - image_paths (list of str): List of file paths to images.
            - labels (list of int): List of numeric labels corresponding to each image.
            - class_to_label (dict): A dictionary mapping class names to numeric labels.
    """
    # Initialize empty lists for paths and labels
    image_paths = []
    labels = []
    
    # Get all subdirectories in the base directory
    class_dirs = [d for d in Path(base_dir).iterdir() if d.is_dir()]
    
    # Create a dictionary to map folder names to numeric labels
    class_to_label = {folder.name: idx for idx, folder in enumerate(sorted(class_dirs))}
    
    # Iterate through each class directory
    for class_dir in class_dirs:
        class_label = class_to_label[class_dir.name]
        
        # Get all image files in the current class directory
        for img_path in class_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_paths.append(str(img_path))
                labels.append(class_label)
    
    return image_paths, labels, class_to_label

