import os
import h5py
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
import numpy as np
from torch.utils.data import random_split
import torch

def crop_image(image: np.ndarray, pix : int = 2) -> np.ndarray:
    """
    Params:
    image: np.ndarray
        An image with shape (68, 68, 3)

    pix: int
        Number of pixels to crop from the top and bottom of the image.

    Crops an image from (68, 68, 3) to (64, 64, 3) by removing
    two pixels from the top and bottom.

    Returns the cropped image if shape matches, otherwise None.
    """
    if image.shape == (68, 68, 3):
        return image[pix:-pix, pix:-pix, :]
    else:
        print(f"Skipping image with unexpected shape: {image.shape}")
        return None

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes an image within each channel.
    im = (im - im.min()) / (im.max() - im.min())
    """
    # Ensure the image is in float32 for precise calculations
    image = image.astype(np.float32)

    # Per-channel min-max normalization
    min_vals = image.min(axis=(0, 1), keepdims=True)  # Shape (1, 1, 3)
    max_vals = image.max(axis=(0, 1), keepdims=True)  # Shape (1, 1, 3)

    # Avoid division by zero by setting zero range to 1
    range_vals = np.maximum(max_vals - min_vals, 1e-5)

    # Normalize the image to [0, 1] per channel
    image_norm = (image - min_vals) / range_vals

    return image_norm

# Updated Dataset class for loading from an HDF5 file
class BBBC021Dataset(Dataset):
    def __init__(self, h5_file: str, transform=None, pix: int = 2):
        self.h5_file = h5_file
        self.transform = transform
        self.pix = pix
        
        # Open the HDF5 file and get the total number of images
        with h5py.File(self.h5_file, 'r') as h5f:
            self.num_images = h5f['images'].shape[0]

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_file, 'r') as h5f:
            image = h5f['images'][idx]
            image_name = h5f['image_names'][idx].decode('utf-8') 

        # Apply custom preprocessing
        image = crop_image(image, pix=self.pix)
        if image is None:
            return None, None
        
        image = normalize_image(image)

        # Make image into a channel-first tensor
        image = np.transpose(image, (2, 0, 1))

        # Apply optional transformations
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, image_name

# DataLoader setup
def load_data(cfg: DictConfig, split: str = 'train', seed: int = 42) -> DataLoader:
    """
    Loads dataset based on the specified type: 'train', 'val', or 'test'.
    
    Args:
        cfg (DictConfig): Configuration file with dataset paths and parameters.
        split (str): Specifies which dataset to load. Options: ['train', 'val', 'test'].
        seed (int): Random seed for dataset splitting.
        
    Returns:
        DataLoader: The requested DataLoader.
    """
    
    torch.manual_seed(seed)  # Set seed for reproducibility

    # Load the full dataset
    dataset = BBBC021Dataset(h5_file=cfg.data.train_path, pix=cfg.data.crop_pixels)

    # Compute dataset sizes
    total_size = len(dataset)
    train_size = int(cfg.data.train_ratio * total_size)
    val_size = int(cfg.data.val_ratio * total_size)
    test_size = total_size - train_size - val_size  # Remaining for test

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Return correct DataLoader
    if split == "train":
        selected_set = train_dataset
    elif split == "val":
        selected_set = val_dataset
    elif split == "test":
        selected_set = test_dataset
    else:
        raise ValueError(f"Invalid dataset split '{split}'. Choose from ['train', 'val', 'test'].")
    
    data_loader = DataLoader(selected_set, 
                             batch_size=cfg.train.batch_size, 
                             shuffle=(split == "train"), 
                             num_workers=4)
    
    return data_loader