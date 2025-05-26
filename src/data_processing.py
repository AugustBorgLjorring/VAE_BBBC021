import os
import h5py
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
import numpy as np
from torch.utils.data import random_split
import torch

# Updated Dataset class for loading from an HDF5 file
class BBBC021Dataset(Dataset):
    def __init__(self, h5_file: str):
        self.h5_file = h5_file
        
        # Open the HDF5 file and get the total number of images
        with h5py.File(self.h5_file, 'r') as h5f:
            self.num_images = h5f['images'].shape[0]

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_file, 'r') as h5f:
            image = h5f['images'][idx]
            image_name = h5f['image_names'][idx].decode('utf-8') 
        
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
    dataset = BBBC021Dataset(h5_file=cfg.data.train_path)

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
