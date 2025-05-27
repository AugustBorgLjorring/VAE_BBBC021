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
        self.h5f = None  # will be lazily opened in each worker

        # Load metadata
        with h5py.File(self.h5_file, 'r') as f:
            self.num_images = f['images'].shape[0]
            self.image_names = [n.decode('utf-8') for n in f['image_names']] # Load names in advance (49 MB)

    def __len__(self):
        return self.num_images

    # Lazily opens HDF5 file for safe multi-worker DataLoader use.
    def _lazy_open(self):
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_file, 'r')

    def __getitem__(self, idx: int):
        self._lazy_open()

        image = self.h5f['images'][idx] # Do not load the entire dataset into memory (22.5 GB)
        image_name = self.image_names[idx]

        # Convert image to torch tensor
        image = torch.from_numpy(image) # dtype is float32

        return image, image_name

class BBBC021Metadata(Dataset):
    def __init__(self, h5_file: str):
        self.h5_file = h5_file
        self.h5f = None  # will be lazily opened in each worker

        # Load metadata
        with h5py.File(self.h5_file, 'r') as f:
            self.length = f['metadata_well'].shape[0]

    def __len__(self):
        return self.length

    def _lazy_open(self):
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_file, 'r')

    def __getitem__(self, idx: int):
        self._lazy_open()

        well = self.h5f['metadata_well'][idx].decode('utf-8')
        compound = self.h5f['metadata_compound'][idx].decode('utf-8')
        concentration = self.h5f['metadata_concentration'][idx].decode('utf-8')
        moa = self.h5f['metadata_moa'][idx].decode('utf-8')

        return well, compound, concentration, moa

# DataLoader setup
def load_data(cfg: DictConfig, split: str = 'train', seed: int = 42, num_workers: int = 4, persistent_workers: bool = True, prefetch_factor: int = 4) -> DataLoader:
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
    elif split == "all":
        selected_set = dataset
    else:
        raise ValueError(f"Invalid dataset split '{split}'. Choose from ['train', 'val', 'test', 'all'].")

    data_loader = DataLoader(selected_set,
                             batch_size=cfg.train.batch_size,
                             shuffle=(split == "train"),
                             drop_last=(split == "train"),
                             pin_memory=True,
                             persistent_workers=persistent_workers,
                             prefetch_factor=prefetch_factor,
                             num_workers=num_workers)
    
    return data_loader


def load_metadata(cfg: DictConfig) -> DataLoader:
    dataset = BBBC021Metadata(h5_file=cfg.data.metadata_path)

    data_loader = DataLoader(dataset,
                             batch_size=cfg.train.batch_size,
                             shuffle=False)

    return data_loader