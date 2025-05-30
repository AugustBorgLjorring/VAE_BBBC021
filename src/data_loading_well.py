import h5py
import torch
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from omegaconf import DictConfig
from torchvision import transforms
import torchvision.transforms.functional as TF

class BBBC021Dataset(Dataset):
    def __init__(self, h5_file: str, transform=None):
        self.h5_file = h5_file
        self.transform = transform
        self.h5f = None
        with h5py.File(self.h5_file, 'r') as f:
            self.num_images = f['images'].shape[0]
            self.image_names = [n.decode('utf-8') for n in f['image_names']]

    def __len__(self):
        return self.num_images

    def _lazy_open(self):
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_file, 'r')

    def __getitem__(self, idx: int):
        self._lazy_open()
        img = self.h5f['images'][idx]
        img = torch.from_numpy(img)  # already float32
        name = self.image_names[idx]

        # Apply transform if present
        if self.transform:
            img = self.transform(img)

        return img, name


class BBBC021Metadata(Dataset):
    def __init__(self, h5_file: str):
        self.h5_file = h5_file
        self.h5f = None
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
        comp = self.h5f['metadata_compound'][idx].decode('utf-8')
        conc = self.h5f['metadata_concentration'][idx].decode('utf-8')
        moa  = self.h5f['metadata_moa'][idx].decode('utf-8')
        return well, comp, conc, moa


def _get_wells(cfg: DictConfig):
    """Load the well ID for every image, in order."""
    with h5py.File(cfg.data.metadata_path, 'r') as f:
        wells = [w.decode('utf-8') for w in f['metadata_well'][:]]
    return wells


class RandomEightWay:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        flip  = random.choice([False, True])
        img = TF.rotate(img, angle)
        if flip:
            img = TF.hflip(img)
        return img


def load_data_by_well(cfg: DictConfig, split: str = 'train', seed: int = 42) -> DataLoader:
    """
    Splits images by well:
      - ~cfg.data.train_ratio of wells → train
      - ~cfg.data.val_ratio of wells   → val
      - remainder                       → test
    """
    # reproducibility
    torch.manual_seed(seed)
    random.seed(seed)

    # Define transforms only for training
    transform = None
    if split == "train":
        transform = transforms.Compose([RandomEightWay()])

    # full image dataset
    dataset = BBBC021Dataset(cfg.data.train_path, transform=transform)

    # map each well → list of image-indices
    wells = _get_wells(cfg)  

    print(f"Loading {split} split")
    print(f"Loaded {len(wells)} wells from {cfg.data.train_path}")

    well2idx = defaultdict(list)
    for idx, w in enumerate(wells):
        well2idx[w].append(idx)

    # shuffle and split wells
    all_wells = list(well2idx.keys())
    random.shuffle(all_wells)
    N = len(all_wells)
    n_train = int(cfg.data.train_ratio * N)
    n_val   = int(cfg.data.val_ratio * N)
    # leftover wells → test
    n_test  = N - n_train - n_val

    print(f"Splitting wells: train {n_train} ({n_train/N:.2%}), val {n_val} ({n_val/N:.2%}), test {n_test} ({n_test/N:.2%})")

    wells_train = all_wells[:n_train]
    wells_val   = all_wells[n_train:n_train + n_val]
    wells_test  = all_wells[n_train + n_val:]

    # flatten indices
    idx_train = [i for w in wells_train for i in well2idx[w]]
    idx_val   = [i for w in wells_val   for i in well2idx[w]]
    idx_test  = [i for w in wells_test  for i in well2idx[w]]

    print(f"Size of splits: train {len(idx_train)} ({len(idx_train)/len(dataset):.2%}), val {len(idx_val)} ({len(idx_val)/len(dataset):.2%}), test {len(idx_test)} ({len(idx_test)/len(dataset):.2%})")

    subsets = {
        'train': Subset(dataset, idx_train),
        'val':   Subset(dataset, idx_val),
        'test':  Subset(dataset, idx_test),
        'all':   dataset
    }

    chosen = subsets.get(split)
    if chosen is None:
        raise ValueError(f"Invalid split '{split}'. Choose from train/val/test/all.")

    loader = DataLoader(
        chosen,
        batch_size=cfg.train.batch_size,
        shuffle=(split == 'train'),
        drop_last=(split == 'train'),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        num_workers=4
    )
    return loader


def load_metadata(cfg: DictConfig) -> DataLoader:
    """Loads the metadata (well, compound, conc, moa) in the same order."""
    ds = BBBC021Metadata(cfg.data.metadata_path)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)
