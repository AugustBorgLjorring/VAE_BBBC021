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


def _get_wells_and_compounds(cfg: DictConfig):
    """Load the well ID and compound for every image, in order."""
    with h5py.File(cfg.data.metadata_path, 'r') as f:
        wells_meta = [w.decode('utf-8') for w in f['metadata_well'][:]]
        compounds_meta = [c.decode('utf-8') for c in f['metadata_compound'][:]]
    return wells_meta, compounds_meta


class RandomEightWay:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        flip  = random.choice([False, True])
        img = TF.rotate(img, angle)
        if flip:
            img = TF.hflip(img)
        return img


def seeded_split_wells(cfg: DictConfig, dataset: Dataset, split: str = 'train', seed: int = 1):
    # 2) Read metadata: well → compound
    wells_meta, compounds_meta = _get_wells_and_compounds(cfg)

    compound_to_wells = defaultdict(list)
    for well, comp in zip(wells_meta, compounds_meta):
        # only append if we haven’t seen this well for this compound yet
        if well not in compound_to_wells[comp]:
            compound_to_wells[comp].append(well)

    # 3) Prepare mutable copy and tracking
    comp_wells = {c: list(ws) for c, ws in compound_to_wells.items()}
    split_wells = {'train': [], 'val': [], 'test': []}
    val_count = test_count = 0
    val_full = test_full = False

    total_wells = sum(len(ws) for ws in comp_wells.values())
    val_target  = int(round(cfg.data.val_ratio  * total_wells))
    test_target = int(round(cfg.data.test_ratio * total_wells))

    # 4) Seeding step: one well per compound in each split
    random.seed(seed)
    for comp in random.sample(list(comp_wells), len(comp_wells)):
        wells = comp_wells[comp]
        random.shuffle(wells)
        if wells:
            split_wells['val'].append(wells.pop(0))
            val_count += 1
        if wells:
            split_wells['test'].append(wells.pop(0))
            test_count += 1
        if wells:
            split_wells['train'].append(wells.pop(0))

    # 5) Collect & shuffle remaining wells
    remaining = [w for ws in comp_wells.values() for w in ws]
    random.shuffle(remaining)

    # 6) Fill val/test to target, rest to train
    for well in remaining:
        if not val_full and val_count < val_target:
            split_wells['val'].append(well)
            val_count += 1
            if val_count >= val_target:
                val_full = True
        elif not test_full and test_count < test_target:
            split_wells['test'].append(well)
            test_count += 1
            if test_count >= test_target:
                test_full = True
        else:
            split_wells['train'].append(well)

    # 7) Map wells → dataset indices
    well2idx = defaultdict(list)
    for idx, w in enumerate(wells_meta):
        well2idx[w].append(idx)

    indices = []
    for w in split_wells[split]:
        indices.extend(well2idx[w])

    print(f"Total wells Train: {len(split_wells['train'])}, Val: {len(split_wells['val'])}, Test: {len(split_wells['test'])}")
    print(f"Total images in {split}: {len(indices)} ({len(indices) / len(dataset):.2%} of total)")

    return indices


def load_data_by_well(cfg: DictConfig, split: str = 'train', seed: int = 1) -> DataLoader:
    """
    Splits wells by compound:
      1) Seed val/test/train with one well per compound
      2) Randomly fill val/test until their targets (cfg.data.val_ratio/test_ratio)
      3) Rest of wells → train
    """
    # Build dataset & transform
    transform = transforms.Compose([RandomEightWay()]) if split == "train" else None
    dataset = BBBC021Dataset(cfg.data.train_path, transform=transform)

    if split != 'all':
        indices = seeded_split_wells(cfg, dataset, split, seed)
        subset = Subset(dataset, indices)
    else:
        subset = dataset

    # Return DataLoader
    return DataLoader(
        subset,
        batch_size=cfg.train.batch_size,
        shuffle=(split == 'train'),
        drop_last=(split == 'train'),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        num_workers=4
    )

def load_metadata(cfg: DictConfig) -> DataLoader:
    """Loads the metadata (well, compound, conc, moa) in the same order."""
    ds = BBBC021Metadata(cfg.data.metadata_path)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)
