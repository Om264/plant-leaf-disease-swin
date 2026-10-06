"""Dataset loading, splitting, transforms, and distributed-aware data loaders."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import datasets, transforms

TRAIN_NAMES = ["train", "training"]
VAL_NAMES = ["val", "valid", "validation"]
TEST_NAMES = ["test", "testing"]


class TransformSubset(Dataset):
    """List of (path, label) samples with a (late-bindable) transform."""

    def __init__(self, samples: Sequence[Tuple[str, int]], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is None:
            raise RuntimeError("Dataset transform not set.")
        return self.transform(img), label


@dataclass
class DatasetBundle:
    train: TransformSubset
    val: TransformSubset
    test: TransformSubset
    class_names: List[str]
    train_labels: List[int]

    def set_transforms(self, train_tf, eval_tf):
        self.train.transform = train_tf
        self.val.transform = eval_tf
        self.test.transform = eval_tf


def build_transforms(cfg: dict):
    """cfg = {"img_size": int, "mean": [...], "std": [...]}.

    RandAugment + random-erasing follow the Swin-Transformer paper's data
    augmentation choices (Liu et al., ICCV 2021, Sec. 4 / official config):
    https://arxiv.org/abs/2103.14030
    """
    size, mean, std = cfg["img_size"], cfg["mean"], cfg["std"]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),
    ])
    return train_tf, build_eval_transform(cfg)


def build_eval_transform(cfg: dict):
    return transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])


def _find_dir(root: Path, names) -> Optional[Path]:
    for n in names:
        if (root / n).is_dir():
            return root / n
    return None


def _samples_for_classes(folder: Path, class_names: List[str]):
    ds = datasets.ImageFolder(str(folder))
    unknown = set(ds.classes) - set(class_names)
    if unknown:
        raise ValueError(f"{folder} has classes not present in training set: {sorted(unknown)}")
    remap = {i: class_names.index(c) for i, c in enumerate(ds.classes)}
    return [(p, remap[l]) for p, l in ds.samples]


def make_eval_dataset(folder, class_names, transform) -> TransformSubset:
    return TransformSubset(_samples_for_classes(Path(folder), class_names), transform)


def build_datasets(data_dir, val_size=0.15, test_size=0.15, seed=42) -> DatasetBundle:
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data_dir not found: {root}")

    train_dir = _find_dir(root, TRAIN_NAMES)
    val_dir = _find_dir(root, VAL_NAMES)
    test_dir = _find_dir(root, TEST_NAMES)

    if train_dir is not None:
        base = datasets.ImageFolder(str(train_dir))
        class_names = base.classes
        train_samples = list(base.samples)
        if val_dir is not None:
            val_samples = _samples_for_classes(val_dir, class_names)
        else:
            print(f"No validation folder - taking {val_size:.0%} of train as validation.")
            tr_idx, va_idx = train_test_split(
                np.arange(len(train_samples)), test_size=val_size,
                stratify=[s[1] for s in train_samples], random_state=seed)
            val_samples = [train_samples[i] for i in va_idx]
            train_samples = [train_samples[i] for i in tr_idx]
        if test_dir is not None:
            test_samples = _samples_for_classes(test_dir, class_names)
        else:
            print("WARNING: no test folder - the validation set is used as test set.")
            test_samples = val_samples
        return DatasetBundle(TransformSubset(train_samples), TransformSubset(val_samples),
                             TransformSubset(test_samples), class_names,
                             [s[1] for s in train_samples])

    print("No train/val/test folders found - creating a stratified split. "
          "(If images of the same leaf/plant exist, split by group first: scripts/split_dataset.py)")
    base = datasets.ImageFolder(str(root))
    samples, labels = base.samples, [s[1] for s in base.samples]
    idx = np.arange(len(samples))
    idx_tv, idx_te = train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)
    idx_tr, idx_va = train_test_split(
        idx_tv, test_size=val_size / (1 - test_size),
        stratify=[labels[i] for i in idx_tv], random_state=seed)
    pick = lambda ids: [samples[i] for i in ids]
    return DatasetBundle(TransformSubset(pick(idx_tr)), TransformSubset(pick(idx_va)),
                         TransformSubset(pick(idx_te)), base.classes, [labels[i] for i in idx_tr])


def make_loaders(bundle: DatasetBundle, batch_size: int, workers: int, pin_memory: bool,
                 distributed: bool = False, world_size: int = 1, rank: int = 0):
    """Train loader uses a DistributedSampler when running under torchrun (world_size>1).
    Val/test loaders stay single-process (run only on rank 0) to keep metric
    computation exact and simple - see train.py.
    """
    kw = dict(num_workers=workers, pin_memory=pin_memory, persistent_workers=workers > 0)
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(bundle.train, num_replicas=world_size, rank=rank,
                                           shuffle=True, drop_last=True)
        train_loader = DataLoader(bundle.train, batch_size=batch_size, sampler=train_sampler,
                                  drop_last=True, **kw)
    else:
        train_loader = DataLoader(bundle.train, batch_size=batch_size, shuffle=True,
                                  drop_last=len(bundle.train) > batch_size, **kw)
    val_loader = DataLoader(bundle.val, batch_size=batch_size, **kw)
    test_loader = DataLoader(bundle.test, batch_size=batch_size, **kw)
    return train_loader, val_loader, test_loader, train_sampler
