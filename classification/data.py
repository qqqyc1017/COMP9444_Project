from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class IP102Split(Dataset):
    """Read official IP102 split files: one image path and integer label per line."""

    def __init__(self, root: Path, split_file: Path, transform: Callable, label_map=None):
        self.root, self.transform = root, transform
        raw_samples: list[tuple[Path, int]] = []
        for line in split_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            path_text, label_text = line.rsplit(maxsplit=1)
            path = Path(path_text)
            candidates = [root / path, root / "images" / path, root / "classification" / path]
            image_path = next((p for p in candidates if p.exists()), candidates[1])
            raw_samples.append((image_path, int(label_text)))
        if label_map is None:
            label_map = {label: i for i, label in enumerate(sorted({y for _, y in raw_samples}))}
        unknown = sorted({y for _, y in raw_samples} - set(label_map))
        if unknown:
            raise ValueError(f"Split {split_file} contains labels absent from training: {unknown}")
        self.label_map = label_map
        self.classes = [str(raw) for raw, _ in sorted(label_map.items(), key=lambda x: x[1])]
        self.samples = [(path, label_map[label]) for path, label in raw_samples]
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def build_transforms(image_size: int = 224):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def _find_split(root: Path, split: str) -> Path | None:
    names = [f"{split}.txt", f"{split}.json"]
    for folder in (root, root / "classification", root / "classification" / "train_test_split"):
        for name in names:
            path = folder / name
            if path.exists() and path.suffix == ".txt":
                return path
    return None


def make_datasets(root: str | Path, image_size: int = 224):
    root = Path(root)
    train_tf, eval_tf = build_transforms(image_size)
    split_files = {s: _find_split(root, s) for s in ("train", "val", "test")}
    if all(split_files.values()):
        train = IP102Split(root, split_files["train"], train_tf)
        class_names = train.classes
        val = IP102Split(root, split_files["val"], eval_tf, train.label_map)
        test = IP102Split(root, split_files["test"], eval_tf, train.label_map)
    elif all((root / s).is_dir() for s in ("train", "val", "test")):
        train = datasets.ImageFolder(root / "train", train_tf)
        val = datasets.ImageFolder(root / "val", eval_tf)
        test = datasets.ImageFolder(root / "test", eval_tf)
        class_names = train.classes
        if val.classes != class_names or test.classes != class_names:
            raise ValueError("train/val/test must contain identical class folders")
    else:
        raise FileNotFoundError(
            f"No supported IP102 layout under {root}. Expected split txt files or "
            "train/, val/, test/ class folders. See README.md."
        )
    return {"train": train, "val": val, "test": test}, class_names


def make_loaders(datasets_by_split, batch_size: int, workers: int):
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
        for split, dataset in datasets_by_split.items()
    }
