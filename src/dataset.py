"""Custom image dataset loader with augmentation."""
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Optional, List


class ClassificationDataset(Dataset):
    """Dataset for folder-structured image classification (ImageFolder style)."""

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.classes: List[str] = sorted(
            [d.name for d in self.root.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = [
            (str(f), self.class_to_idx[f.parent.name])
            for cls_dir in self.root.iterdir() if cls_dir.is_dir()
            for f in cls_dir.glob("*")
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
