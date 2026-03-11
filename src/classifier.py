"""Main image classifier with transfer learning support."""
from __future__ import annotations
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
from .model import build_model
from .dataset import ClassificationDataset
from .trainer import Trainer


class ImageClassifier:
    """High-level interface for image classification."""

    def __init__(
        self,
        model: str = "resnet50",
        num_classes: int = 10,
        pretrained: bool = True,
        device: Optional[str] = None,
    ):
        self.model_name = model
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_model(model, num_classes, pretrained).to(self.device)
        self.class_names: list = []

    def train(
        self,
        train_dir: str,
        val_dir: str,
        epochs: int = 20,
        lr: float = 1e-4,
        batch_size: int = 32,
    ) -> dict:
        """Fine-tune the model on a custom dataset."""
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_ds = ClassificationDataset(train_dir, transform=transform)
        val_ds = ClassificationDataset(val_dir, transform=transform)
        self.class_names = train_ds.classes

        trainer = Trainer(
            model=self.net,
            num_classes=self.num_classes,
            device=self.device,
            lr=lr,
        )
        return trainer.fit(train_ds, val_ds, epochs=epochs, batch_size=batch_size)

    @torch.inference_mode()
    def predict(self, image_path: str) -> Tuple[str, float]:
        """Predict class label for an image."""
        from torchvision import transforms
        self.net.eval()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(self.device)
        logits = self.net(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
        label = self.class_names[idx.item()] if self.class_names else str(idx.item())
        return label, conf.item()

    def save(self, path: str) -> None:
        """Save model weights and class names."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "state_dict": self.net.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "ImageClassifier":
        """Load a saved model checkpoint."""
        ckpt = torch.load(path, map_location="cpu")
        obj = cls(model=ckpt["model_name"], num_classes=ckpt["num_classes"],
                  pretrained=False, device=device)
        obj.net.load_state_dict(ckpt["state_dict"])
        obj.class_names = ckpt.get("class_names", [])
        return obj

    def export_onnx(self, path: str) -> None:
        """Export model to ONNX format for production serving."""
        dummy = torch.zeros(1, 3, 224, 224).to(self.device)
        torch.onnx.export(self.net, dummy, path, opset_version=17,
                          input_names=["image"], output_names=["logits"])
        print(f"Model exported to ONNX: {path}")
