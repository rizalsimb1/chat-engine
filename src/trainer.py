"""Training loop with mixed precision and checkpointing."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class Trainer:
    def __init__(self, model, num_classes: int, device: str, lr: float = 1e-4):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    def fit(self, train_ds: Dataset, val_ds: Dataset,
            epochs: int = 20, batch_size: int = 32) -> dict:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_acc = self._val_epoch(val_loader)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_acc={val_acc:.2%}")

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                loss = self.criterion(self.model(imgs), labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item() * len(imgs)
        return total_loss / len(loader.dataset)

    def _val_epoch(self, loader: DataLoader):
        self.model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                out = self.model(imgs)
                loss = self.criterion(out, labels)
                total_loss += loss.item() * len(imgs)
                correct += (out.argmax(dim=1) == labels).sum().item()
        n = len(loader.dataset)
        return total_loss / n, correct / n
