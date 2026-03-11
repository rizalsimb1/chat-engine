# chat-engine

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Stars](https://img.shields.io/github/stars/rizalsimb1/chat-engine?style=social)
![Issues](https://img.shields.io/github/issues/rizalsimb1/chat-engine)

> A complete image classification pipeline using PyTorch and torchvision. Supports custom datasets, transfer learning from pretrained models (ResNet, EfficientNet), training dashboard, and ONNX export.

## ✨ Features

- ✅ Transfer learning from pretrained ResNet50, EfficientNet-B4, ViT
- ✅ Custom dataset loader with augmentation pipeline
- ✅ Mixed precision training with torch.cuda.amp
- ✅ Training progress dashboard with live loss/accuracy plots
- ✅ Model checkpoint saving and resumption
- ✅ ONNX export for production deployment
- ✅ Grad-CAM visualization for interpretability
- ✅ REST API inference server with FastAPI

## 🛠️ Tech Stack

`Python 3.11+` • `PyTorch 2.x` • `torchvision` • `FastAPI` • `timm` • `PIL` • `matplotlib`

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/rizalsimb1/chat-engine.git
cd chat-engine

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
from classifier import ImageClassifier

# Train from scratch or fine-tune
clf = ImageClassifier(model="resnet50", num_classes=10, pretrained=True)
clf.train(
    train_dir="data/train",
    val_dir="data/val",
    epochs=20,
    lr=1e-4,
)
clf.save("model.pth")

# Inference
clf = ImageClassifier.load("model.pth")
label, confidence = clf.predict("test.jpg")
print(f"{label}: {confidence:.1%}")

```

## 📁 Project Structure

```
chat-engine/
├── src/
│   └── main files
├── tests/
│   └── test files
├── requirements.txt
├── README.md
└── LICENSE
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'feat: add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ by <a href="https://github.com/rizalsimb1">rizalsimb1</a></p>

