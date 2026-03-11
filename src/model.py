"""Model factory — builds pretrained classification models."""
import torch.nn as nn


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        import timm
        model = timm.create_model(
            _timm_name(name),
            pretrained=pretrained,
            num_classes=num_classes,
        )
        return model
    except ImportError:
        pass

    from torchvision import models
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b4(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model


def _timm_name(name: str) -> str:
    mapping = {
        "resnet50": "resnet50",
        "efficientnet_b4": "efficientnet_b4",
        "vit": "vit_base_patch16_224",
    }
    return mapping.get(name, name)
