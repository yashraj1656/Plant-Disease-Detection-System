from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torchvision import models


def _replace_classifier(module: nn.Module, in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )


def create_model(model_name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    name = model_name.lower()
    weights = "DEFAULT" if pretrained else None

    factories: dict[str, Callable[[], nn.Module]] = {
        "resnet18": lambda: models.resnet18(weights=weights),
        "resnet50": lambda: models.resnet50(weights=weights),
        "efficientnet_b0": lambda: models.efficientnet_b0(weights=weights),
        "efficientnet_b3": lambda: models.efficientnet_b3(weights=weights),
    }
    if name not in factories:
        supported = ", ".join(sorted(factories))
        raise ValueError(f"Unsupported model_name '{model_name}'. Use one of: {supported}")

    model = factories[name]()

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = _replace_classifier(model, model.fc.in_features, num_classes, dropout)
    elif hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential) and len(classifier) > 0:
            last = classifier[-1]
            if isinstance(last, nn.Linear):
                new_layers = list(classifier[:-1])
                new_layers.append(nn.Dropout(dropout))
                new_layers.append(nn.Linear(last.in_features, num_classes))
                model.classifier = nn.Sequential(*new_layers)
            else:
                raise ValueError("Could not locate final linear layer in classifier")
        else:
            raise ValueError("Unsupported classifier structure")
    else:
        raise ValueError("Unsupported model architecture")

    return model


@torch.no_grad()
def predict_logits(model: nn.Module, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    logits = model(batch.to(device, non_blocking=True))
    return logits
