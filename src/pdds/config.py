from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str = "artifacts"
    model_name: str = "resnet50"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    dropout: float = 0.2
    pretrained: bool = True
    amp: bool = True
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> TrainConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TrainConfig(**raw)


def save_config(config: TrainConfig, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
