from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import TrainConfig, load_config, save_config
from .data import make_dataloaders
from .model import create_model
from .utils import ensure_dir, save_json, seed_everything


def evaluate_epoch(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            losses.append(loss.item())
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "accuracy": float(accuracy_score(all_targets, all_preds)),
        "macro_f1": float(f1_score(all_targets, all_preds, average="macro")),
    }


def train(config: TrainConfig) -> None:
    seed_everything(config.seed)
    out_dir = ensure_dir(config.output_dir)
    save_config(config, out_dir / "train_config.yaml")

    train_dl, val_dl, test_dl, class_names = make_dataloaders(
        data_dir=config.data_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_name=config.model_name,
        num_classes=len(class_names),
        pretrained=config.pretrained,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")

    history: list[dict[str, float]] = []
    best_f1 = -1.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for step, (images, targets) in enumerate(pbar, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config.amp):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / step)

        scheduler.step()

        train_loss = running_loss / max(len(train_dl), 1)
        val_metrics = evaluate_epoch(model, val_dl, criterion, device)
        result = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(result)
        print(result)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "model_name": config.model_name,
                    "image_size": config.image_size,
                    "dropout": config.dropout,
                },
                out_dir / "best_model.pt",
            )

    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate_epoch(model, test_dl, criterion, device)

    save_json(
        {
            "class_names": class_names,
            "history": history,
            "best_val_macro_f1": best_f1,
            "test": test_metrics,
        },
        out_dir / "metrics.json",
    )
    print("Final test metrics:", test_metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train plant disease classifier")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root with train/val/test")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        if not args.data_dir:
            raise ValueError("--data-dir is required when --config is not provided")
        config = TrainConfig(data_dir=args.data_dir)

    for attr in ("data_dir", "output_dir", "epochs", "batch_size", "model_name", "image_size"):
        value = getattr(args, attr)
        if value is not None:
            setattr(config, attr, value)

    train(config)


if __name__ == "__main__":
    main()
