from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from .data import make_dataloaders
from .model import create_model
from .utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained plant disease model")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="artifacts/eval_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    _, _, test_dl, class_names = make_dataloaders(
        data_dir=args.data_dir,
        image_size=ckpt["image_size"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = create_model(
        model_name=ckpt["model_name"],
        num_classes=len(class_names),
        pretrained=False,
        dropout=ckpt.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    y_true: list[int] = []
    y_pred: list[int] = []
    losses: list[float] = []

    with torch.no_grad():
        for images, labels in test_dl:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            losses.append(criterion(logits, labels).item())
            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "avg_loss": float(sum(losses) / max(1, len(losses))),
        "classification_report": report,
        "confusion_matrix": cm,
    }
    save_json(payload, args.output)
    print(f"Saved evaluation report to {args.output}")


if __name__ == "__main__":
    main()
