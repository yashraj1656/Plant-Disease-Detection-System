from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .data import IMAGENET_MEAN, IMAGENET_STD
from .model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single image prediction")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = create_model(
        model_name=ckpt["model_name"],
        num_classes=len(ckpt["class_names"]),
        pretrained=False,
        dropout=ckpt.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    image_size = ckpt["image_size"]
    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    image = Image.open(Path(args.image)).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    topk = min(args.topk, probs.numel())
    values, indices = torch.topk(probs, k=topk)
    print(f"Image: {args.image}")
    for prob, idx in zip(values.tolist(), indices.tolist()):
        print(f"{ckpt['class_names'][idx]}: {prob:.4f}")


if __name__ == "__main__":
    main()
