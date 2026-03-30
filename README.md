# Plant Disease Detection System (End-to-End)

This repo provides a complete PyTorch pipeline for plant disease image classification:

- Data loading from `train/val/test` folder structure
- Transfer-learning model training (ResNet/EfficientNet)
- Validation + best-checkpoint saving
- Final test evaluation with confusion matrix/report export
- Single-image inference CLI

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Dataset format

Your dataset root must look like:

```text
dataset/
  train/
    healthy/
    early_blight/
    late_blight/
  val/
    healthy/
    early_blight/
    late_blight/
  test/
    healthy/
    early_blight/
    late_blight/
```

Each class folder contains images for that class.

## 3) Train

### Quick start

```bash
PYTHONPATH=src python -m pdds.train \
  --data-dir dataset \
  --output-dir artifacts \
  --model-name resnet50 \
  --epochs 20 \
  --batch-size 32 \
  --image-size 224
```

### Train with YAML config

Create `train_config.yaml`:

```yaml
data_dir: dataset
output_dir: artifacts
model_name: resnet50
image_size: 224
batch_size: 32
num_workers: 4
epochs: 20
lr: 0.0003
weight_decay: 0.0001
label_smoothing: 0.1
dropout: 0.2
pretrained: true
amp: true
seed: 42
```

Run:

```bash
PYTHONPATH=src python -m pdds.train --config train_config.yaml
```

Training outputs:

- `artifacts/best_model.pt`
- `artifacts/metrics.json`
- `artifacts/train_config.yaml`

## 4) Evaluate

```bash
PYTHONPATH=src python -m pdds.evaluate \
  --checkpoint artifacts/best_model.pt \
  --data-dir dataset \
  --output artifacts/eval_report.json
```

This exports class-wise metrics and confusion matrix JSON.

## 5) Predict a single image

```bash
PYTHONPATH=src python -m pdds.predict \
  --checkpoint artifacts/best_model.pt \
  --image sample_leaf.jpg \
  --topk 3
```

## 6) Files

```text
requirements.txt
src/pdds/
  __init__.py
  config.py
  data.py
  model.py
  train.py
  evaluate.py
  predict.py
  utils.py
```

## 7) Notes for high accuracy

- Ensure **no leakage** between train/val/test.
- Use balanced class distribution when possible.
- Start with `resnet50` at 224, then test higher resolution (320/384).
- Tune LR, augmentation strength, and weight decay first.
- Evaluate with per-class recall, not only overall accuracy.

---

If you share your dataset stats + current results, I can help you optimize this code for your specific case.
