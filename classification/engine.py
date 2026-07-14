from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from tqdm import tqdm


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, y_true, y_pred = 0.0, [], []
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp = device.type == "cuda"
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(1).detach().cpu().tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }, y_true, y_pred


@torch.inference_mode()
def benchmark_latency(model, device, image_size=224, warmup=10, repeats=50):
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def save_training_plot(history, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history) + 1)
    axes[0].plot(epochs, [x["train_loss"] for x in history], label="train")
    axes[0].plot(epochs, [x["val_loss"] for x in history], label="val")
    axes[0].set(title="Loss", xlabel="Epoch"); axes[0].legend()
    axes[1].plot(epochs, [x["train_accuracy"] for x in history], label="train")
    axes[1].plot(epochs, [x["val_accuracy"] for x in history], label="val")
    axes[1].set(title="Accuracy", xlabel="Epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def save_confusion(y_true, y_pred, class_names, output: Path):
    size = max(10, min(26, len(class_names) * 0.22))
    fig, ax = plt.subplots(figsize=(size, size))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, include_values=False,
        normalize="true", cmap="Blues", ax=ax, xticks_rotation=90,
    )
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def write_json(data, path: Path):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

