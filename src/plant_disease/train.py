"""Entrenar sin consultar test; seleccionar por F1 macro de validación."""
import argparse
import csv
import json
import os
import platform
import random
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .data import LeafDataset
from .models import build_model
from .prepare import digest, read_manifest


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def class_weights_from_train(rows, classes, mode):
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError("Ponderación desconocida.")
    counts = Counter(row["class_name"] for row in rows if row["split"] == "train")
    if any(counts[name] == 0 for name in classes):
        raise ValueError("Faltan clases en entrenamiento para calcular pesos.")
    total = sum(counts.values())
    return [total / (len(classes) * counts[name]) for name in classes]


def validation_metrics(actual, predicted, classes, loss):
    labels = list(range(len(classes)))
    return dict(split="val", n=len(actual), loss=loss, accuracy=accuracy_score(actual, predicted),
                classes=classes, report=classification_report(actual, predicted, labels=labels,
                    target_names=classes, output_dict=True, zero_division=0),
                confusion_matrix=confusion_matrix(actual, predicted, labels=labels).tolist())


def run_epoch(model, loader, device, optimizer=None, class_weights=None):
    model.train(optimizer is not None)
    if optimizer is not None and hasattr(model, "features"):
        model.features.eval()  # Congela también estadísticas BatchNorm del extractor.
    loss_sum, denominator, actual, predicted = 0.0, 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            numerator = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights, reduction="sum")
            batch_denominator = float(len(labels)) if class_weights is None else class_weights[labels].sum().item()
            loss = numerator / batch_denominator
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += numerator.item()
            denominator += batch_denominator
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(1).detach().cpu().tolist())
    if not actual:
        raise ValueError("La partición no contiene imágenes.")
    return loss_sum / denominator, actual, predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=["cnn", "mobilenet"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("Épocas, batch y threads >= 1; learning rate > 0; weight decay >= 0.")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("La carpeta de salida debe estar vacía.")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    rows = read_manifest(args.manifest, args.root)
    classes = sorted({r["class_name"] for r in rows})
    weights = class_weights_from_train(rows, classes, args.class_weights)
    device = torch.device(args.device)
    model = build_model(args.model, len(classes)).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr,
                                 weight_decay=args.weight_decay)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device) if weights is not None else None
    loaders = {s: DataLoader(LeafDataset(args.root, rows, classes, s), batch_size=args.batch_size,
                            shuffle=s == "train", num_workers=0) for s in ("train", "val")}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(classes=classes, manifest_sha256=digest(args.manifest),
                  python=platform.python_version(), torch=str(torch.__version__),
                  torchvision=str(torchvision.__version__), pretrained=args.model == "mobilenet",
                  augmentation="horizontal_flip + rotation_15, train only", image_size=224,
                  class_weight_values=weights, train_counts=dict(Counter(r["class_name"] for r in rows if r["split"] == "train")),
                  selection_metric="val_f1_macro", validation_loss="unweighted_cross_entropy",
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    repository = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True)
    config["code_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=repository)
    config["code_dirty"] = dirty.returncode != 0
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history, best = [], -1.0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, _, _ = run_epoch(model, loaders["train"], device, optimizer, weight_tensor)
        val_loss, y, pred = run_epoch(model, loaders["val"], device)
        metrics = validation_metrics(y, pred, classes, val_loss)
        score = float(metrics["report"]["macro avg"]["f1-score"])
        record = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_f1_macro=score,
                      val_accuracy=metrics["accuracy"], epoch_seconds=time.perf_counter() - epoch_start)
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best:
            best = score
            torch.save(dict(model_name=args.model, classes=classes, state_dict=model.state_dict(),
                            epoch=epoch, val_f1_macro=score, manifest_sha256=config["manifest_sha256"],
                            config=config), args.out / "best.pt")
            (args.out / "validation_metrics.json").write_text(json.dumps(dict(epoch=epoch, **metrics), indent=2), encoding="utf-8")
            with (args.out / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["path", "actual", "predicted"])
                writer.writerows([[r["path"], classes[a], classes[p]]
                                  for r, a, p in zip(loaders["val"].dataset.rows, y, pred)])
        (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.out / "timing.json").write_text(json.dumps({"training_seconds": time.perf_counter() - start}), encoding="utf-8")
    completed = dict(status="completed", epochs_completed=len(history), best_val_f1_macro=best,
                     finished_at_utc=datetime.now(timezone.utc).isoformat(),
                     artifacts={name: digest(args.out / name) for name in
                                ("best.pt", "config.json", "history.json", "validation_metrics.json", "validation_predictions.csv", "timing.json")})
    (args.out / "completed.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

