"""Entrenar sin consultar test; seleccionar por F1 macro de validación."""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .data import LeafDataset
from .models import build_model
from .prepare import digest, read_manifest


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    if optimizer is not None and hasattr(model, "features"):
        model.features.eval()  # Congela también estadísticas BatchNorm del extractor.
    loss_sum, actual, predicted = 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(1).detach().cpu().tolist())
    return loss_sum / len(actual), actual, predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=["cnn", "mobilenet"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error("Épocas, batch, threads y learning rate deben ser positivos.")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("La carpeta de salida debe estar vacía.")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    rows = read_manifest(args.manifest, args.root)
    classes = sorted({r["class_name"] for r in rows})
    device = torch.device(args.device)
    model = build_model(args.model, len(classes)).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    loaders = {s: DataLoader(LeafDataset(args.root, rows, classes, s), batch_size=args.batch_size,
                            shuffle=s == "train", num_workers=0) for s in ("train", "val")}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(classes=classes, manifest_sha256=digest(args.manifest),
                  python=platform.python_version(), torch=str(torch.__version__),
                  torchvision=str(torchvision.__version__), pretrained=args.model == "mobilenet",
                  augmentation="horizontal_flip + rotation_15, train only", image_size=224)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history, best = [], -1.0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, loaders["train"], device, optimizer)
        val_loss, y, pred = run_epoch(model, loaders["val"], device)
        score = float(f1_score(y, pred, labels=list(range(len(classes))), average="macro", zero_division=0))
        record = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_f1_macro=score)
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best:
            best = score
            torch.save(dict(model_name=args.model, classes=classes, state_dict=model.state_dict(),
                            epoch=epoch, val_f1_macro=score, manifest_sha256=config["manifest_sha256"],
                            config=config), args.out / "best.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.out / "timing.json").write_text(json.dumps({"training_seconds": time.perf_counter() - start}), encoding="utf-8")


if __name__ == "__main__":
    main()

