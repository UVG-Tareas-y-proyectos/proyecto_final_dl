"""Evaluación de test con el checkpoint seleccionado usando validación."""
import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .data import LeafDataset
from .predict import load_checkpoint
from .prepare import digest, read_manifest
from .train import run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "manifest", "checkpoint", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("La salida debe estar vacía.")
    torch.set_num_threads(2)
    model, checkpoint = load_checkpoint(args.checkpoint)
    if digest(args.manifest) != checkpoint["manifest_sha256"]:
        raise ValueError("El manifiesto difiere del utilizado para entrenar.")
    rows = read_manifest(args.manifest, args.root)
    classes = checkpoint["classes"]
    if sorted({r["class_name"] for r in rows}) != classes:
        raise ValueError("Las clases del manifiesto no coinciden con el modelo.")
    dataset = LeafDataset(args.root, rows, classes, "test")
    loss, actual, pred = run_epoch(model, DataLoader(dataset, batch_size=32), "cpu")
    labels = list(range(len(classes)))
    report = classification_report(actual, pred, labels=labels, target_names=classes,
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, pred, labels=labels).tolist()
    metrics = dict(split="test", n=len(actual), loss=loss, accuracy=accuracy_score(actual, pred),
                   report=report, classes=classes, confusion_matrix=matrix,
                   checkpoint_sha256=digest(args.checkpoint), manifest_sha256=digest(args.manifest))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.out / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["real / predicha", *classes])
        writer.writerows([[name, *values] for name, values in zip(classes, matrix)])
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "actual", "predicted"])
        writer.writerows([[r["path"], classes[y], classes[p]] for r, y, p in zip(dataset.rows, actual, pred)])
    print(json.dumps(dict(accuracy=metrics["accuracy"], macro=report["macro avg"]), indent=2))


if __name__ == "__main__":
    main()

