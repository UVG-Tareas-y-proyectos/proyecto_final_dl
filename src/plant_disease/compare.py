"""Comparar checkpoints completos usando exclusivamente validación."""
import argparse
import csv
import json
from pathlib import Path

from .prepare import digest

FAIR_FIELDS = ("manifest_sha256", "classes", "seed", "epochs", "batch_size", "lr",
               "weight_decay", "image_size", "augmentation", "validation_loss")


def read_run(path):
    path = Path(path)
    completed = json.loads((path / "completed.json").read_text(encoding="utf-8"))
    if completed.get("status") != "completed":
        raise ValueError(f"Entrenamiento incompleto: {path.name}")
    for name in ("config.json", "history.json", "validation_metrics.json", "validation_predictions.csv", "timing.json"):
        if digest(path / name) != completed["artifacts"][name]:
            raise ValueError(f"Artefacto modificado: {path.name}/{name}")
    if (path / "best.pt").exists() and digest(path / "best.pt") != completed["artifacts"]["best.pt"]:
        raise ValueError(f"Checkpoint modificado: {path.name}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    history = json.loads((path / "history.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "validation_metrics.json").read_text(encoding="utf-8"))
    timing = json.loads((path / "timing.json").read_text(encoding="utf-8"))
    if completed["epochs_completed"] != config["epochs"] or len(history) != config["epochs"]:
        raise ValueError(f"No terminó el presupuesto de épocas: {path.name}")
    if metrics["split"] != "val" or metrics["classes"] != config["classes"]:
        raise ValueError("Solo se admiten resultados de validación con las clases del entrenamiento.")
    best = max(history, key=lambda epoch: epoch["val_f1_macro"])
    if best["epoch"] != metrics["epoch"] or abs(best["val_f1_macro"] - metrics["report"]["macro avg"]["f1-score"]) > 1e-12:
        raise ValueError(f"El checkpoint no corresponde a la mejor época: {path.name}")
    return config, history, metrics, timing, completed


def compare(runs, out, plots=False):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("La salida debe estar vacía.")
    names = [Path(run).name for run in runs]
    if len(runs) < 2 or len(set(names)) != len(names):
        raise ValueError("Proporciona al menos dos ejecuciones con nombres distintos.")
    loaded = [read_run(run) for run in runs]
    reference = loaded[0][0]
    for config, _, _, _, _ in loaded[1:]:
        for key in FAIR_FIELDS:
            if config[key] != reference[key]:
                raise ValueError(f"Las ejecuciones no son comparables: difiere {key}.")
    records = []
    for name, (config, history, metrics, timing, completed) in zip(names, loaded):
        macro = metrics["report"]["macro avg"]
        records.append(dict(run=name, model=config["model"], class_weights=config["class_weights"],
                            best_epoch=metrics["epoch"], val_n=metrics["n"], val_accuracy=metrics["accuracy"],
                            val_precision_macro=macro["precision"], val_recall_macro=macro["recall"],
                            val_f1_macro=macro["f1-score"], val_loss=metrics["loss"],
                            training_seconds=timing["training_seconds"],
                            checkpoint_sha256=completed["artifacts"]["best.pt"]))
    selected = sorted(records, key=lambda row: (-row["val_f1_macro"], row["run"]))[0]
    report = dict(split="val", selection_metric="val_f1_macro", tie_break="run name ascending",
                  selected_run=selected["run"], selected_checkpoint_sha256=selected["checkpoint_sha256"],
                  manifest_sha256=reference["manifest_sha256"], classes=reference["classes"],
                  seed=reference["seed"], epochs=reference["epochs"], test_evaluated=False, runs=records)
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    if plots:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for name, (_, history, _, _, _) in zip(names, loaded):
            x = [epoch["epoch"] for epoch in history]
            axes[0].plot(x, [epoch["val_loss"] for epoch in history], label=name)
            axes[1].plot(x, [epoch["val_f1_macro"] for epoch in history], label=name)
        for ax, title in zip(axes, ("Entropía cruzada sin ponderar en validación", "F1 macro en validación")):
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Época")
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=8)
        axes[1].set_ylim(0, 1.02)
        fig.savefig(out / "learning_curves.png", dpi=160)
        plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    report = compare(**vars(parser.parse_args()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
