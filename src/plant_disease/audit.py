"""Auditar un manifiesto antes de entrenar, sin ejecutar ni seleccionar modelos."""
import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from PIL import Image

from .prepare import SPLITS, digest, read_manifest


def audit(root, manifest, out, summary=None, plots=False):
    root, manifest, out = Path(root), Path(manifest), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("La salida debe estar vacía para conservar auditorías anteriores.")
    rows = read_manifest(manifest, root)
    preparation = None
    if summary is not None:
        preparation = json.loads(Path(summary).read_text(encoding="utf-8"))
        if preparation["manifest_sha256"] != digest(manifest):
            raise ValueError("El resumen de preparación no corresponde al manifiesto.")
    physical = {p.relative_to(root).as_posix() for p in root.glob("*/*")
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    included = {row["path"] for row in rows}
    excluded = set()
    if preparation is not None:
        excluded = set(preparation["missing_leaf_metadata"] + preparation["damaged_images"] +
                       preparation.get("duplicate_images", []))
        if physical != included | excluded:
            raise ValueError("Hay imágenes sin contabilizar o exclusiones ausentes: descarga o preparación incompleta.")
    classes = sorted({row["class_name"] for row in rows})
    overlap = {}
    for left, right in combinations(SPLITS, 2):
        overlap[f"{left}/{right}"] = {
            key: len({r[key] for r in rows if r["split"] == left} &
                     {r[key] for r in rows if r["split"] == right})
            for key in ("group_id", "sha256")
        }
    image_info = []
    for row in rows:
        with Image.open(root / row["path"]) as image:
            image.load()  # También comprueba que el archivo se decodifica completo.
            image_info.append(dict(path=row["path"], width=image.width, height=image.height,
                                   mode=image.mode, split=row["split"]))
    distribution = []
    for label in classes:
        for split in SPLITS:
            selected = [r for r in rows if r["class_name"] == label and r["split"] == split]
            distribution.append(dict(class_name=label, split=split, images=len(selected),
                                     groups=len({r["group_id"] for r in selected})))
    train_counts = {r["class_name"]: r["images"] for r in distribution if r["split"] == "train"}
    total_train = sum(train_counts.values())
    majority = max(classes, key=lambda name: train_counts[name])
    imbalance = max(train_counts.values()) / min(train_counts.values())
    warnings = []
    if imbalance > 2:
        warnings.append("Desbalance de entrenamiento mayor a 2:1; accuracy puede ocultar errores en la clase minoritaria.")
    if any(r["groups"] < 10 for r in distribution):
        warnings.append("Hay clases con menos de 10 hojas en alguna partición; sus métricas tendrán alta variabilidad.")
    if preparation is None and physical - included:
        warnings.append("Hay imágenes fuera del manifiesto; proporciona --summary para comprobar todas las exclusiones.")
    counts = {split: sum(r["split"] == split for r in rows) for split in SPLITS}
    report = dict(schema_version=1, passed=True, manifest_sha256=digest(manifest),
                  images=len(rows), groups=len({r["group_id"] for r in rows}), classes=classes,
                  physical_images=len(physical), excluded_images=len(excluded),
                  counts=counts, image_fractions={s: counts[s] / len(rows) for s in SPLITS},
                  overlap=overlap, class_distribution=distribution,
                  image_modes=dict(Counter(r["mode"] for r in image_info)),
                  resolutions=dict(Counter(f"{r['width']}x{r['height']}" for r in image_info)),
                  train_imbalance_ratio=imbalance,
                  train_majority_baseline=dict(class_name=majority, accuracy=train_counts[majority] / total_train),
                  suggested_train_class_weights={c: total_train / (len(classes) * train_counts[c]) for c in classes},
                  warnings=warnings,
                  scope="Calidad y separación del dataset; no mide el desempeño predictivo ni identifica todas las similitudes visuales.")
    if preparation is not None:
        report["preparation_exclusions"] = dict(
            missing_leaf_metadata=len(preparation["missing_leaf_metadata"]),
            damaged_images=len(preparation["damaged_images"]),
            duplicates_removed=preparation["duplicates_removed"],
            summary_sha256=digest(summary))
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, records in (("class_distribution.csv", distribution), ("image_inventory.csv", image_info)):
        with (out / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    if plots:
        positions = list(range(len(classes)))
        fig, ax = plt.subplots(figsize=(9, max(3.5, len(classes) * 0.75)), layout="constrained")
        for index, split in enumerate(SPLITS):
            values = [r["images"] for r in distribution if r["split"] == split]
            bars = ax.barh([p + (index - 1) * 0.24 for p in positions], values, height=0.24, label=split)
            ax.bar_label(bars, padding=3, fontsize=8)
        ax.set_yticks(positions, [name.replace("Potato___", "").replace("_", " ") for name in classes])
        ax.set_xlabel("Número de imágenes")
        ax.set_title("PlantVillage · distribución por clase y partición")
        ax.set_xlim(0, max(r["images"] for r in distribution) * 1.2)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(out / "class_distribution.png", dpi=160)
        plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "manifest", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--plots", action="store_true", help="Requiere pip install -e '.[analysis]'.")
    args = parser.parse_args()
    result = audit(**vars(args))
    print(json.dumps({key: result[key] for key in ("passed", "images", "groups", "counts", "train_imbalance_ratio", "warnings")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
