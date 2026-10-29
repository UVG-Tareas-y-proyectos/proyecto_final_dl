"""Inventario y separación determinista por hoja, nunca por imagen."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

FIELDS = ["path", "class_name", "group_id", "sha256", "split"]
SPLITS = ("train", "val", "test")


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def leaf_key(path):
    return Path(path).stem.split("___")[-1].lower()


def prepare(root, leaf_map, out, seed=42):
    root, out = Path(root).resolve(), Path(out)
    if not root.is_dir():
        raise ValueError(f"No existe el directorio de imágenes: {root}")
    if out.exists() and any(out.iterdir()):
        raise ValueError("La salida debe estar vacía para no sobrescribir particiones.")
    metadata = json.loads(Path(leaf_map).read_text(encoding="utf-8"))
    rows, excluded, damaged, duplicates = [], [], [], 0
    duplicate_images = []
    seen_hashes = {}
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    for path in sorted(root.glob("*/*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        rel = path.relative_to(root).as_posix()
        groups = metadata.get(leaf_key(path), [])
        if not isinstance(groups, list) or not groups or not all(isinstance(g, str) and g for g in groups):
            excluded.append(rel)
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError):
            damaged.append(rel)
            continue
        sha = digest(path)
        for group in groups:
            union(groups[0], group)
        if sha in seen_hashes:
            previous = seen_hashes[sha]
            if previous["class_name"] != path.parent.name:
                raise ValueError("Una imagen idéntica tiene etiquetas contradictorias.")
            union(groups[0], previous["group_id"])
            duplicates += 1
            duplicate_images.append(rel)
            continue
        row = dict(path=rel, class_name=path.parent.name, group_id=groups[0], sha256=sha)
        rows.append(row)
        seen_hashes[sha] = row
    by_class = defaultdict(set)
    group_labels = {}
    for row in rows:
        group = row["group_id"] = find(row["group_id"])
        label = row["class_name"]
        if group in group_labels and group_labels[group] != label:
            raise ValueError(f"Grupo con clases contradictorias: {group}")
        group_labels[group] = label
        by_class[label].add(group)
    if len(by_class) < 2:
        raise ValueError("Se necesitan al menos dos clases válidas con metadatos de hoja.")
    assignments = {}
    rng = random.Random(seed)
    for label, members in sorted(by_class.items()):
        groups = sorted(members)
        if len(groups) < 3:
            raise ValueError(f"{label}: se necesitan al menos tres hojas independientes.")
        rng.shuffle(groups)
        nval = max(1, int(len(groups) * 0.15))
        ntest = max(1, int(len(groups) * 0.15))
        for index, group in enumerate(groups):
            assignments[group] = "val" if index < nval else "test" if index < nval + ntest else "train"
    for row in rows:
        row["split"] = assignments[row["group_id"]]
    validate_rows(rows)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(seed=seed, images=len(rows), groups=len(assignments),
                   classes=sorted(by_class), manifest_sha256=digest(manifest),
                   leaf_map_sha256=digest(leaf_map),
                   duplicates_removed=duplicates, missing_leaf_metadata=excluded,
                   duplicate_images=duplicate_images,
                   damaged_images=damaged,
                   counts={s: dict(Counter(r["class_name"] for r in rows if r["split"] == s)) for s in SPLITS})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def validate_rows(rows):
    if not rows:
        raise ValueError("Manifiesto vacío.")
    owners = {"path": {}, "group_id": {}, "sha256": {}}
    classes = set()
    group_labels = {}
    hash_labels = {}
    for row in rows:
        if not all(row.get(k) for k in FIELDS) or row["split"] not in SPLITS:
            raise ValueError("Fila incompleta o partición desconocida.")
        classes.add(row["class_name"])
        for key, labels in (("group_id", group_labels), ("sha256", hash_labels)):
            if row[key] in labels and labels[row[key]] != row["class_name"]:
                raise ValueError(f"Etiquetas contradictorias para {key}={row[key]}")
            labels[row[key]] = row["class_name"]
        for key, seen in owners.items():
            value = row[key]
            if value in seen and (seen[value] != row["split"] or key in {"path", "sha256"}):
                raise ValueError(f"Fuga de datos o ruta repetida: {key}={value}")
            seen[value] = row["split"]
    for split in SPLITS:
        if {r["class_name"] for r in rows if r["split"] == split} != classes:
            raise ValueError(f"Faltan clases en {split}.")


def read_manifest(path, root):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    validate_rows(rows)
    root = Path(root).resolve()
    for row in rows:
        target = (root / row["path"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Ruta fuera del directorio de datos.")
        if digest(target) != row["sha256"]:
            raise ValueError(f"Imagen modificada: {row['path']}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--leaf-map", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    summary = prepare(args.root, args.leaf_map, args.out, args.seed)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"missing_leaf_metadata", "damaged_images"}}, indent=2))


if __name__ == "__main__":
    main()

