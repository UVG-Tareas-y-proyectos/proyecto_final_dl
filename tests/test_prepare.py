import csv
import json

import pytest
from PIL import Image

from plant_disease.prepare import prepare, read_manifest, validate_rows


def fixture_data(tmp_path):
    root = tmp_path / "images"
    metadata = {}
    for cls in ("healthy", "disease"):
        (root / cls).mkdir(parents=True)
        for leaf in range(7):
            for angle in range(2):
                name = f"{cls}{leaf}_{angle}"
                Image.new("RGB", (12, 12), (leaf * 30, angle * 80, 50 if cls == "healthy" else 200)).save(root / cls / f"{name}.png")
                metadata[name] = [f"{cls}:::{leaf}"]
    path = tmp_path / "map.json"
    path.write_text(json.dumps(metadata))
    return root, path


def test_groups_are_disjoint_and_reproducible(tmp_path):
    root, mapping = fixture_data(tmp_path)
    a = prepare(root, mapping, tmp_path / "a")
    b = prepare(root, mapping, tmp_path / "b")
    assert a["manifest_sha256"] == b["manifest_sha256"]
    rows = read_manifest(tmp_path / "a/manifest.csv", root)
    assert len(rows) == 28
    sets = [{r["group_id"] for r in rows if r["split"] == s} for s in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_modified_image_rejected(tmp_path):
    root, mapping = fixture_data(tmp_path)
    prepare(root, mapping, tmp_path / "out")
    next(root.glob("*/*.png")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="modificada"):
        read_manifest(tmp_path / "out/manifest.csv", root)


def test_missing_metadata_excluded(tmp_path):
    root, mapping = fixture_data(tmp_path)
    Image.new("RGB", (12, 12)).save(root / "healthy/unknown.png")
    result = prepare(root, mapping, tmp_path / "out")
    assert result["missing_leaf_metadata"] == ["healthy/unknown.png"]


def test_duplicate_and_leaf_overlap_rejected(tmp_path):
    root, mapping = fixture_data(tmp_path)
    prepare(root, mapping, tmp_path / "out")
    with (tmp_path / "out/manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["sha256"] = next(r["sha256"] for r in rows if r["split"] != rows[0]["split"])
    with pytest.raises(ValueError, match="Fuga"):
        validate_rows(rows)

