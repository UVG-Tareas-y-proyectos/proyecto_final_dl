import csv
import json

import pytest
from PIL import Image

from plant_disease.audit import audit
from plant_disease.prepare import prepare, validate_rows
from test_prepare import fixture_data


def test_audit_counts_exclusions_and_plots(tmp_path):
    root, mapping = fixture_data(tmp_path)
    summary = prepare(root, mapping, tmp_path / "prepared")
    result = audit(root, tmp_path / "prepared/manifest.csv", tmp_path / "audit",
                   summary=tmp_path / "prepared/summary.json", plots=True)
    assert result["images"] == 28
    assert result["groups"] == 14
    assert sum(result["counts"].values()) == 28
    assert result["train_imbalance_ratio"] == 1
    assert result["train_majority_baseline"]["accuracy"] == 0.5
    assert all(number == 0 for pair in result["overlap"].values() for number in pair.values())
    assert result["preparation_exclusions"]["duplicates_removed"] == summary["duplicates_removed"]
    assert (tmp_path / "audit/class_distribution.png").stat().st_size > 1000
    assert len(list(csv.DictReader((tmp_path / "audit/class_distribution.csv").open()))) == 6


def test_audit_rejects_modified_images_and_preserves_old_results(tmp_path):
    root, mapping = fixture_data(tmp_path)
    prepare(root, mapping, tmp_path / "prepared")
    manifest = tmp_path / "prepared/manifest.csv"
    audit(root, manifest, tmp_path / "audit")
    with pytest.raises(ValueError, match="vacía"):
        audit(root, manifest, tmp_path / "audit")
    next(root.glob("*/*.png")).write_bytes(b"modified")
    with pytest.raises(ValueError, match="modificada"):
        audit(root, manifest, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_audit_rejects_unrelated_summary(tmp_path):
    root, mapping = fixture_data(tmp_path)
    summary = prepare(root, mapping, tmp_path / "prepared")
    summary["manifest_sha256"] = "0" * 64
    unrelated = tmp_path / "summary.json"
    unrelated.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="no corresponde"):
        audit(root, tmp_path / "prepared/manifest.csv", tmp_path / "out", summary=unrelated)
    assert not (tmp_path / "out").exists()


def test_audit_detects_files_arriving_after_prepare(tmp_path):
    root, mapping = fixture_data(tmp_path)
    prepare(root, mapping, tmp_path / "prepared")
    Image.new("RGB", (20, 20), "red").save(root / "healthy/arrived_late.png")
    with pytest.raises(ValueError, match="incompleta"):
        audit(root, tmp_path / "prepared/manifest.csv", tmp_path / "audit",
              summary=tmp_path / "prepared/summary.json")
    assert not (tmp_path / "audit").exists()


def test_manifest_rejects_contradictory_leaf_labels(tmp_path):
    root, mapping = fixture_data(tmp_path)
    prepare(root, mapping, tmp_path / "prepared")
    with (tmp_path / "prepared/manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    original = rows[0]
    other = next(r for r in rows if r["class_name"] != original["class_name"] and r["split"] == original["split"])
    other["group_id"] = original["group_id"]
    with pytest.raises(ValueError, match="contradictorias"):
        validate_rows(rows)


def test_manifest_rejects_repeated_image_within_split(tmp_path):
    root, mapping = fixture_data(tmp_path)
    prepare(root, mapping, tmp_path / "prepared")
    with (tmp_path / "prepared/manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    original = rows[0]
    other = next(r for r in rows if r["path"] != original["path"] and r["class_name"] == original["class_name"] and r["split"] == original["split"])
    other["sha256"] = original["sha256"]
    with pytest.raises(ValueError, match="repetida"):
        validate_rows(rows)
