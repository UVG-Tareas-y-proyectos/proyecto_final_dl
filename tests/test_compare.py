import json

import pytest

from plant_disease.compare import compare
from plant_disease.prepare import digest


def make_run(path, score, manifest="same"):
    path.mkdir()
    config = dict(manifest_sha256=manifest, classes=["a", "b"], seed=42, epochs=1,
                  batch_size=32, lr=0.001, weight_decay=0.01, image_size=224,
                  augmentation="train only", validation_loss="unweighted_cross_entropy",
                  model="cnn", class_weights="none")
    history = [dict(epoch=1, val_f1_macro=score, val_loss=0.5)]
    metrics = dict(epoch=1, split="val", n=4, classes=["a", "b"], loss=0.5, accuracy=0.5,
                   report={"macro avg": dict(precision=0.5, recall=0.5, **{"f1-score": score})})
    for name, data in (("config.json", config), ("history.json", history),
                       ("validation_metrics.json", metrics), ("timing.json", {"training_seconds": 1.0})):
        (path / name).write_text(json.dumps(data))
    (path / "validation_predictions.csv").write_text("path,actual,predicted\n")
    artifacts = {name: digest(path / name) for name in
                 ("config.json", "history.json", "validation_metrics.json", "validation_predictions.csv", "timing.json")}
    artifacts["best.pt"] = "not part of this metadata-only fixture"
    (path / "completed.json").write_text(json.dumps(dict(status="completed", epochs_completed=1, artifacts=artifacts)))
    return path


def test_comparison_selects_validation_f1_and_checks_hashes(tmp_path):
    a = make_run(tmp_path / "a", 0.5)
    b = make_run(tmp_path / "b", 0.8)
    report = compare([a, b], tmp_path / "comparison", plots=True)
    assert report["selected_run"] == "b"
    assert report["test_evaluated"] is False
    assert (tmp_path / "comparison/learning_curves.png").stat().st_size > 1000
    (b / "history.json").write_text("[]")
    with pytest.raises(ValueError, match="modificado"):
        compare([a, b], tmp_path / "tampered")


def test_comparison_rejects_different_data_and_unfinished_training(tmp_path):
    a = make_run(tmp_path / "a", 0.5)
    b = make_run(tmp_path / "b", 0.8, manifest="other")
    with pytest.raises(ValueError, match="manifest_sha256"):
        compare([a, b], tmp_path / "different")
    complete = json.loads((b / "completed.json").read_text())
    complete["epochs_completed"] = 0
    (b / "completed.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="presupuesto"):
        compare([a, b], tmp_path / "unfinished")
