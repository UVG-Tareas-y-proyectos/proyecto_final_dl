import json
import math
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from plant_disease.predict import load_checkpoint
from plant_disease.train import class_weights_from_train, run_epoch
from test_prepare import fixture_data
from plant_disease.prepare import prepare


def test_class_weights_use_only_train():
    rows = [dict(class_name=name, split="train") for name in ["a", "a", "a", "b"]]
    rows += [dict(class_name="b", split="val")] * 100
    rows += [dict(class_name="b", split="test")] * 100
    assert class_weights_from_train(rows, ["a", "b"], "balanced") == pytest.approx([2 / 3, 2])
    assert class_weights_from_train(rows, ["a", "b"], "none") is None
    with pytest.raises(ValueError, match="Faltan clases"):
        class_weights_from_train(rows, ["a", "b", "missing"], "balanced")


def test_weighted_epoch_loss_has_correct_denominator():
    logits = torch.tensor([[math.log(3), 0], [math.log(3), 0]], dtype=torch.float32)
    labels = torch.tensor([0, 1])
    weights = torch.tensor([1.0, 3.0])
    dataset = TensorDataset(logits, labels)
    expected = (-math.log(0.75) - 3 * math.log(0.25)) / 4
    for batch_size in (1, 2):
        loss, actual, pred = run_epoch(torch.nn.Identity(), DataLoader(dataset, batch_size=batch_size),
                                       "cpu", class_weights=weights)
        assert loss == pytest.approx(expected)
        assert actual == [0, 1] and pred == [0, 0]


def test_training_cli_persists_validation_without_test(tmp_path):
    root, mapping = fixture_data(tmp_path)
    prepare(root, mapping, tmp_path / "prepared")
    out = tmp_path / "run"
    subprocess.run([sys.executable, "-m", "plant_disease.train", "--root", str(root),
                    "--manifest", str(tmp_path / "prepared/manifest.csv"), "--out", str(out),
                    "--model", "cnn", "--class-weights", "balanced", "--epochs", "1"], check=True,
                   capture_output=True, text=True)
    complete = json.loads((out / "completed.json").read_text())
    metrics = json.loads((out / "validation_metrics.json").read_text())
    assert complete["epochs_completed"] == 1
    assert metrics["split"] == "val" and metrics["n"] == 4
    assert not (out / "test").exists()
    _, checkpoint = load_checkpoint(out / "best.pt")
    assert checkpoint["config"]["class_weight_values"] == [1.0, 1.0]
