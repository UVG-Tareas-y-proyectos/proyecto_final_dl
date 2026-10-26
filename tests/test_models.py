import pytest
import torch
from PIL import Image

from plant_disease.models import build_model
from plant_disease.predict import load_checkpoint, predict_image


@pytest.mark.parametrize("name", ["cnn", "mobilenet"])
def test_output_shape_and_frozen_extractor(name):
    torch.set_num_threads(2)
    model = build_model(name, 3, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 3)
    if name == "mobilenet":
        assert all(not p.requires_grad for p in model.features.parameters())
        assert all(p.requires_grad for p in model.classifier.parameters())


def test_checkpoint_roundtrip_and_rgb_conversion(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model("cnn", 2, pretrained=False).eval()
    path = tmp_path / "model.pt"
    torch.save(dict(model_name="cnn", classes=["a", "b"], state_dict=model.state_dict()), path)
    loaded, checkpoint = load_checkpoint(path)
    image = Image.new("L", (30, 30), 80)
    before = predict_image(model, ["a", "b"], image)
    after = predict_image(loaded, checkpoint["classes"], image)
    assert before == after
    assert sum(r["score"] for r in after) == pytest.approx(1.0)

