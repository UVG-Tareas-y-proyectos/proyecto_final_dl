import torch
from PIL import ImageOps

from .data import image_transform
from .models import build_model


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    if len(set(classes)) != len(classes):
        raise ValueError("Clases repetidas en el checkpoint.")
    model = build_model(checkpoint["model_name"], len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def predict_image(model, classes, image, device="cpu"):
    image = ImageOps.exif_transpose(image).convert("RGB")
    inputs = image_transform()(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(inputs).softmax(1)[0].cpu().tolist()
    return sorted([dict(class_name=name, score=score) for name, score in zip(classes, probabilities)],
                  key=lambda row: row["score"], reverse=True)

