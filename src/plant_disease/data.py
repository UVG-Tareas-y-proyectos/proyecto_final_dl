from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


def image_transform(training=False):
    steps = [transforms.Resize((224, 224))]
    if training:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class LeafDataset(Dataset):
    def __init__(self, root, rows, classes, split):
        self.root = Path(root)
        self.rows = [r for r in rows if r["split"] == split]
        self.class_to_idx = {name: i for i, name in enumerate(classes)}
        self.transform = image_transform(split == "train")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return tensor, self.class_to_idx[row["class_name"]]

