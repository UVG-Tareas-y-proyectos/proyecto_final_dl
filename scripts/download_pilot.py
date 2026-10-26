"""Descarga un piloto real reproducible; no reemplaza el dataset completo."""
import argparse
import concurrent.futures
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "spMohanty/PlantVillage-Dataset"
CLASSES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "plant-disease-course-project"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def api(path):
    return json.loads(fetch(f"https://api.github.com/repos/{REPO}/{path}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/pilot"))
    parser.add_argument("--ref", default="master", help="Guardar el SHA de source.json para repetir exactamente.")
    parser.add_argument("--leaves-per-class", type=int, default=20)
    args = parser.parse_args()
    if args.leaves_per_class < 3:
        parser.error("Se necesitan al menos tres hojas por clase.")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("La salida debe estar vacía.")
    commit = api(f"commits/{urllib.parse.quote(args.ref, safe='')}")["sha"]
    base = f"https://raw.githubusercontent.com/{REPO}/{commit}/"
    metadata_bytes = fetch(base + "leaf-map.json")
    metadata = json.loads(metadata_bytes)
    tree = api(f"git/trees/{commit}")["tree"]
    for directory in ("raw", "color"):
        sha = next(item["sha"] for item in tree if item["path"] == directory)
        tree = api(f"git/trees/{sha}")["tree"]
    paths = []
    rng = random.Random(42)
    for label in CLASSES:
        sha = next(item["sha"] for item in tree if item["path"] == label)
        response = api(f"git/trees/{sha}")
        if response.get("truncated"):
            raise ValueError("Inventario remoto incompleto.")
        groups = {}
        for item in sorted(response["tree"], key=lambda x: x["path"]):
            name = item["path"]
            key = Path(name).stem.split("___")[-1].lower()
            leaves = metadata.get(key, [])
            if item["type"] == "blob" and len(leaves) == 1:
                groups.setdefault(leaves[0], []).append(f"raw/color/{label}/{name}")
        keys = sorted(groups)
        if len(keys) < args.leaves_per_class:
            raise ValueError(f"No hay suficientes hojas para {label}.")
        rng.shuffle(keys)
        for key in keys[:args.leaves_per_class]:
            paths.extend(groups[key][:2])
    args.out.mkdir(parents=True)
    (args.out / "leaf-map.json").write_bytes(metadata_bytes)

    def download(path):
        target = args.out / "color" / Path(path).relative_to("raw/color")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fetch(base + urllib.parse.quote(path)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(download, paths))
    source = dict(repository=REPO, commit=commit, seed=42, leaves_per_class=args.leaves_per_class,
                  max_images_per_leaf=2, classes=CLASSES, images=len(paths), paths=paths,
                  license="CC BY-SA 3.0, según ficha oficial de Hugging Face")
    (args.out / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in source.items() if k != "paths"}, indent=2))


if __name__ == "__main__":
    main()
