"""Descarga un piloto real reproducible; no reemplaza el dataset completo."""
import argparse
import concurrent.futures
import hashlib
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "spMohanty/PlantVillage-Dataset"
CLASSES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "plant-disease-course-project"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)


def verify_blob(content, expected_sha):
    """Verifica el objeto Git esperado en la revisión fijada, antes de guardar."""
    header = f"blob {len(content)}\0".encode("ascii")
    if hashlib.sha1(header + content).hexdigest() != expected_sha:
        raise ValueError("La descarga no coincide con el objeto Git de la imagen.")


def select_images(entries, metadata, label, rng, leaves_per_class=20,
                  max_images_per_leaf=2, all_leaves=False, all_images=False):
    files = [item for item in sorted(entries, key=lambda item: item["path"])
             if item["type"] == "blob" and Path(item["path"]).suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if all_leaves and all_images:
        return files  # El preparador reportará metadatos faltantes o ambiguos.
    groups = {}
    for item in files:
        key = Path(item["path"]).stem.split("___")[-1].lower()
        leaves = metadata.get(key, [])
        if isinstance(leaves, list) and len(leaves) == 1:
            groups.setdefault(leaves[0], []).append(item)
    keys = sorted(groups)
    if not all_leaves and len(keys) < leaves_per_class:
        raise ValueError(f"No hay suficientes hojas para {label}.")
    rng.shuffle(keys)
    chosen = keys if all_leaves else keys[:leaves_per_class]
    return [item for key in chosen for item in
            (groups[key] if all_images else groups[key][:max_images_per_leaf])]


def api(path):
    return json.loads(fetch(f"https://api.github.com/repos/{REPO}/{path}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/raw/pilot"))
    parser.add_argument("--ref", default="master", help="Guardar el SHA de source.json para repetir exactamente.")
    parser.add_argument("--leaves-per-class", type=int, default=20)
    parser.add_argument("--classes", nargs="+", default=CLASSES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-leaves", action="store_true")
    parser.add_argument("--all-images", action="store_true")
    parser.add_argument("--max-images-per-leaf", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    initial, _ = parser.parse_known_args()
    if initial.config:
        config = json.loads(initial.config.read_text(encoding="utf-8"))
        allowed = {action.dest for action in parser._actions} - {"help", "config"}
        if not isinstance(config, dict) or set(config) - allowed:
            parser.error("La configuración contiene campos desconocidos.")
        parser.set_defaults(**config)
    args = parser.parse_args()
    if args.leaves_per_class < 3 or args.max_images_per_leaf < 1 or args.workers < 1:
        parser.error("Hojas >= 3, imágenes por hoja >= 1 y workers >= 1.")
    if len(args.classes) < 2 or len(set(args.classes)) != len(args.classes):
        parser.error("Selecciona al menos dos clases distintas.")
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
    paths, blobs = [], {}
    rng = random.Random(args.seed)
    remote_classes = {item["path"]: item["sha"] for item in tree if item["type"] == "tree"}
    for label in args.classes:
        if label not in remote_classes:
            parser.error(f"Clase desconocida: {label}")
        sha = remote_classes[label]
        response = api(f"git/trees/{sha}")
        if response.get("truncated"):
            raise ValueError("Inventario remoto incompleto.")
        selected = select_images(response["tree"], metadata, label, rng,
                                 args.leaves_per_class, args.max_images_per_leaf,
                                 args.all_leaves, args.all_images)
        for item in selected:
            path = f"raw/color/{label}/{item['path']}"
            paths.append(path)
            blobs[path] = item["sha"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "leaf-map.json").write_bytes(metadata_bytes)

    def download(path):
        target = args.out / "color" / Path(path).relative_to("raw/color")
        target.parent.mkdir(parents=True, exist_ok=True)
        content = fetch(base + urllib.parse.quote(path))
        verify_blob(content, blobs[path])
        target.write_bytes(content)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count, _ in enumerate(pool.map(download, paths), 1):
            if count % 250 == 0:
                print(f"Imágenes verificadas: {count}/{len(paths)}", flush=True)
    source = dict(repository=REPO, commit=commit, seed=args.seed,
                  leaves_per_class=None if args.all_leaves else args.leaves_per_class,
                  max_images_per_leaf=None if args.all_images else args.max_images_per_leaf,
                  all_leaves=args.all_leaves, all_images=args.all_images,
                  classes=args.classes, images=len(paths), paths=paths, remote_blobs=blobs,
                  license="CC BY-SA 3.0, según ficha oficial de Hugging Face")
    (args.out / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in source.items() if k not in {"paths", "remote_blobs"}}, indent=2))


if __name__ == "__main__":
    main()
